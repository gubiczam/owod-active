# A módszer

Amit a kód csinál, pontosan. Ez a specifikáció; hogy mi jött ki belőle, az a `README.md`,
és hogy melyik konzultációs ötlet hol van, az a
`konzultacio_2026-08-25_lefedettseg.md`.

---

## 1. Az anyag: egy detektor-átfutás

`PROB (Deformable-DETR, dino_resnet50)`, checkpoint `exps/SOWODB/PROB/t1.pth`, benchmark
`S-OWODB`. Az exportkor ismert: a 19 task-1 osztály. Minden más — 61 osztály — ismeretlen.

| | jelölt-halmaz | kiértékelő halmaz |
|---|---:|---:|
| kép | 1 600 | 800 |
| régió | 80 000 | 40 000 |
| valódi ismeretlen objektumon ülő régió | 2 758 | 1 063 |
| known objektumon ülő régió | 12 113 | 6 581 |
| háttér | 65 129 | 32 356 |

Képenként a PROB 100 objektum-lekérdezéséből a legjobb 50, a PROB saját objectness-sorrendje
szerint — a válogatáshoz semmilyen annotációt nem használ.

Minden régió hordoz: dobozt (normalizált `cxcywh`), 256 dimenziós dekóder-embeddinget
(egységnyi hosszra normálva betöltéskor), 81 oszlopos osztály-poszteriort (soronként 1-re
normálva), és objectness-t. Ezen kívül — **elkülönítve**, `Candidates.oracle()` mögött — a
benchmark saját annotációját, IoU 0,5-nél illesztve. A `owl/scoring.py` és a
`owl/selection.py` soha nem hívja meg az `oracle()`-t; ezt a `tests/test_owl.py`
`test_scoring_never_reads_an_answer` állítja.

---

## 2. A négy tag

Mind a négy egy szám régiónként, kizárólag detektor-kimenetből. A három additív tagot
**rangnormalizáljuk** a készleten (`rank_normalise`), ami [0, 1]-re teszi őket holtversenyek
átlagolásával — enélkül egy nyers entrópia és egy nyers inverz-gyakoriság nincs azonos
skálán, és a fix súlyok némelyiket észrevétlenül nullára súlyoznák.

| tag | mi ez | implementáció |
|---|---|---|
| `U(x)` | a 81 oszlopos poszterior normalizált Shannon-entrópiája, `log(81)`-gyel osztva | `scoring.uncertainty(method='entropy')` |
| `D(x)` | koszinusz-távolság a **legközelebbi már felcímkézett** régióhoz. Üres halmaznál 1,0 | `scoring.novelty` |
| `w(ĉ(x))` | `-log(n_c / N)`, ahol `n_c` a jelölt klaszterének mérete | `scoring.rarity(method='log_inverse')` |
| `coh(x)` | 0 vagy 1: 0, ha a jelölt zajpont vagy `min_samples`-nél kisebb klaszterben van | `clustering.noise_gate` |

**A `D` két változata**, mert a konzultáción kiderült, hogy két külön dolgot jelent:

* `diversity_source='labelled'` — távolság a növekvő címkézett halmaztól. „Ne vegyük meg
  újra, amit tudunk."
* `mu_batch > 0` — batch-diverzitás: a mohó kiválasztás közben minden kiválasztás lenyomja
  azt, ami rá hasonlít, k-means++ módjára. „Ne vegyünk 600 majdnem azonos régiót."
  Egy rövidlistán fut (`SHORTLIST_FACTOR = 20`), mert a redundancia lefelé mozdíthat egy
  jelöltet, felfelé nem.

**Egy klaszterezés, két tag.** `clustering.fit(method='kmeans', n_clusters=1600)`. Ugyanaz
a partíció adja a `w`-t (klaszterméret) és — `diversity_source='clusters'` esetén — a
`D`-t (a klaszter-centroid távolsága a legközelebbi known-tartalmú klaszter centroidjától).
A partíció minőségi mérőszáma a **known-szennyezés**, nem sziluett-érték: hány known régió
esik olyan klaszterbe, ami a készlet saját known-arányánál kevesebb knownt tartalmaz. A
dúsulás-alap azért kell, mert abszolút többségi szabállyal a mérőszám degenerált — a
készlet 81%-a háttér.

---

## 3. A pontszám, két alakban

### 3.1 A terv egyenlete — arm `plan`

```
s(x) = U(x) + 0.2·D(x) + 0.5·w(ĉ(x))·coh(x)
```

A terv nem adott meg λ-t és γ-t. Ezek egyszer lettek rögzítve, mielőtt bármelyik végpontot
megnéztük volna, és soha nem lettek hangolva — ez az, ami minden összehasonlítást egy
változóssá tesz.

### 3.2 A szorzatos alak — arm `prior_consult_batch`

```
s(x) = P(x) · ( 1 + 0.2·D(x) + 0.5·w(ĉ(x))·coh(x) + 0.3·B(x) )

P(x) = objectness(x) · sqrt(terület(x))        # objektum-szerűség, tanulásmentes
B(x) = batch-diverzitás, a kiválasztás közben frissül
```

Három eltérés a 3.1-től, mindegyik külön armként is fut:

1. **Az objektum-szerűség szorzóként, nem összeadandóként.** Semmi nem menti meg azt a
   régiót, ami nem néz ki objektumnak. Ez az egyetlen tag, amit **nyersen** használunk és
   nem rangnormalizálunk — egy szorzó rangnormalizálása pont a lényegét venné el.
2. **`P` foglalja el az `U` helyét.** Az entrópia kikerül; `objectness × √terület` ingyen
   van, és önmagában is arm (`objectness`), az a bar, amit a szemantikus tagoknak verniük
   kell.
3. **Batch-diverzitás bekapcsolva.**

**Miért ez az alak nyer.** A terv tagjai tudják, *melyik* régiót érdemes előnyben
részesíteni (48%-os tail-arány, a legmagasabb minden arm közül), de nem tudják, hogy
egyáltalán objektum-e (11 tail-objektum). Az ingyenes prior fordítva: talál (48
tail-objektum), de nem torzít (30% tail-arány). A szorzat mind a kettőt megtartja: 61
tail-objektum 47%-os aránynál.

---

## 4. Egy taszk, lépésről lépésre

`owl/runner.py`.

1. **Jelöltképek.** A taszk `candidate_images_per_task` képet kap a még nem használtakból.
2. **Egy detektor-átfutás** rajtuk (`bridge.predict`).
3. **A keret elköltése.** `budget_per_task` régió, `rounds_per_task` körben elosztva.
   Körök között a címkézett halmaz nő, tehát a `D` mozdul — ez a kutatási terv 2. ábrájának
   visszacsatolása, és `rounds=1` az a kontroll, ami kikapcsolja.
4. **Az orákulum válaszol**, a címkézési szabály szerint (`owl/labelling.py`):
   * `box_only` — csak a kiválasztott doboz kap címkét, a képen minden más háttérként tanít;
   * `full_image` — a képen minden annotált objektum megkapja a címkéjét;
   * `known_plus_selected` — a known objektumok ingyen (a detektor már tudja őket), a
     kiválasztott unknown felcímkézve, a többi unknown **ignore**, nem háttér.
   A költséget a szabály határozza meg, nem a keret: a `full_image` képenként több
   annotációs egységbe kerül, és ezt meg is fizeti.
5. **Az exemplar-memória** (`owl/replay.py`): `m_c ∝ n_c^α`, `Σ m_c = M`. Az objektum-szintű
   allokációt egy halmazlefedési heurisztika váltja képlistára, mert egy kép több osztályt
   is kiszolgál. `minimum=1` garantálja, hogy egy tail-osztály ne kerekedjen nullára.
6. **Finomhangolás** (`bridge.train`). `--supervision-mode ft` megtartja az előző taszkok
   dobozait a kiválasztott képeken; `train` eldobja őket. A `box_only` szabály a `train`-re
   képződik le, a másik kettő az `ft`-re.

   Öt PROB-alapértéket **nem** hagyunk magára, és a `tests/test_bridge_flags.py` rögzíti,
   hogy melyiket miért: a `--test-set` alapból egy olyan split-fájlt nevez meg, amit ez a
   protokoll soha nem ír; az `--eval-every` alapból **1**, vagyis minden epoch után
   kiértékelne, holott a kiértékelés a protokoll drága fele és itt külön, szándékos hívás;
   a `--learning-rate` alapból 2e-5, amivel a korábbi munka 0,010 új-osztály mAP50-et mért;
   az `--epochs` alapból 1; a `--seed` pedig egyáltalán nem jutott át, tehát a
   `SEED = 1`-es futás is 0-val keveredett volna.
7. **Kiértékelés** (`bridge.evaluate`) a közös, csökkentett teszthalmazon.

Minden hívás újraindítható: ha a kimenet létezik, a hívás kimarad.

---

## 5. A kiértékelés

A PROB saját `OWEvaluator`-a. Taszkonként kiírt számok:

| metrika | mit válaszol meg |
|---|---|
| `known_mAP50` | mennyit tud összesen |
| `prev_mAP50` | **mennyit felejtett** |
| `new_mAP50` | **mennyit tanult** |
| `U_Recall50` | felismeri-e egyáltalán, hogy valami ismeretlen |
| `forgetting` | `prev_mAP50` esése az előző taszk mérése óta |
| `exchange_rate` | **hány régi mAP-pontot fizettünk egy új pontért** |

A csereárfolyam a döntő szám. A teljes t2-felügyelet 0,20-at fizet. Egy futás, ami 74-et
fizet, nem cserél, hanem veszít.

**A közös, csökkentett teszthalmaz.** A teljes 4 952 képes teszt checkpointonként ~32 perc,
tíz taszkon armonként öt óra. Az `evaluation_subset` megtartja minden deklarált osztályból a
képeket (osztályonként legfeljebb `max_per_class`-t, ami kiegyenlíti a költséget a lánc
mentén), és determinisztikusan mintavételez maradékot. Az így mért previous-class mAP
**mintabecslés**, publikált teljes-teszt számokkal nem összehasonlítható — armok között
viszont igen, mert mind ugyanazon a halmazon fut.
