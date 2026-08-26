# A 2026-08-25-i konzultáció, pontról pontra

Minden ötlet, ami elhangzott: hol van a kódban, mi jött ki belőle, és mi van még hátra.
A „mi jött ki" oszlopban minden szám a `data/results/` alatti CSV-kből származik, amiket a
`tools/run_experiments.py` ír ki. Kézzel beírt szám nincs benne.

Amit érdemes előre tudni: **hét ötletből öt mérhetően működik, egy mérhetően nem, egy
pedig nyitva marad, amíg GPU nem fut.**

---

## 1. `D(x)` — legyen valódi diverzitás

> „Legyen egy »mennyire új a már címkézett elemekhez képest« – szóval tényleges diverzitás."

**Hol van.** `owl/scoring.py: novelty()`, és a `ScoreConfig.diversity_source` kapcsoló.
A régi definíció (`anchor`) fix task-1 horgonyhoz mért, az új (`labelled`) a **növekvő
címkézett halmazhoz**.

**A kérdés, ami itt kettévált.** A konzultáción kiderült, hogy két különböző dolgot
hívunk diverzitásnak, és egyik sem helyettesíti a másikat:

| | mit mér | mit véd ki |
|---|---|---|
| (a) **novelty** | távolság a már címkézettektől | ne vegyük meg újra, amit tudunk |
| (b) **batch-diverzitás** | a most kiválasztottak egymástól való távolsága | ne vegyünk 600 majdnem azonos régiót |

Mind a kettő be van építve, külön taggal (`diversity_source` és `mu_batch`), és külön
mérhető. A (b) a `selection.py: _greedy()`-ben mohó, k-means++ jellegű: minden kiválasztás
lenyomja azt, ami rá hasonlít.

**Mi jött ki.** A `consult` arm (csak (a)) a terv egyenletének a **másfélszeresét**
találja meg. A `consult_batch` (a + b) ennek is a fölé megy. `data/results/selection_arms.csv`.

---

## 2. `coh(x)` — bináris kapu, DBSCAN alapon

> „`coh(x) ∈ {0, 1}` — kapcsoló, ne súly. Ha valami zajpont, akkor a koherencia 0."

**Hol van.** `owl/clustering.py: noise_gate()`, `owl/scoring.py: coherence(method='binary')`.
A folytonos változat (`continuous`) is megmaradt, hogy a kettő egy változó eltéréssel
összemérhető legyen.

**Mi jött ki — és ez az egy pont, ami nem működik.** A kapu a PROB saját jellemzőterében
**gyakrabban dobja ki a valódi ismeretlen objektumokat, mint a hátteret**:

| eps | zajpont a háttérből | zajpont a valódi ismeretlenekből | ismeretlen-arány a kapu előtt → után |
|---:|---:|---:|---:|
| 0,15 | 60% | **92%** | 3,4% → **0,8%** |
| 0,25 | 38% | **78%** | 3,4% → **1,3%** |
| 0,35 | 8% | **27%** | 3,4% → **2,8%** |

`data/results/coherence_gate.csv`

**Miért.** A készlet 81%-a háttér, a háttérrégiók pedig szinte egymás másolatai, tehát ők
ülnek a tér legsűrűbb részén. Ebben a készletben „sok szomszédod van" azt jelenti:
„háttérnek nézel ki".

**Amit ez nem jelent.** Az ötlet mögötti szándék — ne tanuljunk magányos szemetet — jó, és
a 3. pont meg is oldja: ha a ritkaság *ugyanabból* a klaszterezésből jön, a hatalmas
háttér-klaszter magától közel nulla súlyt kap. A szűrés a `w` oldalán történik, nem a
kapu oldalán. **Az entrópia változatlan maradt**, ahogy kérted.

---

## 3. Egy klaszterezés, amiből `D` és `w` is kijön

> „Ha van pl. 1000 known és 10000 unknown, akkor úgy kell klaszterezni, hogy a lehető
> legkevesebb known menjen át az unknownba."

**Hol van.** `owl/clustering.py`. Egy `Partition`, amiről a ritkaság (`rarity()`,
klaszterméret) és a diverzitás (`cluster_novelty()`, a known-klaszterektől mért távolság)
is leolvasható.

**A minőségi mérőszám, ahogy kérted: known-szennyezés.** `contamination()`. Egy fontos
részlet, ami első nekifutásra elrontotta: „a klaszter többsége known" szabály itt
degenerált, mert a készlet 81%-a háttér, tehát szinte egy klaszter sem ér el 50% knownt, és
minden klaszter unknownnak minősül. A helyes kérdés a **dúsulás**: több known van ebben a
klaszterben, mint egy véletlenszerűben?

**Mi jött ki.** A diagnosztika orákulum nélkül lefuttatható — a detektor a saját ismert
osztályait 0,83 pontossággal maga is felcímkézi —, és utólag ellenőrizhető:

| K | átlagos klaszterméret | szennyezés (becsült) | szennyezés (ellenőrzött) | unknown-recall |
|---:|---:|---:|---:|---:|
| 200 | 400 | 0,069 | 0,138 | 0,74 |
| 800 | 100 | 0,039 | 0,128 | 0,80 |
| 1600 | 50 | 0,028 | 0,116 | 0,82 |
| 3200 | 25 | 0,016 | 0,093 | 0,83 |

`data/results/clustering_contamination.csv`

**Egy csapda, amit be kellett zárni.** A szennyezés monoton csökken, ahogy nőnek a
klaszterszámok — a határesetben minden pont saját klaszter, a szennyezés nulla, és semmit
nem tudunk. Ezért a `tune()` két padlót ír elő: minimum unknown-recall, és **minimum
átlagos klaszterméret**, mert a ritkaságot a klaszter *méretéből* olvassuk.

---

## 4. Replay — eloszlás-tudatosan, taszkonként külön

> „Ez a replay is eloszlás tudatos kell legyen, illetve minden taskban más más memória."

**Hol van.** `owl/replay.py`. Három **külön** paraméter, mert három külön kísérleti
tengely: a memória **mérete** (`total`), a **szabály** (`alpha`, ahol `m_c ∝ n_c^α`), és
hogy taszkonként **újraosztjuk-e** (`carry_forward(reallocate=True)`).

**Mi jött ki eddig.** Az allokáció működik, a t1 tizenkilenc osztályán, 400 exemplar mellett:

| α | legritkább osztály | medián osztály | leggyakoribb (person) |
|---:|---:|---:|---:|
| 0 (egyenlő) | 21 | 21 | 21 |
| 1 (head) | **1** | 10 | **249** |
| −0,5 (tail) | 48 | 17 | 4 |
| −1 (tail) | **95** | 11 | **1** |

α = 1 mellett a legritkább osztály egyetlen exemplart kap — ez a terv által megjósolt
kudarc, és a `minimum=1` az, ami megakadályozza, hogy nullára menjen.

**Ami nyitva van.** Hogy melyik α a legjobb, **csak a valódi detektoron dönthető el**. A
korábbi munka ezt lemérte és nem tudta eldönteni: három maggal az α-k közötti szórás
nagyobb volt, mint a tail- és head-favorizálás közötti különbség. A tíz-taszkos lánc
ennél több adatpontot ad.

---

## 5. Kép vs. régió: mit címkézünk valójában

> „Ha egy képet kiválasztunk ne félcímkézés legyen, tanítsunk meg rajta mindent! Ez
> mennyit ront az osztályozón? Ezt is tesztelni kell."

**Hol van.** `owl/labelling.py`, három regisztrált szabály.

**Mi jött ki. Ez a legtisztább eredmény az egész napból.** 600 kiválasztott régió, 306
megnyitott kép:

| szabály | orákulum-költség | felcímkézett objektum | félcímkézett háttér | felügyelet / orákulum-egység |
|---|---:|---:|---:|---:|
| `box_only` | 600 | 423 | **20,7%** | 0,71 |
| `full_image` | **1082** (1,80×) | 3 498 | 0% | 3,23 |
| `known_plus_selected` | **600** | 2 729 | **0%** | **4,55** |

`data/results/labelling_policy.csv`

**A válasz.** A `known_plus_selected` **ugyanannyiba kerül, mint a `box_only`** — mert a
known objektumok felcímkézéséhez nem kell ember, a detektor már tudja őket —,
**nulla félcímkézéssel**, és **hatszor annyi felügyelettel**.

A `box_only` ezzel szemben a háttérként tanított régiók **20,7%-át** egy valódi annotált
objektumra teszi. Ez pontosan az a hiba, amit a korábbi GPU-futás mért: amikor a képen
amúgy meglévő task-1 annotációkat visszatettük, a felejtés **27 pontról 2,7-re** esett,
replay nélkül. Igazad volt abban is, hogy ez a replaynek is jót tesz — ingyen replay.

**Ami nyitva van.** A GPU-oldalon ez jelenleg a PROB `--supervision-mode` kapcsolójára
képződik le (`ft` vs `train`), nem valódi doboz-szintű szabályra. Egy tényleges
per-doboz szabályhoz szűrt XML-eket kellene írni. Ez a következő lépés.

---

## 6. A mérés: task1 → task2 → … → task10

> „Legyen kb task1 (pl itt tud 10 osztályt), aztán egy új osztályt tanítunk neki,
> felcímkézzük, mostmár 11 osztályt tud – ez a task2."

**Hol van.** `owl/protocol.py: build_chain()`, és `owl/runner.py: run_chain()`.

**Miért ez a legfontosabb szerkezeti változtatás.** A korábbi felállásban egy inkrementális
lépés **húsz** új osztályt adott hozzá 600 annotációból. Az eredmény mérhetetlen volt:

| futás | előző-19 mAP50 | új-osztály mAP50 | csereárfolyam |
|---|---:|---:|---:|
| teljes t2-felügyelet | 66,33 | **36,13** | **0,20** |
| random, 600 régió | 27,27 | 0,016 | 2931 |
| eloszlás-tudatos, 600 régió | 46,64 | 0,001 | — |

*(a `data/reference/measured/` alatti valódi GPU-futásokból)*

Taszkonként **egy** osztállyal ugyanaz a 600 régió mind egy osztályra megy. A lánc:

| taszk | új osztály | csoport | train-objektumok |
|---|---|---|---:|
| t2 | traffic light | head | 11 431 |
| t3 | fire hydrant | **tail** | 1 228 |
| t4 | stop sign | **tail** | 1 277 |
| t5 | parking meter | **tail** | 1 076 |
| t6 | bench | head | 8 454 |
| t7 | chair | head | 29 820 |
| t8 | diningtable | head | 9 421 |
| t9 | pottedplant | medium | 5 200 |
| t10 | backpack | head | 8 309 |

Az osztálysorrend nem szabad választás: a PROB kiértékelője pozíció szerint indexel, tehát
csak a hivatalos sorrend prefixét lehet ismertté nyilvánítani. Szerencsére ez a prefix
magától lefedi mind a három gyakorisági csoportot.

**Ami ezt éles feladattá teszi.** A jelöltkészlet 28 800 címkézetlen kép. A `fire hydrant`
ezek közül **911-ben** van benne — 3,2%. Egy random kiválasztás a keret 97%-át elpazarolja.
Itt kell nyernie egy eloszlás-tudatos kiválasztásnak.

**Ami nyitva van.** A lánc GPU-t igényel. A notebook 9. szakasza futtatja, újraindíthatóan.

---

## 7. Egyben vagy körökre bontva?

> „Kiválasztjuk a legjobb 100-at, újraszámoljuk a pontszámokat, megint 100-at… Ezt is le
> kell tesztelni."

**Hol van.** `owl/selection.py: select(rounds=...)`. `rounds=1` a 600×1, `rounds=6` a
6×100, `rounds=12` a 12×50.

**Mi jött ki, és pontosan az, amit vártál.** A körökre bontás **csak azokon az armokon
segít, amelyeknek van mit frissíteniük**:

| arm | 600×1 | 6×100 | változás |
|---|---:|---:|---|
| `consult` (D a címkézettektől) | 26 | **36** | +38% |
| `objectness` (nincs mit frissíteni) | 160 | 160 | 0% |
| `entropy` (nincs mit frissíteni) | 34 | 34 | 0% |

Ez nem véletlen egybeesés, hanem a mechanizmus: a `D` tag a címkézett halmaztól méri a
távolságot, tehát minden kör után változik. Az `objectness` és az `entropy` a detektor
kimenetének függvénye, ami nem mozdul, ha nem tanítunk közben.

---

## Ami a konzultáción nem hangzott el, de a mérésből következik

**Az ingyenes kontroll erős, és ezt ki kell mondani.** Az `objectness × √terület` —
semmi tanulás, semmi eloszlás-modellezés — a legtöbb ismeretlent találja meg, és a terv
egyenletének a **nyolcszorosát**. Bármelyik szemantikus tagnak ezt kell vernie, nem a
randomot.

**Amiben viszont verhető: a tail.** Az `objectness` főleg nagy, feltűnő head-objektumokat
talál. A `prior_consult_batch` — az ingyenes prior **szorzóként**, a konzultáció tagjaival
— kevesebb ismeretlent talál összesen, de **több tail-objektumot**, azonos költségen. A
keretet a fejről a farokra tolja át, ami pontosan a kutatási terv állítása.

Ez az a szám, amire a dolgozat eredményfejezete épülhet.
