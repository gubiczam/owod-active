# Átadás egy következő Claude-nak

Ez a fájl az, amit egy új beszélgetés elején be kell illeszteni. A kutatási tervet
(`owod.docx`) és a konzultációs jegyzetet külön is csatold — azok a szerződés, ez az
állapotjelentés.

---

## 1. Helyzet

BME TDK-kutatás, OWOD + inkrementális tanulás. **Kettő hét van a TDK-ig.** Kell egy
eredménytáblázat, egy dolgozat és egy védés.

| | |
|---|---|
| aktív repó | `github.com/gubiczam/owod-active` (publikus), lokálisan `/Users/gubiczam/Documents/owod-active` |
| PROB-bridge | `gubiczam/PROB` **`feat/daowod-bridge-v2`** ágán — a `main`-en NINCS |
| adat és checkpoint | `/Volumes/AI_SSD`: `datasets/owod_canonical` (120 GB), `PROB/exps/SOWODB/PROB/t1..t4.pth` (478 MB/db) |
| a két törölt elődrepó | `/Volumes/AI_SSD/archive_2026-08-26/*.tar.gz` — ha kell belőlük valami |
| Colab | Pro, T4. A notebook a GitHubról klónozódik minden futásnál. |
| venv | `.venv` a repóban, Python 3.12 |

**Nyelvi konvenció:** kód és azonosítók angolul, a felhasználóval magyarul, a `docs/`
magyarul.

---

## 2. A protokoll, és miért éppen ez

PROB `t1.pth` 19 ismert osztállyal indul, aztán **taszkonként EGY új osztály**, a
benchmark osztálysorrendjének prefixében (a PROB kiértékelője pozíció szerint indexel,
tehát nem lehet ugrálni):

`t2` traffic light (head) → `t3` fire hydrant (tail) → `t4` stop sign (tail) →
`t5` parking meter (tail) → `t6` bench (head) → `t7` chair → `t8` diningtable →
`t9` pottedplant (medium) → `t10` backpack

**Miért egy osztály:** húsz osztály/lépésnél a mért csereárfolyam (hány régi mAP-pont egy
új pontért) **2931** volt, a teljes t2-felügyelet 0,20-a ellenében. Azon a kereten semmi
nem tanul, tehát semmilyen kiválasztási módszer nem különböztethető meg.

---

## 3. Ami MÉRVE van (CPU, a commitolt PROB-átfutáson, 3 mag)

Regenerálható: `python tools/run_experiments.py --seeds 3` → `data/results/*.csv`

**A fő eredmény — tail-objektum 600 régiós kereten:**

| arm | tail | tail-arány | megnyitott kép |
|---|---:|---:|---:|
| `random` | 8,7 | 35% | 501 |
| `plan` (a terv egyenlete pontosan) | 11,0 | 48% | 339 |
| `objectness` (ingyenes kontroll) | 48,0 | 30% | 548 |
| **`prior_consult_batch`** | **61,0** | 47% | 308 |

3/3 magban veri az ingyenes kontrollt (58/64/61 a 48/48/48 ellen). **A mechanizmus:** a
terv tagjai tudják, *melyik* régiót érdemes választani (48% tail-arány, a legmagasabb), de
nem tudják, hogy egyáltalán objektum-e (11 találat). Az ingyenes prior fordítva. A
szorzatuk mindkettőt megtartja.

**Címkézési szabály (konzultáció 5. pont) — a legtisztább eredmény:**

| szabály | orákulum-költség | félcímkézett háttér | felügyelet/egység |
|---|---:|---:|---:|
| `box_only` | 600 | **20,7%** | 0,71 |
| `full_image` | 1082 | 0% | 3,23 |
| **`known_plus_selected`** | **600** | **0%** | **4,55** |

**Ami mérhetően NEM működik:** a konzultáción kért bináris DBSCAN-koherenciakapu. A PROB
jellemzőterében gyakrabban dobja ki a valódi ismeretlen objektumokat (92%), mint a
hátteret (60%) — a háttér a legsűrűbb régió, mert egymás másolatai. **Az ötlet szándéka
viszont teljesül a 3. ponton:** ha a ritkaság ugyanabból a klaszterezésből jön, a hatalmas
háttér-klaszter magától ~0 súlyt kap.

**Klaszterezés (konzultáció 3. pont):** known-szennyezés K=1600-nál 0,028 (detektorból
becsült) / 0,116 (annotációkkal ellenőrzött), unknown-recall 0,82. Orákulum nélkül
futtatható diagnosztika, utólag ellenőrizhető.

---

## 4. Ami NYITVA van, prioritási sorrendben

### (1) A terv fő végpontja implementálva, de GPU-n MÉG NEM MÉRVE

`owl.metrics.unknown_recall_by_group` — dobozszintű U-Recall gyakorisági csoport szerint,
IoU 0,5, pontszám szerint mohó párosítás. Ez a terv megkülönböztető mérése:

> „a tail-U-Recall mérőszámokat értékelem ki, mint az orákulum-költség függvénye […] a
> várt tendencia, hogy az eloszlás-tudatos kiválasztás azonos tail-szintet lényegesen
> kevesebb annotációból ér el."

A notebook kiírja `U_Recall_tail / _medium / _head / _all` + `oracle_cost_so_far`
oszlopokban, és a végén a három arm hasonlító táblázatát. **Ez a dolgozat
eredményfejezete. Ezt kell lefuttatni.**

### (2) `new_mAP50 = 0` — a plaszticitás, és miért nincs

A kiválasztás **nem tudja, melyik a bevezetendő osztály**, és nagyjából a véletlen szintjén
találja meg (mért: 0,80× – 2,39×). Taszkonként 6–50 példányt talál; egy osztály ennyiből
nem tanulható. Nyers erővel nem járható. A jelöltkészlet mért előfordulási arányaiból, és abból, hogy
600 régió 218–393 képet nyit meg:

| cél (fire hydrant, 3,16% előfordulás) | megnyitandó kép | keret/taszk |
|---:|---:|---:|
| 50 példány | 1 444 | ~4 000 régió |
| 100 példány | 2 889 | ~8 000 régió |
| 300 példány | 8 666 | **~24 000 régió** |
| 1 000 példány (a PROB saját t2-je ennyit lát) | 28 887 | ~80 000 régió |

**A hatékonysági görbe három pontja már megvan** a valódi (nem cache-elt) taszkokból:
10 példány → 0,00 · 20 → 0,00 · 50 → **0,70** · [PROB teljes t2: ~1000 → 36,13].
Ez nem hiányzó eredmény, hanem *a terv által kért annotációs hatékonyság-görbe* első
szakasza — és azt mondja, hogy 600 régiós kereten a plaszticitás nem elérhető.

Három lehetséges válasz, döntés kell:
- **(a)** cél-visszacsatolás: az első kör után az orákulum már megnevezett néhány példányt
  a bevezetendő osztályból; a további körök keressenek ezekhez hasonlót. A terv maga írja:
  *„amelyet minden annotációs kör után iteratívan frissítünk"*. Kb. egy napi munka.
- **(b)** csak head-osztályokat vezetünk be — akkor viszont nincs tail az új-osztály
  tengelyen, ami a kutatási kérdés.
- **(c)** kimondjuk eredményként: ezen a kereten a plaszticitás nem elérhető, és a
  dolgozat a felfedezésről + megtartásról szól. **Ez védhető**, mert a terv fő végpontja
  a tail-U-Recall, nem az új-osztály mAP.

### (3) Inkrementális baselinek

Készen fut: naiv fine-tune, megtartott annotációval, random exemplar-replay, **iCaRL
herding** (`replay.herding_order`, tesztelve), egyenletes allokáció, eloszlás-tudatos
allokáció (`m_c ∝ n_c^α`), taszkonként újraosztott memória.

Nincs kész, és ki van mondva: LwF, EWC, BiC a PROB veszteségfüggvényébe nyúlna. **A WA
(Weight Aligning) tiszta checkpoint-műtét és a következő logikus lépés** — de csak azután,
hogy egy valódi GPU-futáson ellenőriztük a PROB osztályozó-fejének szerkezetét.
Részletek: `docs/inkrementalis_baselinek.md`.

---

## 5. A CSAPDÁK — mindegyik egy GPU-sessionbe került

**Olvasd el mindet, mielőtt bármit módosítasz.** Ezek nem hipotézisek, hanem megtörtént
hibák, és mindegyikhez tartozik teszt.

1. **A PROB RÉSZSZÖVEG szerint választ annotáció-szűrőt a split nevéből.**
   `make_coco_transforms` a `train`/`ft`/`val`/`test`-et vizsgálja ebben a sorrendben,
   `OWDetection.__getitem__` utána már csak a `train`/`test`/`ft`-t. Egy `val`-ra
   illeszkedő név olyan ágra kerül, ahol **egyetlen szűrő sem fut** — és `eval` tartalmazza
   a `val`-t. Következmény: a `label_known_class_and_unknown` nem fut, **nincs unknown
   ground truth, tehát a U-Recall mindenhol nulla**, hibaüzenet nélkül. Őr:
   `owl.evaluation_subset.check_split_name`. A split neve `owl_shared_test`.

2. **`seen = prev + current`** — a PROB összeadja a két flaget, tehát a második a
   **növekmény**, nem a futó összeg. `protocol.Task.n_new` az, amit át kell adni. Rossz
   érték: minden mAP hibás, csendben.

3. **A `ft` split csak a `range(0, prev+curr)` osztályokat tartja meg.** Egy kép, amin
   csak későbbi taszkok objektumai vannak, **nulla dobozzal** érkezik, és a collate
   elhasal (`size of tensor a (0) must match the size of tensor b (4)`). A lánc kiszűri
   őket, **és a címkéjüket elteszi** arra a taszkra, ahol az osztályuk bevezethető
   (`reuse_deferred_labels`). Mért arány: a megnyitott képek 28–42%-a.

4. **PROB-alapértékek, amiket nem lehet magukra hagyni** (`tests/test_bridge_flags.py`
   adatként tartja): `--test-set` alapja `owod_all_task_test` (nem létező fájl);
   `--eval-every` alapja **1**, vagyis minden epoch után kiértékel; `--seed` át sem ment;
   `--learning-rate` alapja 2e-5, amivel 0,010 új-osztály mAP50 jött ki.

5. **Az újraindítás összekeveri a konfigurációkat, ha nem ellenőrzöd.** Egy próbafutás és
   egy valódi futás ugyanabba a munkakönyvtárba írt, és a valódi újrahasznosította a
   próbafutás metrikáit — két sor 16 képes teszthalmazon, három 1400-ason, ami 29 pontos
   hamis „felejtésként" jelent meg. Most a `config.json` fingerprintjét összehasonlítja és
   megtagadja a futást. **Külön munkakönyvtár kell minden konfigurációhoz.**

6. **A notebook cellái és az `owl` elcsúsznak.** A cellák abból jönnek, aki utoljára
   mentette; az `owl` a gitből, minden futásnál frissen. Ezért **minden közös érték az
   `owl`-ban lakik** (`evaluation_subset.SHARED_TEST_SET`), és a környezet-cellában van egy
   elcsúszás-őr. Soha ne írj le a notebookba olyan értéket, amit az `owl` is definiál.

7. **A per-osztály AP50 a `coco_eval_bbox` vektorban van**, nem `per_class_AP50` kulcsban:
   `[mAP, mAP, 80 osztály, unknown]` = 83 elem. Három futáson ellenőrizve. Enélkül a
   head/medium/tail bontás — a terv megkülönböztető mérése — nem számolható.

8. **A `MultiScaleDeformableAttention` CUDA-kernel.** Ha nem fordul le, a PROB tiszta
   PyTorch-tartalékon fut: **~3× lassabb** (mért: 2000 képes predict 23,1 perc a
   tartalékon, ~8 perc fordítva). A notebook most kiírja a valódi buildhibát és ennek
   megfelelően árazza a sessiont.

---

## 5b. HA KÖZBEN FUT EGY GPU-LÁNC — koordináció

A felhasználó párhuzamosan futtathat egy Colab-láncot, miközben te dolgozol. **A notebook
minden Run all-nál friss klónt húz a `main`-ről**, tehát amit pusholsz, azt a következő
Run all felveszi. Ez két konkrét veszélyt jelent:

1. **Ha megváltoztatod a `CycleConfig` bármely `RESULT_AFFECTING` mezőjének alapértékét,
   vagy a notebook paraméter-celláját, a folyamatban lévő lánc munkakönyvtára
   használhatatlan lesz** — a fingerprint-őr helyesen megtagadja a folytatást, és a
   felhasználó órákat veszít.
2. Egy félbehagyott arm újraindításához a paramétereknek **bitre azonosnak** kell lenniük.

**Amíg fut egy lánc, ezekhez ne nyúlj:** `owl/runner.py` `CycleConfig` alapértékei, a
notebook 2. (paraméter-) cellája, `owl/protocol.py` osztálysorrend, `owl/selection.py`
`ARMS` szótár, `owl/evaluation_subset.py` `SHARED_TEST_SET`.

**Amihez szabadon nyúlhatsz:** `docs/`, `tools/run_experiments.py` és a CPU-mérések,
`tests/`, README, új modul amit még senki nem hív, és bármi, ami csak *olvassa* az
eredményeket (`data/results/`, elemzés, ábrák, dolgozatszöveg).

Ha mégis kell egy fingerprintet érintő változtatás: **előbb kérdezd meg a felhasználót,
fut-e lánc**, és ha igen, várd meg vagy adj neki új munkakönyvtárnevet.

---

## 6. Hogyan ellenőrizd a saját munkádat — KÖTELEZŐ

```bash
python tools/dry_run_notebook.py      # a TELJES notebook, hamis PROB-bal, ~4 perc
pytest -q -m "not slow"               # 99 teszt, ~55 s
pytest -q -m slow                     # a notebook végigfuttatása tesztként
ruff check owl tools tests
```

**Soha ne adj ki notebookot anélkül, hogy a `dry_run_notebook.py` átmegy.** Három hiba
jutott el élő GPU-sessionbe, mert csak a részeket teszteltem, nem a notebookot. A
próbafuttató minden cellát lefuttat egy névtérben, ahogy a Colab, és a hamis bridge
**úgy tagad meg, ahogy a PROB tenné** — ellenőrzi, hogy a képek és annotációk a lemezen
vannak, alkalmazza a PROB szűrési tartományait, elutasítja a félrevezető splitnevet.

Amikor javítasz egy hibát, **tedd vissza és ellenőrizd, hogy a teszt elkapja.** Ez az
egyetlen bizonyíték, hogy a teszt ér valamit.

---

## 7. A következő két hét — javasolt sorrend

**1–3. nap: futtasd le a fő mérést.** A notebook készen áll: nyisd meg a Colab-linket és
Run all. Három arm (`prior_consult_batch`, `random`, `objectness`), közös időkeretből,
újraindíthatóan. A `MINIMAL_CHAIN = True` felezi, ha a CUDA-kernel nem fordul.
**Amíg ez nincs meg, nincs eredményfejezet.**

**4–5. nap: döntsd el a (2) pontot** (plaszticitás). A védhető minimum a (c): kimondjuk,
hogy ezen a kereten az új-osztály tanulás nem elérhető, és megmutatjuk, mennyi kellene
hozzá — ez maga is az annotációs hatékonyság-görbe, amit a terv kér.

**6–9. nap: a dolgozat.** A `docs/` már tartalmazza a specifikációt (`method.md`), a
konzultáció pontonkénti lefedettségét (`konzultacio_2026-08-25_lefedettseg.md`) és a
baseline-táblát. Az eredményfejezet a GPU-lánc táblázataiból áll.

**10–14. nap: védés, tartalék.** Ha van idő: WA baseline, vagy a cél-visszacsatolás (2a),
vagy egy harmadik mag a CPU-méréseken.

**Amit NE csinálj:** ne kezdj új módszert írni, amíg a fő mérés nem futott le. A repó
képes válaszolni a terv kérdésére; ami hiányzik, az GPU-idő, nem kód.

---

## 8. Amit a felhasználó tud és nem tud

Nem ML-mérnök: a fogalmakat érti, a kódot olvasni tudja, de a hibakeresést nem ő végzi.
**Kifejezetten kérte, hogy egy Colab-linkre kelljen csak Run all-t nyomnia.** Ha valamit
át kell írnia, az kudarc — tedd a repóba előbeállításként.

Türelmes volt öt egymást követő hibán, de joggal mondta, hogy „még mindig nem jó".
**A tanulság: ne küldj ki semmit, amit nem futtattál végig.**
