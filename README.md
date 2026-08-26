# owod-active

**Eloszlás-tudatos aktív annotáció nyílt világú, inkrementális objektumdetektáláshoz.**

A detektor 19 osztályt ismer, a világban 80 van. Körönként 600 régiót címkéztethetünk fel
egy emberrel. **Melyik 600-at kérjük?** És amikor az új osztály ismertté vált, **mit adunk
vissza a régiekből**, hogy ne felejtse el őket?

Alapmodell: **PROB** (Deformable-DETR). Benchmark: **S-OWODB**. Protokoll: **task1 → task10**,
taszkonként egy új osztállyal.

---

## Indulás

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gubiczam/owod-active/blob/main/notebooks/owod_active.ipynb)

**[`notebooks/owod_active.ipynb`](notebooks/owod_active.ipynb) — Futtasd le mindet.**
GPU nélkül, adathalmaz nélkül, néhány perc alatt lefut, és végigmegy a kutatás minden
kérdésén. Magyar magyarázat, angol kód.

**Ha futtatni akarod, és nem csak olvasni:**
[`docs/futtatas.md`](docs/futtatas.md) — négy lépés, sorrendben, a próbafutástól a valódi
láncig.

```bash
pip install -e ".[dev]"
pytest -q                                  # 30 állítás, ~90 másodperc
python tools/run_experiments.py --seeds 3  # minden szám újraszámolva
```

Azért fut GPU nélkül, mert a detektor **egyszer** futott le, offline: a commitolt
`data/pool/sowodb_t1_frozen_pool.npz` a PROB `t1.pth` checkpointjának egyetlen
forward-passe 2 400 benchmark-képen — 120 000 régió dobozzal, 256 dimenziós
dekóder-embeddinggel, poszteriorral, objectness-szel, és a benchmark saját annotációjához
illesztve IoU 0,5-nél. Egy tíz-arm × három-mag összehasonlítás így percekbe kerül és nem
Colab-sessionökbe.

A GPU-ág ugyanabban a notebookban van (`RUN_GPU = True`): ott a PROB súlyai ténylegesen
frissülnek, és a PROB saját kiértékelője ad számot.

---

## Mi jött ki

Minden szám alább a `data/results/` alatti CSV-kből, három maggal, 600 régiós kerettel.
Kézzel beírt szám nincs.

### A fő eredmény: hova megy a keret

| arm | ismeretlen objektum | ebből **tail** | tail-arány | megnyitott kép |
|---|---:|---:|---:|---:|
| `random` | 24,3 | 8,7 | 35% | 501 |
| `plan` — a kutatási terv egyenlete, pontosan | 21,0 | 11,0 | 48% | 339 |
| `entropy` — klasszikus aktív tanulás | 32,0 | 14,0 | 30% | 295 |
| `objectness` — **az ingyenes kontroll** | 155,0 | 48,0 | 30% | 548 |
| **`prior_consult_batch`** — a szintézis | 76,0 | **61,0** | **47%** | **308** |

**A tail oszlop a lényeg, és a mechanizmus tiszta.**

- A terv egyenlete **jól torzít, de nem talál**: 48%-os tail-arány — a legmagasabb a
  táblázatban — mindössze 11 objektumon. A `U + λD + γw·coh` három tagja együtt tudja,
  *melyik* régiót érdemes előnyben részesíteni, de nem tudja, hogy **egyáltalán objektum-e**.
- Az ingyenes `objectness × √terület` **talál, de nem torzít**: 48 tail-objektum, viszont
  csak 30%-os tail-arány — nagy, feltűnő head-objektumokat vesz.
- **A szorzatuk mind a kettőt megtartja**: 61 tail-objektum 47%-os tail-aránynál, és
  ezt **mind a három magban** hozza (58 / 64 / 61 az ingyenes kontroll 48 / 48 / 48-a ellen).

Vagyis: **7,0× a random**, **5,5× a terv saját egyenlete**, és **1,27× a legerősebb
ingyenes alapvonal** — miközben 240-nel kevesebb képet nyit meg ugyanazért a 600 régióért.

### Kép vagy régió: a legtisztább válasz

| szabály | orákulum-költség | felcímkézett objektum | **félcímkézett háttér** |
|---|---:|---:|---:|
| `box_only` | 600 | 423 | **20,7%** |
| `full_image` | 1082 (1,80×) | 3 498 | 0% |
| **`known_plus_selected`** | **600** | **2 729** | **0%** |

A `known_plus_selected` **ugyanannyiba kerül, mint a `box_only`** — a known objektumokhoz
nem kell ember, a detektor már tudja őket —, **nulla félcímkézéssel** és **hatszor annyi
felügyelettel**. A `box_only` ezzel szemben a háttérként tanított régiók 20,7%-át egy
valódi annotált objektumra teszi.

Ez egybevág a korábbi valódi GPU-méréssel: amikor a képen amúgy meglévő task-1
annotációkat visszatettük, a felejtés **27 pontról 2,7-re** esett — replay nélkül.

### Ami mérhetően nem működik

A konzultáción kért **bináris DBSCAN-koherenciakapu**. A PROB saját jellemzőterében a kapu
gyakrabban dobja ki a valódi ismeretlen objektumokat, mint a hátteret:

| eps | zajpont a háttérből | zajpont a valódi ismeretlenekből | ismeretlen-arány kapu előtt → után |
|---:|---:|---:|---:|
| 0,15 | 60% | **92%** | 3,4% → 0,8% |
| 0,25 | 38% | **78%** | 3,4% → 1,3% |

A készlet 81%-a háttér, a háttérrégiók pedig szinte egymás másolatai — ők ülnek a tér
legsűrűbb részén. „Sok szomszédod van" ebben a készletben azt jelenti: „háttérnek nézel ki".

**Az ötlet mögötti szándék viszont jó, és máshol teljesül.** Ha a ritkaság *ugyanabból* a
klaszterezésből jön, a hatalmas háttér-klaszter magától közel nulla súlyt kap: a szűrés a
`w` tag oldalán történik, nem a kapu oldalán.

### Egy klaszterezés, orákulum nélkül ellenőrizhető

A partíció minőségét nem sziluett-értékkel mérjük, hanem **known-szennyezéssel**: hány már
ismert elem esik unknown-jelölt klaszterbe. Ez orákulum nélkül becsülhető — a detektor a
saját ismert osztályait 0,83 pontossággal maga felcímkézi — és utólag ellenőrizhető:

| K | átlagos klaszterméret | szennyezés (becsült) | szennyezés (ellenőrzött) |
|---:|---:|---:|---:|
| 200 | 400 | 0,069 | 0,138 |
| 1600 | 50 | 0,028 | 0,116 |
| 3200 | 25 | 0,016 | 0,093 |

### Mihez mérjük magunkat

A replay-oldali inkrementális módszerek mind ugyanazon a protokollon futnak itt, egyik sem
másik cikk jelentett számából van átemelve: naiv fine-tune (alsó korlát), megtartott
annotációval, random exemplar-replay, **iCaRL herding**
(`replay.herding_order` — az exemplar-válogatás fele, implementálva és tesztelve),
egyenletes allokáció (a mai standard), és a mi eloszlás-tudatos allokációnk. Felső korlát
a PROB saját teljes t2-tanítása.

Ami **nincs kész, és ezt ki is mondjuk**: LwF, EWC és BiC a PROB veszteségfüggvényébe
nyúlna; a WA (Weight Aligning) tiszta checkpoint-műtét és a következő lépés, de csak azután,
hogy egy valódi GPU-futáson ellenőriztük a fej szerkezetét.
A teljes tábla: [`docs/inkrementalis_baselinek.md`](docs/inkrementalis_baselinek.md).

---

## Miért taszkonként egy osztály

Ez a legfontosabb szerkezeti döntés, és a mérés kényszerítette ki.

A korábbi felállásban egy inkrementális lépés **húsz** új osztályt adott hozzá 600
annotációból — osztályonként ~30 régió. Az eredmény mérhetetlen volt:

| valódi GPU-futás | előző-19 mAP50 | új-osztály mAP50 | **csereárfolyam** |
|---|---:|---:|---:|
| teljes t2-felügyelet | 66,33 | 36,13 | **0,20** |
| random, 600 régió | 27,27 | 0,016 | **2931** |

A csereárfolyam azt mondja meg, hány régi mAP-pontot fizetünk egy új pontért. 2931 nem
csere, hanem veszteség. Taszkonként **egy** osztállyal ugyanaz a 600 régió mind egy
osztályra megy — ez az, ami a plaszticitást egyáltalán mérhetővé teszi.

A lánc `t1` (a PROB publikált checkpointja, 19 osztály) után:
traffic light (head) → fire hydrant (tail) → stop sign (tail) → parking meter (tail) →
bench → chair → diningtable → pottedplant (medium) → backpack.

Az osztálysorrend nem szabad választás — a PROB kiértékelője pozíció szerint indexeli az
osztályokat —, de a hivatalos prefix magától lefedi mind a három gyakorisági csoportot.

**Ami éles feladattá teszi.** A jelöltkészlet 28 800 címkézetlen kép. A `fire hydrant`
ezek közül 911-ben van benne: **3,2%**. Egy random kiválasztás a keret 97%-át elpazarolja.

---

## Szerkezet

```
owl/                 tíz modul, modulonként egy fogalom
  protocol.py        a taszk-lánc, és a head/medium/tail csoportok
  proposals.py       amit a detektor javasol; a detektor-mezők és az orákulum elkülönítve
  clustering.py      egy partíció, amiből a ritkaság és a diverzitás is jön
  scoring.py         s(x) négy tagja és maga a pontszám
  selection.py       a keret elköltése: körök, batch-diverzitás, a regisztrált armok
  labelling.py       mit címkéztetünk egy kiválasztott képen, és mibe kerül
  replay.py          az exemplar-memória, az eloszlás-tudatos allokáció és iCaRL herding
  metrics.py         a PROB kiértékelője; felejtés, tanulás, csereárfolyam
  bridge.py          a PROB hívása — az egyetlen modul, ami tud a GPU-ról
  runner.py          a ciklus: szimulálva vagy élesben
  evaluation_subset.py  közös, csökkentett teszthalmaz, hogy tíz taszk megfizethető legyen

notebooks/owod_active.ipynb   EGY notebook, mind a két üzemmóddal
docs/konzultacio_2026-08-25_lefedettseg.md   a konzultáció pontról pontra: hol van, mi jött ki
docs/method.md                a specifikáció: minden tag, minden súly, egy taszk lépésről lépésre
docs/inkrementalis_baselinek.md   mihez mérjük magunkat, és mi az, ami még nincs kész
docs/futtatas.md              mit kell csinálni, sorrendben, a próbafutástól a valódi láncig
data/pool/                    a commitolt PROB-átfutás (60 MB)
data/results/                 minden jelentett szám ide generálódik
data/reference/measured/      korábbi valódi GPU-futások metrikái, hivatkozási pontnak
tests/                        egy állítás publikált állításonként, a rosszul sültekre is
tools/run_experiments.py      minden szám újraszámolása
```

---

## A GPU-ág

`RUN_GPU = True` a notebook paraméter-cellájában. Amire szükség van a Drive-on:

```
OWL/
  checkpoints/SOWODB/t1.pth    a PROB publikált t1 checkpointja (478 MB)
```

**Ez az egyetlen feltöltés.** A 28 800 jelöltkép annotációja a repóban van
(`data/staging/`, 4,6 MB), a képeket pedig a notebook a COCO-ról tölti le igény szerint,
csak azokat, amiket a kiválasztás megnyit. A notebook előellenőrző cellája a `DAOWOD`
mappában is keres, tehát ha egy korábbi futásból már ott van a checkpoint, nem kell újra
feltölteni. A PROB-ot a
[`gubiczam/PROB`](https://github.com/gubiczam/PROB) `feat/daowod-bridge-v2` ága adja, ahol
a `daowod_prob_bridge.py` a `predict` / `train` / `evaluate` igéket kiteszi.

**Költség.** A kiértékelés a drága rész, nem a tanítás: a teljes 4 952 képes teszt
checkpointonként ~32 perc. Ezért a lánc egy közös, csökkentett teszthalmazon mér
(`EVAL_MAX_PER_CLASS = 150` mellett 1 878 kép, ~12 perc). Egy kilenc-taszkos lánc így
armonként nagyjából 4 óra. **Minden hívás újraindítható**: ha a kimenet megvan, a hívás
kimarad, tehát egy megszakadt Colab-session ott folytatja, ahol abbahagyta.

---

## Mit lehet és mit nem lehet állítani

**Lehet.** Melyik pontszám mennyi valódi ismeretlent és mennyi tail-objektumot vesz meg
azonos orákulum-költségen; hogy a körökre bontás csak a frissíthető armoknak segít; hogy a
három címkézési szabály mibe kerül és mennyi félcímkézést okoz; hogy a known-szennyezés
orákulum nélkül becsülhető.

**Nem lehet.** Hogy egyik arm kevesebbet *felejt*, mint a másik. Ezt csak a valódi
detektoron lehet mérni, és a korábbi munka lemérte, hogy a fagyasztott szimuláció az
akvizíciós módszereket felejtés szerint **fordított sorrendbe** rakja. Ezért a
`runner.simulate()` nem is ad detekciós metrikát. Szintén nem lehet: publikált
PROB-számokkal összevetni a csökkentett teszthalmazon mért mAP-ot, és szignifikanciát
állítani három magból — három mag előjelpróbát bír el, p-értéket nem.

**Hatókör.** Egy benchmark, egy alapmodell, három mag. A fagyasztott gerinc miatt a
dobozok soha nem javulnak: az unknown-lefedettség felülről korlátos azzal, amit a `t1.pth`
egyáltalán javasol — a jelölthalmaz ismeretlen objektumainak 13,4%-a.
