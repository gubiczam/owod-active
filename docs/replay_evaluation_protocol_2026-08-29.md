# Replay-kísérlet: kiértékelési protokoll

**Ez a dokumentum a `random + uniform` és `random + tail_favouring` GPU-futások
ELŐTT készült, 2026-08-29-én.** A célja, hogy megkösse, mit fogunk holnap
jelenteni és hogyan értelmezzük — mielőtt bármelyik szám ismert lenne. Ami itt
nincs leírva, az holnap nem lehet a fő állítás.

Commit, ami előtt íródott: `d6d51ea` (Replay Protocol V3). A `random + none`
alapfutás már kész; a másik kettő még nem futott.

---

## 1. A kérdés, egyetlen mondatban

> Fix, objektum-szintű `M = 400`-as rehearsal-keret mellett javítja-e a
> tail-favorizáló allokáció (`α = −0,5`) a ritka osztályok megtartását az
> egyenletes allokációhoz (`α = 0`) és a replay nélküli futáshoz képest, anélkül
> hogy a head/medium megtartás vagy az új-osztály tanulás elfogadhatatlanul
> romlana?

A kiválasztás **random** mind a három karon. A replay az egyetlen kísérleti
változó. Ez a futás **nem** az aktív kiválasztásról szól.

---

## 2. A három kar

| kar | replay | mit mér |
|---|---|---|
| `random__none` | nincs | a felejtés alsó korlátja — **kész** |
| `random__uniform` | `α = 0`, M=400 | a szakirodalmi standard |
| `random__tail_favouring` | `α = −0,5`, M=400 | a B kontribúció |

Minden más azonos: FAST-lánc t1→t2…t6, 600 régió/taszk, 6×100 kör, 2000
jelöltkép, `known_plus_selected`, 5 epoch, lr 2e-4, batch 2, seed 0,
`replay_reallocate=False`, V3 objektum-szintű replay.

---

## 3. Amit jelenteni fogunk

### A. Összesített megtartás
`known_mAP50`, `prev_mAP50`, `forgetting` — taszkonként t2…t6.

### B. Plaszticitás
`new_mAP50` és `new_class_AP50` a taszk által bevezetett osztályra, taszkonként.

### C. Eloszlás-specifikus megtartás
`mAP50_head`, `mAP50_medium`, `mAP50_tail` taszkonként, és ezek eltérése
(a) a replay nélküli alapfutástól, (b) az egyenletes allokációtól.

### D. Per-osztály megtartás
Minden korábban ismert osztályra: horgony-AP, AP minden taszk után, végső AP,
abszolút felejtés, relatív felejtés (`(anchor − final) / anchor`), a tanító
gyakoriság és a frekvencia-csoport.

### E. Replay-összetétel
Taszkonként és karonként: `requested / allocated / delivered`, `replay_images`,
`replay_unique_source_images`, per-osztály kvóta, head/medium/tail bontás,
`from_previous_memory`, `from_new_task`, `evicted`, `added`.

### F. Költség és felügyelet
`asked`, `images_opened`, `images_trainable`, `images_no_supervision`,
`target_objects_in_images`, replay objektum- és képszám.

---

## 4. Hogyan döntünk — előre rögzítve

**Nincs egyetlen skalár pontszám.** Nem definiálunk súlyozott
„stability–plasticity indexet": bármilyen súlyt választanánk, az az eredmény
ismeretében önkényes lenne. Helyette a döntés egy **kétdimenziós
kompromisszum-leolvasás**, és minden megtartási állítás mellett ott áll az
ugyanazon taszkon mért új-osztály AP.

### 4.1 Az elsődleges leolvasás

A `tail_favouring` akkor mondható jobbnak a `uniform`-nál, ha **mindkettő**
teljesül:

1. **Megtartás.** A `mAP50_tail` és a per-osztály tail-felejtés a
   `tail_favouring` javára tér el a t3…t6 taszkok **többségén** — nem csak t6-on.
2. **Ár.** Az `new_mAP50` és a `mAP50_head` romlása nem nagyobb, mint a
   tail-oldali nyereség. Ha a tail nyer 2 pontot és a head 5-öt veszít, az nem
   javulás, hanem átcsoportosítás.

Ha (1) teljesül és (2) nem, azt **kompromisszumként** jelentjük, nem győzelemként.
Ha (1) nem teljesül, azt **negatív eredményként** jelentjük.

### 4.2 Amit előre kimondunk, hogy utólag ne lehessen elmozdítani

* **Egy mag nem alapoz szignifikanciát.** Nem lesz p-érték, nem lesz
  „szignifikánsan jobb". A különbségeket effektusméretként és irányként
  közöljük, a mérés bizonytalanságát pedig kimondjuk.
* **A taszkonkénti pálya számít**, nem csak a végpont. Egy kar, ami t3–t5-ön
  rosszabb és t6-on jobb, nem „jobb kar" — az egy nem-monoton pálya, és így
  kell jelenteni.
* **A t6 fontos, de nem az egyetlen bizonyíték.** A fő táblák taszkonkéntiek.
* **A frekvencia-csoportok összetétele változik a lánc során**, ezért a
  csoportszintű állításokat per-osztály pályákkal kell alátámasztani. Mérve
  (lásd 5. pont): a `tail` sáv a *korábbi* osztályok között t2–t3-on **egyetlen**
  osztály (`bear`), t6-ra négy. Egy „tail mAP50" változás tehát részben
  összetétel-változás, nem megtartás-változás.
* **A replay javíthatja a megtartást pusztán azzal, hogy kevésbé plasztikussá
  teszi a modellt.** Ezért minden megtartási szám mellett ott áll az új-osztály
  AP ugyanazon a taszkon.

### 4.3 A csoport-összetétel kezelése

Két külön mennyiséget jelentünk, és soha nem keverjük:

* **fix-osztályos megtartás** — a 19 t1-osztályra számolt AP, amelyek
  összetétele a lánc során nem változik. Ez az, ami a felejtésről szól.
* **változó csoport-aggregátum** — `mAP50_head/medium/tail` az adott taszkon
  ismert osztályokra. Ez a benchmark szokásos riportja, de a nevezője mozog.

---

## 5. Ami már ma ismert, és ezért nem lehet holnapi „felfedezés"

Ezek CPU-n mérve, a GPU-futások előtt:

* A `tail_favouring` allokáció **valóban a ritkaságot követi**:
  Spearman(kvóta, gyakoriság) = **−0,995** t2-n, −0,42…−0,66 t3…t6-on, míg a
  `uniform` ≈ 0. A `person`/`bear` kvótaarány `uniform`-nál 1,00, a
  `tail_favouring`-nál 0,07–0,21.
* A **head/medium/tail sáv-aggregátum ezt rosszul mutatja**: t6-on a tail
  összeg 68 (uniform) vs 71 (tail_favouring), mert a tail sáv 1–4 osztályt
  tartalmaz. A tömeg, amit a `tail_favouring` a headről levesz, a **medium**
  sávba kerül (t6: 125 → 154). **Ezért a per-osztály kvóta és a
  rangkorreláció az elsődleges replay-összetétel-riport, nem a sáv-összeg.**
* A `replay_reallocate=False` **nem hígítja** a tail-preferenciát: a
  `reallocate=True` futás per-osztály kvótái bitre azonosak; csak a
  képlábnyom változik.
* A replay **képszám** a két kar között t2…t6-on 377–397 vs 387–397 (≤5%
  eltérés), tehát nincs érdemi tanítási-compute confound a karok között.
* A három kar **bitre azonos új-felügyeleti képhalmazt** kap (a kiválasztás
  random, azonos seeddel), tehát az annotációs költség az abszolút számok
  kaveátja, **nem** a karok közti confound. A **valódi**, befejezett
  `random__none` FAST-lánc mért értékei (a `results_random.csv`-ből, nem
  CPU-becslés):

  | taszk | asked | opened | trainable | no-supervision | target objects |
  |---|---:|---:|---:|---:|---:|
  | t2 | 600 | 513 | 364 | 149 | 123 |
  | t3 | 600 | 513 | 365 | 153 | 28 |
  | t4 | 600 | 513 | 379 | 142 | 13 |
  | t5 | 600 | 513 | 380 | 136 | 16 |
  | t6 | 600 | 513 | 405 | 133 | 65 |

  A megnyitott képek **26–30%-a** nem ad felügyelést ezen a taszkon (bankolva a
  későbbire), és az új osztály objektumszáma t4–t5-en **13–16**-ra esik — ezért
  lesz a `new_mAP50` zajos, és ezért nem szabad egyetlen taszk plaszticitására
  következtetést építeni.
* **A ritkaság ezen az osztályhalmazon gyenge előrejelzője a felejtésnek.**
  **[FELTÁRÓ — ELŐD-BIZONYÍTÉK, NEM EZ A FAST-LÁNC.]** Az elődrepó `t1→t2`
  futásaiból származik (20 osztály egyszerre, más protokoll); a befejezett
  `random__none` FAST-lánc per-osztály elemzése még nem készült el, azt a
  `tools/compare_replay.py` állítja elő. A ρ(felejtés, log gyakoriság) −0,41 és +0,29 között
  ingadozik a tanítási beállítástól függően, és a ma esti beállításhoz
  legközelebbi konfiguráción (`ft lr2e-4 e5`) **R² = 0,000** a gyakoriságra és
  0,017 a horgony-AP-ra. Osztályszinten: `aeroplane` (5135 objektum) veszít a
  legtöbbet (67,6 pont), `cat` (4768 — ritkább!) a legkevesebbet (5,1 pont),
  `person` (262 465) a mezőny közepén van.

  **Ezért ezt előre kimondjuk:** ha a `tail_favouring` nem javít, annak a
  legvalószínűbb oka nem az allokátor, hanem az, hogy a gyakoriság nem a
  sebezhetőség tengelye ezen a 19 osztályon. Ezt a `tools/compare_replay.py`
  vulnerability-blokkja méri majd a valódi FAST-láncon, és ez a negatív
  eredmény önmagában is jelentés-értékű — de **nem** indok arra, hogy ma éjjel
  más armot futtassunk.

---

## 6. Amit NEM fogunk állítani

* Hogy bármelyik kar „szignifikánsan" jobb.
* Hogy az eredmény más protokollra (M, α, lánchossz, seed) általánosítható.
* Hogy a `tail` sáv változása osztályszintű bizonyíték nélkül megtartás-változás.
* Hogy publikált PROB/OWOD számokkal közvetlenül összemérhető — a redukált
  kiértékelési split miatt nem az.

---

## 7. Hogyan készül a riport

```bash
python tools/compare_replay.py <workspace-root> --out data/results/replay
```

**Honnan jön a per-osztály AP50.** A `metrics.json`-ban nincs per-osztály
táblának nevezett kulcs; a `coco_eval_bbox` viszont **az**: 83 elem
`[mAP, mAP, 80 osztály, unknown]` elrendezésben, és ezt az evaluátor maga írja.
Ezért nem számolunk újra AP-t a nyers detekciókból — egy második AP-implementáció
csak egy második konvenció-készlet lenne, amit finoman el lehet rontani.
A `owl.metrics.validate_per_class_ap50` minden taszkra visszaépíti a *saját
fájlja* által jelentett `previous_known_AP50` / `current_known_AP50` /
`unknown_AP50` értékeket a vektorból, és ha nem egyeznek, a riport **megtagadja**
a per-osztály táblát ahelyett, hogy hihetőnek látszó, rossz osztályokat közölne.
Minden commitolt valódi GPU-metrikafájlon hat értékes jegyre egyezik.

A `metrics_detections.json` **kereszt-ellenőrzésként** szolgál: per-osztály
*recall* IoU 0,5-nél, ugyanazzal a mohó, pontszám szerint rendezett párosítással,
amit az `unknown_recall_by_group` már használ. Ez **nem AP**, és nem is szabad
AP-ként jelenteni — de egy osztálynak, aminek az AP-je összeomlott, a recallját
is el kell veszítenie, és ha a kettő ellentmond, az AP-tábla a gyanús.

Egy parancs, hat tábla (CSV + Markdown + LaTeX) és hat ábra. Hiányzó vagy
félbehagyott kar nem hiba: a hiányzó cella `—`, és a `depth` riport megmondja,
meddig jutott mindegyik. Részletek: `docs/eredmenyek_vazlat.md`.
