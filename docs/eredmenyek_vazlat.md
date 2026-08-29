# Eredményfejezet — váz

**Ez váz, nem szöveg.** Amelyik szakasz kísérlete még nem futott, ott
`[EREDMÉNY FÜGGŐBEN: …]` áll. Kitalált prózát nem tartalmaz.

Minden állítás mellé oda kell írni, melyik kategóriába tartozik:

| jelölés | jelentése |
|---|---|
| **[MÉRT]** | valódi futásból származó szám |
| **[FELTÁRÓ]** | egy magon, egy futáson tett megfigyelés; irány, nem bizonyíték |
| **[MÓDSZERTANI DÖNTÉS]** | mi választottuk, az eredmény ismerete nélkül |
| **[HIPOTÉZIS]** | előre kimondott, tesztelhető várakozás |
| **[JÖVŐBELI MUNKA]** | nem futott, és ma nem is fog |

A kiértékelést a `docs/replay_evaluation_protocol_2026-08-29.md` köti meg, ami
a GPU-eredmények **előtt** készült. A táblákat és ábrákat a
`python tools/compare_replay.py <workspace> --out data/results/replay` állítja
elő; kézzel beírt szám nincs.

---

## 1. Kísérleti protokoll  **[MÓDSZERTANI DÖNTÉS]**

PROB / S-OWODB, `t1.pth` horgony 19 ismert osztállyal, taszkonként **egy** új
osztály: t2 traffic light (head) → t3 fire hydrant → t4 stop sign → t5 parking
meter (tail) → t6 bench (head).

Kar-független beállítás: random kiválasztás, 600 régió/taszk, 6×100 kör, 2000
jelöltkép/taszk, 50 proposal/kép, `known_plus_selected`, 5 epoch, lr 2e-4,
batch 2, seed 0. Kiértékelés a 892 képes közös splitre, PROB saját
evaluátorával.

Miért egy osztály taszkonként, és miért nem a benchmark eredeti 20-as
lépése: `docs/atadas.md` §2 — húsz osztálynál a mért csereárfolyam 2931 volt,
azon a kereten semmilyen módszer nem különböztethető meg.

Táblázat: **Table 6** (`table6_cost.csv`) — mit kérdezett és min tanult minden
taszk.

## 2. A replay nélküli katasztrofális felejtés  **[MÉRT]**

`random__none`, a teljes FAST-lánc.

**[MÉRT]** A workspace ellenőrizve: `n_tasks=6`, seed 0, 2000 jelöltkép,
600 régió, 6 kör, `known_plus_selected`, 5 epoch, lr 2e-4, batch 2 —
**protokoll-kompatibilis** a ma esti két karral, és nem kell újrafuttatni.
Öt sor: t2 traffic light → t6 bench.

| taszk | asked | opened | trainable | no-supervision | target objects |
|---|---:|---:|---:|---:|---:|
| t2 | 600 | 513 | 364 | 149 | 123 |
| t3 | 600 | 513 | 365 | 153 | 28 |
| t4 | 600 | 513 | 379 | 142 | 13 |
| t5 | 600 | 513 | 380 | 136 | 16 |
| t6 | 600 | 513 | 405 | 133 | 65 |

`[EREDMÉNY FÜGGŐBEN: a per-osztály felejtési tábla és a sebezhetőség-regresszió
ugyanezen workspace-ből — tools/compare_replay.py, egy parancs.]`

Amit itt jelenteni kell: taszkonkénti `known_mAP50` / `prev_mAP50` /
`forgetting`, a 19 t1-osztály per-osztály pályája, és a sávonkénti átlag —
**a per-osztály pálya az elsődleges**, a sáv-aggregátum másodlagos, mert a
nevezője mozog (4.3 pont a protokollban).

Ábrák: **Figure B** (felejtés), **Figure D** (per-osztály felejtés vs log
gyakoriság), **Figure F** (felejtés vs horgony-AP).

## 3. Az objektum-szintű replay protokoll (V3)  **[MÓDSZERTANI DÖNTÉS]**

`Σ_c m_c = |E_k| = leszállított previous-class target = M = 400`, pontosan,
minden karon és minden taszkon.

Miért objektum és nem kép: a V2 képet tárolt egy objektum-allokáció
lefedésére, és PROB egész képeket olvas — mérve `head_favouring` 464, a
`tail_favouring` 1240 objektumot szállított egy 400-as keretre, azaz **2,67×**
szórást, amit az allokációs szabály okozott, nem a terv. Egy kar nem mondható
jobbnak, ha 2,7-szer annyit ismételt.

Hogyan kényszerítjük ki: replay-alias annotációk. Részletek és a PROB-oldali
bizonyítás: `owl/exemplars.py` modul-docstring.

Táblázat: **Table 5** (`table5_replay_composition.csv`).

**Per-osztály AP forrása [MÓDSZERTANI DÖNTÉS].** A `metrics.json` `coco_eval_bbox`
mezője — az evaluátor saját per-osztály AP50 vektora, nem újraszámolt AP. Minden
taszknál visszaépítjük belőle a fájl saját aggregátumait; eltérés esetén a riport
megtagadja a táblát. A `metrics_detections.json` per-osztály **recall**
kereszt-ellenőrzést ad, ami nem AP és nem is jelenthető annak.

## 4. Egyenletes replay  **[EREDMÉNY FÜGGŐBEN: random + uniform FAST-lánc]**

## 5. Eloszlás-tudatos, tail-favorizáló replay  **[EREDMÉNY FÜGGŐBEN: random + tail_favouring FAST-lánc]**

Amit már ma tudunk az allokátorról **[MÉRT, CPU]**: a `tail_favouring` valóban
a ritkaságot követi — Spearman(kvóta, gyakoriság) −0,995 t2-n, −0,42…−0,66
t3…t6-on, szemben a `uniform` ≈ 0-jával; a `bear`/`person` kvótaarány 14:1
helyett 1:1. A sáv-aggregátum ezt elrejti (t6: tail 71 vs 68), mert a tail sáv
1–4 osztályt tartalmaz — **ezért a per-osztály kvóta a fő riport.**

Ábra: **Figure E** (per-osztály allokáció, uniform vs tail-favouring).

## 6. Stability–plasticity kompromisszum  **[EREDMÉNY FÜGGŐBEN]**

Minden megtartási szám mellé az ugyanazon taszkon mért új-osztály AP.

**[MÓDSZERTANI DÖNTÉS]** Nincs egyetlen skalár pontszám; a leolvasás
kétdimenziós, a szabályt a protokoll 4.1 pontja köti meg az eredmény ismerete
nélkül.

Ábrák: **Figure A** (head/medium/tail AP), **Figure C** (új-osztály AP).
Táblák: **Table 1**, **Table 2**, **Table 3**.

## 7. Per-osztály sebezhetőség-elemzés  **[FELTÁRÓ]**

A kérdés: a gyakoriság jó proxy-e a felejtési kockázatra? Ezen múlik, hogy a
frekvencia-vezérelt memória premisszája egyáltalán áll-e.

**[FELTÁRÓ — ELŐD-BIZONYÍTÉK]** Az alábbi az **elődrepó `t1→t2` futásaiból**
származik (20 osztály egyszerre, más protokoll), **nem** a most befejezett
`random__none` FAST-láncból. Irányjelzés, nem ennek a láncnak a bizonyítéka. A
ρ(felejtés, log gyakoriság) −0,41 és +0,29 között ingadozik a tanítási
beállítástól függően; a ma estihez legközelebbi konfiguráción (`ft lr2e-4 e5`)
a gyakoriság R²-e **0,000**, a horgony-AP-é 0,017. Osztályszinten `aeroplane`
(5135 objektum) veszít legtöbbet, `cat` (4768 — ritkább) legkevesebbet.

`[EREDMÉNY FÜGGŐBEN: ugyanez a FAST-láncon, mind a három karra —
tools/compare_replay.py vulnerability-blokk]`

**[JÖVŐBELI MUNKA]** Ha a sebezhetőséget más jelzi előre, mint a gyakoriság,
akkor `m_c ∝ v_c^α` egy vulnerability-proxyval a következő kar. **Ma este nem
fut.**

## 8. Replay-összetétel és költség  **[RÉSZBEN MÉRT]**

**[MÉRT, CPU]** A replay képlábnyoma 377–397 kép taszkonként, a két kar között
<5% eltérés — nincs érdemi tanítási-compute confound. A három kar bitre azonos
új-felügyeleti képhalmazt kap.

`[EREDMÉNY FÜGGŐBEN: a valódi futások Table 5 és Table 6 sorai]`

## 9. Korlátok  **[MÓDSZERTANI]**

1. **Egy mag.** Nincs szignifikancia-állítás, nincs p-érték.
2. **Redukált kiértékelési split** (892 kép): publikált full-test számokkal nem
   összemérhető, karok között igen.
3. **A `tail` sáv 1–4 osztály** a korábbiak között; a csoportszintű állítás
   per-osztály pályák nélkül nem áll meg.
4. **A gyakoriság–nehézség antikorrelál** ezen az osztályhalmazon: a horgony-AP
   head 65,1 / medium 85,1 / tail 87,7 — a „tail" osztályok *magasabbról*
   indulnak, tehát az abszolút felejtés félrevezető, a relatív a helyes mérték.
5. **A plaszticitás alacsony** — a valódi baseline-on mérve: taszkonként
   **13–123** új-osztály objektum kerül a tanításba (t4: 13, t5: 16). Az
   `new_mAP50` ezért zajos, és a replay „javulása" részben plaszticitás-
   csökkenés lehet.
6. **A replay-memória a tanítóhalmaz ~50%-a**: a valódi baseline 364–405
   tanítható képet használ taszkonként, és a replay ehhez 377–397 alias-képet
   ad hozzá. Ez a protokoll tulajdonsága, nem hiba, de a `none` karral való
   összevetésnél ki kell mondani — a replay-karok nagyjából kétszer annyi képet
   látnak lépésenként.

## 10. Következmények a későbbi aktív-kiválasztás kísérletre  **[JÖVŐBELI MUNKA]**

A replay itt izolálva van (a kiválasztás fix random). Ha a replay-kar
kimutatható különbséget ad, az a későbbi selection-kísérlet mérési alapvonala
lesz; ha nem, a selection-kísérletet replay nélkül vagy a nyertes replay-karral
kell futtatni — de nem mindkettőt egyszerre változtatva.
