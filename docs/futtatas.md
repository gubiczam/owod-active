# Futtatás — mit kell csinálnod, sorrendben

Négy lépés. Az első kettőhöz nem kell GPU és nem kell semmit feltölteni.

---

## 1. lépés — Nézd meg, hogy működik (5 perc, semmi előkészület)

Nyisd meg a Colab-badge-et a `README.md`-ben, vagy közvetlenül:

<https://colab.research.google.com/github/gubiczam/owod-active/blob/main/notebooks/owod_active.ipynb>

A paraméter-cellában hagyd így:

```python
RUN_GPU = False
```

**Runtime → Run all.** GPU nem kell hozzá, sőt CPU-runtime-mal is jó.

**Mit fogsz látni.** A taszk-láncot, a jelöltkészletet, a klaszterezés
known-szennyezését, a koherencia-kapu mérését, a nyolc kiválasztási arm
összehasonlítását, a három címkézési szabályt, a replay-allokációt, és a szimulált
láncot. **Ez már önmagában előadható anyag** — minden szám élesben számolódik ki, nem
betöltött.

Ha itt bármi elhasal, ott a hiba, és nem kell rá GPU-időt égetni.

---

## 2. lépés — Tedd fel a checkpointot a Drive-ra (egyszer, ~10 perc feltöltés)

**Egy fájl kell, 478 MB.** Az SSD-ről:

```
/Volumes/AI_SSD/PROB/exps/SOWODB/PROB/t1.pth
```

ide a Drive-on:

```
MyDrive/OWL/checkpoints/SOWODB/t1.pth
```

**Lehet, hogy már ott van.** A korábbi futásokból lehet egy `MyDrive/DAOWOD/checkpoints/SOWODB/t1.pth`
— a notebook előellenőrzése ott is keres, tehát ha megvan, nem kell újra feltölteni.

Az annotációk **nincsenek külön feltöltésre**: mind a 28 800 jelöltkép XML-je a repóban
van (`data/staging/owdetr_pool_annotations.tar.gz`, 4,6 MB), a képeket pedig a notebook a
COCO-ról szedi le futás közben, csak azokat, amiket a kiválasztás megnyit.

---

## 3. lépés — Próbafutás (30–45 perc, GPU)

> A notebook minden cellája le van futtatva itt, a GPU-ág is, hamis PROB-bal
> (`tools/dry_run_notebook.py`, és a tesztek között is fut). Ez azt bizonyítja, hogy a
> vezérlés végigmegy — azt nem, hogy a valódi PROB elfogadja az argumentumokat. Ezért van
> még mindig szükség erre a próbafutásra.


Ez azt bizonyítja, hogy a teljes lánc működik: detektor-átfutás → kiválasztás →
címkézés → replay → tanítás → kiértékelés. **Ne hagyd ki.** Egy elgépelt Drive-útvonal
négy óra után is ugyanúgy kiderül, csak drágábban.

Colabban: **Runtime → Change runtime type → T4 GPU** (A100 se baj, de ehhez nem kell).

```python
RUN_GPU    = True
SMOKE_TEST = True     # ez a lényeg: két rövid taszk, apró teszthalmaz
```

**Runtime → Run all.**

Az első taszk pár percet a képek letöltésével kezd (taszkonként `CANDIDATE_IMAGES` darab
a COCO-ról, ~600 MB 4 000 képnél). Ezt egyszer fizeted meg: egy újraindított futás a már
lemezen lévőket kihagyja.

Az „Előellenőrzés" cella kiírja, mi van meg és mi nincs. Ha valami `MISSING`, ott áll
mellette, honnan kell odatenni, és a cella `CANNOT START`-tal zárul. **Az utána következő
cellák ilyenkor egy olvasható hibaüzenettel állnak meg**, nem `NameError`-ral — de a
diagnózis mindig az előellenőrzés kimenetében van, nem az alatta lévő hibában.

Utána lefordítja a Deformable-DETR CUDA-kernelét (néhány perc). Ha ez elhasal, nem
végzetes: a PROB-nak van tiszta PyTorch-tartaléka, csak lassabb — ki is írja.

**Amit a végén látni akarsz:** egy táblázatot `t2` és `t3` sorral, benne `known_mAP50`,
`prev_mAP50`, `new_mAP50`, `U_Recall50` értékekkel. **A számok itt még nem érnek semmit**
(két taszk, 100 régió, 8 képes teszthalmaz) — csak az számít, hogy *van* szám.

---

## 4. lépés — A valódi lánc (armonként ~4 óra)

```python
RUN_GPU    = True
SMOKE_TEST = False
ARM        = "prior_consult_batch"
```

**Runtime → Run all.** A `TIME_BUDGET_MINUTES = 420` miatt tisztán megáll, mielőtt a
Colab elvágja, és kiírja, melyik taszkokat nem futtatta le.

**Minden újraindítható.** Ha megszakad a session, indítsd újra ugyanezekkel a
paraméterekkel: ami már megvan, azt kihagyja, és ott folytatja, ahol abbahagyta. A
checkpointok és a metrikák a Drive-on gyűlnek, `MyDrive/OWL/work/<arm>/` alatt.

### Ehhez legalább két arm kell

Egy arm önmagában nem eredmény, mert nincs mihez hasonlítani. Futtasd le ugyanezt
**csak az `ARM` átírásával**:

| sorrend | `ARM` | mi ez |
|---|---|---|
| 1. | `"prior_consult_batch"` | a mi módszerünk |
| 2. | `"random"` | a padló — enélkül semmi nem értelmezhető |
| 3. | `"objectness"` | az ingyenes kontroll — ezt kell vernünk |

Három arm ≈ 12 óra, két-három Colab-esten elosztva. **Ha kevés az idő**, vedd
`N_TASKS = 6`-ra: az még mindig lefedi a head-osztályt és mind a három tail-osztályt
(traffic light → fire hydrant → stop sign → parking meter → bench), és armonként ~2
órára jön ki.

Mindegyik arm a saját mappájába ír, tehát nem írják felül egymást, és mindegyik ugyanazt
a közös teszthalmazt használja, tehát összehasonlíthatók.

---

## Mit jelentenek a számok, amikor megjönnek

| oszlop | mit válaszol meg |
|---|---|
| `images_no_supervision` | a keret mekkora része ment olyan képre, amin **még** nincs tanítható osztály |
| `images_from_earlier_tasks` | mennyi korábban kifizetett címke vált most használhatóvá, **ingyen** |
| `prev_mAP50` | **mennyit felejtett** — a korábbi osztályok pontossága |
| `new_mAP50` | **mennyit tanult** — a most bevezetett osztály |
| `U_Recall50` | felismeri-e egyáltalán, hogy valami ismeretlen |
| `forgetting` | `prev_mAP50` esése az előző taszk óta |
| `exchange_rate` | **hány régi mAP-pontot fizettünk egy új pontért** |

A csereárfolyam a döntő. A PROB teljes t2-felügyelete **0,20**-at fizet. A korábbi,
húsz-osztályos felállás **2931**-et fizetett — ezért van most taszkonként egy osztály.
**Ha a lánc 10 alá jön be, az önmagában eredmény.**

---

## Ha elakadsz

| tünet | mi történt |
|---|---|
| `MISSING` az előellenőrzésben | a `t1.pth` nincs ott, ahol keresi — a sor mellett ott az útvonal |
| `GPU: NONE` | Runtime → Change runtime type → T4 GPU, aztán Run all újra |
| `BUILD FAILED` a kernelnél | nem végzetes, csak lassabb; menj tovább |
| `PROB bridge 'train' is missing ...` | a PROB nem a `feat/daowod-bridge-v2` ágról jött; a `bridge.ensure_checkout` ezt kezeli, töröld a `/content/PROB` mappát és futtasd újra |
| a session elvágódott | ugyanezekkel a paraméterekkel Run all — folytatja |
| `unexpected keyword argument` bármelyik `owl` hívásban | a kernel a régi modult tartotta. **Runtime → Restart session**, aztán Run all: a notebook minden futásnál friss klónt húz és kiüríti a betöltött `owl` modulokat |
| a Drive megtelt | a lánc taszkonként csak a két legutolsó checkpointot tartja meg (`keep_checkpoints`), armonként ~1 GB. Ha régebbi futásból maradtak, töröld a `MyDrive/OWL/work/` alatti régi arm-mappákat |
| `NameError` a környezet-cellában | az előellenőrző cella nem futott le, vagy nem ment át. Görgess vissza a kimenetéhez: ott áll, mi hiányzik |
| minden `new_mAP50` nulla | ez volt a régi baj. Ellenőrizd, hogy `LEARNING_RATE = 2e-4` és `EPOCHS = 5` — 2e-5-tel nem tanul |
| `size of tensor a (0) must match the size of tensor b (4)` | egy kép nulla dobozzal érkezett: a PROB `ft` splitje csak a már bevezetett osztályokat tartja meg. Javítva: a lánc kiszűri ezeket, és a címkéjüket elteszi arra a taszkra, ahol az osztályuk bevezethető |
| `FileNotFoundError: .../ImageSets/OWDETR/owod_all_task_test.txt` | a PROB a saját alapértelmezett teszt-splitjét kereste. Javítva: a `train` most átadja a közös teszthalmazt. Ha mégis előjön, régi klónból fut — **Runtime → Restart session**, Run all |
| `FileNotFoundError: .../JPEGImages/xxx.jpg` a `predict` alatt | egy jelöltkép nem töltődött le. A lánc taszkonként szedi le őket, és a nem elérhetőket kidobja — ha ez mégis előjön, a hálózat állt el a futás közben; Run all újra, folytatja |
