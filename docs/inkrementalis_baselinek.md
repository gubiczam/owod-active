# Mihez mérjük magunkat

A kutatási terv és a konzultáció is kéri: ne csak a saját armjainkat mérjük össze, hanem
**más inkrementális tanulási módszerekkel** is, és legyen belőle egy eredménytáblázat.

Ez a doksi azt rögzíti, **melyik módszer hogyan valósul meg itt**, és — ami legalább ilyen
fontos — **melyik nem valósul meg, és miért**. Egy táblázat, amiben nem derül ki, hogy egy
sor referált szám vagy itt futott, nem eredménytáblázat.

---

## A tábla

| módszer | mi ez | hogyan valósul meg itt | állapot |
|---|---|---|---|
| **Joint training** | minden adat egyszerre — a felső korlát | a PROB saját teljes t2-tanítása | **lemérve**: prev 66,33 / új 36,13 |
| **Fine-tune (naiv)** | csak az új adat, semmi védelem — az alsó korlát | `replay_arm="none"`, `supervision_mode="train"` | **fut** a bridge-en |
| **Fine-tune + megtartott annotáció** | az új adaton, de a képen meglévő régi dobozokat nem dobjuk el | `supervision_mode="ft"` | **lemérve**: a felejtés 27 → 2,7 pont |
| **Replay (random exemplar)** | véletlen régi minták a memóriában | `replay_arm="random_images"` | **fut** |
| **iCaRL (exemplar-válogatás)** | herding: az a részhalmaz, aminek az átlaga az osztályátlagot követi | `replay.herding_order`, `replay_arm="herding"` | **implementálva, tesztelve** |
| **Egyenletes allokáció** | osztályonként azonos exemplar-szám — a mai standard | `alpha = 0` | **fut** |
| **Eloszlás-tudatos allokáció (a mi B kontribúciónk)** | `m_c ∝ n_c^α`, `α < 0` | `alpha = -0.5` / `-1.0` | **fut** |
| **Taszkonként újraosztott memória** | nem visszük tovább, minden taszkban újrasúlyozunk | `replay_reallocate=True` | **fut** |
| **WA (Weight Aligning)** | az új osztályok súlyvektorait a régiek normájára skálázzuk | checkpoint-műtét a PROB osztályozó fején | **nincs kész** — lásd lent |
| **BiC** | tanult bias-korrekciós réteg validációs halmazon | a PROB tanítási hurkát kellene módosítani | **nincs kész** |
| **LwF** | disztilláció az előző modellből | új veszteségtag a PROB-ban | **nincs kész** |
| **EWC** | Fisher-alapú súlybüntetés | Fisher-számítás + új veszteségtag | **nincs kész** |

---

## Miért ott húzódik a határ, ahol

Amit a PROB-bridge kitesz: `predict`, `train --labelled-ids --replay-ids
--supervision-mode --epochs --learning-rate`, `evaluate`. Ez pontosan az **adat**-oldali
beavatkozásokat teszi lehetővé: mit adunk a modellnek, milyen arányban, milyen
felügyelettel. Minden replay-alapú módszer ide esik, és ezért van mind a hét első sor
készen.

A **veszteségfüggvény**-oldali módszerek — LwF, EWC, BiC — a PROB tanítási hurkába nyúlnak.
Ezek nem „egy paraméter", hanem a `main_open_world.py` és a kritérium módosítása, és minden
ilyen módosítás egy új dolog, amit hibásan is meg lehet csinálni, csendben. Ezért nem
kerültek bele blindre.

**A WA a kivétel, és a következő lépés.** A Weight Aligning nem tanítás közben történik,
hanem **utána**: a checkpointban az új osztályok osztályozó-súlyvektorait átskálázzuk, hogy
a normájuk megegyezzen a régiekével. Ez tiszta checkpoint-műtét, nem kell hozzá a PROB
tanítóhurkát megérteni, és pontosan azt a torzítást célozza, amiről a kutatási terv beszél:
*„a BiC/WA esetén alkalmazott torzítás éppen a tail ellen hat."* Ezt a terv állítja, és itt
**le lehet mérni** — de csak akkor, ha a checkpoint fejének a szerkezetét egy valódi GPU-s
futáson ellenőriztük. Vakon megírni pont az a fajta munka, amit ez a repó el akar kerülni.

---

## Amit a táblázat mellé ki kell írni

Két dolog, ami nélkül a sorok nem összehasonlíthatók.

1. **Mindegyik itt futott, azonos protokollon.** Egy sem másik cikk jelentett számából van
   átemelve. Ez lassabb, viszont azt jelenti, hogy a sorok között a különbség a módszer, és
   nem a beállítás.
2. **Az orákulum-költség azonos.** Minden arm ugyanazt a `budget_per_task` régiószámot
   kérdezi meg. Ami nem azonos: hány *képet* nyit meg — ez a módszer következménye, és
   ezért külön oszlopban szerepel. A `full_image` címkézési szabály ennél többe kerül, és
   ezt a `labelling.Annotation.oracle_cost` ki is számolja, nem feltételezi.
