# Korábbi valódi GPU-futások

Ezek nem ebben a repóban futottak, hanem az elődjében
(`gubiczam/owod-longtail`), a PROB saját bridge-én keresztül, valódi
súlyfrissítéssel és a PROB hivatalos kiértékelőjével. Azért vannak itt, mert a
számaik **hivatkozási pontok**: ezekhez képest mondjuk meg, mit ér egy új futás.

| fájl | mit tartalmaz |
|---|---|
| `full_t2_supervision_metrics.json` | a felső korlát: a PROB teljes t2-tanítása. prev 66,33 / új 36,13 / csereárfolyam **0,20** |
| `random_b600_metrics.json` | random kiválasztás 600 régióval. prev 27,27 / új 0,016 / csereárfolyam **2931** |
| `mult_prior_shrunk_b*_metrics.json` | az eloszlás-tudatos arm 600 / 3 500 / 20 000 régiónál |
| `objectness_prior_b*_metrics.json` | az ingyenes kontroll ugyanezeken a kereteken |
| `*_ft_lr*_e*_metrics.json` | a plaszticitás-rács: tanulási ráta × lépésszám, fix b600 mellett |
| `real_group_forgetting.csv` | a felejtés head/medium/tail bontásban, felügyeleti mód és replay szerint |
| `efficiency_curve*.csv` | a hatékonysági görbe pontjai |
| `*_result.md` | a hozzájuk tartozó, akkor írt eredményleírások, változatlanul |

**Amiért ez a négy `*_result.md` itt maradt, miközben a többi negyven doksi nem:**
mindegyikben van olyan mért szám, amire ez a repó hivatkozik. A többi
előregisztráció, munkanapló és prezentációs vázlat volt.

**Egy figyelmeztetés, amit ezek közül a legfontosabb.** A
`real_forgetting_result.md` rögzíti, hogy a fagyasztott jellemzőkön alapuló
szimuláció az akvizíciós módszereket felejtés szerint **fordított sorrendbe**
rakja, mint a valódi detektor. Ezért nem ad a `runner.simulate()` detekciós
metrikát, és ezért kell a GPU-ág.
