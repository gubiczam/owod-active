# Method V3 — post-hoc mechanistic audit

**2026-09-03. Diagnostic only. The pre-registered verdict is not revisited.**

```
METHOD_V3_VERDICT = C_DOWNSTREAM_NOT_SUPPORTED        (12/12 trajectories)
```

Nothing here changes that verdict, and nothing here tunes `A`, `C`, `D`, `R`,
`lambda`, `gamma`, an exponent or a threshold. The question is *why* the
experiment came out the way it did, and whether it could have come out any other
way.

Reproduce every table with

```bash
python tools/audit_method_v3.py \
    --results /content/drive/MyDrive/OWL/results/method_v3_selection_transfer \
    --export  /content/drive/MyDrive/OWL/features/dinov2_vitb14_method_v2_v1.npz \
    --views   /content/drive/MyDrive/OWL/features/dinov2_vitb14_stage2_views_v1.npz
```

The tool is **read-only** on the results directory and refuses to write into it.

## 0. What this audit could and could not see

The results and the frozen view export live on Drive; this audit was written on
the local repository. So it splits into two halves, and the split is stated
rather than blurred:

| computable from committed artefacts | needs the Drive artefacts |
|---|---|
| the `A` and `U` rankings and every selection they imply | the real `C` values, hence the `A*C` ranking |
| the whole supervision chain, per arm, exactly | the trajectories' own `labelled_ids.txt` / `replay_ids.txt` |
| the replay memory, per arm and seed, exactly | the per-trajectory detector metrics |
| the invariance bound that decides §1 | the measured overlap that confirms it |

Everything below marked **MEASURED** is computed from committed artefacts and is
exact. Everything marked **PREDICTED** is a bound or an inference that the tool
settles when run with `--views` and `--results`.

---

## 1. Did A and A\*C select the same regions?

### 1.1 The reported coincidence

```
A     images_opened 590   unknown_objects 150   medium+tail 79   background 0.60
A*C   images_opened 590   unknown_objects 150   medium+tail 79   background 0.60
```

### 1.2 The code is not the explanation

`owl.method_v2_stage2.score_c` is a literal row-wise product and `consistency`
is a literal element-wise `min` of two cosines. Neither collapses, neither
returns `A`, and `arm_score("A*C", ...)` passes the real `C` through. A test
divides the product by `A` and recovers `C` exactly. So the coincidence is not a
silently-degenerate score.

### 1.3 The invariance bound — MEASURED

The prefix of size *k* under `A·C` equals the prefix under `A` **only if**

```
A_k / A_(k+1)  >  max(C) / min(C)
```

so the gap ratio at each cut is exactly the dynamic range of `C` that the prefix
can absorb. Measured on the population (8,010 proposals, 839 images):

| budget | A at cut | A below cut | **gap ratio** | within ±0.1% | ±1% | ±5% |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.906855 | 0.906481 | **1.0004128** | 8 | 65 | 276 |
| 200 | 0.881214 | 0.881181 | **1.0000365** | 5 | 87 | 308 |
| 300 | 0.857589 | 0.857542 | **1.0000545** | 8 | 53 | 295 |
| 400 | 0.825636 | 0.825390 | **1.0002981** | 4 | 48 | 216 |
| 500 | 0.779087 | 0.779047 | **1.0000523** | 5 | 38 | 151 |
| 600 | 0.726807 | 0.726786 | **1.0000297** | 4 | 32 | 155 |

The `A` ranking is dense to **three parts in one hundred thousand** at the
600-cut. A cosine similarity between DINOv2 CLS features of a 1.20× crop and its
1.10× / 1.30× versions cannot have a dynamic range below 1.00003 — Stage 2
measured `C`'s unknown-vs-background AUC at 0.6101, and an AUC that far from 0.5
requires real spread.

**PREDICTED: the two arms did not select the same 600 proposals.** The identical
aggregates are an aggregate coincidence, not set identity.

### 1.4 Why the aggregates can match anyway — MEASURED

Only **32** proposals lie within ±1% of `A_600` and **155** within ±5%, so
crossings are confined to a few dozen rows per cut. That band inherits P2's
composition, which is **76.7% background**, and `A` selects 590 distinct images
for 600 regions — so almost every image carries exactly one region. Swapping one
background proposal for another:

* leaves `images_opened` at 590 (one image closes, another opens);
* leaves `unknown_objects` and `medium+tail` untouched (neither row is an object);
* leaves `background_share` at 0.60 after rounding.

So four matching aggregates are exactly what a background↔background swap in a
narrow band produces. The aggregates were never able to detect this difference,
which is itself a finding about the reporting.

### 1.5 What settles it

Two things, both in the tool:

* `audit_prefix_overlap.csv` — proposal/image intersection and Jaccard, entering
  and leaving counts, Spearman on the union and the exact discordant-pair count,
  at all six budgets. Needs `--views`.
* **the authoritative artefact**: each trajectory's own
  `train/labelled_ids.txt`. That is the literal list PROB was handed. If
  `A/seed0` and `A*C/seed0` hold the same list, the *supervision* was identical
  regardless of which proposals were clicked. Needs `--results`.

---

## 2. What actually varied between paired trajectories

### 2.1 PROB's own seed was never varied — MEASURED

`owl.bridge.Bridge.seed` defaults to `0`, and `tools/run_method_v3.py`
constructs the bridge **without passing `seed`**. So `--seed 0` went to PROB in
all twelve trajectories: the dataloader order, the augmentation draws and the
model RNG were seeded identically everywhere.

This corrects a statement in `docs/method_v3_protocol_2026-09-02.md` §2, which
said the seed moves "the exemplar draw and PROB's own `--seed`". It moved only
the exemplar draw.

### 2.2 The seed varied the rehearsal set, and only that — MEASURED

For `A` and `U` the selection is a static ranking, so all three seeds select the
identical 600 regions and hand PROB the identical acquired images. The exemplar
draw, from a pool of **420,304** eligible Task-1 objects, is what moves:

| arm | selection identical across seeds | replay overlap s0∩s1 | s0∩s2 | memory size |
|---|---|---:|---:|---:|
| A | yes | **1** | **1** | 400 |
| U | yes | **1** | **0** | 400 |

The three "seeds" of a deterministic arm are therefore *the same acquisition,
rehearsed on three essentially disjoint 400-object memories*, trained with the
same PROB seed.

### 2.3 There is no common-random-number pairing — MEASURED

The eligible exemplar pool excludes the images the trajectory just bought
(`item.image_id not in spent_images`), so a change in the acquired image set
propagates into a different rehearsal set. Measured sensitivity, arm `A`:

| change to the acquired image set | exemplars shared with the reference |
|---|---:|
| none | 400 / 400 |
| 1 fewer image | **380 / 400** |
| 2 fewer | 380 / 400 |
| 5 fewer | **352 / 400** |
| 10 fewer | 353 / 400 |

So a selection difference of a single image moves ~5% of the rehearsal set, and
five images move ~12%. `A` and `A*C` at the same seed are **not** matched on
rehearsal unless their acquired image sets are identical.

### 2.4 The irreducible residual

PROB never calls `torch.use_deterministic_algorithms`, and MSDeformAttn's
backward accumulates with atomics. Bit-identical inputs therefore still give
different weights.

**Conclusion.** If `labelled_ids.txt` and `replay_ids.txt` match for `A` and
`A*C` at a seed, then the arms received *identical supervision and identical
rehearsal under an identical PROB seed*, and the entire reported difference is
CUDA nondeterminism — i.e. the noise floor. If they differ, the difference is
confounded by a different rehearsal set. **In neither case does the paired
design isolate `C`.** The noise floor of this pipeline has never been measured,
and without it a 1–3 point mAP difference cannot be read at all.

---

## 3. Why acquisition quality did not transfer — and why `new_class_AP50 ≈ 0`

### 3.1 The chain, exactly — MEASURED

```
600 regions  ->  opened images  ->  trainable images  ->  GT boxes PROB keeps
```

PROB's `ft` mode applies `remove_unknown_instances`, which keeps
`category_id in range(0, prev + current)` = the **20** classes declared at t2.
Everything else on an opened image is dropped, not taught as background.

| arm | regions | opened | **barren** | trainable | **supervised boxes** | boxes/region | undeclared dropped | person share | **new-class boxes** | new-class images |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random s0 | 600 | 405 | 133 | 272 | 1,145 | 1.91 | 1,553 | 0.78 | 35 | 15 |
| random s1 | 600 | 413 | 140 | 273 | 1,226 | 2.04 | 1,435 | 0.76 | 37 | 16 |
| random s2 | 600 | 400 | 131 | 269 | 1,162 | 1.94 | 1,415 | 0.77 | 23 | 11 |
| **A** | 600 | **590** | **247** | 343 | **972** | **1.62** | **1,759** | 0.79 | **33** | 15 |
| **U** | 600 | **256** | 46 | 210 | **2,027** | **3.38** | 1,332 | 0.72 | **101** | 27 |

Plus, identically for every trajectory, **400 replay boxes** (uniform, 21 per
Task-1 class).

### 3.2 The exact explanation for `new_class_AP50 ≈ 0`

It is **not** a pipeline bug. All four candidate bugs were checked and cleared:

* **class range** — `traffic light` is `CLASS_ORDER[19]`, inside `range(0, 20)`,
  so it survives `remove_unknown_instances`. The evaluator's AP50 vector shows
  19 non-zero entries then `0.0` at index 19, which is exactly that slot.
* **name mapping** — `traffic light` is spelled identically in the annotations
  and in `CLASS_ORDER`; no cocofication applies to it.
* **`known_plus_selected`** — it does *not* exclude the new class. On the GPU
  path every declared-class box on an opened image enters training whether or
  not it was clicked, so all 33 traffic lights on `A`'s images were supervised.
* **task mapping** — the bridge passes `prev=19, current=1`, so `seen = 20`.

The cause is **supply**, in two stages.

**Stage 1 — no arm could acquire the new class, because it is barely in the
population.** MEASURED attrition of `traffic light` through the frozen
construction:

| stage | proposals | distinct objects |
|---|---:|---:|
| full frozen pool (80,000 rows, 1,600 images) | 65 | 26 |
| after P2 (top 30% by `A`, per-image NMS IoU 0.60) | 14 | — |
| **∩ committed annotations → the Method V3 population** | **3** | **2** |

Two acquirable instances in 8,010 proposals. Measured acquisitions of a
*declared* object, over the 600 regions: `random` **1**, `A` **0**, `U` **0**.

This is *not* an `A` bias against small objects — the data refutes that reading.
Traffic-light proposals sit **above** the pool median on `A` (0.1732 vs 0.1342)
and are **larger** than the median box (normalised area 0.0528 vs 0.0206), and
46% of them are in the top 30% by `A`. The attrition is P2's top-30% plus NMS
(65 → 14) and then the committed-annotation restriction from 1,599 to 839 images
(14 → 3).

**Stage 2 — all new-class supervision therefore arrived as a by-product, and far
too little of it.** The 23–101 traffic-light boxes that did reach training came
from `known_plus_selected`'s free labels on whichever images happened to be
opened, not from selection:

* `A` supervised **33** instances on 15 images; `random` 23–37; `U` **101** on 27.
* Against **766–1,457 `person`** boxes — 72–79% of all supervision — plus 400
  replay boxes of the 19 old classes.
* Five epochs at `lr 2e-4`, batch 2, with `freeze_prob_model=True`: the model
  must grow a 20th classifier row from tens of examples while retaining 19.
* **`U` had 3× more new-class instances than `A` and still scored ≈ 0.** The
  shortage is an order of magnitude, not a margin.

So `new_class_AP50 ≈ 0` for *every* arm was determined before any arm ran: the
new class was neither selectable in the population nor supplied in sufficient
quantity by the free-label channel.

### 3.3 The transfer channel is closed at a single task — MEASURED

`A` acquired **150 distinct unknown objects** and **79 medium+tail objects** —
3.8× and 4.0× `random`. "Unknown" in the frozen pool means "not one of t1's 19
classes", and of the **42** unknown classes present in the population exactly
**one** — `traffic light` — becomes declared at t2. MEASURED, per arm:

| arm | acquired unknown objects | declared at t2 | **dropped by the class filter** |
|---|---:|---:|---:|
| random | 40 | 1 | **39** |
| **A** | **150** | **0** | **150** |
| U | 36 | 0 | **36** |

So **all 150** of `A`'s discoveries were dropped by `remove_unknown_instances`.

`A`'s entire acquisition advantage contributed **zero** supervision. At one task
the acquisition metric and the learning metric are causally disconnected. This is
the transfer bottleneck, and it is structural, not statistical.

`owl.runner.run_chain` implements the channel that would close the gap —
`reuse_deferred_labels` banks an image whose class becomes declarable at a later
task, at no further annotation cost — and the single-task Method V3 design does
not use it.

### 3.4 The primary metric could not respond — MEASURED

Per-class supervised positives among the eight medium+tail classes:

| class | band | random s0 | random s1 | random s2 | A s0 | U s0 |
|---|---|---:|---:|---:|---:|---:|
| aeroplane | medium | 3 | 2 | 2 | 3 | 3 |
| cat | medium | 26 | 19 | 31 | 35 | 24 |
| dog | medium | 16 | 22 | 22 | 28 | 14 |
| train | medium | 10 | 9 | 8 | 8 | 10 |
| elephant | medium | 3 | 1 | 2 | 1 | 7 |
| zebra | medium | 2 | 2 | 2 | 2 | 2 |
| giraffe | medium | 1 | 1 | 1 | 1 | 1 |
| **bear** | **tail** | **0** | **0** | **0** | **0** | **0** |

`bear` — the entire tail band at t2 — received **zero** new supervision in every
arm, and four declared classes (`bus`, `cow`, `sheep`, `bear`) got zero boxes
under `A`. `mAP50_medium_tail` at t2 is therefore a **retention** measurement,
dominated by the 400 identical replay boxes and by whatever the t1 checkpoint
already knew — not an acquisition measurement.

That is why `random = 73.96` and `A = 74.32` are ~0.4 apart while their
acquisition differs fourfold. **The frozen primary metric was structurally
incapable of responding to selection quality at this task.**

---

## 4. Is "600 regions" a fair annotation budget?

**No — it is matched on oracle questions asked, not on supervision delivered.**
MEASURED:

| arm | regions | regions per opened image | supervised boxes | **boxes per region** |
|---|---:|---:|---:|---:|
| A | 600 | 1.02 | 972 | **1.62** |
| random | 600 | 1.45–1.50 | 1,145–1,226 | 1.91–2.04 |
| U | 600 | 2.34 | 2,027 | **3.38** |

`U` receives **2.09×** the supervision of `A` for the identical oracle cost. The
mechanism is `known_plus_selected`: known-class objects on an opened image are
free, so the dominant supervision channel is *how crowded the opened images are*,
and an arm that spreads thinly over sparse images is systematically starved. `A`
opens 590 images for 600 regions and 247 of them (**42%**) contain no declared
class at all.

The budget is self-consistent as an *annotation-cost* comparison — that is a real
and defensible axis, and `owl.discovery` already reports both cost axes. But it
is **not** a supervision-matched comparison, and a detector endpoint is a
function of supervision. The two arms that looked best on acquisition are the two
worst on supervision.

---

## 5. Interpretation, separated

### A. PRE-REGISTERED

**`METHOD_V3_ALLOWED_LADDER` unchanged; `C_DOWNSTREAM_NOT_SUPPORTED`.** All 12
trajectories completed, the criterion was applied mechanically, and it is not
reinterpreted. Consistency `C` did not produce a detectable downstream benefit
over `A` under this protocol.

### B. POST-HOC SUPPORTED

1. `A`'s ranking is dense to 3 × 10⁻⁵ at the 600-cut, so the `A`-prefix cannot be
   invariant to any realistic `C`; the identical aggregates for `A` and `A*C`
   are an aggregate coincidence, and the four reported statistics are blind to
   background↔background swaps in a narrow band.
2. PROB's `--seed` was 0 in all twelve trajectories. The Method V3 seed varied
   only the 400-object rehearsal set, which is essentially disjoint across seeds
   (overlap 0–1 of 400).
3. There is no common-random-number pairing: one changed acquired image moves
   ~5% of the rehearsal set.
4. `new_class_AP50 ≈ 0` is a supply result, not a bug. The population holds
   **2** acquirable `traffic light` objects (65 proposals / 26 objects in the
   full pool, 14 after P2, 3 after the annotation restriction), so no arm could
   select the new class; the 23–101 instances that reached training came from
   the free-label channel, against 766–1,457 `person` boxes over five epochs
   with the objectness head frozen. Class range, name mapping, labelling policy
   and task mapping were each checked and are correct.
5. All **150** of `A`'s acquired unknown objects were dropped by PROB's
   declared-class filter — of the 42 unknown classes in the population, only
   `traffic light` is declared at t2. At a single task, acquisition cannot reach
   the detector.
6. `bear` — the whole tail band — received zero new supervision in every arm, and
   medium classes received 1–35 boxes. `mAP50_medium_tail` at t2 measures
   retention, so the primary metric could not respond to selection.
7. "600 regions" is not supervision-matched: 1.62 boxes/region for `A` against
   3.38 for `U`, a 2.09× gap at identical oracle cost.

### C. HYPOTHESES — each needs a new experiment

1. The `A` vs `A*C` and `A` vs `random` differences are inside the pipeline's
   nondeterminism noise floor. **The noise floor has never been measured.**
2. With enough new-class instances PROB *can* learn a 20th class under this
   fine-tuning recipe. Untested; 101 instances was not enough.
3. Selection quality would transfer over a multi-task chain, where banked
   labels of not-yet-declared classes become supervision later.
4. Matching arms on delivered supervision rather than on regions would change
   the ranking of the arms.
5. `C` might help under a supervision-matched budget. Unsupported either way by
   this run.

### D. INVALID CLAIMS — must not be made from these results

1. That `C` carries no useful selection signal. This run could not have detected
   one: the primary metric measures retention, and the paired design does not
   isolate `C`.
2. That `A` is no better than `random` at acquisition. It is 3.8× better at
   acquiring distinct unknown objects; what failed is the *transfer*.
3. That `random ≈ A` downstream means selection does not matter. Both received
   near-identical *supervision*, which is what the detector saw.
4. That the detector cannot learn new classes. Nothing here tested that with
   adequate data — the population held 2 acquirable instances of the new class.
5. That `A` suppresses small objects, or that box size explains the missing new
   class. Measured and refuted: traffic-light proposals are above the pool
   median on `A` and larger than the median box.
6. Any significance claim. n = 3, the seeds share their acquisition, and the
   noise floor is unknown.
7. That the tail band was measured. `mAP50_tail` at t2 is one class, `bear`,
   which received zero new boxes.
8. That the arms are supervision-matched. They differ by up to 2.09×.

---

## 6. The single recommended next experiment

> **Measure the noise floor and prove the new-class channel can move at all,
> before running any further selection comparison.**

This is option 2 of the four offered — *investigate the new-class learning
pipeline* — narrowed to the cheapest form that can **falsify** the possibility of
ever measuring what Method V3 set out to measure.

**Why not the others.**

* *Equalise annotation semantics and rerun selection* (option 1) is premature.
  With `bear` at zero boxes and medium classes at 1–35, the primary metric still
  measures retention, and with an unknown noise floor a null result would again
  be unreadable. Twelve trajectories, ~5 h, and it reproduces §3.4.
* *Abandon semantic selection for replay B* (option 3) discards a component
  whose selection advantage is real and measured (150 vs 39 distinct unknown
  objects) and whose transfer was never given a working channel. Abandoning on
  this evidence would be the §5-D-1 error.

**What it would consist of** — not implemented, not run:

1. **Noise floor.** One acquisition, held byte-identical, trained and evaluated
   3–4 times with everything fixed including the rehearsal set. This gives the
   standard deviation attributable to CUDA nondeterminism alone, which is the
   number every past and future comparison needs. Cost: ~2 h.
2. **New-class positive control.** The same task and recipe, given a saturated
   new-class supply drawn from outside the candidate population — the benchmark
   holds 11,431 `traffic light` training objects, so a few hundred instances is
   cheap — as an upper bound. If `new_class_AP50` is still ≈ 0 with hundreds of
   instances, the bottleneck is the fine-tuning recipe (epochs, learning rate,
   class balance, frozen objectness head) and **no selection experiment is worth
   running until it is fixed**. If it moves, the recipe is sound and the shortage
   was genuinely supply — which §3.2 shows the frozen population cannot fix,
   because it contains 2 acquirable instances.

Both are single-task, reuse the existing pipeline unchanged, and together cost
less than a third of Method V3. They are diagnostics, not a new contribution
claim, and the outcome determines whether option 1 or a recipe fix is next.

Two smaller repairs are worth folding in whenever the next run happens, and are
recorded here rather than done silently:

* pass a per-trajectory `seed` to `owl.bridge.Bridge` so PROB's own RNG is part
  of the design instead of a constant;
* report `boxes_per_region` and `images_barren` alongside every acquisition
  metric, so a supervision gap like §4's cannot again be invisible in the tables.

---

## 7. Provenance

| | |
|---|---|
| audited verdict | `C_DOWNSTREAM_NOT_SUPPORTED`, 12/12 trajectories |
| tool | `tools/audit_method_v3.py` (read-only on `--results`) |
| tables | `audit_a_gap_structure.csv`, `audit_c_distribution.csv`, `audit_prefix_overlap.csv`, `audit_seed_effect.csv`, `audit_supervision_chain.csv`, `audit_per_class_supervision.csv`, `audit_paired_*.csv`, `audit_summary.json` |
| population | 8,010 proposals on 839 images (P2 ∩ committed annotations) |
| PROB source read | `gubiczam/PROB` @ `4c66be1a52cad9360e09c729e9134aba8fe0b531` |
| Method V3 results | **not modified** |
