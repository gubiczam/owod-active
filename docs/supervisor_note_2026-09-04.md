# Distribution-aware active annotation for open-world detection
## Controlled full-chain benchmark — development-seed results

2026-09-04. Four arms complete at seed 0. **Seed 0 is the development seed; no
statistical claim is made from it.**

---

## 1. The setup

A detector is given a sequence of tasks. At each one it may buy annotation for a
fixed budget, then it is fine-tuned and scored. The question is whether choosing
*what to annotate* helps.

| | |
|---|---|
| detector | PROB (Deformable-DETR, `dino_resnet50`), published S-OWODB `t1.pth` |
| chain | `t1 -> t2 -> t3 -> t4`, **one new class per task** |
| declared | t2 `traffic light` (head) · t3 `fire hydrant` (**tail**) · t4 `stop sign` (**tail**) |
| candidate pool | 2 000 fresh images per task, PROB's own top-50 proposals each, per-image NMS at IoU 0.60 |
| budget | **3 000 oracle answers** per task; `cost(image) = max(1, annotated objects on it)` |
| labelling | full-image: the annotator opens an image and labels everything in it |
| rehearsal | `uniform`, **400 exemplar objects**, identical for every arm |
| training | 5 epochs, lr 2e-4, batch 2, `ft` mode, objectness head frozen |
| evaluation | one **frozen 837-image split**, used for the anchor and every task of every arm |

The chain is genuinely sequential: `t3` fine-tunes `t2`'s own checkpoint *for
that arm*, `t4` fine-tunes `t3`'s. No task restarts from the anchor and no two
arms share a checkpoint; both are asserted against a stubbed detector.

### 1.1 This is not the published S-OWODB task protocol

**Read this before comparing any number to the literature.** The published
S-OWODB split declares 19 / 21 / 20 / 20 classes across four tasks. This chain
declares **one class per task**. It is the repository's own incremental
protocol, chosen because a 21-class step is unaffordable at any annotation
budget available here and because a one-class step makes new-class learning
*visible* — a 21-way average hides every acquisition decision.

**No number in this note may be compared against a published S-OWODB result.**

What the choice buys: the tail band **grows along the chain** — `{bear}` after
t2, `{bear, fire hydrant}` after t3, `{bear, fire hydrant, stop sign}` after t4
— so tail performance at t4 is partly a function of what the selector acquired,
which is the long-tail claim.

---

## 2. The four arms

| arm | what it selects | role |
|---|---|---|
| `random` | uniform over the deduplicated pool | the reference |
| `admissibility` | `A(x) = objectness(x) · sqrt(area(x))`, raw | the strongest learning-free prior this project has |
| `entropy` | normalised Shannon entropy of PROB's class posterior | the standard uncertainty baseline |
| `proposed-v1` | A-gated k-center in frozen DINOv2 crop space | the proposed method |

Raw objectness was **measured and rejected** rather than given a trajectory: on
the committed pool its rank correlation with `A` is 0.281 and their top-600
prefixes are disjoint, but its first 600 picks contain **2** real annotated
objects against `A`'s **284**. Distinct, and degenerate.

---

## 3. Detector results, seed 0

| | random | admissibility | proposed-v1 | entropy |
|---|---:|---:|---:|---:|
| final `known_mAP50` (t4) | **50.13** | 48.34 | 44.89 | 48.03 |
| final `mAP50_tail` (t4) | 36.83 | **48.53** | 37.20 | 47.11 |
| mean `new_class_AP50` (t2–t4) | 2.40 | 7.12 | 0.00 | **7.31** |
| mean `U_Recall50` (t2–t4) | 15.34 | 15.47 | **18.05** | 16.98 |

Against the reference arm:

| vs `random` | admissibility | proposed-v1 | entropy |
|---|---:|---:|---:|
| `known_mAP50` | −1.79 | **−5.24** | −2.10 |
| `mAP50_tail` | **+11.70** | +0.37 | +10.28 |
| mean `new_class_AP50` | **+4.72** | −2.40 | +4.91 |
| mean `U_Recall50` | +0.13 | **+2.71** | +1.64 |

The pre-registered primary contrast, `proposed-v1` vs `admissibility` at t4:
`known_mAP50` **−3.46**, `mAP50_tail` **−11.33**, mean `new_class_AP50`
**−7.12**, mean `U_Recall50` **+2.58**.

---

## 4. Acquisition and supervision cost

*To be filled from this run's own `acquisition.csv`, `supervision_cost.csv` and
`per_class_ap.csv`. The columns are already recorded per arm and per task; the
numbers are not reproduced here because they have not been read off the run.*

| per arm, per task | column |
|---|---|
| objects of the task's own new class acquired | `acquired_new_class` |
| distinct classes acquired | `acquired_classes` |
| acquired now, learnable at t3 / t4 | `acquired_becomes_known_t3/t4` |
| oracle answers charged | `answers_spent` |
| images opened / trainable / barren | `images_opened`, `images_trainable`, `images_barren` |
| boxes the annotator produced | `boxes_labelled` |
| boxes declared at this task | `boxes_supervised` |
| **boxes PROB was actually handed** | `boxes_trained_on` |
| gradient steps taken | `training_iterations` |

Two of these must be quoted whenever an AP difference is discussed:

* **`boxes_labelled` is the matched quantity.** The budget is counted in oracle
  answers precisely so that annotation *effort* is equal. Simulated on the
  committed pool at the real budget, the five selectors land within **6 %** of
  each other on labelled boxes (2 644 – 2 809). The predecessor experiment used
  a *region* budget and handed one arm **2.09×** another's supervision.
* **`boxes_trained_on` is not matched, and that is a result rather than a
  confound.** Which fraction of a labelled image is *currently* learnable is
  not under the selector's control. An arm that buys objects of classes not yet
  declared converts annotation into current-task supervision poorly.

The four completed arms predate the `boxes_trained_on` column;
`tools/backfill_boxes_trained_on.py` recovers it offline from each task's own
`train/labelled_ids.txt` and writes a separate CSV without touching the results.

---

## 5. Interpretation

### 5.1 Proposed-v1 is a negative downstream result

Mean `new_class_AP50` is **0.00** — not small, zero, across all three tasks —
while `known_mAP50` is the lowest of the four and `mAP50_tail` sits at the
reference arm's level. It is worse than `random` on the metric the method
exists to improve.

**Why, mechanistically.** `proposed-v1` simultaneously has the **best**
`U_Recall50` of any arm (18.05, +2.71 over `random`). It found unknown content;
it did not find *enough instances of one class*. Farthest-first traversal leaves
a region of semantic space as soon as that region is covered, so it **caps
per-class multiplicity by construction**: it buys one or two of everything.
Learning a new class needs tens of instances of that one class. The objective is
misaligned with the endpoint, and there is no parameter to blame because the
method has none.

A second mechanism, consistent with the same numbers but **not** established:
the reference the traversal measures against contained the balanced task-1
reference — 19 head classes of street and animal content. The three declared
classes occur *only* in street scenes, i.e. exactly the images most resembling
that reference, so coverage-against-labelled-data may systematically avoid the
images holding the class being learned. This would also explain the lowest
`known_mAP50`: annotation spent where the detector had least to gain.

**What was supposed to separate those two mechanisms is unavailable.** The
`coreset` arm — the same traversal with the admissibility gate switched off —
was to have decided whether the failure came from the gate, from the coverage
objective, or from their interaction, against thresholds fixed before it ran.
Its seed-0 run **terminated with CUDA OOM and reported no detector endpoint**.
**The gate is therefore not causally ruled out.**

### 5.2 High U-Recall did not become new-class AP

U-Recall counts how much unknown content the detector still localises as
*unknown*. `new_class_AP50` counts how well it has learned one *specific*
declared class. Breadth of unknown discovery and depth on a single class are
different quantities, and a coverage objective maximises the first at the
expense of the second. `proposed-v1` is the clean demonstration: best breadth,
zero depth.

This is a result worth reporting in its own right — it says the acquisition
metrics this project measured in earlier work (distinct unknown objects, distinct
unknown classes) are **not** sufficient proxies for incremental learning.

### 5.3 Admissibility and entropy are the strongest current baselines

They agree closely and beat both the reference and the proposed method on
new-class learning: mean `new_class_AP50` 7.12 and 7.31 against `random`'s 2.40,
and `mAP50_tail` 48.53 and 47.11 against 36.83. Both give up ~2 points of
`known_mAP50` relative to `random` (48.34 and 48.03 against 50.13) to do it —
a visible and modest stability/plasticity trade.

`random` has the **highest** `known_mAP50` and the lowest `U_Recall50`: spread
uniformly over the pool, most of what it buys is head-class content the detector
already handles, which protects retention and discovers nothing.

### 5.4 Final tail mAP is not an independent result — but it is not a restatement either

`mAP50_tail` at t4 averages AP over `{bear, fire hydrant, stop sign}`. `bear`
receives **no** new annotation for any arm, so its contribution is a retention
term roughly common to all four. The other two are the classes declared at t3
and t4. The ~11-point separation between {`admissibility`, `entropy`} and
{`random`, `proposed-v1`} is therefore explained by new-class learning and
should not be presented as a second, independent finding.

**Correction to an earlier statement of mine:** the two orderings are *not*
identical. Rank correlation between `mAP50_tail` and mean `new_class_AP50`
across the four arms is **0.6**, not 1.0 — `admissibility` leads on tail while
`entropy` leads on mean new-class AP, and `proposed-v1` edges `random` on tail
while trailing it on new-class AP. The two measure at different points:
`new_class_AP50` scores each class **at its own task**, `mAP50_tail` scores
`fire hydrant` **at t4**, one task of forgetting later. Decomposing the gap needs
`per_class_ap.csv`.

### 5.5 What one development seed can and cannot support

**Can:** that the pipeline runs end to end and sequentially; that a 0.00 on a
metric with adequate supply is a mechanism rather than noise; that the
acquisition and supervision ledgers are matched where they were designed to be;
directions, ranked, with the caveat attached.

**Cannot:** any statement of the form "X is better than Y", any confidence
interval, any significance. The nondeterminism floor of this pipeline is
**unmeasured** — an earlier audit established that the detector's own seed had
never been varied and that paired arms share no common random numbers, so two
identical configurations could differ by an unknown amount. Seeds 1 and 2 for
the three baselines are the next mandatory work.

---

## 6. Proposed-v2

**Awaiting development-seed result.**

Frozen and implemented, **not yet run**. Informativeness first, diversity
second: the admissibility gate, then entropy at or above the median of the gated
population, then farthest-first *inside* that subset with the task-1 reference
removed — DINOv2 demoted from novelty objective to redundancy remover. One new
explicit design choice (the median filter). **Designed after inspecting
Proposed-v1's seed-0 endpoints, therefore development-seed-informed and not
pre-registered.** A kill rule is frozen in code: it earns seeds 1 and 2 only if
seed 0 gives mean `new_class_AP50` ≥ 3.56 and final `known_mAP50` ≥ 44.89.

---

## 6.1 Recommended figures for the meeting

Four, in this order, all produced by `tools/plot_full_owod_benchmark.py` and
`tools/summarize_full_owod_benchmark.py` from this run's own CSVs.

1. **`plots/new_class_AP50.png`** — new-class AP50 against task, arms as lines.
   *Shows the headline outcome in one panel: `admissibility` and `entropy` at
   ~7 across all three tasks and `proposed-v1` flat at zero, so the negative
   result and the baseline result are read off the same axes.*
2. **`plots/acquired_new_class.png`** — objects of the task's own new class
   acquired, against task. *Shows the cause of (1) measured **without the
   detector**: whether an arm ever bought instances of the class it was then
   asked to learn, which separates an acquisition failure from a learning
   failure.*
3. **The annotation-cost table** from `supervision_cost.csv` — per arm and task:
   `boxes_labelled`, `boxes_trained_on`, `images_opened`, `images_barren`,
   `training_iterations`. *Shows the comparison is trustworthy: annotation
   effort is matched to within a few per cent by construction, while what
   actually reached the detector is not, and is reported rather than assumed.*
4. **`plots/annotation_efficiency.png`** — the 2x2 of `known_mAP50`,
   `new_class_AP50`, `U_Recall50` and `mAP50_tail` against **cumulative oracle
   answers**. *Shows the project's actual research axis — performance per unit
   annotation — and carries the stability/plasticity trade honestly, since
   `random` visibly wins the `known_mAP50` panel.*

Two to have open but not to present: `plots/U_Recall50.png`, which makes the
breadth-without-depth contrast explicit against figure 1, and
`plots/mAP50_tail.png`, for the long-tail question with the section 5.4 caveat
attached.

## 7. Stated limitations

1. Not the published S-OWODB task split (§1.1).
2. One development seed; no error bars; nondeterminism floor unmeasured.
3. The gate ablation is missing (CUDA OOM), so Proposed-v1's failure is not
   causally localised.
4. `boxes_trained_on` is not matched across arms; `training_iterations` differ.
5. **Banking recovers only wholly-barren images.** An acquired image that also
   held an already-known class is trained on immediately and never re-offered,
   so its not-yet-declared boxes are never learned from. Measured on the
   candidate index: **67.6 %** of `fire hydrant` objects and **65.8 %** of
   `stop sign` objects sit on such mixed images. This lowers the new-class
   ceiling for **every** arm and is a property of the protocol, not of any
   method. See `docs/full_owod_experiment_log_2026-09.md`.
6. Genuine iterative acquisition (detector rescoring between annotation rounds)
   is not tested; selection is one pass per task.
7. `freeze_prob_model=True` — PROB's own fine-tuning procedure — keeps the
   objectness head fixed, so U-Recall can move only through the classification
   head.
