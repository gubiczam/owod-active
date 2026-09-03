# Full OWOD Active Selection Benchmark V1 — protocol

Frozen 2026-09-03, before any trajectory ran. Code: `owl/active_selection/`.
Research log: `docs/full_owod_experiment_log_2026-09.md`.

---

## 0. What this experiment is, and what it replaces

Method V1 audited the representation. Method V2 audited a better representation
and froze `D_NO_GO / R_NO_GO / C_GO`. Method V3 tested whether the one component
that passed transferred to detector learning at a single transition and froze
`C_DOWNSTREAM_NOT_SUPPORTED`. Its post-hoc audit
(`docs/method_v3_posthoc_audit_2026-09-03.md`) established that the isolated
`t1 -> t2` design **could not have** answered the question it was asked: the
acquirable population held two instances of the only class that becomes
declarable, and the region budget handed one arm 2.09x the supervision of
another.

**None of those verdicts is reopened here.** This is a different experiment with
a different question:

> Across the task sequence, does active selection improve discovery, learning,
> retention and annotation efficiency relative to standard baselines — and does
> a simple coverage-aware method improve on those baselines?

Component diagnostics are over. The endpoint is the detector, at every task.

## 1. The task chain — read this before any number is quoted

The chain is the repository's canonical one, `owl.protocol.build_chain(4)`:

| task | declares | frequency group | train objects | tail band known after |
|---|---|---|---|---|
| t1 | — (PROB's published `t1.pth`, 19 classes) | — | — | `bear` |
| t2 | `traffic light` | head | 11 431 | `bear` |
| t3 | `fire hydrant` | **tail** | 1 228 | `bear`, `fire hydrant` |
| t4 | `stop sign` | **tail** | 1 277 | `bear`, `fire hydrant`, `stop sign` |

**This is not the published S-OWODB task split** (19 / 21 / 20 / 20). It declares
one class per task. No number produced here may be compared against a published
S-OWODB result, and every table says so. The choice is the repository's, not a
new invention, and it is kept for two reasons:

1. a new-class endpoint is measurable at an annotation budget we can afford —
   21 classes at once cannot be;
2. **the tail band grows along the chain.** At t2 it holds one class and
   `mAP50_tail` is a pure retention number. At t4 it holds three, two of them
   learned during the chain, so `mAP50_tail` at t4 is directly a function of
   what the selector acquired. That is the long-tail claim, and it is visible
   only across the sequence — which is exactly what the isolated Method V3
   design destroyed.

Sequentiality is enforced, not assumed: `t3` fine-tunes `t2`'s own checkpoint
*for that arm*, `t4` fine-tunes `t3`'s, no task restarts from the anchor, and no
two arms or seeds share a workspace or a checkpoint.
`tests/test_full_benchmark_chain.py` asserts the lineage against a stubbed
detector.

## 2. The arms

Five, in the pre-declared execution order
(`owl.active_selection.arms.ORDER`):

1. **`random`** — the reference.
2. **`admissibility`** — `A(x) = objectness(x) * sqrt(area(x))`, raw. The bar:
   it has beaten every semantic score this project has built.
3. **`proposed`** — A-gated semantic k-center. Section 3.
4. **`entropy`** — normalised Shannon entropy of PROB's class posterior. The
   standard uncertainty baseline. Measured weak in Method V3 (36 distinct
   unknown objects against admissibility's 150) and kept anyway.
5. **`coreset`** — the same traversal as `proposed` with the gate **off**, over
   the whole deduplicated pool. The recognisable core-set baseline and the
   one-variable ablation of the gate.

**Raw objectness does not get a trajectory, and here is the measurement.** On
the committed pool, raw `objectness` and `A` have Spearman rank correlation
**0.281** and their top-600 prefixes are **disjoint** (Jaccard 0.000) — so they
are genuinely distinct, not redundant. But raw objectness's first 600 picks
contain **2** real annotated objects against `A`'s **284**: it is distinct *and*
degenerate, because it prefers tiny high-confidence boxes. It is reported as a
measurement, not run as an arm. Note that `owl.selection.ARMS['objectness']`
already computes `A`; the name in that older registry is a misnomer.

### 2.1 Shared candidate population

Every arm buys from the same population, or the comparison is between
populations rather than selectors.

* **`P_nms`** — the top 50 proposals per image by PROB's own objectness order,
  then **per-image NMS at IoU 0.60 ordered by `A`**. Deduplication, not
  selection: two proposals on one object cost the annotator once. Applied to
  every arm.
* **`G`** — the top **0.30** of `P_nms` by `A`. The *gate*. Used by `proposed`;
  deliberately not by `coreset`.

Both thresholds are the frozen values of the established `P2` recipe.
`owl.active_selection.population.p2_reference` reproduces that recipe's own
order (gate then NMS) and `tests/test_active_selection.py` asserts it lands on
**15 518** rows at a **0.767** background share — the committed numbers — so
this implementation is pinned to the one Methods V2 and V3 were measured on.
Benchmark V1 deduplicates *before* gating so that an ungated arm has the same
population to select from; that reordering is the only difference and it is
measured (`P_nms` = 63 997 rows, `G` = 19 199 rows, background 0.799 against
`P2`'s 0.767).

## 3. The proposed method, v1

**A-gated semantic k-center.**

1. deduplicate: `P_nms`;
2. gate on object-likeness: keep `G`, the top 30 % by `A`;
3. embed those crops with **frozen DINOv2 ViT-B/14**, the Method V2 crop
   (`owl.semantic_features`: square, 1.20x the larger side, shift-before-shrink,
   never padded, 224x224 bicubic, ImageNet normalisation, CLS token,
   L2-normalised);
   Only `G` is embedded, not the whole pool — 24 000 crops per task instead of
   80 000. That is not only cheaper, it is what the method means: cover the
   semantic space of the *object-like* candidates. The ungated `coreset` control
   embeds all of `P_nms`, which is the point of it.
4. **farthest-first traversal** (k-center greedy) in that space against the
   labelled reference `R`:

       pick argmax_{x in G, unbought} min_{r in R} (1 - cos(x, r))

   `R` starts as the balanced task-1 labelled reference (the frozen
   `ref_t1_dinov2_vitb14_cap1000_v1.npz`) and grows by **every candidate on
   every image opened**, because full-image labelling labels all of them;
5. stop when the next image does not fit in the remaining budget.

**It has no hyperparameter.** That is the reason it was chosen over the additive
form the original plan proposed. `lambda`, `gamma` and `mu` each need a number,
and any number picked after seeing a detector endpoint makes the result a tuned
one. Here the candidate set, the reference set and the traversal are fixed by
this document and there is nothing left to choose.

**Why these components, given what failed.** This is the §6 substitution, stated
explicitly:

| original component | Method V2 verdict | what it becomes here |
|---|---|---|
| `D` — novelty vs the labelled t1 reference | `D_NO_GO` as an unknown-vs-background separator | kept as **coverage**, which is a different quantity: distance to what is already labelled, used to spread the batch, not to classify a region as unknown |
| `R` — semantic rarity | `R_NO_GO` | **removed.** Rarity is not estimated at all; covering under-covered regions of the space is the mechanism instead |
| `C` — crop/view consistency | `C_GO` as a component, `C_DOWNSTREAM_NOT_SUPPORTED` | **removed** |
| `U` — entropy | weak in Method V3 | **removed** from the proposed method; it stays as its own baseline arm |
| `A` — admissibility | strongest thing measured | kept, as the **gate** |

DINOv2 is used only for what it was measured to be good at: semantic relations
among real object-like candidates. Object-versus-background is left to PROB's
objectness, which is what `A` is, because DINOv2 was measured **not** to
separate those (`D_NO_GO`).

## 4. Annotation cost — the confound Method V3 exposed

**Primary unit: the oracle answer, under full-image labelling.**

    cost(image) = max(1, annotated objects on that image)

The annotator is handed an image and labels everything in it. This is the
2026-08-25 consultation's own answer to "only the selected box, or everything in
the image?", and it removes half-labelling by construction.

**Why not regions.** Method V3's audit measured, at a 600-*region* budget:
admissibility 1.62 supervised boxes per region, entropy 3.38 — a 2.09x
difference in what the detector was taught at identical nominal cost.
**Why not images.** The same arms sit at 1.65 and 7.9 boxes per opened image, a
4.8x difference. Counting answers is the only unit that equalises what the
detector is taught.

Opening an image is therefore the acquisition unit: a second selected region on
an already-open image buys nothing and is charged nothing, and the row says so
(`positions_redundant`) rather than reporting a region count that no longer
means anything.

**What this unit does fix, measured before the run.** Simulating all five
selectors on the committed pool at 3 000 answers
(`tools/plan_full_owod_benchmark.py`) gives **2 644 – 2 809 labelled boxes** —
matched to within 6 % — against Method V3's 972 – 2 027, and **140 – 233
trainable images**, matched to within 17 %.

**What it does not fix, also measured.** Of those matched labels, the share whose
class is *already declared* is not matched: at t2 the simulation gives 480
supervised boxes for `admissibility` against 1 812 for `entropy`. That is not a
hidden confound but the mechanism under study — a selector that buys objects of
classes not yet declarable converts annotation labour into current-task
supervision poorly, and the banking of section 6 is precisely what should redeem
it at t3 and t4. It is impossible to match both: which fraction of a labelled
image is currently learnable is not under the selector's control. So it is
reported as an outcome, in every row, and `training_images` /
`training_iterations` must be quoted whenever an AP difference is discussed.

A design matched on *supervised boxes* as well is a pre-declared **sensitivity
comparison for the two finalists after seed 0**, and it is Phase 2 — not a
substitute for the primary unit, which prices what an annotator actually does.

Recorded per arm, per task, always:

`answer_budget`, `answers_spent`, `answers_unspent`, `images_opened`,
`answers_per_image`, `positions_scanned`, `positions_redundant`,
`boxes_labelled`, `boxes_supervised`, `boxes_banked`, `supervised_share`,
`images_barren`, `boxes_supervised_head/medium/tail`, `per_class_supervised`,
`images_trainable`, `images_from_earlier_tasks`, `training_images`,
`training_iterations`.

An image whose objects are all future-task classes yields no supervision now.
It is **banked**, not lost: `reuse_deferred_labels` returns it at the task where
its class is declared, at no further annotation cost. That is a result in its
own right and section 6 measures it.

## 5. Replay

One fixed policy for every arm, so replay cannot become the explanation:
`uniform`, **M = 400 exemplar objects**, protocol version 3 (objects
materialised as alias annotations, so `sum m_c == |E| == 400` exactly), no
reallocation. Alpha is **not** swept in Benchmark V1; the earlier replay
experiment already measured that uniform against tail-favouring is a
stability/plasticity trade rather than a clean win, and distribution-aware
replay is contribution B, revisited after selection works end to end.

**What is and is not paired.** Exemplar identities are recorded per task
(`replay_ids.txt`). The eligible pool excludes the images the task just bought,
so arms that bought different images necessarily draw different exemplars —
Method V3's audit measured that one changed acquired image moves 20 of 400
exemplars. Common random numbers across arms are therefore **impossible** by
construction, and no claim in this benchmark may assume them. What *is* shared:
the anchor checkpoint, the shared evaluation split, the candidate image sample
at each `(seed, task)`, and PROB's `--seed`.

## 6. Task semantics and leakage

At every task the code distinguishes currently-known classes, the newly declared
class, and future-unknown classes. Selectors are handed a
`owl.proposals.Candidates` built by `from_predict`, which carries **no oracle at
all**; the cost function reads per-image object *counts* and never a class or a
box. `tests/test_active_selection.py` asserts every arm runs on a pool with no
oracle.

Future labels are used **only** post hoc, after the budget is committed, to
produce the acquisition table:

`acquired_objects`, `acquired_classes`, `acquired_known_now`,
`acquired_new_class`, `acquired_becomes_known_t3`, `acquired_becomes_known_t4`,
`acquired_stays_unknown`, `acquired_head/medium/tail_objects`.

This is the table that can show what the isolated design could not: an
acquisition that paid nothing at t2 because its class was not yet declarable,
and paid at t3 or t4 because it was banked.

**The new-class supply is no longer degenerate, and this was checked in advance.**
Method V3's population held **2** acquirable `traffic light` objects, which is
why `new_class_AP50` was approximately zero for every arm and why that null was
unreadable. The candidate index here holds 6 703 `traffic light` objects on
2 159 of 28 800 images, 997 `fire hydrant` on 911, and 1 021 `stop sign` on 900.
Simulated at 3 000 answers, the five selectors acquire **21 – 112** `traffic
light` objects. Non-zero for every arm, by a factor of ten or more, before a
single GPU minute was spent.

## 7. Evaluation

One **shared, frozen** split, built once from the chain's three declared classes
at 150 test images per class plus twice that many sampled others — **837
images** — and used for the anchor and for every task of every arm. Forgetting
is therefore a difference between two numbers measured on the same images.

Per task: `known_mAP50`, `prev_mAP50`, `new_mAP50`, `new_class_AP50`, per-class
AP50, `U_Recall50`, WI, A-OSE, `forgetting` (against the value measured on the
checkpoint this task started from), `drop_from_anchor`, `mAP50_head/medium/tail`,
`U_Recall_head/medium/tail`, `exchange_rate`.

Across the chain: mean `known_mAP50` over t2–t4, mean `new_class_AP50`, mean
`U_Recall50`, cumulative forgetting, and annotation-efficiency curves (each
endpoint against cumulative oracle answers). No single scalar is used to rank
arms.

### 7.1 Stated precision limits — computed before the run, not discovered after

The Method V3 audit's lesson was that a null on a metric with no supply is
unreadable. So the supply of every endpoint is stated here, in advance.

**New-class supply per task.** A 2 000-image candidate pool is expected to hold:

| task | class | images in the pool | objects in the pool |
|---|---|---:|---:|
| t2 | `traffic light` | ~149 | ~465 |
| t3 | `fire hydrant` | ~63 | ~69 |
| t4 | `stop sign` | ~62 | ~70 |

An arm opens roughly 300 of the 2 000 images, so a *random* selector is expected
to acquire about 70, 10 and 10 objects of the three new classes and a perfect
one about 465, 69 and 70. **That is a 7x spread for the selector to move, which
is what makes the acquisition endpoint discriminative.** It is also thin for
*learning*: Method V3 delivered 23–101 instances of a new class and measured
`new_class_AP50 ~ 0`. So:

* `acquired_new_class`, `acquired_classes`, `acquired_tail_objects` — **expected
  to be informative.** Detector-free, 7x of headroom, available even if a
  trajectory fails. These are contribution A's direct evidence.
* `known_mAP50`, `forgetting`, `mAP50_head` — **expected to be informative.**
  Driven by how much and what kind of supervision each arm delivered, over 19–22
  classes with thousands of test objects.
* `new_class_AP50` at t3 and t4 — **may be near zero for every arm**, on a
  supply of tens of instances against ~1 200 supervised boxes an epoch. If it
  is, that is reported as *unreadable*, not as "selection does not help
  new-class learning". This is why it is **not** the primary metric.
* `mAP50_tail` — at t2 the band is `{bear}` alone, with 41 training objects in
  the whole benchmark, so it is a pure retention number and is labelled as one.
  At t4 it holds `{bear, fire hydrant, stop sign}` and inherits the two
  new-class supply limits above.
* `U_Recall` — `freeze_prob_model=True` keeps PROB's probabilistic-objectness
  head fixed during fine-tuning, which is PROB's own procedure. Unknown recall
  can therefore move only through the classification head, and a small change
  is expected. Reported, not headlined.

## 8. Endpoints declared in advance

There is **no GO/NO-GO gate** — this is an exploratory benchmark, and inventing
a pass/fail threshold for a first end-to-end run would be theatre. What is
frozen is the *contrast* and the *metric*, so that neither can be chosen after
the numbers arrive:

* **primary**: `proposed` vs `admissibility` at **t4** on `known_mAP50`;
* **long-tail**: `mAP50_tail` at every task, with the band's membership named;
* **acquisition** (detector-free, so available even if a trajectory fails):
  `acquired_classes`;
* **gate ablation**: `proposed` vs `coreset`;
* **reference**: `random`.

## 9. Stopping rules, fixed in advance

An arm may be abandoned mid-benchmark for these reasons and no others:

1. an implementation failure (a crash, a corrupt artefact, a fail-closed guard);
2. a degenerate selector — it opens fewer than 20 images, or spends less than
   half its answer budget;
3. a catastrophic detector failure — `known_mAP50` below 5, i.e. training
   diverged.

**A method merely losing is not a reason to stop it or to omit it.** Every arm
that ran is reported, including in the supervisor material.

## 10. Compute, and the reduction ladder

Priced by `tools/plan_full_owod_benchmark.py` from the project's own measured
basis (`data/reference/gpu_cost_basis.json`) plus a deliberately conservative
DINOv2 crop rate, **before** any training. Measured estimate at the top rung
(epochs 5, 1 200 candidate images): **41.6 – 49.6 min** per arm-task, **8.82 h**
for four arms and **11.30 h** for five, against a 10-hour session ceiling.

**Decision, taken before any training ran:** keep the training schedule at 5
epochs and run the first **four** arms of the pre-declared order in session 1;
`coreset` is the first thing session 2 runs, before any replication seed.
Rationale: under-training is the failure mode that produced
`new_class_AP50 ~ 0` in Method V3, and a second session is recoverable where a
weakened schedule is not.

The ladder below is fixed now, for the case where even that does not hold:

1. epochs `5 -> 3 -> 2`, applied uniformly to every arm and every seed;
2. candidate images per task `2000 -> 1200 -> 800`, likewise.

**Not on the ladder**: dropping a task, dropping or choosing a seed after seeing
results, choosing arms by their numbers, or shortening the chain.

A session that runs out of runtime completes a **prefix of the pre-declared arm
order** and resumes the rest; a stopped arm is never reported as complete. The
order puts the primary contrast first on purpose, so a short session still
yields `random`, `admissibility` and `proposed`.

## 11. Resume and robustness

Per-task `state.json` plus a stored configuration fingerprint that **refuses**
a workspace written under a different configuration rather than blending it.
One workspace per `(arm, seed)`; checkpoints pruned to two per arm. Semantic
features cached per task and keyed on a fingerprint of the exact rows they
describe, so a different population cannot reuse another's geometry. Atomic
JSON writes. A Colab disconnect costs the task in flight, not the session.

## 12. Phase 2, priced but not run

* **supervision-matched sensitivity run for the two finalists** — the same
  chain with the budget counted in *supervised* boxes rather than answers, to
  price the residual of section 4.
* **genuine iterative acquisition** — detector rescoring after each 100 answers
  needs 6 predicts and 6 trains per task instead of 1, about 6x the cost. Out of
  reach this week for four arms; it is the one-shot-versus-`6x100` question and
  it is not answered here. The `rounds_per_task` knob in this repository
  recomputes the *score*, not the detector, and is therefore set to **1** rather
  than being presented as iterative active learning.
* **supervision- and step-matched design** — equalising gradient steps as well
  as boxes.
* **nondeterminism floor** — one acquisition retrained 3–4 times, which the
  Method V3 audit recommended and which nothing in this benchmark provides.
* **controlled long-tail undersampling**, then LVIS. Not started.

## 13. The frozen values

Single machine-readable source of truth. `owl.active_selection.benchmark`
declares the same values in code and `check_protocol()` compares them **as
values**, field by field. Prose above is documentation; this block is the
contract.

```json protocol
{
  "n_tasks": 4,
  "answer_budget_per_task": 3000,
  "candidate_images_per_task": 2000,
  "proposals_per_image": 50,
  "rounds_per_task": 1,
  "replay_arm": "uniform",
  "replay_objects": 400,
  "labelling_policy": "full_image",
  "supervision_mode": "ft",
  "epochs": 5,
  "learning_rate": 0.0002,
  "batch_size": 2,
  "eval_max_per_class": 150,
  "eval_remainder_ratio": 2,
  "seeds": [0, 1, 2],
  "nms_iou": 0.6,
  "admissible_share": 0.3,
  "arms": ["random", "admissibility", "proposed", "entropy", "coreset"],
  "endpoints": {
    "primary_contrast": ["proposed", "admissibility"],
    "primary_task": "t4",
    "primary_metric": "known_mAP50",
    "longtail_metric": "mAP50_tail",
    "acquisition_metric": "acquired_classes",
    "ablation_contrast": ["proposed", "coreset"],
    "reference_arm": "random"
  }
}
```

## 14. Pins

* `owod-active` commit: recorded in `manifest.json` per session.
* PROB: `https://github.com/gubiczam/PROB.git` @
  `4c66be1a52cad9360e09c729e9134aba8fe0b531`, branch `feat/daowod-bridge-v2`.
  Verified reachable by `owl.bridge.verify_remote_commit` in a preflight before
  any long setup. **Not changed** to make a clone convenient; the detector
  implementation is part of the experiment.
* Anchor checkpoint: `checkpoints/SOWODB/t1.pth`, sha256 recorded per session.
* Frozen semantic reference: `ref_t1_dinov2_vitb14_cap1000_v1.npz`.
