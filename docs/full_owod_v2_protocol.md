# Full OWOD Chain V2 — protocol

**Frozen 2026-09-10, before the first V2 trajectory runs.** Notebook:
`notebooks/OWOD_FULL_CHAIN_V2.ipynb`. Implementation:
`owl/active_selection/research_score.py`, `owl/supervision.py`,
`owl/replay.py`.

This document exists so that nothing below can be chosen after a number is
seen. Everything it declares is also declared in code, and
`owl.active_selection.benchmark.check_protocol` compares the two as *values* —
a Method V3 overnight session was lost to an assertion that matched a rendered
float against an English sentence, so prose is documentation and the tagged
block at the end is the contract.

**What V2 is.** V1 (`docs/full_owod_active_benchmark_v1_protocol_2026-09-03.md`)
and the ten-task chain that followed it ran arms that are *not* the research
plan's method: three static rankings, two farthest-first traversals in DINOv2
space, and a cluster-quota allocator. The plan's own equation existed only in
`owl/scoring.py`, which runs on the frozen CPU pool and never trains a
detector. V2 puts the equation on the GPU path and makes the 2026-08-25
consultation's open questions configurable axes rather than fixed choices.

**What V2 is not.** It is not a re-run of V1, it may not be tabulated beside a
V1 number, and it does not replace any committed result. V1's numbers stand as
development evidence. No hyper-parameter below was selected because it won on
the seed-0 results already on disk; where a value was needed, it is either
lifted unchanged from a pre-2026-09 freeze or derived from the data's own
geometry, and each is labelled with which.

---

## 1. The score

```
s(x) = U(x) + λ·D(x) + γ·w(ĉ(x))·coh(x)
```

Equation (1) of the research plan, unchanged in form.

| term | definition | provenance |
|---|---|---|
| `U(x)` | normalised Shannon entropy of PROB's 81-column class posterior, rank-normalised over the round's eligible candidates | plan; the consultation explicitly kept it |
| `D(x)` | see §2 | **consultation redesign** |
| `w(ĉ(x))` | `log((1 + n_cand) / (1 + n_ref))` on the candidate's own cluster, positive part, rank-normalised, forced to 0 on a noise point | plan (`w(c) ∝ 1/n_c`), read at cluster granularity as the consultation asked |
| `coh(x)` | see §3 | **consultation redesign** |
| `λ` | **0.2** | `owl.scoring.ScoreConfig`, fixed 2026-08 before any endpoint was read. Never swept. Not re-chosen here. |
| `γ` | **0.5** | same |

Selection happens **inside the admissibility gate** `G` — the top 30 % of the
deduplicated pool by `A(x) = objectness · √area` — which is upstream of the
score and unchanged from V1. Reason on the record: the plan's terms know
*which* region to prefer but not whether it is an object at all, and `A` is
measured to be the strongest single signal for that.

`U`, `D_labeled` and `w` are rank-normalised once per round. `D_batch` is the
one term used raw, because it changes at every pick: rank-normalising it would
make a candidate's value depend on how many rivals remain rather than on how far
it is from what was just bought, and it would cost a sort per pick. It is
already a bounded cosine distance on `[0, 1]`.

## 2. `D(x)` — three explicit modes

`diversity_mode`:

| mode | definition |
|---|---|
| `labeled_novelty` | cosine distance to the nearest row of the **labelled reference** — the balanced task-1 reference plus everything this trajectory has bought. Grows every round and every task. |
| `batch_diversity` | cosine distance to the nearest candidate **already taken in this round**, initialised to 1.0 and lowered after every pick. |
| `combined` | `0.5·(D_labeled + D_batch)` |
| `none` | `D ≡ 0`, the ablation |

The fixed task-1 anchor the consultation objected to is gone: at t3 the distance
is to what has actually been taught, not to a frozen t1 export.

`batch_diversity` is greedy farthest-first. It is **not** k-means++ — no `D²`
sampling, no randomness, no centroid — and is named for what it does. The equal
weighting in `combined` is an *implementation decision*: it introduces no
parameter that could later be tuned. `D_labeled`, `D_batch` and the final `D`
are logged for every taken candidate in `candidate_log.csv`.

## 3. `coh(x)` — the binary gate, with the continuous form kept

`coherence_mode`:

| mode | definition |
|---|---|
| `dbscan_binary` | `coh ∈ {0, 1}`: 0 for a DBSCAN noise point, 1 for a candidate in a cluster. Core **and** border pass — a border point of a small real cluster is the rare-but-real case the gate must not throw away — and the log records which it was. |
| `continuous` | the plan's original wording: inverse of the k-th nearest neighbour distance, scaled by the population median. |
| `none` | `coh ≡ 1`. **Not** the same as `γ = 0`: the rarity weight still acts, which is what isolates the *gate* from the *weight*. |

**The continuous form is retained deliberately.** The repository holds a
measured negative result for a DBSCAN gate in PROB's own decoder space
(`docs/konzultacio_2026-08-25_lefedettseg.md` §2: at eps 0.15 the gate called
92 % of real unknown objects noise against 60 % of background, because the pool
is 81 % background and background regions are near-duplicates of each other).
V2 gates in **frozen DINOv2 space on the admissible subset**, which is a
different population in a different space, and only the ablation says whether
that changes the answer. A negative result here is a result, not a failure.

**DBSCAN parameters are not tuned.**

* `min_samples` = `max(5, ⌈|G| / (B / c̄)⌉)`, i.e.
  `owl.active_selection.allocation.min_cluster_size`, already frozen on
  2026-09-06 and derived from the **answer budget** `B` and the pool's mean
  image cost `c̄`. Rationale on the record: `B / c̄` is how many images the
  budget can open, so the quotient is "candidates per affordable image", and a
  cluster smaller than that cannot receive a quota worth one image.
* `eps` = the **median** of each candidate's distance to its `min_samples`-th
  nearest neighbour, computed on the round's own eligible candidates and
  recomputed every round. The k-distance graph is the standard way to read a
  density radius; the standard advice — "find the elbow" — is a judgement made
  on the data the result is then read from, and a fixed quantile is the same
  construction with the judgement removed. The quantile is fixed at 0.50 *a
  priori*, and the reason is a statement about the consequence: at
  `eps = median(k-dist)` roughly half the candidates have `min_samples`
  neighbours inside `eps` and become core points, so the gate rejects
  approximately the sparse half of the population.
* PCA to **32** dimensions before clustering, the value
  `owl.clustering.density_coherence` already used, because a DBSCAN radius
  means nothing at 768 dimensions.

Logged per round: cluster count, noise count, noise rate, core count, border
count, accepted, rejected, mean/smallest/largest cluster size, `eps`,
`min_samples`. Logged per taken candidate: cluster id, `core|border|noise`,
`coh`.

## 4. One clustering for `D` and `w`

The partition that produces `coh` is the partition that produces `w`. The
consultation asked for exactly this ("if that is in place then `D` and rarity
both fall out of the clustering").

The partition is fitted in the PCA-reduced space, where a radius means
something. Cluster medoids and the reference-row-to-cluster assignment are then
computed in the **unreduced** space, because that is where the labelled
reference export lives; the partition is taken as given and only the geometry is
redone. `owl.active_selection.research_score.cluster_rarity` refuses a reference
of the wrong width rather than silently comparing two spaces.

**Rarity is never a true class frequency at selection time.** `n_cand` and
`n_ref` are occupancies of an embedding partition. `tests/test_research_score.py`
asserts the term's executable body cannot reach an oracle, a class name or the
frequency groups, and that a research arm runs on a pool whose oracle has been
removed.

## 5. Known-contamination diagnostic

`research_score.known_contamination` clusters the labelled reference **together
with** the candidates and reports: known contamination rate (known rows sitting
in a cluster the candidates also occupy), candidate-only / known-only / mixed
cluster counts, mean cluster purity from the known side, known and candidate
noise rates, and candidate cluster sizes.

**It is a diagnostic and nothing else.** It never enters a score, and a test
asserts the selection path does not call it.

## 6. Annotation policy — a real per-box protocol

`annotation_policy`, implemented in `owl/supervision.py` by writing the
annotation rather than by naming a PROB mode. The old mapping was
`supervision = "train" if policy == "box_only" else "ft"`, which has two
settings for three policies, so `full_image` and `known_plus_selected` were the
same run.

| policy | previously-known objects | the selected object | every other annotated object | cost |
|---|---|---|---|---|
| `selected_box_only` | absent → **background** | supervised | absent → **background** | 1 answer per selected object |
| `full_image` | supervised, free | supervised | supervised if declared; dropped by PROB if not | 1 answer per annotated object on the image |
| `known_plus_selected_ignore_rest` | supervised, **free** | supervised | **ignored** | 1 answer per selected object |

An object of a class no task has declared yet is **banked**, not lost: it is
recorded and rejoins training at the task where its class becomes declarable, at
no further annotation cost (`reuse_deferred_labels`).

### Ignore, stated honestly

Read from the pinned PROB source (`4c66be1`,
`datasets/torchvision_datasets/open_world.py`): `load_instances` parses every
`<object>` and reads only `name` and `bndbox`; `__getitem__` builds `labels`,
`boxes`, `area` and `iscrowd = zeros`; the loss is Deformable-DETR set
prediction. **There is no ignore channel.** `difficult` *is* honoured, but only
by the evaluator (`open_world_eval.py` excludes difficult boxes from both the
positive count and the FP count), never by the training loop. A box left out of
the XML is therefore not ignored — its region is taught as background.

`ignore_mechanism`:

| mechanism | what it does |
|---|---|
| `drop` | leave the object out. **This is not ignore** — the region becomes background. It is what the pipeline did before V2, kept as the named control so the difference the mechanism makes is measurable. |
| `pixel_suppression` | fill the ignored box with the dataset mean colour in a *derived* JPEG. The region then genuinely holds no object, so "background" is the correct target rather than a false one. **Default** for `known_plus_selected_ignore_rest`. |
| `prob_ignore_flag` | the clean route, and a change to the **PROB fork**, not to this repository. Raises with the patch written out. |

`pixel_suppression` is an **implementation decision**, not something either
source document asks for. It trades a false negative for a small distribution
shift, and the shift is reported (`pixels_suppressed_share`,
`images_pixel_suppressed`). The cleaner mechanism is a real ignore mask in
`SetCriterion`; it is specified in the `NotImplementedError` and is the next
step on this axis.

The load-bearing assertion, and the reason this is testable without a GPU:
`owl.supervision.prob_training_target` transcribes PROB's own target
construction, and `owl.supervision.false_background` names the real annotated
objects that end up as background targets. Under
`known_plus_selected_ignore_rest` + `pixel_suppression` that list must be
**empty**; under `drop` it is exactly the ignore set. Both are asserted in
`tests/test_supervision_policy.py`, and so is the fact that today's default
`full_image` still teaches a future-task class as background.

Derived annotations are written as **aliases** (`'8' + id[1:]`), digits-only and
of the source's width, because `OWDetection.convert_image_id` does
`int('2021' + id)` and the evaluator's reverse conversion asserts a 12- or
6-digit remainder. The originals are read and never written — the same image is
also evaluation data. `owl.exemplars` owns the `9` prefix; sharing one would let
either clear the other's annotations mid-task.

## 7. Cost accounting

The primary budget unit is the **oracle answer**, and it is equal across arms by
construction up to the at-most-one-image underspend the ledger allows. What the
answers *became* is reported separately and is **not** equalised, because the
difference is a consequence of the selection policy and is part of the result:

`answers_spent`, `answers_unspent`, `images_opened`, `answers_per_image`,
`objects_labelled`, `objects_supervised`, `objects_ignored`, `objects_banked`,
`known_boxes_reused`, `new_boxes_supervised`, `boxes_labelled`,
`boxes_supervised`, `boxes_banked`, `boxes_trained_on` (+ head/medium/tail),
`images_trainable`, `images_barren`, `images_from_earlier_tasks`,
`training_images`, `training_iterations`, `selections_matching_no_object`,
`pixels_suppressed_share`.

## 8. Replay

`replay_mode` → `m_c ∝ n_c^α`, `Σ m_c = M`, with `owl.replay.allocate` as the
single implementation: capacity-aware, largest-remainder rounding, so the sum is
exact and the tie-breaking is deterministic and documented.

| mode | α | |
|---|---:|---|
| `none` | — | no rehearsal, the lower bound |
| `uniform` | 0 | today's standard; the arm every committed chain holds fixed |
| `proportional` | 1 | head-favouring; the plan's predicted failure, `minimum=1` keeps the tail off zero |
| `tail_aware` | −0.5 | the plan's `α < 0` branch, contribution B's proposal |

`M = 400` exemplar **objects**, unchanged. Replay Protocol V3 makes the budget
exact in the unit the question is about: the memory is a set of exemplar boxes,
each materialised through an alias annotation holding only itself, so
`Σ m_c = |E_k| = delivered = M`. Versions 1 and 2 stored *images* chosen to
cover an object allocation and delivered 464 objects for `head_favouring`
against 1 240 for `tail_favouring` at the same `M = 400` — a 2.67× spread
produced by the allocation rule rather than by design, and the reason the
research plan's remark that "one stored image may contain several classes" is a
correctness constraint here and not a footnote. The object → image mapping is
`owl.exemplars.write_aliases`.

`replay_refresh`: `fixed` keeps incumbents wherever they still serve the new
allocation; `per_task` re-derives the allocation from the class distribution
known **at this task**. `per_task` is the consultation's "a different memory in
every task" and is the main new possibility on this axis. Both re-satisfy the
object budget from scratch, so neither can let the memory drift in size.

## 9. Iterative acquisition

`acquisition_batch_size` in **answers**. `rounds = ⌈B / batch⌉`, and round *r*
may spend `B·r // rounds` in total, so an earlier round's unplaced remainder
carries forward — without the carry the campaign quietly shrinks and a one-shot
arm is no longer comparable to an iterative one.

After each mini-round: the bought candidates join the labelled reference, and
the clustering, `eps`, the rarity, the quota and `D_labeled` are all recomputed
against it. `D_batch` resets, because what the previous round bought is now
covered by `D_labeled`.

**The detector is not retrained between rounds.** Acquisition-score
recomputation and detector retraining are separate knobs on purpose: the
consultation asked for the former and said nothing about the latter, and
conflating them prices a cheap experiment as an expensive one. Detector update
stays once per task.

## 10. The chain

`t1 → acquire → annotate → replay → train → evaluate → t2 → … → t10`, one new
class per task, each task fine-tuning the previous task's own checkpoint for its
own arm.

**This is the repository's own controlled chain, not published S-OWODB.** It
declares one class per task, not S-OWODB's 19/21/20/20 split, and no number from
it may be compared against a published S-OWODB number. Running the published
S-OWODB and M-OWODB protocols from the research plan remains available and is
**not** done here; it is recorded as open in §14.

Evaluation is on the shared reduced split (`EVAL_MAX_PER_CLASS = 150`), so
previous-class mAP is a sample estimate — comparable between arms, not against
published numbers.

## 11. Metrics

Per task, per arm, per seed.

* **detector** — `known_mAP50`, `prev_mAP50`, `new_mAP50`, `new_class_AP50`,
  `U_Recall50`, `forgetting`, `drop_from_anchor`, `exchange_rate`, **`WI08`**,
  **`A_OSE`**.
* **long-tail** — `mAP50_head/medium/tail`, `U_Recall_head/medium/tail`,
  per-class AP50, per-class forgetting.
* **acquisition** — `acquired_classes`, `acquired_objects`,
  `acquired_new_class`, `acquired_known_now`, `acquired_becomes_known_t*`,
  `acquired_stays_unknown`, `acquired_head/medium/tail_objects`,
  `positions_redundant`, and the DBSCAN gate's rejection counts.
* **cost** — §7.
* **chain** — cumulative annotation cost, performance vs annotation cost,
  stability/plasticity trajectory.

`WI08` and `A_OSE` are new to the rows and old to the data: PROB's evaluator has
always returned them and `daowod_prob_bridge` has always normalised `AOSA` to
`A_OSE`, so every committed `metrics.json` holds them. They were simply never
tabulated. A row written before 2026-09-10 reports `None` for them, which is
correct.

## 12. Arms

| arm | role |
|---|---|
| `random` | the reference every active method must beat |
| `entropy` | uncertainty only, `U` — the standard baseline |
| `distribution_aware_iterative_v1` | the **legacy / pre-redesign** method on this path, so the redesign's effect is visible |
| `research_v2_plan` | the plan's equation as literally written: `labeled_novelty` + `continuous` |
| **`research_v2`** | the proposed method: `combined` + `dbscan_binary` + `cluster_rarity` |
| `research_v2_no_gate` | gate ablation, one variable from `research_v2` |
| `research_v2_labeled_only` | `D` = labelled novelty only |
| `research_v2_batch_only` | `D` = intra-batch diversity only |

Replay control: `uniform` for every arm. Distribution-aware replay: `tail_aware`
as a separate cell, held to the primary acquisition arm.

The `research_*` arms are reported as **development-seed-informed**, and that
label is deliberately conservative rather than exact: no V1 endpoint informed a
single term of them, and this document pre-registers them — but they were added
after V1 seed-0 numbers existed, and the convention here is that anything added
after results exist says so.

## 13. Two phases, and the rule that moves an arm between them

A full factorial over score × diversity × coherence × annotation × replay ×
refresh × batch size is not affordable and is not run.

**Phase A — protocol validation.** `t1 → t5`, seed 0. Four arms:

```
random                            + uniform replay
entropy                           + uniform replay
research_v2                       + uniform replay
research_v2                       + tail_aware replay, per_task refresh
```

Purpose: verify that the machinery does what this document says — the per-box
supervision reaches the detector, the gate fires at a rate strictly between 0
and 1, the labelled pool grows across tasks, the rounds recompute, the replay
allocation sums to `M`, and the cost columns reconcile. **Phase A is not an
endpoint comparison.** Its outputs are the diagnostics, not a winner.

**Phase B — the full chain.** `t1 → t10`, seeds 0/1/2, arms fixed **in advance**
to `random`, `entropy`, `distribution_aware_iterative_v1`, `research_v2_plan`,
`research_v2`, plus `research_v2` + `tail_aware`.

### The decision rule, written before Phase A runs

Phase B's arm set is the list above. It is **not** chosen by Phase A's AP.
Phase A can only *remove* an arm, and only for a mechanical reason:

1. a trajectory does not complete (crash, OOM, exhausted budget) → the arm is
   reported as INCOMPLETE and does not enter Phase B;
2. an arm's cost ledger fails to reconcile — `answers_spent` differs from the
   sum of charged image costs, or `objects_supervised + objects_ignored +
   objects_banked ≠ objects_labelled` → the arm is a bug, not a result, and is
   fixed before Phase B rather than reported;
3. the coherence gate degenerates — `noise_rate ∈ {0, 1}` at every task → the
   `dbscan_binary` mode is reported as inapplicable in this space, exactly as
   the 2026-08-25 gate was on PROB's decoder space, and `research_v2` runs with
   `coherence_mode = "continuous"` as its `research_v2_plan` variant already
   does. This is a **pre-registered fallback**, not a tuning step.

Nothing else may drop an arm. In particular: a low AP, a low U-Recall, or a
comparison that looks unfavourable is a result to report, not grounds for
removal.

## 14. Seeds, endpoints, failure criteria

**Seeds.** `[0, 1, 2]`. Seed 0 is development and debug. **The final comparison
reports three seeds**, as mean ± std or as the three individual values. A
seed-0-only difference is a direction, not an effect; the repository has
measured that this pipeline's paired three-seed range is 0.28 on the contrast of
interest, and that range already contains training noise.

**Primary endpoint.** `U_Recall_tail` as a function of `oracle_cost_so_far` —
the research plan's own headline ("the expected tendency is that
distribution-aware selection reaches the same tail level from substantially
fewer annotations"). Primary contrast: `research_v2` vs `entropy`, at t10, three
seeds.

**Secondary endpoints.** `mAP50_tail`; `known_mAP50` (stability);
`new_class_AP50` (plasticity); `forgetting`; `exchange_rate`; `WI08`; `A_OSE`;
`acquired_tail_objects` per answer (detector-free, so available even if a
trajectory fails).

**Failure criteria — declared here so a negative result is a result.**

* The gate is inapplicable if `noise_rate` is 0 or 1 at every task (§13 rule 3).
* `D`'s redesign has no effect if `research_v2` and `research_v2_plan` open the
  same images to within 5 % Jaccard at every task, on every seed.
* The rarity term has no effect if `research_v2_no_gate` and `research_v2` agree
  within the three-seed spread on every secondary endpoint.
* The method does not beat the bar if `research_v2` does not exceed `entropy` on
  primary `U_Recall_tail` at t10 on at least 2 of 3 seeds. **This is reported,
  not tuned away.**

**Open, and not blocking V2.**

* Published S-OWODB / M-OWODB task sequences (§10). Infrastructure exists; the
  runs do not.
* LVIS confirmation, named in the plan's evaluation section.
* A real ignore mask in PROB's `SetCriterion` (§6), which would replace
  `pixel_suppression` with the exact mechanism.
* Per-class *forgetting* is derivable from the committed per-class AP50 vector
  but is not yet a summariser column.

---

## 15. What it costs, and where the number comes from

Not a guess. `tools/plan_full_owod_benchmark.py` models training at 45.5 min per
arm-task on Benchmark V1's 837-image split with its own 11.1 min evaluation
estimate, so the fixed part is **34.4 min**. The t10 session measured evaluation
at **36.7 min on 2,817 images** with `detections=True` — a second forward pass,
which the plan's headline endpoint requires — giving **0.0130 min per test
image**. The chain length sets the split, and the split sets the rest.

| | split | min / arm-task | arm-tasks | GPU-hours |
|---|---:|---:|---:|---:|
| Phase A — `t1→t5`, 3 arms, seed 0 | 918 | 46.4 | 12 | **≈ 9** |
| Phase B — one arm, one seed, `t1→t10` | 2 817 | 71.1 | 9 | **≈ 11** |
| Phase B — 5 arms × 3 seeds | 2 817 | 71.1 | 135 | **≈ 160** |

**160 GPU-hours is not a free-tier session, and pretending otherwise is how a
benchmark gets silently truncated.** Three consequences, decided here rather
than when the clock runs out:

1. the reduction ladder is V1's and is unchanged — epochs 5 → 3 → 2, then
   candidate images 2000 → 1200 → 800, applied uniformly to every arm and seed.
   At epochs 3 / 1200 images Phase B falls to roughly 100 GPU-hours;
2. the launcher stops **between** tasks on a time budget and resumes, so Phase B
   is a sequence of sessions, not one run. `ORDER` fixes which arms a short
   session completes, so the surviving prefix cannot be chosen after the fact;
3. if only one seed is affordable, the result is **exploratory** and says so.
   Dropping seeds 1 and 2 costs the primary endpoint its error bar, and §14's
   "a seed-0 difference is a direction, not an effect" is then the whole finding.

Dropping a task, dropping a seed *after seeing results*, or choosing arms by
their numbers is never on the ladder.

**Where the annotation budget came from.** The frozen-pool ledger
(`tools/plan_full_owod_benchmark.py`) opens 436 images for `research_v2` and 517
for `random` at 3 000 answers under `full_image`, whose measured density is 9.56
annotated objects per candidate image. Under
`known_plus_selected_ignore_rest` an image costs **1** answer, so **300 answers
opens ~300 images** — the same order as the full-image arms, which keeps the
training cost per task comparable while the oracle cost differs by ~10×. That
difference is the annotation-efficiency result, and it is reported as an
outcome, not equalised away.

---

## Frozen values

<!-- FROZEN-V2-BEGIN -->
```json
{
  "n_tasks": 10,
  "seeds": [0, 1, 2],
  "answer_budget_per_task": 3000,
  "candidate_images_per_task": 2000,
  "proposals_per_image": 50,
  "lambda_diversity": 0.2,
  "gamma_rarity": 0.5,
  "eps_quantile": 0.5,
  "pca_dimensions": 32,
  "diversity_mode": "combined",
  "coherence_mode": "dbscan_binary",
  "rarity_mode": "cluster_rarity",
  "annotation_policy": "known_plus_selected_ignore_rest",
  "ignore_mechanism": "pixel_suppression",
  "replay_mode": "uniform",
  "replay_refresh": "fixed",
  "replay_objects": 400,
  "acquisition_batch_size": 100,
  "epochs": 5,
  "learning_rate": 0.0002,
  "batch_size": 2,
  "eval_max_per_class": 150,
  "phase_a_tasks": 5,
  "phase_a_arms": ["random", "entropy", "research_v2"],
  "phase_b_arms": ["random", "entropy", "distribution_aware_iterative_v1",
                   "research_v2_plan", "research_v2"],
  "primary_metric": "U_Recall_tail",
  "primary_contrast": ["research_v2", "entropy"],
  "primary_task": "t10"
}
```
<!-- FROZEN-V2-END -->
