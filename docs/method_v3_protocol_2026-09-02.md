# Method V3 — Selection → Learning Transfer

**Exploratory / prospective. Frozen 2026-09-02, before any downstream detector
endpoint was run.**

This document did not exist before today. It is written from the user's
specification of 2026-09-02 and is frozen before the training launcher may run.

---

## 0. What this experiment is, and what it is not

Method V2 Stage 2 returned, under its own pre-registered thresholds:

```
D_NO_GO      unknown-vs-known AUC      0.6411   (threshold 0.65)
R_NO_GO
C_GO         unknown-vs-background AUC 0.6101   (threshold 0.60)

METHOD_V2_ALLOWED_LADDER = U
```

**That verdict is not reopened here.** D is not rescued because 0.6411 is close
to 0.65. No threshold is changed. No R definition is re-chosen. The Method V2
ladder remains exactly `U`.

Method V3 asks a **new** question:

> Does the only semantic component that passed its frozen Stage-2 gate —
> consistency `C` — produce a real downstream active-learning benefit when
> detector learning is measured under an equal annotation budget?

A negative answer here does not weaken Stage 2's `C_GO`; a positive answer does
not overturn `D_NO_GO` or `R_NO_GO`. The two experiments have different
endpoints.

---

## 1. The four arms — fixed now

| arm | score | source |
|---|---|---|
| `random` | seeded uniform sample without replacement | `owl.selection.select(config.random=True)` |
| `A` | `A(x) = objectness(x) · sqrt(area(x))` | `owl.scoring.admissibility` |
| `U` | normalised Shannon entropy of the class posterior | `owl.scoring.uncertainty(method="entropy")` |
| `A*C` | `A(x) · C(x)`, literal multiplication | `owl.method_v2_stage2.score_c` |

with

```
C(x) = min( cos(z_1.20, z_1.10),  cos(z_1.20, z_1.30) )
```

read **verbatim** from the completed frozen Stage-2 view export
`dinov2_vitb14_stage2_views_v1.npz`. No DINOv2 forward pass is run by Method V3.

No exponent. No `C` threshold. No rescaling. No `A + C`. No normalisation chosen
after an endpoint. No parameter search. `score_c` is the same frozen function
Stage 2 used, called unchanged.

`A` and `U` are used **raw** for ranking. A rank-normalisation of a single term
is order-preserving, so `rank_normalise(A)` and `A` select identically; the raw
form is used so that `A*C` is a literal product of the same `A`.

---

## 2. Seeds and budget — fixed now

```
seeds                 = [0, 1, 2]
acquisition rounds    = 6
budget per round      = 100 regions
total per trajectory  = 600 regions
trajectories          = 4 arms x 3 seeds = 12
```

"Acquisition round" is this repository's own meaning of the term
(`owl/selection.py`, consultation point 7): the budget is spent in six
recomputations of the score inside one incremental task, not in six separate
tasks. All twelve trajectories are attempted. None is terminated early.

**Consequence, stated in advance (1) — the rounds are prefixes.** `A`, `U` and
`A*C` are static scores — they carry no `D`, no rarity, no batch-diversity term —
so re-scoring between rounds cannot move them. For those three arms the six
rounds are therefore exactly the nested prefixes of one ranking: the top 100,
top 200, … top 600. That is not a defect here, it is what makes the per-budget
curve in §7 an exact prefix curve rather than six separate campaigns. `random`
draws without replacement, which gives nested prefixes as well.

**Consequence, stated in advance (2) — what the three seeds vary.** For the same
reason, `A`, `U` and `A*C` select the **identical 600 regions at every seed**.
The seed still changes the exemplar draw inside the fixed `uniform` allocation.
(It does **not** change PROB's own `--seed`: `owl.bridge.Bridge.seed` defaults to
0 and the launcher never overrides it, so PROB was seeded 0 in all twelve
trajectories. Measured in
[`docs/method_v3_posthoc_audit_2026-09-03.md`](method_v3_posthoc_audit_2026-09-03.md)
§2.1, after the run; this sentence originally claimed both.) So for those arms
the three paired
differences measure **training and replay-draw noise on two fixed selections**,
not selection variance. This is stated here, printed by the summariser, and it is
not a flaw — a paired design that holds the selection fixed is the cleaner way to
ask whether one fixed selection trains better than another — but it must never be
reported as if three independent selections had been drawn. Only `random` varies
its selection with the seed.

**Consequence, stated in advance (3) — `C` is used with its own sign.**
`score_C = A · C` is the frozen Stage-2 function, applied unchanged. `C` is a
cosine and could in principle be negative, which would invert the product for
those rows. No clamp, no rescaling and no absolute value is introduced: that
would be a new definition. The runner prints `C`'s mean and its full range before
selecting, so a pathological range is visible in the log rather than silently
absorbed.

---

## 3. What is held identical across the twelve trajectories

* **starting checkpoint** — PROB's published S-OWODB `t1.pth`, from Drive
* **candidate population** — the fixed population of §4, byte-identical
* **training schedule** — `epochs=5`, `learning_rate=2e-4`, `batch_size=2`,
  `supervision_mode=ft`, `freeze_prob_model=True`
* **annotation protocol** — `known_plus_selected` (§6)
* **replay policy** — `uniform`, 400 exemplar **objects**, `alpha=0.0` (§5)
* **evaluation** — one shared test split, built once, identical for every
  trajectory and for the anchor
* **random-seed handling** — one seed per trajectory, driving selection, the
  exemplar draw and PROB's own `--seed`

Only the acquisition ranking changes.

---

## 4. The candidate population, and why it is what it is

```
population = P2  ∩  { images whose benchmark annotation is committed to this repository }

P2                                  15,518 proposals on 1,599 images
committed candidate annotations     data/staging/owdetr_pool_annotations.tar.gz
                                    + data/reference/per_image_class_counts.json
                                    (28,800 images)
=> Method V3 population              8,010 proposals on   839 images
```

Two constraints force this, and both were measured rather than assumed:

1. **`C` exists only on P2.** `tools/export_dinov2_consistency_views.py`
   exported the 1.10x and 1.30x views for P2 only — 15,518 rows. Outside P2
   there is no `C` value, and Method V3 may not compute one (that would be a new
   DINOv2 pass and a new population). Since all four arms must share one
   candidate population, that population is inside P2.
2. **Training needs the annotation on disk.** Of P2's 1,599 images, 839 have a
   committed benchmark annotation XML. The remaining 760 cannot be trained on at
   all, so they cannot be in a population the detector learns from.

The population is verified fail-closed before anything expensive runs: 8,010
rows, 839 images, and P2 itself re-verified at 15,518 rows / 0.767 background by
`owl.method_v2_stage2.verify_p2`.

**Frozen candidate scores.** The proposals, their 256-d decoder embeddings, their
posteriors and their objectness all come from the committed `t1.pth` pass, and
the `C` values from the frozen DINOv2 export. The acquisition function is
therefore **not** re-estimated as the detector adapts. That is a deliberate,
pre-declared simplification — it is what makes the population identical across
arms and it removes the per-task detector pass entirely — and it is a real
difference from `owl.runner.run_chain`, which re-predicts per task.

---

## 5. Replay — one fixed setting, chosen before running

```
replay_arm = "uniform"      owl.replay.ARMS["uniform"] = {"total": 400, "alpha": 0.0}
```

This is the project's established **matched control**: the completed Replay
Protocol V3 experiment (`notebooks/owod_active.ipynb`) compares
`random__uniform` against `random__tail_favouring` on top of the `random__none`
baseline, and `uniform` is the neutral allocation in that comparison.

Method V3 is an experiment about **component A / active selection**. Replay is
therefore held fixed at `uniform` for all four arms and all three seeds. Replay
is not tuned tonight, and no arm gets a different replay scheme. The memory is
`replay_protocol_version = 3` semantics: 400 exemplar **objects**, each
materialised as an alias annotation holding only itself, drawn from the canonical
old-data index `data/reference/t1_replay_class_counts.json` over the 19 Task-1
classes, minus whatever the trajectory just bought.

Measured allocation for `uniform`/400 at this task: 21 objects per class (22 for
`giraffe`), 400 objects on 396 distinct source images.

---

## 6. Annotation semantics — reused unchanged

```
labelling_policy = "known_plus_selected"        owl.labelling.POLICIES
```

* **region cost** — the oracle is charged for the 600 *regions* it is asked
  about. Objects of already-known classes on an opened image are free: the
  detector can already produce them.
* **opened-image semantics** — a chosen region opens its whole image.
* **`known_plus_selected`** — known-class objects labelled, chosen unknowns
  labelled, remaining unknowns marked ignore rather than background.
* **ignored boxes**, **region/object matching**, **oracle matching** (IoU 0.5),
  **per-image NMS** (IoU 0.60, inside P2's construction), **half-label handling**
  — all unchanged, all from the existing modules.
* **distinct-object accounting** — every "objects" number goes through
  `owl.discovery`, the single counter. Proposals are never reported as objects.

**One honest mechanical note about the GPU path**, printed by the notebook
before training: `owl.labelling` prices the annotation, but PROB itself reads
the real annotation XML for every image it is handed and keeps the boxes whose
category falls in `range(0, prev + current)`. So the supervision a trajectory
receives is determined by **which images it opens** and what declared-class
objects those images contain — not by which single box inside an image was
clicked. Selecting an object of a still-undeclared class yields no gradient at
this task. This is a property of the established protocol, not a change to it,
and it is the channel through which selection can reach the detector at all.

---

## 7. What is measured

One incremental task, `owl.protocol.build_chain(2)`:

```
t1   anchor, PROB's published t1.pth, 19 known classes
t2   declares CLASS_ORDER[19] = "traffic light"   (head band, 11,431 train objects)
```

**Detector metrics** at the final budget of 600, from PROB's own evaluator via
`owl.metrics`:

`known_mAP50`, `prev_mAP50`, `new_mAP50`, `new_class_AP50`, `U_Recall50`,
`forgetting`, `drop_from_anchor`, `mAP50_head`, `mAP50_medium`, `mAP50_tail`,
`mAP50_medium_tail`, and — from the per-box detection artefact —
`U_Recall_head / medium / tail / all` with their object counts.

**Selection / oracle-coverage metrics** at every budget mark
(100, 200, 300, 400, 500, 600), from `owl.discovery`:

distinct unknown objects, distinct objects by band (head / medium / tail),
distinct medium+tail objects, unknown classes, proposals per object,
background share of the selection, images opened, oracle cost.

The detector metric is the **primary** evidence. Selection coverage is
explanatory. An area-under-learning-curve is reported for the selection curve,
where the six marks make it well defined.

**Stated precision limits.**

* The detector endpoint is measured at budget 600 only. A detector evaluation at
  each of the six marks would cost 72 evaluations; §9 prices that at far beyond
  the night. The per-budget curve is therefore a *selection* curve, and it is
  labelled as one everywhere it appears.
* At t2 the **tail band holds exactly one known class, `bear`**. `mAP50_tail` is
  therefore a single-class AP and is reported as such. It is never described as
  "tail classes". `mAP50_medium_tail` spans 8 classes: aeroplane, cat, dog,
  train, elephant, zebra, giraffe, bear.
* The shared test split is a reduced split (§8). Previous-class mAP on it is a
  sample estimate and is **not** comparable to published full-test figures. It is
  comparable across the twelve trajectories, which is what this experiment needs.
* Only 56 of the 839 candidate images contain a `traffic light`. `new_class_AP50`
  is therefore driven largely by how many images an arm opens, and is reported as
  a secondary, not as the primary.
* The arms open very different numbers of images for the same 600 regions —
  measured: `A` 590, `A*C` 476, `random` ≈405, `U` 256. At a strict *region*
  budget that is a legitimate difference between the arms, and under
  `known_plus_selected` it is also the main channel by which supervision volume
  differs. Both cost axes are reported (`per_region` and `per_image` in
  `owl.discovery`); the region budget is the primary one, as in A1.

---

## 8. The shared evaluation split

Built once with the established construction
(`owl.evaluation_subset.from_archive`, seed 0), then written once and used by
every trajectory and by the anchor:

```
declared classes      ["traffic light"]
max_per_class         150
remainder_multiplier  3
=> 600 images  (150 required + 450 deterministic remainder)
```

`remainder_multiplier` is 3 rather than the replay experiment's 1 because a
one-class chain would otherwise give a 300-image split, and previous-class mAP
over 19 classes needs the coverage. It is fixed here, before any endpoint, and is
identical for all twelve trajectories and for the anchor. It is not tuned.

The anchor evaluation of `t1.pth` on this split is run **once** and reused by all
twelve trajectories, because it is the same checkpoint on the same split.

---

## 9. Runtime, decided before training

From the project's own measured cost basis, `data/reference/gpu_cost_basis.json`
(T4, 2026-08-26):

```
train    2.053 min fixed + 0.9 s per iteration,  iterations = ceil(images / 2) * epochs
evaluate 0.300 min fixed + 6.462 min per 1,000 test images,  doubled when the
         per-box detection artefact is written
```

Measured inputs (computed on the real population, no detector run):

| arm | images opened @600 | trainable | + replay aliases |
|---|---:|---:|---:|
| `A` | 590 | 343 | 739 |
| `random` | 411 | 275 | 671 |
| `U` | 256 | 210 | 606 |
| `A*C` | not computable without the Drive view export | — | — |

Estimated total for all 12 trajectories at `epochs=5` on a 600-image split:
**≈ 7.2 GPU-hours**, plus the anchor evaluation once (≈ 8 min), the PROB CUDA
kernel build, and the image materialisation. **Target: ≈ 8 hours on a T4.**

**Decision: no training-cost reduction is applied.** The established
`epochs=5, learning_rate=2e-4, batch_size=2` schedule fits the night, so nothing
is reduced and nothing is padded. `tools/plan_method_v3.py` recomputes this
estimate from the same cost basis and prints it before the launcher runs.

For the record, had a reduction been needed the ladder was fixed in advance —
`epochs` 5 → 3 → 2 → 1, applied uniformly to every arm and seed, never by
dropping arms and never by selecting seeds.

Seeds 0, 1, 2 remain the design. A fourth seed is not added: the estimate above
was made before any downstream endpoint existed, and it does not leave a clean
trajectory's worth of headroom.

---

## 10. The primary contrast, fixed before running

```
PRIMARY:  A*C  vs  A   at the final equal annotation budget of 600 regions
```

Primary metric: `mAP50_medium_tail` — the mean per-class AP50 over the medium and
tail classes among the 20 classes known after t2, with the band read from
`data/reference/class_groups.csv` through `owl.protocol.load_groups`.

Secondary contrasts, all reported whatever they show: `U vs A`, `A*C vs U`,
`random vs all`.

---

## 11. `C_DOWNSTREAM_POSITIVE` — the criterion, frozen before execution

The verdict is computed mechanically from the twelve result rows by
`owl.method_v3.evaluate_criterion`. It is

```
C_DOWNSTREAM_POSITIVE
```

if and only if **all three** of the following hold at budget 600:

1. `mean_over_seeds mAP50_medium_tail(A*C)  >  mean_over_seeds mAP50_medium_tail(A)`
2. at least **2 of the 3** paired per-seed differences
   `mAP50_medium_tail(A*C, s) - mAP50_medium_tail(A, s)` are strictly positive
3. `mean_over_seeds known_mAP50(A*C)  >=  mean_over_seeds known_mAP50(A) - 1.0`

and otherwise

```
C_DOWNSTREAM_NOT_SUPPORTED
```

The tolerance in (3) is **1.0 AP50 point**, predeclared here. It exists so that a
medium+tail gain bought by a real collapse of the known classes does not count as
a success; it is not a significance statement. There is no established AP
tolerance elsewhere in this repository, so this number is a declaration, not an
inherited convention.

Which clause failed is printed with the verdict. A failure of (3) alone still
yields `C_DOWNSTREAM_NOT_SUPPORTED`; it is not reinterpreted.

**Numbers appear twice in §11 — in the prose above and in the block of §11.0 —
and the two are kept in step by a test**, which reads every value out of the
block and requires the prose to state the same number in some standard rendering.
The block is authoritative; the prose may not disagree with it.

### 11.0 The criterion, machine-readable

This block is the document's copy of the criterion, and it is the **only** part
of §11 that is checked against the code. `owl.method_v3.check_protocol_criterion`
parses it and compares it **field by field, as values** against
`owl.method_v3.CRITERION`; the notebook calls that function before the training
launcher and refuses to run if the two disagree.

It is here because the first attempt validated the criterion by searching this
document for a rendered phrase, and `f"{1.0:g}"` renders `1.0` as `1` — so a
correct, frozen criterion was reported as a documentation mismatch and the
overnight run stopped before training. A number compared as a number cannot fail
that way. The prose of §11 is for the reader; this block is for the machine.

<!-- METHOD-V3-CRITERION: parsed by owl.method_v3.parse_criterion_block. Keep the
     fence tag `json criterion` exactly; edit only with a deliberate, documented
     change to owl.method_v3.CRITERION. -->

```json criterion
{
  "primary_metric": "mAP50_medium_tail",
  "guard_metric": "known_mAP50",
  "treatment": "A*C",
  "control": "A",
  "budget": 600,
  "minimum_improving_seeds": 2,
  "guard_tolerance": 1.0
}
```

**With three seeds this is a descriptive criterion, not a significance test.**
Individual seed values, the mean, the standard deviation and the three paired
differences are all printed. No p-value is claimed. A bootstrap interval, if
printed, is descriptive only.

---

## 12. What is forbidden tonight

* changing the Method V2 `D` threshold, or rerunning Method V2 at any other
  threshold;
* choosing R1 / R2 / R3 after seeing a result;
* tuning `lambda`, `gamma`, replay, or the training schedule;
* sweeping `C` exponents or `C` thresholds;
* adding or removing an arm after seeing an intermediate result;
* selecting the best seed, or terminating an arm that looks bad;
* using any oracle endpoint to choose a parameter.

The four arms and the criterion above are fixed as of this document.

---

## 13. Pinning

| | |
|---|---|
| OWL repository | `github.com/gubiczam/owod-active`, commit pinned in the notebook |
| PROB repository | `https://github.com/gubiczam/PROB.git`, branch `feat/daowod-bridge-v2` |
| PROB commit | `4c66be1a52cad9360e09c729e9134aba8fe0b531` |
| detector checkpoint | `MyDrive/OWL/checkpoints/SOWODB/t1.pth`, SHA-256 recorded in the run manifest |
| frozen base features | `dinov2_vitb14_method_v2_v1.npz` |
| frozen views | `dinov2_vitb14_stage2_views_v1.npz` |
| frozen pool | `data/pool/sowodb_t1_frozen_pool.npz` |

`ref_t1_dinov2_vitb14_cap1000_v1.npz` is a Stage-2 artefact and is **not** read
by Method V3: `C` needs the base export and the two views, not the reference.

### 13.1 The detector source, verified

`https://github.com/gubiczam/PROB.git` is the **only** PROB URL this project has
ever used — twelve occurrences across the full git history of this repository,
and never any other. It is a fork, and the pinned commit is **fork-only**: it
carries `daowod_prob_bridge.py`, the 1,057-line CLI this project drives PROB
through, which exists on no upstream branch. Verified on 2026-09-03 with
`git ls-remote`:

```
$ git ls-remote --heads https://github.com/gubiczam/PROB.git
874c0553...  refs/heads/feat/daowod-bridge
4c66be1a52cad9360e09c729e9134aba8fe0b531  refs/heads/feat/daowod-bridge-v2
cbd5bfd3...  refs/heads/main

$ git ls-remote https://github.com/orrzohar/PROB.git | grep 4c66be1a
(nothing — the pinned commit is not in upstream history)
```

The commit resolves to *Make --replay-ids do something, or refuse the run*,
2026-08-24. **There is therefore nothing to substitute**, and §13's URL and SHA
may not be changed to make a clone succeed.

A Method V3 attempt on 2026-09-03 died in the PROB setup cell on
`git clone ... exit status 128`. The URL was reachable and the pin present both
before and after, so the failure was transient — the class of failure an
anonymous clone from a shared Colab egress address produces. `exit status 128`
does not distinguish that from a deleted repository, which is why
`owl.bridge.verify_remote_commit` now proves the URL and the SHA **before** pip,
the CUDA kernel build and the smoke test, retries the read-only probe, honours a
pin that has become an interior commit, and refuses to name an alternative
repository in its error. `owl.bridge.local_checkout_matches` accepts an existing
checkout whose origin and `HEAD` are exactly the pinned ones — byte-identical to
a fresh clone, and the recovery if the host is ever truly unavailable.

All of these, plus the population fingerprint and the checkpoint hash, are
written into the machine-readable run manifest
`MyDrive/OWL/results/method_v3_selection_transfer/manifest.json`.
