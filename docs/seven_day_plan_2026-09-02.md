# SEVEN-DAY OWOD RESEARCH EXECUTION PLAN

Written 2026-09-02, before any new experiment ran. Every number below is read off a
file in this repository or off the attached `controlled_lt_fast_v1_comparison` output.
Nothing here is estimated by hand.

Audit basis: `owod.docx`, the 2026-08-25 consultation note, `owl/` (7,422 lines),
`tools/` (5,357), `tests/` (5,179, **246 pass**), `docs/` (1,814),
`data/results/*.csv`, `data/reference/measured/*`, git log to `afcaedf`.

---

## A. Current state

### A.1 The infrastructure is good and should not be rewritten

All seven Aug-25 ideas have code. `tests/` covers them; `ruff` reports 27 findings, all
cosmetic and all in the T1-anchor tooling. The scientific hygiene in this repo is above
what I usually see: acquisition provably never reads `oracle()`
(`test_scoring_never_reads_an_answer`), the frozen-feature surrogate is *forbidden* from
reporting detection metrics because it was measured to rank arms in reverse order, and
`docs/replay_evaluation_protocol_2026-08-29.md` is a genuine pre-registration.

**Reusable as-is:** `owl/scoring.py`, `owl/clustering.py`, `owl/selection.py`,
`owl/labelling.py`, `owl/replay.py`, `owl/exemplars.py` (object-level replay V3),
`owl/protocol.py` (one-class-per-task chain), `owl/metrics.py`
(`unknown_recall_by_group`, per-class AP50 recovery from `coco_eval_bbox`),
`owl/runner.py`, `tools/run_experiments.py`, `tools/compare_replay.py`,
`tools/analyze_chain.py`, `tools/dry_run_notebook.py`, `notebooks/owod_active.ipynb`.

### A.2 Three defects found in the audit

**Defect 1 — the binary coherence gate is a no-op in every arm that reports it.**
Verified directly on the committed pool:

```
kmeans K=1600, min_samples=5  ->  gate closed on 0 of 80,000 candidates (0.0000%)
smallest cluster size = 5;  clusters with size < 5 = 0 of 1600
```

`coherence_method='binary'` and `'off'` therefore return the identical vector.
Consequence, confirmed in `data/results/selection_arms.csv`: `consult` and
`consult_no_gate` are **bitwise identical on all 3 seeds × all 3 round settings** — they
are the same experiment run twice, not a treatment and its control. Every arm the
project describes as carrying "the consultation's binary gate" (`consult`,
`consult_batch`, `consult_shared_cluster`, `prior_consult`, `prior_consult_batch`)
actually runs **ungated**.

The gate was only ever measured under DBSCAN (`data/results/coherence_gate.csv`), which
no arm uses. So Aug-25 point 4 — the supervisor's own request — is **claimed as
implemented and is untested in every result the project reports.** This is the first
thing to fix, and the honest version of it is a good slide.

**Defect 2 — a headline number is single-seed and shrinks by more than half on three.**
`docs/konzultacio_...md` reports the iterative gain as `consult 26 → 36 (+38%)`. Over
the three committed seeds:

| arm | 600×1 unknowns | 6×100 unknowns | change | seed sd |
|---|---:|---:|---:|---:|
| `consult` | 30.7 | 35.3 | **+15.2%** | ±4–6 |
| `random` | 24.3 | 17.0 | −30.1% | **±8.4** |

`random`'s seed spread (±8.4) is larger than `consult`'s iterative gain (+4.6 objects).
On the *total-unknowns* endpoint, H3 is at the noise floor. It is not at the noise floor
on the endpoint that matters — see C.3.

**Defect 3 — arms are compared at equal region cost but unequal training supervision.**
At budget 600, `objectness` opens 548 images and `prior_consult_batch` opens 308. Under
`known_plus_selected` both cost 600 oracle units, but the arm that opens 548 images
harvests far more free known-class boxes. `images_opened` is recorded and never reported
as a supervision axis. This is exactly the "hidden unequal supervision" trap, and it
currently cuts *against* our own method, so fixing it is not self-serving.

### A.3 What is unreadable locally

`OWL/anchors/` does not exist in the mounted Drive, and content reads under
`OWL/work/` time out (`fts_read: Operation timed out`) even though directory listings
succeed. `OWL/work/prior_consult_batch/` has `t2…t6` subdirectories;
`OWL/work/random/` and `OWL/work/objectness/` could not be listed at all.

So I **cannot verify locally** whether the completed `random__none` FAST chain quoted in
`docs/eredmenyek_vazlat.md` still exists. That single fact decides the GPU schedule, and
a 5-minute zero-GPU Colab cell settles it. It is task J.1.

---

## B. What we are actually trying to prove

One sentence: **an estimated data distribution lets a fixed budget be spent better —
annotation budget for discovery (A), memory budget for retention (B).**

Five hypotheses. Each has an independent variable, dependent metrics, controls, an
expected result, and a condition that kills it.

### H1 — Distribution-aware selection buys more *tail* per oracle unit

* **IV** acquisition score.
* **DV** primary: real tail-class unknown objects found per oracle unit; tail share of
  finds. Secondary: distinct tail classes covered; images opened (supervision control).
* **Controls** `random` (floor + noise-floor estimator), `objectness` (learning-free
  prior — the real bar), `plan` (the research plan's exact equation, unchanged).
  Identical candidate pool, identical budget, 3 seeds.
* **Expected** `objectness` wins total unknowns; the full method wins **tail** objects at
  equal oracle cost.
* **Falsified if** the full method fails to beat `objectness` on tail objects in ≥2 of 3
  seeds, **or** if its advantage disappears once images-opened is equalised.

### H2 — Binary coherence suppresses isolated junk *only on an admissible subpool*

* **IV** `coh` ∈ {off, binary on the full pool, binary on the objectness-admissible
  subpool}.
* **DV** share of gated-out candidates that are real unknown objects vs background;
  tail objects retained; noise fraction; eps/min_samples sensitivity.
* **Controls** gate off, at otherwise identical configuration.
* **Expected** on the full pool the gate removes real objects preferentially (already
  measured: 92% vs 60% at eps 0.15) because the pool is 81% background and background is
  the dense region; on the admissible subpool, where the prior has already removed
  background, the noise flag recovers its intended meaning.
* **Falsified if** on the admissible subpool the gate's removal rate on real unknowns
  still exceeds its rate on background. Then coherence is reported as a **negative
  result** and dropped from the method.

### H3 — Iterative selection reallocates the budget from head to tail

Note the endpoint: **not** "finds more", which is at the noise floor (A.2).

* **IV** `rounds` ∈ {1, 6} at fixed total budget 600.
* **DV** tail objects, tail share, total unknowns, images opened.
* **Controls** arms with no labelled-set-dependent term (`objectness`, `entropy`,
  `plan`, `consult_shared_cluster`) must change by **exactly zero** — that is the
  mechanism check, and a non-zero change there means an implementation leak.
* **Expected** only labelled-set-dependent arms move, and they move by shifting budget
  from head to tail rather than by uniform gain.
* **Falsified if** the mechanism check fails, or if the shift is inside `random`'s seed
  spread.

### H4 — `known_plus_selected` removes half-labelling at zero extra oracle cost

* **IV** labelling policy, on **identical** selected regions.
* **DV** reported separately, never merged: oracle cost; opened images; labelled boxes;
  half-labelled background share; then detector forgetting and new-class AP.
* **Controls** same selection, same seed, same training schedule.
* **Expected** cost equal to `box_only`, half-labelling 0%, supervision ~6×, forgetting
  far lower.
* **Falsified if** the forgetting advantage is fully explained by supervision volume
  rather than by removing the half-labelling error. **This one is partly true already
  and must be said out loud** — the policy does buy more supervision, so the honest claim
  is "cheaper *and* better supervised", not "better at equal supervision".

### H5 — Under fixed memory, allocation changes the stability/plasticity trade-off

* **IV** α ∈ {no replay, 0 (uniform), −0.5 (tail-favouring)} at fixed **M = 400
  objects**.
* **DV** primary: per-class relative forgetting on the 19 fixed t1 classes. Secondary:
  head/medium/tail aggregate — secondary because its denominator moves. **Mandatory
  alongside every retention number:** new-class AP on the same task.
* **Controls** identical (random) selection, identical current-task supervision,
  identical init and schedule, and `delivered == M` exactly for every arm (replay V3
  guarantees this; V2 delivered 464 vs 1,240 objects for a 400 budget, a 2.67× confound).
* **Expected** tail-favouring reduces relative forgetting on the rarest t1 classes
  without an equal-or-larger head loss.
* **Falsified by** the pre-registered rule in `docs/replay_evaluation_protocol_2026-08-29.md`
  §4.1.
* **Pre-registered prior:** frequency was already measured to be a *poor* predictor of
  forgetting on these 19 classes (R² ≈ 0.000; `aeroplane` at 5,135 objects forgets 67.6
  points while the rarer `cat` at 4,768 forgets 5.1). **A null result is likely, and it
  is itself a publishable finding** — it says frequency is the wrong allocation axis and
  points at vulnerability-weighted memory as the follow-up.

---

## C. What existing results already tell us

### C.1 The controlled-LT FAST anchors are dead. Do not build on them.

| condition | overall mAP50 | head | medium | **tail** | classes at exactly 0.0 AP |
|---|---:|---:|---:|---:|---:|
| lt10 | 0.507 | 1.336 | 0.047 | **0.000** | 17 of 19 |
| lt50 | 0.105 | 0.247 | 0.044 | **0.000** | 16 of 19 |
| lt100 | 0.0014 | 0.004 | 0.000 | **0.000** | 18 of 19 |

For scale, the **real** T1 anchor in `data/reference/measured/budget0_metrics.json`
scores previous-19 mAP50 = **73.65**, known mAP = 34.98, U-Recall50 = 14.81. The FAST
anchors are ~150–500× below it.

Three reasons this line is finished for this week:

1. **Nothing learned.** Tail mAP is 0.000 in all three conditions. Downstream forgetting
   analysis needs something to forget.
2. **The only moving number is `person`** (9.35 → 1.58 → 0.026). Every delta table
   reduces to one class. The reported Spearman(AP50, log count) of 0.27/0.45/0.39 is
   computed over a vector that is 18/19 zeros — it is not interpretable.
3. **The compute gap is structural, not marginal.** `docs/t1_anchor_training_protocol.md`
   fixes the real recipe at **183,434 optimizer updates**; FAST ran **12,000** = 6.5%.
   Closing that gap is ~8.5 h × several sessions × 3 conditions — the entire week, spent
   on a *precondition* rather than a result.

**And the axis is partly redundant anyway.** `data/reference/longtail/summary.csv`:
natural S-OWODB t1 has ρ = **202.8**, while the most extreme controlled condition is
ρ = 100. The natural benchmark is *more* imbalanced than lt100, and it already has a
converged anchor. The controlled conditions are a cleaner-attribution refinement, not a
prerequisite.

The manifest infrastructure is genuinely good (object count matched at 79,233 across all
three conditions, achieved ρ within 0.14% of requested, SHA-256 pinned). **Keep it,
park it, cite it as the confirmation experiment.**

### C.2 Contribution A already has a real result — and it is not the one in the docs

Three seeds, budget 600, `data/results/selection_arms.csv`:

| arm | unknowns | **tail** | tail share | tail classes | images opened |
|---|---:|---:|---:|---:|---:|
| `random` | 24.3 ± 8.4 | 8.7 | 36% | 7.0 | 501 |
| `entropy` | 32.0 | 14.0 | 44% | 5.0 | 295 |
| `plan` (plan's exact equation) | 21.0 | 11.0 | 52% | 8.0 | 339 |
| `objectness` (free control) | **155.0** | 48.0 | 31% | 9.0 | 548 |
| `prior_consult_batch` 600×1 | 87.3 | 53.0 | 61% | 9.3 | 345 |
| **`prior_consult_batch` 6×100** | 76.0 | **61.0** | **80%** | 9.7 | **308** |

Read honestly: **the free control finds twice as many unknowns as our method and must be
reported as the headline baseline.** What the method changes is *composition* — 61 tail
objects vs 48, at 80% tail share vs 31%, from 308 opened images vs 548. That is the
research plan's own claim ("the same tail level from substantially fewer annotations"),
and it holds 3/3 seeds. It is a narrower claim than "our method finds more unknowns", and
it is the defensible one.

### C.3 The iterative-vs-one-shot result is real on the right endpoint

For `prior_consult_batch`, 600×1 → 6×100 moves total unknowns **down** (87.3 → 76.0) and
tail objects **up** (53.0 → 61.0, +15.1%), with tail share 61% → 80% and images opened
345 → 308. Three seeds, sd ±2.6–3.0 — outside the seed spread.

That is a mechanism, not a gain: recomputing after each round spends later rounds on
rarer material because the labelled pool already covers the common material. **"Iterative
selection reallocates the budget from head to tail" is a better claim than "iterative
finds more", and it is the one the data supports.**

Mechanism check passes: `objectness`, `entropy`, `plan`, `consult_shared_cluster` change
by **exactly 0.0** across rounds, because none has a labelled-set-dependent term.

### C.4 The labelling-policy result is the cleanest thing in the project

`data/results/labelling_policy.csv`, 600 regions → 306 opened images:

| policy | oracle cost | labelled boxes | half-labelled background | supervision / oracle unit |
|---|---:|---:|---:|---:|
| `box_only` | 600 | 423 | **20.7%** | 0.71 |
| `full_image` | 1082 (1.80×) | 3,498 | 0% | 3.23 |
| **`known_plus_selected`** | **600** | 2,729 | **0%** | **4.55** |

And the detector confirms the mechanism (`data/reference/measured/real_group_forgetting.csv`):
restoring the task-1 boxes that `box_only` was discarding cut previous-19 forgetting from
**27.0 → 2.69** points with no replay at all.

### C.5 What the real GPU runs say about replay — and about plasticity

From `real_group_forgetting.csv`, budget 600:

| arm | prev-19 mAP50 | forgetting | U-Recall50 | A-OSE | new-class AP |
|---|---:|---:|---:|---:|---:|
| anchor | 73.65 | 0.00 | 14.81 | 1047 | — |
| full t2 supervision (upper bound) | 66.33 | 7.32 | 22.05 | 3380 | 28–42 |
| `box_only`-equivalent (`train` mode) | 46.64 | 27.01 | 27.21 | 6340 | ~0.00 |
| `known_plus_selected`, no replay | 70.96 | **2.69** | 24.75 | 8554 | ~0.00 |
| `known_plus_selected` + replay α=0 | 70.45 | 3.20 | **27.15** | **3168** | 0.00 |

Two things follow, and both constrain the week:

1. **Replay's measured effect is on known/unknown confusion, not on forgetting.** At
   α=0 forgetting is marginally *worse* (3.20 vs 2.69) while A-OSE drops 8554 → 3168 —
   a 63% reduction in open-set error. That is a real effect and nobody has claimed it
   yet. It should be a reported endpoint for H5, not a footnote.
2. **New-class AP is ~0.00 at a 600-region budget, on every arm.** The measured
   efficiency curve is 10 instances → 0.00, 20 → 0.00, 50 → 0.70, ~1000 → 36.13. So
   **plasticity is not measurable at this budget** and no selection method can be
   distinguished on it. Stating this as a result — with the curve — is the correct move,
   and it is *why* contribution A must be judged on discovery, not on new-class mAP.

### C.6 One existing claim must be retired

`real_group_forgetting.csv` has `old_tail_n = 1`. Every "old tail forgetting" number in
that table is **a single class** (`bear`). Per the project's own rule, that is not a tail
claim. The pre-registration already caught this (§4.2); the results draft must not quote
band aggregates without per-class trajectories.

---

## D. What not to spend time on

| Do not | Why |
|---|---|
| Retrain controlled-LT anchors (LT10/50/100) | 183k updates × 3 conditions ≈ the whole week for a precondition. Natural ρ=203 already exceeds lt100, and a converged natural anchor exists. Park the manifests; cite as confirmation experiment. |
| A 6-task GPU chain for the **selection** arms | New-class AP ≈ 0.00 at 600 regions on every arm (C.5). The chain cannot separate selection methods on plasticity, and its discovery endpoint is already answerable on CPU in minutes. |
| A wide α sweep | The pre-registered prior says frequency ≈ 0 correlation with forgetting on these 19 classes. Three arms (none / 0 / −0.5) test the hypothesis; more α values only invite fitting the prettiest curve. |
| Vulnerability-weighted memory (`m_c ∝ n_c^α · v_c^β`) | The plan's own instruction: only after the simple frequency hypothesis is tested. It is the Day-7 "next question", not this week's work. |
| LwF / EWC / BiC | Requires surgery inside PROB's loss. WA is checkpoint-only and is the right next baseline — after a real GPU run confirms the classifier-head layout. |
| `rounds=12` | Adds nothing measurable over `rounds=6` in three seeds. Drop it and halve the CPU matrix. |
| Broad refactors, and `runner.py:530` while a chain is live | The oracle-cost bug is real but must be fixed **between** chains, never mid-chain — otherwise early and late tasks in one chain follow different rules, which is worse than a known bias. |

---

## E. Revised method after Aug 25

### E.1 The problem the current form has

The plan's additive equation puts a learning-free object prior nowhere, and puts
coherence in a place where it cannot work. Measured cause: the candidate pool is **81%
background**, background regions are near-copies of each other, so **background occupies
the densest part of the feature space.** Any density-based coherence computed on the full
pool therefore rewards background and punishes real objects — which is exactly what
`coherence_gate.csv` shows (92% of real unknowns gated out vs 60% of background at
eps 0.15). The k-means variant avoids that failure only by never firing at all (A.2).

### E.2 The proposed form

```
s(x) = A(x) · [ U(x) + λ·D_known(x) + γ·w(x)·coh_A(x) + μ·B(x | S) ]
```

| term | definition | what it defends against | when it updates |
|---|---|---|---|
| `A(x)` | `objectness(x) · sqrt(area(x))`, **raw, not rank-normalised** — an admissibility factor, so nothing rescues a region that is not object-like | background | never |
| `U(x)` | normalised Shannon entropy of the 81-way posterior ÷ log 81 | confident regions | never (frozen detector) |
| `D_known(x)` | cosine distance to the nearest **already-labelled** region | re-buying what we know | **every round** |
| `w(x)` | `−log(n_c / N)`, `n_c` = size of the candidate's cluster in the shared partition | head redundancy | every round if refit |
| `coh_A(x)` | **binary**, DBSCAN core/noise — computed on the **admissible subpool**, not the full pool | isolated outliers | with the partition |
| `B(x\|S)` | greedy k-means++-style penalty: each pick pushes down what resembles it | 600 near-copies | **within a round** |

**The one substantive change: `coh` is computed on the admissible subpool.** Once `A`
has removed background, "dense" no longer means "background", and the noise flag can
recover its intended meaning. This is not a guess — the existing clustering diagnostics
already point that way: restricting to the top 30% by objectness moves unknown purity
0.384 → **0.454** and unknown recall 0.489 → **0.627**. Day 1 tests it directly, and H2
is written so that a failure is a reportable negative, not a dead end.

### E.3 Why `D` stays two terms rather than one scalar

The supervisor asked whether diversity means distance from the old or spread among the
new. **Both, kept separate**, because they differ in all three ways that matter:

| | `D_known` | `B(x\|S)` |
|---|---|---|
| meaning | historical novelty vs labelled memory | within-batch non-redundancy |
| reference set | grows across rounds | grows within a round |
| ablatable | yes, independently | yes, independently |

Collapsing them into one scalar `D` would make the 6×100-vs-600×1 experiment
uninterpretable: `D_known` is the *only* reason rounds matter, and `B` partly substitutes
for rounds. Measured: `consult` (D_known only) gains +15.2% from rounds;
`consult_shared_cluster` (cluster-D, no labelled dependence) gains **exactly 0.0%**. Two
terms, two mechanisms, two ablations. Cost: one extra hyperparameter (`μ`), fixed once at
0.3 before any endpoint was inspected and never tuned.

### E.4 Rarity without oracle labels

Both `w` and the cluster structure come from **one** partition of the candidate pool
(Aug-25 point 3), so rarity and novelty are not two unrelated distance functions.
Quality is judged by **known contamination with an enrichment baseline** — "does this
cluster hold *more* known content than a random cluster would" — because an absolute
majority rule is degenerate in a pool that is 81% background. Two floors stop the sweep
running away to K=N: minimum unknown recall, and minimum mean cluster size (rarity is
read off cluster *size*). At K=1600: contamination 0.028 estimated / 0.116 verified,
unknown recall 0.82.

Oracle labels are used **only** for retrospective evaluation of the selector, never in
the score. `tests/test_owl.py::test_scoring_never_reads_an_answer` enforces it.

---

## F. Minimum experiment matrix

### F.1 Selection — CPU, free, 3 seeds, `rounds ∈ {1, 6}`

| arm | why it exists |
|---|---|
| `random` | floor, **and the noise-floor estimator** (±8.4 objects) |
| `objectness` | learning-free control; the real bar. **Not removable — it beats us on total unknowns.** |
| `entropy` | supervisor kept entropy; isolates `U` alone |
| `plan` | the plan's exact equation, untouched — the "before" picture |
| `A·(U + λD_known)` | does true novelty add over the free prior |
| `A·(U + λD_known + γw)` | does rarity add |
| `A·(U + λD_known + γw·coh_A)` | **the repaired gate — the new arm** |
| `A·(… + μB)` = full method | does batch diversity add |
| gate ablation: `coh` off / full-pool / admissible | isolates Defect 1, one variable |

9 arms × 2 rounds × 3 seeds. Minutes each. **New reported columns:** `images_opened` and
`labelled_boxes`, so no arm can win on hidden supervision (Defect 3).

### F.2 Labelling policy — CPU accounting + existing GPU evidence

3 policies on **identical** selected regions. Report the six quantities separately
(oracle cost / opened images / labelled boxes / known retained / new revealed / ignored)
and never a single merged "supervision" number.

### F.3 Replay — GPU, the one chain that earns its hours

| arm | replay | tests |
|---|---|---|
| `random__none` | — | forgetting floor |
| `random__uniform` | α=0, M=400 | the literature standard |
| `random__tail_favouring` | α=−0.5, M=400 | contribution B |

Everything else identical: random selection, `known_plus_selected`, t1→t6, 600 regions /
task, 6×100 rounds, 5 epochs, lr 2e-4, batch 2, seed 0, replay V3 object-level,
`delivered == 400` exactly. **Added endpoint: A-OSE**, per C.5.

### F.4 Contingency — A-side detector check (only if F.3 gets a free arm)

Single task t2 only (not a chain), arms {`random`, `objectness`, full method}, fixed
oracle cost. ~1.7 h each. Gives detector-side U-Recall by frequency group — the plan's
headline endpoint — without paying for a 6-task chain that cannot measure plasticity.

**No Cartesian products.** Selection is swept on CPU where it is free; replay is swept on
GPU where it must be; the two are never varied together.

---

## G. Day 1 – Day 7

### DAY 1 — Close the audit, repair the gate
* **Question** Is any reported result affected by the gate being a no-op, and does a gate on the admissible subpool behave as the supervisor intended?
* **Implementation** Zero-GPU Colab cell inventorying Drive and dumping every `results_*.csv`. Add `coherence_scope ∈ {pool, admissible}`; DBSCAN coherence on the objectness-admissible subpool; a test that **fails** if a `binary` arm scores identically to `off`.
* **Experiment** Gate diagnostics: 3 scopes × eps grid × min_samples grid × 3 seeds, reporting removal rate on real unknowns vs background, tail objects retained, noise fraction.
* **GPU** 0. CPU ~1 h.
* **Outputs** `data/results/coherence_scope.csv`, `figures/gate_diagnostic.png`, updated `docs/method.md` + coverage doc, new test.
* **GO/NO-GO** If on the admissible subpool the gate still removes real unknowns more often than background → coherence is reported as a **negative result** and dropped from the method. Either way Day 1 produces a slide.
* **For the supervisor** Deliverable 4, plus the correction that shows we test our own claims.

### DAY 2 — Selection matrix, properly seeded and supervision-controlled
* **Question** Does each term earn its place at equal oracle cost **and** equal opened-image supervision?
* **Implementation** Register the F.1 ablation arms; add `images_opened` / `labelled_boxes` as reported axes; add an equal-opened-images control variant; drop `rounds=12`.
* **Experiment** 9 arms × {1,6} × 3 seeds (5 if it stays under an hour).
* **GPU** 0. CPU ~2 h.
* **Outputs** `data/results/selection_arms_v2.csv`, ablation table, tail-per-oracle-unit figure, `docs/02_method.md`.
* **GO/NO-GO** If the full method fails to beat `objectness` on tail objects in ≥2/3 seeds → report it as negative and make `objectness` the headline baseline. **Do not delete the baseline.**
* **For the supervisor** Deliverables 1, 2, 3, 5.

### DAY 3 — Launch the replay chain (arm 1)
* **Question** What is the forgetting floor with no replay, on a real anchor?
* **Implementation** No new code if Day 1's inventory validates the existing `random__none` workspace against the current fingerprint. Otherwise run it.
* **Experiment** `random__none`, t1→t6.
* **GPU** ~7 h, resumable.
* **Outputs** `work/random__none/results_*.csv`, per-task metrics.
* **GO/NO-GO** If per-class AP50 recovery from `coco_eval_bbox` fails validation, stop and fix before spending two more arms on an unreadable endpoint.
* **For the supervisor** Deliverable 9 (before/learned/forgot curves).

### DAY 4 — Replay arm 2
* **Question** What does the literature-standard uniform memory buy?
* **Experiment** `random__uniform`, α=0, M=400, t1→t6. Strict V3 audit (`delivered == 400`).
* **GPU** ~7 h.
* **Outputs** `work/random__uniform/*`, audit receipt.
* **GO/NO-GO** If `delivered != 400` exactly, the arm is void — fix, do not report.
* **For the supervisor** Deliverables 7, 8.

### DAY 5 — Replay arm 3
* **Question** Does tail-favouring allocation change the stability/plasticity trade-off?
* **Experiment** `random__tail_favouring`, α=−0.5, M=400, t1→t6.
* **GPU** ~7 h.
* **Outputs** `work/random__tail_favouring/*`.
* **GO/NO-GO** Decision follows the **pre-registered** §4.1 rule, not inspection.
* **For the supervisor** Deliverables 8, 10, 11.

### DAY 6 — Analysis, and the contingency arm
* **Question** What do the three arms actually say — per class, not per band?
* **Implementation** `tools/compare_replay.py` (exists). Add A-OSE to the reported endpoints. Vulnerability regression: forgetting vs log frequency vs anchor AP.
* **Experiment** If Day 3 reused an existing chain, spend the freed slot on F.4 (t2-only, 3 selection arms, ~5 h).
* **GPU** 0–5 h.
* **Outputs** 6 tables + 6 figures under `supervisor_update_2026_09/`.
* **GO/NO-GO** Any band-level claim without a per-class trajectory is cut (C.6).
* **For the supervisor** Deliverables 8, 9, 10, 12.

### DAY 7 — The deliverable
* **Question** What did we learn, what failed, what is next?
* **Implementation** Assemble `supervisor_update_2026_09/`; fix `runner.py:530` (safe now — no chain running); commit and push.
* **GPU** 0.
* **Outputs** `SUPERVISOR_SUMMARY.md` answering all 14 questions; `01_research_questions.md` … `05_open_questions.md`; `tables/`, `figures/`, `configs/`, `provenance/`.
* **GO/NO-GO** Every number traces to a committed file. No hand-typed values.

---

## H. GPU / time budget

| day | GPU (T4) | CPU | slack |
|---|---:|---:|---|
| 1 | 0 h | 1 h | full day |
| 2 | 0 h | 2 h | full day |
| 3 | 7 h (resumable) | — | reuse may make it 0 |
| 4 | 7 h | — | resumable |
| 5 | 7 h | — | resumable |
| 6 | 0–5 h | 1 h | the buffer day |
| 7 | 0 h | 2 h | — |
| **total** | **21–26 h** | **6 h** | 1 buffer day + resume everywhere |

Every job is resumable, checkpointed, and fingerprint-guarded — a dropped Colab session
costs minutes, not the run. The heaviest single job is 7 h, so it fits one overnight
session. **No job exceeds one session.** Compare with the abandoned line: three full
controlled anchors at 183,434 updates each would be roughly 60–80 T4-hours for a
precondition.

---

## I. Expected supervisor deliverables

`supervisor_update_2026_09/` — README, `01_research_questions.md`, `02_method.md`,
`03_experiment_protocol.md`, `04_results.md`, `05_open_questions.md`, plus `tables/`,
`figures/`, `configs/`, `provenance/`, and `SUPERVISOR_SUMMARY.md`.

Against your list of 12:

| # | want | expected |
|---|---|---|
| 1 | revised formula | **yes** (E.2) |
| 2 | how the four terms are computed | **yes** (E.2 table + figure) |
| 3 | new D really measures novelty | **yes** — the rounds mechanism check is the proof |
| 4 | DBSCAN / coherence diagnostics | **yes** — including a possible clean negative |
| 5 | 600×1 vs 6×100 | **yes**, 3 seeds, reframed as head→tail reallocation |
| 6 | fair labelling-policy comparison | **yes** — six cost/supervision quantities kept separate |
| 7 | concrete replay protocol | **yes** — V3, `delivered == M` exactly |
| 8 | none / uniform / tail comparison | **yes** — the week's one GPU chain |
| 9 | task-wise before/learned/forgot curves | **yes** for retention; **plasticity reported as ~0 with the efficiency curve explaining why** |
| 10 | head/medium/tail analysis | **yes**, per-class primary, band aggregate secondary |
| 11 | reproducible Colab notebook | **yes** — `notebooks/owod_active.ipynb`, Run-all |
| 12 | concise conclusion | **yes** |

**Not delivered, stated as such:** converged controlled-LT anchors (C.1); detector-side
tail U-Recall vs oracle-cost curve for the selection arms unless Day 6's contingency
runs; any significance claim (one seed on GPU).

Three strong experiments, not fifteen weak ones: **(1)** the selection ablation with a
repaired coherence gate and honest baselines, **(2)** the labelling-policy accounting,
**(3)** the fixed-memory replay comparison.

---

## J. Immediate first implementation task

**J.1 — Drive inventory cell (zero GPU, ~5 min).** A read-only Colab cell that lists
`OWL/work/*` and `OWL/anchors/*`, prints every `config.json` fingerprint against the
current `CycleConfig.RESULT_AFFECTING`, and dumps every `results_*.csv`. It answers the
one thing I could not answer locally (A.3) and it decides whether Day 3 costs 7 GPU-hours
or zero. Nothing else should run before it.

**J.2 — Repair the coherence gate (CPU).** `coherence_scope` option, DBSCAN on the
admissible subpool, plus the regression test that fails when a `binary` arm scores
identically to `off`. This is the fix for the defect that currently invalidates the
project's description of its own headline arm.

I will do J.1 and J.2 on your go-ahead, in that order.

---

## K. Questions for you

Only three, and only the first two are blocking.

1. **GPU reality check.** The plan assumes ~7 usable T4-hours per day on days 3–5. If
   Colab Pro is giving you less than that this week, I will cut the replay chain from
   t1→t6 to t1→t4 (3 increments, ~4 h/arm) rather than drop an arm — losing chain length
   costs less than losing the uniform-vs-tail contrast.

2. **One research decision.** With ~21 GPU-hours, I recommend spending **all** of it on
   contribution B (replay), and making contribution A's case on the CPU discovery
   endpoint plus the existing measured runs. Rationale: A's detector endpoint is
   corrupted by new-class AP ≈ 0.00 at a 600-region budget (C.5), whereas B's endpoint
   (forgetting, A-OSE) has measured dynamic range of 2.7–46.4 points. If you would rather
   split — one replay arm dropped in exchange for the F.4 t2-only selection comparison —
   say so, and I will restructure days 4–6. **My recommendation: don't split unless
   Day 1's inventory hands us a free arm.**

3. **Non-blocking.** Should `docs/` stay Hungarian? This plan is in English because your
   brief was; I will match whatever you prefer for the supervisor folder. The Hungarian
   convention in existing `docs/` is untouched either way.

---

*Nothing in this document was tuned by looking at an endpoint. The three defects in A.2
were found by re-running the committed experiments, and two of the three cut against our
own method.*
