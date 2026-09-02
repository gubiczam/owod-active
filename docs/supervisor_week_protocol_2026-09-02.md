# Supervisor-week protocol — frozen 2026-09-02

Three experiments, frozen **before** any of them ran. The point of this file is that the
hypothesis cannot move after the numbers arrive. If a claim is not written here, it is not
a claim we make this week.

Standing rules for all three:

* **Oracle labels are retrospective only.** They score the selector after the fact. They
  never enter an acquisition score, a clustering, or a gate. Enforced by
  `tests/test_owl.py::test_scoring_never_reads_an_answer`.
* **One seed is descriptive.** CPU experiments run ≥3 seeds and report spread. GPU runs one
  seed and makes no significance claim. No p-values anywhere.
* **Distinct GT objects, not proposals**, for every discovery count. Two proposals on one
  object are one discovery and one oracle answer.
* **Negative results are reported.** A falsified hypothesis is written up as falsified.
* **Fixed before looking:** λ=0.2, γ=0.5, μ=0.3, K=1600. Set before any endpoint was seen,
  never tuned against one.

| | A1 selection | A2 annotation policy | B1 replay |
|---|---|---|---|
| causal question | where to spend annotation budget | what one selected region reveals | where to spend replay memory |
| unit of budget | oracle regions | oracle regions | memory objects |
| compute | CPU, 3+ seeds | CPU + existing GPU evidence | GPU, 1 seed |
| never varied together with | A2, B1 | A1, B1 | A1, A2 |

---

## EXPERIMENT A1 — ACTIVE SELECTION

**Research question.** Under a fixed oracle budget of 600 regions, which acquisition score
puts the most *rare-class* unknown objects in front of the annotator?

**Hypothesis (H1).** A score combining entropy, novelty against the labelled set, and
cluster-estimated rarity — admitted through a learning-free object-likeness factor — finds
more distinct **tail-class** unknown objects per oracle region than random, entropy, or the
learning-free prior alone, even though the prior alone finds more unknowns *in total*.

**Independent variable.** The acquisition score only.

**Controls.**

| control | what it bounds |
|---|---|
| `random` | the floor, **and the seed-noise estimator** |
| `objectness` = `A(x)` alone | the learning-free bar. **Not removable, even though it beats us on total discovery.** |
| `entropy` = `U(x)` alone | isolates the term the supervisor kept |
| `plan` | the research plan's original equation, untouched — the "before" |

**Fixed across arms.** The same committed 80,000-proposal PROB t1 pass (2,400 images, 19
known classes); budget 600 regions; `n_known=19`; one shared k-means partition per seed
(K=1600); λ, γ, μ as above; identical candidate pool and exclusion mask.

**Primary endpoint.** **Distinct tail-class unknown GT objects discovered per 600 oracle
regions.** Chosen because plasticity is not measurable at this budget — measured new-class
AP is ≈0.00 at 600 regions on every arm, with the efficiency curve 10→0.00, 20→0.00,
50→0.70, ~1000→36.13 — so detector mAP cannot separate acquisition methods here.

**Secondary endpoints.** All distinct-object counts unless stated: total unknown objects;
tail share; unknown classes covered; tail classes covered; head/medium/tail composition;
images opened; **proposals per distinct object** (redundancy); selected background /
selected known / selected unknown; oracle regions actually asked.

**Supervision control.** `images_opened` and resulting labelled boxes are reported for every
arm. An arm that opens more images harvests more free known-class boxes under
`known_plus_selected`, so no arm may be called better on discovery while quietly carrying
more supervision.

**Success criterion.** The full method exceeds `objectness` on distinct tail objects in
**≥2 of 3 seeds**, and the advantage survives reporting `images_opened` alongside.

**Falsification / negative reading.** If the full method fails that, we report it as
falsified and name `objectness × √area` as the strongest baseline. If the method wins on
tail while losing on total discovery, we report **both** and frame the contribution as
budget *reallocation*, not as more discovery.

**Expected tables/plots.**
`T-A1.1` arm × endpoint, mean ± sd over seeds. `T-A1.2` term-ablation ladder (rows 4→7).
`F-A1.1` distinct tail objects vs oracle regions, one line per arm.
`F-A1.2` head/medium/tail composition, stacked, one bar per arm.

### A1.1 — Coherence, repaired

**Defect being fixed.** `coherence_method='binary'` closes on **0 of 80,000** candidates at
the configured K=1600 / `min_samples=5`, because the smallest k-means cluster holds 5
members. So `binary` and `off` return identical vectors, and `consult` / `consult_no_gate`
are bitwise identical on 3 seeds × 3 round settings. The gate the supervisor asked for is
untested in every result reported so far.

**Question.** Does binary density coherence remove isolated and background proposals while
preserving real rare unknown structure?

**Sub-hypothesis (H2).** DBSCAN noise on the **full** pool fails because the pool is 81%
background and near-duplicate background occupies the densest region, so "many neighbours"
means "looks like background". On an **objectness-admissible** subpool, where the prior has
already removed most background, the core/noise split recovers its intended meaning.

**Predeclared grid — frozen, and not to be widened.**

```
scope           full_pool | admissible
admissible      top 30% by A(x) = objectness * sqrt(area)      [primary]
                top 10% / 20% / 50%                            [sensitivity only]
eps             0.15 | 0.25 | 0.35 | 0.45
min_samples     5 | 20
pca_dimensions  32
seeds           0, 1, 2
```

**Reported per setting.** Admissible candidates; DBSCAN noise fraction; core+border
fraction; real-unknown rate among coherent points; real-unknown rate among noise points;
background rate among coherent; background rate among noise; distinct real unknown classes
retained; head/medium/tail distinct unknown objects retained; tail-class coverage; known
contamination.

**Predeclared acceptance rule.** The gate is accepted only if some setting in the grid
satisfies **all four**:

1. real-unknown rate among coherent > real-unknown rate among noise — the gate must not
   preferentially delete objects;
2. background rate among noise > background rate among coherent;
3. distinct tail unknown objects retained ≥ **90%** of the pre-gate count;
4. noise fraction ≥ **5%** — the gate must actually do something, which is the direct fix
   for the no-op defect.

**Falsification.** If no setting in the grid satisfies all four, binary coherence is
reported as a **negative result**, `C(x)` is set to constant 1 in the final method, and the
tuning stops there. The full-pool-vs-admissible contrast is reported either way, because it
explains *why* the supervisor's elegant idea does not transfer to this feature space.

**Expected tables/plots.** `T-A1.3` the full grid, one row per setting, accept/reject flag.
`F-A1.3` real-unknown rate among coherent vs among noise, full pool against admissible.

### A1.2 — Novelty and batch diversity, as two separate objectives

The supervisor asked whether diversity means distance from the old or spread among the new.
**Our answer: two different objectives, tested separately.** They differ in reference set,
update cadence, and what they defend against, so collapsing them into one scalar would make
A1.4 uninterpretable.

`D_known(x)` — three candidate definitions were compared on interpretability, compute,
stability across seeds, and compatibility with iterative rounds. **The choice is recorded
before the final comparison runs, and is not made on which produces the best endpoint.**

`B(x|S)` — within-batch redundancy, applied during greedy selection in the manner of
k-means++ seeding: every pick pushes down whatever resembles it.

**Validation, not tuning.** `D_known` must fall as the labelled set grows (it is a distance
to a growing reference set) and must be the *only* reason round count matters. `B(x|S)` must
lower **proposals per distinct object** and mean within-batch cosine similarity. Both are
mechanism checks with a predicted sign, and a wrong sign is an implementation bug.

### A1.3 — Minimum ablation ladder

| # | arm | what it adds |
|---|---|---|
| 1 | `random` | floor + noise estimate |
| 2 | `objectness` = `A` | learning-free bar |
| 3 | `entropy` = `U` | the kept term, alone |
| 4 | `A · (U + λD_known)` | does true novelty pay |
| 5 | `A · (U + λD_known + γR)` | does rarity pay |
| 6 | `A · (U + λD_known + γR·C)` | does the repaired gate pay |
| 7 | `A · (U + λD_known + γR·C + μB)` | does batch diversity pay |
| — | `plan` | the original equation, for reference |

Arms already scientifically equivalent to an existing committed arm are **reused, not
rerun**. `rounds=12` is dropped: it moved nothing beyond `rounds=6` across three seeds.

### A1.4 — One-shot versus iterative

**Question.** Does recomputing labelled-set-dependent novelty and rarity after each 100
selections reallocate later rounds toward rare unknowns?

**Independent variable.** `rounds ∈ {1, 6}` at a fixed total budget of 600 regions.

**Mechanism control — predeclared.** Arms with no labelled-set-dependent term
(`objectness`, `entropy`, `plan`, cluster-sourced novelty) must change by **exactly zero**.
A non-zero change there is an implementation leak, not a finding.

**Reported per round**, cumulative: total unknown proposals; distinct unknown objects;
distinct tail unknowns; tail classes; background selected; known selected; images opened.

**Interpretation rule, fixed now.** If total discovery falls while tail discovery rises, we
report exactly that and call it reallocation. Iterative is **not** called better because one
metric rose. A change inside the `random` arm's seed spread is not a change.

**Expected tables/plots.** `T-A1.4` per-round cumulative table, 600×1 against 6×100.
`F-A1.4` cumulative distinct tail objects against oracle regions spent, per round setting.

---

## EXPERIMENT A2 — ANNOTATION POLICY

**Research question.** When a selected region sits on an image holding several objects, what
should the oracle reveal, and what should the detector train on?

**Hypothesis (H4).** `known_plus_selected` removes half-labelling at the same oracle cost as
`box_only`, because already-known objects need no human. Its detector advantage comes from
**two** sources that must be reported separately: removing the half-labelling error, and
carrying more supervision per oracle unit.

**Independent variable.** The labelling policy, on **identical** selected regions.

**Controls.** `box_only` (cheapest, half-labels), `full_image` (no half-labelling, pays more
oracle units).

**Fixed.** Same selection, seed, budget 600, `known_classes = TASK1`, same training schedule
for any detector number quoted.

**Metrics — annotation cost and training supervision kept strictly apart.**

*Cost side:* selected region budget; images opened; oracle labels requested; half-labelled
image fraction.
*Supervision side:* total labelled boxes; known boxes retained; newly revealed unknown
boxes; ignored/unlabelled objects; supervision-per-oracle-unit ratio.

**Primary endpoint.** Half-labelled background share at equal oracle cost.

**Secondary endpoints.** Supervision per oracle unit; and from existing valid GPU evidence
only — previous-class mAP, new-class AP, forgetting.

**Success criterion.** `known_plus_selected` achieves oracle cost equal to `box_only` with a
half-labelled share of 0, and existing detector evidence shows materially lower forgetting.

**Falsification / negative reading.** If the forgetting advantage is entirely attributable to
supervision volume rather than to removing the half-labelling error, we say so: the honest
claim becomes "cheaper **and** better supervised", not "better at equal supervision". We do
not have a fixed-supervision arm this week and will not imply one.

**No new GPU run for A2** unless the audit finds the existing evidence invalid.

**Expected tables/plots.** `T-A2.1` policy × cost columns and supervision columns, visually
separated. `T-A2.2` the existing detector evidence with its protocol caveats attached.

---

## EXPERIMENT B1 — REPLAY

**Research question.** Given a fixed object-level replay memory M, which previously known
classes should receive the slots?

**Hypothesis (H5).** At fixed M, tail-favouring allocation (α=−0.5) reduces relative
forgetting on the rarest previously known classes compared with uniform (α=0), without an
equal or larger loss on head classes or on new-class learning.

**Independent variable.** The allocation rule only: `none` | `uniform` α=0 | `tail_favouring`
α=−0.5, with `m_c ∝ n_c^α`, `Σ m_c = M = 400` objects.

Head/frequency-proportional (α=1) is **not** in the matrix. It is cheap to add, but the CPU
allocation measurement already shows α=1 drives the rarest class to a single exemplar — the
plan's own predicted failure — and adding a fourth GPU arm costs ~7 T4-hours for a
confirmation. Proposed here, and deliberately not run.

**Controls.** `none` bounds the forgetting floor. `uniform` is the literature standard.

**Fixed across arms.** Random selection with identical seed, so all arms receive a
bit-identical new-supervision image set; `known_plus_selected`; chain t1→t6; 600 regions per
task; 6×100 rounds; 5 epochs; lr 2e-4; batch 2; seed 0; replay protocol V3 object-level;
`replay_reallocate=False`; identical initialisation and schedule; and **`delivered == 400`
exactly on every arm and every task** — V2 delivered 464 against 1,240 objects for a
400 budget, a 2.67× confound caused by the allocation rule rather than by the design.

**Metrics per task.**

*Stability:* previous-class mAP50; per-class forgetting on the **19 fixed t1 classes**;
head/medium/tail forgetting.
*Plasticity:* new-class AP50; current-task mAP50.
*Open-world:* U-Recall50; WI@0.8; A-OSE.
*Memory:* requested / allocated / delivered objects; per-class allocation; replay images;
unique source images; retained / evicted / added.

**Primary endpoint.** Per-class relative forgetting `(anchor − final) / anchor` on the 19
fixed t1 classes, uniform against tail-favouring.

Per-class and not the band aggregate, because the band denominator moves: among *previously
known* classes the tail band holds **one** class (`bear`) at t2–t3 and four by t6, so a band
change is partly composition change. Anchor AP also runs head 65.1 / medium 85.1 / tail
87.7 — rare classes start *higher* — so absolute forgetting is misleading and relative is
the correct measure.

**Secondary endpoints.** Band forgetting (with the caveat attached); new-class AP on the same
task, printed beside every retention number; U-Recall50; **A-OSE**, which existing evidence
suggests is where replay actually acts — at α=0, forgetting was marginally *worse*
(3.20 vs 2.69) while A-OSE fell 8554 → 3168, a 63% reduction nobody has yet claimed.

**Success criterion — the pre-registered two-part reading from
`docs/replay_evaluation_protocol_2026-08-29.md` §4.1, unchanged.** `tail_favouring` beats
`uniform` only if **both** hold: retention favours it on the **majority of t3…t6**, not only
at t6; and the loss in new-class AP and head mAP is no larger than the tail-side gain. If
the first holds and the second does not, it is reported as a **trade-off**, not a win.

**Falsification / negative reading.** If retention does not favour `tail_favouring` on the
majority of t3…t6, H5 is reported as **falsified**. We predeclare the most likely cause:
frequency was already measured to be a poor predictor of forgetting on these 19 classes
(R² ≈ 0.000 on frequency; `aeroplane` at 5,135 objects lost 67.6 points while the rarer `cat`
at 4,768 lost 5.1). A null result therefore indicts the *allocation axis*, not the
allocator, and points at vulnerability-weighted memory as the follow-up. **That follow-up
does not run this week.**

We do not call `tail_favouring` better if it preserves old classes by suppressing new-class
learning. Every retention number is printed next to the plasticity number from the same task.

**Expected tables/plots.**
`T-B1.1` per-task stability × plasticity × open-world, one block per arm.
`T-B1.2` per-class forgetting on the 19 t1 classes, three arms side by side.
`T-B1.3` memory composition: requested / allocated / delivered / retained / evicted / added.
`F-B1.1` per-class relative forgetting against log training frequency, per arm.
`F-B1.2` new-class AP against previous-class mAP — the stability/plasticity plane.
`F-B1.3` A-OSE and U-Recall50 per task, per arm.

---

## Archived, not deleted: the controlled long-tail FAST result

`controlled_lt_fast_v1` (LT10 / LT50 / LT100, 12,000 fixed optimizer updates per condition)
is archived as a **negative / diagnostic** result. Recorded findings:

* The fixed-compute controlled comparison **worked technically**: object count matched at
  79,233 across all three conditions, achieved ρ within 0.14% of requested, SHA-256 pinned
  manifests, deterministic unique-image sampling, hash-linked receipts.
* **12,000 updates is insufficient** for meaningful multi-class T1 learning: overall mAP50
  0.507 / 0.105 / 0.0014, tail mAP50 **0.000 in all three conditions**, 16–18 of 19 classes
  at exactly 0.0 AP. The only class with non-trivial AP is `person`, so every delta table
  reduces to one class and the reported Spearman correlations run over a vector that is
  18/19 zeros.
* The real recipe is **183,434 updates**; FAST ran 6.5% of it.
* **Therefore it is not used as the downstream forgetting anchor.** B1 runs on the converged
  natural anchor (previous-19 mAP50 = 73.65).
* Controlled long-tail **remains planned as a later confirmation experiment.** Note for
  then: natural S-OWODB t1 already has ρ = 202.8, more imbalanced than LT100 at ρ = 100.
* **No GPU is spent on this during the seven-day sprint.**

---

## What this week does not claim

* No significance for any GPU result — one seed.
* No comparability with published PROB/OWOD numbers — the evaluation split is reduced.
* No band-level tail claim without per-class trajectories behind it.
* No detector-level tail U-Recall-versus-oracle-cost curve for the selection arms unless the
  contingency run happens.
* No causal attribution to long-tailedness from natural class-frequency correlation alone.
