# Distribution-aware active annotation — research decision memo

2026-09-06. CPU, read-only. No detector was trained or run. No arm, protocol,
banking rule, replay policy or result was modified. Seed 2 is being prepared and
run separately and nothing here touches it.

Primary specification for this memo: `docs/method.md` (the research plan) and
`docs/konzultacio_2026-08-25_lefedettseg.md` (the supervisor consultation).
Everything numeric below is reproducible with

```bash
python tools/audit_acquisition_ceiling.py --csv data/results/acquisition_ceiling.csv
```

which is read-only and writes nothing into any results directory.

---

## 0. The one-paragraph answer

The D/R/C research direction is **not falsified**, and the reason is precise:
what was tested downstream was a farthest-first *coverage* objective, which is
neither D, nor R, nor C as the plan and the consultation define them. Two of the
three components were tested only as region-level rankings on a different
population and never as an *allocator*; the third was tested downstream in the
one form the plan says it should not take (a multiplicative weight, not a gate).
A cluster-stratified allocator is the closest plan-faithful formulation and its
oracle ceiling is large — 3–5× `random` on tail-class supply. **But a control
with no distribution-awareness at all beats that ceiling**, and it is the finding
that should govern the design: under this benchmark's cost model, the dominant
lever on cumulative tail supply is *how many images you open*, not *which
clusters you spread across*. A distribution-aware method is still worth building.
It must be built against that control, and its endpoint must be per-image, not
absolute.

---

## 1. The gap between the plan and what was tested

`U` = uncertainty · `D` = novelty w.r.t. what is already labelled · `R` = rarity
from the distribution · `C` = coherence gate against isolated outliers.

| plan concept | Aug-25 interpretation | what Stage 2 tested | Proposed-v1 | Proposed-v2 | still untested |
|---|---|---|---|---|---|
| **U** entropy | "entropy is a good signal" | not under test (fixed) | absent | median filter on the gate | — |
| **D** novelty vs labelled | "how new is this compared with already-labelled elements", `diversity_source='labelled'` | `D(x) = 1 − max cos(z, REF-T1)` as a **region ranking**, region-level AUC | farthest-first **coverage** vs a growing reference | **removed** (`reference_scope='trajectory'`) | D as a *term inside a combined score*; D at **image** granularity; D recomputed across rounds in the sequential chain |
| **R** rarity from clustering | "D and R both come out of one clustering"; "cluster known+unknown so as few known leak into unknown" | three region-level scores R1/R2/R3, judged by rank enrichment | absent | absent | **R as a budget allocator** — the form the consultation describes. Never implemented, never run, at any level |
| **C** coherence | "`coh(x) ∈ {0,1}` — a switch, not a weight"; DBSCAN noise | re-specified as *view stability*, `C = min(cos)`; DBSCAN density formulation closed after three failures | absent | absent | **C as a binary gate in a sequential detector run**. V3 ran `A·C`, a weight — explicitly not what the plan asks for |
| **U + D + R·C** | the four-factor score | never combined; the ladder stopped at `U` | no | no | **the combination, in any form** |
| iterative rounds | "pick 100, recompute, pick 100" | not in scope | `ROUNDS_PER_TASK = 1` | `ROUNDS_PER_TASK = 1` | **iterative recomputation inside a task in the sequential benchmark** |

### The six questions, answered directly

1. **Was D ever tested as novelty relative to the accumulated labelled/known
   representation?** *Partly, and not in the form asked for.* Stage 2 measured
   `D(x) = 1 − max cos(z, REF-T1)` against a **fixed, balanced task-1** reference
   and scored it as a region ranking: `D_NO_GO` at unknown-vs-known AUC 0.6411
   against a 0.65 threshold. The **accumulating** reference the consultation
   asked for (`diversity_source='labelled'`, growing after each round) exists in
   `owl/scoring.py` and was measured only in the pre-GPU selection study, where
   it found ~1.5× the plan equation's unknowns. It has **never** been run in the
   sequential detector benchmark. Proposed-v1 used a growing reference but as a
   *coverage objective*, not as a score term.

2. **Was cluster-relative R ever tested?** *No.* R3 — the closest —
   is `log((1+n_cand(c))/(1+n_REF(c)))` used as a **per-region score**, and it
   is not what the consultation describes. Rarity "in the sense of an
   underrepresented but coherent cluster" is an **allocation** statement: it says
   how much budget a cluster receives. No experiment in this repository has ever
   allocated budget across clusters.

3. **Was C ever tested as a binary core-vs-noise gate downstream?** *No.* Two
   distinct things happened and neither is it. The DBSCAN gate was measured on
   **PROB** decoder features and closed for a measured reason: it discards real
   unknowns more often than background (92% vs 60% at eps 0.15) because 81% of
   the pool is background and background sits in the densest region. C was then
   re-specified as view stability, passed its gate (`C_GO`, AUC 0.6101), and was
   run downstream by Method V3 **as a multiplicative weight** `A·C` — which the
   audit then showed selected almost the same regions as `A` alone, because the
   `A` ranking is dense to three parts in 10⁵ at the 600-cut. **A weight on a
   dense ranking is a no-op; a gate is not.** C-as-a-gate, on DINOv2 features,
   inside the object-like set, has never been run.

4. **Was `U + D + R·C` ever combined?** *No.* The Stage-2 ladder stopped at `U`
   and the ladder was never climbed.

5. **Was the representation ever recomputed after an annotation round?**
   *Not in the sequential benchmark.* `ROUNDS_PER_TASK = 1` throughout. The
   pre-GPU study measured that 6×100 helps only arms with something to update
   (`consult` 26 → 36, +38%; `entropy` and `objectness` unchanged at 0%) — which
   is exactly the mechanism argument for iterating a D/R method and exactly why
   iterating entropy would be pointless.

6. **Which negative findings genuinely constrain the new design?**

   **Binding — do not reopen:**
   * DINOv2 must not be the object/background detector (unknown-vs-background
     AUC 0.6835 < 0.76). Semantic structure only *after* an admissibility filter.
   * kNN-density coherence on PROB features is closed: median 20-NN distance runs
     background 0.286 < known 0.371 < unknown-head 0.389 < unknown-tail 0.421,
     and no threshold reverses a monotone ordering.
   * "Cluster size estimates true class frequency" failed at its ceiling
     (ρ ≈ +0.27 *with perfect oracle labels*). Rarity must not be built on the
     claim that cluster size recovers class frequency.
   * A weight on a dense ranking is a no-op (V3's `A·C`).
   * Semantic breadth does not convert to new-class AP: two coverage
     formulations produced the **highest** U-Recall and the **lowest** new-class
     AP of five arms. Any new method must retain an informativeness signal.

   **Not binding — still open:**
   * D as a score *term* rather than a coverage objective; D at image
     granularity; D against an accumulating reference in the sequential chain.
   * R as an **allocator**. Untouched by every negative result above, because
     none of them tested an allocator.
   * C as a **binary gate** on DINOv2 features inside the admissible set.
   * Iterative recomputation within a task.

   **What must not be blurred:** `D_NO_GO` was a verdict on one region-level
   ranking under one reference and one AUC threshold. It is not a verdict on
   diversity.

---

## 2. What the acquisition ceiling actually says

New evidence, computed for this memo from the committed candidate index
(28 800 images, full per-image class counts) under the frozen cost model
`cost(image) = max(1, objects on it)`, 3 000 answers per task, 2 000-image pools.

Every policy but `random` reads oracle labels or oracle costs. **They are
ceilings, not arms** — the best a label-free selector could do with a perfect
representation. `HELD` is the declared-class objects the ledger holds when the
class is declared: bought at this task **plus** banked from an earlier one, which
is the endpoint distribution-aware acquisition actually claims.

| policy | t2 `traffic light` | t3 `fire hydrant` | t4 `stop sign` | images opened |
|---|---:|---:|---:|---:|
| `random` | 64 / 56 | 15 / 28 | 29 / 38 | ~300 |
| **`cheapest*`** | **117 / 99** | **83 / 91** | **130 / 116** | **~900** |
| `stratified*` (perfect clusters) | 60 / 60 | 75 / 65 | 109 / 125 | ~580 |
| `stratified+cheapest*` | 47 / 47 | 66 / 56 | 102 / 102 | ~800 |
| `class-greedy*` (knows the class) | 444 / 512 | 93 / 78 | 95 / 99 | ~300 |
| *measured: `entropy` at t2* | **146** | — | — | — |

*(seed 0 / seed 1)*

Five things follow, and the third is the one that governs the design.

**(a) Acquisition is not saturated.** `class-greedy*` buys **100% of the
declared class present in the pool** within budget, every task. The annotation
budget is not the binding constraint on new-class supply; the selector is.

**(b) The cluster-stratified ceiling is real, and it is a tail effect.**
`stratified*` gives 3–5× `random` at t3 and t4, and *loses slightly* at t2. That
is the plan's own thesis behaving exactly as predicted: equal quota per class
moves budget from head to tail. Two of this chain's three tasks are tail tasks.

**(c) FALSIFIER FIRED — a confound beats the mechanism.** `cheapest*` has no
clustering, no rarity, no embeddings and no distribution-awareness of any kind.
It opens cheap images. It **beats the perfect-cluster stratified ceiling at all
three tasks**. The reason is structural: rare classes live in sparse scenes,
sparse scenes are cheap under a per-object cost model, so "open cheap images"
collects the tail as a side effect — and opening ~900 images instead of ~300 at
identical annotation cost triples incidental tail acquisition and banking. At t4
`cheapest*` (130) even beats `class-greedy*` (95), because volume plus banking
outruns perfect per-task foresight.

**Consequence: a distribution-aware arm that wins without a cost-aware control
proves nothing.** Its win would be attributable to the images it happened to
open being cheap.

**(d) The two levers do not compose naively.** `stratified+cheapest*` is worse
than either parent at every task. Cheapness inside a quota buys more images but
poorer ones; the interaction is real and negative.

**(e) Where distribution-awareness genuinely adds value is per image opened.**
Tail-class objects per image opened, t3/t4: `stratified*` 0.129/0.193 (seed 0)
and 0.108/0.221 (seed 1) against `cheapest*` 0.091/0.146 and 0.100/0.131. Per
task the ratio is noisy (1.08–1.68); **averaged over t3 and t4 it is stable at
1.36 and 1.42**, and against `random` it is 2.19 and 1.54. At t2 (head class)
cheapness wins, as it should. So the primary endpoint must be **held per image
opened**, which cost-awareness cannot inflate, averaged over the two tail
tasks, which is where it is stable.

### Is the cheap-image lever even reachable label-free?

Measured on the 839 images where the frozen S-OWODB pool overlaps the benchmark
index, so treat it as indicative and not as a benchmark-pool measurement:

| predictor | Spearman vs true cost | recall of the truly-cheapest quartile |
|---|---:|---:|
| proposals surviving NMS | **−0.203** | 0.421 (chance 0.25) |
| admissible proposals in `G` | **−0.504** | 0.474 |
| Σ `A` over `G` | **−0.527** | 0.507 |

The sign is **inverted** — images with *more* admissible proposals carry *fewer*
annotated objects — and the signal is real but weak: about **2× chance**. So a
label-free cost-aware arm is implementable, would capture perhaps half the
`cheapest*` effect, and **is a mandatory control** rather than a curiosity. It is
also a partly-uncontrolled difference between the *existing* arms, since
`admissibility` ranks by `A` and `entropy` does not.

### The interaction nobody has costed: banking

The mechanism this design rests on is *buy the tail early and let the ledger
deliver it*. Under `cheapest*` at t4, **87 of 130 held objects are banked**. The
frozen banking rule recovers a label only from a **wholly barren** image, and
`docs/banking_defect_forensics_2026-09-04.md` measures the loss at **68.1%** for
`stop sign` and **68.7%** for `fire hydrant`.

The 2026-09-04 **KEEP V1** recommendation rested on "fixing banking moves ~3
recovered objects to ~10, and ten is still hopeless". **That premise is now
obsolete**: the ceiling is not 10, it is 83–130, and 68% of it is discarded by a
rule we know is wrong. This does not authorise changing anything — the seed-0/1/2
comparison is frozen and must stay frozen — but it is a **protocol-v2 decision
for the supervisor**, and it should be put to them explicitly. Building a method
whose mechanism is early tail purchase, on top of a rule that throws away two
thirds of early tail purchases, is not a fair test of the method.

---

## 3. `distribution_aware_v1` — the recommended method

**Not frozen. Not named Proposed-v3.** Definition first, freeze after §6.

### 3.1 Why not the additive score

The plan's `s(x) = U + λ·D + γ·R·C` is **not** recommended, for three measured
reasons rather than taste:

1. λ and γ are free parameters, and every number chosen after seeing an endpoint
   makes the result tuned. Proposed-v1 was built parameter-free precisely to
   avoid this.
2. Rank-normalising three terms onto [0,1] and adding them makes the *scales*
   commensurable but not the *variances*; on a dense ranking (V3's measured
   three-parts-in-10⁵) the smallest-variance term silently decides nothing and
   the largest silently decides everything.
3. It cannot express the thing the ceiling says matters. Rarity's effect in the
   consultation is **how much budget a region of the space receives** — an
   allocation. A scalar score cannot allocate; it can only reorder.

### 3.2 The method, procedurally

Per task, on that task's own candidate pool, entirely label-free.

```
1. P_nms   per-image NMS at IoU 0.60 ordered by A            [unchanged]
2. G       top 30% of P_nms by A(x) = objectness·√area       [unchanged]
3. Z       frozen DINOv2 ViT-B/14 embeddings of G's crops    [unchanged]
           L2-normalised; the Method-V2 crop, unchanged
4. C(x)    HDBSCAN over Z:  C(x) = 0 if noise, else 1        [the plan's gate]
           clusters k = 1..K over the coherent points
5. R_k     under-representation of cluster k, from the LABELLED reference:
                  R_k = log( (1 + n_cand(k)) / (1 + n_ref(k)) )
           n_ref(k) = reference embeddings whose nearest cluster medoid is k.
           The reference is REF-T1 plus every image this trajectory has bought —
           so R is relative to what is already labelled, and it MOVES.
6. q_k     answer quota, softmax-free and parameter-free:
                  q_k = B · R_k⁺ / Σ_j R_j⁺,     R⁺ = max(R, 0)
           A cluster the labelled set already covers gets nothing.
7. order   within cluster k, descending U (normalised Shannon entropy).
           Emit an interleaved order that walks clusters in descending q_k,
           taking each cluster's next-best-by-U, until each cluster's quota is
           exhausted; then spill the remainder in global U order.
8. spend   hand that order to the existing ledger, unchanged.
```

Steps 1–3 and 8 are the current pipeline verbatim. The method is steps 4–7 and
it produces **a permutation of pool positions** — so it registers as a `ranking`
arm and `owl.active_selection.budget.spend_ranking` consumes it unchanged. **No
change to the task chain, budget, cost model, candidate-pool semantics,
full-image labelling, banking, replay, training, evaluator, checkpoint retention
or result schema.**

### 3.3 Why each piece is there

| piece | plan / Aug-25 provenance | why this form |
|---|---|---|
| `A` gate before embeddings | Stage-1 composition finding | DINOv2 must not be the background detector (AUC 0.6835) |
| `C` = HDBSCAN core-vs-noise, **binary** | "`coh(x) ∈ {0,1}` — a switch, not a weight" | verbatim the consultation. A weight on a dense ranking is a measured no-op |
| one clustering, `R` and `D` from it | "D and R both come out of one clustering" | verbatim the consultation, point 3 |
| `R` from labelled coverage, not cluster size | consultation point 3, corrected | cluster size → class frequency failed at its oracle ceiling (ρ ≈ 0.27); labelled-coverage deficit was never tested as an allocator |
| `R` as **quota**, not score | "as few known as possible leak into unknown" is a partition-quality statement about *allocation* | the ceiling says allocation is where the effect lives |
| `D` implicit in `n_ref(k)` | "how new is this compared with already-labelled elements" | D and R are the same quantity at cluster granularity; separating them would need λ and γ back |
| `U` within cluster | "entropy is a good signal" | keeps the informativeness signal the coverage methods dropped — the measured cause of their failure |
| reference grows | consultation point 7 | makes the score worth recomputing; §5 uses it |

**Why it avoids Proposed-v1/v2's failure modes.** Both maximised *coverage*:
farthest-first leaves a region once covered, so per-class multiplicity is capped
by construction — one or two of everything, never the tens a class needs. That is
the mechanism the supervisor note predicted and both runs confirmed (highest
U-Recall, lowest new-class AP, twice). `distribution_aware_v1` contains **no
traversal**. Within a cluster it takes as many candidates as the quota allows,
ordered by entropy, so multiplicity is bounded by the quota rather than by
coverage — the quantity a rarity rule is supposed to control.

---

## 4. Label-free parameter selection

No parameter is chosen against an oracle endpoint, and none against a detector
endpoint.

| parameter | rule | why it is not a choice |
|---|---|---|
| embedding | frozen DINOv2 ViT-B/14, Method-V2 crop, L2-normalised | already frozen; changing it reopens Stage 1 |
| PCA / whitening | **none** | `whitened32` was the Stage-1 primary but HDBSCAN's own `min_cluster_size` already controls granularity; adding a second reducer adds a dimension count to justify |
| clusterer | **HDBSCAN**, not DBSCAN | chosen *because* it removes `eps`. Selecting eps from a k-distance elbow is a judgement call made on data we would then evaluate |
| `min_cluster_size` | `max(5, ⌈|G| / (B / c̄)⌉)` where `B` = 3 000 answers and `c̄` = the pool's mean image cost | a cluster smaller than one image's worth of budget cannot receive a quota, so this is derived from the budget, not tuned |
| `min_samples` | HDBSCAN default (`= min_cluster_size`) | not touched |
| `R_k` | `log((1+n_cand)/(1+n_ref))`, `R⁺ = max(R,0)` | the Stage-2 R3 form, already frozen in this repository, applied at a new granularity |
| quota | proportional to `R⁺`, no temperature | a temperature is a free parameter |
| `U` | normalised Shannon entropy | unchanged from the baseline arm |

**If the clusterer is still ambiguous** after this, the tie is broken on
**label-free structural criteria only**, declared here in advance and in this
order: (i) noise fraction closest to the 76.7% background share `G` is known to
carry — a gate that keeps everything or nothing is not a gate; (ii) larger mean
cluster size, because the rarity signal is read from cluster *occupancy*;
(iii) higher fraction of the answer budget allocatable, i.e. fewer clusters with
`R⁺ = 0`. Oracle diagnostics are inspected **once**, afterwards, and cannot
reselect the clusterer.

---

## 5. Candidate-side ablations

Cheap (CPU + one cached DINOv2 export), and they separate the components:

| ablation | isolates |
|---|---|
| `U` only | the baseline, already a full detector arm |
| `U + C` | does the coherence gate alone change what is bought? |
| `U + R` (quota, no gate) | is the gate load-bearing, or is the allocator doing all the work? |
| `U + R·C` = `distribution_aware_v1` | the full method |
| **`cost_aware`** | **the mandatory control** — label-free cheap-image preference, ranking images by descending Σ`A` over `G`, no clustering at all |
| `random`, `entropy`, `admissibility` | already measured at seeds 0 and 1 |

`cost_aware` is not optional. §2(c) shows a method could win the primary endpoint
without any distribution-awareness, and this is the arm that detects it.

**Iterative rounds are an ablation, not part of v1.** `ROUNDS_PER_TASK` stays 1
for the frozen comparison. `rounds = 6` for this arm alone is a *separate*
declared experiment, because the reference moves only for this arm and rounds
would therefore change two things at once.

---

## 6. Frozen candidate-side GO / NO-GO

**Written before any oracle diagnostic for this method has been computed.**
Measured on the benchmark's own candidate index at **seeds 0 and 1**, against
`entropy` on the same pools. All five must hold; the first two must hold at both
seeds and the rest at both seeds' mean.

| # | criterion | threshold | why this number |
|---|---|---|---|
| 1 | **different** — Jaccard of opened images vs `entropy` | **< 0.60** | above this the arms cannot produce a distinguishable detector outcome |
| 2 | **tail supply, per image opened** — declared-class objects held at declaration ÷ images opened, **mean over t3 and t4** | **≥ 1.25 × `entropy`** *and* **≥ 1.25 × `cost_aware`** | the per-image form is what cost-awareness cannot inflate (§2e), and the mean over the two tail tasks is the form measured stable (1.36, 1.42) where per-task is not (1.08–1.68). 1.25 sits below the perfect-cluster ceiling, so it is reachable but not free |
| 3 | **not merely cheap** — same quantity against `cost_aware` | already in #2 | the falsifier of §2(c), promoted to a gate |
| 4 | **learnability preserved** — background share of opened images | **≤ `entropy` + 10 pp** | the same tolerance Stage 2 used for R |
| 5 | **no collapse** — distinct unknown classes acquired, and t2 held-per-image | **≥ 0.75 ×** `entropy` on each | a tail gain bought by abandoning the head task is not a win on a mean over three tasks |

**Decision rule, frozen:**

* **any criterion fails → NO-GO.** Record as a negative candidate-side result,
  do not train, do not tune, do not re-choose the clusterer, do not soften a
  threshold. Report which criterion failed and by how much.
* **all pass → GO.** Freeze the method, name it, and run §7.

Criterion 2's threshold, the clusterer, and the ablation set are fixed **now**,
before the diagnostic runs. The oracle is read once, after the run.

---

## 7. If GO: the smallest downstream experiment

**One arm, one seed, the full chain.** `distribution_aware_v1`, seed 0, t2→t4,
everything else identical to the frozen protocol, into
`distribution_aware_v1__seed0`, which no existing trajectory occupies.

**Marked development-seed-informed and not pre-registered**, for the same reason
Proposed-v2 was: seed 0's endpoints informed this design. It goes in
`DEVELOPMENT_SEED_INFORMED`, it does not displace a baseline in `ORDER`, and no
table may present it as pre-registered.

**A kill rule, frozen now**, in the shape the Proposed-v2 rule had: mean
`new_class_AP50 ≥ 3.56` **and** final `known_mAP50 ≥ 44.89`. Below either → STOP,
preserved as a negative result, not tuned, not given seeds 1 and 2.

**`cost_aware` must run too, at seed 0.** Without it a positive result is
uninterpretable — §2(c) is the whole argument. Two arms, not one.

### Cost

| item | GPU | wall |
|---|---:|---:|
| candidate-side diagnostic (§6), cached DINOv2 export, HDBSCAN on ~19 k rows × 3 tasks × 2 seeds | none (CPU) or one T4 hour if the export must be recomputed | ~1 h |
| `distribution_aware_v1` seed 0, t2→t4 | **~2.2 T4-hours** | ~2.2 h |
| `cost_aware` seed 0, t2→t4 | **~2.2 T4-hours** | ~2.2 h |
| **total if GO** | **~4.5 T4-hours** | one Colab session |

Priced from the measured ~2.23 h per baseline arm in the seed-1 session.

**A prerequisite, stated plainly.** The entire measurable difference between the
existing arms lives at t3 and t4, and the training-nondeterminism floor there has
**never been measured**. A 2.2-hour arm evaluated against an unmeasured noise
floor produces a number nobody can interpret. Experiment 2 of
`docs/measurement_experiments_2026-09-05.md` — one fixed acquisition retrained
3–4 times, ~3 GPU hours — should run **before or alongside** this, not after.

---

## 8. What would count as what

| outcome | reading |
|---|---|
| **positive** | GO passed; `distribution_aware_v1` mean `new_class_AP50` exceeds both `entropy` and `cost_aware` at seed 0 **by more than the measured nondeterminism floor**; final `known_mAP50` within the kill rule. Claimable as: *on one development seed, cluster-stratified rarity allocation improved incremental new-class AP over a strong uncertainty baseline and over a cost-matched control.* One seed. A direction. |
| **informative negative** | GO passed — the selector demonstrably bought 1.25× the tail per image — and AP did **not** follow. This is a *strong* result: it separates the acquisition claim from the learning claim and localises the failure in the detector, which is where §7C of the instability forensics already points. It would also make the banking-defect decision (§2) urgent. |
| **inconclusive, and expected if the floor is unmeasured** | any difference at t3/t4 smaller than the nondeterminism floor; or the bimodal ≈0-or-≈20 pattern recurring. Not reportable as a method result at all |
| **negative at the gate** | candidate-side GO fails. Cheapest outcome, and the one that costs no GPU. Record it, and the D/R/C direction has then been tested in its allocator form for the first time |

**Claims permitted if it succeeds** — this list is the ceiling on the write-up:

1. That an oracle-answer budget under full-image labelling equalises annotation
   cost across selectors (already measured: ~6% vs the predecessor's 2.09×).
   Methodological, and independent of any selector.
2. That cluster-stratified rarity allocation increases tail-class supply per
   image opened, on this controlled chain, at these seeds.
3. That the increase is **not** attributable to opening cheaper images, because
   `cost_aware` was run and the endpoint was per-image.
4. If AP follows: that on one development seed it converted to new-class AP.

**Not claimable, whatever happens:** anything about S-OWODB (this is a
one-class-per-task controlled chain and no number here may be compared to a
published S-OWODB result); any statistical effect from one seed; that D/R/C is
"validated"; that coverage-based selection is refuted beyond the two
formulations actually run.

---

## 9. Contribution B — distribution-aware replay

**Do not launch. This is a design note.**

What is already measured, and must be preserved as such:

* the allocator works — `tail_favouring` (α = −0.5) tracks rarity with
  Spearman(quota, frequency) **−0.995** at t2 and −0.42…−0.66 at t3…t6, against
  `uniform`'s ≈ 0; `bear`:`person` quota moves from 14:1 to 1:1;
* object-level materialisation was necessary — image-level storage delivered
  464 objects for `head_favouring` and **1 240** for `tail_favouring` against a
  400 budget, a **2.67×** disparity caused by the allocation rule. Fixed;
  `Σ m_c = |E| = 400` exactly;
* **the downstream α comparison is a negative result and stays one**: across
  three seeds the between-seed spread exceeded the head-vs-tail difference. It is
  a stability/plasticity trade, not a clean win. Nothing here hides that.

**Why α alone was the wrong question.** `m_c ∝ n_c^α` conditions on **one**
variable — the class's own frequency — and the α experiment showed that variable
does not carry enough signal to clear the noise. The four candidates the plan
implicitly offers:

| basis | available now? | argument |
|---|---|---|
| class frequency `n_c` | yes | the α family. Measured insufficient alone |
| **measured forgetting** `Δ AP_c` | **yes** — `per_class_ap.csv` per task, already written | allocates rehearsal to what is *actually* being lost, not to what is *predicted* to be lost. Closes the loop the plan draws |
| class difficulty | partly | confounded with forgetting; no separate estimate exists |
| acquisition scarcity | yes — the ledger records it | double-counts with `n_c` |

**The proposed B experiment, one line:** `m_c ∝ (Δ AP_c)⁺` — allocate the M = 400
exemplar budget in proportion to each previously-known class's **measured AP drop
at the previous task**, with `minimum = 1`, against the frozen `uniform` M = 400
baseline. It is distribution-aware in the plan's sense (allocation follows the
long-tail structure, since tail classes are the ones that drop), it is derived
from the same long-tail principle, and it needs **no new parameter** — no α.

**It is nearly free**: the per-class AP it needs is already written by every
completed trajectory. The cost is one extra arm-equivalent per comparison, and
it should not run until selection is settled. **Not launched.**

---

## 10. Recommendation

> ## IMPLEMENT AND RUN CHEAP DIAGNOSTIC

with the design **revised in two ways** from the sketch that prompted it, both
forced by §2:

1. **`cost_aware` is a mandatory control arm**, candidate-side and downstream.
   Without it, a win by a distribution-aware method is not attributable to
   distribution-awareness — a policy with none at all beats the perfect-cluster
   ceiling on the absolute endpoint.
2. **The primary endpoint is tail supply *per image opened*, not absolute.**
   Absolute held-count is inflatable by opening cheap images; the per-image form
   is where the perfect-cluster ceiling actually wins (1.32–1.41×), and it is
   what criterion 2 gates on.

The diagnostic costs no GPU if the DINOv2 export is cached, it can fail the
method before any training is paid for, and its GO/NO-GO is frozen above. Two
things should be put to the supervisor at the meeting rather than decided here:
the **banking-defect protocol-v2 question** (§2 — the premise of the 2026-09-04
KEEP V1 decision no longer holds), and the **nondeterminism floor** (§7 — without
it, the t3/t4 numbers that carry the entire arm ordering are uninterpretable).
