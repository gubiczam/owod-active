# Method V2 Stage 2 — component diagnostic, frozen before execution

**Written before any Stage-2 oracle diagnostic was computed.** No D, R or C value
on real data has been produced or inspected. This is a pre-registration.

---

## 1. Stage 1 failed. Officially, and it stands.

`docs/method_v2_protocol_2026-09-02.md` §8, on the frozen primary cell
`whitened32` / `P2_admissible_nms`:

| criterion | value | threshold | |
|---|---:|---:|---|
| unknown-class kNN | **0.4387** | ≥ 0.30 | PASS |
| open-pool unknown kNN | **0.0370** | ≥ 0.15 | **FAIL** |
| unknown-vs-background AUC | **0.6835** | ≥ 0.76 | **FAIL** |
| worst-seed margin over PROB 0.1772 | **+0.2581** | ≥ 0.05 | PASS |

### METHOD_V2_REPRESENTATION_FAIL

**That verdict is not revisited, reinterpreted, or recomputed here.** No threshold,
crop, representation, whitening, backbone or population is changed to convert it
into a pass. Stage 2 is a *new, explicitly post-FAIL* experiment with its own
frozen rules.

## 2. Why the work continues anyway

The failure is **localised and diagnosable**, and the two halves of the result
point in opposite directions:

| | DINOv2 | PROB `hs[5]` | reading |
|---|---:|---:|---|
| unknown-class kNN | **0.4387** | 0.1773 | DINOv2 is **2.5× better** at grouping real unknowns by class |
| unknown-tail kNN | **≈0.73–0.74** | 0.2784 | and dramatically better on the classes the plan is about |
| open-pool unknown kNN | 0.0370 | **0.0712** | worse once background is in the neighbour pool |
| unknown-vs-background AUC | 0.6835 | **0.7983** | **worse at telling objects from background** |

So DINOv2 supplies the semantic geometry PROB lacks, and lacks the
object/background discrimination PROB supplies. Both failing criteria are
background-facing; both passing criteria are semantics-facing. Nothing in the
Stage-1 result says the semantic geometry is unusable — it says **DINOv2 must not
be asked to be the background detector.**

The composition this suggests, and which Stage 2 tests the components of:

```
Stage 1   PROB objectness/admissibility + per-image object-level NMS
          -> object/background filtering and duplicate suppression
Stage 2   DINOv2 semantic geometry, operating ONLY inside that object-like set
          -> the semantic components D, R and possibly C
```

The four-factor acquisition idea is preserved:
`s(x) = U(x) + λ·D(x) + γ·R(x)·C(x)`. What changes is that each factor is
computed by the representation that measured well for its job.

## 3. What is fixed, and therefore not under test

| | |
|---|---|
| **P2 is a fixed input population**, not something being re-optimised | admissibility top 30% by `A(x)=objectness·√area`, then per-image NMS IoU 0.60. Must reproduce **n = 15,518**, background ≈ **0.767**, or the run **fails closed**. |
| **DINOv2 embeddings are frozen** | read from the already-exported `dinov2_vitb14_method_v2_v1.npz`. No re-extraction, no new backbone, no new layer, no new crop, no new PCA variant. |
| **whitened32 stays primary** where metric comparability requires it | it is what the Stage-1 and decoder-layer numbers were measured on. `unit` reported as secondary. |
| **Pool** | `split == "pool"` only. `eval` rows are never read. |
| **Not touched** | U, λ, γ, replay, acquisition endpoints, the crop, the NMS threshold, the backbone. |

**No selection or acquisition result is inspected in this stage.** Oracle labels
are used for evaluation only, never inside the construction of D, R or C.

## 4. Component D — semantic novelty

```
D(x) = 1 - max_{r in REF} cos(z_x, z_r)
```

`z` are the frozen DINOv2 embeddings. The candidate's own row is excluded from
`REF` so a proposal cannot be its own reference.

### The labelled reference set, `REF`

**REF-A (primary, no new GPU, oracle-free).** The proposals the *detector itself*
calls a known class — `owl.clustering.predicted_known(posterior, 19)` — taken
within the pool and then **NMS-deduplicated at IoU 0.60 by `A(x)`**, so the
reference is object-level rather than proposal-level and a crowded object cannot
dominate it. This is the repository's established oracle-free stand-in for "what
is already labelled": it is what `clustering.contamination()` already runs on, and
the detector labels its own 19 known classes at 0.83 accuracy.

It mimics the initial labelled state at round 0 without reading a single oracle
label, and it reuses the frozen candidate export — no second representation
pipeline.

**Why not the T1 ground-truth boxes.** They would be legitimate (T1 labels are
*present* at round 0, not future), but they need a separate DINOv2 export over T1
*training* images. REF-B is therefore recorded as an available fallback and is
**not** run in this stage. If it is ever run it must use the identical model,
crop geometry, preprocessing, CLS extraction and L2 normalisation.

### Leakage rules, absolute

* a candidate's oracle class **never** enters D;
* future unknown labels **never** enter `REF`;
* the oracle only *evaluates* whether D ranks unknown and tail objects.

### Diagnostics on fixed P2

D distribution by `oracle_kind` (background / known / unknown) and by unknown
head / medium / tail · ROC AUC unknown-vs-known and unknown-vs-background · rank
enrichment at top **1%, 5%, 10%, 20%, 30%** · distinct unknown objects, distinct
tail objects, unknown classes, tail classes at each fraction.

**The comparison that matters:** D against **`A` alone on exactly the same P2
rows**. If D adds nothing over objectness ordering, it does not belong in the
score.

## 5. Component R — semantic under-representation

**The falsified claim is not reintroduced.** "Cluster size estimates true class
frequency" failed at its *ceiling* under the previous representation
(ρ ≈ +0.27 with perfect oracle labels, flat in pool size) and is not assumed here.

Three definitions, **frozen now**, no grid search, no oracle in construction.
Every parameter reuses a value already frozen in this repository:
`k = K_NEIGHBOURS = 10`, `K = N_CLUSTERS = 120`.

**R1 — candidate semantic density.** Inverse local density among candidates:
`R1(x) = d_k(x)`, the distance to the k-th nearest *other candidate* in P2 DINO
space. Larger = locally sparser. Predeclared as the definition closest to the
falsified isolation signal, and expected to be the weakest.

**R2 — labelled coverage deficit.** A bounded log-ratio of how far the labelled
material is versus how far other candidates are:
```
R2(x) = log( (eps + d_k^REF(x)) / (eps + d_k^cand(x)) ),   eps = 1e-6
```
High when a candidate sits in a region **dense with candidates but far from
anything labelled** — semantically under-covered.

**R3 — semantic partition under-coverage.** One oracle-free k-means partition of
the P2 DINO space (`K = 120`, `random_state = seed`). Each candidate and each
`REF` embedding is assigned to its nearest centroid. For a cluster `c`:
```
R3(x) = log( (1 + n_candidates(c)) / (1 + n_REF(c)) ),   c = cluster(x)
```
High for regions the candidates populate and the labelled set does not.

Sensitivity: one tiny predeclared check only, `k ∈ {10, 20}` for R1/R2 and
`K ∈ {120, 240}` for R3. **k and K are never selected by oracle endpoint
performance.**

### Diagnostics per R

Spearman with true unknown-class frequency and with inverse frequency ·
distributions for unknown head / medium / tail · top-fraction enrichment ·
distinct unknown and tail objects · unknown and tail class coverage.

The question is **not** "does R recover class frequency" — that was answered no.
It is: **does R preferentially surface semantically under-covered unknown
regions, especially medium and tail, without collapsing into background?**

## 6. Component C — consistency, not density

**The density formulation is closed.** DBSCAN core-vs-noise, local kNN density and
isolation penalties are **not** reused as C: three independent operationalisations
failed in the same direction, because median 20-NN distance runs
background 0.286 < known 0.371 < unknown-head 0.389 < unknown-tail 0.421, and no
threshold reverses a monotone ordering.

New hypothesis: **semantic stability under mild deterministic context change.**

Views frozen before computation — exactly two, plus the base:

| view | crop |
|---|---|
| base | the frozen **1.20×** square crop |
| A | same centre, **1.10×** square |
| B | same centre, **1.30×** square |

All three: real pixels only, identical shift-before-shrink geometry, 224×224,
identical DINOv2 preprocessing, identical model. **No colour jitter, no flip, no
stochastic augmentation, no oracle-dependent view choice.**

```
sim_A = cos(z_base, z_A)
sim_B = cos(z_base, z_B)
C(x)  = min(sim_A, sim_B)          # primary
        mean(sim_A, sim_B)         # reported descriptively
```

Exported for **P2 only** (15,518 rows × 2 views = 31,036 crops), cached in Drive
with versioned provenance, never overwriting the base export.

### Diagnostics

C distribution for background / known / unknown and unknown head / medium / tail ·
ROC AUC unknown-vs-background and tail-vs-non-tail-unknown · top and bottom
consistency quantiles · descriptive interaction with D and R.

**Hypothesis:** real semantic objects stay more stable across mild context change
than spurious, fragment or background proposals. **It is not assumed to pass.**

## 7. Distinct-object accounting — mandatory

Every ranking table uses the Day-1 corrected accounting via `owl.discovery`.
Proposal duplicates are **not** object discoveries. Each top-fraction row reports:

proposals · distinct oracle objects · distinct unknown objects · distinct tail
objects · unknown classes · tail classes · proposals-per-object · unique images ·
background share.

## 8. Baselines, on the same fixed P2 rows

`A` = the Stage-1 admissibility ranking · `random` = a seeded shuffle. **No
comparison against historical tables computed under proposal counting.** Everything
is recomputed under distinct-object accounting.

## 9. Frozen GO / NO-GO

This decides only whether D / R / C **deserve a place in a later acquisition
ablation**. It is not a gate on the research continuing.

Predeclared top fractions for the improvement tests: **{5%, 10%, 20%}**.

**D is GO if both:**
1. unknown-vs-known ROC AUC **≥ 0.65**; and
2. at one or more of {5%, 10%, 20%}, D improves **distinct unknown-object count
   OR distinct tail-object count** over the `A`-only ordering on the same fixed P2
   rows by **≥ 10% relative**.

**R is GO if** at least one predeclared R definition shows a **monotonic
head → medium → tail increase in median rarity** *and* improves **medium+tail
class or object coverage** over `A`-only at one or more predeclared top fractions
**without increasing background share by more than 10 percentage points**.

**C is GO if** unknown-vs-background ROC AUC **≥ 0.60**, **or** using C as a
diagnostic filter/weight yields **≥ 10% relative improvement in distinct
unknown-object count** at one or more predeclared top fractions **without reducing
the distinct tail-object count**.

**These criteria are not reinterpreted after results.**

### Resulting permitted ladder

Built in order, stopping at the first component that fails, and every component's
verdict reported separately regardless:

| outcome | ladder |
|---|---|
| D fails | `U` |
| D passes, R and C fail | `U+D` |
| D and R pass, C fails | `U+D+R` |
| all pass | `U+D+R*C` |

If none pass, the negative result is reported and **no replacement semantic term
is invented.**

## 10. Out of scope

Not run, not tuned, not implemented in this stage: acquisition endpoints · λ and
γ · any change to U · replay · alternate backbones, layers, crops or PCA variants ·
any reinterpretation of the Stage-1 verdict.
