# Day 1 results — coherence, novelty, and a counting defect that reversed a conclusion

Run against the committed PROB t1 pass (80,000 proposals, 1,600 images, 19 known
classes; 65,129 background / 12,113 known / 2,758 unknown proposals, carrying
**1,268 distinct unknown objects** of which **354 are tail-class**). Three seeds.
Every number below comes from a file under `data/results/`. Nothing is hand-typed.

Protocol frozen before any of this ran: `docs/supervisor_week_protocol_2026-09-02.md`.

**Three hypotheses were falsified today, and all three cut against our own
method.** That is the headline. The infrastructure is sound; two of the claims
built on it were not.

---

## 1. H2 — binary density coherence: FALSIFIED, on both scopes

`data/results/coherence_scope.csv` — 84 rows, the grid frozen in the protocol,
not widened afterwards.

**Result: 0 of 28 settings accepted, on all 3 seeds.** Neither scope passes.

| scope | settings | accepted | c1 fails | c2 fails | c3 fails | c4 fails |
|---|---:|---:|---:|---:|---:|---:|
| `full_pool` | 8 | **0** | 21/24 | **24/24** | 15/24 | 6/24 |
| `admissible` (top 30% by `A(x)`) | 8 | **0** | **24/24** | **24/24** | 18/24 | 3/24 |
| `admissible_sensitivity` (10/20/50%) | 12 | **0** | 30/36 | 36/36 | 30/36 | 6/36 |

Criterion 2 — *background must be concentrated in the noise points* — fails in
**84 of 84 rows**. Background is not what DBSCAN calls noise. It is what DBSCAN
calls the core.

At the least destructive admissible setting that still gates anything
(share 0.30, eps 0.45, min_samples 20, noise 13.4%):

| | among coherent | among noise |
|---|---:|---:|
| real unknown objects | 7.4% | **13.0%** |
| background | **68.0%** | 47.1% |

**Real unknowns are 1.75× *enriched* among the points the gate throws away.**

### Why — and this is the part worth showing the supervisor

Median distance to the 20th nearest neighbour, PROB decoder features, 32-dim PCA:

| population | n | median 20-NN distance |
|---|---:|---:|
| background | 65,129 | **0.2857** ← densest |
| known objects | 12,113 | 0.3705 |
| unknown, head classes | 1,260 | 0.3885 |
| unknown, medium classes | 562 | 0.4057 |
| unknown, **tail** classes | 936 | **0.4209** ← sparsest |

Local density is **monotone in exactly the wrong direction**. It orders the pool
background → known → unknown-head → unknown-medium → unknown-tail. So in this
feature space density is a proxy for *already familiar*, and isolation is a proxy
for *rare*. Keeping dense points keeps background first and rare unknowns last.

No `eps`, no `min_samples`, and no scope reverses a monotone relationship. **The
gate is not mistuned; it measures the opposite of what it was meant to measure.**

**My own sub-hypothesis was wrong too, and worse than the original.** I predicted
that background dominating density was the cause and that restricting to the
objectness-admissible subpool would repair it. Background dominating density is
real (68.0% vs 47.1%), but restricting the scope made criterion 1 fail *more*
often, not less: 24/24 on the admissible pool against 21/24 on the full pool.
Admissibility raises unknown density everywhere without changing the sign of the
density–objecthood relationship.

### Decision, by the frozen rule

`C(x) = 1` constant. Binary coherence is **removed from the method** and reported
as a negative result. Per the protocol, tuning stops here. The full-pool-versus-
admissible contrast is reported anyway, because it explains *why* an elegant idea
does not transfer to this feature space rather than merely that it did not.

### What replaces it

The isolated-outlier problem the research plan describes is real — section 3
measures it directly — but density is not its solution. The learning-free
admissibility factor is. `A(x) = objectness · √area`, distinct objects:

| top share by `A(x)` | proposals | unknown objects | of pool | tail objects | of pool tail | unknown rate | background rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 8,000 | 476 | 37.5% | **180** | **50.8%** | 0.1609 (**4.67×**) | 0.448 |
| 20% | 16,000 | 621 | 49.0% | 215 | 60.7% | 0.1064 | 0.564 |
| 30% | 24,000 | 772 | 60.9% | 250 | 70.6% | 0.0819 | 0.652 |
| 50% | 40,000 | 953 | 75.2% | 287 | 81.1% | 0.0569 | 0.753 |
| 100% | 80,000 | 1,268 | 100% | 354 | 100% | 0.0345 | 0.814 |

The top 10% of proposals holds **half the tail objects in the benchmark**, and
drops background from 81.4% to 44.8%. That is what suppresses junk here, and it
costs no training and no clustering.

---

## 2. D_known — definition chosen on estimator properties, and a second falsification

`data/results/novelty_definitions.csv`, three seeds. The decision columns are
properties of the estimator, not endpoint performance.

| definition | compute | drift when labelled set grows 100→600 | rank shift | seed sd | unknown rate, top vs bottom decile |
|---|---:|---:|---:|---:|---:|
| `nearest_labelled` | 0.005 s | **0.0363** | **0.3516** | 0.00000 | **6.41×** |
| `nearest_known_prototype` | 0.012 s | **0.0000** | **0.0000** | 0.00000 | 0.08× |
| `nearest_known_cluster` | 0.004 s | **0.0000** | **0.0000** | 0.00058 | 0.22× |

**Primary definition, recorded now, before the A1.3 ladder runs:**

> `D_known(x) = 1 − max_{z ∈ L} cos(x, z)`, the cosine distance to the nearest
> already-labelled embedding, where `L` is the growing labelled pool.

Chosen on **iterative compatibility**, which is structural rather than a result:
the other two definitions have drift of *exactly zero*. They cannot respond to
the annotator answering, so under them A1.4 — one-shot versus iterative — has no
mechanism to test and is guaranteed to return no difference. Prototype and
cluster novelty are also seed-stable and cheap; that does not rescue an estimator
that is constant in the variable the experiment varies.

### The second falsification: novelty-against-known is a background detector

Mean `D_known` to the nearest detector-predicted known prototype, by oracle kind:

| population | mean distance to nearest known prototype |
|---|---:|
| background | **0.3934** ← farthest from what is known |
| known objects | 0.0961 |
| unknown, head | 0.1352 |
| unknown, **tail** | 0.1426 |
| unknown, medium | 0.1574 |

**Real unknown objects sit close to the known-object manifold — 2.6× closer than
background does.** A fire hydrant still looks like an object to an object
detector. What is genuinely unlike the known classes is not a novel object; it is
junk.

So "distance from what is known", used directly, ranks junk first:

| ranked by | unknown rate in top decile | relative to that scope's base |
|---|---:|---:|
| `D_known` (prototype), full pool | 0.0029 | **0.08×** |
| `D_known` (prototype), within admissible 30% | 0.0117 | **0.14×** |
| `D_known` (prototype), within admissible 10% | 0.0587 | **0.37×** |
| `A(x)`, within admissible 30% *(control)* | 0.2242 | **2.74×** |

It never reaches 1.0× — it is **anti-predictive at every scope**. On the same
admissible-30% set, `A(x)`'s top decile holds **120 distinct tail objects** where
`D_known`'s holds **10**.

This is the research plan's own stated worry, now measured: *"a lonely candidate
is simultaneously uncertain, maximally different, and estimated-rare, so pure
diversity- and rarity-based selection attracts precisely the useless outliers."*
The plan was right about the failure and wrong about the fix. Density does not
separate junk from rare-real (section 1); object-likeness does (section 1, table
2).

A note on the 6.41× for `nearest_labelled`: its reference set here is a random
100-proposal sample, which is ~81% background, so "far from the labelled set"
partly means "unlike the background bulk". That is the opposite reference point
from a known-class prototype, and it is why the two definitions have opposite
sign. Worth stating plainly rather than presenting 6.41× as pure novelty signal.

---

## 3. The counting defect — and it reverses the project's headline claim

`tools/run_experiments.py` counted total discovery as distinct `object_id` values
but counted the head/medium/tail breakdown as **proposals**, and put both in one
table. Two boxes on one fire hydrant are one discovery and one oracle answer.

Inflation is **arm-dependent**, because arms differ in how many near-duplicate
boxes they buy per object. Budget 600, three seeds:

| arm | rounds | tail *proposals* | tail *objects* | inflation | proposals/object | images opened |
|---|---:|---:|---:|---:|---:|---:|
| `objectness` | 1 or 6 | 48.0 | **47.0** | **1.02×** | 1.032 | 548 |
| `prior_consult` | 6 | 49.3 | 34.0 | 1.45× | 1.481 | 349 |
| `prior_consult_batch` | 1 | 53.0 | 37.0 | 1.43× | 1.498 | 345 |
| `prior_consult_batch` | 6 | **61.0** | **34.7** | **1.76×** | 1.713 | 308 |

The comparison was biased ~1.7× in favour of the arm the project was advocating.

**Consequence 1 — the H1 claim as previously stated is falsified.** On distinct
objects the learning-free control wins the tail column **47.0 against 34.7**, and
total discovery **155.0 against 76.0**. The previous reading — "our method finds
61 tail objects against objectness's 48" — was 34.7 real objects counted 1.76
times.

**Consequence 2 — the iterative gain is falsified on this endpoint.** For
`prior_consult_batch`, 600×1 → 6×100 moves tail *proposals* 53.0 → 61.0 (+15%)
but tail *objects* 37.0 → 34.7 (**−6%**) and unknown objects 87.3 → 76.0
(**−13%**). Proposals per object rises 1.498 → 1.713. The iterative "gain" was
entirely duplicate boxes on objects already bought.

**Consequence 3 — B(x|S) fails its predicted sign.** Within-batch diversity was
predicted to lower redundancy. Measured on `a_u_d_r` (gate off, three seeds):

| μ | distinct unknown objects | distinct tail objects | proposals/object | images opened |
|---:|---:|---:|---:|---:|
| 0.0 | **76.3** | **35.7** | **2.233** | 331 |
| 0.3 | 63.0 | 32.7 | 2.481 | 299 |

Mean pairwise cosine similarity inside the batch *did* fall as predicted
(0.9026 → 0.8823), so the implementation works. But box-level redundancy **rose**
and discovery fell. Mechanism: two boxes on one object have moderately different
embeddings, while two boxes on *different objects of the same class* have very
similar ones — so a similarity penalty suppresses new objects of a class already
represented, and leaves the duplicates in place. Embedding diversity and object
diversity are not the same objective in this space.

**Fix, so this cannot recur.** All discovery counting now goes through one tested
implementation, `owl/discovery.py`, where every column names its unit, group
counts are distinct objects, group objects are asserted to sum to the total, and
redundancy is its own reported column instead of being folded into a total.
`tests/test_discovery.py` pins each of those.

---

## 4. One cost-axis question, pre-registered rather than exploited

The two arms do not open the same number of images for the same 600 regions —
548 against 308 — so the frozen region axis and an image-based axis rank them
differently:

| endpoint | `objectness` | `prior_consult_batch` 6×100 | winner |
|---|---:|---:|---|
| distinct tail objects per 600 regions **(frozen primary)** | **47.0** | 34.7 | objectness, +35% |
| distinct tail objects per **opened image** | 0.0858 | **0.1127** | our method, +31% |

**A1's primary endpoint stays the frozen one, and by it H1 is falsified.** The
per-image reversal is recorded here as an *observation*, not as a rescue, and it
is pre-registered as a Day-2 question with its own criterion: *is the unit of
annotation cost the region or the opened image?* That is a question about
annotation practice, and A2's cost accounting is where it is answered. It may not
be substituted for the primary endpoint after the fact.

---

## 5. Revised A1 matrix for Day 2

Unchanged in structure; three changes in content.

* **`C(x)` is constant 1.** The two gated rungs (`a_u_d_rc`, `a_u_d_rc_fullpool`)
  still run — minutes of CPU — so the negative result exists as an endpoint
  number and not only as a diagnostic. They are expected to lose.
* **Every endpoint is a distinct-object count**, via `owl/discovery.py`.
* **`rounds ∈ {1, 6}`** only, and A1.4 is reported cumulatively per round.

The open question Day 2 must answer, given that `A(x)` alone currently wins both
frozen endpoints: **does any semantic term earn its place at all, once
object-likeness is accounted for?** The ladder is built to answer that, including
the answer "no". If the answer is no, the honest contribution A becomes *the
measured demonstration that a learning-free objectness prior is a strong baseline
that distribution-aware selection must beat, and that two of the three published
ways of trying attract background instead* — which is a real result, and it is
better found now than in front of the supervisor.

---

## Files

| file | what |
|---|---|
| `data/results/coherence_scope.csv` | 84 rows, the frozen H2 grid, accept/reject per row |
| `data/results/novelty_definitions.csv` | 9 rows, three D_known definitions × 3 seeds |
| `data/results/batch_diversity_validation.csv` | 6 rows, B(x\|S) at μ ∈ {0, 0.3} |
| `owl/discovery.py` | the single distinct-object counter |
| `owl/clustering.py` | `density_coherence`, `CoherenceGate`, `admissible_mask` |
| `owl/scoring.py` | `admissibility`, `coherence(method='density')`, `combination='admissible'` |
| `owl/selection.py` | `ARMS_V2`, `LADDER_V2` — V1 `ARMS` untouched |
| `tools/diagnose_coherence.py` | the H2 grid and its verdict |
| `tools/diagnose_novelty.py` | D_known selection and B(x\|S) validation |
| `tests/test_coherence.py`, `tests/test_discovery.py` | 16 tests pinning both defects |

Committed V1 results and `ARMS` are unchanged, so every earlier number stays
reproducible and its defect stays explainable.
