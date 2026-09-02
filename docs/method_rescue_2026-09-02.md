# METHOD RESCUE / ROOT-CAUSE REPORT

You were right to stop me. Day 1 tested one operationalisation and I wrote it up as if
it had tested the idea. This is the root-cause study, run before any redesign.

**Bottom line.** The conceptual hypothesis is **not** uniformly wrong. Of the eight
requirements in your framing, **two are validated and produce a real measured gain**, one
is falsified *at its ceiling* (which is stronger than falsifying our estimator), one is
falsified across **three independent operationalisations**, and the single largest cause is
a **documented architectural property of PROB** — one that your own bibliography already
cites the fix for. There is one cheap GPU test (~30 min) that decides whether contribution
A can be rescued as designed or must be narrowed.

All numbers from `data/results/{representation_audit,population_audit,selector_rescue}.csv`.
Oracle labels are used only to score representations and selections retrospectively.

---

## A. What exactly Day 1 falsified

Narrowly, and only these:

1. **DBSCAN core/noise on the full 80,000-proposal pool** of L2-normalised final-layer PROB
   decoder embeddings, over eps ∈ {0.15…0.45} × min_samples ∈ {5, 20}.
2. **The same, restricted to the objectness-admissible subpool** (10/20/30/50%).
3. **`D_known` as cosine distance to detector-predicted known-class prototypes**, in that
   same representation.
4. **A within-batch similarity penalty over individual proposal embeddings** (μ = 0.3).
5. **Counting proposals where objects were meant** — a measurement defect, not a hypothesis.

## B. What Day 1 did **not** falsify

Everything else, including the parts that turn out to work:

* that **unknown categories form recoverable groups** — they do (§D);
* that **known structure is explicitly representable** — it is, very well (§D);
* that the **unit of analysis should be the object** — validated, and it is the one change
  that measurably pays (§E, §N);
* that **background must be rejected before semantic reasoning** — validated as necessary,
  though objectness alone is too weak to finish the job (§F);
* that **rarity and coherence** are unachievable *in principle* — what is shown is that they
  are unachievable **in this representation and at this pool size** (§J, §K).

Day 1 also did **not** test the one hypothesis on your list that matters most, because a
sandbox restriction blocked reading PROB's source: **K, wrong source feature.** §C answers
it empirically instead.

---

## C. Exact PROB feature currently used

**I could not read PROB's source.** `/Users/gubiczam/Documents/PROB/daowod_prob_bridge.py`
returns `EPERM` — this session cannot read outside the repository. So the trace below is
**empirical and from the export's own metadata**, not from the code. Flagging that plainly
rather than presenting inference as inspection.

From `data/pool/sowodb_t1_frozen_pool.npz` metadata:

| | |
|---|---|
| declared as | `"PROB decoder output, 256-d, stored float16"` |
| shape | `(120000, 256)`, float16 |
| detector | PROB (Deformable-DETR, `dino_resnet50`), `exps/SOWODB/PROB/t1.pth` |
| per-proposal siblings | `query_index` (int16, 0–99), `posterior_q` (81 uint8), `boxes`, `pred_obj`, `confidence` |
| normalised? | **not in the file** — `owl.proposals._unit()` L2-normalises **on load** |

So: **final decoder layer hidden state, 256-d, query-specific, pre-head** — the tensor that
feeds *both* the class head and the box head, and in PROB also the objectness model. It is
not a classification-head output and not an objectness-head output; it is the shared decoder
state upstream of all of them. Alternatives already in the file: the **81-d class posterior**
(classification-oriented) and `pred_obj`/`confidence` (objectness-oriented). Alternatives
**not** in the file and needing a GPU pass: **any earlier decoder layer**, and backbone/ROI
features.

### What the feature actually encodes — measured

| quantity | value | reading |
|---|---:|---|
| PCA variance in PC1 | **0.597** | one direction owns 60% of the space |
| dims for 90% variance | **7** of 256 | effective dimensionality is ~7, not 256 |
| ρ(PC1, raw embedding norm) | **−0.717** | PC1 *is* the magnitude axis |
| ρ(PC1, `pred_obj`) | **−0.621** | and the magnitude axis is objectness |
| ρ(raw norm, `pred_obj`) | **+0.690** | confirmed directly |
| raw norm sd, known objects | **0.280** | knowns sit at near-constant magnitude |
| raw norm sd, background | **4.870** | background sprays across it |
| η² by `query_index` (100 slots) | **0.1770** | ← the dominant grouping |
| η² by oracle kind (3) | 0.0469 | |
| η² by known class (19), knowns only | 0.0701 | |
| η² by unknown class (58), unknowns only | 0.1014 | |

**The two facts that explain every Day-1 failure:**

1. **The dominant axis is objectness, not semantics.** Any density, Euclidean, or
   nearest-prototype method on this feature is mostly re-deriving `pred_obj` — badly, since
   `pred_obj` is available explicitly and is a better object/background separator
   (AUC 0.776) than the embedding norm (0.606).
2. **The next-strongest structure is *which decoder query fired*, not what is in the box.**
   `query_index` explains 2.5× more variance than known class and 1.7× more than unknown
   class. Clustering this feature substantially clusters **query slots**.

### The literature says this is expected, and names the fix

This failure mode is documented, not unusual:

* DETR query embeddings **couple location and category** for localisation and classification
  simultaneously — the coupling is a recognised obstacle for open-world detection.
  ([OWOD survey, arXiv:2410.11301](https://arxiv.org/html/2410.11301))
* **PROB parameterises objectness as a class-agnostic multivariate Gaussian in the query
  embedding space** — so PROB *deliberately* shapes this exact tensor around objectness.
  ([PROB, arXiv:2212.01424](https://arxiv.org/pdf/2212.01424))
* **Decoupled PROB — reference [9] in your own research plan** — exists to fix "learning
  conflicts between class and objectness predictions in PROB", via **Early Termination of
  Objectness Prediction (ETOP)**, which stops objectness prediction **at particular decoder
  layers**. ([arXiv:2507.13085](https://arxiv.org/abs/2507.13085))
* EW-DETR's Query-Norm Objectness Adapter **decouples direction from magnitude: "feature
  direction encodes class semantics, while feature norm acts as a soft, class-agnostic
  objectness cue."** ([arXiv:2602.20985](https://arxiv.org/pdf/2602.20985))
* On the acquisition side, the established remedy for redundant proposals is NMS-style
  **"candidate fragmentation suppression to avoid pseudo-diversity from redundant
  proposals"**. ([Non-Redundant Informative Sampling, arXiv:2307.08414](https://arxiv.org/abs/2307.08414);
  [Entropy-Based AL with Progressive Diversity Constraint, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Entropy-Based_Active_Learning_for_Object_Detection_With_Progressive_Diversity_Constraint_CVPR_2022_paper.pdf))

EW-DETR's sentence is our measurement restated: norm = objectness, direction = semantics.
And ETOP being **layer-specific** implies earlier decoder layers are less objectness-warped.
**That is the cheap test in §Q.**

---

## D. Is that feature semantically clusterable?

Yes for known classes, weakly but genuinely for unknown classes — and the answer depends
heavily on preprocessing. `data/results/representation_audit.csv`. kNN class agreement,
k=10, **same-object neighbours excluded** (essential: 2.51 proposals sit on the average
object, so without the exclusion "my neighbour shares my class" degenerates into "my
neighbour is me"). Chance = **0.0564**.

| representation | dim | known kNN | unknown kNN | **unknown-tail kNN** | unknown in open pool | unk NMI | unk/known AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `unit_embedding` *(current code)* | 256 | 0.828 | 0.140 | 0.271 | 0.045 | 0.280 | 0.773 |
| `embedding_no_pc1` | 256 | 0.833 | 0.141 | 0.267 | 0.045 | 0.282 | 0.812 |
| **`embedding_whitened32`** | 32 | **0.902** | **0.169** | **0.311** | **0.053** | **0.335** | **0.930** |
| `posterior_hellinger` | 20 | **0.942** | 0.114 | 0.247 | 0.016 | 0.289 | 0.738 |

Four things follow.

1. **Known structure is strongly present** (0.90–0.94). Your requirement 1 — "known
   semantic structure is represented explicitly" — is satisfied.
2. **Unknown class structure exists**: 0.169 against chance 0.0564 = **3.0× chance**.
   Requirement 3 holds, weakly.
3. **Rare unknowns cluster *better* than common ones**: tail 0.311 = **5.5× chance**, nearly
   double the all-unknown figure. That is encouraging and directly relevant — the categories
   the plan cares about are the ones with the most coherent geometry.
4. **But in the open pool it collapses to 0.045–0.053, below within-unknown chance.** An
   unknown proposal's neighbours among all 80,000 are background and knowns. **This is why
   clustering the raw pool cannot find unknown structure — the structure exists and is
   drowned.** Requirement 2 is the binding constraint, not requirement 3.

**Whitening is not cosmetic.** It is the single best preprocessing change measured:
query-slot η² 0.177 → 0.134, known-class η² 0.070 → 0.123, unknown/known AUC 0.773 →
**0.930**, NMI 0.280 → 0.335. The current `unit_embedding` is a poor choice given a
60%-variance dominant axis.

**Division of labour.** The posterior is best for *known* structure (0.942) and worst for
*unknown* structure (0.114, open-pool 0.016) — it is trained to discriminate 19 known
classes, so it maps all unknowns to similar "unknown-ish" vectors. Use the posterior for
known rejection, the whitened embedding for unknown structure. Not one representation.

---

## E. Proposal duplication effect on geometry

Real, large, and it was corrupting both the geometry and the metrics.

* 5,925 GT objects are hit by proposals; **mean 2.51 proposals per object, max 19**;
  1,268 distinct unknown objects behind 2,758 unknown proposals.
* Pair similarities, whitened space: **same object 0.639 · same class different object
  0.142 · different class 0.066**. Same-object similarity is **4.5×** same-class. Duplicates
  form the tightest clusters in the space — they *are* the densest structure.
* In the current `unit_embedding` the three are 0.934 / 0.838 / 0.793 — barely separated,
  which is why raw cosine distances looked uninformative.

**Oracle-free NMS recovers most of what perfect deduplication would give**: unknown-class
NMI 0.386 (NMS) against 0.422 (oracle-perfect dedup), from 0.320 unfiltered. So a
deduplicator is worth having and does not need to be perfect.

---

## F. Background effect on geometry

Background is the primary obstacle, and objectness alone does not remove enough of it.

| population | n | background share | unknown objects | unk NMI in partition | known contamination | unknown recall |
|---|---:|---:|---:|---:|---:|---:|
| raw proposals | 80,000 | **0.814** | 1,268 | 0.282 | 0.119 | 0.452 |
| objectness top 30% | 24,000 | 0.643 | 595 | 0.320 | 0.302 | 0.711 |
| + NMS IoU 0.6 | 16,924 | 0.706 | 529 | 0.386 | 0.273 | **0.820** |
| + known rejection 50% | 8,462 | 0.804 | 420 | **0.454** | 0.193 | 0.734 |
| *[oracle] perfect dedup* | 19,699 | 0.783 | 595 | 0.422 | 0.324 | 0.919 |
| *[oracle] perfect dedup + known reject* | 9,849 | 0.887 | 427 | **0.485** | 0.200 | 0.796 |

**The pipeline you described works on the metric it should improve**: unknown-class NMI
rises **0.282 → 0.454, +61%, entirely with oracle-free steps**, and 0.485 with a perfect
deduplicator. Requirement 3 is not just satisfied — it is *improvable by exactly the
sequence the consultation described.*

**But the population never becomes the one the note imagines.** The Aug-25 note says "e.g.
1000 known and 10000 unknown" — unknowns outnumbering knowns 10:1, background unmentioned.
Reality: even at objectness top 2% the pool is **42% background**, and unknown/known never
exceeds **0.62**. Background share does not fall below 0.64 at any usable operating point,
and *rises* after known rejection. **The imagined population does not exist in this
detector's proposal set**, and that gap — not the clustering algorithm — is the root cause.

---

## G. Best representation found

**PCA-32 whitened, L2-renormalised decoder embedding**, among what is available without a
GPU. Best or tied-best on every structural metric (§D). Cost: one SVD on a 20,000-row
sample, ~1 s.

The genuinely better representation is probably **not in the export at all** — see §Q.

---

## H. Best known-aware clustering formulation

Of your five options, the evidence supports a **hybrid of Option 5 and Option 1**, and
rejects Options 3/4 as objectives:

**Stage 1 — background rejection** by `A(x) = objectness · √area`. Measured, distinct
objects: top 10% of proposals holds **50.8% of all tail objects** and drops background
81.4% → 44.8%; unknown rate 0.0345 → 0.1609 (**4.67×**). This is the strongest single
filter in the whole study.

**Stage 2 — known rejection** by posterior known-mass, mild only (≤25%). AUC for
is-known among objects: posterior **0.705**, whitened prototype 0.591. But see §K — the
gain is threshold-fragile and I do not claim it.

**Stage 3 — semantic clustering** of the residual, whitened space.

**Rejected — Option 4 (k-means/k-means++ with known anchors) *as a budget objective*.**
Cluster-balanced allocation is catastrophic: unknown objects **54.7 ± 9.5** against the
control's 155.0, a 65% loss. The reason is §F: with background still 64–80% of the
population, spreading budget across clusters spends most of it on background clusters,
whereas ranking by `A(x)` concentrates on the object-like tail.

---

## I. Best oracle-free proposal deduplication strategy

**Per-image NMS on `A(x)`, IoU 0.6.** It is the only structural step that produces a robust
gain, and it is **completely insensitive to its one hyperparameter**:

| IoU | proposals kept | unknown objects | tail objects | proposals/object |
|---:|---:|---:|---:|---:|
| control (no NMS) | 80,000 | 155 | 47 | 1.032 |
| 0.3 | 46,863 | **168** | **53** | **1.000** |
| 0.5 | 59,553 | **168** | **53** | **1.000** |
| 0.6 | 63,997 | **168** | **53** | **1.000** |
| 0.7 | 68,561 | **168** | **53** | **1.000** |
| 0.9 | 77,891 | 163 | 50 | 1.012 |

**+8.4% unknown objects and +12.8% tail objects, deterministic, identical across a 0.3–0.7
IoU range**, degrading only at 0.9 where it barely suppresses anything. A flat response
across a wide range is what a real structural effect looks like. It converts duplicate
queries into new discoveries: proposals/object goes to exactly 1.000.

---

## J. Revised definition of rarity — and why I recommend dropping it

**Rarity from estimated group size is falsified at its ceiling.** Not our estimator: the
estimand.

Our cluster-size estimator: ρ(cluster members, true class training frequency) = **+0.004 /
+0.048 / +0.131** across operating points. ρ(cluster distinct source images, frequency) =
+0.032 / +0.048 / +0.057. Essentially zero.

So I measured the **ceiling** — the same count computed with *perfect oracle class labels*,
the best any estimator could do:

| pool | unknown objects | classes present | median objects/class | classes with ≤2 | **ceiling ρ** |
|---|---:|---:|---:|---:|---:|
| full pool (1,600 images) | 1,268 | 58 | 9 | 10 of 58 | **+0.270** |
| at the operating point | 221 | 37 | **3** | **17 of 37** | **+0.213** |

**A perfect rarity estimator on this pool reaches ρ ≈ 0.27.** With median 3 objects per
class and 17 of 37 classes holding ≤2, there is no frequency to estimate.

**And a bigger pool does not fix it** — I tested the obvious rescue:

| candidate images | unknown objects | median/class | ceiling ρ | head/medium/tail median |
|---:|---:|---:|---:|---:|
| 200 | 146 | 2.4 | +0.334 | 5.0 / 1.7 / 2.2 |
| 400 | 321 | 4.4 | +0.399 | 9.7 / 3.2 / 3.5 |
| 800 | 643 | 5.6 | +0.311 | 16.4 / 4.4 / 4.0 |
| 1,600 | 1,268 | 9.0 | **+0.270** | 31.0 / 9.0 / 7.0 |

**Flat-to-declining.** Scaling to 28,800 images would not deliver a usable rarity estimate.
My own proposed rescue is falsified.

The one residual signal is coarse: median objects per class **head 31 / medium 9 / tail 7**.
Head is separable from non-head; **medium and tail are not separable from each other**. Since
the plan's claim is specifically about the tail, a rarity term that cannot tell medium from
tail cannot serve it.

**Recommendation: drop the rarity term.** If it is kept for completeness, it must be
reported with its ceiling attached, and as a head-versus-rest signal only.

---

## K. Revised definition of coherence — falsified three times

Day 1 falsified global DBSCAN. I then implemented your cluster-level reformulation
(≥ m distinct object-like members, from ≥ q images) and it fails the same way:

| coherence rule (at the cleaned operating point) | coherent clusters | real-object rate | incoherent clusters | real-object rate |
|---|---:|---:|---:|---:|
| n_members ≥ median | 41 | **0.0898** | 39 | 0.1043 |
| n_images ≥ median | 41 | **0.0942** | 39 | 0.0996 |
| both ≥ median | 35 | **0.0907** | 45 | 0.1016 |

Coherent is *worse* in all three. **Three independent operationalisations — global
DBSCAN core/noise, admissible-scope DBSCAN, and cluster-level member+image support — all
fail in the same direction.** That is much stronger than one failed configuration.

The mechanism is a monotone ordering that no threshold can reverse (median 20-NN distance):

| population | median 20-NN distance |
|---|---:|
| background | **0.2857** ← densest |
| known objects | 0.3705 |
| unknown, head | 0.3885 |
| unknown, medium | 0.4057 |
| unknown, **tail** | **0.4209** ← sparsest |

**Density is a proxy for "already familiar" and isolation is a proxy for "rare".** In this
representation, "rare real group" and "isolated junk" are not separable by local density,
because rarity and isolation are the *same* signal.

**Recommendation: `C(x) = 1`.** Report as a negative result with the mechanism. The
*intent* — do not buy isolated junk — is served instead by `A(x)`, which is what actually
discriminates junk (unknown rate 4.67× at top 10%).

---

## L. Revised definition of historical novelty

`D_known` as **distance from known** is falsified and the reason is instructive:

| population | mean distance to nearest known prototype |
|---|---:|
| background | **0.3934** ← farthest from what is known |
| known objects | 0.0961 |
| unknown, head | 0.1352 |
| unknown, **tail** | 0.1426 |
| unknown, medium | 0.1574 |

**Real unknown objects sit 2.6× closer to the known manifold than background does.** A fire
hydrant still looks like an object to an object detector; what is genuinely unlike the known
classes is junk. Hence the top decile by `D_known` holds unknowns at **0.08×** the base rate
on the full pool, 0.14× within admissible-30%, 0.37× within admissible-10% — **anti-predictive
at every scope**, against `A(x)`'s **2.74×** on the same set (120 tail objects against 10).

**This is the research plan's own warning, measured**: *"a lonely candidate is
simultaneously uncertain, maximally different, and estimated-rare, so pure diversity- and
rarity-based selection attracts precisely the useless outliers."* The plan diagnosed the
failure correctly and prescribed the wrong cure (density).

**Recommendation.** Novelty survives only in a restricted form: as a **mild rejection
filter** applied *after* `A(x)` admission, not as an additive score. Even then it is
fragile (§O). Its defensible role in the story is the *diagnostic* — it explains why
naive novelty-seeking fails in open-world detection.

---

## M. Revised definition of batch diversity

Falsified as implemented, with a clean mechanism. μ = 0.3 against μ = 0 on the gate-off arm,
three seeds:

| μ | distinct unknown objects | distinct tail objects | proposals/object | mean within-batch similarity |
|---:|---:|---:|---:|---:|
| 0.0 | **76.3** | **35.7** | **2.233** | 0.9026 |
| 0.3 | 63.0 | 32.7 | 2.481 | 0.8823 |

Within-batch similarity fell as intended — the code is correct — but object-level redundancy
**rose** and discovery fell. Mechanism: two boxes on one object have *moderately* different
embeddings, while two boxes on *different objects of the same class* have *very* similar
ones. A similarity penalty therefore suppresses the second instance of a class and leaves
the duplicate in place. **Embedding diversity and object diversity are opposed in this
space.** The literature names exactly this: "pseudo-diversity from redundant proposals",
whose remedy is fragmentation suppression *before* diversity — i.e. NMS, §I.

**Recommendation.** Replace the embedding-similarity penalty with **NMS + a per-cluster
cap**. The cap is the one place semantic structure earns anything:

| arm (3 seeds) | unknown objects | tail objects | tail share | **tail classes** | **tail/image** |
|---|---:|---:|---:|---:|---:|
| `A(x)` control | 155.0 ± 0.0 | 47.0 ± 0.0 | 30.3% | 9.0 | 0.0858 |
| `A(x) + NMS` | 168.0 ± 0.0 | 53.0 ± 0.0 | 31.6% | 9.0 | 0.0883 |
| cluster ALLOCATION K=300 | 54.7 ± 9.5 | 23.7 ± 4.9 | 43.1% | 11.0 | 0.0505 |
| cluster CAP 8/cluster K=300 | 93.0 ± 11.4 | 48.3 ± 4.0 | **52.3%** | **11.3** | **0.1095** |

Capping loses 9% of tail *objects* but gains **+26% tail classes** and **+24% tail objects
per opened image**, at high variance (±11.4). That is a composition-versus-volume trade-off,
reported as such — not a win on the frozen primary endpoint.

---

## N. Revised acquisition algorithm

What the evidence supports, and nothing more:

```
1. A(x) = objectness(x) · sqrt(area(x))          admissibility. Ranks. Learning-free.
2. NMS(IoU 0.6) per image, ordered by A(x)       object-level unit. Oracle-free.
3. take the top `budget` by A(x)
```

Optional, and only if the cost axis is the **opened image** rather than the region:

```
2b. cluster the admitted set in PCA-32 whitened space (K ≈ 300)
3b. rank by A(x) but cap each cluster at ~8 picks
```

Dropped, with measured reasons: `C(x)` (§K), rarity `R(x)` (§J), additive `D_known` (§L),
embedding-similarity batch diversity (§M), cluster-balanced allocation (§H).

`U(x)` (entropy) is untouched and still to be tested inside this structure — it was never
the failing term.

**This is much simpler than the plan's formula, and it is derived rather than assumed.** It
also preserves the plan's actual contribution shape: `A(x)` answers "is this a real object",
NMS answers "is this a *new* object", and the optional cap answers "have we already spent
budget on this semantic region". Your hierarchical framing survives; three of its five
scoring terms do not.

---

## O. Minimal ablation required to test it

Six arms, CPU, 3 seeds, minutes. Primary endpoint unchanged: distinct tail objects per 600
oracle regions. Secondary, reported alongside and never substituted: tail classes, tail per
opened image, proposals/object, images opened.

| # | arm | tests |
|---|---|---|
| 1 | `random` | floor and noise estimate |
| 2 | `A(x)` | the control that must be beaten |
| 3 | `A(x) + NMS` | **does the object-level unit pay** |
| 4 | `A(x) + NMS + U(x)` | does entropy pay once admission is handled |
| 5 | `A(x) + NMS + known-reject 25%` | does mild known rejection pay — expected **fragile** |
| 6 | `A(x) + NMS + cluster cap 8` | composition/volume trade-off on the image axis |

Arm 5's honest prior, pre-registered: 3 seeds give 171.3 ± 1.5 objects and 56.0 ± 1.0 tail
at 25%, against 53.0 at 0% — but **30% gives 51.3, below the no-rejection baseline**. A
non-monotone spike at one threshold is a lucky operating point, not an effect. **I do not
claim it, and arm 5 exists to be reported as negative or fragile.**

Also required, and cheap: rerun the labelling-policy accounting (A2) with
`owl.discovery` so its supervision numbers are in object units too.

---

## P. Expected runtime

| item | cost |
|---|---|
| §O ablation, 6 arms × 3 seeds, CPU | **~15 min** |
| A2 recount in object units, CPU | ~5 min |
| Regenerating every table in this report | ~12 min (3 tools) |
| **§Q decoder-layer test — the only GPU item** | **~30 min T4** |

Total CPU under 40 minutes. The GPU item is one `predict` pass, no training.

---

## Q. GO / NO-GO recommendation for Contribution A

**CONDITIONAL GO**, gated on one ~30-minute GPU test.

**Why not NO-GO.** The idea's requirements 1 and 3 are validated (known structure kNN 0.90;
unknown structure 3.0× chance, tail **5.5×** chance); requirement 6's object-level insight
produces a real, robust, threshold-insensitive gain (+12.8% tail objects); and the pipeline
you described improves unknown-class NMI by **+61%** with oracle-free steps. Requirements 4
and 5 are falsified, but *in this representation* — and the representation is now known to be
the wrong one for the job, for a documented architectural reason.

**Why not unconditional GO.** With the current feature, the entire distribution-aware half
of the method (rarity, coherence, novelty, semantic diversity) contributes nothing, and
rarity is falsified at its ceiling regardless of pool size. Contribution A as written in the
plan cannot be demonstrated on this representation.

### The gate: test an earlier decoder layer

Deformable-DETR's decoder returns all 6 layers (`hs` is `(layers, batch, queries, 256)`);
the bridge exports only the last. **Decoupled PROB's ETOP works by stopping objectness
prediction at particular decoder layers**, which implies the objectness warping is
layer-dependent and that earlier layers are less warped. Our measurement says the final
layer is objectness-dominated (PC1 = 59.7% variance, ρ = −0.72 with the norm).

**Test:** modify the bridge to export `hs[2..5]`, one `predict` pass over the same 1,600
images, then rerun `tools/diagnose_representation.py` per layer. ~30 min T4, no training.

**Decision rule, fixed now:**

* **GO — full contribution A** if any layer reaches **unknown-class kNN agreement ≥ 0.30**
  (≈5× chance, the level the tail already reaches in the final layer) **and** open-pool
  agreement ≥ 0.15. Then rarity and coherence are worth re-testing in that space, and the
  plan's method has a representation that can carry it.
* **GO — narrowed contribution A** otherwise: the contribution becomes *(a)* the measured
  demonstration that a learning-free objectness prior is a strong baseline distribution-aware
  selection must beat, *(b)* object-level deduplication as a real, cheap, reproducible gain,
  and *(c)* a documented root-cause account of why density-, rarity-, and novelty-based
  acquisition fail on a probabilistic-objectness detector — corroborated by Decoupled PROB,
  EW-DETR, and the DETR-coupling literature. That is a genuine, publishable negative result
  with a mechanism, and it is what I would present.
* **NO-GO on the plan's formula either way.** `s(x) = U + λD + γ·w·coh` should not be
  presented as the method. It should be presented as the hypothesis that was tested and the
  measurements that refuted three of its four terms.

**Recommendation on the week.** Run the §Q test and the §O ablation (≈45 min total) before
committing further. Contribution **B (replay)** is unaffected by any of this — it does not
use these features — so the GPU plan for days 3–5 stands unchanged.

---

## Files

| file | what |
|---|---|
| `tools/diagnose_representation.py` | 4 representations × 12 retrospective metrics |
| `tools/diagnose_population.py` | the 6-stage population pipeline, oracle-free and oracle-bounded |
| `tools/diagnose_selector_rescue.py` | 66-row arm comparison with full threshold sensitivity |
| `data/results/representation_audit.csv` | §D |
| `data/results/population_audit.csv` | §E, §F |
| `data/results/selector_rescue.csv` | §I, §M, §N, §O |

**Not done, and stated as such:** PROB source was unreadable (`EPERM`), so §C is empirical
rather than a code trace; UMAP/t-SNE visualisations were not produced because the numeric
diagnostics answered the question and plots are not evidence; earlier decoder layers were not
extracted because that needs the GPU pass in §Q.
