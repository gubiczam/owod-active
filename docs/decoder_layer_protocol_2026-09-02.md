# Decoder-layer representation audit — protocol, frozen before the export ran

A representation *validity* test, not method tuning. **Layer is the only independent
variable.** No acquisition hyperparameter is touched, no training happens.

Written and committed **before** any layer was exported or audited.

---

## 1. Architecture trace — from source, not inference

Read from `github.com/orrzohar/PROB@main` (the upstream of `gubiczam/PROB`). The local
clone is unreadable from this session (`EPERM`), so this is a **source** trace of upstream
rather than of the fork's bridge; §6 pins how the export verifies it matches our own data.

| question | answer | source |
|---|---|---|
| decoder layers | **6** (`--dec_layers` default 6) | `main_open_world.py` |
| queries | **100**; pool metadata agrees (`query_index` 0–99) | argparse + pool meta |
| `hs` shape | `[num_layers, batch, num_queries, hidden_dim]` = `[6, B, 100, 256]` | `deformable_detr.py` |
| is `hs[l]` pre- or post-LayerNorm | **post**. `DeformableTransformerDecoderLayer.forward` ends `tgt = self.norm3(tgt)`; the decoder appends that `output` unchanged — no extra norm | `deformable_transformer.py` |
| class head: all layers or final | **all layers.** `for lvl in range(hs.shape[0]): outputs_class = self.class_embed[lvl](hs[lvl])` | `prob_deformable_detr.py` |
| auxiliary heads | **yes**, one prediction level per decoder layer; `num_pred = decoder.num_layers + 1 if two_stage else decoder.num_layers` | `prob_deformable_detr.py` |
| heads shared across layers? | **cloned** per level under `with_box_refine` (`_get_clones`), otherwise the same module referenced `num_pred` times | `prob_deformable_detr.py` |
| where PROB objectness attaches | **every decoder layer**: `outputs_objectness = self.prob_obj_head[lvl](hs[lvl])` inside the same loop. `out['pred_obj'] = outputs_objectness[-1]` — final layer only in the output dict | `prob_deformable_detr.py` |
| what our bridge exports | `hs[5]` (final layer), 256-d, float16; `pred_obj = outputs_objectness[-1]`; `confidence = exp(-pred_obj / T)` | pool metadata |

### The finding that matters most, and it argues *against* my own hypothesis

```python
class ProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim):
        self.flatten = nn.Flatten(0, 1)
        self.objectness_bn = nn.BatchNorm1d(hidden_dim, affine=False)
    def forward(self, x):
        out = self.flatten(x)
        out = self.objectness_bn(out).unflatten(0, x.shape[:2])
        return out.norm(dim=-1) ** 2
```

`pred_obj` is the **squared L2 norm of the batch-normalised decoder hidden state** — a
Mahalanobis distance to the mean of the query-embedding distribution under a diagonal
covariance taken from BatchNorm's running statistics. PROB's objectness loss
(`--obj_loss_coef 1e-3`) drives `‖BN(h)‖² → 0` for known objects.

Three consequences, all of which follow from the code rather than from our measurements:

1. **`pred_obj` is not an auxiliary signal about the embedding; it is a function of the
   embedding.** So the Day-1 measurement that PC1 of the embedding correlates −0.62 with
   `pred_obj` and −0.72 with the raw norm is an *identity*, not a coincidence.
2. **The objectness objective is a class-agnostic collapse toward a single point** in
   exactly the space we need class structure in. The two objectives are in direct conflict.
   This is what Decoupled PROB (**reference [9] of the research plan**) calls the
   "learning conflict between class and objectness predictions", and its remedy — Early
   Termination of Objectness Prediction — is to stop objectness at some decoder layers.
3. **In stock PROB that collapse is applied at all six layers**, because `prob_obj_head`
   sits inside the per-layer loop with auxiliary supervision.

**Therefore I am lowering my prior on FULL GO before running.** The §Q hypothesis was
"earlier layers are less objectness-warped". The code says no layer escapes the objectness
objective. What is still open, and worth the 30 minutes, is whether the *balance* between
the collapse and the per-layer classification loss differs with depth: `obj_loss_coef` is
only 1e-3, early layers have undergone fewer refinement steps, and the residual stream
accumulates the pressure. That is an empirical question. But the a-priori case is weaker
than the one I made in the rescue report, and this paragraph exists so that a negative
result cannot be presented as a surprise.

---

## 2. What is exported

Same detector, same frozen checkpoint, same 1,600-image pool, **inference only**.

All six layers, `hs[0] … hs[5]`, captured by forward hooks on
`transformer.decoder.layers[0..5]` — no modification to PROB, and no modification to the
existing bridge (which this session cannot read). Hooks capture the layer's own return
value, which the trace above establishes is exactly the tensor appended to `intermediate`
and returned as `hs[l]`.

**Identity alignment is exact, not reconstructed.** The committed pool stores
`query_index` per proposal, so a proposal is keyed by `(image_id, query_index)` and the
export selects precisely those 80,000 keys. No re-derivation of the top-50 objectness
ranking is needed, so no chance of a different ordering.

Carried through unchanged from the existing pool: image id, query index, box, `pred_obj`,
`confidence`, class posterior, and every oracle diagnostic field.

**Built-in correctness gate.** `hs[5]` from the export must reproduce the pool's committed
`embeddings` for the same keys, to float16 tolerance. If mean cosine similarity < 0.999 the
export **fails closed** and no audit runs. That single assertion validates the checkpoint,
the arg reconstruction, the hooks, the image ordering, and the key join at once.

---

## 3. Fixed protocol — identical for every layer

Preprocessing is fixed **across layers**; nothing is chosen per layer.

| stage | fixed choice |
|---|---|
| representations | `raw` (no normalisation) · `unit` (L2-normalised) · `whitened32` (PCA to 32 dims fitted on a seed-0 sample of 20,000 rows, per-axis standardised, L2-renormalised) |
| PCA fit | same protocol, same sample size, same seed, **refitted within each layer** (a basis from another layer is meaningless) |
| populations | **P0** raw 80,000 · **P1** `A(x) = objectness·√area` top 30% · **P2** P1 then per-image NMS at IoU 0.6 ordered by `A(x)` |
| kNN | k = 10, **same-object neighbours always excluded** |
| clustering | k-means, K = 120 on the unknown subset, `random_state` = 0, `n_init` = 3 |
| seeds | 0, 1, 2 for anything stochastic |

Oracle labels score representations retrospectively and never enter any filter. Oracle
deduplication appears only as a labelled *ceiling* row and is excluded from the GO/NO-GO
metrics.

**Why NMS comes before the semantic diagnostics from now on.** The unit of annotation is
approximately an object, not a detector query; same-object similarity (0.639) is 4.5×
same-class similarity (0.142), so duplicates are the densest structure in the space and
dominate any density or neighbour statistic computed without them removed. P0 is retained
as the control that shows what that domination costs.

---

## 4. Primary GO / NO-GO rule — frozen

**FULL GO** if at least one layer other than 5 achieves, on **P2** (the predeclared
oracle-free filtered and deduplicated population), in the `whitened32` representation:

* unknown-class kNN agreement **≥ 0.30**, **and**
* open-pool unknown-class kNN agreement **≥ 0.15**.

Plus two safeguards, both required:

1. **Substantial, not a rounding artefact.** The winning layer must exceed layer 5 on the
   same population by **≥ 0.05 absolute** in unknown-class kNN agreement, and do so **on all
   three seeds**.
2. **Not bought by destroying object/background discrimination.** The winning layer's
   unknown-vs-background AUC must be **≥ 0.95 ×** layer 5's on the same population. A layer
   that finds semantics by forgetting what an object is has not helped a detector.

**Continuous values are reported for every layer whether or not the threshold is met**, and
no layer is selected for barely crossing it.

**NARROWED GO / NO-GO on the formula** otherwise, exactly as written in
`docs/method_rescue_2026-09-02.md` §Q.

## 5. If a layer passes — diagnostics before any selector

In this order, and a selector is built only if these become meaningful:

1. **Novelty** — mean distance to nearest known prototype by oracle kind. Requires real
   unknowns to be *farther* than background, which is the reverse of the final layer.
2. **Unknown clustering** — NMI/ARI/purity against true unknown class.
3. **Rarity** — cluster mass against oracle class frequency, computed from **distinct
   objects and distinct source images only, never proposal counts**; plus head/medium/tail
   separability. Reported against the ceiling (§J of the rescue report: ρ ≈ +0.27 with
   perfect labels on this pool), because a rarity estimator cannot beat its estimand.
4. **Coherence** — the conceptual distinction, not global density: does a *small but
   repeatable* semantic group (≥ m distinct objects from ≥ q images, low known occupancy)
   separate from an *isolated anomaly*?

## 6. Provenance recorded with the export

OWL commit · PROB commit · checkpoint SHA-256 · pool SHA-256 · export version
(`decoder_layers_v1`) · layer indices · GPU name · torch/CUDA versions · wall-clock ·
`hs[5]` validation similarity.

## 7. Anti-p-hacking commitments

Same PCA dimension, same objectness share, same NMS IoU, same K, same k, same seeds for
every layer. Layer is the only thing that varies. The NMS finding
(155 → 168 distinct unknown objects, 47 → 53 distinct tail objects) stands independently of
this test's outcome and is reported either way.
