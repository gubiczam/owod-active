# Method V2 — semantic representation experiment, frozen protocol

**Provenance note, stated plainly:** this document did not exist in the repository
before now. It was written from the specification given on 2026-09-02, **before any
DINOv2 feature was extracted and before any DINOv2 oracle endpoint was computed or
inspected.** It is a pre-registration, not a record of something already run.

This stage answers **one** question:

> Does frozen DINOv2 ViT-B/14 on proposal crops provide a sufficiently semantic
> representation to justify building Method V2's D / R / C terms?

Nothing else is decided here. No λ, no γ, no R, no C, no U change, no acquisition
endpoint, no replay sweep.

---

## 1. Why this experiment exists

`docs/method_rescue_2026-09-02.md` established, with a source trace, that PROB's
final decoder embedding is objectness-dominated by construction:
`pred_obj = ‖BatchNorm(hs[lvl])‖²`, so the objectness objective is a class-agnostic
collapse toward one point in the very space the method needs class structure in,
applied at all six decoder layers. On that representation the measured layer-5
baseline on the decision population was **unknown-class kNN 0.1772, open-pool
unknown kNN 0.0714, unknown-vs-background AUC 0.8000**.

The question is therefore whether the *representation* is the binding constraint. A
frozen, detector-independent semantic backbone answers that directly.

---

## 2. Frozen crop specification

This is the clarification made before extraction. **No artificial grey square
padding.** Crops come from real image pixels only.

1. pool proposal boxes are normalised `cx, cy, w, h`;
2. construct a **square** box centred on the proposal centre;
3. square side = `1.20 × max(proposal_width, proposal_height)` in pixels — 10%
   context on each side relative to the **larger** proposal dimension;
4. clip/shift the square to the image boundaries, **preserving its size where
   possible** (shift before shrinking);
5. if the requested square exceeds an image dimension, use the **largest valid
   square the image supports**, i.e. `min(image_width, image_height)`;
6. crop the real image only — never pad;
7. resize to `224 × 224` with the preprocessing the frozen DINOv2 model expects
   (bicubic resize, ImageNet mean/std normalisation);
8. use the **final CLS representation** (`forward_features(x)["x_norm_clstoken"]`);
9. **L2-normalise** the embeddings.

Rationale for shift-before-shrink: padding would inject a constant synthetic
region whose area varies with how close a proposal sits to the image edge, which
would put an edge-proximity signal into the embedding. Shifting keeps every crop
made of real pixels at the requested scale; shrinking only happens when the image
itself cannot supply the square.

## 3. Frozen backbone

| | |
|---|---|
| model | **DINOv2 ViT-B/14**, hub id `dinov2_vitb14`, repo `facebookresearch/dinov2` |
| weights | frozen, no OWOD fine-tuning of any kind |
| feature | final normed CLS token, dimension **768** |
| eval | `model.eval()` and `torch.inference_mode()` |

**Not compared:** other DINO sizes, CLIP, other layers, other crop margins, other
preprocessing variants. One backbone, one crop, one representation.

## 4. Population

`data/pool/sowodb_t1_frozen_pool.npz`, **`split == "pool"` only**: 80,000
proposals over 1,600 images, 50 per image. `split == "eval"` (40,000 proposals,
800 images) is never loaded, never fitted on, never included.

Expected exactly 80,000 proposals and 1,600 images. A different count is a failure,
not a variation.

Audit populations, reusing the repository's existing definitions rather than new
ones:

| | definition |
|---|---|
| **P0** | all 80,000 pool proposals |
| **P1** | `A(x) = objectness · √area`, top 30% (`owl.clustering.admissible_mask`) |
| **P2** | P1 then per-image NMS at IoU 0.60 ordered by `A(x)` (`tools.diagnose_population.nms_keep`) |

Historical reference values, for **validation, not as targets**:
P0 n=80,000 background≈0.814 · P1 n=24,000 background≈0.652 · P2 n=15,518
background≈0.767. Material disagreement is to be investigated, never forced.

## 5. Identity and export contract

Every row keys to a proposal by `(image_id, query_index)` and carries its source
pool row index. Alignment reuses `owl.decoder_layers.proposal_keys` / `align`, so
the semantic export and the decoder-layer export agree on what a row is.

Output is a new version and never overwrites a decoder-layer export:
`dinov2_vitb14_method_v2_v1.npz`.

Fails closed on: missing images, duplicate proposal identities, wrong proposal or
image count, non-finite features, zero-norm features, dimension drift, alignment
error, or version mismatch.

Provenance records git SHA, model identifier and runtime source/version info,
source pool SHA-256, image root, the crop specification, proposal and image
counts, feature dimension, and device.

## 6. Preprocessing sanity gate

A deterministic smoke mode exports a small fixed subset and verifies crop geometry
is non-empty, dimensionality is constant, features are finite, post-normalisation
L2 norm ≈ 1, and that repeating inference on the same crop in eval mode is
effectively identical. The full 80k export never runs as part of the test suite.

## 7. Metrics

Oracle labels are used **only** for evaluation, never in extraction or in any
filter. Reported for P0, P1, P2, seeds 0/1/2 where randomness exists, reusing the
decoder-layer audit's own metric functions so the comparison is apples-to-apples:

1. known-class kNN accuracy
2. unknown-class kNN accuracy
3. unknown-tail kNN accuracy
4. open-pool unknown semantic NMI
5. unknown-vs-background ROC AUC
6. PCA PC1 explained variance
7. effective dimensionality — PCs explaining 90% of variance

kNN uses k=10 with **same-object neighbours always excluded**: 2.51 proposals sit
on the average annotated object, so without the exclusion "my neighbour shares my
class" degenerates into "my neighbour is me".

**The PROB layer-5 baseline is recomputed from the same pool file, on the same
populations, with the same metric code, in the same run.** It is not quoted from
the earlier document. That makes the comparison exact rather than approximate, and
costs nothing since the pool already carries `hs[5]`.

### Two ambiguities recorded before any result

Both are flagged here rather than resolved silently, because resolving them after
seeing a number would be exactly the failure this document exists to prevent.

**(a) Which quantity carries the 0.15 threshold.** The specification says
"open-pool semantic **NMI** ≥ 0.15". The frozen decoder-layer protocol
(`docs/decoder_layer_protocol_2026-09-02.md` §4) put the 0.15 threshold on
"open-pool unknown-class **kNN** agreement". These are not interchangeable: at
layer-5 / P2 the open-pool kNN was **0.0714**, so 0.15 is a genuine bar, whereas
unknown NMI on the unknown subset was **0.4061**, so 0.15 would already be cleared
by PROB and the criterion would be vacuous.

**Measured before any DINOv2 feature existed.** Running the audit on a fixed-seed
**random unit-norm** matrix of the same shape — pure noise, no information —
produced **open-pool NMI 0.3022 on P2**, twice the 0.15 threshold, while its
open-pool kNN was **0.0038** against PROB's 0.0665. NMI between a 120-cluster
k-means partition and 58 true classes is inflated by cluster count and does not
approach zero for an uninformative space. **So the NMI ≥ 0.15 criterion is
satisfied by noise and cannot discriminate anything.** The kNN form of the same
threshold is discriminative.

Because of that, the audit now measures the random noise floor alongside every
real representation and prints it beside each metric, in the same spirit as the
"chance kNN agreement = 0.0564" line the decoder-layer audit already prints. That
is calibration, not a new threshold, and it changes no decision rule.

Resolution taken: the verdict applies the threshold **literally as specified**, to
open-pool NMI — NMI computed by clustering the whole population and scoring
unknown class against that partition, which is the sense in which it is
"open-pool". **Open-pool unknown kNN is printed beside it with the layer-5 value,
so the stricter reading is visible and nothing is hidden.** If you intended the
kNN form, say so before running; after a result is seen the choice is no longer
free.

**(b) Which representation is primary.** The crop specification ends at
"L2-normalise", so the as-exported unit-norm CLS embedding is the frozen
representation and the verdict is computed on it. The decoder-layer decision used
`whitened32` (PCA-32, per-axis standardised, renormalised). Both are reported on
identical populations; the whitened row exists for apples-to-apples with the
earlier audit. Declared now: **PASS/FAIL is decided on the as-exported
representation.** If the two disagree, that disagreement is reported explicitly.

## 8. Frozen GO / NO-GO

**Primary decision population: P2.** Thresholds are frozen and must not change:

```
PASS requires   unknown-class kNN        >= 0.30
          AND   open-pool semantic NMI   >= 0.15
```

`unknown-vs-background ROC AUC` is reported as a **safeguard** and compared
descriptively to PROB's P2 value (≈0.80). A representation that gains semantics by
losing object/background discrimination is reported as such.

No threshold is added after seeing the result. The CLI ends with exactly one of:

```
METHOD_V2_REPRESENTATION_PASS
METHOD_V2_REPRESENTATION_FAIL
```

with the exact primary metrics printed immediately above it.

## 9. Out of scope at this stage

Not implemented, not tuned, not run: Method V2's R and C terms · λ and γ · any
change to U · active-selection endpoint experiments · the α replay sweep · any
inspection of DINOv2 oracle endpoints outside the predeclared audit above.
