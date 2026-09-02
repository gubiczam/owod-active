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
4. open-pool unknown-class kNN (**primary gate**) and open-pool unknown semantic NMI (**descriptive only** -- see the correction below)
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

### Correction to this protocol, made before any DINOv2 result existed

This section is a record, not a rewrite. The criterion below was **wrong when
first written**, the error was found before any DINOv2 feature was extracted, and
it is corrected here with the reasoning intact. Nothing about the mistake is
hidden, because the value of a pre-registration is precisely that its history is
visible.

**What was wrong.** The first version of this protocol put the 0.15 threshold on
**open-pool NMI**. The frozen decoder-layer protocol
(`docs/decoder_layer_protocol_2026-09-02.md` §4) put it on **open-pool unknown-class
kNN agreement**. Those are not interchangeable, and the NMI form is not a usable
gate.

**The calibration evidence, obtained without any DINOv2 feature.** The audit was
dry-run on a **fixed-seed random unit-norm matrix** of the same shape as the
planned export — pure noise, carrying no information at all:

| representation / P2 | unknown kNN | open-pool kNN | **open-pool NMI** | unknown/bg AUC |
|---|---:|---:|---:|---:|
| random noise, `unit` | 0.0642 | 0.0038 | **0.3022** | 0.4961 |
| random noise, `whitened32` | 0.0652 | — | **0.3356** | 0.5024 |
| PROB `hs[5]`, `unit` | 0.1592 | 0.0665 | 0.3415 | 0.7716 |

**Noise scores open-pool NMI 0.30–0.34, twice the proposed 0.15 threshold.** NMI
between a 120-cluster k-means partition and 58 true classes is inflated by cluster
count and does not approach zero for an uninformative space. A criterion that pure
noise satisfies cannot discriminate anything. The kNN form does: noise reaches
0.0038 where PROB reaches 0.0665.

**The correction.** The 0.15 threshold belongs to **open-pool kNN**, restoring the
gate exactly as the decoder-layer rescue froze it. **NMI is retained as a
descriptive secondary metric with no threshold**, and no NMI threshold is derived
or tuned from the noise observation — the observation explains why NMI is not a
gate, and that is all it is used for. The random noise floor continues to be
measured and printed beside every metric, as calibration in the same spirit as the
"chance kNN agreement = 0.0564" line the decoder-layer audit already prints.

**Primary representation, also corrected.** The first version made the verdict on
the as-exported `unit` representation. It is now **`whitened32`**, because the
frozen thresholds were defined on `whitened32`/P2: deciding on `unit` would change
the preprocessing *and* the backbone at once, and the thresholds would no longer be
comparable to the baseline they were set against. The whitening reuses
`tools.audit_decoder_layers.represent`, the decoder-layer audit's own
implementation and fitting semantics — not a reimplementation. The as-exported
`unit` representation remains a **reported secondary diagnostic**.

## 8. Frozen GO / NO-GO — corrected, pre-result

**Primary population: P2** (admissible + per-image NMS IoU 0.60).
**Primary representation: `whitened32`.** Seeds 0, 1, 2.

A semantic **PASS** requires all four, restoring the decoder-layer rescue's own
decision logic:

```
1  unknown-class kNN                >= 0.30
2  open-pool unknown-class kNN      >= 0.15
3  unknown-vs-background ROC AUC    >= 0.76
4  unknown-class kNN - 0.1772       >= 0.05   for EVERY evaluated seed
```

Where the numbers come from, so none of them is new:

* **1 and 2** are the decoder-layer protocol §4 thresholds verbatim.
* **3** is that protocol's safeguard `AUC >= 0.95 x layer-5 AUC` instantiated at
  the measured layer-5 value of 0.8000, i.e. 0.7600. A representation that gains
  semantics by losing object/background discrimination does not pass.
* **4** is that protocol's "substantial, not a rounding artefact" margin against
  the measured PROB `whitened32`/P2 baseline of **0.1772**, required on **all**
  seeds rather than on the mean. It is technically weaker than criterion 1 once
  0.30 is reached; it is retained explicitly because it belonged to the frozen
  decision logic and dropping it would be a silent change.

**Reported, with no threshold:** open-pool NMI (descriptive only, for the reason
in §7), the `unit` representation on all populations, known-class kNN,
unknown-tail kNN, PC1 variance, dims for 90% variance, P0 and P1, the PROB `hs[5]`
baseline recomputed in the same run, and the random noise floor.

No threshold is added, moved, or derived after a DINOv2 result is seen. The CLI
ends with exactly one of:

```
METHOD_V2_REPRESENTATION_PASS
METHOD_V2_REPRESENTATION_FAIL
```

with the four primary criteria printed immediately above it.

## 9. Out of scope at this stage

Not implemented, not tuned, not run: Method V2's R and C terms · λ and γ · any
change to U · active-selection endpoint experiments · the α replay sweep · any
inspection of DINOv2 oracle endpoints outside the predeclared audit above.
