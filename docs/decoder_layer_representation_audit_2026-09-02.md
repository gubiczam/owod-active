# Decoder-layer representation audit — results

Protocol frozen before anything ran: `docs/decoder_layer_protocol_2026-09-02.md`.
Tables: `data/results/decoder_layer_{representation,population}.csv`.
Plots: `data/results/decoder_layer_{semantic,nuisance,objectness,decision}.png`.

**Status: the layer-5 baseline is measured and committed. Layers 0–4 need one
~35-minute Colab pass (`notebooks/decoder_layer_audit.ipynb`) that this session
cannot run.** The audit code, the decision rule, and the export gate are all in
place and were exercised end-to-end on the committed pool.

---

## 1. Architecture, from source

Read from `github.com/orrzohar/PROB@main`. See the protocol document §1 for the full
table; the three facts that matter:

* **6 decoder layers**, `hs` = `[6, B, 100, 256]`, and `hs[l]` is **post-LayerNorm** —
  `DeformableTransformerDecoderLayer.forward` ends `tgt = self.norm3(tgt)` and the decoder
  appends that value unchanged.
* **Class, box and objectness heads are applied at every layer**, with auxiliary
  supervision: `for lvl in range(hs.shape[0]): ... self.prob_obj_head[lvl](hs[lvl])`.
  Our bridge exports `hs[5]`.
* **`pred_obj = ‖BatchNorm1d(hs[lvl], affine=False)‖²`** — PROB's objectness is the
  squared norm of the batch-normalised hidden state, i.e. a Mahalanobis distance to the
  mean of the query-embedding distribution.

That last point is why this experiment's prior is **weaker than the rescue report
implied**, and the protocol says so before any result: PROB's objectness loss drives
`‖BN(h)‖² → 0` for known objects, which is a **class-agnostic collapse toward one point in
exactly the space we need class structure in**, and it is applied at all six layers. There
is no layer where the objectness objective is absent. What remains open is whether the
*balance* against the per-layer classification loss shifts with depth — `obj_loss_coef` is
only 1e-3, and early layers have undergone fewer refinement steps.

It also means the Day-1 observation that PC1 correlates −0.62 with `pred_obj` and −0.72
with the embedding norm is an **identity, not a coincidence**.

---

## 2. Layer 5 baseline — the bar every other layer must clear

`whitened32`, mean of 3 seeds. Distinct-object counts; kNN with same-object neighbours
excluded; chance kNN agreement among unknowns = 0.0564.

| population | n | background | known kNN | **unknown kNN** | unknown-tail kNN | **open-pool kNN** | NMI | ARI | AUC unk/bg | AUC unk/known |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 raw | 80,000 | 81.4% | 0.9023 | 0.1692 | **0.3111** | 0.0533 | 0.3346 | 0.0363 | 0.8748 | 0.9300 |
| P1 admissible (top 30%) | 24,000 | 65.2% | 0.8747 | 0.1750 | 0.3236 | 0.0653 | 0.3673 | 0.0383 | 0.8142 | 0.8940 |
| **P2 admissible + NMS** | 15,518 | 76.7% | 0.8401 | **0.1772** | 0.2784 | **0.0714** | **0.4061** | 0.0403 | 0.8000 | 0.8981 |

**The decision baseline is `unknown_knn = 0.1772`, `open_pool = 0.0714`,
`AUC unknown/background = 0.8000`** on `whitened32` / P2. A passing layer needs
`unknown_knn ≥ 0.30`, `open_pool ≥ 0.15`, a margin of `≥ 0.05` over 0.1772 on **all three
seeds**, and `AUC ≥ 0.95 × 0.8000 = 0.7600`.

### NMS belongs before every semantic diagnostic — confirmed

Your expectation was right, and it is monotone across all three populations:

| metric | P0 | P1 | **P2** |
|---|---:|---:|---:|
| unknown kNN | 0.1692 | 0.1750 | **0.1772** |
| open-pool kNN | 0.0533 | 0.0653 | **0.0714** |
| unknown NMI | 0.3346 | 0.3673 | **0.4061** |

So P2 is the decision population, and P0 is retained as the control showing what
duplication and background domination cost. Two caveats stated rather than buried:

* **P2's background *share* is higher than P1's** (76.7% vs 65.2%). NMS suppresses
  overlapping *object* boxes far more than background boxes, which do not overlap as much.
  So NMS improves the semantic metrics while making the population *more* background —
  the gain is from removing duplicates, not from removing background.
* **`unknown_tail_knn` falls at P2** (0.2784 vs 0.3236). That is a subset-size artefact,
  not a loss of structure: P2 holds fewer tail objects, so fewer same-class neighbours
  exist to be found at fixed k. kNN agreement at fixed k is **not comparable across
  populations of different size**; NMI is, which is why NMI is the cross-population
  reading.

### Whitening reduces the variance dominance but does not remove the objectness axis

Layer 5, nuisance structure per representation:

| representation | PC1 variance | dims for 90% | ρ(PC1, norm) | ρ(PC1, pred_obj) | η² query | η² known class | η² unknown class |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.6934 | 4 | −0.480 | −0.493 | 0.1964 | 0.0697 | 0.0932 |
| unit | 0.5968 | 7 | −0.717 | −0.621 | 0.1770 | 0.0701 | 0.1014 |
| **whitened32** | **0.1146** | **28** | −0.659 | −0.531 | **0.1340** | **0.1231** | **0.1179** |

Whitening does the work it was chosen for — PC1 falls 0.597 → 0.115, effective
dimensionality rises 7 → 28, class signal overtakes query nuisance in two of three
comparisons — but **ρ(PC1, pred_obj) stays −0.53**: the leading direction is still the
objectness axis, only less dominant. And **η² for query index (0.1340) remains the largest
single structure**, above known class (0.1231) and unknown class (0.1179). Query identity
is not an artefact of scaling.

### Novelty has the wrong sign at layer 5, in every population

Mean distance to the nearest detector-predicted known prototype:

| population | background | known | unknown | unknown tail | sign correct? |
|---|---:|---:|---:|---:|:--:|
| P0 raw | **0.7924** | 0.6127 | 0.7068 | 0.6909 | **no** |
| P2 admissible + NMS | **0.8152** | 0.6342 | 0.7530 | 0.7425 | **no** |

The concept requires real unknowns to be *farther* from known structure than background is.
They are **closer**, in both populations. This reproduces §L of the rescue report under the
frozen protocol, and it is the single number that most cleanly says why
novelty-against-known ranks junk first.

### Duplication at layer 5 (whitened32, P2)

same object **0.4882** · same class, different object **0.2455** · different class
**0.1059**. Same-object similarity is still **2.0×** same-class *after* NMS, which is why
oracle-free deduplication is necessary but not sufficient, and why same-object neighbours
are excluded from every kNN number here.

---

## 3. What is pending, and what it costs

`notebooks/decoder_layer_audit.ipynb` — one config cell, then Run all.

| stage | expected |
|---|---|
| mount, clone, pin OWL + PROB | 2–4 min |
| pip install, MSDA kernel build | 5–12 min (fallback is fine for one pass) |
| materialise the pool's 1,600 images | 2–5 min |
| **export `hs[0..5]`** | **10–25 min** (~8 min with the CUDA kernel, ~20 without) |
| audit, 6 layers × 3 reps × 3 pops × 3 seeds | ~2 min (measured 0.3 min for one layer) |
| plots, persist to Drive | 1–3 min |
| **total** | **~22–50 min, typically ~35** |

Inference only. No training. Peak host RAM ~1.2 GB. Export size 245 MB float16.

**Verified locally before handing it over:** every code cell compiles; 276 tests pass
including 14 new ones for the export contract; `ruff` clean on all new code;
`tools/dry_run_notebook.py` still passes, so the replay notebook is unaffected; the audit
ran end-to-end on the committed pool; the split-name guard accepts `owl_layer_test` and
refuses `owl_layer_eval` / `owl_layer_val` / `pool_train`.

**The export cannot silently mislead.** `hs[5]` must reproduce the pool's committed
`embeddings` at mean cosine ≥ 0.999 or the run stops. That one assertion covers the
checkpoint, the reconstructed model arguments, the hooks, the image ordering and the key
join. Model arguments are not guessed either: `with_box_refine` and `two_stage` are absent
from the published t1 config, so the exporter tries the plausible combinations and keeps
the one whose `state_dict` loads **strictly**.

---

## 4. Results — layers 0 to 4

*Pending the Colab pass. This section will be filled from
`data/results/decoder_layer_population.csv`, and the verdict is computed by
`tools/audit_decoder_layers.decide()` from the frozen rule rather than written by hand.*

---

## 5. The NMS finding stands regardless

Independent of this test's outcome, on the frozen primary endpoint (3 seeds, budget 600,
distinct objects):

| arm | distinct unknown objects | distinct tail objects | proposals/object |
|---|---:|---:|---:|
| `A(x) = objectness·√area` | 155.0 | 47.0 | 1.032 |
| **`A(x)` + per-image NMS** | **168.0** | **53.0** | **1.000** |

**+8.4% unknown objects, +12.8% tail objects, identical at every IoU from 0.3 to 0.7**,
deterministic. This is the object-level insight from the 2026-08-25 consultation, made
oracle-free, and it is the strongest positive result Contribution A currently has.
