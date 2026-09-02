"""The per-layer representation audit. Layer is the only independent variable.

Protocol frozen in ``docs/decoder_layer_protocol_2026-09-02.md`` before the export
ran. Nothing here is chosen per layer: the same three representations, the same
three populations, the same PCA dimension, objectness share, NMS IoU, k, K and
seeds are applied to every layer. That is what makes the comparison one variable
apart, and it is why the constants live at module scope as data rather than as
command-line options.

Without an ``--export`` this audits the committed pool as layer 5 alone. That is
not a shortcut: it exercises the whole path on a laptop and produces the baseline
row that every other layer has to beat, before any GPU time is spent.

    python tools/audit_decoder_layers.py                    # layer 5 baseline, CPU
    python tools/audit_decoder_layers.py --export layers.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from owl import clustering, scoring
from owl import decoder_layers as dl
from owl import proposals as proposals_module
from tools.diagnose_population import nms_keep
from tools.diagnose_representation import (
    K_NEIGHBOURS,
    N_KNOWN_AT_T1,
    knn_agreement,
    load,
    pair_similarity,
    separability,
    variance_explained,
)

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"

# ------------------------------------------------- the frozen protocol, as data ---
PCA_DIMENSIONS = 32
PCA_SAMPLE = 20000
OBJECTNESS_SHARE = 0.30
NMS_IOU = 0.6
N_CLUSTERS = 120
SEEDS = (0, 1, 2)
REPRESENTATIONS = ("raw", "unit", "whitened32")
POPULATIONS = ("P0_raw", "P1_admissible", "P2_admissible_nms")

# --------------------------------------------------- the frozen decision rule ---
GO_UNKNOWN_KNN = 0.30
GO_OPEN_POOL_KNN = 0.15
GO_MARGIN_OVER_FINAL = 0.05        # safeguard 1: substantial, not a rounding artefact
GO_AUC_RETENTION = 0.95            # safeguard 2: do not buy semantics with blindness
DECISION_REPRESENTATION = "whitened32"
DECISION_POPULATION = "P2_admissible_nms"


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def represent(raw: np.ndarray, kind: str, *, seed: int) -> np.ndarray:
    """The three fixed views. Refitted per layer, never transferred between layers."""

    if kind == "raw":
        return raw
    unit = _unit(raw)
    if kind == "unit":
        return unit
    if kind == "whitened32":
        sample = np.random.default_rng(seed).choice(
            unit.shape[0], min(PCA_SAMPLE, unit.shape[0]), replace=False
        )
        centre = unit.mean(axis=0)
        basis = np.linalg.svd(unit[sample] - centre, full_matrices=False)[2][:PCA_DIMENSIONS]
        projected = (unit - centre) @ basis.T
        return _unit(projected / np.maximum(projected.std(axis=0, keepdims=True), 1e-6))
    raise ValueError(kind)


def populations(pool: dict, candidates) -> dict[str, np.ndarray]:
    """P0 raw, P1 admissibility-filtered, P2 additionally NMS-deduplicated.

    Built from ``A(x) = objectness * sqrt(area)`` and boxes only -- both detector
    outputs, neither an annotation. Identical for every layer, because the filter
    must not co-vary with the thing being compared.
    """

    admissibility = scoring.admissibility(candidates)
    everything = np.ones(admissibility.size, dtype=bool)
    admitted = clustering.admissible_mask(admissibility, OBJECTNESS_SHARE)

    saved = pool["pred_obj"].copy()
    pool["pred_obj"] = -admissibility          # NMS must rank by the same score
    try:
        deduplicated = nms_keep(pool, admitted, NMS_IOU)
    finally:
        pool["pred_obj"] = saved

    return {"P0_raw": everything, "P1_admissible": admitted,
            "P2_admissible_nms": deduplicated}


def nuisance_row(features: np.ndarray, pool: dict, *, seed: int) -> dict:
    """Table A: what dominates the space, and how much of it is not semantics."""

    kind, class_name = pool["kind"], pool["class_name"]
    known, unknown = kind == "known", kind == "unknown"
    norms = np.linalg.norm(pool["raw"], axis=1)

    centred = features - features.mean(axis=0)
    sample = np.random.default_rng(seed).choice(
        centred.shape[0], min(PCA_SAMPLE, centred.shape[0]), replace=False
    )
    singular = np.linalg.svd(centred[sample], compute_uv=False)
    spectrum = singular ** 2 / max(float((singular ** 2).sum()), 1e-12)
    basis = np.linalg.svd(centred[sample], full_matrices=False)[2]
    pc1 = centred @ basis[0]

    from scipy.stats import spearmanr
    return {
        "pc1_variance": float(spectrum[0]),
        "dims_for_90pct": int(np.searchsorted(np.cumsum(spectrum), 0.90) + 1),
        "rho_pc1_embedding_norm": float(spearmanr(pc1, norms).statistic),
        "rho_pc1_pred_obj": float(spearmanr(pc1, pool["pred_obj"]).statistic),
        "eta2_query_index": variance_explained(pool["query_index"], features),
        "eta2_oracle_kind": variance_explained(kind, features),
        "eta2_known_class": variance_explained(class_name[known], features[known]),
        "eta2_unknown_class": variance_explained(class_name[unknown], features[unknown]),
    }


def semantic_row(features: np.ndarray, pool: dict, mask: np.ndarray, *, seed: int) -> dict:
    """Tables B, C, D on one population: semantics, open-world separation, duplicates."""

    local = features[mask]
    kind = pool["kind"][mask]
    class_name = pool["class_name"][mask]
    object_id = pool["object_id"][mask]
    group = pool["group"][mask]
    known, unknown = kind == "known", kind == "unknown"
    background = kind == "background"
    inner = {"class_name": class_name, "object_id": object_id, "k": K_NEIGHBOURS}

    row: dict[str, object] = {
        "size": int(mask.sum()),
        "background_share": float(background.mean()),
        "unknown_objects": int(np.unique(object_id[unknown][object_id[unknown] >= 0]).size),
    }

    # B -- semantic structure, same-object neighbours always excluded
    row["known_knn"] = knn_agreement(local, subset=known, **inner)["knn_class_agreement"]
    row["unknown_knn"] = knn_agreement(local, subset=unknown, **inner)["knn_class_agreement"]
    row["unknown_tail_knn"] = knn_agreement(
        local, subset=unknown & (group == "tail"), **inner)["knn_class_agreement"]
    row["open_pool_unknown_knn"] = knn_agreement(
        local, subset=unknown, within_subset=False, **inner)["knn_class_agreement"]

    # C -- open-world separation
    row["auc_object_vs_background"] = separability(local, known | unknown, background, seed=seed)
    row["auc_unknown_vs_background"] = separability(local, unknown, background, seed=seed)
    row["auc_unknown_vs_known"] = separability(local, unknown, known, seed=seed)

    # C -- the novelty question: is background farther from known than real unknowns are?
    predicted = pool["posterior"][mask][:, :N_KNOWN_AT_T1].argmax(axis=1)
    is_known_pred = clustering.predicted_known(pool["posterior"][mask], N_KNOWN_AT_T1)
    rows = [local[is_known_pred & (predicted == index)].mean(axis=0)
            for index in range(N_KNOWN_AT_T1)
            if (is_known_pred & (predicted == index)).any()]
    if rows:
        prototypes = _unit(np.asarray(rows, dtype=np.float32))
        distance = 1.0 - (_unit(local) @ prototypes.T).max(axis=1)
        row["novelty_background"] = float(distance[background].mean()) if background.any() else float("nan")
        row["novelty_known"] = float(distance[known].mean()) if known.any() else float("nan")
        row["novelty_unknown"] = float(distance[unknown].mean()) if unknown.any() else float("nan")
        row["novelty_unknown_tail"] = float(
            distance[unknown & (group == "tail")].mean()
        ) if (unknown & (group == "tail")).any() else float("nan")
        # the sign the concept needs: real unknowns farther from known than background
        row["novelty_sign_correct"] = bool(
            row["novelty_unknown"] > row["novelty_background"]
        )
    # D -- duplication
    row |= pair_similarity(local, {
        "kind": kind, "object_id": object_id, "class_name": class_name}, seed=seed)

    # B -- clusterability of the unknown subset
    index = np.flatnonzero(unknown)
    if index.size > N_CLUSTERS * 2:
        clusters = min(N_CLUSTERS, index.size // 4)
        labels = MiniBatchKMeans(n_clusters=clusters, random_state=seed,
                                 n_init=3, batch_size=4096).fit_predict(local[index])
        row["unknown_nmi"] = float(normalized_mutual_info_score(class_name[index], labels))
        row["unknown_ari"] = float(adjusted_rand_score(class_name[index], labels))
    else:
        row["unknown_nmi"] = float("nan")
        row["unknown_ari"] = float("nan")
    return row


def decide(rows: list[dict]) -> dict:
    """Apply the frozen rule. Reports the reason for every layer, not just the winner."""

    relevant = [
        row for row in rows
        if row["representation"] == DECISION_REPRESENTATION
        and row["population"] == DECISION_POPULATION
    ]
    by_layer: dict[int, list[dict]] = {}
    for row in relevant:
        by_layer.setdefault(int(row["layer"]), []).append(row)

    final = by_layer.get(dl.FINAL_LAYER, [])
    if not final:
        return {"verdict": "INDETERMINATE", "reason": "layer 5 baseline missing"}
    final_knn = float(np.mean([r["unknown_knn"] for r in final]))
    final_auc = float(np.mean([r["auc_unknown_vs_background"] for r in final]))

    verdicts = []
    for layer, group in sorted(by_layer.items()):
        if layer == dl.FINAL_LAYER:
            continue
        knn = [float(r["unknown_knn"]) for r in group]
        open_pool = [float(r["open_pool_unknown_knn"]) for r in group]
        auc = float(np.mean([r["auc_unknown_vs_background"] for r in group]))
        checks = {
            "unknown_knn>=0.30": bool(np.mean(knn) >= GO_UNKNOWN_KNN),
            "open_pool>=0.15": bool(np.mean(open_pool) >= GO_OPEN_POOL_KNN),
            "margin>=0.05_all_seeds": bool(all(v - final_knn >= GO_MARGIN_OVER_FINAL for v in knn)),
            "auc_retained>=0.95x": bool(auc >= GO_AUC_RETENTION * final_auc),
        }
        verdicts.append({
            "layer": layer, "unknown_knn": float(np.mean(knn)),
            "open_pool": float(np.mean(open_pool)), "auc": auc,
            "passes": all(checks.values()), **checks,
        })

    winners = [v for v in verdicts if v["passes"]]
    return {
        "verdict": "FULL_GO" if winners else "NO_LAYER_PASSES",
        "final_layer_unknown_knn": final_knn,
        "final_layer_auc_unknown_vs_background": final_auc,
        "layers": verdicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default=None,
                        help="decoder_layers_v1.npz; omit to audit the pool as layer 5")
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--out-prefix", default="decoder_layer")
    arguments = parser.parse_args()

    pool = load()
    payload = np.load(
        Path(__file__).resolve().parent.parent / "data" / "pool" / "sowodb_t1_frozen_pool.npz",
        allow_pickle=True,
    )
    keep = np.asarray(payload["split"], dtype=str) == "pool"
    pool["raw_boxes"] = payload["boxes"][keep].astype(np.float32)
    candidates = proposals_module.from_frozen_pool(split="pool")

    if arguments.export:
        export = dl.read(arguments.export)
        similarity = dl.validate(export, candidates.embeddings)
        print(f"[gate] hs[5] reproduces the pool at mean cosine {similarity:.6f}  PASS")
        layers = {index: export.layer(index) for index in export.layer_indices}
        print(f"[audit] {len(layers)} layers from {arguments.export}")
    else:
        layers = {dl.FINAL_LAYER: pool["raw"]}
        print("[audit] no export given: auditing the committed pool as layer 5 only.\n"
              "        This is the baseline every other layer must beat.")

    masks = populations(pool, candidates)
    for name, mask in masks.items():
        print(f"  {name:20s} n={int(mask.sum()):6,d}  background={pool['kind'][mask].tolist().count('background') / max(int(mask.sum()), 1):.3f}")

    nuisance: list[dict] = []
    semantic: list[dict] = []
    started = time.time()
    total = len(layers) * len(REPRESENTATIONS) * (1 + len(masks)) * arguments.seeds
    done = 0

    for layer in sorted(layers):
        raw = layers[layer]
        for representation in REPRESENTATIONS:
            for seed in range(arguments.seeds):
                features = represent(raw, representation, seed=seed)
                nuisance.append(
                    {"layer": layer, "representation": representation, "seed": seed}
                    | nuisance_row(features, pool, seed=seed)
                )
                done += 1
                for population, mask in masks.items():
                    semantic.append(
                        {"layer": layer, "representation": representation,
                         "population": population, "seed": seed}
                        | semantic_row(features, pool, mask, seed=seed)
                    )
                    done += 1
                    print(f"  [{done:3d}/{total}] layer {layer} {representation:11s} "
                          f"{population:18s} seed {seed}  "
                          f"unk_knn={semantic[-1]['unknown_knn']:.4f} "
                          f"open={semantic[-1]['open_pool_unknown_knn']:.4f} "
                          f"nmi={semantic[-1]['unknown_nmi']:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, rows in ((f"{arguments.out_prefix}_representation.csv", nuisance),
                       (f"{arguments.out_prefix}_population.csv", semantic)):
        path = RESULTS / name
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

    decision = decide(semantic)
    print("\n" + "=" * 78)
    print(f"FROZEN DECISION RULE on {DECISION_REPRESENTATION} / {DECISION_POPULATION}")
    print("=" * 78)
    print(f"layer 5 baseline: unknown_knn={decision.get('final_layer_unknown_knn', float('nan')):.4f} "
          f"auc_unk/bg={decision.get('final_layer_auc_unknown_vs_background', float('nan')):.4f}")
    for entry in decision.get("layers", []):
        flags = " ".join(f"{k}={'Y' if v else 'N'}" for k, v in entry.items()
                         if isinstance(v, bool) and k != "passes")
        print(f"  layer {entry['layer']}: unk_knn={entry['unknown_knn']:.4f} "
              f"open={entry['open_pool']:.4f} auc={entry['auc']:.4f}  "
              f"{'PASS' if entry['passes'] else 'fail'}  [{flags}]")
    print(f"\nVERDICT: {decision['verdict']}")
    print(f"elapsed {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
