#!/usr/bin/env python
"""Method V2: is frozen DINOv2 semantic enough to justify building D / R / C?

One question, one decision, thresholds frozen in
``docs/method_v2_protocol_2026-09-02.md`` before any feature was extracted.

Every metric function and every population definition is **imported** from the
decoder-layer audit rather than reimplemented, so the comparison is
apples-to-apples by construction and cannot drift through a subtly different
definition. The PROB layer-5 baseline is recomputed here, in the same run, from
the same pool file, on the same populations -- not quoted from the earlier
document -- which makes the comparison exact and costs nothing, since the pool
already carries ``hs[5]``.

Oracle labels are used only to score representations. Nothing here touches
acquisition, and no R, C, lambda or gamma is chosen.

    python tools/audit_dinov2_representation.py --export dinov2_..._v1.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score

from owl import proposals as proposals_module
from owl import semantic_features as sf
from tools.audit_decoder_layers import (
    N_CLUSTERS,
    populations,
    represent,
)
from tools.diagnose_representation import (
    K_NEIGHBOURS,
    knn_agreement,
    load,
    separability,
)

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"

#: Frozen in the protocol, section 8, and corrected before any DINOv2 feature was
#: extracted. Not to be changed after a result is seen.
#:
#: All four are the decoder-layer rescue's own decision logic, instantiated at its
#: measured layer-5 values -- none of them is new to Method V2:
GO_UNKNOWN_KNN = 0.30              # decoder-layer protocol section 4, verbatim
GO_OPEN_POOL_KNN = 0.15            # ditto -- kNN, not NMI. See protocol section 7.
GO_AUC_UNKNOWN_VS_BACKGROUND = 0.76   # 0.95 x the measured layer-5 AUC of 0.8000
GO_MARGIN_OVER_PROB = 0.05         # "substantial, not a rounding artefact"
PROB_P2_UNKNOWN_KNN = 0.1772       # measured PROB whitened32/P2 baseline

DECISION_POPULATION = "P2_admissible_nms"

#: `whitened32`, because the frozen thresholds were defined on whitened32/P2.
#: Deciding on the as-exported `unit` features would change the preprocessing and
#: the backbone at once and leave the thresholds non-comparable with the baseline
#: they were set against. `unit` is reported as a secondary diagnostic.
DECISION_REPRESENTATION = "whitened32"
REPRESENTATIONS = ("whitened32", "unit")

#: A fixed-seed random unit-norm matrix is measured alongside the real
#: representations, purely as a **noise floor**. It is a calibration reference in
#: the same spirit as the "chance kNN agreement" line the decoder-layer audit
#: prints -- not a threshold, and it changes no decision rule.
#:
#: It earns its place: on a dry run with random features, open-pool NMI came out
#: at 0.2927 on P2, because NMI between a 120-cluster partition and 58 classes is
#: inflated by cluster count and does not reach zero for an uninformative space.
#: Any NMI is therefore uninterpretable without the floor beside it.
NOISE_SEED = 12345


def open_pool_nmi(features: np.ndarray, class_name: np.ndarray,
                  unknown: np.ndarray, *, seed: int) -> float:
    """NMI of unknown class against a partition of the **whole** population.

    "Open-pool" is the operative word: the partition is fitted on every row of the
    population, which is what a selector would actually have, rather than on the
    unknown rows alone. Clustering only the unknowns would answer a question
    nobody can ask without the labels.
    """

    if int(unknown.sum()) <= 10 or features.shape[0] < N_CLUSTERS * 2:
        return float("nan")
    clusters = min(N_CLUSTERS, max(features.shape[0] // 4, 2))
    labels = MiniBatchKMeans(
        n_clusters=clusters, random_state=seed, n_init=3, batch_size=4096
    ).fit_predict(features)
    return float(normalized_mutual_info_score(class_name[unknown], labels[unknown]))


def spectrum(features: np.ndarray, *, seed: int) -> tuple[float, int]:
    """PC1 share of variance, and how many PCs reach 90%."""

    centred = features - features.mean(axis=0)
    sample = np.random.default_rng(seed).choice(
        centred.shape[0], min(20000, centred.shape[0]), replace=False
    )
    singular = np.linalg.svd(centred[sample], compute_uv=False)
    share = singular ** 2 / max(float((singular ** 2).sum()), 1e-12)
    return float(share[0]), int(np.searchsorted(np.cumsum(share), 0.90) + 1)


def measure(features: np.ndarray, pool: dict, mask: np.ndarray, *, seed: int) -> dict:
    """The seven predeclared metrics on one population."""

    local = features[mask]
    kind = pool["kind"][mask]
    class_name = pool["class_name"][mask]
    object_id = pool["object_id"][mask]
    group = pool["group"][mask]
    known, unknown = kind == "known", kind == "unknown"
    background = kind == "background"
    inner = {"class_name": class_name, "object_id": object_id, "k": K_NEIGHBOURS}

    pc1, dims = spectrum(local, seed=seed)
    return {
        "size": int(mask.sum()),
        "background_share": float(background.mean()),
        "unknown_objects": int(
            np.unique(object_id[unknown][object_id[unknown] >= 0]).size
        ),
        # 1-3: kNN, same-object neighbours always excluded
        "known_knn": knn_agreement(local, subset=known, **inner)["knn_class_agreement"],
        "unknown_knn": knn_agreement(local, subset=unknown, **inner)["knn_class_agreement"],
        "unknown_tail_knn": knn_agreement(
            local, subset=unknown & (group == "tail"), **inner)["knn_class_agreement"],
        # the stricter reading of the 0.15 threshold, printed for transparency
        "open_pool_unknown_knn": knn_agreement(
            local, subset=unknown, within_subset=False, **inner)["knn_class_agreement"],
        # 4: the metric the frozen rule applies the 0.15 threshold to
        "open_pool_nmi": open_pool_nmi(local, class_name, unknown, seed=seed),
        # 5: the safeguard
        "auc_unknown_vs_background": separability(local, unknown, background, seed=seed),
        "auc_object_vs_background": separability(local, known | unknown, background,
                                                 seed=seed),
        # 6-7
        "pc1_variance": pc1,
        "dims_for_90pct": dims,
    }


def verdict(rows: list[dict], *, source: str = "dinov2_vitb14") -> dict:
    """Apply the corrected frozen rule, and nothing else.

    NMI is carried through for reporting but takes no part in the decision: on
    this population a random unit-norm matrix reaches open-pool NMI 0.30-0.34, so
    a threshold there would be satisfied by noise. See protocol section 7.

    Criterion 4 is checked **per seed**, not on the mean, because that is how the
    decoder-layer protocol wrote it -- a margin that holds on average but fails on
    a seed is the rounding artefact the criterion exists to exclude.
    """

    relevant = [
        row for row in rows
        if row["source"] == source
        and row["representation"] == DECISION_REPRESENTATION
        and row["population"] == DECISION_POPULATION
    ]
    if not relevant:
        return {"verdict": "INDETERMINATE",
                "reason": f"no {source} rows on the decision population"}

    def mean(field: str) -> float:
        values = [float(row[field]) for row in relevant
                  if not np.isnan(float(row[field]))]
        return float(np.mean(values)) if values else float("nan")

    per_seed_knn = [float(row["unknown_knn"]) for row in relevant]
    unknown_knn = mean("unknown_knn")
    open_pool_knn = mean("open_pool_unknown_knn")
    auc = mean("auc_unknown_vs_background")

    checks = {
        "1_unknown_knn>=0.30": bool(unknown_knn >= GO_UNKNOWN_KNN),
        "2_open_pool_knn>=0.15": bool(open_pool_knn >= GO_OPEN_POOL_KNN),
        "3_auc>=0.76": bool(auc >= GO_AUC_UNKNOWN_VS_BACKGROUND),
        "4_margin>=0.05_every_seed": bool(
            per_seed_knn
            and all(value - PROB_P2_UNKNOWN_KNN >= GO_MARGIN_OVER_PROB
                    for value in per_seed_knn)
        ),
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "unknown_knn": unknown_knn,
        "open_pool_unknown_knn": open_pool_knn,
        "auc_unknown_vs_background": auc,
        "open_pool_nmi": mean("open_pool_nmi"),
        "per_seed_unknown_knn": per_seed_knn,
        "worst_seed_margin": (
            min(value - PROB_P2_UNKNOWN_KNN for value in per_seed_knn)
            if per_seed_knn else float("nan")
        ),
        "checks": checks,
        "seeds": len(relevant),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True,
                        help="dinov2_vitb14_method_v2_v1.npz")
    parser.add_argument("--pool", default=str(sf.POOL))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="method_v2_representation.csv")
    arguments = parser.parse_args()

    # ---- the pool, its oracle, and the frozen populations ------------------
    pool = load()
    payload = np.load(arguments.pool, allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == sf.POOL_SPLIT
    pool["raw_boxes"] = payload["boxes"][keep].astype(np.float32)
    candidates = proposals_module.from_frozen_pool(arguments.pool, split=sf.POOL_SPLIT)

    rows_meta = sf.pool_rows(arguments.pool)
    export = sf.read(arguments.export)
    report = sf.validate(export, rows_meta)
    print(f"[gate] export validated: {report}  PASS")
    print(f"[gate] provenance: model={export.provenance.get('model_id')} "
          f"git={str(export.provenance.get('git_sha'))[:8]} "
          f"pool_sha256={str(export.provenance.get('pool_sha256'))[:12]}")

    masks = populations(pool, candidates)
    print("\n[populations] exact n and oracle background fraction:")
    for name, mask in masks.items():
        kinds = pool["kind"][mask]
        print(f"  {name:20s} n={int(mask.sum()):6,d}  "
              f"background={float((kinds == 'background').mean()):.3f}")
    print("  historical reference: P0 80000/0.814 · P1 24000/0.652 · P2 15518/0.767")

    features_dinov2 = export.features()
    noise = np.random.default_rng(NOISE_SEED).normal(
        size=features_dinov2.shape).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    sources = {
        "dinov2_vitb14": features_dinov2,
        "prob_hs5_baseline": pool["raw"],
        "random_noise_floor": noise,
    }

    results: list[dict] = []
    for source, raw in sources.items():
        for representation in REPRESENTATIONS:
            for seed in range(arguments.seeds):
                features = represent(raw, representation, seed=seed)
                for population, mask in masks.items():
                    row = {"source": source, "representation": representation,
                           "population": population, "seed": seed} | measure(
                        features, pool, mask, seed=seed)
                    results.append(row)
                    print(f"  {source:18s} {representation:11s} {population:18s} "
                          f"s{seed}  known={row['known_knn']:.4f} "
                          f"unk={row['unknown_knn']:.4f} "
                          f"tail={row['unknown_tail_knn']:.4f} "
                          f"openNMI={row['open_pool_nmi']:.4f} "
                          f"auc={row['auc_unknown_vs_background']:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / arguments.out
    columns = list(dict.fromkeys(key for row in results for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nwrote {path} ({len(results)} rows)")

    # ---- the frozen decision ----------------------------------------------
    decision = verdict(results)
    def reference(source: str, field: str) -> float:
        values = [
            float(row[field]) for row in results
            if row["source"] == source
            and row["representation"] == DECISION_REPRESENTATION
            and row["population"] == DECISION_POPULATION
            and not np.isnan(float(row[field]))
        ]
        return float(np.mean(values)) if values else float("nan")

    print("\n" + "=" * 78)
    print(f"METHOD V2 REPRESENTATION DECISION  ({DECISION_REPRESENTATION} / "
          f"{DECISION_POPULATION}, mean of {decision.get('seeds', 0)} seeds)")
    print("=" * 78)
    checks = decision["checks"]
    seeds = ", ".join(f"{value:.4f}" for value in decision["per_seed_unknown_knn"])
    print(f"  1  unknown-class kNN         = {decision['unknown_knn']:.4f}   "
          f">= {GO_UNKNOWN_KNN:.2f}    "
          f"{'MET' if checks['1_unknown_knn>=0.30'] else 'NOT MET'}")
    print(f"  2  open-pool unknown kNN     = {decision['open_pool_unknown_knn']:.4f}   "
          f">= {GO_OPEN_POOL_KNN:.2f}    "
          f"{'MET' if checks['2_open_pool_knn>=0.15'] else 'NOT MET'}")
    print(f"  3  unknown/background AUC    = {decision['auc_unknown_vs_background']:.4f}   "
          f">= {GO_AUC_UNKNOWN_VS_BACKGROUND:.2f}    "
          f"{'MET' if checks['3_auc>=0.76'] else 'NOT MET'}")
    print(f"  4  margin over PROB {PROB_P2_UNKNOWN_KNN:.4f}   "
          f"worst seed = {decision['worst_seed_margin']:+.4f}   "
          f">= {GO_MARGIN_OVER_PROB:.2f}    "
          f"{'MET' if checks['4_margin>=0.05_every_seed'] else 'NOT MET'}")
    print(f"     per-seed unknown kNN: {seeds}")
    print(f"\n  open-pool NMI            = {decision['open_pool_nmi']:.4f}   "
          "DESCRIPTIVE ONLY, no threshold -- noise reaches 0.30-0.34 here")
    print("\n  same population, same metric code, for reference:")
    for source, label in (("prob_hs5_baseline", "PROB hs[5]"),
                          ("random_noise_floor", "random noise floor")):
        print(f"    {label:20s} unknown_knn={reference(source, 'unknown_knn'):.4f}  "
              f"openNMI={reference(source, 'open_pool_nmi'):.4f}  "
              f"open_pool_knn={reference(source, 'open_pool_unknown_knn'):.4f}  "
              f"auc={reference(source, 'auc_unknown_vs_background'):.4f}")
    noise = verdict(results, source="random_noise_floor")
    if noise.get("verdict") == "PASS":
        raise SystemExit(
            "the random noise floor PASSES the semantic gate; the gate is broken "
            "and the DINOv2 result would be meaningless. Refusing to report."
        )
    print(f"\n  the noise floor FAILS the corrected gate, as it must "
          f"(openNMI {noise.get('open_pool_nmi', float('nan')):.4f} would have "
          "passed an NMI threshold of 0.15 -- which is why NMI is descriptive only)")
    print()
    print(f"METHOD_V2_REPRESENTATION_{decision['verdict']}")


if __name__ == "__main__":
    main()
