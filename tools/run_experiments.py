"""Recompute every number the README and the notebook report.

Three experiments, all on the committed PROB pass, all on a laptop:

1. ``clustering``  — does one partition give usable rarity and diversity, and
   how much known content does it leak into unknown clusters (consultation §3)?
2. ``selection``   — which score puts the most tail objects in front of the
   annotator at equal cost (consultation §1, §2, §7)?
3. ``labelling``   — what does each labelling policy cost, and how much
   half-labelling does it create (consultation §5)?

Writes CSV into ``data/results/``. Nothing here is hand-typed.

    python tools/run_experiments.py --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import clustering, labelling, proposals, protocol, selection

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"


def write(name: str, rows: list[dict]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(path.parent.parent.parent)}  ({len(rows)} rows)")
    return path


# ---------------------------------------------------------------------------


def experiment_clustering(pool: proposals.Candidates, seeds: int) -> list[dict]:
    """Consultation §3: one clustering, judged by known contamination."""

    oracle = pool.oracle()
    detector_known = clustering.predicted_known(pool.posterior, len(protocol.TASK1))
    truth_known = oracle.kind == "known"
    rows: list[dict] = []
    for seed in range(seeds):
        for k in (200, 400, 800, 1600, 3200):
            partition = clustering.fit(pool.embeddings, n_clusters=k, seed=seed)
            estimated = clustering.contamination(partition, detector_known)
            verified = clustering.contamination(partition, truth_known)
            gate = clustering.noise_gate(partition, minimum_size=5)
            rows.append({
                "seed": seed,
                "n_clusters": k,
                "mean_cluster_size": float(partition.sizes[partition.sizes > 0].mean()),
                "contamination_detector_estimate": estimated["contamination"],
                "contamination_verified": verified["contamination"],
                "unknown_recall_verified": verified["unknown_recall"],
                "gate_open_share": float((gate > 0).mean()),
            })
    return rows


def experiment_dbscan_gate(pool: proposals.Candidates) -> list[dict]:
    """Consultation §2: is the binary noise gate keeping objects or losing them?"""

    oracle = pool.oracle()
    rows: list[dict] = []
    for eps in (0.15, 0.25, 0.35, 0.45):
        partition = clustering.fit(
            pool.embeddings, method="dbscan", eps=eps, min_samples=5,
            pca_dimensions=32, seed=0,
        )
        noise = partition.is_noise
        rows.append({
            "eps": eps,
            "clusters": partition.n_clusters,
            "noise_share_background": float(noise[oracle.kind == "background"].mean()),
            "noise_share_known": float(noise[oracle.kind == "known"].mean()),
            "noise_share_unknown_object": float(noise[oracle.kind == "unknown"].mean()),
            "unknown_purity_before": float((oracle.kind == "unknown").mean()),
            "unknown_purity_after_gate": float((oracle.kind[~noise] == "unknown").mean())
            if (~noise).any() else float("nan"),
        })
    return rows


def experiment_selection(pool: proposals.Candidates, seeds: int, budget: int) -> list[dict]:
    """Consultation §1, §2, §7: which score buys the most tail, at equal cost?"""

    oracle = pool.oracle()
    groups = protocol.load_groups()
    group_of = np.asarray([groups.get(name, "") for name in oracle.class_name])
    is_object = oracle.kind == "unknown"
    rows: list[dict] = []

    for seed in range(seeds):
        partition = clustering.fit(pool.embeddings, n_clusters=1600, seed=seed)
        for name, config in selection.ARMS.items():
            for rounds in (1, 6, 12):
                picked = selection.select(
                    pool, type(config)(**{**config.__dict__, "seed": seed}),
                    budget=budget, rounds=rounds,
                    n_known=len(protocol.TASK1), partition=partition,
                )
                index = picked.indices
                found = is_object[index]
                group = group_of[index][found]
                names = oracle.class_name[index][found]
                rows.append({
                    "seed": seed,
                    "arm": name,
                    "rounds": rounds,
                    "budget": budget,
                    "unknown_objects": int(np.unique(oracle.object_id[index][found]).size),
                    "head": int((group == "head").sum()),
                    "medium": int((group == "medium").sum()),
                    "tail": int((group == "tail").sum()),
                    "classes": int(np.unique(names).size),
                    "tail_classes": int(np.unique(names[group == "tail"]).size),
                    "images_opened": int(picked.images(pool).size),
                })
    return rows


def experiment_labelling(pool: proposals.Candidates, seeds: int, budget: int) -> list[dict]:
    """Consultation §5: what does each labelling policy cost and mis-teach?"""

    rows: list[dict] = []
    for seed in range(seeds):
        partition = clustering.fit(pool.embeddings, n_clusters=1600, seed=seed)
        config = selection.ARMS["prior_consult_batch"]
        picked = selection.select(
            pool, type(config)(**{**config.__dict__, "seed": seed}),
            budget=budget, rounds=6, n_known=len(protocol.TASK1), partition=partition,
        )
        for policy in labelling.POLICIES:
            annotation = labelling.annotate(
                pool, picked, policy=policy, known_classes=protocol.TASK1
            )
            summary = annotation.summary()
            summary |= {
                "seed": seed,
                "half_labelled_share": labelling.half_labelling_rate(annotation, pool),
                "cost_ratio_vs_budget": annotation.oracle_cost / budget,
                "supervision_per_oracle_unit": summary["labelled"] / max(annotation.oracle_cost, 1),
            }
            rows.append(summary)
    return rows


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument(
        "--only", nargs="*",
        choices=["clustering", "dbscan", "selection", "labelling"],
        default=["clustering", "dbscan", "selection", "labelling"],
    )
    arguments = parser.parse_args()

    pool = proposals.from_frozen_pool(split="pool")
    print("pool:", pool.describe())

    if "clustering" in arguments.only:
        write("clustering_contamination.csv", experiment_clustering(pool, arguments.seeds))
    if "dbscan" in arguments.only:
        write("coherence_gate.csv", experiment_dbscan_gate(pool))
    if "selection" in arguments.only:
        write("selection_arms.csv", experiment_selection(pool, arguments.seeds, arguments.budget))
    if "labelling" in arguments.only:
        write("labelling_policy.csv", experiment_labelling(pool, arguments.seeds, arguments.budget))


if __name__ == "__main__":
    main()
