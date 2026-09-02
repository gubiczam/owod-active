"""A1.2: choose D_known, and check that D_known and B(x|S) do what they claim.

The consultation asked whether diversity should mean distance from what is
already labelled, or spread among the newly selected. Our answer is that these
are **two different objectives**, so this tool treats them as two separate
questions.

**Question 1 -- which D_known.** Three definitions are candidates. They are
compared on compute, seed stability, responsiveness to a growing labelled set,
and retrospective discriminative power, and the choice is recorded *before* the
A1.3 ladder runs. It is deliberately not made on which produces the best
endpoint; the decision columns here are properties of the estimator, not of the
result it would produce.

``nearest_labelled``
    cosine distance to the nearest already-labelled embedding. The reference set
    is what the annotator has answered for.
``nearest_known_prototype``
    distance to the nearest mean embedding of the regions the detector itself
    calls a known class. 19 prototypes, no annotation read.
``nearest_known_cluster``
    distance from the candidate's own k-means centroid to the nearest centroid of
    a known-enriched cluster. The "one clustering, both terms" version.

**Question 2 -- does B(x|S) reduce redundancy.** Within-batch diversity has a
predicted sign: it must lower mean pairwise similarity inside the batch, and it
must lower **proposals per distinct object**, because near-duplicate proposals on
one object are the most redundant thing a batch can contain. A wrong sign is an
implementation bug, not a finding.

Oracle labels are used only to score estimators after the fact.

    python tools/diagnose_novelty.py --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import clustering, protocol, scoring, selection
from owl import proposals as proposals_module

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"

BUDGET = 600
ROUNDS = 6
DEFINITIONS = ("nearest_labelled", "nearest_known_prototype", "nearest_known_cluster")


def _known_prototypes(pool: proposals_module.Candidates, n_known: int) -> np.ndarray:
    """One mean embedding per class the detector predicts. No annotation read."""

    predicted = pool.posterior[:, :n_known].argmax(axis=1)
    is_known = clustering.predicted_known(pool.posterior, n_known)
    rows = []
    for index in range(n_known):
        members = pool.embeddings[is_known & (predicted == index)]
        if members.shape[0]:
            rows.append(members.mean(axis=0))
    if not rows:
        return np.zeros((0, pool.embeddings.shape[1]), dtype=np.float32)
    prototypes = np.asarray(rows, dtype=np.float32)
    norm = np.linalg.norm(prototypes, axis=1, keepdims=True)
    return prototypes / np.maximum(norm, 1e-9)


def compute(
    definition: str,
    pool: proposals_module.Candidates,
    *,
    labelled: np.ndarray,
    partition: clustering.Partition,
    n_known: int,
) -> np.ndarray:
    if definition == "nearest_labelled":
        return scoring.novelty(pool, labelled)
    if definition == "nearest_known_prototype":
        prototypes = _known_prototypes(pool, n_known)
        if prototypes.shape[0] == 0:
            return np.ones(len(pool), dtype=np.float32)
        return 1.0 - (pool.embeddings @ prototypes.T).max(axis=1)
    if definition == "nearest_known_cluster":
        is_known = clustering.predicted_known(pool.posterior, n_known)
        labels = partition.labels
        valid = labels >= 0
        counts = np.bincount(labels[valid], minlength=partition.n_clusters)
        known_counts = np.bincount(labels[valid & is_known], minlength=partition.n_clusters)
        enriched = (known_counts / np.maximum(counts, 1)) > float(is_known[valid].mean())
        return scoring.cluster_novelty(partition, enriched)
    raise ValueError(definition)


# ---------------------------------------------------- question 1: D_known ---


def compare_definitions(pool: proposals_module.Candidates, seeds: int) -> list[dict]:
    """Estimator properties, measured. Not endpoint performance."""

    oracle = pool.oracle()
    is_unknown = oracle.kind == "unknown"
    n_known = len(protocol.TASK1)
    rows: list[dict] = []

    # a fixed, arm-independent labelled set, so this measures the estimator and
    # not the acquisition score that happened to fill the set
    rng = np.random.default_rng(0)
    seeded = rng.choice(len(pool), size=100, replace=False)

    for seed in range(seeds):
        partition = clustering.fit(pool.embeddings, n_clusters=1600, seed=seed)
        for definition in DEFINITIONS:
            start = time.perf_counter()
            small = compute(
                definition, pool,
                labelled=pool.embeddings[seeded[:100]],
                partition=partition, n_known=n_known,
            )
            elapsed = time.perf_counter() - start

            # responsiveness: does it move when the labelled set grows 100 -> 600?
            grown = compute(
                definition, pool,
                labelled=pool.embeddings[
                    rng.choice(len(pool), size=600, replace=False)
                ],
                partition=partition, n_known=n_known,
            )
            drift = float(np.abs(grown - small).mean())
            rank_shift = float(
                1.0 - np.corrcoef(
                    scoring.rank_normalise(small), scoring.rank_normalise(grown)
                )[0, 1]
            )

            # retrospective discriminative power: real unknowns in the top vs
            # the bottom decile of the term
            order = np.argsort(-small, kind="stable")
            decile = max(len(pool) // 10, 1)
            top = is_unknown[order[:decile]].mean()
            bottom = is_unknown[order[-decile:]].mean()

            rows.append({
                "seed": seed,
                "definition": definition,
                "seconds": round(elapsed, 4),
                "reference_set_size": (
                    100 if definition == "nearest_labelled"
                    else (n_known if definition == "nearest_known_prototype"
                          else partition.n_clusters)
                ),
                "mean": float(small.mean()),
                "sd": float(small.std()),
                "labelled_growth_drift": drift,
                "labelled_growth_rank_shift": rank_shift,
                "unknown_rate_top_decile": float(top),
                "unknown_rate_bottom_decile": float(bottom),
                "discrimination_ratio": float(top / bottom) if bottom > 0 else float("inf"),
            })
    return rows


# ------------------------------------------------- question 2: B(x | S) -----


def validate_batch_diversity(pool: proposals_module.Candidates, seeds: int) -> list[dict]:
    """Predicted sign: mu > 0 lowers within-batch similarity and redundancy."""

    oracle = pool.oracle()
    groups = protocol.load_groups()
    group_of = np.asarray([groups.get(name, "") for name in oracle.class_name])
    is_unknown = oracle.kind == "unknown"
    rows: list[dict] = []

    base = selection.ARMS_V2["a_u_d_rc"]
    for seed in range(seeds):
        partition = clustering.fit(pool.embeddings, n_clusters=1600, seed=seed)
        for mu in (0.0, 0.3):
            config = type(base)(**{**base.__dict__, "seed": seed, "mu_batch": mu})
            picked = selection.select(
                pool, config, budget=BUDGET, rounds=ROUNDS,
                n_known=len(protocol.TASK1), partition=partition,
            )
            index = picked.indices
            embeddings = pool.embeddings[index]

            # mean pairwise cosine similarity inside the batch, diagonal removed
            similarity = embeddings @ embeddings.T
            n = similarity.shape[0]
            off_diagonal = (similarity.sum() - np.trace(similarity)) / max(n * (n - 1), 1)

            found = is_unknown[index]
            object_ids = oracle.object_id[index][found]
            distinct = int(np.unique(object_ids).size)
            tail_ids = oracle.object_id[index][found & (group_of[index] == "tail")]

            rows.append({
                "seed": seed,
                "mu_batch": mu,
                "asked": int(index.size),
                "mean_pairwise_similarity": float(off_diagonal),
                "unknown_proposals": int(found.sum()),
                "distinct_unknown_objects": distinct,
                "proposals_per_distinct_object": (
                    float(found.sum() / distinct) if distinct else float("nan")
                ),
                "distinct_tail_objects": int(np.unique(tail_ids[tail_ids >= 0]).size),
                "images_opened": int(picked.images(pool).size),
                "selected_background": int((oracle.kind[index] == "background").sum()),
                "selected_known": int((oracle.kind[index] == "known").sum()),
            })
    return rows


# ---------------------------------------------------------------------------


def write(name: str, rows: list[dict]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    arguments = parser.parse_args()

    pool = proposals_module.from_frozen_pool(split="pool")
    print("pool:", pool.describe(), "\n")

    print("--- question 1: which D_known definition ---")
    definitions = compare_definitions(pool, arguments.seeds)
    write("novelty_definitions.csv", definitions)
    for definition in DEFINITIONS:
        subset = [row for row in definitions if row["definition"] == definition]
        print(
            f"  {definition:26s} "
            f"t={np.mean([r['seconds'] for r in subset]):7.3f}s  "
            f"drift={np.mean([r['labelled_growth_drift'] for r in subset]):.4f}  "
            f"rank_shift={np.mean([r['labelled_growth_rank_shift'] for r in subset]):.4f}  "
            f"seed_sd(mean)={np.std([r['mean'] for r in subset]):.5f}  "
            f"unk_top/bottom={np.mean([r['discrimination_ratio'] for r in subset]):6.2f}"
        )

    print("\n--- question 2: does B(x|S) reduce redundancy ---")
    batch = validate_batch_diversity(pool, arguments.seeds)
    write("batch_diversity_validation.csv", batch)
    for mu in (0.0, 0.3):
        subset = [row for row in batch if row["mu_batch"] == mu]
        print(
            f"  mu={mu:.1f}  "
            f"pairwise_sim={np.mean([r['mean_pairwise_similarity'] for r in subset]):.4f}  "
            f"prop/object={np.mean([r['proposals_per_distinct_object'] for r in subset]):.3f}  "
            f"distinct_unk={np.mean([r['distinct_unknown_objects'] for r in subset]):6.1f}  "
            f"distinct_tail={np.mean([r['distinct_tail_objects'] for r in subset]):5.1f}  "
            f"images={np.mean([r['images_opened'] for r in subset]):5.1f}"
        )


if __name__ == "__main__":
    main()
