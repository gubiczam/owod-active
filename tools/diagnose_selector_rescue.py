"""Which structural steps from the consultation actually buy discovery?

Day 1 falsified three operationalisations of the research plan's score. This
tool asks the narrower, answerable question: taking the consultation's *other*
ideas -- object-level unit of analysis, known rejection, cluster-aware budget
allocation -- does any of them beat the learning-free control on the frozen
primary endpoint, distinct tail objects per 600 oracle regions?

Everything here is oracle-free at selection time. The oracle scores the result.

Three families, and the control they must beat:

``A(x)``                    objectness * sqrt(area), ranked, top 600. The control.
``+ NMS``                   per-image IoU suppression first, so the budget is not
                            spent twice on one physical object. This is the
                            consultation's object-level unit, made oracle-free.
``+ known rejection``       drop the most known-like candidates before ranking --
                            "as few known as possible cross into unknown".
``cluster allocation``      spread the budget across semantic clusters instead of
                            ranking by A(x): the "distance to all cluster
                            centroids, like k-means++" idea, as an objective.
``cluster cap``             the gentler version: still rank by A(x), but cap how
                            many picks one cluster may absorb -- structure as a
                            constraint rather than as an objective.

Sensitivity is reported for every threshold that could have been tuned, because
a gain that exists at one threshold and reverses at the next is a lucky operating
point rather than an effect.

    python tools/diagnose_selector_rescue.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import MiniBatchKMeans

from owl import clustering, discovery, protocol, scoring
from owl import proposals as proposals_module
from tools.diagnose_population import nms_keep, whitened
from tools.diagnose_representation import N_KNOWN_AT_T1, UNKNOWN_SLOT, load

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"
BUDGET = 600


def _nms_by_admissibility(pool: dict, admissibility: np.ndarray,
                          mask: np.ndarray, iou: float) -> np.ndarray:
    """NMS keeping the most object-like box of each overlapping group.

    ``nms_keep`` orders by ``-pred_obj``; the control ranks by ``A(x)``, so the
    suppression must use the same order or the two are not one variable apart.
    """

    saved = pool["pred_obj"].copy()
    pool["pred_obj"] = -admissibility
    try:
        return nms_keep(pool, mask, iou)
    finally:
        pool["pred_obj"] = saved


def _known_like(pool: dict, candidates, features: np.ndarray, rule: str) -> np.ndarray:
    """How much a candidate is already explained as a known class. Oracle-free.

    ``posterior``  known-class mass minus unknown-slot mass. The classification
                   view, which the representation audit measured to be best for
                   known structure.
    ``prototype``  cosine similarity to the nearest mean of detector-predicted
                   known regions, in the whitened space.
    """

    posterior = pool["posterior"]
    if rule == "posterior":
        return posterior[:, :N_KNOWN_AT_T1].max(axis=1) - posterior[:, UNKNOWN_SLOT]
    if rule == "prototype":
        predicted = posterior[:, :N_KNOWN_AT_T1].argmax(axis=1)
        is_known = clustering.predicted_known(candidates.posterior, N_KNOWN_AT_T1)
        rows = [
            features[is_known & (predicted == index)].mean(axis=0)
            for index in range(N_KNOWN_AT_T1)
            if (is_known & (predicted == index)).any()
        ]
        prototypes = np.asarray(rows, dtype=np.float32)
        prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-9)
        return (features @ prototypes.T).max(axis=1)
    raise ValueError(rule)


def rank_take(admissibility: np.ndarray, available: np.ndarray, budget: int) -> np.ndarray:
    index = np.flatnonzero(available)
    return index[np.argsort(-admissibility[index], kind="stable")][:budget]


def cluster_cap(admissibility: np.ndarray, available: np.ndarray, labels: np.ndarray,
                per_cluster: int, budget: int) -> np.ndarray:
    """Rank by A(x); refuse a cluster more than ``per_cluster`` picks."""

    index = np.flatnonzero(available)
    order = index[np.argsort(-admissibility[index], kind="stable")]
    taken: dict[int, int] = {}
    picked: list[int] = []
    for position in order:
        if len(picked) >= budget:
            break
        label = int(labels[position])
        if taken.get(label, 0) >= per_cluster:
            continue
        picked.append(int(position))
        taken[label] = taken.get(label, 0) + 1
    return np.asarray(picked, dtype=np.int64)


def cluster_allocate(admissibility: np.ndarray, available: np.ndarray, labels: np.ndarray,
                     budget: int) -> np.ndarray:
    """Round-robin across clusters, most object-like member of each in turn."""

    index = np.flatnonzero(available)
    members: dict[int, list[int]] = {}
    for position in index:
        members.setdefault(int(labels[position]), []).append(int(position))
    for group in members.values():
        group.sort(key=lambda p: -admissibility[p])
    picked: list[int] = []
    order = sorted(members)
    while len(picked) < budget:
        moved = False
        for label in order:
            if members[label]:
                picked.append(members[label].pop(0))
                moved = True
                if len(picked) >= budget:
                    break
        if not moved:
            break
    return np.asarray(picked, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    arguments = parser.parse_args()

    pool = load()
    payload = np.load(
        Path(__file__).resolve().parent.parent / "data" / "pool" / "sowodb_t1_frozen_pool.npz",
        allow_pickle=True,
    )
    keep = np.asarray(payload["split"], dtype=str) == "pool"
    pool["raw_boxes"] = payload["boxes"][keep].astype(np.float32)

    candidates = proposals_module.from_frozen_pool(split="pool")
    groups = protocol.load_groups()
    admissibility = scoring.admissibility(candidates)
    everything = np.ones(admissibility.size, dtype=bool)

    rows: list[dict] = []

    def record(arm: str, selection: np.ndarray, *, seed: int = 0, note: str = "") -> None:
        result = discovery.discovery(candidates, selection, groups=groups)
        rows.append({"arm": arm, "seed": seed, "note": note} | result.row())
        print(f"{arm:46s} s{seed} unk_obj={result.unknown_objects:4d} "
              f"tail_obj={result.objects_by_group['tail']:3d} "
              f"tail%={100 * result.tail_share:5.1f} "
              f"tail_cls={result.classes_by_group['tail']:2d} "
              f"p/obj={result.proposals_per_object:5.3f} "
              f"imgs={result.images_opened:4d} "
              f"tail/img={result.per_image('tail'):.4f}")

    print("=== control, and the object-level unit ===")
    record("A(x) [CONTROL]", rank_take(admissibility, everything, BUDGET))

    nms_masks = {
        iou: _nms_by_admissibility(pool, admissibility, everything, iou)
        for iou in (0.3, 0.5, 0.6, 0.7, 0.9)
    }
    for iou, mask in nms_masks.items():
        record(f"A(x) + NMS iou={iou}", rank_take(admissibility, mask, BUDGET),
               note=f"kept {int(mask.sum())}")

    print("\n=== known rejection, full threshold grid (the tunable knob) ===")
    base = nms_masks[0.6]
    for seed in range(arguments.seeds):
        features = whitened(pool["raw"], seed=seed)
        for rule in ("prototype", "posterior"):
            score = _known_like(pool, candidates, features, rule)
            for share in (0.10, 0.20, 0.25, 0.30, 0.40, 0.50):
                cut = np.quantile(score[base], 1.0 - share)
                record(f"A(x)+NMS - drop {int(share * 100)}% known-like [{rule}]",
                       rank_take(admissibility, base & ~(score >= cut), BUDGET), seed=seed)

    print("\n=== structure as an objective, and as a constraint ===")
    for seed in range(arguments.seeds):
        features = whitened(pool["raw"], seed=seed)
        for n_clusters in (120, 300):
            labels = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=seed, n_init=3, batch_size=4096
            ).fit_predict(features)
            record(f"cluster ALLOCATION K={n_clusters}",
                   cluster_allocate(admissibility, base, labels, BUDGET), seed=seed)
            for per_cluster in (2, 4, 8):
                record(f"cluster CAP {per_cluster}/cluster K={n_clusters}",
                       cluster_cap(admissibility, base, labels, per_cluster, BUDGET), seed=seed)

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "selector_rescue.csv"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
