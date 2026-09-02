"""Is the *population* wrong rather than the idea?

The 2026-08-25 note is explicit about the population it imagines: "if there are
e.g. **1000 known and 10000 unknown**, clustering must be done so that as few
known as possible cross over into the unknown". That is a population of *object
instances* in which unknowns outnumber knowns ten to one, and background is not
mentioned at all.

The pool the code clusters is the opposite: 65,129 background, 12,113 known,
2,758 unknown -- 81% background, and knowns outnumbering unknowns 4.4 to 1 -- and
its 2,758 unknown proposals sit on only 1,268 distinct objects, so the same
physical thing appears 2.5 times on average.

So before concluding anything about the hypothesis, this measures what happens to
the semantic geometry as the population is moved toward the one the idea assumes:

    raw proposals
      -> objectness-filtered            (background rejection, oracle-free)
      -> IoU-deduplicated               (one representative per object, oracle-free)
      -> known-rejected                 (posterior-based, oracle-free)
      -> oracle-deduplicated            (upper bound, DIAGNOSIS ONLY)

Each stage is oracle-free except the last, which exists only to bound what a
perfect deduplicator could deliver.

Oracle labels score the stages; they never select within them.

    python tools/diagnose_population.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from tools.diagnose_representation import (
    K_NEIGHBOURS,
    N_KNOWN_AT_T1,
    UNKNOWN_SLOT,
    knn_agreement,
    load,
)

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def whitened(raw: np.ndarray, *, dimensions: int = 32, seed: int = 0) -> np.ndarray:
    """The representation the audit picked: PCA to 32 dims, whitened, re-normalised.

    Whitening is not cosmetic here. PC1 of the plain unit embedding carries 59.7%
    of the variance and is the objectness axis (rho -0.72 against the raw norm),
    so any Euclidean or density method on the plain embedding is dominated by a
    signal ``pred_obj`` already reports explicitly. Equalising the axes is what
    lets the weaker class-semantic directions participate at all.
    """

    unit = _unit(raw)
    sample = np.random.default_rng(seed).choice(unit.shape[0], min(20000, unit.shape[0]),
                                                replace=False)
    centre = unit.mean(axis=0)
    basis = np.linalg.svd(unit[sample] - centre, full_matrices=False)[2][:dimensions]
    projected = (unit - centre) @ basis.T
    return _unit(projected / np.maximum(projected.std(axis=0, keepdims=True), 1e-6))


# ------------------------------------------------- oracle-free filter stages ---


def objectness_keep(pool: dict, share: float) -> np.ndarray:
    """Top ``share`` by PROB's own objectness ordering. No annotation read."""

    score = -pool["pred_obj"]          # lower pred_obj = more object-like
    keep = max(round(score.size * share), 1)
    mask = np.zeros(score.size, dtype=bool)
    mask[np.argsort(-score, kind="stable")[:keep]] = True
    return mask


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU for normalised cxcywh boxes."""

    cx, cy, w, h = boxes.T
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    left = np.maximum(x1[:, None], x1[None, :])
    top = np.maximum(y1[:, None], y1[None, :])
    right = np.minimum(x2[:, None], x2[None, :])
    bottom = np.minimum(y2[:, None], y2[None, :])
    overlap = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
    union = area[:, None] + area[None, :] - overlap
    return overlap / np.maximum(union, 1e-9)


def nms_keep(pool: dict, candidate: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Per-image IoU suppression, keeping the most object-like box of each cluster.

    This is the oracle-free stand-in for "one representative per object". It is
    ordinary NMS, run inside each image, on the candidates that survived the
    objectness stage.
    """

    keep = np.zeros(candidate.size, dtype=bool)
    order_key = -pool["pred_obj"]
    for image in np.unique(pool["image_ids"][candidate]):
        local = np.flatnonzero(candidate & (pool["image_ids"] == image))
        if local.size == 0:
            continue
        local = local[np.argsort(-order_key[local], kind="stable")]
        iou = _iou_matrix(pool["raw_boxes"][local])
        alive = np.ones(local.size, dtype=bool)
        for position in range(local.size):
            if not alive[position]:
                continue
            keep[local[position]] = True
            alive &= ~((iou[position] > threshold) & (np.arange(local.size) > position))
    return keep


def known_reject(pool: dict, candidate: np.ndarray, *, quantile: float = 0.5) -> np.ndarray:
    """Drop candidates the detector itself explains as a known class.

    Uses the posterior over the 19 classes trained at t1 against the unknown
    slot -- the classification-oriented view, which the representation audit
    measured to be the *best* representation for known structure (kNN class
    agreement 0.942) and the worst for unknown structure (0.114). Using each view
    for what it is good at is the point.

    ``quantile`` is the share of the surviving candidates dropped as known-like,
    fixed rather than tuned against an endpoint.
    """

    posterior = pool["posterior"]
    known_mass = posterior[:, :N_KNOWN_AT_T1].max(axis=1)
    unknown_mass = posterior[:, UNKNOWN_SLOT]
    known_like = known_mass - unknown_mass                # high = explained as known
    index = np.flatnonzero(candidate)
    if index.size == 0:
        return candidate.copy()
    cut = np.quantile(known_like[index], 1.0 - quantile)
    out = candidate.copy()
    out[index[known_like[index] >= cut]] = False
    return out


def oracle_dedup(pool: dict, candidate: np.ndarray) -> np.ndarray:
    """DIAGNOSIS ONLY. One proposal per GT object, chosen by objectness.

    Never available to a selector. It bounds what a perfect deduplicator could
    hand the clustering, so the gap between it and ``nms`` prices the
    deduplicator.
    """

    keep = np.zeros(candidate.size, dtype=bool)
    index = np.flatnonzero(candidate)
    object_id = pool["object_id"]
    best: dict[int, int] = {}
    for position in index:
        identifier = int(object_id[position])
        if identifier < 0:
            keep[position] = True                # background keeps every box
            continue
        current = best.get(identifier)
        if current is None or pool["pred_obj"][position] < pool["pred_obj"][current]:
            best[identifier] = int(position)
    for position in best.values():
        keep[position] = True
    return keep


# ---------------------------------------------------------------- scoring ---


def score_population(name: str, pool: dict, features: np.ndarray,
                     mask: np.ndarray, *, n_clusters: int) -> dict:
    kind = pool["kind"][mask]
    class_name = pool["class_name"][mask]
    object_id = pool["object_id"][mask]
    group = pool["group"][mask]
    local = features[mask]

    unknown = kind == "unknown"
    known = kind == "known"
    background = kind == "background"
    distinct_unknown = np.unique(object_id[unknown][object_id[unknown] >= 0]).size
    distinct_tail = np.unique(
        object_id[unknown & (group == "tail")][object_id[unknown & (group == "tail")] >= 0]
    ).size

    inner = {"class_name": class_name, "object_id": object_id}
    row: dict[str, object] = {
        "population": name,
        "size": int(mask.sum()),
        "background": int(background.sum()),
        "known": int(known.sum()),
        "unknown_proposals": int(unknown.sum()),
        "unknown_objects": int(distinct_unknown),
        "tail_objects": int(distinct_tail),
        "background_share": float(background.mean()),
        "unknown_object_share": float(distinct_unknown / max(mask.sum(), 1)),
        "unknown_to_known_ratio": float(unknown.sum() / max(known.sum(), 1)),
        "proposals_per_unknown_object": float(
            unknown.sum() / max(distinct_unknown, 1)
        ),
    }

    # is there class structure among the unknowns of this population?
    row |= {f"unk_{k}": v for k, v in knn_agreement(
        local, subset=unknown, k=K_NEIGHBOURS, within_subset=True, **inner).items()}
    # and would a clustering of the whole population find it?
    row |= {f"open_{k}": v for k, v in knn_agreement(
        local, subset=unknown, k=K_NEIGHBOURS, within_subset=False, **inner).items()}

    # cluster the whole population, then read the unknown structure off it --
    # which is what a real selector must do
    clusters = min(n_clusters, max(int(mask.sum()) // 4, 2))
    labels = MiniBatchKMeans(
        n_clusters=clusters, random_state=0, n_init=3, batch_size=4096
    ).fit_predict(local)
    row["clusters"] = clusters
    if unknown.sum() > 10:
        row["unk_nmi_in_full_partition"] = float(
            normalized_mutual_info_score(class_name[unknown], labels[unknown])
        )
        row["unk_ari_in_full_partition"] = float(
            adjusted_rand_score(class_name[unknown], labels[unknown])
        )
    else:
        row["unk_nmi_in_full_partition"] = float("nan")
        row["unk_ari_in_full_partition"] = float("nan")

    # known contamination of the clusters an unknown-hunting selector would use,
    # which is the quantity the consultation named
    counts = np.bincount(labels, minlength=clusters)
    known_counts = np.bincount(labels[known], minlength=clusters)
    baseline = float(known.mean())
    known_owned = (known_counts / np.maximum(counts, 1)) > baseline
    row["known_contamination"] = float(
        (known & ~known_owned[labels]).sum() / max(int(known.sum()), 1)
    )
    row["unknown_recall_in_candidate_clusters"] = float(
        (unknown & ~known_owned[labels]).sum() / max(int(unknown.sum()), 1)
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectness-share", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--known-reject", type=float, default=0.5)
    parser.add_argument("--clusters", type=int, default=120)
    arguments = parser.parse_args()

    pool = load()
    payload = np.load(
        Path(__file__).resolve().parent.parent / "data" / "pool" / "sowodb_t1_frozen_pool.npz",
        allow_pickle=True,
    )
    keep = np.asarray(payload["split"], dtype=str) == "pool"
    pool["raw_boxes"] = payload["boxes"][keep].astype(np.float32)

    features = whitened(pool["raw"])
    print(f"representation: PCA-32 whitened unit embedding  {features.shape}\n")

    everything = np.ones(features.shape[0], dtype=bool)
    stages: list[tuple[str, np.ndarray]] = [("raw_proposals", everything)]

    objectness = objectness_keep(pool, arguments.objectness_share)
    stages.append((f"objectness_top{int(arguments.objectness_share * 100)}", objectness))

    nms = nms_keep(pool, objectness, arguments.nms_iou)
    stages.append((f"+nms_iou{arguments.nms_iou}", nms))

    rejected = known_reject(pool, nms, quantile=arguments.known_reject)
    stages.append((f"+known_reject{arguments.known_reject}", rejected))

    stages.append(("[oracle] perfect_dedup", oracle_dedup(pool, objectness)))
    stages.append(("[oracle] perfect_dedup+known_reject",
                   known_reject(pool, oracle_dedup(pool, objectness),
                                quantile=arguments.known_reject)))

    rows = []
    for name, mask in stages:
        row = score_population(name, pool, features, mask, n_clusters=arguments.clusters)
        rows.append(row)
        print(f"{name:38s} n={row['size']:6,d}  bg={row['background_share']:.3f}  "
              f"unk_obj={row['unknown_objects']:5d}  tail_obj={row['tail_objects']:4d}  "
              f"unk/known={row['unknown_to_known_ratio']:5.2f}  "
              f"prop/obj={row['proposals_per_unknown_object']:.2f}")
        print(f"{'':38s}   kNN(within unk)={row['unk_knn_class_agreement']:.4f}  "
              f"kNN(open pool)={row['open_knn_class_agreement']:.4f}  "
              f"NMI={row['unk_nmi_in_full_partition']:.4f}  "
              f"contam={row['known_contamination']:.4f}  "
              f"unk_recall={row['unknown_recall_in_candidate_clusters']:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "population_audit.csv"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
