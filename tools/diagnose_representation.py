"""Is there a semantic feature space in this detector that the idea needs?

The 2026-08-25 consultation asked for a representation in which known structure
is explicit, background is not mistaken for novelty, genuine unknown categories
form groups, and rarity is the size of those groups. Day 1 tested one
operationalisation of that and it failed. This tool asks the prior question:
**does the representation contain the structure the idea assumes exists?**

Answering it before designing another selector, because a selector cannot
recover structure the features do not carry.

Oracle labels are used throughout, and only, to score representations
retrospectively. Nothing here feeds acquisition.

**The measurement that makes or breaks the whole approach** is kNN class
agreement among unknown objects with **same-object neighbours excluded**. Without
that exclusion the number is meaningless: 2.51 proposals sit on the average GT
object, so a proposal's nearest neighbour is usually another box on the same
physical thing, and "my neighbour has my class" becomes "my neighbour is me".
That is the proposal-duplication confound, measured rather than assumed.

    python tools/diagnose_representation.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"
POOL = Path(__file__).resolve().parent.parent / "data" / "pool" / "sowodb_t1_frozen_pool.npz"

N_KNOWN_AT_T1 = 19          # PROB's t1 checkpoint knows 19 classes
UNKNOWN_SLOT = 80           # the posterior's unknown column
K_NEIGHBOURS = 10


# --------------------------------------------------------------- the pool ---


def load() -> dict:
    payload = np.load(POOL, allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == "pool"
    raw = payload["embeddings"][keep].astype(np.float32)
    posterior = payload["posterior_q"][keep].astype(np.float32) / 255.0
    posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-12)
    return {
        "raw": raw,
        "posterior": posterior,
        "query_index": payload["query_index"][keep].astype(int),
        "pred_obj": payload["pred_obj"][keep].astype(np.float32),
        "kind": np.asarray(payload["oracle_kind"], dtype=str)[keep],
        "class_name": np.asarray(payload["oracle_class"], dtype=str)[keep],
        "object_id": payload["oracle_object"][keep].astype(np.int64),
        "group": np.asarray(payload["oracle_group"], dtype=str)[keep],
        "image_ids": np.asarray(payload["image_ids"], dtype=str)[keep],
    }


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def representations(pool: dict) -> dict[str, np.ndarray]:
    """Candidates that are already available or one line of algebra away.

    A different decoder layer, or a backbone ROI feature, would need PROB and a
    GPU pass. These four do not, and they span the hypotheses worth separating:
    is the problem the L2 normalisation, the dominant objectness axis, or the
    feature's nature?
    """

    raw = pool["raw"]
    unit = _unit(raw)

    # centred PCA basis, fitted on a sample -- the pool is 80,000 x 256
    sample = np.random.default_rng(0).choice(raw.shape[0], 20000, replace=False)
    centre = unit.mean(axis=0)
    basis = np.linalg.svd(unit[sample] - centre, full_matrices=False)[2]

    # drop PC1: measured to be the objectness axis (rho -0.72 with the raw norm),
    # so this asks whether semantics is hiding underneath it
    without_pc1 = _unit((unit - centre) - np.outer((unit - centre) @ basis[0], basis[0]))

    # whitening equalises the axes, which is the standard remedy when one
    # direction carries 60% of the variance and Euclidean density is wanted
    projected = (unit - centre) @ basis[:32].T
    whitened = _unit(projected / np.maximum(projected.std(axis=0, keepdims=True), 1e-6))

    # the classification-oriented view: PROB's own posterior over the 19 classes
    # trained at t1 plus the unknown slot. Columns 19..79 are untrained at this
    # checkpoint and are excluded rather than fed in as noise. sqrt is the
    # Hellinger map, which makes Euclidean distance a proper distance between
    # distributions instead of an arbitrary one.
    columns = list(range(N_KNOWN_AT_T1)) + [UNKNOWN_SLOT]
    semantic = pool["posterior"][:, columns]
    semantic = _unit(np.sqrt(semantic / np.maximum(semantic.sum(axis=1, keepdims=True), 1e-12)))

    return {
        "unit_embedding": unit,                 # what the codebase uses today
        "embedding_no_pc1": without_pc1,
        "embedding_whitened32": whitened,
        "posterior_hellinger": semantic,
    }


# -------------------------------------------------------------- diagnostics ---


def knn_agreement(
    features: np.ndarray,
    *,
    subset: np.ndarray,
    class_name: np.ndarray,
    object_id: np.ndarray,
    k: int = K_NEIGHBOURS,
    within_subset: bool = True,
) -> dict[str, float]:
    """Does a proposal's neighbour share its class, once its own object is gone?

    ``within_subset`` restricts the neighbour pool to ``subset`` (asking "is
    there class structure here at all") rather than the whole pool (asking
    "would a clustering of everything find it").

    Same-object neighbours are dropped, always. With 2.51 proposals per object
    they would otherwise supply most of the agreement.
    """

    index = np.flatnonzero(subset)
    if index.size < k + 2:
        return {"knn_class_agreement": float("nan"), "same_object_neighbour_share": float("nan")}

    reference = index if within_subset else np.arange(features.shape[0])
    # ask for extra neighbours so that removing same-object ones still leaves k
    n_ask = min(k + 24, reference.size)
    model = NearestNeighbors(n_neighbors=n_ask, n_jobs=-1).fit(features[reference])
    neighbours = model.kneighbors(features[index], return_distance=False)
    neighbours = reference[neighbours]

    hits, considered, same_object = 0, 0, 0
    for row, origin in enumerate(index):
        candidates = neighbours[row]
        candidates = candidates[candidates != origin]
        same = object_id[candidates] == object_id[origin]
        same &= object_id[origin] >= 0
        same_object += int(same.sum())
        candidates = candidates[~same][:k]
        if candidates.size == 0:
            continue
        hits += int((class_name[candidates] == class_name[origin]).sum())
        considered += candidates.size

    return {
        "knn_class_agreement": hits / considered if considered else float("nan"),
        "same_object_neighbour_share": same_object / (index.size * n_ask),
    }


def separability(features: np.ndarray, positive: np.ndarray, negative: np.ndarray,
                 *, seed: int = 0) -> float:
    """AUC of a nearest-class-mean score. A crude probe, but it needs no fitting.

    Uses held-out halves so the prototype is not built from the points it scores.
    """

    generator = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        pos = np.flatnonzero(positive)
        neg = np.flatnonzero(negative)
        if pos.size < 20 or neg.size < 20:
            return float("nan")
        pos_fit = generator.permutation(pos)[: pos.size // 2]
        neg_fit = generator.permutation(neg)[: neg.size // 2]
        pos_mean = features[pos_fit].mean(axis=0)
        neg_mean = features[neg_fit].mean(axis=0)
        held = np.concatenate([np.setdiff1d(pos, pos_fit), np.setdiff1d(neg, neg_fit)])
        label = np.concatenate([
            np.ones(np.setdiff1d(pos, pos_fit).size),
            np.zeros(np.setdiff1d(neg, neg_fit).size),
        ])
        score = features[held] @ (pos_mean - neg_mean)
        out.append(roc_auc_score(label, score))
    return float(np.mean(out))


def pair_similarity(features: np.ndarray, pool: dict, *, seed: int = 0,
                    pairs: int = 20000) -> dict[str, float]:
    """Three cosine similarities that say what the space actually encodes.

    If same-object similarity is high while same-class-different-object is no
    higher than different-class, the space encodes *this box* and not *this kind
    of thing* -- and a class-level clustering of it cannot work.
    """

    generator = np.random.default_rng(seed)
    unknown = np.flatnonzero(pool["kind"] == "unknown")
    object_id, class_name = pool["object_id"], pool["class_name"]

    by_object: dict[int, list[int]] = {}
    for position in unknown:
        by_object.setdefault(int(object_id[position]), []).append(int(position))

    same_object = [
        (members[0], members[1])
        for members in by_object.values() if len(members) >= 2
    ]

    def sample(items: list, count: int) -> list:
        if not items:
            return []
        picks = generator.choice(len(items), size=min(count, len(items)), replace=False)
        return [items[i] for i in picks]

    def mean_similarity(couples: list) -> float:
        if not couples:
            return float("nan")
        left = features[[a for a, _ in couples]]
        right = features[[b for _, b in couples]]
        return float((left * right).sum(axis=1).mean())

    # same class, different object
    same_class: list[tuple[int, int]] = []
    different_class: list[tuple[int, int]] = []
    for _ in range(pairs):
        a, b = generator.choice(unknown, size=2, replace=False)
        if object_id[a] == object_id[b]:
            continue
        (same_class if class_name[a] == class_name[b] else different_class).append((int(a), int(b)))

    return {
        "sim_same_object": mean_similarity(sample(same_object, pairs)),
        "sim_same_class_other_object": mean_similarity(sample(same_class, pairs)),
        "sim_different_class": mean_similarity(sample(different_class, pairs)),
    }


def cluster_quality(features: np.ndarray, pool: dict, *, subset: np.ndarray,
                    n_clusters: int, seed: int = 0) -> dict[str, float]:
    """NMI/ARI of a k-means partition against true unknown class, on ``subset``."""

    index = np.flatnonzero(subset)
    if index.size < n_clusters * 2:
        return {"nmi": float("nan"), "ari": float("nan"), "cluster_purity": float("nan")}
    labels = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, n_init=3, batch_size=4096
    ).fit_predict(features[index])
    truth = pool["class_name"][index]

    purity = 0
    for value in np.unique(labels):
        members = truth[labels == value]
        if members.size:
            purity += np.bincount(np.unique(members, return_inverse=True)[1]).max()
    return {
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "ari": float(adjusted_rand_score(truth, labels)),
        "cluster_purity": float(purity / index.size),
    }


def variance_explained(labels: np.ndarray, features: np.ndarray) -> float:
    """Between-group share of total sum of squares -- eta^2, all dims pooled."""

    total = float(((features - features.mean(axis=0)) ** 2).sum())
    if total <= 0:
        return float("nan")
    grand = features.mean(axis=0)
    between = 0.0
    for value in np.unique(labels):
        mask = labels == value
        if mask.sum() < 2:
            continue
        between += mask.sum() * float(((features[mask].mean(axis=0) - grand) ** 2).sum())
    return between / total


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=int, default=120)
    arguments = parser.parse_args()

    pool = load()
    kind, class_name = pool["kind"], pool["class_name"]
    known, unknown, background = kind == "known", kind == "unknown", kind == "background"
    tail = pool["group"] == "tail"

    print(f"pool {pool['raw'].shape[0]:,} proposals: "
          f"{background.sum():,} background / {known.sum():,} known / {unknown.sum():,} unknown")
    print(f"distinct unknown objects: {np.unique(pool['object_id'][unknown]).size}\n")

    rows: list[dict] = []
    for name, features in representations(pool).items():
        print(f"--- {name} (dim {features.shape[1]}) ---")
        row: dict[str, object] = {"representation": name, "dim": int(features.shape[1])}

        # what dominates the space
        row["eta2_query_index"] = variance_explained(pool["query_index"], features)
        row["eta2_oracle_kind"] = variance_explained(kind, features)
        row["eta2_known_class"] = variance_explained(class_name[known], features[known])
        row["eta2_unknown_class"] = variance_explained(class_name[unknown], features[unknown])

        # separability probes
        row["auc_object_vs_background"] = separability(features, known | unknown, background)
        row["auc_unknown_vs_background"] = separability(features, unknown, background)
        row["auc_unknown_vs_known"] = separability(features, unknown, known)

        # the decisive measurement, same-object neighbours removed
        row |= {f"known_{k}": v for k, v in knn_agreement(
            features, subset=known, class_name=class_name,
            object_id=pool["object_id"]).items()}
        row |= {f"unknown_{k}": v for k, v in knn_agreement(
            features, subset=unknown, class_name=class_name,
            object_id=pool["object_id"]).items()}
        row |= {f"unknown_tail_{k}": v for k, v in knn_agreement(
            features, subset=unknown & tail, class_name=class_name,
            object_id=pool["object_id"]).items()}
        row |= {f"unknown_openpool_{k}": v for k, v in knn_agreement(
            features, subset=unknown, class_name=class_name,
            object_id=pool["object_id"], within_subset=False).items()}

        row |= pair_similarity(features, pool)
        row |= {f"unknown_{k}": v for k, v in cluster_quality(
            features, pool, subset=unknown, n_clusters=arguments.clusters).items()}

        rows.append(row)
        print(f"    eta2: query={row['eta2_query_index']:.4f} kind={row['eta2_oracle_kind']:.4f} "
              f"known_cls={row['eta2_known_class']:.4f} unk_cls={row['eta2_unknown_class']:.4f}")
        print(f"    AUC:  obj/bg={row['auc_object_vs_background']:.4f} "
              f"unk/bg={row['auc_unknown_vs_background']:.4f} unk/known={row['auc_unknown_vs_known']:.4f}")
        print("    kNN class agreement (same object EXCLUDED):")
        print(f"          known  {row['known_knn_class_agreement']:.4f}   "
              f"unknown {row['unknown_knn_class_agreement']:.4f}   "
              f"unknown-tail {row['unknown_tail_knn_class_agreement']:.4f}   "
              f"unknown-in-open-pool {row['unknown_openpool_knn_class_agreement']:.4f}")
        print(f"    similarity: same_object={row['sim_same_object']:.4f} "
              f"same_class={row['sim_same_class_other_object']:.4f} "
              f"diff_class={row['sim_different_class']:.4f}")
        print(f"    unknown k-means (K={arguments.clusters}): NMI={row['unknown_nmi']:.4f} "
              f"ARI={row['unknown_ari']:.4f} purity={row['unknown_cluster_purity']:.4f}\n")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "representation_audit.csv"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")

    # the reference point every agreement number must beat
    unknown_classes, counts = np.unique(class_name[unknown], return_counts=True)
    chance = float(((counts / counts.sum()) ** 2).sum())
    print(f"\nchance kNN class agreement among unknowns = {chance:.4f} "
          f"({unknown_classes.size} classes, frequency-weighted)")


if __name__ == "__main__":
    main()
