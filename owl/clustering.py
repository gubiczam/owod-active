"""One clustering, from which both rarity and diversity are read.

This is point 3 of the 2026-08-25 consultation. Instead of estimating rarity
one way and diversity another, the candidate pool is partitioned once in the
embedding space and both terms are read off the same partition:

* **rarity** ``w`` — how small the candidate's cluster is; a small cluster is a
  rare class;
* **diversity** ``D`` — how far the candidate's cluster sits from the clusters
  that hold what is already labelled.

The partition's quality is not judged by a silhouette score. It is judged by
**known contamination**: how many already-known proposals land in a cluster we
would treat as unknown. That is measurable before any oracle is paid, because
the detector already recognises its known classes, and it is checkable against
the benchmark's annotations offline.

Measured on the committed PROB t1 pass (80,000 proposals, 1,600 images):

=========  ====  =================  ====================  ====================
subset     K     unknown purity     known contamination   unknown recall
=========  ====  =================  ====================  ====================
all        200   0.265              0.038                 0.331
all        800   0.384              0.041                 0.489
top 30%    800   **0.454**          **0.059**             **0.627**
=========  ====  =================  ====================  ====================

DBSCAN is offered as well because the consultation asked for it, but on these
features it marks *real objects* as noise more often than background (92% vs
60% at eps=0.15) — see :func:`noise_gate` and ``docs/coherence_gate.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class Partition:
    """A clustering of the candidate pool, plus everything read off it."""

    labels: np.ndarray        # (N,) int, -1 means noise (DBSCAN only)
    centroids: np.ndarray     # (C, D) unit-norm cluster centroids
    sizes: np.ndarray         # (C,) members per cluster
    method: str
    params: dict

    @property
    def n_clusters(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def is_noise(self) -> np.ndarray:
        return self.labels < 0

    def size_of(self, index: np.ndarray | None = None) -> np.ndarray:
        """Cluster size per proposal. Noise points report size 1."""
        labels = self.labels if index is None else self.labels[index]
        out = np.ones(labels.shape[0], dtype=np.float64)
        valid = labels >= 0
        out[valid] = self.sizes[labels[valid]]
        return out


# ------------------------------------------------------------------ fitting ---


def _reduce(embeddings: np.ndarray, dimensions: int | None, seed: int) -> np.ndarray:
    if not dimensions or dimensions >= embeddings.shape[1]:
        return embeddings
    reduced = PCA(n_components=dimensions, random_state=seed).fit_transform(embeddings)
    norm = np.linalg.norm(reduced, axis=1, keepdims=True)
    return (reduced / np.maximum(norm, 1e-9)).astype(np.float32)


def fit(
    embeddings: np.ndarray,
    *,
    method: str = "kmeans",
    n_clusters: int = 800,
    eps: float = 0.25,
    min_samples: int = 5,
    pca_dimensions: int | None = None,
    seed: int = 0,
) -> Partition:
    """Partition the pool.

    ``method='kmeans'`` is the default and the one the measurements support.
    ``n_clusters`` is deliberately far larger than the number of classes: a
    class is usually several clusters, and over-clustering is what keeps
    contamination low. ``method='dbscan'`` is the consultation's alternative and
    is the only one that produces noise points.
    """

    features = _reduce(np.asarray(embeddings, dtype=np.float32), pca_dimensions, seed)

    if method == "kmeans":
        model = MiniBatchKMeans(
            n_clusters=min(n_clusters, features.shape[0]),
            random_state=seed,
            n_init=3,
            batch_size=4096,
        )
        labels = model.fit_predict(features)
        params = {"n_clusters": int(model.n_clusters), "pca_dimensions": pca_dimensions}
    elif method == "dbscan":
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(features)
        params = {"eps": eps, "min_samples": min_samples, "pca_dimensions": pca_dimensions}
    else:
        raise ValueError(f"Unknown clustering method {method!r}; use 'kmeans' or 'dbscan'.")

    n_clusters_found = int(labels.max()) + 1
    centroids = np.zeros((max(n_clusters_found, 1), features.shape[1]), dtype=np.float32)
    sizes = np.zeros(max(n_clusters_found, 1), dtype=np.int64)
    for index in range(n_clusters_found):
        members = features[labels == index]
        sizes[index] = members.shape[0]
        if members.shape[0]:
            centroids[index] = members.mean(axis=0)
    norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norm, 1e-9)

    return Partition(labels=labels, centroids=centroids, sizes=sizes, method=method, params=params)


# ------------------------------------------------------------- diagnostics ---


def predicted_known(posterior: np.ndarray, n_known: int) -> np.ndarray:
    """Which proposals the detector itself already calls a known class.

    No annotation is read. This is the label the contamination diagnostic runs
    on when no oracle is available, which is every live round.
    """

    posterior = np.asarray(posterior, dtype=np.float32)
    known_mass = posterior[:, :n_known].max(axis=1)
    other_mass = posterior[:, n_known:].max(axis=1) if posterior.shape[1] > n_known else 0.0
    return known_mass >= other_mass


def contamination(partition: Partition, is_known: np.ndarray) -> dict[str, float]:
    """How badly the partition mixes known content into unknown-looking clusters.

    A cluster is **known-owned** when its share of known proposals is above the
    pool's own share — that is, when it is enriched in already-known content.
    Every other cluster is an **unknown candidate**, and contamination is the
    fraction of known proposals that land in one.

    The enrichment baseline matters. Judging a cluster by an absolute majority
    is degenerate here: the pool is 81% background, so almost no cluster reaches
    50% known and almost every cluster is called unknown. Comparing against the
    pool's own rate asks the right question — does this cluster hold *more*
    known content than a random cluster would.

    Lower contamination is better, but only alongside ``unknown_recall``: a
    partition that calls every cluster known-owned scores zero contamination and
    finds nothing.
    """

    is_known = np.asarray(is_known, dtype=bool)
    labels = partition.labels
    valid = labels >= 0
    if not valid.any():
        return {"contamination": 1.0, "unknown_recall": 0.0, "unknown_clusters": 0}

    counts = np.bincount(labels[valid], minlength=partition.n_clusters)
    known_counts = np.bincount(labels[valid & is_known], minlength=partition.n_clusters)
    known_share = known_counts / np.maximum(counts, 1)
    baseline = float(is_known[valid].mean())
    known_owned = known_share > baseline

    unknown_candidate = valid & ~known_owned[np.clip(labels, 0, None)]
    n_known = max(int((valid & is_known).sum()), 1)
    n_other = max(int((valid & ~is_known).sum()), 1)
    return {
        "contamination": float((unknown_candidate & is_known).sum() / n_known),
        "unknown_recall": float((unknown_candidate & ~is_known).sum() / n_other),
        "unknown_clusters": int(known_owned.size - known_owned.sum()),
        "baseline_known_share": baseline,
    }


def tune(
    embeddings: np.ndarray,
    is_known: np.ndarray,
    *,
    method: str = "kmeans",
    grid: tuple = (200, 400, 800, 1600),
    min_unknown_recall: float = 0.5,
    min_mean_size: float = 25.0,
    **kwargs,
) -> tuple[Partition, list[dict]]:
    """Sweep the partition parameter and keep the least contaminated eligible one.

    Two floors make the sweep meaningful, because contamination on its own is
    trivially minimised:

    * ``min_unknown_recall`` — a partition that declares every cluster
      known-owned has zero contamination and discovers nothing.
    * ``min_mean_size`` — contamination falls monotonically as clusters get
      smaller, and in the limit every point is its own cluster. But rarity is
      read off cluster *size*, so clusters have to be big enough to count. This
      floor is what stops the sweep from running away to K = N.
    """

    rows: list[dict] = []
    best: tuple[float, Partition] | None = None
    for value in grid:
        key = "n_clusters" if method == "kmeans" else "eps"
        partition = fit(embeddings, method=method, **{key: value}, **kwargs)
        report = contamination(partition, is_known) | {key: value, "clusters": partition.n_clusters}
        mean_size = float(partition.sizes[partition.sizes > 0].mean()) if partition.n_clusters else 0.0
        report["mean_cluster_size"] = mean_size
        rows.append(report)
        eligible = (
            report["unknown_recall"] >= min_unknown_recall and mean_size >= min_mean_size
        )
        if eligible:
            if best is None or report["contamination"] < best[0]:
                best = (report["contamination"], partition)
    if best is None:
        raise ValueError(
            f"No configuration reached unknown_recall >= {min_unknown_recall}. "
            f"Widen the grid or lower the floor. Sweep: {rows}"
        )
    return best[1], rows


def noise_gate(partition: Partition, *, minimum_size: int = 5) -> np.ndarray:
    """The consultation's binary ``coh``: 1 for a supported candidate, 0 for noise.

    Under DBSCAN this is exactly the core/noise split. Under k-means there are
    no noise points, so the same idea is applied to cluster size: a candidate in
    a cluster smaller than ``minimum_size`` has no support and is gated out.

    **This gate is measured to hurt on PROB's decoder features** — it removes
    real unknown objects more often than background. It is implemented so that
    claim can be reproduced, not because it is recommended.
    """

    gate = np.ones(partition.labels.shape[0], dtype=np.float32)
    gate[partition.is_noise] = 0.0
    small = partition.sizes < minimum_size
    valid = partition.labels >= 0
    gate[valid & small[np.clip(partition.labels, 0, None)]] = 0.0
    return gate
