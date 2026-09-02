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
        if eligible and (best is None or report["contamination"] < best[0]):
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

    **Measured: on the committed pool this is a no-op, and that is a defect, not
    a property.** At K=1600 the smallest k-means cluster holds 5 members, so a
    ``minimum_size=5`` floor closes on **0 of 80,000** candidates and this
    returns the same vector as no gate at all. It is why ``ARMS['consult']`` and
    ``ARMS['consult_no_gate']`` are bitwise identical on every committed seed.

    Kept unchanged so those committed results stay reproducible. For new work use
    :func:`density_coherence`, which fits DBSCAN on a named scope and therefore
    can actually fire.
    """

    gate = np.ones(partition.labels.shape[0], dtype=np.float32)
    gate[partition.is_noise] = 0.0
    small = partition.sizes < minimum_size
    valid = partition.labels >= 0
    gate[valid & small[np.clip(partition.labels, 0, None)]] = 0.0
    return gate

# ------------------------------------------------- density coherence (A1.1) ---


@dataclass(frozen=True)
class CoherenceGate:
    """A binary ``C(x)``: 1 where a candidate has local density support, 0 where not.

    The consultation asked for ``coh(x) in {0, 1}`` via DBSCAN core-vs-noise. Two
    things had to be fixed before that idea could be tested at all.

    **It was a no-op.** The registered ``binary`` coherence applied a minimum
    *cluster size* under k-means, and on the committed pool at K=1600 the
    smallest k-means cluster holds 5 members, so a ``min_samples=5`` floor closed
    on **0 of 80,000** candidates. ``binary`` and ``off`` returned the identical
    vector, which made ``consult`` and ``consult_no_gate`` the same experiment
    run twice rather than a treatment and its control.

    **On the full pool the idea inverts.** DBSCAN noise on all 80,000 proposals
    marks *real objects* as noise more often than background — 92% against 60% at
    eps 0.15 — because the pool is 81% background, background regions are near
    copies of one another, and so background occupies the densest part of the
    space. In this pool "you have many neighbours" means "you look like
    background", which is the opposite of what the gate is for.

    Hence ``scope``: DBSCAN runs on a *population*, and the population is the
    experiment. On the objectness-admissible subpool the object-likeness prior has
    already removed most background, so density can mean what it was meant to
    mean. ``owl.clustering`` takes no position on whether it does; that is H2, and
    ``tools/diagnose_coherence.py`` answers it against a predeclared grid.

    ``labels`` uses ``-1`` for DBSCAN noise and ``-2`` for a candidate outside the
    scope, so the two reasons a gate is closed stay distinguishable in the
    diagnostic. Discrimination must be judged *within* the scope: comparing
    in-scope core points against out-of-scope candidates would measure
    admissibility, not density.
    """

    gate: np.ndarray       # (N,) float32 in {0, 1}
    scope: np.ndarray      # (N,) bool — the population DBSCAN was fitted on
    labels: np.ndarray     # (N,) int — >=0 cluster, -1 noise, -2 outside scope
    params: dict

    @property
    def n_clusters(self) -> int:
        if not (self.labels >= 0).any():
            return 0
        return self.labels.max().item() + 1

    @property
    def is_noise(self) -> np.ndarray:
        """DBSCAN noise, **within the scope only**."""
        return self.labels == -1

    def summary(self) -> dict[str, float]:
        in_scope = int(self.scope.sum())
        return {
            "scope_size": in_scope,
            "clusters": self.n_clusters,
            # criterion 4 is judged on this: does the gate do anything at all?
            "noise_share_within_scope": (
                float(self.is_noise.sum() / in_scope) if in_scope else float("nan")
            ),
            "gate_open_share_pool": float((self.gate > 0).mean()),
        }


def density_coherence(
    embeddings: np.ndarray,
    *,
    scope: np.ndarray | None = None,
    eps: float = 0.25,
    min_samples: int = 5,
    pca_dimensions: int | None = 32,
    seed: int = 0,
) -> CoherenceGate:
    """Binary coherence from DBSCAN core/border-vs-noise on ``scope``.

    ``scope`` is a boolean mask naming the population to fit on; ``None`` means
    the whole pool. A candidate outside the scope gets ``C(x) = 0``: coherence
    here means "has local support *among object-like candidates*", and something
    the object prior would not admit has none by definition.

    Core and border points are both kept — sklearn gives both a label ``>= 0``
    and only noise gets ``-1`` — because a border point of a small real cluster is
    exactly the rare-but-real case the gate must not throw away.
    """

    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = embeddings.shape[0]
    mask = (
        np.ones(n, dtype=bool)
        if scope is None
        else np.asarray(scope, dtype=bool).reshape(n)
    )

    labels = np.full(n, -2, dtype=np.int64)
    gate = np.zeros(n, dtype=np.float32)
    params = {
        "eps": eps,
        "min_samples": min_samples,
        "pca_dimensions": pca_dimensions,
        "scope_size": int(mask.sum()),
        "pool_size": n,
    }
    if not mask.any():
        return CoherenceGate(gate=gate, scope=mask, labels=labels, params=params)

    # PCA is fitted on the scope, not on the pool: the axes should describe the
    # population being clustered. Fitting on all 80,000 proposals would orient
    # them along the background bulk, which is the variance we just excluded.
    features = _reduce(embeddings[mask], pca_dimensions, seed)
    fitted = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(features)

    labels[mask] = fitted
    gate[mask] = (fitted >= 0).astype(np.float32)
    return CoherenceGate(gate=gate, scope=mask, labels=labels, params=params)


def admissible_mask(scores: np.ndarray, share: float) -> np.ndarray:
    """The top ``share`` of the pool by ``scores`` — the coherence scope.

    Ties are broken by a stable sort, so the mask is reproducible. ``share >= 1``
    admits everything, which is how the ``full_pool`` control is expressed
    without a second code path.
    """

    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if share >= 1.0:
        return np.ones(n, dtype=bool)
    keep = max(round(n * float(share)), 1)
    mask = np.zeros(n, dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:keep]] = True
    return mask
