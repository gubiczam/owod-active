"""The four terms of the selection score, and the score itself.

The research plan's equation::

    s(x) = U(x) + lambda * D(x) + gamma * w(c_hat(x)) * coh(x)

The 2026-08-25 consultation changed what three of the four terms mean. Each
change is a registered option here, so the old and the new definition can be
run one variable apart:

======  ==========================  ==========================================
term    plan / previous code        consultation
======  ==========================  ==========================================
``U``   entropy of the posterior    unchanged -- "az entrópia ötlet jó"
``D``   distance to a *fixed*       distance to the **growing labelled pool**,
        task-1 anchor               plus a batch-diversity part that updates
                                    while the batch is being chosen
``w``   1 / size of a k-means       size of the candidate's cluster in the
        pseudo-class                *single* partition that also produces ``D``
``coh`` continuous local density    **binary gate**: 0 for an unsupported
                                    candidate, 1 otherwise
======  ==========================  ==========================================

Every term is a function of detector outputs and of what has already been
labelled. None of them reads an annotation for a candidate that has not been
paid for; :func:`terms` takes a :class:`~owl.proposals.Candidates` and never
calls ``.oracle()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from owl import clustering
from owl.proposals import Candidates

# --------------------------------------------------------------- utilities ---


def rank_normalise(values: np.ndarray) -> np.ndarray:
    """Map to [0, 1] by rank, ties averaged.

    Every additive term is rank-normalised so a weight means the same thing for
    all of them. A raw entropy and a raw inverse-frequency are not on the same
    scale and adding them with fixed weights would silently weight one of them
    to nothing.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    # average ties so equal inputs get equal output
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return ranks / max(values.size - 1, 1)


def _max_similarity(query: np.ndarray, reference: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Highest cosine similarity from each query row to any reference row."""

    if reference.size == 0:
        return np.zeros(query.shape[0], dtype=np.float32)
    best = np.full(query.shape[0], -1.0, dtype=np.float32)
    for start in range(0, query.shape[0], chunk):
        stop = start + chunk
        best[start:stop] = (query[start:stop] @ reference.T).max(axis=1)
    return best


# ------------------------------------------------------------------- terms ---


def uncertainty(candidates: Candidates, method: str = "entropy") -> np.ndarray:
    """``U(x)`` — how unsure the detector is about this region.

    ``entropy``     normalised Shannon entropy of the class posterior. The
                    consultation kept this one unchanged.
    ``objectness``  the learning-free control: ``objectness * sqrt(area)``.
                    It does no distribution modelling at all and is the bar the
                    semantic terms have to clear.
    """

    if method == "entropy":
        posterior = np.clip(candidates.posterior, 1e-12, 1.0)
        entropy = -(posterior * np.log(posterior)).sum(axis=1)
        return entropy / np.log(posterior.shape[1])
    if method == "objectness":
        return candidates.objectness * np.sqrt(np.maximum(candidates.area, 0.0))
    raise ValueError(f"Unknown uncertainty method {method!r}.")


def novelty(candidates: Candidates, labelled: np.ndarray) -> np.ndarray:
    """``D_a(x)`` — distance to the **growing labelled pool**.

    This is the consultation's correction. The previous code measured distance
    to a fixed task-1 anchor, which answers "does this look like the training
    set" and stops changing. Measuring against everything labelled so far
    answers the question that actually matters — *is this new to us* — and it
    moves every time the annotator answers.

    ``labelled`` is an ``(M, D)`` array of unit-norm embeddings of what has been
    labelled. Empty means everything is maximally novel.
    """

    labelled = np.asarray(labelled, dtype=np.float32).reshape(-1, candidates.embeddings.shape[1])
    if labelled.shape[0] == 0:
        return np.ones(len(candidates), dtype=np.float32)
    return 1.0 - _max_similarity(candidates.embeddings, labelled)


def cluster_novelty(partition: clustering.Partition, known_clusters: np.ndarray) -> np.ndarray:
    """``D_a(x)`` read off the shared partition instead of off raw neighbours.

    Distance from the candidate's own cluster centroid to the nearest centroid
    of a cluster that holds known content. Cheaper than :func:`novelty` and it
    is the version that uses the consultation's "one clustering, both terms"
    structure. ``known_clusters`` is a boolean mask over clusters.
    """

    if not known_clusters.any():
        return np.ones(partition.labels.shape[0], dtype=np.float32)
    similarity = partition.centroids @ partition.centroids[known_clusters].T
    per_cluster = 1.0 - similarity.max(axis=1)
    out = np.ones(partition.labels.shape[0], dtype=np.float32)
    valid = partition.labels >= 0
    out[valid] = per_cluster[partition.labels[valid]]
    return out


def rarity(partition: clustering.Partition, method: str = "log_inverse") -> np.ndarray:
    """``w(c_hat(x))`` — how rare the candidate's estimated class is.

    The estimated class is the candidate's cluster in the shared partition, so
    this term and :func:`cluster_novelty` come from one structure, which is what
    the consultation asked for. A giant cluster is the background blob and gets
    almost no weight; a small but populated cluster is a rare class and gets a
    lot.

    ``log_inverse``  ``-log(n_c / N)`` — the smooth version, and the default.
    ``inverse``      ``1 / n_c`` — the plan's own wording, much sharper.
    """

    sizes = partition.size_of()
    total = max(float(partition.labels.shape[0]), 1.0)
    if method == "log_inverse":
        return -np.log(np.maximum(sizes, 1.0) / total)
    if method == "inverse":
        return 1.0 / np.maximum(sizes, 1.0)
    raise ValueError(f"Unknown rarity method {method!r}.")


def coherence(
    candidates: Candidates,
    partition: clustering.Partition,
    *,
    method: str = "binary",
    minimum_size: int = 5,
    k: int = 5,
) -> np.ndarray:
    """``coh(x)`` — does the candidate have local support, or is it alone.

    ``binary``      the consultation's gate: 0 or 1, nothing in between. A
                    candidate with no cluster support is not something we want
                    to learn, so its rarity contribution is switched off
                    entirely rather than merely reduced.
    ``continuous``  the previous definition, ``1 / (1 + d_k / median(d_k))``.
                    Kept so the two can be compared one variable apart.
    ``off``         constant 1 — the ungated control, which is what isolates
                    what the gate itself does.

    **Measured warning.** On PROB's own decoder features the gate removes real
    unknown objects more often than it removes background, because in a pool
    that is four-fifths background the densest region *is* background. Running
    ``binary`` against ``off`` is the experiment; assuming the gate helps is
    not supported here.
    """

    if method == "off":
        return np.ones(len(candidates), dtype=np.float32)
    if method == "binary":
        return clustering.noise_gate(partition, minimum_size=minimum_size)
    if method == "continuous":
        return _continuous_coherence(candidates, k)
    raise ValueError(f"Unknown coherence method {method!r}.")


#: Continuous coherence depends only on the pool's own geometry, never on what has
#: been labelled, so it is computed once per pool rather than once per round. A
#: twelve-round campaign would otherwise pay for the same k-NN graph twelve times.
_COHERENCE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _continuous_coherence(candidates: Candidates, k: int) -> np.ndarray:
    key = (id(candidates.embeddings), k)
    cached = _COHERENCE_CACHE.get(key)
    if cached is not None:
        return cached

    from sklearn.neighbors import NearestNeighbors

    embeddings = candidates.embeddings
    neighbours = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(embeddings)
    distances = neighbours.kneighbors(embeddings, return_distance=True)[0][:, -1]
    median = float(np.median(distances)) or 1.0
    value = np.clip(1.0 / (1.0 + distances / median), 0.0, 1.0).astype(np.float32)
    _COHERENCE_CACHE[key] = value
    return value


# ------------------------------------------------------------------- score ---


@dataclass(frozen=True)
class ScoreConfig:
    """Everything that defines one arm of the comparison.

    The plan left ``lambda`` and ``gamma`` open. They are fixed here once and
    swept in the notebook; no value is chosen by looking at an endpoint.
    """

    name: str = "plan"
    lambda_diversity: float = 0.2      # weight on D
    gamma_rarity: float = 0.5          # weight on w * coh
    mu_batch: float = 0.0              # weight on batch diversity, 0 = off
    uncertainty_method: str = "entropy"
    diversity_source: str = "labelled"  # 'labelled' | 'clusters' | 'anchor_free'
    rarity_method: str = "log_inverse"
    coherence_method: str = "binary"    # 'binary' | 'continuous' | 'off'
    combination: str = "additive"       # 'additive' | 'multiplicative'
    coherence_minimum_size: int = 5
    cluster_method: str = "kmeans"
    n_clusters: int = 1600
    random: bool = False               # the random baseline short-circuits everything
    seed: int = 0


@dataclass
class Terms:
    """The four terms, already rank-normalised, plus what selection needs later."""

    uncertainty: np.ndarray
    diversity: np.ndarray
    rarity: np.ndarray
    coherence: np.ndarray
    partition: clustering.Partition
    config: ScoreConfig = field(repr=False, default_factory=ScoreConfig)

    #: The learning-free object-likeness factor, kept raw. Rank-normalising a
    #: multiplier would defeat its purpose: the point of ``P`` is that a region
    #: which does not look like an object cannot be rescued by any other term.
    prior: np.ndarray | None = None

    def combine(self, batch_diversity: np.ndarray | None = None) -> np.ndarray:
        """``s(x)``. ``batch_diversity`` is supplied by the greedy selector."""

        semantic = (
            self.config.lambda_diversity * self.diversity
            + self.config.gamma_rarity * self.rarity * self.coherence
        )
        if batch_diversity is not None and self.config.mu_batch:
            semantic = semantic + self.config.mu_batch * batch_diversity

        if self.config.combination == "multiplicative":
            if self.prior is None:
                raise ValueError("A multiplicative score needs the object-likeness prior.")
            return self.prior * (1.0 + semantic)
        return self.uncertainty + semantic

    def table(self) -> dict[str, float]:
        return {
            "mean_U": float(self.uncertainty.mean()),
            "mean_D": float(self.diversity.mean()),
            "mean_w": float(self.rarity.mean()),
            "gate_open_share": float((self.coherence > 0).mean()),
        }


def terms(
    candidates: Candidates,
    config: ScoreConfig,
    *,
    labelled_embeddings: np.ndarray | None = None,
    n_known: int = 19,
    partition: clustering.Partition | None = None,
) -> Terms:
    """Compute the four terms for one round. No annotation is read."""

    if partition is None:
        partition = clustering.fit(
            candidates.embeddings,
            method=config.cluster_method,
            n_clusters=config.n_clusters,
            seed=config.seed,
        )

    u = rank_normalise(uncertainty(candidates, config.uncertainty_method))

    if config.diversity_source == "clusters":
        is_known = clustering.predicted_known(candidates.posterior, n_known)
        counts = np.bincount(
            partition.labels[partition.labels >= 0], minlength=partition.n_clusters
        )
        known_counts = np.bincount(
            partition.labels[(partition.labels >= 0) & is_known],
            minlength=partition.n_clusters,
        )
        baseline = float(is_known.mean())
        known_clusters = (known_counts / np.maximum(counts, 1)) > baseline
        d = cluster_novelty(partition, known_clusters)
    elif config.diversity_source == "labelled":
        reference = (
            np.zeros((0, candidates.embeddings.shape[1]), dtype=np.float32)
            if labelled_embeddings is None
            else labelled_embeddings
        )
        d = novelty(candidates, reference)
    elif config.diversity_source == "anchor_free":
        d = np.zeros(len(candidates), dtype=np.float32)
    else:
        raise ValueError(f"Unknown diversity source {config.diversity_source!r}.")

    return Terms(
        prior=uncertainty(candidates, "objectness"),
        uncertainty=u,
        diversity=rank_normalise(d),
        rarity=rank_normalise(rarity(partition, config.rarity_method)),
        coherence=coherence(
            candidates,
            partition,
            method=config.coherence_method,
            minimum_size=config.coherence_minimum_size,
        ),
        partition=partition,
        config=config,
    )
