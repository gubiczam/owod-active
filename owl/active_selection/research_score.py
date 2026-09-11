"""The research plan's own score, ``s(x) = U + λ·D + γ·w·coh``, on the GPU path.

Protocol document: ``docs/full_owod_v2_protocol.md``.

**Why this module exists.** The repository already had two faithful pieces and
no path that joined them. :mod:`owl.scoring` and :mod:`owl.selection` implement
the plan's equation with every term the 2026-08-25 consultation asked for, but
they only ever ran on the frozen CPU pool through :func:`owl.runner.simulate`.
The sequential GPU chain runs :mod:`owl.active_selection.arms`, whose registered
arms are a *different* method family — three static rankings, two farthest-first
traversals and a cluster-quota allocator. So the arm that trains a detector was
never the arm the plan describes. This module is the missing one, written
against the benchmark's own :class:`~owl.active_selection.population.Population`
and its DINOv2 feature matrix so it can be an arm like any other.

**Nothing here is chosen by looking at an outcome.** ``λ = 0.2`` and ``γ = 0.5``
are lifted unchanged from :class:`owl.scoring.ScoreConfig`, where they were
fixed before any endpoint was inspected and have never been swept.
``min_samples`` is :func:`owl.active_selection.allocation.min_cluster_size`,
already derived from the answer budget. ``eps`` is derived from the candidate
set's own geometry by :func:`eps_from_geometry` and is *recomputed per round*,
so it is a property of the population and not a number a person picked.

The four terms, and what each mode means
========================================

``U(x)`` — unchanged, as the consultation asked. Normalised Shannon entropy of
PROB's class posterior, rank-normalised over the round's eligible candidates.

``D(x)`` — :data:`DIVERSITY_MODES`.

``labeled_novelty``
    cosine distance to the nearest row of the **labelled reference**, which is
    the balanced task-1 reference plus everything this trajectory has bought,
    and which therefore grows every round and every task. This is the
    consultation's complaint about the old fixed task-1 anchor, answered: at t3
    the distance is measured against what has actually been taught, not against
    a frozen t1 export. Computed from embeddings only — no oracle class enters.
``batch_diversity``
    cosine distance to the nearest candidate **already taken in this round**,
    initialised to 1.0 and lowered after every pick. Greedy farthest-first in
    spirit: like ``k-means++`` it re-scores the survivors against what was just
    chosen, but it is *not* ``k-means++`` — there is no ``D^2`` sampling, no
    randomness and no centroid, so it is named for what it does.
``combined``
    ``0.5·(D_labeled + D_batch)``. Both components are on ``[0, 1]``, both are
    logged separately for every pick, and the equal weighting introduces no
    parameter that could be tuned.
``none``
    ``D ≡ 0``. The ablation that isolates what the term is worth.

``coh(x)`` — :data:`COHERENCE_MODES`.

``dbscan_binary``
    the consultation's own proposal: ``coh ∈ {0, 1}``, ``0`` for a DBSCAN noise
    point and ``1`` for a candidate in a cluster. Core and border points both
    pass — a border point of a small real cluster is exactly the rare-but-real
    case the gate must not throw away — and the log records which it was, so the
    distinction is measurable rather than assumed.
``continuous``
    the plan's original wording, "the inverse of the k-th nearest neighbour
    distance", kept as the baseline the binary gate is one variable away from.
    **Not deleted:** the repository has a measured negative result for a DBSCAN
    gate on PROB's own decoder space
    (``docs/konzultacio_2026-08-25_lefedettseg.md`` §2) and an ablation is the
    only thing that says whether DINOv2 space behaves differently.
``none``
    ``coh ≡ 1``, i.e. the gate is open. Not the same as ``γ = 0``: the rarity
    term still acts, which is what isolates the *gate* from the *weight*.

``w(ĉ(x))`` — :data:`RARITY_MODES`. ``cluster_rarity`` reads
``log((1 + n_cand) / (1 + n_ref))`` on the candidate's own cluster, positive
part, from the **same** partition the coherence gate produced. That is the
consultation's "one clustering, and both ``D`` and rarity fall out of it".
``n_ref`` counts labelled-reference rows whose nearest medoid is that cluster,
so rarity means *under-represented in what is already labelled* — never a true
class frequency. :func:`known_contamination` is the separate, post-hoc check
that the partition actually separates new structure, and it is a diagnostic:
nothing in the selection path calls it.

What is deliberately *not* here
===============================

The detector is not re-run between rounds. Rounds recompute the **acquisition
score** — the clustering, the rarity, the reference and both diversity
components — against a labelled pool that grew by what the previous round
bought. Detector retraining stays where the chain puts it, once per task, and
the two are separate knobs on purpose: the consultation asked for the former and
said nothing about the latter, and conflating them would price a cheap
experiment as an expensive one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from owl import clustering, scoring
from owl.active_selection import allocation
from owl.active_selection import budget as ledger_module

DIVERSITY_MODES: tuple[str, ...] = (
    "none", "labeled_novelty", "batch_diversity", "combined",
)
COHERENCE_MODES: tuple[str, ...] = ("none", "continuous", "dbscan_binary")
RARITY_MODES: tuple[str, ...] = ("none", "cluster_rarity")

#: The plan left ``λ`` and ``γ`` open. These are :class:`owl.scoring.ScoreConfig`'s
#: values, fixed on 2026-08 before any endpoint was read and never swept. They
#: are imported as constants rather than restated so the two paths cannot drift.
LAMBDA_DIVERSITY = 0.2
GAMMA_RARITY = 0.5

#: The quantile of the k-distance distribution that becomes ``eps``. Fixed at
#: the median a priori, and the reason is a statement about the *consequence*
#: rather than a preference: at ``eps = median(k-dist)`` roughly half the
#: candidates have at least ``min_samples`` neighbours inside ``eps`` and become
#: core points, so the gate rejects approximately the sparse half of the
#: population. Any other value would have to be argued for from an outcome,
#: which is what this protocol forbids. Exposed only so an ablation can move it.
EPS_QUANTILE = 0.50

#: PCA target for the gate and the partition. Fixed at the value
#: ``owl.clustering.density_coherence`` already used, for the reason given
#: there: DBSCAN's radius means nothing at 768 dimensions, and the axes must
#: describe the population being clustered.
PCA_DIMENSIONS = 32


class ScoreError(ValueError):
    """Raised when the score is asked for something it was not given."""


def _unit(values: np.ndarray) -> np.ndarray:
    """L2-normalise rows, leaving a zero row alone rather than dividing by zero."""

    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-9)


@dataclass(frozen=True)
class ScoreSpec:
    """One configuration of the plan's equation. Everything a run may vary."""

    diversity_mode: str = "combined"
    coherence_mode: str = "dbscan_binary"
    rarity_mode: str = "cluster_rarity"
    lambda_diversity: float = LAMBDA_DIVERSITY
    gamma_rarity: float = GAMMA_RARITY
    eps_quantile: float = EPS_QUANTILE
    pca_dimensions: int | None = PCA_DIMENSIONS
    #: k for the ``continuous`` coherence mode's k-th nearest neighbour.
    coherence_k: int = 10

    def __post_init__(self) -> None:
        if self.diversity_mode not in DIVERSITY_MODES:
            raise ScoreError(
                f"diversity_mode={self.diversity_mode!r}; expected one of "
                f"{DIVERSITY_MODES}")
        if self.coherence_mode not in COHERENCE_MODES:
            raise ScoreError(
                f"coherence_mode={self.coherence_mode!r}; expected one of "
                f"{COHERENCE_MODES}")
        if self.rarity_mode not in RARITY_MODES:
            raise ScoreError(
                f"rarity_mode={self.rarity_mode!r}; expected one of "
                f"{RARITY_MODES}")
        if not 0.0 < self.eps_quantile < 1.0:
            raise ScoreError(f"eps_quantile must be in (0, 1), got {self.eps_quantile}")

    def as_dict(self) -> dict[str, object]:
        return {
            "diversity_mode": self.diversity_mode,
            "coherence_mode": self.coherence_mode,
            "rarity_mode": self.rarity_mode,
            "lambda_diversity": self.lambda_diversity,
            "gamma_rarity": self.gamma_rarity,
            "eps_quantile": self.eps_quantile,
            "pca_dimensions": self.pca_dimensions,
            "coherence_k": self.coherence_k,
        }


@dataclass(frozen=True)
class Gate:
    """The partition and the binary coherence read off it."""

    labels: np.ndarray        # (n,) int, -1 is DBSCAN noise
    status: np.ndarray        # (n,) str, 'core' | 'border' | 'noise'
    coherence: np.ndarray     # (n,) float in {0.0, 1.0}
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchSelection:
    """What the score bought at one task, and the trail it left."""

    images: tuple[str, ...]
    anchors: tuple[int, ...]
    covered: np.ndarray               # (n,) bool, positions on opened images
    picks: tuple[dict, ...]           # one row per taken candidate
    rounds: tuple[dict, ...]          # one row per acquisition round
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------- the terms ---


def eps_from_geometry(
    features: np.ndarray, *, min_samples: int, quantile: float = EPS_QUANTILE
) -> float:
    """``eps`` as the ``quantile`` of the distance to the ``min_samples``-th neighbour.

    The k-distance graph is the standard way to read a density radius off a data
    set, and the standard advice — "look for the elbow" — is a judgement made on
    the very data the result is then read from. A fixed quantile is the same
    construction with the judgement removed: it is computed from the candidate
    features alone, it never sees a detector metric, and it is recomputed for
    every round so it tracks the population rather than a remembered value.
    """

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ScoreError(f"features must be a non-empty 2-D array, got {features.shape}")
    k = int(min_samples)
    if k < 1:
        raise ScoreError(f"min_samples must be at least 1, got {min_samples}")
    if features.shape[0] <= k:
        # Too small to speak about density at all. A radius that admits
        # everything is the honest answer; the caller's diagnostics record the
        # population size next to it.
        return float("inf")

    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(features)
    distances = model.kneighbors(features, return_distance=True)[0][:, -1]
    return float(np.quantile(distances, quantile))


def dbscan_gate(
    features: np.ndarray, *, eps: float, min_samples: int
) -> Gate:
    """DBSCAN core/border-vs-noise, with the three states kept apart.

    sklearn labels core and border points alike (``>= 0``) and only noise
    ``-1``; ``core_sample_indices_`` is what separates the first two. Both pass
    the gate — see the module docstring — but the log distinguishes them,
    because "the gate accepted this" and "the gate accepted this on the strength
    of its own neighbourhood" are different claims and only one of them is
    evidence that a small real cluster was found.
    """

    from sklearn.cluster import DBSCAN

    features = np.asarray(features, dtype=np.float64)
    n = features.shape[0]
    model = DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=-1).fit(features)
    labels = np.asarray(model.labels_, dtype=np.int64)

    status = np.full(n, "border", dtype="<U6")
    status[labels < 0] = "noise"
    status[np.asarray(model.core_sample_indices_, dtype=np.int64)] = "core"
    coherence = (labels >= 0).astype(np.float64)

    unique = np.unique(labels[labels >= 0])
    sizes = [int(np.count_nonzero(labels == k)) for k in unique]
    return Gate(
        labels=labels, status=status, coherence=coherence,
        diagnostics={
            "candidates": int(n),
            "eps": round(float(eps), 6) if np.isfinite(eps) else None,
            "min_samples": int(min_samples),
            "clusters": int(unique.size),
            "noise": int(np.count_nonzero(labels < 0)),
            "noise_rate": round(float(np.mean(labels < 0)), 4) if n else 0.0,
            "core": int(np.count_nonzero(status == "core")),
            "border": int(np.count_nonzero(status == "border")),
            "accepted": int(np.count_nonzero(coherence > 0)),
            "rejected": int(np.count_nonzero(coherence <= 0)),
            "mean_cluster_size": round(float(np.mean(sizes)), 2) if sizes else 0.0,
            "smallest_cluster": min(sizes) if sizes else 0,
            "largest_cluster": max(sizes) if sizes else 0,
        },
    )


def open_gate(n: int) -> Gate:
    """``coherence_mode='none'``: every candidate passes, and there is no partition.

    The labels are all ``0`` rather than ``-1``: a single cluster is what "no
    density structure was consulted" means, and it keeps
    :func:`cluster_rarity` well defined — every candidate is in the same cluster,
    so rarity is constant and the term drops out on its own instead of through a
    special case.
    """

    return Gate(
        labels=np.zeros(int(n), dtype=np.int64),
        status=np.full(int(n), "core", dtype="<U6"),
        coherence=np.ones(int(n), dtype=np.float64),
        diagnostics={"candidates": int(n), "clusters": 1, "noise": 0,
                     "noise_rate": 0.0, "accepted": int(n), "rejected": 0,
                     "gate": "open"},
    )


def continuous_coherence(features: np.ndarray, *, k: int = 10) -> np.ndarray:
    """The plan's original wording: the inverse of the k-th neighbour distance.

    Scaled by the population's median k-distance so the term lands on ``[0, 1]``
    without a magnitude constant, which is what :func:`owl.scoring.coherence`
    does for the CPU path. Kept identical in form so the two are comparable.
    """

    from sklearn.neighbors import NearestNeighbors

    features = np.asarray(features, dtype=np.float64)
    n = features.shape[0]
    if n <= k:
        return np.ones(n, dtype=np.float64)
    model = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(features)
    distances = model.kneighbors(features, return_distance=True)[0][:, -1]
    median = float(np.median(distances)) or 1.0
    return np.clip(1.0 / (1.0 + distances / median), 0.0, 1.0)


def cluster_rarity(
    labels: np.ndarray,
    features: np.ndarray,
    reference: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    """``w(ĉ(x))`` per candidate, from the partition the gate produced.

    ``R_k = log((1 + n_cand_k) / (1 + n_ref_k))``, positive part, where
    ``n_ref_k`` is the number of labelled-reference rows whose nearest cluster
    medoid is ``k``. A cluster the labelled set already covers as well as the
    candidates do gets nothing. Rank-normalised over the candidates so it sits on
    the same ``[0, 1]`` scale as ``U`` and ``D``, then **forced to zero on noise
    points**: a candidate with no cluster has no rarity claim to make, and
    rank-normalisation would otherwise hand the whole tied noise block the
    average rank of that tie.

    Never reads a true class count. ``n_cand`` and ``n_ref`` are occupancies of
    an embedding partition; the oracle's class frequencies are available only to
    :func:`known_contamination` and to the post-hoc summary.

    ``features`` and ``reference`` must be in the **same, unreduced** space.
    ``labels`` may come from the PCA-reduced space — that is where a DBSCAN
    radius means anything — but the medoid a reference row is assigned to has to
    be computed where the reference actually lives, so the partition is taken as
    given and the geometry is redone here. Both sides are L2-normalised first,
    because :func:`owl.active_selection.allocation.medoids` and
    :func:`~owl.active_selection.allocation.reference_counts` are cosine
    arguments and a stray norm would silently reweight them.
    """

    labels = np.asarray(labels, dtype=np.int64)
    features = _unit(np.asarray(features, dtype=np.float64))
    if reference is not None and len(reference):
        reference = _unit(np.asarray(reference, dtype=np.float64))
        if reference.shape[1] != features.shape[1]:
            raise ScoreError(
                f"the candidates are {features.shape[1]}-d and the labelled "
                f"reference is {reference.shape[1]}-d; rarity would be counted "
                "in a space the reference does not live in"
            )
    ids, centres = allocation.medoids(features, labels)
    n_ref = allocation.reference_counts(reference, ids, centres)

    raw = np.zeros(labels.shape[0], dtype=np.float64)
    per_cluster: dict[int, float] = {}
    for identifier in ids:
        identifier = int(identifier)
        members = labels == identifier
        n_cand = int(np.count_nonzero(members))
        value = float(np.log((1.0 + n_cand) / (1.0 + n_ref.get(identifier, 0))))
        per_cluster[identifier] = value
        raw[members] = max(value, 0.0)

    weight = scoring.rank_normalise(raw).astype(np.float64)
    weight[labels < 0] = 0.0
    return weight, {
        "rarity_clusters": int(ids.size),
        "rarity_reference_rows": 0 if reference is None else len(reference),
        "rarity_positive_clusters": int(sum(1 for v in per_cluster.values() if v > 0)),
        "rarity_mean": round(float(weight.mean()), 4) if weight.size else 0.0,
    }


def labeled_novelty(features: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """``D_labeled``: cosine distance to the nearest labelled row, on ``[0, 1]``.

    An empty reference gives ``1.0`` everywhere, which is the correct reading at
    the first purchase task: nothing is labelled, so nothing is redundant.
    """

    features = np.asarray(features, dtype=np.float32)
    if reference is None or len(reference) == 0:
        return np.ones(features.shape[0], dtype=np.float64)
    similarity = scoring._max_similarity(
        features, np.asarray(reference, dtype=np.float32)
    )
    return np.clip(1.0 - similarity.astype(np.float64), 0.0, 1.0)


# ------------------------------------------------------------- the campaign ---


def _combine(
    spec: ScoreSpec,
    uncertainty: np.ndarray,
    d_labeled: np.ndarray,
    d_batch: np.ndarray,
    rarity: np.ndarray,
    coherence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``s(x)`` and the ``D`` that went into it."""

    if spec.diversity_mode == "none":
        diversity = np.zeros_like(uncertainty)
    elif spec.diversity_mode == "labeled_novelty":
        diversity = d_labeled
    elif spec.diversity_mode == "batch_diversity":
        diversity = d_batch
    else:
        diversity = 0.5 * (d_labeled + d_batch)

    score = (
        uncertainty
        + spec.lambda_diversity * diversity
        + spec.gamma_rarity * rarity * coherence
    )
    return score, diversity


def select(
    features: np.ndarray,
    *,
    entropy: np.ndarray,
    image_ids: Sequence[str],
    cost_of: Callable[[str], int],
    budget: int,
    spec: ScoreSpec | None = None,
    rounds: int = 1,
    reference: np.ndarray | None = None,
    excluded_images: frozenset[str] = frozenset(),
    log_picks: bool = True,
) -> ResearchSelection:
    """Spend one task's answer budget under ``s(x) = U + λD + γ·w·coh``.

    ``rounds`` is the consultation's point 7 — "600 at once, or the best 100 then
    recompute". Between rounds the labelled reference grows by what the round
    bought, and the clustering, ``eps``, the rarity and ``D_labeled`` are all
    recomputed against it. The detector is untouched.

    The cost rule is the one every other arm uses
    (:func:`owl.active_selection.budget.spend_ranking`): opening an image costs
    ``max(1, annotated objects on it)`` answers, a second candidate on an open
    image is free and counted as redundant, images bought at an earlier task are
    skipped, and the campaign stops on the first unaffordable image rather than
    skipping it for a cheaper one.
    """

    spec = spec or ScoreSpec()
    features = np.asarray(features, dtype=np.float32)
    entropy = np.asarray(entropy, dtype=np.float64)
    image_ids = np.asarray(image_ids, dtype=str)
    if features.shape[0] != entropy.shape[0] or features.shape[0] != image_ids.shape[0]:
        raise ScoreError(
            f"features {features.shape[0]}, entropy {entropy.shape[0]} and image "
            f"ids {image_ids.shape[0]} describe different populations"
        )
    if int(rounds) < 1:
        raise ScoreError(f"rounds must be at least 1, got {rounds}")

    unique_images = sorted(set(image_ids.tolist()))
    if not unique_images:
        raise ScoreError("the candidate population is empty")
    mean_cost = float(np.mean([cost_of(i) for i in unique_images]))
    min_samples = allocation.min_cluster_size(
        features.shape[0], budget=int(budget), mean_image_cost=mean_cost
    )

    # One PCA for the whole task. The gate and the partition need a space where
    # a radius means something; refitting it per round would move the axes under
    # a comparison whose only intended change is the reference.
    reduced = clustering._reduce(features, spec.pca_dimensions, 0).astype(np.float64)

    ledger = ledger_module.Ledger(budget=int(budget))
    blocks: list[np.ndarray] = (
        [np.asarray(reference, dtype=np.float32)]
        if reference is not None and len(reference) else []
    )
    opened: list[str] = []
    anchors: list[int] = []
    picks: list[dict] = []
    history: list[dict] = []
    taken = np.zeros(features.shape[0], dtype=bool)
    scanned = redundant = 0

    for index in range(1, int(rounds) + 1):
        allowance = allocation.round_allowance(
            int(budget), int(rounds), index, ledger.spent
        )
        eligible = np.flatnonzero(
            ~taken & ~np.isin(image_ids, list(excluded_images) + opened)
        )
        if allowance <= 0 or eligible.size == 0:
            history.append({"round": index, "allowance": int(allowance),
                            "eligible": int(eligible.size), "images": 0,
                            "answers": 0, "stopped": "exhausted"})
            continue

        block = reduced[eligible]
        if spec.coherence_mode == "dbscan_binary":
            eps = eps_from_geometry(
                block, min_samples=min_samples, quantile=spec.eps_quantile
            )
            gate = dbscan_gate(block, eps=eps, min_samples=min_samples)
        elif spec.coherence_mode == "continuous":
            gate = open_gate(eligible.size)
            gate = Gate(
                labels=gate.labels, status=gate.status,
                coherence=continuous_coherence(block, k=spec.coherence_k),
                diagnostics={**gate.diagnostics, "gate": "continuous",
                             "coherence_k": spec.coherence_k},
            )
        else:
            gate = open_gate(eligible.size)

        if spec.rarity_mode == "cluster_rarity":
            stacked = np.vstack(blocks) if blocks else None
            # The partition comes from `block` (reduced); the medoids and the
            # reference assignment are computed on the unreduced features, which
            # is the space the labelled reference is exported in.
            rarity, rarity_diagnostics = cluster_rarity(
                gate.labels, features[eligible], stacked
            )
        else:
            rarity = np.zeros(eligible.size, dtype=np.float64)
            rarity_diagnostics = {"rarity_clusters": 0, "rarity_mean": 0.0}

        uncertainty = scoring.rank_normalise(entropy[eligible]).astype(np.float64)
        needs_labeled = spec.diversity_mode in ("labeled_novelty", "combined")
        d_labeled = (
            scoring.rank_normalise(
                labeled_novelty(features[eligible], np.vstack(blocks) if blocks else None)
            ).astype(np.float64)
            if needs_labeled
            else np.zeros(eligible.size, dtype=np.float64)
        )
        # The one term used raw. It changes at every pick, so rank-normalising it
        # would make a candidate's value depend on how many rivals are left
        # rather than on how far it is from what was just bought — and it would
        # cost a sort per pick. It is already a bounded cosine distance.
        d_batch = np.ones(eligible.size, dtype=np.float64)
        unit = block / np.maximum(
            np.linalg.norm(block, axis=1, keepdims=True), 1e-9
        )

        round_images = 0
        round_answers = 0
        stopped = "budget"
        local_taken = np.zeros(eligible.size, dtype=bool)
        seen: set[str] = set(excluded_images) | set(opened)

        while True:
            score, diversity = _combine(
                spec, uncertainty, d_labeled, d_batch, rarity, gate.coherence
            )
            score = np.where(local_taken, -np.inf, score)
            if not np.isfinite(score).any():
                stopped = "exhausted"
                break
            local = int(np.argmax(score))
            position = int(eligible[local])
            image = str(image_ids[position])
            scanned += 1

            was_redundant = image in seen
            if was_redundant:
                redundant += 1
            else:
                cost = int(cost_of(image))
                if not ledger.affordable(cost) or cost > allowance - round_answers:
                    stopped = "unaffordable"
                    break
                seen.add(image)
                ledger.charge(image, cost)
                opened.append(image)
                anchors.append(position)
                round_images += 1
                round_answers += cost

            if log_picks:
                picks.append({
                    "round": index,
                    "position": position,
                    "image_id": image,
                    "redundant": bool(was_redundant),
                    "U": round(float(uncertainty[local]), 6),
                    "D_labeled": round(float(d_labeled[local]), 6),
                    "D_batch": round(float(d_batch[local]), 6),
                    "D": round(float(diversity[local]), 6),
                    "w": round(float(rarity[local]), 6),
                    "coh": round(float(gate.coherence[local]), 6),
                    "cluster": int(gate.labels[local]),
                    "cluster_status": str(gate.status[local]),
                    "score": round(float(score[local]), 6),
                })

            local_taken[local] = True
            taken[position] = True
            # Farthest-first update. Every taken candidate lowers the batch
            # diversity of whatever resembles it, including a redundant one:
            # under full-image labelling a candidate on an already-open image is
            # annotated too, so it is part of the batch.
            d_batch = np.minimum(
                d_batch, np.clip(1.0 - unit @ unit[local], 0.0, 1.0)
            )

            if ledger.remaining <= 0 or round_answers >= allowance:
                break

        bought = np.isin(image_ids, list(opened))
        blocks = [
            *(blocks[:1] if reference is not None and len(reference) else []),
            *([features[bought]] if bought.any() else []),
        ]
        history.append({
            "round": index, "allowance": int(allowance),
            "eligible": int(eligible.size), "images": round_images,
            "answers": round_answers, "stopped": stopped,
            "reference_rows": int(sum(len(b) for b in blocks)),
            **gate.diagnostics, **rarity_diagnostics,
        })

    covered = np.isin(image_ids, list(opened))
    return ResearchSelection(
        images=tuple(opened), anchors=tuple(anchors), covered=covered,
        picks=tuple(picks), rounds=tuple(history),
        diagnostics=ledger.summary() | {
            "candidates": int(features.shape[0]),
            "mean_image_cost": round(mean_cost, 4),
            "min_samples": int(min_samples),
            "rounds_run": len(history),
            "positions_scanned": scanned,
            "positions_redundant": redundant,
            "positions_taken": int(taken.sum()),
            # The labelled pool as columns rather than only as round-log
            # entries, because "the pool grew during this task" and "it grew
            # along the chain" are both assertions a table has to be able to
            # make. `_final` matches the name the sibling iterative allocator
            # already uses, and it includes what this task bought — it is the
            # pool the next task inherits.
            "reference_rows_initial": int(
                len(reference) if reference is not None else 0),
            "reference_rows_final": int(sum(len(b) for b in blocks)),
            **{f"final_{k}": v for k, v in (history[-1] if history else {}).items()
               if k in ("noise_rate", "clusters", "accepted", "rejected", "eps")},
        } | spec.as_dict(),
    )


# --------------------------------------------------------------- diagnostic ---


def known_contamination(
    candidate_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    min_samples: int,
    eps_quantile: float = EPS_QUANTILE,
    pca_dimensions: int | None = PCA_DIMENSIONS,
) -> dict[str, object]:
    """Point 5: cluster the knowns *with* the candidates and see how much bleeds.

    The consultation's geometric wish, stated as a measurement: "if there are
    1,000 known and 10,000 unknown, cluster so that as few knowns as possible go
    over into the unknown side". This fits one DBSCAN on the union and reports
    how the known rows distributed themselves over the clusters the candidates
    occupy.

    **A diagnostic and nothing else.** It is never called from :func:`select`,
    it cannot change a score, and it reads no oracle class — "known" here means
    "a row of the labelled reference", which is exactly what the trajectory has
    already paid for. ``tests/test_research_score.py`` asserts the selection
    path does not import it.
    """

    candidate_features = np.asarray(candidate_features, dtype=np.float32)
    reference_features = np.asarray(reference_features, dtype=np.float32)
    if reference_features.ndim != 2 or reference_features.shape[0] == 0:
        return {"known_rows": 0, "note": "empty reference; nothing to contaminate"}
    if candidate_features.shape[1] != reference_features.shape[1]:
        raise ScoreError(
            f"candidates are {candidate_features.shape[1]}-d and the reference is "
            f"{reference_features.shape[1]}-d; they are not the same space"
        )

    n_candidates = candidate_features.shape[0]
    union = np.vstack([candidate_features, reference_features])
    reduced = clustering._reduce(union, pca_dimensions, 0).astype(np.float64)
    eps = eps_from_geometry(reduced, min_samples=min_samples, quantile=eps_quantile)
    gate = dbscan_gate(reduced, eps=eps, min_samples=min_samples)

    labels = gate.labels
    is_known = np.zeros(labels.shape[0], dtype=bool)
    is_known[n_candidates:] = True

    ids = np.unique(labels[labels >= 0])
    candidate_only = 0
    known_only = 0
    mixed = 0
    leaked = 0
    purities: list[float] = []
    sizes: list[int] = []
    for identifier in ids:
        members = labels == identifier
        n_known = int(np.count_nonzero(members & is_known))
        n_cand = int(np.count_nonzero(members & ~is_known))
        sizes.append(n_known + n_cand)
        if n_known and n_cand:
            mixed += 1
            leaked += n_known
        elif n_cand:
            candidate_only += 1
        else:
            known_only += 1
        # purity read from the known side: of this cluster's rows, what share is
        # known. A cluster the candidates own scores near zero, which is what a
        # usable "new structure" cluster looks like.
        purities.append(n_known / max(n_known + n_cand, 1))

    total_known = int(np.count_nonzero(is_known))
    return {
        "known_rows": total_known,
        "candidate_rows": int(n_candidates),
        "eps": round(float(eps), 6) if np.isfinite(eps) else None,
        "min_samples": int(min_samples),
        "clusters": int(ids.size),
        "clusters_candidate_only": candidate_only,
        "clusters_known_only": known_only,
        "clusters_mixed": mixed,
        # the headline: knowns sitting in a cluster the candidates also occupy
        "known_contamination_rate": round(leaked / max(total_known, 1), 4),
        "known_leaked_rows": leaked,
        "known_noise_rate": round(
            float(np.mean(labels[is_known] < 0)), 4) if total_known else 0.0,
        "candidate_noise_rate": round(
            float(np.mean(labels[~is_known] < 0)), 4) if n_candidates else 0.0,
        "mean_cluster_purity_known_side": round(
            float(np.mean(purities)), 4) if purities else 0.0,
        "mean_cluster_size": round(float(np.mean(sizes)), 2) if sizes else 0.0,
        "candidate_cluster_sizes": ";".join(
            str(int(np.count_nonzero((labels == k) & ~is_known))) for k in ids[:50]
        ),
    }
