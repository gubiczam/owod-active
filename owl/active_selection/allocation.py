"""Cluster-stratified annotation allocation — the plan's ``R`` as an allocator.

Frozen by ``docs/distribution_aware_decision_memo_2026-09-06.md`` §3.2 and §4.
Nothing in this module may be changed after an oracle outcome has been seen.

**Why an allocator and not a score.** The 2026-08-25 consultation describes
rarity as a statement about how much of the budget a region of the semantic
space receives, and describes coherence as "``coh(x) ∈ {0,1}`` — a switch, not a
weight". A scalar score cannot allocate; it can only reorder. Method V3 measured
what a weight does to a dense ranking: ``A·C`` selected almost exactly what ``A``
selected, because the ``A`` ranking is dense to three parts in 10^5 at the cut.
So ``R`` here changes *quota*, and ``C`` is a gate.

**Why no traversal.** Proposed-v1 and Proposed-v2 both maximised semantic
coverage and both produced the highest ``U_Recall50`` and the lowest
``new_class_AP50`` of five arms. Farthest-first leaves a region once it is
covered, so per-class multiplicity is capped by construction — one or two of
everything, never the tens a class needs. There is no traversal in this module.
Within a cluster the allocator takes as many candidates as the quota allows,
ordered by entropy, so multiplicity is bounded by the quota instead.

**Label-free, by construction.** Nothing here is passed an oracle class, an
oracle box, a future class declaration or an evaluation row. The only oracle-side
quantity in reach is ``cost_of``, the annotation cost function the ledger already
hands to *every* arm, and it enters in exactly two places, both recorded in the
diagnostics: the scalar mean image cost that sets ``min_cluster_size``, and the
charge against a cluster's quota when an image is opened.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

#: Floor on the HDBSCAN minimum cluster size. A cluster of fewer than five
#: candidates cannot support a rarity estimate read from its occupancy.
MIN_CLUSTER_FLOOR = 5


class AllocationError(ValueError):
    """Raised when the allocator is handed something it cannot rank."""


@dataclass(frozen=True)
class Allocation:
    """One task's cluster structure, quotas and emitted order."""

    order: np.ndarray                     # permutation of the input positions
    labels: np.ndarray                    # HDBSCAN label per input row, -1 noise
    coherence: np.ndarray                 # C(x) in {0, 1}
    quota: dict[int, float]               # answers allocated to each cluster
    rarity: dict[int, float]              # R_k, before the positive part
    diagnostics: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.order.size)


def min_cluster_size(n_candidates: int, *, budget: int, mean_image_cost: float) -> int:
    """``max(5, ceil(|G| / (B / c_bar)))`` — derived from the budget, not tuned.

    ``B / c_bar`` is how many images this task's budget can open, so the quotient
    is "candidates per image the campaign can afford". A cluster smaller than
    that cannot receive a quota worth one image, and a rarity signal read from
    its occupancy would be noise. No number here was chosen by looking at an
    outcome: ``B`` is the frozen answer budget and ``c_bar`` is a property of the
    pool.
    """

    if mean_image_cost <= 0:
        raise AllocationError("mean image cost must be positive")
    affordable_images = float(budget) / float(mean_image_cost)
    if affordable_images <= 0:
        raise AllocationError("the budget cannot open a single image")
    return max(MIN_CLUSTER_FLOOR, int(np.ceil(n_candidates / affordable_images)))


def cluster(features: np.ndarray, *, size: int) -> np.ndarray:
    """HDBSCAN over the admissible set's semantic features. Deterministic.

    HDBSCAN rather than DBSCAN **because it removes** ``eps``. Picking ``eps``
    off a k-distance elbow is a judgement call made on the very data the result
    would then be read from; ``min_cluster_size`` is derived from the budget
    instead. ``min_samples`` is left at the library default, which is
    ``min_cluster_size``.
    """

    from sklearn.cluster import HDBSCAN

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise AllocationError(f"features must be a non-empty 2-D array, got {features.shape}")
    if features.shape[0] <= size:
        return np.full(features.shape[0], -1, dtype=np.int64)
    # `copy=True` explicitly: the library default is changing, and a clusterer
    # that may rewrite its input in place has no business touching a feature
    # matrix the caller still needs.
    labels = HDBSCAN(
        min_cluster_size=int(size), min_samples=None, copy=True,
    ).fit_predict(features)
    return np.asarray(labels, dtype=np.int64)


def medoids(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One representative per cluster, exactly and in ``O(n·d)``.

    The medoid is defined as the cluster member minimising the sum of **squared**
    distances to its own cluster. That has a closed form: expanding
    ``sum_j ||x_i - x_j||^2 = n||x_i||^2 - 2 x_i·S + sum_j ||x_j||^2`` and using
    ``||x|| = 1`` for L2-normalised features leaves ``argmax_i x_i·S``, the
    member of greatest cosine similarity to the cluster mean. The definition is
    fixed here, before any outcome, because the exact sum-of-*unsquared*-distances
    medoid is ``O(n^2 d)`` and would be infeasible on a 19,000-row admissible set.
    """

    features = np.asarray(features, dtype=np.float64)
    ids = np.array(sorted({int(v) for v in labels if int(v) >= 0}), dtype=np.int64)
    if ids.size == 0:
        return ids, np.zeros((0, features.shape[1]))
    chosen = np.empty((ids.size, features.shape[1]))
    for position, identifier in enumerate(ids):
        members = np.flatnonzero(labels == identifier)
        block = features[members]
        total = block.sum(axis=0)
        # stable argmax: ties resolve to the lowest member index
        best = int(np.argmax(block @ total))
        chosen[position] = features[members[best]]
    return ids, chosen


def reference_counts(
    reference: np.ndarray | None, ids: np.ndarray, centres: np.ndarray
) -> dict[int, int]:
    """``n_ref(k)`` — reference rows whose nearest cluster medoid is ``k``.

    The reference is the balanced task-1 labelled set plus everything this
    trajectory has already bought, so this is under-representation *relative to
    what is already labelled* — the consultation's ``D``, read at cluster
    granularity. An empty reference makes every cluster maximally
    under-represented, which is the correct reading at the first purchase task.
    """

    counts = {int(k): 0 for k in ids}
    if reference is None or len(reference) == 0 or ids.size == 0:
        return counts
    reference = np.asarray(reference, dtype=np.float64)
    # chunked so a large reference does not materialise an (n_ref x K) block at once
    for start in range(0, reference.shape[0], 4096):
        nearest = np.argmax(reference[start:start + 4096] @ centres.T, axis=1)
        for index in nearest:
            counts[int(ids[index])] += 1
    return counts


def quotas(
    labels: np.ndarray, ids: np.ndarray, n_ref: dict[int, int], *, budget: int
) -> tuple[dict[int, float], dict[int, float]]:
    """``R_k = log((1 + n_cand) / (1 + n_ref))`` and ``q_k = B·R+ / sum R+``.

    The positive part is the whole of the rarity rule: a cluster the labelled set
    already covers as well as the candidates do receives **nothing**. There is no
    temperature and no exponent, because either would be a free parameter chosen
    after the fact.
    """

    rarity: dict[int, float] = {}
    for identifier in ids:
        identifier = int(identifier)
        n_cand = int(np.count_nonzero(labels == identifier))
        rarity[identifier] = float(np.log((1.0 + n_cand) / (1.0 + n_ref.get(identifier, 0))))
    positive = {k: max(v, 0.0) for k, v in rarity.items()}
    total = sum(positive.values())
    if total <= 0:
        return {k: 0.0 for k in rarity}, rarity
    return {k: float(budget) * v / total for k, v in positive.items()}, rarity


def emit(
    labels: np.ndarray,
    quota: dict[int, float],
    entropy: np.ndarray,
    image_ids: Sequence[str],
    cost_of: Callable[[str], int],
) -> tuple[np.ndarray, dict]:
    """Walk the clusters in descending quota, each giving up its next best by U.

    Round-robin, in a fixed order, so a cluster's share of the campaign is set by
    how long its quota keeps it alive rather than by where it sits in the order.
    A cluster is charged only when its pick opens an image no earlier pick opened
    — a second candidate on an open image costs the annotator nothing and must
    not consume a quota either. When every cluster has left, the remaining
    positions spill in one global descending-entropy order, which is what makes
    the result a full permutation and what the noise points fall into.
    """

    entropy = np.asarray(entropy, dtype=np.float64)
    image_ids = np.asarray(image_ids, dtype=str)
    live = sorted(quota, key=lambda k: (-quota[k], k))
    queues = {
        k: list(np.flatnonzero(labels == k)[
            np.argsort(-entropy[labels == k], kind="stable")
        ])
        for k in live
    }
    remaining = dict(quota)
    order: list[int] = []
    taken = np.zeros(labels.shape[0], dtype=bool)
    opened: set[str] = set()
    charged = {k: 0.0 for k in live}

    while live:
        still: list[int] = []
        for identifier in live:
            queue = queues[identifier]
            if not queue or remaining[identifier] <= 0:
                continue
            position = queue.pop(0)
            order.append(int(position))
            taken[position] = True
            image = str(image_ids[position])
            if image not in opened:
                opened.add(image)
                price = float(cost_of(image))
                remaining[identifier] -= price
                charged[identifier] += price
            if queue and remaining[identifier] > 0:
                still.append(identifier)
        live = still

    spill = np.flatnonzero(~taken)
    spill = spill[np.argsort(-entropy[spill], kind="stable")]
    diagnostics = {
        "emitted_under_quota": len(order),
        "spilled": int(spill.size),
        "images_under_quota": len(opened),
        "clusters_funded": int(sum(1 for v in quota.values() if v > 0)),
        "answers_charged": round(float(sum(charged.values())), 3),
    }
    return np.concatenate([np.asarray(order, dtype=np.int64), spill]), diagnostics


def distribution_aware_order(
    features: np.ndarray,
    *,
    entropy: np.ndarray,
    image_ids: Sequence[str],
    cost_of: Callable[[str], int],
    budget: int,
    reference: np.ndarray | None = None,
) -> Allocation:
    """``distribution_aware_v1``: the whole of steps 4-7, as one permutation.

    Steps 1-3 (NMS, the ``A`` gate, the frozen DINOv2 embedding of what survives)
    happen before this is called and are unchanged from the benchmark. Step 8,
    spending the order, is the existing ledger and is unchanged too.
    """

    features = np.asarray(features, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)
    if features.shape[0] != entropy.shape[0] or features.shape[0] != len(image_ids):
        raise AllocationError(
            f"features {features.shape[0]}, entropy {entropy.shape[0]} and image ids "
            f"{len(image_ids)} describe different populations"
        )

    unique_images = sorted({str(i) for i in image_ids})
    mean_cost = float(np.mean([cost_of(i) for i in unique_images]))
    size = min_cluster_size(features.shape[0], budget=budget, mean_image_cost=mean_cost)

    labels = cluster(features, size=size)
    coherence = (labels >= 0).astype(np.int64)
    ids, centres = medoids(features, labels)
    n_ref = reference_counts(reference, ids, centres)
    quota, rarity = quotas(labels, ids, n_ref, budget=budget)
    order, emitted = emit(labels, quota, entropy, image_ids, cost_of)

    funded = [k for k, v in quota.items() if v > 0]
    return Allocation(
        order=order, labels=labels, coherence=coherence, quota=quota, rarity=rarity,
        diagnostics={
            "candidates": int(features.shape[0]),
            "mean_image_cost": round(mean_cost, 4),
            "min_cluster_size": int(size),
            "clusters": int(ids.size),
            "noise": int(np.count_nonzero(labels < 0)),
            "noise_share": round(float(np.mean(labels < 0)), 4),
            "reference_rows": 0 if reference is None else len(reference),
            "clusters_with_zero_quota": int(ids.size - len(funded)),
            "mean_cluster_size": round(
                float(np.mean([int(np.count_nonzero(labels == k)) for k in ids]))
                if ids.size else 0.0, 2),
            **emitted,
        },
    )


def cost_aware_order(
    admissibility: np.ndarray, image_ids: Sequence[str]
) -> tuple[np.ndarray, dict]:
    """The mandatory control: prefer images the detector says are busy.

    Entirely label-free — it reads ``A(x) = objectness·sqrt(area)`` and nothing
    else. It exists because the ceiling audit
    (``tools/audit_acquisition_ceiling.py``) found that simply opening cheap
    images beats a *perfect-cluster* stratified allocator on raw tail counts:
    rare classes live in sparse scenes, sparse scenes are cheap under a
    per-object cost model, and ~900 images for 3,000 answers outruns ~300. The
    proxy is the measured inversion — images carrying **more** admissible
    proposals carry **fewer** annotated objects (Spearman -0.53 against true
    cost, about 2x chance at recovering the cheapest quartile) — so descending
    sum-``A`` is a label-free cheap-image preference.

    Without this arm a win by ``distribution_aware_v1`` would not be
    attributable to distribution-awareness at all.
    """

    admissibility = np.asarray(admissibility, dtype=np.float64)
    image_ids = np.asarray(image_ids, dtype=str)
    totals: dict[str, float] = {}
    for value, image in zip(admissibility, image_ids, strict=True):
        totals[str(image)] = totals.get(str(image), 0.0) + float(value)
    # images by descending sum-A, ties by image id; within an image by descending A
    rank = {image: position for position, image
            in enumerate(sorted(totals, key=lambda i: (-totals[i], i)))}
    keys = np.array([rank[str(i)] for i in image_ids], dtype=np.int64)
    order = np.lexsort((-admissibility, keys)).astype(np.int64)
    return order, {
        "images": len(totals),
        "mean_sum_admissibility": round(float(np.mean(list(totals.values()))), 4),
    }
