"""k-center greedy: the standard core-set criterion, at image granularity.

Farthest-first traversal, as in Sener & Savarese's core-set active learning.
Given a set of already-covered points ``R`` and a candidate set ``X``, take

    argmax_{x in X}  min_{r in R} d(x, r)

add it to ``R``, repeat. ``d`` is cosine distance on unit-norm embeddings, which
is the distance every other cosine in this repository uses.

**One deliberate difference from the textbook version, and it is not a
shortcut.** The textbook picks one *point* per step. Here a step opens an
*image*, and full-image labelling means every annotated object on that image is
labelled — so every candidate on it becomes covered, not just the one that was
farthest. Adding only the chosen point would model an annotation that this
protocol does not perform, and would let one crowded image be bought several
times over. Opening an image therefore adds all of its candidates to ``R`` and
removes them from ``X``.

The criterion has **no hyperparameter**. That is the point of choosing it: the
weights of the earlier additive score (``lambda``, ``gamma``, ``mu``) each needed
a value, and any value chosen after seeing an endpoint would be a tuned result.
Here there is nothing to tune — the candidate set and the reference set are
fixed by the protocol, and the traversal is deterministic given them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from owl.active_selection.budget import Ledger

#: Rows per matrix-multiply block when scoring the pool against new reference
#: points. Bounds peak memory; it does not change the result.
CHUNK = 8192


@dataclass(frozen=True)
class Coverage:
    """What the traversal bought, in the order it bought it."""

    images: tuple[str, ...]
    anchors: tuple[int, ...]          # the candidate position that opened each image
    distances: tuple[float, ...]      # its min-distance to the reference when taken
    ledger: Ledger
    reference_size: int
    covered: np.ndarray               # (n,) bool, candidates on opened images
    diagnostics: dict

    def summary(self) -> dict[str, object]:
        taken = np.asarray(self.distances, dtype=np.float64)
        return self.ledger.summary() | {
            "reference_points": self.reference_size,
            "coverage_first_pick_distance": round(float(taken[0]), 4) if taken.size else None,
            "coverage_last_pick_distance": round(float(taken[-1]), 4) if taken.size else None,
            "coverage_mean_pick_distance": round(float(taken.mean()), 4) if taken.size else None,
            "candidates_covered": int(self.covered.sum()),
        }


def _min_distance(
    features: np.ndarray, reference: np.ndarray, *, chunk: int = CHUNK
) -> np.ndarray:
    """``1 - max cosine`` from every row of ``features`` to any reference row."""

    n = features.shape[0]
    if reference.size == 0:
        return np.full(n, np.inf, dtype=np.float64)
    best = np.full(n, -np.inf, dtype=np.float64)
    for start in range(0, n, chunk):
        stop = start + chunk
        block = features[start:stop] @ reference.T
        best[start:stop] = block.max(axis=1)
    return 1.0 - best


def kcenter_greedy(
    features: np.ndarray,
    image_ids: np.ndarray,
    *,
    cost_of: Callable[[str], int],
    budget: int,
    reference: np.ndarray | None = None,
    candidate: np.ndarray | None = None,
    excluded_images: frozenset[str] = frozenset(),
    tie_break: np.ndarray | None = None,
    chunk: int = CHUNK,
) -> Coverage:
    """Spend ``budget`` oracle answers by farthest-first traversal.

    ``candidate`` restricts which positions may be *chosen* — the admissibility
    gate lives here, and switching it off is the ``coreset`` control. Coverage
    is still credited for every candidate on an opened image, gated or not,
    because the annotator labelled them.

    ``tie_break`` decides between equally distant candidates; higher wins. It
    matters at the very first pick, where an empty reference makes every
    distance infinite. Passing admissibility makes that first pick the most
    object-like region rather than row zero.
    """

    features = np.asarray(features, dtype=np.float32)
    image_ids = np.asarray(image_ids, dtype=str)
    n = features.shape[0]
    if image_ids.shape[0] != n:
        raise ValueError(
            f"features has {n} rows and image_ids {image_ids.shape[0]}; they "
            "must describe the same candidates."
        )
    choosable = (
        np.ones(n, dtype=bool) if candidate is None else np.asarray(candidate, dtype=bool).copy()
    )
    if choosable.shape != (n,):
        raise ValueError(f"candidate has shape {choosable.shape}, expected ({n},).")
    order_key = (
        np.zeros(n, dtype=np.float64)
        if tie_break is None
        else np.asarray(tie_break, dtype=np.float64)
    )

    members: dict[str, np.ndarray] = {}
    for image in np.unique(image_ids):
        members[str(image)] = np.flatnonzero(image_ids == image)
    if excluded_images:
        blocked = np.isin(image_ids, np.asarray(sorted(excluded_images), dtype=str))
        choosable &= ~blocked

    reference = (
        np.zeros((0, features.shape[1]), dtype=np.float32)
        if reference is None
        else np.asarray(reference, dtype=np.float32).reshape(-1, features.shape[1])
    )
    distance = _min_distance(features, reference, chunk=chunk)

    ledger = Ledger(budget=int(budget))
    covered = np.zeros(n, dtype=bool)
    images: list[str] = []
    anchors: list[int] = []
    taken: list[float] = []
    stopped = "pool exhausted"

    while True:
        pool = np.flatnonzero(choosable)
        if pool.size == 0:
            break
        # lexicographic: farthest first, then most object-like, then lowest index
        best = int(pool[np.lexsort((pool, -order_key[pool], -distance[pool]))[0]])
        image = str(image_ids[best])
        cost = cost_of(image)
        if not ledger.affordable(cost):
            stopped = f"next image costs {cost}, {ledger.remaining} answers left"
            break
        ledger.charge(image, cost)
        images.append(image)
        anchors.append(best)
        taken.append(float(distance[best]))

        group = members[image]
        covered[group] = True
        choosable[group] = False
        # everything on the opened image is now labelled, so it is covered
        block = features @ features[group].T
        distance = np.minimum(distance, 1.0 - block.max(axis=1))

    return Coverage(
        images=tuple(images),
        anchors=tuple(anchors),
        distances=tuple(taken),
        ledger=ledger,
        reference_size=int(reference.shape[0]),
        covered=covered,
        diagnostics={
            "choosable_at_start": int(
                (np.asarray(candidate, dtype=bool) if candidate is not None
                 else np.ones(n, dtype=bool)).sum()
            ),
            "stopped_because": stopped,
            "picks": len(images),
        },
    )
