"""The exemplar memory: what we hand back from earlier tasks, and in what mix.

Point 4 of the 2026-08-25 consultation, and contribution B of the research
plan. Replay-based incremental methods keep a memory of old examples and
rehearse them while learning the new class. The standard allocation gives every
class the same number of exemplars. Under a long-tailed distribution that is a
choice against the tail: rare classes learned from less data in the first place,
carry a weaker representation, and are the first to be overwritten.

The tunable allocation, straight from the plan::

    m_c  proportional to  n_c ** alpha,   sum(m_c) = M

======================  ==================================================
``alpha = 0``           equal per class — today's standard
``alpha = 1``           proportional to class size — favours the head
``alpha < 0``           favours the tail; ``-1`` inverts the distribution
======================  ==================================================

Three things are separate experiments and are separate arguments here:

* the **size** of the memory, ``total``;
* the **rule**, ``alpha``;
* whether the memory is **carried forward or re-allocated** every task
  (``reallocate``), which the consultation asked for explicitly — a memory sized
  for three known classes is not the right memory for ten.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Memory:
    """One task's exemplar memory, at image granularity.

    Training runs on images, so the memory is a set of image ids. ``per_class``
    records the object-level allocation it was built to satisfy, which is what
    makes the allocation rule auditable after the fact.
    """

    image_ids: tuple[str, ...]
    per_class: Mapping[str, int]
    alpha: float
    total: int

    def __len__(self) -> int:
        return len(self.image_ids)

    def summary(self) -> dict[str, object]:
        return {
            "images": len(self.image_ids),
            "alpha": self.alpha,
            "budget": self.total,
            "allocated": int(sum(self.per_class.values())),
            "classes": len(self.per_class),
        }


def allocate(
    class_counts: Mapping[str, int],
    *,
    total: int,
    alpha: float = 0.0,
    minimum: int = 1,
) -> dict[str, int]:
    """Split a fixed exemplar budget across classes by ``n_c ** alpha``.

    The requested allocation is capacity-aware: if one class does not contain
    enough available exemplars to use its weighted share, the unused budget is
    redistributed among classes that still have capacity.

    Therefore, whenever the available pool contains at least ``total`` objects,

        sum(m_c) == total

    while no class is ever allocated more exemplars than are available.
    """

    classes = [name for name, count in class_counts.items() if count > 0]
    if not classes or total <= 0:
        return {}

    capacities = np.asarray(
        [int(class_counts[name]) for name in classes],
        dtype=np.int64,
    )
    target = min(int(total), int(capacities.sum()))

    allocation = np.zeros(len(classes), dtype=np.int64)

    # Preserve every represented class when the budget makes that possible.
    requested_minimum = np.minimum(capacities, max(int(minimum), 0))
    minimum_total = int(requested_minimum.sum())

    if minimum_total <= target:
        allocation[:] = requested_minimum
    else:
        # The requested minimum itself does not fit. Allocate one exemplar at a
        # time in weighted order instead of silently exceeding the total.
        weights = np.power(capacities.astype(np.float64), alpha)
        order = np.argsort(-weights, kind="stable")
        remaining = target
        for index in order:
            take = min(int(requested_minimum[index]), remaining)
            allocation[index] += take
            remaining -= take
            if remaining == 0:
                break

    weights = np.power(capacities.astype(np.float64), alpha)

    while int(allocation.sum()) < target:
        remaining_budget = target - int(allocation.sum())
        free_capacity = capacities - allocation
        active = free_capacity > 0

        if not active.any():
            break

        active_weights = np.where(active, weights, 0.0)
        weight_sum = float(active_weights.sum())

        if weight_sum <= 0:
            break

        raw = active_weights / weight_sum * remaining_budget
        grant = np.floor(raw).astype(np.int64)
        grant = np.minimum(grant, free_capacity)
        grant[~active] = 0

        if int(grant.sum()) > 0:
            allocation += grant
            continue

        # With a small remainder every proportional share can be below one.
        # Largest-remainder allocation makes progress deterministically.
        fractional = np.where(active, raw - np.floor(raw), -np.inf)
        index = int(np.argmax(fractional))
        allocation[index] += 1

    return {
        name: int(value)
        for name, value in zip(classes, allocation)
        if value > 0
    }

def build(
    per_image_classes: Mapping[str, Mapping[str, int]],
    known_classes: Sequence[str],
    *,
    total: int,
    alpha: float = 0.0,
    seed: int = 0,
    selector: str = "greedy",
    priority: Sequence[str] | None = None,
) -> Memory:
    """Choose the images that satisfy the allocation as closely as possible.

    ``per_image_classes`` maps an image id to how many objects of each class it
    holds — one image usually serves several classes at once, which is why an
    object-level allocation cannot be turned into an image list by division.

    ``greedy`` repeatedly takes the image that covers the most still-unmet
    demand, which is a set-cover heuristic and reaches the target with far fewer
    images than sampling per class. ``random`` is the control.

    ``priority`` optionally orders the images before selection — pass the output
    of :func:`herding_order` to get iCaRL's criterion instead of set cover. The
    allocation rule is unchanged either way, so the two are one variable apart.
    """

    counts: dict[str, int] = {name: 0 for name in known_classes}
    for classes in per_image_classes.values():
        for name, number in classes.items():
            if name in counts:
                counts[name] += number

    demand = allocate(counts, total=total, alpha=alpha)
    if not demand:
        return Memory(image_ids=(), per_class={}, alpha=alpha, total=total)

    generator = np.random.default_rng(seed)
    image_ids = sorted(per_image_classes)
    if priority is not None:
        ranked = {str(name): rank for rank, name in enumerate(priority)}
        image_ids = sorted(image_ids, key=lambda name: ranked.get(name, len(ranked)))

    if selector == "random":
        chosen = generator.permutation(np.asarray(image_ids, dtype=object))
        picked: list[str] = []
        remaining = dict(demand)
        for image in chosen:
            classes = per_image_classes[image]
            if not any(remaining.get(name, 0) > 0 for name in classes):
                continue
            picked.append(str(image))
            for name, number in classes.items():
                if name in remaining:
                    remaining[name] = max(0, remaining[name] - number)
            if not any(remaining.values()):
                break
        return Memory(tuple(picked), demand, alpha, total)

    if selector != "greedy":
        raise ValueError(f"Unknown selector {selector!r}; use 'greedy' or 'random'.")

    remaining = dict(demand)
    picked = []
    if priority is not None:
        # the order is the decision; walk it and keep whatever still helps
        for image in image_ids:
            if not any(remaining.get(name, 0) > 0 for name in per_image_classes[image]):
                continue
            picked.append(image)
            for name, number in per_image_classes[image].items():
                if name in remaining:
                    remaining[name] = max(0, remaining[name] - number)
            if not any(remaining.values()):
                break
        return Memory(tuple(picked), demand, alpha, total)

    order = generator.permutation(len(image_ids))  # break ties without bias
    pool = [image_ids[i] for i in order]
    while any(value > 0 for value in remaining.values()):
        best_image, best_gain = None, 0
        for image in pool:
            gain = sum(
                min(number, remaining.get(name, 0))
                for name, number in per_image_classes[image].items()
            )
            if gain > best_gain:
                best_image, best_gain = image, gain
        if best_image is None:
            break
        picked.append(best_image)
        pool.remove(best_image)
        for name, number in per_image_classes[best_image].items():
            if name in remaining:
                remaining[name] = max(0, remaining[name] - number)

    return Memory(tuple(picked), demand, alpha, total)


def herding_order(embeddings: np.ndarray) -> np.ndarray:
    """iCaRL's exemplar criterion: keep the set whose mean tracks the class mean.

    Repeatedly take the item that moves the running mean of what is already kept
    closest to the class mean. The result is an *ordering*, so a memory of any
    size is a prefix of it — which is the property iCaRL relies on when the
    per-class budget shrinks as new classes arrive.

    This is the one place where a standard incremental-learning method is
    reproduced rather than referenced. It is the exemplar-*selection* half of
    iCaRL; the nearest-mean-of-exemplars classifier is not applicable here,
    because the detector's own head does the classifying.
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    target = embeddings.mean(axis=0)
    running = np.zeros_like(target)
    remaining = np.ones(embeddings.shape[0], dtype=bool)
    order: list[int] = []
    for step in range(1, embeddings.shape[0] + 1):
        # distance from the class mean if each remaining item were taken next
        candidate = (running + embeddings) / step
        distance = np.linalg.norm(candidate - target, axis=1)
        distance[~remaining] = np.inf
        pick = int(np.argmin(distance))
        order.append(pick)
        remaining[pick] = False
        running = running + embeddings[pick]
    return np.asarray(order, dtype=np.int64)


#: Between tasks the memory is not patched, it is rebuilt: :func:`build` is
#: called again with the class set known now, so the object budget is re-satisfied
#: from scratch every task and cannot drift. ``replay_reallocate`` chooses only
#: *how* it is rebuilt — pass the previous memory as ``priority`` to keep the
#: exemplars we already had wherever they still serve the new allocation, or pass
#: nothing to re-derive it by set cover. An earlier ``carry_forward`` helper
#: unioned the image lists instead, which meant the memory grew every task and by
#: a different amount per arm; that made ``alpha`` arms differ in size as well as
#: in composition, so it was removed rather than fixed.


#: The registered replay arms, one per row of the consultation's table.
ARMS: dict[str, dict] = {
    "none": {"total": 0, "alpha": 0.0},
    "uniform": {"total": 400, "alpha": 0.0},
    "head_favouring": {"total": 400, "alpha": 1.0},
    "tail_favouring": {"total": 400, "alpha": -0.5},
    "tail_inverted": {"total": 400, "alpha": -1.0},
    "random_images": {"total": 400, "alpha": 0.0, "selector": "random"},
    "herding": {"total": 400, "alpha": 0.0, "selector": "herding"},
    "herding_tail": {"total": 400, "alpha": -0.5, "selector": "herding"},
}
