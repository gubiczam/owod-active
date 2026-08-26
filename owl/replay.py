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
    """Split ``total`` exemplars across classes by ``n_c ** alpha``.

    ``minimum`` guarantees every known class survives at all, which matters at
    ``alpha = 1`` where a tail class would otherwise round to zero and be
    forgotten completely — the failure the plan predicts for head-favouring
    allocation.
    """

    classes = [name for name, count in class_counts.items() if count > 0]
    if not classes or total <= 0:
        return {}
    counts = np.array([class_counts[name] for name in classes], dtype=np.float64)

    weights = np.power(counts, alpha)
    weights = weights / weights.sum()
    raw = weights * total

    allocation = np.maximum(np.floor(raw), minimum).astype(np.int64)
    allocation = np.minimum(allocation, counts.astype(np.int64))

    # hand the rounding remainder to whoever was cut hardest, largest first
    while allocation.sum() > total and (allocation > minimum).any():
        victim = int(np.argmax(allocation - minimum))
        allocation[victim] -= 1
    remainder = total - int(allocation.sum())
    if remainder > 0:
        order = np.argsort(-(raw - allocation))
        for index in order[:remainder]:
            if allocation[index] < counts[index]:
                allocation[index] += 1

    return {name: int(value) for name, value in zip(classes, allocation) if value > 0}


def build(
    per_image_classes: Mapping[str, Mapping[str, int]],
    known_classes: Sequence[str],
    *,
    total: int,
    alpha: float = 0.0,
    seed: int = 0,
    selector: str = "greedy",
) -> Memory:
    """Choose the images that satisfy the allocation as closely as possible.

    ``per_image_classes`` maps an image id to how many objects of each class it
    holds — one image usually serves several classes at once, which is why an
    object-level allocation cannot be turned into an image list by division.

    ``greedy`` repeatedly takes the image that covers the most still-unmet
    demand, which is a set-cover heuristic and reaches the target with far fewer
    images than sampling per class. ``random`` is the control.
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


def carry_forward(previous: Memory, new_images: Sequence[str], *, reallocate: bool) -> tuple[str, ...]:
    """Between-task bookkeeping.

    ``reallocate=False`` is the usual thing: keep last task's memory and add to
    it. ``reallocate=True`` is what the consultation asked to test — throw the
    old memory away and size a new one for the classes that are known *now*.
    """

    if reallocate:
        return tuple(new_images)
    return tuple(dict.fromkeys([*previous.image_ids, *new_images]))


#: The registered replay arms, one per row of the consultation's table.
ARMS: dict[str, dict] = {
    "none": {"total": 0, "alpha": 0.0},
    "uniform": {"total": 400, "alpha": 0.0},
    "head_favouring": {"total": 400, "alpha": 1.0},
    "tail_favouring": {"total": 400, "alpha": -0.5},
    "tail_inverted": {"total": 400, "alpha": -1.0},
    "random_images": {"total": 400, "alpha": 0.0, "selector": "random"},
}
