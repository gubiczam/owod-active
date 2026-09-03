"""The annotation ledger: what was asked, what was paid, what PROB was taught.

Benchmark V1 exists because Method V3 held the wrong quantity equal. Its budget
was 600 **regions**, and the audit measured what that bought: 1.62 supervised
boxes per region for the admissibility arm against 3.38 for entropy, a 2.09x
difference in the supervision the detector received at identical nominal cost.
An image budget is worse, not better — the same arms sit at 1.65 and 7.9 boxes
per opened image.

So the unit here is the **oracle answer**, and the policy is full-image
labelling:

    cost(image) = max(1, annotated objects on that image)

The annotator is handed an image and labels everything in it, which is the
2026-08-25 consultation's own answer to point 5 and removes half-labelling by
construction. Opening an image is the only thing that costs anything, so
*opening images is the acquisition unit* — a second selected region on an
already-open image buys nothing and is charged nothing. Every row records that
explicitly rather than reporting a region count that no longer means anything.

Three quantities that must never be conflated, all recorded separately:

``answers``
    what the oracle was charged. Equal across arms by construction, up to the
    at-most-one-image underspend :meth:`Ledger.affordable` allows.
``labelled boxes``
    every annotated object on every opened image. This is what the annotator
    produced, and it is *not* what the detector sees.
``supervised boxes``
    the subset whose class is declared at this task. PROB's ``ft`` mode keeps
    ``category_id in range(0, prev + current)`` and drops the rest, so an
    acquired object of a class that becomes declarable only at t3 contributes
    nothing now. It is banked, not lost — ``reuse_deferred_labels`` returns it at
    the task where its class is declared, at no further cost.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from owl import protocol

#: An opened image costs at least one answer even when the benchmark annotates
#: nothing on it. Without the floor an arm that opens only empty images would
#: acquire images for free and the budget loop would not terminate.
ANSWER_FLOOR = 1


def image_cost(counts: Mapping[str, int] | None) -> int:
    """Oracle answers charged for opening one image."""

    if not counts:
        return ANSWER_FLOOR
    return max(ANSWER_FLOOR, int(sum(int(value) for value in counts.values())))


def cost_function(
    candidate_index: Mapping[str, Mapping[str, int]],
) -> Callable[[str], int]:
    """``image id -> answers``, closed over the benchmark's own object counts."""

    def cost(image_id: str) -> int:
        return image_cost(candidate_index.get(str(image_id)))

    return cost


@dataclass
class Ledger:
    """Spends a fixed number of oracle answers, one image at a time."""

    budget: int
    spent: int = 0
    opened: list[str] = field(default_factory=list)
    costs: list[int] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(int(self.budget) - int(self.spent), 0)

    def affordable(self, cost: int) -> bool:
        """Whether one more image of this cost fits.

        The rule is the same for every arm: the next image in the arm's own
        order is taken if it fits and the campaign **stops** if it does not.
        Skipping an unaffordable image and continuing with a cheaper one would
        bias the tail of every campaign toward sparse images, so the campaign
        underspends by at most one image's cost instead, and that underspend is
        reported.
        """

        return int(cost) <= self.remaining

    def charge(self, image_id: str, cost: int) -> None:
        self.opened.append(str(image_id))
        self.costs.append(int(cost))
        self.spent += int(cost)

    def summary(self) -> dict[str, object]:
        return {
            "answer_budget": int(self.budget),
            "answers_spent": int(self.spent),
            "answers_unspent": int(self.remaining),
            "images_opened": len(self.opened),
            "answers_per_image": round(self.spent / max(len(self.opened), 1), 3),
        }


@dataclass(frozen=True)
class Spend:
    """The outcome of walking one arm's order until the budget is gone."""

    images: tuple[str, ...]        # opened, in the order they were opened
    anchors: tuple[int, ...]       # the proposal position that opened each image
    ledger: Ledger
    scanned: int                   # positions consulted before stopping
    redundant: int                 # positions on an already-open image

    def summary(self) -> dict[str, object]:
        return self.ledger.summary() | {
            "positions_scanned": self.scanned,
            "positions_redundant": self.redundant,
        }


def spend_ranking(
    order: Sequence[int] | np.ndarray,
    image_ids: np.ndarray,
    cost_of: Callable[[str], int],
    *,
    budget: int,
    excluded_images: frozenset[str] = frozenset(),
) -> Spend:
    """Open images in the order a static ranking names them.

    ``excluded_images`` are images bought at an earlier task. They are already
    fully labelled, so re-opening one would be charged for nothing; they are
    skipped rather than treated as free, which would let an arm inflate its
    apparent acquisition.
    """

    image_ids = np.asarray(image_ids, dtype=str)
    ledger = Ledger(budget=int(budget))
    images: list[str] = []
    anchors: list[int] = []
    seen: set[str] = set(excluded_images)
    scanned = 0
    redundant = 0
    for position in np.asarray(order, dtype=np.int64):
        scanned += 1
        image = str(image_ids[position])
        if image in seen:
            redundant += 1
            continue
        cost = cost_of(image)
        if not ledger.affordable(cost):
            break
        seen.add(image)
        ledger.charge(image, cost)
        images.append(image)
        anchors.append(int(position))
    return Spend(
        images=tuple(images), anchors=tuple(anchors),
        ledger=ledger, scanned=scanned, redundant=redundant,
    )


# ------------------------------------------------------------- supervision ---


def supervision(
    candidate_index: Mapping[str, Mapping[str, int]],
    images: Sequence[str],
    *,
    declared: Sequence[str],
    groups: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """What the oracle produced on ``images``, and what PROB will keep of it.

    ``declared`` is the classes known *after* this task, which is exactly the
    set ``remove_unknown_instances`` keeps. Everything else is banked.
    """

    groups = protocol.load_groups() if groups is None else groups
    declared_set = set(declared)
    per_class: dict[str, int] = {}
    for image in dict.fromkeys(str(value) for value in images):
        for name, count in (candidate_index.get(image) or {}).items():
            per_class[name] = per_class.get(name, 0) + int(count)

    supervised = {n: c for n, c in per_class.items() if n in declared_set}
    banked = {n: c for n, c in per_class.items() if n not in declared_set}
    by_group = {"head": 0, "medium": 0, "tail": 0}
    for name, count in supervised.items():
        group = groups.get(name)
        if group in by_group:
            by_group[group] += count

    barren = sum(
        1 for image in dict.fromkeys(str(v) for v in images)
        if not declared_set.intersection(candidate_index.get(image) or {})
    )
    total = sum(per_class.values())
    return {
        "boxes_labelled": total,
        "boxes_supervised": sum(supervised.values()),
        "boxes_banked": sum(banked.values()),
        "supervised_share": round(sum(supervised.values()) / max(total, 1), 4),
        "supervised_classes": len(supervised),
        "banked_classes": len(banked),
        "images_barren": barren,
        "boxes_supervised_head": by_group["head"],
        "boxes_supervised_medium": by_group["medium"],
        "boxes_supervised_tail": by_group["tail"],
        "per_class_supervised": ";".join(
            f"{name}:{count}" for name, count in sorted(supervised.items())
        ),
    }


def acquisition(
    candidate_index: Mapping[str, Mapping[str, int]],
    images: Sequence[str],
    *,
    chain: Sequence[protocol.Task],
    task_index: int,
    groups: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Point 15's table: when does what this task acquired become learnable.

    An object of a class that is unknown now may be declared at a later task of
    the same chain, or never inside it. Both are recorded, per task, so the
    chain can answer whether an acquisition that paid nothing at t2 paid at t3.

    This reads the benchmark's own annotation of images the oracle has **already
    been paid for**, after the budget was committed. It is never an input to a
    selector; :mod:`owl.active_selection.arms` cannot reach it.
    """

    groups = protocol.load_groups() if groups is None else groups
    current = chain[task_index]
    declared_now = set(current.known_classes)
    later: dict[str, set[str]] = {}
    for task in chain[task_index + 1:]:
        if task.new_class:
            later[task.name] = {task.new_class}

    per_class: dict[str, int] = {}
    for image in dict.fromkeys(str(value) for value in images):
        for name, count in (candidate_index.get(image) or {}).items():
            per_class[name] = per_class.get(name, 0) + int(count)

    row: dict[str, object] = {
        "acquired_objects": sum(per_class.values()),
        "acquired_classes": len(per_class),
        "acquired_known_now": sum(c for n, c in per_class.items() if n in declared_now),
        "acquired_new_class": per_class.get(current.new_class or "", 0),
    }
    accounted = set(declared_now)
    for name, members in later.items():
        value = sum(c for n, c in per_class.items() if n in members)
        row[f"acquired_becomes_known_{name}"] = value
        accounted |= members
    row["acquired_stays_unknown"] = sum(
        c for n, c in per_class.items() if n not in accounted
    )
    for group in ("head", "medium", "tail"):
        row[f"acquired_{group}_objects"] = sum(
            c for n, c in per_class.items() if groups.get(n) == group
        )
    return row
