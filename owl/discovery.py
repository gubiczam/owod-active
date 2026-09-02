"""What a selection actually bought, counted in distinct annotated objects.

One concept, one implementation, because counting it two ways produced a wrong
result that survived three seeds and two result documents.

**The defect this module exists to prevent.** ``tools/run_experiments.py``
counted total discovery as distinct ``object_id`` values but counted the
head/medium/tail breakdown as *proposals*. Both numbers went into the same table.
Two proposals sitting on one fire hydrant are one discovery and one oracle
answer, so the group columns were inflated — and not uniformly: inflation is a
property of the arm, because arms differ in how many near-duplicate boxes they
buy per object.

Measured inflation of the tail column at budget 600, three seeds:

======================  ==========  ==========  ===========
arm                     tail        tail        inflation
                        proposals   objects
======================  ==========  ==========  ===========
``objectness``          48.0        47.0        **1.02x**
``prior_consult``       49.3        34.0        1.45x
``prior_consult_batch`` 61.0        34.7        **1.76x**
======================  ==========  ==========  ===========

The comparison was therefore biased by roughly 1.7x in favour of the arm the
project was advocating, and it reversed the conclusion: on distinct objects the
learning-free control wins the tail column 47.0 against 34.7, where the
proposal count said 48.0 against 61.0.

So every discovery quantity in this module is a **distinct-object** count, the
redundancy is reported as its own column rather than being folded into the
totals, and :func:`discovery` is the only place that decides either.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from owl.proposals import Candidates

#: Frequency bands, in report order.
GROUPS: tuple[str, ...] = ("head", "medium", "tail")


@dataclass(frozen=True)
class Discovery:
    """One selection's yield. Distinct objects unless a name says ``proposals``."""

    asked: int                               # oracle regions actually spent
    images_opened: int

    unknown_objects: int                     # distinct annotated unknown objects
    unknown_proposals: int                   # boxes that landed on one of them
    unknown_classes: int
    objects_by_group: Mapping[str, int] = field(default_factory=dict)
    proposals_by_group: Mapping[str, int] = field(default_factory=dict)
    classes_by_group: Mapping[str, int] = field(default_factory=dict)

    selected_background: int = 0
    selected_known: int = 0
    selected_unknown: int = 0                # == unknown_proposals, named for the split

    @property
    def proposals_per_object(self) -> float:
        """Redundancy. 1.0 is one box per object; 1.7 is paying 1.7x for the same find."""

        if not self.unknown_objects:
            return float("nan")
        return self.unknown_proposals / self.unknown_objects

    @property
    def tail_share(self) -> float:
        """Tail's share of distinct discoveries — composition, not volume."""

        if not self.unknown_objects:
            return float("nan")
        return self.objects_by_group.get("tail", 0) / self.unknown_objects

    def per_region(self, name: str = "tail") -> float:
        """Distinct objects of one band per oracle region — A1's primary endpoint."""

        if not self.asked:
            return float("nan")
        return self.objects_by_group.get(name, 0) / self.asked

    def per_image(self, name: str = "tail") -> float:
        """Distinct objects of one band per **opened image**.

        Reported because the two cost axes rank the arms differently, and which
        one is right is a question about annotation practice rather than about
        the selector: an arm that opens 548 images for 600 regions and one that
        opens 308 are not paying the same price outside a strict region budget.
        A1's frozen primary endpoint is :meth:`per_region`; this is a declared
        secondary, and it is not permitted to replace the primary after the fact.
        """

        if not self.images_opened:
            return float("nan")
        return self.objects_by_group.get(name, 0) / self.images_opened

    def row(self) -> dict[str, object]:
        """Flat record for a CSV. Every discovery column says which unit it is in."""

        out: dict[str, object] = {
            "asked": self.asked,
            "images_opened": self.images_opened,
            "unknown_objects": self.unknown_objects,
            "unknown_proposals": self.unknown_proposals,
            "proposals_per_object": round(self.proposals_per_object, 4),
            "unknown_classes": self.unknown_classes,
        }
        for name in GROUPS:
            out[f"{name}_objects"] = self.objects_by_group.get(name, 0)
        for name in GROUPS:
            out[f"{name}_proposals"] = self.proposals_by_group.get(name, 0)
        for name in GROUPS:
            out[f"{name}_classes"] = self.classes_by_group.get(name, 0)
        out |= {
            "tail_share_objects": round(self.tail_share, 4),
            "tail_objects_per_region": round(self.per_region("tail"), 6),
            "tail_objects_per_image": round(self.per_image("tail"), 6),
            "selected_background": self.selected_background,
            "selected_known": self.selected_known,
            "selected_unknown": self.selected_unknown,
        }
        return out


def _distinct(object_ids: np.ndarray) -> int:
    valid = object_ids[object_ids >= 0]
    return int(np.unique(valid).size)


def discovery(
    candidates: Candidates,
    indices: Sequence[int] | np.ndarray,
    *,
    groups: Mapping[str, str],
) -> Discovery:
    """Score one selection retrospectively.

    Reads the oracle, and says so: this is evaluation, never acquisition.
    ``groups`` maps class name to ``head`` / ``medium`` / ``tail``.

    An object with no ``object_id`` (background) is never counted as a discovery,
    and an unknown object matched by several proposals is counted once — the
    annotator was asked once about that object even if several boxes pointed at it.
    """

    index = np.asarray(indices, dtype=np.int64)
    oracle = candidates.oracle()

    kind = oracle.kind[index]
    found = kind == "unknown"
    object_ids = oracle.object_id[index][found]
    class_names = oracle.class_name[index][found]
    band = np.asarray([groups.get(name, "") for name in class_names])

    objects_by_group, proposals_by_group, classes_by_group = {}, {}, {}
    for name in GROUPS:
        mask = band == name
        objects_by_group[name] = _distinct(object_ids[mask])
        proposals_by_group[name] = int(mask.sum())
        classes_by_group[name] = int(np.unique(class_names[mask]).size)

    return Discovery(
        asked=int(index.size),
        images_opened=int(np.unique(candidates.image_ids[index]).size),
        unknown_objects=_distinct(object_ids),
        unknown_proposals=int(found.sum()),
        unknown_classes=int(np.unique(class_names).size),
        objects_by_group=objects_by_group,
        proposals_by_group=proposals_by_group,
        classes_by_group=classes_by_group,
        selected_background=int((kind == "background").sum()),
        selected_known=int((kind == "known").sum()),
        selected_unknown=int(found.sum()),
    )


def cumulative(
    candidates: Candidates,
    indices: Sequence[int] | np.ndarray,
    round_of: Sequence[int] | np.ndarray,
    *,
    groups: Mapping[str, str],
) -> list[dict[str, object]]:
    """One row per round, each scoring everything bought **up to and including** it.

    This is the A1.4 table. Cumulative rather than per-round because the question
    is whether the campaign ends up somewhere different, not whether one round
    differs — and because distinct-object counts do not add across rounds: the
    same object can be bought twice in two different rounds, and that is one
    discovery, which a per-round table would silently double.
    """

    index = np.asarray(indices, dtype=np.int64)
    rounds = np.asarray(round_of, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for value in sorted(set(rounds.tolist())):
        so_far = index[rounds <= value]
        rows.append({"round": int(value)} | discovery(
            candidates, so_far, groups=groups
        ).row())
    return rows
