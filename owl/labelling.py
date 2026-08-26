"""What the annotator actually labels once a region has been chosen.

Point 5 of the 2026-08-25 consultation, and the one that distorts every other
measurement if it is decided wrongly. The selector points at a *region*. The
annotator is handed an *image*, and an image holds several objects.

Three policies, and they differ in what the detector is taught about the boxes
nobody asked about:

``box_only``
    Only the chosen boxes are labelled. Everything else in the image — including
    real objects of classes we already know — arrives as **background**. This is
    half-labelling, and it actively teaches the detector that a car is not a car.
    It is the cheapest per image and the reason forgetting looked catastrophic
    in the earlier work.

``full_image``
    Every annotated object in a chosen image is labelled. No half-labelling, and
    the consultation's default recommendation. It costs more oracle units per
    image, which is why the budget here is counted in regions and not in images:
    an arm that prefers crowded images pays for them.

``known_plus_selected``
    The middle option the consultation asked to run as its own arm. Objects of
    already-known classes are labelled **for free** — the detector can produce
    them itself, no human is needed — the chosen unknowns are labelled, and the
    remaining unknowns are marked **ignore** rather than background, so they
    contribute no gradient in either direction.

The third one is expected to help twice: it avoids half-labelling *and* the
known boxes on the chosen images act as replay that costs nothing. That effect
was already visible in the earlier work, where restoring the task-1 boxes that
were being discarded cut forgetting from 27 points to 2.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from owl.proposals import Candidates
from owl.selection import Selection

POLICIES = ("box_only", "full_image", "known_plus_selected")


@dataclass
class Annotation:
    """The outcome of one annotation round.

    ``oracle_cost`` is the number of regions a human had to look at and answer
    for. It is the quantity every arm is held equal on, so it is computed here
    from the policy rather than assumed to equal the budget.
    """

    policy: str
    images: np.ndarray            # image ids opened
    labelled: np.ndarray          # proposal indices with a usable class label
    background: np.ndarray        # proposal indices taught as background
    ignored: np.ndarray           # proposal indices excluded from the loss
    revealed_classes: tuple[str, ...]
    oracle_cost: int

    def summary(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "images": int(self.images.size),
            "labelled": int(self.labelled.size),
            "background": int(self.background.size),
            "ignored": int(self.ignored.size),
            "classes": len(self.revealed_classes),
            "oracle_cost": self.oracle_cost,
        }


def annotate(
    candidates: Candidates,
    selection: Selection,
    *,
    policy: str = "known_plus_selected",
    known_classes: tuple[str, ...] = (),
) -> Annotation:
    """Ask the oracle, under one policy, and price the answer.

    This is the only place in the codebase that reads
    :meth:`~owl.proposals.Candidates.oracle`. Selection never sees a label; the
    labels appear here, after the budget has been committed.
    """

    if policy not in POLICIES:
        raise ValueError(f"Unknown labelling policy {policy!r}; expected one of {POLICIES}.")

    oracle = candidates.oracle()
    chosen = np.zeros(len(candidates), dtype=bool)
    chosen[selection.indices] = True

    opened = np.unique(candidates.image_ids[selection.indices])
    on_opened = np.isin(candidates.image_ids, opened)

    is_object = oracle.kind != "background"
    is_known = np.isin(oracle.class_name, np.asarray(known_classes, dtype=str))

    if policy == "box_only":
        labelled = chosen & is_object
        # everything else on the opened images is taught as background,
        # including real objects nobody asked about
        background = on_opened & ~labelled
        ignored = np.zeros(len(candidates), dtype=bool)
        cost = int(chosen.sum())

    elif policy == "full_image":
        labelled = on_opened & is_object
        background = on_opened & ~is_object
        ignored = np.zeros(len(candidates), dtype=bool)
        # the annotator answers for every distinct object on every opened image
        cost = _distinct_objects(oracle, on_opened & is_object) + int(
            (on_opened & ~is_object & chosen).sum()
        )

    else:  # known_plus_selected
        labelled = on_opened & is_object & (is_known | chosen)
        ignored = on_opened & is_object & ~labelled
        background = on_opened & ~is_object
        # known objects are free: the detector already produces them
        cost = int(chosen.sum())

    revealed = tuple(sorted(set(oracle.class_name[labelled]) - {""}))
    return Annotation(
        policy=policy,
        images=opened,
        labelled=np.flatnonzero(labelled),
        background=np.flatnonzero(background),
        ignored=np.flatnonzero(ignored),
        revealed_classes=revealed,
        oracle_cost=cost,
    )


def _distinct_objects(oracle, mask: np.ndarray) -> int:
    """Two proposals on the same object cost the annotator once."""

    ids = oracle.object_id[mask]
    return int(np.unique(ids[ids >= 0]).size)


def half_labelling_rate(annotation: Annotation, candidates: Candidates) -> float:
    """The number this policy question is really about.

    The fraction of taught-as-background proposals that in truth sit on a real
    annotated object. Zero under ``full_image`` and under
    ``known_plus_selected``; whatever ``box_only`` costs is the size of the
    error the detector is being trained on.
    """

    if annotation.background.size == 0:
        return 0.0
    kinds = candidates.oracle().kind[annotation.background]
    return float((kinds != "background").mean())
