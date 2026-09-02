"""Discovery counting: distinct objects, and the redundancy kept visible.

The bug these tests pin: total discovery was counted as distinct annotated
objects while the head/medium/tail breakdown was counted as *proposals*, and both
went into one table. Because arms differ in how many near-duplicate boxes they
buy per object, the inflation was arm-dependent — 1.02x for the learning-free
control against 1.76x for the arm being advocated — and it reversed the
conclusion of the comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from owl import discovery
from owl.proposals import Candidates, Oracle

GROUPS = {"fire hydrant": "tail", "bear": "tail", "chair": "head", "sofa": "medium"}


@pytest.fixture
def pool() -> Candidates:
    """Eight proposals: one tail object seen three times, then one of each kind.

    Object 10 (a fire hydrant) carries three proposals. Any counter that treats
    proposals as discoveries will report three tail finds where there is one.
    """

    kind = np.array([
        "unknown", "unknown", "unknown",   # all three on object 10
        "unknown",                          # object 11, a second tail class
        "unknown",                          # object 12, head
        "known", "background", "background",
    ])
    class_name = np.array([
        "fire hydrant", "fire hydrant", "fire hydrant",
        "bear", "chair", "car", "", "",
    ])
    object_id = np.array([10, 10, 10, 11, 12, 90, -1, -1], dtype=np.int64)
    n = kind.size
    return Candidates(
        image_ids=np.array(["a", "a", "a", "b", "b", "c", "c", "d"]),
        boxes=np.tile(np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32), (n, 1)),
        embeddings=np.eye(n, 8, dtype=np.float32),
        posterior=np.full((n, 4), 0.25, dtype=np.float32),
        objectness=np.linspace(0.9, 0.1, n, dtype=np.float32),
        _oracle=Oracle(
            kind=kind, class_name=class_name, object_id=object_id,
            iou=np.full(n, 0.8, dtype=np.float32),
        ),
    )


def test_three_proposals_on_one_object_are_one_discovery(pool):
    result = discovery.discovery(pool, [0, 1, 2], groups=GROUPS)

    assert result.unknown_objects == 1
    assert result.unknown_proposals == 3
    assert result.objects_by_group["tail"] == 1
    assert result.proposals_by_group["tail"] == 3
    assert result.proposals_per_object == 3.0


def test_group_objects_sum_to_total_objects(pool):
    """The invariant the defect broke: bands are in the same unit as the total."""

    result = discovery.discovery(pool, range(len(pool)), groups=GROUPS)

    assert sum(result.objects_by_group.values()) == result.unknown_objects == 3
    assert sum(result.proposals_by_group.values()) == result.unknown_proposals == 5


def test_background_and_known_are_never_discoveries(pool):
    result = discovery.discovery(pool, [5, 6, 7], groups=GROUPS)

    assert result.unknown_objects == 0
    assert result.selected_known == 1
    assert result.selected_background == 2
    assert np.isnan(result.proposals_per_object)
    assert np.isnan(result.tail_share)


def test_the_two_cost_axes_are_both_reported(pool):
    """Per region and per opened image, because they can rank arms differently."""

    result = discovery.discovery(pool, [0, 1, 2, 3], groups=GROUPS)

    assert result.asked == 4
    assert result.images_opened == 2          # images 'a' and 'b'
    assert result.per_region("tail") == pytest.approx(2 / 4)
    assert result.per_image("tail") == pytest.approx(2 / 2)


def test_unknown_class_of_no_declared_group_is_counted_in_no_band(pool):
    """A class missing from the grouping must not be silently filed as head."""

    result = discovery.discovery(pool, [0, 4], groups={"fire hydrant": "tail"})

    assert result.unknown_objects == 2
    assert result.objects_by_group == {"head": 0, "medium": 0, "tail": 1}
    assert sum(result.objects_by_group.values()) < result.unknown_objects


def test_cumulative_does_not_double_count_across_rounds(pool):
    """Distinct objects do not add: the same object bought twice is one find."""

    rows = discovery.cumulative(
        pool, [0, 1, 3], [0, 1, 1], groups=GROUPS
    )

    assert [row["round"] for row in rows] == [0, 1]
    assert rows[0]["unknown_objects"] == 1                 # object 10
    assert rows[1]["unknown_objects"] == 2                 # objects 10 and 11
    assert rows[1]["unknown_proposals"] == 3               # but three boxes paid for
    # naive per-round addition would have said 1 + 2 = 3 tail objects
    assert rows[1]["tail_objects"] == 2


def test_empty_selection_is_answered_not_crashed(pool):
    result = discovery.discovery(pool, [], groups=GROUPS)

    assert result.asked == 0
    assert result.unknown_objects == 0
    assert np.isnan(result.per_region("tail"))
    assert np.isnan(result.per_image("tail"))
    assert isinstance(result.row(), dict)
