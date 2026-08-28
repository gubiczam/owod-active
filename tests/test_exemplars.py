"""The object-level exemplar memory, and the alias annotations that realise it.

What these pin is the one claim Replay Protocol V3 exists to make: the amount of
previous-class supervision a replay arm delivers is the same for every arm, and
only its class composition moves. That claim lives in two places — the selector
takes exactly ``m_c`` objects, and the alias annotation hands PROB exactly those
and nothing else — so both are tested against real XML rather than against a
mock of it.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest

from owl import exemplars, protocol, replay


def write_annotation(root: Path, image_id: str, objects: list[tuple[str, int]]) -> Path:
    """One VOC annotation. ``objects`` is a list of (class name, box offset)."""

    boxes = "".join(
        f"<object><name>{name}</name><difficult>0</difficult>"
        f"<bndbox><xmin>{offset}</xmin><ymin>{offset}</ymin>"
        f"<xmax>{offset + 8}</xmax><ymax>{offset + 8}</ymax></bndbox></object>"
        for name, offset in objects
    )
    path = root / "Annotations" / f"{image_id}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<annotation><folder>OWOD</folder>"
        f"<filename>{image_id}.jpg</filename>"
        "<size><width>640</width><height>480</height><depth>3</depth></size>"
        "<segmented>0</segmented>" + boxes + "</annotation>",
        encoding="utf-8",
    )
    images = root / "JPEGImages"
    images.mkdir(parents=True, exist_ok=True)
    (images / f"{image_id}.jpg").write_bytes(b"pixels-for-" + image_id.encode())
    return path


def boxes_in(path: Path) -> list[tuple[str, str]]:
    """(class name, xmin) for every object in an annotation, in document order."""

    root = ElementTree.parse(path).getroot()
    return [
        (element.findtext("name", ""), element.findtext("bndbox/xmin", ""))
        for element in root.findall("object")
    ]


# --------------------------------------------------------------- the alias id ---


def test_the_alias_is_reversible_and_cannot_collide():
    assert exemplars.alias_id("000000021740") == "900000021740"
    assert exemplars.source_id("900000021740") == "000000021740"

    # PROB casts image ids to int, so a suffix scheme would raise inside the loader
    with pytest.raises(exemplars.ExemplarError, match="not numeric"):
        exemplars.alias_id("000000021740_r")
    # and an id outside the benchmark's zero-padded shape has no reversible alias
    with pytest.raises(exemplars.ExemplarError, match="does not start with zero"):
        exemplars.alias_id("900000021740")


# ------------------------------------------------------------ the object pool ---


def test_the_pool_is_enumerated_as_objects_not_images():
    pool = {"000000000001": {"person": 3, "car": 1}, "000000000002": {"bear": 1}}
    items = exemplars.enumerate_pool(pool, ("person", "car", "bear"))

    assert len(items) == 5
    assert exemplars.capacities(items) == {"person": 3, "car": 1, "bear": 1}
    # three people on one image are three distinct exemplars
    assert sum(1 for i in items if i.class_name == "person") == 3
    assert {i.ordinal for i in items if i.class_name == "person"} == {0, 1, 2}


def test_a_class_outside_the_previous_set_is_not_storable():
    pool = {"000000000001": {"person": 2, "traffic light": 4}}
    items = exemplars.enumerate_pool(pool, ("person",))

    assert exemplars.capacities(items) == {"person": 2}


# -------------------------------------------------------------- the selection ---


@pytest.fixture
def long_tailed_pool():
    """Nineteen classes over the canonical split's own 203x count range.

    Scaled down by ten rather than by more: at ``alpha = -1`` the rarest class is
    allocated about a quarter of the budget, so a pool that capped it would test
    the cap instead of the rule. The real pool holds 1,294 bears and the memory
    wants 101 of them; this holds 129.
    """

    counts = protocol.load_train_counts()
    scaled = {name: max(2, counts[name] // 10) for name in protocol.TASK1}
    pool = {}
    for position, (name, total) in enumerate(sorted(scaled.items())):
        for index in range(total):
            image = f"{position * 10_000 + index:012d}"
            pool.setdefault(image, {})[name] = 1
    return pool


@pytest.mark.parametrize("alpha", [1.0, 0.0, -0.5, -1.0])
def test_the_selection_takes_exactly_the_allocated_objects(long_tailed_pool, alpha):
    """B: per-class delivered counts equal the allocator's m_c, exactly."""

    items = exemplars.enumerate_pool(long_tailed_pool, protocol.TASK1)
    demand = replay.allocate(
        exemplars.capacities(items), total=400, alpha=alpha
    )
    chosen = exemplars.select(items, demand, seed=0)

    assert sum(demand.values()) == 400
    assert len(chosen) == 400
    assert exemplars.delivered_per_class(chosen) == {
        name: count for name, count in demand.items() if count > 0
    }


def test_alpha_moves_the_composition_and_not_the_total(long_tailed_pool):
    items = exemplars.enumerate_pool(long_tailed_pool, protocol.TASK1)
    groups = protocol.load_groups()

    tails = {}
    for alpha in (1.0, -1.0):
        demand = replay.allocate(exemplars.capacities(items), total=400, alpha=alpha)
        chosen = exemplars.select(items, demand, seed=0)
        assert len(chosen) == 400
        tails[alpha] = sum(
            1 for item in chosen if groups.get(item.class_name) == "tail"
        )
    assert tails[-1.0] > tails[1.0] * 3, tails


def test_the_same_objects_are_preferred_whatever_alpha_is(long_tailed_pool):
    """Alpha decides how many of a class, never which ones.

    Otherwise two arms would differ in composition *and* in which individual
    objects they ever saw, and a difference between them could be either.
    """

    items = exemplars.enumerate_pool(long_tailed_pool, protocol.TASK1)
    picked = {}
    for alpha in (0.0, -1.0):
        demand = replay.allocate(exemplars.capacities(items), total=400, alpha=alpha)
        picked[alpha] = exemplars.select(items, demand, seed=0)

    name = "bear"
    few = [i for i in picked[0.0] if i.class_name == name]
    many = [i for i in picked[-1.0] if i.class_name == name]
    assert len(many) > len(few)
    assert set(few) <= set(many), "the smaller allocation is not a prefix of the larger"


def test_selection_is_deterministic_under_the_seed(long_tailed_pool):
    items = exemplars.enumerate_pool(long_tailed_pool, protocol.TASK1)
    demand = replay.allocate(exemplars.capacities(items), total=400, alpha=-0.5)

    assert exemplars.select(items, demand, seed=0) == exemplars.select(
        items, demand, seed=0
    )
    assert exemplars.select(items, demand, seed=1) != exemplars.select(
        items, demand, seed=0
    )


def test_keeping_incumbents_is_what_reallocate_false_means():
    """G: an exemplar already held stays held wherever the quota still allows it."""

    pool = {f"{i:012d}": {"person": 1} for i in range(20)}
    items = exemplars.enumerate_pool(pool, ("person",))

    first = exemplars.select(items, {"person": 5}, seed=0)
    kept = exemplars.select(items, {"person": 5}, incumbent=first, seed=0)
    assert set(kept) == set(first), "the incumbent memory was not preserved"

    # a smaller quota evicts the surplus and nothing else
    smaller = exemplars.select(items, {"person": 3}, incumbent=first, seed=0)
    assert len(smaller) == 3
    assert set(smaller) <= set(first)

    # a larger quota keeps everything and tops up
    larger = exemplars.select(items, {"person": 8}, incumbent=first, seed=0)
    assert len(larger) == 8
    assert set(first) <= set(larger)


def test_reallocating_may_replace_incumbents_but_only_from_the_pool():
    """H: ``True`` re-derives freely, and still cannot leave the bounded pool."""

    pool = {f"{i:012d}": {"person": 1} for i in range(20)}
    items = exemplars.enumerate_pool(pool, ("person",))
    incumbent = exemplars.select(items, {"person": 5}, seed=7)

    bounded = exemplars.enumerate_pool(
        {f"{i:012d}": {"person": 1} for i in range(10)}, ("person",)
    )
    redrawn = exemplars.select(
        bounded, {"person": 5}, incumbent=incumbent, reallocate=True, seed=0
    )
    assert len(redrawn) == 5
    assert set(redrawn) <= set(bounded), "reallocation escaped the eligible pool"


def test_a_demand_the_pool_cannot_meet_is_refused_rather_than_shortened():
    items = exemplars.enumerate_pool({"000000000001": {"person": 2}}, ("person",))
    with pytest.raises(exemplars.ExemplarError, match="pool holds"):
        exemplars.select(items, {"person": 5}, seed=0)


# --------------------------------------------------------------- the aliasing ---


def test_an_alias_holds_exactly_the_selected_boxes(tmp_path):
    """C and D: only the selected boxes, and nothing else that shares the image."""

    write_annotation(tmp_path, "000000000001", [
        ("person", 10),          # person ordinal 0  <- selected
        ("traffic light", 20),   # the class this task introduces
        ("person", 30),          # person ordinal 1
        ("toothbrush", 40),      # a later task's class
        ("bear", 50),            # bear ordinal 0    <- selected
    ])
    original = (tmp_path / "Annotations" / "000000000001.xml").read_bytes()

    chosen = [
        exemplars.Exemplar("000000000001", "person", 0),
        exemplars.Exemplar("000000000001", "bear", 0),
    ]
    mapping = exemplars.write_aliases(chosen, data_root=tmp_path)

    assert mapping == {"900000000001": "000000000001"}
    alias = tmp_path / "Annotations" / "900000000001.xml"
    assert boxes_in(alias) == [("person", "10"), ("bear", "50")]

    # J: the original is byte-identical, and its pixels were not copied over
    assert (tmp_path / "Annotations" / "000000000001.xml").read_bytes() == original
    assert (tmp_path / "JPEGImages" / "900000000001.jpg").exists()
    assert (tmp_path / "JPEGImages" / "000000000001.jpg").read_bytes().startswith(
        b"pixels-for-"
    )


def test_two_selected_exemplars_on_one_image_both_survive(tmp_path):
    """E: an image is stored once and carries every exemplar selected on it."""

    write_annotation(tmp_path, "000000000002", [
        ("person", 10), ("person", 20), ("person", 30), ("car", 40),
    ])
    chosen = [
        exemplars.Exemplar("000000000002", "person", 0),
        exemplars.Exemplar("000000000002", "person", 2),
        exemplars.Exemplar("000000000002", "car", 0),
    ]
    mapping = exemplars.write_aliases(chosen, data_root=tmp_path)

    assert len(mapping) == 1, "one alias per source image"
    alias = tmp_path / "Annotations" / "900000000002.xml"
    assert boxes_in(alias) == [("person", "10"), ("person", "30"), ("car", "40")]


def test_the_alias_keeps_what_probs_loader_reads(tmp_path):
    """PROB reads <filename> and <size>; a missing size raises inside the loader."""

    write_annotation(tmp_path, "000000000003", [("person", 10)])
    exemplars.write_aliases(
        [exemplars.Exemplar("000000000003", "person", 0)], data_root=tmp_path
    )

    root = ElementTree.parse(tmp_path / "Annotations" / "900000000003.xml").getroot()
    assert root.findtext("filename") == "900000000003.jpg"
    assert root.findtext("size/width") == "640"
    assert root.findtext("size/height") == "480"
    assert len(root.findall("object")) == 1


def test_the_cocofied_spelling_is_resolved_the_way_owl_names_classes(tmp_path):
    """The XMLs say 'motorcycle'; every table in owl says 'motorbike'."""

    write_annotation(tmp_path, "000000000004", [("motorcycle", 10), ("person", 20)])
    exemplars.write_aliases(
        [exemplars.Exemplar("000000000004", "motorbike", 0)], data_root=tmp_path
    )

    alias = tmp_path / "Annotations" / "900000000004.xml"
    # the box is kept verbatim, spelling included: PROB's own loader maps it
    assert boxes_in(alias) == [("motorcycle", "10")]


def test_an_alias_that_would_have_no_boxes_is_refused(tmp_path):
    write_annotation(tmp_path, "000000000005", [("person", 10)])
    with pytest.raises(exemplars.ExemplarError, match="only 0 matched"):
        exemplars.write_aliases(
            [exemplars.Exemplar("000000000005", "person", 4)], data_root=tmp_path
        )


def test_writing_a_memory_removes_the_one_before_it(tmp_path):
    """The directory holds one memory, so the budget is verifiable at any moment.

    Two consecutive memories sharing a source image would otherwise leave the
    later annotation in place of the earlier, and counting a finished task's
    boxes afterwards would give the wrong number.
    """

    write_annotation(tmp_path, "000000000007", [("person", 10), ("person", 20)])
    write_annotation(tmp_path, "000000000008", [("bear", 10)])

    exemplars.write_aliases([
        exemplars.Exemplar("000000000007", "person", 0),
        exemplars.Exemplar("000000000007", "person", 1),
        exemplars.Exemplar("000000000008", "bear", 0),
    ], data_root=tmp_path)
    first = sorted(p.name for p in (tmp_path / "Annotations").glob("9*.xml"))
    assert first == ["900000000007.xml", "900000000008.xml"]

    # the next memory drops one image and shrinks the other
    exemplars.write_aliases(
        [exemplars.Exemplar("000000000007", "person", 1)], data_root=tmp_path
    )
    second = sorted(p.name for p in (tmp_path / "Annotations").glob("9*.xml"))
    assert second == ["900000000007.xml"], "the previous memory was left behind"
    assert boxes_in(tmp_path / "Annotations" / "900000000007.xml") == [("person", "20")]

    # and the originals are still untouched
    assert boxes_in(tmp_path / "Annotations" / "000000000007.xml") == [
        ("person", "10"), ("person", "20")
    ]


def test_a_missing_annotation_says_so_rather_than_writing_a_short_memory(tmp_path):
    (tmp_path / "Annotations").mkdir(parents=True)
    with pytest.raises(exemplars.ExemplarError, match="is missing"):
        exemplars.write_aliases(
            [exemplars.Exemplar("000000000006", "person", 0)], data_root=tmp_path
        )
