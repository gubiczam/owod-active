"""The per-box annotation policy, and the one claim that matters most.

Point 6 of the 2026-09 redesign brief and point 5 of the 2026-08-25
consultation: an unselected unknown object must **not** become a background
target. PROB has no ignore channel — ``owl.supervision`` documents the reading
of the pinned source — so the claim has to be checked against what PROB would
actually build from the annotation this repository writes, and that is what
:func:`owl.supervision.prob_training_target` and
:func:`owl.supervision.false_background` are for.

Every test here runs on a synthetic three-object VOC root. No GPU, no dataset,
no checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from owl import protocol, supervision

#: t2 of the canonical chain: nineteen task-1 classes known, `traffic light` new.
PREVIOUSLY_KNOWN = protocol.CLASS_ORDER[:19]
DECLARED = protocol.CLASS_ORDER[:20]
NEW_CLASS = protocol.CLASS_ORDER[19]          # traffic light
FUTURE_CLASS = protocol.CLASS_ORDER[20]       # fire hydrant, declared at t3

WIDTH, HEIGHT = 200, 100

#: One image, four objects, one of each interesting kind.
OBJECTS = (
    ("person", (10, 10, 50, 90)),           # previously known -> free supervision
    (NEW_CLASS, (60, 10, 100, 90)),         # the selected new object
    (NEW_CLASS, (110, 10, 150, 90)),        # a second new-class object, unselected
    (FUTURE_CLASS, (160, 10, 195, 90)),     # a class no task has declared yet
)


def _write_root(tmp_path, objects=OBJECTS, image_id="000000000042"):
    annotations = tmp_path / "Annotations"
    images = tmp_path / "JPEGImages"
    annotations.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    body = "".join(
        f"<object><name>{name}</name><difficult>0</difficult><bndbox>"
        f"<xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
        f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
        for name, b in objects
    )
    (annotations / f"{image_id}.xml").write_text(
        f"<annotation><filename>{image_id}.jpg</filename>"
        f"<size><width>{WIDTH}</width><height>{HEIGHT}</height>"
        f"<depth>3</depth></size>{body}</annotation>",
        encoding="utf-8",
    )
    from PIL import Image

    Image.new("RGB", (WIDTH, HEIGHT), (200, 30, 30)).save(images / f"{image_id}.jpg")
    return tmp_path, image_id


def _selection(box=OBJECTS[1][1]):
    """The second object, as a normalised ``cxcywh`` proposal."""

    x0, y0, x1, y1 = box
    return np.array([[
        (x0 + x1) / 2 / WIDTH, (y0 + y1) / 2 / HEIGHT,
        (x1 - x0) / WIDTH, (y1 - y0) / HEIGHT,
    ]])


def _plan(tmp_path, policy, mechanism="drop"):
    root, image_id = _write_root(tmp_path)
    return image_id, supervision.plan(
        root, {image_id: _selection()},
        policy=policy, previously_known=PREVIOUSLY_KNOWN, declared=DECLARED,
        ignore_mechanism=mechanism,
    )


# --------------------------------------------------------------- the policies ---


def test_known_plus_selected_gives_every_object_the_right_role(tmp_path):
    _, plan = _plan(tmp_path, "known_plus_selected_ignore_rest")
    roles = {(r.class_name, r.ordinal): r for r in plan.roles}

    assert roles[("person", 0)].role == "supervised"
    assert roles[("person", 0)].free is True, "a known box needs no annotator"
    assert roles[(NEW_CLASS, 0)].role == "supervised"
    assert roles[(NEW_CLASS, 0)].free is False, "the selected box is what we paid for"
    assert roles[(NEW_CLASS, 1)].role == "ignored", (
        "an unselected object is not something this policy asked about")
    assert roles[(FUTURE_CLASS, 0)].role == "ignored"

    summary = plan.summary()
    assert summary["known_boxes_reused"] == 1
    assert summary["new_boxes_supervised"] == 1
    assert summary["objects_ignored"] == 2
    assert summary["answers_charged"] == 1


def test_full_image_supervises_every_declared_object(tmp_path):
    _, plan = _plan(tmp_path, "full_image")
    roles = {(r.class_name, r.ordinal): r.role for r in plan.roles}
    assert roles[("person", 0)] == "supervised"
    assert roles[(NEW_CLASS, 0)] == "supervised"
    assert roles[(NEW_CLASS, 1)] == "supervised", (
        "full image means the annotator labelled the one nobody selected too")
    assert roles[(FUTURE_CLASS, 0)] == "banked", "not declared, so not trainable yet"
    assert plan.summary()["objects_ignored"] == 0


def test_selected_box_only_leaves_the_rest_as_background(tmp_path):
    _, plan = _plan(tmp_path, "selected_box_only")
    roles = {(r.class_name, r.ordinal): r.role for r in plan.roles}
    assert roles[(NEW_CLASS, 0)] == "supervised"
    # `person` is a real, already-known object and this policy throws it away.
    assert roles[("person", 0)] == "ignored"
    assert plan.summary()["objects_supervised"] == 1


def test_a_policy_without_an_ignore_set_refuses_an_ignore_mechanism(tmp_path):
    root, image_id = _write_root(tmp_path)
    with pytest.raises(supervision.SupervisionError, match="never performs"):
        supervision.plan(
            root, {image_id: _selection()}, policy="full_image",
            previously_known=PREVIOUSLY_KNOWN, declared=DECLARED,
            ignore_mechanism="pixel_suppression",
        )


def test_an_unmatched_selection_is_counted_not_hidden(tmp_path):
    root, image_id = _write_root(tmp_path)
    # a proposal in the empty top-left corner, overlapping nothing
    empty = np.array([[0.02, 0.02, 0.02, 0.02]])
    plan = supervision.plan(
        root, {image_id: empty}, policy="known_plus_selected_ignore_rest",
        previously_known=PREVIOUSLY_KNOWN, declared=DECLARED,
    )
    assert plan.diagnostics["selections_matching_no_object"] == 1


# ------------------------------------------- the claim: ignore is not background ---


def _target_and_offenders(tmp_path, mechanism):
    """Write the alias, build PROB's target from it, and audit the background."""

    image_id, plan = _plan(
        tmp_path, "known_plus_selected_ignore_rest", mechanism=mechanism)
    written = supervision.write_supervision(plan, data_root=tmp_path)
    (alias,) = written.aliases
    target = supervision.prob_training_target(
        tmp_path / "Annotations" / f"{alias}.xml",
        split_marker="owl_t2_ft", class_order=protocol.CLASS_ORDER,
        n_prev=19, n_current=1,
    )
    suppressed = [
        r.box for r in written.roles
        if r.role == "ignored" and mechanism == "pixel_suppression"
    ]
    offenders = supervision.false_background(
        tmp_path / "Annotations" / f"{image_id}.xml", target,
        suppressed_boxes=suppressed,
    )
    return written, target, offenders


def test_an_ignored_object_never_becomes_a_background_target(tmp_path):
    """The load-bearing assertion of the whole annotation redesign."""

    written, target, offenders = _target_and_offenders(tmp_path, "pixel_suppression")

    # PROB is handed exactly the supervised set, and nothing it drops.
    assert sorted(target["labels"]) == sorted(["person", NEW_CLASS])
    assert target["dropped_by_prob"] == [], (
        "the alias must not contain a class PROB's ft filter would remove; that "
        "removal is the background path this policy exists to close")
    # and no real annotated object is left teaching background.
    assert offenders == [], offenders
    assert written.diagnostics["images_pixel_suppressed"] == 1


def test_drop_is_named_honestly_because_it_does_teach_background(tmp_path):
    """The control. Same policy, no suppression: the ignore set *is* background."""

    _, target, offenders = _target_and_offenders(tmp_path, "drop")
    assert sorted(target["labels"]) == sorted(["person", NEW_CLASS])
    names = sorted(o["class_name"] for o in offenders)
    assert names == sorted([NEW_CLASS, FUTURE_CLASS]), (
        "with ignore_mechanism='drop' the two unselected objects are exactly the "
        "false background, which is what the pipeline did before this module")


def test_full_image_still_teaches_a_future_class_as_background(tmp_path):
    """Why the middle policy is worth running: today's default has the defect too."""

    root, image_id = _write_root(tmp_path)
    plan = supervision.plan(
        root, {image_id: _selection()}, policy="full_image",
        previously_known=PREVIOUSLY_KNOWN, declared=DECLARED,
    )
    written = supervision.write_supervision(plan, data_root=root)
    (alias,) = written.aliases
    target = supervision.prob_training_target(
        root / "Annotations" / f"{alias}.xml", split_marker="owl_t2_ft",
        class_order=protocol.CLASS_ORDER, n_prev=19, n_current=1,
    )
    offenders = supervision.false_background(
        root / "Annotations" / f"{image_id}.xml", target)
    assert [o["class_name"] for o in offenders] == [FUTURE_CLASS]


def test_prob_ignore_flag_refuses_with_the_patch_written_out(tmp_path):
    """The clean mechanism is a PROB change, and saying so is part of the record."""

    _, plan = _plan(
        tmp_path, "known_plus_selected_ignore_rest", mechanism="prob_ignore_flag")
    with pytest.raises(NotImplementedError, match="SetCriterion"):
        supervision.write_supervision(plan, data_root=tmp_path)


# ------------------------------------------------------------------- the alias ---


def test_the_alias_leaves_the_original_annotation_untouched(tmp_path):
    root, image_id = _write_root(tmp_path)
    before = (root / "Annotations" / f"{image_id}.xml").read_bytes()
    plan = supervision.plan(
        root, {image_id: _selection()},
        policy="known_plus_selected_ignore_rest",
        previously_known=PREVIOUSLY_KNOWN, declared=DECLARED)
    supervision.write_supervision(plan, data_root=root)
    assert (root / "Annotations" / f"{image_id}.xml").read_bytes() == before


def test_the_alias_id_is_digits_only_and_keeps_its_width():
    alias = supervision.alias_id("000000000042")
    assert alias == "800000000042"
    assert alias.isdigit() and len(alias) == 12, (
        "PROB's convert_image_id does int('2021' + id) and the evaluator's "
        "reverse conversion asserts a 12- or 6-digit remainder")
    with pytest.raises(supervision.SupervisionError):
        supervision.alias_id("800000000042")
    with pytest.raises(supervision.SupervisionError):
        supervision.alias_id("00000042_r")


def test_the_supervision_alias_cannot_collide_with_a_replay_alias():
    from owl import exemplars

    assert supervision.ALIAS_PREFIX != exemplars.alias_id("000000000042")[0], (
        "owl.exemplars clears 9*.xml and this module clears 8*.xml; sharing a "
        "prefix would let one erase the other's annotations mid-task")


def test_an_image_with_no_supervision_is_counted_not_written(tmp_path):
    """A selection that bought only future-class objects trains nothing now."""

    root, image_id = _write_root(
        tmp_path, objects=((FUTURE_CLASS, (10, 10, 50, 90)),))
    plan = supervision.plan(
        root, {image_id: _selection(box=(10, 10, 50, 90))},
        policy="known_plus_selected_ignore_rest",
        previously_known=PREVIOUSLY_KNOWN, declared=DECLARED)
    assert plan.summary()["objects_banked"] == 1
    written = supervision.write_supervision(plan, data_root=root)
    assert written.aliases == {}
    assert written.diagnostics["images_without_supervision"] == 1


def test_a_second_write_clears_the_previous_generation(tmp_path):
    root, image_id = _write_root(tmp_path)
    plan = supervision.plan(
        root, {image_id: _selection()}, policy="full_image",
        previously_known=PREVIOUSLY_KNOWN, declared=DECLARED)
    supervision.write_supervision(plan, data_root=root)
    first = sorted(p.name for p in (root / "Annotations").glob("8*.xml"))

    other_root, other_id = _write_root(tmp_path, image_id="000000000099")
    plan2 = supervision.plan(
        other_root, {other_id: _selection()}, policy="full_image",
        previously_known=PREVIOUSLY_KNOWN, declared=DECLARED)
    supervision.write_supervision(plan2, data_root=other_root)
    second = sorted(p.name for p in (root / "Annotations").glob("8*.xml"))
    assert first and second and first != second
    assert len(second) == 1, "a stale generation must not survive into the next task"


# ----------------------------------------------------------- PROB's own filter ---


def test_the_reimplemented_target_follows_probs_split_markers(tmp_path):
    """`train` keeps only the current classes, `ft` keeps every declared one."""

    root, image_id = _write_root(tmp_path)
    path = root / "Annotations" / f"{image_id}.xml"
    common = dict(class_order=protocol.CLASS_ORDER, n_prev=19, n_current=1)

    ft = supervision.prob_training_target(path, split_marker="owl_t2_ft", **common)
    assert sorted(ft["labels"]) == sorted(["person", NEW_CLASS, NEW_CLASS])
    assert [o["class_name"] for o in ft["dropped_by_prob"]] == [FUTURE_CLASS]

    train = supervision.prob_training_target(
        path, split_marker="owl_t2_train", **common)
    assert sorted(train["labels"]) == [NEW_CLASS, NEW_CLASS], (
        "remove_prev_class_and_unk_instances keeps range(prev, prev + current)")
    assert "person" in {o["class_name"] for o in train["dropped_by_prob"]}

    test = supervision.prob_training_target(
        path, split_marker="owl_shared_test_t10", **common)
    assert test["dropped_by_prob"] == [], "label_known_class_and_unknown drops nothing"
    assert test["labels"].count("unknown") == 1

    with pytest.raises(supervision.SupervisionError, match="none of PROB's branches"):
        supervision.prob_training_target(path, split_marker="owl_pool", **common)
