"""What the detector is actually taught on an opened image — per box, not per mode.

The 2026-08-25 consultation's point 5, finished. ``owl.labelling`` answers the
*accounting* question on the frozen CPU pool: what three policies cost and how
much half-labelling each causes. It has never reached a detector. On the GPU
path the policy collapsed to PROB's ``--supervision-mode`` switch, and that
switch has two settings for three policies, so ``full_image`` and
``known_plus_selected`` were the **same run**:

    owl/runner.py:  supervision = "train" if policy == "box_only" else "ft"

This module writes the annotation instead of naming a mode, which is the only
place a per-box rule can live.

What PROB does with an annotation, read from the pinned source
==============================================================

``datasets/torchvision_datasets/open_world.py`` at ``4c66be1``:

* ``load_instances`` parses **every** ``<object>`` of ``Annotations/<id>.xml``
  and reads ``name`` and ``bndbox``. It does **not** read ``difficult``, and it
  does not read any other flag.
* ``__getitem__`` then filters by split-name marker — ``ft`` keeps
  ``category_id in range(0, prev + current)`` via ``remove_unknown_instances``
  — and builds ``labels``, ``boxes``, ``area``, ``iscrowd = zeros``.
* the loss is Deformable-DETR set prediction: queries the Hungarian matcher does
  not assign to a target get the no-object class.

**There is therefore no ignore channel.** A box left out of the XML is not
ignored; its region is taught as background. ``difficult`` *is* honoured, but
only by the evaluator (``open_world_eval.py`` excludes difficult boxes from both
the positive count and the FP count) — never by the training loop. So a policy
that says "ignore the rest" cannot be delivered by editing labels alone, and
:data:`IGNORE_MECHANISMS` names the three things that can be done about it
instead of quietly picking one.

The three policies
==================

``selected_box_only``
    Only the objects a selected proposal matched are supervised. Every other
    annotated object on the image — including objects of classes the detector
    already knows — is absent, and therefore taught as background. This is
    half-labelling, it is what ``--supervision-mode train`` delivers, and it is
    kept because it is the measured cause of catastrophic forgetting in this
    project's earlier work (27 points, cut to 2.7 by putting the boxes back).

``full_image``
    Every annotated object of a declared class is supervised. Objects of classes
    not yet declared are dropped by PROB and taught as background. This is
    exactly what the committed chain runs today under ``ft``, so it is the
    baseline the other two are one variable from.

``known_plus_selected_ignore_rest``
    The consultation's middle option. Objects of **previously known** classes
    are supervised and cost nothing — the detector already produces them, no
    human is needed. Objects a selected proposal matched are supervised and cost
    one answer each. Every remaining annotated object is **ignored**: it must not
    become a background target. Which is where :data:`IGNORE_MECHANISMS` comes in.

Ignore, honestly
================

``drop``
    Leave the ignored object out of the XML. This is what the pipeline does
    today and it is **not** ignore — the region becomes background. Offered as
    the named control, so the difference the mechanism makes is measurable
    rather than assumed.
``pixel_suppression``
    Fill the ignored object's box with the dataset mean colour in a *derived*
    JPEG. The region then genuinely holds no object, so "background" is the
    correct target rather than a false one. This is the mechanism that delivers
    the consultation's requirement without touching PROB, and it is an
    implementation decision, not something either source document asks for: it
    trades a false negative for a small distribution shift, and the shift is
    reported (``pixels_suppressed_share``).
``prob_ignore_flag``
    The clean route, and not available from here: a real ignore mask inside
    ``SetCriterion`` — mask the no-object term for queries whose box overlaps an
    ignore region — which is a change to the PROB fork, not to this repository.
    Selecting it raises with the patch spelled out, so the option is on the
    record instead of being an unwritten intention.

Nothing in this module is reachable from a selector. It runs after the ledger
has been charged, which is when a human would have answered.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from owl.evaluation_subset import canonical_class_name

POLICIES: tuple[str, ...] = (
    "selected_box_only", "full_image", "known_plus_selected_ignore_rest",
)
IGNORE_MECHANISMS: tuple[str, ...] = ("drop", "pixel_suppression", "prob_ignore_flag")

#: Roles an annotated object can be given on an opened image.
ROLES: tuple[str, ...] = ("supervised", "ignored", "banked")

#: IoU at which a selected proposal is taken to be *about* an annotated object.
#: The value the whole repository already matches proposals to annotations at
#: (``owl.proposals``, ``docs/method.md``); not introduced here and not swept.
MATCH_IOU = 0.5

#: The alias id prefix for a supervision-filtered annotation. ``owl.exemplars``
#: owns ``9``; this owns ``8``. Both must be digits only, because
#: ``OWDetection.convert_image_id`` does ``int('2021' + id)``, and both keep the
#: source id's width because the evaluator's reverse conversion asserts a 12- or
#: 6-digit remainder.
ALIAS_PREFIX = "8"

#: COCO's mean pixel, as the fill for a suppressed region. PROB normalises with
#: ImageNet statistics, so a mean-coloured patch lands near zero after
#: normalisation — the least informative thing that can be put there. A single
#: fixed colour rather than blur or noise, because it is reproducible from this
#: constant alone.
SUPPRESSION_FILL = (124, 116, 104)


class SupervisionError(ValueError):
    """Raised when a supervision plan cannot be honoured."""


def default_ignore_mechanism(policy: str) -> str:
    """The mechanism a policy uses unless one is named.

    ``selected_box_only`` and ``full_image`` have no ignore set at all, so their
    only honest mechanism is ``drop``. ``known_plus_selected_ignore_rest`` is the
    policy whose whole point is the ignore set, so it defaults to the mechanism
    that actually delivers one.
    """

    if policy not in POLICIES:
        raise SupervisionError(f"Unknown policy {policy!r}; expected one of {POLICIES}")
    return (
        "pixel_suppression"
        if policy == "known_plus_selected_ignore_rest"
        else "drop"
    )


@dataclass(frozen=True)
class ObjectRole:
    """One annotated object on one opened image, and what it becomes."""

    image_id: str
    class_name: str
    ordinal: int              # index among this image's objects of this class
    box: tuple[float, float, float, float]   # xmin, ymin, xmax, ymax in pixels
    role: str                 # one of ROLES
    matched: bool             # a selected proposal pointed at it
    free: bool                # no oracle answer was charged for it
    reason: str


@dataclass(frozen=True)
class Supervision:
    """The plan for one task's opened images, and its accounting."""

    policy: str
    ignore_mechanism: str
    roles: tuple[ObjectRole, ...]
    #: ``alias id -> source id`` for the images an alias was written for. Empty
    #: until :func:`write_supervision` runs.
    aliases: Mapping[str, str] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def of_role(self, role: str) -> tuple[ObjectRole, ...]:
        return tuple(r for r in self.roles if r.role == role)

    def summary(self) -> dict[str, object]:
        supervised = self.of_role("supervised")
        return {
            "annotation_policy": self.policy,
            "ignore_mechanism": self.ignore_mechanism,
            "images_opened": len({r.image_id for r in self.roles}),
            "objects_labelled": len(self.roles),
            "objects_supervised": len(supervised),
            "objects_ignored": len(self.of_role("ignored")),
            "objects_banked": len(self.of_role("banked")),
            "known_boxes_reused": sum(1 for r in supervised if r.free),
            "new_boxes_supervised": sum(1 for r in supervised if not r.free),
            "answers_charged": sum(1 for r in self.roles if not r.free),
        } | dict(self.diagnostics)


# --------------------------------------------------------------- the geometry ---


def iou(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    """IoU of one ``xyxy`` box against many. Zero-area inputs give zero."""

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros(0, dtype=np.float64)
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def cxcywh_to_xyxy(
    boxes: np.ndarray, *, width: float, height: float
) -> np.ndarray:
    """Normalised ``cxcywh`` (PROB's proposal format) to pixel ``xyxy``."""

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx = boxes[:, 0] * width
    cy = boxes[:, 1] * height
    half_w = boxes[:, 2] * width / 2.0
    half_h = boxes[:, 3] * height / 2.0
    return np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)


def read_annotation(path: Path) -> tuple[tuple[int, int], list[dict]]:
    """``(width, height)`` and one record per ``<object>``, in document order.

    Document order is the identity :class:`ObjectRole` and
    :class:`owl.exemplars.Exemplar` both index by, so it is preserved rather
    than sorted.
    """

    tree = ElementTree.parse(path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise SupervisionError(f"{path} has no <size>; a box cannot be placed in it")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))

    records: list[dict] = []
    seen: dict[str, int] = defaultdict(int)
    for element in root.findall("object"):
        name = canonical_class_name(element.findtext("name", ""))
        boxnode = element.find("bndbox")
        if boxnode is None:
            raise SupervisionError(f"{path} has an <object> with no <bndbox>")
        box = tuple(
            float(boxnode.findtext(key, "0"))
            for key in ("xmin", "ymin", "xmax", "ymax")
        )
        records.append({
            "class_name": name, "ordinal": seen[name], "box": box,
            "element": element,
        })
        seen[name] += 1
    return (width, height), records


# ----------------------------------------------------------------- the policy ---


def classify_image(
    image_id: str,
    records: Sequence[Mapping[str, object]],
    selected_boxes: np.ndarray,
    *,
    policy: str,
    previously_known: frozenset[str],
    declared: frozenset[str],
    match_iou: float = MATCH_IOU,
) -> list[ObjectRole]:
    """Give every annotated object on one image its role under ``policy``.

    ``previously_known`` are the classes declared *before* this task — the ones
    the detector can already produce, and therefore the ones a human does not
    have to be paid for. ``declared`` are the classes known *after* this task,
    which is exactly what ``remove_unknown_instances`` keeps: an object outside
    it cannot be trained on now however it was paid for, so it is ``banked``.

    ``selected_boxes`` are the proposals this arm bought on this image, in pixel
    ``xyxy``. An annotated object is *matched* when some selected proposal
    overlaps it at ``match_iou`` or better. A selected proposal that matches
    nothing bought background, which is a real outcome and is counted.
    """

    if policy not in POLICIES:
        raise SupervisionError(f"Unknown policy {policy!r}; expected one of {POLICIES}")

    boxes = np.asarray(selected_boxes, dtype=np.float64).reshape(-1, 4)
    roles: list[ObjectRole] = []
    for record in records:
        box = tuple(float(v) for v in record["box"])  # type: ignore[index]
        name = str(record["class_name"])
        matched = bool(len(boxes)) and bool((iou(box, boxes) >= match_iou).any())

        if policy == "selected_box_only":
            if matched and name in declared:
                role, free, reason = "supervised", False, "selected"
            elif matched:
                role, free, reason = "banked", False, "selected, class not declared"
            else:
                # Absent from the alias, therefore a background target. Named
                # `ignored` would be a lie: `selected_box_only` has no ignore.
                role, free, reason = "ignored", True, "not selected -> background"
        elif policy == "full_image":
            if name in declared:
                role, free = "supervised", not matched
                reason = "on an opened image, class declared"
            else:
                role, free, reason = "banked", not matched, "class not declared yet"
        else:  # known_plus_selected_ignore_rest
            if name in previously_known:
                role, free, reason = "supervised", True, "already known, free"
            elif matched and name in declared:
                role, free, reason = "supervised", False, "selected, class declared"
            elif matched:
                role, free, reason = "banked", False, "selected, class not declared"
            else:
                role, free, reason = "ignored", True, "unselected unknown -> ignore"

        roles.append(ObjectRole(
            image_id=str(image_id), class_name=name,
            ordinal=int(record["ordinal"]), box=box,  # type: ignore[arg-type]
            role=role, matched=matched, free=free, reason=reason,
        ))
    return roles


def plan(
    data_root: Path | str,
    selected: Mapping[str, np.ndarray],
    *,
    policy: str,
    previously_known: Sequence[str],
    declared: Sequence[str],
    ignore_mechanism: str | None = None,
    annotations: Path | str | None = None,
    match_iou: float = MATCH_IOU,
) -> Supervision:
    """Plan the supervision for every opened image. Reads XMLs, writes nothing.

    ``selected`` maps an opened image id to the proposals bought on it, as
    **normalised** ``cxcywh`` — the format ``owl.proposals`` stores. They are
    converted with each image's own ``<size>``.

    ``ignore_mechanism=None`` takes the policy's own default
    (:func:`default_ignore_mechanism`): the two policies that have no ignore set
    get ``drop``, and the one that does gets ``pixel_suppression``. Naming a
    mechanism a policy never performs is refused rather than ignored, because
    the mechanism is reported next to the numbers.
    """

    if policy not in POLICIES:
        raise SupervisionError(f"Unknown policy {policy!r}; expected one of {POLICIES}")
    if ignore_mechanism is None:
        ignore_mechanism = default_ignore_mechanism(policy)
    if ignore_mechanism not in IGNORE_MECHANISMS:
        raise SupervisionError(
            f"Unknown ignore mechanism {ignore_mechanism!r}; expected one of "
            f"{IGNORE_MECHANISMS}")
    if policy != "known_plus_selected_ignore_rest" and ignore_mechanism != "drop":
        raise SupervisionError(
            f"policy={policy!r} has no ignore set, so ignore_mechanism="
            f"{ignore_mechanism!r} would describe a behaviour it never performs. "
            "Pass ignore_mechanism='drop'.")

    data_root = Path(data_root)
    annotations = Path(annotations or data_root / "Annotations")
    known = frozenset(str(v) for v in previously_known)
    declared_set = frozenset(str(v) for v in declared)

    roles: list[ObjectRole] = []
    unmatched_selections = 0
    suppressed_area = 0.0
    total_area = 0.0
    for image_id in sorted(dict.fromkeys(str(v) for v in selected)):
        path = annotations / f"{image_id}.xml"
        if not path.exists():
            raise SupervisionError(
                f"{path} is missing, so the supervision for opened image "
                f"{image_id} cannot be written. The data root's annotations have "
                "to be on disk before a policy is materialised.")
        (width, height), records = read_annotation(path)
        boxes = cxcywh_to_xyxy(selected[image_id], width=width, height=height)
        image_roles = classify_image(
            image_id, records, boxes, policy=policy,
            previously_known=known, declared=declared_set, match_iou=match_iou,
        )
        roles.extend(image_roles)

        annotated = np.asarray([r.box for r in image_roles], dtype=np.float64)
        for proposal in boxes:
            if not len(annotated) or not (
                iou(proposal, annotated) >= match_iou
            ).any():
                unmatched_selections += 1
        total_area += float(width) * float(height)
        if ignore_mechanism == "pixel_suppression":
            for role in image_roles:
                if role.role == "ignored":
                    suppressed_area += max(role.box[2] - role.box[0], 0.0) * max(
                        role.box[3] - role.box[1], 0.0)

    return Supervision(
        policy=policy, ignore_mechanism=ignore_mechanism, roles=tuple(roles),
        diagnostics={
            "selected_boxes": int(sum(len(np.asarray(v).reshape(-1, 4))
                                      for v in selected.values())),
            "selections_matching_no_object": unmatched_selections,
            "pixels_suppressed_share": round(
                suppressed_area / max(total_area, 1.0), 6),
            "match_iou": match_iou,
        },
    )


# ----------------------------------------------------------------- the writer ---


def alias_id(source_id: str) -> str:
    """``'8' + source[1:]`` — injective on ids that do not start with eight."""

    source_id = str(source_id)
    if source_id.startswith(ALIAS_PREFIX):
        raise SupervisionError(
            f"image id {source_id!r} already starts with {ALIAS_PREFIX!r}, so the "
            "supervision alias would collide with a real image")
    if not source_id.isdigit():
        raise SupervisionError(
            f"image id {source_id!r} is not digits only; PROB's "
            "convert_image_id would raise on the alias")
    return ALIAS_PREFIX + source_id[1:]


def write_supervision(
    supervision: Supervision,
    *,
    data_root: Path | str,
    source_annotations: Path | str | None = None,
    source_images: Path | str | None = None,
    clear: bool = True,
) -> Supervision:
    """Materialise the plan as one alias annotation per opened image.

    The alias holds the ``supervised`` objects and nothing else. The original
    annotation is read and never written, because the same image is also
    evaluation data. The alias JPEG is a hard link to the original unless a
    region has to be suppressed, in which case it is a new file.

    Returns the same plan with :attr:`Supervision.aliases` filled in, so the
    caller can put the alias ids in the training split.
    """

    data_root = Path(data_root)
    source_annotations = Path(source_annotations or data_root / "Annotations")
    source_images = Path(source_images or data_root / "JPEGImages")
    out_annotations = data_root / "Annotations"
    out_images = data_root / "JPEGImages"
    out_annotations.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    if supervision.ignore_mechanism == "prob_ignore_flag":
        raise NotImplementedError(
            "ignore_mechanism='prob_ignore_flag' needs a change to the PROB "
            "fork, not to this repository. The patch is: (1) read <difficult> "
            "in OWDetection.load_instances and carry it as an `ignore` mask "
            "alongside `labels`; (2) in SetCriterion.loss_labels, zero the "
            "no-object weight for queries whose predicted box reaches IoU 0.5 "
            "with any ignore box; (3) keep those boxes out of the Hungarian "
            "cost matrix so they are never positive targets either. Until that "
            "lands, use 'pixel_suppression' — which delivers the same "
            "requirement by removing the evidence instead of masking the loss — "
            "or 'drop', which is the honest name for what the pipeline does "
            "today (the region becomes background).")

    if clear:
        for stale in out_annotations.glob(f"{ALIAS_PREFIX}*.xml"):
            stale.unlink()
        for stale in out_images.glob(f"{ALIAS_PREFIX}*.jpg"):
            stale.unlink()

    by_image: dict[str, list[ObjectRole]] = defaultdict(list)
    for role in supervision.roles:
        by_image[role.image_id].append(role)

    mapping: dict[str, str] = {}
    written_empty = 0
    suppressed_images = 0
    for source_id in sorted(by_image):
        keep = [r for r in by_image[source_id] if r.role == "supervised"]
        if not keep:
            # PROB's Normalize transform fails on a zero-box target, and an
            # image with no supervision teaches nothing anyway. It is counted,
            # not written: `images_no_supervision` is already a reported column.
            written_empty += 1
            continue

        source_xml = source_annotations / f"{source_id}.xml"
        tree = ElementTree.parse(source_xml)
        root = tree.getroot()
        alias = alias_id(source_id)

        wanted = {(r.class_name, r.ordinal) for r in keep}
        seen: dict[str, int] = defaultdict(int)
        elements: list[ElementTree.Element] = []
        for element in root.findall("object"):
            name = canonical_class_name(element.findtext("name", ""))
            ordinal = seen[name]
            seen[name] += 1
            if (name, ordinal) in wanted:
                elements.append(element)
        if len(elements) != len(wanted):
            raise SupervisionError(
                f"image {source_id} was planned with {len(wanted)} supervised "
                f"objects but {len(elements)} matched {source_xml}. The plan and "
                "the annotation on disk disagree.")

        alias_root = ElementTree.Element("annotation")
        ElementTree.SubElement(alias_root, "filename").text = f"{alias}.jpg"
        size = root.find("size")
        if size is not None:
            alias_root.append(size)
        for element in elements:
            alias_root.append(element)
        ElementTree.ElementTree(alias_root).write(
            out_annotations / f"{alias}.xml", encoding="utf-8", xml_declaration=True
        )

        ignored = [r for r in by_image[source_id] if r.role == "ignored"]
        target_jpeg = out_images / f"{alias}.jpg"
        source_jpeg = source_images / f"{source_id}.jpg"
        if supervision.ignore_mechanism == "pixel_suppression" and ignored:
            _suppress(source_jpeg, target_jpeg, [r.box for r in ignored])
            suppressed_images += 1
        else:
            _link_or_copy(source_jpeg, target_jpeg)
        mapping[alias] = source_id

    return Supervision(
        policy=supervision.policy,
        ignore_mechanism=supervision.ignore_mechanism,
        roles=supervision.roles,
        aliases=mapping,
        diagnostics=dict(supervision.diagnostics) | {
            "alias_images_written": len(mapping),
            "images_without_supervision": written_empty,
            "images_pixel_suppressed": suppressed_images,
        },
    )


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    if not source.exists():
        raise SupervisionError(f"{source} is missing; the alias has no pixels")
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copyfile(source, target)


def _suppress(source: Path, target: Path, boxes: Sequence[Sequence[float]]) -> None:
    """Write ``source`` to ``target`` with every box filled with the mean colour."""

    from PIL import Image, ImageDraw

    if not source.exists():
        raise SupervisionError(f"{source} is missing; the alias has no pixels")
    with Image.open(source) as handle:
        image = handle.convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x0, y0, x1, y1 = (float(v) for v in box)
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1, y1], fill=SUPPRESSION_FILL)
    if target.exists():
        target.unlink()
    image.save(target, format="JPEG", quality=95)


# ------------------------------------------------- PROB's own target, in code ---


def prob_training_target(
    annotation: Path | str,
    *,
    split_marker: str,
    class_order: Sequence[str],
    n_prev: int,
    n_current: int,
) -> dict[str, object]:
    """PROB's training target for one annotation, reimplemented for testing.

    A faithful transcription of ``OWDetection.load_instances`` and
    ``OWDetection.__getitem__`` at the pinned commit ``4c66be1``, restricted to
    the fields the loss reads. It exists so a test can assert what the detector
    would be taught **without** a GPU, a checkout or a dataset — and so that the
    assertion is against PROB's rule rather than against a description of it.

    ``labels`` and ``boxes`` are the positive targets. Everything else in the
    image is, under Deformable-DETR set prediction, background. That is the whole
    point of the test: ``background_would_include`` names the annotated objects
    that fall into it.
    """

    order = [canonical_class_name(name) for name in class_order]
    known = set(order[: int(n_prev) + int(n_current)])
    if "train" in split_marker:
        # remove_prev_class_and_unk_instances: only the current task's classes
        keep = set(order[int(n_prev): int(n_prev) + int(n_current)])
    elif "test" in split_marker:
        # label_known_class_and_unknown: nothing is dropped, the rest is
        # relabelled. `keep is None` is what encodes that below.
        keep = None
    elif "ft" in split_marker:
        keep = known
    else:
        raise SupervisionError(
            f"split marker {split_marker!r} matches none of PROB's branches; "
            "make_coco_transforms looks for train / ft / val / test and "
            "OWDetection.__getitem__ for train / test / ft.")

    _, records = read_annotation(Path(annotation))
    labels: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    dropped: list[dict] = []
    for record in records:
        name = str(record["class_name"])
        box = tuple(float(v) for v in record["box"])  # type: ignore[arg-type]
        if keep is None:
            labels.append(name if name in known else "unknown")
            boxes.append(box)
        elif name in keep:
            labels.append(name)
            boxes.append(box)
        else:
            dropped.append({"class_name": name, "box": box})
    return {
        "split_marker": split_marker,
        "labels": labels,
        "boxes": boxes,
        "iscrowd": [0] * len(labels),
        # Objects present in *this* annotation that PROB's own filter removes.
        # Their regions become background; nothing else in the annotation does.
        "dropped_by_prob": dropped,
    }


def false_background(
    original_annotation: Path | str,
    target: Mapping[str, object],
    *,
    suppressed_boxes: Sequence[Sequence[float]] = (),
    match_iou: float = MATCH_IOU,
) -> list[dict]:
    """Real annotated objects whose region the detector is taught as background.

    This is the number the whole policy question is about, and it cannot be read
    off the alias alone: the alias is what PROB is *handed*, and a box left out
    of it is invisible there while still being a real object in the pixels. So
    the original annotation is the ground truth, ``target`` is what PROB will
    actually make positive (from :func:`prob_training_target` on the alias), and
    ``suppressed_boxes`` are the regions whose pixels no longer hold an object.

    An object counts as falsely taught as background when it is
    **not** a positive target and **not** suppressed. Under
    ``known_plus_selected_ignore_rest`` with ``pixel_suppression`` this list must
    be empty; under ``drop`` it is exactly the ignore set, which is the
    difference the mechanism makes.
    """

    _, records = read_annotation(Path(original_annotation))
    positives = np.asarray(
        list(target.get("boxes") or []), dtype=np.float64
    ).reshape(-1, 4)
    suppressed = np.asarray(list(suppressed_boxes), dtype=np.float64).reshape(-1, 4)

    offenders: list[dict] = []
    for record in records:
        box = tuple(float(v) for v in record["box"])  # type: ignore[arg-type]
        if len(positives) and (iou(box, positives) >= match_iou).any():
            continue
        if len(suppressed) and (iou(box, suppressed) >= match_iou).any():
            continue
        offenders.append({"class_name": str(record["class_name"]),
                          "ordinal": int(record["ordinal"]), "box": box})
    return offenders
