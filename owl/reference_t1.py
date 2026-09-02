"""REF-T1: the canonical Task-1 labelled reference set, and how it is selected.

Method V2 Stage 2's novelty term is *novelty relative to already-labelled
knowledge*. That makes the reference set part of the method's meaning, not an
implementation detail.

**What was corrected, and why.** The first implementation used REF-A --
``predicted_known(candidate pool) & NMS`` -- which is oracle-free but is **not the
labelled set**. It estimates a pseudo-known manifold *from the same unlabelled
population it then judges*, so ``D`` would have measured "novelty relative to a
manifold inferred from the current pool", a different quantity from the one the
research plan defines. REF-A survives here only as an explicitly secondary
**pseudo-reference** diagnostic and cannot decide ``D_GO``.

REF-T1 is the real thing: ground-truth boxes of the **19 Task-1 classes** from the
canonical T1 *training* annotations. GT is legitimate here precisely because these
are already-labelled training examples at round 0 -- not candidate oracle
information, and not a future task's labels. No T2 or later annotation is read.

**Why the reference must be class-balanced.** The canonical set is 421,243 objects
with a **202.8:1** imbalance (``person`` 262,465 against ``bear`` 1,294). ``D`` is
a *maximum* over the reference set, so a frequency-proportional reference would be
62% ``person`` and ``D`` would collapse into "distance to the nearest person"
rather than "distance from known classes". Balance is a correctness requirement,
not a refinement.

Identity is deterministic: an object is ``(image_id, object_index)`` with
``object_index`` its position in its own annotation file, so the same XML always
yields the same identities and the same selection under a fixed seed.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from owl.decoder_layers import ExportError
from owl.protocol import TASK1

ROOT = Path(__file__).resolve().parent.parent

#: The canonical T1 labelled sources, both committed to this repository.
T1_ANNOTATION_ARCHIVE = ROOT / "data" / "staging" / "owdetr_replay_annotations.tar.gz"
T1_CLASS_COUNTS = ROOT / "data" / "reference" / "t1_replay_class_counts.json"

#: Measured from those files; asserted so a changed input cannot pass silently.
EXPECTED_T1_IMAGES = 89_490
EXPECTED_T1_OBJECTS = 421_243

#: Bump when the selection's meaning changes.
REFERENCE_VERSION = "ref_t1_balanced_v1"

#: Fixed before any selection was made. Not chosen by looking at an endpoint.
SELECTION_SEED = 0

#: The annotation XMLs carry **COCO** class names where this project, following
#: PROB's ``VOC_COCO_CLASS_NAMES``, uses the VOC spelling. Exactly two of the 19
#: Task-1 classes differ, and the committed count file already encodes the
#: resolved convention -- which is how the mismatch was caught: parsing without
#: this map yielded 407,383 objects against the expected 421,243, with
#: ``aeroplane`` and ``motorbike`` at **zero**.
#:
#: Left unmapped, REF-T1 would have contained no example of two of the nineteen
#: known classes, and ``D`` would have reported every aeroplane and motorbike in
#: the candidate pool as novel.
COCO_TO_VOC = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
}


@dataclass(frozen=True)
class ReferenceObjects:
    """A deterministic, class-balanced selection of T1 labelled GT objects."""

    image_ids: np.ndarray      # (N,) str
    object_index: np.ndarray   # (N,) int64, position within its annotation file
    class_name: np.ndarray     # (N,) str, one of the 19 T1 classes
    boxes: np.ndarray          # (N, 4) float32, normalised cxcywh
    keys: np.ndarray           # (N,) str, "image#object_index"
    per_class_cap: int
    provenance: dict

    def __len__(self) -> int:
        return int(self.keys.size)

    @property
    def images(self) -> list[str]:
        return sorted(set(self.image_ids.tolist()))

    def summary(self) -> dict:
        classes, counts = np.unique(self.class_name, return_counts=True)
        return {
            "objects": len(self),
            "images": len(self.images),
            "classes": int(classes.size),
            "per_class_cap": self.per_class_cap,
            "min_per_class": int(counts.min()) if counts.size else 0,
            "max_per_class": int(counts.max()) if counts.size else 0,
            "balanced": bool(counts.size and counts.min() == counts.max()),
        }


def reference_keys(image_ids: np.ndarray, object_index: np.ndarray) -> np.ndarray:
    """``(image_id, object_index)`` as one sortable string key."""

    images = np.asarray(image_ids, dtype=str)
    indices = np.asarray(object_index).astype(np.int64)
    return np.char.add(np.char.add(images, "#"), indices.astype(str))


def class_totals(path: str | Path = T1_CLASS_COUNTS) -> dict[str, int]:
    """Objects per T1 class, from the committed count file. No XML parsing."""

    counts = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(counts) != EXPECTED_T1_IMAGES:
        raise ExportError(
            f"{path} covers {len(counts)} images, expected {EXPECTED_T1_IMAGES}"
        )
    totals = dict.fromkeys(TASK1, 0)
    for classes in counts.values():
        for name, number in classes.items():
            if name in totals:
                totals[name] += int(number)
    if sum(totals.values()) != EXPECTED_T1_OBJECTS:
        raise ExportError(
            f"{path} totals {sum(totals.values())} T1 objects, expected "
            f"{EXPECTED_T1_OBJECTS}"
        )
    return totals


def _normalised_box(node, width: float, height: float) -> tuple[float, float, float, float]:
    box = node.find("bndbox")
    x1 = float(box.find("xmin").text)
    y1 = float(box.find("ymin").text)
    x2 = float(box.find("xmax").text)
    y2 = float(box.find("ymax").text)
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def enumerate_objects(
    archive: str | Path = T1_ANNOTATION_ARCHIVE,
    *,
    classes: tuple[str, ...] = TASK1,
) -> dict[str, list[tuple[str, int, str, tuple[float, float, float, float]]]]:
    """Every T1-class GT object in the archive, grouped by class.

    ``object_index`` is the object's position in its own annotation file, counted
    over **all** objects including non-T1 ones, so the identity does not shift if
    the class filter ever changes.

    Only the 19 T1 classes are collected. A later task's annotation in the same
    file is skipped and never read into the reference.
    """

    grouped: dict[str, list] = {name: [] for name in classes}
    # accept either spelling and store the VOC one the project uses everywhere else
    canonical = {name: name for name in classes}
    canonical.update({
        coco: voc for coco, voc in COCO_TO_VOC.items() if voc in grouped
    })
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if not member.name.endswith(".xml"):
                continue
            stream = handle.extractfile(member)
            if stream is None:
                continue
            tree = ElementTree.parse(stream)
            root = tree.getroot()
            image_id = Path(member.name).stem
            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)
            if width <= 0 or height <= 0:
                raise ExportError(f"{image_id} declares size {width}x{height}")
            for index, node in enumerate(root.findall("object")):
                name = canonical.get(node.find("name").text)
                if name is None:
                    continue
                grouped[name].append(
                    (image_id, index, name, _normalised_box(node, width, height))
                )

    found = sum(len(items) for items in grouped.values())
    if found != EXPECTED_T1_OBJECTS:
        empty = sorted(name for name, items in grouped.items() if not items)
        raise ExportError(
            f"enumerated {found:,} T1 objects, expected {EXPECTED_T1_OBJECTS:,}"
            + (f"; these classes came back empty: {empty}" if empty else "")
            + ". A class-name convention mismatch would silently remove whole "
              "classes from the reference, so this refuses rather than proceeds."
        )
    return grouped


def select_balanced(
    grouped: dict[str, list],
    *,
    per_class_cap: int,
    seed: int = SELECTION_SEED,
) -> ReferenceObjects:
    """Take exactly ``per_class_cap`` objects per class, deterministically.

    Each class's objects are sorted by ``(image_id, object_index)`` -- a canonical
    order that does not depend on archive iteration -- then permuted with a fixed
    seed and truncated. A seeded permutation rather than the first N of the sorted
    order, because COCO ids are arbitrary but not guaranteed unbiased, and the
    permutation makes the choice explicit and reproducible.

    A class holding fewer than ``per_class_cap`` objects contributes all of them
    and the selection is reported as **not balanced**, which callers must surface
    rather than average over.
    """

    if per_class_cap < 1:
        raise ExportError("per_class_cap must be at least 1")

    generator = np.random.default_rng(seed)
    rows: list[tuple[str, int, str, tuple[float, float, float, float]]] = []
    per_class_available = {}
    for name in sorted(grouped):
        items = sorted(grouped[name], key=lambda entry: (entry[0], entry[1]))
        per_class_available[name] = len(items)
        if not items:
            continue
        order = generator.permutation(len(items))[:per_class_cap]
        rows.extend(items[int(position)] for position in sorted(order))

    if not rows:
        raise ExportError("no T1 reference objects were selected")

    image_ids = np.asarray([row[0] for row in rows], dtype=str)
    object_index = np.asarray([row[1] for row in rows], dtype=np.int64)
    class_name = np.asarray([row[2] for row in rows], dtype=str)
    boxes = np.asarray([row[3] for row in rows], dtype=np.float32)
    keys = reference_keys(image_ids, object_index)
    if np.unique(keys).size != keys.size:
        raise ExportError("duplicate (image_id, object_index) identities selected")

    return ReferenceObjects(
        image_ids=image_ids,
        object_index=object_index,
        class_name=class_name,
        boxes=boxes,
        keys=keys,
        per_class_cap=per_class_cap,
        provenance={
            "reference_version": REFERENCE_VERSION,
            "source_archive": str(T1_ANNOTATION_ARCHIVE.name),
            "classes": list(TASK1),
            "per_class_cap": per_class_cap,
            "seed": seed,
            "per_class_available": per_class_available,
            "t1_objects_total": EXPECTED_T1_OBJECTS,
            "t1_images_total": EXPECTED_T1_IMAGES,
            "note": "Task-1 labelled training GT only; no T2 or later annotation "
                    "is read, and no candidate oracle information is used.",
        },
    )


def cost_estimate(grouped: dict[str, list], caps: tuple[int, ...]) -> list[dict]:
    """Objects and *distinct images* a cap would require. Images are the real cost.

    The GPU forward pass over crops is cheap next to fetching the source JPEGs:
    the candidate export needed 1,600 images, and the full T1 set spans 89,490.
    """

    rows = []
    for cap in caps:
        selection = select_balanced(grouped, per_class_cap=cap)
        summary = selection.summary()
        rows.append({
            "per_class_cap": cap,
            "objects": summary["objects"],
            "images": summary["images"],
            "balanced": summary["balanced"],
            "min_per_class": summary["min_per_class"],
            "max_per_class": summary["max_per_class"],
        })
    return rows
