"""A shared, reduced evaluation split, so a ten-task chain is affordable.

PROB's official evaluator over the full 4,952-image test set costs about 32
minutes per checkpoint. A ten-task chain evaluated after every task would spend
five hours on evaluation alone, per arm. This builds a smaller split that keeps
**every** image containing a declared class — so the new-class number is exact —
and samples a deterministic remainder for the previous-class number.

Previous-class mAP on a reduced split is a sample estimate and is **not**
comparable to published full-test figures. It is comparable across arms here,
because every arm is scored on the identical split.
"""

from __future__ import annotations

import random
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

#: PROB's annotation files use COCO names where the VOC benchmark uses its own.
COCOFIED_TO_VOC = {
    "airplane": "aeroplane",
    "dining table": "diningtable",
    "motorcycle": "motorbike",
    "potted plant": "pottedplant",
    "couch": "sofa",
    "tv": "tvmonitor",
}


def canonical_class_name(name: str) -> str:
    """Map an annotation's class name onto the benchmark's own spelling."""

    cleaned = str(name).strip()
    return COCOFIED_TO_VOC.get(cleaned, cleaned)


class EvaluationSubsetError(ValueError):
    """Raised when a shared evaluation subset cannot be constructed."""


@dataclass(frozen=True)
class EvaluationSubset:
    """Image IDs and declared-class coverage in one shared evaluation split."""

    image_ids: tuple[str, ...]
    required_ids: tuple[str, ...]
    sampled_ids: tuple[str, ...]
    object_counts: Mapping[str, int]


def build(
    annotations: Mapping[str, Sequence[str]],
    classes: Sequence[str],
    *,
    seed: int,
    remainder_multiplier: int = 1,
    max_per_class: int | None = None,
) -> EvaluationSubset:
    """Keep the declared-class images and sample a deterministic remainder.

    ``max_per_class`` caps how many images are kept for each declared class.
    Without it a common class such as ``chair`` contributes 1,791 test images
    and drives the whole evaluation cost, while ``parking meter`` contributes
    60. Capping equalises the cost across the chain and is what makes evaluating
    after every one of ten tasks affordable. Rare classes are below any sensible
    cap and are therefore kept whole.
    """

    declared = tuple(dict.fromkeys(canonical_class_name(name) for name in classes))
    if not declared:
        raise EvaluationSubsetError("At least one declared class is required.")
    if remainder_multiplier < 0:
        raise EvaluationSubsetError("remainder_multiplier must be non-negative.")
    wanted = set(declared)
    by_class: dict[str, list[str]] = {name: [] for name in declared}
    for image_id, names in annotations.items():
        present = wanted.intersection(canonical_class_name(name) for name in names)
        for name in present:
            by_class[name].append(str(image_id))
    if max_per_class is not None:
        chooser = random.Random(seed)
        by_class = {
            name: (sorted(chooser.sample(ids, max_per_class)) if len(ids) > max_per_class else ids)
            for name, ids in by_class.items()
        }
    required = sorted({image_id for ids in by_class.values() for image_id in ids})
    if not required:
        raise EvaluationSubsetError("No test image contains a declared class.")
    remaining = sorted({str(value) for value in annotations} - set(required))
    sample_count = min(len(remaining), len(required) * remainder_multiplier)
    sampled = sorted(random.Random(seed).sample(remaining, sample_count))
    image_ids = tuple(sorted([*required, *sampled]))
    selected = set(image_ids)
    counts = {
        name: sum(
            sum(canonical_class_name(value) == name for value in names)
            for image_id, names in annotations.items()
            if str(image_id) in selected
        )
        for name in declared
    }
    return EvaluationSubset(
        image_ids=image_ids,
        required_ids=tuple(required),
        sampled_ids=tuple(sampled),
        object_counts=counts,
    )


def from_archive(
    path: str | Path,
    classes: Sequence[str],
    *,
    seed: int,
    remainder_multiplier: int = 1,
    max_per_class: int | None = None,
) -> EvaluationSubset:
    """Build a subset directly from a tarred VOC annotation directory."""

    annotations: dict[str, tuple[str, ...]] = {}
    with tarfile.open(Path(path)) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            root = ElementTree.fromstring(handle.read())
            annotations[Path(member.name).stem] = tuple(
                element.findtext("name", default="") for element in root.findall("object")
            )
    return build(
        annotations,
        classes,
        seed=seed,
        remainder_multiplier=remainder_multiplier,
        max_per_class=max_per_class,
    )


def from_directory(
    path: str | Path,
    classes: Sequence[str],
    *,
    seed: int,
    remainder_multiplier: int = 1,
) -> EvaluationSubset:
    """Build a subset from extracted VOC XML annotations."""

    annotations = {}
    for source in sorted(Path(path).glob("*.xml")):
        root = ElementTree.parse(source).getroot()
        annotations[source.stem] = tuple(
            element.findtext("name", default="") for element in root.findall("object")
        )
    return build(
        annotations,
        classes,
        seed=seed,
        remainder_multiplier=remainder_multiplier,
    )


def write_image_set(path: str | Path, subset: EvaluationSubset) -> Path:
    """Write one PROB-compatible ImageSet file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(subset.image_ids) + "\n", encoding="utf-8")
    return target
