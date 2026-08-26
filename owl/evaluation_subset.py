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


#: The markers PROB routes a split by, in the order ``make_coco_transforms``
#: tests them. The first match wins, so a name carrying two is not merely
#: ambiguous — it silently picks one behaviour.
SPLIT_MARKERS: tuple[str, ...] = ("train", "ft", "val", "test")

#: What ``OWDetection.__getitem__`` does for each marker. ``val`` is the trap:
#: ``make_coco_transforms`` accepts it, and then the getitem branches test
#: ``train`` / ``test`` / ``ft`` only, so a ``val`` split gets **no filtering at
#: all**.
MARKER_BEHAVIOUR = {
    "train": "remove_prev_class_and_unk_instances — keeps only the current task's classes",
    "ft": "remove_unknown_instances — keeps every class introduced so far",
    "test": "label_known_class_and_unknown — relabels every unseen class to the unknown index",
    "val": "NOTHING — no filtering is applied, so no object is ever labelled unknown",
}


#: The canonical name for this protocol's shared evaluation split.
#:
#: It lives here rather than in the notebook on purpose. The notebook's cells
#: come from whoever last saved them, while ``owl`` is re-cloned on every run —
#: so anything the two must agree on has to live in the package, or a notebook
#: saved before a rename carries the old value and fails in a way that looks
#: like a bug in the new code. That is exactly what happened to this name.
SHARED_TEST_SET = "owl_shared_test"


class SplitNameError(ValueError):
    """Raised when a split name would make PROB do the wrong thing silently."""


def check_split_name(name: str, *, purpose: str = "test") -> str:
    """Fail loudly on a split name PROB would route somewhere unintended.

    PROB picks the annotation filter by **substring** of the split's name. That
    makes an innocuous rename a silent change of meaning, and one case is a trap
    worth naming: ``eval`` contains ``val``, so a split called
    ``owl_shared_eval`` is routed to the ``val`` branch — and the branch that
    applies the filtering tests only ``train`` / ``test`` / ``ft``, so **no
    filtering runs**.

    The consequence is not an error. It is that
    :func:`label_known_class_and_unknown` never runs, so no object is ever
    relabelled to the unknown class, so **U-Recall reads zero everywhere** and
    future-task objects are scored as if their class were already known. A full
    table of plausible, wrong numbers.

    An evaluation split must therefore carry ``test`` and nothing else.
    """

    if purpose not in MARKER_BEHAVIOUR:
        raise SplitNameError(f"Unknown purpose {purpose!r}; expected one of {SPLIT_MARKERS}.")
    present = [marker for marker in SPLIT_MARKERS if marker in name]
    if present != [purpose]:
        raise SplitNameError(
            f"The split name {name!r} carries the markers {present or ['none']}, not "
            f"exactly ['{purpose}']. PROB routes a split by substring, so this one "
            f"would get: {MARKER_BEHAVIOUR.get(present[0], 'no filtering at all') if present else 'no filtering at all'}. "
            f"Rename it so that {purpose!r} is the only marker in it — "
            f"'owl_shared_{purpose}' works, 'owl_shared_eval' does not, because "
            "'eval' contains 'val'."
        )
    return name


def write_image_set(path: str | Path, subset: EvaluationSubset) -> Path:
    """Write one PROB-compatible ImageSet file.

    The file's stem is the split name PROB will be given, so it is checked here:
    see :func:`check_split_name`.
    """

    target = Path(path)
    check_split_name(target.stem, purpose="test")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(subset.image_ids) + "\n", encoding="utf-8")
    return target
