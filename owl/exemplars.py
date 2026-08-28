"""Object-level exemplar rehearsal: the memory is boxes, not images.

Replay Protocol V3. Contribution B of the research plan asks what the *class
composition* of a fixed rehearsal budget does to the stability-plasticity
trade-off, so the budget has to be fixed in the unit the question is about:

    sum_c m_c  ==  |E_k|  ==  delivered previous-class targets  ==  M

Versions 1 and 2 could not hold that. They stored a set of **images** chosen to
cover an object allocation, and PROB trains on whole images, so every
previous-class box that happened to share an image with a chosen exemplar was
rehearsed too. Measured on the canonical pool at ``M = 400``: ``head_favouring``
delivered 464 objects at t6 and ``tail_favouring`` delivered 1,240 — a 2.67x
spread produced by the allocation rule rather than by design. An arm cannot be
said to forget less than another when it rehearsed 2.7 times as much.

**How the budget is enforced.** PROB resolves one annotation per image id
(``OWDetection.__init__`` builds ``imgid2annotations`` from
``Annotations/<id>.xml``) and reads *every* ``<object>`` in it
(``load_instances``). There is no per-image filter and no ignore flag — a box
that survives ``remove_unknown_instances`` becomes a supervised target, and one
that does not is simply absent, which the set-prediction loss reads as
background. So the only place to drop a box without patching PROB is the
annotation itself, and the original annotations must stay untouched because the
same image can be current-task supervision.

Hence **replay aliases**: each source image contributing exemplars gets a second
image id whose XML holds only the selected boxes, and whose JPEG links to the
original. The alias goes into the training split; the original is never modified
and never read for replay.

**Why the alias id is what it is.** ``OWDetection.convert_image_id`` does
``int('2021' + image_id.replace('_', ''))``, so an alias must be digits only —
``000000021740_r`` raises ``ValueError`` inside the loader. The evaluator's
reverse conversion asserts a 12- or 6-digit remainder, so the alias keeps the
source's width. ``'9' + source[1:]`` is injective on ids that do not start with
nine, and the benchmark's largest id is 581,929, so no real image can collide.

**Object identity is (image, class, ordinal)**, not a box coordinate. The
ordinal is the position among that image's objects of that class in XML document
order, which is stable across reads. That keeps selection free of the
filesystem — it needs only per-class counts — while the alias writer, which does
have the XMLs, resolves an ordinal back to a box.
"""

from __future__ import annotations

import shutil
import zlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from owl.evaluation_subset import canonical_class_name


class ExemplarError(ValueError):
    """Raised when the exemplar budget cannot be satisfied exactly."""


@dataclass(frozen=True, order=True)
class Exemplar:
    """One stored rehearsal object.

    ``ordinal`` is the index among the source image's objects **of this class**,
    in XML document order. Two exemplars on one image differ only in ordinal, and
    both survive into the same alias annotation.
    """

    image_id: str
    class_name: str
    ordinal: int

    def as_row(self) -> list:
        return [self.image_id, self.class_name, self.ordinal]

    @classmethod
    def from_row(cls, row: Sequence) -> Exemplar:
        return cls(str(row[0]), str(row[1]), int(row[2]))


# ------------------------------------------------------------------ the pool ---


def enumerate_pool(
    per_image_classes: Mapping[str, Mapping[str, int]],
    classes: Iterable[str],
) -> tuple[Exemplar, ...]:
    """Every object of ``classes`` in ``per_image_classes``, as exemplars.

    This is the *object* granularity the budget is defined in. A pool of 89,490
    images holding 421,243 task-1 objects enumerates to 421,243 exemplars.
    """

    wanted = set(classes)
    out: list[Exemplar] = []
    for image_id in sorted(per_image_classes):
        counts = per_image_classes[image_id]
        for name in sorted(counts):
            if name not in wanted:
                continue
            for ordinal in range(int(counts[name])):
                out.append(Exemplar(str(image_id), name, ordinal))
    return tuple(out)


def capacities(candidates: Iterable[Exemplar]) -> dict[str, int]:
    """How many exemplars of each class the pool can supply.

    This is the ``n_c`` the allocator caps against, and it is read off the
    *currently eligible* pool — never off discarded history.
    """

    out: dict[str, int] = defaultdict(int)
    for exemplar in candidates:
        out[exemplar.class_name] += 1
    return dict(out)


# ------------------------------------------------------------- the selection ---


def _class_order(candidates: Sequence[Exemplar], name: str, seed: int) -> list[Exemplar]:
    """A deterministic, arm-independent order for one class's candidates.

    Seeded from the class *name* rather than from its position in the demand
    dictionary, so changing ``alpha`` changes how many exemplars a class
    contributes and never which ones are preferred. Without that, two arms would
    differ in composition *and* in which individual objects they saw.
    """

    pool = sorted(e for e in candidates if e.class_name == name)
    generator = np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])
    return [pool[index] for index in generator.permutation(len(pool))]


def select(
    candidates: Sequence[Exemplar],
    demand: Mapping[str, int],
    *,
    incumbent: Sequence[Exemplar] = (),
    reallocate: bool = False,
    seed: int = 0,
) -> tuple[Exemplar, ...]:
    """Take exactly ``demand[c]`` exemplars of every class ``c``.

    ``reallocate=False`` keeps the exemplars already held wherever the new quota
    still has room for them, and evicts only the surplus — the incumbent memory
    persists and is topped up. ``reallocate=True`` re-derives the set from the
    same bounded pool. Both return exactly ``sum(demand.values())`` exemplars,
    so the two policies differ in *which* objects are rehearsed and never in how
    many.
    """

    available = set(candidates)
    held = [e for e in incumbent if e in available]
    chosen: list[Exemplar] = []

    for name in sorted(demand):
        quota = int(demand[name])
        if quota <= 0:
            continue
        order = _class_order(candidates, name, seed)
        if len(order) < quota:
            raise ExemplarError(
                f"class {name!r} was allocated {quota} exemplars but the eligible "
                f"pool holds {len(order)}. The allocator caps demand at capacity, "
                "so this means the capacities it was given do not describe the "
                "pool it is being selected from."
            )
        picked: list[Exemplar] = []
        if not reallocate:
            picked.extend([e for e in held if e.class_name == name][:quota])
        taken = set(picked)
        for exemplar in order:
            if len(picked) >= quota:
                break
            if exemplar not in taken:
                picked.append(exemplar)
                taken.add(exemplar)
        chosen.extend(picked)

    total = sum(int(v) for v in demand.values() if int(v) > 0)
    if len(chosen) != total:
        raise ExemplarError(
            f"selected {len(chosen)} exemplars for a demand of {total}; the "
            "budget must be met exactly."
        )
    return tuple(chosen)


def delivered_per_class(exemplars: Iterable[Exemplar]) -> dict[str, int]:
    """What PROB will actually receive, per class. Compare against the demand."""

    out: dict[str, int] = defaultdict(int)
    for exemplar in exemplars:
        out[exemplar.class_name] += 1
    return dict(out)


# ------------------------------------------------------------- the aliasing ---


def alias_id(source_id: str) -> str:
    """The replay id for a source image. ``000000021740 -> 900000021740``.

    Digits only, because ``convert_image_id`` casts to ``int``; same width,
    because the evaluator's reverse conversion asserts 12 or 6 digits. A source
    id that already starts with nine would not be injective, so it is refused
    rather than silently colliding.
    """

    source_id = str(source_id)
    if not source_id.isdigit():
        raise ExemplarError(
            f"image id {source_id!r} is not numeric, and PROB's loader casts "
            "image ids to int, so no alias can be formed for it."
        )
    if not source_id.startswith("0"):
        raise ExemplarError(
            f"image id {source_id!r} does not start with zero. The benchmark's "
            "ids are zero-padded to twelve digits (the largest is 581,929), and "
            "swapping that leading zero for a nine is what makes the alias "
            "reversible and collision-free. An id outside that shape would need "
            "a different scheme."
        )
    return "9" + source_id[1:]


def source_id(alias: str) -> str:
    """The inverse of :func:`alias_id`. ``900000021740 -> 000000021740``."""

    alias = str(alias)
    if not alias.startswith("9") or not alias.isdigit():
        raise ExemplarError(f"{alias!r} is not a replay alias.")
    return "0" + alias[1:]


def write_aliases(
    exemplars: Sequence[Exemplar],
    *,
    data_root: Path,
    source_annotations: Path | None = None,
    source_images: Path | None = None,
    clear: bool = True,
) -> dict[str, str]:
    """Materialise the exemplars as alias annotations. Returns ``alias -> source``.

    One alias per source image, holding **only** the selected boxes of that
    image. Everything else on it — other previous-class objects that were not
    selected, the class the current task is introducing, and classes belonging to
    later tasks — is absent from the alias, so none of it can reach the loss.

    The originals are read and never written. The alias JPEG is a hard link when
    the filesystem allows one and a copy otherwise, so the pixels are not
    duplicated on Drive.

    ``clear`` removes the aliases a previous memory left behind before writing
    this one. Without it the directory accumulates generations: a source image
    held by two consecutive memories keeps whichever annotation was written last,
    so reading a finished task's aliases afterwards shows a *later* task's boxes.
    Nothing in the chain does that — PROB reads the split at training time — but
    a mutable shared directory whose contents depend on when you look is not
    something an experiment should rest on, and clearing makes the object budget
    verifiable at any moment rather than only in the instant it is used.
    """

    data_root = Path(data_root)
    annotations = Path(source_annotations or data_root / "Annotations")
    images = Path(source_images or data_root / "JPEGImages")
    out_annotations = data_root / "Annotations"
    out_images = data_root / "JPEGImages"
    out_annotations.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    if clear:
        for stale in out_annotations.glob("9*.xml"):
            stale.unlink()

    wanted: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for exemplar in exemplars:
        wanted[exemplar.image_id].add((exemplar.class_name, exemplar.ordinal))

    mapping: dict[str, str] = {}
    for source_id in sorted(wanted):
        source_xml = annotations / f"{source_id}.xml"
        if not source_xml.exists():
            raise ExemplarError(
                f"{source_xml} is missing, so the exemplars on image "
                f"{source_id} cannot be filtered. The replay pool's annotations "
                "have to be on disk before the memory is materialised."
            )
        tree = ElementTree.parse(source_xml)
        root = tree.getroot()

        alias = alias_id(source_id)
        keep: list[ElementTree.Element] = []
        seen: dict[str, int] = defaultdict(int)
        for element in root.findall("object"):
            name = canonical_class_name(element.findtext("name", ""))
            ordinal = seen[name]
            seen[name] += 1
            if (name, ordinal) in wanted[source_id]:
                keep.append(element)

        if len(keep) != len(wanted[source_id]):
            raise ExemplarError(
                f"image {source_id} was asked for {len(wanted[source_id])} "
                f"exemplars but only {len(keep)} matched its annotation. The "
                "per-class counts the memory was allocated from disagree with "
                f"{source_xml}."
            )
        if not keep:
            raise ExemplarError(
                f"image {source_id} would produce an alias with no boxes, and "
                "PROB's Normalize transform fails on a zero-box target."
            )

        alias_root = ElementTree.Element("annotation")
        for tag in ("folder", "size", "segmented"):
            element = root.find(tag)
            if element is not None:
                alias_root.append(element)
        filename = ElementTree.SubElement(alias_root, "filename")
        filename.text = f"{alias}.jpg"
        for element in keep:
            alias_root.append(element)

        ElementTree.ElementTree(alias_root).write(
            out_annotations / f"{alias}.xml", encoding="utf-8", xml_declaration=False
        )
        _link(images / f"{source_id}.jpg", out_images / f"{alias}.jpg")
        mapping[alias] = source_id

    return mapping


def _link(source: Path, target: Path) -> None:
    """Point ``target`` at ``source`` without duplicating the pixels if possible."""

    if target.exists():
        return
    if not source.exists():
        raise ExemplarError(
            f"{source} is missing, so no alias image can be provided for it. "
            "The replay pool's JPEGs have to be fetched before the memory is "
            "materialised."
        )
    for attempt in (target.hardlink_to, target.symlink_to):
        try:
            attempt(source)
            return
        except (OSError, NotImplementedError, AttributeError):
            continue
    shutil.copy2(source, target)
