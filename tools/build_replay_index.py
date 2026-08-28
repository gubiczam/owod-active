"""Build the old-data pool the exemplar memory is drawn from.

Replay is rehearsal of knowledge the model already had. In this protocol that is
the split PROB's ``t1.pth`` was trained on — not the candidate pool, which is
what the *selector* buys from and which would make the memory a function of an
arm's own acquisitions.

    python tools/build_replay_index.py \\
        --split /Volumes/AI_SSD/datasets/owod_canonical/ImageSets/OWDETR/owdetr_t1_train.txt \\
        --annotations /Volumes/AI_SSD/datasets/owod_canonical/Annotations

It writes two things, and both ship in the repository so that Colab needs no
attached disk:

``data/reference/t1_replay_class_counts.json``
    ``image_id -> {class_name: count}`` over the task-1 classes only. This is
    what :func:`owl.replay.allocate` reads capacities from.

``data/staging/owdetr_replay_annotations.tar.gz``
    one VOC XML per image in the index. PROB's loader reads these off disk, so a
    memory image without its annotation fails inside a DataLoader worker.

``data/staging/owdetr_replay_manifest.json``
    what this run was given and what it produced, including the SHA-256 of both
    artefacts. The pool is an input to the experiment, so the thesis has to be
    able to say which pool — and a reader has to be able to rebuild it and check.

**Why it subsamples.** The canonical split holds tens of thousands of images and
a memory of a few hundred exemplars will never touch most of them, but Colab has
to carry the annotation of anything the memory *might* select. ``--max-images``
takes a seeded, class-stratified subsample: every task-1 class contributes in
proportion to a rank that keeps the rare ones whole, so the long tail survives
the cut. The subsample is fixed once and is identical for every arm, so it
cannot bias one allocation rule against another — but it *is* a subsample, and
the manifest records the seed and the counts so the thesis can say so.

**On the two spellings.** Six classes are named one way in these XML files and
another in the benchmark's class order (``airplane``/``aeroplane`` and friends).
Unlike ``tools/build_pool_annotations.py``, which copies XMLs through for PROB's
own loader to map, the *index* is read by ``owl`` and therefore uses the
benchmark spelling, via :func:`owl.evaluation_subset.canonical_class_name`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import tarfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl.evaluation_subset import canonical_class_name
from owl.protocol import TASK1

DEFAULT_INDEX = ROOT / "data" / "reference" / "t1_replay_class_counts.json"
DEFAULT_ARCHIVE = ROOT / "data" / "staging" / "owdetr_replay_annotations.tar.gz"
DEFAULT_MANIFEST = ROOT / "data" / "staging" / "owdetr_replay_manifest.json"


def _repo_relative(path: Path) -> str:
    """Repository-relative when it is inside the repository, absolute otherwise."""

    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def digest(path: Path) -> str:
    """SHA-256 of a file, so an artefact can be tied to the run that made it."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def counts_for(path: Path, keep: set[str]) -> dict[str, int]:
    """Per-class object counts for one VOC annotation, restricted to ``keep``."""

    root = ElementTree.parse(path).getroot()
    tally: Counter[str] = Counter()
    for obj in root.iter("object"):
        name = canonical_class_name(obj.findtext("name", ""))
        if name in keep:
            tally[name] += 1
    return dict(tally)


def stratify(index: dict[str, dict[str, int]], limit: int, seed: int) -> dict[str, dict[str, int]]:
    """Keep ``limit`` images, rarest class first, so the tail survives the cut.

    Walking the classes from rarest to most common and taking images round-robin
    means a class with few images contributes all of them before a class with
    many contributes its share. Cutting uniformly at random would instead remove
    the tail in proportion to how small it already is, which is the opposite of
    what an experiment about long-tailed replay needs.
    """

    if limit <= 0 or len(index) <= limit:
        return index

    by_class: dict[str, list[str]] = {name: [] for name in TASK1}
    for image, counts in index.items():
        for name in counts:
            by_class.setdefault(name, []).append(image)

    chooser = random.Random(seed)
    for images in by_class.values():
        images.sort()
        chooser.shuffle(images)

    order = sorted(by_class, key=lambda name: len(by_class[name]))
    kept: dict[str, None] = {}
    cursor = dict.fromkeys(order, 0)
    while len(kept) < limit and any(cursor[n] < len(by_class[n]) for n in order):
        for name in order:
            if len(kept) >= limit:
                break
            while cursor[name] < len(by_class[name]):
                image = by_class[name][cursor[name]]
                cursor[name] += 1
                if image not in kept:
                    kept[image] = None
                    break
    return {image: index[image] for image in kept}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, type=Path,
                        help="the canonical task-1 train split, one image id per line")
    parser.add_argument("--annotations", required=True, type=Path,
                        help="the canonical Annotations/ directory")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--max-images", type=int, default=12000,
        help="cap on the pool, class-stratified (default: %(default)s). "
             "A memory of a few hundred exemplars never needs more, and Colab "
             "has to carry an annotation for every image it might pick.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-archive", action="store_true",
                        help="write the index only, without packing the XMLs")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()

    for path in (arguments.split, arguments.annotations):
        if not path.exists():
            print(f"error: {path} does not exist. The canonical dataset is on the "
                  "external disk; attach it, or pass --split/--annotations.",
                  file=sys.stderr)
            return 1

    keep = set(TASK1)
    index: dict[str, dict[str, int]] = {}
    missing: list[str] = []
    for image_id in read_split(arguments.split):
        path = arguments.annotations / f"{image_id}.xml"
        if not path.exists():
            missing.append(image_id)
            continue
        counts = counts_for(path, keep)
        # an image with no task-1 object can serve no allocation, and PROB's
        # collate fails on a zero-box image
        if counts:
            index[image_id] = counts

    full = len(index)
    index = stratify(index, arguments.max_images, arguments.seed)

    arguments.index.parent.mkdir(parents=True, exist_ok=True)
    arguments.index.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")

    objects: Counter[str] = Counter()
    for counts in index.values():
        objects.update(counts)
    print(f"wrote {arguments.index}  ({len(index)} of {full} eligible images, "
          f"{sum(objects.values())} task-1 objects, "
          f"{arguments.index.stat().st_size / 1e6:.1f} MB)")
    print("objects per class, rarest first:")
    for name, count in sorted(objects.items(), key=lambda item: item[1]):
        print(f"    {name:<12} {count:>7}")
    if missing:
        print(f"NOTE: {len(missing)} images in the split have no annotation; skipped.")

    if arguments.no_archive:
        return 0

    # Deterministic, so the manifest's digest is a check and not just a record:
    # a plain "w:gz" tar stores each source file's mtime and gzip stores its own
    # timestamp, which made two builds of identical content differ in bytes — and
    # a 13 MB artefact that changes on every rebuild is a 13 MB diff for nothing.
    arguments.archive.parent.mkdir(parents=True, exist_ok=True)
    # gzip stores a filename in its header, and takes it from the path *or* from
    # `fileobj.name` — so the same content written to two paths differed in
    # bytes at offset 11. `filename=""` is what makes the artefact a function of
    # its content alone.
    with (
        arguments.archive.open("wb") as raw_output,
        gzip.GzipFile(
            filename="", fileobj=raw_output, mode="wb", compresslevel=9, mtime=0
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for image_id in sorted(index):
            source = arguments.annotations / f"{image_id}.xml"
            info = archive.gettarinfo(source, arcname=f"Annotations/{image_id}.xml")
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as member:
                archive.addfile(info, member)
    print(f"wrote {arguments.archive}  "
          f"({arguments.archive.stat().st_size / 1e6:.1f} MB)")

    manifest = {
        "replay_protocol_version": 3,
        "purpose": "the old-data pool the exemplar memory is drawn from at task 2",
        "split": str(arguments.split),
        "annotations": str(arguments.annotations),
        "classes": list(TASK1),
        "max_images": arguments.max_images,
        "seed": arguments.seed,
        "eligible_images": full,
        "kept_images": len(index),
        "objects": sum(objects.values()),
        "objects_per_class": dict(sorted(objects.items(), key=lambda item: item[1])),
        "images_come_from": "COCO train2017/val2017, downloaded in Colab",
        "index": _repo_relative(arguments.index),
        "index_bytes": arguments.index.stat().st_size,
        "index_sha256": digest(arguments.index),
        "archive": _repo_relative(arguments.archive),
        "archive_bytes": arguments.archive.stat().st_size,
        "archive_sha256": digest(arguments.archive),
        "command": (
            "python tools/build_replay_index.py "
            f"--split {arguments.split} --annotations {arguments.annotations} "
            f"--max-images {arguments.max_images} --seed {arguments.seed}"
        ),
    }
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {arguments.manifest}")
    print(f"  index   sha256 {manifest['index_sha256']}")
    print(f"  archive sha256 {manifest['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
