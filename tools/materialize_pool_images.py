#!/usr/bin/env python
"""Put exactly the pool's 1,600 images and annotations on disk, and nothing else.

The decoder-layer export reads images through PROB's own dataset, so PROB's
preprocessing is applied rather than re-implemented. That needs three things in
the canonical layout: the annotation XMLs, the JPEGs, and an ImageSets file.

* **Annotations** come from the committed archive ``owdetr_pool_annotations.tar.gz``
  -- the same one the replay notebook uses, so no new provenance is introduced.
* **JPEGs** come from COCO's public bucket by zero-padded id, which is how the
  benchmark's images are distributed. Only the ids the pool names are fetched.
* **The ImageSets name is chosen, not incidental.** PROB decides which annotation
  filters to run by *substring* of the split name, and ``owl.evaluation_subset``
  documents the trap: a name matching ``val`` reaches a branch where no filter
  runs at all. ``owl_layer_test`` has ``test`` as its only marker, which is the
  branch whose transform matches inference.

Fails closed: if any pool image cannot be materialised the export would fail its
own gate later, so it is better to stop here and say which ids are missing.

Two modes, sharing one download path so there is only ever one image source:

``--pool`` (default)
    the 1,600 images the frozen candidate pool names, plus its annotation
    archive and the ``ImageSets`` file the exporters read.
``--image-list FILE``
    one ``image_id`` per line. Fetches only JPEGs into the same canonical
    ``<data-root>/JPEGImages/<image_id>.jpg`` layout -- no annotation archive and
    no ``ImageSets`` file, because a caller supplying its own list already has its
    own boxes. Used for REF-T1, whose GT boxes come from
    :mod:`owl.reference_t1` rather than from an extracted archive.

    python tools/materialize_pool_images.py --data-root /content/data/OWOD
    python tools/materialize_pool_images.py --data-root /content/data/OWOD \
        --image-list /content/ref_t1_images.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "pool" / "sowodb_t1_frozen_pool.npz"
POOL_ARCHIVE = ROOT / "data" / "staging" / "owdetr_pool_annotations.tar.gz"
COCO = "https://s3.amazonaws.com/images.cocodataset.org"

#: `test` is the only marker, so PROB applies the evaluation transform. See the
#: module docstring and owl.evaluation_subset.check_split_name.
IMAGE_SET = "owl_layer_test"


def pool_image_ids(pool: Path) -> list[str]:
    payload = np.load(pool, allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == "pool"
    return sorted(set(np.asarray(payload["image_ids"], dtype=str)[keep].tolist()))


def extract(archive: Path, target: Path) -> None:
    """Extract the committed archive, refusing any member that escapes ``target``."""

    target = target.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            destination = (target / member.name).resolve()
            if destination != target and target not in destination.parents:
                raise ValueError(f"archive member escapes the target: {member.name}")
        try:
            handle.extractall(target, filter="data")
        except TypeError:                      # older Python; checked above
            handle.extractall(target)


def valid_image_id(value: str) -> bool:
    """COCO ids are zero-padded 12-digit numbers, and the URL is built from them.

    Checked rather than trusted: an id with a stray character produces a 404 that
    looks identical to a genuinely missing image, and a path separator would
    escape ``JPEGImages`` entirely.
    """

    return len(value) == 12 and value.isdigit()


def parse_image_list(path: str | Path) -> list[str]:
    """One image id per line -> a deduplicated list in first-occurrence order.

    Blank lines and surrounding whitespace are ignored. Duplicates are dropped
    deterministically, keeping the first occurrence, so the fetch order is a
    function of the file alone. Malformed ids are reported together and refused --
    fetching the valid ones and reporting the rest afterwards would leave a
    half-materialised root that a later export would then fail on.
    """

    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"image list {path} does not exist")

    seen: dict[str, None] = {}
    malformed: list[tuple[int, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value:
            continue
        if not valid_image_id(value):
            malformed.append((number, value))
            continue
        seen.setdefault(value, None)

    if malformed:
        shown = ", ".join(f"line {n}: {v!r}" for n, v in malformed[:10])
        raise SystemExit(
            f"{len(malformed)} malformed image id(s) in {path}; expected "
            f"zero-padded 12-digit COCO ids. {shown}"
        )
    if not seen:
        raise SystemExit(f"{path} holds no image ids")
    return list(seen)


def materialise(image_ids: list[str], jpeg: Path, *, workers: int) -> dict:
    """Fetch whatever is missing, validate everything, and report the counts.

    An already-valid JPEG is left untouched: the file is only removed when it
    fails validation, so re-running costs nothing and cannot destroy a good image.
    """

    jpeg.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(image_ids))
    present = [name for name in unique if valid_jpeg(jpeg / f"{name}.jpg")]
    missing = [name for name in unique if name not in set(present)]

    failures: list[tuple[str, str]] = []
    if missing:
        with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            for image_id, error in pool.map(lambda name: fetch(name, jpeg), missing):
                if error:
                    failures.append((image_id, error))

    # validate the end state rather than trusting the download return codes
    readable = [name for name in unique if valid_jpeg(jpeg / f"{name}.jpg")]
    unreadable = sorted(set(unique) - set(readable))
    return {
        "requested": len(image_ids),
        "unique": len(unique),
        "already_present": len(present),
        "downloaded": len(readable) - len(present),
        "failed": len(unreadable),
        "failures": failures,
        "unreadable": unreadable,
    }


def report(counts: dict, jpeg: Path) -> None:
    """Print the counts and fail non-zero if anything is still missing."""

    print(f"  requested        : {counts['requested']:,}")
    print(f"  unique           : {counts['unique']:,}")
    print(f"  already present  : {counts['already_present']:,}")
    print(f"  downloaded       : {counts['downloaded']:,}")
    print(f"  failed           : {counts['failed']:,}")
    if counts["failed"]:
        detail = "\n  ".join(
            f"{name}: {error}" for name, error in counts["failures"][:10]
        ) or "\n  ".join(counts["unreadable"][:10])
        raise SystemExit(
            f"{counts['failed']} of {counts['unique']} images could not be "
            f"materialised under {jpeg}, so an export would fail its own gate:\n"
            f"  {detail}"
        )
    print(f"  all {counts['unique']:,} images readable under {jpeg}")


def valid_jpeg(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        from PIL import Image
        with Image.open(path) as image:
            if image.format != "JPEG":
                return False
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def fetch(image_id: str, jpeg: Path) -> tuple[str, str | None]:
    target = jpeg / f"{image_id}.jpg"
    partial = target.with_name(target.name + ".part")
    errors = []
    for split in ("train2017", "val2017"):
        partial.unlink(missing_ok=True)
        result = subprocess.run(
            ["curl", "--fail", "--silent", "--show-error", "--location",
             "--retry", "3", "--retry-delay", "1", "--connect-timeout", "20",
             "--max-time", "180", "--output", str(partial),
             f"{COCO}/{split}/{image_id}.jpg"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and valid_jpeg(partial):
            partial.replace(target)
            return image_id, None
        lines = (result.stderr or "").strip().splitlines()
        errors.append(f"{split}: rc={result.returncode}: "
                      f"{lines[-1] if lines else 'invalid or empty JPEG'}")
    partial.unlink(missing_ok=True)
    target.unlink(missing_ok=True)
    return image_id, "; ".join(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pool", default=str(POOL))
    parser.add_argument("--image-list", default=None,
                        help="fetch the ids in this file instead of the pool's; "
                             "one id per line, JPEGs only")
    parser.add_argument("--workers", type=int, default=32)
    arguments = parser.parse_args()

    root = Path(arguments.data_root)
    jpeg = root / "JPEGImages"

    # --- image-list mode: JPEGs only, into the same canonical layout ---------
    if arguments.image_list:
        image_ids = parse_image_list(arguments.image_list)
        print(f"image list {arguments.image_list}: {len(image_ids):,} unique ids")
        report(materialise(image_ids, jpeg, workers=arguments.workers), jpeg)
        print(f"ready: {jpeg}")
        return

    image_sets = root / "ImageSets" / "OWDETR"
    for directory in (root, jpeg, image_sets):
        directory.mkdir(parents=True, exist_ok=True)

    image_ids = pool_image_ids(Path(arguments.pool))
    print(f"pool names {len(image_ids)} images")

    if not POOL_ARCHIVE.is_file():
        raise SystemExit(f"missing committed archive {POOL_ARCHIVE}")
    extract(POOL_ARCHIVE, root)
    annotations = root / "Annotations"
    if not annotations.is_dir():
        raise SystemExit(f"{annotations} absent after extracting {POOL_ARCHIVE.name}")
    print(f"annotations extracted to {annotations}")

    report(materialise(image_ids, jpeg, workers=arguments.workers), jpeg)

    path = image_sets / f"{IMAGE_SET}.txt"
    path.write_text("\n".join(image_ids) + "\n", encoding="utf-8")

    from owl.evaluation_subset import check_split_name
    check_split_name(IMAGE_SET)          # refuses a name PROB would route silently
    print(f"wrote {path} ({len(image_ids)} ids), split name checked")
    print(f"ready: {arguments.data_root}")


if __name__ == "__main__":
    main()
