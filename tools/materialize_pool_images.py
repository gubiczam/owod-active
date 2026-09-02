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

    python tools/materialize_pool_images.py --data-root /content/data/OWOD
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
    parser.add_argument("--workers", type=int, default=32)
    arguments = parser.parse_args()

    root = Path(arguments.data_root)
    jpeg = root / "JPEGImages"
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

    missing = [name for name in image_ids if not valid_jpeg(jpeg / f"{name}.jpg")]
    print(f"{len(image_ids) - len(missing)} already present; fetching {len(missing)}")
    failures: list[tuple[str, str]] = []
    if missing:
        with ThreadPoolExecutor(max_workers=min(arguments.workers, len(missing))) as pool_:
            for image_id, error in pool_.map(lambda name: fetch(name, jpeg), missing):
                if error:
                    failures.append((image_id, error))
    if failures:
        detail = "\n  ".join(f"{name}: {error}" for name, error in failures[:10])
        raise SystemExit(
            f"{len(failures)} of {len(image_ids)} pool images could not be "
            f"materialised, so the export would fail its own gate:\n  {detail}"
        )

    path = image_sets / f"{IMAGE_SET}.txt"
    path.write_text("\n".join(image_ids) + "\n", encoding="utf-8")

    from owl.evaluation_subset import check_split_name
    check_split_name(IMAGE_SET)          # refuses a name PROB would route silently
    print(f"wrote {path} ({len(image_ids)} ids), split name checked")
    print(f"ready: {arguments.data_root}")


if __name__ == "__main__":
    main()
