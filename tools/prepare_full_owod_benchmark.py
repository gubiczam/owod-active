#!/usr/bin/env python
"""Build everything Benchmark V1 reads off disk, from a bare Colab runtime.

``Annotations/``      benchmark XML for every candidate image, every replay
                      source image and every test image, from the committed
                      archives. Nothing is downloaded for these.
``ImageSets/OWDETR/`` the one shared evaluation split, built from the chain's
                      three declared classes and written once, so the anchor and
                      every task of every arm are scored on identical images.
``JPEGImages/``       the test split's pixels, fetched from COCO, resumable.

Candidate and replay images are **not** fetched here. Which 1,200 images a task
scores depends on ``(seed, task)`` and which 400 exemplar objects it rehearses on
depends on what it just bought, so the runner fetches those per task through the
same materialiser. What is fetched here is what follows from the protocol alone.

    python tools/prepare_full_owod_benchmark.py --data-root /content/data/OWOD
    python tools/prepare_full_owod_benchmark.py --data-root /content/data/OWOD --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import evaluation_subset
from owl.active_selection import benchmark as bm
from tools.materialize_pool_images import extract, materialise, report, valid_jpeg

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
REPLAY_INDEX = ROOT / "data" / "reference" / "t1_replay_class_counts.json"
ARCHIVES = (
    ROOT / "data" / "staging" / "owdetr_pool_annotations.tar.gz",
    ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz",
    ROOT / "data" / "staging" / "owdetr_replay_annotations.tar.gz",
)
TEST_ARCHIVE = ARCHIVES[1]


def shared_test_split(data_root: Path, *, write: bool = True):
    """Write the one shared evaluation split; return its name and its images."""

    subset = evaluation_subset.from_archive(
        TEST_ARCHIVE, bm.declared_classes(), seed=bm.DEVELOPMENT_SEED,
        remainder_multiplier=bm.EVAL_REMAINDER_RATIO,
        max_per_class=bm.EVAL_MAX_PER_CLASS,
    )
    name = evaluation_subset.SHARED_TEST_SET
    if write:
        target = data_root / "ImageSets" / "OWDETR" / f"{name}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        evaluation_subset.write_image_set(target, subset)
    return name, subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--annotations-only", action="store_true",
                        help="extract the archives and write the split, fetch "
                             "no pixels; what the dry run needs")
    parser.add_argument("--verify-only", action="store_true",
                        help="report what is present and change nothing")
    arguments = parser.parse_args()

    data_root = Path(arguments.data_root)
    jpeg = data_root / "JPEGImages"
    name, subset = shared_test_split(data_root, write=not arguments.verify_only)
    test_ids = sorted(subset.image_ids)

    if arguments.verify_only:
        present = sum(1 for i in test_ids if valid_jpeg(jpeg / f"{i}.jpg"))
        annotations = data_root / "Annotations"
        print(f"[verify] annotations dir: {annotations.is_dir()}")
        print(f"[verify] split {name}: "
              f"{(data_root / 'ImageSets' / 'OWDETR' / f'{name}.txt').is_file()}")
        print(f"[verify] test images: {present:,}/{len(test_ids):,} readable")
        raise SystemExit(0 if present == len(test_ids) else 1)

    for archive in ARCHIVES:
        if not archive.is_file():
            raise SystemExit(f"missing committed archive: {archive}")
        extract(archive, data_root)
    print(f"[prepare] extracted {len(ARCHIVES)} committed annotation archives "
          f"into {data_root}")

    candidate_index = json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    replay_index = json.loads(REPLAY_INDEX.read_text(encoding="utf-8"))
    print(f"[prepare] candidate index {len(candidate_index):,} images; "
          f"replay index {len(replay_index):,} images")
    print(f"[prepare] shared evaluation split {name}: {len(test_ids):,} images "
          f"over {len(bm.declared_classes())} declared classes "
          f"{list(bm.declared_classes())}")

    if arguments.annotations_only:
        print("[prepare] --annotations-only: no pixels fetched")
        return

    counts = materialise(test_ids, jpeg, workers=arguments.workers)
    report(counts, jpeg)
    if counts["unreadable"]:
        raise SystemExit(
            f"{len(counts['unreadable'])} test images could not be fetched. The "
            "evaluation split is shared and frozen, so a missing test image "
            "changes what every arm is scored on. Re-run; the fetch is resumable."
        )


if __name__ == "__main__":
    main()
