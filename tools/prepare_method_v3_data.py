#!/usr/bin/env python
"""Build everything Method V3 reads off disk, from a bare Colab runtime.

What the twelve trajectories actually need under ``--data-root``:

``Annotations/``      the benchmark XML for every candidate image, every replay
                      source image and every test image. All three come from the
                      committed archives; nothing is downloaded for them.
``ImageSets/OWDETR/`` the one shared evaluation split, written here from
                      ``owl.evaluation_subset`` so every trajectory and the
                      anchor are scored on the identical images.
``JPEGImages/``       the pixels. Fetched from COCO, resumable, never
                      re-downloading a file that already validates.

The replay **source** images are not fetched here: which 400 exemplar objects a
trajectory rehearses on depends on its seed and on what it just bought, so the
runner fetches them per trajectory through the same materialiser. What is fetched
here is everything that is a function of the protocol alone.

    python tools/prepare_method_v3_data.py --data-root /content/data/OWOD
    python tools/prepare_method_v3_data.py --data-root /content/data/OWOD --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import evaluation_subset, method_v3, protocol
from tools.materialize_pool_images import extract, materialise, report, valid_jpeg

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "pool" / "sowodb_t1_frozen_pool.npz"
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
ARCHIVES = (
    ROOT / "data" / "staging" / "owdetr_pool_annotations.tar.gz",
    ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz",
    ROOT / "data" / "staging" / "owdetr_replay_annotations.tar.gz",
)
TEST_ARCHIVE = ARCHIVES[1]


def shared_test_split(data_root: Path, *, write: bool = True) -> tuple[str, int]:
    """Write the one shared evaluation split and return its name and size."""

    task = protocol.build_chain(method_v3.N_TASKS)[1]
    subset = evaluation_subset.from_archive(
        TEST_ARCHIVE, [task.new_class], seed=0,
        remainder_multiplier=method_v3.EVAL_REMAINDER_RATIO,
        max_per_class=method_v3.EVAL_MAX_PER_CLASS,
    )
    name = evaluation_subset.SHARED_TEST_SET
    if write:
        target = data_root / "ImageSets" / "OWDETR" / f"{name}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        evaluation_subset.write_image_set(target, subset)
    return name, len(subset.image_ids)


def required_images(data_root: Path) -> dict[str, list[str]]:
    """The image ids that follow from the protocol alone."""

    candidate_index = json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    pool = method_v3.population(POOL, candidate_index)
    candidates = sorted(str(value) for value in pool.image_ids)

    task = protocol.build_chain(method_v3.N_TASKS)[1]
    subset = evaluation_subset.from_archive(
        TEST_ARCHIVE, [task.new_class], seed=0,
        remainder_multiplier=method_v3.EVAL_REMAINDER_RATIO,
        max_per_class=method_v3.EVAL_MAX_PER_CLASS,
    )
    return {"candidates": candidates, "test": sorted(subset.image_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--annotations-only", action="store_true",
                        help="extract the committed archives and write the shared "
                             "split; download no pixels. Used by the orchestration "
                             "dry run, which never reads an image")
    parser.add_argument("--verify-only", action="store_true",
                        help="check the population, the split and what is already "
                             "on disk; download nothing")
    arguments = parser.parse_args()

    data_root = Path(arguments.data_root)
    for archive in ARCHIVES:
        if not archive.is_file():
            raise SystemExit(f"missing committed archive: {archive}")

    sets = required_images(data_root)
    wanted = sorted({*sets["candidates"], *sets["test"]})
    print(f"[population] {len(sets['candidates']):,} candidate images  PASS")
    print(f"[evaluation] {len(sets['test']):,} shared test images")
    print(f"[union]      {len(wanted):,} JPEGs "
          f"(overlap {len(set(sets['candidates']) & set(sets['test'])):,})")

    if arguments.verify_only:
        jpeg = data_root / "JPEGImages"
        present = sum(1 for name in wanted if valid_jpeg(jpeg / f"{name}.jpg"))
        name, size = shared_test_split(data_root, write=False)
        print(f"[split]      {name} would hold {size:,} images")
        print(f"[jpeg]       {present:,} of {len(wanted):,} already valid on disk")
        print("VERIFY ONLY: nothing downloaded, nothing extracted.")
        return

    data_root.mkdir(parents=True, exist_ok=True)
    for archive in ARCHIVES:
        extract(archive, data_root)
    print(f"[annotations] extracted {len(ARCHIVES)} committed archives into {data_root}")

    name, size = shared_test_split(data_root)
    print(f"[split]      wrote {name} with {size:,} images")

    if arguments.annotations_only:
        print(f"ANNOTATIONS ONLY: no pixels fetched under {data_root}")
        return

    counts = materialise(wanted, data_root / "JPEGImages", workers=arguments.workers)
    report(counts, data_root / "JPEGImages")
    print(f"PREPARED {data_root}")


if __name__ == "__main__":
    main()
