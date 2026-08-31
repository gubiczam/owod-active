#!/usr/bin/env python3
"""Verify or fetch the exact COCO JPEG union needed by controlled T1 anchors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import evaluation_subset, protocol, t1_anchor  # noqa: E402

COCO_SOURCE = "https://s3.amazonaws.com/images.cocodataset.org"
DEFAULT_MANIFEST_ROOT = ROOT / "data" / "reference" / "longtail"
DEFAULT_TEST_ARCHIVE = ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz"


def valid_jpeg(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                return False
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def required_ids(conditions: tuple[str, ...], manifest_root: Path, test_archive: Path) -> list[str]:
    image_ids: set[str] = set()
    for condition in conditions:
        _, manifest = t1_anchor.condition_manifest(condition, manifest_root)
        image_ids.update(t1_anchor.load_selection(manifest, repository_root=ROOT))
    subset = evaluation_subset.from_archive(
        test_archive,
        protocol.build_chain(6)[-1].known_classes,
        seed=0,
        remainder_multiplier=t1_anchor.EVALUATION_REMAINDER_MULTIPLIER,
        max_per_class=t1_anchor.EVALUATION_MAX_PER_CLASS,
    )
    if len(subset.image_ids) != 4308:
        raise t1_anchor.AnchorError("Shared evaluation image count changed.")
    image_ids.update(subset.image_ids)
    invalid = [value for value in image_ids if len(value) != 12 or not value.isdigit()]
    if invalid:
        raise t1_anchor.AnchorError(f"Invalid canonical COCO IDs: {sorted(invalid)[:10]}.")
    return sorted(image_ids)


def fetch_one(image_id: str, jpeg_root: Path) -> tuple[str, str | None]:
    target = jpeg_root / f"{image_id}.jpg"
    partial = jpeg_root / f".{image_id}.jpg.part"
    if partial.exists():
        return image_id, f"stale partial exists: {partial}"
    errors: list[str] = []
    for split in ("train2017", "val2017"):
        result = subprocess.run(
            ["curl", "--fail", "--silent", "--show-error", "--location",
             "--retry", "3", "--retry-delay", "1", "--connect-timeout", "20",
             "--max-time", "180", "--output", str(partial),
             f"{COCO_SOURCE}/{split}/{image_id}.jpg"],
            text=True, capture_output=True, check=False,
        )
        if result.returncode == 0 and valid_jpeg(partial):
            partial.replace(target)
            return image_id, None
        if partial.exists():
            partial.unlink()
        detail = (result.stderr or "invalid JPEG response").strip().splitlines()[-1]
        errors.append(f"{split}: {detail}")
    return image_id, "; ".join(errors)


def materialize(arguments: argparse.Namespace) -> dict[str, object]:
    conditions = tuple(item.strip() for item in arguments.conditions.split(",") if item.strip())
    if not conditions or any(item not in t1_anchor.PRIMARY_CONDITIONS for item in conditions):
        raise t1_anchor.AnchorError("Conditions must be an LT-10/LT-50/LT-100 subset.")
    image_ids = required_ids(conditions, arguments.manifest_root, arguments.test_archive)
    if arguments.execute:
        arguments.jpeg_root.mkdir(parents=True, exist_ok=True)
    valid = [image_id for image_id in image_ids
             if valid_jpeg(arguments.jpeg_root / f"{image_id}.jpg")]
    valid_set = set(valid)
    invalid_existing = [
        image_id for image_id in image_ids
        if (arguments.jpeg_root / f"{image_id}.jpg").exists() and image_id not in valid_set
    ]
    if invalid_existing:
        raise t1_anchor.AnchorError(
            f"Canonical JPEG root contains {len(invalid_existing)} invalid files; "
            f"first {invalid_existing[:10]}."
        )
    missing = sorted(set(image_ids) - set(valid))
    failures: list[tuple[str, str | None]] = []
    if arguments.execute and missing:
        with ThreadPoolExecutor(max_workers=min(arguments.workers, len(missing))) as pool:
            failures = [row for row in pool.map(
                lambda value: fetch_one(value, arguments.jpeg_root), missing)
                if row[1] is not None]
    ready = [image_id for image_id in image_ids
             if valid_jpeg(arguments.jpeg_root / f"{image_id}.jpg")]
    report = {
        "schema": "controlled_t1_jpeg_materialization_v1",
        "conditions": list(conditions),
        "source": COCO_SOURCE,
        "required": len(image_ids),
        "already_present": len(valid),
        "downloaded": len(ready) - len(valid),
        "missing": len(image_ids) - len(ready),
        "jpeg_root": str(arguments.jpeg_root.resolve()),
        "execute": arguments.execute,
        "failures": failures[:20],
    }
    if arguments.execute and (failures or len(ready) != len(image_ids)):
        raise t1_anchor.AnchorError(f"COCO JPEG materialization incomplete: {report}.")
    return report


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--conditions", required=True)
    command.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    command.add_argument("--test-archive", type=Path, default=DEFAULT_TEST_ARCHIVE)
    command.add_argument("--jpeg-root", type=Path, required=True)
    command.add_argument("--workers", type=int, default=32)
    command.add_argument("--execute", action="store_true")
    return command


def main() -> int:
    try:
        arguments = parser().parse_args()
        if arguments.workers < 1:
            raise t1_anchor.AnchorError("At least one download worker is required.")
        report = materialize(arguments)
    except (t1_anchor.AnchorError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["execute"] and report["missing"] == 0:
        print("CONTROLLED LT JPEG MATERIALIZATION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
