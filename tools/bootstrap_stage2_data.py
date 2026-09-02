#!/usr/bin/env python
"""Materialise everything Method V2 Stage 2 needs, from a bare runtime.

A fresh Colab VM has an empty ``/content``, so a Run-All notebook has to
bootstrap its own inputs. This is that step, kept out of the notebook so it can
be tested on a laptop.

**What Stage 2 actually needs, established by reading the tools rather than
guessing.** All three exporters read exactly one thing from ``--data-root``:

    <data-root>/JPEGImages/<image_id>.jpg

No ``Annotations/``, no ``ImageSets/``, and no PROB checkout. Boxes come from
committed artefacts instead -- candidate boxes from ``data/pool/...npz`` and
REF-T1 boxes from ``data/staging/owdetr_replay_annotations.tar.gz`` via
:mod:`owl.reference_t1`. ``tools/diagnose_method_v2_stage2.py`` needs no data
root at all: the committed pool plus the three feature exports.

So the whole bootstrap is one image set:

===========================  =========  ============================
candidate pool                   1,600  base export + consistency views
REF-T1 selected sources         14,901  the reference export
overlap                            127
**union to fetch**          **16,374**
===========================  =========  ============================

The frozen REF-T1 identity is asserted *before* anything is fetched, so a run
that would have produced a different reference stops immediately rather than
after half an hour of downloading.

Downloading reuses ``tools/materialize_pool_images.py`` unchanged -- one image
source, one validated download path, resumable, and it already refuses to
overwrite a valid JPEG.

    python tools/bootstrap_stage2_data.py --data-root /content/data/OWOD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import reference_t1 as ref
from owl import semantic_features as sf
from tools.materialize_pool_images import materialise, report

#: The frozen REF-T1 identity, from docs/method_v2_stage2_protocol_2026-09-02.md.
#: Asserted before any download, because a changed manifest means a changed
#: experiment and no amount of fetching would fix it.
EXPECTED_REF_T1_OBJECTS = 19_000
EXPECTED_REF_T1_IMAGES = 14_901
EXPECTED_REF_T1_MANIFEST = (
    "a062fc8f4fd43ea52842725aeaa5eccc0e06eab1894b867b248927bd9d2a2a63"
)

#: The candidate pool's own images, needed by the base export and the views.
EXPECTED_POOL_IMAGES = 1_600
EXPECTED_UNION_IMAGES = 16_374


def frozen_reference() -> ref.ReferenceObjects:
    """Reproduce REF-T1 and refuse unless it is bit-for-bit the frozen one."""

    selection = ref.select_balanced(
        ref.enumerate_objects(),
        per_class_cap=ref.PRIMARY_REF_T1_CAP_PER_CLASS,
    )
    summary = selection.summary()
    fingerprint = selection.provenance["manifest_sha256"]

    problems = []
    if summary["objects"] != EXPECTED_REF_T1_OBJECTS:
        problems.append(
            f"{summary['objects']:,} objects, expected {EXPECTED_REF_T1_OBJECTS:,}")
    if summary["images"] != EXPECTED_REF_T1_IMAGES:
        problems.append(
            f"{summary['images']:,} images, expected {EXPECTED_REF_T1_IMAGES:,}")
    if not summary["balanced"]:
        problems.append("the selection is not exactly class-balanced")
    if fingerprint != EXPECTED_REF_T1_MANIFEST:
        problems.append(
            f"manifest {fingerprint} != frozen {EXPECTED_REF_T1_MANIFEST}")
    if problems:
        raise SystemExit(
            "REF-T1 did not reproduce the frozen reference:\n  "
            + "\n  ".join(problems)
            + "\nThis is a different experiment; refusing to continue."
        )

    print(f"  REF-T1 objects : {summary['objects']:,} "
          f"({summary['min_per_class']}/class x 19, exactly balanced)")
    print(f"  REF-T1 images  : {summary['images']:,}")
    print(f"  manifest SHA256: {fingerprint}  MATCHES FROZEN")
    return selection


def pool_images(pool: Path) -> list[str]:
    """The candidate pool's own image ids, from the committed npz."""

    payload = np.load(pool, allow_pickle=True)
    splits = np.asarray(payload["split"], dtype=str)
    ids = np.asarray(payload["image_ids"], dtype=str)
    images = sorted(set(ids[splits == sf.POOL_SPLIT].tolist()))
    if len(images) != EXPECTED_POOL_IMAGES:
        raise SystemExit(
            f"{pool} names {len(images):,} pool images, expected "
            f"{EXPECTED_POOL_IMAGES:,}"
        )
    return images


def union_image_list(selection: ref.ReferenceObjects, pool: Path) -> list[str]:
    """Every image Stage 2 will open, deduplicated and sorted."""

    images = sorted(set(pool_images(pool)) | set(selection.images))
    if len(images) != EXPECTED_UNION_IMAGES:
        raise SystemExit(
            f"the union is {len(images):,} images, expected "
            f"{EXPECTED_UNION_IMAGES:,}; the inputs are not the frozen ones"
        )
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pool", default=str(sf.POOL))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--image-list-out", default=None,
                        help="also write the union list here, for the record")
    parser.add_argument("--verify-only", action="store_true",
                        help="reproduce the manifest and count images, fetch nothing")
    arguments = parser.parse_args()

    print("Reproducing the frozen REF-T1 reference before fetching anything:")
    selection = frozen_reference()

    images = union_image_list(selection, Path(arguments.pool))
    print(f"\n  candidate pool images  : {EXPECTED_POOL_IMAGES:,}")
    print(f"  REF-T1 source images   : {len(selection.images):,}")
    print(f"  union to materialise   : {len(images):,}")

    if arguments.image_list_out:
        Path(arguments.image_list_out).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.image_list_out).write_text(
            "\n".join(images) + "\n", encoding="utf-8")
        print(f"  wrote {arguments.image_list_out}")

    if arguments.verify_only:
        print("\n--verify-only: nothing fetched")
        return

    jpeg = Path(arguments.data_root) / "JPEGImages"
    print(f"\nMaterialising into {jpeg} (resumable; valid files are reused):")
    report(materialise(images, jpeg, workers=arguments.workers), jpeg)
    print(f"\nready: {jpeg}")


if __name__ == "__main__":
    main()
