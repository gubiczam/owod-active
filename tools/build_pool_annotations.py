"""Pack the candidate images' VOC annotations into one archive for Colab.

PROB's training loader reads one XML per image out of ``Annotations/``. The GPU
chain needs those files for every image the selector might open — the 28,800
images in ``data/reference/per_image_class_counts.json`` — and Colab cannot
mount a 120 GB dataset. This packs exactly those XMLs and nothing else.

    python tools/build_pool_annotations.py \
        --annotations /Volumes/AI_SSD/datasets/owod_canonical/Annotations \
        --output data/staging/owdetr_pool_annotations.tar.gz

Upload the result to Drive once. It does not change between runs.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument(
        "--index", type=Path,
        default=ROOT / "data" / "reference" / "per_image_class_counts.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "staging" / "owdetr_pool_annotations.tar.gz",
    )
    arguments = parser.parse_args()

    image_ids = sorted(json.loads(arguments.index.read_text(encoding="utf-8")))
    source = arguments.annotations
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    written, missing = 0, []
    with tarfile.open(arguments.output, "w:gz") as archive:
        for image_id in image_ids:
            path = source / f"{image_id}.xml"
            if not path.exists():
                missing.append(image_id)
                continue
            archive.add(path, arcname=f"Annotations/{image_id}.xml")
            written += 1

    size = arguments.output.stat().st_size / 1e6
    print(f"wrote {arguments.output}  ({written} annotations, {size:.1f} MB)")
    if missing:
        print(f"MISSING {len(missing)} annotations, first few: {missing[:5]}")
        manifest = arguments.output.with_name(arguments.output.stem + "_missing.json")
        manifest.write_text(json.dumps(missing), encoding="utf-8")
        print(f"full list: {manifest}")
    else:
        print("every candidate image has an annotation.")


if __name__ == "__main__":
    main()
