#!/usr/bin/env python
"""Export the two frozen context views for P2, for Method V2 Stage 2's component C.

C tests **semantic stability under mild deterministic context change**, not
density: three density operationalisations already failed in the same direction,
because in this pool local density orders background < known < unknown-head <
unknown-tail and no threshold reverses a monotone ordering.

Views frozen in ``docs/method_v2_stage2_protocol_2026-09-02.md`` section 6 before
any C value was computed:

===========  ==================================================
base         the frozen 1.20x square crop (the existing export)
view A       same centre, 1.10x square
view B       same centre, 1.30x square
===========  ==================================================

**Only the margin changes.** The square construction, shift-before-shrink
boundary handling, 224x224 bicubic resize, ImageNet normalisation, CLS extraction
and L2 normalisation all come from :mod:`owl.semantic_features`, unchanged --
there is one crop implementation and one preprocessing pipeline, so a difference
between views cannot be an artefact of a second code path. No colour jitter, no
flip, no stochastic augmentation, no oracle-dependent view choice.

Exported for **P2 only** (15,518 rows x 2 views), never overwriting the base
export.

    python tools/export_dinov2_consistency_views.py \\
        --data-root /content/data/OWOD \\
        --out /content/drive/MyDrive/OWL/features/dinov2_vitb14_stage2_views_v1.npz \\
        --smoke-images 4
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import method_v2_stage2 as stage2
from owl import proposals as proposals_module
from owl import semantic_features as sf
from tools.audit_decoder_layers import populations
from tools.diagnose_representation import load
from tools.export_dinov2_features import git_sha, load_backbone

#: Bump when the file's meaning changes.
VIEWS_VERSION = "dinov2_vitb14_stage2_views_v1"

#: The two frozen context margins, beside the base 1.20x.
VIEW_MARGINS = {"view_a": 1.10, "view_b": 1.30}


def p2_rows(pool_path: Path) -> tuple[sf.PoolRows, np.ndarray, dict]:
    """The fixed P2 population, reusing the repository's own construction."""

    rows = sf.pool_rows(pool_path)
    pool = load()
    payload = np.load(pool_path, allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == sf.POOL_SPLIT
    pool["raw_boxes"] = payload["boxes"][keep].astype(np.float32)
    candidates = proposals_module.from_frozen_pool(pool_path, split=sf.POOL_SPLIT)
    masks = populations(pool, candidates)
    mask = masks["P2_admissible_nms"]
    report = stage2.verify_p2(mask, pool["kind"])
    print(f"[p2] {report['rows']:,} rows, background {report['background_share']:.4f}  PASS")
    return rows, mask, report


def embed_views(model, rows: sf.PoolRows, selection: np.ndarray, jpeg: Path, *,
                device: str, batch_size: int, label: str) -> dict[str, dict[str, np.ndarray]]:
    """CLS features for every frozen view of the selected proposals."""

    import torch
    from PIL import Image
    from torch.nn import functional
    from torchvision.transforms import functional as visual

    by_image: dict[str, list[int]] = {}
    for position in selection:
        by_image.setdefault(str(rows.image_ids[position]), []).append(int(position))
    names = sorted(by_image)

    mean = torch.tensor(sf.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(sf.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out: dict[str, dict[str, np.ndarray]] = {view: {} for view in VIEW_MARGINS}
    buffer: dict[str, list] = {"crops": [], "keys": [], "views": []}
    started = time.time()

    def flush() -> None:
        if not buffer["crops"]:
            return
        batch = torch.stack(buffer["crops"]).to(device, non_blocking=True)
        batch = (batch - mean) / std
        with torch.inference_mode():
            features = model.forward_features(batch)["x_norm_clstoken"]
        features = functional.normalize(features.float(), dim=1)
        block = features.cpu().numpy().astype(np.float16)
        for key, view, vector in zip(buffer["keys"], buffer["views"], block):
            out[view][key] = vector
        buffer["crops"], buffer["keys"], buffer["views"] = [], [], []

    for done, name in enumerate(names, start=1):
        with Image.open(jpeg / f"{name}.jpg") as handle:
            image = handle.convert("RGB")
        width, height = image.size
        for position in by_image[name]:
            cx, cy, w, h = rows.boxes[position]
            for view, margin in VIEW_MARGINS.items():
                x0, y0, x1, y1 = sf.square_crop(
                    cx, cy, w, h, width, height, margin=margin)
                crop = image.crop((x0, y0, x1, y1)).resize(
                    (sf.CROP_SIZE, sf.CROP_SIZE), Image.BICUBIC)
                buffer["crops"].append(visual.pil_to_tensor(crop).float() / 255.0)
                buffer["keys"].append(str(rows.keys[position]))
                buffer["views"].append(view)
                if len(buffer["crops"]) >= batch_size:
                    flush()
        if done % 50 == 0 or done == len(names):
            rate = done / max(time.time() - started, 1e-6)
            print(f"[{label}] {done:,}/{len(names):,} images ({rate:.1f} img/s, "
                  f"eta {(len(names) - done) / max(rate, 1e-6) / 60:.1f} min)")
    flush()
    return out


def write(path: Path, keys: np.ndarray, views: dict[str, np.ndarray],
          provenance: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, keys=np.asarray(keys, dtype=str),
        provenance=np.asarray(str(provenance)),
        views_version=np.asarray(VIEWS_VERSION),
        **{view: np.asarray(matrix, dtype=np.float16) for view, matrix in views.items()},
    )
    return path


def read(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    payload = np.load(Path(path), allow_pickle=True)
    version = str(payload["views_version"])
    if version != VIEWS_VERSION:
        raise stage2.Stage2Error(
            f"{path} is {version!r}; this code reads {VIEWS_VERSION!r}"
        )
    import ast
    return (
        np.asarray(payload["keys"], dtype=str),
        {view: payload[view] for view in VIEW_MARGINS},
        ast.literal_eval(str(payload["provenance"])),
    )


def validate(keys: np.ndarray, views: dict[str, np.ndarray]) -> dict:
    """Fail closed on the ways a view export can be quietly wrong."""

    if np.unique(keys).size != keys.size:
        raise stage2.Stage2Error("duplicate proposal identities in the view export")
    report = {"rows": int(keys.size), "views": sorted(views)}
    for view, matrix in views.items():
        features = np.asarray(matrix, dtype=np.float32)
        if features.shape[0] != keys.size:
            raise stage2.Stage2Error(
                f"{view} has {features.shape[0]} rows against {keys.size} keys")
        if features.shape[1] != sf.FEATURE_DIM:
            raise stage2.Stage2Error(
                f"{view} dimension {features.shape[1]}, expected {sf.FEATURE_DIM}")
        if not np.isfinite(features).all():
            raise stage2.Stage2Error(f"{view} holds non-finite features")
        norms = np.linalg.norm(features, axis=1)
        if (norms <= 0).any():
            raise stage2.Stage2Error(f"{view} holds zero-norm features")
        deviation = float(np.abs(norms - 1.0).max())
        if deviation > sf.NORM_TOLERANCE:
            raise stage2.Stage2Error(
                f"{view} is not L2-normalised: worst |‖v‖-1| = {deviation:.2e}")
        report[f"{view}_worst_norm_deviation"] = deviation
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", default=str(sf.POOL))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-images", type=int, default=0,
                        help="run the whole path on N P2 images, validate, write "
                             "nothing, and exit")
    arguments = parser.parse_args()

    import torch

    rows, mask, report = p2_rows(Path(arguments.pool))
    selection = np.flatnonzero(mask)
    jpeg = Path(arguments.data_root) / "JPEGImages"
    if not jpeg.is_dir():
        raise stage2.Stage2Error(
            f"{jpeg} missing -- run tools/materialize_pool_images.py")

    out = Path(arguments.out)
    if out.exists() and not arguments.smoke_images:
        print(f"[resume] {out} exists; verifying instead of re-exporting")
        keys, views, _ = read(out)
        print(f"[resume] {validate(keys, views)}")
        return

    device = arguments.device if torch.cuda.is_available() else "cpu"
    model = load_backbone(device)

    if arguments.smoke_images:
        images = sorted({str(rows.image_ids[p]) for p in selection})[:arguments.smoke_images]
        subset = np.asarray(
            [p for p in selection if str(rows.image_ids[p]) in set(images)],
            dtype=np.int64,
        )
        print(f"[smoke] {len(images)} P2 images, {subset.size} proposals, "
              f"{subset.size * len(VIEW_MARGINS)} crops")
        produced = embed_views(model, rows, subset, jpeg, device=device,
                               batch_size=arguments.batch_size, label="smoke")
        keys = np.asarray([str(rows.keys[p]) for p in subset])
        views = {view: np.stack([produced[view][key] for key in keys])
                 for view in VIEW_MARGINS}
        print(f"[smoke] {validate(keys, views)}  PASS")
        similarity = float((views["view_a"].astype(np.float32)
                            * views["view_b"].astype(np.float32)).sum(axis=1).mean())
        print(f"[smoke] mean cos(view_a, view_b) = {similarity:.4f} "
              "(views differ, as they must, and are not identical copies)")
        if similarity > 0.9999:
            raise stage2.Stage2Error(
                "the two views are effectively identical; the margin is not being "
                "applied and C would be meaningless"
            )
        print("[smoke] the full view export is safe to run")
        return

    started = time.time()
    produced = embed_views(model, rows, selection, jpeg, device=device,
                           batch_size=arguments.batch_size, label="views")
    keys = np.asarray([str(rows.keys[p]) for p in selection])
    views = {view: np.stack([produced[view][key] for key in keys])
             for view in VIEW_MARGINS}

    provenance = {
        "views_version": VIEWS_VERSION,
        "git_sha": git_sha(),
        "model_id": sf.MODEL_ID,
        "model_source": sf.HUB_REPO,
        "base_margin": sf.CROP_MARGIN,
        "view_margins": VIEW_MARGINS,
        "crop": sf.crop_specification(),
        "population": "P2_admissible_nms",
        "p2": report,
        "pool_sha256": sf.sha256(arguments.pool),
        "image_root": str(jpeg),
        "rows": int(keys.size),
        "feature_dim": sf.FEATURE_DIM,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    checked = validate(keys, views)
    write(out, keys, views, provenance)
    Path(str(out) + ".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\n[gate] {checked}  PASS")
    print(f"[done] {out}  in {provenance['wall_clock_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
