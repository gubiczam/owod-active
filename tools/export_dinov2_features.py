#!/usr/bin/env python
"""Export frozen DINOv2 ViT-B/14 CLS features for the pool's proposal crops.

Inference only. The backbone is frozen and never fine-tuned; this stage answers
whether the *representation* was the binding constraint, so training it would
destroy the question.

Protocol: ``docs/method_v2_protocol_2026-09-02.md``. The crop is frozen there and
implemented in :func:`owl.semantic_features.square_crop` -- square, centred,
1.20x the larger proposal side, shifted to fit before being shrunk, **never
padded**, because a grey border's area would vary with how close a proposal sits
to the image edge and would write an edge-proximity signal into the embedding.

Two modes:

``--smoke-images N``
    the whole path on N images, writing nothing: crop geometry, dimensionality,
    finiteness, unit norm, and determinism of a repeated forward pass. Run this
    first; the full export is 80,000 crops.
default
    all 80,000 pool proposals over 1,600 images, in the pool file's row order.

    python tools/export_dinov2_features.py --data-root /content/data/OWOD \\
        --out /content/drive/MyDrive/OWL/features/dinov2_vitb14_method_v2_v1.npz \\
        --smoke-images 4
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import semantic_features as sf

ROOT = Path(__file__).resolve().parent.parent
POOL = sf.POOL


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def preflight(rows: sf.PoolRows, data_root: Path, out: Path) -> Path:
    """Assert what must hold before a GPU session is spent discovering it."""

    problems = []
    jpeg = data_root / "JPEGImages"
    if not jpeg.is_dir():
        problems.append(f"{jpeg} missing -- run tools/materialize_pool_images.py")
    else:
        missing = [name for name in rows.images if not (jpeg / f"{name}.jpg").is_file()]
        if missing:
            problems.append(
                f"{len(missing)} of {len(rows.images)} pool images absent from "
                f"{jpeg} (first: {missing[:3]})"
            )
    if problems:
        raise sf.ExportError("preflight failed:\n  " + "\n  ".join(problems))

    out.parent.mkdir(parents=True, exist_ok=True)
    probe = out.parent / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    print(f"[preflight] {len(rows):,} pool proposals over {len(rows.images):,} images; "
          f"images present; {out.parent} writable")
    return jpeg


def load_backbone(device: str):
    """The frozen DINOv2 ViT-B/14, in eval mode. No fine-tuning, ever."""

    import torch

    model = torch.hub.load(sf.HUB_REPO, sf.MODEL_ID)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def embed_images(
    model, names: list[str], rows: sf.PoolRows, jpeg: Path, *,
    device: str, batch_size: int, label: str,
) -> dict[str, np.ndarray]:
    """CLS features for every pool proposal on ``names``, keyed by proposal key.

    Crops are accumulated across images so the GPU sees full batches: one image
    contributes 50 proposals, which under-fills a sensible batch and would make
    the pass three times longer than it needs to be.
    """

    import torch
    from PIL import Image
    from torch.nn import functional
    from torchvision.transforms import functional as visual

    wanted = {name: np.flatnonzero(rows.image_ids == name) for name in names}
    out: dict[str, np.ndarray] = {}
    buffer_crops: list[torch.Tensor] = []
    buffer_keys: list[str] = []
    mean = torch.tensor(sf.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(sf.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    started = time.time()

    def flush() -> None:
        nonlocal buffer_crops, buffer_keys
        if not buffer_crops:
            return
        batch = torch.stack(buffer_crops).to(device, non_blocking=True)
        batch = (batch - mean) / std
        with torch.inference_mode():
            features = model.forward_features(batch)["x_norm_clstoken"]
        features = functional.normalize(features.float(), dim=1)
        block = features.cpu().numpy().astype(np.float16)
        for key, vector in zip(buffer_keys, block):
            out[key] = vector
        buffer_crops, buffer_keys = [], []

    for done, name in enumerate(names, start=1):
        with Image.open(jpeg / f"{name}.jpg") as handle:
            image = handle.convert("RGB")
        width, height = image.size
        for position in wanted[name]:
            cx, cy, w, h = rows.boxes[position]
            x0, y0, x1, y1 = sf.square_crop(cx, cy, w, h, width, height)
            crop = image.crop((x0, y0, x1, y1)).resize(
                (sf.CROP_SIZE, sf.CROP_SIZE), Image.BICUBIC
            )
            buffer_crops.append(visual.pil_to_tensor(crop).float() / 255.0)
            buffer_keys.append(str(rows.keys[position]))
            if len(buffer_crops) >= batch_size:
                flush()
        if done % 50 == 0 or done == len(names):
            rate = done / max(time.time() - started, 1e-6)
            print(f"[{label}] {done:,}/{len(names):,} images ({rate:.1f} img/s, "
                  f"eta {(len(names) - done) / max(rate, 1e-6) / 60:.1f} min)")
    flush()
    return out


def assemble(features: dict[str, np.ndarray], rows: sf.PoolRows,
             keys: np.ndarray, provenance: dict) -> sf.SemanticExport:
    """Stack into the pool's own row order for the given keys."""

    missing = [key for key in keys.tolist() if key not in features]
    if missing:
        raise sf.ExportError(
            f"{len(missing)} proposals produced no feature (first: {missing[:3]})"
        )
    order = {key: position for position, key in enumerate(rows.keys.tolist())}
    positions = np.asarray([order[key] for key in keys.tolist()], dtype=np.int64)
    return sf.SemanticExport(
        embeddings=np.stack([features[key] for key in keys.tolist()]),
        keys=keys,
        image_ids=rows.image_ids[positions],
        query_index=rows.query_index[positions],
        row_index=rows.row_index[positions],
        provenance=provenance,
    )


def smoke(model, rows: sf.PoolRows, jpeg: Path, *, names: list[str],
          device: str, batch_size: int, provenance: dict) -> None:
    """The sanity gate. Writes nothing; raises rather than reporting a warning."""

    import torch

    print(f"[smoke] {len(names)} images, {int(np.isin(rows.image_ids, names).sum())} "
          "proposals")

    # crop geometry must be non-empty and square for every proposal involved
    from PIL import Image
    for name in names:
        with Image.open(jpeg / f"{name}.jpg") as handle:
            width, height = handle.size
        for position in np.flatnonzero(rows.image_ids == name):
            cx, cy, w, h = rows.boxes[position]
            x0, y0, x1, y1 = sf.square_crop(cx, cy, w, h, width, height)
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise sf.ExportError(
                    f"crop {(x0, y0, x1, y1)} outside {width}x{height} for "
                    f"{rows.keys[position]}"
                )
            if (x1 - x0) != (y1 - y0):
                raise sf.ExportError(f"crop is not square: {(x0, y0, x1, y1)}")
    print("[smoke] crop geometry: non-empty, square, inside the image  PASS")

    produced = embed_images(model, names, rows, jpeg, device=device,
                            batch_size=batch_size, label="smoke")
    keys = rows.keys[np.isin(rows.image_ids, names)]
    export = assemble(produced, rows, keys, provenance)
    report = sf.validate(export, rows, full=False)
    print(f"[smoke] dimension {report['dimension']}, "
          f"worst |‖v‖-1| = {report['worst_norm_deviation']:.2e}  PASS")

    # determinism: the same crops, twice, in eval mode
    again = embed_images(model, names[:1], rows, jpeg, device=device,
                         batch_size=batch_size, label="smoke-repeat")
    shared = sorted(set(again) & set(produced))
    left = np.stack([produced[key] for key in shared]).astype(np.float32)
    right = np.stack([again[key] for key in shared]).astype(np.float32)
    similarity = float((left * right).sum(axis=1).mean())
    drift = float(np.abs(left - right).max())
    if similarity < 1.0 - 1e-4:
        raise sf.ExportError(
            f"repeated inference disagrees: mean cosine {similarity:.6f}. The "
            "model is not deterministic in eval mode."
        )
    print(f"[smoke] repeated inference on {len(shared)} crops: mean cosine "
          f"{similarity:.6f}, max abs drift {drift:.2e}  PASS")
    print("[smoke] the full export is safe to run")
    del torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", default=str(POOL))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-images", type=int, default=0,
                        help="run the full path on the first N images by id, "
                             "validate, write nothing, and exit")
    arguments = parser.parse_args()

    import torch

    rows = sf.pool_rows(arguments.pool)
    out = Path(arguments.out)
    jpeg = preflight(rows, Path(arguments.data_root), out)

    if out.exists() and not arguments.smoke_images:
        print(f"[resume] {out} exists; verifying instead of re-exporting")
        report = sf.validate(sf.read(out), rows)
        print(f"[resume] {report}")
        return

    device = arguments.device if torch.cuda.is_available() else "cpu"
    if device != arguments.device:
        print(f"[device] CUDA unavailable; falling back to {device}")
    model = load_backbone(device)

    provenance = {
        "export_version": sf.EXPORT_VERSION,
        "git_sha": git_sha(),
        "model_id": sf.MODEL_ID,
        "model_source": sf.HUB_REPO,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
        "pool_sha256": sf.sha256(arguments.pool),
        "pool_split": sf.POOL_SPLIT,
        "image_root": str(Path(arguments.data_root) / "JPEGImages"),
        "crop": sf.crop_specification(),
        "proposals": len(rows),
        "images": len(rows.images),
        "feature_dim": sf.FEATURE_DIM,
    }

    if arguments.smoke_images:
        smoke(model, rows, jpeg, names=rows.images[:arguments.smoke_images],
              device=device, batch_size=arguments.batch_size, provenance=provenance)
        return

    started = time.time()
    produced = embed_images(model, rows.images, rows, jpeg, device=device,
                            batch_size=arguments.batch_size, label="export")
    provenance["wall_clock_seconds"] = round(time.time() - started, 1)

    export = assemble(produced, rows, rows.keys, provenance)
    report = sf.validate(export, rows)
    sf.write(out, export)
    Path(str(out) + ".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"\n[gate] {report}  PASS")
    print(f"[done] {out}  in {provenance['wall_clock_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
