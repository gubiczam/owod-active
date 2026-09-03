"""Frozen DINOv2 features for a **live** candidate pool.

``tools/export_dinov2_features.py`` embeds the committed 80,000-proposal pool and
is bound to it. Benchmark V1 needs the same embedding for a pool that does not
exist until the detector has run, once per task per coverage arm, so the crop
algebra is imported from :mod:`owl.semantic_features` rather than restated:
identical margin, identical shift-before-shrink rule, identical 224x224 bicubic
resize, identical ImageNet normalisation, identical CLS token, identical L2
normalisation. If that module changes, both paths change together.

Nothing here is fitted or tuned. The backbone is frozen, inference only.

**Cost.** Roughly 48,000 crops per task at 1,200 candidate images, which is a few
minutes on a T4. The result is cached beside the task's proposals, keyed on the
identity of the rows it describes, so a resumed session does not pay again and a
*different* population cannot silently reuse another one's features.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from owl import semantic_features as sf

#: Bump when the meaning of the file changes. Never overwrite an older version.
EXPORT_VERSION = "dinov2_vitb14_benchmark_v1"


class SemanticError(sf.ExportError):
    """Raised when features cannot be produced or do not describe the pool."""


def row_fingerprint(image_ids: np.ndarray, boxes: np.ndarray) -> str:
    """Identity of the rows a feature matrix describes.

    Boxes are rounded to six decimals before hashing: PROB's own export writes
    float32, and a fingerprint that changed with the last bit of a re-read would
    make the cache useless without making it safer.
    """

    image_ids = np.asarray(image_ids, dtype=str)
    boxes = np.round(np.asarray(boxes, dtype=np.float64), 6)
    digest = hashlib.sha256()
    digest.update(EXPORT_VERSION.encode())
    digest.update(image_ids.tobytes())
    digest.update(boxes.tobytes())
    return digest.hexdigest()


def load_backbone(device: str = "cuda"):
    """The frozen DINOv2 ViT-B/14, eval mode, gradients off. Never fine-tuned."""

    import torch

    model = torch.hub.load(sf.HUB_REPO, sf.MODEL_ID)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def embed(
    image_ids: np.ndarray,
    boxes: np.ndarray,
    jpeg_dir: str | Path,
    *,
    model,
    device: str = "cuda",
    batch_size: int = 128,
    label: str = "dinov2",
) -> np.ndarray:
    """CLS features for every ``(image_id, box)`` row, in the given row order.

    Crops accumulate across images so the GPU sees full batches; one image
    contributes a few dozen proposals, which would under-fill a sensible batch.
    """

    import time

    import torch
    from PIL import Image
    from torch.nn import functional
    from torchvision.transforms import functional as visual

    image_ids = np.asarray(image_ids, dtype=str)
    boxes = np.asarray(boxes, dtype=np.float64)
    jpeg_dir = Path(jpeg_dir)
    out = np.zeros((image_ids.shape[0], sf.FEATURE_DIM), dtype=np.float32)

    mean = torch.tensor(sf.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(sf.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    crops: list = []
    positions: list[int] = []
    started = time.time()

    def flush() -> None:
        nonlocal crops, positions
        if not crops:
            return
        batch = torch.stack(crops).to(device, non_blocking=True)
        batch = (batch - mean) / std
        with torch.inference_mode():
            features = model.forward_features(batch)["x_norm_clstoken"]
        features = functional.normalize(features.float(), dim=1)
        out[np.asarray(positions, dtype=np.int64)] = features.cpu().numpy()
        crops, positions = [], []

    names = list(dict.fromkeys(image_ids.tolist()))
    where = {name: np.flatnonzero(image_ids == name) for name in names}
    for done, name in enumerate(names, start=1):
        path = jpeg_dir / f"{name}.jpg"
        if not path.is_file():
            raise SemanticError(f"{path} is missing; the semantic pass reads real pixels.")
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        for position in where[name]:
            cx, cy, w, h = boxes[position]
            x0, y0, x1, y1 = sf.square_crop(cx, cy, w, h, width, height)
            crop = image.crop((x0, y0, x1, y1)).resize(
                (sf.CROP_SIZE, sf.CROP_SIZE), Image.BICUBIC
            )
            crops.append(visual.pil_to_tensor(crop).float() / 255.0)
            positions.append(int(position))
            if len(crops) >= batch_size:
                flush()
        if done % 100 == 0 or done == len(names):
            rate = done / max(time.time() - started, 1e-6)
            print(f"  [{label}] {done:,}/{len(names):,} images "
                  f"({rate:.1f} img/s, eta {(len(names) - done) / max(rate, 1e-6) / 60:.1f} min)")
    flush()

    norms = np.linalg.norm(out, axis=1)
    if not np.all(np.isfinite(out)):
        raise SemanticError("the semantic pass produced non-finite features.")
    if np.abs(norms - 1.0).max() > sf.NORM_TOLERANCE:
        raise SemanticError(
            f"features are not unit norm (max deviation {np.abs(norms - 1.0).max():.2e}); "
            "every cosine distance downstream assumes they are."
        )
    return out


def write(path: str | Path, features: np.ndarray, fingerprint: str, provenance: dict) -> Path:
    """Atomic write. fp16 on disk, which is what the committed exports use."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    # Through a handle, not a name: np.savez_compressed appends '.npz' to any
    # path that does not already end in it, and the rename would then look for a
    # file that was never written.
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=np.asarray(features, dtype=np.float16),
            fingerprint=np.asarray(fingerprint),
            export_version=np.asarray(EXPORT_VERSION),
            provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
        )
    temporary.replace(path)
    return path


def read(path: str | Path, *, fingerprint: str | None = None) -> np.ndarray:
    """Read a cached matrix, refusing one that describes different rows."""

    payload = np.load(Path(path), allow_pickle=False)
    version = str(payload["export_version"])
    if version != EXPORT_VERSION:
        raise SemanticError(f"{path} is {version!r}; this code reads {EXPORT_VERSION!r}.")
    stored = str(payload["fingerprint"])
    if fingerprint is not None and stored != fingerprint:
        raise SemanticError(
            f"{path} describes rows {stored[:12]} and this pool is {fingerprint[:12]}. "
            "Reusing it would embed one population's geometry in another's "
            "selection; delete the file to recompute."
        )
    return np.asarray(payload["features"], dtype=np.float32)


def cached(
    path: str | Path,
    image_ids: np.ndarray,
    boxes: np.ndarray,
    jpeg_dir: str | Path,
    *,
    model_factory=load_backbone,
    device: str = "cuda",
    batch_size: int = 128,
    label: str = "dinov2",
    provenance: dict | None = None,
) -> np.ndarray:
    """Read the cache if it matches these rows, otherwise embed and write it."""

    path = Path(path)
    fingerprint = row_fingerprint(image_ids, boxes)
    if path.exists():
        return read(path, fingerprint=fingerprint)
    model = model_factory(device)
    features = embed(
        image_ids, boxes, jpeg_dir,
        model=model, device=device, batch_size=batch_size, label=label,
    )
    write(path, features, fingerprint, {
        "rows": int(features.shape[0]),
        "images": len(set(np.asarray(image_ids, dtype=str).tolist())),
        "crop": sf.crop_specification(),
        **(provenance or {}),
    })
    return features


# --------------------------------------------------------------- reference ---


def reference_from_ref_t1(path: str | Path) -> np.ndarray:
    """The balanced task-1 labelled reference, as feature rows.

    This is *already-labelled data* — the split PROB's ``t1.pth`` was trained on
    — so consulting it is not oracle leakage: it is the definition of "diversity
    relative to what we already have", which is what the 2026-08-25
    consultation asked for. It is the frozen Method V2 artefact, unchanged.
    """

    from tools.export_ref_t1_features import read as read_ref

    payload = read_ref(path)
    features = np.asarray(payload["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, 1e-9)).astype(np.float32)


def stack_reference(parts: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate reference blocks, dropping empty ones."""

    blocks = [np.asarray(p, dtype=np.float32) for p in parts if p is not None and len(p)]
    if not blocks:
        return np.zeros((0, sf.FEATURE_DIM), dtype=np.float32)
    return np.vstack(blocks)
