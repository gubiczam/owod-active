#!/usr/bin/env python
"""Export DINOv2 features for REF-T1: the canonical Task-1 labelled reference.

Method V2 Stage 2's ``D`` is *novelty relative to already-labelled knowledge*, so
the reference set is part of the method's meaning. The first implementation used
REF-A -- ``predicted_known(candidate pool) & NMS`` -- which is oracle-free but
estimates a pseudo-known manifold **from the same unlabelled pool it then
judges**, measuring a different quantity. REF-T1 is the real labelled set:
ground-truth boxes of the 19 Task-1 classes from the canonical T1 *training*
annotations. GT is legitimate here because these are already-labelled training
examples at round 0, not candidate oracle information and not a future task's
labels.

**One pipeline, not two.** Crop geometry, context scale, 224x224 resize,
ImageNet normalisation, CLS extraction and L2 normalisation all come from
:mod:`owl.semantic_features` unchanged, and the backbone loader is the candidate
exporter's own. A reference embedded by a second pipeline would make every cosine
against it uninterpretable.

Requires the T1 source images. The selection is class-balanced and nested across
caps under the fixed seed, so exporting at the largest cap yields the smaller ones
as free subsets -- a sensitivity check needs no second GPU pass.

    python tools/export_ref_t1_features.py --data-root /content/data/OWOD \\
        --out /content/drive/MyDrive/OWL/features/ref_t1_dinov2_vitb14_cap1000_v1.npz \\
        --per-class-cap 1000 --smoke-images 4
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

from owl import reference_t1 as ref
from owl import semantic_features as sf
from tools.export_dinov2_features import git_sha, load_backbone

#: Bump when the file's meaning changes.
REF_EXPORT_VERSION = "ref_t1_dinov2_vitb14_v1"


def candidate_overlap(selection: ref.ReferenceObjects, pool: Path) -> dict:
    """How many REF-T1 images also appear in the candidate pool and in eval.

    **Not leakage for the primary protocol**: Task-1 labelled objects are
    legitimately available prior-task supervision, so a candidate sitting on a
    T1-labelled object *should* score low novelty. Same-image references are
    deliberately kept and no no-same-image variant is built for the primary
    analysis.

    Recorded because same-image visual context is a secondary confound worth
    being able to point at later. Overlap with **eval** would be a different
    matter, so it is measured too.
    """

    payload = np.load(pool, allow_pickle=True)
    splits = np.asarray(payload["split"], dtype=str)
    ids = np.asarray(payload["image_ids"], dtype=str)
    reference_images = set(selection.images)
    candidate_images = set(ids[splits == "pool"].tolist())
    eval_images = set(ids[splits == "eval"].tolist())
    return {
        "reference_images": len(reference_images),
        "candidate_pool_images": len(candidate_images),
        "reference_and_candidate_pool": len(reference_images & candidate_images),
        "reference_and_eval": len(reference_images & eval_images),
        "note": "same-image references are kept by design; T1 labels are "
                "legitimate prior-task supervision. Any eval overlap would not be.",
    }


def preflight(selection: ref.ReferenceObjects, data_root: Path, out: Path) -> Path:
    """Assert the images exist before a GPU session discovers they do not."""

    jpeg = data_root / "JPEGImages"
    if not jpeg.is_dir():
        raise ref.ExportError(f"{jpeg} missing")
    missing = [name for name in selection.images if not (jpeg / f"{name}.jpg").is_file()]
    if missing:
        raise ref.ExportError(
            f"{len(missing):,} of {len(selection.images):,} T1 reference images are "
            f"absent from {jpeg} (first: {missing[:3]}). Fetch them before exporting."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    probe = out.parent / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    print(f"[preflight] {len(selection):,} reference objects over "
          f"{len(selection.images):,} images; all present; {out.parent} writable")
    return jpeg


def embed(model, selection: ref.ReferenceObjects, jpeg: Path, *, rows: np.ndarray,
          device: str, batch_size: int, label: str) -> dict[str, np.ndarray]:
    """CLS features for the selected GT boxes, through the frozen crop pipeline."""

    import torch
    from PIL import Image
    from torch.nn import functional
    from torchvision.transforms import functional as visual

    by_image: dict[str, list[int]] = {}
    for position in rows:
        by_image.setdefault(str(selection.image_ids[position]), []).append(int(position))
    names = sorted(by_image)

    mean = torch.tensor(sf.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(sf.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out: dict[str, np.ndarray] = {}
    crops: list = []
    keys: list[str] = []
    started = time.time()

    def flush() -> None:
        nonlocal crops, keys
        if not crops:
            return
        batch = torch.stack(crops).to(device, non_blocking=True)
        batch = (batch - mean) / std
        with torch.inference_mode():
            features = model.forward_features(batch)["x_norm_clstoken"]
        features = functional.normalize(features.float(), dim=1)
        block = features.cpu().numpy().astype(np.float16)
        for key, vector in zip(keys, block):
            out[key] = vector
        crops, keys = [], []

    for done, name in enumerate(names, start=1):
        with Image.open(jpeg / f"{name}.jpg") as handle:
            image = handle.convert("RGB")
        width, height = image.size
        for position in by_image[name]:
            cx, cy, w, h = selection.boxes[position]
            x0, y0, x1, y1 = sf.square_crop(cx, cy, w, h, width, height)
            crop = image.crop((x0, y0, x1, y1)).resize(
                (sf.CROP_SIZE, sf.CROP_SIZE), Image.BICUBIC)
            crops.append(visual.pil_to_tensor(crop).float() / 255.0)
            keys.append(str(selection.keys[position]))
            if len(crops) >= batch_size:
                flush()
        if done % 200 == 0 or done == len(names):
            rate = done / max(time.time() - started, 1e-6)
            print(f"[{label}] {done:,}/{len(names):,} images ({rate:.1f} img/s, "
                  f"eta {(len(names) - done) / max(rate, 1e-6) / 60:.1f} min)")
    flush()
    return out


def validate(keys: np.ndarray, features: np.ndarray, class_name: np.ndarray) -> dict:
    """Fail closed on the ways a reference export can be quietly wrong."""

    features = np.asarray(features, dtype=np.float32)
    if np.unique(keys).size != keys.size:
        raise ref.ExportError("duplicate reference identities in the export")
    if features.shape[0] != keys.size:
        raise ref.ExportError(
            f"{features.shape[0]} rows against {keys.size} keys")
    if features.shape[1] != sf.FEATURE_DIM:
        raise ref.ExportError(
            f"dimension {features.shape[1]}, expected {sf.FEATURE_DIM}")
    if not np.isfinite(features).all():
        raise ref.ExportError("non-finite reference features")
    norms = np.linalg.norm(features, axis=1)
    if (norms <= 0).any():
        raise ref.ExportError("zero-norm reference features")
    deviation = float(np.abs(norms - 1.0).max())
    if deviation > sf.NORM_TOLERANCE:
        raise ref.ExportError(
            f"reference is not L2-normalised: worst |‖v‖-1| = {deviation:.2e}")
    classes, counts = np.unique(class_name, return_counts=True)
    if classes.size != 19:
        raise ref.ExportError(
            f"the reference covers {classes.size} classes, expected all 19 T1 "
            "classes; a missing class would make D report that class as novel"
        )
    return {"rows": int(keys.size), "classes": int(classes.size),
            "min_per_class": int(counts.min()), "max_per_class": int(counts.max()),
            "balanced": bool(counts.min() == counts.max()),
            "worst_norm_deviation": deviation}


def write(path: Path, keys, features, class_name, image_ids, object_index,
          provenance) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, keys=np.asarray(keys, dtype=str),
        embeddings=np.asarray(features, dtype=np.float16),
        class_name=np.asarray(class_name, dtype=str),
        image_ids=np.asarray(image_ids, dtype=str),
        object_index=np.asarray(object_index, dtype=np.int64),
        provenance=np.asarray(str(provenance)),
        export_version=np.asarray(REF_EXPORT_VERSION),
    )
    return path


def read(path: str | Path) -> dict:
    payload = np.load(Path(path), allow_pickle=True)
    version = str(payload["export_version"])
    if version != REF_EXPORT_VERSION:
        raise ref.ExportError(
            f"{path} is {version!r}; this code reads {REF_EXPORT_VERSION!r}")
    import ast
    return {
        "keys": np.asarray(payload["keys"], dtype=str),
        "embeddings": payload["embeddings"],
        "class_name": np.asarray(payload["class_name"], dtype=str),
        "image_ids": np.asarray(payload["image_ids"], dtype=str),
        "object_index": np.asarray(payload["object_index"]).astype(np.int64),
        "provenance": ast.literal_eval(str(payload["provenance"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class-cap", type=int, required=True,
                        help=f"objects per T1 class. The frozen primary value is "
                             f"{ref.PRIMARY_REF_T1_CAP_PER_CLASS}; "
                             f"{ref.SENSITIVITY_CAPS} are the predeclared "
                             "descriptive sensitivity subsets and are literal "
                             "per-class prefixes of it")
    parser.add_argument("--pool", default=str(sf.POOL))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-images", type=int, default=0)
    arguments = parser.parse_args()

    import torch

    grouped = ref.enumerate_objects()
    selection = ref.select_balanced(grouped, per_class_cap=arguments.per_class_cap)
    print(f"[selection] {selection.summary()}")
    print(f"[selection] manifest SHA256 {selection.provenance['manifest_sha256']}")
    if arguments.per_class_cap == ref.PRIMARY_REF_T1_CAP_PER_CLASS:
        print("[selection] this is the FROZEN PRIMARY reference "
              f"({ref.PRIMARY_REF_T1_CAP_PER_CLASS}/class)")
    elif arguments.per_class_cap in ref.SENSITIVITY_CAPS:
        print(f"[selection] this is a predeclared DESCRIPTIVE SENSITIVITY subset "
              f"({arguments.per_class_cap}/class); it cannot carry a verdict")
    else:
        raise ref.ExportError(
            f"--per-class-cap {arguments.per_class_cap} is neither the frozen "
            f"primary ({ref.PRIMARY_REF_T1_CAP_PER_CLASS}) nor a predeclared "
            f"sensitivity subset {ref.SENSITIVITY_CAPS}. Choosing a cap outside "
            "the pre-registration is how a reference size gets tuned; refusing."
        )
    overlap = candidate_overlap(selection, Path(arguments.pool))
    print(f"[overlap] {overlap['reference_and_candidate_pool']:,} reference images "
          f"also in the candidate pool (kept by design); "
          f"{overlap['reference_and_eval']:,} in eval")
    if overlap["reference_and_eval"]:
        raise ref.ExportError(
            f"{overlap['reference_and_eval']} reference images appear in the eval "
            "split. That would be leakage, unlike the candidate-pool overlap."
        )

    out = Path(arguments.out)
    jpeg = preflight(selection, Path(arguments.data_root), out)

    if out.exists() and not arguments.smoke_images:
        print(f"[resume] {out} exists; verifying instead of re-exporting")
        payload = read(out)
        print(f"[resume] {validate(payload['keys'], payload['embeddings'], payload['class_name'])}")
        return

    device = arguments.device if torch.cuda.is_available() else "cpu"
    model = load_backbone(device)

    if arguments.smoke_images:
        images = selection.images[:arguments.smoke_images]
        rows = np.asarray(
            [i for i in range(len(selection))
             if str(selection.image_ids[i]) in set(images)], dtype=np.int64)
        produced = embed(model, selection, jpeg, rows=rows, device=device,
                         batch_size=arguments.batch_size, label="smoke")
        keys = np.asarray([str(selection.keys[i]) for i in rows])
        features = np.stack([produced[k] for k in keys])
        norms = np.linalg.norm(features.astype(np.float32), axis=1)
        print(f"[smoke] {keys.size} reference crops, dim {features.shape[1]}, "
              f"worst |‖v‖-1| = {float(np.abs(norms - 1).max()):.2e}  PASS")
        print("[smoke] the full reference export is safe to run")
        return

    started = time.time()
    rows = np.arange(len(selection), dtype=np.int64)
    produced = embed(model, selection, jpeg, rows=rows, device=device,
                     batch_size=arguments.batch_size, label="ref-t1")
    keys = np.asarray([str(k) for k in selection.keys])
    missing = [k for k in keys.tolist() if k not in produced]
    if missing:
        raise ref.ExportError(f"{len(missing)} reference objects produced no feature")
    features = np.stack([produced[k] for k in keys])

    provenance = {
        "export_version": REF_EXPORT_VERSION,
        "reference": selection.provenance,
        "git_sha": git_sha(),
        "model_id": sf.MODEL_ID,
        "model_source": sf.HUB_REPO,
        "crop": sf.crop_specification(),
        "image_root": str(jpeg),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "objects": int(keys.size),
        "images": len(selection.images),
        "feature_dim": sf.FEATURE_DIM,
        "candidate_overlap": overlap,
        "is_frozen_primary": arguments.per_class_cap == ref.PRIMARY_REF_T1_CAP_PER_CLASS,
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    report = validate(keys, features, selection.class_name)
    write(out, keys, features, selection.class_name, selection.image_ids,
          selection.object_index, provenance)
    Path(str(out) + ".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\n[gate] {report}  PASS")
    print(f"[done] {out}  in {provenance['wall_clock_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
