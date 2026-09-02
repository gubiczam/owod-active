"""The semantic export contract: crop geometry, the file's schema, and its gates.

Method V2 asks whether the *representation* was the binding constraint. PROB's
final decoder embedding is objectness-dominated by construction --
``pred_obj = ||BatchNorm(hs[lvl])||^2``, a class-agnostic collapse toward one
point, applied at every decoder layer -- so a frozen, detector-independent
semantic backbone answers the question directly.

This module holds everything about that export which can be tested on a laptop:
the crop algebra, the schema, and the validation. The GPU half is
``tools/export_dinov2_features.py``.

**Why the crop has no padding.** A grey square around a proposal near an image
edge is a constant synthetic region whose *area varies with how close the proposal
sits to the border*. That is an edge-proximity signal written straight into the
embedding, and it would be indistinguishable from semantics in every metric the
audit computes. So the square is shifted to fit before it is shrunk, and shrunk
only when the image itself cannot supply it. Every crop is real pixels at the
requested scale, or the largest real square the image allows.

Identity is shared with the decoder-layer export rather than reinvented: rows key
on ``(image_id, query_index)`` through :func:`owl.decoder_layers.proposal_keys`,
so the two exports cannot disagree about what a row is.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from owl.decoder_layers import ExportError, align, proposal_keys, sha256

#: Bump when the file's meaning changes. Never overwrite an older version's file.
EXPORT_VERSION = "dinov2_vitb14_method_v2_v1"

#: The frozen backbone. Not swept, not compared against other sizes or CLIP.
MODEL_ID = "dinov2_vitb14"
HUB_REPO = "facebookresearch/dinov2"
FEATURE_DIM = 768

#: The frozen crop, from docs/method_v2_protocol_2026-09-02.md section 2.
CROP_MARGIN = 1.20        # square side = margin * max(proposal_w, proposal_h)
CROP_SIZE = 224           # 224 / 14 = 16 patches, a valid ViT-B/14 input
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: The committed candidate pool. 'eval' is never loaded, never fitted on, never
#: included -- only ``split == "pool"``.
POOL = Path(__file__).resolve().parent.parent / "data" / "pool" / "sowodb_t1_frozen_pool.npz"
POOL_SPLIT = "pool"
EXPECTED_PROPOSALS = 80_000
EXPECTED_IMAGES = 1_600

#: Post-normalisation L2 norms must sit this close to 1.
NORM_TOLERANCE = 1e-3


def crop_specification() -> dict:
    """The frozen crop, as data, so provenance records it rather than describing it."""

    return {
        "margin": CROP_MARGIN,
        "shape": "square, centred on the proposal centre",
        "side_rule": "margin * max(proposal_width_px, proposal_height_px)",
        "boundary_rule": "shift to fit, preserving size; shrink only if the image "
                         "cannot supply the square",
        "padding": "none -- real image pixels only",
        "resize": f"{CROP_SIZE}x{CROP_SIZE} bicubic",
        "normalisation": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "feature": "final normed CLS token (forward_features -> x_norm_clstoken)",
        "postprocess": "L2-normalised",
    }


# ------------------------------------------------------------ crop geometry ---


def square_crop(
    cx: float, cy: float, w: float, h: float,
    width: int, height: int, *, margin: float = CROP_MARGIN,
) -> tuple[int, int, int, int]:
    """Normalised ``cxcywh`` -> an integer pixel square ``(x0, y0, x1, y1)``.

    ``cx, cy, w, h`` are fractions of ``width``/``height``. The returned box is
    exactly square, lies wholly inside the image, and is never padded.

    Order of operations matters and is fixed by the protocol: the requested side
    is capped at what the image can supply, then the box is *shifted* into range.
    Clamping the corners instead would silently produce a non-square crop, and
    padding would inject an edge-proximity signal.
    """

    if width <= 0 or height <= 0:
        raise ExportError(f"image dimensions must be positive; got {width}x{height}")

    side = margin * max(abs(w) * width, abs(h) * height)
    # the largest square this image can supply, and at least one pixel
    side = min(side, float(width), float(height))
    size = max(1, min(round(side), width, height))

    # centre in pixels, then shift the square wholly inside the image
    x0 = round(cx * width - size / 2.0)
    y0 = round(cy * height - size / 2.0)
    x0 = min(max(x0, 0), width - size)
    y0 = min(max(y0, 0), height - size)
    return x0, y0, x0 + size, y0 + size


# ---------------------------------------------------------------- the pool ---


@dataclass(frozen=True)
class PoolRows:
    """The pool split's proposal identities, in the pool file's own row order."""

    image_ids: np.ndarray      # (N,) str
    query_index: np.ndarray    # (N,) int64
    keys: np.ndarray           # (N,) str, "image#query"
    row_index: np.ndarray      # (N,) int64, position in the *unfiltered* npz
    boxes: np.ndarray          # (N, 4) float32 normalised cxcywh

    def __len__(self) -> int:
        return int(self.keys.size)

    @property
    def images(self) -> list[str]:
        return sorted(set(self.image_ids.tolist()))


def pool_rows(path: str | Path) -> PoolRows:
    """Read ``split == 'pool'`` only, and refuse anything else.

    The row index is kept because it is the link back to the unfiltered file: an
    audit that re-derives the subset by filtering again could pick up a different
    order if the file were ever rewritten, whereas a stored index cannot drift.
    """

    payload = np.load(Path(path), allow_pickle=True)
    splits = np.asarray(payload["split"], dtype=str)
    keep = splits == POOL_SPLIT
    if not keep.any():
        raise ExportError(f"no rows with split == {POOL_SPLIT!r} in {path}")

    image_ids = np.asarray(payload["image_ids"], dtype=str)[keep]
    query_index = np.asarray(payload["query_index"])[keep].astype(np.int64)
    keys = proposal_keys(image_ids, query_index)

    if keys.size != EXPECTED_PROPOSALS:
        raise ExportError(
            f"{path} holds {keys.size} pool proposals, expected "
            f"{EXPECTED_PROPOSALS}. The population is not the frozen one."
        )
    distinct_images = np.unique(image_ids).size
    if distinct_images != EXPECTED_IMAGES:
        raise ExportError(
            f"{path} covers {distinct_images} pool images, expected {EXPECTED_IMAGES}"
        )
    if np.unique(keys).size != keys.size:
        duplicates = keys.size - np.unique(keys).size
        raise ExportError(
            f"{duplicates} duplicate (image_id, query_index) identities in the pool "
            "split; a duplicated identity makes the alignment ambiguous."
        )

    return PoolRows(
        image_ids=image_ids,
        query_index=query_index,
        keys=keys,
        row_index=np.flatnonzero(keep).astype(np.int64),
        boxes=payload["boxes"][keep].astype(np.float32),
    )


# --------------------------------------------------------------- the export ---


@dataclass(frozen=True)
class SemanticExport:
    """Frozen-backbone embeddings for the pool's proposals, in pool row order."""

    embeddings: np.ndarray     # (N, FEATURE_DIM) float16 on disk, float32 in use
    keys: np.ndarray           # (N,) str
    image_ids: np.ndarray      # (N,) str
    query_index: np.ndarray    # (N,) int64
    row_index: np.ndarray      # (N,) int64, position in the unfiltered pool npz
    provenance: dict

    def __post_init__(self) -> None:
        n = self.keys.size
        for name in ("embeddings", "image_ids", "query_index", "row_index"):
            value = getattr(self, name)
            if value.shape[0] != n:
                raise ExportError(
                    f"{name} has {value.shape[0]} rows against {n} keys"
                )
        if self.embeddings.ndim != 2:
            raise ExportError(f"embeddings must be 2-d; got {self.embeddings.shape}")

    def features(self) -> np.ndarray:
        """Embeddings as float32. Stored small, evaluated in full precision."""

        return self.embeddings.astype(np.float32)


def write(path: str | Path, export: SemanticExport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=np.asarray(export.embeddings, dtype=np.float16),
        keys=np.asarray(export.keys, dtype=str),
        image_ids=np.asarray(export.image_ids, dtype=str),
        query_index=np.asarray(export.query_index, dtype=np.int64),
        row_index=np.asarray(export.row_index, dtype=np.int64),
        provenance=np.asarray(str(export.provenance)),
        export_version=np.asarray(EXPORT_VERSION),
    )
    return path


def read(path: str | Path) -> SemanticExport:
    payload = np.load(Path(path), allow_pickle=True)
    version = str(payload["export_version"])
    if version != EXPORT_VERSION:
        raise ExportError(
            f"{path} is {version!r}; this code reads {EXPORT_VERSION!r}. Older "
            "exports are not reinterpreted under new semantics."
        )
    return SemanticExport(
        embeddings=payload["embeddings"],
        keys=np.asarray(payload["keys"], dtype=str),
        image_ids=np.asarray(payload["image_ids"], dtype=str),
        query_index=np.asarray(payload["query_index"]).astype(np.int64),
        row_index=np.asarray(payload["row_index"]).astype(np.int64),
        provenance=ast.literal_eval(str(payload["provenance"])),
    )


def validate(export: SemanticExport, rows: PoolRows, *, full: bool = True) -> dict:
    """Refuse an export that is misaligned, degenerate, or the wrong population.

    ``full=False`` relaxes only the population-size checks, for the smoke subset;
    every correctness check still applies. Nothing here warns -- a bad export
    produces numbers that look entirely plausible, so it has to raise.
    """

    features = export.features()

    if features.shape[1] != FEATURE_DIM:
        raise ExportError(
            f"feature dimension {features.shape[1]}, expected {FEATURE_DIM} for "
            f"{MODEL_ID}"
        )
    if np.unique(export.keys).size != export.keys.size:
        raise ExportError("duplicate proposal identities in the export")
    if not np.isfinite(features).all():
        bad = int((~np.isfinite(features)).any(axis=1).sum())
        raise ExportError(f"{bad} rows hold non-finite features")

    norms = np.linalg.norm(features, axis=1)
    if (norms <= 0).any():
        raise ExportError(f"{int((norms <= 0).sum())} rows have zero-norm features")
    deviation = float(np.abs(norms - 1.0).max())
    if deviation > NORM_TOLERANCE:
        raise ExportError(
            f"embeddings are not L2-normalised: worst |‖v‖ - 1| = {deviation:.2e} "
            f"exceeds {NORM_TOLERANCE:.0e}"
        )

    # alignment: every key the export claims must be a pool key, in pool order
    expected = rows.keys if full else rows.keys[np.isin(rows.keys, export.keys)]
    if export.keys.size != expected.size:
        raise ExportError(
            f"the export holds {export.keys.size} rows against {expected.size} "
            "pool rows for this population"
        )
    if not np.array_equal(export.keys, expected):
        align(export.keys, expected)          # raises with the missing count
        raise ExportError(
            "the export's rows are not in the pool's own order; an audit joining "
            "by position would silently compare different proposals."
        )
    if full and export.keys.size != EXPECTED_PROPOSALS:
        raise ExportError(
            f"a full export must hold {EXPECTED_PROPOSALS} rows; got {export.keys.size}"
        )

    return {
        "rows": int(export.keys.size),
        "images": int(np.unique(export.image_ids).size),
        "dimension": int(features.shape[1]),
        "worst_norm_deviation": deviation,
    }


__all__ = [
    "CROP_MARGIN", "CROP_SIZE", "EXPECTED_IMAGES", "EXPECTED_PROPOSALS",
    "EXPORT_VERSION", "FEATURE_DIM", "HUB_REPO", "IMAGENET_MEAN", "IMAGENET_STD",
    "MODEL_ID", "POOL", "POOL_SPLIT", "ExportError", "PoolRows", "SemanticExport",
    "crop_specification", "pool_rows", "read", "sha256", "square_crop",
    "validate", "write",
]
