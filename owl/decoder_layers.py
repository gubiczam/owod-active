"""The decoder-layer export contract: what is in the file, and what makes it valid.

One concept: a multi-layer view of the *same* candidate pool. The committed pool
(`data/pool/sowodb_t1_frozen_pool.npz`) carries PROB's final decoder layer,
``hs[5]``. This module defines a companion export carrying ``hs[0] … hs[5]`` for
exactly the same proposals, so that "which decoder layer" can be varied while
everything else is held fixed.

**Why identity alignment is exact rather than reconstructed.** The pool stores
``query_index`` per proposal, so a proposal is keyed by
``(image_id, query_index)``. The exporter selects those keys directly instead of
re-deriving the pool's top-50-by-objectness ranking, which removes any chance of
the two files disagreeing about which proposal a row is.

**Why the validation gate is a single assertion.** ``hs[5]`` in a correct export
must reproduce the pool's own ``embeddings``. That one check simultaneously
validates the checkpoint, the reconstructed model arguments, the forward hooks,
the image ordering, and the key join. If it fails, nothing downstream is
trustworthy, so :func:`validate` refuses rather than warns.

Nothing here touches a GPU or PROB; that is ``tools/export_decoder_layers.py``.
This module is the schema, the key algebra, and the gate, so they can be tested
on a laptop.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Bump when the file's meaning changes. Never overwrite an older version's file.
EXPORT_VERSION = "decoder_layers_v1"

#: PROB's decoder depth (`--dec_layers` default 6). Layer 5 is what the pool holds.
N_DECODER_LAYERS = 6
FINAL_LAYER = N_DECODER_LAYERS - 1

#: Mean cosine similarity that ``hs[5]`` must reach against the pool's embeddings.
#: The pool stores float16, so exact equality is not available; 0.999 is far above
#: what a wrong layer, a wrong checkpoint, or a misaligned join could reach by
#: accident, and far below float16 round-trip noise.
VALIDATION_SIMILARITY = 0.999


class ExportError(RuntimeError):
    """Raised when an export is missing, misaligned, or fails its own gate."""


def sha256(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def proposal_keys(image_ids: np.ndarray, query_index: np.ndarray) -> np.ndarray:
    """``(image_id, query_index)`` as one sortable string key per proposal.

    A string key rather than a tuple array because it has to survive a round trip
    through ``.npz`` and be usable with :func:`numpy.isin` and
    :func:`numpy.searchsorted` without object dtype.
    """

    images = np.asarray(image_ids, dtype=str)
    queries = np.asarray(query_index).astype(np.int64)
    return np.char.add(np.char.add(images, "#"), queries.astype(str))


def align(source_keys: np.ndarray, target_keys: np.ndarray) -> np.ndarray:
    """Positions in ``source_keys`` for every key in ``target_keys``, in order.

    Raises rather than dropping: a missing key means the export does not cover the
    pool, and silently auditing a subset would compare layers on different
    candidates.
    """

    order = np.argsort(source_keys, kind="mergesort")
    sorted_keys = source_keys[order]
    position = np.searchsorted(sorted_keys, target_keys)
    position = np.clip(position, 0, sorted_keys.size - 1)
    found = sorted_keys[position] == target_keys
    if not found.all():
        missing = int((~found).sum())
        raise ExportError(
            f"{missing} of {target_keys.size} pool proposals are absent from the "
            "export. The export does not cover the audited pool, so layers would "
            "be compared on different candidates."
        )
    return order[position]


@dataclass(frozen=True)
class LayerExport:
    """``hs[0..5]`` for the pool's proposals, in the pool's own row order."""

    features: np.ndarray          # (n_layers, N, 256) float16
    keys: np.ndarray              # (N,) str, aligned to the pool's rows
    layer_indices: tuple[int, ...]
    provenance: dict

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ExportError(f"features must be (layers, N, dim); got {self.features.shape}")
        if self.features.shape[0] != len(self.layer_indices):
            raise ExportError(
                f"{self.features.shape[0]} feature blocks against "
                f"{len(self.layer_indices)} declared layer indices"
            )
        if self.features.shape[1] != self.keys.size:
            raise ExportError(
                f"{self.features.shape[1]} rows against {self.keys.size} keys"
            )

    def layer(self, index: int) -> np.ndarray:
        """``hs[index]`` as float32, in the pool's row order."""

        if index not in self.layer_indices:
            raise ExportError(
                f"layer {index} is not in this export ({self.layer_indices})"
            )
        return self.features[self.layer_indices.index(index)].astype(np.float32)


def write(
    path: str | Path,
    features: np.ndarray,
    keys: np.ndarray,
    layer_indices: tuple[int, ...],
    provenance: dict,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        features=np.asarray(features, dtype=np.float16),
        keys=np.asarray(keys, dtype=str),
        layer_indices=np.asarray(layer_indices, dtype=np.int64),
        provenance=np.asarray(str(provenance)),
        export_version=np.asarray(EXPORT_VERSION),
    )
    return path


def read(path: str | Path) -> LayerExport:
    payload = np.load(Path(path), allow_pickle=True)
    version = str(payload["export_version"])
    if version != EXPORT_VERSION:
        raise ExportError(
            f"{path} is {version!r}; this code reads {EXPORT_VERSION!r}. Older "
            "exports are not reinterpreted under new semantics."
        )
    import ast

    return LayerExport(
        features=payload["features"],
        keys=np.asarray(payload["keys"], dtype=str),
        layer_indices=tuple(int(value) for value in payload["layer_indices"]),
        provenance=ast.literal_eval(str(payload["provenance"])),
    )


def validate(export: LayerExport, pool_embeddings: np.ndarray) -> float:
    """Gate: ``hs[5]`` must reproduce the pool's committed embeddings.

    Returns the mean cosine similarity. Raises when it is below
    :data:`VALIDATION_SIMILARITY`, because every downstream number would then be
    describing an unknown tensor.
    """

    if FINAL_LAYER not in export.layer_indices:
        raise ExportError(
            f"the export omits layer {FINAL_LAYER}, so it cannot be checked "
            "against the pool. Export it even if it is not being audited."
        )
    final = export.layer(FINAL_LAYER)
    reference = np.asarray(pool_embeddings, dtype=np.float32)
    if final.shape != reference.shape:
        raise ExportError(
            f"layer {FINAL_LAYER} is {final.shape}; the pool's embeddings are "
            f"{reference.shape}"
        )

    def unit(matrix: np.ndarray) -> np.ndarray:
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)

    similarity = float((unit(final) * unit(reference)).sum(axis=1).mean())
    if similarity < VALIDATION_SIMILARITY:
        raise ExportError(
            f"layer {FINAL_LAYER} reproduces the pool's embeddings at mean cosine "
            f"{similarity:.6f}, below the required {VALIDATION_SIMILARITY}. The "
            "checkpoint, the reconstructed model arguments, the hooks, the image "
            "order or the key join is wrong. Refusing to audit."
        )
    return similarity
