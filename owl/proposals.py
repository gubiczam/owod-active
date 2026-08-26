"""The candidate pool: what the detector proposes, before anyone is asked.

One concept: a bag of candidate regions with detector-derived fields only.
The oracle fields live in the same object but are kept behind
:meth:`Candidates.oracle`, so a scoring function that reads a label has to say
so out loud. Nothing in :mod:`owl.scoring` touches them.

Two sources produce the same object:

* :func:`from_predict` — a live PROB ``predict`` export, the GPU path;
* :func:`from_frozen_pool` — one committed PROB pass over 2,400 fixed images,
  the CPU path used for development and for the cheap arm sweeps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

BACKGROUND = "background"


@dataclass(frozen=True)
class Oracle:
    """What a human annotator would answer. Never an input to selection.

    ``kind`` is one of ``known`` / ``unknown`` / ``background``, relative to the
    checkpoint that produced the proposals. ``object_id`` identifies the
    annotated object a proposal was matched to, so two proposals on the same
    object cost the annotator once.
    """

    kind: np.ndarray        # (N,) str
    class_name: np.ndarray  # (N,) str, '' for background
    object_id: np.ndarray   # (N,) int, -1 for background
    iou: np.ndarray         # (N,) float

    def __len__(self) -> int:
        return int(self.kind.shape[0])


@dataclass(frozen=True)
class Candidates:
    """Detector proposals over a set of images.

    ``boxes`` are PROB's own normalised ``cxcywh``. ``embeddings`` are the
    decoder outputs, L2-normalised on construction so every cosine distance in
    the codebase means the same thing. ``posterior`` rows sum to one.
    """

    image_ids: np.ndarray    # (N,) str
    boxes: np.ndarray        # (N, 4) float32, normalised cxcywh
    embeddings: np.ndarray   # (N, D) float32, unit norm
    posterior: np.ndarray    # (N, K) float32, rows sum to 1
    objectness: np.ndarray   # (N,) float32, PROB's own objectness in [0, 1]
    _oracle: Oracle | None = None
    meta: dict | None = None

    # ------------------------------------------------------------ basics ---

    def __len__(self) -> int:
        return int(self.image_ids.shape[0])

    @property
    def n_images(self) -> int:
        return int(np.unique(self.image_ids).size)

    @property
    def area(self) -> np.ndarray:
        """Normalised box area, used by the learning-free object-likeness prior."""
        return self.boxes[:, 2] * self.boxes[:, 3]

    def oracle(self) -> Oracle:
        """The answers. Calling this is the explicit admission that a label is read."""
        if self._oracle is None:
            raise ValueError(
                "This pool carries no oracle. Live GPU pools only get answers "
                "after the annotator has been paid; use owl.labelling."
            )
        return self._oracle

    @property
    def has_oracle(self) -> bool:
        return self._oracle is not None

    def take(self, index: np.ndarray) -> Candidates:
        """A sub-pool. ``index`` may be a boolean mask or an integer array."""
        oracle = None
        if self._oracle is not None:
            oracle = Oracle(
                kind=self._oracle.kind[index],
                class_name=self._oracle.class_name[index],
                object_id=self._oracle.object_id[index],
                iou=self._oracle.iou[index],
            )
        return replace(
            self,
            image_ids=self.image_ids[index],
            boxes=self.boxes[index],
            embeddings=self.embeddings[index],
            posterior=self.posterior[index],
            objectness=self.objectness[index],
            _oracle=oracle,
        )

    def describe(self) -> dict[str, object]:
        row: dict[str, object] = {
            "proposals": len(self),
            "images": self.n_images,
            "feature_dim": int(self.embeddings.shape[1]),
            "posterior_dim": int(self.posterior.shape[1]),
        }
        if self.has_oracle:
            kinds, counts = np.unique(self._oracle.kind, return_counts=True)
            row.update({f"oracle_{k}": int(c) for k, c in zip(kinds, counts)})
        return row


# ------------------------------------------------------------ constructors ---


def _unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norm, 1e-9)


def _rows_to_one(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)


def from_predict(path: str | Path) -> Candidates:
    """Read a PROB ``daowod_prob_bridge.py predict`` export.

    No oracle: on the GPU path the answers arrive later, one annotated image at
    a time, through :mod:`owl.labelling`.
    """

    path = Path(path)
    payload = np.load(path, allow_pickle=True)
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    return Candidates(
        image_ids=np.asarray(payload["image_ids"], dtype=str),
        boxes=np.asarray(payload["boxes"], dtype=np.float32),
        embeddings=_unit(payload["embeddings"]),
        posterior=_rows_to_one(payload["posterior"]),
        objectness=np.asarray(payload["objectness"], dtype=np.float32),
        meta=meta,
    )


FROZEN_POOL = Path(__file__).resolve().parent.parent / "data" / "pool" / "sowodb_t1_frozen_pool.npz"


def from_frozen_pool(
    path: str | Path = FROZEN_POOL,
    *,
    split: str = "pool",
) -> Candidates:
    """Read the committed PROB pass. ``split`` is ``pool`` or ``eval``.

    This file is one real forward pass of ``exps/SOWODB/PROB/t1.pth`` over 2,400
    benchmark images, with every proposal matched to the benchmark's own
    annotation at IoU 0.5. It carries an oracle, which is what lets the whole
    annotation cycle run on a laptop.
    """

    payload = np.load(Path(path), allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == split
    if not keep.any():
        raise ValueError(f"No proposals with split={split!r} in {path}.")

    posterior = payload["posterior_q"][keep].astype(np.float32) / 255.0
    oracle = Oracle(
        kind=np.asarray(payload["oracle_kind"], dtype=str)[keep],
        class_name=np.asarray(payload["oracle_class"], dtype=str)[keep],
        object_id=np.asarray(payload["oracle_object"], dtype=np.int64)[keep],
        iou=np.asarray(payload["oracle_iou"], dtype=np.float32)[keep],
    )
    meta = json.loads(str(payload["meta"]))
    meta["split"] = split
    return Candidates(
        image_ids=np.asarray(payload["image_ids"], dtype=str)[keep],
        boxes=np.asarray(payload["boxes"], dtype=np.float32)[keep],
        embeddings=_unit(payload["embeddings"][keep]),
        posterior=_rows_to_one(posterior),
        objectness=np.asarray(payload["confidence"], dtype=np.float32)[keep],
        _oracle=oracle,
        meta=meta,
    )


# ------------------------------------------------------------------ helpers ---


def image_index(candidates: Candidates) -> tuple[np.ndarray, np.ndarray]:
    """``(unique_image_ids, position)`` where ``position[i]`` indexes the image
    proposal ``i`` belongs to. Every image-level aggregation goes through this.
    """

    unique, position = np.unique(candidates.image_ids, return_inverse=True)
    return unique, position
