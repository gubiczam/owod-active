"""The shared candidate population: deduplicate first, gate second.

Every arm of Benchmark V1 buys from the same population, because a comparison
between selectors that see different candidates is a comparison of populations.
Two stages, both oracle-free, both computed from detector output only:

``P_nms``
    per-image non-maximum suppression at IoU 0.60, ordered by ``A(x)``. This is
    deduplication, not selection: two proposals on one object cost the annotator
    once, and the earlier work measured per-image NMS to be clearly useful. It
    applies to **all** arms.
``G``
    the top ``ADMISSIBLE_SHARE`` of ``P_nms`` by ``A(x) = objectness * sqrt(area)``.
    This is the *gate*, and it applies only to the arms that declare it — the
    proposed method uses it, ``coreset`` deliberately does not, and that one
    difference is the ablation.

**Why the order differs from the committed recipe, and how that is checked.**
The established population from Method V2/V3 is ``P2 = gate then NMS`` — 15,518
rows on the frozen pool. Benchmark V1 needs one population that an *ungated* arm
can also select from, so it deduplicates first and gates second.
:func:`p2_reference` reproduces the committed order in this module's own code,
and ``tests/test_active_selection.py`` asserts it lands on 15,518 rows with a
0.767 background share. So the reordering is a documented choice measured
against the original, not a drifted reimplementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from owl import clustering, scoring
from owl.proposals import Candidates

#: Per-image NMS threshold. Frozen from the established P2 recipe
#: (``tools/audit_decoder_layers.NMS_IOU``); not swept here.
NMS_IOU = 0.60

#: The admissibility gate's share, frozen from the same recipe
#: (``OBJECTNESS_SHARE``). Not tuned against any endpoint in this benchmark.
ADMISSIBLE_SHARE = 0.30


@dataclass(frozen=True)
class Population:
    """One task's candidate population and the masks that describe it."""

    candidates: Candidates       # the pool restricted to P_nms
    admissibility: np.ndarray    # (n,) A(x) on that restriction, raw
    gate: np.ndarray             # (n,) bool, the admissible subset G
    kept: np.ndarray             # (N,) bool, P_nms over the incoming pool
    diagnostics: dict

    def __len__(self) -> int:
        return len(self.candidates)


def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU of normalised ``cxcywh`` boxes.

    Same algebra as ``tools/diagnose_population._iou_matrix``, on the box layout
    :class:`owl.proposals.Candidates` carries rather than on the frozen pool's
    dict, so one implementation serves the live path.
    """

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 0))
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    area = np.maximum(w, 0.0) * np.maximum(h, 0.0)
    left = np.maximum(x1[:, None], x1[None, :])
    top = np.maximum(y1[:, None], y1[None, :])
    right = np.minimum(x2[:, None], x2[None, :])
    bottom = np.minimum(y2[:, None], y2[None, :])
    overlap = np.maximum(right - left, 0.0) * np.maximum(bottom - top, 0.0)
    union = area[:, None] + area[None, :] - overlap
    return overlap / np.maximum(union, 1e-9)


def per_image_nms(
    boxes: np.ndarray,
    image_ids: np.ndarray,
    order_key: np.ndarray,
    *,
    candidate: np.ndarray | None = None,
    iou_threshold: float = NMS_IOU,
) -> np.ndarray:
    """Keep the highest-``order_key`` box of each overlapping group, per image.

    ``candidate`` restricts the input; positions outside it are never kept, which
    is what lets the committed "gate then NMS" order be reproduced by the same
    function that Benchmark V1 calls with no restriction at all.
    """

    boxes = np.asarray(boxes, dtype=np.float64)
    image_ids = np.asarray(image_ids)
    order_key = np.asarray(order_key, dtype=np.float64)
    n = image_ids.shape[0]
    candidate = np.ones(n, dtype=bool) if candidate is None else np.asarray(candidate, dtype=bool)

    keep = np.zeros(n, dtype=bool)
    for image in np.unique(image_ids[candidate]):
        local = np.flatnonzero(candidate & (image_ids == image))
        if local.size == 0:
            continue
        local = local[np.argsort(-order_key[local], kind="stable")]
        iou = iou_matrix(boxes[local])
        alive = np.ones(local.size, dtype=bool)
        positions = np.arange(local.size)
        for position in positions:
            if not alive[position]:
                continue
            keep[local[position]] = True
            alive &= ~((iou[position] > iou_threshold) & (positions > position))
    return keep


def p2_reference(
    candidates: Candidates,
    *,
    share: float = ADMISSIBLE_SHARE,
    iou_threshold: float = NMS_IOU,
) -> np.ndarray:
    """The **committed** population order: gate, then NMS. For the pin test only.

    Benchmark V1 does not select from this — see the module docstring — but the
    number it produces is what proves this module's NMS and gate agree with the
    implementation Method V2 and V3 were measured on.
    """

    admissibility = scoring.admissibility(candidates)
    admitted = clustering.admissible_mask(admissibility, share)
    return per_image_nms(
        candidates.boxes, candidates.image_ids, admissibility,
        candidate=admitted, iou_threshold=iou_threshold,
    )


def build(
    candidates: Candidates,
    *,
    share: float = ADMISSIBLE_SHARE,
    iou_threshold: float = NMS_IOU,
) -> Population:
    """``P_nms`` and its admissible subset ``G``, for one task's fresh pool."""

    admissibility = scoring.admissibility(candidates)
    kept = per_image_nms(
        candidates.boxes, candidates.image_ids, admissibility,
        iou_threshold=iou_threshold,
    )
    index = np.flatnonzero(kept)
    restricted = candidates.take(index)
    inside = admissibility[index]
    gate = clustering.admissible_mask(inside, share)
    return Population(
        candidates=restricted,
        admissibility=inside,
        gate=gate,
        kept=kept,
        diagnostics={
            "proposals_in": len(candidates),
            "proposals_after_nms": int(index.size),
            "nms_survival": round(float(index.size) / max(len(candidates), 1), 4),
            "images": int(np.unique(restricted.image_ids).size),
            "admissible": int(gate.sum()),
            "admissible_share": share,
            "nms_iou": iou_threshold,
        },
    )
