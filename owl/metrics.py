"""Reading PROB's official evaluator, and the head/medium/tail resolution on top.

The parser is deliberately unchanged from the earlier work: it is the code that
already reads real ``OWEvaluator.summarize`` logs from the GPU runs kept in
``data/reference/measured/``, so a rewrite would trade working code for risk.

What the task chain adds on top is :func:`task_row` — one row per task, holding
the three numbers the consultation asked to see side by side: how much was
forgotten, how much was learned, and whether unknowns are still being found.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GROUPS: tuple[str, ...] = ("head", "medium", "tail")


class MetricsError(ValueError):
    """Raised when the evaluator's output cannot be read as a complete result."""


@dataclass(frozen=True)
class Evaluation:
    """One official evaluation, parsed. Field names follow the proposal's metrics."""

    #: mAP50 over every currently known class. The proposal's "ismert mAP".
    known_map50: float
    #: Recall50 on the pooled unknown category. The proposal's "U-Recall".
    unknown_recall50: float
    #: Wilderness Impact, keyed by recall level and then by IoU threshold — the
    wilderness_impact: Mapping[float, Mapping[float, float]]
    #: Absolute Open-Set Error, keyed by IoU threshold.
    absolute_ose: Mapping[float, float]
    #: AP50 for each class by name, in the evaluator's own class order.
    per_class_ap50: Mapping[str, float]
    #: mAP50 restricted to classes introduced in earlier tasks; None at task 1.
    previous_map50: float | None = None
    #: mAP50 restricted to classes introduced by the current task.
    current_map50: float | None = None
    unknown_ap50: float | None = None
    raw: str = field(default="", repr=False)

    def wi_at(self, recall_level: float = 0.8, iou: float = 50.0) -> float:
        """WI is conventionally quoted at recall 0.8 and IoU 50; say which."""
        if recall_level not in self.wilderness_impact:
            raise MetricsError(
                f"Wilderness Impact was not reported at recall {recall_level}; "
                f"available: {sorted(self.wilderness_impact)}"
            )
        at_recall = self.wilderness_impact[recall_level]
        if iou not in at_recall:
            raise MetricsError(
                f"Wilderness Impact at recall {recall_level} was not reported at IoU "
                f"{iou}; available: {sorted(at_recall)}"
            )
        return at_recall[iou]

    def aose_at(self, iou: float = 50.0) -> float:
        if iou not in self.absolute_ose:
            raise MetricsError(
                f"A-OSE was not reported at IoU {iou}; available: "
                f"{sorted(self.absolute_ose)}"
            )
        return self.absolute_ose[iou]


_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

#: ``summarize`` prints torch tensors and numpy scalars through ``str``/``repr``, so
_WRAPPED = re.compile(r"(?:np\.\w+|tensor|array)\(\s*([^(),]*)[^()]*\)")


def _unwrap(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _WRAPPED.sub(r"\1", text)
    return text


def _search_scalar(text: str, label: str) -> float | None:
    match = re.search(rf"^{re.escape(label)}:?\s*(\S.*?)\s*$", text, re.MULTILINE)
    if not match:
        return None
    try:
        return float(_unwrap(match.group(1)))
    except ValueError:
        return None


def _search_nested_mapping(text: str, label: str) -> dict[float, dict[float, float]]:
    """Parse ``{0.1: {50: 0.03}, 0.2: {50: 0.05}}`` — recall level, then IoU."""

    parsed = _search_literal(text, label)
    if parsed is None:
        return {}
    result: dict[float, dict[float, float]] = {}
    for key, value in parsed.items():
        if not isinstance(value, dict):
            raise MetricsError(
                f"{label} was printed as a flat mapping; the evaluator nests it by "
                "IoU, so this log was produced by different code."
            )
        result[float(key)] = {float(inner): float(number) for inner, number in value.items()}
    return result


def _search_mapping(text: str, label: str) -> dict[float, float]:
    """Parse a printed flat dict such as ``{50: 6291.0}``."""

    parsed = _search_literal(text, label)
    if parsed is None:
        return {}
    return {float(key): float(value) for key, value in parsed.items()}


def _search_literal(text: str, label: str) -> dict | None:
    match = re.search(rf"^{re.escape(label)}:?\s*(\{{.*\}})\s*$", text, re.MULTILINE)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(_unwrap(match.group(1)))
    except (ValueError, SyntaxError):  # pragma: no cover - malformed evaluator output
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_evaluation(text: str, *, class_names: Sequence[str] | None = None) -> Evaluation:
    """Read one ``OWEvaluator.summarize`` block out of a training/eval log."""

    known = _search_scalar(text, "Known AP50")
    if known is None:
        raise MetricsError(
            "no 'Known AP50' line found: this is not a completed evaluation. "
            "Check that the run reached OWEvaluator.summarize rather than dying earlier."
        )
    unknown_recall = _search_scalar(text, "Unknown Recall50")
    if unknown_recall is None:
        raise MetricsError("no 'Unknown Recall50' line found; U-Recall is unavailable.")

    per_class: dict[str, float] = {}
    for name, value in re.findall(rf"^([A-Za-z][\w \-]*?)\s+({_FLOAT})\s*$", text, re.MULTILINE):
        key = name.strip()
        if class_names is not None and key not in class_names:
            continue
        if key in {"detection mAP50", "detection mAP"}:
            continue
        per_class[key] = float(value)

    return Evaluation(
        known_map50=known,
        unknown_recall50=unknown_recall,
        wilderness_impact=_search_nested_mapping(text, "Wilderness Impact"),
        absolute_ose=_search_mapping(text, "Absolute OSE (total_num_unk_det_as_known)"),
        per_class_ap50=per_class,
        previous_map50=_search_scalar(text, "Prev class AP50"),
        current_map50=_search_scalar(text, "Current class AP50"),
        unknown_ap50=_search_scalar(text, "Unknown AP50"),
        raw=text,
    )


def grouped_map(
    evaluation: Evaluation, groups: Mapping[str, Sequence[str]]
) -> dict[str, float | None]:
    """Mean AP50 per frequency group — the proposal's "csoportonkénti mAP"."""

    result: dict[str, float | None] = {}
    for group, names in groups.items():
        values = [
            evaluation.per_class_ap50[name]
            for name in names
            if name in evaluation.per_class_ap50
        ]
        result[group] = sum(values) / len(values) if values else None
    return result


def forgetting(
    before: Mapping[str, float | None], after: Mapping[str, float | None]
) -> dict[str, float | None]:
    """Per-group forgetting: ``before - after``."""

    result: dict[str, float | None] = {}
    for group in set(before) | set(after):
        start, end = before.get(group), after.get(group)
        result[group] = None if start is None or end is None else start - end
    return result


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many."""

    if boxes.size == 0:
        return np.zeros(0)
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    overlap = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return overlap / np.maximum(area + areas - overlap, 1e-9)


def unknown_recall_by_group(
    artifact: Mapping[str, object] | str,
    *,
    known_classes: Sequence[str],
    groups: Mapping[str, str],
    iou_threshold: float = 0.5,
    minimum_score: float = 0.0,
) -> dict[str, dict[str, float]]:
    """U-Recall split by the true class's frequency group.

    **This is the research plan's headline endpoint.** The plan asks for
    "csoportonkénti mAP és U-Recall ... valamint a tail-U-Recall ... mint az
    orákulum-költség függvénye", and predicts that distribution-aware selection
    reaches the same tail level from far fewer annotations. The aggregate U-Recall
    the official evaluator reports cannot show that: it averages over every
    unknown class at once, which is exactly the structure the research is about.

    Computed from the detections artifact, which holds the same post-processed
    detections the official numbers come from — so this is a decomposition of
    U-Recall, not a re-implementation of it. The artifact's ground truth carries
    each object's **true** class name (``ground_truth_records`` reads
    ``load_instances`` before PROB relabels unknowns), which is what makes the
    grouping possible at all.

    An unknown object counts as recalled when some detection of the unknown class
    overlaps it at ``iou_threshold``. Detections are matched greedily by
    descending score and each is used once, so two detections on one object
    cannot recall two.
    """

    if isinstance(artifact, (str, Path)):
        artifact = json.loads(Path(artifact).read_text(encoding="utf-8"))
    if artifact.get("schema") != "daowod_detections_v1":
        raise MetricsError(
            f"Unexpected detections schema {artifact.get('schema')!r}; this reader "
            "understands 'daowod_detections_v1'."
        )

    unknown_name = artifact["unknown_class_name"]
    known = set(known_classes)

    truth: dict[str, list[tuple[str, list[float]]]] = {}
    for record in artifact["ground_truth"]:
        name = record["class_name"]
        if name in known or name == unknown_name:
            continue                       # a known object is not an unknown to find
        truth.setdefault(record["image_id"], []).append((name, record["box"]))

    found: dict[str, list[tuple[float, list[float]]]] = {}
    for record in artifact["detections"]:
        if record["class_name"] != unknown_name or record["score"] < minimum_score:
            continue
        found.setdefault(record["image_id"], []).append((record["score"], record["box"]))

    tally = {group: [0, 0] for group in (*GROUPS, "unassigned")}
    for image_id, objects in truth.items():
        boxes = np.asarray([box for _, box in objects], dtype=float)
        names = [name for name, _ in objects]
        claimed = np.zeros(len(objects), dtype=bool)

        for _, box in sorted(found.get(image_id, ()), key=lambda item: -item[0]):
            overlaps = _iou(np.asarray(box, dtype=float), boxes)
            overlaps[claimed] = -1.0
            best = int(np.argmax(overlaps)) if overlaps.size else -1
            if best >= 0 and overlaps[best] >= iou_threshold:
                claimed[best] = True

        for name, hit in zip(names, claimed):
            group = groups.get(name, "unassigned")
            if group not in tally:
                group = "unassigned"
            tally[group][1] += 1
            tally[group][0] += int(hit)

    result = {
        group: {
            "recalled": recalled,
            "objects": total,
            "recall": 100.0 * recalled / total if total else None,
        }
        for group, (recalled, total) in tally.items()
    }
    recalled = sum(v["recalled"] for v in result.values())
    objects = sum(v["objects"] for v in result.values())
    result["all"] = {
        "recalled": recalled,
        "objects": objects,
        "recall": 100.0 * recalled / objects if objects else None,
    }
    return result


def from_bridge_metrics(path, *, class_names=None) -> Evaluation:
    """Read the JSON the PROB bridge writes next to every evaluation.

    Preferred over :func:`parse_evaluation`: the bridge writes the evaluator's
    numbers straight out, so nothing depends on how a log happened to be
    formatted. The log parser stays for older runs and for reading a
    still-running job.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    official = payload.get("official_metrics", {})
    per_class = per_class_ap50(payload, class_names=class_names)
    return Evaluation(
        known_map50=float(payload.get("known_AP50", payload.get("known_mAP", 0.0))),
        unknown_recall50=float(payload.get("U_Recall", 0.0)),
        wilderness_impact={0.8: {50.0: float(payload.get("WI", 0.0))}},
        absolute_ose={50.0: float(payload.get("A_OSE", 0.0))},
        per_class_ap50=per_class,
        previous_map50=_optional(payload.get("previous_known_AP50")),
        current_map50=_optional(payload.get("current_known_AP50")),
        unknown_ap50=_optional(payload.get("unknown_AP50")),
        raw=json.dumps(official),
    )


def _optional(value) -> float | None:
    return None if value is None else float(value)


#: Where per-class AP50 actually lives. The bridge writes no ``per_class_AP50``
#: key, but it does write ``coco_eval_bbox``, and that vector is
#: ``[mAP, mAP, <80 classes in the evaluator's order>, unknown]`` — 83 entries.
#:
#: This matters more than it looks: the head/medium/tail decomposition is the
#: research plan's distinguishing form of evaluation, and without per-class
#: numbers it cannot be computed at all. It was available in every metrics file
#: already. Verified against three committed GPU runs: the mean over
#: ``CLASS_ORDER[:prev]`` reproduces ``previous_known_AP50`` exactly, the mean
#: over ``CLASS_ORDER[prev:prev + current]`` reproduces ``current_known_AP50``,
#: and the last entry reproduces ``unknown_AP50``.
COCO_EVAL_BBOX_OFFSET = 2
COCO_EVAL_BBOX_CLASSES = 80


def per_class_ap50(payload: Mapping[str, object], *, class_names=None) -> dict[str, float]:
    """AP50 per class name, read out of ``coco_eval_bbox``.

    Falls back to a ``per_class_AP50`` mapping if a future bridge writes one.
    Returns an empty mapping rather than guessing when the vector is the wrong
    length, because a misaligned per-class table is worse than none: it would
    attribute one class's score to another and the head/medium/tail split would
    be quietly wrong.
    """

    explicit = payload.get("per_class_AP50")
    if isinstance(explicit, Mapping) and explicit:
        return {
            str(name): float(value) for name, value in explicit.items()
            if class_names is None or name in class_names
        }

    from owl.protocol import CLASS_ORDER

    vector = payload.get("coco_eval_bbox") or []
    expected = COCO_EVAL_BBOX_OFFSET + COCO_EVAL_BBOX_CLASSES + 1
    if len(vector) != expected:
        return {}

    start = COCO_EVAL_BBOX_OFFSET
    values = vector[start : start + COCO_EVAL_BBOX_CLASSES]
    result = {
        name: float(value)
        for name, value in zip(CLASS_ORDER, values)
        if class_names is None or name in class_names
    }
    if class_names is None or "unknown" in class_names:
        result["unknown"] = float(vector[-1])
    return result



def infer_introduced_counts(payload: Mapping[str, object]) -> dict[str, object]:
    """Recover the evaluator's own slice boundaries from the file itself.

    ``OWEvaluator.summarize`` computes

        PK_AP50 = AP[:, o50][:prev_intro_cls].mean()
        CK_AP50 = AP[:, o50][prev_intro_cls:prev_intro_cls + curr_intro_cls].mean()

    over an ``AP`` array indexed by ``CLASS_NAMES`` position, and publishes that
    same array as ``coco_eval_bbox`` — see ``datasets/open_world_eval.py``::

        self.coco_eval['bbox'].stats = torch.cat(
            [self.AP[:, o50].mean(dim=0, keepdim=True),
             self.AP.flatten().mean(dim=0, keepdim=True), self.AP.flatten()])

    but it never writes ``prev_intro_cls`` into the metrics file. Assuming a
    value is what silently broke this check: 19 happens to be right at t2 and
    wrong at every task after it, so the error grew with the chain and looked
    like a layout problem.

    So rather than assume, search for the prefix length whose mean reproduces
    the reported ``previous_known_AP50``. A match is useful only when it is
    unique: zero-heavy AP vectors can make several prefix lengths reproduce the
    same aggregate. Protocol counts remain the source of truth; this recovery is
    a diagnostic cross-check, or a fallback when there is exactly one solution.
    """

    vector = payload.get("coco_eval_bbox") or []
    expected = COCO_EVAL_BBOX_OFFSET + COCO_EVAL_BBOX_CLASSES + 1
    if len(vector) != expected:
        return {"found": False, "reason": f"coco_eval_bbox has {len(vector)} entries"}

    values = [float(v) for v in vector[COCO_EVAL_BBOX_OFFSET:]]
    reported_prev = payload.get("previous_known_AP50")
    reported_current = payload.get("current_known_AP50")

    n_prev = 0
    if reported_prev is not None:
        target = float(reported_prev)
        matches = [
            k for k in range(1, len(values))
            if abs(sum(values[:k]) / k - target) <= 1e-3
        ]
        if not matches:
            return {"found": False,
                    "reason": "no prefix of coco_eval_bbox averages to "
                              f"previous_known_AP50 ({target:.6f})"}
        if len(matches) != 1:
            return {
                "found": False,
                "reason": "multiple prefixes of coco_eval_bbox average to "
                          f"previous_known_AP50 ({target:.6f}): {matches}",
                "ambiguous": True,
                "previous_candidates": matches,
            }
        n_prev = matches[0]

    n_current = 0
    if reported_current is not None:
        target = float(reported_current)
        matches = [
            k for k in range(1, len(values) - n_prev + 1)
            if abs(sum(values[n_prev:n_prev + k]) / k - target) <= 1e-3
        ]
        if not matches:
            return {"found": False,
                    "reason": "no window after the previous classes averages to "
                              f"current_known_AP50 ({target:.6f})"}
        if len(matches) != 1:
            return {
                "found": False,
                "reason": "multiple windows after the previous classes average to "
                          f"current_known_AP50 ({target:.6f}): {matches}",
                "ambiguous": True,
                "current_candidates": matches,
            }
        n_current = matches[0]

    return {"found": True, "previous_introduced_classes": n_prev,
            "current_introduced_classes": n_current}


def validate_per_class_ap50(
    payload: Mapping[str, object],
    *,
    n_prev: int | None = None,
    n_current: int | None = None,
    tolerance: float = 1e-3,
) -> dict[str, object]:
    """Check the per-class vector against the aggregates the evaluator reported.

    ``coco_eval_bbox`` is not labelled as a per-class table anywhere in the
    metrics file, and the file does not record how many classes had been
    introduced when it was written. Both are needed to check it, and **neither
    may be guessed** — a wrong class count is exactly what made this check fail
    from t3 onward while passing at t2.

    ``n_prev`` and ``n_current`` should therefore be passed by the caller, which
    knows them from the protocol (``owl.protocol.Task.n_prev`` / ``n_new`` are
    what the runner handed the bridge). When they are not passed, they are
    *recovered from the file* by :func:`infer_introduced_counts` rather than
    assumed. When both are available they are compared, and a disagreement is
    itself a failure — the file and the protocol must describe the same run.
    """

    per_class = per_class_ap50(payload)
    if not per_class:
        vector = payload.get("coco_eval_bbox") or []
        return {
            "usable": False,
            "reason": (
                f"coco_eval_bbox has {len(vector)} entries, not the "
                f"{COCO_EVAL_BBOX_OFFSET + COCO_EVAL_BBOX_CLASSES + 1} this reader "
                "understands, and no per_class_AP50 mapping was written"
            ),
            "checks": [],
        }

    from owl.protocol import CLASS_ORDER

    inferred = infer_introduced_counts(payload)
    declared_prev = payload.get("previous_introduced_classes")
    declared_current = payload.get("current_introduced_classes")

    if n_prev is None:
        n_prev = (int(declared_prev) if declared_prev is not None
                  else inferred.get("previous_introduced_classes"))
    if n_current is None:
        n_current = (int(declared_current) if declared_current is not None
                     else inferred.get("current_introduced_classes"))
    if n_prev is None or n_current is None:
        return {"usable": False, "checks": [],
                "reason": inferred.get("reason", "the class counts are unknown")}
    n_prev, n_current = int(n_prev), int(n_current)
    if n_prev < 0 or n_current < 0 or n_prev + n_current > len(CLASS_ORDER):
        return {"usable": False, "checks": [],
                "reason": f"invalid class counts: prev={n_prev}, current={n_current}"}

    checks: list[dict] = []

    def compare(label: str, names: Sequence[str], reported: object) -> None:
        values = [per_class[name] for name in names if name in per_class]
        if not values or reported is None:
            return
        rebuilt = sum(values) / len(values)
        checks.append({
            "quantity": label,
            "classes": len(values),
            "rebuilt": rebuilt,
            "reported": float(reported),
            "agrees": abs(rebuilt - float(reported)) <= tolerance,
        })

    if n_prev:
        compare("previous_known_AP50", CLASS_ORDER[:n_prev],
                payload.get("previous_known_AP50"))
    if n_current:
        compare("current_known_AP50", CLASS_ORDER[n_prev:n_prev + n_current],
                payload.get("current_known_AP50"))
    if n_prev + n_current:
        compare("known_AP50", CLASS_ORDER[:n_prev + n_current],
                payload.get("known_AP50"))
    unknown = payload.get("unknown_AP50")
    if unknown is not None and "unknown" in per_class:
        checks.append({
            "quantity": "unknown_AP50",
            "classes": 1,
            "rebuilt": per_class["unknown"],
            "reported": float(unknown),
            "agrees": abs(per_class["unknown"] - float(unknown)) <= tolerance,
        })

    disagreed = [c["quantity"] for c in checks if not c["agrees"]]
    # the protocol and the file must describe the same run
    if inferred.get("found") and inferred.get("previous_introduced_classes") != n_prev:
        disagreed.append(
            f"the file's own aggregates imply prev_intro_cls="
            f"{inferred['previous_introduced_classes']}, the caller said {n_prev}")
    if inferred.get("found") and inferred.get("current_introduced_classes") != n_current:
        disagreed.append(
            f"the file's own aggregates imply curr_intro_cls="
            f"{inferred['current_introduced_classes']}, the caller said {n_current}")
    if declared_prev is not None and int(declared_prev) != n_prev:
        disagreed.append(
            f"the file declares prev_intro_cls={int(declared_prev)}, "
            f"the caller said {n_prev}")
    if declared_current is not None and int(declared_current) != n_current:
        disagreed.append(
            f"the file declares curr_intro_cls={int(declared_current)}, "
            f"the caller said {n_current}")

    return {
        "usable": bool(checks) and not disagreed,
        "reason": (
            "" if not disagreed else
            f"the per-class vector does not reproduce {disagreed}; the layout of "
            "coco_eval_bbox or the class counts are not what this reader assumes"
        ),
        "checks": checks,
        "n_classes": len(per_class),
        "previous_introduced_classes": n_prev,
        "current_introduced_classes": n_current,
        "counts_recovered_from_file": inferred.get("found", False),
        "count_inference_reason": inferred.get("reason", ""),
    }


def per_class_recall(
    artifact: Mapping[str, object] | str,
    *,
    classes: Sequence[str],
    iou_threshold: float = 0.5,
    minimum_score: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Per-class recall at ``iou_threshold`` from the detections artefact.

    This is **not** a second AP implementation and must not be reported as AP:
    AP integrates precision over recall and depends on the score ranking of every
    detection, which is exactly the part a hand-rolled reimplementation gets
    subtly wrong. What the artefact supports without that risk is recall, and it
    is computed with the same greedy, score-ordered, one-detection-per-object
    matching :func:`unknown_recall_by_group` already uses.

    Its purpose is a cross-check: a class whose AP50 collapsed should also lose
    recall, and a class the evaluator says is fine should still be found. When
    the two disagree, the per-class AP table is the one to distrust.
    """

    if isinstance(artifact, (str, Path)):
        artifact = json.loads(Path(artifact).read_text(encoding="utf-8"))
    if artifact.get("schema") != "daowod_detections_v1":
        raise MetricsError(
            f"Unexpected detections schema {artifact.get('schema')!r}; this reader "
            "understands 'daowod_detections_v1'."
        )

    wanted = set(classes)
    truth: dict[tuple[str, str], list[list[float]]] = {}
    for record in artifact["ground_truth"]:
        name = record["class_name"]
        if name in wanted:
            truth.setdefault((record["image_id"], name), []).append(record["box"])

    found: dict[tuple[str, str], list[tuple[float, list[float]]]] = {}
    for record in artifact["detections"]:
        name = record["class_name"]
        if name in wanted and record["score"] >= minimum_score:
            found.setdefault((record["image_id"], name), []).append(
                (record["score"], record["box"]))

    tally = {name: [0, 0] for name in wanted}
    for (image_id, name), boxes in truth.items():
        matrix = np.asarray(boxes, dtype=float)
        claimed = np.zeros(len(boxes), dtype=bool)
        for _, box in sorted(found.get((image_id, name), ()), key=lambda item: -item[0]):
            overlaps = _iou(np.asarray(box, dtype=float), matrix)
            overlaps[claimed] = -1.0
            best = int(np.argmax(overlaps)) if overlaps.size else -1
            if best >= 0 and overlaps[best] >= iou_threshold:
                claimed[best] = True
        tally[name][0] += int(claimed.sum())
        tally[name][1] += len(boxes)

    return {
        name: {
            "recalled": recalled,
            "objects": total,
            "recall": 100.0 * recalled / total if total else None,
        }
        for name, (recalled, total) in tally.items()
    }


# ------------------------------------------------- the chain's reporting row ---


def task_row(
    evaluation: Evaluation,
    *,
    task: str,
    new_class: str | None,
    anchor_known_map50: float | None = None,
    previous_baseline: float | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """One task of the chain, as the notebook prints it.

    ``previous_baseline`` is the previous-class mAP50 measured *before* this
    task trained, so ``forgetting`` is the drop this task caused rather than the
    distance from the pretrained anchor.
    """

    row: dict[str, object] = {
        "task": task,
        "new_class": new_class or "—",
        "known_mAP50": evaluation.known_map50,
        "prev_mAP50": evaluation.previous_map50,
        "new_mAP50": evaluation.current_map50,
        "U_Recall50": evaluation.unknown_recall50,
    }
    if new_class and new_class in evaluation.per_class_ap50:
        row["new_class_AP50"] = evaluation.per_class_ap50[new_class]
    if previous_baseline is not None and evaluation.previous_map50 is not None:
        row["forgetting"] = previous_baseline - evaluation.previous_map50
    if anchor_known_map50 is not None:
        row["drop_from_anchor"] = anchor_known_map50 - evaluation.known_map50
    if groups:
        for group, value in grouped_map(evaluation, groups).items():
            row[f"mAP50_{group}"] = value
    return row


def exchange_rate(row: Mapping[str, object]) -> float | None:
    """Old mAP50 points paid per new mAP50 point gained.

    The single number that says whether an incremental step was worth taking.
    Full t2 supervision measured 0.20 in the earlier work; a step that pays 74
    is not a trade, it is a loss with a rounding error attached.
    """

    forgot = row.get("forgetting")
    learned = row.get("new_mAP50") or row.get("new_class_AP50")
    if forgot is None or not learned:
        return None
    return float(forgot) / float(learned)


def group_membership(
    class_names: Sequence[str], groups: Mapping[str, str]
) -> dict[str, list[str]]:
    """Invert ``class -> group`` into ``group -> classes``, restricted to ``class_names``."""

    result: dict[str, list[str]] = {group: [] for group in GROUPS}
    for name in class_names:
        group = groups.get(name)
        if group in result:
            result[group].append(name)
    return result
