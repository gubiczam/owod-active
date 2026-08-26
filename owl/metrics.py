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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

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


def grouped_unknown_recall(
    detections: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, Sequence[str]],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, float | None]:
    """Tail-resolved U-Recall, the proposal's headline metric for Contribution A."""

    recalled = dict.fromkeys(groups, 0)
    present = dict.fromkeys(groups, 0)
    membership = {name: group for group, names in groups.items() for name in names}

    for image_id, truth in ground_truth.items():
        found = list(detections.get(image_id, ()))
        for class_name in truth:
            group = membership.get(class_name)
            if group is None:
                continue
            present[group] += 1
            if class_name in found:
                found.remove(class_name)
                recalled[group] += 1

    return {
        group: (recalled[group] / present[group] if present[group] else None) for group in groups
    }


def from_bridge_metrics(path, *, class_names=None) -> Evaluation:
    """Read the JSON the PROB bridge writes next to every evaluation.

    Preferred over :func:`parse_evaluation`: the bridge writes the evaluator's
    numbers straight out, so nothing depends on how a log happened to be
    formatted. The log parser stays for older runs and for reading a
    still-running job.
    """

    import json
    from pathlib import Path as _Path

    payload = json.loads(_Path(path).read_text(encoding="utf-8"))
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
