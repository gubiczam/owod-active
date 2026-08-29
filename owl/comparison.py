"""Reading finished replay runs back off disk and comparing them.

:mod:`owl.analysis` answers the *selection* study's question — tail U-Recall
against oracle cost, one line per selection arm. This module answers the
*replay* study's question, which needs different things from the same workspaces:

* per-class AP at every task, which is not in ``results_<arm>.csv`` at all. It
  lives in each ``<task>_<arm>/metrics.json`` as ``coco_eval_bbox``, and the
  anchor's is in ``anchor_metrics.json``. Group aggregates cannot substitute:
  the ``tail`` band holds one class at t2 and four at t6, so a change in
  ``mAP50_tail`` is partly a change of denominator.
* the replay composition the chain recorded per task, so a retention difference
  can be read against the memory that produced it.
* the annotation and supervision cost, so retention is never quoted without the
  plasticity and the compute that came with it.

Every reader tolerates a run that is missing, partial, or has no anchor: the
comparison is meant to be run *while* the GPU is still working, and a table
whose later columns are ``None`` is more useful than an exception.

Nothing here writes into a workspace.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from owl import metrics, protocol
from owl.analysis import AnalysisError, _number

#: The frequency bands, in the order every table and figure lists them.
GROUPS: tuple[str, ...] = ("head", "medium", "tail")

#: The run names this study expects. A workspace root may hold any subset.
EXPECTED: tuple[str, ...] = (
    "random__none", "random__uniform", "random__tail_favouring",
)


@dataclass(frozen=True)
class Run:
    """One ``<selection>__<replay>`` workspace, as far as it got."""

    name: str
    selection_arm: str
    replay_arm: str
    path: Path
    rows: list[dict]
    config: dict
    #: class -> AP50 on the starting checkpoint. Empty when the run predates the
    #: anchor evaluation or it has not been written yet.
    anchor_ap: dict[str, float] = field(default_factory=dict)
    #: task name -> class -> AP50, for every task that finished.
    per_task_ap: dict[str, dict[str, float]] = field(default_factory=dict)
    #: task name -> the validation report for that task's per-class vector. The
    #: per-class AP is not a labelled table in the metrics file; it is read out
    #: of ``coco_eval_bbox``, so every task records whether that vector
    #: reproduced the evaluator's own aggregates.
    per_class_checks: dict[str, dict] = field(default_factory=dict)
    #: task name -> class -> recall at IoU 0.5, from the detections artefact.
    #: An independent cross-check, never reported as AP.
    per_task_recall: dict[str, dict[str, dict]] = field(default_factory=dict)

    @property
    def per_class_ap_is_validated(self) -> bool:
        return bool(self.per_class_checks) and all(
            report.get("usable") for report in self.per_class_checks.values())

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def tasks(self) -> list[str]:
        return [str(row.get("task")) for row in self.rows]

    def row(self, task: str) -> dict | None:
        for row in self.rows:
            if str(row.get("task")) == task:
                return row
        return None

    @property
    def final_task(self) -> str | None:
        return self.tasks[-1] if self.rows else None


# ------------------------------------------------------------------ reading ---


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{k: _number(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def _read_metrics(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _per_class(path: Path, *, n_prev: int | None = None,
               n_current: int | None = None) -> tuple[dict[str, float], dict]:
    """Per-class AP50 and the report saying whether it may be trusted.

    The metrics file has no key named for a per-class table. What it has is
    ``coco_eval_bbox``, an 83-entry vector laid out as
    ``[mAP, mAP, <80 classes>, unknown]``, and :mod:`owl.metrics` is the one
    place that knows those offsets. Reading it here rather than recomputing AP
    from the raw detections is deliberate: these are the evaluator's own
    numbers, and a second AP implementation would be a second set of conventions
    to get subtly wrong.

    Empty rather than raising: a task whose evaluation died leaves a file the
    reader should skip, not a comparison that cannot be produced at all.
    """

    payload = _read_metrics(path)
    if not payload:
        return {}, {"usable": False, "reason": f"{path.name} is missing or unreadable",
                    "checks": []}
    found = metrics.per_class_ap50(payload)
    report = metrics.validate_per_class_ap50(
        payload, n_prev=n_prev, n_current=n_current)
    # A warning next to Table 4 is not enough: once validation says that the
    # layout or slice counts are wrong, those plausible-looking class values
    # must not flow into the table, forgetting analysis, or vulnerability fit.
    return (found if report.get("usable") else {}), report


def _recall(metrics_path: Path, classes: Sequence[str]) -> dict[str, dict]:
    """Per-class recall from the detections artefact this task wrote, if any.

    Used only as a cross-check on the AP table — never as a substitute for it.
    """

    payload = _read_metrics(metrics_path)
    artefact = payload.get("detections_path")
    if not artefact:
        return {}
    path = Path(artefact)
    if not path.is_absolute():
        path = metrics_path.parent / path
    if not path.exists():
        # the chain writes it beside the metrics file under a fixed name
        path = metrics_path.with_name(f"{metrics_path.stem}_detections.json")
    if not path.exists():
        return {}
    try:
        return metrics.per_class_recall(path, classes=classes)
    except (metrics.MetricsError, OSError, json.JSONDecodeError, KeyError):
        return {}


def load_run(directory: Path) -> Run | None:
    """One workspace directory, or ``None`` if it holds no finished task."""

    directory = Path(directory)
    results = sorted(directory.glob("results_*.csv"))
    if not results:
        return None
    rows = _read_rows(results[0])
    if not rows:
        return None

    name = directory.name
    selection, _, replay = name.partition("__")
    config: dict = {}
    stamp = directory / "config.json"
    if stamp.exists():
        try:
            config = json.loads(stamp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
    selection = str(config.get("arm", selection or results[0].stem[len("results_"):]))
    replay = str(config.get("replay_arm", replay or "unknown"))

    # The metrics file does not record how many classes had been introduced when
    # it was written, and PROB slices its aggregates by exactly that number. The
    # chain does know it: `Task.n_prev` and `Task.n_new` are what the runner
    # handed the bridge, so they are the authority here. Guessing 19 for every
    # task is what made this check pass at t2 and fail at t3 onward.
    n_tasks = int(config.get("n_tasks") or (len(rows) + 1))
    try:
        chain = {task.name: task for task in protocol.build_chain(n_tasks)}
    except protocol.ProtocolError:
        chain = {}
    anchor_task = protocol.build_chain(2)[0]

    anchor, anchor_check = _per_class(
        directory / "anchor_metrics.json",
        n_prev=anchor_task.n_prev, n_current=anchor_task.n_new)
    per_task: dict[str, dict[str, float]] = {}
    checks: dict[str, dict] = {}
    recall: dict[str, dict[str, dict]] = {}
    if anchor_check.get("checks"):
        checks["anchor"] = anchor_check
    for row in rows:
        task = str(row.get("task"))
        step = chain.get(task)
        metrics_path = directory / f"{task}_{selection}" / "metrics.json"
        found, report = _per_class(
            metrics_path,
            n_prev=step.n_prev if step else None,
            n_current=step.n_new if step else None)
        if found:
            per_task[task] = found
        if report.get("checks") or found:
            checks[task] = report
        measured = _recall(metrics_path, list(protocol.TASK1))
        if measured:
            recall[task] = measured

    return Run(name=name, selection_arm=selection, replay_arm=replay,
               path=directory, rows=rows, config=config,
               anchor_ap=anchor, per_task_ap=per_task,
               per_class_checks=checks, per_task_recall=recall)


def load_runs(
    root: str | Path,
    *,
    include: Sequence[str] | None = None,
) -> dict[str, Run]:
    """The registered replay runs under ``root``, keyed by directory name.

    ``root`` may be the workspace root that holds one directory per run, or a
    single run directory. Directories without a results file are skipped
    silently — a run that has not started yet is not an error.

    By default only :data:`EXPECTED` is considered. A Drive workspace
    accumulates directories from every earlier study — ``objectness``,
    ``prior_consult_batch``, a bare ``random`` — and those are different
    experiments, not arms of this one. Reporting them as incompatible is
    correct but noisy; the registered experiment should simply not look at
    them. Pass ``include`` to widen the set deliberately.
    """

    root = Path(root)
    if not root.exists():
        raise AnalysisError(f"{root} does not exist.")

    wanted = set(EXPECTED if include is None else include)
    found: dict[str, Run] = {}
    single = load_run(root)
    if single is not None:
        found[single.name] = single
    else:
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            if child.name not in wanted:
                continue
            run = load_run(child)
            if run is not None:
                found[run.name] = run
    if not found:
        raise AnalysisError(
            f"No finished run under {root} among {sorted(wanted)}. A run "
            "directory holds results_<arm>.csv next to its per-task metrics; "
            "download the workspace from Drive, not the notebook output. Pass "
            "--include to consider directories outside the registered experiment."
        )
    return {name: found[name] for name in sorted(found, key=_order)}


def _order(name: str) -> tuple[int, str]:
    """None, then uniform, then tail-favouring: baseline first, always."""

    return (EXPECTED.index(name), "") if name in EXPECTED else (len(EXPECTED), name)


# ------------------------------------------------------------ compatibility ---

#: Fields that must agree for two runs to be comparable at all. ``replay_arm``
#: is deliberately absent: it is the variable under study.
MUST_MATCH: tuple[str, ...] = (
    "n_tasks", "budget_per_task", "rounds_per_task", "candidate_images_per_task",
    "proposals_per_image", "arm", "labelling_policy", "replay_reallocate",
    "replay_protocol_version", "reuse_deferred_labels", "epochs", "learning_rate",
    "batch_size", "n_clusters", "seed",
)


def compatibility(runs: Mapping[str, Run], *, reference: str | None = None) -> list[dict]:
    """Which result-affecting settings differ between runs. Empty means none.

    This is the check that a completed baseline really belongs in the same table
    as tonight's arms — a chain run at ``n_tasks=2`` or a different seed is a
    different experiment, and the fingerprint guard only protects a workspace
    from *itself*, never one workspace from another.
    """

    if not runs:
        return []
    names = list(runs)
    reference = reference if reference in runs else names[0]
    base = runs[reference].config
    out: list[dict] = []
    for name in names:
        if name == reference:
            continue
        other = runs[name].config
        for field_name in MUST_MATCH:
            here, there = base.get(field_name, "(absent)"), other.get(field_name, "(absent)")
            if here != there:
                out.append({"field": field_name, "reference": reference,
                            "reference_value": here, "run": name, "value": there})
    return out


def depth_report(runs: Mapping[str, Run]) -> list[dict]:
    """How far each run got, and whether its anchor and per-class data exist."""

    return [
        {
            "run": name,
            "selection_arm": run.selection_arm,
            "replay_arm": run.replay_arm,
            "tasks_finished": len(run),
            "last_task": run.final_task,
            "has_anchor_ap": bool(run.anchor_ap),
            "tasks_with_per_class_ap": len(run.per_task_ap),
            "per_class_ap_validated": run.per_class_ap_is_validated,
            "tasks_with_recall_crosscheck": len(run.per_task_recall),
        }
        for name, run in runs.items()
    ]


# ------------------------------------------------------------------- tables ---

#: Table 1's per-task quantities, in reporting order.
TASK_METRICS: tuple[str, ...] = (
    "known_mAP50", "prev_mAP50", "new_mAP50", "forgetting",
    "mAP50_head", "mAP50_medium", "mAP50_tail",
)


def _tasks_of(runs: Mapping[str, Run]) -> list[str]:
    """Every task any run reached, in chain order."""

    seen: dict[str, None] = {}
    for run in runs.values():
        for task in run.tasks:
            seen.setdefault(task, None)
    return sorted(seen, key=lambda name: int(str(name).lstrip("t") or 0))


def table_task_comparison(runs: Mapping[str, Run]) -> list[dict]:
    """TABLE 1 — every metric, every task, one column group per run."""

    out: list[dict] = []
    for task in _tasks_of(runs):
        row: dict = {"task": task}
        for name, run in runs.items():
            source = run.row(task) or {}
            row["new_class"] = row.get("new_class") or source.get("new_class")
            for metric in TASK_METRICS:
                row[f"{name}:{metric}"] = source.get(metric)
        out.append(row)
    return out


def table_delta_versus_baseline(
    runs: Mapping[str, Run], *, baseline: str = "random__none"
) -> list[dict]:
    """TABLE 2 — each replay run minus the no-replay run, per task."""

    if baseline not in runs:
        return []
    out: list[dict] = []
    for task in _tasks_of(runs):
        base_row = runs[baseline].row(task) or {}
        for name, run in runs.items():
            if name == baseline:
                continue
            source = run.row(task) or {}
            row: dict = {"task": task, "run": name, "baseline": baseline}
            for metric in TASK_METRICS:
                row[f"delta_{metric}"] = _delta(source.get(metric), base_row.get(metric))
            out.append(row)
    return out


def table_tail_versus_uniform(
    runs: Mapping[str, Run],
    *,
    treatment: str = "random__tail_favouring",
    control: str = "random__uniform",
) -> list[dict]:
    """TABLE 3 — the contribution's own comparison, per task."""

    if treatment not in runs or control not in runs:
        return []
    out: list[dict] = []
    for task in _tasks_of(runs):
        here = runs[treatment].row(task) or {}
        there = runs[control].row(task) or {}
        row: dict = {"task": task, "treatment": treatment, "control": control}
        for metric in TASK_METRICS:
            row[f"delta_{metric}"] = _delta(here.get(metric), there.get(metric))
        out.append(row)
    return out


def _delta(here: object, there: object) -> float | None:
    if isinstance(here, (int, float)) and isinstance(there, (int, float)):
        return float(here) - float(there)
    return None


def table_per_class(runs: Mapping[str, Run]) -> list[dict]:
    """TABLE 4 — per-class retention, which the group aggregates cannot show.

    The reference AP is the anchor when the run recorded one. Relative
    forgetting is reported only where the anchor AP is large enough for the
    ratio to mean anything: dividing a two-point drop by an anchor of 0.3 is a
    number, not a measurement.
    """

    counts = protocol.load_train_counts()
    groups = protocol.load_groups()
    known = list(protocol.TASK1)

    reference: dict[str, float] = {}
    for run in runs.values():
        if run.anchor_ap:
            reference = run.anchor_ap
            break

    out: list[dict] = []
    for name in known:
        row: dict = {
            "class": name,
            "train_objects": counts.get(name),
            "group": groups.get(name),
            "anchor_AP50": reference.get(name),
        }
        for run_name, run in runs.items():
            final = run.final_task
            per_class = run.per_task_ap.get(final or "", {})
            value = per_class.get(name)
            row[f"{run_name}:final_AP50"] = value
            # the independent signal: recall at IoU 0.5 from the detections
            # artefact. Not AP, and never reported as AP — but a class whose AP
            # collapsed should have lost recall too, and a disagreement means
            # the AP table is the thing to distrust.
            measured = run.per_task_recall.get(final or "", {}).get(name) or {}
            row[f"{run_name}:final_recall50"] = measured.get("recall")
            row[f"{run_name}:test_objects"] = measured.get("objects")
            anchor = reference.get(name)
            row[f"{run_name}:forgetting"] = (
                None if value is None or anchor is None else anchor - value
            )
            row[f"{run_name}:relative_forgetting"] = (
                None if value is None or anchor is None or anchor < 1.0
                else (anchor - value) / anchor
            )
        out.append(row)

    # what replay bought, per class, relative to the baseline
    baseline = "random__none"
    if baseline in runs:
        for row in out:
            base = row.get(f"{baseline}:final_AP50")
            for run_name in runs:
                if run_name == baseline:
                    continue
                row[f"{run_name}:gain_over_none"] = _delta(
                    row.get(f"{run_name}:final_AP50"), base)
        uniform, tail = "random__uniform", "random__tail_favouring"
        if uniform in runs and tail in runs:
            for row in out:
                row["tail_extra_over_uniform"] = _delta(
                    row.get(f"{tail}:final_AP50"), row.get(f"{uniform}:final_AP50"))
    return out


def parse_per_class_quota(text: object) -> dict[str, int]:
    """``'bear:56;car:8'`` as the chain writes it."""

    if not isinstance(text, str) or not text.strip():
        return {}
    out: dict[str, int] = {}
    for piece in text.split(";"):
        name, _, value = piece.rpartition(":")
        if name and value.strip().isdigit():
            out[name] = int(value)
    return out


def table_replay_composition(runs: Mapping[str, Run]) -> list[dict]:
    """TABLE 5 — the memory that produced each task's numbers."""

    groups = protocol.load_groups()
    counts = protocol.load_train_counts()
    out: list[dict] = []
    for name, run in runs.items():
        for row in run.rows:
            quota = parse_per_class_quota(row.get("replay_per_class"))
            band = {g: sum(v for n, v in quota.items() if groups.get(n) == g)
                    for g in GROUPS}
            ranked = sorted(quota.items(), key=lambda item: -item[1])
            out.append({
                "run": name,
                "replay_arm": run.replay_arm,
                "task": row.get("task"),
                "requested": row.get("replay_requested_objects"),
                "allocated": row.get("replay_allocated_objects"),
                "delivered": row.get("replay_delivered_objects"),
                "replay_images": row.get("replay_images"),
                "unique_source_images": row.get("replay_unique_source_images"),
                "head_objects": band["head"],
                "medium_objects": band["medium"],
                "tail_objects": band["tail"],
                "quota_min": min(quota.values()) if quota else None,
                "quota_median": _median(sorted(quota.values())) if quota else None,
                "quota_max": max(quota.values()) if quota else None,
                "most_represented": ", ".join(f"{n}:{v}" for n, v in ranked[:3]),
                "least_represented": ", ".join(f"{n}:{v}" for n, v in ranked[-3:]),
                "rho_quota_frequency": _spearman(
                    [counts[n] for n in quota if n in counts],
                    [v for n, v in quota.items() if n in counts]),
                "retained": row.get("replay_from_previous_memory"),
                "added": row.get("replay_added"),
                "evicted": row.get("replay_evicted"),
            })
    return out


def table_cost(runs: Mapping[str, Run]) -> list[dict]:
    """TABLE 6 — annotation and supervision, so retention is never quoted alone."""

    out: list[dict] = []
    for name, run in runs.items():
        for row in run.rows:
            out.append({
                "run": name,
                "task": row.get("task"),
                "asked": row.get("asked"),
                "images_opened": row.get("images_opened"),
                "images_trainable": row.get("images_trainable"),
                "images_no_supervision": row.get("images_no_supervision"),
                "images_from_earlier_tasks": row.get("images_from_earlier_tasks"),
                "target_objects_in_images": row.get("target_objects_in_images"),
                "replay_objects": row.get("replay_delivered_objects"),
                "replay_images": row.get("replay_images"),
                "oracle_cost_so_far": row.get("oracle_cost_so_far"),
            })
    return out


# ---------------------------------------------------------------- statistics ---


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return (float(values[middle - 1]) + float(values[middle])) / 2


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for position, index in enumerate(order):
        out[index] = float(position)
    return out


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation. ``None`` below three points, where it means nothing."""

    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else None


def _least_squares(rows: Sequence[Sequence[float]], target: Sequence[float]):
    """Ordinary least squares with an intercept, via normal equations.

    Small and explicit rather than a dependency: the design matrix here is at
    most 19 x 3, and numpy's lstsq would hide the fact that this is descriptive.
    """

    import numpy as np

    design = np.column_stack([np.ones(len(rows))] + [
        np.asarray([row[i] for row in rows], dtype=float)
        for i in range(len(rows[0]))
    ]) if rows and rows[0] else None
    if design is None or design.shape[0] <= design.shape[1]:
        return None
    y = np.asarray(target, dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    residual = float(((y - predicted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    return {
        "coefficients": [float(c) for c in coefficients],
        "r_squared": 1.0 - residual / total if total else None,
        "n": int(design.shape[0]),
    }


def vulnerability(run: Run) -> dict:
    """Is forgetting explained by frequency, by anchor AP, or by both?

    The premise of a frequency-weighted memory is that rarity predicts what gets
    forgotten. On one run this is descriptive only — direction and magnitude,
    never a p-value — but it is the check that decides whether the premise holds
    on *this* class set at all.
    """

    counts = protocol.load_train_counts()
    groups = protocol.load_groups()
    final = run.final_task
    after = run.per_task_ap.get(final or "", {})
    if not run.anchor_ap or not after:
        return {"available": False,
                "reason": "the run has no anchor AP or no per-class AP at its last task"}

    names, frequency, anchor, forgetting, relative = [], [], [], [], []
    for name in protocol.TASK1:
        start, end = run.anchor_ap.get(name), after.get(name)
        if start is None or end is None:
            continue
        names.append(name)
        frequency.append(math.log(counts[name]))
        anchor.append(start)
        forgetting.append(start - end)
        relative.append((start - end) / start if start >= 1.0 else float("nan"))

    usable = [i for i, value in enumerate(relative) if not math.isnan(value)]
    out: dict = {
        "available": True,
        "run": run.name,
        "task": final,
        "n_classes": len(names),
        "classes": names,
        "rho_forgetting_vs_log_frequency": _spearman(frequency, forgetting),
        "rho_forgetting_vs_anchor_ap": _spearman(anchor, forgetting),
        "rho_relative_forgetting_vs_log_frequency": _spearman(
            [frequency[i] for i in usable], [relative[i] for i in usable]),
        "group_means": {
            group: _mean([forgetting[i] for i, n in enumerate(names)
                          if groups.get(n) == group])
            for group in GROUPS
        },
    }
    if len(names) >= 5:
        out["ols_frequency_only"] = _least_squares(
            [[f] for f in frequency], forgetting)
        out["ols_anchor_only"] = _least_squares([[a] for a in anchor], forgetting)
        out["ols_frequency_and_anchor"] = _least_squares(
            [[f, a] for f, a in zip(frequency, anchor)], forgetting)
    return out


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


# ------------------------------------------------------------------ figures ---

#: One colour per run, fixed so every figure in the thesis agrees with the next.
STYLE: dict[str, dict] = {
    "random__none":           {"color": "#666666", "marker": "o", "label": "no replay"},
    "random__uniform":        {"color": "#1f77b4", "marker": "s", "label": "uniform (α=0)"},
    "random__tail_favouring": {"color": "#d62728", "marker": "^", "label": "tail (α=−0.5)"},
}


def _style(name: str) -> dict:
    return STYLE.get(name, {"color": "#999999", "marker": "x", "label": name})


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:                                # pragma: no cover
        raise AnalysisError(
            "matplotlib is not installed; run \"pip install -e '.[plots]'\" or pass "
            "--no-plots to get the tables only."
        ) from error
    return plt


def _save(figure, path: Path, formats: Iterable[str]) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        target = path.with_suffix(f".{suffix}")
        figure.savefig(target, dpi=200, bbox_inches="tight")
        written.append(target)
    return written


def _series(run: Run, metric: str) -> tuple[list[str], list[float]]:
    tasks, values = [], []
    for row in run.rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            tasks.append(str(row.get("task")))
            values.append(float(value))
    return tasks, values


def plot_figures(
    runs: Mapping[str, Run], out: Path, *, formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    """Figures A-F. A run that has not produced a quantity is skipped, not faked."""

    plt = _pyplot()
    out = Path(out)
    written: list[Path] = []
    counts = protocol.load_train_counts()

    # A: head / medium / tail AP over the chain
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    drew = False
    for column, group in enumerate(GROUPS):
        for name, run in runs.items():
            tasks, values = _series(run, f"mAP50_{group}")
            if not values:
                continue
            drew = True
            style = _style(name)
            axes[column].plot(tasks, values, marker=style["marker"],
                              color=style["color"], label=style["label"])
        axes[column].set_title(f"{group} mAP50")
        axes[column].grid(True, alpha=0.3)
        axes[column].set_xlabel("task")
    axes[0].set_ylabel("mAP50")
    if drew:
        axes[-1].legend(fontsize=8)
        written += _save(figure, out / "figure_a_group_ap", formats)
    plt.close(figure)

    # B: forgetting, and C: new-class AP — the two halves of the trade-off
    for key, metric, ylabel, title in (
        ("figure_b_forgetting", "forgetting", "forgetting (mAP50 points)",
         "Previous-class forgetting per task"),
        ("figure_c_new_class_ap", "new_mAP50", "new-class mAP50",
         "Plasticity: the class this task introduced"),
    ):
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        drew = False
        for name, run in runs.items():
            tasks, values = _series(run, metric)
            if not values:
                continue
            drew = True
            style = _style(name)
            axis.plot(tasks, values, marker=style["marker"], color=style["color"],
                      label=style["label"])
        axis.set_xlabel("task")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        if drew:
            axis.legend(fontsize=8)
            written += _save(figure, out / key, formats)
        plt.close(figure)

    # D: per-class final forgetting against log training frequency
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    drew = False
    for name, run in runs.items():
        final = run.final_task
        after = run.per_task_ap.get(final or "", {})
        if not run.anchor_ap or not after:
            continue
        x, y = [], []
        for class_name in protocol.TASK1:
            start, end = run.anchor_ap.get(class_name), after.get(class_name)
            if start is None or end is None:
                continue
            x.append(math.log10(counts[class_name]))
            y.append(start - end)
        if not x:
            continue
        drew = True
        style = _style(name)
        axis.scatter(x, y, color=style["color"], marker=style["marker"],
                     label=style["label"], alpha=0.8)
    axis.set_xlabel("log10 training objects")
    axis.set_ylabel("final forgetting (AP50 points)")
    axis.set_title("Is forgetting explained by rarity?")
    axis.grid(True, alpha=0.3)
    if drew:
        axis.legend(fontsize=8)
        written += _save(figure, out / "figure_d_forgetting_vs_frequency", formats)
    plt.close(figure)

    # E: the replay allocation itself, uniform against tail-favouring
    figure, axis = plt.subplots(figsize=(8.0, 4.0))
    drew = False
    order = sorted(protocol.TASK1, key=lambda n: counts[n])
    for name, run in runs.items():
        if run.replay_arm == "none" or not run.rows:
            continue
        quota = parse_per_class_quota(run.rows[0].get("replay_per_class"))
        if not quota:
            continue
        drew = True
        style = _style(name)
        axis.plot(range(len(order)), [quota.get(n, 0) for n in order],
                  marker=style["marker"], color=style["color"], label=style["label"])
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(order, rotation=60, ha="right", fontsize=7)
    axis.set_xlabel("task-1 class, rarest first")
    axis.set_ylabel("exemplar objects at t2")
    axis.set_title("Replay allocation across classes (M = 400)")
    axis.grid(True, alpha=0.3)
    if drew:
        axis.legend(fontsize=8)
        written += _save(figure, out / "figure_e_replay_allocation", formats)
    plt.close(figure)

    # F: forgetting against anchor AP, with rarity visible as marker size
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    drew = False
    for name, run in runs.items():
        final = run.final_task
        after = run.per_task_ap.get(final or "", {})
        if not run.anchor_ap or not after:
            continue
        x, y, sizes = [], [], []
        for class_name in protocol.TASK1:
            start, end = run.anchor_ap.get(class_name), after.get(class_name)
            if start is None or end is None:
                continue
            x.append(start)
            y.append(start - end)
            sizes.append(12 + 34 * (math.log10(counts[class_name]) - 3.0))
        if not x:
            continue
        drew = True
        style = _style(name)
        axis.scatter(x, y, s=[max(8, s) for s in sizes], color=style["color"],
                     marker=style["marker"], label=style["label"], alpha=0.75)
    axis.set_xlabel("anchor AP50 (how well the class started)")
    axis.set_ylabel("final forgetting (AP50 points)")
    axis.set_title("Vulnerability: difficulty or rarity? (marker size = log frequency)")
    axis.grid(True, alpha=0.3)
    if drew:
        axis.legend(fontsize=8)
        written += _save(figure, out / "figure_f_forgetting_vs_anchor", formats)
    plt.close(figure)

    # a run that reached a band with no members reports None, and a group whose
    # membership changed is not a retention change — see docs/
    if not written:
        raise AnalysisError(
            "No run held enough data to draw anything. A chain that has not "
            "written per-class metrics yet produces tables but no figures."
        )
    return written
