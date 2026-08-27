"""Reading a finished chain back off disk, and answering the plan's question.

The GPU chain writes one ``results_<arm>.csv`` per arm into its workspace and
prints its tables into the Colab session. A session is not a result chapter: it
dies with the runtime, its tables are plain text, and the plan's headline
endpoint is not a table at all but a *curve* —

    "a tail-U-Recall mérőszámokat értékelem ki, mint az orákulum-költség
    függvénye […] a várt tendencia, hogy az eloszlás-tudatos kiválasztás azonos
    tail-szintet lényegesen kevesebb annotációból ér el."

So this module reads the CSVs and produces three things: the curve, the
comparison at equal cost, and the number the sentence above actually predicts —
how much annotation each arm needed to reach the *same* tail level.

One correction happens here rather than in the runner. ``oracle_cost_so_far`` is
written as ``(task index + 1) * budget_per_task``, which assumes every task spent
its whole budget. :func:`owl.selection.select` caps each round's quota at the
candidates still available, and the ``exclude`` mask grows with every task, so a
depleted pool spends less than the budget — and depletes differently per arm,
because the arms buy different regions. That is the x-axis of the headline
result, so it cannot be assumed. The regions actually paid for are in the
``asked`` column; its cumulative sum is the honest cost. Both are reported, and
:func:`cost_discrepancy` says whether they differ at all on a given run.

Nothing here writes into a chain's workspace, and nothing here is imported by
the runner or the notebook: this runs after the GPU is done, on a laptop.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: The metric the research plan is about. Everything else is a diagnostic.
HEADLINE = "U_Recall_tail"

#: Recall columns the chain writes, in the order the tables should show them.
RECALL_COLUMNS: tuple[str, ...] = (
    "U_Recall_tail", "U_Recall_medium", "U_Recall_head", "U_Recall_all",
)


class AnalysisError(ValueError):
    """Raised when the CSVs cannot answer the question that was asked of them."""


# ------------------------------------------------------------------ reading ---


@dataclass(frozen=True)
class Arm:
    """One arm's chain, as it was written to disk, one row per finished task."""

    name: str
    rows: list[dict]
    path: Path

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> list:
        """The column, or a list of ``None`` if the chain never wrote it."""

        return [row.get(name) for row in self.rows]


def _number(text: object) -> object:
    """CSV holds text. Empty, ``—`` and ``None`` mean the chain wrote nothing.

    A missing recall is not a zero: ``unknown_recall_by_group`` returns ``None``
    for a group with no objects in the test set, and plotting that as 0.0 would
    invent a measurement.
    """

    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped in ("", "—", "None", "nan"):
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def read_rows(path: str | Path) -> list[dict]:
    """One ``results_<arm>.csv``, with its numbers parsed."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{k: _number(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise AnalysisError(f"{path} has a header and no rows: no task finished.")
    return rows


def load_arms(root: str | Path) -> dict[str, Arm]:
    """Every arm under ``root``, keyed by arm name.

    The chain writes ``<workspace>/<arm>/results_<arm>.csv``, so the search is
    recursive and the arm name comes from the file, not from the directory —
    a workspace copied under a new name still reports the arm it actually ran.
    """

    root = Path(root)
    if root.is_file():
        found = [root]
    elif root.is_dir():
        found = sorted(root.glob("**/results_*.csv"))
    else:
        raise AnalysisError(f"{root} does not exist.")
    if not found:
        raise AnalysisError(
            f"No results_*.csv under {root}. The chain writes one per arm into "
            "<workspace>/<arm>/; download that directory from Drive, not the "
            "notebook output."
        )

    arms: dict[str, Arm] = {}
    for path in found:
        name = path.stem[len("results_"):] or path.parent.name
        if name in arms:
            raise AnalysisError(
                f"Two files claim to be arm {name!r}: {arms[name].path} and {path}. "
                "Each configuration needs its own workspace."
            )
        arms[name] = Arm(name=name, rows=read_rows(path), path=path)
    return {name: arms[name] for name in sorted(arms, key=_arm_order)}


def _arm_order(name: str) -> tuple[int, str]:
    """Arms in the order :data:`owl.selection.ARMS` declares them.

    Any fixed order would do; what matters is that every table and the figure
    list the arms the same way, so a reader comparing two of them is not
    silently comparing different columns. Alphabetical order would change as
    soon as an arm is renamed, and the glob's order changes with the directory
    layout. A name the scorer does not know sorts last, alphabetically.
    """

    from owl import selection

    order = list(selection.ARMS)
    return (order.index(name), "") if name in order else (len(order), name)


# --------------------------------------------------------------------- cost ---


def real_cost(arm: Arm) -> list[float]:
    """Cumulative regions the oracle was actually asked about, per task.

    This is the x-axis. It is a cumulative sum of ``asked`` rather than a
    multiple of the budget, because a task whose candidate pool ran dry spent
    less than the budget and the difference is invisible in the recorded column.
    """

    asked = arm.column("asked")
    if any(value is None for value in asked):
        raise AnalysisError(
            f"Arm {arm.name!r} has no 'asked' column, so the real oracle cost "
            f"cannot be recovered from {arm.path}."
        )
    running, out = 0.0, []
    for value in asked:
        running += float(value)
        out.append(running)
    return out


def recorded_cost(arm: Arm) -> list[float | None]:
    """What the chain wrote as ``oracle_cost_so_far``: budget × task index."""

    return [None if v is None else float(v) for v in arm.column("oracle_cost_so_far")]


def cost_discrepancy(arms: Mapping[str, Arm]) -> list[dict]:
    """Per task, the recorded cost against the real one. Empty if they agree.

    An empty result means the pool never ran dry and the recorded x-axis was
    right all along — which is the outcome to hope for, and the reason this is
    reported rather than silently corrected.
    """

    out: list[dict] = []
    for name, arm in arms.items():
        for row, recorded, real in zip(arm.rows, recorded_cost(arm), real_cost(arm)):
            if recorded is None or abs(recorded - real) < 1e-9:
                continue
            out.append({
                "arm": name, "task": row.get("task"),
                "recorded": recorded, "real": real, "difference": recorded - real,
                "asked": row.get("asked"),
            })
    return out


# -------------------------------------------------------------------- curve ---


def curve(arm: Arm, metric: str = HEADLINE, *, cost: str = "real") -> list[tuple[float, float]]:
    """``(oracle cost, metric)`` for every task that measured the metric.

    Tasks where the metric is missing are dropped, not zero-filled: a frequency
    group with no objects in the shared test set has no recall to report.
    """

    costs = real_cost(arm) if cost == "real" else recorded_cost(arm)
    points = [
        (float(c), float(row[metric]))
        for c, row in zip(costs, arm.rows)
        if c is not None and row.get(metric) is not None
    ]
    return points


def cost_to_reach(points: Sequence[tuple[float, float]], level: float) -> float | None:
    """The oracle cost at which the curve first reaches ``level``.

    Linearly interpolated between the two bracketing measurements. ``None`` when
    the arm never reaches the level — which is itself the answer to the plan's
    question, and must not be reported as a large number.
    """

    previous: tuple[float, float] | None = None
    for cost, value in points:
        if value >= level:
            if previous is None or value == previous[1]:
                return cost
            span = value - previous[1]
            return previous[0] + (level - previous[1]) / span * (cost - previous[0])
        previous = (cost, value)
    return None


# ------------------------------------------------------------------- tables ---


def per_arm_table(arm: Arm, metric: str = HEADLINE) -> list[dict]:
    """What one arm bought, task by task. The denominator is in the table.

    ``unknown_objects_tail`` is there because a recall computed over three
    objects is not a result, and the only way to see that is to print it.
    """

    rows = []
    for row, real in zip(arm.rows, real_cost(arm)):
        out = {
            "task": row.get("task"),
            "new_class": row.get("new_class"),
            "asked": row.get("asked"),
            "oracle_cost": real,
        }
        for column in RECALL_COLUMNS:
            if any(r.get(column) is not None for r in arm.rows):
                out[column] = row.get(column)
        objects = f"{metric.replace('U_Recall_', 'unknown_objects_')}"
        if any(r.get(objects) is not None for r in arm.rows):
            out[objects] = row.get(objects)
        for column in ("known_mAP50", "new_mAP50", "forgetting"):
            if any(r.get(column) is not None for r in arm.rows):
                out[column] = row.get(column)
        rows.append(out)
    return rows


def comparison(arms: Mapping[str, Arm], metric: str = HEADLINE) -> list[dict]:
    """The arms side by side, to the depth every one of them reached.

    A longer chain against a shorter one is not a comparison, so the table stops
    where the shortest arm stopped; :func:`depth_report` says what was cut.
    """

    if not arms:
        return []
    depth = min(len(arm) for arm in arms.values())
    rows = []
    costs = {name: real_cost(arm) for name, arm in arms.items()}
    for index in range(depth):
        first = next(iter(arms.values()))
        row: dict = {
            "task": first.rows[index].get("task"),
            "new_class": first.rows[index].get("new_class"),
        }
        for name, arm in arms.items():
            row[name] = arm.rows[index].get(metric)
            row[f"{name}_cost"] = costs[name][index]
        rows.append(row)
    return rows


def efficiency(
    arms: Mapping[str, Arm], *, reference: str, metric: str = HEADLINE
) -> list[dict]:
    """The plan's prediction, stated as a number: cost to reach the same level.

    For every tail level the reference arm reached, how much annotation each arm
    needed to get there. "Lényegesen kevesebb annotációból" is a claim about
    this column and no other: an arm that reaches the same level at a lower cost
    has demonstrated it, and one that never reaches it has not.
    """

    if reference not in arms:
        raise AnalysisError(
            f"No arm named {reference!r} in {sorted(arms)}; --reference names the "
            "baseline the others are measured against."
        )
    curves = {name: curve(arm, metric) for name, arm in arms.items()}
    reference_curve = curves[reference]
    if not reference_curve:
        raise AnalysisError(
            f"The reference arm {reference!r} never measured {metric}: nothing to "
            "compare against."
        )

    rows = []
    for _, level in reference_curve:
        row: dict = {"level": level}
        for name, points in curves.items():
            reached = cost_to_reach(points, level)
            row[name] = reached
            # A first measurement already at or above the level bounds the cost
            # from above; the chain never looked below it, so it is not evidence
            # that less would have done.
            row[f"{name}_bounded"] = bool(points) and points[0][1] >= level
        baseline = row.get(reference)
        for name in curves:
            if name == reference:
                continue
            reached = row.get(name)
            row[f"{name}_saving"] = (
                None if reached is None or baseline is None else baseline - reached
            )
        rows.append(row)
    return rows


def depth_report(arms: Mapping[str, Arm]) -> list[dict]:
    """How far each arm got, and what the comparison had to drop."""

    if not arms:
        return []
    depth = min(len(arm) for arm in arms.values())
    return [
        {
            "arm": name,
            "tasks_finished": len(arm),
            "oracle_cost": real_cost(arm)[-1] if len(arm) else 0.0,
            "tasks_dropped_from_comparison": len(arm) - depth,
            "last_task": arm.rows[-1].get("task") if len(arm) else None,
        }
        for name, arm in arms.items()
    ]


# ------------------------------------------------------------------ writing ---


def _show(value: object, digits: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _columns(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return list(dict.fromkeys(key for row in rows for key in row))


def to_markdown(rows: Sequence[Mapping[str, object]], digits: int = 2) -> str:
    """A table the thesis can take verbatim."""

    if not rows:
        return "*(empty)*"
    columns = _columns(rows)
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_show(row.get(c), digits) for c in columns) + " |")
    return "\n".join(lines)


def to_latex(rows: Sequence[Mapping[str, object]], digits: int = 2,
             caption: str | None = None, label: str | None = None) -> str:
    """The same table as a ``tabular``. Underscores in headers are escaped."""

    if not rows:
        return "% (empty)"
    columns = _columns(rows)

    def escape(text: object) -> str:
        # An em dash is what the tables print for a missing value, and it is not
        # in a default LaTeX font encoding.
        return str(text).replace("_", r"\_").replace("—", "--")

    body = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline",
            " & ".join(escape(c) for c in columns) + r" \\", r"\hline"]
    for row in rows:
        body.append(" & ".join(escape(_show(row.get(c), digits)) for c in columns) + r" \\")
    body += [r"\hline", r"\end{tabular}"]
    if caption or label:
        body = ([r"\begin{table}[h]", r"\centering"] + body
                + ([rf"\caption{{{escape(caption)}}}"] if caption else [])
                + ([rf"\label{{{label}}}"] if label else [])
                + [r"\end{table}"])
    return "\n".join(body)


def write_csv(rows: Sequence[Mapping[str, object]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_columns(rows))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_curves(
    arms: Mapping[str, Arm], path: str | Path, *, metric: str = HEADLINE,
    formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    """The figure the plan asks for: ``metric`` against real oracle cost.

    matplotlib is an optional dependency (``pip install -e '.[plots]'``) because
    nothing else in the repository draws anything, and the tables have to be
    obtainable without it.
    """

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:                                # pragma: no cover
        raise AnalysisError(
            "matplotlib is not installed; run \"pip install -e '.[plots]'\" or pass "
            "--no-plot to get the tables only."
        ) from error

    drawn = {name: curve(arm, metric) for name, arm in arms.items()}
    drawn = {name: points for name, points in drawn.items() if points}
    if not drawn:
        raise AnalysisError(
            f"No arm measured {metric}, so there is no curve to draw. A chain run "
            "with measure_grouped_recall=False writes no grouped recall."
        )

    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    for name, points in drawn.items():
        axes.plot([c for c, _ in points], [v for _, v in points],
                  marker="o", label=name)
    axes.set_xlabel("oracle cost — regions actually annotated (cumulative)")
    axes.set_ylabel(f"{metric.replace('U_Recall_', 'U-Recall ')} (%)")
    group = metric.replace("U_Recall_", "")
    axes.set_title(f"Unknown recall on the {group} classes against annotation cost")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        target = path.with_suffix(f".{suffix}")
        figure.savefig(target, dpi=200)
        written.append(target)
    plt.close(figure)
    return written
