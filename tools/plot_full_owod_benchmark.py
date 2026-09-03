#!/usr/bin/env python
"""Publication-simple matplotlib figures from the summariser's CSVs.

Seven figures, one endpoint each, arms as lines and tasks on the x axis, plus the
annotation-efficiency curves. No styling beyond what makes a projected figure
readable, no second y axis, no arm hidden because its line is low.

A dry-run summary is stamped ``DRY RUN`` across every panel, so a stubbed figure
cannot be pasted into a slide by accident.

    python tools/plot_full_owod_benchmark.py --results <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl.active_selection import arms as arm_registry

TASKS = ("t2", "t3", "t4")

#: One figure per endpoint: (csv, column, title, y label).
PANELS = (
    ("per_task_metrics.csv", "known_mAP50", "Known-class mAP50 vs task", "mAP50"),
    ("per_task_metrics.csv", "new_class_AP50", "New-class AP50 vs task", "AP50"),
    ("per_task_metrics.csv", "U_Recall50", "U-Recall vs task", "U-Recall"),
    ("per_task_metrics.csv", "forgetting", "Forgetting caused by each task", "mAP50 points lost"),
    ("per_task_metrics.csv", "mAP50_tail", "Tail-class mAP50 vs task", "mAP50"),
    ("per_task_metrics.csv", "mAP50_head", "Head-class mAP50 vs task", "mAP50"),
    ("per_task_metrics.csv", "mAP50_medium", "Medium-class mAP50 vs task", "mAP50"),
    ("acquisition.csv", "acquired_classes", "Distinct classes acquired per task", "classes"),
    ("acquisition.csv", "acquired_new_class",
     "Objects of the task's own new class acquired", "objects"),
    ("acquisition.csv", "acquired_tail_objects", "Tail-class objects acquired", "objects"),
    ("supervision_cost.csv", "boxes_supervised",
     "Supervised boxes reaching the detector", "boxes"),
    ("supervision_cost.csv", "boxes_labelled",
     "Boxes the oracle labelled (the matched cost)", "boxes"),
)

EFFICIENCY = (
    ("known_mAP50", "Known-class mAP50"),
    ("new_class_AP50", "New-class AP50"),
    ("U_Recall50", "U-Recall"),
    ("mAP50_tail", "Tail-class mAP50"),
)


def read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row: dict, column: str) -> float | None:
    raw = row.get(column)
    if raw in (None, "", "—"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def stamp(axis, dry_run: bool) -> None:
    if dry_run:
        axis.text(0.5, 0.5, "DRY RUN", transform=axis.transAxes,
                  fontsize=40, color="red", alpha=0.25,
                  ha="center", va="center", rotation=20, zorder=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None, help="defaults to <results>/plots")
    parser.add_argument("--dpi", type=int, default=150)
    arguments = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = Path(arguments.results)
    out = Path(arguments.out) if arguments.out else results / "plots"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    dry_run = bool(manifest.get("dry_run"))

    cache = {name: read(results / name) for name in
             ("per_task_metrics.csv", "acquisition.csv", "supervision_cost.csv",
              "annotation_efficiency.csv")}
    written: list[Path] = []

    for source, column, title, ylabel in PANELS:
        rows = cache.get(source) or []
        series: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for row in rows:
            got = value(row, column)
            if got is None or row["task"] not in TASKS:
                continue
            key = (row["arm"], row.get("seed", "0"))
            series.setdefault(key, []).append((TASKS.index(row["task"]), got))
        if not series:
            continue
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        for arm in arm_registry.ORDER:
            for (name, seed), points in sorted(series.items()):
                if name != arm:
                    continue
                points.sort()
                axis.plot([p[0] for p in points], [p[1] for p in points],
                          marker="o", label=f"{name}" if seed in ("0", 0) else
                          f"{name} (seed {seed})")
        axis.set_xticks(range(len(TASKS)))
        axis.set_xticklabels([f"{t}\n{d}" for t, d in zip(
            TASKS, ("traffic light\n(head)", "fire hydrant\n(tail)", "stop sign\n(tail)")
        )], fontsize=8)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        stamp(axis, dry_run)
        figure.tight_layout()
        path = out / f"{column}.png"
        figure.savefig(path, dpi=arguments.dpi)
        plt.close(figure)
        written.append(path)

    rows = cache.get("annotation_efficiency.csv") or []
    if rows:
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
        for axis, (column, label) in zip(axes.ravel(), EFFICIENCY):
            for arm in arm_registry.ORDER:
                points = sorted(
                    (float(r["cumulative_answers"]), value(r, column))
                    for r in rows if r["arm"] == arm and value(r, column) is not None
                )
                if points:
                    axis.plot([p[0] for p in points], [p[1] for p in points],
                              marker="o", label=arm)
            axis.set_xlabel("cumulative oracle answers")
            axis.set_ylabel(label)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
            stamp(axis, dry_run)
        figure.suptitle("Annotation efficiency: endpoint against oracle answers", fontsize=12)
        figure.tight_layout()
        path = out / "annotation_efficiency.png"
        figure.savefig(path, dpi=arguments.dpi)
        plt.close(figure)
        written.append(path)

    print(f"wrote {len(written)} figures to {out}")
    for path in written:
        print(f"  {path.name}")
    if dry_run:
        print("Every figure is stamped DRY RUN: the detector was stubbed.")


if __name__ == "__main__":
    main()
