"""One command that turns the replay runs into tomorrow's result chapter.

    python tools/compare_replay.py /path/to/work --out data/results/replay

``/path/to/work`` is the Drive workspace root that holds one directory per run
(``random__none``, ``random__uniform``, ``random__tail_favouring``). Any subset
works: a run that has not started is skipped, a run that stopped early is
compared as far as it got, and a missing quantity prints as ``—`` rather than
being invented.

It writes six tables in three forms each — CSV for further work, Markdown for
the thesis, LaTeX for the paper — plus a machine-readable ``summary.json`` and
six figures. Nothing here writes into a workspace; it only reads.

**The first thing it prints is the compatibility check**, because a table that
silently mixes a two-task sanity run with a five-task chain is worse than no
table. Read that section before the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import comparison
from owl.analysis import AnalysisError, to_latex, to_markdown, write_csv
from owl.runner import table

DEFAULT_OUT = ROOT / "data" / "results" / "replay"


def emit(rows, out: Path, stem: str, *, digits: int, caption: str | None = None) -> None:
    """One table, in the three forms the thesis needs. Empty tables are skipped."""

    if not rows:
        print(f"  ({stem}: nothing to report yet)")
        return
    write_csv(rows, out / f"{stem}.csv")
    (out / f"{stem}.md").write_text(to_markdown(rows, digits), encoding="utf-8")
    (out / f"{stem}.tex").write_text(
        to_latex(rows, digits, caption=caption, label=f"tab:{stem}"), encoding="utf-8")
    print(table(rows, digits=digits))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workspace", type=Path,
                        help="the workspace root, or a single run directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline", default="random__none",
                        help="the run the deltas are measured against (default: %(default)s)")
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--no-plots", action="store_true",
                        help="tables only; matplotlib is an optional dependency")
    arguments = parser.parse_args()

    try:
        runs = comparison.load_runs(arguments.workspace)
    except AnalysisError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    digits = arguments.digits

    # ---- what is on disk, and may it be compared at all? -------------------
    print(f"{'=' * 78}\nwhat the runs left behind\n{'=' * 78}")
    depth = comparison.depth_report(runs)
    print(table(depth, digits=digits))
    emit(depth, out, "depth", digits=digits, caption="Runs found")

    missing = [name for name in comparison.EXPECTED if name not in runs]
    if missing:
        print(f"\nnot present yet: {missing}")

    # ---- where do the per-class numbers come from, and may they be trusted? --
    print(f"\n{'=' * 78}\nper-class AP provenance\n{'=' * 78}")
    print("Source: coco_eval_bbox in each task's metrics.json — the evaluator's own")
    print("per-class AP50, read by owl.metrics.per_class_ap50. The metrics file has")
    print("no key named for a per-class table, so every vector is checked against the")
    print("aggregates the same file reports (previous/current_known_AP50, unknown_AP50).")
    provenance = []
    for name, run in runs.items():
        for task, report in sorted(run.per_class_checks.items()):
            provenance.append({
                "run": name, "task": task,
                "usable": report.get("usable"),
                "classes": report.get("n_classes"),
                "checks": ", ".join(
                    f"{c['quantity'].replace('_AP50', '')}"
                    f"{'=' if c['agrees'] else '!='}{c['rebuilt']:.4f}"
                    for c in report.get("checks", [])),
                "reason": report.get("reason") or "",
            })
    if provenance:
        print()
        print(table(provenance, digits=digits))
        emit(provenance, out, "per_class_provenance", digits=digits,
             caption="Per-class AP source and its validation")
    unusable = [r for r in provenance if not r["usable"]]
    if unusable:
        print("\n*** Some tasks' per-class vectors do not reproduce their own file's")
        print("*** aggregates. Table 4 for those tasks is NOT trustworthy; treat the")
        print("*** recall cross-check as the only per-class evidence there.")

    print(f"\n{'=' * 78}\ncompatibility — may these runs share a table?\n{'=' * 78}")
    clashes = comparison.compatibility(runs, reference=arguments.baseline)
    if clashes:
        print(table(clashes, digits=digits))
        print("\n*** These runs differ in settings that change what the numbers mean.")
        print("*** Comparing them would not be a replay comparison. Fix or exclude")
        print("*** the offending run before quoting anything below.")
        emit(clashes, out, "compatibility", digits=digits,
             caption="Result-affecting settings that differ between runs")
    else:
        print("Every run agrees on every result-affecting setting except replay_arm.")

    # ---- the six tables ----------------------------------------------------
    sections = [
        ("TABLE 1 — task-wise comparison", "table1_task_comparison",
         comparison.table_task_comparison(runs),
         "Retention, plasticity and group AP per task, by replay strategy"),
        (f"TABLE 2 — delta versus {arguments.baseline}", "table2_delta_vs_baseline",
         comparison.table_delta_versus_baseline(runs, baseline=arguments.baseline),
         "Each replay strategy minus the no-replay baseline"),
        ("TABLE 3 — tail-favouring versus uniform", "table3_tail_vs_uniform",
         comparison.table_tail_versus_uniform(runs),
         "The contribution's own comparison"),
        ("TABLE 4 — per-class retention", "table4_per_class",
         comparison.table_per_class(runs),
         "Per-class AP, forgetting and what replay bought, by class frequency"),
        ("TABLE 5 — replay composition", "table5_replay_composition",
         comparison.table_replay_composition(runs),
         "The exemplar memory that produced each task's numbers"),
        ("TABLE 6 — annotation and supervision cost", "table6_cost",
         comparison.table_cost(runs),
         "What each task was charged and what it trained on"),
    ]
    for title, stem, rows, caption in sections:
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        emit(rows, out, stem, digits=digits, caption=caption)

    # ---- is rarity the right proxy for vulnerability? ----------------------
    print(f"\n{'=' * 78}\nvulnerability: does rarity predict forgetting?\n{'=' * 78}")
    reports = {name: comparison.vulnerability(run) for name, run in runs.items()}
    for name, report in reports.items():
        if not report.get("available"):
            print(f"  {name}: not available — {report.get('reason')}")
            continue
        print(f"  {name} (at {report['task']}, n={report['n_classes']} classes)")
        print(f"    rho(forgetting, log frequency) = "
              f"{_show(report['rho_forgetting_vs_log_frequency'])}")
        print(f"    rho(forgetting, anchor AP)     = "
              f"{_show(report['rho_forgetting_vs_anchor_ap'])}")
        print("    mean forgetting by band        = "
              + ", ".join(f"{g}={_show(v)}" for g, v in report["group_means"].items()))
        for key in ("ols_frequency_only", "ols_anchor_only", "ols_frequency_and_anchor"):
            fit = report.get(key)
            if fit:
                print(f"    {key:26s} R^2={_show(fit['r_squared'])} "
                      f"coefficients={[round(c, 3) for c in fit['coefficients']]}")
    print("\n  One run, one seed: these are descriptive. No p-values, no significance.")

    # ---- machine-readable summary -----------------------------------------
    summary = {
        "runs": {name: {"selection_arm": run.selection_arm,
                        "replay_arm": run.replay_arm,
                        "tasks_finished": len(run),
                        "last_task": run.final_task,
                        "path": str(run.path)}
                 for name, run in runs.items()},
        "missing": missing,
        "compatibility_clashes": clashes,
        "vulnerability": reports,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                      encoding="utf-8")

    # ---- figures -----------------------------------------------------------
    if not arguments.no_plots:
        try:
            written = comparison.plot_figures(runs, out)
            print(f"\nfigures: {len(written)} files in {out}")
        except AnalysisError as error:
            print(f"\nno figures: {error}")

    print(f"\ntables, summary.json and figures written into {out}")
    return 0


def _show(value) -> str:
    return "—" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
