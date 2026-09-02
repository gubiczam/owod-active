#!/usr/bin/env python
"""Turn the twelve Method V3 trajectories into the tables the plan asks for.

Reads only ``result.json`` and ``selection_curve.csv`` from each trajectory
directory. Prints, in order:

1. one row per (arm, seed) with the detector metrics;
2. the primary contrast ``A*C`` vs ``A`` at budget 600, with the three paired
   differences;
3. the secondary contrasts ``U vs A``, ``A*C vs U``, ``random vs all``;
4. the per-budget selection curves at 100 .. 600 with their area under curve;
5. the mechanically generated verdict.

The verdict comes from :func:`owl.method_v3.evaluate_criterion` and from nowhere
else, so this tool cannot change what counts as a success.

    python tools/summarize_method_v3.py \\
        --results /content/drive/MyDrive/OWL/results/method_v3_selection_transfer
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import method_v3
from owl.runner import table

#: Detector columns, in report order. Every one is produced by owl.metrics from
#: PROB's own evaluator; nothing here is invented for this experiment.
DETECTOR_COLUMNS: tuple[str, ...] = (
    "known_mAP50", "prev_mAP50", "new_mAP50", "new_class_AP50",
    "mAP50_head", "mAP50_medium", "mAP50_tail", "mAP50_medium_tail",
    "forgetting", "drop_from_anchor",
    "U_Recall50", "U_Recall_head", "U_Recall_medium", "U_Recall_tail", "U_Recall_all",
)

#: Selection columns. Distinct objects unless the name says proposals.
SELECTION_COLUMNS: tuple[str, ...] = (
    "images_opened", "oracle_cost", "unknown_objects", "medium_tail_objects",
    "head_objects", "medium_objects", "tail_objects", "unknown_classes",
    "proposals_per_object", "background_share_of_selection",
)


def load(results: Path) -> list[dict]:
    rows: list[dict] = []
    for arm, seed in method_v3.trajectories():
        directory = results / method_v3.trajectory_name(arm, seed)
        found = method_v3.load_trajectory(directory)
        if found is not None:
            rows.append(found)
            continue
        status = directory / "status.json"
        state = "missing"
        if status.exists():
            import json
            state = json.loads(status.read_text(encoding="utf-8")).get("status", "?")
        print(f"  [{arm} seed{seed}] {state} — not included")
    return rows


def curves(results: Path) -> list[dict]:
    rows: list[dict] = []
    for arm, seed in method_v3.trajectories():
        path = results / method_v3.trajectory_name(arm, seed) / "selection_curve.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "arm": arm, "seed": seed, "budget": int(row["budget"]),
                    "unknown_objects": int(row["unknown_objects"]),
                    "medium_objects": int(row["medium_objects"]),
                    "tail_objects": int(row["tail_objects"]),
                    "medium_tail_objects": int(row["medium_objects"])
                    + int(row["tail_objects"]),
                    "unknown_classes": int(row["unknown_classes"]),
                    "images_opened": int(row["images_opened"]),
                    "selected_background": int(row["selected_background"]),
                })
    return rows


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.fmean(values), statistics.stdev(values)


def _by_arm(rows: list[dict], column: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for row in rows:
        if column in row and row[column] is not None:
            out.setdefault(str(row["arm"]), {})[int(row["seed"])] = float(row[column])
    return out


def contrast(rows: list[dict], treatment: str, control: str, column: str) -> dict:
    """One arm against another on one column, with the paired seed differences."""

    series = _by_arm(rows, column)
    left, right = series.get(treatment, {}), series.get(control, {})
    shared = sorted(set(left) & set(right))
    paired = {seed: left[seed] - right[seed] for seed in shared}
    mean_left, sd_left = _mean_sd([left[s] for s in shared])
    mean_right, sd_right = _mean_sd([right[s] for s in shared])
    return {
        "metric": column,
        f"{control} mean±sd": f"{mean_right:.2f}±{sd_right:.2f}",
        f"{treatment} mean±sd": f"{mean_left:.2f}±{sd_left:.2f}",
        "paired delta": (
            f"{statistics.fmean(paired.values()):+.2f}" if paired else "—"
        ),
        "per-seed delta": " ".join(f"s{s}:{paired[s]:+.2f}" for s in shared) or "—",
        "seeds improving": sum(1 for value in paired.values() if value > 0),
        "n": len(shared),
    }


def area_under_curve(marks: list[int], values: list[float]) -> float:
    """Trapezoidal area over the budget axis, normalised by the budget span.

    Well defined here because every arm reports at the same six marks. Reported
    for the **selection** curve only: the detector endpoint exists at budget 600
    alone, so a detector AULC would be an interpolation of one point.
    """

    if len(marks) < 2:
        return float("nan")
    total = 0.0
    for (x0, y0), (x1, y1) in zip(zip(marks, values), zip(marks[1:], values[1:])):
        total += (x1 - x0) * (y0 + y1) / 2.0
    return total / (marks[-1] - marks[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None,
                        help="directory for the CSV/JSON copies (default: --results)")
    arguments = parser.parse_args()

    results = Path(arguments.results)
    out = Path(arguments.out) if arguments.out else results
    rows = load(results)
    curve_rows = curves(results)

    print("=" * 78)
    print("METHOD V3 — SELECTION -> LEARNING TRANSFER (exploratory/prospective)")
    print("=" * 78)
    stubbed = (results / "DRY_RUN").exists() or any(
        row.get("dry_run") for row in rows
    )
    if stubbed:
        for _ in range(3):
            print("!!! DRY RUN — PROB WAS STUBBED OUT. EVERY DETECTOR NUMBER "
                  "AND THE VERDICT BELOW ARE SYNTHETIC AND MEAN NOTHING. !!!")
    print("Method V2 Stage 2 stands unchanged: D_NO_GO, R_NO_GO, C_GO, ladder = U.")
    print(f"trajectories complete: {len(rows)} of {len(method_v3.trajectories())}")
    if rows:
        tail_band = rows[0].get("known_tail_classes") or []
        print(f"tail band at this task: {tail_band}"
              + ("  — a SINGLE class; reported as one class, never as "
                 "'tail classes'" if len(tail_band) == 1 else ""))
        print(f"medium+tail classes:    {rows[0].get('known_medium_tail_classes')}")
        print("selection determinism:  A, U and A*C are static rankings, so their "
              "600 picks are identical at every seed. The three paired differences "
              "therefore measure training and replay-draw noise on two FIXED "
              "selections, not selection variance. random varies with the seed.")
    print()

    if not rows:
        print("No complete trajectory yet. Run tools/run_method_v3.py.")
        return

    # ---- 1. per-trajectory ------------------------------------------------
    print("PER-TRAJECTORY DETECTOR METRICS  (budget 600)")
    detector = [
        {"arm": row["arm"], "seed": row["seed"]}
        | {name: row.get(name) for name in DETECTOR_COLUMNS}
        for row in rows
    ]
    print(table(detector))
    print()
    print("PER-TRAJECTORY SELECTION / ORACLE COVERAGE  (explanatory)")
    coverage = [
        {"arm": row["arm"], "seed": row["seed"]}
        | {name: row.get(name) for name in SELECTION_COLUMNS}
        for row in rows
    ]
    print(table(coverage))
    print()

    # ---- 2. the primary contrast ------------------------------------------
    print("=" * 78)
    print(f"PRIMARY CONTRAST: {method_v3.CRITERION.treatment} - "
          f"{method_v3.CRITERION.control} @ budget {method_v3.CRITERION.budget}")
    print("=" * 78)
    primary_metrics = (
        "mAP50_medium_tail", "mAP50_medium", "mAP50_tail",
        "new_mAP50", "new_class_AP50", "U_Recall_all", "known_mAP50",
        "medium_tail_objects", "unknown_objects", "background_share_of_selection",
    )
    print(table([
        contrast(rows, method_v3.CRITERION.treatment,
                 method_v3.CRITERION.control, column)
        for column in primary_metrics
    ]))
    print()

    # ---- 3. the secondary contrasts ---------------------------------------
    print("SECONDARY CONTRASTS")
    for treatment, control in (("U", "A"), ("A*C", "U"),
                               ("A", "random"), ("U", "random"), ("A*C", "random")):
        print(f"\n  {treatment} vs {control}")
        print(table([
            contrast(rows, treatment, control, column)
            for column in ("mAP50_medium_tail", "known_mAP50", "new_mAP50",
                           "U_Recall_all", "unknown_objects")
        ]))
    print()

    # ---- 4. the per-budget selection curves -------------------------------
    print("=" * 78)
    print("PER-BUDGET SELECTION CURVES  (selection coverage, NOT a detector curve)")
    print("The detector endpoint exists at budget 600 only; see protocol §7.")
    print("=" * 78)
    for column in ("medium_tail_objects", "unknown_objects", "unknown_classes",
                   "images_opened"):
        print(f"\n  {column}  (mean over seeds)")
        grid = []
        for arm in method_v3.ARMS:
            row: dict[str, object] = {"arm": arm}
            series: list[float] = []
            for mark in method_v3.BUDGET_MARKS:
                values = [
                    float(entry[column]) for entry in curve_rows
                    if entry["arm"] == arm and entry["budget"] == mark
                ]
                mean = statistics.fmean(values) if values else float("nan")
                row[str(mark)] = mean
                series.append(mean)
            row["AULC"] = area_under_curve(list(method_v3.BUDGET_MARKS), series)
            grid.append(row)
        print(table(grid))
    print()

    # ---- 5. the verdict ---------------------------------------------------
    print("=" * 78)
    print("FROZEN CRITERION")
    print("=" * 78)
    print(method_v3.CRITERION.statement())
    print()
    try:
        verdict = method_v3.evaluate_criterion(rows)
    except method_v3.MethodV3Error as error:
        print(f"VERDICT NOT COMPUTED: {error}")
        return
    for name, ok in verdict.clauses.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    detail = verdict.detail
    print("  paired differences: "
          + " ".join(f"s{s}:{v:+.3f}" for s, v in detail["paired_differences"].items()))
    print(f"  {method_v3.CRITERION.control} mean {detail['primary_mean_control']:.3f} "
          f"± {detail['primary_sd_control']:.3f} | "
          f"{method_v3.CRITERION.treatment} mean {detail['primary_mean_treatment']:.3f} "
          f"± {detail['primary_sd_treatment']:.3f}")
    print(f"  guard {method_v3.CRITERION.guard_metric} delta "
          f"{detail['guard_delta']:+.3f} "
          f"(tolerance {method_v3.CRITERION.guard_tolerance:g})")
    print()
    print("=" * 78)
    print(verdict.label)
    print("=" * 78)
    print("n = 3 seeds. This is a descriptive criterion, not a significance test; "
          "no p-value is claimed.")
    if not verdict.positive:
        print(f"failed clauses: {', '.join(verdict.failed_clauses())}")

    method_v3.write_rows(out / "method_v3_trajectories.csv", detector)
    method_v3.write_rows(out / "method_v3_coverage.csv", coverage)
    if curve_rows:
        method_v3.write_rows(out / "method_v3_selection_curve.csv", curve_rows)
    method_v3.write_json(out / "method_v3_summary.json", {
        "dry_run": bool(stubbed),
        "verdict": verdict.label,
        "clauses": dict(verdict.clauses),
        "detail": dict(verdict.detail),
        "trajectories_complete": len(rows),
        "trajectories_expected": len(method_v3.trajectories()),
    })
    print(f"\nwrote {out / 'method_v3_summary.json'}")


if __name__ == "__main__":
    main()
