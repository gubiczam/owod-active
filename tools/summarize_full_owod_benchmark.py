#!/usr/bin/env python
"""Turn Benchmark V1's per-trajectory artefacts into the tables and CSVs.

Reads only what is on disk, changes nothing, and refuses to invent a single
scalar ranking of the arms. Six tables, in the order the supervisor material
uses them:

1. **detector, per task** — the endpoint. known / previous / new-class mAP50,
   U-Recall, WI, A-OSE, forgetting, and the head/medium/tail decomposition.
2. **annotation cost** — what each arm was charged and what the detector
   actually received from it. This is the table Method V3 did not have, and
   without which its AP differences could not be interpreted.
3. **acquisition** — what was bought, and at which later task it becomes
   learnable.
4. **across the chain** — means over t2–t4 and cumulative forgetting.
5. **the declared contrasts** — the primary one, the gate ablation, and every
   arm against the reference. Nothing else is presented as a headline.
6. **annotation efficiency** — each endpoint against cumulative oracle answers.

    python tools/summarize_full_owod_benchmark.py --results <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import metrics, protocol, runner
from owl.active_selection import arms as arm_registry
from owl.active_selection import benchmark as bm

TASKS = ("t2", "t3", "t4")

DETECTOR_COLUMNS = (
    "known_mAP50", "prev_mAP50", "new_mAP50", "new_class_AP50", "U_Recall50",
    "forgetting", "drop_from_anchor", "mAP50_head", "mAP50_medium", "mAP50_tail",
    "U_Recall_head", "U_Recall_medium", "U_Recall_tail", "exchange_rate",
)
COST_COLUMNS = (
    "answers_spent", "answers_unspent", "images_opened", "answers_per_image",
    "images_trainable", "images_barren", "images_from_earlier_tasks",
    "boxes_labelled", "boxes_supervised", "boxes_banked", "supervised_share",
    "boxes_supervised_head", "boxes_supervised_medium", "boxes_supervised_tail",
    # what PROB was actually handed, which is not what the task bought
    "boxes_trained_on", "boxes_trained_on_head", "boxes_trained_on_medium",
    "boxes_trained_on_tail",
    "training_images", "training_iterations",
)
ACQUISITION_COLUMNS = (
    "acquired_objects", "acquired_classes", "acquired_new_class",
    "acquired_known_now", "acquired_becomes_known_t3", "acquired_becomes_known_t4",
    "acquired_stays_unknown", "acquired_head_objects", "acquired_medium_objects",
    "acquired_tail_objects",
)


def number(value):
    if value in (None, "", "—"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_trajectory(directory: Path) -> list[dict]:
    path = directory / "results.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [{k: number(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def per_class_rows(results: Path, trajectory: str, arm: str, seed: object) -> list[dict]:
    """Per-class AP50 at every task, from each task's own metrics file."""

    groups = protocol.load_groups()
    chain = {task.name: task for task in bm.chain()}
    rows: list[dict] = []
    for directory in sorted((results / trajectory).glob("t*_*")):
        task_name = directory.name.split("_", 1)[0]
        path = directory / "metrics.json"
        if task_name not in chain or not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            per_class = metrics.per_class_ap50(
                payload, class_names=chain[task_name].known_classes
            )
        except metrics.MetricsError:
            continue
        for name, value in per_class.items():
            rows.append({
                "trajectory": trajectory, "arm": arm, "seed": seed,
                "task": task_name, "class_name": name,
                "group": groups.get(name, "—"),
                "introduced": "new" if name == chain[task_name].new_class else "old",
                "AP50": value,
            })
    return rows


def mean(values) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.fmean(clean), 3) if clean else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None, help="defaults to --results")
    parser.add_argument("--digits", type=int, default=2)
    arguments = parser.parse_args()

    results = Path(arguments.results)
    out = Path(arguments.out) if arguments.out else results
    manifest_path = results / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path} is missing; nothing to summarise.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("dry_run"):
        banner = "!" * 78
        line = "  DRY RUN — PROB AND DINOv2 WERE STUBBED. NO NUMBER BELOW IS A RESULT."
        for _ in range(3):
            print(banner)
            print(line)
            print(banner)

    print("=" * 78)
    print("FULL OWOD ACTIVE SELECTION BENCHMARK V1 — results")
    print("=" * 78)
    print(manifest["endpoint_statement"])
    print("\nChain (one class per task; NOT the published S-OWODB split):")
    print(runner.table([
        {"task": t["task"], "declares": t["new_class"] or "— (anchor)",
         "known_after": t["known_after"], "tail_band": ", ".join(t["tail_band"])}
        for t in manifest["chain"]
    ], digits=arguments.digits))

    loaded: dict[str, list[dict]] = {}
    order: list[dict] = []
    for entry in manifest["trajectories"]:
        rows = read_trajectory(results / entry["trajectory"])
        if not rows:
            continue
        loaded[entry["trajectory"]] = rows
        order.append(entry)
    if not loaded:
        raise SystemExit("no trajectory produced a results.csv.")

    incomplete = [e for e in manifest["trajectories"] if e["status"] != "COMPLETE"]
    if incomplete:
        print("\nNOT COMPLETE — reported as such, never as a result:")
        print(runner.table([
            {k: v for k, v in e.items() if k != "tasks"} for e in incomplete
        ], digits=arguments.digits))

    # ---- 1. the detector, per task ------------------------------------
    detector: list[dict] = []
    for entry in order:
        for row in loaded[entry["trajectory"]]:
            detector.append(
                {"arm": entry["arm"], "seed": entry["seed"], "task": row["task"],
                 "declares": row.get("new_class")}
                | {c: row.get(c) for c in DETECTOR_COLUMNS}
            )
    print("\n[1] DETECTOR, PER TASK")
    print(runner.table(detector, digits=arguments.digits))
    write_csv(out / "per_task_metrics.csv", detector)

    # ---- 2. annotation cost --------------------------------------------
    cost: list[dict] = []
    for entry in order:
        for row in loaded[entry["trajectory"]]:
            cost.append(
                {"arm": entry["arm"], "seed": entry["seed"], "task": row["task"]}
                | {c: row.get(c) for c in COST_COLUMNS}
            )
    print("\n[2] ANNOTATION COST AND WHAT REACHED THE DETECTOR")
    print(runner.table(cost, digits=arguments.digits))
    write_csv(out / "supervision_cost.csv", cost)
    spread = [r["boxes_labelled"] for r in cost if isinstance(r.get("boxes_labelled"), float)]
    if spread:
        print(f"  boxes_labelled across all arms and tasks: {min(spread):.0f} – "
              f"{max(spread):.0f} (ratio {max(spread) / max(min(spread), 1):.2f}x). "
              "Method V3's region budget gave 2.09x.")
    trained = [r["boxes_trained_on"] for r in cost
               if isinstance(r.get("boxes_trained_on"), float)]
    if trained:
        print(f"  boxes_trained_on (what PROB was handed, banked labels "
              f"included): {min(trained):.0f} – {max(trained):.0f} "
              f"(ratio {max(trained) / max(min(trained), 1):.2f}x)")
    supervised = [r["boxes_supervised"] for r in cost
                  if isinstance(r.get("boxes_supervised"), float)]
    if supervised:
        print(f"  boxes_supervised: {min(supervised):.0f} – {max(supervised):.0f} "
              f"(ratio {max(supervised) / max(min(supervised), 1):.2f}x) — the "
              "residual of protocol section 4, an outcome and not a matched quantity.")

    # ---- 3. acquisition -------------------------------------------------
    acquisition: list[dict] = []
    for entry in order:
        for row in loaded[entry["trajectory"]]:
            acquisition.append(
                {"arm": entry["arm"], "seed": entry["seed"], "task": row["task"],
                 "declares": row.get("new_class")}
                | {c: row.get(c) for c in ACQUISITION_COLUMNS}
            )
    print("\n[3] ACQUISITION, AND WHEN IT BECOMES LEARNABLE")
    print(runner.table(acquisition, digits=arguments.digits))
    write_csv(out / "acquisition.csv", acquisition)

    # ---- 4. across the chain --------------------------------------------
    chain_rows: list[dict] = []
    for entry in order:
        rows = loaded[entry["trajectory"]]
        chain_rows.append({
            "arm": entry["arm"], "seed": entry["seed"],
            "tasks": len(rows),
            "mean_known_mAP50": mean([r.get("known_mAP50") for r in rows]),
            "final_known_mAP50": rows[-1].get("known_mAP50"),
            "mean_new_class_AP50": mean([r.get("new_class_AP50") for r in rows]),
            "mean_U_Recall50": mean([r.get("U_Recall50") for r in rows]),
            "cumulative_forgetting": mean([r.get("forgetting") for r in rows]),
            "final_mAP50_tail": rows[-1].get("mAP50_tail"),
            "total_answers": sum(
                r.get("answers_spent") or 0 for r in rows),
            "total_boxes_supervised": sum(
                r.get("boxes_supervised") or 0 for r in rows),
            "distinct_classes_acquired_t2": rows[0].get("acquired_classes"),
        })
    print("\n[4] ACROSS THE CHAIN (t2–t4)")
    print(runner.table(chain_rows, digits=arguments.digits))
    write_csv(out / "chain_summary.csv", chain_rows)

    # ---- forgetting, its own file ---------------------------------------
    forgetting = [
        {"arm": e["arm"], "seed": e["seed"], "task": r["task"],
         "prev_mAP50": r.get("prev_mAP50"), "forgetting": r.get("forgetting"),
         "drop_from_anchor": r.get("drop_from_anchor"),
         "exchange_rate": r.get("exchange_rate")}
        for e in order for r in loaded[e["trajectory"]]
    ]
    write_csv(out / "forgetting.csv", forgetting)

    per_class: list[dict] = []
    for entry in order:
        per_class.extend(per_class_rows(
            results, entry["trajectory"], entry["arm"], entry["seed"]
        ))
    write_csv(out / "per_class_ap.csv", per_class)

    # ---- 5. the declared contrasts --------------------------------------
    endpoints = manifest["endpoints"]
    by_arm = {(e["arm"], e["seed"]): loaded[e["trajectory"]] for e in order}
    seeds = sorted({e["seed"] for e in order})

    def endpoint(arm: str, seed, metric: str, task: str | None):
        rows = by_arm.get((arm, seed))
        if not rows:
            return None
        if task is None:
            return mean([r.get(metric) for r in rows])
        for row in rows:
            if row["task"] == task:
                return row.get(metric)
        return None

    contrasts: list[dict] = []
    pairs = [
        ("primary", tuple(endpoints["primary_contrast"])),
        ("gate ablation", tuple(endpoints["ablation_contrast"])),
    ] + [
        ("vs reference", (arm, endpoints["reference_arm"]))
        for arm in arm_registry.ORDER if arm != endpoints["reference_arm"]
    ]
    watched = [
        (endpoints["primary_metric"], endpoints["primary_task"]),
        (endpoints["longtail_metric"], endpoints["primary_task"]),
        ("new_class_AP50", None),
        ("U_Recall50", None),
        (endpoints["acquisition_metric"], "t2"),
    ]
    for label, (treatment, control) in pairs:
        for metric, task in watched:
            for seed in seeds:
                a = endpoint(treatment, seed, metric, task)
                b = endpoint(control, seed, metric, task)
                if a is None or b is None:
                    continue
                contrasts.append({
                    "contrast": label, "treatment": treatment, "control": control,
                    "metric": metric, "at": task or "mean t2–t4", "seed": seed,
                    "treatment_value": a, "control_value": b,
                    "difference": round(float(a) - float(b), 3),
                })
    print("\n[5] THE DECLARED CONTRASTS")
    print(runner.table(contrasts, digits=arguments.digits))
    write_csv(out / "contrasts.csv", contrasts)

    # ---- 6. annotation efficiency ---------------------------------------
    efficiency: list[dict] = []
    for entry in order:
        spent = 0.0
        for row in loaded[entry["trajectory"]]:
            spent += row.get("answers_spent") or 0
            efficiency.append({
                "arm": entry["arm"], "seed": entry["seed"], "task": row["task"],
                "cumulative_answers": spent,
                "known_mAP50": row.get("known_mAP50"),
                "new_class_AP50": row.get("new_class_AP50"),
                "U_Recall50": row.get("U_Recall50"),
                "mAP50_tail": row.get("mAP50_tail"),
                "acquired_classes": row.get("acquired_classes"),
            })
    print("\n[6] ANNOTATION EFFICIENCY")
    print(runner.table(efficiency, digits=arguments.digits))
    write_csv(out / "annotation_efficiency.csv", efficiency)

    # ---- the frozen kill rule, applied mechanically ----------------------
    kill: dict[str, object] | None = None
    rule = manifest.get("kill_rule") or bm.KILL_RULE.as_dict()
    for entry in order:
        if entry["arm"] != rule["arm"] or entry["seed"] != rule["seed"]:
            continue
        rows = loaded[entry["trajectory"]]
        kill = bm.KILL_RULE.decide(
            mean([r.get("new_class_AP50") for r in rows]),
            rows[-1].get("known_mAP50") if rows else None,
        )
        print(f"\n[kill rule] {manifest.get('kill_rule_statement') or bm.KILL_RULE.statement()}")
        print(f"  {entry['trajectory']}: {kill['verdict']} — "
              + "; ".join(kill["reasons"]))
        if kill["verdict"] == "STOP":
            print("  Preserve it as a negative development result. Do not tune "
                  "it. Do not run its replication seeds.")

    for line in manifest.get("provenance", []):
        print(f"\n[provenance] {line}")

    bm.write_json(out / "summary.json", {
        "experiment": manifest["experiment"],
        "dry_run": bool(manifest.get("dry_run")),
        "endpoints": endpoints,
        "endpoint_statement": manifest["endpoint_statement"],
        "trajectories": [
            {k: v for k, v in e.items() if k != "tasks"}
            for e in manifest["trajectories"]
        ],
        "chain_summary": chain_rows,
        "contrasts": contrasts,
        "kill_rule": rule,
        "kill_rule_outcome": kill,
        "provenance": manifest.get("provenance", []),
        "reporting_rules": manifest["reporting_rules"],
    })

    print("\nHOW THESE NUMBERS MAY BE REPORTED")
    for line in manifest["reporting_rules"]:
        print(f"  - {line}")
    print(f"\nwrote per_task_metrics.csv, supervision_cost.csv, acquisition.csv, "
          f"chain_summary.csv, forgetting.csv, per_class_ap.csv, contrasts.csv, "
          f"annotation_efficiency.csv, summary.json to {out}")


if __name__ == "__main__":
    main()
