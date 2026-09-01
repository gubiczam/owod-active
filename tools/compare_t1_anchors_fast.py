#!/usr/bin/env python3
"""Create the descriptive three-condition FAST comparison after every DONE gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import longtail, t1_anchor, t1_anchor_fast


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise t1_anchor.AnchorError(f"FAST comparison output already exists: {path}.")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def delta_rows(control: dict[str, object], treatment: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for name in ("overall", "head", "medium", "tail"):
        control_value = control["overall_mAP50"] if name == "overall" else control["group_mAP50"][name]
        treatment_value = treatment["overall_mAP50"] if name == "overall" else treatment["group_mAP50"][name]
        rows.append({"scope": "aggregate", "name": name,
                     "control_AP50": control_value, "treatment_AP50": treatment_value,
                     "delta_AP50": float(treatment_value) - float(control_value)})
    by_control = {row["class_name"]: row for row in control["classes"]}
    for row in treatment["classes"]:
        previous = by_control[row["class_name"]]
        rows.append({"scope": "class", "name": row["class_name"],
                     "control_AP50": previous["anchor_AP50"],
                     "treatment_AP50": row["anchor_AP50"],
                     "delta_AP50": float(row["anchor_AP50"])
                     - float(previous["anchor_AP50"])})
    return rows


def figures(output: Path, metrics_by_condition: dict[str, dict[str, object]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ("lt10", "lt50", "lt100")
    written = []

    def save(name: str) -> None:
        path = output / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        written.append(name)

    plt.figure(figsize=(5, 3.5))
    plt.bar(order, [metrics_by_condition[item]["overall_mAP50"] for item in order])
    plt.ylabel("T1 mAP50")
    plt.title("FAST fixed-compute anchor performance")
    save("overall_ap50_by_condition.png")

    plt.figure(figsize=(6, 3.8))
    x = range(len(order))
    width = 0.25
    for offset, group in enumerate(("head", "medium", "tail")):
        plt.bar([value + (offset - 1) * width for value in x],
                [metrics_by_condition[item]["group_mAP50"][group] for item in order],
                width=width, label=group)
    plt.xticks(list(x), order)
    plt.ylabel("mAP50")
    plt.legend()
    plt.title("Controlled frequency groups")
    save("group_ap50_by_condition.png")

    plt.figure(figsize=(6, 4))
    for condition in order:
        rows = metrics_by_condition[condition]["classes"]
        plt.scatter([math.log(float(row["train_count"])) for row in rows],
                    [row["anchor_AP50"] for row in rows], label=condition, s=22)
    plt.xlabel("log controlled training count")
    plt.ylabel("per-class AP50")
    plt.legend()
    save("per_class_ap50_vs_log_frequency.png")

    plt.figure(figsize=(6, 4))
    baseline = {row["class_name"]: row for row in metrics_by_condition["lt10"]["classes"]}
    for condition in ("lt50", "lt100"):
        rows = metrics_by_condition[condition]["classes"]
        plt.plot([row["rank"] for row in rows],
                 [float(row["anchor_AP50"])
                  - float(baseline[row["class_name"]]["anchor_AP50"]) for row in rows],
                 marker="o", markersize=3, label=f"{condition} - lt10")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("controlled rank")
    plt.ylabel("AP50 change")
    plt.legend()
    save("ap50_change_vs_controlled_rank.png")
    return written


def compare(work_root: Path, output: Path) -> dict[str, object]:
    metrics_by_condition = {}
    sources = {}
    plan_fingerprints = set()
    final_updates = set()
    for condition in ("lt10", "lt50", "lt100"):
        workspace = t1_anchor_fast.workspace(work_root, condition)
        if t1_anchor_fast.workspace_state(workspace, condition) != "DONE":
            raise t1_anchor.AnchorError(
                "FAST comparison requires LT10, LT50, and LT100 all to be DONE.")
        metrics_path = workspace / "anchor_metrics.json"
        metadata_path = (workspace / f"t1_fast_{condition}.pth").with_suffix(".metadata.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metrics.get("recipe_version") != t1_anchor_fast.FAST_RECIPE_VERSION \
                or metrics.get("condition") != condition:
            raise t1_anchor.AnchorError("FAST comparison encountered a non-FAST receipt.")
        metrics_by_condition[condition] = metrics
        plan_fingerprints.add(metadata["plan_fingerprint"])
        final_updates.add(metadata["final_optimizer_updates"])
        sources[condition] = {
            "metrics": str(metrics_path), "metrics_sha256": longtail.sha256_file(metrics_path),
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "recipe_fingerprint": metadata["recipe_fingerprint"],
        }
    if len(plan_fingerprints) != 1 or len(final_updates) != 1:
        raise t1_anchor.AnchorError("FAST anchors do not share one frozen budget plan.")
    output.mkdir(parents=True, exist_ok=True)
    summaries = [{
        "condition": condition,
        "overall_mAP50": metrics_by_condition[condition]["overall_mAP50"],
        "head_mAP50": metrics_by_condition[condition]["group_mAP50"]["head"],
        "medium_mAP50": metrics_by_condition[condition]["group_mAP50"]["medium"],
        "tail_mAP50": metrics_by_condition[condition]["group_mAP50"]["tail"],
        "spearman_AP50_log_train_count": metrics_by_condition[condition][
            "learnability_descriptives"]["spearman_AP50_log_train_frequency"],
    } for condition in ("lt10", "lt50", "lt100")]
    per_class = [{
        "condition": condition, "class": row["class_name"], "rank": row["rank"],
        "group": row["group"], "training_count": row["train_count"],
        "AP50": row["anchor_AP50"],
    } for condition in ("lt10", "lt50", "lt100")
      for row in metrics_by_condition[condition]["classes"]]
    write_csv(output / "table_anchor_metrics.csv", tuple(summaries[0]), summaries)
    write_csv(output / "table_per_class.csv", tuple(per_class[0]), per_class)
    for treatment, control in (("lt50", "lt10"), ("lt100", "lt10"), ("lt100", "lt50")):
        rows = delta_rows(metrics_by_condition[control], metrics_by_condition[treatment])
        write_csv(output / f"table_delta_{treatment}_vs_{control}.csv", tuple(rows[0]), rows)
    figure_names = figures(output, metrics_by_condition)
    payload = {
        "schema": "controlled_t1_anchor_fast_comparison_v1",
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "conditions": ["lt10", "lt50", "lt100"],
        "final_optimizer_updates_per_condition": next(iter(final_updates)),
        "plan_fingerprint": next(iter(plan_fingerprints)),
        "summaries": summaries, "sources": sources, "figures": figure_names,
        "one_seed_descriptive_only": True, "significance_claims": False,
        "scientific_wording": (
            "fixed-compute controlled comparison of annotated supervision imbalance; "
            "not historical reproduction and not evidence of convergence"
        ),
    }
    t1_anchor.write_json_once_or_verify(output / "summary.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--work-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    return command


def main() -> int:
    try:
        arguments = parser().parse_args()
        payload = compare(arguments.work_root.resolve(), arguments.output.resolve())
    except (t1_anchor.AnchorError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload["summaries"], indent=2, sort_keys=True))
    print("CONTROLLED T1 ANCHOR FAST COMPARISON COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
