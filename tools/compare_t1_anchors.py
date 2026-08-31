#!/usr/bin/env python3
"""Combine completed controlled-LT T1 anchor metrics without authorizing T2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import longtail, t1_anchor  # noqa: E402


def write_csv_once(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise t1_anchor.AnchorError(f"Comparison output already exists: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def compare(work_root: Path, conditions: tuple[str, ...], output: Path) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    classes: list[dict[str, object]] = []
    sources: dict[str, dict[str, str]] = {}
    for condition in conditions:
        workspace = work_root / f"t1_anchor__{condition}__seed0"
        if t1_anchor.workspace_state(workspace, condition) != "DONE":
            raise t1_anchor.AnchorError(f"{condition} is not a validated DONE anchor.")
        metrics_path = workspace / "anchor_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("condition") != condition:
            raise t1_anchor.AnchorError(f"{condition} metrics belong to another condition.")
        summaries.append({
            "condition": condition,
            "overall_mAP50": metrics["overall_mAP50"],
            "head_mAP50": metrics["group_mAP50"]["head"],
            "medium_mAP50": metrics["group_mAP50"]["medium"],
            "tail_mAP50": metrics["group_mAP50"]["tail"],
            "spearman_AP50_log_train_frequency": metrics[
                "learnability_descriptives"]["spearman_AP50_log_train_frequency"],
            "minimum_AP50": metrics["learnability_descriptives"]["minimum_AP50"],
            "exact_zero_AP50_classes": ",".join(
                metrics["learnability_descriptives"]["exact_zero_AP50_classes"]),
        })
        classes.extend({
            "condition": condition,
            "class_name": row["class_name"],
            "train_count": row["train_count"],
            "rank": row["rank"],
            "group": row["group"],
            "AP50": row["anchor_AP50"],
        } for row in metrics["classes"])
        sources[condition] = {
            "metrics": str(metrics_path),
            "metrics_sha256": longtail.sha256_file(metrics_path),
            "checkpoint_sha256": str(metrics["checkpoint_sha256"]),
            "recipe_fingerprint": str(metrics["recipe_fingerprint"]),
        }
    output.mkdir(parents=True, exist_ok=True)
    write_csv_once(
        output / "anchor_summary.csv",
        ("condition", "overall_mAP50", "head_mAP50", "medium_mAP50", "tail_mAP50",
         "spearman_AP50_log_train_frequency", "minimum_AP50", "exact_zero_AP50_classes"),
        summaries,
    )
    write_csv_once(
        output / "anchor_per_class.csv",
        ("condition", "class_name", "train_count", "rank", "group", "AP50"),
        classes,
    )
    payload = {
        "schema": "controlled_t1_anchor_comparison_v1",
        "conditions": list(conditions),
        "summaries": summaries,
        "classes": classes,
        "sources": sources,
        "one_seed_descriptive_only": True,
        "incremental_training_authorized": False,
        "lt100_exact_zero_tail_classes": [
            row["class_name"] for row in classes
            if row["condition"] == "lt100" and row["group"] == "tail" and row["AP50"] == 0
        ],
    }
    t1_anchor.write_json_once_or_verify(output / "anchor_comparison.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--work-root", type=Path, required=True)
    command.add_argument(
        "--conditions", default=",".join(t1_anchor.PRIMARY_CONDITIONS),
        help="comma-separated completed conditions",
    )
    command.add_argument("--output", type=Path, required=True)
    return command


def main() -> int:
    arguments = parser().parse_args()
    conditions = tuple(item.strip() for item in arguments.conditions.split(",") if item.strip())
    if not conditions or any(item not in t1_anchor.PRIMARY_CONDITIONS for item in conditions):
        print("error: conditions must be a non-empty LT-10/LT-50/LT-100 subset", file=sys.stderr)
        return 2
    try:
        payload = compare(arguments.work_root.resolve(), conditions, arguments.output.resolve())
    except (t1_anchor.AnchorError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload["summaries"], indent=2, sort_keys=True))
    print("CONTROLLED LT ANCHOR COMPARISON COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
