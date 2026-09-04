#!/usr/bin/env python
"""Recover ``boxes_trained_on`` for trajectories that ran before it was recorded.

Session 1's four arms completed before the ledger separated *what a task bought*
from *what PROB was handed*, so their rows carry `boxes_supervised` and not
`boxes_trained_on`. The missing number is not lost: PROB was handed exactly the
ids in each task's own ``train/labelled_ids.txt``, and the benchmark's per-image
class counts say what is on them. This recomputes it the way
``remove_unknown_instances`` would — ``category_id in range(0, prev + current)``
— and writes a separate CSV.

**It never touches the results it reads.** No ``state.json``, no ``results.csv``,
no ``metrics.json`` is rewritten; the output goes to ``--out``, which must not be
the results directory. A completed trajectory stays exactly as it was measured.

    python tools/backfill_boxes_trained_on.py \\
        --results /content/drive/MyDrive/OWL/results/full_owod_active_benchmark_v1 \\
        --out /content/drive/MyDrive/OWL/results/full_owod_active_benchmark_v1/backfill
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import protocol, runner
from owl.active_selection import benchmark as bm
from owl.active_selection import budget as annotation_budget

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"


def labelled_ids(task_dir: Path) -> list[str]:
    """The ids PROB was handed for this task, in the order it was handed them."""

    path = task_dir / "train" / "labelled_ids.txt"
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").split() if line.strip()]


def replay_ids(task_dir: Path) -> list[str]:
    path = task_dir / "train" / "replay_ids.txt"
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").split() if line.strip()]


def rows_for(results: Path, entry: dict, candidate_index, groups) -> list[dict]:
    chain = bm.chain()
    out: list[dict] = []
    for task in chain[1:]:
        task_dir = results / entry["trajectory"] / f"{task.name}_{entry['arm']}"
        handed = labelled_ids(task_dir)
        if not handed:
            continue
        ledger = annotation_budget.supervision(
            candidate_index, handed, declared=task.known_classes, groups=groups,
        )
        aliases = replay_ids(task_dir)
        out.append({
            "trajectory": entry["trajectory"],
            "arm": entry["arm"],
            "seed": entry["seed"],
            "task": task.name,
            "declares": task.new_class,
            "images_handed_to_prob": len(handed),
            "replay_alias_images": len(aliases),
            "boxes_trained_on": ledger["boxes_supervised"],
            "boxes_trained_on_head": ledger["boxes_supervised_head"],
            "boxes_trained_on_medium": ledger["boxes_supervised_medium"],
            "boxes_trained_on_tail": ledger["boxes_supervised_tail"],
            "boxes_on_those_images_total": ledger["boxes_labelled"],
            "boxes_dropped_as_undeclared": ledger["boxes_banked"],
            "classes_supervised": ledger["supervised_classes"],
            "per_class_supervised": ledger["per_class_supervised"],
            "source": "backfilled from train/labelled_ids.txt",
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True,
                        help="must differ from --results; nothing there is rewritten")
    parser.add_argument("--digits", type=int, default=2)
    arguments = parser.parse_args()

    results = Path(arguments.results).resolve()
    out = Path(arguments.out).resolve()
    if out == results:
        raise SystemExit(
            "--out must not be the results directory. This tool recomputes a "
            "column for trajectories that are already measured; writing beside "
            "them risks overwriting the measurement it is describing."
        )

    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    candidate_index = json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    groups = protocol.load_groups()

    rows: list[dict] = []
    for entry in manifest["trajectories"]:
        if entry.get("status") != "COMPLETE":
            continue
        rows.extend(rows_for(results, entry, candidate_index, groups))
    if not rows:
        raise SystemExit("no completed trajectory carried a train/labelled_ids.txt")

    out.mkdir(parents=True, exist_ok=True)
    target = out / "boxes_trained_on.csv"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = target.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)

    print(runner.table([
        {k: v for k, v in row.items()
         if k not in ("trajectory", "per_class_supervised", "source")}
        for row in rows
    ], digits=arguments.digits))

    print(f"\nwrote {target}")
    print(f"read-only on {results}: nothing there was modified")
    totals: dict[str, int] = {}
    for row in rows:
        totals[row["arm"]] = totals.get(row["arm"], 0) + int(row["boxes_trained_on"])
    print("\nboxes PROB was handed over the whole chain, by arm:")
    for arm, total in sorted(totals.items(), key=lambda item: -item[1]):
        print(f"  {arm:16s} {total:6d}")


if __name__ == "__main__":
    main()
