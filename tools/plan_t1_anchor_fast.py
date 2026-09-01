#!/usr/bin/env python3
"""Freeze the equal FAST step budget from timing receipts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import longtail, t1_anchor, t1_anchor_fast


def plan(arguments: argparse.Namespace) -> dict[str, object]:
    receipts = {}
    for condition in t1_anchor_fast.CONDITION_ORDER:
        path = t1_anchor_fast.workspace(arguments.work_root, condition) / \
            "cuda_benchmark_fast_v1.json"
        if not path.is_file():
            raise t1_anchor.AnchorError(f"Missing FAST benchmark receipt: {path}.")
        receipts[condition] = json.loads(path.read_text(encoding="utf-8"))
    if arguments.output.is_file():
        payload = json.loads(arguments.output.read_text(encoding="utf-8"))
        t1_anchor_fast.validate_plan(payload)
        if payload.get("total_gpu_budget_hours") != arguments.total_gpu_budget_hours \
                or payload.get("default_optimizer_updates_per_condition") != \
                arguments.default_updates:
            raise t1_anchor.AnchorError("Existing frozen FAST plan uses another declared budget.")
        for condition, receipt in receipts.items():
            receipt_identity = t1_anchor_fast.benchmark_identity(receipt)
            expected_sha = longtail.sha256_bytes(
                (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
            if payload["benchmark_receipt_sha256"][condition] != expected_sha:
                raise t1_anchor.AnchorError("Existing FAST plan names another benchmark receipt.")
            if payload["benchmark_identities"][condition] != receipt_identity \
                    or payload["seconds_per_update"][condition] != \
                    receipt_identity["seconds_per_optimizer_update"]:
                raise t1_anchor.AnchorError("Existing FAST plan changed measured runtime evidence.")
        return payload
    payload = t1_anchor_fast.plan_experiment(
        receipts, total_budget_hours=arguments.total_gpu_budget_hours,
        default_updates=arguments.default_updates,
        final_evaluation_seconds=arguments.final_evaluation_seconds,
        safety_reserve_seconds=arguments.safety_reserve_seconds,
        setup_seconds=arguments.setup_seconds,
    )
    t1_anchor.write_json_once_or_verify(arguments.output, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--work-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--total-gpu-budget-hours", type=float, default=14.0)
    command.add_argument("--default-updates", type=int, default=12_000)
    command.add_argument("--final-evaluation-seconds", type=int, default=1_800)
    command.add_argument("--safety-reserve-seconds", type=int, default=1_800)
    command.add_argument("--setup-seconds", type=int, default=900)
    return command


def main() -> int:
    try:
        payload = plan(parser().parse_args())
    except (t1_anchor.AnchorError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print("FAST RECIPE PLAN")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("Decision:", payload["decision"])
    return 0 if payload["decision"] == "GO" else 3


if __name__ == "__main__":
    raise SystemExit(main())
