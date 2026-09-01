#!/usr/bin/env python3
"""Evaluate one controlled T1 anchor and write its learnability record."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import bridge, longtail, metrics, t1_anchor  # noqa: E402

DEFAULT_MANIFEST_ROOT = ROOT / "data" / "reference" / "longtail"


def git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def runtime_probe(python: str, prob_root: Path) -> dict[str, object]:
    code = """
import json, sys, torch, torchvision
from models.ops.functions import ms_deform_attn_func as wrapper
from models.ops.modules import ms_deform_attn as downstream
print(json.dumps({
    'python_version': sys.version,
    'torch_version': torch.__version__,
    'torchvision_version': torchvision.__version__,
    'cuda_version': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'wrapper_msda': bool(wrapper.MSDA_AVAILABLE),
    'downstream_msda': bool(downstream.MSDA_AVAILABLE),
}))
"""
    result = subprocess.run(
        [python, "-c", code], cwd=prob_root, text=True, capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(f"Evaluation runtime probe failed: {result.stderr[-2000:]}")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if not payload["cuda_available"] or not payload["wrapper_msda"] \
            or not payload["downstream_msda"]:
        raise t1_anchor.AnchorError("Evaluation requires CUDA and compiled MSDA.")
    return payload


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    workspace = arguments.workspace.resolve()
    checkpoint = workspace / f"t1_{arguments.condition}.pth"
    metadata_path = checkpoint.with_suffix(".metadata.json")
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise t1_anchor.AnchorError("Complete checkpoint and metadata are required.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    t1_anchor.validate_training_metadata(metadata)
    if metadata.get("global_step") != t1_anchor.TOTAL_OPTIMIZER_UPDATES:
        raise t1_anchor.AnchorError("Final evaluation requires all 183,434 updates.")
    if metadata["condition"] != arguments.condition:
        raise t1_anchor.AnchorError("Checkpoint metadata belongs to another condition.")
    if longtail.sha256_file(checkpoint) != metadata["checkpoint_sha256"]:
        raise t1_anchor.AnchorError("Checkpoint SHA-256 differs from its metadata.")
    if git_value(ROOT, "rev-parse", "HEAD") != metadata["owl_commit"]:
        raise t1_anchor.AnchorError("Evaluator OWL commit differs from the training commit.")
    if git_value(arguments.prob_root, "rev-parse", "HEAD") != metadata["prob_commit"]:
        raise t1_anchor.AnchorError("Evaluator PROB commit differs from the training commit.")
    _, manifest = t1_anchor.condition_manifest(arguments.condition, arguments.manifest_root)
    view_path = workspace / "training_view.json"
    if not view_path.is_file():
        raise t1_anchor.AnchorError("Training-view provenance is missing.")
    view = json.loads(view_path.read_text(encoding="utf-8"))
    data_root = Path(str(view["data_root"]))
    split = data_root / "ImageSets" / "OWDETR" / f"{t1_anchor.EVALUATION_SPLIT}.txt"
    if not split.is_file():
        raise t1_anchor.AnchorError("The fixed shared evaluation split is missing.")
    split_sha = longtail.sha256_file(split)
    if split_sha != t1_anchor.EVALUATION_SPLIT_SHA256 \
            or metadata["evaluation_split_sha256"] != split_sha:
        raise t1_anchor.AnchorError("The fixed shared evaluation split identity changed.")
    output = workspace / "anchor_metrics.json"
    if output.exists():
        raise t1_anchor.AnchorError(f"Refusing to overwrite anchor metrics {output}.")
    plan = {
        "condition": arguments.condition,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "evaluation_split": t1_anchor.EVALUATION_SPLIT,
        "evaluation_split_sha256": split_sha,
        "output": str(output),
        "execute": arguments.execute,
    }
    if not arguments.execute:
        return plan

    runtime = runtime_probe(arguments.python, arguments.prob_root)
    for name in ("python_version", "torch_version", "torchvision_version", "cuda_version"):
        if runtime.get(name) != metadata.get(name):
            raise t1_anchor.AnchorError(
                f"Evaluation {name} differs from the training environment."
            )

    raw_output = workspace / "anchor_bridge_metrics.json"
    if raw_output.exists():
        raise t1_anchor.AnchorError(
            f"Partial evaluator output requires inspection: {raw_output}.")
    instrument = bridge.Bridge(
        prob_root=arguments.prob_root,
        data_root=data_root,
        device="cuda",
        num_workers=2,
        seed=0,
        log_dir=workspace / "logs",
    )
    written = instrument.evaluate(
        checkpoint=checkpoint,
        test_set=t1_anchor.EVALUATION_SPLIT,
        output=raw_output,
        n_prev=0,
        n_current=19,
        detections=False,
    )
    raw = json.loads(written.read_text(encoding="utf-8"))
    raw["test_set_sha256"] = split_sha
    validation = metrics.validate_per_class_ap50(raw, n_prev=0, n_current=19)
    if not validation["usable"]:
        raise t1_anchor.AnchorError(f"PROB per-class AP50 validation failed: {validation}.")
    payload = t1_anchor.anchor_metrics_payload(
        condition=arguments.condition,
        manifest=manifest,
        bridge_metrics=raw,
        checkpoint_sha256=str(metadata["checkpoint_sha256"]),
        recipe_fingerprint=str(metadata["recipe_fingerprint"]),
    )
    values = [float(row["anchor_AP50"]) for row in payload["classes"]]
    frequencies = [math.log(float(row["train_count"])) for row in payload["classes"]]

    def ranks(items: list[float]) -> list[float]:
        order = sorted(range(len(items)), key=items.__getitem__)
        result = [0.0] * len(items)
        index = 0
        while index < len(order):
            end = index + 1
            while end < len(order) and items[order[end]] == items[order[index]]:
                end += 1
            rank = (index + end - 1) / 2.0
            for position in order[index:end]:
                result[position] = rank
            index = end
        return result

    def correlation(left: list[float], right: list[float]) -> float | None:
        left_rank, right_rank = ranks(left), ranks(right)
        left_mean = statistics.mean(left_rank)
        right_mean = statistics.mean(right_rank)
        numerator = sum(
            (x - left_mean) * (y - right_mean) for x, y in zip(left_rank, right_rank))
        denominator = math.sqrt(
            sum((x - left_mean) ** 2 for x in left_rank)
            * sum((y - right_mean) ** 2 for y in right_rank)
        )
        return numerator / denominator if denominator else None

    payload["learnability_descriptives"] = {
        "minimum_AP50": min(values),
        "median_AP50": statistics.median(values),
        "maximum_AP50": max(values),
        "exact_zero_AP50_classes": [
            row["class_name"] for row in payload["classes"]
            if float(row["anchor_AP50"]) == 0.0
        ],
        "post_hoc_threshold_used": False,
        "spearman_AP50_log_train_frequency": correlation(values, frequencies),
        "incremental_training_authorized": False,
        "decision": "manual preregistered learnability review required",
    }
    t1_anchor.write_json_once_or_verify(output, payload)
    per_class = workspace / "per_class.csv"
    temporary_csv = per_class.with_name(".per_class.csv.tmp")
    if per_class.exists() or temporary_csv.exists():
        raise t1_anchor.AnchorError("Per-class evaluation output already exists.")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("condition", "class_name", "rank", "group", "train_count", "AP50"),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in payload["classes"]:
            writer.writerow({
                "condition": arguments.condition,
                "class_name": row["class_name"],
                "rank": row["rank"],
                "group": row["group"],
                "train_count": row["train_count"],
                "AP50": row["anchor_AP50"],
            })
    temporary_csv.replace(per_class)
    done = {
        "schema": "controlled_t1_anchor_done_v2",
        "recipe_version": t1_anchor.RECIPE_VERSION,
        "condition": arguments.condition,
        "global_step": t1_anchor.TOTAL_OPTIMIZER_UPDATES,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "metrics_sha256": longtail.sha256_file(output),
        "per_class_csv_sha256": longtail.sha256_file(per_class),
        "recipe_fingerprint": metadata["recipe_fingerprint"],
        "incremental_training_authorized": False,
    }
    t1_anchor.write_json_once_or_verify(workspace / "DONE.json", done)
    return plan | {"metrics": payload}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--condition", choices=t1_anchor.PRIMARY_CONDITIONS, required=True)
    command.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    command.add_argument("--prob-root", type=Path, required=True)
    command.add_argument("--workspace", type=Path, required=True)
    command.add_argument("--python", default=sys.executable)
    command.add_argument("--execute", action="store_true")
    return command


def main() -> int:
    try:
        report = evaluate(parser().parse_args())
    except (t1_anchor.AnchorError, longtail.LongTailError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
