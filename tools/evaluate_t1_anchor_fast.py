#!/usr/bin/env python3
"""Run the one mandatory final evaluation for a completed FAST anchor."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import bridge, longtail, metrics, t1_anchor, t1_anchor_fast

DEFAULT_MANIFEST_ROOT = ROOT / "data" / "reference" / "longtail"


def git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=root, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def validate_cuda_runtime(python: str, prob_root: Path, metadata: dict[str, object]) -> None:
    code = """import json,sys,torch,torchvision
from models.ops.functions import ms_deform_attn_func as w
from models.ops.modules import ms_deform_attn as d
print(json.dumps({'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'wrapper':bool(w.MSDA_AVAILABLE),'downstream':bool(d.MSDA_AVAILABLE)}))"""
    result = subprocess.run([python, "-c", code], cwd=prob_root, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError("FAST final evaluation CUDA probe failed.")
    runtime = json.loads(result.stdout.splitlines()[-1])
    expected = {"python": metadata["python_version"], "torch": metadata["torch_version"],
                "torchvision": metadata["torchvision_version"],
                "cuda": metadata["cuda_version"]}
    if not runtime["available"] or not runtime["wrapper"] or not runtime["downstream"] \
            or any(runtime[key] != value for key, value in expected.items()):
        raise t1_anchor.AnchorError("FAST final evaluation runtime differs from training.")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0
        for position in order[index:end]:
            result[position] = rank
        index = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    left_rank, right_rank = ranks(left), ranks(right)
    left_mean, right_mean = statistics.mean(left_rank), statistics.mean(right_rank)
    numerator = sum((x - left_mean) * (y - right_mean)
                    for x, y in zip(left_rank, right_rank, strict=True))
    denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left_rank)
                            * sum((y - right_mean) ** 2 for y in right_rank))
    return numerator / denominator if denominator else None


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    workspace = arguments.workspace.resolve()
    t1_anchor_fast.validate_workspace_path(workspace, arguments.condition)
    checkpoint = workspace / f"t1_fast_{arguments.condition}.pth"
    metadata_path = checkpoint.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    t1_anchor_fast.validate_training_metadata(metadata)
    if metadata.get("recipe_version") != t1_anchor_fast.FAST_RECIPE_VERSION \
            or metadata.get("global_step") != metadata.get("final_optimizer_updates"):
        raise t1_anchor.AnchorError("FAST final evaluation requires complete frozen training.")
    if longtail.sha256_file(checkpoint) != metadata.get("checkpoint_sha256"):
        raise t1_anchor.AnchorError("FAST final checkpoint SHA-256 changed.")
    if git_value(ROOT, "rev-parse", "HEAD") != metadata["owl_commit"] \
            or git_value(ROOT, "status", "--porcelain") \
            or git_value(arguments.prob_root, "rev-parse", "HEAD") != t1_anchor.PINNED_PROB_COMMIT:
        raise t1_anchor.AnchorError("FAST evaluation checkout identity changed.")
    _, manifest = t1_anchor.condition_manifest(arguments.condition, arguments.manifest_root)
    view = json.loads((workspace / "training_view.json").read_text(encoding="utf-8"))
    data_root = Path(str(view["data_root"]))
    split = data_root / "ImageSets" / "OWDETR" / f"{t1_anchor.EVALUATION_SPLIT}.txt"
    if longtail.sha256_file(split) != t1_anchor.EVALUATION_SPLIT_SHA256:
        raise t1_anchor.AnchorError("FAST fixed evaluation split changed.")
    output = workspace / "anchor_metrics.json"
    plan = {"condition": arguments.condition, "checkpoint": str(checkpoint),
            "output": str(output), "execute": arguments.execute}
    if not arguments.execute:
        return plan
    validate_cuda_runtime(arguments.python, arguments.prob_root, metadata)
    raw_output = workspace / "anchor_bridge_metrics.json"
    instrument = bridge.Bridge(
        prob_root=arguments.prob_root, data_root=data_root, device="cuda",
        num_workers=2, seed=0, log_dir=workspace / "logs")
    instrument.evaluate(
        checkpoint=checkpoint, test_set=t1_anchor.EVALUATION_SPLIT,
        output=raw_output, n_prev=0, n_current=19, detections=False)
    raw = json.loads(raw_output.read_text(encoding="utf-8"))
    raw["test_set_sha256"] = t1_anchor.EVALUATION_SPLIT_SHA256
    validation = metrics.validate_per_class_ap50(raw, n_prev=0, n_current=19)
    if not validation["usable"]:
        raise t1_anchor.AnchorError(f"FAST per-class AP50 is unusable: {validation}.")
    payload = t1_anchor.anchor_metrics_payload(
        condition=arguments.condition, manifest=manifest, bridge_metrics=raw,
        checkpoint_sha256=str(metadata["checkpoint_sha256"]),
        recipe_fingerprint=str(metadata["recipe_fingerprint"]))
    payload["schema"] = "controlled_t1_anchor_fast_metrics_v1"
    payload["recipe_version"] = t1_anchor_fast.FAST_RECIPE_VERSION
    payload["final_optimizer_updates"] = metadata["final_optimizer_updates"]
    payload["image_presentations"] = metadata["image_presentations"]
    values = [float(row["anchor_AP50"]) for row in payload["classes"]]
    frequencies = [math.log(float(row["train_count"])) for row in payload["classes"]]
    payload["learnability_descriptives"] = {
        "minimum_AP50": min(values), "median_AP50": statistics.median(values),
        "maximum_AP50": max(values),
        "exact_zero_AP50_classes": [row["class_name"] for row in payload["classes"]
                                    if float(row["anchor_AP50"]) == 0.0],
        "spearman_AP50_log_train_frequency": spearman(values, frequencies),
        "one_seed_descriptive_only": True, "significance_claims": False,
    }
    t1_anchor.write_json_once_or_verify(output, payload)
    per_class = workspace / "per_class.csv"
    temporary = per_class.with_name(".per_class.csv.tmp")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=("condition", "class", "rank", "group", "training_count", "AP50"),
        lineterminator="\n")
    writer.writeheader()
    for row in payload["classes"]:
        writer.writerow({"condition": arguments.condition, "class": row["class_name"],
                         "rank": row["rank"], "group": row["group"],
                         "training_count": row["train_count"], "AP50": row["anchor_AP50"]})
    expected_csv = buffer.getvalue()
    if per_class.is_file():
        if per_class.read_text(encoding="utf-8") != expected_csv:
            raise t1_anchor.AnchorError("Existing FAST per-class CSV differs from evaluation.")
    else:
        if temporary.exists() and temporary.read_text(encoding="utf-8") != expected_csv:
            raise t1_anchor.AnchorError("Interrupted FAST per-class CSV differs from evaluation.")
        if not temporary.exists():
            temporary.write_text(expected_csv, encoding="utf-8")
        temporary.replace(per_class)
    done = {
        "schema": "controlled_t1_anchor_fast_done_v1",
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": arguments.condition,
        "global_step": metadata["global_step"],
        "final_optimizer_updates": metadata["final_optimizer_updates"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "metrics_sha256": longtail.sha256_file(output),
        "per_class_csv_sha256": longtail.sha256_file(per_class),
        "recipe_fingerprint": metadata["recipe_fingerprint"],
        "plan_fingerprint": metadata["plan_fingerprint"],
    }
    t1_anchor.write_json_once_or_verify(workspace / "DONE.json", done)
    return plan | {"metrics": payload, "state": "DONE"}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--condition", choices=t1_anchor_fast.CONDITION_ORDER, required=True)
    command.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    command.add_argument("--prob-root", type=Path, required=True)
    command.add_argument("--workspace", type=Path, required=True)
    command.add_argument("--python", default=sys.executable)
    command.add_argument("--execute", action="store_true")
    return command


def main() -> int:
    try:
        report = evaluate(parser().parse_args())
    except (t1_anchor.AnchorError, longtail.LongTailError, OSError,
            json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
