#!/usr/bin/env python3
"""Benchmark or train one compute-matched FAST anchor without touching V2."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import longtail, t1_anchor, t1_anchor_fast

DEFAULT_MANIFEST_ROOT = ROOT / "data" / "reference" / "longtail"


def git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=root, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def identity(arguments: argparse.Namespace):
    _, manifest = t1_anchor.condition_manifest(arguments.condition, arguments.manifest_root)
    initialization_sha = longtail.sha256_file(arguments.initialization)
    if initialization_sha != arguments.initialization_sha:
        raise t1_anchor.AnchorError("FAST initialization SHA-256 mismatch.")
    sidecar = arguments.initialization.with_suffix(".initialization.json")
    initialization = json.loads(sidecar.read_text(encoding="utf-8"))
    t1_anchor.validate_initialization_metadata(initialization, arguments.initialization)
    workspace = arguments.workspace.resolve()
    t1_anchor_fast.validate_workspace_path(workspace, arguments.condition)
    view = json.loads((workspace / "training_view.json").read_text(encoding="utf-8"))
    expected = {
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": arguments.condition,
        "manifest_scientific_sha256": manifest["scientific_sha256"],
        "initialization_sha256": initialization_sha,
        "prob_commit": t1_anchor.PINNED_PROB_COMMIT,
        "owl_commit": arguments.owl_commit,
    }
    if any(view.get(key) != value for key, value in expected.items()):
        raise t1_anchor.AnchorError("FAST training view identity changed.")
    data_root = Path(str(view["data_root"]))
    split = data_root / "ImageSets" / "OWDETR" / "owl_anchor_train.txt"
    if longtail.sha256_file(split) != view["split_sha256"]:
        raise t1_anchor.AnchorError("FAST training split changed.")
    return manifest, initialization, view, data_root, split


def recipe_for(arguments: argparse.Namespace, manifest: dict[str, object],
               initialization: dict[str, object], plan: dict[str, object]):
    runtime = plan["benchmark_identities"][arguments.condition]
    recipe = t1_anchor_fast.FastRecipe(
        condition=arguments.condition,
        manifest_sha256=str(manifest["scientific_sha256"]),
        owl_commit=arguments.owl_commit,
        initialization_sha256=arguments.initialization_sha,
        initialization_model_state_sha256=str(initialization["model_state_sha256"]),
        python_version=str(runtime["python_version"]),
        torch_version=str(runtime["torch_version"]),
        torchvision_version=str(runtime["torchvision_version"]),
        cuda_version=str(runtime["cuda_version"]),
        final_optimizer_updates=int(plan["frozen_optimizer_updates_per_condition"]),
    )
    recipe.validate()
    return recipe


def prob_arguments(data_root: Path, output_dir: Path) -> list[str]:
    config = t1_anchor_fast.SCIENTIFIC_CONFIG
    return [
        "--", "--output_dir", str(output_dir), "--dataset", str(config["dataset"]),
        "--data_root", str(data_root), "--device", "cuda",
        "--PREV_INTRODUCED_CLS", "0", "--CUR_INTRODUCED_CLS", "19",
        "--num_classes", str(config["num_classes"]), "--train_set", "owl_anchor_train",
        "--test_set", t1_anchor.EVALUATION_SPLIT, "--epochs", "1", "--lr_drop", "1",
        "--model_type", str(config["model_type"]), "--backbone", str(config["backbone"]),
        "--num_queries", str(config["num_queries"]), "--num_feature_levels",
        str(config["num_feature_levels"]), "--position_embedding",
        str(config["position_embedding"]), "--position_embedding_scale",
        str(config["position_embedding_scale"]), "--enc_layers", str(config["encoder_layers"]),
        "--dec_layers", str(config["decoder_layers"]), "--enc_n_points",
        str(config["encoder_attention_points"]), "--dec_n_points",
        str(config["decoder_attention_points"]), "--hidden_dim", str(config["hidden_dim"]),
        "--dim_feedforward", str(config["feedforward_dim"]), "--dropout",
        str(config["dropout"]), "--nheads", str(config["attention_heads"]),
        "--set_cost_class", str(config["matcher_class_cost"]), "--set_cost_bbox",
        str(config["matcher_bbox_cost"]), "--set_cost_giou",
        str(config["matcher_giou_cost"]), "--cls_loss_coef",
        str(config["classification_loss_coefficient"]), "--bbox_loss_coef",
        str(config["bbox_loss_coefficient"]), "--giou_loss_coef",
        str(config["giou_loss_coefficient"]), "--focal_alpha", str(config["focal_alpha"]),
        "--obj_loss_coef", str(config["objectness_loss_coefficient"]), "--obj_temp",
        str(config["objectness_temperature"]), "--top_unk", str(config["top_unknown"]),
        "--featdim", str(config["feature_dim"]), "--nc_loss_coef",
        str(config["novelty_loss_coefficient"]), "--nc_epoch",
        str(config["novelty_start_epoch"]), "--bbox_thresh", str(config["bbox_threshold"]),
        "--unk_conf_w", str(config["unknown_confidence_weight"]), "--batch_size", "2",
        "--lr", str(config["learning_rate"]), "--lr_backbone",
        str(config["backbone_learning_rate"]), "--lr_linear_proj_mult",
        str(float(config["linear_projection_learning_rate"]) / float(config["learning_rate"])),
        "--weight_decay", str(config["weight_decay"]), "--clip_max_norm",
        str(config["clip_max_norm"]), "--seed", "0", "--num_workers", "0",
        "--eval_every", "1000000", "--wandb_project", "",
    ]


def command_for(arguments: argparse.Namespace, manifest: dict[str, object],
                view: dict[str, object], data_root: Path, split: Path,
                recipe: t1_anchor_fast.FastRecipe | None) -> tuple[list[str], Path, Path]:
    output_dir = arguments.workspace / ("benchmark_fast_v1" if arguments.benchmark else "train")
    result_path = arguments.workspace / (
        "cuda_benchmark_fast_v1.json" if arguments.benchmark else "training_session_fast_v1.json")
    command = [
        arguments.python, "-u", str(ROOT / "tools" / "run_prob_t1_anchor_fast.py"),
        "--prob-root", str(arguments.prob_root), "--owl-root", str(ROOT),
        "--condition", arguments.condition, "--manifest-sha",
        str(manifest["scientific_sha256"]), "--initialization",
        str(arguments.initialization), "--initialization-sha", arguments.initialization_sha,
        "--train-split-sha", longtail.sha256_file(split), "--result", str(result_path),
    ]
    if arguments.benchmark:
        command += ["--benchmark", "--warmup-iterations", "5",
                    "--measured-iterations", str(arguments.benchmark_iterations)]
    else:
        command += ["--plan", str(arguments.plan), "--benchmark-receipt",
                    str(arguments.workspace / "cuda_benchmark_fast_v1.json"),
                    "--recipe-fingerprint", recipe.fingerprint()]
        if arguments.stop_at_unix is not None:
            command += ["--stop-at-unix", str(arguments.stop_at_unix)]
    command += prob_arguments(data_root, output_dir)
    return command, output_dir, result_path


def execute(arguments: argparse.Namespace) -> dict[str, object]:
    if git_value(ROOT, "rev-parse", "HEAD") != arguments.owl_commit \
            or git_value(ROOT, "status", "--porcelain"):
        raise t1_anchor.AnchorError("FAST execution requires its clean reviewed OWL commit.")
    if git_value(arguments.prob_root, "rev-parse", "HEAD") != t1_anchor.PINNED_PROB_COMMIT:
        raise t1_anchor.AnchorError("FAST execution requires pinned PROB.")
    manifest, initialization, view, data_root, split = identity(arguments)
    benchmark_path = arguments.workspace / "cuda_benchmark_fast_v1.json"

    def validate_benchmark(payload: dict[str, object]) -> None:
        t1_anchor_fast.benchmark_identity(payload)
        expected = {
            "condition": arguments.condition,
            "manifest_sha256": manifest["scientific_sha256"],
            "initialization_sha256": arguments.initialization_sha,
            "train_split_sha256": longtail.sha256_file(split),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise t1_anchor.AnchorError("FAST benchmark differs from current training inputs.")

    plan = None
    recipe = None
    if not arguments.benchmark:
        if arguments.plan is None:
            raise t1_anchor.AnchorError("FAST training requires a frozen plan receipt.")
        if arguments.execute and arguments.stop_at_unix is None:
            raise t1_anchor.AnchorError("FAST training requires the global wall-clock deadline.")
        plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
        t1_anchor_fast.validate_plan(plan)
        recipe = recipe_for(arguments, manifest, initialization, plan)
        if not benchmark_path.is_file() \
                or longtail.sha256_file(benchmark_path) != plan["benchmark_receipt_sha256"][
                    arguments.condition]:
            raise t1_anchor.AnchorError("FAST condition benchmark differs from its frozen plan.")
        validate_benchmark(json.loads(benchmark_path.read_text(encoding="utf-8")))
        t1_anchor.write_json_once_or_verify(arguments.workspace / "recipe_fast_v1.json", {
            "schema": t1_anchor_fast.FAST_SCHEMA,
            "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
            "fingerprint": recipe.fingerprint(), "recipe": recipe.payload(),
            "plan_fingerprint": plan["plan_fingerprint"],
        })
    command, output_dir, result_path = command_for(
        arguments, manifest, view, data_root, split, recipe)
    report = {
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": arguments.condition, "benchmark": arguments.benchmark,
        "command": command, "command_shell": shlex.join(command),
        "execute": arguments.execute,
    }
    if not arguments.execute:
        return report
    if arguments.benchmark and result_path.is_file():
        receipt = json.loads(result_path.read_text(encoding="utf-8"))
        validate_benchmark(receipt)
        return report | {"receipt": receipt, "cached": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = arguments.workspace / "TRAINING.json"
    if not arguments.benchmark:
        temporary_marker = marker.with_name(".TRAINING.json.tmp")
        temporary_marker.write_text(
            json.dumps(t1_anchor_fast.training_marker_payload(arguments.condition),
                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_marker, marker)
    try:
        result = subprocess.run(command, cwd=arguments.prob_root,
                                env=os.environ | {"PYTHONHASHSEED": "0"}, check=False)
    finally:
        if not arguments.benchmark and marker.is_file():
            marker.unlink()
    if result.returncode:
        raise t1_anchor.AnchorError(f"FAST adapter exited with status {result.returncode}.")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if arguments.benchmark:
        validate_benchmark(payload)
        return report | {"receipt": payload, "cached": False}
    resume = arguments.workspace / "train" / "resume_latest.pth"
    if payload["checkpoint_sha256"] != longtail.sha256_file(resume):
        raise t1_anchor.AnchorError("FAST session checkpoint hash mismatch.")
    if payload["global_step"] < recipe.final_optimizer_updates:
        return report | {"training": payload, "state": "INCOMPLETE_RESUMABLE"}
    if payload["global_step"] != recipe.final_optimizer_updates:
        raise t1_anchor.AnchorError("FAST exceeded its frozen optimizer budget.")
    final = arguments.workspace / f"t1_fast_{arguments.condition}.pth"
    checkpoint_sha = t1_anchor.copy_checkpoint(resume, final)
    metadata = {
        "schema": t1_anchor_fast.FAST_SCHEMA,
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": arguments.condition, "manifest_sha256": recipe.manifest_sha256,
        "owl_commit": recipe.owl_commit, "prob_commit": recipe.prob_commit,
        "initialization_sha256": recipe.initialization_sha256,
        "recipe_fingerprint": recipe.fingerprint(), "recipe": recipe.payload(),
        "plan_fingerprint": plan["plan_fingerprint"], "seed": 0,
        "global_step": recipe.final_optimizer_updates,
        "final_optimizer_updates": recipe.final_optimizer_updates,
        "image_presentations": recipe.total_image_presentations,
        "lr_drop_update": recipe.lr_drop_update,
        "checkpoint": str(final), "checkpoint_sha256": checkpoint_sha,
        "class_order": list(t1_anchor.protocol.TASK1),
        "train_objects": int(manifest["selected_objects"]),
        "train_images": int(manifest["selected_images"]),
        "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
        "started_at": payload["started_at"],
        "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "gpu": payload["gpu"], "torch_version": payload["torch_version"],
        "torchvision_version": recipe.torchvision_version,
        "python_version": recipe.python_version, "cuda_version": payload["cuda_version"],
        "msda": {"backend": payload["msda"], "extension": payload["msda_extension"]},
        "benchmark_receipt_sha256": plan["benchmark_receipt_sha256"][arguments.condition],
        "resume_guarantee": "next unseen unique batch plus complete RNG state",
        "command": command,
    }
    t1_anchor_fast.validate_training_metadata(metadata)
    t1_anchor.write_json_once_or_verify(final.with_suffix(".metadata.json"), metadata)
    return report | {"training": payload, "metadata": metadata,
                     "state": "TRAINED_PENDING_EVAL"}


def parser_cli() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--condition", choices=t1_anchor_fast.CONDITION_ORDER, required=True)
    command.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    command.add_argument("--prob-root", type=Path, required=True)
    command.add_argument("--workspace", type=Path, required=True)
    command.add_argument("--initialization", type=Path, required=True)
    command.add_argument("--initialization-sha", required=True)
    command.add_argument("--owl-commit", required=True)
    command.add_argument("--python", default=sys.executable)
    command.add_argument("--benchmark", action="store_true")
    command.add_argument("--benchmark-iterations", type=int, default=20)
    command.add_argument("--plan", type=Path)
    command.add_argument("--stop-at-unix", type=float)
    command.add_argument("--execute", action="store_true")
    return command


def main() -> int:
    try:
        arguments = parser_cli().parse_args()
        if arguments.benchmark_iterations != 20:
            raise t1_anchor.AnchorError("FAST benchmark requires exactly twenty measured updates.")
        report = execute(arguments)
    except (t1_anchor.AnchorError, longtail.LongTailError, OSError,
            json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
