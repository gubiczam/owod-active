#!/usr/bin/env python3
"""Benchmark or advance one controlled T1 anchor under fixed-step Recipe V2."""

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

from owl import longtail, t1_anchor  # noqa: E402

DEFAULT_MANIFEST_ROOT = ROOT / "data" / "reference" / "longtail"


def git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=root, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def recipe_for(arguments: argparse.Namespace):
    _, manifest = t1_anchor.condition_manifest(arguments.condition, arguments.manifest_root)
    initialization_sha = longtail.sha256_file(arguments.initialization)
    if initialization_sha != arguments.initialization_sha:
        raise t1_anchor.AnchorError("Initialization SHA-256 mismatch.")
    sidecar = arguments.initialization.with_suffix(".initialization.json")
    if not sidecar.is_file():
        raise t1_anchor.AnchorError("Initialization provenance sidecar is missing.")
    initialization = json.loads(sidecar.read_text(encoding="utf-8"))
    t1_anchor.validate_initialization_metadata(initialization, arguments.initialization)
    recipe = t1_anchor.AnchorRecipe(
        condition=arguments.condition,
        manifest_sha256=str(manifest["scientific_sha256"]),
        owl_commit=arguments.owl_commit,
        initialization_sha256=initialization_sha,
        initialization_model_state_sha256=str(initialization["model_state_sha256"]),
        python_version=str(initialization["python_version"]),
        torch_version=str(initialization["torch_version"]),
        torchvision_version=str(initialization["torchvision_version"]),
        cuda_version=str(initialization["cuda_version"]),
    )
    recipe.validate()
    return manifest, recipe, initialization


def validate_workspace(arguments: argparse.Namespace, recipe: t1_anchor.AnchorRecipe,
                       manifest: dict[str, object]) -> dict[str, object]:
    workspace = arguments.workspace.resolve()
    t1_anchor.validate_anchor_workspace(
        workspace=workspace, condition=arguments.condition,
        output_checkpoint=workspace / f"t1_{arguments.condition}.pth",
        allow_existing_output=True)
    path = workspace / "training_view.json"
    if not path.is_file():
        raise t1_anchor.AnchorError("Run the materializing V2 preflight first.")
    view = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "condition": arguments.condition,
        "manifest_scientific_sha256": recipe.manifest_sha256,
        "recipe_fingerprint": recipe.fingerprint(),
        "images": int(manifest["selected_images"]),
        "evaluation_images": 4308,
        "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
    }
    if any(view.get(key) != value for key, value in expected.items()):
        raise t1_anchor.AnchorError("Training view differs from the fixed Recipe V2 identity.")
    data_root = Path(str(view["data_root"]))
    train_split = data_root / "ImageSets" / "OWDETR" / "owl_anchor_train.txt"
    if not train_split.is_file() or longtail.sha256_file(train_split) != view["split_sha256"]:
        raise t1_anchor.AnchorError("Condition training split is missing or changed.")
    return view


def command_for(arguments: argparse.Namespace, recipe: t1_anchor.AnchorRecipe,
                view: dict[str, object], *, smoke: bool,
                write_smoke_splits: bool = False) -> tuple[list[str], Path, Path]:
    del write_smoke_splits
    workspace = arguments.workspace.resolve()
    output_dir = workspace / ("smoke_v2" if smoke else "train")
    data_root = Path(str(view["data_root"]))
    split = data_root / "ImageSets" / "OWDETR" / "owl_anchor_train.txt"
    result = output_dir / ("benchmark.json" if smoke else "session.json")
    command = [
        arguments.python, "-u", str(ROOT / "tools" / "run_prob_t1_anchor_v2.py"),
        "--prob-root", str(arguments.prob_root), "--owl-root", str(ROOT),
        "--condition", recipe.condition,
        "--recipe-fingerprint", recipe.fingerprint(),
        "--manifest-sha", recipe.manifest_sha256,
        "--initialization", str(arguments.initialization.resolve()),
        "--initialization-sha", recipe.initialization_sha256,
        "--train-split-sha", longtail.sha256_file(split), "--result", str(result),
    ]
    if smoke:
        command += ["--benchmark", "--warmup-iterations", "5",
                    "--measured-iterations", str(arguments.benchmark_iterations)]
    elif arguments.stop_at_unix is not None:
        command += ["--stop-at-unix", str(arguments.stop_at_unix)]
    command += _prob_arguments(recipe, data_root, output_dir)
    return command, output_dir, output_dir / "resume_latest.pth"


def _prob_arguments(recipe: t1_anchor.AnchorRecipe, data_root: Path,
                    output_dir: Path) -> list[str]:
    return [
        "--", "--output_dir", str(output_dir), "--dataset", recipe.dataset,
        "--data_root", str(data_root), "--device", recipe.device,
        "--PREV_INTRODUCED_CLS", "0", "--CUR_INTRODUCED_CLS", "19",
        "--num_classes", str(recipe.num_classes), "--train_set", "owl_anchor_train",
        "--test_set", t1_anchor.EVALUATION_SPLIT,
        "--epochs", str(recipe.reference_epochs), "--lr_drop", "31",
        "--model_type", recipe.model_type, "--backbone", recipe.backbone,
        "--num_queries", str(recipe.num_queries), "--num_feature_levels",
        str(recipe.num_feature_levels), "--position_embedding", recipe.position_embedding,
        "--position_embedding_scale", str(recipe.position_embedding_scale),
        "--enc_layers", str(recipe.encoder_layers), "--dec_layers", str(recipe.decoder_layers),
        "--enc_n_points", str(recipe.encoder_attention_points), "--dec_n_points",
        str(recipe.decoder_attention_points), "--hidden_dim", str(recipe.hidden_dim),
        "--dim_feedforward", str(recipe.feedforward_dim), "--dropout", str(recipe.dropout),
        "--nheads", str(recipe.attention_heads), "--set_cost_class",
        str(recipe.matcher_class_cost), "--set_cost_bbox", str(recipe.matcher_bbox_cost),
        "--set_cost_giou", str(recipe.matcher_giou_cost), "--cls_loss_coef",
        str(recipe.classification_loss_coefficient), "--bbox_loss_coef",
        str(recipe.bbox_loss_coefficient), "--giou_loss_coef",
        str(recipe.giou_loss_coefficient), "--focal_alpha", str(recipe.focal_alpha),
        "--obj_loss_coef", str(recipe.objectness_loss_coefficient), "--obj_temp",
        str(recipe.objectness_temperature), "--top_unk", str(recipe.top_unknown),
        "--featdim", str(recipe.feature_dim), "--nc_loss_coef",
        str(recipe.novelty_loss_coefficient), "--nc_epoch", str(recipe.novelty_start_epoch),
        "--bbox_thresh", str(recipe.bbox_threshold), "--unk_conf_w",
        str(recipe.unknown_confidence_weight), "--batch_size", str(recipe.batch_size),
        "--lr", str(recipe.learning_rate), "--lr_backbone",
        str(recipe.backbone_learning_rate), "--lr_linear_proj_mult",
        str(recipe.linear_projection_learning_rate / recipe.learning_rate),
        "--weight_decay", str(recipe.weight_decay), "--clip_max_norm",
        str(recipe.clip_max_norm), "--seed", str(recipe.seed), "--num_workers",
        str(recipe.num_workers), "--eval_every", "1000000", "--wandb_project", "",
    ]


def execute(arguments: argparse.Namespace) -> dict[str, object]:
    prob_root = arguments.prob_root.resolve()
    if git_value(ROOT, "rev-parse", "HEAD") != arguments.owl_commit:
        raise t1_anchor.AnchorError("OWL checkout does not match the reviewed scientific commit.")
    if git_value(ROOT, "status", "--porcelain"):
        raise t1_anchor.AnchorError("Scientific training requires a clean OWL checkout.")
    if git_value(prob_root, "rev-parse", "HEAD") != t1_anchor.PINNED_PROB_COMMIT:
        raise t1_anchor.AnchorError("PROB checkout does not match the pinned commit.")
    manifest, recipe, _initialization = recipe_for(arguments)
    view = validate_workspace(arguments, recipe, manifest)
    command, output_dir, resume_path = command_for(
        arguments, recipe, view, smoke=arguments.smoke_only)
    plan = {
        "recipe_version": t1_anchor.RECIPE_VERSION,
        "condition": arguments.condition, "recipe_fingerprint": recipe.fingerprint(),
        "command": command, "command_shell": shlex.join(command),
        "workspace": str(arguments.workspace.resolve()), "output_dir": str(output_dir),
        "smoke_only": arguments.smoke_only, "execute": arguments.execute,
    }
    if not arguments.execute:
        return plan
    if not arguments.smoke_only:
        receipt = arguments.workspace / "cuda_training_smoke_v2.json"
        if not receipt.is_file():
            raise t1_anchor.AnchorError("Run the Recipe V2 CUDA benchmark before training.")
        smoke = json.loads(receipt.read_text(encoding="utf-8"))
        if smoke.get("recipe_fingerprint") != recipe.fingerprint():
            raise t1_anchor.AnchorError("CUDA benchmark belongs to another recipe.")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=prob_root,
                            env=os.environ | {"PYTHONHASHSEED": str(recipe.seed)}, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(f"Recipe V2 adapter exited with status {result.returncode}.")
    result_path = output_dir / ("benchmark.json" if arguments.smoke_only else "session.json")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if arguments.smoke_only:
        return plan | {"smoke": _publish_smoke(arguments, recipe, payload)}
    return plan | _publish_training(arguments, recipe, manifest, command, resume_path, payload)


def _publish_smoke(arguments: argparse.Namespace, recipe: t1_anchor.AnchorRecipe,
                   payload: dict[str, object]) -> dict[str, object]:
    if payload.get("warmup_iterations") != 5 \
            or payload.get("measured_iterations") != arguments.benchmark_iterations \
            or not payload.get("real_model_forward_matcher_criterion_backward_optimizer"):
        raise t1_anchor.AnchorError("Recipe V2 CUDA benchmark is incomplete.")
    seconds = float(payload["seconds_per_optimizer_update"])
    receipt_payload = payload | {
        "recipe_fingerprint": recipe.fingerprint(),
        "initialization_sha256": recipe.initialization_sha256,
        "estimated_training_hours": t1_anchor.TOTAL_OPTIMIZER_UPDATES * seconds / 3600,
        "checkpoint_overhead_and_final_evaluation_excluded": True,
    }
    receipt = arguments.workspace / "cuda_training_smoke_v2.json"
    temporary = receipt.with_name(f".{receipt.name}.tmp")
    temporary.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, receipt)
    return receipt_payload


def _publish_training(arguments: argparse.Namespace, recipe: t1_anchor.AnchorRecipe,
                      manifest: dict[str, object], command: list[str], resume_path: Path,
                      payload: dict[str, object]) -> dict[str, object]:
    if not resume_path.is_file() or payload["checkpoint_sha256"] != longtail.sha256_file(resume_path):
        raise t1_anchor.AnchorError("Training session lacks a valid atomic resume checkpoint.")
    if int(payload["global_step"]) < t1_anchor.TOTAL_OPTIMIZER_UPDATES:
        return {"training": payload, "state": "INCOMPLETE RESUMABLE"}
    if int(payload["global_step"]) != t1_anchor.TOTAL_OPTIMIZER_UPDATES:
        raise t1_anchor.AnchorError("Training exceeded the preregistered update budget.")
    alias = arguments.workspace / f"t1_{arguments.condition}.pth"
    checkpoint_sha = t1_anchor.copy_checkpoint(resume_path, alias)
    metadata = {
        "schema": t1_anchor.ANCHOR_SCHEMA, "recipe_version": t1_anchor.RECIPE_VERSION,
        "condition": arguments.condition, "manifest_sha256": manifest["scientific_sha256"],
        "owl_commit": recipe.owl_commit, "prob_commit": recipe.prob_commit,
        "initialization_sha256": recipe.initialization_sha256,
        "recipe_fingerprint": recipe.fingerprint(), "recipe": recipe.payload(),
        "seed": recipe.seed, "reference_epochs": recipe.reference_epochs,
        "global_step": t1_anchor.TOTAL_OPTIMIZER_UPDATES,
        "optimizer_steps": t1_anchor.TOTAL_OPTIMIZER_UPDATES,
        "image_presentations": t1_anchor.TOTAL_IMAGE_PRESENTATIONS,
        "checkpoint": str(alias), "checkpoint_sha256": checkpoint_sha,
        "class_order": list(recipe.class_order), "train_objects": int(manifest["selected_objects"]),
        "train_images": int(manifest["selected_images"]),
        "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
        "started_at": payload["started_at"],
        "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "gpu": payload["gpu"],
        "torch_version": payload["torch_version"],
        "torchvision_version": recipe.torchvision_version,
        "python_version": recipe.python_version, "cuda_version": payload["cuda_version"],
        "msda": {"backend": payload["msda"], "extension": payload["msda_extension"]},
        "command": command,
        "resume_guarantee": "exact global step, deterministic batch suffix, and full RNG state",
    }
    t1_anchor.validate_training_metadata(metadata)
    t1_anchor.write_json_once_or_verify(alias.with_suffix(".metadata.json"), metadata)
    return {"training": payload, "metadata": metadata,
            "state": "TRAINING COMPLETE; FINAL EVALUATION REQUIRED"}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--condition", choices=t1_anchor.PRIMARY_CONDITIONS, required=True)
    command.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    command.add_argument("--prob-root", type=Path, required=True)
    command.add_argument("--workspace", type=Path, required=True)
    command.add_argument("--initialization", type=Path, required=True)
    command.add_argument("--initialization-sha", required=True)
    command.add_argument("--owl-commit", required=True)
    command.add_argument("--python", default=sys.executable)
    command.add_argument("--smoke-only", action="store_true")
    command.add_argument("--benchmark-iterations", type=int, default=20)
    command.add_argument("--stop-at-unix", type=float)
    command.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    command.add_argument("--execute", action="store_true")
    return command


def main() -> int:
    try:
        arguments = parser().parse_args()
        if arguments.benchmark_iterations < 2:
            raise t1_anchor.AnchorError("Benchmark requires at least two measured updates.")
        report = execute(arguments)
    except (t1_anchor.AnchorError, longtail.LongTailError, OSError,
            json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("execute") and report.get("smoke_only") and "smoke" in report:
        print("CONTROLLED LT ANCHOR RECIPE V2 BENCHMARK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
