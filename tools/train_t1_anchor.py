#!/usr/bin/env python3
"""Launch one reviewed controlled-LT T1 anchor through pinned PROB.

The default is a command-only dry-run.  ``--execute --smoke-only`` must succeed
on the live T4 before ``--execute`` is accepted for the 41-epoch run.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import bridge, longtail, t1_anchor  # noqa: E402

DEFAULT_MANIFEST_ROOT = ROOT / "data" / "reference" / "longtail"


def git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def load_checkpoint_summary(python: str, prob_root: Path, checkpoint: Path) -> dict[str, object]:
    code = (
        "import json,torch; p=torch.load(r'" + str(checkpoint) + "',map_location='cpu',"
        "weights_only=False); print(json.dumps({'epoch':p.get('epoch'),"
        "'keys':sorted(p),'model_keys':len(p.get('model',{})),'has_optimizer':"
        "'optimizer' in p,'has_scheduler':'lr_scheduler' in p}))"
    )
    result = subprocess.run(
        [python, "-c", code], cwd=prob_root, text=True, capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(f"Cannot validate PROB checkpoint: {result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def verify_initialization_contents(
    python: str, prob_root: Path, checkpoint: Path, expected_state_sha: str,
    *, strict_model: bool,
) -> dict[str, object]:
    code = f"""
import argparse, json, sys, torch
sys.path.insert(0, {str(ROOT)!r})
from owl.t1_anchor import model_state_sha256
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
state_sha = model_state_sha256(checkpoint['model'])
payload = {{
    'epoch': checkpoint.get('epoch'),
    'model_keys': len(checkpoint.get('model', {{}})),
    'model_state_sha256': state_sha,
    'has_optimizer': 'optimizer' in checkpoint,
    'has_scheduler': 'lr_scheduler' in checkpoint,
    'strict_model_load': False,
}}
if {strict_model!r}:
    import main_open_world
    from models import build_model
    args = argparse.Namespace(**checkpoint['args'])
    model, _, _, _ = build_model(args, mode='prob')
    model.load_state_dict(checkpoint['model'], strict=True)
    payload['strict_model_load'] = True
print('OWL_INIT_VERIFY=' + json.dumps(payload, sort_keys=True))
"""
    result = subprocess.run(
        [python, "-c", code, str(checkpoint)], cwd=prob_root,
        text=True, capture_output=True, check=False,
    )
    markers = [
        line.removeprefix("OWL_INIT_VERIFY=") for line in result.stdout.splitlines()
        if line.startswith("OWL_INIT_VERIFY=")
    ]
    if result.returncode or not markers:
        raise t1_anchor.AnchorError(
            "Initialization tensor/architecture validation failed:\n"
            + (result.stdout + result.stderr)[-3000:]
        )
    payload = json.loads(markers[-1])
    if payload != {
        "epoch": -1,
        "model_keys": 579,
        "model_state_sha256": expected_state_sha,
        "has_optimizer": False,
        "has_scheduler": False,
        "strict_model_load": strict_model,
    }:
        raise t1_anchor.AnchorError(f"Initialization content is not exact: {payload}.")
    return payload


def probe_cuda(python: str, prob_root: Path) -> dict[str, object]:
    code = """
import json, sys, torch, torchvision
from models.ops.functions import ms_deform_attn_func as wrapper
from models.ops.modules import ms_deform_attn as downstream
payload = {
    'torch_version': torch.__version__,
    'torchvision_version': torchvision.__version__,
    'python_version': sys.version,
    'cuda_version': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'gpu_memory_gib': (
        torch.cuda.get_device_properties(0).total_memory / (1 << 30)
        if torch.cuda.is_available() else None
    ),
    'wrapper_msda': bool(wrapper.MSDA_AVAILABLE),
    'downstream_msda': bool(downstream.MSDA_AVAILABLE),
    'extension_path': getattr(wrapper.MSDA, '__file__', None),
}
print(json.dumps(payload))
if not payload['cuda_available'] or not payload['wrapper_msda'] or not payload['downstream_msda']:
    raise SystemExit(2)
"""
    result = subprocess.run(
        [python, "-c", code], cwd=prob_root, text=True, capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(
            "A full anchor requires CUDA and compiled MSDA; probe failed:\n"
            + (result.stdout + result.stderr)[-3000:]
        )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if float(payload["gpu_memory_gib"]) < 14.0:
        raise t1_anchor.AnchorError(
            f"A T4-class run requires at least 14 GiB GPU memory; found "
            f"{payload['gpu_memory_gib']:.2f} GiB."
        )
    return payload


def recipe_for(
    arguments: argparse.Namespace,
) -> tuple[dict[str, object], t1_anchor.AnchorRecipe, dict[str, object]]:
    _, manifest = t1_anchor.condition_manifest(arguments.condition, arguments.manifest_root)
    initialization_sha = longtail.sha256_file(arguments.initialization)
    if initialization_sha != arguments.initialization_sha:
        raise t1_anchor.AnchorError("Initialization SHA-256 mismatch.")
    sidecar = arguments.initialization.with_suffix(".initialization.json")
    if not sidecar.is_file():
        raise t1_anchor.AnchorError("Initialization provenance sidecar is missing.")
    initialization = json.loads(sidecar.read_text(encoding="utf-8"))
    t1_anchor.validate_initialization_metadata(initialization, arguments.initialization)
    expected = {
        "schema": "controlled_t1_initialization_v1",
        "sha256": initialization_sha,
        "prob_commit": t1_anchor.PINNED_PROB_COMMIT,
        "dino_backbone_sha256": t1_anchor.DINO_SHA256,
        "seed": t1_anchor.TRAINING_SEED,
        "class_order": list(t1_anchor.protocol.TASK1),
    }
    for name, value in expected.items():
        if initialization.get(name) != value:
            raise t1_anchor.AnchorError(
                f"Initialization provenance field {name!r} does not match the recipe.")
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


def validate_workspace(
    arguments: argparse.Namespace, recipe: t1_anchor.AnchorRecipe,
    manifest: dict[str, object],
) -> dict[str, object]:
    workspace = arguments.workspace.resolve()
    t1_anchor.validate_anchor_workspace(
        workspace=workspace,
        condition=arguments.condition,
        output_checkpoint=workspace / f"t1_{arguments.condition}.pth",
    )
    view_path = workspace / "training_view.json"
    if not view_path.is_file():
        raise t1_anchor.AnchorError("Run the full materializing anchor preflight first.")
    view = json.loads(view_path.read_text(encoding="utf-8"))
    if view.get("condition") != arguments.condition:
        raise t1_anchor.AnchorError("Training view belongs to another condition.")
    if view.get("recipe_fingerprint") != recipe.fingerprint():
        raise t1_anchor.AnchorError("Training view was prepared for another recipe.")
    data_root = Path(str(view["data_root"]))
    persistent_data = (workspace / "data" / "OWOD").resolve()
    local_condition_data = (
        data_root.name == "OWOD"
        and data_root.parent.name == arguments.condition
        and "controlled_lt" in data_root.parts
    )
    if data_root.resolve() != persistent_data and not local_condition_data:
        raise t1_anchor.AnchorError("Training view data root is not condition-isolated.")
    expected_view = {
        "manifest_scientific_sha256": recipe.manifest_sha256,
        "images": int(manifest["selected_images"]),
        "objects": int(manifest["selected_objects"]),
        "evaluation_images": 4308,
        "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
    }
    for name, value in expected_view.items():
        if view.get(name) != value:
            raise t1_anchor.AnchorError(
                f"Training view field {name!r} differs from the reviewed recipe."
            )
    for required in (
        data_root / "Annotations",
        data_root / "JPEGImages",
        data_root / "ImageSets" / "OWDETR" / "owl_anchor_train.txt",
        data_root / "ImageSets" / "OWDETR" / f"{t1_anchor.EVALUATION_SPLIT}.txt",
    ):
        if not required.exists():
            raise t1_anchor.AnchorError(f"Prepared data root is incomplete: {required}.")
    return view


def command_for(
    arguments: argparse.Namespace, recipe: t1_anchor.AnchorRecipe,
    view: dict[str, object], *, smoke: bool, write_smoke_splits: bool = False,
) -> tuple[list[str], Path, Path]:
    workspace = arguments.workspace.resolve()
    data_root = Path(str(view["data_root"]))
    output_dir = workspace / ("smoke" if smoke else "train")
    train_set = "owl_anchor_train"
    test_set = t1_anchor.EVALUATION_SPLIT
    epochs = recipe.epochs
    eval_every = recipe.evaluation_every
    if smoke:
        train_source = data_root / "ImageSets" / "OWDETR" / "owl_anchor_train.txt"
        test_source = data_root / "ImageSets" / "OWDETR" / f"{test_set}.txt"
        train_ids = train_source.read_text(encoding="utf-8").splitlines()[
            :recipe.batch_size * arguments.benchmark_iterations
        ]
        test_ids = test_source.read_text(encoding="utf-8").splitlines()[:recipe.batch_size]
        train_set = "owl_anchor_smoke_train"
        test_set = "owl_anchor_smoke_test"
        if write_smoke_splits:
            image_sets = data_root / "ImageSets" / "OWDETR"
            (image_sets / f"{train_set}.txt").write_text(
                "\n".join(train_ids) + "\n", encoding="utf-8")
            (image_sets / f"{test_set}.txt").write_text(
                "\n".join(test_ids) + "\n", encoding="utf-8")
        epochs = 1
        eval_every = 1

    command = [
        arguments.python, "-u", str(ROOT / "tools" / "run_prob_t1_anchor.py"),
        "--prob-root", str(arguments.prob_root), "--",
        "--output_dir", str(output_dir),
        "--dataset", recipe.dataset,
        "--data_root", str(data_root),
        "--device", recipe.device,
        "--PREV_INTRODUCED_CLS", "0",
        "--CUR_INTRODUCED_CLS", "19",
        "--num_classes", str(recipe.num_classes),
        "--train_set", train_set,
        "--test_set", test_set,
        "--epochs", str(epochs),
        "--lr_drop", str(recipe.lr_drop_epoch),
        "--model_type", recipe.model_type,
        "--backbone", recipe.backbone,
        "--num_queries", str(recipe.num_queries),
        "--num_feature_levels", str(recipe.num_feature_levels),
        "--position_embedding", recipe.position_embedding,
        "--position_embedding_scale", str(recipe.position_embedding_scale),
        "--enc_layers", str(recipe.encoder_layers),
        "--dec_layers", str(recipe.decoder_layers),
        "--enc_n_points", str(recipe.encoder_attention_points),
        "--dec_n_points", str(recipe.decoder_attention_points),
        "--hidden_dim", str(recipe.hidden_dim),
        "--dim_feedforward", str(recipe.feedforward_dim),
        "--dropout", str(recipe.dropout),
        "--nheads", str(recipe.attention_heads),
        "--set_cost_class", str(recipe.matcher_class_cost),
        "--set_cost_bbox", str(recipe.matcher_bbox_cost),
        "--set_cost_giou", str(recipe.matcher_giou_cost),
        "--cls_loss_coef", str(recipe.classification_loss_coefficient),
        "--bbox_loss_coef", str(recipe.bbox_loss_coefficient),
        "--giou_loss_coef", str(recipe.giou_loss_coefficient),
        "--focal_alpha", str(recipe.focal_alpha),
        "--obj_loss_coef", str(recipe.objectness_loss_coefficient),
        "--obj_temp", str(recipe.objectness_temperature),
        "--top_unk", str(recipe.top_unknown),
        "--featdim", str(recipe.feature_dim),
        "--nc_loss_coef", str(recipe.novelty_loss_coefficient),
        "--nc_epoch", str(recipe.novelty_start_epoch),
        "--bbox_thresh", str(recipe.bbox_threshold),
        "--unk_conf_w", str(recipe.unknown_confidence_weight),
        "--batch_size", str(recipe.batch_size),
        "--lr", str(recipe.learning_rate),
        "--lr_backbone", str(recipe.backbone_learning_rate),
        "--lr_linear_proj_mult", str(
            recipe.linear_projection_learning_rate / recipe.learning_rate),
        "--weight_decay", str(recipe.weight_decay),
        "--clip_max_norm", str(recipe.clip_max_norm),
        "--seed", str(recipe.seed),
        "--num_workers", str(recipe.num_workers),
        "--eval_every", str(eval_every),
        "--wandb_project", "",
    ]
    official = output_dir / "checkpoint.pth"
    if arguments.resume and not smoke:
        if not official.is_file():
            raise t1_anchor.AnchorError(f"Cannot resume; checkpoint is absent: {official}.")
        command += ["--resume", str(official)]
    else:
        command += ["--pretrain", str(arguments.initialization.resolve())]
    return command, output_dir, official


def execute(arguments: argparse.Namespace) -> dict[str, object]:
    prob_root = arguments.prob_root.resolve()
    if git_value(ROOT, "rev-parse", "HEAD") != arguments.owl_commit:
        raise t1_anchor.AnchorError("OWL checkout does not match the reviewed commit.")
    if git_value(ROOT, "status", "--porcelain"):
        raise t1_anchor.AnchorError("Training is forbidden from an uncommitted OWL working tree.")
    if git_value(prob_root, "rev-parse", "HEAD") != t1_anchor.PINNED_PROB_COMMIT:
        raise t1_anchor.AnchorError("PROB checkout does not match the pinned commit.")
    manifest, recipe, initialization = recipe_for(arguments)
    view = validate_workspace(arguments, recipe, manifest)
    command, output_dir, official = command_for(
        arguments, recipe, view, smoke=arguments.smoke_only,
        write_smoke_splits=arguments.execute,
    )
    plan = {
        "condition": arguments.condition,
        "recipe_fingerprint": recipe.fingerprint(),
        "command": command,
        "command_shell": shlex.join(command),
        "workspace": str(arguments.workspace.resolve()),
        "output_dir": str(output_dir),
        "smoke_only": arguments.smoke_only,
        "resume": arguments.resume,
        "execute": arguments.execute,
    }
    if not arguments.execute:
        return plan

    initialization_check = verify_initialization_contents(
        arguments.python, prob_root, arguments.initialization.resolve(),
        recipe.initialization_model_state_sha256,
        strict_model=arguments.smoke_only,
    )
    cuda = probe_cuda(arguments.python, prob_root)
    for name in ("torch_version", "torchvision_version", "python_version", "cuda_version"):
        if cuda.get(name) != initialization.get(name):
            raise t1_anchor.AnchorError(
                f"Runtime {name} differs from the shared initialization environment: "
                f"{cuda.get(name)!r} != {initialization.get(name)!r}.")
    smoke_receipt = arguments.workspace / "cuda_training_smoke.json"
    if not arguments.smoke_only:
        if not smoke_receipt.is_file():
            raise t1_anchor.AnchorError("Run --execute --smoke-only before full training.")
        smoke = json.loads(smoke_receipt.read_text(encoding="utf-8"))
        if smoke.get("recipe_fingerprint") != recipe.fingerprint():
            raise t1_anchor.AnchorError("CUDA smoke belongs to another recipe.")
        if smoke.get("initialization_sha256") != recipe.initialization_sha256:
            raise t1_anchor.AnchorError("CUDA smoke used another initialization.")
        if not smoke.get(
                "forward_matcher_criterion_weighted_loss_backward_finite_gradients_optimizer"):
            raise t1_anchor.AnchorError("CUDA smoke did not prove the full training operation.")
        benchmark = smoke.get("benchmark", {})
        if not benchmark.get("finite_gradients") or float(
                benchmark.get("seconds_per_iteration", 0)) <= 0:
            raise t1_anchor.AnchorError("CUDA smoke has no valid exact-path benchmark.")
    if output_dir.exists() and not arguments.resume:
        raise t1_anchor.AnchorError(f"Refusing to overwrite existing run directory {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.UTC).isoformat()
    start_clock = time.monotonic()
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(recipe.seed)
    timing_path = output_dir / "benchmark_timing.json"
    if arguments.smoke_only:
        environment["OWL_T1_TIMING_PATH"] = str(timing_path)
    result = subprocess.run(command, cwd=prob_root, env=environment, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(f"Pinned PROB training exited with status {result.returncode}.")
    summary = load_checkpoint_summary(arguments.python, prob_root, official)
    expected_epoch = 0 if arguments.smoke_only else recipe.epochs - 1
    if summary.get("epoch") != expected_epoch or not summary.get("has_optimizer") \
            or not summary.get("has_scheduler") or summary.get("model_keys") != 579:
        raise t1_anchor.AnchorError(f"Incomplete PROB checkpoint: {summary}.")
    ended = datetime.datetime.now(datetime.UTC).isoformat()
    if arguments.smoke_only:
        if not timing_path.is_file():
            raise t1_anchor.AnchorError("Exact-path smoke did not produce benchmark timing.")
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if timing.get("iterations") != arguments.benchmark_iterations \
                or not timing.get("finite_gradients"):
            raise t1_anchor.AnchorError(f"Invalid exact-path benchmark receipt: {timing}.")
        reload_metrics = arguments.workspace / "smoke_reload_evaluation.json"
        instrument = bridge.Bridge(
            prob_root=prob_root,
            data_root=Path(str(view["data_root"])),
            device="cuda",
            num_workers=recipe.num_workers,
            seed=recipe.seed,
            log_dir=arguments.workspace / "logs",
        )
        instrument.evaluate(
            checkpoint=official,
            test_set="owl_anchor_smoke_test",
            output=reload_metrics,
            n_prev=0,
            n_current=19,
            detections=False,
        )
        if not reload_metrics.is_file():
            raise t1_anchor.AnchorError("Reloaded smoke checkpoint evaluation did not complete.")
        receipt = {
            "schema": "controlled_t1_cuda_smoke_v1",
            "condition": arguments.condition,
            "recipe_fingerprint": recipe.fingerprint(),
            "initialization_sha256": recipe.initialization_sha256,
            "checkpoint_sha256": longtail.sha256_file(official),
            "checkpoint": summary,
            "initialization_check": initialization_check,
            "cuda": cuda,
            "benchmark": timing,
            "estimated_training_hours": {
                condition: (
                    t1_anchor.optimizer_steps(
                        int(t1_anchor.condition_manifest(
                            condition, arguments.manifest_root)[1]["selected_images"]),
                        t1_anchor.AnchorRecipe(
                            **(recipe.payload() | {
                                "condition": condition,
                                "manifest_sha256": t1_anchor.EXPECTED_MANIFEST_SHA256[condition],
                            })
                        ),
                    ) * float(timing["seconds_per_iteration"]) / 3600.0
                )
                for condition in t1_anchor.PRIMARY_CONDITIONS
            },
            "actual_filtered_training_images": recipe.batch_size * arguments.benchmark_iterations,
            "actual_evaluation_images": recipe.batch_size,
            "forward_matcher_criterion_weighted_loss_backward_finite_gradients_optimizer": True,
            "checkpoint_save_strict_initialization_load_reload_evaluation": True,
            "reload_evaluation_sha256": longtail.sha256_file(reload_metrics),
            "started_at": started,
            "ended_at": ended,
        }
        t1_anchor.write_json_once_or_verify(smoke_receipt, receipt)
        return plan | {"smoke": receipt}

    alias = arguments.workspace / f"t1_{arguments.condition}.pth"
    checkpoint_sha = t1_anchor.copy_checkpoint(official, alias)
    metadata = {
        "schema": t1_anchor.ANCHOR_SCHEMA,
        "condition": arguments.condition,
        "manifest_sha256": manifest["scientific_sha256"],
        "owl_commit": recipe.owl_commit,
        "prob_commit": recipe.prob_commit,
        "initialization_sha256": recipe.initialization_sha256,
        "recipe_fingerprint": recipe.fingerprint(),
        "recipe": recipe.payload(),
        "seed": recipe.seed,
        "epochs": recipe.epochs,
        "optimizer_steps": t1_anchor.optimizer_steps(int(manifest["selected_images"]), recipe),
        "checkpoint": str(alias),
        "checkpoint_sha256": checkpoint_sha,
        "class_order": list(recipe.class_order),
        "train_objects": int(manifest["selected_objects"]),
        "train_images": int(manifest["selected_images"]),
        "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
        "started_at": started,
        "ended_at": ended,
        "elapsed_seconds": time.monotonic() - start_clock,
        "gpu": cuda["gpu"],
        "torch_version": cuda["torch_version"],
        "torchvision_version": cuda["torchvision_version"],
        "python_version": cuda["python_version"],
        "cuda_version": cuda["cuda_version"],
        "msda": cuda,
        "command": command,
        "resume_requested": arguments.resume,
        "resume_guarantee": "epoch/model/optimizer/scheduler restored; RNG stream is not exact",
    }
    t1_anchor.validate_training_metadata(metadata)
    t1_anchor.write_json_once_or_verify(alias.with_suffix(".metadata.json"), metadata)
    return plan | {"metadata": metadata}


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
    command.add_argument("--resume", action="store_true")
    command.add_argument("--execute", action="store_true")
    return command


def main() -> int:
    try:
        arguments = parser().parse_args()
        if arguments.benchmark_iterations < 2:
            raise t1_anchor.AnchorError("Benchmark smoke requires at least two iterations.")
        report = execute(arguments)
    except (t1_anchor.AnchorError, longtail.LongTailError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("execute") and report.get("smoke_only") and "smoke" in report:
        print("CONTROLLED LT ANCHOR PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
