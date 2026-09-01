#!/usr/bin/env python3
"""Run the exact pinned-PROB FAST benchmark or fixed-step training loop."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path


class DisabledWandb:
    config = None

    @staticmethod
    def log(*_args: object, **_kwargs: object) -> None:
        return None


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--prob-root", type=Path, required=True)
    command.add_argument("--owl-root", type=Path, required=True)
    command.add_argument("--condition", required=True)
    command.add_argument("--manifest-sha", required=True)
    command.add_argument("--initialization", type=Path, required=True)
    command.add_argument("--initialization-sha", required=True)
    command.add_argument("--train-split-sha", required=True)
    command.add_argument("--result", type=Path, required=True)
    command.add_argument("--benchmark", action="store_true")
    command.add_argument("--plan", type=Path)
    command.add_argument("--benchmark-receipt", type=Path)
    command.add_argument("--recipe-fingerprint")
    command.add_argument("--stop-at-unix", type=float)
    command.add_argument("--warmup-iterations", type=int, default=5)
    command.add_argument("--measured-iterations", type=int, default=20)
    command.add_argument("prob_arguments", nargs=argparse.REMAINDER)
    return command


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    outer = parser().parse_args()
    forwarded = list(outer.prob_arguments)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    sys.path.insert(0, str(outer.owl_root.resolve()))
    sys.path.insert(0, str(outer.prob_root.resolve()))
    os.chdir(outer.prob_root.resolve())
    sys.modules["wandb"] = DisabledWandb()

    import main_open_world
    import numpy as np
    import torch
    import torchvision
    import util.misc as utils
    from models import build_model
    from models.ops.functions import ms_deform_attn_func as msda_wrapper
    from models.ops.modules import ms_deform_attn as msda_downstream
    from torch.utils.data import DataLoader

    from owl import longtail, t1_anchor, t1_anchor_fast

    prob_parser = argparse.ArgumentParser(
        "FAST pinned PROB arguments", parents=[main_open_world.get_args_parser()])
    args = prob_parser.parse_args(forwarded)
    if args.wandb_project or args.batch_size != 2 or args.num_workers != 0:
        raise RuntimeError("FAST requires offline logging, batch size 2, and num_workers 0.")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FAST scientific execution requires CUDA.")
    if not msda_wrapper.MSDA_AVAILABLE or not msda_downstream.MSDA_AVAILABLE:
        raise RuntimeError("FAST scientific execution requires compiled MSDA.")

    plan = None
    final_updates = t1_anchor_fast.DEFAULT_UPDATES
    benchmark_sha = None
    if not outer.benchmark:
        if outer.plan is None or outer.benchmark_receipt is None or not outer.recipe_fingerprint:
            raise RuntimeError("FAST training requires frozen plan, benchmark, and recipe identity.")
        plan = json.loads(outer.plan.read_text(encoding="utf-8"))
        t1_anchor_fast.validate_plan(plan)
        final_updates = int(plan["frozen_optimizer_updates_per_condition"])
        benchmark = json.loads(outer.benchmark_receipt.read_text(encoding="utf-8"))
        t1_anchor_fast.benchmark_identity(benchmark)
        if benchmark["condition"] != outer.condition:
            raise RuntimeError("FAST benchmark receipt belongs to another condition.")
        runtime_expected = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
        if any(benchmark.get(key) != value for key, value in runtime_expected.items()):
            raise RuntimeError("FAST training runtime differs from its live benchmark.")
        benchmark_sha = longtail.sha256_file(outer.benchmark_receipt)
        if plan["benchmark_receipt_sha256"][outer.condition] != benchmark_sha:
            raise RuntimeError("Frozen FAST plan names another benchmark receipt.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda")
    model, criterion, _postprocessors, _selection = build_model(args, mode="prob")
    model.to(device)
    dataset_train, _dataset_val = main_open_world.get_datasets(args)

    def matches(name: str, keywords: list[str]) -> bool:
        return any(keyword in name for keyword in keywords)

    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad
                    and not matches(n, args.lr_backbone_names)
                    and not matches(n, args.lr_linear_proj_names)], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad
                    and matches(n, args.lr_backbone_names)], "lr": args.lr_backbone},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad
                    and matches(n, args.lr_linear_proj_names)],
         "lr": args.lr * args.lr_linear_proj_mult},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = output_dir / "resume_latest.pth"
    global_step = 0
    selected = t1_anchor_fast.ordered_indices(
        len(dataset_train), outer.condition, final_updates)
    selection_sha = hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode()).hexdigest()

    if resume_path.is_file() and not outer.benchmark:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        expected = {
            "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
            "recipe_fingerprint": outer.recipe_fingerprint,
            "manifest_sha256": outer.manifest_sha,
            "initialization_sha256": outer.initialization_sha,
            "condition": outer.condition,
            "seed": 0,
            "train_split_sha256": outer.train_split_sha,
            "prob_commit": t1_anchor.PINNED_PROB_COMMIT,
            "final_optimizer_updates": final_updates,
            "benchmark_receipt_sha256": benchmark_sha,
            "plan_fingerprint": plan["plan_fingerprint"],
            "selected_order_sha256": selection_sha,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("FAST resume checkpoint identity changed.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state["global_step"])
        if state.get("batch_offset") != global_step:
            raise RuntimeError("FAST resume batch offset differs from its completed updates.")
        scheduler = state.get("lr_scheduler", {})
        if scheduler.get("drop_update") != t1_anchor_fast.lr_drop_after_updates(final_updates) \
                or scheduler.get("last_completed_update") != global_step \
                or [float(item) for item in scheduler.get("base_lrs", [])] != base_lrs:
            raise RuntimeError("FAST resume scheduler state changed.")
        random.setstate(state["rng_state"]["python"])
        np.random.set_state(state["rng_state"]["numpy"])
        torch.set_rng_state(state["rng_state"]["torch_cpu"])
        torch.cuda.set_rng_state_all(state["rng_state"]["torch_cuda"])
    else:
        initialization = torch.load(outer.initialization, map_location="cpu", weights_only=False)
        model.load_state_dict(initialization["model"], strict=True)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    def apply_lr(update_index: int) -> None:
        scale = t1_anchor_fast.lr_scale_for_update(update_index, final_updates)
        for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base_lr * scale

    def checkpoint(reason: str) -> None:
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "lr_scheduler": {
                "type": "fast_explicit_update_space_floor_31_over_41",
                "base_lrs": base_lrs,
                "drop_update": t1_anchor_fast.lr_drop_after_updates(final_updates),
                "last_completed_update": global_step,
            },
            "global_step": global_step, "batch_offset": global_step,
            "selected_order_sha256": selection_sha,
            "sampling_seed": t1_anchor_fast.sampling_seed(outer.condition),
            "sampling_policy": t1_anchor_fast.SAMPLING_POLICY,
            "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch_cpu": torch.get_rng_state(),
                          "torch_cuda": torch.cuda.get_rng_state_all()},
            "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
            "recipe_fingerprint": outer.recipe_fingerprint,
            "manifest_sha256": outer.manifest_sha,
            "initialization_sha256": outer.initialization_sha,
            "prob_commit": t1_anchor.PINNED_PROB_COMMIT,
            "train_split_sha256": outer.train_split_sha,
            "condition": outer.condition, "seed": 0,
            "final_optimizer_updates": final_updates,
            "benchmark_receipt_sha256": benchmark_sha,
            "plan_fingerprint": plan["plan_fingerprint"],
            "reason": reason, "args": args,
        }
        temporary = resume_path.with_name(f".{resume_path.name}.tmp")
        torch.save(payload, temporary)
        verified = torch.load(temporary, map_location="cpu", weights_only=False)
        if verified["global_step"] != global_step \
                or verified["selected_order_sha256"] != selection_sha:
            raise RuntimeError("FAST atomic checkpoint validation failed.")
        os.replace(temporary, resume_path)
        atomic_json(output_dir / "resume_progress.json", {
            "schema": "controlled_t1_anchor_fast_progress_v1",
            "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
            "condition": outer.condition, "global_step": global_step,
            "final_optimizer_updates": final_updates,
            "recipe_fingerprint": outer.recipe_fingerprint,
            "plan_fingerprint": plan["plan_fingerprint"], "reason": reason,
        })

    def update(samples: object, targets: list[dict[str, object]], index: int) -> None:
        apply_lr(index)
        samples = samples.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        model.train()
        criterion.train()
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = deepcopy(criterion.weight_dict)
        losses = sum(loss_dict[key] * weight_dict[key] for key in loss_dict if key in weight_dict)
        if not math.isfinite(float(losses.detach().item())):
            raise RuntimeError("FAST weighted loss is non-finite.")
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise RuntimeError("FAST gradient is non-finite.")
        optimizer.step()

    started = datetime.datetime.now(datetime.UTC).isoformat()
    measured: list[float] = []
    data_times: list[float] = []
    compute_times: list[float] = []
    benchmark_target = outer.warmup_iterations + outer.measured_iterations
    target = benchmark_target if outer.benchmark else final_updates
    stop_reason = "complete"
    torch.cuda.reset_peak_memory_stats()
    batches = t1_anchor_fast.remaining_batches(
        len(dataset_train), outer.condition, final_updates, global_step)
    if outer.benchmark:
        batches = batches[:benchmark_target]
    loader_generator = torch.Generator()
    loader_generator.manual_seed(t1_anchor_fast.sampling_seed(outer.condition))
    loader = DataLoader(
        dataset_train, batch_sampler=batches, collate_fn=utils.collate_fn,
        num_workers=0, pin_memory=True, generator=loader_generator)
    iterator = iter(loader)
    while global_step < target:
        if global_step >= target:
            break
        if outer.stop_at_unix is not None and time.time() >= (
                outer.stop_at_unix - int(plan["safety_reserve_seconds"])):
            stop_reason = "global_gpu_budget_soft_stop"
            checkpoint(stop_reason)
            break
        data_started = time.monotonic()
        try:
            samples, targets = next(iterator)
        except StopIteration:
            break
        torch.cuda.synchronize()
        data_elapsed = time.monotonic() - data_started
        compute_started = time.monotonic()
        update(samples, targets, global_step)
        torch.cuda.synchronize()
        compute_elapsed = time.monotonic() - compute_started
        global_step += 1
        if outer.benchmark and global_step == outer.warmup_iterations:
            torch.cuda.reset_peak_memory_stats()
        elif outer.benchmark and global_step > outer.warmup_iterations:
            measured.append(data_elapsed + compute_elapsed)
            data_times.append(data_elapsed)
            compute_times.append(compute_elapsed)
        if not outer.benchmark and (
                global_step % t1_anchor_fast.CHECKPOINT_INTERVAL_UPDATES == 0
                or global_step == final_updates):
            checkpoint("training_complete" if global_step == final_updates else "periodic")

    runtime = {
        "gpu": torch.cuda.get_device_name(0), "python_version": sys.version,
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda, "msda": "compiled",
        "msda_extension": getattr(msda_wrapper.MSDA, "__file__", None),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    if outer.benchmark:
        if len(measured) != outer.measured_iterations:
            raise RuntimeError("FAST benchmark did not complete every measured update.")
        payload = {
            "schema": "controlled_t1_anchor_fast_benchmark_v1",
            "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
            "condition": outer.condition, "manifest_sha256": outer.manifest_sha,
            "initialization_sha256": outer.initialization_sha,
            "train_split_sha256": outer.train_split_sha,
            "warmup_iterations": outer.warmup_iterations,
            "measured_iterations": outer.measured_iterations,
            "seconds_per_optimizer_update": sum(measured) / len(measured),
            "seconds_per_data_batch": sum(data_times) / len(data_times),
            "seconds_per_compute_update": sum(compute_times) / len(compute_times),
            "measured_seconds": measured,
            "finite_gradients": True,
            "exact_fast_path": (
                "data+augmentation+forward+matcher+criterion+weighted_loss+backward+"
                "finite_gradients+optimizer_step"
            ),
            "started_at": started,
            "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
        } | runtime
    else:
        payload = {
            "schema": "controlled_t1_anchor_fast_session_v1",
            "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
            "condition": outer.condition, "global_step": global_step,
            "final_optimizer_updates": final_updates,
            "state": "TRAINED_PENDING_EVAL" if global_step == final_updates
            else "INCOMPLETE_RESUMABLE",
            "reason": stop_reason, "checkpoint": str(resume_path),
            "checkpoint_sha256": longtail.sha256_file(resume_path),
            "started_at": started,
            "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
        } | runtime
    atomic_json(outer.result, payload)
    print("OWL_T1_FAST_RESULT=" + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
