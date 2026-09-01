#!/usr/bin/env python3
"""Execute Recipe V2's fixed-step loop using the pinned PROB model and losses."""

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


def outer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-root", type=Path, required=True)
    parser.add_argument("--owl-root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--recipe-fingerprint", required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--initialization-sha", required=True)
    parser.add_argument("--train-split-sha", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--stop-at-unix", type=float)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("prob_arguments", nargs=argparse.REMAINDER)
    return parser


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    outer = outer_parser().parse_args()
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
    import util.misc as utils
    from models import build_model
    from models.ops.functions import ms_deform_attn_func as msda_wrapper
    from models.ops.modules import ms_deform_attn as msda_downstream
    from torch.utils.data import DataLoader

    from owl import longtail, t1_anchor

    parser = argparse.ArgumentParser(
        "Recipe V2 pinned PROB arguments", parents=[main_open_world.get_args_parser()])
    args = parser.parse_args(forwarded)
    if args.wandb_project or args.batch_size != 2 or args.num_workers != 0:
        raise RuntimeError("Recipe V2 requires offline logging, batch size 2, and num_workers 0.")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Recipe V2 scientific execution requires CUDA.")
    if not msda_wrapper.MSDA_AVAILABLE or not msda_downstream.MSDA_AVAILABLE:
        raise RuntimeError("Recipe V2 scientific execution requires compiled MSDA.")

    torch.manual_seed(t1_anchor.TRAINING_SEED)
    torch.cuda.manual_seed_all(t1_anchor.TRAINING_SEED)
    np.random.seed(t1_anchor.TRAINING_SEED)
    random.seed(t1_anchor.TRAINING_SEED)
    device = torch.device("cuda")
    model, criterion, _postprocessors, _exemplar_selection = build_model(args, mode="prob")
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
    if resume_path.is_file() and not outer.benchmark:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        expected = {
            "recipe_version": t1_anchor.RECIPE_VERSION,
            "recipe_fingerprint": outer.recipe_fingerprint,
            "manifest_sha256": outer.manifest_sha,
            "initialization_sha256": outer.initialization_sha,
            "condition": outer.condition,
            "seed": t1_anchor.TRAINING_SEED,
            "train_split_sha256": outer.train_split_sha,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Resume checkpoint identity differs from Recipe V2.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state["global_step"])
        scheduler_state = state.get("lr_scheduler", {})
        if scheduler_state.get("type") != "explicit_update_space_step_drop" \
                or scheduler_state.get("drop_update") != t1_anchor.LR_DROP_UPDATE \
                or scheduler_state.get("last_completed_update") != global_step \
                or [float(value) for value in scheduler_state.get("base_lrs", [])] != base_lrs:
            raise RuntimeError("Resume checkpoint has incompatible update-space LR state.")
        saved_epoch, saved_offset = t1_anchor.reference_position(global_step)
        if (state.get("reference_epoch"), state.get("step_in_reference_epoch")) != (
                saved_epoch, saved_offset):
            raise RuntimeError("Resume checkpoint contains an inconsistent global position.")
        random.setstate(state["rng_state"]["python"])
        np.random.set_state(state["rng_state"]["numpy"])
        torch.set_rng_state(state["rng_state"]["torch_cpu"])
        torch.cuda.set_rng_state_all(state["rng_state"]["torch_cuda"])
    else:
        initialization = torch.load(outer.initialization, map_location="cpu", weights_only=False)
        model.load_state_dict(initialization["model"], strict=True)
        # The shared initialization contains weights only. Reset the complete stochastic
        # stream after construction so every condition starts from the same seed policy.
        torch.manual_seed(t1_anchor.TRAINING_SEED)
        torch.cuda.manual_seed_all(t1_anchor.TRAINING_SEED)
        np.random.seed(t1_anchor.TRAINING_SEED)
        random.seed(t1_anchor.TRAINING_SEED)

    def apply_lr(update_index: int) -> None:
        scale = t1_anchor.learning_rate_scale_for_update(update_index)
        for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base_lr * scale

    def checkpoint(reason: str) -> None:
        reference_epoch, offset = t1_anchor.reference_position(global_step)
        next_indices = () if global_step == t1_anchor.TOTAL_OPTIMIZER_UPDATES else \
            t1_anchor.reference_epoch_indices(
                len(dataset_train), outer.condition, reference_epoch, t1_anchor.TRAINING_SEED)
        sampler_fingerprint = hashlib.sha256(
            json.dumps(next_indices, separators=(",", ":")).encode()).hexdigest()
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "lr_scheduler": {
                "type": "explicit_update_space_step_drop",
                "base_lrs": base_lrs,
                "drop_update": t1_anchor.LR_DROP_UPDATE,
                "next_update_scale": None if global_step == t1_anchor.TOTAL_OPTIMIZER_UPDATES
                else t1_anchor.learning_rate_scale_for_update(global_step),
                "last_completed_update": global_step,
            },
            "global_step": global_step, "reference_epoch": reference_epoch,
            "step_in_reference_epoch": offset,
            "sampling": {
                "algorithm": "python_random_sample_without_replacement_v1",
                "reference_epoch_seed": None if global_step == t1_anchor.TOTAL_OPTIMIZER_UPDATES
                else t1_anchor.reference_epoch_seed(outer.condition, reference_epoch),
                "reference_epoch_permutation_sha256": sampler_fingerprint,
            },
            "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch_cpu": torch.get_rng_state(),
                          "torch_cuda": torch.cuda.get_rng_state_all()},
            "recipe_version": t1_anchor.RECIPE_VERSION,
            "recipe_fingerprint": outer.recipe_fingerprint,
            "manifest_sha256": outer.manifest_sha,
            "initialization_sha256": outer.initialization_sha,
            "train_split_sha256": outer.train_split_sha,
            "condition": outer.condition, "seed": t1_anchor.TRAINING_SEED,
            "reason": reason, "args": args,
        }
        temporary = resume_path.with_name(f".{resume_path.name}.tmp")
        torch.save(payload, temporary)
        verified = torch.load(temporary, map_location="cpu", weights_only=False)
        if verified["global_step"] != global_step or verified["recipe_fingerprint"] != outer.recipe_fingerprint:
            raise RuntimeError("Atomic checkpoint validation failed.")
        os.replace(temporary, resume_path)

    def one_update(samples: object, targets: list[dict[str, object]], update_index: int) -> float:
        apply_lr(update_index)
        samples = samples.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        model.train()
        criterion.train()
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = deepcopy(criterion.weight_dict)
        if update_index // t1_anchor.UPDATES_PER_REFERENCE_EPOCH < args.nc_epoch:
            for key in weight_dict:
                if "NC" in key:
                    weight_dict[key] = 0
        losses = sum(loss_dict[key] * weight_dict[key] for key in loss_dict if key in weight_dict)
        if not math.isfinite(float(losses.detach().item())):
            raise RuntimeError("Non-finite weighted training loss.")
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise RuntimeError("Non-finite gradient in Recipe V2 training.")
        optimizer.step()
        return float(losses.detach().item())

    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    runtime = {
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "msda": "compiled",
        "msda_extension": getattr(msda_wrapper.MSDA, "__file__", None),
    }
    measured: list[float] = []
    compute_measured: list[float] = []
    data_measured: list[float] = []
    benchmark_total = outer.warmup_iterations + outer.measured_iterations
    target_step = benchmark_total if outer.benchmark else t1_anchor.TOTAL_OPTIMIZER_UPDATES
    stop_reason = "complete"
    while global_step < target_step:
        batches = t1_anchor.remaining_reference_batches(
            len(dataset_train), outer.condition, global_step, t1_anchor.TRAINING_SEED)
        if outer.benchmark:
            batches = batches[:target_step - global_step]
        loader = DataLoader(dataset_train, batch_sampler=batches, collate_fn=utils.collate_fn,
                            num_workers=0, pin_memory=True)
        iterator = iter(loader)
        while global_step < target_step:
            if outer.stop_at_unix is not None and time.time() >= (
                    outer.stop_at_unix - t1_anchor.SOFT_STOP_RESERVE_SECONDS):
                stop_reason = "gpu_budget_soft_stop"
                checkpoint(stop_reason)
                break
            data_started = time.monotonic()
            try:
                samples, targets = next(iterator)
            except StopIteration:
                break
            torch.cuda.synchronize()
            data_elapsed = time.monotonic() - data_started
            update_started = time.monotonic()
            one_update(samples, targets, global_step)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - update_started
            global_step += 1
            if outer.benchmark and global_step > outer.warmup_iterations:
                measured.append(data_elapsed + elapsed)
                compute_measured.append(elapsed)
                data_measured.append(data_elapsed)
            if not outer.benchmark and (
                    global_step % t1_anchor.CHECKPOINT_INTERVAL_UPDATES == 0
                    or global_step % t1_anchor.UPDATES_PER_REFERENCE_EPOCH == 0
                    or global_step == t1_anchor.TOTAL_OPTIMIZER_UPDATES):
                checkpoint("periodic" if global_step < target_step else "training_complete")
        if stop_reason != "complete":
            break

    if outer.benchmark:
        if len(measured) != outer.measured_iterations:
            raise RuntimeError("Benchmark did not complete all measured iterations.")
        payload = {
            "schema": "controlled_t1_cuda_benchmark_v2",
            "recipe_version": t1_anchor.RECIPE_VERSION,
            "condition": outer.condition,
            "warmup_iterations": outer.warmup_iterations,
            "measured_iterations": outer.measured_iterations,
            "seconds_per_optimizer_update": sum(measured) / len(measured),
            "seconds_per_compute_update": sum(compute_measured) / len(compute_measured),
            "seconds_per_data_batch": sum(data_measured) / len(data_measured),
            "measured_seconds": measured,
            "finite_gradients": True,
            "real_model_forward_matcher_criterion_backward_optimizer": True,
            "started_at": started_at,
            "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
        } | runtime
    else:
        payload = {
            "schema": "controlled_t1_training_session_v2",
            "recipe_version": t1_anchor.RECIPE_VERSION,
            "condition": outer.condition,
            "global_step": global_step,
            "reference_epoch": t1_anchor.reference_position(global_step)[0],
            "step_in_reference_epoch": t1_anchor.reference_position(global_step)[1],
            "state": "TRAINING COMPLETE; FINAL EVALUATION REQUIRED" if global_step == target_step
            else "INCOMPLETE RESUMABLE",
            "reason": stop_reason,
            "checkpoint": str(resume_path),
            "checkpoint_sha256": longtail.sha256_file(resume_path),
            "started_at": started_at,
            "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
        } | runtime
    atomic_json(outer.result, payload)
    print("OWL_T1_V2_RESULT=" + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
