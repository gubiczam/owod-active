#!/usr/bin/env python3
"""Run pinned PROB with local-only experiment logging.

Pinned PROB passes its global ``wandb`` object to every training epoch even
when no W&B project is configured.  This isolated adapter preserves PROB's
training path while replacing only that external side effect with a no-op.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


class DisabledWandb:
    """The two methods/attributes reachable in pinned PROB's training path."""

    config = None

    @staticmethod
    def init(*_arguments: object, **_keywords: object) -> None:
        raise RuntimeError("W&B initialization is forbidden for controlled T1 anchors.")

    @staticmethod
    def log(*_arguments: object, **_keywords: object) -> None:
        return None


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--prob-root", type=Path, required=True)
    command.add_argument("prob_arguments", nargs=argparse.REMAINDER)
    return command


def main() -> int:
    arguments = parser().parse_args()
    prob_root = arguments.prob_root.resolve()
    forwarded = list(arguments.prob_arguments)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if not forwarded:
        parser().error("PROB arguments are required after '--'.")
    sys.path.insert(0, str(prob_root))
    os.chdir(prob_root)
    disabled_wandb = DisabledWandb()
    sys.modules["wandb"] = disabled_wandb
    import main_open_world  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from daowod_prob_bridge import compatible_torch_load  # noqa: PLC0415

    main_open_world.wandb = disabled_wandb
    timing_path = os.environ.get("OWL_T1_TIMING_PATH")
    original_train_one_epoch = main_open_world.train_one_epoch

    def measured_train_one_epoch(*train_arguments: object, **train_keywords: object):
        model = train_arguments[0]
        data_loader = train_arguments[2]
        torch.cuda.synchronize()
        started = time.monotonic()
        result = original_train_one_epoch(*train_arguments, **train_keywords)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        nonfinite = [
            name for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
        ]
        if nonfinite:
            raise RuntimeError(f"Non-finite gradients after training batch: {nonfinite[:10]}")
        if timing_path:
            payload = {
                "iterations": len(data_loader),
                "seconds": elapsed,
                "seconds_per_iteration": elapsed / len(data_loader),
                "finite_gradients": True,
            }
            Path(timing_path).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    main_open_world.train_one_epoch = measured_train_one_epoch
    prob_parser = argparse.ArgumentParser(
        "Controlled T1 PROB training", parents=[main_open_world.get_args_parser()])
    prob_arguments = prob_parser.parse_args(forwarded)
    if prob_arguments.wandb_project:
        raise RuntimeError("Controlled T1 anchors require an empty W&B project.")
    with compatible_torch_load(torch):
        main_open_world.main(prob_arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
