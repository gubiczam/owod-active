"""Contracts for the compute-limited controlled T1 anchor FAST experiment."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import random
import socket
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from owl import longtail, protocol, t1_anchor

FAST_RECIPE_VERSION = "controlled_t1_anchor_fast_v1"
FAST_SCHEMA = FAST_RECIPE_VERSION
FAST_PROTOCOL_VERSION = 1
DEFAULT_UPDATES = 12_000
MAX_UNIQUE_UPDATES = 17_730
BATCH_SIZE = 2
DEFAULT_TOTAL_GPU_BUDGET_HOURS = 14.0
MAX_TOTAL_GPU_BUDGET_HOURS = 15.0
DEFAULT_SAFETY_RESERVE_SECONDS = 1_800
DEFAULT_FINAL_EVALUATION_SECONDS = 1_800
DEFAULT_SETUP_SECONDS = 900
CHECKPOINT_INTERVAL_UPDATES = 1_000
CONDITION_ORDER = ("lt100", "lt50", "lt10")
SAMPLING_POLICY = "single_deterministic_uniform_sample_without_replacement"

SCIENTIFIC_CONFIG = {
    "architecture": "PROB Deformable-DETR",
    "model_type": "prob",
    "backbone": "dino_resnet50",
    "pretrained_backbone": True,
    "dino_backbone_sha256": t1_anchor.DINO_SHA256,
    "num_classes": 81,
    "class_order": list(protocol.TASK1),
    "dataset": "OWDETR",
    "optimizer": "AdamW",
    "learning_rate": 2e-4,
    "backbone_learning_rate": 2e-5,
    "linear_projection_learning_rate": 2e-5,
    "weight_decay": 1e-4,
    "lr_drop_gamma": 0.1,
    "clip_max_norm": 0.1,
    "num_queries": 100,
    "num_feature_levels": 4,
    "with_box_refine": False,
    "two_stage": False,
    "masks": False,
    "dilation": False,
    "unmatched_boxes": False,
    "invalid_class_logits": False,
    "novelty_classification_branch": False,
    "freeze_probabilistic_model": False,
    "remove_difficult": False,
    "cache_mode": False,
    "position_embedding": "sine",
    "position_embedding_scale": 6.283185307179586,
    "encoder_layers": 6,
    "decoder_layers": 6,
    "encoder_attention_points": 4,
    "decoder_attention_points": 4,
    "hidden_dim": 256,
    "feedforward_dim": 1024,
    "dropout": 0.1,
    "attention_heads": 8,
    "matcher_class_cost": 2,
    "matcher_bbox_cost": 5,
    "matcher_giou_cost": 2,
    "classification_loss_coefficient": 2,
    "bbox_loss_coefficient": 5,
    "giou_loss_coefficient": 2,
    "objectness_loss_coefficient": 1e-3,
    "objectness_temperature": 1,
    "focal_alpha": 0.25,
    "auxiliary_loss": True,
    "top_unknown": 5,
    "feature_dim": 1024,
    "novelty_loss_coefficient": 2,
    "novelty_start_epoch": 0,
    "bbox_threshold": 0.3,
    "unknown_confidence_weight": 1,
    "augmentation": (
        "RandomHorizontalFlip; RandomSelect(RandomResize[480..800], "
        "RandomResize[400,500,600]+RandomSizeCrop[384,600]+RandomResize[480..800]); "
        "max_size=1333; ImageNet normalization"
    ),
    "replay": "none",
    "active_selection": "none",
    "device": "cuda",
    "num_workers": 0,
    "duration_policy": "fixed_global_optimizer_updates",
    "checkpoint_interval_updates": CHECKPOINT_INTERVAL_UPDATES,
    "class_balanced_sampler": False,
    "oversampling": False,
    "loss_reweighting": False,
    "evaluation_split": t1_anchor.EVALUATION_SPLIT,
    "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
    "evaluation_cadence": "mandatory_final_only",
}


def lr_drop_after_updates(final_updates: int) -> int:
    validate_update_count(final_updates)
    return final_updates * 31 // 41


def validate_update_count(final_updates: int) -> None:
    if not isinstance(final_updates, int) or not 1 <= final_updates <= MAX_UNIQUE_UPDATES:
        raise t1_anchor.AnchorError(
            f"FAST updates must be an integer from 1 through {MAX_UNIQUE_UPDATES}.")


@dataclass(frozen=True)
class FastRecipe:
    condition: str
    manifest_sha256: str
    owl_commit: str
    initialization_sha256: str
    initialization_model_state_sha256: str
    python_version: str
    torch_version: str
    torchvision_version: str
    cuda_version: str
    final_optimizer_updates: int = DEFAULT_UPDATES
    recipe_version: str = FAST_RECIPE_VERSION
    schema: str = FAST_SCHEMA
    protocol_version: int = FAST_PROTOCOL_VERSION
    prob_commit: str = t1_anchor.PINNED_PROB_COMMIT
    seed: int = t1_anchor.TRAINING_SEED
    batch_size: int = BATCH_SIZE
    sampling_policy: str = SAMPLING_POLICY

    @property
    def total_image_presentations(self) -> int:
        return self.final_optimizer_updates * self.batch_size

    @property
    def lr_drop_update(self) -> int:
        return lr_drop_after_updates(self.final_optimizer_updates)

    def validate(self) -> None:
        if self.recipe_version != FAST_RECIPE_VERSION or self.schema != FAST_SCHEMA:
            raise t1_anchor.AnchorError("A non-FAST receipt cannot launch FAST science.")
        if self.condition not in CONDITION_ORDER:
            raise t1_anchor.AnchorError("FAST condition must be LT-10, LT-50, or LT-100.")
        if self.manifest_sha256 != t1_anchor.EXPECTED_MANIFEST_SHA256[self.condition]:
            raise t1_anchor.AnchorError("FAST manifest differs from the reviewed controlled LT input.")
        if self.prob_commit != t1_anchor.PINNED_PROB_COMMIT:
            raise t1_anchor.AnchorError("FAST changed the pinned PROB commit.")
        if self.seed != 0 or self.batch_size != 2:
            raise t1_anchor.AnchorError("FAST requires seed 0 and batch size 2.")
        if self.sampling_policy != SAMPLING_POLICY:
            raise t1_anchor.AnchorError("FAST unique-image sampling policy changed.")
        validate_update_count(self.final_optimizer_updates)
        for value, length, label in (
            (self.owl_commit, 40, "owl_commit"),
            (self.initialization_sha256, 64, "initialization_sha256"),
            (self.initialization_model_state_sha256, 64, "initialization_model_state_sha256"),
        ):
            t1_anchor._lower_hex(value, length, label)
        if not all((self.python_version, self.torch_version,
                    self.torchvision_version, self.cuda_version)):
            raise t1_anchor.AnchorError("FAST recipe lacks its exact runtime identity.")

    def payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self) | {
            "total_image_presentations": self.total_image_presentations,
            "lr_drop_update": self.lr_drop_update,
            "lr_schedule": "explicit_update_space_floor_31_over_41",
            "scientific_config": SCIENTIFIC_CONFIG,
            "scientific_wording": (
                "fixed-compute controlled comparison; not a historical reproduction "
                "and not evidence of convergence"
            ),
        }

    def fingerprint(self) -> str:
        return longtail.sha256_bytes(longtail.canonical_json_bytes(self.payload()))


def sampling_seed(condition: str, seed: int = 0) -> int:
    if condition not in CONDITION_ORDER or seed != 0:
        raise t1_anchor.AnchorError("Invalid FAST deterministic sampling identity.")
    identity = f"{FAST_RECIPE_VERSION}\0{seed}\0{condition}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")


def ordered_indices(
    dataset_size: int, condition: str, final_updates: int, seed: int = 0,
) -> tuple[int, ...]:
    validate_update_count(final_updates)
    required = BATCH_SIZE * final_updates
    if required > dataset_size:
        raise t1_anchor.AnchorError("FAST unique-image selection exceeds this condition dataset.")
    generator = random.Random(sampling_seed(condition, seed))
    permutation = list(range(dataset_size))
    generator.shuffle(permutation)
    return tuple(permutation[:required])


def remaining_batches(
    dataset_size: int, condition: str, final_updates: int, global_step: int, seed: int = 0,
) -> tuple[tuple[int, int], ...]:
    if not 0 <= global_step <= final_updates:
        raise t1_anchor.AnchorError("FAST global_step is outside its frozen budget.")
    selected = ordered_indices(dataset_size, condition, final_updates, seed)
    batches = tuple(zip(selected[::2], selected[1::2], strict=True))
    return batches[global_step:]


def lr_scale_for_update(update_index: int, final_updates: int) -> float:
    validate_update_count(final_updates)
    if not 0 <= update_index < final_updates:
        raise t1_anchor.AnchorError("FAST update index is outside its frozen budget.")
    return 1.0 if update_index < lr_drop_after_updates(final_updates) else 0.1


def benchmark_identity(receipt: Mapping[str, object]) -> dict[str, object]:
    forbidden = [key for key in receipt if "ap" in key.lower() or "metric" in key.lower()]
    if forbidden:
        raise t1_anchor.AnchorError(f"FAST benchmark receipt contains scientific results: {forbidden}.")
    required = {
        "schema", "recipe_version", "condition", "warmup_iterations",
        "measured_iterations", "seconds_per_optimizer_update", "peak_gpu_memory_bytes",
        "gpu", "python_version", "torch_version", "torchvision_version", "cuda_version",
        "manifest_sha256",
        "initialization_sha256", "train_split_sha256",
        "finite_gradients", "exact_fast_path",
    }
    missing = required - set(receipt)
    if missing:
        raise t1_anchor.AnchorError(f"FAST benchmark receipt lacks {sorted(missing)}.")
    condition = str(receipt["condition"])
    if condition not in CONDITION_ORDER:
        raise t1_anchor.AnchorError("FAST benchmark names an invalid condition.")
    expected = {
        "schema": "controlled_t1_anchor_fast_benchmark_v1",
        "recipe_version": FAST_RECIPE_VERSION,
        "warmup_iterations": 5,
        "measured_iterations": 20,
        "manifest_sha256": t1_anchor.EXPECTED_MANIFEST_SHA256[condition],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise t1_anchor.AnchorError("FAST benchmark identity is not exact.")
    if float(receipt["seconds_per_optimizer_update"]) <= 0 \
            or int(receipt["peak_gpu_memory_bytes"]) <= 0:
        raise t1_anchor.AnchorError("FAST benchmark timing or memory is invalid.")
    expected_path = (
        "data+augmentation+forward+matcher+criterion+weighted_loss+backward+"
        "finite_gradients+optimizer_step"
    )
    if receipt["finite_gradients"] is not True or receipt["exact_fast_path"] != expected_path:
        raise t1_anchor.AnchorError("FAST benchmark did not exercise the complete training path.")
    return {key: receipt[key] for key in sorted(required)}


def plan_experiment(
    receipts: Mapping[str, Mapping[str, object]], *, total_budget_hours: float = 14.0,
    default_updates: int = DEFAULT_UPDATES,
    final_evaluation_seconds: int = DEFAULT_FINAL_EVALUATION_SECONDS,
    safety_reserve_seconds: int = DEFAULT_SAFETY_RESERVE_SECONDS,
    setup_seconds: int = DEFAULT_SETUP_SECONDS,
) -> dict[str, object]:
    if set(receipts) != set(CONDITION_ORDER):
        raise t1_anchor.AnchorError("FAST planning requires benchmark receipts for all conditions.")
    if not 0 < total_budget_hours <= MAX_TOTAL_GPU_BUDGET_HOURS:
        raise t1_anchor.AnchorError("TOTAL_GPU_BUDGET_HOURS must be positive and at most 15.0.")
    if safety_reserve_seconds < 1_200:
        raise t1_anchor.AnchorError("FAST planning requires at least twenty minutes of reserve.")
    if final_evaluation_seconds <= 0 or setup_seconds < 0:
        raise t1_anchor.AnchorError("FAST evaluation and setup estimates must be valid.")
    validate_update_count(default_updates)
    identities = {condition: benchmark_identity(receipts[condition])
                  for condition in CONDITION_ORDER}
    runtime_fields = ("gpu", "python_version", "torch_version", "torchvision_version",
                      "cuda_version")
    if any(len({identity[field] for identity in identities.values()}) != 1
           for field in runtime_fields):
        raise t1_anchor.AnchorError("FAST condition benchmarks do not share one live runtime.")
    seconds_per_update = {
        condition: float(identities[condition]["seconds_per_optimizer_update"])
        for condition in CONDITION_ORDER
    }
    slowest = max(seconds_per_update.values())
    total_seconds = total_budget_hours * 3600
    fixed_overhead = 3 * final_evaluation_seconds + safety_reserve_seconds + setup_seconds
    available_training = total_seconds - fixed_overhead
    capacity = math.floor(available_training / (3 * slowest)) if available_training > 0 else 0
    candidate = min(default_updates, MAX_UNIQUE_UPDATES, capacity)
    frozen_updates = candidate if candidate == default_updates else candidate // 1_000 * 1_000
    go = frozen_updates >= 1_000
    training_seconds = frozen_updates * slowest * 3 if go else 0.0
    projected = training_seconds + fixed_overhead
    if go and projected > total_seconds:
        raise t1_anchor.AnchorError("FAST planner produced a budget-overrunning plan.")
    now = datetime.datetime.now(datetime.UTC)
    payload = {
        "schema": "controlled_t1_anchor_fast_plan_v1",
        "recipe_version": FAST_RECIPE_VERSION,
        "decision": "GO" if go else "NO-GO",
        "decision_basis": "runtime_only_no_AP_or_scientific_metrics",
        "condition_order": list(CONDITION_ORDER),
        "default_optimizer_updates_per_condition": default_updates,
        "frozen_optimizer_updates_per_condition": frozen_updates if go else None,
        "batch_size": BATCH_SIZE,
        "image_presentations_per_condition": BATCH_SIZE * frozen_updates if go else None,
        "lr_drop_update": lr_drop_after_updates(frozen_updates) if go else None,
        "sampling_policy": SAMPLING_POLICY,
        "max_unique_optimizer_updates": MAX_UNIQUE_UPDATES,
        "seconds_per_update": seconds_per_update,
        "planning_seconds_per_update": slowest,
        "training_seconds_per_condition": frozen_updates * slowest if go else None,
        "final_evaluation_seconds_per_condition": final_evaluation_seconds,
        "setup_seconds": setup_seconds,
        "safety_reserve_seconds": safety_reserve_seconds,
        "projected_total_seconds": projected,
        "total_gpu_budget_hours": total_budget_hours,
        "latest_safe_expected_completion_utc": (
            now + datetime.timedelta(seconds=projected)).isoformat() if go else None,
        "benchmark_identities": identities,
        "benchmark_receipt_sha256": {
            condition: longtail.sha256_bytes(
                (json.dumps(dict(receipts[condition]), indent=2, sort_keys=True) + "\n").encode())
            for condition in CONDITION_ORDER
        },
        "created_at": now.isoformat(),
    }
    payload["plan_fingerprint"] = longtail.sha256_bytes(longtail.canonical_json_bytes(payload))
    return payload


def validate_plan(payload: Mapping[str, object]) -> None:
    if payload.get("schema") != "controlled_t1_anchor_fast_plan_v1" \
            or payload.get("recipe_version") != FAST_RECIPE_VERSION \
            or payload.get("decision") != "GO":
        raise t1_anchor.AnchorError("FAST training requires a frozen GO plan receipt.")
    updates = int(payload["frozen_optimizer_updates_per_condition"])
    validate_update_count(updates)
    expected_constants = {
        "decision_basis": "runtime_only_no_AP_or_scientific_metrics",
        "condition_order": list(CONDITION_ORDER),
        "batch_size": BATCH_SIZE,
        "sampling_policy": SAMPLING_POLICY,
        "max_unique_optimizer_updates": MAX_UNIQUE_UPDATES,
        "image_presentations_per_condition": BATCH_SIZE * updates,
        "lr_drop_update": lr_drop_after_updates(updates),
    }
    if any(payload.get(key) != value for key, value in expected_constants.items()):
        raise t1_anchor.AnchorError("FAST plan's frozen scientific budget is inconsistent.")
    identities = payload.get("benchmark_identities")
    seconds = payload.get("seconds_per_update")
    hashes = payload.get("benchmark_receipt_sha256")
    if not isinstance(identities, Mapping) or not isinstance(seconds, Mapping) \
            or not isinstance(hashes, Mapping) \
            or set(identities) != set(CONDITION_ORDER) \
            or set(seconds) != set(CONDITION_ORDER) \
            or set(hashes) != set(CONDITION_ORDER):
        raise t1_anchor.AnchorError("FAST plan lacks all three exact benchmark identities.")
    checked_identities = {
        condition: benchmark_identity(identities[condition]) for condition in CONDITION_ORDER
    }
    runtime_fields = ("gpu", "python_version", "torch_version", "torchvision_version",
                      "cuda_version")
    if any(len({identity[field] for identity in checked_identities.values()}) != 1
           for field in runtime_fields):
        raise t1_anchor.AnchorError("FAST plan mixes benchmark runtimes.")
    if any(float(seconds[condition]) != float(
            checked_identities[condition]["seconds_per_optimizer_update"])
           for condition in CONDITION_ORDER):
        raise t1_anchor.AnchorError("FAST plan timing differs from its benchmark identities.")
    for condition in CONDITION_ORDER:
        t1_anchor._lower_hex(str(hashes[condition]), 64, "benchmark_receipt_sha256")
    slowest = max(float(value) for value in seconds.values())
    if float(payload.get("planning_seconds_per_update", 0)) != slowest:
        raise t1_anchor.AnchorError("FAST plan did not conservatively use the slowest benchmark.")
    total_seconds = float(payload["total_gpu_budget_hours"]) * 3600
    if not 0 < float(payload["total_gpu_budget_hours"]) <= MAX_TOTAL_GPU_BUDGET_HOURS \
            or int(payload["safety_reserve_seconds"]) < 1_200 \
            or int(payload["final_evaluation_seconds_per_condition"]) <= 0 \
            or int(payload["setup_seconds"]) < 0:
        raise t1_anchor.AnchorError("FAST plan has an invalid global budget or reserve.")
    fixed_overhead = (
        3 * int(payload["final_evaluation_seconds_per_condition"])
        + int(payload["safety_reserve_seconds"]) + int(payload["setup_seconds"])
    )
    capacity = math.floor((total_seconds - fixed_overhead) / (3 * slowest))
    default = int(payload["default_optimizer_updates_per_condition"])
    candidate = min(default, MAX_UNIQUE_UPDATES, capacity)
    expected_updates = candidate if candidate == default else candidate // 1_000 * 1_000
    if updates != expected_updates:
        raise t1_anchor.AnchorError("FAST frozen step count does not follow the timing-only policy.")
    projected = 3 * updates * slowest + fixed_overhead
    if float(payload.get("training_seconds_per_condition", -1)) != updates * slowest \
            or float(payload.get("projected_total_seconds", -1)) != projected \
            or projected > total_seconds:
        raise t1_anchor.AnchorError("FAST plan's projected wall-clock budget is inconsistent.")
    fingerprint = dict(payload)
    claimed = str(fingerprint.pop("plan_fingerprint", ""))
    actual = longtail.sha256_bytes(longtail.canonical_json_bytes(fingerprint))
    if claimed != actual:
        raise t1_anchor.AnchorError("FAST plan receipt fingerprint changed.")


def workspace(root: str | Path, condition: str) -> Path:
    if condition not in CONDITION_ORDER:
        raise t1_anchor.AnchorError("Invalid FAST condition workspace.")
    return Path(root) / f"t1_anchor_fast__{condition}__seed0"


def validate_workspace_path(path: str | Path, condition: str) -> None:
    if Path(path).resolve().name != f"t1_anchor_fast__{condition}__seed0":
        raise t1_anchor.AnchorError("FAST workspace is not condition- and recipe-isolated.")
    if Path(path).resolve().name.startswith("t1_anchor__"):
        raise t1_anchor.AnchorError("FAST and full V2 workspaces cannot collide.")


def validate_training_metadata(payload: Mapping[str, object]) -> None:
    required = {
        "schema", "recipe_version", "condition", "manifest_sha256", "owl_commit",
        "prob_commit", "initialization_sha256", "recipe_fingerprint", "plan_fingerprint",
        "seed", "global_step", "final_optimizer_updates", "image_presentations",
        "lr_drop_update", "checkpoint_sha256", "class_order", "evaluation_split_sha256",
        "benchmark_receipt_sha256", "recipe",
    }
    missing = required - set(payload)
    if missing:
        raise t1_anchor.AnchorError(f"FAST final metadata lacks {sorted(missing)}.")
    if payload["schema"] != FAST_SCHEMA or payload["recipe_version"] != FAST_RECIPE_VERSION:
        raise t1_anchor.AnchorError("FAST final metadata has another recipe schema.")
    condition = str(payload["condition"])
    if condition not in CONDITION_ORDER \
            or payload["manifest_sha256"] != t1_anchor.EXPECTED_MANIFEST_SHA256[condition]:
        raise t1_anchor.AnchorError("FAST final metadata changed its condition manifest.")
    updates = int(payload["final_optimizer_updates"])
    validate_update_count(updates)
    if payload["global_step"] != updates \
            or payload["image_presentations"] != BATCH_SIZE * updates \
            or payload["lr_drop_update"] != lr_drop_after_updates(updates):
        raise t1_anchor.AnchorError("FAST final metadata has an unequal or incomplete budget.")
    if payload["prob_commit"] != t1_anchor.PINNED_PROB_COMMIT \
            or payload["seed"] != 0 \
            or tuple(payload["class_order"]) != protocol.TASK1 \
            or payload["evaluation_split_sha256"] != t1_anchor.EVALUATION_SPLIT_SHA256:
        raise t1_anchor.AnchorError("FAST final scientific identity changed.")
    recipe = payload["recipe"]
    if not isinstance(recipe, Mapping) \
            or longtail.sha256_bytes(longtail.canonical_json_bytes(recipe)) != \
            payload["recipe_fingerprint"] \
            or recipe.get("condition") != condition \
            or recipe.get("manifest_sha256") != payload["manifest_sha256"] \
            or recipe.get("initialization_sha256") != payload["initialization_sha256"] \
            or recipe.get("prob_commit") != payload["prob_commit"] \
            or recipe.get("final_optimizer_updates") != updates \
            or recipe.get("total_image_presentations") != BATCH_SIZE * updates:
        raise t1_anchor.AnchorError("FAST final recipe fingerprint or identity changed.")
    for name, length in (("owl_commit", 40), ("manifest_sha256", 64),
                         ("initialization_sha256", 64), ("recipe_fingerprint", 64),
                         ("plan_fingerprint", 64), ("checkpoint_sha256", 64),
                         ("benchmark_receipt_sha256", 64)):
        t1_anchor._lower_hex(str(payload[name]), length, name)


def _boot_identity() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text(encoding="utf-8").strip() if path.is_file() else socket.gethostname()


def training_marker_payload(condition: str) -> dict[str, object]:
    if condition not in CONDITION_ORDER:
        raise t1_anchor.AnchorError("Invalid FAST active-training marker condition.")
    return {
        "schema": "controlled_t1_anchor_fast_training_marker_v1",
        "recipe_version": FAST_RECIPE_VERSION,
        "condition": condition, "pid": os.getpid(), "boot_identity": _boot_identity(),
    }


def _training_marker_active(path: Path, condition: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return False
        if payload != training_marker_payload(condition) | {"pid": payload.get("pid")}:
            return False
        if payload["boot_identity"] != _boot_identity():
            return False
        os.kill(int(payload["pid"]), 0)
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def workspace_state(path: str | Path, condition: str) -> str:
    root = Path(path)
    validate_workspace_path(root, condition)
    if _training_marker_active(root / "TRAINING.json", condition):
        return "TRAINING"
    if (root / "FAILED.json").is_file():
        return "FAILED"
    done = root / "DONE.json"
    final = root / f"t1_fast_{condition}.pth"
    metadata = final.with_suffix(".metadata.json")
    metrics = root / "anchor_metrics.json"
    per_class = root / "per_class.csv"
    resume = root / "train" / "resume_latest.pth"
    if done.is_file():
        try:
            receipt = json.loads(done.read_text(encoding="utf-8"))
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            metric_payload = json.loads(metrics.read_text(encoding="utf-8"))
            validate_training_metadata(meta)
            expected = {
                "schema": "controlled_t1_anchor_fast_done_v1",
                "recipe_version": FAST_RECIPE_VERSION,
                "condition": condition,
                "global_step": meta["final_optimizer_updates"],
                "final_optimizer_updates": meta["final_optimizer_updates"],
                "checkpoint_sha256": longtail.sha256_file(final),
                "metrics_sha256": longtail.sha256_file(metrics),
                "per_class_csv_sha256": longtail.sha256_file(per_class),
                "recipe_fingerprint": meta["recipe_fingerprint"],
                "plan_fingerprint": meta["plan_fingerprint"],
            }
            if not all(receipt.get(key) == value for key, value in expected.items()):
                return "FAILED"
            if meta.get("global_step") != meta.get("final_optimizer_updates") \
                    or meta.get("checkpoint_sha256") != expected["checkpoint_sha256"]:
                return "FAILED"
            if metric_payload.get("schema") != "controlled_t1_anchor_fast_metrics_v1" \
                    or metric_payload.get("recipe_version") != FAST_RECIPE_VERSION \
                    or metric_payload.get("condition") != condition \
                    or metric_payload.get("checkpoint_sha256") != expected["checkpoint_sha256"] \
                    or metric_payload.get("recipe_fingerprint") != meta.get("recipe_fingerprint"):
                return "FAILED"
            return "DONE"
        except (OSError, KeyError, t1_anchor.AnchorError, json.JSONDecodeError):
            return "FAILED"
    if final.is_file() and metadata.is_file():
        try:
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            validate_training_metadata(meta)
            if meta.get("recipe_version") == FAST_RECIPE_VERSION \
                    and meta.get("global_step") == meta.get("final_optimizer_updates") \
                    and meta.get("checkpoint_sha256") == longtail.sha256_file(final):
                return "TRAINED_PENDING_EVAL"
        except (OSError, t1_anchor.AnchorError, json.JSONDecodeError):
            pass
        return "FAILED"
    if any(item.exists() for item in (final, metadata, metrics, per_class)):
        return "FAILED"
    if resume.is_file():
        return "INCOMPLETE_RESUMABLE"
    if (root / "train").exists() and any((root / "train").iterdir()):
        return "FAILED"
    return "READY"
