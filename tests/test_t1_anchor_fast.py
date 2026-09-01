"""Scientific contracts for the compute-matched controlled T1 FAST recipe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from owl import longtail, t1_anchor, t1_anchor_fast

ROOT = Path(__file__).resolve().parent.parent


def recipe(condition: str, updates: int = 12_000) -> t1_anchor_fast.FastRecipe:
    return t1_anchor_fast.FastRecipe(
        condition=condition,
        manifest_sha256=t1_anchor.EXPECTED_MANIFEST_SHA256[condition],
        owl_commit="1" * 40,
        initialization_sha256="2" * 64,
        initialization_model_state_sha256="3" * 64,
        python_version="3.13.9", torch_version="2.11.0+cu128",
        torchvision_version="0.26.0+cu128", cuda_version="12.8",
        final_optimizer_updates=updates,
    )


def benchmark(condition: str, seconds: float = 0.9) -> dict[str, object]:
    return {
        "schema": "controlled_t1_anchor_fast_benchmark_v1",
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": condition,
        "warmup_iterations": 5, "measured_iterations": 20,
        "seconds_per_optimizer_update": seconds,
        "peak_gpu_memory_bytes": 10_000,
        "gpu": "Tesla T4", "python_version": "3.13.9",
        "torch_version": "2.11.0+cu128", "torchvision_version": "0.26.0+cu128",
        "cuda_version": "12.8",
        "manifest_sha256": t1_anchor.EXPECTED_MANIFEST_SHA256[condition],
        "initialization_sha256": "2" * 64, "train_split_sha256": "4" * 64,
        "finite_gradients": True,
        "exact_fast_path": (
            "data+augmentation+forward+matcher+criterion+weighted_loss+backward+"
            "finite_gradients+optimizer_step"
        ),
    }


def test_fast_recipe_version_and_default_equal_budget_are_exact():
    recipes = [recipe(condition) for condition in t1_anchor_fast.CONDITION_ORDER]
    assert t1_anchor_fast.FAST_RECIPE_VERSION == "controlled_t1_anchor_fast_v1"
    assert all(item.final_optimizer_updates == 12_000 for item in recipes)
    assert all(item.batch_size == 2 for item in recipes)
    assert all(item.total_image_presentations == 24_000 for item in recipes)
    ignored = {"condition", "manifest_sha256"}
    common = [{key: value for key, value in item.payload().items() if key not in ignored}
              for item in recipes]
    assert common[0] == common[1] == common[2]
    assert len({item.fingerprint() for item in recipes}) == 3


def test_unique_sampling_covers_exact_full_run_and_is_deterministic():
    selected = t1_anchor_fast.ordered_indices(35_460, "lt100", 12_000)
    assert len(selected) == 24_000 == len(set(selected))
    assert selected == t1_anchor_fast.ordered_indices(35_460, "lt100", 12_000)
    assert selected != t1_anchor_fast.ordered_indices(35_808, "lt50", 12_000)
    assert t1_anchor_fast.ordered_indices(35_460, "lt100", 10_000) == selected[:20_000]


def test_resume_is_exact_next_unseen_batch_without_replay():
    selected = t1_anchor_fast.ordered_indices(37_429, "lt10", 12_000)
    batches = tuple(zip(selected[::2], selected[1::2], strict=True))
    remaining = t1_anchor_fast.remaining_batches(37_429, "lt10", 12_000, 4_321)
    assert remaining == batches[4_321:]
    consumed = set(selected[: 4_321 * 2])
    assert consumed.isdisjoint({item for batch in remaining for item in batch})


def test_resume_validates_order_and_isolates_loader_rng():
    source = (ROOT / "tools" / "run_prob_t1_anchor_fast.py").read_text(encoding="utf-8")
    assert '"selected_order_sha256": selection_sha' in source
    assert 'state.get("batch_offset") != global_step' in source
    assert "generator=loader_generator" in source


def test_unique_policy_caps_updates_at_half_smallest_dataset():
    assert t1_anchor_fast.MAX_UNIQUE_UPDATES == 17_730
    assert len(t1_anchor_fast.ordered_indices(35_460, "lt100", 17_730)) == 35_460
    with pytest.raises(t1_anchor.AnchorError, match="17730"):
        t1_anchor_fast.validate_update_count(17_731)


def test_lr_drop_uses_floor_and_lower_lr_starts_on_next_one_based_update():
    assert t1_anchor_fast.lr_drop_after_updates(12_000) == 9_073
    assert t1_anchor_fast.lr_scale_for_update(9_072, 12_000) == 1.0  # update 9,073
    assert t1_anchor_fast.lr_scale_for_update(9_073, 12_000) == 0.1  # update 9,074


def test_benchmark_receipt_is_runtime_only_and_cannot_inspect_ap():
    receipt = benchmark("lt100")
    assert t1_anchor_fast.benchmark_identity(receipt)["measured_iterations"] == 20
    with pytest.raises(t1_anchor.AnchorError, match="scientific results"):
        t1_anchor_fast.benchmark_identity(receipt | {"AP50": 1.0})


def test_runtime_plan_freezes_default_budget_when_it_fits():
    receipts = {condition: benchmark(condition, seconds)
                for condition, seconds in zip(t1_anchor_fast.CONDITION_ORDER,
                                              (0.92, 0.88, 0.9), strict=True)}
    plan = t1_anchor_fast.plan_experiment(receipts, total_budget_hours=14.0)
    assert plan["decision"] == "GO"
    assert plan["frozen_optimizer_updates_per_condition"] == 12_000
    assert plan["image_presentations_per_condition"] == 24_000
    assert plan["decision_basis"] == "runtime_only_no_AP_or_scientific_metrics"
    t1_anchor_fast.validate_plan(plan)


def test_runtime_plan_rejects_mixed_benchmark_stacks():
    receipts = {condition: benchmark(condition)
                for condition in t1_anchor_fast.CONDITION_ORDER}
    receipts["lt10"]["gpu"] = "another GPU"
    with pytest.raises(t1_anchor.AnchorError, match="one live runtime"):
        t1_anchor_fast.plan_experiment(receipts)


def test_runtime_plan_reduces_all_conditions_equally_or_fails_closed():
    receipts = {condition: benchmark(condition, 1.0)
                for condition in t1_anchor_fast.CONDITION_ORDER}
    reduced = t1_anchor_fast.plan_experiment(receipts, total_budget_hours=4.0)
    assert reduced["frozen_optimizer_updates_per_condition"] == 2_000
    assert reduced["image_presentations_per_condition"] == 4_000
    blocked = t1_anchor_fast.plan_experiment(receipts, total_budget_hours=1.5)
    assert blocked["decision"] == "NO-GO"
    assert blocked["frozen_optimizer_updates_per_condition"] is None


def test_frozen_plan_cannot_change_after_scientific_receipt_exists(tmp_path):
    receipts = {condition: benchmark(condition) for condition in t1_anchor_fast.CONDITION_ORDER}
    plan = t1_anchor_fast.plan_experiment(receipts)
    path = tmp_path / "fast_recipe_plan.json"
    t1_anchor.write_json_once_or_verify(path, plan)
    with pytest.raises(t1_anchor.AnchorError, match="immutable metadata differs"):
        t1_anchor.write_json_once_or_verify(
            path, plan | {"frozen_optimizer_updates_per_condition": 11_000})


def test_v2_and_fast_workspaces_cannot_collide(tmp_path):
    fast = t1_anchor_fast.workspace(tmp_path, "lt100")
    assert fast.name == "t1_anchor_fast__lt100__seed0"
    t1_anchor_fast.validate_workspace_path(fast, "lt100")
    with pytest.raises(t1_anchor.AnchorError, match="recipe-isolated"):
        t1_anchor_fast.validate_workspace_path(
            tmp_path / "t1_anchor__lt100__seed0", "lt100")


def test_live_training_marker_reports_training_state(tmp_path):
    workspace = t1_anchor_fast.workspace(tmp_path, "lt50")
    workspace.mkdir(parents=True)
    marker = workspace / "TRAINING.json"
    marker.write_text(
        json.dumps(t1_anchor_fast.training_marker_payload("lt50")), encoding="utf-8")
    assert t1_anchor_fast.workspace_state(workspace, "lt50") == "TRAINING"


def test_incomplete_checkpoint_is_not_done_and_final_hash_is_enforced(tmp_path):
    workspace = t1_anchor_fast.workspace(tmp_path, "lt100")
    assert t1_anchor_fast.workspace_state(workspace, "lt100") == "READY"
    resume = workspace / "train" / "resume_latest.pth"
    resume.parent.mkdir(parents=True)
    resume.write_bytes(b"resume")
    assert t1_anchor_fast.workspace_state(workspace, "lt100") == "INCOMPLETE_RESUMABLE"
    final = workspace / "t1_fast_lt100.pth"
    final.write_bytes(b"final")
    item = recipe("lt100")
    metadata = {
        "schema": t1_anchor_fast.FAST_SCHEMA,
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": "lt100",
        "manifest_sha256": t1_anchor.EXPECTED_MANIFEST_SHA256["lt100"],
        "owl_commit": "1" * 40,
        "prob_commit": t1_anchor.PINNED_PROB_COMMIT,
        "initialization_sha256": "2" * 64,
        "recipe_fingerprint": item.fingerprint(),
        "recipe": item.payload(),
        "plan_fingerprint": "5" * 64,
        "seed": 0,
        "global_step": 12_000, "final_optimizer_updates": 12_000,
        "image_presentations": 24_000, "lr_drop_update": 9_073,
        "checkpoint_sha256": longtail.sha256_file(final),
        "class_order": list(t1_anchor.protocol.TASK1),
        "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
        "benchmark_receipt_sha256": "6" * 64,
    }
    final.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert t1_anchor_fast.workspace_state(workspace, "lt100") == "TRAINED_PENDING_EVAL"
    final.write_bytes(b"changed")
    assert t1_anchor_fast.workspace_state(workspace, "lt100") == "FAILED"


def test_done_requires_hash_linked_fast_evaluation_schema(tmp_path):
    condition = "lt10"
    workspace = t1_anchor_fast.workspace(tmp_path, condition)
    workspace.mkdir(parents=True)
    final = workspace / f"t1_fast_{condition}.pth"
    final.write_bytes(b"final")
    item = recipe(condition)
    metadata = {
        "schema": t1_anchor_fast.FAST_SCHEMA,
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": condition, "manifest_sha256": item.manifest_sha256,
        "owl_commit": item.owl_commit, "prob_commit": item.prob_commit,
        "initialization_sha256": item.initialization_sha256,
        "recipe_fingerprint": item.fingerprint(), "recipe": item.payload(),
        "plan_fingerprint": "5" * 64,
        "seed": 0, "global_step": 12_000, "final_optimizer_updates": 12_000,
        "image_presentations": 24_000, "lr_drop_update": 9_073,
        "checkpoint_sha256": longtail.sha256_file(final),
        "class_order": list(t1_anchor.protocol.TASK1),
        "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
        "benchmark_receipt_sha256": "6" * 64,
    }
    final.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    metrics = {
        "schema": "controlled_t1_anchor_fast_metrics_v1",
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": condition, "checkpoint_sha256": metadata["checkpoint_sha256"],
        "recipe_fingerprint": metadata["recipe_fingerprint"],
    }
    metrics_path = workspace / "anchor_metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    per_class = workspace / "per_class.csv"
    per_class.write_text("condition,class,AP50\n", encoding="utf-8")
    done = {
        "schema": "controlled_t1_anchor_fast_done_v1",
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": condition, "global_step": 12_000,
        "final_optimizer_updates": 12_000,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "metrics_sha256": longtail.sha256_file(metrics_path),
        "per_class_csv_sha256": longtail.sha256_file(per_class),
        "recipe_fingerprint": metadata["recipe_fingerprint"],
        "plan_fingerprint": metadata["plan_fingerprint"],
    }
    (workspace / "DONE.json").write_text(json.dumps(done), encoding="utf-8")
    assert t1_anchor_fast.workspace_state(workspace, condition) == "DONE"
    metrics_path.write_text("{}", encoding="utf-8")
    assert t1_anchor_fast.workspace_state(workspace, condition) == "FAILED"


def test_comparison_requires_all_three_done(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "compare_t1_anchors_fast", ROOT / "tools" / "compare_t1_anchors_fast.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with pytest.raises(t1_anchor.AnchorError, match="all to be DONE"):
        module.compare(tmp_path, tmp_path / "controlled_lt_fast_v1_comparison")


def test_complete_comparison_writes_exact_tables_and_figures(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "compare_t1_anchors_fast_complete", ROOT / "tools" / "compare_t1_anchors_fast.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(t1_anchor_fast, "workspace_state", lambda *_args: "DONE")
    for condition_index, condition in enumerate(("lt10", "lt50", "lt100")):
        workspace = t1_anchor_fast.workspace(tmp_path, condition)
        workspace.mkdir(parents=True)
        checkpoint = workspace / f"t1_fast_{condition}.pth"
        checkpoint.write_bytes(condition.encode())
        metadata = {
            "plan_fingerprint": "7" * 64, "final_optimizer_updates": 12_000,
            "checkpoint_sha256": longtail.sha256_file(checkpoint),
            "recipe_fingerprint": str(condition_index + 1) * 64,
        }
        checkpoint.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        rows = [{
            "class_name": name, "rank": index + 1,
            "group": "head" if index < 7 else "medium" if index < 13 else "tail",
            "train_count": index + 2,
            "anchor_AP50": float(index + condition_index),
        } for index, name in enumerate(t1_anchor.protocol.TASK1)]
        metrics = {
            "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
            "condition": condition,
            "overall_mAP50": 9.0 + condition_index,
            "group_mAP50": {"head": 7.0, "medium": 9.0, "tail": 11.0},
            "learnability_descriptives": {"spearman_AP50_log_train_frequency": 1.0},
            "classes": rows,
        }
        (workspace / "anchor_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    output = tmp_path / "controlled_lt_fast_v1_comparison"
    payload = module.compare(tmp_path, output)
    assert payload["one_seed_descriptive_only"] is True
    expected = {
        "summary.json", "table_anchor_metrics.csv", "table_per_class.csv",
        "table_delta_lt50_vs_lt10.csv", "table_delta_lt100_vs_lt10.csv",
        "table_delta_lt100_vs_lt50.csv", "overall_ap50_by_condition.png",
        "group_ap50_by_condition.png", "per_class_ap50_vs_log_frequency.png",
        "ap50_change_vs_controlled_rank.png",
    }
    assert {path.name for path in output.iterdir()} == expected


def test_fast_and_full_v2_versions_remain_distinct():
    assert t1_anchor.RECIPE_VERSION == "controlled_t1_anchor_v2"
    assert t1_anchor_fast.FAST_RECIPE_VERSION != t1_anchor.RECIPE_VERSION
    assert t1_anchor.PINNED_PROB_COMMIT == \
        "4c66be1a52cad9360e09c729e9134aba8fe0b531"
