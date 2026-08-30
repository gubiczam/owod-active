"""Contract tests for the production one-click Colab notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "owod_active.ipynb"
OWL_SHA = "ae2d2ab1bdeb7a9c30992448d0a839c3458451e9"
PROB_SHA = "4c66be1a52cad9360e09c729e9134aba8fe0b531"


def payload() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells() -> list[str]:
    return ["".join(item["source"]) for item in payload()["cells"]
            if item["cell_type"] == "code"]


def cell(stage: int) -> str:
    matches = [source for source in code_cells() if source.startswith(f"# {stage} —")]
    assert len(matches) == 1, (stage, len(matches))
    return matches[0]


def notebook_function(stage: int, name: str):
    tree = ast.parse(cell(stage))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert any(node.name == name for node in functions), name
    namespace: dict = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 f"stage {stage}:{name}", "exec"), namespace)  # noqa: S102
    return namespace[name]


def test_notebook_is_valid_compilable_json_with_exactly_stages_0_through_12():
    document = payload()
    assert document["nbformat"] == 4
    assert document["metadata"]["accelerator"] == "GPU"
    assert len(code_cells()) == 13
    for stage in range(13):
        compile(cell(stage), f"stage {stage}", "exec")


def test_production_parameters_are_exact_and_have_no_smoke_or_optional_mode():
    namespace: dict = {}
    exec(compile(cell(0), "parameters", "exec"), namespace)  # noqa: S102
    expected = {
        "RUN_GPU": True, "FAST_CHAIN": True, "SELECTION_ARM": "random",
        "LABELLING_POLICY": "known_plus_selected", "N_TASKS": 6,
        "BUDGET_PER_TASK": 600, "ROUNDS_PER_TASK": 6,
        "CANDIDATE_IMAGES": 2000, "PROPOSALS_PER_IMAGE": 50,
        "REPLAY_REALLOCATE": False, "EPOCHS": 5,
        "LEARNING_RATE": 2e-4, "BATCH_SIZE": 2, "N_CLUSTERS": 1600,
        "SEED": 0, "REPLAY_ARMS": ("uniform", "tail_favouring"),
        "TIME_BUDGET_MINUTES": 420, "SESSION_CEILING_MINUTES": 840,
    }
    for name, value in expected.items():
        assert namespace[name] == value, name
    assert namespace["EXPERIMENT_AUDIT"]["workspaces"] == (
        "random__none", "random__uniform", "random__tail_favouring")
    assert "SMOKE_TEST" not in namespace and "MINIMAL_CHAIN" not in namespace


def test_reviewed_repositories_are_immutable_and_origin_checked():
    parameters, checkout, prob = cell(0), cell(2), cell(3)
    assert f'OWL_COMMIT = "{OWL_SHA}"' in parameters
    assert f'PROB_COMMIT = "{PROB_SHA}"' in parameters
    for command in ("clone\", \"--filter=blob:none\", \"--no-checkout",
                    "remote\", \"get-url\", \"origin", "fetch\", \"--depth\", \"1",
                    "reset\", \"--hard", "rev-parse\", \"HEAD"):
        assert command in checkout
    assert "PROB_COMMIT" in prob and "PROB_SHA == PROB_COMMIT" in prob
    assert "PROB_BRANCH" not in "\n".join(code_cells())


def test_drive_mount_precedes_clones_and_is_the_only_interactive_operation():
    joined = "\n".join(code_cells())
    assert joined.index('drive.mount("/content/drive"') < joined.index("ensure_pinned_checkout")
    assert 'DRIVE_ROOT = "/content/drive/MyDrive/OWL"' in cell(0)
    assert ".owod_write_probe" in cell(1)
    assert "input(" not in joined and "getpass(" not in joined


def test_dependencies_are_installed_without_floating_upgrades_or_runtime_restart():
    joined = "\n".join((cell(2), cell(3)))
    assert "pip\", \"install" in joined and 'f"{ROOT}[plots]"' in joined
    assert 'str(PROB / "requirements.txt")' not in joined
    for requirement in ("einops==0.5.0", "Cython==3.1.3", "pycocotools==2.0.5",
                        "wandb==0.18.7", "pandas==2.3.2", "seaborn==0.13.2",
                        "tqdm==4.67.1", "jedi==0.19.2"):
        assert requirement in joined
    assert "--no-build-isolation" in joined and "--no-deps" in joined
    assert "--only-binary=:all:" in joined
    assert "runtime_probe" in joined and "import main_open_world" in joined
    assert "sys.version_info[:2] == (3, 13)" in joined
    assert "CocoEvaluator" in joined and "NPY_OWNDATA" in joined
    assert "evaluator.coco_eval" in joined and "stats[0]" in joined
    assert "PROB CUDA model/loss/evaluator smoke" in joined
    assert "weighted.backward()" in joined and "MSDA_AVAILABLE" in joined
    assert "OWDetection" in joined and "voc_eval" in joined
    assert "scikit-image==0.19.2" not in joined and "pandas==1.5.1" not in joined
    assert "/usr/bin/python" not in joined and "/usr/local/bin/python" not in joined
    assert "--upgrade" not in joined and "restart" not in joined.lower()
    assert "MultiScaleDeformableAttention" in joined and "probe_prob_msda" in joined
    assert "requires jedi, which is not installed" in joined
    assert "bootstrap_package_probe = pip_check()" in joined
    assert "if bootstrap_package_probe.returncode != 0:" in joined
    assert "Python package consistency check failed after bootstrap repair" in joined


def test_msda_backend_is_runtime_authoritative_and_cuda_smoke_gated():
    prob, preflight, final = cell(3), cell(5), cell(12)

    # Regression for the real T4 failure: the no-torch bare import remains only
    # diagnostic and is never equated with the wrapper's dispatch decision.
    assert 'RAW_MSDA = run_json_probe(' in prob
    assert 'from models.ops.functions import ms_deform_attn_func as msda_func' in prob
    assert 'from models.ops.modules import ms_deform_attn as msda_module' in prob
    assert 'PREBUILD_PROB_MSDA.get("available") is not True' in prob
    assert 'RAW_MSDA.get("ok") != POSTBUILD_PROB_MSDA.get("available")' in prob
    assert 'assert MSDA_AVAILABLE ==' not in prob
    assert 'sys.argv[2]' not in prob

    # Both compiled and fallback branches must be observed inside the actual
    # detector forward and survive criterion, backward, and postprocessing.
    assert 'dispatch_counts = {"compiled": 0, "fallback": 0}' in prob
    assert 'ObservedCompiledDispatch' in prob and 'observed_fallback' in prob
    assert 'assert dispatch_counts[' in prob
    assert 'weighted.backward()' in prob and 'postprocessors["bbox"]' in prob
    assert 'torch.cuda.synchronize()' in prob
    assert 'compatible_keys' in prob and 'T1 checkpoint has no compatible' in prob
    assert 'if prob_smoke.returncode != 0:' in prob
    assert 'ENVIRONMENT_PREFLIGHT_OK = True' in prob
    assert prob.index('if prob_smoke.returncode != 0:') < prob.index(
        'ENVIRONMENT_PREFLIGHT_OK = True')

    # A failed/missing smoke result cannot reach anchor evaluation or training.
    assert '"CUDA model smoke": ENVIRONMENT_PREFLIGHT_OK' in preflight
    assert 'PROB_MSDA_BACKEND' in final and 'MSDA_BUILT' not in final


def test_canonical_data_and_shared_split_are_package_owned():
    data = cell(4)
    assert 'DATA = Path("/content/data/OWOD")' in data
    for name in ("owdetr_pool_annotations.tar.gz", "owdetr_test_annotations.tar.gz",
                 "owdetr_replay_annotations.tar.gz", "per_image_class_counts.json",
                 "t1_replay_class_counts.json"):
        assert name in data
    assert "evaluation_subset.SHARED_TEST_SET" in data
    assert '"owl_shared_test"' not in data
    assert "extract_committed_archive" in data and "destination" in data


def test_preflight_is_fail_closed_and_covers_every_required_input():
    preflight = cell(5)
    for label in ("CUDA model smoke", "GPU visible", "GPU memory >= 14 GiB", "torch CUDA",
                  "package consistency", "OWL exact SHA", "PROB exact SHA",
                  "Drive writable", "Drive free >= 8 GiB", "local free >= 12 GiB",
                  "checkpoint", "canonical data root",
                  "completed baseline exists", "baseline fingerprint",
                  "target fingerprints", "target artefact integrity",
                  "Replay Protocol V3", "workspace isolation",
                  "PROB bridge flags"):
        assert label in preflight
    assert "workspace_problem" in preflight and "results tasks" in preflight
    assert "PREFLIGHT FAILED" in preflight
    assert "No GPU evaluation or training was started" in preflight
    assert "PREFLIGHT_OK = True" in preflight


def test_legacy_no_replay_baseline_exception_is_exactly_one_field_narrow():
    decide = notebook_function(5, "legacy_no_replay_baseline_decision")
    expected = {
        "arm": "random", "replay_arm": "none", "seed": 0,
        "replay_protocol_version": 3,
    }
    legacy = {name: value for name, value in expected.items()
              if name != "replay_protocol_version"}

    differences, note = decide(
        workspace_name="random__none", stored=legacy, expected=expected,
        completed=True, integrity_ok=True, no_replay_artifacts=True)
    assert differences == {}
    assert note == {
        "workspace": "random__none",
        "stored": "absent",
        "normalized_for_compatibility": "no-replay-only",
        "reason": "replay_arm=none; replay protocol inactive",
        "historical_config_modified": False,
    }

    rejected = []
    for workspace, replay_arm in (
        ("random__uniform", "uniform"),
        ("random__tail_favouring", "tail_favouring"),
        ("random__none", "uniform"),
    ):
        candidate_expected = dict(expected, replay_arm=replay_arm)
        candidate_stored = dict(candidate_expected)
        candidate_stored.pop("replay_protocol_version")
        rejected.append(decide(
            workspace_name=workspace, stored=candidate_stored,
            expected=candidate_expected, completed=True, integrity_ok=True,
            no_replay_artifacts=True))

    other_mismatch = dict(legacy, seed=7)
    rejected.append(decide(
        workspace_name="random__none", stored=other_mismatch, expected=expected,
        completed=True, integrity_ok=True, no_replay_artifacts=True))
    rejected.append(decide(
        workspace_name="random__none", stored=legacy, expected=expected,
        completed=False, integrity_ok=True, no_replay_artifacts=True))
    rejected.append(decide(
        workspace_name="random__none", stored=legacy, expected=expected,
        completed=True, integrity_ok=False, no_replay_artifacts=True))
    rejected.append(decide(
        workspace_name="random__none", stored=legacy, expected=expected,
        completed=True, integrity_ok=True, no_replay_artifacts=False))
    rejected.append(decide(
        workspace_name="random__none", stored=dict(legacy, replay_protocol_version=2),
        expected=expected, completed=True, integrity_ok=True,
        no_replay_artifacts=True))
    unrelated_missing = dict(legacy)
    unrelated_missing.pop("seed")
    rejected.append(decide(
        workspace_name="random__none", stored=unrelated_missing, expected=expected,
        completed=True, integrity_ok=True, no_replay_artifacts=True))

    assert all(differences and note is None for differences, note in rejected)


def test_legacy_projection_never_rewrites_drive_and_comparisons_record_provenance():
    preflight, anchor, final_compare = cell(5), cell(6), cell(11)
    assert "BASELINE_CONFIG_BYTES" in preflight
    assert "Historical random__none config.json was modified" in preflight
    assert "destination.symlink_to" in preflight
    assert "COMPARISON_WORKSPACE" in preflight
    assert "load_compatible_run(BASELINE)" in anchor
    assert "build_comparison_workspace()" in anchor
    assert "legacy_baseline_replay_protocol" in final_compare
    assert "assert_historical_baseline_unchanged()" in final_compare


def test_baseline_is_exact_and_anchor_validation_precedes_any_replay_training():
    anchor = cell(6)
    assert "EXPECTED_TASKS" in anchor and "require_anchor=False" in anchor
    assert '[*anchor_command, "--dry-run"]' in anchor
    assert "81-class anchor" in anchor
    assert "per_task_recall" in anchor and "per_class_checks" in anchor
    assert "forgetting" in anchor
    assert "allow_partial=True" in anchor and "wanted_tasks" in anchor
    assert "comparison.compatibility" in anchor
    assert "compare_replay.py" in anchor and "--no-plots" in anchor
    joined = "\n".join(code_cells())
    assert joined.index('[*anchor_command, "--dry-run"]') < joined.index("runner.run_chain(")


def test_uniform_runs_and_validates_before_tail_is_scheduled():
    sources = code_cells()
    uniform_run = next(i for i, source in enumerate(sources)
                       if 'run_replay_arm("uniform")' in source)
    uniform_audit = next(i for i, source in enumerate(sources)
                         if 'validate_replay_workspace("uniform")' in source)
    tail_run = next(i for i, source in enumerate(sources)
                    if 'run_replay_arm("tail_favouring")' in source)
    tail_audit = next(i for i, source in enumerate(sources)
                      if 'validate_replay_workspace("tail_favouring")' in source)
    assert uniform_run < uniform_audit < tail_run < tail_audit


def test_each_replay_run_is_isolated_resumable_and_has_its_own_420_minute_cap():
    parameters, preflight, run = cell(0), cell(5), cell(7)
    assert 'PLANNED_RUNS = ("random__uniform", "random__tail_favouring")' in parameters
    assert 'BASELINE_NAME = "random__none"' in parameters
    assert 'WORK / f"random__{arm}"' in preflight
    assert "fingerprint_differences" in preflight
    assert "time_budget_minutes=per_run_budget" in run
    assert "min(TIME_BUDGET_MINUTES, SESSION_CEILING_MINUTES - gpu_minutes_before)" in run
    assert "SESSION_CEILING_MINUTES" in run
    assert "resumed and completed" in run and "validated and skipped" in run
    assert "workspace=target" in run
    assert 'workspace=WORK / "random"' not in "\n".join(code_cells())


def test_replay_v3_audit_checks_objects_aliases_lifecycle_and_resume_identity():
    audit = cell(8)
    for fragment in (
        '== diagnostics["allocated_objects"]', '== diagnostics["delivered_objects"]',
        "budget == 400", "len(current) == budget", "sum(per_class.values()) == budget",
        "exemplars.alias_id", "exemplars.source_id", "canonical old-data pool",
        "E_(k-1) union L_(k-1)", "identity_sha256", "replay_v3_audit.json",
        "per_class_validated", "recall_crosschecks",
    ):
        assert fragment in audit


def test_final_comparison_is_strict_complete_and_persisted_only_after_validation():
    final_compare = cell(11)
    for stem in ("table1_task_comparison", "table2_delta_vs_baseline",
                 "table3_tail_vs_uniform", "table4_per_class",
                 "table5_replay_composition", "table6_cost"):
        assert stem in final_compare
    for format_name in (".csv", ".md", ".tex", '"png"', '"pdf"'):
        assert format_name in final_compare
    assert 'DRIVE / "comparisons" / COMPARISON_NAME' in final_compare
    assert "compatibility_clashes" in final_compare
    assert final_compare.index("missing = sorted") < final_compare.index("shutil.copytree")


def test_success_marker_can_only_be_reached_after_all_validations():
    final = cell(12)
    assert "EXPERIMENT_COMPLETE = True" in final
    assert final.rstrip().endswith('print("EXPERIMENT COMPLETE")')
    assert sum("EXPERIMENT COMPLETE" in source for source in code_cells()) == 1
    assert all("EXPERIMENT COMPLETE" not in cell(stage) for stage in range(12))


@pytest.mark.slow
def test_whole_notebook_runs_with_colab_gpu_network_prob_and_baseline_faked(capsys):
    import sys

    tools = str(ROOT / "tools")
    sys.path.insert(0, tools)
    try:
        import dry_run_notebook
        dry_run_notebook.run(run_gpu=True, verbose=False)
    finally:
        sys.path.remove(tools)
    from owl import bridge as live_bridge

    assert live_bridge.Bridge.__name__ == "Bridge"
    output = capsys.readouterr().out
    assert "GPU branch: pinned baseline + 2 replay runs" in output
    assert "all notebook audits" in output
