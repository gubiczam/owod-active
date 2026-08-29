"""The replay comparison, exercised on workspaces the chain really wrote.

The point of these is that tomorrow's analysis must not be written tomorrow.
Every table is produced today against workspaces built by ``runner.run_chain``
itself, including the two situations that will actually occur: a run that has
not started, and a run that stopped part-way through the chain.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_run_chain import OLD_DATA, FakeBridge, prob_data_root

from owl import comparison, protocol, runner

ROOT = Path(__file__).resolve().parent.parent


def build_run(root: Path, replay_arm: str, *, n_tasks: int = 4,
              time_budget: float | None = None, seed: int = 0) -> Path:
    """One real chain against the fake bridge, left on disk like the GPU leaves it."""

    index = {f"{i:012d}": {"traffic light": 1, protocol.TASK1[i % 19]: 1}
             for i in range(400)}
    workspace = root / f"random__{replay_arm}"
    config = runner.CycleConfig(
        n_tasks=n_tasks, budget_per_task=20, rounds_per_task=2,
        candidate_images_per_task=40, proposals_per_image=4, n_clusters=8,
        arm="random", replay_arm=replay_arm, keep_checkpoints=0, seed=seed,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        runner.run_chain(
            FakeBridge(), config, workspace=workspace, candidate_index=index,
            replay_index=OLD_DATA, replay_root=prob_data_root(root, index, OLD_DATA),
            start_checkpoint=root / "t1.pth", test_set="owl_shared_test",
            chain=protocol.build_chain(n_tasks), prepare_images=lambda ids: ids,
            time_budget_minutes=time_budget,
        )
    return workspace


@pytest.fixture(scope="module")
def three_runs(tmp_path_factory):
    """The full comparison: baseline plus both replay arms."""

    root = tmp_path_factory.mktemp("work")
    for arm in ("none", "uniform", "tail_favouring"):
        build_run(root, arm)
    return root


# ------------------------------------------------------------------ reading ---


def test_the_runs_are_found_and_ordered_baseline_first(three_runs):
    runs = comparison.load_runs(three_runs)

    assert list(runs) == list(comparison.EXPECTED), "the baseline must come first"
    for name, run in runs.items():
        assert run.replay_arm == name.split("__", 1)[1]
        assert run.selection_arm == "random"
        assert len(run) == 3, f"{name} did not finish its chain"


def test_per_class_ap_is_read_from_the_metrics_files_not_the_csv(three_runs):
    """The CSV has no per-class column; the group aggregates cannot substitute."""

    runs = comparison.load_runs(three_runs)
    run = runs["random__none"]

    assert run.anchor_ap, "the anchor's per-class AP was not picked up"
    assert set(run.per_task_ap) == set(run.tasks)
    for name in ("person", "bear", "car"):
        assert name in run.anchor_ap
        assert name in run.per_task_ap[run.final_task]


def test_the_per_class_vector_is_validated_against_the_file_that_carries_it(three_runs):
    """The metrics file has no key named for a per-class table.

    What it has is ``coco_eval_bbox``, and nothing in the file says those 83
    entries are ``[mAP, mAP, 80 classes, unknown]``. So every task's vector is
    checked by rebuilding the aggregates the same file reports; a layout change
    has to surface as a refusal, not as a plausible table of the wrong classes.
    """

    runs = comparison.load_runs(three_runs)
    for name, run in runs.items():
        assert run.per_class_checks, f"{name} recorded no provenance"
        assert run.per_class_ap_is_validated, f"{name} failed its own cross-check"
        for task, report in run.per_class_checks.items():
            assert report["usable"], (name, task, report.get("reason"))
            assert report["checks"], "nothing was actually compared"
            for check in report["checks"]:
                assert check["agrees"], (name, task, check)


def test_a_vector_that_does_not_reproduce_its_own_aggregates_is_refused(tmp_path):
    """The failure this guards: silently reporting the wrong classes."""

    from owl import metrics

    workspace = build_run(tmp_path, "none")
    task_dir = next(p for p in workspace.iterdir() if p.name.endswith("_random"))
    payload = json.loads((task_dir / "metrics.json").read_text())

    # a vector of the right length whose contents no longer match the aggregates
    payload["coco_eval_bbox"] = [30.0, 30.0, *[7.0] * 80, 0.5]
    report = metrics.validate_per_class_ap50(payload)
    assert report["usable"] is False
    assert "does not reproduce" in report["reason"]

    # and a vector of the wrong length is refused before it is read at all
    payload["coco_eval_bbox"] = [1.0, 2.0, 3.0]
    report = metrics.validate_per_class_ap50(payload)
    assert report["usable"] is False
    assert "entries" in report["reason"]


def test_the_recall_crosscheck_comes_from_the_detections_artefact(three_runs):
    """An independent per-class signal, and it must never be called AP."""

    runs = comparison.load_runs(three_runs)
    run = runs["random__none"]
    assert run.per_task_recall, "no detections artefact was read"

    measured = run.per_task_recall[run.final_task]
    for name, entry in measured.items():
        assert name in protocol.TASK1
        assert entry["objects"] >= 0
        if entry["objects"]:
            assert 0.0 <= entry["recall"] <= 100.0

    rows = {row["class"]: row for row in comparison.table_per_class(runs)}
    assert "random__none:final_recall50" in rows["person"]
    assert "random__none:test_objects" in rows["person"]


def test_the_detections_artefact_is_written_for_every_replay_arm(three_runs):
    """Requirement: the same artefact for none, uniform and tail_favouring."""

    runs = comparison.load_runs(three_runs)
    for name, run in runs.items():
        assert set(run.per_task_recall) == set(run.tasks), (
            f"{name} is missing a detections artefact for some task")


def test_a_run_that_has_not_started_is_absent_rather_than_fatal(tmp_path):
    build_run(tmp_path, "none")
    (tmp_path / "random__uniform").mkdir()          # created but never run

    runs = comparison.load_runs(tmp_path)

    assert list(runs) == ["random__none"]
    assert comparison.table_delta_versus_baseline(runs) == []
    assert comparison.table_tail_versus_uniform(runs) == []
    # the tables that only need the baseline still come out
    assert comparison.table_task_comparison(runs)
    assert comparison.table_per_class(runs)
    assert comparison.table_cost(runs)


def test_a_partial_run_is_compared_as_far_as_it_got(tmp_path):
    """The overnight case: one arm finished, the next was cut off."""

    build_run(tmp_path, "none")
    build_run(tmp_path, "uniform", time_budget=1)   # stops after one task

    runs = comparison.load_runs(tmp_path)
    assert len(runs["random__none"]) == 3
    assert len(runs["random__uniform"]) == 1

    rows = comparison.table_task_comparison(runs)
    assert len(rows) == 3, "every task any run reached must appear"
    later = rows[-1]
    assert later["random__none:known_mAP50"] is not None
    assert later["random__uniform:known_mAP50"] is None, "a gap must stay a gap"


def test_an_empty_root_says_what_to_download(tmp_path):
    with pytest.raises(comparison.AnalysisError, match="No finished run"):
        comparison.load_runs(tmp_path)


# ------------------------------------------------------------ compatibility ---


def test_runs_from_different_protocols_are_reported_not_silently_compared(tmp_path):
    """A two-task sanity run and a five-task chain are not the same experiment."""

    build_run(tmp_path, "none")
    build_run(tmp_path, "uniform", n_tasks=3)       # a shorter chain

    runs = comparison.load_runs(tmp_path)
    clashes = comparison.compatibility(runs, reference="random__none")

    assert clashes, "a different chain length must be flagged"
    assert any(c["field"] == "n_tasks" for c in clashes)


def test_only_the_replay_arm_may_differ(three_runs):
    runs = comparison.load_runs(three_runs)
    assert comparison.compatibility(runs, reference="random__none") == []
    assert "replay_arm" not in comparison.MUST_MATCH, (
        "the variable under study cannot be required to match")


# ------------------------------------------------------------------- tables ---


def test_the_deltas_are_measured_against_the_named_baseline(three_runs):
    runs = comparison.load_runs(three_runs)
    table1 = {(r["task"]): r for r in comparison.table_task_comparison(runs)}
    table2 = comparison.table_delta_versus_baseline(runs)

    assert table2, "there are two replay runs to compare"
    for row in table2:
        source = table1[row["task"]]
        expected = (source[f"{row['run']}:known_mAP50"]
                    - source["random__none:known_mAP50"])
        assert row["delta_known_mAP50"] == pytest.approx(expected)


def test_the_per_class_table_carries_frequency_and_group(three_runs):
    runs = comparison.load_runs(three_runs)
    rows = comparison.table_per_class(runs)

    assert len(rows) == len(protocol.TASK1)
    by_class = {row["class"]: row for row in rows}
    assert by_class["person"]["group"] == "head"
    assert by_class["bear"]["group"] == "tail"
    assert by_class["person"]["train_objects"] > by_class["bear"]["train_objects"]
    for row in rows:
        assert row["anchor_AP50"] is not None
        for name in runs:
            assert f"{name}:final_AP50" in row
            assert f"{name}:forgetting" in row


def test_relative_forgetting_is_withheld_where_it_would_be_noise(three_runs):
    """Dividing a drop by an anchor of 0.3 is a number, not a measurement."""

    runs = comparison.load_runs(three_runs)
    rows = comparison.table_per_class(runs)
    for row in rows:
        if row["anchor_AP50"] is not None and row["anchor_AP50"] < 1.0:
            assert row["random__none:relative_forgetting"] is None


def test_the_replay_composition_table_reports_the_budget_and_its_shape(three_runs):
    runs = comparison.load_runs(three_runs)
    rows = comparison.table_replay_composition(runs)

    budgets = {"none": 0, "uniform": 400, "tail_favouring": 400}
    for row in rows:
        expected = budgets[row["replay_arm"]]
        assert row["requested"] == row["allocated"] == row["delivered"] == expected
        if expected:
            assert row["head_objects"] + row["medium_objects"] + row["tail_objects"] \
                == expected
            assert row["replay_images"] == row["unique_source_images"]


def test_the_composition_table_shows_tail_favouring_actually_favours_rarity(three_runs):
    """The claim the whole arm exists to make, checked on the real allocator."""

    runs = comparison.load_runs(three_runs)
    rows = {(r["run"], r["task"]): r for r in comparison.table_replay_composition(runs)}
    first = min(r["task"] for r in comparison.table_replay_composition(runs))

    uniform = rows[("random__uniform", first)]
    tail = rows[("random__tail_favouring", first)]
    assert uniform["rho_quota_frequency"] is not None
    assert tail["rho_quota_frequency"] < uniform["rho_quota_frequency"], (
        "tail-favouring must correlate more negatively with frequency than uniform")
    assert tail["quota_max"] > uniform["quota_max"], "the rarest class must gain"


def test_the_cost_table_keeps_supervision_beside_retention(three_runs):
    runs = comparison.load_runs(three_runs)
    rows = comparison.table_cost(runs)
    for row in rows:
        for column in ("asked", "images_opened", "images_trainable",
                       "images_no_supervision", "replay_objects", "replay_images"):
            assert column in row


# ---------------------------------------------------------- vulnerability ---


def test_vulnerability_reports_direction_without_claiming_significance(three_runs):
    runs = comparison.load_runs(three_runs)
    report = comparison.vulnerability(runs["random__none"])

    assert report["available"] is True
    assert report["n_classes"] == len(protocol.TASK1)
    assert set(report["group_means"]) == set(comparison.GROUPS)
    assert "p_value" not in json.dumps(report), "one seed cannot support a p-value"
    for key in ("ols_frequency_only", "ols_anchor_only", "ols_frequency_and_anchor"):
        assert report[key]["n"] == len(protocol.TASK1)


def test_vulnerability_says_so_when_the_data_is_not_there(tmp_path):
    workspace = build_run(tmp_path, "none")
    (workspace / "anchor_metrics.json").unlink()

    run = comparison.load_run(workspace)
    report = comparison.vulnerability(run)
    assert report["available"] is False
    assert "anchor" in report["reason"]


# ------------------------------------------------------------ the whole tool ---


def test_the_command_line_tool_runs_on_a_real_workspace(three_runs, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "compare_replay.py"), str(three_runs),
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True, check=False,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    out = tmp_path / "out"
    for stem in ("depth", "table1_task_comparison", "table2_delta_vs_baseline",
                 "table3_tail_vs_uniform", "table4_per_class",
                 "table5_replay_composition", "table6_cost"):
        for suffix in ("csv", "md", "tex"):
            assert (out / f"{stem}.{suffix}").exists(), f"{stem}.{suffix} missing"
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["runs"]) == set(comparison.EXPECTED)
    assert summary["compatibility_clashes"] == []
    assert "no p-values" in result.stdout or "No p-values" in result.stdout \
        or "no significance" in result.stdout.lower()


def test_the_tool_works_today_with_only_the_baseline(tmp_path):
    """It has to be useful before tonight's runs exist, not only after."""

    root = tmp_path / "work"
    root.mkdir()
    build_run(root, "none")

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "compare_replay.py"), str(root),
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True, check=False,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not present yet" in result.stdout
    assert "random__uniform" in result.stdout
    assert (tmp_path / "out" / "table4_per_class.csv").exists()
