"""The result chapter, built from what the chain actually wrote.

The repository's standing lesson is that testing the parts is not testing the
thing: five bugs reached a live GPU session because the pieces were each correct.
So the first test here does not hand the analyser a CSV of its own invention —
it runs the real chain against the fake bridge, lets ``run_chain`` write its
``results_<arm>.csv``, and analyses *that*. If a column is ever renamed, this
fails rather than the thesis.

The rest are the edge cases that decide whether a number in the thesis is true:
a cost axis that assumes a budget was fully spent, a missing recall silently
read as zero, and a long arm compared against a short one.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_run_chain import FakeBridge

from owl import analysis, protocol, runner

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def index():
    """A candidate index: 400 images, each holding one of the declared classes."""

    declared = [task.new_class for task in protocol.build_chain(4)[1:]]
    return {
        f"img{i:04d}": {declared[i % len(declared)]: 1 + (i % 3),
                        protocol.TASK1[i % len(protocol.TASK1)]: 1}
        for i in range(400)
    }


@pytest.fixture
def workspace(tmp_path, index):
    """Two arms of the real chain, run against the fake bridge, left on disk."""

    config = runner.CycleConfig(
        n_tasks=4, budget_per_task=20, rounds_per_task=2,
        candidate_images_per_task=40, proposals_per_image=4,
        n_clusters=8, replay_arm="tail_favouring",
    )
    root = tmp_path / "work"
    for arm in ("random", "prior_consult_batch"):
        runner.run_chain(
            FakeBridge(), replace(config, arm=arm), workspace=root / arm,
            candidate_index=index, start_checkpoint=tmp_path / "t1.pth",
            test_set="owl_shared_test", chain=protocol.build_chain(4),
            prepare_images=lambda ids: ids,
        )
    return root


def write_arm(root: Path, arm: str, rows: list[dict]) -> Path:
    """A hand-made ``results_<arm>.csv``, in the layout the chain uses."""

    path = root / arm / f"results_{arm}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


# --------------------------------------------------------------- end to end ---


def test_the_analyser_reads_what_the_chain_actually_wrote(workspace):
    """No invented schema: the CSVs under test come from ``run_chain`` itself."""

    arms = analysis.load_arms(workspace)
    assert sorted(arms) == ["prior_consult_batch", "random"]

    for name, arm in arms.items():
        assert len(arm) == 3, f"{name} did not finish the three-task chain"
        rows = analysis.per_arm_table(arm, analysis.HEADLINE)
        assert [row["task"] for row in rows] == ["t2", "t3", "t4"]
        assert analysis.HEADLINE in rows[0], "the plan's endpoint is missing from the table"
        assert rows[0]["oracle_cost"] > 0

        points = analysis.curve(arm)
        assert points and all(cost > 0 for cost, _ in points)
        assert [c for c, _ in points] == sorted(c for c, _ in points)

    comparison = analysis.comparison(arms)
    assert len(comparison) == 3
    assert {"random", "prior_consult_batch"} <= set(comparison[0])

    efficiency = analysis.efficiency(arms, reference="random")
    assert efficiency and "random" in efficiency[0]


def test_the_command_line_tool_runs_on_that_workspace(workspace, tmp_path):
    """The tool, not its pieces — the same reason ``dry_run_notebook`` exists."""

    out = tmp_path / "out"
    process = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "analyze_chain.py"), str(workspace),
         "--out", str(out), "--reference", "random"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "at equal oracle cost" in process.stdout
    for name in ("comparison.csv", "comparison.md", "comparison.tex",
                 "efficiency.csv", "arm_random.csv", "depth.csv",
                 "tail_recall_vs_cost.png", "tail_recall_vs_cost.pdf"):
        assert (out / name).exists(), f"{name} was not written"


# ------------------------------------------------------------------ the cost ---


def test_the_real_cost_is_what_was_asked_not_what_was_budgeted(tmp_path):
    """The bug this module corrects, put in front of the analyser deliberately.

    ``oracle_cost_so_far`` is written as budget × task index. ``selection.select``
    caps its quota at the candidates still available, so a depleted pool spends
    less — and the plan's x-axis is exactly this number.
    """

    write_arm(tmp_path, "random", [
        {"task": "t2", "asked": 600, "oracle_cost_so_far": 600, "U_Recall_tail": 10.0},
        {"task": "t3", "asked": 600, "oracle_cost_so_far": 1200, "U_Recall_tail": 12.0},
        {"task": "t4", "asked": 240, "oracle_cost_so_far": 1800, "U_Recall_tail": 13.0},
    ])
    arms = analysis.load_arms(tmp_path)

    assert analysis.real_cost(arms["random"]) == [600.0, 1200.0, 1440.0]
    drift = analysis.cost_discrepancy(arms)
    assert len(drift) == 1
    assert drift[0]["task"] == "t4"
    assert drift[0]["difference"] == pytest.approx(360.0)
    assert analysis.curve(arms["random"])[-1] == (1440.0, 13.0)


def test_a_full_budget_reports_no_discrepancy(tmp_path):
    """The correction must stay silent when there is nothing to correct.

    Otherwise the warning is noise and gets ignored on the run that matters.
    """

    write_arm(tmp_path, "random", [
        {"task": "t2", "asked": 600, "oracle_cost_so_far": 600, "U_Recall_tail": 10.0},
        {"task": "t3", "asked": 600, "oracle_cost_so_far": 1200, "U_Recall_tail": 12.0},
    ])
    assert analysis.cost_discrepancy(analysis.load_arms(tmp_path)) == []


def test_a_chain_without_the_asked_column_refuses_to_guess(tmp_path):
    write_arm(tmp_path, "random", [{"task": "t2", "oracle_cost_so_far": 600}])
    with pytest.raises(analysis.AnalysisError, match="asked"):
        analysis.real_cost(analysis.load_arms(tmp_path)["random"])


# --------------------------------------------------------------- the metric ---


def test_a_missing_recall_is_not_a_zero(tmp_path):
    """``unknown_recall_by_group`` returns None for a group with no objects.

    Read as 0.0 it would look like a measured failure to find the tail, which is
    the opposite of "the test set contained no tail object to find".
    """

    write_arm(tmp_path, "random", [
        {"task": "t2", "asked": 600, "U_Recall_tail": "", "unknown_objects_tail": 0},
        {"task": "t3", "asked": 600, "U_Recall_tail": 12.0, "unknown_objects_tail": 25},
    ])
    arm = analysis.load_arms(tmp_path)["random"]

    assert arm.rows[0]["U_Recall_tail"] is None
    assert analysis.curve(arm) == [(1200.0, 12.0)]
    assert analysis.per_arm_table(arm)[0]["U_Recall_tail"] is None


def test_cost_to_reach_interpolates_between_measurements():
    points = [(600.0, 10.0), (1200.0, 20.0)]
    assert analysis.cost_to_reach(points, 10.0) == 600.0
    assert analysis.cost_to_reach(points, 15.0) == pytest.approx(900.0)
    assert analysis.cost_to_reach(points, 20.0) == 1200.0


def test_an_unreached_level_is_none_not_a_large_number():
    """"Never got there" and "got there expensively" are different findings."""

    assert analysis.cost_to_reach([(600.0, 10.0), (1200.0, 12.0)], 40.0) is None
    assert analysis.cost_to_reach([], 1.0) is None


def test_efficiency_says_which_arm_needed_less_annotation(tmp_path):
    """The plan's prediction, stated as a number and checked against a case."""

    write_arm(tmp_path, "random", [
        {"task": "t2", "asked": 600, "U_Recall_tail": 5.0},
        {"task": "t3", "asked": 600, "U_Recall_tail": 10.0},
    ])
    write_arm(tmp_path, "prior_consult_batch", [
        {"task": "t2", "asked": 600, "U_Recall_tail": 10.0},
        {"task": "t3", "asked": 600, "U_Recall_tail": 20.0},
    ])
    rows = analysis.efficiency(analysis.load_arms(tmp_path), reference="random")

    at_ten = next(row for row in rows if row["level"] == 10.0)
    assert at_ten["random"] == 1200.0
    assert at_ten["prior_consult_batch"] == 600.0
    assert at_ten["prior_consult_batch_saving"] == pytest.approx(600.0)
    # its first measurement was already at the level, so 600 is an upper bound
    assert at_ten["prior_consult_batch_bounded"] is True
    assert at_ten["random_bounded"] is False


def test_an_arm_that_never_reaches_the_level_reports_nothing(tmp_path):
    write_arm(tmp_path, "random", [{"task": "t2", "asked": 600, "U_Recall_tail": 30.0}])
    write_arm(tmp_path, "objectness", [{"task": "t2", "asked": 600, "U_Recall_tail": 4.0}])
    rows = analysis.efficiency(analysis.load_arms(tmp_path), reference="random")
    assert rows[0]["objectness"] is None
    assert rows[0]["objectness_saving"] is None


# ----------------------------------------------------------- partial chains ---


def test_the_comparison_stops_where_the_shortest_arm_stopped(tmp_path):
    """A four-task arm against a one-task arm is not a result, it is a mistake."""

    write_arm(tmp_path, "random", [
        {"task": f"t{i}", "asked": 600, "U_Recall_tail": float(i)} for i in (2, 3, 4, 5)
    ])
    write_arm(tmp_path, "objectness", [
        {"task": "t2", "asked": 600, "U_Recall_tail": 2.0},
    ])
    arms = analysis.load_arms(tmp_path)

    assert len(analysis.comparison(arms)) == 1
    dropped = {row["arm"]: row["tasks_dropped_from_comparison"]
               for row in analysis.depth_report(arms)}
    assert dropped == {"random": 3, "objectness": 0}


def test_a_chain_with_no_grouped_recall_says_so_instead_of_drawing_nothing(tmp_path):
    """``measure_grouped_recall=False`` writes no U_Recall_tail at all."""

    write_arm(tmp_path, "random", [{"task": "t2", "asked": 600, "known_mAP50": 40.0}])
    arms = analysis.load_arms(tmp_path)
    with pytest.raises(analysis.AnalysisError, match="U_Recall_tail"):
        analysis.plot_curves(arms, tmp_path / "figure")
    with pytest.raises(analysis.AnalysisError, match="never measured"):
        analysis.efficiency(arms, reference="random")


# ------------------------------------------------------------------ reading ---


def test_an_empty_workspace_says_what_to_download(tmp_path):
    with pytest.raises(analysis.AnalysisError, match="results_"):
        analysis.load_arms(tmp_path)
    with pytest.raises(analysis.AnalysisError, match="does not exist"):
        analysis.load_arms(tmp_path / "nope")


def test_two_files_claiming_the_same_arm_are_refused(tmp_path):
    """Trap 5: one workspace per configuration, or the metrics get mixed."""

    write_arm(tmp_path / "a", "random", [{"task": "t2", "asked": 600}])
    write_arm(tmp_path / "b", "random", [{"task": "t2", "asked": 600}])
    with pytest.raises(analysis.AnalysisError, match="own workspace"):
        analysis.load_arms(tmp_path)


def test_a_header_with_no_rows_is_not_a_finished_chain(tmp_path):
    path = tmp_path / "random" / "results_random.csv"
    path.parent.mkdir(parents=True)
    path.write_text("task,asked\n", encoding="utf-8")
    with pytest.raises(analysis.AnalysisError, match="no task finished"):
        analysis.load_arms(tmp_path)


def test_the_tables_render_missing_values_and_underscores(tmp_path):
    rows = [{"task": "t2", "U_Recall_tail": 12.345, "new_mAP50": None}]
    markdown = analysis.to_markdown(rows)
    assert "| 12.35 |" in markdown and "| — |" in markdown

    latex = analysis.to_latex(rows, caption="c", label="tab:x")
    assert r"U\_Recall\_tail" in latex
    assert r"\caption{c}" in latex and "12.35" in latex
    assert "—" not in latex, "an em dash breaks a plain LaTeX build"
