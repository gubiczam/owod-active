"""The replay comparison, exercised on workspaces the chain really wrote.

The point of these is that tomorrow's analysis must not be written tomorrow.
Every table is produced today against workspaces built by ``runner.run_chain``
itself, including the two situations that will actually occur: a run that has
not started, and a run that stopped part-way through the chain.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
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


def test_t3_through_t6_use_the_growing_protocol_prefix(tmp_path):
    """Regression for the real failure: only t2 has a 19-class prefix."""

    workspace = build_run(tmp_path, "none", n_tasks=6)
    run = comparison.load_run(workspace)
    assert run is not None
    expected = {"t2": 19, "t3": 20, "t4": 21, "t5": 22, "t6": 23}
    assert set(run.per_task_ap) == set(expected)
    for task, n_prev in expected.items():
        report = run.per_class_checks[task]
        assert report["usable"] is True, (task, report)
        assert report["previous_introduced_classes"] == n_prev
        assert report["current_introduced_classes"] == 1
        assert {check["quantity"] for check in report["checks"]} >= {
            "previous_known_AP50", "current_known_AP50", "known_AP50",
        }


def test_a_vector_that_does_not_reproduce_its_own_aggregates_is_refused(tmp_path):
    """The failure this guards: silently reporting the wrong classes."""

    from owl import metrics

    workspace = build_run(tmp_path, "none")
    task_dir = next(p for p in workspace.iterdir() if p.name.endswith("_random"))
    payload = json.loads((task_dir / "metrics.json").read_text())

    original = list(payload["coco_eval_bbox"])

    # A vector of the right length whose contents no longer match the file's own
    # aggregates. Told the right class counts, the rebuild disagrees; left to
    # recover them, no prefix averages to the reported value at all. Either way
    # it is refused, and the reason says which.
    payload["coco_eval_bbox"] = [30.0, 30.0, *[7.0] * 80, 0.5]
    told = metrics.validate_per_class_ap50(payload, n_prev=19, n_current=1)
    assert told["usable"] is False
    assert "does not reproduce" in told["reason"]

    recovered = metrics.validate_per_class_ap50(payload)
    assert recovered["usable"] is False
    assert "no prefix" in recovered["reason"]

    # and a vector of the wrong length is refused before it is read at all
    payload["coco_eval_bbox"] = [1.0, 2.0, 3.0]
    report = metrics.validate_per_class_ap50(payload)
    assert report["usable"] is False
    assert "entries" in report["reason"]

    # the untouched file recovers its own counts and passes
    payload["coco_eval_bbox"] = original
    good = metrics.validate_per_class_ap50(payload)
    assert good["usable"] is True
    assert good["counts_recovered_from_file"] is True


def test_ambiguous_count_inference_is_diagnostic_not_authoritative():
    """Zero-heavy real AP vectors can reproduce one mean at several lengths."""

    from owl import metrics

    payload = {
        "coco_eval_bbox": [0.0, 0.0, *[0.0] * 81],
        "previous_known_AP50": 0.0,
        "current_known_AP50": 0.0,
        "known_AP50": 0.0,
        "unknown_AP50": 0.0,
    }
    inferred = metrics.infer_introduced_counts(payload)
    assert inferred["found"] is False
    assert inferred["ambiguous"] is True
    assert len(inferred["previous_candidates"]) > 1

    # The protocol counts are still sufficient to validate every aggregate;
    # ambiguity in the optional cross-check must not overrule that authority.
    told = metrics.validate_per_class_ap50(payload, n_prev=19, n_current=1)
    assert told["usable"] is True
    assert told["counts_recovered_from_file"] is False
    assert "multiple prefixes" in told["count_inference_reason"]

    untold = metrics.validate_per_class_ap50(payload)
    assert untold["usable"] is False
    assert "multiple prefixes" in untold["reason"]


def test_invalid_per_class_values_are_withheld_from_tables(tmp_path):
    """A provenance warning must not leave the rejected values reportable."""

    workspace = build_run(tmp_path, "none")
    task = max(workspace.glob("t*_random"))
    path = task / "metrics.json"
    payload = json.loads(path.read_text())
    payload["coco_eval_bbox"] = [30.0, 30.0, *[7.0] * 80, 0.5]
    path.write_text(json.dumps(payload))

    run = comparison.load_run(workspace)
    assert run is not None
    assert run.per_class_checks[run.final_task]["usable"] is False
    assert run.final_task not in run.per_task_ap
    rows = comparison.table_per_class({run.name: run})
    assert all(row[f"{run.name}:final_AP50"] is None for row in rows)


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


def test_default_discovery_ignores_historical_workspaces(tmp_path):
    build_run(tmp_path, "none")
    historical = build_run(tmp_path, "uniform")
    historical.rename(tmp_path / "objectness")

    runs = comparison.load_runs(tmp_path)
    assert list(runs) == ["random__none"]

    deliberately_included = comparison.load_runs(tmp_path, include=["objectness"])
    assert list(deliberately_included) == ["objectness"]


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


# ------------------------------------------------------------------- anchor ---


def _patch_recorded_split(
    workspace: Path, image_ids: list[str], *, include_ground_truth: bool = True,
) -> None:
    """Make the fake run carry the split provenance the real bridge writes."""

    for path in workspace.glob("t*_random/metrics_detections.json"):
        payload = json.loads(path.read_text())
        payload["dataset"] = "OWDETR"
        payload["class_names"] = [*protocol.CLASS_ORDER, "unknown"]
        payload["image_count"] = len(image_ids)
        payload["ground_truth"] = (
            [{"image_id": image_id, "class_name": "unknown", "box": [0, 0, 1, 1]}
             for image_id in image_ids]
            if include_ground_truth else []
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        metrics_path = path.with_name("metrics.json")
        metrics_payload = json.loads(metrics_path.read_text())
        metrics_payload["test_set"] = "owl_shared_test"
        metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")


def _run_anchor_tool(workspace: Path, root: Path, *extra: str):
    (root / "t1.pth").write_bytes(b"fake t1")
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "evaluate_anchor.py"),
         "--workspace", str(workspace), "--checkpoint", str(root / "t1.pth"),
         "--prob-root", str(root / "PROB"), "--data-root", str(root / "DATA"),
         "--dry-run", *extra],
        capture_output=True, text=True, check=False,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )


def _prepare_anchor_inputs(workspace: Path, root: Path) -> list[str]:
    """The committed split plus the exact XML tree PROB would read."""

    from owl import evaluation_subset

    config = json.loads((workspace / "config.json").read_text())
    chain = protocol.build_chain(int(config["n_tasks"]))
    archive = ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz"
    subset = evaluation_subset.from_archive(
        archive, [task.new_class for task in chain[1:]], seed=int(config["seed"]),
        remainder_multiplier=1, max_per_class=150)
    wanted = set(subset.image_ids)
    annotations = root / "DATA" / "Annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    (root / "DATA" / "JPEGImages").mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if not member.isfile() or not member.name.endswith(".xml") \
                    or Path(member.name).stem not in wanted:
                continue
            source = handle.extractfile(member)
            assert source is not None
            (annotations / Path(member.name).name).write_bytes(source.read())
    _patch_recorded_split(workspace, list(subset.image_ids))
    return list(subset.image_ids)


def test_the_anchor_tool_refuses_a_split_the_chain_was_not_scored_on(tmp_path):
    """An anchor measured on a different split is worse than no anchor."""

    workspace = build_run(tmp_path, "none")
    (workspace / "anchor_metrics.json").unlink(missing_ok=True)
    _patch_recorded_split(workspace, ["not-the-real-split"])

    result = _run_anchor_tool(workspace, tmp_path)
    assert result.returncode == 1
    assert "does not match the one the chain was scored on" in result.stdout + result.stderr


def test_the_anchor_tool_verifies_the_split_before_touching_prob(tmp_path):
    """It rebuilds the chain's own split and proves it matches, then stops."""

    workspace = build_run(tmp_path, "none")
    (workspace / "anchor_metrics.json").unlink(missing_ok=True)
    _prepare_anchor_inputs(workspace, tmp_path)

    split = tmp_path / "DATA" / "ImageSets" / "OWDETR" / "owl_shared_test.txt"
    split.parent.mkdir(parents=True)
    split.write_text("sentinel\n")

    result = _run_anchor_tool(workspace, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "split verified" in result.stdout
    assert "PROB was not called" in result.stdout
    assert not (workspace / "anchor_metrics.json").exists(), "dry run must not write it"
    assert split.read_text() == "sentinel\n", "dry run must not mutate the data root"


def test_the_anchor_tool_requires_the_exact_image_ids_not_only_the_count(tmp_path):
    workspace = build_run(tmp_path, "none")
    (workspace / "anchor_metrics.json").unlink(missing_ok=True)
    image_ids = _prepare_anchor_inputs(workspace, tmp_path)
    wrong = [*image_ids]
    wrong[-1] = "same-count-different-image"
    _patch_recorded_split(workspace, wrong)

    result = _run_anchor_tool(workspace, tmp_path)
    assert result.returncode == 1
    assert "1 missing, 1 stray" in result.stdout + result.stderr


def test_the_anchor_tool_requires_the_same_image_order(tmp_path):
    workspace = build_run(tmp_path, "none")
    (workspace / "anchor_metrics.json").unlink(missing_ok=True)
    image_ids = _prepare_anchor_inputs(workspace, tmp_path)
    _patch_recorded_split(workspace, list(reversed(image_ids)))

    result = _run_anchor_tool(workspace, tmp_path)
    assert result.returncode == 1
    assert "0 missing, 0 stray" in result.stdout + result.stderr


def test_the_anchor_tool_requires_the_annotations_prob_will_read(tmp_path):
    workspace = build_run(tmp_path, "none")
    (workspace / "anchor_metrics.json").unlink(missing_ok=True)
    image_ids = _prepare_anchor_inputs(workspace, tmp_path)
    changed = tmp_path / "DATA" / "Annotations" / f"{image_ids[0]}.xml"
    changed.write_text("<annotation />")

    result = _run_anchor_tool(workspace, tmp_path)
    assert result.returncode == 1
    assert "differ from the committed archive" in result.stdout + result.stderr


def _run_anchor_main(monkeypatch, workspace: Path, root: Path, fake_bridge) -> int:
    spec = importlib.util.spec_from_file_location(
        "evaluate_anchor_under_test", ROOT / "tools" / "evaluate_anchor.py")
    assert spec is not None and spec.loader is not None
    evaluate_anchor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate_anchor)

    (root / "t1.pth").write_bytes(b"fake t1")
    fake_bridge.check = lambda: {"fake": True}
    monkeypatch.setattr(evaluate_anchor.bridge, "Bridge", lambda **kwargs: fake_bridge)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_anchor.py", "--workspace", str(workspace),
        "--checkpoint", str(root / "t1.pth"), "--prob-root", str(root / "PROB"),
        "--data-root", str(root / "DATA"),
    ])
    return evaluate_anchor.main()


def test_anchor_is_published_only_after_a_successful_staged_evaluation(
    tmp_path, monkeypatch,
):
    workspace = build_run(tmp_path, "none")
    (workspace / "anchor_metrics.json").unlink(missing_ok=True)
    _prepare_anchor_inputs(workspace, tmp_path)
    before = {path.relative_to(workspace) for path in workspace.rglob("*")}
    fake = FakeBridge()

    assert _run_anchor_main(monkeypatch, workspace, tmp_path, fake) == 0
    output = workspace / "anchor_metrics.json"
    assert output.exists()
    after = {path.relative_to(workspace) for path in workspace.rglob("*")}
    assert after - before == {Path("anchor_metrics.json")}
    assert not (tmp_path / "DATA" / "ImageSets").exists()


def test_a_failed_staged_evaluation_does_not_leave_an_anchor(tmp_path, monkeypatch):
    class InvalidMetricsBridge(FakeBridge):
        def evaluate(self, **kwargs):
            path = super().evaluate(**kwargs)
            payload = json.loads(path.read_text())
            payload["known_AP50"] += 10.0
            path.write_text(json.dumps(payload))
            return path

    workspace = build_run(tmp_path, "none")
    (workspace / "anchor_metrics.json").unlink(missing_ok=True)
    _prepare_anchor_inputs(workspace, tmp_path)

    assert _run_anchor_main(
        monkeypatch, workspace, tmp_path, InvalidMetricsBridge()) == 1
    assert not (workspace / "anchor_metrics.json").exists()


def test_the_anchor_tool_never_overwrites_an_existing_anchor(tmp_path):
    workspace = build_run(tmp_path, "none")
    assert (workspace / "anchor_metrics.json").exists()
    before = (workspace / "anchor_metrics.json").read_bytes()

    result = _run_anchor_tool(workspace, tmp_path)
    assert result.returncode == 0
    assert "already exists" in result.stdout
    assert (workspace / "anchor_metrics.json").read_bytes() == before


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
