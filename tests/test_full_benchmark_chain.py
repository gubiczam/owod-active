"""Benchmark V1's orchestration, against a stubbed detector.

These are the assertions the protocol makes about the *chain*, and every one of
them is something Method V3 either could not have made or got wrong:

* the chain is sequential — ``t3`` fine-tunes its own arm's ``t2`` checkpoint;
* no two arms share a checkpoint or a workspace;
* the oracle answers are matched across arms;
* a completed task resumes from disk and reproduces its row exactly;
* a workspace written under a different configuration is refused, not blended;
* PROB's seed is actually passed — V3's audit found it left at 0 for all twelve
  of its trajectories because the launcher never passed it;
* the protocol document and the code agree, compared as values.
"""

from __future__ import annotations

import itertools
import json
import shutil
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from owl import protocol, runner
from owl.active_selection import arms, benchmark
from owl.active_selection import budget as annotation_budget

# ------------------------------------------------------------- the protocol ---


def test_the_document_and_the_code_agree():
    assert benchmark.check_protocol()["agrees"]


def test_a_protocol_without_the_block_is_refused(tmp_path):
    path = tmp_path / "p.md"
    path.write_text("# no block here\n", encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkError, match="no ```json protocol"):
        benchmark.check_protocol(path)


def test_a_changed_value_in_the_document_is_caught(tmp_path):
    text = benchmark.PROTOCOL_PATH.read_text(encoding="utf-8")
    tampered = text.replace('"epochs": 5,', '"epochs": 4,')
    assert tampered != text
    path = tmp_path / "p.md"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkError, match="epochs: document 4, code 5"):
        benchmark.check_protocol(path)


def test_a_missing_field_is_caught_rather_than_ignored(tmp_path):
    text = benchmark.PROTOCOL_PATH.read_text(encoding="utf-8")
    # `epochs` is not the last field in the block, so dropping its line leaves
    # valid JSON — which is what makes "absent" a distinct failure from "invalid".
    tampered = text.replace('  "epochs": 5,\n', "")
    assert tampered != text
    path = tmp_path / "p.md"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkError, match="absent from the document"):
        benchmark.check_protocol(path)


def test_the_frozen_values_cover_every_declared_constant():
    frozen = benchmark.frozen_values()
    for name in ("epochs", "seeds", "arms", "endpoints", "labelling_policy",
                 "answer_budget_per_task", "nms_iou", "admissible_share"):
        assert name in frozen


# ------------------------------------------------------------------- chain ---


def test_the_chain_is_four_tasks_declaring_one_class_each():
    chain = benchmark.chain()
    assert [t.name for t in chain] == ["t1", "t2", "t3", "t4"]
    assert [t.new_class for t in chain[1:]] == [
        "traffic light", "fire hydrant", "stop sign"
    ]
    assert chain[0].is_anchor


def test_the_tail_band_grows_along_the_chain():
    """The reason the chain is run to t4 at all."""

    bands = [benchmark.tail_band(t) for t in benchmark.chain()]
    assert bands[1] == ("bear",)
    assert len(bands[2]) == 2
    assert len(bands[3]) == 3
    assert bands[1][0] in bands[3]


def test_the_shared_split_is_built_on_the_chains_own_classes():
    assert benchmark.declared_classes() == (
        "traffic light", "fire hydrant", "stop sign"
    )


# ------------------------------------------------------------------ config ---


def test_the_cycle_config_is_the_frozen_one():
    config = benchmark.cycle_config("proposed", 0)
    assert config.budget_unit == "answers"
    assert config.budget_per_task == benchmark.ANSWER_BUDGET_PER_TASK
    assert config.replay_arm == benchmark.REPLAY_ARM
    assert config.labelling_policy == benchmark.LABELLING_POLICY
    assert config.rounds_per_task == 1
    assert config.epochs == benchmark.EPOCHS


def test_every_arm_and_seed_gets_its_own_workspace_name():
    names = {
        benchmark.trajectory_name(arm, seed)
        for arm in arms.ORDER for seed in benchmark.SEEDS
    }
    assert len(names) == len(arms.ORDER) * len(benchmark.SEEDS)


def test_an_unregistered_arm_is_refused():
    with pytest.raises(benchmark.BenchmarkError, match="Unknown arm"):
        benchmark.cycle_config("magic", 0)


def test_an_undeclared_seed_is_refused():
    with pytest.raises(benchmark.BenchmarkError, match="not one of the declared seeds"):
        benchmark.cycle_config("random", 7)


def test_answers_unit_refuses_to_run_without_a_selector(tmp_path):
    config = benchmark.cycle_config("random", 0)
    with pytest.raises(ValueError, match="prices an \\*image\\*"):
        runner.run_chain(
            None, config, workspace=tmp_path, candidate_index={},
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        )


def test_regions_unit_refuses_a_selector(tmp_path):
    config = runner.CycleConfig(budget_unit="regions", replay_arm="none")
    with pytest.raises(ValueError, match="cannot be run under budget_unit='regions'"):
        runner.run_chain(
            None, config, workspace=tmp_path, candidate_index={},
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
            selector=lambda *a, **k: None,
        )


def test_a_legacy_workspace_stays_resumable():
    """``budget_unit`` joins the fingerprint only when it is not the old default.

    Absent means "written by code that meant something else" and must count as
    differing — but refusing every pre-2026-09-03 workspace on that basis would
    strand the completed Replay-V3 chains for no scientific gain.
    """

    legacy = runner.CycleConfig()
    assert "budget_unit" not in legacy.fingerprint()
    new = benchmark.cycle_config("random", 0)
    assert new.fingerprint()["budget_unit"] == "answers"


def test_the_prob_seed_is_actually_passed():
    """Method V3's audit found ``--seed`` left at 0 for all twelve trajectories."""

    from owl.bridge import Bridge

    assert Bridge.__dataclass_fields__["seed"].default == 0
    source = (Path(__file__).resolve().parent.parent / "tools"
              / "run_full_owod_benchmark.py").read_text(encoding="utf-8")
    assert "seed=seed," in source, (
        "the launcher must hand each trajectory's seed to the bridge; the "
        "Method V3 launcher did not and every trajectory ran on seed 0"
    )


# ---------------------------------------------------- the selector callback ---


@pytest.fixture
def small_index():
    """400 images, each holding one declared class and one task-1 class."""

    declared = [task.new_class for task in benchmark.chain()[1:]]
    return {
        f"{i:012d}": {declared[i % len(declared)]: 1 + (i % 3),
                      protocol.TASK1[i % len(protocol.TASK1)]: 1}
        for i in range(400)
    }


@pytest.fixture
def small_config():
    """The benchmark's own config, shrunk so the whole chain runs in seconds."""

    from dataclasses import replace

    return replace(
        benchmark.cycle_config("random", 0),
        candidate_images_per_task=40, proposals_per_image=4,
        budget_per_task=30, epochs=1, replay_arm="none",
    )


def _lineage_bridge():
    """The shared fake, plus the two fields the lineage assertions need.

    ``tests.test_run_chain.FakeBridge`` does not record which checkpoint a train
    call started from; adding it there would change a fixture forty other tests
    depend on, so it is recorded here instead.
    """

    from tests.test_run_chain import FakeBridge

    class LineageBridge(FakeBridge):
        def train(self, labelled_ids, **kwargs):
            produced = super().train(labelled_ids, **kwargs)
            self.calls[-1] |= {
                "previous": str(kwargs["previous_checkpoint"]),
                "output": str(produced),
            }
            return produced

    return LineageBridge()


def _run(tmp_path, index, config, arm, *, features_for=None, workspace=None):
    from tests.test_run_chain import prob_data_root

    data_root = prob_data_root(tmp_path, index)
    (tmp_path / "t1.pth").write_bytes(b"anchor")
    bridge = _lineage_bridge()
    seen: list[dict] = []

    def spy(path, image_ids, boxes, jpeg_dir, **kwargs):
        generator = np.random.default_rng(len(seen))
        block = generator.normal(size=(len(image_ids), 8)).astype(np.float32)
        block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-9)
        seen.append({"rows": len(image_ids), "path": Path(path)})
        return block

    selector = benchmark.make_selector(
        arm, candidate_index=index, jpeg_dir=data_root / "JPEGImages",
        ref_t1="unused-in-this-test",
        features_for=features_for or spy,
        reference_for=lambda _: np.zeros((0, 8), dtype=np.float32),
    )
    from dataclasses import replace as _replace

    results = runner.run_chain(
        bridge, _replace(config, arm=arm),
        workspace=workspace or (tmp_path / "work" / arm),
        candidate_index=index,
        start_checkpoint=tmp_path / "t1.pth",
        test_set="owl_shared_test",
        chain=benchmark.chain(),
        selector=selector,
    )
    return results, bridge, seen


def test_the_selector_is_handed_a_pool_with_no_answers(tmp_path, small_index, small_config):
    """A live pool has no oracle at all, so no selector can consult one."""

    inspected: list[bool] = []

    def watching(candidates, *, task, task_dir, used_images, budget, seed):
        inspected.append(candidates.has_oracle)
        images = [i for i in sorted(set(candidates.image_ids.tolist()))
                  if i not in used_images][:6]
        return arms.ArmSelection(arm="watch", images=tuple(images), anchors=(),
                                 row={"arm": "watch", "answers_spent": len(images)})

    from tests.test_run_chain import FakeBridge, prob_data_root

    prob_data_root(tmp_path, small_index)
    (tmp_path / "t1.pth").write_bytes(b"anchor")
    runner.run_chain(
        FakeBridge(), small_config,
        workspace=tmp_path / "w", candidate_index=small_index,
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=benchmark.chain(), selector=watching,
    )
    assert inspected == [False, False, False]


def test_every_arm_reaches_t4_from_its_own_previous_checkpoint(
    tmp_path, small_index, small_config
):
    for arm in ("random", "admissibility", "entropy", "proposed", "coreset",
                "proposed_v2"):
        results, bridge, _ = _run(tmp_path / arm, small_index, small_config, arm)
        assert [r.task for r in results] == ["t2", "t3", "t4"], arm
        trains = [c for c in bridge.calls if c["verb"] == "train"]
        assert len(trains) == 3, arm
        assert Path(trains[0]["previous"]).name == "t1.pth", arm
        for earlier, later in itertools.pairwise(trains):
            assert later["previous"] == earlier["output"], (
                f"{arm}: t{trains.index(later) + 3} did not train from the "
                "checkpoint its own previous task produced"
            )


def test_two_arms_never_share_a_checkpoint(tmp_path, small_index, small_config):
    produced = {}
    for arm in ("random", "entropy"):
        _, bridge, _ = _run(tmp_path / arm, small_index, small_config, arm)
        produced[arm] = [
            c["output"] for c in bridge.calls if c["verb"] == "train"
        ]
    assert not set(produced["random"]) & set(produced["entropy"])


def test_the_answer_budget_is_matched_across_arms(tmp_path, small_index, small_config):
    spent = {}
    for arm in ("random", "admissibility", "entropy", "proposed"):
        results, _, _ = _run(tmp_path / arm, small_index, small_config, arm)
        spent[arm] = [r.selection_row["answers_spent"] for r in results]
    flat = [v for values in spent.values() for v in values]
    assert max(flat) <= small_config.budget_per_task
    # every arm spends the budget up to at most one image's cost
    assert max(flat) - min(flat) <= 4, spent


def test_a_coverage_arm_carries_its_reference_forward(tmp_path, small_index, small_config):
    results, _, seen = _run(tmp_path / "p", small_index, small_config, "proposed")
    assert len(seen) == 3, "one semantic pass per task"
    points = [r.selection_row["reference_points"] for r in results]
    assert points[0] == 0
    assert points[1] > 0 and points[2] > points[1], (
        f"the labelled reference must grow along the chain, got {points}"
    )


def test_a_gated_arm_embeds_fewer_rows_than_the_ungated_control(
    tmp_path, small_index, small_config
):
    _, _, gated = _run(tmp_path / "g", small_index, small_config, "proposed")
    _, _, ungated = _run(tmp_path / "u", small_index, small_config, "coreset")
    assert gated[0]["rows"] < ungated[0]["rows"]


def test_a_resumed_chain_reproduces_its_rows(tmp_path, small_index, small_config):
    workspace = tmp_path / "shared"
    first, _, _ = _run(tmp_path / "a", small_index, small_config, "random",
                       workspace=workspace)
    again, bridge, _ = _run(tmp_path / "a", small_index, small_config, "random",
                            workspace=workspace)
    assert [r.flat() for r in first] == [r.flat() for r in again]
    assert not [c for c in bridge.calls if c["verb"] == "train"], (
        "a completed chain retrained on resume"
    )


def test_a_workspace_from_another_configuration_is_refused(
    tmp_path, small_index, small_config
):
    from dataclasses import replace

    workspace = tmp_path / "shared"
    _run(tmp_path / "a", small_index, small_config, "random", workspace=workspace)
    with pytest.raises(RuntimeError, match="different configuration"):
        _run(tmp_path / "a", small_index,
             replace(small_config, epochs=small_config.epochs + 1), "random",
             workspace=workspace)


def test_the_row_prices_what_the_detector_received(tmp_path, small_index, small_config):
    results, _, _ = _run(tmp_path / "r", small_index, small_config, "random")
    for row in (r.selection_row for r in results):
        assert row["boxes_labelled"] >= row["boxes_supervised"]
        assert row["boxes_labelled"] == row["boxes_supervised"] + row["boxes_banked"]
        assert row["training_iterations"] > 0
        assert row["images_opened"] >= row["images_trainable"] - row[
            "images_from_earlier_tasks"]
        # what was bought and what PROB was handed are separate quantities
        assert row["boxes_trained_on"] > 0
        assert row["boxes_trained_on"] == (
            row["boxes_trained_on_head"] + row["boxes_trained_on_medium"]
            + row["boxes_trained_on_tail"]
        )


def test_what_prob_was_handed_matches_what_prob_would_keep(
    tmp_path, small_index, small_config
):
    """The accounting is checked against PROB's own filter, not against itself.

    ``remove_unknown_instances`` keeps ``category_id in range(0, prev + current)``
    on a fine-tuning split. ``boxes_trained_on`` claims to be exactly that count
    over the images handed to ``train``, so it is recomputed here from the
    annotations on disk, the way the loader would.
    """

    from xml.etree import ElementTree

    from owl.evaluation_subset import canonical_class_name
    from tests.test_run_chain import prob_data_root

    data_root = prob_data_root(tmp_path / "x", small_index)
    results, bridge, _ = _run(tmp_path / "x", small_index, small_config, "random")
    trains = [c for c in bridge.calls if c["verb"] == "train"]
    chain = benchmark.chain()
    for result, call, task in zip(results, trains, chain[1:], strict=True):
        declared = set(protocol.CLASS_ORDER[: task.n_prev + task.n_new])
        counted = 0
        for image_id in call["images"]:
            root = ElementTree.parse(
                data_root / "Annotations" / f"{image_id}.xml"
            ).getroot()
            counted += sum(
                1 for element in root.findall("object")
                if canonical_class_name(element.findtext("name", "")) in declared
            )
        assert result.selection_row["boxes_trained_on"] == counted, (
            f"{task.name}: the row says "
            f"{result.selection_row['boxes_trained_on']} supervised boxes and "
            f"PROB would keep {counted}"
        )


def test_an_object_bought_early_is_credited_to_the_task_that_can_learn_it(
    tmp_path, small_index, small_config
):
    results, _, _ = _run(tmp_path / "b", small_index, small_config, "random")
    t2, t3, t4 = (r.selection_row for r in results)
    # Every image in this index holds one declared class, and a third of them
    # hold the class t3 declares, so t2 must credit some of them forward.
    assert t2["acquired_becomes_known_t3"] > 0
    assert t2["acquired_becomes_known_t4"] > 0
    # t3 can still credit t4; t4 is the last task, so there is nothing after it.
    assert t3["acquired_becomes_known_t4"] > 0
    assert "acquired_becomes_known_t3" not in t3
    assert not any(key.startswith("acquired_becomes_known") for key in t4)


def test_reference_blocks_only_read_earlier_tasks(tmp_path):
    for index in (2, 3, 4):
        directory = tmp_path / f"t{index}_proposed"
        directory.mkdir()
        with (directory / "coverage_reference.npz").open("wb") as handle:
            np.savez_compressed(handle, features=np.zeros((index, 3), dtype=np.float16))
    blocks = benchmark.reference_blocks(tmp_path / "t4_proposed", task_index=4)
    assert [len(b) for b in blocks] == [2, 3]
    assert benchmark.reference_blocks(tmp_path / "t2_proposed", task_index=2) == []


def test_make_selector_refuses_a_coverage_arm_without_a_reference(tmp_path):
    with pytest.raises(benchmark.BenchmarkError, match="task-1 reference"):
        benchmark.make_selector(
            "proposed", candidate_index={}, jpeg_dir=tmp_path, ref_t1=None
        )


def test_make_selector_does_not_need_a_reference_for_a_ranking_arm(tmp_path):
    selector = benchmark.make_selector(
        "random", candidate_index={}, jpeg_dir=tmp_path, ref_t1=None
    )
    assert callable(selector)


# --------------------------------------------------------------- manifest ---


def test_the_manifest_records_the_pins_and_the_rules():
    payload = benchmark.manifest(
        trajectories=[{"trajectory": "random__seed0", "status": "COMPLETE"}],
        owl_commit="a" * 40, prob_commit="b" * 40,
        prob_repository="https://example.invalid/PROB.git",
        checkpoint="/x/t1.pth", checkpoint_sha256="c" * 64,
        test_set="owl_shared_test", test_images=837,
    )
    assert payload["pins"]["prob_commit"] == "b" * 40
    assert payload["frozen"]["epochs"] == benchmark.EPOCHS
    assert payload["reporting_rules"] == list(benchmark.REPORTING)
    assert [t["task"] for t in payload["chain"]] == ["t1", "t2", "t3", "t4"]
    assert payload["evaluation"]["images"] == 837
    assert payload["dry_run"] is False


def test_write_json_is_atomic(tmp_path):
    target = tmp_path / "deep" / "m.json"
    benchmark.write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert not list(tmp_path.rglob("*.part"))


# ------------------------------------------------------------- the runner ---


def test_the_row_records_supervision_and_steps_for_every_run():
    """The two columns without which an AP difference cannot be interpreted."""

    source = (Path(__file__).resolve().parent.parent / "owl" / "runner.py").read_text(
        encoding="utf-8"
    )
    assert '"training_iterations"' in source
    assert "annotation_budget.supervision(" in source
    assert "annotation_budget.acquisition(" in source


def test_budget_unit_is_result_affecting():
    names = {f.name for f in fields(runner.CycleConfig)}
    assert "budget_unit" in names
    assert any(name == "budget_unit" for name, _ in runner.CycleConfig.LATER_ADDITIONS)


def test_the_chain_used_by_the_benchmark_is_the_repositorys_own():
    assert benchmark.chain() == protocol.build_chain(benchmark.N_TASKS)


def test_the_ledger_is_not_reachable_from_a_selector():
    """A selector is handed a cost function that reads counts, never classes."""

    cost = annotation_budget.cost_function({"i1": {"person": 3}})
    assert cost("i1") == 3
    assert cost("absent") == annotation_budget.ANSWER_FLOOR

# -------------------------------------------- lineage across a broken session ---


def test_a_session_that_died_mid_chain_resumes_onto_its_own_checkpoint(
    tmp_path, small_index, small_config
):
    """The lineage check that only a *partial* resume can make.

    A completed chain resumes by restoring every task and training nothing, so
    it proves nothing about lineage. The dangerous case is a session that
    finished t2 and died: t3 must fine-tune the t2 checkpoint that is on disk,
    not restart from the anchor.
    """

    workspace = tmp_path / "broken"
    first, _, _ = _run(tmp_path / "a", small_index, small_config, "random",
                       workspace=workspace)
    assert [r.task for r in first] == ["t2", "t3", "t4"]

    # Drop t4 as though the session had died in it. t3's checkpoint is on disk —
    # `keep_checkpoints` never prunes the newest, which is the invariant the
    # resume depends on and the next test is about.
    shutil.rmtree(workspace / "t4_random")
    t3_checkpoint = workspace / "t3_random" / "checkpoint.pth"
    assert t3_checkpoint.is_file()

    again, bridge, _ = _run(tmp_path / "a", small_index, small_config, "random",
                            workspace=workspace)
    trains = [c for c in bridge.calls if c["verb"] == "train"]
    assert len(trains) == 1, "only t4 should have retrained"
    assert Path(trains[0]["previous"]) == t3_checkpoint, (
        f"t4 resumed from {trains[0]['previous']} instead of t3's own checkpoint"
    )
    assert [r.task for r in again] == ["t2", "t3", "t4"]
    # and the restored rows are identical to the ones originally written
    assert [r.flat() for r in again[:2]] == [r.flat() for r in first[:2]]


def test_a_lost_previous_checkpoint_stops_the_chain_instead_of_restarting_it(
    tmp_path, small_index, small_config
):
    """A silent restart from the anchor would report a chain that never ran."""

    workspace = tmp_path / "pruned"
    _run(tmp_path / "a", small_index, small_config, "random", workspace=workspace)
    shutil.rmtree(workspace / "t4_random")
    (workspace / "t3_random" / "checkpoint.pth").unlink()

    with pytest.raises(RuntimeError, match="break the checkpoint lineage"):
        _run(tmp_path / "a", small_index, small_config, "random", workspace=workspace)


def test_the_first_task_is_allowed_to_start_from_the_anchor(
    tmp_path, small_index, small_config
):
    """The guard must not fire on a fresh chain."""

    results, bridge, _ = _run(tmp_path / "fresh", small_index, small_config, "random")
    trains = [c for c in bridge.calls if c["verb"] == "train"]
    assert Path(trains[0]["previous"]).name == "t1.pth"
    assert len(results) == 3


# ---------------------------------------- isolation from the earlier methods ---


def test_the_results_directory_is_not_any_earlier_experiments(tmp_path):
    """Method V1, V2 and V3 results must be unreachable from this launcher."""

    source = (Path(__file__).resolve().parent.parent / "tools"
              / "run_full_owod_benchmark.py").read_text(encoding="utf-8")
    for earlier in ("method_v3_selection_transfer", "method_v2", "replay_v3",
                    "dinov2_vitb14_method_v2_v1", "dinov2_vitb14_stage2_views_v1"):
        assert earlier not in source, earlier
    notebook = (Path(__file__).resolve().parent.parent / "notebooks"
                / "full_owod_active_benchmark_v1.ipynb").read_text(encoding="utf-8")
    assert "results/full_owod_active_benchmark_v1" in notebook
    for earlier in ("results/method_v3_selection_transfer", "results/method_v2"):
        assert earlier not in notebook, earlier


def test_the_only_frozen_artefact_reused_is_the_labelled_reference():
    """Nothing recomputes a Method V2 export, and nothing overwrites one."""

    notebook = (Path(__file__).resolve().parent.parent / "notebooks"
                / "full_owod_active_benchmark_v1.ipynb").read_text(encoding="utf-8")
    assert "ref_t1_dinov2_vitb14_cap1000_v1.npz" in notebook
    # the two Stage-2 exports belong to Method V3 and are not touched here
    assert "dinov2_vitb14_stage2_views_v1" not in notebook


def test_the_manifest_records_the_seed_and_the_replay_policy(tmp_path):
    source = (Path(__file__).resolve().parent.parent / "tools"
              / "run_full_owod_benchmark.py").read_text(encoding="utf-8")
    for field in ('"prob_seed": seed', '"replay_arm": bm.REPLAY_ARM',
                  '"replay_objects": bm.REPLAY_OBJECTS',
                  '"checkpoint_lineage"'):
        assert field in source, field


# ------------------------------------------- recovering a column after the fact ---


def _fake_results(root: Path, arm: str, index: dict) -> Path:
    """A results tree shaped like a completed trajectory, ids files included."""

    trajectory = f"{arm}__seed0"
    images = sorted(index)
    for offset, task in enumerate(benchmark.chain()[1:]):
        train = root / trajectory / f"{task.name}_{arm}" / "train"
        train.mkdir(parents=True, exist_ok=True)
        handed = images[offset : offset + 2]
        (train / "labelled_ids.txt").write_text("\n".join(handed) + "\n", encoding="utf-8")
        (train / "replay_ids.txt").write_text("alias-1\nalias-2\n", encoding="utf-8")
    benchmark.write_json(root / "manifest.json", {
        "trajectories": [
            {"trajectory": trajectory, "arm": arm, "seed": 0, "status": "COMPLETE"}
        ],
    })
    return root


def test_boxes_trained_on_is_recoverable_from_the_id_files(tmp_path, monkeypatch):
    """Session 1 ran before the column existed; the number is not lost."""

    from tools import backfill_boxes_trained_on as backfill

    index = {
        "000000000001": {"person": 3, "traffic light": 2, "banana": 4},
        "000000000002": {"car": 1, "fire hydrant": 2},
        "000000000003": {"bear": 1, "stop sign": 5},
        "000000000004": {"banana": 7},
    }
    results = _fake_results(tmp_path / "results", "random", index)
    monkeypatch.setattr(backfill, "CANDIDATE_INDEX", tmp_path / "index.json")
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")

    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    rows = backfill.rows_for(
        results, manifest["trajectories"][0], index, protocol.load_groups()
    )
    assert [r["task"] for r in rows] == ["t2", "t3", "t4"]

    # t2 is handed images 1 and 2; `traffic light` is declared, `fire hydrant`
    # is not yet, and `banana` never is in this chain.
    t2 = rows[0]
    assert t2["boxes_trained_on"] == 3 + 2 + 1        # person, traffic light, car
    assert t2["boxes_dropped_as_undeclared"] == 4 + 2  # banana, fire hydrant
    assert t2["boxes_on_those_images_total"] == 12
    assert t2["replay_alias_images"] == 2
    assert t2["images_handed_to_prob"] == 2

    # t3 is handed images 2 and 3. `fire hydrant` is declared now and counts;
    # `stop sign` is not declared until t4, so its five boxes are dropped here
    # and appear only in t4's count. That the same image yields a different
    # supervised total at two tasks is the whole point of the column.
    t3, t4 = rows[1], rows[2]
    assert t3["boxes_trained_on"] == 1 + 2 + 1          # car, hydrant, bear
    assert t3["boxes_dropped_as_undeclared"] == 5       # the stop signs
    assert t3["boxes_trained_on_tail"] == 1 + 2         # bear, fire hydrant

    # t4 is handed images 3 and 4; now the stop signs count.
    assert t4["boxes_trained_on"] == 1 + 5              # bear, stop sign
    assert t4["boxes_dropped_as_undeclared"] == 7       # the bananas
    assert t4["boxes_trained_on_tail"] == 1 + 5


def test_the_backfill_refuses_to_write_into_the_results_it_reads(tmp_path):
    """A tool that recomputes a measurement must not be able to overwrite it."""

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve().parent.parent / "tools"
             / "backfill_boxes_trained_on.py"),
         "--results", str(tmp_path), "--out", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "must not be the results directory" in result.stdout + result.stderr


# ----------------------------------------------------------- Proposed-v2 ---


def test_v2_never_consults_the_task_one_reference(tmp_path, small_index, small_config):
    """REF-T1 is removed from v2's objective, so it must not even be read."""

    def forbidden(_path):
        raise AssertionError("v2 read REF-T1; its reference is trajectory-only")

    from tests.test_run_chain import prob_data_root

    data_root = prob_data_root(tmp_path, small_index)
    (tmp_path / "t1.pth").write_bytes(b"anchor")

    def features(path, image_ids, boxes, jpeg_dir, **kwargs):
        block = np.random.default_rng(len(image_ids)).normal(
            size=(len(image_ids), 8)).astype(np.float32)
        return block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-9)

    selector = benchmark.make_selector(
        "proposed_v2", candidate_index=small_index,
        jpeg_dir=data_root / "JPEGImages", ref_t1=None,
        features_for=features, reference_for=forbidden,
    )
    from dataclasses import replace

    results = runner.run_chain(
        _lineage_bridge(), replace(small_config, arm="proposed_v2"),
        workspace=tmp_path / "v2", candidate_index=small_index,
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=benchmark.chain(), selector=selector,
    )
    assert [r.task for r in results] == ["t2", "t3", "t4"]


def test_v2_needs_no_ref_t1_but_v1_still_does(tmp_path):
    assert callable(benchmark.make_selector(
        "proposed_v2", candidate_index={}, jpeg_dir=tmp_path, ref_t1=None))
    with pytest.raises(benchmark.BenchmarkError, match="task-1 reference"):
        benchmark.make_selector(
            "proposed", candidate_index={}, jpeg_dir=tmp_path, ref_t1=None)


def test_v2_starts_with_an_empty_reference_and_grows_it(
    tmp_path, small_index, small_config
):
    results, _, seen = _run(tmp_path / "v2", small_index, small_config, "proposed_v2")
    points = [r.selection_row["reference_points"] for r in results]
    assert points[0] == 0, "t2 must start from nothing bought"
    assert points[1] > 0 and points[2] > points[1], points
    # and the first pick of t2 had nothing to measure against
    assert results[0].selection_row["coverage_picks_without_reference"] == 1
    assert results[0].selection_row["coverage_first_pick_distance"] is None
    assert results[1].selection_row["coverage_picks_without_reference"] == 0
    assert len(seen) == 3, "one semantic pass per task"


def test_v2_embeds_fewer_rows_than_v1(tmp_path, small_index, small_config):
    _, _, v1 = _run(tmp_path / "a", small_index, small_config, "proposed")
    _, _, v2 = _run(tmp_path / "b", small_index, small_config, "proposed_v2")
    assert v2[0]["rows"] < v1[0]["rows"]


def test_v2_records_its_own_reference_scope(tmp_path, small_index, small_config):
    results, _, _ = _run(tmp_path / "v2", small_index, small_config, "proposed_v2")
    for row in (r.selection_row for r in results):
        assert row["reference_scope"] == "trajectory"
        assert row["population_ranked"] < row["population_admissible"]


def test_adding_v2_does_not_move_an_existing_arms_fingerprint():
    """The four measured seed-0 trajectories must stay resumable."""

    for arm in ("random", "admissibility", "proposed", "entropy"):
        fingerprint = benchmark.cycle_config(arm, 0).fingerprint()
        assert fingerprint["arm"] == arm
        assert fingerprint["budget_per_task"] == benchmark.ANSWER_BUDGET_PER_TASK
        assert fingerprint["epochs"] == benchmark.EPOCHS
        assert fingerprint["replay_arm"] == benchmark.REPLAY_ARM
        assert fingerprint["budget_unit"] == "answers"
        assert "informative" not in fingerprint
        assert set(fingerprint) == set(
            runner.CycleConfig.RESULT_AFFECTING) | {"budget_unit"}


def test_v2_gets_its_own_workspace_and_does_not_touch_proposed():
    assert benchmark.trajectory_name("proposed_v2", 0) == "proposed_v2__seed0"
    assert benchmark.trajectory_name("proposed", 0) == "proposed__seed0"
    assert benchmark.cycle_config("proposed_v2", 0).arm == "proposed_v2"


# ------------------------------------------------------------- kill rule ---


def test_the_kill_rule_thresholds_are_the_observed_ones():
    assert benchmark.KILL_RULE.arm == "proposed_v2"
    assert benchmark.KILL_RULE.seed == benchmark.DEVELOPMENT_SEED
    assert benchmark.KILL_RULE.minimum_mean_new_class_ap50 == 3.56
    assert benchmark.KILL_RULE.minimum_final_known_map50 == 44.89


def test_the_kill_rule_needs_both_conditions():
    rule = benchmark.KILL_RULE
    assert rule.decide(4.0, 46.0)["verdict"] == "PROCEED"
    assert rule.decide(3.55, 46.0)["verdict"] == "STOP"
    assert rule.decide(4.0, 44.88)["verdict"] == "STOP"
    assert rule.decide(3.56, 44.89)["verdict"] == "PROCEED"   # boundary is inclusive
    assert rule.decide(None, 46.0)["verdict"] == "INCOMPLETE"


def test_the_kill_rule_reaches_the_manifest():
    payload = benchmark.manifest(
        trajectories=[], owl_commit="a" * 40, prob_commit="b" * 40,
        prob_repository="x", checkpoint="c", checkpoint_sha256=None,
        test_set="owl_shared_test", test_images=837,
    )
    assert payload["kill_rule"] == benchmark.KILL_RULE.as_dict()
    assert payload["development_seed_informed"] == ["proposed_v2"]
    assert any("not\npre-registered" in line or "not pre-registered" in line
               for line in payload["provenance"])
    assert any("CUDA OOM" in line for line in payload["provenance"])


def test_the_provenance_says_the_gate_is_not_ruled_out():
    joined = " ".join(benchmark.PROVENANCE)
    assert "NOT been causally ruled out" in joined
    assert "development-seed-informed" in joined
    assert "supporting development evidence only" in joined


def test_the_allocator_config_is_set_before_torch_can_load():
    from tools import run_full_owod_benchmark as launcher

    assert launcher.ALLOCATOR_CONFIG == "expandable_segments:True"
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert source.index("def configure_allocator") < source.index("def main()")
    assert "configure_allocator()" in source.split("def main() -> None:")[1][:200]


# --------------------------------------------- seed isolation, replication ---


def test_a_seed_cannot_overwrite_another_seeds_trajectory():
    """Two independent mechanisms, because the seed-0 results are irreplaceable."""

    for arm in ("random", "admissibility", "entropy"):
        # 1. different directories
        assert benchmark.trajectory_name(arm, 0) != benchmark.trajectory_name(arm, 1)
        assert benchmark.trajectory_name(arm, 1).endswith("__seed1")
        # 2. and if they were ever pointed at one, the fingerprint refuses it
        zero = benchmark.cycle_config(arm, 0).fingerprint()
        one = benchmark.cycle_config(arm, 1).fingerprint()
        assert zero != one
        assert {k for k in zero if zero[k] != one[k]} == {"seed"}


def test_a_seed_one_run_refuses_a_seed_zero_workspace(
    tmp_path, small_index, small_config
):
    """The fingerprint guard is what makes the directory scheme belt-and-braces."""

    from dataclasses import replace

    workspace = tmp_path / "shared"
    _run(tmp_path / "a", small_index, small_config, "random", workspace=workspace)
    with pytest.raises(RuntimeError, match="different configuration"):
        _run(tmp_path / "a", small_index, replace(small_config, seed=1), "random",
             workspace=workspace)


def test_the_replication_session_names_only_surviving_baselines():
    """`proposed`, `proposed_v2` and `coreset` are excluded for recorded reasons."""

    import json as _json

    notebook = _json.loads(
        (Path(__file__).resolve().parent.parent / "notebooks"
         / "full_owod_active_benchmark_v1.ipynb").read_text(encoding="utf-8")
    )
    source = "".join(notebook["cells"][1]["source"])
    import re as _re

    named = _re.findall(
        r'"([a-z_]+)"', _re.search(r"SESSION_ARMS = \(([^)]*)\)", source, _re.DOTALL).group(1)
    )
    seeds = _re.search(r"SEEDS = \(([^)]*)\)", source).group(1)
    assert named == ["random", "admissibility", "entropy"], named
    assert seeds.strip().rstrip(",") == "1", seeds
    for excluded in ("proposed", "proposed_v2", "coreset"):
        assert excluded not in named


def test_the_kill_rule_stops_proposed_v2_on_its_measured_seed_zero():
    """0.06 against a 3.56 floor. Recorded so the verdict cannot drift."""

    outcome = benchmark.KILL_RULE.decide(0.06, 48.07)
    assert outcome["verdict"] == "STOP"
    assert any("0.06" in reason for reason in outcome["reasons"])
    # the guard threshold was met; the new-class threshold is what stopped it
    assert not any("known_mAP50" in reason for reason in outcome["reasons"])
