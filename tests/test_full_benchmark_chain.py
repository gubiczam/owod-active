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
    for arm in ("random", "admissibility", "entropy", "proposed", "coreset"):
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
