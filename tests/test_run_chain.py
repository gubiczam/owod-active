"""The GPU chain's control flow, exercised without a GPU.

The failure this file exists for: ``run_chain`` sampled candidate images and
handed them straight to the detector, and nothing had put the JPEGs on disk. The
detector reads them off the filesystem, so it died on the first one — nine
minutes into a session, after the annotations had been extracted and the CUDA
extension built.

No amount of testing ``owl`` in isolation catches that, because every individual
piece was correct. What was missing was a *step*. So this file stands a fake
bridge in for PROB, runs the whole chain against it, and asserts on the order
and the arguments of the calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from owl import protocol, runner


class FakeBridge:
    """Records what PROB would have been asked to do, and fakes plausible output."""

    def __init__(self, feature_dim: int = 32) -> None:
        self.calls: list[dict] = []
        self.feature_dim = feature_dim

    # -- the three verbs -----------------------------------------------------

    def predict(self, image_ids, *, checkpoint, output, n_prev, n_current,
                max_proposals_per_image=50):
        self.calls.append({
            "verb": "predict", "images": list(image_ids),
            "n_prev": n_prev, "n_current": n_current,
            "per_image": max_proposals_per_image,
            "checkpoint": str(checkpoint),
        })
        output = Path(output)
        if output.exists():
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(len(self.calls))
        rows = max(len(image_ids) * 4, 8)
        ids = np.asarray([image_ids[i % len(image_ids)] for i in range(rows)], dtype=object)
        n_known = max(n_current, 1)
        np.savez_compressed(
            output,
            image_ids=ids,
            confidence=generator.random(rows),
            embeddings=generator.normal(size=(rows, self.feature_dim)),
            posterior=generator.random((rows, n_known + 1)),
            predicted_labels=generator.integers(0, n_known, rows),
            boxes=generator.random((rows, 4)) * 0.5 + 0.25,
            objectness=generator.random(rows),
        )
        output.with_suffix(".json").write_text(json.dumps({"proposal_count": rows}))
        return output

    def train(self, labelled_ids, *, previous_checkpoint, output_checkpoint, output_dir,
              n_prev, n_current, test_set, replay_ids=(), supervision_mode="ft",
              epochs=1, learning_rate=2e-4, batch_size=1, eval_every=10**6):
        # PROB reads a validation split during training and its default names a
        # file this protocol never writes, so a caller that omits this is broken.
        assert test_set, "train was not told which test set to build the val loader from"
        assert eval_every > epochs, (
            "PROB would evaluate inside the training loop; evaluation is the "
            "expensive half of this protocol and is a separate call here")
        self.calls.append({
            "verb": "train", "images": list(labelled_ids), "replay": list(replay_ids),
            "n_prev": n_prev, "n_current": n_current, "supervision": supervision_mode,
        })
        output_checkpoint = Path(output_checkpoint)
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_checkpoint.write_bytes(b"fake")
        return output_checkpoint

    def evaluate(self, *, checkpoint, test_set, output, n_prev, n_current, batch_size=4):
        self.calls.append({
            "verb": "evaluate", "n_prev": n_prev, "n_current": n_current,
            "test_set": test_set,
        })
        output = Path(output)
        if output.exists():
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "known_AP50": 40.0, "U_Recall": 20.0, "WI": 0.03, "A_OSE": 1000,
            "previous_known_AP50": 60.0, "current_known_AP50": 5.0,
            "unknown_AP50": 0.5, "per_class_AP50": {},
        }), encoding="utf-8")
        return output

    def cost_report(self):
        return {"total": float(len(self.calls))}

    # -- helpers for the assertions ------------------------------------------

    def verbs(self) -> list[str]:
        return [call["verb"] for call in self.calls]

    def of(self, verb: str) -> list[dict]:
        return [call for call in self.calls if call["verb"] == verb]


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
def index_with_barren_images():
    """Half the pool holds only classes no task in this chain ever declares.

    PROB's fine-tuning split keeps only the classes introduced so far, so those
    images arrive with zero boxes and its collate function fails on them. They
    have to be filtered out before training, and counted.
    """
    declared = [task.new_class for task in protocol.build_chain(4)[1:]]
    future = "toothbrush"       # task 4 of the benchmark; never declared here
    assert future not in declared and future not in protocol.TASK1
    pool = {}
    for i in range(400):
        if i % 2:
            pool[f"img{i:04d}"] = {future: 2}
        else:
            pool[f"img{i:04d}"] = {declared[i % len(declared)]: 1,
                                   protocol.TASK1[i % len(protocol.TASK1)]: 1}
    return pool


@pytest.fixture
def config():
    return runner.CycleConfig(
        n_tasks=4, budget_per_task=20, rounds_per_task=2,
        candidate_images_per_task=40, proposals_per_image=4,
        n_clusters=8, replay_arm="tail_favouring",
    )


def test_the_chain_asks_for_its_images_before_the_detector_runs(tmp_path, index, config):
    """The bug this file was written for."""

    asked: list[list[str]] = []

    def prepare(image_ids):
        asked.append(list(image_ids))
        return image_ids

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=prepare,
    )

    assert len(asked) == 3, "prepare_images must be called once per incremental task"
    for requested, call in zip(asked, fake.of("predict")):
        assert requested == call["images"], "the detector got images nobody fetched"


def test_an_unavailable_image_is_dropped_rather_than_killing_the_run(tmp_path, index, config):
    """COCO occasionally will not serve an image. That must cost one image, not the chain."""

    def prepare(image_ids):
        return list(image_ids)[:-5]        # five could not be fetched

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=prepare,
    )
    for call in fake.of("predict"):
        assert len(call["images"]) == config.candidate_images_per_task - 5


def test_the_chain_refuses_to_predict_on_nothing(tmp_path, index, config):
    fake = FakeBridge()
    with pytest.raises(RuntimeError, match="no usable candidate images"):
        runner.run_chain(
            fake, config, workspace=tmp_path, candidate_index=index,
            start_checkpoint=tmp_path / "t1.pth", test_set="eval",
            chain=protocol.build_chain(4), prepare_images=lambda ids: [],
        )


def test_images_with_no_currently_known_object_are_not_trained_on(
    tmp_path, index_with_barren_images, config
):
    """The loader fails on them rather than skipping them, so we must filter."""

    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index_with_barren_images,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    known_by_task = {t.name: set(t.known_classes) for t in protocol.build_chain(4)[1:]}
    for row, call in zip(results, fake.of("train")):
        known = known_by_task[row.task]
        for image in call["images"] + call["replay"]:
            assert any(name in known for name in index_with_barren_images[image]), (
                f"{image} would arrive at PROB with zero boxes")


def test_the_wasted_half_of_the_budget_is_reported_not_hidden(
    tmp_path, index_with_barren_images, config
):
    """An image the oracle was paid for and PROB cannot use is a measurement."""

    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index_with_barren_images,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    for row in results:
        selection = row.selection_row
        assert selection["images_no_supervision"] > 0, "half the pool is barren"
        assert (
            selection["images_trainable"] + selection["images_no_supervision"]
            == selection["images_opened"]
        )


def test_a_label_paid_for_early_is_used_when_its_class_becomes_declarable(
    tmp_path, config
):
    """The oracle answered; the answer keeps its value even if we cannot use it yet."""

    from dataclasses import replace

    chain = protocol.build_chain(4)
    late = chain[-1].new_class          # declarable only at the last task
    early = chain[1].new_class
    pool = {}
    for i in range(400):
        pool[f"img{i:04d}"] = {late: 2} if i % 2 else {early: 1}

    banked = FakeBridge()
    results = runner.run_chain(
        banked, replace(config, reuse_deferred_labels=True), workspace=tmp_path / "on",
        candidate_index=pool, start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=chain, prepare_images=lambda ids: ids,
    )
    reused = [row.selection_row["images_from_earlier_tasks"] for row in results]
    assert reused[0] == 0, "nothing is banked before the first task"
    assert sum(reused) > 0, "images holding the late class were never picked back up"

    # and with the ledger off, nothing comes back
    discarded = FakeBridge()
    off = runner.run_chain(
        discarded, replace(config, reuse_deferred_labels=False), workspace=tmp_path / "off",
        candidate_index=pool, start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=chain, prepare_images=lambda ids: ids,
    )
    assert all(row.selection_row["images_from_earlier_tasks"] == 0 for row in off)
    trained_with = sum(len(c["images"]) for c in banked.of("train"))
    trained_without = sum(len(c["images"]) for c in discarded.of("train"))
    assert trained_with > trained_without, "banking must add supervision, not just bookkeeping"


def test_no_image_is_trained_on_twice_through_the_ledger(tmp_path, index, config):
    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    seen: set[str] = set()
    for call in fake.of("train"):
        fresh = set(call["images"])
        assert not (fresh & seen), "an image reached training twice as new supervision"
        seen |= fresh


def test_a_task_with_too_little_trainable_content_says_so(tmp_path, config):
    """PROB drops the last partial batch, so too few images trains on nothing."""

    from dataclasses import replace

    barren = {f"img{i:04d}": {"toothbrush": 1} for i in range(400)}
    fake = FakeBridge()
    with pytest.raises(RuntimeError, match="trainable images"):
        runner.run_chain(
            fake, replace(config, batch_size=2), workspace=tmp_path,
            candidate_index=barren, start_checkpoint=tmp_path / "t1.pth",
            test_set="eval", chain=protocol.build_chain(4),
            prepare_images=lambda ids: ids,
        )


def test_the_class_counts_follow_probs_convention_end_to_end(tmp_path, index, config):
    """seen = prev + current, at every call, for every task."""

    fake = FakeBridge()
    chain = protocol.build_chain(4)
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=chain, prepare_images=lambda ids: ids,
    )
    for task, call in zip(chain[1:], fake.of("predict")):
        # the detector still only knows the previous classes
        assert call["n_prev"] + call["n_current"] == task.n_prev
    for task, call in zip(chain[1:], fake.of("train")):
        assert call["n_prev"] + call["n_current"] == task.n_current
        assert call["n_current"] == 1
    for task, call in zip(chain[1:], fake.of("evaluate")):
        assert call["n_prev"] + call["n_current"] == task.n_current


def test_the_calls_happen_in_the_only_order_that_makes_sense(tmp_path, index, config):
    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    assert fake.verbs() == ["predict", "train", "evaluate"] * 3


def test_each_task_trains_from_the_previous_tasks_checkpoint(tmp_path, index, config):
    """A chain that restarts from t1 every task is not a chain."""

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    used = [call["checkpoint"] for call in fake.of("predict")]
    assert used[0].endswith("t1.pth")
    assert len(set(used)) == 3, "every task must predict with its own predecessor"


def test_replay_grows_and_never_holds_a_future_class(tmp_path, index, config):
    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    sizes = [row.replay_row["images"] for row in results]
    assert sizes == sorted(sizes), f"the memory shrank: {sizes}"
    # every replayed image is one an earlier task actually opened
    opened = {image for call in fake.of("train") for image in call["images"]}
    for call in fake.of("train")[1:]:
        assert set(call["replay"]) <= opened


def test_the_labelling_policy_reaches_probs_supervision_flag(tmp_path, index, config):
    """box_only is the only policy that discards previous-task boxes."""

    from dataclasses import replace

    for policy, expected in (("box_only", "train"),
                             ("full_image", "ft"),
                             ("known_plus_selected", "ft")):
        fake = FakeBridge()
        runner.run_chain(
            fake, replace(config, labelling_policy=policy),
            workspace=tmp_path / policy, candidate_index=index,
            start_checkpoint=tmp_path / "t1.pth", test_set="eval",
            chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
        )
        assert {call["supervision"] for call in fake.of("train")} == {expected}, policy


def test_a_resumed_run_refetches_nothing_and_retrains_nothing(tmp_path, index, config):
    """A cut-off Colab session must not pay twice."""

    arguments = {
        "workspace": tmp_path,
        "candidate_index": index,
        "start_checkpoint": tmp_path / "t1.pth",
        "test_set": "eval",
        "chain": protocol.build_chain(4),
    }
    first = FakeBridge()
    runner.run_chain(first, config, prepare_images=lambda ids: ids, **arguments)

    fetches: list[int] = []
    second = FakeBridge()
    runner.run_chain(
        second, config, prepare_images=lambda ids: (fetches.append(1), ids)[1], **arguments
    )
    assert fetches == [], "a cached detector pass must not trigger a download"


def test_old_checkpoints_are_pruned_so_drive_does_not_fill(tmp_path, index, config):
    """478 MB each x nine tasks x three arms fills a free Drive."""

    from dataclasses import replace

    fake = FakeBridge()
    runner.run_chain(
        fake, replace(config, keep_checkpoints=2), workspace=tmp_path,
        candidate_index=index, start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    surviving = sorted(p.name for p in tmp_path.rglob("checkpoint.pth"))
    assert len(surviving) == 2, f"expected two checkpoints, found {surviving}"
    # and the metrics of every task are still there, which is what resume reads
    assert len(list(tmp_path.rglob("metrics.json"))) == 3


def test_keeping_every_checkpoint_is_still_possible(tmp_path, index, config):
    from dataclasses import replace

    fake = FakeBridge()
    runner.run_chain(
        fake, replace(config, keep_checkpoints=0), workspace=tmp_path,
        candidate_index=index, start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    assert len(list(tmp_path.rglob("checkpoint.pth"))) == 3


def test_the_time_budget_stops_the_chain_and_says_what_it_skipped(tmp_path, index, config, capsys):
    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
        time_budget_minutes=1,      # one fake call's worth
    )
    assert len(results) < 3
    assert "Not run:" in capsys.readouterr().out


def test_the_chain_writes_its_rows_as_it_goes(tmp_path, index, config):
    """If the session dies, whatever finished must already be on disk."""

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index,
        start_checkpoint=tmp_path / "t1.pth", test_set="eval",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    written = tmp_path / f"results_{config.arm}.csv"
    assert written.exists()
    assert len(written.read_text(encoding="utf-8").strip().splitlines()) == 4  # header + 3
