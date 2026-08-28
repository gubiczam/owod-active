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
import zlib
from pathlib import Path

import numpy as np
import pytest

from owl import exemplars, protocol, replay, runner
from owl.evaluation_subset import check_split_name


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
        # Seeded from the *inputs*, the way PROB is: the same checkpoint over the
        # same images returns the same proposals. Seeding from the call counter
        # instead made the fake's output depend on how many calls this bridge had
        # already served, so a resumed session got different proposals from an
        # unbroken one for reasons that exist only in the fake.
        # Not the checkpoint *path*: two workspaces hold the same chain under
        # different directory names, and PROB's output depends on the weights,
        # not on where they are stored. The class counts stand in for the
        # checkpoint, since they advance with it.
        digest = zlib.crc32(
            "|".join([*map(str, image_ids), str(n_prev), str(n_current)]).encode("utf-8")
        )
        generator = np.random.default_rng(digest)
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
        check_split_name(test_set, purpose="test")
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

    def evaluate(self, *, checkpoint, test_set, output, n_prev, n_current,
                 batch_size=4, detections=True):
        check_split_name(test_set, purpose="test")
        output = Path(output)
        if output.exists():
            # a cached artefact is not work; the resume tests read `calls` as
            # "what this session actually paid for"
            return output
        self.calls.append({
            "verb": "evaluate", "n_prev": n_prev, "n_current": n_current,
            "test_set": test_set,
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "known_AP50": 40.0, "U_Recall": 20.0, "WI": 0.03, "A_OSE": 1000,
            "previous_known_AP50": 60.0, "current_known_AP50": 5.0,
            "unknown_AP50": 0.5,
            # the real bridge writes no per_class_AP50; it writes this vector,
            # shaped [mAP, mAP, <80 classes>, unknown]
            "coco_eval_bbox": [30.0, 30.0, *[float(i % 40) for i in range(80)], 0.5],
                }
        output.write_text(json.dumps(payload), encoding="utf-8")

        if detections:
            # the same shape the bridge writes, so the grouped-recall reader is
            # exercised rather than merely imported
            from owl import protocol as _protocol

            artefact = output.with_name(f"{output.stem}_detections.json")
            unknown = _protocol.CLASS_ORDER[n_prev + n_current:][:6]
            truth, found = [], []
            for index, name in enumerate(unknown):
                box = [10.0 * index, 0.0, 10.0 * index + 8.0, 8.0]
                truth.append({"image_id": "000000000000", "class_name": name, "box": box})
                if index % 2 == 0:                      # half of them recalled
                    found.append({"image_id": "000000000000", "class_name": "unknown",
                                  "score": 0.9, "box": box})
            artefact.write_text(json.dumps({
                "schema": "daowod_detections_v1", "unknown_class_name": "unknown",
                "ground_truth": truth, "detections": found,
            }), encoding="utf-8")
            payload["detections_path"] = str(artefact)
            output.write_text(json.dumps(payload), encoding="utf-8")
        return output

    def cost_report(self):
        return {"total": float(len(self.calls))}

    # -- helpers for the assertions ------------------------------------------

    def verbs(self) -> list[str]:
        return [call["verb"] for call in self.calls]

    def of(self, verb: str) -> list[dict]:
        return [call for call in self.calls if call["verb"] == verb]


#: The old-data pool: what existed before the chain started, which for this
#: protocol is the split PROB's ``t1.pth`` was trained on. Disjoint from the
#: candidate pool by construction, so a replay object can never be mistaken for
#: something an arm bought. The ids are twelve digits because
#: ``OWDetection.convert_image_id`` casts an image id to ``int`` — and none of
#: them starts with nine, which is what ``owl.exemplars.alias_id`` reserves.
OLD_DATA = {
    f"{500000 + i:012d}": {protocol.TASK1[i % len(protocol.TASK1)]: 1 + (i % 4)}
    for i in range(600)
}


def prob_data_root(tmp_path, *indices):
    """A PROB-shaped ``data_root``: one VOC annotation and one JPEG per image.

    The exemplar memory is materialised by filtering real annotations, so the
    tests need real ones. Objects are written grouped by class name in sorted
    order, which is the document order ``owl.exemplars`` counts ordinals in.
    """

    root = tmp_path / "prob_data"
    annotations, images = root / "Annotations", root / "JPEGImages"
    annotations.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    for index in indices:
        for image_id, counts in index.items():
            target = annotations / f"{image_id}.xml"
            if target.exists():
                continue
            boxes, offset = [], 0
            for name in sorted(counts):
                for _ in range(int(counts[name])):
                    offset += 10
                    boxes.append(
                        f"<object><name>{name}</name><difficult>0</difficult>"
                        f"<bndbox><xmin>{offset}</xmin><ymin>{offset}</ymin>"
                        f"<xmax>{offset + 8}</xmax><ymax>{offset + 8}</ymax>"
                        "</bndbox></object>"
                    )
            target.write_text(
                "<annotation><folder>OWOD</folder>"
                f"<filename>{image_id}.jpg</filename>"
                "<size><width>640</width><height>480</height><depth>3</depth></size>"
                "<segmented>0</segmented>" + "".join(boxes) + "</annotation>",
                encoding="utf-8",
            )
            (images / f"{image_id}.jpg").write_bytes(b"not-a-real-jpeg")
    return root


def replay_sources(train_call) -> set[str]:
    """The physical images behind a training call's replay ids.

    PROB is handed *aliases* — a second id per source image whose annotation
    holds only the selected exemplar boxes. Every assertion about which images
    were rehearsed has to resolve them back.
    """

    return {exemplars.source_id(alias) for alias in train_call["replay"]}


@pytest.fixture
def index():
    """A candidate index: 400 images, each holding one of the declared classes."""
    declared = [task.new_class for task in protocol.build_chain(4)[1:]]
    return {
        f"{i:012d}": {declared[i % len(declared)]: 1 + (i % 3),
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
            pool[f"{i:012d}"] = {future: 2}
        else:
            pool[f"{i:012d}"] = {declared[i % len(declared)]: 1,
                                   protocol.TASK1[i % len(protocol.TASK1)]: 1}
    return pool


@pytest.fixture
def config():
    return runner.CycleConfig(
        n_tasks=4, budget_per_task=20, rounds_per_task=2,
        candidate_images_per_task=40, proposals_per_image=4,
        n_clusters=8, replay_arm="tail_favouring",
    )


def test_the_chain_asks_for_its_images_before_the_detector_and_before_training(
    tmp_path, index, config
):
    """Twice per task, and both calls matter.

    The first covers the candidate pool the detector is about to read. The second
    covers what training and replay are about to read, which is a different and
    smaller set — and which a resumed run needs even when the detector pass is
    cached, because Drive keeps the proposals and /content keeps nothing.
    """

    asked: list[list[str]] = []

    def prepare(image_ids):
        asked.append([str(i) for i in image_ids])
        return image_ids

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=prepare,
    )

    assert len(asked) == 6, "two calls per incremental task"

    offered: set[str] = set()
    for index_of_task, call in enumerate(fake.of("predict")):
        candidates, training = asked[2 * index_of_task], asked[2 * index_of_task + 1]
        assert candidates == call["images"], "the detector got images nobody fetched"

        offered |= set(candidates)
        # Training reads the selected images plus the replay memory, and the
        # memory is drawn from the old-data pool and from earlier tasks — so it
        # is not a subset of this task's pool, and at a later task it can even be
        # larger than it. The invariant that does hold: nothing is read that was
        # never offered, by either pool.
        assert set(training) <= offered | set(OLD_DATA), (
            f"task {index_of_task + 2} would read images from no pool: "
            f"{sorted(set(training) - offered)[:3]}"
        )
        assert training, "training asked for nothing"


def test_an_unavailable_image_is_dropped_rather_than_killing_the_run(tmp_path, index, config):
    """COCO occasionally will not serve an image. That must cost one image, not the chain."""

    def prepare(image_ids):
        # the low ids are candidate images; the memory's sources are the high
        # ones, and a memory that cannot be put on disk is a separate failure
        # with its own test below
        return list(image_ids)[5:]         # five could not be fetched

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=prepare,
    )
    for call in fake.of("predict"):
        assert len(call["images"]) == config.candidate_images_per_task - 5


def test_the_chain_refuses_to_predict_on_nothing(tmp_path, index, config):
    fake = FakeBridge()
    with pytest.raises(RuntimeError, match="no usable candidate images"):
        runner.run_chain(
            fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
            chain=protocol.build_chain(4), prepare_images=lambda ids: [],
        )


def test_images_with_no_currently_known_object_are_not_trained_on(
    tmp_path, index_with_barren_images, config
):
    """The loader fails on them rather than skipping them, so we must filter."""

    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index_with_barren_images, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index_with_barren_images, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    known_by_task = {t.name: set(t.known_classes) for t in protocol.build_chain(4)[1:]}
    for row, call in zip(results, fake.of("train")):
        known = known_by_task[row.task]
        physical = list(call["images"]) + sorted(replay_sources(call))
        for image in physical:
            counts = index_with_barren_images.get(image) or OLD_DATA[image]
            assert any(name in known for name in counts), (
                f"{image} would arrive at PROB with zero boxes")


def test_the_wasted_half_of_the_budget_is_reported_not_hidden(
    tmp_path, index_with_barren_images, config
):
    """An image the oracle was paid for and PROB cannot use is a measurement."""

    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index_with_barren_images, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index_with_barren_images, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
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
        pool[f"{i:012d}"] = {late: 2} if i % 2 else {early: 1}

    banked = FakeBridge()
    results = runner.run_chain(
        banked, replace(config, reuse_deferred_labels=True), workspace=tmp_path / "on",
        candidate_index=pool, start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test", replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, pool, OLD_DATA),
        chain=chain, prepare_images=lambda ids: ids,
    )
    reused = [row.selection_row["images_from_earlier_tasks"] for row in results]
    assert reused[0] == 0, "nothing is banked before the first task"
    assert sum(reused) > 0, "images holding the late class were never picked back up"

    # and with the ledger off, nothing comes back
    discarded = FakeBridge()
    off = runner.run_chain(
        discarded, replace(config, reuse_deferred_labels=False), workspace=tmp_path / "off",
        candidate_index=pool, start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test", replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, pool, OLD_DATA),
        chain=chain, prepare_images=lambda ids: ids,
    )
    assert all(row.selection_row["images_from_earlier_tasks"] == 0 for row in off)
    trained_with = sum(len(c["images"]) for c in banked.of("train"))
    trained_without = sum(len(c["images"]) for c in discarded.of("train"))
    assert trained_with > trained_without, "banking must add supervision, not just bookkeeping"


def test_no_image_is_trained_on_twice_through_the_ledger(tmp_path, index, config):
    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
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

    barren = {f"{i:012d}": {"toothbrush": 1} for i in range(400)}
    fake = FakeBridge()
    with pytest.raises(RuntimeError, match="trainable images"):
        runner.run_chain(
            fake, replace(config, batch_size=2), workspace=tmp_path,
            candidate_index=barren, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, barren, OLD_DATA),
            start_checkpoint=tmp_path / "t1.pth",
            test_set="owl_shared_test", chain=protocol.build_chain(4),
            prepare_images=lambda ids: ids,
        )


def test_the_class_counts_follow_probs_convention_end_to_end(tmp_path, index, config):
    """seen = prev + current, at every call, for every task."""

    fake = FakeBridge()
    chain = protocol.build_chain(4)
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=chain, prepare_images=lambda ids: ids,
    )
    for task, call in zip(chain[1:], fake.of("predict")):
        # the detector still only knows the previous classes
        assert call["n_prev"] + call["n_current"] == task.n_prev
    for task, call in zip(chain[1:], fake.of("train")):
        assert call["n_prev"] + call["n_current"] == task.n_current
        assert call["n_current"] == 1
    # the first evaluate is the anchor: the starting checkpoint on its own 19
    # classes, which is what task 2 measures its forgetting against
    anchor, *per_task = fake.of("evaluate")
    assert anchor["n_prev"] + anchor["n_current"] == chain[0].n_current
    for task, call in zip(chain[1:], per_task):
        assert call["n_prev"] + call["n_current"] == task.n_current


def test_the_calls_happen_in_the_only_order_that_makes_sense(tmp_path, index, config):
    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    # the anchor is scored once, before anything is trained; then the cycle
    assert fake.verbs() == ["evaluate"] + ["predict", "train", "evaluate"] * 3


def test_each_task_trains_from_the_previous_tasks_checkpoint(tmp_path, index, config):
    """A chain that restarts from t1 every task is not a chain."""

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    used = [call["checkpoint"] for call in fake.of("predict")]
    assert used[0].endswith("t1.pth")
    assert len(set(used)) == 3, "every task must predict with its own predecessor"


def test_a_task_never_rehearses_on_its_own_fresh_data(tmp_path, index, config):
    """The invariant the whole replay experiment rests on.

    Replay is rehearsal of what was known *before* a step. The previous version
    put the task's own freshly labelled images into the pool and then drew the
    memory from it, so at task 2 the memory was a subset of the images being
    trained on — a 100% intersection — and every later task rehearsed a third of
    its own acquisition. An arm measured that way is not being compared on how
    well it retains old knowledge.
    """

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(5), prepare_images=lambda ids: ids,
    )

    trains = fake.of("train")
    assert len(trains) >= 3, "the chain has to run far enough to have a history"

    for step, call in enumerate(trains, start=2):
        current, replayed = set(call["images"]), replay_sources(call)
        assert not (replayed & current), (
            f"t{step} rehearsed on {len(replayed & current)} of the images it was "
            "learning from in the same step"
        )
        # and the aliases are never the physical ids, so nothing can be handed to
        # PROB twice under two names
        assert not (set(call["replay"]) & current)

    # t2 has no earlier task to draw on, so its memory must come from the
    # declared old-data pool and from nowhere else
    assert trains[0]["replay"], "t2 got no rehearsal at all"
    assert replay_sources(trains[0]) <= set(OLD_DATA), (
        "t2 rehearsed on something that is not old data"
    )


def test_an_image_the_selector_just_bought_is_not_also_rehearsed(tmp_path, config):
    """The two pools overlap in reality, so the invariant must not rest on luck.

    Measured on the committed indices: 1,800 of the 12,000 old-data images are
    also in the candidate pool. When the selector buys one of those, nothing
    stops the memory from drawing it as well, and PROB would receive the same
    image twice in one step. Here the candidate pool is *entirely* inside the
    old-data pool, so the collision is certain rather than occasional.
    """

    declared = [task.new_class for task in protocol.build_chain(4)[1:]]
    shared = {
        f"{i:012d}": {declared[i % len(declared)]: 1 + (i % 3),
                        protocol.TASK1[i % len(protocol.TASK1)]: 2}
        for i in range(400)
    }

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=shared,
        replay_index=shared,                      # the pools are the same pool
        replay_root=prob_data_root(tmp_path, shared),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )

    for step, call in enumerate(fake.of("train"), start=2):
        overlap = set(call["images"]) & replay_sources(call)
        assert not overlap, (
            f"t{step} was handed {len(overlap)} image(s) as both new supervision "
            f"and rehearsal: {sorted(overlap)[:3]}"
        )
        assert call["replay"], f"t{step} got no rehearsal at all"


def test_what_a_task_learned_becomes_rehearsable_only_afterwards(
    tmp_path, index, config
):
    """The other half of the temporal rule, and the one a weak test would miss.

    Excluding the current task is not enough on its own: a memory that only ever
    held the task-1 pool would also pass that check while quietly never
    rehearsing anything the chain itself acquired. What t2 learned has to be
    eligible from t3 onward.
    """

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(5), prepare_images=lambda ids: ids,
    )

    trains = fake.of("train")
    learned_at_t2 = set(trains[0]["images"])
    eligible_later = set()
    for call in trains[1:]:
        eligible_later |= replay_sources(call)

    assert learned_at_t2 & eligible_later, (
        "nothing t2 learned ever became rehearsable; the memory is frozen on the "
        "old-data pool instead of following the chain"
    )


def test_the_memory_holds_its_object_budget_instead_of_growing(
    tmp_path, index, config
):
    """M is an object budget (docs/method.md, step 5), and it must not drift.

    The earlier implementation unioned each task's memory with the last one, so
    the memory grew every task and by a different amount per arm — which meant
    two alpha values differed in how much rehearsal they received as well as in
    how it was distributed, and no comparison between them meant anything.
    """

    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(5), prepare_images=lambda ids: ids,
    )

    budget = replay.ARMS[config.replay_arm]["total"]
    allocated = [row.replay_row["delivered_objects"] for row in results]
    assert allocated == [budget] * len(results), (
        f"the object budget drifted across tasks: {allocated}, expected {budget}"
    )


def test_two_alpha_arms_differ_in_composition_not_in_size(tmp_path, index, config):
    """The comparison is only valid if the total is held equal."""

    from dataclasses import replace

    composition = {}
    for arm in ("head_favouring", "tail_favouring"):
        fake = FakeBridge()
        results = runner.run_chain(
            fake, replace(config, replay_arm=arm), workspace=tmp_path / arm,
            candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
            chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
        )
        composition[arm] = [row.replay_row for row in results]

    budget = replay.ARMS["head_favouring"]["total"]
    for arm, rows in composition.items():
        assert [r["delivered_objects"] for r in rows] == [budget] * len(rows), arm

    head = composition["head_favouring"][0]
    tail = composition["tail_favouring"][0]
    assert head["delivered_objects"] == tail["delivered_objects"]
    assert head["per_class"] != tail["per_class"], (
        "the two allocation rules produced an identical class composition")


def test_a_replay_arm_without_an_old_data_pool_is_refused(tmp_path, index, config):
    """The candidate pool is not a stand-in for old data, and must not become one."""

    with pytest.raises(ValueError, match="replay_index"):
        runner.run_chain(
            FakeBridge(), config, workspace=tmp_path, candidate_index=index,
            replay_root=prob_data_root(tmp_path, index, OLD_DATA),
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
            chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
        )


def test_a_replay_arm_without_a_place_to_write_aliases_is_refused(
    tmp_path, index, config
):
    """An object budget is only real if the filtered annotations get written."""

    with pytest.raises(ValueError, match="replay_root"):
        runner.run_chain(
            FakeBridge(), config, workspace=tmp_path, candidate_index=index,
            replay_index=OLD_DATA,
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
            chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
        )


def test_a_memory_that_cannot_be_put_on_disk_stops_the_run(tmp_path, index, config):
    """Silently rehearsing on 340 of 400 objects would break the comparison."""

    def prepare(image_ids):
        # the memory's source images are the high ids and never arrive
        return [i for i in image_ids if not str(i).startswith("00000050")]

    with pytest.raises(RuntimeError, match="exemplar objects on disk"):
        runner.run_chain(
            FakeBridge(), config, workspace=tmp_path, candidate_index=index,
            replay_index=OLD_DATA,
            replay_root=prob_data_root(tmp_path, index, OLD_DATA),
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
            chain=protocol.build_chain(4), prepare_images=prepare,
        )


def test_running_without_replay_needs_no_old_data_pool(tmp_path, index, config):
    from dataclasses import replace

    results = runner.run_chain(
        FakeBridge(), replace(config, replay_arm="none"), workspace=tmp_path,
        candidate_index=index, start_checkpoint=tmp_path / "t1.pth",
        test_set="owl_shared_test", chain=protocol.build_chain(4),
        prepare_images=lambda ids: ids,
    )
    assert all(row.replay_row["images"] == 0 for row in results)


def test_the_labelling_policy_reaches_probs_supervision_flag(tmp_path, index, config):
    """box_only is the only policy that discards previous-task boxes."""

    from dataclasses import replace

    for policy, expected in (("box_only", "train"),
                             ("full_image", "ft"),
                             ("known_plus_selected", "ft")):
        fake = FakeBridge()
        runner.run_chain(
            fake, replace(config, labelling_policy=policy),
            workspace=tmp_path / policy, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
            chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
        )
        assert {call["supervision"] for call in fake.of("train")} == {expected}, policy


def test_a_resumed_run_refetches_nothing_and_retrains_nothing(tmp_path, index, config):
    """A cut-off Colab session must not pay twice."""

    arguments = {
        "workspace": tmp_path,
        "candidate_index": index,
        "replay_index": OLD_DATA,
        "replay_root": prob_data_root(tmp_path, index, OLD_DATA),
        "start_checkpoint": tmp_path / "t1.pth",
        "test_set": "owl_shared_test",
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


def test_a_resumed_chain_rehearses_on_exactly_what_an_unbroken_one_did(
    tmp_path, index, config
):
    """Replay state has to survive a lost Colab runtime, or arms are not comparable.

    The memory is rebuilt from the eligible pool at the top of every task rather
    than carried in a variable, so the thing that has to be restored is *which
    tasks have finished*. If that were restored wrongly the resumed half of a
    chain would rehearse on a different memory from the first half, inside one
    results table.
    """

    arguments = {
        "candidate_index": index,
        "replay_index": OLD_DATA,
        "replay_root": prob_data_root(tmp_path, index, OLD_DATA),
        "start_checkpoint": tmp_path / "t1.pth",
        "test_set": "owl_shared_test",
        "chain": protocol.build_chain(5),
        "prepare_images": lambda ids: ids,
    }

    unbroken = FakeBridge()
    runner.run_chain(unbroken, config, workspace=tmp_path / "whole", **arguments)

    # the same chain, cut off after one task and continued in a new session
    broken = FakeBridge()
    runner.run_chain(broken, config, workspace=tmp_path / "cut",
                     time_budget_minutes=1, **arguments)
    resumed = FakeBridge()
    runner.run_chain(resumed, config, workspace=tmp_path / "cut", **arguments)

    straight = [sorted(call["replay"]) for call in unbroken.of("train")]
    interrupted = [sorted(call["replay"])
                   for call in broken.of("train") + resumed.of("train")]

    assert len(interrupted) == len(straight) > 2, (
        f"the interrupted run did not cover the chain: {len(interrupted)} vs "
        f"{len(straight)}"
    )
    assert interrupted == straight, "a resumed chain rehearsed on a different memory"


def test_every_replay_arm_delivers_the_same_number_of_objects(tmp_path, index, config):
    """A: the claim Replay Protocol V3 exists to support.

    ``requested == allocated == delivered == M`` for every arm and every task.
    Under V2 the same three arms delivered 464, 768 and 1,240 objects at t6 for a
    budget of 400, because the memory was stored as images and PROB reads whole
    images.
    """

    from dataclasses import replace

    budget = replay.ARMS["uniform"]["total"]
    per_arm = {}
    for arm in ("head_favouring", "uniform", "tail_favouring"):
        fake = FakeBridge()
        results = runner.run_chain(
            fake, replace(config, replay_arm=arm, n_tasks=6),
            workspace=tmp_path / arm, candidate_index=index, replay_index=OLD_DATA,
            replay_root=prob_data_root(tmp_path, index, OLD_DATA),
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
            chain=protocol.build_chain(6), prepare_images=lambda ids: ids,
        )
        assert len(results) == 5, arm
        per_arm[arm] = results
        for row in results:
            diagnostics = row.replay_row
            assert diagnostics["requested_objects"] == budget, (arm, row.task)
            assert diagnostics["allocated_objects"] == budget, (arm, row.task)
            assert diagnostics["delivered_objects"] == budget, (arm, row.task)
            # B: the per-class breakdown sums to the budget and is the allocation
            counts = dict(
                pair.split(":") for pair in diagnostics["per_class"].split(";")
            )
            assert sum(int(v) for v in counts.values()) == budget, (arm, row.task)
            # one alias per source image, so nothing was handed over twice
            assert diagnostics["images"] == diagnostics["unique_source_images"]

    # and the arms really are different memories, not the same one relabelled
    compositions = {
        arm: [row.replay_row["per_class"] for row in rows]
        for arm, rows in per_arm.items()
    }
    assert compositions["head_favouring"] != compositions["tail_favouring"]


def test_a_discarded_exemplar_cannot_come_back_from_the_canonical_pool(tmp_path, config):
    """F: the memory is bounded by E_(k-1) and the immediately previous task.

    The pool here is built so that resurrection would be *attractive*: the
    old-data index is full of a rare class the candidate images never contain, so
    from t3 onward the allocator would happily draw more of it — and may only use
    the ones the memory itself kept. If the canonical pool were reopened, t3
    would hold rare-class objects that t2 had discarded.
    """

    from dataclasses import replace

    rare = "bear"
    declared = [task.new_class for task in protocol.build_chain(5)[1:]]
    assert rare not in declared

    # the candidate pool: the declared classes plus one common previous class,
    # and never the rare one
    candidates = {
        f"{i:012d}": {declared[i % len(declared)]: 1, "person": 2}
        for i in range(400)
    }
    # the old-data pool: mostly the rare class, on images the selector cannot buy
    old = {
        f"{700000 + i:012d}": ({rare: 2} if i % 2 else {rare: 1, "person": 1})
        for i in range(300)
    }

    fake = FakeBridge()
    runner.run_chain(
        fake, replace(config, replay_arm="tail_favouring", n_tasks=5),
        workspace=tmp_path, candidate_index=candidates, replay_index=old,
        replay_root=prob_data_root(tmp_path, candidates, old),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(5), prepare_images=lambda ids: ids,
    )

    stored = []
    for task in ("t2", "t3", "t4", "t5"):
        state = json.loads(
            (tmp_path / f"{task}_{config.arm}" / "state.json").read_text()
        )
        stored.append((
            task,
            {tuple(row) for row in state["exemplars"]},
            set(state["previous_task_images"]),
        ))

    assert stored[0][1], "t2 stored no exemplars"
    for index in range(1, len(stored)):
        task, current, _ = stored[index]
        _, previous_memory, previous_labelled = stored[index - 1]
        outside = {
            item for item in current
            if item not in previous_memory and item[0] not in previous_labelled
        }
        assert not outside, (
            f"{task} holds {len(outside)} exemplar(s) that were neither in the "
            f"previous memory nor acquired at the previous task: "
            f"{sorted(outside)[:3]}"
        )

    # and the rare class really was discarded between tasks, so the test had
    # something to catch rather than passing vacuously
    evicted = [
        json.loads((tmp_path / f"{task}_{config.arm}" / "state.json").read_text())
        for task in ("t3", "t4", "t5")
    ]
    assert any(state["replay_row"]["evicted"] > 0 for state in evicted), (
        "nothing was ever evicted, so resurrection was never possible anyway"
    )


def test_a_resumed_chain_stores_bit_identical_exemplars(tmp_path, index, config):
    """I: object identities, aliases and diagnostics all survive a lost runtime."""

    arguments = {
        "candidate_index": index,
        "replay_index": OLD_DATA,
        "replay_root": prob_data_root(tmp_path, index, OLD_DATA),
        "start_checkpoint": tmp_path / "t1.pth",
        "test_set": "owl_shared_test",
        "chain": protocol.build_chain(5),
        "prepare_images": lambda ids: ids,
    }

    whole = FakeBridge()
    runner.run_chain(whole, config, workspace=tmp_path / "whole", **arguments)

    cut = FakeBridge()
    runner.run_chain(cut, config, workspace=tmp_path / "cut",
                     time_budget_minutes=1, **arguments)
    resumed = FakeBridge()
    runner.run_chain(resumed, config, workspace=tmp_path / "cut", **arguments)

    def state_of(root, task):
        return json.loads((root / f"{task}_{config.arm}" / "state.json").read_text())

    for task in ("t2", "t3", "t4", "t5"):
        straight = state_of(tmp_path / "whole", task)
        interrupted = state_of(tmp_path / "cut", task)
        assert straight["exemplars"] == interrupted["exemplars"], task
        assert straight["previous_task_images"] == interrupted["previous_task_images"], task
        assert straight["replay_row"] == interrupted["replay_row"], task

    straight_aliases = [sorted(call["replay"]) for call in whole.of("train")]
    interrupted_aliases = [sorted(call["replay"])
                           for call in cut.of("train") + resumed.of("train")]
    assert interrupted_aliases == straight_aliases


def test_an_old_workspace_cannot_resume_under_the_new_replay_meaning(
    tmp_path, index, config
):
    """Version 1 built the memory after the fact; version 2 builds it before.

    The arm name is identical in both, so nothing in the stored fingerprint would
    otherwise stop a version-1 workspace from being continued as version 2 and
    the two halves ending up in one table.
    """

    stored = config.fingerprint()
    del stored["replay_protocol_version"]          # a workspace from before the field
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(RuntimeError, match="replay_protocol_version"):
        runner.run_chain(
            FakeBridge(), config, workspace=tmp_path, candidate_index=index,
            replay_index=OLD_DATA, start_checkpoint=tmp_path / "t1.pth",
            replay_root=prob_data_root(tmp_path, index, OLD_DATA),
            test_set="owl_shared_test", chain=protocol.build_chain(4),
            prepare_images=lambda ids: ids,
        )


def test_the_first_incremental_task_reports_forgetting_against_the_anchor(
    tmp_path, index, config
):
    """t2 moves the weights furthest, and used to be the one task with no number.

    Forgetting is measured against the previous task's known mAP50, and task 2
    has no previous task inside the chain — so without scoring the starting
    checkpoint the column was empty exactly where it matters most.
    """

    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index,
        replay_index=OLD_DATA, start_checkpoint=tmp_path / "t1.pth",
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        test_set="owl_shared_test", chain=protocol.build_chain(4),
        prepare_images=lambda ids: ids,
    )

    first = results[0].flat()
    assert first["task"] == "t2"
    assert first["forgetting"] is not None, "t2 still has no forgetting baseline"
    assert first["drop_from_anchor"] is not None

    # and the anchor is scored on the starting checkpoint, before any training
    assert (tmp_path / "anchor_metrics.json").exists()
    assert fake.of("evaluate")[0]["n_prev"] + fake.of("evaluate")[0]["n_current"] == 19


def test_old_checkpoints_are_pruned_so_drive_does_not_fill(tmp_path, index, config):
    """478 MB each x nine tasks x three arms fills a free Drive."""

    from dataclasses import replace

    fake = FakeBridge()
    runner.run_chain(
        fake, replace(config, keep_checkpoints=2), workspace=tmp_path,
        candidate_index=index, start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test", replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
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
        candidate_index=index, start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test", replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    assert len(list(tmp_path.rglob("checkpoint.pth"))) == 3


def test_the_time_budget_stops_the_chain_and_says_what_it_skipped(tmp_path, index, config, capsys):
    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
        time_budget_minutes=1,      # one fake call's worth
    )
    assert len(results) < 3
    assert "Not run:" in capsys.readouterr().out


def test_a_second_arm_gets_the_whole_time_budget_it_was_handed(
    tmp_path, index, config, capsys
):
    """The notebook drives every arm through ONE bridge, and that is the trap.

    ``Bridge.cost_report()['total']`` is cumulative over everything that bridge
    object has ever run. The notebook loop hands each arm the budget that is
    *left* — ``TIME_BUDGET_MINUTES - spent``. So comparing the running total
    against the remaining budget compares two different clocks, and the second
    arm stops after one task while there is still budget for four.

    That is exactly the signature of a workspace where the first arm finished
    and the rest hold one task each. So this drives two arms through one bridge
    the way the notebook does, and asserts the second one is not cut short.
    """

    from dataclasses import replace

    fake = FakeBridge()
    chain = protocol.build_chain(4)          # three incremental tasks
    per_task = 3                             # predict + train + evaluate
    budget = float((len(chain) - 1) * per_task * 2)   # room for both arms in full

    finished = {}
    spent = 0.0
    for arm in ("prior_consult_batch", "random"):
        finished[arm] = len(runner.run_chain(
            fake, replace(config, arm=arm), workspace=tmp_path / arm,
            candidate_index=index, start_checkpoint=tmp_path / "t1.pth", replay_index=OLD_DATA,
            replay_root=prob_data_root(tmp_path, index, OLD_DATA),
            test_set="owl_shared_test", chain=chain,
            time_budget_minutes=budget - spent, prepare_images=lambda ids: ids,
        ))
        spent = fake.cost_report()["total"]

    assert finished["prior_consult_batch"] == len(chain) - 1
    assert finished["random"] == len(chain) - 1, (
        "the second arm stopped early: run_chain measured the bridge's lifetime "
        f"total against this call's own budget. Finished: {finished}"
    )
    assert "Not run:" not in capsys.readouterr().out


def test_the_frequency_split_reaches_the_reported_row(tmp_path, index, config):
    """head/medium/tail is the plan's distinguishing evaluation; it must be in the table."""

    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    for row in results:
        flat = row.flat()
        assert any(key.startswith("mAP50_") for key in flat), flat.keys()


def test_the_chain_writes_its_rows_as_it_goes(tmp_path, index, config):
    """If the session dies, whatever finished must already be on disk."""

    fake = FakeBridge()
    runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    written = tmp_path / f"results_{config.arm}.csv"
    assert written.exists()
    assert len(written.read_text(encoding="utf-8").strip().splitlines()) == 4  # header + 3


def test_a_workspace_from_a_different_configuration_is_refused(tmp_path, index, config):
    """The bug that produced an unusable results table.

    Resuming is keyed on output files existing. A smoke run and a real run shared
    a workspace, so the real run silently reused the smoke run's checkpoints and
    metrics for the tasks it had reached. Two rows were then measured on a
    sixteen-image evaluation split and three on a fourteen-hundred-image one, and
    the difference read as a twenty-nine-point swing in forgetting.
    """

    from dataclasses import replace

    smoke = replace(config, budget_per_task=10, epochs=1)
    runner.run_chain(
        FakeBridge(), smoke, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    with pytest.raises(RuntimeError, match="different configuration") as caught:
        runner.run_chain(
            FakeBridge(), config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
            chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
        )
    message = str(caught.value)
    assert "budget_per_task" in message and "epochs" in message
    assert "rm -rf" in message, "the message must say how to proceed"


def test_the_same_configuration_still_resumes(tmp_path, index, config):
    first = FakeBridge()
    runner.run_chain(
        first, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    second = FakeBridge()
    runner.run_chain(
        second, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    assert not second.of("train"), "a completed chain must not retrain"


def test_bookkeeping_only_changes_do_not_block_a_resume(tmp_path, index, config):
    """keep_checkpoints does not change any number, so it must not refuse."""

    from dataclasses import replace

    runner.run_chain(
        FakeBridge(), config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    runner.run_chain(
        FakeBridge(), replace(config, keep_checkpoints=0), workspace=tmp_path,
        candidate_index=index, start_checkpoint=tmp_path / "t1.pth", replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        test_set="owl_shared_test", chain=protocol.build_chain(4),
        prepare_images=lambda ids: ids,
    )


def test_a_resumed_chain_restores_what_it_had_already_bought(tmp_path, index, config):
    """Resuming without the accumulated state is not resuming, it is restarting.

    ``used_images`` stops a later task re-buying images the oracle already
    answered for, and the ledger and the memory are what the replay allocation is
    built from. A chain that forgets them selects duplicates and reallocates a
    memory it had already paid for, and nothing about the output says so.
    """

    from dataclasses import replace

    stop_early = replace(config, n_tasks=4)
    first = FakeBridge()
    partial = runner.run_chain(
        first, stop_early, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
        time_budget_minutes=4,      # two tasks' worth of fake calls
    )
    assert 0 < len(partial) < 3, f"expected a partial chain, got {len(partial)}"
    bought = {image for row in partial for image in first.of("train")[0]["images"]}

    second = FakeBridge()
    complete = runner.run_chain(
        second, stop_early, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    assert len(complete) == 3, "the resumed run must finish the chain"

    # the restored rows are the originals, not recomputed ones
    for before, after in zip(partial, complete):
        assert before.flat() == after.flat(), f"{after.task} changed on resume"

    # and no image the first run trained on is bought again by the second
    fresh = {image for call in second.of("train") for image in call["images"]}
    assert not (fresh & bought), "a resumed task re-bought images already paid for"


def test_resuming_does_not_retrain_a_task_whose_checkpoint_was_pruned(tmp_path, index, config):
    """Checkpoints are pruned to save Drive; completion is keyed on metrics."""

    from dataclasses import replace

    tight = replace(config, keep_checkpoints=1)
    runner.run_chain(
        FakeBridge(), tight, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    assert len(list(tmp_path.rglob("checkpoint.pth"))) == 1

    again = FakeBridge()
    runner.run_chain(
        again, tight, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    assert not again.calls, f"a finished chain did work again: {again.verbs()}"


def test_the_plans_headline_endpoint_reaches_the_table(tmp_path, index, config):
    """U-Recall by frequency group, against oracle cost.

    The plan asks for "csoportonkénti mAP és U-Recall ... valamint a tail-U-Recall
    ... mint az orákulum-költség függvénye", and predicts that distribution-aware
    selection reaches the same tail level from fewer annotations. The aggregate
    U-Recall the evaluator prints averages over every unknown class at once and
    therefore cannot show that at all.
    """

    fake = FakeBridge()
    results = runner.run_chain(
        fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=lambda ids: ids,
    )
    for row in results:
        flat = row.flat()
        for column in ("U_Recall_head", "U_Recall_medium", "U_Recall_tail",
                       "U_Recall_all", "oracle_cost_so_far"):
            assert column in flat, f"{row.task} is missing {column}"
        assert flat["unknown_objects_all"] > 0

    costs = [row.flat()["oracle_cost_so_far"] for row in results]
    assert costs == sorted(costs) and costs[0] == config.budget_per_task, costs


def test_turning_the_grouped_recall_off_skips_the_second_forward_pass(tmp_path, index, config):
    """It is the expensive half of evaluation, so it must be optional and visible."""

    from dataclasses import replace

    fake = FakeBridge()
    results = runner.run_chain(
        fake, replace(config, measure_grouped_recall=False), workspace=tmp_path,
        candidate_index=index, start_checkpoint=tmp_path / "t1.pth", replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        test_set="owl_shared_test", chain=protocol.build_chain(4),
        prepare_images=lambda ids: ids,
    )
    assert not any("_detections.json" in p.name for p in tmp_path.rglob("*.json"))
    assert "U_Recall_tail" not in results[0].flat()


def test_a_cached_detector_pass_does_not_imply_the_images_are_still_there(
    tmp_path, index, config
):
    """The bug that killed a live GPU session, twice removed from its cause.

    Drive persists between Colab sessions; ``/content`` does not. So a resumed
    run finds ``proposals.npz`` on Drive, skips the detector pass — and then
    trains on JPEGs that were downloaded into a ``/content`` that no longer
    exists. The failure surfaces as ``FileNotFoundError`` inside a DataLoader
    worker, long after the cause.

    Gating the download on "is the detector pass cached" was the mistake: what
    reads the images is the training, not the caching.
    """

    class DemandsImages(FakeBridge):
        """A bridge that fails the way PROB fails when a JPEG is missing."""

        def __init__(self, present: set[str]):
            super().__init__()
            self.present = present

        def train(self, labelled_ids, **kwargs):
            # PROB reads JPEGImages/<id>.jpg for every id in the split. The
            # labelled ids were fetched; the replay ids are aliases, which exist
            # only as links onto a fetched source.
            missing = [i for i in labelled_ids if i not in self.present]
            missing += [i for i in kwargs.get("replay_ids", ())
                        if exemplars.source_id(i) not in self.present]
            if missing:
                raise FileNotFoundError(
                    f"No such file or directory: JPEGImages/{missing[0]}.jpg"
                )
            return super().train(labelled_ids, **kwargs)

    # first session: everything is fetched, and the run is cut short
    fetched: set[str] = set()

    def fetch(ids):
        fetched.update(str(i) for i in ids)
        return ids

    first = DemandsImages(fetched)
    runner.run_chain(
        first, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=fetch,
        time_budget_minutes=4,
    )
    assert first.of("predict"), "the first session must have run the detector"

    # second session: Drive kept the proposals, /content kept nothing
    survives_restart: set[str] = set()

    def fetch_again(ids):
        survives_restart.update(str(i) for i in ids)
        return ids

    second = DemandsImages(survives_restart)
    results = runner.run_chain(
        second, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
        start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
        chain=protocol.build_chain(4), prepare_images=fetch_again,
    )
    assert len(results) == 3, "the resumed chain must finish"
    assert any(call["verb"] == "predict" and False for call in second.calls) is False
    # the point: training images were fetched even though the detector was cached
    trained = {i for call in second.of("train") for i in call["images"]}
    trained |= {i for call in second.of("train") for i in replay_sources(call)}
    assert trained, "nothing was trained on"
    assert trained <= survives_restart, "trained on images nobody fetched this session"


def test_the_run_stops_clearly_when_the_training_downloads_fail(tmp_path, index, config):
    """The candidate pool arrives, the training subset does not. Say which."""

    calls = {"n": 0}

    def flaky(ids):
        calls["n"] += 1
        return ids if calls["n"] % 2 else []     # the pool works, training does not

    fake = FakeBridge()
    with pytest.raises(RuntimeError, match="training images on disk"):
        runner.run_chain(
            fake, config, workspace=tmp_path, candidate_index=index, replay_index=OLD_DATA,
        replay_root=prob_data_root(tmp_path, index, OLD_DATA),
            start_checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
            chain=protocol.build_chain(4), prepare_images=flaky,
        )
