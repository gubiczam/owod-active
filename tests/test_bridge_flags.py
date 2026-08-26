"""What PROB is actually told on the command line.

The failure this file exists for: ``train`` never passed ``--test-set``, so PROB
fell back to its own default — ``owod_all_task_test``, a split file this protocol
never writes — and died inside its dataset constructor after loading the
training images. And it never passed ``--eval-every``, whose PROB default is 1,
which would have evaluated after every epoch and multiplied the wall clock of
the expensive half of the protocol by the number of epochs.

Neither is findable by faking the bridge: the bug is in what PROB assumes when
we say nothing. So this file asserts on the command line itself, and names, for
every flag, the default that would apply if we left it out.

``PROB_DEFAULTS`` is the knowledge worth keeping. Every entry is a default that
is wrong for this protocol, so every entry must appear on the command line.
"""

from __future__ import annotations

import pytest

from owl import bridge

#: flag -> (PROB's own default, why it is wrong here)
PROB_DEFAULTS = {
    "--test-set": (
        "owod_all_task_test",
        (
            "a split file this protocol never writes; PROB builds a validation "
            "dataset during training and fails on the missing file"
        ),
    ),
    "--eval-every": (
        "1",
        (
            "evaluates after every epoch, and evaluation is the expensive half of "
            "the protocol — five epochs would cost five unwanted evaluations"
        ),
    ),
    "--learning-rate": (
        "2e-05",
        (
            "the rate the earlier work measured 0.010 new-class mAP50 at; PROB's own "
            "training default is 2e-4"
        ),
    ),
    "--epochs": ("1", "one epoch does not learn a class from a few hundred regions"),
    "--supervision-mode": (
        "train",
        (
            "a 'train' split strips previous-task boxes, which is what made "
            "forgetting look catastrophic"
        ),
    ),
    "--seed": (
        "0",
        (
            "PROB seeds itself independently, so a run with SEED = 1 would still "
            "shuffle and initialise exactly like SEED = 0"
        ),
    ),
}

#: Flags that agree with PROB's default and are still passed explicitly, because
#: the choice is consequential enough that it should be readable in a log.
DELIBERATE_AGREEMENTS = {
    "--freeze-prob-model": "decides whether 'unknown' means the same thing after "
                           "an incremental step as before it",
}


@pytest.fixture
def instrument(tmp_path):
    """A bridge that records command lines instead of running anything."""
    return bridge.Bridge(prob_root=tmp_path, data_root=tmp_path / "data", dry_run=True)


def command_for(instrument, verb: str) -> list[str]:
    matching = [c for c in instrument.commands if verb in c]
    assert len(matching) == 1, f"expected one {verb} command, got {len(matching)}"
    return matching[0]


def test_train_names_every_default_that_would_be_wrong(instrument, tmp_path):
    (tmp_path / "t1.pth").write_bytes(b"x")
    instrument.train(
        ["a", "b"], previous_checkpoint=tmp_path / "t1.pth",
        output_checkpoint=tmp_path / "out" / "checkpoint.pth",
        output_dir=tmp_path / "out", n_prev=19, n_current=1,
        test_set="owl_shared_eval", epochs=5, learning_rate=2e-4,
    )
    command = command_for(instrument, "train")
    for flag, (default, why) in PROB_DEFAULTS.items():
        assert flag in command, f"{flag} is left to PROB's default {default!r}: {why}"
    for flag, why in DELIBERATE_AGREEMENTS.items():
        assert flag in command or f"--no-{flag[2:]}" in command, f"{flag}: {why}"


def test_train_disables_evaluation_during_training(instrument, tmp_path):
    (tmp_path / "t1.pth").write_bytes(b"x")
    instrument.train(
        ["a"], previous_checkpoint=tmp_path / "t1.pth",
        output_checkpoint=tmp_path / "o" / "c.pth", output_dir=tmp_path / "o",
        n_prev=19, n_current=1, test_set="owl_shared_test", epochs=5,
    )
    command = command_for(instrument, "train")
    every = int(command[command.index("--eval-every") + 1])
    assert every > 5, "PROB would evaluate inside the training loop"


def test_train_requires_a_test_set_rather_than_defaulting(tmp_path):
    """Making it keyword-only and required is what stops this recurring."""
    import inspect

    parameter = inspect.signature(bridge.Bridge.train).parameters["test_set"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_predict_and_evaluate_carry_the_class_counts(instrument, tmp_path):
    (tmp_path / "t1.pth").write_bytes(b"x")
    instrument.predict(["a"], checkpoint=tmp_path / "t1.pth",
                       output=tmp_path / "p.npz", n_prev=0, n_current=19)
    instrument.evaluate(checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
                        output=tmp_path / "m.json", n_prev=19, n_current=1)
    for verb, prev, current in (("predict", "0", "19"), ("evaluate", "19", "1")):
        command = command_for(instrument, verb)
        assert command[command.index("--prev-introduced-classes") + 1] == prev
        assert command[command.index("--current-introduced-classes") + 1] == current


def test_evaluate_writes_the_detections_the_plan_needs(instrument, tmp_path):
    """The per-box artefact costs a second forward pass and is on by default.

    An earlier version of this test asserted the opposite — that
    ``--no-detections`` is always passed, because the artefact doubles the
    evaluation and nothing read it. That premise is gone: the research plan's
    headline endpoint is U-Recall split by frequency group, which cannot be
    computed from the aggregate the evaluator prints, and the artefact is where
    the per-box detections live. So the default flipped, and the flag now appears
    only when the decomposition is deliberately given up.
    """

    (tmp_path / "t1.pth").write_bytes(b"x")
    instrument.evaluate(checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
                        output=tmp_path / "on.json", n_prev=19, n_current=1)
    assert "--no-detections" not in command_for(instrument, "evaluate")

    off = bridge.Bridge(prob_root=tmp_path, data_root=tmp_path, dry_run=True)
    off.evaluate(checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
                 output=tmp_path / "off.json", n_prev=19, n_current=1, detections=False)
    assert "--no-detections" in off.commands[0]


def test_the_seed_reaches_prob_on_every_verb(instrument, tmp_path):
    """A seed that only owl honours is not a seed."""
    seeded = bridge.Bridge(prob_root=tmp_path, data_root=tmp_path, dry_run=True, seed=7)
    (tmp_path / "t1.pth").write_bytes(b"x")
    seeded.predict(["a"], checkpoint=tmp_path / "t1.pth", output=tmp_path / "p.npz",
                   n_prev=0, n_current=19)
    seeded.train(["a"], previous_checkpoint=tmp_path / "t1.pth",
                 output_checkpoint=tmp_path / "o" / "c.pth", output_dir=tmp_path / "o",
                 n_prev=19, n_current=1, test_set="owl_shared_test")
    seeded.evaluate(checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
                    output=tmp_path / "m.json", n_prev=19, n_current=1)
    for command in seeded.commands:
        assert command[command.index("--seed") + 1] == "7"


def test_every_command_names_the_dataset_and_its_root(instrument, tmp_path):
    (tmp_path / "t1.pth").write_bytes(b"x")
    instrument.predict(["a"], checkpoint=tmp_path / "t1.pth",
                       output=tmp_path / "p.npz", n_prev=0, n_current=19)
    instrument.train(["a"], previous_checkpoint=tmp_path / "t1.pth",
                     output_checkpoint=tmp_path / "o" / "c.pth", output_dir=tmp_path / "o",
                     n_prev=19, n_current=1, test_set="owl_shared_test")
    instrument.evaluate(checkpoint=tmp_path / "t1.pth", test_set="owl_shared_test",
                        output=tmp_path / "m.json", n_prev=19, n_current=1)
    for command in instrument.commands:
        assert "--dataset" in command and "--data-root" in command


# --------------------------------------------------- how PROB routes a split ---


def test_a_split_named_eval_gets_no_filtering_and_that_is_the_trap():
    """The most expensive kind of bug: one that returns numbers instead of an error.

    PROB picks the annotation filter by substring of the split's name.
    ``make_coco_transforms`` tests train / ft / val / test in that order, and
    ``OWDetection.__getitem__`` then tests train / test / ft only. So a name
    matching ``val`` — and ``eval`` matches ``val`` — reaches a branch where
    **no filter runs at all**.

    What that costs: ``label_known_class_and_unknown`` is what relabels every
    not-yet-known object to the unknown class index. Without it there is no
    unknown ground truth, so U-Recall reads zero for every arm at every task,
    and future-task objects are scored as if their class were already known.
    """

    from owl.evaluation_subset import MARKER_BEHAVIOUR, SplitNameError, check_split_name

    assert check_split_name("owl_shared_test") == "owl_shared_test"
    for wrong in ("owl_shared_eval", "owl_eval", "live_cycle_eval", "shared_val"):
        with pytest.raises(SplitNameError, match="routes a split by substring"):
            check_split_name(wrong)
    # and a name carrying two markers is rejected rather than silently resolved
    with pytest.raises(SplitNameError):
        check_split_name("train_test_split")
    assert "NOTHING" in MARKER_BEHAVIOUR["val"]


def test_evaluate_refuses_a_misrouting_split_name(instrument, tmp_path):
    from owl.evaluation_subset import SplitNameError

    (tmp_path / "t1.pth").write_bytes(b"x")
    with pytest.raises(SplitNameError):
        instrument.evaluate(checkpoint=tmp_path / "t1.pth", test_set="owl_shared_eval",
                            output=tmp_path / "m.json", n_prev=19, n_current=1)


def test_writing_an_image_set_checks_the_name_it_will_be_known_by(tmp_path):
    from owl import evaluation_subset as module

    subset = module.EvaluationSubset(("a", "b"), ("a",), ("b",), {"chair": 1})
    good = module.write_image_set(tmp_path / "owl_shared_test.txt", subset)
    assert good.read_text().split() == ["a", "b"]
    with pytest.raises(module.SplitNameError):
        module.write_image_set(tmp_path / "owl_shared_eval.txt", subset)
