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
        n_prev=19, n_current=1, test_set="eval", epochs=5,
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
    instrument.evaluate(checkpoint=tmp_path / "t1.pth", test_set="eval",
                        output=tmp_path / "m.json", n_prev=19, n_current=1)
    for verb, prev, current in (("predict", "0", "19"), ("evaluate", "19", "1")):
        command = command_for(instrument, verb)
        assert command[command.index("--prev-introduced-classes") + 1] == prev
        assert command[command.index("--current-introduced-classes") + 1] == current


def test_evaluate_asks_for_no_detection_dump(instrument, tmp_path):
    """Writing per-image detections doubles the evaluation's cost and we never read them."""
    (tmp_path / "t1.pth").write_bytes(b"x")
    instrument.evaluate(checkpoint=tmp_path / "t1.pth", test_set="eval",
                        output=tmp_path / "m.json", n_prev=19, n_current=1)
    assert "--no-detections" in command_for(instrument, "evaluate")


def test_the_seed_reaches_prob_on_every_verb(instrument, tmp_path):
    """A seed that only owl honours is not a seed."""
    seeded = bridge.Bridge(prob_root=tmp_path, data_root=tmp_path, dry_run=True, seed=7)
    (tmp_path / "t1.pth").write_bytes(b"x")
    seeded.predict(["a"], checkpoint=tmp_path / "t1.pth", output=tmp_path / "p.npz",
                   n_prev=0, n_current=19)
    seeded.train(["a"], previous_checkpoint=tmp_path / "t1.pth",
                 output_checkpoint=tmp_path / "o" / "c.pth", output_dir=tmp_path / "o",
                 n_prev=19, n_current=1, test_set="eval")
    seeded.evaluate(checkpoint=tmp_path / "t1.pth", test_set="eval",
                    output=tmp_path / "m.json", n_prev=19, n_current=1)
    for command in seeded.commands:
        assert command[command.index("--seed") + 1] == "7"


def test_every_command_names_the_dataset_and_its_root(instrument, tmp_path):
    (tmp_path / "t1.pth").write_bytes(b"x")
    instrument.predict(["a"], checkpoint=tmp_path / "t1.pth",
                       output=tmp_path / "p.npz", n_prev=0, n_current=19)
    instrument.train(["a"], previous_checkpoint=tmp_path / "t1.pth",
                     output_checkpoint=tmp_path / "o" / "c.pth", output_dir=tmp_path / "o",
                     n_prev=19, n_current=1, test_set="eval")
    instrument.evaluate(checkpoint=tmp_path / "t1.pth", test_set="eval",
                        output=tmp_path / "m.json", n_prev=19, n_current=1)
    for command in instrument.commands:
        assert "--dataset" in command and "--data-root" in command
