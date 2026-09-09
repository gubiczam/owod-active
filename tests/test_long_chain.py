"""Running the frozen protocol over a longer chain, without changing it.

The 2026-08-25 consultation asked for the protocol to be measured through
``t1 -> ... -> t10``. ``owl.protocol.build_chain`` always supported that; what
was missing was a way to say so without editing a frozen constant, and a way to
stop the two chains from being mistaken for one experiment.

The property that matters most here is the *negative* one: Benchmark V1 must be
bit-identical to what it was measured as. Everything long-chain is opt-in.
"""

from __future__ import annotations

import itertools

import pytest

from owl import evaluation_subset, protocol
from owl.active_selection import benchmark as bm

# ------------------------------------------------------- V1 is untouched ---


def test_the_frozen_four_task_defaults_have_not_moved():
    assert bm.N_TASKS == 4
    assert [t.name for t in bm.chain()] == ["t1", "t2", "t3", "t4"]
    assert bm.declared_classes() == ("traffic light", "fire hydrant", "stop sign")
    assert bm.cycle_config("random", 0).n_tasks == 4
    assert bm.check_protocol()["agrees"], "the protocol document no longer agrees"


def test_four_tasks_keeps_the_bare_split_name():
    """Renaming V1's split would orphan every completed trajectory."""

    assert evaluation_subset.shared_test_set_name(4) == evaluation_subset.SHARED_TEST_SET


# --------------------------------------------------------- the long chain ---


def test_the_ten_task_chain_declares_one_class_per_task():
    chain = bm.chain(10)
    assert [t.name for t in chain] == [f"t{i}" for i in range(1, 11)]
    assert bm.declared_classes(10) == (
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "chair", "diningtable", "pottedplant", "backpack",
    )
    # every task adds exactly one, and the anchor's 19 are never re-declared
    for previous, task in itertools.pairwise(chain):
        assert len(task.known_classes) == len(previous.known_classes) + 1
        assert task.new_class not in previous.known_classes


def test_the_long_chain_spans_all_three_frequency_bands():
    """The scientific reason to run it: t3/t4 are tail-and-scarce, t6-t10 are not."""

    groups = protocol.load_groups()
    bands = {groups[c] for c in bm.declared_classes(10)}
    assert bands == {"head", "medium", "tail"}, bands
    assert len({c for c in bm.chain(4)[-1].known_classes if groups.get(c) == "tail"}) == 3
    assert len({c for c in bm.chain(10)[-1].known_classes if groups.get(c) == "tail"}) == 4


def test_cycle_config_takes_the_chain_length_and_nothing_else_moves():
    short, long = bm.cycle_config("random", 0), bm.cycle_config("random", 0, n_tasks=10)
    assert (short.n_tasks, long.n_tasks) == (4, 10)
    for field in ("budget_per_task", "budget_unit", "rounds_per_task",
                  "candidate_images_per_task", "proposals_per_image",
                  "labelling_policy", "replay_arm", "epochs", "learning_rate"):
        assert getattr(short, field) == getattr(long, field), field


@pytest.mark.parametrize("length", [1, 0, -3])
def test_a_degenerate_chain_is_refused(length):
    with pytest.raises(bm.BenchmarkError):
        bm.cycle_config("random", 0, n_tasks=length)


# ------------------------------- the two experiments cannot be confused ---


def test_a_longer_chain_gets_its_own_split_under_its_own_name():
    """Same name would let one silently overwrite the other in a data root."""

    from pathlib import Path

    from tools.prepare_full_owod_benchmark import shared_test_split

    short_name, short = shared_test_split(Path("/nonexistent"), write=False)
    long_name, long = shared_test_split(Path("/nonexistent"), write=False, n_tasks=10)

    assert short_name != long_name
    assert (short_name, len(short.image_ids)) == ("owl_shared_test", 837)
    assert (long_name, len(long.image_ids)) == ("owl_shared_test_t10", 2817)
    assert len(long.image_ids) > 3 * len(short.image_ids), (
        "the long chain must be measured on its own, much larger split -- which "
        "is exactly why its numbers are not comparable with V1's")


@pytest.mark.parametrize("length", [4, 7, 10])
def test_every_split_name_passes_probs_marker_guard(length):
    """PROB routes annotation filtering by substring; `test` must be the only one."""

    name = evaluation_subset.shared_test_set_name(length)
    assert evaluation_subset.check_split_name(name) == name


def test_both_tools_expose_the_chain_length():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for tool in ("run_full_owod_benchmark.py", "prepare_full_owod_benchmark.py"):
        help_text = subprocess.run(
            [sys.executable, str(root / "tools" / tool), "--help"],
            capture_output=True, text=True, check=True).stdout
        assert "--n-tasks" in help_text, tool
    assert "NOT comparable" in subprocess.run(
        [sys.executable, str(root / "tools" / "run_full_owod_benchmark.py"), "--help"],
        capture_output=True, text=True, check=True).stdout


# ----------------------------------------------- the manifest tells the truth ---


def _manifest(n_tasks=None):
    return bm.manifest(
        trajectories=[], owl_commit="a" * 40, prob_commit="b" * 40,
        prob_repository="x", checkpoint="c", checkpoint_sha256=None,
        test_set="owl_shared_test", test_images=837, n_tasks=n_tasks,
    )


def test_the_manifest_records_the_chain_actually_run():
    """`frozen` states the *protocol's* n_tasks; the record must state the run's.

    Without this a ten-task session wrote a manifest describing a four-task
    chain under V1's experiment name -- the provenance file would have been
    wrong about the experiment it recorded.
    """

    short, long = _manifest(), _manifest(10)
    assert short["chain_length"] == 4 and long["chain_length"] == 10
    assert [c["task"] for c in short["chain"]][-1] == "t4"
    assert [c["task"] for c in long["chain"]][-1] == "t10"
    assert len(long["chain"]) == 10


def test_a_longer_chain_gets_its_own_experiment_name():
    assert _manifest()["experiment"] == "full_owod_active_benchmark_v1"
    assert _manifest(10)["experiment"] == "full_owod_chain_t10"


def test_the_manifest_states_the_comparability_limit_in_words():
    """A reader of the file must not have to infer it."""

    short, long = _manifest(), _manifest(10)
    assert short["comparable_with_benchmark_v1"] is True
    assert long["comparable_with_benchmark_v1"] is False
    assert "different shared evaluation split" in long["comparability_note"]
    assert "must NOT be placed in the same table" in long["comparability_note"]


def test_the_frozen_block_still_states_the_protocol_value():
    """`check_protocol` pins this against the document; it must not follow the run."""

    assert _manifest(10)["frozen"]["n_tasks"] == 4
    assert bm.check_protocol()["agrees"]


# --------------------------------------------------------- the T10 notebook ---


def _t10_cells(kind="code"):
    import json
    from pathlib import Path

    payload = json.loads(
        (Path(__file__).resolve().parent.parent / "notebooks"
         / "full_owod_chain_t10.ipynb").read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in payload["cells"] if c["cell_type"] == kind]


def test_the_notebook_asks_for_ten_tasks_everywhere_it_matters():
    code = "\n".join(_t10_cells())
    assert "N_TASKS = 10" in code
    assert '"--n-tasks", str(N_TASKS)' in code
    assert code.count('"--n-tasks"') >= 2, "prepare and the launcher both need it"
    assert 'SESSION_ARMS = ("random", "admissibility")' in code
    assert "SEEDS = (0,)" in code
    for excluded in ("entropy", "distribution_aware_iterative_v1"):
        assert f'"{excluded}")' not in code.split("SESSION_ARMS")[1][:80], excluded


def test_the_notebook_writes_to_its_own_results_directory():
    code = "\n".join(_t10_cells())
    assert 'RESULTS_RELATIVE = "results/full_owod_chain_t10"' in code
    assert "full_owod_active_benchmark_v1" not in code.split("RESULTS_RELATIVE")[1][:200]


def test_the_notebook_keeps_the_frozen_value_frozen():
    code = "\n".join(_t10_cells())
    assert 'assert bm.N_TASKS == 4' in code
    assert "shared_test_set_name(N_TASKS) \\\n    != evaluation_subset.shared_test_set_name(bm.N_TASKS)" in code


def test_the_notebook_states_the_second_experiment_rule_and_the_banking_limit():
    markdown = "\n".join(_t10_cells("markdown"))
    assert "SECOND EXPERIMENT" in markdown
    assert "2,817" in markdown and "837" in markdown
    assert "68.3" in markdown and "81.2" in markdown
    assert "published S-OWODB" in markdown  # markdown carries **not** in bold


def test_the_notebook_pin_fails_closed_until_it_is_set():
    """It cannot contain the SHA of the commit containing it, so it must refuse."""

    import re

    source = _t10_cells()[0]
    pin = re.search(r'OWL_COMMIT = "([^"]*)"', source).group(1)
    assert pin == "PIN_AFTER_PUSH", pin
    assert len(pin) != 40, "a placeholder must not look like a real SHA"
    assert "assert len(PROB_COMMIT) == 40 and len(OWL_COMMIT) == 40" in source, (
        "the 40-character assertion is what makes the placeholder fail closed")
