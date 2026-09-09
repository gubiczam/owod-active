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


def test_the_notebook_pins_the_revision_that_carries_the_chain_support():
    """A full SHA that exists, and whose code is what the notebook needs.

    The pin necessarily lags the commit that sets it by one -- a notebook cannot
    contain the SHA of the commit containing it -- so what must hold is that
    nothing the notebook *executes* differs between the two. Checked here on
    ``owl/`` and on every tool the notebook invokes; the clean-room validator
    proves the same thing by running the pinned code.
    """

    import re
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    source = _t10_cells()[0]
    pin = re.search(r'OWL_COMMIT = "([0-9a-f]*)"', source).group(1)
    assert len(pin) == 40, f"OWL_COMMIT is {pin!r}, not a full 40-character SHA"
    assert "assert len(PROB_COMMIT) == 40 and len(OWL_COMMIT) == 40" in source, (
        "the 40-character assertion must stay; it is what catches a short pin")
    assert subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{pin}^{{commit}}"],
        capture_output=True, check=False).returncode == 0, f"{pin} is not a commit"

    # the pinned revision must already carry the chain support
    listing = subprocess.run(
        ["git", "-C", str(root), "show", f"{pin}:tools/run_full_owod_benchmark.py"],
        capture_output=True, text=True, check=True).stdout
    assert '"--n-tasks"' in listing, "the pinned launcher has no --n-tasks"

    invoked = set(re.findall(r'"tools"\s*/\s*"([a-z0-9_]+\.py)"',
                             "\n".join(_t10_cells())))
    assert invoked, "no tool invocations found"
    for tree in ["owl/", *(f"tools/{t}" for t in sorted(invoked))]:
        drift = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", pin, "HEAD", "--", tree],
            capture_output=True, text=True, check=True).stdout.strip()
        assert not drift, (
            f"{tree} differs between the pinned {pin[:12]} and HEAD:\n{drift}\n"
            "Re-pin, or the session runs code this tree no longer has.")


def test_the_notebooks_prose_describes_the_session_it_actually_runs():
    """The header comment and the configuration must not contradict.

    This notebook was derived from the four-task seed-2 replication notebook,
    and it inherited that notebook's header: "the three surviving baseline arms
    at seed 2", including the claim that "Seeds 0 and 1 are both complete for
    these three arms". It runs two arms at seed 0 on a chain where nothing is
    complete. Every executable line was right and every sentence above them was
    wrong, which is the version of this defect a reviewer is least likely to
    catch -- so it is asserted rather than remembered.
    """

    parameters = _t10_cells()[0]
    prose = "\n".join(line for line in parameters.splitlines()
                      if line.strip().startswith("#"))

    # It must not describe a seed or an arm count this session does not run.
    for stale in ("seed 2", "Seed 2", "seed-2", "three surviving",
                  "the third of the pre-registered seed set"):
        assert stale not in prose, (
            f"the header comment still says {stale!r}, but SEEDS = (0,) and "
            "SESSION_ARMS has two arms")

    # And it must name what this session is: two arms, seed 0, ten tasks.
    assert "seed 0" in prose, "the header must say which seed this is"
    assert "ten-task" in prose or "ten task" in prose
    # The excluded arms stay named with their reasons -- that is provenance,
    # not decoration, and dropping it is how an exclusion becomes invisible.
    for excluded in ("proposed", "proposed_v2", "coreset"):
        assert excluded in prose, f"{excluded} must stay recorded as excluded"
    # Replication seeds must be described as a later session, never as a
    # promise that can be quietly withdrawn once seed 0's numbers are visible.
    assert "seeds 1, 2" in prose or "seeds 1 and 2" in prose


def test_the_notebook_states_it_cannot_overwrite_a_four_task_result():
    """Both mechanisms, in the notebook and in the code."""

    parameters = _t10_cells()[0]
    assert "results/full_owod_chain_t10" in parameters
    assert "fingerprint" in parameters, (
        "the header must say why a workspace collision is refused, not only "
        "that the directory differs")

    from owl import runner

    assert "n_tasks" in runner.CycleConfig.RESULT_AFFECTING
    assert (runner.CycleConfig(n_tasks=4).fingerprint()
            != runner.CycleConfig(n_tasks=10).fingerprint())


def test_the_notebook_states_the_runtime_for_the_split_it_actually_scores():
    """22 h, not the 15-18 h inherited from the four-task planner.

    ``plan_full_owod_benchmark.py`` sizes evaluation from Benchmark V1's 837
    images, and cell [8/10] prints that table. This chain is scored on
    ``owl_shared_test_t10`` -- 2,817 images, with a second forward pass for
    ``detections=True`` -- so evaluation costs 36.7 min per arm-task instead of
    11.1, eighteen times over. Getting this wrong does not corrupt a result; it
    tells the operator one Run all will finish when it will not, which is how a
    chain gets abandoned half-done and reported as a failure of the method.
    """

    body = _t10_cells()[8]
    assert "15-18 hours" not in body, "the four-task figure must not be restated"
    assert "22 hours" in body, "the corrected figure must be stated"
    assert "2,817" in body and "837" in body, (
        "the comment must show both splits, so the correction can be checked")
    assert "MORE THAN ONCE" in body, (
        "the operator must be told one session will not finish")
    assert "restored from state.json" in body


def test_the_time_budget_is_below_the_expected_runtime_on_purpose():
    """A budget under the runtime is the resume design, not an error."""

    parameters = _t10_cells()[0]
    assert "TIME_BUDGET_MINUTES = 1200" in parameters
    body = _t10_cells()[8]
    assert "BELOW that" in body and "deliberate" in body, (
        "a reader who notices 1200 min < 22 h must find the reason here, or "
        "they will 'fix' it by raising the budget and lose the graceful stop")
