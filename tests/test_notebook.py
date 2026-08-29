"""The notebook's cells, executed — because compiling them proves nothing.

The failure this file exists for: the preflight cell assigned ``DRIVE`` only on
its ``RUN_GPU = False`` branch, so on the GPU branch the next cell died with
``NameError: name 'DRIVE' is not defined`` — instead of printing the diagnosis it
was written to print. A syntax check cannot see that, and neither can a static
import audit, because the name *is* assigned somewhere in the file.

So these tests run the cells the way Colab does: one namespace, in order, with
``google.colab`` and the GPU replaced by stubs.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "owod_active.ipynb"

#: Names the GPU cells read that the preflight cell is responsible for setting.
PREFLIGHT_CONTRACT = ("PREFLIGHT_OK", "DRIVE", "CHECKPOINT", "POOL_ANNOTATIONS",
                      "TEST_ANNOTATIONS")


def code_cells() -> list[str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in payload["cells"] if c["cell_type"] == "code"]


def cell_containing(marker: str) -> str:
    """The first cell holding ``marker``. Later cells may read the same name."""
    matches = [src for src in code_cells() if marker in src]
    assert matches, f"no cell contains {marker!r}"
    return matches[0]


def _with_flags(source: str, **flags: bool) -> str:
    """Rewrite top-level boolean assignments, whatever they are set to now."""

    import re

    for name, value in flags.items():
        source, count = re.subn(
            rf"^{name} = \w+", f"{name} = {value}", source, count=1, flags=re.MULTILINE
        )
        assert count == 1, f"{name} is not assigned once at the top level"
    return source


def test_the_notebook_has_exactly_one_parameters_cell():
    assert cell_containing("# ============================== PARAMETERS")


def test_every_code_cell_compiles():
    for index, source in enumerate(code_cells()):
        compile(source, f"cell {index}", "exec")


@pytest.mark.parametrize("run_gpu", [False, True])
def test_the_preflight_cell_sets_its_whole_contract_on_both_branches(run_gpu, monkeypatch):
    """The bug this whole file was written for.

    Whichever branch runs, every name the later cells read must exist
    afterwards. On the GPU branch it may legitimately report failure — what it
    may not do is leave a name undefined.
    """

    if run_gpu:
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
        google = types.ModuleType("google")
        google.colab = colab
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.colab", colab)

    import subprocess as real_subprocess

    def fake_run(command, **kwargs):
        # a GPU that is present but a Drive that is empty: the realistic first run
        if command and "nvidia-smi" in str(command[0]):
            return real_subprocess.CompletedProcess(command, 0, "Tesla T4, 15360 MiB\n", "")
        return real_subprocess.CompletedProcess(command, 0, "", "")

    namespace = {
        "RUN_GPU": run_gpu,
        "DRIVE_ROOT": "/content/drive/MyDrive/OWL",
        "Path": Path,
        "subprocess": types.SimpleNamespace(run=fake_run),
        "ROOT": NOTEBOOK.parent.parent,
    }
    exec(cell_containing("PREFLIGHT_OK"), namespace)  # noqa: S102 - that is the point

    for name in PREFLIGHT_CONTRACT:
        assert name in namespace, f"the preflight left {name} undefined on RUN_GPU={run_gpu}"

    if not run_gpu:
        assert namespace["PREFLIGHT_OK"] is False
    else:
        # the annotations ship in the repository, so those two must resolve even
        # when Drive is empty; only the checkpoint can be missing
        assert namespace["POOL_ANNOTATIONS"] is not None
        assert namespace["TEST_ANNOTATIONS"] is not None
        assert namespace["DRIVE"] == Path("/content/drive/MyDrive/OWL")


def test_the_gpu_cells_gate_on_the_preflight_verdict_not_on_a_bare_name():
    """Every GPU cell after the preflight must refuse to run rather than crash."""

    sources = code_cells()
    preflight = next(i for i, src in enumerate(sources) if "PREFLIGHT_OK" in src)
    downstream = sources[preflight + 1:]
    gating = [src for src in downstream if "PREFLIGHT_OK" in src]
    assert gating, "no cell after the preflight checks its verdict"
    # and the old broken guard must not come back
    assert not any("assert DRIVE is not None" in src for src in sources)


def test_the_smoke_test_only_shrinks_the_gpu_chain():
    """SMOKE_TEST must not quietly reduce the CPU results to a toy."""

    parameters = cell_containing("SMOKE = dict(")
    assert "if RUN_GPU and SMOKE_TEST:" in parameters
    namespace: dict = {}
    exec(parameters.replace("RUN_GPU = True", "RUN_GPU = False"), namespace)  # noqa: S102
    assert namespace["N_TASKS"] == 10, "the CPU path must keep the full chain"
    assert namespace["BUDGET_PER_TASK"] == 600


@pytest.mark.slow
def test_the_whole_notebook_runs_with_prob_and_colab_faked(capsys):
    """The end-to-end check: every cell, GPU branch, real owl, fake PROB.

    Two of the three bugs that reached a live GPU session are caught here and
    nowhere else, because both were failures of the *notebook* rather than of
    any module in it. Takes a couple of minutes: it runs the real arm sweep over
    the real 80,000-proposal pool, which is most of the cost and also most of
    the value.
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import dry_run_notebook

    dry_run_notebook.run(run_gpu=True, verbose=False)

    # What this asserts is that the GPU branch ran at all — everything the branch
    # had to *achieve* (both arms, three tasks each, eighteen PROB calls, every
    # metric column present) is asserted inside ``run`` itself, which raises.
    # Retyping its summary sentence here was trap 6 applied to the test suite:
    # the sentence changed when the chain grew a second arm, and the copy in this
    # file went red for a reason that had nothing to do with the notebook.
    output = capsys.readouterr().out
    assert "GPU branch:" in output, output
    assert "grouped recall" in output, output


@pytest.mark.slow
def test_the_cpu_branch_runs_and_skips_the_chain(capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import dry_run_notebook

    dry_run_notebook.run(run_gpu=False, verbose=False)
    assert "GPU chain correctly skipped" in capsys.readouterr().out


def test_the_notebook_retypes_nothing_owl_already_defines():
    """The recurring failure was drift between the cells and the package.

    The cells come from whoever last saved the notebook; owl is re-cloned on
    every run. So a value that exists in both places will eventually disagree,
    and the split name did: a notebook saved before the rename carried
    ``owl_shared_eval`` and failed against the new guard, which reads like a bug
    in the new code. The fix is not a better error message — it is to stop
    duplicating the value.
    """

    from owl import evaluation_subset

    joined = "\n".join(code_cells())
    assert "SHARED_TEST_SET" in joined, "the notebook must read the name from owl"
    assert f'"{evaluation_subset.SHARED_TEST_SET}"' not in joined, (
        "the split name is retyped in the notebook; import it instead"
    )
    assert '"owl_shared_eval"' not in joined


def test_the_drift_guard_names_what_is_missing():
    guard = cell_containing("_REQUIRED")
    for expected in ("prepare_images", "proposals_per_image", "SHARED_TEST_SET",
                     "per_class_ap50", "reuse_deferred_labels"):
        assert expected in guard, f"{expected} is not covered by the drift guard"


def test_the_cuda_extension_check_asks_a_fresh_interpreter():
    """A wheel installed a moment ago is invisible to the process that installed it.

    Asking in-process reported a working build as failed, which costs nothing but
    three times the projected wall clock and a wrong decision about the run.
    """

    cell = cell_containing("MultiScaleDeformableAttention")
    assert "msda_available" in cell
    assert "sys.executable" in cell and "-c" in cell
    assert "invalidate_caches" in cell


@pytest.mark.parametrize(
    ("run_gpu", "smoke", "fast", "tasks", "candidates", "epochs"),
    [
        (True, True, True, 3, 300, 1),      # smoke wins over fast
        (True, False, True, 6, 2000, 5),    # the weekend chain
        (True, False, False, 10, 4000, 5),  # the full chain
        (False, True, True, 10, 4000, 5),   # CPU is never shrunk
    ],
)
def test_the_presets_resolve_the_way_the_run_guide_says(
    run_gpu, smoke, fast, tasks, candidates, epochs
):
    """Three flags, four meanings. A revert costs one boolean, not four values.

    The user has to revert the notebook whenever owl gains something, which loses
    their edits — so the configurations they actually run are presets in the
    repository rather than numbers they retype.
    """

    # Substitute the assignment whatever its committed value is, so this test
    # keeps testing the presets rather than the current defaults.
    source = _with_flags(
        cell_containing("# ============================== PARAMETERS"),
        RUN_GPU=run_gpu, SMOKE_TEST=smoke, FAST_CHAIN=fast,
    )
    namespace: dict = {}
    exec(compile(source, "parameters", "exec"), namespace)  # noqa: S102
    assert namespace["N_TASKS"] == tasks
    assert namespace["CANDIDATE_IMAGES"] == candidates
    assert namespace["EPOCHS"] == epochs


def test_the_notebook_runs_every_arm_from_one_press():
    """A single arm's numbers have nothing to be measured against.

    The arms of the current study are the *replay* arms: selection is held fixed
    at random so that what varies is the class composition of a fixed rehearsal
    budget. The notebook loops them rather than asking the user to edit a name
    and press Run all once per arm — which is what produced the mixed
    ``REPLAY_ARM`` / ``REPLAY_ARMS`` state this test now guards against.
    """

    parameters = cell_containing("# ============================== PARAMETERS")
    namespace: dict = {}
    exec(compile(parameters, "parameters", "exec"), namespace)  # noqa: S102

    from owl import replay, selection

    arms = namespace["ARMS"]
    replay_arms = namespace["REPLAY_ARMS"]
    assert arms == ("random",), "the replay study holds selection fixed"
    assert len(replay_arms) > 1, "one replay arm has nothing to be compared with"
    for arm in arms:
        assert arm in selection.ARMS, f"{arm} is not a registered selection arm"
    for replay_arm in replay_arms:
        assert replay_arm in replay.ARMS, f"{replay_arm} is not a registered replay arm"


def test_no_cell_reaches_for_a_singular_replay_arm():
    """The failure this guards: a Run all that dies on a CPU diagnostic.

    The parameter cell was edited to sweep ``REPLAY_ARMS`` while three later
    cells still read a singular ``REPLAY_ARM``, so a fresh runtime raised
    ``NameError`` on the simulation cell — after the CPU sections had already
    spent minutes. Every cell must take its replay arm from the sweep, either by
    naming one explicitly or by looping.
    """

    for index, source in enumerate(code_cells()):
        for number, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "REPLAY_ARM" not in line:
                continue
            assert "REPLAY_ARMS" in line, (
                f"cell {index} line {number} reads a singular REPLAY_ARM, which "
                f"the parameter cell no longer defines: {stripped}"
            )


def test_the_experiment_audit_names_what_the_run_will_do():
    """Requirement of the overnight run: the configuration is visible up front."""

    parameters = cell_containing("# ============================== PARAMETERS")
    namespace: dict = {}
    exec(compile(parameters, "parameters", "exec"), namespace)  # noqa: S102
    assert callable(namespace["describe_experiment"])

    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        namespace["describe_experiment"]()
    printed = buffer.getvalue()

    for expected in ("selection arm", "replay arm", "tasks", "candidate images",
                     "annotation budget", "selection rounds", "epochs", "seed",
                     "workspaces", "max time"):
        assert expected in printed, f"the audit does not report {expected!r}"
    for arm in namespace["ARMS"]:
        for replay_arm in namespace["REPLAY_ARMS"]:
            assert f"work/{arm}__{replay_arm}" in printed, (
                f"the audit does not name the workspace for {arm}__{replay_arm}")


def test_each_run_gets_its_own_time_budget_and_skips_are_named():
    """Sharing one budget truncates whichever run goes second.

    At the measured V3 cost a five-task chain is about 263 minutes, so a single
    420-minute pot leaves the second run 157 and it stops after three tasks —
    producing two arms that cannot be compared with each other. Each run gets
    its own cap; the session ceiling is what stops scheduling, and a run that
    never started has to be named rather than silently missing.
    """

    chain = cell_containing("for arm in ARMS:")
    assert "for replay_arm in REPLAY_ARMS:" in chain, "the replay arms must be swept"
    assert "remaining = TIME_BUDGET_MINUTES" in chain, "the cap must be per run"
    assert "TIME_BUDGET_MINUTES - spent" not in chain, "that is the shared budget again"
    assert "session_ceiling = TIME_BUDGET_MINUTES * len(planned)" in chain
    assert "not started" in chain, "a run that never ran must be named"
    assert "Run all again" in chain, "the user must be told it resumes"
    assert 'workspace=WORK / run' in chain, "each run needs its own workspace"
    assert 'replace(base_config, arm=arm, replay_arm=replay_arm)' in chain, (
        "the singular config field is what the loop must fill")


def test_the_notebook_is_committed_ready_to_run():
    """The user's ask: open the link, press Run all, change nothing."""

    parameters = cell_containing("# ============================== PARAMETERS")
    namespace: dict = {}
    exec(compile(parameters, "parameters", "exec"), namespace)  # noqa: S102
    assert namespace["RUN_GPU"] is True
    assert namespace["SMOKE_TEST"] is False, "the smoke test has already passed"
    assert namespace["FAST_CHAIN"] is True
    assert namespace["N_TASKS"] == 6 and namespace["CANDIDATE_IMAGES"] == 2000


def test_the_comparison_only_uses_the_depth_every_arm_reached():
    """A five-task arm against a two-task arm is not a result."""

    results = cell_containing("tail U-Recall at equal oracle cost")
    assert "min(len(rows) for rows in by_arm.values()" in results
    assert "every arm reached" in results
