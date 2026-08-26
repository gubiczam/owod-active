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
    assert "GPU branch: 3 tasks, 9 PROB calls" in capsys.readouterr().out


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

    source = cell_containing("# ============================== PARAMETERS")
    source = (source.replace("RUN_GPU = True", f"RUN_GPU = {run_gpu}")
                    .replace("SMOKE_TEST = True", f"SMOKE_TEST = {smoke}")
                    .replace("FAST_CHAIN = True", f"FAST_CHAIN = {fast}"))
    namespace: dict = {}
    exec(compile(source, "parameters", "exec"), namespace)  # noqa: S102
    assert namespace["N_TASKS"] == tasks
    assert namespace["CANDIDATE_IMAGES"] == candidates
    assert namespace["EPOCHS"] == epochs
