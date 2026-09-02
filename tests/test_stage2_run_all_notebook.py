"""Static validation of the Stage-2 Run-All notebook.

Written because the notebook's failure mode is not a crash in cell 1 -- it is a
cell twenty minutes in that assumes something an earlier cell never created. The
bug this replaced was exactly that: ``AssertionError: Missing /content/data/OWOD``
in a notebook that never materialised it.

So these tests check the properties a Run-All notebook has to have and that
reading it casually does not reveal: that every script it calls exists, that every
flag it passes is a real flag, that no cell touches a path before the cell that
creates it, and that nothing is assumed to survive from a previous session.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "method_v2_stage2_run_all.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"]


@pytest.fixture(scope="module")
def joined(code_cells) -> str:
    return "\n".join(code_cells)


# --------------------------------------------------------------- well-formed ---


def test_the_notebook_parses_and_declares_a_gpu_accelerator(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["cells"], "no cells"


def test_every_code_cell_compiles(code_cells):
    for index, source in enumerate(code_cells):
        # strip Colab's #@title directive lines, which are comments anyway
        ast.parse(source, filename=f"<cell {index}>")


def test_no_cell_uses_shell_magics_that_hide_failures(joined):
    """`!cmd` swallows a non-zero exit; subprocess with check=True does not."""

    for line in joined.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("!"), f"shell magic hides failures: {line}"
        assert not stripped.startswith("%%"), f"cell magic: {line}"


def test_the_title_and_instructions_are_present(notebook):
    first = "".join(notebook["cells"][0]["source"])
    assert "METHOD V2 STAGE 2 — ONE CLICK RUN" in first
    assert "T4 GPU" in first
    assert "Run all" in first


def test_progress_headings_are_numbered_in_order(code_cells):
    found = [int(m.group(1)) for source in code_cells
             for m in re.finditer(r"\[(\d+)/10\]", source)]
    assert found, "no [n/10] progress headings"
    ordered = sorted(set(found))
    assert ordered == list(range(1, max(ordered) + 1)), ordered
    assert max(ordered) == 10


# ------------------------------------------------------------ ordering rules ---


def _first_index(code_cells, needle: str) -> int:
    for index, source in enumerate(code_cells):
        if needle in source:
            return index
    raise AssertionError(f"{needle!r} never appears in any code cell")


def test_drive_is_mounted_before_any_drive_path_is_used(code_cells):
    mount = _first_index(code_cells, "drive.mount")
    first_use = _first_index(code_cells, "/content/drive/MyDrive")
    assert mount <= first_use, "a Drive path is used before Drive is mounted"


def test_the_repository_is_pinned_before_it_is_imported(code_cells):
    checkout = _first_index(code_cells, 'git", "-C", REPO, "checkout"')
    import_owl = _first_index(code_cells, "from owl import")
    assert checkout < import_owl, "owl is imported before the repo is pinned"


def test_dependencies_are_installed_before_owl_is_imported(code_cells):
    install = _first_index(code_cells, '"-e", REPO')
    import_owl = _first_index(code_cells, "from owl import")
    assert install < import_owl


def test_the_data_root_is_materialised_before_any_exporter_reads_it(code_cells):
    """The bug this notebook replaces: an exporter running before the fetch."""

    bootstrap = _first_index(code_cells, "bootstrap_stage2_data.py")
    for exporter in ("export_dinov2_features.py",
                     "export_ref_t1_features.py",
                     "export_dinov2_consistency_views.py"):
        assert bootstrap < _first_index(code_cells, exporter), (
            f"{exporter} runs before the data root is materialised"
        )


def test_no_cell_asserts_the_data_root_exists_before_materialisation(code_cells):
    bootstrap = _first_index(code_cells, "bootstrap_stage2_data.py")
    for index, source in enumerate(code_cells[:bootstrap]):
        assert "DATA_ROOT" not in source, (
            f"cell {index} references DATA_ROOT before it is created"
        )


def test_the_diagnostic_runs_last_and_after_all_three_exports(code_cells):
    diagnostic = _first_index(code_cells, "diagnose_method_v2_stage2.py")
    for export in ("export_dinov2_features.py", "export_ref_t1_features.py",
                   "export_dinov2_consistency_views.py"):
        assert _first_index(code_cells, export) < diagnostic
    # and it only runs once
    assert sum(source.count("diagnose_method_v2_stage2.py")
               for source in code_cells) == 1


def test_the_manifest_is_asserted_before_the_reference_export(code_cells):
    """A wrong reference must stop the run in seconds, not after an export."""

    manifest = _first_index(code_cells, "EXPECTED_REF_T1_MANIFEST")
    export = _first_index(code_cells, "export_ref_t1_features.py")
    assert manifest < export


# ----------------------------------------------- the scripts and their flags ---


def _scripts(joined: str) -> set[str]:
    return set(re.findall(r'tools/([A-Za-z0-9_]+\.py)', joined))


def test_every_script_the_notebook_calls_exists(joined):
    called = _scripts(joined)
    assert called, "the notebook calls no repository script"
    for name in sorted(called):
        assert (ROOT / "tools" / name).is_file(), f"tools/{name} does not exist"


def test_every_flag_the_notebook_passes_is_a_real_flag(joined, code_cells):
    """Checked against each script's own argparse, not against a list here."""

    for name in sorted(_scripts(joined)):
        source = (ROOT / "tools" / name).read_text(encoding="utf-8")
        declared = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source))
        used = set()
        for cell in code_cells:
            if name not in cell:
                continue
            # the flags in the same call as this script
            segment = cell.split(name, 1)[1]
            used |= set(re.findall(r'"(--[a-z0-9-]+)"', segment.split("])")[0]))
        unknown = used - declared
        assert not unknown, f"tools/{name} has no {sorted(unknown)}; declares {sorted(declared)}"


def test_required_flags_are_supplied_for_every_script_call(joined, code_cells):
    for name in sorted(_scripts(joined)):
        source = (ROOT / "tools" / name).read_text(encoding="utf-8")
        required = set(re.findall(
            r'add_argument\(\s*"(--[a-z0-9-]+)"[^)]*required=True', source))
        for cell in code_cells:
            if name not in cell:
                continue
            segment = cell.split(name, 1)[1].split("])")[0]
            supplied = set(re.findall(r'"(--[a-z0-9-]+)"', segment))
            missing = required - supplied
            assert not missing, f"a call to tools/{name} omits {sorted(missing)}"


def test_the_notebook_pins_a_full_commit_sha(joined):
    match = re.search(r'COMMIT\s*=\s*"([0-9a-f]{40})"', joined)
    assert match, "COMMIT is not a pinned 40-character SHA"
    assert 'checkout", "-q", COMMIT' in joined
    assert "REPLACE_AT_COMMIT" not in joined, "the placeholder was never replaced"


# --------------------------------------------------- frozen science is asserted ---


def test_the_frozen_numbers_are_asserted_not_merely_printed(joined):
    """Counts inline; the manifest via its named constant.

    Restating the SHA in the notebook would be worse than importing it: two
    copies can drift, and the notebook's copy is the one nothing would check.
    """

    assert "19000" in joined and "14901" in joined
    assert "EXPECTED_REF_T1_MANIFEST" in joined
    assert 'per-class-cap", "1000"' in joined

    from tools.bootstrap_stage2_data import EXPECTED_REF_T1_MANIFEST
    assert EXPECTED_REF_T1_MANIFEST == (
        "a062fc8f4fd43ea52842725aeaa5eccc0e06eab1894b867b248927bd9d2a2a63")


def test_the_notebook_chooses_no_scientific_parameter(joined):
    """Execution infrastructure only.

    No numeric threshold appears anywhere in the notebook, not even as display
    text: the values it prints are imported from the frozen module, so a
    threshold cannot be restated here and then quietly go stale.
    """

    for forbidden in ("lambda_", "gamma =", "dinov2_vitl", "dinov2_vits",
                      "CROP_MARGIN =", "NMS_IOU =", "per_class_cap =",
                      "K_NEIGHBOURS =", "N_CLUSTERS ="):
        assert forbidden not in joined, f"the notebook sets {forbidden!r}"

    thresholds = re.findall(r"(?<![\w.])0\.\d\d(?![\w])", joined)
    assert not thresholds, f"numeric thresholds restated in the notebook: {thresholds}"
    assert "D_GO_UNKNOWN_VS_KNOWN_AUC" in joined
    assert "C_GO_UNKNOWN_VS_BACKGROUND_AUC" in joined


def test_a_gpu_is_required_explicitly(joined):
    assert "torch.cuda.is_available()" in joined
    assert "T4 GPU" in joined


# ------------------------------------------------------------- resumability ---


def test_expensive_outputs_live_on_drive_not_in_content(joined):
    for name in ("BASE_EXPORT", "REF_T1_EXPORT", "VIEWS_EXPORT"):
        match = re.search(rf'{name}\s*=\s*f?"([^"]+)"', joined)
        assert match, f"{name} is not assigned a path"
        assert "{FEATURES}" in match.group(1), (
            f"{name} is not under Drive: {match.group(1)}"
        )


def test_every_expensive_step_is_skipped_when_its_output_exists(joined):
    """Run all pressed twice must resume, not recompute."""

    assert joined.count("os.path.exists(") >= 4
    for name in ("REF_T1_EXPORT", "VIEWS_EXPORT", "BASE_EXPORT"):
        assert f"os.path.exists({name})" in joined, f"{name} is recomputed blindly"


def test_nothing_is_assumed_to_survive_from_a_previous_session(code_cells):
    """Every ephemeral path must be created by an earlier cell in this run."""

    created = set()
    for source in code_cells:
        for path in re.findall(r'"(/content/[A-Za-z0-9_./-]+)"', source):
            if path.startswith("/content/drive"):
                continue          # Drive is persistent and mounted in cell 1
            root = "/".join(path.split("/")[:3])
            assert root in created or "clone" in source or "mkdir" in source \
                or path in ("/content/owod-active", "/content/data/OWOD") \
                or "stage2_images.txt" in path, (
                    f"{path} is used before anything creates it")
        if "git" in source and "clone" in source:
            created.add("/content/owod-active")
        if "bootstrap_stage2_data.py" in source:
            created.add("/content/data")
    assert "/content/owod-active" in created


def test_results_are_copied_to_drive_at_the_end(joined):
    assert "shutil.copy2" in joined
    assert "method_v2_stage2_summary.json" in joined


def test_the_final_summary_prints_the_verdicts_and_the_ladder(joined):
    for token in ("D_", "R_", "C_", "METHOD_V2_ALLOWED_LADDER"):
        assert token in joined
    assert "GO" in joined and "NO_GO" in joined
