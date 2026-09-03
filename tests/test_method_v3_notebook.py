"""Static validation of the overnight Method V3 notebook.

A notebook is JSON that nothing type-checks and that only fails once a GPU has
been paid for. These tests read the committed file and check the things that
would otherwise be discovered eight hours in: a flag the launcher does not have,
a Drive path used before Drive is mounted, a threshold restated in the notebook
that has drifted from the module the verdict is computed from, and — the one that
matters most tonight — a training launcher that runs before the success criterion
has been printed.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from owl import method_v3

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "method_v3_selection_transfer_overnight.ipynb"

#: The cell that starts the twelve real trajectories. Everything that must
#: happen first is expressed relative to this index.
LAUNCHER_TAG = "[8/9]"
TOTAL_CELLS = 9


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook):
    return [
        "".join(cell["source"])
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]


@pytest.fixture(scope="module")
def markdown(notebook):
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )


def executable_source(cell: str) -> str:
    """The cell with its comments removed, for bans that are about code.

    A ban on ``:g}`` has to be a ban on *formatting a value*, not on a comment
    explaining why that formatting was a bug — otherwise the note that documents
    the fix trips the test that enforces it.
    """

    import io
    import tokenize

    kept = []
    for token in tokenize.generate_tokens(io.StringIO(cell).readline):
        if token.type != tokenize.COMMENT:
            kept.append(token.string)
    return "\n".join(kept)


def quotes(text: str) -> str:
    """Compare source text without caring which quote character was used.

    ``repr`` gives single quotes and the notebook is written with double ones;
    normalising both sides keeps the comparison about the *values* rather than
    about a formatting choice.
    """

    return text.replace("'", '"')


def index_of(code_cells, tag):
    """The cell that *is* the step, matched on its own heading, not on a mention.

    A later cell may talk about ``[8/9]`` in a comment; the heading is what
    identifies the cell, so only the first line counts for a ``[n/9]`` tag.
    """

    if re.fullmatch(r"\[\d+/\d+\]", tag):
        matches = [i for i, source in enumerate(code_cells)
                   if tag in source.splitlines()[0]]
    else:
        matches = [i for i, source in enumerate(code_cells) if tag in source]
    assert len(matches) == 1, (tag, matches)
    return matches[0]


# ------------------------------------------------------------- well-formed ---


def test_the_notebook_is_valid_json_with_the_expected_shape(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "T4"


def test_every_code_cell_compiles(code_cells):
    for index, source in enumerate(code_cells):
        ast.parse(source)  # raises with the cell's own line number


def test_no_shell_magic_can_swallow_a_failure(code_cells):
    """``!cmd`` and ``%run`` return no status; a failed step would look fine."""

    for source in code_cells:
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("!"), line
            assert not stripped.startswith(("%run", "%%bash", "%%sh")), line


def test_the_cells_are_numbered_in_order(code_cells):
    tags = [
        re.search(r"\[(\d+)/(\d+)\]", source).groups()
        for source in code_cells
        if re.search(r"\[(\d+)/(\d+)\]", source)
    ]
    assert [int(n) for n, _ in tags] == list(range(1, TOTAL_CELLS + 1))
    assert {int(total) for _, total in tags} == {TOTAL_CELLS}


def test_the_title_says_how_to_run_it(markdown):
    for needle in ("METHOD V3", "SELECTION → LEARNING TRANSFER",
                   "OVERNIGHT ONE-CLICK EXPERIMENT", "T4 GPU",
                   "Runtime → Run all", "Come back tomorrow"):
        assert needle in markdown, needle


def test_the_title_does_not_reopen_method_v2(markdown):
    assert "D_NO_GO" in markdown and "C_GO" in markdown
    assert "stands unchanged" in markdown
    assert "exploratory" in markdown.lower()


# ------------------------------------------------------------------- order ---


def test_drive_is_mounted_before_any_drive_path_is_used(code_cells):
    mount = index_of(code_cells, "drive.mount")
    for index, source in enumerate(code_cells):
        if index <= mount:
            continue
        if "/content/drive" in source:
            continue                       # the parameter cell only names it
        assert True
    for index, source in enumerate(code_cells[:mount]):
        assert "DRIVE /" not in source and "FEATURES /" not in source, index


def test_the_repository_is_pinned_and_installed_before_owl_is_imported(code_cells):
    pin = index_of(code_cells, "ensure_pinned_checkout(Path(\"/content/owod-active\")")
    imports = [i for i, s in enumerate(code_cells) if "from owl import" in s]
    assert imports and min(imports) == pin


def test_prob_is_pinned_and_smoke_tested_before_the_launcher(code_cells):
    prob = index_of(code_cells, "ensure_pinned_checkout(Path(\"/content/PROB\")")
    launcher = index_of(code_cells, LAUNCHER_TAG)
    assert prob < launcher
    assert "PROB CUDA model/loss/evaluator smoke" in code_cells[prob]
    assert "ENVIRONMENT_PREFLIGHT_OK = True" in code_cells[prob]


def test_the_data_root_is_built_before_the_launcher_and_before_it_is_asserted(code_cells):
    prepare = index_of(code_cells, "prepare_method_v3_data.py")
    launcher = index_of(code_cells, LAUNCHER_TAG)
    assert prepare < launcher
    for index, source in enumerate(code_cells[:prepare]):
        assert 'DATA / "Annotations"' not in source, index


def test_the_frozen_artefacts_are_checked_before_the_launcher(code_cells):
    artefacts = index_of(code_cells, "BASE_EXPORT = FEATURES /")
    launcher = index_of(code_cells, LAUNCHER_TAG)
    assert artefacts < launcher
    assert "assert not missing" in code_cells[artefacts]


def test_the_criterion_is_printed_before_the_training_launcher(code_cells):
    """The explicit requirement: show the criterion before real training runs."""

    criterion = index_of(code_cells, "CRITERION_STATEMENT")
    launcher = index_of(code_cells, LAUNCHER_TAG)
    assert criterion < launcher
    cell = code_cells[criterion]
    assert "method_v3.CRITERION.statement()" in cell
    assert "FROZEN CRITERION" in cell


def test_the_protocol_is_checked_structurally_not_by_prose_matching(code_cells):
    """The regression that stopped an overnight run before it trained anything.

    The notebook used to search the protocol document for
    ``f"{guard_tolerance:g} AP50 point"``. ``f"{1.0:g}"`` is ``"1"`` and the
    document says ``"1.0"``, so a correct, frozen criterion failed a
    documentation check. The notebook must delegate to
    ``owl.method_v3.check_protocol_criterion``, which compares values.
    """

    cell = code_cells[index_of(code_cells, "CRITERION_STATEMENT")]
    assert "method_v3.check_protocol_criterion()" in cell
    assert "_protocol_text" not in cell
    assert "read_text" not in cell


def test_no_cell_searches_a_document_for_a_formatted_criterion_value(code_cells):
    """Bans the whole class, not just the one phrase that failed.

    Two things are forbidden anywhere in the notebook: formatting a criterion
    field into a string with a format spec (``:g`` silently drops ``.0``), and
    substring-searching a document for a value that came out of the module.
    """

    joined = "\n".join(executable_source(source) for source in code_cells)
    for banned in (":g}", ":.1f}", ":.2f}",
                   "AP50 point", "of the 3", "of the 2"):
        assert banned not in joined, banned
    for field in method_v3.CRITERION.__dataclass_fields__:
        assert f"CRITERION.{field}:" not in joined, field
    assert "_protocol_text" not in joined
    assert "_needle" not in joined


def test_the_notebook_never_hardcodes_the_criterion_values(code_cells):
    """No criterion field name or number may be typed into the notebook at all."""

    joined = "\n".join(executable_source(source) for source in code_cells)
    for value in ("mAP50_medium_tail", "known_mAP50 tolerance",
                  "guard_tolerance", "minimum_improving_seeds"):
        assert value not in joined, value


def test_the_summariser_runs_after_the_launcher_and_exactly_once(code_cells):
    launcher = index_of(code_cells, LAUNCHER_TAG)
    summarise = index_of(code_cells, "summarize_method_v3.py")
    assert summarise > launcher
    assert sum(s.count("summarize_method_v3.py") for s in code_cells) == 1


def test_the_launcher_is_invoked_from_exactly_one_cell(code_cells):
    """One launch site. A second one would double-charge the night."""

    invocations = [
        index for index, source in enumerate(code_cells)
        if 'str(ROOT / "tools" / "run_method_v3.py")' in source
    ]
    assert invocations == [index_of(code_cells, LAUNCHER_TAG)]
    assert sum(source.count("_launch = [") for source in code_cells) == 1


# ------------------------------------------------------- the actual CLIs ---


def scripts_called(code_cells):
    found = set()
    for source in code_cells:
        found.update(re.findall(r'"tools" / "([a-z0-9_]+\.py)"', source))
    return found


def test_every_script_the_notebook_calls_exists(code_cells):
    called = scripts_called(code_cells)
    assert called, "no tool call found at all"
    for name in called:
        assert (ROOT / "tools" / name).is_file(), name


def test_every_flag_the_notebook_passes_exists_in_that_script(code_cells):
    """Checked against each script's own argparse, not against a copy of it."""

    for source in code_cells:
        for match in re.finditer(
            r'"tools" / "([a-z0-9_]+\.py)"\)?,?\n?(.*?)\]', source, re.DOTALL
        ):
            name, arguments = match.group(1), match.group(2)
            flags = set(re.findall(r'"(--[a-z0-9-]+)"', arguments))
            if not flags:
                continue
            help_text = subprocess.run(
                [sys.executable, str(ROOT / "tools" / name), "--help"],
                capture_output=True, text=True, check=True, cwd=ROOT,
            ).stdout
            for flag in flags:
                assert flag in help_text, (name, flag)


def test_the_launcher_is_given_every_required_flag(code_cells):
    launcher = code_cells[index_of(code_cells, LAUNCHER_TAG)]
    for flag in ("--prob-root", "--data-root", "--checkpoint", "--export",
                 "--views", "--out", "--time-budget-minutes"):
        assert flag in launcher, flag
    assert "--dry-run" not in launcher, "the overnight cell must not stub PROB out"


def test_the_launcher_reads_the_manifest_and_refuses_a_stubbed_one(code_cells):
    launcher = code_cells[index_of(code_cells, LAUNCHER_TAG)]
    assert 'MANIFEST["dry_run"] is False' in launcher
    assert f"len(COMPLETE) == {len(method_v3.trajectories())}" in launcher


# -------------------------------------------------------------- the pins ---


def test_a_full_forty_character_sha_is_pinned_for_prob(code_cells):
    source = code_cells[0]
    prob = re.search(r'PROB_COMMIT = "([0-9a-f]+)"', source)
    assert prob and len(prob.group(1)) == 40
    assert prob.group(1) == "4c66be1a52cad9360e09c729e9134aba8fe0b531"


def test_a_full_forty_character_sha_is_pinned_for_owl(code_cells):
    source = code_cells[0]
    owl = re.search(r'OWL_COMMIT = "([0-9a-f]{40})"', source)
    assert owl, "OWL_COMMIT must be a full 40-character SHA"
    assert 'len(OWL_COMMIT) == 40' in source


def test_the_notebook_asserts_the_frozen_design_from_the_module(code_cells):
    """The notebook's own assertions are compared against the live module.

    Written this way so that changing ``owl.method_v3`` without changing the
    notebook fails here, rather than the test merely checking that some string
    resembling the design appears somewhere.
    """

    joined = quotes("\n".join(code_cells))
    assert quotes(f"method_v3.ARMS == {method_v3.ARMS!r}") in joined
    assert quotes(f"method_v3.SEEDS == {method_v3.SEEDS!r}") in joined
    assert (f"len(method_v3.trajectories()) == {len(method_v3.trajectories())}"
            in joined)


def test_no_scientific_threshold_is_restated_in_the_notebook(code_cells):
    """Every number the verdict depends on is read from the pinned module.

    The Stage-2 notebook restated 0.65 and 0.60 as display text and they could
    have drifted from ``owl.method_v2_stage2``. Same rule here: the tolerance,
    the improving-seed count and the metric names appear only as attribute
    reads.
    """

    joined = "\n".join(code_cells)
    forbidden = (
        "guard_tolerance = 1.0", "= 1.0  #", "mAP50_medium_tail(A*C)",
        "at least 2 of 3", "0.65", "0.6101", "0.6411",
    )
    for needle in forbidden:
        assert needle not in joined, needle
    assert "method_v3.CRITERION" in joined


def test_the_budget_and_the_arms_are_not_typed_as_literals(code_cells):
    joined = "\n".join(code_cells)
    assert "BUDGET_PER_TASK = 600" not in joined
    assert "EPOCHS = 5" not in joined
    assert "REPLAY_ARM" not in joined or "method_v3.REPLAY_ARM" in joined


# ------------------------------------------------------ resume and Drive ---


def test_a_gpu_is_required(code_cells):
    joined = "\n".join(code_cells)
    assert "cuda_available" in joined or "torch.cuda.is_available()" in joined
    assert "CUDA is unavailable" in joined


def test_every_expensive_output_lands_on_drive(code_cells):
    joined = "\n".join(code_cells)
    assert 'RESULTS_RELATIVE = "results/method_v3_selection_transfer"' in joined
    assert "RESULTS = DRIVE / RESULTS_RELATIVE" in joined
    assert '"--out", str(RESULTS)' in joined


def test_nothing_is_assumed_to_survive_a_previous_session(code_cells):
    """``/content`` is wiped between sessions; only Drive persists."""

    joined = "\n".join(code_cells)
    assert "ensure_pinned_checkout" in joined            # repo re-cloned
    assert "prepare_method_v3_data.py" in joined         # data root rebuilt
    assert "force_remount=False" in joined               # Drive re-mounted


def test_the_run_is_resumable_and_says_so(code_cells):
    launcher = code_cells[index_of(code_cells, LAUNCHER_TAG)]
    assert "resume" in launcher.lower()
    assert "--time-budget-minutes" in launcher
    assert "Run all again" in launcher


def test_a_partial_run_stops_with_an_actionable_message(code_cells):
    launcher = code_cells[index_of(code_cells, LAUNCHER_TAG)]
    assert "still to run" in launcher
    assert "incomplete design" in launcher


def test_the_final_cell_prints_the_verdict_and_the_pins(code_cells):
    final = code_cells[-1]
    assert 'SUMMARY["verdict"]' in final
    assert "OWL SHA" in final and "PROB SHA" in final
    assert "checkpoint sha256" in final
    assert "D_NO_GO, R_NO_GO, C_GO, allowed ladder U" in final
    assert 'SUMMARY["dry_run"] is False' in final


def test_the_verdict_can_only_be_one_of_the_two_frozen_strings(code_cells):
    labels = ("C_DOWNSTREAM_POSITIVE", "C_DOWNSTREAM_NOT_SUPPORTED")
    assert quotes(f"{labels!r}") in quotes(code_cells[-1])
    # and both are really the two the module can produce
    from owl.method_v3 import Verdict
    assert Verdict("C_DOWNSTREAM_POSITIVE", {}, {}).positive
    assert not Verdict("C_DOWNSTREAM_NOT_SUPPORTED", {}, {}).positive


def test_no_outcome_dependent_branching_around_the_launcher(code_cells):
    """No cell may choose what to run based on a metric it has just seen."""

    launcher_index = index_of(code_cells, LAUNCHER_TAG)
    for source in code_cells[: launcher_index + 1]:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            text = ast.dump(node)
            for metric in ("mAP50", "U_Recall", "AP50", "verdict", "medium_tail"):
                assert metric not in text, metric


def test_the_notebook_names_no_arm_conditionally(code_cells):
    """All four arms are scheduled by owl.method_v3, not by the notebook."""

    joined = "\n".join(code_cells)
    assert "--only-arm" not in joined
    assert "--only-seed" not in joined
