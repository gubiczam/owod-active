"""Static validation of the Benchmark V1 notebook.

A notebook is JSON that nothing type-checks and that only fails once a GPU has
been paid for. Every test here checks something that would otherwise be
discovered hours into a session: a flag the launcher does not have, a Drive path
used before Drive is mounted, a constant restated in the notebook that has
drifted from the module the results are computed from, an ``!`` shell magic that
swallows a failure, or the one that costs the most — a launcher that runs before
the protocol check and the runtime decision have been printed.

``tools/dry_run_full_owod_benchmark.py`` is the dynamic half: it executes these
same cells. These tests are the half that runs in a second.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from owl.active_selection import arms, benchmark

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "full_owod_active_benchmark_v1.ipynb"

TOTAL_CELLS = 10
PARAMETERS_TAG = "[1/10]"
DRIVE_TAG = "[2/10]"
OWL_TAG = "[3/10]"
PROB_PREFLIGHT_TAG = "[4/10]"
PROB_SETUP_TAG = "[5/10]"
ARTEFACTS_TAG = "[6/10]"
DATA_TAG = "[7/10]"
PLAN_TAG = "[8/10]"
LAUNCHER_TAG = "[9/10]"
SUMMARY_TAG = "[10/10]"


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook):
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def markdown(notebook):
    return "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )


def index_of(code_cells, tag):
    """The cell that *is* the step, matched on its heading, not on a mention."""

    if re.fullmatch(r"\[\d+/\d+\]", tag):
        matches = [i for i, s in enumerate(code_cells) if tag in s.splitlines()[0]]
    else:
        matches = [i for i, s in enumerate(code_cells) if tag in s]
    assert len(matches) == 1, (tag, matches)
    return matches[0]


# ------------------------------------------------------------- well-formed ---


def test_the_notebook_is_valid_json_with_the_expected_shape(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "T4"


def test_every_code_cell_compiles(code_cells):
    for source in code_cells:
        ast.parse(source)


def test_there_are_exactly_the_declared_cells(code_cells):
    assert len(code_cells) == TOTAL_CELLS


def test_the_cells_are_numbered_in_order(code_cells):
    tags = [re.search(r"\[(\d+)/(\d+)\]", s.splitlines()[0]) for s in code_cells]
    assert all(tags), "every code cell needs an [n/m] heading"
    assert [int(t.group(1)) for t in tags] == list(range(1, TOTAL_CELLS + 1))
    assert {int(t.group(2)) for t in tags} == {TOTAL_CELLS}


def test_no_shell_magic_can_swallow_a_failure(code_cells):
    """``!cmd`` and ``%run`` return no status; a failed step would look fine."""

    for source in code_cells:
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("!"), line
            assert not stripped.startswith(("%run", "%%bash", "%%sh")), line


#: The repository's tools, every one of which must be run through `_streamed`.
LONG_STEPS = (
    "prepare_full_owod_benchmark.py", "plan_full_owod_benchmark.py",
    "run_full_owod_benchmark.py", "summarize_full_owod_benchmark.py",
    "plot_full_owod_benchmark.py",
)


def test_every_long_step_streams_its_output(code_cells):
    """A captured step hides the traceback of the step that failed.

    Which is the one thing a 3 a.m. Run all must not do. So every invocation of
    one of this repository's tools must go through ``_streamed``. The exception
    is ``--help``, which the preflight reads as a *string* to check the pinned
    launcher's flags; there is no traceback to lose there.

    Parsed, not grepped: cell 7's docstring explains why ``capture_output`` is
    wrong here, and a substring ban would trip on the note that documents the fix.
    """

    banned = {"run", "Popen", "check_call", "check_output", "_capture", "_checked"}
    for index, source in enumerate(code_cells):
        mentions = any(tool in source for tool in LONG_STEPS)
        if mentions:
            assert "_streamed(" in source, (
                index,
                "a cell that runs one of this repository's tools must stream it",
            )
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            called = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if called not in banned:
                continue
            rendered = ast.unparse(node)
            if "--help" in rendered:
                continue
            for tool in LONG_STEPS:
                assert tool not in rendered, (index, called, tool)


def test_drive_is_mounted_before_any_drive_path_is_used(code_cells):
    mount = index_of(code_cells, DRIVE_TAG)
    for index, source in enumerate(code_cells[:mount]):
        assert "DRIVE /" not in source, index
        assert "RESULTS" not in source or index == 0, index


def test_the_prob_source_is_verified_before_it_is_cloned(code_cells):
    assert index_of(code_cells, PROB_PREFLIGHT_TAG) < index_of(code_cells, PROB_SETUP_TAG)


def test_the_plan_and_the_protocol_check_run_before_the_launcher(code_cells):
    """The runtime decision is printed before any training, not after."""

    plan = index_of(code_cells, PLAN_TAG)
    launcher = index_of(code_cells, LAUNCHER_TAG)
    assert plan < launcher
    assert "check_protocol" in code_cells[plan]
    assert "plan_full_owod_benchmark.py" in code_cells[plan]
    # It may read the launcher's --help to check its flags; it must not run it.
    assert '_streamed([\n' not in code_cells[plan].replace(" ", "")
    for line in code_cells[plan].splitlines():
        if "run_full_owod_benchmark.py" in line:
            assert "--help" in line or "_help" in line or '"tools"' in line, line


def test_the_launcher_is_the_last_thing_before_the_summary(code_cells):
    launcher = index_of(code_cells, LAUNCHER_TAG)
    summary = index_of(code_cells, SUMMARY_TAG)
    assert summary == launcher + 1
    assert "run_full_owod_benchmark.py" in code_cells[launcher]
    assert "summarize_full_owod_benchmark.py" in code_cells[summary]
    assert "plot_full_owod_benchmark.py" in code_cells[summary]


def test_the_data_root_is_built_before_the_launcher(code_cells):
    assert index_of(code_cells, DATA_TAG) < index_of(code_cells, LAUNCHER_TAG)


def test_the_frozen_artefacts_are_checked_before_the_launcher(code_cells):
    artefacts = index_of(code_cells, ARTEFACTS_TAG)
    assert artefacts < index_of(code_cells, LAUNCHER_TAG)
    assert "ref_t1_dinov2_vitb14_cap1000_v1.npz" in code_cells[artefacts]


# ------------------------------------------------------------------- pins ---


def test_the_pins_are_full_shas(code_cells):
    source = code_cells[index_of(code_cells, PARAMETERS_TAG)]
    for name in ("OWL_COMMIT", "PROB_COMMIT"):
        match = re.search(rf'{name} = "([0-9a-f]+)"', source)
        assert match, name
        assert len(match.group(1)) == 40, name


def test_the_prob_pin_is_the_projects_frozen_one(code_cells):
    from owl.bridge import PROB_REPOSITORY

    source = code_cells[index_of(code_cells, PARAMETERS_TAG)]
    assert "4c66be1a52cad9360e09c729e9134aba8fe0b531" in source
    assert PROB_REPOSITORY in source


def test_the_owl_pin_is_a_commit_that_exists_here(code_cells):
    """A stale pin is the failure that cost two overnight sessions."""

    source = code_cells[index_of(code_cells, PARAMETERS_TAG)]
    commit = re.search(r'OWL_COMMIT = "([0-9a-f]{40})"', source).group(1)
    result = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True, check=False,
    )
    assert result.returncode == 0, (
        f"OWL_COMMIT {commit} is not a commit in this repository. Re-pin the "
        "notebook to the commit that carries the code it imports."
    )


def test_the_notebook_names_the_api_it_needs(code_cells):
    """A stale pin must report which symbol it is missing, by name."""

    source = code_cells[index_of(code_cells, OWL_TAG)]
    for name in ("bm.check_protocol", "bm.make_selector", "run_chain.selector",
                 "coverage.kcenter_greedy", "population.p2_reference"):
        assert name in source, name


# -------------------------------------------------- no restated constants ---


def test_the_notebook_restates_no_scientific_constant(code_cells):
    """Every number is read from the pinned module, never typed twice.

    A number typed into a notebook is a number that can drift away from the
    module the results are actually computed from.
    """

    joined = "\n".join(code_cells)
    for value in (str(benchmark.ANSWER_BUDGET_PER_TASK),
                  str(benchmark.CANDIDATE_IMAGES_PER_TASK)):
        # allowed only inside the assertion that pins them to the module
        occurrences = [
            line for line in joined.splitlines()
            if value in line and not line.strip().startswith("#")
        ]
        for line in occurrences:
            assert "bm." in line, f"{value} restated without reading the module: {line}"


def test_the_arms_come_from_the_registry(code_cells):
    source = code_cells[index_of(code_cells, PARAMETERS_TAG)]
    match = re.search(r"SESSION_ARMS = \(([^)]*)\)", source, re.DOTALL)
    assert match
    named = re.findall(r'"([a-z_]+)"', match.group(1))
    assert named, "no arms named"
    assert set(named) <= set(arms.ARMS), named
    assert len(set(named)) == len(named), f"an arm is named twice: {named}"

    # Relative order follows the registry. This used to assert the *prefix* of
    # ORDER, which encoded "a short session still yields the primary contrast" —
    # true while the whole pre-declared order was being run and a timeout would
    # truncate it. A session may now name a deliberate subset (seed-1 replicates
    # the three surviving baselines; `proposed`, `proposed_v2` and `coreset` are
    # excluded for recorded reasons), so the prefix property is obsolete. What
    # must still hold is that nobody reorders the arms.
    positions = [arms.ORDER.index(name) for name in named]
    assert positions == sorted(positions), (
        f"{named} is not in the registry's order {list(arms.ORDER)}"
    )


def test_the_reporting_rules_are_printed_before_the_launcher(code_cells):
    plan = code_cells[index_of(code_cells, PLAN_TAG)]
    assert "bm.REPORTING" in plan


def test_the_endpoints_are_printed_from_the_module(code_cells):
    plan = code_cells[index_of(code_cells, PLAN_TAG)]
    assert "bm.ENDPOINTS.statement()" in plan


def test_the_launcher_refuses_a_stubbed_manifest(code_cells):
    launcher = code_cells[index_of(code_cells, LAUNCHER_TAG)]
    assert 'assert not _manifest.get("dry_run")' in launcher


# ---------------------------------------------------------------- markdown ---


def test_the_markdown_states_what_the_chain_is_not(markdown):
    lowered = markdown.lower()
    assert "one class per task" in lowered
    assert "s-owodb" in lowered
    assert "resum" in lowered


def test_the_markdown_names_the_protocol_and_the_arms(markdown):
    assert "full_owod_active_benchmark_v1_protocol_2026-09-03.md" in markdown
    for arm in arms.ORDER:
        assert arm in markdown, arm


def test_the_launcher_flags_the_notebook_uses_all_exist(code_cells):
    launcher = code_cells[index_of(code_cells, LAUNCHER_TAG)]
    help_text = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_full_owod_benchmark.py"), "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    for flag in re.findall(r'"(--[a-z-]+)"', launcher):
        assert flag in help_text, flag


# ------------------------------------- the committed notebook is launch-ready ---
#
# These pin the operational contract of the file as committed, because the
# failure they guard against is not a wrong number in a table — it is a Colab
# session that dies in cell [1/10] or, worse, runs the wrong arms for seven
# hours.


def test_the_owl_pin_is_a_full_forty_character_sha(code_cells):
    """Cell [1/10] asserts len == 40; a short SHA fails the run immediately."""

    source = code_cells[index_of(code_cells, PARAMETERS_TAG)]
    match = re.search(r'OWL_COMMIT = "([0-9a-f]*)"', source)
    assert match, "OWL_COMMIT is absent or not a plain hex literal"
    assert len(match.group(1)) == 40, (
        f"OWL_COMMIT is {len(match.group(1))} characters, not 40. The notebook's "
        "own assertion refuses a short SHA, so this would fail in [1/10]."
    )
    assert 'assert len(PROB_COMMIT) == 40 and len(OWL_COMMIT) == 40' in source, (
        "the 40-character assertion must stay; it is what catches a short pin"
    )


def test_the_pinned_revision_carries_the_code_the_run_imports(code_cells):
    """The pin may lag HEAD, but only by commits that do not touch ``owl/``.

    This is the reproducibility claim in one assertion: whatever the notebook
    pins must give a byte-identical ``owl/`` to the working tree, so the run
    executes the code this repository currently tests.
    """

    source = code_cells[index_of(code_cells, PARAMETERS_TAG)]
    commit = re.search(r'OWL_COMMIT = "([0-9a-f]{40})"', source).group(1)
    assert subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True, check=False,
    ).returncode == 0, f"{commit} is not a commit in this repository"

    drift = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", commit, "HEAD", "--", "owl/"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert not drift, (
        f"owl/ differs between the pinned {commit[:12]} and HEAD:\n{drift}\n"
        "Re-pin the notebook, or the session runs code this tree no longer has."
    )


def test_the_session_is_the_seed_one_replication(code_cells):
    source = code_cells[index_of(code_cells, PARAMETERS_TAG)]
    named = re.findall(
        r'"([a-z_]+)"',
        re.search(r"SESSION_ARMS = \(([^)]*)\)", source, re.DOTALL).group(1),
    )
    assert named == ["random", "admissibility", "entropy"], named
    seeds = re.search(r"SEEDS = \(([^)]*)\)", source).group(1)
    assert seeds.strip().rstrip(",") == "1", seeds


def test_no_excluded_arm_can_launch(code_cells):
    """`proposed`, `proposed_v2` and `coreset` may be *named* but not selected."""

    source = code_cells[index_of(code_cells, PARAMETERS_TAG)]
    body = re.search(r"SESSION_ARMS = \(([^)]*)\)", source, re.DOTALL).group(1)
    for excluded in ("proposed", "proposed_v2", "coreset"):
        assert f'"{excluded}"' not in body, f"{excluded} is in SESSION_ARMS"
    # and the launcher is handed SESSION_ARMS, not a literal list of arms
    launcher = code_cells[index_of(code_cells, LAUNCHER_TAG)]
    assert "*SESSION_ARMS" in launcher or "SESSION_ARMS" in launcher


def test_bridge_is_imported_before_any_cell_uses_it(code_cells):
    """The `NameError: name 'bridge' is not defined` class of failure."""

    imports = [i for i, s in enumerate(code_cells)
               if re.search(r"^from owl import .*\bbridge\b", s, re.MULTILINE)]
    uses = [i for i, s in enumerate(code_cells)
            if re.search(r"(?<!import )\bbridge\.", s)]
    assert imports, "no cell imports `bridge`"
    assert uses, "no cell uses `bridge` — this test would be vacuous"
    assert min(imports) < min(uses), (
        f"`bridge` is used in cell {min(uses)} but only imported in "
        f"{min(imports)}; a clean Run all would raise NameError"
    )


def test_the_first_bridge_user_names_the_cell_it_needs(code_cells):
    """A bare NameError says nothing; this says which cell to run."""

    first = min(i for i, s in enumerate(code_cells)
                if re.search(r"(?<!import )\bbridge\.", s))
    assert '"bridge" not in globals()' in code_cells[first]
    assert "[3/10]" in code_cells[first]


def test_the_time_budget_covers_the_priced_session(code_cells):
    """Three baseline arms at ~2.23 h each is ~6.7 h; the cap must exceed it."""

    source = code_cells[index_of(code_cells, PARAMETERS_TAG)]
    budget = int(re.search(r"TIME_BUDGET_MINUTES = (\d+)", source).group(1))
    assert budget >= 420, f"{budget} min is under the ~6.7 h this session needs"


def test_no_cell_advertises_a_stale_runtime(code_cells, markdown):
    """A nine-hour banner on a seven-hour session is operationally misleading."""

    joined = "\n".join(code_cells) + markdown
    assert "nine hours" not in joined
