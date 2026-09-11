"""The V2 notebook: its parameter cell, its pre-registration, and Run-all safety.

A notebook is the one artefact unit tests normally cannot see, and this project
has lost two GPU sessions to notebook defects that every module passed. So the
notebook's *text* is asserted here — the config cell it ships with, the pin it
refuses to run without, the flags it hands the launcher — and its *execution* is
asserted by ``tools/dry_run_owod_full_chain_v2.py``, which runs every cell with
PROB and Colab faked.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from owl import replay, supervision
from owl.active_selection import arms
from owl.active_selection import v2 as v2_protocol

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "OWOD_FULL_CHAIN_V2.ipynb"
PROTOCOL = ROOT / "docs" / "full_owod_v2_protocol.md"


@pytest.fixture(scope="module")
def cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


@pytest.fixture(scope="module")
def code(cells) -> list[str]:
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def _run_cell(code: list[str]) -> str:
    """The cell that actually invokes the launcher.

    Discriminated by the argument it *passes*, not by the script's name or by a
    flag's name: the preflight cell names the script too — it reads its `--help`
    to check the flags exist — and it lists every flag name as a string, so both
    "run_full_owod_benchmark.py" and '"--out"' match two cells.
    """

    matches = [c for c in code if '"--prob-root", str(PROB)' in c]
    assert len(matches) == 1, f"{len(matches)} cells invoke the launcher"
    return matches[0]


@pytest.fixture(scope="module")
def parameters(code) -> dict[str, object]:
    """The literal assignments of the parameter cell, parsed not executed.

    Parsing rather than exec'ing is deliberate: the cell imports torch-adjacent
    things and asserts on a placeholder SHA, and what is under test is the
    values it *declares*.
    """

    tree = ast.parse(code[0])
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.isupper():
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return values


# ------------------------------------------------------------ the config cell ---


def test_the_config_cell_names_every_axis_the_brief_asks_for(parameters):
    for name in ("TASK_END", "SEEDS", "ANNOTATION_POLICY", "REPLAY_MODE",
                 "REPLAY_REFRESH", "COHERENCE_MODE", "DIVERSITY_MODE",
                 "ACQUISITION_BATCH_SIZE"):
        assert name in parameters, name


def test_every_declared_mode_is_a_registered_one(parameters):
    assert parameters["ANNOTATION_POLICY"] in supervision.POLICIES
    assert parameters["IGNORE_MECHANISM"] in supervision.IGNORE_MECHANISMS
    assert parameters["REPLAY_MODE"] in replay.MODES
    assert parameters["REPLAY_REFRESH"] in replay.REFRESH
    from owl.active_selection import research_score

    assert parameters["COHERENCE_MODE"] in research_score.COHERENCE_MODES
    assert parameters["DIVERSITY_MODE"] in research_score.DIVERSITY_MODES
    assert set(parameters["SESSION_ARMS"]) <= set(arms.ARMS)


def test_the_shipped_config_agrees_with_the_protocol_document(parameters):
    """The same comparison cell [4/12] makes, made here so CI catches a drift."""

    configuration = v2_protocol.Configuration(
        task_end=int(parameters["TASK_END"]),
        seeds=tuple(parameters["SEEDS"]),
        annotation_policy=str(parameters["ANNOTATION_POLICY"]),
        ignore_mechanism=str(parameters["IGNORE_MECHANISM"]),
        replay_mode=str(parameters["REPLAY_MODE"]),
        replay_refresh=str(parameters["REPLAY_REFRESH"]),
        coherence_mode=str(parameters["COHERENCE_MODE"]),
        diversity_mode=str(parameters["DIVERSITY_MODE"]),
        acquisition_batch_size=int(parameters["ACQUISITION_BATCH_SIZE"]),
        answer_budget=int(parameters["ANSWER_BUDGET"]),
        arms=tuple(parameters["SESSION_ARMS"]),
        phase=str(parameters["PHASE"]),
    )
    agreement = v2_protocol.check(configuration)
    assert agreement.agrees, agreement.statement()
    assert set(configuration.arms) <= set(
        v2_protocol.phase(configuration.phase).arms), (
        "the notebook ships arms this phase did not pre-register")


def test_the_shipped_config_is_phase_a(parameters):
    """What ships is the cheap validation pass, not the expensive comparison.

    Opening the notebook and pressing Run all should not start a three-seed
    ten-task chain by accident.
    """

    assert parameters["PHASE"] == "A"
    assert parameters["SEEDS"] == [0]
    assert int(parameters["TASK_END"]) <= v2_protocol.phase("A").task_end


def test_the_owl_pin_is_a_placeholder_that_refuses_to_run(parameters, code):
    """A notebook cannot hold the SHA of the commit containing it.

    So it ships a placeholder and stops in the first cell. The alternative — a
    stale SHA from a previous commit — would run silently against code that is
    not the code under test.
    """

    assert not re.fullmatch(r"[0-9a-f]{40}", str(parameters["OWL_COMMIT"]))
    assert 'assert len(OWL_COMMIT) == 40' in code[0]
    assert re.fullmatch(r"[0-9a-f]{40}", str(parameters["PROB_COMMIT"])), (
        "the PROB pin is a real commit and must stay one")


def test_it_writes_somewhere_no_committed_result_lives(parameters):
    assert parameters["RESULTS_RELATIVE"] == "results/full_owod_chain_v2"
    for occupied in ("full_owod_active_benchmark_v1", "full_owod_chain_t10"):
        assert occupied not in str(parameters["RESULTS_RELATIVE"])


# -------------------------------------------------------------- the launcher ---


def test_every_v2_flag_the_notebook_passes_exists_on_the_launcher(code):
    """The preflight checks this at run time; this checks it at commit time."""

    run_cell = _run_cell(code)
    passed = set(re.findall(r'"(--[a-z0-9-]+)"', run_cell))

    import subprocess
    import sys

    help_text = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_full_owod_benchmark.py"), "--help"],
        check=True, capture_output=True, text=True).stdout
    missing = sorted(flag for flag in passed if flag not in help_text)
    assert not missing, missing


def test_the_launcher_flags_of_the_configuration_round_trip():
    flags = v2_protocol.phase("B").launcher_flags()
    assert "--annotation-policy" in flags
    assert "--acquisition-batch-size" in flags
    # arms in the registry's order, never the caller's
    index = flags.index("--arms")
    named = []
    for value in flags[index + 1:]:
        if value.startswith("--"):
            break
        named.append(value)
    assert named == [a for a in arms.ORDER if a in set(named)]


# --------------------------------------------------------------- Run-all safety ---


def test_every_name_is_defined_before_it_is_used():
    """The static check that replaces a 3 a.m. NameError in cell nine."""

    import subprocess
    import sys

    finished = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "audit_notebook_dataflow.py"),
         "--quiet", str(NOTEBOOK)],
        check=False, capture_output=True, text=True)
    assert finished.returncode == 0, finished.stdout + finished.stderr


def test_the_run_cell_is_resumable_and_says_so(code):
    run_cell = _run_cell(code)
    assert "resume" in run_cell.lower()
    # The guard that a stubbed manifest can never be read as a result.
    assert 'assert not _manifest.get("dry_run")' in run_cell


def test_the_evaluation_split_is_named_for_this_chain_length(code):
    """The defect this notebook was written after finding.

    `prepare_full_owod_benchmark.py --n-tasks N` writes ONLY the split for N,
    and `evaluation_subset.SHARED_TEST_SET` is Benchmark V1's four-task name. A
    cell that prepares a five-task chain and then asserts the four-task file
    exists dies on a fresh `/content` — which is what the committed t10 notebook
    did, while asserting two cells earlier that the two names must differ.
    """

    for notebook in (NOTEBOOK, ROOT / "notebooks" / "full_owod_chain_t10.ipynb"):
        text = notebook.read_text(encoding="utf-8")
        assert "evaluation_subset.SHARED_TEST_SET}.txt" not in text, notebook.name
        assert "shared_test_set_name(N_TASKS)}.txt" in text, notebook.name


def test_the_markdown_says_what_may_not_be_claimed(cells):
    markdown = "\n".join(
        "".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    assert "full_owod_v2_protocol.md" in markdown
    for claim in ("S-OWODB", "Not a re-run", "Not a tuned method"):
        assert claim in markdown, claim
    # Phase A must not be presented as a comparison.
    assert "not an endpoint comparison" in markdown


def test_the_protocol_document_freezes_what_the_code_declares():
    values = v2_protocol.frozen(PROTOCOL)
    from owl.active_selection import research_score

    assert values["lambda_diversity"] == research_score.LAMBDA_DIVERSITY
    assert values["gamma_rarity"] == research_score.GAMMA_RARITY
    assert values["eps_quantile"] == research_score.EPS_QUANTILE
    assert values["pca_dimensions"] == research_score.PCA_DIMENSIONS
    for arm in values["phase_a_arms"] + values["phase_b_arms"]:
        assert arm in arms.ARMS, arm
    # The primary arm's registered spec is the protocol's spec.
    spec = arms.ARMS["research_v2"].score_spec
    assert spec.diversity_mode == values["diversity_mode"]
    assert spec.coherence_mode == values["coherence_mode"]
    assert spec.rarity_mode == values["rarity_mode"]


def test_the_protocol_declares_its_failure_criteria():
    """A pre-registration without failure criteria is not one."""

    text = PROTOCOL.read_text(encoding="utf-8")
    for required in ("Failure criteria", "Primary endpoint", "decision rule",
                     "U_Recall_tail", "reported,", "not tuned"):
        assert required in text, required
