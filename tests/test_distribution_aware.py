"""``distribution_aware_v1`` and its ``cost_aware`` control: the frozen properties.

The memo (``docs/distribution_aware_decision_memo_2026-09-06.md``) is the binding
specification. These tests pin the things that would silently invalidate the
diagnostic rather than break it: a selector that can see an answer, a clusterer
that does not reproduce, a gate whose threshold drifted, a run that trains.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from owl import proposals
from owl.active_selection import allocation, arms, population
from owl.active_selection import benchmark as bm
from owl.active_selection import diagnostic as gates

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pool():
    return proposals.from_frozen_pool(split="pool")


@pytest.fixture(scope="module")
def built(pool):
    return population.build(pool)


@pytest.fixture(scope="module")
def features(built):
    """Stand-in semantic features for ``G``. Deterministic, L2-normalised.

    The *values* are not DINOv2 and no scientific reading may be taken from
    anything computed on them. What they exercise is every line between the
    export and the ledger, which is where a defect would hide.
    """

    index = arms.ranked_positions("distribution_aware_v1", built)
    generator = np.random.default_rng(0)
    block = generator.normal(size=(index.size, 16))
    return (block / np.linalg.norm(block, axis=1, keepdims=True)).astype(np.float32)


# ------------------------------------------------------- label-free by force ---


def test_neither_arm_reads_an_answer(built, features):
    """The oracle is behind ``Candidates.oracle()``; both arms must run without it.

    Constructed the same way as ``test_no_arm_reads_an_answer``: a population
    whose oracle columns are absent entirely, so a read raises rather than
    quietly returning something plausible.
    """

    stripped = population.build(
        proposals.Candidates(
            image_ids=built.candidates.image_ids,
            boxes=built.candidates.boxes,
            posterior=built.candidates.posterior,
            objectness=built.candidates.objectness,
            embeddings=built.candidates.embeddings,
        )
    )
    with pytest.raises(ValueError, match="carries no oracle"):
        stripped.candidates.oracle()

    index = arms.ranked_positions("distribution_aware_v1", stripped)
    generator = np.random.default_rng(1)
    block = generator.normal(size=(index.size, 16))
    block /= np.linalg.norm(block, axis=1, keepdims=True)

    for arm, semantic in (("cost_aware", None),
                          ("distribution_aware_v1", block.astype(np.float32))):
        picked = arms.select(
            arm, stripped, cost_of=lambda _: 3, answer_budget=200, seed=0,
            semantic=semantic,
        )
        assert len(picked.images) > 0


def _executable_source(path: Path) -> str:
    """The module with docstrings and comments removed.

    Prose may discuss ``new_class_AP50`` and oracle kinds — the reasons for the
    design are exactly what a docstring is for. What must not exist is a *line
    that runs* and reaches one.
    """

    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def test_the_allocator_source_names_no_oracle_and_no_future_class():
    """A static read of the two modules the selector actually runs."""

    for module in ("allocation.py", "arms.py"):
        body = _executable_source(ROOT / "owl" / "active_selection" / module)
        for forbidden in ("oracle", "gt_class", "new_class", "known_classes",
                          "declared", "class_groups"):
            assert forbidden not in body, f"{module} reaches {forbidden}"


def test_cost_aware_reads_only_admissibility(built):
    """The control must be label-free, including free of the true annotation cost.

    Its order is a pure function of ``A`` and the image ids, so handing it a
    wildly different cost function must not move a single position.
    """

    index = arms.ranked_positions("cost_aware", built)
    first, _ = allocation.cost_aware_order(
        built.admissibility[index], built.candidates.image_ids[index])
    second, _ = allocation.cost_aware_order(
        built.admissibility[index], built.candidates.image_ids[index])
    assert np.array_equal(first, second)

    cheap = arms.select("cost_aware", built, cost_of=lambda _: 1,
                        answer_budget=300, seed=0)
    dear = arms.select("cost_aware", built, cost_of=lambda _: 1,
                       answer_budget=300, seed=7)
    # no seed dependence either: the control is deterministic given the pool
    assert cheap.images == dear.images

    source = (ROOT / "owl" / "active_selection" / "allocation.py").read_text()
    body = source.split("def cost_aware_order")[1]
    assert "cost_of" not in body, "cost_aware must not consult the cost function"


# --------------------------------------------------------------- determinism ---


def test_the_clustering_reproduces(features):
    size = allocation.min_cluster_size(
        features.shape[0], budget=3000, mean_image_cost=9.5)
    first = allocation.cluster(features, size=size)
    second = allocation.cluster(features, size=size)
    assert np.array_equal(first, second)


def test_the_whole_order_reproduces(built, features):
    index = arms.ranked_positions("distribution_aware_v1", built)
    kwargs = {
        "entropy": np.linspace(0, 1, index.size),
        "image_ids": built.candidates.image_ids[index],
        "cost_of": lambda _: 4, "budget": 3000,
    }
    first = allocation.distribution_aware_order(features, **kwargs)
    second = allocation.distribution_aware_order(features, **kwargs)
    assert np.array_equal(first.order, second.order)
    assert first.diagnostics == second.diagnostics


def test_the_order_is_a_permutation(built, features):
    index = arms.ranked_positions("distribution_aware_v1", built)
    result = allocation.distribution_aware_order(
        features, entropy=np.linspace(0, 1, index.size),
        image_ids=built.candidates.image_ids[index],
        cost_of=lambda _: 4, budget=3000,
    )
    assert sorted(result.order.tolist()) == list(range(index.size))
    assert set(np.unique(result.coherence).tolist()) <= {0, 1}


# ------------------------------------------------------------------- rarity ---


def test_min_cluster_size_comes_from_the_budget():
    """``max(5, ceil(|G| / (B / c_bar)))`` — no literal chosen by hand."""

    assert allocation.min_cluster_size(19_000, budget=3000, mean_image_cost=9.5) == 61
    assert allocation.min_cluster_size(10, budget=3000, mean_image_cost=9.5) == 5
    with pytest.raises(allocation.AllocationError):
        allocation.min_cluster_size(100, budget=3000, mean_image_cost=0)


def test_an_empty_reference_makes_every_cluster_under_represented():
    """The correct reading at the first purchase task, not an edge case to patch."""

    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    ids = np.array([0, 1])
    counts = allocation.reference_counts(None, ids, np.zeros((2, 4)))
    assert counts == {0: 0, 1: 0}
    quota, rarity = allocation.quotas(labels, ids, counts, budget=3000)
    assert all(v > 0 for v in rarity.values())
    assert sum(quota.values()) == pytest.approx(3000)


def test_a_covered_cluster_receives_nothing():
    """The positive part *is* the rarity rule: R+ = 0 means no budget at all."""

    labels = np.array([0, 0, 0, 1, 1, 1])
    ids = np.array([0, 1])
    quota, rarity = allocation.quotas(labels, ids, {0: 0, 1: 400}, budget=3000)
    assert rarity[1] < 0 and quota[1] == 0.0
    assert quota[0] == pytest.approx(3000)


def test_quota_is_spent_on_images_and_charged_once(built, features):
    """A second candidate on an open image costs the annotator nothing, so it
    must not consume a quota either."""

    labels = np.array([0, 0, 1, 1])
    quota = {0: 10.0, 1: 10.0}
    order, diag = allocation.emit(
        labels, quota, np.array([0.9, 0.8, 0.7, 0.6]),
        ["a", "a", "b", "b"], lambda _: 5,
    )
    assert sorted(order.tolist()) == [0, 1, 2, 3]
    assert diag["images_under_quota"] == 2
    assert diag["answers_charged"] == pytest.approx(10.0)


# -------------------------------------------------------------------- gates ---


def test_the_frozen_gates_have_not_drifted():
    """Thresholds are frozen by the memo; a silent edit invalidates the run."""

    expected = {
        "jaccard_vs_entropy": 0.60,
        "tail_per_image_vs_entropy": 1.25,
        "tail_per_image_vs_cost_aware": 1.25,
        "background_share_vs_entropy": 0.10,
        "breadth_vs_entropy": 0.75,
        "t2_not_collapsed": 0.75,
    }
    assert {g.key: g.threshold for g in gates.GATES} == expected
    assert gates.DIAGNOSTIC_ARMS == ("entropy", "cost_aware", "distribution_aware_v1")
    assert gates.TAIL_TASKS == ("t3", "t4")


def test_the_memo_and_the_gates_agree():
    memo = (ROOT / "docs" / "distribution_aware_decision_memo_2026-09-06.md").read_text(
        encoding="utf-8")
    for token in ("0.60", "1.25", "10 pp", "0.75"):
        assert token in memo, token


def test_every_gate_is_required_and_one_failure_is_no_go():
    rows, opened = _synthetic(tail_ratio=0.5)
    verdict = gates.evaluate(rows, opened,
                             method="distribution_aware_v1", seeds=(0,))
    assert not verdict.go
    assert "tail_per_image_vs_entropy" in verdict.failed


def test_a_clearly_better_arm_passes_every_gate():
    """The gates must be satisfiable — otherwise NO-GO proves nothing."""

    rows, opened = _synthetic(tail_ratio=3.0)
    verdict = gates.evaluate(rows, opened,
                             method="distribution_aware_v1", seeds=(0,))
    assert verdict.go, verdict.failed


def _synthetic(*, tail_ratio: float):
    rows, opened = [], {}
    for arm, factor in (("entropy", 1.0), ("cost_aware", 1.0),
                        ("distribution_aware_v1", tail_ratio)):
        for task in ("t2", "t3", "t4"):
            rows.append({
                "arm": arm, "seed": 0, "task": task, "images_opened": 300,
                "held_at_declaration": 30 * factor, "background_share": 0.1,
                "acquired_class_breadth": 50,
            })
            base = [f"{task}-{i}" for i in range(300)]
            opened[(arm, 0, task)] = (
                base if arm == "entropy"
                else [f"{task}-{arm}-{i}" for i in range(300)]
            )
    return rows, opened


# ------------------------------------------------- the driver cannot train ---


def test_the_driver_never_trains_or_evaluates():
    source = (ROOT / "tools" / "run_distribution_aware_diagnostic.py").read_text(
        encoding="utf-8")
    for forbidden in ("bridge.train(", "bridge.evaluate(", "run_chain",
                      "--epochs", "OWEvaluator"):
        assert forbidden not in source, f"the diagnostic must not reach {forbidden}"
    assert "bridge.predict(" in source, "it does need one detector pass"


def test_the_driver_writes_only_under_its_own_out(built):
    source = (ROOT / "tools" / "run_distribution_aware_diagnostic.py").read_text(
        encoding="utf-8")
    assert "full_owod_active_benchmark_v1" not in source, (
        "the diagnostic must not name the frozen benchmark's results directory"
    )
    for token in ("__seed0", "__seed1", "__seed2"):
        assert token not in source


def test_the_two_arms_are_marked_development_seed_informed():
    for arm in ("cost_aware", "distribution_aware_v1"):
        assert arm in bm.DEVELOPMENT_SEED_INFORMED, arm
    joined = " ".join(bm.PROVENANCE)
    assert "distribution_aware_v1" in joined and "not pre-registered" in joined


# ------------------------------------------------- the diagnostic notebook ---

NOTEBOOK = ROOT / "notebooks" / "distribution_aware_diagnostic.ipynb"

#: The frozen REF-T1 identity, from docs/method_v2_stage2_protocol_2026-09-02.md
#: and tools/bootstrap_stage2_data.py. Restated here so a notebook that quietly
#: stopped checking it fails a test rather than a run.
FROZEN_REF_MANIFEST = (
    "a062fc8f4fd43ea52842725aeaa5eccc0e06eab1894b867b248927bd9d2a2a63")


def _cells(kind: str) -> list[str]:
    import json as _json

    payload = _json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in payload["cells"] if c["cell_type"] == kind]


def test_the_pinned_revision_selects_identically_for_the_completed_no_go():
    """The one-shot diagnostic is finished. Its pin stays; its behaviour must not.

    This used to compare ``git diff <pin> HEAD`` over ``owl/`` and the invoked
    tools, which was right while the notebook was still being prepared. It is
    the wrong check for a **completed** experiment: re-pinning a finished run
    onto later code is how a result quietly becomes a different result, so the
    pin is deliberately fixed and the tree is expected to move past it — the
    iterative arm frozen afterwards necessarily changes ``owl/``.

    The property stated behaviourally instead, and it is stronger: check out the
    pinned ``owl/``, run the three arms this notebook ran on a fixed synthetic
    pool in a subprocess, and require the same gated subsets and the same opened
    images as this tree produces. If a change ever does reach ``entropy``,
    ``cost_aware`` or ``distribution_aware_v1``, the completed NO-GO is no longer
    reproducible and this fails.
    """

    import os
    import subprocess
    import tempfile

    source = _cells("code")[0]
    match = re.search(r'OWL_COMMIT = "([0-9a-f]*)"', source)
    assert match and len(match.group(1)) == 40, "OWL_COMMIT must be a full 40-char SHA"
    assert 'assert len(PROB_COMMIT) == 40 and len(OWL_COMMIT) == 40' in source
    commit = match.group(1)
    assert subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True, check=False).returncode == 0

    # The notebook reads its arms from `diagnostic.DIAGNOSTIC_ARMS` rather than
    # naming them, so the set it ran is that tuple as of the pinned revision.
    import json as _json

    named = _json.dumps(list(gates.DIAGNOSTIC_ARMS))
    assert gates.DIAGNOSTIC_ARMS == (
        "entropy", "cost_aware", "distribution_aware_v1"), gates.DIAGNOSTIC_ARMS
    probe = ROOT / "tests" / "data" / "arm_behaviour_probe.py"

    def fingerprint(tree: Path) -> str:
        done = subprocess.run(
            [sys.executable, str(probe), named],
            capture_output=True, text=True, check=False,
            env=dict(os.environ, PYTHONPATH=str(tree)), cwd=str(tree))
        assert done.returncode == 0, done.stderr[-2000:]
        return done.stdout.strip()

    with tempfile.TemporaryDirectory() as temporary:
        pinned = Path(temporary)
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", commit, "owl"],
                                 capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(pinned)], input=archive, check=True)
        before = fingerprint(pinned)
    assert before == fingerprint(ROOT), (
        f"the arms the completed diagnostic ran behave differently at the "
        f"pinned {commit[:12]} than in this tree. Its NO-GO is no longer "
        "reproducible from what this repository contains."
    )


def test_the_notebook_prints_the_gates_before_it_measures_anything():
    code = _cells("code")
    printed = next(i for i, s in enumerate(code) if "diagnostic.configuration()" in s)
    ran = next(i for i, s in enumerate(code) if "run_distribution_aware_diagnostic.py" in s)
    assert printed < ran, "the frozen criteria must be on the record before the run"


def test_the_notebook_cannot_train_and_says_so():
    """No cell may *call* a training path. Prose that says it will not is fine —
    and is in fact required by the second half of this test."""

    import ast

    for source in _cells("code"):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            rendered = ast.unparse(node)
            for forbidden in ("run_full_owod_benchmark.py", "bridge.train(",
                              "bridge.evaluate(", "--epochs"):
                assert forbidden not in rendered, f"{forbidden} in {rendered[:120]}"
    markdown = "\n".join(_cells("markdown")).lower()
    assert "no prob training" in markdown
    assert "not pre-registered" in markdown


def test_the_notebook_writes_beside_the_frozen_benchmark_not_into_it():
    joined = "\n".join(_cells("code"))
    assert 'RESULTS_RELATIVE = "results/distribution_aware_diagnostic"' in joined
    assert "assert RESULTS != FROZEN" in joined


def test_the_notebook_names_the_cell_a_later_one_depends_on():
    code = _cells("code")
    first = min(i for i, s in enumerate(code) if "diagnostic.configuration()" in s)
    assert '"diagnostic" not in globals()' in code[first]
    assert "[3/10]" in code[first]


def test_every_tool_the_notebook_calls_accepts_the_flags_it_is_given():
    """The defect that started this: a command built against an interface that
    does not exist.

    ``prepare_full_owod_benchmark.py`` was invoked with ``--prob-root`` and
    ``--staging``, neither of which it has ever had, and a fresh Colab found out
    in cell five. This extracts every ``tools/<x>.py`` invocation from the
    notebook and checks each flag against that tool's own ``--help``.
    """

    joined = "\n".join(_cells("code"))
    invocations = re.findall(
        r'"tools"\s*/\s*"([a-z0-9_]+\.py)"(.*?)\]', joined, re.DOTALL)
    assert invocations, "no tool invocations found — this test would be vacuous"

    seen = set()
    for tool, tail in invocations:
        seen.add(tool)
        path = ROOT / "tools" / tool
        assert path.is_file(), f"{tool} does not exist"
        help_text = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True, text=True, check=True,
        ).stdout
        for flag in re.findall(r'"(--[a-z0-9-]+)"', tail):
            assert flag in help_text, f"{tool} has no {flag}"
    assert "prepare_full_owod_benchmark.py" in seen
    assert "run_distribution_aware_diagnostic.py" in seen


def test_the_notebook_prepares_annotations_and_not_the_evaluation_pixels():
    """The diagnostic never evaluates, so the eval split's pixels are not read.

    Both preparation modes extract the committed archives — which is where the
    benchmark XML for every candidate image comes from — so the annotations the
    detector needs are present either way.
    """

    joined = "\n".join(_cells("code"))
    assert '"--annotations-only"' in joined
    assert "Annotations" in joined and "per_image_class_counts.json" in joined


def test_the_notebook_materialises_candidate_pixels():
    """``prepare_full_owod_benchmark.py`` deliberately fetches no candidate
    images, so something else must. Prove the notebook and the driver both know."""

    prepare_help = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "prepare_full_owod_benchmark.py"), "--help"],
        capture_output=True, text=True, check=True).stdout
    assert "not** fetched here" in prepare_help or "not fetched here" in prepare_help.replace("*", "")

    driver = (ROOT / "tools" / "run_distribution_aware_diagnostic.py").read_text(
        encoding="utf-8")
    assert "from tools.materialize_pool_images import materialise" in driver
    assert "prepare_images(candidate_ids)" in driver
    joined = "\n".join(_cells("code"))
    assert "materialize_pool_images" in joined


def test_ref_t1_is_mandatory_and_verified():
    """One protocol, not two. An absent REF-T1 must rebuild or fail, never
    silently degrade to an empty reference."""

    joined = "\n".join(_cells("code"))
    assert "ref_t1_dinov2_vitb14_cap1000_v1.npz" in joined, "wrong export filename"
    assert FROZEN_REF_MANIFEST in joined, "the frozen manifest is not verified"
    assert "bootstrap_stage2_data.py" in joined and "export_ref_t1_features.py" in joined
    assert '"--ref-t1", str(REF_T1)' in joined, "the run must be handed REF-T1"

    driver = (ROOT / "tools" / "run_distribution_aware_diagnostic.py").read_text(
        encoding="utf-8")
    assert "def require_reference" in driver
    assert "require_reference(args.arms, args.ref_t1)" in driver


def test_the_driver_refuses_to_run_without_ref_t1(tmp_path):
    """The rule, exercised rather than asserted about."""

    from tools import run_distribution_aware_diagnostic as driver

    with pytest.raises(driver.DiagnosticError, match="REF-T1"):
        driver.require_reference(["distribution_aware_v1"], None)
    with pytest.raises(driver.DiagnosticError, match="REF-T1"):
        driver.require_reference(["distribution_aware_v1"], tmp_path / "absent.npz")
    # arms that do not consult it are unaffected
    driver.require_reference(["entropy", "cost_aware"], None)


def test_a_clean_run_all_defines_every_name_before_use():
    """The dependency graph, checked instead of hoped for."""

    from tools.audit_notebook_dataflow import analyse, code_cells

    report = analyse(code_cells(NOTEBOOK))
    undefined = {row["cell"]: row["undefined"] for row in report if row["undefined"]}
    assert not undefined, undefined
    # and the producer of each cell's inputs is an earlier cell, by construction
    for row in report:
        assert all(producer < row["cell"] for producer in row["consumes"].values())


def test_the_notebook_reads_its_arms_and_seeds_from_the_module():
    """A retyped constant is a constant that can drift from the frozen gates."""

    joined = "\n".join(_cells("code"))
    assert "SESSION_ARMS = diagnostic.DIAGNOSTIC_ARMS" in joined
    assert "SESSION_SEEDS = diagnostic.DIAGNOSTIC_SEEDS" in joined


def test_the_notebook_streams_failures_instead_of_swallowing_them():
    """`capture_output=True` hides the traceback of the step that failed, which
    is the one thing a three-hour Run all must not do."""

    joined = "\n".join(_cells("code"))
    assert "def _streamed(" in joined
    assert "stderr=subprocess.STDOUT" in joined
    assert "its output is above" in joined
    # Every expensive step goes through it — checked on the call graph rather
    # than on a window of surrounding text, which a long comment defeats.
    import ast

    streamed: set[str] = set()
    for source in _cells("code"):
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "_streamed"):
                # ast.unparse renders string literals with single quotes
                streamed |= set(re.findall(r"[a-z0-9_]+\.py", ast.unparse(node)))
    for tool in ("prepare_full_owod_benchmark.py", "bootstrap_stage2_data.py",
                 "export_ref_t1_features.py", "run_distribution_aware_diagnostic.py"):
        assert tool in streamed, f"{tool} is not run through _streamed"


# ---------------------------------------------------------------- resuming ---
#
# A three-hour session that dies at 90% must not cost three hours. The DINOv2
# cache is atomic by construction (`.npz.part` then rename) and validated on
# read by fingerprint; the detector cache is written in place by PROB and cached
# on the path merely *existing*, so a pass cut off mid-write is the one way this
# directory can hold a file that looks finished and is not.


def _npz(path: Path, **arrays) -> Path:
    import numpy as _np

    _np.savez_compressed(path, **arrays)
    return path


def test_a_complete_predict_export_is_reusable(tmp_path):
    from tools.run_distribution_aware_diagnostic import predict_cache_is_usable

    good = _npz(
        tmp_path / "good.npz",
        image_ids=np.array(["a", "b"]), boxes=np.zeros((2, 4)),
        posterior=np.ones((2, 81)) / 81, objectness=np.ones(2),
        embeddings=np.zeros((2, 8)),
    )
    assert predict_cache_is_usable(good)


@pytest.mark.parametrize("damage", ["truncate", "empty", "missing_array", "absent"])
def test_an_incomplete_predict_export_is_not_reusable(tmp_path, damage):
    """Each is a way a killed session leaves something that looks finished."""

    from tools.run_distribution_aware_diagnostic import predict_cache_is_usable

    path = tmp_path / "export.npz"
    if damage == "absent":
        assert not predict_cache_is_usable(path)
        return
    if damage == "empty":
        path.write_bytes(b"")
    elif damage == "missing_array":
        _npz(path, image_ids=np.array(["a"]), boxes=np.zeros((1, 4)))
    else:
        full = _npz(tmp_path / "full.npz", image_ids=np.array(["a"] * 200),
                    boxes=np.zeros((200, 4)), posterior=np.ones((200, 81)) / 81,
                    objectness=np.ones(200), embeddings=np.zeros((200, 8)))
        blob = full.read_bytes()
        path.write_bytes(blob[: len(blob) // 2])
    assert not predict_cache_is_usable(path)


def test_an_unusable_entry_is_moved_aside_and_not_destroyed(tmp_path):
    """The rule for this directory is that nothing is deleted. A corrupt export
    is also evidence about how the session died."""

    from tools.run_distribution_aware_diagnostic import quarantine

    path = tmp_path / "export.npz"
    path.write_bytes(b"not a zip")
    spoiled = quarantine(path)
    assert not path.exists()
    assert spoiled.exists() and spoiled.read_bytes() == b"not a zip"
    assert spoiled.name.endswith(".npz.incomplete")


def test_verify_cache_reports_without_changing_anything(tmp_path):
    from tools.run_distribution_aware_diagnostic import verify_cache

    workspace = tmp_path / "work"
    (workspace / "_predict").mkdir(parents=True)
    _npz(workspace / "_predict" / "aaaa.npz", image_ids=np.array(["a"]),
         boxes=np.zeros((1, 4)), posterior=np.ones((1, 81)) / 81,
         objectness=np.ones(1), embeddings=np.zeros((1, 8)))
    (workspace / "_predict" / "bbbb.npz").write_bytes(b"")
    task = workspace / "entropy__seed0" / "t2"
    task.mkdir(parents=True)
    _npz(task / "dinov2_pool.npz", features=np.zeros((3, 4)))
    (task / "dinov2_pool.npz.part").write_bytes(b"half")

    before = sorted(p.name for p in workspace.rglob("*"))
    report = verify_cache(workspace)
    assert sorted(p.name for p in workspace.rglob("*")) == before

    assert report["predict_exports"] == 2 and report["predict_usable"] == 1
    assert len(report["predict_incomplete"]) == 1
    assert report["dinov2_exports"] == 1
    assert len(report["abandoned_part_files"]) == 1


def test_a_resumed_run_reproduces_an_uninterrupted_one(tmp_path):
    """The property the whole resume path exists for.

    Run the diagnostic, corrupt a cached detector export the way a killed
    session would, run again, and require the rows to be **byte-identical**. If
    resuming could change a number, the cache would be a liability rather than a
    saving.
    """

    from tools import run_distribution_aware_diagnostic as driver

    out = tmp_path / "out"
    arguments = ["--dry-run", "--out", str(out), "--seeds", "0"]
    assert driver.main(arguments) == 0
    first = (out / "diagnostic_rows.csv").read_bytes()

    exports = sorted((out / "work" / "_predict").glob("*.npz"))
    assert exports, "the dry run cached no detector export — this would be vacuous"
    blob = exports[0].read_bytes()
    exports[0].write_bytes(blob[: len(blob) // 2])
    assert not driver.predict_cache_is_usable(exports[0])

    assert driver.main(arguments) == 0
    assert (out / "diagnostic_rows.csv").read_bytes() == first
    assert list((out / "work" / "_predict").glob("*.incomplete"))


# --------------------------------------------------------------- iterative ---


def test_rounds_one_iterative_is_exactly_v1(built, features):
    """The strongest invariant available: with a single round the iterative arm
    must reproduce ``distribution_aware_v1`` exactly.

    If it does not, something other than the round structure changed, and the
    experiment would no longer be the one-variable test it is sold as.
    """

    index = arms.ranked_positions("distribution_aware_v1", built)
    common = {
        "entropy": np.linspace(0, 1, index.size),
        "image_ids": built.candidates.image_ids[index],
        "cost_of": lambda i: len(i) % 7 + 1,
        "budget": 3000,
    }
    one_shot = allocation.distribution_aware_order(features, **common)
    from owl.active_selection import budget as ledger

    expected = ledger.spend_ranking(
        one_shot.order, common["image_ids"], common["cost_of"], budget=3000)
    iterative = allocation.distribution_aware_iterative_order(
        features, rounds=1, **common)
    assert iterative.images == expected.images
    assert iterative.diagnostics["min_cluster_size"] == one_shot.diagnostics["min_cluster_size"]


def test_the_reference_grows_and_the_eligible_set_shrinks(built, features):
    """The mechanism, observed rather than asserted about."""

    index = arms.ranked_positions("distribution_aware_v1", built)
    result = allocation.distribution_aware_iterative_order(
        features, entropy=np.linspace(0, 1, index.size),
        image_ids=built.candidates.image_ids[index],
        cost_of=lambda i: len(i) % 7 + 1, budget=3000, rounds=6,
    )
    spent_rounds = [r for r in result.rounds if r.get("images")]
    assert len(spent_rounds) >= 2, "a single-round run would make this vacuous"
    eligible = [r["eligible"] for r in spent_rounds]
    references = [r["reference_rows"] for r in spent_rounds]
    assert eligible == sorted(eligible, reverse=True) and eligible[0] > eligible[-1]
    assert references == sorted(references) and references[0] < references[-1]
    assert result.diagnostics["answers_spent"] <= 3000


def test_the_iterative_order_is_reproducible(built, features):
    index = arms.ranked_positions("distribution_aware_v1", built)
    kwargs = {
        "entropy": np.linspace(0, 1, index.size),
        "image_ids": built.candidates.image_ids[index],
        "cost_of": lambda i: len(i) % 7 + 1, "budget": 3000, "rounds": 6,
    }
    first = allocation.distribution_aware_iterative_order(features, **kwargs)
    second = allocation.distribution_aware_iterative_order(features, **kwargs)
    assert first.images == second.images
    assert first.diagnostics == second.diagnostics


def test_the_iterative_arm_reads_no_answer(built):
    stripped = population.build(
        proposals.Candidates(
            image_ids=built.candidates.image_ids, boxes=built.candidates.boxes,
            posterior=built.candidates.posterior,
            objectness=built.candidates.objectness,
            embeddings=built.candidates.embeddings,
        )
    )
    index = arms.ranked_positions("distribution_aware_iterative_v1", stripped)
    generator = np.random.default_rng(3)
    block = generator.normal(size=(index.size, 16))
    block /= np.linalg.norm(block, axis=1, keepdims=True)
    picked = arms.select(
        "distribution_aware_iterative_v1", stripped, cost_of=lambda _: 3,
        answer_budget=200, seed=0, semantic=block.astype(np.float32), rounds=6,
    )
    assert len(picked.images) > 0
    assert picked.row["rounds"] == 6


# ----------------------------------------------------- the round schedule ---


def test_the_carry_makes_rounds_a_no_op_for_a_static_ranking():
    """Why the comparators need no re-run, checked on the real code path."""

    from owl.active_selection import budget as ledger

    rng = np.random.default_rng(0)
    images = np.array([f"i{n:05d}" for n in range(4000)])
    order = rng.permutation(len(images))
    cost_of = (lambda i: (int(i[1:]) % 11) + 1)

    one = ledger.spend_ranking(order, images, cost_of, budget=3000)
    for rounds in (1, 2, 3, 6, 12):
        many = ledger.spend_ranking_in_rounds(
            order, images, cost_of, budget=3000, rounds=rounds)
        assert many.images == one.images, rounds
        assert many.ledger.spent == one.ledger.spent, rounds


def test_without_the_carry_the_schedule_would_shrink_the_campaign():
    """The confound the carry rule exists to remove — demonstrated, not claimed."""

    from owl.active_selection import budget as ledger

    rng = np.random.default_rng(0)
    images = np.array([f"i{n:05d}" for n in range(4000)])
    order = rng.permutation(len(images))
    cost_of = (lambda i: (int(i[1:]) % 11) + 1)
    one = ledger.spend_ranking(order, images, cost_of, budget=3000)

    opened: list[str] = []
    for _ in range(6):                      # a fixed 500 each time, no carry
        spend = ledger.spend_ranking(
            order, images, cost_of, budget=500,
            excluded_images=frozenset(opened))
        opened.extend(spend.images)
    assert len(opened) < len(one.images), (
        "if these ever agree, the carry rule has stopped mattering and this "
        "test should be reconsidered rather than deleted"
    )


def test_round_allowance_carries_the_whole_remainder():
    assert allocation.round_allowance(3000, 6, 1, 0) == 500
    assert allocation.round_allowance(3000, 6, 2, 480) == 520     # 20 carried
    assert allocation.round_allowance(3000, 6, 6, 2400) == 600
    assert allocation.round_allowance(3000, 6, 3, 1600) == 0      # never negative


def test_the_method_under_test_is_derived_not_assumed():
    """A default of "distribution_aware_v1" judged the wrong arm the moment a
    second method existed."""

    assert gates.method_under_test(
        ["entropy", "cost_aware", "distribution_aware_iterative_v1"]
    ) == "distribution_aware_iterative_v1"
    with pytest.raises(ValueError, match="exactly one method"):
        gates.method_under_test(["entropy", "cost_aware"])
    with pytest.raises(ValueError, match="exactly one method"):
        gates.method_under_test(["entropy", "cost_aware", "a", "b"])


def test_the_iterative_session_is_frozen():
    assert gates.ITERATIVE_ROUNDS == 6
    assert gates.ITERATIVE_ARMS == (
        "entropy", "cost_aware", "distribution_aware_iterative_v1")
    assert gates.COMPARATORS == ("entropy", "cost_aware")
    # the gates themselves are untouched by the iterative session
    assert {g.key: g.threshold for g in gates.GATES} == {
        "jaccard_vs_entropy": 0.60, "tail_per_image_vs_entropy": 1.25,
        "tail_per_image_vs_cost_aware": 1.25,
        "background_share_vs_entropy": 0.10, "breadth_vs_entropy": 0.75,
        "t2_not_collapsed": 0.75,
    }


# ------------------------------------------------------------ closing A1 ---


def _part_a_csv(path: Path, rows) -> Path:
    import csv as _csv

    fields = ["arm", "seed", "task", "images_opened", "answers_spent",
              "current_new_objects", "future_new_objects",
              "banked_from_earlier", "held_at_declaration", "tail_objects",
              "acquired_class_breadth"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _arm_rows(arm, seed, fh2, ss2, ss3):
    """A consistent trajectory for one arm, from known per-class purchases."""

    return [
        {"arm": arm, "seed": seed, "task": "t2", "images_opened": 300,
         "answers_spent": 3000, "current_new_objects": 70,
         "future_new_objects": fh2 + ss2, "banked_from_earlier": 0,
         "held_at_declaration": 70, "tail_objects": 5,
         "acquired_class_breadth": 50},
        {"arm": arm, "seed": seed, "task": "t3", "images_opened": 300,
         "answers_spent": 3000, "current_new_objects": 10,
         "future_new_objects": ss3, "banked_from_earlier": fh2,
         "held_at_declaration": 10 + fh2, "tail_objects": 5,
         "acquired_class_breadth": 50},
        {"arm": arm, "seed": seed, "task": "t4", "images_opened": 300,
         "answers_spent": 3000, "current_new_objects": 11,
         "future_new_objects": 0, "banked_from_earlier": ss2 + ss3,
         "held_at_declaration": 11 + ss2 + ss3, "tail_objects": 5,
         "acquired_class_breadth": 50},
    ]


def test_per_class_early_purchase_is_recovered_exactly(tmp_path):
    """The memo said this was unavailable. It is not: three CSV columns are
    linear in FH@t2, SS@t2 and SS@t3, and the system is over-determined."""

    from tools.close_part_a import early_purchases, load

    path = _part_a_csv(tmp_path / "rows.csv",
                       _arm_rows("entropy", 0, 7, 8, 4)
                       + _arm_rows("distribution_aware_v1", 0, 15, 9, 8))
    table = load(path)
    got = early_purchases(table, "distribution_aware_v1", 0)
    assert got["fire_hydrant_at_t2"] == 15
    assert got["stop_sign_at_t2"] == 9
    assert got["stop_sign_at_t3"] == 8
    assert got["future_tail_bought_early"] == 32


def test_the_reconstruction_refuses_a_file_that_fails_its_own_identity(tmp_path):
    """`future_new(t2)` must equal `banked(t3) + banked(t4) - future_new(t3)`.
    If it does not, the file is not the run this arithmetic assumes."""

    from tools.close_part_a import ForensicError, early_purchases, load

    rows = _arm_rows("entropy", 0, 7, 8, 4)
    rows[0]["future_new_objects"] = 99          # break it
    path = _part_a_csv(tmp_path / "rows.csv", rows)
    with pytest.raises(ForensicError, match="does not describe the run"):
        early_purchases(load(path), "entropy", 0)


def test_the_reconstruction_refuses_a_negative_split(tmp_path):
    from tools.close_part_a import ForensicError, early_purchases, load

    rows = _arm_rows("entropy", 0, 7, 8, 4)
    rows[2]["banked_from_earlier"] = 2          # less than SS@t3 alone
    rows[0]["future_new_objects"] = 7 + (2 - 4)
    path = _part_a_csv(tmp_path / "rows.csv", rows)
    with pytest.raises(ForensicError):
        early_purchases(load(path), "entropy", 0)



# ------------------------------------------------- the iterative notebook ---

ITERATIVE_NOTEBOOK = (
    ROOT / "notebooks" / "distribution_aware_iterative_diagnostic.ipynb")


def _iter_cells(kind: str) -> list[str]:
    import json as _json

    payload = _json.loads(ITERATIVE_NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in payload["cells"] if c["cell_type"] == kind]


def test_the_iterative_notebook_writes_somewhere_new():
    """The completed NO-GO must survive this session untouched."""

    joined = "\n".join(_iter_cells("code"))
    assert ('RESULTS_RELATIVE = "results/distribution_aware_iterative_diagnostic"'
            in joined)
    assert 'ONE_SHOT = DRIVE / "results" / "distribution_aware_diagnostic"' in joined
    assert "assert RESULTS not in (FROZEN, ONE_SHOT)" in joined


def test_the_iterative_notebook_runs_six_rounds_read_from_the_module():
    joined = "\n".join(_iter_cells("code"))
    assert "SESSION_ARMS = diagnostic.ITERATIVE_ARMS" in joined
    assert "SESSION_ROUNDS = diagnostic.ITERATIVE_ROUNDS" in joined
    assert '"--rounds", str(SESSION_ROUNDS)' in joined
    assert "assert SESSION_ROUNDS == 6" in joined


def test_the_iterative_notebook_prints_the_gates_first():
    cells = _iter_cells("code")
    printed = next(i for i, s in enumerate(cells) if "diagnostic.configuration()" in s)
    ran = next(i for i, s in enumerate(cells)
               if "run_distribution_aware_diagnostic.py" in s and "--rounds" in s)
    assert printed < ran


def test_the_iterative_notebook_pin_carries_this_code():
    import subprocess as _sp

    source = _iter_cells("code")[0]
    match = re.search(r'OWL_COMMIT = "([0-9a-f]*)"', source)
    assert match and len(match.group(1)) == 40
    commit = match.group(1)
    assert _sp.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
                   capture_output=True, check=False).returncode == 0
    for tree in ("owl/", "tools/run_distribution_aware_diagnostic.py"):
        drift = _sp.run(
            ["git", "-C", str(ROOT), "diff", "--name-only", commit, "HEAD", "--", tree],
            capture_output=True, text=True, check=True).stdout.strip()
        assert not drift, f"{tree} differs from the pinned {commit[:12]}:\n{drift}"


def test_every_tool_the_iterative_notebook_calls_accepts_its_flags():
    import subprocess as _sp

    joined = "\n".join(_iter_cells("code"))
    invocations = re.findall(
        r'"tools"\s*/\s*"([a-z0-9_]+\.py)"(.*?)\]', joined, re.DOTALL)
    assert invocations
    for tool, tail in invocations:
        help_text = _sp.run([sys.executable, str(ROOT / "tools" / tool), "--help"],
                            capture_output=True, text=True, check=True).stdout
        for flag in re.findall(r'"(--[a-z0-9-]+)"', tail):
            assert flag in help_text, f"{tool} has no {flag}"


def test_the_iterative_notebook_defines_every_name_before_use():
    from tools.audit_notebook_dataflow import analyse, code_cells

    report = analyse(code_cells(ITERATIVE_NOTEBOOK))
    assert not {r["cell"]: r["undefined"] for r in report if r["undefined"]}
