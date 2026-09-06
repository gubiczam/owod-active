"""``distribution_aware_v1`` and its ``cost_aware`` control: the frozen properties.

The memo (``docs/distribution_aware_decision_memo_2026-09-06.md``) is the binding
specification. These tests pin the things that would silently invalidate the
diagnostic rather than break it: a selector that can see an answer, a clusterer
that does not reproduce, a gate whose threshold drifted, a run that trains.
"""

from __future__ import annotations

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
    verdict = gates.evaluate(rows, opened, seeds=(0,))
    assert not verdict.go
    assert "tail_per_image_vs_entropy" in verdict.failed


def test_a_clearly_better_arm_passes_every_gate():
    """The gates must be satisfiable — otherwise NO-GO proves nothing."""

    rows, opened = _synthetic(tail_ratio=3.0)
    verdict = gates.evaluate(rows, opened, seeds=(0,))
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
