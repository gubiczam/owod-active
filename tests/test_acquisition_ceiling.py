"""The ceiling audit must stay a *ceiling* audit: read-only, and honest.

The numbers this tool produced decide whether a third selection method is built
at all, so two things are pinned. That it cannot train, evaluate or write into a
results directory — it is allowed to read oracle labels precisely because it
never touches the experiment. And that its ceiling policies really are ceilings:
a policy with perfect foreknowledge of the declared class must buy all of it,
otherwise the "acquisition is not saturated" conclusion is an artefact of a
stopping bug rather than a fact about the budget.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from owl import protocol
from owl.active_selection import benchmark as bm

ceiling = pytest.importorskip("tools.audit_acquisition_ceiling")

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools" / "audit_acquisition_ceiling.py").read_text(encoding="utf-8")


def test_the_tool_cannot_train_or_evaluate():
    for forbidden in (".train(", ".evaluate(", "bridge.", "subprocess"):
        assert forbidden not in SOURCE, f"a ceiling audit must not call {forbidden}"


def test_the_tool_writes_only_where_it_is_told():
    """One `open("w")`, behind `--csv`. Nothing near a trajectory directory."""

    assert SOURCE.count('.open("w"') == 1
    assert "full_owod_active_benchmark_v1" not in SOURCE
    assert "results/" not in SOURCE.replace("data/results/", "")


def test_the_cost_model_matches_the_ledger():
    """If this drifts from `owl.active_selection.budget`, the ceilings are wrong."""

    from owl.active_selection import budget

    counts = {"a": {"person": 3, "car": 1}, "b": {}, "c": {"bear": 1}}
    for image, objects in counts.items():
        assert ceiling.image_cost(counts, image) == budget.image_cost(objects)
    assert ceiling.image_cost(counts, "missing") == budget.ANSWER_FLOOR


def test_spend_stops_rather_than_skipping_an_unaffordable_image():
    """The ledger's own rule. Skipping would bias the campaign tail to sparse
    images and quietly inflate every ceiling in the table."""

    counts = {"a": {"x": 2}, "big": {"x": 50}, "c": {"x": 1}}
    opened, spent = ceiling.spend(["a", "big", "c"], counts, budget=5)
    assert opened == ["a"] and spent == 2, (opened, spent)


def test_perfect_foreknowledge_buys_the_whole_declared_class():
    """The finding this tool exists for: the budget is not the constraint.

    `class-greedy*` knows which class is declared next. On the real candidate
    index it must acquire **every** instance of that class present in the task
    pool -- which is what makes "acquisition is not saturated" a fact about the
    budget rather than an accident of the simulation.
    """

    counts = ceiling.load_index()
    images = np.array(sorted(counts), dtype=str)
    known = set(protocol.TASK1)
    chain = protocol.build_chain(bm.N_TASKS)

    rows = ceiling.trajectory(
        "class-greedy*", counts, images, known, chain, 0,
        budget=bm.ANSWER_BUDGET_PER_TASK,
        pool_size=bm.CANDIDATE_IMAGES_PER_TASK,
    )
    assert len(rows) == bm.N_TASKS - 1
    for row in rows:
        assert row["acquired_this_task"] == row["available_in_pool"], row
        assert row["answers_spent"] <= bm.ANSWER_BUDGET_PER_TASK


def test_random_is_the_only_policy_without_an_asterisk():
    """The asterisk is the oracle warning; losing it would let a ceiling be read
    as an implementable arm."""

    assert set(ceiling.POLICIES) - {"random"} == {
        name for name in ceiling.POLICIES if name.endswith("*")
    }
