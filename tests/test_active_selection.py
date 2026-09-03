"""The selectors, the population, the ledger and the traversal.

What these tests are for, in order of how much they would cost to get wrong:

* the population **reproduces the committed one**. Methods V2 and V3 were
  measured on ``P2`` = 15,518 rows at a 0.767 background share; a new NMS
  implementation that disagrees would silently change what every arm selects
  from, and no downstream number would say so.
* **no arm can read an answer.** Every arm runs on a pool with no oracle at all,
  which is what a live GPU pool is.
* k-center greedy is **the standard criterion**, checked on synthetic clusters
  where the right answer is known by construction.
* the ledger's arithmetic, because the whole benchmark rests on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from owl import proposals, scoring, selection
from owl.active_selection import arms, benchmark, budget, coverage, population


@pytest.fixture(scope="module")
def pool():
    return proposals.from_frozen_pool(split="pool")


@pytest.fixture(scope="module")
def built(pool):
    return population.build(pool)


# ------------------------------------------------------------- population ---


def test_p2_reproduces_the_committed_population(pool):
    """The committed recipe, in this module's code, on the committed numbers."""

    mask = population.p2_reference(pool)
    assert int(mask.sum()) == 15_518
    background = float((pool.oracle().kind[mask] == "background").mean())
    assert background == pytest.approx(0.767, abs=0.002)


def test_p2_agrees_with_the_original_implementation(pool):
    from tools.audit_decoder_layers import populations
    from tools.diagnose_representation import load

    reference = load()
    payload = np.load(proposals.FROZEN_POOL, allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == "pool"
    reference["raw_boxes"] = payload["boxes"][keep].astype(np.float32)
    original = populations(reference, pool)["P2_admissible_nms"]
    assert np.array_equal(population.p2_reference(pool), original)


def test_benchmark_population_deduplicates_then_gates(built, pool):
    assert len(built) < len(pool)
    assert built.diagnostics["proposals_after_nms"] == len(built)
    assert built.gate.sum() == pytest.approx(
        round(len(built) * population.ADMISSIBLE_SHARE), abs=1
    )
    # the gate is the top share by A, so its minimum A is above the rest's maximum
    inside, outside = built.admissibility[built.gate], built.admissibility[~built.gate]
    assert inside.min() >= outside.max()


def test_nms_keeps_one_box_per_overlapping_group():
    boxes = np.array([
        [0.5, 0.5, 0.4, 0.4],      # kept: highest order key
        [0.51, 0.51, 0.4, 0.4],    # suppressed by the first
        [0.1, 0.1, 0.05, 0.05],    # disjoint, kept
    ])
    images = np.array(["a", "a", "a"])
    keep = population.per_image_nms(boxes, images, np.array([3.0, 2.0, 1.0]))
    assert keep.tolist() == [True, False, True]


def test_nms_does_not_suppress_across_images():
    boxes = np.array([[0.5, 0.5, 0.4, 0.4]] * 2)
    keep = population.per_image_nms(boxes, np.array(["a", "b"]), np.array([2.0, 1.0]))
    assert keep.all()


# ------------------------------------------------------------------- arms ---


def test_no_arm_reads_an_answer(built):
    """A live pool carries no oracle. Every arm must still run."""

    import dataclasses

    naked = dataclasses.replace(built.candidates, _oracle=None)
    assert not naked.has_oracle
    bare = population.Population(
        candidates=naked, admissibility=built.admissibility, gate=built.gate,
        kept=built.kept, diagnostics=built.diagnostics,
    )
    cost_of = lambda _: 5
    for name, spec in arms.ARMS.items():
        features = None
        if spec.needs_semantic:
            index = np.flatnonzero(bare.gate) if spec.gated else np.arange(len(bare))
            features = bare.candidates.embeddings[index]
        picked = arms.select(
            name, bare, cost_of=cost_of, answer_budget=100, seed=0,
            semantic=features,
        )
        assert len(picked) == 20, name


@pytest.mark.parametrize(
    ("arm", "committed"), [("entropy", "entropy"), ("admissibility", "objectness")]
)
def test_ranking_matches_the_committed_selector(built, arm, committed):
    """The same order the registered ScoreConfig produces, without a k-means fit."""

    mine = arms.ranking(arm, built, seed=0)[:600]
    theirs = selection.select(
        built.candidates, selection.ARMS[committed], budget=600, rounds=1,
        partition=_trivial_partition(len(built)),
    ).indices
    assert mine.tolist() == theirs.tolist()


def _trivial_partition(n: int):
    from owl import clustering

    return clustering.Partition(
        labels=np.zeros(n, dtype=np.int64),
        centroids=np.zeros((1, 1), dtype=np.float32),
        sizes=np.asarray([n]), method="trivial", params={},
    )


def test_admissibility_is_not_raw_objectness(pool):
    """Distinct, and the distinctness is why raw objectness gets no trajectory."""

    a = scoring.admissibility(pool)
    raw = pool.objectness
    top_a = set(np.argsort(-a, kind="mergesort")[:600].tolist())
    top_raw = set(np.argsort(-raw, kind="mergesort")[:600].tolist())
    assert not top_a & top_raw, "the two prefixes were measured to be disjoint"
    kind = pool.oracle().kind
    objects_a = int((kind[sorted(top_a)] != "background").sum())
    objects_raw = int((kind[sorted(top_raw)] != "background").sum())
    assert objects_a > 100 and objects_raw < 10, (objects_a, objects_raw)


def test_a_semantic_arm_refuses_a_pool_it_was_not_given_features_for(built):
    with pytest.raises(arms.ArmError, match="no features were supplied"):
        arms.select("proposed", built, cost_of=lambda _: 1, answer_budget=10, seed=0)


def test_a_ranking_arm_refuses_features_nothing_would_read(built):
    with pytest.raises(arms.ArmError, match="does not read semantic features"):
        arms.select("random", built, cost_of=lambda _: 1, answer_budget=10, seed=0,
                    semantic=np.zeros((len(built), 4), dtype=np.float32))


def test_a_gated_arm_wants_features_for_the_gate_only(built):
    with pytest.raises(arms.ArmError, match="the admissible subset G"):
        arms.select("proposed", built, cost_of=lambda _: 1, answer_budget=10, seed=0,
                    semantic=built.candidates.embeddings)


def test_gated_and_ungated_arms_see_different_candidate_counts(built):
    cost_of = lambda _: 4
    gated = arms.select(
        "proposed", built, cost_of=cost_of, answer_budget=200, seed=0,
        semantic=built.candidates.embeddings[np.flatnonzero(built.gate)],
    )
    ungated = arms.select(
        "coreset", built, cost_of=cost_of, answer_budget=200, seed=0,
        semantic=built.candidates.embeddings,
    )
    assert gated.row["coverage_candidates"] < ungated.row["coverage_candidates"]
    assert gated.images != ungated.images


def test_random_is_reproducible_and_seed_dependent(built):
    cost_of = lambda _: 3
    one = arms.select("random", built, cost_of=cost_of, answer_budget=90, seed=0)
    again = arms.select("random", built, cost_of=cost_of, answer_budget=90, seed=0)
    other = arms.select("random", built, cost_of=cost_of, answer_budget=90, seed=1)
    assert one.images == again.images
    assert one.images != other.images


def test_the_declared_order_holds_every_registered_arm():
    assert set(arms.ORDER) == set(arms.ARMS)
    assert arms.ORDER[:3] == ("random", "admissibility", "proposed"), (
        "the primary contrast and its reference must survive a short session"
    )


# ----------------------------------------------------------------- budget ---


def test_an_image_costs_its_objects_but_never_zero():
    assert budget.image_cost({"person": 3, "car": 1}) == 4
    assert budget.image_cost({}) == budget.ANSWER_FLOOR
    assert budget.image_cost(None) == budget.ANSWER_FLOOR


def test_the_ledger_stops_rather_than_skipping_an_unaffordable_image():
    ledger = budget.Ledger(budget=10)
    ledger.charge("a", 7)
    assert ledger.remaining == 3
    assert not ledger.affordable(4)
    assert ledger.affordable(3)


def test_spending_a_ranking_opens_each_image_once():
    images = np.array(["a", "a", "b", "c"])
    spend = budget.spend_ranking(
        [0, 1, 2, 3], images, lambda name: {"a": 2, "b": 3, "c": 4}[name], budget=5
    )
    assert spend.images == ("a", "b")
    assert spend.redundant == 1          # the second 'a'
    assert spend.ledger.spent == 5
    assert spend.scanned == 4            # 'c' was consulted and did not fit


def test_an_excluded_image_is_skipped_not_charged():
    images = np.array(["a", "b"])
    spend = budget.spend_ranking(
        [0, 1], images, lambda _: 1, budget=5, excluded_images=frozenset({"a"})
    )
    assert spend.images == ("b",)
    assert spend.ledger.spent == 1


def test_supervision_separates_declared_from_banked():
    index = {"i1": {"person": 2, "traffic light": 1}, "i2": {"fire hydrant": 3}}
    row = budget.supervision(index, ["i1", "i2"], declared=("person", "traffic light"))
    assert row["boxes_labelled"] == 6
    assert row["boxes_supervised"] == 3
    assert row["boxes_banked"] == 3
    assert row["images_barren"] == 1


def test_an_image_counted_twice_is_charged_once():
    index = {"i1": {"person": 2}}
    row = budget.supervision(index, ["i1", "i1"], declared=("person",))
    assert row["boxes_labelled"] == 2


def test_acquisition_says_when_an_object_becomes_learnable():
    chain = benchmark.chain()
    index = {"i1": {"person": 1, "fire hydrant": 2, "stop sign": 1, "banana": 5}}
    row = budget.acquisition(index, ["i1"], chain=chain, task_index=1)
    assert row["acquired_becomes_known_t3"] == 2      # fire hydrant, declared at t3
    assert row["acquired_becomes_known_t4"] == 1      # stop sign, declared at t4
    assert row["acquired_stays_unknown"] == 5         # banana, never in this chain
    assert row["acquired_known_now"] == 1


# --------------------------------------------------------------- coverage ---


def test_kcenter_visits_every_cluster_before_revisiting_one():
    """Farthest-first on well-separated clusters takes one from each, in turn."""

    generator = np.random.default_rng(0)
    centres = np.eye(4, dtype=np.float32)
    features, images = [], []
    for cluster in range(4):
        for member in range(5):
            vector = centres[cluster] + 0.01 * generator.normal(size=4)
            features.append(vector / np.linalg.norm(vector))
            images.append(f"c{cluster}m{member}")
    features = np.asarray(features, dtype=np.float32)
    result = coverage.kcenter_greedy(
        features, np.asarray(images), cost_of=lambda _: 1, budget=4,
        tie_break=np.arange(len(images), dtype=float),
    )
    clusters = [name[:2] for name in result.images]
    assert sorted(clusters) == ["c0", "c1", "c2", "c3"]


def test_kcenter_avoids_what_the_reference_already_covers():
    centres = np.eye(3, dtype=np.float32)
    features = np.asarray([centres[0], centres[1], centres[2]], dtype=np.float32)
    result = coverage.kcenter_greedy(
        features, np.asarray(["a", "b", "c"]), cost_of=lambda _: 1, budget=1,
        reference=centres[:2],
    )
    assert result.images == ("c",)


def test_opening_an_image_covers_everything_on_it():
    features = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = coverage.kcenter_greedy(
        features, np.asarray(["a", "a", "b"]), cost_of=lambda _: 1, budget=1,
        tie_break=np.asarray([1.0, 0.0, 0.0]),
    )
    assert result.images == ("a",)
    assert result.covered.tolist() == [True, True, False]


def test_the_tie_break_decides_the_first_pick_when_nothing_is_covered():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = coverage.kcenter_greedy(
        features, np.asarray(["a", "b"]), cost_of=lambda _: 1, budget=1,
        tie_break=np.asarray([0.0, 9.0]),
    )
    assert result.images == ("b",)


def test_coverage_reports_why_it_stopped():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = coverage.kcenter_greedy(
        features, np.asarray(["a", "b"]), cost_of=lambda _: 100, budget=1,
    )
    assert result.images == ()
    assert "100" in result.diagnostics["stopped_because"]


def test_coverage_refuses_mismatched_inputs():
    with pytest.raises(ValueError, match="same candidates"):
        coverage.kcenter_greedy(
            np.zeros((3, 2), dtype=np.float32), np.asarray(["a"]),
            cost_of=lambda _: 1, budget=1,
        )
