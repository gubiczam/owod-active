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
            # through the arm's own definition, so a new arm cannot quietly be
            # handed the wrong subset here and pass anyway
            features = bare.candidates.embeddings[arms.ranked_positions(name, bare)]
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

# ----------------------------------- the failure mode Method V3 walked into ---


def test_the_gated_arm_only_ever_picks_inside_the_gate(built):
    """A-gating is applied *before* semantic selection, not after it."""

    index = np.flatnonzero(built.gate)
    picked = arms.select(
        "proposed", built, cost_of=lambda _: 4, answer_budget=400, seed=0,
        semantic=built.candidates.embeddings[index],
    )
    assert picked.anchors, "nothing was selected"
    inside = set(index.tolist())
    assert all(anchor in inside for anchor in picked.anchors), (
        "a pick landed outside the admissible subset; the gate is not being "
        "applied before the traversal"
    )
    # and the images it opened are the images of those gated candidates
    opened = set(picked.images)
    assert opened == {
        str(built.candidates.image_ids[a]) for a in picked.anchors
    }


def test_coverage_does_not_reduce_to_the_admissibility_ranking(built):
    """The whole point, and the exact way Method V3 failed to test it.

    There, ``A`` and ``A*C`` reported four identical aggregate statistics and
    nobody checked whether the *proposals* differed — the audit later found the
    ranking was dense to 3 parts in 100,000, so the aggregates were blind. Here
    the check is on the selection itself: a coverage traversal that had collapsed
    into "take the top of the tie-break" would open exactly the images the
    admissibility arm opens, in the same order.
    """

    index = np.flatnonzero(built.gate)
    cost_of = lambda _: 4
    coverage_arm = arms.select(
        "proposed", built, cost_of=cost_of, answer_budget=400, seed=0,
        semantic=built.candidates.embeddings[index],
    )
    ranking_arm = arms.select(
        "admissibility", built, cost_of=cost_of, answer_budget=400, seed=0
    )
    assert coverage_arm.images != ranking_arm.images
    shared = set(coverage_arm.images) & set(ranking_arm.images)
    assert len(shared) < 0.9 * len(coverage_arm.images), (
        f"{len(shared)} of {len(coverage_arm.images)} opened images are the "
        "admissibility arm's; the traversal is behaving like a static ranking"
    )


def test_the_traversal_reorders_its_own_tie_break(built):
    """Picks must not come out in descending admissibility order."""

    index = np.flatnonzero(built.gate)
    picked = arms.select(
        "proposed", built, cost_of=lambda _: 4, answer_budget=200, seed=0,
        semantic=built.candidates.embeddings[index],
    )
    order = built.admissibility[list(picked.anchors)]
    assert not np.all(np.diff(order) <= 0), (
        "every pick was less admissible than the one before it, which is what a "
        "pure A ranking looks like"
    )


def test_acquisition_changes_what_the_traversal_wants_next(built):
    """Coverage is updated by acquisition; it is not a fixed score.

    Two traversals over the same pool, one starting from an empty reference and
    one from a reference that already covers part of the space, must not agree —
    otherwise ``R`` is decorative.
    """

    index = np.flatnonzero(built.gate)
    features = built.candidates.embeddings[index]
    cost_of = lambda _: 4
    fresh = coverage.kcenter_greedy(
        features, built.candidates.image_ids[index], cost_of=cost_of, budget=200,
        tie_break=built.admissibility[index],
    )
    seeded = coverage.kcenter_greedy(
        features, built.candidates.image_ids[index], cost_of=cost_of, budget=200,
        reference=features[: len(features) // 2],
        tie_break=built.admissibility[index],
    )
    assert fresh.images != seeded.images
    assert seeded.reference_size > 0 and fresh.reference_size == 0


def test_every_pick_is_at_least_as_far_as_the_next_one_would_have_been(built):
    """Farthest-first: the distances it accepts are non-increasing.

    Opening an image can only shrink the remaining minimum distances, so the
    sequence of accepted distances cannot rise. A rise would mean the traversal
    is not taking the argmax.
    """

    index = np.flatnonzero(built.gate)
    result = coverage.kcenter_greedy(
        built.candidates.embeddings[index], built.candidates.image_ids[index],
        cost_of=lambda _: 4, budget=200,
        reference=built.candidates.embeddings[index][:50],
        tie_break=built.admissibility[index],
    )
    taken = np.asarray(result.distances)
    assert taken.size > 5
    assert np.all(np.diff(taken) <= 1e-6), taken[:10]

# ------------------------------------------- no oracle reaches the ranking ---


def test_no_selection_module_can_reach_an_answer():
    """Static: the acquisition path never calls ``Candidates.oracle()``.

    ``tests/test_owl.py::test_scoring_never_reads_an_answer`` makes the same
    guarantee for the additive score. This is its counterpart for the modules
    Benchmark V1 added, and it is a source check rather than a behavioural one
    because a call added on a rarely-taken branch would pass the behavioural
    test for months.
    """

    import inspect

    from owl.active_selection import population as population_module

    for module in (arms, coverage, population_module, budget):
        source = inspect.getsource(module)
        assert ".oracle()" not in source, module.__name__
        assert "oracle_kind" not in source, module.__name__
        assert "class_name" not in source or module is budget, module.__name__


def test_the_cost_function_reads_counts_and_never_a_class():
    """The one thing the selector *is* handed that touches the annotation.

    It answers "how many objects are on this image", which is the price of a
    question. It cannot answer "which classes", and a selector that wanted to
    could not get there through this interface.
    """

    import inspect

    source = inspect.getsource(budget.cost_function)
    assert "sum(" in inspect.getsource(budget.image_cost)
    # the closure returns an int, so there is nothing else to read off it
    cost = budget.cost_function({"i1": {"person": 2, "bear": 1}})
    assert cost("i1") == 3
    assert isinstance(cost("i1"), int)
    assert "class" not in source


def test_the_future_label_helpers_are_not_imported_by_the_selectors():
    """``acquisition`` reads future classes; it must run only after the budget.

    It lives in :mod:`owl.active_selection.budget` beside the cost function, so
    the separation cannot be by module. It is enforced by the call site instead:
    :func:`owl.runner.run_chain` calls it *after* the selector has returned and
    the images have been committed, and the arm registry never mentions it.
    """

    import inspect

    assert "acquisition" not in inspect.getsource(arms)
    assert "acquisition" not in inspect.getsource(coverage)
    runner_source = inspect.getsource(
        __import__("owl.runner", fromlist=["runner"])
    )
    selector_call = runner_source.index("bought = selector(")
    acquisition_call = runner_source.index("annotation_budget.acquisition(")
    assert selector_call < acquisition_call, (
        "the future-label table is computed before the selector has spent the "
        "budget"
    )


# ---------------------------------------------------------- Proposed-v2 ---
#
# Development-seed-informed, appended 2026-09-04, NOT pre-registered. These
# tests pin what it is; `tests/test_full_benchmark_chain.py` pins that it runs
# the chain and that it leaves the earlier arms alone.


def test_v2_ranks_the_gated_above_median_uncertainty_subset(built):
    from owl import scoring as scoring_module

    index = arms.ranked_positions("proposed_v2", built)
    gate = np.flatnonzero(built.gate)
    entropy = scoring_module.uncertainty(built.candidates, "entropy")

    assert set(index.tolist()) <= set(gate.tolist()), "v2 left the gate"
    threshold = float(np.median(entropy[built.gate]))
    assert entropy[index].min() >= threshold
    # `>=` keeps half the gate, plus any ties sitting exactly at the median
    assert gate.size // 2 <= index.size <= gate.size // 2 + 1 + gate.size // 100


def test_the_v2_median_is_taken_inside_the_gate_not_over_the_pool(built):
    """The trap: a pool-wide median keeps an unfixed share of the gate."""

    from owl import scoring as scoring_module

    entropy = scoring_module.uncertainty(built.candidates, "entropy")
    inside = float(np.median(entropy[built.gate]))
    whole = float(np.median(entropy))
    assert inside != whole, "the fixture cannot distinguish the two medians"

    index = arms.ranked_positions("proposed_v2", built)
    pool_wide = np.flatnonzero(built.gate & (entropy >= whole))
    assert index.size != pool_wide.size
    assert entropy[index].min() >= inside


def test_v2_ranks_about_half_of_what_v1_ranks(built):
    v1 = arms.ranked_positions("proposed", built).size
    v2 = arms.ranked_positions("proposed_v2", built).size
    assert 0.45 * v1 <= v2 <= 0.55 * v1
    assert arms.ranked_share("proposed_v2") == pytest.approx(
        arms.ranked_share("proposed") * 0.5
    )


def test_v2_does_not_change_what_the_earlier_arms_rank(built):
    """Adding an arm must not move a measured one."""

    assert np.array_equal(
        arms.ranked_positions("proposed", built), np.flatnonzero(built.gate)
    )
    for name in ("random", "admissibility", "entropy", "coreset"):
        assert np.array_equal(
            arms.ranked_positions(name, built), np.arange(len(built))
        )


def test_the_earlier_arms_keep_their_defaults():
    for name in ("random", "admissibility", "entropy", "proposed", "coreset"):
        spec = arms.ARMS[name]
        assert spec.informative is False, name
        assert spec.reference_scope == "labelled", name
    v2 = arms.ARMS["proposed_v2"]
    assert v2.informative is True
    assert v2.reference_scope == "trajectory"
    assert v2.gated is True and v2.needs_semantic is True


def test_v2_is_last_in_the_declared_order():
    """It was designed after seeing results, so it may not displace a baseline."""

    assert arms.ORDER[-1] == "proposed_v2"
    assert arms.ORDER[:5] == (
        "random", "admissibility", "proposed", "entropy", "coreset"
    )
    assert set(arms.ORDER) == set(arms.ARMS)


def test_v2_refuses_features_for_the_wrong_subset(built):
    with pytest.raises(arms.ArmError, match="U >= median"):
        arms.select(
            "proposed_v2", built, cost_of=lambda _: 1, answer_budget=10, seed=0,
            semantic=built.candidates.embeddings[np.flatnonzero(built.gate)],
        )


def test_v2_only_ever_picks_inside_its_own_subset(built):
    index = arms.ranked_positions("proposed_v2", built)
    picked = arms.select(
        "proposed_v2", built, cost_of=lambda _: 4, answer_budget=400, seed=0,
        semantic=built.candidates.embeddings[index],
    )
    assert picked.anchors
    assert set(picked.anchors) <= set(index.tolist())


# ------------------------------------------- the empty-reference first pick ---


def test_an_empty_reference_takes_the_most_admissible_candidate_first():
    """v2's t2 starts with nothing bought. The rule is stated, not inherited."""

    features = np.eye(4, dtype=np.float32)
    result = coverage.kcenter_greedy(
        features, np.asarray(["a", "b", "c", "d"]),
        cost_of=lambda _: 1, budget=1,
        tie_break=np.asarray([0.1, 0.9, 0.5, 0.4]),
    )
    assert result.images == ("b",)


def test_a_tie_at_the_maximum_breaks_to_the_lowest_index():
    features = np.eye(4, dtype=np.float32)
    result = coverage.kcenter_greedy(
        features, np.asarray(["a", "b", "c", "d"]),
        cost_of=lambda _: 1, budget=1,
        tie_break=np.asarray([0.1, 0.9, 0.5, 0.9]),
    )
    assert result.images == ("b",), "the later equal candidate won"


def test_an_undefined_first_distance_is_none_and_never_infinity():
    """`inf` would reach the manifest as a value strict JSON rejects."""

    import json
    import math

    features = np.eye(3, dtype=np.float32)
    result = coverage.kcenter_greedy(
        features, np.asarray(["a", "b", "c"]), cost_of=lambda _: 1, budget=3,
        tie_break=np.asarray([0.9, 0.5, 0.1]),
    )
    assert result.distances[0] is None
    assert all(d is not None and math.isfinite(d) for d in result.distances[1:])
    summary = result.summary()
    assert summary["coverage_first_pick_distance"] is None
    assert summary["coverage_picks_without_reference"] == 1
    assert summary["coverage_mean_pick_distance"] is not None
    json.dumps(summary, allow_nan=False)          # raises on inf/nan


def test_a_non_empty_reference_leaves_no_undefined_distance():
    features = np.eye(3, dtype=np.float32)
    result = coverage.kcenter_greedy(
        features, np.asarray(["a", "b", "c"]), cost_of=lambda _: 1, budget=2,
        reference=features[:1], tie_break=np.asarray([0.9, 0.5, 0.1]),
    )
    assert all(d is not None for d in result.distances)
    assert result.summary()["coverage_picks_without_reference"] == 0


# ------------------------------------------------------------ CUDA memory ---


def test_releasing_the_backbone_is_safe_without_torch():
    from owl.active_selection import semantic

    report = semantic.release(device="cpu")
    assert report["gc_collected"] >= 0
    assert report["torch"] in (True, False)


def test_the_semantic_pass_releases_the_backbone_even_when_it_fails(tmp_path, monkeypatch):
    """In a `finally`: a failed pass must not leave a full card behind."""

    from owl.active_selection import semantic

    released: list[dict] = []
    monkeypatch.setattr(semantic, "release", lambda **kw: released.append(kw) or {})

    class Backbone:
        pass

    def exploding(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(semantic, "embed", exploding)
    with pytest.raises(RuntimeError, match="out of memory"):
        semantic.cached(
            tmp_path / "f.npz", np.asarray(["1"]), np.zeros((1, 4)), tmp_path,
            model_factory=lambda _d: Backbone(), device="cuda",
        )
    assert released == [{"device": "cuda"}], "the backbone was not released"


def test_the_semantic_pass_releases_the_backbone_on_success(tmp_path, monkeypatch):
    from owl.active_selection import semantic

    released: list[dict] = []
    monkeypatch.setattr(semantic, "release", lambda **kw: released.append(kw) or {})
    monkeypatch.setattr(
        semantic, "embed",
        lambda *a, **k: np.eye(1, semantic.sf.FEATURE_DIM, dtype=np.float32),
    )
    semantic.cached(
        tmp_path / "f.npz", np.asarray(["1"]), np.zeros((1, 4)), tmp_path,
        model_factory=lambda _d: object(), device="cuda",
    )
    assert released == [{"device": "cuda"}]
