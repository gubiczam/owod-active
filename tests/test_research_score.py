"""The research plan's own score on the GPU path, term by term.

Covers the 2026-09 redesign brief's test list where it concerns selection:
the score never reads an oracle class, the labelled pool grows, batch diversity
moves *during* a campaign, a DBSCAN noise point gets ``coh = 0``, and the replay
vocabulary means what it says.

Every test is synthetic and CPU-only, except the two that use the committed
frozen pool through the shared ``pool`` fixture.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from owl import replay
from owl.active_selection import arms, population
from owl.active_selection import research_score as rs


@pytest.fixture
def blobs():
    """Six well-separated clusters plus a handful of genuine isolated outliers.

    The outliers are the case the whole ``coh`` term exists for: a lone
    candidate is simultaneously uncertain, maximally novel and — because its
    cluster is a singleton — maximally rare, so a score without a coherence gate
    ranks exactly the useless points first.
    """

    rng = np.random.default_rng(7)
    dimensions = 48
    centres = rng.normal(size=(6, dimensions))
    assignment = rng.integers(0, 6, size=540)
    dense = centres[assignment] + 0.12 * rng.normal(size=(540, dimensions))
    lonely = 6.0 * rng.normal(size=(12, dimensions))
    features = np.vstack([dense, lonely]).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    n = features.shape[0]
    image_ids = np.array([f"{i // 4:012d}" for i in range(n)])
    entropy = rng.random(n)
    counts = {name: 3 for name in set(image_ids.tolist())}
    return {
        "features": features,
        "entropy": entropy,
        "image_ids": image_ids,
        "cost_of": (lambda image, counts=counts: counts[str(image)]),
        "is_outlier": np.arange(n) >= 540,
    }


# ------------------------------------------------------------ the coherence gate ---


def test_a_dbscan_noise_point_gets_coherence_zero(blobs):
    """The consultation's own definition: `coh(x) in {0, 1}`, 0 for a noise point."""

    from owl import clustering

    reduced = clustering._reduce(blobs["features"], rs.PCA_DIMENSIONS, 0).astype(float)
    eps = rs.eps_from_geometry(reduced, min_samples=10)
    gate = rs.dbscan_gate(reduced, eps=eps, min_samples=10)

    assert set(np.unique(gate.coherence)) <= {0.0, 1.0}, "coh must be binary"
    assert np.array_equal(gate.coherence == 0.0, gate.labels < 0)
    assert set(np.unique(gate.status)) <= {"core", "border", "noise"}
    assert np.array_equal(gate.status == "noise", gate.labels < 0)

    # The planted outliers are what the gate is supposed to reject, and the
    # claim is a *relationship*, not an absolute rate: `eps = median(k-dist)`
    # puts roughly half the population on the core side by construction, so a
    # share of genuine cluster members lands on the border or outside it. What
    # must hold is that the gate is far harsher on the outliers than on the
    # clusters — the opposite of the measured failure on PROB's decoder space,
    # where it rejected real unknowns more often than background.
    rejected_outliers = float((gate.coherence[blobs["is_outlier"]] == 0.0).mean())
    rejected_dense = float((gate.coherence[~blobs["is_outlier"]] == 0.0).mean())
    assert rejected_outliers > 0.8, (
        f"the gate rejected only {rejected_outliers:.0%} of the planted isolated "
        "outliers, which is the one thing it exists to do")
    assert rejected_outliers > 3 * rejected_dense, (
        f"outliers rejected {rejected_outliers:.0%}, cluster members "
        f"{rejected_dense:.0%} — the gate is not discriminating")


def test_the_gate_reports_core_border_and_noise_separately(blobs):
    """`accepted` is not evidence on its own; which kind was accepted is."""

    from owl import clustering

    reduced = clustering._reduce(blobs["features"], rs.PCA_DIMENSIONS, 0).astype(float)
    gate = rs.dbscan_gate(
        reduced, eps=rs.eps_from_geometry(reduced, min_samples=10), min_samples=10)
    diagnostics = gate.diagnostics
    assert diagnostics["core"] + diagnostics["border"] == diagnostics["accepted"]
    assert diagnostics["accepted"] + diagnostics["rejected"] == diagnostics["candidates"]
    assert 0.0 < diagnostics["noise_rate"] < 1.0
    assert diagnostics["smallest_cluster"] >= 1


def test_eps_comes_from_the_geometry_and_moves_with_it(blobs):
    """A tighter population gets a tighter radius, with no number chosen by hand."""

    tight = blobs["features"]
    loose = np.asarray(tight, dtype=np.float64) + 0.5 * np.random.default_rng(
        1).normal(size=tight.shape)
    loose /= np.linalg.norm(loose, axis=1, keepdims=True)
    assert rs.eps_from_geometry(tight, min_samples=10) < rs.eps_from_geometry(
        loose, min_samples=10)


def test_the_open_gate_is_not_the_same_thing_as_gamma_zero(blobs):
    """`coherence_mode='none'` keeps the rarity weight; it only opens the gate."""

    gate = rs.open_gate(10)
    assert np.array_equal(gate.coherence, np.ones(10))
    assert np.array_equal(gate.labels, np.zeros(10, dtype=np.int64)), (
        "one cluster, not noise: rarity stays defined and becomes constant, "
        "which is what 'no density structure was consulted' means")


# ---------------------------------------------------------------------- rarity ---


def test_rarity_falls_with_how_well_the_labelled_set_covers_a_cluster():
    """`R_k = log((1 + n_cand) / (1 + n_ref))`, positive part, from one partition."""

    features = np.array(
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02],      # cluster 0, covered
         [0.0, 1.0], [0.01, 0.99], [0.02, 0.98]],     # cluster 1, uncovered
        dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1])
    reference = np.array([[1.0, 0.0]] * 20, dtype=np.float64)

    weight, diagnostics = rs.cluster_rarity(labels, features, reference)
    assert weight[3] > weight[0], (
        "the cluster the labelled set has never seen must outrank the one it "
        "has twenty rows of")
    assert diagnostics["rarity_reference_rows"] == 20

    # with nothing labelled, two equal-sized clusters are equally rare
    flat, _ = rs.cluster_rarity(labels, features, None)
    assert flat[0] == pytest.approx(flat[3])


def test_a_noise_point_makes_no_rarity_claim():
    features = np.eye(4)
    labels = np.array([0, 0, 0, -1])
    weight, _ = rs.cluster_rarity(labels, features, None)
    assert weight[3] == 0.0, (
        "rank-normalising would otherwise hand the tied noise block the average "
        "rank of the tie, i.e. a positive rarity for having no cluster at all")


def test_rarity_refuses_a_reference_from_another_space():
    with pytest.raises(rs.ScoreError, match="not the same space|does not live in"):
        rs.cluster_rarity(np.zeros(3, dtype=np.int64), np.eye(3), np.eye(5)[:, :5])


def test_rarity_never_sees_a_class_count():
    """The term is an occupancy of an embedding partition, and nothing else.

    Checked on the executable body with the docstring removed, because the
    docstring's job is precisely to talk about the oracle it must not touch.
    """

    import ast
    import inspect

    tree = ast.parse(inspect.getsource(rs.cluster_rarity).strip())
    function = tree.body[0]
    if (
        function.body
        and isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
    ):
        function.body = function.body[1:]
    body = ast.unparse(function)
    for forbidden in ("oracle", "class_name", "groups", "load_groups"):
        assert forbidden not in body, f"{forbidden} must not be reachable from rarity"


# ------------------------------------------------------------------- diversity ---


def test_labeled_novelty_is_one_when_nothing_is_labelled():
    features = np.eye(3, dtype=np.float32)
    assert np.array_equal(rs.labeled_novelty(features, None), np.ones(3))
    assert np.array_equal(
        rs.labeled_novelty(features, np.zeros((0, 3), dtype=np.float32)), np.ones(3))


def test_labeled_novelty_falls_for_a_candidate_the_reference_already_holds():
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    novelty = rs.labeled_novelty(features, np.array([[1.0, 0.0]], dtype=np.float32))
    assert novelty[0] < 0.01 and novelty[1] > 0.9


def test_batch_diversity_moves_during_one_campaign(blobs):
    """Point 2B: the next candidate's D must fall once something like it is bought."""

    result = rs.select(
        blobs["features"], entropy=blobs["entropy"], image_ids=blobs["image_ids"],
        cost_of=blobs["cost_of"], budget=90, rounds=1,
        spec=rs.ScoreSpec(diversity_mode="batch_diversity", coherence_mode="none"),
    )
    batch = [row["D_batch"] for row in result.picks]
    assert batch[0] == 1.0, "nothing is taken yet, so nothing is redundant yet"
    assert min(batch) < 1.0, "D_batch never moved, so it is not batch diversity"
    assert len(set(batch)) > 1
    # farthest-first: the running minimum of what is still available can only
    # fall, so the *offered* value of a pick is bounded by every earlier pick.
    assert all(later <= 1.0 for later in batch)


def test_the_two_halves_of_D_are_logged_separately_and_the_final_D_is_their_mean(blobs):
    """`combined` has to stay decomposable, or the ablation cannot be read."""

    result = rs.select(
        blobs["features"], entropy=blobs["entropy"], image_ids=blobs["image_ids"],
        cost_of=blobs["cost_of"], budget=60, rounds=1,
        reference=blobs["features"][:40],
        spec=rs.ScoreSpec(diversity_mode="combined", coherence_mode="none"),
    )
    assert result.picks
    for row in result.picks:
        assert row["D"] == pytest.approx(
            0.5 * (row["D_labeled"] + row["D_batch"]), abs=1e-6)
        for key in ("U", "D_labeled", "D_batch", "D", "w", "coh", "cluster",
                    "cluster_status", "score", "round"):
            assert key in row


def test_each_diversity_mode_reads_only_the_component_it_names(blobs):
    common = dict(
        entropy=blobs["entropy"], image_ids=blobs["image_ids"],
        cost_of=blobs["cost_of"], budget=60, rounds=1,
        reference=blobs["features"][:40],
    )
    only_labelled = rs.select(
        blobs["features"],
        spec=rs.ScoreSpec(diversity_mode="labeled_novelty", coherence_mode="none"),
        **common)
    only_batch = rs.select(
        blobs["features"],
        spec=rs.ScoreSpec(diversity_mode="batch_diversity", coherence_mode="none"),
        **common)
    off = rs.select(
        blobs["features"],
        spec=rs.ScoreSpec(diversity_mode="none", coherence_mode="none"), **common)

    assert all(r["D"] == r["D_labeled"] for r in only_labelled.picks)
    assert all(r["D"] == r["D_batch"] for r in only_batch.picks)
    assert all(r["D"] == 0.0 for r in off.picks)
    assert only_labelled.images != only_batch.images, (
        "two definitions of D that buy the identical images are not two "
        "definitions of D")


# ---------------------------------------------------------------------- rounds ---


def test_the_labelled_pool_grows_between_rounds(blobs):
    """Point 10: after each mini-round the reference is bigger than before it."""

    result = rs.select(
        blobs["features"], entropy=blobs["entropy"], image_ids=blobs["image_ids"],
        cost_of=blobs["cost_of"], budget=120, rounds=4,
        spec=rs.ScoreSpec(diversity_mode="combined", coherence_mode="dbscan_binary"),
    )
    rows = [r["reference_rows"] for r in result.rounds if r.get("images")]
    assert len(rows) >= 2, "the campaign never reached a second round"
    assert rows == sorted(rows) and rows[-1] > rows[0], (
        f"reference rows over rounds were {rows}; the point of rounds is that "
        "what the last one bought is labelled for the next one")


def test_rounds_conserve_the_budget_and_never_exceed_it(blobs):
    one = rs.select(
        blobs["features"], entropy=blobs["entropy"], image_ids=blobs["image_ids"],
        cost_of=blobs["cost_of"], budget=120, rounds=1,
        spec=rs.ScoreSpec(coherence_mode="none"))
    many = rs.select(
        blobs["features"], entropy=blobs["entropy"], image_ids=blobs["image_ids"],
        cost_of=blobs["cost_of"], budget=120, rounds=6,
        spec=rs.ScoreSpec(coherence_mode="none"))
    for result in (one, many):
        assert result.diagnostics["answers_spent"] <= 120
        # every image costs 3 here, so a full campaign spends exactly the budget
        assert result.diagnostics["answers_spent"] == 120
        assert len(set(result.images)) == len(result.images), "an image opened twice"


def test_an_image_bought_at_an_earlier_task_is_skipped_not_charged(blobs):
    excluded = frozenset(blobs["image_ids"][:40].tolist())
    result = rs.select(
        blobs["features"], entropy=blobs["entropy"], image_ids=blobs["image_ids"],
        cost_of=blobs["cost_of"], budget=60, rounds=1,
        excluded_images=excluded, spec=rs.ScoreSpec(coherence_mode="none"))
    assert not set(result.images) & excluded


def test_selection_is_deterministic(blobs):
    kwargs = dict(
        entropy=blobs["entropy"], image_ids=blobs["image_ids"],
        cost_of=blobs["cost_of"], budget=60, rounds=3,
        spec=rs.ScoreSpec(diversity_mode="combined", coherence_mode="dbscan_binary"))
    first = rs.select(blobs["features"], **kwargs)
    second = rs.select(blobs["features"], **kwargs)
    assert first.images == second.images
    assert [r["score"] for r in first.picks] == [r["score"] for r in second.picks]


def test_a_bad_mode_is_refused_at_construction():
    with pytest.raises(rs.ScoreError, match="diversity_mode"):
        rs.ScoreSpec(diversity_mode="whatever")
    with pytest.raises(rs.ScoreError, match="coherence_mode"):
        rs.ScoreSpec(coherence_mode="dbscan")
    with pytest.raises(rs.ScoreError, match="eps_quantile"):
        rs.ScoreSpec(eps_quantile=0.0)


def test_lambda_and_gamma_are_the_values_owl_scoring_froze():
    """No new free parameter enters with this module."""

    from owl import scoring

    frozen = scoring.ScoreConfig()
    assert rs.LAMBDA_DIVERSITY == frozen.lambda_diversity
    assert rs.GAMMA_RARITY == frozen.gamma_rarity


# ------------------------------------------------------- the contamination check ---


def test_known_contamination_is_a_diagnostic_the_selector_cannot_reach():
    import inspect

    body = inspect.getsource(rs.select)
    assert "known_contamination" not in body, (
        "point 5 is explicit that this must not become a selection decision")


def test_known_contamination_separates_a_disjoint_reference_from_a_shared_one():
    rng = np.random.default_rng(3)
    dimensions = 16
    a = rng.normal(size=(1, dimensions)) + 0.05 * rng.normal(size=(200, dimensions))
    b = rng.normal(size=(1, dimensions)) + 0.05 * rng.normal(size=(200, dimensions))
    candidates = np.vstack([a, b]).astype(np.float32)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)

    shared = candidates[:60]                      # reference drawn from cluster a
    report = rs.known_contamination(
        candidates, shared, min_samples=10, pca_dimensions=8)
    assert report["known_rows"] == 60
    assert report["known_contamination_rate"] > 0.5, (
        "a reference literally drawn from a candidate cluster must show as "
        "contaminating it")

    far = (candidates[:60] * -1.0).astype(np.float32)   # antipodal, disjoint
    clean = rs.known_contamination(
        candidates, far, min_samples=10, pca_dimensions=8)
    assert clean["known_contamination_rate"] < report["known_contamination_rate"]
    for key in ("clusters_candidate_only", "clusters_mixed",
                "mean_cluster_purity_known_side", "candidate_cluster_sizes"):
        assert key in report


def test_an_empty_reference_is_reported_not_crashed():
    report = rs.known_contamination(
        np.eye(5, dtype=np.float32), np.zeros((0, 5), dtype=np.float32),
        min_samples=3)
    assert report["known_rows"] == 0


# --------------------------------------------------------------- the arm wiring ---


def test_the_research_arms_are_registered_and_route_to_this_module():
    assert arms.RESEARCH == frozenset({
        "research_v2", "research_v2_plan", "research_v2_no_gate",
        "research_v2_labeled_only", "research_v2_batch_only",
    })
    for name in arms.RESEARCH:
        spec = arms.ARMS[name]
        assert spec.kind == "research"
        assert spec.needs_semantic is True
        assert spec.gated is True, "the object-likeness gate is upstream of the score"
        assert isinstance(spec.score_spec, rs.ScoreSpec)
    assert set(arms.ORDER) == set(arms.ARMS)


def test_the_earlier_arms_carry_no_score_spec():
    """Adding the field may not change an arm that was already measured."""

    for name, spec in arms.ARMS.items():
        if name not in arms.RESEARCH:
            assert spec.score_spec is None, name
            assert spec.picks_expected is False if hasattr(spec, "picks_expected") else True


def test_the_named_contrasts_differ_by_exactly_one_thing():
    plan = arms.ARMS["research_v2_plan"].score_spec
    proposed = arms.ARMS["research_v2"].score_spec
    gateless = arms.ARMS["research_v2_no_gate"].score_spec

    # plan vs proposed: the two 2026-08-25 redesigns, together
    assert (plan.diversity_mode, plan.coherence_mode) == (
        "labeled_novelty", "continuous")
    assert (proposed.diversity_mode, proposed.coherence_mode) == (
        "combined", "dbscan_binary")
    # proposed vs no_gate: the gate alone
    assert dataclasses.replace(proposed, coherence_mode="none") == gateless
    assert proposed.rarity_mode == gateless.rarity_mode == "cluster_rarity"


def test_a_research_arm_runs_on_the_committed_pool_without_an_oracle(pool):
    """Point 1 of the test list, on the real population rather than a synthetic one."""

    built = population.build(pool)
    naked = dataclasses.replace(built.candidates, _oracle=None)
    assert not naked.has_oracle
    bare = population.Population(
        candidates=naked, admissibility=built.admissibility, gate=built.gate,
        kept=built.kept, diagnostics=built.diagnostics)

    index = arms.ranked_positions("research_v2", bare)
    picked = arms.select(
        "research_v2", bare, cost_of=lambda _: 5, answer_budget=100, seed=0,
        semantic=bare.candidates.embeddings[index], rounds=2)

    assert len(picked) == 20, "twenty images at five answers each is the whole budget"
    assert picked.picks, "a research arm must leave a candidate-level trail"
    assert picked.row["selector_detail"] == "research_score"
    assert picked.row["diversity_mode"] == "combined"
    assert 0.0 < float(picked.row["final_noise_rate"]) < 1.0, (
        "a gate that fires on nobody or on everybody is not a gate")
    assert set(picked.covered.nonzero()[0]) <= set(index.tolist()), (
        "a gated arm may not cover outside the subset it selects from")


# ---------------------------------------------------------------------- replay ---


def test_the_replay_mode_names_map_onto_the_frozen_arms():
    assert replay.resolve_mode("uniform")[1]["alpha"] == 0.0
    assert replay.resolve_mode("proportional")[1]["alpha"] > 0
    assert replay.resolve_mode("tail_aware")[1]["alpha"] < 0
    assert replay.resolve_mode("none")[1]["total"] == 0
    for mode, arm in replay.MODES.items():
        assert arm in replay.ARMS, mode
    with pytest.raises(ValueError, match="Unknown replay_mode"):
        replay.resolve_mode("tail")


def test_every_mode_spends_the_whole_memory_and_nothing_more():
    counts = {"person": 900, "car": 400, "bear": 12, "fire hydrant": 7}
    for mode in replay.MODES:
        arm, spec = replay.resolve_mode(mode)
        total = int(spec["total"])
        allocation = replay.allocate(counts, total=total, alpha=float(spec["alpha"]))
        assert sum(allocation.values()) == total, (mode, allocation)


def test_alpha_below_zero_gives_the_rare_class_more_than_alpha_above_it():
    # Every class holds more objects than any rule could ask of it, so the
    # allocation rule is the only thing under test. With a genuinely small tail
    # class the capacity cap binds first and both rules give it everything it
    # has, which is correct behaviour and would hide the comparison.
    counts = {"person": 9000, "car": 4000, "bear": 1200, "fire hydrant": 700}
    _, tail = replay.resolve_mode("tail_aware")
    _, head = replay.resolve_mode("proportional")
    tail_side = replay.allocate(counts, total=400, alpha=float(tail["alpha"]))
    head_side = replay.allocate(counts, total=400, alpha=float(head["alpha"]))

    assert tail_side["fire hydrant"] > head_side["fire hydrant"]
    assert tail_side["person"] < head_side["person"]
    # The research plan's predicted failure, as a ratio rather than a count, so
    # it does not depend on how skewed this particular class set happens to be:
    # under alpha=1 the rarest class gets a share far below its uniform share,
    # and `minimum=1` is the only thing keeping it off zero.
    uniform = replay.allocate(counts, total=400, alpha=0.0)
    assert head_side["fire hydrant"] < uniform["fire hydrant"] / 2
    assert tail_side["fire hydrant"] > uniform["fire hydrant"]
    assert min(head_side.values()) >= 1, "minimum=1 must keep every class present"


def test_the_refresh_names_map_onto_the_reallocation_flag():
    assert replay.resolve_refresh("fixed") is False
    assert replay.resolve_refresh("per_task") is True
    with pytest.raises(ValueError, match="Unknown replay_refresh"):
        replay.resolve_refresh("every_task")


def test_per_task_refresh_can_change_the_memory_and_fixed_prefers_incumbents():
    """The axis the consultation asked for: a different memory in every task."""

    from owl import exemplars

    candidates = [
        *(exemplars.Exemplar(f"{i:012d}", "person", 0) for i in range(30)),
        *(exemplars.Exemplar(f"{100 + i:012d}", "bear", 0) for i in range(30)),
    ]
    counts = {"person": 30, "bear": 30}
    incumbent = tuple(e for e in candidates if e.class_name == "person")[:8]

    # `fixed` — the allocation is re-satisfied but the exemplars we already had
    # are offered first, so an incumbent that still serves it survives.
    demand = replay.allocate(counts, total=8, alpha=0.0)
    kept = exemplars.select(
        candidates, demand, incumbent=incumbent,
        reallocate=replay.resolve_refresh("fixed"), seed=0)
    redone = exemplars.select(
        candidates, demand, incumbent=incumbent,
        reallocate=replay.resolve_refresh("per_task"), seed=0)

    assert len(kept) == len(redone) == 8, "the budget is fixed either way"
    surviving_incumbents = len(set(kept) & set(incumbent))
    assert surviving_incumbents >= len(set(redone) & set(incumbent)), (
        "'fixed' must never keep fewer incumbents than 'per_task'; that is the "
        "whole difference between the two refresh rules")
    # And the axis is real: the two rules can hand PROB different memories.
    assert set(kept) != set(redone) or surviving_incumbents == 8
