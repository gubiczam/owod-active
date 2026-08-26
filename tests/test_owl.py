"""One assertion per published claim, including the ones that came out badly."""

from __future__ import annotations

import numpy as np
import pytest

from owl import clustering, labelling, metrics, proposals, protocol, replay, scoring, selection

# ------------------------------------------------------------------ protocol ---


def test_chain_declares_one_class_per_task():
    chain = protocol.build_chain(10)
    assert len(chain) == 10
    assert chain[0].is_anchor and chain[0].n_current == 19
    assert [t.n_current for t in chain] == list(range(19, 29))
    assert all(t.new_class is not None for t in chain[1:])


def test_chain_follows_the_evaluators_class_order():
    """PROB indexes classes by position, so a task may not skip one."""
    chain = protocol.build_chain(10)
    declared = [t.new_class for t in chain[1:]]
    assert declared == list(protocol.CLASS_ORDER[19:28])


def test_chain_spans_all_three_frequency_groups():
    groups = protocol.load_groups()
    declared = {groups[t.new_class] for t in protocol.build_chain(10)[1:]}
    assert declared == {"head", "medium", "tail"}


def test_chain_refuses_to_run_off_the_end():
    with pytest.raises(protocol.ProtocolError):
        protocol.build_chain(100)


# ----------------------------------------------------------------- proposals ---


def test_frozen_pool_splits_are_disjoint_and_populated(pool):
    evaluation = proposals.from_frozen_pool(split="eval")
    assert len(pool) == 80_000 and len(evaluation) == 40_000
    assert not set(pool.image_ids) & set(evaluation.image_ids)


def test_embeddings_are_unit_norm_and_posteriors_are_distributions(pool):
    assert np.allclose(np.linalg.norm(pool.embeddings, axis=1), 1.0, atol=1e-4)
    assert np.allclose(pool.posterior.sum(axis=1), 1.0, atol=1e-4)


def test_a_pool_without_answers_refuses_to_invent_them(pool):
    bare = proposals.Candidates(
        pool.image_ids, pool.boxes, pool.embeddings, pool.posterior, pool.objectness
    )
    with pytest.raises(ValueError, match="no oracle"):
        bare.oracle()


# ---------------------------------------------------------------- clustering ---


def test_contamination_uses_enrichment_not_a_bare_majority(pool):
    """An absolute-majority rule is degenerate on a pool that is 81% background."""
    partition = clustering.fit(pool.embeddings, n_clusters=800, seed=0)
    report = clustering.contamination(partition, pool.oracle().kind == "known")
    assert 0.0 < report["contamination"] < 0.5
    assert report["unknown_recall"] > 0.5
    assert report["baseline_known_share"] == pytest.approx(0.1514, abs=0.01)


def test_more_clusters_leak_less_known_content(pool):
    known = pool.oracle().kind == "known"
    leaks = [
        clustering.contamination(clustering.fit(pool.embeddings, n_clusters=k, seed=0), known)[
            "contamination"
        ]
        for k in (200, 800, 3200)
    ]
    assert leaks == sorted(leaks, reverse=True)


def test_the_tuner_refuses_a_partition_that_discovers_nothing(pool):
    known = clustering.predicted_known(pool.posterior, 19)
    with pytest.raises(ValueError, match="unknown_recall"):
        clustering.tune(pool.embeddings, known, grid=(200,), min_unknown_recall=1.01)


def test_the_detector_can_name_its_own_known_classes_without_an_annotation(pool):
    """The contamination diagnostic has to run before any oracle is paid."""
    estimate = clustering.predicted_known(pool.posterior, 19)
    truth = pool.oracle().kind == "known"
    precision = float((truth & estimate).sum() / max(estimate.sum(), 1))
    assert precision > 0.75


# ------------------------------------------------------------------- scoring ---


def test_rank_normalisation_is_uniform_and_ties_average():
    values = np.array([5.0, 1.0, 1.0, 9.0])
    ranked = scoring.rank_normalise(values)
    assert ranked.max() == 1.0
    assert ranked[1] == ranked[2]                 # ties share an averaged rank
    assert ranked[1] < ranked[0] < ranked[3]      # and order is preserved
    assert scoring.rank_normalise(np.array([3.0, 1.0, 2.0])).tolist() == [1.0, 0.0, 0.5]


def test_novelty_grows_from_an_empty_labelled_pool(pool):
    pool = pool.take(np.arange(500))
    empty = scoring.novelty(pool, np.zeros((0, 256), dtype=np.float32))
    assert np.allclose(empty, 1.0)
    seen = scoring.novelty(pool, pool.embeddings[:100])
    assert seen[:100].max() < 1e-4  # a labelled item is not novel to itself


def test_scoring_never_reads_an_answer(pool):
    """The whole design rests on this: selection is blind to the oracle."""
    pool = pool.take(np.arange(4000))
    blind = proposals.Candidates(
        pool.image_ids, pool.boxes, pool.embeddings, pool.posterior, pool.objectness
    )
    terms = scoring.terms(blind, scoring.ScoreConfig(n_clusters=50))
    assert np.isfinite(terms.combine()).all()


def test_a_multiplicative_score_needs_the_prior():
    terms = scoring.Terms(
        uncertainty=np.zeros(3), diversity=np.zeros(3), rarity=np.zeros(3),
        coherence=np.ones(3),
        partition=clustering.Partition(np.zeros(3, int), np.zeros((1, 2)), np.array([3]), "k", {}),
        config=scoring.ScoreConfig(combination="multiplicative"),
    )
    with pytest.raises(ValueError, match="object-likeness"):
        terms.combine()


# ----------------------------------------------------------------- selection ---


def test_budget_splits_evenly_across_rounds():
    assert selection._split_budget(600, 6) == [100] * 6
    assert selection._split_budget(601, 6) == [101, 100, 100, 100, 100, 100]
    assert sum(selection._split_budget(600, 12)) == 600


def test_every_arm_spends_exactly_the_budget_once(pool):
    pool = pool.take(np.arange(20_000))
    partition = clustering.fit(pool.embeddings, n_clusters=400, seed=0)
    for name, config in selection.ARMS.items():
        picked = selection.select(pool, config, budget=200, rounds=2, partition=partition)
        assert len(picked) == 200, name
        assert np.unique(picked.indices).size == 200, name


def test_the_free_objectness_prior_beats_the_plans_own_equation(pool, partition):
    """Measured, and it is not the result the plan predicted."""
    kind = pool.oracle().kind
    found = {}
    for name in ("random", "plan", "objectness"):
        picked = selection.select(
            pool, selection.ARMS[name], budget=600, rounds=1, partition=partition
        )
        found[name] = int((kind[picked.indices] == "unknown").sum())
    assert found["objectness"] > 5 * found["plan"]
    assert found["plan"] <= found["random"] * 1.5  # the plan's score is not ahead of random


def test_the_consultations_corrections_beat_the_plans_equation_on_tail(pool, partition):
    """The point of the 2026-08-25 meeting, measured."""
    oracle = pool.oracle()
    groups = protocol.load_groups()
    group_of = np.asarray([groups.get(name, "") for name in oracle.class_name])

    def tail_found(arm: str, rounds: int) -> int:
        picked = selection.select(
            pool, selection.ARMS[arm], budget=600, rounds=rounds, partition=partition
        )
        index = picked.indices
        found = oracle.kind[index] == "unknown"
        return int((group_of[index][found] == "tail").sum())

    assert tail_found("prior_consult_batch", 6) > tail_found("objectness", 1)
    assert tail_found("consult_batch", 6) > tail_found("plan", 1)


def test_recomputing_between_rounds_only_helps_a_score_that_can_move(pool, partition):
    """Consultation §7: 6x100 beats 600x1 exactly when D depends on what was picked."""
    kind = pool.oracle().kind

    def found(arm: str, rounds: int) -> int:
        picked = selection.select(
            pool, selection.ARMS[arm], budget=600, rounds=rounds, partition=partition
        )
        return int((kind[picked.indices] == "unknown").sum())

    assert found("consult", 6) > found("consult", 1)     # D tracks the labelled pool
    assert found("objectness", 6) == found("objectness", 1)  # nothing to update


# ----------------------------------------------------------------- labelling ---


def test_box_only_teaches_real_objects_as_background(pool, partition):
    """The half-labelling error, quantified. This is why the policy matters."""
    picked = selection.select(
        pool, selection.ARMS["prior_consult_batch"], budget=600, rounds=6, partition=partition
    )
    rates = {
        policy: labelling.half_labelling_rate(
            labelling.annotate(pool, picked, policy=policy, known_classes=protocol.TASK1), pool
        )
        for policy in labelling.POLICIES
    }
    assert rates["box_only"] > 0.15
    assert rates["full_image"] == 0.0
    assert rates["known_plus_selected"] == 0.0


def test_known_plus_selected_costs_no_more_than_box_only(pool, partition):
    """Known objects are free: the detector already produces them."""
    picked = selection.select(
        pool, selection.ARMS["prior_consult_batch"], budget=600, rounds=6, partition=partition
    )
    annotations = {
        policy: labelling.annotate(pool, picked, policy=policy, known_classes=protocol.TASK1)
        for policy in labelling.POLICIES
    }
    assert annotations["known_plus_selected"].oracle_cost == annotations["box_only"].oracle_cost
    assert annotations["full_image"].oracle_cost > annotations["box_only"].oracle_cost
    # and it hands the trainer far more supervision for that same price
    assert (
        annotations["known_plus_selected"].labelled.size
        > 4 * annotations["box_only"].labelled.size
    )


# -------------------------------------------------------------------- replay ---


def test_allocation_interpolates_between_uniform_and_size_proportional():
    counts = {"head": 10_000, "medium": 1_000, "tail": 100}
    uniform = replay.allocate(counts, total=300, alpha=0.0)
    head_favouring = replay.allocate(counts, total=300, alpha=1.0)
    tail_favouring = replay.allocate(counts, total=300, alpha=-1.0)
    assert uniform["head"] == uniform["tail"]
    assert head_favouring["head"] > head_favouring["tail"]
    assert tail_favouring["tail"] > tail_favouring["head"]


def test_no_class_is_ever_allocated_to_zero():
    """A tail class rounded to zero is a class forgotten outright."""
    counts = {"head": 262_465, "tail": 1_294}
    allocation = replay.allocate(counts, total=50, alpha=1.0, minimum=1)
    assert allocation["tail"] >= 1


def test_allocation_respects_the_total():
    counts = protocol.load_train_counts()
    subset = {name: counts[name] for name in protocol.TASK1}
    for alpha in (-1.0, -0.5, 0.0, 1.0):
        assert sum(replay.allocate(subset, total=400, alpha=alpha).values()) <= 400


def test_greedy_memory_covers_the_allocation_with_fewer_images_than_random():
    per_image = {
        f"img{i:03d}": {"a": (i % 3), "b": (i % 5), "c": 1 if i % 7 == 0 else 0}
        for i in range(300)
    }
    per_image = {k: {c: n for c, n in v.items() if n} for k, v in per_image.items()}
    greedy = replay.build(per_image, ("a", "b", "c"), total=60, alpha=0.0, selector="greedy")
    sampled = replay.build(per_image, ("a", "b", "c"), total=60, alpha=0.0, selector="random")
    assert 0 < len(greedy) <= len(sampled)


def test_herding_tracks_the_class_mean_better_than_sampling():
    """iCaRL's exemplar criterion, reproduced here rather than cited."""
    generator = np.random.default_rng(0)
    points = np.vstack([generator.normal(0, 1, (50, 8)), generator.normal(5, 1, (50, 8))])
    order = replay.herding_order(points)
    assert sorted(order.tolist()) == list(range(100))       # it is a permutation

    target = points.mean(axis=0)
    for size in (5, 10, 20):
        herded = np.linalg.norm(points[order[:size]].mean(axis=0) - target)
        sampled = np.mean([
            np.linalg.norm(points[generator.choice(100, size, replace=False)].mean(0) - target)
            for _ in range(100)
        ])
        assert herded < sampled / 2, size


def test_herding_prefixes_are_nested():
    """A shrinking per-class budget has to be a prefix, not a re-selection."""
    generator = np.random.default_rng(1)
    points = generator.normal(0, 1, (40, 6))
    order = replay.herding_order(points)
    assert order[:5].tolist() == order[:10].tolist()[:5]


def test_memory_carry_forward_versus_reallocation():
    previous = replay.Memory(("a", "b"), {}, 0.0, 10)
    assert replay.carry_forward(previous, ["b", "c"], reallocate=False) == ("a", "b", "c")
    assert replay.carry_forward(previous, ["b", "c"], reallocate=True) == ("b", "c")


# ------------------------------------------------------------------- metrics ---


def test_the_exchange_rate_reproduces_the_measured_gpu_runs():
    """Full supervision pays 0.20 old points per new point; b600 random pays thousands."""
    root = protocol.GROUPS_PATH.parent / "measured"
    full = metrics.from_bridge_metrics(root / "full_t2_supervision_metrics.json")
    random_run = metrics.from_bridge_metrics(root / "random_b600_metrics.json")
    anchor = 73.649
    full_rate = metrics.exchange_rate(
        metrics.task_row(full, task="t2", new_class=None, previous_baseline=anchor)
    )
    random_rate = metrics.exchange_rate(
        metrics.task_row(random_run, task="t2", new_class=None, previous_baseline=anchor)
    )
    assert full_rate == pytest.approx(0.203, abs=0.01)
    assert random_rate > 1000
