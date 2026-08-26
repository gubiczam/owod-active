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


def test_the_increment_is_what_prob_is_told_not_the_running_total():
    """PROB adds the two flags: seen = prev + current. Getting this wrong makes
    every reported mAP wrong, silently, so it is pinned here."""
    chain = protocol.build_chain(10)
    for task in chain[1:]:
        assert task.n_new == 1
        assert task.n_prev + task.n_new == task.n_current
    # the failure mode this guards against
    last = chain[-1]
    assert last.n_prev + last.n_current != last.n_current, "sanity: they differ"
    assert last.n_prev + last.n_new == 28


def test_the_bridge_rejects_impossible_class_counts():
    from owl import bridge as bridge_module

    instrument = bridge_module.Bridge(prob_root="/tmp/x", data_root="/tmp/y", dry_run=True)
    with pytest.raises(bridge_module.BridgeError, match="seen = prev \\+ current"):
        instrument._call("train", [], n_prev=0, n_current=0, label="bad")


def test_chain_refuses_to_run_off_the_end():
    with pytest.raises(protocol.ProtocolError):
        protocol.build_chain(100)


def test_every_owl_data_source_uses_the_benchmark_spelling():
    """Six classes are spelled two ways and it is a real trap.

    The raw VOC annotation files say ``dining table``, ``potted plant``,
    ``couch``, ``tv``, ``airplane``, ``motorcycle``. The benchmark's class order
    says ``diningtable``, ``pottedplant``, ``sofa``, ``tvmonitor``,
    ``aeroplane``, ``motorbike``. Two of the ten tasks declare classes on that
    list, so if one of our own tables drifted to the COCO spelling those tasks
    would silently count zero instances of their own target.

    PROB's loader does the mapping itself
    (``datasets/torchvision_datasets/open_world.py``, VOC_CLASS_NAMES_COCOFIED),
    so the archives are correct as they come off disk. This pins the other side:
    everything inside owl speaks the benchmark's spelling.
    """
    import json

    from owl import evaluation_subset

    cocofied = ["airplane", "dining table", "motorcycle", "potted plant", "couch", "tv"]
    benchmark = ["aeroplane", "diningtable", "motorbike", "pottedplant", "sofa", "tvmonitor"]

    root = protocol.GROUPS_PATH.parent
    index = json.loads((root / "per_image_class_counts.json").read_text(encoding="utf-8"))
    sources = {
        "candidate index": {name for counts in index.values() for name in counts},
        "class groups": set(protocol.load_groups()),
        "class order": set(protocol.CLASS_ORDER),
    }
    for label, names in sources.items():
        assert not names & set(cocofied), f"{label} drifted to the COCO spelling"
        assert set(benchmark) <= names, f"{label} is missing benchmark-spelled classes"

    for coco, voc in zip(cocofied, benchmark):
        assert evaluation_subset.canonical_class_name(coco) == voc


def test_the_declared_classes_are_reachable_in_the_candidate_pool():
    """A task whose class appears in no candidate image cannot be learned."""
    import json

    index = json.loads(
        (protocol.GROUPS_PATH.parent / "per_image_class_counts.json").read_text(encoding="utf-8")
    )
    for task in protocol.build_chain(10)[1:]:
        images = sum(1 for counts in index.values() if task.new_class in counts)
        assert images >= 300, f"{task.new_class} is in only {images} candidate images"


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


def test_per_class_ap50_is_read_out_of_the_vector_the_bridge_actually_writes():
    """The bridge writes no per_class_AP50 key. It writes coco_eval_bbox.

    Without per-class numbers the head/medium/tail decomposition — the research
    plan's distinguishing form of evaluation — cannot be computed at all. It was
    available in every metrics file already, in a vector shaped
    ``[mAP, mAP, <80 classes>, unknown]``.
    """

    import json

    root = protocol.GROUPS_PATH.parent / "measured"
    payload = json.loads(
        (root / "full_t2_supervision_metrics.json").read_text(encoding="utf-8")
    )
    assert "per_class_AP50" not in payload
    per_class = metrics.per_class_ap50(payload)
    assert len(per_class) == 81                      # 80 classes plus unknown
    assert per_class["unknown"] == pytest.approx(payload["unknown_AP50"], abs=1e-4)


@pytest.mark.parametrize(
    ("filename", "n_prev", "n_current"),
    [
        ("full_t2_supervision_metrics.json", 19, 21),
        ("random_b600_metrics.json", 19, 21),
        ("objectness_prior_b600_metrics.json", 19, 21),
    ],
)
def test_the_per_class_vector_reproduces_the_reported_aggregates(filename, n_prev, n_current):
    """The alignment check. A misaligned table would attribute one class's score
    to another and the head/medium/tail split would be quietly wrong."""

    import json
    from statistics import mean

    root = protocol.GROUPS_PATH.parent / "measured"
    payload = json.loads((root / filename).read_text(encoding="utf-8"))
    per_class = metrics.per_class_ap50(payload)
    scores = [per_class[name] for name in protocol.CLASS_ORDER]

    assert mean(scores[:n_prev]) == pytest.approx(payload["previous_known_AP50"], abs=1e-3)
    assert mean(scores[n_prev : n_prev + n_current]) == pytest.approx(
        payload["current_known_AP50"], abs=1e-3
    )
    assert mean(scores[: n_prev + n_current]) == pytest.approx(payload["known_AP50"], abs=1e-3)


def test_a_wrong_length_vector_yields_nothing_rather_than_a_guess():
    assert metrics.per_class_ap50({"coco_eval_bbox": [1.0, 2.0, 3.0]}) == {}
    assert metrics.per_class_ap50({}) == {}


def test_the_frequency_split_of_a_real_run_is_computable():
    """The plan's headline evaluation, on a measured checkpoint."""

    root = protocol.GROUPS_PATH.parent / "measured"
    evaluation = metrics.from_bridge_metrics(root / "full_t2_supervision_metrics.json")
    membership = metrics.group_membership(protocol.CLASS_ORDER[:40], protocol.load_groups())
    grouped = metrics.grouped_map(evaluation, membership)
    assert set(grouped) == {"head", "medium", "tail"}
    assert all(value is not None and value > 0 for value in grouped.values())


# ------------------------------- the plan's headline endpoint ---------------


def _artifact(objects, detections, unknown="unknown"):
    return {
        "schema": "daowod_detections_v1",
        "unknown_class_name": unknown,
        "ground_truth": objects,
        "detections": detections,
    }


def test_unknown_recall_splits_by_the_true_class_of_the_object():
    """The aggregate U-Recall cannot answer the plan's question; this can."""

    objects = [
        {"image_id": "a", "class_name": "fire hydrant", "box": [0, 0, 10, 10]},
        {"image_id": "a", "class_name": "chair", "box": [50, 50, 60, 60]},
        {"image_id": "a", "class_name": "person", "box": [20, 20, 30, 30]},
    ]
    detections = [
        {"image_id": "a", "class_name": "unknown", "score": 0.9, "box": [0, 0, 10, 10]},
    ]
    result = metrics.unknown_recall_by_group(
        _artifact(objects, detections),
        known_classes=["person"],
        groups={"fire hydrant": "tail", "chair": "head", "person": "head"},
    )
    assert result["tail"] == {"recalled": 1, "objects": 1, "recall": 100.0}
    assert result["head"] == {"recalled": 0, "objects": 1, "recall": 0.0}
    assert result["all"]["objects"] == 2, "a known object is not an unknown to find"


def test_two_detections_on_one_object_recall_it_once():
    objects = [{"image_id": "a", "class_name": "fire hydrant", "box": [0, 0, 10, 10]}]
    detections = [
        {"image_id": "a", "class_name": "unknown", "score": 0.9, "box": [0, 0, 10, 10]},
        {"image_id": "a", "class_name": "unknown", "score": 0.8, "box": [1, 1, 9, 9]},
    ]
    result = metrics.unknown_recall_by_group(
        _artifact(objects, detections), known_classes=[],
        groups={"fire hydrant": "tail"},
    )
    assert result["tail"]["recalled"] == 1


def test_a_detection_that_does_not_overlap_recalls_nothing():
    objects = [{"image_id": "a", "class_name": "fire hydrant", "box": [0, 0, 10, 10]}]
    detections = [
        {"image_id": "a", "class_name": "unknown", "score": 0.9, "box": [80, 80, 90, 90]},
    ]
    result = metrics.unknown_recall_by_group(
        _artifact(objects, detections), known_classes=[],
        groups={"fire hydrant": "tail"},
    )
    assert result["tail"]["recall"] == 0.0


def test_a_known_class_detection_does_not_count_as_discovery():
    """Only detections of the unknown class recall an unknown object."""

    objects = [{"image_id": "a", "class_name": "fire hydrant", "box": [0, 0, 10, 10]}]
    detections = [
        {"image_id": "a", "class_name": "car", "score": 0.99, "box": [0, 0, 10, 10]},
    ]
    result = metrics.unknown_recall_by_group(
        _artifact(objects, detections), known_classes=[],
        groups={"fire hydrant": "tail"},
    )
    assert result["tail"]["recall"] == 0.0


def test_a_foreign_schema_is_refused_rather_than_misread():
    with pytest.raises(metrics.MetricsError, match="daowod_detections_v1"):
        metrics.unknown_recall_by_group(
            {"schema": "something_else"}, known_classes=[], groups={}
        )


def test_the_groups_of_the_real_benchmark_partition_the_unknowns():
    """Every class the chain can meet must land in a named frequency group."""

    groups = protocol.load_groups()
    chain = protocol.build_chain(10)
    for name in protocol.unknown_classes(chain[1]):
        assert groups.get(name) in metrics.GROUPS, f"{name} has no frequency group"
