"""Method V2 Stage 2 components: construction, accounting, and the frozen gates.

Two properties carry most of the weight.

**No oracle in construction.** D, R and C must be computable from detector
outputs and the labelled reference set alone. If an oracle label could reach
them, every diagnostic downstream would be measuring its own answer.

**Distinct-object accounting.** Under proposal counting an earlier comparison
inflated one arm by 1.76x against a control's 1.02x and reversed its conclusion.
Every ranking table here goes through ``owl.discovery``.
"""

from __future__ import annotations

import numpy as np
import pytest

from owl import method_v2_stage2 as stage2
from owl.proposals import Candidates, Oracle

GROUPS = {"fire hydrant": "tail", "bear": "tail", "sofa": "medium", "chair": "head"}


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def tiny_pool() -> Candidates:
    """Ten proposals: one tail object seen three times, plus knowns and background."""

    kind = np.array([
        "unknown", "unknown", "unknown",   # all three on object 10 (tail)
        "unknown",                          # object 11, tail, other class
        "unknown",                          # object 12, head
        "known", "known",                   # objects 90, 91
        "background", "background", "background",
    ])
    class_name = np.array([
        "fire hydrant", "fire hydrant", "fire hydrant",
        "bear", "chair", "car", "person", "", "", "",
    ])
    object_id = np.array([10, 10, 10, 11, 12, 90, 91, -1, -1, -1], dtype=np.int64)
    n = kind.size
    generator = np.random.default_rng(0)
    posterior = generator.random((n, 81)).astype(np.float32)
    posterior /= posterior.sum(axis=1, keepdims=True)
    return Candidates(
        image_ids=np.array(["a", "a", "a", "b", "b", "c", "c", "c", "d", "d"]),
        boxes=np.tile(np.array([0.5, 0.5, 0.2, 0.2], np.float32), (n, 1)),
        embeddings=_unit(generator.normal(size=(n, 8)).astype(np.float32)),
        posterior=posterior,
        objectness=np.linspace(0.9, 0.1, n, dtype=np.float32),
        _oracle=Oracle(kind=kind, class_name=class_name, object_id=object_id,
                       iou=np.full(n, 0.8, np.float32)),
    )


# ------------------------------------------------- the fixed input population ---


def test_p2_must_reproduce_or_fail_closed():
    """P2 is an input to Stage 2, so drift would silently change every component."""

    kind = np.array(["background"] * stage2.EXPECTED_P2_ROWS, dtype=str)
    kind[: int(stage2.EXPECTED_P2_ROWS * (1 - stage2.EXPECTED_P2_BACKGROUND))] = "unknown"
    mask = np.ones(stage2.EXPECTED_P2_ROWS, dtype=bool)

    report = stage2.verify_p2(mask, kind)
    assert report["rows"] == 15_518

    with pytest.raises(stage2.Stage2Error, match="expected 15,518"):
        stage2.verify_p2(mask[:-1], kind[:-1])


def test_p2_background_share_is_checked_not_just_the_row_count():
    kind = np.array(["unknown"] * stage2.EXPECTED_P2_ROWS, dtype=str)

    with pytest.raises(stage2.Stage2Error, match="background share"):
        stage2.verify_p2(np.ones(stage2.EXPECTED_P2_ROWS, dtype=bool), kind)


def test_the_real_p2_reproduces_15518_rows():
    """The frozen population, from the committed pool, through the repo's own code."""

    from owl import proposals as proposals_module
    from owl import semantic_features as sf
    from tools.audit_decoder_layers import populations
    from tools.diagnose_representation import load

    pool = load()
    payload = np.load(sf.POOL, allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == sf.POOL_SPLIT
    pool["raw_boxes"] = payload["boxes"][keep].astype(np.float32)
    candidates = proposals_module.from_frozen_pool(sf.POOL, split=sf.POOL_SPLIT)

    masks = populations(pool, candidates)
    report = stage2.verify_p2(masks["P2_admissible_nms"], pool["kind"])

    assert report["rows"] == 15_518
    assert report["background_share"] == pytest.approx(0.767, abs=0.001)
    # and no eval row can have reached it
    assert int(keep.sum()) == 80_000


# ------------------------------------------------------------------ component D ---


def test_novelty_is_one_minus_the_best_cosine():
    reference = _unit(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    features = _unit(np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32))

    values = stage2.novelty(features, reference)

    assert values[0] == pytest.approx(0.0, abs=1e-6)         # identical to a reference
    assert values[1] == pytest.approx(1.0 - 2 ** -0.5, abs=1e-6)


def test_novelty_excludes_a_candidate_from_being_its_own_reference():
    """A predicted-known candidate is itself a reference vector; without the
    exclusion its D would be 0 for the trivial reason that it matched itself."""

    features = _unit(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    reference = features.copy()
    mapping = np.array([0, 1], dtype=np.int64)

    without = stage2.novelty(features, reference)
    with_exclusion = stage2.novelty(features, reference, exclude_self=mapping)

    assert np.allclose(without, 0.0, atol=1e-6)
    assert (with_exclusion > 0.5).all()


def test_novelty_refuses_an_empty_or_mismatched_reference():
    features = _unit(np.ones((3, 4), dtype=np.float32))

    with pytest.raises(stage2.Stage2Error, match="reference set is empty"):
        stage2.novelty(features, np.zeros((0, 4), dtype=np.float32))
    with pytest.raises(stage2.Stage2Error, match="against candidate dim"):
        stage2.novelty(features, np.ones((2, 5), dtype=np.float32))


def test_pseudo_reference_construction_reads_no_oracle_label(tiny_pool):
    """REF-A comes from the posterior and NMS only. Handed no labels, by signature."""

    mask = stage2.pseudo_reference_mask(
        tiny_pool.posterior, tiny_pool.objectness, tiny_pool.boxes,
        tiny_pool.image_ids, nms=np.ones(len(tiny_pool), dtype=bool),
    )

    assert mask.dtype == bool
    assert mask.shape == (len(tiny_pool),)
    # the same call with a permuted oracle must give an identical answer
    permuted = Oracle(
        kind=tiny_pool.oracle().kind[::-1],
        class_name=tiny_pool.oracle().class_name[::-1],
        object_id=tiny_pool.oracle().object_id[::-1],
        iou=tiny_pool.oracle().iou[::-1],
    )
    assert permuted is not None                    # the oracle exists but is unused
    again = stage2.pseudo_reference_mask(
        tiny_pool.posterior, tiny_pool.objectness, tiny_pool.boxes,
        tiny_pool.image_ids, nms=np.ones(len(tiny_pool), dtype=bool),
    )
    assert np.array_equal(mask, again)


def test_pseudo_reference_construction_is_narrowed_by_nms(tiny_pool):
    everything = stage2.pseudo_reference_mask(
        tiny_pool.posterior, tiny_pool.objectness, tiny_pool.boxes,
        tiny_pool.image_ids, nms=np.ones(len(tiny_pool), dtype=bool))
    suppressed = np.zeros(len(tiny_pool), dtype=bool)
    suppressed[:4] = True
    narrowed = stage2.pseudo_reference_mask(
        tiny_pool.posterior, tiny_pool.objectness, tiny_pool.boxes,
        tiny_pool.image_ids, nms=suppressed)

    assert narrowed.sum() <= everything.sum()
    assert not (narrowed & ~suppressed).any()


# ------------------------------------------------------------------ component R ---


def test_r1_is_larger_for_an_isolated_point():
    clustered = np.repeat(np.eye(1, 6, dtype=np.float32), 12, axis=0)
    clustered += np.random.default_rng(0).normal(0, 0.001, clustered.shape)
    stray = np.zeros((1, 6), dtype=np.float32)
    stray[0, 3] = 1.0
    features = _unit(np.vstack([clustered, stray]))

    values = stage2.rarity_r1(features, k=3)

    assert values[-1] > values[:-1].max()


def test_r2_is_high_where_candidates_are_dense_and_labels_are_far():
    """R2 = log(labelled k-th distance / candidate k-th distance).

    Both candidate groups get the *same* jitter, so their candidate-side density
    is matched and the only thing that differs is how far the labelled material
    sits. That isolates what R2 claims to measure: coverage deficit, not isolation.
    """

    generator = np.random.default_rng(1)
    uncovered = np.zeros((12, 6), dtype=np.float32)
    uncovered[:, 0] = 1.0
    covered = np.zeros((12, 6), dtype=np.float32)
    covered[:, 4] = 1.0
    jitter = 0.01
    uncovered = _unit(uncovered + generator.normal(0, jitter, uncovered.shape))
    covered = _unit(covered + generator.normal(0, jitter, covered.shape))
    features = np.vstack([uncovered, covered])
    # the labelled set sits on `covered` and nowhere near `uncovered`
    reference = _unit(covered + generator.normal(0, jitter, covered.shape))

    values = stage2.rarity_r2(features, reference, k=3)

    assert values[:12].mean() > values[12:].mean()
    assert values[12:].mean() < 1.0      # covered region scores near zero


def test_r3_scores_clusters_the_labelled_set_does_not_populate():
    generator = np.random.default_rng(3)
    left = _unit(np.tile(np.eye(1, 6, dtype=np.float32), (40, 1))
                 + generator.normal(0, 0.01, (40, 6)))
    right = np.zeros((40, 6), dtype=np.float32)
    right[:, 3] = 1.0
    right = _unit(right + generator.normal(0, 0.01, (40, 6)))
    features = np.vstack([left, right])
    reference = left[:20]                     # labels cover only the left cluster

    values = stage2.rarity_r3(features, reference, n_clusters=2, seed=0)

    assert values[40:].mean() > values[:40].mean()


def test_the_r_definitions_reuse_frozen_repository_parameters():
    """k and K are not chosen for this experiment; drift here is a silent change."""

    from tools.audit_decoder_layers import N_CLUSTERS
    from tools.diagnose_representation import K_NEIGHBOURS

    assert stage2.K_NEIGHBOURS == K_NEIGHBOURS == 10
    assert stage2.N_CLUSTERS == N_CLUSTERS == 120


# ------------------------------------------------------------------ component C ---


def test_consistency_is_the_minimum_of_the_two_view_similarities():
    base = _unit(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    view_a = _unit(np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    view_b = _unit(np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    measured = stage2.consistency(base, view_a, view_b)

    assert measured["consistency"][0] == pytest.approx(2 ** -0.5, abs=1e-6)
    assert measured["consistency"][1] == pytest.approx(2 ** -0.5, abs=1e-6)
    assert measured["consistency_mean"][0] == pytest.approx((1 + 2 ** -0.5) / 2, abs=1e-6)
    assert (measured["consistency"] <= measured["consistency_mean"] + 1e-6).all()


def test_consistency_refuses_views_that_do_not_cover_the_same_rows():
    base = _unit(np.ones((4, 3), dtype=np.float32))

    with pytest.raises(stage2.Stage2Error, match="same rows in the same order"):
        stage2.consistency(base, base[:3], base)


def test_the_frozen_view_margins_are_1_10_and_1_30():
    from owl import semantic_features as sf
    from tools.export_dinov2_consistency_views import VIEW_MARGINS

    assert sf.CROP_MARGIN == 1.20
    assert VIEW_MARGINS == {"view_a": 1.10, "view_b": 1.30}


def test_the_views_use_the_one_frozen_crop_implementation():
    """Only the margin varies, so a view difference cannot be a second code path."""

    from owl import semantic_features as sf
    from tools.export_dinov2_consistency_views import VIEW_MARGINS

    base = sf.square_crop(0.5, 0.5, 0.2, 0.1, 640, 480)
    narrow = sf.square_crop(0.5, 0.5, 0.2, 0.1, 640, 480,
                            margin=VIEW_MARGINS["view_a"])
    wide = sf.square_crop(0.5, 0.5, 0.2, 0.1, 640, 480,
                          margin=VIEW_MARGINS["view_b"])

    for box in (base, narrow, wide):
        assert (box[2] - box[0]) == (box[3] - box[1])          # square
        assert 0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 480
    assert (narrow[2] - narrow[0]) < (base[2] - base[0]) < (wide[2] - wide[0])


# ----------------------------------------------------- ranking and accounting ---


def test_rank_table_counts_distinct_objects_not_proposals(tiny_pool):
    rows = np.arange(len(tiny_pool))
    # rank the three duplicates of object 10 first
    scores = np.array([9.0, 8.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    table = stage2.rank_table(scores, tiny_pool, rows, groups=GROUPS,
                              fractions=(0.3,), name="dupes")

    assert table[0]["proposals"] == 3
    assert table[0]["unknown_objects"] == 1          # one object, three boxes
    assert table[0]["unknown_proposals"] == 3
    assert table[0]["tail_objects"] == 1
    assert table[0]["proposals_per_object"] == pytest.approx(3.0)


def test_rank_table_reports_background_share_and_unique_images(tiny_pool):
    rows = np.arange(len(tiny_pool))
    scores = np.linspace(1.0, 0.0, len(tiny_pool))

    table = stage2.rank_table(scores, tiny_pool, rows, groups=GROUPS,
                              fractions=(1.0,), name="all")

    assert table[0]["proposals"] == len(tiny_pool)
    assert table[0]["background_share"] == pytest.approx(0.3)
    assert table[0]["unique_images"] == 4
    assert table[0]["distinct_oracle_objects"] == 5


def test_rank_table_refuses_a_score_population_mismatch(tiny_pool):
    with pytest.raises(stage2.Stage2Error, match="scores against"):
        stage2.rank_table(np.zeros(3), tiny_pool, np.arange(len(tiny_pool)),
                          groups=GROUPS)


def test_group_summary_separates_the_oracle_strata(tiny_pool):
    oracle = tiny_pool.oracle()
    scores = np.where(oracle.kind == "unknown", 1.0, 0.0)

    rows = stage2.group_summary(scores, oracle.kind,
                                np.array([GROUPS.get(n, "") for n in oracle.class_name]),
                                name="probe")

    by_stratum = {row["stratum"]: row for row in rows}
    assert by_stratum["unknown_all"]["median"] == 1.0
    assert by_stratum["background"]["median"] == 0.0
    assert by_stratum["unknown_tail"]["n"] == 4        # 3 on object 10 + 1 on 11


# ------------------------------------------------------ the frozen GO/NO-GO ---


def _table(fraction, unknown_objects, tail_objects, medium_objects=0,
           medium_classes=0, tail_classes=0, background_share=0.5):
    return [{"fraction": fraction, "unknown_objects": unknown_objects,
             "tail_objects": tail_objects, "medium_objects": medium_objects,
             "medium_classes": medium_classes, "tail_classes": tail_classes,
             "background_share": background_share}]


def test_d_needs_both_the_auc_and_a_ten_percent_gain():
    baseline = _table(0.10, 100, 20)

    both = stage2.evaluate_d(unknown_vs_known_auc=0.70,
                             table=_table(0.10, 111, 20), baseline=baseline)
    assert both["go"] is True

    auc_only = stage2.evaluate_d(unknown_vs_known_auc=0.70,
                                 table=_table(0.10, 105, 21), baseline=baseline)
    assert auc_only["go"] is False
    assert auc_only["checks"]["relative_gain>=10pct_at_some_fraction"] is False

    gain_only = stage2.evaluate_d(unknown_vs_known_auc=0.64,
                                  table=_table(0.10, 200, 40), baseline=baseline)
    assert gain_only["go"] is False
    assert gain_only["checks"]["unknown_vs_known_auc>=0.65"] is False


def test_d_accepts_a_tail_gain_alone():
    """Either endpoint may carry the improvement, per the frozen wording."""

    verdict = stage2.evaluate_d(unknown_vs_known_auc=0.80,
                                table=_table(0.05, 100, 23),
                                baseline=_table(0.05, 100, 20))

    assert verdict["go"] is True


def test_r_needs_monotone_medians_and_coverage_within_the_background_budget():
    baseline = _table(0.10, 100, 20, medium_objects=10, medium_classes=5,
                      tail_classes=5, background_share=0.50)
    good = _table(0.10, 100, 25, medium_objects=15, medium_classes=7,
                  tail_classes=6, background_share=0.55)

    passing = stage2.evaluate_r({"R2": {
        "medians": {"head": 0.1, "medium": 0.2, "tail": 0.3},
        "table": good, "baseline": baseline}})
    assert passing["go"] is True
    assert passing["passing_definitions"] == ["R2"]

    not_monotone = stage2.evaluate_r({"R2": {
        "medians": {"head": 0.5, "medium": 0.2, "tail": 0.3},
        "table": good, "baseline": baseline}})
    assert not_monotone["go"] is False

    too_much_background = stage2.evaluate_r({"R2": {
        "medians": {"head": 0.1, "medium": 0.2, "tail": 0.3},
        "table": _table(0.10, 100, 25, medium_objects=15, medium_classes=7,
                        tail_classes=6, background_share=0.65),
        "baseline": baseline}})
    assert too_much_background["go"] is False


def test_r_passes_if_any_one_predeclared_definition_passes():
    baseline = _table(0.10, 100, 20, medium_objects=10, medium_classes=5,
                      tail_classes=5, background_share=0.50)
    good = _table(0.10, 100, 25, medium_objects=15, medium_classes=7,
                  tail_classes=6, background_share=0.52)

    verdict = stage2.evaluate_r({
        "R1": {"medians": {"head": 0.5, "medium": 0.2, "tail": 0.1},
               "table": good, "baseline": baseline},
        "R3": {"medians": {"head": 0.1, "medium": 0.2, "tail": 0.3},
               "table": good, "baseline": baseline},
    })

    assert verdict["go"] is True
    assert verdict["passing_definitions"] == ["R3"]


def test_c_passes_on_the_auc_or_on_the_filter_gain():
    baseline = _table(0.10, 100, 20)

    on_auc = stage2.evaluate_c(unknown_vs_background_auc=0.61)
    assert on_auc["go"] is True

    on_filter = stage2.evaluate_c(unknown_vs_background_auc=0.50,
                                  table=_table(0.10, 115, 20), baseline=baseline)
    assert on_filter["go"] is True

    losing_the_tail = stage2.evaluate_c(unknown_vs_background_auc=0.50,
                                        table=_table(0.10, 130, 19),
                                        baseline=baseline)
    assert losing_the_tail["go"] is False


def test_the_ladder_stops_at_the_first_failure():
    assert stage2.allowed_ladder(False, True, True) == "U"
    assert stage2.allowed_ladder(True, False, True) == "U+D"
    assert stage2.allowed_ladder(True, True, False) == "U+D+R"
    assert stage2.allowed_ladder(True, True, True) == "U+D+R*C"


def test_the_verdict_prints_every_component_regardless_of_the_ladder():
    verdict = stage2.Stage2Verdict(
        d={"go": True}, r={"go": False}, c={"go": True})

    assert verdict.lines() == [
        "D_GO", "R_NO_GO", "C_GO", "METHOD_V2_ALLOWED_LADDER = U+D",
    ]


def test_the_frozen_thresholds_are_the_protocol_values():
    assert stage2.D_GO_UNKNOWN_VS_KNOWN_AUC == 0.65
    assert stage2.D_GO_RELATIVE_IMPROVEMENT == 0.10
    assert stage2.R_GO_RELATIVE_IMPROVEMENT == 0.10
    assert stage2.R_GO_MAX_BACKGROUND_INCREASE == 0.10
    assert stage2.C_GO_UNKNOWN_VS_BACKGROUND_AUC == 0.60
    assert stage2.C_GO_RELATIVE_IMPROVEMENT == 0.10
    assert stage2.GO_FRACTIONS == (0.05, 0.10, 0.20)
    assert stage2.REPORT_FRACTIONS == (0.01, 0.05, 0.10, 0.20, 0.30)


# ------------------------------- REF-T1: the corrected primary reference ---


def test_ref_t1_alias_map_covers_the_two_coco_spellings():
    """Without it, aeroplane and motorbike would be absent from the reference.

    Parsing the archive without the map yielded 407,383 objects against the
    expected 421,243, with those two classes at exactly zero -- so D would have
    reported every aeroplane and motorbike in the pool as novel.
    """

    from owl.protocol import TASK1
    from owl.reference_t1 import COCO_TO_VOC

    assert COCO_TO_VOC == {"airplane": "aeroplane", "motorcycle": "motorbike"}
    for voc in COCO_TO_VOC.values():
        assert voc in TASK1


def test_ref_t1_class_totals_match_the_committed_count_file():
    from owl.protocol import TASK1
    from owl.reference_t1 import EXPECTED_T1_OBJECTS, class_totals

    totals = class_totals()

    assert set(totals) == set(TASK1)
    assert sum(totals.values()) == EXPECTED_T1_OBJECTS == 421_243
    assert min(totals.values()) == 1_294          # bear
    assert max(totals.values()) == 262_465        # person


def test_ref_t1_enumeration_fails_closed_on_a_convention_mismatch(monkeypatch):
    """A wrong total must raise, naming the empty classes, not proceed quietly."""

    from owl import reference_t1 as ref

    monkeypatch.setattr(ref, "COCO_TO_VOC", {})     # simulate the missing alias

    with pytest.raises(ref.ExportError) as error:
        ref.enumerate_objects()
    message = str(error.value)
    assert "expected 421,243" in message
    assert "aeroplane" in message and "motorbike" in message


def test_ref_t1_selection_is_exactly_balanced_and_deterministic():
    from owl import reference_t1 as ref

    grouped = ref.enumerate_objects()
    first = ref.select_balanced(grouped, per_class_cap=100)
    second = ref.select_balanced(grouped, per_class_cap=100)

    assert np.array_equal(first.keys, second.keys)
    summary = first.summary()
    assert summary["objects"] == 19 * 100
    assert summary["classes"] == 19
    assert summary["balanced"] is True
    assert summary["min_per_class"] == summary["max_per_class"] == 100
    assert np.unique(first.keys).size == first.keys.size


def test_ref_t1_selections_are_nested_across_caps():
    """One export at the largest cap gives the smaller caps as free subsets."""

    from owl import reference_t1 as ref

    grouped = ref.enumerate_objects()
    small = set(ref.select_balanced(grouped, per_class_cap=50).keys.tolist())
    large = set(ref.select_balanced(grouped, per_class_cap=100).keys.tolist())

    assert small <= large
    assert len(small) * 2 == len(large)


def test_ref_t1_holds_only_task_1_classes():
    """No T2 or later annotation may reach the reference."""

    from owl import reference_t1 as ref
    from owl.protocol import CLASS_ORDER, TASK1

    selection = ref.select_balanced(ref.enumerate_objects(), per_class_cap=50)

    present = set(selection.class_name.tolist())
    assert present == set(TASK1)
    future = set(CLASS_ORDER[len(TASK1):])
    assert not (present & future)


def test_ref_t1_images_do_not_touch_the_eval_split():
    from owl import reference_t1 as ref
    from owl import semantic_features as sf

    selection = ref.select_balanced(ref.enumerate_objects(), per_class_cap=100)
    payload = np.load(sf.POOL, allow_pickle=True)
    splits = np.asarray(payload["split"], dtype=str)
    ids = np.asarray(payload["image_ids"], dtype=str)

    assert not (set(selection.images) & set(ids[splits == "eval"].tolist()))


def test_ref_t1_rejects_a_nonsense_cap():
    from owl import reference_t1 as ref

    with pytest.raises(ref.ExportError, match="at least 1"):
        ref.select_balanced({"bear": [("a", 0, "bear", (0.5, 0.5, 0.1, 0.1))]},
                            per_class_cap=0)


# --------------------------------------------- the frozen C ranking, A * C ---


def test_score_c_is_exactly_a_times_c():
    admissibility = np.array([2.0, 4.0, 1.0])
    consistency = np.array([0.5, 0.25, 1.0])

    assert np.allclose(stage2.score_c(admissibility, consistency),
                       [1.0, 1.0, 1.0])


def test_score_c_refuses_a_shape_mismatch():
    with pytest.raises(stage2.Stage2Error, match="row-wise on the same P2 rows"):
        stage2.score_c(np.ones(4), np.ones(3))


# ------------------------------- the corrected R_GO and zero denominators ---


def test_r_requires_the_object_endpoint_and_class_gain_cannot_rescue_it():
    """Class coverage is descriptive: a class-only gain must not pass R."""

    baseline = _table(0.10, 100, 20, medium_objects=10, medium_classes=5,
                      tail_classes=5, background_share=0.50)
    # medium+tail objects flat (30 -> 30) but classes up 10 -> 14
    class_only = _table(0.10, 100, 20, medium_objects=10, medium_classes=8,
                        tail_classes=6, background_share=0.50)

    verdict = stage2.evaluate_r({"R2": {
        "medians": {"head": 0.1, "medium": 0.2, "tail": 0.3},
        "table": class_only, "baseline": baseline}})

    assert verdict["go"] is False
    entry = verdict["definitions"]["R2"]
    assert entry["monotone_head_medium_tail"] is True
    assert entry["object_gain_and_background_at_same_fraction"] is False
    assert entry["fractions"][0.10]["medium_tail_class_gain"] > 0.10   # reported


def test_r_needs_the_gain_and_the_background_budget_at_the_same_fraction():
    """Gain at 5% paid for by background at 20% must not pass."""

    baseline = [
        {"fraction": 0.05, "unknown_objects": 50, "tail_objects": 10,
         "medium_objects": 10, "medium_classes": 5, "tail_classes": 5,
         "background_share": 0.50},
        {"fraction": 0.20, "unknown_objects": 100, "tail_objects": 20,
         "medium_objects": 10, "medium_classes": 5, "tail_classes": 5,
         "background_share": 0.50},
    ]
    split = [
        # gain at 5%, but background blows the budget there
        {"fraction": 0.05, "unknown_objects": 50, "tail_objects": 10,
         "medium_objects": 20, "medium_classes": 5, "tail_classes": 5,
         "background_share": 0.75},
        # background fine at 20%, but no gain there
        {"fraction": 0.20, "unknown_objects": 100, "tail_objects": 20,
         "medium_objects": 10, "medium_classes": 5, "tail_classes": 5,
         "background_share": 0.50},
    ]

    verdict = stage2.evaluate_r({"R2": {
        "medians": {"head": 0.1, "medium": 0.2, "tail": 0.3},
        "table": split, "baseline": baseline}})

    assert verdict["go"] is False
    assert verdict["definitions"]["R2"]["satisfying_fraction"] is None


def test_a_zero_baseline_cannot_satisfy_a_relative_gain():
    """An infinite improvement over nothing is an artefact, not a pass."""

    baseline = _table(0.05, 0, 0)

    verdict = stage2.evaluate_d(unknown_vs_known_auc=0.90,
                                table=_table(0.05, 50, 10), baseline=baseline)

    assert verdict["go"] is False
    assert np.isnan(verdict["gains"][0.05]["unknown_objects"])
    assert verdict["checks"]["relative_gain>=10pct_at_some_fraction"] is False


def test_a_nan_auc_cannot_pass_a_threshold():
    verdict = stage2.evaluate_c(unknown_vs_background_auc=float("nan"))

    assert verdict["go"] is False
    assert verdict["checks"]["unknown_vs_background_auc>=0.60"] is False
