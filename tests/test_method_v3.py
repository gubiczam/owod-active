"""Method V3: the population, the four arms, the frozen criterion, the run.

Two things these tests exist to prevent.

1. **A criterion that moves.** The verdict must be a pure function of the twelve
   result rows and of nothing else, and the three clauses must be exactly the
   ones written in ``docs/method_v3_protocol_2026-09-02.md``. Several tests pin
   each clause independently, including the case where only the guard fails.
2. **A stubbed run reported as real.** ``--dry-run`` writes real-looking files.
   ``dry_run`` is part of the trajectory fingerprint, so a real run must refuse
   to resume a stubbed trajectory; that refusal is tested.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from owl import method_v3, protocol, replay, scoring, selection

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
REPLAY_INDEX = ROOT / "data" / "reference" / "t1_replay_class_counts.json"
POOL_PATH = ROOT / "data" / "pool" / "sowodb_t1_frozen_pool.npz"


@pytest.fixture(scope="session")
def candidate_index():
    return json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def v3_population(candidate_index):
    return method_v3.population(POOL_PATH, candidate_index)


@pytest.fixture(scope="session")
def synthetic_consistency(v3_population):
    """A deterministic stand-in for the frozen C, in the real cosine range.

    The real values live in a Drive artefact this suite has no access to. What
    the tests need from C is only that it is a per-row vector of the right length
    that changes the product ranking, which this provides without pretending to
    be DINOv2.
    """

    generator = np.random.default_rng(11)
    return generator.uniform(0.55, 0.99, size=len(v3_population)).astype(np.float32)


# ------------------------------------------------------------- population ---


def test_population_reproduces_the_frozen_counts(v3_population):
    assert len(v3_population) == method_v3.EXPECTED_POPULATION_ROWS == 8_010
    assert v3_population.image_ids.size == method_v3.EXPECTED_POPULATION_IMAGES == 839


def test_population_is_inside_p2(v3_population):
    assert int(v3_population.p2_mask.sum()) == 15_518
    assert not (v3_population.keep_mask & ~v3_population.p2_mask).any()
    assert int(v3_population.within_p2.sum()) == len(v3_population)


def test_population_only_keeps_images_with_a_committed_annotation(
    v3_population, candidate_index
):
    assert {str(v) for v in v3_population.image_ids} <= set(candidate_index)


def test_population_fails_closed_when_the_annotation_index_shrinks(candidate_index):
    fewer = dict(list(candidate_index.items())[:100])
    with pytest.raises(method_v3.MethodV3Error, match="did not reproduce"):
        method_v3.population(POOL_PATH, fewer)


def test_population_carries_a_stable_key_fingerprint(v3_population, candidate_index):
    again = method_v3.population(POOL_PATH, candidate_index)
    assert again.provenance["keys_sha256"] == v3_population.provenance["keys_sha256"]


# ------------------------------------------------------------------- arms ---


def test_the_four_arms_are_exactly_the_frozen_ones():
    assert method_v3.ARMS == ("random", "A", "U", "A*C")
    assert method_v3.SEEDS == (0, 1, 2)
    assert (method_v3.BUDGET, method_v3.ROUNDS) == (600, 6)
    assert method_v3.BUDGET_MARKS == (100, 200, 300, 400, 500, 600)
    assert len(method_v3.trajectories()) == 12


def test_arm_a_is_the_frozen_admissibility(v3_population):
    score = method_v3.arm_score("A", v3_population.candidates)
    assert np.allclose(score, scoring.admissibility(v3_population.candidates))


def test_arm_u_is_the_frozen_normalised_entropy(v3_population):
    score = method_v3.arm_score("U", v3_population.candidates)
    assert np.allclose(
        score, scoring.uncertainty(v3_population.candidates, "entropy")
    )
    assert 0.0 <= score.min() and score.max() <= 1.0


def test_arm_a_times_c_is_the_literal_product(v3_population, synthetic_consistency):
    score = method_v3.arm_score(
        "A*C", v3_population.candidates, consistency=synthetic_consistency
    )
    expected = (
        scoring.admissibility(v3_population.candidates) * synthetic_consistency
    )
    assert np.allclose(score, expected)


def test_arm_a_times_c_has_no_exponent_and_no_threshold(
    v3_population, synthetic_consistency
):
    """A power, a threshold or a rescaling of C would all break this identity."""

    a = scoring.admissibility(v3_population.candidates)
    score = method_v3.arm_score(
        "A*C", v3_population.candidates, consistency=synthetic_consistency
    )
    nonzero = a > 0
    assert np.allclose(score[nonzero] / a[nonzero], synthetic_consistency[nonzero])


def test_random_arm_has_no_ranking(v3_population):
    assert method_v3.arm_score("random", v3_population.candidates) is None


def test_a_times_c_refuses_to_run_without_consistency(v3_population):
    with pytest.raises(method_v3.MethodV3Error, match="frozen consistency"):
        method_v3.arm_score("A*C", v3_population.candidates)


def test_unknown_arm_is_refused(v3_population):
    with pytest.raises(method_v3.MethodV3Error, match="unknown arm"):
        method_v3.arm_score("A+C", v3_population.candidates)


def test_every_arm_config_switches_off_every_semantic_weight():
    for arm in method_v3.ARMS:
        config = method_v3.arm_config(arm, 0)
        assert config.lambda_diversity == 0.0
        assert config.gamma_rarity == 0.0
        assert config.mu_batch == 0.0
        assert config.coherence_method == "off"
        assert config.random is (arm == "random")


# -------------------------------------------------------------- selection ---


@pytest.mark.parametrize("arm", method_v3.ARMS)
def test_every_arm_spends_exactly_the_budget(arm, v3_population, synthetic_consistency):
    picked = method_v3.select_for_arm(
        v3_population.candidates, arm, 0, consistency=synthetic_consistency
    )
    assert len(picked) == method_v3.BUDGET
    assert np.unique(picked.indices).size == method_v3.BUDGET
    assert sorted(set(picked.round_of.tolist())) == list(range(method_v3.ROUNDS))
    assert [int((picked.round_of == r).sum()) for r in range(method_v3.ROUNDS)] == [100] * 6


def test_rounds_are_nested_prefixes_of_one_ranking(v3_population):
    """The property the per-budget curve depends on. Protocol §2."""

    picked = method_v3.select_for_arm(v3_population.candidates, "A", 0)
    ranking = np.argsort(
        -scoring.admissibility(v3_population.candidates), kind="mergesort"
    )
    assert picked.indices.tolist() == ranking[: method_v3.BUDGET].tolist()


def test_static_arms_select_identically_at_every_seed(
    v3_population, synthetic_consistency
):
    """Stated in advance in the protocol, and printed by the summariser."""

    for arm in ("A", "U", "A*C"):
        picks = [
            method_v3.select_for_arm(
                v3_population.candidates, arm, seed, consistency=synthetic_consistency
            ).indices.tolist()
            for seed in method_v3.SEEDS
        ]
        assert picks[0] == picks[1] == picks[2]


def test_random_arm_does_vary_with_the_seed(v3_population):
    picks = [
        method_v3.select_for_arm(v3_population.candidates, "random", seed).indices.tolist()
        for seed in method_v3.SEEDS
    ]
    assert picks[0] != picks[1] and picks[1] != picks[2]


def test_arms_choose_materially_different_regions(v3_population, synthetic_consistency):
    chosen = {
        arm: set(method_v3.select_for_arm(
            v3_population.candidates, arm, 0, consistency=synthetic_consistency
        ).indices.tolist())
        for arm in method_v3.ARMS
    }
    for left in method_v3.ARMS:
        for right in method_v3.ARMS:
            if left < right:
                assert chosen[left] != chosen[right], (left, right)


def test_selection_never_reads_the_oracle(v3_population, monkeypatch):
    """Acquisition may not touch a label. Enforced, not assumed."""

    def refuse(self):
        raise AssertionError("selection read the oracle")

    monkeypatch.setattr(type(v3_population.candidates), "oracle", refuse)
    for arm in method_v3.ARMS:
        method_v3.select_for_arm(
            v3_population.candidates, arm, 0,
            consistency=np.full(len(v3_population), 0.8, dtype=np.float32),
        )


# ------------------------------- the precomputed hook in owl.selection ---


def test_precomputed_refuses_a_config_that_still_weights_a_term(v3_population):
    config = scoring.ScoreConfig(name="x", lambda_diversity=0.2, gamma_rarity=0.0,
                                 coherence_method="off")
    with pytest.raises(ValueError, match="still weights"):
        selection.select(
            v3_population.candidates, config, budget=10,
            precomputed=np.zeros(len(v3_population)),
        )


def test_precomputed_refuses_the_random_arm(v3_population):
    with pytest.raises(ValueError, match="no ranking"):
        selection.select(
            v3_population.candidates, method_v3.arm_config("random", 0),
            budget=10, precomputed=np.zeros(len(v3_population)),
        )


def test_precomputed_refuses_a_wrong_length(v3_population):
    with pytest.raises(ValueError, match="expected"):
        selection.select(
            v3_population.candidates, method_v3.arm_config("A", 0),
            budget=10, precomputed=np.zeros(7),
        )


def test_precomputed_is_opt_in_and_leaves_the_v1_arms_alone(pool, partition):
    """The existing ladder must be bit-identical: V1 results stay reproducible."""

    config = selection.ARMS["prior_consult_batch"]
    before = selection.select(
        pool, config, budget=50, rounds=2, n_known=19, partition=partition
    )
    after = selection.select(
        pool, config, budget=50, rounds=2, n_known=19, partition=partition,
        precomputed=None,
    )
    assert before.indices.tolist() == after.indices.tolist()


# -------------------------------------------------------------- criterion ---


def _row(arm, seed, primary, guard=50.0):
    return {
        "arm": arm, "seed": seed, "budget": method_v3.BUDGET,
        "mAP50_medium_tail": primary, "known_mAP50": guard,
    }


def _design(treatment, control, guard_treatment=50.0, guard_control=50.0):
    return [
        *[_row("A", s, control[s], guard_control) for s in method_v3.SEEDS],
        *[_row("A*C", s, treatment[s], guard_treatment) for s in method_v3.SEEDS],
    ]


def test_criterion_matches_the_frozen_statement():
    criterion = method_v3.CRITERION
    assert criterion.primary_metric == "mAP50_medium_tail"
    assert criterion.guard_metric == "known_mAP50"
    assert (criterion.treatment, criterion.control) == ("A*C", "A")
    assert criterion.budget == 600
    assert criterion.minimum_improving_seeds == 2
    assert criterion.guard_tolerance == 1.0


def test_the_protocol_declares_the_criterion_as_the_same_values():
    """Structured equality, which is the whole point of the block.

    The previous version of this check searched the document for the rendered
    phrase ``f"{guard_tolerance:g} AP50 point"``. That renders ``1.0`` as ``1``
    while the document says ``1.0``, so a correct and frozen criterion was
    reported as a documentation mismatch and stopped an overnight run before it
    trained anything. Values, not wording.
    """

    report = method_v3.check_protocol_criterion()
    assert report["criterion"] == asdict(method_v3.CRITERION)
    assert Path(report["protocol"]) == method_v3.PROTOCOL_PATH


def test_the_protocol_block_holds_exactly_the_frozen_numbers():
    declared = method_v3.parse_criterion_block(
        method_v3.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert declared == {
        "primary_metric": "mAP50_medium_tail",
        "guard_metric": "known_mAP50",
        "treatment": "A*C",
        "control": "A",
        "budget": 600,
        "minimum_improving_seeds": 2,
        "guard_tolerance": 1.0,
    }


def test_a_difference_in_number_formatting_does_not_fail_the_check(tmp_path):
    """``1`` and ``1.0`` are the same number and must both be accepted."""

    text = method_v3.PROTOCOL_PATH.read_text(encoding="utf-8")
    rewritten = text.replace('"guard_tolerance": 1.0', '"guard_tolerance": 1')
    assert rewritten != text
    target = tmp_path / "protocol.md"
    target.write_text(rewritten, encoding="utf-8")
    report = method_v3.check_protocol_criterion(target)
    assert report["criterion"]["guard_tolerance"] == 1.0


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ('"guard_tolerance": 1.0', '"guard_tolerance": 2.0'),
        ('"minimum_improving_seeds": 2', '"minimum_improving_seeds": 1'),
        ('"budget": 600', '"budget": 300'),
        ('"primary_metric": "mAP50_medium_tail"', '"primary_metric": "known_mAP50"'),
        ('"treatment": "A*C"', '"treatment": "U"'),
        ('"control": "A"', '"control": "random"'),
        ('"guard_metric": "known_mAP50"', '"guard_metric": "mAP50_head"'),
    ],
)
def test_a_difference_in_value_does_fail_the_check(tmp_path, before, after):
    text = method_v3.PROTOCOL_PATH.read_text(encoding="utf-8")
    assert text.count(before) >= 1
    target = tmp_path / "protocol.md"
    target.write_text(text.replace(before, after), encoding="utf-8")
    with pytest.raises(method_v3.MethodV3Error, match="different criteria"):
        method_v3.check_protocol_criterion(target)


def test_a_dropped_or_added_criterion_field_fails_the_check(tmp_path):
    text = method_v3.PROTOCOL_PATH.read_text(encoding="utf-8")
    # A field with a trailing comma, so what is left is still valid JSON and the
    # missing-field branch is what gets exercised rather than the parser.
    dropped = tmp_path / "dropped.md"
    dropped.write_text(
        text.replace('  "minimum_improving_seeds": 2,\n', ""), encoding="utf-8")
    with pytest.raises(method_v3.MethodV3Error, match="not declared in the protocol"):
        method_v3.check_protocol_criterion(dropped)

    added = tmp_path / "added.md"
    added.write_text(
        text.replace('"guard_tolerance": 1.0',
                     '"guard_tolerance": 1.0,\n  "p_value": 0.05'),
        encoding="utf-8")
    with pytest.raises(method_v3.MethodV3Error,
                       match="declared but not a criterion field"):
        method_v3.check_protocol_criterion(added)


def test_the_known_retention_clause_cannot_be_dropped_from_the_protocol(tmp_path):
    """Removing the guard is a change of criterion, not a wording change."""

    text = method_v3.PROTOCOL_PATH.read_text(encoding="utf-8")
    target = tmp_path / "protocol.md"
    target.write_text(
        text.replace('  "guard_metric": "known_mAP50",\n', ""), encoding="utf-8")
    with pytest.raises(method_v3.MethodV3Error, match="guard_metric"):
        method_v3.check_protocol_criterion(target)


def test_a_missing_or_duplicated_block_fails_closed(tmp_path):
    text = method_v3.PROTOCOL_PATH.read_text(encoding="utf-8")
    block = method_v3.CRITERION_BLOCK.search(text)
    assert block

    absent = tmp_path / "absent.md"
    absent.write_text(text.replace(block.group(0), ""), encoding="utf-8")
    with pytest.raises(method_v3.MethodV3Error, match="holds 0"):
        method_v3.check_protocol_criterion(absent)

    twice = tmp_path / "twice.md"
    twice.write_text(text + "\n" + block.group(0) + "\n", encoding="utf-8")
    with pytest.raises(method_v3.MethodV3Error, match="holds 2"):
        method_v3.check_protocol_criterion(twice)


def test_a_malformed_block_fails_closed(tmp_path):
    text = method_v3.PROTOCOL_PATH.read_text(encoding="utf-8")
    target = tmp_path / "broken.md"
    target.write_text(
        text.replace('"guard_tolerance": 1.0', '"guard_tolerance": '),
        encoding="utf-8")
    with pytest.raises(method_v3.MethodV3Error, match="not valid JSON"):
        method_v3.check_protocol_criterion(target)


def test_a_missing_protocol_document_fails_closed(tmp_path):
    with pytest.raises(method_v3.MethodV3Error, match="protocol document is missing"):
        method_v3.check_protocol_criterion(tmp_path / "nope.md")


def test_the_protocol_names_both_verdict_labels(tmp_path):
    text = method_v3.PROTOCOL_PATH.read_text(encoding="utf-8")
    target = tmp_path / "protocol.md"
    target.write_text(
        text.replace("C_DOWNSTREAM_NOT_SUPPORTED", "SOMETHING_ELSE"),
        encoding="utf-8")
    with pytest.raises(method_v3.MethodV3Error, match="never names the verdict"):
        method_v3.check_protocol_criterion(target)


def test_the_protocol_prose_does_not_contradict_the_block():
    """The block is authoritative, but the reader must not be told otherwise.

    Checked tolerantly on purpose: a value may appear as ``1`` or ``1.0`` or
    ``1.00`` and all of those are the same number. What is rejected is prose that
    states a *different* number from the block.
    """

    text = method_v3.PROTOCOL_PATH.read_text(encoding="utf-8")
    section = text[text.index("## 11."):text.index("## 12.")]
    declared = method_v3.parse_criterion_block(text)

    for field in ("guard_tolerance", "minimum_improving_seeds", "budget"):
        value = declared[field]
        renderings = {f"{value:g}", str(value), f"{float(value):.1f}",
                      str(int(value)) if float(value).is_integer() else str(value)}
        assert any(rendering in section for rendering in renderings), (
            field, value, sorted(renderings))
    for field in ("primary_metric", "guard_metric", "treatment", "control"):
        assert declared[field] in section, field


def test_all_three_clauses_pass_gives_positive():
    verdict = method_v3.evaluate_criterion(
        _design({0: 11.0, 1: 12.0, 2: 13.0}, {0: 10.0, 1: 10.0, 2: 10.0})
    )
    assert verdict.label == "C_DOWNSTREAM_POSITIVE"
    assert verdict.positive and not verdict.failed_clauses()


def test_a_mean_gain_carried_by_one_seed_is_not_enough():
    """Clause 2 exists for exactly this: +30 on one seed, worse on two."""

    verdict = method_v3.evaluate_criterion(
        _design({0: 40.0, 1: 9.0, 2: 9.0}, {0: 10.0, 1: 10.0, 2: 10.0})
    )
    assert verdict.label == "C_DOWNSTREAM_NOT_SUPPORTED"
    assert verdict.clauses["mean_improves"]
    assert not verdict.clauses["majority_of_paired_seeds_improve"]


def test_a_majority_without_a_mean_gain_is_not_enough():
    verdict = method_v3.evaluate_criterion(
        _design({0: 10.5, 1: 10.5, 2: 1.0}, {0: 10.0, 1: 10.0, 2: 10.0})
    )
    assert verdict.label == "C_DOWNSTREAM_NOT_SUPPORTED"
    assert verdict.clauses["majority_of_paired_seeds_improve"]
    assert not verdict.clauses["mean_improves"]


def test_a_gain_bought_by_collapsing_the_known_classes_is_not_positive():
    verdict = method_v3.evaluate_criterion(_design(
        {0: 20.0, 1: 20.0, 2: 20.0}, {0: 10.0, 1: 10.0, 2: 10.0},
        guard_treatment=40.0, guard_control=50.0,
    ))
    assert verdict.label == "C_DOWNSTREAM_NOT_SUPPORTED"
    assert verdict.failed_clauses() == ("known_map_within_tolerance",)


def test_the_guard_tolerance_is_exactly_one_point():
    inside = method_v3.evaluate_criterion(_design(
        {0: 11.0, 1: 11.0, 2: 11.0}, {0: 10.0, 1: 10.0, 2: 10.0},
        guard_treatment=49.0, guard_control=50.0,
    ))
    outside = method_v3.evaluate_criterion(_design(
        {0: 11.0, 1: 11.0, 2: 11.0}, {0: 10.0, 1: 10.0, 2: 10.0},
        guard_treatment=48.9, guard_control=50.0,
    ))
    assert inside.positive and not outside.positive


def test_a_verdict_is_refused_on_an_incomplete_design():
    rows = _design({0: 11.0, 1: 12.0, 2: 13.0}, {0: 10.0, 1: 10.0, 2: 10.0})
    with pytest.raises(method_v3.MethodV3Error, match="missing seeds"):
        method_v3.evaluate_criterion([row for row in rows if row["seed"] != 2])


def test_a_verdict_ignores_rows_at_another_budget():
    rows = _design({0: 11.0, 1: 12.0, 2: 13.0}, {0: 10.0, 1: 10.0, 2: 10.0})
    noise = [dict(row, budget=100, mAP50_medium_tail=999.0) for row in rows]
    assert method_v3.evaluate_criterion(rows + noise).positive


def test_the_verdict_records_the_three_paired_differences():
    verdict = method_v3.evaluate_criterion(
        _design({0: 11.0, 1: 12.0, 2: 13.0}, {0: 10.0, 1: 10.0, 2: 10.0})
    )
    assert verdict.detail["paired_differences"] == {"0": 1.0, "1": 2.0, "2": 3.0}
    assert verdict.detail["n_seeds"] == 3
    assert "no significance claim" in verdict.detail["significance"]


# ---------------------------------------------------- the frozen constants ---


def test_the_replay_arm_is_the_established_matched_control():
    assert method_v3.REPLAY_ARM == "uniform"
    assert replay.ARMS["uniform"] == {"total": 400, "alpha": 0.0}


def test_the_annotation_and_training_settings_are_the_established_ones():
    assert method_v3.LABELLING_POLICY == "known_plus_selected"
    assert method_v3.SUPERVISION_MODE == "ft"
    assert (method_v3.EPOCHS, method_v3.LEARNING_RATE, method_v3.BATCH_SIZE) == (
        5, 2e-4, 2)


def test_the_task_is_one_incremental_step_declaring_one_class():
    chain = protocol.build_chain(method_v3.N_TASKS)
    assert len(chain) == 2 and chain[0].is_anchor
    assert chain[1].new_class == protocol.CLASS_ORDER[protocol.N_TASK1]
    assert chain[1].n_new == 1


def test_the_tail_band_holds_one_class_and_medium_tail_holds_eight():
    task = protocol.build_chain(method_v3.N_TASKS)[1]
    groups = protocol.load_groups()
    medium_tail = method_v3.medium_tail_classes(task.known_classes, groups)
    tail = [n for n in task.known_classes if groups.get(n) == "tail"]
    assert tail == ["bear"], tail
    assert len(medium_tail) == 8, medium_tail


def test_the_protocol_prints_the_gpu_path_supervision_note(v3_population,
                                                           candidate_index):
    text = method_v3.annotation_protocol(method_v3.TrajectoryInputs(
        pool=v3_population, candidate_index=candidate_index, replay_index={},
        replay_root=Path("/nowhere"), start_checkpoint=Path("/nowhere/t1.pth"),
        test_set="owl_shared_test",
    ))
    for needle in ("known_plus_selected", "uniform", "IoU 0.5", "IoU 0.60",
                   "owl.discovery", "WHICH IMAGES", "600 regions in 6 rounds"):
        assert needle in text, needle


# ------------------------------------------------------------ bookkeeping ---


def test_trajectory_names_are_filesystem_safe():
    names = [method_v3.trajectory_name(a, s) for a, s in method_v3.trajectories()]
    assert len(set(names)) == 12
    assert all("*" not in name and "/" not in name for name in names)
    assert "a_times_c__seed2" in names


def test_write_json_is_atomic_and_leaves_no_partial(tmp_path):
    target = tmp_path / "deep" / "result.json"
    method_v3.write_json(target, {"status": "complete"})
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "complete"
    assert not list(tmp_path.rglob("*.partial"))


def test_an_incomplete_trajectory_is_not_loaded_as_a_result(tmp_path):
    method_v3.write_json(tmp_path / "result.json", {"status": "failed"})
    assert method_v3.load_trajectory(tmp_path) is None
    method_v3.write_json(tmp_path / "result.json", {"status": "running"})
    assert method_v3.load_trajectory(tmp_path) is None
    method_v3.write_json(tmp_path / "result.json", {"status": "complete", "arm": "A"})
    assert method_v3.load_trajectory(tmp_path)["arm"] == "A"


def test_a_corrupt_trajectory_fails_closed(tmp_path):
    (tmp_path / "result.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(method_v3.MethodV3Error, match="corrupt"):
        method_v3.load_trajectory(tmp_path)


def test_a_missing_trajectory_is_simply_absent(tmp_path):
    assert method_v3.load_trajectory(tmp_path) is None


def test_dry_run_is_part_of_the_trajectory_identity(v3_population):
    real = method_v3.fingerprint("A", 0, v3_population, test_set="t", dry_run=False)
    stubbed = method_v3.fingerprint("A", 0, v3_population, test_set="t", dry_run=True)
    assert real["dry_run"] is False and stubbed["dry_run"] is True
    assert real != stubbed


def test_the_fingerprint_carries_every_result_affecting_setting(v3_population):
    print_ = method_v3.fingerprint("A", 0, v3_population, test_set="owl_shared_test")
    for key in ("arm", "seed", "budget", "rounds", "replay_arm", "replay_total",
                "labelling_policy", "epochs", "learning_rate", "batch_size",
                "test_set", "eval_max_per_class", "eval_remainder_ratio",
                "population_rows", "population_keys_sha256", "criterion"):
        assert key in print_, key


# ------------------------------------ the whole orchestration, PROB stubbed ---


@pytest.fixture(scope="module")
def stubbed_run(tmp_path_factory):
    """One stubbed 12-trajectory walk, reused by the tests below.

    Proves the schedule, the resume behaviour, the manifest and the summariser
    without a GPU. Every detector number is synthetic; the point is the plumbing.
    """

    workspace = tmp_path_factory.mktemp("method_v3_dry")
    data_root = workspace / "data"
    fake = workspace / "fake"
    fake.mkdir()

    from owl import semantic_features as sf
    from tools.export_dinov2_consistency_views import VIEW_MARGINS, VIEWS_VERSION

    rows = sf.pool_rows(POOL_PATH)
    generator = np.random.default_rng(3)

    def unit(count):
        values = generator.normal(size=(count, sf.FEATURE_DIM)).astype(np.float32)
        return (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float16)

    base = unit(rows.keys.size)
    sf.write(fake / "base.npz", sf.SemanticExport(
        embeddings=base, keys=rows.keys, image_ids=rows.image_ids,
        query_index=rows.query_index, row_index=rows.row_index,
        provenance={"model_id": "SYNTHETIC-test", "note": "not DINOv2"},
    ))
    pool = method_v3.population(
        POOL_PATH, json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    )
    anchor = base[pool.p2_mask].astype(np.float32)

    def near(matrix):
        moved = matrix + 0.3 * generator.normal(size=matrix.shape).astype(np.float32)
        return (moved / np.linalg.norm(moved, axis=1, keepdims=True)).astype(np.float16)

    np.savez(
        fake / "views.npz",
        keys=np.asarray(rows.keys[pool.p2_mask], dtype=str),
        view_a=near(anchor), view_b=near(anchor),
        provenance=np.asarray(str({"view_margins": VIEW_MARGINS})),
        views_version=np.asarray(VIEWS_VERSION),
    )

    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "prepare_method_v3_data.py"),
         "--data-root", str(data_root), "--annotations-only"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    out = workspace / "results"
    command = [
        sys.executable, str(ROOT / "tools" / "run_method_v3.py"),
        "--prob-root", str(ROOT), "--data-root", str(data_root),
        "--checkpoint", str(POOL_PATH),
        "--export", str(fake / "base.npz"), "--views", str(fake / "views.npz"),
        "--out", str(out), "--dry-run",
    ]
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return {"out": out, "data_root": data_root, "fake": fake,
            "command": command, "first": first.stdout, "second": second.stdout}


def test_the_stubbed_run_schedules_all_twelve(stubbed_run):
    for arm, seed in method_v3.trajectories():
        assert f"{arm}/s{seed}" in stubbed_run["first"]
    assert "ALL 12 TRAJECTORIES COMPLETE" in stubbed_run["first"]


def test_every_trajectory_writes_a_complete_record(stubbed_run):
    for arm, seed in method_v3.trajectories():
        directory = stubbed_run["out"] / method_v3.trajectory_name(arm, seed)
        record = method_v3.load_trajectory(directory)
        assert record is not None, directory
        assert record["arm"] == arm and record["seed"] == seed
        assert record["oracle_cost"] == method_v3.BUDGET
        assert record["asked"] == method_v3.BUDGET
        assert record["replay_objects"] == 400
        assert record["dry_run"] is True
        assert record["half_labelled_share"] == 0.0
        assert (directory / "selection_curve.csv").is_file()
        assert (directory / "status.json").is_file()


def test_the_selection_curve_has_the_six_budget_marks(stubbed_run):
    import csv

    path = (stubbed_run["out"] / method_v3.trajectory_name("A*C", 1)
            / "selection_curve.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["budget"]) for row in rows] == list(method_v3.BUDGET_MARKS)
    assert [int(row["asked"]) for row in rows] == list(method_v3.BUDGET_MARKS)
    counts = [int(row["unknown_objects"]) for row in rows]
    assert counts == sorted(counts), "cumulative discoveries cannot decrease"


def test_distinct_objects_never_exceed_the_proposals_that_found_them(stubbed_run):
    for arm, seed in method_v3.trajectories():
        record = method_v3.load_trajectory(
            stubbed_run["out"] / method_v3.trajectory_name(arm, seed))
        assert record["unknown_objects"] <= record["unknown_proposals"]
        bands = sum(record[f"{band}_objects"] for band in ("head", "medium", "tail"))
        assert bands == record["unknown_objects"], (arm, seed)


def test_the_second_run_resumes_instead_of_repeating(stubbed_run):
    assert stubbed_run["second"].count("already complete") == 12
    assert "done in" not in stubbed_run["second"]


def test_the_manifest_pins_everything_needed_to_re_derive_the_run(stubbed_run):
    manifest = json.loads(
        (stubbed_run["out"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "method_v3_selection_transfer"
    assert manifest["dry_run"] is True
    assert manifest["arms"] == list(method_v3.ARMS)
    assert manifest["seeds"] == list(method_v3.SEEDS)
    assert manifest["replay"]["arm"] == "uniform"
    assert manifest["does_not_reopen"]["method_v2_stage2"]["allowed_ladder"] == "U"
    assert manifest["population"]["rows"] == 8_010
    assert manifest["criterion"]["guard_tolerance"] == 1.0
    assert manifest["checkpoint_sha256"]
    assert all(state == method_v3.STATUS_COMPLETE
               for state in manifest["trajectory_status"].values())


def test_the_directory_is_marked_as_stubbed(stubbed_run):
    assert (stubbed_run["out"] / "DRY_RUN").is_file()


def test_a_real_run_refuses_to_resume_a_stubbed_trajectory(stubbed_run,
                                                           v3_population):
    """The one mistake that must be impossible: synthetic numbers reported as real.

    Driven straight at :func:`owl.method_v3.run_trajectory` rather than through
    the launcher, because the launcher's ``bridge.check()`` refuses a missing
    PROB checkout first and would prove the wrong guard.
    """

    class Unused:
        def cost_report(self):
            return {"total": 0.0}

    workspace = stubbed_run["out"] / method_v3.trajectory_name("A", 0)
    assert method_v3.load_trajectory(workspace) is not None
    inputs = method_v3.TrajectoryInputs(
        pool=v3_population, candidate_index={}, replay_index={},
        replay_root=workspace, start_checkpoint=POOL_PATH,
        test_set="owl_shared_test",
        consistency=np.full(len(v3_population), 0.8, dtype=np.float32),
    )
    with pytest.raises(method_v3.MethodV3Error) as raised:
        method_v3.run_trajectory(
            Unused(), "A", 0, workspace=workspace, inputs=inputs, dry_run=False
        )
    message = str(raised.value)
    assert "different configuration" in message
    assert "dry_run" in message
    # and the stubbed result is still there, untouched
    assert method_v3.load_trajectory(workspace)["dry_run"] is True


def test_a_failed_trajectory_is_visibly_failed(stubbed_run, tmp_path):
    """A crash must leave FAILED on disk and must not be counted as complete."""

    class Exploding:
        def cost_report(self):
            return {"total": 0.0}

        def evaluate(self, **_):
            raise RuntimeError("synthetic evaluation failure")

    pool = method_v3.population(
        POOL_PATH, json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    )
    inputs = method_v3.TrajectoryInputs(
        pool=pool, candidate_index={}, replay_index={},
        replay_root=tmp_path, start_checkpoint=POOL_PATH, test_set="owl_shared_test",
        consistency=np.full(len(pool), 0.8, dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        method_v3.run_trajectory(
            Exploding(), "A", 0, workspace=tmp_path / "boom", inputs=inputs
        )
    state = json.loads(
        (tmp_path / "boom" / "status.json").read_text(encoding="utf-8"))
    assert state["status"] == method_v3.STATUS_FAILED
    assert "synthetic evaluation failure" in state["error"]
    assert method_v3.load_trajectory(tmp_path / "boom") is None


def test_the_summariser_reports_the_verdict_and_marks_the_dry_run(stubbed_run):
    probe = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "summarize_method_v3.py"),
         "--results", str(stubbed_run["out"])],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    text = probe.stdout
    assert "DRY RUN" in text and "SYNTHETIC" in text
    assert "PRIMARY CONTRAST: A*C - A @ budget 600" in text
    assert "PER-BUDGET SELECTION CURVES" in text
    assert "NOT a detector curve" in text
    assert "tail band at this task: ['bear']" in text
    assert "SINGLE class" in text
    assert "selection determinism" in text
    for label in ("U vs A", "A*C vs U", "A vs random"):
        assert label in text, label
    assert ("C_DOWNSTREAM_POSITIVE" in text
            or "C_DOWNSTREAM_NOT_SUPPORTED" in text)
    summary = json.loads(
        (stubbed_run["out"] / "method_v3_summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["trajectories_complete"] == 12


def test_the_summariser_declines_a_verdict_on_a_partial_design(stubbed_run, tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    for arm, seed in method_v3.trajectories():
        if arm == "A*C" and seed == 2:
            continue
        source = stubbed_run["out"] / method_v3.trajectory_name(arm, seed)
        target = partial / method_v3.trajectory_name(arm, seed)
        target.mkdir()
        (target / "result.json").write_text(
            (source / "result.json").read_text(encoding="utf-8"), encoding="utf-8")
    probe = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "summarize_method_v3.py"),
         "--results", str(partial)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "VERDICT NOT COMPUTED" in probe.stdout
    assert "missing seeds" in probe.stdout


# --------------------------------------------------------------- the plan ---


def test_the_planner_refuses_to_reduce_the_schedule_silently():
    probe = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "plan_method_v3.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "DECISION: no reduction" in probe.stdout
    assert f"epochs={method_v3.EPOCHS}" in probe.stdout
    assert "C_DOWNSTREAM_POSITIVE  iff" in probe.stdout


def test_the_planner_prices_all_four_arms_and_the_anchor():
    probe = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "plan_method_v3.py")],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    for arm in method_v3.ARMS:
        assert arm in probe.stdout
    assert "12 trajectories (h)" in probe.stdout


def test_the_preparer_reports_the_union_without_downloading(tmp_path):
    probe = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "prepare_method_v3_data.py"),
         "--data-root", str(tmp_path), "--verify-only"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "839 candidate images  PASS" in probe.stdout
    assert "600 shared test images" in probe.stdout
    assert "nothing downloaded" in probe.stdout
    assert not any(tmp_path.iterdir())
