"""The post-hoc audit: read-only, and its arithmetic pinned.

Two properties matter most here. The audit must never write into the frozen
Method V3 results directory, and the numbers it reports must be the ones the
document quotes — an audit whose tables drift from its prose is worse than none.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from owl import method_v3, protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import audit_method_v3 as audit

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
POOL = ROOT / "data" / "pool" / "sowodb_t1_frozen_pool.npz"


@pytest.fixture(scope="module")
def candidate_index():
    return json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def audited(candidate_index):
    from owl import scoring

    pool = method_v3.population(POOL, candidate_index)
    return pool, scoring.admissibility(pool.candidates)


# ------------------------------------------------------------- arithmetic ---


def test_spearman_is_one_on_a_monotone_transform():
    values = np.linspace(1.0, 5.0, 50)
    assert audit.spearman(values, values ** 3) == pytest.approx(1.0)
    assert audit.spearman(values, -values) == pytest.approx(-1.0)


def test_jaccard_matches_its_definition():
    import math

    assert audit.jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)
    assert audit.jaccard({1}, {1}) == 1.0
    assert audit.jaccard({1}, {2}) == 0.0
    assert math.isnan(audit.jaccard(set(), set()))


def test_discordant_pairs_counts_only_real_inversions():
    left = np.array([3.0, 2.0, 1.0])
    index = np.array([0, 1, 2])
    assert audit.discordant_pairs(left, left, index) == 0
    assert audit.discordant_pairs(left, -left, index) == 3
    # one swap of the middle pair
    right = np.array([3.0, 1.0, 2.0])
    assert audit.discordant_pairs(left, right, index) == 1


def test_prefix_is_the_top_k_by_score():
    score = np.array([0.1, 0.9, 0.5, 0.7])
    assert audit.prefix(score, 2).tolist() == [1, 3]


def test_c_distribution_reports_every_requested_percentile():
    row = audit.c_distribution(np.linspace(0.5, 1.0, 1001))
    for percentile in (0, 1, 5, 25, 50, 75, 95, 99, 100):
        assert f"p{percentile}" in row
    assert row["p0"] == pytest.approx(0.5)
    assert row["p100"] == pytest.approx(1.0)
    assert row["dynamic_range_max_over_min"] == pytest.approx(2.0)


# ----------------------------------------------- the numbers the doc quotes ---


def test_the_gap_ratio_at_every_cut_is_far_below_any_cosine_spread(audited):
    """The bound §1.3 rests on: A is dense to ~1e-5 at the cuts."""

    _, admissibility = audited
    rows = audit.a_gap_structure(admissibility)
    assert [row["budget"] for row in rows] == list(method_v3.BUDGET_MARKS)
    for row in rows:
        assert 1.0 < row["gap_ratio"] < 1.001, row
    final = rows[-1]
    assert final["budget"] == 600
    assert final["gap_ratio"] == pytest.approx(1.0000297, abs=1e-6)
    assert final["within_0.01"] == 32


def test_the_supervision_chain_reproduces_the_documented_numbers(audited,
                                                                candidate_index):
    pool, _ = audited
    row = audit.supervision_chain(
        pool.candidates, "A", 0, candidate_index=candidate_index)
    assert row["regions"] == 600
    assert row["images_opened"] == 590
    assert row["images_trainable"] == 343
    assert row["images_barren"] == 247
    assert row["supervised_boxes"] == 972
    assert row["boxes_per_region"] == pytest.approx(1.62, abs=0.005)
    assert row["undeclared_boxes_dropped"] == 1759
    assert row["new_class_boxes"] == 33
    assert row["new_class_images"] == 15
    assert row["person_boxes"] == 766
    assert row["acquired_unknown_objects"] == 150
    assert row["acquired_medium_tail_objects"] == 79
    assert row["trainable_source"] == "recomputed"


def test_u_receives_twice_the_supervision_of_a_for_the_same_budget(audited,
                                                                  candidate_index):
    """§4: the budget is matched on oracle cost, not on supervision."""

    pool, _ = audited
    a = audit.supervision_chain(
        pool.candidates, "A", 0, candidate_index=candidate_index)
    u = audit.supervision_chain(
        pool.candidates, "U", 0, candidate_index=candidate_index)
    assert a["regions"] == u["regions"] == 600
    assert u["supervised_boxes"] == 2027
    assert u["supervised_boxes"] / a["supervised_boxes"] == pytest.approx(
        2.09, abs=0.01)
    # and the acquisition ordering is the opposite way round
    assert a["acquired_unknown_objects"] > 4 * u["acquired_unknown_objects"]


def test_the_tail_band_received_no_new_supervision_in_any_arm(audited,
                                                              candidate_index):
    """§3.4: mAP50_tail at t2 is one class, and it got nothing."""

    pool, _ = audited
    task = protocol.build_chain(method_v3.N_TASKS)[1]
    groups = protocol.load_groups()
    tail = [n for n in task.known_classes if groups.get(n) == "tail"]
    assert tail == ["bear"]
    for arm in ("random", "A", "U"):
        row = audit.supervision_chain(
            pool.candidates, arm, 0, candidate_index=candidate_index)
        assert row["per_class"].get("bear", 0) == 0, arm


def test_only_the_new_class_among_the_unknowns_is_declarable_at_t2(audited):
    """§3.3, stated exactly rather than tidily.

    "Unknown" in the frozen pool means "not one of t1's 19 classes", so a
    traffic-light proposal is unknown there and *is* declared at t2. Exactly one
    of the 42 unknown classes crosses that line, which is why the acquisition
    advantage cannot reach the detector.
    """

    pool, _ = audited
    task = protocol.build_chain(method_v3.N_TASKS)[1]
    declared = set(protocol.CLASS_ORDER[: task.n_current])
    oracle = pool.candidates.oracle()
    unknown = oracle.kind == "unknown"
    names = set(oracle.class_name[unknown].tolist())
    assert len(names) == 42
    assert names & declared == {task.new_class} == {"traffic light"}


def test_the_new_class_is_almost_absent_from_the_candidate_population(audited):
    """§3.2 stage 1: two acquirable instances, so no arm could select it."""

    import numpy as np

    pool, _ = audited
    oracle = pool.candidates.oracle()
    new_class = oracle.class_name == "traffic light"
    assert int(new_class.sum()) == 3
    assert np.unique(oracle.object_id[new_class]).size == 2


def test_the_new_class_is_not_suppressed_by_box_size(audited):
    """§3.2: the tidy 'A hides small objects' story is measured and refused."""

    import numpy as np

    from owl import proposals as proposals_module
    from owl import scoring
    from owl import semantic_features as sf

    whole = proposals_module.from_frozen_pool(sf.POOL, split="pool")
    oracle = whole.oracle()
    new_class = oracle.class_name == "traffic light"
    admissibility = scoring.admissibility(whole)
    assert np.median(admissibility[new_class]) > np.median(admissibility)
    assert np.median(whole.area[new_class]) > np.median(whole.area)


def test_acquired_unknowns_are_dropped_by_the_declared_class_filter(audited):
    """§3.3: the per-arm drop counts the document quotes."""

    import numpy as np

    pool, _ = audited
    task = protocol.build_chain(method_v3.N_TASKS)[1]
    declared = sorted(set(protocol.CLASS_ORDER[: task.n_current]))
    oracle = pool.candidates.oracle()
    unknown = oracle.kind == "unknown"

    expected = {"random": (40, 1), "A": (150, 0), "U": (36, 0)}
    for arm, (total, declarable) in expected.items():
        index = method_v3.select_for_arm(pool.candidates, arm, 0).indices
        chosen = unknown[index]
        ids = oracle.object_id[index][chosen]
        names = oracle.class_name[index][chosen]
        assert np.unique(ids[ids >= 0]).size == total, arm
        keep = (ids >= 0) & np.isin(names, declared)
        assert np.unique(ids[keep]).size == declarable, arm


def test_the_seed_moves_the_rehearsal_set_and_not_the_selection(audited,
                                                               candidate_index):
    pool, _ = audited
    replay_index = json.loads(
        (ROOT / "data" / "reference" / "t1_replay_class_counts.json")
        .read_text(encoding="utf-8"))
    picks = [
        method_v3.select_for_arm(pool.candidates, "A", seed).indices.tolist()
        for seed in method_v3.SEEDS
    ]
    assert picks[0] == picks[1] == picks[2]

    memories = [
        audit.replay_memory(pool.candidates, "A", seed,
                            candidate_index=candidate_index,
                            replay_index=replay_index)
        for seed in method_v3.SEEDS
    ]
    assert all(len(memory) == 400 for memory in memories)
    assert len(memories[0] & memories[1]) <= 5
    assert len(memories[0] & memories[2]) <= 5


def test_prob_was_seeded_identically_in_every_trajectory():
    """§2.1: the bridge is built without a seed, so PROB got 0 everywhere."""

    from owl.bridge import Bridge

    assert Bridge.__dataclass_fields__["seed"].default == 0
    launcher = (ROOT / "tools" / "run_method_v3.py").read_text(encoding="utf-8")
    construction = launcher[launcher.index("bridge = Bridge("):]
    construction = construction[: construction.index(")")]
    assert "seed=" not in construction


# --------------------------------------------------------------- read-only ---


def test_the_audit_refuses_to_write_into_the_results_directory(tmp_path):
    probe = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "audit_method_v3.py"),
         "--results", str(tmp_path), "--out", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert probe.returncode != 0
    assert "must not be --results" in probe.stdout + probe.stderr


def test_the_audit_runs_offline_and_writes_only_to_out(tmp_path):
    out = tmp_path / "audit"
    probe = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "audit_method_v3.py"),
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "POST-HOC MECHANISTIC AUDIT" in probe.stdout
    assert "C_DOWNSTREAM_NOT_SUPPORTED — not revisited here" in probe.stdout
    assert "SKIPPED — pass --export and --views" in probe.stdout
    for name in ("audit_a_gap_structure.csv", "audit_seed_effect.csv",
                 "audit_supervision_chain.csv", "audit_per_class_supervision.csv",
                 "audit_summary.json"):
        assert (out / name).is_file(), name
    summary = json.loads((out / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict_under_audit"] == "C_DOWNSTREAM_NOT_SUPPORTED"
    assert summary["prob_seed_varies_across_trajectories"] is False


def test_the_audit_reads_the_authoritative_training_lists(tmp_path):
    """When the artefacts are there, labelled_ids.txt overrides recomputation."""

    results = tmp_path / "results"
    for arm, seed in method_v3.trajectories():
        directory = results / method_v3.trajectory_name(arm, seed) / "train"
        directory.mkdir(parents=True)
        (directory / "labelled_ids.txt").write_text(
            "000000000139\n000000000285\n", encoding="utf-8")
        (directory / "replay_ids.txt").write_text(
            f"alias_{arm}_{seed}\n", encoding="utf-8")
    artefacts = audit.trajectory_artefacts(results)
    assert artefacts[("A", 0)]["labelled"] == ["000000000139", "000000000285"]

    rows = audit.paired_identity(artefacts, "A*C", "A")
    assert len(rows) == 3
    for row in rows:
        assert row["labelled_identical"] is True
        assert row["replay_identical"] is False
        assert row["replay_intersection"] == 0


def test_the_documented_findings_appear_in_the_audit_document():
    """The prose and the tables must not drift apart."""

    text = (ROOT / "docs" / "method_v3_posthoc_audit_2026-09-03.md").read_text(
        encoding="utf-8")
    assert "C_DOWNSTREAM_NOT_SUPPORTED" in text
    for needle in ("1.0000297", "8,010", "839", "972", "2,027", "1,759",
                   "33", "101", "150", "79", "247", "1.62", "3.38", "2.09",
                   "420,304", "bear", "0.1732", "0.1342", "11,431"):
        assert needle in text, needle
    for heading in ("A. PRE-REGISTERED", "B. POST-HOC SUPPORTED",
                    "C. HYPOTHESES", "D. INVALID CLAIMS"):
        assert heading in text, heading
    assert "single recommended next experiment" in text
