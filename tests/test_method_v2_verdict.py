"""The corrected Method V2 semantic gate: all four criteria, and nothing else.

The gate was wrong when first written -- the 0.15 threshold was placed on
open-pool NMI instead of open-pool kNN -- and a dry run on random features showed
why that mattered: pure noise reaches open-pool NMI 0.30-0.34, so the criterion
was satisfied by a representation carrying no information. The correction was made
before any DINOv2 feature existed. These tests pin the corrected rule so it cannot
drift back, and pin the property that motivated the correction: **noise must
fail.**
"""

from __future__ import annotations

import pytest

from tools.audit_dinov2_representation import (
    DECISION_POPULATION,
    DECISION_REPRESENTATION,
    GO_AUC_UNKNOWN_VS_BACKGROUND,
    GO_MARGIN_OVER_PROB,
    GO_OPEN_POOL_KNN,
    GO_UNKNOWN_KNN,
    PROB_P2_UNKNOWN_KNN,
    verdict,
)


def rows(*, unknown_knn, open_pool_knn=0.20, auc=0.85, nmi=0.40,
         source="dinov2_vitb14", representation=None, population=None):
    """Per-seed rows on the decision cell. ``unknown_knn`` may be a list."""

    values = unknown_knn if isinstance(unknown_knn, (list, tuple)) else [unknown_knn] * 3
    return [
        {
            "source": source,
            "representation": representation or DECISION_REPRESENTATION,
            "population": population or DECISION_POPULATION,
            "seed": seed,
            "unknown_knn": value,
            "open_pool_unknown_knn": open_pool_knn,
            "auc_unknown_vs_background": auc,
            "open_pool_nmi": nmi,
        }
        for seed, value in enumerate(values)
    ]


# ---------------------------------------------------------- the frozen values ---


def test_the_frozen_constants_are_the_decoder_layer_rescue_values():
    """None of the four numbers is new to Method V2; drift here is a silent change."""

    assert GO_UNKNOWN_KNN == 0.30
    assert GO_OPEN_POOL_KNN == 0.15
    assert GO_AUC_UNKNOWN_VS_BACKGROUND == 0.76      # 0.95 x layer-5 AUC 0.8000
    assert GO_MARGIN_OVER_PROB == 0.05
    assert PROB_P2_UNKNOWN_KNN == 0.1772
    assert DECISION_REPRESENTATION == "whitened32"
    assert DECISION_POPULATION == "P2_admissible_nms"


# ------------------------------------------------------------------ the gate ---


def test_all_four_criteria_met_passes():
    result = verdict(rows(unknown_knn=0.35))

    assert result["verdict"] == "PASS"
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("kwargs", "failing"),
    [
        ({"unknown_knn": 0.29}, "1_unknown_knn>=0.30"),
        ({"unknown_knn": 0.35, "open_pool_knn": 0.14}, "2_open_pool_knn>=0.15"),
        ({"unknown_knn": 0.35, "auc": 0.75}, "3_auc>=0.76"),
    ],
)
def test_each_criterion_can_fail_on_its_own(kwargs, failing):
    result = verdict(rows(**kwargs))

    assert result["verdict"] == "FAIL"
    assert result["checks"][failing] is False
    assert sum(not ok for ok in result["checks"].values()) >= 1


def test_the_margin_is_checked_per_seed_not_on_the_mean():
    """A margin that holds on average but fails on one seed is the artefact the
    criterion exists to exclude. Mean 0.2272 clears +0.05; the worst seed does not.
    """

    per_seed = [0.32, 0.32, 0.2172]        # mean 0.2857, worst margin +0.0400
    result = verdict(rows(unknown_knn=per_seed))

    assert result["checks"]["4_margin>=0.05_every_seed"] is False
    assert result["worst_seed_margin"] == pytest.approx(0.2172 - PROB_P2_UNKNOWN_KNN)
    assert result["verdict"] == "FAIL"


def test_nmi_takes_no_part_in_the_decision():
    """The correction's whole point: NMI is reported, never decisive.

    Noise reaches 0.30-0.34 on this population, so an NMI threshold would be met
    by a representation with no information in it.
    """

    passing = verdict(rows(unknown_knn=0.35, nmi=0.0))
    failing = verdict(rows(unknown_knn=0.10, nmi=0.99))

    assert passing["verdict"] == "PASS"      # passes with NMI at zero
    assert failing["verdict"] == "FAIL"      # fails with NMI at ceiling
    assert "nmi" not in " ".join(passing["checks"])


def test_noise_like_metrics_fail_the_corrected_gate():
    """The measured random-noise floor, as observed on the synthetic dry run."""

    result = verdict(rows(unknown_knn=0.0652, open_pool_knn=0.0038,
                          auc=0.5024, nmi=0.3356,
                          source="random_noise_floor"),
                     source="random_noise_floor")

    assert result["verdict"] == "FAIL"
    assert result["checks"] == {
        "1_unknown_knn>=0.30": False,
        "2_open_pool_knn>=0.15": False,
        "3_auc>=0.76": False,
        "4_margin>=0.05_every_seed": False,
    }


def test_the_prob_baseline_itself_fails_the_gate():
    """Sanity: the gate is a bar PROB does not clear, or it measures nothing."""

    result = verdict(rows(unknown_knn=PROB_P2_UNKNOWN_KNN, open_pool_knn=0.0714,
                          auc=0.8000, source="prob_hs5_baseline"),
                     source="prob_hs5_baseline")

    assert result["verdict"] == "FAIL"
    assert result["checks"]["3_auc>=0.76"] is True        # PROB clears the safeguard
    assert result["checks"]["1_unknown_knn>=0.30"] is False


# ----------------------------------------------------------------- plumbing ---


def test_only_the_decision_cell_is_consulted():
    """Rows from another representation or population must not leak in."""

    mixed = (
        rows(unknown_knn=0.35)
        + rows(unknown_knn=0.05, representation="unit")
        + rows(unknown_knn=0.05, population="P0_raw")
    )

    result = verdict(mixed)

    assert result["seeds"] == 3
    assert result["verdict"] == "PASS"


def test_a_missing_decision_cell_is_indeterminate_not_a_pass():
    result = verdict(rows(unknown_knn=0.35, population="P0_raw"))

    assert result["verdict"] == "INDETERMINATE"
    assert "checks" not in result
