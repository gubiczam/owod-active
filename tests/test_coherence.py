"""The coherence gate: that it fires, on what population, and with what meaning.

These tests exist because of a defect that survived three seeds and two result
documents: ``coherence_method='binary'`` closed on **0 of 80,000** candidates, so
``ARMS['consult']`` and ``ARMS['consult_no_gate']`` were bitwise identical and
the gate the 2026-08-25 consultation asked for was never actually tested.

Two kinds of test follow. The first kind *pins the defect*, so the committed V1
results stay explainable rather than mysterious. The second kind *guards the
repair*: a gate that silently stops firing must fail a test rather than quietly
turn an ablation into a duplicate.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from owl import clustering, scoring, selection

# --------------------------------------------------------------- fixtures ---


@pytest.fixture
def blobs() -> np.ndarray:
    """Three tight clusters plus scattered singletons, unit-norm.

    Small enough to run in milliseconds, structured enough that "core" and
    "noise" are unambiguous: the scattered points are far from everything.
    """

    generator = np.random.default_rng(0)
    centres = np.eye(3, 16, dtype=np.float32)
    clustered = np.repeat(centres, 40, axis=0) + generator.normal(0, 0.01, (120, 16))
    scattered = generator.normal(0, 1.0, (12, 16))
    points = np.vstack([clustered, scattered]).astype(np.float32)
    return points / np.linalg.norm(points, axis=1, keepdims=True)


# ------------------------------------------------- pinning the V1 defect ---


def test_kmeans_binary_gate_is_a_no_op_when_every_cluster_clears_the_floor(blobs):
    """Why ``consult`` and ``consult_no_gate`` came out identical.

    ``noise_gate`` gates on cluster *size*. k-means produces no noise points at
    all, so whenever every cluster holds at least ``minimum_size`` members the
    gate is open everywhere and is indistinguishable from having no gate. On the
    committed pool at K=1600 the smallest cluster holds 5 members, which is
    exactly the configured floor -- hence 0 of 80,000 gated.
    """

    partition = clustering.fit(blobs, method="kmeans", n_clusters=3, seed=0)
    gate = clustering.noise_gate(partition, minimum_size=5)

    assert gate.min() == 1.0
    assert np.array_equal(gate, np.ones_like(gate)), (
        "documented V1 behaviour: the k-means size gate is a no-op here"
    )


# ---------------------------------------------------- guarding the repair ---


def test_density_gate_actually_fires(blobs):
    """The repair's whole point: some candidate must be gated out.

    If this ever passes vacuously the ablation has silently become a duplicate
    of its own control, which is the failure mode that produced the defect.
    """

    gate = clustering.density_coherence(
        blobs, eps=0.2, min_samples=5, pca_dimensions=None, seed=0
    )

    assert 0.0 < float((gate.gate == 0).mean()) < 1.0, (
        "a gate that closes on nobody, or on everybody, tests nothing"
    )
    assert gate.n_clusters >= 1


def test_density_gate_keeps_dense_points_and_drops_scattered_ones(blobs):
    """Predicted sign. The 120 clustered points survive, the 12 strays do not."""

    gate = clustering.density_coherence(
        blobs, eps=0.2, min_samples=5, pca_dimensions=None, seed=0
    )

    clustered_kept = gate.gate[:120].mean()
    scattered_kept = gate.gate[120:].mean()
    assert clustered_kept > scattered_kept
    assert clustered_kept == 1.0


def test_scope_restricts_the_fit_and_closes_the_gate_outside_it(blobs):
    """Out of scope means ``C(x) = 0`` and label ``-2``, distinct from noise ``-1``.

    Keeping the two reasons apart is what lets the diagnostic judge
    discrimination *within* the scope. Merged, it would measure admissibility.
    """

    scope = np.zeros(blobs.shape[0], dtype=bool)
    scope[:120] = True
    gate = clustering.density_coherence(
        blobs, scope=scope, eps=0.2, min_samples=5, pca_dimensions=None, seed=0
    )

    assert gate.params["scope_size"] == 120
    assert (gate.gate[120:] == 0).all()
    assert (gate.labels[120:] == -2).all()
    assert not gate.is_noise[120:].any(), "outside the scope is not DBSCAN noise"
    assert gate.summary()["scope_size"] == 120


def test_admissible_mask_takes_the_top_share_and_is_deterministic():
    scores = np.arange(100, dtype=np.float64)

    mask = clustering.admissible_mask(scores, 0.30)
    assert mask.sum() == 30
    assert mask[70:].all() and not mask[:70].any()

    assert np.array_equal(mask, clustering.admissible_mask(scores, 0.30))
    assert clustering.admissible_mask(scores, 1.0).all()


def test_empty_scope_is_answered_not_crashed(blobs):
    gate = clustering.density_coherence(
        blobs, scope=np.zeros(blobs.shape[0], dtype=bool), pca_dimensions=None
    )

    assert (gate.gate == 0).all()
    assert gate.n_clusters == 0
    assert np.isnan(gate.summary()["noise_share_within_scope"])


# ------------------------------------------------------- the arm contract ---


def test_every_gated_arm_in_the_ladder_has_a_gate_that_can_fire(blobs):
    """The regression guard the defect asked for.

    An arm whose ``coherence_method`` is not ``'off'`` must produce a coherence
    vector that differs from the ungated one. If a future parameter change makes
    a gate vacuous again, this fails instead of turning an ablation rung into a
    copy of its control.
    """

    from owl.proposals import Candidates, Oracle

    generator = np.random.default_rng(1)
    n = blobs.shape[0]
    posterior = generator.random((n, 81)).astype(np.float32)
    posterior /= posterior.sum(axis=1, keepdims=True)
    pool = Candidates(
        image_ids=np.array([f"im{i // 4}" for i in range(n)]),
        boxes=np.column_stack([
            generator.random(n), generator.random(n),
            np.full(n, 0.2), np.full(n, 0.2),
        ]).astype(np.float32),
        embeddings=blobs,
        posterior=posterior,
        objectness=generator.random(n).astype(np.float32),
        _oracle=Oracle(
            kind=np.array(["background"] * n),
            class_name=np.array([""] * n),
            object_id=np.full(n, -1, dtype=np.int64),
            iou=np.zeros(n, dtype=np.float32),
        ),
    )
    partition = clustering.fit(blobs, n_clusters=3, seed=0)
    ungated = np.ones(n, dtype=np.float32)

    checked = 0
    for name, config in selection.ARMS_V2.items():
        if config.coherence_method in ("off", "continuous") or config.random:
            continue
        tuned = type(config)(**{
            **config.__dict__,
            "coherence_eps": 0.2,
            "coherence_min_samples": 5,
            "coherence_pca_dimensions": None,
        })
        vector = scoring.terms(pool, tuned, partition=partition).coherence
        assert not np.array_equal(vector, ungated), (
            f"arm {name!r} declares coherence {config.coherence_method!r} but "
            "scores identically to no gate at all"
        )
        checked += 1
    assert checked >= 2, "the ladder must contain gated arms to guard"


def test_admissible_combination_keeps_uncertainty_and_multiplicative_drops_it(blobs):
    """The two multiplicative forms differ by exactly one thing, on purpose.

    ``multiplicative`` is ``A * (1 + semantic)`` -- the form the committed
    2026-08 arms measured, kept bit-identical. ``admissible`` is
    ``A * (U + semantic)``, the decomposition the 2026-09-02 protocol freezes.
    """

    n = blobs.shape[0]
    terms = scoring.Terms(
        uncertainty=np.full(n, 0.25, dtype=np.float32),
        diversity=np.zeros(n, dtype=np.float32),
        rarity=np.zeros(n, dtype=np.float32),
        coherence=np.ones(n, dtype=np.float32),
        partition=clustering.fit(blobs, n_clusters=3, seed=0),
        prior=np.full(n, 2.0, dtype=np.float32),
    )

    terms.config = scoring.ScoreConfig(combination="multiplicative")
    assert np.allclose(terms.combine(), 2.0)          # 2 * (1 + 0)

    terms.config = scoring.ScoreConfig(combination="admissible")
    assert np.allclose(terms.combine(), 0.5)          # 2 * (0.25 + 0)


def test_the_ladder_adds_one_term_per_rung():
    """Each rung differs from the one above it in as few fields as possible."""

    rungs = ("a_u", "a_u_d", "a_u_d_r", "a_u_d_rc", "a_u_d_rc_batch")
    for lower, upper in itertools.pairwise(rungs):
        a = selection.ARMS_V2[lower].__dict__
        b = selection.ARMS_V2[upper].__dict__
        changed = {k for k in a if a[k] != b[k]} - {"name"}
        assert len(changed) <= 2, f"{lower} -> {upper} changes {changed}"

    # and the scope control differs from its sibling in the scope alone
    a = selection.ARMS_V2["a_u_d_rc"].__dict__
    b = selection.ARMS_V2["a_u_d_rc_fullpool"].__dict__
    assert {k for k in a if a[k] != b[k]} - {"name"} == {"coherence_admissible_share"}
