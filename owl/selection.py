"""Spending the annotation budget: how many rounds, and which regions.

Two questions from the 2026-08-25 consultation live here.

**Point 7 — one shot or several.** Given 600 regions to label, do we score once
and take the top 600, or take the best 100, recompute, and repeat six times?
The second is expected to be better because diversity and rarity both update
with what has already been chosen, and it is more expensive. :func:`select`
takes ``rounds`` so ``600x1``, ``6x100`` and ``12x50`` are one parameter apart.

**Point 1b — batch diversity.** Novelty against the labelled pool stops the
selector re-buying what we already know. It does nothing about a batch that is
600 near-copies of each other. The greedy update in :func:`select` adds a term
that grows as a candidate resembles what this batch has already taken, in the
manner of k-means++ seeding. With ``rounds > 1`` some of this happens for free;
with ``mu_batch > 0`` it happens inside a round as well.

The budget is counted in **regions**, because that is what an annotator is
asked about. Converting regions into images -- and deciding what else on a
chosen image gets labelled -- is :mod:`owl.labelling`, deliberately separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from owl import scoring
from owl.proposals import Candidates
from owl.scoring import ScoreConfig, Terms

#: How many candidates per budget slot compete in the greedy batch-diversity
#: pass. Twenty is enough room for redundancy to reorder a batch and small
#: enough that the pass costs seconds rather than minutes.
SHORTLIST_FACTOR = 20


@dataclass
class Selection:
    """What the selector asks the oracle about."""

    indices: np.ndarray          # positions in the candidate pool, in pick order
    round_of: np.ndarray         # which round each pick belongs to
    scores: np.ndarray           # the score each pick had when it was taken
    config: ScoreConfig

    def __len__(self) -> int:
        return int(self.indices.size)

    def images(self, candidates: Candidates) -> np.ndarray:
        """The distinct images the annotator ends up opening."""
        return np.unique(candidates.image_ids[self.indices])


def select(
    candidates: Candidates,
    config: ScoreConfig,
    *,
    budget: int,
    rounds: int = 1,
    labelled_embeddings: np.ndarray | None = None,
    n_known: int = 19,
    exclude: np.ndarray | None = None,
    partition=None,
    precomputed: np.ndarray | None = None,
) -> Selection:
    """Spend ``budget`` regions over ``rounds`` recomputations.

    ``exclude`` is a boolean mask of proposals already paid for in earlier
    tasks; they are never asked about twice.

    Between rounds the labelled pool grows by what was just picked, so ``D``
    moves — that is the feedback loop the research plan draws in figure 2, and
    ``rounds=1`` is the control that switches it off.

    ``precomputed`` is a ranking supplied from outside instead of built from
    :func:`owl.scoring.terms`. It exists because a score can be a *product* of
    terms — ``A(x) * C(x)``, frozen in Method V3 — and :class:`ScoreConfig`
    describes only the additive-plus-admissibility family. Passing it is an
    explicit statement that the config's own terms are not used, so a config
    that still carries a non-zero weight is refused rather than silently
    ignored. The round structure, the exclusion of what has been bought, and the
    order of the picks are the same code path as every other arm.
    """

    n = len(candidates)
    if precomputed is not None:
        precomputed = np.asarray(precomputed, dtype=np.float64)
        if precomputed.shape != (n,):
            raise ValueError(
                f"precomputed has shape {precomputed.shape}, expected ({n},)."
            )
        if config.random:
            raise ValueError(
                "A random arm has no ranking; pass precomputed=None for it."
            )
        weighted = {
            "lambda_diversity": config.lambda_diversity,
            "gamma_rarity": config.gamma_rarity,
            "mu_batch": config.mu_batch,
        }
        active = {name: value for name, value in weighted.items() if value}
        if active:
            raise ValueError(
                "A precomputed ranking replaces the config's own terms, but this "
                f"config still weights {sorted(active)}. Zero them, or drop "
                "precomputed."
            )

    available = np.ones(n, dtype=bool)
    if exclude is not None:
        available &= ~np.asarray(exclude, dtype=bool)

    generator = np.random.default_rng(config.seed)
    reference = (
        np.zeros((0, candidates.embeddings.shape[1]), dtype=np.float32)
        if labelled_embeddings is None
        else np.asarray(labelled_embeddings, dtype=np.float32)
    )

    per_round = _split_budget(budget, rounds)
    picked: list[int] = []
    picked_round: list[int] = []
    picked_score: list[float] = []

    for round_index, quota in enumerate(per_round):
        quota = min(quota, int(available.sum()))
        if quota <= 0:
            break

        if config.random:
            choice = generator.choice(np.flatnonzero(available), size=quota, replace=False)
            chosen_scores = np.zeros(quota)
        elif precomputed is not None:
            pool = np.flatnonzero(available)
            choice = pool[np.argsort(-precomputed[pool], kind="mergesort")][:quota]
            chosen_scores = precomputed[choice]
        else:
            round_terms = scoring.terms(
                candidates,
                config,
                labelled_embeddings=reference,
                n_known=n_known,
                partition=partition,
            )
            choice, chosen_scores = _greedy(
                round_terms, candidates, available, quota, config
            )

        picked.extend(int(i) for i in choice)
        picked_round.extend([round_index] * len(choice))
        picked_score.extend(float(s) for s in chosen_scores)
        available[choice] = False
        reference = np.vstack([reference, candidates.embeddings[choice]])

    return Selection(
        indices=np.asarray(picked, dtype=np.int64),
        round_of=np.asarray(picked_round, dtype=np.int64),
        scores=np.asarray(picked_score, dtype=np.float64),
        config=config,
    )


def _split_budget(budget: int, rounds: int) -> list[int]:
    """Even split, remainder to the earliest rounds. ``600, 6 -> [100]*6``."""

    if rounds < 1:
        raise ValueError("rounds must be at least 1.")
    base, extra = divmod(int(budget), int(rounds))
    return [base + (1 if i < extra else 0) for i in range(rounds)]


def _greedy(
    round_terms: Terms,
    candidates: Candidates,
    available: np.ndarray,
    quota: int,
    config: ScoreConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Take ``quota`` candidates, optionally penalising within-batch redundancy."""

    base = round_terms.combine()
    pool = np.flatnonzero(available)

    if not config.mu_batch:
        order = pool[np.argsort(-base[pool], kind="mergesort")][:quota]
        return order, base[order]

    # k-means++ flavoured: each pick pushes down everything that looks like it.
    #
    # Only a shortlist competes. Redundancy can move a candidate down the
    # ranking but never up, so anything far below the cut cannot reach the
    # batch, and scoring the whole pool once per pick would cost 600 x 80,000
    # dot products for a result that is identical.
    shortlist = pool[np.argsort(-base[pool], kind="mergesort")][: quota * SHORTLIST_FACTOR]
    embeddings = candidates.embeddings[shortlist]
    shortlist_base = base[shortlist]
    similarity_to_batch = np.zeros(shortlist.size, dtype=np.float32)
    alive = np.ones(shortlist.size, dtype=bool)
    chosen: list[int] = []
    chosen_scores: list[float] = []
    for _ in range(quota):
        adjusted = shortlist_base - config.mu_batch * scoring.rank_normalise(similarity_to_batch)
        adjusted[~alive] = -np.inf
        local = int(np.argmax(adjusted))
        if not np.isfinite(adjusted[local]):
            break
        chosen.append(int(shortlist[local]))
        chosen_scores.append(float(adjusted[local]))
        alive[local] = False
        similarity_to_batch = np.maximum(similarity_to_batch, embeddings @ embeddings[local])
    return np.asarray(chosen, dtype=np.int64), np.asarray(chosen_scores)


# ------------------------------------------------------------------- arms ---

#: The registered arms. Every comparison in the notebook is one of these against
#: another, and each differs from ``plan`` in as few fields as possible.
ARMS: dict[str, ScoreConfig] = {
    "random": ScoreConfig(name="random", random=True),
    "entropy": ScoreConfig(
        name="entropy", lambda_diversity=0.0, gamma_rarity=0.0
    ),
    "objectness": ScoreConfig(
        name="objectness", uncertainty_method="objectness",
        lambda_diversity=0.0, gamma_rarity=0.0,
    ),
    "plan": ScoreConfig(
        name="plan", diversity_source="clusters", coherence_method="continuous",
    ),
    "consult": ScoreConfig(
        name="consult", diversity_source="labelled", coherence_method="binary",
    ),
    "consult_batch": ScoreConfig(
        name="consult_batch", diversity_source="labelled",
        coherence_method="binary", mu_batch=0.3,
    ),
    "consult_no_gate": ScoreConfig(
        name="consult_no_gate", diversity_source="labelled", coherence_method="off",
    ),
    "consult_shared_cluster": ScoreConfig(
        name="consult_shared_cluster", diversity_source="clusters",
        coherence_method="binary",
    ),
    # The synthesis: the free object-likeness prior as an admissibility factor,
    # with the consultation's terms deciding between the regions it admits.
    "prior_consult": ScoreConfig(
        name="prior_consult", combination="multiplicative",
        diversity_source="labelled", coherence_method="binary",
    ),
    "prior_consult_batch": ScoreConfig(
        name="prior_consult_batch", combination="multiplicative",
        diversity_source="labelled", coherence_method="binary", mu_batch=0.3,
    ),
}

# --------------------------------------------------------------- arms, V2 ---

#: The 2026-09-02 ablation ladder, frozen in
#: ``docs/supervisor_week_protocol_2026-09-02.md`` §A1.3.
#:
#: Separate from :data:`ARMS` on purpose. ``ARMS`` and every CSV under
#: ``data/results/`` that was produced from it stay untouched and reproducible;
#: three of those arms are known to be defective as *controls* and this dict is
#: where the repaired versions live:
#:
#: * ``consult`` and ``consult_no_gate`` in ``ARMS`` are bitwise identical,
#:   because ``coherence_method='binary'`` closes on 0 of 80,000 candidates. The
#:   gate-on / gate-off contrast here is ``a_u_d_r`` (off) against ``a_u_d_rc``
#:   (density), which is a real one variable apart.
#: * ``ARMS`` had no way to vary the *scope* a density gate is fitted on, which
#:   is the whole of H2. ``a_u_d_rc_fullpool`` is that control.
#: * ``combination='multiplicative'`` drops ``U``; ``'admissible'`` keeps it, per
#:   the frozen decomposition. Both are present, so the difference is measurable
#:   rather than assumed.
#:
#: Each rung adds exactly one term to the one above it, so the ladder reads as an
#: ablation and not as a set of unrelated configurations.
ARMS_V2: dict[str, ScoreConfig] = {
    # --- controls ---------------------------------------------------------
    # identical to ARMS['random'] / ['objectness'] / ['entropy']: reused rather
    # than redefined, so the V1 and V2 tables agree on their shared rows.
    # gamma=0 makes the coherence factor inert, so these declare 'off' instead
    # of inheriting the default 'binary' and implying a gate they never apply.
    "random": ScoreConfig(name="random", random=True, coherence_method="off"),
    "objectness": ScoreConfig(
        name="objectness", uncertainty_method="objectness",
        lambda_diversity=0.0, gamma_rarity=0.0, coherence_method="off",
    ),
    "entropy": ScoreConfig(
        name="entropy", lambda_diversity=0.0, gamma_rarity=0.0,
        coherence_method="off",
    ),
    "plan": ScoreConfig(
        name="plan", diversity_source="clusters", coherence_method="continuous",
    ),
    # --- the ladder: A * (U + lam*D_known + gam*R*C + mu*B) ---------------
    "a_u": ScoreConfig(                      # does U pay, once A admits?
        name="a_u", combination="admissible",
        lambda_diversity=0.0, gamma_rarity=0.0, coherence_method="off",
    ),
    "a_u_d": ScoreConfig(                    # + novelty against the labelled pool
        name="a_u_d", combination="admissible", diversity_source="labelled",
        gamma_rarity=0.0, coherence_method="off",
    ),
    "a_u_d_r": ScoreConfig(                  # + rarity, gate OFF -- the gate control
        name="a_u_d_r", combination="admissible", diversity_source="labelled",
        coherence_method="off",
    ),
    "a_u_d_rc": ScoreConfig(                 # + the repaired density gate
        name="a_u_d_rc", combination="admissible", diversity_source="labelled",
        coherence_method="density", coherence_admissible_share=0.30,
    ),
    "a_u_d_rc_batch": ScoreConfig(           # + within-batch diversity: full method
        name="a_u_d_rc_batch", combination="admissible", diversity_source="labelled",
        coherence_method="density", coherence_admissible_share=0.30, mu_batch=0.3,
    ),
    # --- H2's scope control: the same gate fitted on the whole pool -------
    "a_u_d_rc_fullpool": ScoreConfig(
        name="a_u_d_rc_fullpool", combination="admissible", diversity_source="labelled",
        coherence_method="density", coherence_admissible_share=1.0,
    ),
}

#: Rungs of the ladder in ablation order, for tables that must not be alphabetised.
LADDER_V2: tuple[str, ...] = (
    "random", "objectness", "entropy", "plan",
    "a_u", "a_u_d", "a_u_d_r", "a_u_d_rc", "a_u_d_rc_batch", "a_u_d_rc_fullpool",
)
