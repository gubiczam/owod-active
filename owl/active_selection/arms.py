"""The five arms of Benchmark V1, and what each one is a control for.

Every arm is a function of detector output and of what has already been
labelled. None of them can reach an annotation of an unbought region:
:func:`select` is handed a :class:`~owl.active_selection.population.Population`
and a cost function, and the cost function reads only object *counts* of images —
never a class, never a box. ``tests/test_active_selection.py`` asserts that
calling any arm on a pool whose oracle is absent still works.

======================  ==================================================
arm                     what it is for
======================  ==================================================
``random``              the reference every active method must beat.
``entropy``             the standard uncertainty baseline. Normalised Shannon
                        entropy of PROB's class posterior. Measured weak at
                        finding unknown objects in Method V3 (36 objects
                        against admissibility's 150) and kept anyway, because
                        it is the baseline the literature and the supervisor
                        both expect.
``admissibility``       ``A(x) = objectness(x) * sqrt(area(x))``. The
                        learning-free prior that has beaten every semantic
                        score this project has built, and therefore the bar.
                        Identical to ``owl.selection.ARMS['objectness']``,
                        whose name is a misnomer: raw objectness on its own is
                        a different and degenerate ranking (Spearman 0.28
                        against ``A``, top-600 Jaccard 0.000, and 2 real
                        objects in its first 600 picks against ``A``'s 284),
                        which is why it does not get a trajectory.
``coreset``             k-center greedy over the **whole** deduplicated pool in
                        frozen DINOv2 space. The recognisable core-set
                        baseline, and the ungated control for the proposed
                        method.
``proposed``            the same traversal restricted to the admissible subset
                        ``G``. One variable away from ``coreset``: the gate.
======================  ==================================================

**The proposed method, stated once.** Gate on object-likeness, then cover the
semantic space of what is not yet labelled:

    1. deduplicate the pool (per-image NMS at IoU 0.60, ordered by ``A``);
    2. keep the top 30% by ``A`` — the frozen admissibility share;
    3. embed those crops with frozen DINOv2 ViT-B/14, the Method V2 crop;
    4. farthest-first traversal in that space, initialised against the labelled
       reference (the balanced task-1 reference, plus every image bought so
       far), where opening an image covers everything annotated on it.

It carries **no** free parameter. That is deliberate: ``lambda``, ``gamma`` and
``mu`` of the earlier additive score each needed a number, and any number chosen
after seeing a detector endpoint would have made the result a tuned one. It also
uses each representation for the thing it was measured to be good at — PROB's
objectness for object-versus-background, which DINOv2 was measured *not* to
separate (``D_NO_GO``), and DINOv2 for semantic relations among real
object-like candidates, which is what it was measured to improve.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from owl import scoring
from owl.active_selection import allocation as allocation_module
from owl.active_selection import budget as ledger_module
from owl.active_selection import coverage as coverage_module
from owl.active_selection import population as population_module
from owl.active_selection import research_score
from owl.active_selection.population import Population


@dataclass(frozen=True)
class Arm:
    """One registered selector."""

    name: str
    kind: str                 # 'ranking' | 'coverage'
    needs_semantic: bool      # requires DINOv2 features for this task's pool
    gated: bool               # restricted to the admissible subset G
    reference_aware: bool     # consults what has already been labelled
    description: str

    #: Restrict further to the candidates whose normalised entropy is at or
    #: above the **median of the gated population**. An explicit new design
    #: choice, introduced by Proposed-v2; every earlier arm leaves it off and is
    #: therefore bit-identical to what it was measured as.
    informative: bool = False

    #: What "already labelled" means for a coverage arm.
    #:
    #: ``labelled``    the balanced task-1 reference plus everything this
    #:                 trajectory has bought. What ``proposed`` and ``coreset``
    #:                 were measured with.
    #: ``trajectory``  **only** what this trajectory has bought. Proposed-v2,
    #:                 which stops using DINOv2 as a distance-to-REF-T1 novelty
    #:                 score at all.
    reference_scope: str = "labelled"

    #: For a ``research`` arm: which configuration of
    #: ``s(x) = U + λ·D + γ·w·coh`` it is. ``None`` for every other kind, so no
    #: earlier arm's behaviour can be changed by this field existing.
    score_spec: research_score.ScoreSpec | None = None

    @property
    def slug(self) -> str:
        return self.name


ARMS: dict[str, Arm] = {
    "random": Arm(
        name="random", kind="ranking", needs_semantic=False, gated=False,
        reference_aware=False,
        description="uniform draw over the deduplicated pool",
    ),
    "entropy": Arm(
        name="entropy", kind="ranking", needs_semantic=False, gated=False,
        reference_aware=False,
        description="normalised Shannon entropy of PROB's class posterior",
    ),
    "admissibility": Arm(
        name="admissibility", kind="ranking", needs_semantic=False, gated=False,
        reference_aware=False,
        description="A(x) = objectness * sqrt(area), used raw",
    ),
    "coreset": Arm(
        name="coreset", kind="coverage", needs_semantic=True, gated=False,
        reference_aware=True,
        description="k-center greedy in frozen DINOv2 space, ungated",
    ),
    "proposed": Arm(
        name="proposed", kind="coverage", needs_semantic=True, gated=True,
        reference_aware=True,
        description="A-gated k-center greedy in frozen DINOv2 space",
    ),
    # ------------------------------------------------------------ v2, 2026-09-04
    # Designed AFTER inspecting Proposed-v1's seed-0 endpoints. It is
    # development-seed-informed and is NOT pre-registered; the protocol and the
    # research log say so, and so must any table that reports it.
    #
    # Informativeness first, diversity second. v1 maximised semantic coverage
    # and, on seed 0, produced mean new_class_AP50 = 0.00 with the best
    # U_Recall of any arm: it bought breadth and no depth, because
    # farthest-first leaves a region once that region is covered and so caps
    # per-class multiplicity by construction. v2 keeps the traversal but
    # demotes it to redundancy removal inside an informative subset, and stops
    # measuring distance to REF-T1 — a signal Method V2 already froze as
    # ``D_NO_GO``.
    "proposed_v2": Arm(
        name="proposed_v2", kind="coverage", needs_semantic=True, gated=True,
        reference_aware=True, informative=True, reference_scope="trajectory",
        description="A-gated, above-median-U, DINOv2 farthest-first for "
                    "redundancy removal only",
    ),
    # ------------------------------------------- 2026-09-06, the D/R/C allocator
    # Frozen by docs/distribution_aware_decision_memo_2026-09-06.md. Like
    # `proposed_v2` it is development-seed-informed and NOT pre-registered, and
    # every table that reports it must say so.
    #
    # It is a *ranking* arm that reads semantic features, which no earlier arm
    # was: the plan's `R` is an allocation over clusters, and an allocation
    # produces an order, not a traversal. `owl.active_selection.allocation`
    # holds the algorithm and the reasons for each of its parts.
    "distribution_aware_v1": Arm(
        name="distribution_aware_v1", kind="ranking", needs_semantic=True,
        gated=True, reference_aware=True, reference_scope="labelled",
        description="A-gated HDBSCAN; coherence gate, cluster rarity quota "
                    "against the labelled reference, entropy within cluster",
    ),
    # ------------------------------------------ 2026-09-06, the iterative form
    # Frozen by docs/iterative_decision_memo_2026-09-06.md. The ONLY scientific
    # difference from `distribution_aware_v1` is that the clustering, the rarity
    # and the quota are recomputed after each annotation round against a
    # reference that has grown by what the round bought.
    #
    # It exists because v1 tested a degenerate case of the specification: `R` is
    # defined against what is already labelled, and one-shot froze that for a
    # whole task. Rounds were on the register from the 2026-08-25 consultation,
    # before v1's NO-GO, so this is completing a pre-registration rather than
    # reacting to a result.
    "distribution_aware_iterative_v1": Arm(
        name="distribution_aware_iterative_v1", kind="iterative",
        needs_semantic=True, gated=True, reference_aware=True,
        reference_scope="labelled",
        description="distribution_aware_v1 with the reference, clusters, rarity "
                    "and quota recomputed after each annotation round",
    ),
    # The mandatory control. The ceiling audit found that simply opening cheap
    # images beats a perfect-cluster stratified allocator on raw tail counts, so
    # without this arm a win by the allocator is not attributable to
    # distribution-awareness. Label-free: it reads `A` and nothing else.
    "cost_aware": Arm(
        name="cost_aware", kind="ranking", needs_semantic=False, gated=True,
        reference_aware=False,
        description="A-gated, images ordered by descending sum-A over G — a "
                    "label-free cheap-image preference, no clustering",
    ),
    # ------------------------------------------ V2, the research score itself
    # Frozen by docs/full_owod_v2_protocol.md before the V2 chain ran.
    #
    # These are the first arms on the GPU path that compute the research plan's
    # own equation, `s(x) = U + λ·D + γ·w·coh`. Until now that equation existed
    # only in `owl.scoring`, which runs on the frozen CPU pool and never trains
    # a detector; the arms above are a different method family. See
    # `owl.active_selection.research_score` for every term.
    #
    # They are listed in `benchmark.DEVELOPMENT_SEED_INFORMED` and reported as
    # not-pre-registered, and the label is deliberately conservative rather than
    # exact: their *design* comes from the research plan and the 2026-08-25
    # consultation, and no V1 endpoint informed a single term of it — but they
    # were added after V1's seed-0 numbers existed, and the repository's
    # convention is that anything added after results exist carries the label.
    # `docs/full_owod_v2_protocol.md` is their pre-registration for V2.
    "research_v2": Arm(
        name="research_v2", kind="research", needs_semantic=True, gated=True,
        reference_aware=True, reference_scope="labelled",
        score_spec=research_score.ScoreSpec(
            diversity_mode="combined", coherence_mode="dbscan_binary",
            rarity_mode="cluster_rarity",
        ),
        description="U + 0.2*(labelled novelty + intra-batch)/2 + 0.5*rarity*"
                    "DBSCAN gate, A-gated, recomputed every round",
    ),
    #: The plan's equation as it is literally written in the PDF: `D` as
    #: distance to what is labelled and `coh` as the inverse k-th neighbour
    #: distance. The one variable between this and `research_v2` is the pair of
    #: 2026-08-25 redesigns, so the contrast measures the consultation.
    "research_v2_plan": Arm(
        name="research_v2_plan", kind="research", needs_semantic=True, gated=True,
        reference_aware=True, reference_scope="labelled",
        score_spec=research_score.ScoreSpec(
            diversity_mode="labeled_novelty", coherence_mode="continuous",
            rarity_mode="cluster_rarity",
        ),
        description="the plan's equation as written: labelled-novelty D, "
                    "continuous coherence",
    ),
    #: Gate ablation. `coh ≡ 1` — not the same as γ=0, because the rarity weight
    #: still acts. It is what separates "the gate helped" from "the weight helped".
    "research_v2_no_gate": Arm(
        name="research_v2_no_gate", kind="research", needs_semantic=True,
        gated=True, reference_aware=True, reference_scope="labelled",
        score_spec=research_score.ScoreSpec(
            diversity_mode="combined", coherence_mode="none",
            rarity_mode="cluster_rarity",
        ),
        description="research_v2 with the coherence gate held open",
    ),
    #: The two halves of `D`, each alone, so the combined term is decomposable.
    "research_v2_labeled_only": Arm(
        name="research_v2_labeled_only", kind="research", needs_semantic=True,
        gated=True, reference_aware=True, reference_scope="labelled",
        score_spec=research_score.ScoreSpec(
            diversity_mode="labeled_novelty", coherence_mode="dbscan_binary",
            rarity_mode="cluster_rarity",
        ),
        description="research_v2 with D = labelled novelty only",
    ),
    "research_v2_batch_only": Arm(
        name="research_v2_batch_only", kind="research", needs_semantic=True,
        gated=True, reference_aware=True, reference_scope="labelled",
        score_spec=research_score.ScoreSpec(
            diversity_mode="batch_diversity", coherence_mode="dbscan_binary",
            rarity_mode="cluster_rarity",
        ),
        description="research_v2 with D = intra-batch diversity only",
    ),
}

#: Execution priority, fixed before the first trajectory ran. A session that
#: runs out of runtime completes a prefix of this list, so the arms that survive
#: a short session are the ones the primary contrast needs — the proposed method,
#: the bar it must clear, and the reference. It is **not** a licence to drop an
#: arm because of what its numbers turned out to be; see the protocol's
#: "stopping rules" section for the only reasons an arm may be abandoned.
#: ``proposed_v2`` is **appended**, not inserted. It was designed after seeing
#: seed-0 results, so it may not displace a baseline in the execution order; a
#: short session completes the pre-registered prefix first and reaches v2 last.
#: ``cost_aware`` and ``distribution_aware_v1`` are **appended** for the same
#: reason ``proposed_v2`` was: both were designed after seeing seed-0 results and
#: may not displace a pre-registered baseline in the execution order.
#: ``research_*`` are appended for the same reason: they were added after V1's
#: seed-0 numbers existed, so they may not displace a pre-registered baseline in
#: the execution order even though the V2 protocol pre-registers them.
ORDER: tuple[str, ...] = (
    "random", "admissibility", "proposed", "entropy", "coreset", "proposed_v2",
    "cost_aware", "distribution_aware_v1", "distribution_aware_iterative_v1",
    "research_v2", "research_v2_plan", "research_v2_no_gate",
    "research_v2_labeled_only", "research_v2_batch_only",
)


#: The ranking arms whose order comes from an allocation rather than a score.
#: Membership is what routes an arm away from the frozen `ranking()` path, so an
#: arm that is not in here cannot have its measured behaviour changed by this
#: module's 2026-09-06 additions.
ALLOCATED: frozenset[str] = frozenset({"distribution_aware_v1", "cost_aware"})

#: The arm whose selection is a loop over rounds rather than one ranking. It
#: needs its own branch in :func:`select` because the ledger is consulted
#: *between* recomputations, not once at the end.
ITERATIVE: frozenset[str] = frozenset({"distribution_aware_iterative_v1"})

#: The arms that compute the research plan's own equation. Membership routes an
#: arm to :mod:`owl.active_selection.research_score`, so an arm outside this set
#: cannot have its measured behaviour changed by that module existing.
RESEARCH: frozenset[str] = frozenset(
    name for name, spec in ARMS.items() if spec.kind == "research"
)


class ArmError(ValueError):
    """Raised when an arm is asked for something it was not given."""


@dataclass(frozen=True)
class ArmSelection:
    """What one arm bought at one task."""

    arm: str
    images: tuple[str, ...]
    anchors: tuple[int, ...]
    row: dict[str, object]
    #: Positions on the opened images — everything full-image labelling paid
    #: for, not only the region that triggered the purchase. The coverage arms
    #: carry this forward as next task's labelled reference.
    covered: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    #: Candidate-level trail, one row per taken candidate: ``U``, ``D_labeled``,
    #: ``D_batch``, ``D``, ``w``, ``coh``, cluster id and core/border/noise
    #: status. Only a ``research`` arm fills this; everything else leaves it
    #: empty, which is what keeps the earlier arms' rows byte-identical.
    picks: tuple[dict, ...] = ()

    #: The proposal box that opened each image, aligned with :attr:`images`, in
    #: normalised ``cxcywh``. A per-box annotation policy needs to know *which*
    #: region the answer was about; only :func:`ArmSelection.anchors` knew that,
    #: and those are positions in the arm's own population, which the runner
    #: cannot resolve. Filled by
    #: :func:`owl.active_selection.benchmark.make_selector`.
    anchor_boxes: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.float32))

    def __len__(self) -> int:
        return len(self.images)


def ranking(
    arm: str, pool: Population, *, seed: int
) -> np.ndarray:
    """The order a static arm would consult positions in.

    Identical to what :func:`owl.selection.select` produces for the
    corresponding registered config — pinned by
    ``test_ranking_matches_the_committed_selector`` — but computed without
    fitting a k-means partition, which the additive path needs for terms this
    benchmark weights at zero and which costs minutes per task.
    """

    candidates = pool.candidates
    n = len(candidates)
    if arm == "random":
        generator = np.random.default_rng(seed)
        return generator.choice(n, size=n, replace=False).astype(np.int64)
    if arm == "entropy":
        score = scoring.rank_normalise(scoring.uncertainty(candidates, "entropy"))
    elif arm == "admissibility":
        score = scoring.rank_normalise(scoring.admissibility(candidates))
    else:
        raise ArmError(f"{arm!r} is not a ranking arm; kinds are {sorted(ARMS)}.")
    return np.argsort(-score, kind="mergesort").astype(np.int64)


def allocated_ranking(
    arm: str,
    pool: Population,
    index: np.ndarray,
    *,
    cost_of: Callable[[str], int],
    answer_budget: int,
    semantic: np.ndarray | None,
    reference: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    """Pool positions for the two arms frozen on 2026-09-06.

    Both are *ranking* arms that select inside the admissible subset ``G``, so
    the order they build is over ``index`` and is mapped back to pool positions
    before the ledger sees it. Neither reads an oracle class, an oracle box or a
    future declaration; ``cost_of`` is the same annotation cost function the
    ledger hands every arm.
    """

    if arm == "cost_aware":
        order, diagnostics = allocation_module.cost_aware_order(
            pool.admissibility[index], pool.candidates.image_ids[index],
        )
        return index[order], {"selector_detail": "sum_A_per_image"} | diagnostics

    if arm != "distribution_aware_v1":
        raise ArmError(f"{arm!r} has no allocated ranking")

    features = np.asarray(semantic, dtype=np.float32)
    if features.shape[0] != index.size:
        raise ArmError(
            f"semantic has {features.shape[0]} rows and arm {arm!r} selects over "
            f"{index.size} candidates (the admissible subset G); the export does "
            "not describe what this arm ranks."
        )
    entropy = scoring.uncertainty(pool.candidates, "entropy")[index]
    result = allocation_module.distribution_aware_order(
        features,
        entropy=entropy,
        image_ids=pool.candidates.image_ids[index],
        cost_of=cost_of,
        budget=int(answer_budget),
        reference=reference,
    )
    return index[result.order], {"selector_detail": "cluster_quota"} | result.diagnostics


def ranked_positions(arm: str, pool: Population) -> np.ndarray:
    """The pool positions ``arm`` may choose from, in ascending order.

    **One definition, two callers**, and that is the point:
    :func:`owl.active_selection.benchmark.make_selector` uses it to decide which
    crops to embed and :func:`select` uses it to decide what the traversal may
    take. Computing the subset twice is how a feature matrix comes to describe a
    different population from the one being selected over — the failure the
    export fingerprint exists to catch, and which is better not created.

    Two restrictions, applied in this order:

    1. the **admissibility gate** — the top
       :data:`owl.active_selection.population.ADMISSIBLE_SHARE` by ``A``;
    2. for an ``informative`` arm, ``U >= median(U)`` **within what step 1
       kept**. The median is taken over the gated population, not over the
       whole pool: taking it over the pool would keep whatever share of the gate
       happened to sit above the pool's middle, which is not a fixed quantity
       and not the stated rule.
    """

    spec = ARMS[arm]
    keep = (
        np.asarray(pool.gate, dtype=bool).copy()
        if spec.gated
        else np.ones(len(pool), dtype=bool)
    )
    if spec.informative:
        entropy = scoring.uncertainty(pool.candidates, "entropy")
        threshold = float(np.median(entropy[keep]))
        keep &= entropy >= threshold
    return np.flatnonzero(keep)


def ranked_share(arm: str) -> float:
    """The fraction of ``P_nms`` an arm ranks, for cost estimates only.

    Exact for the gate (a rank fraction) and nominal for the median filter,
    which keeps half of the gate up to ties at the median.
    """

    spec = ARMS[arm]
    share = population_module.ADMISSIBLE_SHARE if spec.gated else 1.0
    return share * (0.5 if spec.informative else 1.0)


def select(
    arm: str,
    pool: Population,
    *,
    cost_of: Callable[[str], int],
    answer_budget: int,
    seed: int,
    semantic: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    excluded_images: frozenset[str] = frozenset(),
    rounds: int = 1,
) -> ArmSelection:
    """Spend one task's annotation budget with ``arm``.

    ``semantic`` is the DINOv2 feature matrix for **this task's** deduplicated
    pool, in its row order. Required by the coverage arms and refused for the
    others, so a run cannot quietly pay for an export nothing reads.
    """

    if arm not in ARMS:
        raise ArmError(f"Unknown arm {arm!r}; registered: {sorted(ARMS)}.")
    spec = ARMS[arm]
    if spec.needs_semantic and semantic is None:
        raise ArmError(
            f"Arm {arm!r} selects in DINOv2 space and no features were supplied. "
            "Run the semantic export for this task's pool first."
        )
    if not spec.needs_semantic and semantic is not None:
        raise ArmError(
            f"Arm {arm!r} does not read semantic features, but some were passed. "
            "Paying for an export nothing consults would misreport the cost of "
            "this arm."
        )

    if spec.kind == "research":
        features = np.asarray(semantic, dtype=np.float32)
        index = ranked_positions(arm, pool)
        if features.shape[0] != index.size:
            raise ArmError(
                f"semantic has {features.shape[0]} rows and arm {arm!r} selects "
                f"over {index.size} candidates (the admissible subset G); the "
                "export does not describe what this arm ranks."
            )
        result = research_score.select(
            features,
            entropy=scoring.uncertainty(pool.candidates, "entropy")[index],
            image_ids=pool.candidates.image_ids[index],
            cost_of=cost_of, budget=int(answer_budget), spec=spec.score_spec,
            rounds=int(rounds), reference=reference,
            excluded_images=frozenset(excluded_images),
        )
        covered = np.zeros(len(pool), dtype=bool)
        covered[index[result.covered]] = True
        row = (
            {"arm": arm, "selector": spec.kind, "rounds": int(rounds),
             "selector_detail": "research_score"}
            | result.diagnostics
            | {"round_log": json.dumps(list(result.rounds))}
        )
        return ArmSelection(
            arm=arm, images=result.images,
            anchors=tuple(int(index[a]) for a in result.anchors),
            row=row, covered=covered, picks=result.picks,
        )

    if spec.kind == "iterative":
        features = np.asarray(semantic, dtype=np.float32)
        index = ranked_positions(arm, pool)
        if features.shape[0] != index.size:
            raise ArmError(
                f"semantic has {features.shape[0]} rows and arm {arm!r} selects "
                f"over {index.size} candidates (the admissible subset G); the "
                "export does not describe what this arm ranks."
            )
        result = allocation_module.distribution_aware_iterative_order(
            features,
            entropy=scoring.uncertainty(pool.candidates, "entropy")[index],
            image_ids=pool.candidates.image_ids[index],
            cost_of=cost_of, budget=int(answer_budget), rounds=int(rounds),
            reference=reference, excluded_images=frozenset(excluded_images),
        )
        covered = np.zeros(len(pool), dtype=bool)
        covered[index[result.covered]] = True
        row = (
            {"arm": arm, "selector": spec.kind, "rounds": int(rounds),
             "selector_detail": "cluster_quota_iterative"}
            | result.diagnostics
            # The same ledger columns the ranking arms report, so one CSV
            # describes both and `diagnostic.evaluate` needs no special case.
            | {"answer_budget": int(answer_budget),
               "answers_unspent": int(answer_budget) - int(result.diagnostics["answers_spent"]),
               "answers_per_image": round(
                   int(result.diagnostics["answers_spent"])
                   / max(len(result.images), 1), 3),
               "round_log": json.dumps(list(result.rounds))}
        )
        return ArmSelection(
            arm=arm, images=result.images,
            anchors=tuple(int(index[a]) for a in result.anchors),
            row=row, covered=covered,
        )

    if spec.kind == "ranking":
        # The three pre-registered ranking arms keep the exact code path they
        # were measured on. The two frozen on 2026-09-06 select inside `G` and
        # build their order from an allocation, so they take the branch below;
        # nothing about `random`, `entropy` or `admissibility` moves.
        detail: dict[str, object] = {}
        if arm in ALLOCATED:
            index = ranked_positions(arm, pool)
            order, detail = allocated_ranking(
                arm, pool, index, cost_of=cost_of, answer_budget=answer_budget,
                semantic=semantic, reference=reference,
            )
        else:
            order = ranking(arm, pool, seed=seed)
        # The round schedule is applied to every arm, not only to the one that
        # recomputes. For a static ranking it is provably a no-op under the
        # carry rule, and asserting that on the real code path is what lets the
        # completed one-shot rows stand as the 6-round rows.
        spend = ledger_module.spend_ranking_in_rounds(
            order, pool.candidates.image_ids, cost_of,
            budget=answer_budget, rounds=int(rounds),
            excluded_images=excluded_images,
        )
        row: dict[str, object] = (
            {"arm": arm, "selector": spec.kind} | spend.summary() | detail
        )
        covered = np.isin(
            np.asarray(pool.candidates.image_ids, dtype=str),
            np.asarray(spend.images, dtype=str),
        )
        return ArmSelection(
            arm=arm, images=spend.images, anchors=spend.anchors,
            row=row, covered=covered,
        )

    features = np.asarray(semantic, dtype=np.float32)
    # A gated arm never selects outside G and never covers outside it either, so
    # it is handed features for G alone — cheaper, and it is what the method
    # means: cover the semantic space of the *object-like* candidates. An
    # informative arm narrows that again to the upper half by entropy. An ungated
    # arm gets the whole deduplicated pool, which is the point of the control.
    index = ranked_positions(arm, pool)
    if features.shape[0] != index.size:
        raise ArmError(
            f"semantic has {features.shape[0]} rows and arm {arm!r} selects over "
            f"{index.size} candidates ("
            + ("the admissible subset G" if spec.gated else "the deduplicated pool")
            + (", narrowed to U >= median(U) within it" if spec.informative else "")
            + "); the export does not describe what this arm ranks."
        )
    result = coverage_module.kcenter_greedy(
        features, pool.candidates.image_ids[index],
        cost_of=cost_of, budget=answer_budget,
        reference=reference,
        excluded_images=excluded_images,
        tie_break=pool.admissibility[index],
    )
    covered = np.zeros(len(pool), dtype=bool)
    covered[index[result.covered]] = True
    row = (
        {"arm": arm, "selector": spec.kind, "gated": spec.gated}
        | result.summary()
        | {"coverage_stop": result.diagnostics["stopped_because"],
           "coverage_candidates": int(index.size)}
    )
    return ArmSelection(
        arm=arm,
        images=result.images,
        # positions in the *pool*, not in the arm's own restricted view
        anchors=tuple(int(index[a]) for a in result.anchors),
        row=row, covered=covered,
    )
