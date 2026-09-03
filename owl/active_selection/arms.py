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

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from owl import scoring
from owl.active_selection import budget as ledger_module
from owl.active_selection import coverage as coverage_module
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
}

#: Execution priority, fixed before the first trajectory ran. A session that
#: runs out of runtime completes a prefix of this list, so the arms that survive
#: a short session are the ones the primary contrast needs — the proposed method,
#: the bar it must clear, and the reference. It is **not** a licence to drop an
#: arm because of what its numbers turned out to be; see the protocol's
#: "stopping rules" section for the only reasons an arm may be abandoned.
ORDER: tuple[str, ...] = ("random", "admissibility", "proposed", "entropy", "coreset")


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

    if spec.kind == "ranking":
        order = ranking(arm, pool, seed=seed)
        spend = ledger_module.spend_ranking(
            order, pool.candidates.image_ids, cost_of,
            budget=answer_budget, excluded_images=excluded_images,
        )
        row: dict[str, object] = {"arm": arm, "selector": spec.kind} | spend.summary()
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
    # it is handed features for G alone. That is not only cheaper — 24,000 crops
    # per task instead of 80,000 — it is what the method means: cover the
    # semantic space of the *object-like* candidates. An ungated arm gets the
    # whole deduplicated pool, which is the point of the control.
    index = np.flatnonzero(pool.gate) if spec.gated else np.arange(len(pool))
    if features.shape[0] != index.size:
        raise ArmError(
            f"semantic has {features.shape[0]} rows and arm {arm!r} selects over "
            f"{index.size} candidates ("
            + ("the admissible subset G" if spec.gated else "the deduplicated pool")
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
