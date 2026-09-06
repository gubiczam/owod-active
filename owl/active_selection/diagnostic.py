"""The frozen candidate-side GO/NO-GO for ``distribution_aware_v1``.

Criteria, thresholds, aggregation and the decision rule are frozen by
``docs/distribution_aware_decision_memo_2026-09-06.md`` §6, written before any
oracle diagnostic for this method existed. **After an outcome is visible none of
it may be changed** — not a threshold, not an aggregation, not a definition, not
which criteria are required.

This module holds two things and nothing else: the frozen configuration, and the
arithmetic that turns measured rows into a verdict. The selector never imports
it, so nothing here can reach acquisition.

**Definitions pinned here rather than left to the caller**, because a phrase in a
memo is not an implementation:

``background_share``
    the share of **opened images that carry no annotated object at all**. The
    memo says "background share of opened images"; at image granularity under
    full-image labelling that is the barren-image share, and it is what the
    committed candidate index (per-image class counts) can support. An arm that
    opens empty images is spending answers on background.
``tail supply per image opened``
    declared-class objects the ledger **holds at declaration** — bought at this
    task or banked from an earlier one — divided by images opened, averaged over
    ``t3`` and ``t4``. Per task this ratio is noisy (the ceiling audit measured
    1.08-1.68); averaged over the two tail tasks it was stable (1.36, 1.42),
    which is why the memo aggregates before comparing.
``acquired class breadth``
    distinct classes not known at ``t1`` among the opened images' objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

#: The two tail tasks. ``t2`` declares a head class and the memo treats it
#: separately, as a no-collapse check rather than as evidence for the method.
TAIL_TASKS: tuple[str, ...] = ("t3", "t4")

#: Seeds the diagnostic is measured at.
DIAGNOSTIC_SEEDS: tuple[int, ...] = (0, 1)

#: The arms compared. ``entropy`` is the strong baseline; ``cost_aware`` is the
#: mandatory control without which a win is not attributable to
#: distribution-awareness.
DIAGNOSTIC_ARMS: tuple[str, ...] = ("entropy", "cost_aware", "distribution_aware_v1")


@dataclass(frozen=True)
class Gate:
    """One frozen criterion."""

    key: str
    statement: str
    threshold: float
    #: ``per_seed`` must hold at every seed; ``mean`` at the mean over seeds.
    aggregation: str
    #: Which arm the measurement is compared against, if any.
    against: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


#: **Frozen 2026-09-06, before any oracle diagnostic for this method was run.**
GATES: tuple[Gate, ...] = (
    Gate(
        key="jaccard_vs_entropy",
        statement="Jaccard of opened images against entropy is below 0.60 — "
                  "above it the arms cannot produce a distinguishable detector "
                  "outcome",
        threshold=0.60, aggregation="per_seed", against="entropy",
    ),
    Gate(
        key="tail_per_image_vs_entropy",
        statement="tail supply per image opened, mean over t3 and t4, is at "
                  "least 1.25x entropy",
        threshold=1.25, aggregation="per_seed", against="entropy",
    ),
    Gate(
        key="tail_per_image_vs_cost_aware",
        statement="tail supply per image opened, mean over t3 and t4, is at "
                  "least 1.25x cost_aware — the falsifier, promoted to a gate",
        threshold=1.25, aggregation="per_seed", against="cost_aware",
    ),
    Gate(
        key="background_share_vs_entropy",
        statement="background share of opened images is no more than 10 "
                  "percentage points above entropy",
        threshold=0.10, aggregation="mean", against="entropy",
    ),
    Gate(
        key="breadth_vs_entropy",
        statement="distinct acquired classes not known at t1 is at least 0.75x "
                  "entropy",
        threshold=0.75, aggregation="mean", against="entropy",
    ),
    Gate(
        key="t2_not_collapsed",
        statement="t2 held-per-image is at least 0.75x entropy — a tail gain "
                  "bought by abandoning the head task is not a win on a mean "
                  "over three tasks",
        threshold=0.75, aggregation="mean", against="entropy",
    ),
)

#: Every gate is required. The memo's decision rule: any failure is NO-GO, and
#: the method is preserved as a negative candidate-side result without tuning.
DECISION = (
    "All six gates must hold. Any failure -> NO-GO: record the negative, do not "
    "train, do not tune, do not re-choose the clusterer, do not soften a "
    "threshold. All pass -> GO: freeze and run the minimal downstream "
    "experiment."
)


def configuration() -> dict[str, object]:
    """The machine-readable frozen gate configuration, printed before the run."""

    return {
        "frozen": "docs/distribution_aware_decision_memo_2026-09-06.md",
        "frozen_on": "2026-09-06",
        "arms": list(DIAGNOSTIC_ARMS),
        "seeds": list(DIAGNOSTIC_SEEDS),
        "tail_tasks": list(TAIL_TASKS),
        "primary_endpoint": "held_at_declaration / images_opened, mean over t3 and t4",
        "gates": [gate.as_dict() for gate in GATES],
        "decision": DECISION,
    }


# ------------------------------------------------------------- measurement ---


def _by(rows: Sequence[Mapping[str, object]], arm: str, seed: int) -> list[dict]:
    return [dict(r) for r in rows if r["arm"] == arm and int(r["seed"]) == seed]


def tail_per_image(rows: Sequence[Mapping[str, object]], arm: str, seed: int) -> float:
    """Mean over t3 and t4 of held-at-declaration per image opened."""

    picked = {r["task"]: r for r in _by(rows, arm, seed)}
    values = []
    for task in TAIL_TASKS:
        row = picked[task]
        values.append(float(row["held_at_declaration"]) / max(int(row["images_opened"]), 1))
    return sum(values) / len(values)


def mean_over_tasks(rows, arm: str, seed: int, key: str) -> float:
    picked = _by(rows, arm, seed)
    return sum(float(r[key]) for r in picked) / max(len(picked), 1)


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(map(str, left)), set(map(str, right))
    union = a | b
    return len(a & b) / len(union) if union else 0.0


@dataclass(frozen=True)
class Outcome:
    """One gate's measured value and verdict."""

    gate: Gate
    measured: dict[int, float]
    aggregate: float
    passed: bool
    detail: str = ""

    def row(self) -> dict[str, object]:
        return {
            "gate": self.gate.key,
            "threshold": self.gate.threshold,
            "aggregation": self.gate.aggregation,
            "against": self.gate.against,
            "measured_per_seed": {str(k): round(v, 4) for k, v in self.measured.items()},
            "aggregate": round(self.aggregate, 4),
            "verdict": "PASS" if self.passed else "FAIL",
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Verdict:
    """The frozen decision, and every gate that produced it."""

    outcomes: tuple[Outcome, ...]
    go: bool
    failed: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": "GO" if self.go else "NO-GO",
            "failed": list(self.failed),
            "gates": [o.row() for o in self.outcomes],
        }


def _decide(gate: Gate, measured: dict[int, float]) -> Outcome:
    values = list(measured.values())
    aggregate = sum(values) / max(len(values), 1)
    if gate.aggregation == "per_seed":
        candidate = min(values) if gate.key != "jaccard_vs_entropy" else max(values)
    else:
        candidate = aggregate
    if gate.key == "jaccard_vs_entropy":
        passed = candidate < gate.threshold
    elif gate.key == "background_share_vs_entropy":
        passed = candidate <= gate.threshold
    else:
        passed = candidate >= gate.threshold
    return Outcome(gate=gate, measured=measured, aggregate=aggregate, passed=passed,
                   detail=f"decided on {gate.aggregation} value {candidate:.4f}")


def evaluate(
    rows: Sequence[Mapping[str, object]],
    opened: Mapping[tuple[str, int, str], Sequence[str]],
    *,
    method: str = "distribution_aware_v1",
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> Verdict:
    """Apply the frozen gates to measured rows. No thresholds are read from data."""

    measurements: dict[str, dict[int, float]] = {gate.key: {} for gate in GATES}
    for seed in seeds:
        mine = {task: opened[(method, seed, task)] for _, _, task in
                [k for k in opened if k[0] == method and k[1] == seed]}
        theirs = {task: opened[("entropy", seed, task)] for task in mine}
        pooled_mine = [i for task in sorted(mine) for i in mine[task]]
        pooled_theirs = [i for task in sorted(theirs) for i in theirs[task]]
        measurements["jaccard_vs_entropy"][seed] = jaccard(pooled_mine, pooled_theirs)

        mine_tail = tail_per_image(rows, method, seed)
        measurements["tail_per_image_vs_entropy"][seed] = (
            mine_tail / max(tail_per_image(rows, "entropy", seed), 1e-9)
        )
        measurements["tail_per_image_vs_cost_aware"][seed] = (
            mine_tail / max(tail_per_image(rows, "cost_aware", seed), 1e-9)
        )
        measurements["background_share_vs_entropy"][seed] = (
            mean_over_tasks(rows, method, seed, "background_share")
            - mean_over_tasks(rows, "entropy", seed, "background_share")
        )
        measurements["breadth_vs_entropy"][seed] = (
            mean_over_tasks(rows, method, seed, "acquired_class_breadth")
            / max(mean_over_tasks(rows, "entropy", seed, "acquired_class_breadth"), 1e-9)
        )
        t2_mine = next(r for r in _by(rows, method, seed) if r["task"] == "t2")
        t2_theirs = next(r for r in _by(rows, "entropy", seed) if r["task"] == "t2")
        measurements["t2_not_collapsed"][seed] = (
            (float(t2_mine["held_at_declaration"]) / max(int(t2_mine["images_opened"]), 1))
            / max(float(t2_theirs["held_at_declaration"])
                  / max(int(t2_theirs["images_opened"]), 1), 1e-9)
        )

    outcomes = tuple(_decide(gate, measurements[gate.key]) for gate in GATES)
    failed = tuple(o.gate.key for o in outcomes if not o.passed)
    return Verdict(outcomes=outcomes, go=not failed, failed=failed)
