"""Full OWOD Active Selection Benchmark V1 — the selectors and the ledger.

Protocol: ``docs/full_owod_active_benchmark_v1_protocol_2026-09-03.md``.

The package holds the four things Benchmark V1 needs that the repository did not
already have, and nothing else. Everything about the *chain* — the per-task
detector pass, the replay memory, the fine-tune, PROB's evaluator, resume — is
:func:`owl.runner.run_chain`, which has already run on a GPU for six tasks and is
not reimplemented here.

:mod:`~owl.active_selection.population`
    the shared candidate population: per-image NMS, and the admissibility gate.
    Pinned against the established ``P2`` recipe, so a new implementation that
    disagrees with the committed one fails a test rather than a benchmark.
:mod:`~owl.active_selection.budget`
    the annotation ledger. What a region costs, what the oracle answers, what
    PROB is actually taught, and which acquired unknown becomes declarable at
    which later task.
:mod:`~owl.active_selection.coverage`
    k-center greedy (farthest-first traversal) — the standard core-set
    criterion, at image granularity because opening an image labels everything
    in it.
:mod:`~owl.active_selection.arms`
    the arm registry: five selectors, each a function of detector output and of
    what has already been labelled. None of them reads an annotation.
"""

from owl.active_selection import (
    arms,
    budget,
    coverage,
    population,
)

__all__ = ["arms", "budget", "coverage", "population"]
