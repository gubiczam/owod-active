"""Distribution-aware active annotation for open-world incremental detection.

Ten modules, one concept each. Read them in this order:

``protocol``    the task chain — one new class per task, starting from PROB's t1
``proposals``   what the detector proposes; detector fields and oracle fields kept apart
``clustering``  one partition, from which both rarity and diversity are read
``scoring``     the four terms of s(x) and the score itself
``selection``   spending the budget: rounds, batch diversity, the registered arms
``labelling``   what gets labelled on a chosen image, and what that costs
``replay``      the exemplar memory and its distribution-aware allocation
``metrics``     PROB's evaluator, read; forgetting, learning and the exchange rate
``bridge``      calling PROB — the only module that knows about a GPU
``runner``      the cycle that ties them together, simulated or real
"""

__version__ = "0.1.0"

from owl import (  # noqa: F401
    bridge,
    clustering,
    labelling,
    metrics,
    proposals,
    protocol,
    replay,
    runner,
    scoring,
    selection,
)

__all__ = [
    "bridge", "clustering", "labelling", "metrics", "proposals",
    "protocol", "replay", "runner", "scoring", "selection",
]
