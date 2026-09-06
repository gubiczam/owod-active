"""Fingerprint what a given revision's arms actually select.

Run with a checked-out ``owl/`` first on ``sys.path``. A completed experiment's
notebook keeps its pin — re-pinning a finished run onto later code is how a
result quietly becomes a different result — so the property that must hold is
not "the tree has not moved" but "the arms this notebook runs have not moved".

Deliberately uses a small synthetic pool and synthetic features: behavioural
equivalence does not need the frozen 120k-row pool or a real DINOv2 export, and
this keeps the comparison to a couple of seconds.
"""
import hashlib
import json
import sys

import numpy as np

from owl.active_selection import arms, population
from owl.proposals import Candidates

ARMS = json.loads(sys.argv[1])
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1

rng = np.random.default_rng(20260906)
n_images, per_image = 60, 25
n = n_images * per_image
candidates = Candidates(
    image_ids=np.repeat(np.array([f"img{i:04d}" for i in range(n_images)]), per_image),
    boxes=np.clip(rng.random((n, 4)) * 0.6 + 0.2, 0.05, 0.95).astype(np.float32),
    posterior=rng.dirichlet(np.full(81, 0.5), size=n).astype(np.float32),
    objectness=rng.random(n).astype(np.float32),
    embeddings=rng.normal(size=(n, 16)).astype(np.float32),
)
pool = population.build(candidates)
cost_of = (lambda i: len(i) % 7 + 1)


def features_for(size):
    block = np.random.default_rng(4242).normal(size=(size, 12))
    return (block / np.linalg.norm(block, axis=1, keepdims=True)).astype(np.float32)


digest = hashlib.sha256()
for arm in ARMS:
    spec = arms.ARMS[arm]
    index = arms.ranked_positions(arm, pool)
    digest.update(arm.encode() + index.tobytes())

    # Revision-tolerant on purpose: this file is executed against *older*
    # checkouts, which is the whole point of it, so anything added after the
    # oldest revision it must run on is read defensively.
    allocated = getattr(arms, "ALLOCATED", frozenset())
    semantic = features_for(index.size) if spec.needs_semantic else None
    if not spec.needs_semantic and spec.kind == "ranking" and arm not in allocated:
        for seed in (0, 1, 2):
            order = arms.ranking(arm, pool, seed=seed)
            digest.update(str(seed).encode() + order.tobytes())

    kwargs = {}
    try:
        kwargs = {"rounds": ROUNDS}
        spend = arms.select(arm, pool, cost_of=cost_of, answer_budget=200,
                            seed=0, semantic=semantic, **kwargs)
    except TypeError:                       # a revision predating `rounds`
        spend = arms.select(arm, pool, cost_of=cost_of, answer_budget=200,
                            seed=0, semantic=semantic)
    digest.update("|".join(spend.images).encode())
print(digest.hexdigest())
