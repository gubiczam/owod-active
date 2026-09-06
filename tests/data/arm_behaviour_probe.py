"""Print a fingerprint of what the session's arms select, for one revision.

Run with a checked-out `owl/` first on sys.path. Deliberately uses a small
synthetic pool: behavioural equivalence does not need the frozen 120k-row pool
and this keeps the comparison to a couple of seconds.
"""
import hashlib
import json
import sys

import numpy as np

from owl.active_selection import arms, population
from owl.proposals import Candidates

ARMS = json.loads(sys.argv[1])
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

digest = hashlib.sha256()
for arm in ARMS:
    for seed in (0, 1, 2):
        order = arms.ranking(arm, pool, seed=seed)
        digest.update(arm.encode() + str(seed).encode() + order.tobytes())
    digest.update(arms.ranked_positions(arm, pool).tobytes())
    spend = arms.select(arm, pool, cost_of=lambda i: len(i) % 7 + 1,
                        answer_budget=200, seed=0)
    digest.update("|".join(spend.images).encode())
print(digest.hexdigest())
