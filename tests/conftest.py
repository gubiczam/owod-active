"""Shared fixtures.

The frozen pool is 80,000 proposals and a k-means over it takes a few seconds.
Both are computed once per session and handed to every test that needs them, so
the suite stays under a minute instead of re-clustering thirty times.
"""

from __future__ import annotations

import pytest

from owl import clustering, proposals


@pytest.fixture(scope="session")
def pool():
    return proposals.from_frozen_pool(split="pool")


@pytest.fixture(scope="session")
def partition(pool):
    return clustering.fit(pool.embeddings, n_clusters=1600, seed=0)
