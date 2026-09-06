#!/usr/bin/env python3
"""What is the *most* any image-level acquisition policy could deliver?

Read-only. CPU. Committed artefacts only. No detector, no training, no writes
into any results directory.

Why this exists. Before designing a third selection method it is worth knowing
whether selection has any headroom left at all, and — if it does — which
mechanism the headroom belongs to. This tool answers both by simulating
*hypothetical* policies against the benchmark's own candidate index
(``data/reference/per_image_class_counts.json``, 28 800 images) under the frozen
cost model::

    cost(image) = max(1, annotated objects on it)          budget 3 000 / task

Oracle use. Every policy here except ``random`` reads oracle labels or oracle
costs. **They are not selectors and none of them may be implemented as one.**
They are *ceilings*: the best a real, label-free selector could ever do if its
representation were perfect. A ceiling that sits at the level a real arm already
reached says the mechanism is exhausted; a ceiling far above it says the
mechanism is worth building; and a *confound* ceiling that beats the mechanism's
own ceiling says the mechanism is not what would be doing the work.

The endpoint is deliberately cumulative. Distribution-aware acquisition does not
claim to buy more of the declared class at its own task — it claims to buy the
tail *early*, while it is still unknown, so that the ledger already holds it when
the class is declared. So ``held`` counts objects of the declared class bought at
this task **and** at any earlier one.

Caveats, stated rather than buried:

* the candidate sample is drawn with the protocol's key ``(seed, task.index)``
  but is **not** byte-identical to a trajectory's own draw, so ``random`` here is
  a structural expectation and not the ``random`` arm's measured behaviour;
* ``held`` is what the *annotation ledger* holds. It is not what PROB is handed —
  the banking defect (``docs/banking_defect_forensics_2026-09-04.md``) means only
  part of it is delivered — and it is certainly not AP;
* one class per task, this repository's controlled chain, not S-OWODB.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from owl import protocol
from owl.active_selection import benchmark as bm

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"

Counts = Mapping[str, Mapping[str, int]]
Policy = Callable[..., Sequence[str]]


def load_index(path: Path = INDEX) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_cost(counts: Mapping[str, Mapping[str, int]], image: str) -> int:
    """The frozen cost model, restated here so the tool is self-contained."""

    return max(1, sum(counts.get(image, {}).values()))


def spend(order: Sequence[str], counts: Counts, budget: int) -> tuple[list[str], int]:
    """Open images in ``order`` until the next one does not fit.

    Same stopping rule as :class:`owl.active_selection.budget.Ledger` — an
    unaffordable image ends the campaign rather than being skipped, so the tail
    of a campaign is not biased toward sparse images.
    """

    opened: list[str] = []
    spent = 0
    for image in order:
        cost = image_cost(counts, image)
        if cost > budget - spent:
            break
        opened.append(image)
        spent += cost
    return opened, spent


# ----------------------------------------------------------------- policies ---
#
# Signature: (pool, counts, known, seed, task_index, budget) -> order of images.


def policy_random(pool, counts, known, seed, task_index, budget):
    return list(np.random.default_rng((seed, task_index, 7)).permutation(pool))


def policy_cheapest(pool, counts, known, seed, task_index, budget):
    """CONFOUND CEILING. No labels, no clustering — just open cheap images.

    Rare classes live on sparse images and sparse images are cheap, so this
    collects the tail with no distribution-awareness whatsoever. If it matches
    the stratified ceiling, stratification is not the mechanism.
    """

    jitter = np.random.default_rng((seed, task_index, 3)).random(len(pool))
    costs = np.array([image_cost(counts, i) for i in pool])
    return [pool[k] for k in np.lexsort((jitter, costs))]


def policy_class_greedy(pool, counts, known, seed, task_index, budget, *, target):
    """SUPPLY CEILING. Perfect foreknowledge of which class is declared next."""

    return sorted(
        pool,
        key=lambda i: (-counts.get(i, {}).get(target, 0) / image_cost(counts, i), i),
    )


def _stratified(pool, counts, known, seed, task_index, budget, *, within):
    """Equal answer quota to every unknown class present. Perfect clusters.

    This is the ceiling of the cluster-stratified idea: clusters that are exactly
    the true classes, and a rarity rule that is exactly equal-quota. No real
    clusterer can beat it.
    """

    present = sorted({c for i in pool for c in counts.get(i, {}) if c not in known})
    quota = budget / max(len(present), 1)
    order: list[str] = []
    seen: set[str] = set()
    for name in present:
        members = sorted(
            (i for i in pool if counts.get(i, {}).get(name, 0)),
            key=lambda i: within(i, name),
        )
        spent = 0.0
        for image in members:
            if image in seen:
                continue
            cost = image_cost(counts, image)
            if spent + cost > quota:
                break
            order.append(image)
            seen.add(image)
            spent += cost
    rest = np.random.default_rng((seed, task_index, 11)).permutation(pool)
    return order + [i for i in rest if i not in seen]


def policy_stratified(pool, counts, known, seed, task_index, budget):
    """Within a cluster, the image richest in that class per answer."""

    return _stratified(
        pool, counts, known, seed, task_index, budget,
        within=lambda i, c: (-counts.get(i, {}).get(c, 0) / image_cost(counts, i), i),
    )


def policy_stratified_cheapest(pool, counts, known, seed, task_index, budget):
    """Within a cluster, the cheapest image. Tests whether the two compose."""

    return _stratified(
        pool, counts, known, seed, task_index, budget,
        within=lambda i, c: (image_cost(counts, i), i),
    )


POLICIES: dict[str, Policy] = {
    "random": policy_random,
    "cheapest*": policy_cheapest,
    "stratified*": policy_stratified,
    "stratified+cheapest*": policy_stratified_cheapest,
    "class-greedy*": policy_class_greedy,
}


def trajectory(name, counts, images, known, chain, seed, *, budget, pool_size):
    """Walk one policy down the whole chain, carrying the ledger forward."""

    policy = POLICIES[name]
    used: set[str] = set()
    held: dict[str, int] = {}
    rows = []
    for task in chain[1:]:
        free = np.array([i for i in images if i not in used], dtype=str)
        pool = list(
            np.random.default_rng((seed, task.index))
            .choice(free, size=min(pool_size, free.size), replace=False)
        )
        target = task.new_class
        kwargs = {"target": target} if name == "class-greedy*" else {}
        order = policy(pool, counts, known, seed, task.index, budget, **kwargs)
        opened, spent = spend(order, counts, budget)

        this = sum(counts.get(i, {}).get(target, 0) for i in opened)
        banked = held.get(target, 0)
        available = sum(counts.get(i, {}).get(target, 0) for i in pool)
        unknown_objects = sum(
            n for i in opened for c, n in counts.get(i, {}).items() if c not in known
        )
        for image in opened:
            for name_, n in counts.get(image, {}).items():
                held[name_] = held.get(name_, 0) + n
        used.update(opened)

        rows.append({
            "policy": name, "seed": seed, "task": task.name, "class": target,
            "available_in_pool": available,
            "images_opened": len(opened), "answers_spent": spent,
            "acquired_this_task": this, "banked_from_earlier": banked,
            "held_at_declaration": this + banked,
            "held_per_image": round((this + banked) / max(len(opened), 1), 4),
            "unknown_objects": unknown_objects,
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--csv", type=Path, help="write the rows here")
    args = parser.parse_args(argv)

    counts = load_index()
    images = np.array(sorted(counts), dtype=str)
    known = set(protocol.TASK1)
    chain = protocol.build_chain(bm.N_TASKS)

    unknown = sorted({c for v in counts.values() for c in v if c not in known})
    declared = [t.new_class for t in chain[1:]]
    print(f"candidate index : {len(images)} images, {len(unknown)} unknown classes")
    print(f"declared in this chain : {declared} — "
          f"{len(declared)} of {len(unknown)}")
    print(f"budget : {bm.ANSWER_BUDGET_PER_TASK} answers/task over "
          f"{bm.CANDIDATE_IMAGES_PER_TASK} candidate images\n")

    rows = [r for name in args.policies for seed in args.seeds
            for r in trajectory(name, counts, images, known, chain, seed,
                                budget=bm.ANSWER_BUDGET_PER_TASK,
                                pool_size=bm.CANDIDATE_IMAGES_PER_TASK)]

    header = (f"{'policy':<22}{'seed':>5}{'task':>5} {'class':<14}"
              f"{'avail':>7}{'imgs':>6}{'ans':>6}{'this':>6}{'bank':>6}"
              f"{'HELD':>6}{'/img':>7}")
    print(header)
    print("-" * len(header))
    last = None
    for r in rows:
        if last is not None and r["policy"] != last:
            print()
        last = r["policy"]
        print(f"{r['policy']:<22}{r['seed']:>5}{r['task']:>5} {r['class']:<14}"
              f"{r['available_in_pool']:>7}{r['images_opened']:>6}"
              f"{r['answers_spent']:>6}{r['acquired_this_task']:>6}"
              f"{r['banked_from_earlier']:>6}{r['held_at_declaration']:>6}"
              f"{r['held_per_image']:>7.3f}")

    print("\n* = reads oracle labels or oracle costs. These are CEILINGS, not arms.")
    print("HELD = declared-class objects the ledger holds when the class is declared.")

    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
