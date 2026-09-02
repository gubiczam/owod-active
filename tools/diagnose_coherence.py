"""H2: does binary density coherence remove junk while keeping rare real objects?

The consultation asked for ``coh(x) in {0, 1}`` from DBSCAN core-vs-noise. Two
problems had to be fixed before the idea could be tested at all:

1. **The registered gate was a no-op.** ``coherence_method='binary'`` applied a
   minimum k-means *cluster size*, and at K=1600 the smallest cluster holds 5
   members, so a ``min_samples=5`` floor closed on 0 of 80,000 candidates.
   ``ARMS['consult']`` and ``ARMS['consult_no_gate']`` are therefore bitwise
   identical on every committed seed -- one experiment run twice, not a
   treatment and its control.

2. **On the full pool the idea inverts.** 81% of the pool is background,
   background regions are near copies of each other, so background occupies the
   densest part of the feature space. "Many neighbours" means "looks like
   background", and the gate deletes real objects preferentially.

So the *scope* DBSCAN is fitted on is the experiment, and this tool measures it
against the grid frozen in ``docs/supervisor_week_protocol_2026-09-02.md`` before
it ran. **The grid is not to be widened**, and the acceptance rule is evaluated
in code below rather than by eye, so it cannot drift once the numbers are in.

Discrimination is judged **within the scope**. Comparing in-scope core points
against out-of-scope candidates would measure admissibility, not density, and
would make every admissible setting look like a success.

Oracle labels appear here and only here, as a retrospective diagnostic. The
selector never sees them.

    python tools/diagnose_coherence.py --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import clustering, protocol, scoring
from owl import proposals as proposals_module

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"

# ----------------------------------------------------- the predeclared grid ---
#
# Frozen 2026-09-02, before the first run. Widening it after seeing a result
# would turn a falsification test into a search, so it lives here as data.

PRIMARY_SHARE = 0.30                       # the scope the method will use
SHARE_GRID = (0.10, 0.20, 0.30, 0.50)      # sensitivity only
EPS_GRID = (0.15, 0.25, 0.35, 0.45)
MIN_SAMPLES_GRID = (5, 20)
PCA_DIMENSIONS = 32

# ------------------------------------------------------- the decision rule ---
#
# All four must hold for one setting for the gate to enter the method.

MIN_TAIL_RETAINED = 0.90   # criterion 3: keep >= 90% of in-scope tail objects
MIN_NOISE_SHARE = 0.05     # criterion 4: the gate must actually do something


def _distinct(object_ids: np.ndarray) -> int:
    """Distinct annotated objects, not proposals. Two boxes on one object = one."""

    valid = object_ids[object_ids >= 0]
    return int(np.unique(valid).size)


def evaluate(
    pool: proposals_module.Candidates,
    gate: clustering.CoherenceGate,
    *,
    group_of: np.ndarray,
) -> dict:
    """Retrospective scoring of one gate setting, judged within its own scope."""

    oracle = pool.oracle()
    kind = oracle.kind
    scope = gate.scope
    coherent = scope & (gate.gate > 0)
    noise = gate.is_noise                       # in-scope DBSCAN noise only

    def rate(mask: np.ndarray, value: str) -> float:
        n = int(mask.sum())
        return float((kind[mask] == value).sum() / n) if n else float("nan")

    unknown_in_scope = scope & (kind == "unknown")
    unknown_kept = coherent & (kind == "unknown")

    def tail_objects(mask: np.ndarray) -> int:
        return _distinct(oracle.object_id[mask & (group_of == "tail")])

    tail_scope = tail_objects(unknown_in_scope)
    tail_kept = tail_objects(unknown_kept)
    tail_pool = tail_objects(kind == "unknown")

    known_in_scope = int((scope & (kind == "known")).sum())
    known_kept = int((coherent & (kind == "known")).sum())

    row = {
        # --- what the scope is -------------------------------------------
        "scope_size": int(scope.sum()),
        "scope_share_of_pool": float(scope.mean()),
        "clusters": gate.n_clusters,
        # --- what the gate does, inside the scope ------------------------
        "noise_share_within_scope": float(noise.sum() / max(int(scope.sum()), 1)),
        "core_border_share_within_scope": float(coherent.sum() / max(int(scope.sum()), 1)),
        # --- criteria 1 and 2: who gets deleted --------------------------
        "unknown_rate_coherent": rate(coherent, "unknown"),
        "unknown_rate_noise": rate(noise, "unknown"),
        "background_rate_coherent": rate(coherent, "background"),
        "background_rate_noise": rate(noise, "background"),
        # --- what real structure survives --------------------------------
        "unknown_objects_scope": _distinct(oracle.object_id[unknown_in_scope]),
        "unknown_objects_kept": _distinct(oracle.object_id[unknown_kept]),
        "unknown_classes_kept": int(
            np.unique(oracle.class_name[unknown_kept]).size
        ),
        "head_objects_kept": _distinct(
            oracle.object_id[unknown_kept & (group_of == "head")]
        ),
        "medium_objects_kept": _distinct(
            oracle.object_id[unknown_kept & (group_of == "medium")]
        ),
        "tail_objects_kept": tail_kept,
        "tail_classes_kept": int(
            np.unique(oracle.class_name[unknown_kept & (group_of == "tail")]).size
        ),
        # criterion 3 is judged against the scope; the pool ratio is what
        # admissibility itself costs, and the two must not be confused
        "tail_retained_vs_scope": float(tail_kept / tail_scope) if tail_scope else float("nan"),
        "tail_retained_vs_pool": float(tail_kept / tail_pool) if tail_pool else float("nan"),
        # --- known contamination -----------------------------------------
        "known_share_coherent": rate(coherent, "known"),
        "known_retained_vs_scope": (
            float(known_kept / known_in_scope) if known_in_scope else float("nan")
        ),
    }

    # --------------------------------------------- the four frozen criteria ---
    c1 = row["unknown_rate_coherent"] > row["unknown_rate_noise"]
    c2 = row["background_rate_noise"] > row["background_rate_coherent"]
    c3 = row["tail_retained_vs_scope"] >= MIN_TAIL_RETAINED
    c4 = row["noise_share_within_scope"] >= MIN_NOISE_SHARE
    row |= {
        "c1_objects_not_preferentially_deleted": bool(c1),
        "c2_background_concentrated_in_noise": bool(c2),
        "c3_tail_retained": bool(c3),
        "c4_gate_does_something": bool(c4),
        "accepted": bool(c1 and c2 and c3 and c4),
    }
    return row


def run(pool: proposals_module.Candidates, seeds: int) -> list[dict]:
    groups = protocol.load_groups()
    oracle = pool.oracle()
    group_of = np.asarray([groups.get(name, "") for name in oracle.class_name])

    admissibility = scoring.admissibility(pool)
    rows: list[dict] = []

    # settings: the primary grid (both scopes x eps x min_samples), then the
    # admissible-share sensitivity at the primary min_samples only.
    settings: list[tuple[str, float, float, int]] = []
    for eps in EPS_GRID:
        for min_samples in MIN_SAMPLES_GRID:
            settings.append(("full_pool", 1.0, eps, min_samples))
            settings.append(("admissible", PRIMARY_SHARE, eps, min_samples))
    for share in SHARE_GRID:
        if share == PRIMARY_SHARE:
            continue                      # already covered by the primary grid
        for eps in EPS_GRID:
            settings.append(("admissible_sensitivity", share, eps, MIN_SAMPLES_GRID[0]))

    total = len(settings) * seeds
    done = 0
    for seed in range(seeds):
        for scope_name, share, eps, min_samples in settings:
            mask = clustering.admissible_mask(admissibility, share)
            gate = clustering.density_coherence(
                pool.embeddings,
                scope=mask,
                eps=eps,
                min_samples=min_samples,
                pca_dimensions=PCA_DIMENSIONS,
                seed=seed,
            )
            row = {
                "seed": seed,
                "scope": scope_name,
                "admissible_share": share,
                "eps": eps,
                "min_samples": min_samples,
            } | evaluate(pool, gate, group_of=group_of)
            rows.append(row)
            done += 1
            print(
                f"  [{done:3d}/{total}] {scope_name:22s} share={share:.2f} "
                f"eps={eps:.2f} ms={min_samples:2d} seed={seed}  "
                f"noise={row['noise_share_within_scope']:.3f} "
                f"unk_coh={row['unknown_rate_coherent']:.4f} "
                f"unk_noise={row['unknown_rate_noise']:.4f} "
                f"tail_ret={row['tail_retained_vs_scope']:.3f} "
                f"{'ACCEPT' if row['accepted'] else ''}"
            )
    return rows


def verdict(rows: list[dict]) -> None:
    """State H2's outcome from the frozen rule, per scope, over all seeds."""

    print("\n" + "=" * 78)
    print("H2 VERDICT  (rule frozen in docs/supervisor_week_protocol_2026-09-02.md)")
    print("=" * 78)

    seeds = sorted({row["seed"] for row in rows})
    for scope in ("full_pool", "admissible", "admissible_sensitivity"):
        subset = [row for row in rows if row["scope"] == scope]
        if not subset:
            continue
        keys = sorted({(row["admissible_share"], row["eps"], row["min_samples"])
                       for row in subset})
        all_seed_accepts = []
        for key in keys:
            matching = [
                row for row in subset
                if (row["admissible_share"], row["eps"], row["min_samples"]) == key
            ]
            if len(matching) == len(seeds) and all(row["accepted"] for row in matching):
                all_seed_accepts.append(key)
        print(f"\n{scope}: {len(all_seed_accepts)} of {len(keys)} settings accepted "
              f"on all {len(seeds)} seeds")
        for share, eps, min_samples in all_seed_accepts:
            print(f"    share={share:.2f} eps={eps:.2f} min_samples={min_samples}")

        # why the rejections happened, which is the interesting part
        counts = {name: 0 for name in
                  ("c1_objects_not_preferentially_deleted",
                   "c2_background_concentrated_in_noise",
                   "c3_tail_retained", "c4_gate_does_something")}
        for row in subset:
            for name in counts:
                if not row[name]:
                    counts[name] += 1
        print(f"    failures out of {len(subset)} rows: "
              + ", ".join(f"{k.split('_')[0]}={v}" for k, v in counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="coherence_scope.csv")
    arguments = parser.parse_args()

    pool = proposals_module.from_frozen_pool(split="pool")
    print("pool:", pool.describe())
    print(f"grid: shares={SHARE_GRID} eps={EPS_GRID} min_samples={MIN_SAMPLES_GRID} "
          f"pca={PCA_DIMENSIONS} (frozen)\n")

    rows = run(pool, arguments.seeds)

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / arguments.out
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path} ({len(rows)} rows)")

    verdict(rows)


if __name__ == "__main__":
    main()
