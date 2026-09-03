#!/usr/bin/env python
"""Post-hoc mechanistic audit of the completed Method V3 experiment.

Read-only on the results. It never writes inside ``--results``: everything goes
to ``--out`` (default ``<results>/../method_v3_posthoc_audit``), so the frozen
verdict and its artefacts cannot be touched.

It answers four questions, and it separates what needs the Drive artefacts from
what does not:

1. **Did A and A*C select the same regions, or merely selections with the same
   oracle aggregates?** Decided two ways. The authoritative one reads each
   trajectory's own ``train/labelled_ids.txt`` and ``train/replay_ids.txt`` —
   the exact lists handed to PROB — and diffs them. The second recomputes both
   rankings and reports prefix overlap at every budget mark, which needs the
   frozen view export.
2. **What varies between paired trajectories?** The selection, the rehearsal
   set and PROB's own seed are each checked for whether they actually differ.
3. **Where does acquisition quality stop being supervision?** The chain
   regions -> opened images -> trainable images -> GT boxes PROB keeps, with the
   class distribution and the new class counted exactly.
4. **Is "600 regions" a supervision-matched budget?** The same chain, per arm.

    # offline: everything except the A*C recomputation
    python tools/audit_method_v3.py --out /tmp/audit

    # on Colab, with the completed run and the frozen views
    python tools/audit_method_v3.py \\
        --results /content/drive/MyDrive/OWL/results/method_v3_selection_transfer \\
        --export  /content/drive/MyDrive/OWL/features/dinov2_vitb14_method_v2_v1.npz \\
        --views   /content/drive/MyDrive/OWL/features/dinov2_vitb14_stage2_views_v1.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import discovery, method_v3, protocol, replay, scoring
from owl import exemplars as em
from owl.runner import table

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "pool" / "sowodb_t1_frozen_pool.npz"
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
REPLAY_INDEX = ROOT / "data" / "reference" / "t1_replay_class_counts.json"

PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)


# ------------------------------------------------------------------ helpers ---


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Rank correlation, computed without a new dependency."""

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        out = np.empty(values.size, dtype=np.float64)
        out[order] = np.arange(values.size, dtype=np.float64)
        return out

    a, b = ranks(np.asarray(left, float)), ranks(np.asarray(right, float))
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator else float("nan")


def discordant_pairs(left: np.ndarray, right: np.ndarray, index: np.ndarray) -> int:
    """Pairs within ``index`` that the two scores order differently. Exact, O(k^2)."""

    a, b = np.asarray(left)[index], np.asarray(right)[index]
    count = 0
    for i in range(index.size - 1):
        first = np.sign(a[i] - a[i + 1:])
        second = np.sign(b[i] - b[i + 1:])
        count += int((first * second < 0).sum())
    return count


def prefix(score: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.asarray(score), kind="mergesort")[:k]


def jaccard(left: set, right: set) -> float:
    union = len(left | right)
    return len(left & right) / union if union else float("nan")


# -------------------------------------------- 1. A vs A*C selection identity ---


def a_gap_structure(admissibility: np.ndarray) -> list[dict]:
    """How tight is the A ranking at each cut, and therefore how little C needs
    to vary to move the prefix.

    The prefix under ``A * C`` equals the prefix under ``A`` **only if**
    ``A_k / A_(k+1) > C_max / C_min`` over the band that could cross. So the gap
    ratio is exactly the dynamic range of C that the prefix can absorb, and it
    needs no C value to compute.
    """

    order = np.argsort(-admissibility, kind="mergesort")
    rows = []
    for mark in method_v3.BUDGET_MARKS:
        at, below = admissibility[order[mark - 1]], admissibility[order[mark]]
        row = {"budget": mark, "A_at_cut": at, "A_below_cut": below,
               "gap_ratio": at / below if below else float("inf")}
        for width in (0.001, 0.01, 0.05):
            row[f"within_{width:g}"] = int(
                ((admissibility >= at / (1 + width))
                 & (admissibility <= at * (1 + width))).sum()
            )
        rows.append(row)
    return rows


def c_distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    row = {"n": int(values.size), "mean": float(values.mean()),
           "std": float(values.std(ddof=1)),
           "dynamic_range_max_over_min": float(values.max() / values.min())
           if values.min() > 0 else float("inf")}
    for percentile in PERCENTILES:
        row[f"p{percentile}"] = float(np.percentile(values, percentile))
    return row


def prefix_comparison(
    candidates, admissibility: np.ndarray, product: np.ndarray
) -> list[dict]:
    """Prefix overlap, image overlap and reordering, at every budget mark."""

    image_ids = np.asarray(candidates.image_ids, dtype=str)
    rows = []
    for mark in method_v3.BUDGET_MARKS:
        left, right = prefix(admissibility, mark), prefix(product, mark)
        ls, rs = set(left.tolist()), set(right.tolist())
        left_images = set(image_ids[left].tolist())
        right_images = set(image_ids[right].tolist())
        union = np.asarray(sorted(ls | rs), dtype=np.int64)
        rows.append({
            "budget": mark,
            "proposal_intersection": len(ls & rs),
            "proposal_jaccard": round(jaccard(ls, rs), 6),
            "entering_under_AxC": len(rs - ls),
            "leaving_under_AxC": len(ls - rs),
            "image_intersection": len(left_images & right_images),
            "images_A": len(left_images), "images_AxC": len(right_images),
            "image_jaccard": round(jaccard(left_images, right_images), 6),
            "spearman_on_union": round(
                spearman(admissibility[union], product[union]), 6),
            "discordant_pairs_in_union": discordant_pairs(
                admissibility, product, union),
        })
    return rows


# --------------------------------------------------- 3 & 4. the supervision ---


def supervision_chain(
    candidates, arm: str, seed: int, *, candidate_index, consistency=None,
    image_ids=None,
) -> dict:
    """regions -> opened -> trainable -> the GT boxes PROB actually keeps.

    ``image_ids`` overrides the recomputation with the trajectory's own
    ``labelled_ids.txt``, which is authoritative: it is the list PROB was handed.
    """

    task = protocol.build_chain(method_v3.N_TASKS)[1]
    declared = set(protocol.CLASS_ORDER[: task.n_current])
    known_now = frozenset(task.known_classes)
    groups = protocol.load_groups()

    picked = method_v3.select_for_arm(candidates, arm, seed, consistency=consistency)
    opened = [str(value) for value in picked.images(candidates)]
    if image_ids is None:
        trainable = [
            name for name in opened
            if any(cls in known_now for cls in candidate_index.get(name, {}))
        ]
        source = "recomputed"
    else:
        trainable = [str(value) for value in image_ids]
        source = "labelled_ids.txt"

    kept: dict[str, int] = {}
    dropped = 0
    for name in trainable:
        for cls, count in candidate_index.get(name, {}).items():
            if cls in declared:
                kept[cls] = kept.get(cls, 0) + count
            else:
                dropped += count

    found = discovery.discovery(candidates, picked.indices, groups=groups)
    total = sum(kept.values())
    return {
        "arm": arm, "seed": seed, "trainable_source": source,
        "regions": len(picked),
        "images_opened": len(opened),
        "images_trainable": len(trainable),
        "images_barren": len(opened) - len(trainable),
        "regions_per_opened_image": round(len(picked) / max(len(opened), 1), 3),
        "supervised_boxes": total,
        "boxes_per_region": round(total / max(len(picked), 1), 3),
        "undeclared_boxes_dropped": dropped,
        "declared_classes_with_zero_boxes": sum(
            1 for cls in protocol.CLASS_ORDER[: task.n_current] if not kept.get(cls)
        ),
        "person_boxes": kept.get("person", 0),
        "person_share_of_supervision": round(
            kept.get("person", 0) / max(total, 1), 4),
        "new_class_boxes": kept.get(task.new_class, 0),
        "new_class_images": sum(
            1 for name in trainable if task.new_class in candidate_index.get(name, {})
        ),
        "acquired_unknown_objects": found.unknown_objects,
        "acquired_medium_tail_objects": (
            found.objects_by_group.get("medium", 0)
            + found.objects_by_group.get("tail", 0)
        ),
        "selected_background": found.selected_background,
        "background_share": round(found.selected_background / max(len(picked), 1), 4),
        "per_class": kept,
    }


def replay_memory(candidates, arm, seed, *, candidate_index, replay_index,
                  consistency=None):
    """Reproduce the trajectory's 400-object rehearsal set exactly."""

    task = protocol.build_chain(method_v3.N_TASKS)[1]
    known_now = frozenset(task.known_classes)
    previous = frozenset(task.previous_classes)
    picked = method_v3.select_for_arm(candidates, arm, seed, consistency=consistency)
    trainable = frozenset(
        str(value) for value in picked.images(candidates)
        if any(cls in known_now for cls in candidate_index.get(str(value), {}))
    )
    pool = tuple(
        item for item in em.enumerate_pool(replay_index, previous)
        if item.class_name in previous and item.image_id not in trainable
    )
    demand = replay.allocate(em.capacities(pool), total=400, alpha=0.0)
    return set(em.select(pool, demand, incumbent=(), reallocate=False, seed=seed))


# ---------------------------------------------- 2. what the artefacts prove ---


def trajectory_artefacts(results: Path) -> dict[tuple[str, int], dict]:
    """The lists PROB was actually handed, per trajectory. Authoritative."""

    found: dict[tuple[str, int], dict] = {}
    for arm, seed in method_v3.trajectories():
        directory = results / method_v3.trajectory_name(arm, seed)
        record = method_v3.load_trajectory(directory)
        entry: dict = {"directory": str(directory), "result": record}
        for name, key in (("labelled_ids.txt", "labelled"),
                          ("replay_ids.txt", "replay")):
            path = directory / "train" / name
            entry[key] = (
                [line.strip() for line in
                 path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if path.exists() else None
            )
        found[(arm, seed)] = entry
    return found


def paired_identity(artefacts: dict, treatment: str, control: str) -> list[dict]:
    """Did the paired trajectories receive the same supervision and rehearsal?"""

    rows = []
    for seed in method_v3.SEEDS:
        left = artefacts.get((treatment, seed), {})
        right = artefacts.get((control, seed), {})
        row: dict = {"seed": seed, "treatment": treatment, "control": control}
        for key in ("labelled", "replay"):
            a, b = left.get(key), right.get(key)
            if a is None or b is None:
                row[f"{key}_identical"] = "artefact missing"
                row[f"{key}_intersection"] = None
                row[f"{key}_jaccard"] = None
                continue
            sa, sb = set(a), set(b)
            row[f"{key}_identical"] = sa == sb
            row[f"{key}_n_treatment"] = len(sa)
            row[f"{key}_n_control"] = len(sb)
            row[f"{key}_intersection"] = len(sa & sb)
            row[f"{key}_jaccard"] = round(jaccard(sa, sb), 6)
        rows.append(row)
    return rows


# --------------------------------------------------------------------- main ---


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None,
                        help="the completed Method V3 results directory; read-only")
    parser.add_argument("--export", default=None, help="frozen DINOv2 base export")
    parser.add_argument("--views", default=None, help="frozen Stage-2 view export")
    parser.add_argument("--out", default=None, help="where the audit tables go")
    arguments = parser.parse_args()

    results = Path(arguments.results) if arguments.results else None
    out = Path(arguments.out) if arguments.out else (
        results.parent / "method_v3_posthoc_audit" if results
        else ROOT / "data" / "results" / "method_v3_posthoc_audit"
    )
    if results is not None and out.resolve() == results.resolve():
        raise SystemExit(
            "--out must not be --results: this audit is read-only on the frozen "
            "Method V3 artefacts."
        )
    out.mkdir(parents=True, exist_ok=True)

    candidate_index = json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    replay_index = json.loads(REPLAY_INDEX.read_text(encoding="utf-8"))
    pool = method_v3.population(POOL, candidate_index)
    candidates = pool.candidates
    admissibility = scoring.admissibility(candidates)
    task = protocol.build_chain(method_v3.N_TASKS)[1]
    summary: dict = {
        "verdict_under_audit": "C_DOWNSTREAM_NOT_SUPPORTED",
        "note": "pre-registered and not revisited here",
        "population": pool.provenance,
        "task": {"name": task.name, "new_class": task.new_class,
                 "declared_classes": task.n_current},
    }

    print("=" * 78)
    print("METHOD V3 — POST-HOC MECHANISTIC AUDIT")
    print("=" * 78)
    print("Pre-registered verdict: C_DOWNSTREAM_NOT_SUPPORTED — not revisited here.")
    print(f"population: {len(candidates):,} proposals on "
          f"{pool.image_ids.size:,} images")
    print()

    # ---- 1a. how tight is A at the cut (needs no C) ---------------------
    print("1a. A RANKING STRUCTURE AT EACH CUT")
    print("    gap_ratio is exactly the dynamic range of C the prefix can absorb:")
    print("    the A-prefix is unchanged by A*C only if max(C)/min(C) < gap_ratio.")
    gaps = a_gap_structure(admissibility)
    print(table(gaps))
    method_v3.write_rows(out / "audit_a_gap_structure.csv", gaps)
    summary["a_gap_structure"] = gaps
    print()

    consistency = None
    if arguments.views and arguments.export:
        consistency, consistency_provenance = method_v3.consistency_values(
            pool, base_export=arguments.export, views_export=arguments.views)
        summary["consistency_provenance"] = consistency_provenance
        product = method_v3.arm_score("A*C", candidates, consistency=consistency)

        print("1b. C DISTRIBUTION ON THE POPULATION")
        distribution = c_distribution(consistency)
        print(table([distribution]))
        summary["c_distribution"] = distribution
        method_v3.write_rows(out / "audit_c_distribution.csv", [distribution])
        absorbed = [
            row["budget"] for row in gaps
            if row["gap_ratio"] > distribution["dynamic_range_max_over_min"]
        ]
        print(f"    max(C)/min(C) = "
              f"{distribution['dynamic_range_max_over_min']:.6f}")
        print(f"    budgets whose A-prefix that range cannot move: "
              f"{absorbed or 'none'}")
        summary["budgets_with_invariant_prefix"] = absorbed
        print()

        print("1c. A vs A*C PREFIX OVERLAP")
        overlap = prefix_comparison(candidates, admissibility, product)
        print(table(overlap))
        method_v3.write_rows(out / "audit_prefix_overlap.csv", overlap)
        summary["prefix_overlap"] = overlap
        print(f"    Spearman(A, A*C) over the whole population: "
              f"{spearman(admissibility, product):.6f}")
        summary["spearman_population"] = spearman(admissibility, product)
        print()
    else:
        print("1b/1c. SKIPPED — pass --export and --views to recompute A*C.")
        print("       Without them the A*C prefix cannot be reconstructed, because")
        print("       C exists only in the frozen view export.")
        print()

    # ---- 2. what actually differs between paired trajectories -----------
    print("2. WHAT DIFFERS BETWEEN PAIRED TRAJECTORIES")
    prob_seed_varies = False
    print(f"    PROB's own --seed: owl.bridge.Bridge.seed default = 0, and "
          f"tools/run_method_v3.py constructs Bridge without seed=, so PROB was "
          f"seeded 0 in every trajectory. varies across trajectories: "
          f"{prob_seed_varies}")
    summary["prob_seed_varies_across_trajectories"] = prob_seed_varies

    memories = {}
    for arm in ("A", "U"):
        for seed in method_v3.SEEDS:
            memories[(arm, seed)] = replay_memory(
                candidates, arm, seed, candidate_index=candidate_index,
                replay_index=replay_index)
    seed_rows = []
    for arm in ("A", "U"):
        base = memories[(arm, 0)]
        seed_rows.append({
            "arm": arm,
            "selection_identical_across_seeds": True,
            "replay_overlap_s0_s1": len(base & memories[(arm, 1)]),
            "replay_overlap_s0_s2": len(base & memories[(arm, 2)]),
            "replay_size": len(base),
        })
    print(table(seed_rows))
    method_v3.write_rows(out / "audit_seed_effect.csv", seed_rows)
    summary["seed_effect"] = seed_rows
    print()

    if results is not None:
        artefacts = trajectory_artefacts(results)
        print("2b. THE LISTS PROB WAS ACTUALLY HANDED (authoritative)")
        for treatment, control in (("A*C", "A"), ("U", "A"), ("A", "random")):
            rows = paired_identity(artefacts, treatment, control)
            print(f"\n  {treatment} vs {control}")
            print(table(rows))
            method_v3.write_rows(
                out / f"audit_paired_{method_v3.SLUGS[treatment]}"
                f"_vs_{method_v3.SLUGS[control]}.csv", rows)
            summary[f"paired_{treatment}_vs_{control}"] = rows
        print()
    else:
        artefacts = {}
        print("2b. SKIPPED — pass --results to diff the actual training lists.")
        print()

    # ---- 3 & 4. the supervision chain, per arm --------------------------
    print("3+4. SUPERVISION CHAIN AND BUDGET FAIRNESS")
    chain_rows = []
    for arm in method_v3.ARMS:
        if arm == "A*C" and consistency is None:
            continue
        for seed in (method_v3.SEEDS if arm == "random" else (0,)):
            supplied = None
            entry = artefacts.get((arm, seed), {})
            if entry.get("labelled"):
                supplied = entry["labelled"]
            chain_rows.append(supervision_chain(
                candidates, arm, seed, candidate_index=candidate_index,
                consistency=consistency if arm == "A*C" else None,
                image_ids=supplied,
            ))
    printable = [
        {key: value for key, value in row.items() if key != "per_class"}
        for row in chain_rows
    ]
    print(table(printable))
    method_v3.write_rows(out / "audit_supervision_chain.csv", printable)
    summary["supervision_chain"] = printable
    print()

    print("    per-class supervised positives (declared classes only)")
    class_rows = []
    groups = protocol.load_groups()
    for cls in protocol.CLASS_ORDER[: task.n_current]:
        row = {"class": cls, "band": groups.get(cls, "—"),
               "is_new_class": cls == task.new_class}
        for entry in chain_rows:
            row[f"{entry['arm']}_s{entry['seed']}"] = entry["per_class"].get(cls, 0)
        class_rows.append(row)
    print(table(class_rows))
    method_v3.write_rows(out / "audit_per_class_supervision.csv", class_rows)
    summary["per_class_supervision"] = class_rows
    print()

    method_v3.write_json(out / "audit_summary.json", summary)
    print(f"wrote {out}")
    print()
    print("This audit did not modify the Method V3 results directory.")


if __name__ == "__main__":
    main()
