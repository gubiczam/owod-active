#!/usr/bin/env python
"""Price Benchmark V1 on the CPU, and print the reduction decision, before any GPU runs.

Two jobs, both of which must happen before a session is spent:

1. **Simulate the annotation ledger.** Every arm's selector is run on the
   committed frozen pool with the real answer budget, so the number of images
   each one opens — which is what the training cost is proportional to — is
   measured rather than guessed. The coverage arms are run on PROB decoder
   embeddings here because the frozen DINOv2 export lives on Drive; that
   substitution changes *which* images they open and therefore is a **cost
   proxy only**, never a scientific result, and the output says so.

2. **Decide the compute reduction, in advance.** The ladder is fixed here, not
   after seeing a clock: epochs 5 -> 3 -> 2, then candidate images 1200 -> 800.
   If the estimate at the top rung fits the ceiling, nothing is reduced and the
   script says so. Reducing the *number of arms* is not on the ladder; the
   pre-declared arm order in ``owl.active_selection.arms.ORDER`` handles a short
   session by completing a prefix and resuming the rest.

    python tools/plan_full_owod_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import evaluation_subset, proposals, protocol, runner
from owl.active_selection import arms as arm_registry
from owl.active_selection import benchmark as bm
from owl.active_selection import budget as annotation_budget
from owl.active_selection import population as population_module

ROOT = Path(__file__).resolve().parent.parent
COST_BASIS = ROOT / "data" / "reference" / "gpu_cost_basis.json"
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
TEST_ARCHIVE = ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz"

#: Deliberately conservative. The real rate on a T4 is very likely higher; a
#: plan that over-estimates stops a session cleanly, one that under-estimates
#: loses it. The notebook prints the measured rate at the first task.
DINO_CROPS_PER_MINUTE = 6_000.0

#: Fixed before the first trajectory. Applied uniformly to every arm and every
#: seed, never to a subset.
EPOCH_LADDER: tuple[int, ...] = (5, 3, 2)
CANDIDATE_LADDER: tuple[int, ...] = (2000, 1200, 800)

#: One session. Above this the ladder is descended; the chain is never truncated.
HOURS_CEILING = 10.0


def cost_basis() -> dict:
    return json.loads(COST_BASIS.read_text(encoding="utf-8"))


def minutes(
    basis: dict,
    *,
    candidate_images: int,
    train_images: int,
    eval_images: int,
    epochs: int,
    batch_size: int,
    dino_crops: int,
) -> dict[str, float]:
    """One arm at one task, in minutes, from the project's measured basis."""

    predict = candidate_images / 1000.0 * basis["predict_minutes_per_1000_images"]
    iterations = (train_images // max(batch_size, 1)) * epochs
    train = basis["train_minutes_fixed_overhead"] + iterations * basis[
        "train_seconds_per_iteration"
    ] / 60.0
    # `detections=True` costs a second forward pass over the split.
    evaluate = basis["evaluate_minutes_fixed_overhead"] + 2 * (
        eval_images / 1000.0 * basis["evaluate_minutes_per_1000_images_reference_run"]
    )
    dino = dino_crops / DINO_CROPS_PER_MINUTE
    return {
        "predict": predict, "train": train, "evaluate": evaluate, "dino": dino,
        "iterations": float(iterations),
        "total": predict + train + evaluate + dino,
    }


def simulate_arms(
    candidate_index: dict, *, answer_budget: int, seed: int
) -> list[dict[str, object]]:
    """Run every selector on the committed pool and measure what it opens."""

    pool_candidates = proposals.from_frozen_pool(split="pool")
    pool = population_module.build(pool_candidates)
    # The frozen pool's images are a subset of the candidate index, so the cost
    # function is the real one; only the DINOv2 space is substituted.
    cost_of = annotation_budget.cost_function(candidate_index)
    groups = protocol.load_groups()
    tasks = bm.chain()
    rows: list[dict[str, object]] = []
    for name in arm_registry.ORDER:
        spec = arm_registry.ARMS[name]
        semantic = None
        if spec.needs_semantic:
            ranked = np.flatnonzero(pool.gate) if spec.gated else np.arange(len(pool))
            semantic = pool.candidates.embeddings[ranked]
        picked = arm_registry.select(
            name, pool, cost_of=cost_of, answer_budget=answer_budget,
            seed=seed, semantic=semantic,
        )
        ledger = annotation_budget.supervision(
            candidate_index, picked.images,
            declared=tasks[1].known_classes, groups=groups,
        )
        acquired = annotation_budget.acquisition(
            candidate_index, picked.images, chain=tasks, task_index=1, groups=groups,
        )
        # An image with no declared class cannot be trained on now — PROB's split
        # keeps only the classes introduced so far — so the training set is the
        # rest. Pricing the run on `images_opened` would over-estimate it by the
        # barren share, which for the admissibility arm is 70%.
        trainable = len(picked.images) - int(ledger["images_barren"])
        rows.append({
            "arm": name,
            "proxy_space": "PROB decoder (cost proxy)" if spec.needs_semantic else "—",
            "images_opened": len(picked.images),
            "answers_spent": picked.row["answers_spent"],
            "answers_per_image": picked.row["answers_per_image"],
            "boxes_labelled": ledger["boxes_labelled"],
            "boxes_supervised_t2": ledger["boxes_supervised"],
            "barren_t2": int(ledger["images_barren"]),
            "trainable_t2": trainable,
            "new_class_objects": acquired["acquired_new_class"],
            "known_at_t3": acquired.get("acquired_becomes_known_t3", 0),
            "known_at_t4": acquired.get("acquired_becomes_known_t4", 0),
            "classes": acquired["acquired_classes"],
            "tail_objects": acquired["acquired_tail_objects"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-images", type=int, default=bm.REPLAY_OBJECTS,
                        help="alias images the memory materialises; at most one "
                             "per exemplar object, so M is the upper bound")
    parser.add_argument("--json", default=None, help="also write the plan here")
    arguments = parser.parse_args()

    print("=" * 78)
    print("FULL OWOD ACTIVE SELECTION BENCHMARK V1 — plan")
    print("=" * 78)
    report = bm.check_protocol()
    print(f"[protocol] {report['path']} agrees with the code on "
          f"{report['fields']} frozen fields")

    tasks = bm.chain()
    print("\n[chain] the repository's canonical chain — ONE class per task, NOT "
          "the published S-OWODB 19/21/20/20 split")
    print(runner.table([
        {"task": t.name, "declares": t.new_class or "— (pretrained anchor)",
         "group": protocol.load_groups().get(t.new_class, "—") if t.new_class else "—",
         "known_after": t.n_current, "tail_band": ", ".join(bm.tail_band(t))}
        for t in tasks
    ]))

    print("\n[arms] pre-declared execution order")
    print(runner.table([
        {"order": i + 1, "arm": a, "kind": arm_registry.ARMS[a].kind,
         "dinov2": arm_registry.ARMS[a].needs_semantic,
         "gated": arm_registry.ARMS[a].gated,
         "what it is for": arm_registry.ARMS[a].description}
        for i, a in enumerate(arm_registry.ORDER)
    ]))

    print(f"\n[endpoints] {bm.ENDPOINTS.statement()}")

    subset = evaluation_subset.from_archive(
        TEST_ARCHIVE, bm.declared_classes(), seed=bm.DEVELOPMENT_SEED,
        remainder_multiplier=bm.EVAL_REMAINDER_RATIO,
        max_per_class=bm.EVAL_MAX_PER_CLASS,
    )
    eval_images = len(subset.image_ids)
    print(f"\n[evaluation] one shared split of {eval_images:,} images "
          f"({len(subset.required_ids):,} hold a declared class); used for the "
          "anchor and every task of every arm")

    candidate_index = json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    print(f"\n[ledger] simulated on the committed frozen pool at "
          f"{bm.ANSWER_BUDGET_PER_TASK:,} oracle answers")
    simulated = simulate_arms(
        candidate_index, answer_budget=bm.ANSWER_BUDGET_PER_TASK,
        seed=bm.DEVELOPMENT_SEED,
    )
    print(runner.table(simulated))
    print("  The two coverage rows are a COST PROXY: they traverse PROB decoder "
          "space, not DINOv2.\n  They predict runtime, not selection quality.")

    opened = max(int(row["trainable_t2"]) for row in simulated)
    basis = cost_basis()
    print(f"\n[runtime] worst-case {opened} trainable images + "
          f"{arguments.replay_images} replay aliases per task")

    decision = None
    plan_rows: list[dict[str, object]] = []
    for epochs in EPOCH_LADDER:
        for candidate_images in CANDIDATE_LADDER:
            # 0.80 is the measured NMS survival on the committed pool; a gated
            # arm then embeds only the admissible share of that, which is why
            # `proposed` is roughly three times cheaper than `coreset`.
            deduplicated = candidate_images * bm.PROPOSALS_PER_IMAGE * 0.80
            per_task = {}
            for name in arm_registry.ORDER:
                spec = arm_registry.ARMS[name]
                crops = 0
                if spec.needs_semantic:
                    crops = int(deduplicated * (
                        population_module.ADMISSIBLE_SHARE if spec.gated else 1.0
                    ))
                per_task[name] = minutes(
                    basis,
                    candidate_images=candidate_images,
                    train_images=opened + arguments.replay_images,
                    eval_images=eval_images,
                    epochs=epochs,
                    batch_size=bm.BATCH_SIZE,
                    dino_crops=crops,
                )
            n_incremental = len(tasks) - 1
            session = sum(v["total"] for v in per_task.values()) * n_incremental
            session += basis["evaluate_minutes_fixed_overhead"] + eval_images / 1000.0 * basis[
                "evaluate_minutes_per_1000_images_reference_run"
            ]
            row = {
                "epochs": epochs, "candidate_images": candidate_images,
                "min_per_arm_task_min": round(min(v["total"] for v in per_task.values()), 1),
                "min_per_arm_task_max": round(max(v["total"] for v in per_task.values()), 1),
                "five_arms_hours": round(session / 60.0, 2),
                "four_arms_hours": round(
                    (session - sum(
                        v["total"] for k, v in per_task.items()
                        if k == arm_registry.ORDER[-1]
                    ) * n_incremental) / 60.0, 2),
            }
            plan_rows.append(row)
            if decision is None and row["five_arms_hours"] <= HOURS_CEILING:
                decision = ("five arms", epochs, candidate_images, row["five_arms_hours"])
            if decision is None and row["four_arms_hours"] <= HOURS_CEILING:
                decision = ("four arms", epochs, candidate_images, row["four_arms_hours"])
    print(runner.table(plan_rows))

    print("\n[decision] taken now, before any real training:")
    top = plan_rows[0]
    if top["five_arms_hours"] <= HOURS_CEILING:
        print(f"  NO REDUCTION. epochs={EPOCH_LADDER[0]}, "
              f"candidate_images={CANDIDATE_LADDER[0]}, all "
              f"{len(arm_registry.ORDER)} arms, {top['five_arms_hours']:.2f} h "
              f"<= {HOURS_CEILING:.0f} h ceiling.")
    elif top["four_arms_hours"] <= HOURS_CEILING:
        print(f"  KEEP the training schedule; run the first "
              f"{len(arm_registry.ORDER) - 1} arms of the pre-declared order in "
              f"session 1 ({top['four_arms_hours']:.2f} h) and finish "
              f"{arm_registry.ORDER[-1]!r} at the start of session 2.")
        print("  Rationale: under-training is the failure mode that produced "
              "new_class_AP50 ~ 0 in Method V3.\n  Spending a second session is "
              "recoverable; a weakened training schedule is not.")
    else:
        print(f"  DESCEND the ladder: {decision}")
    print(f"  Ladder, fixed in advance: epochs {list(EPOCH_LADDER)}, then "
          f"candidate images {list(CANDIDATE_LADDER)}.")
    print("  Never on the ladder: dropping a task, dropping a seed after seeing "
          "results, or\n  choosing arms by their numbers.")

    print("\n[reporting rules]")
    for line in bm.REPORTING:
        print(f"  - {line}")

    if arguments.json:
        bm.write_json(arguments.json, {
            "chain": [t.name for t in tasks],
            "arms": list(arm_registry.ORDER),
            "eval_images": eval_images,
            "ledger_simulation": simulated,
            "runtime": plan_rows,
            "epoch_ladder": list(EPOCH_LADDER),
            "candidate_ladder": list(CANDIDATE_LADDER),
            "hours_ceiling": HOURS_CEILING,
        })
        print(f"\nwrote {arguments.json}")


if __name__ == "__main__":
    main()
