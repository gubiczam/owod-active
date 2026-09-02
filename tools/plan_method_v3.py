#!/usr/bin/env python
"""Price Method V3 before it runs, and print the frozen protocol and criterion.

CPU only. No detector, no DINOv2, no oracle endpoint. It exists because the
runtime decision — whether the training schedule has to be reduced — must be
documented *before* real training, and because the criterion must be visible
before the launcher.

    python tools/plan_method_v3.py
    python tools/plan_method_v3.py --views /content/drive/.../views.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import evaluation_subset, method_v3, protocol, replay
from owl import exemplars as em

ROOT = Path(__file__).resolve().parent.parent
COST_BASIS = ROOT / "data" / "reference" / "gpu_cost_basis.json"
TEST_ARCHIVE = ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz"
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
REPLAY_INDEX = ROOT / "data" / "reference" / "t1_replay_class_counts.json"

#: The reduction ladder, fixed in advance. Applied uniformly to every arm and
#: seed, or not at all. Never by dropping arms and never by selecting seeds.
EPOCH_LADDER: tuple[int, ...] = (5, 3, 2, 1)

#: Above this the night is gone and the schedule must be reduced. Protocol §9.
HOURS_CEILING = 10.0


def train_minutes(images: int, epochs: int, basis: dict) -> float:
    iterations = math.ceil(images / method_v3.BATCH_SIZE) * epochs
    return (
        basis["train_minutes_fixed_overhead"]
        + basis["train_seconds_per_iteration"] * iterations / 60.0
    )


def evaluate_minutes(images: int, basis: dict, *, detections: bool) -> float:
    once = (
        basis["evaluate_minutes_fixed_overhead"]
        + basis["evaluate_minutes_per_1000_images_reference_run"] * images / 1000.0
    )
    return once * (2 if detections else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", default=None,
                        help="frozen Stage-2 view export; when given, the A*C arm "
                             "is priced from its real selection instead of being "
                             "bounded by the other arms")
    parser.add_argument("--export", default=None, help="frozen DINOv2 base export")
    parser.add_argument("--out", default=None, help="write the estimate as JSON")
    arguments = parser.parse_args()

    basis = json.loads(COST_BASIS.read_text(encoding="utf-8"))
    candidate_index = json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    replay_index = json.loads(REPLAY_INDEX.read_text(encoding="utf-8"))

    pool = method_v3.population(
        ROOT / "data" / "pool" / "sowodb_t1_frozen_pool.npz", candidate_index
    )
    print(f"[population] {pool.provenance['rows']:,} proposals on "
          f"{pool.provenance['images']:,} images  PASS")

    task = protocol.build_chain(method_v3.N_TASKS)[1]
    known_now = frozenset(task.known_classes)

    subset = evaluation_subset.from_archive(
        TEST_ARCHIVE, [task.new_class], seed=0,
        remainder_multiplier=method_v3.EVAL_REMAINDER_RATIO,
        max_per_class=method_v3.EVAL_MAX_PER_CLASS,
    )
    n_test = len(subset.image_ids)
    print(f"[evaluation] shared split {n_test:,} images "
          f"({len(subset.required_ids)} required + {len(subset.sampled_ids)} remainder)")

    spec = replay.ARMS[method_v3.REPLAY_ARM]
    eligible = em.enumerate_pool(replay_index, frozenset(task.previous_classes))
    demand = replay.allocate(
        em.capacities(eligible), total=int(spec["total"]), alpha=float(spec["alpha"])
    )
    chosen = em.select(eligible, demand, incumbent=(), reallocate=False, seed=0)
    replay_images = len({item.image_id for item in chosen})
    print(f"[replay] {method_v3.REPLAY_ARM}: {len(chosen)} objects on "
          f"{replay_images} alias images, identical for all four arms")

    consistency = None
    if arguments.views and arguments.export:
        consistency, provenance = method_v3.consistency_values(
            pool, base_export=arguments.export, views_export=arguments.views
        )
        print(f"[C] {provenance['definition']}  mean {provenance['mean']:.4f}")

    # ---- what each arm opens, so the training cost is real rather than assumed
    per_arm: dict[str, dict[str, int]] = {}
    for arm in method_v3.ARMS:
        if arm == "A*C" and consistency is None:
            continue
        picked = method_v3.select_for_arm(
            pool.candidates, arm, 0, consistency=consistency
        )
        opened = [str(v) for v in picked.images(pool.candidates)]
        trainable = [
            image for image in opened
            if any(name in known_now for name in candidate_index.get(image, {}))
        ]
        per_arm[arm] = {
            "images_opened": len(opened),
            "images_trainable": len(trainable),
            "training_images": len(trainable) + replay_images,
        }

    if "A*C" not in per_arm:
        # No view export here: bound A*C by the arms that are known, so the
        # estimate is a range rather than a guess presented as a number.
        bounds = [row["training_images"] for row in per_arm.values()]
        per_arm["A*C"] = {
            "images_opened": -1, "images_trainable": -1,
            "training_images": round(sum(bounds) / len(bounds)),
            "note": "estimated as the mean of the priced arms; needs --views to be exact",
        }

    print()
    print("arm      opened  trainable  +replay  train(min)  eval(min)")
    estimate: dict[str, object] = {"per_arm": per_arm, "test_images": n_test}
    for arm in method_v3.ARMS:
        row = per_arm[arm]
        t = train_minutes(row["training_images"], method_v3.EPOCHS, basis)
        e = evaluate_minutes(n_test, basis, detections=True)
        row["train_minutes"] = round(t, 1)
        row["evaluate_minutes"] = round(e, 1)
        print(f"{arm:8s} {row['images_opened']:6d} {row['images_trainable']:10d} "
              f"{row['training_images']:8d} {t:11.1f} {e:10.1f}")

    print()
    print("epochs   12 trajectories (h)   verdict")
    ladder: list[dict[str, object]] = []
    for epochs in EPOCH_LADDER:
        total = sum(
            len(method_v3.SEEDS) * (
                train_minutes(per_arm[arm]["training_images"], epochs, basis)
                + evaluate_minutes(n_test, basis, detections=True)
            )
            for arm in method_v3.ARMS
        )
        anchor = evaluate_minutes(n_test, basis, detections=False)
        hours = (total + anchor) / 60.0
        ladder.append({"epochs": epochs, "hours": round(hours, 2)})
        print(f"{epochs:6d}   {hours:19.2f}   "
              f"{'fits' if hours <= HOURS_CEILING else 'over the ceiling'}")

    chosen_epochs = next(
        (row["epochs"] for row in ladder if row["hours"] <= HOURS_CEILING), 1
    )
    estimate |= {
        "ladder": ladder,
        "hours_ceiling": HOURS_CEILING,
        "epochs_chosen": chosen_epochs,
        "epochs_frozen_in_protocol": method_v3.EPOCHS,
        "anchor_evaluations": 1,
        "trajectories": len(method_v3.trajectories()),
    }

    print()
    if chosen_epochs != method_v3.EPOCHS:
        raise SystemExit(
            f"The cost basis says epochs={method_v3.EPOCHS} needs more than "
            f"{HOURS_CEILING} h; the protocol's reduction ladder chooses "
            f"epochs={chosen_epochs}. Update the protocol and owl.method_v3 "
            "deliberately — this tool does not reduce the schedule behind your back."
        )
    print(f"DECISION: no reduction. epochs={method_v3.EPOCHS} "
          f"(the established schedule) fits at "
          f"{ladder[0]['hours']:.2f} h of GPU work for all 12 trajectories.")
    print("Reduction ladder, had it been needed, applied uniformly to every arm "
          f"and seed: epochs {' -> '.join(str(e) for e in EPOCH_LADDER)}")

    print()
    print(method_v3.CRITERION.statement())

    if arguments.out:
        method_v3.write_json(arguments.out, estimate)
        print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
