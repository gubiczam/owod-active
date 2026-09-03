#!/usr/bin/env python
"""Run Benchmark V1: every arm, every seed, the whole chain, resumable.

One workspace per ``(arm, seed)``. Inside it :func:`owl.runner.run_chain` walks
t2 -> t3 -> t4, fine-tuning each task from the previous task's own checkpoint,
and writes ``state.json`` and ``metrics.json`` per task. A killed session resumes
at the task it died on; a *finished* trajectory is skipped entirely.

The arms run in the pre-declared order ``owl.active_selection.arms.ORDER``, so a
session that runs out of runtime completes a prefix of it and the next session
continues. Nothing is dropped because of what its numbers were.

    python tools/run_full_owod_benchmark.py \\
        --prob-root /content/PROB --data-root /content/data/OWOD \\
        --checkpoint /content/drive/MyDrive/OWL/checkpoints/SOWODB/t1.pth \\
        --ref-t1 /content/drive/MyDrive/OWL/features/ref_t1_dinov2_vitb14_cap1000_v1.npz \\
        --out /content/drive/MyDrive/OWL/results/full_owod_active_benchmark_v1 \\
        --seeds 0 --time-budget-minutes 600
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import metrics, runner
from owl.active_selection import arms as arm_registry
from owl.active_selection import benchmark as bm
from owl.bridge import PROB_REPOSITORY, Bridge
from tools.materialize_pool_images import materialise
from tools.prepare_full_owod_benchmark import shared_test_split

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
REPLAY_INDEX = ROOT / "data" / "reference" / "t1_replay_class_counts.json"


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def synthetic_features(path, image_ids, boxes, jpeg_dir, **_):
    """Deterministic stand-in for the DINOv2 pass. ``--dry-run`` only.

    Unit-norm, seeded on the rows it describes, so the traversal is exercised
    end to end and is reproducible — but it is noise, and a dry-run manifest
    says ``dry_run: true`` so its selections can never be read as results.
    """

    import numpy as np

    from owl.active_selection import semantic

    fingerprint = semantic.row_fingerprint(image_ids, boxes)
    generator = np.random.default_rng(int(fingerprint[:8], 16))
    features = generator.normal(size=(len(image_ids), 32)).astype(np.float32)
    return features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True), 1e-9
    )


def empty_reference(_path):
    """No labelled reference. ``--dry-run`` only.

    With an empty reference every distance is infinite and the traversal falls
    through to its admissibility tie-break, which is the branch a real run takes
    at its very first pick — so the dry run exercises it rather than skipping it.
    """

    import numpy as np

    return np.zeros((0, 32), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="the t1 anchor")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref-t1", default=None,
                        help="frozen balanced task-1 DINOv2 reference; required "
                             "by the coverage arms")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fetch-workers", type=int, default=32)
    parser.add_argument("--dino-batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[bm.DEVELOPMENT_SEED])
    parser.add_argument("--arms", nargs="+", default=list(arm_registry.ORDER),
                        choices=list(arm_registry.ARMS),
                        help="defaults to the pre-declared order; naming a "
                             "subset is for debugging and for finishing a "
                             "session, never for choosing by results")
    parser.add_argument("--time-budget-minutes", type=float, default=None,
                        help="stop cleanly before Colab takes the runtime away")
    parser.add_argument("--dry-run", action="store_true",
                        help="stub the detector and the semantic pass; proves "
                             "the orchestration on a laptop")
    arguments = parser.parse_args()

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(arguments.data_root)
    jpeg = data_root / "JPEGImages"

    report = bm.check_protocol()
    print(f"[protocol] {report['fields']} frozen fields agree with "
          f"{Path(report['path']).name}")
    print(f"[endpoints] {bm.ENDPOINTS.statement()}")
    for line in bm.REPORTING:
        print(f"  ! {line}")

    candidate_index = json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    replay_index = json.loads(REPLAY_INDEX.read_text(encoding="utf-8"))
    test_set, subset = shared_test_split(data_root, write=True)
    print(f"[evaluation] shared split {test_set}: {len(subset.image_ids):,} images")

    wanted = [a for a in arm_registry.ORDER if a in set(arguments.arms)]
    needs_reference = any(arm_registry.ARMS[a].needs_semantic for a in wanted)
    if needs_reference and not arguments.dry_run and arguments.ref_t1 is None:
        raise SystemExit(
            "The coverage arms measure distance to what is already labelled and "
            "--ref-t1 was not given. Pass the frozen "
            "ref_t1_dinov2_vitb14_cap1000_v1.npz export, or run without "
            f"{[a for a in wanted if arm_registry.ARMS[a].needs_semantic]}."
        )

    if arguments.dry_run:
        from tools.dry_run_notebook import FakeBridge

        bridge_for = lambda seed: FakeBridge(
            prob_root=arguments.prob_root, data_root=data_root,
        )
        features_for = synthetic_features
        reference_for = empty_reference
        print("[dry run] PROB and DINOv2 are stubbed. No number below is a result.")
    else:
        from owl.active_selection import semantic

        def bridge_for(seed: int) -> Bridge:
            # Method V3's audit found `--seed` left at 0 for all twelve of its
            # trajectories, because the launcher never passed it. Passed here.
            return Bridge(
                prob_root=Path(arguments.prob_root), data_root=data_root,
                device=arguments.device, num_workers=arguments.num_workers,
                seed=seed, log_dir=out / "logs",
            )

        features_for = semantic.cached
        reference_for = semantic.reference_from_ref_t1

    def prepare_images(image_ids):
        counts = materialise(
            [str(v) for v in image_ids], jpeg, workers=arguments.fetch_workers
        )
        lost = set(counts["unreadable"])
        return [str(v) for v in dict.fromkeys(str(i) for i in image_ids) if v not in lost]

    if arguments.dry_run:
        def prepare_images(image_ids):
            out_ids = []
            jpeg.mkdir(parents=True, exist_ok=True)
            for image_id in dict.fromkeys(str(i) for i in image_ids):
                target = jpeg / f"{image_id}.jpg"
                if not target.exists():
                    target.write_bytes(b"\xff")
                out_ids.append(image_id)
            return out_ids

    # ---- the anchor, evaluated once and shared ---------------------------
    #
    # run_chain evaluates its start checkpoint into its own workspace, so five
    # arms would pay for the identical evaluation five times. It is cached on the
    # output path existing, so one real evaluation copied into each workspace
    # skips the other four.
    chain = bm.chain()
    # The shared split's pixels must be on disk before anything is evaluated,
    # and /content does not survive a Colab session even though Drive does. A
    # real run finds them already fetched and this is a validating no-op.
    present = prepare_images(subset.image_ids)
    if len(present) != len(subset.image_ids):
        raise SystemExit(
            f"{len(subset.image_ids) - len(present)} of {len(subset.image_ids)} "
            "shared evaluation images are not on disk. The split is frozen, so a "
            "missing test image changes what every arm is scored on. Run "
            "tools/prepare_full_owod_benchmark.py first."
        )
    anchor_path = out / "anchor_metrics.json"
    if not anchor_path.exists():
        bridge_for(bm.DEVELOPMENT_SEED).evaluate(
            checkpoint=Path(arguments.checkpoint), test_set=test_set,
            output=anchor_path, n_prev=chain[0].n_prev, n_current=chain[0].n_new,
            detections=False,
        )
    anchor = metrics.from_bridge_metrics(anchor_path)
    print(f"[anchor] {chain[0].n_current} known classes, "
          f"mAP50 {anchor.known_map50:.2f} — every arm's t2 forgetting starts here")

    started = time.monotonic()
    trajectories: list[dict[str, object]] = []
    for seed in arguments.seeds:
        for arm in wanted:
            name = bm.trajectory_name(arm, seed)
            workspace = out / name
            workspace.mkdir(parents=True, exist_ok=True)
            # run_chain evaluates its start checkpoint into its own workspace and
            # caches on the file existing, so seeding each workspace with the one
            # shared anchor skips four identical evaluations.
            local_anchor = workspace / "anchor_metrics.json"
            if not local_anchor.exists():
                shutil.copyfile(anchor_path, local_anchor)

            elapsed = (time.monotonic() - started) / 60.0
            left = (
                None if arguments.time_budget_minutes is None
                else arguments.time_budget_minutes - elapsed
            )
            if left is not None and left <= 0:
                print(f"[session] {elapsed:.0f} min used; not started: {name}. "
                      "Re-run to continue — it resumes.")
                trajectories.append({"trajectory": name, "arm": arm, "seed": seed,
                                     "status": "NOT_STARTED"})
                continue

            print("=" * 78)
            print(f"[{name}] {arm_registry.ARMS[arm].description}")
            print("=" * 78)
            config = bm.cycle_config(arm, seed)
            selector = bm.make_selector(
                arm,
                candidate_index=candidate_index,
                jpeg_dir=jpeg,
                ref_t1=arguments.ref_t1 or ("dry-run" if arguments.dry_run else None),
                device=arguments.device,
                batch_size=arguments.dino_batch_size,
                features_for=features_for,
                reference_for=reference_for,
            )
            budget_for_arm = (
                None if left is None
                else min(left, bm.ARM_TIME_BUDGET_MINUTES)
            )
            try:
                results = runner.run_chain(
                    bridge_for(seed), config,
                    workspace=workspace,
                    candidate_index=candidate_index,
                    start_checkpoint=Path(arguments.checkpoint),
                    test_set=test_set,
                    chain=chain,
                    time_budget_minutes=budget_for_arm,
                    prepare_images=prepare_images,
                    replay_index=replay_index,
                    replay_root=data_root,
                    selector=selector,
                )
            except Exception as error:                    # noqa: BLE001
                # A failed trajectory is visibly FAILED. It is never left to look
                # like a completed one, and the session continues so the other
                # arms are not lost with it.
                print(f"[{name}] FAILED: {type(error).__name__}: {error}")
                bm.write_json(workspace / "FAILED.json", {
                    "trajectory": name, "error": f"{type(error).__name__}: {error}",
                })
                trajectories.append({"trajectory": name, "arm": arm, "seed": seed,
                                     "status": "FAILED",
                                     "error": f"{type(error).__name__}: {error}"})
                continue

            rows = [row.flat() for row in results]
            _write_rows(rows, workspace / "results.csv")
            complete = len(rows) == len(chain) - 1
            trajectories.append({
                "trajectory": name, "arm": arm, "seed": seed,
                "status": "COMPLETE" if complete else "INCOMPLETE",
                "tasks": [row["task"] for row in rows],
                "dry_run": bool(arguments.dry_run),
                "known_mAP50_final": rows[-1].get("known_mAP50") if rows else None,
            })
            print(f"[{name}] {'complete' if complete else 'INCOMPLETE'}: "
                  f"{len(rows)} of {len(chain) - 1} tasks, "
                  f"{(time.monotonic() - started) / 60.0:.0f} min into the session")

    # Merge, never overwrite. Session 2 runs the arms session 1 did not reach,
    # and a manifest rewritten from this session's arms alone would erase the
    # record of the ones already finished — and with it the summariser's ability
    # to compute any contrast between them.
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged = {row["trajectory"]: row for row in previous.get("trajectories", [])}
        merged.update({row["trajectory"]: row for row in trajectories})
        trajectories = [merged[name] for name in sorted(merged)]

    checkpoint = Path(arguments.checkpoint)
    bm.write_json(manifest_path, bm.manifest(
        trajectories=trajectories,
        owl_commit=git_sha(ROOT),
        prob_commit=git_sha(Path(arguments.prob_root)),
        prob_repository=PROB_REPOSITORY,
        checkpoint=str(checkpoint),
        checkpoint_sha256=(
            bm.sha256(checkpoint) if checkpoint.is_file() and not arguments.dry_run
            else None
        ),
        test_set=test_set,
        test_images=len(subset.image_ids),
        dry_run=bool(arguments.dry_run),
    ))
    shutil.copyfile(bm.PROTOCOL_PATH, out / bm.PROTOCOL_PATH.name)
    print(f"\n[manifest] {manifest_path}")
    print(runner.table([
        {k: v for k, v in row.items() if k != "tasks"} for row in trajectories
    ]))
    failed = [r for r in trajectories if r["status"] != "COMPLETE"]
    if failed:
        print(f"\n{len(failed)} of {len(trajectories)} trajectories are not "
              "complete. Re-run to continue; they resume.")


def _write_rows(rows, path: Path) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = Path(path).with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


if __name__ == "__main__":
    main()
