#!/usr/bin/env python
"""Run the twelve Method V3 trajectories. Resumable, fail-visible, no branching.

Four arms x three seeds, in a fixed order, all attempted. Every trajectory gets
its own directory under ``--out``; a completed one is skipped on its
``result.json``, an interrupted one is re-run, a corrupt one stops the run.

    python tools/run_method_v3.py \\
        --prob-root /content/PROB --data-root /content/data/OWOD \\
        --checkpoint /content/drive/MyDrive/OWL/checkpoints/SOWODB/t1.pth \\
        --export  /content/drive/MyDrive/OWL/features/dinov2_vitb14_method_v2_v1.npz \\
        --views   /content/drive/MyDrive/OWL/features/dinov2_vitb14_stage2_views_v1.npz \\
        --out     /content/drive/MyDrive/OWL/results/method_v3_selection_transfer

``--dry-run`` walks the whole orchestration with PROB stubbed out, which is what
the tests and the notebook's preflight use: it proves the schedule, the
resumability and the manifest without a GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import method_v3, protocol
from owl.bridge import Bridge
from tools.materialize_pool_images import materialise
from tools.prepare_method_v3_data import shared_test_split

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "pool" / "sowodb_t1_frozen_pool.npz"
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
REPLAY_INDEX = ROOT / "data" / "reference" / "t1_replay_class_counts.json"


class StubBridge:
    """PROB replaced by deterministic synthetic files. ``--dry-run`` only.

    It exists so the whole orchestration — the schedule, the selection, the
    annotation accounting, the replay memory, the resume logic, the manifest, the
    tables and the criterion — can be proven on a laptop without a GPU and
    without a detector endpoint. Its numbers are a hash of (arm, seed) and mean
    nothing.

    A stubbed trajectory can never be mistaken for a real one: ``dry_run`` is
    part of :func:`owl.method_v3.fingerprint`, so a real run pointed at a stubbed
    directory refuses to resume it.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def cost_report(self) -> dict[str, float]:
        return {"total": 0.0}

    def _synthetic(self, seed_text: str) -> list[float]:
        generator = np.random.default_rng(
            abs(hash(seed_text)) % (2**32)
        )
        return [float(value) for value in generator.uniform(0.0, 40.0, size=83)]

    def _detections(self, path: Path, generator) -> Path:
        """A minimal ``daowod_detections_v1`` artefact, so the grouped-U-Recall
        decomposition is exercised rather than skipped in a dry run."""

        from owl.protocol import CLASS_ORDER, N_TASK1

        unknown_pool = list(CLASS_ORDER[N_TASK1 + 1:])
        truth, found = [], []
        for index in range(200):
            image = f"{index:012d}"
            name = unknown_pool[index % len(unknown_pool)]
            box = [10.0, 10.0, 60.0, 60.0]
            truth.append({"image_id": image, "class_name": name, "box": box})
            if generator.random() < 0.5:
                found.append({"image_id": image, "class_name": "unknown",
                              "score": 0.9, "box": box})
        method_v3.write_json(path, {
            "schema": "daowod_detections_v1",
            "dry_run": True,
            "unknown_class_name": "unknown",
            "ground_truth": truth,
            "detections": found,
        })
        return path

    def evaluate(self, *, checkpoint, test_set, output, n_prev, n_current,
                 detections=True, batch_size=4):
        output = Path(output)
        if output.exists():
            return output
        vector = self._synthetic(f"{output.parent.name}|{checkpoint}|{test_set}")
        artefact = None
        if detections:
            artefact = str(self._detections(
                output.with_name(output.stem + "_detections.json"),
                np.random.default_rng(abs(hash(str(output))) % (2**32)),
            ))
        method_v3.write_json(output, {
            "dry_run": True,
            "detections_path": artefact,
            "known_AP50": float(np.mean(vector[2:2 + n_prev + n_current])),
            "previous_known_AP50": (
                float(np.mean(vector[2:2 + n_prev])) if n_prev else None),
            "current_known_AP50": (
                float(np.mean(vector[2 + n_prev:2 + n_prev + n_current]))
                if n_current else None),
            "unknown_AP50": vector[-1],
            "U_Recall": vector[-1],
            "WI": 0.0, "A_OSE": 0.0,
            "coco_eval_bbox": vector,
            "official_metrics": {},
        })
        self.calls.append({"verb": "evaluate", "output": str(output)})
        return output

    def train(self, labelled_ids, *, previous_checkpoint, output_checkpoint,
              output_dir, n_prev, n_current, test_set, replay_ids=(),
              supervision_mode="ft", epochs=5, learning_rate=2e-4, batch_size=2):
        output_checkpoint = Path(output_checkpoint)
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if not output_checkpoint.exists():
            output_checkpoint.write_text("dry-run stub checkpoint", encoding="utf-8")
        self.calls.append({
            "verb": "train", "images": len(labelled_ids),
            "replay": len(replay_ids), "epochs": epochs,
            "supervision_mode": supervision_mode,
        })
        return output_checkpoint


def git_sha(path: Path) -> str:
    probe = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path,
        capture_output=True, text=True, check=False,
    )
    return probe.stdout.strip() if probe.returncode == 0 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export", required=True, help="frozen DINOv2 base export")
    parser.add_argument("--views", required=True, help="frozen Stage-2 view export")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fetch-workers", type=int, default=32)
    parser.add_argument("--time-budget-minutes", type=float, default=None,
                        help="stop cleanly between trajectories once this much GPU "
                             "time has been spent; the next run resumes")
    parser.add_argument("--dry-run", action="store_true",
                        help="stub PROB out and walk the orchestration only")
    # Debugging aids, and deliberately not a way to shape the result: the
    # verdict is computed by tools/summarize_method_v3.py, which refuses to
    # produce one unless A and A*C are present at every seed. Running a subset
    # cannot terminate an arm or select a seed out of the design.
    parser.add_argument("--only-arm", default=None, choices=list(method_v3.ARMS),
                        help="run one arm only (debugging; the verdict still "
                             "needs the complete 4x3 design)")
    parser.add_argument("--only-seed", type=int, default=None,
                        choices=list(method_v3.SEEDS),
                        help="run one seed only (debugging; same restriction)")
    arguments = parser.parse_args()

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(arguments.data_root)

    # ---- the frozen inputs, verified before any GPU time is spent ---------
    candidate_index = json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    replay_index = json.loads(REPLAY_INDEX.read_text(encoding="utf-8"))
    pool = method_v3.population(POOL, candidate_index)
    print(f"[population] {pool.provenance['rows']:,} proposals on "
          f"{pool.provenance['images']:,} images  PASS")

    consistency, consistency_provenance = method_v3.consistency_values(
        pool, base_export=arguments.export, views_export=arguments.views
    )
    print(f"[C] {consistency_provenance['definition']}")
    print(f"[C] mean {consistency_provenance['mean']:.4f}, "
          f"range [{consistency_provenance['min']:.4f}, "
          f"{consistency_provenance['max']:.4f}]  read verbatim, no DINOv2 pass")

    test_set, test_images = shared_test_split(data_root, write=not arguments.dry_run)
    print(f"[evaluation] {test_set} — {test_images:,} images, identical everywhere")

    print()
    print(method_v3.annotation_protocol(method_v3.TrajectoryInputs(
        pool=pool, candidate_index=candidate_index, replay_index=replay_index,
        replay_root=data_root, start_checkpoint=Path(arguments.checkpoint),
        test_set=test_set,
    )))
    print()
    print(method_v3.CRITERION.statement())
    print()

    if arguments.dry_run:
        bridge = StubBridge()
        (out / "DRY_RUN").write_text(
            "Every result under this directory came from tools/run_method_v3.py "
            "--dry-run: PROB was stubbed out and the numbers are synthetic.\n",
            encoding="utf-8",
        )
        print("[dry run] PROB is stubbed out; every detector number below is "
              "synthetic and the directory is marked DRY_RUN")
    else:
        bridge = Bridge(
            prob_root=Path(arguments.prob_root), data_root=data_root,
            device=arguments.device, num_workers=arguments.num_workers,
            log_dir=out / "logs",
        )
        bridge.check()
        (out / "DRY_RUN").unlink(missing_ok=True)

    def placeholder_images(image_ids):
        """``--dry-run`` only: touch a placeholder so the orchestration can walk.

        The alias materialisation links a JPEG per exemplar source, so the path
        cannot be skipped. Nothing reads the pixels in a dry run — PROB is
        stubbed — so a one-byte file proves the plumbing without the network.
        """

        jpeg = data_root / "JPEGImages"
        jpeg.mkdir(parents=True, exist_ok=True)
        wanted = [str(value) for value in image_ids]
        for name in wanted:
            target = jpeg / f"{name}.jpg"
            if not target.exists():
                target.write_bytes(b"\0")
        return list(dict.fromkeys(wanted))

    def prepare_images(image_ids):
        """Fetch what is missing and return only the ids that really arrived.

        The materialiser reports counts; what the runner needs is the surviving
        ids, so the end state on disk is re-checked rather than the download
        return codes trusted.
        """

        wanted = [str(value) for value in image_ids]
        jpeg = data_root / "JPEGImages"
        counts = materialise(wanted, jpeg, workers=arguments.fetch_workers)
        lost = set(counts["unreadable"])
        if lost:
            print(f"  [images] {len(lost)} of {len(set(wanted))} could not be "
                  f"fetched; first: {sorted(lost)[:5]}")
        return [name for name in dict.fromkeys(wanted) if name not in lost]

    inputs = method_v3.TrajectoryInputs(
        pool=pool,
        candidate_index=candidate_index,
        replay_index=replay_index,
        replay_root=data_root,
        start_checkpoint=Path(arguments.checkpoint),
        test_set=test_set,
        consistency=consistency,
        anchor_metrics=None,
        prepare_images=placeholder_images if arguments.dry_run else prepare_images,
        provenance={
            "owl_commit": git_sha(ROOT),
            "prob_commit": git_sha(Path(arguments.prob_root)),
            "consistency": consistency_provenance,
            "population": pool.provenance,
        },
    )

    # ---- the anchor once, then reused by all twelve -----------------------
    anchor_task = protocol.build_chain(method_v3.N_TASKS)[0]
    shared_anchor = out / "anchor_metrics.json"
    bridge.evaluate(
        checkpoint=Path(arguments.checkpoint), test_set=test_set,
        output=shared_anchor, n_prev=anchor_task.n_prev,
        n_current=anchor_task.n_new, detections=False,
    )
    if shared_anchor.exists():
        inputs.anchor_metrics = shared_anchor

    # ---- the twelve trajectories ------------------------------------------
    scheduled = [
        (arm, seed) for arm, seed in method_v3.trajectories()
        if (arguments.only_arm is None or arm == arguments.only_arm)
        and (arguments.only_seed is None or seed == arguments.only_seed)
    ]
    print(f"[schedule] {len(scheduled)} trajectories: "
          + ", ".join(f"{arm}/s{seed}" for arm, seed in scheduled))

    status: dict[str, str] = {}
    rows: list[dict] = []
    started = bridge.cost_report()["total"]
    for arm, seed in scheduled:
        name = method_v3.trajectory_name(arm, seed)
        elapsed = bridge.cost_report()["total"] - started
        if (arguments.time_budget_minutes is not None
                and elapsed >= arguments.time_budget_minutes
                and method_v3.load_trajectory(out / name) is None):
            remaining = [
                method_v3.trajectory_name(a, s)
                for a, s in scheduled[scheduled.index((arm, seed)):]
            ]
            print(f"Stopping cleanly: {elapsed:.0f} min spent. "
                  f"Not run: {remaining}. Run again to resume.")
            for other in remaining:
                status[other] = "not run"
            break
        row = method_v3.run_trajectory(
            bridge, arm, seed, workspace=out / name, inputs=inputs,
            dry_run=arguments.dry_run,
        )
        status[name] = row.get("status", method_v3.STATUS_COMPLETE)
        rows.append(row)

    for arm, seed in method_v3.trajectories():
        status.setdefault(method_v3.trajectory_name(arm, seed), "not scheduled")

    method_v3.write_json(out / "manifest.json", method_v3.manifest(
        owl_sha=git_sha(ROOT),
        prob_sha=git_sha(Path(arguments.prob_root)),
        checkpoint=arguments.checkpoint,
        pool=pool,
        consistency_provenance=consistency_provenance,
        test_set=test_set,
        test_images=test_images,
        scheduled=scheduled,
        completed=status,
        dry_run=arguments.dry_run,
        runtime_estimate={
            "gpu_minutes_this_run": round(
                bridge.cost_report()["total"] - started, 1),
            "per_verb": {k: round(v, 1) for k, v in bridge.cost_report().items()},
        },
    ))

    complete = sum(1 for value in status.values() if value == method_v3.STATUS_COMPLETE)
    print()
    print(f"[manifest] {out / 'manifest.json'}")
    print(f"[status] {complete} of {len(method_v3.trajectories())} trajectories complete")
    if complete < len(method_v3.trajectories()):
        print("Run again to resume the rest; nothing completed is repeated.")
    else:
        print("ALL 12 TRAJECTORIES COMPLETE")


if __name__ == "__main__":
    main()
