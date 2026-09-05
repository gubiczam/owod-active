#!/usr/bin/env python
"""Re-score existing seed-0 checkpoints on a larger held-out split. NO TRAINING.

This is a **separate evaluation**, not a replacement endpoint. It writes to its
own directory, never touches
``results/full_owod_active_benchmark_v1/``, and no number it produces may be
mixed into the frozen benchmark table — a different evaluation set is a
different measurement.

**What a larger split can and cannot fix, measured before it was built.** The
current 837-image shared split already contains *every* test image holding a
``fire hydrant`` (101 objects, 86 images) or a ``stop sign`` (75, 69), because
the per-class cap of 150 never binds for them. Those two classes gain **nothing**
from a larger split; their support is the whole benchmark test set. What does
gain: ``bear`` 2 -> 71 objects (**35.5x**), the 22 declared classes 4,237 ->
18,599 (4.4x), and unknown-class objects 2,548 -> 18,182 (7.1x), which is the
support behind U-Recall, WI and A-OSE. So this experiment repairs
``mAP50_tail``, ``known_mAP50`` and the open-world metrics. It does not repair
t3/t4 new-class AP, and nothing can: there is no more test data for those two
classes.

**Only ten checkpoints exist.** ``keep_checkpoints=2`` prunes all but the newest
two per arm, so after a three-task chain each arm retains **t3 and t4** and its
t2 checkpoint is gone. t2 cannot be re-scored without retraining, which this
tool will not do.

    python tools/run_large_eval.py --prob-root /content/PROB \\
        --data-root /content/data/OWOD \\
        --results   .../results/full_owod_active_benchmark_v1 \\
        --out       .../results/large_eval_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import evaluation_subset, metrics, protocol, runner
from owl.active_selection import benchmark as bm
from owl.bridge import Bridge
from tools.materialize_pool_images import materialise

ROOT = Path(__file__).resolve().parent.parent
TEST_ARCHIVE = ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz"
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
REPLAY_INDEX = ROOT / "data" / "reference" / "t1_replay_class_counts.json"

#: ``test`` and nothing else. PROB routes a split by substring, and ``eval``
#: contains ``val`` — a split called ``large_eval`` is routed to the ``val``
#: branch, where no annotation filtering runs at all: U-Recall would read zero
#: everywhere and future-task objects would be scored as already known. A full
#: table of plausible, wrong numbers. See owl.evaluation_subset.check_split_name.
LARGE_TEST_SET = "owl_large_test"

#: Every incremental task. The default is all of them *on purpose*: t2's
#: checkpoint has been pruned in every arm, and a default that quietly omitted it
#: would hide that fact instead of reporting it.
ALL_TASKS = ("t2", "t3", "t4")


def build_split(scope: str) -> tuple[list[str], dict]:
    """The evaluation image ids, and what they contain.

    ``full``     every image in the benchmark's test archive.
    ``declared`` every test image holding at least one of the 22 declared
                 classes — cheaper, and it keeps all of ``bear``, ``traffic
                 light``, ``fire hydrant`` and ``stop sign``.
    """

    import tarfile
    from xml.etree import ElementTree

    from owl.evaluation_subset import canonical_class_name

    per: dict[str, dict[str, int]] = {}
    with tarfile.open(TEST_ARCHIVE) as handle:
        for member in handle:
            if not member.name.endswith(".xml"):
                continue
            stem = member.name.rsplit("/", 1)[-1][:-4]
            counts: dict[str, int] = {}
            for node in ElementTree.parse(handle.extractfile(member)).getroot().findall("object"):
                name = canonical_class_name(node.findtext("name", ""))
                counts[name] = counts.get(name, 0) + 1
            per[stem] = counts

    declared = set(bm.chain()[-1].known_classes)
    if scope == "full":
        chosen = sorted(per)
    elif scope == "declared":
        chosen = sorted(i for i in per if declared & set(per[i]))
    else:
        raise SystemExit(f"unknown scope {scope!r}; expected 'full' or 'declared'")
    return chosen, per


def leakage_check(image_ids: list[str]) -> dict[str, int]:
    """Nothing evaluated on may ever have been trainable. Fail closed."""

    candidate = set(json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8")))
    replay = set(json.loads(REPLAY_INDEX.read_text(encoding="utf-8")))
    chosen = set(image_ids)
    overlap = {
        "with_candidate_pool": len(chosen & candidate),
        "with_replay_pool": len(chosen & replay),
    }
    if any(overlap.values()):
        raise SystemExit(
            "LEAKAGE: the proposed evaluation split shares images with data that "
            f"is trained on — {overlap}. Refusing to evaluate."
        )
    return overlap


def surviving_checkpoints(results: Path, manifest: dict) -> list[dict]:
    """Every (arm, seed, task) whose checkpoint is still on disk."""

    found = []
    for entry in manifest["trajectories"]:
        if entry.get("status") != "COMPLETE":
            continue
        for task in bm.chain()[1:]:
            path = (results / entry["trajectory"] / f"{task.name}_{entry['arm']}"
                    / "checkpoint.pth")
            found.append({
                "trajectory": entry["trajectory"], "arm": entry["arm"],
                "seed": entry["seed"], "task": task.name,
                "checkpoint": path, "present": path.is_file(),
            })
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results", required=True, help="the frozen benchmark; read-only")
    parser.add_argument("--out", required=True, help="must differ from --results")
    parser.add_argument("--scope", default="full", choices=("full", "declared"))
    parser.add_argument("--tasks", nargs="+", default=list(ALL_TASKS))
    parser.add_argument("--detections", action="store_true",
                        help="second forward pass, for U-Recall by frequency "
                             "group. Doubles the cost; the aggregate U-Recall, "
                             "WI and A-OSE come from the metrics file without it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fetch-workers", type=int, default=32)
    parser.add_argument("--time-budget-minutes", type=float, default=None)
    parser.add_argument("--plan-only", action="store_true",
                        help="print the split, the leakage check and the cost, "
                             "then stop without evaluating")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    results, out = Path(arguments.results).resolve(), Path(arguments.out).resolve()
    if out == results:
        raise SystemExit(
            "--out must differ from --results. This is a NEW evaluation; the "
            "frozen benchmark endpoint is preserved, not replaced."
        )
    data_root = Path(arguments.data_root)

    evaluation_subset.check_split_name(LARGE_TEST_SET, purpose="test")
    image_ids, per_image = build_split(arguments.scope)
    overlap = leakage_check(image_ids)

    declared = bm.chain()[-1].known_classes
    focus = ("traffic light", "fire hydrant", "stop sign", "bear")
    current = set(evaluation_subset.from_archive(
        TEST_ARCHIVE, bm.declared_classes(), seed=bm.DEVELOPMENT_SEED,
        remainder_multiplier=bm.EVAL_REMAINDER_RATIO,
        max_per_class=bm.EVAL_MAX_PER_CLASS).image_ids)

    print("=" * 96)
    print("LARGE EVALUATION OF EXISTING CHECKPOINTS — evaluation only, no training")
    print("=" * 96)
    print(f"split name  : {LARGE_TEST_SET}  (routed by PROB on the 'test' marker)")
    print(f"scope       : {arguments.scope} — {len(image_ids):,} images "
          f"(frozen benchmark split: {len(current):,})")
    print(f"leakage     : {overlap} — zero overlap with anything trainable")
    print()
    rows = []
    for name in focus:
        cur_o = sum(per_image[i].get(name, 0) for i in current)
        cur_i = sum(1 for i in current if name in per_image[i])
        new_o = sum(per_image[i].get(name, 0) for i in image_ids)
        new_i = sum(1 for i in image_ids if name in per_image[i])
        rows.append({"class": name, "current_objects": cur_o, "current_images": cur_i,
                     "large_objects": new_o, "large_images": new_i,
                     "gain": f"{new_o / max(cur_o, 1):.1f}x"})
    print(runner.table(rows))

    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    checkpoints = [c for c in surviving_checkpoints(results, manifest)
                   if c["task"] in set(arguments.tasks)]
    live = [c for c in checkpoints if c["present"]]
    missing = [c for c in checkpoints if not c["present"]]
    print(f"\ncheckpoints : {len(live)} present, {len(missing)} pruned "
          f"(keep_checkpoints={bm.KEEP_CHECKPOINTS} retains the newest two per arm)")
    if missing:
        print("  pruned, cannot be re-scored without retraining:")
        print(runner.table([{k: c[k] for k in ("arm", "seed", "task")} for c in missing]))

    basis = json.loads((ROOT / "data" / "reference" / "gpu_cost_basis.json"
                        ).read_text(encoding="utf-8"))
    passes = 2 if arguments.detections else 1
    per_ckpt = (basis["evaluate_minutes_fixed_overhead"]
                + passes * len(image_ids) / 1000
                * basis["evaluate_minutes_per_1000_images_reference_run"])
    print(f"\ncost        : {per_ckpt:.1f} min per checkpoint "
          f"({'2 passes, detections on' if arguments.detections else '1 pass'}) "
          f"-> {len(live) * per_ckpt / 60:.2f} h for {len(live)}")
    print(f"downloads   : up to {len(set(image_ids) - current):,} test JPEGs not yet on disk")

    if arguments.plan_only:
        print("\n--plan-only: nothing evaluated.")
        return

    # ---- the split file, then the pixels --------------------------------
    subset = evaluation_subset.EvaluationSubset(
        image_ids=tuple(image_ids), required_ids=tuple(image_ids),
        sampled_ids=(), object_counts={
            name: sum(per_image[i].get(name, 0) for i in image_ids) for name in declared},
    )
    target = data_root / "ImageSets" / "OWDETR" / f"{LARGE_TEST_SET}.txt"
    if not arguments.dry_run:
        evaluation_subset.write_image_set(target, subset)
        counts = materialise(list(image_ids), data_root / "JPEGImages",
                             workers=arguments.fetch_workers)
        if counts["unreadable"]:
            raise SystemExit(
                f"{len(counts['unreadable'])} evaluation images could not be "
                "fetched. The split must be identical for every arm, so a "
                "missing image would change what an arm is scored on.")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        evaluation_subset.write_image_set(target, subset)
        jpeg = data_root / "JPEGImages"
        jpeg.mkdir(parents=True, exist_ok=True)
        for image in image_ids:
            path = jpeg / f"{image}.jpg"
            if not path.exists():
                path.write_bytes(b"\xff")

    if arguments.dry_run:
        from tools.dry_run_notebook import FakeBridge
        bridge = FakeBridge(prob_root=arguments.prob_root, data_root=data_root)
    else:
        bridge = Bridge(prob_root=Path(arguments.prob_root), data_root=data_root,
                        device=arguments.device, seed=bm.DEVELOPMENT_SEED,
                        log_dir=out / "logs")

    chain = {task.name: task for task in bm.chain()}
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    table: list[dict] = []
    per_class_rows: list[dict] = []
    groups = protocol.load_groups()

    for record in live:
        elapsed = (time.monotonic() - started) / 60
        if arguments.time_budget_minutes and elapsed >= arguments.time_budget_minutes:
            print(f"stopping cleanly at {elapsed:.0f} min; re-run to continue")
            break
        task = chain[record["task"]]
        destination = out / record["trajectory"] / f"{record['task']}_metrics.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [{record['trajectory']}/{record['task']}] evaluating on "
              f"{len(image_ids):,} images")
        path = bridge.evaluate(
            checkpoint=record["checkpoint"], test_set=LARGE_TEST_SET,
            output=destination, n_prev=task.n_prev, n_current=task.n_new,
            detections=arguments.detections,
        )
        evaluation = metrics.from_bridge_metrics(path)
        row = metrics.task_row(
            evaluation, task=task.name, new_class=task.new_class,
            groups=metrics.group_membership(task.known_classes, groups),
        )
        table.append({"arm": record["arm"], "seed": record["seed"],
                      "split": LARGE_TEST_SET, "images": len(image_ids), **row})
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            for name, value in metrics.per_class_ap50(
                    payload, class_names=task.known_classes).items():
                per_class_rows.append({
                    "arm": record["arm"], "seed": record["seed"],
                    "task": task.name, "class_name": name,
                    "group": groups.get(name, "—"),
                    "introduced": "new" if name == task.new_class else "old",
                    "AP50": value})
        except metrics.MetricsError as error:
            print(f"    per-class AP unavailable: {error}")

    if table:
        print("\n" + runner.table(table))
        _write(out / "large_eval_metrics.csv", table)
        _write(out / "large_eval_per_class_ap.csv", per_class_rows)
    bm.write_json(out / "manifest.json", {
        "experiment": "large_eval_v1",
        "note": "A SEPARATE evaluation of existing checkpoints on a larger "
                "held-out split. Not a replacement for the frozen benchmark "
                "endpoint; numbers from the two splits may not be mixed.",
        "dry_run": bool(arguments.dry_run),
        "split": {"name": LARGE_TEST_SET, "scope": arguments.scope,
                  "images": len(image_ids), "leakage": overlap,
                  "frozen_split_images": len(current)},
        "detections": bool(arguments.detections),
        "checkpoints_evaluated": [
            {k: (str(v) if isinstance(v, Path) else v) for k, v in c.items()}
            for c in live],
        "checkpoints_pruned": [
            {k: (str(v) if isinstance(v, Path) else v) for k, v in c.items()}
            for c in missing],
        "source_results": str(results),
    })
    print(f"\nwrote {out}")
    print(f"read-only on {results}: the frozen benchmark endpoint is untouched")


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


if __name__ == "__main__":
    main()
