"""Score the T1 checkpoint on a finished chain's own evaluation split.

Per-class forgetting needs a reference the chain never produced: the completed
``random__none`` baseline predates the anchor evaluation that ``run_chain`` now
performs, so its workspace has no ``anchor_metrics.json`` and every
``anchor − final`` in the analysis has nothing to subtract from.

This runs **one evaluation and nothing else**. It does not train, does not touch
a checkpoint, does not write into any task directory, and refuses to overwrite
an anchor that already exists. The evaluation is staged in a temporary data
root, so the only persistent file it adds is
``<workspace>/anchor_metrics.json`` — the same name, from the same bridge verb,
with the same arguments ``run_chain`` would have used.

    python tools/evaluate_anchor.py \\
        --workspace  /content/drive/MyDrive/OWL/work/random__none \\
        --checkpoint /content/drive/MyDrive/OWL/checkpoints/SOWODB/t1.pth \\
        --prob-root  /content/PROB \\
        --data-root  /content/data/OWOD

**Why the result is protocol-identical, and how that is checked rather than
asserted.** The evaluation split is a deterministic function of the committed
annotation archive, the declared classes of the chain, and three numbers
(``seed``, ``max_per_class``, ``remainder_multiplier``). Rebuilding it with the
same inputs reproduces it exactly. But rather than trust that, the tool compares
the rebuilt split against the ``image_count`` the finished chain recorded in its
own detections artefact, and against the image ids that artefact's ground truth
mentions. If the rebuilt split is not the one the chain was scored on, it stops
before spending any GPU time.

The class counts are the anchor's own: ``prev = 0``, ``current = 19``. PROB
slices its aggregates at exactly those, and reports no ``PK_AP50`` when nothing
has been introduced yet — which is why the anchor's metrics file carries a null
there and the reader must not read that as a missing field.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import bridge, evaluation_subset, metrics, protocol

DEFAULT_ARCHIVE = ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz"


def recorded_splits(workspace: Path, selection_arm: str) -> list[dict]:
    """What every finished task says it was scored on."""

    found: list[dict] = []
    for task_dir in sorted(workspace.glob(f"t*_{selection_arm}")):
        payload_path = task_dir / "metrics.json"
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        artefact = payload_path.with_name(f"{payload_path.stem}_detections.json")
        if not artefact.exists():
            continue
        detections = json.loads(artefact.read_text(encoding="utf-8"))
        ground_truth = detections.get("ground_truth", ())
        found.append({
            "task": task_dir.name,
            "metrics_test_set": payload.get("test_set"),
            "test_set": detections.get("test_set"),
            "dataset": detections.get("dataset"),
            "image_count": detections.get("image_count"),
            "ground_truth_images": {
                str(record["image_id"]) for record in ground_truth
            },
            "ground_truth_image_order": tuple(dict.fromkeys(
                str(record["image_id"]) for record in ground_truth)),
            "class_names": detections.get("class_names"),
            "n_prev": detections.get("previous_introduced_classes"),
            "n_current": detections.get("current_introduced_classes"),
        })
    return found


def archive_annotations(archive: Path, image_ids: set[str]) -> dict[str, bytes]:
    """The exact committed XML bytes for the selected images."""

    found: dict[str, bytes] = {}
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            image_id = Path(member.name).stem
            if not member.isfile() or not member.name.endswith(".xml") \
                    or image_id not in image_ids:
                continue
            source = handle.extractfile(member)
            if source is not None:
                found[image_id] = source.read()
    return found


def verify_annotations(archive: Path, data_root: Path, image_ids: set[str]) -> str:
    """Prove that PROB will read the same XMLs used to construct the split."""

    committed = archive_annotations(archive, image_ids)
    missing_archive = image_ids - set(committed)
    if missing_archive:
        return f"the archive is missing {len(missing_archive)} selected annotations"
    missing_disk: list[str] = []
    changed: list[str] = []
    for image_id, expected in committed.items():
        path = data_root / "Annotations" / f"{image_id}.xml"
        try:
            actual = path.read_bytes()
        except OSError:
            missing_disk.append(image_id)
            continue
        if actual != expected:
            changed.append(image_id)
    if missing_disk:
        return f"data root is missing {len(missing_disk)} selected annotations"
    if changed:
        return f"{len(changed)} selected annotations differ from the committed archive"
    return ""


def make_staging_data_root(source: Path, target: Path, subset) -> Path:
    """A temporary PROB data root: source data, private ImageSets state."""

    target.mkdir(parents=True)
    for name in ("Annotations", "JPEGImages"):
        origin = (source / name).resolve()
        if not origin.is_dir():
            raise FileNotFoundError(f"Missing {origin}")
        (target / name).symlink_to(origin, target_is_directory=True)
    evaluation_subset.write_image_set(
        target / "ImageSets" / "OWDETR" /
        f"{evaluation_subset.SHARED_TEST_SET}.txt", subset)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, type=Path,
                        help="the finished run directory, e.g. .../work/random__none")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="the T1 checkpoint the chain started from")
    parser.add_argument("--prob-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="PROB's --data-root: holds Annotations/, JPEGImages/, ImageSets/")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    # These three are notebook parameters rather than CycleConfig fields, so they
    # cannot be read back from config.json. They default to the notebook's own
    # values and the rebuilt split is verified against the chain's record below.
    parser.add_argument("--eval-max-per-class", type=int, default=150)
    parser.add_argument("--eval-remainder-ratio", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true",
                        help="rebuild and verify the split, then stop before PROB")
    arguments = parser.parse_args()

    workspace = arguments.workspace
    if not workspace.is_dir():
        print(f"error: {workspace} is not a directory", file=sys.stderr)
        return 1

    output = workspace / "anchor_metrics.json"
    if output.exists():
        print(f"{output} already exists — nothing to do. Delete it deliberately if "
              "you want it recomputed.")
        return 0
    if not arguments.checkpoint.is_file():
        print(f"error: checkpoint {arguments.checkpoint} is missing", file=sys.stderr)
        return 1
    if not arguments.archive.is_file():
        print(f"error: annotation archive {arguments.archive} is missing", file=sys.stderr)
        return 1

    stamp = workspace / "config.json"
    if not stamp.exists():
        print(f"error: {stamp} is missing, so the chain's own settings cannot be "
              "read and the split cannot be reproduced.", file=sys.stderr)
        return 1
    config = json.loads(stamp.read_text(encoding="utf-8"))
    n_tasks = int(config["n_tasks"])
    seed = int(config["seed"])
    selection_arm = str(config["arm"])

    chain = protocol.build_chain(n_tasks)
    anchor_task = chain[0]
    declared = [task.new_class for task in chain[1:]]

    print(f"workspace      {workspace}")
    print(f"chain          n_tasks={n_tasks}, arm={selection_arm}, seed={seed}")
    print(f"declared       {declared}")

    subset = evaluation_subset.from_archive(
        arguments.archive, declared, seed=seed,
        remainder_multiplier=arguments.eval_remainder_ratio,
        max_per_class=arguments.eval_max_per_class,
    )
    rebuilt = set(subset.image_ids)
    print(f"rebuilt split  {len(rebuilt)} images "
          f"(max_per_class={arguments.eval_max_per_class}, "
          f"remainder={arguments.eval_remainder_ratio})")

    # ---- is this the split the chain was actually scored on? ---------------
    recorded = recorded_splits(workspace, selection_arm)
    if not recorded:
        print("\n*** The chain recorded no detections artefact, so the rebuilt split "
              "cannot be verified against it. Refusing: an anchor scored on a "
              "different split is worse than no anchor.", file=sys.stderr)
        return 1

    expected_classes = [*protocol.CLASS_ORDER, "unknown"]
    task_by_name = {task.name: task for task in chain}
    for item in recorded:
        task_name = item["task"].split("_", 1)[0]
        task = task_by_name.get(task_name)
        print(f"chain recorded {item['image_count']} images "
              f"on split {item['test_set']!r} (from {item['task']})")
        mismatch = (
            item["image_count"] != len(rebuilt)
            or item["ground_truth_images"] != rebuilt
            or item["ground_truth_image_order"] != subset.image_ids
            or item["test_set"] != evaluation_subset.SHARED_TEST_SET
            or item["metrics_test_set"] != evaluation_subset.SHARED_TEST_SET
            or item["dataset"] not in (None, "OWDETR")
            or item["class_names"] not in (None, expected_classes)
            or task is None
            or item["n_prev"] not in (None, task.n_prev)
            or item["n_current"] not in (None, task.n_new)
        )
        if mismatch:
            missing = rebuilt - item["ground_truth_images"]
            stray = item["ground_truth_images"] - rebuilt
            print("\n*** The rebuilt split does not match the one the chain was "
                  f"scored on ({len(missing)} missing, {len(stray)} stray IDs), or "
                  "the recorded dataset/class-count metadata differs. Refusing.",
                  file=sys.stderr)
            return 1
    print("split verified: every task records exactly the rebuilt image-ID set, "
          "dataset, evaluator order and class counts.")

    annotation_problem = verify_annotations(
        arguments.archive, arguments.data_root, rebuilt)
    if annotation_problem:
        print(f"\n*** {annotation_problem}. Refusing: matching image IDs alone do "
              "not prove matching evaluation data.", file=sys.stderr)
        return 1
    print("annotations    exact byte match with the committed archive")

    first_train_metadata = workspace / f"t2_{selection_arm}" / "checkpoint.train.json"
    if first_train_metadata.exists():
        trained = json.loads(first_train_metadata.read_text(encoding="utf-8"))
        recorded_checkpoint = trained.get("previous_checkpoint")
        print(f"chain started  {recorded_checkpoint}")
        if recorded_checkpoint and Path(recorded_checkpoint).resolve() != \
                arguments.checkpoint.resolve():
            print("\n*** The supplied checkpoint path differs from the one recorded "
                  "for t2. A path is not a content hash, but this mismatch is enough "
                  "to refuse the anchor.", file=sys.stderr)
            return 1
    else:
        print("checkpoint     no t2 training metadata; exact historical checkpoint "
              "identity cannot be proven from this workspace")

    if arguments.dry_run:
        print("\n--dry-run: inputs are verified; PROB was not called and nothing was written.")
        return 0

    # ---- one evaluation, and nothing else ----------------------------------
    with tempfile.TemporaryDirectory(prefix="owl-anchor-") as temporary:
        staging = Path(temporary)
        staged_data = make_staging_data_root(
            arguments.data_root, staging / "data", subset)
        staged_output = staging / "anchor_metrics.json"
        prob = bridge.Bridge(
            prob_root=arguments.prob_root, data_root=staged_data,
            log_dir=staging / "logs", num_workers=2, seed=seed)
        print(prob.check())
        print(f"\nevaluating {arguments.checkpoint} with prev={anchor_task.n_prev}, "
              f"current={anchor_task.n_new} — the anchor's own class counts")
        path = prob.evaluate(
            checkpoint=arguments.checkpoint,
            test_set=evaluation_subset.SHARED_TEST_SET,
            output=staged_output,
            n_prev=anchor_task.n_prev, n_current=anchor_task.n_new,
            detections=False,
        )

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        report = metrics.validate_per_class_ap50(
            payload, n_prev=anchor_task.n_prev, n_current=anchor_task.n_new)
        per_class = metrics.per_class_ap50(payload)
        print(f"\nstaged {path}")
        if report["usable"]:
            # Exclusive creation preserves the no-overwrite guarantee even if
            # another process wrote the anchor while PROB was evaluating.
            with output.open("xb") as destination:
                destination.write(Path(path).read_bytes())
    print(f"  known_AP50            {payload.get('known_AP50')}")
    print(f"  per-class entries     {len(per_class)}")
    print(f"  per-class validated   {report['usable']}  {report.get('reason', '')}")
    for check in report.get("checks", ()):
        print(f"    {check['quantity']:22s} n={check['classes']:2d} "
              f"rebuilt={check['rebuilt']:.4f} reported={check['reported']:.4f} "
              f"{'OK' if check['agrees'] else 'MISMATCH'}")
    if not report["usable"]:
        print("\n*** The anchor's own per-class vector does not validate. Do not use "
              "it as a forgetting reference.", file=sys.stderr)
        return 1
    print(f"\nwrote {output}")
    print("\nThe anchor used the supplied checkpoint, the chain's verified split, "
          "the configured PROB evaluator and the anchor's class counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
