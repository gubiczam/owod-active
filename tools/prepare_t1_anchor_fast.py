#!/usr/bin/env python3
"""Prepare recipe-isolated controlled-LT data views for FAST benchmarking."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import evaluation_subset, longtail, protocol, t1_anchor, t1_anchor_fast

DEFAULT_MANIFEST_ROOT = ROOT / "data" / "reference" / "longtail"
DEFAULT_TRAIN_ARCHIVE = ROOT / "data" / "staging" / "owdetr_replay_annotations.tar.gz"
DEFAULT_TEST_ARCHIVE = ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz"


def git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=root, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def paths(workspace: Path, data_root: Path) -> dict[str, Path]:
    return {
        "data_root": data_root,
        "annotations": data_root / "Annotations",
        "jpeg_link": data_root / "JPEGImages",
        "train_split": data_root / "ImageSets" / "OWDETR" / "owl_anchor_train.txt",
        "test_split": data_root / "ImageSets" / "OWDETR" / f"{t1_anchor.EVALUATION_SPLIT}.txt",
        "metadata": workspace / "training_view.json",
    }


def prepare(arguments: argparse.Namespace) -> dict[str, object]:
    manifest_path, manifest = t1_anchor.condition_manifest(
        arguments.condition, arguments.manifest_root)
    selection = t1_anchor.load_selection(manifest, repository_root=ROOT)
    initialization_sha = longtail.sha256_file(arguments.initialization)
    if initialization_sha != arguments.initialization_sha:
        raise t1_anchor.AnchorError("Shared initialization SHA-256 mismatch.")
    sidecar = arguments.initialization.with_suffix(".initialization.json")
    initialization = json.loads(sidecar.read_text(encoding="utf-8"))
    t1_anchor.validate_initialization_metadata(initialization, arguments.initialization)
    if git_value(ROOT, "rev-parse", "HEAD") != arguments.owl_commit \
            or git_value(ROOT, "status", "--porcelain"):
        raise t1_anchor.AnchorError("FAST preparation requires its clean reviewed OWL commit.")
    if git_value(arguments.prob_root, "rev-parse", "HEAD") != t1_anchor.PINNED_PROB_COMMIT:
        raise t1_anchor.AnchorError("FAST preparation requires pinned PROB.")
    dino = arguments.prob_root / "models" / "dino_resnet50_pretrain.pth"
    if not dino.is_file() or longtail.sha256_file(dino) != t1_anchor.DINO_SHA256:
        raise t1_anchor.AnchorError("Pinned DINO backbone is missing or changed.")

    workspace = t1_anchor_fast.workspace(arguments.work_root.resolve(), arguments.condition)
    t1_anchor_fast.validate_workspace_path(workspace, arguments.condition)
    workspace.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(arguments.work_root).free < arguments.minimum_free_gib * (1 << 30):
        raise t1_anchor.AnchorError("Insufficient persistent Drive space for FAST checkpoints.")
    view_paths = paths(workspace, arguments.data_root.resolve())
    source_before = longtail.sha256_file(arguments.train_annotations)
    test_before = longtail.sha256_file(arguments.test_annotations)
    data_identity = {
        "schema": "controlled_t1_anchor_fast_data_view_v1",
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": arguments.condition,
        "manifest_scientific_sha256": manifest["scientific_sha256"],
        "initialization_sha256": initialization_sha,
        "prob_commit": t1_anchor.PINNED_PROB_COMMIT,
        "owl_commit": arguments.owl_commit,
        "data_root": str(view_paths["data_root"]),
    }
    if view_paths["metadata"].is_file() and view_paths["data_root"].is_dir():
        view = json.loads(view_paths["metadata"].read_text(encoding="utf-8"))
        if any(view.get(key) != value for key, value in data_identity.items()):
            raise t1_anchor.AnchorError("Existing FAST view has another immutable identity.")
        if t1_anchor.annotations_tree_sha256(view_paths["annotations"]) != view.get(
                "combined_annotations_tree_sha256"):
            raise t1_anchor.AnchorError("Existing FAST annotation tree changed.")
        if longtail.sha256_file(view_paths["train_split"]) != view.get("split_sha256"):
            raise t1_anchor.AnchorError("Existing FAST training split changed.")
    else:
        if view_paths["data_root"].exists() and any(view_paths["data_root"].iterdir()):
            raise t1_anchor.AnchorError("Partial FAST materialization requires a fresh data root.")
        view = t1_anchor.materialize_training_view(
            manifest=manifest, selection=selection,
            source_annotations=arguments.train_annotations,
            annotations_dir=view_paths["annotations"], split_path=view_paths["train_split"])
        t1_anchor.copy_evaluation_annotations(
            arguments.test_annotations, view_paths["annotations"])
        subset = evaluation_subset.from_archive(
            arguments.test_annotations, protocol.build_chain(6)[-1].known_classes,
            seed=0, remainder_multiplier=t1_anchor.EVALUATION_REMAINDER_MULTIPLIER,
            max_per_class=t1_anchor.EVALUATION_MAX_PER_CLASS)
        evaluation_subset.write_image_set(view_paths["test_split"], subset)
        if longtail.sha256_file(view_paths["test_split"]) != t1_anchor.EVALUATION_SPLIT_SHA256:
            raise t1_anchor.AnchorError("FAST shared evaluation split identity changed.")
        t1_anchor.verify_jpegs(sorted(set(selection) | set(subset.image_ids)), arguments.jpeg_root)
        t1_anchor.link_jpeg_root(arguments.jpeg_root, view_paths["jpeg_link"])
        view |= data_identity | {
            "evaluation_images": len(subset.image_ids),
            "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
            "evaluation_annotations_sha256": test_before,
            "combined_annotations_tree_sha256": t1_anchor.annotations_tree_sha256(
                view_paths["annotations"]),
        }
        t1_anchor.write_json_once_or_verify(view_paths["metadata"], view)
    if longtail.sha256_file(arguments.train_annotations) != source_before \
            or longtail.sha256_file(arguments.test_annotations) != test_before:
        raise t1_anchor.AnchorError("Canonical annotation sources changed during FAST preparation.")
    provenance = data_identity | {
        "manifest_path": str(manifest_path),
        "selection_ledger_sha256": manifest["selection"]["sha256"],
        "source_annotations_sha256": source_before,
        "evaluation_annotations_sha256": test_before,
        "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
        "dino_backbone_sha256": t1_anchor.DINO_SHA256,
        "scientific_wording": (
            "fixed-compute comparison of annotated supervision imbalance; filtered XMLs "
            "do not change natural scene prevalence"
        ),
    }
    t1_anchor.write_json_once_or_verify(workspace / "data_provenance.json", provenance)
    return {
        "recipe_version": t1_anchor_fast.FAST_RECIPE_VERSION,
        "condition": arguments.condition,
        "workspace": str(workspace),
        "training_view": view,
        "state": t1_anchor_fast.workspace_state(workspace, arguments.condition),
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--condition", choices=t1_anchor_fast.CONDITION_ORDER, required=True)
    command.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    command.add_argument("--train-annotations", type=Path, default=DEFAULT_TRAIN_ARCHIVE)
    command.add_argument("--test-annotations", type=Path, default=DEFAULT_TEST_ARCHIVE)
    command.add_argument("--prob-root", type=Path, required=True)
    command.add_argument("--work-root", type=Path, required=True)
    command.add_argument("--data-root", type=Path, required=True)
    command.add_argument("--jpeg-root", type=Path, required=True)
    command.add_argument("--initialization", type=Path, required=True)
    command.add_argument("--initialization-sha", required=True)
    command.add_argument("--owl-commit", required=True)
    command.add_argument("--minimum-free-gib", type=int, default=10)
    return command


def main() -> int:
    try:
        report = prepare(parser().parse_args())
    except (t1_anchor.AnchorError, longtail.LongTailError, OSError,
            json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    print("CONTROLLED T1 ANCHOR FAST DATA PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
