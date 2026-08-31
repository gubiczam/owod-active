#!/usr/bin/env python3
"""Create one shared initialization or prepare one controlled T1 data root.

No training is performed.  The full preflight is intentionally unavailable in
an uncommitted OWL checkout; use ``preflight --protocol-only`` during review.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import evaluation_subset, longtail, protocol, t1_anchor  # noqa: E402

DEFAULT_MANIFEST_ROOT = ROOT / "data" / "reference" / "longtail"
DEFAULT_TRAIN_ARCHIVE = ROOT / "data" / "staging" / "owdetr_replay_annotations.tar.gz"
DEFAULT_TEST_ARCHIVE = ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz"


def git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False)
    if result.returncode:
        raise t1_anchor.AnchorError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def create_initialization(arguments: argparse.Namespace) -> dict[str, object]:
    prob_root = arguments.prob_root.resolve()
    if git_value(prob_root, "rev-parse", "HEAD") != t1_anchor.PINNED_PROB_COMMIT:
        raise t1_anchor.AnchorError("Initialization requires the exact pinned PROB commit.")
    dino = prob_root / "models" / "dino_resnet50_pretrain.pth"
    if not dino.is_file() or longtail.sha256_file(dino) != t1_anchor.DINO_SHA256:
        raise t1_anchor.AnchorError("DINO backbone file is absent or has the wrong SHA-256.")
    output = arguments.output.resolve()
    sidecar = output.with_suffix(".initialization.json")
    if output.exists() or sidecar.exists():
        raise t1_anchor.AnchorError(f"Refusing to overwrite initialization artefact {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(prob_root))
    previous_directory = Path.cwd()
    os.chdir(prob_root)
    try:
        import numpy as np
        import torch
        import torchvision
        import main_open_world
        from models import build_model

        random.seed(t1_anchor.TRAINING_SEED)
        np.random.seed(t1_anchor.TRAINING_SEED)
        torch.manual_seed(t1_anchor.TRAINING_SEED)
        args = main_open_world.get_args_parser().parse_args([])
        args.dataset = "OWDETR"
        args.PREV_INTRODUCED_CLS = 0
        args.CUR_INTRODUCED_CLS = 19
        args.num_classes = 81
        args.model_type = "prob"
        args.backbone = "dino_resnet50"
        args.pretrained_backbone = True
        args.obj_loss_coef = 1e-3
        args.obj_temp = 1
        args.seed = t1_anchor.TRAINING_SEED
        args.exemplar_replay_selection = False
        model, _, _, _ = build_model(args, mode="prob")
        state = model.state_dict()
        state_sha = t1_anchor.model_state_sha256(state)
        payload = {
            "model": state,
            "epoch": -1,
            "args": vars(args),
            "initialization": {
                "schema": "controlled_t1_initialization_v1",
                "seed": t1_anchor.TRAINING_SEED,
                "prob_commit": t1_anchor.PINNED_PROB_COMMIT,
                "dino_backbone_sha256": t1_anchor.DINO_SHA256,
                "model_state_sha256": state_sha,
            },
        }
        torch.save(payload, output)
        loaded = torch.load(output, map_location="cpu", weights_only=True)
        if (t1_anchor.model_state_sha256(loaded["model"]) != state_sha
                or loaded.get("epoch") != -1):
            raise t1_anchor.AnchorError("Serialized initialization did not round-trip exactly.")
        metadata = t1_anchor.initialization_metadata(
            path=output,
            prob_commit=t1_anchor.PINNED_PROB_COMMIT,
            torch_version=str(torch.__version__),
            torchvision_version=str(torchvision.__version__),
            python_version=sys.version,
            cuda_version=torch.version.cuda,
            model_state_sha256=state_sha,
        )
    finally:
        os.chdir(previous_directory)
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _view_paths(workspace: Path, data_root: Path | None = None) -> dict[str, Path]:
    data_root = data_root.resolve() if data_root is not None else workspace / "data" / "OWOD"
    return {
        "data_root": data_root,
        "annotations": data_root / "Annotations",
        "jpeg_link": data_root / "JPEGImages",
        "train_split": data_root / "ImageSets" / "OWDETR" / "owl_anchor_train.txt",
        "test_split": data_root / "ImageSets" / "OWDETR" / f"{t1_anchor.EVALUATION_SPLIT}.txt",
        "metadata": workspace / "training_view.json",
    }


def existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate == candidate.parent:
            raise t1_anchor.AnchorError(f"No existing parent for {path}.")
        candidate = candidate.parent
    return candidate


def protocol_report(arguments: argparse.Namespace) -> tuple[dict[str, object], list[str]]:
    manifest_path, manifest = t1_anchor.condition_manifest(
        arguments.condition, arguments.manifest_root)
    selection = t1_anchor.load_selection(manifest, repository_root=ROOT)
    missing: list[str] = []
    owl_commit = str(arguments.owl_commit)
    if len(owl_commit) != 40:
        missing.append("reviewed OWL commit")
    if not arguments.initialization.is_file():
        missing.append(f"shared seed-0 initialization {arguments.initialization}")
        initialization_sha = ""
    else:
        initialization_sha = longtail.sha256_file(arguments.initialization)
        if initialization_sha != arguments.initialization_sha:
            raise t1_anchor.AnchorError("Initialization SHA-256 does not match the required value.")
        sidecar = arguments.initialization.with_suffix(".initialization.json")
        if not sidecar.is_file():
            raise t1_anchor.AnchorError("Initialization provenance sidecar is missing.")
        initialization = json.loads(sidecar.read_text(encoding="utf-8"))
        t1_anchor.validate_initialization_metadata(initialization, arguments.initialization)
    recipe = None
    if len(owl_commit) == 40 and initialization_sha:
        recipe = t1_anchor.AnchorRecipe(
            condition=arguments.condition,
            manifest_sha256=str(manifest["scientific_sha256"]),
            owl_commit=owl_commit,
            initialization_sha256=initialization_sha,
            initialization_model_state_sha256=str(initialization["model_state_sha256"]),
            python_version=str(initialization["python_version"]),
            torch_version=str(initialization["torch_version"]),
            torchvision_version=str(initialization["torchvision_version"]),
            cuda_version=str(initialization["cuda_version"]),
        )
        recipe.validate()
    workspace = arguments.work_root.resolve() / f"t1_anchor__{arguments.condition}__seed0"
    report = {
        "schema": t1_anchor.ANCHOR_SCHEMA,
        "condition": arguments.condition,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["scientific_sha256"],
        "selection_images": len(selection),
        "selection_objects": manifest["selected_objects"],
        "workspace": str(workspace),
        "output_checkpoint": str(workspace / f"t1_{arguments.condition}.pth"),
        "owl_commit": owl_commit or None,
        "prob_commit": t1_anchor.PINNED_PROB_COMMIT,
        "initialization": str(arguments.initialization),
        "initialization_sha256": initialization_sha or None,
        "recipe": recipe.payload() if recipe else None,
        "recipe_fingerprint": recipe.fingerprint() if recipe else None,
        "optimizer_steps": (
            t1_anchor.optimizer_steps(int(manifest["selected_images"]), recipe)
            if recipe else None
        ),
        "missing_execution_inputs": missing,
        "execution_ready": False,
    }
    return report, missing


def preflight(arguments: argparse.Namespace) -> dict[str, object]:
    report, missing = protocol_report(arguments)
    if arguments.protocol_only:
        return report
    if missing:
        raise t1_anchor.AnchorError("Anchor inputs are incomplete: " + "; ".join(missing))

    owl_root = ROOT.resolve()
    current = git_value(owl_root, "rev-parse", "HEAD")
    if current != arguments.owl_commit:
        raise t1_anchor.AnchorError(f"OWL checkout is {current}, not reviewed {arguments.owl_commit}.")
    if git_value(owl_root, "status", "--porcelain"):
        raise t1_anchor.AnchorError("Expensive anchor training requires a clean OWL checkout.")
    prob_root = arguments.prob_root.resolve()
    if git_value(prob_root, "rev-parse", "HEAD") != t1_anchor.PINNED_PROB_COMMIT:
        raise t1_anchor.AnchorError("PROB checkout is not the pinned scientific commit.")
    dino = prob_root / "models" / "dino_resnet50_pretrain.pth"
    if not dino.is_file() or longtail.sha256_file(dino) != t1_anchor.DINO_SHA256:
        raise t1_anchor.AnchorError("Pinned DINO backbone is missing or has the wrong hash.")

    workspace = Path(str(report["workspace"]))
    output_checkpoint = Path(str(report["output_checkpoint"]))
    historical_checkpoint = arguments.work_root.parent / "checkpoints" / "SOWODB" / "t1.pth"
    t1_anchor.validate_anchor_workspace(
        workspace=workspace,
        condition=arguments.condition,
        output_checkpoint=output_checkpoint,
        historical_checkpoint=historical_checkpoint,
        allow_existing_output=True,
    )
    if output_checkpoint.exists():
        metadata_path = output_checkpoint.with_suffix(".metadata.json")
        if not metadata_path.is_file():
            raise t1_anchor.AnchorError("Existing final checkpoint lacks validated metadata.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        t1_anchor.validate_training_metadata(metadata)
        if metadata.get("condition") != arguments.condition \
                or metadata.get("checkpoint_sha256") != longtail.sha256_file(output_checkpoint) \
                or metadata.get("recipe_fingerprint") != report["recipe_fingerprint"]:
            raise t1_anchor.AnchorError("Existing final checkpoint provenance is invalid.")
    free = shutil.disk_usage(arguments.work_root).free
    if free < arguments.minimum_free_gib * (1 << 30):
        raise t1_anchor.AnchorError(
            f"Only {free / (1 << 30):.1f} GiB free; require {arguments.minimum_free_gib} GiB."
        )
    local_target = arguments.data_root if arguments.data_root is not None else workspace
    local_free = shutil.disk_usage(existing_parent(local_target)).free
    if local_free < arguments.minimum_local_free_gib * (1 << 30):
        raise t1_anchor.AnchorError(
            f"Only {local_free / (1 << 30):.1f} GiB free for local data; require "
            f"{arguments.minimum_local_free_gib} GiB."
        )

    if not arguments.materialize:
        raise t1_anchor.AnchorError("Full preflight requires --materialize for the filtered XML view.")
    paths = _view_paths(workspace, arguments.data_root)
    workspace.mkdir(parents=True, exist_ok=True)
    probe = workspace / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    _, manifest = t1_anchor.condition_manifest(arguments.condition, arguments.manifest_root)
    selection = t1_anchor.load_selection(manifest, repository_root=ROOT)
    source_before = longtail.sha256_file(arguments.train_annotations)
    test_before = longtail.sha256_file(arguments.test_annotations)
    if paths["metadata"].is_file() and paths["data_root"].is_dir():
        view = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        expected = {
            "condition": arguments.condition,
            "manifest_scientific_sha256": manifest["scientific_sha256"],
            "recipe_fingerprint": report["recipe_fingerprint"],
            "data_root": str(paths["data_root"]),
        }
        if any(view.get(name) != value for name, value in expected.items()):
            raise t1_anchor.AnchorError("Existing training view belongs to another recipe.")
        if t1_anchor.annotations_tree_sha256(paths["annotations"]) != view.get(
                "combined_annotations_tree_sha256"):
            raise t1_anchor.AnchorError("Existing materialized annotation tree changed.")
        if longtail.sha256_file(paths["train_split"]) != view.get("split_sha256"):
            raise t1_anchor.AnchorError("Existing materialized training split changed.")
        if longtail.sha256_file(paths["test_split"]) != t1_anchor.EVALUATION_SPLIT_SHA256:
            raise t1_anchor.AnchorError("Existing shared evaluation split changed.")
        train_ids = paths["train_split"].read_text(encoding="utf-8").splitlines()
        test_ids = paths["test_split"].read_text(encoding="utf-8").splitlines()
        t1_anchor.verify_jpegs(sorted(set(train_ids) | set(test_ids)), arguments.jpeg_root)
        if paths["jpeg_link"].resolve() != arguments.jpeg_root.resolve():
            raise t1_anchor.AnchorError("Existing JPEG link points at another source.")
    else:
        if paths["data_root"].exists() and any(paths["data_root"].iterdir()):
            raise t1_anchor.AnchorError(
                f"Partial data materialization requires a fresh target: {paths['data_root']}."
            )
        view = t1_anchor.materialize_training_view(
            manifest=manifest,
            selection=selection,
            source_annotations=arguments.train_annotations,
            annotations_dir=paths["annotations"],
            split_path=paths["train_split"],
        )
        t1_anchor.copy_evaluation_annotations(arguments.test_annotations, paths["annotations"])
        subset = evaluation_subset.from_archive(
            arguments.test_annotations,
            protocol.build_chain(6)[-1].known_classes,
            seed=0,
            remainder_multiplier=t1_anchor.EVALUATION_REMAINDER_MULTIPLIER,
            max_per_class=t1_anchor.EVALUATION_MAX_PER_CLASS,
        )
        evaluation_subset.write_image_set(paths["test_split"], subset)
        evaluation_sha = longtail.sha256_file(paths["test_split"])
        if evaluation_sha != t1_anchor.EVALUATION_SPLIT_SHA256:
            raise t1_anchor.AnchorError(
                f"Shared evaluation split identity changed: {evaluation_sha}."
            )
        all_images = sorted(set(selection) | set(subset.image_ids))
        t1_anchor.verify_jpegs(all_images, arguments.jpeg_root)
        t1_anchor.link_jpeg_root(arguments.jpeg_root, paths["jpeg_link"])
        view |= {
            "data_root": str(paths["data_root"]),
            "evaluation_images": len(subset.image_ids),
            "evaluation_split_sha256": evaluation_sha,
            "evaluation_annotations_sha256": test_before,
            "recipe_fingerprint": report["recipe_fingerprint"],
            "combined_annotations_tree_sha256": t1_anchor.annotations_tree_sha256(
                paths["annotations"]),
        }
    if longtail.sha256_file(arguments.train_annotations) != source_before:
        raise t1_anchor.AnchorError("Canonical training annotation archive was modified.")
    if longtail.sha256_file(arguments.test_annotations) != test_before:
        raise t1_anchor.AnchorError("Canonical evaluation annotation archive was modified.")
    t1_anchor.write_json_once_or_verify(paths["metadata"], view)
    t1_anchor.write_json_once_or_verify(workspace / "config.json", report["recipe"])
    t1_anchor.write_json_once_or_verify(workspace / "recipe.json", {
        "schema": t1_anchor.ANCHOR_SCHEMA,
        "fingerprint": report["recipe_fingerprint"],
        "recipe": report["recipe"],
    })
    t1_anchor.write_json_once_or_verify(workspace / "provenance.json", {
        "schema": "controlled_t1_anchor_provenance_v1",
        "condition": arguments.condition,
        "manifest_sha256": report["manifest_sha256"],
        "selection_ledger_sha256": manifest["selection"]["sha256"],
        "source_index_sha256": manifest["source_index"]["sha256"],
        "source_annotations_sha256": manifest["source_annotations"]["sha256"],
        "evaluation_annotations_sha256": manifest["evaluation_split_sha256"],
        "owl_commit": report["owl_commit"],
        "prob_commit": report["prob_commit"],
        "initialization_sha256": report["initialization_sha256"],
        "dino_backbone_sha256": t1_anchor.DINO_SHA256,
        "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
    })
    report["training_view"] = view
    report["execution_ready"] = True
    report["missing_execution_inputs"] = []
    return report


def common_preflight_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--condition", choices=t1_anchor.PRIMARY_CONDITIONS, required=True)
    command.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    command.add_argument("--train-annotations", type=Path, default=DEFAULT_TRAIN_ARCHIVE)
    command.add_argument("--test-annotations", type=Path, default=DEFAULT_TEST_ARCHIVE)
    command.add_argument("--prob-root", type=Path, required=True)
    command.add_argument("--work-root", type=Path, required=True)
    command.add_argument(
        "--data-root", type=Path,
        help="isolated local OWOD data root; defaults inside the persistent workspace",
    )
    command.add_argument("--jpeg-root", type=Path, required=True)
    command.add_argument("--initialization", type=Path, required=True)
    command.add_argument("--initialization-sha", default="")
    command.add_argument("--owl-commit", default="")
    command.add_argument("--minimum-free-gib", type=int, default=10)
    command.add_argument("--minimum-local-free-gib", type=int, default=10)
    command.add_argument("--materialize", action="store_true")
    command.add_argument("--protocol-only", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    initialization = commands.add_parser("create-initialization")
    initialization.add_argument("--prob-root", type=Path, required=True)
    initialization.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("preflight")
    common_preflight_arguments(check)
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command == "create-initialization":
            report = create_initialization(arguments)
            label = "INITIALIZATION CREATED"
        else:
            report = preflight(arguments)
            label = "DATA PREFLIGHT PASS" if report["execution_ready"] else "PROTOCOL-ONLY"
    except (t1_anchor.AnchorError, longtail.LongTailError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    print(label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
