"""Controlled-long-tail T1 anchor provenance and training-view contracts.

This module prepares data and fingerprints only.  PROB remains the owner of
model construction, optimization, checkpointing, and evaluation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree

from owl import evaluation_subset, longtail, metrics, protocol

ANCHOR_PROTOCOL_VERSION = 1
ANCHOR_SCHEMA = "controlled_t1_anchor_v1"
PINNED_PROB_COMMIT = "4c66be1a52cad9360e09c729e9134aba8fe0b531"
DINO_SHA256 = "156f8c4166a23dc2951ae811e39d76a06269c565932edf647c0187e65cd7aa7c"
HISTORICAL_T1_SHA256 = "dba5390bffdfdf63058a995f241696df8d06b7fb859aecc8292d9ea02d459a22"
EXPECTED_MANIFEST_SHA256 = {
    "original": "f25ae1b235f87cefe2044e81ca6753cc46bd5b81b3085b019459de4a8113b032",
    "lt10": "b3c751fa1034a499d592391d87afc145b0cc8b11bf4a255512fd4f52ca094f0f",
    "lt50": "9525f6f40958c1282b739a79c6c196d4c7942e35dc605a777a0f45b245547f0f",
    "lt100": "5d5e9b2287c97748c135f4412696201449b7086683f58facc876c3c62d8a4e2d",
}
PRIMARY_CONDITIONS = ("lt10", "lt50", "lt100")
TRAINING_SEED = 0
EVALUATION_SPLIT = evaluation_subset.SHARED_TEST_SET
EVALUATION_SPLIT_SHA256 = "f37a3bb0916dd8462fceb35f60364fed75d3a00cebd3e0ce72775dbf79d76c27"
EVALUATION_MAX_PER_CLASS = 150
EVALUATION_REMAINDER_MULTIPLIER = 1


class AnchorError(ValueError):
    """Raised when an anchor input or output is scientifically ambiguous."""


def model_state_sha256(state: Mapping[str, object]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes independently of torch.save."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _lower_hex(value: str, length: int, label: str) -> str:
    text = str(value)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise AnchorError(f"{label} must be an exact {length}-character lowercase hex value.")
    return text


def condition_manifest(
    condition: str, manifest_root: str | Path,
) -> tuple[Path, dict[str, object]]:
    condition = str(condition).lower()
    if condition not in longtail.CONDITIONS:
        raise AnchorError(f"Unknown condition {condition!r}.")
    path = Path(manifest_root) / f"{condition}.json"
    if not path.is_file():
        raise AnchorError(f"Missing controlled-LT manifest: {path}.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    longtail.verify_manifest(manifest)
    if manifest.get("condition") != condition:
        raise AnchorError(f"{path} claims condition {manifest.get('condition')!r}.")
    expected = EXPECTED_MANIFEST_SHA256[condition]
    if manifest.get("scientific_sha256") != expected:
        raise AnchorError(
            f"{condition} manifest identity changed: {manifest.get('scientific_sha256')} != "
            f"reviewed {expected}."
        )
    classes = [str(row["class_name"]) for row in manifest.get("classes", [])]
    if tuple(classes) != longtail.class_ranking({
        str(row["class_name"]): int(row["original_count"])
        for row in manifest["classes"]
    }):
        raise AnchorError(f"{condition} manifest class ranking is not canonical.")
    if set(classes) != set(protocol.TASK1) or len(classes) != 19:
        raise AnchorError("The anchor manifest must cover exactly the nineteen T1 classes.")
    return path, manifest


def _repo_reference(path: str, repository_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root / candidate


def load_selection(
    manifest: Mapping[str, object], *, repository_root: str | Path,
) -> dict[str, dict[str, list[int]]]:
    selection_ref = manifest["selection"]
    path = _repo_reference(str(selection_ref["path"]), Path(repository_root))
    if not path.is_file():
        raise AnchorError(f"Missing selected-object ledger: {path}.")
    actual = longtail.sha256_file(path)
    if actual != selection_ref["sha256"]:
        raise AnchorError(f"Selection hash mismatch for {path}: {actual}.")
    payload = longtail.read_gzip_json(path)
    if payload.get("protocol") != longtail.PROTOCOL_NAME:
        raise AnchorError(f"Selection ledger {path} uses another protocol.")
    if payload.get("condition") != manifest["condition"]:
        raise AnchorError(f"Selection ledger {path} uses another condition.")
    selection = payload.get("objects")
    if not isinstance(selection, dict):
        raise AnchorError(f"Selection ledger {path} has no object map.")
    return selection


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.xml")):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def annotations_tree_sha256(root: str | Path) -> str:
    """Public deterministic identity for an isolated XML materialization."""

    path = Path(root)
    if not path.is_dir():
        raise AnchorError(f"Annotation directory is missing: {path}.")
    return _tree_sha256(path)


def write_json_once_or_verify(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Atomically create immutable metadata, or verify an identical existing file."""

    target = Path(path)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise AnchorError(f"Existing immutable metadata differs: {target}.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        raise AnchorError(f"Interrupted metadata write requires inspection: {temporary}.")
    temporary.write_bytes(encoded)
    temporary.replace(target)
    return target


def _canonical_objects(root: ElementTree.Element) -> dict[str, list[ElementTree.Element]]:
    result: dict[str, list[ElementTree.Element]] = defaultdict(list)
    for item in root.findall("object"):
        name = evaluation_subset.canonical_class_name(item.findtext("name", default=""))
        result[name].append(item)
    return result


def materialize_training_view(
    *,
    manifest: Mapping[str, object],
    selection: Mapping[str, Mapping[str, Sequence[int]]],
    source_annotations: str | Path,
    annotations_dir: str | Path,
    split_path: str | Path,
) -> dict[str, object]:
    """Materialize filtered XML aliases, refusing overwrite or source mutation."""

    source = Path(source_annotations)
    output = Path(annotations_dir)
    split = Path(split_path)
    if not source.is_file():
        raise AnchorError(f"Missing canonical training annotation archive: {source}.")
    source_before = longtail.sha256_file(source)
    if source_before != manifest["source_annotations"]["sha256"]:
        raise AnchorError("Canonical training annotation archive hash mismatch.")
    if output.exists() and any(output.iterdir()):
        raise AnchorError(f"Training-view directory is not empty: {output}.")
    if split.exists():
        raise AnchorError(f"Training split already exists: {split}.")
    output.mkdir(parents=True, exist_ok=True)
    split.parent.mkdir(parents=True, exist_ok=True)

    wanted_images = set(selection)
    seen: set[str] = set()
    achieved: Counter[str] = Counter()
    with tarfile.open(source) as archive:
        for member in archive.getmembers():
            image_id = Path(member.name).stem
            if image_id not in wanted_images or not member.isfile() or not member.name.endswith(".xml"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise AnchorError(f"Cannot read source XML {member.name}.")
            root = ElementTree.fromstring(handle.read())
            by_class = _canonical_objects(root)
            keep: set[int] = set()
            for class_name, ordinals in selection[image_id].items():
                objects = by_class.get(class_name, [])
                for ordinal in ordinals:
                    if not isinstance(ordinal, int) or ordinal < 0 or ordinal >= len(objects):
                        raise AnchorError(
                            f"Invalid selected object {image_id}/{class_name}/{ordinal}."
                        )
                    keep.add(id(objects[ordinal]))
                    achieved[class_name] += 1
            for item in list(root.findall("object")):
                if id(item) not in keep:
                    root.remove(item)
            if not root.findall("object"):
                raise AnchorError(f"Filtered view unexpectedly emptied {image_id}.")
            payload = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
            (output / f"{image_id}.xml").write_bytes(payload)
            seen.add(image_id)

    missing = wanted_images - seen
    if missing:
        raise AnchorError(
            f"Canonical archive lacks {len(missing)} selected XMLs; first {sorted(missing)[:10]}."
        )
    expected = {
        str(row["class_name"]): int(row["achieved_count"])
        for row in manifest["classes"]
    }
    if {name: achieved[name] for name in protocol.TASK1} != expected:
        raise AnchorError("Materialized XML counts do not match the reviewed manifest.")
    image_ids = sorted(wanted_images)
    split.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    if longtail.sha256_file(source) != source_before:
        raise AnchorError("Canonical training annotations changed during materialization.")
    return {
        "schema": "controlled_t1_training_view_v1",
        "condition": manifest["condition"],
        "manifest_scientific_sha256": manifest["scientific_sha256"],
        "source_annotations_sha256": source_before,
        "annotations_tree_sha256": _tree_sha256(output),
        "split_sha256": longtail.sha256_file(split),
        "images": len(image_ids),
        "objects": sum(achieved.values()),
        "class_counts": {name: achieved[name] for name in protocol.TASK1},
    }


def verify_jpegs(image_ids: Sequence[str], jpeg_root: str | Path) -> None:
    root = Path(jpeg_root)
    if not root.is_dir():
        raise AnchorError(f"Canonical JPEG root is missing: {root}.")
    missing = [image_id for image_id in image_ids if not (root / f"{image_id}.jpg").is_file()]
    if missing:
        raise AnchorError(
            f"Canonical JPEG root lacks {len(missing)} selected images; first {missing[:10]}."
        )


def validate_anchor_workspace(
    *, workspace: str | Path, condition: str, output_checkpoint: str | Path,
    historical_checkpoint: str | Path | None = None,
    allow_existing_output: bool = False,
) -> None:
    """Reject ambiguous, shared, existing, or historical anchor targets."""

    root = Path(workspace).resolve()
    expected = f"t1_anchor__{condition}__seed0"
    if root.name != expected or root.name in longtail.HISTORICAL_WORKSPACES:
        raise AnchorError(f"Workspace must be isolated as {expected!r}.")
    output = Path(output_checkpoint).resolve()
    if output.exists() and not allow_existing_output:
        raise AnchorError(f"Refusing to overwrite anchor checkpoint {output}.")
    if historical_checkpoint is not None and output == Path(historical_checkpoint).resolve():
        raise AnchorError("Anchor output collides with the historical T1 checkpoint.")


@dataclass(frozen=True)
class AnchorRecipe:
    """The reviewed single-T4 fixed recipe; not an exact historical reproduction."""

    schema: str = ANCHOR_SCHEMA
    anchor_protocol_version: int = ANCHOR_PROTOCOL_VERSION
    condition: str = ""
    manifest_sha256: str = ""
    owl_commit: str = ""
    prob_commit: str = PINNED_PROB_COMMIT
    initialization_sha256: str = ""
    initialization_model_state_sha256: str = ""
    python_version: str = ""
    torch_version: str = ""
    torchvision_version: str = ""
    cuda_version: str = ""
    dino_backbone_sha256: str = DINO_SHA256
    architecture: str = "PROB Deformable-DETR"
    model_type: str = "prob"
    backbone: str = "dino_resnet50"
    pretrained_backbone: bool = True
    num_classes: int = 81
    class_order: tuple[str, ...] = protocol.TASK1
    dataset: str = "OWDETR"
    train_supervision: str = "condition-filtered T1 XML aliases"
    evaluation_split: str = EVALUATION_SPLIT
    evaluation_split_sha256: str = EVALUATION_SPLIT_SHA256
    epochs: int = 41
    duration_policy: str = "same_epochs"
    batch_size: int = 2
    optimizer: str = "AdamW"
    learning_rate: float = 2e-4
    backbone_learning_rate: float = 2e-5
    linear_projection_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    lr_drop_epoch: int = 31
    lr_drop_gamma: float = 0.1
    clip_max_norm: float = 0.1
    seed: int = TRAINING_SEED
    num_workers: int = 2
    num_queries: int = 100
    num_feature_levels: int = 4
    with_box_refine: bool = False
    two_stage: bool = False
    masks: bool = False
    dilation: bool = False
    position_embedding: str = "sine"
    position_embedding_scale: float = 6.283185307179586
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_points: int = 4
    decoder_attention_points: int = 4
    hidden_dim: int = 256
    feedforward_dim: int = 1024
    dropout: float = 0.1
    attention_heads: int = 8
    matcher_class_cost: float = 2
    matcher_bbox_cost: float = 5
    matcher_giou_cost: float = 2
    classification_loss_coefficient: float = 2
    bbox_loss_coefficient: float = 5
    giou_loss_coefficient: float = 2
    objectness_loss_coefficient: float = 1e-3
    objectness_temperature: float = 1
    focal_alpha: float = 0.25
    auxiliary_loss: bool = True
    unmatched_boxes: bool = False
    top_unknown: int = 5
    feature_dim: int = 1024
    invalid_class_logits: bool = False
    novelty_classification_branch: bool = False
    novelty_loss_coefficient: float = 2
    novelty_start_epoch: int = 0
    bbox_threshold: float = 0.3
    unknown_confidence_weight: float = 1
    freeze_probabilistic_model: bool = False
    remove_difficult: bool = False
    cache_mode: bool = False
    evaluation_every: int = 5
    device: str = "cuda"
    augmentation: str = (
        "RandomHorizontalFlip; RandomSelect(RandomResize[480..800], "
        "RandomResize[400,500,600]+RandomSizeCrop[384,600]+RandomResize[480..800]); "
        "max_size=1333; ImageNet normalization"
    )
    replay: str = "none"
    active_selection: str = "none"
    class_balanced_sampler: bool = False
    oversampling: bool = False
    loss_reweighting: bool = False

    def validate(self) -> None:
        if self.condition not in longtail.CONDITIONS:
            raise AnchorError(f"Invalid recipe condition {self.condition!r}.")
        _lower_hex(self.manifest_sha256, 64, "manifest_sha256")
        _lower_hex(self.owl_commit, 40, "owl_commit")
        _lower_hex(self.prob_commit, 40, "prob_commit")
        _lower_hex(self.initialization_sha256, 64, "initialization_sha256")
        _lower_hex(
            self.initialization_model_state_sha256, 64,
            "initialization_model_state_sha256",
        )
        _lower_hex(self.dino_backbone_sha256, 64, "dino_backbone_sha256")
        if self.prob_commit != PINNED_PROB_COMMIT:
            raise AnchorError("The anchor recipe changed the pinned PROB commit.")
        if self.manifest_sha256 != EXPECTED_MANIFEST_SHA256[self.condition]:
            raise AnchorError("The anchor recipe changed the reviewed LT manifest.")
        if self.class_order != protocol.TASK1 or len(self.class_order) != 19:
            raise AnchorError("The anchor recipe changed the exact T1 class order.")
        if self.seed != TRAINING_SEED:
            raise AnchorError("The first controlled anchor study is preregistered at seed 0.")
        if self.replay != "none" or self.active_selection != "none":
            raise AnchorError("T1 anchor training cannot use replay or active selection.")
        if self.oversampling or self.class_balanced_sampler or self.loss_reweighting:
            raise AnchorError("The controlled intervention is supervision frequency only.")
        if any((self.with_box_refine, self.two_stage, self.masks, self.dilation,
                self.unmatched_boxes, self.invalid_class_logits,
                self.novelty_classification_branch, self.freeze_probabilistic_model,
                self.remove_difficult, self.cache_mode)):
            raise AnchorError("The fixed anchor recipe enabled an unreviewed PROB branch.")
        if self.evaluation_split_sha256 != EVALUATION_SPLIT_SHA256:
            raise AnchorError("The fixed shared evaluation split identity changed.")
        for name in ("python_version", "torch_version", "torchvision_version", "cuda_version"):
            if not getattr(self, name):
                raise AnchorError(f"The recipe lacks pinned runtime field {name!r}.")
        identity = {
            "condition", "manifest_sha256", "owl_commit", "initialization_sha256",
            "initialization_model_state_sha256", "python_version", "torch_version",
            "torchvision_version", "cuda_version",
        }
        expected = asdict(AnchorRecipe())
        changed = {
            name: (expected[name], value)
            for name, value in asdict(self).items()
            if name not in identity and value != expected[name]
        }
        if changed:
            raise AnchorError(f"Fixed anchor recipe settings changed: {changed}.")

    def payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    def fingerprint(self) -> str:
        return longtail.sha256_bytes(longtail.canonical_json_bytes(self.payload()))


def optimizer_steps(images: int, recipe: AnchorRecipe) -> int:
    if images < recipe.batch_size:
        raise AnchorError("Training view is smaller than one drop-last batch.")
    return (images // recipe.batch_size) * recipe.epochs


def initialization_metadata(
    *, path: str | Path, prob_commit: str, torch_version: str, python_version: str,
    torchvision_version: str, cuda_version: str | None, model_state_sha256: str,
) -> dict[str, object]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise AnchorError(f"Initialization checkpoint is missing: {checkpoint}.")
    return {
        "schema": "controlled_t1_initialization_v1",
        "path": str(checkpoint),
        "sha256": longtail.sha256_file(checkpoint),
        "model_state_sha256": _lower_hex(model_state_sha256, 64, "model_state_sha256"),
        "prob_commit": _lower_hex(prob_commit, 40, "prob_commit"),
        "dino_backbone_sha256": DINO_SHA256,
        "seed": TRAINING_SEED,
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "python_version": python_version,
        "cuda_version": cuda_version,
        "epoch": -1,
        "class_order": list(protocol.TASK1),
    }


def validate_initialization_metadata(
    payload: Mapping[str, object], checkpoint: str | Path,
) -> None:
    required = {
        "schema", "path", "sha256", "model_state_sha256", "prob_commit",
        "dino_backbone_sha256", "seed", "torch_version", "torchvision_version",
        "python_version", "cuda_version", "epoch", "class_order",
    }
    missing = required - set(payload)
    if missing:
        raise AnchorError(f"Initialization provenance is missing {sorted(missing)}.")
    if payload["schema"] != "controlled_t1_initialization_v1":
        raise AnchorError("Initialization provenance has another schema.")
    if longtail.sha256_file(checkpoint) != payload["sha256"]:
        raise AnchorError("Initialization checkpoint SHA-256 differs from its provenance.")
    _lower_hex(str(payload["model_state_sha256"]), 64, "model_state_sha256")
    if payload["prob_commit"] != PINNED_PROB_COMMIT:
        raise AnchorError("Initialization provenance uses another PROB commit.")
    if payload["dino_backbone_sha256"] != DINO_SHA256:
        raise AnchorError("Initialization provenance uses another DINO backbone.")
    if payload["seed"] != TRAINING_SEED or payload["epoch"] != -1:
        raise AnchorError("Initialization provenance is not the shared seed-0 pre-T1 state.")
    if tuple(payload["class_order"]) != protocol.TASK1:
        raise AnchorError("Initialization provenance changed the T1 class order.")
    for name in ("torch_version", "torchvision_version", "python_version"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise AnchorError(f"Initialization provenance has invalid {name!r}.")


REQUIRED_METADATA = {
    "schema", "condition", "manifest_sha256", "owl_commit", "prob_commit",
    "initialization_sha256", "recipe_fingerprint", "seed", "epochs",
    "optimizer_steps", "checkpoint_sha256", "class_order", "train_objects",
    "train_images", "started_at", "ended_at", "gpu", "torch_version",
    "torchvision_version", "python_version", "cuda_version", "msda", "command",
    "evaluation_split_sha256",
}


def validate_training_metadata(payload: Mapping[str, object]) -> None:
    missing = REQUIRED_METADATA - set(payload)
    if missing:
        raise AnchorError(f"Anchor metadata is missing {sorted(missing)}.")
    if payload["schema"] != ANCHOR_SCHEMA:
        raise AnchorError("Anchor metadata has another schema.")
    if tuple(payload["class_order"]) != protocol.TASK1:
        raise AnchorError("Anchor metadata changed the T1 class order.")
    for name in (
        "manifest_sha256", "initialization_sha256", "recipe_fingerprint",
        "checkpoint_sha256",
    ):
        _lower_hex(str(payload[name]), 64, name)
    for name in ("owl_commit", "prob_commit"):
        _lower_hex(str(payload[name]), 40, name)


def controlled_groups(manifest: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {name: [] for name in longtail.GROUPS}
    for row in manifest["classes"]:
        grouped[str(row["group"])].append(str(row["class_name"]))
    result = {name: tuple(values) for name, values in grouped.items()}
    if tuple(map(len, result.values())) != (7, 6, 6):
        raise AnchorError("Controlled groups must contain 7/6/6 classes.")
    return result


def anchor_metrics_payload(
    *, condition: str, manifest: Mapping[str, object], bridge_metrics: Mapping[str, object],
    checkpoint_sha256: str, recipe_fingerprint: str,
) -> dict[str, object]:
    per_class = metrics.per_class_ap50(bridge_metrics)
    missing = set(protocol.TASK1) - set(per_class)
    if missing:
        raise AnchorError(f"Anchor evaluation lacks T1 AP50 for {sorted(missing)}.")
    groups = controlled_groups(manifest)
    group_map = {
        group: sum(float(per_class[name]) for name in names) / len(names)
        for group, names in groups.items()
    }
    rows = []
    for row in manifest["classes"]:
        rows.append({
            "class_name": row["class_name"],
            "rank": row["rank"],
            "group": row["group"],
            "train_count": row["achieved_count"],
            "anchor_AP50": float(per_class[str(row["class_name"])]),
        })
    overall = sum(float(per_class[name]) for name in protocol.TASK1) / len(protocol.TASK1)
    reported_overall = float(bridge_metrics.get("known_AP50", overall))
    if abs(reported_overall - overall) > 1e-4:
        raise AnchorError(
            f"Evaluator known mAP50 {reported_overall} disagrees with the per-class mean "
            f"{overall}."
        )
    return {
        "schema": "controlled_t1_anchor_metrics_v1",
        "condition": condition,
        "manifest_sha256": manifest["scientific_sha256"],
        "checkpoint_sha256": _lower_hex(checkpoint_sha256, 64, "checkpoint_sha256"),
        "recipe_fingerprint": _lower_hex(recipe_fingerprint, 64, "recipe_fingerprint"),
        "evaluation_split": EVALUATION_SPLIT,
        "evaluation_split_sha256": bridge_metrics.get("test_set_sha256"),
        "overall_mAP50": reported_overall,
        "group_mAP50": group_map,
        "classes": rows,
        "source_bridge_metrics": dict(bridge_metrics),
    }


def condition_forgetting(
    anchor: Mapping[str, object], final: Mapping[str, object], *, epsilon: float = 1e-12,
) -> list[dict[str, object]]:
    """Condition-specific T1->T6 forgetting; cross-condition anchors are refused."""

    if anchor.get("condition") != final.get("condition"):
        raise AnchorError("Forgetting must use the same condition's T1 and T6 metrics.")
    anchor_rows = {str(row["class_name"]): row for row in anchor.get("classes", [])}
    final_ap = final.get("per_class_AP50")
    if not isinstance(final_ap, Mapping):
        raise AnchorError("Final metrics have no per_class_AP50 mapping.")
    rows: list[dict[str, object]] = []
    for name in protocol.TASK1:
        if name not in anchor_rows or name not in final_ap:
            raise AnchorError(f"Forgetting inputs lack class {name!r}.")
        start = float(anchor_rows[name]["anchor_AP50"])
        end = float(final_ap[name])
        absolute = start - end
        rows.append({
            "condition": anchor["condition"],
            "class_name": name,
            "anchor_AP50": start,
            "final_AP50": end,
            "absolute_forgetting": absolute,
            "relative_forgetting": absolute / (start + epsilon),
        })
    return rows


def copy_evaluation_annotations(source_archive: str | Path, annotations_dir: str | Path) -> None:
    """Add immutable evaluation XMLs to an isolated condition data root."""

    output = Path(annotations_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(Path(source_archive)) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            target = output / Path(member.name).name
            if target.exists():
                raise AnchorError(f"Train/evaluation annotation collision at {target.name}.")
            handle = archive.extractfile(member)
            if handle is None:
                raise AnchorError(f"Cannot read evaluation XML {member.name}.")
            target.write_bytes(handle.read())


def link_jpeg_root(source: str | Path, target: str | Path) -> None:
    """Expose canonical JPEGs without copying or modifying them."""

    source_path = Path(source).resolve()
    target_path = Path(target)
    if not source_path.is_dir():
        raise AnchorError(f"Canonical JPEG directory is missing: {source_path}.")
    if target_path.exists() or target_path.is_symlink():
        raise AnchorError(f"Refusing to replace existing JPEGImages path: {target_path}.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.symlink_to(source_path, target_is_directory=True)


def copy_checkpoint(source: str | Path, destination: str | Path) -> str:
    """Publish a stable alias once, never overwrite a checkpoint."""

    source_path, destination_path = Path(source), Path(destination)
    if not source_path.is_file():
        raise AnchorError(f"PROB did not produce checkpoint {source_path}.")
    if destination_path.exists():
        raise AnchorError(f"Refusing to overwrite checkpoint {destination_path}.")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.tmp")
    if temporary.exists():
        raise AnchorError(f"Interrupted checkpoint publication requires inspection: {temporary}.")
    shutil.copy2(source_path, temporary)
    temporary.replace(destination_path)
    return longtail.sha256_file(destination_path)


def workspace_state(workspace: str | Path, condition: str) -> str:
    """Classify a persistent anchor workspace without mutating it."""

    root = Path(workspace)
    done = root / "DONE.json"
    final = root / f"t1_{condition}.pth"
    metadata = final.with_suffix(".metadata.json")
    metrics_path = root / "anchor_metrics.json"
    raw_metrics = root / "anchor_bridge_metrics.json"
    per_class = root / "per_class.csv"
    checkpoint = root / "train" / "checkpoint.pth"
    if done.is_file():
        if not all(path.is_file() for path in (final, metadata, metrics_path, per_class)):
            return "INCOMPLETE NON-RESUMABLE"
        try:
            payload = json.loads(done.read_text(encoding="utf-8"))
            metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            validate_training_metadata(metadata_payload)
        except (AnchorError, json.JSONDecodeError):
            return "INCOMPLETE NON-RESUMABLE"
        expected = {
            "checkpoint_sha256": longtail.sha256_file(final),
            "metrics_sha256": longtail.sha256_file(metrics_path),
            "per_class_csv_sha256": longtail.sha256_file(per_class),
            "condition": condition,
        }
        consistent = (
            all(payload.get(key) == value for key, value in expected.items())
            and metadata_payload.get("condition") == condition
            and metadata_payload.get("checkpoint_sha256") == expected["checkpoint_sha256"]
            and metrics_payload.get("condition") == condition
            and metrics_payload.get("checkpoint_sha256") == expected["checkpoint_sha256"]
            and payload.get("recipe_fingerprint")
            == metadata_payload.get("recipe_fingerprint")
            == metrics_payload.get("recipe_fingerprint")
        )
        return "DONE" if consistent else "INCOMPLETE NON-RESUMABLE"
    if metrics_path.exists() or raw_metrics.exists() or per_class.exists():
        return "INCOMPLETE NON-RESUMABLE"
    if final.is_file() and metadata.is_file():
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            validate_training_metadata(payload)
        except (AnchorError, json.JSONDecodeError):
            return "INCOMPLETE NON-RESUMABLE"
        if payload.get("condition") != condition \
                or payload.get("checkpoint_sha256") != longtail.sha256_file(final):
            return "INCOMPLETE NON-RESUMABLE"
        return "INCOMPLETE RESUMABLE"
    if final.exists() or metadata.exists():
        return "INCOMPLETE NON-RESUMABLE"
    if checkpoint.is_file():
        return "INCOMPLETE RESUMABLE"
    if (root / "train").exists():
        return "INCOMPLETE NON-RESUMABLE"
    return "READY"
