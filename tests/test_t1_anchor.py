"""Scientific contracts for controlled-long-tail T1 anchor preparation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
from argparse import Namespace
from pathlib import Path
from xml.etree import ElementTree

import pytest

from owl import evaluation_subset, longtail, protocol, t1_anchor

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_ROOT = ROOT / "data" / "reference" / "longtail"


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepare_tool():
    return load_tool("prepare_t1_anchor_training")


@pytest.fixture(scope="module")
def train_tool():
    return load_tool("train_t1_anchor")


@pytest.fixture(scope="module")
def compare_tool():
    return load_tool("compare_t1_anchors")


@pytest.fixture(scope="module")
def materialize_tool():
    return load_tool("materialize_t1_anchor_images")


def test_condition_manifest_resolution_and_reviewed_hashes_are_exact():
    for condition, expected in t1_anchor.EXPECTED_MANIFEST_SHA256.items():
        path, manifest = t1_anchor.condition_manifest(condition, MANIFEST_ROOT)
        assert path == MANIFEST_ROOT / f"{condition}.json"
        assert manifest["scientific_sha256"] == expected
        assert {row["class_name"] for row in manifest["classes"]} == set(protocol.TASK1)


def test_shared_evaluation_split_identity_is_exact(tmp_path):
    subset = evaluation_subset.from_archive(
        ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz",
        protocol.build_chain(6)[-1].known_classes,
        seed=0,
        remainder_multiplier=t1_anchor.EVALUATION_REMAINDER_MULTIPLIER,
        max_per_class=t1_anchor.EVALUATION_MAX_PER_CLASS,
    )
    output = tmp_path / "owl_shared_test.txt"
    evaluation_subset.write_image_set(output, subset)
    assert len(subset.image_ids) == 4308
    assert longtail.sha256_file(output) == t1_anchor.EVALUATION_SPLIT_SHA256


def recipe(condition: str, *, initialization="2" * 64):
    return t1_anchor.AnchorRecipe(
        condition=condition,
        manifest_sha256=t1_anchor.EXPECTED_MANIFEST_SHA256[condition],
        owl_commit="1" * 40,
        initialization_sha256=initialization,
        initialization_model_state_sha256="3" * 64,
        python_version="3.13.9",
        torch_version="2.11.0+cu128",
        torchvision_version="0.26.0+cu128",
        cuda_version="12.8",
    )


def test_recipe_is_identical_except_condition_manifest_and_fingerprint():
    recipes = [recipe(condition) for condition in t1_anchor.PRIMARY_CONDITIONS]
    payloads = [item.payload() for item in recipes]
    ignored = {"condition", "manifest_sha256"}
    common = [{key: value for key, value in payload.items() if key not in ignored}
              for payload in payloads]
    assert common[0] == common[1] == common[2]
    assert len({item.fingerprint() for item in recipes}) == 3
    assert all(item.class_order == protocol.TASK1 and len(item.class_order) == 19
               for item in recipes)
    assert all(item.seed == 0 and item.replay == item.active_selection == "none"
               for item in recipes)


def test_recipe_rejects_wrong_initialization_and_scientific_rebalancing():
    with pytest.raises(t1_anchor.AnchorError, match="initialization_sha256"):
        recipe("lt10", initialization="not-a-sha").validate()
    payload = recipe("lt10").payload()
    payload["oversampling"] = True
    with pytest.raises(t1_anchor.AnchorError, match="supervision frequency only"):
        t1_anchor.AnchorRecipe(**payload).validate()


def _xml(image_id: str, classes: list[str]) -> bytes:
    root = ElementTree.Element("annotation")
    ElementTree.SubElement(root, "filename").text = f"{image_id}.jpg"
    size = ElementTree.SubElement(root, "size")
    ElementTree.SubElement(size, "width").text = "64"
    ElementTree.SubElement(size, "height").text = "64"
    ElementTree.SubElement(size, "depth").text = "3"
    for name in classes:
        item = ElementTree.SubElement(root, "object")
        ElementTree.SubElement(item, "name").text = name
        box = ElementTree.SubElement(item, "bndbox")
        for field, value in zip(("xmin", "ymin", "xmax", "ymax"), (1, 1, 10, 10)):
            ElementTree.SubElement(box, field).text = str(value)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _training_metadata(checkpoint: Path, condition: str = "lt10") -> dict[str, object]:
    return {
        "schema": t1_anchor.ANCHOR_SCHEMA,
        "condition": condition,
        "manifest_sha256": "1" * 64,
        "owl_commit": "2" * 40,
        "prob_commit": "3" * 40,
        "initialization_sha256": "4" * 64,
        "recipe_fingerprint": "5" * 64,
        "seed": 0, "epochs": 41, "optimizer_steps": 10,
        "checkpoint_sha256": longtail.sha256_file(checkpoint),
        "class_order": list(protocol.TASK1),
        "train_objects": 79_233, "train_images": 37_429,
        "evaluation_split_sha256": t1_anchor.EVALUATION_SPLIT_SHA256,
        "started_at": "start", "ended_at": "end", "gpu": "T4",
        "torch_version": "x", "torchvision_version": "v", "python_version": "p",
        "cuda_version": "y", "msda": "compiled", "command": ["python"],
    }


def test_filtered_training_view_keeps_exact_selected_boxes_and_never_mutates_source(tmp_path):
    source = tmp_path / "source.tar.gz"
    first, second = "000000000001", "000000000002"
    with tarfile.open(source, "w:gz") as archive:
        for image_id, values in (
            (first, [*protocol.TASK1, protocol.TASK1[0]]),
            (second, [protocol.TASK1[1], "traffic light"]),
        ):
            payload = _xml(image_id, values)
            info = tarfile.TarInfo(f"Annotations/{image_id}.xml")
            info.size = len(payload)
            import io
            archive.addfile(info, io.BytesIO(payload))
    before = longtail.sha256_file(source)
    selection = {
        first: {name: [0] for name in protocol.TASK1},
        second: {protocol.TASK1[1]: [0]},
    }
    expected = {name: 1 for name in protocol.TASK1}
    expected[protocol.TASK1[1]] = 2
    manifest = {
        "condition": "lt10",
        "scientific_sha256": "1" * 64,
        "source_annotations": {"sha256": before},
        "classes": [
            {"class_name": name, "achieved_count": expected[name]}
            for name in protocol.TASK1
        ],
    }
    annotations = tmp_path / "view" / "Annotations"
    split = tmp_path / "view" / "ImageSets" / "OWDETR" / "owl_anchor_train.txt"
    result = t1_anchor.materialize_training_view(
        manifest=manifest, selection=selection, source_annotations=source,
        annotations_dir=annotations, split_path=split)
    assert result["images"] == 2 and result["objects"] == 20
    assert result["class_counts"] == expected
    assert split.read_text(encoding="utf-8").splitlines() == [first, second]
    kept_first = ElementTree.parse(annotations / f"{first}.xml").getroot().findall("object")
    assert len(kept_first) == 19  # the second aeroplane was filtered independently
    kept_second = ElementTree.parse(annotations / f"{second}.xml").getroot().findall("object")
    assert len(kept_second) == 1  # future-class traffic light is absent
    assert longtail.sha256_file(source) == before


def test_jpeg_validation_is_exact(tmp_path):
    (tmp_path / "000000000001.jpg").write_bytes(b"jpeg")
    t1_anchor.verify_jpegs(["000000000001"], tmp_path)
    with pytest.raises(t1_anchor.AnchorError, match="lacks 1"):
        t1_anchor.verify_jpegs(["000000000001", "000000000002"], tmp_path)


def test_workspace_isolation_and_historical_checkpoint_protection(tmp_path):
    good = tmp_path / "t1_anchor__lt10__seed0"
    t1_anchor.validate_anchor_workspace(
        workspace=good, condition="lt10", output_checkpoint=good / "t1_lt10.pth")
    with pytest.raises(t1_anchor.AnchorError, match="isolated"):
        t1_anchor.validate_anchor_workspace(
            workspace=tmp_path / "shared", condition="lt10",
            output_checkpoint=tmp_path / "shared" / "t1_lt10.pth")
    historical = tmp_path / "checkpoints" / "SOWODB" / "t1.pth"
    with pytest.raises(t1_anchor.AnchorError, match="historical"):
        t1_anchor.validate_anchor_workspace(
            workspace=good, condition="lt10", output_checkpoint=historical,
            historical_checkpoint=historical)


def test_workspace_state_distinguishes_ready_resume_done_and_corruption(tmp_path):
    workspace = tmp_path / "t1_anchor__lt10__seed0"
    assert t1_anchor.workspace_state(workspace, "lt10") == "READY"
    checkpoint = workspace / "train" / "checkpoint.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"resume")
    assert t1_anchor.workspace_state(workspace, "lt10") == "INCOMPLETE RESUMABLE"
    final = workspace / "t1_lt10.pth"
    final.write_bytes(b"final")
    metadata = _training_metadata(final)
    final.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert t1_anchor.workspace_state(workspace, "lt10") == "INCOMPLETE RESUMABLE"
    metrics_path = workspace / "anchor_metrics.json"
    metrics = {
        "condition": "lt10",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "recipe_fingerprint": metadata["recipe_fingerprint"],
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    per_class = workspace / "per_class.csv"
    per_class.write_text("condition,class_name\n", encoding="utf-8")
    done = {
        "condition": "lt10",
        "checkpoint_sha256": longtail.sha256_file(final),
        "metrics_sha256": longtail.sha256_file(metrics_path),
        "per_class_csv_sha256": longtail.sha256_file(per_class),
        "recipe_fingerprint": metadata["recipe_fingerprint"],
    }
    (workspace / "DONE.json").write_text(json.dumps(done), encoding="utf-8")
    assert t1_anchor.workspace_state(workspace, "lt10") == "DONE"
    final.write_bytes(b"tampered")
    assert t1_anchor.workspace_state(workspace, "lt10") == "INCOMPLETE NON-RESUMABLE"


def test_evaluation_annotation_copy_is_source_immutable(tmp_path):
    archive = tmp_path / "eval.tar.gz"
    payload = _xml("000000000003", [protocol.TASK1[0]])
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("Annotations/000000000003.xml")
        info.size = len(payload)
        import io
        handle.addfile(info, io.BytesIO(payload))
    before = longtail.sha256_file(archive)
    destination = tmp_path / "view" / "Annotations"
    t1_anchor.copy_evaluation_annotations(archive, destination)
    assert (destination / "000000000003.xml").read_bytes() == payload
    assert longtail.sha256_file(archive) == before


def test_protocol_only_preflight_fails_closed_and_writes_nothing(prepare_tool, tmp_path):
    arguments = Namespace(
        condition="lt10", manifest_root=MANIFEST_ROOT,
        train_annotations=ROOT / "data" / "staging" / "owdetr_replay_annotations.tar.gz",
        test_annotations=ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz",
        prob_root=tmp_path / "PROB", work_root=tmp_path / "work",
        jpeg_root=tmp_path / "JPEGImages", initialization=tmp_path / "missing.pth",
        initialization_sha="", owl_commit="", minimum_free_gib=1,
        minimum_local_free_gib=1, data_root=None,
        materialize=False, protocol_only=True,
    )
    before = list(tmp_path.rglob("*"))
    report = prepare_tool.preflight(arguments)
    assert report["execution_ready"] is False
    assert report["recipe_fingerprint"] is None
    assert report["workspace"].endswith("t1_anchor__lt10__seed0")
    assert list(tmp_path.rglob("*")) == before == []


def test_initialization_sha_and_provenance_validation(tmp_path):
    checkpoint = tmp_path / "prob_t1_seed0_init.pth"
    checkpoint.write_bytes(b"shared initialization")
    payload = {
        "schema": "controlled_t1_initialization_v1",
        "path": str(checkpoint),
        "sha256": longtail.sha256_file(checkpoint),
        "model_state_sha256": "a" * 64,
        "prob_commit": t1_anchor.PINNED_PROB_COMMIT,
        "dino_backbone_sha256": t1_anchor.DINO_SHA256,
        "seed": 0,
        "torch_version": "2.x",
        "torchvision_version": "0.x",
        "python_version": "3.13.x",
        "cuda_version": "12.x",
        "epoch": -1,
        "class_order": list(protocol.TASK1),
    }
    t1_anchor.validate_initialization_metadata(payload, checkpoint)
    checkpoint.write_bytes(b"different")
    with pytest.raises(t1_anchor.AnchorError, match="SHA-256"):
        t1_anchor.validate_initialization_metadata(payload, checkpoint)


def test_command_propagates_seed_recipe_and_no_replay_without_dry_run_writes(
    train_tool, tmp_path,
):
    data_root = tmp_path / "t1_anchor__lt10__seed0" / "data" / "OWOD"
    image_sets = data_root / "ImageSets" / "OWDETR"
    image_sets.mkdir(parents=True)
    (image_sets / "owl_anchor_train.txt").write_text("a\nb\n", encoding="utf-8")
    (image_sets / f"{t1_anchor.EVALUATION_SPLIT}.txt").write_text("c\nd\n", encoding="utf-8")
    arguments = Namespace(
        python="python", prob_root=tmp_path / "PROB",
        workspace=tmp_path / "t1_anchor__lt10__seed0",
        initialization=tmp_path / "init.pth", resume=False,
        benchmark_iterations=2,
    )
    command, _, _ = train_tool.command_for(
        arguments, recipe("lt10"), {"data_root": str(data_root)},
        smoke=True, write_smoke_splits=False)
    joined = " ".join(command)
    assert "run_prob_t1_anchor.py" in joined
    assert "--seed 0" in joined
    assert "--PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19" in joined
    assert "--batch_size 2" in joined and "--epochs 1" in joined
    assert "replay" not in joined and "exemplar" not in joined
    assert not (image_sets / "owl_anchor_smoke_train.txt").exists()


def test_prob_adapter_disables_external_wandb_without_changing_forwarded_args(tmp_path):
    prob = tmp_path / "PROB"
    prob.mkdir()
    receipt = tmp_path / "receipt.txt"
    (prob / "torch.py").write_text("class cuda:\n @staticmethod\n def synchronize(): pass\n",
                                     encoding="utf-8")
    (prob / "daowod_prob_bridge.py").write_text(
        "from contextlib import contextmanager\n"
        "@contextmanager\n"
        "def compatible_torch_load(torch):\n yield\n",
        encoding="utf-8",
    )
    (prob / "main_open_world.py").write_text(
        "import argparse, wandb\n"
        "def get_args_parser():\n"
        " p=argparse.ArgumentParser(add_help=False); "
        "p.add_argument('--wandb_project',default='PROB'); "
        "p.add_argument('--receipt'); return p\n"
        "def train_one_epoch(*args,**kwargs): return {}\n"
        "def main(args):\n"
        " wandb.log({'training': 1}); open(args.receipt,'w').write(args.wandb_project)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_prob_t1_anchor.py"),
         "--prob-root", str(prob), "--", "--wandb_project", "", "--receipt", str(receipt)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert receipt.read_text(encoding="utf-8") == ""


def test_metadata_checkpoint_hash_and_schema(tmp_path):
    checkpoint = tmp_path / "t1_lt10.pth"
    checkpoint.write_bytes(b"checkpoint")
    payload = _training_metadata(checkpoint)
    t1_anchor.validate_training_metadata(json.loads(json.dumps(payload)))
    damaged = dict(payload)
    damaged.pop("optimizer_steps")
    with pytest.raises(t1_anchor.AnchorError, match="optimizer_steps"):
        t1_anchor.validate_training_metadata(damaged)


def test_anchor_metric_schema_groups_and_condition_specific_forgetting():
    _, manifest = t1_anchor.condition_manifest("lt10", MANIFEST_ROOT)
    raw = {
        "known_AP50": 10.0,
        "per_class_AP50": {name: float(index + 1) for index, name in enumerate(protocol.TASK1)},
        "test_set_sha256": "a" * 64,
    }
    anchor = t1_anchor.anchor_metrics_payload(
        condition="lt10", manifest=manifest, bridge_metrics=raw,
        checkpoint_sha256="b" * 64, recipe_fingerprint="c" * 64)
    assert len(anchor["classes"]) == 19
    assert set(anchor["group_mAP50"]) == {"head", "medium", "tail"}
    final = {
        "condition": "lt10",
        "per_class_AP50": {name: 1.0 for name in protocol.TASK1},
    }
    rows = t1_anchor.condition_forgetting(anchor, final)
    assert len(rows) == 19
    assert rows[0]["absolute_forgetting"] == rows[0]["anchor_AP50"] - 1.0
    with pytest.raises(t1_anchor.AnchorError, match="same condition"):
        t1_anchor.condition_forgetting(anchor, final | {"condition": "lt50"})


def test_step_counts_quantify_same_epoch_image_difference():
    expected = {"lt10": 767_274, "lt50": 734_064, "lt100": 726_930}
    for condition, steps in expected.items():
        _, manifest = t1_anchor.condition_manifest(condition, MANIFEST_ROOT)
        assert t1_anchor.optimizer_steps(int(manifest["selected_images"]), recipe(condition)) == steps


def test_exact_three_condition_jpeg_union_is_stable(materialize_tool):
    image_ids = materialize_tool.required_ids(
        t1_anchor.PRIMARY_CONDITIONS,
        MANIFEST_ROOT,
        ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz",
    )
    assert len(image_ids) == 46_685
    assert image_ids == sorted(set(image_ids))
    assert all(len(image_id) == 12 and image_id.isdigit() for image_id in image_ids)


def test_dedicated_notebook_is_static_compilable_and_fail_closed():
    notebook_path = ROOT / "notebooks" / "train_controlled_lt_anchors.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    code = [source for cell, source in zip(notebook["cells"], sources)
            if cell.get("cell_type") == "code"]
    for index, source in enumerate(code):
        compile(source, f"{notebook_path.name}:cell-{index}", "exec")
    joined = "\n".join(sources)
    assert 'PROB_COMMIT = "4c66be1a52cad9360e09c729e9134aba8fe0b531"' in joined
    assert 'ALLOW_BUDGET_OVERRUN = False' in joined
    assert 'BENCHMARK_ITERATIONS = 20' in joined
    assert "pip\", \"install\", \"-r" not in notebook_path.read_text(encoding="utf-8")
    assert "CONTROLLED LT ANCHOR PREFLIGHT PASS" not in joined  # emitted only by the tool
    assert "ANCHOR BENCHMARK/RESUME POINT COMPLETE" in joined


def test_combined_report_requires_valid_done_anchors(compare_tool, tmp_path):
    condition = "lt10"
    workspace = tmp_path / f"t1_anchor__{condition}__seed0"
    workspace.mkdir()
    final = workspace / f"t1_{condition}.pth"
    final.write_bytes(b"anchor")
    metadata = _training_metadata(final, condition)
    final.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    classes = [
        {
            "class_name": name,
            "train_count": index + 1,
            "rank": index + 1,
            "group": "head" if index < 7 else "medium" if index < 13 else "tail",
            "anchor_AP50": float(index),
        }
        for index, name in enumerate(protocol.TASK1)
    ]
    metrics = {
        "condition": condition,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "recipe_fingerprint": metadata["recipe_fingerprint"],
        "overall_mAP50": 9.0,
        "group_mAP50": {"head": 3.0, "medium": 9.5, "tail": 15.5},
        "learnability_descriptives": {
            "spearman_AP50_log_train_frequency": 1.0,
            "minimum_AP50": 0.0,
            "exact_zero_AP50_classes": [protocol.TASK1[0]],
        },
        "classes": classes,
    }
    metrics_path = workspace / "anchor_metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    per_class = workspace / "per_class.csv"
    per_class.write_text("condition,class_name\n", encoding="utf-8")
    done = {
        "condition": condition,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "metrics_sha256": longtail.sha256_file(metrics_path),
        "per_class_csv_sha256": longtail.sha256_file(per_class),
        "recipe_fingerprint": metadata["recipe_fingerprint"],
    }
    (workspace / "DONE.json").write_text(json.dumps(done), encoding="utf-8")
    payload = compare_tool.compare(tmp_path, (condition,), tmp_path / "comparison")
    assert payload["incremental_training_authorized"] is False
    assert len(payload["classes"]) == 19
    assert (tmp_path / "comparison" / "anchor_summary.csv").is_file()
