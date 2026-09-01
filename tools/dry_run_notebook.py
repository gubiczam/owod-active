"""Execute every notebook cell, GPU branch included, with PROB and Colab faked.

The point: two bugs reached the user's GPU session — a preflight that left a name
undefined, and a chain that asked the detector for images nobody had downloaded.
Neither was findable by unit-testing ``owl``, because every part was correct on
its own. What was wrong was the *notebook*, and nothing was running it.

So this runs it. One namespace, cells in order, exactly as Colab would, with the
things a laptop cannot provide replaced:

* ``google.colab.drive`` — a stub;
* ``subprocess.run`` — git, pip, curl and nvidia-smi answered locally, so no
  network and no clone;
* ``owl.bridge.Bridge`` — a fake that writes plausible proposals, checkpoints
  and metrics, and records what PROB would have been asked to do.

Everything else is the real code on the real committed data. If this passes, the
notebook's control flow works; what it cannot check is whether PROB itself likes
the arguments.

    python tools/dry_run_notebook.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path
from xml.etree import ElementTree as _ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "owod_active.ipynb"
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- the fakes ---


from owl.evaluation_subset import check_split_name


class FakeBridge:
    """Stands in for PROB. Writes output shaped the way the real bridge does."""

    #: The evaluator's class order, in PROB's own indexing. An annotation's
    #: category_id is its position here, which is what the filtering ranges over.
    CLASS_ORDER: tuple[str, ...] = ()

    def __init__(self, *, prob_root, data_root, feature_dim=64, **_):
        from owl import protocol

        self.prob_root = Path(prob_root)
        self.data_root = Path(data_root)
        self.feature_dim = feature_dim
        self.calls: list[dict] = []
        self.CLASS_ORDER = protocol.CLASS_ORDER

    def _boxes_after_filtering(self, image_id: str, n_prev: int, n_current: int) -> int:
        """How many boxes survive `remove_unknown_instances`.

        PROB keeps ``category_id in range(0, prev + current)`` on a fine-tuning
        split. An image whose objects all fall outside that range arrives with
        zero boxes, and the collate function fails on it rather than skipping it.
        Modelling this is the whole point: without it the fake accepts input the
        real loader rejects.
        """

        from xml.etree import ElementTree

        from owl.evaluation_subset import canonical_class_name

        path = self.data_root / "Annotations" / f"{image_id}.xml"
        known = set(self.CLASS_ORDER[: n_prev + n_current])
        root = ElementTree.parse(path).getroot()
        return sum(
            1 for element in root.findall("object")
            if canonical_class_name(element.findtext("name", "")) in known
        )

    def check(self):
        return {"fake": True, "prob_root": str(self.prob_root)}

    def predict(self, image_ids, *, checkpoint, output, n_prev, n_current,
                max_proposals_per_image=50):
        image_ids = [str(v) for v in image_ids]
        missing = [i for i in image_ids
                   if not (self.data_root / "JPEGImages" / f"{i}.jpg").exists()]
        if missing:
            # exactly how the real run failed
            raise RuntimeError(
                f"PROB predict would fail: {len(missing)} of {len(image_ids)} images "
                f"are not on disk, e.g. {missing[:3]}"
            )
        for image_id in image_ids:
            annotation = self.data_root / "Annotations" / f"{image_id}.xml"
            if not annotation.exists():
                raise RuntimeError(f"PROB predict would fail: missing {annotation}")
        self.calls.append({"verb": "predict", "images": image_ids,
                           "n_prev": n_prev, "n_current": n_current})
        output = Path(output)
        if output.exists():
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(len(self.calls))
        rows = len(image_ids) * min(max_proposals_per_image, 8)
        known = max(n_prev + n_current, 1)
        np.savez_compressed(
            output,
            image_ids=np.asarray([image_ids[i % len(image_ids)] for i in range(rows)],
                                 dtype=object),
            confidence=generator.random(rows),
            embeddings=generator.normal(size=(rows, self.feature_dim)),
            posterior=generator.random((rows, known + 1)),
            predicted_labels=generator.integers(0, known, rows),
            boxes=generator.random((rows, 4)) * 0.5 + 0.25,
            objectness=generator.random(rows),
        )
        output.with_suffix(".json").write_text(json.dumps({"proposal_count": rows}))
        return output

    def train(self, labelled_ids, *, previous_checkpoint, output_checkpoint, output_dir,
              n_prev, n_current, test_set, replay_ids=(), supervision_mode="ft",
              eval_every=10**6, **_):
        assert test_set, "train was not told which test set to build the val loader from"
        check_split_name(test_set, purpose="test")
        split = self.data_root / "ImageSets" / "OWDETR" / f"{test_set}.txt"
        assert split.exists(), (
            f"PROB train would fail: it builds a validation dataset from {split}, "
            "which does not exist")
        assert Path(previous_checkpoint).exists(), (
            f"PROB train would fail: no checkpoint at {previous_checkpoint}")
        empty = []
        for image_id in list(labelled_ids) + list(replay_ids):
            assert (self.data_root / "Annotations" / f"{image_id}.xml").exists(), (
                f"PROB train would fail: no annotation for {image_id}")
            if self._boxes_after_filtering(image_id, n_prev, n_current) == 0:
                empty.append(image_id)
        assert not empty, (
            f"PROB train would fail: {len(empty)} of "
            f"{len(labelled_ids) + len(replay_ids)} images arrive with zero boxes "
            f"after remove_unknown_instances, e.g. {empty[:3]}. The collate "
            "function raises 'size of tensor a (0) must match the size of tensor "
            "b (4)' on the first one."
        )
        self.calls.append({"verb": "train", "images": list(labelled_ids),
                           "replay": list(replay_ids), "n_prev": n_prev,
                           "n_current": n_current, "supervision": supervision_mode,
                           "test_set": test_set})
        output_checkpoint = Path(output_checkpoint)
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_checkpoint.write_bytes(b"fake checkpoint")
        output_checkpoint.with_suffix(".train.json").write_text(json.dumps({
            "previous_checkpoint": str(previous_checkpoint),
            "output_checkpoint": str(output_checkpoint),
        }), encoding="utf-8")
        return output_checkpoint

    def evaluate(self, *, checkpoint, test_set, output, n_prev, n_current,
                 detections=True, **_):
        output = Path(output)
        if output.exists():
            return output
        check_split_name(test_set, purpose="test")
        assert Path(checkpoint).exists()
        split = self.data_root / "ImageSets" / "OWDETR" / f"{test_set}.txt"
        assert split.exists(), f"PROB evaluate would fail: no image set at {split}"
        split_ids = split.read_text().split()
        for image_id in split_ids:
            assert (self.data_root / "JPEGImages" / f"{image_id}.jpg").exists(), (
                f"PROB evaluate would fail: test image {image_id} is not on disk")
        self.calls.append({"verb": "evaluate", "n_prev": n_prev,
                           "n_current": n_current, "test_set": test_set})
        output.parent.mkdir(parents=True, exist_ok=True)
        step = len([c for c in self.calls if c["verb"] == "evaluate"])
        payload = {
            # every aggregate is a mean over a slice of the same AP array the
            # file publishes as coco_eval_bbox, and PK_AP50 is absent when no
            # class has been introduced yet — exactly what the bridge writes.
            "known_AP50": (
                sum(float(i % 40) for i in range(n_prev + n_current))
                / (n_prev + n_current) if n_prev + n_current else 0.0),
            "U_Recall": 20.0 - step,
            "previous_known_AP50": (
                sum(float(i % 40) for i in range(n_prev)) / n_prev if n_prev else None),
            "current_known_AP50": (
                sum(float(i % 40) for i in range(n_prev, n_prev + n_current))
                / n_current if n_current else 0.0),
            "unknown_AP50": 0.4, "WI": 0.03, "A_OSE": 1200,
            "test_set": test_set,
            "coco_eval_bbox": [30.0, 30.0, *[float(i % 40) for i in range(80)], 0.4],
                }
        output.write_text(json.dumps(payload), encoding="utf-8")

        if detections:
            # the same shape the bridge writes, so the grouped-recall reader is
            # exercised rather than merely imported
            from owl import protocol as _protocol

            artefact = output.with_name(f"{output.stem}_detections.json")
            unknown = _protocol.CLASS_ORDER[n_prev + n_current:][:6]
            truth, found = [], []
            # Every split id appears first, in exact split order. The anchor tool
            # validates this recorded order before it will bless historical data.
            for index, image_id in enumerate(split_ids):
                name = _protocol.TASK1[index % len(_protocol.TASK1)]
                box = [0.0, 0.0, 8.0, 8.0]
                truth.append({"image_id": image_id, "class_name": name, "box": box})
                found.append({"image_id": image_id, "class_name": name,
                              "score": 0.95, "box": box})
            for index, name in enumerate(unknown):
                box = [10.0 * index, 0.0, 10.0 * index + 8.0, 8.0]
                truth.append({"image_id": split_ids[0], "class_name": name, "box": box})
                if index % 2 == 0:                      # half of them recalled
                    found.append({"image_id": split_ids[0], "class_name": "unknown",
                                  "score": 0.9, "box": box})
            artefact.write_text(json.dumps({
                "schema": "daowod_detections_v1", "unknown_class_name": "unknown",
                "test_set": test_set, "dataset": "OWDETR",
                "image_count": len(split_ids),
                "class_names": [*_protocol.CLASS_ORDER, "unknown"],
                "previous_introduced_classes": n_prev,
                "current_introduced_classes": n_current,
                "ground_truth": truth, "detections": found,
            }), encoding="utf-8")
            payload["detections_path"] = str(artefact)
            output.write_text(json.dumps(payload), encoding="utf-8")
        return output

    def cost_report(self):
        return {"total": float(len(self.calls)), "calls": len(self.calls)}


def fake_subprocess(jpeg_dir: Path):
    """git / pip / curl / nvidia-smi, answered without a network."""

    import re
    import subprocess as real

    installed_versions = {
        "pandas": "2.3.2", "seaborn": "0.13.2", "tqdm": "4.67.1",
    }

    def run(command, **kwargs):
        text = [str(part) for part in command]
        joined = " ".join(text)
        if any(part.endswith(("evaluate_anchor.py", "compare_replay.py")) for part in text):
            return real.run(
                command,
                check=kwargs.pop("check", False),
                **kwargs,
            )
        if "nvidia-smi" in joined:
            return real.CompletedProcess(command, 0, "Tesla T4, 15360 MiB\n", "")
        if len(text) >= 4 and text[1:4] == ["-m", "pip", "check"]:
            if "jedi" not in installed_versions:
                conflict = "ipython 7.34.0 requires jedi, which is not installed.\n"
                return real.CompletedProcess(command, 1, conflict, "")
            return real.CompletedProcess(command, 0, "No broken requirements found.\n", "")
        if len(text) >= 4 and text[1:3] == ["-m", "pip"] and "install" in text:
            for part in text:
                match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ]+)", part)
                if match:
                    installed_versions[match.group(1).lower()] = match.group(2)
            return real.CompletedProcess(command, 0, "", "")
        if "-c" in text:
            script = text[text.index("-c") + 1]
            if "OWOD_RAW_MSDA_PROBE=" in script:
                payload = {
                    "ok": False,
                    "path": None,
                    "error": "ImportError: dry-run extension needs torch loaded first",
                }
                return real.CompletedProcess(
                    command, 0, "OWOD_RAW_MSDA_PROBE=" + json.dumps(payload) + "\n", "")
            if "OWOD_PROB_MSDA_PROBE=" in script:
                prob_root = Path(kwargs.get("cwd", ROOT))
                payload = {
                    "probe_ok": True,
                    "python": "3.13.7",
                    "executable": text[0],
                    "torch": "2.8.0+cu126",
                    "torchvision": "0.23.0+cu126",
                    "torch_cuda": "12.6",
                    "cuda_available": True,
                    "gpu": "Tesla T4 (dry run)",
                    "numpy": np.__version__,
                    "scipy": "1.16.1",
                    "sklearn": "1.7.1",
                    "pillow": "11.3.0",
                    "matplotlib": "3.10.5",
                    "pandas": "2.3.2",
                    "einops": "0.5.0",
                    "pycocotools": "2.0.5",
                    "extension_after_torch": {
                        "ok": True, "path": "/fake/MultiScaleDeformableAttention.so",
                        "error": None,
                    },
                    "available": True,
                    "backend": "compiled",
                    "wrapper_path": str(prob_root / "models" / "ops" / "functions" /
                                        "ms_deform_attn_func.py"),
                    "downstream_path": str(prob_root / "models" / "ops" / "modules" /
                                           "ms_deform_attn.py"),
                    "extension_path": "/fake/MultiScaleDeformableAttention.so",
                    "error": None,
                }
                return real.CompletedProcess(
                    command, 0, "OWOD_PROB_MSDA_PROBE=" + json.dumps(payload) + "\n", "")
            if "OWOD_MSDA_RESULT=" in script:
                payload = {
                    "available": True,
                    "backend": "compiled",
                    "dispatch_counts": {"compiled": 12, "fallback": 0},
                }
                output = (
                    "OWOD_MSDA_RESULT=" + json.dumps(payload) + "\n"
                    "MSDA backend: compiled\n"
                    "PROB CUDA model/loss/evaluator smoke: PASS\n"
                )
                return real.CompletedProcess(command, 0, output, "")
            version_match = re.search(r"version\(['\"]([^'\"]+)['\"]\)", script)
            if version_match:
                version = installed_versions.get(version_match.group(1).lower())
                return real.CompletedProcess(
                    command, 0 if version else 1, (version + "\n") if version else "", "")
            direct_import = re.fullmatch(r"import ([A-Za-z0-9_.]+)", script)
            if direct_import:
                module = direct_import.group(1).split(".", 1)[0].lower()
                return real.CompletedProcess(
                    command, 0 if module in installed_versions else 1, "", "")
            # The CUDA and complete PROB runtime probes are successful fakes.
            return real.CompletedProcess(command, 0, "fake subprocess probe: PASS\n", "")
        if text[0] == "which":
            return real.CompletedProcess(command, 0, "/usr/local/cuda/bin/nvcc\n", "")
        if text[0] == "curl":
            output_option = "--output" if "--output" in text else "-o"
            target = Path(text[text.index(output_option) + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8)).save(target, format="JPEG")
            return real.CompletedProcess(command, 0, "", "")
        if "git" in text[0] and text[1:4] == ["remote", "get-url", "origin"]:
            cwd = Path(kwargs.get("cwd", ROOT))
            repository = ("https://github.com/gubiczam/PROB.git"
                          if cwd.name == "PROB" else
                          "https://github.com/gubiczam/owod-active.git")
            return real.CompletedProcess(command, 0, repository + "\n", "")
        if "git" in text[0] and text[1:3] == ["rev-parse", "HEAD"]:
            cwd = Path(kwargs.get("cwd", ROOT))
            commit = ("4c66be1a52cad9360e09c729e9134aba8fe0b531"
                      if cwd.name == "PROB" else
                      "ae2d2ab1bdeb7a9c30992448d0a839c3458451e9")
            return real.CompletedProcess(command, 0, commit + "\n", "")
        return real.CompletedProcess(command, 0, "", "")

    return run


# ------------------------------------------------------------------- harness ---

#: Lines rewritten before execution, each of which MUST match exactly once. A
#: notebook edit that breaks one of these fails the dry run loudly rather than
#: letting it test something other than what ships.
def substitutions(workspace: Path) -> list[tuple[str, str]]:
    return [
        ('Path("/content/owod-active")', f'Path("{ROOT}")'),
        ('DATA = Path("/content/data/OWOD")', f'DATA = Path("{workspace / "OWOD"}")'),
        ('Path("/content/PROB")', f'Path("{workspace / "PROB"}")'),
        ('Path("/content/owod_preflight_comparison")',
         f'Path("{workspace / "precompare"}")'),
        ('Path("/content/owod_comparison_replay_v3_fast_seed0")',
         f'Path("{workspace / "comparison"}")'),
        ('Path("/content/owod_no_replay_compatibility_view")',
         f'Path("{workspace / "compatibility_view"}")'),
        ('DRIVE_FREE_GB >= 8.0', 'DRIVE_FREE_GB >= 0.0'),
        ('LOCAL_FREE_GB >= 12.0', 'LOCAL_FREE_GB >= 0.0'),
    ]


def cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def run(run_gpu: bool, *, verbose: bool) -> None:
    from owl import bridge as original_bridge_module

    original_bridge_class = original_bridge_module.Bridge
    workspace = Path(tempfile.mkdtemp(prefix="owl-dry-"))
    drive_root = workspace / "drive" / "MyDrive" / "OWL"
    (drive_root / "checkpoints" / "SOWODB").mkdir(parents=True, exist_ok=True)
    (drive_root / "checkpoints" / "SOWODB" / "t1.pth").write_bytes(b"fake t1")
    prob_root = workspace / "PROB"
    (prob_root / ".git").mkdir(parents=True)
    (prob_root / "models" / "ops").mkdir(parents=True)
    (prob_root / "requirements.txt").write_text("", encoding="utf-8")

    # google.colab
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
    google = types.ModuleType("google")
    google.colab = colab
    sys.modules.setdefault("google", google)
    sys.modules["google.colab"] = colab

    fake_run = fake_subprocess(workspace)

    # A CUDA-capable torch surface is enough for the notebook's preflight. OWL's
    # exercised code is numpy/scikit-learn and never reaches this stub.
    torch = types.ModuleType("torch")
    torch.__version__ = "2.8.0-dry"
    torch.version = types.SimpleNamespace(cuda="12.6")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda _: "Tesla T4 (dry run)")
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = torch

    namespace: dict = {}
    legacy_baseline_config_path: Path | None = None
    legacy_baseline_config_bytes: bytes | None = None
    try:
        for index, cell in enumerate(cells()):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])

            # Static dry-runs exercise notebook control flow under the repository's
            # local test interpreter; the production cell retains its Python-3.13 gate.
            source = source.replace(
                "assert sys.version_info[:2] == (3, 13), sys.version",
                "assert len(sys.version_info[:2]) == 2, sys.version",
            )

            if "# ============================== PARAMETERS" in source:
                source = source.replace("RUN_GPU = True", f"RUN_GPU = {run_gpu}")
                source = source.replace(
                    'DRIVE_ROOT = "/content/drive/MyDrive/OWL"',
                    f'DRIVE_ROOT = "{drive_root}"')
                # Preserve the production assertions, then shrink subsequent cells.
                source += (
                    "\nN_TASKS, BUDGET_PER_TASK, ROUNDS_PER_TASK = 4, 40, 2"
                    "\nCANDIDATE_IMAGES, PROPOSALS_PER_IMAGE = 60, 4"
                    "\nEVAL_MAX_PER_CLASS, EVAL_REMAINDER_RATIO = 3, 0"
                    "\nN_CLUSTERS, TIME_BUDGET_MINUTES = 64, 10_000\n"
                )

            for before, after in substitutions(workspace):
                if before in source:
                    assert source.count(before) == 1, f"{before!r} matched twice"
                    source = source.replace(before, after)

            namespace["subprocess"] = types.SimpleNamespace(run=fake_run)
            buffer = io.StringIO()
            try:
                with redirect_stdout(buffer):
                    exec(compile(source, f"cell {index}", "exec"), namespace)  # noqa: S102
            except Exception:
                print(buffer.getvalue())
                print(f"\n*** cell {index} raised ***\n")
                raise
            # The environment cell purges sys.modules and re-imports owl, which
            # restores the real Bridge. So the fake goes in *after* it runs, not
            # before — the same staleness trap the cell exists to close.
            if "from owl import" in source:
                namespace["bridge"].Bridge = FakeBridge

            # The real Drive input is a completed historical baseline. Seed the
            # same shape after canonical data preparation and before preflight.
            if source.startswith("# 4 —"):
                assert namespace["fetch_images"](namespace["subset"].image_ids) == \
                    list(namespace["subset"].image_ids)
                config = namespace["runner"].CycleConfig(
                    n_tasks=namespace["N_TASKS"],
                    budget_per_task=namespace["BUDGET_PER_TASK"],
                    rounds_per_task=namespace["ROUNDS_PER_TASK"],
                    candidate_images_per_task=namespace["CANDIDATE_IMAGES"],
                    proposals_per_image=namespace["PROPOSALS_PER_IMAGE"],
                    arm="random", labelling_policy="known_plus_selected",
                    replay_arm="none", replay_reallocate=False,
                    replay_protocol_version=3, epochs=namespace["EPOCHS"],
                    learning_rate=namespace["LEARNING_RATE"],
                    batch_size=namespace["BATCH_SIZE"],
                    n_clusters=namespace["N_CLUSTERS"], seed=namespace["SEED"],
                    measure_grouped_recall=True,
                )
                baseline_bridge = FakeBridge(
                    prob_root=prob_root, data_root=namespace["DATA"])
                namespace["runner"].run_chain(
                    baseline_bridge, config,
                    workspace=drive_root / "work" / "random__none",
                    candidate_index=namespace["candidate_index"],
                    replay_index=namespace["replay_index"],
                    replay_root=namespace["DATA"],
                    start_checkpoint=drive_root / "checkpoints" / "SOWODB" / "t1.pth",
                    test_set=namespace["TEST_SET"], chain=namespace["chain"],
                    prepare_images=namespace["fetch_images"],
                )
                legacy_baseline_config_path = (
                    drive_root / "work" / "random__none" / "config.json")
                legacy_config = json.loads(legacy_baseline_config_path.read_text())
                assert legacy_config.pop("replay_protocol_version") == 3
                legacy_baseline_config_path.write_text(
                    json.dumps(legacy_config, indent=2), encoding="utf-8")
                baseline_states = sorted(
                    (drive_root / "work" / "random__none").glob("t*_random/state.json"))
                assert len(baseline_states) == 3
                legacy_states = [json.loads(path.read_text()) for path in baseline_states]
                legacy_states[0]["replay_row"] = {}
                legacy_states[0].pop("exemplars", None)
                legacy_states[1]["replay_row"] = {"images": 0, "per_class": ""}
                legacy_states[1]["exemplars"] = []
                legacy_states[2].pop("replay_row", None)
                legacy_states[2].pop("exemplars", None)
                for path, state in zip(baseline_states, legacy_states, strict=True):
                    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                legacy_baseline_config_bytes = legacy_baseline_config_path.read_bytes()

            if verbose:
                print(f"--- cell {index} ---")
                print(buffer.getvalue().rstrip() or "(no output)")
            else:
                print(f"cell {index:2d} ok")

        # Run the orchestration/reporting tail a second time against the same
        # workspaces. No predict/train/evaluate call may be added: this is the
        # notebook-level proof that a reconnect + Run all reuses valid work.
        if run_gpu:
            calls_before_rerun = len(namespace["prob_bridge"].calls)
            for index, item in enumerate(cells()):
                if item["cell_type"] != "code":
                    continue
                source = "".join(item["source"])
                if not any(source.startswith(f"# {stage} —") for stage in range(7, 13)):
                    continue
                for before, after in substitutions(workspace):
                    source = source.replace(before, after)
                namespace["subprocess"] = types.SimpleNamespace(run=fake_run)
                with redirect_stdout(io.StringIO()):
                    exec(compile(source, f"rerun cell {index}", "exec"), namespace)  # noqa: S102
            assert len(namespace["prob_bridge"].calls) == calls_before_rerun
            assert namespace["RUN_STATUS"]["random__uniform"] == "validated and skipped"
            assert namespace["RUN_STATUS"]["random__tail_favouring"] == \
                "validated and skipped"
            print("rerun idempotency: valid replay work was skipped with no new PROB calls")
    finally:
        original_bridge_module.Bridge = original_bridge_class
        sys.modules["owl.bridge"] = original_bridge_module
        import owl as owl_package
        owl_package.bridge = original_bridge_module
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch

    # ---- what the run must have achieved ---------------------------------
    if run_gpu:
        fake = namespace["prob_bridge"]
        by_arm = namespace["by_arm"]
        assert namespace["PROB_COMPAT_INSTALLED"] == [
            "einops==0.5.0", "pycocotools==2.0.5", "wandb==0.18.7",
            "jedi==0.19.2",
        ]
        for label in (
            "CUDA model smoke", "package consistency", "baseline fingerprint",
            "target fingerprints", "Replay Protocol V3",
        ):
            assert namespace["checks"][label], label
            print(f"{label:30s} PASS")
        assert namespace["LEGACY_BASELINE_COMPATIBILITY"] == {
            "workspace": "random__none",
            "stored": "absent",
            "normalized_for_compatibility": "no-replay-only",
            "reason": "replay_arm=none; replay protocol inactive",
            "historical_config_modified": False,
        }
        assert legacy_baseline_config_path is not None
        assert legacy_baseline_config_path.read_bytes() == legacy_baseline_config_bytes
        assert namespace["summary"]["legacy_baseline_replay_protocol"] == \
            namespace["LEGACY_BASELINE_COMPATIBILITY"]
        assert set(by_arm) == {"random__uniform", "random__tail_favouring"}, sorted(by_arm)
        verbs = [call["verb"] for call in fake.calls]
        # each arm scores the starting checkpoint once — that is what task 2
        # measures its forgetting against — and then runs the cycle per task
        assert verbs == (["evaluate"] + ["predict", "train", "evaluate"] * 3) * 2, verbs
        # The production notebook refuses a missing old-data index. When it is
        # present, both configured runs must actually rehearse through aliases.
        index_path = ROOT / "data" / "reference" / "t1_replay_class_counts.json"
        assert index_path.exists() and namespace["replay_index"]
        from owl import exemplars as _exemplars
        from owl import replay as _replay

        for call in [c for c in fake.calls if c["verb"] == "train"]:
            assert call["replay"], "configured replay training received no aliases"
            assert all(str(i).startswith("9") for i in call["replay"])
        for run, rows in by_arm.items():
            budget = _replay.ARMS[run.rsplit("__", 1)[1]]["total"]
            for row in rows:
                diagnostics = row.replay_row
                assert (
                    diagnostics["requested_objects"]
                    == diagnostics["allocated_objects"]
                    == diagnostics["delivered_objects"]
                    == budget
                ), f"{run} {row.task}: object budget not held: {diagnostics}"
        written = list((namespace["DATA"] / "Annotations").glob("9*.xml"))
        assert written, "no replay alias annotation was written"
        boxes = sum(len(_ET.parse(path).getroot().findall("object")) for path in written)
        print(f"replay branch exercised: {len(namespace['replay_index'])} "
              f"old-data images, arms {tuple(namespace['REPLAY_ARMS'])}, "
              f"{len(written)} alias annotations holding {boxes} boxes; "
              "every run held its own object budget on every task")
        assert _exemplars.source_id(written[0].stem).startswith("0")

        for arm, results in by_arm.items():
            assert len(results) == 3, f"{arm}: expected three tasks, got {len(results)}"
            for row in results:
                flat = row.flat()
                for column in ("known_mAP50", "prev_mAP50", "new_mAP50", "U_Recall50",
                               "U_Recall_tail", "oracle_cost_so_far", "forgetting",
                               "drop_from_anchor"):
                    assert flat.get(column) is not None, f"{arm} {row.task}: no {column}"
        assert namespace["EXPERIMENT_COMPLETE"] is True
        assert (drive_root / "comparisons" / "replay_v3_fast_seed0" / "summary.json").is_file()
        print("\nGPU branch: pinned baseline + 2 replay runs passed all notebook "
              "audits, comparison generation, and persistence.")
    else:
        raise AssertionError("the production notebook intentionally requires a GPU")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print each cell's output")
    arguments = parser.parse_args()
    run(run_gpu=True, verbose=arguments.verbose)
    print("DRY RUN PASSED")


if __name__ == "__main__":
    main()
