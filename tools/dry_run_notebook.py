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

    python tools/dry_run_notebook.py            # GPU branch
    python tools/dry_run_notebook.py --cpu      # CPU branch
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

import numpy as np

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
        return output_checkpoint

    def evaluate(self, *, checkpoint, test_set, output, n_prev, n_current, **_):
        check_split_name(test_set, purpose="test")
        assert Path(checkpoint).exists()
        split = self.data_root / "ImageSets" / "OWDETR" / f"{test_set}.txt"
        assert split.exists(), f"PROB evaluate would fail: no image set at {split}"
        for image_id in split.read_text().split():
            assert (self.data_root / "JPEGImages" / f"{image_id}.jpg").exists(), (
                f"PROB evaluate would fail: test image {image_id} is not on disk")
            break
        self.calls.append({"verb": "evaluate", "n_prev": n_prev,
                           "n_current": n_current, "test_set": test_set})
        output = Path(output)
        if output.exists():
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        step = len([c for c in self.calls if c["verb"] == "evaluate"])
        output.write_text(json.dumps({
            "known_AP50": 60.0 - step, "U_Recall": 20.0 - step,
            "previous_known_AP50": 70.0 - 2 * step, "current_known_AP50": 3.0 + step,
            "unknown_AP50": 0.4, "WI": 0.03, "A_OSE": 1200, "per_class_AP50": {},
        }), encoding="utf-8")
        return output

    def cost_report(self):
        return {"total": float(len(self.calls)), "calls": len(self.calls)}


def fake_subprocess(jpeg_dir: Path):
    """git / pip / curl / nvidia-smi, answered without a network."""

    import subprocess as real

    def run(command, **kwargs):
        text = [str(part) for part in command]
        joined = " ".join(text)
        if "nvidia-smi" in joined:
            return real.CompletedProcess(command, 0, "Tesla T4, 15360 MiB\n", "")
        if text[0] == "curl":
            target = Path(text[text.index("-o") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\xff\xd8\xff\xe0fake jpeg")
            return real.CompletedProcess(command, 0, "", "")
        if "git" in text[0] and "log" in text:
            return real.CompletedProcess(command, 0, "deadbee dry run\n", "")
        return real.CompletedProcess(command, 0, "", "")

    return run


# ------------------------------------------------------------------- harness ---

#: Lines rewritten before execution, each of which MUST match exactly once. A
#: notebook edit that breaks one of these fails the dry run loudly rather than
#: letting it test something other than what ships.
def substitutions(workspace: Path) -> list[tuple[str, str]]:
    return [
        ('ROOT = Path("/content/owod-active")', f'ROOT = Path("{ROOT}")'),
        ('subprocess.run(["rm", "-rf", str(ROOT)], check=True)', "pass"),
        ('subprocess.run(["git", "clone", "--depth", "1", OWL_REPOSITORY, str(ROOT)], check=True)',
         "pass"),
        ('subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)], check=True)',
         "pass"),
        ('DATA = Path("/content/data/OWOD")', f'DATA = Path("{workspace / "OWOD"}")'),
        ('PROB = bridge.ensure_checkout(Path("/content/PROB"))',
         f'PROB = Path("{workspace / "PROB"}"); PROB.mkdir(parents=True, exist_ok=True)'),
    ]


def cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def run(run_gpu: bool, *, verbose: bool) -> None:
    workspace = Path(tempfile.mkdtemp(prefix="owl-dry-"))
    drive_root = workspace / "drive" / "MyDrive" / "OWL"
    (drive_root / "checkpoints" / "SOWODB").mkdir(parents=True, exist_ok=True)
    (drive_root / "checkpoints" / "SOWODB" / "t1.pth").write_bytes(b"fake t1")

    # google.colab
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
    google = types.ModuleType("google")
    google.colab = colab
    sys.modules.setdefault("google", google)
    sys.modules["google.colab"] = colab

    fake_run = fake_subprocess(workspace)

    namespace: dict = {}
    try:
        for index, cell in enumerate(cells()):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])

            if index == 2:  # the parameters cell
                source = source.replace("RUN_GPU = True", f"RUN_GPU = {run_gpu}")
                source = source.replace("RUN_GPU = False", f"RUN_GPU = {run_gpu}")
                source = source.replace(
                    'DRIVE_ROOT = "/content/drive/MyDrive/OWL"',
                    f'DRIVE_ROOT = "{drive_root}"')
                # keep the dry run quick: the CPU sections are the slow part
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

            if verbose:
                print(f"--- cell {index} ---")
                print(buffer.getvalue().rstrip() or "(no output)")
            else:
                print(f"cell {index:2d} ok")
    finally:
        pass

    # ---- what the run must have achieved ---------------------------------
    if run_gpu:
        fake = namespace["prob_bridge"]
        verbs = [call["verb"] for call in fake.calls]
        assert verbs == ["predict", "train", "evaluate"] * 3, verbs
        results = namespace["gpu_results"]
        assert len(results) == 3, f"expected three tasks, got {len(results)}"
        for row in results:
            flat = row.flat()
            for column in ("known_mAP50", "prev_mAP50", "new_mAP50", "U_Recall50"):
                assert flat.get(column) is not None, f"{row.task} has no {column}"
        print("\nGPU branch: 3 tasks, 9 PROB calls, every metric present.")
    else:
        assert namespace["gpu_results"] == []
        print("\nCPU branch: complete, GPU chain correctly skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="run the RUN_GPU=False branch")
    parser.add_argument("--verbose", action="store_true", help="print each cell's output")
    arguments = parser.parse_args()
    run(run_gpu=not arguments.cpu, verbose=arguments.verbose)
    print("DRY RUN PASSED")


if __name__ == "__main__":
    main()
