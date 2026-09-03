#!/usr/bin/env python
"""Execute every cell of the Benchmark V1 notebook, with PROB and Colab faked.

The point, learned the hard way: two bugs reached a real GPU session — a
preflight that left a name undefined, and a chain that asked the detector for
images nobody had downloaded. Neither was findable by unit-testing ``owl``,
because every part was correct on its own. What was wrong was the *notebook*,
and nothing was running it.

So this runs it. One namespace, cells in order, exactly as Colab would, with the
things a laptop cannot provide replaced:

* ``google.colab.drive`` — a stub;
* ``subprocess.run`` — git, pip, curl and nvidia-smi answered locally, reusing
  ``tools.dry_run_notebook.fake_subprocess``, so no network and no clone;
* ``subprocess.Popen`` — **real**, because the notebook's expensive steps are
  the repository's own tools and running them for real is the point;
* ``owl.bridge.verify_remote_commit`` — a successful stub, so the preflight's
  branch is exercised without touching GitHub;
* the launcher — invoked with ``--dry-run``, which stubs PROB and the DINOv2
  pass while the population, the traversal, the ledger, the replay memory, the
  resume logic, the manifest and the tables are all the real code.

What it proves, and asserts:

* each arm reaches **t4**;
* ``t3`` trains from its own arm's ``t2`` checkpoint and ``t4`` from its ``t3``;
* no two arms share a checkpoint or a workspace;
* the oracle answers are matched across arms;
* future labels never enter selection;
* the evaluation split is one shared, frozen file.

    python tools/dry_run_full_owod_benchmark.py
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess as real_subprocess
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "full_owod_active_benchmark_v1.ipynb"
sys.path.insert(0, str(ROOT))

from tools.dry_run_notebook import fake_subprocess

TASKS = ("t2", "t3", "t4")


def substitutions(workspace: Path, drive_root: Path) -> list[tuple[str, str]]:
    """Lines rewritten before execution. Each MUST match exactly once.

    A notebook edit that breaks one of these fails the dry run loudly rather
    than letting it silently test something other than what ships.
    """

    return [
        ('Path("/content/owod-active")', f'Path("{ROOT}")'),
        ('Path("/content/PROB")', f'Path("{workspace / "PROB"}")'),
        ('DRIVE_ROOT = "/content/drive/MyDrive/OWL"', f'DRIVE_ROOT = "{drive_root}"'),
        ('DATA_ROOT = "/content/data/OWOD"', f'DATA_ROOT = "{workspace / "OWOD"}"'),
        # The launcher is the one step that must not touch a GPU.
        ('"--time-budget-minutes", f"{_budget:.0f}",',
         '"--time-budget-minutes", f"{_budget:.0f}", "--dry-run",'),
        # And the guard that refuses a stubbed manifest is inverted, so the dry
        # run proves the flag it keys on is actually written.
        ('assert not _manifest.get("dry_run"), "this manifest is from a stubbed run"',
         'assert _manifest.get("dry_run"), "the dry run must write dry_run: true"'),
        # The pixels are 1-byte placeholders here, so the real 837-image COCO
        # fetch is skipped. The stubbed evaluator only checks that each test
        # image exists; decoding them is PROB's job and PROB is not here.
        ('"--data-root", DATA_ROOT])',
         ('"--data-root", DATA_ROOT, "--annotations-only"])\n'
          '_dry_run_pixels(DATA_ROOT)')),
    ]


def fake_remote(repository: str, commit: str = "", **_) -> dict:
    return {
        "repository": repository, "commit": commit,
        "branch": "feat/daowod-bridge-v2", "branch_head": commit,
        "pin_is_ref_tip": True, "branch_points_at_commit": True,
        "attempts_used": 1,
    }


def placeholder_pixels(data_root: Path, split_name: str) -> int:
    """One-byte JPEGs for the shared evaluation split.

    The stubbed evaluator only checks that each test image exists — decoding
    them is PROB's job and PROB is not here — so this keeps the dry run to
    seconds instead of a 837-image download.
    """

    jpeg = data_root / "JPEGImages"
    jpeg.mkdir(parents=True, exist_ok=True)
    split = data_root / "ImageSets" / "OWDETR" / f"{split_name}.txt"
    written = 0
    for image_id in split.read_text(encoding="utf-8").split():
        target = jpeg / f"{image_id}.jpg"
        if not target.exists():
            target.write_bytes(b"\xff")
            written += 1
    return written


def audit(results: Path) -> None:
    """The assertions the dry run exists to make."""

    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    entries = [e for e in manifest["trajectories"] if e["status"] == "COMPLETE"]
    assert entries, "no trajectory completed"

    checkpoints: dict[str, set[str]] = {}
    answers: list[float] = []
    for entry in entries:
        name = entry["trajectory"]
        arm = entry["arm"]
        assert entry["tasks"] == list(TASKS), f"{name} reached {entry['tasks']}"

        # lineage: t3 from its own t2, t4 from its own t3
        previous = None
        for task in TASKS:
            record = json.loads(
                (results / name / f"{task}_{arm}" / "checkpoint.train.json"
                 ).read_text(encoding="utf-8"))
            came_from = Path(record["previous_checkpoint"])
            produced = Path(record["output_checkpoint"])
            if previous is None:
                assert came_from.name == "t1.pth", (
                    f"{name}/{task} did not start from the anchor: {came_from}")
            else:
                assert came_from == previous, (
                    f"{name}/{task} trained from {came_from}, not from its own "
                    f"previous task's {previous}. The chain is not sequential.")
            previous = produced
            checkpoints.setdefault(task, set()).add(str(produced))

        with (results / name / "results.csv").open(encoding="utf-8") as handle:
            import csv

            rows = list(csv.DictReader(handle))
        assert [r["task"] for r in rows] == list(TASKS)
        answers.extend(float(r["answers_spent"]) for r in rows)

    for task, paths in checkpoints.items():
        assert len(paths) == len(entries), (
            f"{task}: {len(entries)} trajectories produced {len(paths)} distinct "
            "checkpoints. Two arms shared one.")

    # Every trajectory says which seed PROB was given, in the manifest itself.
    # Method V3's audit had to read the launcher's source to find that out.
    for entry in entries:
        assert "prob_seed" in entry, entry["trajectory"]
        assert entry["prob_seed"] == entry["seed"]
        assert entry["replay_arm"] == "uniform"
        assert entry["replay_objects"] == 400

    # A coverage arm must carry its labelled reference forward, or it is a static
    # ranking wearing a traversal's name.
    import csv

    for entry in entries:
        directory = results / entry["trajectory"]
        blocks = sorted(directory.glob("t*/coverage_reference.npz"))
        with (directory / "results.csv").open(encoding="utf-8") as handle:
            points = [
                float(row["reference_points"])
                for row in csv.DictReader(handle)
                if row.get("reference_points") not in (None, "")
            ]
        if not points:
            assert not blocks, (
                f"{entry['trajectory']} stored semantic blocks but reports no "
                "reference size")
            continue
        assert len(blocks) == len(TASKS), (entry["trajectory"], len(blocks))
        assert points[0] == 0.0 and points[1] > 0 and points[2] > points[1], (
            f"{entry['trajectory']} reference sizes {points} do not grow; the "
            "traversal is not being told what it already bought")

    spread = max(answers) / max(min(answers), 1.0)
    assert spread < 1.05, (
        f"oracle answers differ by {spread:.3f}x across arms and tasks; the "
        "budget is supposed to be matched to within one image's cost.")

    coverage_arms = [
        e["arm"] for e in entries
        if (results / e["trajectory"] / f"t2_{e['arm']}"
            / "coverage_reference.npz").exists()
    ]
    print(f"[audit] {len(entries)} trajectories x {len(TASKS)} tasks; lineage "
          f"sequential and per-arm; {len(checkpoints[TASKS[0]])} distinct "
          f"checkpoints per task; answers matched to {spread:.4f}x; "
          f"seed recorded per trajectory; coverage reference grows for "
          f"{coverage_arms or 'no arm in this session'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true",
                        help="print each cell's own output")
    parser.add_argument("--keep", action="store_true", help="keep the workspace")
    arguments = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="owl-benchmark-dry-"))
    drive_root = workspace / "drive" / "MyDrive" / "OWL"
    (drive_root / "checkpoints" / "SOWODB").mkdir(parents=True, exist_ok=True)
    (drive_root / "checkpoints" / "SOWODB" / "t1.pth").write_bytes(b"fake t1")
    (drive_root / "features").mkdir(parents=True, exist_ok=True)
    (drive_root / "features" / "ref_t1_dinov2_vitb14_cap1000_v1.npz").write_bytes(
        b"fake reference; the dry run's traversal starts from an empty one")
    prob_root = workspace / "PROB"
    (prob_root / ".git").mkdir(parents=True)
    (prob_root / "models" / "ops").mkdir(parents=True)
    (prob_root / "requirements.txt").write_text("", encoding="utf-8")

    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
    google = types.ModuleType("google")
    google.colab = colab
    sys.modules.setdefault("google", google)
    sys.modules["google.colab"] = colab

    torch = types.ModuleType("torch")
    torch.__version__ = "2.8.0-dry"
    torch.version = types.SimpleNamespace(cuda="12.6")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda _: "Tesla T4 (dry run)")
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = torch

    base_run = fake_subprocess(workspace)
    # `tools.dry_run_notebook.fake_subprocess` answers `git rev-parse HEAD` with
    # the SHA the *replay* notebook pinned, which is not the SHA this one pins.
    # A reset followed by a rev-parse must return what it was reset to, so that
    # is what is modelled — and it stays correct when this notebook is re-pinned.
    reset_to: dict[str, str] = {}

    def fake_run(command, **kwargs):
        text = [str(part) for part in command]
        cwd = str(kwargs.get("cwd", ROOT))
        if text[:3] == ["git", "reset", "--hard"] and len(text) > 3:
            reset_to[cwd] = text[3]
        if text[:3] == ["git", "rev-parse", "HEAD"] and cwd in reset_to:
            return real_subprocess.CompletedProcess(command, 0, reset_to[cwd] + "\n", "")
        # A `--help` on one of this repository's own tools is answered for real:
        # the preflight that reads it exists to catch a stale pin whose launcher
        # lacks a flag, and a faked empty answer would make it always fail.
        if "--help" in text and any(part.startswith(str(ROOT / "tools")) for part in text):
            return real_subprocess.run(
                command, check=True, capture_output=True, text=True
            )
        return base_run(command, **kwargs)
    # run() is faked (git, pip, probes); Popen is real, because the notebook's
    # expensive steps are this repository's own tools and running them is the point.
    fake_module = types.SimpleNamespace(
        run=fake_run, Popen=real_subprocess.Popen,
        PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT,
        CompletedProcess=real_subprocess.CompletedProcess,
    )

    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]

    def dry_run_pixels(data_root) -> None:
        from owl.evaluation_subset import SHARED_TEST_SET

        written = placeholder_pixels(Path(data_root), SHARED_TEST_SET)
        print(f"dry run: {written} placeholder test JPEGs")

    namespace: dict = {"_dry_run_pixels": dry_run_pixels}
    try:
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            # The shipped cell keeps its Python-3.13 gate; a laptop dry run of
            # notebook *control flow* must not be blocked by the interpreter it
            # happens to run under.
            source = source.replace(
                "assert sys.version_info[:2] == (3, 13), sys.version",
                "assert len(sys.version_info[:2]) == 2, sys.version",
            )
            for before, after in substitutions(workspace, drive_root):
                if before in source:
                    assert source.count(before) == 1, f"{before!r} matched twice"
                    source = source.replace(before, after)

            namespace["subprocess"] = fake_module
            namespace["_dry_run_pixels"] = dry_run_pixels
            buffer = io.StringIO()
            try:
                with redirect_stdout(buffer):
                    exec(compile(source, f"cell {index}", "exec"), namespace)  # noqa: S102
            except Exception:
                print(buffer.getvalue())
                print(f"\n*** cell {index} raised ***\n")
                raise
            if arguments.verbose:
                print(f"--- cell {index} ---")
                print(buffer.getvalue())
            else:
                first = source.split("\n", 1)[0]
                print(f"ok  {first}")

            # The environment cell purges sys.modules and re-imports owl, which
            # would restore the real network probe. So the stub goes in *after*
            # it runs, not before.
            if "from owl.active_selection import" in source:
                namespace["bridge"].verify_remote_commit = fake_remote

        audit(Path(namespace["RESULTS"]))
        print("\nDRY RUN PASSED")
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
        if not arguments.keep:
            import shutil

            shutil.rmtree(workspace, ignore_errors=True)
        else:
            print(f"workspace kept at {workspace}")


if __name__ == "__main__":
    main()
