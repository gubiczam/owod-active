#!/usr/bin/env python
"""Execute every cell of the V2 notebook, with PROB, Colab and DINOv2 faked.

The V1 dry run (``tools/dry_run_full_owod_benchmark.py``) exists because two
bugs reached a real GPU session and neither was findable by unit-testing ``owl``:
every part was correct on its own and the *notebook* was wrong. V2 adds four
things that only a notebook run exercises end to end — a research score with a
DBSCAN gate, a per-box annotation policy that writes derived annotations, a
policy-dependent cost function, and a round schedule that actually reaches the
arm — so it gets its own.

It reuses the V1 harness's stubs (``google.colab``, ``torch``, the faked
``subprocess.run``) and differs in what it substitutes and what it asserts.

    python tools/dry_run_owod_full_chain_v2.py

What it proves, and asserts:

* every cell runs in order on a clean namespace — the V1 run's own lesson;
* the pre-registration check passes for the shipped parameter cell;
* each arm walks the whole chain, each task from its **own** previous checkpoint;
* the per-box policy reaches the detector: alias annotations are written, PROB is
  handed alias ids, and the supervision ledger reconciles
  (``supervised + ignored + banked == labelled``);
* the DBSCAN gate fires on some candidates and not all of them;
* ``D_batch`` moves during a round, and ``D`` is the mean of its two halves;
* the acquisition rounds recompute against a labelled pool that grows;
* the oracle answers are matched across arms.

The DINOv2 features are seeded noise, so **no selection here is a result**. The
manifest says ``dry_run: true`` and the audit refuses a manifest that does not.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess as real_subprocess
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NOTEBOOK = ROOT / "notebooks" / "OWOD_FULL_CHAIN_V2.ipynb"

from tools.dry_run_full_owod_benchmark import (
    fake_remote,
    placeholder_pixels,
)
from tools.dry_run_notebook import fake_subprocess

#: A two-task chain and one arm pair. The notebook ships Phase A (t1 -> t5,
#: three arms); a dry run has to prove the control flow, not the protocol's
#: length, and every extra task is another stubbed evaluation.
DRY_TASKS = ("t2",)
DRY_ARMS = ("random", "research_v2")


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
        # The pin is a placeholder in the shipped notebook, on purpose: a real
        # run must paste the SHA of the commit that carries the V2 code, and the
        # parameter cell refuses to continue without it. A dry run is not a real
        # run, so it substitutes a syntactically valid SHA and the faked git
        # answers it.
        ('OWL_COMMIT = "REPLACE_WITH_THE_40_CHAR_SHA_OF_THE_V2_COMMIT"',
         'OWL_COMMIT = "0" * 40'),
        # Two tasks, two arms. Named here rather than in the notebook, so what
        # ships is the protocol's configuration.
        ('TASK_END = 5', f'TASK_END = {1 + len(DRY_TASKS)}'),
        ('SESSION_ARMS = ("random", "entropy", "research_v2")',
         f'SESSION_ARMS = {DRY_ARMS!r}'),
        # The budget and the batch are NOT substituted. They are frozen values
        # and cell [4/12] compares them against the protocol document, so
        # shrinking them here would either fail that check or require faking it
        # — and 300 answers in 3 rounds of 100 is what the protocol runs, which
        # is what a dry run should exercise.
        # The launcher is the one step that must not touch a GPU.
        ('"--time-budget-minutes", f"{_budget:.0f}",',
         '"--time-budget-minutes", f"{_budget:.0f}", "--dry-run",'),
        # And the guard that refuses a stubbed manifest is inverted, so the dry
        # run proves the flag it keys on is actually written.
        ('assert not _manifest.get("dry_run"), "this manifest is from a stubbed run"',
         'assert _manifest.get("dry_run"), "the dry run must write dry_run: true"'),
        # Placeholder pixels instead of the real COCO fetch. The stubbed
        # evaluator only checks that each test image exists; decoding them is
        # PROB's job and PROB is not here. The per-box policy DOES open the
        # JPEGs it suppresses, so those are written as real images below.
        ('"--data-root", DATA_ROOT, "--n-tasks", str(N_TASKS)])',
         ('"--data-root", DATA_ROOT, "--n-tasks", str(N_TASKS),\n'
          '           "--annotations-only"])\n'
          '_dry_run_pixels(DATA_ROOT)')),
    ]


def real_pixels(data_root: Path) -> int:
    """Decodable JPEGs for every annotated image, as hard links to one file.

    ``pixel_suppression`` opens the source JPEG and paints over a box, so a
    one-byte placeholder is not enough for the images the policy touches — and
    which images it touches is exactly what the selector decides, so all 28,800
    need pixels. Encoding 28,800 files would dominate the dry run's wall clock,
    so one 32x32 image is encoded and hard-linked; the suppression writer never
    modifies a source in place (it writes a new alias), so sharing an inode is
    safe. Only Pillow ever decodes these: PROB is stubbed.
    """

    from PIL import Image

    jpeg = data_root / "JPEGImages"
    jpeg.mkdir(parents=True, exist_ok=True)
    template = jpeg / "_dry_run_template.jpg"
    Image.new("RGB", (32, 32), (90, 120, 150)).save(template, format="JPEG")

    written = 0
    for annotation in (data_root / "Annotations").glob("*.xml"):
        target = jpeg / f"{annotation.stem}.jpg"
        if target.exists() and target.stat().st_size > 4:
            continue
        if target.exists():
            target.unlink()
        try:
            target.hardlink_to(template)
        except OSError:
            import shutil

            shutil.copyfile(template, target)
        written += 1
    return written


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def audit(results: Path) -> None:
    """The assertions this dry run exists to make."""

    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    entries = [e for e in manifest["trajectories"] if e["status"] == "COMPLETE"]
    assert entries, "no trajectory completed"

    configuration = json.loads(
        (results / "v2_configuration.json").read_text(encoding="utf-8"))
    assert configuration["annotation_policy"] == "known_plus_selected_ignore_rest"
    assert configuration["ignore_mechanism"] == "pixel_suppression"

    answers: list[float] = []
    checkpoints: dict[str, set[str]] = {}
    saw_research_arm = False

    for entry in entries:
        name, arm = entry["trajectory"], entry["arm"]
        assert entry["tasks"] == list(DRY_TASKS), f"{name} reached {entry['tasks']}"

        # lineage: every task starts from its own arm's previous checkpoint
        previous = None
        for task in DRY_TASKS:
            record = json.loads(
                (results / name / f"{task}_{arm}" / "checkpoint.train.json"
                 ).read_text(encoding="utf-8"))
            came_from = Path(record["previous_checkpoint"])
            if previous is None:
                assert came_from.name == "t1.pth", (
                    f"{name}/{task} did not start from the anchor: {came_from}")
            else:
                assert came_from == previous, (
                    f"{name}/{task} trained from {came_from}, not from its own "
                    "previous task's checkpoint. The chain is not sequential.")
            previous = Path(record["output_checkpoint"])
            checkpoints.setdefault(task, set()).add(str(previous))

        table = rows(results / name / "results.csv")
        assert [r["task"] for r in table] == list(DRY_TASKS)
        answers.extend(number(r["answers_spent"]) for r in table)

        # --- the per-box annotation policy actually reached the detector ------
        for row in table:
            labelled = number(row.get("objects_labelled"))
            assert labelled > 0, (
                f"{name}/{row['task']} recorded no annotated objects, so the "
                "per-box policy did not run at all")
            parts = (
                number(row.get("objects_supervised"))
                + number(row.get("objects_ignored"))
                + number(row.get("objects_banked"))
            )
            assert parts == labelled, (
                f"{name}/{row['task']} ledger does not reconcile: "
                f"supervised+ignored+banked={parts} against labelled={labelled}")
            assert number(row.get("objects_ignored")) > 0, (
                f"{name}/{row['task']} ignored nothing, so "
                "known_plus_selected_ignore_rest is indistinguishable from "
                "full_image here and the dry run proves nothing")
            assert row.get("label_supervision_from", "").startswith("per-box"), (
                f"{name}/{row['task']} says supervision came from "
                f"{row.get('label_supervision_from')!r}; the policy did not "
                "take over")
            assert number(row.get("training_images")) > 0

        # --- the research score's own trail -----------------------------------
        if arm not in ("random", "entropy", "admissibility"):
            saw_research_arm = True
            log = results / name / f"{DRY_TASKS[0]}_{arm}" / "candidate_log.csv"
            assert log.is_file(), f"{name} left no candidate_log.csv"
            picks = rows(log)
            assert picks, "the candidate log is empty"

            statuses = {row["cluster_status"] for row in picks}
            assert statuses <= {"core", "border", "noise"}, statuses
            assert {row["coh"] for row in picks} <= {"0.0", "1.0"}, (
                "coh must be binary under dbscan_binary")
            # A gate that fires on nobody, or on everybody, is not a gate. On a
            # stubbed noise feature matrix it may legitimately accept every
            # *taken* candidate, so the population-level rate is what is checked.
            gate_rate = number(
                [r for r in table][0].get("final_noise_rate"), default=-1.0)
            assert 0.0 < gate_rate < 1.0, (
                f"{name} reports final_noise_rate={gate_rate}; the DBSCAN gate "
                "either rejected nothing or rejected everything")

            batch = [number(row["D_batch"]) for row in picks]
            assert batch[0] == 1.0, "nothing was taken yet, so nothing was redundant"
            assert min(batch) < 1.0, (
                "D_batch never moved during a round, so intra-batch diversity "
                "is not being recomputed")
            for row in picks:
                assert abs(
                    number(row["D"])
                    - 0.5 * (number(row["D_labeled"]) + number(row["D_batch"]))
                ) < 1e-5, row

            # the rounds recompute against a pool that grows
            schedule = json.loads(table[0]["round_log"])
            assert len(schedule) > 1, (
                f"{name} ran {len(schedule)} round; the schedule did not reach "
                "the arm, which is the defect --acquisition-batch-size exists "
                "to avoid")
            reference = [r["reference_rows"] for r in schedule if r.get("images")]
            assert reference == sorted(reference), reference

    assert saw_research_arm, (
        f"no research arm ran; {DRY_ARMS} must include one or this dry run "
        "exercises none of the V2 score")

    for task, paths in checkpoints.items():
        assert len(paths) == len(entries), (
            f"{task}: {len(entries)} trajectories produced {len(paths)} distinct "
            "checkpoints. Two arms shared one.")

    spread = max(answers) / max(min(answers), 1.0)
    assert spread < 1.15, (
        f"oracle answers differ by {spread:.3f}x across arms and tasks; the "
        "budget is supposed to be matched to within one image's cost.")

    print(f"[audit] {len(entries)} trajectories x {len(DRY_TASKS)} task(s); "
          f"lineage sequential and per-arm; answers matched to {spread:.4f}x; "
          "per-box supervision reconciles; DBSCAN gate fires; D_batch moves; "
          "rounds recompute against a growing reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true",
                        help="print each cell's own output")
    parser.add_argument("--keep", action="store_true", help="keep the workspace")
    arguments = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="owl-v2-dry-"))
    drive_root = workspace / "drive" / "MyDrive" / "OWL"
    (drive_root / "checkpoints" / "SOWODB").mkdir(parents=True, exist_ok=True)
    (drive_root / "checkpoints" / "SOWODB" / "t1.pth").write_bytes(b"fake t1")
    (drive_root / "features").mkdir(parents=True, exist_ok=True)
    (drive_root / "features" / "ref_t1_dinov2_vitb14_cap1000_v1.npz").write_bytes(
        b"fake reference; the dry run's reference starts empty")
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
            return real_subprocess.run(command, check=True, capture_output=True,
                                       text=True)
        return base_run(command, **kwargs)

    fake_module = types.SimpleNamespace(
        run=fake_run, Popen=real_subprocess.Popen,
        PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT,
        CompletedProcess=real_subprocess.CompletedProcess,
    )

    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]

    def dry_run_pixels(data_root) -> None:
        from owl.evaluation_subset import shared_test_set_name

        data_root = Path(data_root)
        placeholders = placeholder_pixels(
            data_root, shared_test_set_name(1 + len(DRY_TASKS)))
        # The per-box policy opens and rewrites the JPEGs it suppresses, so
        # every annotated image needs pixels Pillow can actually decode.
        decodable = real_pixels(data_root)
        print(f"dry run: {placeholders} placeholder test JPEGs, "
              f"{decodable} decodable annotated JPEGs")

    namespace: dict = {"_dry_run_pixels": dry_run_pixels}
    # A substitution that matches nothing is the failure mode that hurts: the
    # cell runs unmodified and the dry run quietly tests something other than
    # what it claims to. The docstring promised "exactly once"; this enforces
    # the other half of it.
    matched: set[str] = set()
    try:
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            source = source.replace(
                "assert sys.version_info[:2] == (3, 13), sys.version",
                "assert len(sys.version_info[:2]) == 2, sys.version",
            )
            for before, after in substitutions(workspace, drive_root):
                if before in source:
                    assert source.count(before) == 1, f"{before!r} matched twice"
                    source = source.replace(before, after)
                    matched.add(before)

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
                print("ok ", source.split("\n", 1)[0])

            # The environment cell purges sys.modules and re-imports owl, which
            # would restore the real network probe. So the stub goes in *after*
            # it runs, not before.
            if "from owl.active_selection import" in source:
                namespace["bridge"].verify_remote_commit = fake_remote

        expected = {before for before, _ in substitutions(workspace, drive_root)}
        assert matched == expected, (
            "these substitutions matched no cell, so the notebook ran "
            f"unmodified where the dry run assumed otherwise: {expected - matched}")
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
