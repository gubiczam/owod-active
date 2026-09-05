"""The evaluation-only re-scoring driver.

What these tests are for, in order of what they would cost to get wrong:

* **the split name.** PROB routes a split by substring and ``eval`` contains
  ``val``, so a split called ``large_eval`` goes to the ``val`` branch where no
  annotation filtering runs — U-Recall reads zero everywhere and future-task
  objects are scored as already known. A full table of plausible, wrong numbers.
* **leakage.** Nothing evaluated on may ever have been trainable.
* **the frozen endpoint.** This is a *separate* evaluation; it must be unable to
  write where the benchmark's own results live.
* **the counts**, because the whole design rests on them: a larger split repairs
  ``bear`` and the open-world metrics and does nothing whatever for
  ``fire hydrant`` and ``stop sign``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from owl.active_selection import benchmark as bm
from owl.evaluation_subset import SplitNameError, check_split_name
from tools import run_large_eval as driver

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------ the split name ---


def test_the_split_name_carries_only_the_test_marker():
    assert check_split_name(driver.LARGE_TEST_SET, purpose="test")


@pytest.mark.parametrize("name", ["large_eval", "owl_large_eval", "owl_eval_test"])
def test_a_name_containing_eval_is_refused(name):
    """`eval` contains `val`; PROB would skip annotation filtering entirely."""

    with pytest.raises(SplitNameError, match="val"):
        check_split_name(name, purpose="test")


# ------------------------------------------------------------------ leakage ---


def test_the_leakage_check_passes_on_the_real_test_split():
    image_ids, _ = driver.build_split("full")
    assert driver.leakage_check(image_ids) == {
        "with_candidate_pool": 0, "with_replay_pool": 0}


def test_the_leakage_check_refuses_a_trainable_image():
    candidate = json.loads(driver.CANDIDATE_INDEX.read_text(encoding="utf-8"))
    intruder = next(iter(candidate))
    with pytest.raises(SystemExit, match="LEAKAGE"):
        driver.leakage_check(["000000000001", intruder])


# ------------------------------------------------------------- the two scopes ---


@pytest.fixture(scope="module")
def scopes():
    full, per = driver.build_split("full")
    declared, _ = driver.build_split("declared")
    return full, declared, per


def test_the_scopes_are_nested_and_sized_as_measured(scopes):
    full, declared, _ = scopes
    assert len(full) == 4_952
    assert len(declared) == 3_864
    assert set(declared) <= set(full)


def test_a_larger_split_cannot_help_the_two_tail_classes(scopes):
    """The design's central measured fact, and the reason it is not oversold.

    The frozen 837-image split already holds *every* test image with a fire
    hydrant or a stop sign, because the 150-per-class cap never binds for them.
    """

    from owl import evaluation_subset

    full, declared, per = scopes
    current = set(evaluation_subset.from_archive(
        driver.TEST_ARCHIVE, bm.declared_classes(), seed=bm.DEVELOPMENT_SEED,
        remainder_multiplier=bm.EVAL_REMAINDER_RATIO,
        max_per_class=bm.EVAL_MAX_PER_CLASS).image_ids)

    def objects(split, name):
        return sum(per[i].get(name, 0) for i in split)

    for name, count in (("fire hydrant", 101), ("stop sign", 75)):
        assert objects(current, name) == count
        assert objects(declared, name) == count
        assert objects(full, name) == count, f"{name} gained support it cannot have"

    # what it does repair
    assert objects(current, "bear") == 2
    assert objects(full, "bear") == 71
    assert objects(full, "traffic light") == 637
    assert objects(current, "traffic light") == 534


# --------------------------------------------------------------- checkpoints ---


def _tree(root: Path, *, prune_t2: bool = True) -> Path:
    trajectories = []
    for arm in ("random", "admissibility", "proposed", "entropy", "proposed_v2"):
        name = f"{arm}__seed0"
        for task in ("t2", "t3", "t4"):
            directory = root / name / f"{task}_{arm}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "metrics.json").write_text("{}", encoding="utf-8")
            if task != "t2" or not prune_t2:
                (directory / "checkpoint.pth").write_bytes(b"fake")
        trajectories.append({"trajectory": name, "arm": arm, "seed": 0,
                             "status": "COMPLETE"})
    bm.write_json(root / "manifest.json", {"trajectories": trajectories})
    return root


def test_the_pruned_t2_checkpoints_are_reported_not_hidden(tmp_path):
    root = _tree(tmp_path / "frozen")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    found = driver.surviving_checkpoints(root, manifest)
    assert len(found) == 15
    assert sum(1 for c in found if c["present"]) == 10
    assert {c["task"] for c in found if not c["present"]} == {"t2"}


def test_every_task_is_examined_by_default():
    """A default that omitted t2 would hide that its checkpoint is gone."""

    assert driver.ALL_TASKS == ("t2", "t3", "t4")


def test_an_incomplete_trajectory_is_skipped(tmp_path):
    root = _tree(tmp_path / "frozen")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["trajectories"][0]["status"] = "FAILED"
    found = driver.surviving_checkpoints(root, manifest)
    assert {c["arm"] for c in found} == {
        "admissibility", "proposed", "entropy", "proposed_v2"}


# ------------------------------------------------- the frozen endpoint is safe ---


def test_writing_into_the_frozen_results_is_refused(tmp_path):
    root = _tree(tmp_path / "frozen")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_large_eval.py"),
         "--prob-root", str(tmp_path), "--data-root", str(tmp_path),
         "--results", str(root), "--out", str(root)],
        capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "must differ from --results" in result.stdout + result.stderr


def test_the_driver_never_trains():
    source = (ROOT / "tools" / "run_large_eval.py").read_text(encoding="utf-8")
    assert ".train(" not in source
    assert "bridge.evaluate(" in source
