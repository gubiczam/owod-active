"""Scientific-contract tests for the controlled S-OWODB long-tail protocol."""

from __future__ import annotations

import copy
import importlib.util
import json
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest

from owl import longtail, protocol, runner

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def prepare_tool():
    spec = importlib.util.spec_from_file_location(
        "prepare_longtail_no_replay_under_test",
        ROOT / "tools" / "prepare_longtail_no_replay.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source_index():
    return longtail.read_source_index(
        protocol.GROUPS_PATH.parent / "t1_replay_class_counts.json")


@pytest.fixture(scope="module")
def source_counts(source_index):
    return longtail.object_counts(source_index)


def test_canonical_t1_source_identity_and_ranking_are_exact(source_index, source_counts):
    assert len(source_index) == 89_490
    assert sum(source_counts.values()) == 421_243
    ranking = longtail.class_ranking(source_counts)
    assert ranking[:4] == ("person", "car", "bird", "boat")
    assert ranking[-4:] == ("giraffe", "cat", "train", "bear")
    assert source_counts[ranking[0]] / source_counts[ranking[-1]] == pytest.approx(
        202.8323029366306)


def test_rank_thirds_are_fixed_and_meaningful(source_counts):
    ranking = longtail.class_ranking(source_counts)
    groups = longtail.rank_groups(ranking)
    sizes = Counter(groups.values())
    assert sizes == {"head": 7, "medium": 6, "tail": 6}
    assert all(groups[name] == "head" for name in ranking[:7])
    assert all(groups[name] == "medium" for name in ranking[7:13])
    assert all(groups[name] == "tail" for name in ranking[13:])
    assert protocol.load_groups()["bear"] == "tail"  # historical semantics remain separate


def test_fixed_total_exponential_schedules_hit_the_preregistered_severities(source_counts):
    total = longtail.matched_controlled_total(source_counts)
    assert total == 79_233
    expected_edges = {
        "lt10": (10_432, 1_043, 10.001917545541707),
        "lt50": (15_730, 315, 49.93650793650794),
        "lt100": (18_025, 180, 100.13888888888889),
    }
    previous = None
    for condition, (largest, smallest, ratio) in expected_edges.items():
        targets = longtail.condition_targets(source_counts, condition)
        values = list(targets.values())
        assert sum(values) == total
        assert values[0] == largest and values[-1] == smallest
        assert longtail.achieved_rho(targets) == pytest.approx(ratio)
        assert all(left >= right > 0 for left, right in zip(values, values[1:]))
        assert all(targets[name] <= source_counts[name] for name in targets)
        if previous is not None:
            assert targets != previous
        previous = targets


def test_controlled_conditions_keep_every_class_and_are_distinct(source_counts):
    conditions = {
        condition: longtail.condition_targets(source_counts, condition)
        for condition in longtail.CONDITIONS
    }
    assert len({tuple(targets.items()) for targets in conditions.values()}) == 4
    assert all(
        set(targets) == set(longtail.CONTROLLED_CLASSES)
        and all(value > 0 for value in targets.values())
        for targets in conditions.values()
    )


def test_original_is_the_complete_untouched_object_population(source_counts):
    original = longtail.condition_targets(source_counts, "original")
    assert original == {
        name: source_counts[name] for name in longtail.class_ranking(source_counts)
    }
    assert sum(original.values()) == 421_243
    assert longtail.achieved_rho(original) == pytest.approx(202.8323029366306)


def small_multi_object_index():
    index = {}
    for rank, name in enumerate(longtail.CONTROLLED_CLASSES):
        image_id = f"{rank + 1:012d}"
        index[image_id] = {name: 4}
    # Two classes share one detector image; selection must control boxes, not
    # mistakenly count the whole image for both classes.
    index["999999999999"] = {
        longtail.CONTROLLED_CLASSES[0]: 3,
        longtail.CONTROLLED_CLASSES[1]: 2,
    }
    return index


def test_object_selection_is_seeded_exact_unique_and_multi_object_safe():
    index = small_multi_object_index()
    targets = {name: 2 for name in longtail.CONTROLLED_CLASSES}
    first = longtail.select_objects(index, targets, seed=0)
    again = longtail.select_objects(copy.deepcopy(index), targets, seed=0)
    other = longtail.select_objects(index, targets, seed=1)
    assert first == again and first != other
    assert longtail.selection_counts(first) == targets
    identities = [
        (image, name, ordinal)
        for image, values in first.items()
        for name, ordinals in values.items()
        for ordinal in ordinals
    ]
    assert len(identities) == len(set(identities)) == 38
    longtail.verify_selection(index, first, targets)


def test_deterministic_selection_ledger_is_path_independent(tmp_path):
    index = small_multi_object_index()
    targets = {name: 2 for name in longtail.CONTROLLED_CLASSES}
    selection = longtail.select_objects(index, targets, seed=0)
    payload = longtail.selection_payload("lt10", selection)
    one = longtail.write_gzip_json(tmp_path / "one.json.gz", payload)
    two = longtail.write_gzip_json(tmp_path / "different-name.json.gz", payload)
    assert one.read_bytes() == two.read_bytes()
    assert longtail.read_gzip_json(one) == payload


def test_manifest_hash_is_deterministic_and_covers_source_selection_and_groups(source_counts):
    targets = longtail.condition_targets(source_counts, "lt10")
    arguments = {
        "condition": "lt10",
        "source_counts": source_counts,
        "targets": targets,
        "selected_images": 12_345,
        "seed": 0,
        "source_index_path": "source.json",
        "source_index_sha256": "1" * 64,
        "source_annotations_path": "source.tar.gz",
        "source_annotations_sha256": "2" * 64,
        "selection_path": "selection.json.gz",
        "selection_sha256": "3" * 64,
        "test_split_sha256": "4" * 64,
        "controlled_total": 79_233,
    }
    first = longtail.build_manifest(**arguments)
    second = longtail.build_manifest(**copy.deepcopy(arguments))
    assert first == second
    longtail.verify_manifest(first)
    damaged = copy.deepcopy(first)
    damaged["classes"][0]["achieved_count"] -= 1
    with pytest.raises(longtail.LongTailError, match="hash mismatch"):
        longtail.verify_manifest(damaged)


def test_source_scientific_content_is_never_mutated(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(small_multi_object_index(), sort_keys=True), encoding="utf-8")
    before = longtail.sha256_file(source)
    index = longtail.read_source_index(source)
    targets = {name: 1 for name in longtail.CONTROLLED_CLASSES}
    longtail.select_objects(index, targets, seed=0)
    assert longtail.sha256_file(source) == before


def test_release_manifests_and_selections_validate_without_source_or_test_mutation(
    prepare_tool,
):
    protected = [
        ROOT / "data" / "reference" / "t1_replay_class_counts.json",
        ROOT / "data" / "staging" / "owdetr_replay_annotations.tar.gz",
        ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz",
    ]
    before = {path: longtail.sha256_file(path) for path in protected}
    manifests = prepare_tool.validate_manifests(ROOT / "data" / "reference" / "longtail")
    assert tuple(manifests) == longtail.CONDITIONS
    assert {path: longtail.sha256_file(path) for path in protected} == before


def test_longtail_fingerprint_isolated_and_historical_fingerprint_unchanged():
    historical = runner.CycleConfig().fingerprint()
    assert not any("longtail" in name for name in historical)
    config = longtail.LongTailCycleConfig(
        longtail_condition="lt50",
        longtail_manifest_sha256="1" * 64,
        longtail_source_sha256="2" * 64,
        longtail_anchor_sha256="3" * 64,
        longtail_owl_commit="4" * 40,
        longtail_prob_commit="5" * 40,
    )
    fingerprint = config.fingerprint()
    assert runner.CycleConfig().fingerprint() == historical
    assert fingerprint["controlled_longtail_protocol_version"] == 1
    assert fingerprint["longtail_condition"] == "lt50"
    assert fingerprint["longtail_manifest_sha256"] == "1" * 64
    assert longtail.fingerprint_sha256(config) == longtail.sha256_bytes(
        longtail.canonical_json_bytes(fingerprint))


def test_dry_run_workspace_names_cannot_collide_with_completed_history():
    names = {longtail.workspace_name(condition) for condition in longtail.CONDITIONS}
    assert names == {
        "random__none__original", "random__none__lt10",
        "random__none__lt50", "random__none__lt100",
    }
    assert not names & longtail.HISTORICAL_WORKSPACES
    assert longtail.workspace_name("lt10", seed=1) == "random__none__lt10__seed1"


def _prepare_arguments(prepare_tool, tmp_path, *, protocol_only, owl_commit=""):
    return Namespace(
        manifest_root=ROOT / "data" / "reference" / "longtail",
        anchor_root=tmp_path / "anchors",
        work_root=tmp_path / "work",
        owl_commit=owl_commit,
        prob_commit=prepare_tool.PINNED_PROB_COMMIT,
        seed=0,
        protocol_only=protocol_only,
    )


def test_protocol_only_dry_run_is_read_only_and_reports_the_anchor_gate(
    prepare_tool, tmp_path,
):
    arguments = _prepare_arguments(prepare_tool, tmp_path, protocol_only=True)
    historical = arguments.work_root / "random__none"
    historical.mkdir(parents=True)
    sentinel = historical / "completed-result.json"
    sentinel.write_bytes(b"frozen historical result")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*") if path.is_file()
    }
    report = prepare_tool.prepare(arguments)
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*") if path.is_file()
    }
    assert after == before
    assert sentinel.read_bytes() == b"frozen historical result"
    assert report["execution_ready"] is False
    assert len(report["runs"]) == 4
    assert all(row["replay"] == "none" for row in report["runs"])
    assert all(row["selection"] == "random" for row in report["runs"])
    assert all(row["fingerprint"] is None for row in report["runs"])
    with pytest.raises(longtail.LongTailError, match="not provenance-complete"):
        prepare_tool.prepare(_prepare_arguments(prepare_tool, tmp_path, protocol_only=False))


def test_complete_dry_run_computes_distinct_exact_fingerprints_without_writing(
    prepare_tool, tmp_path,
):
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    for condition in longtail.CONDITIONS:
        (anchors / prepare_tool.DEFAULT_ANCHOR_NAMES[condition]).write_bytes(
            f"condition-specific-{condition}".encode()
        )
    arguments = _prepare_arguments(
        prepare_tool, tmp_path, protocol_only=False, owl_commit="a" * 40)
    before = {path: path.read_bytes() for path in anchors.iterdir()}
    report = prepare_tool.prepare(arguments)
    assert report["execution_ready"] is True
    hashes = {row["fingerprint_sha256"] for row in report["runs"]}
    assert len(hashes) == 4 and None not in hashes
    assert all(row["fingerprint"]["replay_arm"] == "none" for row in report["runs"])
    assert all(row["fingerprint"]["arm"] == "random" for row in report["runs"])
    assert all(row["fingerprint"]["n_tasks"] == 6 for row in report["runs"])
    assert {path: path.read_bytes() for path in anchors.iterdir()} == before
    assert not arguments.work_root.exists()


def test_dry_run_refuses_a_controlled_workspace_with_another_fingerprint(
    prepare_tool, tmp_path,
):
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    for condition in longtail.CONDITIONS:
        (anchors / prepare_tool.DEFAULT_ANCHOR_NAMES[condition]).write_bytes(condition.encode())
    workspace = tmp_path / "work" / longtail.workspace_name("lt10")
    workspace.mkdir(parents=True)
    stamp = workspace / "config.json"
    stamp.write_text(json.dumps({"n_tasks": 999}), encoding="utf-8")
    before = stamp.read_bytes()
    with pytest.raises(longtail.LongTailError, match="different fingerprint"):
        prepare_tool.prepare(_prepare_arguments(
            prepare_tool, tmp_path, protocol_only=False, owl_commit="b" * 40))
    assert stamp.read_bytes() == before
