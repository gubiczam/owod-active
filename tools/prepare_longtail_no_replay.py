#!/usr/bin/env python3
"""Fail-closed dry preparation for controlled-long-tail no-replay chains.

This command is deliberately read-only.  It verifies the four versioned
dataset manifests and computes the exact run fingerprints only when the
condition-specific T1 anchors and the reviewed OWL commit are available.  It
does not create workspaces, train anchors, or launch a GPU chain.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import longtail, protocol  # noqa: E402

PINNED_PROB_COMMIT = "4c66be1a52cad9360e09c729e9134aba8fe0b531"
DEFAULT_MANIFEST_ROOT = ROOT / "data" / "reference" / "longtail"
DEFAULT_ANCHOR_NAMES = {
    condition: f"t1_{condition}.pth" for condition in longtail.CONDITIONS
}


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _targets(manifest: dict[str, object]) -> dict[str, int]:
    return {
        str(row["class_name"]): int(row["achieved_count"])
        for row in manifest["classes"]
    }


def _xml_ids(path: Path) -> set[str]:
    with tarfile.open(path) as archive:
        return {
            Path(member.name).stem
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".xml")
        }


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, int, int], ...] | None:
    """Read-only metadata snapshot used to prove this dry-run touched no history."""

    if not root.exists():
        return None
    return tuple(
        (
            str(path.relative_to(root)),
            path.lstat().st_mode,
            path.lstat().st_size,
            path.lstat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
    )


def validate_manifests(manifest_root: Path) -> dict[str, dict[str, object]]:
    """Verify every scientific hash, selection identity, count, and split guard."""

    manifests: dict[str, dict[str, object]] = {}
    common: dict[str, str] | None = None
    source_index: dict[str, dict[str, int]] | None = None
    evaluation_ids: set[str] | None = None
    for condition in longtail.CONDITIONS:
        path = manifest_root / f"{condition}.json"
        if not path.is_file():
            raise longtail.LongTailError(f"Missing controlled manifest: {path}.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        longtail.verify_manifest(manifest)
        if manifest.get("protocol") != longtail.PROTOCOL_NAME:
            raise longtail.LongTailError(f"{path} has the wrong protocol name.")
        if manifest.get("protocol_version") != longtail.PROTOCOL_VERSION:
            raise longtail.LongTailError(f"{path} has the wrong protocol version.")
        if manifest.get("condition") != condition:
            raise longtail.LongTailError(f"{path} claims condition {manifest.get('condition')!r}.")

        source = manifest["source_index"]
        annotations = manifest["source_annotations"]
        selection_ref = manifest["selection"]
        source_path = _repo_path(str(source["path"]))
        annotations_path = _repo_path(str(annotations["path"]))
        selection_path = _repo_path(str(selection_ref["path"]))
        for candidate, recorded in (
            (source_path, source["sha256"]),
            (annotations_path, annotations["sha256"]),
            (selection_path, selection_ref["sha256"]),
        ):
            if not candidate.is_file():
                raise longtail.LongTailError(f"Missing manifest input: {candidate}.")
            actual = longtail.sha256_file(candidate)
            if actual != recorded:
                raise longtail.LongTailError(
                    f"Hash mismatch for {candidate}: {actual} != {recorded}."
                )

        if source_index is None:
            source_index = longtail.read_source_index(source_path)
            evaluation_archive = ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz"
            if not evaluation_archive.is_file():
                raise longtail.LongTailError(
                    f"Missing evaluation annotation archive: {evaluation_archive}."
                )
            evaluation_sha = longtail.sha256_file(evaluation_archive)
            if evaluation_sha != manifest["evaluation_split_sha256"]:
                raise longtail.LongTailError("The evaluation split hash does not match the manifest.")
            evaluation_ids = _xml_ids(evaluation_archive)

        asserted_common = {
            "source_index_sha256": str(source["sha256"]),
            "source_annotations_sha256": str(annotations["sha256"]),
            "evaluation_split_sha256": str(manifest["evaluation_split_sha256"]),
        }
        if common is None:
            common = asserted_common
        elif asserted_common != common:
            raise longtail.LongTailError("Controlled conditions do not share exact source splits.")

        payload = longtail.read_gzip_json(selection_path)
        if not isinstance(payload, dict) or payload.get("condition") != condition:
            raise longtail.LongTailError(f"{selection_path} has the wrong condition.")
        if payload.get("protocol") != longtail.PROTOCOL_NAME:
            raise longtail.LongTailError(f"{selection_path} has the wrong protocol.")
        selection = payload.get("objects")
        if not isinstance(selection, dict):
            raise longtail.LongTailError(f"{selection_path} has no object mapping.")
        targets = _targets(manifest)
        longtail.verify_selection(source_index, selection, targets)
        if len(selection) != int(manifest["selected_images"]):
            raise longtail.LongTailError(f"{condition} selected-image count does not match.")
        if sum(targets.values()) != int(manifest["selected_objects"]):
            raise longtail.LongTailError(f"{condition} selected-object count does not match.")
        ranked = [str(row["class_name"]) for row in manifest["classes"]]
        if tuple(ranked) != longtail.class_ranking(longtail.object_counts(source_index)):
            raise longtail.LongTailError(f"{condition} class ranking is not canonical.")
        values = [targets[name] for name in ranked]
        if any(left < right or right <= 0 for left, right in zip(values, values[1:])):
            raise longtail.LongTailError(f"{condition} is not positive and monotonic.")
        if evaluation_ids is not None and set(selection) & evaluation_ids:
            raise longtail.LongTailError(f"{condition} leaks canonical evaluation images.")
        manifests[condition] = manifest
    return manifests


def frozen_config(
    *, condition: str, manifest: dict[str, object], anchor_sha256: str,
    owl_commit: str, prob_commit: str, seed: int,
) -> longtail.LongTailCycleConfig:
    """The exact seed-0 pilot settings, extended only by LT provenance."""

    return longtail.LongTailCycleConfig(
        n_tasks=6,
        budget_per_task=600,
        rounds_per_task=6,
        candidate_images_per_task=2000,
        proposals_per_image=50,
        arm="random",
        labelling_policy="known_plus_selected",
        replay_arm="none",
        replay_reallocate=False,
        replay_protocol_version=3,
        epochs=5,
        learning_rate=2e-4,
        batch_size=2,
        n_clusters=1600,
        seed=seed,
        measure_grouped_recall=True,
        reuse_deferred_labels=True,
        keep_checkpoints=2,
        longtail_condition=condition,
        longtail_manifest_sha256=str(manifest["scientific_sha256"]),
        longtail_source_sha256=str(manifest["source_annotations"]["sha256"]),
        longtail_anchor_sha256=anchor_sha256,
        longtail_owl_commit=owl_commit,
        longtail_prob_commit=prob_commit,
    )


def prepare(arguments: argparse.Namespace) -> dict[str, object]:
    manifests = validate_manifests(arguments.manifest_root)
    if arguments.prob_commit != PINNED_PROB_COMMIT:
        raise longtail.LongTailError(
            f"PROB must remain pinned to {PINNED_PROB_COMMIT}, got {arguments.prob_commit}."
        )
    if arguments.seed not in (0, 1, 2):
        raise longtail.LongTailError("Only preregistered seeds 0, 1, and 2 may be prepared.")

    work_root = arguments.work_root.resolve()
    historical = {work_root / name for name in longtail.HISTORICAL_WORKSPACES}
    historical_before = {path: _tree_snapshot(path) for path in historical}
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    owl_valid = len(arguments.owl_commit) == 40 and all(
        character in "0123456789abcdef" for character in arguments.owl_commit
    )
    if not owl_valid:
        missing.append("reviewed 40-character OWL scientific commit")

    for condition in longtail.CONDITIONS:
        workspace = work_root / longtail.workspace_name(condition, seed=arguments.seed)
        if workspace in historical:
            raise longtail.LongTailError(f"Workspace collision with completed history: {workspace}.")
        anchor = arguments.anchor_root / DEFAULT_ANCHOR_NAMES[condition]
        anchor_sha = longtail.sha256_file(anchor) if anchor.is_file() else ""
        if not anchor_sha:
            missing.append(f"{condition} T1 anchor: {anchor}")
        row: dict[str, object] = {
            "condition": condition,
            "manifest_scientific_sha256": manifests[condition]["scientific_sha256"],
            "workspace": str(workspace),
            "anchor": str(anchor),
            "anchor_sha256": anchor_sha or None,
            "task_chain": [task.name for task in protocol.build_chain(6)],
            "selection": "random",
            "replay": "none",
            "seed": arguments.seed,
            "fingerprint": None,
            "fingerprint_sha256": None,
        }
        if owl_valid and anchor_sha:
            config = frozen_config(
                condition=condition, manifest=manifests[condition],
                anchor_sha256=anchor_sha, owl_commit=arguments.owl_commit,
                prob_commit=arguments.prob_commit, seed=arguments.seed,
            )
            fingerprint = config.fingerprint()
            row["fingerprint"] = fingerprint
            row["fingerprint_sha256"] = longtail.fingerprint_sha256(config)
            stamp = workspace / "config.json"
            if stamp.exists():
                stored = json.loads(stamp.read_text(encoding="utf-8"))
                differences = {
                    name: (stored.get(name, "(absent)"), value)
                    for name, value in fingerprint.items()
                    if name not in stored or stored[name] != value
                }
                if differences:
                    raise longtail.LongTailError(
                        f"Existing workspace {workspace} has a different fingerprint: "
                        f"{differences}."
                    )
        rows.append(row)

    historical_after = {path: _tree_snapshot(path) for path in historical}
    if historical_after != historical_before:
        raise longtail.LongTailError("Historical workspace configuration changed during dry-run.")

    ready = not missing
    report: dict[str, object] = {
        "protocol": longtail.PROTOCOL_NAME,
        "protocol_version": longtail.PROTOCOL_VERSION,
        "execution_ready": ready,
        "missing_execution_inputs": missing,
        "prob_commit": arguments.prob_commit,
        "owl_commit": arguments.owl_commit or None,
        "read_only": True,
        "runs": rows,
    }
    if not ready and not arguments.protocol_only:
        raise longtail.LongTailError(
            "Day-2 execution is not provenance-complete: " + "; ".join(missing)
        )
    return report


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    command.add_argument("--anchor-root", type=Path, required=True)
    command.add_argument("--work-root", type=Path, required=True)
    command.add_argument("--owl-commit", default="")
    command.add_argument("--prob-commit", default=PINNED_PROB_COMMIT)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument(
        "--protocol-only", action="store_true",
        help="report missing anchors/OWL commit instead of failing; never execution-ready",
    )
    return command


def main() -> int:
    try:
        report = prepare(parser().parse_args())
    except (longtail.LongTailError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
