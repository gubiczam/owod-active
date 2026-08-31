#!/usr/bin/env python3
"""Build deterministic controlled-long-tail manifests for S-OWODB T1.

This command does not train a model and never writes into the canonical source
archives.  It writes one exact object-identity ledger and one small manifest for
ORIGINAL, LT-10, LT-50, and LT-100, plus diagnostic CSV/PNG outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl import longtail  # noqa: E402
from owl.evaluation_subset import canonical_class_name  # noqa: E402

DEFAULT_SOURCE_INDEX = ROOT / "data" / "reference" / "t1_replay_class_counts.json"
DEFAULT_SOURCE_ANNOTATIONS = ROOT / "data" / "staging" / "owdetr_replay_annotations.tar.gz"
DEFAULT_TEST_ANNOTATIONS = ROOT / "data" / "staging" / "owdetr_test_annotations.tar.gz"
DEFAULT_OUTPUT = ROOT / "data" / "reference" / "longtail"


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def archive_ids(path: Path) -> set[str]:
    with tarfile.open(path) as archive:
        return {
            Path(member.name).stem
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".xml")
        }


def verify_source_xml(
    archive_path: Path, index: dict[str, dict[str, int]],
) -> None:
    """The committed count index must describe the source XMLs exactly."""

    seen: set[str] = set()
    allowed = set(longtail.CONTROLLED_CLASSES)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            image_id = Path(member.name).stem
            handle = archive.extractfile(member)
            if handle is None:
                raise longtail.LongTailError(f"Cannot read {member.name}.")
            root = ElementTree.fromstring(handle.read())
            counts: Counter[str] = Counter()
            for item in root.findall("object"):
                name = canonical_class_name(item.findtext("name", default=""))
                if name in allowed:
                    counts[name] += 1
            expected = index.get(image_id)
            if expected is None:
                raise longtail.LongTailError(
                    f"Source archive contains unindexed T1 image {image_id}."
                )
            if dict(counts) != expected:
                raise longtail.LongTailError(
                    f"Source XML counts differ for {image_id}: {dict(counts)} != {expected}."
                )
            seen.add(image_id)
    missing = set(index) - seen
    if missing:
        raise longtail.LongTailError(
            f"Source archive is missing {len(missing)} indexed XMLs; first: {sorted(missing)[:10]}."
        )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV diagnostics were still written.")
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    styles = {
        "original": ("#333333", "o"),
        "lt10": ("#1f77b4", "s"),
        "lt50": ("#ff7f0e", "^"),
        "lt100": ("#d62728", "D"),
    }
    for condition in longtail.CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        color, marker = styles[condition]
        axis.plot(
            [int(row["rank"]) + 1 for row in selected],
            [int(row["achieved_count"]) for row in selected],
            marker=marker,
            color=color,
            linewidth=1.8,
            markersize=4,
            label=condition.upper().replace("LT", "LT-"),
        )
    axis.set_yscale("log")
    axis.set_xlabel("Original-frequency class rank")
    axis.set_ylabel("Controlled T1 training objects (log scale)")
    axis.set_xticks(range(1, len(longtail.CONTROLLED_CLASSES) + 1))
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, metadata={"Software": "owl controlled long-tail v1"})
    plt.close(figure)


def build(arguments: argparse.Namespace) -> list[dict[str, object]]:
    for path in (
        arguments.source_index, arguments.source_annotations, arguments.test_annotations
    ):
        if not path.is_file():
            raise longtail.LongTailError(f"Missing required source artefact: {path}.")

    before = {
        path: longtail.sha256_file(path)
        for path in (
            arguments.source_index, arguments.source_annotations, arguments.test_annotations
        )
    }
    index = longtail.read_source_index(arguments.source_index)
    counts = longtail.object_counts(index)
    controlled_total = longtail.matched_controlled_total(counts)
    if not arguments.skip_xml_verification:
        verify_source_xml(arguments.source_annotations, index)
    test_ids = archive_ids(arguments.test_annotations)
    leakage = set(index) & test_ids
    if leakage:
        raise longtail.LongTailError(
            f"Canonical T1 train/test leakage: {len(leakage)} images; first {sorted(leakage)[:10]}."
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    per_class: list[dict[str, object]] = []
    for condition in longtail.CONDITIONS:
        targets = longtail.condition_targets(
            counts, condition, controlled_total=controlled_total)
        selection = longtail.select_objects(index, targets, seed=arguments.seed)
        selection_path = arguments.output / f"{condition}_objects.json.gz"
        longtail.write_gzip_json(
            selection_path, longtail.selection_payload(condition, selection))
        selection_sha = longtail.sha256_file(selection_path)
        manifest = longtail.build_manifest(
            condition=condition,
            source_counts=counts,
            targets=targets,
            selected_images=len(selection),
            seed=arguments.seed,
            source_index_path=relative(arguments.source_index),
            source_index_sha256=before[arguments.source_index],
            source_annotations_path=relative(arguments.source_annotations),
            source_annotations_sha256=before[arguments.source_annotations],
            selection_path=relative(selection_path),
            selection_sha256=selection_sha,
            test_split_sha256=before[arguments.test_annotations],
            controlled_total=controlled_total,
        )
        longtail.verify_manifest(manifest)
        manifest_path = arguments.output / f"{condition}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest_file_sha = longtail.sha256_file(manifest_path)
        class_rows = [dict(row, condition=condition) for row in manifest["classes"]]
        per_class.extend(class_rows)
        groups = Counter(str(row["group"]) for row in class_rows)
        values = [int(value) for value in targets.values()]
        summaries.append({
            "condition": condition,
            "requested_rho": manifest["requested_rho"] or "natural",
            "achieved_rho": f"{float(manifest['achieved_rho']):.9f}",
            "images": len(selection),
            "objects": sum(values),
            "n_max": max(values),
            "n_min": min(values),
            "head_classes": groups["head"],
            "medium_classes": groups["medium"],
            "tail_classes": groups["tail"],
            "scientific_sha256": manifest["scientific_sha256"],
            "manifest_file_sha256": manifest_file_sha,
        })

    write_csv(
        arguments.output / "summary.csv",
        list(summaries[0]),
        summaries,
    )
    write_csv(
        arguments.output / "per_class.csv",
        ["condition", "class_name", "original_count", "target_count",
         "achieved_count", "rank", "group"],
        per_class,
    )
    plot_curves(arguments.output / "frequency_curves.png", per_class)

    after = {path: longtail.sha256_file(path) for path in before}
    if after != before:
        raise longtail.LongTailError("A canonical source artefact was modified during generation.")
    print(f"controlled total (LT-10/LT-50/LT-100): {controlled_total}")
    print("condition  images  objects  n_max  n_min  achieved_rho  groups")
    for row in summaries:
        print(
            f"{row['condition']:<9} {row['images']:>7} {row['objects']:>8} "
            f"{row['n_max']:>6} {row['n_min']:>6} {float(row['achieved_rho']):>13.6f} "
            f"{row['head_classes']}/{row['medium_classes']}/{row['tail_classes']}"
        )
    return summaries


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    command.add_argument("--source-annotations", type=Path, default=DEFAULT_SOURCE_ANNOTATIONS)
    command.add_argument("--test-annotations", type=Path, default=DEFAULT_TEST_ANNOTATIONS)
    command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument(
        "--skip-xml-verification", action="store_true",
        help="skip the expensive source XML/index equality audit (never use for release manifests)",
    )
    return command


def main() -> int:
    try:
        build(parser().parse_args())
    except longtail.LongTailError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
