#!/usr/bin/env python3
"""The cheap candidate-side diagnostic for ``distribution_aware_v1``.

Selection only. **No PROB training, no PROB evaluation, no checkpoint is
written.** The one detector call is ``predict``, which scores candidate images
and is what every arm needs to exist at all.

Frozen by ``docs/distribution_aware_decision_memo_2026-09-06.md``. The gates live
in :mod:`owl.active_selection.diagnostic` and are printed as machine-readable
JSON before anything is measured, so the criteria are on the record ahead of the
outcome.

**One deviation from the benchmark, declared before the run.** A real trajectory
scores task *n*'s pool with the checkpoint task *n−1* produced for that arm. This
diagnostic trains nothing, so there are no per-arm checkpoints and every task of
every arm is scored with the **t1 anchor**. That holds the detector fixed and
makes the comparison purely about selection — which is what a candidate-side
diagnostic is for — but it means ``entropy`` here is entropy under the anchor,
not under the arm's own evolving model. Stated, not hidden; it applies
identically to all three arms.

Everything else is the benchmark's own code: the candidate draw keyed on
``(seed, task.index)``, per-image NMS, the ``A`` gate, the cached DINOv2 export,
the growing labelled reference, ``arms.select`` and the annotation ledger.

Oracle labels are read **after** selection and only to describe what was bought.
``tests/test_distribution_aware.py`` asserts the selector cannot reach them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

# The repository root, so `tools.` is importable when this file is run directly.
# `run_full_owod_benchmark.py` does the same, for the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from owl import bridge as bridge_module
from owl import proposals, protocol
from owl.active_selection import arms as arm_registry
from owl.active_selection import benchmark as bm
from owl.active_selection import budget as annotation_budget
from owl.active_selection import diagnostic as gates
from owl.active_selection import population as population_module
from owl.active_selection import semantic
from tools.materialize_pool_images import materialise

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_INDEX = ROOT / "data" / "reference" / "per_image_class_counts.json"
CLASS_GROUPS = ROOT / "data" / "reference" / "class_groups.csv"


class DiagnosticError(RuntimeError):
    """Raised when the diagnostic is asked for something it must not do."""


def require_reference(selected: Sequence[str], ref_t1: Path | None) -> None:
    """REF-T1 is a scientific prerequisite, not a convenience.

    The memo defines ``distribution_aware_v1``'s reference as **REF-T1 plus every
    image this trajectory has bought**, and ``R_k`` is read against it. Without
    REF-T1 the reference at ``t2`` is empty, every cluster reads as maximally
    under-represented and the quota degenerates toward proportional-to-size —
    a *different method*, silently substituted because an old Drive file was
    missing. One protocol, or none: this refuses rather than falling back.
    """

    needs = [a for a in selected
             if arm_registry.ARMS[a].needs_semantic
             and arm_registry.ARMS[a].reference_scope == "labelled"]
    if not needs:
        return
    if ref_t1 is None or not Path(ref_t1).is_file():
        raise DiagnosticError(
            f"{needs} measure under-representation against REF-T1 plus what the "
            f"trajectory has bought, and --ref-t1 is {ref_t1 or 'absent'}. "
            "Running without it would silently measure a different method: an "
            "empty reference at t2 makes every cluster maximally "
            "under-represented. Build the frozen export with "
            "tools/export_ref_t1_features.py --per-class-cap 1000, or run only "
            "arms that do not consult it."
        )


# ------------------------------------------------------------------ oracle ---


def class_groups() -> dict[str, str]:
    with CLASS_GROUPS.open(encoding="utf-8") as handle:
        return {r["class_name"]: r["group"] for r in csv.DictReader(handle)}


def oracle_row(
    opened: Sequence[str],
    counts: Mapping[str, Mapping[str, int]],
    *,
    known: frozenset[str],
    current_new: str,
    future_new: frozenset[str],
    groups: Mapping[str, str],
    held: Mapping[str, int],
) -> dict[str, object]:
    """Describe what an arm bought. Read **after** selection, never before."""

    objects: dict[str, int] = {}
    barren = 0
    for image in opened:
        on_image = counts.get(str(image), {})
        if not on_image:
            barren += 1
        for name, n in on_image.items():
            objects[name] = objects.get(name, 0) + int(n)

    unknown = {k: v for k, v in objects.items() if k not in known}
    by_group = {g: 0 for g in ("head", "medium", "tail")}
    for name, n in objects.items():
        group = groups.get(name)
        if group in by_group:
            by_group[group] += n

    acquired_now = objects.get(current_new, 0)
    return {
        "images_opened": len(opened),
        "background_share": round(barren / max(len(opened), 1), 4),
        "acquired_class_breadth": len(unknown),
        "objects_total": sum(objects.values()),
        "current_new_objects": acquired_now,
        "banked_from_earlier": int(held.get(current_new, 0)),
        "held_at_declaration": acquired_now + int(held.get(current_new, 0)),
        "future_new_objects": sum(objects.get(c, 0) for c in future_new),
        "head_objects": by_group["head"],
        "medium_objects": by_group["medium"],
        "tail_objects": by_group["tail"],
        "unknown_objects": sum(unknown.values()),
    }


# ------------------------------------------------------------------ driver ---


def run_arm(
    arm: str,
    *,
    seed: int,
    chain,
    all_images: Sequence[str],
    counts: Mapping[str, Mapping[str, int]],
    groups: Mapping[str, str],
    workspace: Path,
    predict_for: Callable[..., object],
    features_for: Callable[..., np.ndarray],
    reference_for: Callable[..., np.ndarray],
    ref_t1: Path | None,
    prepare_images: Callable[[Sequence[str]], Sequence[str]] | None,
    budget: int,
    pool_size: int,
) -> tuple[list[dict], dict[tuple[str, int, str], list[str]]]:
    spec = arm_registry.ARMS[arm]
    cost_of = annotation_budget.cost_function(counts)
    known = frozenset(protocol.TASK1)
    declared = [t.new_class for t in chain[1:]]

    used: set[str] = set()
    held: dict[str, int] = {}
    rows: list[dict] = []
    opened_by_task: dict[tuple[str, int, str], list[str]] = {}

    for position, task in enumerate(chain[1:]):
        task_dir = workspace / f"{arm}__seed{seed}" / task.name
        task_dir.mkdir(parents=True, exist_ok=True)

        generator = np.random.default_rng([seed, task.index])
        pool_ids = np.asarray([i for i in all_images if i not in used], dtype=object)
        take = min(pool_size, pool_ids.size)
        candidate_ids = [str(v) for v in generator.choice(pool_ids, size=take, replace=False)]

        # The pixels. `prepare_full_owod_benchmark.py` deliberately does not fetch
        # candidate images — which 2,000 a task scores depends on `(seed, task)`
        # and on what the arm has already bought — so the caller that draws the
        # pool is the one that must materialise it. The benchmark launcher does
        # this through the same materialiser; so does this.
        if prepare_images is not None:
            available = [str(v) for v in prepare_images(candidate_ids)]
            dropped = len(candidate_ids) - len(available)
            if dropped:
                print(f"  [{task.name}/{arm}] {dropped} of {len(candidate_ids)} "
                      "candidate images could not be fetched; dropped", flush=True)
            candidate_ids = available
        if not candidate_ids:
            raise DiagnosticError(
                f"{task.name}/{arm} has no usable candidate images. PROB reads "
                "JPEGs off disk, so either the fetch failed for all of them or "
                "the data root is wrong."
            )

        # Content-addressed, because at t2 every arm draws the *same* pool
        # (`used` is empty and the draw is keyed on `(seed, task)` alone). Keying
        # the cache on the image list rather than on the arm turns 18 detector
        # passes into 14 — 8,000 images of T4 time, for free and with no effect
        # on any number, since identical input gives identical output.
        digest = hashlib.sha256("\n".join(sorted(candidate_ids)).encode()).hexdigest()
        shared = workspace / "_predict" / f"{digest[:16]}.npz"
        shared.parent.mkdir(parents=True, exist_ok=True)
        export = predict_for(candidate_ids, task=task, output=shared)
        candidates = proposals.from_predict(export)
        pool = population_module.build(candidates)

        features = reference = None
        ranked = arm_registry.ranked_positions(arm, pool)
        if spec.needs_semantic:
            features = features_for(
                task_dir / "dinov2_pool.npz",
                pool.candidates.image_ids[ranked],
                pool.candidates.boxes[ranked],
                device="cuda", batch_size=128,
                label=f"{task.name}/{arm} dinov2",
                provenance={"task": task.name, "arm": arm, "seed": seed},
            )
            anchor = [reference_for(ref_t1)] if (
                ref_t1 is not None and spec.reference_scope == "labelled") else []
            reference = semantic.stack_reference([
                *anchor, *bm.reference_blocks(task_dir, task_index=task.index),
            ])

        picked = arm_registry.select(
            arm, pool, cost_of=cost_of, answer_budget=budget, seed=seed,
            semantic=features, reference=reference,
            excluded_images=frozenset(used),
        )

        if spec.needs_semantic and features is not None:
            block = np.asarray(features[picked.covered[ranked]], dtype=np.float16)
            temporary = task_dir / "coverage_reference.npz.part"
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, features=block,
                                    task=np.asarray(task.name), arm=np.asarray(arm))
            os.replace(temporary, task_dir / "coverage_reference.npz")

        row = oracle_row(
            picked.images, counts, known=known, current_new=task.new_class,
            future_new=frozenset(declared[position + 1:]), groups=groups, held=held,
        )
        row = {"arm": arm, "seed": seed, "task": task.name,
               "class": task.new_class} | dict(picked.row) | row
        row["tail_per_image"] = round(
            row["held_at_declaration"] / max(row["images_opened"], 1), 4)
        row["population_ranked"] = int(ranked.size)
        rows.append(row)
        opened_by_task[(arm, seed, task.name)] = list(picked.images)

        for image in picked.images:
            for name, n in counts.get(str(image), {}).items():
                held[name] = held.get(name, 0) + int(n)
        used.update(picked.images)

    return rows, opened_by_task


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prob-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="the t1 anchor, used for every task")
    parser.add_argument("--ref-t1", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(gates.DIAGNOSTIC_SEEDS))
    parser.add_argument("--arms", nargs="+", default=list(gates.DIAGNOSTIC_ARMS))
    parser.add_argument("--fetch-workers", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true",
                        help="stub the detector and DINOv2; exercises every other line")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("FROZEN GATE CONFIGURATION — printed before anything is measured")
    print("=" * 78)
    print(json.dumps(gates.configuration(), indent=2))
    print("=" * 78, flush=True)

    counts = json.loads(CANDIDATE_INDEX.read_text(encoding="utf-8"))
    groups = class_groups()
    chain = protocol.build_chain(bm.N_TASKS)
    all_images = sorted(counts)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("\n*** DRY RUN: the detector and DINOv2 are stubbed. Every number "
              "below is plumbing, not evidence. ***\n")
        predict_for, features_for, reference_for, prepare = _stubs(counts)
    else:
        if not (args.prob_root and args.data_root and args.checkpoint):
            raise DiagnosticError(
                "a real run needs --prob-root, --data-root and --checkpoint (the t1 "
                "anchor). Pass --dry-run to exercise the control flow without them."
            )
        require_reference(args.arms, args.ref_t1)
        bridge = bridge_module.Bridge(
            prob_root=args.prob_root, data_root=args.data_root,
            device=args.device, num_workers=args.num_workers, seed=0,
            log_dir=args.out / "logs",
        )
        bridge.check()

        def predict_for(image_ids, *, task, output):
            return bridge.predict(
                image_ids, checkpoint=args.checkpoint, output=output,
                n_prev=0, n_current=task.n_prev,
                max_proposals_per_image=bm.PROPOSALS_PER_IMAGE,
            )

        jpeg = Path(args.data_root) / "JPEGImages"

        def prepare(image_ids):
            got = materialise([str(v) for v in image_ids], jpeg,
                              workers=args.fetch_workers)
            lost = set(got["unreadable"])
            return [v for v in dict.fromkeys(str(i) for i in image_ids)
                    if v not in lost]

        features_for = _wrap_features(semantic.cached, args.data_root)
        reference_for = semantic.reference_from_ref_t1

    rows: list[dict] = []
    opened: dict[tuple[str, int, str], list[str]] = {}
    for seed in args.seeds:
        for arm in args.arms:
            print(f"\n--- {arm} seed {seed} ---", flush=True)
            got, by_task = run_arm(
                arm, seed=seed, chain=chain, all_images=all_images, counts=counts,
                groups=groups, workspace=args.out / "work",
                predict_for=predict_for, features_for=features_for,
                reference_for=reference_for, ref_t1=args.ref_t1,
                prepare_images=prepare, budget=bm.ANSWER_BUDGET_PER_TASK,
                pool_size=bm.CANDIDATE_IMAGES_PER_TASK,
            )
            rows.extend(got)
            opened.update(by_task)
            for row in got:
                print(f"  {row['task']} {row['class']:<14} images "
                      f"{row['images_opened']:>4} answers {row['answers_spent']:>5} "
                      f"held {row['held_at_declaration']:>4} "
                      f"/img {row['tail_per_image']:.4f}", flush=True)

    fields = sorted({k for r in rows for k in r})
    with (args.out / "diagnostic_rows.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    verdict = gates.evaluate(rows, opened, seeds=args.seeds)
    payload = {"configuration": gates.configuration(), **verdict.as_dict()}
    (args.out / "verdict.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    for outcome in verdict.outcomes:
        row = outcome.row()
        print(f"{row['verdict']:>4}  {row['gate']:<32} threshold {row['threshold']:<6} "
              f"measured {row['measured_per_seed']}")
    print("=" * 78)
    print(f"VERDICT: {'GO' if verdict.go else 'NO-GO'}")
    if verdict.failed:
        print("failed:", ", ".join(verdict.failed))
    print(f"\nwrote {args.out / 'diagnostic_rows.csv'} and {args.out / 'verdict.json'}")
    return 0


def _wrap_features(cached, data_root: Path):
    jpeg_dir = Path(data_root) / "JPEGImages"

    def features_for(path, image_ids, boxes, **kwargs):
        return cached(path, image_ids, boxes, jpeg_dir, **kwargs)

    return features_for


def _stubs(counts):
    """Deterministic fakes so the whole control flow runs without a GPU.

    They stand in for the detector and DINOv2 **only**. The population, the
    ``A`` gate, HDBSCAN, the rarity quota, the reference bookkeeping, the ledger
    and the gates are the real code. Nothing they produce is a scientific
    result and the driver prints so.
    """

    def predict_for(image_ids, *, task, output):
        generator = np.random.default_rng(abs(hash((task.name, len(image_ids)))) % 2**32)
        n = len(image_ids) * bm.PROPOSALS_PER_IMAGE
        posterior = generator.dirichlet(np.full(81, 0.4), size=n).astype(np.float32)
        payload = {
            "image_ids": np.repeat(np.asarray(image_ids, dtype=str), bm.PROPOSALS_PER_IMAGE),
            "boxes": np.clip(generator.random((n, 4)) * 0.6 + 0.2, 0.05, 0.95).astype(np.float32),
            "posterior": posterior,
            "objectness": generator.random(n).astype(np.float32),
            "embeddings": generator.normal(size=(n, 32)).astype(np.float32),
        }
        np.savez_compressed(output, **payload)
        return output

    def features_for(path, image_ids, boxes, **kwargs):
        generator = np.random.default_rng(len(image_ids))
        block = generator.normal(size=(len(image_ids), 24))
        return (block / np.linalg.norm(block, axis=1, keepdims=True)).astype(np.float32)

    def reference_for(_path):
        generator = np.random.default_rng(11)
        block = generator.normal(size=(500, 24))
        return (block / np.linalg.norm(block, axis=1, keepdims=True)).astype(np.float32)

    def prepare(image_ids):
        """Exercise the materialisation branch without a network.

        The real path fetches candidate pixels here, and a dry run that skipped
        it would leave exactly the branch that was missing untested — which is
        how the missing fetch survived review in the first place.
        """

        return [str(i) for i in dict.fromkeys(str(v) for v in image_ids)]

    return predict_for, features_for, reference_for, prepare


if __name__ == "__main__":
    raise SystemExit(main())
