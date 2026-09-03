"""Full OWOD Active Selection Benchmark V1: the frozen protocol, in code.

Protocol document: ``docs/full_owod_active_benchmark_v1_protocol_2026-09-03.md``.
Every number this module declares appears there and nowhere else is authoritative;
:func:`check_protocol` compares the two as *values*, because a Method V3 overnight
session was lost to an assertion that matched a protocol's English prose.

What this benchmark is, stated so it cannot be overclaimed later:

* a **true sequential chain**. ``t3`` fine-tunes ``t2``'s own checkpoint for its
  own arm; no task restarts from the anchor and no two arms share a checkpoint.
  ``tests/test_full_benchmark_chain.py`` asserts the lineage.
* the repository's **canonical task chain**, which declares **one class per
  task** — ``t2`` traffic light, ``t3`` fire hydrant, ``t4`` stop sign. It is
  *not* S-OWODB's published 19/21/20/20 split, and no result here may be
  compared against a published S-OWODB number. What it buys is a measurable
  new-class endpoint at an affordable annotation budget, and a tail band that
  **grows with the chain** — ``{bear}`` at t2, ``{bear, fire hydrant}`` at t3,
  ``{bear, fire hydrant, stop sign}`` at t4 — so tail AP at t4 is directly a
  function of what the selector acquired.
* budgeted in **oracle answers**, not regions. See
  :mod:`owl.active_selection.budget` for why the region unit was abandoned.
* **exploratory**. There is no GO/NO-GO gate. The primary contrast is declared
  in advance so that the arm order and the metric cannot be chosen after the
  numbers arrive, and the reporting rules in :data:`REPORTING` say what may and
  may not be claimed from a single seed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from owl import protocol, runner
from owl.active_selection import arms as arm_registry
from owl.active_selection import budget as annotation_budget
from owl.active_selection import population as population_module
from owl.active_selection import semantic
from owl.active_selection.population import ADMISSIBLE_SHARE, NMS_IOU

ROOT = Path(__file__).resolve().parent.parent.parent
PROTOCOL_PATH = ROOT / "docs" / "full_owod_active_benchmark_v1_protocol_2026-09-03.md"

# ------------------------------------------------------------- the protocol ---

#: t1 anchor plus three incremental tasks. The chain, not a slice of it.
N_TASKS = 4

#: Oracle answers per task, under full-image labelling. Chosen from the
#: benchmark's own measured density — 9.56 annotated objects per candidate image
#: — to open roughly 300 images and deliver roughly the supervision Method V3's
#: random arm received, so the two are on comparable footing. Fixed before any
#: trajectory ran.
ANSWER_BUDGET_PER_TASK = 3000

#: Fresh images the detector scores at each task. Sampled per (seed, task) from
#: the 28,800-image candidate index, disjoint from what earlier tasks bought.
#:
#: 2,000 rather than 1,200 — the value the project's own completed six-task GPU
#: chain used — because the detector pass costs 4 minutes per thousand images
#: while training costs twenty-five, so a larger pool is the cheapest way to put
#: more of the rare declared class within reach of the selector. At 2,000 the
#: pool is expected to hold ~150 images with a traffic light, ~63 with a fire
#: hydrant and ~62 with a stop sign; at 1,200 it would hold ~90, ~38 and ~37.
CANDIDATE_IMAGES_PER_TASK = 2000

#: PROB offers 100 queries per image; the top 50 by its own objectness order is
#: what every export in this project keeps.
PROPOSALS_PER_IMAGE = 50

#: One. Not six.
#:
#: The repository's ``rounds_per_task`` recomputes the *score* with a grown
#: labelled pool; it does not re-run the detector. For ``random``, ``entropy``
#: and ``admissibility`` the score does not depend on the labelled pool at all,
#: so six rounds provably return the same prefix as one — Method V3 measured
#: exactly that. For the two coverage arms the traversal is already sequential
#: by construction: its reference grows at every pick. So six rounds would buy
#: nothing here and calling them "iterative active learning" would be the
#: mislabelling the consultation warned against. Genuine detector rescoring
#: every 100 answers is Phase 2 and is priced in the protocol.
ROUNDS_PER_TASK = 1

#: One fixed replay policy for every arm, so replay cannot become the
#: explanation. ``uniform`` at M=400 exemplar *objects* is the project's
#: established matched control and the setting Method V3 held fixed.
REPLAY_ARM = "uniform"
REPLAY_OBJECTS = 400

#: Full-image labelling: every annotated object on an opened image is labelled.
#: This is what PROB's ``ft`` supervision mode actually delivers on the GPU path
#: — it keeps every declared-class box of every image it is handed — so naming
#: the policy anything else would describe a behaviour the detector never
#: performs.
LABELLING_POLICY = "full_image"
SUPERVISION_MODE = "ft"

# --- training, unchanged from every previous GPU run in this project ---------
EPOCHS = 5
LEARNING_RATE = 2e-4
BATCH_SIZE = 2

#: The shared evaluation split: every declared class of the whole chain capped
#: at 150 test images, plus twice that many sampled others. Built **once** for
#: the chain and used for the anchor and for every task of every arm, so
#: forgetting is a difference between two numbers measured on the same images.
EVAL_MAX_PER_CLASS = 150
EVAL_REMAINDER_RATIO = 2

#: Seeds. Seed 0 is the development and first-result seed; 1 and 2 are the
#: replication seeds and are run only after a proposed selector is frozen.
SEEDS: tuple[int, ...] = (0, 1, 2)
DEVELOPMENT_SEED = 0

#: How much of a session one arm may consume before the chain stops cleanly.
#: A stopped arm resumes; it is never reported as complete.
ARM_TIME_BUDGET_MINUTES = 260.0

#: Keep two checkpoints per arm: the one a task starts from and the one it
#: wrote. Five arms x 478 MB x 3 tasks would not fit on a free Drive.
KEEP_CHECKPOINTS = 2


class BenchmarkError(ValueError):
    """Raised when the protocol and the code disagree, or an input is wrong."""


@dataclass(frozen=True)
class Endpoints:
    """What was declared before the first trajectory ran.

    No threshold, no verdict. The point of freezing these is that the *contrast*
    and the *metric* are chosen in advance, so a result cannot be assembled
    afterwards from whichever pair of arms happened to differ.
    """

    primary_contrast: tuple[str, str] = ("proposed", "admissibility")
    primary_task: str = "t4"
    primary_metric: str = "known_mAP50"
    #: The long-tail endpoint. Reported at every task, and the reason the chain
    #: is run to t4 at all: the tail band holds one class at t2 and three at t4.
    longtail_metric: str = "mAP50_tail"
    #: The acquisition endpoint, measured without the detector and therefore
    #: available even if a trajectory fails.
    acquisition_metric: str = "acquired_classes"
    ablation_contrast: tuple[str, str] = ("proposed", "coreset")
    reference_arm: str = "random"

    def statement(self) -> str:
        return (
            f"Primary contrast: {self.primary_contrast[0]} vs "
            f"{self.primary_contrast[1]} at {self.primary_task}, on "
            f"{self.primary_metric}, with {self.longtail_metric} as the "
            f"long-tail endpoint and {self.acquisition_metric} as the "
            f"detector-free acquisition endpoint. Gate ablation: "
            f"{self.ablation_contrast[0]} vs {self.ablation_contrast[1]}. "
            f"Reference arm: {self.reference_arm}. Seeds {list(SEEDS)}; seed "
            f"{DEVELOPMENT_SEED} is the development seed and a difference "
            "measured on it alone is exploratory."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "primary_contrast": list(self.primary_contrast),
            "primary_task": self.primary_task,
            "primary_metric": self.primary_metric,
            "longtail_metric": self.longtail_metric,
            "acquisition_metric": self.acquisition_metric,
            "ablation_contrast": list(self.ablation_contrast),
            "reference_arm": self.reference_arm,
        }


ENDPOINTS = Endpoints()

#: What a single-seed result may and may not be reported as. Printed by the
#: summariser next to every table, so the caveat travels with the numbers.
REPORTING: tuple[str, ...] = (
    (
        "One seed gives no error bar. Method V3's audit measured that PROB's "
        "seed was never varied and that paired arms share no common random "
        "numbers, so the nondeterminism floor of this pipeline is still "
        "unmeasured. A seed-0-only difference is a direction, not an effect."
    ),
    (
        "The tail band holds one class at t2 (bear) and gains one at each later "
        "task. Never write 'tail classes improve' for a band of one class; "
        "name it."
    ),
    (
        "Equal oracle answers means equal labelled boxes, not equal gradient "
        "steps: arms open different numbers of images. training_iterations is "
        "in every row and must be quoted whenever an AP difference is discussed."
    ),
    (
        "The chain declares one class per task and is not the published "
        "S-OWODB task split. No number here may be compared against a "
        "published S-OWODB result."
    ),
)


def chain() -> tuple[protocol.Task, ...]:
    """The benchmark's task chain: t1 anchor, then t2, t3, t4."""

    return protocol.build_chain(N_TASKS)


def declared_classes() -> tuple[str, ...]:
    """The classes the chain introduces — what the shared eval split is built on."""

    return tuple(task.new_class for task in chain()[1:] if task.new_class)


def tail_band(task: protocol.Task, groups: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The tail classes known after ``task``. One at t2, three at t4."""

    groups = protocol.load_groups() if groups is None else groups
    return tuple(name for name in task.known_classes if groups.get(name) == "tail")


# ------------------------------------------------------------------ configs ---


def cycle_config(arm: str, seed: int) -> runner.CycleConfig:
    """The frozen :class:`owl.runner.CycleConfig` for one trajectory."""

    if arm not in arm_registry.ARMS:
        raise BenchmarkError(
            f"Unknown arm {arm!r}; registered: {sorted(arm_registry.ARMS)}."
        )
    if seed not in SEEDS:
        raise BenchmarkError(f"seed={seed} is not one of the declared seeds {SEEDS}.")
    return runner.CycleConfig(
        n_tasks=N_TASKS,
        budget_per_task=ANSWER_BUDGET_PER_TASK,
        budget_unit="answers",
        rounds_per_task=ROUNDS_PER_TASK,
        candidate_images_per_task=CANDIDATE_IMAGES_PER_TASK,
        proposals_per_image=PROPOSALS_PER_IMAGE,
        arm=arm,
        labelling_policy=LABELLING_POLICY,
        replay_arm=REPLAY_ARM,
        replay_reallocate=False,
        replay_protocol_version=3,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        seed=seed,
        measure_grouped_recall=True,
        reuse_deferred_labels=True,
        keep_checkpoints=KEEP_CHECKPOINTS,
    )


def trajectory_name(arm: str, seed: int) -> str:
    """Workspace name. One directory per (arm, seed); never shared."""

    return f"{arm}__seed{seed}"


# ------------------------------------------------------------- the selector ---

_TASK_DIR = re.compile(r"^t(\d+)_")


def reference_blocks(task_dir: Path, *, task_index: int) -> list[np.ndarray]:
    """Semantic features of everything this arm bought at **earlier** tasks.

    Read off disk rather than held in memory, because a resumed session restores
    a finished task from ``state.json`` and never re-runs its selection. Each
    task writes exactly one block under its own directory, so re-running a task
    overwrites its own contribution and nothing is double-counted.
    """

    blocks: list[np.ndarray] = []
    workspace = Path(task_dir).parent
    for path in sorted(workspace.glob("t*/coverage_reference.npz")):
        match = _TASK_DIR.match(path.parent.name)
        if match is None or int(match.group(1)) >= task_index:
            continue
        payload = np.load(path, allow_pickle=False)
        blocks.append(np.asarray(payload["features"], dtype=np.float32))
    return blocks


def make_selector(
    arm: str,
    *,
    candidate_index: Mapping[str, Mapping[str, int]],
    jpeg_dir: str | Path,
    ref_t1: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 128,
    features_for: Callable[..., np.ndarray] = semantic.cached,
    reference_for: Callable[..., np.ndarray] = semantic.reference_from_ref_t1,
) -> Callable[..., arm_registry.ArmSelection]:
    """Build the ``selector`` callback :func:`owl.runner.run_chain` expects.

    ``features_for`` and ``reference_for`` are injected so a dry run can
    exercise the whole control flow without a GPU, a DINOv2 download or the
    frozen reference export. They are the *only* two seams: the population, the
    traversal, the ledger, the cost accounting and the cross-task reference
    bookkeeping are the same code in a dry run as in a real one.
    """

    spec = arm_registry.ARMS[arm]
    cost_of = annotation_budget.cost_function(candidate_index)
    jpeg_dir = Path(jpeg_dir)
    if spec.needs_semantic and ref_t1 is None:
        raise BenchmarkError(
            f"Arm {arm!r} covers semantic space relative to what is already "
            "labelled, and the balanced task-1 reference was not given. Pass "
            "the frozen ref_t1 export, or run an arm that does not consult it."
        )

    def selector(candidates, *, task, task_dir, used_images, budget, seed):
        task_dir = Path(task_dir)
        pool = population_module.build(candidates)
        print(f"  [{task.name}/{arm}] population {len(pool):,} of "
              f"{len(candidates):,} proposals after NMS on "
              f"{pool.diagnostics['images']:,} images; "
              f"{pool.diagnostics['admissible']:,} admissible")

        features = None
        reference = None
        if spec.needs_semantic:
            # Only the rows this arm ranks are embedded: G for a gated arm, the
            # whole deduplicated pool for the ungated control. The cache is keyed
            # on a fingerprint of exactly those rows, so the two arms cannot
            # reuse each other's file.
            ranked = np.flatnonzero(pool.gate) if spec.gated else np.arange(len(pool))
            features = features_for(
                task_dir / "dinov2_pool.npz",
                pool.candidates.image_ids[ranked],
                pool.candidates.boxes[ranked],
                jpeg_dir,
                device=device,
                batch_size=batch_size,
                label=f"{task.name}/{arm} dinov2",
                provenance={"task": task.name, "arm": arm, "seed": seed},
            )
            reference = semantic.stack_reference([
                reference_for(ref_t1),
                *reference_blocks(task_dir, task_index=task.index),
            ])

        picked = arm_registry.select(
            arm, pool,
            cost_of=cost_of,
            answer_budget=int(budget),
            seed=seed,
            semantic=features,
            reference=reference,
            excluded_images=frozenset(used_images),
        )

        if spec.needs_semantic and features is not None:
            # `covered` is indexed on the pool; the features are indexed on the
            # rows the arm ranked. Map through `ranked` rather than assuming they
            # are the same length, which for a gated arm they are not.
            block = np.asarray(features[picked.covered[ranked]], dtype=np.float16)
            # `np.savez_compressed` appends '.npz' unless the name already ends
            # in it, so a '.part' temporary must be written through a handle or
            # the atomic rename looks for a file numpy never created.
            temporary = task_dir / "coverage_reference.npz.part"
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, features=block,
                                    task=np.asarray(task.name), arm=np.asarray(arm))
            os.replace(temporary, task_dir / "coverage_reference.npz")

        row = dict(picked.row) | {
            "population_proposals": pool.diagnostics["proposals_after_nms"],
            "population_images": pool.diagnostics["images"],
            "population_admissible": pool.diagnostics["admissible"],
        }
        return arm_registry.ArmSelection(
            arm=picked.arm, images=picked.images, anchors=picked.anchors,
            row=row, covered=picked.covered,
        )

    return selector


# ------------------------------------------------- protocol self-consistency ---

#: The tagged block in the protocol document that holds the frozen values. One
#: source of truth, compared as values. Prose is documentation; this is the
#: contract. A Method V3 overnight run was lost to a substring match against a
#: rendered float, which is what this construction exists to prevent.
CRITERION_BLOCK = re.compile(r"^```json protocol\n(.*?)^```", re.DOTALL | re.MULTILINE)


def frozen_values() -> dict[str, object]:
    """Everything the protocol document must agree with, as data."""

    return {
        "n_tasks": N_TASKS,
        "answer_budget_per_task": ANSWER_BUDGET_PER_TASK,
        "candidate_images_per_task": CANDIDATE_IMAGES_PER_TASK,
        "proposals_per_image": PROPOSALS_PER_IMAGE,
        "rounds_per_task": ROUNDS_PER_TASK,
        "replay_arm": REPLAY_ARM,
        "replay_objects": REPLAY_OBJECTS,
        "labelling_policy": LABELLING_POLICY,
        "supervision_mode": SUPERVISION_MODE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "eval_max_per_class": EVAL_MAX_PER_CLASS,
        "eval_remainder_ratio": EVAL_REMAINDER_RATIO,
        "seeds": list(SEEDS),
        "nms_iou": NMS_IOU,
        "admissible_share": ADMISSIBLE_SHARE,
        "arms": list(arm_registry.ORDER),
        "endpoints": ENDPOINTS.as_dict(),
    }


def parse_protocol_block(text: str) -> dict[str, object]:
    match = CRITERION_BLOCK.search(text)
    if match is None:
        raise BenchmarkError(
            "the protocol document holds no ```json protocol``` block. That "
            "block is the single machine-readable source of the frozen values; "
            "without it the code and the document cannot be compared."
        )
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise BenchmarkError(
            f"the protocol's ```json protocol``` block is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise BenchmarkError("the protocol block must be a JSON object.")
    return payload


def check_protocol(path: str | Path | None = None) -> dict[str, object]:
    """Assert the document and the code declare the same values."""

    path = Path(path) if path is not None else PROTOCOL_PATH
    if not path.is_file():
        raise BenchmarkError(f"{path} is missing; the protocol must be committed.")
    stated = parse_protocol_block(path.read_text(encoding="utf-8"))
    expected = frozen_values()

    missing = sorted(set(expected) - set(stated))
    extra = sorted(set(stated) - set(expected))
    differing = {
        name: (stated[name], value)
        for name, value in expected.items()
        if name in stated and stated[name] != value
    }
    if missing or extra or differing:
        lines = []
        if missing:
            lines.append(f"    absent from the document: {missing}")
        if extra:
            lines.append(f"    in the document but not in the code: {extra}")
        for name, (was, now) in sorted(differing.items()):
            lines.append(f"    {name}: document {was!r}, code {now!r}")
        raise BenchmarkError(
            f"{path} and owl.active_selection.benchmark disagree:\n"
            + "\n".join(lines)
            + "\nOne of them is wrong. Decide which, in the research log."
        )
    return {"path": str(path), "fields": len(expected), "agrees": True}


# -------------------------------------------------------------- provenance ---


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Atomic: a killed session leaves the previous file, never half of one."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                         encoding="utf-8")
    os.replace(temporary, path)
    return path


def manifest(
    *,
    trajectories: Sequence[Mapping[str, object]],
    owl_commit: str,
    prob_commit: str,
    prob_repository: str,
    checkpoint: str,
    checkpoint_sha256: str | None,
    test_set: str,
    test_images: int,
    dry_run: bool = False,
) -> dict[str, object]:
    """The machine-readable record of one session."""

    return {
        "experiment": "full_owod_active_benchmark_v1",
        "dry_run": bool(dry_run),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "frozen": frozen_values(),
        "endpoints": ENDPOINTS.as_dict(),
        "endpoint_statement": ENDPOINTS.statement(),
        "reporting_rules": list(REPORTING),
        "chain": [
            {"task": task.name, "new_class": task.new_class,
             "known_after": task.n_current, "tail_band": list(tail_band(task))}
            for task in chain()
        ],
        "pins": {
            "owl_commit": owl_commit,
            "prob_repository": prob_repository,
            "prob_commit": prob_commit,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "evaluation": {
            "test_set": test_set,
            "images": test_images,
            "max_per_class": EVAL_MAX_PER_CLASS,
            "remainder_ratio": EVAL_REMAINDER_RATIO,
            "shared": "one split for the anchor and every task of every arm",
        },
        "trajectories": [dict(row) for row in trajectories],
        "config_example": asdict(cycle_config(arm_registry.ORDER[0], DEVELOPMENT_SEED)),
    }
