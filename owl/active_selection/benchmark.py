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

from owl import protocol, replay, runner, supervision
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


@dataclass(frozen=True)
class KillRule:
    """When a development-seed-informed arm does **not** get replication seeds.

    Frozen before Proposed-v2's first trajectory. Both thresholds are values
    already on the record from the seed-0 baseline run, not numbers chosen to
    be reachable:

    * ``3.56`` is half of the admissibility arm's seed-0 mean
      ``new_class_AP50`` of 7.12 — clear less than half the bar and the method
      has not addressed the failure it was designed for;
    * ``44.89`` is Proposed-v1's own seed-0 final ``known_mAP50`` — fall below
      it and v2 has not even improved on the version it replaces.

    Failing either preserves v2 as a negative development result. It is not
    tuned afterwards and it does not get seeds 1 and 2.
    """

    arm: str = "proposed_v2"
    seed: int = DEVELOPMENT_SEED
    minimum_mean_new_class_ap50: float = 3.56
    minimum_final_known_map50: float = 44.89

    def statement(self) -> str:
        return (
            f"{self.arm} proceeds to seeds {[s for s in SEEDS if s != self.seed]} "
            f"only if its seed-{self.seed} run gives mean new_class_AP50 >= "
            f"{self.minimum_mean_new_class_ap50} AND final known_mAP50 >= "
            f"{self.minimum_final_known_map50}. Failing either, it is preserved "
            "as a negative development result, not tuned."
        )

    def decide(
        self, mean_new_class_ap50: float | None, final_known_map50: float | None
    ) -> dict[str, object]:
        """Mechanical. No judgement is applied at reporting time."""

        if mean_new_class_ap50 is None or final_known_map50 is None:
            return {"verdict": "INCOMPLETE", "reasons": ["a required endpoint is absent"]}
        reasons = []
        if mean_new_class_ap50 < self.minimum_mean_new_class_ap50:
            reasons.append(
                f"mean new_class_AP50 {mean_new_class_ap50:.2f} < "
                f"{self.minimum_mean_new_class_ap50}")
        if final_known_map50 < self.minimum_final_known_map50:
            reasons.append(
                f"final known_mAP50 {final_known_map50:.2f} < "
                f"{self.minimum_final_known_map50}")
        return {
            "verdict": "STOP" if reasons else "PROCEED",
            "reasons": reasons or ["both thresholds met"],
            "mean_new_class_AP50": mean_new_class_ap50,
            "final_known_mAP50": final_known_map50,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm, "seed": self.seed,
            "minimum_mean_new_class_ap50": self.minimum_mean_new_class_ap50,
            "minimum_final_known_map50": self.minimum_final_known_map50,
        }


KILL_RULE = KillRule()

#: Arms whose design followed inspection of a detector endpoint. Reported as
#: such, always. Everything else in :data:`owl.active_selection.arms.ORDER` was
#: fixed before the first trajectory ran.
#:
#: The ``research_*`` arms carry the label conservatively rather than exactly.
#: No V1 endpoint informed a single term of them — every term comes from the
#: research plan and the 2026-08-25 consultation, and
#: ``docs/full_owod_v2_protocol.md`` pre-registers them before the V2 chain
#: runs. But they were *added* after V1 seed-0 numbers existed, and the
#: convention here is that anything added after results exist says so.
DEVELOPMENT_SEED_INFORMED: tuple[str, ...] = (
    "proposed_v2", "cost_aware", "distribution_aware_v1",
    "distribution_aware_iterative_v1",
    "research_v2", "research_v2_plan", "research_v2_no_gate",
    "research_v2_labeled_only", "research_v2_batch_only",
)

PROVENANCE: tuple[str, ...] = (
    (
        "distribution_aware_v1 and its mandatory cost_aware control were frozen "
        "on 2026-09-06 in docs/distribution_aware_decision_memo_2026-09-06.md, "
        "after seed-0 and seed-1 endpoints were known. They are "
        "development-seed-informed, not pre-registered, and no table may "
        "present them otherwise."
    ),
    (
        "Proposed-v2 was designed after inspection of Proposed-v1 "
        "development-seed (seed 0) results and is therefore "
        "development-seed-informed, not pre-registered."
    ),
    (
        "The coreset gate ablation is unavailable: its seed-0 detector run "
        "terminated with CUDA OOM and reported no endpoint. The A gate has "
        "therefore NOT been causally ruled out as a cause of Proposed-v1's "
        "failure."
    ),
    (
        "The CPU candidate-side diagnostics that informed Proposed-v2 are "
        "supporting development evidence only. They are oracle counts over an "
        "already-committed pool, not a detector result."
    ),
    (
        "The research_* arms are the research plan's own equation, "
        "s(x) = U + lambda*D + gamma*w*coh, with the 2026-08-25 redesigns of D "
        "and coh. Their design reads no V1 endpoint: lambda and gamma are the "
        "values owl.scoring froze in 2026-08, min_samples comes from the answer "
        "budget and eps from the candidate geometry. They are pre-registered by "
        "docs/full_owod_v2_protocol.md and are nonetheless reported as "
        "development-seed-informed, because they were added after V1 seed-0 "
        "numbers existed and that is the more conservative claim."
    ),
)

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


def chain(n_tasks: int | None = None) -> tuple[protocol.Task, ...]:
    """The task chain: t1 anchor, then one new class per task.

    ``n_tasks`` defaults to :data:`N_TASKS`, the frozen Benchmark V1 length, so
    every existing caller is unchanged. It is a parameter because the
    2026-08-25 consultation asked for the protocol to be measured through
    ``t1 -> ... -> t10`` and ``owl.protocol.build_chain`` has always supported
    it; what was missing was a way to say so without editing a frozen constant.

    A chain of a different length is a **different experiment**, not a longer
    version of this one: it declares more classes, so
    :func:`declared_classes` differs, so the shared evaluation split differs,
    so every metric is measured on a different image set. Callers that mix the
    two are comparing across measurements.
    """

    return protocol.build_chain(N_TASKS if n_tasks is None else int(n_tasks))


def declared_classes(n_tasks: int | None = None) -> tuple[str, ...]:
    """The classes the chain introduces — what the shared eval split is built on."""

    return tuple(task.new_class for task in chain(n_tasks)[1:] if task.new_class)


def tail_band(task: protocol.Task, groups: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The tail classes known after ``task``. One at t2, three at t4."""

    groups = protocol.load_groups() if groups is None else groups
    return tuple(name for name in task.known_classes if groups.get(name) == "tail")


# ------------------------------------------------------------------ configs ---


def cycle_config(
    arm: str, seed: int, *, n_tasks: int | None = None,
    answer_budget: int | None = None,
    acquisition_batch_size: int | None = None,
    annotation_policy: str | None = None,
    ignore_mechanism: str | None = None,
    replay_mode: str | None = None,
    replay_refresh: str | None = None,
) -> runner.CycleConfig:
    """The frozen :class:`owl.runner.CycleConfig` for one trajectory.

    Everything except the chain length is frozen for V1. ``n_tasks`` defaults to
    :data:`N_TASKS`; passing another value runs the same protocol over a longer
    chain and is only meaningful into a separate results directory with its own
    evaluation split.

    **The V2 axes.** Every keyword below defaults to ``None`` and ``None`` means
    "the V1 frozen value", so a call that passes only ``arm`` and ``seed``
    produces byte-identically the configuration every committed trajectory ran.
    Naming one opens the axis ``docs/full_owod_v2_protocol.md`` declares:

    ``annotation_policy`` / ``ignore_mechanism``
        the per-box policy of :mod:`owl.supervision`, written as filtered
        annotations rather than mapped onto PROB's two-valued
        ``--supervision-mode``.
    ``replay_mode`` / ``replay_refresh``
        the ``m_c ∝ n_c**alpha`` vocabulary of :data:`owl.replay.MODES` and
        :data:`owl.replay.REFRESH`.
    ``acquisition_batch_size``
        answers per mini-round. ``rounds = ceil(budget / batch)``, which is how
        the consultation's "600 at once, or the best 100 then recompute" is
        expressed. Refuses a batch larger than the budget rather than silently
        collapsing to one round.
    ``answer_budget``
        the per-task oracle budget. It has to move with the annotation policy,
        because the *price of an image* is a property of the policy — see
        :func:`owl.active_selection.budget.cost_function`.
    """

    if arm not in arm_registry.ARMS:
        raise BenchmarkError(
            f"Unknown arm {arm!r}; registered: {sorted(arm_registry.ARMS)}."
        )
    if seed not in SEEDS:
        raise BenchmarkError(f"seed={seed} is not one of the declared seeds {SEEDS}.")
    length = N_TASKS if n_tasks is None else int(n_tasks)
    if length < 2:
        raise BenchmarkError(f"a chain needs the anchor and at least one task, got {length}")
    budget = ANSWER_BUDGET_PER_TASK if answer_budget is None else int(answer_budget)
    if budget < 1:
        raise BenchmarkError(f"answer_budget must be positive, got {budget}")

    rounds = ROUNDS_PER_TASK
    if acquisition_batch_size is not None:
        batch = int(acquisition_batch_size)
        if batch < 1 or batch > budget:
            raise BenchmarkError(
                f"acquisition_batch_size={batch} must be between 1 and the "
                f"answer budget {budget}. A batch larger than the budget would "
                "silently collapse to one round, which is a different "
                "experiment wearing an iterative name.")
        rounds = -(-budget // batch)          # ceil

    replay_arm = REPLAY_ARM
    if replay_mode is not None:
        replay_arm, _ = replay.resolve_mode(replay_mode)
    reallocate = (
        False if replay_refresh is None else replay.resolve_refresh(replay_refresh))

    if annotation_policy is not None and annotation_policy not in supervision.POLICIES:
        raise BenchmarkError(
            f"annotation_policy={annotation_policy!r}; expected one of "
            f"{supervision.POLICIES}")
    if ignore_mechanism is not None and annotation_policy is None:
        raise BenchmarkError(
            "ignore_mechanism was given without an annotation_policy, so it "
            "would describe a mechanism nothing uses.")

    return runner.CycleConfig(
        n_tasks=length,
        budget_per_task=budget,
        budget_unit="answers",
        rounds_per_task=rounds,
        candidate_images_per_task=CANDIDATE_IMAGES_PER_TASK,
        proposals_per_image=PROPOSALS_PER_IMAGE,
        arm=arm,
        labelling_policy=LABELLING_POLICY,
        annotation_policy=annotation_policy,
        ignore_mechanism=ignore_mechanism,
        replay_arm=replay_arm,
        replay_reallocate=reallocate,
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


def reference_blocks(
    task_dir: Path, *, task_index: int, run_id: str | None = None
) -> list[np.ndarray]:
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
        try:
            payload = np.load(path, allow_pickle=False)
            features = np.asarray(payload["features"], dtype=np.float32)
            stamp = str(payload["run_id"]) if "run_id" in payload.files else None
        except Exception as error:
            raise BenchmarkError(
                f"{path} is unreadable ({error}). It is an earlier task's "
                "contribution to this arm's semantic reference, so continuing "
                "would select against an incomplete reference. Re-run the arm; "
                "every task rewrites its own block."
            ) from error
        if run_id is not None and stamp != run_id:
            # Written by a different process. The caller walks the chain from
            # its first task, so every block it legitimately needs is rewritten
            # in *this* run — a block from another run is a leftover, and using
            # it would let an interrupted trajectory contaminate a later task
            # silently rather than loudly.
            print(f"  [reference] ignoring {path.parent.name}/{path.name} from an "
                  f"earlier run ({stamp}); this run rewrites it", flush=True)
            continue
        blocks.append(features)
    return blocks


def make_selector(
    arm: str,
    *,
    candidate_index: Mapping[str, Mapping[str, int]],
    jpeg_dir: str | Path,
    ref_t1: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 128,
    annotation_policy: str = LABELLING_POLICY,
    rounds: int = ROUNDS_PER_TASK,
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
    # The price of an image belongs to the annotation policy. `LABELLING_POLICY`
    # is the default, so every committed trajectory is charged exactly what it
    # was charged before.
    cost_of = annotation_budget.cost_function(candidate_index, annotation_policy)
    jpeg_dir = Path(jpeg_dir)
    wants_ref_t1 = spec.needs_semantic and spec.reference_scope == "labelled"
    if wants_ref_t1 and ref_t1 is None:
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
            # Only the rows this arm ranks are embedded, and *the same function*
            # decides that here and inside the traversal — computing the subset
            # twice is how a feature matrix comes to describe a different
            # population from the one being selected over. The cache is keyed on
            # a fingerprint of exactly these rows, so no two arms can reuse each
            # other's file.
            ranked = arm_registry.ranked_positions(arm, pool)
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
            # `labelled` measures distance to the balanced task-1 reference plus
            # this trajectory's purchases — what `proposed` and `coreset` were
            # measured with. `trajectory` measures distance to the purchases
            # alone: Proposed-v2 stops using DINOv2 as a distance-to-REF-T1
            # novelty score, which Method V2 already froze as D_NO_GO, and uses
            # it only to tell near-duplicates apart. At t2 that reference is
            # empty, and `coverage.kcenter_greedy` documents what the first pick
            # then is.
            anchor_reference = [reference_for(ref_t1)] if wants_ref_t1 else []
            reference = semantic.stack_reference([
                *anchor_reference,
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
            rounds=int(rounds),
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

        # The candidate-level trail a research arm leaves. Written next to the
        # task rather than folded into the row: it is one line per taken
        # candidate, which is the granularity the consultation asked for
        # ("log D_labeled, D_batch and the final D per candidate") and far too
        # long for a metrics row. The row keeps the aggregates.
        if picked.picks:
            picks_path = task_dir / "candidate_log.csv"
            fields = list(picked.picks[0])
            temporary = picks_path.with_suffix(".csv.part")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                import csv as _csv

                writer = _csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(picked.picks)
            os.replace(temporary, picks_path)

        row = dict(picked.row) | {
            "population_proposals": pool.diagnostics["proposals_after_nms"],
            "population_images": pool.diagnostics["images"],
            "population_admissible": pool.diagnostics["admissible"],
            "population_ranked": int(arm_registry.ranked_positions(arm, pool).size),
            "reference_scope": spec.reference_scope,
        }
        # The anchor positions index this task's `pool`, which the runner never
        # sees. Resolve them to boxes here, where the pool is in scope, so a
        # per-box annotation policy has something to match the oracle against.
        anchor_boxes = (
            np.asarray(pool.candidates.boxes[list(picked.anchors)], dtype=np.float32)
            if picked.anchors
            else np.zeros((0, 4), dtype=np.float32)
        )
        return arm_registry.ArmSelection(
            arm=picked.arm, images=picked.images, anchors=picked.anchors,
            row=row, covered=picked.covered, picks=picked.picks,
            anchor_boxes=anchor_boxes,
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
        "development_seed_informed": list(DEVELOPMENT_SEED_INFORMED),
        "kill_rule": KILL_RULE.as_dict(),
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
    n_tasks: int | None = None,
) -> dict[str, object]:
    """The machine-readable record of one session.

    ``n_tasks`` is the chain length actually run. It matters for provenance and
    not only for tidiness: ``frozen`` records the **protocol's** frozen
    ``n_tasks`` (4, which :func:`check_protocol` pins against the document), so
    without this the manifest of a ten-task session would describe a four-task
    chain and carry V1's experiment name. A session of a different length gets
    its own name and an explicit statement that its numbers are not comparable
    with V1's, because the evaluation split differs.
    """

    length = N_TASKS if n_tasks is None else int(n_tasks)
    return {
        "experiment": ("full_owod_active_benchmark_v1" if length == N_TASKS
                       else f"full_owod_chain_t{length}"),
        "chain_length": length,
        "comparable_with_benchmark_v1": length == N_TASKS,
        "comparability_note": (
            "same chain length and therefore the same shared evaluation split"
            if length == N_TASKS else
            f"a {length}-task chain declares "
            f"{len(declared_classes(length))} classes against V1's "
            f"{len(declared_classes())}, so it is measured on a different "
            "shared evaluation split. These numbers must NOT be placed in the "
            "same table as Benchmark V1's."
        ),
        "dry_run": bool(dry_run),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "frozen": frozen_values(),
        "endpoints": ENDPOINTS.as_dict(),
        "endpoint_statement": ENDPOINTS.statement(),
        "reporting_rules": list(REPORTING),
        "provenance": list(PROVENANCE),
        "development_seed_informed": list(DEVELOPMENT_SEED_INFORMED),
        "kill_rule": KILL_RULE.as_dict(),
        "kill_rule_statement": KILL_RULE.statement(),
        "chain": [
            {"task": task.name, "new_class": task.new_class,
             "known_after": task.n_current, "tail_band": list(tail_band(task))}
            for task in chain(length)
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
