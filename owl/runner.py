"""The cycle. One task: score, ask, label, rehearse, fine-tune, evaluate.

Two ways to run the same configuration.

:func:`simulate`
    CPU, minutes, no dataset. Runs on the committed PROB pass and answers
    **what a score selects**: how many real unknown objects the budget buys, in
    which frequency group, from how many classes, at what oracle cost. This is
    where arms are swept, because a sweep here costs seconds.

:func:`run_chain`
    GPU, hours, the real thing. PROB's weights are updated, PROB's own evaluator
    scores the checkpoint, and the numbers are mAP and U-Recall.

**They answer different questions and the split is not a matter of taste.** The
earlier work checked the frozen-feature surrogate against the real detector and
found it ranks acquisition methods on forgetting in the *reverse* order. So a
simulation may be used to compare what a score selects, and may not be used to
claim one arm forgets less than another. :func:`simulate` refuses to report a
detection metric for that reason.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from owl import clustering, labelling, metrics, proposals, protocol, replay, selection
from owl.bridge import Bridge


@dataclass
class CycleConfig:
    """Everything one run of the chain needs. The notebook edits this and only this."""

    # --- the protocol -----------------------------------------------------
    n_tasks: int = 10
    budget_per_task: int = 600          # regions the oracle is asked about
    rounds_per_task: int = 6            # 1 = one shot, 6 = 6x100 (consultation, point 7)
    candidate_images_per_task: int = 4000
    proposals_per_image: int = 50        # PROB offers 100; the top 50 by its own
                                        # objectness order is what the frozen pool
                                        # keeps, and it halves the export's size

    # --- the four experimental variables ----------------------------------
    arm: str = "prior_consult_batch"    # owl.selection.ARMS
    labelling_policy: str = "known_plus_selected"   # owl.labelling.POLICIES
    replay_arm: str = "tail_favouring"  # owl.replay.ARMS
    replay_reallocate: bool = False     # re-size the memory every task

    # --- training ---------------------------------------------------------
    epochs: int = 5
    learning_rate: float = 2e-4
    batch_size: int = 2
    n_clusters: int = 1600
    seed: int = 0

    #: Write the per-box detections artefact and decompose U-Recall by frequency
    #: group. This is the research plan's headline endpoint — "tail-U-Recall as a
    #: function of oracle cost" — and it cannot come from the aggregate the
    #: evaluator prints. It costs a second forward pass over the evaluation
    #: split, so turning it off halves evaluation and gives up the main result.
    measure_grouped_recall: bool = True

    #: An image the oracle labelled whose objects are all future-task classes
    #: cannot be trained on yet — PROB's split keeps only the classes introduced
    #: so far. The label is still ours: we paid for it. With this on, such an
    #: image is banked and joins the training set at the task where its class
    #: becomes declarable, at no further annotation cost. This is the research
    #: plan's feedback loop applied to the annotation ledger, and turning it off
    #: is what measures what it is worth.
    reuse_deferred_labels: bool = True

    #: How many task checkpoints to keep. Each is 478 MB, so a nine-task chain
    #: writes 4.3 GB and three arms fill a free Drive. Two is the minimum that
    #: keeps a resumed session working: the one a task starts from, and the one
    #: it produced. Set to 0 to keep every checkpoint.
    keep_checkpoints: int = 2

    def describe(self) -> dict[str, object]:
        return asdict(self)

    #: Fields whose value changes what the numbers mean. Everything except
    #: bookkeeping: a run that differs in any of these is a different experiment
    #: and may not reuse another one's cached tasks.
    RESULT_AFFECTING = (
        "n_tasks", "budget_per_task", "rounds_per_task", "candidate_images_per_task",
        "proposals_per_image", "arm", "labelling_policy", "replay_arm",
        "replay_reallocate", "reuse_deferred_labels", "epochs", "learning_rate",
        "batch_size", "n_clusters", "seed",
    )

    def fingerprint(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.RESULT_AFFECTING}


@dataclass
class TaskResult:
    task: str
    new_class: str | None
    selection_row: dict
    annotation_row: dict
    replay_row: dict
    evaluation_row: dict | None = None

    def flat(self) -> dict[str, object]:
        row: dict[str, object] = {"task": self.task, "new_class": self.new_class or "—"}
        row |= self.selection_row
        row |= {f"label_{k}": v for k, v in self.annotation_row.items() if k != "policy"}
        row |= {f"replay_{k}": v for k, v in self.replay_row.items()}
        if self.evaluation_row:
            row |= self.evaluation_row
        return row


# ------------------------------------------------------------- simulation ---


def simulate(
    candidates: proposals.Candidates,
    config: CycleConfig,
    *,
    chain: Sequence[protocol.Task] | None = None,
    partition: clustering.Partition | None = None,
) -> list[TaskResult]:
    """Walk the chain on the committed pool and report what each budget buys.

    No detection metric is produced. What comes out is the composition of the
    annotation: real objects found, split by frequency group, and what the
    oracle was charged.
    """

    chain = chain or protocol.build_chain(config.n_tasks)
    groups = protocol.load_groups()
    oracle = candidates.oracle()
    group_of = np.asarray([groups.get(name, "") for name in oracle.class_name])

    if partition is None:
        partition = clustering.fit(
            candidates.embeddings, n_clusters=config.n_clusters, seed=config.seed
        )

    arm = selection.ARMS[config.arm]
    spent = np.zeros(len(candidates), dtype=bool)
    labelled_embeddings = np.zeros((0, candidates.embeddings.shape[1]), dtype=np.float32)
    memory = replay.Memory((), {}, 0.0, 0)
    results: list[TaskResult] = []

    for task in chain[1:]:
        picked = selection.select(
            candidates, arm,
            budget=config.budget_per_task,
            rounds=config.rounds_per_task,
            labelled_embeddings=labelled_embeddings,
            n_known=task.n_prev,
            exclude=spent,
            partition=partition,
        )
        annotation = labelling.annotate(
            candidates, picked,
            policy=config.labelling_policy,
            known_classes=task.previous_classes,
        )

        index = picked.indices
        is_object = oracle.kind[index] != "background"
        found = group_of[index][is_object]
        target = oracle.class_name[index] == task.new_class

        results.append(
            TaskResult(
                task=task.name,
                new_class=task.new_class,
                selection_row={
                    "asked": len(picked),
                    "objects": int(is_object.sum()),
                    "head": int((found == "head").sum()),
                    "medium": int((found == "medium").sum()),
                    "tail": int((found == "tail").sum()),
                    "classes_seen": int(np.unique(oracle.class_name[index][is_object]).size),
                    "target_instances": int(np.unique(oracle.object_id[index][target]).size),
                },
                annotation_row=annotation.summary()
                | {"half_labelled": labelling.half_labelling_rate(annotation, candidates)},
                replay_row=memory.summary(),
            )
        )

        spent[index] = True
        labelled_embeddings = np.vstack(
            [labelled_embeddings, candidates.embeddings[annotation.labelled]]
        )

    return results


# ----------------------------------------------------------- the real chain ---


def run_chain(
    bridge: Bridge,
    config: CycleConfig,
    *,
    workspace: Path,
    candidate_index: Mapping[str, Mapping[str, int]],
    start_checkpoint: Path,
    test_set: str,
    chain: Sequence[protocol.Task] | None = None,
    time_budget_minutes: float | None = None,
    prepare_images: Callable[[Sequence[str]], Sequence[str]] | None = None,
) -> list[TaskResult]:
    """Run the task chain on the GPU, one checkpoint per task, resumable.

    Every artefact is keyed by ``workspace / task / arm``, and the bridge skips
    any call whose output already exists — so a Colab session that is cut off
    resumes at the task it died on rather than at the beginning.

    ``time_budget_minutes`` stops the chain cleanly before the runtime is lost
    and prints which tasks were not run. Nothing is silently truncated.

    ``prepare_images`` is called with each task's candidate image ids before the
    detector runs, and returns the ids that are actually usable. The detector
    reads JPEGs off disk and dies on the first one that is missing, so something
    has to put them there; on Colab that is a download from COCO, and only the
    images a task actually offers the selector are worth fetching. Returning a
    shorter list is how an unavailable image is dropped instead of killing the
    run. It is not called when the task's detector pass is already cached.
    """

    chain = chain or protocol.build_chain(config.n_tasks)
    groups = protocol.load_groups()
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # Resuming is keyed on files existing, which is fast and which silently makes
    # two different experiments into one. A smoke run and a real run share this
    # workspace, so the real run reused the smoke run's checkpoints and metrics
    # for the tasks the smoke run had reached — and the table came out with two
    # tasks measured on a sixteen-image evaluation split and the rest on a
    # fourteen-hundred-image one, which reads as a twenty-nine-point swing in
    # forgetting that never happened. So the stored configuration is compared,
    # and a run that differs is refused rather than blended.
    stamp = workspace / "config.json"
    if stamp.exists():
        stored = json.loads(stamp.read_text(encoding="utf-8"))
        current = config.fingerprint()
        differing = {
            name: (stored.get(name), value)
            for name, value in current.items()
            if name in stored and stored[name] != value
        }
        if differing:
            lines = "\n".join(
                f"    {name}: stored {was!r}, now {now!r}"
                for name, (was, now) in sorted(differing.items())
            )
            raise RuntimeError(
                f"{workspace} holds results from a different configuration:\n{lines}\n"
                "Resuming would mix them into one table. Either delete that "
                f"directory —\n    rm -rf '{workspace}'\n"
                "— or point `workspace` somewhere else."
            )
    stamp.write_text(json.dumps(config.fingerprint(), indent=2), encoding="utf-8")

    arm = selection.ARMS[config.arm]
    replay_spec = dict(replay.ARMS[config.replay_arm])
    replay_selector = replay_spec.pop("selector", "greedy")

    generator = np.random.default_rng(config.seed)
    all_images = np.asarray(sorted(candidate_index), dtype=object)
    used_images: set[str] = set()
    labelled_history: dict[str, list[str]] = {}
    #: Every image the oracle has ever answered for, and whether it has been
    #: trained on yet. An entry that is not yet trainable is not lost.
    ledger: set[str] = set()
    trained_on: set[str] = set()
    memory = replay.Memory((), {}, 0.0, 0)
    checkpoint = Path(start_checkpoint)
    results: list[TaskResult] = []
    written: list[Path] = []
    previous_baseline: float | None = None
    elapsed = 0.0

    for task in chain[1:]:
        task_dir = workspace / f"{task.name}_{config.arm}"
        task_dir.mkdir(parents=True, exist_ok=True)
        state_path = task_dir / "state.json"

        # ---- 0. already finished? ----------------------------------------
        #
        # A completed task is one that has metrics, not one that still has its
        # checkpoint: checkpoints are pruned to save Drive, so keying the skip on
        # them makes a resumed run retrain work it had already paid for. The
        # accumulated state — what has been opened, banked, trained on, and held
        # in memory — is restored from disk, because a chain resumed without it
        # would select images it had already bought and rebuild a memory it had
        # already allocated.
        if state_path.exists() and (task_dir / "metrics.json").exists():
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            used_images.update(saved["used_images"])
            ledger.update(saved["ledger"])
            trained_on.update(saved["trained_on"])
            labelled_history[task.name] = saved["labelled"]
            memory = replay.Memory(
                image_ids=tuple(saved["memory"]), per_class=saved["memory_per_class"],
                alpha=saved["memory_alpha"], total=saved["memory_total"],
            )
            previous_baseline = saved["known_map50"]
            restored = Path(saved["checkpoint"])
            if restored.exists():
                checkpoint = restored
            results.append(TaskResult(
                task=task.name, new_class=task.new_class,
                selection_row=saved["selection_row"],
                annotation_row=saved["annotation_row"],
                replay_row=saved["replay_row"],
                evaluation_row=saved["evaluation_row"],
            ))
            print(f"  [{task.name}] already done; restored from {state_path.name}")
            continue

        if time_budget_minutes is not None and elapsed >= time_budget_minutes:
            remaining = [t.name for t in chain[chain.index(task):]]
            print(f"Stopping cleanly: {elapsed:.0f} min used. Not run: {remaining}")
            break

        known_now = set(task.known_classes)

        # ---- 1. this task's candidate images ----------------------------
        pool = np.asarray([i for i in all_images if i not in used_images], dtype=object)
        take = min(config.candidate_images_per_task, pool.size)
        candidate_ids = [
            str(v) for v in generator.choice(pool, size=take, replace=False)
        ]

        # ---- 2. make sure the images exist, then one detector pass -------
        proposals_path = task_dir / "proposals.npz"
        if prepare_images is not None and not proposals_path.exists():
            available = [str(value) for value in prepare_images(candidate_ids)]
            dropped = len(candidate_ids) - len(available)
            if dropped:
                print(f"  [{task.name}] {dropped} of {len(candidate_ids)} candidate "
                      f"images could not be fetched; dropped")
            candidate_ids = available
        if not candidate_ids:
            raise RuntimeError(
                f"{task.name} has no usable candidate images. The detector reads JPEGs "
                "off disk, so either prepare_images failed for all of them or the "
                "dataset root is wrong."
            )

        export = bridge.predict(
            candidate_ids,
            checkpoint=checkpoint,
            output=proposals_path,
            n_prev=0, n_current=task.n_prev,
            max_proposals_per_image=config.proposals_per_image,
        )
        candidates = proposals.from_predict(export)

        # ---- 3. spend the budget ----------------------------------------
        # Cluster once per task, not once per round. The partition depends only on
        # the pool's geometry, which does not move while the budget is spent, and
        # a k-means over 200,000 proposals costs minutes — at six rounds that is
        # the difference between one evening and three.
        task_partition = clustering.fit(
            candidates.embeddings, method=arm.cluster_method,
            n_clusters=config.n_clusters, seed=config.seed,
        )
        picked = selection.select(
            candidates, arm,
            budget=config.budget_per_task,
            rounds=config.rounds_per_task,
            n_known=task.n_prev,
            partition=task_partition,
        )
        opened = [str(v) for v in picked.images(candidates)]
        used_images.update(opened)

        # ---- 4. what the oracle's answers are worth ----------------------
        #
        # PROB's fine-tuning split keeps only the classes introduced so far
        # (`remove_unknown_instances`: category_id in range(0, prev + curr)). An
        # image whose objects are all future-task classes therefore arrives with
        # zero boxes, and the collate function fails on it rather than skipping
        # it — `size of tensor a (0) must match the size of tensor b (4)`.
        #
        # So such an image cannot be trained on. That is not only a loader
        # constraint, it is a result: the oracle was paid for it and it yields no
        # supervision at this task, and how often an arm does that is worth
        # knowing. Both numbers go in the row.
        def usable(image: str, known: frozenset = frozenset(known_now)) -> bool:
            """Would PROB see at least one box on this image at this task?"""
            return any(name in known for name in candidate_index.get(image, {}))

        ledger.update(opened)
        trainable = [image for image in opened if usable(image)]
        barren = len(opened) - len(trainable)

        deferred: list[str] = []
        if config.reuse_deferred_labels:
            # labels paid for at an earlier task whose class is declarable now
            deferred = sorted(
                image for image in ledger - trained_on - set(opened) if usable(image)
            )
            trainable = list(dict.fromkeys([*trainable, *deferred]))

        if barren:
            print(f"  [{task.name}] {barren} of {len(opened)} opened images hold no "
                  f"class known after this task; banked for a later one")
        if deferred:
            print(f"  [{task.name}] {len(deferred)} images banked at earlier tasks "
                  "became trainable and cost nothing extra")
        if len(trainable) < config.batch_size:
            raise RuntimeError(
                f"{task.name} kept only {len(trainable)} trainable images of "
                f"{len(opened)} opened, and PROB's loader drops the last partial "
                f"batch, so it needs at least {config.batch_size}. Raise "
                "budget_per_task, or lower batch_size."
            )

        found = sum(candidate_index.get(image, {}).get(task.new_class, 0) for image in opened)
        with_target = sum(1 for image in opened if task.new_class in candidate_index.get(image, {}))
        labelled_history[task.name] = trainable
        trained_on.update(trainable)

        # ---- 5. the exemplar memory --------------------------------------
        if replay_spec["total"] > 0:
            history = {
                image: candidate_index.get(image, {})
                for images in labelled_history.values() for image in images
            }
            fresh = replay.build(
                history, task.previous_classes,
                total=replay_spec["total"], alpha=replay_spec["alpha"],
                seed=config.seed, selector=replay_selector,
            )
            carried = replay.carry_forward(
                memory, fresh.image_ids, reallocate=config.replay_reallocate
            )
            # same loader constraint: a replay image with no currently-known
            # object arrives empty and fails the collate
            carried = tuple(
                image for image in carried
                if any(name in known_now for name in candidate_index.get(image, {}))
            )
            memory = replay.Memory(
                image_ids=carried,
                per_class=fresh.per_class, alpha=fresh.alpha, total=fresh.total,
            )

        # ---- 6. fine-tune -------------------------------------------------
        supervision = "train" if config.labelling_policy == "box_only" else "ft"
        checkpoint = bridge.train(
            trainable,
            previous_checkpoint=checkpoint,
            output_checkpoint=task_dir / "checkpoint.pth",
            output_dir=task_dir / "train",
            n_prev=task.n_prev, n_current=task.n_new,
            test_set=test_set,
            replay_ids=memory.image_ids,
            supervision_mode=supervision,
            epochs=config.epochs, learning_rate=config.learning_rate,
            batch_size=config.batch_size,
        )

        # ---- 7. score it ---------------------------------------------------
        metrics_path = bridge.evaluate(
            checkpoint=checkpoint, test_set=test_set,
            output=task_dir / "metrics.json",
            n_prev=task.n_prev, n_current=task.n_new,
            detections=config.measure_grouped_recall,
        )
        evaluation = metrics.from_bridge_metrics(metrics_path)
        # The head/medium/tail split is the research plan's distinguishing form
        # of evaluation, and it needs per-class AP50 — which lives in the metrics
        # file's coco_eval_bbox vector. See owl.metrics.per_class_ap50.
        row = metrics.task_row(
            evaluation, task=task.name, new_class=task.new_class,
            previous_baseline=previous_baseline,
            groups=metrics.group_membership(task.known_classes, groups),
        )
        row["exchange_rate"] = metrics.exchange_rate(row)

        # The plan's headline endpoint: how much of the *tail* the detector still
        # finds as unknown, at this point on the oracle-cost axis.
        artefact = json.loads(metrics_path.read_text(encoding="utf-8")).get("detections_path")
        if artefact and Path(artefact).exists():
            by_group = metrics.unknown_recall_by_group(
                Path(artefact), known_classes=task.known_classes, groups=groups,
            )
            for name in ("head", "medium", "tail", "all"):
                row[f"U_Recall_{name}"] = by_group[name]["recall"]
                row[f"unknown_objects_{name}"] = by_group[name]["objects"]
            row["oracle_cost_so_far"] = (len(results) + 1) * config.budget_per_task
        previous_baseline = evaluation.known_map50

        result = TaskResult(
                task=task.name, new_class=task.new_class,
                selection_row={
                    "asked": len(picked),
                    "images_opened": len(opened),
                    "images_trainable": len(trainable),
                    "images_no_supervision": barren,
                    "images_from_earlier_tasks": len(deferred),
                    "target_objects_in_images": found,
                    "images_with_target": with_target,
                },
                annotation_row={"policy": config.labelling_policy,
                                "supervision": supervision},
                replay_row=memory.summary(),
                evaluation_row=row,
        )
        results.append(result)

        state_path.write_text(json.dumps({
            "used_images": sorted(used_images),
            "ledger": sorted(ledger),
            "trained_on": sorted(trained_on),
            "labelled": labelled_history[task.name],
            "memory": list(memory.image_ids),
            "memory_per_class": dict(memory.per_class),
            "memory_alpha": memory.alpha,
            "memory_total": memory.total,
            "known_map50": evaluation.known_map50,
            "checkpoint": str(checkpoint),
            "selection_row": result.selection_row,
            "annotation_row": result.annotation_row,
            "replay_row": result.replay_row,
            "evaluation_row": result.evaluation_row,
        }, indent=2), encoding="utf-8")

        elapsed = bridge.cost_report()["total"]
        _write_rows(results, workspace / f"results_{config.arm}.csv")
        written.append(Path(checkpoint))
        freed = _prune_checkpoints(written, config.keep_checkpoints)
        print(f"  [{task.name}] {elapsed:.0f} min spent so far; "
              f"{len(chain) - 1 - len(results)} tasks left"
              + (f"; freed {freed / 1e9:.1f} GB of checkpoints" if freed else ""))

    return results


def _prune_checkpoints(written: list[Path], keep: int) -> int:
    """Delete all but the newest ``keep`` checkpoints. Returns bytes freed.

    A resumed run needs the checkpoint a task starts from and the one it wrote,
    and nothing older — the metrics of every earlier task are already on disk.
    Keeping all of them costs 478 MB each, which is what fills a free Drive
    somewhere in the third arm and stops the chain for a reason that has nothing
    to do with the research.
    """

    if keep <= 0:
        return 0
    freed = 0
    for path in written[:-keep]:
        if path.exists():
            freed += path.stat().st_size
            path.unlink()
    return freed


def _write_rows(results: Sequence[TaskResult], path: Path) -> None:
    import csv

    rows = [r.flat() for r in results]
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: Sequence[Mapping[str, object]], digits: int = 2) -> str:
    """Plain-text table. The notebook prints everything through this."""

    rows = [dict(row) for row in rows]
    if not rows:
        return "(empty)"
    columns = list(dict.fromkeys(key for row in rows for key in row))

    def show(value: object) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    widths = {c: max(len(c), *(len(show(r.get(c))) for r in rows)) for c in columns}
    lines = ["  ".join(c.ljust(widths[c]) for c in columns)]
    lines.append("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        lines.append("  ".join(show(row.get(c)).ljust(widths[c]) for c in columns))
    return "\n".join(lines)
