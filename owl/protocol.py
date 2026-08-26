"""The task chain: which class becomes known when, and how classes are grouped.

One concept: the incremental protocol. Task 1 is PROB's published S-OWODB
checkpoint, which knows 19 classes. Every later task declares exactly one new
class, so a budget of a few hundred regions is spent on one class instead of
being split across twenty. That single change is what makes new-class learning
measurable at an affordable annotation budget; see docs/why_one_class.md.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- S-OWODB ---

TASK1: tuple[str, ...] = (
    "aeroplane", "bicycle", "bird", "boat", "bus", "car",
    "cat", "cow", "dog", "horse", "motorbike", "sheep", "train",
    "elephant", "bear", "zebra", "giraffe", "truck", "person",
)

TASK2: tuple[str, ...] = (
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "chair", "diningtable",
    "pottedplant", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "microwave", "oven", "toaster", "sink",
    "refrigerator", "bed", "toilet", "sofa",
)

TASK3: tuple[str, ...] = (
    "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
)

TASK4: tuple[str, ...] = (
    "laptop", "mouse", "remote", "keyboard", "cell phone", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "tvmonitor", "bottle",
)

#: The evaluator's class order. PROB indexes classes by position in this tuple,
#: so a class can only be declared known once every class before it is known.
CLASS_ORDER: tuple[str, ...] = TASK1 + TASK2 + TASK3 + TASK4

#: How many classes PROB's t1.pth was trained on.
N_TASK1 = len(TASK1)


class ProtocolError(ValueError):
    """Raised when a requested chain does not fit the evaluator's class order."""


# ------------------------------------------------------------- task chain ---


@dataclass(frozen=True)
class Task:
    """One step of the chain.

    ``index`` is 1-based and matches the paper's ``task_k`` naming: task 1 is
    the pretrained checkpoint, task 2 is the first thing we train.
    """

    index: int
    new_class: str | None          # None for task 1, which declares nothing
    known_classes: tuple[str, ...]  # everything known *after* this task ran
    n_prev: int                     # classes known before this task
    n_current: int                  # classes known after it

    @property
    def name(self) -> str:
        return f"t{self.index}"

    @property
    def previous_classes(self) -> tuple[str, ...]:
        """The classes whose mAP measures forgetting."""
        return self.known_classes[: self.n_prev]

    @property
    def is_anchor(self) -> bool:
        return self.new_class is None


def build_chain(n_tasks: int = 10, *, start: int = N_TASK1) -> tuple[Task, ...]:
    """Build a ``n_tasks``-long chain that declares one new class per task.

    ``start`` is the position in :data:`CLASS_ORDER` of the first class to
    declare. It defaults to 19, so the chain begins exactly where PROB's
    published t1 checkpoint ends and no class is skipped.
    """

    if n_tasks < 1:
        raise ProtocolError("A chain needs at least the anchor task.")
    new_classes = CLASS_ORDER[start : start + n_tasks - 1]
    if len(new_classes) < n_tasks - 1:
        raise ProtocolError(
            f"CLASS_ORDER holds {len(CLASS_ORDER) - start} classes after position "
            f"{start}, which is not enough for {n_tasks - 1} incremental tasks."
        )

    tasks = [
        Task(
            index=1,
            new_class=None,
            known_classes=CLASS_ORDER[:start],
            n_prev=0,
            n_current=start,
        )
    ]
    for offset, class_name in enumerate(new_classes):
        n_prev = start + offset
        tasks.append(
            Task(
                index=offset + 2,
                new_class=class_name,
                known_classes=CLASS_ORDER[: n_prev + 1],
                n_prev=n_prev,
                n_current=n_prev + 1,
            )
        )
    return tuple(tasks)


def unknown_classes(task: Task) -> tuple[str, ...]:
    """Classes that still count as unknown after ``task`` — the U-Recall set."""

    return CLASS_ORDER[task.n_current :]


# ------------------------------------------------------ head / medium / tail ---

#: Where the frequency grouping is read from. The file is a copy of a measured
#: count over the benchmark's own annotations, not a guess.
GROUPS_PATH = Path(__file__).resolve().parent.parent / "data" / "reference" / "class_groups.csv"


def load_groups(path: str | Path = GROUPS_PATH) -> Mapping[str, str]:
    """class name -> 'head' | 'medium' | 'tail', by training-set object count."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["class_name"]: row["group"] for row in csv.DictReader(handle)}


def load_train_counts(path: str | Path = GROUPS_PATH) -> Mapping[str, int]:
    """class name -> number of annotated objects in the benchmark's train split."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["class_name"]: int(row["train_objects"]) for row in csv.DictReader(handle)}


def group_of(class_names: Sequence[str], groups: Mapping[str, str] | None = None) -> list[str]:
    groups = load_groups() if groups is None else groups
    return [groups.get(name, "unknown") for name in class_names]


def describe_chain(tasks: Sequence[Task]) -> list[dict[str, object]]:
    """One row per task, ready to print. Used by the notebook's first table."""

    groups = load_groups()
    counts = load_train_counts()
    rows: list[dict[str, object]] = []
    for task in tasks:
        rows.append(
            {
                "task": task.name,
                "new_class": task.new_class or "— (pretrained anchor)",
                "group": groups.get(task.new_class, "—") if task.new_class else "—",
                "train_objects": counts.get(task.new_class, 0) if task.new_class else 0,
                "known_after": task.n_current,
                "unknown_after": len(CLASS_ORDER) - task.n_current,
            }
        )
    return rows
