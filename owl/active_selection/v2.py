"""Full OWOD Chain V2: the frozen protocol as values, and the phase presets.

Protocol document: ``docs/full_owod_v2_protocol.md``. Everything the V2
notebook declares is compared against the tagged JSON block in that document
**as values**, not as prose — a Method V3 overnight session was lost to an
assertion that matched a rendered float against an English sentence, and
:mod:`owl.active_selection.benchmark` already carries the same construction for
V1.

Why a separate module rather than more constants in ``benchmark``: V1's numbers
are frozen and results depend on them, so nothing here may be able to move one.
This module reads the V2 document and hands back a configuration; it does not
redefine a V1 value, and :func:`phase` refuses an arm that is not registered.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from owl import replay, supervision
from owl.active_selection import arms as arm_registry
from owl.active_selection import research_score

ROOT = Path(__file__).resolve().parent.parent.parent
PROTOCOL_PATH = ROOT / "docs" / "full_owod_v2_protocol.md"

_BLOCK = re.compile(
    r"<!--\s*FROZEN-V2-BEGIN\s*-->\s*```json\s*(?P<body>.*?)```\s*<!--\s*FROZEN-V2-END\s*-->",
    re.DOTALL,
)


class V2Error(ValueError):
    """Raised when the V2 protocol and a configuration disagree."""


def frozen(path: str | Path | None = None) -> dict[str, object]:
    """The tagged JSON block of the protocol document. One source of truth."""

    path = Path(path or PROTOCOL_PATH)
    match = _BLOCK.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise V2Error(
            f"{path} has no FROZEN-V2 block. The protocol's values live in that "
            "block and nowhere else is authoritative.")
    return json.loads(match.group("body"))


@dataclass(frozen=True)
class Configuration:
    """One V2 run, in the vocabulary the notebook's parameter cell uses."""

    task_end: int = 10
    seeds: tuple[int, ...] = (0, 1, 2)
    annotation_policy: str = "known_plus_selected_ignore_rest"
    ignore_mechanism: str = "pixel_suppression"
    replay_mode: str = "uniform"
    replay_refresh: str = "fixed"
    coherence_mode: str = "dbscan_binary"
    diversity_mode: str = "combined"
    acquisition_batch_size: int = 100
    answer_budget: int = 300
    arms: tuple[str, ...] = ()
    phase: str = "B"

    def __post_init__(self) -> None:
        if self.annotation_policy not in supervision.POLICIES:
            raise V2Error(
                f"annotation_policy={self.annotation_policy!r}; expected one of "
                f"{supervision.POLICIES}")
        if self.ignore_mechanism not in supervision.IGNORE_MECHANISMS:
            raise V2Error(
                f"ignore_mechanism={self.ignore_mechanism!r}; expected one of "
                f"{supervision.IGNORE_MECHANISMS}")
        if self.replay_mode not in replay.MODES:
            raise V2Error(
                f"replay_mode={self.replay_mode!r}; expected one of "
                f"{sorted(replay.MODES)}")
        if self.replay_refresh not in replay.REFRESH:
            raise V2Error(
                f"replay_refresh={self.replay_refresh!r}; expected one of "
                f"{sorted(replay.REFRESH)}")
        if self.coherence_mode not in research_score.COHERENCE_MODES:
            raise V2Error(
                f"coherence_mode={self.coherence_mode!r}; expected one of "
                f"{research_score.COHERENCE_MODES}")
        if self.diversity_mode not in research_score.DIVERSITY_MODES:
            raise V2Error(
                f"diversity_mode={self.diversity_mode!r}; expected one of "
                f"{research_score.DIVERSITY_MODES}")
        unknown = [a for a in self.arms if a not in arm_registry.ARMS]
        if unknown:
            raise V2Error(f"unregistered arms: {unknown}")
        if self.task_end < 2:
            raise V2Error(f"task_end={self.task_end}; a chain needs the anchor and one task")
        if not self.seeds:
            raise V2Error("at least one seed is required")

    def arm_order(self) -> tuple[str, ...]:
        """The arms in the registry's pre-declared execution order.

        Never in the order the notebook happened to list them: a session that
        runs out of runtime must complete a prefix that is fixed in advance, so
        which arms survive a short session cannot be chosen after the fact.
        """

        wanted = set(self.arms)
        return tuple(a for a in arm_registry.ORDER if a in wanted)

    def score_mismatch(self) -> tuple[str, ...]:
        """Arms whose frozen ``ScoreSpec`` contradicts the notebook's modes.

        The notebook names ``COHERENCE_MODE`` and ``DIVERSITY_MODE`` because the
        brief asks for a human-readable config cell, but the *authoritative*
        value is the arm's own registered :class:`ScoreSpec` — an arm is a frozen
        object and a notebook variable may not silently redefine one. So the two
        are compared and a disagreement is named rather than resolved.
        """

        problems: list[str] = []
        for name in self.arm_order():
            spec = arm_registry.ARMS[name].score_spec
            if spec is None or name != "research_v2":
                # Only the primary arm is expected to match the cell; the
                # ablations exist precisely to differ from it.
                continue
            if spec.coherence_mode != self.coherence_mode:
                problems.append(
                    f"{name} is registered with coherence_mode="
                    f"{spec.coherence_mode!r} but the notebook says "
                    f"{self.coherence_mode!r}")
            if spec.diversity_mode != self.diversity_mode:
                problems.append(
                    f"{name} is registered with diversity_mode="
                    f"{spec.diversity_mode!r} but the notebook says "
                    f"{self.diversity_mode!r}")
        return tuple(problems)

    def launcher_flags(self) -> list[str]:
        """The V2 flags for ``tools/run_full_owod_benchmark.py``."""

        return [
            "--n-tasks", str(self.task_end),
            "--seeds", *[str(s) for s in self.seeds],
            "--arms", *self.arm_order(),
            "--annotation-policy", self.annotation_policy,
            "--ignore-mechanism", self.ignore_mechanism,
            "--replay-mode", self.replay_mode,
            "--replay-refresh", self.replay_refresh,
            "--acquisition-batch-size", str(self.acquisition_batch_size),
            "--answer-budget", str(self.answer_budget),
        ]

    def as_dict(self) -> dict[str, object]:
        return asdict(self) | {"arm_order": list(self.arm_order())}


#: The two phases of §13, as presets. Phase A validates the machinery on a short
#: chain; Phase B is the full comparison. The arm sets are read from the protocol
#: document, so neither can be edited here without editing the pre-registration.
PHASES: tuple[str, ...] = ("A", "B")


def phase(
    name: str, path: str | Path | None = None, **overrides: object
) -> Configuration:
    """The frozen configuration for phase ``A`` or ``B``.

    ``overrides`` exist for a debug run and for the dry run, which needs a
    two-task chain and one seed. They are recorded in the manifest like anything
    else, so a run that used one cannot be reported as the protocol's.
    """

    if name not in PHASES:
        raise V2Error(f"phase={name!r}; expected one of {PHASES}")
    values = frozen(path)

    if name == "A":
        arms = tuple(values["phase_a_arms"])
        base = Configuration(
            task_end=int(values["phase_a_tasks"]),
            seeds=(0,),
            arms=arms,
            phase="A",
            annotation_policy=str(values["annotation_policy"]),
            ignore_mechanism=str(values["ignore_mechanism"]),
            replay_mode=str(values["replay_mode"]),
            replay_refresh=str(values["replay_refresh"]),
            coherence_mode=str(values["coherence_mode"]),
            diversity_mode=str(values["diversity_mode"]),
            acquisition_batch_size=int(values["acquisition_batch_size"]),
            answer_budget=300,
        )
    else:
        base = Configuration(
            task_end=int(values["n_tasks"]),
            seeds=tuple(int(s) for s in values["seeds"]),
            arms=tuple(values["phase_b_arms"]),
            phase="B",
            annotation_policy=str(values["annotation_policy"]),
            ignore_mechanism=str(values["ignore_mechanism"]),
            replay_mode=str(values["replay_mode"]),
            replay_refresh=str(values["replay_refresh"]),
            coherence_mode=str(values["coherence_mode"]),
            diversity_mode=str(values["diversity_mode"]),
            acquisition_batch_size=int(values["acquisition_batch_size"]),
            answer_budget=300,
        )
    if not overrides:
        return base
    from dataclasses import replace as _replace

    return _replace(base, **overrides)          # type: ignore[arg-type]


@dataclass(frozen=True)
class Agreement:
    """Whether a notebook's parameter cell matches the frozen protocol."""

    agrees: bool
    checked: int
    disagreements: tuple[str, ...] = field(default_factory=tuple)
    path: str = ""

    def statement(self) -> str:
        if self.agrees:
            return (f"{self.checked} V2 fields agree with "
                    f"{Path(self.path).name}")
        return (f"{len(self.disagreements)} of {self.checked} V2 fields "
                f"disagree with {Path(self.path).name}: "
                + "; ".join(self.disagreements))


def check(
    configuration: Configuration | Mapping[str, object],
    path: str | Path | None = None,
) -> Agreement:
    """Compare a configuration against the protocol's own values.

    Only the fields the document freezes are compared, and the comparison is on
    values. A field the document does not freeze — ``arms`` for a debug subset,
    ``seeds`` for a development run — is not checked here; the manifest records
    it instead.
    """

    path = Path(path or PROTOCOL_PATH)
    values = frozen(path)
    given = (
        configuration.as_dict()
        if isinstance(configuration, Configuration)
        else dict(configuration)
    )

    comparable = {
        "annotation_policy": "annotation_policy",
        "ignore_mechanism": "ignore_mechanism",
        "replay_mode": "replay_mode",
        "replay_refresh": "replay_refresh",
        "coherence_mode": "coherence_mode",
        "diversity_mode": "diversity_mode",
        "acquisition_batch_size": "acquisition_batch_size",
    }
    disagreements: list[str] = []
    for field_name, frozen_key in comparable.items():
        if field_name not in given:
            continue
        want = values[frozen_key]
        got = given[field_name]
        if type(want) is int or type(got) is int:
            same = int(want) == int(got)          # type: ignore[arg-type]
        else:
            same = str(want) == str(got)
        if not same:
            disagreements.append(f"{field_name}: notebook {got!r} vs protocol {want!r}")

    # Phase B is the only configuration the document fixes the chain length and
    # the seed set for; Phase A is explicitly shorter and single-seed.
    if str(given.get("phase")) == "B":
        if int(given.get("task_end", 0)) != int(values["n_tasks"]):
            disagreements.append(
                f"task_end: notebook {given.get('task_end')!r} vs protocol "
                f"{values['n_tasks']!r}")
        if sorted(int(s) for s in given.get("seeds", ())) != sorted(
            int(s) for s in values["seeds"]
        ):
            disagreements.append(
                f"seeds: notebook {list(given.get('seeds', ()))} vs protocol "
                f"{values['seeds']}")

    return Agreement(
        agrees=not disagreements,
        checked=len(comparable) + (2 if str(given.get("phase")) == "B" else 0),
        disagreements=tuple(disagreements),
        path=str(path),
    )
