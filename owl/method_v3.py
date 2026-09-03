"""Method V3 — selection to learning transfer. Exploratory, and frozen.

One concept: **does the acquisition ranking change what the detector learns**,
at an equal annotation budget, with everything else held identical?

Method V2 Stage 2 returned ``D_NO_GO``, ``R_NO_GO``, ``C_GO`` and left the
allowed ladder at ``U``. Nothing here reopens that. This module runs a *new*
experiment on the one component that passed its own gate: it takes the frozen
consistency values, builds the frozen product ranking ``A(x) * C(x)``, spends an
equal region budget under four rankings, fine-tunes PROB once per trajectory and
scores it with PROB's own evaluator.

The protocol is owned by ``docs/method_v3_protocol_2026-09-02.md``, which was
written before any downstream detector endpoint existed. Everything this module
hard-codes is there with its justification; nothing here may be chosen after a
result is seen.

Why this is not :func:`owl.runner.run_chain`. ``run_chain`` re-predicts a fresh
candidate pool per task, and ``C`` exists only for the frozen P2 population — so
a live pool has no ``C`` and the arms could not share one candidate population.
Method V3 is therefore a **single incremental task over a fixed population**, and
it reuses ``owl.selection``, ``owl.labelling``, ``owl.replay``, ``owl.exemplars``,
``owl.discovery``, ``owl.metrics`` and ``owl.bridge`` unchanged. ``run_chain`` and
every result produced from it are untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from owl import discovery as discovery_module
from owl import exemplars as exemplar_memory
from owl import labelling, metrics, protocol, replay, scoring, selection
from owl import method_v2_stage2 as stage2
from owl import proposals as proposals_module
from owl.bridge import Bridge

# =============================================================== the design ===

#: The four arms, fixed 2026-09-02 before any downstream endpoint. Display names.
ARMS: tuple[str, ...] = ("random", "A", "U", "A*C")

#: Filesystem/CSV-safe tokens. ``A*C`` cannot be a path component.
SLUGS: Mapping[str, str] = {"random": "random", "A": "a", "U": "u", "A*C": "a_times_c"}

SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGET: int = 600
ROUNDS: int = 6
#: Where the per-budget selection curve is reported. Exactly the round prefixes.
BUDGET_MARKS: tuple[int, ...] = (100, 200, 300, 400, 500, 600)

#: One incremental task: the published anchor, then ``CLASS_ORDER[19]``.
N_TASKS: int = 2

#: Held fixed for every arm and seed. §3 and §5 of the protocol.
REPLAY_ARM: str = "uniform"
LABELLING_POLICY: str = "known_plus_selected"
EPOCHS: int = 5
LEARNING_RATE: float = 2e-4
BATCH_SIZE: int = 2
SUPERVISION_MODE: str = "ft"

#: The shared evaluation split, built once. §8.
EVAL_MAX_PER_CLASS: int = 150
EVAL_REMAINDER_RATIO: int = 3

#: The population, verified fail-closed before anything expensive runs. §4.
EXPECTED_POPULATION_ROWS: int = 8_010
EXPECTED_POPULATION_IMAGES: int = 839

#: The frozen artefacts Method V3 reads. REF-T1 is deliberately absent: ``C``
#: needs the base export and the two views, not the Stage-2 reference.
BASE_EXPORT_NAME = "dinov2_vitb14_method_v2_v1.npz"
VIEWS_EXPORT_NAME = "dinov2_vitb14_stage2_views_v1.npz"


class MethodV3Error(RuntimeError):
    """Raised when a frozen input, population or trajectory does not reproduce."""


# ================================================================ population ===


@dataclass(frozen=True)
class Population:
    """The fixed candidate population, plus what proves it is the frozen one."""

    candidates: proposals_module.Candidates   # already restricted
    keys: np.ndarray                          # (N,) "image#query", restricted
    p2_mask: np.ndarray                       # (80000,) bool, P2 over the whole pool
    keep_mask: np.ndarray                     # (80000,) bool, P2 ∩ annotated images
    within_p2: np.ndarray                     # (15518,) bool, the same restriction
    provenance: dict

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def image_ids(self) -> np.ndarray:
        return np.unique(self.candidates.image_ids)


def population(
    pool_path: str | Path,
    candidate_index: Mapping[str, Mapping[str, int]],
) -> Population:
    """P2 restricted to images whose benchmark annotation is committed here.

    Fail-closed on both counts. A population that does not reproduce is not
    silently trained on: every number below would be measured on something else.
    """

    from owl import semantic_features as sf
    from tools.audit_decoder_layers import populations
    from tools.diagnose_representation import load

    pool_path = Path(pool_path)
    pool = load()
    payload = np.load(pool_path, allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == sf.POOL_SPLIT
    pool["raw_boxes"] = payload["boxes"][keep].astype(np.float32)

    whole = proposals_module.from_frozen_pool(pool_path, split=sf.POOL_SPLIT)
    rows = sf.pool_rows(pool_path)
    p2 = populations(pool, whole)["P2_admissible_nms"]
    p2_report = stage2.verify_p2(p2, pool["kind"])

    image_ids = np.asarray(whole.image_ids, dtype=str)
    annotated = np.asarray(sorted(set(image_ids[p2]) & set(candidate_index)), dtype=str)
    keep_mask = p2 & np.isin(image_ids, annotated)

    n_rows = int(keep_mask.sum())
    n_images = int(np.unique(image_ids[keep_mask]).size)
    if n_rows != EXPECTED_POPULATION_ROWS or n_images != EXPECTED_POPULATION_IMAGES:
        raise MethodV3Error(
            f"the Method V3 population holds {n_rows:,} proposals on {n_images:,} "
            f"images, expected {EXPECTED_POPULATION_ROWS:,} on "
            f"{EXPECTED_POPULATION_IMAGES:,}. The fixed candidate population did "
            "not reproduce; investigate rather than proceed."
        )

    return Population(
        candidates=whole.take(np.flatnonzero(keep_mask)),
        keys=rows.keys[keep_mask],
        p2_mask=p2,
        keep_mask=keep_mask,
        within_p2=keep_mask[p2],
        provenance={
            "pool": str(pool_path),
            "pool_sha256": _sha256(pool_path),
            "p2": p2_report,
            "rows": n_rows,
            "images": n_images,
            "restriction": "P2 ∩ images with a committed benchmark annotation",
            "keys_sha256": hashlib.sha256(
                "\n".join(str(k) for k in rows.keys[keep_mask]).encode()
            ).hexdigest(),
        },
    )


def consistency_values(
    pool: Population,
    *,
    base_export: str | Path,
    views_export: str | Path,
) -> tuple[np.ndarray, dict]:
    """``C(x)`` for the population, read verbatim from the frozen Stage-2 exports.

    No DINOv2 forward pass, no re-cropping, no re-normalisation: the base export
    and the two view exports are validated against the frozen pool identity and
    then fed to :func:`owl.method_v2_stage2.consistency` unchanged.
    """

    from owl import semantic_features as sf
    from tools.export_dinov2_consistency_views import read as read_views
    from tools.export_dinov2_consistency_views import validate as validate_views

    rows = sf.pool_rows(pool.provenance["pool"])
    export = sf.read(base_export)
    base_report = sf.validate(export, rows)
    features = export.features()

    keys, views, view_provenance = read_views(views_export)
    views_report = validate_views(keys, views)
    expected = np.asarray([str(k) for k in rows.keys[pool.p2_mask]], dtype=str)
    if not np.array_equal(keys, expected):
        raise MethodV3Error(
            "the view export does not cover exactly the P2 rows in P2 order; "
            "Method V3 refuses to realign a frozen Stage-2 artefact"
        )

    on_p2 = stage2.consistency(
        features[pool.p2_mask],
        views["view_a"].astype(np.float32),
        views["view_b"].astype(np.float32),
    )["consistency"]
    values = np.asarray(on_p2, dtype=np.float32)[pool.within_p2]
    if values.size != len(pool):
        raise MethodV3Error(
            f"C covers {values.size} rows against {len(pool)} population rows"
        )
    return values, {
        "base_export": str(base_export),
        "base_export_sha256": _sha256(base_export),
        "base_validation": base_report,
        "views_export": str(views_export),
        "views_export_sha256": _sha256(views_export),
        "views_validation": views_report,
        "view_margins": view_provenance.get("view_margins"),
        "definition": "C(x) = min(cos(z_1.20, z_1.10), cos(z_1.20, z_1.30))",
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


# ===================================================================== arms ===


def arm_score(
    name: str,
    candidates: proposals_module.Candidates,
    *,
    consistency: np.ndarray | None = None,
) -> np.ndarray | None:
    """The ranking one arm selects by. ``None`` means "the random arm".

    Reads detector outputs and the frozen ``C`` only. No oracle is touched: this
    is acquisition, and :meth:`owl.proposals.Candidates.oracle` is not called
    anywhere in this function or in what it calls.
    """

    if name not in ARMS:
        raise MethodV3Error(f"unknown arm {name!r}; the four arms are {ARMS}")
    if name == "random":
        return None
    if name == "A":
        return np.asarray(scoring.admissibility(candidates), dtype=np.float64)
    if name == "U":
        return np.asarray(
            scoring.uncertainty(candidates, "entropy"), dtype=np.float64
        )
    # A*C -- the literal product, through Stage 2's own frozen function.
    if consistency is None:
        raise MethodV3Error("arm 'A*C' needs the frozen consistency values")
    consistency = np.asarray(consistency, dtype=np.float32)
    if consistency.shape[0] != len(candidates):
        raise MethodV3Error(
            f"C covers {consistency.shape[0]} rows against {len(candidates)} candidates"
        )
    return np.asarray(
        stage2.score_c(scoring.admissibility(candidates), consistency), dtype=np.float64
    )


def arm_config(name: str, seed: int) -> scoring.ScoreConfig:
    """One arm's :class:`~owl.scoring.ScoreConfig`.

    Every semantic weight is zero and the coherence factor is declared ``off``:
    the ranking is supplied to :func:`owl.selection.select` precomputed, so no
    term may be reintroduced by a default. ``random=True`` short-circuits to the
    seeded uniform draw.
    """

    return scoring.ScoreConfig(
        name=name,
        random=(name == "random"),
        lambda_diversity=0.0,
        gamma_rarity=0.0,
        mu_batch=0.0,
        coherence_method="off",
        combination="additive",
        seed=seed,
    )


def select_for_arm(
    candidates: proposals_module.Candidates,
    name: str,
    seed: int,
    *,
    consistency: np.ndarray | None = None,
    budget: int = BUDGET,
    rounds: int = ROUNDS,
) -> selection.Selection:
    """Spend ``budget`` regions over ``rounds`` under one arm's ranking."""

    return selection.select(
        candidates,
        arm_config(name, seed),
        budget=budget,
        rounds=rounds,
        precomputed=arm_score(name, candidates, consistency=consistency),
    )


# ================================================================ criterion ===


@dataclass(frozen=True)
class Criterion:
    """``C_DOWNSTREAM_POSITIVE``, frozen before execution. Protocol §11."""

    primary_metric: str = "mAP50_medium_tail"
    guard_metric: str = "known_mAP50"
    treatment: str = "A*C"
    control: str = "A"
    budget: int = BUDGET
    minimum_improving_seeds: int = 2
    guard_tolerance: float = 1.0

    def statement(self) -> str:
        return (
            f"C_DOWNSTREAM_POSITIVE  iff, at budget {self.budget}:\n"
            f"  (1) mean {self.primary_metric}({self.treatment}) > "
            f"mean {self.primary_metric}({self.control})\n"
            f"  (2) at least {self.minimum_improving_seeds} of {len(SEEDS)} paired "
            f"seed differences {self.treatment} - {self.control} are > 0\n"
            f"  (3) mean {self.guard_metric}({self.treatment}) >= "
            f"mean {self.guard_metric}({self.control}) - {self.guard_tolerance}\n"
            f"otherwise C_DOWNSTREAM_NOT_SUPPORTED"
        )


#: The one criterion. Instantiated at import so it cannot be parameterised later.
CRITERION = Criterion()

#: Where the criterion is written down for a human. §11.0 of that document holds
#: a machine-readable copy, and :func:`check_protocol_criterion` compares it.
PROTOCOL_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs" / "method_v3_protocol_2026-09-02.md"
)

#: The fence the structured copy lives behind. A tagged fence rather than a bare
#: ``json`` one so an unrelated JSON example in the document can never be picked
#: up as the criterion.
CRITERION_BLOCK = re.compile(r"^```json criterion\n(.*?)^```", re.DOTALL | re.MULTILINE)


def parse_criterion_block(text: str) -> dict:
    """The criterion as the protocol document declares it, as values.

    Exactly one tagged block must be present. Returning parsed JSON rather than a
    matched phrase is the whole point: the first version of this check searched
    the document for ``f"{guard_tolerance:g} AP50 point"``, which renders ``1.0``
    as ``1`` while the document says ``1.0`` — a correct, frozen criterion
    reported as a mismatch, and an overnight run stopped before it trained
    anything. Numbers are compared as numbers here, so a difference in rendering
    cannot fail and a difference in value cannot pass.
    """

    found = CRITERION_BLOCK.findall(text)
    if len(found) != 1:
        raise MethodV3Error(
            f"the protocol holds {len(found)} ```json criterion``` blocks, "
            "expected exactly 1. The criterion must have one machine-readable "
            "declaration, not zero and not two."
        )
    try:
        payload = json.loads(found[0])
    except json.JSONDecodeError as error:
        raise MethodV3Error(
            f"the protocol's criterion block is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise MethodV3Error(
            f"the protocol's criterion block is a {type(payload).__name__}, "
            "expected an object"
        )
    return payload


def check_protocol_criterion(path: str | Path | None = None) -> dict:
    """Fail closed unless the document and :data:`CRITERION` agree, value by value.

    Compares the field *set* and each field's *value*. ``1`` and ``1.0`` are the
    same number and are accepted as such; ``1.0`` and ``2.0`` are not, and neither
    is a missing or an extra field. Wording, emphasis and number formatting in the
    document's prose are deliberately not examined — they are for the reader.
    """

    path = PROTOCOL_PATH if path is None else Path(path)
    if not Path(path).is_file():
        raise MethodV3Error(f"the protocol document is missing: {path}")
    text = Path(path).read_text(encoding="utf-8")

    declared = parse_criterion_block(text)
    frozen = asdict(CRITERION)

    missing = sorted(set(frozen) - set(declared))
    extra = sorted(set(declared) - set(frozen))
    differing = {
        name: (declared[name], frozen[name])
        for name in sorted(set(frozen) & set(declared))
        if declared[name] != frozen[name]
    }
    if missing or extra or differing:
        lines = []
        if missing:
            lines.append(f"    not declared in the protocol: {missing}")
        if extra:
            lines.append(f"    declared but not a criterion field: {extra}")
        for name, (there, here) in differing.items():
            lines.append(f"    {name}: protocol {there!r}, owl.method_v3 {here!r}")
        raise MethodV3Error(
            f"{path} and owl.method_v3.CRITERION declare different criteria:\n"
            + "\n".join(lines)
            + "\n"
            "This is a real disagreement about what counts as a success, not a "
            "formatting difference — the two are compared as values. Decide which "
            "is right and change both deliberately."
        )

    # The two verdict labels are format-free strings, so checking that the
    # document names both of them costs nothing and catches a document that
    # describes some other decision procedure entirely.
    for label in ("C_DOWNSTREAM_POSITIVE", "C_DOWNSTREAM_NOT_SUPPORTED"):
        if label not in text:
            raise MethodV3Error(f"{path} never names the verdict {label!r}")

    return {"protocol": str(path), "criterion": frozen,
            "statement": CRITERION.statement()}


@dataclass(frozen=True)
class Verdict:
    """The mechanical outcome. ``label`` is one of the two frozen strings."""

    label: str
    clauses: Mapping[str, bool]
    detail: Mapping[str, object]

    @property
    def positive(self) -> bool:
        return self.label == "C_DOWNSTREAM_POSITIVE"

    def failed_clauses(self) -> tuple[str, ...]:
        return tuple(name for name, ok in self.clauses.items() if not ok)


def medium_tail_classes(
    known_classes: Sequence[str], groups: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """The medium and tail known classes, in the evaluator's own class order."""

    groups = protocol.load_groups() if groups is None else groups
    return tuple(
        name for name in known_classes if groups.get(name) in ("medium", "tail")
    )


def evaluate_criterion(
    rows: Sequence[Mapping[str, object]], criterion: Criterion = CRITERION
) -> Verdict:
    """Apply the frozen criterion to the twelve result rows. No branching on data.

    ``rows`` are the flat per-trajectory records written by
    :func:`run_trajectory`; only ``arm``, ``seed`` and the two metrics are read.
    """

    def series(arm: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for row in rows:
            if row.get("arm") != arm or int(row.get("budget", -1)) != criterion.budget:
                continue
            out[int(row["seed"])] = float(row[criterion.primary_metric])
        return out

    def guard(arm: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for row in rows:
            if row.get("arm") != arm or int(row.get("budget", -1)) != criterion.budget:
                continue
            out[int(row["seed"])] = float(row[criterion.guard_metric])
        return out

    treatment, control = series(criterion.treatment), series(criterion.control)
    missing = [s for s in SEEDS if s not in treatment or s not in control]
    if missing:
        raise MethodV3Error(
            f"the criterion needs {criterion.treatment} and {criterion.control} at "
            f"every seed {SEEDS}; missing seeds {missing}. A verdict is not "
            "computed from an incomplete design."
        )

    paired = {s: treatment[s] - control[s] for s in SEEDS}
    guard_treatment = float(np.mean([guard(criterion.treatment)[s] for s in SEEDS]))
    guard_control = float(np.mean([guard(criterion.control)[s] for s in SEEDS]))
    mean_treatment = float(np.mean([treatment[s] for s in SEEDS]))
    mean_control = float(np.mean([control[s] for s in SEEDS]))

    clauses = {
        "mean_improves": mean_treatment > mean_control,
        "majority_of_paired_seeds_improve": (
            sum(1 for value in paired.values() if value > 0)
            >= criterion.minimum_improving_seeds
        ),
        "known_map_within_tolerance": (
            guard_treatment >= guard_control - criterion.guard_tolerance
        ),
    }
    return Verdict(
        label=(
            "C_DOWNSTREAM_POSITIVE" if all(clauses.values())
            else "C_DOWNSTREAM_NOT_SUPPORTED"
        ),
        clauses=clauses,
        detail={
            "criterion": asdict(criterion),
            "primary_mean_treatment": mean_treatment,
            "primary_mean_control": mean_control,
            "primary_sd_treatment": float(
                np.std([treatment[s] for s in SEEDS], ddof=1)
            ),
            "primary_sd_control": float(np.std([control[s] for s in SEEDS], ddof=1)),
            "paired_differences": {str(s): paired[s] for s in SEEDS},
            "improving_seeds": sum(1 for value in paired.values() if value > 0),
            "guard_mean_treatment": guard_treatment,
            "guard_mean_control": guard_control,
            "guard_delta": guard_treatment - guard_control,
            "n_seeds": len(SEEDS),
            "significance": "descriptive only; n=3 supports no significance claim",
        },
    )


# ============================================================== bookkeeping ===


def trajectory_name(arm: str, seed: int) -> str:
    """``a_times_c__seed1``. One directory per trajectory, resumable by name."""

    if arm not in ARMS:
        raise MethodV3Error(f"unknown arm {arm!r}")
    return f"{SLUGS[arm]}__seed{int(seed)}"


def trajectories() -> tuple[tuple[str, int], ...]:
    """All twelve, in a fixed order. Every one is attempted."""

    return tuple((arm, seed) for arm in ARMS for seed in SEEDS)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: object) -> Path:
    """Atomic write: a killed runtime leaves either the old file or the new one."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)
    return path


def write_rows(path: str | Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Atomic CSV. Columns are the union of the rows' keys, in first-seen order."""

    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def fingerprint(
    arm: str, seed: int, pool: Population, *, test_set: str, dry_run: bool = False
) -> dict:
    """What makes two trajectories the same experiment. Compared before resuming.

    ``dry_run`` is part of the identity on purpose: a stubbed orchestration walk
    writes real-looking files, and the one thing that must never happen is a real
    run resuming a stubbed trajectory and reporting its synthetic numbers. With
    the flag inside the fingerprint that mistake fails closed instead.
    """

    return {
        "experiment": "method_v3_selection_transfer",
        "dry_run": bool(dry_run),
        "arm": arm,
        "seed": int(seed),
        "arms": list(ARMS),
        "budget": BUDGET,
        "rounds": ROUNDS,
        "n_tasks": N_TASKS,
        "replay_arm": REPLAY_ARM,
        "replay_total": int(replay.ARMS[REPLAY_ARM]["total"]),
        "replay_alpha": float(replay.ARMS[REPLAY_ARM]["alpha"]),
        "labelling_policy": LABELLING_POLICY,
        "supervision_mode": SUPERVISION_MODE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "test_set": test_set,
        "eval_max_per_class": EVAL_MAX_PER_CLASS,
        "eval_remainder_ratio": EVAL_REMAINDER_RATIO,
        "population_rows": pool.provenance["rows"],
        "population_images": pool.provenance["images"],
        "population_keys_sha256": pool.provenance["keys_sha256"],
        "criterion": asdict(CRITERION),
    }


STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_RUNNING = "running"


def load_trajectory(directory: str | Path) -> dict | None:
    """A finished trajectory's record, or ``None`` if it is absent or not complete.

    A ``failed`` or ``running`` record is deliberately **not** returned as a
    result: an interrupted trajectory must be re-run, never reported.
    """

    path = Path(directory) / "result.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MethodV3Error(
            f"{path} is corrupt ({error}). Delete that trajectory directory and "
            "re-run it; a corrupt result is never treated as complete."
        ) from error
    return payload if payload.get("status") == STATUS_COMPLETE else None


# ============================================================== the trajectory ===


@dataclass
class TrajectoryInputs:
    """Everything one trajectory needs that is not the arm or the seed."""

    pool: Population
    candidate_index: Mapping[str, Mapping[str, int]]
    replay_index: Mapping[str, Mapping[str, int]]
    replay_root: Path
    start_checkpoint: Path
    test_set: str
    consistency: np.ndarray | None = None
    anchor_metrics: Path | None = None
    prepare_images: object | None = None
    provenance: dict = field(default_factory=dict)


def annotation_protocol(inputs: TrajectoryInputs) -> str:
    """The exact protocol, printed before the real run. Protocol §5, §6."""

    task = protocol.build_chain(N_TASKS)[1]
    spec = replay.ARMS[REPLAY_ARM]
    return "\n".join([
        "METHOD V3 — ANNOTATION, REPLAY AND TRAINING PROTOCOL (frozen)",
        (f"  task                 {task.name}: declares {task.new_class!r} "
         f"({protocol.load_groups().get(task.new_class)} band); "
         f"{task.n_prev} -> {task.n_current} known classes"),
        (f"  candidate population {inputs.pool.provenance['rows']:,} proposals on "
         f"{inputs.pool.provenance['images']:,} images "
         f"({inputs.pool.provenance['restriction']})"),
        (f"  budget               {BUDGET} regions in {ROUNDS} rounds of "
         f"{BUDGET // ROUNDS}; region = one proposal the oracle is asked about"),
        (f"  labelling policy     {LABELLING_POLICY} — known-class objects on an "
         "opened image are free, chosen unknowns are labelled, remaining unknowns "
         "are ignored rather than taught as background"),
        "  opened image         a chosen region opens its whole image",
        "  oracle matching      proposal to benchmark annotation at IoU 0.5",
        "  per-image NMS        IoU 0.60, inside P2's frozen construction",
        "  half-labelling       0 by construction under this policy",
        ("  distinct objects     every object count goes through owl.discovery; "
         "proposals are never reported as objects"),
        (f"  replay               {REPLAY_ARM}: {spec['total']} exemplar OBJECTS, "
         f"alpha={spec['alpha']}, protocol version 3, drawn from the canonical "
         "old-data index over the 19 Task-1 classes, identical for all four arms"),
        (f"  training             supervision_mode={SUPERVISION_MODE}, "
         f"epochs={EPOCHS}, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}, "
         "freeze_prob_model=True"),
        (f"  evaluation           PROB's own evaluator on {inputs.test_set}, "
         "with the per-box detection artefact, identical for every trajectory"),
        ("  GPU-path note        owl.labelling prices the annotation; PROB itself "
         "reads each opened image's real annotation XML and keeps the boxes whose "
         "category falls in range(0, prev + current). Supervision is therefore "
         "decided by WHICH IMAGES an arm opens, not by which box inside an image "
         "was clicked, and a still-undeclared class yields no gradient at this "
         "task. Established protocol, unchanged."),
    ])


def run_trajectory(
    bridge: Bridge,
    arm: str,
    seed: int,
    *,
    workspace: Path,
    inputs: TrajectoryInputs,
    dry_run: bool = False,
) -> dict:
    """One arm at one seed: select, annotate, rehearse, fine-tune, evaluate.

    Resumable and fail-visible. A completed trajectory is skipped on its
    ``result.json``; an interrupted one is re-run; a corrupt one stops the run.
    """

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    stamp = workspace / "config.json"
    expected = fingerprint(
        arm, seed, inputs.pool, test_set=inputs.test_set, dry_run=dry_run
    )

    if stamp.exists():
        stored = json.loads(stamp.read_text(encoding="utf-8"))
        differing = {
            name: (stored.get(name, "(absent)"), value)
            for name, value in expected.items()
            if name not in stored or stored[name] != value
        }
        if differing:
            lines = "\n".join(
                f"    {name}: stored {was!r}, now {now!r}"
                for name, (was, now) in sorted(differing.items())
            )
            raise MethodV3Error(
                f"{workspace} holds a different configuration:\n{lines}\n"
                f"Delete it —\n    rm -rf '{workspace}'\n— or run elsewhere."
            )

    done = load_trajectory(workspace)
    if done is not None:
        print(f"  [{arm} seed{seed}] already complete; restored from result.json")
        return done

    write_json(stamp, expected)
    write_json(workspace / "status.json", {
        "status": STATUS_RUNNING, "arm": arm, "seed": seed,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    try:
        row = _run_trajectory_body(
            bridge, arm, seed, workspace, inputs, dry_run=dry_run
        )
    except BaseException as error:
        write_json(workspace / "status.json", {
            "status": STATUS_FAILED, "arm": arm, "seed": seed,
            "error": f"{type(error).__name__}: {error}",
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        raise

    write_json(workspace / "status.json", {
        "status": STATUS_COMPLETE, "arm": arm, "seed": seed,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return row


def _run_trajectory_body(
    bridge: Bridge, arm: str, seed: int, workspace: Path, inputs: TrajectoryInputs,
    *, dry_run: bool = False,
) -> dict:
    chain = protocol.build_chain(N_TASKS)
    anchor_task, task = chain[0], chain[1]
    groups = protocol.load_groups()
    candidates = inputs.pool.candidates
    started = time.time()

    # ---- 0. the anchor, measured once and reused ---------------------------
    #
    # Forgetting is "what did this step cost", so it needs the previous-class
    # mAP of the checkpoint the step started from. It is the same checkpoint on
    # the same split for all twelve trajectories, so a shared anchor is copied in
    # and `bridge.evaluate` skips the call.
    anchor_path = workspace / "anchor_metrics.json"
    if inputs.anchor_metrics is not None and not anchor_path.exists():
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        anchor_path.write_text(
            Path(inputs.anchor_metrics).read_text(encoding="utf-8"), encoding="utf-8"
        )
    anchor_path = bridge.evaluate(
        checkpoint=Path(inputs.start_checkpoint), test_set=inputs.test_set,
        output=anchor_path, n_prev=anchor_task.n_prev, n_current=anchor_task.n_new,
        detections=False,
    )
    anchor = metrics.from_bridge_metrics(anchor_path)

    # ---- 1. spend the budget ----------------------------------------------
    picked = select_for_arm(
        candidates, arm, seed, consistency=inputs.consistency
    )
    if len(picked) != BUDGET:
        raise MethodV3Error(
            f"{arm} seed{seed} spent {len(picked)} regions, not {BUDGET}"
        )

    # ---- 2. price the answers ---------------------------------------------
    annotation = labelling.annotate(
        candidates, picked, policy=LABELLING_POLICY,
        known_classes=task.previous_classes,
    )
    if annotation.oracle_cost != BUDGET:
        raise MethodV3Error(
            f"{arm} seed{seed} was charged {annotation.oracle_cost} oracle units "
            f"for a {BUDGET}-region budget under {LABELLING_POLICY!r}"
        )

    # ---- 3. what the selection actually bought (evaluation, not acquisition) --
    found = discovery_module.discovery(candidates, picked.indices, groups=groups)
    curve = discovery_module.cumulative(
        candidates, picked.indices, picked.round_of, groups=groups
    )
    for entry, mark in zip(curve, BUDGET_MARKS):
        entry["budget"] = mark
        entry["arm"] = arm
        entry["seed"] = seed
    write_rows(workspace / "selection_curve.csv", curve)

    # ---- 4. which opened images PROB can be trained on --------------------
    known_now = frozenset(task.known_classes)
    opened = [str(value) for value in picked.images(candidates)]
    trainable = [
        image for image in opened
        if any(name in known_now for name in inputs.candidate_index.get(image, {}))
    ]
    barren = len(opened) - len(trainable)
    if len(trainable) < BATCH_SIZE:
        raise MethodV3Error(
            f"{arm} seed{seed} kept only {len(trainable)} trainable images of "
            f"{len(opened)} opened; PROB drops the last partial batch and needs "
            f"{BATCH_SIZE}"
        )

    # ---- 5. the exemplar memory: 400 objects of old knowledge --------------
    spec = replay.ARMS[REPLAY_ARM]
    replay_budget, replay_alpha = int(spec["total"]), float(spec["alpha"])
    previous = frozenset(task.previous_classes)
    spent_images = frozenset(trainable)

    def eligible(available: set[str] | None = None):
        pool = exemplar_memory.enumerate_pool(inputs.replay_index, previous)
        return tuple(
            item for item in pool
            if item.class_name in previous
            and item.image_id not in spent_images
            and (available is None or item.image_id in available)
        )

    def build(pool):
        demand = replay.allocate(
            exemplar_memory.capacities(pool), total=replay_budget, alpha=replay_alpha
        )
        return exemplar_memory.select(
            pool, demand, incumbent=(), reallocate=False, seed=seed
        ), demand

    eligible_pool = eligible()
    chosen, demand = build(eligible_pool)

    # ---- 6. everything training will read must be on disk -----------------
    if inputs.prepare_images is not None:
        for _ in range(3):
            sources = {item.image_id for item in chosen}
            wanted = sorted({*trainable, *sources})
            present = {str(value) for value in inputs.prepare_images(wanted)}
            trainable = [image for image in trainable if image in present]
            if sources <= present:
                break
            eligible_pool = eligible(available=present)
            chosen, demand = build(eligible_pool)
        if len(trainable) < BATCH_SIZE:
            raise MethodV3Error(
                f"{arm} seed{seed} has only {len(trainable)} training images on "
                "disk; the downloads failed"
            )
        if len(chosen) != replay_budget:
            raise MethodV3Error(
                f"{arm} seed{seed} could only put {len(chosen)} of {replay_budget} "
                "exemplar objects on disk. Rehearsing on a short memory would make "
                "this trajectory incomparable with the others, which is the one "
                "thing the object budget exists to prevent."
            )

    aliases = tuple(sorted(
        exemplar_memory.write_aliases(chosen, data_root=Path(inputs.replay_root))
    ))

    # ---- 7. fine-tune ------------------------------------------------------
    checkpoint = bridge.train(
        trainable,
        previous_checkpoint=Path(inputs.start_checkpoint),
        output_checkpoint=workspace / "checkpoint.pth",
        output_dir=workspace / "train",
        n_prev=task.n_prev, n_current=task.n_new, test_set=inputs.test_set,
        replay_ids=aliases, supervision_mode=SUPERVISION_MODE,
        epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
    )

    # ---- 8. score it -------------------------------------------------------
    metrics_path = bridge.evaluate(
        checkpoint=checkpoint, test_set=inputs.test_set,
        output=workspace / "metrics.json",
        n_prev=task.n_prev, n_current=task.n_new, detections=True,
    )
    evaluation = metrics.from_bridge_metrics(metrics_path)
    membership = metrics.group_membership(task.known_classes, groups)
    medium_tail = medium_tail_classes(task.known_classes, groups)
    row = metrics.task_row(
        evaluation, task=task.name, new_class=task.new_class,
        previous_baseline=anchor.known_map50,
        anchor_known_map50=anchor.known_map50, groups=membership,
    )
    row["mAP50_medium_tail"] = metrics.grouped_map(
        evaluation, {"medium_tail": medium_tail}
    )["medium_tail"]

    artefact = json.loads(metrics_path.read_text(encoding="utf-8")).get("detections_path")
    if artefact and Path(artefact).exists():
        by_group = metrics.unknown_recall_by_group(
            Path(artefact), known_classes=task.known_classes, groups=groups
        )
        for name in ("head", "medium", "tail", "all"):
            row[f"U_Recall_{name}"] = by_group[name]["recall"]
            row[f"unknown_objects_{name}"] = by_group[name]["objects"]

    # ---- 9. one flat record, written atomically ---------------------------
    # ``found.row()`` first, then the trajectory's own keys. Both dictionaries
    # carry ``images_opened`` and both carry a ``tail_classes``, and they do NOT
    # mean the same thing: Discovery counts *how many* tail classes the selection
    # touched, while the band membership is a *list of names*. Merging the other
    # way round silently replaced the name list with a count. The band lists are
    # renamed so the two can never collide again, and the explicit keys are
    # applied last so their meaning wins.
    record = {
        **found.row(),
        "status": STATUS_COMPLETE,
        "dry_run": bool(dry_run),
        "arm": arm,
        "seed": seed,
        "budget": BUDGET,
        "task": task.name,
        "new_class": task.new_class,
        "known_medium_tail_classes": list(medium_tail),
        "known_tail_classes": list(membership.get("tail", ())),
        "oracle_cost": annotation.oracle_cost,
        "images_opened": len(opened),
        "images_trainable": len(trainable),
        "images_no_supervision": barren,
        "replay_requested_objects": replay_budget,
        "replay_allocated_objects": int(sum(demand.values())),
        "replay_objects": len(chosen),
        "replay_alias_images": len(aliases),
        "replay_per_class": exemplar_memory.delivered_per_class(chosen),
        "half_labelled_share": labelling.half_labelling_rate(annotation, candidates),
        "background_share_of_selection": found.selected_background / BUDGET,
        "anchor_known_mAP50": anchor.known_map50,
        "minutes": (time.time() - started) / 60.0,
        "checkpoint": str(checkpoint),
        "metrics_path": str(metrics_path),
        **{key: value for key, value in row.items() if key not in ("task", "new_class")},
        "medium_tail_objects": (
            found.objects_by_group.get("medium", 0)
            + found.objects_by_group.get("tail", 0)
        ),
        "provenance": dict(inputs.provenance),
    }
    write_json(workspace / "result.json", record)
    write_rows(workspace / "result.csv", [
        {k: v for k, v in record.items()
         if not isinstance(v, (dict, list))}
    ])
    print(f"  [{arm} seed{seed}] done in {record['minutes']:.1f} min; "
          f"medium+tail mAP50 {record['mAP50_medium_tail']:.2f}, "
          f"known mAP50 {record['known_mAP50']:.2f}")
    return record


# ================================================================== manifest ===


def manifest(
    *,
    owl_sha: str,
    prob_sha: str,
    checkpoint: str | Path,
    pool: Population,
    consistency_provenance: Mapping[str, object],
    test_set: str,
    test_images: int,
    scheduled: Sequence[tuple[str, int]],
    completed: Mapping[str, str],
    dry_run: bool = False,
    runtime_estimate: Mapping[str, object] | None = None,
) -> dict:
    """The machine-readable run manifest. Everything needed to re-derive the run."""

    return {
        "experiment": "method_v3_selection_transfer",
        "dry_run": bool(dry_run),
        "label": "Method V3 — exploratory/prospective Selection→Learning Transfer",
        "protocol": "docs/method_v3_protocol_2026-09-02.md",
        "does_not_reopen": {
            "method_v2_stage2": {
                "D": "NO_GO", "R": "NO_GO", "C": "GO",
                "allowed_ladder": "U",
                "note": "thresholds unchanged; Method V3 asks a new question",
            }
        },
        "owl_commit": owl_sha,
        "prob_commit": prob_sha,
        "prob_branch": "feat/daowod-bridge-v2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint) if Path(checkpoint).exists() else None,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "budget": BUDGET,
        "rounds": ROUNDS,
        "budget_marks": list(BUDGET_MARKS),
        "replay": {"arm": REPLAY_ARM, **replay.ARMS[REPLAY_ARM]},
        "labelling_policy": LABELLING_POLICY,
        "training": {
            "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "supervision_mode": SUPERVISION_MODE,
        },
        "evaluation": {
            "test_set": test_set, "images": int(test_images),
            "max_per_class": EVAL_MAX_PER_CLASS,
            "remainder_multiplier": EVAL_REMAINDER_RATIO,
            "detections": True,
        },
        "population": pool.provenance,
        "consistency": dict(consistency_provenance),
        "criterion": {**asdict(CRITERION), "statement": CRITERION.statement()},
        "scheduled": [{"arm": arm, "seed": seed} for arm, seed in scheduled],
        "trajectory_status": dict(completed),
        "runtime_estimate": dict(runtime_estimate or {}),
        "platform": {"python": platform.python_version(), "node": platform.node()},
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
