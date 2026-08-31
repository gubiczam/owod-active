"""Controlled long-tail training views for the fixed S-OWODB T1 classes.

The historical protocol starts from a detector that already knows the nineteen
T1 classes.  Those are consequently the fixed population on which forgetting
is measured at every later task.  This module controls their *training-object*
distribution without touching source XMLs, the candidate pool, or evaluation.

The controlled conditions use a common total object count and an exponential
rank schedule.  Individual objects are selected by a seeded SHA-256 order per
class, which is independent of Python, NumPy, filesystem, and dictionary order.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from owl import protocol, runner

PROTOCOL_VERSION = 1
PROTOCOL_NAME = "sowodb_controlled_t1_longtail"
CONTROLLED_CLASSES: tuple[str, ...] = protocol.TASK1
CONTROLLED_RHOS: tuple[int, ...] = (10, 50, 100)
CONDITIONS: tuple[str, ...] = ("original", "lt10", "lt50", "lt100")
GROUPS: tuple[str, ...] = ("head", "medium", "tail")
HISTORICAL_WORKSPACES: frozenset[str] = frozenset({
    "random__none", "random__uniform", "random__tail_favouring",
})


class LongTailError(ValueError):
    """Raised when a controlled training view is incomplete or inconsistent."""


@dataclass(frozen=True, order=True)
class ObjectIdentity:
    """One source annotation object, ordinal within its canonical class."""

    image_id: str
    class_name: str
    ordinal: int


def canonical_json_bytes(payload: object) -> bytes:
    """Canonical scientific content; paths and timestamps must be supplied explicitly."""

    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source_index(path: str | Path) -> dict[str, dict[str, int]]:
    """Read and validate the committed canonical T1 object-count index."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise LongTailError("The source index must be a non-empty image mapping.")
    index: dict[str, dict[str, int]] = {}
    allowed = set(CONTROLLED_CLASSES)
    for image_id, values in raw.items():
        image_id = str(image_id)
        if len(image_id) != 12 or not image_id.isdigit():
            raise LongTailError(f"Invalid canonical image id {image_id!r}.")
        if not isinstance(values, dict):
            raise LongTailError(f"Image {image_id} has a non-object count row.")
        counts: dict[str, int] = {}
        for name, value in values.items():
            if name not in allowed:
                raise LongTailError(f"Image {image_id} contains non-T1 class {name!r}.")
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise LongTailError(f"Image {image_id} has invalid {name!r} count {value!r}.")
            counts[name] = int(value)
        if counts:
            index[image_id] = counts
    missing = allowed - {name for counts in index.values() for name in counts}
    if missing:
        raise LongTailError(f"Source index has no objects for {sorted(missing)}.")
    return dict(sorted(index.items()))


def object_counts(index: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for values in index.values():
        counts.update({name: int(value) for name, value in values.items()})
    return {name: counts[name] for name in CONTROLLED_CLASSES}


def class_ranking(counts: Mapping[str, int]) -> tuple[str, ...]:
    """Descending original frequency, with evaluator order as the tie-break."""

    position = {name: index for index, name in enumerate(CONTROLLED_CLASSES)}
    if set(counts) != set(CONTROLLED_CLASSES):
        raise LongTailError("Class counts must cover exactly the nineteen T1 classes.")
    if any(int(counts[name]) <= 0 for name in CONTROLLED_CLASSES):
        raise LongTailError("Every controlled class must have at least one source object.")
    return tuple(sorted(CONTROLLED_CLASSES, key=lambda name: (-int(counts[name]), position[name])))


def rank_groups(ranking: Sequence[str]) -> dict[str, str]:
    """Top/middle/bottom rank thirds; for 19 classes the sizes are 7/6/6."""

    if len(set(ranking)) != len(ranking) or not ranking:
        raise LongTailError("Ranking must contain unique classes.")
    return {
        name: GROUPS[min(2, rank * 3 // len(ranking))]
        for rank, name in enumerate(ranking)
    }


def schedule_weights(class_count: int, rho: float) -> tuple[float, ...]:
    """``rho**(-r/(C-1))`` for ranks ``r = 0..C-1``."""

    if class_count < 2 or not math.isfinite(rho) or rho < 1:
        raise LongTailError("The schedule needs at least two classes and rho >= 1.")
    return tuple(rho ** (-rank / (class_count - 1)) for rank in range(class_count))


def maximum_feasible_total(counts: Mapping[str, int], rho: float) -> int:
    """Largest continuous-schedule total that needs no duplicate source object."""

    ranking = class_ranking(counts)
    weights = schedule_weights(len(ranking), rho)
    scale = min(int(counts[name]) / weight for name, weight in zip(ranking, weights))
    return math.floor(scale * sum(weights))


def matched_controlled_total(
    counts: Mapping[str, int], rhos: Sequence[float] = CONTROLLED_RHOS,
) -> int:
    """Maximum common total feasible for every requested controlled severity."""

    if not rhos:
        raise LongTailError("At least one controlled rho is required.")
    return min(maximum_feasible_total(counts, rho) for rho in rhos)


def target_counts(
    counts: Mapping[str, int], rho: float, *, total: int,
) -> dict[str, int]:
    """Largest-remainder integer schedule with exact total and no oversampling."""

    ranking = class_ranking(counts)
    weights = schedule_weights(len(ranking), rho)
    if total <= 0 or total > maximum_feasible_total(counts, rho):
        raise LongTailError(
            f"Total {total} is infeasible for rho={rho:g} without oversampling; "
            f"maximum is {maximum_feasible_total(counts, rho)}."
        )
    ideals = [total * weight / sum(weights) for weight in weights]
    targets = [math.floor(value) for value in ideals]
    remaining = total - sum(targets)
    order = sorted(
        range(len(ranking)),
        key=lambda rank: (-(ideals[rank] - targets[rank]), rank),
    )
    for rank in order:
        if remaining == 0:
            break
        name = ranking[rank]
        if targets[rank] < int(counts[name]):
            targets[rank] += 1
            remaining -= 1
    if remaining:
        raise LongTailError(f"Could not place {remaining} rounded target objects.")
    result = dict(zip(ranking, targets))
    if any(result[name] > int(counts[name]) for name in ranking):
        raise LongTailError("A target exceeds its source-class capacity.")
    values = [result[name] for name in ranking]
    if any(left < right for left, right in zip(values, values[1:])):
        raise LongTailError("Integer rounding broke the monotonic rank schedule.")
    if values[-1] <= 0 or sum(values) != total:
        raise LongTailError("The target schedule lost a class or changed the total.")
    return result


def condition_targets(
    counts: Mapping[str, int], condition: str, *, controlled_total: int | None = None,
) -> dict[str, int]:
    condition = str(condition).lower()
    if condition not in CONDITIONS:
        raise LongTailError(f"Unknown condition {condition!r}; expected {CONDITIONS}.")
    if condition == "original":
        return {name: int(counts[name]) for name in class_ranking(counts)}
    rho = int(condition.removeprefix("lt"))
    total = matched_controlled_total(counts) if controlled_total is None else controlled_total
    return target_counts(counts, rho, total=total)


def achieved_rho(targets: Mapping[str, int]) -> float:
    values = [int(value) for value in targets.values()]
    if not values or min(values) <= 0:
        raise LongTailError("Achieved rho requires positive class counts.")
    return max(values) / min(values)


def workspace_name(condition: str, *, seed: int = 0) -> str:
    """An isolated no-replay workspace name that cannot shadow the pilot."""

    condition = str(condition).lower()
    if condition not in CONDITIONS or seed < 0:
        raise LongTailError(f"Invalid condition/seed: {condition!r}, {seed!r}.")
    suffix = "" if seed == 0 else f"__seed{seed}"
    name = f"random__none__{condition}{suffix}"
    if name in HISTORICAL_WORKSPACES:
        raise LongTailError(f"Controlled workspace collides with historical {name}.")
    return name


def enumerate_objects(
    index: Mapping[str, Mapping[str, int]],
) -> dict[str, tuple[ObjectIdentity, ...]]:
    by_class: dict[str, list[ObjectIdentity]] = defaultdict(list)
    for image_id in sorted(index):
        for name in CONTROLLED_CLASSES:
            for ordinal in range(int(index[image_id].get(name, 0))):
                by_class[name].append(ObjectIdentity(image_id, name, ordinal))
    return {name: tuple(by_class[name]) for name in CONTROLLED_CLASSES}


def _selection_key(identity: ObjectIdentity, seed: int) -> tuple[bytes, ObjectIdentity]:
    material = (
        f"{PROTOCOL_NAME}|v{PROTOCOL_VERSION}|{seed}|{identity.class_name}|"
        f"{identity.image_id}|{identity.ordinal}"
    ).encode("utf-8")
    return hashlib.sha256(material).digest(), identity


def select_objects(
    index: Mapping[str, Mapping[str, int]], targets: Mapping[str, int], *, seed: int,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Select exact per-class object identities into a filtered training view."""

    available = enumerate_objects(index)
    selected: list[ObjectIdentity] = []
    for name in class_ranking(object_counts(index)):
        wanted = int(targets.get(name, 0))
        if wanted <= 0 or wanted > len(available[name]):
            raise LongTailError(
                f"Class {name!r} requests {wanted} of {len(available[name])} objects."
            )
        if wanted == len(available[name]):
            selected.extend(available[name])
        else:
            ordered = sorted(available[name], key=lambda item: _selection_key(item, seed))
            selected.extend(ordered[:wanted])
    if len(selected) != len(set(selected)):
        raise LongTailError("Selected object identities are not unique.")
    rows: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for identity in sorted(selected):
        rows[identity.image_id][identity.class_name].append(identity.ordinal)
    result = {
        image_id: {
            name: tuple(ordinals)
            for name, ordinals in sorted(values.items())
        }
        for image_id, values in sorted(rows.items())
    }
    verify_selection(index, result, targets)
    return result


def selection_counts(selection: Mapping[str, Mapping[str, Sequence[int]]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for values in selection.values():
        counts.update({name: len(ordinals) for name, ordinals in values.items()})
    return {name: counts[name] for name in CONTROLLED_CLASSES}


def verify_selection(
    index: Mapping[str, Mapping[str, int]],
    selection: Mapping[str, Mapping[str, Sequence[int]]],
    targets: Mapping[str, int],
) -> None:
    identities: set[ObjectIdentity] = set()
    for image_id, values in selection.items():
        if image_id not in index:
            raise LongTailError(f"Selected image {image_id} is absent from the source index.")
        for name, ordinals in values.items():
            capacity = int(index[image_id].get(name, 0))
            for ordinal in ordinals:
                identity = ObjectIdentity(image_id, name, int(ordinal))
                if ordinal < 0 or ordinal >= capacity:
                    raise LongTailError(f"Selected object {identity} is outside its source XML.")
                if identity in identities:
                    raise LongTailError(f"Duplicate selected object identity {identity}.")
                identities.add(identity)
    if selection_counts(selection) != {
        name: int(targets[name]) for name in CONTROLLED_CLASSES
    }:
        raise LongTailError("Selected-object counts do not match the target schedule.")


def selection_payload(
    condition: str,
    selection: Mapping[str, Mapping[str, Sequence[int]]],
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "condition": condition,
        "objects": {
            image_id: {name: list(ordinals) for name, ordinals in values.items()}
            for image_id, values in selection.items()
        },
    }


def write_gzip_json(path: str | Path, payload: object) -> Path:
    """Write deterministic gzip JSON (no filename or wall-clock header fields)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        target.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as stream,
    ):
        stream.write(canonical_json_bytes(payload))
    return target


def read_gzip_json(path: str | Path) -> object:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def build_manifest(
    *,
    condition: str,
    source_counts: Mapping[str, int],
    targets: Mapping[str, int],
    selected_images: int,
    seed: int,
    source_index_path: str,
    source_index_sha256: str,
    source_annotations_path: str,
    source_annotations_sha256: str,
    selection_path: str,
    selection_sha256: str,
    test_split_sha256: str,
    controlled_total: int,
) -> dict[str, object]:
    ranking = class_ranking(source_counts)
    groups = rank_groups(ranking)
    requested = None if condition == "original" else int(condition.removeprefix("lt"))
    classes = [
        {
            "class_name": name,
            "original_count": int(source_counts[name]),
            "target_count": int(targets[name]),
            "achieved_count": int(targets[name]),
            "rank": rank,
            "group": groups[name],
        }
        for rank, name in enumerate(ranking)
    ]
    payload: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "condition": condition,
        "scope": "the 19 fixed T1 classes whose retention is measured through T2-T6",
        "source_dataset_identity": "S-OWODB canonical COCO training annotations",
        "source_split_identity": "owdetr_t1_train",
        "source_index": {"path": source_index_path, "sha256": source_index_sha256},
        "source_annotations": {
            "path": source_annotations_path,
            "sha256": source_annotations_sha256,
        },
        "evaluation_split_sha256": test_split_sha256,
        "requested_rho": requested,
        "achieved_rho": achieved_rho(targets),
        "seed": seed,
        "sampling_algorithm": "per-class seeded SHA-256 object order without replacement",
        "sampling_unit": "annotated object; training-only filtered XML view",
        "target_formula": "n_r = k * rho^(-r/(C-1)); largest-remainder integer rounding",
        "total_policy": (
            "untouched canonical total" if condition == "original"
            else f"fixed common controlled total={controlled_total}"
        ),
        "selected_images": selected_images,
        "selected_objects": sum(int(value) for value in targets.values()),
        "selection": {"path": selection_path, "sha256": selection_sha256},
        "group_definition": "original-frequency rank thirds: ranks 0-6 / 7-12 / 13-18",
        "classes": classes,
    }
    payload["scientific_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def verify_manifest(manifest: Mapping[str, object]) -> None:
    payload = dict(manifest)
    recorded = str(payload.pop("scientific_sha256", ""))
    actual = sha256_bytes(canonical_json_bytes(payload))
    if recorded != actual:
        raise LongTailError(f"Manifest scientific hash mismatch: {recorded} != {actual}.")


@dataclass
class LongTailCycleConfig(runner.CycleConfig):
    """A run-chain config whose fingerprint cannot collide with historical work."""

    controlled_longtail_protocol_version: int = PROTOCOL_VERSION
    longtail_condition: str = ""
    longtail_manifest_sha256: str = ""
    longtail_source_sha256: str = ""
    longtail_anchor_sha256: str = ""
    longtail_owl_commit: str = ""
    longtail_prob_commit: str = ""

    LONGTAIL_RESULT_AFFECTING = (
        "controlled_longtail_protocol_version",
        "longtail_condition",
        "longtail_manifest_sha256",
        "longtail_source_sha256",
        "longtail_anchor_sha256",
        "longtail_owl_commit",
        "longtail_prob_commit",
    )

    def fingerprint(self) -> dict[str, object]:
        values = {
            name: getattr(self, name)
            for name in self.LONGTAIL_RESULT_AFFECTING
        }
        if self.longtail_condition not in CONDITIONS:
            raise LongTailError(f"Invalid long-tail condition {self.longtail_condition!r}.")
        for name in (
            "longtail_manifest_sha256", "longtail_source_sha256", "longtail_anchor_sha256"
        ):
            value = str(values[name])
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise LongTailError(f"{name} must be an exact lowercase SHA-256.")
        for name in ("longtail_owl_commit", "longtail_prob_commit"):
            value = str(values[name])
            if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
                raise LongTailError(f"{name} must be an exact lowercase Git commit.")
        return super().fingerprint() | values


def fingerprint_sha256(config: LongTailCycleConfig) -> str:
    return sha256_bytes(canonical_json_bytes(config.fingerprint()))
