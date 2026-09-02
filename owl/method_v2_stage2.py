"""Method V2 Stage 2: do DINO semantics earn D, R and C inside Stage 1's output?

Stage 1 failed officially — ``METHOD_V2_REPRESENTATION_FAIL`` — and that verdict
stands. It failed on the two background-facing criteria (open-pool kNN 0.0370,
unknown-vs-background AUC 0.6835, both worse than PROB) while passing the two
semantics-facing ones (unknown-class kNN 0.4387 against PROB's 0.1773,
unknown-tail ≈0.73). So the reading is not "the semantic geometry is unusable" but
"DINOv2 must not be asked to be the background detector".

This module holds the components of the composition that follows from that:
PROB objectness plus NMS keeps the object/background job, and DINO semantics
operate only inside the object-like set it produces.

Everything here is **construction plus accounting**. Oracle labels appear only in
:func:`rank_table` and the evaluators, which score a ranking after the fact —
never in D, R or C. ``tests/test_method_v2_stage2.py`` pins that separation.

Protocol: ``docs/method_v2_stage2_protocol_2026-09-02.md``, frozen before this ran.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from owl import clustering, discovery
from owl.decoder_layers import ExportError

#: Reused from the repository rather than chosen for this experiment.
K_NEIGHBOURS = 10          # tools.diagnose_representation.K_NEIGHBOURS
N_CLUSTERS = 120           # tools.audit_decoder_layers.N_CLUSTERS
NMS_IOU = 0.60             # tools.audit_decoder_layers.NMS_IOU
N_KNOWN_AT_T1 = 19

#: The fixed input population. A different count is a failure, not a variation.
EXPECTED_P2_ROWS = 15_518
EXPECTED_P2_BACKGROUND = 0.767
P2_BACKGROUND_TOLERANCE = 0.002

#: Rank fractions reported; the GO tests use only the predeclared subset.
REPORT_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.30)
GO_FRACTIONS = (0.05, 0.10, 0.20)

#: Frozen thresholds, protocol section 9. Not reinterpreted after results.
D_GO_UNKNOWN_VS_KNOWN_AUC = 0.65
D_GO_RELATIVE_IMPROVEMENT = 0.10
R_GO_RELATIVE_IMPROVEMENT = 0.10
R_GO_MAX_BACKGROUND_INCREASE = 0.10      # percentage points, as a fraction
C_GO_UNKNOWN_VS_BACKGROUND_AUC = 0.60
C_GO_RELATIVE_IMPROVEMENT = 0.10

EPS = 1e-6
GROUPS = ("head", "medium", "tail")


class Stage2Error(ExportError):
    """Raised when the fixed population or an input does not reproduce."""


# ------------------------------------------------------- the fixed population ---


def verify_p2(mask: np.ndarray, kind: np.ndarray) -> dict:
    """Fail closed unless P2 reproduces exactly.

    P2 is an *input* to Stage 2, not something being re-optimised, so a drifted
    population would silently change what every component below is measured on.
    """

    rows = int(np.asarray(mask, dtype=bool).sum())
    background = float((np.asarray(kind)[mask] == "background").mean())
    if rows != EXPECTED_P2_ROWS:
        raise Stage2Error(
            f"P2 holds {rows:,} rows, expected {EXPECTED_P2_ROWS:,}. The fixed "
            "input population did not reproduce; investigate rather than proceed."
        )
    if abs(background - EXPECTED_P2_BACKGROUND) > P2_BACKGROUND_TOLERANCE:
        raise Stage2Error(
            f"P2 background share {background:.4f}, expected "
            f"{EXPECTED_P2_BACKGROUND:.3f} +/- {P2_BACKGROUND_TOLERANCE}"
        )
    return {"rows": rows, "background_share": background}


def reference_mask(
    posterior: np.ndarray,
    admissibility: np.ndarray,
    boxes: np.ndarray,
    image_ids: np.ndarray,
    *,
    nms: np.ndarray | None = None,
) -> np.ndarray:
    """REF-A: detector-predicted-known proposals, NMS-deduplicated. Oracle-free.

    The repository's established stand-in for "what is already labelled" --
    ``clustering.predicted_known`` is what ``clustering.contamination`` already
    runs on, and the detector recognises its own 19 classes at 0.83 accuracy. It
    mimics the round-0 labelled state without reading one oracle label.

    Deduplicated because a crowded object otherwise contributes a dozen nearly
    identical reference vectors and dominates every nearest-neighbour query made
    against the set. ``nms`` is injected so the caller supplies the repository's
    own suppression rather than this module growing a second implementation.
    """

    predicted = clustering.predicted_known(posterior, N_KNOWN_AT_T1)
    if not predicted.any():
        raise Stage2Error("no proposal is predicted-known; REF-A would be empty")
    if nms is None:
        return predicted
    return predicted & np.asarray(nms, dtype=bool)


# ------------------------------------------------------------------ component D ---


def novelty(features: np.ndarray, reference: np.ndarray,
            *, exclude_self: np.ndarray | None = None,
            chunk: int = 2048) -> np.ndarray:
    """``D(x) = 1 - max cos(z_x, z_r)`` over the labelled reference set.

    ``features`` and ``reference`` must be L2-normalised, which the frozen export
    guarantees, so a dot product *is* the cosine.

    ``exclude_self`` maps each candidate row to its index inside ``reference`` (or
    -1). A predicted-known candidate is itself a reference vector, and without the
    exclusion its D would be exactly 0 for the trivial reason that it matched
    itself.
    """

    features = np.asarray(features, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if reference.shape[0] == 0:
        raise Stage2Error("the labelled reference set is empty")
    if reference.shape[1] != features.shape[1]:
        raise Stage2Error(
            f"reference dim {reference.shape[1]} against candidate dim "
            f"{features.shape[1]}"
        )

    best = np.empty(features.shape[0], dtype=np.float32)
    for start in range(0, features.shape[0], chunk):
        stop = min(start + chunk, features.shape[0])
        similarity = features[start:stop] @ reference.T
        if exclude_self is not None:
            for offset, position in enumerate(exclude_self[start:stop]):
                if position >= 0:
                    similarity[offset, position] = -np.inf
        best[start:stop] = similarity.max(axis=1)
    return (1.0 - best).astype(np.float32)


# ------------------------------------------------------------------ component R ---


def _kth_distance(query: np.ndarray, reference: np.ndarray, k: int,
                  *, drop_self: bool) -> np.ndarray:
    """Distance to the k-th nearest reference row, optionally ignoring row i == i."""

    n_ask = min(k + (1 if drop_self else 0), reference.shape[0])
    if n_ask < 1:
        raise Stage2Error("not enough reference rows for the requested k")
    model = NearestNeighbors(n_neighbors=n_ask, n_jobs=-1).fit(reference)
    distances = model.kneighbors(query, return_distance=True)[0]
    return distances[:, -1].astype(np.float32)


def rarity_r1(features: np.ndarray, *, k: int = K_NEIGHBOURS) -> np.ndarray:
    """R1 -- inverse local density among candidates: distance to the k-th candidate.

    Predeclared as the definition closest to the isolation signal that already
    failed three times, and expected to be the weakest of the three. It is here so
    that expectation is tested rather than assumed.
    """

    return _kth_distance(features, features, k, drop_self=True)


def rarity_r2(features: np.ndarray, reference: np.ndarray,
              *, k: int = K_NEIGHBOURS) -> np.ndarray:
    """R2 -- labelled coverage deficit, as a log ratio of the two k-th distances.

    High where a candidate sits in a region **dense with candidates but far from
    anything labelled**: semantically under-covered rather than merely isolated.
    The log keeps it stable when either distance approaches zero.
    """

    candidate = _kth_distance(features, features, k, drop_self=True)
    labelled = _kth_distance(features, reference, k, drop_self=False)
    return np.log((EPS + labelled) / (EPS + candidate)).astype(np.float32)


def rarity_r3(features: np.ndarray, reference: np.ndarray,
              *, n_clusters: int = N_CLUSTERS, seed: int = 0) -> np.ndarray:
    """R3 -- semantic partition under-coverage, from one oracle-free k-means.

    Candidates and reference vectors are assigned to the same centroids, and a
    cluster scores high when the candidates populate it and the labelled set does
    not. This is *coverage*, not cluster size: the falsified claim was that
    cluster size estimates true class frequency, and nothing here assumes it.
    """

    model = MiniBatchKMeans(
        n_clusters=min(n_clusters, max(features.shape[0] // 4, 2)),
        random_state=seed, n_init=3, batch_size=4096,
    ).fit(features)
    candidate_labels = model.labels_
    reference_labels = model.predict(reference)
    size = model.n_clusters
    candidates = np.bincount(candidate_labels, minlength=size)
    labelled = np.bincount(reference_labels, minlength=size)
    score = np.log((1.0 + candidates) / (1.0 + labelled))
    return score[candidate_labels].astype(np.float32)


# ------------------------------------------------------------------ component C ---


def consistency(base: np.ndarray, view_a: np.ndarray,
                view_b: np.ndarray) -> dict[str, np.ndarray]:
    """Semantic stability across the two frozen context views.

    ``C = min(sim_A, sim_B)``; the mean is returned for description only. All
    three inputs are L2-normalised, so the row-wise dot product is the cosine.

    Deliberately not a density measure: three density operationalisations already
    failed in the same direction, because in this pool local density orders
    background < known < unknown-head < unknown-tail and no threshold reverses a
    monotone ordering.
    """

    for name, matrix in (("view_a", view_a), ("view_b", view_b)):
        if matrix.shape != base.shape:
            raise Stage2Error(
                f"{name} is {matrix.shape}, base is {base.shape}; the views must "
                "cover exactly the same rows in the same order"
            )
    similarity_a = (base * view_a).sum(axis=1).astype(np.float32)
    similarity_b = (base * view_b).sum(axis=1).astype(np.float32)
    return {
        "sim_a": similarity_a,
        "sim_b": similarity_b,
        "consistency": np.minimum(similarity_a, similarity_b),
        "consistency_mean": ((similarity_a + similarity_b) / 2.0).astype(np.float32),
    }


# ------------------------------------------------------- ranking and accounting ---


def rank_table(
    scores: np.ndarray,
    candidates,
    rows: np.ndarray,
    *,
    groups: Mapping[str, str],
    fractions: Sequence[float] = REPORT_FRACTIONS,
    name: str = "",
) -> list[dict]:
    """Top-fraction accounting for one ranking, in **distinct objects**.

    ``rows`` are positions in the full candidate pool, so the returned counts are
    directly comparable with every other ranking scored on the same P2 rows.

    Distinct-object counting is not optional. Under proposal counting the same
    comparison once inflated an arm by 1.76x against a control's 1.02x and
    reversed its conclusion; ``owl.discovery`` is the single implementation that
    prevents a second occurrence.
    """

    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[0] != rows.shape[0]:
        raise Stage2Error(
            f"{scores.shape[0]} scores against {rows.shape[0]} population rows"
        )
    order = rows[np.argsort(-scores, kind="stable")]
    oracle = candidates.oracle()

    out = []
    for fraction in fractions:
        take = max(1, round(order.size * fraction))
        selected = order[:take]
        found = discovery.discovery(candidates, selected, groups=groups)
        kinds = oracle.kind[selected]
        row = {"ranking": name, "fraction": fraction, "proposals": int(take)} | {
            key: value for key, value in found.row().items()
            if key not in ("asked",)
        }
        row |= {
            "distinct_oracle_objects": int(
                np.unique(oracle.object_id[selected][oracle.object_id[selected] >= 0]).size
            ),
            "background_share": float((kinds == "background").mean()),
            "known_share": float((kinds == "known").mean()),
            "unique_images": int(np.unique(candidates.image_ids[selected]).size),
        }
        out.append(row)
    return out


def group_summary(scores: np.ndarray, kind: np.ndarray, group: np.ndarray,
                  *, name: str = "") -> list[dict]:
    """Median/quartile of a score per oracle stratum. Evaluation only."""

    strata: list[tuple[str, np.ndarray]] = [
        ("background", kind == "background"),
        ("known", kind == "known"),
        ("unknown_all", kind == "unknown"),
    ]
    strata += [(f"unknown_{band}", (kind == "unknown") & (group == band))
               for band in GROUPS]

    out = []
    for label, mask in strata:
        values = np.asarray(scores)[mask]
        out.append({
            "score": name, "stratum": label, "n": int(mask.sum()),
            "median": float(np.median(values)) if values.size else float("nan"),
            "mean": float(values.mean()) if values.size else float("nan"),
            "q25": float(np.quantile(values, 0.25)) if values.size else float("nan"),
            "q75": float(np.quantile(values, 0.75)) if values.size else float("nan"),
        })
    return out


def auc(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    """ROC AUC of ``scores`` separating two oracle strata. Evaluation only."""

    if positive.sum() < 2 or negative.sum() < 2:
        return float("nan")
    mask = positive | negative
    return float(roc_auc_score(positive[mask], np.asarray(scores)[mask]))


# --------------------------------------------------------------- the GO tests ---


def _at(table: Sequence[Mapping], fraction: float) -> Mapping | None:
    for row in table:
        if abs(float(row["fraction"]) - fraction) < 1e-9:
            return row
    return None


def _relative_gain(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        return float("inf") if candidate > 0 else 0.0
    return (candidate - baseline) / baseline


def evaluate_d(*, unknown_vs_known_auc: float, table: Sequence[Mapping],
               baseline: Sequence[Mapping]) -> dict:
    """Protocol section 9: AUC >= 0.65 AND >= 10% relative gain at some fraction."""

    gains = {}
    improved = False
    for fraction in GO_FRACTIONS:
        row, reference = _at(table, fraction), _at(baseline, fraction)
        if row is None or reference is None:
            continue
        objects = _relative_gain(float(row["unknown_objects"]),
                                 float(reference["unknown_objects"]))
        tail = _relative_gain(float(row["tail_objects"]),
                              float(reference["tail_objects"]))
        gains[fraction] = {"unknown_objects": objects, "tail_objects": tail}
        if max(objects, tail) >= D_GO_RELATIVE_IMPROVEMENT:
            improved = True
    checks = {
        "unknown_vs_known_auc>=0.65": bool(
            not np.isnan(unknown_vs_known_auc)
            and unknown_vs_known_auc >= D_GO_UNKNOWN_VS_KNOWN_AUC
        ),
        "relative_gain>=10pct_at_some_fraction": bool(improved),
    }
    return {"component": "D", "go": all(checks.values()), "checks": checks,
            "unknown_vs_known_auc": unknown_vs_known_auc, "gains": gains}


def evaluate_r(definitions: Mapping[str, Mapping]) -> dict:
    """Protocol section 9: monotone head->medium->tail median, plus coverage gain.

    ``definitions`` maps an R name to
    ``{"medians": {band: value}, "table": [...], "baseline": [...]}``.
    R is GO if **any one** predeclared definition satisfies both conditions.
    """

    per_definition = {}
    for name, payload in definitions.items():
        medians = payload["medians"]
        monotone = bool(
            medians.get("head") is not None
            and medians["head"] <= medians["medium"] <= medians["tail"]
        )
        coverage = False
        background_ok = False
        details = {}
        for fraction in GO_FRACTIONS:
            row = _at(payload["table"], fraction)
            reference = _at(payload["baseline"], fraction)
            if row is None or reference is None:
                continue
            mid_tail_objects = (float(row["medium_objects"]) + float(row["tail_objects"]))
            mid_tail_reference = (float(reference["medium_objects"])
                                  + float(reference["tail_objects"]))
            mid_tail_classes = (float(row["medium_classes"]) + float(row["tail_classes"]))
            mid_tail_class_ref = (float(reference["medium_classes"])
                                  + float(reference["tail_classes"]))
            object_gain = _relative_gain(mid_tail_objects, mid_tail_reference)
            class_gain = _relative_gain(mid_tail_classes, mid_tail_class_ref)
            increase = float(row["background_share"]) - float(reference["background_share"])
            details[fraction] = {
                "medium_tail_object_gain": object_gain,
                "medium_tail_class_gain": class_gain,
                "background_increase": increase,
            }
            if (max(object_gain, class_gain) >= R_GO_RELATIVE_IMPROVEMENT
                    and increase <= R_GO_MAX_BACKGROUND_INCREASE):
                coverage = True
                background_ok = True
        per_definition[name] = {
            "monotone_head_medium_tail": monotone,
            "coverage_gain_within_background_budget": bool(coverage and background_ok),
            "medians": dict(medians),
            "fractions": details,
            "go": bool(monotone and coverage and background_ok),
        }
    winners = [name for name, entry in per_definition.items() if entry["go"]]
    return {"component": "R", "go": bool(winners), "passing_definitions": winners,
            "definitions": per_definition}


def evaluate_c(*, unknown_vs_background_auc: float,
               table: Sequence[Mapping] | None = None,
               baseline: Sequence[Mapping] | None = None) -> dict:
    """Protocol section 9: AUC >= 0.60 OR a >= 10% gain that keeps the tail."""

    auc_ok = bool(
        not np.isnan(unknown_vs_background_auc)
        and unknown_vs_background_auc >= C_GO_UNKNOWN_VS_BACKGROUND_AUC
    )
    filter_ok = False
    details = {}
    if table is not None and baseline is not None:
        for fraction in GO_FRACTIONS:
            row, reference = _at(table, fraction), _at(baseline, fraction)
            if row is None or reference is None:
                continue
            gain = _relative_gain(float(row["unknown_objects"]),
                                  float(reference["unknown_objects"]))
            tail_kept = float(row["tail_objects"]) >= float(reference["tail_objects"])
            details[fraction] = {"unknown_object_gain": gain, "tail_kept": tail_kept}
            if gain >= C_GO_RELATIVE_IMPROVEMENT and tail_kept:
                filter_ok = True
    checks = {
        "unknown_vs_background_auc>=0.60": auc_ok,
        "filter_gain>=10pct_keeping_tail": bool(filter_ok),
    }
    return {"component": "C", "go": bool(auc_ok or filter_ok), "checks": checks,
            "unknown_vs_background_auc": unknown_vs_background_auc,
            "fractions": details}


def allowed_ladder(d_go: bool, r_go: bool, c_go: bool) -> str:
    """The permitted acquisition ladder, built in order and stopped at a failure.

    D is the gateway: R and C are semantic refinements of a novelty score, so a
    ladder containing R without D would not be one of the four outcomes the
    protocol enumerates.
    """

    if not d_go:
        return "U"
    if not r_go:
        return "U+D"
    if not c_go:
        return "U+D+R"
    return "U+D+R*C"


@dataclass(frozen=True)
class Stage2Verdict:
    d: dict
    r: dict
    c: dict

    @property
    def ladder(self) -> str:
        return allowed_ladder(self.d["go"], self.r["go"], self.c["go"])

    def lines(self) -> list[str]:
        return [
            f"D_{'GO' if self.d['go'] else 'NO_GO'}",
            f"R_{'GO' if self.r['go'] else 'NO_GO'}",
            f"C_{'GO' if self.c['go'] else 'NO_GO'}",
            f"METHOD_V2_ALLOWED_LADDER = {self.ladder}",
        ]
