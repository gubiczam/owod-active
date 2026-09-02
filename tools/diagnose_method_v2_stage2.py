#!/usr/bin/env python
"""Method V2 Stage 2: do D, R and C earn a place in a later acquisition ablation?

Diagnostic only. **This runs no acquisition, tunes no lambda or gamma, and does
not touch U or replay.** Protocol and every threshold were frozen in
``docs/method_v2_stage2_protocol_2026-09-02.md`` before this script existed.

Stage 1 failed officially and that verdict is not revisited here. It failed on the
two background-facing criteria and passed the two semantics-facing ones, so Stage 1
keeps the object/background job and DINO semantics are tested only inside the
object-like population it produces.

Oracle labels are used to *score* rankings and never inside D, R or C.

    python tools/diagnose_method_v2_stage2.py \\
        --export  /content/drive/MyDrive/OWL/features/dinov2_vitb14_method_v2_v1.npz \\
        --views   /content/drive/MyDrive/OWL/features/dinov2_vitb14_stage2_views_v1.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import spearmanr

from owl import method_v2_stage2 as stage2
from owl import proposals as proposals_module
from owl import protocol as owl_protocol
from owl import scoring
from owl import semantic_features as sf
from tools.audit_decoder_layers import populations
from tools.diagnose_population import nms_keep
from tools.diagnose_representation import load

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"
SEEDS = (0, 1, 2)


def write_csv(name: str, rows: list[dict]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="frozen DINOv2 base export")
    parser.add_argument("--ref-t1", required=True,
                        help="REF-T1 export from tools/export_ref_t1_features.py -- "
                             "the canonical Task-1 labelled reference, and the only "
                             "reference that decides D_GO")
    parser.add_argument("--views", default=None,
                        help="Stage-2 view export; omit to skip component C")
    parser.add_argument("--with-pseudo-reference", action="store_true",
                        help="also report the REF-A pseudo-known manifold as a "
                             "secondary diagnostic (never decides a verdict)")
    parser.add_argument("--pool", default=str(sf.POOL))
    arguments = parser.parse_args()

    # ---- the fixed population, verified before anything is computed --------
    pool = load()
    payload = np.load(arguments.pool, allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == sf.POOL_SPLIT
    pool["raw_boxes"] = payload["boxes"][keep].astype(np.float32)
    candidates = proposals_module.from_frozen_pool(arguments.pool, split=sf.POOL_SPLIT)
    rows_meta = sf.pool_rows(arguments.pool)

    masks = populations(pool, candidates)
    p2 = masks["P2_admissible_nms"]
    report = stage2.verify_p2(p2, pool["kind"])
    print(f"[p2] {report['rows']:,} rows, background "
          f"{report['background_share']:.4f}  PASS (fixed input, not re-optimised)")

    export = sf.read(arguments.export)
    sf.validate(export, rows_meta)
    features = export.features()
    print(f"[export] validated, {features.shape}, model "
          f"{export.provenance.get('model_id')}")

    groups = owl_protocol.load_groups()
    kind, group = pool["kind"], pool["group"]
    p2_rows = np.flatnonzero(p2)
    admissibility = scoring.admissibility(candidates)

    # ---- REF-T1: the canonical Task-1 labelled reference. PRIMARY. ----------
    from tools.export_ref_t1_features import read as read_reference
    from tools.export_ref_t1_features import validate as validate_reference

    ref_payload = read_reference(arguments.ref_t1)
    ref_report = validate_reference(
        ref_payload["keys"], ref_payload["embeddings"], ref_payload["class_name"])
    reference = ref_payload["embeddings"].astype(np.float32)
    print(f"[REF-T1] {ref_report}  PASS")
    print("[REF-T1] Task-1 labelled GT boxes only; no candidate oracle label and "
          "no future-task annotation is read")
    # REF-T1 is disjoint from the candidate pool by construction: its rows are GT
    # objects, not proposals, so no candidate can be its own reference.
    position_in_reference = None

    local = features[p2]
    local_kind, local_group = kind[p2], group[p2]
    unknown = local_kind == "unknown"
    known = local_kind == "known"
    background = local_kind == "background"

    # ---- baselines on exactly the same P2 rows -----------------------------
    baseline_a = stage2.rank_table(
        admissibility[p2], candidates, p2_rows, groups=groups, name="A_admissibility")
    random_rank = stage2.rank_table(
        np.random.default_rng(0).permutation(p2_rows.size).astype(float),
        candidates, p2_rows, groups=groups, name="random")

    rankings = list(baseline_a) + list(random_rank)
    summary: dict = {"p2": report,
                     "reference": "REF-T1 (canonical Task-1 labelled GT)",
                     "reference_vectors": int(reference.shape[0]),
                     "seeds": list(SEEDS)}

    # ================= component D =========================================
    d_values = stage2.novelty(local, reference, exclude_self=position_in_reference)
    d_rows = stage2.group_summary(d_values, local_kind, local_group, name="D")
    d_table = stage2.rank_table(d_values, candidates, p2_rows, groups=groups, name="D")
    rankings += d_table
    d_auc_known = stage2.auc(d_values, unknown, known)
    d_auc_background = stage2.auc(d_values, unknown, background)
    d_verdict = stage2.evaluate_d(unknown_vs_known_auc=d_auc_known,
                                  table=d_table, baseline=baseline_a)
    d_rows += [{"score": "D", "stratum": "auc_unknown_vs_known", "n": 0,
                "median": d_auc_known, "mean": d_auc_known, "q25": "", "q75": ""},
               {"score": "D", "stratum": "auc_unknown_vs_background", "n": 0,
                "median": d_auc_background, "mean": d_auc_background,
                "q25": "", "q75": ""}]
    # secondary, never decisive: the REF-A pseudo-known manifold, for comparison
    if arguments.with_pseudo_reference:
        saved = pool["pred_obj"].copy()
        pool["pred_obj"] = -admissibility
        try:
            deduplicated = nms_keep(pool, np.ones(len(candidates), dtype=bool),
                                    stage2.NMS_IOU)
        finally:
            pool["pred_obj"] = saved
        pseudo_mask = stage2.pseudo_reference_mask(
            candidates.posterior, admissibility, pool["raw_boxes"],
            pool["image_ids"], nms=deduplicated)
        pseudo_rows = np.flatnonzero(pseudo_mask)
        mapping = np.full(len(candidates), -1, dtype=np.int64)
        mapping[pseudo_rows] = np.arange(pseudo_rows.size)
        pseudo_values = stage2.novelty(local, features[pseudo_mask],
                                       exclude_self=mapping[p2])
        print(f"[REF-A] SECONDARY pseudo-known manifold, {pseudo_rows.size:,} "
              "vectors; reported only, decides nothing")
        d_rows += stage2.group_summary(pseudo_values, local_kind, local_group,
                                       name="D_pseudo_REF_A_secondary")
        rankings += stage2.rank_table(pseudo_values, candidates, p2_rows,
                                      groups=groups,
                                      name="D_pseudo_REF_A_secondary")
        summary["D_pseudo_reference_secondary"] = {
            "vectors": int(pseudo_rows.size),
            "auc_unknown_vs_known": stage2.auc(pseudo_values, unknown, known),
            "note": "pseudo-known manifold estimated from the candidate pool; not "
                    "the labelled set and not part of any verdict",
        }

    write_csv("method_v2_stage2_d.csv", d_rows)
    summary["D"] = d_verdict | {"auc_unknown_vs_background": d_auc_background,
                                "reference": "REF-T1", "reference_report": ref_report}

    # ================= component R =========================================
    counts = owl_protocol.load_train_counts()
    true_frequency = np.asarray(
        [counts.get(name, np.nan) for name in pool["class_name"][p2]], dtype=float)

    r_rows: list[dict] = []
    definitions: dict[str, dict] = {}
    for name, values in (
        ("R1_candidate_density", stage2.rarity_r1(local)),
        ("R2_labelled_deficit", stage2.rarity_r2(local, reference)),
        ("R3_partition_undercoverage", stage2.rarity_r3(local, reference, seed=0)),
    ):
        # R3 is the only stochastic definition. Seed 0 carries the verdict, as
        # declared; seeds 1 and 2 are reported as stability rows so a k-means
        # initialisation cannot be the thing that decides R_GO unnoticed.
        strata = stage2.group_summary(values, local_kind, local_group, name=name)
        table = stage2.rank_table(values, candidates, p2_rows, groups=groups, name=name)
        rankings += table
        usable = unknown & np.isfinite(true_frequency)
        rho = (float(spearmanr(values[usable], true_frequency[usable]).statistic)
               if usable.sum() > 5 else float("nan"))
        medians = {band: float(np.median(values[unknown & (local_group == band)]))
                   if (unknown & (local_group == band)).any() else None
                   for band in stage2.GROUPS}
        definitions[name] = {"medians": medians, "table": table,
                             "baseline": baseline_a}
        r_rows += strata + [
            {"score": name, "stratum": "spearman_true_frequency", "n": int(usable.sum()),
             "median": rho, "mean": rho, "q25": "", "q75": ""},
            {"score": name, "stratum": "spearman_inverse_frequency",
             "n": int(usable.sum()), "median": -rho, "mean": -rho, "q25": "", "q75": ""},
        ]
    for seed in SEEDS[1:]:
        stability = stage2.rarity_r3(local, reference, seed=seed)
        r_rows += [row | {"score": f"R3_partition_undercoverage_seed{seed}"}
                   for row in stage2.group_summary(
                       stability, local_kind, local_group,
                       name=f"R3_partition_undercoverage_seed{seed}")]

    write_csv("method_v2_stage2_r.csv", r_rows)
    r_verdict = stage2.evaluate_r(definitions)
    summary["R"] = {key: value for key, value in r_verdict.items()
                    if key != "definitions"}
    summary["R"]["definitions"] = {
        name: {k: v for k, v in entry.items() if k != "fractions"}
        for name, entry in r_verdict["definitions"].items()
    }

    # ================= component C =========================================
    if arguments.views:
        from tools.export_dinov2_consistency_views import read as read_views
        from tools.export_dinov2_consistency_views import validate as validate_views

        keys, views, view_provenance = read_views(arguments.views)
        validate_views(keys, views)
        expected = np.asarray([str(k) for k in rows_meta.keys[p2]])
        if not np.array_equal(keys, expected):
            raise stage2.Stage2Error(
                "the view export's rows are not the P2 rows in P2 order; C would "
                "compare different proposals."
            )
        measured = stage2.consistency(
            local, views["view_a"].astype(np.float32), views["view_b"].astype(np.float32))
        c_values = measured["consistency"]
        c_rows = stage2.group_summary(c_values, local_kind, local_group, name="C_min")
        c_rows += stage2.group_summary(measured["consistency_mean"], local_kind,
                                       local_group, name="C_mean_descriptive")
        c_auc = stage2.auc(c_values, unknown, background)
        tail = unknown & (local_group == "tail")
        c_auc_tail = stage2.auc(c_values, tail, unknown & ~tail)
        # C as a weight on the Stage-1 ranking, the protocol's filter/weight test
        weighted = stage2.rank_table(
            stage2.score_c(admissibility[p2], c_values),
            candidates, p2_rows, groups=groups, name="A_times_C_frozen")
        rankings += weighted
        c_verdict = stage2.evaluate_c(unknown_vs_background_auc=c_auc,
                                      table=weighted, baseline=baseline_a)
        c_rows += [
            {"score": "C_min", "stratum": "auc_unknown_vs_background", "n": 0,
             "median": c_auc, "mean": c_auc, "q25": "", "q75": ""},
            {"score": "C_min", "stratum": "auc_tail_vs_other_unknown", "n": 0,
             "median": c_auc_tail, "mean": c_auc_tail, "q25": "", "q75": ""},
        ]
        write_csv("method_v2_stage2_c.csv", c_rows)
        summary["C"] = c_verdict | {"auc_tail_vs_other_unknown": c_auc_tail,
                                    "views_provenance": view_provenance.get("view_margins")}
    else:
        c_verdict = {"component": "C", "go": False,
                     "checks": {"views_not_supplied": False},
                     "unknown_vs_background_auc": float("nan"),
                     "note": "no --views export supplied; C not evaluated"}
        summary["C"] = c_verdict
        print("[C] no --views export supplied; C is reported NO_GO for absence of "
              "evidence, not for failing its test")

    write_csv("method_v2_stage2_rankings.csv", rankings)

    verdict = stage2.Stage2Verdict(d=d_verdict, r=r_verdict, c=c_verdict)
    summary["ladder"] = verdict.ladder
    (RESULTS / "method_v2_stage2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"wrote {RESULTS / 'method_v2_stage2_summary.json'}")

    print("\n" + "=" * 78)
    print("METHOD V2 STAGE 2 — component verdicts (frozen criteria, section 9)")
    print("=" * 78)
    print(f"  D  unknown-vs-known AUC   = {d_auc_known:.4f}  >= "
          f"{stage2.D_GO_UNKNOWN_VS_KNOWN_AUC:.2f}")
    for fraction, gain in d_verdict["gains"].items():
        print(f"     top {fraction:.0%}: unknown-object gain over A "
              f"{gain['unknown_objects']:+.1%}, tail-object gain "
              f"{gain['tail_objects']:+.1%}")
    print(f"  R  passing definitions: {r_verdict['passing_definitions'] or 'none'}")
    print("     primary coverage endpoint: distinct medium+tail oracle OBJECTS; "
          "class coverage is descriptive")
    for name, entry in r_verdict["definitions"].items():
        print(f"     {name:30s} monotone={entry['monotone_head_medium_tail']}  "
              f"object_gain+background_same_fraction="
              f"{entry['object_gain_and_background_at_same_fraction']}  "
              f"at={entry['satisfying_fraction']}  medians={entry['medians']}")
    print(f"  C  unknown-vs-background AUC = "
          f"{c_verdict.get('unknown_vs_background_auc', float('nan')):.4f}  >= "
          f"{stage2.C_GO_UNKNOWN_VS_BACKGROUND_AUC:.2f}")
    print()
    for line in verdict.lines():
        print(line)


if __name__ == "__main__":
    main()
