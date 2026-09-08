#!/usr/bin/env python3
"""Consultation figures, from committed artefacts only. No GPU, no network.

Every panel is drawn from a file in this repository, and each one prints the
file and columns it used so a number on a slide can be traced back in one step.

**Nothing is fabricated.** Panels that need the sequential benchmark's own
result files are declared and *skipped with a reason* unless ``--benchmark`` is
given, rather than being drawn from remembered numbers.

    python tools/build_consultation_figures.py --out docs/figures
    python tools/build_consultation_figures.py --out docs/figures \\
        --benchmark /path/to/per_task_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MEASURED = ROOT / "data" / "reference" / "measured" / "real_group_forgetting.csv"
LABELLING = ROOT / "data" / "results" / "labelling_policy.csv"
COHERENCE = ROOT / "data" / "results" / "coherence_gate.csv"
BATCH = ROOT / "data" / "results" / "batch_diversity_validation.csv"
ROUNDS = ROOT / "data" / "results" / "selection_arms.csv"

INK, WARM, COOL, MUTED = "#1b1b1b", "#c1553b", "#3d6b8e", "#8a8a8a"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def frame(title: str, size=(7.2, 4.2)):
    figure, axes = plt.subplots(figsize=size)
    axes.set_title(title, loc="left", fontsize=11, color=INK, pad=12)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.tick_params(colors=INK, labelsize=9)
    axes.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axes.set_axisbelow(True)
    return figure, axes


def save(figure, out: Path, name: str, source: str, caveat: str) -> None:
    figure.tight_layout()
    figure.savefig(out / f"{name}.png", dpi=200)
    plt.close(figure)
    print(f"  {name}.png\n      source : {source}\n      caveat : {caveat}")


# ---------------------------------------------------------------- panels ---


def figure_forgetting(out: Path) -> None:
    """The headline: where catastrophic forgetting actually came from."""

    wanted = {
        "mult_prior_shrunk_b600": "t1 boxes on the\nselected image\nDISCARDED",
        "mult_prior_shrunk_b600_ft_noreplay": "t1 boxes KEPT\nno replay",
        "mult_prior_shrunk_b600_replay_a0": "t1 boxes KEPT\n+ uniform replay",
    }
    table = {r["point"]: r for r in rows(MEASURED)}
    labels = [wanted[k] for k in wanted]
    values = [float(table[k]["previous19_forgetting"]) for k in wanted]
    kept = [float(table[k]["previous19_map50"]) for k in wanted]

    figure, axes = frame("Forgetting is dominated by discarded free annotation, "
                         "not by the absence of replay")
    bars = axes.bar(labels, values, color=[WARM, COOL, COOL], width=0.55)
    for bar, value, previous in zip(bars, values, kept, strict=True):
        axes.text(bar.get_x() + bar.get_width() / 2, value + 0.8,
                  f"{value:.2f}\nprev-19 mAP50 {previous:.2f}",
                  ha="center", fontsize=9, color=INK)
    axes.set_ylabel("previous-19 forgetting (mAP50 points)", fontsize=9)
    axes.set_ylim(0, max(values) * 1.28)
    save(figure, out, "fig1_forgetting_decomposition",
         f"{MEASURED.relative_to(ROOT)} — previous19_forgetting, previous19_map50",
         "One seed, one task step, predecessor protocol (600 regions). "
         "Not the current sequential benchmark.")


def figure_cost_is_not_supervision(out: Path) -> None:
    table = rows(LABELLING)
    names = [r["policy"] for r in table]
    per_unit = [float(r["supervision_per_oracle_unit"]) for r in table]
    cost = [float(r["cost_ratio_vs_budget"]) for r in table]
    half = [float(r["half_labelled_share"]) for r in table]

    figure, axes = frame("Equal annotation cost is not equal supervision")
    positions = range(len(names))
    axes.bar(positions, per_unit, color=[WARM, MUTED, COOL], width=0.55)
    for x, (unit, ratio, share) in enumerate(zip(per_unit, cost, half, strict=True)):
        axes.text(x, unit + 0.12,
                  f"{unit:.2f} per unit\ncost {ratio:.2f}x\nhalf-labelled {share:.1%}",
                  ha="center", fontsize=9, color=INK)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(names, fontsize=9)
    axes.set_ylabel("labelled objects per oracle unit", fontsize=9)
    axes.set_ylim(0, max(per_unit) * 1.32)
    save(figure, out, "fig2_cost_is_not_supervision",
         f"{LABELLING.relative_to(ROOT)} — supervision_per_oracle_unit, "
         "cost_ratio_vs_budget, half_labelled_share",
         "Accounting only, one seed, 306 images. known_plus_selected has never "
         "been run on the detector.")


def figure_acquisition_vs_learning(out: Path) -> None:
    table = [r for r in rows(MEASURED) if r["revealed_unknown_objects"]]
    figure, axes = frame("Finding unknown objects did not become new-class AP")
    for row in table:
        found = float(row["revealed_unknown_objects"])
        new = max(float(row["new_head_map50"]), float(row["new_medium_map50"]),
                  float(row["new_tail_map50"]))
        axes.scatter(found, new, s=54, color=COOL, zorder=3)
        axes.annotate(f"{row['arm']} b{row['budget']}", (found, new),
                      textcoords="offset points", xytext=(7, 4),
                      fontsize=7.5, color=MUTED)
    full = next(r for r in rows(MEASURED) if r["point"] == "full_t2_supervision")
    axes.axhline(float(full["new_head_map50"]), color=WARM, linewidth=1.1,
                 linestyle="--")
    axes.text(axes.get_xlim()[1], float(full["new_head_map50"]) + 0.6,
              f"full t2 supervision  {float(full['new_head_map50']):.1f}",
              ha="right", fontsize=8, color=WARM)
    axes.set_xscale("log")
    axes.set_xlabel("distinct unknown objects revealed by acquisition (log)", fontsize=9)
    axes.set_ylabel("best new-class AP50 of any band", fontsize=9)
    save(figure, out, "fig3_acquisition_vs_learning",
         f"{MEASURED.relative_to(ROOT)} — revealed_unknown_objects, new_*_map50",
         "Predecessor protocol, one seed per point. Descriptive: the points are "
         "different budgets and arms, not a controlled sweep.")


def figure_coherence_gate(out: Path) -> None:
    table = rows(COHERENCE)
    eps = [float(r["eps"]) for r in table]
    background = [100 * float(r["noise_share_background"]) for r in table]
    unknown = [100 * float(r["noise_share_unknown_object"]) for r in table]

    figure, axes = frame("The DBSCAN coherence gate rejects real unknowns harder "
                         "than background")
    axes.plot(eps, unknown, "o-", color=WARM, label="real unknown objects discarded")
    axes.plot(eps, background, "o-", color=COOL, label="background discarded")
    for x, y in zip(eps, unknown, strict=True):
        axes.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=8, color=WARM)
    axes.set_xlabel("DBSCAN eps", fontsize=9)
    axes.set_ylabel("share discarded as noise (%)", fontsize=9)
    axes.legend(frameon=False, fontsize=9)
    save(figure, out, "fig4_coherence_gate",
         f"{COHERENCE.relative_to(ROOT)} — noise_share_background, "
         "noise_share_unknown_object",
         "PROB decoder features only. Says nothing about a coherence gate on a "
         "different representation.")


def figure_batch_diversity(out: Path) -> None:
    table = rows(BATCH)
    seeds = sorted({r["seed"] for r in table})
    off = [float(next(r for r in table if r["seed"] == s and r["mu_batch"] == "0.0")
                 ["distinct_unknown_objects"]) for s in seeds]
    on = [float(next(r for r in table if r["seed"] == s and r["mu_batch"] == "0.3")
                ["distinct_unknown_objects"]) for s in seeds]

    figure, axes = frame("Batch diversity made the batch less redundant and found "
                         "fewer distinct objects")
    for index, (a, b) in enumerate(zip(off, on, strict=True)):
        axes.plot([0, 1], [a, b], "o-", color=COOL, alpha=0.85)
        axes.annotate(f"seed {seeds[index]}", (1, b), textcoords="offset points",
                      xytext=(8, -3), fontsize=8, color=MUTED)
    axes.set_xticks([0, 1])
    axes.set_xticklabels(["mu_batch = 0", "mu_batch = 0.3"], fontsize=9)
    axes.set_ylabel("distinct unknown objects in 600 picks", fontsize=9)
    save(figure, out, "fig5_batch_diversity",
         f"{BATCH.relative_to(ROOT)} — distinct_unknown_objects, mu_batch",
         "Region-level selection study on the frozen pool, three seeds. Never "
         "run downstream on the detector.")


def figure_rounds(out: Path) -> None:
    table = rows(ROUNDS)
    arms = ("entropy", "objectness", "consult", "prior_consult_batch")
    figure, axes = frame("Iterative rounds: a no-op for static scores, and not a "
                         "reliable gain for the rest")
    width = 0.2
    for offset, rounds in enumerate(("1", "6", "12")):
        heights, positions = [], []
        for index, arm in enumerate(arms):
            values = [float(r["unknown_objects"]) for r in table
                      if r["arm"] == arm and r["rounds"] == rounds]
            heights.append(sum(values) / len(values))
            positions.append(index + (offset - 1) * width)
        axes.bar(positions, heights, width=width,
                 color=[MUTED, COOL, WARM][offset],
                 label={"1": "600x1", "6": "6x100", "12": "12x50"}[rounds])
    axes.set_xticks(range(len(arms)))
    axes.set_xticklabels(arms, fontsize=8.5)
    axes.set_ylabel("distinct unknown objects (mean of 3 seeds)", fontsize=9)
    axes.legend(frameon=False, fontsize=9)
    save(figure, out, "fig6_rounds",
         f"{ROUNDS.relative_to(ROOT)} — arm, rounds, unknown_objects",
         "Means over three seeds hide a real disagreement: consult gains "
         "+36%/+7%/+8% and prior_consult_batch loses on all three. Show the "
         "per-seed numbers when asked.")


# ------------------------------------------------- benchmark, if supplied ---


def figure_benchmark(out: Path, path: Path) -> None:
    table = rows(path)
    field = next((f for f in ("new_class_AP50", "new_mAP50", "new_class_ap50")
                  if f in table[0]), None)
    if field is None:
        raise SystemExit(
            f"{path} has no new-class AP column; columns are {sorted(table[0])}")
    arms = ["random", "admissibility", "entropy"]
    figure, axes = frame("New-class AP50 per seed — three paired seeds, shown "
                         "per seed and never averaged across a disagreement")
    for index, arm in enumerate(arms):
        points = [(r["seed"], float(r[field])) for r in table if r["arm"] == arm]
        for seed, value in points:
            axes.scatter(index, value, s=52, color=COOL, zorder=3)
            axes.annotate(f"s{seed}", (index, value), textcoords="offset points",
                          xytext=(8, -3), fontsize=8, color=MUTED)
        if points:
            mean = sum(v for _, v in points) / len(points)
            axes.plot([index - 0.2, index + 0.2], [mean, mean], color=WARM, lw=2)
    axes.set_xticks(range(len(arms)))
    axes.set_xticklabels(arms, fontsize=9)
    axes.set_ylabel("mean new-class AP50 over t2-t4", fontsize=9)
    save(figure, out, "fig7_three_seed_new_ap",
         f"{path} — arm, seed, {field}",
         "Three seeds. A 3/3 sign agreement, not a significance test. Each seed "
         "changes acquisition and PROB's training seed together.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    parser.add_argument("--benchmark", type=Path,
                        help="the sequential benchmark's per-task metrics CSV, "
                             "if you have extracted it from Drive")
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)

    print(f"writing to {arguments.out}\n")
    for build in (figure_forgetting, figure_cost_is_not_supervision,
                  figure_acquisition_vs_learning, figure_coherence_gate,
                  figure_batch_diversity, figure_rounds):
        build(arguments.out)

    if arguments.benchmark and arguments.benchmark.is_file():
        figure_benchmark(arguments.out, arguments.benchmark)
    else:
        print("\n  fig7_three_seed_new_ap.png  SKIPPED — needs the benchmark's own\n"
              "      per-task metrics CSV, which is on Drive and not in this\n"
              "      repository. Pass --benchmark <path> once extracted. It is\n"
              "      NOT drawn from remembered numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
