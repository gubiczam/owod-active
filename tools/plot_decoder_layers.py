"""Four plots for the decoder-layer audit. No decorative figures.

Each answers one question the tables answer less legibly:

1. semantic quality against depth -- the thing the experiment is about;
2. query-index nuisance against depth -- how much of the space is "which query
   fired" rather than "what is in the box";
3. objectness entanglement against depth -- ``pred_obj`` is
   ``||BatchNorm(hs[lvl])||^2``, so this is how much of the representation is the
   objectness objective;
4. unknown kNN and NMI against depth, with the frozen decision thresholds drawn,
   so a reader can see whether a layer passes rather than being told.

Plots are diagnostic, not evidence. The CSVs are the evidence.
"""

from __future__ import annotations

import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.audit_decoder_layers import (
    DECISION_POPULATION,
    DECISION_REPRESENTATION,
    GO_OPEN_POOL_KNN,
    GO_UNKNOWN_KNN,
    POPULATIONS,
)

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"


def read(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} missing; run tools/audit_decoder_layers.py first")
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def mean_by_layer(rows: list[dict], field: str, **filters) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        value = row.get(field)
        if isinstance(value, float) and not math.isnan(value):
            grouped[int(row["layer"])].append(value)
    layers = sorted(grouped)
    return layers, [st.mean(grouped[layer]) for layer in layers]


def _finish(axis, title: str, ylabel: str) -> None:
    axis.set_xlabel("decoder layer (5 = the one PROB exports)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(fontsize=8)


def main() -> None:
    nuisance = read("decoder_layer_representation.csv")
    semantic = read("decoder_layer_population.csv")

    # 1 -- semantic quality against depth, one line per population
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for population in POPULATIONS:
        layers, values = mean_by_layer(
            semantic, "unknown_knn",
            representation=DECISION_REPRESENTATION, population=population)
        if layers:
            axis.plot(layers, values, marker="o", label=population)
    axis.axhline(GO_UNKNOWN_KNN, linestyle="--", color="crimson",
                 label=f"GO threshold {GO_UNKNOWN_KNN}")
    axis.axhline(0.0564, linestyle=":", color="grey", label="chance 0.056")
    _finish(axis, f"Unknown-class kNN agreement ({DECISION_REPRESENTATION})",
            "kNN class agreement, same object excluded")
    figure.tight_layout()
    figure.savefig(RESULTS / "decoder_layer_semantic.png", dpi=150)
    plt.close(figure)

    # 2 -- query-index nuisance against depth
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for field, label in (("eta2_query_index", "query index (nuisance)"),
                         ("eta2_known_class", "known class (signal)"),
                         ("eta2_unknown_class", "unknown class (signal)")):
        layers, values = mean_by_layer(nuisance, field,
                                       representation=DECISION_REPRESENTATION)
        if layers:
            axis.plot(layers, values, marker="o", label=label)
    _finish(axis, "What the space encodes, by depth", "variance explained (eta squared)")
    figure.tight_layout()
    figure.savefig(RESULTS / "decoder_layer_nuisance.png", dpi=150)
    plt.close(figure)

    # 3 -- objectness entanglement against depth
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for field, label in (("pc1_variance", "PC1 share of variance"),
                         ("rho_pc1_pred_obj", "rho(PC1, pred_obj)"),
                         ("rho_pc1_embedding_norm", "rho(PC1, embedding norm)")):
        layers, values = mean_by_layer(nuisance, field,
                                       representation=DECISION_REPRESENTATION)
        if layers:
            axis.plot(layers, values, marker="o", label=label)
    axis.axhline(0.0, linewidth=0.8, color="black")
    _finish(axis, "Objectness entanglement by depth\n(pred_obj = ||BatchNorm(hs[l])||^2)",
            "value")
    figure.tight_layout()
    figure.savefig(RESULTS / "decoder_layer_objectness.png", dpi=150)
    plt.close(figure)

    # 4 -- the decision view: both GO metrics plus NMI, on the decision population
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    filters = {"representation": DECISION_REPRESENTATION, "population": DECISION_POPULATION}
    for field, label, threshold in (
        ("unknown_knn", "unknown kNN", GO_UNKNOWN_KNN),
        ("open_pool_unknown_knn", "open-pool unknown kNN", GO_OPEN_POOL_KNN),
        ("unknown_nmi", "unknown NMI", None),
    ):
        layers, values = mean_by_layer(semantic, field, **filters)
        if not layers:
            continue
        line = axis.plot(layers, values, marker="o", label=label)[0]
        if threshold is not None:
            axis.axhline(threshold, linestyle="--", linewidth=0.9,
                         color=line.get_color(), alpha=0.6)
    _finish(axis, f"Frozen decision metrics on {DECISION_POPULATION}\n"
                  "(dashed = GO thresholds)", "value")
    figure.tight_layout()
    figure.savefig(RESULTS / "decoder_layer_decision.png", dpi=150)
    plt.close(figure)

    for name in ("semantic", "nuisance", "objectness", "decision"):
        print(f"wrote {RESULTS / f'decoder_layer_{name}.png'}")


if __name__ == "__main__":
    main()
