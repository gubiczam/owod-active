"""Turn a finished GPU chain into the thesis's result chapter.

The chain writes ``<workspace>/<arm>/results_<arm>.csv`` next to its checkpoints
on Drive. Download that workspace and point this at it:

    python tools/analyze_chain.py /path/to/workspace
    python tools/analyze_chain.py /path/to/workspace --reference random --out out/

It prints, and writes into ``--out``:

1. the figure the research plan asks for — tail U-Recall against oracle cost,
   one line per arm;
2. the arms side by side at equal cost, to the depth every one of them reached;
3. the plan's prediction as a number: how much annotation each arm needed to
   reach the same tail level as the baseline;
4. a warning if the recorded ``oracle_cost_so_far`` overstates what the oracle
   was really asked (see :mod:`owl.analysis`).

Tables come out as CSV, Markdown and LaTeX. Nothing is hand-typed into the
thesis, and nothing here touches a running chain: it only reads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import analysis
from owl.runner import table

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "results" / "chain"


def _emit(rows, out: Path, stem: str, *, digits: int, caption: str | None = None) -> None:
    """One table, in the three forms the thesis needs."""

    if not rows:
        return
    analysis.write_csv(rows, out / f"{stem}.csv")
    (out / f"{stem}.md").write_text(analysis.to_markdown(rows, digits), encoding="utf-8")
    (out / f"{stem}.tex").write_text(
        analysis.to_latex(rows, digits, caption=caption, label=f"tab:{stem}"),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workspace", type=Path,
                        help="the chain's workspace, or a single results_<arm>.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"where the tables and the figure go (default: {DEFAULT_OUT})")
    parser.add_argument("--metric", default=analysis.HEADLINE,
                        help="the endpoint to compare (default: %(default)s)")
    parser.add_argument("--reference", default="random",
                        help="the arm the others are measured against (default: %(default)s)")
    parser.add_argument("--arms", default=None,
                        help="comma-separated subset of arms to read")
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the figure (matplotlib is an optional dependency)")
    args = parser.parse_args()

    try:
        arms = analysis.load_arms(args.workspace)
    except analysis.AnalysisError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.arms:
        wanted = [name.strip() for name in args.arms.split(",") if name.strip()]
        missing = [name for name in wanted if name not in arms]
        if missing:
            print(f"error: no such arm(s): {missing}; found {sorted(arms)}", file=sys.stderr)
            return 1
        arms = {name: arms[name] for name in wanted}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- what is actually on disk -----------------------------------------
    print(f"{'=' * 78}\nwhat the chain left behind\n{'=' * 78}")
    depth = analysis.depth_report(arms)
    print(table(depth, digits=args.digits))
    _emit(depth, out, "depth", digits=args.digits)
    partial = [row["arm"] for row in depth if row["tasks_dropped_from_comparison"]]
    if partial:
        print(f"\nPartial: {partial} ran longer than the shortest arm; the comparison "
              "stops\nwhere the shortest one stopped. Run all again to finish the rest.")

    # ---- is the recorded x-axis the real one? ------------------------------
    drift = analysis.cost_discrepancy(arms)
    if drift:
        print(f"\n{'=' * 78}\noracle_cost_so_far overstates the real cost\n{'=' * 78}")
        print(table(drift, digits=args.digits))
        print("\nThe recorded column is budget × task index; the pool ran dry, so fewer\n"
              "regions were actually bought. Every cost below is the real one — the\n"
              "cumulative sum of 'asked'. Quote that in the thesis, not the recorded one.")
        _emit(drift, out, "cost_discrepancy", digits=args.digits,
              caption="Recorded against real oracle cost")
    else:
        print("\nRecorded and real oracle cost agree: no task ran out of candidates.")

    # ---- per arm ------------------------------------------------------------
    for name, arm in arms.items():
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        rows = analysis.per_arm_table(arm, args.metric)
        print(table(rows, digits=args.digits))
        _emit(rows, out, f"arm_{name}", digits=args.digits,
              caption=f"The {name} arm, task by task")

    # ---- the comparison ------------------------------------------------------
    if len(arms) > 1:
        print(f"\n{'=' * 78}\n{args.metric} at equal oracle cost\n{'=' * 78}")
        rows = analysis.comparison(arms, args.metric)
        print(table(rows, digits=args.digits))
        _emit(rows, out, "comparison", digits=args.digits,
              caption=f"{args.metric} at equal oracle cost")

        if args.reference in arms:
            print(f"\n{'=' * 78}\nannotation needed to reach the same level as "
                  f"'{args.reference}'\n{'=' * 78}")
            rows = analysis.efficiency(arms, reference=args.reference, metric=args.metric)
            print(table(rows, digits=args.digits))
            print("\nLower is better: it is the oracle cost at which the arm first reaches\n"
                  "that level. '—' means the arm never reached it. A 'bounded' column set\n"
                  "means the arm's first measurement was already there, so its cost is an\n"
                  "upper bound — nothing was measured below it.")
            _emit(rows, out, "efficiency", digits=args.digits,
                  caption=f"Oracle cost to reach the tail level of '{args.reference}'")
        else:
            print(f"\nNo arm named '{args.reference}', so the efficiency table is skipped. "
                  f"Pass\n--reference with one of {sorted(arms)}.")
    else:
        print("\nOnly one arm. A single arm's numbers have nothing to be measured "
              "against;\nrun the baselines before quoting them.")

    # ---- the figure ----------------------------------------------------------
    if not args.no_plot:
        try:
            written = analysis.plot_curves(arms, out / "tail_recall_vs_cost",
                                           metric=args.metric)
            print("\nfigure: " + ", ".join(str(p) for p in written))
        except analysis.AnalysisError as error:
            print(f"\nno figure: {error}")

    print(f"\ntables written into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
