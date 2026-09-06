#!/usr/bin/env python3
"""Close A1 from ``diagnostic_rows.csv``. Read-only; no selector, no detector.

**Per-class early purchase is exactly recoverable**, which the 2026-09-06 memo
did not realise. Nothing in the CSV is a per-class counter, but three of its
columns are linear in the quantities we want, and the system is determined.

Write ``FH2`` for fire hydrants on images opened at t2, ``SS2`` and ``SS3`` for
stop signs on images opened at t2 and t3. From
``tools/run_distribution_aware_diagnostic.oracle_row``:

* ``future_new_objects`` sums the classes declared at *later* tasks, so it is
  ``FH2 + SS2`` at t2, ``SS3`` at t3 and ``0`` at t4;
* ``banked_from_earlier`` is ``held[declared class]`` read **before** this task's
  purchases are added, so it is ``0`` at t2, ``FH2`` at t3 and ``SS2 + SS3`` at t4.

Hence

    FH2 = banked(t3)
    SS3 = future_new(t3)
    SS2 = banked(t4) - future_new(t3)

and the system is over-determined, which gives a free audit:

    future_new(t2)  ==  banked(t3) + banked(t4) - future_new(t3)

If that identity fails the file does not describe the run this arithmetic
assumes, and the tool refuses to report rather than printing a plausible number.

What it cannot do: split ``SS2`` further, attribute any object to an image, or
say anything about banking survival. The opened image identities were never
persisted, so no banking counterfactual is available from this file. That is
reported as unavailable, never estimated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TASKS = ("t2", "t3", "t4")
NUMERIC = ("images_opened", "answers_spent", "current_new_objects",
           "future_new_objects", "banked_from_earlier", "held_at_declaration",
           "tail_objects", "acquired_class_breadth")


class ForensicError(ValueError):
    """Raised when the file cannot support the arithmetic asked of it."""


def load(path: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ForensicError(f"{path} is empty")
    missing = [c for c in (*NUMERIC, "arm", "seed", "task") if c not in rows[0]]
    if missing:
        raise ForensicError(f"{path} has no {missing}; is it a diagnostic_rows.csv?")
    table: dict[tuple[str, int, str], dict[str, float]] = {}
    for row in rows:
        key = (row["arm"], int(row["seed"]), row["task"])
        if key in table:
            raise ForensicError(f"duplicate row for {key}")
        table[key] = {c: float(row[c]) for c in NUMERIC}
    return table


def early_purchases(table, arm: str, seed: int) -> dict[str, float]:
    """The per-class reconstruction, with its own consistency check."""

    try:
        rows = {t: table[(arm, seed, t)] for t in TASKS}
    except KeyError as error:
        raise ForensicError(f"{arm} seed {seed} is missing {error}") from error

    fh2 = rows["t3"]["banked_from_earlier"]
    ss3 = rows["t3"]["future_new_objects"]
    ss2 = rows["t4"]["banked_from_earlier"] - ss3

    predicted = fh2 + ss2
    observed = rows["t2"]["future_new_objects"]
    if abs(predicted - observed) > 1e-6:
        raise ForensicError(
            f"{arm} seed {seed}: future_new_objects at t2 is {observed:.0f} but "
            f"banked(t3) + banked(t4) - future_new(t3) is {predicted:.0f}. The "
            "file does not describe the run this arithmetic assumes; refusing "
            "to report a per-class split."
        )
    if ss2 < 0:
        raise ForensicError(f"{arm} seed {seed}: SS2 is negative ({ss2:.0f})")

    return {
        "fire_hydrant_at_t2": fh2,
        "stop_sign_at_t2": ss2,
        "stop_sign_at_t3": ss3,
        "future_objects_at_t2": observed,
        "future_objects_at_t3": ss3,
        "future_tail_bought_early": fh2 + ss2 + ss3,
        "images_opened_t2": rows["t2"]["images_opened"],
        "images_opened_t3": rows["t3"]["images_opened"],
        "images_opened_before_t4": rows["t2"]["images_opened"] + rows["t3"]["images_opened"],
        "answers_t2": rows["t2"]["answers_spent"],
        "answers_before_t4": rows["t2"]["answers_spent"] + rows["t3"]["answers_spent"],
        "held_t3": rows["t3"]["held_at_declaration"],
        "held_t4": rows["t4"]["held_at_declaration"],
    }


def report(path: Path, arms, seeds, as_json: bool) -> int:
    table = load(path)
    got = {(a, s): early_purchases(table, a, s) for a in arms for s in seeds}

    if as_json:
        print(json.dumps({f"{a}__seed{s}": v for (a, s), v in got.items()}, indent=2))
        return 0

    print("=" * 96)
    print("A1 — FUTURE-CLASS OBJECTS BOUGHT BEFORE THE CLASS WAS DECLARED")
    print("=" * 96)
    head = (f"{'arm':<24}{'seed':>5}{'FH@t2':>8}{'SS@t2':>8}{'SS@t3':>8}"
            f"{'early tail':>12}{'imgs':>7}{'per img':>9}{'per 1k ans':>12}")
    print(head); print("-" * len(head))
    for (arm, seed), v in got.items():
        images = v["images_opened_before_t4"]
        answers = v["answers_before_t4"]
        print(f"{arm:<24}{seed:>5}{v['fire_hydrant_at_t2']:>8.0f}"
              f"{v['stop_sign_at_t2']:>8.0f}{v['stop_sign_at_t3']:>8.0f}"
              f"{v['future_tail_bought_early']:>12.0f}{images:>7.0f}"
              f"{v['future_tail_bought_early'] / max(images, 1):>9.4f}"
              f"{1000 * v['future_tail_bought_early'] / max(answers, 1):>12.2f}")

    print()
    print("=" * 96)
    print("RATIO AGAINST entropy — reported per seed, never averaged across a disagreement")
    print("=" * 96)
    head = (f"{'arm':<24}{'seed':>5}{'raw':>10}{'per image':>12}{'per answer':>12}")
    print(head); print("-" * len(head))
    for arm in arms:
        if arm == "entropy":
            continue
        for seed in seeds:
            mine, theirs = got[(arm, seed)], got[("entropy", seed)]
            raw = mine["future_tail_bought_early"] / max(theirs["future_tail_bought_early"], 1e-9)
            per_i = ((mine["future_tail_bought_early"] / max(mine["images_opened_before_t4"], 1))
                     / max(theirs["future_tail_bought_early"] / max(theirs["images_opened_before_t4"], 1), 1e-9))
            per_a = ((mine["future_tail_bought_early"] / max(mine["answers_before_t4"], 1))
                     / max(theirs["future_tail_bought_early"] / max(theirs["answers_before_t4"], 1), 1e-9))
            print(f"{arm:<24}{seed:>5}{raw:>10.3f}{per_i:>12.3f}{per_a:>12.3f}")

    print()
    print("A2 — NOT RECOVERABLE FROM PERSISTED ARTEFACTS.")
    print("  Banking survival is a property of WHICH images were opened, and the")
    print("  opened image identities were never written to disk. No per-arm")
    print("  counterfactual is computed here and none is estimated.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--arms", nargs="+",
                        default=["entropy", "cost_aware", "distribution_aware_v1"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)
    return report(arguments.csv, arguments.arms, arguments.seeds, arguments.json)


if __name__ == "__main__":
    raise SystemExit(main())
