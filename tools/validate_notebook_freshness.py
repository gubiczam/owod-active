#!/usr/bin/env python3
"""Clean-room check that a Run-all notebook would survive a fresh Colab.

Unit tests pass against the *working tree*. A fresh Colab runs the **pinned
commit**, and the gap between those two is where "Run all works" turns out not
to. So this checks out the pin into a throwaway directory and exercises the
notebook against *that*, in a subprocess with nothing of this tree on the path.

What it actually executes:

1. the pin is a full 40-character SHA and resolves in this repository;
2. cell [1/10]'s parameter block, in an isolated namespace — it is pure
   constants and assertions, so it either passes or names what it wants;
3. the pinned checkout imports, and cell [3/10]'s API checks and registry
   assertions run against it. This is the cell that killed a Run all with a
   hardcoded ``arm_registry.ORDER`` tuple, and it is checked against the code
   the notebook will really import rather than the code sitting here;
4. the frozen session configuration read from the pinned modules;
5. every ``--flag`` in the long cell exists in the pinned tool's own ``--help``;
6. the keys the last cell reads exist in the contract the driver writes.

What it cannot do: mount Drive, clone PROB, build MSDeformAttn, or run the
detector. Those need Colab and are reported as such, never as passed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEEDS_COLAB = (
    "google.colab drive mount",
    "PROB clone, compat installs and the MSDeformAttn build",
    "the DINOv2 backbone download",
    "bridge.predict on a GPU",
)


class ValidationError(RuntimeError):
    pass


def cells(path: Path, kind: str = "code") -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in payload["cells"] if c["cell_type"] == kind]


def checkout(commit: str, into: Path) -> Path:
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", commit],
        capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(into)], input=archive, check=True)
    return into


def in_checkout(tree: Path, code: str) -> str:
    """Run a snippet with only the pinned tree importable."""

    done = subprocess.run(
        [sys.executable, "-c", code], cwd=str(tree), capture_output=True, text=True,
        env={"PYTHONPATH": str(tree), "PATH": "/usr/bin:/bin", "HOME": str(tree)},
        check=False,
    )
    if done.returncode != 0:
        raise ValidationError(done.stderr[-3000:] or done.stdout[-3000:])
    return done.stdout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("notebook", type=Path)
    arguments = parser.parse_args(argv)

    code = cells(arguments.notebook)
    parameters = code[0]
    print(f"notebook: {arguments.notebook}  ({len(code)} code cells)")

    # ---- 1. the pin ------------------------------------------------------
    match = re.search(r'OWL_COMMIT = "([0-9a-f]*)"', parameters)
    if not match or len(match.group(1)) != 40:
        raise ValidationError("OWL_COMMIT is not a full 40-character SHA")
    commit = match.group(1)
    subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
                   capture_output=True, check=True)
    print(f"[1] pin resolves: {commit}")

    # ---- 2. cell [1/10] in an isolated namespace -------------------------
    namespace: dict = {}
    exec(compile(parameters, "<cell 1>", "exec"), namespace)  # noqa: S102
    for name in ("OWL_COMMIT", "PROB_COMMIT", "DRIVE_ROOT", "RESULTS_RELATIVE",
                 "DATA_ROOT", "SESSION_STARTED"):
        if name not in namespace:
            raise ValidationError(f"cell [1/10] does not define {name}")
    print(f"[2] cell [1/10] executes; results -> {namespace['RESULTS_RELATIVE']}")

    with tempfile.TemporaryDirectory() as temporary:
        tree = checkout(commit, Path(temporary))
        if not (tree / "owl" / "active_selection" / "arms.py").is_file():
            raise ValidationError("the pinned checkout has no owl/active_selection")

        # ---- 3. cell [3/10]'s checks, against the PINNED code ------------
        body = code[2]
        start = body.index("from owl import bridge")
        end = body.index("OWL_SHA =")
        print(in_checkout(tree, body[start:end]).strip())
        print("[3] cell [3/10] API checks and registry assertions pass on the pin")

        # ---- 4. the frozen session configuration -------------------------
        frozen = json.loads(in_checkout(tree, (
            "import json;"
            "from owl.active_selection import diagnostic as d;"
            "print(json.dumps({'arms': list(d.ITERATIVE_ARMS),"
            " 'seeds': list(d.DIAGNOSTIC_SEEDS), 'rounds': d.ITERATIVE_ROUNDS,"
            " 'gates': {g.key: g.threshold for g in d.GATES}}))"
        )))
        expected_gates = {
            "jaccard_vs_entropy": 0.60, "tail_per_image_vs_entropy": 1.25,
            "tail_per_image_vs_cost_aware": 1.25,
            "background_share_vs_entropy": 0.10, "breadth_vs_entropy": 0.75,
            "t2_not_collapsed": 0.75,
        }
        if frozen["arms"] != ["entropy", "cost_aware", "distribution_aware_iterative_v1"]:
            raise ValidationError(f"arms are {frozen['arms']}")
        if frozen["seeds"] != [0, 1] or frozen["rounds"] != 6:
            raise ValidationError(f"seeds {frozen['seeds']}, rounds {frozen['rounds']}")
        if frozen["gates"] != expected_gates:
            raise ValidationError(f"gates moved: {frozen['gates']}")
        print(f"[4] frozen on the pin: arms {frozen['arms']}, "
              f"seeds {frozen['seeds']}, rounds {frozen['rounds']}, 6 gates unchanged")

        # ---- 5. every flag the long cell passes exists on the pin --------
        long_cell = next(c for c in code if "--rounds" in c)
        for tool, tail in re.findall(
                r'"tools"\s*/\s*"([a-z0-9_]+\.py)"(.*?)\]', long_cell, re.DOTALL):
            help_text = subprocess.run(
                [sys.executable, str(tree / "tools" / tool), "--help"],
                cwd=str(tree), capture_output=True, text=True, check=True).stdout
            missing = [f for f in re.findall(r'"(--[a-z0-9-]+)"', tail)
                       if f not in help_text]
            if missing:
                raise ValidationError(f"{tool} on the pin has no {missing}")
            print(f"[5] {tool}: every flag the notebook passes exists")

        # ---- 6. the last cell's output contract --------------------------
        final = code[-1]
        keys = set(re.findall(r'_verdict\["([a-z_]+)"\]', final)) | set(
            re.findall(r'_gate\["([a-z_]+)"\]', final))
        contract = json.loads(in_checkout(tree, (
            "import json;"
            "from owl.active_selection import diagnostic as d;"
            "v = d.Verdict(outcomes=(d._decide(d.GATES[0], {0: 0.1}),), go=False,"
            "              failed=('x',));"
            "payload = v.as_dict();"
            "print(json.dumps(sorted(payload) + sorted(payload['gates'][0])))"
        )))
        missing = sorted(keys - set(contract))
        if missing:
            raise ValidationError(f"the last cell reads {missing}, not in verdict.json")
        print(f"[6] cell [{len(code)}/{len(code)}] reads {sorted(keys)} — all present")

    print("\nCLEAN-ROOM VALIDATION PASSED for everything checkable off-GPU.")
    print("Still requires a Colab T4 and cannot be checked here:")
    for item in NEEDS_COLAB:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
