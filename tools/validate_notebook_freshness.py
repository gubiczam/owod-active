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


def _check_iterative_frozen(tree: Path) -> None:
    """The iterative diagnostic's frozen session, read off the pinned checkout."""

    frozen = json.loads(in_checkout(tree, (
        "import json;"
        "from owl.active_selection import diagnostic as d;"
        "print(json.dumps({'arms': list(d.ITERATIVE_ARMS),"
        " 'seeds': list(d.DIAGNOSTIC_SEEDS), 'rounds': d.ITERATIVE_ROUNDS,"
        " 'gates': {g.key: g.threshold for g in d.GATES}}))"
    )))
    expected = {
        "jaccard_vs_entropy": 0.60, "tail_per_image_vs_entropy": 1.25,
        "tail_per_image_vs_cost_aware": 1.25,
        "background_share_vs_entropy": 0.10, "breadth_vs_entropy": 0.75,
        "t2_not_collapsed": 0.75,
    }
    if frozen["arms"] != ["entropy", "cost_aware", "distribution_aware_iterative_v1"]:
        raise ValidationError(f"arms are {frozen['arms']}")
    if frozen["seeds"] != [0, 1] or frozen["rounds"] != 6:
        raise ValidationError(f"seeds {frozen['seeds']}, rounds {frozen['rounds']}")
    if frozen["gates"] != expected:
        raise ValidationError(f"gates moved: {frozen['gates']}")
    print(f"[4] frozen on the pin: arms {frozen['arms']}, seeds {frozen['seeds']}, "
          f"rounds {frozen['rounds']}, 6 gates unchanged")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--assume-pin", default=None,
                        help="validate against this revision instead of the "
                             "notebook's own OWL_COMMIT. For the one case where "
                             "a notebook cannot yet carry a real pin: the "
                             "commit that introduces it. Pass the SHA of that "
                             "commit; the notebook must still fail closed until "
                             "the pin is set for real.")
    arguments = parser.parse_args(argv)

    code = cells(arguments.notebook)
    parameters = code[0]
    print(f"notebook: {arguments.notebook}  ({len(code)} code cells)")

    # ---- 1. the pin ------------------------------------------------------
    match = re.search(r'OWL_COMMIT = "([^"]*)"', parameters)
    if match is None:
        raise ValidationError("the notebook has no OWL_COMMIT literal")
    declared = match.group(1)
    if arguments.assume_pin:
        commit = arguments.assume_pin
        if len(declared) == 40:
            raise ValidationError(
                f"--assume-pin was given but the notebook already pins "
                f"{declared}; validate the real pin instead")
        print(f"[1] pin NOT SET in the notebook ({declared!r}) — validating "
              f"against --assume-pin instead.\n"
              f"    The 40-character assertion in cell [1/10] keeps the notebook "
              f"failing closed until the pin is set for real.")
    else:
        if len(declared) != 40:
            raise ValidationError(
                f"OWL_COMMIT is {declared!r}, not a full 40-character SHA. If "
                "this is the commit that introduces the notebook, pass "
                "--assume-pin <sha>.")
        commit = declared
    subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
                   capture_output=True, check=True)
    print(f"[1] validating against revision: {commit}")

    # ---- 2. cell [1/10] in an isolated namespace -------------------------
    namespace: dict = {}
    # Under --assume-pin the cell is executed *as it will be once pinned*: the
    # placeholder is substituted first. Without this the cell's own
    # fail-closed assertion -- which is the behaviour we want it to have --
    # would stop us from validating anything after it.
    executable = (parameters.replace(declared, commit)
                  if arguments.assume_pin else parameters)
    exec(compile(executable, "<cell 1>", "exec"), namespace)  # noqa: S102
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
        # Cell [3/10] legitimately uses names cell [1/10] defined -- Run all
        # shares one namespace -- so carry the simple literals across, exactly
        # as the notebook would. Anything not a plain literal is left out and
        # will surface as a NameError, which is the correct signal.
        preamble = "".join(
            f"{name} = {value!r}\n" for name, value in namespace.items()
            if not name.startswith("_")
            and isinstance(value, (str, int, float, bool, tuple, list))
        )
        print(in_checkout(tree, preamble + body[start:end]).strip())
        print("[3] cell [3/10] API checks and registry assertions pass on the pin")

        # ---- 4. the frozen session configuration -------------------------
        #
        # Only meaningful for a notebook that declares the iterative diagnostic
        # session. A different notebook asserts its own frozen values inside
        # cell [3/10], which step 3 has already executed against the pin.
        if "ITERATIVE_ARMS" not in "".join(code):
            print("[4] skipped — this notebook declares no iterative session; "
                  "its frozen values were asserted in step 3")
        else:
            _check_iterative_frozen(tree)

        # ---- 5. every flag the notebook passes exists on the pin --------
        #
        # Every cell, not only the long one: the defect that started this was a
        # command built against an interface that did not exist, and the last
        # cell invokes two more tools of its own.
        invocations = re.findall(
            r'"tools"\s*/\s*"([a-z0-9_]+\.py)"(.*?)\]', "\n".join(code), re.DOTALL)
        if not invocations:
            raise ValidationError("no cell invokes a tool — nothing to validate")
        for tool, tail in invocations:
            path = tree / "tools" / tool
            if not path.is_file():
                raise ValidationError(f"{tool} does not exist on the pin")
            help_text = subprocess.run(
                [sys.executable, str(path), "--help"],
                cwd=str(tree), capture_output=True, text=True, check=True).stdout
            flags = re.findall(r'"(--[a-z0-9-]+)"', tail)
            missing = [f for f in flags if f not in help_text]
            if missing:
                raise ValidationError(f"{tool} on the pin has no {missing}")
            print(f"[5] {tool}: {len(flags)} flag(s) checked, all present")

        # ---- 6. the last cell's output contract --------------------------
        final = code[-1]
        reads = sorted(set(re.findall(
            r'_(?:verdict|manifest)\["([a-z_]+)"\]', final)))
        if not reads:
            # Say so rather than printing a pass over an empty set. This
            # notebook's last cell consumes summary.json, which only exists
            # after a real run, so its contract cannot be checked off-GPU.
            consumed = sorted(set(re.findall(r'_[a-z]+\["([a-z_]+)"\]', final)))
            print(f"[6] cell [{len(code)}/{len(code)}] reads {consumed} from a file "
                  "a real run produces — NOT checkable off-GPU, reported as such")
        else:
            produced = json.loads(in_checkout(tree, _CONTRACT_PROBE))
            missing = sorted(set(reads) - set(produced))
            if missing:
                raise ValidationError(
                    f"the last cell reads {missing}, absent from both the verdict "
                    f"and the manifest contract ({produced})")
            print(f"[6] cell [{len(code)}/{len(code)}] reads {reads} — all present")

    print("\nCLEAN-ROOM VALIDATION PASSED for everything checkable off-GPU.")
    print("Still requires a Colab T4 and cannot be checked here:")
    for item in NEEDS_COLAB:
        print(f"  - {item}")
    return 0


#: Every key a last cell may legitimately read, from both contracts this
#: repository writes: the diagnostic's verdict and the benchmark's manifest.
_CONTRACT_PROBE = (
    "import json;"
    "from owl.active_selection import diagnostic as d;"
    "from owl.active_selection import benchmark as b;"
    "v = d.Verdict(outcomes=(d._decide(d.GATES[0], {0: 0.1}),), go=False,"
    "              failed=('x',)).as_dict();"
    "m = b.manifest(trajectories=[], owl_commit='a'*40, prob_commit='b'*40,"
    "               prob_repository='x', checkpoint='c', checkpoint_sha256=None,"
    "               test_set='t', test_images=1);"
    "print(json.dumps(sorted(set(v) | set(v['gates'][0]) | set(m))))"
)


if __name__ == "__main__":
    raise SystemExit(main())
