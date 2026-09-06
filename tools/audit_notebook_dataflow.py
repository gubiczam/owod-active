#!/usr/bin/env python3
"""Does a clean ``Runtime -> Run all`` define every name before it is used?

Static, read-only, no execution. It answers the one question a fresh-Colab
failure keeps asking in a different disguise: *which earlier cell was supposed
to produce this?* A `NameError` in cell nine at 3 a.m. is the same defect as an
`unrecognized arguments` in cell five — a consumer whose producer is not there.

What it does: walk the code cells in order, accumulate the names each one binds
(assignments, imports, defs, classes, with/for/except targets, comprehension and
lambda scopes handled), and report any name a cell *loads* that no earlier cell
bound and that is not a builtin. It also reports, per cell, which names it
consumes from earlier cells — which is the dependency table, computed rather
than written by hand.

It cannot see a file that does not exist on disk, only a name that does not
exist in the namespace. Artefact dependencies are checked by the notebook's own
assertions and by ``tests/test_distribution_aware.py``.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
from pathlib import Path

#: Names Colab or IPython provide that no cell binds.
AMBIENT = frozenset({"get_ipython", "display", "In", "Out", "exit", "quit", "__file__"})


class Scope(ast.NodeVisitor):
    """Names a module-level chunk binds, and names it loads from outside."""

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.loaded: set[str] = set()

    # -- bindings ----------------------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.bound.add(node.id)
        elif isinstance(node.ctx, ast.Load) and node.id not in self.bound:
            self.loaded.add(node.id)

    def visit_alias(self, node: ast.alias) -> None:
        name = node.asname or node.name.split(".")[0]
        self.bound.add(name)

    def _function(self, node) -> None:
        self.bound.add(node.name)
        # A function body runs later; treat its free names as loads, and its
        # parameters and locals as bound inside it only.
        inner = Scope()
        inner.bound |= {a.arg for a in node.args.args + node.args.kwonlyargs}
        if node.args.vararg:
            inner.bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            inner.bound.add(node.args.kwarg.arg)
        for statement in node.body:
            inner.visit(statement)
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(default)
        self.loaded |= inner.loaded - inner.bound - self.bound

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Lambda(self, node: ast.Lambda) -> None:
        inner = Scope()
        inner.bound |= {a.arg for a in node.args.args + node.args.kwonlyargs}
        inner.visit(node.body)
        self.loaded |= inner.loaded - inner.bound - self.bound

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # `except E as name:` binds `name` for the handler body. Without this the
        # audit reports every such target as undefined, and a real finding would
        # be lost in the noise.
        if node.name:
            self.bound.add(node.name)
        if node.type is not None:
            self.visit(node.type)
        for statement in node.body:
            self.visit(statement)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        for statement in node.body:
            self.visit(statement)
        for base in node.bases:
            self.visit(base)

    def _comprehension(self, node) -> None:
        inner = Scope()
        inner.bound |= self.bound
        for generator in node.generators:
            inner.visit(generator.iter)
            inner.visit(generator.target)
            for condition in generator.ifs:
                inner.visit(condition)
        for child in ast.iter_child_nodes(node):
            if child not in node.generators:
                inner.visit(child)
        self.loaded |= inner.loaded - self.bound

    visit_ListComp = _comprehension
    visit_SetComp = _comprehension
    visit_GeneratorExp = _comprehension
    visit_DictComp = _comprehension


def code_cells(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in payload["cells"] if c["cell_type"] == "code"]


def analyse(cells: list[str]) -> list[dict]:
    known: set[str] = set(dir(builtins)) | set(AMBIENT)
    produced: dict[str, int] = {}
    report: list[dict] = []
    for index, source in enumerate(cells, start=1):
        scope = Scope()
        for statement in ast.parse(source).body:
            scope.visit(statement)
        undefined = sorted(n for n in scope.loaded if n not in known)
        consumed = sorted({n: produced[n] for n in scope.loaded if n in produced})
        report.append({
            "cell": index,
            "title": source.splitlines()[0].strip("# ").strip(),
            "binds": sorted(scope.bound),
            "consumes": {n: produced[n] for n in consumed},
            "undefined": undefined,
        })
        for name in scope.bound:
            produced.setdefault(name, index)
        known |= scope.bound
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    report = analyse(code_cells(arguments.notebook))
    failures = [row for row in report if row["undefined"]]
    if not arguments.quiet:
        for row in report:
            print(f"\n[{row['cell']}] {row['title'][:70]}")
            if row["consumes"]:
                pairs = ", ".join(f"{n} <- [{c}]" for n, c in row["consumes"].items())
                print(f"    consumes: {pairs}")
            if row["undefined"]:
                print(f"    UNDEFINED: {row['undefined']}")
    if failures:
        print("\nFAIL — a clean Run all would raise NameError:")
        for row in failures:
            print(f"  cell [{row['cell']}]: {row['undefined']}")
        return 1
    print(f"\nPASS — {len(report)} cells, every name defined before it is used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
