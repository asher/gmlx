"""The command line's test paths, grouped by directory.

pytest 9.1 shows a conftest's fixtures only below the directory node it
was collected for, and collects a directory once for each run of
consecutive paths in it. With ``tests/serve/a.py tests/b.py
tests/serve/c.py`` the second ``tests/serve`` node sees no fixture of
``tests/serve/conftest.py``, its autouse ones included.
docs/upstream-defects.md has the entry. :func:`grouped` orders the paths
so each directory's are consecutive, keeping the order of first appearance
at every level.
"""
from __future__ import annotations

from pathlib import PurePath


def grouped(args: list[str]) -> list[str]:
    tree: dict = {}
    for i, arg in enumerate(args):
        node = tree
        for part in PurePath(arg.split("::", 1)[0]).parts[:-1]:
            node = node.setdefault(("dir", part), {})
        node[("arg", i)] = None
    out: list[str] = []

    def walk(node: dict) -> None:
        for (kind, key), sub in node.items():
            if kind == "arg":
                out.append(args[key])
            else:
                walk(sub)

    walk(tree)
    return out
