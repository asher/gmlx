#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""List what changed upstream in the code gmlx owns, for triage at a bump.

Each registered owned file names its upstream package and source file, the
functions it takes when it is an extract, and the baseline its vendor
script's ``--check`` recorded at vendor time. The report splits the owned
file and the new upstream source into units (functions, methods, each
property setter and deleter apart from its getter, the non-method body
of each class, named module assignments), applies the
baseline's vendor edits to the upstream text, and lists every unit that
differs. It uses the baseline to tell who changed each unit:

- when the new source has the baseline sha256, upstream changed nothing
  and every difference is a gmlx change;
- otherwise ``--old`` names the package directory of the baseline version,
  checked against the baseline sha256, and each unit is split into an
  upstream change, a gmlx change, or both.

The output is a Markdown list to triage, and ``--json`` writes the same
rows. It exits 0 whatever it finds, and 2 when an input does not match
its baseline.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Owned:
    path: str             # the owned file, relative to the repository
    package: str          # the upstream import name
    source: str           # the upstream file, relative to the package
    baseline: str         # the vendor-time baseline, relative to the repository
    extract: tuple[str, ...] = ()   # the upstream functions an extract takes


REGISTRY = (
    Owned("gmlx/cache/kv_cache.py", "mlx_vlm", "models/cache.py",
          "scripts/vendor_cache_baseline.json"),
    Owned("gmlx/cache/attention.py", "mlx_vlm", "models/base.py",
          "scripts/vendor_cache_baseline.json",
          extract=("create_attention_mask", "create_ssm_mask",
                   "align_attention_mask_to_scores",
                   "quantized_scaled_dot_product_attention",
                   "_turboquant_attention_applies",
                   "scaled_dot_product_attention")),
)

# Row order: what needs a decision first.
KINDS = ("upstream", "both", "upstream added", "upstream removed",
         "changed", "added or removed", "gmlx", "gmlx added", "gmlx removed")


class Mismatch(Exception):
    pass


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _method_key(node) -> str:
    """A property setter or deleter has its getter's name, so it keys as
    ``name.setter`` or ``name.deleter``."""
    for d in node.decorator_list:
        if (isinstance(d, ast.Attribute) and d.attr in ("setter", "deleter")
                and isinstance(d.value, ast.Name) and d.value.id == node.name):
            return f"{node.name}.{d.attr}"
    return node.name


def units(text: str) -> dict[str, str]:
    """name -> source text of each unit of a module."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)

    def seg(node) -> str:
        start = min([d.lineno for d in getattr(node, "decorator_list", [])]
                    + [node.lineno])
        return "".join(lines[start - 1:node.end_lineno])

    out: dict[str, str] = {}
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], "value", None), ast.Constant):
        body = body[1:]  # the module docstring is the vendor's own
    module = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = seg(node)
        elif isinstance(node, ast.ClassDef):
            start = min([d.lineno for d in node.decorator_list]
                        + [node.lineno])
            rest = ["".join(lines[start - 1:node.body[0].lineno - 1])]
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{_method_key(sub)}"] = seg(sub)
                else:
                    rest.append(seg(sub))
            out[node.name] = "".join(rest)
        elif (isinstance(node, ast.AnnAssign)
              and isinstance(node.target, ast.Name)):
            out[node.target.id] = seg(node)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            out[node.targets[0].id] = seg(node)
        else:
            module.append(seg(node))
    out["<module>"] = "".join(module)
    return out


def apply_edits(text: str, edits) -> tuple[str, list[str]]:
    """The text with each vendor edit that still applies, and the old
    strings of those that do not."""
    stale = []
    for e in edits:
        if text.count(e["old"]) == e["count"]:
            text = text.replace(e["old"], e["new"])
        else:
            stale.append(e["old"])
    return text, stale


def package_dir(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise Mismatch(f"{package} is not installed")
    return Path(list(spec.submodule_search_locations)[0])


def package_version(pkg: Path) -> str:
    m = re.search(r'__version__ = "([^"]+)"',
                  (pkg / "version.py").read_text())
    return m.group(1) if m else "unknown"


def _pick(u: dict[str, str], extract) -> dict[str, str]:
    if not extract:
        return u
    return {k: v for k, v in u.items() if k.split(".")[0] in extract}


def classify(owned: dict[str, str], new: dict[str, str],
             old: dict[str, str] | None) -> dict[str, str]:
    """unit -> kind for every unit that differs."""
    rows = {}
    for name in sorted(set(owned) | set(new) | set(old or ())):
        o, n = owned.get(name), new.get(name)
        if o == n:
            continue
        if old is None:
            rows[name] = "changed" if o is not None and n is not None \
                else "added or removed"
            continue
        b = old.get(name)
        if o is not None and n is not None:
            up, us = b != n, b != o
            rows[name] = ("both" if up and us else
                          "upstream" if up else "gmlx")
        elif n is None:
            rows[name] = "upstream removed" if b is not None else "gmlx added"
        else:
            rows[name] = "gmlx removed" if b is not None else "upstream added"
    return rows


def report(entry: Owned, new_pkg: Path, old_pkg: Path | None,
           root: Path = ROOT) -> dict:
    base = json.loads((root / entry.baseline).read_text())
    src = base["sources"][entry.source]
    edits = base["outputs"][entry.path]["edits"]
    new_text = (new_pkg / entry.source).read_text()
    new_sha = sha256(new_text)
    old_text = None
    if new_sha == src["sha256"]:
        old_text = new_text
    elif old_pkg is not None:
        old_text = (old_pkg / entry.source).read_text()
        if sha256(old_text) != src["sha256"]:
            raise Mismatch(f"{old_pkg / entry.source} is not the baseline "
                           f"{base['version']} source")
    new_edited, stale = apply_edits(new_text, edits)
    old_units = None
    if old_text is not None:
        old_edited, old_stale = apply_edits(old_text, edits)
        if old_stale:
            raise Mismatch(f"{entry.path}: a baseline edit does not apply "
                           "to the baseline source")
        old_units = _pick(units(old_edited), entry.extract)
    owned_units = units((root / entry.path).read_text())
    new_units = _pick(units(new_edited), entry.extract)
    if entry.extract:
        owned_units.pop("<module>", None)
        new_units.pop("<module>", None)
        if old_units is not None:
            old_units.pop("<module>", None)
    rows = classify(owned_units, new_units, old_units)
    return {
        "path": entry.path,
        "package": entry.package,
        "source": entry.source,
        "baseline": {"version": base["version"], "sha256": src["sha256"]},
        "new": {"version": package_version(new_pkg), "sha256": new_sha,
                "unchanged": new_sha == src["sha256"]},
        "split": old_units is not None,
        "stale_edits": stale,
        "rows": [{"unit": k, "change": v} for k, v in
                 sorted(rows.items(), key=lambda kv: (KINDS.index(kv[1]),
                                                      kv[0]))],
    }


def markdown(results) -> str:
    out = ["# Owned code drift", ""]
    for r in results:
        b, n = r["baseline"], r["new"]
        out += [f"## {r['path']}", "",
                f"{r['package']} {r['source']}: baseline {b['version']} "
                f"({b['sha256'][:12]}), new {n['version']} "
                f"({n['sha256'][:12]}, "
                f"{'unchanged' if n['unchanged'] else 'changed'}).", ""]
        if not r["split"]:
            out += ["Upstream changed and no --old source was given, so "
                    "the rows do not say who changed each unit.", ""]
        for old in r["stale_edits"]:
            out.append(f"Vendor edit no longer applies: {old!r}")
        if r["stale_edits"]:
            out.append("")
        if not r["rows"]:
            out += ["No unit differs.", ""]
            continue
        out += ["| Unit | Change |", "|---|---|"]
        out += [f"| `{row['unit']}` | {row['change']} |" for row in r["rows"]]
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new", type=Path,
                    help="upstream package directory to compare against "
                         "(default: the installed package)")
    ap.add_argument("--old", type=Path,
                    help="package directory of the baseline version, which "
                         "splits upstream changes from gmlx changes")
    ap.add_argument("--json", type=Path, help="also write the rows here")
    args = ap.parse_args()
    results = []
    try:
        for entry in REGISTRY:
            new_pkg = args.new or package_dir(entry.package)
            results.append(report(entry, new_pkg, args.old))
    except Mismatch as e:
        print(f"drift_report: {e}", file=sys.stderr)
        return 2
    print(markdown(results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
