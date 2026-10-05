#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""List what an upstream release changes in the copy, for triage at a port.

The manifest (gmlx/_vendor/manifest.json) records every copied file with
the sha256 upstream shipped and its status. The report compares each
package of the copy with a new release of it (``--new mlx_vlm=DIR``, the
package directory) and, when given, the recorded release (``--old``). It
applies the codemod of scripts/vendor_upstream.py to both upstream texts
first, so the import rewrite never shows as a gmlx change.

A file-level summary comes first. It names at most ``LISTED`` files of
each change, and every file gmlx edits, aliases, extracts from or mirrors
beside them. Unit rows follow for the files gmlx edits (edited and alias
entries), the files gmlx imports from, and the entries of ``EXTRACTS``:
owned files that take units of an upstream file, and gmlx code that
mirrors upstream units beside the copy. A unit is a function, a method, a
property setter or deleter apart from its getter, the non-method body of
a class, or a named module assignment. Each row says who changed the
unit:

- where the new file has the recorded sha256, upstream changed nothing,
  and every difference is a gmlx change;
- otherwise the old file, checked against the recorded sha256, splits
  each unit into an upstream change, a gmlx change, or both. A verbatim
  or codemod file of the copy is the recorded release after the rewrite,
  so it serves as the old file when none is given. Without an old file
  the rows say only that the unit differs.

A unit an edited file imports from gmlx in place of its own body keeps
its row when upstream changes it, and the row names the gmlx file that
owns it. An old file that does not match the record is listed as
unverifiable, and the report goes on. The output is Markdown, and
``--json`` writes the same data. It exits 0 whatever it finds, and 2 on a
usage error.
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
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = Path("gmlx") / "_vendor"
PREFIX = "gmlx._vendor."


@dataclass(frozen=True)
class Extract:
    path: str             # the owned file, relative to the repository
    package: str          # the upstream import name
    source: str           # the upstream file, relative to the package
    # The upstream top-level names the owned file takes, or () for every
    # upstream name the owned file defines.
    names: tuple[str, ...] = ()
    # (upstream name, owned name) for a unit the owned file renames.
    renames: tuple[tuple[str, str], ...] = ()
    # True for gmlx code that mirrors upstream units beside the copy, which
    # keeps its own; False for owned bodies the copy imports in place of its
    # own.
    mirror: bool = False


VLM, LM = "mlx_vlm", "mlx_lm"  # gmlx: stock-name

# Owned files that take part of an upstream file, and gmlx code that
# mirrors upstream units. An alias entry of the manifest names each owned
# file that takes a whole one.
EXTRACTS = (
    Extract("gmlx/cache/attention.py", VLM, "models/base.py",
            ("create_attention_mask", "create_ssm_mask",
             "align_attention_mask_to_scores",
             "quantized_scaled_dot_product_attention",
             "_turboquant_attention_applies",
             "scaled_dot_product_attention")),
    Extract("gmlx/cache/attention.py", LM, "models/base.py",
            ("create_attention_mask", "create_ssm_mask",
             "quantized_scaled_dot_product_attention",
             "scaled_dot_product_attention")),
    Extract("gmlx/cache/kv_cache.py", LM, "models/base.py",
            ("create_causal_mask",)),
    Extract("gmlx/cache/kv_cache.py", LM, "models/cache.py",
            ("_BaseCache", "ArraysCache", "BatchKVCache",
             "BatchRotatingKVCache", "CacheList", "ChunkedKVCache",
             "ConcatenateKVCache", "KVCache", "QuantizedKVCache",
             "RotatingKVCache", "create_attention_mask", "dynamic_roll",
             "make_prompt_cache")),
    # The K3 conv ties its output to the carried state, which the copy's
    # does not.
    Extract("gmlx/models/kimi_k3.py", LM, "models/kimi_linear.py",
            ("ShortConv1d",), mirror=True),
    # gmlx builds its own APC manager from the env vars from_env reads.
    Extract("gmlx/cache/apc_manager.py", VLM, "apc.py", ("from_env",),
            (("from_env", "build_apc_manager"),), mirror=True),
)

# File changes, in the order they need a decision.
CHANGES = ("both", "upstream only", "removed upstream", "new upstream file",
           "gmlx only", "excluded", "unchanged")
# Unit changes, in the order they need a decision.
KINDS = ("upstream", "both", "upstream added", "upstream removed",
         "changed", "added or removed", "gmlx", "gmlx added", "gmlx removed")
# How many files of a change the Markdown names before it only counts.
LISTED = 25

_ALIAS = re.compile(r"^import (gmlx\.[\w.]+) as _owned$", re.MULTILINE)


class UsageError(Exception):
    pass


def _tool():
    """scripts/vendor_upstream.py, for its codemod, manifest and paths."""
    name = "_vendor_upstream"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parent / "vendor_upstream.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        body = body[1:]  # the module docstring
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
        elif b is None:
            rows[name] = "upstream added"
        else:
            # A unit the copy dropped still changes upstream: its owner,
            # the gmlx module the copy imports it from, has to take it.
            rows[name] = "upstream" if b != n else "gmlx removed"
    return rows


def owners(text: str, root: Path) -> dict[str, str]:
    """name -> the gmlx file it comes from, for each name a copied file
    imports from gmlx at module level in place of its own body."""
    out = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom) and node.level == 0
                and (node.module or "").split(".")[0] == "gmlx"
                and not node.module.startswith(PREFIX)):
            base = node.module.replace(".", "/")
            path = next((p for p in (f"{base}.py", f"{base}/__init__.py")
                         if (root / p).is_file()), f"{base}.py")
            for a in node.names:
                out[a.asname or a.name] = path
    return out


def package_dir(package: str) -> Path:
    """The directory of ``package`` in the installed stock distribution,
    found through the distribution's metadata. Nothing is imported, so
    neither the copy under gmlx._vendor nor an import hook stands in."""
    name = package.replace("_", "-")
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        raise UsageError(f"{name} is not installed: pass --new "
                         f"{package}=DIR") from None
    path = Path(str(dist.locate_file(package)))
    if not (path / "__init__.py").is_file():
        raise UsageError(f"{name} {dist.version} has no {package} package "
                         f"at {path}: pass --new {package}=DIR")
    return path


def package_version(package: str, pkg: Path) -> str:
    """The release a package directory holds, from its version file."""
    try:
        text = (pkg / _tool().VERSION_FILES[package]).read_text()
    except OSError:
        return "unknown"
    m = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", text)
    return m.group(1) if m else "unknown"


def imported_files(root: Path, files) -> set[str]:
    """The copied files gmlx code imports from or names in a module
    string, found by AST in every gmlx module outside the copy."""
    names: set[str] = set()
    for py in sorted((root / "gmlx").rglob("*.py")):
        if py.relative_to(root).parts[:2] == VENDOR.parts:
            continue
        try:
            tree = ast.parse(py.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom)
                    and (node.module or "").startswith(PREFIX)):
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                names.update(a.name for a in node.names
                             if a.name.startswith(PREFIX))
            elif (isinstance(node, ast.Constant)
                  and isinstance(node.value, str)
                  and node.value.startswith(PREFIX)):
                names.add(node.value.split(":")[0])
    out = set()
    for name in names:
        parts = name[len(PREFIX):].split(".")
        for n in range(len(parts), 0, -1):
            base = "/".join(parts[:n])
            hit = next((f for f in (f"{base}.py", f"{base}/__init__.py")
                        if f in files), None)
            if hit is not None:
                out.add(hit)
                break
    return out


def _read(pkg: Path, rel: str) -> bytes | None:
    """A file of a release, or None. A wheel ships the LICENSE in its
    dist-info beside the package directory, and a source checkout at its
    root, the package directory's parent."""
    path = pkg / rel
    if path.is_file():
        return path.read_bytes()
    if rel == "LICENSE":
        for info in sorted(pkg.parent.glob(f"{pkg.name}-*.dist-info")):
            for cand in (info / "licenses" / "LICENSE", info / "LICENSE"):
                if cand.is_file():
                    return cand.read_bytes()
        checkout = pkg.parent
        if any((checkout / m).exists()
               for m in (".git", "pyproject.toml", "setup.py")):
            cand = checkout / "LICENSE"
            if cand.is_file():
                return cand.read_bytes()
    return None


def _texts(new_pkg: Path, old_pkg: Path | None, rel: str, upstream: str):
    """The new text or None, the old text or None, and whether the old
    side is known: True when the new file or the old file has the recorded
    sha256, False with no old release, None when the old file is missing
    or is not the recorded one."""
    new = _read(new_pkg, rel)
    if new is not None and sha256(new) == upstream:
        return new, new, True
    if old_pkg is None:
        return new, None, False
    old = _read(old_pkg, rel)
    if old is not None and sha256(old) == upstream:
        return new, old, True
    return new, None, None


def _unit_rows(owned_text: str, new: bytes | None, old: bytes | None, *,
               ex: Extract | None = None, owned_by: dict | None = None,
               old_is_copy: bool = False) -> list[dict] | str:
    """The unit rows of one upstream file against its owned text. ``ex``
    restricts them to the units an extract or a mirror takes, under their
    upstream names. ``owned_by`` names the gmlx file of each unit the
    copied file imports in place of its body. ``old_is_copy``: the old text
    is the copy's own file, which the rewrite already holds."""
    rewrite = _tool().rewrite_text
    back = {own: up for up, own in (ex.renames if ex else ())}

    def upstream_unit(key, text):
        top, dot, rest = key.partition(".")
        if top not in back:
            return key, text
        if not dot:  # the def or class line carries the owned name
            text = re.sub(rf"\b(def|class)\s+{re.escape(top)}\b",
                          rf"\1 {back[top]}", text, count=1)
        return back[top] + dot + rest, text

    try:
        owned = dict(upstream_unit(k, v)
                     for k, v in units(owned_text).items())
        new_u = units(rewrite(new.decode())) if new is not None else {}
        old_u = (None if old is None else units(
            old.decode() if old_is_copy else rewrite(old.decode())))
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        return f"unparsable: {type(e).__name__}"
    if ex is not None:
        def tops(u):
            return {k.split(".")[0] for k in u if k != "<module>"}

        keep = (set(ex.names) if ex.names else
                tops(owned) & (tops(new_u) | tops(old_u or {})))

        def cut(u):
            return {k: v for k, v in u.items() if k.split(".")[0] in keep}

        owned, new_u = cut(owned), cut(new_u)
        old_u = None if old_u is None else cut(old_u)
    rows = []
    for k, v in sorted(classify(owned, new_u, old_u).items(),
                       key=lambda kv: (KINDS.index(kv[1]), kv[0])):
        row = {"unit": k, "change": v}
        owner = (owned_by or {}).get(k.split(".")[0])
        if owner is not None and k not in owned:
            row["owner"] = owner
        rows.append(row)
    return rows


def _copy_old(root: Path, package: str, rel: str, entry: dict,
              known) -> tuple[bytes | None, bool]:
    """The copy's own file as the old text, for a verbatim or codemod entry
    with no verified old release: it is that release after the rewrite."""
    if known is True or entry["status"] not in ("verbatim", "codemod"):
        return None, False
    path = root / VENDOR / package / rel
    return (path.read_bytes(), True) if path.is_file() else (None, False)


def _change(status: str, new: bytes | None, upstream: str) -> str:
    if status == "excluded":
        return "excluded"
    if new is None:
        return "removed upstream"
    edited = status in ("edited", "alias")
    if sha256(new) == upstream:
        return "gmlx only" if edited else "unchanged"
    return "both" if edited else "upstream only"


def report_package(package: str, new_pkg: Path, old_pkg: Path | None,
                   root: Path = ROOT) -> dict:
    manifest = _tool().read_manifest(root)
    prefix = f"{package}/"
    entries = {name[len(prefix):]: e for name, e in manifest["files"].items()
               if name.startswith(prefix)}
    imported = imported_files(root, set(manifest["files"]))
    known_word = {True: "verified", False: "not given", None: "unverifiable"}
    files, sections = [], []
    for rel, entry in sorted(entries.items()):
        status = entry["status"]
        new, old, known = _texts(new_pkg, old_pkg, rel, entry["upstream"])
        change = _change(status, new, entry["upstream"])
        files.append({"path": rel, "status": status, "change": change,
                      "old": known_word[known]})
        if (change in ("excluded", "unchanged") or not rel.endswith(".py")
                or (status not in ("edited", "alias")
                    and prefix + rel not in imported)):
            continue
        owned_path = root / VENDOR / package / rel
        if status == "alias":
            m = _ALIAS.search(owned_path.read_text())
            if m is not None:
                owned_path = root / (m.group(1).replace(".", "/") + ".py")
        owned_text = owned_path.read_text()
        copy_old, from_copy = _copy_old(root, package, rel, entry, known)
        sections.append({
            "path": rel, "status": status, "change": change,
            "owned": owned_path.relative_to(root).as_posix(),
            "split": known is True or from_copy,
            "rows": _unit_rows(
                owned_text, new, copy_old if from_copy else old,
                owned_by=owners(owned_text, root) if status == "edited"
                else None, old_is_copy=from_copy)})
    for ex in EXTRACTS:
        if ex.package != package or ex.source not in entries:
            continue
        entry = entries[ex.source]
        new, old, known = _texts(new_pkg, old_pkg, ex.source,
                                 entry["upstream"])
        copy_old, from_copy = _copy_old(root, package, ex.source, entry,
                                        known)
        sections.append({
            "path": ex.source, "status": "mirror" if ex.mirror else "extract",
            "owned": ex.path,
            "change": _change("edited", new, entry["upstream"]),
            "split": known is True or from_copy,
            "rows": _unit_rows((root / ex.path).read_text(), new,
                               copy_old if from_copy else old, ex=ex,
                               old_is_copy=from_copy)})
    shipped = {p.relative_to(new_pkg).as_posix()
               for p in new_pkg.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    for rel in sorted(shipped - set(entries)):
        files.append({"path": rel, "status": None,
                      "change": "new upstream file", "old": "not given"})
    return {
        "package": package,
        "copy": manifest["packages"].get(package, {}).get("version"),
        "new": {"dir": str(new_pkg),
                "version": package_version(package, new_pkg)},
        "old": (None if old_pkg is None else
                {"dir": str(old_pkg),
                 "version": package_version(package, old_pkg)}),
        "files": files,
        "units": sections,
    }


def markdown(results) -> str:
    out = ["# Upstream drift", ""]
    for r in results:
        old = r["old"]
        out += [f"## {r['package']}", "",
                f"The copy holds {r['copy']}. The new release is "
                f"{r['new']['version']} ({r['new']['dir']}). "
                + (f"The old release is {old['version']} ({old['dir']})."
                   if old else "No old release was given."), "",
                "| Change | Files |", "|---|---|"]
        by = {c: [f for f in r["files"] if f["change"] == c] for c in CHANGES}
        out += [f"| {c} | {len(by[c])} |" for c in CHANGES if by[c]]
        out.append("")
        # gmlx code that owns or mirrors units of a file, by upstream path.
        takers: dict[str, list[str]] = {}
        for s in r["units"]:
            if s["status"] in ("extract", "mirror"):
                takers.setdefault(s["path"], []).append(
                    f"{s['status']} {s['owned']}")
        for c in CHANGES[:4]:
            if not by[c]:
                continue
            # A file gmlx edits, aliases, extracts from or mirrors is always
            # named; the limit applies to the rest.
            pinned = [f for f in by[c] if f["status"] in ("edited", "alias")
                      or f["path"] in takers]
            rest = [f for f in by[c] if f not in pinned]
            shown = pinned + rest[:LISTED]
            out.append(f"{c.capitalize()}:")
            for f in by[c]:
                if f not in shown:
                    continue
                notes = ([f["status"]] if f["status"] else []) \
                    + takers.get(f["path"], [])
                out.append(f"- `{f['path']}`"
                           + (f" ({', '.join(notes)})" if notes else ""))
            if len(rest) > LISTED:
                out.append(f"- and {len(rest) - LISTED} more, listed in "
                           f"the --json output")
            out.append("")
        bad = [f for f in r["files"] if f["old"] == "unverifiable"]
        if bad:
            out.append("Unverifiable: the old file is missing or is not the "
                       "recorded one, so its rows do not say who changed a "
                       "unit.")
            out += [f"- `{f['path']}`" for f in bad]
            out.append("")
        for s in r["units"]:
            out += [f"### `{s['path']}` ({s['status']}, {s['change']})", "",
                    f"Owned text: `{s['owned']}`."
                    + ("" if s["split"] else " The rows are unsplit."), ""]
            if isinstance(s["rows"], str):
                out += [s["rows"], ""]
            elif not s["rows"]:
                out += ["No unit differs.", ""]
            elif any("owner" in row for row in s["rows"]):
                out += ["| Unit | Change | Owner |", "|---|---|---|"]
                out += [f"| `{row['unit']}` | {row['change']} | "
                        + (f"`{row['owner']}`" if "owner" in row else "")
                        + " |" for row in s["rows"]]
                out.append("")
            else:
                out += ["| Unit | Change |", "|---|---|"]
                out += [f"| `{row['unit']}` | {row['change']} |"
                        for row in s["rows"]]
                out.append("")
    return "\n".join(out)


def _dirs(values, flag) -> dict[str, Path]:
    out = {}
    roots = _tool().ROOTS
    for v in values or ():
        package, sep, path = v.partition("=")
        if not sep or package not in roots:
            raise UsageError(f"{flag} takes PACKAGE=DIR, PACKAGE one of "
                             f"{', '.join(roots)}: {v!r}")
        if not (Path(path) / "__init__.py").is_file():
            raise UsageError(f"{flag} {v}: not a package directory")
        out[package] = Path(path)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new", action="append", metavar="PACKAGE=DIR",
                    help="the package directory of a new release, per "
                         "package (default: the installed stock "
                         "distribution, when there is one)")
    ap.add_argument("--old", action="append", metavar="PACKAGE=DIR",
                    help="the package directory of the release the copy "
                         "was taken from, which splits upstream changes "
                         "from gmlx changes")
    ap.add_argument("--package", action="append", metavar="PACKAGE",
                    help="compare only this package (repeatable; default: "
                         "each package given --new, else both)")
    ap.add_argument("--json", type=Path, help="also write the report here")
    args = ap.parse_args(argv)
    try:
        new, old = _dirs(args.new, "--new"), _dirs(args.old, "--old")
        packages = args.package or sorted(new) or list(_tool().ROOTS)
        results = []
        for package in packages:
            if package not in _tool().ROOTS:
                raise UsageError(f"--package {package}: not a copied package")
            new_pkg = new.get(package) or package_dir(package)
            results.append(report_package(package, new_pkg,
                                          old.get(package)))
    except UsageError as e:
        print(f"drift_report: {e}", file=sys.stderr)
        return 2
    print(markdown(results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
