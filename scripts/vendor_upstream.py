#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# vendor_upstream: skip
"""Copy mlx-vlm and mlx-lm into gmlx/_vendor and keep the copy honest.

gmlx carries both packages whole under ``gmlx/_vendor``, imported as
``gmlx._vendor.mlx_vlm`` and ``gmlx._vendor.mlx_lm``. The copy is upstream's
files with only their module references rewritten, and from then on it is
gmlx code. ``gmlx/_vendor/manifest.json`` records every copied file with the
sha256 upstream shipped, a status and, where the file in the tree differs,
its sha256.

Subcommands:

  copy SRC       Copy both packages from SRC, a site-packages directory or a
                 directory of wheels. Every file must match the sha256 in its
                 distribution's RECORD. Writes the files, each package's
                 LICENSE and the manifest. With --packages, the packages
                 copied replace their entries in an existing manifest.
  rewrite PATH   Rewrite module references under each PATH in place (by
                 default gmlx, tests, scripts and bench). With --check,
                 change nothing and exit 1 if a file would change.
  reverse FILE   Print a copied file as upstream shipped it.
  manifest       Recompute each file's status and sha256 from the tree.
  verify         Check the tree against the manifest. Exits 1 on a mismatch.

Rewrite rules. They work on tokens, so no line is reformatted.

  R1  import heads at level 0: ``import mlx_vlm.x [as y]`` and
      ``from mlx_vlm.x import y``. Relative imports stay as they are.
  R2  a bare use of the root name bound by ``import mlx_vlm.x`` without an
      alias.
  R3  a string literal whose whole value is a module path at a root, with an
      optional ``:attr`` or a trailing dot (importlib, sys.modules,
      find_spec, logger names).
  R4  the leading literal of an f-string that starts with a root and a dot.
  R5  a raw or escape-free string that parses as Python and imports a root,
      such as a script a test passes to a child process. R1 to R4 apply
      inside it.

Comments, docstrings, messages, URLs and paths that name a root are left as
they are and listed in the report. So are lines that compare module names
by root or prefix, and every ``startswith`` test on a string at a root,
since a rewrite can make such a check vacuous. A file with a
``# vendor_upstream: skip`` comment line is never rewritten. A line with a
``# gmlx: stock-name`` comment keeps its names as they are, for code that
must name a stock module, such as a reader of names an earlier release
recorded; verify accepts such a line and no other.

The reverse is the removal of the ``gmlx._vendor.`` prefix before a root. It
is exact because no upstream file contains that prefix; ``copy`` checks.

Statuses: ``verbatim`` (upstream bytes), ``codemod`` (upstream bytes after
the rewrite), ``edited`` (changed by hand; carries a ``# gmlx:`` comment),
``alias`` (replaced by a module that aliases an owned gmlx module; carries
a ``# gmlx:`` comment), ``excluded`` (not in the tree) and ``license``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import bisect
import collections
import csv
import hashlib
import io
import json
import re
import sys
import tokenize
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = Path("gmlx") / "_vendor"
MANIFEST = VENDOR / "manifest.json"
PREFIX = "gmlx._vendor."
ROOTS = ("mlx_vlm", "mlx_lm")
DISTS = {"mlx_vlm": "mlx_vlm", "mlx_lm": "mlx_lm"}
TARGET = {r: PREFIX + r for r in ROOTS}
SKIP_PRAGMA = re.compile(r"^#\s*vendor_upstream:\s*skip\s*$", re.MULTILINE)
EDIT_MARKER = re.compile(r"#\s*gmlx:")
STOCK_NAME = re.compile(r"#\s*gmlx:\s*stock-name\b")
VERSION_FILES = {"mlx_vlm": "version.py", "mlx_lm": "_version.py"}
REWRITE_PATHS = ("gmlx", "tests", "scripts", "bench")
# Directories under gmlx/_vendor that this tool does not copy or verify.
OTHER_VENDORED = ("misaki",)

_MODPATH = re.compile(
    r"^(?P<root>mlx_vlm|mlx_lm)(?:\.[A-Za-z_]\w*)*\.?(?::[A-Za-z_][\w.]*)?$")
_WORD = re.compile(r"(?<![\w.])(mlx_vlm|mlx_lm)\b")
_DEPTH = re.compile(r"__module__|__name__|__package__|__spec__|__qualname__")
_SPLIT_INDEX = re.compile(
    r"\.(?:r?split|partition|rpartition)\(\s*['\"]\.['\"]")
_COMPARE = re.compile(
    r"__module__\s*(==|!=|in\b)|__module__\.startswith|__name__\.startswith")
_ROOT_PREFIX = re.compile(r"^(mlx_vlm|mlx_lm)(\.|$)")
_EMBEDDED_IMPORT = re.compile(
    r"(^|\n)\s*(import|from)\s+(mlx_vlm|mlx_lm)\b"
    r"|import_module\(\s*[\"'](mlx_vlm|mlx_lm)")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- rewrite --------------------------------------------------------------


class _Edits:
    """Replacements by (row, col) span, applied back to front."""

    def __init__(self, text: str):
        self.text = text
        self.offsets = [0]
        for line in text.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.edits: set[tuple[int, int, str]] = set()

    def add(self, start, end, repl):
        s = self.offsets[start[0] - 1] + start[1]
        e = self.offsets[end[0] - 1] + end[1]
        self.edits.add((s, e, repl))

    def apply(self) -> str:
        out = self.text
        for s, e, r in sorted(self.edits, reverse=True):
            out = out[:s] + r + out[e:]
        return out


def _string_prefix(tok: str) -> tuple[str, str]:
    i = 0
    while i < len(tok) and tok[i] not in "'\"":
        i += 1
    q = tok[i:i + 3] if tok[i:i + 3] in ('"""', "'''") else tok[i]
    return tok[:i], q


def _embedded_code(tok: str, prefix: str, q: str, val: str) -> bool:
    raw = tok[len(prefix) + len(q):len(tok) - len(q)]
    if "r" not in prefix.lower() and raw != val:
        return False  # escapes in a cooked string: a hand edit
    if not _EMBEDDED_IMPORT.search(raw):
        return False
    try:
        ast.parse(raw)
    except SyntaxError:
        return False
    return True


def analyze(text: str, path: str = "<text>", *, apply: bool = True):
    """Return (new text, rule counts, mentions left, review lines)."""
    counts: collections.Counter = collections.Counter()
    left: list[dict] = []
    review: list[dict] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return text, {"error": f"syntax: {e}"}, left, review
    edits = _Edits(text)
    src_lines = text.splitlines(keepends=True)
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    starts = [t.start for t in toks]
    # Lines whose names stay stock: every token on them is left as it is.
    stock_lines = {t.start[0] for t in toks if t.type == tokenize.COMMENT
                   and STOCK_NAME.search(t.string)}

    def cpos(lineno: int, col: int) -> tuple[int, int]:
        # ast columns count UTF-8 bytes; tokenize columns count characters.
        line = src_lines[lineno - 1] if lineno - 1 < len(src_lines) else ""
        return lineno, len(line.encode()[:col].decode("utf-8", "replace"))

    imports: list[tuple[str, ast.AST]] = []
    bound: set[str] = set()
    names = {}
    docstrings = set()
    loggers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute) and node.func.attr == "startswith":
            args = [e for a in node.args
                    for e in (a.elts if isinstance(a, ast.Tuple) else [a])]
            if any(isinstance(a, ast.Constant) and isinstance(a.value, str)
                   and _ROOT_PREFIX.match(a.value) for a in args):
                review.append({"path": path, "line": node.lineno,
                               "kind": "startswith test on a root string"})
        if getattr(node, "lineno", None) in stock_lines:
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and \
                    node.module.split(".")[0] in ROOTS:
                imports.append(("from", node))
                counts[f"R1 from {node.module.split('.')[0]} import"] += 1
        elif isinstance(node, ast.Import):
            heads = [a for a in node.names if a.name.split(".")[0] in ROOTS]
            if heads:
                imports.append(("import", node))
            for a in heads:
                root = a.name.split(".")[0]
                if a.asname is None:
                    bound.add(root)
                    counts[f"R1 import {root}"] += 1
                else:
                    counts[f"R1 import {root} as"] += 1
        elif isinstance(node, ast.Name) and node.id in ROOTS:
            names[cpos(node.lineno, node.col_offset)] = node
        elif isinstance(node, ast.arg) and node.arg in ROOTS:
            review.append({"path": path, "line": node.lineno,
                           "kind": "parameter named like a root"})
        elif isinstance(node, ast.alias) and node.asname in ROOTS:
            review.append({"path": path, "line": node.lineno,
                           "kind": "alias named like a root"})
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    body[0].value, ast.Constant) and isinstance(
                    body[0].value.value, str):
                docstrings.add(cpos(body[0].value.lineno,
                                    body[0].value.col_offset))
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(
                f, "id", None)
            if fname == "getLogger" and node.args and isinstance(
                    node.args[0], ast.Constant):
                loggers.add(cpos(node.args[0].lineno, node.args[0].col_offset))

    # R1: the head name of each import clause.
    for kind, node in imports:
        lo = bisect.bisect_left(starts, cpos(node.lineno, node.col_offset))
        hi = bisect.bisect_right(starts, cpos(node.end_lineno,
                                              node.end_col_offset))
        want = False
        for t in toks[lo:hi]:
            if t.type in (tokenize.NL, tokenize.COMMENT):
                continue
            if t.type == tokenize.NAME and t.string in ("import", "from"):
                if kind == "from" and t.string == "import":
                    break
                want = True
                continue
            if kind == "import" and t.type == tokenize.OP and t.string == ",":
                want = True
                continue
            if t.type == tokenize.NAME and want:
                if t.string in ROOTS:
                    edits.add(t.start, t.end, TARGET[t.string])
                want = False

    # R2: bare names bound by a no-alias import.
    for pos, node in names.items():
        if isinstance(node.ctx, ast.Load) and node.id in bound:
            edits.add(pos, (pos[0], pos[1] + len(node.id)), TARGET[node.id])
            counts[f"R2 bare name {node.id}"] += 1
        else:
            review.append({"path": path, "line": pos[0],
                           "kind": f"name {node.id} not bound by a no-alias "
                                   f"import ({type(node.ctx).__name__})"})

    fstring_start = False
    fs_start = getattr(tokenize, "FSTRING_START", -1)
    fs_middle = getattr(tokenize, "FSTRING_MIDDLE", -1)
    for t in toks:
        tt, ts = t.type, t.string
        if t.start[0] in stock_lines:
            if tt != tokenize.COMMENT and _WORD.search(ts):
                counts["left stock-name line"] += 1
            fstring_start = False
            continue
        if tt == tokenize.STRING:
            prefix, q = _string_prefix(ts)
            body = len(prefix) + len(q)
            if "b" in prefix.lower():
                pass
            elif "f" in prefix.lower():
                # An f-string token before Python 3.12.
                m = re.match(r"(mlx_vlm|mlx_lm)\.", ts[body:])
                if m:
                    s = (t.start[0], t.start[1] + body)
                    edits.add(s, (s[0], s[1] + len(m.group(1))),
                              TARGET[m.group(1)])
                    counts["R4 f-string head"] += 1
                elif _WORD.search(ts):
                    counts["left f-string"] += 1
                    left.append({"path": path, "line": t.start[0],
                                 "kind": "f-string", "text": ts[:120]})
            else:
                try:
                    val = ast.literal_eval(ts)
                except Exception:  # noqa: S110 - not a plain literal; skipped
                    val = None
                m = _MODPATH.match(val) if isinstance(val, str) else None
                if m and ts[body:body + len(m.group("root"))] == m.group("root"):
                    root = m.group("root")
                    kind = ("R3 logger name" if t.start in loggers
                            else "R3 bare root string" if val == root
                            else "R3 module-path string")
                    s = (t.start[0], t.start[1] + body)
                    edits.add(s, (s[0], s[1] + len(root)), TARGET[root])
                    counts[kind] += 1
                    if val == root:
                        review.append({
                            "path": path, "line": t.start[0], "text": ts,
                            "kind": "bare root string rewritten; check root "
                                    "comparisons"})
                elif isinstance(val, str) and _WORD.search(val):
                    if _embedded_code(ts, prefix, q, val):
                        raw = ts[body:len(ts) - len(q)]
                        new_raw, sub, _, _ = analyze(raw, path + ":<embedded>")
                        if "error" not in sub and new_raw != raw:
                            edits.add((t.start[0], t.start[1] + body),
                                      (t.end[0], t.end[1] - len(q)), new_raw)
                            counts["R5 embedded code string"] += 1
                            fstring_start = False
                            continue
                    cat = ("docstring" if t.start in docstrings else
                           "path-like" if re.search(r"(mlx_vlm|mlx_lm)/", val)
                           else "message or other")
                    counts[f"left {cat}"] += 1
                    left.append({"path": path, "line": t.start[0],
                                 "kind": cat, "text": val[:120]})
            fstring_start = False
        elif tt == fs_start:
            fstring_start = True
        elif tt == fs_middle:
            m = re.match(r"(mlx_vlm|mlx_lm)\.", ts)
            if fstring_start and m:
                edits.add(t.start, (t.start[0], t.start[1] + len(m.group(1))),
                          TARGET[m.group(1)])
                counts["R4 f-string head"] += 1
            elif _WORD.search(ts):
                counts["left f-string"] += 1
                left.append({"path": path, "line": t.start[0],
                             "kind": "f-string", "text": ts[:120]})
            fstring_start = False
        elif tt == tokenize.COMMENT:
            if _WORD.search(ts):
                counts["left comment"] += 1
        elif tt != tokenize.NL:
            fstring_start = False

    for n, line in enumerate(text.splitlines(), 1):
        if _DEPTH.search(line) and _SPLIT_INDEX.search(line):
            review.append({"path": path, "line": n, "kind": "module-name split",
                           "text": line.strip()[:140]})
        if _COMPARE.search(line):
            review.append({"path": path, "line": n,
                           "kind": "module-name comparison",
                           "text": line.strip()[:140]})

    new = edits.apply() if apply else text
    if new != text:
        try:
            ast.parse(new)
        except SyntaxError as e:
            return text, {"error": f"rewrite broke syntax: {e}"}, left, review
    return new, dict(counts), left, review


def rewrite_text(text: str) -> str:
    """The forward rewrite of one file's text."""
    if SKIP_PRAGMA.search(text) or not _WORD.search(text):
        return text
    new, counts, _, _ = analyze(text)
    if "error" in counts:
        raise ValueError(counts["error"])
    return new


def reverse_text(text: str) -> str:
    for root in ROOTS:
        text = text.replace(PREFIX + root, root)
    return text


def _py_files(top: Path):
    if top.is_file():
        yield top
        return
    for p in sorted(top.rglob("*.py")):
        if "__pycache__" not in p.parts:
            yield p


def cmd_rewrite(args) -> int:
    base = Path(args.root).resolve()
    totals: collections.Counter = collections.Counter()
    changed, skipped, errors = [], [], []
    left, review, per_file = [], [], {}
    for top in args.paths or REWRITE_PATHS:
        top = Path(top) if Path(top).is_absolute() else base / top
        if not top.exists():
            continue
        for p in _py_files(top):
            text = p.read_text(encoding="utf-8")
            if not _WORD.search(text):
                continue
            rel = p.resolve().relative_to(base).as_posix()
            if SKIP_PRAGMA.search(text):
                skipped.append(rel)
                continue
            new, counts, lf, rv = analyze(text, rel, apply=True)
            if "error" in counts:
                errors.append({"path": rel, "error": counts["error"]})
                continue
            totals.update(counts)
            if counts:
                per_file[rel] = counts
            left += lf
            review += rv
            if new != text:
                changed.append(rel)
                if not args.check:
                    p.write_text(new, encoding="utf-8")
    report = {"files_changed": len(changed), "totals": dict(sorted(totals.items())),
              "errors": errors, "skipped": skipped, "changed": changed,
              "left": left, "review": review, "per_file": per_file}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1) + "\n")
    if args.check:
        for rel in changed:
            print(f"would rewrite: {rel}")
        for e in errors:
            print(f"error: {e['path']}: {e['error']}")
        return 1 if changed or errors else 0
    print(json.dumps({k: report[k] for k in ("files_changed", "totals")}
                     | {"errors": len(errors), "left": len(left),
                        "review": len(review), "skipped": len(skipped)},
                     indent=1))
    return 1 if errors else 0


def cmd_reverse(args) -> int:
    sys.stdout.write(reverse_text(Path(args.file).read_text(encoding="utf-8")))
    return 0


# -- copy -------------------------------------------------------------------


def _record_sha(entry: str) -> str | None:
    if not entry.startswith("sha256="):
        return None
    digest = base64.urlsafe_b64decode(entry[7:] + "=" * (-len(entry[7:]) % 4))
    return digest.hex()


class _Source:
    """One distribution: its files under the package and its RECORD."""

    def __init__(self, pkg: str, src: Path):
        self.pkg = pkg
        self.zip = None
        wheels = sorted(src.glob(f"{DISTS[pkg]}-*.whl")) if src.is_dir() else []
        if src.is_file() and src.suffix == ".whl":
            wheels = [src]
        if wheels:
            if len(wheels) > 1:
                raise SystemExit(f"more than one {pkg} wheel in {src}")
            self.zip = zipfile.ZipFile(wheels[0])
            infos = [n for n in self.zip.namelist()
                     if re.match(rf"{DISTS[pkg]}-[^/]+\.dist-info/RECORD$", n)]
            if len(infos) != 1:
                raise SystemExit(f"{wheels[0]}: no single RECORD")
            self.dist = infos[0].rsplit("/", 1)[0]
            self.origin = wheels[0].name
        else:
            infos = sorted(src.glob(f"{DISTS[pkg]}-*.dist-info"))
            if len(infos) != 1:
                raise SystemExit(f"{src}: no single {pkg} dist-info")
            self.root = src
            self.dist = infos[0].name
            self.origin = self.dist
        self.version = self.dist[len(DISTS[pkg]) + 1:-len(".dist-info")]
        self.record = {}
        for row in csv.reader(io.StringIO(self.read(f"{self.dist}/RECORD")
                                          .decode())):
            if row:
                self.record[row[0]] = _record_sha(row[1]) if len(row) > 1 else None

    def read(self, name: str) -> bytes:
        if self.zip is not None:
            return self.zip.read(name)
        return (self.root / name).read_bytes()

    def files(self) -> list[str]:
        out = []
        for name, digest in self.record.items():
            if not name.startswith(self.pkg + "/"):
                continue
            if "__pycache__" in name.split("/") or name.endswith(".pyc"):
                continue
            if digest is None:
                raise SystemExit(f"{self.pkg}: RECORD has no sha256 for {name}")
            out.append(name)
        return sorted(out)

    def checked(self, name: str) -> bytes:
        data = self.read(name)
        if sha256(data) != self.record.get(name):
            raise SystemExit(f"{self.origin}: {name} does not match RECORD")
        return data

    def license(self) -> bytes | None:
        for name in (f"{self.dist}/licenses/LICENSE", f"{self.dist}/LICENSE"):
            if name in self.record:
                return self.checked(name)
        return None

    def requires(self) -> list[str]:
        meta = self.read(f"{self.dist}/METADATA").decode()
        return [ln.split(":", 1)[1].strip() for ln in meta.splitlines()
                if ln.startswith("Requires-Dist:")]


def cmd_copy(args) -> int:
    root = Path(args.root)
    src = Path(args.src).expanduser()
    manifest = {"prefix": PREFIX, "packages": {}, "files": {}}
    if (root / MANIFEST).exists():
        manifest = read_manifest(root)
        if manifest.get("prefix") != PREFIX:
            raise SystemExit(f"{MANIFEST}: prefix {manifest.get('prefix')!r} "
                             f"is not {PREFIX!r}")
        for pkg in args.packages:
            manifest["packages"].pop(pkg, None)
            manifest["files"] = {k: v for k, v in manifest["files"].items()
                                 if not k.startswith(pkg + "/")}
    for pkg in args.packages:
        s = _Source(pkg, src)
        dest = root / VENDOR / pkg
        if dest.exists():
            raise SystemExit(f"{dest} exists; remove it first")
        names = s.files()
        for name in names:
            data = s.checked(name)
            if PREFIX.encode() in data:
                raise SystemExit(f"{name} already contains {PREFIX!r}; the "
                                 "reverse would not be exact")
            out = root / VENDOR / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            manifest["files"][name] = {"status": "verbatim",
                                       "upstream": sha256(data)}
        lic = s.license()
        if lic is None:
            raise SystemExit(f"{s.origin}: no LICENSE")
        (dest / "LICENSE").write_bytes(lic)
        manifest["files"][f"{pkg}/LICENSE"] = {"status": "license",
                                               "upstream": sha256(lic)}
        manifest["packages"][pkg] = {"version": s.version, "source": s.origin,
                                     "files": len(names),
                                     "requires": s.requires()}
    write_manifest(root, manifest)
    print(json.dumps({k: {"version": v["version"], "files": v["files"]}
                      for k, v in manifest["packages"].items()}))
    return 0


# -- manifest and verify -----------------------------------------------------


def write_manifest(root: Path, manifest: dict) -> None:
    """One file per line, so a diff of the manifest reads file by file."""
    head = {k: v for k, v in manifest.items() if k != "files"}
    lines = ["{"]
    for k, v in head.items():
        lines.append(f" {json.dumps(k)}: {json.dumps(v, sort_keys=True)},")
    lines.append(' "files": {')
    items = sorted(manifest["files"].items())
    for i, (name, entry) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        lines.append(f"  {json.dumps(name)}: "
                     f"{json.dumps(entry, sort_keys=True)}{comma}")
    lines += [" }", "}"]
    (root / MANIFEST).write_text("\n".join(lines) + "\n")


def read_manifest(root: Path) -> dict:
    return json.loads((root / MANIFEST).read_text())


def status_of(path: Path, entry: dict) -> str:
    """What the file in the tree is, relative to what upstream shipped."""
    if not path.exists():
        return "excluded"
    data = path.read_bytes()
    if entry["status"] == "license":
        return "license"
    if sha256(data) == entry["upstream"]:
        return "verbatim"
    text = data.decode("utf-8")
    # The codemod's output exactly: an edit that only adds the prefix
    # reverses to upstream too, but the rewrite does not produce it.
    upstream = reverse_text(text)
    if (sha256(upstream.encode()) == entry["upstream"]
            and path.suffix == ".py" and rewrite_text(upstream) == text):
        return "codemod"
    if re.search(r"^sys\.modules\[__name__\]\s*=", text, re.MULTILINE):
        return "alias"
    return "edited"


def _tree_files(root: Path, packages) -> set[str]:
    out = set()
    for pkg in packages:
        for p in (root / VENDOR / pkg).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                out.add(p.relative_to(root / VENDOR).as_posix())
    return out


def cmd_manifest(args) -> int:
    root = Path(args.root)
    manifest = read_manifest(root)
    counts = collections.Counter()
    for name, entry in manifest["files"].items():
        path = root / VENDOR / name
        entry["status"] = status_of(path, entry)
        entry.pop("sha256", None)
        if entry["status"] != "excluded":
            digest = sha256(path.read_bytes())
            if digest != entry["upstream"]:
                entry["sha256"] = digest
        counts[entry["status"]] += 1
    extra = _tree_files(root, manifest["packages"]) - set(manifest["files"])
    if extra:
        print("files under the copy that upstream did not ship:")
        for name in sorted(extra):
            print(f"  {name}")
        return 1
    write_manifest(root, manifest)
    print(json.dumps(dict(sorted(counts.items()))))
    return 0


def verify(root: Path) -> list[str]:
    """Every way the tree differs from the manifest."""
    problems = []
    manifest = read_manifest(root)
    if manifest.get("prefix") != PREFIX:
        problems.append(f"manifest prefix {manifest.get('prefix')!r} is not "
                        f"{PREFIX!r}")
    for name, entry in sorted(manifest["files"].items()):
        path = root / VENDOR / name
        status = status_of(path, entry)
        if status != entry["status"]:
            problems.append(f"{name}: is {status}, manifest says "
                            f"{entry['status']}")
            continue
        if status == "excluded":
            continue
        data = path.read_bytes()
        if sha256(data) != entry.get("sha256", entry["upstream"]):
            problems.append(f"{name}: sha256 differs from the manifest")
        if status in ("edited", "alias") and not EDIT_MARKER.search(
                data.decode("utf-8")):
            problems.append(f"{name}: {status} without a '# gmlx:' comment")
        if name.endswith(".py") and status != "license":
            text = data.decode("utf-8")
            if not SKIP_PRAGMA.search(text) and rewrite_text(text) != text:
                problems.append(f"{name}: a module reference is not rewritten")
    extra = _tree_files(root, manifest["packages"]) - set(manifest["files"])
    for name in sorted(extra):
        problems.append(f"{name}: under the copy but not in the manifest")
    for d in sorted((root / VENDOR).iterdir()):
        if d.is_dir() and d.name != "__pycache__" and d.name not in (
                *manifest["packages"], *OTHER_VENDORED):
            problems.append(f"{d.name}: a package under the copy that the "
                            f"manifest does not list")
    for pkg, info in manifest["packages"].items():
        vf = root / VENDOR / pkg / VERSION_FILES.get(pkg, "version.py")
        if vf.exists():
            m = re.search(r"__version__\s*=\s*[\"']([^\"']+)", vf.read_text())
            if not m or m.group(1) != info["version"]:
                problems.append(f"{pkg}: version file says "
                                f"{m.group(1) if m else None}, manifest says "
                                f"{info['version']}")
    return problems


def cmd_verify(args) -> int:
    problems = verify(Path(args.root))
    for p in problems:
        print(p)
    if not problems:
        m = read_manifest(Path(args.root))
        c = collections.Counter(e["status"] for e in m["files"].values())
        print(f"ok: {len(m['files'])} files, {dict(sorted(c.items()))}")
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO),
                    help="repository root (default: this checkout)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("copy", help="copy both packages into gmlx/_vendor")
    c.add_argument("src", help="site-packages directory, wheel directory or "
                               "a wheel")
    c.add_argument("--packages", nargs="+", default=list(ROOTS),
                   choices=ROOTS)
    c.set_defaults(fn=cmd_copy)
    r = sub.add_parser("rewrite", help="rewrite module references in place")
    r.add_argument("paths", nargs="*",
                   help="files or directories, relative to --root (default: "
                        + ", ".join(REWRITE_PATHS) + ")")
    r.add_argument("--check", action="store_true",
                   help="change nothing; exit 1 if a file would change")
    r.add_argument("--report", help="write a JSON report here")
    r.set_defaults(fn=cmd_rewrite)
    v = sub.add_parser("reverse", help="print a copied file as upstream "
                                       "shipped it")
    v.add_argument("file")
    v.set_defaults(fn=cmd_reverse)
    m = sub.add_parser("manifest", help="recompute statuses and sha256")
    m.set_defaults(fn=cmd_manifest)
    k = sub.add_parser("verify", help="check the tree against the manifest")
    k.set_defaults(fn=cmd_verify)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
