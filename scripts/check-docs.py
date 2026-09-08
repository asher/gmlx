#!/usr/bin/env python3
"""Lint the Markdown docs against the house style and check every link.

Checks (each a failure):
  - bold/italic emphasis outside code fences
  - dash asides (" - " or " -- ") in prose
  - "&" in a heading
  - a table cell over 160 characters (the generated family table is exempt)
  - a relative link, image, or in-repo GitHub link whose file or #anchor
    does not resolve (GitHub slug rules)
  - a ```yaml fence opener with trailing text (the docs tests would skip it)
  - non-ASCII bytes (tests/test_ascii_hygiene.py enforces this too)

Report-only modes, never failures:
  --ownership   backticked --flags outside docs/cli.md and env names outside
                docs/env-vars.md and docs/internals/debug-switches.md
  --style       banned words and phrases from the writing contract

  python scripts/check-docs.py             # lint README.md, CONTRIBUTING.md, docs/**
  python scripts/check-docs.py --ownership
  python scripts/check-docs.py --style
Stdlib only, so the CI lint job can run it without installing the package.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CELL_MAX = 160
_GITHUB_BLOB = re.compile(r"https://github\.com/asher/gmlx/blob/main/([^)#\s\"]+)(#[^)\s\"]*)?")
_GITHUB_RAW = re.compile(r"https://raw\.githubusercontent\.com/asher/gmlx/main/([^)\s\"]+)")
_MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MD_IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_HTML_REF = re.compile(r"\b(?:href|src|srcset)=\"([^\"]+)\"")
_ITALIC = re.compile(r"(?<![\w*\\])\*(?!\s)[^*\n]+?(?<!\s)\*(?![\w*])")
_FLAG_TICK = re.compile(r"`(--[a-z][a-z0-9-]*)")
_ENV_TICK = re.compile(r"`((?:GMLX|MLX_VLM|APC|KV)_[A-Z0-9_]+|PREFILL_STEP_SIZE|TOP_LOGPROBS_K|MAX_KV_SIZE|QUANTIZED_KV_START)")
_BANNED = [
    r"\bDetails:", r"\bNote:", r"\bhonest\b", r"\bdeliberately\b",
    r"\bload-bearing\b", r"\bjust\b", r"\bsimply\b", r"\bthe point is\b",
    r"\bas of this writing\b",
]
_ENV_OWNERS = {"docs/env-vars.md", "docs/internals/debug-switches.md"}
_FLAG_OWNER = "docs/cli.md"


def doc_files() -> list:
    files = [_REPO / "README.md", _REPO / "CONTRIBUTING.md"]
    files += sorted((_REPO / "docs").rglob("*.md"))
    return [f for f in files if f.is_file()]


def slugify(heading: str) -> str:
    """GitHub's anchor rule: strip markup, lowercase, drop punctuation except
    hyphen and underscore, spaces to hyphens."""
    h = heading.strip()
    h = re.sub(r"`", "", h)
    h = re.sub(r"\*", "", h)
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h)
    h = h.lower()
    h = re.sub(r"[^\w\- ]", "", h)
    h = h.replace(" ", "-")
    return h


def anchors_of(path: Path) -> set:
    seen: Counter = Counter()
    out = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not m:
            continue
        s = slugify(m.group(2))
        n = seen[s]
        seen[s] += 1
        out.add(s if n == 0 else f"{s}-{n}")
    return out


def prose_lines(text: str):
    """(lineno, line) for lines outside code fences."""
    in_fence = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield i, line


def _is_family_row(line: str) -> bool:
    return line.startswith("|") and "temperature=" in line


def check_file(path: Path, anchor_cache: dict) -> list:
    rel = path.relative_to(_REPO).as_posix()
    problems = []
    raw = path.read_bytes()
    if any(b > 0x7F for b in raw):
        problems.append(f"{rel}: non-ASCII bytes")
    text = raw.decode("utf-8", errors="replace")

    for n, line in enumerate(text.splitlines(), 1):
        if line.startswith("```yaml") and line.strip() != "```yaml":
            problems.append(f"{rel}:{n}: yaml fence opener has trailing text")

    for n, line in prose_lines(text):
        if "**" in line:
            problems.append(f"{rel}:{n}: bold emphasis")
        elif _ITALIC.search(line) and not line.lstrip().startswith(("* ", "- ", "|")):
            problems.append(f"{rel}:{n}: italic emphasis")
        elif _ITALIC.search(line) and line.lstrip().startswith("|"):
            if not _is_family_row(line):
                problems.append(f"{rel}:{n}: italic emphasis")
        if re.match(r"^#{1,6}\s", line):
            if "&" in line:
                problems.append(f"{rel}:{n}: '&' in heading")
            continue
        if re.match(r"^\s*[-*]\s", line) or line.lstrip().startswith(">"):
            body = re.sub(r"^\s*[-*>]\s*", "", line)
        else:
            body = line
        body_nocode = re.sub(r"`[^`]*`", "", body)
        if " -- " in body_nocode or re.search(r"(?<!\|)\s-\s(?!\|)", body_nocode):
            problems.append(f"{rel}:{n}: dash aside")
        if line.lstrip().startswith("|") and not _is_family_row(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            for c in cells:
                if len(c) > _CELL_MAX and not re.match(r"^:?-+:?$", c):
                    problems.append(f"{rel}:{n}: table cell {len(c)} chars")
                    break

    targets = []
    for m in _MD_LINK.finditer(text):
        targets.append(m.group(1))
    for m in _MD_IMG.finditer(text):
        targets.append(m.group(1))
    for m in _HTML_REF.finditer(text):
        targets.append(m.group(1))
    for t in targets:
        t = t.strip("<>")
        if t.startswith(("mailto:", "data:")):
            continue
        anchor = None
        if t.startswith("http"):
            m = _GITHUB_BLOB.match(t)
            if m:
                target = _REPO / m.group(1)
                anchor = (m.group(2) or "")[1:] or None
            else:
                m = _GITHUB_RAW.match(t)
                if not m:
                    continue
                target = _REPO / m.group(1)
        elif t.startswith("#"):
            target = path
            anchor = t[1:]
        else:
            file_part, _, anchor = t.partition("#")
            target = (path.parent / file_part).resolve()
            anchor = anchor or None
        if not target.exists():
            problems.append(f"{rel}: broken link {t}")
            continue
        if anchor and target.suffix == ".md":
            if target not in anchor_cache:
                anchor_cache[target] = anchors_of(target)
            if anchor not in anchor_cache[target]:
                problems.append(f"{rel}: missing anchor {t}")
    return problems


def report_ownership(files: list) -> None:
    print("backticked flags outside docs/cli.md, env names outside their owners:")
    for path in files:
        rel = path.relative_to(_REPO).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        prose = "\n".join(line for _, line in prose_lines(text))
        flags = Counter(_FLAG_TICK.findall(prose)) if rel != _FLAG_OWNER else Counter()
        envs = Counter(_ENV_TICK.findall(prose)) if rel not in _ENV_OWNERS else Counter()
        if not flags and not envs:
            continue
        print(f"  {rel}: {sum(flags.values())} flag mentions "
              f"({len(flags)} distinct), {sum(envs.values())} env mentions "
              f"({len(envs)} distinct)")
        top = flags.most_common(5)
        if top:
            print("    flags: " + ", ".join(f"{k} x{v}" for k, v in top))
        top = envs.most_common(5)
        if top:
            print("    env:   " + ", ".join(f"{k} x{v}" for k, v in top))


def report_style(files: list) -> None:
    print("banned words and phrases (writing contract):")
    pats = [re.compile(p, re.IGNORECASE) for p in _BANNED]
    for path in files:
        rel = path.relative_to(_REPO).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        hits: Counter = Counter()
        for _, line in prose_lines(text):
            for p in pats:
                if p.search(line):
                    hits[p.pattern] += 1
        if hits:
            print(f"  {rel}: " + ", ".join(f"{k} x{v}" for k, v in hits.most_common()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ownership", action="store_true",
                    help="report flag and env mentions outside their owner files")
    ap.add_argument("--style", action="store_true",
                    help="report banned words and phrases")
    ap.add_argument("paths", nargs="*", help="files to lint (default: all docs)")
    args = ap.parse_args(argv)

    files = [Path(p).resolve() for p in args.paths] if args.paths else doc_files()
    if args.ownership:
        report_ownership(files)
        return 0
    if args.style:
        report_style(files)
        return 0

    anchor_cache: dict = {}
    problems = []
    for f in files:
        problems += check_file(f, anchor_cache)
    for p in problems:
        print(p)
    kinds = Counter(p.split(": ", 1)[1].split(" ")[0] for p in problems)
    if problems:
        sys.stdout.flush()
        print(f"{len(problems)} problems in {len(files)} files: "
              + ", ".join(f"{k} {v}" for k, v in kinds.most_common()), file=sys.stderr)
        return 1
    print(f"{len(files)} files clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
