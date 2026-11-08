#!/usr/bin/env python3
"""Write the snapshot of llama.cpp's byte-level pre-tokenizer table.

  python scripts/llama_pre_snapshot.py ~/src/llama.cpp
  python scripts/llama_pre_snapshot.py ~/src/llama.cpp --out FILE

It reads src/llama-vocab.cpp of a llama.cpp checkout: the regex list of
each pre type from the BPE tokenizer, and for each `tokenizer.ggml.pre`
name its pre type and the flags the vocab load block sets. The default
output is tests/load/data/llama_cpp_pre.json, which
tests/load/test_tokenizer_pre_table.py reads. The file is ASCII.

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "tests" / "load" / "data" / "llama_cpp_pre.json"
_PREFIX = "LLAMA_VOCAB_PRE_TYPE_"
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "'": "'",
            "0": "\0"}


def _strings_and_code(text: str):
    """Yield ("str", value) for each C string literal and ("code", chunk)
    for the text between them, with // and /* */ comments removed."""
    i, n, code = 0, len(text), []
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            i = text.find("\n", i)
            i = n if i < 0 else i
        elif text.startswith("/*", i):
            i = text.index("*/", i) + 2
        elif c == '"':
            if code:
                yield "code", "".join(code)
                code = []
            i += 1
            out = []
            while text[i] != '"':
                if text[i] == "\\":
                    out.append(_ESCAPES[text[i + 1]])
                    i += 2
                else:
                    out.append(text[i])
                    i += 1
            i += 1
            yield "str", "".join(out)
        else:
            code.append(c)
            i += 1
    if code:
        yield "code", "".join(code)


def _block(text: str, start: int) -> str:
    """The text inside the braces that open at or after ``start``."""
    i = text.index("{", start)
    depth, k, in_str = 0, i, False
    while k < len(text):
        c = text[k]
        if in_str:
            if c == "\\":
                k += 2
                continue
            if c == '"':
                in_str = False
        elif text.startswith("//", k):
            k = text.find("\n", k)
            continue
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:k]
        k += 1
    raise ValueError("unbalanced braces")


def _regex_table(src: str) -> dict[str, list[str]]:
    """Pre type -> its ordered regex list, from the BPE tokenizer switch."""
    start = src.index("llm_tokenizer_bpe(const llama_vocab & vocab)")
    body = _block(src, src.index("switch (vocab.get_pre_type())", start))
    table: dict[str, list[str]] = {}
    pending: list[str] = []
    pos = 0
    pattern = re.compile(
        r"case\s+" + _PREFIX + r"(\w+)\s*:|default\s*:|regex_exprs\s*=\s*\{")
    for m in pattern.finditer(body):
        if m.start() < pos:
            continue
        if m.group(0).startswith("case"):
            pending.append(m.group(1))
        elif m.group(0).startswith("default"):
            pending.append("DEFAULT")
        else:
            inner = _block(body, m.end() - 1)
            exprs: list[str] = []
            joined = False
            for kind, chunk in _strings_and_code(inner):
                if kind == "str":
                    if joined and exprs:
                        exprs[-1] += chunk
                    else:
                        exprs.append(chunk)
                    joined = True
                elif "," in chunk:
                    joined = False
            for name in pending:
                table[name] = exprs
            pending = []
            pos = m.end() - 1 + len(inner) + 2
    return table


_FLAG = re.compile(r"\b(ignore_merges|add_bos|clean_spaces)\s*=\s*(true|false)\s*;")


def _name_table(src: str) -> dict[str, dict]:
    """Pre name -> pre type and the flags its load branch sets."""
    start = src.index("if (type == LLAMA_VOCAB_TYPE_BPE) {")
    end = src.index("} else if (type == LLAMA_VOCAB_TYPE_SPM) {", start)
    block = src[start:end]
    defaults = {"ignore_merges": False, "add_bos": False, "clean_spaces": True}
    head = block[:block.index("if (tokenizer_pre.empty())")]
    for m in _FLAG.finditer(head):
        defaults[m.group(1)] = m.group(2) == "true"
    names: dict[str, dict] = {}
    branches = re.split(r"\}\s*else\s+if\s*\(", block)
    for branch in branches[1:]:
        cond, _, body = branch.partition(") {")
        listed = re.findall(r'tokenizer_pre\s*==\s*"([^"]+)"', cond)
        kind = re.search(r"pre_type\s*=\s*" + _PREFIX + r"(\w+)\s*;", body)
        if not listed or kind is None:
            continue
        entry = dict(defaults, pre_type=kind.group(1))
        for m in _FLAG.finditer(body):
            entry[m.group(1)] = m.group(2) == "true"
        entry["null_bos"] = "special_bos_id = LLAMA_TOKEN_NULL" in body
        for name in listed:
            names[name] = {k: entry[k] for k in (
                "pre_type", "ignore_merges", "add_bos", "clean_spaces",
                "null_bos")}
    return names


def snapshot(checkout: Path) -> dict:
    src = (checkout / "src" / "llama-vocab.cpp").read_text(encoding="utf-8")
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--short=9", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    names = _name_table(src)
    regex = _regex_table(src)
    used = {e["pre_type"] for e in names.values()}
    missing = sorted(used - set(regex) - {"DEFAULT"})
    if missing:
        raise SystemExit(f"pre types with no regex list: {missing}")
    return {
        "llama_cpp_commit": commit,
        "source": "src/llama-vocab.cpp",
        "names": dict(sorted(names.items())),
        "regex": {k: regex[k] for k in sorted(regex)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="a llama.cpp checkout")
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()
    data = snapshot(args.checkout.expanduser())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1, ensure_ascii=True) + "\n",
                        encoding="ascii")
    print(f"{len(data['names'])} names, {len(data['regex'])} pre types, "
          f"llama.cpp {data['llama_cpp_commit']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
