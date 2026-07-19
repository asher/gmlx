#!/usr/bin/env python3
"""Tripwire for the repo ASCII policy: non-ASCII may appear only in the
allowlisted glyph modules (rendered TUI furniture, tokenizer logic, or test
fixtures), and the docs tree must be pure ASCII. Typography in messages,
comments, and docstrings is ASCII everywhere else. CPU-only."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Each entry earns its place with rendered UI glyphs, glyph logic, or fixtures.
_ALLOWED = {
    "gmlx/chat.py",              # status-line middot separators
    "gmlx/menubar.py",           # status dots, emoji, HIG menu ellipses
    "gmlx/reasoning.py",         # braille spinner, box drawing, check mark
    "gmlx/render.py",            # markdown rules, borders, bullets
    "gmlx/sessions.py",          # separators in exported markdown
    "gmlx/spinner.py",           # braille frames
    "gmlx/talk.py",              # VU-meter blocks, status glyphs
    "gmlx/talk_memory.py",       # bullet char in list-marker-strip regex
    "gmlx/tokenizer.py",         # U+2581 / U+0120 tokenizer markers
    "gmlx/tts.py",               # speech-text sanitizer dash/fraction glyphs
    "tests/e2e/checks.py",           # U+FFFD degeneration detection
    "tests/test_chat.py",            # asserts chat status-line separators
    "tests/test_e2e_checks.py",      # U+FFFD fixtures
    "tests/test_menubar.py",         # asserts menubar glyph labels
    "tests/test_reasoning.py",       # asserts spinner/box output
    "tests/test_render.py",          # asserts rendered rules; CJK width fixture
    "tests/test_server_patches.py",  # multilingual tokenizer fixtures
    "tests/test_tokenizer.py",       # U+FFFD control-token / byte-level fixtures
    "tests/test_tts.py",             # multi-script sanitizer fixtures
    "tests/test_talk_audio.py",      # asserts VU-meter block output
}


def _non_ascii_files(*dirs) -> set:
    out = set()
    for d in dirs:
        for p in sorted((_ROOT / d).rglob("*.py")):
            rel = p.relative_to(_ROOT).as_posix()
            if rel.startswith("gmlx/_vendor/"):
                continue                 # vendored code keeps upstream style
            if any(b > 0x7F for b in p.read_bytes()):
                out.add(rel)
    return out


def test_code_non_ascii_only_in_glyph_allowlist():
    offenders = _non_ascii_files("gmlx", "scripts", "tests")
    crept = offenders - _ALLOWED
    assert not crept, f"non-ascii crept into: {sorted(crept)}"


def test_docs_are_pure_ascii():
    files = [_ROOT / "README.md", _ROOT / "CONTRIBUTING.md"]
    files += [p for p in sorted((_ROOT / "docs").rglob("*"))
              if p.suffix in (".md", ".tape")]
    bad = [f.relative_to(_ROOT).as_posix() for f in files
           if f.is_file() and any(b > 0x7F for b in f.read_bytes())]
    assert not bad, f"non-ascii in docs: {bad}"
