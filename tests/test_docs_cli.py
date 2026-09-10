#!/usr/bin/env python3
"""Tripwire for `docs/cli.md`: every long option each verb's parser defines must
appear backticked in that verb's section, and every verb in the umbrella has a
`## gmlx <verb>` heading. The parsers are captured by intercepting parse_args,
so no verb executes: no model, no server, no network. CPU-only."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

_DOC = Path(__file__).resolve().parent.parent / "docs" / "cli.md"

# Gaps the doc has today. Phase 3 of the docs overhaul empties both sets; a
# flag or heading listed here that the doc later covers fails below so the
# entry gets removed.
_KNOWN_GAPS: dict = {"missing_headings": set()}

# Experimental flag families the reference deliberately leaves out.
_UNDOCUMENTED = re.compile(r"^--(over-|inject-|bench-chat-seed$)")


class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


@pytest.fixture
def capture(monkeypatch):
    """Run a verb entry point up to its parse_args call and return the parser."""
    def fake_parse_args(self, args=None, namespace=None):
        raise _Captured(self)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)

    def run(fn, argv):
        try:
            fn(argv)
        except _Captured as c:
            return c.parser
        raise AssertionError(f"{fn.__name__}{argv}: parse_args never reached")
    return run


def _long_options(parser) -> set:
    out = set()
    for a in parser._actions:
        if a.help is argparse.SUPPRESS or isinstance(a, argparse._HelpAction):
            continue
        if any(s in ("-h", "--help", "--help-all") for s in a.option_strings):
            continue
        out.update(s for s in a.option_strings if s.startswith("--"))
    return out


def _entry(verb):
    """(callable, argv) reaching the verb's parser, plus a set of option strings
    to subtract (service install repeats every serve flag)."""
    if verb in ("serve", "init", "sync-models", "launch", "stop", "restart",
                "status", "logs"):
        from gmlx.serve import server
        return server.main, ([] if verb == "serve" else [verb]), set()
    if verb == "run":
        from gmlx.commands import cli
        return cli.main, [], set()
    if verb == "chat":
        chat = pytest.importorskip("gmlx.tui.chat")
        return chat.cmd_chat, [], set()
    if verb == "talk":
        talk = pytest.importorskip("gmlx.talk.main")
        return talk.cmd_talk, [], set()
    if verb == "train":
        from gmlx.commands.train import cmd_train
        return cmd_train, [], set()
    if verb == "doctor":
        from gmlx.commands.doctor import cmd_doctor
        return cmd_doctor, [], set()
    if verb == "completion":
        from gmlx.commands.completion import cmd_completion
        return cmd_completion, [], set()
    from gmlx.commands import manage
    return getattr(manage, f"cmd_{verb}"), [], set()


def _verbs():
    from gmlx.commands.cli import _VERBS
    return list(_VERBS)


def _sections() -> dict:
    """`## heading text` -> body, for every h2 in the doc."""
    text = _DOC.read_text()
    out = {}
    parts = re.split(r"(?m)^## ", text)
    for part in parts[1:]:
        head, _, body = part.partition("\n")
        out[head.strip().replace("`", "")] = body
    return out


def _section_for(verb: str, sections: dict) -> str:
    strict = f"gmlx {verb}"
    if strict in sections:
        return sections[strict]
    if verb in ("stop", "restart", "status", "logs", "service"):
        return sections.get("gmlx serve", "")
    for head, body in sections.items():
        if re.match(rf"{re.escape(verb)}\b", head):
            return body
    return ""


def _backticked(text: str, flag: str) -> bool:
    pat = re.compile(r"`[^`\n]*(?<![\w-])" + re.escape(flag) + r"(?![\w-])[^`\n]*`")
    return bool(pat.search(text))


def test_doc_exists():
    assert _DOC.is_file(), f"missing {_DOC}"


def test_every_verb_has_a_heading():
    sections = _sections()
    missing = {v for v in _verbs() if f"gmlx {v}" not in sections}
    stale = _KNOWN_GAPS["missing_headings"] - missing
    assert not stale, f"drop from _KNOWN_GAPS, headings now present: {sorted(stale)}"
    new = missing - _KNOWN_GAPS["missing_headings"]
    assert not new, f"verbs with no `## gmlx <verb>` heading: {sorted(new)}"


@pytest.mark.parametrize("verb", _verbs())
def test_verb_flags_documented(verb, capture):
    if verb == "service":
        from gmlx.serve import server
        serve_flags = _long_options(capture(server.main, []))
        flags = set()
        for action in ("install", "uninstall", "status"):
            flags |= _long_options(capture(server.main, ["service", action]))
        flags -= serve_flags
    else:
        fn, argv, subtract = _entry(verb)
        flags = _long_options(capture(fn, argv)) - subtract
    if verb == "launch":
        menubar = pytest.importorskip("gmlx.commands.menubar")
        flags |= _long_options(capture(menubar.cmd_menubar, []))
    flags = {f for f in flags if not _UNDOCUMENTED.match(f)}
    section = _section_for(verb, _sections())
    assert section, f"no section found for {verb}"
    missing = {f for f in flags if not _backticked(section, f)}
    known = _KNOWN_GAPS.get(verb, set())
    stale = known - missing
    assert not stale, f"{verb}: drop from _KNOWN_GAPS, now documented: {sorted(stale)}"
    new = missing - known
    assert not new, f"{verb}: flags absent from its docs/cli.md section: {sorted(new)}"
