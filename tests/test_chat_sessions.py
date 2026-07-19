#!/usr/bin/env python3
"""`gmlx.sessions`: schema round-trip, atomic writes, listing/lookup, and
markdown export. Pure file logic - no model, no terminal."""
from __future__ import annotations

import json

import pytest

from gmlx import sessions as ss


def _doc(model="/m/x.gguf", msgs=None):
    return {
        "model": {"path": model, "arch": "qwen35"},
        "system_prompt": "BE TERSE",
        "sampling": {"temp": 0.7},
        "reasoning": "show",
        "render": "lite",
        "theme": "nord",
        "colorblind": False,
        "thinking_budget": None,
        "messages": msgs
        or [
            {"role": "user", "content": "hi", "ts": "2026-07-01T10:00:00"},
            {
                "role": "assistant",
                "content": "<think>hmm</think>Hello.",
                "ts": "2026-07-01T10:00:05",
                "canceled": False,
                "think_open": False,
                "stats": {"gen_tokens": 5},
            },
        ],
    }


def test_sessions_dir_honors_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert ss.sessions_dir() == tmp_path / "gmlx" / "chats"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert str(ss.sessions_dir()).endswith(".local/share/gmlx/chats")


def test_save_load_round_trip():
    path = ss.save_session(_doc(), "trip")
    assert path.exists() and not list(path.parent.glob("*.tmp"))
    doc, p = ss.load_session("trip")
    assert p == path
    assert doc["version"] == ss.SCHEMA_VERSION
    assert doc["created"] and doc["updated"]
    assert doc["system_prompt"] == "BE TERSE"
    assert doc["messages"][1]["content"] == "<think>hmm</think>Hello."
    # a second save keeps created, bumps nothing else structurally
    doc2, _ = ss.load_session("trip")
    ss.save_session(doc2, "trip")
    doc3, _ = ss.load_session("trip")
    assert doc3["created"] == doc["created"]


def test_load_by_explicit_path(tmp_path):
    p = ss.save_session(_doc(), "elsewhere")
    doc, path = ss.load_session(str(p))
    assert path == p and doc["model"]["arch"] == "qwen35"


def test_load_rejects_other_versions():
    p = ss.session_path("badver")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": 99, "messages": []}))
    with pytest.raises(ValueError, match="version"):
        ss.load_session("badver")


def test_default_session_name_slug():
    import re

    name = ss.default_session_name("/m/Qwen3.6-27B-UD-Q6_K_XL.gguf")
    assert " " not in name and name == name.lower()
    assert re.search(r"-\d{8}-\d{6}$", name)          # timestamp suffix


def test_list_sessions_and_latest_for_model():
    ss.save_session(_doc("/m/a.gguf"), "older")
    doc = _doc("/m/b.gguf")
    ss.save_session(doc, "newer")
    # force a strictly later 'updated' on the second file

    raw = json.loads(ss.session_path("newer").read_text())
    raw["updated"] = "2999-01-01T00:00:00"
    ss.session_path("newer").write_text(json.dumps(raw))
    rows = ss.list_sessions()
    assert [r["name"] for r in rows] == ["newer", "older"]
    assert rows[0]["turns"] == 1 and rows[0]["model"] == "b.gguf"
    assert ss.latest_for_model("/m/a.gguf") == "older"
    assert ss.latest_for_model("/m/none.gguf") is None


def test_split_thinking():
    assert ss.split_thinking("<think>steps</think>Answer.") == ("steps", "Answer.")
    assert ss.split_thinking("closing</think>Ans", think_open=True) == (
        "closing",
        "Ans",
    )
    assert ss.split_thinking("plain") == ("", "plain")


def test_export_markdown(tmp_path):
    doc = _doc()
    doc["messages"].append(
        {
            "role": "user",
            "content": "look",
            "images": ["/pics/cat.png"],
            "ts": "",
        }
    )
    doc["messages"].append(
        {"role": "assistant", "content": "A cat.", "canceled": True, "ts": ""}
    )
    out = ss.export_markdown(doc, str(tmp_path / "t.md"))
    text = out.read_text()
    assert text.startswith("# Chat - x.gguf")
    assert "## System\n\nBE TERSE" in text
    assert "<details><summary>thinking</summary>" in text
    assert "hmm" in text and "Hello." in text
    assert "<think>" not in text                      # markers never exported
    assert "*[image: `/pics/cat.png`]*" in text
    assert "## Assistant *(canceled)*" in text


def test_stray_json_never_dos_the_session_ui(monkeypatch, tmp_path):
    # One malformed .json in the chats dir (hand-edited, or a stray non-session
    # file) must not crash /sessions, --resume, or latest_for_model.
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    ss.save_session(_doc(), "good")
    d = ss.sessions_dir()
    (d / "list.json").write_text("[1, 2]")            # non-object JSON
    (d / "null.json").write_text("null")
    (d / "strmodel.json").write_text(json.dumps(
        {"version": 0, "model": "/m/x.gguf", "messages": ["hi"]}))
    names = [s["name"] for s in ss.list_sessions()]
    assert "good" in names
    assert "list" not in names and "null" not in names
    assert ss.latest_for_model("/m/x.gguf") == "good"
    with pytest.raises(ValueError):
        ss.load_session(str(d / "list.json"))
