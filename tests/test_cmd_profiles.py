#!/usr/bin/env python3
"""``gmlx profiles`` - the built-in family sampling table, user-profile
listing, and per-model resolution. CPU-only: configs are written to tmp_path
and the model files are fakes (family comes from the explicit `family:` key,
so no GGUF header is read)."""
from __future__ import annotations

import json

from gmlx import manage


def _cfg(tmp_path, body: str) -> str:
    g = tmp_path / "m.gguf"
    g.write_text("x")
    p = tmp_path / "cfg.yaml"
    p.write_text(body.replace("PATH", str(g)))
    return str(p)


def test_profiles_table_without_config(monkeypatch, capsys):
    """The table form needs no config: every family + its intents print."""
    from gmlx import config as cfgmod
    monkeypatch.setattr(cfgmod, "default_config_paths", lambda: [])
    rc = manage.cmd_profiles([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "qwen3.6" in out and "gemma" in out and "gpt-oss" in out
    assert "@coding" in out and "@reasoning-high" in out
    assert "no server config found" in out


def test_profiles_table_lists_user_profiles_and_models(tmp_path, capsys):
    cfg = _cfg(tmp_path, (
        "profiles:\n"
        "  slow: {sampling: {temperature: 0.2}}\n"
        "  coding: {sampling: {temperature: 0.42}}\n"
        "models:\n"
        "  qw: {path: PATH, family: qwen3.6}\n"))
    rc = manage.cmd_profiles(["--config", cfg])
    assert rc == 0
    out = capsys.readouterr().out
    assert "slow: temperature=0.2" in out
    assert "(replaces the built-in intent)" in out    # `coding` shadows
    assert "qw" in out and "qwen3.6" in out


def test_profiles_model_resolution_rows(tmp_path, capsys):
    """Per-model form prints base + every addressable profile fully resolved,
    with the shaping layers (rule, overrides) named."""
    cfg = _cfg(tmp_path, (
        "profiles:\n"
        "  slow: {sampling: {temperature: 0.2}}\n"
        "rules:\n"
        "  - {match: 'qw*', profile: slow}\n"
        "models:\n"
        "  qw:\n"
        "    path: PATH\n"
        "    family: qwen3.6\n"
        "    overrides: {sampling: {top_k: 50}}\n"))
    rc = manage.cmd_profiles(["qw", "--config", cfg])
    assert rc == 0
    out = capsys.readouterr().out
    assert "family qwen3.6" in out
    assert "matched rule profile" in out and "slow" in out
    assert "model overrides" in out and "top_k=50" in out
    # base carries the rule profile's t=0.2 over the family's 1.0; @instruct
    # switches to the card's non-thinking point.
    assert "temperature=0.2" in out
    assert "enable_thinking=False" in out


def test_profiles_model_json_resolves_alias(tmp_path, capsys):
    cfg = _cfg(tmp_path, (
        "models:\n"
        "  qw: {path: PATH, family: qwen3.6}\n"
        "aliases: {fast: qw@coding}\n"))
    rc = manage.cmd_profiles(["fast", "--config", cfg, "--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["model"] == "qw"
    assert doc["family"] == "qwen3.6"
    assert doc["resolved"]["@coding"]["sampling"]["temperature"] == 0.6
    assert doc["resolved"]["base"]["sampling"]["temperature"] == 1.0


def test_profiles_unknown_model_errors(tmp_path, capsys):
    cfg = _cfg(tmp_path, "models:\n  qw: {path: PATH}\n")
    rc = manage.cmd_profiles(["nope", "--config", cfg])
    assert rc == 2
    assert "unknown model" in capsys.readouterr().err


def test_profiles_kill_switch_drops_builtin_rows(tmp_path, capsys):
    """family_defaults: false removes the built-in rows and says so."""
    cfg = _cfg(tmp_path, (
        "server: {family_defaults: false}\n"
        "profiles:\n"
        "  slow: {sampling: {temperature: 0.2}}\n"
        "models:\n"
        "  qw: {path: PATH, family: qwen3.6}\n"))
    rc = manage.cmd_profiles(["qw", "--config", cfg])
    assert rc == 0
    out = capsys.readouterr().out
    assert "family_defaults: false" in out
    assert "@coding" not in out          # built-ins not addressable
    assert "@slow" in out                # user profiles still are


def test_profiles_model_form_without_config_errors(monkeypatch, capsys):
    from gmlx import config as cfgmod
    monkeypatch.setattr(cfgmod, "default_config_paths", lambda: [])
    rc = manage.cmd_profiles(["some-id"])
    assert rc == 2
    assert "needs a server config" in capsys.readouterr().err
