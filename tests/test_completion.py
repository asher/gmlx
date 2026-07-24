#!/usr/bin/env python3
"""Shell-completion engine (`gmlx __complete`) + `gmlx completion zsh`.

CPU-only: every candidate is computed from argparse help, the verb table, and a
temp config - no model, no server, no shell.
"""

from __future__ import annotations

import textwrap

import pytest

from gmlx import cli, completion, lifecycle


def _vals(lines):
    """The value column (before any tab) of candidate lines."""
    return [ln.split("\t", 1)[0] for ln in lines]


# Verb-level (first word) completion.

def test_verb_completion_lists_verbs_and_alias():
    vals = _vals(completion._complete([""]))
    for v in ("run", "chat", "serve", "launch", "list", "ps", "completion"):
        assert v in vals
    assert "ls" in vals                     # the list alias is offered too
    assert "__complete" not in vals         # the hidden helper never shows


def test_every_verb_has_a_description():
    # Drift guard: a new dispatchable verb must get a one-liner here.
    for v in cli._VERBS:
        assert completion._VERB_DESC.get(v), f"missing _VERB_DESC for {v!r}"


def test_verb_candidates_carry_descriptions():
    line = next(ln for ln in completion._complete([""]) if ln.startswith("run\t"))
    assert "generate" in line


# Flag completion (scraped from each verb's own --help).

def test_run_flag_completion():
    vals = _vals(completion._complete(["run", "--"]))
    for f in ("--max-tokens", "--temp", "--mmproj", "--speculative"):
        assert f in vals


def test_flag_help_survives_wrapping():
    # `-n LINES, --lines LINES` wraps its help onto the next line in argparse output;
    # the scraper must still attach it.
    lines = completion._complete(["logs", "--"])
    n = next(ln for ln in lines if ln.startswith("-n\t"))
    assert "history" in n


def test_service_borrows_serve_flags():
    vals = _vals(completion._complete(["service", "install", "--"]))
    assert "--host" in vals and "--port" in vals


# Value completion: files vs config models vs enums.

def test_path_flag_value_defers_to_files():
    lines = completion._complete(["run", "--config", ""])
    assert lines and lines[0] == "::files"


def test_non_path_flag_value_offers_nothing():
    # --temp takes a float we can't enumerate: no candidates, and crucially no files.
    assert completion._complete(["run", "--temp", ""]) == []


def _write_cfg(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(textwrap.dedent("""
        server:
          model_dirs: []
          assistants:
            helper:
              model: qwen-fast
        models:
          qwen-fast:
            path: /tmp/qwen.gguf
          gemma-vlm:
            path: /tmp/gemma.gguf
        aliases:
          q: qwen-fast
    """).strip())
    return str(cfg)


def test_model_positional_lists_config_ids_and_files(tmp_path):
    cfg = _write_cfg(tmp_path)
    lines = completion._complete(["run", "--config", cfg, ""])
    assert lines[0] == "::files"            # a path is always acceptable too
    vals = _vals(lines)
    assert "qwen-fast" in vals and "gemma-vlm" in vals
    assert "q" in vals                      # aliases included
    alias = next(ln for ln in lines if ln.startswith("q\t"))
    assert "alias -> qwen-fast" in alias
    assert "helper" in vals                 # served assistants included
    helper = next(ln for ln in lines if ln.startswith("helper\t"))
    assert "assistant -> qwen-fast" in helper


def test_model_positional_drops_off_once_model_given(tmp_path):
    cfg = _write_cfg(tmp_path)
    # A model is already in place; the next bare word is not a second model.
    assert completion._complete(["run", "--config", cfg, "qwen-fast", ""]) == []


def test_talk_positional_lists_model_ids_without_files(tmp_path):
    cfg = _write_cfg(tmp_path)
    lines = completion._complete(["talk", "--config", cfg, ""])
    assert "::files" not in lines           # a served id, never a path on disk
    vals = _vals(lines)
    assert "qwen-fast" in vals and "q" in vals
    # Once a model is chosen, no more positional candidates.
    assert completion._complete(["talk", "--config", cfg, "qwen-fast", ""]) == []


def test_ls_alias_canonicalizes_for_flags():
    vals = _vals(completion._complete(["ls", "--"]))
    assert "--config" in vals and "--json" in vals


# Positional value sources that aren't config models.

def test_launch_completes_harnesses_and_menubar():
    vals = _vals(completion._complete(["launch", ""]))
    for h in ("opencode", "claude-code", "menubar"):
        assert h in vals
    # Once a harness is chosen, no more positional candidates.
    assert completion._complete(["launch", "opencode", ""]) == []


def test_service_completes_actions():
    vals = _vals(completion._complete(["service", ""]))
    assert vals == ["install", "uninstall", "status"] or set(vals) == {
        "install", "uninstall", "status"}


def test_validate_positional_defers_to_files():
    assert completion._complete(["validate", ""]) == ["::files"]


def test_unknown_verb_yields_nothing():
    assert completion._complete(["frobnicate", ""]) == []


# Live endpoint completion: --host/--port/--url/--base-url from running servers.

_FAKE_RUNS = [
    {"host": "127.0.0.1", "port": 8080,
     "url": "http://127.0.0.1:8080", "managed_by": "detach"},
    {"host": "127.0.0.1", "port": 8081,
     "url": "http://127.0.0.1:8081", "managed_by": "detach"},
    {"host": "0.0.0.0", "port": 9090,
     "url": "http://0.0.0.0:9090", "managed_by": "launchd"},
]


def test_port_value_completes_running_ports(monkeypatch):
    monkeypatch.setattr(lifecycle, "list_runs", lambda: list(_FAKE_RUNS))
    vals = _vals(completion._complete(["serve", "--port", ""]))
    assert vals == ["9090", "8081", "8080"] or set(vals) == {"8080", "8081", "9090"}
    # The host + how-it's-managed ride along as the description.
    line = next(ln for ln in completion._complete(["stop", "--port", ""])
                if ln.startswith("9090\t"))
    assert "0.0.0.0" in line and "launchd" in line


def test_host_value_completes_and_dedupes(monkeypatch):
    monkeypatch.setattr(lifecycle, "list_runs", lambda: list(_FAKE_RUNS))
    vals = _vals(completion._complete(["status", "--host", ""]))
    assert set(vals) == {"127.0.0.1", "0.0.0.0"}   # three runs, two distinct hosts


def test_url_value_completes_running_urls(monkeypatch):
    monkeypatch.setattr(lifecycle, "list_runs", lambda: list(_FAKE_RUNS))
    vals = _vals(completion._complete(["ps", "--url", ""]))
    assert "http://0.0.0.0:9090" in vals
    assert "http://127.0.0.1:8080" in vals


def test_base_url_value_appends_v1(monkeypatch):
    monkeypatch.setattr(lifecycle, "list_runs", lambda: list(_FAKE_RUNS))
    vals = _vals(completion._complete(["launch", "--base-url", ""]))
    assert "http://127.0.0.1:8080/v1" in vals     # base-url wants the /v1 suffix
    assert "http://127.0.0.1:8080" not in vals


def test_endpoint_value_empty_with_no_servers(monkeypatch):
    monkeypatch.setattr(lifecycle, "list_runs", lambda: [])
    # Nothing running -> no candidates, and crucially no spurious ::files.
    assert completion._complete(["serve", "--port", ""]) == []
    assert completion._complete(["ps", "--url", ""]) == []


# Robustness: completion must never raise.

def test_complete_swallows_bad_config(tmp_path, capsys):
    bad = tmp_path / "broken.yaml"
    bad.write_text("models: [this is not: valid: yaml")
    assert completion.cmd_complete(["run", "--config", str(bad), ""]) == 0
    # Whatever it prints, it exits cleanly (no traceback).
    assert "Traceback" not in capsys.readouterr().err


def test_cmd_complete_prints_lines(capsys):
    assert completion.cmd_complete([""]) == 0
    out = capsys.readouterr().out
    assert "run" in out and "serve" in out


# `completion zsh` script emission.

def test_completion_zsh_emits_script(capsys):
    assert completion.cmd_completion(["zsh"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("#compdef gmlx")  # the command name
    assert "gmlx __complete" in out         # the live callback
    assert "funcstack[1] == _gmlx" in out  # dual eval/fpath idiom
    assert "compdef _gmlx gmlx" in out


def test_completion_bash_emits_script(capsys):
    assert completion.cmd_completion(["bash"]) == 0
    out = capsys.readouterr().out
    assert "complete -F _gmlx gmlx" in out  # the command name
    assert "gmlx __complete" in out         # same live callback as zsh
    assert "COMPREPLY" in out


def test_completion_fish_emits_script(capsys):
    assert completion.cmd_completion(["fish"]) == 0
    out = capsys.readouterr().out
    assert "complete -c gmlx" in out        # the command name
    assert "gmlx __complete" in out         # same live callback as zsh/bash
    assert "__fish_complete_path" in out     # ::files is handled in fish too


def test_completion_no_shell_prints_help(capsys):
    assert completion.cmd_completion([]) == 0
    assert "completion script" in capsys.readouterr().out.lower()


def test_completion_rejects_unknown_shell(capsys):
    with pytest.raises(SystemExit):
        completion.cmd_completion(["powershell"])  # not implemented; argparse rejects
    assert "powershell" in capsys.readouterr().err


# Umbrella routing.

def test_umbrella_routes_hidden_complete(capsys):
    assert cli.umbrella_main(["__complete", ""]) == 0
    assert "run" in capsys.readouterr().out


def test_umbrella_routes_completion_verb(capsys):
    assert cli.umbrella_main(["completion", "zsh"]) == 0
    assert capsys.readouterr().out.startswith("#compdef gmlx")


def test_completion_is_a_known_verb():
    assert "completion" in cli._VERBS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
