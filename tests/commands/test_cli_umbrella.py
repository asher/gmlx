#!/usr/bin/env python3
"""`gmlx <verb>` umbrella dispatch - each verb routes to the right entry
point (all share one console entry). CPU-only:
every target entry point is a recording stub.
"""

from __future__ import annotations

import pytest

import gmlx.tui.chat as chat  # noqa: E402
import gmlx.commands.cli as cli  # noqa: E402
import gmlx.commands.manage as manage  # noqa: E402
import gmlx.serve.server as server  # noqa: E402


@pytest.fixture
def routes(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "main", lambda argv, prog=None: seen.__setitem__("run", argv) or 0)
    monkeypatch.setattr(chat, "cmd_chat",
                        lambda argv, prog=None: seen.__setitem__("chat", argv) or 0)
    monkeypatch.setattr(server, "main",
                        lambda argv, prog=None: seen.__setitem__("server", argv) or 0)
    monkeypatch.setattr(manage, "cmd_validate",
                        lambda argv, prog=None: seen.__setitem__("validate", argv) or 0)
    monkeypatch.setattr(manage, "cmd_pull",
                        lambda argv, prog=None: seen.__setitem__("pull", argv) or 0)
    monkeypatch.setattr(manage, "cmd_rm",
                        lambda argv, prog=None: seen.__setitem__("rm", argv) or 0)
    return seen


def test_rm_verb(routes):
    assert cli.umbrella_main(["rm", "old-model", "--yes"]) == 0
    assert routes["rm"] == ["old-model", "--yes"]


def test_run_verb(routes):
    assert cli.umbrella_main(["run", "model.gguf", "--prompt", "hi"]) == 0
    assert routes["run"] == ["model.gguf", "--prompt", "hi"]


def test_version_flag(capsys):
    import gmlx
    for flag in ("--version", "-V", "version"):
        assert cli.umbrella_main([flag]) == 0
        out = capsys.readouterr().out
        assert out.strip() == f"gmlx {gmlx.__version__}"


def test_chat_verb(routes):
    assert cli.umbrella_main(["chat", "model.gguf", "--temp", "0.7"]) == 0
    assert routes["chat"] == ["model.gguf", "--temp", "0.7"]


def test_keyboard_interrupt_exits_130(routes, monkeypatch, capsys):
    def boom(argv, prog=None):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "main", boom)
    assert cli.umbrella_main(["run", "model.gguf"]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_dispatch_error_is_one_line(routes, monkeypatch, capsys):
    def boom(argv, prog=None):
        raise OSError("connection refused")
    monkeypatch.setattr(manage, "cmd_pull", boom)
    assert cli.umbrella_main(["pull", "hf:o/r/m.gguf"]) == 1
    assert "error: connection refused" in capsys.readouterr().err


def test_serve_verb_passes_through(routes):
    assert cli.umbrella_main(["serve", "model.gguf", "--port", "9"]) == 0
    assert routes["server"] == ["model.gguf", "--port", "9"]


def test_init_verb_prefixes_subcommand(routes):
    assert cli.umbrella_main(["init", "--models-dir", "x"]) == 0
    assert routes["server"] == ["init", "--models-dir", "x"]


def test_launch_verb_prefixes_subcommand(routes):
    assert cli.umbrella_main(["launch", "opencode"]) == 0
    assert routes["server"] == ["launch", "opencode"]


def test_validate_verb(routes):
    assert cli.umbrella_main(["validate", "hf:o/r/m.gguf"]) == 0
    assert routes["validate"] == ["hf:o/r/m.gguf"]


def test_pull_verb(routes):
    assert cli.umbrella_main(["pull", "hf:o/r/m.gguf", "--to", "."]) == 0
    assert routes["pull"] == ["hf:o/r/m.gguf", "--to", "."]


def test_ls_aliases_list(routes, monkeypatch):
    monkeypatch.setattr(manage, "cmd_list",
                        lambda argv, prog=None: routes.__setitem__("list", argv) or 0)
    assert cli.umbrella_main(["ls", "--json"]) == 0
    assert routes["list"] == ["--json"]          # `ls` canonicalizes to `list`


def test_no_args_prints_help(routes, capsys):
    assert cli.umbrella_main([]) == 0
    assert "usage: gmlx <command>" in capsys.readouterr().out
    assert routes == {}                  # nothing dispatched


def test_unknown_verb_errors(routes, capsys):
    rc = cli.umbrella_main(["frobnicate"])
    assert rc == 2
    assert "unknown command" in capsys.readouterr().err


def test_stray_gguf_path_hints_run(routes, capsys):
    rc = cli.umbrella_main(["model.gguf", "--prompt", "hi"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "gmlx run model.gguf" in err


def test_unknown_verb_suggests_close_match(routes, capsys):
    rc = cli.umbrella_main(["serv"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "did you mean 'serve'?" in err
    assert "usage:" not in err        # two-line suggestion, not the full wall
    assert routes == {}


def test_help_verb_routes_to_subcommand_help(routes):
    assert cli.umbrella_main(["help", "validate"]) == 0
    assert routes["validate"] == ["--help"]


def test_umbrella_one_shot_example_is_runnable(routes, capsys):
    cli.umbrella_main([])
    out = capsys.readouterr().out
    assert "--prompt" in out          # `run` defines --prompt, not -p
    assert " -p " not in out


def test_deleted_cwd_gives_actionable_error(tmp_path, monkeypatch, capsys):
    # A shell sitting in a deleted directory must get a clear message, not a
    # bare "[Errno 2] No such file or directory" from some abspath() deep in
    # a verb.
    import os

    import gmlx.commands.cli as cli

    def _gone():
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(os, "getcwd", _gone)
    assert cli.umbrella_main(["chat", "whatever"]) == 1
    err = capsys.readouterr().err
    assert "working directory no longer exists" in err


# condensed --help vs --help-all (run + chat)
def test_run_help_is_condensed(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "common options" in out
    assert "--prompt" in out and "--reasoning" in out
    assert "--help-all" in out and "further flags" in out
    assert "--bench-chat-seed" not in out       # the long tail is demoted


def test_run_help_all_is_complete(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help-all"])
    out = capsys.readouterr().out
    assert "--bench-chat-seed" in out and "--kv-bits" in out
    assert "--stream-experts" in out


def test_chat_help_is_condensed(capsys):
    import gmlx.tui.chat as chat
    with pytest.raises(SystemExit):
        chat.cmd_chat(["--help"])
    out = capsys.readouterr().out
    assert "common options" in out
    assert "--assistant" in out and "--reasoning" in out
    assert "--logit-bias" not in out            # the long tail is demoted


def test_chat_help_all_is_complete(capsys):
    import gmlx.tui.chat as chat
    with pytest.raises(SystemExit):
        chat.cmd_chat(["--help-all"])
    out = capsys.readouterr().out
    assert "--logit-bias" in out and "--kv-bits" in out


# Help always wins: a value-taking flag must not swallow a following help
# token (`chat --thinking --help` used to die with "expected one argument").
def test_help_after_value_flag_is_hoisted(routes):
    assert cli.umbrella_main(["chat", "--thinking", "--help"]) == 0
    assert routes["chat"] == ["--help"]
    assert cli.umbrella_main(["run", "--profile", "--help-all"]) == 0
    assert routes["run"] == ["--help-all"]
    assert cli.umbrella_main(["serve", "model.gguf", "--config", "-h"]) == 0
    assert routes["server"] == ["-h"]


def test_help_hoist_stops_at_double_dash(routes):
    # Past a `--` separator nothing is a flag; the argv passes through intact.
    assert cli.umbrella_main(["rm", "old-model", "--", "--help"]) == 0
    assert routes["rm"] == ["old-model", "--", "--help"]
