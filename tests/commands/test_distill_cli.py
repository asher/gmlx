"""``gmlx distill``: the umbrella routes each action to its own handler, the
help paths exit clean, an unknown action is refused, ``cache --validate``
runs the validator on a minted cache, and completion offers the actions
and each action's flags. No model loads."""
from __future__ import annotations

import contextlib
import io

import pytest

from gmlx.commands import cli, completion, distill

pytest.importorskip("tokenizers")


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            rc = cli.umbrella_main(argv)
        except SystemExit as e:
            rc = e.code
    return rc, buf.getvalue()


def test_verb_registered():
    assert "distill" in cli._VERBS
    assert completion._VERB_DESC["distill"]


@pytest.mark.parametrize("argv", [["distill"], ["distill", "--help"], ["distill", "help"]])
def test_umbrella_help(argv):
    rc, out = _run(argv)
    assert rc == 0
    for action in distill._ACTIONS:
        assert f"  {action}" in out


@pytest.mark.parametrize("action", distill._ACTIONS)
def test_action_help_exits_clean(action):
    rc, out = _run(["distill", action, "--help"])
    assert rc == 0
    assert f"usage: gmlx distill {action}" in out


def test_action_help_wins_over_a_value_flag():
    rc, out = _run(["distill", "cache", "--max-len", "5", "--help"])
    assert rc == 0 and "usage: gmlx distill cache" in out


def test_unknown_action_refused():
    rc, out = _run(["distill", "bogus"])
    assert rc == 2 and "unknown action" in out


@pytest.mark.parametrize("action", distill._ACTIONS)
def test_every_action_reaches_its_handler(monkeypatch, action):
    seen = {}

    def fake(argv, prog):
        seen["argv"], seen["prog"] = argv, prog
        return 7
    monkeypatch.setitem(distill.HANDLERS, action, fake)
    assert distill.cmd_distill([action, "--x", "1"], prog="gmlx distill") == 7
    assert seen == {"argv": ["--x", "1"], "prog": f"gmlx distill {action}"}


def test_cache_requires_the_three_paths():
    rc, out = _run(["distill", "cache", "--teacher", "t.gguf"])
    assert rc == 2 and "--out" in out


def test_train_refuses_both_scale_conventions():
    rc, out = _run(["distill", "train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf",
                    "--iters", "1", "--lora-scale", "2", "--lora-alpha", "32"])
    assert rc == 2 and "one multiplier" in out


def test_cache_validate_runs_the_validator(tmp_path):
    from distill.test_distill_lib import _tiny_cache, tok_bl  # noqa: F401
    tok = tok_bl.__wrapped__()
    _tiny_cache(tmp_path / "c", tok)
    rc, out = _run(["distill", "cache", "--validate", str(tmp_path / "c")])
    assert rc == 0 and out.strip().endswith("valid")
    (tmp_path / "c" / "manifest.json").write_text("{}")
    rc, out = _run(["distill", "cache", "--validate", str(tmp_path / "c")])
    assert rc == 1


def test_completion_offers_actions_then_flags():
    vals = [ln.split("\t", 1)[0] for ln in completion._complete(["distill", ""])]
    assert vals == list(distill._ACTIONS)
    flags = [ln.split("\t", 1)[0] for ln in completion._complete(["distill", "align", "--"])]
    assert "--materialize" in flags and "--teacher" not in flags
    assert completion._complete(["distill", "cache", "--teacher", ""]) == ["::files"]
    assert completion._complete(["distill", "train", "--loss", ""]) == ["bucketed", "paper", "renorm"]


def test_train_refuses_a_non_gguf_student(tmp_path, capsys):
    from gmlx.commands.distill import cmd_train
    (tmp_path / "config.json").write_text("{}")
    rc = cmd_train(["--view", str(tmp_path), "--student", str(tmp_path), "--adapter-out",
                    str(tmp_path / "a.gguf"), "--iters", "1"])
    assert rc == 2
    assert "GGUF" in capsys.readouterr().err
