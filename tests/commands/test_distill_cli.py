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


def test_argparse_errors_print_as_refusals():
    rc, out = _run(["distill", "gen", "--no-such-flag"])
    assert rc == 2
    assert "usage: gmlx distill gen" in out and "[gen] refuse:" in out


def test_flag_aliases_share_a_destination():
    from gmlx.commands.distill import _filter_parser, _gen_parser
    assert _filter_parser("gmlx distill filter").parse_args(["--in", "a", "--out", "b", "--min-words", "1"]).min_words == 1
    assert _filter_parser("gmlx distill filter").parse_args(["--in", "a", "--out", "b", "--min-tokens", "2"]).min_words == 2
    assert _gen_parser("gmlx distill gen").parse_args(["--out", "o", "--model", "s.gguf"]).teacher == "s.gguf"


def test_gen_refuses_a_missing_model(tmp_path, capsys):
    from gmlx.commands.distill import cmd_gen
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"id": "a", "messages": [{"role": "user", "content": "hi"}]}\n')
    rc = cmd_gen(["--teacher", str(tmp_path / "none.gguf"), "--prompts", str(prompts),
                  "--out", str(tmp_path / "o.jsonl")])
    assert rc == 2
    assert "[gen] refuse: no model at" in capsys.readouterr().err


def test_align_refuses_missing_student_and_tables(tmp_path, capsys):
    from gmlx.commands.distill import cmd_align
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    rc = cmd_align(["--cache", str(cache), "--student", str(tmp_path / "none.gguf"), "--out", str(tmp_path / "v")])
    assert rc == 2 and "[align] refuse: no student at" in capsys.readouterr().err
    student = tmp_path / "s.gguf"
    student.write_bytes(b"")
    rc = cmd_align(["--cache", str(cache), "--student", str(student), "--out", str(tmp_path / "v"),
                    "--tables", str(tmp_path / "no-tables")])
    assert rc == 2 and "[align] refuse: no tables.json in" in capsys.readouterr().err
    assert not (tmp_path / "v").exists()


def test_bad_render_kwargs_refuse(tmp_path, capsys):
    from gmlx.commands.distill import cmd_align, cmd_cache, cmd_eval, cmd_gen
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    student = tmp_path / "s.gguf"
    student.write_bytes(b"")
    bad = '{"enable_thinking": True}'
    rc = cmd_cache(["--teacher", str(student), "--corpus", str(tmp_path / "c.jsonl"), "--out", str(tmp_path / "o"),
                    "--frame-kwargs", bad])
    assert rc == 2 and "[cache] refuse: --frame-kwargs" in capsys.readouterr().err
    rc = cmd_align(["--cache", str(cache), "--student", str(student), "--out", str(tmp_path / "v"),
                    "--frame-kwargs", bad])
    assert rc == 2 and "[align] refuse: --frame-kwargs" in capsys.readouterr().err
    rc = cmd_eval(["--student", str(student), "--md", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json"),
                   "--frame-kwargs", bad])
    assert rc == 2 and "[eval] refuse:" in capsys.readouterr().err
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"id": "a", "messages": [{"role": "user", "content": "hi"}]}\n')
    rc = cmd_gen(["--teacher", str(student), "--prompts", str(prompts), "--out", str(tmp_path / "g.jsonl"),
                  "--chat-template-kwargs", bad])
    assert rc == 2 and "[gen] refuse: --chat-template-kwargs" in capsys.readouterr().err


def test_filter_refuses_a_malformed_jsonl_line(tmp_path, capsys):
    from gmlx.commands.distill import cmd_filter
    bad = tmp_path / "in.jsonl"
    bad.write_text('{"id": "a", "messages": [{"role": "user", "content": "hi"}]}\nnot json\n')
    rc = cmd_filter(["--in", str(bad), "--out", str(tmp_path / "out.jsonl")])
    err = capsys.readouterr().err
    assert rc == 2 and "[filter] refuse:" in err and "line 2" in err


@pytest.mark.parametrize("argv", [
    ["train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf", "--iters", "1", "--save-every", "0"],
    ["train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf", "--iters", "1", "--val-every", "0"],
    ["train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf", "--iters", "1",
     "--report-every", "-1"],
    ["train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf", "--iters", "1", "--batch-size", "0"],
    ["gen", "--teacher", "t.gguf", "--prompts", "p.jsonl", "--out", "o.jsonl", "--concurrency", "0"],
    ["gen", "--teacher", "t.gguf", "--prompts", "p.jsonl", "--out", "o.jsonl", "--report-every", "0"],
    ["cache", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "d", "--rows-per-shard", "0"],
])
def test_zero_cadence_and_size_flags_are_refused_at_parse_time(argv):
    """A zero interval would divide the loop by zero after the load; the
    parser refuses it."""
    rc, out = _run(["distill", *argv])
    assert rc == 2 and "positive" in out, out
