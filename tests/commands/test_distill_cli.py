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
    # a shard cut short by a kill is reported, not raised on
    shard = tmp_path / "c" / "batch-00000.safetensors"
    shard.write_bytes(shard.read_bytes()[:100])
    rc, out = _run(["distill", "cache", "--validate", str(tmp_path / "c")])
    assert rc == 1 and "shard 0 sha256 mismatch" in out
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
    ["cache", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "d", "--top-k", "0"],
    ["cache", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "d", "--max-len", "0"],
    ["cache", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "d", "--hidden-dim", "0"],
    ["align", "--cache", "c", "--student", "s.gguf", "--out", "v", "--kprime", "0"],
    ["train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf", "--iters", "0"],
    ["train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf", "--iters", "1",
     "--val-batches", "0"],
    ["eval", "--student", "s.gguf", "--md", "r.md", "--json", "r.json", "--batch-size", "0"],
    ["eval", "--student", "s.gguf", "--md", "r.md", "--json", "r.json", "--max-len", "0"],
    ["eval", "--student", "s.gguf", "--md", "r.md", "--json", "r.json", "--chat-max-len", "0"],
    ["train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf", "--iters", "1",
     "--lora-rank", "0"],
    ["eval", "--student", "s.gguf", "--md", "r.md", "--json", "r.json", "--gsm8k-max-tokens", "0"],
    ["eval", "--student", "s.gguf", "--md", "r.md", "--json", "r.json", "--chat-max-tokens", "-1"],
    ["eval", "--student", "s.gguf", "--md", "r.md", "--json", "r.json", "--task-limit", "0"],
    ["eval", "--student", "s.gguf", "--md", "r.md", "--json", "r.json", "--kld-rows", "0"],
    ["gen", "--teacher", "t.gguf", "--prompts", "p.jsonl", "--out", "o.jsonl", "--thinking-budget", "0"],
    ["gen", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "o.jsonl", "--prefix-chars", "0"],
    ["cache", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "d", "--limit-docs", "0"],
    ["cache", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "d", "--max-rows", "0"],
    ["filter", "--in", "a.jsonl", "--out", "o.jsonl", "--ngram", "0"],
    ["filter", "--in", "a.jsonl", "--out", "o.jsonl", "--max-line-repeats", "0"],
    ["census", "--without", "a", "--with", "b", "--out", "c.json", "--max-rows", "0"],
])
def test_zero_cadence_and_size_flags_are_refused_at_parse_time(argv):
    """A zero interval would divide the loop by zero after the load; the
    parser refuses it."""
    rc, out = _run(["distill", *argv])
    assert rc == 2 and "positive" in out, out


@pytest.mark.parametrize("argv", [
    ["train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf", "--iters", "1", "--seed", "-1"],
    ["align", "--cache", "c", "--student", "s.gguf", "--out", "v", "--seed", "-1"],
    ["gen", "--teacher", "t.gguf", "--prompts", "p.jsonl", "--out", "o.jsonl", "--seed", "-1"],
    ["gen", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "o.jsonl", "--docs", "-1"],
    ["gen", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "o.jsonl", "--min-chars", "-1"],
    ["cache", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "d", "--hidden", "--hidden-seed", "-1"],
])
def test_negative_seeds_and_counts_are_refused_at_parse_time(argv):
    """A negative seed would fail in the generator after the load; the
    parser refuses it, and zero stays valid where it means all."""
    rc, out = _run(["distill", *argv])
    assert rc == 2 and "at least 0" in out, out


@pytest.mark.parametrize("argv", [
    ["align", "--cache", "c", "--student", "s.gguf", "--out", "v", "--max-chunk-len", "0"],
    ["align", "--cache", "c", "--student", "s.gguf", "--out", "v", "--gamma", "0"],
    ["train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf", "--iters", "1", "--gamma", "-1"],
])
def test_non_positive_chunk_knobs_are_refused_at_parse_time(argv):
    """A zero chunk length keeps no chunk and a zero gamma keeps every
    chunk the log of zero would drop; the parser refuses both."""
    rc, out = _run(["distill", *argv])
    assert rc == 2 and "positive" in out, out


def test_distill_paths_expand_a_home_relative_argument(tmp_path, monkeypatch):
    """A quoted ~ reaches the parser unexpanded; every path argument
    expands it, so the refusal names the real path and no literal ~
    directory is ever written."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    rc, out = _run(["distill", "eval", "--student", "~/nope.gguf", "--md", "~/r.md", "--json", "~/r.json"])
    assert rc == 2 and str(tmp_path / "nope.gguf") in out and "~/" not in out
    assert not (tmp_path / "~").exists()


def test_distill_gen_tokenizer_expands_a_home_relative_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = tmp_path / "p.jsonl"
    p.write_text('{"id": "a", "messages": [{"role": "user", "content": "hi"}]}\n')
    rc, out = _run(["distill", "gen", "--base-url", "http://127.0.0.1:9/v1", "--prompts", str(p), "--out",
                    str(tmp_path / "o.jsonl"), "--thinking", "--thinking-budget", "40", "--tokenizer", "~/t.gguf"])
    assert rc == 2 and f"cannot load the tokenizer from {tmp_path / 't.gguf'}" in out and "~/" not in out


def test_cache_counts_take_positive_integers():
    for flag in ("--max-tokens", "--trunk"):
        rc, out = _run(["distill", "cache", "--teacher", "t.gguf", "--corpus", "c.jsonl", "--out", "d", flag, "0"])
        assert rc == 2 and "a positive integer is required, got 0" in out



def test_filter_help_names_the_trace_repeat_threshold():
    rc, out = _run(["distill", "filter", "--help"])
    assert rc == 0 and "--max-trace-repeat" in out



@pytest.mark.parametrize("argv", [
    ["gen", "--teacher", "t.gguf", "--prompts", "p.jsonl", "--out", "o.jsonl", "--max-tokens", "0"],
    ["filter", "--in", "a.jsonl", "--out", "o.jsonl", "--max-reply-tokens", "0"],
])
def test_zero_token_budgets_are_refused_at_parse_time(argv):
    rc, out = _run(["distill", *argv])
    assert rc == 2 and "positive" in out, out


@pytest.mark.parametrize("flag", ["--max-repeat", "--max-trace-repeat", "--max-non-ascii"])
@pytest.mark.parametrize("value", ["-0.1", "1.5", "x"])
def test_fraction_flags_take_a_value_in_the_unit_interval(flag, value):
    rc, out = _run(["distill", "filter", "--in", "a.jsonl", "--out", "o.jsonl", flag, value])
    assert rc == 2 and "a fraction between 0 and 1 is required" in out, out


_TRAIN = ["distill", "train", "--view", "v", "--student", "s.gguf", "--adapter-out", "a.gguf", "--iters", "1"]


@pytest.mark.parametrize("flag,value,word", [
    ("--warmup", "5", "fraction"),
    ("--lora-dropout", "1.0", "below 1"),
    ("--lora-dropout", "-0.1", "below 1"),
    ("--lr", "0", "positive"),
    ("--weight-decay", "-1", "at least 0"),
    ("--dk", "-1", "at least 0"),
    ("--alm", "-0.5", "at least 0"),
    ("--ce", "-1", "at least 0"),
    ("--hs", "-1", "at least 0"),
])
def test_train_flags_refuse_values_out_of_range_at_parse_time(flag, value, word):
    """A warmup past 1 never reaches the peak rate, a dropout of 1 raises
    after the student loads, and a negative rate or loss weight trains the
    wrong way; the parser refuses each."""
    rc, out = _run(_TRAIN + [flag, value])
    assert rc == 2 and word in out


def test_train_refuses_when_every_loss_weight_is_zero():
    rc, out = _run(_TRAIN + ["--dk", "0", "--alm", "0", "--ce", "0"])
    assert rc == 2 and "[train] refuse: every loss weight is 0" in out


@pytest.mark.parametrize("argv, word", [
    (_TRAIN + ["--lora-scale", "0"], "nonzero"),
    (_TRAIN + ["--lora-alpha", "nan"], "nonzero"),
    (_TRAIN + ["--lr", "inf"], "positive"),
    (_TRAIN + ["--weight-decay", "inf"], "at least 0"),
    (["distill", "align", "--cache", "c", "--student", "s.gguf", "--out", "v", "--w-mid", "-1"], "at least 0"),
    (["distill", "cache", "--teacher", "t.gguf", "--corpus", "c", "--out", "o", "--cache-limit-gb", "-1"],
     "at least 0"),
])
def test_float_flags_refuse_zero_nan_and_infinite_values_at_parse_time(argv, word):
    """A LoRA scale of 0 trains an adapter that changes nothing, and inf or
    nan anywhere reaches the optimizer as inf or nan; the parser refuses
    each."""
    rc, out = _run(argv)
    assert rc == 2 and word in out


_GEN = ["distill", "gen", "--out", "o.jsonl", "--prompts", "p.jsonl", "--base-url", "http://h"]
_CACHE = ["distill", "cache", "--teacher", "t.gguf", "--corpus", "c", "--out", "o"]
_ALIGN = ["distill", "align", "--cache", "c", "--student", "s.gguf", "--out", "v"]
_EVAL = ["distill", "eval", "--student", "s.gguf", "--md", "r.md", "--json", "r.json"]
_CENSUS = ["distill", "census", "--without", "a", "--with", "b", "--out", "c.json"]


@pytest.mark.parametrize("argv, word", [
    (_GEN + ["--temperature", "nan"], "at least 0"),
    (_GEN + ["--top-p", "2"], "between 0 and 1"),
    (_GEN + ["--min-p", "-1"], "between 0 and 1"),
    (_GEN + ["--timeout", "0"], "positive"),
    (_GEN + ["--startup-timeout", "nan"], "positive"),
    (_CACHE + ["--logits-cap-gb", "nan"], "positive"),
    (_CACHE + ["--max-disk-gb", "nan"], "at least 0"),
    (_CACHE + ["--expert-bytes-gb", "-1"], "at least 0"),
    (_ALIGN + ["--max-disk-gb", "inf"], "at least 0"),
    (_ALIGN + ["--val-fraction", "5"], "between 0 and 1"),
    (_EVAL + ["--decontam-threshold", "nan"], "between 0 and 1"),
    (_CENSUS + ["--delta-threshold", "nan"], "finite"),
])
def test_the_remaining_float_flags_refuse_nan_and_out_of_range_values_at_parse_time(argv, word):
    """A nan cap never refuses anything (every comparison is False), a nan
    sampling value fails every request after the server started, and a
    validation fraction past 1 holds out the wrong rows after the whole
    row loop; the parser refuses each."""
    rc, out = _run(argv)
    assert rc == 2 and word in out


@pytest.mark.parametrize("port", ["0", "70000"])
def test_gen_refuses_a_port_outside_the_tcp_range_at_parse_time(port):
    """Port 0 serves on a random port that the readiness poll never finds,
    and a port past 65535 raises inside the socket call; the parser
    refuses both."""
    rc, out = _run(_GEN + ["--port", port])
    assert rc == 2 and "a port between 1 and 65535 is required" in out
