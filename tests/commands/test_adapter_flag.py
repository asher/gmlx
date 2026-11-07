"""The shared ``--adapter PATH[:STRENGTH]`` flag: the strength rule, and
each command that takes it parses it the same way."""
from __future__ import annotations

import argparse

import pytest

from gmlx.commands.adapter_flag import (duplicate_adapter, format_adapter,
                                        parse_adapter)


@pytest.mark.parametrize("text, want", [
    ("a.safetensors", ("a.safetensors", 1.0)),
    ("a.safetensors:0.5", ("a.safetensors", 0.5)),
    ("a.safetensors:-1", ("a.safetensors", -1.0)),
    ("a.safetensors:1e-1", ("a.safetensors", 0.1)),
    ("my:lora.safetensors", ("my:lora.safetensors", 1.0)),
    ("my:lora.safetensors:2", ("my:lora.safetensors", 2.0)),
    ("dir/0.5", ("dir/0.5", 1.0)),
    ("a.safetensors:strong", ("a.safetensors:strong", 1.0)),
    (":0.5", (":0.5", 1.0)),
])
def test_the_adapter_tail_is_a_strength_only_when_it_is_a_float(text, want):
    assert parse_adapter(text) == want


@pytest.mark.parametrize("tail", ["1_0", "1_000.5", "1e1_0", "-0_5"])
def test_a_strength_with_an_underscore_is_refused(tail):
    assert float(tail)      # Python reads it, as 10 for the first one
    with pytest.raises(argparse.ArgumentTypeError,
                       match="without underscores"):
        parse_adapter(f"a.safetensors:{tail}")
    # An underscore in the path, or in a tail that is no number, is fine.
    assert parse_adapter("my_lora.safetensors:0.5") == \
        ("my_lora.safetensors", 0.5)
    assert parse_adapter("dir:my_lora_1") == ("dir:my_lora_1", 1.0)


@pytest.mark.parametrize("text", ["a.gguf:inf", "a.gguf:nan", ""])
def test_a_strength_that_is_not_finite_or_no_path_is_refused(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_adapter(text)


@pytest.mark.parametrize("path, strength", [
    ("/m/a.gguf", 0.5), ("/m/x:y.gguf", 1.0), ("/m/dir:2", 1e-5),
])
def test_a_formatted_value_parses_back(path, strength):
    assert parse_adapter(format_adapter(path, strength)) == (path, strength)


def test_a_path_named_twice_is_found():
    assert duplicate_adapter([("a.gguf", 1.0), ("./b.gguf", 1.0)]) is None
    assert duplicate_adapter([("a.gguf", 1.0), ("./a.gguf", 0.5)]) == "./a.gguf"


def _parsers():
    from gmlx.commands import capture, cli, kld
    from gmlx.serve import server
    from gmlx.tui import chat

    serve = argparse.ArgumentParser()
    server._add_serve_args(serve)
    return {
        "run": lambda argv: cli._build_parser().parse_args(["m.gguf", *argv]),
        "chat": lambda argv: chat._build_parser().parse_args(["m.gguf", *argv]),
        "serve": lambda argv: serve.parse_args(["m.gguf", *argv]),
        "capture": lambda argv: capture._build_parser("gmlx capture").parse_args(
            ["m.gguf", "--prompts", "a=a.jsonl", "--out", "o", *argv]),
        "kld": lambda argv: kld.build_parser().parse_args(
            ["m.gguf", "--text", "t.txt", *argv]),
    }


@pytest.mark.parametrize("command", ["run", "chat", "serve", "capture", "kld"])
def test_every_text_command_parses_the_flag_one_way(command):
    parse = _parsers()[command]
    args = parse(["--adapter", "my:dir/a.gguf:0.5", "--adapter", "b.gguf"])
    assert args.adapter == [("my:dir/a.gguf", 0.5), ("b.gguf", 1.0)]
    assert parse([]).adapter is None


class _Installed(Exception):
    pass


@pytest.mark.parametrize("argv", [
    ["--prompt", "hi"],
    ["--prompt", "hi", "--speculative"],
    ["--bench", "16"],
    ["--bench-depths", "16"],
], ids=["generate", "mtp", "bench", "bench-depths"])
def test_run_folds_the_strength_into_the_install(tmp_path, monkeypatch, argv):
    """run publishes no row vector, so each path installs with fold on."""
    from gmlx.commands import cli

    gguf, lora = tmp_path / "m.gguf", tmp_path / "a.gguf"
    gguf.write_bytes(b"GGUF")
    lora.write_bytes(b"GGUF")
    seen = []

    def loader(name, *out):
        return lambda *a, **k: (seen.append(name), out)[1]

    monkeypatch.setattr(cli, "maybe_load_from_config", lambda *a, **k: None)
    mtp = "--speculative" in argv
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda *a, **k: mtp)
    monkeypatch.setattr("gmlx.commands.tool_preflight.check_or_exit",
                        lambda *a, **k: None)
    monkeypatch.setattr("gmlx.load.loader.load_model",
                        loader("plain", object(), {}, object()))
    monkeypatch.setattr("gmlx.spec.mtp_load.load_mtp_model",
                        loader("mtp", object(), object(), {}, object()))

    def install(args, m, config, *, fold=True):
        seen.append((args.adapter, fold))
        raise _Installed

    monkeypatch.setattr(cli, "_apply_cli_adapter", install)
    with pytest.raises(_Installed):
        cli.main([str(gguf), "--adapter", f"{lora}:0.5", *argv])
    assert seen == ["mtp" if mtp else "plain", ([(str(lora), 0.5)], True)]
