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


@pytest.mark.parametrize("command", ["run", "chat", "serve", "capture"])
def test_every_text_command_parses_the_flag_one_way(command):
    parse = _parsers()[command]
    args = parse(["--adapter", "my:dir/a.gguf:0.5", "--adapter", "b.gguf"])
    assert args.adapter == [("my:dir/a.gguf", 0.5), ("b.gguf", 1.0)]
    assert parse([]).adapter is None
