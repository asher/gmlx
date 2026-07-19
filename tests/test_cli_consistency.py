#!/usr/bin/env python3
"""Cross-verb CLI consistency (the P2 flag-alignment pass): the serve flag
aliases (--from-hf-cache/--hf-cache), the unified recursion toggle, serve
--print-config, chat --no-chat-template, and train's --config model resolution.
All CPU-only - no GGUF is parsed and no server starts."""
from __future__ import annotations

import argparse

import pytest

from gmlx import chat as chatmod
from gmlx import server as srv
from gmlx import train as trainmod


def _serve_ns(argv):
    ap = argparse.ArgumentParser()
    srv._add_serve_args(ap)
    return ap.parse_args(argv)


# --- run/chat: -v/--verbose parity (load diagnostics vs spinner) ---
def test_run_chat_verbose_flag_parity():
    from gmlx import cli as climod

    for build in (climod._build_parser, chatmod._build_parser):
        ap = build()
        ns = ap.parse_args(["m.gguf"])
        assert ns.verbose is False
        assert ap.parse_args(["m.gguf", "-v"]).verbose is True
        assert ap.parse_args(["m.gguf", "--verbose"]).verbose is True


# --- serve: hf-cache alias + recursion toggle + print-config flag ---
def test_serve_hf_cache_aliases_same_dest():
    assert _serve_ns(["--hf-cache"]).hf_cache is True
    assert _serve_ns(["--from-hf-cache"]).hf_cache is True
    assert _serve_ns([]).hf_cache is False


def test_serve_recursion_defaults_shallow_and_toggles():
    assert _serve_ns([]).recursive is False
    assert _serve_ns(["-r"]).recursive is True
    assert _serve_ns(["--recursive"]).recursive is True
    assert _serve_ns(["--no-recursive"]).recursive is False


def test_serve_print_config_flag():
    assert _serve_ns([]).print_config is False
    assert _serve_ns(["--print-config"]).print_config is True


# --- serve --print-config end-to-end (single-model mode; no GGUF read) ---
def test_serve_print_config_dumps_and_exits(tmp_path, capsys):
    g = tmp_path / "Qwen3-0.6B-Q4_K_M.gguf"
    g.write_text("x")
    rc = srv._cmd_serve(["--print-config", str(g)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# effective gmlx config" in out
    assert "host: 127.0.0.1" in out
    assert "models:" in out
    assert str(g) in out          # the resolved model path is in the dump


def test_serve_print_config_round_trips_through_load_config(tmp_path, capsys):
    """--print-config output is a valid --config file: the dump uses the on-disk
    schema shape and every key survives load_config."""
    from gmlx.config import load_config
    g = tmp_path / "Qwen3-0.6B-Q4_K_M.gguf"
    g.write_text("x")
    cfg_in = tmp_path / "cfg.yaml"
    cfg_in.write_text(
        "server:\n"
        "  port: 9090\n"
        "  defaults: {ttl_s: 600}\n"
        "profiles:\n"
        "  slow: {sampling: {temperature: 0.2, max_tokens: 2048}}\n"
        "  slower: {extends: slow, load: {kv_bits: 8}}\n"
        "rules:\n"
        "  - {match: 'qw*', profile: slow}\n"
        f"models:\n"
        f"  qw:\n"
        f"    path: {g}\n"
        f"    family: qwen3.6\n"
        f"    profiles: {{coding: {{sampling: {{min_p: 0.05}}}}}}\n"
        f"    overrides: {{sampling: {{top_k: 50}}}}\n"
        "aliases: {fast: qw@coding}\n")
    rc = srv._cmd_serve(["--print-config", "--config", str(cfg_in)])
    assert rc == 0
    out = capsys.readouterr().out
    dumped = tmp_path / "dumped.yaml"
    dumped.write_text(out)
    cfg = load_config(dumped)                 # must not raise
    assert cfg.port == 9090
    assert cfg.defaults.ttl_s == 600
    assert cfg.profiles["slower"].extends == "slow"
    assert cfg.rules[0].match == "qw*"
    assert cfg.models["qw"].family == "qwen3.6"
    assert cfg.models["qw"].profiles == {"coding": {"sampling": {"min_p": 0.05}}}
    assert cfg.aliases == {"fast": "qw@coding"}


def test_serve_background_broken_config_fails_fast(tmp_path, capsys):
    """A broken config used to background silently with defaults; the parent
    now surfaces the ConfigError and exits."""
    cfg_in = tmp_path / "cfg.yaml"
    cfg_in.write_text("server:\n  bogus_key: 1\n")
    rc = srv._cmd_serve(["--config", str(cfg_in)])
    assert rc == 2
    assert "bogus_key" in capsys.readouterr().err


# --- chat: --no-chat-template parses (the run<->chat alignment) ---
def test_chat_no_chat_template_flag():
    ap = chatmod._build_parser()
    assert ap.parse_args(["model.gguf"]).no_chat_template is False
    assert ap.parse_args(["model.gguf", "--no-chat-template"]).no_chat_template is True


def test_train_bad_data_dir_fails_before_load(tmp_path, monkeypatch, capsys):
    # A path-shaped --data that isn't a dataset dir must fail BEFORE the model
    # load, not as an HF repo-id error after the base is in memory.
    base = tmp_path / "base.gguf"
    base.write_text("x")

    def _never(*a, **kw):
        raise AssertionError("train_lora must not run with a bad --data")

    monkeypatch.setattr(trainmod, "train_lora", _never)
    rc = trainmod.cmd_train([str(base), "--data", "./no-such-data",
                             "--adapter-out", str(tmp_path / "a.gguf")])
    assert rc == 2
    assert "no such directory" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = trainmod.cmd_train([str(base), "--data", str(empty),
                             "--adapter-out", str(tmp_path / "a.gguf")])
    assert rc == 2
    assert "train.jsonl" in capsys.readouterr().err


# --- train: --config resolves a model id/alias to its path (like run/chat) ---
def test_train_config_resolves_model_id(tmp_path, monkeypatch):
    base = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    base.write_text("x")
    (tmp_path / "train.jsonl").write_text("{}\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"models:\n  mybase:\n    path: {base}\n")

    seen = {}

    def _fake_train(gguf_path, data, out_path, **kw):
        seen["base"] = gguf_path
        return out_path, 3

    monkeypatch.setattr(trainmod, "train_lora", _fake_train)
    rc = trainmod.cmd_train(
        ["mybase", "--config", str(cfg), "--data", str(tmp_path),
         "--adapter-out", str(tmp_path / "a.gguf")])
    assert rc == 0
    assert seen["base"] == str(base)          # id resolved to the config path


def test_train_bare_path_passes_through(tmp_path, monkeypatch):
    base = tmp_path / "base.gguf"
    base.write_text("x")
    (tmp_path / "train.jsonl").write_text("{}\n")
    seen = {}
    monkeypatch.setattr(
        trainmod, "train_lora",
        lambda g, d, o, **kw: (seen.update(base=g), (o, 1))[1])
    rc = trainmod.cmd_train(
        [str(base), "--data", str(tmp_path), "--adapter-out", str(tmp_path / "a.gguf")])
    assert rc == 0
    assert seen["base"] == str(base)          # an on-disk path is used verbatim


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
