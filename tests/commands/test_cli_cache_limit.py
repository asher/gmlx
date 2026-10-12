#!/usr/bin/env python3
"""The in-process verbs (gmlx run, gmlx chat) bound the MLX buffer cache
with the serve policy before the model loads. Unbounded, MLX keeps freed
buffers up to its memory limit (1.5x the working set), and a deep prompt
piles them up (7.5 GiB cached after an 80K-token run --mtp prefill).
CPU-only: every load seam is a stub."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

import gmlx.commands.cli as cli
import gmlx.gen.prefill_decay as prefill_decay
import gmlx.serve.capacity as capacity
import gmlx.serve.memory as memory

GIB = memory.GIB
WS = 100 * GIB
AUTO = pytest.approx(5 * GIB, abs=1)     # 5 percent of the working set


@pytest.fixture(autouse=True)
def no_family_defaults(monkeypatch):
    """Keep argparse defaults: family seeding would fill flags we didn't pass."""
    monkeypatch.setenv("GMLX_NO_FAMILY_DEFAULTS", "1")


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUF")
    return str(p)


@pytest.fixture
def events(monkeypatch):
    """A fixed working set; records each cache limit set and each model
    load, in order. The process limit itself is left alone."""
    monkeypatch.delenv("GMLX_CACHE_LIMIT_GB", raising=False)
    monkeypatch.setattr(capacity, "working_set_bytes", lambda: float(WS))
    monkeypatch.setattr(prefill_decay, "_NOTED_CACHE_LIMIT", None)
    seen: list = []

    def record(n):
        seen.append(int(n))
        return 0

    monkeypatch.setattr(mx, "set_cache_limit", record)
    return seen


@pytest.fixture
def plain(monkeypatch, events):
    import gmlx.gen.generation as generation
    import gmlx.load.loader as loader

    def fake_load(*a, **k):
        events.append("load")
        return object(), {}, object()

    monkeypatch.setattr(loader, "load_model", fake_load)
    monkeypatch.setattr(generation, "generate", lambda *a, **k: None)
    return events


@pytest.fixture
def spec(monkeypatch, events):
    import gmlx.gen.generation as generation
    import gmlx.spec.mtp_load as mtp_load

    class _Tok:
        chat_template = None

    def fake_load(*a, **k):
        events.append("load")
        return object(), object(), {}, _Tok()

    monkeypatch.setattr(mtp_load, "load_mtp_model", fake_load)
    monkeypatch.setattr(
        generation, "generate_speculative",
        lambda *a, **k: {"text": "", "tokens": 1, "elapsed_s": 0.1,
                         "prefill_s": 0.0, "decode_tps": 10.0,
                         "accept_rate": 0.5, "mean_accept_len": 1.5,
                         "rounds": 1})
    return events


def test_run_bounds_the_cache_before_the_load(gguf, plain):
    assert cli.main([gguf, "--prompt", "hi", "--no-mtp"]) == 0
    assert plain == [AUTO, "load"]
    assert prefill_decay._NOTED_CACHE_LIMIT == AUTO


def test_run_mtp_bounds_the_cache_before_the_load(gguf, spec):
    assert cli.main([gguf, "--prompt", "hi", "--mtp"]) == 0
    assert spec == [AUTO, "load"]


def test_run_takes_the_env_limit(gguf, plain, monkeypatch):
    monkeypatch.setenv("GMLX_CACHE_LIMIT_GB", "2")
    assert cli.main([gguf, "--prompt", "hi", "--no-mtp"]) == 0
    assert plain == [2 * GIB, "load"]


def test_run_env_off_keeps_the_mlx_default(gguf, plain, monkeypatch):
    monkeypatch.setenv("GMLX_CACHE_LIMIT_GB", "off")
    assert cli.main([gguf, "--prompt", "hi", "--no-mtp"]) == 0
    assert plain == ["load"]


def _chat_args(gguf, assistant=False):
    return SimpleNamespace(assistant=assistant, gguf=gguf, mmproj=None,
                           draft_gguf=None, stream_experts=False,
                           verbose=False)


def test_chat_bounds_the_cache_before_the_load(gguf, events, monkeypatch):
    import gmlx.commands.tool_preflight as tool_preflight
    import gmlx.tui.chat as chat

    monkeypatch.setattr(tool_preflight, "check_or_exit", lambda *a, **k: None)
    monkeypatch.setattr(chat, "_backend_plain_text",
                        lambda args, kv: events.append("load") or "backend")
    backend = chat._load_chat_backend(_chat_args(gguf), {},
                                      speculative=False, vlm_mtp=False)
    assert backend == "backend"
    assert events == [AUTO, "load"]


def test_chat_assistant_loads_nothing_and_sets_no_limit(gguf, events):
    import gmlx.tui.chat as chat

    chat._load_chat_backend(_chat_args(gguf, assistant=True), {},
                            speculative=False, vlm_mtp=False)
    assert events == []
