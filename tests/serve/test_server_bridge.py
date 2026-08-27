#!/usr/bin/env python3
"""``mlx_lm.server`` bridge (server_bridge_lm.py): GGUF routing through the patched
``ModelProvider._load``. CPU-only - sentinel original loader and a fake
``load_model``, so no GGUF files, no real load."""
from __future__ import annotations

import types

import pytest

pytest.importorskip("mlx_lm")

from gmlx import server_bridge_lm as server_bridge  # noqa: E402


class _Provider:
    """Duck-typed ``ModelProvider`` self for the patched ``_load``."""

    is_distributed = False

    def __init__(self):
        self.model_key = "old"
        self.model = "old"
        self.tokenizer = "old"
        self.draft_model = "old"
        self.is_batchable = True
        self._tokenizer_config = {}
        self.cli_args = types.SimpleNamespace(use_default_chat_template=False)


@pytest.fixture
def installed(monkeypatch):
    """Install the bridge over a sentinel original ``_load``; returns
    ``(patched_load, original_calls)``."""
    from mlx_lm import server as _server

    original_calls = []

    def sentinel(self, model_path, adapter_path=None, draft_model_path=None):
        original_calls.append((model_path, adapter_path, draft_model_path))
        return "ORIG"

    monkeypatch.setattr(_server.ModelProvider, server_bridge._BRIDGE_FLAG,
                        False, raising=False)
    monkeypatch.setattr(_server.ModelProvider, "_load", sentinel)
    server_bridge.install_gguf_bridge()
    return _server.ModelProvider._load, original_calls


def test_gguf_adapter_raises_not_silent_noop(installed, monkeypatch):
    """--adapter on the GGUF route was a silent no-op (embedded in model_key,
    never applied); it must raise, pointing at the route that supports it."""
    patched, original_calls = installed
    monkeypatch.setattr(server_bridge, "load_model",
                        lambda *a, **k: pytest.fail("must not load the bare base"))
    provider = _Provider()
    with pytest.raises(ValueError, match="adapter_path is not supported"):
        patched(provider, "/m/x.gguf", "/m/adapter")
    assert original_calls == []
    assert provider.model == "old"      # state untouched, no half-loaded model


def test_non_gguf_adapter_falls_through_to_original(installed):
    patched, original_calls = installed
    assert patched(_Provider(), "mlx-community/Qwen3-4B", "/m/adapter") == "ORIG"
    assert original_calls == [("mlx-community/Qwen3-4B", "/m/adapter", None)]


def test_gguf_without_adapter_loads_sequential(installed, monkeypatch, capsys):
    patched, original_calls = installed
    tokenizer = types.SimpleNamespace(chat_template="t")
    monkeypatch.setattr(server_bridge, "load_model",
                        lambda p, chat_template=None, verbose=True:
                        ("MODEL", {}, tokenizer))
    monkeypatch.setattr(server_bridge, "make_prompt_cache", lambda m: None)
    provider = _Provider()
    patched(provider, "/m/x.gguf", None, "/m/draft.gguf")  # draft ignored
    assert original_calls == []
    assert provider.model == "MODEL"
    assert provider.tokenizer is tokenizer
    # The key keeps the caller's original draft path: upstream load()
    # recomputes it per request and reloads on any mismatch.
    assert provider.model_key == ("/m/x.gguf", None, "/m/draft.gguf")
    assert provider.draft_model is None
    assert provider.is_batchable is False
    # the dropped --draft-model is warned, not silent (the adapter case raises;
    # this one is unvalidated-but-tolerated, so a warning is the right loudness).
    assert "--draft-model is ignored" in capsys.readouterr().err


def test_gguf_without_draft_does_not_warn(installed, monkeypatch, capsys):
    patched, _ = installed
    tokenizer = types.SimpleNamespace(chat_template="t")
    monkeypatch.setattr(server_bridge, "load_model",
                        lambda p, chat_template=None, verbose=True:
                        ("MODEL", {}, tokenizer))
    monkeypatch.setattr(server_bridge, "make_prompt_cache", lambda m: None)
    patched(_Provider(), "/m/x.gguf", None, None)
    assert "--draft-model" not in capsys.readouterr().err


def test_register_resolved_models_waits_for_build_lock():
    """Reload's registry rebuild must serialize with residency's build lock:
    clearing the drafter stash mid-build strands the drafter that build is
    about to consume."""
    import threading
    import time
    import types

    from gmlx import server_bridge_vlm as bridge
    from gmlx.config import build_config

    cfg = build_config({"server": {}, "models": {}})
    lock = threading.Lock()
    pool = types.SimpleNamespace(_build_lock=lock)
    import importlib
    mod = importlib.import_module("mlx_vlm.server")
    saved = getattr(mod, "_kq_residency_pool", None)
    mod._kq_residency_pool = pool
    try:
        done = threading.Event()
        with lock:  # a build is in flight
            t = threading.Thread(
                target=lambda: (bridge.register_resolved_models(cfg),
                                done.set()))
            t.start()
            time.sleep(0.05)
            assert not done.is_set()  # blocked on the build lock
        t.join(5)
        assert done.is_set()
    finally:
        mod._kq_residency_pool = saved
