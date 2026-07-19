#!/usr/bin/env python3
"""VLM serving bridge: mmproj association + the VLM branch of
``load_serveable_model`` + bridge routing. CPU-only - a fake ``load_vlm_model``
and a sentinel original loader, so no GPU, no GGUF files, no real model load."""
from __future__ import annotations

import os
import sys
import types

import pytest

pytest.importorskip("mlx_vlm")

from gmlx import server_bridge_vlm as serving  # noqa: E402

_ENV_KEYS = ("MLX_VLM_GGUF_MMPROJ", "MLX_VLM_GGUF_HF_SOURCE")


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Each test sees an empty registry, no association env vars, and no in-flight
    build spec."""
    serving._GGUF_VLM_REGISTRY.clear()
    serving.set_build_spec(None)
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield
    serving._GGUF_VLM_REGISTRY.clear()
    serving.set_build_spec(None)


# mmproj association: registry + env fallback
def test_register_and_resolve_keyed_by_abspath():
    serving.register_gguf_vlm("/m/llm.gguf", "/m/mmproj.gguf", hf_source="hf/repo")
    spec = serving._resolve_vlm_spec("/m/llm.gguf")
    assert spec == {"mmproj_path": "/m/mmproj.gguf", "hf_source": "hf/repo"}
    # A relative path resolving to the same file hits the same entry.
    here = os.path.basename("/m/llm.gguf")  # 'llm.gguf'
    serving.register_gguf_vlm("./x.gguf", "./y.gguf")
    assert serving._resolve_vlm_spec(os.path.abspath("./x.gguf"))[
        "mmproj_path"
    ] == os.path.abspath("./y.gguf")
    assert here  # silence lint; abspath keying is the real assertion above


def test_unregistered_text_gguf_resolves_none():
    assert serving._resolve_vlm_spec("/m/plain-text.gguf") is None


def test_env_fallback_for_single_model_launch(monkeypatch):
    monkeypatch.setenv("MLX_VLM_GGUF_MMPROJ", "/env/mmproj.gguf")
    monkeypatch.setenv("MLX_VLM_GGUF_HF_SOURCE", "env/repo")
    spec = serving._resolve_vlm_spec("/some/unregistered.gguf")
    assert spec == {"mmproj_path": "/env/mmproj.gguf", "hf_source": "env/repo"}


def test_registry_wins_over_env(monkeypatch):
    monkeypatch.setenv("MLX_VLM_GGUF_MMPROJ", "/env/mmproj.gguf")
    serving.register_gguf_vlm("/m/llm.gguf", "/m/explicit.gguf")
    assert serving._resolve_vlm_spec("/m/llm.gguf")["mmproj_path"] == "/m/explicit.gguf"


# load_serveable_model VLM branch (fake load_vlm_model)
class _FakeConfig:
    model_type = "gemma4_unified"
    image_token_id = 258880


class _FakeVLMModel:
    def __init__(self):
        self.config = _FakeConfig()

    def get_input_embeddings(self, *a, **k):  # pragma: no cover - presence only
        raise NotImplementedError


@pytest.fixture
def fake_load_vlm_model(monkeypatch):
    """Stub ``gmlx.vlm.load_vlm_model`` without importing the heavy real
    module; records its call and returns ``(model, config_dict, processor)``
    (the loader's order, which the server branch must reorder)."""
    calls = []
    processor = object()

    def _fake(gguf_path, mmproj_path, *, hf_source=None, verbose=True):
        calls.append(
            {"gguf": gguf_path, "mmproj": mmproj_path,
             "hf_source": hf_source, "verbose": verbose}
        )
        return _FakeVLMModel(), {"model_type": "gemma4_unified"}, processor

    mod = types.ModuleType("gmlx.vlm")
    mod.load_vlm_model = _fake
    monkeypatch.setitem(sys.modules, "gmlx.vlm", mod)
    return calls, processor


def test_vlm_branch_returns_model_processor_config(fake_load_vlm_model):
    calls, processor = fake_load_vlm_model
    model, proc, config = serving.load_serveable_model(
        "/m/llm.gguf", mmproj_path="/m/mmproj.gguf", hf_source="hf/repo"
    )
    # Reordered (model, processor, config) and config is the model's dataclass,
    # not the synthesized dict the loader returns 2nd.
    assert isinstance(model, _FakeVLMModel)
    assert proc is processor
    assert config is model.config
    assert config.model_type == "gemma4_unified"
    # Loaded quietly, with hf_source threaded and mmproj passed through verbatim.
    assert calls == [{"gguf": "/m/llm.gguf", "mmproj": "/m/mmproj.gguf",
                      "hf_source": "hf/repo", "verbose": False}]


def test_text_branch_does_not_touch_vlm_loader(fake_load_vlm_model, monkeypatch):
    calls, _ = fake_load_vlm_model
    tokenizer = types.SimpleNamespace(eos_token_id=1)  # settable; no _tokenizer
    monkeypatch.setattr(serving, "load_model",
                        lambda *a, **k: (_FakeVLMModel(), {}, tokenizer),
                        raising=True)
    monkeypatch.setattr(serving, "TextOnlyModel",
                        lambda raw, config=None: types.SimpleNamespace(config={}),
                        raising=True)
    monkeypatch.setattr(serving, "_build_detokenizer", lambda backend: object())
    monkeypatch.setattr(serving, "StoppingCriteria",
                        lambda eos, backend: object(), raising=True)
    serving.load_serveable_model("/m/plain.gguf")  # no mmproj
    assert calls == []  # the VLM loader was never invoked for a text GGUF


# bridge routing: text vs VLM vs non-gguf
@pytest.fixture
def installed_bridge(monkeypatch):
    """Install the bridge against a sentinel original loader and a spy
    ``load_serveable_model``; returns ``(patched_loader, original_calls,
    serveable_calls)``."""
    import importlib

    gen = importlib.import_module("mlx_vlm.server.generation")
    original_calls = []
    serveable_calls = []

    def sentinel(model_path, adapter_path=None):
        original_calls.append((model_path, adapter_path))
        return ("ORIG", model_path)

    def spy(model_path, *, mmproj_path=None, hf_source=None,
            speculative=False, draft_gguf_path=None, chat_template=None,
            adapter_gguf=None, stream=None, moe_expert_mass=None,
            feeder_prefill=None, feeder_decode=None):
        serveable_calls.append((model_path, mmproj_path, hf_source, chat_template))
        return ("SERVEABLE", model_path, mmproj_path)

    # Fresh install over the sentinel so `original` is the sentinel.
    monkeypatch.setattr(gen, serving._BRIDGE_FLAG, False, raising=False)
    monkeypatch.setattr(gen, "load_model_resources", sentinel, raising=False)
    monkeypatch.setattr(serving, "load_serveable_model", spy)
    serving.install_gguf_server_bridge()
    return gen.load_model_resources, original_calls, serveable_calls


def test_non_gguf_falls_through_to_original(installed_bridge):
    patched, original_calls, serveable_calls = installed_bridge
    assert patched("mlx-community/Qwen2.5-VL-3B")[0] == "ORIG"
    assert original_calls == [("mlx-community/Qwen2.5-VL-3B", None)]
    assert serveable_calls == []


def test_text_gguf_routes_without_mmproj(installed_bridge):
    patched, _, serveable_calls = installed_bridge
    patched("/m/text.gguf")
    assert serveable_calls == [("/m/text.gguf", None, None, None)]


def test_vlm_gguf_routes_with_resolved_mmproj(installed_bridge):
    patched, _, serveable_calls = installed_bridge
    serving.register_gguf_vlm("/m/llm.gguf", "/m/mmproj.gguf", hf_source="hf/repo")
    patched("/m/llm.gguf")
    # VLM keeps its mmproj-synthesized processor template - no chat_template threaded.
    assert serveable_calls == [("/m/llm.gguf", "/m/mmproj.gguf", "hf/repo", None)]


def test_text_route_threads_build_spec_chat_template(installed_bridge):
    """The config-resolved spec's chat-template override reaches the text loader
    through the bridge. The bridge reads the *build* spec (published by residency
    under its build lock), NOT the request-thread ``_active_spec`` ContextVar -
    the bridge runs in the engine's load worker thread where that var is invisible
    (see test_serving_mtp.test_chat_template_crosses_into_worker_thread)."""
    patched, _, serveable_calls = installed_bridge
    spec = types.SimpleNamespace(chat_template="{{ pirate }}")
    serving.set_build_spec(spec)
    try:
        patched("/m/text.gguf")
    finally:
        serving.set_build_spec(None)
    assert serveable_calls == [("/m/text.gguf", None, None, "{{ pirate }}")]


def test_install_is_idempotent():
    import importlib

    gen = importlib.import_module("mlx_vlm.server.generation")
    serving.install_gguf_server_bridge()
    first = gen.load_model_resources
    serving.install_gguf_server_bridge()
    assert gen.load_model_resources is first
