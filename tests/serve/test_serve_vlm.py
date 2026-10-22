#!/usr/bin/env python3
"""VLM serving bridge: mmproj association + the VLM branch of
``load_serveable_model`` + bridge routing. CPU-only - a fake ``load_vlm_model``
and a sentinel original loader, so no GPU, no GGUF files, no real model load."""
from __future__ import annotations

import os
import sys
import types

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import gmlx.serve.bridge_vlm as serving  # noqa: E402

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
        # A real mlx-vlm Model holds its text tower here. The streaming
        # placement must go on the tower, not on the wrapper.
        self.language_model = object()

    def get_input_embeddings(self, *a, **k):  # pragma: no cover - presence only
        raise NotImplementedError


@pytest.fixture
def fake_load_vlm_model(monkeypatch):
    """Stub ``gmlx.load.vlm.load_vlm_model`` without importing the heavy real
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

    mod = types.ModuleType("gmlx.load.vlm")
    mod.load_vlm_model = _fake
    monkeypatch.setitem(sys.modules, "gmlx.load.vlm", mod)
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


@pytest.fixture
def spy_placement(monkeypatch):
    """Record the placement/lever installers instead of running them, so the
    branch is testable without a streamed model. Each entry is
    ``(target, args, kwargs)`` keyed by installer name."""
    import gmlx.stream.expert_streaming as expert_streaming
    import gmlx.stream.moe_experts as moe_experts

    seen: dict[str, tuple] = {}

    def _spy(name, module=expert_streaming):
        def _fake(target, *args, **kwargs):
            seen[name] = (target, args, kwargs)
        monkeypatch.setattr(module, name, _fake, raising=True)

    _spy("install_expert_streaming")
    _spy("install_unflagged_tables")
    _spy("configure_stream_cpu")
    _spy("install_moe_experts_override")
    _spy("install_moe_expert_mass", moe_experts)
    _spy("install_moe_miss_shed", moe_experts)
    _spy("install_moe_layer_shed", moe_experts)
    return seen


def test_vlm_stream_experts_places_on_the_text_tower(
        fake_load_vlm_model, spy_placement):
    model, _proc, _config = serving.load_serveable_model(
        "/m/llm.gguf", mmproj_path="/m/mmproj.gguf", stream="experts",
        feeder_prefill=True, feeder_decode=False,
    )
    target, _args, kwargs = spy_placement["install_expert_streaming"]
    # The tower, not the VLM wrapper: the vision tower stays on the GPU.
    assert target is model.language_model
    # explicit: `stream: experts` streams a model under the budget too.
    assert kwargs == {"gguf_path": "/m/llm.gguf", "explicit": True,
                      "feeder_prefill": True, "feeder_decode": False,
                      "fast_disk": None}
    assert "configure_stream_cpu" not in spy_placement


@pytest.mark.parametrize("mmproj", [None, "/m/mmproj.gguf"])
def test_a_build_with_no_placement_serves_its_tables_from_the_file(
        fake_load_vlm_model, spy_placement, monkeypatch, mmproj):
    # Tables that alone bring the model under the budget, as residency
    # priced the build: the text tower's own install serves them.
    tokenizer = types.SimpleNamespace(eos_token_id=1)
    raw = _FakeVLMModel()
    monkeypatch.setattr(serving, "load_model",
                        lambda *a, **k: (raw, {}, tokenizer))
    monkeypatch.setattr(serving, "TextOnlyModel",
                        lambda raw, config=None: types.SimpleNamespace(config={}))
    monkeypatch.setattr(serving, "_build_detokenizer", lambda backend: object())
    monkeypatch.setattr(serving, "StoppingCriteria",
                        lambda eos, backend: object())
    model, _proc, _config = serving.load_serveable_model(
        "/m/llm.gguf", mmproj_path=mmproj, feeder_prefill=True,
        feeder_decode=False)
    target, args, kwargs = spy_placement["install_unflagged_tables"]
    assert target is (model.language_model if mmproj else raw)
    assert args == ("/m/llm.gguf",)
    assert kwargs == {"feeder_prefill": True, "feeder_decode": False,
                      "fast_disk": None}
    assert "install_expert_streaming" not in spy_placement


def test_stream_fast_disk_reaches_the_placement(
        fake_load_vlm_model, spy_placement):
    # A model's stream_fast_disk goes to its own feeder, not the process env.
    serving.load_serveable_model(
        "/m/llm.gguf", mmproj_path="/m/mmproj.gguf", stream="experts",
        fast_disk="on")
    assert spy_placement["install_expert_streaming"][2]["fast_disk"] == "on"


def test_vlm_stream_experts_installs_the_moe_levers(
        fake_load_vlm_model, spy_placement):
    model, _proc, _config = serving.load_serveable_model(
        "/m/llm.gguf", mmproj_path="/m/mmproj.gguf", stream="experts",
        moe_experts=4, moe_expert_mass=0.8, moe_miss_shed=0.9,
        moe_layer_shed=0.1,
    )
    for name, value in (("install_moe_experts_override", 4),
                        ("install_moe_expert_mass", 0.8),
                        ("install_moe_miss_shed", 0.9),
                        ("install_moe_layer_shed", 0.1)):
        target, args, _kwargs = spy_placement[name]
        assert target is model.language_model
        assert args == (value,)


def test_vlm_stream_cpu_is_refused(fake_load_vlm_model):
    # stream: cpu moves the device for the process, so it moves the vision
    # tower to the CPU with the text tower.
    with pytest.raises(NotImplementedError, match="stream"):
        serving.load_serveable_model(
            "/m/llm.gguf", mmproj_path="/m/mmproj.gguf", stream="cpu")


def test_vlm_mtp_stream_is_refused(fake_load_vlm_model):
    # The engine loads the MTP drafter after this function, and the drafter
    # gets no placement.
    with pytest.raises(NotImplementedError, match="VLM x MTP"):
        serving.load_serveable_model(
            "/m/llm.gguf", mmproj_path="/m/mmproj.gguf", stream="experts",
            speculative=True)


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

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    original_calls = []
    serveable_calls = []

    def sentinel(model_path, adapter_path=None):
        original_calls.append((model_path, adapter_path))
        return ("ORIG", model_path)

    def spy(model_path, *, mmproj_path=None, hf_source=None,
            speculative=False, draft_gguf_path=None, chat_template=None,
            adapter_gguf=None, stream=None, moe_experts=None,
            moe_expert_mass=None, moe_miss_shed=None, moe_layer_shed=None,
            moe_prestage=None, feeder_prefill=None, feeder_decode=None,
            fast_disk=None):
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


@pytest.mark.parametrize("path,model_type", [
    ("mlx-community/gemma-4-e4b-it-4bit", "gemma4"),
    ("/m/diffusion.gguf", "diffusion_gemma"),
], ids=["mlx-gemma4", "diffusion-gguf"])
def test_every_load_enables_the_system_text_fix(monkeypatch, path, model_type):
    """The stock loader of an MLX-format Gemma-4 and the DiffusionGemma
    branch never reach the VLM loader, which was the only one to enable the
    fix, so their system prompt reached the template as a list."""
    import importlib

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    processor = types.SimpleNamespace()
    loaded = ("MODEL", processor, types.SimpleNamespace(model_type=model_type))
    monkeypatch.setattr(gen, serving._BRIDGE_FLAG, False, raising=False)
    monkeypatch.setattr(gen, "load_model_resources",
                        lambda model_path, adapter_path=None: loaded,
                        raising=False)
    monkeypatch.setattr(serving, "load_serveable_model",
                        lambda *a, **k: loaded)
    monkeypatch.setattr(serving, "_warn_if_template_drops_text",
                        lambda *a: None)
    serving.install_gguf_server_bridge()
    assert gen.load_model_resources(path) is loaded
    assert getattr(processor, "_gmlx_system_as_text", False) is True


def test_install_is_idempotent():
    import importlib

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    serving.install_gguf_server_bridge()
    first = gen.load_model_resources
    serving.install_gguf_server_bridge()
    assert gen.load_model_resources is first


@pytest.mark.parametrize("make_error, raised, match", [
    pytest.param(lambda: MemoryError("the load ran out of memory"),
                 MemoryError, None, id="out of memory"),
    pytest.param(lambda: ModuleNotFoundError(
        "No module named 'gmlx.spec.mtp_load'", name="gmlx.spec.mtp_load"),
                 RuntimeError, "gmlx restart", id="first-party module gone"),
    pytest.param(lambda: ModuleNotFoundError(
        "No module named 'numpy.linalg'", name="numpy.linalg"),
                 ModuleNotFoundError, None, id="third-party module missing"),
])
def test_a_failed_load_ends_the_streaming_caps_lift(monkeypatch, make_error,
                                                    raised, match):
    """The server outlives a load that raises, so the caps lift of
    `serve --stream-experts` ends with that load, as a streamed install
    ends it: no later load or prompt runs on (400, 100000). A first-party
    module missing after startup still names `gmlx restart`, and any
    other error reaches the caller as it was raised."""
    import importlib

    import gmlx.serve.cb_phase as cb_phase

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    writes = []
    monkeypatch.setitem(cb_phase._state, "kq", types.SimpleNamespace(
        set_cb_caps=lambda ops, mb: writes.append((ops, mb)) or cb_phase.COARSE))
    for key, value in (("phase", None), ("caps", None), ("lifted", False),
                       ("pre_lift", None)):
        monkeypatch.setitem(cb_phase._state, key, value)
    cb_phase.note_lift()                # the argv lift, latched at init
    error = make_error()

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(gen, serving._BRIDGE_FLAG, False, raising=False)
    monkeypatch.setattr(gen, "load_model_resources",
                        lambda model_path, adapter_path=None: None,
                        raising=False)
    monkeypatch.setattr(serving, "load_serveable_model", fail)
    serving.install_gguf_server_bridge()
    with pytest.raises(raised, match=match) as info:
        gen.load_model_resources("/m/text.gguf")
    if raised is RuntimeError:
        assert info.value.__cause__ is error
    else:
        assert info.value is error
    assert writes == [cb_phase.FINE]
    assert cb_phase._state["pre_lift"] is None
