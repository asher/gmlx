#!/usr/bin/env python3
"""MTP (speculative) serving bridge: registration + resolution, the speculative
branch of ``load_serveable_model`` (drafter stash + target/processor), the
drafter-load injection patch, and bridge routing. CPU-only - a fake
``load_mtp_model`` and sentinel loaders, so no GPU, no GGUF files, no real load."""
from __future__ import annotations

import os
import sys
import threading
import types

import pytest

pytest.importorskip("mlx_vlm")

from gmlx import server_bridge_vlm as serving  # noqa: E402

_ENV_KEYS = (
    "MLX_VLM_GGUF_SPECULATIVE",
    "MLX_VLM_GGUF_DRAFT",
    "MLX_VLM_DRAFT_MODEL",
    "MLX_VLM_DRAFT_KIND",
    "GMLX_DRAFT_BLOCK_SIZE",
)


def _resolved_stub(**overrides):
    """Spec stub cut from a real config.ResolvedModel (not a bare
    SimpleNamespace): dataclass field renames/removals fail here instead of
    passing silently against a hand-rolled attribute set."""
    import dataclasses

    from gmlx.config import ResolvedModel

    base = ResolvedModel(
        id="stub", path="/m/stub.gguf", sampling={}, load={}, cache={},
        system=None, speculative=False, mmproj=None, draft_gguf=None,
        pin=False, ttl_s=None)
    return dataclasses.replace(base, **overrides)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Each test sees empty registries/stash, no association env vars, and no
    request-bound active spec or in-flight build spec."""
    serving._GGUF_MTP_REGISTRY.clear()
    serving._MTP_DRAFTER_STASH.clear()
    serving._GGUF_VLM_REGISTRY.clear()
    serving._active_spec.set(None)
    serving.set_build_spec(None)
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield
    serving._GGUF_MTP_REGISTRY.clear()
    serving._MTP_DRAFTER_STASH.clear()
    serving._active_spec.set(None)
    serving.set_build_spec(None)


# registration + resolution: registry + env fallback
def test_register_native_head_keyed_by_abspath():
    serving.register_gguf_mtp("/m/target.gguf")  # native head: no companion
    assert serving._resolve_mtp_spec("/m/target.gguf") == {"draft_gguf_path": None}
    # A relative path resolving to the same file hits the same entry.
    serving.register_gguf_mtp("./t.gguf")
    assert serving._resolve_mtp_spec(os.path.abspath("./t.gguf")) == {
        "draft_gguf_path": None
    }


def test_register_assistant_keeps_draft_abspath():
    serving.register_gguf_mtp("/m/target.gguf", draft_gguf_path="./draft.gguf")
    assert serving._resolve_mtp_spec("/m/target.gguf") == {
        "draft_gguf_path": os.path.abspath("./draft.gguf")
    }


def test_unregistered_gguf_resolves_none():
    assert serving._resolve_mtp_spec("/m/plain.gguf") is None


def test_env_fallback_native_and_assistant(monkeypatch):
    monkeypatch.setenv("MLX_VLM_GGUF_SPECULATIVE", "1")
    assert serving._resolve_mtp_spec("/some/unreg.gguf") == {"draft_gguf_path": None}
    monkeypatch.setenv("MLX_VLM_GGUF_DRAFT", "/env/draft.gguf")
    assert serving._resolve_mtp_spec("/some/unreg.gguf") == {
        "draft_gguf_path": "/env/draft.gguf"
    }


def test_env_falsey_does_not_enable(monkeypatch):
    monkeypatch.setenv("MLX_VLM_GGUF_SPECULATIVE", "0")
    assert serving._resolve_mtp_spec("/some/unreg.gguf") is None


def test_registry_wins_over_env(monkeypatch):
    monkeypatch.setenv("MLX_VLM_GGUF_SPECULATIVE", "1")
    monkeypatch.setenv("MLX_VLM_GGUF_DRAFT", "/env/draft.gguf")
    serving.register_gguf_mtp("/m/target.gguf")  # native, no draft
    assert serving._resolve_mtp_spec("/m/target.gguf") == {"draft_gguf_path": None}


def test_serve_speculative_flag_routes_speculative(installed_bridge, tmp_path):
    """The `serve model.gguf --speculative` promise (server.py --speculative
    help): the flag flows through the REAL serve parser + single-model config +
    registration, _resolve_mtp_spec returns a native-head spec, and the next
    load through the bridge routes speculative."""
    import argparse

    from gmlx import server

    patched, _, serveable_calls = installed_bridge
    gguf = tmp_path / "target.gguf"
    gguf.write_bytes(b"GGUF")
    ap = argparse.ArgumentParser()
    server._add_serve_args(ap)
    a = ap.parse_args([str(gguf), "--speculative"])
    try:
        serving.register_resolved_models(server._single_model_cfg(a))
        assert serving._resolve_mtp_spec(str(gguf)) == {"draft_gguf_path": None}
        serving.set_build_spec(_resolved_stub(path=str(gguf), speculative=True))
        patched(str(gguf))
        assert serveable_calls == [
            (str(gguf), None, None, True, None, None, None)]
    finally:
        serving.clear_resolved_models()


# load_serveable_model speculative branch (fake load_mtp_model)
class _FakeTarget:
    """Stand-in for the MTPTextTarget the loader returns: only ``.config`` (the
    3rd return element) and ``.language_model`` matter to the server."""

    def __init__(self):
        self.config = {"model_type": "qwen3_5_moe"}
        self.language_model = object()


@pytest.fixture
def fake_load_mtp_model(monkeypatch):
    """Stub ``gmlx.mtp_load.load_mtp_model`` (imported lazily inside the
    speculative branch). Records its call; returns ``(model, drafter, config,
    tokenizer)`` in the loader's order."""
    calls = []
    drafter = object()

    def _fake(gguf_path, *, draft_gguf_path=None, chat_template=None, verbose=True):
        calls.append(
            {"gguf": gguf_path, "draft_gguf_path": draft_gguf_path,
             "chat_template": chat_template, "verbose": verbose}
        )
        tok = types.SimpleNamespace(eos_token_ids={1})
        return _FakeTarget(), drafter, {"model_type": "qwen3_5_moe"}, tok

    mtp_load = types.ModuleType("gmlx.mtp_load")
    mtp_load.load_mtp_model = _fake
    monkeypatch.setitem(sys.modules, "gmlx.mtp_load", mtp_load)
    # The processor builder reaches StoppingCriteria / detokenizer - keep CPU-cheap.
    monkeypatch.setattr(serving, "_build_detokenizer", lambda backend: object())
    monkeypatch.setattr(serving, "StoppingCriteria",
                        lambda eos, backend: object(), raising=True)
    return calls, drafter


def test_speculative_branch_returns_target_and_stashes_drafter(fake_load_mtp_model):
    calls, drafter = fake_load_mtp_model
    model, proc, config = serving.load_serveable_model(
        "/m/target.gguf", speculative=True
    )
    assert isinstance(model, _FakeTarget)
    # config IS the model's config (one object, like the text path) and is now
    # readable BOTH ways: attribute (the server's request preprocessor reads
    # model.config.model_type) and dict (the target's get_input_embeddings reads
    # config.get(...)).
    assert config is model.config
    assert config.model_type == "qwen3_5_moe"          # attribute access
    assert config.get("model_type") == "qwen3_5_moe"   # dict access
    assert isinstance(proc, serving._GgufServerProcessor)
    # Drafter stashed by absolute path for the drafter-load patch to pick up.
    assert serving._MTP_DRAFTER_STASH[os.path.abspath("/m/target.gguf")] == (
        drafter, "mtp"
    )
    # Loaded quietly, no companion for the native-head shape.
    assert calls == [{"gguf": "/m/target.gguf", "draft_gguf_path": None,
                      "chat_template": None, "verbose": False}]


def test_speculative_branch_threads_draft_gguf(fake_load_mtp_model):
    calls, _ = fake_load_mtp_model
    serving.load_serveable_model(
        "/m/target.gguf", speculative=True, draft_gguf_path="/m/draft.gguf"
    )
    assert calls[0]["draft_gguf_path"] == "/m/draft.gguf"


def test_speculative_branch_threads_chat_template(fake_load_mtp_model):
    calls, _ = fake_load_mtp_model
    serving.load_serveable_model(
        "/m/target.gguf", speculative=True, chat_template="/tmpl/x.jinja"
    )
    assert calls[0]["chat_template"] == "/tmpl/x.jinja"


# load_serveable_model VLM x MTP branch (fake load_vlm_mtp_model)
@pytest.fixture
def fake_load_vlm_mtp_model(monkeypatch):
    """Stub ``gmlx.mtp_load.load_vlm_mtp_model`` (lazy import in the VLM x MTP
    branch). Records its call; returns ``(model, drafter, config, tokenizer,
    processor)`` in the loader's order. The VLM model's config is attribute-readable
    already (unlike the text MTP wrapper's dict), so no _AttrDict promotion."""
    calls = []
    drafter = object()

    def _fake(gguf_path, mmproj_path, *, hf_source=None, draft_gguf_path=None,
              chat_template=None, verbose=True):
        calls.append(
            {"gguf": gguf_path, "mmproj": mmproj_path, "hf_source": hf_source,
             "draft_gguf_path": draft_gguf_path, "chat_template": chat_template,
             "verbose": verbose}
        )
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="qwen3_5"),
            language_model=object())
        tok = types.SimpleNamespace(eos_token_ids={1})
        processor = object()  # the loader hands back an engine-ready VLM processor
        return model, drafter, {"model_type": "qwen3_5"}, tok, processor

    mtp_load = types.ModuleType("gmlx.mtp_load")
    mtp_load.load_vlm_mtp_model = _fake
    monkeypatch.setitem(sys.modules, "gmlx.mtp_load", mtp_load)
    return calls, drafter


def test_vlm_mtp_branch_returns_vlm_and_stashes_drafter(fake_load_vlm_mtp_model):
    calls, drafter = fake_load_vlm_mtp_model
    model, proc, config = serving.load_serveable_model(
        "/m/llm.gguf", mmproj_path="/m/mmproj.gguf", speculative=True,
        draft_gguf_path="/m/draft.gguf",
    )
    # config IS the VLM model's own dataclass config (attribute-readable), not the
    # synthesized text processor wrapper - the engine reads model.config.model_type.
    assert config is model.config
    assert config.model_type == "qwen3_5"
    assert proc is not None  # the VLM processor passes through (not _make_text_processor)
    # Drafter stashed by absolute path for the drafter-load patch to pick up.
    assert serving._MTP_DRAFTER_STASH[os.path.abspath("/m/llm.gguf")] == (
        drafter, "mtp")
    assert calls[0]["mmproj"] == "/m/mmproj.gguf"
    assert calls[0]["draft_gguf_path"] == "/m/draft.gguf"
    assert calls[0]["verbose"] is False


def test_vlm_mtp_branch_native_head_no_companion(fake_load_vlm_mtp_model):
    # native-head shape (qwen): no --draft-gguf, drafter from the LLM GGUF's nextn
    calls, _ = fake_load_vlm_mtp_model
    serving.load_serveable_model(
        "/m/llm.gguf", mmproj_path="/m/mmproj.gguf", speculative=True)
    assert calls[0]["draft_gguf_path"] is None


def test_text_branch_does_not_touch_mtp_loader(fake_load_mtp_model, monkeypatch):
    calls, _ = fake_load_mtp_model
    tokenizer = types.SimpleNamespace(eos_token_id=1)
    monkeypatch.setattr(serving, "load_model",
                        lambda *a, **k: (_FakeTarget(), {}, tokenizer), raising=True)
    monkeypatch.setattr(serving, "TextOnlyModel",
                        lambda raw, config=None: types.SimpleNamespace(config={}),
                        raising=True)
    serving.load_serveable_model("/m/plain.gguf")  # not speculative
    assert calls == []  # the MTP loader was never invoked for a plain text GGUF


# drafter-load injection patch
@pytest.fixture
def installed_drafter_patch(monkeypatch):
    """Install the drafter-load patch over a sentinel original; returns
    ``(patched_load_drafter, original_calls)``."""
    import importlib

    drafters = importlib.import_module("mlx_vlm.speculative.drafters")
    original_calls = []

    def sentinel(path_or_repo, kind=None, **kwargs):
        original_calls.append((path_or_repo, kind))
        return ("ORIG_DRAFTER", path_or_repo)

    monkeypatch.setattr(drafters, serving._DRAFTER_PATCH_FLAG, False, raising=False)
    monkeypatch.setattr(drafters, "load_drafter", sentinel, raising=False)
    serving._install_drafter_injection()
    return drafters.load_drafter, original_calls


def test_stashed_path_returns_in_memory_drafter(installed_drafter_patch):
    patched, original_calls = installed_drafter_patch
    drafter = object()
    serving._MTP_DRAFTER_STASH[os.path.abspath("/m/target.gguf")] = (drafter, "mtp")
    assert patched("/m/target.gguf", kind="mtp") == (drafter, "mtp")
    assert original_calls == []  # never hit the disk loader
    # Popped on consume: the engine owns the drafter now; a lingering stash
    # entry would pin its weights across evictions/reloads.
    assert serving._MTP_DRAFTER_STASH == {}


def test_unstashed_path_falls_through(installed_drafter_patch):
    patched, original_calls = installed_drafter_patch
    assert patched("/some/other/drafter", kind="mtp")[0] == "ORIG_DRAFTER"
    assert original_calls == [("/some/other/drafter", "mtp")]


def test_draft_block_size_override_sets_config(installed_drafter_patch, monkeypatch):
    patched, _ = installed_drafter_patch
    drafter = types.SimpleNamespace(config=types.SimpleNamespace(block_size=1))
    serving._MTP_DRAFTER_STASH[os.path.abspath("/m/target.gguf")] = (drafter, "mtp")
    monkeypatch.setenv("GMLX_DRAFT_BLOCK_SIZE", "3")
    result = patched("/m/target.gguf", kind="mtp")
    assert result[0].config.block_size == 3       # serve --draft-block-size N


def test_draft_block_size_override_noop_without_env(installed_drafter_patch):
    patched, _ = installed_drafter_patch
    drafter = types.SimpleNamespace(config=types.SimpleNamespace(block_size=7))
    serving._MTP_DRAFTER_STASH[os.path.abspath("/m/t.gguf")] = (drafter, "mtp")
    result = patched("/m/t.gguf", kind="mtp")
    assert result[0].config.block_size == 7       # untouched (no env)


def test_drafter_patch_is_idempotent():
    import importlib

    drafters = importlib.import_module("mlx_vlm.speculative.drafters")
    serving._install_drafter_injection()
    first = drafters.load_drafter
    serving._install_drafter_injection()
    assert drafters.load_drafter is first


# bridge routing: text vs VLM vs MTP vs non-gguf
@pytest.fixture
def installed_bridge(monkeypatch):
    """Install the bridge over a sentinel original loader and a spy
    ``load_serveable_model``; returns ``(patched_loader, original_calls,
    serveable_calls)``. The drafter patch installs against the real module
    (harmless - only the stash drives it)."""
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
        serveable_calls.append(
            (model_path, mmproj_path, hf_source, speculative, draft_gguf_path,
             chat_template, adapter_gguf)
        )
        return ("SERVEABLE", model_path)

    monkeypatch.setattr(gen, serving._BRIDGE_FLAG, False, raising=False)
    monkeypatch.setattr(gen, "load_model_resources", sentinel, raising=False)
    monkeypatch.setattr(serving, "load_serveable_model", spy)
    serving.install_gguf_server_bridge()
    return gen.load_model_resources, original_calls, serveable_calls


def test_non_gguf_falls_through_to_original(installed_bridge):
    patched, original_calls, serveable_calls = installed_bridge
    assert patched("mlx-community/Qwen3-4B")[0] == "ORIG"
    assert original_calls == [("mlx-community/Qwen3-4B", None)]
    assert serveable_calls == []


def test_plain_text_gguf_routes_non_speculative(installed_bridge):
    patched, _, serveable_calls = installed_bridge
    patched("/m/text.gguf")
    assert serveable_calls == [("/m/text.gguf", None, None, False, None, None, None)]


def test_mtp_gguf_routes_speculative_and_sets_env(installed_bridge, monkeypatch):
    patched, _, serveable_calls = installed_bridge
    serving.register_gguf_mtp("/m/target.gguf", draft_gguf_path="/m/draft.gguf")
    patched("/m/target.gguf")
    assert serveable_calls == [
        ("/m/target.gguf", None, None, True, "/m/draft.gguf", None, None)
    ]
    # The stock drafter-load block fires only when these are set.
    assert os.environ["MLX_VLM_DRAFT_MODEL"] == "/m/target.gguf"
    assert os.environ["MLX_VLM_DRAFT_KIND"] == "mtp"


def test_gguf_adapter_path_raises_not_silently_dropped(installed_bridge):
    """A GGUF base + an adapter_path is the unbuilt LoRA inference seam: the
    bridge must surface it, not silently drop the adapter and serve the bare base."""
    patched, _, serveable_calls = installed_bridge
    with pytest.raises(NotImplementedError, match="adapter inference"):
        patched("/m/text.gguf", "/m/adapter")
    assert serveable_calls == []   # never loaded the bare base under a dropped adapter


def test_non_gguf_adapter_path_flows_to_original(installed_bridge):
    """A non-GGUF id keeps mlx-vlm's adapter handling - adapter_path reaches the
    stock loader untouched."""
    patched, original_calls, _ = installed_bridge
    patched("mlx-community/Qwen3-4B", "/m/adapter")
    assert original_calls == [("mlx-community/Qwen3-4B", "/m/adapter")]


def test_vlm_mtp_when_both_registered(installed_bridge):
    patched, _, serveable_calls = installed_bridge
    serving.register_gguf_vlm("/m/llm.gguf", "/m/mmproj.gguf")
    serving.register_gguf_mtp("/m/llm.gguf")
    patched("/m/llm.gguf")
    # Both registered -> VLM x MTP: the call carries mmproj AND speculative, and the
    # drafter-load block is armed (env keyed to this GGUF) like the text MTP branch.
    assert serveable_calls == [
        ("/m/llm.gguf", "/m/mmproj.gguf", None, True, None, None, None)
    ]
    assert os.environ.get("MLX_VLM_DRAFT_MODEL") == "/m/llm.gguf"


# per-id speculative gate: the env window overrides the path-keyed registry
# (one GGUF backing both a speculative id and a plain "lossless oracle" id).
# The model builds in the engine's WORKER thread, so the signal must be
# process-global (env), NOT a request-thread ContextVar.
def test_explicit_speculative_off_overrides_registry(monkeypatch):
    """``MLX_VLM_GGUF_SPECULATIVE=0`` (set per build by the residency env window)
    forces a non-speculative resolution even when the SAME GGUF is registered for
    MTP by a sibling id - the lossless-oracle case (one GGUF, spec-on + spec-off)."""
    serving.register_gguf_mtp("/m/shared.gguf", draft_gguf_path="/m/d.gguf")
    assert serving._resolve_mtp_spec("/m/shared.gguf") == {
        "draft_gguf_path": os.path.abspath("/m/d.gguf")}     # registry, no env
    monkeypatch.setenv("MLX_VLM_GGUF_SPECULATIVE", "0")
    assert serving._resolve_mtp_spec("/m/shared.gguf") is None  # env "0" wins


def test_bridge_speculative_off_routes_plain_and_clears_drafter_env(
        installed_bridge, monkeypatch):
    """With the env window at "0", a GGUF registered MTP by a sibling id routes as
    plain text and clears stale drafter env (left by a prior speculative build), so
    the engine's load_drafter block can't fire for the oracle."""
    patched, _, serveable_calls = installed_bridge
    serving.register_gguf_mtp("/m/shared.gguf")
    monkeypatch.setenv("MLX_VLM_GGUF_SPECULATIVE", "0")
    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", "/m/shared.gguf")  # stale, prior MTP build
    monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
    patched("/m/shared.gguf")
    assert serveable_calls == [
        ("/m/shared.gguf", None, None, False, None, None, None)]
    assert "MLX_VLM_DRAFT_MODEL" not in os.environ
    assert "MLX_VLM_DRAFT_KIND" not in os.environ


def test_stale_drafter_env_never_reaches_a_non_gguf_load(monkeypatch):
    """A speculative GGUF load sets MLX_VLM_DRAFT_MODEL/KIND process-wide; a later
    NON-GGUF (plain HF) load must not see them - the stock loader's drafter block
    would otherwise hand the K-quant MTP drafter to an unrelated model. The seam
    pops both vars before branch dispatch, so the stock loader runs clean."""
    import importlib

    gen = importlib.import_module("mlx_vlm.server.generation")
    env_at_original = []

    def sentinel(model_path, adapter_path=None):
        env_at_original.append(
            {k: os.environ.get(k)
             for k in ("MLX_VLM_DRAFT_MODEL", "MLX_VLM_DRAFT_KIND")})
        return ("ORIG", model_path)

    monkeypatch.setattr(gen, serving._BRIDGE_FLAG, False, raising=False)
    monkeypatch.setattr(gen, "load_model_resources", sentinel, raising=False)
    monkeypatch.setattr(serving, "load_serveable_model",
                        lambda *a, **k: ("SERVEABLE", a, k))
    serving.install_gguf_server_bridge()
    patched = gen.load_model_resources

    serving.register_gguf_mtp("/m/target.gguf")
    patched("/m/target.gguf")                       # speculative: sets drafter env
    assert os.environ["MLX_VLM_DRAFT_MODEL"] == "/m/target.gguf"
    patched("mlx-community/Qwen3-4B")               # non-GGUF: must run clean
    assert env_at_original == [
        {"MLX_VLM_DRAFT_MODEL": None, "MLX_VLM_DRAFT_KIND": None}]
    assert "MLX_VLM_DRAFT_MODEL" not in os.environ
    assert "MLX_VLM_DRAFT_KIND" not in os.environ


def test_stale_drafter_env_cleared_on_vlm_branch(installed_bridge, monkeypatch):
    """The VLM branch clears stale drafter env too - every branch of the load
    seam starts from a clean drafter state."""
    patched, _, serveable_calls = installed_bridge
    serving.register_gguf_vlm("/m/llm.gguf", "/m/mmproj.gguf")
    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", "/m/other.gguf")  # stale
    monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
    patched("/m/llm.gguf")
    assert serveable_calls == [
        ("/m/llm.gguf", "/m/mmproj.gguf", None, False, None, None, None)]
    assert "MLX_VLM_DRAFT_MODEL" not in os.environ
    assert "MLX_VLM_DRAFT_KIND" not in os.environ


def test_drop_mtp_stash_is_path_keyed_and_tolerant():
    """The residency teardown hook: drops exactly the evicted model's stash entry;
    a path with no stash is a no-op."""
    drafter = object()
    serving._MTP_DRAFTER_STASH[os.path.abspath("/m/a.gguf")] = (drafter, "mtp")
    serving._MTP_DRAFTER_STASH[os.path.abspath("/m/b.gguf")] = (drafter, "mtp")
    serving.drop_mtp_stash("/m/a.gguf")
    serving.drop_mtp_stash("/m/never-stashed.gguf")
    assert list(serving._MTP_DRAFTER_STASH) == [os.path.abspath("/m/b.gguf")]


def test_speculative_signal_crosses_thread_boundary_via_env(
        installed_bridge, monkeypatch):
    """The regression that the prior same-thread test missed: the model loads in
    the engine's generation WORKER thread (ResponseGenerator._initialize_model),
    where a request-thread ContextVar is invisible. So the decision must ride the
    process-global env window, not ``_active_spec``. Set a *misleading* speculative
    spec on the main-thread ContextVar and register the path MTP, but set the env
    window to "0"; calling the bridge from a SEPARATE thread (as the engine does)
    must route PLAIN - following env - and see ``get_active_spec()`` is None there."""
    patched, _, serveable_calls = installed_bridge
    serving.register_gguf_mtp("/m/shared.gguf")
    serving._active_spec.set(                                 # main-thread ContextVar...
        types.SimpleNamespace(speculative=True, chat_template=None))
    monkeypatch.setenv("MLX_VLM_GGUF_SPECULATIVE", "0")       # ...env window says non-spec
    seen = {}

    def worker():
        seen["active_spec"] = serving.get_active_spec()      # invisible across the thread
        patched("/m/shared.gguf")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["active_spec"] is None                       # ContextVar did NOT cross
    assert serveable_calls == [
        ("/m/shared.gguf", None, None, False, None, None, None)]


# per-profile chat_template: the SAME worker-thread seam. The profile's
# chat-template override (config path) must reach the load bridge, which runs
# in the engine's generation WORKER thread - so it rides the process-global build
# spec (set by residency under its build lock), NOT the request-thread ContextVar.
def test_chat_template_crosses_into_worker_thread(installed_bridge):
    """RED before the fix: the bridge reads ``get_active_spec().chat_template``,
    but that ContextVar is invisible in the engine's load worker thread, so a
    per-profile chat_template is silently dropped. After the fix the bridge reads
    the build spec - set by residency under its build lock - which DOES cross the
    boundary. Set a *misleading* template on the main-thread ContextVar and the real
    one on the build spec; the worker must thread the build-spec template through."""
    patched, _, serveable_calls = installed_bridge
    serving._active_spec.set(                                 # main-thread ContextVar...
        types.SimpleNamespace(speculative=False, chat_template="WRONG-FROM-CONTEXTVAR"))
    serving.set_build_spec(                                   # ...build spec is the truth
        types.SimpleNamespace(speculative=False, chat_template="CUSTOM-TEMPLATE"))
    seen = {}

    def worker():
        seen["active_spec"] = serving.get_active_spec()      # None across the thread
        patched("/m/text.gguf")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["active_spec"] is None                       # ContextVar did NOT cross
    assert serveable_calls == [
        ("/m/text.gguf", None, None, False, None, "CUSTOM-TEMPLATE", None)]


# GGUF LoRA adapter: rides the SAME build-spec channel as chat_template,
# and is applied to the loaded base in load_serveable_model (text path only).
def test_adapter_threads_into_serveable_via_build_spec(installed_bridge):
    """A config `adapter:` rides the build-spec channel (worker-thread-safe, like
    chat_template) and reaches the text loader as the 7th serveable arg."""
    patched, _, serveable_calls = installed_bridge
    serving.set_build_spec(
        types.SimpleNamespace(chat_template=None, adapter="/m/ad.lora.gguf"))
    patched("/m/text.gguf")
    assert serveable_calls == [
        ("/m/text.gguf", None, None, False, None, None, "/m/ad.lora.gguf")]


def test_moe_expert_mass_threads_into_serveable_via_build_spec(
        installed_bridge, monkeypatch):
    """A config `moe_expert_mass:` rides the build-spec channel with the stream
    placement and reaches the text loader (where the fan-out filter is
    installed)."""
    patched, _, _ = installed_bridge
    seen = {}

    def recorder(model_path, **kw):
        seen.update(kw)
        return ("SERVEABLE", model_path)

    monkeypatch.setattr(serving, "load_serveable_model", recorder)
    serving.set_build_spec(types.SimpleNamespace(
        chat_template=None, adapter=None, stream="experts",
        moe_expert_mass=0.9))
    patched("/m/text.gguf")
    assert seen["stream"] == "experts"
    assert seen["moe_expert_mass"] == 0.9


def test_adapter_rides_build_spec_not_active_spec_across_thread(installed_bridge):
    """Like chat_template, the adapter must cross into the engine's load WORKER
    thread via the build spec, not the request-thread ContextVar. A misleading
    adapter on ``_active_spec`` is ignored; the build-spec adapter wins."""
    patched, _, serveable_calls = installed_bridge
    serving._active_spec.set(
        types.SimpleNamespace(speculative=False, chat_template=None,
                              adapter="WRONG-FROM-CONTEXTVAR"))
    serving.set_build_spec(
        types.SimpleNamespace(chat_template=None, adapter="/m/right.lora.gguf"))
    seen = {}

    def worker():
        seen["active_spec"] = serving.get_active_spec()
        patched("/m/text.gguf")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["active_spec"] is None
    assert serveable_calls == [
        ("/m/text.gguf", None, None, False, None, None, "/m/right.lora.gguf")]


def test_text_branch_applies_adapter(fake_load_mtp_model, monkeypatch):
    """The text path hands the *raw* model + config to ``_apply_gguf_adapter`` (the
    leaves-wrap seam) when an adapter is given."""
    raw = _FakeTarget()
    tokenizer = types.SimpleNamespace(eos_token_id=1)
    monkeypatch.setattr(serving, "load_model",
                        lambda *a, **k: (raw, {"model_type": "qwen3"}, tokenizer),
                        raising=True)
    monkeypatch.setattr(serving, "TextOnlyModel",
                        lambda r, config=None: types.SimpleNamespace(config={}),
                        raising=True)
    applied = []
    monkeypatch.setattr(
        serving, "_apply_gguf_adapter",
        lambda r, cfg, ad, base_gguf_path=None:
        applied.append((r, cfg, ad, base_gguf_path)))
    serving.load_serveable_model("/m/text.gguf", adapter_gguf="/m/ad.lora.gguf")
    # base_gguf_path rides along so the adapter's arch is gated against the
    # base's before install (clean mismatch error, not missing-targets).
    assert applied == [(raw, {"model_type": "qwen3"}, "/m/ad.lora.gguf",
                        "/m/text.gguf")]


def test_vlm_plus_adapter_raises_not_silently_dropped():
    with pytest.raises(NotImplementedError, match="VLM base"):
        serving.load_serveable_model("/m/llm.gguf", mmproj_path="/m/mm.gguf",
                                     adapter_gguf="/m/ad.lora.gguf")


def test_speculative_plus_adapter_raises_not_silently_dropped():
    with pytest.raises(NotImplementedError, match="speculative/MTP base"):
        serving.load_serveable_model("/m/t.gguf", speculative=True,
                                     adapter_gguf="/m/ad.lora.gguf")


def test_prompt_step_caps_mtp_hidden_capture():
    """Chunked MTP prefill must retain only the trailing hidden_capture_limit
    positions for window-limited drafters, not pin the whole prompt's hidden."""
    import mlx.core as mx
    from mlx_vlm.generate.ar import PromptProcessingBatch

    from gmlx.spec_engine import install_full_prompt_mtp_prefill

    install_full_prompt_mtp_prefill()

    D, LIMIT, TOTAL, CHUNK = 8, 128, 1024, 256

    class _Out:
        def __init__(self, n):
            self.hidden_states = [mx.zeros((1, n, D))]

    class _Drafter:
        hidden_capture_limit = LIMIT

        def prefill_from_target_hidden(self, *a, **k):
            pass

    class _Batch(PromptProcessingBatch):
        def __init__(self):
            self.draft_kind = "mtp"
            self.draft_model = _Drafter()
            self.model = lambda ids, **kw: _Out(ids.shape[1])
            self.prompt_cache = []
            self.prefill_step_size = CHUNK
            self._mtp_full_input_ids = mx.zeros((1, TOTAL), dtype=mx.int32)
            self._input_ids = mx.zeros((1, TOTAL), dtype=mx.int32)
            self._inputs_embeds = mx.zeros((1, TOTAL, D))
            self._mtp_chunk_hiddens = []
            self._processed_prompt_columns = 0
            self._prompt_kwargs = {}
            self._prompt_length_aware_keys = []

        def needs_processing(self):
            return self._inputs_embeds.shape[1] > 1

        def _next_apc_checkpoint_column(self):
            return None

        def _prompt_kwargs_for_step(self, n):
            return {}

        def _store_apc_exact_checkpoints(self):
            pass

    b = _Batch()
    while b.needs_processing():
        assert b.prompt_step() > 0
    total = sum(int(h.shape[1]) for h in b._mtp_chunk_hiddens)
    assert total == LIMIT


def test_load_vlm_mtp_model_applies_chat_template(monkeypatch):
    """The chat_template param must land on the tokenizer AND the processor
    (which snapshots the tokenizer's template at construction)."""
    import gmlx.mtp_load as mtp_load
    import gmlx.vlm as vlm_mod

    class _Tok:
        chat_template = "{{ gguf }}"
        eos_token = "</s>"
        eos_token_id = 2
        bos_token_id = 1

        def get_vocab(self):
            return {"</s>": 2}

    class _Proc:
        chat_template = "{{ gguf }}"

    class _LM:
        speculative_logits_from_hidden = object()
        rollback_speculative_cache = object()

    class _Model:
        language_model = _LM()

    tok = _Tok()
    monkeypatch.setattr(
        vlm_mod, "load_vlm_model",
        lambda *a, **k: (_Model(), {"model_type": "gemma4"}, _Proc(), tok),
    )
    monkeypatch.setattr(
        mtp_load, "_load_gemma4_assistant_drafter", lambda *a, **k: "drafter")
    monkeypatch.setattr(mtp_load, "_wire_big_model", lambda m: None)

    model, drafter, config, tokenizer, processor = mtp_load.load_vlm_mtp_model(
        "/m/t.gguf", "/m/mmproj.gguf",
        draft_gguf_path="/m/draft.gguf",
        chat_template="{{ override }}",
    )
    assert tok.chat_template == "{{ override }}"
    assert processor.chat_template == "{{ override }}"
