"""Serve-side MTP failure handling and drafter-source logging.

A failed speculative build degrades the model to a plain load instead of
taking the server down, and the drafter's real source is logged in place
of the engine's line (which names the stash key - the target GGUF - and
reads as the model drafting for itself)."""

from __future__ import annotations

import inspect
import logging
import os

import pytest

import gmlx.serve.bridge_vlm as bridge


class _Drafter:
    _gmlx_draft_source = None


class _DFlash2Drafter:
    kind_label = "dflash2"

    def __init__(self, source=None):
        self._gmlx_draft_source = source


class _QwenMTPDrafter:
    _gmlx_draft_source = None


_QwenMTPDrafter.__name__ = "QwenMTPDrafter"


def test_degrade_clears_drafter_state(monkeypatch):
    path = "/tmp/fake-target.gguf"
    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", path)
    monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
    bridge._MTP_DRAFTER_STASH[os.path.abspath(path)] = (_Drafter(), "mtp")
    bridge._degrade_failed_mtp(path, "ValueError: no companion")
    assert "MLX_VLM_DRAFT_MODEL" not in os.environ
    assert "MLX_VLM_DRAFT_KIND" not in os.environ
    assert os.path.abspath(path) not in bridge._MTP_DRAFTER_STASH


def _record(msg, args):
    return logging.LogRecord(
        "gmlx._vendor.mlx_vlm.server", logging.INFO, __file__, 1, msg, args, None
    )


def test_filter_drops_only_the_stash_keyed_line():
    f = bridge._DrafterSourceFilter()
    path = "/tmp/stash-target.gguf"
    bridge._MTP_DRAFTER_STASH[os.path.abspath(path)] = (_Drafter(), "mtp")
    try:
        rec = _record("Loading speculative drafter (%s): %s", ("mtp", path))
        assert not f.filter(rec)
        other = _record("Loading speculative drafter (%s): %s",
                        ("mtp", "/tmp/unmanaged.gguf"))
        assert f.filter(other)
        ready = _record("Drafter ready; speculative decoding enabled.", ())
        assert f.filter(ready)
    finally:
        bridge._MTP_DRAFTER_STASH.pop(os.path.abspath(path), None)


def test_filter_target_string_still_in_upstream_source():
    # The filter keys on the engine's literal message prefix; an upstream
    # rename would silently stop the suppression, so pin it to the source.
    generation = pytest.importorskip("gmlx._vendor.mlx_vlm.server.generation")
    src = inspect.getsource(generation)
    assert "Loading speculative drafter" in src


def test_drafter_source_lines(caplog):
    with caplog.at_level(logging.INFO, logger="gmlx.serve.bridge_vlm"):
        bridge._log_drafter_source(
            "/m/target.gguf", _DFlash2Drafter("/m/drafter.gguf"),
            "/m/drafter.gguf")
        bridge._log_drafter_source("/m/target.gguf", _QwenMTPDrafter(), None)
        bridge._log_drafter_source(
            "/m/target.gguf", _DFlash2Drafter("/m/found.gguf"), None)
    lines = [r.getMessage() for r in caplog.records]
    assert any("companion" in ln and "/m/drafter.gguf" in ln
               and "dflash2" in ln for ln in lines)
    assert any("native MTP head of target.gguf" in ln for ln in lines)
    assert any("autodetected companion" in ln and "/m/found.gguf" in ln
               and "dflash2" in ln for ln in lines)


# stale-install import translation: a first-party submodule can only go
# missing when the install changed on disk under the running server
def test_first_party_import_error_gets_the_restart_hint():
    exc = ModuleNotFoundError("No module named 'gmlx.spec.mtp_load'",
                              name="gmlx.spec.mtp_load")
    with pytest.raises(RuntimeError, match="gmlx restart") as e:
        bridge._raise_if_first_party_import(exc)
    assert e.value.__cause__ is exc
    assert "gmlx.spec.mtp_load" in str(e.value)


def test_third_party_import_error_passes_through():
    for name in ("numpy.linalg", "misaki", None):
        exc = ModuleNotFoundError(f"No module named {name!r}", name=name)
        assert bridge._raise_if_first_party_import(exc) is None


@pytest.mark.parametrize("per_layer", [False, True], ids=["tied", "per-layer"])
def test_an_mtp_entry_refuses_an_embedding_adapter_before_any_build(
        tmp_path, monkeypatch, per_layer):
    """The header check runs before the MTP try, so a refused adapter is
    never built, degraded and rebuilt plain."""
    import importlib
    import types

    pytest.importorskip("gguf")
    from lora_gguf import mint_adapter, mint_base

    base = mint_base(tmp_path / "base.gguf", tied=True)
    ad = mint_adapter(tmp_path / "a.gguf", embd=not per_layer, per_layer=per_layer)
    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    monkeypatch.setattr(gen, bridge._BRIDGE_FLAG, False, raising=False)
    monkeypatch.setattr(gen, "load_model_resources", gen.load_model_resources)
    monkeypatch.setattr(bridge, "load_serveable_model",
                        lambda *a, **k: pytest.fail("built for a refused adapter"))
    monkeypatch.setattr(bridge, "_degrade_failed_mtp",
                        lambda *a, **k: pytest.fail("degraded for a refused adapter"))
    monkeypatch.setitem(bridge._GGUF_MTP_REGISTRY, os.path.abspath(base),
                        {"draft_gguf_path": None})
    monkeypatch.setattr(bridge, "_current_build_spec",
                        types.SimpleNamespace(effective_adapters=lambda: (ad,)))
    bridge.install_gguf_server_bridge()
    with pytest.raises(ValueError, match="a.gguf: "):
        gen.load_model_resources(base, None)
