"""Serve-side MTP failure handling and drafter-source logging.

A failed speculative build degrades the model to a plain load instead of
taking the server down, and the drafter's real source is logged in place
of the engine's line (which names the stash key - the target GGUF - and
reads as the model drafting for itself)."""

from __future__ import annotations

import logging
import os

import gmlx.serve.bridge_vlm as bridge


class _Drafter:
    pass


class _DFlash2Drafter:
    kind_label = "dflash2"


class _QwenMTPDrafter:
    pass


_QwenMTPDrafter.__name__ = "QwenMTPDrafter"


def test_degrade_clears_drafter_state(monkeypatch):
    path = "/tmp/fake-target.gguf"
    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", path)
    monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
    bridge._MTP_DRAFTER_STASH[os.path.abspath(path)] = (_Drafter(), "mtp")
    bridge._degrade_failed_mtp(path, ValueError("no companion"))
    assert "MLX_VLM_DRAFT_MODEL" not in os.environ
    assert "MLX_VLM_DRAFT_KIND" not in os.environ
    assert os.path.abspath(path) not in bridge._MTP_DRAFTER_STASH


def _record(msg, args):
    return logging.LogRecord(
        "mlx_vlm.server", logging.INFO, __file__, 1, msg, args, None
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


def test_drafter_source_lines(caplog):
    with caplog.at_level(logging.INFO, logger="gmlx.serve.bridge_vlm"):
        bridge._log_drafter_source(
            "/m/target.gguf", _DFlash2Drafter(), "/m/drafter.gguf")
        bridge._log_drafter_source("/m/target.gguf", _QwenMTPDrafter(), None)
        bridge._log_drafter_source("/m/target.gguf", _DFlash2Drafter(), None)
    lines = [r.getMessage() for r in caplog.records]
    assert any("companion" in ln and "drafter.gguf" in ln and "dflash2" in ln
               for ln in lines)
    assert any("native MTP head of target.gguf" in ln for ln in lines)
    assert any("autodetected dflash2 companion" in ln for ln in lines)
