"""Decode concurrency control: env resolution + serve-path injection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_vlm")

import gmlx.serve.decode_batch as db  # noqa: E402


def test_default(monkeypatch):
    monkeypatch.delenv("GMLX_DECODE_BATCH", raising=False)
    assert db.decode_batch() == db.DEFAULT_DECODE_BATCH


def test_env_override(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_BATCH", "4")
    assert db.decode_batch() == 4


def test_zero_restores_upstream(monkeypatch):
    from mlx_vlm.generate.ar import DEFAULT_COMPLETION_BATCH_SIZE

    monkeypatch.setenv("GMLX_DECODE_BATCH", "0")
    assert db.decode_batch() == int(DEFAULT_COMPLETION_BATCH_SIZE)


def test_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_BATCH", "lots")
    assert db.decode_batch() == db.DEFAULT_DECODE_BATCH
    monkeypatch.setenv("GMLX_DECODE_BATCH", "-3")
    assert db.decode_batch() == db.DEFAULT_DECODE_BATCH


def test_stash_wrapper_injects(monkeypatch):
    from mlx_vlm.generate import ar

    import gmlx.spec.engine as spec_engine

    seen = {}

    class _BG:
        def __init__(self, model, processor, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(ar, "BatchGenerator", _BG)
    spec_engine._install_apc_manager_stash()

    monkeypatch.setenv("GMLX_DECODE_BATCH", "5")
    ar.BatchGenerator(SimpleNamespace(), None)
    assert seen["completion_batch_size"] == 5

    seen.clear()
    ar.BatchGenerator(SimpleNamespace(), None, completion_batch_size=3)
    assert seen["completion_batch_size"] == 3


def test_stash_wrapper_clamps_full_width_prefill_group(monkeypatch):
    # Stock admission needs prefill_batch_size free slots to form a
    # prompt batch; a prefill group >= the width freezes insertion
    # while any row decodes (8/8 served FIFO). The wrapper clamps the
    # group to 1 only in that regime.
    from mlx_vlm.generate import ar

    import gmlx.spec.engine as spec_engine

    class _BG:
        def __init__(self, model, processor, **kwargs):
            self.completion_batch_size = kwargs.get(
                "completion_batch_size", 32)
            self.prefill_batch_size = kwargs.get("prefill_batch_size", 8)

    monkeypatch.setattr(ar, "BatchGenerator", _BG)
    spec_engine._install_apc_manager_stash()

    monkeypatch.setenv("GMLX_DECODE_BATCH", "8")
    gen = ar.BatchGenerator(SimpleNamespace(), None)
    assert gen.completion_batch_size == 8
    assert gen.prefill_batch_size == 1

    gen = ar.BatchGenerator(SimpleNamespace(), None,
                            completion_batch_size=32)
    assert gen.prefill_batch_size == 8
