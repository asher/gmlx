"""A streamable table the wire load maps outside MLX's tracker: MLX never
counts or wires it, and its install leaves the wired limit free. An
mlx-kquant that cannot map it that way loads it tracked, and the install
pins the limit as for any tracked table."""
from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

import gmlx.stream.table_stream as ts
from gmlx.upstream.patching import wrapped_with
from helpers import _real_apple_gpu

gguf = pytest.importorskip("gguf")

TABLE = "per_layer_token_embd.weight"
ROWS, DIMS = 4096, 512


class _Node:
    pass


@pytest.fixture(autouse=True)
def _fresh_tables(monkeypatch):
    monkeypatch.setattr(ts, "_UNTRACKED", {})
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.delenv("GMLX_STREAM_PLE_COMPOSE", raising=False)
    # The stream flag latch: the load streams the table for sure.
    ts.force_table_stream(True)
    yield
    ts.force_table_stream(False)


@pytest.fixture
def flash_next_gguf(tmp_path):
    """A qwen4exp GGUF whose PLE table is iq4_nl, as in Flash-Next, at
    4096 rows of 512 weights (1.2 MB). Returns its path and wire bytes."""
    rng = np.random.default_rng(5)
    blocks = DIMS // 32
    wire = np.zeros((ROWS, blocks, 18), dtype=np.uint8)
    wire[:, :, 1] = 0x3C          # fp16 1.0 scale, little-endian
    wire[:, :, 2:] = rng.integers(0, 256, (ROWS, blocks, 16), dtype=np.uint8)
    wire = wire.reshape(ROWS, blocks * 18)
    path = str(tmp_path / "Flash-Next-Q4.gguf")
    w = gguf.GGUFWriter(path, "qwen4exp")
    w.add_tensor(TABLE, wire, raw_dtype=gguf.GGMLQuantizationType.IQ4_NL)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), np.float32),
                 raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, wire


def _model(arrays):
    """A qwen4exp model shell holding the loaded table."""
    from mlx_kquant.nn import KQuantEmbedding

    emb = KQuantEmbedding(ROWS, DIMS, "iq4_nl")
    emb.weight = arrays[TABLE]
    emb.scales = arrays["per_layer_token_embd.scales"]
    model = _Node()
    model.model_type = "qwen4_exp"
    model.model = _Node()
    model.model.ple_embed = emb
    return model


def _settled():
    mx.synchronize()
    return mx.get_active_memory()


@pytest.mark.skipif(not ts.untracked_tables_supported(),
                    reason="mlx-kquant has no load_gguf(untracked=...)")
@pytest.mark.skipif(not _real_apple_gpu(),
                    reason="untracked views need the Metal backend")
def test_a_streamed_table_loads_untracked_and_leaves_the_limit_free(
        flash_next_gguf, monkeypatch):
    """The load maps the table outside MLX's tracker: active memory does
    not grow by it. The loader flags it, its install pins nothing, and
    rows gathered on the CPU stream match a tracked copy."""
    import mlx_kquant as kq

    from gmlx.load.wire import load_gguf_wire_bytes

    path, wire = flash_next_gguf
    ref = _model(kq.load_gguf(path, zero_copy=False)[0])
    mx.eval(ref.model.ple_embed.parameters())
    before = _settled()
    arrays = load_gguf_wire_bytes(path, zero_copy=True, shards=[path])[0]
    mx.eval(arrays[TABLE])
    assert _settled() - before < wire.nbytes // 4
    model = _model(arrays)
    assert ts.mark_untracked_tables(model, [path]) == wire.nbytes
    limit = mx.set_wired_limit
    ts.install_table_streaming(model)
    assert not wrapped_with(mx.set_wired_limit, "_kq_no_sweep")
    assert mx.set_wired_limit is limit
    ids = mx.array([[0, 7, 2048], [4095, 7, 1]], dtype=mx.int32)
    got = model.model.ple_embed(ids)
    want = ref.model.ple_embed(ids)
    assert bool(mx.all(got == want))


@pytest.mark.parametrize("error", [
    TypeError("load_gguf() got an unexpected keyword argument 'untracked'"),
    RuntimeError("untracked: no Metal buffer view for this tensor"),
])
def test_a_table_that_cannot_load_untracked_loads_tracked_and_pins(
        flash_next_gguf, monkeypatch, error):
    """An mlx-kquant without load_gguf(untracked=...), or one that cannot
    view the table that way, reloads the shard tracked. The loader flags
    nothing, and the install pins the wired limit."""
    import gmlx.load.wire as wire_mod

    path, wire = flash_next_gguf
    real = wire_mod.kq.load_gguf
    calls = []

    def kq_load(*args, **kwargs):
        calls.append(sorted(kwargs.get("untracked", ())))
        if "untracked" in kwargs:
            raise error
        return real(*args)

    monkeypatch.setattr(wire_mod.kq, "load_gguf", kq_load)
    arrays = wire_mod.load_gguf_wire_bytes(path, zero_copy=True,
                                           shards=[path])[0]
    assert calls == [[TABLE], []]
    model = _model(arrays)
    assert ts.mark_untracked_tables(model, [path]) == 0
    ts.install_table_streaming(model)
    assert wrapped_with(mx.set_wired_limit, "_kq_no_sweep")
