#!/usr/bin/env python3
"""Streamable lookup-table components (gmlx.stream.table_stream): the
descriptor, the selection-ladder test, the CPU-stream gather wrap (parity +
as_linear guard), the load-time warm-touch exclusion, the pin-set exclusion,
and the loader ladder end to end on a tiny synthetic model. Pure CPU."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

import gmlx.stream.table_stream as ts
from gmlx.stream.expert_streaming import install_expert_streaming


def _kquant_table(rows=8, dims=32, seed=7):
    from gguf import quants
    from gguf.constants import GGMLQuantizationType
    from mlx_kquant.nn import KQuantEmbedding

    rng = np.random.default_rng(seed)
    emb = KQuantEmbedding(rows, dims, "q8_0")
    wires = quants.quantize(
        rng.standard_normal((rows, dims)).astype(np.float32) * 0.1,
        GGMLQuantizationType.Q8_0,
    ).astype(np.uint8)
    emb.weight = mx.array(wires.reshape(rows, -1))
    mx.eval(emb.parameters())
    return emb


class _Node:
    pass


@pytest.fixture(autouse=True)
def _clear_force_latch():
    yield
    ts.force_table_stream(False)


def _table_model(model_type="qwen4_exp", rows=8, dims=32):
    model = _Node()
    model.model_type = model_type
    inner = _Node()
    inner.ple_embed = _kquant_table(rows, dims)
    model.model = inner
    return model


# --- descriptor -----------------------------------------------------------


def test_descriptor_resolves_declared_table():
    model = _table_model()
    tabs = ts.streamable_tables_for(model)
    assert len(tabs) == 1
    tier, mod = tabs[0]
    assert tier.gguf_name == "per_layer_token_embd.weight"
    assert mod is model.model.ple_embed
    assert ts.table_bytes(model) == int(mod.weight.nbytes)


def _engram_model(layer_ids=(1, 3), n_layers=5):
    """A deepseek_v41-shaped stand-in: the tables hang off indexed layers,
    so the descriptor must be a callable and the path must index a list."""
    model = _Node()
    model.model_type = "deepseek_v41"
    inner = _Node()
    inner.layers = []
    for i in range(n_layers):
        layer = _Node()
        layer.engram = None
        if i in layer_ids:
            layer.engram = _Node()
            layer.engram.embed = _kquant_table(rows=8 + i)
        inner.layers.append(layer)
    model.model = inner
    return model


def test_descriptor_resolves_indexed_layer_tables():
    model = _engram_model()
    tabs = ts.streamable_tables_for(model)
    assert [t.gguf_name for t, _ in tabs] == [
        "blk.1.engram_embd.weight", "blk.3.engram_embd.weight"]
    assert [t.param_path for t, _ in tabs] == [
        "model.layers.1.engram.embed", "model.layers.3.engram.embed"]
    assert [m for _, m in tabs] == [
        model.model.layers[1].engram.embed,
        model.model.layers[3].engram.embed]
    assert ts.table_bytes(model) == sum(
        int(m.weight.nbytes) for _, m in tabs)


def test_descriptor_out_of_range_index_resolves_to_nothing():
    model = _engram_model(layer_ids=(1,))
    model.model.layers = model.model.layers[:1]   # the table's layer is gone
    assert ts.streamable_tables_for(model) == []


def test_streamable_table_names_by_arch():
    qwen = ts.streamable_table_names("qwen4exp")
    ds = ts.streamable_table_names("deepseek41")
    assert qwen.fullmatch("per_layer_token_embd.weight")
    assert not qwen.fullmatch("blk.1.engram_embd.weight")
    assert ds.fullmatch("blk.1.engram_embd.weight")
    assert ds.fullmatch("blk.14.engram_embd.weight")
    assert not ds.fullmatch("blk.1.engram_q.weight")
    # The PLE-keyed conversion's one table for every engram layer.
    assert ds.fullmatch("per_layer_token_embd.weight")
    assert not ds.fullmatch("blk.1.engram_gate.weight")
    assert ts.streamable_table_names("llama") is None
    assert ts.streamable_table_names(None) is None


def test_descriptor_empty_for_unknown_arch_and_missing_path():
    assert ts.streamable_tables_for(_table_model(model_type="llama")) == []
    bare = _Node()
    bare.model_type = "qwen4_exp"  # declared arch, table absent
    assert ts.streamable_tables_for(bare) == []
    assert ts.table_bytes(bare) == 0


# --- selection ladder -----------------------------------------------------


def test_selection_streams_table_only_when_it_clears_budget(monkeypatch):
    model = _table_model()
    tbytes = ts.table_bytes(model)
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    # over budget, table alone clears it -> stream
    assert ts.table_stream_selected(model, tbytes + 100, 150)
    # over budget, remainder still over -> compose (default on)
    assert ts.table_stream_selected(model, tbytes + 100, 50)
    monkeypatch.setenv("GMLX_STREAM_PLE_COMPOSE", "0")
    assert not ts.table_stream_selected(model, tbytes + 100, 50)
    monkeypatch.delenv("GMLX_STREAM_PLE_COMPOSE")
    # fits -> no streaming
    assert not ts.table_stream_selected(model, tbytes + 100, tbytes + 200)
    # unknown budget -> conservative no
    assert not ts.table_stream_selected(model, tbytes + 100, None)


def test_selection_forced_by_stream_cpu(monkeypatch):
    model = _table_model()
    tbytes = ts.table_bytes(model)
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    fits = tbytes + 200
    assert not ts.table_stream_selected(model, tbytes + 100, fits)
    # --stream-cpu streams the experts whatever the size; the table rides
    assert ts.table_stream_selected(model, tbytes + 100, fits,
                                    force_stream=True)
    # the latch reaches callers that never see the flag (the warm touch)
    ts.force_table_stream()
    assert ts.table_stream_selected(model, tbytes + 100, fits)
    # an explicit off still wins
    monkeypatch.setenv("GMLX_STREAM_PLE", "0")
    assert not ts.table_stream_selected(model, tbytes + 100, fits,
                                        force_stream=True)


def test_selection_env_override(monkeypatch):
    model = _table_model()
    tbytes = ts.table_bytes(model)
    monkeypatch.setenv("GMLX_STREAM_PLE", "0")
    assert not ts.table_stream_selected(model, tbytes + 100, 150)
    monkeypatch.setenv("GMLX_STREAM_PLE", "1")
    # forced even on a fits-in-RAM model (the overhead A/B)
    assert ts.table_stream_selected(model, tbytes + 100, tbytes + 200)


# --- wrap: parity, guard, idempotence ------------------------------------


def test_wrapped_table_gather_is_bit_exact():
    model = _table_model()
    emb = model.model.ple_embed
    rows = mx.array([[0, 3, 7], [5, 1, 2]], dtype=mx.int32)
    ref = emb(rows)
    mx.eval(ref)

    offloaded, names = ts.install_table_streaming(model)
    assert offloaded == int(emb.weight.nbytes)
    assert names == ["per_layer_token_embd.weight"]
    assert emb.__class__.__name__ == "KQuantEmbedding_TableStream"
    assert ts.table_streaming_active(model)

    out = emb(rows)
    mx.eval(out)
    assert out.dtype == ref.dtype
    assert bool(mx.all(out == ref))  # bit-exact, not allclose


def test_a_gather_only_table_streams_at_its_output_dtype():
    """MXFP8 scales reach 2^127. A streamed table decodes like the resident
    one, straight to the output dtype, not through float16."""
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantEmbedding

    if "mxfp8" not in kq.codecs():
        pytest.skip("this mlx-kquant has no mxfp8 codec")
    rows, dims = 4, 64
    raw = np.full((rows, dims // 32, 33), 0x38, dtype=np.uint8)  # e4m3 1.0
    raw[:, :, 0] = np.array([[150], [100], [127], [200]], dtype=np.uint8)
    emb = KQuantEmbedding(rows, dims, "mxfp8")
    emb.weight = mx.array(raw.reshape(rows, -1))
    model = _Node()
    model.model_type = "qwen4_exp"
    model.model = _Node()
    model.model.ple_embed = emb
    ids = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    ref = emb(ids)
    mx.eval(ref)
    ts.install_table_streaming(model)
    out = emb(ids)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out))) and bool(mx.all(out == ref))


def test_wrapped_table_as_linear_raises():
    model = _table_model()
    ts.install_table_streaming(model)
    with pytest.raises(RuntimeError, match="wire"):
        model.model.ple_embed.as_linear(mx.zeros((1, 32)))


def test_install_idempotent():
    model = _table_model()
    b1, _ = ts.install_table_streaming(model)
    cls = model.model.ple_embed.__class__
    b2, _ = ts.install_table_streaming(model)
    assert b1 == b2
    assert model.model.ple_embed.__class__ is cls  # no double wrap


def test_streamed_table_array_ids_cover_weight():
    model = _table_model()
    assert ts.streamed_table_array_ids(model) == set()  # not yet streamed
    ts.install_table_streaming(model)
    ids = ts.streamed_table_array_ids(model)
    emb = model.model.ple_embed
    assert id(emb.weight) in ids and id(emb.scales) in ids


# --- warm-touch exclusion -------------------------------------------------


def test_warm_touch_excludes_table_iff_it_will_stream(monkeypatch):
    model = _table_model()
    emb = model.model.ple_embed
    tbytes = ts.table_bytes(model)
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    # fits-in-RAM, no force: table is touched like any other weight
    assert ts.warm_touch_exclusions(model, tbytes + 100, tbytes + 200) == set()
    # over budget with the table clearing it: the ladder will stream it
    skip = ts.warm_touch_exclusions(model, tbytes + 100, 150)
    assert id(emb.weight) in skip
    # --stream-cpu latched before the load: must also skip
    ts.force_table_stream()
    skip = ts.warm_touch_exclusions(model, tbytes + 100, tbytes + 200)
    assert id(emb.weight) in skip and id(emb.scales) in skip
    ts.force_table_stream(False)
    assert ts.warm_touch_exclusions(model, tbytes + 100, tbytes + 200) == set()
    # forced on a fits model (the P1 overhead A/B): must also skip
    monkeypatch.setenv("GMLX_STREAM_PLE", "1")
    skip = ts.warm_touch_exclusions(model, tbytes + 100, tbytes + 200)
    assert id(emb.weight) in skip
    monkeypatch.setenv("GMLX_STREAM_PLE", "0")
    assert ts.warm_touch_exclusions(model, tbytes + 100, 150) == set()


def test_warm_touch_skips_the_expert_stacks_under_a_stream_flag(monkeypatch):
    """A stream flag streams the experts of a model that fits; the load-time
    touch must not wire them first."""
    from mlx.utils import tree_flatten

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    model, glu = _moe_table_model()
    model.modules = lambda: [model.model.ple_embed, glu]
    experts = {id(a) for _, a in tree_flatten(glu.parameters())}
    fits = 1 << 40
    assert ts.warm_touch_exclusions(model, 100, fits) == set()
    ts.force_table_stream()
    skip = ts.warm_touch_exclusions(model, 100, fits)
    assert experts <= skip
    assert id(model.model.ple_embed.weight) in skip


# --- pin-set exclusion ----------------------------------------------------


def test_every_token_ranges_excludes_streamable_names(tmp_path):
    from gguf import GGUFReader

    from gmlx.stream.pin_weights import every_token_ranges, _PAGE

    p = tmp_path / "m.gguf"
    from gguf import GGUFWriter

    w = GGUFWriter(str(p), "llama")
    w.add_uint32("llama.block_count", 1)
    w.add_tensor("blk.0.ffn_gate_exps.weight",
                 np.zeros((4, 8, 16), dtype=np.float16))
    w.add_tensor("blk.0.attn_q.weight", np.zeros((16, 16), dtype=np.float16))
    # large enough that a page-aligned range around a neighbor cannot
    # accidentally cover it whole
    w.add_tensor("per_layer_token_embd.weight",
                 np.zeros((4096, 16), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    excl = frozenset({"per_layer_token_embd.weight"})
    rs = every_token_ranges(str(p), exclude_names=excl)[str(p)]
    table = next(t for t in GGUFReader(str(p)).tensors
                 if t.name == "per_layer_token_embd.weight")
    off = int(table.data_offset)
    end = off + int(table.n_bytes)
    # the table's interior (past page-alignment slack) is not pinned
    inner = any(a + _PAGE <= off and end <= a + n - _PAGE for a, n in rs)
    assert not inner
    # without the exclusion it is pinned (guards against a silently
    # over-broad exclusion)
    rs_all = every_token_ranges(str(p))[str(p)]
    assert any(a <= off and end <= a + n for a, n in rs_all)


# --- loader ladder, end to end -------------------------------------------


def _moe_table_model(n_experts=4, in_dims=32, hidden=64):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "load"))
    from test_offload import _kquant_glu

    model = _Node()
    model.model_type = "qwen4_exp"
    inner = _Node()
    inner.ple_embed = _kquant_table()
    model.model = inner
    glu = _kquant_glu(n_experts, in_dims, hidden)
    layer = _Node()
    layer.modules = lambda: [glu]
    model.layers = [layer]

    def parameters():
        return {"glu": glu.parameters(),
                "table": {"weight": inner.ple_embed.weight,
                          "scales": inner.ple_embed.scales}}

    model.parameters = parameters
    return model, glu


def _fake_budget(monkeypatch, budget_bytes):
    monkeypatch.setattr(
        mx, "device_info",
        lambda: {"max_recommended_working_set_size": budget_bytes / 0.9})


def test_ladder_streams_table_and_keeps_experts_resident(monkeypatch):
    from mlx.utils import tree_flatten

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    model, glu = _moe_table_model()
    total = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    tbytes = ts.table_bytes(model)
    # over budget; experts + rest fit once the table streams
    _fake_budget(monkeypatch, total - tbytes // 2)

    n, offloaded = install_expert_streaming(model)
    assert n == 1
    assert ts.table_streaming_active(model)
    # experts wrapped but NOT streaming (resident fast path)
    assert not getattr(glu, "_kq_cpu_only", False)


def test_the_stream_flag_streams_the_experts_over_the_budget(monkeypatch):
    """--stream-experts over the budget, on a model whose table alone
    brings it under: the table and the experts stream (compose)."""
    from mlx.utils import tree_flatten

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    model, glu = _moe_table_model()
    total = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    _fake_budget(monkeypatch, total - ts.table_bytes(model) // 2)
    monkeypatch.setattr("gmlx.gen.prefill_decay.deduct_untracked_weights",
                        lambda n, key=None: None)
    install_expert_streaming(model, explicit=True)
    assert ts.table_streaming_active(model)
    assert getattr(glu, "_kq_cpu_only", False)
    assert model._kq_streaming is True


def _file_table_model():
    """A table model whose table the load reads from the file: no array,
    a source of the header's size. Returns the model, its glu, the bytes
    left in arrays and the table bytes."""
    from types import SimpleNamespace
    from typing import Any, cast

    from mlx.utils import tree_flatten

    from gmlx.stream.table_pread import _pread_class

    model, glu = cast("tuple[Any, Any]", _moe_table_model())
    table = model.model.ple_embed
    tbytes = ts.table_bytes(model)
    del table["weight"]
    del table["scales"]
    table._kq_table_source = SimpleNamespace(nbytes=tbytes)
    table.__class__ = _pread_class(table.__class__)
    model.parameters = lambda: {"glu": glu.parameters()}
    flat = cast("list[tuple[str, Any]]", tree_flatten(model.parameters()))
    rest = sum(a.nbytes for _, a in flat)
    assert ts.file_table_bytes(model) == tbytes
    return model, glu, rest, tbytes


def test_the_stream_flag_streams_the_experts_beside_a_table_read_from_the_file(
        monkeypatch):
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    model, glu, rest, tbytes = _file_table_model()
    _fake_budget(monkeypatch, rest + tbytes // 2)
    monkeypatch.setattr("gmlx.gen.prefill_decay.deduct_untracked_weights",
                        lambda n, key=None: None)
    install_expert_streaming(model, explicit=True)
    assert getattr(glu, "_kq_cpu_only", False)
    assert model._kq_streaming is True


@pytest.mark.parametrize("over_without_tables", [False, True])
def test_the_stream_flag_s_arena_share_keys_on_the_tables_streaming(
        monkeypatch, capsys, over_without_tables):
    """A table read from the file has no array, but the model is as large
    as its header says. With --stream-experts, a model the tables bring
    under the budget gets the forced arena share, so the arena does not
    wire every expert, and its -v line says the flag streams it; one
    still over it gets the full arena, and its line names the budget."""
    import types

    import gmlx.gen.prefill_decay as decay
    import gmlx.load.loadlog as loadlog
    import gmlx.stream.budget as sbudget
    import gmlx.stream.pin_weights as pins
    import gmlx.stream.prefetch as prefetch

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    model, _, rest, tbytes = _file_table_model()
    _fake_budget(monkeypatch, rest - 1 if over_without_tables
                 else rest + tbytes // 2)
    monkeypatch.setattr(prefetch, "maybe_make_prefetcher",
                        lambda path: types.SimpleNamespace(offsets={}))
    monkeypatch.setattr(pins, "cast_copies", lambda m, p: types.SimpleNamespace(
        dead_bytes=0, names=frozenset()))
    monkeypatch.setattr(pins, "maybe_pin_weights", lambda *a, **kw: None)
    monkeypatch.setattr(sbudget, "kv_room_bytes",
                        lambda p, **kw: types.SimpleNamespace(bytes=0))
    monkeypatch.setattr(decay, "note_streamed_tracked_bytes",
                        lambda *a, **kw: None)
    monkeypatch.setattr("gmlx.gen.prefill_decay.deduct_untracked_weights",
                        lambda n, key=None: None)
    seen = []

    class Reached(Exception):
        pass

    def arena(*a, **kw):
        seen.append(kw.get("forced"))
        raise Reached

    monkeypatch.setattr("gmlx.stream.expert_streaming._decode_arena_bytes",
                        arena)
    monkeypatch.setattr(loadlog, "_LAST_VERBOSE", True)
    with pytest.raises(Reached):
        install_expert_streaming(model, gguf_path="m.gguf", explicit=True,
                                 feeder_prefill=False, feeder_decode=False)
    assert seen == [not over_without_tables]
    out, err = capsys.readouterr()
    why = (" > ~0 GB wired budget" if over_without_tables
           else ", streaming forced by --stream-experts or stream: experts")
    assert f"[stream] streaming: model 0 GB{why} - 1 MoE layers' " in out + err


def test_the_stream_flag_under_the_budget_streams_table_and_experts(monkeypatch):
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    model, glu = _moe_table_model()
    _fake_budget(monkeypatch, 1 << 40)
    monkeypatch.setattr("gmlx.gen.prefill_decay.deduct_untracked_weights",
                        lambda n, key=None: None)
    install_expert_streaming(model, explicit=True)
    assert ts.table_streaming_active(model)
    assert getattr(glu, "_kq_cpu_only", False)


def test_the_stream_plan_is_the_ladder(monkeypatch):
    """One rule for the install and the serve build's header prediction."""
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.delenv("GMLX_STREAM_PLE_COMPOSE", raising=False)
    plan = ts.stream_plan
    # Over the budget, the table alone brings it under: table-only
    # without the flag, compose with it.
    assert plan(150, 30, 130) == (True, False, False)
    assert plan(150, 30, 130, explicit=True, latched=True) == (True, True, False)
    # Still over post-table: compose; no table: experts only.
    assert plan(150, 10, 130, explicit=True, latched=True) == (True, True, False)
    assert plan(150, 0, 130, explicit=True, latched=True) == (False, True, False)
    # Under the budget the flag forces a stream: compose.
    assert plan(100, 30, 130, explicit=True, latched=True) == (True, True, False)
    # No flag under the budget: nothing streams.
    assert plan(100, 30, 130) == (False, False, False)
    monkeypatch.setenv("GMLX_STREAM_PLE_COMPOSE", "0")
    assert plan(150, 10, 130, explicit=True, latched=True) == (False, True, True)


def _fresh_accounting(monkeypatch):
    import gmlx.gen.prefill_decay as pd

    for name in ("_UNTRACKED_WEIGHTS", "_UNTRACKED_OWNERS", "_STREAMED_TRACKED",
                 "_STREAMED_CAP", "_BUILD_CREDIT"):
        monkeypatch.setattr(pd, name, {})
    monkeypatch.setattr(pd, "_WEIGHTS_OWNER", None)
    return pd


def test_compose_credits_nothing_for_zero_copy_weights(monkeypatch):
    """Zero-copy weights sit in the untracked registry, so a compose install
    deducts the table and the experts and credits neither; read after the
    table deduct, the table looked tracked and the experts credited it."""
    from mlx.utils import tree_flatten

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    pd = _fresh_accounting(monkeypatch)
    model, glu = _moe_table_model()
    key = ("/models/moe.gguf",)
    model._kq_weights_key = key
    total = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    experts = sum(a.nbytes for _, a in tree_flatten(glu.parameters()))
    pd.note_untracked_weights(float(total), key=key)
    _fake_budget(monkeypatch, 1 << 40)
    install_expert_streaming(model, explicit=True)
    assert ts.table_streaming_active(model)
    assert getattr(glu, "_kq_cpu_only", False)
    assert pd.streamed_tracked_bytes() == 0
    assert pd.untracked_weight_bytes() == total - ts.table_bytes(model) - experts


def test_compose_credits_both_for_tracked_weights(monkeypatch):
    from mlx.utils import tree_flatten

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    pd = _fresh_accounting(monkeypatch)
    model, glu = _moe_table_model()
    model._kq_weights_key = ("/models/moe.gguf",)
    experts = sum(a.nbytes for _, a in tree_flatten(glu.parameters()))
    _fake_budget(monkeypatch, 1 << 40)
    install_expert_streaming(model, explicit=True)
    assert pd.streamed_tracked_bytes() == ts.table_bytes(model) + experts


def test_compose_frees_each_byte_once_in_a_mixed_regime(monkeypatch):
    """Part of the weights tracked (casts, repacks), the rest zero-copy: each
    streamed share is credited out of the tracked bytes and deducted from
    the untracked registration only for the rest."""
    from mlx.utils import tree_flatten

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    pd = _fresh_accounting(monkeypatch)
    model, glu = _moe_table_model()
    rest = mx.zeros((1 << 18,), dtype=mx.float32)
    params = model.parameters
    model.parameters = lambda: {**params(), "rest": rest}
    key = ("/models/moe.gguf",)
    model._kq_weights_key = key
    total = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    table = ts.table_bytes(model)
    experts = sum(a.nbytes for _, a in tree_flatten(glu.parameters()))
    tracked = table + experts // 2        # the table and half the experts
    pd.note_untracked_weights(float(total - tracked), key=key)
    _fake_budget(monkeypatch, 1 << 40)
    install_expert_streaming(model, explicit=True)
    assert getattr(glu, "_kq_cpu_only", False)
    assert pd.streamed_tracked_bytes() == tracked
    assert pd.untracked_weight_bytes() == total - table - experts


def _serve_build(monkeypatch, tracked_parts, table_rows=8, ws=1 << 40):
    """A table model through a serve build: the window credits the table
    and the experts, the load adds ``tracked_parts`` to active memory and
    registers the rest untracked. Returns the model, its part sizes, the
    active-memory box and the accounting module."""
    from mlx.utils import tree_flatten

    pd = _fresh_accounting(monkeypatch)
    model, glu = _moe_table_model()
    model.model.ple_embed = _kquant_table(rows=table_rows)
    rest = mx.zeros((4096,), dtype=mx.float32)
    params = model.parameters
    model.parameters = lambda: {**params(), "rest": rest}
    model._kq_weights_key = ("/models/moe.gguf",)
    total = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    size = {"T": ts.table_bytes(model),
            "E": sum(a.nbytes for _, a in tree_flatten(glu.parameters()))}
    size["N"] = total - size["T"] - size["E"]
    box = {"active": 0.0}
    monkeypatch.setattr(mx, "get_active_memory", lambda: box["active"])
    monkeypatch.setattr(mx, "device_info",
                        lambda: {"max_recommended_working_set_size": ws})
    pd.set_untracked_weights_owner("entry")
    pd.note_build_credit("entry", float(size["T"] + size["E"]))
    tracked = sum(size[p] for p in tracked_parts)
    box["active"] = float(tracked)
    pd.note_untracked_weights(float(total - tracked), key=model._kq_weights_key)
    pd.set_untracked_weights_owner(None)
    return model, glu, size, box, pd


@pytest.mark.parametrize("tracked_parts,table_rows", [
    (set(), 8), ({"N", "T", "E"}, 8), ({"E"}, 8),
    ({"E"}, 1024),              # a table at least the experts' size
])
def test_a_compose_install_reads_the_real_free(monkeypatch, tracked_parts,
                                               table_rows):
    """The real installer and unmap release on each regime: free is ws
    minus the resident weights across the walk, after the install and
    after the stacks unmap, and the serve build credit is gone."""
    from gmlx.stream.expert_streaming import _release_unmapped_credit

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    ws = 1 << 40
    model, glu, size, box, pd = _serve_build(monkeypatch, tracked_parts,
                                             table_rows, ws)
    walk = []
    monkeypatch.setattr("gmlx.stream.prefetch.maybe_make_prefetcher",
                        lambda path: walk.append(pd.headroom_bytes()))
    install_expert_streaming(model, explicit=True)
    assert ts.table_streaming_active(model)
    assert getattr(glu, "_kq_cpu_only", False)
    assert walk == [ws - size["N"]]
    assert pd.build_credit_bytes() == 0
    assert pd.headroom_bytes() == ws - size["N"]
    freed = size["E"] if "E" in tracked_parts else 0
    box["active"] -= freed
    _release_unmapped_credit(model, freed)
    assert pd.headroom_bytes() == ws - size["N"]


@pytest.mark.parametrize("tracked_parts", [set(), {"N", "T", "E"}, {"E"}])
def test_a_table_only_install_reads_the_real_free(monkeypatch, tracked_parts):
    from mlx.utils import tree_flatten

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    model, glu = _moe_table_model()
    total = sum(a.nbytes for _, a in tree_flatten(model.parameters())) + 4 * 4096
    table = ts.table_bytes(model)
    ws = int((total - table // 2) / 0.9) + 1       # over, but not post-table
    model, glu, size, box, pd = _serve_build(monkeypatch, tracked_parts, 8, ws)
    # A plain build: the bridge's install takes the table-only ladder.
    install_expert_streaming(model)
    assert ts.table_streaming_active(model)
    assert not getattr(glu, "_kq_cpu_only", False)
    assert pd.build_credit_bytes() == 0
    assert pd.headroom_bytes() == ws - size["N"] - size["E"]


def test_unmapped_stacks_give_back_their_credit(monkeypatch):
    from gmlx.stream.expert_streaming import _release_unmapped_credit

    pd = _fresh_accounting(monkeypatch)
    model, _ = _moe_table_model()
    key = ("/models/moe.gguf",)
    model._kq_weights_key = key
    pd.note_streamed_tracked_bytes(1000.0, key=key, source="experts",
                                   cap=1000.0)
    _release_unmapped_credit(model, 0)            # untracked regime: nothing
    assert pd.streamed_tracked_bytes() == 1000.0
    _release_unmapped_credit(model, 600.0)
    assert pd.streamed_tracked_bytes() == 400.0


def test_ladder_composes_when_table_insufficient(monkeypatch):
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    model, glu = _moe_table_model()
    # over budget even post-table: compose by default - both stream
    _fake_budget(monkeypatch, 10)
    install_expert_streaming(model)
    assert ts.table_streaming_active(model)
    assert getattr(glu, "_kq_cpu_only", False)


def test_ladder_compose_kill_switch(monkeypatch):
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_STREAM_PLE_COMPOSE", "0")
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    model, glu = _moe_table_model()
    _fake_budget(monkeypatch, 10)
    install_expert_streaming(model)
    assert not ts.table_streaming_active(model)  # table stays resident
    assert getattr(glu, "_kq_cpu_only", False)


def test_ladder_compose_env_streams_both(monkeypatch):
    monkeypatch.setenv("GMLX_STREAM_PLE", "1")
    monkeypatch.setenv("GMLX_STREAM_PLE_COMPOSE", "1")
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    model, glu = _moe_table_model()
    _fake_budget(monkeypatch, 10)  # over budget even post-table
    install_expert_streaming(model)
    assert ts.table_streaming_active(model)
    assert getattr(glu, "_kq_cpu_only", False)  # experts stream too


def test_ladder_streams_table_under_force_stream(monkeypatch):
    # --stream-cpu on a model that fits: the table used to stay resident,
    # which on deepseek41 means holding 60 GiB of engram tables.
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    model, glu = _moe_table_model()
    _fake_budget(monkeypatch, 1 << 40)       # far under budget
    install_expert_streaming(model, force_stream=True)
    assert ts.table_streaming_active(model)
    assert getattr(glu, "_kq_cpu_only", False)


def test_ladder_force_stream_honors_compose_kill_switch(monkeypatch):
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_STREAM_PLE_COMPOSE", "0")
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    model, glu = _moe_table_model()
    _fake_budget(monkeypatch, 1 << 40)
    install_expert_streaming(model, force_stream=True)
    assert not ts.table_streaming_active(model)
    assert getattr(glu, "_kq_cpu_only", False)


def test_ladder_env_disable(monkeypatch):
    monkeypatch.setenv("GMLX_STREAM_PLE", "0")
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    model, glu = _moe_table_model()
    from mlx.utils import tree_flatten

    total = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    _fake_budget(monkeypatch, total - ts.table_bytes(model) // 2)
    install_expert_streaming(model)
    assert not ts.table_streaming_active(model)
    assert getattr(glu, "_kq_cpu_only", False)  # falls through to experts


# --- P1: PLE embedding parity, streamed vs resident ----------------------


def _tiny_ple():
    import dataclasses

    from gmlx.models.qwen4_exp.model import ModelArgs, PLEEmbedding

    sizes = [10 + i for i in range(16)]
    offsets = [sum(sizes[:i]) for i in range(16)]
    kw = dict(
        ple_ngram_size=3,
        ple_heads_per_ngram=8,
        ple_eos_token_id=9,
        ple_embed_dim=32,
        ple_layer_multipliers=[23703573157769, 20109073645365,
                               8052911324071],
        ple_head_vocab_sizes=sizes,
        ple_head_offsets=offsets,
    )
    flds = {f.name for f in dataclasses.fields(ModelArgs)}
    ple = PLEEmbedding(ModelArgs(**{k: v for k, v in kw.items() if k in flds}))
    return ple, sum(sizes)


@pytest.mark.parametrize("case", ["fresh", "eos_history", "trimmed"])
def test_ple_embedding_parity_streamed_vs_resident(case):
    ple, n_rows = _tiny_ple()
    table = _kquant_table(rows=n_rows, dims=32)
    model = _Node()
    model.model_type = "qwen4_exp"
    inner = _Node()
    inner.ple_embed = table
    model.model = inner

    if case == "fresh":
        ids = mx.array([[1, 4, 2, 7]])
        cache_r, cache_s = [None] * 4, [None] * 4
    elif case == "eos_history":
        # EOS mid-sequence cuts the n-gram context
        ids = mx.array([[3, 9, 5, 1]])
        cache_r, cache_s = [None] * 4, [None] * 4
    else:  # trimmed: verify-rollback shape - history rewound to a prefix
        ids = mx.array([[6, 2]])
        hist = mx.array([[4, 1]])
        cache_r = [None, None, None, mx.array(hist)]
        cache_s = [None, None, None, mx.array(hist)]

    ref = ple(ids, cache_r, table)
    mx.eval(ref)

    ts.install_table_streaming(model)
    out = ple(ids, cache_s, table)
    mx.eval(out)

    assert out.dtype == ref.dtype
    assert bool(mx.all(out == ref))
    # cache history advanced identically on both paths
    if cache_r[3] is not None:
        assert bool(mx.all(cache_r[3] == cache_s[3]))


def test_ple_embedding_parity_batch_rows():
    # batched decode shape: B=3 rows with distinct histories
    ple, n_rows = _tiny_ple()
    table = _kquant_table(rows=n_rows, dims=32)
    model = _Node()
    model.model_type = "qwen4_exp"
    inner = _Node()
    inner.ple_embed = table
    model.model = inner

    ids = mx.array([[1, 2], [3, 4], [9, 5]])
    hist = mx.array([[7, 0], [9, 9], [2, 6]])
    ref = ple(ids, [None, None, None, mx.array(hist)], table)
    mx.eval(ref)
    ts.install_table_streaming(model)
    out = ple(ids, [None, None, None, mx.array(hist)], table)
    mx.eval(out)
    assert bool(mx.all(out == ref))


def test_table_stream_is_dedicated_and_does_not_leak_default():
    s = ts.table_stream()
    assert s is ts.table_stream()  # singleton
    # A table forward must not rebind the CPU device default
    # (``with mx.stream(...)`` would, permanently - mlx 0.32.1).
    model = _table_model()
    ts.install_table_streaming(model)
    before = mx.default_stream(mx.cpu)
    out = model.model.ple_embed(mx.array([[0, 1]], dtype=mx.int32))
    mx.eval(out)
    assert mx.default_stream(mx.cpu) == before


def _iq4nl_table(rows=8, dims=32, seed=11):
    """IQ4_NL wire bytes crafted directly (gguf-py cannot encode it):
    per 32-elem block, fp16 scale 1.0 + 16 random nibble bytes. Parity
    needs identical bytes through both paths, not a real encoder."""
    from mlx_kquant.nn import KQuantEmbedding, bytes_per_row

    rng = np.random.default_rng(seed)
    bpr = bytes_per_row("iq4_nl", dims)
    blocks = dims // 32
    wires = np.zeros((rows, bpr), dtype=np.uint8)
    for b in range(blocks):
        o = b * 18
        wires[:, o] = 0x00
        wires[:, o + 1] = 0x3C  # fp16 1.0, little-endian
        wires[:, o + 2:o + 18] = rng.integers(
            0, 256, (rows, 16), dtype=np.uint8)
    emb = KQuantEmbedding(rows, dims, "iq4_nl")
    emb.weight = mx.array(wires)
    mx.eval(emb.parameters())
    return emb


def test_wrapped_table_parity_iq4_nl():
    # the real PLE table's codec
    model = _Node()
    model.model_type = "qwen4_exp"
    inner = _Node()
    inner.ple_embed = _iq4nl_table()
    model.model = inner
    emb = inner.ple_embed
    rows = mx.array([[0, 3, 7], [5, 1, 2]], dtype=mx.int32)
    ref = emb(rows)
    mx.eval(ref)
    ts.install_table_streaming(model)
    out = emb(rows)
    mx.eval(out)
    assert out.dtype == ref.dtype
    assert bool(mx.all(out == ref))


# --- fallback-state accounting: pin exclusion + arena sizing --------------


def test_fallback_excludes_streamable_from_pin(monkeypatch):
    import gmlx.stream.pin_weights as pw

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_STREAM_PLE_COMPOSE", "0")  # resident-table state
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    seen = {}

    def _capture(gguf_path, exclude_names=frozenset(), reserved_bytes=0):
        seen["exclude"] = set(exclude_names)
        return None

    monkeypatch.setattr(pw, "maybe_pin_weights", _capture)
    model, glu = _moe_table_model()
    _fake_budget(monkeypatch, 10)  # forces experts-stream fallback
    install_expert_streaming(model, gguf_path="/nonexistent.gguf")
    assert not ts.table_streaming_active(model)
    assert seen.get("exclude") == {"per_layer_token_embd.weight"}


def test_streamed_table_bytes_keys_on_streamed_state(monkeypatch):
    # Declared-but-resident (fallback) tables stay charged: only actually
    # streamed tables report bytes for the arena subtraction.
    model, _ = _moe_table_model()
    assert ts.streamed_table_bytes(model) == 0
    ts.install_table_streaming(model)
    assert ts.streamed_table_bytes(model) == ts.table_bytes(model) > 0


def test_arena_sizing_excludes_streamable_bytes(monkeypatch):
    from gmlx.stream.budget import _decode_arena_bytes
    import gmlx.stream.budget as budget

    monkeypatch.delenv("GMLX_DECODE_ARENA_GB", raising=False)
    monkeypatch.setattr(budget, "_available_ram_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        mx, "device_info",
        lambda: {"memory_size": 137 * 10**9,
                 "max_recommended_working_set_size": 120 * 10**9})
    G = 1 << 30
    offsets = {"s": [(0, 0, 100 * G, 0, "")]}  # 100 GiB of experts
    total = 160 * G  # + 54 GiB table + 6 GiB genuine every-token
    base = _decode_arena_bytes(total, offsets, budget=108 * G)
    fixed = _decode_arena_bytes(total, offsets, budget=108 * G,
                                streamable_bytes=54 * G)
    assert fixed - base >= 53 * G
    assert fixed > 40 * G


# --- tables read from the file (past the device buffer ceiling) -----------


def _file_backed_model():
    """A declared table with no ``weight``: too large for any MTLBuffer,
    so ``table_pread`` reads its rows and drops the array."""
    model = _table_model()
    mod = model.model.ple_embed
    del mod["weight"]
    mod._kq_table_source = _Node()
    mod._kq_table_source.nbytes = 101 << 30
    mod._kq_table_streamed = True
    return model, mod


def test_file_backed_table_is_still_declared():
    model, mod = _file_backed_model()
    assert [m for _, m in ts.streamable_tables_for(model)] == [mod]


def test_file_backed_table_weighs_nothing_in_the_budget_seams():
    """The figure both callers subtract counts live arrays only. Charging
    the file bytes made the ladder pick table-only streaming and leave
    163 GB of experts resident."""
    model, _mod = _file_backed_model()
    assert ts.table_bytes(model) == 0
    assert ts.streamed_table_bytes(model) == 0


def test_selection_skips_a_table_that_is_already_off_the_books(monkeypatch):
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    model, _mod = _file_backed_model()
    # Deeply over budget: the experts must stream, and no arithmetic on
    # the table may say otherwise.
    assert not ts.table_stream_selected(model, 137 << 30, 103 << 30)
    assert ts.install_table_streaming(model) == (0, ["per_layer_token_embd.weight"])
