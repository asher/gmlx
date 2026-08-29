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
from gmlx.load.loader import install_expert_streaming


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
    # over budget, remainder still over -> fall back (no table stream)
    assert not ts.table_stream_selected(model, tbytes + 100, 50)
    # fits -> no streaming
    assert not ts.table_stream_selected(model, tbytes + 100, tbytes + 200)
    # unknown budget -> conservative no
    assert not ts.table_stream_selected(model, tbytes + 100, None)


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
    # forced on a fits model (the P1 overhead A/B): must also skip
    monkeypatch.setenv("GMLX_STREAM_PLE", "1")
    skip = ts.warm_touch_exclusions(model, tbytes + 100, tbytes + 200)
    assert id(emb.weight) in skip
    monkeypatch.setenv("GMLX_STREAM_PLE", "0")
    assert ts.warm_touch_exclusions(model, tbytes + 100, 150) == set()


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

    deducted = []
    monkeypatch.setattr(
        "gmlx.load.loader.deduct_untracked_weights",
        lambda n, key=None: deducted.append(n))

    n, offloaded = install_expert_streaming(model)
    assert n == 1
    assert ts.table_streaming_active(model)
    # experts wrapped but NOT streaming (resident fast path)
    assert not getattr(glu, "_kq_cpu_only", False)
    assert deducted == [tbytes]


def test_ladder_falls_back_to_experts_when_table_insufficient(monkeypatch):
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    model, glu = _moe_table_model()
    # over budget even without the table: v1 has no compose - table stays
    # resident, experts stream (today's behavior)
    _fake_budget(monkeypatch, 10)
    install_expert_streaming(model)
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
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    seen = {}

    def _capture(gguf_path, exclude_names=frozenset()):
        seen["exclude"] = set(exclude_names)
        return None

    monkeypatch.setattr(pw, "maybe_pin_weights", _capture)
    model, glu = _moe_table_model()
    _fake_budget(monkeypatch, 10)  # forces experts-stream fallback
    install_expert_streaming(model, gguf_path="/nonexistent.gguf")
    assert not ts.table_streaming_active(model)
    assert seen.get("exclude") == {"per_layer_token_embd.weight"}


def test_arena_sizing_excludes_streamable_bytes(monkeypatch):
    from gmlx.load.loader import _decode_arena_bytes
    import gmlx.load.loader as loader

    monkeypatch.delenv("GMLX_DECODE_ARENA_GB", raising=False)
    monkeypatch.setattr(loader, "_available_ram_bytes", lambda *a, **k: None)
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
