"""The expert store of the fused MoE block (GMLX_MOE_EXPERT_STORE), full
by default for qwen4_exp and off for other models.

Off, nothing is registered and the block reads its mapped stacks. Full, the
load builds each block's store before anything wires the stacks, the module
tree keeps placeholders of the same shape and byte count, the loader's
dicts let the mapped arrays go, and the block serves every width from the
store. A forward that needs the mapped stacks then raises. A load over the
wired budget, one that streams its experts, one on the CPU device and a
training load build no store, and a block whose fold turns off at some
width keeps its stacks. Real kq kernels on the GPU."""
from __future__ import annotations

import gc
import os
import weakref
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import gmlx.load.expert_store as es
import gmlx.load.modules as modules

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the fused MoE kernels need a real GPU and a GPU default device",
                allow_module_level=True)

D, INTER, E, K = 256, 256, 8, 3


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("GMLX_MOE_EXPERT_STORE", raising=False)
    monkeypatch.delenv("GMLX_MOE_EXPERT_STORE_BLOCKS", raising=False)
    for name in list(os.environ):
        if name.startswith("KQ_MOE_STORE_"):
            monkeypatch.delenv(name)
    for name in ("_BLOCKS", "_DONE"):
        getattr(es, name).clear()
    es._GONE.clear()
    es._BUILT[0] = False
    es.STATS.clear()
    yield
    for name in ("_BLOCKS", "_DONE"):
        getattr(es, name).clear()
    es._GONE.clear()
    es._BUILT[0] = False


class _Two(nn.Module):
    def __init__(self, b0, b1):
        super().__init__()
        self.layers = [b0, b1]

    def __call__(self, x):
        return self.layers[1](self.layers[0](x))


def _block(seed, codec):
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

    from gmlx._vendor.mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    block = Qwen3NextSparseMoeBlock(SimpleNamespace(
        hidden_size=D, moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER, norm_topk_prob=True,
        num_experts=E, num_experts_per_tok=K))

    def q(leaf, shape, c):
        w = mx.array((rng.standard_normal(shape) * 0.3).astype(np.float32))
        leaf.weight, leaf.scales = kq.quantize(w, c)
        return leaf

    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(block.switch_mlp, name, q(
            KQuantSwitchLinear(E, o, i, False, codec), (E, o, i), codec))
        setattr(block.shared_expert, name, q(
            KQuantLinear(i, o, False, "q8_0"), (o, i), "q8_0"))
    block.eval()
    return block


def _model(codec="q5_0", seed=0):
    """Two fused blocks, weights evaluated, as after load_weights."""
    import mlx_kquant as kq

    if not hasattr(kq, "moe_store_pack") or not kq.moe_store_supported(
            codec, "q8_0"):
        pytest.skip("this mlx-kquant has no expert store for the codec")
    m = _Two(_block(seed, codec), _block(seed + 10, codec))
    modules.install_fused_moe_glu(m)
    if any(type(b).__name__ != "_FusedKQuantMoeBlock" for b in m.layers):
        pytest.skip("the fused MoE block is not installable in this build")
    mx.eval(m.parameters())
    return m


def _x(t, seed=1):
    mx.random.seed(seed)
    return (mx.random.normal((1, t, D)) * 0.5).astype(mx.bfloat16)


def _f32(a):
    mx.eval(a)
    return np.array(a.astype(mx.float32))


def _stacks(m):
    return [getattr(b.switch_mlp, n).weight for b in m.layers
            for n in ("gate_proj", "up_proj", "down_proj")]


def test_off_registers_nothing_and_reads_the_mapped_stacks():
    m = _model()
    assert not es._BLOCKS
    assert es.build_at_load(m) == 0
    before = [id(a) for a in _stacks(m)]
    out = m(_x(4))
    mx.eval(out)
    assert [id(a) for a in _stacks(m)] == before
    assert all(b.__dict__.get("_kq_expert_store") is None for b in m.layers)
    assert es.held(m) == (0, set())


@pytest.mark.parametrize("codec", ["q5_0", "iq4_nl", "q8_0", "q4_k"])
def test_full_store_holds_the_experts_once_and_serves_every_width(
        monkeypatch, codec):
    want_m = _model(codec)
    want = {t: _f32(want_m(_x(t))) for t in (1, 2, 5, 8, 12, 40)}

    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    m = _model(codec)
    assert len(es._BLOCKS) == 2
    mapped = _stacks(m)
    shapes = [tuple(a.shape) for a in mapped]
    nbytes = sum(a.nbytes for a in mapped)
    loader_dict = {f"w{i}": a for i, a in enumerate(mapped)}
    loader_dict["other"] = mx.zeros((4,))
    # The loader passes two dicts that name the same arrays.
    hf_dict = dict(loader_dict)
    refs = [weakref.ref(a) for a in mapped]
    del mapped

    assert es.build_at_load(m, drop=(loader_dict, hf_dict)) == 2
    assert list(loader_dict) == list(hf_dict) == ["other"]
    gc.collect()
    assert all(r() is None for r in refs), "a mapped stack is still held"

    held = _stacks(m)
    # Placeholders: the stack's shape and byte count over one expert.
    assert [tuple(a.shape) for a in held] == shapes
    assert es.held(m) == (nbytes, {id(a) for a in held})
    assert es.STATS["blocks"] == 2 and es.STATS["bytes"] == nbytes
    assert all(b.__dict__["_kq_store_only"] for b in m.layers)
    for t, w in want.items():
        got = _f32(m(_x(t)))
        assert got.shape == w.shape
        # The store GLU rounds a weight to half before its products.
        assert np.abs(got - w).max() <= 2e-2 * np.abs(w).max(), t


@pytest.mark.parametrize("codec", ["q5_0", "iq4_nl"])
def test_full_store_wide_rows_equal_the_in_place_sorted_path(
        monkeypatch, codec):
    """From GMLX_MOE_EXPERT_STORE_WIDE_ROWS rows a full store runs the
    members' sorted path with the segment GEMM reading the planes: the
    same bits as the in-place block at those widths, bfloat16 and the
    float32 residual of the hyper-connection models."""
    import mlx_kquant as kq

    if not getattr(kq, "moe_store_seg_supported", lambda *a: False)(codec, D):
        pytest.skip("this mlx-kquant has no wide store kernel for the codec")
    widths = (24, 40, 300)
    want_m = _model(codec)
    want = {t: _f32(want_m(_x(t))) for t in widths}
    want32 = _f32(want_m(_x(40).astype(mx.float32)))

    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE_WIDE_ROWS", "24")
    m = _model(codec)
    assert es.build_at_load(m) == 2
    for b in m.layers:
        assert b.switch_mlp.__dict__["_kq_wide_store"] is b._kq_expert_store
        assert es.wide(b, 24) and not es.wide(b, 23)
    for t in widths:
        assert np.array_equal(_f32(m(_x(t))), want[t]), t
    out32 = m(_x(40).astype(mx.float32))
    assert out32.dtype == mx.float32
    assert np.array_equal(_f32(out32), want32)
    # Below the lever the tile kernels serve, within their rounding.
    got = _f32(m(_x(23)))
    w = _f32(want_m(_x(23)))
    assert np.abs(got - w).max() <= 2e-2 * np.abs(w).max()


@pytest.mark.parametrize("k", [3, 8, 10])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_wide_combine_has_the_bits_of_the_unsort_product_and_sum(k, dtype):
    """The one-kernel mix of the wide form against the three ops it stands
    for: gather to route order, product with the scores, sum over routes."""
    t, n = 40, 256
    rng = np.random.default_rng(k)
    y = mx.array(rng.standard_normal((t * k, n)).astype(np.float32)).astype(dtype)
    scores = mx.array(rng.dirichlet(np.ones(k), t).astype(np.float32)).astype(dtype)
    order = mx.array(rng.permutation(t * k).astype(np.uint32))
    want = (y[mx.argsort(order)].reshape(t, k, n) * scores[..., None]).sum(axis=-2)
    got = es.wide_combine(y, order, scores)
    assert got.dtype == dtype
    assert np.array_equal(_f32(got), _f32(want))


def _q4_block(seed, gu_codec, down_codec):
    """A Flash-Next MoE block on kq weights: routed gate and up in
    gu_codec, routed down in down_codec, a q8_0 shared expert."""
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

    import gmlx.models.qwen4_exp.model as qm

    rng = np.random.default_rng(seed)
    b = qm.SparseMoeBlock(SimpleNamespace(
        hidden_size=D, moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER, norm_topk_prob=True,
        num_experts=E, num_experts_per_tok=K))

    def q(leaf, shape, c):
        w = mx.array((rng.standard_normal(shape) * 0.3).astype(np.float32))
        leaf.weight, leaf.scales = kq.quantize(w, c)
        return leaf

    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        c = down_codec if name == "down_proj" else gu_codec
        setattr(b.switch_mlp, name, q(
            KQuantSwitchLinear(E, o, i, False, c), (E, o, i), c))
        setattr(b.shared_expert, name, q(
            KQuantLinear(i, o, False, "q8_0"), (o, i), "q8_0"))
    b.eval()
    return b


def test_qwen4exp_block_mixes_in_the_wide_kernel_with_the_same_bits(
        monkeypatch):
    """The Flash-Next block on a full store at a wide width: the mixed
    wide forward is taken, and its output has the bits of the op chain
    (GMLX_MOE_WIDE_COMBINE=0)."""
    import mlx_kquant as kq

    codec = "q5_0"
    if not getattr(kq, "moe_store_seg_supported", lambda *a: False)(codec, D):
        pytest.skip("this mlx-kquant has no wide store kernel for the codec")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE_WIDE_ROWS", "24")
    m = _Two(_q4_block(0, codec, codec), _q4_block(10, codec, codec))
    modules.install_fused_moe_glu(m)
    if any(type(b).__name__ != "_FusedKQuantMoeBlock" for b in m.layers):
        pytest.skip("the fused MoE block is not installable in this build")
    mx.eval(m.parameters())
    assert es.build_at_load(m) == 2
    calls = []
    real = es.wide_combine
    monkeypatch.setattr(
        es, "wide_combine", lambda *a: calls.append(1) or real(*a))
    x = _x(40)
    got = [_f32(b(x)) for b in m.layers]
    assert len(calls) == 2
    # Below the wide width, and on a float32 input, the block's own forms.
    m.layers[0](_x(23))
    m.layers[0](x.astype(mx.float32))
    assert len(calls) == 2
    monkeypatch.setenv("GMLX_MOE_WIDE_COMBINE", "0")
    want = [_f32(b(x)) for b in m.layers]
    assert len(calls) == 2
    assert all(np.array_equal(g, w) for g, w in zip(got, want))


def test_wide_rows_zero_or_a_codec_with_no_wide_loader_keeps_the_tiles(
        monkeypatch):
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE_WIDE_ROWS", "0")
    m = _model("q5_0")
    es.build_at_load(m)
    assert not es.wide(m.layers[0], 4096)
    assert "_kq_wide_store" not in m.layers[0].switch_mlp.__dict__
    mx.eval(m(_x(40)))
    for name in ("_BLOCKS", "_DONE"):
        getattr(es, name).clear()
    es._BUILT[0] = False
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE_WIDE_ROWS", "24")
    m = _model("q8_0")
    es.build_at_load(m)
    assert not es.wide(m.layers[0], 4096)
    mx.eval(m(_x(40)))


def test_blocks_lever_stages_the_build(monkeypatch):
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE_BLOCKS", "1")
    m = _model()
    assert es.build_at_load(m) == 1
    assert m.layers[0].__dict__.get("_kq_store_only")
    assert m.layers[1].__dict__.get("_kq_expert_store") is None
    out = m(_x(3))
    mx.eval(out)


def test_a_load_over_the_wired_budget_builds_no_store(monkeypatch):
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    want = _f32(_model()(_x(4)))
    es._BLOCKS.clear()
    m = _model()
    monkeypatch.setattr(es, "_stream_bound", lambda model: True)
    lines = []
    assert es.build_at_load(m, log=lines.append) == 0
    assert any("streams its experts" in line for line in lines)
    assert es.held(m) == (0, set())
    assert np.array_equal(_f32(m(_x(4))), want)


def test_a_store_only_block_refuses_a_forward_that_needs_the_stacks(
        monkeypatch):
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    m = _model()
    es.build_at_load(m)
    b = m.layers[0]
    b.train()
    with pytest.raises(RuntimeError, match="GMLX_MOE_EXPERT_STORE=full"):
        b(_x(2))
    b.eval()
    mx.eval(b(_x(2)))


def test_a_bounded_store_keeps_the_mapped_stacks(monkeypatch):
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "0.001")
    m = _model()
    before = [id(a) for a in _stacks(m)]
    assert es.build_at_load(m) == 0
    out = m(_x(8))
    mx.eval(out)
    assert es.STATS["mode"] == "cap"
    assert [id(a) for a in _stacks(m)] == before
    assert es.held(m) == (0, set())


def _stack_file(tmp_path, m, monkeypatch, *, spoil=False):
    """The model's stacks written to one file at odd offsets, as a GGUF
    holds them, and the header scan answered from it."""
    import gmlx.stream.prefetch as prefetch

    path = str(tmp_path / "stacks.bin")
    ranges = {}
    with open(path, "wb") as f:
        for li, b in enumerate(m.layers):
            for kind in ("gate", "up", "down"):
                w = getattr(b.switch_mlp, f"{kind}_proj").weight
                f.write(b"\x5a" * (37 + 11 * li))
                raw = bytearray(memoryview(w).cast("B"))
                if spoil and kind == "up":
                    raw[-1] ^= 0xFF
                ranges.setdefault(li, []).append(
                    (path, f.tell(), len(raw), w.shape[0], kind))
                f.write(raw)
        f.write(b"\x5a" * 5)
    monkeypatch.setattr(prefetch, "expert_offset_map", lambda p: ranges)
    return path


def test_full_store_reads_the_stacks_from_the_file(monkeypatch, tmp_path):
    want_m = _model()
    want = {t: _f32(want_m(_x(t))) for t in (1, 4, 12)}
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    by_map = _model()
    es.build_at_load(by_map)
    ref = {t: _f32(by_map(_x(t))) for t in want}
    assert es.STATS["mapped"] == 2

    m = _model()
    path = _stack_file(tmp_path, m, monkeypatch)
    lines = []
    assert es.build_at_load(m, paths=[path], log=lines.append) == 2
    assert es.STATS["mapped"] == 0 and es.STATS["stage_bytes"] > 0
    assert any("read from the file" in line for line in lines)
    for t in want:
        got = _f32(m(_x(t)))
        # The same bytes packed: the same store, bit for bit.
        assert np.array_equal(got, ref[t]), t
        assert np.abs(got - want[t]).max() <= 2e-2 * np.abs(want[t]).max()


def test_a_file_range_that_differs_from_the_tensor_packs_from_the_mapping(
        monkeypatch, tmp_path):
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    by_map = _model()
    es.build_at_load(by_map)
    ref = _f32(by_map(_x(5)))

    m = _model()
    path = _stack_file(tmp_path, m, monkeypatch, spoil=True)
    lines = []
    assert es.build_at_load(m, paths=[path], log=lines.append) == 2
    assert es.STATS["mapped"] == 2
    assert any("packed from the mapping" in line for line in lines)
    assert np.array_equal(_f32(m(_x(5))), ref)


def test_planes_lists_the_store_arrays_of_a_full_store(monkeypatch):
    m = _model()
    assert es.planes(m) == []
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    m = _model()
    es.build_at_load(m)
    planes = es.planes(m)
    assert sum(a.nbytes for a in planes) == es.STATS["bytes"]
    assert not {id(a) for a in planes} & es.held(m)[1]


def _stack_gguf(tmp_path, m, codec="q5_0", base=0, before=None):
    """The model's expert stacks as a GGUF with the routed-expert tensor
    names, the codec's raw bytes, layer 0 at block ``base``; ``before``'s
    stacks first, from block 0."""
    import gguf

    path = str(tmp_path / "stacks.gguf")
    w = gguf.GGUFWriter(path, "qwen3next")
    qt = getattr(gguf.GGMLQuantizationType, codec.upper())
    blocks = ([] if before is None else list(enumerate(before.layers))) + [
        (base + li, b) for li, b in enumerate(m.layers)]
    for li, b in blocks:
        for kind in ("gate", "up", "down"):
            raw = np.array(getattr(b.switch_mlp, f"{kind}_proj").weight)
            w.add_tensor(f"blk.{li}.ffn_{kind}_exps.weight", raw,
                         raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_release_maps_the_stacks_back_for_a_mode_the_store_cannot_serve(
        monkeypatch, tmp_path):
    want_m = _model()
    want = _f32(want_m(_x(6)))
    raw = [np.array(a) for a in _stacks(want_m)]
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    m = _model()
    path = _stack_gguf(tmp_path, m)
    assert es.build_at_load(m, paths=[path], log=lambda s: None) == 2
    assert es.STATS["mapped"] == 0
    lines = []
    assert es.release(m, "this load streams its experts",
                      log=lines.append) == 2
    assert "streams its experts" in lines[0]
    assert es.held(m) == (0, set()) and es.planes(m) == []
    assert not any(b.__dict__.get("_kq_store_only") for b in m.layers)
    for a, r in zip(_stacks(m), raw, strict=True):
        assert np.array_equal(np.array(a), r)
    # The in-place path again, bit for bit.
    assert np.array_equal(_f32(m(_x(6))), want)
    assert es.release(m, "again") == 0


def test_an_adapter_release_maps_each_file_once_and_counts_the_stacks(
        monkeypatch, tmp_path):
    """An adapter on the shared experts of two store-only blocks maps the
    file once for both; the stacks mapped again join the load's untracked
    weight bytes, and the verify records that hold the planes go now."""
    import mlx_kquant as kq

    import gmlx.gen.prefill_decay as pd
    import gmlx.models.qwen4_exp.model  # noqa: F401  (the records' owner)

    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    m = _model()
    path = _stack_gguf(tmp_path, m)
    assert es.build_at_load(m, paths=[path], log=lambda s: None) == 2
    key = (path,)
    object.__setattr__(m, "_kq_weights_key", key)
    monkeypatch.setattr(pd, "_UNTRACKED_WEIGHTS", {key: 1000.0})
    object.__setattr__(m.layers[1], "_verify_fused", (0, {"k": "plain"}, []))
    maps = []
    real = kq.load_gguf
    monkeypatch.setattr(kq, "load_gguf",
                        lambda p, *a: maps.append(p) or real(p, *a))
    modules._stamp_gate_slots(
        dict(m.named_modules()),
        [f"layers.{i}.shared_expert.gate_proj" for i in (0, 1)], 0)
    assert maps == [path]
    assert all(b.__dict__["_kq_lora_slots"] == (0,) for b in m.layers)
    assert pd.untracked_weight_bytes_for(key) == 1000 + sum(
        a.nbytes for a in _stacks(m))
    assert vars(m.layers[1])["_verify_fused"] == (-1, {}, [])


def test_a_drafter_reads_and_maps_back_its_own_block(monkeypatch, tmp_path):
    """An MTP drafter's layer 0 is block num_hidden_layers of its sidecar,
    which also holds the target's blocks from 0: the store reads the
    drafter's block from the file and maps that block back."""
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    target = _model(seed=40)
    es._BLOCKS.clear()
    m = _model()
    raw = [np.array(a) for a in _stacks(m)]
    object.__setattr__(m, "_kq_store_blk_base", 2)
    path = _stack_gguf(tmp_path, m, base=2, before=target)
    with es.drafter_load():
        assert es.build_at_load(m, paths=[path], log=lambda s: None) == 2
    assert es.STATS["mapped"] == 0
    assert es.release(m, "this load streams its experts",
                      log=lambda s: None) == 2
    for a, r in zip(_stacks(m), raw, strict=True):
        assert np.array_equal(np.array(a), r)


def test_a_drafter_whose_file_block_is_not_known_builds_no_store(
        monkeypatch, tmp_path):
    """A drafter loader that does not give its layers' block in a file it
    shares with the target builds no store: its module path names the
    target's blocks, which the store would read and a release map back."""
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    target = _model(seed=40)
    es._BLOCKS.clear()
    m = _model()
    path = _stack_gguf(tmp_path, m, base=2, before=target)
    lines = []
    with es.drafter_load():
        assert es.build_at_load(m, paths=[path], log=lines.append) == 0
    assert any("drafter's blocks" in line for line in lines)
    assert es.held(m) == (0, set()) and not es._BLOCKS


@pytest.mark.parametrize("target,off", [
    ("blk.0.ffn_down_shexp.weight", True),
    ("blk.0.ffn_down_exps.weight", True),
    ("blk.0.attn_q.weight", False)])
def test_an_adapter_on_a_block_s_experts_loads_without_the_store(
        monkeypatch, tmp_path, target, off):
    """A serve or CLI load with an adapter that takes a block's experts or
    shared expert builds no store: the install would give it up after the
    build's load time. An adapter elsewhere leaves the store on."""
    import gguf

    import gmlx.commands.cli as cli
    import gmlx.serve.bridge_vlm as bridge

    path = str(tmp_path / "adapter.gguf")
    w = gguf.GGUFWriter(path, "qwen4exp")
    for suffix in (".lora_a", ".lora_b"):
        w.add_tensor(target + suffix, np.zeros((4, 8), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    seen = []

    class _Loaded(Exception):
        pass

    def load_model(*a, **k):
        seen.append(es.store_mode("qwen4_exp"))
        raise _Loaded

    monkeypatch.setattr(bridge, "load_model", load_model)
    with pytest.raises(_Loaded):
        bridge.load_serveable_model("base.gguf", adapter_gguf=path)
    args = SimpleNamespace(adapter=[(path, 1.0)], gguf="base.gguf")
    with cli._check_cli_adapter(args):
        seen.append(es.store_mode("qwen4_exp"))
    want = None if off else ("full", 0)
    assert seen == [want, want] and es.store_mode("qwen4_exp") == ("full", 0)


def test_the_drafter_loaders_run_in_the_drafter_scope(monkeypatch):
    import gmlx.spec.mtp_load as ml

    seen = []

    class _Config(dict):
        def get(self, *a):
            seen.append(es._DRAFTER.get())
            raise LookupError

    monkeypatch.setattr(ml, "_assistant_kind", lambda *a: "dflash")
    monkeypatch.setattr(ml, "_load_dflash_drafter",
                        lambda *a, **k: seen.append(es._DRAFTER.get()))
    ml._load_assistant_drafter("d.gguf", None, {}, zero_copy=True, log=print)
    with pytest.raises(LookupError):
        ml._load_mtp_drafter({}, {}, "qwen35moe", _Config(), None)
    assert seen == [True, True] and not es._DRAFTER.get()


@pytest.mark.parametrize("how", ["flag", "memory"])
def test_a_load_that_streams_its_experts_builds_no_store(
        monkeypatch, tmp_path, how):
    import gmlx.stream.table_stream as ts

    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    m = _model()
    path = _stack_gguf(tmp_path, m)
    assert es.stored_file_ranges([path])
    if how == "flag":
        monkeypatch.setattr(ts, "_FORCED", True)
    else:
        monkeypatch.setattr(ts, "_EXPERTS_FOR_MEMORY", {})
        ts.stream_experts_for_memory(path, "unwired")
    assert es.stored_file_ranges([path]) == {}
    lines = []
    assert es.build_at_load(m, paths=[path], log=lines.append) == 0
    assert any("streams its experts" in line for line in lines)
    assert es.held(m) == (0, set())


def test_a_training_or_cpu_load_builds_no_store(monkeypatch):
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    with es.off("training"):
        m = _model()
        assert es.build_at_load(m, log=lambda s: None) == 0
    assert not es._BLOCKS and es.held(m) == (0, set())
    m = _model()
    mx.set_default_device(mx.cpu)
    try:
        assert es.build_at_load(m, log=lambda s: None) == 0
    finally:
        mx.set_default_device(mx.gpu)
    assert es.held(m) == (0, set())


def test_a_block_whose_fold_turns_off_keeps_its_stacks(monkeypatch):
    """A split-codec block skips the fold at 2 and 3 rows and runs its
    members there, on the mapped stacks: no store for it."""
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    m = _model()
    object.__setattr__(m.layers[0], "_kq_fold_skip_t", (2, 3))
    assert es.build_at_load(m, log=lambda s: None) == 1
    assert not m.layers[0].__dict__.get("_kq_store_only")
    assert m.layers[1].__dict__.get("_kq_store_only")
    mx.eval(m(_x(2)), m(_x(3)), m(_x(5)))


def test_a_second_load_gets_its_bounded_store(monkeypatch):
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "0.001")
    first = _model()
    mx.eval(first(_x(4)))
    assert first.layers[0].__dict__.get("_kq_expert_store") is not None
    m = _model(seed=3)
    mx.eval(m(_x(4)))
    assert m.layers[0].__dict__.get("_kq_expert_store") is not None


def test_release_without_the_shard_paths_raises(monkeypatch):
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    m = _model()
    es.build_at_load(m, log=lambda s: None)
    with pytest.raises(RuntimeError, match="load without the store"):
        es.release(m, "this load streams its experts")


def test_the_load_builds_the_store_kernels_ahead_of_their_first_call(
        monkeypatch):
    """A full store's load builds the pipelines of every store kernel its
    blocks reach on a thread, so no forward builds one; the lever at 0
    starts no thread."""
    import mlx_kquant as kq

    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE_WIDE_ROWS", "24")
    es._WARM[0] = None
    m = _model("iq4_nl")
    assert es.build_at_load(m) == 2
    th = es._WARM[0]
    assert th is not None
    th.join(60)
    assert not th.is_alive() and es.STATS["warm_pipelines"] > 0
    built = len(kq.kernel_keys())
    for t in (1, 2, 5, 8, 12, 40):
        mx.eval(m(_x(t)))
    store = ("kq_store_", "kq_moe_store_plan", "kq_moe_store_combine",
             "kq_moe_tile_plan", "kq_moe_tile_combine")
    late = [k for k in kq.kernel_keys()[built:] if k.startswith(store)]
    assert late == []

    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE_WARM", "0")
    es._WARM[0] = None
    assert es.build_at_load(_model("iq4_nl")) == 2
    assert es._WARM[0] is None


def _down_ok(codec):
    import mlx_kquant as kq

    if not getattr(kq, "moe_store_seg_supported", lambda *a: False)(
            codec, INTER):
        pytest.skip("this mlx-kquant has no wide store kernel for the codec")


@pytest.mark.parametrize("codec", ["q5_0", "iq4_nl"])
def test_down_store_holds_the_down_stack_and_leaves_gate_and_up_mapped(
        monkeypatch, codec):
    """GMLX_MOE_EXPERT_STORE=down: the down stack is held once in the
    store, gate and up stay the mapped arrays and run the in-place GLU.
    Narrow widths differ from in place only by the store down's rounding;
    the members' sorted widths are the in-place bits (the segment GEMM on
    the planes), bfloat16 and a float32 residual."""
    _down_ok(codec)
    narrow, wide = (1, 2, 5, 8, 12), (24, 40, 300)
    want_m = _model(codec)
    want = {t: _f32(want_m(_x(t))) for t in narrow + wide}
    want32 = _f32(want_m(_x(40).astype(mx.float32)))

    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "down")
    m = _model(codec)
    mapped = _stacks(m)
    gate_up = [id(a) for i, a in enumerate(mapped) if i % 3 != 2]
    down = [a for i, a in enumerate(mapped) if i % 3 == 2]
    down_shapes = [tuple(a.shape) for a in down]
    down_bytes = sum(a.nbytes for a in down)
    loader_dict = {f"w{i}": a for i, a in enumerate(mapped)}
    refs = [weakref.ref(a) for a in down]
    del mapped, down

    lines = []
    assert es.build_at_load(m, drop=(loader_dict,), log=lines.append) == 2
    assert any("down stacks" in line for line in lines)
    assert sorted(loader_dict) == ["w0", "w1", "w3", "w4"]
    gc.collect()
    assert all(r() is None for r in refs), "a mapped down stack is still held"

    now = _stacks(m)
    assert [id(a) for i, a in enumerate(now) if i % 3 != 2] == gate_up
    held = [a for i, a in enumerate(now) if i % 3 == 2]
    assert [tuple(a.shape) for a in held] == down_shapes
    assert es.held(m) == (down_bytes, {id(a) for a in held})
    assert es.STATS["mode"] == "down" and es.STATS["bytes"] == down_bytes
    planes = es.planes(m)
    assert sum(a.nbytes for a in planes) == down_bytes
    for b in m.layers:
        assert b.__dict__["_kq_store_only"] and b.__dict__["_kq_store_down"]
        assert b._kq_expert_store.g == () and b._kq_expert_store.u == ()
        assert "_kq_wide_store" not in b.switch_mlp.__dict__
    for t in narrow:
        got = _f32(m(_x(t)))
        # The store down's sums below 8 rows are not the in-place order.
        assert np.abs(got - want[t]).max() <= 2e-2 * np.abs(want[t]).max(), t
    for t in wide:
        assert np.array_equal(_f32(m(_x(t))), want[t]), t
    out32 = m(_x(40).astype(mx.float32))
    assert out32.dtype == mx.float32
    assert np.array_equal(_f32(out32), want32)


def test_down_store_reads_only_the_down_stack_from_the_file(
        monkeypatch, tmp_path):
    _down_ok("q5_0")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "down")
    by_map = _model()
    es.build_at_load(by_map, log=lambda s: None)
    ref = {t: _f32(by_map(_x(t))) for t in (2, 12, 40)}
    assert es.STATS["mapped"] == 2

    m = _model()
    path = _stack_file(tmp_path, m, monkeypatch)
    down_bytes = sum(b.switch_mlp.down_proj.weight.nbytes for b in m.layers)
    assert es.build_at_load(m, paths=[path], log=lambda s: None) == 2
    assert es.STATS["mapped"] == 0
    # One staging buffer, of the down stack's size.
    assert es.STATS["stage_bytes"] == down_bytes // 2
    for t, w in ref.items():
        assert np.array_equal(_f32(m(_x(t))), w), t


def test_down_store_refuses_a_forward_that_needs_the_down_stack(monkeypatch):
    _down_ok("q5_0")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "down")
    m = _model()
    es.build_at_load(m, log=lambda s: None)
    b = m.layers[0]
    b.train()
    with pytest.raises(RuntimeError, match="GMLX_MOE_EXPERT_STORE"):
        b(_x(2))
    with pytest.raises(RuntimeError, match="GMLX_MOE_EXPERT_STORE"):
        b(_x(40))
    b.eval()
    mx.eval(b(_x(2)), b(_x(40)))
    # The leaf read as a plain gather has no mapped stack behind it.
    with pytest.raises(RuntimeError, match="GMLX_MOE_EXPERT_STORE=down"):
        b.switch_mlp.down_proj(
            mx.zeros((1, 1, 1, 1, INTER), dtype=mx.bfloat16),
            mx.zeros((1, 1, K), dtype=mx.uint32))


def test_down_store_release_maps_the_down_stack_back(monkeypatch, tmp_path):
    from mlx_kquant.nn import KQuantSwitchLinear

    _down_ok("q5_0")
    want_m = _model()
    want = {t: _f32(want_m(_x(t))) for t in (6, 40)}
    raw = [np.array(a) for a in _stacks(want_m)]
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "down")
    m = _model()
    path = _stack_gguf(tmp_path, m)
    gate_up = [id(a) for i, a in enumerate(_stacks(m)) if i % 3 != 2]
    assert es.build_at_load(m, paths=[path], log=lambda s: None) == 2
    assert es.release(m, "this load streams its experts",
                      log=lambda s: None) == 2
    assert es.held(m) == (0, set()) and es.planes(m) == []
    for b in m.layers:
        assert not b.__dict__.get("_kq_store_only")
        assert not b.__dict__.get("_kq_store_down")
        assert type(b.switch_mlp.down_proj) is KQuantSwitchLinear
    now = _stacks(m)
    assert [id(a) for i, a in enumerate(now) if i % 3 != 2] == gate_up
    for a, r in zip(now, raw, strict=True):
        assert np.array_equal(np.array(a), r)
    for t, w in want.items():
        assert np.array_equal(_f32(m(_x(t))), w), t


def test_down_store_warms_the_down_kernels_only(monkeypatch):
    import mlx_kquant as kq

    _down_ok("iq4_nl")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "down")
    es._WARM[0] = None
    m = _model("iq4_nl")
    assert es.build_at_load(m, log=lambda s: None) == 2
    th = es._WARM[0]
    assert th is not None
    th.join(60)
    assert not th.is_alive() and es.STATS["warm_pipelines"] > 0
    built = len(kq.kernel_keys())
    for t in (1, 2, 5, 8, 12, 40):
        mx.eval(m(_x(t)))
    store = ("kq_store_", "kq_moe_store_plan", "kq_moe_store_combine",
             "kq_moe_tile_plan", "kq_moe_tile_combine")
    late = [k for k in kq.kernel_keys()[built:] if k.startswith(store)]
    assert late == []
    assert not [k for k in kq.kernel_keys() if k.startswith("kq_store_glu")
                and "iq4_nl" in k and k in kq.kernel_keys()[built:]]


def test_store_mode_values(monkeypatch):
    for v, want in (("", None), ("0", None), ("off", None),
                    ("full", ("full", 0)), ("down", ("down", 0)),
                    ("2", ("cap", 2 * 2**30))):
        monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", v)
        assert es.store_mode() == want
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "up")
    with pytest.raises(ValueError, match="full, down or a size"):
        es.store_mode()


def test_the_default_store_follows_the_model_type(monkeypatch):
    """Unset: full for Flash-Next, its vision wrapper and its MTP head,
    which holds the target's config and a MoE layer but no model_type."""
    from types import SimpleNamespace

    import gmlx.models.qwen4_exp.model as q4

    monkeypatch.delenv("GMLX_MOE_EXPERT_STORE", raising=False)
    text = SimpleNamespace(model_type="qwen4_exp")
    head = SimpleNamespace(config=SimpleNamespace(text=q4.ModelArgs()))
    for model, want in ((text, ("full", 0)),
                        (SimpleNamespace(language_model=text), ("full", 0)),
                        (head, ("full", 0)),
                        (SimpleNamespace(model_type="qwen3_5_moe"), None)):
        assert es.store_mode(es.model_type_of(model)) == want


def test_qwen4exp_wide_gate_up_run_as_one_kernel_with_the_same_bits(
        monkeypatch):
    """Gate and up in a paired codec: the wide form runs them and the
    SwiGLU as one kq kernel, with the bits of two calls and the activation
    (GMLX_MOE_WIDE_PAIR=0). A codec with no paired kernel keeps two calls."""
    import mlx_kquant as kq

    if not (hasattr(kq, "moe_store_seg_pair")
            and kq.moe_store_seg_supported("iq4_xs", D)
            and kq.moe_store_seg_supported("q5_0", INTER)):
        pytest.skip("this mlx-kquant has no paired wide store kernel")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE", "full")
    monkeypatch.setenv("GMLX_MOE_EXPERT_STORE_WIDE_ROWS", "24")
    m = _Two(_q4_block(0, "iq4_xs", "q5_0"), _q4_block(10, "q5_0", "q5_0"))
    modules.install_fused_moe_glu(m)
    if any(type(b).__name__ != "_FusedKQuantMoeBlock" for b in m.layers):
        pytest.skip("the fused MoE block is not installable in this build")
    mx.eval(m.parameters())
    assert es.build_at_load(m) == 2
    calls = []
    real = kq.moe_store_seg_pair
    monkeypatch.setattr(
        kq, "moe_store_seg_pair",
        lambda *a, **k: calls.append(a[3]) or real(*a, **k))
    x = _x(40)
    got = [_f32(b(x)) for b in m.layers]
    assert calls == ["iq4_xs"]
    monkeypatch.setenv("GMLX_MOE_WIDE_PAIR", "0")
    want = [_f32(b(x)) for b in m.layers]
    assert calls == ["iq4_xs"]
    assert all(np.array_equal(g, w) for g, w in zip(got, want))
