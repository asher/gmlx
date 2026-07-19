#!/usr/bin/env python3
"""MoE expert CPU offload (``install_expert_streaming``): the SwitchGLU wrapper
must be numerically transparent, idempotent, layer-scoped via ``n_layers``,
and must leave non-expert modules untouched. Pure CPU - tiny synthetic models,
no GGUF, no GPU (under KQUANT_FORCE_CPU the stream context is a no-op device
hop, which still exercises the wrapper mechanics end to end)."""

from __future__ import annotations


import mlx.core as mx  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402
from mlx_lm.utils import _get_classes  # noqa: E402

from gmlx.config_synth import synthesize_config  # noqa: E402
from gmlx.loader import (  # noqa: E402
    _resolve_prefill_step,
    install_expert_streaming,
    moe_streaming_active,
)

from test_config_synth import _qwen3next_meta  # noqa: E402


def _tiny_moe_model():
    c = synthesize_config(_qwen3next_meta(), tensor_shapes={})
    Model, ModelArgs = _get_classes(c)
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    return model, c


def test_offload_wrapper_is_numerically_transparent():
    mx.random.seed(1)
    glu = SwitchGLU(16, 32, 4)
    mx.eval(glu.parameters())
    x = mx.random.normal((2, 3, 16))
    inds = mx.array(
        [[[0, 2], [2, 1], [1, 3]], [[3, 0], [0, 1], [2, 3]]], dtype=mx.uint32
    )
    ref = glu(x, inds)
    mx.eval(ref)

    class _Holder:  # minimal "model" with one layer holding the SwitchGLU
        pass

    layer = _Holder()
    layer.modules = lambda: [glu]
    model = _Holder()
    model.layers = [layer]
    n, nbytes = install_expert_streaming(model)
    assert n == 1 and nbytes > 0
    assert glu.__class__.__name__ == "SwitchGLU_CPUOffload"
    out = glu(x, inds)
    mx.eval(out)
    assert mx.allclose(ref, out, atol=1e-6, rtol=1e-6)


def test_offload_model_wide_counts_and_idempotence():
    model, c = _tiny_moe_model()
    n_layers = c["num_hidden_layers"]
    n, nbytes = install_expert_streaming(model)
    assert n == n_layers  # every-layer MoE on qwen3_next
    assert nbytes > 0
    # second install is a no-op (no double wrap, no double count)
    n2, _ = install_expert_streaming(model)
    assert n2 == 0
    # non-expert modules untouched
    for layer in model.layers:
        assert "CPUOffload" not in type(layer).__name__
        attn = getattr(layer, "self_attn", None) or layer.linear_attn
        assert "CPUOffload" not in type(attn).__name__
    # forward still runs and matches an unwrapped twin
    twin, _ = _tiny_moe_model()
    toks = mx.array([[1, 2, 3, 4]])
    ref = twin(toks, cache=twin.make_cache())
    out = model(toks, cache=model.make_cache())
    mx.eval(ref, out)
    assert mx.allclose(ref, out, atol=1e-5, rtol=1e-5)


def test_offload_n_layers_scopes_the_wrap():
    model, c = _tiny_moe_model()
    n, _ = install_expert_streaming(model, n_layers=2)
    assert n == 2
    wrapped = [
        "CPUOffload" in type(layer.mlp.switch_mlp).__name__ for layer in model.layers
    ]
    assert wrapped == [True, True, False, False]


def test_offload_overram_streaming_mode(monkeypatch):
    """Models larger than the GPU wired budget enter streaming mode: the
    mlx-lm residency sweep (mx.set_wired_limit) is neutralized, and expert
    calls stay on the CPU stream even above the GPU-routing threshold -
    a GPU expert call would wire the buffers it references."""
    model, _c = _tiny_moe_model()
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )  # ~1 KB budget
    import importlib

    mlg = importlib.import_module("mlx_lm.generate")
    real_set_wired = mx.set_wired_limit
    real_wired_ctx = mlg.wired_limit
    try:
        n, _ = install_expert_streaming(model)
        assert n > 0
        # sweep neutralized: set_wired_limit is the no-op marker
        assert getattr(mx.set_wired_limit, "_kq_no_sweep", False)
        assert mx.set_wired_limit(10**12) == 0
        # and mlx-lm's wired_limit context is the quiet variant (no
        # per-generation large-model warning), still usable as a context
        assert getattr(mlg.wired_limit, "_kq_no_sweep", False)
        with mlg.wired_limit(model, None):
            pass
        # expert modules pinned to CPU regardless of the staging threshold
        monkeypatch.setenv("GMLX_STREAM_GPU_TOKENS", "1")
        cpu_streams = []
        real_stream = mx.stream

        def spy(s):
            cpu_streams.append(s)
            return real_stream(s)

        monkeypatch.setattr(mx, "stream", spy)
        glu = model.layers[0].mlp.switch_mlp
        assert getattr(glu, "_kq_cpu_only", False)
        x = mx.random.normal((1, 4, _c["hidden_size"]))
        inds = mx.zeros((1, 4, 2), dtype=mx.uint32)
        out = glu(x, inds)  # 4 tokens >= 1, but streaming forces CPU
        mx.eval(out)
        assert len(cpu_streams) == 1
    finally:
        mx.set_wired_limit = real_set_wired
        mlg.wired_limit = real_wired_ctx


def _holder_model(glu):
    class _Holder:
        pass

    layer = _Holder()
    layer.modules = lambda: [glu]
    model = _Holder()
    model.layers = [layer]
    return model


def test_offload_prefill_threshold_routes_large_calls_off_cpu(monkeypatch):
    """Calls with >= GMLX_STREAM_GPU_TOKENS tokens skip the CPU stream
    context (prefill staging); decode-shaped calls and threshold 0 use it.
    Both branches stay numerically transparent."""
    mx.random.seed(2)
    glu = SwitchGLU(16, 32, 4)
    mx.eval(glu.parameters())

    x_small = mx.random.normal((1, 3, 16))
    i_small = mx.array([[[0, 2], [2, 1], [1, 3]]], dtype=mx.uint32)
    x_big = mx.random.normal((1, 6, 16))
    i_big = mx.array(
        [[[0, 2], [2, 1], [1, 3], [3, 0], [0, 1], [2, 3]]], dtype=mx.uint32
    )
    ref_small, ref_big = glu(x_small, i_small), glu(x_big, i_big)
    mx.eval(ref_small, ref_big)

    class _Holder:
        pass

    layer = _Holder()
    layer.modules = lambda: [glu]
    model = _Holder()
    model.layers = [layer]
    install_expert_streaming(model)

    cpu_streams = []
    real_stream = mx.stream

    def spy(s):
        cpu_streams.append(s)
        return real_stream(s)

    monkeypatch.setattr(mx, "stream", spy)
    monkeypatch.setenv("GMLX_STREAM_GPU_TOKENS", "4")

    out = glu(x_small, i_small)  # 3 tokens < 4 -> CPU stream
    mx.eval(out)
    assert len(cpu_streams) == 1
    assert mx.allclose(ref_small, out, atol=1e-6, rtol=1e-6)

    cpu_streams.clear()
    out = glu(x_big, i_big)  # 6 tokens >= 4 -> default (GPU) stream
    mx.eval(out)
    assert cpu_streams == []
    assert mx.allclose(ref_big, out, atol=1e-6, rtol=1e-6)

    monkeypatch.setenv("GMLX_STREAM_GPU_TOKENS", "0")  # routing disabled
    cpu_streams.clear()
    out = glu(x_big, i_big)
    mx.eval(out)
    assert len(cpu_streams) == 1
    assert mx.allclose(ref_big, out, atol=1e-6, rtol=1e-6)


def test_inram_all_gpu_auto_policy(monkeypatch):
    """In-RAM installs route decode calls to the GPU stream by default (the
    residency sweep wires the whole model either way, so the CPU hop has no
    memory benefit). An explicit GMLX_STREAM_GPU_TOKENS=0 forces CPU decode."""
    mx.random.seed(6)
    glu = SwitchGLU(16, 32, 4)
    mx.eval(glu.parameters())
    model = _holder_model(glu)
    install_expert_streaming(model)
    assert getattr(glu, "_kq_gpu_tokens_default", None) == 1

    cpu_streams = []
    real_stream = mx.stream

    def spy(s):
        cpu_streams.append(s)
        return real_stream(s)

    monkeypatch.setattr(mx, "stream", spy)
    x = mx.random.normal((1, 1, 16))
    inds = mx.array([[[0, 2]]], dtype=mx.uint32)
    mx.eval(glu(x, inds))  # decode-sized, but auto-policy -> GPU stream
    assert cpu_streams == []

    monkeypatch.setenv("GMLX_STREAM_GPU_TOKENS", "0")
    mx.eval(glu(x, inds))  # explicit opt-out -> CPU stream
    assert len(cpu_streams) == 1


def test_prefetcher_window_and_pass_reset(monkeypatch):
    """on_layer advances a depth-ahead advisory window, advises each layer
    once per pass, and a backward layer jump (next prefill chunk) resets.
    Runs in advise mode; the default pread mode shares the same window
    bookkeeping and differs only in the per-layer population call."""
    from gmlx.prefetch import ExpertPrefetcher

    monkeypatch.setenv("GMLX_STREAM_PREFETCH_MODE", "advise")
    offsets = {li: [] for li in range(6)}  # empty ranges: _advise no-ops
    pf = ExpertPrefetcher(offsets, depth=2)
    calls = []
    pf._advise = calls.append
    pf._pool.submit = lambda fn, *a: fn(*a)  # synchronous for the test

    pf.on_layer(0)
    assert calls == [0, 1, 2]
    pf.on_layer(1)
    assert calls == [0, 1, 2, 3]
    pf.on_layer(3)  # gap (dense layer skipped): window catches up
    assert calls == [0, 1, 2, 3, 4, 5]
    pf.on_layer(5)  # window past the last layer: nothing new
    assert calls == [0, 1, 2, 3, 4, 5]
    pf.on_layer(0)  # next chunk: pass reset, re-advise
    assert calls[6:] == [0, 1, 2]
    pf.enabled = False
    pf.on_layer(1)
    assert len(calls) == 9
    pf.close()


def test_prefetch_offset_regex():
    from gmlx.prefetch import _EXPS_RE

    assert _EXPS_RE.fullmatch("blk.7.ffn_gate_exps.weight").group(1) == "7"
    assert _EXPS_RE.fullmatch("blk.61.ffn_down_exps.weight")
    assert _EXPS_RE.fullmatch("blk.0.ffn_gate_up_exps.weight")
    # non-expert weights stay out: shared experts, dense ffn, norms
    assert _EXPS_RE.fullmatch("blk.3.ffn_gate_shexp.weight") is None
    assert _EXPS_RE.fullmatch("blk.3.ffn_gate.weight") is None
    assert _EXPS_RE.fullmatch("blk.3.ffn_norm.weight") is None


def test_streaming_prefill_paces_prefetcher(monkeypatch):
    """In streaming mode with a prefetcher attached, a prefill-sized call
    evals its input (layer-paced execution) and advances the advisory
    window; decode-sized calls pull only their routed experts (on_decode)."""
    mx.random.seed(7)
    glu = SwitchGLU(16, 32, 4)
    mx.eval(glu.parameters())
    model = _holder_model(glu)
    model.parameters = lambda: {"glu": glu.parameters()}  # exceeds the budget
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    real_set_wired = mx.set_wired_limit
    try:
        install_expert_streaming(model)
        assert getattr(glu, "_kq_cpu_only", False)

        class _FakePF:
            enabled = True
            layers = []
            decode_calls = []

            def on_layer(self, li):
                self.layers.append(li)

            def on_decode(self, li, expert_ids):
                self.decode_calls.append((li, list(expert_ids)))

        pf = _FakePF()
        object.__setattr__(glu, "_kq_prefetcher", pf)
        object.__setattr__(glu, "_kq_li", 5)

        x = mx.random.normal((1, 40, 16))  # 40 tokens >= prefill threshold
        inds = mx.zeros((1, 40, 2), dtype=mx.uint32)
        ref = mx.array(glu(x, inds))
        assert pf.layers == [5]
        assert pf.decode_calls == []

        x1 = mx.random.normal((1, 1, 16))  # decode-sized: routed-expert pull
        i1 = mx.zeros((1, 1, 2), dtype=mx.uint32)
        mx.eval(glu(x1, i1))
        assert pf.layers == [5]
        assert pf.decode_calls == [(5, [0])]

        pf.enabled = False  # kill switch honored
        mx.eval(glu(x, inds))
        mx.eval(glu(x1, i1))
        assert pf.layers == [5]
        assert pf.decode_calls == [(5, [0])]
        mx.eval(ref)
    finally:
        mx.set_wired_limit = real_set_wired


def _kquant_glu(n_experts=4, in_dims=32, hidden=64, seed=3):
    """SwitchGLU whose three projections are q8_0 KQuantSwitchLinears."""
    import numpy as np
    from gguf import quants
    from gguf.constants import GGMLQuantizationType
    from mlx_kquant.nn import KQuantSwitchLinear

    rng = np.random.default_rng(seed)
    glu = SwitchGLU(in_dims, hidden, n_experts)
    dims = {
        "gate_proj": (hidden, in_dims),
        "up_proj": (hidden, in_dims),
        "down_proj": (in_dims, hidden),
    }
    for name, (o, i) in dims.items():
        sub = KQuantSwitchLinear(n_experts, o, i, False, "q8_0")
        wires = [
            quants.quantize(
                rng.standard_normal((o, i)).astype(np.float32) * 0.1,
                GGMLQuantizationType.Q8_0,
            ).astype(np.uint8)
            for _ in range(n_experts)
        ]
        sub.weight = mx.array(np.stack(wires, 0))
        setattr(glu, name, sub)
    mx.eval(glu.parameters())
    return glu


def test_moe_experts_override_targets_offloaded_router(monkeypatch):
    """--moe-experts rewrites the router fan-out (experts per TOKEN) on blocks
    whose experts are offloaded: top_k / num_experts_per_tok on the owning
    block AND on a DeepSeek-style gate submodule. Guards: k<1 and k>experts
    raise; a model with no offloaded experts is a no-op."""
    import mlx.nn as nn

    from gmlx.loader import install_moe_experts_override

    class _Gate(nn.Module):  # DeepSeek-style: top_k lives on the gate
        def __init__(self):
            super().__init__()
            self.top_k = 2
            self.weight = mx.zeros((4, 32))

    class _Block(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.num_experts_per_tok = 2
            self.gate = _Gate()
            self.switch_mlp = glu

    class _Layer(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.mlp = block

    class _Shell:
        pass

    glu = _kquant_glu()
    block = _Block(glu)
    model = _Shell()
    model.layers = [_Layer(block)]
    model.parameters = lambda: {"glu": glu.parameters()}

    assert install_moe_experts_override(model, 1) == 0  # nothing offloaded yet

    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)
    assert getattr(glu, "_kq_cpu_only", False)

    import pytest

    with pytest.raises(ValueError):
        install_moe_experts_override(model, 0)
    with pytest.raises(ValueError):
        install_moe_experts_override(model, 5)  # only 4 experts in the stack

    assert install_moe_experts_override(model, 1) == 1
    assert block.num_experts_per_tok == 1
    assert block.gate.top_k == 1

    x = mx.random.normal((1, 3, 32))
    i1 = mx.array([[[0], [2], [1]]], dtype=mx.uint32)  # k=1-shaped routing
    out = glu(x, i1)
    # Reference arm is offloaded too (same seed = identical weights) so both
    # sides run the CPU expert kernels: Metal vs NEON kquant matmuls are not
    # bit-identical, and only the top-k override is under test here.
    ref_glu = _kquant_glu()
    ref_model = _Shell()
    ref_model.layers = [_Layer(_Block(ref_glu))]
    ref_model.parameters = lambda: {"glu": ref_glu.parameters()}
    install_expert_streaming(ref_model)
    ref = ref_glu(x, i1)
    mx.eval(out, ref)
    assert mx.allclose(ref, out, atol=1e-5, rtol=1e-5)


def test_streaming_prefill_default_resolution(monkeypatch):
    """Prefill-width policy: an explicit request always wins; STREAMING-mode
    models default to 8192 (each prefill chunk re-streams the expert lane, so
    fewer chunks win ~linearly); plain and in-RAM offloaded models pass None
    through to mlx-lm's own default."""
    model, _c = _tiny_moe_model()
    assert not moe_streaming_active(model)
    assert _resolve_prefill_step(model, None) == (None, False)

    n, _ = install_expert_streaming(model)  # tiny model: in-RAM mode
    assert n > 0
    assert not moe_streaming_active(model)
    assert _resolve_prefill_step(model, None) == (None, False)
    assert _resolve_prefill_step(model, 4096) == (4096, False)

    model2, _c2 = _tiny_moe_model()
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )  # force streaming: ~1 KB wired budget
    real_set_wired = mx.set_wired_limit
    try:
        install_expert_streaming(model2)
        assert moe_streaming_active(model2)
        assert _resolve_prefill_step(model2, None) == (8192, True)
        assert _resolve_prefill_step(model2, 4096) == (4096, False)
    finally:
        mx.set_wired_limit = real_set_wired
