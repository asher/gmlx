#!/usr/bin/env python3
"""Native-fp (MXFP4 / NVFP4) ggml->MLX de-interleave: pure-numpy layout proofs,
plus the wire-mode dispatch seam (repack gating, module install, CPU parity).

The de-interleave is a positional byte/nibble shuffle plus a scale-byte
passthrough - no dequant, no precision change. The first section proves
exactly that, on the CPU with numpy only (no mlx, no GPU): the codes land at
the right *sequence* positions in MLX's packed-uint32 layout, and the
per-group scale bytes survive unchanged and in order. For MXFP4 a full E2M1
numeric dequant of both layouts is compared bit-for-bit; for NVFP4 (no GGUF to
validate the UE4M3 scale against yet) the shuffle + scale passthrough are
proven directly.

The second section covers ``GMLX_NATIVE_FP`` wire mode: the resolver,
the CLI env preset, ``install_kquant_modules`` dispatch, and numeric parity
of the wire ``KQuantSwitchLinear`` (CPU stream) and packed
``NativeFPSwitchLinear`` (GPU) against a ggml-order dequant reference.

The MLX-side bit-exact ``mx.dequantize`` cross-check is a separate GPU gate.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from gmlx.native_fp import (  # noqa: E402
    E2M1_VALUES,
    mxfp4_deinterleave,
    nvfp4_deinterleave,
)


def _unpack_seq_nibbles(packed: np.ndarray) -> np.ndarray:
    """MLX packed uint32 -> sequential 4-bit codes (8 per word, code ``k`` in bits
    ``4k..4k+3``). Independent of the repack internals - this is what MLX's
    fp4 kernel reads, so matching it proves the packing is correct."""
    packed = np.ascontiguousarray(packed).astype(np.uint32)
    nwords = packed.shape[-1]
    flat = packed.reshape(-1, nwords)
    out = np.empty((flat.shape[0], nwords * 8), dtype=np.uint8)
    for k in range(8):
        out[:, k::8] = ((flat >> np.uint32(4 * k)) & np.uint32(0xF)).astype(np.uint8)
    return out.reshape(*packed.shape[:-1], nwords * 8)


def _ggml_mxfp4_dequant(raw: np.ndarray) -> np.ndarray:
    """Reference ggml block_mxfp4 dequant in sequence order: value at position
    ``j`` = ``E2M1[qs[j]&0xF] * 2^(e-127)``; position ``j+16`` uses ``qs[j]>>4``.
    Returns ``(..., nblk*32)`` float32."""
    raw = np.ascontiguousarray(raw, np.uint8)
    *lead, last = raw.shape
    nblk = last // 17
    blocks = raw.reshape(*lead, nblk, 17)
    e = blocks[..., 0].astype(np.int32)               # E8M0 scale byte / block
    qs = blocks[..., 1:]                               # (..., nblk, 16)
    codes = np.concatenate([qs & 0xF, qs >> 4], axis=-1)   # (..., nblk, 32) seq
    scale = np.exp2((e - 127).astype(np.float32))      # (..., nblk)
    vals = E2M1_VALUES[codes] * scale[..., None]
    return vals.reshape(*lead, nblk * 32)


def test_mxfp4_deinterleave_dequant_matches_ggml():
    rng = np.random.RandomState(0)
    nblk = 5
    # Full-range nibbles (every code exercised); moderate E8M0 exponent so
    # 2^(e-127) stays well inside float32 -> the comparison is exact.
    qs = rng.randint(0, 256, size=(2, nblk, 16)).astype(np.uint8)
    e = rng.randint(120, 135, size=(2, nblk, 1)).astype(np.uint8)
    raw = np.concatenate([e, qs], axis=-1).reshape(2, nblk * 17)

    packed, scales = mxfp4_deinterleave(raw)
    assert packed.dtype == np.uint32 and packed.shape == (2, nblk * 4)
    assert scales.dtype == np.uint8 and scales.shape == (2, nblk)

    # Scale bytes pass through unchanged, in block order.
    assert np.array_equal(scales, e.reshape(2, nblk))

    # Dequant of the MLX layout == dequant of the ggml layout, bit-for-bit.
    codes_mlx = _unpack_seq_nibbles(packed)                       # (2, nblk*32)
    scale_mlx = np.exp2((scales.astype(np.int32) - 127).astype(np.float32))
    vals_mlx = E2M1_VALUES[codes_mlx] * np.repeat(scale_mlx, 32, axis=-1)
    assert np.array_equal(vals_mlx, _ggml_mxfp4_dequant(raw))


def test_mxfp4_packed_geometry_matches_switchlinear():
    # A row of `in_dims` weights -> in_dims/8 uint32 words + in_dims/32 scales,
    # which is exactly NativeFPSwitchLinear's (packed_per_row, scales_per_row).
    in_dims = 256
    nblk = in_dims // 32                       # 32 vals / mxfp4 block
    raw = np.zeros((nblk * 17,), dtype=np.uint8)
    packed, scales = mxfp4_deinterleave(raw)
    assert packed.shape == (in_dims // 8,)     # bits=4 -> in_dims*4/32
    assert scales.shape == (in_dims // 32,)


def test_nvfp4_deinterleave_shuffle_and_scale_passthrough():
    rng = np.random.RandomState(1)
    nblk = 3
    raw = rng.randint(0, 256, size=(2, nblk * 36)).astype(np.uint8)

    packed, scales = nvfp4_deinterleave(raw)
    assert packed.dtype == np.uint32 and packed.shape == (2, nblk * 8)
    assert scales.dtype == np.uint8 and scales.shape == (2, nblk * 4)

    blocks = raw.reshape(2, nblk, 36)
    # Four UE4M3 sub-scale bytes per block, in order.
    assert np.array_equal(scales, blocks[..., :4].reshape(2, nblk * 4))

    # 64 values per block = four 16-value groups; group g = qs[8g:8g+8] with the
    # two-halves split (byte&0xF -> pos j, byte>>4 -> pos j+8).
    qs = blocks[..., 4:].reshape(2, nblk, 4, 8)
    expected = np.concatenate([qs & 0xF, qs >> 4], axis=-1)       # (2,nblk,4,16)
    codes_mlx = _unpack_seq_nibbles(packed).reshape(2, nblk, 4, 16)
    assert np.array_equal(codes_mlx, expected)


def test_nvfp4_packed_geometry():
    # group_size=16, bits=4: in_dims/8 words + in_dims/16 scales.
    in_dims = 128
    nblk = in_dims // 64                        # 64 vals / nvfp4 block
    raw = np.zeros((nblk * 36,), dtype=np.uint8)
    packed, scales = nvfp4_deinterleave(raw)
    assert packed.shape == (in_dims // 8,)
    assert scales.shape == (in_dims // 16,)


# ---------------------------------------------------------------------------
# Wire mode (GMLX_NATIVE_FP): resolver, env preset, install dispatch, and
# numeric parity through the installed modules. These need mlx (and, where
# marked, an mlx-kquant build carrying the fp4 codecs / a Metal device).
# ---------------------------------------------------------------------------


def _kq_has_fp4() -> bool:
    import mlx_kquant as kq

    return {"mxfp4", "nvfp4"} <= set(kq.codecs())


def _tiny_switch_model(n_experts=4, out_dims=64, in_dims=128, bias=False):
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchLinear

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = SwitchLinear(in_dims, out_dims, n_experts,
                                        bias=bias)

    return Tiny()


def test_preset_native_fp_wire_env():
    from gmlx.loader import preset_native_fp_wire_env

    with mock.patch.dict(os.environ):
        os.environ.pop("GMLX_NATIVE_FP", None)
        # no streaming placement -> untouched (auto stays in charge)
        preset_native_fp_wire_env(
            SimpleNamespace(stream_cpu=False, stream_experts=False))
        assert "GMLX_NATIVE_FP" not in os.environ
        # args without the flags at all (bridge surfaces) -> untouched
        preset_native_fp_wire_env(SimpleNamespace())
        assert "GMLX_NATIVE_FP" not in os.environ
        preset_native_fp_wire_env(
            SimpleNamespace(stream_cpu=True, stream_experts=False))
        assert os.environ["GMLX_NATIVE_FP"] == "wire"
    with mock.patch.dict(os.environ, {"GMLX_NATIVE_FP": "packed"}):
        # an explicit user override always wins over the placement preset
        preset_native_fp_wire_env(
            SimpleNamespace(stream_cpu=False, stream_experts=True))
        assert os.environ["GMLX_NATIVE_FP"] == "packed"


def test_resolve_native_fp_wire_modes(monkeypatch):
    from gmlx import loader

    meta = {"blk.0.ffn_gate_exps.weight": "mxfp4"}
    weights = {"blk.0.ffn_gate_exps.weight": np.zeros(64, np.uint8)}
    log = lambda *_: None  # noqa: E731
    kq_fp4 = SimpleNamespace(codecs=lambda: ["q4_k", "mxfp4", "nvfp4"])
    kq_old = SimpleNamespace(codecs=lambda: ["q4_k"])

    # no native-fp tensors -> False regardless of mode
    monkeypatch.setenv("GMLX_NATIVE_FP", "wire")
    monkeypatch.setattr(loader, "kq", kq_fp4)
    assert not loader._resolve_native_fp_wire(
        weights, {"a.weight": "q4_k"}, log)

    # explicit wire
    assert loader._resolve_native_fp_wire(weights, meta, log)

    # explicit wire on a build without the codecs: loud, never a silent repack
    monkeypatch.setattr(loader, "kq", kq_old)
    with pytest.raises(RuntimeError, match="mxfp4"):
        loader._resolve_native_fp_wire(weights, meta, log)

    # explicit packed
    monkeypatch.setenv("GMLX_NATIVE_FP", "packed")
    monkeypatch.setattr(loader, "kq", kq_fp4)
    assert not loader._resolve_native_fp_wire(weights, meta, log)

    # auto: fits the wired budget -> packed kernels; over budget -> wire
    monkeypatch.setenv("GMLX_NATIVE_FP", "auto")
    monkeypatch.setattr(loader, "mx", SimpleNamespace(
        device_info=lambda: {"max_recommended_working_set_size": 10 ** 12}))
    assert not loader._resolve_native_fp_wire(weights, meta, log)
    monkeypatch.setattr(loader, "mx", SimpleNamespace(
        device_info=lambda: {"max_recommended_working_set_size": 50}))
    assert loader._resolve_native_fp_wire(weights, meta, log)

    # auto on a build without the codecs degrades to packed
    monkeypatch.setattr(loader, "kq", kq_old)
    assert not loader._resolve_native_fp_wire(weights, meta, log)


def test_install_packed_default_unchanged():
    import mlx.core as mx

    from gmlx.modules import NativeFPSwitchLinear, install_kquant_modules

    model = _tiny_switch_model()
    n = install_kquant_modules(model, {"experts.weight": "mxfp4"})
    assert n == 1
    m = model.experts
    assert isinstance(m, NativeFPSwitchLinear)
    assert m.weight.dtype == mx.uint32


@pytest.mark.parametrize("codec,bpr", [("mxfp4", 68), ("nvfp4", 72)])
def test_install_wire_dispatches_kquant_switchlinear(codec, bpr):
    if not _kq_has_fp4():
        pytest.skip("mlx-kquant build lacks the fp4 codecs")
    import mlx.core as mx

    from gmlx.modules import KQuantSwitchLinear, install_kquant_modules

    model = _tiny_switch_model(n_experts=4, out_dims=64, in_dims=128)
    n = install_kquant_modules(model, {"experts.weight": codec},
                               native_fp_wire=True)
    assert n == 1
    m = model.experts
    assert isinstance(m, KQuantSwitchLinear)
    assert m.kquant_type == codec
    # zero-copy wire geometry + the (1,) scales placeholder kq.load_gguf emits
    assert m.weight.dtype == mx.uint8 and m.weight.shape == (4, 64, bpr)
    assert m.scales.dtype == mx.uint8 and m.scales.shape == (1,)


@pytest.mark.parametrize("native_fp_wire", [False, True])
def test_install_native_fp_non_switch_fails_loud(native_fp_wire):
    import mlx.nn as nn

    from gmlx.modules import install_kquant_modules

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(32, 32, bias=False)

    with pytest.raises(NotImplementedError, match="mxfp4"):
        install_kquant_modules(Tiny(), {"proj.weight": "mxfp4"},
                               native_fp_wire=native_fp_wire)


def test_expert_gpu_ok_gates_cpu_only_codecs(monkeypatch):
    from gmlx import loader
    from gmlx.loader import _kq_expert_gpu_ok

    # Stub the capability registry so the test is independent of which
    # codecs the installed kq build has Metal kernels for (fp4 flipped to
    # has_matmul=True when the plain-leaf families landed).
    monkeypatch.setattr(
        loader, "kq",
        SimpleNamespace(codec_has_matmul=lambda c: c != "fake_cpu_only"))
    proj = lambda c: SimpleNamespace(kquant_type=c)  # noqa: E731
    glu = lambda g, u, d: SimpleNamespace(  # noqa: E731
        gate_proj=proj(g), up_proj=proj(u), down_proj=proj(d))
    assert _kq_expert_gpu_ok(glu("q4_k", "q4_k", "q6_k"))
    assert _kq_expert_gpu_ok(glu("mxfp4", "mxfp4", "mxfp4"))
    assert not _kq_expert_gpu_ok(glu("fake_cpu_only", "q4_k", "q4_k"))
    assert not _kq_expert_gpu_ok(glu("q4_k", "q4_k", "fake_cpu_only"))
    # non-kquant projections (packed native-fp, plain float) don't gate
    assert _kq_expert_gpu_ok(SimpleNamespace(gate_proj=object()))
    # older kq builds without the query never gate
    monkeypatch.setattr(loader, "kq", SimpleNamespace())
    assert _kq_expert_gpu_ok(glu("fake_cpu_only", "q4_k", "q4_k"))


def _synth_mxfp4_wire(rng, n_experts, out_dims, in_dims):
    # Moderate E8M0 exponents: every nibble code exercised, dynamic range kept
    # small enough for a global-relative error metric to be meaningful.
    nblk = in_dims // 32
    e = rng.randint(121, 132,
                    size=(n_experts, out_dims, nblk, 1)).astype(np.uint8)
    qs = rng.randint(0, 256,
                     size=(n_experts, out_dims, nblk, 16)).astype(np.uint8)
    return np.concatenate([e, qs], axis=-1).reshape(
        n_experts, out_dims, nblk * 17)


def _ue4m3_to_fp32(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, np.uint8)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    man = (b & 0x7).astype(np.float32)
    val = np.where(exp == 0, man * 2.0 ** -9,
                   (1.0 + man * 0.125) * np.exp2((exp - 7).astype(np.float32)))
    return np.where((b == 0) | (b == 0x7F), 0.0, val).astype(np.float32)


def _ggml_nvfp4_dequant(raw: np.ndarray) -> np.ndarray:
    """Reference ggml block_nvfp4 dequant in sequence order: 36 B block = four
    UE4M3 sub-scale bytes + four 8-byte two-halves groups of 16 values."""
    raw = np.ascontiguousarray(raw, np.uint8)
    *lead, last = raw.shape
    nblk = last // 36
    blocks = raw.reshape(*lead, nblk, 36)
    sc = _ue4m3_to_fp32(blocks[..., :4])                    # (..., nblk, 4)
    qs = blocks[..., 4:].reshape(*lead, nblk, 4, 8)
    codes = np.concatenate([qs & 0xF, qs >> 4], axis=-1)    # (..., nblk, 4, 16)
    vals = E2M1_VALUES[codes] * sc[..., None]
    return vals.reshape(*lead, nblk * 64)


def _synth_nvfp4_wire(rng, n_experts, out_dims, in_dims):
    nblk = in_dims // 64
    raw = rng.randint(
        0, 256, size=(n_experts, out_dims, nblk, 36)).astype(np.uint8)
    raw[..., :4] = rng.randint(0x30, 0x41, size=raw[..., :4].shape)
    return raw.reshape(n_experts, out_dims, nblk * 36)


def _moe_reference(deq, x, idx):
    # SwitchGLU calling convention: x (T,1,1,K), indices (T,topk) -> the
    # gather_qmm batch broadcast yields (T,topk,1,out).
    n_tok, topk = idx.shape
    out = np.empty((n_tok, topk, 1, deq.shape[1]), np.float32)
    for t in range(n_tok):
        for j in range(topk):
            out[t, j, 0] = deq[idx[t, j]] @ x[t, 0, 0]
    return out


@pytest.mark.parametrize("codec", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize(
    "n_tok,topk,n_experts,tol",
    [
        (3, 2, 4, 2e-2),    # decode regime: NEON GEMV on q8 activations
        (96, 1, 2, 5e-3),   # >16 rows/expert: f32 dequant + sgemm (bf16 out)
    ],
)
def test_wire_switchlinear_parity(codec, n_tok, topk, n_experts, tol):
    """The installed wire module reproduces the ggml-order dequant reference
    on the CPU stream (the --stream-cpu decode path); the packed module reproduces it
    on the GPU. Same GGUF bytes in, same MoE output out."""
    if not _kq_has_fp4():
        pytest.skip("mlx-kquant build lacks the fp4 codecs")
    import mlx.core as mx

    from gmlx.modules import KQuantSwitchLinear

    out_dims, in_dims = 32, 128
    rng = np.random.RandomState(7)
    if codec == "mxfp4":
        raw = _synth_mxfp4_wire(rng, n_experts, out_dims, in_dims)
        deq = _ggml_mxfp4_dequant(raw)
    else:
        raw = _synth_nvfp4_wire(rng, n_experts, out_dims, in_dims)
        deq = _ggml_nvfp4_dequant(raw)
    x = rng.uniform(-1, 1, size=(n_tok, 1, 1, in_dims)).astype(np.float32)
    idx = rng.randint(0, n_experts, size=(n_tok, topk)).astype(np.uint32)
    ref = _moe_reference(deq, x, idx)
    denom = np.max(np.abs(ref))

    wire = KQuantSwitchLinear(n_experts, out_dims, in_dims, False, codec)
    wire.weight = mx.array(raw)
    with mx.stream(mx.cpu):
        got = wire(mx.array(x), mx.array(idx)).astype(mx.float32)
    assert np.max(np.abs(np.array(got) - ref)) / denom < tol


def test_wire_vs_packed_switchlinear_parity_mxfp4():
    """Cross-check the two dispatch targets against each other: packed
    (de-interleaved bytes through stock mx.gather_qmm on the GPU) and wire
    (untouched bytes through kq.gather_qmm on the CPU stream) agree."""
    if not _kq_has_fp4():
        pytest.skip("mlx-kquant build lacks the fp4 codecs")
    import mlx.core as mx

    if not mx.metal.is_available():
        pytest.skip("packed arm needs the stock Metal mxfp4 kernels")
    from gmlx.modules import KQuantSwitchLinear, NativeFPSwitchLinear

    n_experts, out_dims, in_dims, n_tok, topk = 4, 32, 128, 3, 2
    rng = np.random.RandomState(11)
    raw = _synth_mxfp4_wire(rng, n_experts, out_dims, in_dims)
    x = rng.uniform(-1, 1, size=(n_tok, 1, 1, in_dims)).astype(np.float32)
    idx = rng.randint(0, n_experts, size=(n_tok, topk)).astype(np.uint32)
    ref = _moe_reference(_ggml_mxfp4_dequant(raw), x, idx)
    denom = np.max(np.abs(ref))

    wire = KQuantSwitchLinear(n_experts, out_dims, in_dims, False, "mxfp4")
    wire.weight = mx.array(raw)
    with mx.stream(mx.cpu):
        y_wire = wire(mx.array(x), mx.array(idx)).astype(mx.float32)
    y_wire = np.array(y_wire)

    packed_w, scales = mxfp4_deinterleave(raw)
    packed = NativeFPSwitchLinear(n_experts, out_dims, in_dims, False, "mxfp4")
    packed.weight = mx.array(packed_w)
    packed.scales = mx.array(scales)
    y_packed = np.array(
        packed(mx.array(x), mx.array(idx)).astype(mx.float32))

    assert np.max(np.abs(y_packed - ref)) / denom < 5e-3
    assert np.max(np.abs(y_wire - y_packed)) / denom < 2.5e-2
