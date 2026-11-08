"""Int8 attention on MiniMax-H3: the refusals before any weight loads, the
step's int8 term in the memory plan, a tiny H3 whose default forward
never reaches mlx-kquant's int8 ops, and on a NAX GPU the tiny H3
through ``sdpa_qk8``. The refusals, the plan and the default forward need
no kernel."""
from __future__ import annotations

import os
import types

import mlx.core as mx
import mlx_kquant as kq
import numpy as np
import pytest
from mlx.utils import tree_flatten
from video_fixtures import (H3_HEADER, H3_IDENTITY_FASTH3, fasth3_view,
                            fixture_view, load_json, qk8_kernel, vsa_kernel)

from gmlx.video import fingerprint as vfp
from gmlx.video import layout, memfit
from gmlx.video import loader as vloader
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import model as hm


@pytest.fixture
def int8_load(monkeypatch):
    """``load_dit`` with int8 attention on the header fixture as fl2va,
    or on a file that identifies as FastH3 with ``fast``, which records
    whether the weights load and returns the model."""
    from gmlx.image import loader as iloader

    loads = []

    def load(*a, **k):
        loads.append(a)
        return types.SimpleNamespace(attention="dense"), None

    monkeypatch.setattr(iloader, "load_dit", load)

    def go(fast: bool = False):
        view, variant = fixture_view(H3_HEADER), "fl2va"
        if fast:
            row = load_json(H3_IDENTITY_FASTH3)["files"][
                "fastvideo_fasth3_8step_v2_pruned-Q4_K_M.gguf"]
            monkeypatch.setattr(vfp, "weight_identity",
                                lambda view, names: row["variant_hash"])
            monkeypatch.setattr(vfp, "layout_identity",
                                lambda view, fam: row["layout_hash"])
            view, variant = fasth3_view(), None
        monkeypatch.setattr(vloader, "identify_dit",
                            lambda path, **kw: vfp.identify(view, **kw))
        model, _ = vloader.load_dit("h3.gguf", variant=variant,
                                    attention="int8", log=lambda *x: None)
        return model
    go.loads = loads
    return go


_WHAT = "MiniMax-H3 fl2va runs with int8 attention"


def test_a_kquant_without_sdpa_qk8_is_refused(int8_load, monkeypatch):
    monkeypatch.delattr(kq, "sdpa_qk8", raising=False)
    with pytest.raises(vfp.VideoModelError) as ei:
        int8_load()
    assert str(ei.value) == (
        f"{_WHAT}, and the installed mlx-kquant {kq.__version__} has no "
        f"sdpa_qk8; update mlx-kquant to 0.5.10 or later")
    assert int8_load.loads == []


def test_a_gpu_without_nax_is_refused(int8_load, monkeypatch):
    qk8_kernel(monkeypatch, nax=False)
    with pytest.raises(vfp.VideoModelError) as ei:
        int8_load()
    assert str(ei.value) == (
        f"{_WHAT}, whose kernel needs a GPU with NAX units (M5 or newer), "
        f"and this Mac's GPU has none")
    assert int8_load.loads == []
    monkeypatch.setattr(kq, "nax_available", lambda: True)
    assert int8_load().attention == "int8"
    assert len(int8_load.loads) == 1


def test_fasth3_with_int8_attention_is_refused(int8_load, monkeypatch):
    vsa_kernel(monkeypatch)
    qk8_kernel(monkeypatch)
    with pytest.raises(vfp.VideoModelError) as ei:
        int8_load(fast=True)
    assert str(ei.value) == (
        "MiniMax-H3 fasth3 runs video sparse attention, and int8 "
        "attention applies to dense attention only")
    assert int8_load.loads == []


def test_an_int8_step_holds_the_int8_q_and_k_beside_a_dense_one():
    """The int8 copies of q and k of H3's 56 heads of 128, and their
    float32 scales, one a head and row: 2 x 56 x 128 + 2 x 56 x 4 bytes
    a packed row over the dense step, and no other term moves."""
    kw = dict(dit=12 << 30, encoder=20 << 30, vae=1 << 30,
              audio_vae=1 << 30, rows=37735, text=1024, audio_latents=150,
              pixels=1344 * 768, budget=100 << 30)
    dense = memfit.plan_memory("h3", **kw)
    int8 = memfit.plan_memory("h3", attention="int8", **kw)
    step = int8.terms["denoise"].pop("step") - dense.terms["denoise"].pop(
        "step")
    assert step == (2 * 56 * 128 + 2 * 56 * 4) * 37735
    assert int8.terms == dense.terms


# A tiny H3, head dim 128 as the int8 kernel takes, in bfloat16 but for
# the family's float32 weights, as the loader leaves them.

TINY = {"num_attention_heads": 2, "attention_head_dim": 128,
        "hidden_size": 64, "num_layers": 2, "num_refiner_layers": 1,
        "ffn_dim": 48, "in_channels": 24, "audio_in_channels": 32,
        "patch_size": [1, 2, 2], "text_dim": 40, "rope_freq_dim": 16,
        "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5,
        "adaln_grid": 1025, "adaln_basis": 8}


def _tiny() -> hm.H3Transformer:
    mx.random.seed(0)
    m = hm.H3Transformer(TINY)
    w = {k: 0.05 * mx.random.normal(v.shape)
         for k, v in tree_flatten(m.parameters())}
    for k in w:
        if k.endswith(".weight") and "norm" in k.split(".")[-2]:
            w[k] = 1 + w[k]
        elif k.startswith("blocks.") and k.endswith("adaln_proj.linear.bias"):
            w[k] = 20 * w[k]
        if not any(p in k for p in hf.FAMILY.fp32_keep):
            w[k] = w[k].astype(mx.bfloat16)
    w["rope.inv_freq"] = mx.array(1 / 10000 ** (np.arange(16) / 16),
                                  mx.float32)
    m.load_weights(list(w.items()))
    return m


def _forward(model: hm.H3Transformer,
             chunk: int | None = hf.ATTENTION_CHUNK) -> tuple:
    """The velocity of a 256x256 clip of 22 frames with 8 text rows: 466
    packed rows, from the same inputs on every call."""
    vs, aus = hf.latent_shapes(22, 256, 256)
    lay = layout.pack(hf.TIMELINE, 8, *vs[2:], aus[3])
    key = mx.random.split(mx.random.key(1), 3)
    v = mx.random.normal(vs, key=key[0])
    a = mx.random.normal(aus, key=key[1])
    text = model.caption(mx.random.normal((1, 8, 40), key=key[2]))
    out = model(v, a, 0.25, 0.6, text, lay, chunk=chunk)
    mx.eval(out)
    return out


def _refuse(*a, **k):
    raise AssertionError("the int8 op ran")


def test_the_default_forward_is_dense_bit_for_bit(monkeypatch):
    """The default forward reaches neither int8 op, and gives the bits of
    a forward set back to dense after int8, which does reach them."""
    monkeypatch.setattr(kq, "sdpa_qk8_inputs", _refuse, raising=False)
    monkeypatch.setattr(kq, "sdpa_qk8", _refuse, raising=False)
    model = _tiny()
    assert model.attention == "dense"
    default = _forward(model)
    model.attention = "int8"
    with pytest.raises(AssertionError, match="the int8 op ran"):
        _forward(model)
    model.attention = "dense"
    for x, y in zip(default, _forward(model)):
        assert mx.array_equal(x, y)
    with pytest.raises(ValueError, match="runs dense or int8 attention, "
                                         "not 'int4'"):
        model.attention = "int4"


# On a NAX GPU with an mlx-kquant that has sdpa_qk8.


def _no_kernel() -> str | None:
    """Why the int8 kernel does not run here, or None."""
    if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
        return "the int8 kernel runs on the GPU"
    if getattr(kq, "sdpa_qk8", None) is None:
        return "mlx-kquant has no sdpa_qk8"
    if not kq.nax_available():
        return "the int8 kernel needs a GPU with NAX units"
    return None


_NO_KERNEL = _no_kernel()
nax_only = pytest.mark.skipif(_NO_KERNEL is not None,
                              reason=_NO_KERNEL or "")


def _rel(a, b) -> float:
    a = np.array(a.astype(mx.float32), np.float64)
    b = np.array(b.astype(mx.float32), np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# A sdpa_qk8 call on random bf16 inputs sits 9.4e-3 in relative L2 from
# MLX's attention (lab, kq-i8 checks). A sanity margin of 10 covers the
# tiny model's two blocks and random weights; a wrong layout, scale or
# slice gives a distance near 1.
LAB_CALL_REL = 9.4e-3
SANITY_MARGIN = 10
INT8_VS_DENSE = LAB_CALL_REL * SANITY_MARGIN


@nax_only
def test_int8_attention_runs_sdpa_qk8_in_each_block(monkeypatch):
    """Each DiT block quantizes q and k once and calls sdpa_qk8 once on
    466 rows, and the velocity is within INT8_VS_DENSE, the lab's per
    call distance times a sanity margin of 10, of the dense forward's.
    The bound was set before the op first ran; a miss is a finding."""
    model = _tiny()
    dense = _forward(model)
    calls = []
    stock_inputs, stock_attn = kq.sdpa_qk8_inputs, kq.sdpa_qk8
    monkeypatch.setattr(kq, "sdpa_qk8_inputs", lambda *a: (
        calls.append("inputs") or stock_inputs(*a)))
    monkeypatch.setattr(kq, "sdpa_qk8", lambda *a: (
        calls.append("attn") or stock_attn(*a)))
    model.attention = "int8"
    got = _forward(model)
    assert calls == ["inputs", "attn"] * TINY["num_layers"]
    for x, y in zip(got, dense):
        assert _rel(x, y) < INT8_VS_DENSE


@nax_only
def test_int8_chunks_have_the_whole_calls_bits():
    """sdpa_qk8 gives a chunk of whole 64-row query blocks the bits of its
    rows of one call, so a forward in 64-row chunks, the last one ragged,
    equals a forward of one call a block, bit for bit: each chunk reads
    its own rows of the int8 q and their scales."""
    model = _tiny()
    model.attention = "int8"
    whole = _forward(model, chunk=None)
    for x, y in zip(whole, _forward(model, chunk=64)):
        assert mx.array_equal(x, y)
