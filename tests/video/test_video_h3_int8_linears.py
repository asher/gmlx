"""Int8 linears on MiniMax-H3: the refusals before any weight loads, the
conversion of a tiny H3 GGUF's Q4_K block linears to ``Int8Linear``
leaves, the file's linears as the default bit for bit, a LoRA on the int8
base, and the GGUF's bytes after an int8 load. The conversion and forward
tests run on mlx-kquant's ops on a NAX GPU, and everywhere on reference
ops written from the ops' formula in MLX."""
from __future__ import annotations

import hashlib
import os
import re
import types

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq
import numpy as np
import pytest
from mlx_kquant.nn import KQuantLinear
from video_fixtures import (H3_HEADER, H3_IDENTITY_FASTH3, fasth3_view,
                            fixture_view, int8_kernel, load_json, vsa_kernel)

from gmlx.image.ops import int8_linear
from gmlx.image.ops.int8_linear import Int8Linear
from gmlx.video import fingerprint as vfp
from gmlx.video import layout, pipeline
from gmlx.video import loader as vloader
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import model as hm

_QUIET = {"log": lambda *a: None}


# The refusals, on the published file's header with no weights.


@pytest.fixture
def linears_load(monkeypatch):
    """``load_dit`` with int8 linears on the header fixture as fl2va, or
    on a file that identifies as FastH3 with ``fast``. The image loader
    records its arguments and the conversion records its patterns."""
    from gmlx.image import loader as iloader

    loads, converted = [], []

    def load(*a, **k):
        loads.append(k)
        model = types.SimpleNamespace(attention="dense", _linears="file")
        model.convert_linears = types.MethodType(
            hm.H3Transformer.convert_linears, model)
        return model, None

    monkeypatch.setattr(iloader, "load_dit", load)
    monkeypatch.setattr(int8_linear, "convert", lambda model, patterns: (
        converted.append(patterns), 200)[1])

    def go(fast: bool = False, attention: str = "dense"):
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
                                    attention=attention, linears="int8",
                                    **_QUIET)
        return model
    go.loads, go.converted = loads, converted
    return go


_WHAT = "MiniMax-H3 fl2va runs with int8 linears"


@pytest.mark.parametrize("op", ["int8_matmul", "int8_quantize_rows"])
def test_a_kquant_without_the_int8_ops_is_refused(linears_load, monkeypatch,
                                                  op):
    int8_kernel(monkeypatch)
    monkeypatch.delattr(kq, op)
    with pytest.raises(vfp.VideoModelError) as ei:
        linears_load()
    assert str(ei.value) == (
        f"{_WHAT}, and the installed mlx-kquant {kq.__version__} has no "
        f"int8_matmul; update mlx-kquant to 0.5.11 or later")
    assert linears_load.loads == []


def test_a_gpu_without_nax_is_refused_and_one_with_nax_converts(
        linears_load, monkeypatch):
    """Without NAX units nothing loads. With them the load gate counts
    the int8 DiT's bytes over its file and the largest conversion beside
    it (both from the header, design 3.5), and the family's block linears
    convert once the load returns."""
    int8_kernel(monkeypatch, nax=False)
    with pytest.raises(vfp.VideoModelError) as ei:
        linears_load()
    assert str(ei.value) == (
        f"{_WHAT}, whose kernel needs a GPU with NAX units (M5 or newer), "
        f"and this Mac's GPU has none")
    assert linears_load.loads == []
    monkeypatch.setattr(kq, "nax_available", lambda: True)
    model = linears_load()
    (k,) = linears_load.loads
    assert k["gate_extra"] == 8_441_753_600 + 5 * 28672 * 5376 + 4 * 28672
    assert k["gate_hint"] == ("Run with int8 off or attention, or try a "
                              "smaller DiT quant")
    assert linears_load.converted == [hf.FAMILY.int8_linears]
    assert (model._linears, model.attention) == ("int8", "dense")
    assert linears_load(attention="int8").attention == "int8"


def test_fasth3_converts_its_linears_and_refuses_int8_attention(
        linears_load, monkeypatch):
    """FastH3 runs its video sparse attention with int8 linears: the load
    gate counts H3's 200 block linears, not the gates, and they convert.
    Int8 attention on top is refused, since it replaces dense attention
    only, so the gate's hint names int8 off alone."""
    vsa_kernel(monkeypatch)
    int8_kernel(monkeypatch)
    model = linears_load(fast=True)
    (k,) = linears_load.loads
    assert k["gate_extra"] == 8_441_753_600 + 5 * 28672 * 5376 + 4 * 28672
    assert k["gate_hint"] == "Run with int8 off, or try a smaller DiT quant"
    assert linears_load.converted == [hf.FAMILY.int8_linears]
    assert model._linears == "int8"
    with pytest.raises(vfp.VideoModelError) as ei:
        linears_load(fast=True, attention="int8")
    assert str(ei.value) == (
        "MiniMax-H3 fasth3 runs video sparse attention, and int8 "
        "attention applies to dense attention only")
    assert len(linears_load.loads) == 1


def test_the_family_converts_the_four_linears_of_each_block():
    """qkv_proj, out_proj, fc1 and fc2 of each of the 50 blocks, fc2's
    input in groups of 1024 columns, and no token refiner linear."""
    view = fixture_view(H3_HEADER)
    names = {n.removesuffix(".weight") for n in view.tensors
             if n.endswith(".weight") and len(view.tensors[n].shape) == 2}
    model = types.SimpleNamespace(named_modules=lambda: [(n, None)
                                                         for n in names])
    paths = int8_linear.int8_paths(model, hf.FAMILY.int8_linears)  # type: ignore[arg-type]
    assert len(paths) == 200
    assert {re.sub(r"^blocks\.\d+\.", "", p) for p in paths} == {
        "attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"}
    assert {p: g for p, g in paths.items() if g} == {
        f"blocks.{i}.mlp.fc2": 1024 for i in range(50)}


# A tiny H3 GGUF: the published file's tensors, codecs and order at
# hidden 256 (Q4_K takes rows of 256), 2 heads of 128, ffn 2048 (fc2's
# input in two groups of 1024), 2 blocks and 1 refiner block.

_DIMS = {5376: 256, 7168: 256, 21504: 768, 28672: 4096, 14336: 2048,
         96768: 18 * 256, 10752: 2 * 256, 5120: 64}
TEXT_DIM = 64
DEPTH = 2


def _tiny_tensors(seed: int = 0) -> dict:
    """name -> (numpy array, codec) of the tiny GGUF."""
    from image_fixtures import bf16_bits

    rng = np.random.default_rng(seed)
    out = {}
    for name, shape, codec in load_json(H3_HEADER)["tensors"]:
        m = re.match(r"(token_refiner\.)?blocks\.(\d+)\.(.*)", name)
        names = [name]
        if m:
            if m.group(2) != "0":
                continue
            names = [f"{m.group(1) or ''}blocks.{i}.{m.group(3)}"
                     for i in range(1 if m.group(1) else DEPTH)]
        for n in names:
            dims = tuple(reversed([_DIMS.get(d, d) for d in shape]))
            if n == "rope.inv_freq":
                a = (1 / 10000 ** (np.arange(16) / 16)).astype(np.float32)
            elif codec == "Q4_K":
                w = (rng.standard_normal(dims) / 16).astype(np.float32)
                a = np.array(kq.quantize(mx.array(w), "q4_k")[0])
            else:
                a = 0.05 * rng.standard_normal(dims)
                if re.search(r"norm\d?\.weight$", n):
                    a = 1 + a
                elif n.startswith("blocks.") and n.endswith(
                        "adaln_proj.linear.bias"):
                    a = 20 * a
                a = a.astype(np.float32)
                if codec == "F16":
                    a = a.astype(np.float16)
                elif codec == "BF16":
                    a = bf16_bits(a)
            out[n] = (a, codec)
    return out


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory) -> str:
    from image_fixtures import write_gguf

    return write_gguf(tmp_path_factory.mktemp("h3") / "h3-tiny.gguf",
                      _tiny_tensors(), arch=None)


def _identify(path, *, family=None, variant=None):
    """The tiny file as fl2va: its shapes match no published H3."""
    from gmlx.image.fingerprint import view_from_scans
    from gmlx.image.loader import _scan_shard

    view = view_from_scans([_scan_shard(path)])
    return vfp.VideoIdentified(
        family=hf.FAMILY, config=hf.synth_config(view), prefix="",
        layout="qkv", variant="fl2va", identity_hash=None, layout_hash=None,
        view=view)


def _load(monkeypatch, path: str, linears: str | None = None):
    """The tiny DiT, with the loader's default linears when ``linears`` is
    None. For int8 the refusal sees a NAX GPU during the load only."""
    monkeypatch.setattr(vloader, "identify_dit", _identify)
    kw = {} if linears is None else {"linears": linears}
    with monkeypatch.context() as m:
        if linears == "int8":
            m.setattr(kq, "nax_available", lambda: True)
        model, _ = vloader.load_dit(path, **kw, **_QUIET)
    return model


def _forward(model, chunk: int | None = hf.ATTENTION_CHUNK) -> tuple:
    """The velocity of a 256x256 clip of 22 frames with 8 text rows: 466
    packed rows, from the same inputs on every call."""
    vs, aus = hf.latent_shapes(22, 256, 256)
    lay = layout.pack(hf.TIMELINE, 8, *vs[2:], aus[3])
    key = mx.random.split(mx.random.key(1), 3)
    v = mx.random.normal(vs, key=key[0])
    a = mx.random.normal(aus, key=key[1])
    text = model.caption(mx.random.normal((1, 8, TEXT_DIM), key=key[2]))
    out = model(v, a, 0.25, 0.6, text, lay, chunk=chunk)
    mx.eval(out)
    return out


def _refuse(*a, **k):
    raise AssertionError("an int8 linear op ran")


def test_the_files_linears_are_the_default_bit_for_bit(tiny_gguf,
                                                       monkeypatch):
    """The default load and ``linears="file"`` keep the file's K-quant
    leaves, never reach an int8 linear op, and give the same bits."""
    monkeypatch.setattr(kq, "int8_quantize_rows", _refuse, raising=False)
    monkeypatch.setattr(kq, "int8_matmul", _refuse, raising=False)
    default = _load(monkeypatch, tiny_gguf)
    file = _load(monkeypatch, tiny_gguf, "file")
    for m in (default, file):
        assert m.linears == "file"
        assert not any(isinstance(x, Int8Linear)
                       for _, x in m.named_modules())
    for x, y in zip(_forward(default), _forward(file)):
        assert mx.array_equal(x, y)


# The ops' formula (mlx-kquant's int8_quantize_rows and int8_matmul
# docstrings) in MLX ops: a group's int8 sums are exact in float32 here,
# 127 x 127 x 1024 being under 2^24.


def _ref_quantize_rows(x, *, group_size=None, stream=None):
    xf = x.astype(mx.float32)
    k = xf.shape[-1]
    g = k if group_size is None or group_size >= k else group_size
    lead = xf.shape[:-1]
    xg = xf.reshape(*lead, k // g, g)
    m = mx.max(mx.abs(xg), axis=-1, keepdims=True)
    s = mx.where(m > 0, m / 127, mx.array(1 / 127, mx.float32))
    codes = mx.clip(mx.round(xg / s), -127, 127).astype(mx.int8)
    return codes.reshape(xf.shape), s.reshape(*lead, k // g)


def _int_product(a: mx.array, b: mx.array) -> mx.array:
    """``a @ b.T`` of int8 codes held in float32. The codes and every sum
    under 2^24 are exact in float32, and in TF32, which keeps integers to
    2^11, so the product does not depend on MLX_ENABLE_TF32. Refuses an
    operand that is not codes, so no reference multiplies a scaled
    float."""
    for t in (a, b):
        if not (mx.all(t == mx.round(t)).item()
                and mx.max(mx.abs(t)).item() <= 127):
            raise AssertionError("a reference product takes int8 codes")
    if a.shape[-1] * 127 * 127 >= 1 << 24:
        raise AssertionError("a reference product's sums pass 2^24")
    return a @ b.T


def _ref_matmul(x8, x_scales, w8, w_scales, *, group_size=None,
                out_dtype=mx.bfloat16, stream=None):
    # As mlx-kquant's op: w_scales is float32 [N], one a row.
    if w_scales.shape != w8.shape[:1] or w_scales.dtype != mx.float32:
        raise ValueError(f"w_scales is {w_scales.dtype} {w_scales.shape}")
    k = x8.shape[-1]
    g = k if group_size is None else group_size
    y = mx.zeros((*x8.shape[:-1], w8.shape[0]), mx.float32)
    for i in range(k // g):
        cols = slice(i * g, (i + 1) * g)
        acc = _int_product(x8[..., cols].astype(mx.float32),
                           w8[:, cols].astype(mx.float32))
        y = y + acc * x_scales[..., i:i + 1]
    return (y * w_scales).astype(out_dtype)


def _no_kernel() -> str | None:
    """Why mlx-kquant's int8 linear ops do not run here, or None."""
    if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
        return "the int8 linear kernel runs on the GPU"
    if getattr(kq, "int8_matmul", None) is None:
        return "mlx-kquant has no int8_matmul"
    if not kq.nax_available():
        return "the int8 linear kernel needs a GPU with NAX units"
    return None


_NO_KERNEL = _no_kernel()
nax_only = pytest.mark.skipif(_NO_KERNEL is not None,
                              reason=_NO_KERNEL or "")


@pytest.fixture(params=["reference", pytest.param("mlx_kquant",
                                                  marks=nax_only)])
def ops(request, monkeypatch) -> str:
    """The int8 linear ops a test runs: the reference ops in MLX, or
    mlx-kquant's on a NAX GPU."""
    if request.param == "reference":
        monkeypatch.setattr(kq, "int8_quantize_rows", _ref_quantize_rows,
                            raising=False)
        monkeypatch.setattr(kq, "int8_matmul", _ref_matmul, raising=False)
    return request.param


_PATHS = [f"blocks.{i}.{p}" for i in range(DEPTH)
          for p in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")]


def _leaf(model, path: str):
    return dict(model.named_modules())[path]


def _sim_codes(w: mx.array) -> tuple:
    """The lab sim's h3c weight codes and scales (design 6.2): symmetric
    per output row, scale max |w| / 127."""
    sw = mx.max(mx.abs(w), axis=-1, keepdims=True) / 127
    sw = mx.where(sw > 0, sw, 1 / 127)
    return mx.clip(mx.round(w / sw), -127, 127), sw


# The conversion's codes against the sim's formula: within 1 code, and
# equal at 99.8 percent or more (sdpa_qk8's margin, design 6.1).
CODE_AGREEMENT = 0.998


def test_the_conversion_makes_each_block_linear_int8(tiny_gguf, monkeypatch,
                                                     ops):
    """The 4 x depth block linears become Int8Linear leaves, fc2 with
    activation groups of 1024; the refiner's K-quant linears and the
    adaLN projections stay as loaded. Each weight's codes and scales are
    int8_quantize_rows of the file's float32 dequantized weight bit for
    bit, and agree with the sim's formula. No K-quant leaf of a converted
    path is left reachable."""
    file = _load(monkeypatch, tiny_gguf, "file")
    model = _load(monkeypatch, tiny_gguf, "int8")
    assert (model.linears, file.linears) == ("int8", "file")
    mods, before = dict(model.named_modules()), dict(file.named_modules())
    assert sorted(p for p, m in mods.items()
                  if isinstance(m, Int8Linear)) == sorted(_PATHS)
    assert not any(isinstance(m, KQuantLinear) and p.startswith("blocks.")
                   for p, m in mods.items())
    for p, m in before.items():
        if p not in _PATHS and isinstance(m, (KQuantLinear, nn.Linear)):
            assert type(mods[p]) is type(m), p
    assert isinstance(mods["token_refiner.blocks.0.mlp.fc1"], KQuantLinear)
    for path in _PATHS:
        leaf, base = mods[path], before[path]
        assert leaf.act_group == (1024 if path.endswith("fc2") else None)
        w = kq.dequantize(base["weight"], base["scales"], base.kquant_type,
                          mx.float32)
        codes, scales = kq.int8_quantize_rows(w)
        # The op gives a weight [N, K] scales [N, 1]; int8_matmul takes
        # them as [N].
        assert scales.shape == (w.shape[0], 1)
        assert leaf.weight.dtype == mx.int8
        assert leaf.scales.shape == (w.shape[0],)
        assert leaf.scales.dtype == mx.float32
        assert mx.array_equal(leaf.weight, codes)
        assert mx.array_equal(leaf.scales, scales.reshape(-1))
        sim, _sw = _sim_codes(w)
        diff = mx.abs(leaf.weight.astype(mx.float32) - sim)
        assert mx.max(diff).item() <= 1
        assert mx.mean(diff == 0).item() >= CODE_AGREEMENT, path


def _spy(monkeypatch) -> list:
    """Each Int8Linear call's (path-free) group, input and output."""
    calls = []
    call = Int8Linear.__call__

    def spy(self, x):
        y = call(self, x)
        calls.append((self.act_group, x, y))
        return y
    monkeypatch.setattr(Int8Linear, "__call__", spy)
    return calls


def _rel(a, b) -> float:
    a = np.array(a.astype(mx.float32), np.float64)
    b = np.array(b.astype(mx.float32), np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _sim_product(x: mx.array, w: mx.array, group: int | None) -> mx.array:
    """The lab sim's h3c product (competitor_product, design 6.2) on the
    float32 weight ``w``, cast to x's dtype. The sim's grouped form,
    ``(xi * sx) @ wi.T``, multiplies a scaled float, which TF32 rounds to
    about 1e-3; this takes each group's integer product and scales it
    after, the same sum in exact arithmetic, in group order as the
    kernel."""
    xf = x.astype(mx.float32)
    lead, k = xf.shape[:-1], xf.shape[-1]
    wi, sw = _sim_codes(w)
    g = group or k
    xg = xf.reshape(*lead, k // g, g)
    sx = mx.max(mx.abs(xg), axis=-1, keepdims=True) / 127
    sx = mx.where(sx > 0, sx, 1 / 127)
    xi = mx.clip(mx.round(xg / sx), -127, 127)
    y = mx.zeros((*lead, w.shape[0]), mx.float32)
    for i in range(k // g):
        y = y + _int_product(xi[..., i, :], wi[:, i * g:(i + 1) * g]) \
            * sx[..., i, :]
    return (y * sw.reshape(-1)).astype(x.dtype)


def test_the_references_multiply_codes_and_match_the_sims_form():
    """The reference product refuses a scaled operand, and the grouped
    sim product equals the lab's ``(xi * sx) @ wi.T`` taken in float64,
    to float32 rounding."""
    with pytest.raises(AssertionError, match="takes int8 codes"):
        _int_product(mx.array([[0.5, 1.0]]), mx.array([[1.0, 1.0]]))
    rng = np.random.default_rng(6)
    x = rng.standard_normal((5, 2048)).astype(np.float32)
    w = (rng.standard_normal((32, 2048)) / 16).astype(np.float32)
    got = _sim_product(mx.array(x), mx.array(w), 1024)
    xg = x.reshape(5, 2, 1024)
    sx = np.abs(xg).max(-1, keepdims=True) / np.float32(127)
    xi = np.clip(np.round(xg / sx), -127, 127).astype(np.float64)
    sw = np.abs(w).max(-1, keepdims=True) / np.float32(127)
    wi = np.clip(np.round(w / sw), -127, 127)
    want = ((xi * sx).reshape(5, 2048) @ wi.T.astype(np.float64)) \
        * sw.reshape(-1)
    np.testing.assert_allclose(np.array(got), want, rtol=1e-5, atol=1e-6)


# G3's bar for one int8 call against the sim's float32 product rounded
# to bf16, as design 6.1 fixed it before any kernel ran: G3 measured
# 1.02e-4, margin about 10x. A miss is a finding.
CALL_REL = 1e-3


def test_an_int8_forward_runs_the_int8_ops_on_each_block_linear(
        tiny_gguf, monkeypatch, ops):
    """A forward runs int8_matmul on each of the 4 x depth block linears
    once, fc2 with groups of 1024, and each call's output is within
    CALL_REL, G3's bar, of the sim's product on the same input and
    weight."""
    file = _load(monkeypatch, tiny_gguf, "file")
    model = _load(monkeypatch, tiny_gguf, "int8")
    groups = []
    matmul = kq.int8_matmul
    monkeypatch.setattr(kq, "int8_matmul", lambda *a, **k: (
        groups.append(k.get("group_size")), matmul(*a, **k))[1])
    calls = _spy(monkeypatch)
    _forward(model)
    assert groups == [None, None, None, 1024] * DEPTH
    assert len(calls) == len(_PATHS)
    worst = 0.0
    for path, (group, x, y) in zip(_PATHS, calls):
        base = _leaf(file, path)
        w = kq.dequantize(base["weight"], base["scales"], base.kquant_type,
                          mx.float32)
        assert y.dtype == x.dtype
        err = _rel(y, _sim_product(x, w, group))
        worst = max(worst, err)
        assert err <= CALL_REL, (path, err)
    print(f"tiny H3 int8 linears ({ops}): worst call {worst:.3e}")


@nax_only
def test_chunked_fc1_rows_have_the_whole_calls_bits(tiny_gguf, monkeypatch):
    """fc1 in 64-row chunks, the last ragged, gives the velocity bits of
    fc1 over all rows at once: each chunk quantizes its own rows, and
    int8_matmul gives a slice of rows the bits of those rows of a whole
    call."""
    model = _load(monkeypatch, tiny_gguf, "int8")
    monkeypatch.setattr(hm, "MLP_ROWS", 1 << 30)
    whole = _forward(model)
    monkeypatch.setattr(hm, "MLP_ROWS", 64)
    for x, y in zip(whole, _forward(model)):
        assert mx.array_equal(x, y)


def _lora_file(path, seed: int = 4) -> str:
    """A LoRA in ComfyUI's form on each block linear of the tiny DiT."""
    from lora_fixtures import write_keys

    rng = np.random.default_rng(seed)
    dims = {"attn.qkv_proj": (768, 256), "attn.out_proj": (256, 256),
            "mlp.fc1": (4096, 256), "mlp.fc2": (256, 2048)}
    keys = {}
    for p in _PATHS:
        out, inp = dims[p.split(".", 2)[2]]
        stem = f"diffusion_model.{p}"
        keys[f"{stem}.lora_A.weight"] = (
            rng.standard_normal((4, inp)) * 0.05).astype(np.float32)
        keys[f"{stem}.lora_B.weight"] = (
            rng.standard_normal((out, 4)) * 0.05).astype(np.float32)
        keys[f"{stem}.alpha"] = np.asarray(4.0, np.float32)
    return write_keys(path, keys)


@pytest.mark.parametrize("placement", ["block", "split"])
def test_a_lora_runs_on_the_int8_base_as_its_product_and_the_stock_delta(
        tiny_gguf, monkeypatch, tmp_path, ops, placement):
    """Each wrapped block linear keeps its Int8Linear base, and its call
    is the base's product plus the stock delta bit for bit. The forms are
    own or block, never the low-rank kernel or a split, which need a
    K-quant or float base, under the default placement and under
    GMLX_IMAGE_LORA_PLACEMENT=split alike (design 4)."""
    from gmlx.image.lora import install

    monkeypatch.setattr(install, "_PLACEMENT", placement)
    model = _load(monkeypatch, tiny_gguf, "int8")
    pipe = pipeline.VideoPipeline(dit=model, family=hf.FAMILY,  # type: ignore[arg-type]
                                  variant="fl2va", log=lambda *a: None)
    tid = pipeline.adapter_table(pipe, [(_lora_file(tmp_path / "l.safetensors"),
                                         0.8)])
    prev = install.activate(pipe, tid)
    try:
        rng = np.random.default_rng(5)
        mods = dict(model.named_modules())
        for path in _PATHS:
            leaf = mods[path]
            assert isinstance(leaf, install.ImageLoRALinear), path
            assert isinstance(leaf.base, Int8Linear), path
            k = 2048 if path.endswith("fc2") else 256
            x = mx.array(rng.standard_normal((1, 70, k)).astype(np.float32)
                         ).astype(mx.bfloat16)
            f = leaf._state.factors(path, x.dtype)
            assert isinstance(f, (install._Own, install._Block)), (path, f)
            if isinstance(f, install._Own):
                z = x @ f.down.T
                z = (z.astype(mx.float32) * f.scale).astype(z.dtype)
                want = install._add(leaf.base(x), z, f.up.T)
            else:
                want = install._add(leaf.base(x), x @ f.a_t, f.b_t)
            assert mx.array_equal(leaf(x), want), path
            assert not mx.array_equal(leaf(x), leaf.base(x)), path
    finally:
        install.restore(pipe, prev)


def test_an_int8_load_and_forward_leave_the_file_as_it_was(tiny_gguf,
                                                           monkeypatch):
    """The conversion reads the file's mapped K-quant bytes and writes
    nothing back through the mapping (design 6.2)."""
    monkeypatch.setattr(kq, "int8_quantize_rows", _ref_quantize_rows,
                        raising=False)
    monkeypatch.setattr(kq, "int8_matmul", _ref_matmul, raising=False)

    def digest() -> str:
        with open(tiny_gguf, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    before = digest()
    model = _load(monkeypatch, tiny_gguf, "int8")
    _forward(model)
    del model
    mx.clear_cache()
    assert digest() == before


def test_the_conversion_clears_the_cache_after_each_matrix(tiny_gguf,
                                                           monkeypatch):
    """Each matrix's float32 weight goes back to MLX's cache before the
    next one is read, and the cache is cleared then: MLX reuses a freed
    buffer only near its own size, so a cache kept across the four sizes
    of a block would hold more than memfit.int8_transient counts."""
    events = []
    monkeypatch.setattr(kq, "int8_quantize_rows", lambda *a, **k: (
        events.append("quantize"), _ref_quantize_rows(*a, **k))[1],
        raising=False)
    monkeypatch.setattr(kq, "int8_matmul", _ref_matmul, raising=False)
    clear = mx.clear_cache
    monkeypatch.setattr(mx, "clear_cache", lambda: (
        events.append("clear"), clear())[1])
    cached, read = [], int8_linear._float_weight
    monkeypatch.setattr(int8_linear, "_float_weight", lambda *a: (
        cached.append(mx.get_cache_memory()), read(*a))[1])
    events.clear()
    _load(monkeypatch, tiny_gguf, "int8")
    run = [e for e in events if e in ("quantize", "clear")]
    first = run.index("quantize")
    assert run[first:first + 2 * len(_PATHS)] == (
        ["quantize", "clear"] * len(_PATHS))
    # The weight is freed before the clear, so each later read starts on
    # an empty cache.
    assert cached[1:] == [0] * (len(_PATHS) - 1)


def test_an_int8_linear_takes_one_scale_a_row_as_n_not_n_by_1():
    """int8_quantize_rows gives a weight [N, K] scales [N, 1], and
    int8_matmul takes w_scales [N]: an Int8Linear refuses the op's own
    shape, so a conversion that kept it fails at the leaf, not in the
    kernel."""
    w = mx.zeros((64, 32), mx.int8)
    with pytest.raises(ValueError, match="one float32 scale an output row"):
        Int8Linear(w, mx.ones((64, 1), mx.float32))
    with pytest.raises(ValueError, match="one float32 scale an output row"):
        Int8Linear(w, mx.ones((64,), mx.float16))
    assert Int8Linear(w, mx.ones((64,), mx.float32)).scales.shape == (64,)


def test_a_leaf_the_conversion_cannot_take_changes_nothing(monkeypatch):
    """A bias or an input width off a multiple of 32 is refused before
    any leaf changes."""
    monkeypatch.setattr(kq, "int8_quantize_rows", _ref_quantize_rows,
                        raising=False)
    for leaf, why in ((nn.Linear(64, 32), "blocks.0.mlp.fc2 has a bias"),
                      (nn.Linear(40, 32, bias=False),
                       "blocks.0.mlp.fc2 takes 40 columns, and int8 linears "
                       "take a multiple of 32")):
        root = nn.Module()
        root.blocks = [nn.Module()]
        root.blocks[0].mlp = nn.Module()
        root.blocks[0].mlp.fc1 = nn.Linear(64, 64, bias=False)
        root.blocks[0].mlp.fc2 = leaf
        with pytest.raises(ValueError) as ei:
            int8_linear.convert(root, {"blocks.*.mlp.fc1": None,
                                       "blocks.*.mlp.fc2": None})
        assert str(ei.value).startswith(why)
        assert type(root.blocks[0].mlp.fc1) is nn.Linear
