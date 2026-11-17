"""MiniMax-H3-Image's int8 setting, as H3 video's: the four block linears
it converts and the bytes they hold, read from the release header; the
load's refusals and its gate; the int8 terms of a step; and a LoRA on the
int8 linears of the tiny DiT. CPU-only, no release weights. The product
gate of each setting on the release file is in
test_image_check3_h3_image.py."""
from __future__ import annotations

import dataclasses
from collections import Counter
from types import SimpleNamespace

import mlx.core as mx
import mlx_kquant as kq
import numpy as np
import pytest
from image_fixtures import Z_HEADER, fixture_view
from lora_fixtures import h3_diffusers_keys, reference_delta, write_keys
from oracle_fixture import committed_fixture

from gmlx.image import fingerprint as ifp
from gmlx.image import int8 as neutral
from gmlx.image import loader, memfit
from gmlx.image.families.h3_image import family as hi

HEADER = "h3_image_pruned_ud_q4_k_xl.header.json"
GiB = 1 << 30
# The int8 weight and float32 scale a row of the 200 converted matrices,
# less their K-quant bytes in the release file.
DELTA = 19_279_769_600 - 12_942_698_496
# fc1 at float32, its int8 codes and its scales: the largest conversion.
TRANSIENT = 5 * 28672 * 5376 + 4 * 28672


def _kernels(monkeypatch, nax: bool) -> None:
    """An mlx-kquant with the int8 ops, on a GPU with NAX units or
    without, so a test does not depend on the machine's kq."""
    for name in ("sdpa_qk8_inputs", "sdpa_qk8", "int8_quantize_rows",
                 "int8_matmul"):
        monkeypatch.setattr(kq, name, getattr(kq, name, lambda *a, **k: None),
                            raising=False)
    monkeypatch.setattr(kq, "nax_available", lambda: nax)


def test_int8_linears_convert_the_four_linears_of_each_block():
    """H3 video's block linears less its VSA gate, which this DiT lacks:
    qkv, out, fc1 and fc2 of each of the 50 blocks, and the bytes the int8
    DiT holds over the file and while one matrix converts, exact from the
    header."""
    view = fixture_view(HEADER)
    fam = hi.FAMILY
    assert fam.int8_attention
    got = neutral.int8_matrices(view, fam.strip_prefixes, fam.int8_linears)
    assert Counter((n, k) for n, k, _ in got) == {
        (21504, 5376): 50, (5376, 7168): 50, (28672, 5376): 50,
        (5376, 14336): 50}
    assert sum(b for *_, b in got) == 12_942_698_496
    assert neutral.int8_delta(view, fam.strip_prefixes,
                              fam.int8_linears) == DELTA
    assert neutral.int8_transient(view, fam.strip_prefixes,
                                  fam.int8_linears) == TRANSIENT
    held = {v: memfit.int8_held(view, fam, v) for v in neutral.INT8}
    assert held == {"off": 0, "attention": 0, "linears": DELTA,
                    "all": DELTA}


@pytest.fixture
def gated(monkeypatch):
    """``load_dit`` on a header, stopped at its load gate, which records
    the bytes it counts and its hint."""
    import gmlx.load.preflight as pre

    class Gated(Exception):
        pass

    seen = []

    def gate(weight_bytes, model_id, *, what, hint):
        seen.append((weight_bytes, hint))
        raise Gated

    monkeypatch.setattr(pre, "find_split_shards", lambda path: [path])
    monkeypatch.setattr(loader, "_scan_shard", lambda s: SimpleNamespace(
        truncated=False, kv={}, skipped={}))
    monkeypatch.setattr(loader, "gate_load", gate)

    def go(int8: str, header: str = HEADER, variant: str = "pruned"):
        view = fixture_view(header)
        monkeypatch.setattr(loader, "view_from_scans", lambda scans: view)
        ident = ifp.identify(view, variant=variant)
        n = len(seen)
        with pytest.raises(Gated):
            loader.load_dit("/d.gguf", identified=ident, int8=int8,
                            log=lambda *a: None)
        assert len(seen) == n + 1
        return seen[-1]
    go.seen = seen
    return go


def test_the_load_gate_counts_the_int8_dit_and_its_largest_conversion(
        gated, monkeypatch):
    _kernels(monkeypatch, nax=True)
    view = fixture_view(HEADER)
    dense = memfit.file_bytes(view) + memfit.cast_bytes(view, "bfloat16")
    plain = (dense, "Try a smaller DiT quant")
    assert gated("off") == plain
    assert gated("attention") == plain
    wide = (dense + DELTA + TRANSIENT,
            "Run with int8 off or attention, or try a smaller DiT quant")
    assert gated("linears") == wide
    assert gated("all") == wide


@pytest.mark.parametrize("int8,part", [("attention", "attention"),
                                       ("linears", "linears"),
                                       ("all", "attention")])
def test_a_gpu_without_nax_is_refused_before_the_gate(gated, monkeypatch,
                                                      int8, part):
    _kernels(monkeypatch, nax=False)
    with pytest.raises(ifp.ImageModelError) as e:
        gated(int8)
    assert str(e.value) == (
        f"MiniMax-H3-Image pruned runs with int8 {part}, whose kernel needs "
        "a GPU with NAX units (M5 or newer), and this Mac's GPU has none")
    assert gated.seen == []


@pytest.mark.parametrize("int8,why", [
    ("attention", "gmlx runs Z-Image with dense attention only"),
    ("all", "gmlx runs Z-Image with dense attention only"),
    ("linears", "gmlx runs Z-Image with its file's linears only")])
def test_a_family_without_int8_refuses_each_part(gated, monkeypatch, int8,
                                                 why):
    _kernels(monkeypatch, nax=True)
    with pytest.raises(ifp.ImageModelError) as e:
        gated(int8, Z_HEADER, "turbo")
    assert str(e.value) == why
    assert gated.seen == []
    gated("off", Z_HEADER, "turbo")


def test_an_int8_step_holds_its_int8_rows_beside_a_dense_one():
    """Each row of a step holds the int8 q and k of 56 heads of 128 and a
    float32 scale a head with int8 attention, and the int8 copy of fc2's
    14336 wide input with a float32 scale per 1024 columns with int8
    linears, as H3 video prices them; no other term moves."""
    row = memfit.PEAKS["h3_image"]
    kw = dict(family="h3_image", height=1024, width=1024,
              budget=100 * GiB, overcommit=False)
    dense = memfit.plan_memory(13 * GiB, 18 * GiB, GiB, **kw)
    rows = dense.rows * memfit.image_rows(row, 1024, 1024, None)
    q_k, fc2 = 2 * 56 * 128 + 2 * 56 * 4, 14336 + 14 * 4
    for attention, linears, more in (("int8", "file", q_k),
                                     ("dense", "int8", fc2),
                                     ("int8", "int8", q_k + fc2)):
        got = memfit.plan_memory(13 * GiB, 18 * GiB, GiB, attention=attention,
                                 linears=linears, **kw)
        assert got.terms["step_peak"] - dense.terms["step_peak"] == \
            rows * more
        assert {k: v for k, v in got.terms.items() if k != "step_peak"} == \
            {k: v for k, v in dense.terms.items() if k != "step_peak"}


def _quantize_rows(x, *, group_size=None, stream=None):
    # Symmetric codes a row or a row's group, as mlx-kquant's op.
    xf = x.astype(mx.float32)
    k = xf.shape[-1]
    g = k if group_size is None or group_size >= k else group_size
    xg = xf.reshape(*xf.shape[:-1], k // g, g)
    s = mx.max(mx.abs(xg), axis=-1, keepdims=True) / 127
    s = mx.where(s > 0, s, mx.array(1 / 127, mx.float32))
    codes = mx.clip(mx.round(xg / s), -127, 127).astype(mx.int8)
    return codes.reshape(xf.shape), s.reshape(*xf.shape[:-1], k // g)


def _matmul(x8, x_scales, w8, w_scales, *, group_size=None,
            out_dtype=mx.bfloat16, stream=None):
    k = x8.shape[-1]
    g = k if group_size is None else group_size
    x = (x8.astype(mx.float32).reshape(*x8.shape[:-1], k // g, g)
         * x_scales[..., None]).reshape(x8.shape)
    return ((x @ w8.astype(mx.float32).T) * w_scales).astype(out_dtype)


# The block linears the tiny DiT converts: its fc2 takes 48 columns, off
# int8's multiple of 32, which the release's 14336 is on.
_TINY_INT8 = {k: v for k, v in hi.INT8_LINEARS.items()
              if not k.endswith("fc2")}


@pytest.mark.parametrize("placement", ["block", "split"])
def test_a_lora_runs_on_the_int8_base_as_its_product_and_the_stock_delta(
        monkeypatch, tmp_path, placement):
    """A LoRA in diffusers' names on the tiny DiT with int8 linears: each
    converted block linear keeps its Int8Linear base, its call is the
    base's product plus the table's delta bit for bit, and that delta is
    the file's, q, k and v on thirds of qkv_proj and fc1's halves swapped
    back. The forms are own or block, never the low-rank kernel or a
    split, which need a K-quant or float base, under either placement."""
    import h3_image_gates as hg

    from gmlx.image import api, lora
    from gmlx.image.lora import install
    from gmlx.image.ops.int8_linear import Int8Linear

    monkeypatch.setattr(kq, "int8_quantize_rows", _quantize_rows,
                        raising=False)
    monkeypatch.setattr(kq, "int8_matmul", _matmul, raising=False)
    monkeypatch.setattr(install, "_PLACEMENT", placement)
    weights, wm = committed_fixture("selftest_h3_image_tiny_weights")
    weights = {k: np.asarray(v, np.float32) for k, v in weights.items()}
    strength = 0.8
    with mx.stream(mx.cpu):
        pipe = api.ImagePipeline(
            dit=hg.tiny_model(weights, wm["inputs"]["config"]),
            family=dataclasses.replace(hi.FAMILY, min_pixels=None),
            variant="pruned")
        paths = sorted(p for p, _ in pipe.dit.named_modules()
                       if any(p.endswith(s) for s in (
                           "attn.qkv_proj", "attn.out_proj", "mlp.fc1"))
                       and p.startswith("blocks."))
        assert pipe.dit.convert_linears(_TINY_INT8) == len(paths) == 6
        rng = np.random.default_rng(3)
        mods = dict(pipe.dit.named_modules())
        adapter = {}
        for p in paths:
            out, inp = mods[p].weight.shape
            adapter[p] = (
                (rng.standard_normal((4, inp)) * 0.2).astype(np.float32),
                (rng.standard_normal((out, 4)) * 0.2).astype(np.float32),
                2.0)
        path = write_keys(tmp_path / "l.safetensors",
                          h3_diffusers_keys(adapter))
        ad = lora.load_adapter(pipe, path)
        tid = api.define_patch_table(pipe,
                                     lora.table_entries([(ad, strength)]))
        prev = install.activate(pipe, tid)
        try:
            mods = dict(pipe.dit.named_modules())
            for p in paths:
                leaf = mods[p]
                assert isinstance(leaf, install.ImageLoRALinear), p
                assert isinstance(leaf.base, Int8Linear), p
                down, up, alpha = adapter[p]
                x = mx.array(rng.standard_normal((1, 70, down.shape[1]))
                             .astype(np.float32)).astype(mx.bfloat16)
                f = leaf._state.factors(p, x.dtype)
                assert isinstance(f, (install._Own, install._Block)), (p, f)
                if isinstance(f, install._Own):
                    z = x @ f.down.T
                    z = (z.astype(mx.float32) * f.scale).astype(z.dtype)
                    want = install._add(leaf.base(x), z, f.up.T)
                else:
                    want = install._add(leaf.base(x), x @ f.a_t, f.b_t)
                assert mx.array_equal(leaf(x), want), p
                assert not mx.array_equal(leaf(x), leaf.base(x)), p
                # The delta of the float32 factors is the file's.
                f = leaf._state.factors(p, mx.float32)
                if isinstance(f, install._Own):
                    delta = f.scale * (f.up @ f.down)
                else:
                    delta = (f.a_t @ f.b_t).T
                ref = reference_delta([(down, up, alpha, None, strength)],
                                      (up.shape[0], down.shape[1]))
                err = np.abs(np.array(delta, np.float64) - ref).max()
                assert err <= 1e-6 * np.abs(ref).max(), (p, err)
        finally:
            install.restore(pipe, prev)

