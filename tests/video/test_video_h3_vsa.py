"""FastH3's video sparse attention: the tile map of gmlx's packed layout
against FastVideo's index arrays at three FastH3 shapes, the refusals,
the views a sparse block hands the call, and on a NAX GPU the call in a
tiny FastH3 block, its pause points and the test hook that pins and reads
back its routing. The tile map, the refusals and the views need no
kernel."""
from __future__ import annotations

import json
import math
import os
import tempfile

import mlx.core as mx
import mlx_kquant as kq
import numpy as np
import pytest
from mlx.utils import tree_flatten
from video_fixtures import (FASTH3_FV_GEOMETRY, FIXTURES, H3_IDENTITY_FASTH3,
                            fasth3_view, load_json)

from gmlx.image import pacing
from gmlx.image.float32 import f32_products_exact, split
from gmlx.video import layout, rope
from gmlx.video import fingerprint as vfp
from gmlx.video import loader as vloader
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import model as hm
from gmlx.video.families.h3 import vsa

_SHAPES = {"512x512x22": (512, 512, 22), "768x768x39": (768, 768, 39),
           "1344x768x124": (1344, 768, 124)}


def _layout(width: int, height: int, frames: int, text: int = 56,
            anchors: tuple = ()) -> layout.Layout:
    vs, aus = hf.latent_shapes(frames, width, height)
    return layout.pack(hf.TIMELINE, text, *vs[2:], aus[3], anchors)


def _fv() -> tuple:
    z = np.load(FIXTURES / FASTH3_FV_GEOMETRY)
    return z, json.loads(bytes(z["meta"]).decode())


def test_the_fixture_is_fastvideos_revision():
    manifest = load_json("manifest.json")[FASTH3_FV_GEOMETRY]
    _, meta = _fv()
    assert meta["fastvideo_revision"] == manifest["revision"]
    assert meta["tile_size"] == vsa.TILE


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_the_tile_map_is_fastvideos(shape):
    """gmlx's layout gives FastVideo's segments: the dense text and audio
    rows, then the video grid (latent frames, H / 2, W / 2). Their tile
    map is FastVideo's, array for array."""
    z, meta = _fv()
    want = meta["shapes"][shape]
    lay = _layout(*_SHAPES[shape])
    segs = vsa.segments(lay, hf.CONSTANTS.values["patch_size"])
    assert [list(s) if isinstance(s, tuple) else s
            for s in segs] == want["segments"]
    got = vsa.tile_map(segs)
    for name in ("tile_gather_index", "prefix_gather_index", "untile_index",
                 "variable_block_sizes"):
        a, b = getattr(got, name), z[f"{shape}/{name}"]
        assert a.dtype == np.int32 and np.array_equal(a, b), name
    assert (got.num_prefix_tiles, got.prefix_length) == (
        want["num_prefix_tiles"], want["prefix_length"])
    assert [list(s) for s in got.video_tile_spans] == want["video_tile_spans"]
    assert got.untile_index.size == lay.rows == want["total_seq_length"]


# A keyframe with its audio, and an audio anchor alone.
_ANCHORS = {"keyframe": layout.AnchorRows(frame=0, video_t=1, audio_t=2),
            "audio": layout.AnchorRows(frame=0, audio_t=5)}
_TEXT_TO_VIDEO = ("MiniMax-H3 fasth3 runs text to video only, and this "
                  "request has 1 anchor; run it without keyframes, guides "
                  "or audio anchors")


@pytest.mark.parametrize("anchor", sorted(_ANCHORS))
def test_a_layout_with_an_anchor_does_not_reach_the_tile_map(anchor):
    """FastH3 runs text to video only, so no anchor rows, video or
    audio, are tiled."""
    lay = _layout(512, 512, 22, anchors=(_ANCHORS[anchor],))
    with pytest.raises(ValueError) as ei:
        vsa.segments(lay, (1, 2, 2))
    assert str(ei.value) == _TEXT_TO_VIDEO


def test_a_layout_with_vision_rows_does_not_reach_the_tile_map():
    vs, aus = hf.latent_shapes(22, 512, 512)
    lay = layout.pack(hf.TIMELINE, 56, *vs[2:], aus[3], spans=((7, 30),))
    with pytest.raises(ValueError, match="fasth3 runs text to video only, "
                                         "and this prompt holds images"):
        vsa.segments(lay, (1, 2, 2))


@pytest.fixture
def fasth3_load(monkeypatch):
    """``load_dit`` on a file that identifies as FastH3, which records
    whether the weights load."""
    from gmlx.image import loader as iloader

    row = load_json(H3_IDENTITY_FASTH3)["files"][
        "fastvideo_fasth3_8step_v2_pruned-Q4_K_M.gguf"]
    monkeypatch.setattr(vfp, "weight_identity",
                        lambda view, names: row["variant_hash"])
    monkeypatch.setattr(vfp, "layout_identity",
                        lambda view, fam: row["layout_hash"])
    monkeypatch.setattr(vloader, "identify_dit",
                        lambda path, **kw: vfp.identify(fasth3_view(), **kw))
    loads = []

    def load(*a, **k):
        loads.append(a)
        return object(), None

    monkeypatch.setattr(iloader, "load_dit", load)

    def go():
        vloader.load_dit("fasth3.gguf", log=lambda *x: None)
    go.loads = loads
    return go


def test_a_kquant_without_vsa_attention_is_refused(fasth3_load, monkeypatch):
    monkeypatch.delattr(kq, "vsa_attention", raising=False)
    with pytest.raises(vfp.VideoModelError) as ei:
        fasth3_load()
    assert str(ei.value) == (
        f"MiniMax-H3 fasth3 runs video sparse attention, and the installed "
        f"mlx-kquant {kq.__version__} has no vsa_attention; update "
        f"mlx-kquant to a release that has it")
    assert fasth3_load.loads == []


def test_a_gpu_without_nax_is_refused(fasth3_load, monkeypatch):
    monkeypatch.setattr(kq, "vsa_attention", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(kq, "nax_available", lambda: False)
    with pytest.raises(vfp.VideoModelError) as ei:
        fasth3_load()
    assert str(ei.value) == (
        "MiniMax-H3 fasth3 runs video sparse attention, whose kernel needs "
        "a GPU with NAX units (M5 or newer), and this Mac's GPU has none")
    assert fasth3_load.loads == []
    monkeypatch.setattr(kq, "nax_available", lambda: True)
    fasth3_load()
    assert len(fasth3_load.loads) == 1


# A tiny FastH3 model, head dim 128 as the kernel takes.

TINY = {"num_attention_heads": 2, "attention_head_dim": 128,
        "hidden_size": 64, "num_layers": 2, "num_refiner_layers": 1,
        "ffn_dim": 48, "in_channels": 24, "audio_in_channels": 32,
        "patch_size": [1, 2, 2], "text_dim": 40, "rope_freq_dim": 16,
        "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5,
        "adaln_grid": 1025, "adaln_basis": 8}
# 22 frames at 256x256: 8 text rows, 74 audio rows and a (6, 8, 8) video
# grid, so 3 dense tiles and 8 video tiles, of which each video query
# tile keeps 2 at sparsity 0.8.
_CLIP = (256, 256, 22)
# Gate weights this many times the others' scale, which makes the gate
# term about 0.7 of the attention rows.
GATE_SCALE = 4.0


def _tiny(sparsity: float) -> hm.H3Transformer:
    """Random weights in bfloat16, but for the family's float32 ones, as
    the loader leaves them."""
    mx.random.seed(0)
    m = hm.H3Transformer({**TINY, "vsa_sparsity": sparsity})
    w = {k: 0.05 * mx.random.normal(v.shape)
         for k, v in tree_flatten(m.parameters())}
    for k in w:
        if k.endswith(".weight") and "norm" in k.split(".")[-2]:
            w[k] = 1 + w[k]
        elif k.endswith("to_gate_compress.weight"):
            w[k] = GATE_SCALE * w[k]
        if not any(p in k for p in hf.FAMILY.fp32_keep):
            w[k] = w[k].astype(mx.bfloat16)
    w["rope.inv_freq"] = mx.array(1 / 10000 ** (np.arange(16) / 16),
                                  mx.float32)
    m.load_weights(list(w.items()))
    return m


def _gathers(a: mx.array) -> bool:
    """Whether the graph of ``a`` holds a gather."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.dot")
        mx.export_to_dot(path, a)
        with open(path) as f:
            return '"Gather"' in f.read()


def test_a_sparse_block_hands_the_call_views_not_gathers(monkeypatch):
    """q, k and v reach vsa_attention as views of the block's products,
    with no gather in their graphs. MLX's GPU gather misreads a view that
    spans more than INT32_MAX elements of its buffer, as v of the fused
    qkv does from 99,866 rows (docs/upstream-defects.md)."""
    model = _tiny(hf.VSA_SPARSITY)
    lay = _layout(*_CLIP, text=8)
    cos, sin = rope.tables(lay.positions, model.rope.inv_freq,
                           TINY["attention_head_dim"])
    h = mx.random.normal((1, lay.rows, TINY["hidden_size"])).astype(
        mx.bfloat16)
    # Evaluated inputs leave only the block's own ops in the graphs.
    mx.eval(h, cos, sin, model.parameters())
    seen = {}

    def attend(q, k, v, gate, geom, sparsity):
        seen.update(q=q, k=k, v=v)
        return mx.zeros(q.shape, q.dtype)

    monkeypatch.setattr(vsa, "attend", attend)
    model.blocks[0].attention(h, cos, sin, vsa=object())
    assert {n: _gathers(a) for n, a in seen.items()} == {
        "q": False, "k": False, "v": False}


# On a NAX GPU: the call in a tiny FastH3 block.


def _no_kernel() -> str | None:
    """Why the sparse kernel does not run here, or None."""
    if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
        return "the sparse kernel runs on the GPU"
    if getattr(kq, "vsa_attention", None) is None:
        return "mlx-kquant has no vsa_attention"
    if not kq.nax_available():
        return "the sparse kernel needs a GPU with NAX units"
    return None


_NO_KERNEL = _no_kernel()
nax_only = pytest.mark.skipif(_NO_KERNEL is not None,
                              reason=_NO_KERNEL or "")


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _tile_means(rows: np.ndarray, m: vsa.TileMap) -> np.ndarray:
    """float32 [heads, tiles, D] means of each tile's packed rows."""
    tile = m.untile_index // vsa.TILE
    out = np.zeros((m.variable_block_sizes.size,) + rows.shape[1:])
    np.add.at(out, tile, rows.astype(np.float64))
    out /= m.variable_block_sizes[:, None, None]
    return out.transpose(1, 0, 2)


@nax_only
def test_keeping_every_tile_is_dense_attention_plus_the_gate_term(
        monkeypatch):
    """At a sparsity that keeps every tile, a block's sparse attention is
    its dense attention plus the gate term: each row's gate of ``h``
    times the softmax of its tile's pooled scores over the pooled v."""
    model = _tiny(0.01)
    lay = _layout(*_CLIP, text=8)
    plan = model.plan(lay, 0.25, 0.25)
    blk = model.blocks[0]
    h = mx.random.normal((1, lay.rows, TINY["hidden_size"])).astype(
        mx.bfloat16)
    seen = {}
    stock = vsa.attend

    def attend(q, k, v, gate, geom, sparsity):
        seen.update(q=q, k=k, v=v, gate=gate)
        return stock(q, k, v, gate, geom, sparsity)

    monkeypatch.setattr(vsa, "attend", attend)
    got = blk.attention(h, plan.cos, plan.sin, vsa=plan.vsa)
    dense = blk.attention(h, plan.cos, plan.sin)
    m = vsa.tile_map(vsa.segments(lay, model.cfg.patch))
    q, k, v = (np.array(seen[x].astype(mx.float32)) for x in "qkv")
    gate = np.array(blk.attn.to_gate_compress(h)[0].astype(mx.float32))
    np.testing.assert_array_equal(
        gate.reshape(seen["gate"].shape), np.array(seen["gate"].astype(
            mx.float32)))
    qp, kp, vp = (_tile_means(x, m) for x in (q, k, v))
    scores = qp @ kp.transpose(0, 2, 1) / math.sqrt(q.shape[-1])
    p = np.exp(scores - scores.max(-1, keepdims=True))
    coarse = (p / p.sum(-1, keepdims=True)) @ vp
    tile = m.untile_index // vsa.TILE
    term = coarse.transpose(1, 0, 2)[tile] * gate.reshape(q.shape)
    term = term.reshape(1, lay.rows, -1)
    dense = np.array(dense.astype(mx.float32))
    assert np.linalg.norm(term) > 0.5 * np.linalg.norm(dense)
    assert _rel(np.array(got.astype(mx.float32)), dense + term) < 0.01


@nax_only
def test_a_sparse_block_pauses_after_qkv_and_after_the_call(monkeypatch):
    """A paced forward evaluates q, k, v and the gate before RoPE, then
    the sparse call's own output, so the call's transients are gone
    before out_proj and the feed-forward run."""
    model = _tiny(hf.VSA_SPARSITY)
    lay = _layout(*_CLIP, text=8)
    plan = model.plan(lay, 0.25, 0.6)
    h = mx.random.normal((1, lay.rows, TINY["hidden_size"])).astype(
        mx.bfloat16)
    calls, seen = [], []
    stock = vsa.attend

    def attend(*a):
        calls.append(stock(*a))
        return calls[-1]

    monkeypatch.setattr(vsa, "attend", attend)
    with pacing.paced(seen.append):
        model.blocks[0].attention(h, plan.cos, plan.sin, vsa=plan.vsa)
    assert [len(s) for s in seen] == [4, 1]
    assert seen[1][0] is calls[0]


@nax_only
def test_the_pin_hook_reads_back_and_replays_the_routing(monkeypatch):
    """A traced forward records each call's block indices and float32
    pooled q and k means and runs the shipped call's bits. Pinned to
    those indices, a forward gives the same velocity; pinned to the other
    tiles, another one."""
    model = _tiny(hf.VSA_SPARSITY)
    lay = _layout(*_CLIP, text=8)
    m = vsa.tile_map(vsa.segments(lay, model.cfg.patch))
    vs, aus = hf.latent_shapes(_CLIP[2], *_CLIP[:2])
    xv, xa = mx.random.normal(vs), mx.random.normal(aus)
    text = model.caption(mx.random.normal((1, 8, 40)))
    seen = []
    stock = vsa.attend

    def attend(q, k, *rest):
        seen.append((np.array(q.astype(mx.float32)),
                     np.array(k.astype(mx.float32))))
        return stock(q, k, *rest)

    monkeypatch.setattr(vsa, "attend", attend)

    def run():
        return model(xv, xa, 0.25, 0.25, text, lay)

    shipped = run()
    with vsa.traced() as free:
        own = run()
    assert len(free.calls) == TINY["num_layers"]
    # 3 dense tiles, then 8 video tiles of which each query tile keeps 2.
    assert (m.num_prefix_tiles, m.variable_block_sizes.size) == (3, 11)
    for c, (q, k) in zip(free.calls, seen[TINY["num_layers"]:]):
        assert (c.pinned, c.shipped) == (False, True)
        assert c.block_idx.shape == (2, 8, 3 + 2)
        assert c.q_pooled.dtype == c.k_pooled.dtype == np.float32
        np.testing.assert_allclose(c.q_pooled, _tile_means(q, m), rtol=1e-5,
                                   atol=1e-6)
        np.testing.assert_allclose(c.k_pooled, _tile_means(k, m), rtol=1e-5,
                                   atol=1e-6)
        # float32 products on every device, TF32 or not.
        np.testing.assert_allclose(
            c.scores, c.q_pooled @ c.k_pooled.transpose(0, 2, 1)
            / math.sqrt(128), rtol=1e-5, atol=1e-6)
    for a, b in zip(shipped, own):
        assert mx.array_equal(a, b)
    pins = [c.block_idx for c in free.calls]
    with vsa.traced(lambda n: pins[n]) as pinned:
        same = run()
    assert [c.pinned for c in pinned.calls] == [True, True]
    for a, b in zip(same, shipped):
        assert mx.array_equal(a, b)

    def other(n):
        idx = pins[n].copy()
        for h in range(idx.shape[0]):
            for t in range(idx.shape[1]):
                rest = sorted(set(range(3, 11)) - set(idx[h, t, 3:]))
                idx[h, t, 3:] = rest[:2]
        return idx

    with vsa.traced(other):
        moved = run()
    assert not mx.array_equal(moved[0], shipped[0])


@nax_only
def test_the_hook_keeps_the_shipped_bits_with_float32_scores(monkeypatch):
    """On a GPU that multiplies float32 at TF32, mlx-kquant scores the
    routing on split operands: qh @ kh.T + (qh @ kl.T + ql @ kh.T), k
    transposed after its split. The hook records those scores bit for bit
    and its own pieces give the shipped call's bits with them, and not
    with the direct product."""
    if f32_products_exact():
        pytest.skip("this GPU multiplies float32 at float32")
    lay = _layout(*_CLIP, text=8)
    geom = vsa.geometry(vsa.segments(lay, hf.CONSTANTS.values["patch_size"]))
    q, k, v, gate = (mx.random.normal((lay.rows, 2, 128),
                                      key=mx.random.key(i)).astype(
                                          mx.bfloat16) for i in range(4))

    def call():
        with vsa.traced() as trace:
            vsa.attend(q, k, v, gate, geom, hf.VSA_SPARSITY)
        return trace.calls[0]

    c = call()
    assert c.shipped is True
    qh, ql = split(mx.array(c.q_pooled))
    kh, kl = (x.transpose(0, 2, 1) for x in split(mx.array(c.k_pooled)))
    want = (qh @ kh + (qh @ kl + ql @ kh)) / math.sqrt(128)
    np.testing.assert_array_equal(c.scores, np.array(want))
    monkeypatch.setattr(vsa, "f32_products_exact", lambda: True)
    assert call().shipped is False
