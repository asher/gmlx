"""FastH3's video sparse attention: the tile map of gmlx's packed layout
against FastVideo's index arrays at three FastH3 shapes, the refusal of
a video anchor, and on a NAX GPU the call in a tiny FastH3 block and the
test hook that pins and reads back its routing. The tile map and the
refusal are CPU-only."""
from __future__ import annotations

import json
import math
import os

import mlx.core as mx
import mlx_kquant as kq
import numpy as np
import pytest
from mlx.utils import tree_flatten
from video_fixtures import FASTH3_FV_GEOMETRY, FIXTURES, load_json

from gmlx.video import layout
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


def test_a_video_anchor_is_refused():
    """FastH3 is text to video: a video anchor would be a second video
    region, which its attention does not run."""
    lay = _layout(512, 512, 22, anchors=(
        layout.AnchorRows(frame=0, video_t=1, audio_t=2),))
    with pytest.raises(ValueError) as ei:
        vsa.segments(lay, (1, 2, 2))
    assert str(ei.value) == (
        "FastH3's video sparse attention runs over one video region, the "
        "target video, and this layout also has 1 video anchor segment")


# On a NAX GPU: a tiny FastH3 block, head dim 128 as the kernel takes.

nax_only = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available()
    or getattr(kq, "vsa_attention", None) is None
    or not getattr(kq, "nax_available", lambda: False)(),
    reason="the sparse kernel runs on a NAX GPU")

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
def test_the_pin_hook_reads_back_and_replays_the_routing(monkeypatch):
    """A traced forward records each call's block indices and float32
    pooled q and k means and runs the shipped call's bits. Pinned to
    those indices, a forward gives the same velocity; pinned to the other
    tiles, another one."""
    model = _tiny(0.8)
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
        # The GPU's float32 matmul may run at TF32.
        np.testing.assert_allclose(
            c.scores, c.q_pooled @ c.k_pooled.transpose(0, 2, 1)
            / math.sqrt(128), rtol=1e-2, atol=1e-3)
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

