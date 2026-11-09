"""The H3 model's own conventions on a tiny model: patch order, stereo
packing, the curve table blend, which modulation row each segment reads,
and the sign of the velocity. CPU-only."""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from gmlx.image import ops, pacing
from gmlx.video import api, layout, schedule
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import model as hm
from video_fixtures import ROW_REGIME_IDS, ROW_REGIMES

TINY = {"num_attention_heads": 2, "attention_head_dim": 128,
        "hidden_size": 64, "num_layers": 2, "num_refiner_layers": 1,
        "ffn_dim": 48, "in_channels": 24, "audio_in_channels": 32,
        "patch_size": [1, 2, 2], "text_dim": 40, "rope_freq_dim": 16,
        "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5,
        "adaln_grid": 1025, "adaln_basis": 8}


def tiny_weights() -> dict:
    """Random weights for the tiny model, with norm weights near one and
    each block's adaLN bias at unit scale, so every block moves the
    velocity."""
    mx.random.seed(0)
    m = hm.H3Transformer(TINY)
    out = {k: 0.05 * mx.random.normal(v.shape)
           for k, v in tree_flatten(m.parameters())}
    for k in out:
        if k.endswith(".weight") and "norm" in k.split(".")[-2]:
            out[k] = 1 + out[k]
        elif k.startswith("blocks.") and k.endswith("adaln_proj.linear.bias"):
            out[k] = 20 * out[k]
    out["rope.inv_freq"] = mx.array(1 / 10000 ** (np.arange(16) / 16),
                                    mx.float32)
    return out


@pytest.fixture(scope="module")
def tiny():
    m = hm.H3Transformer(TINY)
    m.load_weights(list(tiny_weights().items()))
    m.act_dtype = mx.float32
    return m


def test_patch_rows_are_time_height_width_with_channel_major_features():
    lat = mx.arange(24 * 2 * 4 * 6).reshape(1, 24, 2, 4, 6)
    rows = hm.patchify(lat, (1, 2, 2))
    assert rows.shape == (2 * 2 * 3, 96)
    a = np.array(lat)[0]
    # Row (t=1, h=0, w=2), feature (c=5, ph=1, pw=0).
    assert int(rows[1 * 6 + 0 * 3 + 2, 5 * 4 + 1 * 2 + 0]) == a[5, 1, 1, 4]
    back = hm.unpatchify(rows, (24, 2, 4, 6), (1, 2, 2))
    assert mx.array_equal(back, lat)


def test_stereo_audio_packs_all_of_channel_0_then_channel_1():
    lat = mx.arange(32 * 2 * 5).reshape(1, 32, 2, 5)
    rows = hm.pack_audio(lat)
    assert rows.shape == (10, 32)
    a = np.array(lat)[0]
    np.testing.assert_array_equal(np.array(rows[3]), a[:, 0, 3])
    np.testing.assert_array_equal(np.array(rows[5 + 3]), a[:, 1, 3])
    assert mx.array_equal(hm.unpack_audio(rows), lat)


def test_the_curve_blends_the_two_rows_around_t(tiny):
    tab = np.array(tiny.adaln_t_table)
    got = np.array(tiny.curve([0.0, 1.0, 0.5, 0.25 + 0.3 / 1024, -1.0]))
    np.testing.assert_array_equal(got[0], tab[0])
    np.testing.assert_array_equal(got[1], tab[1024])
    np.testing.assert_array_equal(got[2], tab[512])
    w = np.float32(0.25 + 0.3 / 1024) * np.float32(1024) - np.float32(256)
    np.testing.assert_allclose(got[3], tab[256] + w * (tab[257] - tab[256]),
                               rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(got[4], tab[0])


def test_each_segment_reads_its_tag_at_its_time(tiny):
    lay = layout.pack(hf.TIMELINE, 8, 2, 4, 4, 8,
                      (layout.AnchorRows(0, video_t=1, audio_t=2),))
    plan = tiny.plan(lay, 0.25, 0.6)
    assert plan.times == (0.25, 0.6, 0.999, 1.0)
    tv, ta, cv, ca = range(4)
    v, t, a = hf.TAGS["video"], hf.TAGS["text"], hf.TAGS["audio"]
    assert [r for _, _, r in plan.segs] == [3 * tv + t, 3 * cv + v,
                                            3 * ca + a, 3 * ta + a,
                                            3 * tv + v]
    assert plan.heads == {"video": tv, "audio": ta}


def test_the_modulation_row_of_a_tag_is_its_slice_of_the_linear(tiny):
    blk = tiny.blocks[0]
    t_emb = tiny.curve([0.25, 0.6])
    full = np.array(blk.adaln_proj.linear(t_emb))
    mod = blk.modulation(t_emb, mx.float32)
    d = TINY["hidden_size"]
    for i in range(2):
        for tag in range(3):
            for k in range(6):
                np.testing.assert_array_equal(
                    np.array(mod[k][3 * i + tag]),
                    full[i, (tag * 6 + k) * d:(tag * 6 + k + 1) * d])


def test_the_velocity_is_the_negated_head(tiny):
    vs, aus = hf.latent_shapes(5, 64, 64)
    lay = layout.pack(hf.TIMELINE, 8, *vs[2:], aus[3])
    v, a = mx.random.normal(vs), mx.random.normal(aus)
    text = tiny.caption(mx.random.normal((1, 8, 40)))
    taps: dict = {}
    vv, va = tiny(v, a, 0.25, 0.6, text, lay, taps=taps)
    assert vv.shape == vs and va.shape == aus and vv.dtype == mx.float32
    assert mx.array_equal(-hm.pack_audio(va), taps["final_layer.audio_out"])
    assert mx.array_equal(-hm.patchify(vv, (1, 2, 2)),
                          taps["final_layer.video_out"])


def test_the_segment_order_does_not_change_the_velocity(tiny):
    """Attention sees every row, and each row keeps its position and
    modulation, so a family that orders the segments otherwise gets the
    same velocity, anchors included."""
    vs, aus = hf.latent_shapes(5, 64, 64)
    anchor = layout.AnchorRows(frame=4, video_t=1, audio_t=2)
    v, a = mx.random.normal(vs), mx.random.normal(aus)
    text = tiny.caption(mx.random.normal((1, 8, 40)))
    cond = {"cond_video": [mx.random.normal((4, 96))],
            "cond_audio": [mx.random.normal((4, 32))]}
    got = []
    # The CPU stream keeps the float32 products exact.
    with mx.stream(mx.cpu):
        for order in (hf.SEGMENT_ORDER, ("audio", "video", "anchors",
                                         "text")):
            tl = dataclasses.replace(hf.TIMELINE, order=order)
            lay = layout.pack(tl, 8, *vs[2:], aus[3], (anchor,))
            got.append(tiny(v, a, 0.25, 0.6, text, lay, **cond))
    for x, y in zip(*got):
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=1e-4,
                                   atol=1e-5)


TWO = (layout.AnchorRows(frame=0, video_t=2, audio_t=1),
       layout.AnchorRows(frame=4, video_t=1, audio_t=2))


def _two_anchors(tiny):
    vs, aus = hf.latent_shapes(5, 64, 64)
    lay = layout.pack(hf.TIMELINE, 8, *vs[2:], aus[3], TWO)
    v, a = mx.random.normal(vs), mx.random.normal(aus)
    text = tiny.caption(mx.random.normal((1, 8, 40)))
    cv = [mx.random.normal((8, 96)), mx.random.normal((4, 96))]
    ca = [mx.random.normal((2, 32)), mx.random.normal((4, 32))]
    return v, a, text, lay, cv, ca


def test_each_anchor_segment_holds_that_anchors_rows(tiny):
    v, a, text, lay, cv, ca = _two_anchors(tiny)
    x = tiny.embed(v, a, text, lay, cv, ca)
    want = {"cond": [tiny.video_patch_proj(c) for c in cv],
            "cond_audio": [tiny.audio_patch_proj(c) for c in ca]}
    seen = {"cond": 0, "cond_audio": 0}
    for s in lay.segments:
        if s.kind in seen:
            got = x[0, s.start:s.stop]
            np.testing.assert_allclose(np.array(got),
                                       np.array(want[s.kind][seen[s.kind]]),
                                       rtol=1e-6, atol=1e-6)
            seen[s.kind] += 1
    assert seen == {"cond": 2, "cond_audio": 2}


def test_anchor_rows_in_the_wrong_order_are_refused(tiny):
    v, a, text, lay, cv, ca = _two_anchors(tiny)
    with pytest.raises(ValueError, match="next cond segment has 8 rows, "
                                         "the inputs give 4 rows"):
        tiny.embed(v, a, text, lay, cv[::-1], ca)
    with pytest.raises(ValueError, match="next cond_audio segment has 4 "
                                         "rows, the inputs give none"):
        tiny.embed(v, a, text, lay, cv, ca[:1])
    with pytest.raises(ValueError, match=r"\['cond'\] segments the layout"):
        tiny.embed(v, a, text, lay, [*cv, cv[1]], ca)


def test_a_large_update_keeps_the_residual_under_it():
    """MiniMax-H3's block 0 writes about -2e4 into one channel of each
    text row over an input of about 40, where the bfloat16 step is 128.
    The gated residual adds in float32 and rounds once, as the
    reference's addcmul_ does, so each sum is the nearest bfloat16 to the
    true one. Rounding the product first puts a sum up to a step off."""
    rng = np.random.default_rng(0)
    x = mx.array(40 + rng.standard_normal((1, 24, 64)), mx.bfloat16)
    y = mx.array(-2e4 * (1 + 0.1 * rng.standard_normal((1, 24, 64))),
                 mx.bfloat16)
    gate = mx.array(rng.uniform(0.5, 1.0, (2, 64)), mx.bfloat16)
    segs = ((0, 8, 1), (8, 24, 0))
    got = np.array(hm.gated(x, gate, y, segs).astype(mx.float32),
                   np.float64)
    xf, yf, gf = (np.array(v.astype(mx.float32), np.float64)
                  for v in (x, y, gate))
    rows = gf[[1] * 8 + [0] * 16]
    truth = xf + rows[None] * yf    # exact: bfloat16 inputs, float64
    step = 2.0 ** (np.floor(np.log2(np.abs(got))) - 7)
    assert np.all(np.abs(got - truth) <= step / 2)


def test_chunked_attention_matches_one_call(tiny):
    vs, aus = hf.latent_shapes(22, 128, 128)
    lay = layout.pack(hf.TIMELINE, 8, *vs[2:], aus[3])
    v, a = mx.random.normal(vs), mx.random.normal(aus)
    text = tiny.caption(mx.random.normal((1, 8, 40)))
    one = tiny(v, a, 0.25, 0.6, text, lay, chunk=None)
    many = tiny(v, a, 0.25, 0.6, text, lay, chunk=40)
    for x, y in zip(one, many):
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=1e-4,
                                   atol=1e-5)


def test_the_stock_feed_forward_pauses_after_each_row_chunk(tiny,
                                                            monkeypatch):
    """Off the fused kernel, as under a live LoRA, the feed-forward runs
    ``MLP_ROWS`` rows a product with a pause point after each, so a
    paced forward holds one product at a time."""
    monkeypatch.setattr(hm, "MLP_ROWS", 16)
    blk = tiny.blocks[0]
    h = mx.random.normal((1, 40, TINY["hidden_size"]))
    seen = []

    def pacer(arrays):
        seen.append(tuple(arrays[0].shape))
        mx.eval(*arrays)

    with pacing.paced(pacer):
        got = blk.hidden(h)
    ffn = TINY["ffn_dim"]
    assert seen == [(1, 16, ffn), (1, 16, ffn), (1, 8, ffn)]
    a, b = mx.split(blk.mlp.fc1(h), 2, axis=-1)
    np.testing.assert_allclose(np.array(got), np.array(ops.swiglu(a, b)),
                               rtol=1e-6, atol=1e-7)


def test_a_block_pauses_after_qkv_before_rope(tiny):
    """A paced forward evaluates q, k and v before RoPE, so a LoRA
    delta's second qkv product is gone before RoPE's copies exist."""
    blk = tiny.blocks[0]
    vs, aus = hf.latent_shapes(22, 128, 128)
    lay = layout.pack(hf.TIMELINE, 8, *vs[2:], aus[3])
    plan = tiny.plan(lay, 0.25, 0.6)
    h = mx.random.normal((1, lay.rows, TINY["hidden_size"]))
    seen = []
    with pacing.paced(lambda arrays: seen.append(len(arrays))):
        blk.attention(h, plan.cos, plan.sin, chunk=None)
    assert seen == [3]


@pytest.mark.parametrize("given", [False, True])
def test_a_run_steps_each_stream_on_its_own_schedule(tiny, monkeypatch,
                                                     given):
    """Three forwards with a different velocity each: each stream moves by
    its own sigma drops, so another table or shift lands elsewhere, and
    each forward's plan, which the model reads in place of the times
    passed beside it, puts each head at its stream's time. Tables passed
    in, such as an oracle's, replace the shifts' tables."""
    vel = {"video": (1.0, 3.0, 7.0), "audio": (2.0, 5.0, 11.0)}
    seen = []

    def fake(self, video, audio, tv, ta, *a, plan, **k):
        i = len(seen)
        seen.append(tuple(plan.times[plan.heads[s]]
                          for s in ("video", "audio")))
        return (mx.full(video.shape, vel["video"][i]),
                mx.full(audio.shape, vel["audio"][i]))

    monkeypatch.setattr(hm.H3Transformer, "__call__", fake)
    vs, aus = hf.latent_shapes(5, 64, 64)
    lay = layout.pack(hf.TIMELINE, 8, *vs[2:], aus[3])
    shifts = {"video": 6.0, "audio": 2.0} if given else hf.SHIFTS
    sig = {k: schedule.sigmas(3, shifts[k]) for k in vel}
    got = api.sample(tiny, mx.zeros(vs), mx.zeros(aus),
                     mx.zeros((1, 8, 64)), lay, steps=3, shifts=hf.SHIFTS,
                     sigmas=sig if given else None)
    for x, k in zip(got, ("video", "audio")):
        s = sig[k]
        want = sum(float(s[i + 1] - s[i]) * vel[k][i] for i in range(3))
        np.testing.assert_allclose(np.array(x), want, rtol=1e-6)
    assert seen == [(api.stream_time(sig["video"][i]),
                     api.stream_time(sig["audio"][i])) for i in range(3)]
    assert seen[0] == (0.0, 0.0)


@pytest.mark.parametrize("size, frames, text, rows, every", ROW_REGIMES,
                         ids=ROW_REGIME_IDS)
def test_the_default_pacer_evaluates_on_the_clips_rows(size, frames, text,
                                                       rows, every):
    """Every point of a clip of 32768 rows or more evaluates, whatever its
    caption, and a smaller clip evaluates every ceil(32768 / rows)
    points."""
    vs, aus = hf.latent_shapes(frames, *size)
    lay = layout.pack(hf.TIMELINE, text, *vs[2:], aus[3])
    assert lay.rows == rows
    hit = []
    budget = api.RowBudget(lay.rows, evaluate=hit.append)
    for i in range(48):
        budget((i,))
    assert hit == list(range(every - 1, 48, every))


def test_a_run_counts_the_layouts_rows_and_starts_each_step_again(
        tiny, monkeypatch):
    """The default pacer of a run counts each point as the layout's rows
    and starts again after each forward, so each step evaluates at the
    same points."""
    vs, aus = hf.latent_shapes(5, 64, 64)
    lay = layout.pack(hf.TIMELINE, 8, *vs[2:], aus[3])
    monkeypatch.setattr(api, "EVAL_ROWS", 3 * lay.rows)
    hit, made = [], []
    real = api.RowBudget

    def budget(rows, threshold, on_point=None):
        made.append((rows, threshold))
        return real(rows, threshold, on_point=on_point,
                    evaluate=lambda *a: hit.append(int(a[0].item())))

    def fake(self, video, audio, *a, **k):
        for i in range(8):
            pacing.point(mx.array(i))
        return mx.zeros(video.shape), mx.zeros(audio.shape)

    monkeypatch.setattr(api, "RowBudget", budget)
    monkeypatch.setattr(hm.H3Transformer, "__call__", fake)
    api.sample(tiny, mx.zeros(vs), mx.zeros(aus), mx.zeros((1, 8, 64)), lay,
               steps=2, shifts=hf.SHIFTS)
    assert made == [(lay.rows, 3 * lay.rows)]
    assert hit == [2, 5, 2, 5]


def test_float16_is_refused(tiny):
    with pytest.raises(hm.H3DtypeError, match="bfloat16 or float32"):
        hm.H3Transformer(TINY).act_dtype = mx.float16


def test_vision_rows_run_at_the_video_time_with_the_video_tag(tiny):
    """ComfyUI's tag runs over the prefix (model.py:677-688): a vision
    block with its markers runs at the text's time, the video's, with the
    video tag; the text around it keeps the text tag."""
    lay = layout.pack(hf.TIMELINE, 20, 2, 4, 4, 8, spans=((6, 11),))
    plan = tiny.plan(lay, 0.25, 0.6)
    tv = plan.times.index(0.25)
    v, t = hf.TAGS["video"], hf.TAGS["text"]
    assert [(a, b, r) for a, b, r in plan.segs][:3] == [
        (0, 6, 3 * tv + t), (6, 11, 3 * tv + v), (11, 20, 3 * tv + t)]


def test_the_prefix_takes_its_rows_by_segment(tiny):
    vs, aus = hf.latent_shapes(5, 64, 64)
    v, a = mx.random.normal(vs), mx.random.normal(aus)
    text = mx.random.normal((1, 20, TINY["hidden_size"]))
    plain = layout.pack(hf.TIMELINE, 20, *vs[2:], aus[3])
    split = layout.pack(hf.TIMELINE, 20, *vs[2:], aus[3],
                        spans=((6, 11), (14, 20)))
    assert mx.array_equal(tiny.embed(v, a, text, split),
                          tiny.embed(v, a, text, plain))
    with pytest.raises(ValueError, match="19 text rows for a layout of 20"):
        tiny.embed(v, a, text[:, 1:], split)


def test_cond_rows_patchify_and_mix_each_anchor(tiny):
    from gmlx.video.anchors import Anchor

    z = mx.random.normal((1, 24, 1, 4, 4))
    s = mx.random.normal((1, 32, 2, 3))
    noise = [mx.random.normal((4, 96))]
    cv, ca = tiny.cond_rows((Anchor(0, video=z), Anchor(4, audio=s)), 5,
                            noise=noise)
    rule = hf.ANCHOR_RULES["video"]
    want = rule.aug * hm.patchify(z, (1, 2, 2)) + (1 - rule.aug) * noise[0]
    np.testing.assert_allclose(np.array(cv[0]), np.array(want), rtol=1e-6,
                               atol=1e-7)
    assert len(cv) == len(ca) == 1
    assert mx.array_equal(ca[0], hm.pack_audio(s))
