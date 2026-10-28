"""The H3 model's own conventions on a tiny model: patch order, stereo
packing, the curve table blend, which modulation row each segment reads,
and the sign of the velocity. CPU-only."""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from gmlx.video import api, layout, schedule
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import model as hm

TINY = {"num_attention_heads": 2, "attention_head_dim": 128,
        "hidden_size": 64, "num_layers": 2, "num_refiner_layers": 1,
        "ffn_dim": 48, "in_channels": 24, "audio_in_channels": 32,
        "patch_size": [1, 2, 2], "text_dim": 40, "rope_freq_dim": 16,
        "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5,
        "adaln_grid": 1025, "adaln_basis": 8}


@pytest.fixture(scope="module")
def tiny():
    mx.random.seed(0)
    m = hm.H3Transformer(TINY)
    m.update(tree_map(lambda p: 0.05 * mx.random.normal(p.shape),
                      m.parameters()))
    m.rope.inv_freq = mx.array(1 / 10000 ** (np.arange(16) / 16), mx.float32)
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
    for order in (hf.SEGMENT_ORDER, ("audio", "video", "anchors", "text")):
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


def test_a_run_steps_each_stream_on_its_own_schedule(tiny, monkeypatch):
    """Three forwards with a different velocity each: each stream moves by
    its own sigma drops, so another table or shift lands elsewhere, and
    each forward's plan, which the model reads in place of the times
    passed beside it, puts each head at its stream's time."""
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
    got = api.sample(tiny, mx.zeros(vs), mx.zeros(aus),
                     mx.zeros((1, 8, 64)), lay, steps=3, shifts=hf.SHIFTS)
    sig = {k: schedule.sigmas(3, hf.SHIFTS[k]) for k in vel}
    for x, k in zip(got, ("video", "audio")):
        s = sig[k]
        want = sum(float(s[i + 1] - s[i]) * vel[k][i] for i in range(3))
        np.testing.assert_allclose(np.array(x), want, rtol=1e-6)
    assert seen == [(api.stream_time(sig["video"][i]),
                     api.stream_time(sig["audio"][i])) for i in range(3)]
    assert seen[0] == (0.0, 0.0)


def test_float16_is_refused(tiny):
    with pytest.raises(hm.H3DtypeError, match="bfloat16 or float32"):
        hm.H3Transformer(TINY).act_dtype = mx.float16
