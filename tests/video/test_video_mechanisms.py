"""The shared video mechanisms on H3's values: the packed layout and its
positions, the RoPE pair order the q and k kernel takes, the per-stream
schedules, and the anchor noise and time rule. CPU-only."""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image.ops import attention
from gmlx.video import anchors, layout, rope, schedule
from gmlx.video.families.h3 import family as hf

TL = hf.TIMELINE


def test_t2v_packs_text_then_audio_then_video():
    """64x64, 5 frames: 2 latent frames of 2x2 rows, 8 audio frames a
    channel. Text sits at its index, the targets start at the cursor."""
    vs, aus = hf.latent_shapes(5, 64, 64)
    assert vs == (1, 24, 2, 4, 4) and aus == (1, 32, 2, 8)
    lay = layout.pack(TL, 8, 2, 4, 4, 8)
    assert [(s.kind, s.start, s.stop) for s in lay.segments] == [
        ("text", 0, 8), ("audio", 8, 24), ("video", 24, 32)]
    p = lay.positions
    assert np.array_equal(p[:8, 0], np.arange(8)) and not p[:8, 1:].any()
    # The 2x2 grid of a square frame spans 0 and 16 on each axis.
    np.testing.assert_array_equal(p[8:16, 0], 8 + np.arange(8))
    np.testing.assert_array_equal(p[8:24, 0], np.tile(8 + np.arange(8), 2))
    assert set(p[8:16, 2]) == {0.0} and set(p[16:24, 2]) == {16.0}
    assert not p[8:24, 1].any()
    np.testing.assert_allclose(p[24:28, 0], 8.0)
    np.testing.assert_allclose(p[28:32, 0], 8 + 5 / 3)
    np.testing.assert_array_equal(p[24:28, 1:], [[0, 0], [0, 16], [16, 0],
                                                 [16, 16]])


def test_video_frames_advance_on_the_five_frame_clock():
    t = layout.video_times(TL, 7, 0.0)
    np.testing.assert_allclose(t, np.cumsum([0, 1, 4, 4, 4, 4, 1]) * 5 / 3)


def test_a_wide_canvas_centers_its_grid_on_the_square_root_area():
    frame, ws = layout.frame_grid(TL, 48, 84)    # 768x1344
    area = np.sqrt(48 * 84)
    hs = (np.arange(24) * (48 / area / 24) + (1 - 48 / area) / 2) * 32
    np.testing.assert_allclose(ws, (np.arange(42) * (84 / area / 42)
                                    + (1 - 84 / area) / 2) * 32)
    np.testing.assert_allclose(frame[::42, 0], hs)
    assert frame.shape == (24 * 42, 2)


def test_anchor_rows_sit_after_the_text_at_their_frame_time():
    a = layout.AnchorRows(frame=17, video_t=2, audio_t=8)
    lay = layout.pack(TL, 8, 7, 32, 32, 37, (a,))
    kinds = [(s.kind, s.rows) for s in lay.segments]
    assert kinds == [("text", 8), ("cond", 512), ("cond_audio", 16),
                     ("audio", 74), ("video", 1792)]
    cond = lay.span("cond")
    at = 8 + 5 / 3 * 17
    np.testing.assert_allclose(lay.positions[cond.start, 0], at)
    np.testing.assert_allclose(lay.positions[cond.start + 256, 0], at + 5 / 3)
    ca = lay.span("cond_audio")
    np.testing.assert_allclose(lay.positions[ca.start:ca.start + 8, 0],
                               at + np.arange(8))
    # The targets still start at the cursor after the text.
    assert lay.positions[lay.span("audio").start, 0] == 8.0
    assert lay.positions[lay.span("video").start, 0] == 8.0


def test_the_family_order_places_the_segments():
    """Another order moves the segments and keeps every row's position."""
    a = layout.AnchorRows(frame=4, video_t=1, audio_t=2)
    base = layout.pack(TL, 8, 2, 4, 4, 8, (a,))
    tl = dataclasses.replace(TL, order=("audio", "video", "anchors", "text"))
    lay = layout.pack(tl, 8, 2, 4, 4, 8, (a,))
    assert [s.kind for s in lay.segments] == ["audio", "video", "cond",
                                              "cond_audio", "text"]
    for kind in ("text", "cond", "cond_audio", "audio", "video"):
        x, y = base.span(kind), lay.span(kind)
        np.testing.assert_array_equal(lay.positions[y.start:y.stop],
                                      base.positions[x.start:x.stop])


@pytest.mark.parametrize("order", [("text", "refs", "audio", "video"),
                                   ("text", "audio", "video")],
                         ids=["unknown", "missing"])
def test_an_order_without_every_group_is_refused(order):
    with pytest.raises(ValueError, match="is not an order of"):
        layout.pack(dataclasses.replace(TL, order=order), 8, 2, 4, 4, 8)


def _split_half(x: np.ndarray, ang: np.ndarray, rot: int) -> np.ndarray:
    """The reference's rotation: pair (i, i + rot / 2) for i < rot / 2,
    dims past rot untouched."""
    h = rot // 2
    c = np.cos(ang)[None, :, None, :]
    s = np.sin(ang)[None, :, None, :]
    x1, x2 = x[..., :h], x[..., h:rot]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c, x[..., rot:]],
                          axis=-1)


def test_pair_halves_puts_each_rotated_pair_side_by_side():
    x = mx.arange(128)
    got = np.array(rope.pair_halves(x, 96))
    want = np.concatenate([np.stack([np.arange(48), 48 + np.arange(48)],
                                    axis=1).reshape(-1), np.arange(96, 128)])
    np.testing.assert_array_equal(got, want)


def test_the_kernel_pair_order_rotates_as_the_reference():
    """Norm then rotate in the paired order: the reference's split-half
    rotation of the normed heads, in that order."""
    rng = np.random.default_rng(0)
    n, heads, d = 6, 2, 128
    q = rng.standard_normal((1, n, heads, d)).astype(np.float32)
    w = (1 + 0.1 * rng.standard_normal(d)).astype(np.float32)
    pos = rng.uniform(0, 40, (n, 3))
    inv = (1.0 / 10000 ** (np.arange(16) / 16)).astype(np.float32)
    cos, sin = rope.tables(pos, mx.array(inv), d)
    assert cos.shape == (n, 64)
    qn = q / np.sqrt((q * q).mean(-1, keepdims=True) + 1e-5) * w
    ang = (pos.astype(np.float32)[:, :, None] * inv).reshape(n, 48)
    want = _split_half(qn, ang, 96)
    qq = rope.pair_halves(mx.array(q), 96)
    ww = rope.pair_halves(mx.array(w), 96)
    got, _ = attention.qk_norm_rope(qq, qq, ww, ww, cos, sin, 1e-5)
    got = np.array(got.transpose(0, 2, 1, 3))
    np.testing.assert_allclose(got, np.array(rope.pair_halves(
        mx.array(want), 96)), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("video_shift", [12.0, 6.0])
def test_the_audio_schedule_is_the_video_one_reshifted(video_shift):
    """Stepping audio on its own shift-3 schedule is ComfyUI's audio sigma
    from the video sigma."""
    sig = schedule.stream_sigmas(8, {"video": video_shift, "audio": 3.0})
    got = [schedule.time_shift(float(v), video_shift, 3.0)
           for v in sig["video"]]
    np.testing.assert_allclose(got, sig["audio"], rtol=1e-6, atol=1e-7)


# ComfyUI's time_shift_sigma on a float32 torch sigma; at each of these an
# all-float32 or all-float64 version lands one or two float32 steps away.
@pytest.mark.parametrize("from_shift,to_shift,sigma,want", [
    (1.3, 3.0, 0.42500001192092896, 0.630407989025116),
    (2.7, 1.7, 0.574999988079071, 0.46000000834465027),
    (1.7, 2.9, 0.125, 0.19594596326351166),
    (2.3, 1.3, 0.625, 0.48507457971572876)])
def test_time_shift_rounds_as_comfyui(from_shift, to_shift, sigma, want):
    assert schedule.time_shift(sigma, from_shift, to_shift) == want


def test_video_anchor_rows_take_a_little_noise_from_a_restarted_seed():
    rule = hf.ANCHOR_RULES["video"]
    z1 = mx.ones((4, 96))
    z2 = mx.zeros((4, 96))
    a1, a2 = anchors.augment([z1, z2], rule, seed=7)
    noise = (a2 / (1 - rule.aug))
    np.testing.assert_allclose(np.array(a1), np.array(rule.aug * z1
                                                      + (1 - rule.aug)
                                                      * noise), atol=1e-6)
    again = anchors.augment([z2], rule, seed=7)[0]
    assert mx.array_equal(again, a2)
    assert not mx.array_equal(anchors.augment([z2], rule, seed=8)[0], a2)
    assert float(mx.abs(noise).mean()) > 0.5


def test_audio_anchor_rows_stay_clean_and_run_clean():
    rule = hf.ANCHOR_RULES["audio"]
    z = mx.random.normal((16, 32))
    assert mx.array_equal(anchors.augment([z], rule, seed=3)[0], z)
    assert anchors.pinned(0.3, rule) == 1.0
    assert anchors.pinned(0.3, hf.ANCHOR_RULES["video"]) == 0.999
    assert anchors.pinned(0.9995, hf.ANCHOR_RULES["video"]) == 0.9995


def test_a_negative_frame_counts_from_the_end():
    assert anchors.resolve_frame(-1, 22) == 21
    assert anchors.resolve_frame(0, 22) == 0
    with pytest.raises(ValueError, match="outside a clip of 22 frames"):
        anchors.resolve_frame(22, 22)


@pytest.mark.parametrize("n,want", [(1, 5), (5, 5), (6, 22), (22, 22),
                                    (120, 124), (124, 124), (361, 362)])
def test_frame_counts_snap_up_to_the_grid(n, want):
    assert hf.align_frames(n) == want


def test_clip_lengths_size_their_latents():
    assert hf.latent_shapes(124, 1344, 768) == ((1, 24, 37, 48, 84),
                                                (1, 32, 2, 207))
    assert hf.latent_shapes(22, 512, 512) == ((1, 24, 7, 32, 32),
                                              (1, 32, 2, 37))
    with pytest.raises(ValueError, match="17 k"):
        hf.latent_shapes(24, 512, 512)
    with pytest.raises(ValueError, match="multiple of 32"):
        hf.latent_shapes(22, 500, 512)
