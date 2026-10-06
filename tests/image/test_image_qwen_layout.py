"""The token layout of Qwen-Image 2.1 against the diffusers oracle's, from
the committed fixture selftest_qwen_image21_layout: the rotary position of
every token, the expansion of image slots into reference tokens, the
modulation row each token reads, the prefill segments and the rotary
frequencies. scripts/image_oracle/check3_qwen_image21.py --self-test
reads each of them back from the reference's own outputs. The cases are a
text prefix, one reference image, two reference images with text between
them, two adjacent reference images, and a longer prefix with a 12 x 20
target. The rest covers the layout's refusals and its memos.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import qwen_tiny as qt
from oracle_fixture import committed_fixture

from gmlx.image.families.qwen_image21 import layout
from gmlx.image.families.qwen_image21 import model as qm

CASES = ("text", "ref1", "ref2", "adjacent", "wide")
# The axes of the official config: frame, height and width.
REAL_AXES = (16, 56, 56)


@pytest.fixture(scope="module")
def oracle():
    return committed_fixture("selftest_qwen_image21_layout")


def _case(oracle, case: str):
    arrays, m = oracle
    spec = m["inputs"]["cases"][case]
    n_rows, slots, refs = qt.CASES[case]
    assert (spec["rows"], tuple(spec["image_slots"]),
            tuple(map(tuple, spec["ref_shapes"]))) == (n_rows, slots, refs)
    lay = layout.prefix_layout(n_rows, slots, refs)
    got = {k.split(".", 1)[1]: v for k, v in arrays.items()
           if k.startswith(case + ".")}
    return lay, tuple(spec["target_tokens"]), got


@pytest.mark.parametrize("case", CASES)
def test_positions_equal_the_oracle(oracle, case):
    lay, (h, w), want = _case(oracle, case)
    target = layout.target_positions(lay.position, h, w)
    got = np.concatenate([lay.positions, target])
    assert got.shape == want["positions"].shape == (lay.tokens + h * w, 3)
    assert np.array_equal(got, want["positions"])
    # The frame axis of the target sits where the prefix stopped.
    assert int(want["positions"][lay.tokens, 0]) == lay.position
    # A text token has one position on all three axes.
    text = ~want["image_pad_mask"]
    assert (got[text] == got[text][:, :1]).all()


def test_the_positions_of_one_reference_image_and_the_target():
    """Three text rows, a 2 x 4 reference in two slots, four text rows and
    a 4 x 6 target, written out."""
    lay = layout.prefix_layout(9, (3, 4), ((2, 4),))
    pos = lay.positions.tolist()
    assert pos[:3] == [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    # The grid is centered on zero and its frame axis holds position 3.
    assert pos[3:11] == [[3, r, c] for r in (-1, 0) for c in (-2, -1, 0, 1)]
    # The block advances the position by its larger side, 4.
    assert pos[11:] == [[p, p, p] for p in (7, 8, 9, 10)]
    assert lay.position == 11
    target = layout.target_positions(lay.position, 4, 6).tolist()
    assert target == [[11, r, c] for r in (-2, -1, 0, 1)
                      for c in (-3, -2, -1, 0, 1, 2)]


@pytest.mark.parametrize("case", CASES)
def test_slot_expansion_equals_the_oracle(oracle, case):
    """Each image slot of the encoder rows becomes four tokens, which the
    reference latents overwrite in order."""
    lay, (h, w), want = _case(oracle, case)
    n_rows, slots, refs = qt.CASES[case]
    n_text = lay.text_rows.size
    assert n_text == n_rows - len(slots)
    assert lay.tokens == n_text + layout.SLOT_TOKENS * len(slots)
    # The oracle numbers the encoder rows and then the image tokens, the
    # references first. gmlx numbers the text rows and then the reference
    # tokens.
    source = np.where(lay.source < n_text,
                      lay.text_rows[np.minimum(lay.source, n_text - 1)],
                      n_rows + lay.source - n_text)
    n_ref = sum(a * b for a, b in refs)
    target = n_rows + n_ref + np.arange(h * w)
    assert np.array_equal(np.concatenate([source, target]),
                          want["joint_source"])
    image = np.concatenate([lay.source >= n_text, np.ones(h * w, bool)])
    assert np.array_equal(image, want["image_pad_mask"])
    # No slot row of the encoder reaches the sequence.
    assert not set(want["joint_source"].tolist()) & set(slots)


@pytest.mark.parametrize("case", CASES)
def test_segments_and_image_blocks_equal_the_oracle(oracle, case):
    lay, (h, w), want = _case(oracle, case)
    assert [(s, e, bool(t)) for s, e, t in want["segments"].tolist()] == list(
        lay.segments)
    ids = np.full(lay.tokens + h * w, -1)
    blocks = [(s, e) for s, e, text in lay.segments if not text]
    for i, (s, e) in enumerate(blocks):
        ids[s:e] = i
    ids[lay.tokens:] = len(blocks)
    assert np.array_equal(ids, want["image_ids"])
    assert len(blocks) == len(qt.CASES[case][2])


@pytest.mark.parametrize("case", CASES)
def test_modulation_rows_equal_the_oracle(oracle, case):
    """Every prefix token reads the t = 0 row, row 1 of the oracle's
    table, and every target token the sampled row, row 0. The target is
    the tail of the sequence."""
    lay, (h, w), want = _case(oracle, case)
    tail = np.arange(lay.tokens + h * w) >= lay.tokens
    assert np.array_equal(want["target_token_mask"], tail)
    assert np.array_equal(want["modulation_row"], np.where(tail, 0, 1))


@pytest.mark.parametrize("case", ["text", "ref1", "ref2"])
def test_the_prefill_reads_t_zero_and_the_target_the_sampled_row(
        case, monkeypatch):
    """The split of the forward follows the oracle's row selection: the
    blocks of ``prefill_prefix`` take the modulation of t = 0 and the
    blocks of ``forward_target`` that of the sampled timestep."""
    model = qt.model_of(qt.weights())
    cond, lat = qt.conditioning(case), qt.latents()
    made, used = [], []
    real_at = qm.QwenImage21DiT.modulation_at
    real_prefill, real_target = qm.Block.prefill, qm.Block.target

    def modulation_at(self, t):
        mod = real_at(self, t)
        made.append((np.asarray(t, np.float32).tolist(), mod))
        return mod

    def prefill(self, x, mod, *args):
        used.append(("prefix", x.shape[1], mod))
        return real_prefill(self, x, mod, *args)

    def target(self, x, mod, *args):
        used.append(("target", x.shape[1], mod))
        return real_target(self, x, mod, *args)

    monkeypatch.setattr(qm.QwenImage21DiT, "modulation_at", modulation_at)
    monkeypatch.setattr(qm.Block, "prefill", prefill)
    monkeypatch.setattr(qm.Block, "target", target)
    sigma = 0.625
    with mx.stream(mx.cpu):
        prefix = qm.prefill_prefix(model, cond)
        mx.eval(qm.forward_target(model, lat, sigma, cond, prefix))
    (t0, zero), (t1, sampled) = made
    assert t0 == [0.0] and t1 == [625.0]
    n, t = model.cfg.n_layers, lat.shape[2] * lat.shape[3]
    assert [(kind, rows) for kind, rows, _ in used] == (
        [("prefix", prefix.tokens)] * n + [("target", t)] * n)
    assert all(mod is zero for kind, _, mod in used if kind == "prefix")
    assert all(mod is sampled for kind, _, mod in used if kind == "target")


def test_rotary_frequencies_equal_the_oracle_within_one_ulp(oracle):
    arrays, m = oracle
    theta = m["inputs"]["theta"]
    off = {}
    for d in m["inputs"]["freq_dims"]:
        got, want = layout.rope_freqs(d, theta), arrays[f"rope_freqs.{d}"]
        assert got.dtype == want.dtype == np.float32
        ulps = np.abs(got.view(np.int32).astype(np.int64)
                      - want.view(np.int32).astype(np.int64))
        off[d] = (int(ulps.max()), int((ulps > 0).sum()))
    print(f"qwen rotary frequencies against torch's, (ulps, elements off) "
          f"by dim: {off}")
    # gmlx rounds the float64 power once. torch's float32 power is one ulp
    # from that at some elements of the 56-wide axes and equal elsewhere.
    for d in (4, 6, 16):
        assert off[d] == (0, 0)
    assert off[56][0] <= 1
    assert REAL_AXES == (16, 56, 56)
    # Text sits on all three axes at one shared position, so the 56-wide
    # axes hold every position of the tables, as the frame axis does. One
    # frequency of a 56-wide axis is one ulp off torch's. Over all
    # positions that moves an entry of the cos and sin tables by 1.53e-5
    # at most, and by 1.2e-7 within 64 positions of the center, where the
    # image tokens of 2048 pixels sit. Against torch's own tables, with
    # its cos and sin, the two were measured at 1.5e-5, at position 7729,
    # and 1.5e-7.
    ours, torchs = layout.rope_freqs(56, theta), arrays["rope_freqs.56"]

    def table_gap(lo: int, hi: int) -> tuple[float, int]:
        pos = np.arange(lo, hi, dtype=np.float32)[:, None]
        a, b = pos * ours[None], pos * torchs[None]
        gap = np.maximum(np.abs(np.cos(a) - np.cos(b)),
                         np.abs(np.sin(a) - np.sin(b))).max(axis=1)
        return float(gap.max()), lo + int(gap.argmax())

    (full, at), (center, _) = table_gap(-1024, 8192), table_gap(-64, 64)
    print(f"qwen rotary table gap against torch's frequencies on a "
          f"56-wide axis: {full:.2e} at position {at} over all positions, "
          f"{center:.2e} within 64 of the center")
    assert full <= 1.6e-5 and center <= 1.6e-7


def test_rope_tables_are_the_cos_and_sin_of_one_float32_product():
    pos = np.array([[5, -2, 1], [8191, -1024, 0]])
    cos, sin = layout.rope_tables(pos, REAL_AXES, 10000.0)
    assert cos.shape == sin.shape == (2, sum(REAL_AXES) // 2)
    assert cos.dtype == sin.dtype == mx.float32
    f = [layout.rope_freqs(d, 10000.0) for d in REAL_AXES]
    ang = np.concatenate([np.float32(p) * fr for p, fr in zip(pos[1], f)])
    assert ang.dtype == np.float32
    assert np.array_equal(np.array(cos)[1], np.cos(ang))
    assert np.array_equal(np.array(sin)[1], np.sin(ang))


def test_positions_outside_the_tables_are_refused():
    layout.prefix_layout(8192)
    with pytest.raises(ValueError, match="0 to 8192 is outside -1024 to 8191"):
        layout.prefix_layout(8193)
    layout.target_positions(0, 2048, 2)
    with pytest.raises(ValueError, match="outside -1024 to 8191"):
        layout.target_positions(0, 2050, 2)
    with pytest.raises(ValueError, match="outside -1024 to 8191"):
        layout.target_positions(8192, 2, 2)


def test_bad_slots_and_reference_shapes_are_refused():
    for slots, refs, text in (
            ((4, 3), ((2, 4),), "ascending rows"),
            ((3, 3), ((2, 4),), "ascending rows"),
            ((3, 9), ((2, 4),), "ascending rows"),
            ((3, 4, 5), ((2, 4),), "3 image slots hold 12 latent tokens, "
                                   "and the reference images have 8"),
            ((3, 5), ((2, 4),), "2x4 latent tokens needs 2 consecutive"),
            ((3, 4), ((1, 6), (1, 2)), "1x6 latent tokens needs 1.5 "
                                       "consecutive"),
            ((2, 3, 4), ((2, 2), (4, 2)), None)):
        if text is None:
            layout.prefix_layout(9, slots, refs)
            continue
        with pytest.raises(ValueError, match=text):
            layout.prefix_layout(9, slots, refs)
    with pytest.raises(ValueError, match="3x4 latent tokens needs 3 "
                                         "consecutive"):
        layout.prefix_layout(9, (2, 3, 5), ((3, 4),))


def test_the_memos_change_speed_only():
    key = (9, (3, 4), ((2, 4),))
    layout.clear_caches()
    lay = layout.prefix_layout(*key)
    assert layout.prefix_layout(*key) is lay
    for a in (lay.text_rows, lay.source, lay.positions):
        assert not a.flags.writeable
    cos, sin = layout.prefix_tables((4, 6, 6), 10000.0, *key)
    assert layout.prefix_tables((4, 6, 6), 10000.0, *key)[0] is cos
    tcos, tsin = layout.target_tables((4, 6, 6), 10000.0, lay.position, 4, 6)
    layout.clear_caches()
    again = layout.prefix_layout(*key)
    assert again is not lay and again.segments == lay.segments
    assert np.array_equal(again.positions, lay.positions)
    assert np.array_equal(again.source, lay.source)
    cos2, sin2 = layout.prefix_tables((4, 6, 6), 10000.0, *key)
    assert cos2 is not cos
    assert np.array_equal(np.array(cos2), np.array(cos))
    assert np.array_equal(np.array(sin2), np.array(sin))
    tcos2, tsin2 = layout.target_tables((4, 6, 6), 10000.0, lay.position,
                                        4, 6)
    assert np.array_equal(np.array(tcos2), np.array(tcos))
    assert np.array_equal(np.array(tsin2), np.array(tsin))
    # The tables are those of the positions.
    want = layout.rope_tables(lay.positions, (4, 6, 6), 10000.0)
    assert np.array_equal(np.array(cos2), np.array(want[0]))
