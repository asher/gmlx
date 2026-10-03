"""The image memory plan: the formulas on the two real header inventories,
lever order, refusal text, overcommit, and budgets of a 16 GiB and a
128 GiB machine. CPU."""
from __future__ import annotations

import pytest
from image_fixtures import Q_HEADER, Z_HEADER, fixture_view

from gmlx.image import memfit
from gmlx.image.memfit import GiB, ImageMemoryError, plan_memory

ENCODER = int(2.32 * GiB)       # Qwen3-4B Q4_K_M, no output.weight
VAE = 167_000_000               # FLUX.1 autoencoder in bf16
BUDGET_16 = 17_179_869_184 - 8e9    # 16 GiB less the 8e9 byte reserve
BUDGET_128 = 100e9


@pytest.fixture(scope="module")
def z_view():
    return fixture_view(Z_HEADER)


@pytest.fixture(scope="module")
def q_view():
    return fixture_view(Q_HEADER)


def _z(view, **kw):
    kw.setdefault("budget", BUDGET_128)
    kw.setdefault("overcommit", False)
    return plan_memory(view, ENCODER, VAE, family="z_image", **kw)


def test_dit_bytes_come_from_the_header_codecs(z_view, q_view):
    assert memfit.file_bytes(z_view) == 5_017_582_848
    assert round(memfit.file_bytes(z_view) / GiB, 2) == 4.67
    assert memfit.file_bytes(q_view) == 4_199_538_688
    assert memfit.tensor_bytes((256, 3840), "Q4_K") == 256 * 3840 // 256 * 144
    assert memfit.file_bytes(123) == 123 and memfit.file_bytes(None) == 0


def test_z_image_rows_and_step_formula(z_view):
    row = memfit.PEAKS["z_image"]
    assert memfit.image_rows(row, 1024, 1024, 22) == 4128
    assert memfit.image_rows(row, 1024, 1024, None) == 4608
    assert memfit.image_rows(row, 2048, 2048, 512) == 16384 + 512
    p = _z(z_view, height=1024, width=1024, caption_tokens=22)
    formula = 4128 * (11520 + 2 * 10240) * 2
    assert 0.24 < formula / GiB < 0.25
    assert p.terms["step_peak"] == int(formula * memfit.Z_IMAGE_LIVE)
    assert p.terms["encode_peak"] == int(22 * 3 * 9728 * 2
                                         * memfit.ENCODE_LIVE)
    assert p.terms["decode_peak"] == memfit.vae_peak("decode", 1024, 1024,
                                                     itemsize=2)
    assert p.rows == 1 and not p.levers


def test_qwen_image_rows_and_prefix_cache(q_view):
    row = memfit.PEAKS["qwen_image21"]
    assert memfit.image_rows(row, 1024, 1024, None) == 4096
    assert round(4096 * (24576 + 12288) * 2 / GiB, 2) == 0.28
    assert row.prefix(200, 0, 0) == 100 * (1 << 20)
    assert row.prefix(0, 1, 1024) == 2 * GiB
    p = plan_memory(q_view, 0, 0, family="qwen_image21", height=1024,
                    width=1024, cfg=4.0, text_tokens=200, refs=1,
                    ref_px=1024, budget=BUDGET_128, overcommit=False)
    assert p.terms["prefix_cache"] == 2 * (100 * (1 << 20) + 2 * GiB)
    assert p.rows == 2


def test_a_16_gib_machine_releases_the_encoder_and_tiles_at_1024(z_view):
    p = _z(z_view, height=1024, width=1024, budget=BUDGET_16)
    assert p.fits and p.levers == ("release the encoder", "tiled decode")
    assert p.release_encoder and p.tiled_decode
    assert p.phases["denoise"] == p.phases["encode"] - ENCODER \
        - p.terms["encode_peak"] + p.terms["step_peak"]


def test_levers_apply_in_order(z_view, monkeypatch):
    # A decode of 8 bytes per pixel, channel and itemsize keeps every
    # phase in play, so each lever is the one that makes the plan fit.
    fit = memfit.VaeFit(0.0, 8.0)
    monkeypatch.setattr(memfit, "VAE_DECODE",
                        {c: {2: fit, 4: fit} for c in ("exact", "winograd")})
    tiled = memfit.TiledFit(0.0, 1.0, 0.0)
    monkeypatch.setattr(memfit, "VAE_TILED", {"decode": {2: tiled, 4: tiled}})

    def levers(gib, **kw):
        return _z(z_view, height=2048, width=2048, cfg=3.0,
                  budget=gib * GiB, **kw).levers

    rows, release, tiled = memfit.LEVERS
    assert levers(16.0) == ()
    assert levers(15.5) == (rows,)
    assert levers(14.0) == (rows, release)
    assert levers(12.0) == (rows, release, tiled)
    assert levers(14.0, batch_rows=False) == (release,)


def test_tiled_decode_halves_the_decode_peak_at_2048(z_view):
    # Tiles hold their activations at each GroupNorm, so at 2048 the tiled
    # decode is still the largest phase, at under half the whole decode.
    whole = _z(z_view, height=2048, width=2048)
    assert whole.fits and not whole.levers
    in_tiles = memfit.vae_peak("decode", 2048, 2048, itemsize=2, tiled=True)
    need = whole.phases["decode"] - whole.terms["decode_peak"] + in_tiles
    assert need > whole.phases["denoise"]
    tiled = _z(z_view, height=2048, width=2048, budget=need + 1)
    assert tiled.fits and tiled.tiled_decode
    assert tiled.terms["decode_peak"] == memfit.vae_peak(
        "decode", 2048, 2048, itemsize=2, tiled=True)
    assert tiled.terms["decode_peak"] < whole.terms["decode_peak"] / 2


def test_a_128_gib_machine_fits_both_sizes_without_levers(z_view):
    for size in (1024, 2048):
        p = _z(z_view, height=size, width=size, budget=BUDGET_128)
        assert p.fits and p.levers == ()


def test_refusal_names_every_term_in_gib_and_the_largest(z_view):
    with pytest.raises(ImageMemoryError) as e:
        _z(z_view, height=2048, width=2048, budget=BUDGET_16)
    msg = str(e.value)
    for term in ("dit 4.67 GiB", "encoder 2.32 GiB", "step_peak",
                 "decode_peak", "budget of 8.55 GiB"):
        assert term in msg
    assert "The largest term is decode_peak; try a lower resolution" in msg
    assert "GMLX_OVERCOMMIT=1 overrides" in msg
    assert e.value.plan.levers == ("release the encoder", "tiled decode")
    assert not e.value.plan.fits


def test_a_borrowed_encoder_is_never_released(z_view):
    p = _z(z_view, height=1024, width=1024, budget=12 * GiB,
           encoder_owned=False)
    assert p.fits and p.levers == ("tiled decode",)
    assert not p.release_encoder
    with pytest.raises(ImageMemoryError, match="borrowed encoder"):
        _z(z_view, height=1024, width=1024, budget=BUDGET_16,
           encoder_owned=False)


def test_overcommit_returns_the_plan(z_view, monkeypatch):
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    p = plan_memory(z_view, ENCODER, VAE, family="z_image", height=2048,
                    width=2048, budget=BUDGET_16)
    assert not p.fits


def test_budget_comes_from_capacity_by_default(z_view, monkeypatch):
    from gmlx.serve import capacity

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: BUDGET_16)
    p = plan_memory(z_view, ENCODER, VAE, family="z_image", height=1024,
                    width=1024)
    assert p.budget == BUDGET_16
    assert p.levers == ("release the encoder", "tiled decode")
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: BUDGET_128)
    p = plan_memory(z_view, ENCODER, VAE, family="z_image", height=2048,
                    width=2048)
    assert p.budget == BUDGET_128 and p.fits and not p.levers


def test_reference_refusal_names_what_fits(q_view):
    with pytest.raises(ImageMemoryError) as e:
        plan_memory(q_view, ENCODER, 0, family="qwen_image21", height=1024,
                    width=1024, cfg=4.0, refs=16, ref_px=1024,
                    budget=60 * GiB, overcommit=False)
    msg = str(e.value)
    assert "The largest term is prefix_cache" in msg
    assert "; 13 references of 1024 px fit, or 16 of 928 px" in msg


def test_fp32_vae_on_fp16_activations(z_view):
    p = _z(z_view, height=1024, width=1024, dtype="float16")
    assert p.terms["vae"] == VAE
    fp32 = memfit.vae_peak("decode", 1024, 1024, itemsize=4)
    assert p.terms["decode_peak"] == fp32
    assert fp32 > memfit.vae_peak("decode", 1024, 1024, itemsize=2)
    vae16 = memfit.vae_bytes(_TinyVae(), 2)
    vae32 = memfit.vae_bytes(_TinyVae(), 4)
    assert vae32 == 2 * vae16 == 2 * 2 * 30


class _TinyVae:
    shapes = {"a": (2, 3), "b": (4, 6)}


def test_unknown_family_is_refused(z_view):
    with pytest.raises(ValueError, match="known: qwen_image21, z_image"):
        plan_memory(z_view, 0, 0, family="flux", height=8, width=8,
                    budget=1e12, overcommit=False)


# Footprints in GiB of VAE passes under bounded_cache, measured on an M5
# Max: whole passes at 256 to 1024 pixels on each conv path, and tiled
# passes in shrunk tiles on the exact path. The fitted rows must stay above every one.
MEASURED = {
    ("exact", "decode", 2): {256: 0.83, 512: 2.35, 768: 3.48, 1024: 4.45},
    ("exact", "decode", 4): {256: 1.65, 512: 3.77, 768: 5.03, 1024: 7.01},
    ("exact", "encode", 2): {256: 0.47, 512: 1.67, 768: 2.79, 1024: 2.87},
    ("exact", "encode", 4): {256: 0.94, 512: 3.44, 768: 3.75, 1024: 4.64},
    ("winograd", "decode", 2): {256: 2.19, 512: 3.35, 768: 4.60,
                                1024: 6.64},
    ("winograd", "decode", 4): {256: 2.91, 512: 4.55, 768: 7.24,
                                1024: 10.42},
    ("winograd", "encode", 2): {256: 1.08, 512: 2.49, 768: 2.92,
                                1024: 4.60},
    ("winograd", "encode", 4): {256: 2.24, 512: 3.25, 768: 4.87,
                                1024: 7.61},
}
MEASURED_TILED = {
    ("decode", 2): {768: 2.29, 1024: 3.20, 2048: 7.09},
    ("decode", 4): {768: 3.07, 1024: 4.40, 2048: 12.74},
    ("encode", 2): {768: 1.72, 1024: 2.73, 2048: 5.28},
    ("encode", 4): {768: 2.68, 1024: 3.41, 2048: 9.18},
}


def test_the_vae_rows_cover_the_measured_footprints():
    for (conv, op, item), sizes in MEASURED.items():
        for size, gib in sizes.items():
            assert memfit.vae_peak(op, size, size, itemsize=item,
                                   conv=conv) >= gib * GiB, (conv, op, size)
    for (op, item), sizes in MEASURED_TILED.items():
        for size, gib in sizes.items():
            assert memfit.vae_peak(op, size, size, itemsize=item, tiled=True,
                                   conv="exact") >= gib * GiB, (op, size)


def _limit_now() -> int:
    import mlx.core as mx

    lim = mx.set_cache_limit(0)
    mx.set_cache_limit(lim)
    return lim


@pytest.fixture
def cache_limit():
    import mlx.core as mx

    before = mx.set_cache_limit(1 << 40)
    yield
    mx.set_cache_limit(before)


def test_bounded_cache_lowers_the_limit_for_its_span(cache_limit):
    with memfit.bounded_cache(5 << 20):
        assert _limit_now() == 5 << 20
        with memfit.bounded_cache(1 << 20):
            assert _limit_now() == 5 << 20
        assert _limit_now() == 5 << 20
    assert _limit_now() == 1 << 40
    with pytest.raises(KeyError):
        with memfit.bounded_cache(5 << 20):
            raise KeyError("x")
    assert _limit_now() == 1 << 40


def test_bounded_cache_never_raises_a_tighter_limit(cache_limit):
    import mlx.core as mx

    mx.set_cache_limit(1 << 20)
    with memfit.bounded_cache(5 << 20):
        assert _limit_now() == 1 << 20
    assert _limit_now() == 1 << 20


def test_bounded_cache_keeps_a_limit_set_inside_it(cache_limit):
    import mlx.core as mx

    with memfit.bounded_cache(5 << 20):
        mx.set_cache_limit(7 << 20)
    assert _limit_now() == 7 << 20


def test_bounded_cache_restores_once_across_threads(cache_limit):
    import threading

    inside = threading.Barrier(2)
    leave = threading.Event()

    def worker():
        with memfit.bounded_cache(5 << 20):
            inside.wait()
            leave.wait()

    t = threading.Thread(target=worker)
    t.start()
    with memfit.bounded_cache(5 << 20):
        inside.wait()
    assert _limit_now() == 5 << 20
    leave.set()
    t.join()
    assert _limit_now() == 1 << 40


def _unit_fits(monkeypatch):
    unit = {2: memfit.VaeFit(0.0, 1.0), 4: memfit.VaeFit(0.0, 1.0)}
    rows = {"exact": unit, "winograd": unit}
    monkeypatch.setattr(memfit, "VAE_DECODE", rows)
    monkeypatch.setattr(memfit, "VAE_ENCODE", rows)
    tiled = {2: memfit.TiledFit(0.0, 0.125, 0.0),
             4: memfit.TiledFit(0.0, 0.125, 0.0)}
    monkeypatch.setattr(memfit, "VAE_TILED",
                        {"decode": tiled, "encode": tiled})


def test_the_tile_layout_is_the_one_tiling_runs():
    from gmlx.image.vae import tiling

    for px in range(8, 4104, 8):
        lat = px // 8
        length = tiling.tile_length(lat, memfit.TILE_PX // 8,
                                    memfit.OVERLAP_PX // 8)
        starts = tiling.tile_starts(lat, length, memfit.OVERLAP_PX // 8)
        assert memfit.tile_layout(px) == (len(starts), 8 * length), px


def test_vae_tiling_picks_whole_then_tiles_then_refuses(monkeypatch):
    _unit_fits(monkeypatch)
    kw = dict(itemsize=2, channels=128, overcommit=False)
    whole = 2048 * 2048 * 128 * 2
    # The 5 x 5 tiles of 464 pixels at k_tiles = 0.125.
    assert memfit.tile_layout(2048) == (5, 464)
    tiles = int(0.125 * 25 * 464 * 464 * 128 * 2)
    assert memfit.vae_peak("decode", 2048, 2048, itemsize=2) == whole
    assert memfit.vae_peak("decode", 2048, 2048, itemsize=2,
                           tiled=True) == tiles
    assert memfit.vae_tiling("decode", 2048, 2048, free=whole, **kw) is False
    assert memfit.vae_tiling("decode", 2048, 2048, free=whole - 1,
                             **kw) is True
    assert memfit.vae_tiling("decode", 2048, 2048, tiled=True, free=tiles,
                             **kw) is True
    with pytest.raises(ImageMemoryError, match="try a smaller image") as e:
        memfit.vae_tiling("decode", 2048, 2048, free=tiles - 1, **kw)
    msg = str(e.value)
    assert f"needs {tiles / GiB:.2f} GiB" in msg and "in tiles" in msg
    with pytest.raises(ImageMemoryError,
                       match=f"a tiled pass needs {tiles / GiB:.2f} GiB"):
        memfit.vae_tiling("decode", 2048, 2048, tiled=False,
                          free=whole - 1, **kw)
    assert memfit.vae_tiling("decode", 2048, 2048, tiled=False, free=0,
                             itemsize=2, overcommit=True) is False


def test_vae_tiling_runs_a_small_image_whole(monkeypatch):
    _unit_fits(monkeypatch)
    one = 512 * 512 * 128 * 4
    assert memfit.vae_tiling("encode", 512, 512, itemsize=4, tiled=True,
                             free=one, overcommit=False) is False
    with pytest.raises(ImageMemoryError, match="whole needs"):
        memfit.vae_tiling("encode", 512, 512, itemsize=4, free=one - 1,
                          overcommit=False)
    assert memfit.vae_peak("encode", 512, 512, itemsize=4, batch=3) == 3 * one


def test_vae_tiling_without_a_free_measure_never_refuses(monkeypatch):
    _unit_fits(monkeypatch)
    monkeypatch.setattr(memfit, "free_bytes", lambda: None)
    assert memfit.vae_tiling("decode", 4096, 4096, itemsize=2) is False
    assert memfit.vae_tiling("decode", 4096, 4096, itemsize=2,
                             tiled=True) is True
