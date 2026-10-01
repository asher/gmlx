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


def test_a_16_gib_machine_releases_the_encoder_at_1024(z_view):
    p = _z(z_view, height=1024, width=1024, budget=BUDGET_16)
    assert p.fits and p.levers == ("release the encoder",)
    assert p.release_encoder and not p.tiled_decode
    assert p.phases["denoise"] == p.phases["encode"] - ENCODER \
        - p.terms["encode_peak"] + p.terms["step_peak"]


def test_levers_apply_in_order(z_view):
    def levers(gib, **kw):
        return _z(z_view, height=2048, width=2048, cfg=3.0,
                  budget=gib * GiB, **kw).levers

    rows, release, tiled = memfit.LEVERS
    assert levers(16.5) == ()
    assert levers(15.6) == (rows,)
    assert levers(14.0) == (rows, release)
    assert levers(12.0) == (rows, release, tiled)
    assert levers(14.0, batch_rows=False) == (release,)


def test_tiled_decode_bounds_the_decode_phase(z_view):
    whole = _z(z_view, height=2048, width=2048)
    assert whole.fits and not whole.levers
    tiled = _z(z_view, height=2048, width=2048,
               budget=whole.phases["denoise"] + 1)
    assert tiled.tiled_decode
    assert tiled.terms["decode_peak"] < whole.terms["decode_peak"] / 8


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
    assert "The largest term is dit; try a smaller DiT quant" in msg
    assert "GMLX_OVERCOMMIT=1 overrides" in msg
    assert e.value.plan.levers == ("release the encoder", "tiled decode")
    assert not e.value.plan.fits


def test_a_borrowed_encoder_is_never_released(z_view):
    p = _z(z_view, height=1024, width=1024, budget=BUDGET_16,
           encoder_owned=False)
    assert p.fits and p.levers == ("tiled decode",)
    assert not p.release_encoder
    with pytest.raises(ImageMemoryError, match="borrowed encoder"):
        _z(z_view, height=2048, width=2048, budget=BUDGET_16,
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
    assert p.budget == BUDGET_16 and p.levers == ("release the encoder",)
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
    vae16 = memfit.vae_bytes(_TinyVae(), 2)
    vae32 = memfit.vae_bytes(_TinyVae(), 4)
    assert vae32 == 2 * vae16 == 2 * 2 * 30
    assert memfit.VAE_K[4] * 4 > memfit.VAE_K[2] * 2


class _TinyVae:
    shapes = {"a": (2, 3), "b": (4, 6)}


def test_unknown_family_is_refused(z_view):
    with pytest.raises(ValueError, match="known: qwen_image21, z_image"):
        plan_memory(z_view, 0, 0, family="flux", height=8, width=8,
                    budget=1e12, overcommit=False)
