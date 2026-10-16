"""The image memory plan: the formulas on the two real header inventories,
lever order with a budget that forces each lever, the mmproj at the bytes
it holds once loaded, refusal text, overcommit, and budgets of a 16 GiB
and a 128 GiB machine. CPU."""
from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

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
    assert p.terms["step_peak"] == memfit.Z_IMAGE_STEP.bytes(formula)
    assert p.terms["encode_peak"] == int(22 * 3 * 9728 * 2
                                         * memfit.ENCODE_LIVE)
    assert p.terms["decode_peak"] == memfit.vae_peak(
        "decode", 1024, 1024, family="z_image", itemsize=2)
    assert p.rows == 1 and not p.levers


def test_qwen_image_rows_and_prefix_cache(q_view):
    row = memfit.PEAKS["qwen_image21"]
    pre = row.prefix
    assert memfit.image_rows(row, 1024, 1024, None) == 4096
    assert round(4096 * (24576 + 12288) * 2 / GiB, 2) == 0.28
    # 200 text rows cache 100 MiB, and one 1024 pixel reference 2 GiB.
    assert pre.rows(200, 0, 0) == 200 and pre.rows(0, 1, 1024) == 4096
    assert pre.rows(200, 3, 512) == 200 + 3 * 1024
    assert pre.cache_bytes(200, 2) == 100 * (1 << 20)
    assert pre.cache_bytes(4096, 2) == 2 * GiB
    p = plan_memory(q_view, 0, 0, family="qwen_image21", height=1024,
                    width=1024, cfg=4.0, text_tokens=200, refs=1,
                    ref_px=1024, budget=BUDGET_128, overcommit=False)
    assert p.terms["prefix_cache"] == 2 * (100 * (1 << 20) + 2 * GiB)
    assert p.rows == 2
    # The cache follows the activation dtype.
    p = plan_memory(q_view, 0, 0, family="qwen_image21", height=1024,
                    width=1024, text_tokens=200, dtype="float32",
                    budget=BUDGET_128, overcommit=False)
    assert p.terms["prefix_cache"] == 2 * 100 * (1 << 20)


def test_the_qwen_row_holds_the_sizes_of_the_header(q_view):
    from gmlx.image import fingerprint as fp

    got = fp.identify(q_view)
    cfg, row = got.config, memfit.PEAKS[got.family.name]
    assert (row.prefix.blocks, row.prefix.dim) == (
        cfg["num_layers"], cfg["inner_dim"]) == (32, 4096)
    # The widest outputs of a row: the fused gate and up product, and the
    # gated half that the MLP's last product reads.
    assert row.width == 2 * cfg["ffn_dim"] + cfg["ffn_dim"] == 36864
    # One token is one latent pixel of the 16x VAE.
    assert row.token_px == row.prefix.token_px == 16 * cfg["patch_size"]
    assert row.caption_rows == 0 and row.row_multiple == 1


def test_a_qwen_step_holds_the_joined_keys_and_the_prefill(q_view):
    row = memfit.PEAKS["qwen_image21"]
    kw = dict(family="qwen_image21", height=1024, width=1024,
              budget=BUDGET_128, overcommit=False)
    image = 4096 * 36864 * 2

    def join(rows):
        return 2 * rows * 4096 * 2

    # Text to image: 4096 target rows over the K and V of 512 text rows.
    p = plan_memory(q_view, 0, 0, **kw)
    assert p.terms["prefix_cache"] == 2 * 32 * 512 * 4096 * 2 == GiB // 4
    assert p.terms["step_peak"] == row.step.bytes(image + join(512 + 4096),
                                                  1, join(512))
    assert p.phases["denoise"] == (p.terms["dit"] + p.terms["prefix_cache"]
                                   + p.terms["step_peak"])
    assert memfit.prefix_terms(row, 4096, 512, 0, 0, 2) == (
        GiB // 4, image + join(512 + 4096), 512 * 36864 * 2)
    # Two 1024 pixel references: the prefill of 8704 prefix rows is wider
    # than one step of 4096 target rows, and narrower than two.
    prefix = 512 + 2 * 4096
    step, prefill = image + join(prefix + 4096), prefix * 36864 * 2
    assert step < prefill < 2 * step
    p = plan_memory(q_view, 0, 0, refs=2, ref_px=1024, **kw)
    assert p.rows == 1 and p.terms["step_peak"] == row.prefill.bytes(prefill)
    assert row.prefill.bytes(prefill) > row.step.bytes(step)
    p = plan_memory(q_view, 0, 0, refs=2, ref_px=1024, cfg=4.0, **kw)
    assert p.rows == 2
    assert p.terms["step_peak"] == row.step_peak(step, prefill, 2,
                                                 join(prefix))
    assert p.terms["prefix_cache"] == 2 * 32 * prefix * 4096 * 2 * 2
    # One 512 pixel reference under guidance: the two rows of a step are
    # the term.
    prefix = 512 + 1024
    step, prefill = image + join(prefix + 4096), prefix * 36864 * 2
    p = plan_memory(q_view, 0, 0, refs=1, ref_px=512, cfg=4.0, **kw)
    assert p.terms["step_peak"] == row.step.bytes(step, 2, join(prefix))
    assert row.step.bytes(step, 2, join(prefix)) > row.prefill.bytes(prefill)
    # A family without a prefix has neither term.
    z = memfit.PEAKS["z_image"]
    assert memfit.prefix_terms(z, 4128, 512, 2, 1024, 2) == (
        0, 4128 * z.width * 2, 0)


def test_the_negative_row_counts_whenever_guidance_is_not_one(q_view, z_view):
    """The rule of api.sample: a negative row runs unless cfg is 1.0."""
    kw = dict(family="qwen_image21", height=1024, width=1024,
              budget=BUDGET_128, overcommit=False)
    one = plan_memory(q_view, 0, 0, cfg=1.0, **kw)
    assert one.rows == 1
    for cfg in (0.5, 4.0, 0.0, -1.0):
        p = plan_memory(q_view, 0, 0, cfg=cfg, **kw)
        assert p.rows == 2, cfg
        assert p.terms["prefix_cache"] == 2 * one.terms["prefix_cache"]
    z = dict(height=1024, width=1024)
    assert _z(z_view, cfg=1.0, **z).rows == 1
    assert _z(z_view, cfg=0.5, **z).terms == _z(z_view, cfg=4.0, **z).terms
    assert _z(z_view, cfg=0.5, **z).rows == 2


def test_prefix_rows_of_the_conditionings_replace_the_counts(q_view,
                                                             z_view):
    row = memfit.PEAKS["qwen_image21"]
    kw = dict(family="qwen_image21", height=1024, width=1024,
              budget=BUDGET_128, overcommit=False)
    by_count = plan_memory(q_view, 0, 0, refs=1, ref_px=1024, **kw)
    by_rows = plan_memory(q_view, 0, 0, prefix_rows=[512 + 4096], **kw)

    def later(terms: dict) -> dict:
        return {k: v for k, v in terms.items() if k != "encode_peak"}

    assert later(by_rows.terms) == later(by_count.terms)
    # The count says the request has a reference to encode: its image
    # rows and the tower. Rows alone are a conditioning made already.
    assert by_count.terms["encode_peak"] - by_rows.terms["encode_peak"] == \
        int(1024 * row.encode_row_bytes * memfit.ENCODE_LIVE
            + row.encode_tower_bytes)
    # Under guidance each conditioning has its own cache, and the step
    # and the prefill are those of the longer prefix.
    pos = memfit.prefix_row_terms(row, 4096, 4608, 2)
    neg = memfit.prefix_row_terms(row, 4096, 77, 2)
    assert pos[1] > neg[1] and pos[2] > neg[2]
    for rows in ([4608, 77], [77, 4608]):
        g = plan_memory(q_view, 0, 0, cfg=4.0, prefix_rows=rows, **kw)
        assert g.rows == 2
        assert g.terms["prefix_cache"] == pos[0] + neg[0]
        assert g.terms["step_peak"] == max(
            row.step.bytes(pos[1], 2, row.prefix.join_bytes(4608, 2)),
            row.prefill.bytes(pos[2]))
    # A conditioning the request does not name counts as text rows, and
    # one it does not run is left out.
    h = plan_memory(q_view, 0, 0, cfg=4.0, prefix_rows=[4608], **kw)
    assert h.terms["prefix_cache"] == pos[0] + GiB // 4
    assert plan_memory(q_view, 0, 0, prefix_rows=[4608, 77],
                       **kw).terms == by_rows.terms
    # A family without a prefix has no rows to read.
    z = dict(height=1024, width=1024)
    assert _z(z_view, prefix_rows=[4608], **z).terms == _z(z_view, **z).terms


def test_other_prefix_states_stay_up_to_the_bound(q_view):
    kw = dict(family="qwen_image21", height=1024, width=1024,
              budget=BUDGET_128, overcommit=False)
    own = plan_memory(q_view, 0, 0, **kw).terms["prefix_cache"]
    assert own == GiB // 4

    def cache(other: int, bound: int) -> int:
        return plan_memory(q_view, 0, 0, prefix_other=other,
                           prefix_bound=bound, **kw).terms["prefix_cache"]

    # All of a small state stays, a large one is shed to the bound, and
    # nothing else stays once the request's own states pass the bound.
    assert cache(GiB // 8, GiB) == own + GiB // 8
    assert cache(2 * GiB, GiB) == GiB
    assert cache(2 * GiB, GiB // 8) == own
    assert cache(GiB, 0) == own


def test_the_reference_hint_counts_every_phase(q_view):
    """A request whose encode phase does not fit whatever its references
    are: the hint names no count and no area."""
    with pytest.raises(ImageMemoryError) as e:
        plan_memory(q_view, ENCODER, 0, family="qwen_image21", height=1024,
                    width=1024, refs=2, ref_px=1024, caption_tokens=10 ** 7,
                    budget=60 * GiB, overcommit=False)
    assert e.value.plan.phases["denoise"] <= 60 * GiB
    assert e.value.plan.phases["encode"] > 60 * GiB
    assert ("; the request does not fit without reference images either"
            in str(e.value))


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
    tiled = memfit.TiledFit(0.0, 1.0, 0.0)
    _z_row(monkeypatch,
           whole={"decode": {c: {2: fit, 4: fit}
                             for c in ("exact", "winograd")}},
           tiled={"decode": {2: tiled, 4: tiled}})

    def levers(gib, **kw):
        return _z(z_view, height=2048, width=2048, cfg=3.0,
                  budget=gib * GiB, **kw).levers

    rows, release, tiled, _ = memfit.LEVERS
    assert levers(20.0) == ()
    assert levers(17.0) == (rows,)
    assert levers(14.0) == (rows, release)
    assert levers(12.0) == (rows, release, tiled)
    assert levers(14.0, batch_rows=False) == (release,)


def test_tiled_decode_halves_the_decode_peak_at_2048(z_view):
    # Tiles hold their activations at each GroupNorm, so at 2048 the tiled
    # decode is still the largest phase, at under half the whole decode.
    whole = _z(z_view, height=2048, width=2048)
    assert whole.fits and not whole.levers
    in_tiles = memfit.vae_peak("decode", 2048, 2048, family="z_image",
                               itemsize=2, tiled=True)
    need = whole.phases["decode"] - whole.terms["decode_peak"] + in_tiles
    assert need > whole.phases["denoise"]
    tiled = _z(z_view, height=2048, width=2048, budget=need + 1)
    assert tiled.fits and tiled.tiled_decode
    assert tiled.terms["decode_peak"] == in_tiles
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
                 "decode_peak", "budget of 8.54 GiB"):
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
    with pytest.raises(ImageMemoryError, match="is not released in this request"):
        _z(z_view, height=1024, width=1024, budget=BUDGET_16,
           encoder_owned=False)


def _mmproj_view():
    """The header of a small mmproj: 16-bit and 32-bit float tensors and a
    quantized one."""
    from types import SimpleNamespace as T

    return T(tensors={
        "v.patch_embd.weight": T(shape=(64, 3, 16, 16), codec="F16",
                                 nbytes=64 * 3 * 256 * 2),
        "v.blk.0.attn.q.weight": T(shape=(64, 64), codec="BF16", nbytes=0),
        "v.blk.0.ln1.bias": T(shape=(64,), codec="F32", nbytes=64 * 4),
        "mm.0.weight": T(shape=(256, 64), codec="Q8_0", nbytes=0)})


def test_an_mmproj_counts_at_the_bytes_it_holds_once_loaded(q_view, z_view):
    """The load of a Qwen3-VL tower casts its float tensors to float32, so
    a 16-bit tensor is resident at twice its file bytes."""
    view = _mmproj_view()
    floats = 64 * 3 * 256 + 64 * 64 + 64
    q8 = memfit.tensor_bytes((256, 64), "Q8_0")
    on_file = (64 * 3 * 256 + 64 * 64) * 2 + 64 * 4 + q8
    assert memfit.file_bytes(view) == on_file
    assert memfit.mmproj_bytes(view) == on_file
    loaded = memfit.mmproj_bytes(view, fp32=True)
    assert loaded == floats * 4 + q8 > on_file
    assert memfit.mmproj_bytes(None) == 0 == memfit.mmproj_bytes(None,
                                                                 fp32=True)
    # A byte count is what the caller measured, and stands.
    assert memfit.mmproj_bytes(123, fp32=True) == 123
    kw = dict(height=1024, width=1024, budget=BUDGET_128, overcommit=False)
    assert memfit.PEAKS["qwen_image21"].mmproj_fp32 is True
    assert memfit.PEAKS["z_image"].mmproj_fp32 is False
    q = plan_memory(q_view, ENCODER, 0, view, family="qwen_image21", **kw)
    z = plan_memory(z_view, ENCODER, VAE, view, family="z_image", **kw)
    assert (q.terms["mmproj"], z.terms["mmproj"]) == (loaded, on_file)
    # It is resident in every phase of a plan that keeps the encoder.
    bare = plan_memory(q_view, ENCODER, 0, family="qwen_image21", **kw)
    assert {k: q.phases[k] - bare.phases[k] for k in q.phases} == dict.fromkeys(
        q.phases, loaded)
    assert q.weight_bytes == bare.weight_bytes + loaded


def test_the_release_of_the_encoder_frees_its_mmproj(q_view):
    mmproj = GiB // 2
    kw = dict(family="qwen_image21", height=1024, width=1024,
              overcommit=False)
    free = plan_memory(q_view, ENCODER, VAE, mmproj, budget=BUDGET_128, **kw)
    assert free.levers == () and free.peak == free.phases["decode"]
    p = plan_memory(q_view, ENCODER, VAE, mmproj, budget=free.peak - 1, **kw)
    assert p.levers == ("release the encoder",) and p.release_encoder
    # The encode phase holds both, and the phases after it neither.
    assert p.phases["encode"] == free.phases["encode"]
    for phase in ("denoise", "decode"):
        assert p.phases[phase] == free.phases[phase] - ENCODER - mmproj
    # An mmproj alone is a reason to release, and none is a lever only
    # when something is resident to free.
    alone = plan_memory(q_view, 0, VAE, mmproj, budget=free.peak - ENCODER
                        - 1, **kw)
    assert alone.levers == ("release the encoder",)
    assert alone.phases["decode"] == free.peak - ENCODER - mmproj
    nothing = plan_memory(q_view, 0, VAE, budget=free.peak - ENCODER - mmproj
                          - 1, **kw)
    assert nothing.levers == ("tiled decode",)


def test_each_qwen_lever_has_a_budget_that_forces_it(q_view):
    """The plan's order on Qwen-Image 2.1: sequential rows, release of the
    encoder, tiled decode, and then the refusal."""
    rows, release, tiled, _ = memfit.LEVERS
    mmproj = GiB // 2

    def plan(budget, **kw):
        kw.setdefault("height", 1024)
        kw.setdefault("width", 1024)
        return plan_memory(q_view, ENCODER, VAE, mmproj,
                           family="qwen_image21", budget=budget,
                           overcommit=False, **kw)

    # Sequential rows: a guided request above 1024 pixels, where the
    # decode is in tiles already and two rows of a step are the peak.
    big = dict(height=1536, width=1536, cfg=4.0)
    free = plan(BUDGET_128, **big)
    assert free.levers == () and free.rows == 2 and free.tiled_decode
    assert free.peak == free.phases["denoise"]
    p = plan(free.peak - 1, **big)
    assert p.levers == (rows,) and p.rows == 1 and not p.release_encoder
    assert p.terms["step_peak"] < free.terms["step_peak"]
    assert p.terms["prefix_cache"] == free.terms["prefix_cache"]
    # A request without guidance has one row, and no such lever.
    assert plan(BUDGET_128, height=1536, width=1536).rows == 1
    # Release of the encoder: the next lever of that request.
    q = plan(p.peak - 1, **big)
    assert q.levers == (rows, release) and q.release_encoder
    assert q.peak == p.peak - ENCODER - mmproj
    # Tiled decode: a request at 1024 pixels, where a whole decode is the
    # peak, with an encoder that is not released.
    free = plan(BUDGET_128, encoder_owned=False)
    assert free.levers == () and free.peak == free.phases["decode"]
    p = plan(free.peak - 1, encoder_owned=False)
    assert p.levers == (tiled,) and p.tiled_decode
    assert p.terms["decode_peak"] < free.terms["decode_peak"]
    # All three in order on a guided request at 1024 pixels, where a lever
    # is taken although the one before it left the plan over the budget.
    guided = plan(BUDGET_128, cfg=4.0)
    assert guided.levers == () and guided.peak == guided.phases["decode"]
    assert plan(guided.peak - 1, cfg=4.0).levers == (rows, release)
    after = guided.peak - ENCODER - mmproj
    last = plan(after - 1, cfg=4.0)
    assert last.levers == (rows, release, tiled)
    # The refusal, one byte under the last plan, names every term in GiB.
    with pytest.raises(ImageMemoryError) as e:
        plan(last.peak - 1, cfg=4.0)
    msg, refused = str(e.value), e.value.plan
    assert refused.levers == (rows, release, tiled) and not refused.fits
    assert refused.phases == last.phases
    for name in ("dit", "encoder", "vae", "mmproj", "prefix_cache",
                 "step_peak", "decode_peak", "encode_peak"):
        value = refused.terms[name]
        assert value > 0 and f"{name} {value / GiB:.2f} GiB" in msg, name
    # One byte apart, the peak rounds up and the budget down.
    up = math.ceil(last.peak / GiB * 100) / 100
    assert f"peak {up:.2f} GiB against a budget of {up - 0.01:.2f} GiB" in msg
    assert "levers sequential rows, release the encoder, tiled decode" in msg
    assert "GMLX_OVERCOMMIT=1 overrides" in msg


def test_overcommit_returns_the_plan(z_view, monkeypatch):
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    p = plan_memory(z_view, ENCODER, VAE, family="z_image", height=2048,
                    width=2048, budget=BUDGET_16)
    assert not p.fits


def test_the_weights_gate_refuses_over_budget_and_defers_to_capacity(
        z_view, monkeypatch):
    from gmlx.serve import capacity

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    seen = []
    monkeypatch.setattr(capacity, "preload_gate",
                        lambda b, m: seen.append((b, m)))
    p = _z(z_view, height=1024, width=1024, budget=BUDGET_16)
    memfit.gate_weights(p, "z.gguf")
    assert seen == [(float(p.weight_bytes), "z.gguf")]
    small = dataclasses.replace(p, budget=4 * GiB)
    with pytest.raises(ImageMemoryError, match=(
            r"The image model weights of 7\.\d\d GiB exceed the working "
            r"budget of 4\.00 GiB\. The largest term is dit; try a smaller "
            r"DiT quant\.")):
        memfit.gate_weights(small, "z.gguf")
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    memfit.gate_weights(small, "z.gguf")
    assert len(seen) == 1


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
    assert ("; the largest that fit are 9 of the 16 reference images at "
            "this reference area, or all 16 at a reference area of "
            "768 x 768 px") in msg
    # What the message names fits, and one step more does not.
    kw = dict(family="qwen_image21", height=1024, width=1024, cfg=4.0,
              budget=60 * GiB, overcommit=False)
    for refs, px, fits in ((9, 1024, True), (10, 1024, False),
                           (16, 768, True), (16, 800, False)):
        if fits:
            assert plan_memory(q_view, ENCODER, 0, refs=refs, ref_px=px,
                               **kw).fits
        else:
            with pytest.raises(ImageMemoryError):
                plan_memory(q_view, ENCODER, 0, refs=refs, ref_px=px, **kw)


def test_reference_refusal_reads_the_sizes_of_the_request(q_view):
    """The references of a request by their own sizes, both conditionings
    of a guided one counted, and the VAE encode of each priced."""
    refs = [(1024, 1024), (768, 1376), (1024, 1024)]
    kw = dict(family="qwen_image21", height=1024, width=1024,
              overcommit=False)
    fit = plan_memory(q_view, ENCODER, VAE, ref_sizes=refs,
                      budget=BUDGET_128, **kw)
    # Unguided the three fit a budget that the guided request passes.
    budget = fit.peak + GiB
    g = dict(kw, cfg=4.0, budget=budget)
    assert plan_memory(q_view, ENCODER, VAE, ref_sizes=refs, budget=budget,
                       **kw).fits

    def named(sizes, **over) -> tuple[int, int]:
        with pytest.raises(ImageMemoryError) as e:
            plan_memory(q_view, ENCODER, VAE, ref_sizes=sizes,
                        **{**g, **over})
        m = re.search(r"; the largest that fit are (\d+) of the 3 reference "
                      r"images at this reference area, or all 3 at a "
                      r"reference area of (\d+) x \2 px\. ", str(e.value))
        assert m, str(e.value)
        return int(m[1]), int(m[2])

    def at(side: int) -> list:
        return [(memfit._scaled(h, side / 1024),
                 memfit._scaled(w, side / 1024)) for h, w in refs]

    count, side = named(refs)
    assert 0 < count < 3 and 256 <= side < 1024 and side % 32 == 0
    # What the message names fits, and one step more does not.
    assert plan_memory(q_view, ENCODER, VAE, ref_sizes=refs[:count],
                       **g).fits
    assert plan_memory(q_view, ENCODER, VAE, ref_sizes=at(side), **g).fits
    for sizes in (refs[:count + 1], at(side + 32)):
        with pytest.raises(ImageMemoryError):
            plan_memory(q_view, ENCODER, VAE, ref_sizes=sizes, **g)
    # Without guidance the same budget holds one conditioning, and all
    # three fit: the guided refusal counted both.
    assert at(1024) == refs
    # A reference area under the family's smallest is never named.
    # The budget is just over the least the request takes without them.
    tight = BUDGET_128
    while True:
        try:
            tight = plan_memory(q_view, ENCODER, VAE, cfg=4.0,
                                budget=tight - 1, **kw).peak
        except ImageMemoryError:
            break
    tight += GiB // 32
    with pytest.raises(ImageMemoryError) as e:
        plan_memory(q_view, ENCODER, VAE, ref_sizes=refs,
                    **{**g, "budget": tight})
    assert ("the largest that fit are 0 of the 3 reference images at this "
            "reference area, or all 3 at no smaller reference area"
            ) in str(e.value)
    assert memfit.PEAKS["qwen_image21"].prefix.min_ref_px == 256


def test_fp32_vae_on_fp16_activations(z_view):
    p = _z(z_view, height=1024, width=1024, dtype="float16")
    assert p.terms["vae"] == VAE
    fp32 = memfit.vae_peak("decode", 1024, 1024, family="z_image",
                           itemsize=4)
    assert p.terms["decode_peak"] == fp32
    assert fp32 > memfit.vae_peak("decode", 1024, 1024, family="z_image",
                                  itemsize=2)
    vae16 = memfit.vae_bytes(_TinyVae(), 2)
    vae32 = memfit.vae_bytes(_TinyVae(), 4)
    assert vae32 == 2 * vae16 == 2 * 2 * 30


class _TinyVae:
    shapes = {"a": (2, 3), "b": (4, 6)}


def test_unknown_family_is_refused(z_view):
    with pytest.raises(ValueError, match="known: flux2_klein, flux3_klein, "
                                         "krea2, qwen_image21, z_image"):
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


# Footprints in GiB of one Z-Image-Turbo Q4_K_M step through api.sample
# under bounded_cache on an M5 Max, by rows. The step row must stay above
# every one.
MEASURED_STEP = {288: 0.54, 1056: 1.28, 4128: 2.19, 4608: 2.24}


def test_the_step_row_covers_the_measured_footprints():
    row = memfit.PEAKS["z_image"]
    for rows, gib in MEASURED_STEP.items():
        assert row.step.bytes(rows * row.width * 2) >= gib * GiB, rows


# The measured cells of one Qwen-Image 2.1 Q4_K_M request through
# api.sample under bounded_cache on an M5 Max: text requests and requests
# with reference images, on the stock ops and with the kernel routes in
# force. See the file's "what".
QWEN_CELLS = (Path(__file__).resolve().parents[1] / "fixtures" / "image"
              / "memfit" / "qwen_image21_cells.json")
# Two cells of 16384 image rows on the stock ops, by text rows: the
# footprint of a step over the settled process, state included.
MEASURED_QWEN_STEP_2048 = {512: 5.88, 36: 5.37}
ROOM = 0.25 * GiB               # what a fit leaves above every cell


def _qwen_cells() -> list[dict]:
    doc = json.loads(QWEN_CELLS.read_text())
    cells = []
    for c in doc["cells"]:
        image = (c["height"] // 16) * (c["width"] // 16)
        prefix = c["text_rows"] + c["refs"] * (c["ref_px"] // 16) ** 2
        conds = 2 if c["guided"] else 1
        cells.append(dict(c, image=image, prefix=prefix, conds=conds,
                          rows=c.get("rows", conds)))
    for text, gib in MEASURED_QWEN_STEP_2048.items():
        cells.append(dict(image=16384, prefix=text, conds=1, rows=1,
                          routes=[],
                          refs=0, ref_px=0, guided=False, steps_gib=[gib],
                          state_bytes=text * 2 * 32 * 4096 * 2,
                          prefill_gib=None))
    return cells


def _room(row, cell: dict) -> tuple[float, float]:
    """Bytes the row's plan leaves above the cell's largest step, and
    above its prefill alone."""
    state, step, prefill = memfit.prefix_row_terms(
        row, cell["image"], cell["prefix"], 2)
    assert state == cell["state_bytes"]
    n = cell["conds"]
    read = row.prefix.join_bytes(cell["prefix"], 2)
    plan = n * state + row.step_peak(step, prefill, cell["rows"], read)
    alone = math.inf
    if cell["prefill_gib"] is not None:
        alone = (state + (row.prefill or row.step).bytes(prefill)
                 - cell["prefill_gib"] * GiB)
    return plan - max(cell["steps_gib"]) * GiB, alone


def test_the_qwen_rows_cover_every_measured_cell():
    row = memfit.PEAKS["qwen_image21"]
    assert row.step is memfit.QWEN_IMAGE21_STEP
    assert row.prefill is memfit.QWEN_IMAGE21_PREFILL
    cells = _qwen_cells()
    # Both paths, text requests and requests with references, guided and
    # not, are among the cells, with one to four references.
    for routes, counts in ((True, (50, 34, 35)), (False, (93, 30, 80))):
        mine = [c for c in cells if bool(c["routes"]) is routes]
        assert (sum(1 for c in mine if not c["refs"]),
                sum(1 for c in mine if c["refs"] and c["guided"]),
                sum(1 for c in mine if c["refs"] and not c["guided"])) == \
            counts
        sizes = {(c["refs"], c["ref_px"]) for c in mine if c["refs"]}
        assert {(n, px) for n in (1, 2) for px in (256, 512, 1024)} | {
            (4, 1024)} <= sizes
        assert any(n == 3 for n, _ in sizes)
    # On the stock ops, one reference of 544 to 736 pixels, where a step
    # reads a long prefix.
    assert {px for c in cells if not c["routes"] and c["refs"] == 1
            for px in [c["ref_px"]]} >= set(range(544, 737, 32))
    # Two guided requests planned one row at a time.
    assert sum(1 for c in cells if c["conds"] == 2 and c["rows"] == 1) == 2
    for c in cells:
        step, alone = _room(row, c)
        assert step >= ROOM and alone >= ROOM, c
    # The rows are fits, not ceilings far above the readings: a text
    # request on the stock ops is priced within 1 GiB of its reading up
    # to 4096 image rows, and within 1.5 GiB at 16384.
    text = [c for c in cells if not c["refs"] and not c["routes"]]
    assert max(_room(row, c)[0] for c in text
               if c["image"] <= 4096) <= 1.25 * GiB
    assert max(_room(row, c)[0] for c in text) <= 1.5 * GiB
    assert min(_room(row, c)[0] for c in cells) <= 0.35 * GiB


KREA2_CELLS = QWEN_CELLS.with_name("krea2_cells.json")


def test_the_krea2_row_covers_every_measured_step():
    row = memfit.PEAKS["krea2"]
    assert row.step is memfit.KREA2_STEP
    cells = json.loads(KREA2_CELLS.read_text())["cells"]
    assert {c["size"] for c in cells} >= {256, 512, 768, 1024, 1280, 1536}
    assert sum(c["guided"] for c in cells) == 2
    room = []
    for c in cells:
        rows = memfit.image_rows(row, c["size"], c["size"], c["text_rows"])
        plan = row.step.bytes(rows * row.width * 2, 2 if c["guided"] else 1)
        room.append(plan - max(c["steps_gib"]) * GiB)
    assert min(room) >= ROOM
    # A fit, not a ceiling: the closest cell is within 0.35 GiB, and none
    # is priced more than 1.75 GiB above its reading.
    assert min(room) <= 0.35 * GiB and max(room) <= 1.75 * GiB


FLUX3_CELLS = QWEN_CELLS.with_name("flux3_klein_cells.json")


def _flux3_room(row, c: dict) -> float:
    """Bytes the row's step leaves above the cell's largest step: the
    image rows, the text rows and the tokens of the references."""
    rows = memfit.image_rows(row, c["size"], c["size"], c["text_rows"])
    rows += c.get("refs", 0) * (c.get("ref_px", 0) // row.ref_token_px) ** 2
    plan = row.step.bytes(rows * row.width * 2, 2 if c["guided"] else 1)
    return plan - max(c["steps_gib"]) * GiB


def test_the_flux3_row_covers_every_measured_step():
    row = memfit.PEAKS["flux3_klein"]
    assert row.step is memfit.FLUX3_KLEIN_STEP
    cells = json.loads(FLUX3_CELLS.read_text())["cells"]
    assert {c["size"] for c in cells} >= {256, 512, 768, 1024, 1280, 1536,
                                          2048}
    assert {c["text_rows"] for c in cells} == {80, 640, 4096}
    assert sum(c["guided"] for c in cells) == 4
    assert sum(bool(c.get("refs")) for c in cells) == 3
    room = [_flux3_room(row, c) for c in cells]
    assert min(room) >= ROOM
    # A fit, not a ceiling: the closest cell is within 0.35 GiB, and none
    # is priced more than 1.25 GiB above its reading.
    assert min(room) <= 0.35 * GiB and max(room) <= 1.25 * GiB


def test_a_thinner_flux3_row_misses_a_measured_cell():
    row = memfit.PEAKS["flux3_klein"]
    cells = json.loads(FLUX3_CELLS.read_text())["cells"]

    def worst(**change) -> float:
        thin = dataclasses.replace(row, step=dataclasses.replace(
            row.step, **change))
        return min(_flux3_room(thin, c) for c in cells)

    assert worst() >= ROOM
    assert worst(base=2.35 * GiB) < ROOM
    assert worst(k=2.7) < ROOM
    # The second row of a guided step at Krea 2's slope.
    assert worst(row_k=0.5) < ROOM
    # Without the reference tokens in the rows, two 1024 pixel references
    # are priced under their reading.
    two = next(c for c in cells if c.get("refs") == 2)
    alone = dataclasses.replace(row, ref_token_px=10 ** 9)
    assert _flux3_room(alone, two) < ROOM


def test_the_flux3_plan_counts_the_references_and_the_most_text():
    """The plan prices the most text rows the encoder gives, and each
    reference's tokens in every step."""
    row = memfit.PEAKS["flux3_klein"]
    kw = dict(family="flux3_klein", height=1024, width=768,
              budget=BUDGET_128, overcommit=False)
    item = 2

    def step(rows, conds=1):
        return row.step.bytes(rows * row.width * item, conds)

    text = 64 * 48 + 4096
    p = plan_memory(0, 0, 0, **kw)
    assert p.terms["step_peak"] == step(text)
    refs = [(1024, 1024), (512, 768)]
    p = plan_memory(0, 0, 0, ref_sizes=refs, **kw)
    assert p.terms["step_peak"] == step(text + 64 * 64 + 32 * 48)
    p = plan_memory(0, 0, 0, refs=2, ref_px=512, cfg=3.5, **kw)
    assert p.rows == 2 and p.terms["step_peak"] == step(text + 2 * 32 * 32,
                                                        2)
    # A caption the caller knows is priced at its own rows.
    p = plan_memory(0, 0, 0, caption_tokens=80, **kw)
    assert p.terms["step_peak"] == step(64 * 48 + 80)


def test_a_flux3_request_plan_prices_the_text_rows_of_its_prompt():
    """The conditioning of a request gives the text rows of every step and
    of its encode, the more of a guided pair's, and a plan without one
    prices the family's most."""
    import mlx.core as mx

    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning
    from gmlx.image.registry import get_family

    row = memfit.PEAKS["flux3_klein"]
    dit = SimpleNamespace(act_dtype=mx.bfloat16, cfg=SimpleNamespace(patch=1))
    pipe = api.ImagePipeline(dit=dit, family=get_family("flux3_klein"),
                             variant="distilled")
    kw = dict(height=1024, width=768, budget=BUDGET_128, overcommit=False)

    def cond(rows):
        return Conditioning(mx.zeros((rows, 4)))

    def step(rows, conds=1):
        return row.step.bytes(rows * row.width * 2, conds)

    image = 64 * 48
    p = api.plan_request(pipe, conds=cond(80), cfg=1.0, **kw)
    assert p.terms["step_peak"] == step(image + 80)
    assert p.terms["encode_peak"] == int(
        row.encode_base + 80 * row.encode_row_bytes * memfit.ENCODE_LIVE)
    p = api.plan_request(pipe, conds=[cond(160), cond(80)], cfg=3.5, **kw)
    assert p.rows == 2 and p.terms["step_peak"] == step(image + 160, 2)
    assert api.plan_request(pipe, cfg=1.0, **kw).terms[
        "step_peak"] == step(image + 4096)
    p = api.plan_request(pipe, conds=cond(80), cfg=1.0, caption_tokens=640,
                         **kw)
    assert p.terms["step_peak"] == step(image + 640)


SHORT = ("z_image", "krea2", "qwen_image21", "flux3_klein")


def _short_cells(family: str) -> list[dict]:
    """The measured cells of a short prompt on ``family``: the size, the
    text rows of each conditioning, its references, and the GiB of the
    largest step and of a prefill alone where one was read."""
    def cell(h, w, text, gib, refs=0, px=0, guided=False, prefill=None,
             state=None):
        return dict(height=h, width=w, text=text, refs=refs, ref_px=px,
                    guided=guided, gib=gib, prefill=prefill, state=state)

    if family == "z_image":
        # Every cell but the one of 4608 rows ran a caption of at most 32
        # tokens.
        return [cell(s, s, 22, MEASURED_STEP[rows])
                for s, rows in ((256, 288), (512, 1056), (1024, 4128))]
    if family == "qwen_image21":
        out = []
        for c in _qwen_cells():
            text = c["prefix"] - c["refs"] * (c["ref_px"] // 16) ** 2
            if text != 36:
                continue
            side = math.isqrt(c["image"]) * 16
            out.append(cell(c.get("height", side), c.get("width", side), text,
                            max(c["steps_gib"]), c["refs"], c["ref_px"],
                            c["guided"], c["prefill_gib"], c["state_bytes"]))
        return out
    path, short = {"krea2": (KREA2_CELLS, 36),
                   "flux3_klein": (FLUX3_CELLS, 80)}[family]
    return [cell(c["size"], c["size"], c["text_rows"], max(c["steps_gib"]),
                 c.get("refs", 0), c.get("ref_px", 0), c["guided"])
            for c in json.loads(path.read_text())["cells"]
            if c["text_rows"] == short]


def _short_plan(family: str, c: dict) -> memfit.MemoryPlan:
    """The plan of the cell's request on ``family`` from conditionings of
    its text rows, as a request plans after its encode."""
    import mlx.core as mx

    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning
    from gmlx.image.registry import get_family

    fam = get_family(family)
    scale = int(fam.constants.values["vae_scale_factor"])
    dit = SimpleNamespace(act_dtype=mx.bfloat16, cfg=SimpleNamespace(
        patch=memfit.PEAKS[family].token_px // scale))
    pipe = api.ImagePipeline(dit=dit, family=fam, variant="")
    refs = tuple(SimpleNamespace(height=c["ref_px"], width=c["ref_px"],
                                 sha256=str(i)) for i in range(c["refs"]))
    conds = [Conditioning(mx.zeros((c["text"], 4)), refs=refs)
             for _ in range(2 if c["guided"] else 1)]
    return api.plan_request(pipe, height=c["height"], width=c["width"],
                            cfg=3.5 if c["guided"] else 1.0, conds=conds,
                            budget=BUDGET_128, overcommit=False)


def _short_room(family: str, c: dict) -> float:
    """Bytes the plan of the cell's request holds over its largest step
    and over its prefill: its prefix states and its step term."""
    p = _short_plan(family, c)
    if c["state"] is not None:
        assert p.terms["prefix_cache"] == (2 if c["guided"] else 1) * c[
            "state"]
    held = p.terms["prefix_cache"] + p.terms["step_peak"]
    return held - max(c["gib"], c["prefill"] or 0) * GiB


@pytest.mark.parametrize("family", SHORT)
def test_a_short_prompt_plan_covers_every_measured_cell(family):
    """A request plan made from the conditionings of a short prompt holds
    ROOM above every measured cell of that prompt. Z-Image's plan prices
    the most caption rows, since at its own rows it holds less than
    ROOM."""
    row = memfit.PEAKS[family]
    cells = _short_cells(family)
    assert len(cells) == {"z_image": 3, "krea2": 3, "qwen_image21": 79,
                          "flux3_klein": 13}[family]
    room = [_short_room(family, c) for c in cells]

    def formula(c, caption):
        rows = memfit.image_rows(row, c["height"], c["width"], caption)
        if row.ref_token_px:
            rows += c["refs"] * (c["ref_px"] // row.ref_token_px) ** 2
        return rows * row.width * 2

    if family == "z_image":
        assert not memfit.prices_prompt(family)
        for c in cells:
            assert _short_plan(family, c).terms["step_peak"] == \
                row.step.bytes(formula(c, None))
        own = [row.step.bytes(formula(c, c["text"])) - c["gib"] * GiB
               for c in cells]
        assert min(own) < ROOM
    else:
        assert memfit.prices_prompt(family)
        # A step that joins the text rows is priced at the cell's own.
        for c in cells if row.caption_rows else ():
            assert _short_plan(family, c).terms["step_peak"] == \
                row.step.bytes(formula(c, c["text"]),
                               2 if c["guided"] else 1)
    assert min(room) >= ROOM


def test_the_flux3_encode_term_covers_every_measured_encode():
    row = memfit.PEAKS["flux3_klein"]
    encodes = json.loads(FLUX3_CELLS.read_text())["encode"]
    assert [e["rows"] for e in encodes] == [80, 640, 2080, 4096]
    kw = dict(family="flux3_klein", height=512, width=512,
              budget=BUDGET_128, overcommit=False)
    room = []
    for e in encodes:
        p = plan_memory(0, ENCODER, 0, caption_tokens=e["rows"], **kw)
        assert p.terms["encode_peak"] == int(
            row.encode_base
            + e["rows"] * row.encode_row_bytes * memfit.ENCODE_LIVE)
        room.append(p.terms["encode_peak"] - e["gib"] * GiB)
    assert min(room) >= ROOM
    # The plan of an unknown prompt prices the most rows, under 2.5 GiB,
    # where Krea 2's rows alone would price 7.1 GiB.
    p = plan_memory(0, ENCODER, 0, **kw)
    assert p.terms["encode_peak"] == int(
        row.encode_base + 4096 * row.encode_row_bytes * memfit.ENCODE_LIVE)
    assert p.terms["encode_peak"] < 2.5 * GiB
    assert 4096 * memfit.PEAKS["krea2"].encode_row_bytes \
        * memfit.ENCODE_LIVE > 7 * GiB
    # Without the base the shortest prompt is priced under its reading.
    assert 80 * row.encode_row_bytes * memfit.ENCODE_LIVE < \
        encodes[0]["gib"] * GiB


FLUX2_CELLS = QWEN_CELLS.with_name("flux2_klein_cells.json")


def test_the_flux2_row_covers_every_measured_step():
    """The 4B cells: the 512 text rows of every prompt, sizes to 2048
    pixels, one, two and four 1024 pixel references and guided Base
    steps. Their rows count as FLUX.3 klein's do."""
    row = memfit.PEAKS["flux2_klein"]
    assert row.step is memfit.FLUX2_KLEIN_4B_STEP
    assert row.width == 36864 and row.caption_rows == row.text_rows == 512
    assert not memfit.prices_prompt("flux2_klein")
    cells = json.loads(FLUX2_CELLS.read_text())["cells"]
    assert {c["size"] for c in cells} == {256, 512, 768, 1024, 1280, 1536,
                                          2048}
    assert {c["text_rows"] for c in cells} == {512}
    assert {c["size"] for c in cells if c["guided"]} == {512, 1024, 1536}
    assert sorted(c["refs"] for c in cells if c.get("refs")) == [1, 2, 4]
    assert {c["ref_px"] for c in cells if c.get("refs")} == {1024}
    room = [_flux3_room(row, c) for c in cells]
    assert min(room) >= ROOM
    # A fit, not a ceiling: the closest cell is within 0.35 GiB, and none
    # is priced more than 1.25 GiB above its reading.
    assert min(room) <= 0.35 * GiB and max(room) <= 1.25 * GiB


def test_a_thinner_flux2_row_misses_a_measured_cell():
    row = memfit.PEAKS["flux2_klein"]
    cells = json.loads(FLUX2_CELLS.read_text())["cells"]

    def worst(**change) -> float:
        thin = dataclasses.replace(row, step=dataclasses.replace(
            row.step, **change))
        return min(_flux3_room(thin, c) for c in cells)

    assert worst() >= ROOM
    assert worst(base=2.52 * GiB) < ROOM
    assert worst(k=3.0) < ROOM
    # Without the reference tokens in the rows, two 1024 pixel references
    # are priced under their reading.
    two = next(c for c in cells if c.get("refs") == 2)
    alone = dataclasses.replace(row, ref_token_px=10 ** 9)
    assert _flux3_room(alone, two) < ROOM


def test_the_flux2_plan_counts_the_references_and_its_text_rows():
    """Every prompt is 512 text rows, so the plan prices them whatever the
    caller knows, and each reference's tokens in every step."""
    row = memfit.PEAKS["flux2_klein"]
    kw = dict(family="flux2_klein", height=1024, width=768,
              budget=BUDGET_128, overcommit=False)

    def step(rows, conds=1):
        return row.step.bytes(rows * row.width * 2, conds)

    text = 64 * 48 + 512
    assert plan_memory(0, 0, 0, **kw).terms["step_peak"] == step(text)
    p = plan_memory(0, 0, 0, ref_sizes=[(1024, 1024), (512, 768)], **kw)
    assert p.terms["step_peak"] == step(text + 64 * 64 + 32 * 48)
    p = plan_memory(0, 0, 0, refs=2, ref_px=512, cfg=4.0, **kw)
    assert p.rows == 2 and p.terms["step_peak"] == step(text + 2 * 32 * 32,
                                                        2)


def test_the_flux2_encode_term_covers_every_measured_encode():
    """An encode is 512 rows whatever the prompt, on either encoder file."""
    row = memfit.PEAKS["flux2_klein"]
    encodes = json.loads(FLUX2_CELLS.read_text())["encode"]
    assert {e["rows"] for e in encodes} == {512}
    assert {e["encoder"] for e in encodes} == {"Qwen3-4B-Q8_0.gguf",
                                               "Qwen3-4B-Q4_K_M.gguf"}
    p = plan_memory(0, ENCODER, 0, family="flux2_klein", height=512,
                    width=512, budget=BUDGET_128, overcommit=False)
    assert p.terms["encode_peak"] == int(
        row.encode_base + 512 * row.encode_row_bytes * memfit.ENCODE_LIVE)
    room = [p.terms["encode_peak"] - e["gib"] * GiB for e in encodes]
    assert min(room) >= ROOM
    # Without the base every encode is priced under its reading.
    assert 512 * row.encode_row_bytes * memfit.ENCODE_LIVE < min(
        e["gib"] for e in encodes) * GiB


def test_a_thinner_qwen_row_misses_a_measured_cell():
    row = memfit.PEAKS["qwen_image21"]
    cells = _qwen_cells()

    def worst(**change) -> float:
        thin = dataclasses.replace(row, **change)
        return min(min(_room(thin, c)) for c in cells)

    assert worst() >= ROOM
    step = row.step

    def thin(**change) -> float:
        return worst(step=dataclasses.replace(step, **change))

    # The step base the stock ops read over.
    assert thin(base=1.1 * GiB) < ROOM and thin(base=1.0 * GiB) < 0.1 * GiB
    # Without the term of the prefix a step reads, a step on the stock ops
    # at 736x736 with 1000 text rows and one 640 pixel reference is over
    # its plan; with half the term or a lower bound there is no room.
    assert thin(read_k=0.0) < 0
    assert thin(read_k=8.0) < 0.05 * GiB
    assert thin(read_cap=0.6 * GiB) < ROOM
    # The second row of a guided step is not free.
    assert thin(row_k=0.0) < ROOM
    # The prefill fit of another family, the step's own fit, or none
    # leave the first step after the prefill of one 1024 pixel reference
    # less than the room.
    assert worst(prefill=memfit.Z_IMAGE_STEP) < ROOM
    assert worst(prefill=memfit.StepFit(1.25 * GiB, 5.5)) < ROOM
    assert worst(prefill=None) < ROOM
    # Both under the rows that were in force before these cells.
    assert worst(step=memfit.StepFit(1.0 * GiB, 4.0),
                 prefill=memfit.Z_IMAGE_STEP) < -0.1 * GiB


def test_the_one_row_lever_frees_what_its_cells_measured():
    """A guided request planned one row at a time reads 0.07 and 0.13 GiB
    under its two rows together. The plan credits the lever within
    0.1 GiB of that, and both plans stay above their cells."""
    row = memfit.PEAKS["qwen_image21"]
    cells = _qwen_cells()
    pairs = 0
    for one in (c for c in cells if c["conds"] == 2 and c["rows"] == 1):
        boths = [c for c in cells if c["rows"] == 2 and all(
            c[k] == one[k] for k in ("width", "height", "text_rows", "refs",
                                     "ref_px", "routes"))]
        assert boths
        both = max(boths, key=lambda c: max(c["steps_gib"]))
        saved = (max(both["steps_gib"]) - max(one["steps_gib"])) * GiB
        assert 0.05 * GiB < saved < 0.15 * GiB
        _, step, prefill = memfit.prefix_row_terms(row, one["image"],
                                                    one["prefix"], 2)
        read = row.prefix.join_bytes(one["prefix"], 2)
        credit = (row.step_peak(step, prefill, 2, read)
                  - row.step_peak(step, prefill, 1, read))
        assert abs(credit - saved) < 0.1 * GiB, (credit / GiB, saved / GiB)
        assert min(_room(row, one)) >= ROOM and min(_room(row, both)) >= ROOM
        pairs += 1
    assert pairs == 2


def test_a_reference_token_holds_its_cache_bytes():
    """Every measured state is 2 x 32 x 4096 elements a prefix row in
    bf16, whatever the references."""
    row = memfit.PEAKS["qwen_image21"]
    per_row = 2 * 32 * 4096 * 2
    assert row.prefix.cache_bytes(1, 2) == per_row == 524288
    seen = set()
    for c in _qwen_cells():
        assert c["state_bytes"] == c["prefix"] * per_row
        seen.add((c["refs"], c["ref_px"]))
    assert {(2, 1024), (4, 1024), (0, 0)} <= seen
    assert row.prefix.rows(512, 2, 1024) == 512 + 2 * 4096


def test_a_prefill_keeps_its_own_fit(q_view):
    row = memfit.PEAKS["qwen_image21"]
    assert row.prefill.k > row.step.k and row.prefill.base >= row.step.base
    assert row.step_peak(GiB, 0) == row.step.bytes(GiB)
    assert row.step_peak(0, GiB) == row.prefill.bytes(GiB)
    assert row.step_peak(GiB, GiB) == row.prefill.bytes(GiB)
    # Nine 1024 pixel references under guidance: the prefill is the term.
    kw = dict(family="qwen_image21", height=1024, width=1024, cfg=4.0,
              budget=BUDGET_128, overcommit=False)
    p = plan_memory(q_view, 0, 0, refs=9, ref_px=1024, **kw)
    _, step, prefill = memfit.prefix_terms(row, 4096, 512, 9, 1024, 2)
    assert p.terms["step_peak"] == row.prefill.bytes(prefill)
    assert row.prefill.bytes(prefill) > row.step.bytes(2 * step)
    # A row without a prefill fit prices both by its step fit.
    z = memfit.PEAKS["z_image"]
    assert z.prefill is None
    assert z.step_peak(GiB, 2 * GiB) == z.step.bytes(2 * GiB)
    assert z.step_peak(GiB, 0) == z.step.bytes(GiB)


# MLX peak in GiB of one Qwen3-VL-8B Q4_K_M encode over the encoder with
# its tower, on an M5 Max in fresh processes, by (references, reference
# side, feature rows). The tower ran on the GPU with split float32
# products, one block at a time.
MEASURED_QWEN_ENCODE = {(0, 0, 614): 0.51, (1, 256, 82): 0.46,
                        (2, 256, 754): 0.67, (1, 512, 274): 0.38,
                        (2, 512, 1138): 0.77, (1, 1024, 1042): 1.25,
                        (2, 1024, 2674): 1.32}


def test_the_encode_phase_counts_the_image_rows_and_the_tower(q_view):
    row = memfit.PEAKS["qwen_image21"]
    kw = dict(family="qwen_image21", height=1024, width=1024,
              budget=BUDGET_128, overcommit=False)
    for (refs, side, rows), gib in MEASURED_QWEN_ENCODE.items():
        slots = refs * (side // 32) ** 2
        p = plan_memory(q_view, ENCODER, 0, caption_tokens=rows - slots,
                        ref_sizes=[(side, side)] * refs, **kw)
        assert p.terms["encode_peak"] == int(
            rows * row.encode_row_bytes * memfit.ENCODE_LIVE
            + (row.encode_tower_bytes if refs else 0))
        if refs:
            assert p.terms["encode_peak"] >= gib * GiB + ROOM, (refs, side)
        else:
            assert p.terms["encode_peak"] >= gib * GiB
    # Without the tower term the encode of one 1024 pixel reference is
    # under its reading.
    assert 1042 * row.encode_row_bytes * memfit.ENCODE_LIVE < 1.25 * GiB
    # A family without references has no such term.
    assert memfit.PEAKS["z_image"].encode_tower_bytes == 0


def _sized(**tensors):
    """A header view of tensors with their bytes."""
    return SimpleNamespace(tensors={
        name: SimpleNamespace(nbytes=int(n), shape=(), codec="F32")
        for name, n in tensors.items()})


# MLX peak in GiB of each part's load over what stays resident, on the
# Qwen-Image 2.1 files in fresh processes on an M5 Max.
MEASURED_LOAD = {"encoder": 0.478, "encoder and mmproj": 1.550,
                 "vae gguf to bf16": 1.331, "vae gguf to float32": 1.200,
                 "vae safetensors to bf16": 0.711}
HEAD_BYTES = 510_504_960        # output.weight of the Qwen3-VL-8B Q4_K_M
MMPROJ_FILE = 1_162_548_160     # the tensors of its BF16 mmproj
VAE_ELEMENTS = 337_740_404      # of the Qwen-Image 2.1 VAE


def test_the_load_term_covers_the_measured_loads():
    enc = _sized(**{"output.weight": HEAD_BYTES, "blk.0.w": 4 << 30})
    mm = _sized(**{"v.w": MMPROJ_FILE})
    vae = SimpleNamespace(shapes={"w": (VAE_ELEMENTS,)})
    t, room = memfit.load_transient, memfit.LOAD_ROOM
    assert t() == 0 and t(5 << 30, 1 << 30, 2 << 30) == 0
    assert t(enc) == int(room * HEAD_BYTES)
    assert t(enc) >= MEASURED_LOAD["encoder"] * GiB
    assert t(enc, None, mm) == int(room * (HEAD_BYTES + MMPROJ_FILE))
    assert t(enc, None, mm) >= MEASURED_LOAD["encoder and mmproj"] * GiB
    # An mmproj without its encoder loads nothing.
    assert t(None, None, mm) == 0
    assert t(None, vae) == int(room * VAE_ELEMENTS * 4)
    assert t(None, vae) >= max(v for k, v in MEASURED_LOAD.items()
                               if k.startswith("vae")) * GiB
    # The bytes named alone are under two of the readings.
    assert HEAD_BYTES < MEASURED_LOAD["encoder"] * GiB
    assert VAE_ELEMENTS * 4 < MEASURED_LOAD["vae gguf to bf16"] * GiB
    # The parts load one after the other: the larger of the two.
    assert t(enc, vae, mm) == max(t(enc, None, mm), t(None, vae))


def test_the_dit_load_term_is_the_16_bit_tensors_it_casts():
    view = SimpleNamespace(tensors={
        "a": SimpleNamespace(shape=(), codec="F16", nbytes=8 << 20),
        "b": SimpleNamespace(shape=(), codec="BF16", nbytes=4 << 20),
        "c": SimpleNamespace(shape=(), codec="Q4_K", nbytes=2 << 20),
        "d": SimpleNamespace(shape=(), codec="F32", nbytes=1 << 20)})
    assert memfit.cast_bytes(view, "bfloat16") == 8 << 20
    assert memfit.cast_bytes(view, "float16") == 4 << 20
    assert memfit.cast_bytes(1 << 30, "bfloat16") == 0
    assert memfit.cast_bytes(None, "bfloat16") == 0
    assert memfit.dit_transient(view, "bf16") == int(
        memfit.LOAD_ROOM * (8 << 20))


def test_a_plan_made_before_the_load_has_a_load_phase(q_view):
    kw = dict(family="qwen_image21", height=512, width=512,
              overcommit=False)
    loaded = plan_memory(q_view, ENCODER, VAE, budget=BUDGET_128, **kw)
    assert "load" not in loaded.phases and "load_peak" not in loaded.terms
    peak = int(1.4 * GiB)
    p = plan_memory(q_view, ENCODER, VAE, load_peak=peak, budget=BUDGET_128,
                    **kw)
    assert list(p.phases)[0] == "load"
    assert p.phases["load"] == p.weight_bytes + peak
    assert {k: v for k, v in p.phases.items() if k != "load"} == \
        loaded.phases
    # The release of the encoder is no lever of the load: a request whose
    # other phases fit with it is refused when its load does not.
    released = plan_memory(q_view, ENCODER, VAE,
                           budget=loaded.weight_bytes + GiB, **kw)
    assert released.fits and released.release_encoder
    with pytest.raises(ImageMemoryError) as e:
        plan_memory(q_view, ENCODER, VAE, load_peak=peak,
                    budget=loaded.weight_bytes + GiB, **kw)
    assert "load_peak 1.40 GiB" in str(e.value)
    assert e.value.plan.peak == e.value.plan.phases["load"]


def _refused(*args, **kw):
    """The plan of a request that fits no budget, with every lever on."""
    with pytest.raises(ImageMemoryError) as e:
        plan_memory(*args, budget=1, **kw)
    return e.value.plan


def test_the_staged_load_is_the_last_lever_and_prices_each_phase(q_view):
    """Stub sizes on the Qwen-Image 2.1 header: the staged load is taken
    only when the plan does not fit without it, each of its phases has its
    price, and the refusal comes after it."""
    mm, adapters, caches = GiB // 2, 300 << 20, 100 << 20
    parts = (int(0.6 * GiB), int(0.3 * GiB))
    kw = dict(family="qwen_image21", height=512, width=512, cfg=4.0,
              adapters=adapters, cache_bytes=caches, overcommit=False,
              load_peak=max(parts))
    args = (q_view, ENCODER, VAE, mm)
    dit = memfit.file_bytes(q_view)
    whole = _refused(*args, **kw)
    staged = _refused(*args, load_parts=parts, **kw)
    assert not whole.staged_load and whole.levers == memfit.LEVERS[:2]
    assert staged.staged_load and staged.levers == (
        *memfit.LEVERS[:2], "staged load")
    assert staged.release_encoder and staged.rows == 1
    assert staged.peak < whole.peak == whole.phases["load"]
    # A request that fits without the lever has the plan it has without
    # the parts, at every budget down to the last byte that fits.
    for budget in (BUDGET_128, 20 * GiB, whole.peak + GiB, whole.peak):
        p = plan_memory(*args, budget=budget, load_parts=parts, **kw)
        assert p == plan_memory(*args, budget=budget, **kw)
        assert p.fits and not p.staged_load and "staged load" not in p.levers
        assert "load_encoder" not in p.phases
    # One byte under it the unstaged plan is refused, and the staged one
    # fits.
    with pytest.raises(ImageMemoryError):
        plan_memory(*args, budget=whole.peak - 1, **kw)
    p = plan_memory(*args, budget=whole.peak - 1, load_parts=parts, **kw)
    assert p.fits and p.staged_load and p.levers[-1] == "staged load"
    assert list(p.phases) == ["load_encoder", "encode", "load", "denoise",
                              "decode"]
    t = p.terms
    first = ENCODER + mm + adapters + caches
    # Each load phase holds the room of a staged load over its part.
    room = memfit.STAGE_ROOM
    assert room >= GiB // 10
    assert t["load_encoder_peak"] == parts[0] + room
    assert t["load_peak"] == parts[1] + room
    assert p.phases["load_encoder"] == first + parts[0] + room
    assert p.phases["encode"] == first + t["encode_peak"]
    assert p.phases["load"] == (dit + VAE + adapters + caches + parts[1]
                                + room)
    # After the release the phases are those of the unstaged plan.
    assert {k: p.phases[k] for k in ("denoise", "decode")} == {
        k: whole.phases[k] for k in ("denoise", "decode")}
    assert p.phases["denoise"] == (dit + VAE + adapters + caches
                                   + t["prefix_cache"] + t["step_peak"])
    # The refusal comes after the staged load, and names its phases.
    with pytest.raises(ImageMemoryError) as e:
        plan_memory(*args, budget=staged.peak - 1, load_parts=parts, **kw)
    msg = str(e.value)
    assert e.value.plan.staged_load and e.value.plan.phases == staged.phases
    assert "levers sequential rows, release the encoder, staged load" in msg
    assert f"load_encoder {staged.phases['load_encoder'] / GiB:.2f}" in msg
    for name in ("load_encoder_peak", "load_peak"):
        assert f"{name} {staged.terms[name] / GiB:.2f} GiB" in msg
    # An encoder the request keeps has no such lever.
    kept = _refused(*args, load_parts=parts, encoder_owned=False, **kw)
    assert not kept.staged_load and "staged load" not in kept.levers
    # An init image and a reference keep their phase after the load.
    edit = _refused(*args, load_parts=parts, ref_sizes=[(256, 256)], **kw)
    assert list(edit.phases) == ["load_encoder", "encode", "load", "init",
                                 "denoise", "decode"]
    assert edit.phases["init"] == (dit + VAE + adapters + caches
                                   + edit.terms["prefix_cache"]
                                   + edit.terms["init_peak"])


# The Qwen-Image 2.1 files beside the Q4_K_M DiT of the header fixture:
# the encoder without its output matrix, the mmproj as loaded, the VAE in
# bf16, and what the load of each part holds over that, in GiB.
Q_PARTS = dict(encoder=4.20, mmproj=2.15, vae=0.63, load_text=0.52,
               load_mmproj=1.71, load_vae=1.38)


def _q16(q_view, size, cfg, refs=(), budget=BUDGET_16, overcommit=False):
    g = {k: int(v * GiB) for k, v in Q_PARTS.items()}
    first = g["load_mmproj"] if refs else g["load_text"]
    return plan_memory(
        q_view, g["encoder"], g["vae"], g["mmproj"] if refs else 0,
        family="qwen_image21", height=size, width=size, cfg=cfg,
        ref_sizes=[(r, r) for r in refs], budget=budget,
        overcommit=overcommit,
        load_peak=max(first, g["load_vae"]),
        load_parts=(first, g["load_vae"]))


@pytest.mark.parametrize("size, cfg, refs", [
    (512, 1.0, ()), (512, 4.0, ()), (1024, 1.0, ()), (1024, 4.0, ()),
    (512, 1.0, (256,)), (512, 4.0, (256,)), (1024, 4.0, (256,))])
def test_a_16_gib_machine_runs_qwen_image_staged(q_view, size, cfg, refs):
    """What a 16 GB Mac runs of Qwen-Image 2.1 at Q4_K_M: no request fits
    with every part loaded together, and these fit staged."""
    p = _q16(q_view, size, cfg, refs)
    assert p.fits and p.staged_load and p.release_encoder
    assert p.tiled_decode is (size > 512)
    assert p.rows == 1
    # On a budget that holds every part the same request loads them
    # together.
    free = _q16(q_view, size, cfg, refs, budget=BUDGET_128)
    assert free.fits and not free.staged_load and not free.levers


@pytest.mark.parametrize("size, cfg, refs, phase, peak", [
    (512, 4.0, (512,), "decode", 8.92), (1024, 4.0, (512,), "denoise", 9.14),
    (512, 1.0, (512,), "encode", 8.69), (1024, 1.0, (1024,), "denoise",
                                         10.70)])
def test_a_16_gib_machine_refuses_the_larger_qwen_edits(q_view, size, cfg,
                                                        refs, phase, peak):
    with pytest.raises(ImageMemoryError) as e:
        _q16(q_view, size, cfg, refs)
    p = e.value.plan
    assert p.staged_load and p.levers[-1] == "staged load"
    assert max(p.phases, key=p.phases.get) == phase
    assert abs(p.peak / GiB - peak) < 0.02
    assert "against a budget of 8.54 GiB" in str(e.value)


# MLX peak in GiB of each phase of staged runs of those files under the
# budget of a 16 GiB machine, on an M5 Max, 4 steps, seed 1, by (size,
# guidance, references). The requests the plan refuses ran under
# GMLX_OVERCOMMIT=1. The step and VAE terms are footprints under the
# cache bound, which an MLX peak leaves out, so those phases read well
# under their plans.
MEASURED_STAGED = {
    (384, 4.0, ()): dict(load_encoder=4.681, encode=4.379, load=5.862,
                         denoise=5.032, decode=5.342),
    (768, 4.0, ()): dict(load_encoder=4.681, encode=4.379, load=5.862,
                         denoise=5.288, decode=5.603),
    (1536, 4.0, ()): dict(load_encoder=4.681, encode=4.379, load=5.862,
                          denoise=6.108, decode=5.672),
    (512, 4.0, (256, 256)): dict(load_encoder=7.902, encode=6.839,
                                 load=5.866, init=4.887, denoise=5.652,
                                 decode=6.126),
    (512, 4.0, (512, 512)): dict(load_encoder=7.902, encode=7.817,
                                 load=5.878, init=5.363, denoise=7.072,
                                 decode=7.639),
}


@pytest.mark.parametrize("cell", sorted(MEASURED_STAGED), ids=str)
def test_the_staged_phases_cover_the_measured_runs(q_view, cell):
    """Every phase of a staged plan is ROOM or more over its reading. The
    load terms alone leave less, and STAGE_ROOM is that room."""
    size, cfg, refs = cell
    p = _q16(q_view, size, cfg, refs, overcommit=True)
    assert p.staged_load
    for phase, read in MEASURED_STAGED[cell].items():
        assert p.phases[phase] >= read * GiB + ROOM, (phase, p.phases[phase]
                                                      / GiB)
    assert memfit.STAGE_ROOM == ROOM
    for phase in ("load_encoder", "load"):
        bare = p.phases[phase] - memfit.STAGE_ROOM
        assert MEASURED_STAGED[cell][phase] * GiB < bare < (
            MEASURED_STAGED[cell][phase] * GiB + ROOM)


def test_z_image_stages_its_load_only_when_a_budget_needs_it(z_view):
    """Z-Image at Q4_K_M fits a 16 GiB machine with every part loaded
    together, as without the lever. With a larger encoder quant the load
    of all parts is over that budget, and the request runs staged."""
    parts = (int(0.02 * GiB), int(0.34 * GiB))
    kw = dict(load_peak=max(parts), load_parts=parts, budget=BUDGET_16)
    p = _z(z_view, height=1024, width=1024, **kw)
    assert p.fits and not p.staged_load
    assert p == _z(z_view, height=1024, width=1024, budget=BUDGET_16,
                   load_peak=max(parts))
    assert p.levers == ("release the encoder", "tiled decode")
    q8 = int(4.3 * GiB)             # Qwen3-4B at Q8_0

    def big(size, **more):
        return plan_memory(z_view, q8, VAE, family="z_image", height=size,
                           width=size, overcommit=False, **more)

    for size, levers in ((512, ("release the encoder", "staged load")),
                         (1024, ("release the encoder", "tiled decode",
                                 "staged load"))):
        with pytest.raises(ImageMemoryError) as e:
            big(size, load_peak=max(parts), budget=BUDGET_16)
        # The misfit is every part resident together, at the load and
        # at the encode.
        whole = e.value.plan.phases
        assert e.value.plan.peak == max(whole["load"], whole["encode"])
        staged = big(size, **kw)
        assert staged.fits and staged.staged_load and staged.levers == levers
        assert staged.phases["load_encoder"] == (q8 + parts[0]
                                                 + memfit.STAGE_ROOM)
        assert staged.peak == staged.phases["decode"]


def test_the_vae_rows_cover_the_measured_footprints():
    for (conv, op, item), sizes in MEASURED.items():
        for size, gib in sizes.items():
            assert memfit.vae_peak(op, size, size, family="z_image",
                                   itemsize=item,
                                   conv=conv) >= gib * GiB, (conv, op, size)
    for (op, item), sizes in MEASURED_TILED.items():
        for size, gib in sizes.items():
            assert memfit.vae_peak(op, size, size, family="z_image",
                                   itemsize=item, tiled=True,
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


def _z_row(monkeypatch, **fits) -> None:
    """The Z-Image VAE row with ``fits`` in place of its own."""
    monkeypatch.setitem(memfit.VAE_ROWS, "z_image", dataclasses.replace(
        memfit.VAE_ROWS["z_image"], **fits))


def _unit_fits(monkeypatch):
    unit = {2: memfit.VaeFit(0.0, 1.0), 4: memfit.VaeFit(0.0, 1.0)}
    rows = {"exact": unit, "winograd": unit}
    tiled = {2: memfit.TiledFit(0.0, 0.125, 0.0),
             4: memfit.TiledFit(0.0, 0.125, 0.0)}
    _z_row(monkeypatch, whole={"decode": rows, "encode": rows},
           tiled={"decode": tiled, "encode": tiled})


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
    kw = dict(family="z_image", itemsize=2, channels=128, overcommit=False)
    whole = 2048 * 2048 * 128 * 2
    # The 5 x 5 tiles of 464 pixels at k_tiles = 0.125.
    assert memfit.tile_layout(2048) == (5, 464)
    tiles = int(0.125 * 25 * 464 * 464 * 128 * 2)
    assert memfit.vae_peak("decode", 2048, 2048, family="z_image",
                           itemsize=2) == whole
    assert memfit.vae_peak("decode", 2048, 2048, family="z_image",
                           itemsize=2, tiled=True) == tiles
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
                             family="z_image", itemsize=2,
                             overcommit=True) is False


def test_vae_tiling_runs_a_small_image_whole(monkeypatch):
    _unit_fits(monkeypatch)
    one = 512 * 512 * 128 * 4
    kw = dict(family="z_image", itemsize=4)
    assert memfit.vae_tiling("encode", 512, 512, tiled=True, free=one,
                             overcommit=False, **kw) is False
    with pytest.raises(ImageMemoryError, match="whole needs"):
        memfit.vae_tiling("encode", 512, 512, free=one - 1, overcommit=False,
                          **kw)
    assert memfit.vae_peak("encode", 512, 512, batch=3, **kw) == 3 * one


def test_vae_tiling_without_a_free_measure_never_refuses(monkeypatch):
    _unit_fits(monkeypatch)
    monkeypatch.setattr(memfit, "free_bytes", lambda: None)
    kw = dict(family="z_image", itemsize=2)
    assert memfit.vae_tiling("decode", 4096, 4096, **kw) is True
    assert memfit.vae_tiling("decode", 4096, 4096, tiled=False,
                             **kw) is False
    assert memfit.vae_tiling("decode", 4096, 4096, tiled=True, **kw) is True


@pytest.mark.parametrize("family", ["z_image", "qwen_image21"])
def test_without_a_free_measure_the_automatic_choice_is_tiles(family,
                                                              monkeypatch):
    """Nothing is refused, and no VAE runs an image larger than one tile
    whole unless the caller forces it."""
    monkeypatch.setattr(memfit, "free_bytes", lambda: None)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    for op in ("decode", "encode"):
        for item in (2, 4):
            kw = dict(family=family, itemsize=item)
            assert memfit.vae_tiling(op, 4096, 4096, **kw) is True
            assert memfit.vae_tiling(op, 528, 512, **kw) is True
            assert memfit.vae_tiling(op, 512, 512, **kw) is False
            assert memfit.vae_tiling(op, 4096, 4096, tiled=True, **kw) is True
            assert memfit.vae_tiling(op, 4096, 4096, tiled=False,
                                     **kw) is False
            assert memfit.vae_tiling(op, 512, 512, tiled=True, **kw) is False
