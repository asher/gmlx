"""The rope scaling resolver against a float32 transcription of llama.cpp.

The reference below transcribes ggml's rope_yarn, rope_yarn_ramp,
ggml_rope_yarn_corr_dims and ggml_rope_cache_init (the angle is built by
repeated multiplication, as ggml does), the hparams reads of
llama_model::load_hparams and the cparams block of llama_context's
constructor, all in float32.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gmlx.load import rope_scaling as rs
from gmlx.load.rope_scaling import RopeOverride

F = np.float32


# llama.cpp reference


def _ref_hparams(meta: dict, arch: str) -> dict:
    k = f"{arch}.rope."
    h = {
        "n_ctx_train": meta.get(f"{arch}.context_length", 0),
        "n_ctx_orig_yarn": meta.get(k + "scaling.original_context_length",
                                    meta.get(f"{arch}.context_length", 0)),
        "freq_base": F(meta.get(k + "freq_base", 10000.0)),
        "type": meta.get(k + "scaling.type", "linear"),
        "rope_attn_factor": F(meta.get(k + "scaling.attn_factor", 1.0)),
        "yarn_ext_factor": F(-1.0),
        "yarn_attn_factor": F(1.0),
        "yarn_beta_fast": F(32.0),
        "yarn_beta_slow": F(1.0),
        "log_mul": F(0.0),
    }
    scale = meta.get(k + "scaling.factor", 0.0)
    h["freq_scale_train"] = F(1.0) if scale == 0.0 else F(1.0) / F(scale)
    if arch in ("mistral3",):
        h["yarn_beta_fast"] = F(meta.get(k + "scaling.yarn_beta_fast", 32.0))
        h["yarn_beta_slow"] = F(meta.get(k + "scaling.yarn_beta_slow", 1.0))
        h["log_mul"] = F(meta.get(k + "scaling.yarn_log_multiplier", 0.0))
    if arch in ("deepseek2",):
        lm = meta.get(k + "scaling.yarn_log_multiplier")
        if lm is not None:
            h["log_mul"] = F(lm) / F(0.1)
    return h


def _ref_cparams(h: dict, arch: str, p: dict) -> dict:
    """p holds the llama.cpp params: unset is -1 for the yarn floats, 0 for
    freq base, freq scale and the original context, None for the type."""
    c = {}
    c["ext_factor"] = p["ext"] if p["ext"] >= 0 else h["yarn_ext_factor"]
    c["attn_factor"] = p["attn"] if p["attn"] >= 0 else h["yarn_attn_factor"]
    c["beta_fast"] = p["bf"] if p["bf"] >= 0 else h["yarn_beta_fast"]
    c["beta_slow"] = p["bs"] if p["bs"] >= 0 else h["yarn_beta_slow"]
    c["freq_base"] = h["freq_base"] if p["base"] == 0 else F(p["base"])
    c["freq_scale"] = h["freq_scale_train"] if p["fs"] == 0 else F(p["fs"])
    c["n_ctx_orig"] = (p["orig"] if p["orig"] != 0 else
                       h["n_ctx_orig_yarn"] if h["n_ctx_orig_yarn"] != 0 else
                       h["n_ctx_train"])
    c["type"] = p["type"] if p["type"] is not None else h["type"]
    if c["type"] == "none":
        c["freq_scale"] = F(1.0)
    if c["ext_factor"] < 0:
        c["ext_factor"] = F(1.0) if c["type"] == "yarn" else F(0.0)

    def get_mscale(scale, mscale):
        return F(1.0) if scale <= 1.0 else F(0.1) * mscale * F(math.log(scale)) + F(1.0)

    if c["ext_factor"] != 0:
        factor = F(1.0) / c["freq_scale"]
        if h["log_mul"] != 0:
            mscale = F(1.0)
            mscale_all = h["log_mul"]
            if arch == "deepseek2" and mscale_all != 1.0:
                mscale = mscale_all
            c["attn_factor"] = get_mscale(factor, mscale) / get_mscale(factor, mscale_all)
        else:
            c["attn_factor"] = get_mscale(factor, F(1.0))
        c["attn_factor"] = c["attn_factor"] * (F(1.0) / (F(1.0) + F(0.1) * F(math.log(factor))))
    c["attn_factor"] = F(c["attn_factor"] * h["rope_attn_factor"])
    return c


def _ref_corr_dims(n_dims, n_ctx_orig, base, beta_fast, beta_slow):
    def corr_dim(n_rot):
        return F(n_dims) * F(math.log(F(n_ctx_orig) / (F(n_rot) * F(2) * F(math.pi)))) / (
            F(2) * F(math.log(F(base))))
    start = F(math.floor(corr_dim(beta_fast)))
    end = F(math.ceil(corr_dim(beta_slow)))
    return max(F(0), start), min(F(n_dims - 1), end)


def _ref_rope_yarn(theta_extrap, freq_scale, corr, i0, ext_factor, mscale):
    theta_interp = F(freq_scale * theta_extrap)
    theta = theta_interp
    if ext_factor != 0.0:
        y = F((i0 // 2 - corr[0]) / max(F(0.001), F(corr[1] - corr[0])))
        ramp = F(1) - min(F(1), max(F(0), y))
        ramp_mix = F(ramp * ext_factor)
        theta = F(theta_interp * (F(1) - ramp_mix) + theta_extrap * ramp_mix)
        mscale = F(mscale * (F(1.0) + F(0.1) * F(math.log(F(1.0) / freq_scale))))
    return theta, F(mscale)


def _ref_angles(pos, n_dims, c):
    """ggml_rope_cache_init: per-pair angle and magnitude at ``pos``."""
    theta_scale = F(F(c["freq_base"]) ** F(-2.0 / n_dims))
    corr = _ref_corr_dims(n_dims, c["n_ctx_orig"], c["freq_base"],
                          c["beta_fast"], c["beta_slow"])
    theta = F(pos)
    out, mag = [], None
    for i0 in range(0, n_dims, 2):
        t, mag = _ref_rope_yarn(theta, c["freq_scale"], corr, i0,
                                c["ext_factor"], c["attn_factor"])
        out.append(t)
        theta = F(theta * theta_scale)
    return np.array(out, dtype=np.float32), mag


def _params(**kw):
    p = {"ext": -1.0, "attn": -1.0, "bf": -1.0, "bs": -1.0, "base": 0.0,
         "fs": 0.0, "orig": 0, "type": None}
    p.update(kw)
    return p


def _override_from(p: dict) -> RopeOverride:
    return RopeOverride(
        rope_scaling=p["type"],
        rope_freq_scale=p["fs"] or None,
        rope_freq_base=p["base"] or None,
        yarn_orig_ctx=p["orig"] or None,
        yarn_ext_factor=None if p["ext"] < 0 else p["ext"],
        yarn_attn_factor=None if p["attn"] < 0 else p["attn"],
        yarn_beta_fast=None if p["bf"] < 0 else p["bf"],
        yarn_beta_slow=None if p["bs"] < 0 else p["bs"],
    )


# Header and option sets. Head widths and bases follow the real models.
CASES = {
    # Qwen3.8-Flash-Next model card: no header scaling, factor 4 from the
    # options, original context 262144.
    "model-card": ("qwen4exp", {
        "qwen4exp.context_length": 262144,
        "qwen4exp.rope.freq_base": 10000000.0,
    }, _params(type="yarn", fs=0.25, orig=262144), 64),
    "model-card-factor-2": ("qwen4exp", {
        "qwen4exp.context_length": 262144,
        "qwen4exp.rope.freq_base": 10000000.0,
    }, _params(type="yarn", fs=0.5), 64),
    "gpt-oss-header": ("gpt-oss", {
        "gpt-oss.context_length": 131072,
        "gpt-oss.rope.freq_base": 150000.0,
        "gpt-oss.rope.scaling.type": "yarn",
        "gpt-oss.rope.scaling.factor": 32.0,
        "gpt-oss.rope.scaling.original_context_length": 4096,
    }, _params(), 64),
    "deepseek4-header": ("deepseek4", {
        "deepseek4.context_length": 1048576,
        "deepseek4.rope.freq_base": 10000.0,
        "deepseek4.rope.scaling.type": "yarn",
        "deepseek4.rope.scaling.factor": 16.0,
        "deepseek4.rope.scaling.original_context_length": 65536,
    }, _params(), 64),
    "deepseek2-log-mul": ("deepseek2", {
        "deepseek2.context_length": 163840,
        "deepseek2.rope.freq_base": 10000.0,
        "deepseek2.rope.scaling.type": "yarn",
        "deepseek2.rope.scaling.factor": 40.0,
        "deepseek2.rope.scaling.original_context_length": 4096,
        "deepseek2.rope.scaling.yarn_log_multiplier": 0.0707,
    }, _params(), 64),
    "mistral3-log-mul-beta": ("mistral3", {
        "mistral3.context_length": 262144,
        "mistral3.rope.freq_base": 1000000.0,
        "mistral3.rope.scaling.type": "yarn",
        "mistral3.rope.scaling.factor": 16.0,
        "mistral3.rope.scaling.original_context_length": 16384,
        "mistral3.rope.scaling.yarn_beta_fast": 4.0,
        "mistral3.rope.scaling.yarn_beta_slow": 1.0,
        "mistral3.rope.scaling.yarn_log_multiplier": 1.0,
    }, _params(), 128),
    "ext-factor-0": ("qwen4exp", {
        "qwen4exp.context_length": 262144,
        "qwen4exp.rope.freq_base": 10000000.0,
    }, _params(type="yarn", fs=0.25, ext=0.0, attn=1.5), 64),
    "linear-header": ("llama", {
        "llama.context_length": 32768,
        "llama.rope.freq_base": 500000.0,
        "llama.rope.scaling.type": "linear",
        "llama.rope.scaling.factor": 8.0,
    }, _params(), 128),
    "linear-option": ("qwen3", {
        "qwen3.context_length": 40960,
        "qwen3.rope.freq_base": 1000000.0,
    }, _params(fs=0.5), 128),
    "none-over-header": ("gpt-oss", {
        "gpt-oss.context_length": 131072,
        "gpt-oss.rope.freq_base": 150000.0,
        "gpt-oss.rope.scaling.type": "yarn",
        "gpt-oss.rope.scaling.factor": 32.0,
        "gpt-oss.rope.scaling.original_context_length": 4096,
    }, _params(type="none"), 64),
    "none": ("qwen3", {
        "qwen3.context_length": 40960,
        "qwen3.rope.freq_base": 1000000.0,
    }, _params(), 128),
    "attn-factor-header": ("qwen3", {
        "qwen3.context_length": 32768,
        "qwen3.rope.freq_base": 1000000.0,
        "qwen3.rope.scaling.type": "yarn",
        "qwen3.rope.scaling.factor": 4.0,
        "qwen3.rope.scaling.original_context_length": 32768,
        "qwen3.rope.scaling.attn_factor": 0.8,
        # llama.cpp reads no beta key for qwen3; neither may we.
        "qwen3.rope.scaling.yarn_beta_fast": 8.0,
    }, _params(bf=16.0, bs=2.0), 128),
    "beta-key-not-read-for-qwen3": ("qwen3", {
        "qwen3.context_length": 32768,
        "qwen3.rope.freq_base": 1000000.0,
        "qwen3.rope.scaling.type": "yarn",
        "qwen3.rope.scaling.factor": 4.0,
        "qwen3.rope.scaling.original_context_length": 32768,
        "qwen3.rope.scaling.yarn_beta_fast": 8.0,
        "qwen3.rope.scaling.yarn_log_multiplier": 0.1,
    }, _params(), 128),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_resolve_matches_llama_cpp(name):
    arch, meta, p, dims = CASES[name]
    c = _ref_cparams(_ref_hparams(meta, arch), arch, p)
    got = rs.resolve(meta, arch, _override_from(p))
    assert got.type == c["type"]
    assert F(got.freq_scale) == c["freq_scale"]
    assert got.original_context == c["n_ctx_orig"]
    assert F(got.ext_factor) == F(c["ext_factor"])
    assert F(got.beta_fast) == F(c["beta_fast"])
    assert F(got.beta_slow) == F(c["beta_slow"])
    assert F(got.attn_factor) == pytest.approx(float(c["attn_factor"]), rel=1e-6)


@pytest.mark.parametrize("name", sorted(CASES))
def test_table_and_scale_match_llama_cpp(name):
    arch, meta, p, dims = CASES[name]
    c = _ref_cparams(_ref_hparams(meta, arch), arch, p)
    ref, ref_mag = _ref_angles(1, dims, c)
    got = rs.resolve(meta, arch, _override_from(p))
    table = rs.inv_freq(dims, float(c["freq_base"]), got)
    np.testing.assert_allclose(table, ref, rtol=4e-6, atol=0)
    assert got.mscale == pytest.approx(float(ref_mag), rel=1e-6)
    # The mx.fast.rope convention is the reciprocal.
    fr = np.asarray(rs.freqs(dims, float(c["freq_base"]), got))
    np.testing.assert_allclose(1.0 / fr, ref, rtol=4e-6)


# llama.cpp loaders that set rope_freq_scale_train_swa to the trained scale.
_SWA_TRAIN = ("gpt-oss", "gemma2")


@pytest.mark.parametrize("arch", ["qwen3", "gpt-oss"])
@pytest.mark.parametrize("p", [_params(type="yarn", fs=0.25, base=500000.0),
                               _params(), _params(type="none")],
                         ids=["options", "header", "none"])
def test_sliding_layers_match_llama_cpp(arch, p):
    """A sliding layer rotates with llama_model::get_rope_freq_scale and
    get_rope_freq_base (llama-model.cpp): the trained sliding scale (1, or
    the trained scale for the archs above) and the trained base. The
    extrapolation mix, the attention factor, the betas and the original
    context stay the run's."""
    meta = {
        f"{arch}.context_length": 131072,
        f"{arch}.rope.freq_base": 150000.0,
        f"{arch}.rope.scaling.type": "yarn",
        f"{arch}.rope.scaling.factor": 32.0,
        f"{arch}.rope.scaling.original_context_length": 4096,
    }
    dims = 64
    h = _ref_hparams(meta, arch)
    c = dict(_ref_cparams(h, arch, p))
    c["freq_scale"] = h["freq_scale_train"] if arch in _SWA_TRAIN else F(1.0)
    c["freq_base"] = h["freq_base"]
    ref, ref_mag = _ref_angles(1, dims, c)
    got = rs.resolve(meta, arch, _override_from(p)).for_sliding()
    assert got.freq_base == float(h["freq_base"])
    np.testing.assert_allclose(rs.inv_freq(dims, got.freq_base, got), ref,
                               rtol=4e-6, atol=0)
    assert got.mscale == pytest.approx(float(ref_mag), rel=1e-6)


def test_model_card_numbers():
    meta = CASES["model-card"][1]
    s = rs.resolve(meta, "qwen4exp", RopeOverride(
        rope_scaling="yarn", rope_scale=4.0, yarn_orig_ctx=262144))
    assert s.factor == 4.0
    # 0.1 ln 4 + 1, as cos and sin are multiplied by it.
    assert s.mscale == pytest.approx(0.1 * math.log(4.0) + 1.0, rel=1e-6)
    assert s.active and s.user_set
    assert rs.effective_context(262144, s) == 1048576
    s2 = rs.resolve(meta, "qwen4exp", RopeOverride(rope_scaling="yarn",
                                                   rope_scale=2.0))
    assert rs.effective_context(262144, s2) == 524288


def test_none_is_inactive_and_keeps_context():
    s = rs.resolve(CASES["none"][1], "qwen3")
    assert not s.active
    assert s.identity() is None
    assert rs.effective_context(40960, s) == 40960
    np.testing.assert_array_equal(
        rs.inv_freq(128, 1e6, s),
        np.power(1e6, -np.arange(64) * 2 / 128).astype(np.float32))


def test_header_yarn_keeps_trained_context():
    # A header factor already describes the trained context.
    s = rs.resolve(CASES["gpt-oss-header"][1], "gpt-oss")
    assert s.active and not s.user_set
    assert rs.effective_context(131072, s) == 131072


def test_linear_option_does_not_extend_context():
    s = rs.resolve(CASES["linear-option"][1], "qwen3",
                   RopeOverride(rope_scale=2.0))
    assert s.type == "linear" and s.factor == 2.0
    assert rs.effective_context(40960, s) == 40960


def test_identity_tracks_every_number():
    meta = CASES["model-card"][1]
    a = rs.resolve(meta, "qwen4exp", RopeOverride(rope_scaling="yarn", rope_scale=4))
    b = rs.resolve(meta, "qwen4exp", RopeOverride(rope_scaling="yarn", rope_scale=2))
    c = rs.resolve(meta, "qwen4exp", RopeOverride(rope_scaling="yarn", rope_scale=4,
                                                  yarn_beta_fast=16))
    assert len({a.identity(), b.identity(), c.identity()}) == 3
    assert a.identity() == rs.resolve(meta, "qwen4exp", RopeOverride(
        rope_scaling="yarn", rope_freq_scale=0.25)).identity()


def test_override_validation():
    with pytest.raises(ValueError, match="not one of"):
        RopeOverride(rope_scaling="dynamic")
    with pytest.raises(ValueError, match="positive"):
        RopeOverride(rope_scale=0.0)
    with pytest.raises(ValueError, match="disagree"):
        RopeOverride(rope_scale=4.0, rope_freq_scale=0.5)
    RopeOverride(rope_scale=4.0, rope_freq_scale=0.25)
    with pytest.raises(ValueError, match="positive integer"):
        RopeOverride(yarn_orig_ctx=0)


def test_from_config_keys():
    o = RopeOverride.from_config({"type": "yarn", "factor": 4,
                                  "original_context": 262144})
    assert o == RopeOverride(rope_scaling="yarn", rope_scale=4.0,
                             yarn_orig_ctx=262144)
    with pytest.raises(ValueError, match="unknown key"):
        RopeOverride.from_config({"rope_type": "yarn"})
    assert RopeOverride.from_config(None).is_empty()


def test_unknown_header_type_raises():
    with pytest.raises(ValueError, match="unknown rope scaling type"):
        rs.resolve({"qwen3.rope.scaling.type": "dynamic"}, "qwen3")


def test_float32_phase_near_one_million():
    """At a position near 1,000,000 the float32 angle, as the rope kernels
    and ggml form it, stays within a bounded phase error of the float64
    angle, and mx.fast.rope with the table rotates within that bound."""
    mx = pytest.importorskip("mlx.core")
    meta = CASES["model-card"][1]
    s = rs.resolve(meta, "qwen4exp", RopeOverride(rope_scaling="yarn",
                                                  rope_scale=4.0))
    dims, base, pos = 64, 1e7, 999_983
    table = rs.inv_freq(dims, base, s)
    # Float64 reference angle from a float64 table.
    j = np.arange(dims // 2, dtype=np.float64)
    ex = base ** (-2 * j / dims)
    lo, hi = rs.corr_dims(dims, base, s)
    ramp = 1 - np.clip((j - lo) / max(0.001, hi - lo), 0, 1)
    inv64 = 0.25 * ex * (1 - ramp) + ex * ramp
    ang64 = pos * inv64
    ang32 = (np.float32(pos) * table).astype(np.float64)
    err = np.abs(np.angle(np.exp(1j * (ang32 - ang64))))
    # float32 spacing at 1e6 is 0.0625; the table adds pos * 1 ulp.
    bound = 0.0625 / 2 + pos * np.abs(table.astype(np.float64) - inv64).max() + 1e-6
    assert err.max() <= bound
    assert bound < 0.1
    # llama.cpp's angles at the same position stay inside the same bound.
    c = _ref_cparams(_ref_hparams(meta, "qwen4exp"), "qwen4exp",
                     _params(type="yarn", fs=0.25))
    ref, _ = _ref_angles(pos, dims, c)
    ref_err = np.abs(np.angle(np.exp(1j * (ref.astype(np.float64) - ang64))))
    assert ref_err.max() < 2 * bound
    # mx.fast.rope with the table against the float64 rotation.
    x = mx.ones((1, 1, 1, dims), dtype=mx.float32)
    y = mx.fast.rope(x, dims, traditional=False, base=None, scale=1.0,
                     offset=pos, freqs=rs.freqs(dims, base, s))
    y = np.asarray(y).reshape(dims)
    h = dims // 2
    want = np.concatenate([np.cos(ang64) - np.sin(ang64),
                           np.sin(ang64) + np.cos(ang64)])
    assert np.abs(y - want).max() <= 2 * bound + 1e-5
    assert np.abs(y[:h] - want[:h]).max() < 0.1
