"""Check 5 of the image acceptance gate for Qwen-Image 2.1: the VAE.

The tiny tests run on the CPU against the two self-test fixtures, whose
arrays are committed beside their manifests. The oracle is diffusers'
AutoencoderKLQwenImage21 at the tiny config with seeded weights, and the
tests write the same weights as a safetensors file in the official layout,
whose sha256 the manifest holds, and pack it into a GGUF. From either file
the fp32 decode and the fp32 encoder mean must stay within the manifest's
max abs, and the bf16 decode above its PSNR. The same holds for the file
without its time_conv tensors. Nine wrong models, each one departure from
the reference, must fail the fp32 gate. The second fixture holds ten more
tiny configs with other widths, levels and time flags. A config with the
official levels and time flags is read from a file, and the others are
built from the config, since a file cannot give them.

The fp32 gate of the tiny models is 5e-5 max abs: about 7 times the
largest reading of a right model over these configs, 6.9e-6, where the
smallest reading of a wrong model is 0.33.

The official tests need the VAE file. gmlx decodes the oracle's latents
and encodes its image with the official VAE, read from a GGUF packed from
the oracle's safetensors when one is found, else from that safetensors
file, and holds the manifest's tolerances. The oracle runs every fp32
pass again in float64, and each fp32 gate follows from its own fp32
against float64 gap on that tensor.

The fp32 decode is held to the oracle's fp32 decode at the manifest's max
abs, twice that gap: 1.2e-4 at 512 pixels and 1.3e-4 at 1024, where gmlx
on an M5 Max with TF32 off reads 3.0e-5 and 4.9e-5. The latents of the
image with alpha decode to a real alpha channel, half of it neither
opaque nor clear, and that decode is held the same way.

The fp32 encoder mean is held to the oracle's float64 mean by relative
L2, at twice the oracle's own fp32 against float64 relative L2, on the
GPU and on the MLX CPU stream. Max abs is the wrong measure for this
tensor. Its largest value is 36 to 66 times its rms, so the largest error
of a right implementation moves with the one value it falls on: against
the manifest's max abs gate the GPU reads 0.39 to 0.85 of it and the CPU
stream 0.46 to 0.94, and against twice the oracle's own max abs gap the
GPU reads 1.05 at 512 pixels with alpha. By relative L2 the two read 0.46
to 0.64 of the gate on every image, and the GPU with TF32 on reads 60 to
85 times it. One value of a mean that is off by the old max abs gate
still fails the new one. The max abs is printed beside its share of the
manifest's gate.

The fp32 items are gated with MLX_ENABLE_TF32=0 in a child process, as in
the Z-Image check. Only the default conv path gates: the other path is
read and printed, and in bf16 its decode at 1024 pixels reads 48.06 dB
against a gate of 49.21. The official tests are skipped unless
GMLX_TEST_IMAGE_DIR holds the VAE and GMLX_IMAGE_FIXTURES the fixture of
the committed manifest, which the oracle wrote at 512 and 1024 pixels.

Every pass of an official test goes through ``tiling.decode`` or
``tiling.encode``, which price it by the VAE row of the memory plan and
refuse one that does not fit. Each test forces its passes whole or in
tiles and states the price of its largest pass over the weights, and
none runs a whole pass above 1024 pixels, where a bf16 decode is priced
at 6.80 GiB against 23.03 GiB at 2048. A tiled decode is the whole decode
and a tiled encode the whole encode, and the last two tests hold that on
the official file.
scripts/image_oracle/check5_qwen_image21_vae.py writes the fixtures."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from image_dir import sha256, vae_files
from oracle_fixture import ORACLE, committed_fixture, fixture, fx
from qwen_vae_fixtures import (FAMILY, WRONG, build, module_config, pruned,
                               tiny, tiny_file, weights)

import mlx.core as mx

NAME = "check5_qwen_image21_vae"
TINY = "selftest_qwen_image21_vae_tiny"
CASES = "selftest_qwen_image21_vae_cases"
TINY_FP32_MAX_ABS = 5e-5
OFFICIAL_FLAGS = [False, True, True, True]
TILED_ENCODE_REL_L2 = 0.05
TILED_ROUND_TRIP_DB = 50.0
# The largest side that an official test runs whole.
WHOLE_MAX = 1024


def to_unit(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float64) / 2 + 0.5, 0.0, 1.0)


def _tags(m) -> list:
    return [f"{h}x{w}" for h, w in m["inputs"]["sizes"]]


def _decode(vae, latents, **kw) -> np.ndarray:
    """The RGBA decode of model-space latents, NCHW."""
    from gmlx.image.vae import tiling

    out = tiling.decode(vae, latents, alpha=True, **kw)
    return np.array(out).transpose(0, 3, 1, 2)


def _encode_mean(vae, image) -> np.ndarray:
    from gmlx.image import memfit

    x = mx.array(image.transpose(0, 2, 3, 1))
    with memfit.bounded_cache():
        mean, _ = vae.encode(x)
        return np.array(mean.astype(mx.float32)).transpose(0, 3, 1, 2)


def _encode_latents(vae, image, **kw) -> np.ndarray:
    from gmlx.image.vae import tiling

    return np.array(tiling.encode(vae, image.transpose(0, 2, 3, 1), **kw))


def _guarded_mean(vae, image, m, **kw) -> np.ndarray:
    """The encoder mean through the guarded encode. That gives model-space
    latents, and the oracle's statistics put them back on the VAE's
    scale."""
    shape = (1, -1, 1, 1)
    lat = _encode_latents(vae, image, **kw).astype(np.float64)
    return (lat * np.array(m["inputs"]["latents_std"]).reshape(shape)
            + np.array(m["inputs"]["latents_mean"]).reshape(shape))


def _whole_tags(m) -> list:
    """The manifest's sizes for a test that runs each whole."""
    big = [s for s in m["inputs"]["sizes"] if max(s) > WHOLE_MAX]
    assert not big, (f"no official test runs a whole pass above {WHOLE_MAX} "
                     f"pixels, and the oracle's sizes hold {big}")
    return _tags(m)


def readings(vae, arrays, s: str) -> dict:
    """Max abs of the decode, the two encoder means and the model-space
    latents against the oracle's fp32 arrays at size ``s``."""
    image = arrays[f"image_{s}"]
    rgba = np.concatenate([image, arrays[f"alpha_{s}"]], axis=1)
    return {
        "decode": fx.max_abs(_decode(vae, arrays[f"latents_{s}"],
                                     tiled=False, free=1e12),
                             arrays[f"decoded_{s}"]),
        "encode": fx.max_abs(_encode_mean(vae, image),
                             arrays[f"encode_mean_{s}"]),
        "encode_rgba": fx.max_abs(_encode_mean(vae, rgba),
                                  arrays[f"encode_mean_rgba_{s}"]),
        "latents": fx.max_abs(_encode_latents(vae, image, tiled=False,
                                              free=1e12),
                              arrays[f"latents_{s}"]),
    }


def tolerances(m, s: str) -> dict:
    t = m["tolerances"]["sizes"][s]
    return {"decode": t["decode_fp32_max_abs"],
            "encode": t["encode_fp32_max_abs"],
            "encode_rgba": t["encode_fp32_max_abs"],
            "latents": t["encode_fp32_max_abs"]}


def failed(got: dict, tol: dict) -> list:
    return sorted(k for k in got if not got[k] <= tol[k])


# The tiny model: CPU, committed arrays.


@pytest.fixture(scope="module")
def selftest():
    return committed_fixture(TINY)


@pytest.fixture(scope="module")
def files(selftest, tmp_path_factory):
    from gmlx.image.vae import pack_vae

    d = tmp_path_factory.mktemp("qwen_vae_tiny")
    st = tiny_file(d)
    gg = pack_vae(st, d / "vae.gguf", source_repo="selftest",
                  source_revision="selftest",
                  config_json=json.dumps(tiny.TINY)).path
    return {"safetensors": st, "gguf": gg}


def _load(path, dtype):
    from gmlx.image.vae import load_vae

    return load_vae(path, family=FAMILY, dtype=dtype)


def test_the_tiny_file_is_the_one_the_oracle_ran(selftest, files):
    _, m = selftest
    want = m["files"]["vae"]
    assert os.path.getsize(files["safetensors"]) == want["size"]
    assert sha256(files["safetensors"]) == want["sha256"]
    assert m["inputs"]["config"] == tiny.TINY
    assert m["inputs"]["weights_seed"] == tiny.SEED
    assert m["oracle"]["packages"]["diffusers"]["source"].endswith(
        "e0abab83b5df05de9e7abd788643c1a7c1e42e28.tar.gz")


def test_the_tiny_fp32_gate_is_5e_5(selftest, cases):
    for _, m in (selftest, cases):
        assert m["tolerances"]["fp32_max_abs"] == TINY_FP32_MAX_ABS
        assert "5e-5" in m["tolerances"]["rule"]
    for s in _tags(selftest[1]):
        t = selftest[1]["tolerances"]["sizes"][s]
        assert t["decode_fp32_max_abs"] == TINY_FP32_MAX_ABS
        assert t["encode_fp32_max_abs"] == TINY_FP32_MAX_ABS


def test_the_tiny_fixture_exercises_the_clamp_and_the_alpha(selftest):
    arrays, m = selftest
    for s in _tags(m):
        r = m["results"]["sizes"][s]
        dec = arrays[f"decoded_{s}"]
        assert dec.shape[1] == 4 and np.abs(dec).max() == 1.0
        assert 0.01 < r["decode_clamped_share"] < 0.5
        assert np.ptp(dec[:, 3]) > 0.5
        assert r["encode_rgba_vs_opaque"]["max_abs"] > 0.05
        assert arrays[f"alpha_{s}"].min() == -1.0


def test_the_latent_statistics_are_the_oracle_ones(selftest, files):
    _, m = selftest
    cfg = _load(files["safetensors"], "fp32").config
    assert list(cfg.latents_mean) == m["inputs"]["latents_mean"]
    assert list(cfg.latents_std) == m["inputs"]["latents_std"]


def test_the_guarded_encode_gives_the_encoder_mean(selftest, files):
    """The official tests read the encoder mean from the guarded encode."""
    arrays, m = selftest
    with mx.stream(mx.cpu):
        vae = _load(files["safetensors"], "fp32")
        for s in _tags(m):
            image = arrays[f"image_{s}"]
            got = _guarded_mean(vae, image, m, tiled=False, free=1e12)
            tol = m["tolerances"]["sizes"][s]["encode_fp32_max_abs"]
            assert fx.max_abs(got, _encode_mean(vae, image)) < 2e-6
            assert fx.max_abs(got, arrays[f"encode_mean_{s}"]) <= tol
        assert _whole_tags(m) == _tags(m)
        big = {"inputs": {"sizes": [[512, 512], [1024, 1040]]}}
        with pytest.raises(AssertionError, match="above 1024 pixels"):
            _whole_tags(big)


@pytest.mark.parametrize("fmt", ["safetensors", "gguf"])
def test_tiny_fp32_decode_and_encode_are_within_max_abs(selftest, files,
                                                        fmt):
    arrays, m = selftest
    with mx.stream(mx.cpu):
        vae = _load(files[fmt], "fp32")
        assert vae.source_format == fmt
        for s in _tags(m):
            got, tol = readings(vae, arrays, s), tolerances(m, s)
            for k in got:
                print(f"[check5] tiny {fmt} {s} fp32 {k} max abs "
                      f"{got[k]:.3e}, tolerance {tol[k]:.1e}, room "
                      f"{tol[k] / got[k]:.0f}x")
            assert failed(got, tol) == []


@pytest.mark.parametrize("form", ["all_12", "the_weights"])
def test_the_tiny_file_without_time_conv_is_within_max_abs(selftest,
                                                           tmp_path, form):
    from gmlx.image.vae import pack_vae

    arrays, m = selftest
    t = pruned()[form]
    assert len(t) == {"all_12": 226, "the_weights": 232}[form]
    st = tiny_file(tmp_path, t, "pruned.safetensors")
    gg = pack_vae(st, tmp_path / "pruned.gguf", source_repo="selftest",
                  source_revision="selftest",
                  config_json=json.dumps(tiny.TINY)).path
    with mx.stream(mx.cpu):
        for path in (st, gg):
            vae = _load(path, "fp32")
            assert list(vae.config.temporal) == OFFICIAL_FLAGS
            for s in _tags(m):
                got, tol = readings(vae, arrays, s), tolerances(m, s)
                print(f"[check5] tiny without {form} of time_conv, "
                      f"{vae.source_format} {s} fp32: " + ", ".join(
                          f"{k} {got[k]:.3e}" for k in got))
                assert failed(got, tol) == []


@pytest.mark.parametrize("fmt", ["safetensors", "gguf"])
def test_tiny_bf16_decode_psnr_and_encode(selftest, files, fmt):
    arrays, m = selftest
    with mx.stream(mx.cpu):
        vae = _load(files[fmt], "bf16")
        assert vae.dtype == mx.bfloat16
        for s in _tags(m):
            t = m["tolerances"]["sizes"][s]
            got = _decode(vae, arrays[f"latents_{s}"], tiled=False,
                          free=1e12)
            ref = arrays[f"decoded_{s}"]
            db = fx.psnr(to_unit(got), to_unit(ref), 1.0)
            own = m["results"]["sizes"][s]["decode_bf16_vs_fp32"]["psnr_db"]
            rel = fx.rel_l2(_encode_mean(vae, arrays[f"image_{s}"]),
                            arrays[f"encode_mean_{s}"])
            print(f"[check5] tiny {fmt} {s} bf16 decode PSNR {db:.2f} dB, "
                  f"gate {t['decode_bf16_psnr_db']:.2f} dB, the oracle's "
                  f"own {own:.2f} dB; bf16 encode rel L2 {rel:.3e}, "
                  f"tolerance {t['encode_bf16_rel_l2']:.3e}")
            assert got.dtype == np.float32 and np.abs(got).max() <= 1.0
            assert db >= t["decode_bf16_psnr_db"]
            assert rel <= t["encode_bf16_rel_l2"]


@pytest.mark.parametrize("wrong", sorted(WRONG))
def test_a_wrong_model_fails_the_fp32_gate(selftest, files, wrong,
                                           monkeypatch):
    arrays, m = selftest
    depart, must_fail = WRONG[wrong]
    with mx.stream(mx.cpu):
        vae = _load(files["safetensors"], "fp32")
        depart(monkeypatch, vae)
        for s in _tags(m):
            got, tol = readings(vae, arrays, s), tolerances(m, s)
            bad = failed(got, tol)
            print(f"[check5] tiny wrong model {wrong} {s}: " + ", ".join(
                f"{k} {got[k]:.3e}" for k in got) + f"; fails {bad}")
            assert set(must_fail) <= set(bad)


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_tiny_gguf_and_safetensors_weights_are_bit_identical(files, dtype):
    g, s = _load(files["gguf"], dtype), _load(files["safetensors"], dtype)
    a, b = weights(g), weights(s)
    assert a.keys() == b.keys() and len(a) == 226
    assert all(np.array_equal(a[k], b[k]) for k in a)
    if dtype == "fp32":
        src = tiny.weights()
        for k, v in a.items():
            want = src[k].transpose(0, 2, 3, 1) if src[k].ndim == 4 \
                else src[k]
            assert np.array_equal(v, want), k


# Ten more tiny configs: CPU, committed arrays.


@pytest.fixture(scope="module")
def cases():
    return committed_fixture(CASES)


def _from_a_file(config: dict) -> bool:
    return (len(config["dim_mult"]) == 5
            and config["temperal_downsample"] == OFFICIAL_FLAGS)


def case_readings(vae, arrays, name: str) -> dict:
    """Max abs of the decode and of the encoder mean and log variance
    against the oracle's fp32 arrays of a case."""
    lat, image = tiny.case_inputs(name)
    mean, logvar = vae.encode(mx.array(image.transpose(0, 2, 3, 1)))
    return {
        "decode": fx.max_abs(_decode(vae, lat, tiled=False, free=1e12),
                             arrays[f"decoded_{name}"]),
        "mean": fx.max_abs(np.array(mean).transpose(0, 3, 1, 2),
                           arrays[f"mean_{name}"]),
        "logvar": fx.max_abs(np.array(logvar).transpose(0, 3, 1, 2),
                             arrays[f"logvar_{name}"]),
    }


def test_the_cases_are_the_ones_the_oracle_ran(cases):
    arrays, m = cases
    assert m["inputs"]["cases"] == list(tiny.CASES) and len(tiny.CASES) == 10
    assert sum(_from_a_file(tiny.case_config(n)) for n in tiny.CASES) == 5
    for name in tiny.CASES:
        r = m["results"]["cases"][name]
        config = tiny.case_config(name)
        assert r["config"] == config
        assert r["size"] == list(tiny.CASES[name][1])
        src = tiny.weights(config)
        assert r["tensors"] == len(src)
        assert tiny.digest(src[k] for k in sorted(src)) == r["weights_sha256"]
        assert tiny.digest(tiny.case_inputs(name)) == r["inputs_sha256"]
        dec = arrays[f"decoded_{name}"]
        assert dec.shape == (1, 4, *tiny.CASES[name][1])
        # The clamp acts in every case.
        assert np.abs(dec).max() == 1.0
        assert 0.05 < r["decode_clamped_share"] < 0.5
        # The oracle's own fp32 rounding is far inside the gate.
        assert max(r["fp32_vs_fp64_max_abs"].values()) < TINY_FP32_MAX_ABS / 5


@pytest.mark.parametrize("name", list(tiny.CASES))
def test_a_tiny_config_is_within_max_abs(cases, tmp_path, name):
    arrays, m = cases
    config = tiny.case_config(name)
    tol = m["tolerances"]["fp32_max_abs"]
    with mx.stream(mx.cpu):
        if _from_a_file(config):
            vae = _load(tiny_file(tmp_path, tiny.weights(config)), "fp32")
        else:
            vae = build(config)
        assert vae.config == module_config(config)
        got = case_readings(vae, arrays, name)
    print(f"[check5] case {name} fp32: " + ", ".join(
        f"{k} max abs {v:.3e}" for k, v in got.items())
        + f", tolerance {tol:.1e}, room {tol / max(got.values()):.1f}x")
    assert max(got.values()) <= tol


def test_a_file_cannot_give_the_model_other_time_flags(cases, tmp_path):
    from gmlx.image.vae.reader import VAEError

    arrays, m = cases
    tol = m["tolerances"]["fp32_max_abs"]
    # time_conv at the first downsample, where the official config halves
    # no time, is refused.
    for name in ("other_time_flags", "all_time_flags"):
        path = tiny_file(tmp_path, tiny.weights(tiny.case_config(name)),
                         f"{name}.safetensors")
        with pytest.raises(VAEError, match="unexpected.*encoder.down_blocks"
                           r".0.downsampler.time_conv"):
            _load(path, "fp32")
    # A file with no time_conv is the official layout without them, so it
    # reads with the official flags and is another model than the config
    # with no time flags.
    name = "no_time_flags"
    path = tiny_file(tmp_path, tiny.weights(tiny.case_config(name)),
                     f"{name}.safetensors")
    with mx.stream(mx.cpu):
        vae = _load(path, "fp32")
        assert list(vae.config.temporal) == OFFICIAL_FLAGS
        got = case_readings(vae, arrays, name)
    print(f"[check5] the {name} file read with the official flags: "
          + ", ".join(f"{k} max abs {v:.3e}" for k, v in got.items()))
    assert min(got.values()) > 1000 * tol


# The official VAE: needs the file and the fixture of the committed
# manifest.

official = pytest.mark.integration


@pytest.fixture(scope="module")
def check5():
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"no committed {NAME} manifest: the oracle has not run "
                    "on the official Qwen-Image 2.1 VAE")
    return fixture(NAME)


@pytest.fixture(scope="module")
def vae_path(check5):
    from gmlx.image.vae import read_header

    want = check5[1]["files"]["vae"]["sha256"]
    found = vae_files(FAMILY)
    for p in found:
        if p.suffix == ".gguf" and read_header(str(p)).metadata.get(
                "gmlx.vae.source_sha256") == want:
            return p
    for p in found:
        if p.suffix == ".safetensors" and sha256(p) == want:
            return p
    pytest.skip("the fixture's VAE is not under GMLX_TEST_IMAGE_DIR")


@pytest.fixture(scope="module")
def vaes(vae_path):
    from gmlx.image.vae import load_vae

    cache = {}

    def get(dtype, conv=None):
        if (dtype, conv) not in cache:
            cache[dtype, conv] = load_vae(vae_path, family=FAMILY,
                                          dtype=dtype, conv=conv)
        return cache[dtype, conv]

    return get


def _tf32() -> bool:
    from block_gates import tf32

    return tf32()


def _numerics() -> str:
    return "TF32 on" if _tf32() else "TF32 off"


def _gate_without_tf32(request) -> None:
    """Rerun this test in a child process with TF32 off, where it gates."""
    env = dict(os.environ, MLX_ENABLE_TF32="0")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
         request.node.nodeid], cwd=request.config.rootpath, env=env,
        capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "[check5]" in line:
            print(line[line.index("[check5]"):])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


@official
def test_latent_statistics_are_the_oracle_ones(check5, vaes):
    _, m = check5
    cfg = vaes("fp32").config
    assert list(cfg.latents_mean) == m["inputs"]["latents_mean"]
    assert list(cfg.latents_std) == m["inputs"]["latents_std"]


def _decodes(arrays, m, s: str) -> list:
    """The decodes the fixture holds at size ``s``, each (name, latents
    key, decoded key, prefix of its tolerances): the latents of the
    opaque image, and where the oracle decoded them the latents of the
    image with alpha, whose alpha channel is a ramp, a clear disk and
    noise."""
    rows = [("opaque", f"latents_{s}", f"decoded_{s}", "decode")]
    if f"latents_rgba_{s}" in arrays:
        rows.append(("alpha", f"latents_rgba_{s}", f"decoded_rgba_{s}",
                     "decode_rgba"))
    return rows


@official
def test_the_fixture_decodes_real_transparency(check5):
    """The latents of the opaque image decode to an alpha near one, so the
    fixture also holds the decode of the latents of the image with alpha
    at the first size."""
    arrays, m = check5
    first = _tags(m)[0]
    assert [name for name, *_ in _decodes(arrays, m, first)] == [
        "opaque", "alpha"]
    r = m["results"]["sizes"][first]
    alpha = to_unit(arrays[f"decoded_rgba_{first}"][:, 3])
    between = float(np.mean((alpha > 0.01) & (alpha < 0.99)))
    assert between == pytest.approx(r["decode_rgba_alpha_share_between"])
    assert between > 0.4 and float(np.mean(alpha <= 0.01)) > 0.05
    assert (alpha.min(), alpha.max()) == (0.0, 1.0)
    # Against that, the alpha of the opaque image's latents.
    opaque = to_unit(arrays[f"decoded_{first}"][:, 3])
    assert float(np.mean(opaque > 0.99)) > 0.9
    # Each fp32 gate is twice the oracle's own gap against float64.
    for s in _tags(m):
        t, r = m["tolerances"]["sizes"][s], m["results"]["sizes"][s]
        assert t["decode_fp32_max_abs"] == pytest.approx(
            2 * r["decode_fp32_vs_fp64"]["max_abs"])
        assert t["encode_rgba_fp32_max_abs"] == pytest.approx(max(
            1e-3, 2 * r["encode_rgba_fp32_vs_fp64"]["max_abs"]))
        assert fx.max_abs(arrays[f"decoded_{s}"],
                          arrays[f"decoded_f64_{s}"]) == pytest.approx(
            r["decode_fp32_vs_fp64"]["max_abs"])
    t, r = m["tolerances"]["sizes"][first], m["results"]["sizes"][first]
    assert t["decode_rgba_fp32_max_abs"] == pytest.approx(
        2 * r["decode_rgba_fp32_vs_fp64"]["max_abs"])


@official
def test_fp32_decode_is_within_max_abs(check5, vaes, vae_path, request):
    """Whole fp32 decodes, forced. At 1024 pixels the largest is priced at
    10.26 GiB on the exact conv path and 15.49 GiB on Winograd. The gate
    is twice the oracle's own fp32 against float64 gap on that decode.
    The max abs is over the four channels, so the alpha is held to the
    oracle's fourth channel at the tolerance of the colors, on the decode
    whose alpha is near one and on the one whose alpha is a ramp, and its
    own reading is printed."""
    from gmlx.image.vae.conv import CONV, CONV_PATHS

    arrays, m = check5
    bad = []
    for conv in CONV_PATHS:
        vae = vaes("fp32", conv)
        for s in _whole_tags(m):
            for name, lat, dec, key in _decodes(arrays, m, s):
                got = _decode(vae, arrays[lat], tiled=False)
                ref = arrays[dec]
                assert got.shape == ref.shape and got.shape[1] == 4
                err = fx.max_abs(got, ref)
                alpha = fx.max_abs(got[:, 3:], ref[:, 3:])
                tol = m["tolerances"]["sizes"][s][f"{key}_fp32_max_abs"]
                print(f"[check5] {vae_path.name} {s} fp32 {conv} decode, "
                      f"{name}, {_numerics()}, max abs {err:.3e} (alpha "
                      f"{alpha:.3e}), {err / tol:.2f} of its gate "
                      f"{tol:.3e}, rel L2 {fx.rel_l2(got, ref):.3e}")
                if conv == CONV and err > tol:
                    bad.append((s, name))
    if _tf32():
        _gate_without_tf32(request)
    else:
        assert not bad


@official
def test_bf16_decode_psnr(check5, vaes):
    """Whole bf16 decodes, forced. At 1024 pixels the largest is priced at
    6.80 GiB on the exact conv path and 8.91 GiB on Winograd. The alpha
    alone, against the oracle's fourth channel, is held to the gate of the
    four channels where it is near one, and to the oracle's own bf16 gap
    on the alpha channel where it is a ramp."""
    from gmlx.image.vae.conv import CONV, CONV_PATHS

    arrays, m = check5
    bad = []
    for conv in CONV_PATHS:
        vae = vaes("bf16", conv)
        for s in _whole_tags(m):
            for name, lat, dec, key in _decodes(arrays, m, s):
                got = _decode(vae, arrays[lat], tiled=False)
                ref = arrays[dec]
                assert got.shape == ref.shape and got.shape[1] == 4
                db = fx.psnr(to_unit(got), to_unit(ref), 1.0)
                alpha = fx.psnr(to_unit(got[:, 3:]), to_unit(ref[:, 3:]),
                                1.0)
                t = m["tolerances"]["sizes"][s]
                tol = t[f"{key}_bf16_psnr_db"]
                tol_a = t.get(f"{key}_bf16_alpha_psnr_db", tol)
                print(f"[check5] {s} bf16 {conv} decode, {name}, PSNR "
                      f"{db:.2f} dB, gate {tol:.2f} dB; alpha {alpha:.2f} "
                      f"dB, gate {tol_a:.2f} dB; max abs "
                      f"{fx.max_abs(got, ref):.3e}")
                if conv == CONV and (db < tol or alpha < tol_a):
                    bad.append((s, name))
    assert not bad


# The encoder means of the fixture: the image, the array of the oracle's
# mean, its fp32 against float64 gap in the results, and the manifest's
# max abs tolerance.
ENCODED = (("opaque", "encode_mean", "encode_fp32_vs_fp64",
            "encode_fp32_max_abs"),
           ("alpha", "encode_mean_rgba", "encode_rgba_fp32_vs_fp64",
            "encode_rgba_fp32_max_abs"))


def _encoder_mean_rows(vae, arrays, m, device: str) -> list:
    """The fp32 encoder means on ``device`` that miss the gate: relative
    L2 against the oracle's float64 mean, at twice the oracle's own fp32
    against float64 relative L2."""
    bad = []
    for s in _whole_tags(m):
        image = arrays[f"image_{s}"]
        rgba = np.concatenate([image, arrays[f"alpha_{s}"]], axis=1)
        for what, key, gap, max_key in ENCODED:
            got = _guarded_mean(vae, image if what == "opaque" else rgba, m,
                                tiled=False)
            rel = fx.rel_l2(got, arrays[f"{key}_f64_{s}"])
            gate = 2 * m["results"]["sizes"][s][gap]["rel_l2"]
            err = fx.max_abs(got, arrays[f"{key}_{s}"])
            tol = m["tolerances"]["sizes"][s][max_key]
            print(f"[check5] {s} fp32 encode mean, {what}, {device}, "
                  f"{_numerics()}, rel L2 against float64 {rel:.3e}, "
                  f"{rel / gate:.2f} of its gate {gate:.3e}; max abs "
                  f"against fp32 {err:.3e}, {err / tol:.2f} of the "
                  f"manifest's {tol:.3e}")
            if not rel <= gate:
                bad.append((device, s, what))
    return bad


@official
def test_fp32_encoder_mean(check5, vaes, request):
    """Whole fp32 encodes, forced, through the guarded encode, on the GPU
    with exact float32 products and on the MLX CPU stream. At 1024 pixels
    the largest is priced at 5.53 GiB."""
    arrays, m = check5
    bad = _encoder_mean_rows(vaes("fp32"), arrays, m, "GPU")
    if _tf32():
        _gate_without_tf32(request)
        return
    with mx.stream(mx.cpu):
        bad += _encoder_mean_rows(vaes("fp32"), arrays, m, "CPU stream")
    assert not bad


@official
def test_bf16_encoder_mean(check5, vaes):
    """Whole bf16 encodes, forced, through the guarded encode. At 1024
    pixels the largest is priced at 3.69 GiB."""
    arrays, m = check5
    for s in _whole_tags(m):
        got = _guarded_mean(vaes("bf16"), arrays[f"image_{s}"], m,
                            tiled=False)
        ref = arrays[f"encode_mean_{s}"]
        tol = m["tolerances"]["sizes"][s]["encode_bf16_rel_l2"]
        rel = fx.rel_l2(got, ref)
        print(f"[check5] {s} bf16 encode mean max abs "
              f"{fx.max_abs(got, ref):.3e}, rel L2 {rel:.3e}, tolerance "
              f"{tol:.3e}")
        assert rel <= tol, s


@official
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_a_tiled_decode_is_the_whole_decode_at_the_largest_size(
        check5, vaes, dtype):
    """A tiled decode beside a whole one, both forced. The part of the
    decoder that reads the whole latent runs once, and the tiles of the
    rest are cropped at the reach of a cut, so the two are one result: in
    fp32 they are equal in bits, with TF32 on or off, and in bf16 they
    are held within 2e-3 max abs, the rounding bound of the tiny model,
    and read 0 on an M5 Max. At 1024 pixels the whole pass is priced at
    6.80 GiB in bf16 and 10.26 GiB in fp32, and the tiled one at 2.91 and
    4.65 GiB."""
    from gmlx.image import memfit
    from gmlx.image.vae import tiling

    arrays, m = check5
    s = _whole_tags(m)[-1]
    h, w = m["inputs"]["sizes"][-1]
    assert h > 512
    vae = vaes(dtype)
    kw = tiling.price_args(vae, "decode", 1)
    price = memfit.vae_peak("decode", h, w, **kw)
    # The automatic choice is whole with room for it and tiles without.
    assert memfit.vae_tiling("decode", h, w, free=price, **kw) is False
    assert memfit.vae_tiling("decode", h, w, free=price - 1, **kw) is True
    lat = arrays[f"latents_{s}"]
    tiled = _decode(vae, lat, tiled=True)
    whole = _decode(vae, lat, tiled=False)
    err = fx.max_abs(tiled, whole)
    ref = arrays[f"decoded_{s}"]
    db = fx.psnr(to_unit(tiled), to_unit(ref), 1.0)
    print(f"[check5] {s} {dtype} tiled decode, {_numerics()}: max abs "
          f"{err:.3e} against the whole decode, PSNR {db:.2f} dB against "
          f"the oracle")
    if dtype == "fp32":
        assert np.array_equal(tiled, whole)
    else:
        assert err <= 2e-3
        assert db >= m["tolerances"]["sizes"][s]["decode_bf16_psnr_db"]


@official
def test_tiled_bf16_encode_at_the_largest_size(check5, vaes):
    """A tiled bf16 encode beside a whole one, forced, and two whole
    decodes, forced. The tiles of an encode are cropped at the reach of a
    cut, so it is the whole encode up to the rounding of the device. At
    1024 pixels the whole encode is priced at 3.69 GiB and each whole
    decode at 6.80 GiB."""
    arrays, m = check5
    s = _whole_tags(m)[-1]
    vae = vaes("bf16")
    image = arrays[f"image_{s}"]
    whole = _encode_latents(vae, image, tiled=False)
    tiled = _encode_latents(vae, image, tiled=True)
    rel = fx.rel_l2(tiled, whole)
    a = _decode(vae, whole, tiled=False)
    b = _decode(vae, tiled, tiled=False)
    db = fx.psnr(to_unit(b), to_unit(a), 1.0)
    print(f"[check5] {s} bf16 tiled encode rel L2 {rel:.3e} against the "
          f"whole encode, round trip {db:.2f} dB against the whole round "
          f"trip")
    assert m["inputs"]["sizes"][-1][0] > 512
    assert rel < TILED_ENCODE_REL_L2
    assert db >= TILED_ROUND_TRIP_DB
