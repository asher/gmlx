"""Check 5 of the image acceptance gate for MiniMax-H3-Image: the VAE's
decode and encode against ComfyUI's MiniMaxH3ImageVAE on the same file.

scripts/image_oracle/check5_h3_image_vae.py writes the fixtures: every
encode and decode in float64, the fp32 latents that every decode takes,
the class's draws for its sampled encode, and the oracle's own fp32 and
float16 errors against float64. Each gate is twice the oracle's error:
fp32 and a float16 encode by relative L2, a float16 decode by PSNR, at the
oracle's PSNR less 20 log10(2) dB. The fp32 gates run with TF32 off. The
encode holds the mean, and the sample from the mean and the clamped log
variance with the class's draws, normalized, as ComfyUI encodes.

The self-test runs ComfyUI's class at a tiny size on seeded weights; its
fixture is committed with the model's names and shapes, so these tests
write the same file and run every gate on the CPU tier. Each model of
WRONG gets one rule wrong and must read at least twice its gate at one
item. The VAE loads with ``load_vae(..., rows={"h3_image": ROW})``, as the
family is not discovered yet. The cases on the published file need it
under KQUANT_TEST_GGUF_DIR and the fixture under GMLX_IMAGE_FIXTURES.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from block_gates import exact_process, rerun_exact
from oracle_fixture import ORACLE, ROOT, committed, committed_fixture, fixture
from oracle_fixture import fx

from gmlx.image import pacing
from gmlx.image.h3 import tiles, vit
from gmlx.image.vae import h3_image as hv
from gmlx.image.vae import reader

TINY = "selftest_h3_image_vae"
REAL = "check5_h3_image_vae"
TWICE_DB = 20 * math.log10(2)
ROWS = {"h3_image": hv.ROW}
SPATIAL = hv.H3ImageVAEConfig().spatial


def _seeded():
    path = ROOT / "scripts" / "video_oracle" / "h3_vae_tiny.py"
    spec = importlib.util.spec_from_file_location("_h3_vae_tiny_i", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seeded = _seeded()


def _load(path, dtype):
    return reader.load_vae(path, family="h3_image", dtype=dtype, rows=ROWS)


def _image(a) -> np.ndarray:
    """(1, C, 1, H, W) to (1, H, W, C)."""
    return np.moveaxis(np.asarray(a)[:, :, 0], 1, -1)


def _share(got, ref: np.ndarray, tol: dict, mode: str) -> float:
    """``got`` against ``ref`` as a share of its gate in ``mode``: over 1
    misses, and a wrong shape misses by any measure."""
    got = np.asarray(got, np.float64)
    if got.shape != ref.shape:
        return math.inf
    if mode == "fp32":
        return fx.rel_l2(got, ref) / tol["fp32_rel_l2"]
    if "f16_psnr_db" in tol:
        return 10 ** ((tol["f16_psnr_db"] - fx.psnr(got, ref, 1.0)) / 20)
    return fx.rel_l2(got, ref) / tol["f16_rel_l2"]


def shares(path, arrays: dict, m: dict, mode: str) -> dict:
    """Each gated item as a share of its gate in ``mode``."""
    tol, cases = m["tolerances"]["items"], m["inputs"]["cases"]
    vae = _load(path, mx.float32 if mode == "fp32" else mx.float16)
    out = {}

    def gate(key, got, ref):
        out[key] = _share(got, ref, tol[key], mode)

    for tag in cases["images"]:
        mean, logvar = vae.encode(mx.array(_image(arrays[f"input_{tag}"])))
        gate(f"mean {tag}", mean.transpose(0, 3, 1, 2),
             arrays[f"mean_f64_{tag}"][:, :, 0])
        eps = vae.to_vae_layout(mx.array(arrays[f"noise_{tag}"][:, :, 0]))
        gate(f"encode {tag}",
             vae.to_model_space(mean + mx.exp(0.5 * logvar) * eps),
             arrays[f"latents_f64_{tag}"][:, :, 0])
    for tag in [*cases["images"], *cases["latents"]]:
        z = vae.to_vae_space(mx.array(arrays[f"latents_fp32_{tag}"][:, :, 0]))
        gate(f"decode {tag}", (vae.decode(z) + 1) / 2,
             _image(arrays[f"decoded_f64_{tag}"]))
    del vae
    mx.clear_cache()
    return out


def worst_line(label: str, got: dict) -> str:
    top = max(got, key=got.get)
    return f"{label}: {len(got)} items, worst {got[top]:.2f} of its gate at {top}"


def _items(cases: dict) -> set:
    return ({f"{k} {t}" for t in cases["images"] for k in ("encode", "mean")}
            | {f"decode {t}" for t in [*cases["images"], *cases["latents"]]})


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check5_h3_image_vae.py and commit "
                    "the manifest it writes")
    return committed(name)


@pytest.mark.parametrize("name", (TINY, REAL))
def test_the_manifests_follow_the_rule(name):
    """Every gate is twice the oracle's own error beside it, and every item
    of the cases has one."""
    m = _manifest(name)
    gaps, tol = m["results"]["gaps"], m["tolerances"]["items"]
    assert set(tol) == set(gaps) == _items(m["inputs"]["cases"])
    for k, g in gaps.items():
        t = tol[k]
        assert t["fp32_rel_l2"] == 2 * g["fp32_vs_f64"]["rel_l2"], k
        if k.startswith("decode"):
            assert t["f16_psnr_db"] == pytest.approx(
                g["f16_vs_f64"]["psnr_db"] - TWICE_DB, abs=1e-9), k
        else:
            assert t["f16_rel_l2"] == 2 * g["f16_vs_f64"]["rel_l2"], k


@pytest.mark.parametrize("name", (TINY, REAL))
def test_each_case_tiles_as_the_oracle(name):
    m = _manifest(name)
    for tag, (h, w) in m["inputs"]["cases"]["images"].items():
        plan = m["inputs"]["plans"][tag]
        for axis, n in (("tiles_y", h), ("tiles_x", w)):
            got = tiles.split_tiles(n, tiles.TILE, tiles.MIN_OVERLAP, SPATIAL)
            assert dict(zip(("starts", "lengths", "overlaps"), got)) == \
                plan[axis], (tag, axis)


def test_the_latent_statistics_are_comfyuis():
    """ComfyUI's constants, which the published file lacks, in float32."""
    stats = committed(TINY)["inputs"]["latent_stats"]
    for k in ("latents_mean", "latents_std"):
        want = np.asarray(stats[k], np.float32)
        assert np.asarray(getattr(hv, k.upper()), np.float32).tobytes() \
            == want.tobytes(), k


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    arrays, m = committed_fixture(TINY)
    path = tmp_path_factory.mktemp("h3_image_vae") / "vae.safetensors"
    digest = seeded.write_safetensors(path,
                                      seeded.weights(m["inputs"]["shapes"]))
    return path, digest, arrays, m


def test_the_seeded_file_is_the_oracle_file(tiny):
    path, digest, _, m = tiny
    assert digest == m["inputs"]["file_sha256"]


def test_the_tiny_model_in_fp32_passes_its_gates(tiny, request):
    if not exact_process():
        return rerun_exact(request, "h3 image check5 tiny fp32")
    path, _, arrays, m = tiny
    got = shares(path, arrays, m, "fp32")
    assert set(got) == _items(m["inputs"]["cases"])
    print(worst_line("h3 image check5 tiny fp32", got))
    over = {k: v for k, v in got.items() if not v <= 1}
    assert not over, over


def test_the_tiny_model_in_float16_passes_its_gates(tiny):
    path, _, arrays, m = tiny
    got = shares(path, arrays, m, "f16")
    assert set(got) == _items(m["inputs"]["cases"])
    print(worst_line("h3 image check5 tiny f16", got))
    over = {k: v for k, v in got.items() if not v <= 1}
    assert not over, over


# Wrong models, each one rule easy to get wrong, run in fp32.

def _zero_pad(mp):
    def pad(x, axis, before, after):
        widths = [(0, 0)] * x.ndim
        widths[axis] = (before, after)
        return mx.pad(x, widths)
    mp.setattr(vit, "reflect_pad", pad)


def _rotate_half_over_the_head(mp):
    """Pairs (i, i + 32) across the whole head, the frequencies on the
    first pairs, as rotate-half reads a head that rotates every dim."""
    def rope(x, cos, sin):
        half, rest = x.shape[-1] // 2, x.shape[-1] // 2 - cos.shape[-1]
        c = mx.concatenate([cos, mx.ones((cos.shape[0], rest))], axis=-1)
        s = mx.concatenate([sin, mx.zeros((sin.shape[0], rest))], axis=-1)
        c, s = c[None, :, None, :], s[None, :, None, :]
        a, b = x[..., :half], x[..., half:]
        return mx.concatenate([a * c - b * s, a * s + b * c], axis=-1)
    mp.setattr(vit, "split_half_rope", rope)


def _positions_unscaled(mp):
    """Each token at its index on each axis, as a RoPE over pixel rows
    reads it, in place of its centered place on [-1, 1]."""
    def coords(*sizes):
        grid = mx.meshgrid(*[mx.arange(n, dtype=mx.float32) for n in sizes],
                           indexing="ij")
        return mx.stack([g.reshape(-1) for g in grid], axis=-1)
    mp.setattr(vit, "token_coords", coords)


def _tiles_not_blended(mp):
    mp.setattr(vit, "blend", lambda a, b, extent, axis: b)


def _pixels_not_denormalized(mp):
    mp.setattr(hv.H3ImageVAE, "finish",
               lambda self, raw: mx.clip(raw, 0.0, 1.0) * 2.0 - 1.0)


WRONG = {"zero pad": _zero_pad,
         "rotate half over the head": _rotate_half_over_the_head,
         "positions unscaled": _positions_unscaled,
         "tiles not blended": _tiles_not_blended,
         "pixels not denormalized": _pixels_not_denormalized}


@pytest.mark.parametrize("name", list(WRONG))
def test_each_wrong_model_misses_a_gate(tiny, request, monkeypatch, name):
    if not exact_process():
        return rerun_exact(request, f"h3 image check5 wrong {name}")
    path, _, arrays, m = tiny
    WRONG[name](monkeypatch)
    got = shares(path, arrays, m, "fp32")
    print(worst_line(f"h3 image check5 wrong {name}", got))
    assert max(got.values()) >= 2, got


class _Evaluating:
    def __init__(self):
        self.points = 0

    def __call__(self, arrays):
        self.points += 1
        mx.eval(*arrays)


def test_the_decoder_pauses_per_block_with_the_same_image(tiny):
    """A pacer sees one point after each decoder block of each forward, a
    row of two tiles a forward over 2x2 tiles, and the image equals that
    of a decode with no pacer, bit for bit."""
    vae = _load(tiny[0], mx.float32)
    z = mx.random.normal((1, 17, 17, 24), key=mx.random.key(1))
    with mx.stream(mx.cpu):
        plain = vae.decode(z)
        pacer = _Evaluating()
        with pacing.paced(pacer):
            paced = vae.decode(z)
    assert pacer.points == 2 * vae.config.layers
    assert np.array_equal(np.array(plain), np.array(paced))


def test_the_mask_token_is_read_and_left_unused(tiny):
    """The file's decoder.mask_token is the one tensor of the file the
    module holds no parameter for, as ComfyUI keeps it unused."""
    header = reader.read_header(str(tiny[0]), own_names=True)
    vae = _load(tiny[0], mx.float32)
    from mlx.utils import tree_flatten

    params = {k for k, _ in tree_flatten(vae.parameters())}
    assert set(header.shapes) - params == {"decoder.mask_token"}


def _find(root: Path, want: dict) -> str:
    for p in sorted(root.rglob(want["name"])):
        if p.is_file() and os.path.getsize(p) == want["size"]:
            if fx.file_identity(str(p))["sha256"] != want["sha256"]:
                pytest.fail(f"{p} is not the file the oracle read")
            return str(p)
    pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")


@pytest.mark.integration
@pytest.mark.parametrize("mode", ("fp32", "f16"))
def test_check5_on_the_published_file(gguf_dir, request, mode):
    m = _manifest(REAL)
    path = _find(gguf_dir, m["files"]["vae"])
    if mode == "fp32" and not exact_process():
        return rerun_exact(request, "h3 image check5 fp32")
    arrays, m = fixture(REAL)
    got = shares(path, arrays, m, mode)
    assert set(got) == _items(m["inputs"]["cases"])
    print(worst_line(f"h3 image check5 {mode}", got))
    over = {k: v for k, v in got.items() if not v <= 1}
    assert not over, over
