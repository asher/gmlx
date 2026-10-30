"""Check 5 of the video acceptance gate for MiniMax-H3: the video VAE, TAEH3
and the audio VAE against ComfyUI's classes on the same files.

scripts/video_oracle/check5_h3.py writes the fixtures: every encode and
decode in float64, the fp32 latents that every decode takes, and the
oracle's own fp32 and float16 errors against float64. Each gate is twice
the oracle's error: fp32 and a float16 encode by relative L2, a float16
decode by PSNR, at the oracle's PSNR less 20 log10(2) dB. The fp32 gates
run with TF32 off, and the audio VAE's also in the caller's process, where
its products stay float32.

The self-test runs ComfyUI's classes at a tiny size on seeded weights. Its
fixture is committed with each model's names and shapes, so these tests
write the same files, check their sha256 and run every gate on the CPU
tier. Each model of WRONG gets one rule wrong and must read at least
twice its gate at one item.

The cases on the published files need them under KQUANT_TEST_GGUF_DIR and
the fixture under GMLX_VIDEO_FIXTURES, and skip until both are there.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from block_gates import exact_process, rerun_exact, tf32
from video_oracle_fixture import (ROOT, committed, committed_fixture,
                                  fixture, fixture_present, fx)

from gmlx.image.vae.group_norm import group_norm
from gmlx.image.vae.reader import VAEError
from gmlx.image import float32 as f32
from gmlx.video.vae import chunking, h3_audio, h3_video
from gmlx.video.vae import tiny as taeh3
from gmlx.video.vae import reader as vr

TINY = "selftest_h3_vae"
REAL = "check5_h3"
FAMILIES = {"vae": "h3", "tiny": "h3_tiny", "audio_vae": "h3_audio"}
TWICE_DB = 20 * math.log10(2)
# Pixels per latent on each spatial axis.
SPATIAL = 16


def _seeded():
    path = ROOT / "scripts" / "video_oracle" / "h3_vae_tiny.py"
    spec = importlib.util.spec_from_file_location("_h3_vae_tiny_t", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seeded = _seeded()


def _last(a) -> np.ndarray:
    """Channels first to channels last."""
    return np.moveaxis(np.asarray(a), 1, -1)


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


def shares(files: dict, arrays: dict, m: dict, mode: str,
           roles=tuple(FAMILIES)) -> dict:
    """Each gated item of the models in ``roles`` as a share of its gate
    in ``mode``. The audio VAE runs in fp32 alone."""
    tol, cases = m["tolerances"]["items"], m["inputs"]["cases"]
    dt = mx.float32 if mode == "fp32" else mx.float16
    coded = [*cases["clips"], *cases["images"]]
    out = {}

    def gate(key, got, ref):
        out[key] = _share(got, ref, tol[key], mode)

    if "vae" in roles:
        vae = vr.load_vae(files["vae"], family="h3", dtype=dt)
        for tag in coded:
            gate(f"encode {tag}", vae.encode(mx.array(
                _last(arrays[f"input_{tag}"]))), arrays[f"latents_f64_{tag}"])
        for tag in [*coded, *cases["latents"]]:
            gate(f"decode {tag}", vae.decode(mx.array(
                arrays[f"latents_fp32_{tag}"])),
                _last(arrays[f"decoded_f64_{tag}"]))
        del vae
        mx.clear_cache()
    if "tiny" in roles:
        tae = vr.load_vae(files["tiny"], family="h3_tiny", dtype=dt)
        for tag in _tiny_tags(cases):
            gate(f"tiny {tag}", tae.decode(mx.array(
                arrays[f"latents_fp32_{tag}"])),
                _last(arrays[f"tiny_f64_{tag}"]))
        del tae
        mx.clear_cache()
    if "audio_vae" in roles and mode == "fp32":
        aud = vr.load_vae(files["audio_vae"], family="h3_audio",
                          dtype=mx.float32)
        gate("audio encode", aud.encode(mx.array(arrays["wave"])),
             arrays["wave_latents_f64"])
        gate("audio decode", aud.decode(mx.array(arrays["wave_latents_fp32"])),
             arrays["wave_decoded_f64"])
        del aud
        mx.clear_cache()
    return out


def worst_line(label: str, got: dict) -> str:
    top = max(got, key=got.get)
    return f"{label}: {len(got)} items, worst {got[top]:.2f} of its gate at {top}"


AUDIO = {"audio encode", "audio decode"}


def _tiny_tags(cases: dict) -> list:
    """The cases TAEH3 decodes: the clips, the images and the seeded
    latents of more than one frame."""
    return [*cases["clips"], *cases["images"],
            *(t for t, (f, _, _) in cases["latents"].items() if f > 1)]


def _items(cases: dict) -> set:
    coded = [*cases["clips"], *cases["images"]]
    return ({f"encode {t}" for t in coded}
            | {f"decode {t}" for t in [*coded, *cases["latents"]]}
            | {f"tiny {t}" for t in _tiny_tags(cases)}
            | AUDIO)


@pytest.mark.parametrize("name", (TINY, REAL))
def test_the_manifests_follow_the_rule(name):
    """Every gate is twice the oracle's own error beside it, and every item
    of the cases has one."""
    m = committed(name)
    gaps, tol = m["results"]["gaps"], m["tolerances"]["items"]
    assert set(tol) == set(gaps) == _items(m["inputs"]["cases"])
    for k, g in gaps.items():
        t = tol[k]
        assert t["fp32_rel_l2"] == 2 * g["fp32_vs_f64"]["rel_l2"], k
        if k.startswith("audio"):
            assert set(t) == {"fp32_rel_l2"}, k
        elif k.startswith("encode"):
            assert t["f16_rel_l2"] == 2 * g["f16_vs_f64"]["rel_l2"], k
        else:
            assert t["f16_psnr_db"] == pytest.approx(
                g["f16_vs_f64"]["psnr_db"] - TWICE_DB, abs=1e-9), k


def _tiles(n: int) -> dict:
    got = chunking.split_tiles(n, h3_video.TILE, h3_video.MIN_OVERLAP,
                               SPATIAL)
    return dict(zip(("starts", "lengths", "overlaps"), got))


@pytest.mark.parametrize("name", (TINY, REAL))
def test_each_case_plan_is_the_oracle_plan(name):
    """The latent count, the tiles of both axes and the decode chunks of
    each case are the ones the class computed."""
    m = committed(name)
    cases, c = m["inputs"]["cases"], h3_video.CHUNKS
    shapes = {**{t: v for t, v in cases["clips"].items()},
              **{t: [1, *v] for t, v in cases["images"].items()}}
    plans = m["inputs"]["plans"]
    assert set(plans) == {*shapes, *cases["latents"]}
    for tag, (frames, h, w) in shapes.items():
        plan = plans[tag]
        lat = c.latents(frames)
        assert plan["latents"] == lat, tag
        assert plan["tiles_y"] == _tiles(h), tag
        assert plan["tiles_x"] == _tiles(w), tag
        if frames > 1:
            assert plan["encode_pieces"] == -(-frames // c.clip), tag
            assert (plan["decode_pad_latents"], plan["decode_runs"]) == \
                c.runs(lat), tag
            assert (plan["tokens_chunk_size"], plan["token_overlap"],
                    plan["token_drop"], plan["frame_pre_padding"],
                    plan["frame_overlap"]) == (c.tokens, c.overlap, c.drop,
                                               c.pre_frames, c.fade), tag


def test_the_tile_and_chunk_schedules_are_the_oracle_schedules():
    """The tiles of every side from 16 to 2048 pixels in steps of 16, and
    the decode chunks and frames out of 1 to 92 latents."""
    tables = committed(TINY)["inputs"]["plan_tables"]
    assert len(tables["tiles"]) == 128 and len(tables["chunks"]) == 92
    for n, want in tables["tiles"].items():
        assert _tiles(int(n)) == want, n
    c = h3_video.CHUNKS
    for n, want in tables["chunks"].items():
        assert c.frames(int(n)) == want["frames"], n
        if int(n) > 1:
            assert c.runs(int(n)) == (want["pad"], want["runs"]), n


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The seeded files written from the manifest's shapes, their sha256,
    the fixture's arrays and its manifest."""
    arrays, m = committed_fixture(TINY)
    d = tmp_path_factory.mktemp("h3_vae")
    files, digests = {}, {}
    for role, names in m["inputs"]["shapes"].items():
        files[role] = d / f"{role}.safetensors"
        digests[role] = seeded.write_safetensors(files[role],
                                                 seeded.weights(names))
    return files, digests, arrays, m


def test_the_seeded_files_are_the_oracle_files(tiny):
    files, digests, _, m = tiny
    assert digests == m["inputs"]["files_sha256"]
    assert set(files) == set(FAMILIES)


def test_the_tiny_models_in_fp32_pass_their_gates(tiny, request):
    if not exact_process():
        return rerun_exact(request, "h3 check5 tiny fp32")
    files, _, arrays, m = tiny
    got = shares(files, arrays, m, "fp32")
    assert set(got) == _items(m["inputs"]["cases"])
    print(worst_line("h3 check5 tiny fp32", got))
    over = {k: v for k, v in got.items() if not v <= 1}
    assert not over, over


def test_the_tiny_models_in_float16_pass_their_gates(tiny):
    files, _, arrays, m = tiny
    got = shares(files, arrays, m, "f16")
    assert set(got) == _items(m["inputs"]["cases"]) - AUDIO
    print(worst_line("h3 check5 tiny f16", got))
    over = {k: v for k, v in got.items() if not v <= 1}
    assert not over, over


def test_the_tiny_audio_vae_passes_its_gates_in_this_process(tiny):
    """The audio VAE's fp32 gates hold in the caller's process too, TF32
    or not: its products stay float32."""
    files, _, arrays, m = tiny
    got = shares(files, arrays, m, "fp32", ("audio_vae",))
    assert set(got) == AUDIO
    print(worst_line("h3 check5 tiny audio, this process", got))
    over = {k: v for k, v in got.items() if not v <= 1}
    assert not over, over


def test_tf32_products_miss_the_audio_gates(tiny, monkeypatch):
    """With its products at TF32, as MLX runs float32 on an M5-class GPU,
    the audio VAE misses a gate."""
    if not tf32():
        pytest.skip("this process multiplies float32 at float32 precision")
    for mod in (f32, h3_audio):
        monkeypatch.setattr(mod, "f32_products_exact", lambda: True)
    files, _, arrays, m = tiny
    got = shares(files, arrays, m, "fp32", ("audio_vae",))
    print(worst_line("h3 check5 tiny audio, TF32 products", got))
    assert max(got.values()) > 2, got


# Wrong models, each one rule easy to get wrong, run in fp32 on the models
# of its roles.

def _group_norm_per_clip(mp):
    """Statistics over the whole piece, as an image VAE's norm reads a
    stack of frames."""
    def call(self, x):
        b, t, h, w, c = x.shape
        y = group_norm(x.reshape(b, t * h, w, c), self.weight, self.bias,
                       h3_video.GROUPS, h3_video.GROUP_EPS)
        return y.reshape(x.shape)
    mp.setattr(h3_video.FrameGroupNorm, "__call__", call)


def _zero_pad(mp):
    def pad(x, axis, before, after):
        widths = [(0, 0)] * x.ndim
        widths[axis] = (before, after)
        return mx.pad(x, widths)
    mp.setattr(h3_video, "reflect_pad", pad)


def _front_frames_replicated(mp):
    """Each causal conv padded in front with copies of the first frame, as
    other causal video VAEs pad, in place of zeros."""
    def call(self, x, sides=None):
        pt, ph, pw = self.pad
        top, bottom, left, right = sides or (ph, ph, pw, pw)
        x = h3_video.reflect_pad(h3_video.reflect_pad(x, 2, top, bottom), 3,
                                 left, right)
        if pt:
            x = mx.concatenate([mx.repeat(x[:, :1], 2 * pt, axis=1), x], 1)
        y = h3_video.conv3d(x, self.weight, self.stride)
        return y + self.bias.astype(y.dtype)
    mp.setattr(h3_video.CausalConv3d, "__call__", call)


def _first_latents_dropped(mp):
    mp.setattr(h3_video, "drop_latents",
               lambda m: m[:, h3_video.CHUNKS.drop:])


def _rotate_half_over_the_head(mp):
    """Pairs (i, i + 32) across the whole head, the frequencies on the
    first pairs, as rotate-half reads a head that rotates every dim."""
    def rope(x, cos, sin):
        half, rest = x.shape[-1] // 2, x.shape[-1] // 2 - cos.shape[-1]
        c = mx.concatenate([cos, mx.ones((cos.shape[0], rest))], axis=-1)
        s = mx.concatenate([sin, mx.zeros((sin.shape[0], rest))], axis=-1)
        c, s = c[None, :, None, :], s[None, :, None, :]
        x1, x2 = x[..., :half], x[..., half:]
        return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)
    mp.setattr(h3_video, "split_half_rope", rope)


def _tiles_blend_raw_neighbors(mp):
    """Each decoded tile blended with its neighbors as decoded, not as
    already blended."""
    def tiled(self, z, frames=slice(None)):
        h, w = z.shape[2] * SPATIAL, z.shape[3] * SPATIAL
        ys, ylen, yo = chunking.split_tiles(h, h3_video.TILE,
                                            h3_video.MIN_OVERLAP, SPATIAL)
        xs, xlen, xo = chunking.split_tiles(w, h3_video.TILE,
                                            h3_video.MIN_OVERLAP, SPATIAL)
        raw = [[self._tile(z[:, :, y // SPATIAL:(y + yl) // SPATIAL,
                             x // SPATIAL:(x + xl) // SPATIAL])[:, frames]
                for x, xl in zip(xs, xlen)] for y, yl in zip(ys, ylen)]
        canvas = mx.zeros((*raw[0][0].shape[:2], h, w, 3))
        for i, (y, yl) in enumerate(zip(ys, ylen)):
            for j, (x, xl) in enumerate(zip(xs, xlen)):
                tile = raw[i][j]
                if i > 0:
                    tile = h3_video.blend(raw[i - 1][j], tile, yo[i - 1], 2)
                if j > 0:
                    tile = h3_video.blend(raw[i][j - 1], tile, xo[j - 1], 3)
                canvas[:, :, y:y + yl, x:x + xl] = tile
        return canvas
    mp.setattr(h3_video.H3VideoVAE, "_tiled_decode", tiled)


def _encode_tiles_blend_blended(mp):
    """Each encoded tile blended with its neighbors as already blended,
    as the decoder's tiles blend."""
    def tiled(self, x):
        sp = self.cfg.spatial
        ys, ylen, yo = chunking.split_tiles(x.shape[2], h3_video.TILE,
                                            h3_video.MIN_OVERLAP, sp)
        xs, xlen, xo = chunking.split_tiles(x.shape[3], h3_video.TILE,
                                            h3_video.MIN_OVERLAP, sp)
        rows: list = []
        for i, (y, yl) in enumerate(zip(ys, ylen)):
            row: list = []
            for j, (x0, xl) in enumerate(zip(xs, xlen)):
                m = self._moments(x[:, :, y:y + yl, x0:x0 + xl])
                if i > 0:
                    m = h3_video.blend(rows[i - 1][j], m, yo[i - 1] // sp, 2)
                if j > 0:
                    m = h3_video.blend(row[j - 1], m, xo[j - 1] // sp, 3)
                row.append(m)
            rows.append(row)
        out = []
        for i, row in enumerate(rows):
            parts = []
            for j, m in enumerate(row):
                if i < len(rows) - 1:
                    m = m[:, :, :m.shape[2] - yo[i] // sp]
                if j < len(row) - 1:
                    m = m[:, :, :, :m.shape[3] - xo[j] // sp]
                parts.append(m)
            out.append(mx.concatenate(parts, axis=3))
        return mx.concatenate(out, axis=2)
    mp.setattr(h3_video.H3VideoVAE, "_tiled_moments", tiled)


class _NoFade(chunking.TimeChunks):
    fade = property(lambda self: 0)


def _no_crossfade(mp):
    c = h3_video.CHUNKS
    mp.setattr(h3_video, "CHUNKS", _NoFade(c.clip, c.ratio, c.drop))


def _audio_left_pad(mp):
    real = h3_audio.H3AudioVAE.encode

    def encode(self, waveform):
        pad = -waveform.shape[-1] % h3_audio.HOP
        return real(self, mx.pad(waveform, [(0, 0), (0, 0), (pad, 0)]))
    mp.setattr(h3_audio.H3AudioVAE, "encode", encode)


def _exact_gelu(mp):
    mp.setattr(h3_audio.nn, "gelu_approx", nn.gelu)


def _snake_alpha_as_a_log(mp):
    mp.setattr(h3_audio.Snake, "__call__", lambda self, x: h3_audio.snake(
        x, mx.exp(self.alpha), mx.exp(self.alpha)))


# name: (the wrong rule, the models it reaches)
WRONG = {"GroupNorm over the whole piece": (_group_norm_per_clip, ("vae",)),
         "zeros in place of reflect padding": (_zero_pad, ("vae",)),
         "first frames replicated in front": (_front_frames_replicated,
                                              ("vae",)),
         "the first latents dropped": (_first_latents_dropped, ("vae",)),
         "rotate-half over the whole head": (_rotate_half_over_the_head,
                                             ("vae",)),
         "decode tiles blended with raw neighbors": (
             _tiles_blend_raw_neighbors, ("vae",)),
         "encode tiles blended with blended neighbors": (
             _encode_tiles_blend_blended, ("vae",)),
         "no crossfade between runs": (_no_crossfade, ("vae",)),
         "audio padded on the left": (_audio_left_pad, ("audio_vae",)),
         "exact GELU for the tanh form": (_exact_gelu, ("audio_vae",)),
         "Snake alpha read as a log": (_snake_alpha_as_a_log,
                                       ("audio_vae",))}


@pytest.mark.parametrize("name", list(WRONG))
def test_a_wrong_model_misses_a_gate(tiny, request, monkeypatch, name):
    if not exact_process():
        return rerun_exact(request, "h3 check5 wrong")
    files, _, arrays, m = tiny
    apply, roles = WRONG[name]
    apply(monkeypatch)
    got = shares(files, arrays, m, "fp32", roles)
    print(worst_line(f"h3 check5 wrong, {name}", got))
    # The right models read at most about half of each gate.
    assert max(got.values()) > 2, got


# The reader.

def test_the_audio_vae_runs_in_float32_alone(tiny):
    files = tiny[0]
    with pytest.raises(VAEError, match="runs in float32, not"):
        vr.load_vae(files["audio_vae"], family="h3_audio", dtype=mx.float16)


def test_the_bfloat16_the_image_vaes_take_is_refused(tiny):
    files = tiny[0]
    with pytest.raises(VAEError, match="runs in float16 or float32, not"):
        vr.load_vae(files["vae"], family="h3", dtype="bf16")


@pytest.mark.parametrize("role,family,holds", [
    ("vae", "h3_audio", "the MiniMax-H3 video VAE, the h3 VAE"),
    ("tiny", "h3", "TAEH3, the tiny MiniMax-H3 preview decoder, the "
     "h3_tiny VAE")])
def test_a_file_of_another_vae_is_refused_by_name(tiny, role, family, holds):
    files = tiny[0]
    with pytest.raises(VAEError, match=holds):
        vr.load_vae(files[role], family=family, dtype=None)


def test_a_file_missing_a_tensor_is_refused(tiny, tmp_path):
    _, _, _, m = tiny
    names = dict(m["inputs"]["shapes"]["vae"])
    del names["decoder.transformer_blocks.1.norm1.weight"]
    path = tmp_path / "vae.safetensors"
    seeded.write_safetensors(path, seeded.weights(names))
    with pytest.raises(VAEError, match="transformer_blocks.1.norm1.weight"):
        vr.load_vae(path, family="h3", dtype=None)


def _with_metadata(src, dst, meta: dict) -> None:
    data = Path(src).read_bytes()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + n])
    header["__metadata__"] = meta
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    Path(dst).write_bytes(struct.pack("<Q", len(h)) + h + data[8 + n:])


def test_a_decoder_tuned_for_stills_alone_is_refused(tiny, tmp_path):
    """ComfyUI decodes a still from its latent alone with a decoder whose
    metadata sets h3_t1_direct; gmlx refuses the file by that name."""
    path = tmp_path / "vae.safetensors"
    _with_metadata(tiny[0]["vae"], path, {"h3_t1_direct": "false"})
    vr.load_vae(path, family="h3", dtype=None)
    _with_metadata(tiny[0]["vae"], path, {"h3_t1_direct": "true",
                                          "h3_t1_output_slice": "3"})
    with pytest.raises(VAEError, match=re.escape(f"{path} sets h3_t1_direct")):
        vr.load_vae(path, family="h3", dtype=None)


def _causal(q, k, v):
    s = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float64), k) \
        / q.shape[-1] ** 0.5
    s = s + np.triu(np.full(s.shape[-2:], -np.inf), 1)
    p = np.exp(s - s.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ v


@pytest.mark.parametrize("split", [False, True], ids=["sdpa", "split"])
def test_the_audio_attention_runs_in_query_blocks(monkeypatch, split):
    """The audio attention runs 16 query rows at a time over the keys up to
    each block's end, and gives causal attention: SDPA on a device that
    multiplies float32 exactly (the CPU here), split products in a TF32
    process."""
    calls = []
    if split:
        monkeypatch.setattr(h3_audio, "f32_products_exact", lambda: False)
        real = h3_audio.exact_matmul

        def spy(a, b, **kw):
            calls.append((*a.shape[2:], *b.shape[2:]))
            return real(a, b, **kw)
        monkeypatch.setattr(h3_audio, "exact_matmul", spy)
    else:
        real = mx.fast.scaled_dot_product_attention

        def spy(q, k, v, **kw):
            calls.append((q.shape[2], k.shape[2], v.shape[2]))
            return real(q, k, v, **kw)
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    monkeypatch.setattr(h3_audio, "QUERY_BLOCK", 16)
    rng = np.random.default_rng(3)
    q, k, v = (rng.standard_normal((2, 8, 81, 32)).astype(np.float32)
               for _ in range(3))
    with mx.stream(mx.default_device() if split else mx.cpu):
        got = h3_audio.causal_attention(*map(mx.array, (q, k, v)),
                                        32 ** -0.5)
    want_calls = []
    for i in range(0, 81, 16):
        r, e = min(16, 81 - i), min(i + 16, 81)
        # scores (r, d) by (d, e), then values (r, e) by (e, d)
        want_calls += [(r, 32, 32, e), (r, e, e, 32)] if split \
            else [(r, e, e)]
    assert calls == want_calls, calls
    want = _causal(q, k, v)
    gap = np.linalg.norm(np.array(got, np.float64) - want) \
        / np.linalg.norm(want)
    assert gap < 1e-6, gap


def test_an_image_family_is_unknown_to_the_video_rows(tiny):
    with pytest.raises(VAEError, match="no VAE is known for family 'flux1'; "
                       "known: h3, h3_audio, h3_tiny"):
        vr.load_vae(tiny[0]["vae"], family="flux1", dtype=None)


# The published files.

@pytest.fixture(scope="module")
def real(gguf_dir):
    """The published files the oracle read, found by name, size and
    sha256 under KQUANT_TEST_GGUF_DIR."""
    m = committed(REAL)
    fixture_present(REAL)
    files = {}
    for role, want in m["files"].items():
        for p in sorted(Path(gguf_dir).rglob(want["name"])):
            if p.stat().st_size == want["size"]:
                break
        else:
            pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
        if fx.file_identity(p)["sha256"] != want["sha256"]:
            pytest.fail(f"{p} is not the file the oracle read")
        files[role] = p
    return files


@pytest.mark.integration
def test_check5_in_fp32(real, request):
    if not exact_process():
        return rerun_exact(request, "h3 check5 fp32")
    mx.set_cache_limit(2 << 30)
    arrays, m = fixture(REAL)
    got = shares(real, arrays, m, "fp32")
    assert set(got) == _items(m["inputs"]["cases"])
    print(worst_line("h3 check5 fp32", got))
    over = {k: v for k, v in got.items() if not v <= 1}
    assert not over, over


@pytest.mark.integration
def test_check5_audio_in_this_process(real):
    mx.set_cache_limit(2 << 30)
    arrays, m = fixture(REAL)
    got = shares(real, arrays, m, "fp32", ("audio_vae",))
    assert set(got) == AUDIO
    print(worst_line("h3 check5 audio, this process", got))
    over = {k: v for k, v in got.items() if not v <= 1}
    assert not over, over


@pytest.mark.integration
def test_check5_in_float16(real):
    mx.set_cache_limit(2 << 30)
    arrays, m = fixture(REAL)
    got = shares(real, arrays, m, "f16")
    assert set(got) == _items(m["inputs"]["cases"]) - AUDIO
    print(worst_line("h3 check5 f16", got))
    over = {k: v for k, v in got.items() if not v <= 1}
    assert not over, over


# Memory: each decoder holds one run at a time.

metal_only = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="reads MLX's Metal peak")


def _settled() -> int:
    """MLX's active memory once the GPU has finished, so the buffers the
    last eval freed are not counted."""
    mx.synchronize()
    return mx.get_active_memory()


def _peak(run) -> int:
    """MLX's peak over ``run()`` above the memory active before it."""
    base = _settled()
    mx.reset_peak_memory()
    run()
    mx.synchronize()
    return mx.get_peak_memory() - base


def _decode(model, latents: mx.array) -> tuple[int, int]:
    """The peak of a decode whose frames are dropped as they come, as a
    writer streams them, and the most it holds between two runs, both
    above the memory active before it."""
    mx.eval(latents)
    base = _settled()
    mx.reset_peak_memory()
    held = 0
    for frames in model.decode_frames(latents):
        del frames
        held = max(held, _settled() - base)
    return mx.get_peak_memory() - base, held


@metal_only
@pytest.mark.parametrize("split,d,gate", [(False, 32, 3), (True, 64, 12)],
                         ids=["sdpa", "split"])
def test_the_audio_attention_holds_one_query_block(monkeypatch, split, d,
                                                    gate):
    """Eight query blocks peak at a few arrays of one block's scores, as one
    block does: about 1.6 on SDPA at head dim 32, where MLX holds the
    scores, and about 8 on split products. With no eval between blocks
    they read about 5 and 35."""
    monkeypatch.setattr(h3_audio, "f32_products_exact", lambda: not split)
    monkeypatch.setattr(h3_audio, "QUERY_BLOCK", 128)
    q, k, v = (mx.random.normal((2, 8, 1024, d), key=mx.random.key(i))
               for i in range(3))
    mx.eval(q, k, v)
    block = 2 * 8 * 128 * 1024 * 4
    peak = _peak(lambda: mx.eval(
        h3_audio.causal_attention(q, k, v, d ** -0.5)))
    print(f"h3 check5 audio attention peak: {peak / block:.2f} blocks")
    assert peak < gate * block, peak / block


@metal_only
def test_taeh3_holds_one_run_at_a_time(tiny):
    """A run peaks under 24 times its frames, a clip of four runs as one
    run does, and between runs it holds the carry and the frames of one
    run. A carry copied in the run's own eval reads about 34 times."""
    tae = vr.load_vae(tiny[0]["tiny"], family="h3_tiny", dtype=mx.float16)
    mx.eval(tae.parameters())
    z = mx.random.normal((1, 24, 20, 8, 8), key=mx.random.key(0))
    (one, _), (four, held) = (_decode(tae, z[:, :, :n]) for n in (5, 20))
    carry = sum(3 * w * (8 * 2 ** i) ** 2 * 2
                for i, w in enumerate(taeh3.WIDTHS[:3]))
    grow = math.prod(taeh3.TIME_GROW)
    frames = (taeh3.RUN * grow - grow + 1) * (8 * 16) ** 2 * 3 * 4
    print(f"h3 check5 taeh3 peak: one run {one / 2**20:.1f} MiB, four "
          f"{four / 2**20:.1f} MiB; held {held / 2**20:.2f} MiB against "
          f"{(carry + frames) / 2**20:.2f}")
    assert one < 24 * frames, (one, frames)
    assert four < 1.3 * one, (one, four)
    assert held < 1.25 * (carry + frames), (held, carry, frames)


@metal_only
def test_the_video_decoder_holds_one_run_canvas_at_a_time(tiny):
    """Three runs of 2x3 tiles peak at one tile's peak and under 2.6 run
    canvases, and between yields hold under half a canvas: the tail the
    next run fades from, and at the last run the tail's finished frames.
    A slice that keeps the canvas alive copies it at each tile write and
    reads over 3; one that keeps the run before alive holds a canvas
    between runs."""
    vae = vr.load_vae(tiny[0]["vae"], family="h3", dtype=mx.float16)
    mx.eval(vae.parameters())
    z = mx.random.normal((1, 24, 17, 24, 33), key=mx.random.key(0))
    c = h3_video.CHUNKS
    run = c.tokens + c.overlap
    canvas = run * c.ratio * 24 * 33 * SPATIAL ** 2 * 3 * 4
    zt = z[:, :, :run, :16, :16].transpose(0, 2, 3, 4, 1)
    mx.eval(zt)
    tile = _peak(lambda: vae._tile(zt))
    peak, held = _decode(vae, z)
    print(f"h3 check5 decode peak: one tile and {(peak - tile) / canvas:.2f}"
          f" run canvases; held {held / canvas:.2f}")
    assert peak - tile < 2.6 * canvas, (peak, tile, canvas)
    assert held < 0.5 * canvas, (held, canvas)
