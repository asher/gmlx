"""Reference images of FLUX.2 klein: the size check, pixel cap and grid
crop of the reference's encode_image_refs, the refusals, and the
conditioning a pipeline makes of a prompt with references, whose encoder
reads the prompt alone and whose references reach the DiT as VAE latents
in bf16, normalized by the VAE's batch norm. The schedule of an edit
counts the image's tokens alone. CPU tier."""
from __future__ import annotations

import dataclasses
import io
import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from test_image_api import _Encoder
from test_image_flux3_edit import _RefDit, _Vae

from gmlx.image import api
from gmlx.image.conditioning import Conditioning
from gmlx.image.families.flux2_klein import edit
from gmlx.image.families.flux2_klein import family as ff
from gmlx.image.families.flux2_klein import model as fm
from gmlx.image.registry import get_family

PIL = pytest.importorskip("PIL")


def _picture(w: int, h: int, seed: int = 0, channels: int = 3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, channels)).astype(np.uint8)


# SPDX-FileCopyrightText: Copyright 2025 Black Forest Labs, Inc.
# _official transcribes cap_min_pixels, cap_pixels and
# center_crop_to_multiple_of_x of src/flux2/sampling.py of BFL's FLUX.2
# reference at commit 50fe516 on sizes (Apache-2.0; see
# licenses/flux2-LICENSE).
def _official(width: int, height: int, limit: int):
    """(resized size, crop box) of a reference, or None when refused."""
    if width < 64 or height < 64:
        return None
    if width / height > 8 or height / width > 8:
        return None
    if width * height > limit:
        scale = math.sqrt(limit / (width * height))
        width, height = int(width * scale), int(height * scale)
    new_w, new_h = (width // 16) * 16, (height // 16) * 16
    left, top = (width - new_w) // 2, (height - new_h) // 2
    return (width, height), (left, top, left + new_w, top + new_h)


ONE, SEVERAL = 2024 ** 2, 1024 ** 2
SIZES = [(600, 400), (400, 400), (608, 400), (2400, 1800), (64, 64),
         (63, 400), (512, 64), (513, 64), (64, 512), (64, 513), (2024, 2024),
         (2025, 2024), (4096, 4096), (1366, 768), (1031, 777), (8000, 1000),
         (1, 1)]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("limit", [ONE, SEVERAL])
def test_the_size_of_a_reference_is_the_reference_rule(size, limit):
    want = _official(*size, limit)
    if want is None:
        with pytest.raises(ValueError):
            edit.check_size(*size)
        return
    edit.check_size(*size)
    assert edit.fitted_size(*size, limit) == want[0]


def test_the_rows_name_their_source():
    r = ff.REFERENCES
    assert r.values == {"max_refs": 10, "one_max_pixels": ONE,
                        "multi_max_pixels": SEVERAL, "min_side": 64,
                        "max_aspect": 8, "grid": 16}
    assert r.repo == "github.com/black-forest-labs/flux2"
    assert r.revision == ff.PROMPT.revision
    parts = ff.FAMILY.edit
    assert parts is not None and parts.max_refs == 10 and parts.text_only
    assert parts.prepare is None and parts.encode is None
    one = _picture(600, 400)
    got = parts.prepare_all([one])
    assert got.crop is None and len(got) == 1
    assert parts.size(got) == (592, 400)


@pytest.mark.parametrize("size", [(600, 400), (1031, 777), (2400, 1800),
                                  (512, 64)])
@pytest.mark.parametrize("count", [1, 2])
def test_a_reference_is_fitted_with_lanczos_and_cropped_centered(size,
                                                                 count):
    from PIL import Image

    src = _picture(*size)
    limit = ONE if count == 1 else SEVERAL
    (w, h), box = _official(*size, limit)
    fitted = Image.fromarray(src)
    if (w, h) != size:
        fitted = fitted.resize((w, h), Image.Resampling.LANCZOS)
    want = np.asarray(fitted.crop(box))
    refs = edit.prepare([src] * count)
    assert refs.crop is None and len(refs) == count
    for ref in refs:
        assert (ref.pixels[..., 3] == 255).all()
        assert np.array_equal(ref.pixels[..., :3], want)
    assert edit.default_size(refs) == (box[2] - box[0], box[3] - box[1])
    assert edit.default_size(()) is None


def test_one_reference_keeps_more_pixels_than_each_of_several():
    src = _picture(2400, 1800)
    (one,) = edit.prepare([src])
    two = edit.prepare([src, src])
    assert (one.width, one.height) == (2336, 1744)
    assert (two[0].width, two[0].height) == (1168, 880)


def test_a_reference_is_upright_and_its_alpha_dropped_not_composited():
    from PIL import Image

    src = _picture(80, 72, channels=4)
    src[..., 3] = 0
    (ref,) = edit.prepare([src])
    assert (ref.pixels[..., 3] == 255).all()
    assert np.array_equal(ref.pixels[..., :3], src[4:68, :, :3])
    # The EXIF orientation applies before the rule: 6 turns it a quarter.
    im = Image.fromarray(_picture(96, 64))
    exif = im.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    with Image.open(io.BytesIO(buf.getvalue())) as jpg:
        jpg.load()
        (turned,) = edit.prepare([jpg])
    assert (turned.width, turned.height) == (64, 96)
    # A prepared reference prepares again to itself.
    again = edit.prepare([ref])
    assert np.array_equal(again[0].pixels, ref.pixels)


def test_the_rules_refuse_what_they_cannot_prepare():
    ok = _picture(64, 64)
    with pytest.raises(edit.BadReference, match="side under 64") as e:
        edit.prepare([ok, _picture(63, 400)])
    assert e.value.index == 1
    with pytest.raises(edit.BadReference, match="than 8 to 1") as e:
        edit.prepare([_picture(513, 64)])
    assert e.value.index == 0
    with pytest.raises(ValueError, match="at most 10 references"):
        edit.prepare([ok] * 11)
    with pytest.raises(edit.BadReference, match="uint8"):
        edit.prepare([np.zeros((64, 64, 3), np.float32)])


def test_the_dit_reads_its_references_in_bf16_with_the_ids_of_their_index():
    """encode_image_refs gives its tokens in bf16, and each reference its
    time coordinate 10 + 10 i."""
    rng = np.random.default_rng(0)
    a = mx.array(rng.standard_normal((1, 8, 2, 3)).astype(np.float32))
    b = mx.array(rng.standard_normal((1, 8, 1, 2)).astype(np.float32))
    refs = fm.Flux2KleinDiT.references(None, [a, b])
    assert refs.tokens.dtype == mx.bfloat16
    toks = np.array(refs.tokens.astype(mx.float32))
    plain = np.concatenate([np.array(fm.tokens(z)) for z in (a, b)], axis=1)
    want = np.concatenate([np.array(fm.tokens(z).astype(mx.bfloat16)
                                    .astype(mx.float32)) for z in (a, b)],
                          axis=1)
    assert np.array_equal(toks, want)
    assert not np.array_equal(toks, plain)
    assert sorted(set(refs.ids[:, 0].tolist())) == [10.0, 20.0]


@pytest.fixture
def pipe(monkeypatch):
    from gmlx.image.vae import tiling

    monkeypatch.setattr(tiling, "_plan", lambda *a, **k: False)
    fam = dataclasses.replace(get_family("z_image"),
                              edit=get_family("flux2_klein").edit)
    return api.ImagePipeline(dit=_RefDit(), family=fam, variant="turbo",
                             encoder=_Encoder(), vae=_Vae())


def test_a_text_only_edit_encodes_the_prompt_alone(pipe):
    a, b = _picture(600, 400, 1), _picture(400, 400, 2)
    text = api.encode_text(pipe, "ab")
    cond = api.encode_text(pipe, "ab", images=[a, b], ref_latents=False)
    assert pipe.encoder.encoded == ["ab"]
    assert cond.key == text.key and cond.features is text.features
    assert cond.image_slots == () and cond.pending
    assert [r.identity for r in cond.refs] == [
        r.identity for r in edit.prepare([a, b])]
    assert pipe.vae.calls == []
    done = api.encode_text(pipe, "ab", images=[a, b])
    assert not done.pending and len(done.ref_latents) == 2
    assert pipe.vae.calls == [(1, 400, 592, 3), (1, 400, 400, 3)]


def test_references_take_no_area_and_prepared_ones_pass_through(pipe):
    a = _picture(600, 400)
    refs = api.references(pipe, [a])
    assert refs.crop is None and [(r.width, r.height) for r in refs] == [
        (592, 400)]
    assert api.references(pipe, list(refs)) == tuple(refs)
    with pytest.raises(ValueError, match="takes no reference area"):
        api.references(pipe, [a], area=512 * 512)
    with pytest.raises(ValueError, match="at most 10 reference images"):
        api.references(pipe, [a] * 11)


class _Rows:
    """A zero velocity that keeps the sigma and the reference count of
    each row it runs."""

    def __init__(self):
        self.cfg = SimpleNamespace(in_channels=4, patch=1, flow_map=False)
        self.act_dtype = mx.float32
        self.rows = []

    def references(self, latents):
        return [np.array(z) for z in latents]

    def __call__(self, latents, sigma, cap_feats, refs=None):
        self.rows.append((float(np.asarray(sigma)[0]),
                          0 if refs is None else len(refs)))
        return mx.zeros_like(latents)


def test_the_schedule_of_an_edit_counts_the_image_tokens_alone():
    """get_schedule reads the image's tokens, as the reference's
    denoise does, and never the tokens of its references."""
    fam = get_family("flux2_klein")
    dit = _Rows()
    pipe = api.ImagePipeline(dit=dit, family=fam, variant="distilled")
    ref = mx.zeros((4, 64, 64))
    cond = Conditioning(mx.zeros((3, 4)), ref_latents=(ref,))
    with mx.stream(mx.cpu):
        api.sample(pipe, cond, height=256, width=256, steps=4, seed=0)
    image = fam.schedule(4, tokens=16 * 16)
    counted = fam.schedule(4, tokens=16 * 16 + 64 * 64)
    assert not np.allclose(image, counted)
    assert [s for s, _ in dit.rows] == [float(s) for s in image[:-1]]
    assert {n for _, n in dit.rows} == {1}
    assert np.array_equal(api.image_sigmas(pipe, 4, 256, 256), image)


def _tiny_flux2_vae():
    """A tiny FLUX.2 autoencoder on its initial weights, in float32, with
    batch norm statistics far from 0 and 1."""
    from gmlx.image.vae import flux2_ae

    vae = flux2_ae.Flux2AE(flux2_ae.Flux2AEConfig(
        in_channels=3, out_channels=3, latent_channels=8,
        block_out_channels=(32, 64), layers_per_block=1,
        norm_num_groups=32, eps=1e-6))
    rng = np.random.default_rng(5)
    n = vae.config.model_channels
    vae.bn.running_mean = mx.array(rng.standard_normal(n), mx.float32)
    vae.bn.running_var = mx.array(rng.uniform(0.5, 2.0, n), mx.float32)
    return vae


def test_reference_latents_are_normalized_by_the_batch_norm(monkeypatch):
    """A reference's latents are the mean of the encode, patchified 2 x 2
    channel first and normalized by the batch norm statistics, as the
    reference's encode does, and not the patchified mean alone."""
    from gmlx.image.vae import tiling

    monkeypatch.setattr(tiling, "_plan", lambda *a, **k: False)
    vae = _tiny_flux2_vae()
    pipe = api.ImagePipeline(dit=_Rows(), family=get_family("flux2_klein"),
                             variant="distilled", vae=vae)
    (ref,) = edit.prepare([_picture(96, 64, 7)])
    cond = Conditioning(mx.zeros((3, 4)), refs=(ref,))
    with mx.stream(mx.cpu):
        (got,) = api.encode_references(pipe, cond).ref_latents
        x = mx.array(ref.pixels[..., :3].astype(np.float32) / 255.0)[None]
        mean, _ = vae.encode(x * 2 - 1)
        mx.eval(got, mean)
    m = np.array(mean.astype(mx.float32))
    b, hh, ww, c = m.shape
    patched = (m.reshape(b, hh // 2, 2, ww // 2, 2, c)
               .transpose(0, 5, 2, 4, 1, 3)
               .reshape(b, c * 4, hh // 2, ww // 2))[0]
    mu = np.array(vae.bn.running_mean)[:, None, None]
    sd = np.sqrt(np.array(vae.bn.running_var) + vae.config.bn_eps)[:, None,
                                                                    None]
    want = (patched - mu) / sd
    got = np.array(got)
    assert got.shape == want.shape == (32, 64 // 4, 96 // 4)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)
    assert not np.allclose(got, patched, atol=0.1)
