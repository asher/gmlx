"""Reference images of FLUX.3 klein: the canvas, pad and crop rules of the
reference pipeline, the refusals, and the conditioning a pipeline makes of
a prompt with references, whose encoder reads the prompt alone and whose
references reach the DiT as VAE latents. CPU tier."""
from __future__ import annotations

import dataclasses
import io
import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from test_image_api import _Dit, _Encoder

from gmlx.image import api
from gmlx.image.conditioning import Conditioning, Reference
from gmlx.image.families.flux3_klein import edit
from gmlx.image.families.flux3_klein import family as ff
from gmlx.image.registry import get_family

PIL = pytest.importorskip("PIL")


def _picture(w: int, h: int, seed: int = 0, channels: int = 3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, channels)).astype(np.uint8)


def _official_canvas(width: int, height: int) -> tuple:
    # reference_canvas of src/flux3/pipeline.py, as it is written.
    most, low, high = 4 * 1024 * 1024, 64, 4096

    def ceil16(v):
        return -(-v // 16) * 16

    scale = min(1.0, math.sqrt(most / (width * height)),
                high / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(high, max(low, ceil16(v))) for v in fit)
        if canvas[0] * canvas[1] <= most:
            return fit, canvas
        scale *= 0.995


SIZES = [(600, 400), (400, 400), (608, 400), (2400, 1800), (40, 24),
         (5000, 200), (200, 5000), (1, 1), (10, 12), (2048, 2048),
         (2049, 2047), (4096, 4096), (10000, 3), (1366, 768), (17, 4000)]


@pytest.mark.parametrize("size", SIZES)
def test_the_canvas_of_one_reference_is_the_reference_pipeline_rule(size):
    fit, canvas = edit.reference_canvas(*size)
    assert (fit, canvas) == _official_canvas(*size)
    assert all(v % 16 == 0 and 64 <= v <= 4096 for v in canvas)
    assert fit[0] <= canvas[0] and fit[1] <= canvas[1]
    assert canvas[0] * canvas[1] <= 4 * 1024 * 1024


def test_the_rows_name_their_source():
    r = ff.REFERENCES
    assert r.values == {"max_refs": 10, "match_max_pixels": 4 * 1024 * 1024,
                        "multi_max_pixels": 1024 * 1024, "min_side": 64,
                        "max_side": 4096, "grid": 16}
    assert r.repo == "github.com/black-forest-labs/flux-3-image-klein"
    parts = ff.FAMILY.edit
    assert parts is not None and parts.max_refs == 10 and parts.text_only
    assert parts.prepare is None and parts.encode is None
    one = _picture(600, 400)
    assert parts.prepare_all([one]).crop == edit.prepare([one]).crop
    assert parts.size(edit.prepare([one])) == (608, 400)
    # Qwen-Image 2.1 runs its references through the encoder.
    assert not get_family("qwen_image21").edit.text_only


def test_one_reference_is_padded_by_reflection_to_its_canvas():
    src = _picture(600, 400)
    ref, box = edit.prepare_reference(src)
    assert (ref.width, ref.height) == (608, 400) and box == (4, 0, 604, 400)
    px = ref.pixels
    assert (px[..., 3] == 255).all()
    assert np.array_equal(px[:, 4:604, :3], src)
    want = np.pad(src, ((0, 0), (4, 4), (0, 0)), mode="reflect")
    assert np.array_equal(px[..., :3], want)
    assert np.array_equal(px[:, 0, :3], src[:, 4])
    # One that fills its canvas has no box.
    full, box = edit.prepare_reference(_picture(608, 400))
    assert box is None and (full.width, full.height) == (608, 400)


def test_a_reference_of_16_pixels_or_less_pads_symmetric():
    src = _picture(10, 12)
    ref, box = edit.prepare_reference(src)
    assert (ref.width, ref.height) == (64, 64)
    assert box == (27, 26, 37, 38)
    want = np.pad(src, ((26, 26), (27, 27), (0, 0)), mode="symmetric")
    assert np.array_equal(ref.pixels[..., :3], want)
    wide = edit.prepare_reference(_picture(40, 24))[0]
    want = np.pad(_picture(40, 24), ((20, 20), (12, 12), (0, 0)),
                  mode="reflect")
    assert np.array_equal(wide.pixels[..., :3], want)


def test_a_large_reference_is_resized_with_lanczos_to_its_fit():
    from PIL import Image

    src = _picture(2400, 1800)
    fit, canvas = edit.reference_canvas(2400, 1800)
    ref, box = edit.prepare_reference(src)
    want = np.asarray(Image.fromarray(src).resize(
        fit, Image.Resampling.LANCZOS))
    left, top = (canvas[0] - fit[0]) // 2, (canvas[1] - fit[1]) // 2
    assert box == (left, top, left + fit[0], top + fit[1])
    assert np.array_equal(ref.pixels[top:top + fit[1], left:left + fit[0], :3],
                          want)


def test_several_references_fit_one_mib_each_and_crop_to_the_grid():
    from PIL import Image

    big = _picture(1600, 1000)
    got = edit.prepare_multi_reference(big)
    scale = math.sqrt(1024 * 1024 / (1600 * 1000))
    w, h = int(1600 * scale), int(1000 * scale)
    small = np.asarray(Image.fromarray(big).resize(
        (w, h), Image.Resampling.LANCZOS))
    gw, gh = w // 16 * 16, h // 16 * 16
    left, top = (w - gw) // 2, (h - gh) // 2
    assert (got.width, got.height) == (gw, gh) == (1280, 800)
    assert np.array_equal(got.pixels[..., :3],
                          small[top:top + gh, left:left + gw])
    # Under one MiB a reference keeps its pixels and loses its edges only.
    src = _picture(600, 400)
    got = edit.prepare_multi_reference(src)
    assert (got.width, got.height) == (592, 400)
    assert np.array_equal(got.pixels[..., :3], src[:, 4:596])
    refs = edit.prepare([big, src])
    assert refs.crop is None and len(refs) == 2
    assert edit.default_size(refs) == (1280, 800)
    assert edit.default_size(()) is None


def test_a_reference_is_upright_and_its_alpha_dropped_not_composited():
    from PIL import Image

    src = _picture(40, 24, channels=4)
    src[..., 3] = 0
    ref = edit.prepare_multi_reference(src)
    assert (ref.pixels[..., 3] == 255).all()
    assert np.array_equal(ref.pixels[..., :3], src[4:20, 4:36, :3])
    # The EXIF orientation applies before the rule: 6 turns it a quarter.
    im = Image.fromarray(_picture(48, 32))
    exif = im.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    with Image.open(io.BytesIO(buf.getvalue())) as jpg:
        jpg.load()
        turned = edit.prepare_multi_reference(jpg)
    assert (turned.width, turned.height) == (32, 48)
    # A prepared reference prepares again to itself by its own rule.
    again = edit.prepare([ref, ref])
    assert np.array_equal(again[0].pixels, ref.pixels)
    one, box = edit.prepare_reference(_picture(600, 400))
    again = edit.prepare([one])
    assert again.crop is None and np.array_equal(again[0].pixels, one.pixels)


def test_the_rules_refuse_what_they_cannot_prepare():
    ok = _picture(64, 64)
    with pytest.raises(edit.BadReference, match="no 16 x 16 pixel tile") as e:
        edit.prepare([ok, _picture(10, 40)])
    assert e.value.index == 1
    with pytest.raises(ValueError, match="at most 10 references"):
        edit.prepare([ok] * 11)
    with pytest.raises(edit.BadReference, match="uint8") as e:
        edit.prepare([np.zeros((8, 8, 3), np.float32)])
    assert e.value.index == 0


# A pipeline: the toy DiT that reads the tokens of its references, the
# toy encoder without a vision tower, and a VAE of RGB that counts.

class _RefDit(_Dit):
    def __init__(self):
        super().__init__()
        self.seen = []

    def references(self, latents):
        return [np.array(z) for z in latents]

    def __call__(self, latents, sigma, cap_feats, refs=None):
        v = super().__call__(latents, sigma, cap_feats)
        if refs is not None:
            self.seen.append(refs)
            v = v + sum(float(r.mean()) * (i + 1) for i, r in enumerate(refs))
        return v


class _Vae:
    """Latents (4, h / 16, w / 16) that are a function of the pixels."""
    dtype = mx.float32
    lockstep = False
    config = SimpleNamespace(scale=16, latent_channels=4, in_channels=3)

    def __init__(self):
        self.calls = []

    def encode(self, x):
        self.calls.append(tuple(x.shape))
        b, h, w, c = x.shape
        assert c == 3
        pooled = x.reshape(b, h // 16, 16, w // 16, 16, c).mean(axis=(2, 4))
        mean = mx.concatenate([pooled, pooled[..., :1]], axis=-1)
        return mean, mx.zeros_like(mean)

    def to_model_space(self, mean):
        return mean.transpose(0, 3, 1, 2)


@pytest.fixture
def pipe(monkeypatch):
    from gmlx.image.vae import tiling

    monkeypatch.setattr(tiling, "_plan", lambda *a, **k: False)
    fam = dataclasses.replace(get_family("z_image"),
                              edit=get_family("flux3_klein").edit)
    return api.ImagePipeline(dit=_RefDit(), family=fam, variant="turbo",
                             encoder=_Encoder(), vae=_Vae())


def test_a_text_only_edit_encodes_the_prompt_alone(pipe):
    a, b = _picture(600, 400, 1), _picture(400, 400, 2)
    text = api.encode_text(pipe, "ab")
    cond = api.encode_text(pipe, "ab", images=[a, b], ref_latents=False)
    # No vision tower: the encoder read the prompt once, and the
    # conditioning of the edit is that of the text with its references.
    assert pipe.encoder.encoded == ["ab"]
    assert cond.key == text.key and cond.features is text.features
    assert cond.image_slots == () and cond.pending
    assert [r.identity for r in cond.refs] == [
        r.identity for r in edit.prepare([a, b])]
    assert api.conditioning_cached(pipe, "ab", images=[a, b])
    assert pipe.vae.calls == []
    done = api.encode_text(pipe, "ab", images=[a, b])
    assert not done.pending and len(done.ref_latents) == 2
    assert pipe.vae.calls == [(1, 400, 592, 3), (1, 400, 400, 3)]
    assert api.encode_text(pipe, "ab") is text


def test_references_take_no_area_and_prepared_ones_pass_through(pipe):
    a = _picture(600, 400)
    refs = api.references(pipe, [a])
    assert refs.crop == (4, 0, 604, 400)
    assert api.references(pipe, list(refs)) == tuple(refs)
    with pytest.raises(ValueError, match="takes no reference area"):
        api.references(pipe, [a], area=512 * 512)
    with pytest.raises(ValueError, match="at most 10 reference images"):
        api.references(pipe, [a] * 11)
    # A mix prepares again by the rule of the count.
    mixed = api.references(pipe, [refs[0], _picture(400, 400)])
    assert [(r.width, r.height) for r in mixed] == [(608, 400), (400, 400)]


def test_a_conditioning_without_its_reference_latents_does_not_sample(pipe):
    cond = api.encode_text(pipe, "ab", images=[_picture(64, 64)],
                           ref_latents=False)
    with pytest.raises(ValueError, match="encode_references first"):
        api.sample(pipe, cond, height=32, width=32, seed=0, steps=1)
    assert pipe.dit.seen == []


def test_the_reference_latents_reach_the_dit_in_their_order(pipe):
    a, b = _picture(64, 48, 1), _picture(32, 32, 2)
    cond = api.encode_text(pipe, "ab", images=[a, b])
    x = api.sample(pipe, cond, height=32, width=32, seed=0, steps=2)
    lat = [np.array(z)[None] for z in cond.ref_latents]
    assert len(pipe.dit.seen) == 2
    for seen in pipe.dit.seen:
        assert [s.shape for s in seen] == [(1, 4, 3, 4), (1, 4, 2, 2)]
        assert all(np.array_equal(s, z) for s, z in zip(seen, lat))
    flipped = dataclasses.replace(cond, ref_latents=cond.ref_latents[::-1],
                                  ref_hashes=())
    y = api.sample(pipe, flipped, height=32, width=32, seed=0, steps=2)
    assert not np.array_equal(np.array(x), np.array(y))
    plain = Conditioning(cond.features, cond.key)
    z = api.sample(pipe, plain, height=32, width=32, seed=0, steps=2)
    assert not np.array_equal(np.array(x), np.array(z))
    assert isinstance(cond.refs[0], Reference)


def test_the_plan_counts_a_reference_given_twice_in_every_step(pipe):
    """The DiT runs one token block per reference, so one image given four
    times prices the step of four distinct images of its size, and the VAE
    encodes it once."""
    flux = dataclasses.replace(pipe, family=get_family("flux3_klein"),
                               vae=None, encoder=None)
    kw = dict(height=1024, width=1024, cfg=1.0, budget=100e9,
              overcommit=False)

    def plan(images):
        cond = api.encode_text(pipe, "ab", images=images, ref_latents=False)
        return cond, api.plan_request(flux, conds=cond, **kw)

    a = _picture(1024, 1024, 1)
    same, four = plan([a] * 4)
    assert len(same.refs) == 4 and len({r.sha256 for r in same.refs}) == 1
    assert api._ref_sizes([same]) == [(1024, 1024)]
    _, distinct = plan([a] + [_picture(1024, 1024, s) for s in (2, 3, 4)])
    _, one = plan([a])
    assert four.phases["denoise"] == distinct.phases["denoise"]
    assert four.phases["denoise"] > one.phases["denoise"]
    assert four.terms["init_peak"] == one.terms["init_peak"]
