"""The reference images of a MiniMax-H3-Image edit by ComfyUI's
MiniMaxH3ImageConditioning rules: the two copies of each reference, the
prompt that holds their vision blocks, the encode, the default size, the
key, and the memory rows a plan prices for them. CPU-only, no weights."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image import memfit, vision_tap
from gmlx.image.conditioning import Reference
from gmlx.image.families.h3_image import edit as he
from gmlx.image.families.h3_image import family as hi
from gmlx.image.h3 import pictures
from gmlx.image.h3.encoder import IMAGE_PAD, VISION_END, VISION_START, spec


def _image(w: int, h: int, channels: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, channels), dtype=np.uint8)


@pytest.mark.parametrize("size, vae, vision", [
    # A short edge under 1024 scales up to it for both copies.
    ((640, 480), (1376, 1024), (1376, 1024)),
    # A short edge of 1024 to 2048 keeps its size for the VAE, each side
    # rounded half to even as Python's round does: 62.5 grid steps is 62.
    ((3000, 2000), (3008, 1984), (1536, 1024)),
    # Past 2048 the VAE copy scales down to it.
    ((6000, 3000), (4096, 2048), (2048, 1024)),
    ((1024, 3072), (1024, 3072), (1024, 3072)),
])
def test_each_reference_has_comfyuis_two_copies(size, vae, vision):
    """The VAE copy's short edge goes into 1024 to 2048 and the vision
    copy's to 1024, each side on the 32 pixel grid (ComfyUI's
    _size_short_edge), each copy Lanczos from the image."""
    from PIL import Image

    img = _image(*size)
    ref = he.prepare([img])[0]
    assert (ref.width, ref.height) == vae
    assert (ref.vision.shape[1], ref.vision.shape[0]) == vision
    pil = Image.fromarray(img)
    want = np.asarray(pil.resize(vae, Image.Resampling.LANCZOS))
    assert np.array_equal(ref.pixels[..., :3], want)
    assert (ref.pixels[..., 3] == 255).all()
    want = np.asarray(pil.resize(vision, Image.Resampling.LANCZOS))
    assert np.array_equal(ref.vision, want)


def test_a_copy_at_its_images_size_is_the_images_bytes_and_alpha_is_dropped():
    """An image already at a copy's size reaches it untouched, and an
    alpha channel is dropped, not composited, as ComfyUI's image loader
    and _resize_to give it."""
    img = _image(1024, 1536, 4, seed=1)
    img[..., 3] = 0
    ref = he.prepare([img])[0]
    assert np.array_equal(ref.pixels[..., :3], img[..., :3])
    assert np.array_equal(ref.vision, img[..., :3])
    assert he.default_size([ref]) == (1024, 1536)


def test_references_outside_the_family_shapes_or_count_are_refused():
    with pytest.raises(he.BadReference, match="3100x1000 pixels is outside "
                       "the shapes MiniMax-H3-Image takes, 1:3 to 3:1") as e:
        he.prepare([_image(1024, 1024), _image(3100, 1000)])
    assert e.value.index == 1
    with pytest.raises(ValueError, match="at most 16 references, and the "
                                         "request has 17"):
        he.prepare([_image(64, 64)] * 17)
    assert hi.FAMILY.edit.max_refs == 16
    assert he.default_size(()) is None


class _Pieces:
    """A tokenizer that tokenizes each text it knows."""

    def __init__(self, pieces: dict):
        self.pieces = pieces

    def encode(self, text, add_special_tokens=False):
        return list(self.pieces[text])


def _handle(pieces: dict, model=None):
    from gmlx._vendor.mlx_vlm.models.qwen3_vl.processing_qwen3_vl import \
        Qwen3VLImageProcessor

    return SimpleNamespace(
        spec=spec(), tokenizer=_Pieces(pieces), model=model,
        processor=SimpleNamespace(image_processor=Qwen3VLImageProcessor()))


PIECES = {"Picture 1: ": [10, 11, 12], "Picture 2: ": [10, 11, 13],
          "a cat": [7, 8]}


def test_the_prompt_reads_each_vision_copy_after_its_label():
    """Each reference is its label, "Picture i: ", and a vision block of
    one pad per 32 x 32 pixels of its vision copy, before the prompt."""
    refs = he.prepare([_image(1024, 1024), _image(2048, 4096)])
    h = _handle(PIECES)
    ids = he.token_ids(h, "a cat", refs)
    blocks = [32 * 32, 32 * 64]
    want = []
    for i, n in enumerate(blocks, 1):
        want += PIECES[f"Picture {i}: "]
        want += [VISION_START] + [IMAGE_PAD] * n + [VISION_END]
    assert ids == want + [7, 8]
    assert ids == pictures.picture_ids(
        h.spec, h.tokenizer, "a cat", [(1024, 1024), (1024, 2048)],
        "Picture {i}: ")[0]
    # The rendered prompt's bytes bound every row but the pads past one a
    # block.
    rows = len(ids) - sum(n - 1 for n in blocks)
    from gmlx.image.text_encoder import text_rows_at_most

    assert text_rows_at_most(h.spec, he.render("a cat", 2)) >= rows


def test_the_encode_returns_the_features_and_the_vision_rows(monkeypatch):
    """The tower reads each vision copy at its size, and the image slots
    are each block's rows less its markers."""
    refs = he.prepare([_image(1024, 1024), _image(1024, 2048, seed=2)])
    seen = []

    def tap(handle, ids, blocks, pad):
        grids = [g for _, g in blocks]
        seen.append(grids)
        return mx.arange(len(ids) * 4, dtype=mx.float32).reshape(-1, 4), []

    monkeypatch.setattr(vision_tap, "tap", tap)
    h = _handle(PIECES, model=SimpleNamespace(vision_tower=object()))
    ids, feats, slots = he.encode(h, "a cat", refs)
    assert seen == [[[1, 64, 64], [1, 128, 64]]]
    assert feats.shape == (len(ids), 4) and feats.dtype == mx.float32
    starts = [i for i, t in enumerate(ids) if t == VISION_START]
    want = [r for s, n in zip(starts, (1024, 2048))
            for r in range(s + 1, s + 1 + n)]
    assert list(slots) == want
    assert all(ids[r] == IMAGE_PAD for r in slots)
    # A reference without its vision copy is refused by name.
    bare = Reference(refs[0].pixels)
    with pytest.raises(ValueError, match="needs its vision copy"):
        he.encode(h, "a cat", (bare,))


def test_the_vision_copy_joins_a_references_identity():
    ref = he.prepare([_image(1024, 1024)])[0]
    bare = Reference(ref.pixels)
    assert ref.identity[:3] == bare.identity
    assert ref.identity[3] == f"vision {ref.vision_sha256}"
    other = Reference(ref.pixels, vision=ref.vision[::-1].copy())
    assert other.identity != ref.identity
    with pytest.raises(ValueError, match=r"vision copy must be \(H, W, 3\)"):
        Reference(ref.pixels, vision=ref.pixels)


def test_the_template_id_names_the_label_and_the_pixel_bounds():
    assert len(he.template_id()) == 64
    assert he.BOUNDS == (3136, 12845056)
    assert he.LABEL == "Picture {i}: "


def test_the_vision_rows_a_plan_prices_bound_each_copy():
    """From a reference's VAE size alone, the plan prices at least the
    vision rows of its vision copy made from the original image, and at
    most five grid steps of the long side more."""
    row = memfit.PEAKS["h3_image"]
    slack = []
    for w in range(256, 8200, 37):
        for h in range(max(256, -(-w // 3)), min(8200, 3 * w) + 1, 53):
            vw, vh = he.short_edge(w, h, 1024, 2048)
            xw, xh = he.short_edge(w, h, 1024, 1024)
            actual = (xw // 32) * (xh // 32) + 2
            slack.append(memfit.vision_rows(row, [(vh, vw)]) - actual)
    assert len(slack) > 10_000
    assert 0 <= min(slack) and max(slack) <= 5 * 32, (min(slack),
                                                       max(slack))
    assert memfit.vision_rows(memfit.PEAKS["z_image"], [(1024, 1024)]) == 0


def test_a_plan_prices_an_edits_reference_and_vision_rows_and_its_tower():
    """An edit's step joins each reference's latent rows, a token of 32
    pixels, and its vision rows beside the caption; its encode runs the
    vision rows beside the text and the tower."""
    gib = 1 << 30

    def plan(**kw):
        return memfit.plan_memory(4 * gib, 4 * gib, gib // 4, gib,
                                  family="h3_image", height=1024,
                                  width=1024, caption_tokens=100,
                                  text_tokens=100, budget=96 * gib,
                                  overcommit=False, **kw)

    row = memfit.PEAKS["h3_image"]

    def step(rows):
        return row.step_peak(rows * row.width * 2, 0)

    plain = plan()
    assert plain.terms["step_peak"] == step(32 * 32 + 100)
    refs = [(1024, 1024), (2048, 1024)]
    edit = plan(ref_sizes=refs)
    vision = memfit.vision_rows(row, refs)
    assert edit.terms["step_peak"] == step(32 * 32 + 100 + 32 * 32
                                           + 64 * 32 + vision)
    given = plan(ref_sizes=refs, vision=1024 + 2048)
    assert given.terms["step_peak"] == step(32 * 32 + 100 + 32 * 32
                                            + 64 * 32 + 1024 + 2048)
    assert row.mmproj_fp32 and row.encode_tower_bytes > 4 * gib
    enc = edit.phases["encode"] - plain.phases["encode"]
    assert enc == (row.encode_tower_bytes + vision * row.encode_row_bytes
                   * memfit.ENCODE_LIVE)


def test_an_edits_conditioning_prices_its_vision_rows_apart_from_its_text():
    """The rows a known conditioning gives the plan: its vision rows, and
    the rest of its features as the caption."""
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    pipe = SimpleNamespace(family=hi.FAMILY, vae_scale=16,
                           dit=SimpleNamespace(cfg=SimpleNamespace(patch=2)))
    slots = tuple(range(5, 5 + 1024))
    cond = Conditioning(mx.zeros((1040, 8)), "k", image_slots=slots,
                        ref_latents=(mx.zeros((24, 64, 64)),))
    terms = api.plan_terms(pipe, cond, height=1024, width=1024)
    assert terms["vision"] == 1024 and terms["caption_tokens"] == 16
    assert terms["step_refs"] == [(1024, 1024)]
    plain = api.plan_terms(pipe, Conditioning(mx.zeros((16, 8)), "t"),
                           height=1024, width=1024)
    assert plain["vision"] == 0 and plain["caption_tokens"] == 16
