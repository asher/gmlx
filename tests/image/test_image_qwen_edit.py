"""Reference images of Qwen-Image 2.1: the resize rule, the template with
image slots, the token ids, the tap of a tiny random Qwen3-VL against the
forward of the copied mlx-vlm class, and the conditioning a pipeline makes
of a prompt with images. CPU tier."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from image_fixtures import load_json

from gmlx.image import api, memfit
from gmlx.image import text_encoder as te
from gmlx.image.conditioning import (Conditioning, Reference,
                                     conditioning_key, latents_hash)
from gmlx.image.families.qwen_image21 import edit
from gmlx.image.families.qwen_image21 import family as qf
from gmlx.image.registry import get_family

PAD, HIDDEN = 150, 64
MANIFEST = "oracle/check2_qwen_image21_edit.manifest.json"


def _picture(w: int, h: int, seed: int = 0, alpha: bool = False):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 4 if alpha else 3)).astype(np.uint8)
    return img


def _official_dimensions(area: int, ratio: float) -> tuple:
    # calculate_dimensions of the reference pipeline, as it is written.
    width = math.sqrt(area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


@pytest.mark.parametrize("area", [512 * 512, 1024 * 1024, 768 * 768])
@pytest.mark.parametrize("size", [(768, 768), (640, 480), (480, 640),
                                  (1920, 1080), (1000, 333), (17, 400),
                                  (4032, 3024), (720, 1280)])
def test_the_size_of_a_reference_is_the_reference_pipeline_rule(area, size):
    w, h = size
    assert edit.dimensions(area, w / h) == _official_dimensions(area, w / h)


def test_the_sizes_of_the_oracle_cases():
    m = load_json(MANIFEST)
    area = m["inputs"]["output_resolution"] ** 2
    sizes = {n: edit.dimensions(area, r["width"] / r["height"])
             for n, r in m["inputs"]["images"].items()}
    for case, lay in m["inputs"]["layout"].items():
        assert [list(sizes[n]) for n in lay["images"]] == lay["sizes"], case
        assert [w * h // 1024 for w, h in lay["sizes"]] == lay["image_rows"]
    assert edit.dimensions(1024 * 1024, 16 / 9) == (1376, 768)
    assert edit.dimensions(1024 * 1024, 1.0) == (1024, 1024)


def test_a_reference_is_rgba_at_the_family_size():
    from PIL import Image

    area = 512 * 512
    rgb = _picture(640, 480)
    ref = edit.prepare(rgb, area)
    assert isinstance(ref, Reference)
    assert (ref.width, ref.height) == (576, 448)
    assert ref.pixels.dtype == np.uint8 and ref.pixels.shape == (448, 576, 4)
    assert bool((ref.pixels[..., 3] == 255).all())
    assert not ref.pixels.flags.writeable
    # The resize is Pillow's Lanczos on the RGBA image.
    want = Image.fromarray(rgb).convert("RGBA").resize(
        (576, 448), Image.Resampling.LANCZOS)
    assert np.array_equal(ref.pixels, np.asarray(want))
    assert not np.array_equal(ref.pixels, np.asarray(
        Image.fromarray(rgb).convert("RGBA").resize(
            (576, 448), Image.Resampling.BICUBIC)))
    # A PIL image, a grey image and an image with alpha are taken.
    assert edit.prepare(Image.fromarray(rgb), area).sha256 == ref.sha256
    assert edit.prepare(rgb[..., 0], area).pixels.shape == (448, 576, 4)
    rgba = _picture(640, 480, alpha=True)
    with_alpha = edit.prepare(rgba, area)
    assert len(set(with_alpha.pixels[..., 3].ravel().tolist())) > 2
    # An image at its size is the image, and its hash names size and bytes.
    again = edit.prepare(ref.pixels, area)
    assert np.array_equal(again.pixels, ref.pixels)
    assert again.sha256 == ref.sha256 != with_alpha.sha256
    assert ref.identity == (ref.sha256, 448, 576)
    tall = Reference(np.ascontiguousarray(
        ref.pixels.reshape(576, 448, 4)))
    assert tall.sha256 != ref.sha256
    assert edit.slots(ref) == 18 * 14 and edit.slots(tall) == 18 * 14


def test_references_that_the_encoder_cannot_take_are_refused():
    with pytest.raises(ValueError, match="uint8 array"):
        edit.prepare(np.zeros((64, 64, 3), np.float32), 512 * 512)
    with pytest.raises(ValueError, match="too long for its area"):
        edit.prepare(_picture(4000, 10), 512 * 512)
    with pytest.raises(ValueError, match="under the 65536 pixels"):
        edit.prepare(_picture(640, 480), 128 * 128)
    with pytest.raises(ValueError, match=r"must be \(H, W, 4\) uint8"):
        Reference(np.zeros((8, 8, 3), np.uint8))


@pytest.mark.parametrize("w, h", [(640, 480), (480, 640), (1000, 600),
                                  (1200, 1000), (5000, 100), (300, 300)])
def test_a_small_reference_names_the_least_side_that_takes_it(w, h):
    """At 256 x 256 pixels a reference of some aspect ratios comes out
    under the encoder's least area. The refusal names the least side of
    a square area that takes it, and that side does."""
    side = edit.least_side(w / h)
    assert 256 <= side <= 320
    edit.prepare(_picture(w, h), side * side)
    if side > 256:
        with pytest.raises(edit.SmallReference) as e:
            edit.prepare(_picture(w, h), (side - 1) ** 2)
        assert e.value.side == side
        assert f"area of {side} x {side} pixels" in str(e.value)
    assert max(edit.least_side(r / 64) for r in range(64, 64 * 200, 7)) <= 320


def test_a_reference_sha256_given_must_be_its_own():
    px = _picture(32, 32, alpha=True)
    ref = Reference(px)
    assert Reference(px, sha256=ref.sha256).sha256 == ref.sha256
    with pytest.raises(ValueError, match="not that of the reference pixels"):
        Reference(px, sha256="0" * 64)


def test_the_encoder_sees_the_alpha_over_white():
    px = np.zeros((32, 32, 4), np.uint8)
    px[..., :3] = (10, 200, 90)
    px[:16, :, 3] = 255
    px[16:24, :, 3] = 128
    got = edit.over_white(Reference(px))
    assert got.shape == (32, 32, 3) and got.dtype == np.uint8
    assert got[0, 0].tolist() == [10, 200, 90]
    assert got[31, 0].tolist() == [255, 255, 255]
    a = 128 / 255
    want = np.array([10, 200, 90]) * a + 255 * (1 - a)
    assert np.abs(got[20, 0] - want).max() <= 1.0


def test_the_template_is_the_reference_render():
    """The rendered template of each oracle case, as the reference
    pipeline's own call rendered it."""
    m = load_json(MANIFEST)
    for case, row in m["inputs"]["cases"].items():
        assert edit.render(row["prompt"], len(row["images"])) == (
            m["inputs"]["rendered"][case]), case
    one = edit.render("x", 1)
    assert one.count("<|image_pad|>") == 1 and "<image1>" in one
    three = edit.render("x", 3)
    assert ("<image1><|vision_start|><|image_pad|><|vision_end|> <image2>"
            "<|vision_start|><|image_pad|><|vision_end|> <image3>"
            "<|vision_start|><|image_pad|><|vision_end|>x<|im_end|>") in three
    # An empty prompt is one space, as in the text template.
    assert edit.render("", 1) == edit.render(" ", 1)
    # The edit template opens with the system turn of the text template.
    assert one.startswith(qf.PROMPT.values["system_turn"])
    assert qf.PROMPT.repo == "huggingface/diffusers"
    assert len(qf.REFERENCES.revision) == 40


def test_the_template_id_names_the_template_and_the_dropped_rows(monkeypatch):
    base = edit.template_id()
    assert base != te.encoder_spec("qwen_image21").template_id
    for key, value in (("image_slot", "<|vision_start|><|image_pad|>"
                                      "<|vision_end|>"),
                       ("image_slot_join", ""), ("drop_tokens", 13)):
        with monkeypatch.context() as mp:
            mp.setitem(qf.PROMPT.values, key, value)
            assert edit.template_id() != base, key
    assert edit.template_id() == base


class _Tokenizer:
    """One id per character under 100, and one id per special token."""
    special = {"<|image_pad|>": PAD, "<|vision_start|>": 148,
               "<|vision_end|>": 149, "<|im_start|>": 146, "<|im_end|>": 147}

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        out, i = [], 0
        while i < len(text):
            for word, tid in self.special.items():
                if text.startswith(word, i):
                    out.append(tid)
                    i += len(word)
                    break
            else:
                out.append(ord(text[i]) % 100)
                i += 1
        return out


def _spec(layers: int = 4, drop: int = 3):
    return replace(te.encoder_spec("qwen_image21"), hidden=HIDDEN,
                   layers=layers, tap_layers=(layers,), drop=drop,
                   identity=())


def _tiny_qwen3_vl(seed: int = 0, layers: int = 4, **tower):
    from mlx.utils import tree_flatten

    from gmlx._vendor.mlx_vlm.models.qwen3_vl import config as vc
    from gmlx._vendor.mlx_vlm.models.qwen3_vl import qwen3_vl
    from gmlx._vendor.mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLImageProcessor)

    text = vc.TextConfig(
        model_type="qwen3_vl", num_hidden_layers=layers, hidden_size=HIDDEN,
        intermediate_size=96, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=160, num_key_value_heads=2, head_dim=16, rope_theta=1e6,
        max_position_embeddings=1024,
        rope_scaling={"type": "default", "mrope_section": [2, 3, 3]})
    vision = vc.VisionConfig(**{
        **dict(depth=3, hidden_size=32, intermediate_size=48,
               out_hidden_size=HIDDEN, num_heads=4, patch_size=16,
               spatial_merge_size=2, temporal_patch_size=2,
               num_position_embeddings=64, deepstack_visual_indexes=[0, 1]),
        **tower})
    model = qwen3_vl.Model(vc.ModelConfig(
        text_config=text, vision_config=vision, model_type="qwen3_vl",
        image_token_id=PAD, video_token_id=151, vocab_size=160))
    rng = np.random.default_rng(seed)
    model.load_weights([
        (k, mx.array((rng.standard_normal(v.shape) * 0.2
                      + (1.0 if "norm" in k and k.endswith("weight")
                         else 0.0)).astype(np.float32)))
        for k, v in tree_flatten(model.parameters())])
    model.eval()
    proc = SimpleNamespace(image_processor=Qwen3VLImageProcessor(
        patch_size=16, merge_size=2, min_pixels=65536, max_pixels=1 << 22))
    return model, proc


def _text_twin(vl):
    """The text class of the tiny encoder, holding its language weights,
    as the text loader builds a Qwen3-VL GGUF without its mmproj."""
    from mlx.utils import tree_flatten

    from gmlx._vendor.mlx_lm.models import qwen3

    model = qwen3.Model(qwen3.ModelArgs(
        model_type="qwen3", hidden_size=HIDDEN, num_hidden_layers=4,
        intermediate_size=96, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=160, num_key_value_heads=2, max_position_embeddings=1024,
        rope_theta=1e6, head_dim=16, tie_word_embeddings=False))
    weights = dict(tree_flatten(vl.language_model.parameters()))
    model.load_weights([(k, weights[k]) for k, _ in
                        tree_flatten(model.parameters())])
    model.eval()
    return model


def test_a_text_prompt_taps_the_same_with_and_without_the_tower():
    """The text class rotates with the multimodal class's rope for the
    tap, so a server that loads the tower conditions a text prompt as the
    CLI does without it."""
    vl, _ = _tiny_qwen3_vl()
    text = _text_twin(vl)
    own = text.model.layers[0].self_attn.rope
    prompt = "a red bicycle leaning on a wall in the rain"
    with_tower = te.EncoderHandle(vl, _Tokenizer(), _spec(), owned=True)
    without = te.EncoderHandle(text, _Tokenizer(), _spec(), owned=True)
    a, b = with_tower.encode(prompt), without.encode(prompt)
    assert a.ids == b.ids and len(a.ids) > 20
    assert mx.array_equal(a.hidden[0], b.hidden[0]).item()
    assert text.model.layers[0].self_attn.rope is own
    assert type(own) is nn.RoPE


@pytest.fixture(scope="module")
def handle():
    model, proc = _tiny_qwen3_vl()
    return te.EncoderHandle(model, _Tokenizer(), _spec(), owned=True,
                            source_key="tiny+mmproj", processor=proc)


@pytest.fixture(scope="module")
def refs():
    area = 320 * 320
    return [edit.prepare(_picture(300, 260, seed=1), area),
            edit.prepare(_picture(200, 320, seed=2, alpha=True), area)]


def test_each_slot_takes_one_pad_token_per_image_row(handle, refs):
    tok = handle.tokenizer
    assert [(r.width, r.height) for r in refs] == [(352, 288), (256, 416)]
    ids = edit.token_ids(handle, "ab", refs)
    assert ids.count(PAD) == 11 * 9 + 8 * 13
    plain = tok.encode(edit.render("ab", 2), add_special_tokens=False)
    at = [i for i, t in enumerate(plain) if t == PAD]
    assert ids[:at[0]] == plain[:at[0]]
    assert ids[at[0]:at[0] + 99] == [PAD] * 99
    gap = plain[at[0] + 1:at[1]]
    assert ids[at[0] + 99:at[0] + 99 + len(gap)] == gap
    assert ids[-len(plain[at[1] + 1:]):] == plain[at[1] + 1:]
    with pytest.raises(ValueError, match="a prompt cannot hold"):
        edit.token_ids(handle, "a <|image_pad|> b", refs)
    with pytest.raises(ValueError, match="needs a reference"):
        edit.token_ids(handle, "ab", [])


def _class_tap(handle, ids, refs, layer: int):
    """The features by the copied mlx-vlm class itself: its own join of
    the image features, its positions and its deepstack, for all the
    references in one call."""
    pixels, grids = [], []
    for r in refs:
        p, g = edit._pixels(handle, r)
        pixels.append(p)
        grids.append(g)
    x = mx.array(np.asarray(ids, np.int32).reshape(1, -1))
    h = handle.model.hidden_state_at_layer(
        x, layer, pixel_values=mx.concatenate(pixels, axis=0),
        image_grid_thw=mx.array(grids))
    return np.array(h[0, handle.spec.drop:])


def test_the_tap_with_images_is_the_forward_of_the_class(handle, refs):
    with mx.stream(mx.cpu):
        ids, feats, slots = edit.encode(handle, "paint it red", refs)
        want = _class_tap(handle, ids, refs, 4)
        assert feats.dtype == mx.float32
        got = np.array(feats)
        assert got.shape == (len(ids) - 3, HIDDEN) == want.shape
        assert list(slots) == [i - 3 for i, t in enumerate(ids) if t == PAD]
        assert len(slots) == 99 + 104 and slots[0] > 0
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
        # What a wrong encoder moves on this model: every one of these
        # is far from the right features.
        for name, other in (
                ("one deepstack block short",
                 np.array(edit.encode(handle, "paint it red", refs,
                                      deepstack=1)[1])),
                ("the references swapped",
                 np.array(edit.encode(handle, "paint it red",
                                      refs[::-1])[1])[:got.shape[0]]),
                ("the tap one layer early", _class_tap(handle, ids, refs, 3)),
                ("the final norm applied", np.array(
                    handle.model.language_model.model.norm(feats)))):
            gap = np.linalg.norm(other - got) / np.linalg.norm(got)
            assert gap > 1e-2, (name, gap)
        # One reference alone, and the same prompt as a text prompt.
        _, one, s1 = edit.encode(handle, "paint it red", refs[:1])
        assert one.shape[0] == len(s1) + (got.shape[0] - len(slots)) - len(
            handle.tokenizer.encode(" <image2><|vision_start|>"
                                    "<|vision_end|>",
                                    add_special_tokens=False))
        text = handle.encode("paint it red")
        assert text.features.shape[0] < one.shape[0]
        assert text.features.dtype == mx.float32


def test_a_split_product_is_float32_in_this_process():
    """``exact_matmul`` holds float32 precision whatever MLX's TF32 switch
    reads in this process, at the magnitudes of the tower's residual
    stream, for ``a @ b`` and ``a @ b.T``."""
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((96, 160)) * 3000).astype(np.float32)
    b = rng.standard_normal((160, 72)).astype(np.float32) / 12
    want = a.astype(np.float64) @ b.astype(np.float64)
    for got in (edit.exact_matmul(mx.array(a), mx.array(b)),
                edit.exact_matmul(mx.array(a), mx.array(b.T.copy()),
                                  b_t=True)):
        assert got.dtype == mx.float32
        gap = np.linalg.norm(np.array(got, np.float64) - want) \
            / np.linalg.norm(want)
        assert gap < 2e-6, gap
    # The high part keeps 10 mantissa bits, and the parts sum to the input.
    high, low = edit._split(mx.array(a))
    assert not (np.array(high).view(np.uint32) & 0x1FFF).any()
    assert np.array_equal(np.array(high + low), a)
    # A matrix read from bfloat16 is its own high part: the product that
    # leaves it whole is the split product, bit for bit.
    b16 = mx.array(b).astype(mx.bfloat16).astype(mx.float32)
    assert edit.tf32_exact(b16).item() and not edit.tf32_exact(
        mx.array(b)).item()
    for t in (False, True):
        w = b16.T if t else b16
        assert mx.array_equal(
            edit.exact_matmul(mx.array(a), w, b_t=t, b_exact=True),
            edit.exact_matmul(mx.array(a), w, b_t=t)).item()


def _linears(module) -> dict:
    return {p: id(m) for p, m in module.named_modules()
            if isinstance(m, (nn.Linear, edit.ExactLinear))}


def _gap(a, b) -> float:
    a, b = np.array(a, np.float64), np.array(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("products, attention, where", [
    (True, True, "plain"), (True, False, "plain"), (False, True, "split"),
    (False, False, "cpu")])
def test_the_tower_computes_float32_where_it_runs(handle, refs, monkeypatch,
                                                  products, attention, where):
    """The tower runs on the default device as it is where float32
    products are exact. Where they are at TF32 its float32 linear layers
    split each product and its blocks are evaluated one at a time while it
    runs and only then, or, where the fused attention is not float32 at
    its head size, it runs on the CPU."""
    seen = []
    tower = handle.model.vision_tower
    leaves = _linears(tower)
    blocks = list(tower.blocks)
    assert leaves and blocks

    class Watched(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = tower

        def __call__(self, pixels, grid):
            kinds = {type(m) for m in tower.modules()
                     if isinstance(m, (nn.Linear, edit.ExactLinear))}
            kinds |= {type(b) for b in tower.blocks}
            seen.append((mx.default_stream(mx.default_device()), kinds))
            return tower(pixels, grid)

    monkeypatch.setattr(handle.model, "vision_tower", Watched())
    monkeypatch.setattr(edit, "f32_products_exact", lambda: products)
    monkeypatch.setattr(edit, "sdpa_exact", lambda d: d == 8 and attention)
    edit.encode(handle, "x", refs)
    stream = mx.default_stream(mx.cpu if where == "cpu"
                               else mx.default_device())
    kinds = ({edit.ExactLinear, edit.EvaluatedBlock} if where == "split"
             else {nn.Linear, type(blocks[0])})
    assert seen == [(stream, kinds)] * 2
    assert _linears(tower) == leaves
    assert all(a is b for a, b in zip(tower.blocks, blocks, strict=True))


@pytest.mark.parametrize("bf16", [False, True], ids=["float32", "bfloat16"])
def test_the_split_tower_is_the_float32_tower_on_this_device(refs, bf16):
    """At the head size of the Qwen3-VL tower, 72, the fused attention is
    float32 on this device, and the tower on the default device gives the
    outputs of the tower on the CPU to float32 rounding, with weights of
    full float32 or read from bfloat16, which are not split. On a device
    that multiplies float32 at TF32, the tower without the split does
    not."""
    from mlx.utils import tree_map

    model, proc = _tiny_qwen3_vl(hidden_size=144, num_heads=2,
                                 intermediate_size=96)
    tower = model.vision_tower
    if bf16:
        tower.update(tree_map(
            lambda a: a.astype(mx.bfloat16).astype(mx.float32)
            if a.dtype == mx.float32 else a, tower.parameters()))
    assert edit._head_dims(tower) == {72}
    assert edit.sdpa_exact(72)
    h = te.EncoderHandle(model, _Tokenizer(), _spec(), owned=True,
                         source_key="tiny72+mmproj", processor=proc)
    for ref in refs:
        pixels, grid = edit._pixels(h, ref)
        grid = mx.array([grid])
        with mx.stream(mx.cpu):
            f, d = tower(pixels, grid)
            want = [f, *d]
        with edit.float32_products(tower):
            f, d = tower(pixels, grid)
            got = [f, *d]
            split = [m for _, m in tower.named_modules()
                     if isinstance(m, edit.ExactLinear)]
        if not edit.f32_products_exact():
            assert split and {m.weight_exact for m in split} == {bf16}
        assert max(_gap(g, w) for g, w in zip(got, want)) < 1e-5
        if not edit.f32_products_exact():
            f, d = tower(pixels, grid)
            assert max(_gap(g, w) for g, w in zip([f, *d], want)) > 1e-4


def test_the_attention_check_reads_the_device(monkeypatch):
    monkeypatch.setattr(edit, "_SDPA_EXACT", {})
    with mx.stream(mx.cpu):
        assert edit.sdpa_exact(8) and edit.sdpa_exact(72)
        cpu = str(mx.default_device())
    assert set(edit._SDPA_EXACT) == {(cpu, 8), (cpu, 72)}
    with mx.stream(mx.cpu):
        monkeypatch.setattr(edit, "_F32_EXACT", {})
        assert edit.f32_products_exact() is True
        assert edit._F32_EXACT == {str(mx.default_device()): True}


def test_the_split_layers_go_back_when_the_tower_raises(handle, monkeypatch):
    tower = handle.model.vision_tower
    before = _linears(tower)
    blocks = list(tower.blocks)
    monkeypatch.setattr(edit, "f32_products_exact", lambda: False)
    monkeypatch.setattr(edit, "sdpa_exact", lambda d: True)
    with pytest.raises(RuntimeError, match="stop"):
        with edit.float32_products(tower):
            assert {type(m) for m in tower.modules() if isinstance(
                m, (nn.Linear, edit.ExactLinear))} == {edit.ExactLinear}
            assert all(w.block is b for w, b in zip(tower.blocks, blocks,
                                                    strict=True))
            raise RuntimeError("stop")
    assert _linears(tower) == before
    assert all(a is b for a, b in zip(tower.blocks, blocks, strict=True))
    assert {type(m) for p, m in tower.named_modules()
            if p in before} == {nn.Linear}


def test_an_encoder_without_its_tower_or_with_another_pad_is_refused(refs):
    from gmlx._vendor.mlx_lm.models import qwen3

    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=HIDDEN, num_hidden_layers=2,
        intermediate_size=96, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=160, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=1e6, head_dim=16, tie_word_embeddings=True)
    text = te.EncoderHandle(qwen3.Model(args), _Tokenizer(), _spec(2),
                            owned=True)
    assert not text.vision and text.vision_bytes() == 0
    with pytest.raises(ValueError, match="load the pipeline with the "
                                         "encoder's mmproj"):
        edit.encode(text, "x", refs)
    model, proc = _tiny_qwen3_vl()
    model.config.image_token_id = PAD - 1
    other = te.EncoderHandle(model, _Tokenizer(), _spec(), owned=True,
                             processor=proc)
    assert other.vision and other.vision_bytes() > 0
    with pytest.raises(ValueError, match="image pad token"):
        edit.encode(other, "x", refs)
    released = te.EncoderHandle(*_tiny_qwen3_vl()[:1], _Tokenizer(), _spec(),
                                owned=True)
    released.release()
    with pytest.raises(RuntimeError, match="was released"):
        edit.encode(released, "x", refs)


# A pipeline: the tiny encoder with its tower, and a VAE that counts.

class _Vae:
    """Latents (8, h / 16, w / 16) that are a function of the pixels."""
    dtype = mx.float32
    lockstep = False
    config = SimpleNamespace(scale=16, latent_channels=8, in_channels=4)

    def __init__(self):
        self.calls = []

    def encode(self, x):
        self.calls.append(tuple(x.shape))
        b, h, w, c = x.shape
        assert c == 4
        pooled = x.reshape(b, h // 16, 16, w // 16, 16, c).mean(axis=(2, 4))
        mean = mx.concatenate([pooled, pooled * 2], axis=-1)
        return mean, mx.zeros_like(mean)

    def to_model_space(self, mean):
        return mean.transpose(0, 3, 1, 2)


@pytest.fixture
def pipe(handle, monkeypatch):
    from gmlx.image.vae import tiling

    monkeypatch.setattr(tiling, "_plan", lambda *a, **k: False)
    dit = SimpleNamespace(cfg=SimpleNamespace(in_channels=8, patch=1),
                          act_dtype=mx.bfloat16)
    return api.ImagePipeline(dit=dit, family=get_family("qwen_image21"),
                             variant="base", encoder=handle, vae=_Vae())


def test_references_follow_the_family_rule_and_its_cap(pipe):
    images = [_picture(300, 260, seed=i) for i in range(3)]
    refs = api.references(pipe, images, area=320 * 320)
    assert [(r.width, r.height) for r in refs] == [(352, 288)] * 3
    assert api.references(pipe, refs) == refs
    full = api.references(pipe, images[:1])
    assert full[0].width * full[0].height == 1088 * 960
    cap = pipe.family.edit.max_refs
    assert cap == qf.REFERENCES.values["max_refs"] <= 16
    with pytest.raises(ValueError, match=f"takes at most {cap} reference "
                                         f"images, and the request has "
                                         f"{cap + 1}"):
        api.references(pipe, [images[0]] * (cap + 1))
    with pytest.raises(ValueError, match="outside 65536 to 1048576"):
        api.references(pipe, images, area=2048 * 2048)
    z = replace(pipe, family=get_family("z_image"))
    assert api.references(z, []) == ()
    with pytest.raises(ValueError, match="Z-Image takes no reference "
                                         "images"):
        api.references(z, images)
    with pytest.raises(ValueError, match="Z-Image takes no reference "
                                         "images"):
        api.encode_text(z, "x", images=images)


def test_a_prompt_with_images_is_a_conditioning_with_its_references(pipe):
    images = [_picture(300, 260, seed=1), _picture(200, 320, seed=2)]
    refs = api.references(pipe, images, area=320 * 320)
    cond = api.encode_text(pipe, "paint it red", images=refs,
                           ref_latents=False)
    assert cond.pending and cond.refs == refs and not cond.ref_latents
    assert len(cond.image_slots) == 99 + 104
    assert cond.features.shape == (len(edit.token_ids(
        pipe.encoder, "paint it red", refs)) - 3, HIDDEN)
    assert api.encode_text(pipe, "paint it red", images=refs,
                           ref_latents=False) is cond
    assert pipe.conditionings.bytes == cond.features.nbytes
    # The key holds the images in order, the prompt and the template.
    ids = edit.token_ids(pipe.encoder, "paint it red", refs)
    assert cond.key == conditioning_key(
        family="qwen_image21", encoder="tiny+mmproj",
        template=edit.template_id(), ids=ids, tap_layers=(4,),
        dtype=pipe.encoder.dtype_name, images=[r.identity for r in refs])
    keys = {cond.key}
    for prompt, order in (("paint it red", refs[::-1]),
                          ("paint it blue", refs),
                          ("paint it red", refs[:1])):
        other = api.encode_text(pipe, prompt, images=order,
                                ref_latents=False)
        assert other.key not in keys
        keys.add(other.key)
    # Other pixels at the same size are another key.
    changed = np.array(refs[0].pixels)
    changed[0, 0, 0] ^= 1
    other = api.encode_text(pipe, "paint it red",
                            images=(Reference(changed), refs[1]),
                            ref_latents=False)
    assert other.key not in keys
    # A step refuses a conditioning without its latents by name.
    with pytest.raises(ValueError, match="encode_references first"):
        with api.state_scope(pipe, cond):
            pass


def test_a_text_prompt_keeps_its_key_and_never_reads_the_tower(pipe,
                                                               monkeypatch):
    """The text path of a pipeline is what it was before reference
    images: the key is the sha256 of the same document, and the tower
    never runs."""
    def boom(*a, **k):
        raise AssertionError("a text prompt ran the vision tower")

    monkeypatch.setattr(pipe.encoder.model, "vision_tower", boom)
    cond = api.encode_text(pipe, "a red bicycle")
    enc = pipe.encoder
    ids = enc.token_ids("a red bicycle")
    doc = {"family": "qwen_image21", "encoder": "tiny+mmproj",
           "adapters": "", "template": enc.spec.template_id,
           "ids": [int(i) for i in ids], "tap_layers": [4],
           "dtype": enc.dtype_name}
    assert cond.key == hashlib.sha256(json.dumps(
        doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert not cond.image_slots and not cond.refs and not cond.pending
    assert api.encode_references(pipe, cond) is cond
    assert pipe.vae.calls == []
    (h,) = te.tap(enc.model, ids, layers=(4,), activation="float32")
    assert mx.array_equal(cond.features, h[0, 3:]).item()


def test_a_text_encoder_load_without_an_mmproj_never_loads_the_tower(
        monkeypatch, tmp_path):
    import gmlx.load.loader as loader
    import gmlx.load.vlm as vlm

    calls = []
    monkeypatch.setattr(loader, "load_model",
                        lambda path: calls.append(("text", path)) or (
                            "model", {"c": 1}, "tok"))
    monkeypatch.setattr(vlm, "load_vlm_model",
                        lambda path, mmproj, **k: calls.append(
                            ("vlm", path, mmproj, k)) or (
                            "model", {"c": 1}, "proc", "tok"))
    assert te._load_owned("enc.gguf")[:4] == ("model", {"c": 1}, "tok", None)
    assert calls == [("text", "enc.gguf")]
    calls.clear()
    assert te._load_owned("enc.gguf", "mm.gguf")[:4] == (
        "model", {"c": 1}, "tok", "proc")
    assert calls == [("vlm", "enc.gguf", "mm.gguf",
                      {"return_tokenizer": True})]
    # Z-Image's encoder takes no mmproj, by name.
    with pytest.raises(te.ImageModelError, match="takes no mmproj"):
        te.load_encoder("enc.gguf", te.Z_IMAGE, mmproj="mm.gguf")
    with pytest.raises(ValueError, match="pass text_encoder"):
        api.load("dit.gguf", mmproj="mm.gguf")


def test_reference_latents_come_from_the_vae_once_per_image(pipe):
    images = [_picture(300, 260, seed=1), _picture(200, 320, seed=2)]
    refs = api.references(pipe, images, area=320 * 320)
    pos = api.encode_text(pipe, "paint it red", images=refs,
                          ref_latents=False)
    neg = api.encode_text(pipe, "blurry", images=refs, ref_latents=False)
    done = api.encode_references(pipe, [pos, neg])
    assert [tuple(z.shape) for z in done[0].ref_latents] == [(8, 18, 22),
                                                            (8, 26, 16)]
    # One VAE pass per image, whatever the conditionings that hold it.
    assert pipe.vae.calls == [(1, 288, 352, 4), (1, 416, 256, 4)]
    for c, src in zip(done, (pos, neg)):
        assert not c.pending and c.key == src.key and c.refs == refs
        assert c.features is src.features
        assert c.ref_hashes == tuple(latents_hash(z) for z in c.ref_latents)
    assert done[0].ref_latents[0] is done[1].ref_latents[0]
    assert api.encode_references(pipe, done[0]) is done[0]
    # The VAE reads all four channels on the [-1, 1] scale.
    px = refs[1].pixels.astype(np.float32) / 255.0 * 2 - 1
    want = px.reshape(26, 16, 16, 16, 4).mean(axis=(1, 3))
    got = np.array(done[0].ref_latents[1])
    np.testing.assert_allclose(got[:4].transpose(1, 2, 0), want, atol=1e-6)
    # The default of encode_text does both stages.
    whole = api.encode_text(pipe, "paint it red", images=refs)
    assert whole.key == pos.key and whole.ref_hashes == done[0].ref_hashes
    assert len(pipe.vae.calls) == 2
    # The hash of latents names their shape and their float32 bits.
    z = done[0].ref_latents[0]
    assert latents_hash(z) == latents_hash(mx.array(np.array(z)))
    assert latents_hash(z) != latents_hash(z + 1e-3)
    assert latents_hash(z) != latents_hash(z.reshape(8, 22, 18))
    with pytest.raises(ValueError, match="1 hashes for 2 reference"):
        Conditioning(pos.features, "k", ref_latents=done[0].ref_latents,
                     ref_hashes=("a",))


def test_two_references_of_one_size_keep_their_own_latents_and_order(pipe):
    """Two different images of one resized size in one request, and the
    same two in the other order: each image has its own latents, from one
    VAE pass each, and the order is part of the conditioning."""
    images = [_picture(300, 260, seed=3), _picture(300, 260, seed=4)]
    a, b = api.references(pipe, images, area=320 * 320)
    assert (a.width, a.height) == (b.width, b.height) == (352, 288)
    assert a.sha256 != b.sha256
    ab = api.encode_text(pipe, "swap them", images=(a, b))
    ba = api.encode_text(pipe, "swap them", images=(b, a))
    assert pipe.vae.calls == [(1, 288, 352, 4)] * 2
    assert ab.key != ba.key and ab.refs == (a, b) and ba.refs == (b, a)
    for cond, order in ((ab, (a, b)), (ba, (b, a))):
        for z, ref in zip(cond.ref_latents, order):
            px = ref.pixels.astype(np.float32) / 255.0 * 2 - 1
            want = px.reshape(18, 16, 22, 16, 4).mean(axis=(1, 3))
            np.testing.assert_allclose(np.array(z)[:4].transpose(1, 2, 0),
                                       want, atol=1e-6)
    assert ab.ref_hashes == ba.ref_hashes[::-1]
    assert ab.ref_hashes[0] != ab.ref_hashes[1]
    # The image rows of each order come from the tower in that order.
    assert ab.image_slots == ba.image_slots
    assert not np.allclose(np.array(ab.features), np.array(ba.features))


def test_the_plan_prices_the_references_of_its_conditionings(pipe):
    """The prefix rows of a conditioning with references come from the
    sizes of its images before their latents exist, and are the same
    after. The VAE encodes of the references are the init phase."""
    images = [_picture(300, 260, seed=1), _picture(200, 320, seed=2)]
    refs = api.references(pipe, images, area=320 * 320)
    pos = api.encode_text(pipe, "paint it red", images=refs,
                          ref_latents=False)
    text = api.encode_text(pipe, "paint it red")
    assert api._prefix_rows(pipe, pos) == (
        pos.features.shape[0] - len(pos.image_slots) + 18 * 22 + 26 * 16)
    done = api.encode_references(pipe, pos)
    assert api._prefix_rows(pipe, done) == api._prefix_rows(pipe, pos)
    assert api._ref_sizes([pos, done, text]) == [(288, 352), (416, 256)]
    bare = replace(pipe, vae=None)
    kw = dict(height=512, width=512, cfg=1.0, budget=100e9, overcommit=False)
    plain = api.plan_request(bare, conds=text, **kw)
    plan = api.plan_request(bare, conds=pos, **kw)
    assert "init" not in plain.phases and "init" in plan.phases
    row = memfit.PEAKS["qwen_image21"]
    assert plan.terms["prefix_cache"] == row.prefix.cache_bytes(
        api._prefix_rows(pipe, pos), 2)
    assert plan.terms["init_peak"] == max(
        memfit.vae_peak("encode", h, w, family="qwen_image21", itemsize=2)
        for h, w in ((288, 352), (416, 256)))
    assert not plan.tiled_encode
    # A guided request with the same images holds two prefixes and encodes
    # each image once.
    neg = api.encode_text(pipe, "blurry", images=refs, ref_latents=False)
    both = api.plan_request(bare, conds=[pos, neg], **{**kw, "cfg": 4.0})
    assert both.terms["prefix_cache"] == sum(row.prefix.cache_bytes(
        api._prefix_rows(pipe, c), 2) for c in (pos, neg))
    assert both.terms["init_peak"] == plan.terms["init_peak"]


def test_the_init_phase_runs_a_reference_in_tiles_when_whole_does_not_fit():
    # The prefix rows of a text prompt, so that the init phase is the
    # widest one.
    kw = dict(family="qwen_image21", height=256, width=256, cfg=1.0,
              overcommit=False, prefix_rows=[36])
    price = dict(family="qwen_image21", itemsize=2)
    whole = memfit.vae_peak("encode", 1024, 1024, **price)
    cut = memfit.vae_peak("encode", 1024, 1024, tiled=True, **price)
    small = memfit.vae_peak("encode", 512, 512, **price)
    assert small < cut < whole
    sizes = [(512, 512), (1024, 1024)]
    roomy = memfit.plan_memory(0, 0, 0, ref_sizes=sizes, budget=100e9, **kw)
    assert not roomy.tiled_encode and roomy.terms["init_peak"] == whole
    held = roomy.phases["init"] - whole
    assert roomy.vae_pass("encode") == (False, 100e9 - held)
    assert roomy.phases["denoise"] < held + cut
    tight = memfit.plan_memory(0, 0, 0, ref_sizes=sizes,
                               budget=held + whole - 1, **kw)
    assert tight.fits and tight.tiled_encode and not tight.levers
    assert tight.terms["init_peak"] == cut
    # References of one tile have no tiles to run in: the request that
    # does not fit them whole is refused.
    low = memfit.plan_memory(0, 0, 0, ref_sizes=sizes[:1], budget=100e9,
                             **kw)
    assert low.terms["init_peak"] == small and not low.tiled_encode
    with pytest.raises(memfit.ImageMemoryError, match="does not fit"):
        memfit.plan_memory(0, 0, 0, ref_sizes=sizes[:1],
                           budget=low.phases["init"] - 1, **kw)
    # The encodes run one at a time: more references of a size cost no
    # more in this phase than one, beside their prefix rows.
    two = memfit.plan_memory(0, 0, 0, ref_sizes=[sizes[1]] * 2,
                             budget=100e9, **kw)
    assert two.terms["init_peak"] == whole


@pytest.mark.parametrize("side, plan_tiles, want", [
    (512, True, False),         # one tile runs whole
    (1024, False, False),
    (1024, True, True),
])
def test_a_reference_is_encoded_whole_or_in_tiles_as_the_plan_says(
        pipe, monkeypatch, side, plan_tiles, want):
    from gmlx.image.vae import tiling

    seen = []

    def encode(vae, x, *, tiled, mode, free):
        seen.append((tuple(x.shape), tiled, mode, free))
        return mx.zeros((1, 8, x.shape[1] // 16, x.shape[2] // 16))

    monkeypatch.setattr(tiling, "encode", encode)
    ref = Reference(np.zeros((side, side, 4), np.uint8))
    plan = SimpleNamespace(phases={"init": 1},
                           vae_pass=lambda op: (plan_tiles, 7.0))
    z = api._reference_latents(pipe, ref, plan)
    assert seen == [((1, side, side, 4), want, "mode", 7.0)]
    assert z.shape == (8, side // 16, side // 16)
    # Without a plan, or with a plan without a budget, the pass decides
    # by the free memory it reads.
    for other in (None, SimpleNamespace(phases={"init": 1},
                                        vae_pass=lambda op: (True, None)),
                  SimpleNamespace(phases={})):
        pipe.reference_latents.clear()
        seen.clear()
        api._reference_latents(pipe, ref, other)
        assert seen == [((1, side, side, 4), None, "mode", None)]
