"""A MiniMax-H3 prompt with keyframes: the ids, the vision spans and their
tags, and the tower's input, against ComfyUI's
(scripts/video_oracle/keyframes_h3.py). CPU-only, no weights; the ids on
the real tokenizer need the encoder GGUF under KQUANT_TEST_GGUF_DIR."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import host_box
import numpy as np
import pytest
from video_oracle_fixture import committed, committed_fixture

from gmlx.video import encoder
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import keyframes as kf

NAME = "keyframes_h3"
SPEC = hf.FAMILY.encoder()
CASES = committed(NAME)["results"]["cases"]


class _Pieces:
    """A tokenizer that knows the texts ComfyUI tokenized alone."""

    def __init__(self, pieces: dict):
        self.pieces = pieces
        self.asked: list = []

    def encode(self, text, add_special_tokens=False):
        self.asked.append(text)
        return list(self.pieces[text])


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_ids_spans_and_tags_are_comfyuis(case):
    """Each label and the prompt are tokenized alone, a label counts the
    images given (a last frame alone is <Picture 1>), a block holds its
    markers and one pad per 32 x 32 pixels, and each span covers a block
    with its markers, which is ComfyUI's tag run widened by one each
    side."""
    c = CASES[case]
    tok = _Pieces(c["pieces"])
    ids, spans = kf.keyframe_ids(SPEC, tok, c["prompt"],
                                 [tuple(s) for s in c["sizes"]])
    assert ids == c["ids"]
    assert kf.token_tags(len(ids), spans).tolist() == c["tags"]
    assert kf.spans_of(c["tags"]) == spans
    assert len(spans) == len(c["sizes"])
    for (start, stop), (w, h) in zip(spans, c["sizes"]):
        assert ids[start] == hf.VISION_START and ids[stop - 1] == \
            hf.VISION_END
        assert stop - start == (w // 32) * (h // 32) + 2
    assert sorted(set(tok.asked)) == sorted(c["pieces"])


def test_the_block_sizes_of_the_shipped_canvases():
    assert kf.keyframe_ids(SPEC, _Pieces({"<Picture 1>: ": [1] * 6}), "",
                           [(1344, 768)])[0].count(hf.IMAGE_PAD) == 1008
    ids, spans = kf.keyframe_ids(SPEC, _Pieces({"<Picture 1>: ": [1] * 6}),
                                 "", [(768, 768)])
    assert ids.count(hf.IMAGE_PAD) == 576 and len(ids) == 584
    with pytest.raises(ValueError, match="1350x768 pixels is not on the "
                                         "vision tower's 32-pixel grid"):
        kf.keyframe_ids(SPEC, _Pieces({}), "", [(1350, 768)])


def test_tags_that_are_not_zero_and_one_are_refused():
    with pytest.raises(ValueError, match="hold 0 and 1 alone"):
        kf.spans_of([1, 0, 2])
    assert kf.spans_of([0, 0, 1, 1, 0]) == ((0, 2), (4, 5))
    assert kf.spans_of([1, 1]) == ()


def _processor():
    from gmlx._vendor.mlx_vlm.models.qwen3_vl.processing_qwen3_vl import \
        Qwen3VLImageProcessor

    return SimpleNamespace(image_processor=Qwen3VLImageProcessor())


def test_the_tower_input_is_comfyuis():
    """process_qwen2vl_images at patch 16 and mean and std 0.5 on the
    fixture's 64x64 image, from the same uint8 pixels."""
    arrays, _ = committed_fixture(NAME)
    handle = SimpleNamespace(processor=_processor())
    pix, grid = kf.pixels(handle, arrays["image_64"])
    assert grid == arrays["grid_64"].reshape(-1).tolist() == [1, 4, 4]
    np.testing.assert_allclose(np.array(pix), arrays["pixels_64"], rtol=0,
                               atol=1e-6)


def test_a_1344x768_keyframe_keeps_its_grid():
    """The vendored default cap of 1003520 pixels would shrink it; the
    family's bounds keep it whole."""
    handle = SimpleNamespace(processor=_processor())
    image = np.zeros((768, 1344, 3), np.uint8)
    _, grid = kf.pixels(handle, image)
    assert grid == [1, 48, 84]
    proc = handle.processor.image_processor
    assert proc(np.transpose(image, (2, 0, 1))[None])["image_grid_thw"][0] \
        .tolist() != [1, 48, 84]


def test_a_processor_that_resizes_or_normalizes_otherwise_is_refused():
    handle = SimpleNamespace(processor=_processor())
    proc = handle.processor.image_processor
    proc.image_mean = [0.48145466, 0.4578275, 0.40821073]
    with pytest.raises(ValueError, match="image_mean"):
        kf.pixels(handle, np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(ValueError, match="no image processor"):
        kf.pixels(SimpleNamespace(processor=None),
                  np.zeros((64, 64, 3), np.uint8))


@pytest.fixture(scope="module")
def encoder_tokenizer(gguf_index):
    for p in gguf_index.get(SPEC.arch, []):
        if Path(p).name.startswith("Qwen3-VL-32B-Instruct"):
            host_box.real_files()
            return encoder.header_tokenizer(hf.FAMILY, p)
    pytest.skip("no Qwen3-VL-32B encoder GGUF under KQUANT_TEST_GGUF_DIR")


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_ids_on_the_encoders_tokenizer_are_comfyuis(encoder_tokenizer,
                                                        case):
    c = CASES[case]
    ids, _ = kf.keyframe_ids(SPEC, encoder_tokenizer, c["prompt"],
                             [tuple(s) for s in c["sizes"]])
    assert ids == c["ids"]


def _tiny_h3_encoder(layers: int = 3):
    """A tiny random Qwen3-VL with its tower and H3's vision ids, built as
    tests/image/test_image_qwen_edit.py builds its own, and the H3 spec
    cut to its size."""
    import dataclasses

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from gmlx._vendor.mlx_vlm.models.qwen3_vl import config as vc
    from gmlx._vendor.mlx_vlm.models.qwen3_vl import qwen3_vl
    from gmlx.image import text_encoder as te

    hidden, vocab = 64, 151680
    text = vc.TextConfig(
        model_type="qwen3_vl", num_hidden_layers=layers, hidden_size=hidden,
        intermediate_size=96, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=vocab, num_key_value_heads=2, head_dim=16,
        rope_theta=1e6, max_position_embeddings=4096,
        rope_scaling={"type": "default", "mrope_section": [2, 3, 3]})
    vision = vc.VisionConfig(
        depth=2, hidden_size=32, intermediate_size=48, out_hidden_size=hidden,
        num_heads=4, patch_size=16, spatial_merge_size=2,
        temporal_patch_size=2, num_position_embeddings=64,
        deepstack_visual_indexes=[0, 1])
    model = qwen3_vl.Model(vc.ModelConfig(
        text_config=text, vision_config=vision, model_type="qwen3_vl",
        image_token_id=hf.IMAGE_PAD, video_token_id=151656,
        vision_start_token_id=hf.VISION_START, vocab_size=vocab))
    rng = np.random.default_rng(0)
    model.load_weights([
        (k, mx.array((rng.standard_normal(v.shape) * 0.2
                      + (1.0 if "norm" in k and k.endswith("weight")
                         else 0.0)).astype(np.float32)))
        for k, v in tree_flatten(model.parameters())])
    model.eval()
    spec = dataclasses.replace(SPEC, hidden=hidden, layers=layers,
                               tap_layers=(layers,), identity=(),
                               vision_identity=())
    pieces = {"<Picture 1>: ": [11, 12, 13], "<Picture 2>: ": [11, 12, 14],
              "a fox": [21, 22]}
    return te.EncoderHandle(model, _Pieces(pieces), spec, owned=True,
                            source_key="tiny+mmproj", processor=_processor())


def test_the_encode_with_keyframes_is_the_forward_of_the_class():
    """The tap of the shared vision_tap on H3's ids: each image through
    the tower alone, its rows in its block, its positions and deepstack
    as the copied mlx-vlm class gives them in one call."""
    import mlx.core as mx

    handle = _tiny_h3_encoder()
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 256, (64, 96, 3), dtype=np.uint8),
              rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)]
    with mx.stream(mx.cpu):
        feats, spans = kf.encode(handle, "a fox", images)
        ids, want_spans = kf.keyframe_ids(handle.spec, handle.tokenizer,
                                          "a fox", [(96, 64), (64, 64)])
        blocks = [kf.pixels(handle, im) for im in images]
        x = mx.array(np.asarray(ids, np.int32)[None])
        want = handle.model.hidden_state_at_layer(
            x, handle.spec.tap_layers[0],
            pixel_values=mx.concatenate([p for p, _ in blocks], axis=0),
            image_grid_thw=mx.array([g for _, g in blocks]))
    assert spans == want_spans == ((3, 3 + 6 + 2), (14, 14 + 4 + 2))
    assert feats.shape == (1, len(ids), 64) and feats.dtype == mx.float32
    np.testing.assert_allclose(np.array(feats[0]), np.array(want[0]),
                               rtol=1e-4, atol=1e-4)
    with mx.stream(mx.cpu):
        swapped, _ = kf.encode(handle, "a fox", images[::-1])
    gap = np.linalg.norm(np.array(swapped[0, -2:] - feats[0, -2:]))
    assert gap > 1e-2 * np.linalg.norm(np.array(feats[0, -2:]))


def test_keyframes_need_the_tower():
    handle = _tiny_h3_encoder()
    handle.model.vision_tower = None
    with pytest.raises(ValueError, match="load the encoder with the entry's "
                                         "mmproj"):
        kf.encode(handle, "a fox", [np.zeros((64, 64, 3), np.uint8)])
