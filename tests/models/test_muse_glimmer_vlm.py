"""Muse Glimmer GGUF + mmproj: the loader builds the vendored VLM class.

mlx-vlm 0.6.15 ships its own ``mlx_vlm.models.muse_glimmer`` for the HF
checkpoint. Its ``_encode_image`` raises without the ``image_grid_thw`` that
only its own processor makes, and its language model has no speculative
hooks. When that class won the registration, every image request failed and
the DFlash drafter was dropped. These tests drive the loader's own resolution
(``build_vlm_model``, which calls mlx-vlm's ``get_model_and_args``) on a tiny
config. CPU-only, no GGUF.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

import gmlx.load.vlm as gvlm  # noqa: E402
import gmlx.models.muse_glimmer.tools as muse_tools  # noqa: E402
import gmlx.models.muse_glimmer.vlm_model as vm  # noqa: E402
from gmlx.load.mtp_target import (  # noqa: E402
    _MTP_TARGET_HOOKS_BY_TYPE,
    _spec_hook_key,
)

HIDDEN = 64
IMAGE_TOKEN = 60
# 56x84 pixels is a 4x6 patch grid: two 4x4 windows (the second one partial),
# a resampled position grid, and 2x3 soft tokens after the 2x2 merge.
IMAGE_HW = (56, 84)
SOFT_TOKENS = 6


def _config() -> dict:
    return {
        "model_type": "muse_glimmer",
        "text_config": {
            "model_type": "muse_glimmer",
            "hidden_size": HIDDEN,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 64,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 32,
        },
        "vision_config": {
            "model_type": "muse_glimmer",
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "image_size": 56,
            "patch_size": 14,
            "num_channels": 3,
            "projection_dim": HIDDEN,
            "adapter_hidden_size": 48,
            "spatial_merge_size": 2,
            "sparse_factor": 4,
            "num_position_embeddings": 16,
        },
        "vocab_size": 64,
        "image_token_id": IMAGE_TOKEN,
        "image_token_index": IMAGE_TOKEN,
    }


@pytest.fixture(scope="module")
def model():
    # The same two calls the GGUF VLM loader makes before build_vlm_model.
    vm.ensure_registered()
    muse_tools.ensure_registered()
    mx.random.seed(0)
    built, _ = gvlm.build_vlm_model(_config())
    mx.eval(built.parameters())
    return built


def test_loader_builds_the_vendored_class(model):
    assert type(model) is vm.Model
    assert type(model.language_model) is vm.LanguageModel


def test_language_model_carries_the_drafter_hooks(model):
    # mtp_load's VLM branch probes exactly this row on model.language_model
    # and drops the drafter when any hook is missing.
    hooks = _MTP_TARGET_HOOKS_BY_TYPE[_spec_hook_key("muse_glimmer")]
    lm = model.language_model
    assert [h for h in hooks if not hasattr(lm, h)] == []
    assert hasattr(lm, "set_dflash_capture")


def test_image_encodes_without_image_grid_thw(model):
    # The GGUF image processor emits pixel_values and image_sizes only.
    ids = mx.array([[1, 2] + [IMAGE_TOKEN] * SOFT_TOKENS + [3]])
    pixels = mx.random.normal((1, 3, *IMAGE_HW))
    text_only = model.get_input_embeddings(ids).inputs_embeds
    merged = model.get_input_embeddings(
        ids, pixels, image_sizes=[IMAGE_HW]).inputs_embeds
    assert merged.shape == (1, ids.shape[1], HIDDEN)
    image_rows = slice(2, 2 + SOFT_TOKENS)
    assert not mx.allclose(merged[:, image_rows], text_only[:, image_rows])
    assert mx.array_equal(merged[:, :2], text_only[:, :2])

    out = model(ids, pixels, cache=model.make_cache(), image_sizes=[IMAGE_HW])
    assert out.logits.shape == (1, ids.shape[1], 64)


# The GGUF processor: the chat template already starts with BOS, and the
# tokenizer adds one of its own, so the processor must not add a second.

BOS = "<|begin_of_text|>"


def _processor():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = [BOS, "<|patch|>", "<|image_start|>", "<|image_end|>"]
    vocab = {"[UNK]": 0, "hi": 1, "there": 2,
             **{t: 3 + i for i, t in enumerate(specials)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(specials)
    tok.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A", special_tokens=[(BOS, vocab[BOS])])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token=BOS, unk_token="[UNK]")
    mm_meta = {"clip.vision.patch_size": 14, "clip.vision.spatial_merge_size": 2}
    return gvlm._synthesize_muse_glimmer_processor(fast, mm_meta), vocab[BOS]


def _ids(feat):
    return [int(t) for t in feat["input_ids"][0].tolist()]


def test_processor_keeps_one_bos_on_a_text_prompt():
    proc, bos_id = _processor()
    assert _ids(proc(text=f"{BOS}hi there")) == [bos_id, 1, 2]
    # Without the template's BOS the tokenizer adds its own, once.
    assert _ids(proc(text="hi there")) == [bos_id, 1, 2]


def test_processor_keeps_one_bos_on_an_image_prompt():
    from PIL import Image

    proc, bos_id = _processor()
    image = Image.new("RGB", (IMAGE_HW[1], IMAGE_HW[0]))
    ids = _ids(proc(images=[image], text=f"{BOS}<|patch|> hi"))
    # BOS, <|image_start|>, one <|patch|> per soft token, <|image_end|>, hi
    assert ids == [bos_id, 5] + [4] * SOFT_TOKENS + [6, 1]
