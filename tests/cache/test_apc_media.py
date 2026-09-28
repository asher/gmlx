"""Config-declared media ids reach mlx-vlm's APC media guard."""

import importlib
from types import MethodType, SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm import apc as _apc
from mlx_vlm.generate import ar

import gmlx.load.vlm as gvlm
from gmlx.cache.apc_media import install_media_token_ids
from gmlx.load.config_synth import synthesize_config

from test_config_synth import (
    _GLM5NEXT_SHAPES,
    _MUSE_GLIMMER_SHAPES,
    _glm5next_meta,
    _muse_glimmer_meta,
)


def test_media_token_ids_folded_into_reader():
    assert install_media_token_ids()
    assert install_media_token_ids()  # idempotent
    apc = importlib.import_module("mlx_vlm.apc")
    cfg = SimpleNamespace(image_token_id=7, media_token_ids=[32, 33, 34])
    assert apc.multimodal_token_ids_from_config(cfg) == {7, 32, 33, 34}
    # dict configs (the stock reader ignores them) and configs without it
    assert apc.multimodal_token_ids_from_config(
        {"media_token_ids": [40]}) >= {40}
    assert 40 not in apc.multimodal_token_ids_from_config(
        SimpleNamespace(image_token_id=7))


def test_guard_moves_prefix_past_expanded_block():
    """A prefix that ends inside an expanded block moves up past its last
    media id (the stock rule), so no restored prefix cuts a block."""
    install_media_token_ids()
    from mlx_vlm.apc import media_safe_prefix_min

    vocab = 32
    block = [vocab + t for t in (1, 1, 0, 2, 2, 3, 1, 1, 4)]
    ids = [1, 2, 3] + block + [4, 5, 6]
    media = {vocab + t for t in range(5)}
    lo = media_safe_prefix_min(mx.array(ids), media)
    assert lo == len(ids) - 3, lo
    # text-only prompts keep a zero floor
    assert media_safe_prefix_min(mx.array([1, 2, 3]), media) == 0


def test_language_model_config_carries_media_ids():
    """The serve engine reads the config off the language model."""
    from gmlx.models.deepseek_v4.model import ModelArgs
    from gmlx.models.deepseek_v4.vlm_model import ModelConfig

    text = ModelArgs(model_type="deepseek_v4", vocab_size=32,
                     hidden_size=8, intermediate_size=8,
                     num_hidden_layers=1, num_attention_heads=1)
    cfg = ModelConfig(text_config=text, vocab_size=32)
    assert cfg.media_token_ids == [32, 33, 34, 35, 36]
    assert text.media_token_ids == [32, 33, 34, 35, 36]
    install_media_token_ids()
    from mlx_vlm.apc import multimodal_token_ids_from_config
    assert multimodal_token_ids_from_config(text) >= {32, 33, 34, 35, 36}


_HIDDEN = 64


def _gemma4_config(model_type: str, **text) -> dict:
    """A tiny gemma-4 container config; the ids fall to the defaults."""
    text_type = ("gemma4_unified_text" if model_type == "gemma4_unified"
                 else "gemma4_text")
    return {
        "model_type": model_type,
        "text_config": {
            "model_type": text_type, "hidden_size": _HIDDEN,
            "num_hidden_layers": 4, "intermediate_size": 128,
            "num_attention_heads": 4, "head_dim": 16, "global_head_dim": 16,
            "num_key_value_heads": 2, "num_global_key_value_heads": 2,
            "num_kv_shared_layers": 0, "vocab_size": 128,
            "vocab_size_per_layer_input": 128,
            "hidden_size_per_layer_input": 0, "sliding_window": 64,
            "sliding_window_pattern": 2, "attention_k_eq_v": False, **text},
        "vision_config": {},
    }


def _synth_config(model_type: str) -> dict:
    """The loader's config for a tiny glm5_next or muse_glimmer pair."""
    meta, shapes = {"glm5_next": (_glm5next_meta(), _GLM5NEXT_SHAPES),
                    "muse_glimmer": (_muse_glimmer_meta(),
                                     _MUSE_GLIMMER_SHAPES)}[model_type]
    hidden = synthesize_config(meta, shapes)["hidden_size"]
    clip = {
        "clip.vision.block_count": 1,
        "clip.vision.embedding_length": 32,
        "clip.vision.feed_forward_length": 64,
        "clip.vision.attention.head_count": 2,
        "clip.vision.patch_size": 14,
        "clip.vision.image_size": 448,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.projection_dim": hidden,
    }
    return gvlm.synthesize_vlm_config(model_type, meta, shapes, clip)


def _guard(lm):
    """A stand-in for the batch engine the server builds on ``lm``, with
    the stock media guard helpers."""
    gen = SimpleNamespace(model=lm)
    for name in ("_apc_media_token_ids", "_apc_safe_prefix_lookup_min",
                 "_apc_suffix_is_text_only", "_apc_prefix_has_media_tokens"):
        setattr(gen, name, MethodType(getattr(ar.BatchGenerator, name), gen))
    return gen


def _model(model_type: str, text: dict | None):
    config = (_gemma4_config(model_type, **(text or {}))
              if model_type.startswith("gemma4")
              else _synth_config(model_type))
    return gvlm.build_vlm_model(config)[0]


@pytest.mark.parametrize("model_type,text,mirrored", [
    ("gemma4", {"use_bidirectional_attention": "vision"}, True),
    ("gemma4_unified", {}, True),
    ("gemma4", {}, False),
    ("gemma4_unified", {"use_bidirectional_attention": None}, False),
    ("glm5_next", None, False),
    ("muse_glimmer", None, False),
])
def test_language_model_guard_sees_the_container_ids_with_bidirectional_attention(
        model_type, text, mirrored):
    """These archs name their image id on the container config only, and
    the server builds the batch engine on the language model. The loader
    mirrors the ids onto the text config when that config turns on
    bidirectional image attention, and the spec target the MTP path swaps
    in reads them too. Causal models keep a guard that sees no image id."""
    install_media_token_ids()
    model = _model(model_type, text)
    container = _apc.multimodal_token_ids_from_config(model.config)
    image = int(model.config.image_token_id)
    assert image in container
    lms = [model.language_model]
    if model_type.startswith("gemma4"):
        gvlm._swap_spec_language_model(model, model_type, log=lambda _m: None)
        assert model.language_model is not lms[0]
        lms.append(model.language_model)
    for lm in lms:
        seen = _guard(lm)._apc_media_token_ids()
        if mirrored:
            assert seen >= container
        else:
            assert image not in seen


class _Store:
    """An APC store holding the prompt's first ``cached`` columns, as block
    tier blocks or as an exact checkpoint."""

    def __init__(self, tier: str, cached: int):
        self.tier, self.cached = tier, cached

    def lookup_prefix(self, ids_list, extra_hash):
        return (["block"], self.cached) if self.tier == "block" else ([], 0)

    def lookup_exact_cache(self, ids_list, extra_hash, min_prefix_tokens):
        if self.tier == "exact" and self.cached >= min_prefix_tokens:
            return "checkpoint", self.cached
        return None, 0

    def lookup_prefix_disk_cache(self, ids_list, **_kwargs):
        return None, 0

    def release(self, blocks):
        pass


def _restored_len(lm, ids, store) -> int:
    """The prefix the stock lookup restores for ``ids`` from ``store``."""
    gen = _guard(lm)
    plan = _apc.apc_lookup_plan(
        store, ids, extra_hash=0, apc_mode="block",
        safe_lookup_min=gen._apc_safe_prefix_lookup_min(ids),
        suffix_is_text_only=lambda n: gen._apc_suffix_is_text_only(ids, n),
        prefix_has_media=lambda n: gen._apc_prefix_has_media_tokens(ids, n))
    return 0 if plan is None else plan["prefix_len"]


def _last_logits(lm, embeds, types, prefix: int, tier: str) -> mx.array:
    """Last-column logits with the first ``prefix`` columns restored. Block
    tier KV comes from a one-pass prefill of the whole prompt; an exact
    checkpoint is the KV of a prefill that stopped at ``prefix``."""
    def forward(cols, cache):
        out = lm(None, inputs_embeds=embeds[:, cols], cache=cache,
                 mm_token_type_ids=types[:, cols])
        return out.logits[:, -1, :].astype(mx.float32)

    cache = lm.make_cache()
    if prefix and tier == "block":
        forward(slice(None), cache)
        for c in cache:
            c.trim(embeds.shape[1] - prefix)
    elif prefix:
        forward(slice(0, prefix), cache)
    return forward(slice(prefix, None), cache)


@pytest.mark.parametrize("model_type,text", [
    ("gemma4", {"use_bidirectional_attention": "vision"}),
    ("gemma4_unified", {}),
])
@pytest.mark.parametrize("tier,cached", [("block", 16), ("exact", 32)])
def test_restored_image_prompt_matches_a_cold_prefill(
        model_type, text, tier, cached):
    """With bidirectional image attention, a one-pass prefill lets every
    image column see the whole image. A restore that ends inside the image
    leaves the rest of it to a suffix pass, where it sees only earlier
    columns, so the guard must decline it. A checkpoint past the image
    still restores."""
    install_media_token_ids()
    model, _ = gvlm.build_vlm_model(_gemma4_config(model_type, **text))
    lm = model.language_model
    assert lm.config.use_bidirectional_attention == "vision"
    mx.eval(lm.parameters())
    image = int(model.config.image_token_id)
    n, lo, hi = 40, 10, 26
    ids = [image if lo <= i < hi else 1 + i for i in range(n)]
    types = mx.array([[int(lo <= i < hi) for i in range(n)]])
    embeds = mx.random.normal((1, n, _HIDDEN), key=mx.random.key(0)) * 0.5

    cold = _last_logits(lm, embeds, types, 0, tier)
    prefix = _restored_len(lm, ids, _Store(tier, cached))
    served = _last_logits(lm, embeds, types, prefix, tier)
    diff = mx.max(mx.abs(served - cold)).item()
    assert diff <= 1e-4, (prefix, diff)
    assert prefix == (cached if cached >= hi else 0)


@pytest.mark.parametrize("model_type,text", [
    ("gemma4", {}), ("glm5_next", None), ("muse_glimmer", None)])
def test_causal_models_reuse_a_block_prefix_inside_an_image(model_type, text):
    """With causal attention an image column sees only earlier columns in
    a one-pass prefill too, so a block prefix that ends inside the image
    restores, and on gemma-4 the logits match a cold prefill."""
    install_media_token_ids()
    model = _model(model_type, text)
    lm = model.language_model
    image = int(model.config.image_token_id)
    n, lo, hi = 40, 10, 26
    ids = [image if lo <= i < hi else 1 + i for i in range(n)]
    assert _restored_len(lm, ids, _Store("block", 16)) == 16
    if model_type == "gemma4":
        mx.eval(lm.parameters())
        types = mx.array([[int(lo <= i < hi) for i in range(n)]])
        embeds = mx.random.normal((1, n, _HIDDEN), key=mx.random.key(0)) * 0.5
        cold = _last_logits(lm, embeds, types, 0, "block")
        served = _last_logits(lm, embeds, types, 16, "block")
        assert mx.max(mx.abs(served - cold)).item() <= 1e-4
