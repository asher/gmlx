"""Config-declared media ids reach mlx-vlm's APC media guard."""

import importlib
from types import SimpleNamespace

import mlx.core as mx

from gmlx.cache.apc_media import install_media_token_ids


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
