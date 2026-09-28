"""Bug fixes ported from mlx-vlm 0.7.3 onto the vendored qwen3.5 base
(gmlx.models.qwen35.language, mlx-vlm 0.6.15). Each test builds the
failing input on CPU and checks the vendored body against the 0.7.3 stock
body. No weights: get_rope_index reads only the config."""

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm.models.qwen3_5 import language as _StockL
from mlx_vlm.models.qwen3_5.config import ModelConfig, VisionConfig

from gmlx.models.qwen35.config import TextConfig
from gmlx.models.qwen35.language import LanguageModel

_VIDEO, _START, _END = 125, 126, 127


def _configs():
    text = TextConfig(
        model_type="qwen3_5", hidden_size=64, intermediate_size=128,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_hidden_layers=4, num_attention_heads=4,
        rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=2,
        max_position_embeddings=2048, tie_word_embeddings=True, head_dim=32,
        rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                         "rope_theta": 100000, "partial_rotary_factor": 0.25},
        full_attention_interval=4,
    )
    cfg = ModelConfig(
        text_config=text,
        vision_config=VisionConfig(
            depth=1, hidden_size=32, intermediate_size=64, out_hidden_size=64,
            num_heads=2, patch_size=14, num_position_embeddings=16,
            deepstack_visual_indexes=[]),
        model_type="qwen3_5", vocab_size=128, image_token_id=124,
        video_token_id=_VIDEO, vision_start_token_id=_START,
        vision_end_token_id=_END,
    )
    return text, cfg


def test_get_rope_index_splits_video_grids_per_temporal_group():
    """A 2-frame-group video arrives as one (2, 4, 4) grid row and two
    vision blocks of 4 merged tokens each. The 0.6.15 body indexes one grid
    row per block and read past the first block (a ValueError on this
    input); 0.7.3 splits the row into two (1, 4, 4) rows first."""
    text, cfg = _configs()
    lm = LanguageModel(text, cfg)
    block = [_START] + [_VIDEO] * 4 + [_END]
    ids = mx.array([[1] + block + [7, 8] + block + [5]])
    grid = mx.array([[2, 4, 4]])

    pos, deltas = lm.get_rope_index(ids, video_grid_thw=grid)
    split_pos, split_deltas = lm.get_rope_index(
        ids, video_grid_thw=mx.array([[1, 4, 4], [1, 4, 4]]))
    ref_pos, ref_deltas = _StockL.LanguageModel(text, cfg).get_rope_index(
        ids, video_grid_thw=grid)

    assert pos.tolist() == split_pos.tolist() == ref_pos.tolist()
    assert deltas.tolist() == split_deltas.tolist() == ref_deltas.tolist()
    # Tokens 10-13 are the second block; its t stream resumes after the
    # four text positions between the blocks (4-7).
    assert pos[0, 0].tolist()[10:14] == [8, 8, 8, 8]
