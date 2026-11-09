"""The Gemma 4 audio tower's local attention mask.

A query frame sees the 11 frames before it and itself. transformers
(sliding_window_mask_function with context_left - 1) and llama.cpp's
gemma4a mask agree on this. The mask here is a port of llama.cpp's loop
over one block.
"""

from __future__ import annotations

import numpy as np

from gmlx._vendor.mlx_vlm.models.gemma4.audio import AudioEncoder
from gmlx._vendor.mlx_vlm.models.gemma4.config import AudioConfig


def _llama_cpp_block_mask(chunk: int, past: int) -> np.ndarray:
    """tools/mtmd/clip.cpp, PROJECTOR_TYPE_GEMMA4A, for a block far from
    either end of the clip."""
    context = chunk + past
    mask = np.zeros((chunk, context), dtype=bool)
    b = 5
    for q in range(chunk):
        gq = b * chunk + q
        for k in range(context):
            gk = b * chunk - past + k
            mask[q, k] = gk <= gq and (gq - gk) < past
    return mask


def test_audio_mask_matches_llama_cpp():
    config = AudioConfig(hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=2)
    got = np.array(AudioEncoder(config)._build_causal_valid_mask())
    want = _llama_cpp_block_mask(config.attention_chunk_size,
                                 config.attention_context_left - 1)
    np.testing.assert_array_equal(got, want)
    # Itself and the 11 frames before it.
    assert got.sum(axis=1).tolist() == [12] * config.attention_chunk_size
