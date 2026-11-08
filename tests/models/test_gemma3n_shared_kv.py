"""Gemma 3n's KV-shared layers attend over their source layer's keys at the
positions of the tokens in the pass.

The last layers of Gemma 3n read the keys and values of an earlier layer of
the same type. They must rotate their queries to the tokens' positions, as
the source layer does, and read the keys the source layer attended over.
A batch cache moves its offset array in place and stores more state than
the keys and values, so the batch engine exercises both. Tiny random
models on the CPU, for the mlx-lm and mlx-vlm copies of the model.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

_TEXT = dict(
    model_type="gemma3n_text", hidden_size=64, num_hidden_layers=4,
    intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, rms_norm_eps=1e-6, vocab_size=32,
    vocab_size_per_layer_input=32, num_kv_shared_layers=2,
    sliding_window=512, max_position_embeddings=1024, rope_theta=1e6,
    rope_local_base_freq=1e4, final_logit_softcapping=30.0,
    layer_types=["sliding_attention", "full_attention"] * 2,
    activation_sparsity_pattern=[0.95, 0, 0, 0],
    hidden_size_per_layer_input=16, altup_num_inputs=4,
    altup_coef_clip=120.0, altup_correct_scale=True, altup_active_idx=0,
    laurel_rank=8, rope_scaling=None)


def _mlx_lm():
    from gmlx._vendor.mlx_lm.models import gemma3n

    return gemma3n.LanguageModel(gemma3n.TextConfig.from_dict(_TEXT))


def _mlx_vlm():
    from gmlx._vendor.mlx_vlm.models.gemma3n.config import TextConfig
    from gmlx._vendor.mlx_vlm.models.gemma3n.language import LanguageModel

    return LanguageModel(TextConfig.from_dict(
        {**_TEXT, "intermediate_size": [128] * 4}))


_COPIES = {"lm": _mlx_lm, "vlm": _mlx_vlm}


def _model(copy):
    mx.random.seed(0)
    model = _COPIES[copy]()
    mx.eval(model.parameters())
    return model


def _logits(model, ids, cache):
    out = model(mx.array(ids), cache=cache)
    return getattr(out, "logits", out).astype(mx.float32)


_IDS = [3, 9, 4, 7, 2, 5, 11, 8]


@pytest.mark.parametrize("copy", sorted(_COPIES))
def test_decode_matches_one_prefill(copy):
    """Token-by-token decode after a short prefill gives the logits of one
    prefill over the whole prompt."""
    model = _model(copy)
    want = _logits(model, [_IDS], model.make_cache())[0]
    cache = model.make_cache()
    got = [_logits(model, [_IDS[:3]], cache)[0]]
    got += [_logits(model, [[t]], cache)[0] for t in _IDS[3:]]
    np.testing.assert_allclose(np.array(mx.concatenate(got)), np.array(want),
                               atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("copy", sorted(_COPIES))
def test_left_padded_batch_rows_match_each_row_alone(copy):
    """A left-padded batch cache prefills and decodes each row as the row
    does alone."""
    from gmlx._vendor.mlx_lm.generate import _make_cache

    model = _model(copy)
    rows = [_IDS, _IDS[2:6]]
    pad = [0, len(_IDS) - 4]
    cache = _make_cache(model, pad, None)
    batch = [[0] * p + r for p, r in zip(pad, rows)]
    got = [_logits(model, batch, cache)[:, -1]]
    got += [_logits(model, [[t], [t]], cache)[:, -1] for t in (6, 1)]
    for i, row in enumerate(rows):
        alone = model.make_cache()
        want = [_logits(model, [row], alone)[0, -1]]
        want += [_logits(model, [[t]], alone)[0, -1] for t in (6, 1)]
        np.testing.assert_allclose(
            np.array(mx.stack([g[i] for g in got])), np.array(mx.stack(want)),
            atol=2e-4, rtol=2e-4)
