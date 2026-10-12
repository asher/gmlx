"""The rope scaling options reach the text tower of a VLM load: Qwen3-VL's
language model, and Qwen3-Omni's under its thinker. Tiny synthesized
pairs, no weights."""

from __future__ import annotations

import numpy as np
import pytest

import gmlx.load.vlm as gvlm
from gmlx._vendor.mlx_vlm.models.rope_utils import MRoPERotaryEmbedding
from gmlx.load import rope_scaling as rs
from gmlx.load.rope_scaling import RopeOverride
from test_config_synth import _QWEN3VL_SHAPES, _base_meta
from test_qwen3vl import _MM_META, _llm_meta

YARN4 = RopeOverride(rope_scaling="yarn", rope_scale=4.0)


def _omni_pair():
    m = _base_meta("qwen3vlmoe")
    m["qwen3vlmoe.rope.dimension_sections"] = [4, 2, 2, 0]
    m["qwen3vlmoe.expert_count"] = 4
    m["qwen3vlmoe.expert_used_count"] = 2
    m["qwen3vlmoe.expert_feed_forward_length"] = 32
    mm = {k: v for k, v in _MM_META.items()
          if k != "clip.vision.is_deepstack_layers"}
    mm.update({
        "clip.has_audio_encoder": True,
        "clip.audio.embedding_length": 32,
        "clip.audio.block_count": 1,
        "clip.audio.attention.head_count": 2,
        "clip.audio.feed_forward_length": 64,
        "clip.audio.num_mel_bins": 16,
        "clip.audio.projection_dim": 64,
    })
    return m, {}, mm


_PAIRS = {
    "qwen3_vl": (lambda: (_llm_meta(), _QWEN3VL_SHAPES, _MM_META), 5000000.0),
    "qwen3_omni_moe": (_omni_pair, 10000.0),
}


@pytest.mark.parametrize("kind", sorted(_PAIRS))
def test_text_tower_takes_the_options(kind):
    pair, base = _PAIRS[kind]
    meta, shapes, mm = pair()
    with rs.override_scope(YARN4):
        cfg = gvlm.synthesize_vlm_config(kind, meta, dict(shapes), mm)
    model, cfg = gvlm.build_vlm_model(cfg)
    assert gvlm.scale_text_rope(model, cfg) > 0
    if kind == "qwen3_omni_moe":
        tower, text = model.thinker.language_model, cfg["thinker_config"]["text_config"]
    else:
        tower, text = model.language_model, cfg["text_config"]
    s = rs.from_config(text)
    assert s.factor == 4.0 and s.mscale > 1.0
    ropes = [m.rotary_emb for _, m in tower.named_modules()
             if isinstance(getattr(m, "rotary_emb", None), MRoPERotaryEmbedding)]
    assert ropes
    for r in ropes:
        np.testing.assert_array_equal(np.asarray(r.inv_freq),
                                      rs.inv_freq(int(r.dim), base, s))
        assert r.attention_scaling == pytest.approx(s.mscale)
