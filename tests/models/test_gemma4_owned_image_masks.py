"""The owned Gemma 4 text model masks image blocks as transformers does.

transformers builds the masks under bidirectional image attention with
``create_masks_for_vision_model``: full-attention layers stay causal, and
sliding layers take AND(window, OR(causal, block)). llama.cpp's gemma4
graph does the same. mlx-vlm 0.7.3 ORs each block into both layer types
and past the sliding window.

The owned model builds the causal masks, then rebuilds the sliding mask
of a chunk that holds image or video tokens with the block overlay on the
chunk's own columns inside the window. On tiny random weights shared with
transformers it gives transformers' logits for gemma4 and gemma4_unified
text configs, and each chunk after a cached prefix gets the rows of the
one-pass masks. The stock half runs in a child that applies no loader
install (gemma4_probes.py). Runs on the CPU stream, whose float32 matmuls
match torch's closely.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from gmlx._vendor.mlx_vlm.models.gemma4.config import TextConfig as G4TextConfig
from gmlx._vendor.mlx_vlm.models.gemma4_unified.config import TextConfig as G4UTextConfig

from gmlx.models.gemma4 import language as owned

import gemma4_probes as probes

torch = pytest.importorskip("torch")
hg = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
hgc = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")
hu = pytest.importorskip(
    "transformers.models.gemma4_unified.modeling_gemma4_unified")
huc = pytest.importorskip(
    "transformers.models.gemma4_unified.configuration_gemma4_unified")

DRIVER = Path(__file__).with_name("gemma4_probes.py")
_N, _HIDDEN, _VOCAB = 40, 64, 128
_LAYERS = ["sliding_attention", "full_attention"] * 3
# (window, image blocks): a window wider than the block isolates the full
# layers; a narrower one also checks the window on the sliding overlay.
_CASES = {"text": (64, []), "wide": (64, [(10, 30)]),
          "narrow": (8, [(6, 20), (24, 34)])}
_TOL = 2e-4


@pytest.fixture(scope="module")
def stock() -> dict:
    r = subprocess.run([sys.executable, str(DRIVER), "--cpu"],
                       capture_output=True, text=True, env=dict(os.environ),
                       timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout)


def _text(window):
    return dict(vocab_size=_VOCAB, hidden_size=_HIDDEN, intermediate_size=128,
                num_hidden_layers=len(_LAYERS), num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, global_head_dim=16,
                num_global_key_value_heads=2, sliding_window=window,
                layer_types=_LAYERS, final_logit_softcapping=30.0,
                initializer_range=0.2, hidden_size_per_layer_input=0,
                vocab_size_per_layer_input=_VOCAB, num_kv_shared_layers=0,
                attention_k_eq_v=False, use_bidirectional_attention="vision")


@functools.lru_cache(maxsize=None)
def _models(model_type, window):
    """transformers' causal LM and the owned language model on the same
    weights."""
    if model_type == "gemma4":
        hf_mod, tc, cfg_cls = hg, hgc.Gemma4TextConfig(**_text(window)), \
            G4TextConfig
        hf_cls, text_type = hg.Gemma4ForCausalLM, "gemma4_text"
    else:
        hf_mod, tc, cfg_cls = hu, huc.Gemma4UnifiedTextConfig(
            **_text(window)), G4UTextConfig
        hf_cls, text_type = hu.Gemma4UnifiedForCausalLM, "gemma4_unified_text"
    tc._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = hf_cls(tc).eval().float()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
    cfg = cfg_cls.from_dict({**_text(window), "model_type": text_type,
                             "rope_parameters": tc.rope_parameters,
                             "use_double_wide_mlp": False})
    lm = owned.LanguageModel(cfg)
    lm.load_weights([(k, mx.array(v.detach().numpy()))
                     for k, v in hf.state_dict().items()
                     if not k.startswith("lm_head")], strict=True)
    mx.eval(lm.parameters())
    return hf_mod, tc, hf, lm


def _types(blocks, n=_N):
    t = np.zeros((1, n), dtype=np.int64)
    for a, b in blocks:
        t[0, a:b] = 1
    return t


def _embeds(n=_N, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((1, n, _HIDDEN)) * 0.5).astype(np.float32)


def _reference(hf_mod, tc, hf, embeds, types):
    """transformers' logits under its vision-model masks."""
    e, t = torch.from_numpy(embeds), torch.from_numpy(types)
    masks = hf_mod.create_masks_for_vision_model(
        config=tc, inputs_embeds=e, attention_mask=None, past_key_values=None,
        position_ids=torch.arange(e.shape[1])[None],
        block_sequence_ids=hf_mod.get_block_sequence_ids_for_mask(
            t, device=e.device))
    with torch.no_grad():
        return hf(inputs_embeds=e, attention_mask=masks,
                  use_cache=False).logits[0].numpy()


def _one_pass_sliding(types, window):
    """AND(window, OR(causal, block)) over one row of token types."""
    t = types[0]
    vision = (t == 1) | (t == 2)
    starts = vision & ~np.concatenate([[False], vision[:-1]])
    ids = np.where(vision, np.cumsum(starts) - 1, -1)
    q, k = np.arange(len(t))[:, None], np.arange(len(t))[None, :]
    block = (ids[:, None] != -1) & (ids[:, None] == ids[None, :])
    return (k > q - window) & ((k <= q) | block)


@pytest.mark.parametrize("case", ["text", "wide", "narrow"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_logits_match_transformers(model_type, case):
    window, blocks = _CASES[case]
    hf_mod, tc, hf, lm = _models(model_type, window)
    embeds, types = _embeds(), _types(blocks)
    ref = _reference(hf_mod, tc, hf, embeds, types)
    with mx.stream(mx.cpu):
        out = lm(None, inputs_embeds=mx.array(embeds), cache=lm.make_cache(),
                 mm_token_type_ids=mx.array(types.astype(np.int32)))
        got = np.array(out.logits[0].astype(mx.float32))
    assert np.abs(got - ref).max() <= _TOL


@pytest.mark.parametrize("start,end", [(6, 20), (24, 34)])
@pytest.mark.parametrize("whole_types", [False, True],
                         ids=["chunk_types", "prompt_types"])
def test_a_chunk_gets_the_one_pass_rows(start, end, whole_types):
    """A chunk holding one whole block, after a cached prefix, gets the
    rows of the one-pass masks for its queries, over the keys the caches
    hold (the sliding cache keeps the last window - 1 prefix keys). Token
    types for the whole prompt are sliced at the cache offset."""
    window, blocks = _CASES["narrow"]
    _hf_mod, _tc, _hf, lm = _models("gemma4_unified", window)
    tm = lm.model
    types_np = _types(blocks)
    types = mx.array(types_np.astype(np.int32))
    embeds = mx.array(_embeds())
    sliding = _one_pass_sliding(types_np, window)
    kept = min(window - 1, start)
    with mx.stream(mx.cpu):
        cache = lm.make_cache()
        mx.eval(lm(None, inputs_embeds=embeds[:, :start], cache=cache,
                   mm_token_type_ids=types[:, :start]).logits)
        cache = cache + [None] * (len(tm.layers) - len(cache))
        passed = types if whole_types else types[:, start:end]
        masks = tm._make_masks(embeds[:, start:end], cache, passed)
    for layer, mask in zip(tm.layers, masks):
        if layer.layer_type == "full_attention":
            assert isinstance(mask, str) and mask == "causal"
            continue
        want = sliding[start:end, start - kept:end]
        got = np.array(mx.broadcast_to(mask, (1, 1) + want.shape))[0, 0]
        assert (got == want).all()


def test_the_sliding_layers_share_one_mask():
    """One mask object per layer type: the batched route's pad relay keys
    on that identity."""
    window, blocks = _CASES["narrow"]
    _hf_mod, _tc, _hf, lm = _models("gemma4", window)
    tm = lm.model
    masks = tm._make_masks(mx.zeros((1, _N, _HIDDEN)),
                           [None] * len(tm.layers),
                           mx.array(_types(blocks).astype(np.int32)))
    for kind in ("sliding_attention", "full_attention"):
        ids = {id(m) for layer, m in zip(tm.layers, masks)
               if layer.layer_type == kind}
        assert len(ids) == 1, kind


def test_owned_masks_keep_the_block_to_sliding_layers_in_the_window():
    assert probes.image_mask_spread(owned) == {
        "full_attention": {"causal": True},
        "sliding_attention": {"past_window": False, "block_ahead": True},
    }


def test_stock_masks_put_the_block_in_every_layer_past_the_window(stock):
    assert stock["image_mask_spread"] == {
        "full_attention": {"causal": False},
        "sliding_attention": {"past_window": True, "block_ahead": True},
    }
