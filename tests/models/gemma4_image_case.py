"""A tiny Gemma 4 text case under bidirectional image attention.

Shared by the owned image mask and chunking tests. The module imports no
torch: ``hf_models`` and ``reference`` build transformers' side only when
a test asks for it, and skip that test when torch or transformers is
missing. ``owned_lm`` builds the owned language model on random weights
without transformers.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np
import pytest
from gmlx._vendor.mlx_vlm.models.gemma4.config import TextConfig as G4TextConfig
from gmlx._vendor.mlx_vlm.models.gemma4_unified.config import TextConfig as G4UTextConfig

from gmlx.models.gemma4 import language as owned

N, HIDDEN, VOCAB = 40, 64, 128
LAYERS = ["sliding_attention", "full_attention"] * 3
# (window, image blocks): a window wider than the block isolates the full
# layers, and a narrower one also checks the window on the sliding overlay.
CASES = {"text": (64, []), "wide": (64, [(10, 30)]),
         "narrow": (8, [(6, 20), (24, 34)])}
TOL = 2e-4
_TEXT_TYPE = {"gemma4": "gemma4_text", "gemma4_unified": "gemma4_unified_text"}


def text(window) -> dict:
    return dict(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128,
                num_hidden_layers=len(LAYERS), num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, global_head_dim=16,
                num_global_key_value_heads=2, sliding_window=window,
                layer_types=LAYERS, final_logit_softcapping=30.0,
                initializer_range=0.2, hidden_size_per_layer_input=0,
                vocab_size_per_layer_input=VOCAB, num_kv_shared_layers=0,
                attention_k_eq_v=False, use_bidirectional_attention="vision")


def _config(model_type, window, rope_parameters=None):
    cfg = {**text(window), "model_type": _TEXT_TYPE[model_type],
           "use_double_wide_mlp": False}
    if rope_parameters is not None:
        cfg["rope_parameters"] = rope_parameters
    cls = G4TextConfig if model_type == "gemma4" else G4UTextConfig
    return cls.from_dict(cfg)


@functools.lru_cache(maxsize=None)
def owned_lm(model_type, window):
    """The owned language model on random weights."""
    mx.random.seed(0)
    lm = owned.LanguageModel(_config(model_type, window))
    mx.eval(lm.parameters())
    return lm


@functools.lru_cache(maxsize=None)
def hf_models(model_type, window):
    """transformers' causal LM and the owned language model on the same
    weights. Skips the calling test without torch or transformers."""
    torch = pytest.importorskip("torch")
    if model_type == "gemma4":
        hf_mod = pytest.importorskip(
            "transformers.models.gemma4.modeling_gemma4")
        cfg_mod = pytest.importorskip(
            "transformers.models.gemma4.configuration_gemma4")
        tc = cfg_mod.Gemma4TextConfig(**text(window))
        hf_cls = hf_mod.Gemma4ForCausalLM
    else:
        hf_mod = pytest.importorskip(
            "transformers.models.gemma4_unified.modeling_gemma4_unified")
        cfg_mod = pytest.importorskip(
            "transformers.models.gemma4_unified."
            "configuration_gemma4_unified")
        tc = cfg_mod.Gemma4UnifiedTextConfig(**text(window))
        hf_cls = hf_mod.Gemma4UnifiedForCausalLM
    tc._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = hf_cls(tc).eval().float()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
    lm = owned.LanguageModel(_config(model_type, window, tc.rope_parameters))
    lm.load_weights([(k, mx.array(v.detach().numpy()))
                     for k, v in hf.state_dict().items()
                     if not k.startswith("lm_head")], strict=True)
    mx.eval(lm.parameters())
    return hf_mod, tc, hf, lm


def types(blocks, n=N) -> np.ndarray:
    t = np.zeros((1, n), dtype=np.int64)
    for a, b in blocks:
        t[0, a:b] = 1
    return t


def embeds(n=N, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((1, n, HIDDEN)) * 0.5).astype(np.float32)


def reference(hf_mod, tc, hf, e, t) -> np.ndarray:
    """transformers' logits under its vision-model masks."""
    import torch

    e, t = torch.from_numpy(e), torch.from_numpy(t)
    masks = hf_mod.create_masks_for_vision_model(
        config=tc, inputs_embeds=e, attention_mask=None, past_key_values=None,
        position_ids=torch.arange(e.shape[1])[None],
        block_sequence_ids=hf_mod.get_block_sequence_ids_for_mask(
            t, device=e.device))
    with torch.no_grad():
        return hf(inputs_embeds=e, attention_mask=masks,
                  use_cache=False).logits[0].numpy()


def one_pass_sliding(t, window) -> np.ndarray:
    """AND(window, OR(causal, block)) over one row of token types."""
    t = t[0]
    vision = (t == 1) | (t == 2)
    starts = vision & ~np.concatenate([[False], vision[:-1]])
    ids = np.where(vision, np.cumsum(starts) - 1, -1)
    q, k = np.arange(len(t))[:, None], np.arange(len(t))[None, :]
    block = (ids[:, None] != -1) & (ids[:, None] == ids[None, :])
    return (k > q - window) & ((k <= q) | block)


def one_pass_logits(lm, e, t) -> np.ndarray:
    """The owned model's logits over one row in one pass, on the CPU."""
    with mx.stream(mx.cpu):
        out = lm(None, inputs_embeds=mx.array(e), cache=lm.make_cache(),
                 mm_token_type_ids=mx.array(t.astype(np.int32)))
        return np.array(out.logits[0].astype(mx.float32))
