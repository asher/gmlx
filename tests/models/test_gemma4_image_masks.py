"""Gemma-4 masks under bidirectional image attention follow transformers'
create_masks_for_vision_model: full-attention layers stay causal, and
sliding layers take AND(window, OR(causal, block)).

Tiny random-weight gemma4 ("vision") and gemma4_unified models carry the
same weights in transformers and in gmlx. The plain language model and the
MTP target match transformers on one row and on a left-padded batch, and
the stock, carried and owned mask builders give the same masks. Runs on
the CPU stream, whose fp32 matmuls match torch's closely.
"""

import functools

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.gemma4 import language as g4

import gmlx.load.vlm as gvlm
import gmlx.models.gemma4.owned as gemma4_owned
import gmlx.models.gemma4.sync as gemma4_sync
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.models.gemma4.image_masks import image_masks
from gmlx.load.mtp_target import _vlm_spec_language_model

from test_gemma4_image_audio import _served, serve_env  # noqa: F401

torch = pytest.importorskip("torch")
hg = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
hgc = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")
hu = pytest.importorskip(
    "transformers.models.gemma4_unified.modeling_gemma4_unified")
huc = pytest.importorskip(
    "transformers.models.gemma4_unified.configuration_gemma4_unified")

_N, _HIDDEN, _VOCAB = 40, 64, 128
_LAYERS = ["sliding_attention", "full_attention"] * 3
# (window, image blocks): a window wider than the block isolates the full
# layers; a narrower one also checks the window on the sliding overlay.
_CASES = {"wide": (64, [(10, 30)]), "narrow": (8, [(6, 20), (24, 34)])}
_TOL = 2e-4


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
    """transformers' causal LM and gmlx's plain language model and MTP
    target, on the same weights."""
    assert gemma4_sync.install_gemma4_nosync()
    mtp_prefill.install_full_prompt_mtp_prefill()
    if model_type == "gemma4":
        hf_mod, tc = hg, hgc.Gemma4TextConfig(**_text(window))
        hf_cls, text_type = hg.Gemma4ForCausalLM, "gemma4_text"
    else:
        hf_mod, tc = hu, huc.Gemma4UnifiedTextConfig(**_text(window))
        hf_cls, text_type = hu.Gemma4UnifiedForCausalLM, "gemma4_unified_text"
    tc._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = hf_cls(tc).eval().float()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
    model, _ = gvlm.build_vlm_model({
        "model_type": model_type, "vision_config": {},
        "text_config": {**_text(window), "model_type": text_type,
                        "rope_parameters": tc.rope_parameters,
                        "use_double_wide_mlp": False}})
    lm = model.language_model
    lm.load_weights([(k, mx.array(v.detach().numpy()))
                     for k, v in hf.state_dict().items()
                     if not k.startswith("lm_head")], strict=True)
    mx.eval(lm.parameters())
    _cls, build = _vlm_spec_language_model(model_type)
    target = build(model.config)
    target.load_weights(list(tree_flatten(lm.parameters())))
    mx.eval(target.parameters())
    return hf_mod, tc, hf, lm, target


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


def _gmlx(m, embeds, types):
    with mx.stream(mx.cpu):
        out = m(None, inputs_embeds=mx.array(embeds), cache=m.make_cache(),
                mm_token_type_ids=mx.array(types.astype(np.int32)))
        return np.array(out.logits[0].astype(mx.float32))


@pytest.mark.parametrize("path", ["plain", "mtp"])
@pytest.mark.parametrize("case", ["text", "wide", "narrow"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_logits_match_transformers(model_type, case, path):
    window, blocks = _CASES["wide" if case == "text" else case]
    hf_mod, tc, hf, lm, target = _models(model_type, window)
    embeds = _embeds()
    types = _types([] if case == "text" else blocks)
    ref = _reference(hf_mod, tc, hf, embeds, types)
    got = _gmlx(lm if path == "plain" else target, embeds, types)
    assert np.abs(got - ref).max() <= _TOL


@pytest.mark.parametrize("path", ["plain", "mtp"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_a_left_padded_batch_matches_transformers(model_type, path,
                                                   serve_env):  # noqa: F811
    """An image row beside a shorter text row, left-padded, served in one
    pass: each row matches transformers run on that row alone."""
    window, blocks = _CASES["narrow"]
    hf_mod, tc, hf, lm, target = _models(model_type, window)
    m = lm if path == "plain" else target
    short = 24
    image_e, text_e = _embeds(), _embeds(short, seed=8)
    image_t, text_t = _types(blocks), _types([], short)
    embeds = np.concatenate(
        [image_e, np.concatenate(
            [np.zeros((1, _N - short, _HIDDEN), np.float32), text_e], 1)], 0)
    types = np.concatenate([image_t, _types([], _N)], 0).astype(np.int32)
    rows = [list(range(1, _N + 1)), list(range(1, short + 1))]
    with mx.stream(mx.cpu):
        chunks, got = _served(m, rows, mx.array(embeds), mx.array(types),
                              path == "mtp")
        got = np.array(got)
    assert chunks == []
    for row, (e, t) in enumerate(((image_e, image_t), (text_e, text_t))):
        ref = _reference(hf_mod, tc, hf, e, t)[-1]
        assert np.abs(got[row] - ref).max() <= _TOL, row


@pytest.mark.parametrize("case", ["wide", "narrow"])
def test_every_builder_gives_the_same_masks(case):
    """The stock builder (GMLX_G4_NOSYNC=0), the carried nosync builder and
    the owned MTP builder give the same masks, and the full-attention mask
    is the causal one."""
    window, blocks = _CASES[case]
    _hf_mod, _tc, _hf, lm, _target = _models("gemma4_unified", window)
    tm = lm.model
    installed = g4.Gemma4TextModel._make_masks
    builders = [installed._gmlx_orig, installed._gmlx_body,
                gemma4_owned._owned_make_masks]
    h = mx.zeros((1, _N, _HIDDEN))
    types = mx.array(_types(blocks).astype(np.int32))
    got = []
    for build in builders:
        cache = lm.make_cache()
        cache = cache + [None] * (len(tm.layers) - len(cache))
        got.append(image_masks(tm, h, cache, types, build))
    full = g4.create_causal_mask(_N)
    for masks in got:
        for layer, mask in zip(tm.layers, masks):
            if layer.layer_type == "full_attention":
                assert (isinstance(mask, str) and mask == "causal") or (
                    mx.array_equal(mask, full).item())
    for masks in got[1:]:
        for a, b in zip(got[0], masks):
            assert type(a) is type(b)
            assert a == b if isinstance(a, str) else mx.array_equal(a, b).item()
