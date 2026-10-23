"""DFlash v1 on a tiny random Gemma 4 target (no GGUF, no weights).

Covers the packed-hidden capture on the owned Gemma 4 language model, on
both the return_hidden forward and the inner verify forward, the drafter's
embedding scale, and the engine contract: the rounds emit the target's own
greedy chain.
"""

import mlx.core as mx
import pytest

from gmlx._vendor.mlx_vlm.models.gemma4.config import TextConfig
from gmlx.models.gemma4.language import LanguageModel
from gmlx.spec.dflash_drafter import DFlashConfig, DFlashDrafter
from gmlx.spec.helpers import _mtp_verify_without_logits

# Six layers, the last two KV-shared and per-layer inputs on, so a capture
# sits in the shared region as Bolt-E4B's layer 35 does.
_TEXT = dict(
    model_type="gemma4_text", hidden_size=64, num_hidden_layers=6,
    intermediate_size=128, num_attention_heads=4, head_dim=16,
    global_head_dim=32, rms_norm_eps=1e-6, vocab_size=128,
    vocab_size_per_layer_input=128, num_key_value_heads=2,
    num_kv_shared_layers=2, hidden_size_per_layer_input=8,
    sliding_window=32, sliding_window_pattern=3, tie_word_embeddings=True,
    final_logit_softcapping=30.0,
)
CAPTURE = (1, 4)
H = _TEXT["hidden_size"]
BLOCK = 4
N_GEN = 16
PROMPT = mx.array([[3, 17, 42, 99, 7, 63, 5, 28, 11]])
# Tied embeddings collapse most random greedy chains onto one token. This
# draw yields 9 distinct tokens over N_GEN steps at a 0.09 minimum top-2
# margin.
SEED = 62


def _target(seed=SEED):
    mx.random.seed(seed)
    lm = LanguageModel(TextConfig(**_TEXT))
    mx.eval(lm.parameters())
    return lm


def _drafter(lm, *, scale=1.0):
    config = DFlashConfig(
        hidden_size=H, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        rms_norm_eps=1e-6, vocab_size=_TEXT["vocab_size"],
        max_position_embeddings=1024, rope_theta=10000.0, block_size=BLOCK,
        mask_token_id=4, target_layer_ids=list(CAPTURE),
        num_target_layers=_TEXT["num_hidden_layers"],
        layer_types=["sliding_attention"] * 2, sliding_window=512,
        final_logit_softcapping=_TEXT["final_logit_softcapping"],
        input_embedding_scale=scale)
    drafter = DFlashDrafter(config)
    mx.eval(drafter.parameters())
    drafter.reset(lm)
    return drafter


def _reference_captures(lm):
    sink: list = []
    lm(PROMPT, cache=lm.make_cache(), capture_layer_ids=list(CAPTURE),
       hidden_sink=sink)
    return sink


def test_unarmed_target_hands_out_the_bare_hidden():
    lm = _target()
    out = lm(PROMPT, cache=lm.make_cache(), return_hidden=True)
    assert out.hidden_states[-1].shape == (1, PROMPT.shape[1], H)
    h = lm.model(PROMPT, cache=lm.make_cache(), skip_final_norm=True)
    assert h.shape[-1] == H


def test_armed_forward_packs_final_then_captures_in_order():
    lm = _target()
    plain = lm(PROMPT, cache=lm.make_cache(), return_hidden=True)
    caps = _reference_captures(lm)
    lm.set_dflash_capture(CAPTURE)
    out = lm(PROMPT, cache=lm.make_cache(), return_hidden=True)
    packed = out.hidden_states[-1]
    assert packed.shape == (1, PROMPT.shape[1], H * (1 + len(CAPTURE)))
    assert mx.array_equal(out.logits, plain.logits).item()
    assert mx.array_equal(packed[..., :H], plain.hidden_states[-1]).item()
    for i, cap in enumerate(caps):
        assert mx.array_equal(packed[..., H * (i + 1):H * (i + 2)], cap).item()


def test_armed_verify_packs_and_the_hooks_read_the_trunk():
    """The engine verifies through ``lm.model(..., skip_final_norm=True)``,
    not through the language model's call."""
    lm = _target()
    block = mx.array([[5, 7, 9, 11]])
    cache = lm.make_cache()
    lm(PROMPT, cache=cache)
    want = lm.speculative_logits_from_hidden(
        _mtp_verify_without_logits(lm, block, cache).hidden)
    lm.set_dflash_capture(CAPTURE)
    cache = lm.make_cache()
    lm(PROMPT, cache=cache)
    packed = _mtp_verify_without_logits(lm, block, cache).hidden
    assert packed.shape[-1] == H * (1 + len(CAPTURE))
    assert mx.array_equal(lm.speculative_logits_from_hidden(packed), want).item()
    assert mx.array_equal(lm.speculative_argmax_from_hidden(packed),
                          mx.argmax(want, axis=-1)).item()
    assert lm.speculative_draft_hidden(packed) is packed


def test_drafter_embeds_with_its_own_scale_alone():
    """The converter folds Gemma's sqrt(hidden) into dflash.embedding_scale,
    so the target's embed_scale must not multiply it again."""
    lm = _target()
    drafter = _drafter(lm, scale=H ** 0.5)
    assert lm.model.embed_scale == pytest.approx(H ** 0.5)
    assert drafter.embed_scale == pytest.approx(H ** 0.5)
    assert _drafter(lm).embed_scale == 1.0


def _engine_reference(lm, n):
    cache = lm.make_cache()
    logits = lm(PROMPT, cache=cache).logits[0, -1]
    ref = []
    for _ in range(n):
        t = int(mx.argmax(logits).item())
        ref.append(t)
        logits = lm(mx.array([[t]]), cache=cache).logits[0, -1]
    return ref


def test_engine_rounds_emit_the_greedy_chain():
    import gmlx.spec.speculative as sp

    lm = _target()
    ref = _engine_reference(lm, N_GEN)
    assert len(set(ref)) >= 4
    drafter = _drafter(lm, scale=H ** 0.5)
    lm.set_dflash_capture(CAPTURE)
    cache = lm.make_cache()
    out = lm(PROMPT, cache=cache, return_hidden=True, return_shared_kv=True)
    b = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    sp._buffer_mtp_target_cache(cache, drafter, BLOCK)
    got = [b, *sp._owned_decode_rounds(
        lm, drafter, lm, cache, hidden=out.hidden_states[-1], b=b,
        shared_kv=out.shared_kv_states, seed_tokens=PROMPT, emitted=1,
        max_tokens=N_GEN, sampler=None, draft_block_size=BLOCK)]
    assert got[:N_GEN] == ref
