"""Hy4-flash MTP on a tiny random config: the rollback of a rejected draft,
greedy decode against plain decode, and the draft block's tensor set."""
from __future__ import annotations

import logging

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from gmlx.models.deepseek_v4.cache import MTP_UNDO_WIDTH
from gmlx.models.hy_v4_flash.mtp import (
    HyV4FlashMTPConfig,
    HyV4FlashMTPDrafter,
    HyV4FlashSpecLM,
)
from test_hy_v4_flash_model import _args

PREFILL_CHUNK = 8
N_GEN = 24
# A verify forward runs two tokens where plain decode runs one, and the two
# round differently. A step whose top two logits are closer than this can
# flip, which is a tie and not a lost token. The greedy tests use weights
# and a prompt (seeds 5 and 11) whose every step clears it.
GREEDY_TIE_TOL = 1e-2


@pytest.fixture(autouse=True)
def _on_cpu():
    # Exact fp32, as test_hy_v4_flash_model.py.
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _randomize(module, seed, scale):
    mx.random.seed(seed)
    params = dict(tree_flatten(module.parameters()))
    module.load_weights(
        [(k, mx.random.normal(v.shape, scale=scale).astype(v.dtype))
         for k, v in params.items()])
    mx.eval(module.parameters())


def _lm(seed=7, scale=0.5):
    lm = HyV4FlashSpecLM(_args(mtp_num_attention_heads=2))
    _randomize(lm, seed, scale)
    return lm


def _drafter(args, cls=HyV4FlashMTPDrafter):
    drafter = cls(HyV4FlashMTPConfig(text_config=args))
    _randomize(drafter, 3, 0.05)
    return drafter


def _target(lm):
    from gmlx.load.mtp_target import MTPTextTarget

    return MTPTextTarget(lm, {"model_type": "hy_v4_flash"})


# The window is 6 keys and a pool 4 tokens: these prompts end before and
# after the window fills, and leave every pool remainder at the rollback.
@pytest.mark.parametrize("accepted", [0, 2])
@pytest.mark.parametrize("prompt_len", [3, 9, 12, 14])
def test_a_rejected_draft_leaves_the_caches_of_the_accepted_prefix(
        prompt_len, accepted):
    lm = _lm()
    vocab = lm.args.vocab_size
    mx.random.seed(1)
    prompt = mx.random.randint(0, vocab, (1, prompt_len))
    block = mx.random.randint(0, vocab, (1, 4))
    nxt = mx.random.randint(0, vocab, (1, 2))
    kept = accepted + 1

    cache_a = lm.make_cache()
    out = lm(prompt, cache=cache_a, return_hidden=True)
    assert out.hidden_states[-1].shape == (1, prompt_len, lm.args.hidden_size)
    hidden, _, sink = lm.speculative_verify_hidden(block, cache_a)
    assert hidden.shape[1] == 4
    lm.rollback_speculative_cache(cache_a, sink, accepted, 4)
    # Two more tokens: the second reads an n-gram that spans the rollback.
    logits_a = lm(nxt, cache=cache_a).logits

    cache_b = lm.make_cache()
    lm(mx.concatenate([prompt, block[:, :kept]], axis=1), cache=cache_b)
    logits_b = lm(nxt, cache=cache_b).logits

    assert float(mx.abs(logits_a - logits_b).max()) < 2e-3
    slot_a = lm.model._history_slot(cache_a)
    slot_b = lm.model._history_slot(cache_b)
    assert mx.array_equal(slot_a[0], slot_b[0]).item()


def test_a_rollback_without_its_undo_log_fails():
    lm = _lm()
    cache = lm.make_cache()
    # 8 + 4 ends on a pool boundary: the trim crosses a finished pool,
    # which only the undo log can rewind.
    lm(mx.random.randint(0, 8, (1, 8)), cache=cache)
    _, _, sink = lm.speculative_verify_hidden(
        mx.random.randint(0, 8, (1, 4)), cache)
    for entry in cache:
        for leaf in getattr(entry, "caches", None) or ():
            if hasattr(leaf, "_undo"):
                leaf._undo = None
    with pytest.raises(RuntimeError, match="untrimmable"):
        lm.rollback_speculative_cache(cache, sink, 2, 4)


def _greedy_reference(lm, prompt, n_gen):
    """Plain greedy decode: the tokens and each step's top-2 margin."""
    cache = lm.make_cache()
    i = 0
    while i < prompt.shape[1]:
        logits = lm(prompt[:, i:i + PREFILL_CHUNK], cache=cache).logits
        i += PREFILL_CHUNK

    def step(row):
        srt = mx.sort(row)
        return int(mx.argmax(row).item()), float((srt[-1] - srt[-2]).item())

    tok, gap = step(logits[0, -1])
    out, gaps = [tok], [gap]
    for _ in range(n_gen - 1):
        logits = lm(mx.array([[tok]], dtype=mx.int32), cache=cache).logits
        tok, gap = step(logits[0, -1])
        out.append(tok)
        gaps.append(gap)
    return out, gaps


def _spec_decode(target, lm, drafter, prompt, n_gen, block=2):
    from gmlx.spec.speculative import stream_speculative

    drafter.reset(target)
    return list(stream_speculative(
        target, drafter, prompt, prompt_cache=lm.make_cache(),
        max_tokens=n_gen, sampler=None, draft_block_size=block,
        prefill_chunk=PREFILL_CHUNK))


class _OracleDrafter(HyV4FlashMTPDrafter):
    """Drafts a known continuation, so every draft is accepted, but for
    the script positions in ``_miss``, which it gets wrong."""

    def __init__(self, config):
        super().__init__(config)
        self._script = []
        self._miss = frozenset()
        self._pos = 0

    def reset(self, target_model, left_padding=None):
        out = super().reset(target_model, left_padding)
        self._pos = 1  # the engine samples the first token itself
        return out

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler,
                    token_dtype=mx.int32, greedy=False):
        tok = super().draft_block(
            last_bonus, hidden, cache, block_size, sampler, token_dtype,
            greedy)
        n = int(tok.shape[1])
        window = self._script[self._pos:self._pos + n]
        if len(window) == n:
            vocab = int(self.config.text_config.vocab_size)
            tok = mx.array(
                [[(int(t) + (self._pos + i in self._miss)) % vocab
                  for i, t in enumerate(window)]], dtype=token_dtype)
        return tok

    def accept_verified_tokens(self, verify_hidden, draft_tokens, accepted,
                               new_tokens, sampler, token_dtype=mx.int32,
                               greedy=False):
        self._pos += int(accepted) + 1
        super().accept_verified_tokens(
            verify_hidden, draft_tokens, accepted, new_tokens, sampler,
            token_dtype, greedy)


def _prompt(lm):
    # 41 tokens: past the window, and more keys than the indexer selects.
    mx.random.seed(11)
    return mx.random.randint(0, lm.args.vocab_size, (1, 41)).astype(mx.int32)


@pytest.mark.parametrize("block", [2, 4, MTP_UNDO_WIDTH])
def test_greedy_mtp_decode_gives_the_plain_tokens_through_rejections(
        caplog, block):
    lm = _lm(seed=5)
    drafter = _drafter(lm.args)
    prompt = _prompt(lm)

    ref, gaps = _greedy_reference(lm, prompt, N_GEN)
    with caplog.at_level(logging.WARNING, logger="gmlx.spec.speculative"):
        spec = _spec_decode(_target(lm), lm, drafter, prompt, N_GEN, block)

    assert min(gaps) > GREEDY_TIE_TOL
    assert spec == ref
    # A random block against a random target: its drafts are rejected.
    assert any(a == 0 for a in drafter.accept_lens)
    # The block was seeded a chunk at a time during prefill, and decode
    # took that cache over: its keys cover the prompt and what followed.
    assert not [r for r in caplog.records if "streamed seed" in r.message]
    assert drafter._cache[0][0].offset >= prompt.shape[1] - 1


@pytest.mark.parametrize("block", [2, 4, MTP_UNDO_WIDTH])
def test_greedy_mtp_decode_gives_the_plain_tokens_when_drafts_hold(block):
    lm = _lm(seed=5)
    prompt = _prompt(lm)
    ref, _ = _greedy_reference(lm, prompt, N_GEN)
    oracle = _drafter(lm.args, cls=_OracleDrafter)
    oracle._script = ref
    spec = _spec_decode(_target(lm), lm, oracle, prompt, N_GEN, block)

    assert spec == ref
    accepts = list(oracle.accept_lens)
    # The last round can be cut short by the token budget.
    assert sum(1 for a in accepts if a == block - 1) >= len(accepts) - 1


def test_greedy_mtp_decode_gives_the_plain_tokens_when_a_deep_block_holds_in_part():
    lm = _lm(seed=5)
    prompt = _prompt(lm)
    ref, _ = _greedy_reference(lm, prompt, N_GEN)
    oracle = _drafter(lm.args, cls=_OracleDrafter)
    oracle._script = ref
    oracle._miss = frozenset({2, 3, 8, 12, 17, 21})
    spec = _spec_decode(
        _target(lm), lm, oracle, prompt, N_GEN, MTP_UNDO_WIDTH)

    assert spec == ref
    # Rounds that keep none, some and all of their drafts.
    assert len(set(oracle.accept_lens)) >= 4


# 41 rows is more pools than the indexer selects. The four draft steps of
# a block of 6 finish a pool from every remainder.
@pytest.mark.parametrize("accepted", [0, 2, 5])
@pytest.mark.parametrize("prompt_len", [41, 42, 43, 44])
def test_the_accept_finds_the_block_s_cache_without_its_draft_steps(
        prompt_len, accepted):
    lm = _lm()
    target = _target(lm)
    vocab, width = lm.args.vocab_size, lm.args.hidden_size
    mx.random.seed(2)
    prompt = mx.random.randint(0, vocab, (1, prompt_len)).astype(mx.int32)
    states = mx.random.normal((1, prompt_len, width))
    drafts = mx.random.randint(0, vocab, (1, 5)).astype(mx.int32)
    verify = mx.random.normal((1, 6, width))

    def round_at(block):
        drafter = _drafter(lm.args)
        drafter.reset(target)
        drafter.prefill_from_target_hidden(
            prompt, states, 3, None, greedy=True)
        drafter.draft_block(
            3, states[:, -1:], None, block, None, greedy=True)
        drafter.accept_verified_tokens(
            verify, drafts, accepted, [5], None, greedy=True)
        keys, pool = drafter._cache[0].caches
        counts = keys.offset, pool._plen, pool.remainder
        # The next round's first draft and a step past it read the cache.
        nxt = drafter.draft_block(
            5, verify[:, -1:], None, 3, None, greedy=True)
        return *counts, nxt

    shallow, deep = round_at(2), round_at(MTP_UNDO_WIDTH)
    assert deep[:3] == shallow[:3]
    assert shallow[0] == prompt_len + accepted + 1
    assert mx.array_equal(deep[3], shallow[3]).item()


def test_a_verify_block_at_16_experts_a_token_keeps_the_decode_expert_route():
    """The released file routes 16 experts a token, so a verify of 4 rows
    is 64 (row, expert) pairs, where the expert kernels of other models
    start sorting by expert."""
    from gmlx.models.hy_v4_flash.model import HyV4FlashMoE

    moe = HyV4FlashMoE(_args(num_experts_per_tok=16, n_routed_experts=32))
    widest = 16 * MTP_UNDO_WIDTH
    assert moe.switch_mlp._kq_fused_pairs > widest
    assert moe.gate._fused_pairs > widest
    assert HyV4FlashMoE(_args()).switch_mlp._kq_fused_pairs == 64


def test_the_drafter_meets_the_engine_contract():
    from gmlx.load.mtp_target import (
        _MTP_TARGET_HOOKS_BY_TYPE,
        _mtp_target_classes,
    )
    from gmlx.spec.drafter_protocol import validate_drafter

    for hook in _MTP_TARGET_HOOKS_BY_TYPE["hy_v4_flash"]:
        assert callable(getattr(HyV4FlashSpecLM, hook, None)), hook
    assert _mtp_target_classes("hy_v4_flash")[0] is HyV4FlashSpecLM

    args = _args(mtp_num_attention_heads=2)
    drafter = _drafter(args)
    validate_drafter(drafter)
    with pytest.raises(NotImplementedError):
        drafter.make_cache(left_padding=[0])
    with pytest.raises(ValueError, match="block of 2 to"):
        HyV4FlashMTPDrafter(HyV4FlashMTPConfig(
            text_config=args, block_size=MTP_UNDO_WIDTH + 1))
    # A deeper ask drafts the deepest block a round can undo.
    from gmlx.spec.helpers import _resolve_block_total

    assert _resolve_block_total(drafter, 9) == MTP_UNDO_WIDTH
    # The block is a sparse layer at the head count the file gives it.
    attn = drafter.layers[0].self_attn
    assert attn.num_heads == 2 and attn.indexer is not None
    assert not attn.window


# The released file's block 42, as gguf lists it.
_BLOCK_TENSORS = (
    "attn_gate.weight", "attn_k_b.weight", "attn_kv_a_mqa.weight",
    "attn_kv_a_norm.weight", "attn_norm.weight", "attn_output.weight",
    "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
    "attn_sinks.weight", "attn_v_b.weight", "exp_probs_b.bias",
    "ffn_down_exps.weight", "ffn_down_shexp.weight", "ffn_gate_exps.weight",
    "ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_norm.weight",
    "ffn_routed_down.weight", "ffn_routed_up.weight", "ffn_up_exps.weight",
    "ffn_up_shexp.weight", "indexer.attn_k.weight",
    "indexer.attn_q_b.weight", "indexer.k_norm.weight",
    "indexer.proj.weight", "indexer_compressor_ape.weight",
    "indexer_compressor_gate.weight", "indexer_compressor_norm.weight",
    "indexer_compressor_value.weight", "nextn.eh_proj.weight",
    "nextn.enorm.weight", "nextn.hnorm.weight",
    "nextn.shared_head_norm.weight", "post_attention_norm.weight",
    "post_ffw_norm.weight",
)


def test_the_file_s_draft_block_fills_the_drafter_and_nothing_else():
    from gmlx.load.loader import _FP32_KEEP_BY_MODEL_TYPE
    from gmlx.load.wire import remap_mtp_arrays

    args = _args(mtp_num_attention_heads=2)
    drafter = _drafter(args)
    params = {k for k, _ in tree_flatten(drafter.parameters())}

    blk = args.num_hidden_layers
    arrays = {f"blk.{blk}.{t}": mx.zeros((4, 4)) for t in _BLOCK_TENSORS}
    arrays["blk.0.attn_q_a.weight"] = mx.zeros((4, 4))  # trunk: ignored

    weights, _, stats = remap_mtp_arrays(
        arrays, {}, "hy_v4_flash", first_mtp_block=blk, num_mtp_layers=1)
    assert stats["mapped"] == len(_BLOCK_TENSORS) and stats["skipped"] == 0
    assert set(weights) == params, (
        f"only in the remap: {sorted(set(weights) - params)}; only in the "
        f"drafter: {sorted(params - set(weights))}")

    # The arrays the trunk keeps in fp32 stay fp32 on the block too.
    pins = _FP32_KEEP_BY_MODEL_TYPE["hy_v4_flash"]
    for name in ("layers.0.self_attn.sinks", "layers.0.mlp.gate.weight",
                 "layers.0.mlp.gate.e_score_correction_bias",
                 "layers.0.self_attn.indexer.weights_proj.weight",
                 "layers.0.self_attn.indexer.ape"):
        assert name in params and any(s in name for s in pins), name
