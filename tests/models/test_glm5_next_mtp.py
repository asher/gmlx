#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""glm5_next MTP: verify-sink rollback parity, drafter cycle, greedy identity.

The losslessness contract is drafter-independent (the verify walk emits the
target's own tokens), so the gates that matter are the cache paths:

- the explicit rollback test: a verify block + rollback must leave every
  cache leaf (KDA conv tails + recurrent state via sink replay, MLA latent
  KV + pool cache via trim/undo) equal to a clean forward over the kept
  prefix, across all four pool-phase residues;
- reject rounds (random drafter vs random target) and accept rounds (oracle
  drafter) through the real engine loop must be token-identical to a plain
  greedy decode over the same prefill chunking.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx.models.glm5_next.mtp import (
    Glm5NextMTPConfig,
    Glm5NextMTPDrafter,
    Glm5NextSpecLM,
)

from test_glm5_next import _tiny_args

PREFILL_CHUNK = 8
N_GEN = 24

# The greedy-identity A/B is only defined where the target's argmax is
# numerically unambiguous: the verify forward derives tokens through the
# qL = 2 path, whose FP rounding differs from the reference's 1-token decode
# path by ~1e-3 in logit space. A top-2 margin below the floor can flip the
# argmax - a tie-break artifact, not a lost token.
GREEDY_TIE_TOL = 1e-2


def _randomize(module, seed=0, scale=0.05):
    mx.random.seed(seed)
    flat = [(k, mx.random.normal(v.shape) * scale)
            for k, v in nn.utils.tree_flatten(module.parameters())]
    module.load_weights(flat, strict=False)
    for layer in getattr(module, "layers", []):
        if getattr(layer, "is_linear", False):
            layer.self_attn.a_folded = -mx.random.uniform(
                low=1.0, high=4.0, shape=layer.self_attn.a_folded.shape)
    mx.eval(module.parameters())


def _build_lm(seed=0, scale=0.05):
    lm = Glm5NextSpecLM(_tiny_args())
    _randomize(lm, seed=seed, scale=scale)
    return lm


def _build_drafter(args, cls=Glm5NextMTPDrafter, block_size=2):
    drafter = cls(Glm5NextMTPConfig(text_config=args, block_size=block_size))
    _randomize(drafter, seed=3)
    return drafter


def _cache_leaf_states(caches):
    out = []
    for c in caches:
        st = c.state
        out.append(list(st) if isinstance(st, (list, tuple)) else [st])
    return out


def _assert_states_close(a, b, atol):
    assert len(a) == len(b)
    for ca, cb in zip(a, b):
        assert len(ca) == len(cb)
        for x, y in zip(ca, cb):
            if isinstance(x, (list, tuple)):
                assert isinstance(y, (list, tuple))
                _assert_states_close([x], [y], atol)
                continue
            if x is None or y is None:
                assert x is None and y is None
                continue
            assert x.shape == y.shape, (x.shape, y.shape)
            if x.size == 0:  # the latent KV's zero-width value tensor
                continue
            if x.dtype in (mx.int32, mx.int64, mx.uint32, mx.uint64):
                assert mx.array_equal(x, y)
            else:
                diff = mx.abs(x.astype(mx.float32) - y.astype(mx.float32))
                assert float(diff.max()) < atol, float(diff.max())


# All four pool-phase residues at the rollback boundary: with kpool = 4,
# prompt + 3 kept tokens ends at offsets 6, 12, 15, 17 (mod 4 = 2, 0, 3, 1),
# and 9 / 12 / 14 also push total keys past the tiny n_select = 11 so the
# verify forward itself runs with sparse selection engaged.
@pytest.mark.parametrize("prompt_len", [3, 9, 12, 14])
def test_verify_rollback_matches_prefix_forward(prompt_len):
    """A verify block of 4 followed by rollback to accepted = 2 (3 kept)
    leaves every cache leaf (KDA conv tails + fp32 recurrent state, MLA
    latent KV, pool cache) equal to a clean forward over prompt + the 3
    kept tokens, and the next decode step's logits agree."""
    lm = _build_lm()
    args = lm.args
    vocab = args.vocab_size
    mx.random.seed(1)
    prompt = mx.random.randint(0, vocab, (1, prompt_len))
    block = mx.random.randint(0, vocab, (1, 4))
    nxt = mx.random.randint(0, vocab, (1, 1))

    # Speculative path: prefill, verify 4, rollback to accepted=2.
    cache_a = lm.make_cache()
    out = lm(prompt, cache=cache_a, return_hidden=True)
    assert out.hidden_states[-1].shape == (1, prompt_len, args.hidden_size)
    raw, _, sink = lm.speculative_verify_hidden(block, cache_a)
    assert raw.shape[1] == 4
    assert len(sink) == sum(1 for lt in args.layer_types
                            if lt == "linear_attention")
    lm.rollback_speculative_cache(cache_a, sink, 2, 4)
    logits_a = lm(nxt, cache=cache_a).logits
    mx.eval(logits_a)

    # Reference: one forward over prompt + the 3 kept tokens.
    cache_b = lm.make_cache()
    lm(mx.concatenate([prompt, block[:, :3]], axis=1), cache=cache_b)
    logits_b = lm(nxt, cache=cache_b).logits
    mx.eval(logits_b)

    for ca, cb in zip(cache_a, cache_b):
        if hasattr(ca, "caches"):
            assert ca[0].offset == cb[0].offset
            assert (ca[1]._plen, ca[1].remainder) == (
                cb[1]._plen, cb[1].remainder)
    _assert_states_close(
        _cache_leaf_states(cache_a), _cache_leaf_states(cache_b), 2e-4)
    assert float(mx.abs(logits_a - logits_b).max()) < 2e-3


def test_rollback_refuses_untrimmable_pool():
    """Once the pool undo log is consumed the rollback must fail loudly,
    never silently under-trim."""
    lm = _build_lm()
    cache = lm.make_cache()
    # 8 + 4 ends on a pool boundary (remainder 0): the trim must cross a
    # completed window, which only the undo log can rewind.
    lm(mx.random.randint(0, 8, (1, 8)), cache=cache)
    _, _, sink = lm.speculative_verify_hidden(
        mx.random.randint(0, 8, (1, 4)), cache)
    for entry in cache:
        if hasattr(entry, "caches"):
            entry[1]._undo = None  # simulate a consumed undo log
    with pytest.raises(RuntimeError, match="untrimmable"):
        lm.rollback_speculative_cache(cache, sink, 2, 4)


def test_logits_from_hidden_matches_forward():
    lm = _build_lm()
    ids = mx.random.randint(0, lm.args.vocab_size, (1, 5))
    out = lm(ids, return_hidden=True)
    ref = lm(ids).logits
    got = lm.speculative_logits_from_hidden(out.hidden_states[-1])
    assert float(mx.abs(got - ref).max()) < 1e-5
    assert mx.array_equal(
        lm.speculative_argmax_from_hidden(out.hidden_states[-1]),
        mx.argmax(ref, axis=-1))


def _wrap_target(lm):
    from gmlx.load.loader import MTPTextTarget

    return MTPTextTarget(lm, {"model_type": "glm5_next"})


def _greedy_reference(lm, prompt, n_gen):
    """Plain greedy decode; returns (tokens, per-step top-2 logit margins)."""
    cache = lm.make_cache()
    i = 0
    while i < prompt.shape[1]:
        logits = lm(prompt[:, i:i + PREFILL_CHUNK], cache=cache).logits
        i += PREFILL_CHUNK

    def _step(row):
        srt = mx.sort(row)
        return int(mx.argmax(row).item()), float((srt[-1] - srt[-2]).item())

    tok, gap = _step(logits[0, -1])
    out, gaps = [tok], [gap]
    for _ in range(n_gen - 1):
        logits = lm(mx.array([[tok]], dtype=mx.int32), cache=cache).logits
        tok, gap = _step(logits[0, -1])
        out.append(tok)
        gaps.append(gap)
    return out, gaps


def _spec_decode(target, lm, drafter, prompt, n_gen):
    from gmlx.spec.speculative import stream_speculative

    drafter.reset(target)
    return list(
        stream_speculative(
            target,
            drafter,
            prompt,
            prompt_cache=lm.make_cache(),
            max_tokens=n_gen,
            sampler=None,
            draft_block_size=2,
            prefill_chunk=PREFILL_CHUNK,
        )
    )


class _OracleDrafter(Glm5NextMTPDrafter):
    """Drafts a known continuation -> every draft accepted (no rollback)."""

    def __init__(self, config, script=None):
        super().__init__(config)
        self._script = script or []
        self._pos = 0

    def reset(self, target_model, left_padding=None):
        out = super().reset(target_model, left_padding)
        self._pos = 1  # position 0 is the engine-sampled first token
        return out

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler,
                    token_dtype=mx.int32, greedy=False):
        tok = super().draft_block(
            last_bonus, hidden, cache, block_size, sampler, token_dtype,
            greedy)
        n = int(tok.shape[1])
        window = self._script[self._pos:self._pos + n]
        if len(window) == n:
            tok = mx.array([[int(t) for t in window]], dtype=token_dtype)
        return tok

    def accept_verified_tokens(self, verify_hidden, draft_tokens, accepted,
                               new_tokens, sampler, token_dtype=mx.int32,
                               greedy=False):
        self._pos += int(accepted) + 1
        super().accept_verified_tokens(
            verify_hidden, draft_tokens, accepted, new_tokens, sampler,
            token_dtype, greedy)


def test_mtp_greedy_identity_reject_path():
    # scale 0.2 / seed 5 with a seed-7 prompt: every greedy step's top-2
    # margin clears GREEDY_TIE_TOL (probed), so the full trajectory gates
    # the rollback.
    lm = _build_lm(seed=5, scale=0.2)
    target = _wrap_target(lm)
    drafter = _build_drafter(lm.args)
    # 41 > n_select = 11: decode + verify run with sparse selection engaged.
    mx.random.seed(7)
    prompt = mx.random.randint(0, lm.args.vocab_size, (1, 41)).astype(mx.int32)

    ref, gaps = _greedy_reference(lm, prompt, N_GEN)
    spec = _spec_decode(target, lm, drafter, prompt, N_GEN)

    # Compare only the prefix above the tie-break floor (see GREEDY_TIE_TOL);
    # a flip desyncs the whole tail, so stop at the first ambiguous step.
    k = next((i for i, g in enumerate(gaps) if g < GREEDY_TIE_TOL), N_GEN)
    assert k >= 8, f"greedy trajectory too tie-dense to gate rollback (k={k})"
    assert spec[:k] == ref[:k]
    # A random drafter against a random target: rejections must dominate, so
    # the rollback path is genuinely exercised.
    assert any(a == 0 for a in drafter.accept_lens)


def test_mtp_greedy_identity_accept_path():
    lm = _build_lm(seed=5, scale=0.2)
    target = _wrap_target(lm)
    mx.random.seed(7)
    prompt = mx.random.randint(0, lm.args.vocab_size, (1, 41)).astype(mx.int32)

    ref, _ = _greedy_reference(lm, prompt, N_GEN)
    oracle = _build_drafter(lm.args, cls=_OracleDrafter)
    oracle._script = ref
    spec = _spec_decode(target, lm, oracle, prompt, N_GEN)

    assert spec == ref
    accepts = list(oracle.accept_lens)
    assert sum(1 for a in accepts if a == 1) >= len(accepts) - 1, (
        "oracle drafts must (near-)all be accepted (accept-path gate; the "
        "final round may be budget-clamped below the full draft width)")


def test_speclm_hooks_match_loader_contract():
    from gmlx.load.loader import _MTP_TARGET_HOOKS_BY_TYPE, _mtp_target_classes

    for hook in _MTP_TARGET_HOOKS_BY_TYPE["glm5_next"]:
        assert callable(getattr(Glm5NextSpecLM, hook, None)), hook
    cls, build = _mtp_target_classes("glm5_next")
    assert cls is Glm5NextSpecLM


def test_drafter_validates_and_rejects_batch():
    from gmlx.spec.drafter_protocol import validate_drafter

    args = _tiny_args()
    drafter = _build_drafter(args)
    validate_drafter(drafter)
    assert drafter.cap_at_configured_depth
    assert drafter.requires_owned_engine
    assert not drafter.supports_kv_sidecar
    with pytest.raises(NotImplementedError):
        drafter.make_cache(left_padding=[0])
    with pytest.raises(NotImplementedError):
        drafter.inject_rows(None, None, None, None)
    # v1 pins the rollout depth: the accept-path pool trim rewinds through a
    # one-update undo log, so a deeper block would silently desync.
    with pytest.raises(ValueError, match="block_size=2"):
        Glm5NextMTPDrafter(
            Glm5NextMTPConfig(text_config=args, block_size=3))
    # Default rollout feed is the PRE-norm hidden (GMLX_MTP_POSTNORM_FEED
    # flips it); the identity must hold so the A/B is a real toggle.
    h = mx.random.normal((1, 1, args.hidden_size))
    if drafter._postnorm_feed:
        assert mx.allclose(drafter._next_hidden(h), drafter.norm(h))
    else:
        assert drafter._next_hidden(h) is h


def test_drafter_cache_is_latent_pool_pair():
    from gmlx.models.deepseek_v4.cache import PoolingCache

    drafter = _build_drafter(_tiny_args())
    (cache,) = drafter.make_cache()
    # Dual-origin rule: the CacheList/KVCache come from the construction
    # module (either cache origin), so assert by name, never isinstance.
    assert type(cache).__name__ == "CacheList"
    assert type(cache[0]).__name__ == "KVCache"
    # The pool cache IS identity-checked: every APC seam keys on this class.
    assert isinstance(cache[1], PoolingCache)
    assert cache[1].ratio == 4
    assert cache[1].quantizable is False


def test_fp32_pins_cover_drafter_tree():
    from gmlx.load.loader import _FP32_KEEP_BY_MODEL_TYPE

    pins = _FP32_KEEP_BY_MODEL_TYPE["glm5_next"]
    for name in (
        "layers.0.mlp.gate.weight",
        "layers.0.mlp.gate.e_score_correction_bias",
        "layers.0.self_attn.indexer.weights_proj.weight",
        "layers.0.self_attn.indexer.compressor.ape",
    ):
        assert any(s in name for s in pins), name
    for name in (
        "layers.0.self_attn.q_b_proj.weight",
        "layers.0.mlp.switch_mlp.gate_proj.weight",
        "fc.weight",
        "norm.weight",
    ):
        assert not any(s in name for s in pins), name


def test_mtp_remap_covers_closed_tensor_set():
    """The real GGUF's MTP block (blk.45) is exactly 29 tensors: a full DSA
    decoder layer (MLA + indexer + sigmoid MoE, NO hyper-connection rows)
    plus the four nextn extras. The remap must map all of them onto drafter
    params, covering the full drafter tree (both directions closed)."""
    from mlx.utils import tree_flatten

    from gmlx.load.loader import remap_mtp_arrays

    args = _tiny_args()
    drafter = _build_drafter(args)
    params = {k for k, _ in tree_flatten(drafter.parameters())}

    blk = args.num_hidden_layers  # 4: the block past the tiny trunk
    layer_tensors = (
        "attn_norm.weight", "ffn_norm.weight", "attn_output.weight",
        "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
        "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
        "attn_k_b.weight", "attn_v_b.weight",
        "indexer.attn_q_b.weight", "indexer.attn_k.weight",
        "indexer.k_norm.weight", "indexer.k_norm.bias",
        "indexer.proj.weight", "indexer_compressor_ape.weight",
        "indexer_compressor_gate.weight",
        "ffn_gate_inp.weight", "exp_probs_b.bias",
        "ffn_gate_exps.weight", "ffn_up_exps.weight",
        "ffn_down_exps.weight", "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    )
    arrays = {f"blk.{blk}.{t}": mx.zeros((4, 4)) for t in layer_tensors}
    for t in ("eh_proj", "enorm", "hnorm", "shared_head_norm"):
        arrays[f"blk.{blk}.nextn.{t}.weight"] = mx.zeros((4, 4))
    # Trunk tensors must be ignored by the MTP remap.
    arrays["blk.0.attn_q.weight"] = mx.zeros((4, 4))

    weights, meta, stats = remap_mtp_arrays(
        arrays, {}, "glm5next", first_mtp_block=blk, num_mtp_layers=1)
    assert stats["mapped"] == 29 and stats["skipped"] == 0
    produced = set(weights)
    assert produced == params, (
        f"remap/drafter tree mismatch: only-remap={sorted(produced - params)} "
        f"only-drafter={sorted(params - produced)}")
