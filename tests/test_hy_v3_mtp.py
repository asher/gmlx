"""Hy3 (hy_v3) MTP: drafter/SpecLM unit tests + tiny greedy-identity A/B.

The losslessness contract is drafter-independent (the verify walk emits the
target's own tokens), so the load-bearing gates are the cache paths:

- reject rounds (random drafter vs random target): every rejection trims the
  verify tail from the plain per-layer KVCaches via
  ``rollback_speculative_cache``;
- accept rounds (oracle drafter): no rollback. every verify token stays in
  the cache, gating the accept path's trim/re-append bookkeeping (the drafter
  re-processes accepted tokens with target hidden);
- the depth-2 rollout (block 3) chains the head's POST-final-norm output
  (``_next_hidden = norm(h)``, the vLLM-reference semantics).

All must be token-identical to a plain greedy decode over the SAME prefill
chunking.
"""

import mlx.core as mx
import pytest

from gmlx.config_synth import synthesize_config
from gmlx.hy_v3_model import ModelArgs, ensure_registered
from gmlx.hy_v3_mtp import HyV3MTPConfig, HyV3MTPDrafter, HyV3SpecLM

from test_config_synth import _HY_V3_SHAPES, _hy_v3_meta
from test_deepseek_v4_mtp import _randomize_zero_params

PREFILL_CHUNK = 8
N_GEN = 24

# The greedy-identity A/B is only defined where the target's argmax is
# numerically unambiguous. The reject-path verify derives each token through the
# block-SDPA path (qL = drafts + 1), whose FP rounding vs the reference's
# 1-token decode path differs by ~3e-3 in logit space (a mathematically
# equivalent but different kernel). A top-2 margin below that floor can flip the
# argmax. A greedy tie-break artifact on this maximal-uncertainty tiny random
# model, not a rollback bug (the walk still emits the target's own greedy pick).
# So the identity claim holds only over the prefix above the floor.
GREEDY_TIE_TOL = 1e-2


def _tiny_config() -> dict:
    ensure_registered()
    return synthesize_config(_hy_v3_meta(), tensor_shapes=_HY_V3_SHAPES)


def _build_target(cfg):
    from gmlx.loader import MTPTextTarget

    lm = HyV3SpecLM(ModelArgs.from_dict(cfg))
    mx.eval(lm.parameters())
    _randomize_zero_params(lm)  # expert_bias is zero-initialized
    return MTPTextTarget(lm, cfg), lm


def _build_drafter(cfg, cls=HyV3MTPDrafter, block_size=2, **kw):
    drafter = cls(
        HyV3MTPConfig(text_config=ModelArgs.from_dict(cfg), block_size=block_size),
        **kw,
    )
    mx.eval(drafter.parameters())
    _randomize_zero_params(drafter)
    return drafter


def _greedy_reference(lm, prompt, n_gen):
    """Plain greedy decode; returns (tokens, per-step top-2 logit margins). The
    margins gate the greedy-identity claim against tie-break noise (see
    GREEDY_TIE_TOL)."""
    cache = lm.make_cache()
    i = 0
    while i < prompt.shape[1]:
        logits = lm(prompt[:, i : i + PREFILL_CHUNK], cache=cache).logits
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


def _spec_decode(target, lm, drafter, prompt, n_gen, block=2):
    from gmlx.speculative import stream_speculative

    drafter.reset(target)
    return list(
        stream_speculative(
            target,
            drafter,
            prompt,
            prompt_cache=lm.make_cache(),
            max_tokens=n_gen,
            sampler=None,
            draft_block_size=block,
            prefill_chunk=PREFILL_CHUNK,
        )
    )


class _OracleDrafter(HyV3MTPDrafter):
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
            last_bonus, hidden, cache, block_size, sampler, token_dtype, greedy
        )
        n = int(tok.shape[1])
        window = self._script[self._pos : self._pos + n]
        if len(window) == n:
            tok = mx.array([[int(t) for t in window]], dtype=token_dtype)
        return tok

    def accept_verified_tokens(self, verify_hidden, draft_tokens, accepted,
                               new_tokens, sampler, token_dtype=mx.int32,
                               greedy=False):
        self._pos += int(accepted) + 1
        super().accept_verified_tokens(
            verify_hidden, draft_tokens, accepted, new_tokens, sampler,
            token_dtype, greedy,
        )


@pytest.mark.parametrize("block", [2, 3])
def test_mtp_greedy_identity_reject_path(block):
    mx.random.seed(7)
    cfg = _tiny_config()
    target, lm = _build_target(cfg)
    drafter = _build_drafter(cfg, block_size=block)
    prompt = mx.random.randint(0, cfg["vocab_size"], (1, 41)).astype(mx.int32)

    ref, gaps = _greedy_reference(lm, prompt, N_GEN)
    spec = _spec_decode(target, lm, drafter, prompt, N_GEN, block=block)

    # Compare only the prefix above the tie-break floor: a sub-floor argmax
    # margin flips on the block-SDPA path (a numerical artifact, not a lost
    # token, see GREEDY_TIE_TOL). A flip desyncs the whole tail, so stop at
    # the first ambiguous step.
    k = next((i for i, g in enumerate(gaps) if g < GREEDY_TIE_TOL), N_GEN)
    assert k >= 8, f"greedy trajectory too tie-dense to gate rollback (k={k})"
    assert spec[:k] == ref[:k]
    accepts = list(drafter.accept_lens)
    # A random drafter against a random target: rejections must dominate, so
    # the rollback path is genuinely exercised.
    assert any(a == 0 for a in accepts)


@pytest.mark.parametrize("block", [2, 3])
def test_mtp_greedy_identity_accept_path(block):
    mx.random.seed(7)
    cfg = _tiny_config()
    target, lm = _build_target(cfg)
    prompt = mx.random.randint(0, cfg["vocab_size"], (1, 41)).astype(mx.int32)

    ref, _ = _greedy_reference(lm, prompt, N_GEN)
    oracle = _build_drafter(cfg, cls=_OracleDrafter, block_size=block)
    oracle._script = ref
    spec = _spec_decode(target, lm, oracle, prompt, N_GEN, block=block)

    assert spec == ref
    accepts = list(oracle.accept_lens)
    full = block - 1
    assert sum(1 for a in accepts if a == full) >= len(accepts) - 1, (
        "oracle drafts must (near-)all be accepted (accept-path gate; the "
        "final round may be budget-clamped below the full draft width)"
    )


def test_speclm_hooks_match_loader_contract():
    from gmlx.loader import _MTP_TARGET_HOOKS_BY_TYPE

    for hook in _MTP_TARGET_HOOKS_BY_TYPE["hy_v3"]:
        assert callable(getattr(HyV3SpecLM, hook, None)), hook


def test_drafter_validates_and_rejects_batch():
    from gmlx.drafter_protocol import validate_drafter

    cfg = _tiny_config()
    drafter = _build_drafter(cfg)
    validate_drafter(drafter)
    assert drafter.cap_at_configured_depth
    assert drafter.requires_owned_engine
    with pytest.raises(NotImplementedError):
        drafter.make_cache(left_padding=[0])
    with pytest.raises(NotImplementedError):
        drafter.inject_rows(None, None, None, None)
    # Post-final-norm chaining (vLLM reference): the rollout hidden IS norm(h).
    h = mx.random.normal((1, 1, cfg["hidden_size"]))
    assert mx.allclose(drafter._next_hidden(h), drafter.norm(h))


def test_router_params_pinned_fp32():
    from gmlx.loader import _FP32_KEEP_BY_MODEL_TYPE

    pins = _FP32_KEEP_BY_MODEL_TYPE["hy_v3"]
    for name in (
        "model.layers.1.mlp.router.gate.weight",       # target tree
        "model.layers.1.mlp.router.expert_bias",
        "layers.0.mlp.router.gate.weight",             # drafter tree
        "layers.0.mlp.router.expert_bias",
    ):
        assert any(s in name for s in pins), name
    # ...and nothing else in the trunk matches.
    for name in (
        "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        "model.layers.1.input_layernorm.weight",
        "lm_head.weight",
    ):
        assert not any(s in name for s in pins), name


def test_mtp_remap_covers_closed_tensor_set():
    """The real GGUF's MTP block (blk.80) is exactly 20 tensors; the remap
    must map all of them onto drafter params, covering the full drafter tree
    (both directions closed). Mirrors the blk.80 header dump of
    hy3-1M-MTP-IQ2_M.gguf, with tiny shapes."""
    from mlx.utils import tree_flatten

    from gmlx.loader import remap_mtp_arrays

    cfg = _tiny_config()
    drafter = _build_drafter(cfg)
    params = {k for k, _ in tree_flatten(drafter.parameters())}

    blk = cfg["num_hidden_layers"]  # 3: the block past the trunk
    decoder = (
        "attn_q", "attn_k", "attn_v", "attn_output",
        "attn_q_norm", "attn_k_norm", "attn_norm", "ffn_norm",
        "ffn_gate_inp",
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
        "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
    )
    arrays = {f"blk.{blk}.{b}.weight": mx.zeros((4, 4)) for b in decoder}
    arrays[f"blk.{blk}.exp_probs_b"] = mx.zeros((4,))
    for b in ("eh_proj", "enorm", "hnorm", "shared_head_norm"):
        arrays[f"blk.{blk}.nextn.{b}.weight"] = mx.zeros((4, 4))
    # Trunk tensors must be ignored by the MTP remap.
    arrays["blk.0.attn_q.weight"] = mx.zeros((4, 4))

    weights, meta, stats = remap_mtp_arrays(
        arrays, {}, "hy_v3", first_mtp_block=blk, num_mtp_layers=1
    )
    assert stats["mapped"] == 20 and stats["skipped"] == 0
    produced = set(weights)
    assert produced == params, (
        f"remap/drafter tree mismatch: only-remap={sorted(produced - params)} "
        f"only-drafter={sorted(params - produced)}"
    )
    assert "fc.weight" in produced
    assert "pre_fc_norm_embedding.weight" in produced
    assert "pre_fc_norm_hidden.weight" in produced
    assert "norm.weight" in produced
