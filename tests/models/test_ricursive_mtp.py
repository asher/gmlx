"""Ripple speculative target: the packed-hidden capture seam and the
verify and rollback walk, on a tiny random model past its sliding window."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from gmlx.load.config_synth import synthesize_config
from gmlx.models.ricursive.model import ModelArgs
from gmlx.models.ricursive.mtp import RicursiveSpecLM
from gmlx.spec.helpers import _buffer_mtp_target_cache

from test_config_synth import _RICURSIVE_SHAPES, _ricursive_meta

CAPTURE = (0, 3)
N_GEN = 16
BLOCK = 4
# A draw whose greedy chain has a clear top-2 margin at every step, so the
# identity claim is not vacuous.
SEED = 3
# Verify runs a block SDPA whose rounding differs from 1-token decode. A
# gap above this floor is a real divergence.
GREEDY_TIE_TOL = 1e-3


def _build():
    mx.random.seed(SEED)
    cfg = synthesize_config(_ricursive_meta(), tensor_shapes=_RICURSIVE_SHAPES)
    lm = RicursiveSpecLM(ModelArgs.from_dict(cfg))
    mx.eval(lm.parameters())
    return lm, cfg


def _spec_cache(lm):
    cache = lm.make_cache()
    _buffer_mtp_target_cache(
        cache, SimpleNamespace(config=SimpleNamespace(block_size=BLOCK)), BLOCK)
    return cache


def _packed_width(cfg):
    return cfg["hidden_size"] * (1 + len(CAPTURE))


def test_hooks_match_the_loader_contract():
    from gmlx.load.mtp_target import (
        _MTP_TARGET_HOOKS_BY_TYPE,
        _mtp_target_classes,
    )

    for hook in _MTP_TARGET_HOOKS_BY_TYPE["ricursive"]:
        assert callable(getattr(RicursiveSpecLM, hook, None)), hook
    cls, build = _mtp_target_classes("ricursive")
    assert cls is RicursiveSpecLM
    _, cfg = _build()
    assert isinstance(build(cfg), RicursiveSpecLM)


def test_capture_packs_and_the_trunk_slice_recovers_it():
    lm, cfg = _build()
    ids = mx.array([[1, 2, 3, 4, 5, 6]])
    bare, _ = lm.speculative_verify_hidden(ids, lm.make_cache())
    assert bare.shape[-1] == cfg["hidden_size"]

    lm.set_dflash_capture(CAPTURE)
    packed, _ = lm.speculative_verify_hidden(ids, lm.make_cache())
    assert packed.shape[-1] == _packed_width(cfg)
    trunk = lm._dflash_trunk(packed)
    mx.eval(bare, trunk)
    assert float(mx.abs(trunk - bare).max().item()) == 0.0

    lm.set_dflash_capture(())
    ref = lm.speculative_logits_from_hidden(bare)
    lm.set_dflash_capture(CAPTURE)
    got = lm.speculative_logits_from_hidden(packed)
    assert float(mx.abs(ref - got).max().item()) == 0.0
    am = lm.speculative_argmax_from_hidden(packed)
    assert am.tolist() == mx.argmax(got, axis=-1).tolist()


def test_return_hidden_packs_the_prefill():
    lm, cfg = _build()
    lm.set_dflash_capture(CAPTURE)
    out = lm(mx.array([[1, 2, 3]]), cache=lm.make_cache(), return_hidden=True)
    assert out.hidden_states[0].shape[-1] == _packed_width(cfg)
    assert out.logits.shape[-1] == cfg["vocab_size"]


def test_trunk_is_materialized_before_the_logit_head(monkeypatch):
    """The real head is a quantized kernel that reads its buffer directly,
    so a strided view of the packed hidden must not reach it."""
    lm, cfg = _build()
    lm.set_dflash_capture(CAPTURE)
    calls = []
    real = mx.contiguous
    monkeypatch.setattr(
        mx, "contiguous",
        lambda x, *a, **k: (calls.append(x.shape), real(x, *a, **k))[1])
    lm.speculative_logits_from_hidden(mx.zeros((1, 3, _packed_width(cfg))))
    assert calls


def test_rollback_trims_every_layer_past_the_window():
    lm, cfg = _build()
    assert cfg["sliding_window"] < 10
    cache = _spec_cache(lm)
    lm.speculative_verify_hidden(mx.array([list(range(1, 11))]), cache)
    before = [c.offset for c in cache]
    lm.speculative_verify_hidden(mx.array([[7] * BLOCK]), cache)
    lm.rollback_speculative_cache(cache, None, 1, BLOCK)
    assert [c.offset for c in cache] == [b + 2 for b in before]
    lm.speculative_verify_hidden(mx.array([[5] * BLOCK]), cache)
    offsets = [c.offset for c in cache]
    lm.rollback_speculative_cache(cache, None, BLOCK - 1, BLOCK)
    assert [c.offset for c in cache] == offsets


def test_rollback_refuses_a_rotated_stock_ring():
    lm, _ = _build()
    cache = lm.make_cache()
    lm.speculative_verify_hidden(mx.array([list(range(1, 11))]), cache)
    with pytest.raises(RuntimeError, match="untrimmable"):
        lm.rollback_speculative_cache(cache, None, 0, BLOCK)


@pytest.mark.parametrize("armed", [False, True])
def test_verify_walk_is_token_identical_to_greedy(armed):
    """Whatever the drafts, the walk emits the target's own greedy tokens.
    Every draft is wrong, so every round rejects and rolls back, and the
    context runs well past the sliding window."""
    lm, cfg = _build()
    vocab = cfg["vocab_size"]
    prompt = mx.array([[1, 2, 3, 4, 5, 6, 7]])

    cache = lm.make_cache()
    ref_logits = lm.head(lm.model(prompt, cache))[0, -1]
    ref, margins = [], []
    for _ in range(N_GEN):
        top = mx.sort(ref_logits)[-2:]
        margins.append(float((top[1] - top[0]).item()))
        t = int(mx.argmax(ref_logits).item())
        ref.append(t)
        ref_logits = lm.head(lm.model(mx.array([[t]]), cache))[0, -1]
    assert min(margins) > GREEDY_TIE_TOL, (
        "the pinned draw has no clear greedy chain; choose another SEED")

    if armed:
        lm.set_dflash_capture(CAPTURE)
    cache2 = _spec_cache(lm)
    hid, _ = lm.speculative_verify_hidden(prompt, cache2)
    tok = int(lm.speculative_argmax_from_hidden(hid)[0, -1].item())
    got, accepts = [tok], []
    while len(got) < N_GEN:
        drafts = [(ref[len(got)] + 1 + i) % vocab for i in range(BLOCK - 1)]
        hid, _ = lm.speculative_verify_hidden(
            mx.array([[tok] + drafts]), cache2)
        rows = lm.speculative_argmax_from_hidden(hid)[0].tolist()
        accepted = 0
        for i, d in enumerate(drafts):
            if int(rows[i]) != d:
                break
            accepted += 1
        accepts.append(accepted)
        got.extend(drafts[:accepted])
        tok = int(rows[accepted])
        got.append(tok)
        lm.rollback_speculative_cache(cache2, None, accepted, BLOCK)

    assert got[:N_GEN] == ref
    assert accepts == [0] * len(accepts)
