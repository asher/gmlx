#!/usr/bin/env python3
"""Flint shared-expert fold: the shared expert is two routed experts wide
(1024 next to 512), so it rides the fused gathers as two slots where the
kq build has those kernels, and stays a separate MLP where it does not.
The eligibility tests fake the kq probes. The output test runs the real
kernels on q4_1 weights and skips on a kq build without the slot count."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_kquant as kq
from helpers import _real_apple_gpu

import gmlx.load.modules as modules
from gmlx.load.modules import install_fused_moe_glu, install_hyv3_shexp_fold
from gmlx.models.flint.model import FlintMoE, ModelArgs

D, INTER = 256, 256


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _args(shared=2 * INTER, experts=8, top_k=3):
    return ModelArgs(
        model_type="flint", hidden_size=D, num_hidden_layers=1,
        intermediate_size=512, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, vocab_size=64, layer_types=["full_attention"],
        num_experts=experts, num_experts_per_tok=top_k,
        moe_intermediate_size=INTER, shared_expert_intermediate_size=shared,
        routed_scaling_factor=0.0, norm_topk_prob=True)


def _kq_moe(shared=2 * INTER, codec="q4_1", shexp_codec="q4_1", weights=False):
    """A Flint MoE block on KQuant projections: zeros by default (the
    eligibility reads shapes and codecs only), or random weights
    quantized to the codecs."""
    from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

    args = _args(shared)
    block = FlintMoE(args)
    rng = np.random.default_rng(5)

    def wire(shape, c):
        w = rng.standard_normal(shape).astype(np.float32) / np.sqrt(shape[-1])
        q = kq.quantize(mx.array(w).reshape(-1, shape[-1]), c)[0]
        return q.reshape(*shape[:-1], q.shape[-1])

    e = args.num_experts
    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        m = KQuantSwitchLinear(e, o, i, False, codec)
        if weights:
            m.weight = wire((e, o, i), codec)
        setattr(block.switch_mlp, name, m)
    for name, (o, i) in (("gate_proj", (shared, D)), ("up_proj", (shared, D)),
                         ("down_proj", (D, shared))):
        m = KQuantLinear(i, o, False, shexp_codec)
        if weights:
            m.weight = wire((o, i), shexp_codec)
        setattr(block.shared_experts, name, m)
    if weights:
        block.gate.weight = mx.array(
            rng.standard_normal((e, D)).astype(np.float32) / np.sqrt(D))
    block.eval()
    return block


def _installed(blk):
    shell = _Shell(blk)
    return install_fused_moe_glu(shell), install_hyv3_shexp_fold(shell)


def _fake_probes(monkeypatch, answer):
    calls = []

    def glu(*a, **k):
        calls.append(("glu", a, k))
        if isinstance(answer, Exception):
            raise answer
        return answer

    def mix(*a, **k):
        calls.append(("mix", a, k))
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(kq, "shexp_glu_combo_has_kernel", glu, raising=False)
    monkeypatch.setattr(kq, "shexp_mix_combo_has_kernel", mix, raising=False)
    return calls


needs_fused = pytest.mark.skipif(
    not modules._FusedMoeCaps().has_base,
    reason="installed mlx-kquant has no fused MoE gathers")


@needs_fused
def test_two_wide_shared_expert_is_stamped_with_its_slot_count(monkeypatch):
    calls = _fake_probes(monkeypatch, True)
    blk = _kq_moe()
    assert _installed(blk) == (1, 1)
    assert blk.switch_mlp._kq_shexp_mod is blk.shared_experts
    glu = next(c for c in calls if c[0] == "glu")
    mix = next(c for c in calls if c[0] == "mix")
    assert glu[1] == ("q4_1", "q4_1", "q4_1") and glu[2] == {"shared_slots": 2}
    assert mix[1] == ("q4_1", "q4_1")
    assert mix[2] == {"shared_slots": 2, "slot_width": INTER}


@needs_fused
@pytest.mark.parametrize("answer", [False, TypeError("no shared_slots")])
def test_kq_without_the_kernels_keeps_the_shared_expert_apart(monkeypatch, answer):
    _fake_probes(monkeypatch, answer)
    blk = _kq_moe()
    assert _installed(blk) == (1, 0)
    assert getattr(blk.switch_mlp, "_kq_shexp_mod", None) is None


@needs_fused
def test_shared_width_off_a_multiple_is_not_stamped(monkeypatch):
    calls = _fake_probes(monkeypatch, True)
    blk = _kq_moe(shared=INTER * 3 // 2)
    assert _installed(blk) == (1, 0)
    assert calls == []


@needs_fused
def test_one_wide_shared_expert_needs_no_probe(monkeypatch):
    calls = _fake_probes(monkeypatch, False)
    blk = _kq_moe(shared=INTER)
    assert _installed(blk) == (1, 1)
    assert calls == []


def _multi_kernels():
    try:
        return bool(kq.shexp_mix_combo_has_kernel(
            "q4_1", "q4_1", shared_slots=2, slot_width=INTER))
    except (AttributeError, TypeError):
        return False


@needs_fused
@pytest.mark.skipif(not _real_apple_gpu(), reason="GPU kernels")
@pytest.mark.skipif(not _multi_kernels(),
                    reason="installed mlx-kquant has no multi-slot shared expert")
@pytest.mark.parametrize("t", [1, 2])
def test_folded_two_wide_shared_expert_matches_the_block(t):
    blk = _kq_moe(weights=True)
    x = mx.array(np.random.default_rng(9).standard_normal((1, t, D))
                 .astype(np.float32)).astype(mx.bfloat16)
    want = blk(x).astype(mx.float32)
    assert _installed(blk) == (1, 1)
    seen = []
    real = kq.moe_glu_gather_shexp_kq

    def spy(*a, **k):
        h = real(*a, **k)
        seen.append(h.shape)
        return h

    kq.moe_glu_gather_shexp_kq = spy
    try:
        got = blk(x).astype(mx.float32)
    finally:
        kq.moe_glu_gather_shexp_kq = real
    # three routed slots and the two shared ones
    assert seen == [(t, 5, INTER)]
    err = float(mx.abs(got - want).max() / mx.abs(want).max())
    assert err < 2e-2, err


needs_mix = pytest.mark.skipif(
    not hasattr(kq, "gather_mix"), reason="installed mlx-kquant has no gather_mix")


@needs_fused
@needs_mix
@pytest.mark.skipif(not _real_apple_gpu(), reason="GPU kernels")
@pytest.mark.skipif(not _multi_kernels(),
                    reason="installed mlx-kquant has no multi-slot shared expert")
@pytest.mark.parametrize("concat", [True, False])
def test_stamped_prefill_mixes_in_one_dispatch(monkeypatch, concat):
    """At sorting widths the stamped block mixes through gather_mix and
    adds the shared expert itself, so its output is the unstamped
    block's, bit for bit."""
    monkeypatch.setattr(modules, "_GATEUP_CONCAT_ENABLED", concat)
    blk = _kq_moe(weights=True)
    assert _installed(blk) == (1, 1)
    x = mx.array(np.random.default_rng(3).standard_normal((1, 64, D))
                 .astype(np.float32)).astype(mx.bfloat16)
    calls = []
    real = kq.gather_mix
    monkeypatch.setattr(
        kq, "gather_mix", lambda *a, **k: calls.append(1) or real(*a, **k))
    got = blk(x)
    mx.eval(got)
    assert calls == [1]
    object.__setattr__(blk.switch_mlp, "_kq_shexp_mod", None)
    want = blk(x)
    mx.eval(want)
    assert got.shape == want.shape == (1, 64, D)
    assert mx.array_equal(got, want).item()


@needs_fused
@needs_mix
def test_stamped_prefill_stays_unmixed_off_the_gpu_and_when_streamed():
    """The stream wrapper's token split joins unmixed leaves, and a CPU
    stream would pull the shared expert off the GPU: both keep the
    stamped return unmixed for the block to mix."""
    blk = _kq_moe()
    assert _installed(blk)[0] == 1
    glu = blk.switch_mlp
    object.__setattr__(glu, "_kq_shexp_mod", blk.shared_experts)
    inds = mx.zeros((1, 64, 3), dtype=mx.uint32)
    scores = mx.ones((1, 64, 3), dtype=mx.bfloat16)
    with mx.stream(mx.gpu):
        assert glu._kq_mix_prefill_ok(inds, scores)
    with mx.stream(mx.cpu):
        assert not glu._kq_mix_prefill_ok(inds, scores)
    object.__setattr__(glu, "_kq_scores_sink", True)
    with mx.stream(mx.gpu):
        assert not glu._kq_mix_prefill_ok(inds, scores)
    object.__setattr__(glu, "_kq_shexp_mod", None)
    with mx.stream(mx.gpu):
        assert glu._kq_mix_prefill_ok(inds, scores)
