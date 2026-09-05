"""kvarn on the B=1 MTP path: serve spec-cache conversion, shared-KV
declines, rollback contracts, and the R1 reject-cycle bit-equality gate."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

import mlx.core as mx

pytest.importorskip("mlx_vlm.generate.ar")

from mlx_lm.models.cache import KVCache, RotatingKVCache  # noqa: E402
from mlx_vlm.generate import ar  # noqa: E402
from mlx_vlm.server import generation as gen  # noqa: E402
from mlx_vlm.speculative import utils as su  # noqa: E402

import mlx_kquant as kq  # noqa: E402

import gmlx.spec.engine as spec_engine  # noqa: E402
from gmlx.cache.kvarn_cache import KVarNKVCache  # noqa: E402
from kvarn_testlib import Args, D, H, filled, needs_kvarn_ops, tokens  # noqa: E402


class _SSMCache:
    """ArraysCache stand-in: not a KVCache, no to_quantized."""

    def is_trimmable(self):
        return False


class _FakeLM:
    def __init__(self, **kw):
        self.args = Args(**kw)

    def make_cache(self):
        return [KVCache(), _SSMCache(), KVCache()]


class _ReadbackLM(_FakeLM):
    def speculative_logits_from_hidden(self, hidden):
        return hidden


class _Drafter:
    uses_shared_kv = False

    class config:
        block_size = 2


@pytest.fixture
def restorable(monkeypatch):
    # Start from the stock function even when an earlier module installed
    # the wrap (once per process, with the boot env it saw), and undo this
    # test's install at teardown.
    for mod in (su, ar, gen):
        fn = mod.make_speculative_prompt_cache
        monkeypatch.setattr(
            mod, "make_speculative_prompt_cache", getattr(fn, "_gmlx_orig", fn)
        )
    monkeypatch.delenv("KV_BITS", raising=False)
    monkeypatch.delenv("KV_TAIL_TOKENS", raising=False)
    return monkeypatch


def _mk(lm=None, batch_size=1, make_cache=None):
    return ar.make_speculative_prompt_cache(
        lm or _FakeLM(),
        draft_kind="mtp",
        batch_size=batch_size,
        left_padding=[0] * batch_size,
        make_cache=make_cache
        or (lambda lm, lp: pytest.fail("B=1 mtp bypass must not call make_cache")),
    )


# -- env params --------------------------------------------------------------


def test_params_kvarn_scheme_alone(restorable):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    assert spec_engine._spec_kv_quant_params() == dict(
        scheme="kvarn", kv_bits=6, value_bits=6, tail_tokens=1024)


def test_params_kvarn_bits_and_tail(restorable):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    restorable.setenv("KV_BITS", "4")
    restorable.setenv("KV_TAIL_TOKENS", "256")
    assert spec_engine._spec_kv_quant_params() == dict(
        scheme="kvarn", kv_bits=4, value_bits=4, tail_tokens=256)


def test_params_kvarn_malformed(restorable):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    restorable.setenv("KV_BITS", "4.5")
    assert spec_engine._spec_kv_quant_params() is None


def test_params_kvarn_kill_switch(restorable):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    restorable.setenv("GMLX_SPEC_KV_QUANT", "0")
    assert spec_engine._spec_kv_quant_params() is None


def test_params_affine_unchanged(restorable):
    restorable.setenv("KV_BITS", "8")
    assert spec_engine._spec_kv_quant_params() == dict(
        scheme="uniform", kv_bits=8, kv_group_size=64)
    restorable.setenv("KV_QUANT_SCHEME", "turboquant")
    assert spec_engine._spec_kv_quant_params() is None


def _stamp(lm, verdict="full", **single):
    from types import SimpleNamespace

    lm._gmlx_kv_policy = SimpleNamespace(
        single=SimpleNamespace(verdict=verdict, **single))
    return lm


def test_stamped_params_rule_the_boot_env(restorable):
    # No stamp: None, so the boot env decides. A stamp: its scheme and
    # widths, {} when it quantizes nothing.
    assert spec_engine._stamped_spec_params(_FakeLM()) is None
    kv = _stamp(_FakeLM(), scheme="kvarn", bits=4, value_bits=None,
                tail_tokens=256)
    assert spec_engine._stamped_spec_params(kv) == dict(
        scheme="kvarn", kv_bits=4, value_bits=4, tail_tokens=256)
    kv = _stamp(_FakeLM(), scheme="kvarn", bits=6, value_bits=5,
                tail_tokens=None)
    assert spec_engine._stamped_spec_params(kv) == dict(
        scheme="kvarn", kv_bits=6, value_bits=5, tail_tokens=1024)
    off = _stamp(_FakeLM(), scheme="uniform", bits=None, group_size=64)
    assert spec_engine._stamped_spec_params(off) == {}
    aff = _stamp(_FakeLM(), scheme="uniform", bits=8, group_size=32)
    assert spec_engine._stamped_spec_params(aff) == dict(
        scheme="uniform", kv_bits=8, kv_group_size=32)


def test_stamped_params_keep_a_declined_stamp_fp16():
    # A dropped stamp still carries its requested width (a qat id under
    # affine, an MLA model under kvarn); the B=1 spec cache must not
    # re-arm what the load declined.
    for scheme in ("uniform", "kvarn"):
        dropped = _stamp(_FakeLM(), verdict="dropped", scheme=scheme, bits=8,
                         group_size=64, value_bits=None, tail_tokens=None)
        assert spec_engine._stamped_spec_params(dropped) == {}
    err = _stamp(_FakeLM(), verdict="error", scheme="uniform", bits=8,
                 group_size=64)
    assert spec_engine._stamped_spec_params(err) == {}


def test_stamped_params_honor_a_zero_tail():
    # tail 0 disables the fp16 tail; it is not the default's absence.
    kv = _stamp(_FakeLM(), scheme="kvarn", bits=6, value_bits=None,
                tail_tokens=0)
    assert spec_engine._stamped_spec_params(kv) == dict(
        scheme="kvarn", kv_bits=6, value_bits=6, tail_tokens=0)


# -- serve B=1 conversion ----------------------------------------------------


@needs_kvarn_ops
def test_b1_mtp_kvarn_converts(restorable, kvarn_ops_ok, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="gmlx.spec.engine")
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    spec_engine.install_spec_kv_quant()
    caches = _mk()
    # The shared carve-out holds the last layer of a deep stack fp16.
    assert type(caches[0]) is KVarNKVCache
    assert type(caches[2]) is KVCache
    assert isinstance(caches[1], _SSMCache)
    assert caches[0].k_bits == 6 and caches[0].tail_cap == 1024
    assert not hasattr(caches[0], "bits")
    assert "[kv] MTP spec path" in caplog.text


@needs_kvarn_ops
def test_b1_mtp_kvarn_env_widths(restorable, kvarn_ops_ok):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    restorable.setenv("KV_BITS", "4")
    restorable.setenv("KV_TAIL_TOKENS", "256")
    spec_engine.install_spec_kv_quant()
    caches = _mk()
    assert caches[0].k_bits == 4 and caches[0].v_bits == 4
    assert caches[0].tail_cap == 256


@needs_kvarn_ops
def test_b1_mtp_kvarn_from_the_stamp(restorable, kvarn_ops_ok):
    # Boot env says fp16; this model was loaded at kvarn k4 tail 256.
    restorable.delenv("KV_QUANT_SCHEME", raising=False)
    spec_engine.install_spec_kv_quant()
    lm = _stamp(_FakeLM(), scheme="kvarn", bits=4, value_bits=None,
                tail_tokens=256)
    caches = _mk(lm=lm)
    assert type(caches[0]) is KVarNKVCache
    assert caches[0].k_bits == 4 and caches[0].tail_cap == 256
    # and the reverse: boot env kvarn, stamp says fp16
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    off = _stamp(_FakeLM(), scheme="uniform", bits=None, group_size=64)
    assert all(type(c) is not KVarNKVCache for c in _mk(lm=off))


def test_readback_target_declines(restorable, kvarn_ops_ok):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    spec_engine.install_spec_kv_quant()
    caches = _mk(lm=_ReadbackLM())
    assert all(type(c) is not KVarNKVCache for c in caches)


@needs_kvarn_ops
def test_qwen35_arch_converts(restorable, kvarn_ops_ok):
    # The dispatch arm lifted the qwen3.5 bypass: the arch converts like
    # any other 128-dim stack (recurrent layers stay untouched).
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    spec_engine.install_spec_kv_quant()
    caches = _mk(lm=_FakeLM(model_type="qwen3_5"))
    assert sum(type(c) is KVarNKVCache for c in caches) == 1
    assert type(caches[1]) is _SSMCache


def test_batch_passthrough(restorable, kvarn_ops_ok):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    spec_engine.install_spec_kv_quant()
    sentinel = ["stock"]
    out = _mk(batch_size=2, make_cache=lambda lm, lp: sentinel)
    assert out is sentinel


def test_rotating_stack_declines(restorable, kvarn_ops_ok):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    spec_engine.install_spec_kv_quant()

    class _RotatingLM(_FakeLM):
        def make_cache(self):
            return [RotatingKVCache(max_size=64), KVCache()]

    caches = _mk(lm=_RotatingLM())
    assert all(type(c) is not KVarNKVCache for c in caches)


def test_reads_kv_back_detection():
    class _Owned(_ReadbackLM):
        def speculative_verify_hidden(self, x, pc):
            return x, {}

    assert spec_engine._mtp_reads_kv_back(_ReadbackLM())
    assert not spec_engine._mtp_reads_kv_back(_Owned())
    assert not spec_engine._mtp_reads_kv_back(_FakeLM())


def test_spec_probes_read_through_the_serve_wrapper(monkeypatch):
    # serve hands the spec cache builder an MTPTextTarget; the hooks and
    # the rollback live on the language model it wraps
    from types import SimpleNamespace

    class _LM(_ReadbackLM):
        def rollback_speculative_cache(self, caches, gdn_states, accepted,
                                       block_size):
            return accepted

    inner = _LM()
    wrapper = SimpleNamespace(language_model=inner,
                              config={"model_type": "qwen3_5"})
    assert spec_engine._mtp_reads_kv_back(wrapper)
    spec_engine._harden_spec_target(wrapper)
    assert getattr(inner.rollback_speculative_cache, "_gmlx_kvarn_guard",
                   False)
    monkeypatch.setenv("GMLX_QWEN_OWNED", "0")
    assert "stock fallback" in spec_engine.mtp_kv_decline(wrapper)


# -- shared-KV readback guard ------------------------------------------------


def test_shared_kv_readback_guard():
    from gmlx.spec.helpers import _mtp_shared_kv_from_prompt_cache

    class _Layer:
        layer_type = "full_attention"

    class _Inner:
        layers = [_Layer(), _Layer()]

    class _LM:
        model = _Inner()

    pc = [KVarNKVCache(), KVarNKVCache()]
    assert _mtp_shared_kv_from_prompt_cache(_LM(), pc) == {}


# -- CLI/MTP plumbing --------------------------------------------------------


def test_mtp_run_flags_keep_kvarn():
    from gmlx.commands.cli import mtp_dropped_run_flags

    ns = argparse.Namespace(
        stop=None,
        logit_bias=None,
        repetition_penalty=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        xtc_probability=0.0,
        quantized_kv_start=0,
        max_kv_size=None,
        over_generation=0,
        inject_critique=None,
        thinking_budget=None,
        prefill_step_size=None,
        kv_quant_scheme="kvarn",
    )
    assert mtp_dropped_run_flags(ns) == []


def test_mtp_setup_declines_shared_kv_drafter(kvarn_ops_ok, capsys):
    from gmlx.gen.generation import setup_kvarn_mtp_cache

    class _SharedDrafter:
        pass  # no uses_shared_kv attr: conservative default True

    assert setup_kvarn_mtp_cache(_FakeLM(), _SharedDrafter(), None, 1024, 2) is None
    assert "reads the target KV back" in capsys.readouterr().err


@needs_kvarn_ops
def test_mtp_setup_builds_and_warns_wide_block(kvarn_ops_ok, capsys):
    from gmlx.gen.generation import setup_kvarn_mtp_cache

    pc = setup_kvarn_mtp_cache(_FakeLM(), _Drafter(), None, 1024, 8)
    assert pc is not None
    assert sum(type(c) is KVarNKVCache for c in pc) == 1
    err = capsys.readouterr().err
    assert "[kv] kvarn6 tail=1024 -> quantized 1/3 attn layers" in err
    assert "verify width 9 (block 8 + 1) exceeds the fused kvarn route" in err


def test_mtp_setup_declines_a_window_stack(kvarn_ops_ok, capsys):
    # serve's rule: a sliding-window layer declines the whole MTP stack.
    from gmlx.gen.generation import setup_kvarn_mtp_cache

    class _WindowLM(_FakeLM):
        def make_cache(self):
            return [RotatingKVCache(max_size=64), KVCache()]

    assert setup_kvarn_mtp_cache(_WindowLM(), _Drafter(), None, 1024, 2) is None
    assert "sliding-window cache stack cannot quantize" in capsys.readouterr().err


# -- rollback contracts ------------------------------------------------------


@needs_kvarn_ops
def test_hy3_rollback_with_kvarn_leaves():
    from gmlx.models.hy_v3.mtp import HyV3SpecLM

    caches = [filled(130, tail=256, seed=s) for s in (0, 1)]
    HyV3SpecLM.rollback_speculative_cache(None, caches, None, 1, 4)
    assert all(c.offset == 128 for c in caches)


@needs_kvarn_ops
def test_hy3_rollback_consults_probe_before_mutating():
    from gmlx.models.hy_v3.mtp import HyV3SpecLM

    class _ProbeStuck:
        def is_trimmable(self):
            return True

        def _can_trim(self, n):
            return False

        def trim(self, n):
            pytest.fail("refused probe must block the mutation phase")

    live = filled(130, tail=256)
    with pytest.raises(RuntimeError, match="untrimmable"):
        HyV3SpecLM.rollback_speculative_cache(None, [live, _ProbeStuck()], None, 1, 4)
    assert live.offset == 130  # two-phase: nothing mutated


class _Refuser:
    def __init__(self, limit):
        self.limit = limit

    def is_trimmable(self):
        return True

    def _can_trim(self, n):
        return n <= self.limit

    def trim(self, n):
        pytest.fail("refused pre-check must block the stock body")


def test_rollback_guard_pre_checks_trim():
    from gmlx.gen.generation import harden_mtp_rollback

    calls = []

    class _LM:
        def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
            calls.append((accepted, block_size))
            return accepted

    lm = _LM()
    harden_mtp_rollback(lm)
    wrapped = lm.rollback_speculative_cache
    harden_mtp_rollback(lm)
    assert lm.rollback_speculative_cache is wrapped  # idempotent

    assert wrapped([_Refuser(1), None], None, 2, 4) == 2  # trim 1: fits
    assert calls == [(2, 4)]
    with pytest.raises(RuntimeError, match="refuse trim"):
        wrapped([_Refuser(1)], None, 0, 4)  # trim 3: refused, stock not run
    assert calls == [(2, 4)]
    # batched accepted: the guard checks the max-row trim, like stock
    wrapped([_Refuser(1)], None, mx.array([0, 2]), 4)
    assert len(calls) == 2


@needs_kvarn_ops
def test_rollback_guard_installed_by_mtp_setup(kvarn_ops_ok):
    from gmlx.gen.generation import setup_kvarn_mtp_cache

    class _RollbackLM(_FakeLM):
        def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
            return accepted

    lm = _RollbackLM()
    pc = setup_kvarn_mtp_cache(lm, _Drafter(), None, 1024, 2)
    assert pc is not None
    assert getattr(lm.rollback_speculative_cache, "_gmlx_kvarn_guard", False)


# -- R1: reject cycles straddling seal boundaries ----------------------------


@needs_kvarn_ops
def test_mtp_reject_cycles_straddle_seals():
    # The R1 invariant: rejected draft KV must leave zero trace even when a
    # draft block straddles a 128-token seal boundary (eager seal quantizes
    # tokens the verify then rejects; trim must reopen via the horizon).
    # Every rejection count 1..3 crosses boundaries at every phase over the
    # walk, and each round's trim must report exactly what was asked.
    rng = np.random.default_rng(11)
    true_k, true_v = tokens(700, seed=11)
    a = KVarNKVCache(tail_tokens=256)
    pos = 0
    rounds = 0
    while pos < 560:
        r = rounds % 4
        keep = 4 - r
        gk = mx.array(rng.standard_normal((1, H, r, D)).astype(np.float16))
        gv = mx.array(rng.standard_normal((1, H, r, D)).astype(np.float16))
        blk_k = mx.concatenate([true_k[:, :, pos : pos + keep], gk], axis=2)
        blk_v = mx.concatenate([true_v[:, :, pos : pos + keep], gv], axis=2)
        a.update_and_fetch(blk_k, blk_v)
        if r:
            assert a.trim(r) == r, f"round {rounds}: trim({r}) refused"
        pos += keep
        rounds += 1
        assert a.offset == pos
    ref = KVarNKVCache(tail_tokens=256)
    ref.update_and_fetch(true_k[:, :, :pos], true_v[:, :, :pos])
    assert (a.offset, a.n_sealed, a.live_len) == (
        ref.offset,
        ref.n_sealed,
        ref.live_len,
    )
    for x, y in zip(a.materialize(), ref.materialize(), strict=True):
        assert np.array_equal(np.array(x), np.array(y))


# -- preemption lift ---------------------------------------------------------


@needs_kvarn_ops
def test_kvarn_lift_cache_recovers_original_domain():
    """MTP preemption lifts a B=1 kvarn cache into a one-row BatchKVCache.
    materialize() returns rotated K/V; feeding that to stock SDPA attends
    rotated keys with an un-rotated query -- no crash, just wrong logits.
    The lift must use the original-domain accessor."""
    from mlx_vlm.models.cache import BatchKVCache

    k, v = tokens(300, seed=7)
    c = KVarNKVCache(tail_tokens=256)
    c.update_and_fetch(k, v)
    c._gmlx_cascade = "stamp"

    lifted = spec_engine.kvarn_lift_cache(c)
    assert type(lifted) is BatchKVCache
    assert lifted.offset == 300
    assert lifted._gmlx_cascade == "stamp"
    assert hasattr(lifted, "filter") and hasattr(lifted, "extend")

    got_k = lifted.keys[..., :300, :].astype(mx.float32)
    # Original domain, not the rotated one the SDPA route reads.
    err = mx.abs(got_k - k.astype(mx.float32)).max().item()
    assert err < 0.2, err
    rot = kq.kvarn_rotate(k).astype(mx.float32)
    rot_err = mx.abs(got_k - rot).max().item()
    assert rot_err > err, (err, rot_err)
    # The verbatim tail rows survive the round trip exactly.
    tail = mx.abs(got_k[:, :, -256:] - k[:, :, -256:].astype(mx.float32))
    assert tail.max().item() == 0.0


@needs_kvarn_ops
def test_kvarn_lift_matches_stock_attention():
    """The decisive check: one attention step over the lifted cache must
    agree with the same step over the un-preempted fp16 history."""
    k, v = tokens(260, seed=11)
    c = KVarNKVCache(tail_tokens=256)
    c.update_and_fetch(k, v)
    lifted = spec_engine.kvarn_lift_cache(c)

    q = mx.random.normal((1, H, 1, D)).astype(mx.float16)
    scale = D ** -0.5

    def attend(keys, values):
        s = (q.astype(mx.float32) @ keys.astype(mx.float32).transpose(
            0, 1, 3, 2)) * scale
        return mx.softmax(s, axis=-1) @ values.astype(mx.float32)

    ref = attend(k, v)
    got = attend(lifted.keys[..., :260, :], lifted.values[..., :260, :])
    assert mx.abs(got - ref).max().item() < 5e-3


def test_preemption_gate_admits_what_the_lift_handles():
    """The gate and the lift must agree. A kvarn cache the lift knows how
    to promote but the gate refuses makes MTP preemption dead code: the
    queued request waits for the live generation instead of joining it."""
    from mlx_vlm.models.cache import KVCache, QuantizedKVCache

    from gmlx.cache.kvarn_cache import KVarNRotatingKVCache

    assert spec_engine.batch_liftable(KVarNKVCache(tail_tokens=256))
    # offset counts evicted tokens the rotating buffers no longer hold
    assert not spec_engine.batch_liftable(KVarNRotatingKVCache(2048, tail_tokens=256))
    assert spec_engine.batch_liftable(KVCache())
    assert spec_engine.batch_liftable(QuantizedKVCache(group_size=64, bits=8))

    class _Opaque:
        pass

    assert not spec_engine.batch_liftable(_Opaque())


@needs_kvarn_ops
def test_injected_kvarn_row_lifts_to_fp16_batch():
    """A B=1 kvarn row admitted into a live MTP batch lifts through the
    original-domain recovery, not the class merge KVarNKVCache lacks
    (27B speculative stress: every concurrent admission raised)."""
    from mlx_vlm.models.cache import BatchKVCache

    from gmlx.spec.speculative import _lift_injected_cache, _lift_live_cache

    k, v = tokens(300, seed=3)
    base = KVCache()
    base.update_and_fetch(k, v)
    live = BatchKVCache.merge([base])
    c = filled(300, tail=256, seed=9)
    lifted = _lift_injected_cache(live, c)
    assert type(lifted) is BatchKVCache
    live.extend(lifted)
    assert live.keys.shape[0] == 2
    assert live.offset.tolist() == [300, 300]
    k9, _ = tokens(300, seed=9)
    got = live.keys[1:, :, :300, :].astype(mx.float32)
    assert mx.abs(got - k9.astype(mx.float32)).max().item() < 0.2
    assert type(_lift_live_cache(filled(64, tail=256))) is BatchKVCache
