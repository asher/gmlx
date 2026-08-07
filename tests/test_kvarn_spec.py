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

from gmlx import spec_engine  # noqa: E402
from gmlx.kvarn_cache import KVarNKVCache  # noqa: E402

_NEEDS_GPU = pytest.mark.skipif(
    mx.default_device() != mx.gpu,
    reason="kvarn quantize/rotate dispatch Metal kernels; needs the GPU device",
)

H = 2
D = 128


def _tokens(n, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, H, n, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, H, n, D)).astype(np.float16))
    return k, v


def _filled(n, tail=256, seed=0):
    c = KVarNKVCache(tail_tokens=tail)
    c.update_and_fetch(*_tokens(n, seed))
    return c


class _SSMCache:
    """ArraysCache stand-in: not a KVCache, no to_quantized."""

    def is_trimmable(self):
        return False


class _Args:
    def __init__(self, **kw):
        self.model_type = kw.pop("model_type", "llama")
        self.head_dim = kw.pop("head_dim", 128)
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeLM:
    def __init__(self, **kw):
        self.args = _Args(**kw)

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
def _ops_ok(monkeypatch):
    from gmlx import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    monkeypatch.delenv("GMLX_KVARN_BITS", raising=False)


@pytest.fixture
def restorable(monkeypatch):
    # Identity-setattr records the current attrs so the install's direct
    # module assignment is undone at teardown.
    for mod in (su, ar, gen):
        monkeypatch.setattr(
            mod, "make_speculative_prompt_cache", mod.make_speculative_prompt_cache
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
    assert spec_engine._spec_kv_quant_params() == ("kvarn", None, 1024)


def test_params_kvarn_bits_and_tail(restorable):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    restorable.setenv("KV_BITS", "4")
    restorable.setenv("KV_TAIL_TOKENS", "256")
    assert spec_engine._spec_kv_quant_params() == ("kvarn", 4, 256)


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
    assert spec_engine._spec_kv_quant_params() == ("affine", 8, 64)
    restorable.setenv("KV_QUANT_SCHEME", "turboquant")
    assert spec_engine._spec_kv_quant_params() is None


# -- serve B=1 conversion ----------------------------------------------------


@_NEEDS_GPU
def test_b1_mtp_kvarn_converts(restorable, _ops_ok, capsys):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    spec_engine.install_spec_kv_quant()
    caches = _mk()
    assert type(caches[0]) is KVarNKVCache
    assert type(caches[2]) is KVarNKVCache
    assert isinstance(caches[1], _SSMCache)
    assert caches[0].k_bits == 6 and caches[0].tail_cap == 1024
    assert not hasattr(caches[0], "bits")
    assert "[kv] MTP spec path" in capsys.readouterr().out


@_NEEDS_GPU
def test_b1_mtp_kvarn_env_widths(restorable, _ops_ok):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    restorable.setenv("KV_BITS", "4")
    restorable.setenv("KV_TAIL_TOKENS", "256")
    spec_engine.install_spec_kv_quant()
    caches = _mk()
    assert caches[0].k_bits == 4 and caches[0].v_bits == 4
    assert caches[0].tail_cap == 256


def test_readback_target_declines(restorable, _ops_ok):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    spec_engine.install_spec_kv_quant()
    caches = _mk(lm=_ReadbackLM())
    assert all(type(c) is not KVarNKVCache for c in caches)


def test_bypass_arch_declines(restorable, _ops_ok):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    spec_engine.install_spec_kv_quant()
    caches = _mk(lm=_FakeLM(model_type="qwen3_5"))
    assert all(type(c) is not KVarNKVCache for c in caches)


def test_batch_passthrough(restorable, _ops_ok):
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    spec_engine.install_spec_kv_quant()
    sentinel = ["stock"]
    out = _mk(batch_size=2, make_cache=lambda lm, lp: sentinel)
    assert out is sentinel


def test_rotating_stack_declines(restorable, _ops_ok):
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


# -- shared-KV readback guard ------------------------------------------------


def test_shared_kv_readback_guard():
    from gmlx.spec_helpers import _mtp_shared_kv_from_prompt_cache

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
    from gmlx.cli import mtp_dropped_run_flags

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


def test_mtp_setup_declines_shared_kv_drafter(_ops_ok, capsys):
    from gmlx.generation import setup_kvarn_mtp_cache

    class _SharedDrafter:
        pass  # no uses_shared_kv attr: conservative default True

    assert setup_kvarn_mtp_cache(_FakeLM(), _SharedDrafter(), None, 1024, 2) is None
    assert "reads the target KV back" in capsys.readouterr().err


@_NEEDS_GPU
def test_mtp_setup_builds_and_warns_wide_block(_ops_ok, capsys):
    from gmlx.generation import setup_kvarn_mtp_cache

    pc = setup_kvarn_mtp_cache(_FakeLM(), _Drafter(), None, 1024, 6)
    assert pc is not None
    assert sum(type(c) is KVarNKVCache for c in pc) == 2
    err = capsys.readouterr().err
    assert "[kv] kvarn6 KV cache" in err
    assert "verify width 6 exceeds the fused kvarn route" in err


# -- rollback contracts ------------------------------------------------------


@_NEEDS_GPU
def test_hy3_rollback_with_kvarn_leaves():
    from gmlx.hy_v3_mtp import HyV3SpecLM

    caches = [_filled(130, seed=s) for s in (0, 1)]
    HyV3SpecLM.rollback_speculative_cache(None, caches, None, 1, 4)
    assert all(c.offset == 128 for c in caches)


@_NEEDS_GPU
def test_hy3_rollback_consults_probe_before_mutating():
    from gmlx.hy_v3_mtp import HyV3SpecLM

    class _ProbeStuck:
        def is_trimmable(self):
            return True

        def _can_trim(self, n):
            return False

        def trim(self, n):
            pytest.fail("refused probe must block the mutation phase")

    live = _filled(130)
    with pytest.raises(RuntimeError, match="untrimmable"):
        HyV3SpecLM.rollback_speculative_cache(None, [live, _ProbeStuck()], None, 1, 4)
    assert live.offset == 130  # two-phase: nothing mutated


# -- R1: reject cycles straddling seal boundaries ----------------------------


@_NEEDS_GPU
def test_mtp_reject_cycles_straddle_seals():
    # The R1 invariant: rejected draft KV must leave zero trace even when a
    # draft block straddles a 128-token seal boundary (eager seal quantizes
    # tokens the verify then rejects; trim must reopen via the horizon).
    # Every rejection count 1..3 crosses boundaries at every phase over the
    # walk, and each round's trim must report exactly what was asked.
    rng = np.random.default_rng(11)
    true_k, true_v = _tokens(700, seed=11)
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
