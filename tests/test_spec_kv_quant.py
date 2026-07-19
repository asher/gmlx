"""B=1 MTP KV_BITS: construction conversion, trim rollback, quantized verify."""

import os

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.generate.ar")
q35l = pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_lm.models.cache import KVCache, QuantizedKVCache  # noqa: E402
from mlx_vlm.generate import ar  # noqa: E402
from mlx_vlm.server import generation as gen  # noqa: E402
from mlx_vlm.speculative import utils as su  # noqa: E402

from gmlx import qwen35_verify_fold, spec_engine  # noqa: E402


class _SSMCache:
    """ArraysCache stand-in: not a KVCache, no to_quantized."""

    def is_trimmable(self):
        return False


class _FakeLM:
    def make_cache(self):
        return [KVCache(), _SSMCache(), KVCache()]


@pytest.fixture
def restorable(monkeypatch):
    # Identity-setattr records the current attrs so the install's direct
    # module assignment is undone at teardown.
    for mod in (su, ar, gen):
        monkeypatch.setattr(
            mod, "make_speculative_prompt_cache",
            mod.make_speculative_prompt_cache)
    return monkeypatch


def _mk(batch_size=1, make_cache=None):
    return ar.make_speculative_prompt_cache(
        _FakeLM(),
        draft_kind="mtp",
        batch_size=batch_size,
        left_padding=[0] * batch_size,
        make_cache=make_cache or (lambda lm, lp: pytest.fail(
            "B=1 mtp bypass must not call make_cache")),
    )


def test_b1_mtp_converts(restorable):
    restorable.setenv("KV_BITS", "4")
    spec_engine.install_spec_kv_quant()
    caches = _mk()
    assert isinstance(caches[0], QuantizedKVCache)
    assert isinstance(caches[2], QuantizedKVCache)
    assert isinstance(caches[1], _SSMCache)
    assert caches[0].bits == 4 and caches[0].group_size == 64
    assert caches[0].offset == 0 and caches[0].is_trimmable()
    # idempotent: second install keeps the same wrapper
    wrapped = su.make_speculative_prompt_cache
    spec_engine.install_spec_kv_quant()
    assert su.make_speculative_prompt_cache is wrapped


def test_group_size_env(restorable):
    restorable.setenv("KV_BITS", "8")
    restorable.setenv("KV_GROUP_SIZE", "32")
    spec_engine.install_spec_kv_quant()
    caches = _mk()
    assert caches[0].bits == 8 and caches[0].group_size == 32


def test_no_env_no_patch(restorable):
    restorable.delenv("KV_BITS", raising=False)
    before = su.make_speculative_prompt_cache
    spec_engine.install_spec_kv_quant()
    assert su.make_speculative_prompt_cache is before


def test_kill_switch(restorable):
    restorable.setenv("KV_BITS", "4")
    restorable.setenv("GMLX_SPEC_KV_QUANT", "0")
    before = su.make_speculative_prompt_cache
    spec_engine.install_spec_kv_quant()
    assert su.make_speculative_prompt_cache is before


@pytest.mark.parametrize(
    "bits,scheme", [("4", "turboquant"), ("1.6", "uniform")]
)
def test_non_affine_stays_fp16(restorable, bits, scheme):
    restorable.setenv("KV_BITS", bits)
    restorable.setenv("KV_QUANT_SCHEME", scheme)
    before = su.make_speculative_prompt_cache
    spec_engine.install_spec_kv_quant()
    assert su.make_speculative_prompt_cache is before


def test_batch_passthrough(restorable):
    restorable.setenv("KV_BITS", "4")
    spec_engine.install_spec_kv_quant()
    sentinel = ["stock"]
    out = _mk(batch_size=2, make_cache=lambda lm, lp: sentinel)
    assert out is sentinel


def _fill(c, parts):
    for p in parts:
        c.update_and_fetch(p, p)


def test_quantized_trim_rollback_exact():
    # The rollback premise: suffix trim on packed KV is a pure offset move
    # (packing is per-token along head_dim), so trim + re-append lands
    # bit-identically to a straight-line fill.
    mx.random.seed(7)
    k1 = mx.random.normal((1, 2, 40, 64)).astype(mx.bfloat16)
    blk = mx.random.normal((1, 2, 4, 64)).astype(mx.bfloat16)
    k3 = mx.random.normal((1, 2, 3, 64)).astype(mx.bfloat16)
    a = QuantizedKVCache(group_size=64, bits=4)
    _fill(a, [k1, blk])
    assert a.trim(3) == 3 and a.offset == 41
    a.update_and_fetch(k3, k3)
    assert a.offset == 44
    b = QuantizedKVCache(group_size=64, bits=4)
    _fill(b, [k1, blk[:, :, :1, :], k3])
    for xa, xb in zip(a.state, b.state):
        for pa, pb in zip(xa, xb):
            assert mx.array_equal(pa, pb).item()


def _dequant_ref(q, qc, scale):
    # upstream per-token verify loop on dequantized KV
    keys = mx.dequantize(*qc.keys, group_size=qc.group_size, bits=qc.bits)
    values = mx.dequantize(*qc.values, group_size=qc.group_size, bits=qc.bits)
    keys = keys[..., : qc.offset, :].astype(q.dtype)
    values = values[..., : qc.offset, :].astype(q.dtype)
    L = q.shape[2]
    prefix = qc.offset - L
    return mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + 1, :],
                keys[:, :, : prefix + i + 1, :],
                values[:, :, : prefix + i + 1, :],
                scale=scale,
                mask=None,
            )
            for i in range(L)
        ],
        axis=2,
    )


def test_fold_claims_quantized_b1():
    assert qwen35_verify_fold.install_qwen35_verify_fold()
    fn = q35l._target_verify_left_padded_attention
    mx.random.seed(3)
    scale = 64**-0.5
    qc = QuantizedKVCache(group_size=64, bits=8)
    prefix = mx.random.normal((1, 2, 512, 64)).astype(mx.bfloat16)
    qc.update_and_fetch(prefix, prefix)
    blk = mx.random.normal((1, 2, 4, 64)).astype(mx.bfloat16)
    keys, values = qc.update_and_fetch(blk, blk)
    q = mx.random.normal((1, 8, 4, 64)).astype(mx.bfloat16)
    # ref FIRST: upstream quantized SDPA scales its queries arg in place
    # (mlx-lm `queries *= scale`), so q is unusable after the fold call.
    ref = _dequant_ref(q, qc, scale)
    mx.eval(ref)
    out = fn(q, keys, values, cache=qc, scale=scale, mask="causal")
    assert out is not None and out.shape == ref.shape
    err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    # The CPU backend's quantized SDPA accumulates in a different order:
    # 0.027 max err on this seed vs <=0.02 on Metal. Same quantization, both
    # within q8 noise; keep the Metal bound tight.
    tol = 3e-2 if os.environ.get("KQUANT_FORCE_CPU") else 2e-2
    assert err < tol, f"quantized fold err={err}"


def test_batch_sdpa_tuple_defers():
    assert qwen35_verify_fold.install_qwen35_verify_fold()
    fn = q35l.scaled_dot_product_attention
    orig = fn._gmlx_orig
    mx.random.seed(5)
    qc = QuantizedKVCache(group_size=64, bits=8)
    pref = mx.random.normal((2, 2, 128, 64)).astype(mx.bfloat16)
    keys, values = qc.update_and_fetch(pref, pref)
    q = mx.random.normal((2, 8, 1, 64)).astype(mx.bfloat16)
    # fresh copy per call: quantized SDPA scales queries in place upstream
    out = fn(q + 0, keys, values, cache=qc, scale=64**-0.5, mask=None)
    ref = orig(q + 0, keys, values, cache=qc, scale=64**-0.5, mask=None)
    assert mx.array_equal(out, ref).item()
