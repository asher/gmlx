"""The owned quantized SDPA: the grouped batch mask, the prefill route and
the decode route.

Stock mlx-lm's quantized SDPA applies a 4D batch mask to 5D grouped
scores as-is: the mask's batch axis lands in the kv-head slot, which
raises when B != n_kv_heads and misapplies the mask silently when they
are equal. The mlx-vlm body the owned function carries aligns the mask,
and every quantized SDPA name resolves to it. The mlx-lm repro runs in a
child that never imports gmlx.

Query widths from GMLX_KV8_PREFILL_FLASH_MIN_L up dequantize the wire
and run fused SDPA; those tests pin numerics against the dequantized
reference and the upstream body, the threshold and the switch. At decode
width, 8-bit K/V in 64-wide groups go to mlx_kquant's sdpa_decode_gqa
when its build takes quantized operands and row starts; those tests pin
the starts a batch cache's make_mask records, the fallbacks and the
switch. Where a route does not apply, the result must equal the upstream
body's exactly.

The upstream body scales ``queries`` in place, so every call here takes
a fresh copy.
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_lm.models import base as lm_base
from mlx_vlm.models import base as vlm_base

import gmlx.cache.attention as owned_attention
import gmlx.cache.kv_cache as owned
from stock_twin import stock_twin

qsdpa = owned_attention.quantized_scaled_dot_product_attention
_stock = stock_twin(qsdpa, origin="mlx_vlm.models.base")


def _case(B=2, nq=8, nkv=4, L=1, kv=64, d=64, pads=(5, 0),
          dtype=mx.float32):
    mx.random.seed(11)
    q = mx.random.normal((B, nq, L, d)).astype(dtype)
    k = mx.random.normal((B, nkv, kv, d)).astype(dtype)
    v = mx.random.normal((B, nkv, kv, d)).astype(dtype)
    qk = mx.quantize(k, group_size=64, bits=8)
    qv = mx.quantize(v, group_size=64, bits=8)
    pos = mx.arange(kv)[None, None, None, :]
    mask = pos >= mx.array(list(pads))[:B, None, None, None]
    mx.eval(q, mask, *qk, *qv)
    return q, qk, qv, mask


def _ref(q, qk, qv, mask, scale=0.125):
    kd = mx.dequantize(*qk, group_size=64, bits=8)
    vd = mx.dequantize(*qv, group_size=64, bits=8)
    return mx.fast.scaled_dot_product_attention(
        mx.array(q), kd, vd, scale=scale, mask=mask)


def _exact(a, b) -> bool:
    return mx.abs(a - b).max().item() == 0.0


# The stock mlx-lm body, in a child with no gmlx on the import path.
_STOCK_LM_CHILD = r'''
import json
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)
from mlx_lm.models.base import quantized_scaled_dot_product_attention as qsdpa

assert not any(m == "gmlx" or m.startswith("gmlx.") for m in sys.modules)


def case(B, nq, nkv, pads):
    mx.random.seed(11)
    q = mx.random.normal((B, nq, 1, 64))
    k = mx.random.normal((B, nkv, 64, 64))
    v = mx.random.normal((B, nkv, 64, 64))
    qk = mx.quantize(k, group_size=64, bits=8)
    qv = mx.quantize(v, group_size=64, bits=8)
    pos = mx.arange(64)[None, None, None, :]
    mask = pos >= mx.array(list(pads))[:B, None, None, None]
    ref = mx.fast.scaled_dot_product_attention(
        mx.array(q), mx.dequantize(*qk, group_size=64, bits=8),
        mx.dequantize(*qv, group_size=64, bits=8), scale=0.125, mask=mask)
    return q, qk, qv, mask, ref


out = {}
q, qk, qv, mask, ref = case(2, 8, 4, (5, 0))
try:
    mx.eval(qsdpa(mx.array(q), qk, qv, 0.125, mask))
    out["raises"] = None
except ValueError as e:
    out["raises"] = "ValueError"
q, qk, qv, mask, ref = case(2, 8, 2, (9, 0))
bad = qsdpa(mx.array(q), qk, qv, 0.125, mask)
out["misapply_err"] = mx.abs(bad - ref).max().item()
print(json.dumps(out))
'''


def _without_gmlx_env():
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    keep = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
            if p and os.path.abspath(p) != root]
    env["PYTHONPATH"] = os.pathsep.join(keep)
    return env


def test_stock_mlx_lm_body_has_both_mask_defects():
    # B != n_kv_heads raises on the 5D/4D broadcast; B == n_kv_heads
    # broadcasts the mask's batch dim into the kv-head slot, silently.
    r = subprocess.run([sys.executable, "-c", _STOCK_LM_CHILD],
                       capture_output=True, text=True, timeout=300,
                       env=_without_gmlx_env(), cwd="/")
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["raises"] == "ValueError"
    assert got["misapply_err"] > 0.05


def test_every_quantized_sdpa_name_is_the_owned_function():
    assert (lm_base.quantized_scaled_dot_product_attention
            is vlm_base.quantized_scaled_dot_product_attention is qsdpa)
    assert _stock is not qsdpa


@pytest.mark.parametrize("B,nkv,pads", [(2, 4, (5, 0)), (2, 2, (9, 0)),
                                        (3, 2, (7, 2, 0))],
                         ids=["raise", "misapply", "three-rows"])
def test_owned_body_has_neither_mask_defect(B, nkv, pads):
    q, qk, qv, mask = _case(B=B, nq=8, nkv=nkv, pads=pads)
    got = qsdpa(mx.array(q), qk, qv, 0.125, mask)
    err = mx.abs(got - _ref(q, qk, qv, mask)).max().item()
    assert err < 2e-2, f"B={B} nkv={nkv} err={err}"


def test_ungrouped_and_b1_calls_run_the_upstream_body(monkeypatch):
    monkeypatch.setenv("GMLX_QSDPA_KQ", "0")
    for B, nq, nkv, pads in ((2, 2, 2, (5, 0)), (1, 8, 4, (3,))):
        q, qk, qv, mask = _case(B=B, nq=nq, nkv=nkv, pads=pads)
        got = qsdpa(mx.array(q), qk, qv, 0.125, mask)
        assert _exact(got, _stock(mx.array(q), qk, qv, 0.125, mask))


def test_prefill_route_matches_the_dequantized_reference():
    # The route runs the reference's own ops, so it matches exactly; the
    # upstream body does not.
    q, qk, qv, mask = _case(B=2, nq=8, nkv=4, L=64, kv=256, pads=(5, 0))
    got = qsdpa(mx.array(q), qk, qv, 0.125, mask)
    assert _exact(got, _ref(q, qk, qv, mask))
    stock = _stock(mx.array(q), qk, qv, 0.125, mask)
    assert not _exact(stock, _ref(q, qk, qv, mask))


def test_prefill_route_matches_the_upstream_numerics():
    q, qk, qv, mask = _case(B=2, nq=8, nkv=4, L=32, kv=256, pads=(5, 0))
    got = qsdpa(mx.array(q), qk, qv, 0.125, mask)
    stock = _stock(mx.array(q), qk, qv, 0.125, mask)
    err = mx.abs(got - stock).max().item()
    assert err < 2e-2, f"prefill route vs upstream body err={err}"


def test_prefill_route_takes_a_causal_string_mask():
    q, qk, qv, _ = _case(B=1, nq=8, nkv=4, L=16, kv=64)
    got = qsdpa(mx.array(q), qk, qv, 0.125, "causal")
    stock = _stock(mx.array(q), qk, qv, 0.125, "causal")
    err = mx.abs(got - stock).max().item()
    assert err < 2e-2, f"causal prefill route vs upstream body err={err}"


def test_prefill_route_threshold_and_switch(monkeypatch):
    # Below the threshold: the upstream body, exactly.
    q, qk, qv, mask = _case(B=1, nq=8, nkv=4, L=4, kv=64, pads=(3,))
    got = qsdpa(mx.array(q), qk, qv, 0.125, mask)
    assert _exact(got, _stock(mx.array(q), qk, qv, 0.125, mask))
    q, qk, qv, mask = _case(B=1, nq=8, nkv=4, L=64, kv=256, pads=(3,))
    assert not _exact(qsdpa(mx.array(q), qk, qv, 0.125, mask),
                      _stock(mx.array(q), qk, qv, 0.125, mask))
    # A raised threshold, then the switch: the upstream body again.
    monkeypatch.setenv("GMLX_KV8_PREFILL_FLASH_MIN_L", "128")
    got = qsdpa(mx.array(q), qk, qv, 0.125, mask)
    assert _exact(got, _stock(mx.array(q), qk, qv, 0.125, mask))
    monkeypatch.delenv("GMLX_KV8_PREFILL_FLASH_MIN_L")
    monkeypatch.setenv("GMLX_KV8_PREFILL_FLASH", "0")
    got = qsdpa(mx.array(q), qk, qv, 0.125, mask)
    assert _exact(got, _stock(mx.array(q), qk, qv, 0.125, mask))


def test_owned_sdpa_reaches_the_prefill_route():
    q, qk, qv, mask = _case(B=2, nq=8, nkv=4, L=64, kv=256, pads=(5, 0))
    cache = SimpleNamespace(bits=8, group_size=64)
    got = owned_attention.scaled_dot_product_attention(
        mx.array(q), qk, qv, cache, 0.125, mask)
    assert _exact(got, _ref(q, qk, qv, mask))


def _strided_case(B=1, nq=8, nkv=4, L=16, kv=63, d=64, dtype=mx.float16,
                  group=64, bits=8):
    """Quantized tuples as the live caches produce them: step-rounded zero
    buffers fetched as lazy slices, strided when kv is not a step
    multiple."""
    mx.random.seed(11)
    q = mx.random.normal((B, nq, L, d)).astype(dtype)
    k = mx.random.normal((B, nkv, kv, d)).astype(dtype)
    v = mx.random.normal((B, nkv, kv, d)).astype(dtype)
    step = 256
    alloc = (kv + step - 1) // step * step

    def store(x):
        w, s, b = mx.quantize(x, group_size=group, bits=bits)
        bufs = tuple(
            mx.zeros((B, nkv, alloc, t.shape[-1]), dtype=t.dtype)
            for t in (w, s, b))
        for buf, t in zip(bufs, (w, s, b)):
            buf[..., :kv, :] = t
        return tuple(buf[..., :kv, :] for buf in bufs)

    qk, qv = store(k), store(v)
    mx.eval(q, *qk, *qv)
    return q, qk, qv


def _cpu_dequant(qt, group=64, bits=8):
    """Ground truth independent of the Metal kernel under test."""
    with mx.stream(mx.cpu):
        return mx.dequantize(*qt, group_size=group, bits=bits)


def _cpu_ref(q, qk, qv, mask, scale=0.125, group=64, bits=8):
    kd = _cpu_dequant(qk, group, bits)
    vd = _cpu_dequant(qv, group, bits)
    return mx.fast.scaled_dot_product_attention(
        mx.array(q), kd, vd, scale=scale, mask=mask)


@pytest.mark.parametrize("group,bits", [(64, 8), (32, 8), (64, 4), (32, 4)])
@pytest.mark.parametrize("kv", [63, 256])
def test_prefill_route_dequantizes_strided_cache_slices(group, bits, kv):
    # kv=256 is the step-aligned (contiguous) control.
    q, qk, qv = _strided_case(L=16, kv=kv, group=group, bits=bits)
    got = qsdpa(mx.array(q), qk, qv, 0.125, "causal", group_size=group,
                bits=bits)
    ref = _cpu_ref(q, qk, qv, "causal", group=group, bits=bits)
    err = mx.abs(got - ref).max().item()
    assert err < 2e-3, f"g={group} b={bits} kv={kv} err={err}"


def test_prefill_route_on_live_batch_cache_slices():
    mx.random.seed(11)
    c = owned.BatchQuantizedKVCache([0], group_size=64, bits=8)
    k = mx.random.normal((1, 4, 63, 64)).astype(mx.float16)
    v = mx.random.normal((1, 4, 63, 64)).astype(mx.float16)
    qk, qv = c.update_and_fetch(k, v)
    assert qk[0].shape[2] == 63  # sliced from the 256-step buffer
    q = mx.random.normal((1, 8, 16, 64)).astype(mx.float16)
    got = qsdpa(mx.array(q), qk, qv, 0.125, "causal")
    err = mx.abs(got - _cpu_ref(q, qk, qv, "causal")).max().item()
    assert err < 2e-3, f"prefill route on batch cache slices err={err}"


# mlx #4381 fixed the encode-order bug in 0.32.2. The canaries are a
# version bump gate: fail-closed in both directions, never delete on
# failure.
_MLX_STRIDED_FIX = (0, 32, 2)


def _mlx_version_tuple():
    parts = []
    for tok in mx.__version__.split(".")[:3]:
        num = "".join(ch for ch in tok if ch.isdigit())
        parts.append(int(num or 0))
    return tuple(parts)


def _operand_buffers(kv=63, alloc=256, group=64, bits=8):
    mx.random.seed(11)
    k = mx.random.normal((1, 4, kv, 64)).astype(mx.float16)
    w, s, b = mx.quantize(k, group_size=group, bits=bits)
    bufs = []
    for t in (w, s, b):
        buf = mx.zeros((1, 4, alloc, t.shape[-1]), dtype=t.dtype)
        buf[..., :kv, :] = t
        bufs.append(buf)
    mx.eval(*bufs)
    return bufs, kv


@pytest.mark.parametrize(
    "strided",
    [(w, s, b) for w in (0, 1) for s in (0, 1) for b in (0, 1)
     if (w, s, b) != (0, 0, 0)],
    ids=lambda t: "w%ds%db%d" % t)
def test_mlx_strided_operand_dequantize_canary(strided):
    # Encode-order bug: the biases contiguity copy was encoded after
    # w and scales were bound, so dequantize read the copy's buffers.
    # Corruption tracks the biases operand exactly. Pre-fix versions
    # must reproduce it and fixed versions must not. On failure
    # re-check the pin; never delete the test.
    # GPU vs CPU dequantize differ ~2e-3 in fp16, so clean is < 1e-2
    # and corrupt is > 1.
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal-only regression")
    bufs, kv = _operand_buffers()
    ops = [buf[..., :kv, :] if st else mx.contiguous(buf[..., :kv, :])
           for buf, st in zip(bufs, strided)]
    got = mx.dequantize(*ops, group_size=64, bits=8)
    ref = _cpu_dequant(tuple(mx.contiguous(buf[..., :kv, :])
                             for buf in bufs))
    err = mx.abs(got.astype(mx.float32)
                 - ref.astype(mx.float32)).max().item()
    broken = _mlx_version_tuple() < _MLX_STRIDED_FIX
    if broken and strided[2]:
        assert err > 1.0, (
            f"strided-biases dequantize correct on mlx {mx.__version__} "
            f"(err={err}): upstream fix backported? re-gate the canary "
            "and re-check the pin")
    else:
        assert err < 1e-2, (
            f"strided-operand dequantize corrupt on mlx {mx.__version__} "
            f"(strided w/s/b={strided}, err={err}): upstream regression")


def test_mlx_strided_input_quantize_canary():
    # mx.quantize was never affected. Pinned on both sides of the
    # fix so KVCache.to_quantized stays certified.
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal-only regression")
    mx.random.seed(11)
    kv = 63
    k = mx.random.normal((1, 4, kv, 64)).astype(mx.float16)
    buf = mx.zeros((1, 4, 256, 64), dtype=mx.float16)
    buf[..., :kv, :] = k
    mx.eval(buf)
    sl = buf[..., :kv, :]
    got = mx.quantize(sl, group_size=64, bits=8)
    ref = mx.quantize(mx.contiguous(sl), group_size=64, bits=8)
    for g, r in zip(got, ref):
        assert mx.array_equal(g, r).item(), (
            f"quantize of a strided input diverged on mlx {mx.__version__}")


def _decode_kernel_ready() -> bool:
    return (mx.default_device().type == mx.DeviceType.gpu
            and owned_attention._kq_decode_kernel() is not None)


needs_kq = pytest.mark.skipif(
    not _decode_kernel_ready(),
    reason="q8 and starts sdpa_decode_gqa unavailable (build or cpu device)")


def _fresh_memo(monkeypatch):
    memo = type(owned_attention._STARTS_MEMO)()
    monkeypatch.setattr(owned_attention, "_STARTS_MEMO", memo)
    return memo


@pytest.fixture
def spy(monkeypatch):
    """Record the starts of each decode kernel call; the real kernel runs."""
    real = owned_attention._kq_decode_kernel()
    calls = []

    def kernel(*a, **kw):
        calls.append(kw.get("starts"))
        return real(*a, **kw)

    monkeypatch.setattr(owned_attention, "_kq_decode_kernel", lambda: kernel)
    _fresh_memo(monkeypatch)
    return calls


@needs_kq
def test_decode_route_takes_a_recorded_left_pad_mask(spy):
    q, qk, qv, mask = _case(B=3, nq=8, nkv=4, kv=512, pads=(37, 5, 0),
                            dtype=mx.float16)
    owned_attention._register_starts(mask, mx.array([37, 5, 0]))
    got = qsdpa(mx.array(q), qk, qv, 0.125, mask)
    assert len(spy) == 1 and spy[0].tolist() == [37, 5, 0]
    err = mx.abs(got - _ref(q, qk, qv, mask)).max().item()
    assert err < 2e-2, f"decode route vs dequantized reference err={err}"


@needs_kq
def test_make_mask_records_the_starts_and_the_call_routes(spy):
    mx.random.seed(11)
    pads = [37, 5, 0]
    c = owned.BatchQuantizedKVCache(pads, group_size=64, bits=8)
    k = mx.random.normal((3, 4, 511, 64)).astype(mx.float16)
    c.update_and_fetch(k, k)
    # Per step, the mask comes first (it counts the new token), then the
    # update.
    mask = c.make_mask(1)
    qk, qv = c.update_and_fetch(
        mx.random.normal((3, 4, 1, 64)).astype(mx.float16),
        mx.random.normal((3, 4, 1, 64)).astype(mx.float16))
    q = mx.random.normal((3, 8, 1, 64)).astype(mx.float16)
    got = owned_attention.scaled_dot_product_attention(
        mx.array(q), qk, qv, c, 0.125, mask)
    assert len(spy) == 1 and spy[0].tolist() == pads
    err = mx.abs(got - _ref(q, qk, qv, mask)).max().item()
    assert err < 2e-2, f"decode route through the cache err={err}"


@needs_kq
def test_decode_route_without_a_mask_and_with_causal(spy):
    q, qk, qv, _ = _case(B=2, nq=8, nkv=4, kv=256, dtype=mx.float16)
    got = qsdpa(mx.array(q), qk, qv, 0.125, None)
    got_c = qsdpa(mx.array(q), qk, qv, 0.125, "causal")
    assert spy == [None, None]
    full = mx.ones((2, 1, 1, 256), dtype=mx.bool_)
    err = mx.abs(got - _ref(q, qk, qv, full)).max().item()
    assert err < 2e-2, f"maskless decode route err={err}"
    assert _exact(got_c, got)


@needs_kq
def test_decode_route_leaves_other_calls_to_the_upstream_body(spy):
    q, qk, qv, mask = _case(B=2, nq=8, nkv=4, kv=256, pads=(9, 0),
                            dtype=mx.float16)
    # A mask no cache recorded.
    got = qsdpa(mx.array(q), qk, qv, 0.125, mask)
    assert _exact(got, _stock(mx.array(q), qk, qv, 0.125, mask))
    # An additive float mask.
    fmask = mx.where(mask, mx.array(0.0, mx.float16),
                     mx.array(-mx.inf, mx.float16))
    got = qsdpa(mx.array(q), qk, qv, 0.125, fmask)
    assert _exact(got, _stock(mx.array(q), qk, qv, 0.125, fmask))
    # float32 queries.
    q32, qk32, qv32, m32 = _case(B=2, nq=8, nkv=4, kv=256, pads=(9, 0))
    got = qsdpa(mx.array(q32), qk32, qv32, 0.125, m32)
    assert _exact(got, _stock(mx.array(q32), qk32, qv32, 0.125, m32))
    assert spy == []


@needs_kq
def test_decode_route_switch(spy, monkeypatch):
    q, qk, qv, mask = _case(B=2, nq=8, nkv=4, kv=256, pads=(9, 0),
                            dtype=mx.float16)
    owned_attention._register_starts(mask, mx.array([9, 0]))
    monkeypatch.setenv("GMLX_QSDPA_KQ", "0")
    got = qsdpa(mx.array(q), qk, qv, 0.125, mask)
    assert spy == []
    assert _exact(got, _stock(mx.array(q), qk, qv, 0.125, mask))
    monkeypatch.delenv("GMLX_QSDPA_KQ")
    qsdpa(mx.array(q), qk, qv, 0.125, mask)
    assert len(spy) == 1


def test_decode_route_declines_on_the_cpu_device(monkeypatch):
    looked = []
    monkeypatch.setattr(owned_attention, "_kq_decode_kernel",
                        lambda: looked.append(1))
    q, qk, qv, _ = _case(B=2, nq=8, nkv=4, kv=256, dtype=mx.float16)
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        qsdpa(mx.array(q), qk, qv, 0.125, None)
    finally:
        mx.set_default_device(prev)
    assert looked == []


def test_the_starts_record_keys_on_the_mask_object(monkeypatch):
    memo = _fresh_memo(monkeypatch)
    pos = mx.arange(64)[None, None, None, :]
    mask = pos >= mx.array([7, 0])[:, None, None, None]
    owned_attention._register_starts(mask, mx.array([7, 0]))
    a = owned_attention._registered_starts(mask)
    b = owned_attention._registered_starts(mask)
    assert a is b and a.tolist() == [7, 0] and a.dtype == mx.int32
    assert owned_attention._registered_geometry(mask) is None
    # Another object with the same contents has no record.
    twin = pos >= mx.array([7, 0])[:, None, None, None]
    assert owned_attention._registered_starts(twin) is None
    for i in range(owned_attention._STARTS_MEMO_CAP + 2):
        m = pos >= mx.array([i, 0])[:, None, None, None]
        owned_attention._register_starts(m, mx.array([i, 0]))
    assert len(memo) == owned_attention._STARTS_MEMO_CAP


def test_make_mask_records_only_plain_decode_masks(monkeypatch):
    _fresh_memo(monkeypatch)
    c = owned.BatchQuantizedKVCache([3, 0], group_size=64, bits=8)
    k = mx.zeros((2, 4, 8, 64), mx.float16)
    c.update_and_fetch(k, k)
    decode = c.make_mask(1)
    starts = owned_attention._registered_starts(decode)
    assert starts.tolist() == [3, 0] and starts.dtype == mx.int32
    for mask in (c.make_mask(4), c.make_mask(1, window_size=4)):
        assert owned_attention._registered_starts(mask) is None
