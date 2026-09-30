"""qwen3.5 folded verify attention: numerics vs the upstream per-token loop."""

import mlx.core as mx
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm.models.qwen3_5.language")
from gmlx._vendor.mlx_vlm.models.cache import KVCache
from gmlx.models.qwen35 import language as q35

import gmlx.models.qwen35.verify_fold as qwen35_verify_fold


def _rand(hq, hkv, qL, kL, d=64):
    q = mx.random.normal((1, hq, qL, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hkv, kL, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, kL, d)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v


def _loop_ref(q, k, v, scale):
    # the upstream per-token verify loop: qL sliced qL==1 calls, mask=None
    L = q.shape[2]
    prefix = k.shape[2] - L
    return mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + 1, :],
                k[:, :, : prefix + i + 1, :],
                v[:, :, : prefix + i + 1, :],
                scale=scale,
                mask=None,
            )
            for i in range(L)
        ],
        axis=2,
    )


def _install():
    assert qwen35_verify_fold.install_qwen35_verify_fold()
    fn = q35._target_verify_left_padded_attention
    assert hasattr(fn, "_gmlx_orig")
    return fn


@pytest.mark.parametrize("hq,hkv,qL", [(32, 2, 4), (16, 4, 3), (8, 8, 2)])
def test_folded_matches_per_token_loop(hq, hkv, qL):
    fn = _install()
    scale = 64**-0.5
    q, k, v = _rand(hq, hkv, qL, kL=512 + qL)
    out = fn(q, k, v, cache=KVCache(), scale=scale, mask="causal")
    assert out is not None and out.shape == q.shape
    ref = _loop_ref(q, k, v, scale)
    err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    assert err < 2e-2, f"hq/hkv={hq}/{hkv} qL={qL} err={err}"


def test_folded_defers_non_b1_and_decode():
    fn = _install()
    scale = 64**-0.5
    # qL == 1 (decode) must fall through to upstream (None for no padding)
    q, k, v = _rand(8, 2, 1, kL=64)
    assert fn(q, k, v, cache=KVCache(), scale=scale, mask=None) is None
    # quantized-style cache (has .bits) must fall through untouched
    q, k, v = _rand(8, 2, 4, kL=64)

    class _QCache(KVCache):
        bits = 4

    assert fn(q, k, v, cache=_QCache(), scale=scale, mask="causal") is None


def test_install_idempotent_and_killable(monkeypatch):
    fn1 = _install()
    assert qwen35_verify_fold.install_qwen35_verify_fold()
    fn2 = q35._target_verify_left_padded_attention
    assert fn2 is fn1  # no double wrap
    assert fn2._gmlx_orig is fn1._gmlx_orig
    sdpa1 = q35.scaled_dot_product_attention
    assert hasattr(sdpa1, "_gmlx_orig")
    monkeypatch.setenv("GMLX_QWEN35_VERIFY_FOLD", "0")
    monkeypatch.setattr(qwen35_verify_fold, "_installed", False)
    assert not qwen35_verify_fold.install_qwen35_verify_fold()


_GLOBALS = ("scaled_dot_product_attention", "_target_verify_left_padded_attention")


def _fresh_fold(monkeypatch):
    """Module globals without any linked wrap, and the install flag reset.
    Both are restored after the test."""
    for name in _GLOBALS:
        fn = getattr(q35, name)
        while getattr(fn, "_gmlx_orig", None) is not None:
            fn = fn._gmlx_orig
        monkeypatch.setattr(q35, name, fn)
    monkeypatch.setattr(qwen35_verify_fold, "_installed", False)


def _fold_layers(fn):
    n = 0
    while fn is not None:
        n += bool(getattr(fn, qwen35_verify_fold._FLAG, False))
        fn = getattr(fn, "_gmlx_orig", None)
    return n


def test_the_batch_fold_passes_a_kvarn_view_to_the_owned_sdpa(monkeypatch):
    """The fold sits over the owned SDPA, whose KVarN route takes the
    view: the fold must hand it down untouched."""
    from gmlx.cache import kvarn_sdpa as ks
    from gmlx.cache.kvarn_cache import KVarNView

    _fresh_fold(monkeypatch)
    assert qwen35_verify_fold.install_qwen35_verify_fold()
    sdpa = q35.scaled_dot_product_attention
    assert _fold_layers(sdpa) == 1
    claimed = []
    monkeypatch.setattr(ks, "kvarn_attention",
                        lambda q, cache, *a: claimed.append(cache) or q)
    view = KVarNView(object(), "k")
    q = mx.zeros((2, 8, 4, 64), dtype=mx.bfloat16)
    assert sdpa(q, view, view, cache=None, scale=1.0, mask=None) is q
    assert claimed == [view.cache]


def test_the_verify_fold_passes_a_kvarn_view_through(monkeypatch):
    """The verify fold reads keys.ndim on array keys. A KVarN view goes to
    the function under the fold as it came."""
    from gmlx.cache.kvarn_cache import KVarNView

    seen = []
    out = mx.zeros((1,))

    def under(queries, keys, values, *, cache, scale, mask):
        seen.append((keys, values, cache, scale, mask))
        return out

    fold = qwen35_verify_fold._fold_verify(q35, under, None)
    kv, vv = KVarNView(object(), "k"), KVarNView(object(), "v")
    q = mx.zeros((1, 8, 4, 64), dtype=mx.bfloat16)
    assert fold(q, kv, vv, cache=None, scale=0.5, mask="causal") is out
    assert len(seen) == 1 and seen[0][0] is kv and seen[0][1] is vv
    assert seen[0][2:] == (None, 0.5, "causal")


def test_reinstall_under_a_later_wrap_keeps_one_fold(monkeypatch):
    """A test or a reload resets the install flag after another installer
    wrapped the fold. The install took that wrap's original for its own,
    stacked a second fold on it and dropped the wrap."""
    _fresh_fold(monkeypatch)
    assert qwen35_verify_fold.install_qwen35_verify_fold()
    tops = {}
    for name in _GLOBALS:
        cur = getattr(q35, name)

        def later(*args, _cur=cur, **kwargs):
            return _cur(*args, **kwargs)

        later.__dict__["_gmlx_orig"] = cur
        setattr(q35, name, later)
        tops[name] = later
    monkeypatch.setattr(qwen35_verify_fold, "_installed", False)
    assert qwen35_verify_fold.install_qwen35_verify_fold()
    for name, top in tops.items():
        assert getattr(q35, name) is top, name
        assert _fold_layers(top) == 1, name


class _BatchCache:
    """Stand-in for BatchKVCache: left_padding attr, no .bits."""

    def __init__(self, pads):
        self.left_padding = mx.array(pads, dtype=mx.int32)


def _rand_batch(B, hq, hkv, qL, kL, d=64):
    q = mx.random.normal((B, hq, qL, d)).astype(mx.bfloat16)
    k = mx.random.normal((B, hkv, kL, d)).astype(mx.bfloat16)
    v = mx.random.normal((B, hkv, kL, d)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v


def _batch_mask(pads, qL, kL):
    # replicate BatchKVCache.make_mask: left-pad + bottom-right causal
    j = mx.arange(kL)[None, None, None, :]
    i = mx.arange(qL)[None, None, :, None]
    p = mx.array(pads, dtype=mx.int32)[:, None, None, None]
    return (j >= p) & (j <= kL - qL + i)


def test_batch_fold_matches_padded_mask():
    _install()
    fn = q35.scaled_dot_product_attention
    orig = fn._gmlx_orig
    scale = 64**-0.5
    B, qL, kL = 3, 4, 4608
    pads = [0, 512, 1024]
    q, k, v = _rand_batch(B, 8, 2, qL, kL)
    cache = _BatchCache(pads)
    mask = _batch_mask(pads, qL, kL)
    out = fn(q, k, v, cache=cache, scale=scale, mask=mask)
    ref = orig(q, k, v, cache=cache, scale=scale, mask=mask)
    assert out.shape == ref.shape == q.shape
    err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    assert err < 2e-2, f"batch fold err={err}"
    # pads must have been identity-cached after the host sync
    assert getattr(cache, "_gmlx_pads_cache")[1] == pads


def test_batch_fold_eligibility():
    _install()
    fn = q35.scaled_dot_product_attention
    orig = fn._gmlx_orig
    scale = 64**-0.5
    pads = [0, 256]
    cache = _BatchCache(pads)
    # decode (qL == 1) defers: identical to orig on the same args
    q, k, v = _rand_batch(2, 8, 2, 1, 4608)
    mask = _batch_mask(pads, 1, 4608)
    out = fn(q, k, v, cache=cache, scale=scale, mask=mask)
    ref = orig(q, k, v, cache=cache, scale=scale, mask=mask)
    assert mx.array_equal(out, ref).item()
    # shallow KV (< 4096) defers
    q, k, v = _rand_batch(2, 8, 2, 4, 512)
    mask = _batch_mask(pads, 4, 512)
    out = fn(q, k, v, cache=cache, scale=scale, mask=mask)
    ref = orig(q, k, v, cache=cache, scale=scale, mask=mask)
    assert mx.array_equal(out, ref).item()
    # array mask on a cache without left_padding defers (unknown structure)
    q, k, v = _rand_batch(2, 8, 2, 4, 4608)
    full_mask = _batch_mask([0, 0], 4, 4608)
    out = fn(q, k, v, cache=object(), scale=scale, mask=full_mask)
    ref = orig(q, k, v, cache=object(), scale=scale, mask=full_mask)
    assert mx.array_equal(out, ref).item()
