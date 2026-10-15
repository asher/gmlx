"""fused_min_rows matches what the installed MLX runs fused.

A prefill chunk the table calls fused must not allocate a score tensor,
and a chunk one row narrower than a head dim's fused width must. An MLX
that changes either rule fails here before depth decay misprices it.
"""

import os

import mlx.core as mx
import pytest

import gmlx.gen.prefill_decay as pd

pytestmark = pytest.mark.skipif(
    os.environ.get("KQUANT_FORCE_CPU") == "1" or not mx.metal.is_available(),
    reason="fused attention dispatch is a Metal path")

HEADS, KV_HEADS = 8, 2


def _score_bytes(dim, q_len, k_len, mask):
    """Peak bytes one prefill chunk allocates beyond its inputs."""
    mx.random.seed(0)
    q = mx.random.normal((1, HEADS, q_len, dim)).astype(mx.bfloat16)
    k = mx.random.normal((1, KV_HEADS, k_len, dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, KV_HEADS, k_len, dim)).astype(mx.bfloat16)
    if mask == "array":
        rows = mx.arange(k_len - q_len, k_len)[:, None]
        mask = rows >= mx.arange(k_len)[None]
    mx.eval(q, k, v, mask if isinstance(mask, mx.array) else q)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=dim ** -0.5, mask=mask)
    mx.eval(out)
    extra = mx.get_peak_memory() - base - out.nbytes
    del q, k, v, out
    mx.clear_cache()
    return extra


def _dense(q_len, k_len):
    return HEADS * q_len * k_len * 2


@pytest.fixture(autouse=True)
def _no_buffer_cache():
    old = mx.set_cache_limit(0)
    yield
    mx.set_cache_limit(old)


@pytest.mark.parametrize("mask", ["causal", "array"])
@pytest.mark.parametrize("k_len", [2048, 16384])
@pytest.mark.parametrize("dim", [64, 72, 80, 96, 128, 256])
def test_fused_width_allocates_no_scores(dim, k_len, mask):
    rows = pd.fused_min_rows(dim)
    if not rows:
        pytest.skip(f"head dim {dim} is not fused on mlx {mx.__version__}")
    for q_len in (rows, 1024, 2048):
        if q_len > k_len:
            continue
        extra = _score_bytes(dim, q_len, k_len, mask)
        assert extra < 0.05 * _dense(q_len, k_len), (dim, q_len, k_len)


@pytest.mark.parametrize("mask", ["causal", "array"])
@pytest.mark.parametrize("dim", [256])
def test_one_row_under_the_fused_width_materializes(dim, mask):
    rows = pd.fused_min_rows(dim)
    if not rows:
        pytest.skip(f"head dim {dim} is not fused on mlx {mx.__version__}")
    q_len, k_len = rows - 1, 16384
    assert _score_bytes(dim, q_len, k_len, mask) > 0.5 * _dense(q_len, k_len)


@pytest.mark.parametrize("dim", [192, 512])
def test_unfused_head_dims_materialize(dim):
    assert pd.fused_min_rows(dim) == 0
    q_len, k_len = 1024, 8192
    assert _score_bytes(dim, q_len, k_len, "causal") > (
        0.5 * _dense(q_len, k_len))


def test_head_dim_256_without_a_mask_materializes():
    # The table covers causal and array masks only; a provider must not arm
    # for an arch whose prefill passes no mask.
    q_len, k_len = 1024, 8192
    assert _score_bytes(256, q_len, k_len, None) > 0.5 * _dense(q_len, k_len)
