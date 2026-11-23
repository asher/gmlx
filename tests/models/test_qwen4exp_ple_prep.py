"""The PLE layer's prefill conv in one kernel (ple_prep.conv_silu) against
the op chain it replaces: concat, dilated depthwise conv1d, silu."""
from __future__ import annotations

import dataclasses
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import gmlx.models.qwen4_exp.model as q4
from gmlx.models.qwen4_exp import ple_prep

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the PLE prefill kernel needs a real GPU and a GPU default device",
                allow_module_level=True)

K, DIL = 4, 3
L = (K - 1) * DIL


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(a, b).item())


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("t", [1, 2, 8, 9, 40, 300])
def test_conv_silu_has_the_bits_of_the_op_chain(t, dtype):
    C = 256
    rng = np.random.default_rng(t)
    state = mx.array(rng.standard_normal((1, L, C)).astype(np.float32)).astype(dtype)
    rows = mx.array(rng.standard_normal((1, t, C)).astype(np.float32)).astype(dtype)
    conv = nn.Conv1d(C, C, kernel_size=K, dilation=DIL, groups=C, bias=False)
    conv.weight = mx.array((rng.standard_normal(conv.weight.shape) * 0.4)
                           .astype(np.float32)).astype(dtype)
    assert ple_prep.conv_supported(state, rows, conv.weight, DIL)
    want = nn.silu(conv(mx.concatenate([state, rows], axis=1)))
    got = ple_prep.conv_silu(state, rows, conv.weight, DIL)
    assert _same(got, want)


def test_shapes_the_conv_kernel_does_not_take():
    C = 64
    state = mx.zeros((1, L, C), dtype=mx.bfloat16)
    rows = mx.zeros((1, 5, C), dtype=mx.bfloat16)
    w = mx.zeros((C, K, 1), dtype=mx.bfloat16)
    assert ple_prep.conv_supported(state, rows, w, DIL)
    assert not ple_prep.conv_supported(state, rows, w.astype(mx.float32), DIL)
    assert not ple_prep.conv_supported(state[:, 1:], rows, w, DIL)
    two = lambda a: mx.concatenate([a, a], axis=0)
    assert not ple_prep.conv_supported(two(state), two(rows), w, DIL)


def _layer():
    heads = 16
    kw = dict(
        hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=1, head_dim=16, hc_count=4,
        ple_ngram_size=3, ple_heads_per_ngram=8, ple_eos_token_id=0,
        ple_embed_dim=8, ple_conv_kernel=K,
        ple_layer_multipliers=[23703573157769, 20109073645365, 8052911324071],
        ple_head_vocab_sizes=[101] * heads,
        ple_head_offsets=[101 * i for i in range(heads)],
        layer_types=["linear_attention"])
    flds = {f.name for f in dataclasses.fields(q4.ModelArgs)}
    args = q4.ModelArgs(**{k: v for k, v in kw.items() if k in flds})
    mx.random.seed(4)
    layer = q4.PLELayer(args)
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    table = nn.Embedding(101 * heads, 8)
    table.set_dtype(mx.bfloat16)
    return layer, table


@pytest.mark.parametrize("rows", [5, 40])
def test_layer_prefill_takes_the_conv_kernel_with_the_same_bits(monkeypatch, rows):
    """A prefill of rows fewer and more than the conv state takes the kernel,
    and GMLX_Q4_PLE_PREFILL_FUSE=0 gives the same output and conv state."""
    layer, table = _layer()
    h = (mx.random.normal((1, rows, 4, 64), key=mx.random.key(1)) * 0.5
         ).astype(mx.bfloat16)
    ids = mx.random.randint(1, 5000, (1, rows), key=mx.random.key(2))
    calls = []
    real = ple_prep.conv_silu
    monkeypatch.setattr(ple_prep, "conv_silu",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    runs = []
    for off in (False, True):
        if off:
            monkeypatch.setenv("GMLX_Q4_PLE_PREFILL_FUSE", "0")
        else:
            monkeypatch.delenv("GMLX_Q4_PLE_PREFILL_FUSE", raising=False)
        cache = [None, None, mx.random.normal((1, L, 256), key=mx.random.key(3)
                                              ).astype(mx.bfloat16), None]
        out = layer(h, ids, cache, table)
        mx.eval(out, cache[2])
        runs.append((out, cache[2]))
    assert len(calls) == 1
    assert _same(runs[0][0], runs[1][0])
    assert _same(runs[0][1], runs[1][1])
