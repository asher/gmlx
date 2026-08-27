"""PLE n-gram row hash: fused kernel vs eager op chain."""

from __future__ import annotations

import dataclasses
import os
import re

import mlx.core as mx
import numpy as np
import pytest

import gmlx.qwen4_exp_model as q4
from gmlx.qwen4_exp_model import ModelArgs, PLEEmbedding

_EOS = 248044
_PLE_KW = dict(
    ple_ngram_size=3,
    ple_heads_per_ngram=8,
    ple_eos_token_id=_EOS,
    ple_embed_dim=160,
    ple_layer_multipliers=[23703573157769, 20109073645365, 8052911324071],
    ple_head_vocab_sizes=[
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
        20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
        20000153, 20000159, 20000161, 20000171,
    ],
    ple_head_offsets=[
        0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228,
        140000297, 160000374, 180000455, 200000548, 220000655, 240000802,
        260000955, 280001114, 300001275,
    ],
)


def _real_apple_gpu() -> bool:
    if os.environ.get("KQUANT_FORCE_CPU"):
        return False
    try:
        name = str(mx.device_info().get("device_name", ""))
    except Exception:
        return False
    return bool(re.search(r"Apple M\d", name))


def _ple():
    flds = {f.name for f in dataclasses.fields(ModelArgs)}
    return PLEEmbedding(ModelArgs(
        **{k: v for k, v in _PLE_KW.items() if k in flds}))


def _rows(ple, ids, prev, fused):
    if fused:
        os.environ.pop("GMLX_Q4_PLE_FUSED_HASH", None)
    else:
        os.environ["GMLX_Q4_PLE_FUSED_HASH"] = "0"
    q4._ple_hash_kernel = None
    try:
        out = ple.row_ids(mx.array(ids), None, prev=mx.array(prev))
        mx.eval(out)
    finally:
        os.environ.pop("GMLX_Q4_PLE_FUSED_HASH", None)
        q4._ple_hash_kernel = None
    return np.array(out)


@pytest.mark.skipif(not _real_apple_gpu(),
                    reason="fused PLE hash is a Metal kernel")
@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (3, 1), (2, 7), (2, 63)])
def test_ple_hash_fused_matches_eager(shape):
    B, T = shape
    ple = _ple()
    rng = np.random.default_rng(B * 100 + T)
    ids = rng.integers(0, 248000, size=(B, T))
    ids[rng.random((B, T)) < 0.15] = _EOS
    prev = rng.integers(0, 248000, size=(B, 2))
    prev[rng.random((B, 2)) < 0.3] = _EOS
    eager = _rows(ple, ids.astype(np.int64), prev.astype(np.int64), False)
    fused = _rows(ple, ids.astype(np.int64), prev.astype(np.int64), True)
    assert (eager == fused).all()


def test_ple_hash_gate_off_on_cpu():
    # KQUANT_FORCE_CPU streams run the eager chain; the gate must not
    # hand back a Metal kernel there.
    if _real_apple_gpu():
        pytest.skip("CPU-stream-only check")
    q4._ple_hash_kernel = None
    try:
        assert q4._ple_hash() is None
    finally:
        q4._ple_hash_kernel = None
