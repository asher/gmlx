"""QSAKVCache base and tail segments: decode appends go to the tail and
leave the base alone, the streams read back equal to one buffer through
appends, rollback trims and folds, the two-segment gather matches an index
into the whole stream, and attention over the segments matches attention
over one buffer. The pooled block keys are split the same way."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from helpers import _real_apple_gpu

import gmlx.models.qwen4_exp.model as q4
from gmlx.models.qwen4_exp.model import Attention, ModelArgs, QSAKVCache


gpu_only = pytest.mark.skipif(
    not _real_apple_gpu(), reason="needs a real Apple GPU")

H, D, DI = 2, 16, 8


class _Ref:
    """The three streams as plain NumPy arrays."""

    def __init__(self):
        self.k = np.zeros((1, H, 0, D), np.float32)
        self.v = np.zeros((1, H, 0, D), np.float32)
        self.ik = np.zeros((1, 0, DI), np.float32)

    def push(self, k, v, ik):
        self.k = np.concatenate([self.k, np.asarray(k)], axis=2)
        self.v = np.concatenate([self.v, np.asarray(v)], axis=2)
        self.ik = np.concatenate([self.ik, np.asarray(ik)], axis=1)

    def trim(self, n):
        self.k, self.v = self.k[:, :, :-n], self.v[:, :, :-n]
        self.ik = self.ik[:, :-n]


def _rows(n, seed):
    mx.random.seed(seed)
    return (mx.random.normal((1, H, n, D)), mx.random.normal((1, H, n, D)),
            mx.random.normal((1, n, DI)))


def _check_full(c, ref):
    k, v = c.kv_full()
    assert c.offset == ref.k.shape[2]
    assert np.array_equal(np.asarray(k), ref.k)
    assert np.array_equal(np.asarray(v), ref.v)


def _check_state(c, ref):
    k, v, ik = c.state
    assert np.array_equal(np.asarray(k), ref.k)
    assert np.array_equal(np.asarray(v), ref.v)
    assert np.array_equal(np.asarray(ik), ref.ik)
    assert c._tl == 0


# (rows, in the tail) per append; negative rows is a rollback trim
_SCRIPT = [37] + [1] * 20 + [-3, 3, 2, 20] + [1] * 9 + [-12, 4, 1, 1]


@pytest.mark.parametrize("cap", [4, 16, 1024])
def test_streams_match_one_buffer(monkeypatch, cap):
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", str(cap))
    c, ref = QSAKVCache(ratio=4), _Ref()
    for i, n in enumerate(_SCRIPT):
        if n < 0:
            assert c.trim(-n) == -n
            ref.trim(-n)
        else:
            k, v, ik = _rows(n, 100 + i)
            in_tail = c.append_qsa(k, v, ik)
            assert in_tail == (i > 0 and n <= min(8, cap))
            ref.push(k, v, ik)
        _check_full(c, ref)
        if i % 7 == 3:
            _check_state(c, ref)
    _check_state(c, ref)


def test_tail_zero_keeps_one_buffer(monkeypatch):
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", "0")
    c, ref = QSAKVCache(ratio=4), _Ref()
    for i, n in enumerate([12, 1, 1, 3]):
        k, v, ik = _rows(n, 200 + i)
        assert c.append_qsa(k, v, ik) is False
        ref.push(k, v, ik)
    assert c._kt is None
    _check_state(c, ref)


def test_decode_appends_leave_the_base(monkeypatch):
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", "16")
    c = QSAKVCache(ratio=4)
    c.append_qsa(*_rows(40, 1))
    base = (c._kb, c._vb, c._ib)
    for i in range(16):
        assert c.append_qsa(*_rows(1, 300 + i))
        assert all(a is b for a, b in zip((c._kb, c._vb, c._ib), base))
    assert c._tl == 16 and c.offset == 56
    # the next row does not fit: the tail folds into the base first
    assert c.append_qsa(*_rows(1, 400))
    assert c._tl == 1 and c.offset == 57


def test_finished_blocks_span_segments(monkeypatch):
    """A block whose raw keys straddle the base and the tail pools the
    same rows as one buffer."""
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", "16")
    c, ref = QSAKVCache(ratio=4), _Ref()
    for i, n in enumerate([10, 1, 1, 1, 1, 1, 1]):
        k, v, ik = _rows(n, 500 + i)
        c.append_qsa(k, v, ik)
        ref.push(k, v, ik)
    got = c.finished_blocks(4, lambda raw, start: raw.reshape(1, -1, 4, DI).sum(axis=2))
    want = ref.ik[:, :16].reshape(1, 4, 4, DI).sum(axis=2)
    assert np.allclose(np.asarray(got), want, atol=1e-5)


def _finish(raw, start):
    m = raw.shape[1] // 4
    return raw.reshape(1, m, 4, DI).sum(axis=2) + (start + mx.arange(m))[None, :, None]


@pytest.mark.parametrize("cap", [0, 8, 1024])
def test_block_segments_match_one_buffer(monkeypatch, cap):
    """A decode step's block joins the tail and leaves the base alone; the
    segments read back as the blocks of one buffer through rollbacks and
    tail folds."""
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", str(cap))
    c, ref = QSAKVCache(ratio=4), _Ref()
    script = [37] + [1] * 9 + [-6, 4, 4, 1, 8, -3] + [1] * 14 + [-30, 2, 4, 40, 1, 4]
    tailed = False
    for i, n in enumerate(script):
        if n < 0:
            c.trim(-n)
            ref.trim(-n)
        else:
            k, v, ik = _rows(n, 800 + i)
            c.append_qsa(k, v, ik)
            ref.push(k, v, ik)
        nb = c.offset // 4
        if not nb:
            continue
        held = c.blocks
        base, tail = c.block_segments(nb, _finish)
        want = ref.ik[:, :nb * 4].reshape(1, nb, 4, DI).sum(axis=2) + np.arange(nb)[None, :, None]
        got = base if tail is None else mx.concatenate([base, tail], axis=1)
        assert np.allclose(np.asarray(got), want, atol=1e-4)
        assert np.allclose(np.asarray(c.finished_blocks(nb, _finish)), want, atol=1e-4)
        if tail is not None:
            tailed = True
            assert c.blocks is held and tail.shape[1] <= cap // 4
    assert tailed == (cap > 0)


def _gather_case(monkeypatch, kernel: bool):
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", "16")
    if not kernel:
        monkeypatch.setattr(q4, "_kv_gather", lambda: None)
    c, ref = QSAKVCache(ratio=4), _Ref()
    for i, n in enumerate([41, 1, 1, 1, 3, 1]):
        k, v, ik = _rows(n, 600 + i)
        c.append_qsa(k, v, ik)
        ref.push(k, v, ik)
    assert c._tl == 7
    rows = mx.array([[0, 47, 40, 41, 5, 46, 40, 12, 33]], dtype=mx.int32)
    k, v = c.gather_kv(rows)
    idx = np.asarray(rows)[0]
    assert np.array_equal(np.asarray(k), ref.k[:, :, idx])
    assert np.array_equal(np.asarray(v), ref.v[:, :, idx])
    # a rollback into the base moves the boundary; stale base rows past it
    # are not read
    c.trim(9)
    ref.trim(9)
    c.append_qsa(*(r := _rows(2, 700)))
    ref.push(*r)
    rows = mx.array([[38, 39, 40, 0]], dtype=mx.int32)
    k, _ = c.gather_kv(rows)
    assert np.array_equal(np.asarray(k), ref.k[:, :, np.asarray(rows)[0]])


def test_gather_ops_match_one_buffer(monkeypatch):
    _gather_case(monkeypatch, kernel=False)


@gpu_only
def test_gather_kernel_matches_one_buffer(monkeypatch):
    _gather_case(monkeypatch, kernel=True)


def _layer(head_dim, dtype, budget=8):
    args = ModelArgs(hidden_size=128, num_hidden_layers=1,
                     num_attention_heads=12, num_key_value_heads=2,
                     head_dim=head_dim, indexer_budget=budget,
                     compress_ratios=[4], layer_types=["full_attention"])
    mx.random.seed(13)
    layer = Attention(args, 0)
    layer.set_dtype(dtype)
    for lin in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                layer.indexer.q_proj, layer.indexer.k_proj):
        lin.weight = (mx.random.normal(lin.weight.shape) * 0.05).astype(dtype)
    return layer


def _decode(layer, widths, cap, monkeypatch, prefill=41):
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", str(cap))
    mx.random.seed(31)
    cache = QSAKVCache(ratio=4)
    dtype = layer.q_proj.weight.dtype
    outs = [layer(mx.random.normal((1, prefill, 128)).astype(dtype), mask=None,
                  cache=cache)]
    for w in widths:
        outs.append(layer(mx.random.normal((1, w, 128)).astype(dtype),
                          mask=None, cache=cache))
        if w > 1:
            cache.trim(w - 1)  # verify rollback keeps the first row
    mx.eval(*outs)
    return outs, cache


@gpu_only
@pytest.mark.parametrize("head_dim, dtype", [
    (256, mx.bfloat16),   # paged kernel at L=1
    (64, mx.float32),     # gathered path
])
def test_segmented_attention_matches_one_buffer(monkeypatch, head_dim, dtype):
    """Decode and verify-width steps past the sparse boundary, with
    rollbacks and tail folds, against the same steps on one buffer."""
    layer = _layer(head_dim, dtype)
    widths = [1, 1, 3, 1, 4, 1, 1, 2, 1, 1, 1, 1, 3, 1]
    got, cache = _decode(layer, widths, 4, monkeypatch)
    ref, _ = _decode(layer, widths, 0, monkeypatch)
    assert cache._kt is not None
    for a, b in zip(got, ref):
        assert mx.array_equal(a, b).item()


@gpu_only
def test_two_segment_scores_match_one_buffer(monkeypatch):
    """Past 512 blocks the fused scorer runs once per block segment. The
    selection, and so the output, equals the one-buffer run."""
    layer = _layer(256, mx.bfloat16, budget=2048)
    widths = [1, 1, 3, 1, 4, 1, 1, 2, 1, 1, 1, 1, 3, 1, 4, 4, 1, 1, 1, 1]
    got, cache = _decode(layer, widths, 1024, monkeypatch, prefill=2301)
    ref, _ = _decode(layer, widths, 0, monkeypatch, prefill=2301)
    assert cache._bt is not None and cache.n_blocks > cache._nbb >= 512
    for a, b in zip(got[1:], ref[1:]):
        assert mx.array_equal(a, b).item()
