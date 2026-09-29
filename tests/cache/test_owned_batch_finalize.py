"""A batch KV cache's finalize rebinds left_padding.

The stock finalize adds the rolled tail to ``left_padding`` in place, so the
identity-keyed pad memos keep the pads from before the trim. The owned
BatchKVCache, BatchRotatingKVCache and BatchQuantizedKVCache rebind it. CPU
checks go through the memo readers and through the vendored qwen3.5
rollback, and a GPU model check runs a B=2 qwen3.5 speculative rollback
with accept counts 4 and 0, then one decode, against single-stream on both
rows. Each check also runs the stock finalize body, which must fail it.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

import gmlx.cache.kv_cache as owned
from gmlx.models.qwen35.language import _qwen3_5_left_padding_info
from gmlx.models.qwen35.verify_fold import _pads_list
from stock_twin import stock_method

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="qwen3_5 GDN forward is Metal-only")


@pytest.fixture(params=["owned", "stock"])
def body(request, monkeypatch):
    """Run each check on the owned finalize and on the stock one, where
    it must fail."""
    if request.param == "stock":
        for cls in (owned.BatchKVCache, owned.BatchRotatingKVCache,
                    owned.BatchQuantizedKVCache):
            monkeypatch.setattr(cls, "finalize",
                                stock_method(cls, "finalize"))
    return request.param


def _kv(S, D=8):
    return mx.ones((2, 2, S, D))


def _read_memos(c):
    return _qwen3_5_left_padding_info(c), _pads_list(c)


def _trim_refreshes(c, pads) -> bool:
    before = c.left_padding
    old = before.tolist()
    c.finalize()
    assert c.left_padding.tolist() == pads
    return (_read_memos(c) == ((tuple(pads), max(pads)), pads)
            and before.tolist() == old)


def _expect(body, ok):
    assert ok is (body == "owned"), body


def test_right_padding_trim_refreshes_the_pad_memos(body):
    c = owned.BatchKVCache(left_padding=[0, 0])
    c.update_and_fetch(_kv(6), _kv(6))
    assert _read_memos(c) == (((0, 0), 0), [0, 0])
    c.prepare(right_padding=[0, 2])
    _expect(body, _trim_refreshes(c, [0, 2]))


def test_rotating_lengths_trim_refreshes_the_pad_memos(body):
    c = owned.BatchRotatingKVCache(max_size=16, left_padding=[0, 0])
    c.prepare(lengths=[6, 4], right_padding=[0, 2])
    c.update_and_fetch(_kv(6), _kv(6))
    _read_memos(c)
    _expect(body, _trim_refreshes(c, [0, 2]))


def test_quantized_right_padding_trim_refreshes_the_pad_memos(body):
    c = owned.BatchQuantizedKVCache(left_padding=[0, 0], group_size=64,
                                    bits=8)
    c.update_and_fetch(_kv(6, 64), _kv(6, 64))
    _read_memos(c)
    c.prepare(right_padding=[0, 2])
    _expect(body, _trim_refreshes(c, [0, 2]))


def test_ragged_rollback_refreshes_every_pad_reader(body):
    from gmlx.models.gemma4.batched_sdpa import _pads as gemma4_pads
    from gmlx.models.qwen35.language import LanguageModel
    from gmlx.upstream.cascade_sdpa import _pads as cascade_pads

    c = owned.BatchKVCache(left_padding=[0, 5])
    c.update_and_fetch(_kv(12), _kv(12))

    def readers():
        return (_qwen3_5_left_padding_info(c)[0], _pads_list(c),
                cascade_pads(c), gemma4_pads(c))

    assert readers() == ((0, 5), [0, 5], [0, 5], [0, 5])
    # Row 1 accepted two fewer drafts, so the rollback right-trims it through
    # prepare and finalize. Called unbound: with KV-only caches the stock
    # body never reads self.
    LanguageModel.rollback_speculative_cache(None, [c], [], [2, 0], 3)
    assert c.left_padding.tolist() == [0, 7]
    _expect(body, readers() == ((0, 7), [0, 7], [0, 7], [0, 7]))


def test_finalize_without_a_tail_keeps_the_array():
    c = owned.BatchKVCache(left_padding=[1, 0])
    c.update_and_fetch(_kv(6), _kv(6))
    lp = c.left_padding
    c.finalize()
    assert c.left_padding is lp


# The GPU runs production numerics: float32 matmuls at TF32 precision on
# M5-class GPUs, where the batch and a single-stream row differ by rounding.
# Over 40 weight seeds that difference reached 0.0148 on row 0 and 0.0107 on
# row 1, and each row's tolerance is about three times its worst case. With
# the stock finalize, row 1 attends its four rolled-in drafts and differs
# by 0.177 to 0.444 over the same seeds, at least 5.5 times its tolerance.
_ROW_TOL = (0.045, 0.032)


@_NEEDS_GPU
def test_ragged_rollback_then_decode_matches_single_stream(monkeypatch):
    import gmlx.models.qwen35.owned as qwen35_owned
    from gmlx.models.qwen35.cache import Qwen35ArraysCache
    from gmlx.models.qwen35.config import TextConfig

    mx.random.seed(11)
    cfg = TextConfig(
        model_type="qwen3_5", hidden_size=64, intermediate_size=128,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, num_hidden_layers=4, num_attention_heads=4,
        rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=2,
        max_position_embeddings=2048, tie_word_embeddings=True, head_dim=32,
        rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                         "rope_theta": 100000, "partial_rotary_factor": 0.25},
        full_attention_interval=4,
    )
    top = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=124, video_token_id=125, vision_start_token_id=126)
    lm = qwen35_owned.language_model_class("qwen3_5")(cfg, top)
    mx.eval(lm.parameters())

    prompts = [[3, 17, 42, 9, 8, 7, 6, 5, 4], [11, 12, 13, 14]]
    first, nxt = [20, 21], [50, 51]
    blocks = [[30, 31, 32, 33, 34], [40, 41, 42, 43, 44]]
    accepted, width = [4, 0], 5

    def verify(ids, cache):
        return lm(ids, cache=cache, capture_layer_ids=[], return_hidden=True,
                  return_shared_kv=True)

    refs = []
    for r in range(2):
        c = lm.make_cache()
        lm(mx.array([prompts[r]]), cache=c)
        lm(mx.array([[first[r]]]), cache=c)
        out = verify(mx.array([blocks[r]]), c)
        lm.rollback_speculative_cache(c, out.gdn_states, accepted[r], width)
        refs.append(lm(mx.array([[nxt[r]]]), cache=c).logits)

    def batched():
        rows = []
        for r in range(2):
            c = lm.make_cache()
            lm(mx.array([prompts[r]]), cache=c)
            rows.append(c)
        batch = [
            (owned.BatchKVCache if isinstance(a, owned.KVCache)
             else Qwen35ArraysCache).merge([a, b])
            for a, b in zip(*rows)
        ]
        lm(mx.array([[t] for t in first]), cache=batch)
        out = verify(mx.array(blocks), batch)
        lm.rollback_speculative_cache(batch, out.gdn_states, accepted, width)
        logits = lm(mx.array([[t] for t in nxt]), cache=batch).logits
        return [mx.abs(logits[r : r + 1] - refs[r]).max().item()
                for r in range(2)]

    fixed = batched()
    assert all(d <= tol for d, tol in zip(fixed, _ROW_TOL)), fixed
    # Control: the stock finalize must fail this check by a wide margin, or
    # the tolerance could not tell the bug from rounding.
    monkeypatch.setattr(owned.BatchKVCache, "finalize",
                        stock_method(owned.BatchKVCache, "finalize"))
    stock = batched()
    assert stock[1] > 3 * _ROW_TOL[1], stock
