"""batch_finalize_fix: a batch KV cache's finalize rebinds left_padding.

Stock finalize adds the rolled tail to ``left_padding`` in place, so the
identity-keyed pad memos keep the pads from before the trim. CPU checks per
class origin through the memo readers and through the vendored qwen3.5
rollback, and a GPU model check: a B=2 qwen3.5 speculative rollback with
accept counts 2 and 0, then one decode, matches single-stream on both rows.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx_vlm.models.cache  # noqa: F401  (both origins for cache_types)
import pytest

from gmlx.cache.compat import cache_types
from gmlx.models.qwen35.language import _qwen3_5_left_padding_info
from gmlx.models.qwen35.verify_fold import _pads_list
from gmlx.upstream.batch_finalize_fix import install_batch_finalize_fix

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="qwen3_5 GDN forward is Metal-only")


@pytest.fixture(autouse=True)
def installed():
    install_batch_finalize_fix()


def _kv(S, D=8):
    return mx.ones((2, 2, S, D))


def _read_memos(c):
    return _qwen3_5_left_padding_info(c), _pads_list(c)


def _check_trim(c, pads):
    before = c.left_padding
    old = before.tolist()
    c.finalize()
    assert _read_memos(c) == ((tuple(pads), max(pads)), pads)
    assert c.left_padding.tolist() == pads
    assert before.tolist() == old


@pytest.mark.parametrize("cls", cache_types("BatchKVCache"),
                         ids=lambda c: c.__module__)
def test_right_padding_trim_refreshes_the_pad_memos(cls):
    c = cls(left_padding=[0, 0])
    c.update_and_fetch(_kv(6), _kv(6))
    assert _read_memos(c) == (((0, 0), 0), [0, 0])
    c.prepare(right_padding=[0, 2])
    _check_trim(c, [0, 2])


@pytest.mark.parametrize("cls", cache_types("BatchRotatingKVCache"),
                         ids=lambda c: c.__module__)
def test_rotating_lengths_trim_refreshes_the_pad_memos(cls):
    c = cls(max_size=16, left_padding=[0, 0])
    c.prepare(lengths=[6, 4], right_padding=[0, 2])
    c.update_and_fetch(_kv(6), _kv(6))
    _read_memos(c)
    _check_trim(c, [0, 2])


def test_quantized_right_padding_trim_refreshes_the_pad_memos():
    (cls,) = cache_types("BatchQuantizedKVCache")
    c = cls(left_padding=[0, 0], group_size=64, bits=8)
    c.update_and_fetch(_kv(6, 64), _kv(6, 64))
    _read_memos(c)
    c.prepare(right_padding=[0, 2])
    _check_trim(c, [0, 2])


@pytest.mark.parametrize("cls", cache_types("BatchKVCache"),
                         ids=lambda c: c.__module__)
def test_stock_ragged_rollback_refreshes_every_pad_reader(cls):
    from gmlx.models.gemma4.batched_sdpa import _pads as gemma4_pads
    from gmlx.models.qwen35.language import LanguageModel
    from gmlx.upstream.cascade_sdpa import _pads as cascade_pads

    c = cls(left_padding=[0, 5])
    c.update_and_fetch(_kv(12), _kv(12))

    def readers():
        return (_qwen3_5_left_padding_info(c)[0], _pads_list(c),
                cascade_pads(c), gemma4_pads(c))

    assert readers() == ((0, 5), [0, 5], [0, 5], [0, 5])
    # Row 1 accepted two fewer drafts, so the rollback right-trims it through
    # prepare and finalize. Called unbound: with KV-only caches the stock
    # body never reads self.
    LanguageModel.rollback_speculative_cache(None, [c], [], [2, 0], 3)
    assert readers() == ((0, 7), [0, 7], [0, 7], [0, 7])
    assert c.left_padding.tolist() == [0, 7]


def test_finalize_without_a_tail_keeps_the_array():
    (cls,) = cache_types("BatchKVCache")[-1:]
    c = cls(left_padding=[1, 0])
    c.update_and_fetch(_kv(6), _kv(6))
    lp = c.left_padding
    c.finalize()
    assert c.left_padding is lp


def test_install_is_idempotent():
    (cls,) = cache_types("BatchKVCache")[-1:]
    fn = cls.finalize
    install_batch_finalize_fix()
    assert cls.finalize is fn


@_NEEDS_GPU
def test_ragged_rollback_then_decode_matches_single_stream():
    from mlx_vlm.models.cache import BatchKVCache, KVCache

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
    blocks = [[30, 31, 32], [40, 41, 42]]
    accepted = [2, 0]

    def verify(ids, cache):
        return lm(ids, cache=cache, capture_layer_ids=[], return_hidden=True,
                  return_shared_kv=True)

    refs = []
    for r in range(2):
        c = lm.make_cache()
        lm(mx.array([prompts[r]]), cache=c)
        lm(mx.array([[first[r]]]), cache=c)
        out = verify(mx.array([blocks[r]]), c)
        lm.rollback_speculative_cache(c, out.gdn_states, accepted[r], 3)
        refs.append(lm(mx.array([[nxt[r]]]), cache=c).logits)

    rows = []
    for r in range(2):
        c = lm.make_cache()
        lm(mx.array([prompts[r]]), cache=c)
        rows.append(c)
    batch = [
        (BatchKVCache if isinstance(a, KVCache) else Qwen35ArraysCache).merge(
            [a, b])
        for a, b in zip(*rows)
    ]
    lm(mx.array([[t] for t in first]), cache=batch)
    out = verify(mx.array(blocks), batch)
    lm.rollback_speculative_cache(batch, out.gdn_states, accepted, 3)
    logits = lm(mx.array([[t] for t in nxt]), cache=batch).logits
    for r in range(2):
        diff = mx.abs(logits[r : r + 1] - refs[r]).max().item()
        assert diff < 1e-2, (r, diff)
