"""The native MTP head writes its KV in place across a round boundary.

The engine runs accept through ``draft_call``, which async-evaluates the
head's seed with its cache state. A draft step built while that work runs
cannot donate the head's KV buffer, so its append copies all of the K and
V: one head buffer per round, 0.5 GiB at 128K on Qwen3.8-27B.
``draft_block`` waits for the seed first. Tiny head, deep zero cache.
"""

from __future__ import annotations

import dataclasses
import importlib

import mlx.core as mx
import mlx.nn as nn
import pytest

from helpers import _real_apple_gpu

pytest.importorskip("gmlx._vendor.mlx_vlm.speculative.drafters.qwen3_5_mtp.config")

from gmlx.spec.helpers import _SpeculativeSamplerRNG  # noqa: E402
from gmlx.spec.mtp_drafter import QwenMTPDrafter  # noqa: E402
from test_vlm_spec_target_swap import _q35_text_cfg  # noqa: E402

_DEPTH = 1 << 18
_ROUNDS = 4


class _Target(nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)


def _head():
    text = _q35_text_cfg()
    cfg = importlib.import_module(
        "gmlx._vendor.mlx_vlm.speculative.drafters.qwen3_5_mtp.config")
    head = QwenMTPDrafter(cfg.Qwen3_5MTPConfig.from_dict({
        "model_type": "qwen3_5_mtp",
        "text_config": {f.name: getattr(text, f.name)
                        for f in dataclasses.fields(text)}}))
    mx.random.seed(11)
    target = _Target(text.vocab_size, text.hidden_size)
    mx.eval(head.parameters(), target.parameters())
    head.reset(target)
    return head, text


def _deep(head, text):
    """Each head layer holds ``_DEPTH`` zero rows; returns the K+V bytes."""
    n = 0
    for c in head._cache:
        shape = (1, text.num_key_value_heads, _DEPTH + 512, text.head_dim)
        c.keys = mx.zeros(shape, mx.float32)
        c.values = mx.zeros(shape, mx.float32)
        c.offset = _DEPTH
        mx.eval(c.keys, c.values)
        n += c.keys.nbytes + c.values.nbytes
    return n


@pytest.mark.skipif(not _real_apple_gpu(),
                    reason="buffer donation under in-flight work is a Metal property")
def test_the_first_draft_append_of_a_round_does_not_copy_the_head_kv():
    head, text = _head()
    kv = _deep(head, text)
    rng = _SpeculativeSamplerRNG(head, enabled=False)
    head._seed_token = mx.array([[3]])
    head._seed_hidden = mx.zeros((1, 1, text.hidden_size))
    over = []
    for r in range(_ROUNDS):
        mx.clear_cache()
        a0 = mx.get_active_memory()
        mx.reset_peak_memory()
        drafts = head.draft_block(mx.array([5]), None, None, 3, None,
                                  greedy=True)
        mx.eval(drafts, [c.state for c in head._cache])
        over.append(mx.get_peak_memory() - a0)
        verify = mx.random.normal((1, 3, text.hidden_size), key=mx.random.key(r))
        rng.draft_call(head.accept_verified_tokens, verify, drafts, 1, [7],
                       None, mx.int32, greedy=True)
    mx.eval([c.state for c in head._cache])
    assert max(over) < kv // 2, (over, kv)
