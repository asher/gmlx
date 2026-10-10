#!/usr/bin/env python3
"""Owned-engine prefill: the last chunk's full logits are released once the
first token is taken. The decode rounds read neither the chunk's output nor
its logits, and a held 2048-row chunk of a 248K vocabulary pins about
1 GB for the whole reply. The first logits row is a view of that array, so
dropping the output alone does not free it."""

from __future__ import annotations

import gc
import types

import mlx.core as mx
import pytest

from gmlx.spec.speculative import stream_speculative

T = 64
V = 1 << 16
LOGITS_BYTES = T * V * 4


class _StubLM:
    def rollback_speculative_cache(self, *a, **k):
        pass

    def __call__(self, ids, cache=None, return_hidden=False,
                 return_shared_kv=False, **kw):
        n = int(ids.shape[1])
        return types.SimpleNamespace(
            logits=mx.zeros((1, n, V)),
            hidden_states=[mx.zeros((1, n, 4))],
            shared_kv_states={},
        )


@pytest.mark.parametrize("greedy", [True, False])
def test_first_token_releases_the_prefill_logits(greedy):
    model = types.SimpleNamespace(language_model=_StubLM())
    drafter = types.SimpleNamespace(config=types.SimpleNamespace(
        block_size=2))
    prompt = mx.zeros((1, T), dtype=mx.int32)
    sampler = None if greedy else (lambda lp: mx.argmax(lp, axis=-1))
    mx.eval(prompt)
    gc.collect()
    base = mx.get_active_memory()
    gen = stream_speculative(model, drafter, prompt, prompt_cache=[],
                             max_tokens=1, sampler=sampler, prefill_chunk=T)
    assert next(gen) == 0
    gc.collect()
    held = mx.get_active_memory() - base
    gen.close()
    assert held < LOGITS_BYTES // 8, (
        f"{held} bytes still active after the first token")
