#!/usr/bin/env python3
"""Owned-engine prefill split: a target with ``prefill_split_last`` gets the
mlx-lm split (chunk n-1, step the last token at T=1); everything else keeps
the full-prompt chunking."""

from __future__ import annotations

import types

import mlx.core as mx

from gmlx.spec.speculative import stream_speculative


class _StubLM:
    def __init__(self, split_last):
        if split_last:
            self.prefill_split_last = True
        self.calls = []

    def rollback_speculative_cache(self, *a, **k):
        pass

    def __call__(self, ids, cache=None, return_hidden=False,
                 return_shared_kv=False, **kw):
        self.calls.append(int(ids.shape[1]))
        T = int(ids.shape[1])
        return types.SimpleNamespace(
            logits=mx.zeros((1, T, 8)),
            hidden_states=[mx.zeros((1, T, 4))],
            shared_kv_states={},
        )


def _prefill_calls(split_last):
    lm = _StubLM(split_last)
    model = types.SimpleNamespace(language_model=lm)
    drafter = types.SimpleNamespace(config=types.SimpleNamespace(
        block_size=2))   # no prefill_from_target_hidden
    gen = stream_speculative(
        model, drafter, mx.zeros((1, 7), dtype=mx.int32),
        prompt_cache=[], max_tokens=1, prefill_chunk=4)
    toks = list(gen)
    assert len(toks) == 1
    return lm.calls


def test_split_last_prefill():
    assert _prefill_calls(True) == [4, 2, 1]


def test_default_full_prefill():
    assert _prefill_calls(False) == [4, 3]
