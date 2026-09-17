"""The DeepSeek-V4.1 prefill tail: with the prompt end armed on the first
cache entry, a layer past the last kv-source layer runs only the rows the
layers after it can still reach, and the chunked pass matches the one
that runs every row (last-row logits, the decode steps after it)."""
from __future__ import annotations

import importlib.util
import pathlib

import mlx.core as mx
import pytest

from gmlx.gen import prefill_tail

_SIBLING = pathlib.Path(__file__).with_name("test_deepseek_v41_model.py")
_PROMPT = mx.array([[(i * 7 + 5) % 60 for i in range(30)]])


def _model_module():
    spec = importlib.util.spec_from_file_location("_v41_model_tests_tail", _SIBLING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(model, prompt, chunks, ends=()):
    """Feed ``chunks`` rows then the last token alone, as the generation
    loop does; ``ends`` = {chunk index: prompt end to arm before it}."""
    cache = model.make_cache()
    outs, pos = [], 0
    ends = dict(ends)
    for i, n in enumerate(chunks):
        if i in ends:
            cache[0]._gmlx_prefill_end = ends[i]
        outs.append(model(prompt[:, pos:pos + n], cache=cache))
        mx.eval([c.state for c in cache])
        pos += n
    logits = model(prompt[:, pos:], cache=cache)
    steps = [model(mx.array([[33]]), cache=cache), model(mx.array([[41]]), cache=cache)]
    mx.eval(logits, steps)
    return outs, logits, steps, cache


def _same(a, b):
    return mx.allclose(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("over", [
    {},
    dict(candidate_source_layer=4, candidate_topk_blocks=1, candidate_block_size=2),
])
def test_armed_chunks_match_the_full_pass(over):
    tm = _model_module()
    model = tm._randomized(tm.Model(tm._args(**over)))
    chunks = (8, 8, 8, 5)
    full = _run(model, _PROMPT, chunks)
    tail = _run(model, _PROMPT, chunks, ends={0: 30})
    # window 4, six layers, kv sources at 2 and 4: layers 4 and 5 keep
    # rows 23+ and 26+, so the last layer has nothing before the end.
    assert [o.shape[1] for o in tail[0]] == [0, 0, 0, 0]
    assert [o.shape[1] for o in full[0]] == list(chunks)
    assert _same(full[1][:, -1], tail[1][:, -1])
    for a, b in zip(full[2], tail[2]):
        assert _same(a, b)
    assert tail[3][0]._gmlx_prefill_end is None
    # The last layer saw its 3 prep rows, the last token and 2 steps.
    assert tail[3][-1].offset == 6


def test_rearming_at_a_checkpoint_stays_exact():
    """An end at a checkpoint column mid-prompt (serve path) then the
    real end: every row up to the checkpoint is exact, so the run after
    it matches the plain chunked pass."""
    tm = _model_module()
    model = tm._randomized(tm.Model(tm._args()))
    chunks = (8, 8, 8, 5)
    full = _run(model, _PROMPT, chunks)
    tail = _run(model, _PROMPT, chunks, ends={0: 16, 2: 30})
    assert tail[0][1].shape[1] > 0
    assert _same(full[1][:, -1], tail[1][:, -1])
    for a, b in zip(full[2], tail[2]):
        assert _same(a, b)


def test_tail_is_off_for_batches_and_by_env(monkeypatch):
    tm = _model_module()
    model = tm._randomized(tm.Model(tm._args()))
    prompt = mx.concatenate([_PROMPT, _PROMPT[:, ::-1]], axis=0)
    cache = model.make_cache()
    cache[0]._gmlx_prefill_end = 30
    out = model(prompt[:, :8], cache=cache)
    assert out.shape[:2] == (2, 8)
    monkeypatch.setenv("GMLX_DS41_PREFILL_TAIL", "0")
    outs, _, _, _ = _run(model, _PROMPT, (8, 8, 8, 5), ends={0: 30})
    assert [o.shape[1] for o in outs] == [8, 8, 8, 5]


class _Batch:
    def __init__(self, rows, cols, depth, col=None, done=0, draft=None):
        self._inputs_embeds = mx.zeros((rows, cols, 4))
        self.prompt_cache = [_Cache(depth), _Cache(depth)]
        self._col = col
        self._processed_prompt_columns = done
        self.draft_model = draft

    def _next_apc_checkpoint_column(self):
        return self._col


class _Cache:
    def __init__(self, offset):
        self.offset = offset


def test_arm_batch_ends_at_the_prompt_or_the_next_checkpoint():
    b = _Batch(1, 10, depth=5)
    assert prefill_tail.arm_batch(b) == 15
    assert b.prompt_cache[0]._gmlx_prefill_end == 15
    assert prefill_tail.arm_batch(_Batch(1, 10, depth=5, col=8, done=2)) == 11
    assert prefill_tail.arm_batch(_Batch(2, 10, depth=5)) is None
    assert prefill_tail.arm_batch(_Batch(1, 10, depth=5, draft=object())) is None
    assert prefill_tail.arm_prefill_tail([], 4) is None
