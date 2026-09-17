"""merge_prefill_tail: a short last prefill chunk folds into the chunks
before it, never growing a chunk past 9/8 of the step."""
from __future__ import annotations

import pytest

from gmlx.stream.expert_streaming import merge_prefill_tail


@pytest.mark.parametrize("n_tokens,step,want", [
    (8208, 8192, 8208),      # 8192 + 16: one chunk
    (16417, 8192, 8209),     # 8192 + 8192 + 33: 8209 + 8208
    (8192 + 1024, 8192, 9216),   # tail at the limit merges
    (8192 + 1025, 8192, 8192),   # past it, keep the step
    (16384, 8192, 8192),     # no tail
    (8000, 8192, 8192),      # under one chunk
    (100, 0, 0),
    (100, None, None),
])
def test_merge(n_tokens, step, want):
    assert merge_prefill_tail(step, n_tokens) == want


def test_switch_keeps_the_step(monkeypatch):
    monkeypatch.setenv("GMLX_STREAM_PREFILL_TAIL_MERGE", "0")
    assert merge_prefill_tail(8192, 8208) == 8192


def test_generation_widens_the_streamed_step(monkeypatch):
    """The CLI generate path folds the tail before handing the step on."""
    import gmlx.gen.generation as gen
    from gmlx.stream import expert_streaming as es

    monkeypatch.setattr(es, "_resolve_prefill_step", lambda m, r: (8192, True))
    monkeypatch.setattr(es, "moe_streaming_active", lambda m: True)
    seen = {}

    class Tok:
        bos_token = None
        chat_template = None

        def encode(self, prompt, add_special_tokens=True):
            return [1] * 8209

    def fake_stream(model, tokenizer, prompt, **kw):
        seen.update(kw)
        return iter(())

    import importlib

    mg = importlib.import_module("mlx_lm.generate")   # not the function
    monkeypatch.setattr(mg, "stream_generate", fake_stream)
    try:
        list(gen.generate(object(), Tok(), "x" * 10, max_tokens=1,
                          apply_chat_template=False, verbose=False))
    except Exception:  # noqa: BLE001 - only the step hand-off is under test
        pass
    assert seen["prefill_step_size"] == 8208
