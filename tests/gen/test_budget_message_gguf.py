"""The thinking-budget message on real weights, on the serve plain path: the
criteria the server builds for a request with a budget and a message forces
the message's ids, then the end marker, after the budget's thinking tokens,
and the reply's reasoning ends with the message. A budget of 0 on a prompt
that opens the block lets no thinking token through.

``integration``; picks the smallest ``qwen35`` GGUF under
``KQUANT_TEST_GGUF_DIR``, else a ``qwen3`` one, and skips without either.
The decode is greedy and runs without a drafter.
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest

import mlx.core as mx

pytest.importorskip("gmlx._vendor.mlx_vlm")

pytestmark = pytest.mark.integration

MESSAGE = "\n\nI am out of time, so I will answer now."
N_TOKENS = 40


def _pick(gguf_index):
    for arch in ("qwen35", "qwen3"):
        paths = gguf_index.get(arch)
        if paths:
            return min(paths, key=os.path.getsize)
    pytest.skip("no qwen35 or qwen3 GGUF under KQUANT_TEST_GGUF_DIR "
                f"(have: {sorted(gguf_index)})")


def _criteria(tok, ids, budget):
    from gmlx.serve.patches import chat_behavior

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    cls = gen.ResponseGenerator
    original = cls._make_thinking_budget_criteria
    try:
        chat_behavior.install_thinking_budget_fix()
        args = gen.GenerationArguments(
            max_tokens=N_TOKENS, thinking_budget=budget,
            thinking_budget_message=MESSAGE)
        return cls._make_thinking_budget_criteria(
            SimpleNamespace(tokenizer=tok), args, mx.array([ids]))
    finally:
        cls._make_thinking_budget_criteria = original


@pytest.mark.parametrize("budget", [8, 0])
def test_budget_message_then_end_marker_on_the_plain_path(gguf_index,
                                                          monkeypatch, budget):
    from gmlx import load_model
    from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
    from gmlx.gen.thinking_budget import (
        budget_message_ids,
        marker_ids,
        prompt_reasoning_tokens,
    )
    from gmlx.models.vlm_text_only import TextOnlyModel

    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    raw, _config, tok = load_model(_pick(gguf_index), verbose=False)
    from gmlx.serve.bridge_vlm import _ensure_text_embedding_probe

    model = TextOnlyModel(raw)
    _ensure_text_embedding_probe(model, raw)  # as the server's load does
    lm = model.language_model
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Explain why the sky is blue."}],
        add_generation_prompt=True, tokenize=False, enable_thinking=True)
    ids = list(tok.encode(prompt, add_special_tokens=False))
    start, end = marker_ids(tok, "<think>"), marker_ids(tok, "</think>")
    seed = prompt_reasoning_tokens(ids, start, (end,))
    crit = _criteria(tok, ids, budget)
    assert crit.thinking_token_count == seed

    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    gen = ar.BatchGenerator(lm, proc, prefill_step_size=None,
                            max_tokens=N_TOKENS,
                            stream=mx.default_stream(mx.default_device()))
    (uid,) = gen.insert(
        [ids], N_TOKENS, thinking_budget_criteria=[crit],
        prompt_kwargs=[{"inputs_embeds": lm.input_embeds(mx.array([ids]))}])
    out = []
    while gen.has_work:
        _p, rs = gen.next()
        out += [int(r.token) for r in rs if r.uid == uid]
    gen.close()

    forced = list(budget_message_ids(tok, MESSAGE)) + list(end)
    # The plain path forces right after the budget's last thinking token.
    if crit._prompt_open:
        cut = max(budget - seed, 0)
    elif start[0] in out:
        cut = out.index(start[0]) + 1 + budget
    else:
        pytest.skip("the model wrote no thinking block")
    if end[0] in out[:cut]:
        pytest.skip("the model closed its thinking inside the budget")
    assert out[cut:cut + len(forced)] == forced
    text = tok.decode(out)
    reasoning = text.split("</think>", 1)[0]
    assert reasoning.endswith(MESSAGE)
