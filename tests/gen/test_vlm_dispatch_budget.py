"""thinking_budget on the copied mlx-vlm ``stream_generate`` path, with the
real Muse Spark and Muse Glimmer token ids.

ATEM's reasoning start marker is five tokens and the generation prompt
ends two tokens into it. The criteria must match both markers on all
their tokens, complete the start marker on the generated rest, and read
the prompt's open state from its last start and end markers. The model is
a script: it writes its text until a close is forced, then the rest.
"""

from __future__ import annotations

import contextlib
import importlib
import types

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import atem_tokens as at  # noqa: E402
import mlx.core as mx  # noqa: E402

_DISPATCH = importlib.import_module("gmlx._vendor.mlx_vlm.generate.dispatch")

_PROMPT = "<|start|>user<|message|>hi<|eot|><|start|>assistant"
_ANSWER = "<|start|>assistant to=user<|message|>Hello world foo bar<|eot|>"
_REASON = " to=self<|message|>Thinking hard about it now.<|eom|>"


class _Detok:
    def __init__(self):
        self.last_segment = ""

    def add_token(self, token, skip_special_token_ids=()):
        pass

    def finalize(self):
        pass


def _scripted_step(script, after):
    """A ``generate_step`` that yields ``script`` and, once the criteria
    forces a token, that token and then ``after``. It pops the forced id
    after each yield, as the copied ``generate_step`` does."""

    def generate_step(input_ids, model, pixel_values, mask, **kw):
        crit = kw.get("thinking_budget_criteria")
        src, rest, nxt = iter(script), None, None
        while True:
            if nxt is not None:
                tid, src = nxt, None
                rest = rest or iter(after)
            else:
                tid = next(src if src is not None else rest, None)
                if tid is None:
                    return
            yield tid, None
            nxt = crit.pop_forced_token_id() if crit is not None else None

    return generate_step


def _generate(monkeypatch, tok, script, after, *, prompt=_PROMPT, budget=3):
    ids = tok.encode(prompt)
    monkeypatch.setattr(
        _DISPATCH, "_prepare_generation_inputs",
        lambda *a, **k: (mx.array([ids]), None, None, None))
    monkeypatch.setattr(_DISPATCH, "is_diffusion_model", lambda *a, **k: False)
    monkeypatch.setattr(_DISPATCH, "make_streaming_detokenizer",
                        lambda processor: _Detok())
    monkeypatch.setattr(_DISPATCH, "wired_limit",
                        lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(_DISPATCH.cache, "make_prompt_cache",
                        lambda *a, **k: [])
    monkeypatch.setattr(_DISPATCH, "generate_step",
                        _scripted_step(tok.encode(script), tok.encode(after)))
    tok.stopping_criteria = lambda t: False
    model = types.SimpleNamespace(config=types.SimpleNamespace(),
                                  language_model=None)
    out = list(_DISPATCH.stream_generate(
        model, tok, prompt, thinking_budget=budget,
        thinking_start_token=at.SELF_OPEN, thinking_end_token=at.EOM,
        enable_thinking=True))
    return tok, tok.decode(out[-1].token_ids)


@pytest.mark.parametrize("template", ["spark", "glimmer"])
def test_budget_caps_the_reasoning_block(monkeypatch, template):
    tok, text = _generate(monkeypatch, at.AtemTokenizer(template),
                          _REASON + _ANSWER, _ANSWER)
    assert text == " to=self<|message|>Thinking hard about\n<|eom|>" + _ANSWER
    assert not tok.thinking_budget_criteria.in_thinking


def test_budget_leaves_commentary_and_the_answer_alone(monkeypatch):
    head = " to=commentary<|message|>Hello world foo bar<|eom|><|start|>assistant"
    tok, text = _generate(monkeypatch, at.AtemTokenizer(),
                          head + _REASON + _ANSWER, _ANSWER)
    assert text == (head + " to=self<|message|>Thinking hard about\n<|eom|>"
                    + _ANSWER)
    # The reasoning tokens up to the cap and the forced newline. The
    # commentary counts for nothing.
    assert tok.thinking_budget_criteria.thinking_token_count == 5


def test_a_prompt_with_closed_reasoning_does_not_start_inside(monkeypatch):
    prompt = (_PROMPT + " to=self<|message|>Thinking<|eom|>"
              "<|start|>assistant to=user<|message|>Hello<|eot|>"
              "<|start|>user<|message|>hi<|eot|><|start|>assistant")
    tok, text = _generate(monkeypatch, at.AtemTokenizer(), _ANSWER[18:], "",
                          prompt=prompt)
    assert text == _ANSWER[18:]
    assert tok.thinking_budget_criteria.thinking_token_count == 0


def test_a_prompt_that_opens_reasoning_counts_from_the_start(monkeypatch):
    prompt = _PROMPT + " to=self<|message|>"
    tok, text = _generate(monkeypatch, at.AtemTokenizer(),
                          "Thinking hard about it now.<|eom|>" + _ANSWER,
                          _ANSWER, prompt=prompt)
    assert text == "Thinking hard about\n<|eom|>" + _ANSWER
