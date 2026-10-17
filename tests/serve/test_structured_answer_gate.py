"""A ``response_format`` grammar constrains only the final answer of an
ATEM model (Muse Spark, Muse Glimmer), with the real token ids.

The <|eom|> that ends reasoning also ends commentary and tool calls, and
the answer text starts after its own header,
"<|start|>assistant to=user<|message|>". llama.cpp's Glimmer parser
applies the schema there too: any number of reasoning messages, then the
header, then the JSON. So the grammar waits for the header, in plain
decode and on speculative rounds, whether thinking is on or not.

The wrapped processor is a recorder. A generated token counts as
constrained when the recorder ran on the logits it was sampled from.
"""

from __future__ import annotations

import importlib
import types

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import atem_tokens as at  # noqa: E402
import mlx.core as mx  # noqa: E402

from gmlx._vendor.mlx_vlm.structured import (  # noqa: E402
    LLGuidanceLogitsProcessor,
    ThinkingAwareLogitsProcessor,
)
from gmlx.spec import row_procs  # noqa: E402
from gmlx.upstream.patching import root  # noqa: E402

_GEN = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")

_PROMPT = "<|start|>user<|message|>hi<|eot|><|start|>assistant"
_COMMENTARY = " to=commentary<|message|>Working on it.<|eom|><|start|>assistant"
_REASON = " to=self<|message|>Thinking hard about it now.<|eom|>"
_HEADER = "<|start|>assistant to=user<|message|>"
_CONTENT = "Hello world foo bar<|eot|>"


class _Recorder:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, logits):
        self.calls += 1
        return logits


def _serve_proc(tok, *, enable_thinking=True):
    """The request processor as the server wraps it for an ATEM request."""
    rec = _Recorder()
    me = _GEN.ResponseGenerator.__new__(_GEN.ResponseGenerator)
    me.tokenizer = tok
    args = types.SimpleNamespace(
        logit_bias=None, repetition_penalty=None, repetition_context_size=20,
        presence_penalty=None, presence_context_size=20,
        frequency_penalty=None, frequency_context_size=20,
        logits_processors=[rec], enable_thinking=enable_thinking,
        thinking_start_token=at.SELF_OPEN, thinking_end_token=at.EOM)
    make = root(_GEN.ResponseGenerator._make_logits_processors)
    (proc,) = make(me, args, mx.array([tok.encode(_PROMPT)]))
    return proc, rec


def _batch_decode(proc, rec, prompt, gen):
    """The batch loop: the prompt at prefill, then each sampled token
    through ``process_last_token``. Returns the constrained tokens."""
    logits, out = mx.zeros((1, 8)), []
    before = rec.calls
    proc(mx.array(prompt), logits)
    if rec.calls > before:
        out.append(gen[0])
    for k, t in enumerate(gen[:-1]):
        before = rec.calls
        proc.process_last_token(t, logits)
        if rec.calls > before:
            out.append(gen[k + 1])
    return out


def _step_decode(proc, rec, prompt, gen):
    """The single-stream loop: the prompt's last token, then the growing
    context."""
    logits, out, ctx = mx.zeros((1, 8)), [], [prompt[-1]]
    for t in gen:
        before = rec.calls
        proc(mx.array(ctx), logits)
        if rec.calls > before:
            out.append(t)
        ctx.append(t)
    return out


@pytest.mark.parametrize("template", ["spark", "glimmer"])
@pytest.mark.parametrize("drive", [_batch_decode, _step_decode])
@pytest.mark.parametrize("head", ["", _COMMENTARY], ids=["plain", "commentary"])
def test_the_grammar_waits_for_the_answer_header(template, drive, head):
    tok = at.AtemTokenizer(template)
    proc, rec = _serve_proc(tok)
    gen = tok.encode(head + _REASON + _HEADER + _CONTENT)
    out = drive(proc, rec, tok.encode(_PROMPT), gen)
    assert tok.decode(out) == _CONTENT


@pytest.mark.parametrize("drive", [_batch_decode, _step_decode])
def test_an_answer_without_reasoning_is_constrained_after_its_header(drive):
    tok = at.AtemTokenizer()
    proc, rec = _serve_proc(tok)
    gen = tok.encode(" to=user<|message|>" + _CONTENT)
    assert tok.decode(drive(proc, rec, tok.encode(_PROMPT), gen)) == _CONTENT


def test_the_header_is_gated_with_thinking_off():
    # The template reasons and writes the header either way.
    tok = at.AtemTokenizer()
    proc, rec = _serve_proc(tok, enable_thinking=False)
    assert isinstance(proc, ThinkingAwareLogitsProcessor)
    gen = tok.encode(_REASON + _HEADER + _CONTENT)
    out = _batch_decode(proc, rec, tok.encode(_PROMPT), gen)
    assert tok.decode(out) == _CONTENT


def test_reset_and_clone_start_from_the_prompt():
    tok = at.AtemTokenizer()
    proc, rec = _serve_proc(tok)
    gen = tok.encode(_COMMENTARY + _REASON + _HEADER + _CONTENT)
    _batch_decode(proc, rec, tok.encode(_PROMPT), gen)
    assert proc._active
    proc.reset()
    assert not proc._active
    assert tok.decode(_step_decode(proc, rec, tok.encode(_PROMPT), gen)) == _CONTENT
    twin = proc.clone()
    assert not twin._active
    twin.processor = rec
    assert tok.decode(_step_decode(twin, rec, tok.encode(_PROMPT), gen)) == _CONTENT


def test_speculative_rows_start_the_processor_at_the_header():
    tok = at.AtemTokenizer()
    proc, _ = _serve_proc(tok)
    prompt = tok.encode(_PROMPT)
    rows = row_procs.SpecRowProcessors.from_processors([proc], prompt)
    gate = rows.steps[0].gate
    eom, start = at.PIECES[at.EOM], at.PIECES["<|start|>"]
    head = tok.encode(_COMMENTARY + _REASON)
    rows.start(head[0])
    rows.commit(head[1:])
    assert gate.start is None
    # A commentary or reasoning <|eom|> among the drafts starts nothing.
    assert rows.plan([eom, start]) is None
    header = tok.encode(_HEADER)
    plan = rows.plan(header + [tok.encode("Hello")[0]])
    n = len(rows.context)
    assert plan.starts[0] == [None] * len(header) + [n + len(header) - 1] * 2
    rows.commit(header)
    assert gate.start == n + len(header) - 1


class _FakeMatcher:
    def __init__(self):
        self.consumed = []

    def is_stopped(self):
        return False

    def consume_token(self, t):
        self.consumed.append(int(t))
        return True


def test_speculative_grammar_consumes_only_the_answer(monkeypatch):
    tok = at.AtemTokenizer()
    proc, _ = _serve_proc(tok)
    inner = LLGuidanceLogitsProcessor.__new__(LLGuidanceLogitsProcessor)
    inner.llg_tokenizer = types.SimpleNamespace(vocab_size=202048)
    proc.processor = inner
    matcher = _FakeMatcher()
    monkeypatch.setattr(row_procs, "_fresh_matcher", lambda p: matcher)
    rows = row_procs.SpecRowProcessors.from_processors(
        [proc], tok.encode(_PROMPT))
    g = rows.grammar
    head = tok.encode(_COMMENTARY + _REASON)
    rows.start(head[0])
    rows.commit(head[1:])
    assert not g.active and matcher.consumed == []
    assert g.masks([at.PIECES[at.EOM], at.PIECES["<|start|>"]]) is None
    rows.commit(tok.encode(_HEADER + _CONTENT))
    assert tok.decode(matcher.consumed) == _CONTENT
