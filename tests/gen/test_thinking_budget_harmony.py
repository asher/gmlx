"""Thinking budget on harmony channels (Beam, gpt-oss), with the real Beam
token ids. No model is needed.

The generation prompt ends at ``<|start|>assistant``, so the model writes
the whole analysis header itself. The family markers are that header and
``<|end|>``. The model is a script: it writes its text until a close is
forced, then starts its ``after`` text over at each forced close.

The three budget paths are the CLI logits processor, the MTP round hook
and the serve criteria. Each must cap analysis only, take a bare new header
or ``<|constrain|>`` as the end of a block, let a final answer or a tool
call through, and stand down on a model that keeps reopening. With no tools
declared, a forced close also opens the final channel.
"""

from __future__ import annotations

import pytest

import beam_tokens as bt  # noqa: E402
import mlx.core as mx  # noqa: E402

from gmlx.gen.thinking_budget import (
    BUDGET_WRAP_PHRASE,
    AnswerGate,
    _thinking_token_seqs,
    answer_opener_seqs,
    has_answer_header,
    make_mtp_finish_hook,
    make_thinking_budget_processor,
    seq_thinking_budget_criteria_cls,
)

_PROMPT = "<|start|>user<|message|>hi<|end|><|start|>assistant"
_REASON = "<|channel|>analysis<|message|>Thinking hard about it now.<|end|>"
_STEER = "<|start|>assistant<|channel|>final<|message|>"
_ANSWER = "Hello world foo bar"
_FINAL = _STEER + _ANSWER
_REOPEN = ("<|start|>assistant<|channel|>analysis<|message|>"
           "Again more thought here.<|end|>")
_TOOL = ("<|start|>assistant to=bash<|channel|>analysis <|constrain|>xml"
         '<|message|><rfl:parameter name="command">\nls -la\n</rfl:parameter>')
_STAND_DOWN = "[thinking] budget stood down after {} forced closes"


def _tok():
    return bt.BeamTokenizer()


def _markers():
    return {"start_token": bt.START, "end_token": bt.END}


def _after(after, tok):
    """The script for each forced close in turn. A string is the script
    for every close; a list gives one per close, the last one repeating."""
    scripts = [after] if isinstance(after, str) else list(after)
    n = 0
    while True:
        yield iter(tok.encode(scripts[min(n, len(scripts) - 1)]))
        n += 1


def _one_hot_id(out):
    row = out[0]
    n_open = int(mx.sum(row > -1e8).item())
    return int(mx.argmax(row).item()) if n_open == 1 else None


def _cli_run(p, script, after, tok):
    """Drive the logits processor the way mlx-lm does."""
    tokens = [tok.encode(_PROMPT)[-1]]
    src, afters = iter(tok.encode(script)), _after(after, tok)
    emitted, forcing = [], False
    logits = mx.zeros((1, bt.VOCAB))
    out = p(mx.array(tokens), logits)
    while len(emitted) < 600:
        forced = _one_hot_id(out)
        if forced is not None:
            forcing = True
            tid = forced
        else:
            if forcing:
                forcing = False
                src = next(afters)
            tid = next(src, None)
            if tid is None:
                break
        emitted.append(tid)
        tokens.append(tid)
        out = p(mx.array(tokens), logits)
    return tok.decode(emitted)


def _mtp_run(h, script, after, tok):
    """Drive the MTP hook with a round boundary before every token."""
    src, afters = iter(tok.encode(script)), _after(after, tok)
    emitted = []
    while len(emitted) < 600:
        seq = h.take_forced()
        if seq is not None:
            for t in seq:
                h.observe(t)
                emitted.append(t)
            src = next(afters)
            continue
        tid = next(src, None)
        if tid is None:
            break
        h.observe(tid)
        emitted.append(tid)
    return tok.decode(emitted)


def _serve_criteria(tok, budget, tools=False):
    cls = seq_thinking_budget_criteria_cls()
    return cls(tok, budget, thinking_end_token=bt.END,
               thinking_start_token=bt.START, enable_thinking=True,
               prompt_open_thinking=False,
               prompt_ids=tok.encode(_PROMPT)[-16:], tools=tools)


def _serve_run(crit, script, after, tok):
    """Drive the criteria as the batched decode loop does."""
    src, afters = iter(tok.encode(script)), _after(after, tok)
    emitted, nxt, forcing = [], None, False
    while len(emitted) < 600:
        if nxt is not None:
            tid, forcing = nxt, True
        else:
            if forcing:
                forcing = False
                src = next(afters)
            tid = next(src, None)
            if tid is None:
                break
        emitted.append(tid)
        crit(tid)
        nxt = crit.pop_forced_token_id()
    return tok.decode(emitted)


def test_the_beam_fixture_tokenizer_matches():
    tok = _tok()
    for text, real in bt.TABLE["encodings"].items():
        assert tok.encode(text) == real, text


def test_beam_resolves_the_harmony_pair_not_think_tags():
    """Beam's vocab holds <think> and </think> as single tokens, and Beam
    never uses them. The template names the harmony pair."""
    tok = _tok()
    start, end = _thinking_token_seqs(tok)
    assert tok.decode(start) == "<|channel|>analysis<|message|>"
    assert tok.decode(end) == "<|end|>"
    start, end = _thinking_token_seqs(tok, **_markers())
    assert tok.decode(start) == bt.START


def test_family_markers_for_beam_and_gpt_oss():
    from gmlx.gen.profiles import detect_family, family_base

    for arch in ("beam", "gpt-oss"):
        s = family_base(detect_family(arch))["sampling"]
        assert s["thinking_start_token"] == bt.START, arch
        assert s["thinking_end_token"] == bt.END, arch
        assert has_answer_header(s["thinking_start_token"]), arch


@pytest.mark.parametrize("budget, cut", [
    (0, "<|channel|>analysis<|message|>Thinking\n<|end|>"),
    (3, "<|channel|>analysis<|message|>Thinking hard about it"
        + BUDGET_WRAP_PHRASE + "<|end|>"),
])
def test_budget_caps_analysis_then_the_answer_runs(budget, cut):
    tok = _tok()
    p = make_thinking_budget_processor(tok, budget, start_in_thinking=False,
                                       **_markers())
    text = _cli_run(p, _REASON + _FINAL, _ANSWER, tok)
    assert text == cut + _STEER + _ANSWER
    assert p._spent and not p.done and not p._answer_pending


def test_mtp_and_serve_closes_open_the_final_channel():
    tok = _tok()
    cut = "<|channel|>analysis<|message|>Thinking\n<|end|>"
    h = make_mtp_finish_hook(tok, budget=0, prompt_ids=tok.encode(_PROMPT),
                             **_markers())
    assert _mtp_run(h, _REASON + _FINAL, _ANSWER, tok) == cut + _STEER + _ANSWER
    assert not h._answer_pending and h._strikes == 0
    crit = _serve_criteria(tok, 0)
    assert _serve_run(crit, _REASON + _FINAL, _ANSWER, tok) == cut + _STEER + _ANSWER
    assert not crit._answer_pending and crit._strikes == 0


@pytest.mark.parametrize("path", ["cli", "mtp", "serve"])
def test_with_tools_declared_the_close_is_end_only(path):
    """The model may answer or call a tool after the close, so it writes
    no header."""
    tok = _tok()
    cut = "<|channel|>analysis<|message|>Thinking\n<|end|>"
    if path == "cli":
        b = make_thinking_budget_processor(tok, 0, start_in_thinking=False,
                                           tools=True, **_markers())
        text = _cli_run(b, _REASON + _TOOL, _TOOL, tok)
    elif path == "mtp":
        b = make_mtp_finish_hook(tok, budget=0, prompt_ids=tok.encode(_PROMPT),
                                 tools=True, **_markers())
        text = _mtp_run(b, _REASON + _TOOL, _TOOL, tok)
    else:
        b = _serve_criteria(tok, 0, tools=True)
        text = _serve_run(b, _REASON + _TOOL, _TOOL, tok)
    assert text == cut + _TOOL
    assert b._strikes == 0 and not b.done


def test_think_tag_formats_get_no_steer():
    from gmlx.gen.thinking_budget import answer_steer_ids

    tok = _tok()
    assert answer_steer_ids(tok, tok.encode(bt.START)) == tok.encode(_STEER)
    assert answer_steer_ids(tok, tok.encode("<|start|>")) == []
    assert answer_steer_ids(tok, ()) == []


@pytest.mark.parametrize("close", ["<|start|>assistant", "<|constrain|>"])
def test_a_bare_header_or_constrain_ends_the_count(close):
    """llama.cpp's harmony parser ends analysis there too. The block ends
    naturally, so the budget disarms and the answer runs."""
    tok = _tok()
    p = make_thinking_budget_processor(tok, 6, start_in_thinking=False,
                                       **_markers())
    head = "<|channel|>analysis<|message|>Thinking hard"
    rest = "" if close == "<|start|>assistant" else "<|start|>assistant"
    script = head + close + rest + "<|channel|>final<|message|>Hello world foo bar"
    assert _cli_run(p, script, "", tok) == script
    assert p.done and not p._spent


def test_constrain_in_a_tool_header_is_not_a_close():
    """Outside a block, <|constrain|> is the tool-call constraint. It ends
    nothing and takes no strike."""
    tok = _tok()
    p = make_thinking_budget_processor(tok, 0, start_in_thinking=False,
                                       tools=True, **_markers())
    text = _cli_run(p, _REASON + _TOOL, _TOOL, tok)
    assert text.endswith(_TOOL)
    assert p._strikes == 0 and not p.done


@pytest.mark.parametrize("after", [_FINAL, _TOOL])
def test_answer_or_tool_call_after_a_forced_close_takes_no_strike(after):
    tok = _tok()
    p = make_thinking_budget_processor(tok, 0, start_in_thinking=False,
                                       tools=True, **_markers())
    _cli_run(p, _REASON + after, after, tok)
    assert p._strikes == 0 and not p.done and not p._answer_pending


def test_tool_call_parses_after_a_forced_close():
    from gmlx._vendor.mlx_vlm.tools.extractor import process_tool_calls
    from gmlx.models.beam import tools

    tok = _tok()
    p = make_thinking_budget_processor(tok, 0, start_in_thinking=False,
                                       tools=True, **_markers())
    text = _cli_run(p, _REASON + _TOOL, _TOOL, tok)
    calls = process_tool_calls(text, tools, None).calls
    assert [c["function"]["name"] for c in calls] == ["bash"]
    assert calls[0]["function"]["arguments"] == '{"command": "ls -la"}'


def test_cli_recloses_one_reopen_and_stays_armed():
    tok = _tok()
    p = make_thinking_budget_processor(tok, 0, start_in_thinking=False,
                                       **_markers())
    head = "<|start|>assistant<|channel|>analysis<|message|>"
    text = _cli_run(p, _REASON + _FINAL, [_REOPEN + _ANSWER, _ANSWER], tok)
    # The reopen is closed on the spot, before any of its thought.
    assert head + "\n<|end|>" + _STEER in text
    assert "Again" not in text
    assert text.endswith(_FINAL)
    assert p._forced_closes == 2 and not p.done


def test_mtp_recloses_one_reopen_and_stays_armed():
    tok = _tok()
    h = make_mtp_finish_hook(tok, budget=0, prompt_ids=tok.encode(_PROMPT),
                             **_markers())
    text = _mtp_run(h, _REASON + _FINAL, [_REOPEN + _ANSWER, _ANSWER], tok)
    assert h._forced_closes == 2 and not h.done
    assert text.endswith(_FINAL)


def test_serve_recloses_one_reopen_and_stays_armed():
    tok = _tok()
    crit = _serve_criteria(tok, 0)
    text = _serve_run(crit, _REASON + _FINAL, [_REOPEN + _ANSWER, _ANSWER], tok)
    assert crit._forced_closes == 2 and not crit.done
    assert text.endswith(_FINAL)


@pytest.mark.parametrize("path, closes", [("cli", 4), ("mtp", 3), ("serve", 4)])
def test_a_model_that_keeps_reopening_stands_the_budget_down(capsys, path,
                                                             closes):
    """Three strikes on every path. The CLI and serve write the fourth close
    and then stand down; MTP stands down in place of it."""
    tok = _tok()
    if path == "cli":
        b = make_thinking_budget_processor(tok, 0, start_in_thinking=False,
                                           **_markers())
        _cli_run(b, _REASON, _REOPEN, tok)
    elif path == "mtp":
        b = make_mtp_finish_hook(tok, budget=0, prompt_ids=tok.encode(_PROMPT),
                                 **_markers())
        _mtp_run(b, _REASON, _REOPEN, tok)
    else:
        b = _serve_criteria(tok, 0)
        _serve_run(b, _REASON, _REOPEN, tok)
    assert b.done
    assert b._strikes == 3
    assert b._forced_closes == closes
    err = capsys.readouterr().err
    assert err.count("budget stood down") == 1
    assert _STAND_DOWN.format(closes) in err


def test_serve_soft_close_ends_the_block_without_a_strike():
    tok = _tok()
    crit = _serve_criteria(tok, 6)
    script = ("<|channel|>analysis<|message|>Thinking hard<|start|>assistant"
              "<|channel|>final<|message|>Hello world foo bar")
    assert _serve_run(crit, script, "", tok) == script
    assert not crit.in_thinking and crit._strikes == 0
    assert crit._forced_closes == 0


def test_response_format_gate_opens_at_the_final_header():
    """The answer is the final channel, with or without a JSON constraint.
    The gate must not open at the analysis close."""
    tok = _tok()
    seqs = answer_opener_seqs(tok, bt.START, bt.END)
    for header in ("<|channel|>final<|message|>",
                   "<|channel|>final <|constrain|>json<|message|>"):
        gate = AnswerGate(seqs, tok.encode(_PROMPT)[-8:])
        stream = tok.encode(_REASON + "<|start|>assistant" + header)
        hit = gate.advance(stream)
        assert hit == len(stream) - 1, header
        assert gate.restart(tok.encode(_PROMPT)[-8:]).find(
            tok.encode(_REASON)) is None
