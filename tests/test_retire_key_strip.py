"""next_turn_lcp under strip-mode templates (qwen3.5-class).

The 122B edge cert failed on pool churn: retirement stored full-depth
chains because the no-trailing-message render keeps the thinking block
(strip only fires once a later user query arrives), so every chat next
turn diverged at the strip point and the stored suffix was dead weight.
Without tools the prediction must append a dummy user probe so the
template strips and the LCP lands at the replay boundary.
"""

from types import SimpleNamespace

from gmlx import retire_key


def _ords(s):
    return [ord(c) for c in s]


def _strip_render(processor, config, msgs, **kw):
    """Qwen-shaped toy template: assistant thinking survives only when
    no later user message follows."""
    out = []
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "user":
            out.append(f"U:{m['content']};")
        elif role == "assistant":
            later_user = any(
                x.get("role") == "user" for x in msgs[i + 1:])
            think = ""
            if not later_user and m.get("reasoning_content"):
                think = f"<think>{m['reasoning_content']}</think>"
            out.append(f"A:{think}{m.get('content') or ''};")
    if kw.get("add_generation_prompt"):
        out.append("A:")
    return "".join(out)


def _ctx(**kw_extra):
    gen_text = "<think>xy</think>ok"
    tok = SimpleNamespace(
        decode=lambda t, **k: gen_text, eos_token=None)
    return {
        "render": _strip_render,
        "preprocess": _ords,
        "processor": SimpleNamespace(tokenizer=tok),
        "config": None,
        "messages": [{"role": "user", "content": "hi"}],
        "kw": dict(kw_extra),
    }


def _seq_and_gen():
    prompt = "U:hi;A:"
    gen_text = "<think>xy</think>ok"
    return _ords(prompt) + _ords(gen_text), _ords(gen_text)


def test_no_tools_caps_at_strip_point():
    seq, gen = _seq_and_gen()
    lcp = retire_key.next_turn_lcp(_ctx(), seq, gen)
    # Next-turn render "U:hi;A:ok;U:0;" diverges from the stored
    # think-bearing chain right after the assistant header.
    assert lcp == len("U:hi;A:")


def test_tools_present_keeps_tool_continuation_render():
    seq, gen = _seq_and_gen()
    lcp = retire_key.next_turn_lcp(
        _ctx(tools=[{"type": "function"}]), seq, gen)
    # No trailing message: the toy template keeps the think block, so
    # the chain matches through the whole generation.
    assert lcp == len(seq)


def test_non_strip_template_unchanged_by_probe():
    def keep_render(processor, config, msgs, **kw):
        out = []
        for m in msgs:
            if m.get("role") == "user":
                out.append(f"U:{m['content']};")
            elif m.get("role") == "assistant":
                think = (f"<think>{m['reasoning_content']}</think>"
                         if m.get("reasoning_content") else "")
                out.append(f"A:{think}{m.get('content') or ''};")
        if kw.get("add_generation_prompt"):
            out.append("A:")
        return "".join(out)

    ctx = _ctx()
    ctx["render"] = keep_render
    seq, gen = _seq_and_gen()
    lcp = retire_key.next_turn_lcp(ctx, seq, gen)
    # Divergence at the user header: the full generation still matches.
    assert lcp == len(seq)
