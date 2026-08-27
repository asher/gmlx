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


def test_prompt_stable_demotes_last_assistant():
    # Boundary placement must see the future demotion: with a prior
    # assistant carrying reasoning in the messages, p_stable ends
    # before its think block, not at the prompt tail.
    ctx = _ctx()
    ctx["messages"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok", "reasoning_content": "xy"},
        {"role": "user", "content": "more"},
    ]
    # Live prompt render: assistant is not last -> think stripped by
    # _strip_render, but simulate the normalizer keeping it for the
    # last assistant by rendering the prompt with think present.
    prompt = "U:hi;A:<think>xy</think>ok;U:more;A:"
    prompt_ids = _ords(prompt)
    lcp = retire_key.prompt_stable_lcp(ctx, prompt_ids)
    # Demoted render "U:hi;A:ok;U:more;U:0;" diverges at the think.
    assert lcp == len("U:hi;A:")


def test_keep_template_caps_at_think_via_client_echo():
    # Keep-mode template (dwarfstar deepseek4 chat-v2 class): renders
    # attached reasoning verbatim wherever it appears. Standard clients
    # never resend reasoning_content (the DeepSeek API contract), so the
    # probe echoes content only and the next turn diverges at the think
    # block, not at the user header. Predicting len(seq) here stored a
    # dead full-depth chain every turn on the live DSv4-Flash serve.
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
    assert lcp == len("U:hi;A:")


def test_preserve_thinking_kwarg_keeps_reasoning_in_echo():
    # A truthy preserve_thinking template kwarg declares the
    # keep-reasoning protocol: the client resends reasoning_content and
    # the template renders it, so the probe echoes the fields and the
    # full generation replays.
    def keep_render(processor, config, msgs, **kw):
        out = []
        for m in msgs:
            if m.get("role") == "user":
                out.append(f"U:{m['content']};")
            elif m.get("role") == "assistant":
                think = (f"<think>{m['reasoning_content']}</think>"
                         if kw.get("preserve_thinking")
                         and m.get("reasoning_content") else "")
                out.append(f"A:{think}{m.get('content') or ''};")
        if kw.get("add_generation_prompt"):
            out.append("A:")
        return "".join(out)

    ctx = _ctx(preserve_thinking=True)
    ctx["render"] = keep_render
    seq, gen = _seq_and_gen()
    lcp = retire_key.next_turn_lcp(ctx, seq, gen)
    assert lcp == len(seq)

    # Flag present but false: standard echo, strip prediction holds.
    ctx = _ctx(preserve_thinking=False)
    ctx["render"] = keep_render
    lcp = retire_key.next_turn_lcp(ctx, seq, gen)
    assert lcp == len("U:hi;A:")
