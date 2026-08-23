"""Next-turn retirement keys: predict what the client will replay.

A finished request's APC entry is only useful if the next turn's rendered
prompt starts with the stored key. Chat templates do not replay generated
text verbatim: thinking blocks are stripped or re-emitted with normalized
whitespace, and tool calls are re-serialized from the structured
``tool_calls`` field. A key built as ``full_ids + generated`` therefore
diverges from the replayed prefix at or before the first tool call, and
the entry can never match.

The fix is provenance, not guessing: the server-side capture (installed by
``server_patches.apc.install_retire_render_capture``) records each chat
request's processed messages and template kwargs keyed by the exact token
ids the request forwarded. At retirement, ``next_turn_lcp`` re-renders the
hypothetical next turn (messages plus this completion, parsed into the
same assistant-message shape the API response carries) through the same
template and tokenizer, and returns the longest common prefix with the
sequence we actually forwarded. When the request declares tools the
render appends no trailing message (a tool result may come next, and
strip-mode templates keep the chain's thinking for tool continuations);
without tools the next client message can only be a user turn, so the
render appends a dummy user probe and strip-mode templates apply their
strip, landing the LCP at the true replay boundary. Non-strip templates
diverge at the user header either way, so the probe changes nothing for
them.

Prediction errors are safe by construction: lookup still requires an exact
token-prefix match, so a wrong LCP costs reuse, never correctness.
"""

from __future__ import annotations

import collections
import json
import logging
import re
import threading

_log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_TEXT_MEMO: collections.OrderedDict = collections.OrderedDict()
_IDS_MEMO: collections.OrderedDict = collections.OrderedDict()
# Sized for the soak/prod ceiling: 8 sessions + 4 spike + cancels can
# hold >16 requests between submit and retire; an evicted ctx silently
# degrades retirement to a full-depth store.
_TEXT_MEMO_CAP = 64
_IDS_MEMO_CAP = 64

# Mirrors the endpoint's control-token scrub of post-tool-call text.
_CONTROL_TOKEN_RE = re.compile(r"<\|[^>]+\|>|<[^>]+>")


def register_render(text: str, ctx: dict) -> None:
    """Record a rendered chat prompt's context, keyed by the prompt text."""
    with _LOCK:
        _TEXT_MEMO[text] = ctx
        while len(_TEXT_MEMO) > _TEXT_MEMO_CAP:
            _TEXT_MEMO.popitem(last=False)


def register_ids(prompt_text: str, ids, preprocess) -> None:
    """Re-key a rendered prompt's context by its forwarded token ids.

    Called from the submit hop, where the prompt text and its tokenization
    coexist. ``preprocess`` is the generator's own text-to-ids callable so
    the retirement render tokenizes through the identical path (special
    tokens, model-type gates) as the live request did.
    """
    with _LOCK:
        ctx = _TEXT_MEMO.get(prompt_text)
        if ctx is None:
            return
        ctx = dict(ctx)
        ctx["preprocess"] = preprocess
        _IDS_MEMO[tuple(int(t) for t in ids)] = ctx
        while len(_IDS_MEMO) > _IDS_MEMO_CAP:
            _IDS_MEMO.popitem(last=False)


def lookup_render_ctx(full_ids) -> dict | None:
    """Fetch (without consuming) the render context for a request's ids."""
    with _LOCK:
        return _IDS_MEMO.get(tuple(int(t) for t in full_ids))


def _decode_generated(ctx: dict, generated: list[int]) -> str:
    processor = ctx["processor"]
    tokenizer = getattr(processor, "tokenizer", processor)
    text = tokenizer.decode(list(generated), skip_special_tokens=False)
    eos = getattr(tokenizer, "eos_token", None)
    stop = ctx.get("stop_token_text") or eos
    for tok in {stop, eos} - {None}:
        if tok and text.endswith(tok):
            text = text[: -len(tok)]
    return text


def _thinking_markers(kw: dict):
    """The same (open, close) marker pairs the response splitter tries,
    including its built-in defaults -- the request kwargs rarely carry
    explicit thinking tokens."""
    from mlx_vlm.server.responses_state import ThinkingStreamState
    return ThinkingStreamState._build_open_close_markers(
        kw.get("thinking_start_token"), kw.get("thinking_end_token"))


def _gen_prompt_text(ctx: dict) -> str:
    """The request's generation-prompt render (memoized on the ctx):
    qwen-style templates open the thinking block inside the prompt, so
    the generated text alone never shows the start marker."""
    text = ctx.get("_gen_prompt")
    if text is None:
        text = ""
        try:
            render = ctx.get("render")
            if render is not None:
                kw = dict(ctx.get("kw") or {})
                kw["add_generation_prompt"] = True
                out = render(ctx["processor"], ctx["config"],
                             list(ctx["messages"]), **kw)
                if isinstance(out, str):
                    text = out
        except Exception:
            text = ""
        ctx["_gen_prompt"] = text
    return text


def _virtually_finish(ctx: dict, text: str) -> str:
    """Close an open thinking block in a mid-decode partial.

    A partial prediction asks what the next turn would replay if the
    reply finished here. With no end marker in the text the splitter
    reads the whole partial as content, so every mid-thinking render
    diverged at the (empty) think block and the snapshot ring froze on
    its first tick -- the virtual closer restores the honest question.
    """
    try:
        markers = _thinking_markers(ctx.get("kw") or {})
    except Exception:
        return text
    for sm, em in markers:
        if em in text:
            return text
        if sm in text:
            return text + em
    prompt = _gen_prompt_text(ctx)
    for sm, em in markers:
        if prompt.rfind(sm) > prompt.rfind(em):
            return text + em
    return text


def truncated_thinking(text, pairs, prompt) -> bool:
    """Whether ``text`` is reasoning cut off inside a prompt-opened think
    block: no marker from any pair appears in the text, and ``prompt``
    ends inside an open block. Shared rule for the server's non-stream
    splitter and the retirement mirror -- each supplies the prompt from
    its own source (request contextvar vs. generation-prompt memo)."""
    if not text or not isinstance(prompt, str) or not prompt:
        return False
    for sm, em in pairs:
        if sm in text or em in text:
            return False
    return any(prompt.rfind(sm) > prompt.rfind(em) for sm, em in pairs)


_CHANNEL_HINTS = ("<|channel|>", "<|channel>", "<|start|>assistant")


def _channel_split(full_text: str):
    """Segment channel-marker reasoning (harmony analysis channels,
    gemma-style thought channels) that the tag-pair splitter cannot see.
    Harmony templates refuse messages whose content carries raw
    ``<|channel|>`` tags, so an unsplit echo makes every next-turn
    render prediction fail and retirement stores full-depth unmatchable
    keys. Returns (reasoning, content), or (None, None) when no channel
    marker is present or nothing classified as reasoning."""
    if not any(m in full_text for m in _CHANNEL_HINTS):
        return None, None
    from .reasoning import ReasoningFilter

    f = ReasoningFilter()
    spans = f.feed(full_text) + f.flush()
    reasoning = "".join(t for t, m in spans if m == "reason").strip()
    content = "".join(t for t, m in spans if m == "answer").strip()
    if not reasoning:
        return None, None
    return reasoning, content


def build_assistant_message(ctx: dict, full_text: str) -> dict:
    """Parse a completion into the assistant message a client echoes back.

    Mirrors the server's response construction: thinking split via the
    request's thinking tokens, tool calls parsed by the template-inferred
    tool module with arguments normalized to dicts (the endpoint
    json-decodes them before templating), and post-tool text scrubbed of
    control tokens.
    """
    from mlx_vlm.server import openai as srv

    kw = ctx.get("kw") or {}
    ts = kw.get("thinking_start_token")
    te = kw.get("thinking_end_token")
    reasoning, content = srv._split_thinking(full_text, ts, te)
    if reasoning is None:
        ch_reasoning, ch_content = _channel_split(full_text)
        if ch_reasoning is not None:
            reasoning, content = ch_reasoning, ch_content
    if reasoning is None and content and truncated_thinking(
            full_text, _thinking_markers(kw), _gen_prompt_text(ctx)):
        reasoning, content = content, ""

    tool_calls = None
    tools = kw.get("tools")
    parser = srv._infer_tool_parser_from_processor(ctx["processor"])
    tool_module = srv.load_tool_module(parser) if parser else None
    if tool_module is not None:
        tc = srv.process_tool_calls(
            model_output=full_text, tool_module=tool_module, tools=tools)
        if tc.get("calls"):
            tool_calls = []
            for call in tc["calls"]:
                call = dict(call) if isinstance(call, dict) else call
                if isinstance(call, dict) and "function" in call:
                    fn = dict(call["function"])
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            fn["arguments"] = {}
                    call["function"] = fn
                tool_calls.append(call)
            _, content = srv._split_thinking(tc.get("remaining_text") or "",
                                             ts, te)
            if content:
                content = _CONTROL_TOKEN_RE.sub("", content).strip()
            content = content or None

    msg: dict = {"role": "assistant", "content": content or ""}
    if reasoning:
        msg["reasoning_content"] = reasoning
        msg["reasoning"] = reasoning
        # Harmony templates read analysis text from this field and raise
        # on raw channel tags in content; think-tag templates ignore it.
        msg["thinking"] = reasoning
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _render_ids(ctx: dict, msgs: list) -> list[int] | None:
    """Render ``msgs`` through the request's own template and tokenizer
    (no generation prompt) and return the token ids. The single render
    path behind every prefix prediction; a second renderer diverging
    silently is how the turn-boundary bug stayed invisible."""
    render = ctx.get("render")
    preprocess = ctx.get("preprocess")
    if render is None or preprocess is None:
        return None
    kw = dict(ctx.get("kw") or {})
    kw["add_generation_prompt"] = False
    text = render(ctx["processor"], ctx["config"], msgs, **kw)
    if not isinstance(text, str):
        return None
    raw = preprocess(text)
    ids = raw["input_ids"] if isinstance(raw, dict) else raw
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


def predict_next_ids(ctx: dict, assistant_msg: dict | None) -> list[int] | None:
    """Render and tokenize the hypothetical next-turn prefix.

    ``assistant_msg=None`` renders the request's own messages with no
    reply appended -- the render-stable core a next turn extends."""
    msgs = list(ctx["messages"])
    if assistant_msg is not None:
        msgs.append(assistant_msg)
    return _render_ids(ctx, msgs)


def _lcp_len(seq, nxt) -> int:
    n = min(len(seq), len(nxt))
    lcp = 0
    while lcp < n and int(seq[lcp]) == int(nxt[lcp]):
        lcp += 1
    return lcp


def system_prefix_lcp(ctx: dict, prompt_ids) -> int | None:
    """Token length of the prefix every sibling request shares: the LCP
    of two probe renders that differ only in a dummy first user turn.

    Sibling fan-out requests share the system prompt and tool schemas
    but diverge at the first user message; this is the deepest position
    one checkpoint can serve all of them. A system-only render cannot
    measure it on every template (gemma folds the system prompt into
    the first user turn and renders a lone system message to almost
    nothing), so the offset comes from a divergence probe instead:
    render leading-system + user "0" and leading-system + user "1"
    through the same template, kwargs (tools included), and tokenizer
    as the live request, and take where they split. A folding template
    folds both probes identically, so the split lands exactly where
    real siblings diverge. Clamped by the LCP with the live
    prompt in case a template leaks user content into the header.
    Memoized on the ctx; media prompts return None (expanded media ids
    cannot be re-encoded from text)."""
    if not ctx or ctx.get("media"):
        return None
    memo = ctx.get("_p_system")
    if memo is not None:
        return memo if memo >= 0 else None
    lcp = None
    try:
        msgs = list(ctx.get("messages") or ())
        lead = []
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "system":
                lead.append(m)
            else:
                break
        # No leading system block, or nothing after it (a system-only
        # prompt is covered by the terminal checkpoint already).
        if lead and len(lead) < len(msgs):
            p1 = _render_ids(ctx, lead + [{"role": "user", "content": "0"}])
            p2 = _render_ids(ctx, lead + [{"role": "user", "content": "1"}])
            if p1 and p2:
                lcp = min(_lcp_len(p1, p2), _lcp_len(p1, prompt_ids))
    except Exception:
        _log.debug("system-prefix prediction failed", exc_info=True)
        lcp = None
    ctx["_p_system"] = -1 if lcp is None else lcp
    return lcp


# Model types whose prompt-stable prediction already failed once: the
# first failure warns with the traceback, repeats log at debug -- one
# broken template must not stack a warning onto every request. Keyed by
# model_type, not object identity: a recycled id would inherit
# "already warned" across a model swap.
_P_STABLE_WARNED: set[str] = set()


def prompt_stable_lcp(ctx: dict, prompt_ids) -> int | None:
    """Longest prefix of the live prompt render that survives a next-turn
    re-render.

    The same messages rendered without the generation prompt: positions
    past the LCP (the gen-prompt tail and any template-opened thinking
    block) are replaced when the client echoes the conversation back, so
    a checkpoint above it can never be adopted by turn 2. Memoized on the
    ctx; returns None when the context is missing, carries media, or
    prediction fails.
    """
    if not ctx or ctx.get("media"):
        return None
    memo = ctx.get("_p_stable")
    if memo is not None:
        return memo if memo >= 0 else None
    lcp = None
    try:
        if (ctx.get("kw") or {}).get("tools"):
            nxt = predict_next_ids(ctx, None)
        else:
            # Same demotion probe as next_turn_lcp: a toolless next
            # turn appends a user message, demoting the prompt's last
            # assistant so the template strips its think block. Without
            # it p_stable lands past think tokens the next render drops
            # (the 9B probe: every record diverged at the prior
            # assistant's think-open, position 358).
            nxt = _render_ids(ctx, list(ctx["messages"]) + [
                {"role": "user", "content": "0"}])
        if nxt:
            seq = [int(t) for t in prompt_ids]
            n = min(len(seq), len(nxt))
            lcp = 0
            while lcp < n and seq[lcp] == nxt[lcp]:
                lcp += 1
    except Exception:
        cfg = ctx.get("config")
        key = str(getattr(cfg, "model_type", None)
                  or (cfg.get("model_type") if isinstance(cfg, dict) else None)
                  or type(cfg).__name__)
        if key in _P_STABLE_WARNED:
            _log.debug("prompt-stable prediction failed", exc_info=True)
        else:
            _P_STABLE_WARNED.add(key)
            _log.warning(
                "prompt-stable prediction failed; turn-boundary reuse is "
                "off for this model (repeats log at debug)", exc_info=True)
        lcp = None
    ctx["_p_stable"] = -1 if lcp is None else lcp
    return lcp


def next_turn_lcp(ctx: dict, seq: list[int], generated: list[int],
                  *, partial: bool = False) -> int | None:
    """Longest common prefix of ``seq`` with the predicted next-turn render.

    ``partial=True`` marks a mid-decode call (the decode-time snapshot
    tick): an open thinking block is virtually closed before the split,
    since the finished reply this predicts for would close it. At
    retirement the text is what the client will actually echo, so no
    closer is applied. Returns None (caller keeps today's behavior)
    when the context is missing, carries media (re-encoding text cannot
    reproduce expanded media token ids), or any step of the prediction
    fails.
    """
    if not ctx or ctx.get("media"):
        return None
    try:
        full_text = _decode_generated(ctx, generated)
        if partial:
            full_text = _virtually_finish(ctx, full_text)
        msg = build_assistant_message(ctx, full_text)
        if (ctx.get("kw") or {}).get("tools"):
            nxt = predict_next_ids(ctx, msg)
        else:
            # No tools declared: the next message can only be a user
            # turn. The dummy probe makes strip-mode templates apply
            # their think-strip, so the LCP lands where the replay
            # actually diverges instead of at the full stored chain.
            nxt = _render_ids(ctx, list(ctx["messages"]) + [
                msg, {"role": "user", "content": "0"}])
        if not nxt:
            return None
        n = min(len(seq), len(nxt))
        lcp = 0
        while lcp < n and seq[lcp] == nxt[lcp]:
            lcp += 1
        return lcp
    except Exception:
        _log.warning("retire-key prediction failed; storing under the "
                     "forwarded key", exc_info=True)
        return None


def _reset_for_tests() -> None:
    with _LOCK:
        _TEXT_MEMO.clear()
        _IDS_MEMO.clear()
