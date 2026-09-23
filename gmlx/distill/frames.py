"""Row frames for the teacher pass: how a row's bytes are placed relative to a
chat template (continue, chat, reply, reply-think), and which positions of
the row are targets."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import numpy as np

from gmlx.load.tokenizer import hf_inner

from .align import Alignment
from .constants import FRAME_KINDS  # noqa: F401
from .tokens import adds_bos, encode_with_byte_ends

# ---------------------------------------------------------------------------
# framed rows: a conversation rendered through a tokenizer's chat template,
# with targets on the assistant spans only
# ---------------------------------------------------------------------------

CONTINUE_INSTRUCTION = "Continue the following text."


def parse_render_kwargs(spec: str | None) -> dict:
    """--frame-kwargs value: a JSON object, or a path to a JSON file."""
    if not spec:
        return {}
    text = spec
    if not spec.lstrip().startswith("{"):
        p = Path(spec).expanduser()
        text = p.read_text(encoding="utf-8") if p.is_file() else spec
    kw = json.loads(text)
    if not isinstance(kw, dict):
        raise ValueError("--frame-kwargs must be a JSON object")
    return kw


def has_chat_template(tokenizer) -> bool:
    return bool(getattr(hf_inner(tokenizer), "chat_template", None))


def set_render_kwargs(tokenizer, kwargs: dict | None) -> None:
    """Attach the keyword arguments every template render of this tokenizer
    passes to apply_chat_template (enable_thinking, reasoning_effort,
    date_string, ...). Each side of a pair carries its own: the teacher's
    live in the cache manifest's frame block, the student's in view.json."""
    hf_inner(tokenizer).distill_render_kwargs = dict(kwargs or {})


def render_kwargs(tokenizer) -> dict:
    return dict(getattr(hf_inner(tokenizer), "distill_render_kwargs", None) or {})


def resolve_render_kwargs(tokenizer, inherit: dict | None = None, override: dict | None = None) -> dict:
    """A side's render settings: the stable defaults for its template, then
    every inherited setting this template reads (the thinking switch a
    cache was rendered with, the date pinned to the other side's), then
    the user's overrides."""
    kw = default_render_kwargs(tokenizer, date=(inherit or {}).get("date_string"))
    tpl = getattr(hf_inner(tokenizer), "chat_template", None) or ""
    kw.update({k: v for k, v in (inherit or {}).items() if k in tpl})
    kw.update(override or {})
    return kw


def default_render_kwargs(tokenizer, date: str | None = None) -> dict:
    """Render settings that keep a template's output stable across days:
    templates that read date_string (Llama 3.x) get today's date pinned in
    the Llama format, so the cached render and every later re-render of the
    same conversation agree byte for byte."""
    tpl = getattr(hf_inner(tokenizer), "chat_template", None) or ""
    kw: dict = {}
    if "date_string" in tpl:
        kw["date_string"] = date or today_string()
    return kw


def today_string() -> str:
    """Today in the Llama date format."""
    import datetime
    return datetime.date.today().strftime("%d %b %Y")


def _fold_system(messages: list[dict]) -> list[dict] | None:
    """The system message folded into the first user turn, for templates
    that reject the system role; None when there is nothing to fold."""
    if not messages or messages[0].get("role") != "system" or len(messages) < 2:
        return None
    if messages[1].get("role") != "user":
        return None
    sys_c = (messages[0].get("content") or "").strip()
    user = dict(messages[1])
    user["content"] = (sys_c + "\n\n" + (user.get("content") or "")) if sys_c else user.get("content", "")
    return [user] + list(messages[2:])


def apply_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool = False, **extra) -> str:
    """apply_chat_template with the tokenizer's render settings applied.
    A template that rejects the system role gets the system message folded
    into the first user turn; any other template error (a tool role the
    template cannot render, roles that do not alternate) is raised as a
    ValueError so the caller can drop and count the conversation."""
    inner = hf_inner(tokenizer)
    kw = dict(render_kwargs(tokenizer))
    kw.update(extra)
    try:
        return inner.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)
    except Exception as e:  # jinja TemplateError, ValueError from the template
        folded = _fold_system(messages)
        if folded is not None:
            try:
                return inner.apply_chat_template(folded, tokenize=False,
                                                 add_generation_prompt=add_generation_prompt, **kw)
            except Exception as e2:
                raise ValueError(f"template rejects the conversation: {str(e2)[:120]}") from None
        raise ValueError(f"template rejects the conversation: {str(e)[:120]}") from None


def render_frame(tokenizer, messages: list[dict]) -> str:
    """The frame an assistant reply is written behind: the way the template
    renders a completed assistant turn after messages, everything before
    the content, which is the template's own record of a reply given
    without reasoning (harmony's `<|channel|>final<|message|>`, GLM's,
    granite's and Qwen's closed empty thinking block). When that prefix is
    a strict prefix of the generation prompt the generation prompt wins,
    since the template then puts a preamble in front of every reply at
    inference (gemma-4's empty thought channel). A tokenizer without a
    chat template renders the messages as plain text, contents separated
    by blank lines."""
    if not has_chat_template(tokenizer):
        return _plain_render(tokenizer, messages, gen_prompt=True)
    gen = apply_template(tokenizer, messages, add_generation_prompt=True)
    done = apply_template(tokenizer, list(messages) + [{"role": "assistant", "content": _PROBE}])
    k = done.rfind(_PROBE)
    if k < 0:
        return gen
    pre = done[:k]
    if len(pre) < len(gen) and gen.startswith(pre):
        return gen
    return pre


FRAME_PREFIX_KINDS = ("continue", "model")


def frame_prefix(tokenizer, kind: str, instruction: str = CONTINUE_INSTRUCTION) -> str:
    """A bpb window prefix rendered from the tokenizer's own template, with
    no leading BOS (the scorer adds it): "continue" is the frame over one
    user turn holding the instruction, "model" is the model-turn header
    alone."""
    inner = hf_inner(tokenizer)
    if kind == "continue":
        frame = render_frame(tokenizer, [{"role": "user", "content": instruction}])
    elif kind == "model":
        # the header is what the frame adds beyond the user turn rendered
        # on its own, whatever the template ends that turn with
        msgs = [{"role": "user", "content": _PROBE}]
        r = render_frame(tokenizer, msgs)
        if has_chat_template(tokenizer):
            plain = apply_template(tokenizer, msgs)
        else:
            plain = _plain_render(tokenizer, msgs, gen_prompt=False)
        if len(r) > len(plain) and r.startswith(plain):
            frame = r[len(plain):]
        else:
            k = r.rfind(_PROBE)
            nl = r.find("\n", k)
            frame = r[nl + 1:] if nl >= 0 else r[k + len(_PROBE):]
    else:
        raise ValueError(f"unknown frame prefix kind {kind!r}")
    bos = getattr(inner, "bos_token", None)
    if bos and frame.startswith(bos):
        frame = frame[len(bos):]
    return frame


def continue_messages(text: str, instruction: str = CONTINUE_INSTRUCTION) -> list[dict]:
    return [{"role": "user", "content": instruction}, {"role": "assistant", "content": text}]


def _plain_render(tokenizer, messages: list[dict], *, gen_prompt: bool) -> str:
    """No-template render: BOS (when the tokenizer adds one), then the
    message contents separated by blank lines, each assistant content
    followed by the EOS string; with gen_prompt a blank line closes the
    text so a reply can follow."""
    inner = hf_inner(tokenizer)
    bos = getattr(inner, "bos_token", None) if adds_bos(tokenizer) else None
    eos = getattr(inner, "eos_token", None) or ""
    parts = []
    for m in messages:
        c = m.get("content") or ""
        if not c.strip():
            continue
        if m.get("role") == "assistant":
            c = c + eos
        parts.append(c)
    text = "\n\n".join(parts)
    if gen_prompt and parts:
        text += "\n\n"
    return (bos or "") + text


def render_row(tokenizer, messages: list[dict], *, open_tail: bool,
               close_tail: bool = False, last_only: bool = False,
               reason_target: bool = False) -> tuple[bytes, list[tuple[int, int, int]]]:
    """Render a conversation and return (text bytes, target spans).

    A span is (content start, content end, target end): the content bytes
    are identical in every render of the conversation and align across
    tokenizers, the target end extends over the end-of-turn marker that
    follows the content in this render (equal to the content end on an
    open tail).

    With open_tail the last message is an assistant turn rendered as frame
    + content: render_frame over the earlier messages, then the content
    verbatim with no end-of-turn marker (the continue framing of a raw text
    window). With open_tail and close_tail together the same frame + content
    render is closed by the template's own turn-end marker (the first of
    assistant_tails) and the span's target end extends over it, so the
    position after the content predicts the stop: the continue framing of
    a window that ends its document. Otherwise the whole conversation goes through the template and
    every assistant content span found in the render is a target, extended
    over the turn-end marker that follows it (assistant_tails); a last
    turn the template leaves unmarked (GLM) gets the marker appended;
    with last_only only the final assistant turn's span is kept (a reply
    row) and every earlier turn is context. A reasoning_content the
    template renders before a content is skipped over, never a target,
    unless reason_target is set: then the span starts at the trace, so
    the trace, the closing marker the template puts after it and the
    content are all targets behind the thinking-mode generation prompt
    (the frame is then `<think>` plus its newline on the Qwen family).
    Raises ValueError when the template rejects the conversation or an
    assistant content cannot be located in the render (a template that
    rewrites content). A token that straddles a span edge is never a
    target, so a frame ending in a space that merges with the content's
    first token loses that token as a target; the chat templates in use
    end their frame with a newline or a special token."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("a framed row ends with an assistant message")
    if open_tail:
        frame = render_frame(tokenizer, messages[:-1])
        content = messages[-1]["content"]
        fb = frame.encode("utf-8")
        b1 = len(fb) + len(content.encode("utf-8"))
        if close_tail:
            tails = assistant_tails(tokenizer)
            tail = (tails[0] if tails else "").encode("utf-8")
            return fb + content.encode("utf-8") + tail, [(len(fb), b1, b1 + len(tail))]
        return fb + content.encode("utf-8"), [(len(fb), b1, b1)]
    if has_chat_template(tokenizer):
        rendered = apply_template(tokenizer, messages)
    else:
        rendered = _plain_render(tokenizer, messages, gen_prompt=False)
    tails = assistant_tails(tokenizer)
    spans = []
    cursor = 0
    for i, m in enumerate(messages):
        c = (m.get("content") or "")
        if m.get("role") != "assistant":
            if c.strip():
                k = rendered.find(c.strip(), cursor)
                if k >= 0:
                    cursor = k + len(c.strip())
            continue
        cs = c.strip()
        if not cs:
            continue   # a tool-call-only turn: context, not a target
        rc = (m.get("reasoning_content") or "").strip()
        k0 = None
        # the markup around a reply holds the reply's own characters (a
        # reply of "a" sits inside "assistant", "hi" inside "</think>"), so
        # the content is located by rendering the conversation once more
        # with this content replaced by a probe: everything before it
        # renders the same, and the probe's position is the content's
        k = _probe_start(tokenizer, messages, i, rendered, c, cursor)
        if k is not None:
            if rc and reason_target:
                kr = rendered.rfind(rc, cursor, k)
                if kr >= 0:
                    k0 = kr
        else:
            # a template that renders the turn differently around a probe
            # (one that splits the content at a think tag): search after
            # the header the template renders for the turn
            cursor = max(cursor, _header_end(tokenizer, messages[:i], rendered))
            if rc:
                kr = rendered.find(rc, cursor)
                if kr >= 0:
                    cursor = kr + len(rc)
                    if reason_target:
                        k0 = kr
            k = rendered.find(cs, cursor)
            if k < 0:
                raise ValueError("assistant content not found in the rendered conversation")
        end = k + len(cs)
        b0 = len(rendered[:k if k0 is None else k0].encode("utf-8"))
        b1 = len(rendered[:end].encode("utf-8"))
        te = _tail_end(rendered, end, tails)
        if te is not None:
            end = te
        elif tails and not rendered[end:].strip():
            rendered = rendered + tails[-1]
            end = len(rendered)
        b2 = len(rendered[:end].encode("utf-8"))
        spans.append((b0, b1, b2))
        cursor = end
    if not spans:
        raise ValueError("no assistant span in the conversation")
    if last_only:
        spans = spans[-1:]
    return rendered.encode("utf-8"), spans


def _probe_start(tokenizer, messages: list[dict], i: int, rendered: str, content: str,
                 cursor: int) -> int | None:
    """Where the content of turn ``i`` starts in ``rendered``: the
    conversation is rendered once more with that content replaced by a
    probe, and when everything before the probe matches the render, the
    probe's position is the content's (after any leading whitespace the
    template keeps). None when the template renders the turn differently
    around the probe or the render does not hold the content there."""
    probe = list(messages)
    probe[i] = dict(messages[i], content=_PROBE)
    try:
        if has_chat_template(tokenizer):
            r2 = apply_template(tokenizer, probe)
        else:
            r2 = _plain_render(tokenizer, probe, gen_prompt=False)
    except ValueError:
        return None
    p = r2.find(_PROBE, cursor)
    if p < 0 or r2[:p] != rendered[:p]:
        return None
    cs = content.strip()
    for q in (p, p + len(content) - len(content.lstrip())):
        if rendered.startswith(cs, q):
            return q
    return None


def _header_end(tokenizer, prior: list[dict], rendered: str) -> int:
    """Where the assistant turn after ``prior`` can start in ``rendered``:
    the length of the frame the template renders a reply behind
    (``render_frame``) when the render carries it whole, else the common
    prefix of the two backed off to the last whitespace or marker close,
    0 when the template cannot render the frame."""
    try:
        hdr = render_frame(tokenizer, prior)
    except ValueError:
        return 0
    h = len(os.path.commonprefix([hdr, rendered]))
    # a frame can leave the render before its end (a think block whose
    # content the template moves, a reply that starts with a tag), and
    # the common prefix then ends inside the reply: back off to the last
    # boundary the frame renders; a frame carried whole ends where it ends,
    # marker or not ("[/INST]", "<|message|>")
    if h < len(hdr):
        while h > 0 and not (rendered[h - 1].isspace() or rendered[h - 1] == ">"):
            h -= 1
    return h


def _tail_end(rendered: str, end: int, tails: list[str]) -> int | None:
    """Where the turn-end marker after a content ending at ``end`` stops,
    or None when no marker follows: the marker sits right after the
    content, or after the whitespace the content ended in."""
    j = end
    while j < len(rendered) and rendered[j].isspace():
        j += 1
    for pos in range(end, j + 1):
        for tail in tails:
            if tail and rendered.startswith(tail, pos):
                return pos + len(tail)
    return None



def row_render_args(kind: str | None) -> dict:
    """render_row keyword arguments for a row's frame kind: "continue" is
    an open tail, "continue-closed" an open tail closed by the turn-end
    marker, "chat" a whole conversation through the template with every
    assistant turn a target, "reply" the same render with the final
    assistant turn as the only target, "reply-think" a reply row whose
    target starts at the final turn's reasoning trace."""
    if kind == "continue":
        return {"open_tail": True}
    if kind == "continue-closed":
        return {"open_tail": True, "close_tail": True}
    if kind == "reply":
        return {"open_tail": False, "last_only": True}
    if kind == "reply-think":
        return {"open_tail": False, "last_only": True, "reason_target": True}
    return {"open_tail": False}


_PROBE = "zqxj"


def _tails_cache(inner) -> dict:
    """The tail cache on the tokenizer object itself (a module-level map
    keyed by id() would outlive the tokenizer and could answer for another
    one at the same address); a throwaway dict on an object that refuses
    new attributes."""
    cache = getattr(inner, "_gmlx_tails", None)
    if cache is None:
        cache = {}
        try:
            inner._gmlx_tails = cache
        except AttributeError:
            pass
    return cache


def _special_strings(inner) -> list[str]:
    """Special, added and end-of-generation token strings, longest first:
    the vocabulary a template's turn markers are drawn from."""
    out = set(getattr(inner, "all_special_tokens", None) or [])
    with contextlib.suppress(Exception):
        out.update(inner.get_added_vocab().keys())
    for i in getattr(inner, "_gguf_eos_token_ids", None) or []:
        t = inner.convert_ids_to_tokens(int(i))
        if isinstance(t, str) and t:
            out.add(t)
    return sorted(out, key=len, reverse=True)


def assistant_tails(tokenizer) -> list[str]:
    """The markers a template puts after an assistant content, in the order
    they are tried: first what follows the content of a final turn
    ("<turn|>\\n", "<|im_end|>\\n", "<|return|>"), then what follows it
    when another turn comes after, cut after the first special or
    end-of-generation token string plus any newline ("<|end|>" on harmony,
    "<|user|>" on GLM). Measured on probe conversations rendered with the
    tokenizer's render settings; an empty list for a template that marks
    nothing."""
    inner = hf_inner(tokenizer)
    cache = _tails_cache(inner)
    # the template is part of the key: a copied tokenizer given another
    # template carries the copy's cache along
    key = (json.dumps(render_kwargs(tokenizer), sort_keys=True), has_chat_template(tokenizer),
           str(getattr(inner, "chat_template", None)))
    if key in cache:
        return cache[key]
    if not has_chat_template(tokenizer):
        eos = getattr(inner, "eos_token", None) or ""
        cache[key] = [eos] if eos else []
        return cache[key]
    tails: list[str] = []
    probe = [{"role": "user", "content": "q"}, {"role": "assistant", "content": _PROBE}]
    try:
        r = apply_template(tokenizer, probe)
        k = r.rfind(_PROBE)
        single = r[k + len(_PROBE):] if k >= 0 else ""
        if single:
            tails.append(single)
        r2 = apply_template(tokenizer, probe + [{"role": "user", "content": "q2"},
                                                {"role": "assistant", "content": "kkkk"}])
        k = r2.find(_PROBE)
        j = r2.find("q2", k) if k >= 0 else -1
        rest = r2[k + len(_PROBE):j] if k >= 0 and j > k else ""
        multi = ""
        if rest:
            cut = -1
            for tok in _special_strings(inner):
                i = rest.find(tok)
                if i >= 0 and (cut < 0 or i + len(tok) < cut):
                    cut = i + len(tok)
            if cut > 0:
                while cut < len(rest) and rest[cut] == "\n":
                    cut += 1
                multi = rest[:cut]
        if multi and multi not in tails:
            tails.append(multi)
    except ValueError:
        pass
    cache[key] = tails
    return tails


def target_mask(ends: np.ndarray, spans: list | None) -> np.ndarray:
    """Per position t, whether token t + 1 lies inside a target span (its
    bytes [ends[t], ends[t+1]) within [content start, target end)). Without
    spans every position with a successor is a target."""
    n = len(ends)
    m = np.zeros(max(n, 0), dtype=bool)
    if n < 2:
        return m
    if spans is None:
        m[:-1] = True
        return m
    e = np.asarray(ends, dtype=np.int64)
    starts = e[:-1]
    stops = e[1:]
    for sp in spans:
        b0, b2 = int(sp[0]), int(sp[-1])
        m[:-1] |= (starts >= b0) & (stops <= b2) & (stops > starts)
    return m


def fit_conversation(tokenizer, msgs: list[dict], max_len: int, tb) -> tuple | None:
    """Render a conversation within max_len tokens: trailing turns are
    dropped (always ending on an assistant turn) until the render fits.
    Returns (ids, ends, text, messages, spans, flagged) or None when
    nothing fits or the template cannot locate an assistant content."""
    msgs = list(msgs)
    while msgs and msgs[-1].get("role") != "assistant":
        msgs = msgs[:-1]
    while len(msgs) >= 2:
        try:
            text, spans = render_row(tokenizer, msgs, open_tail=False)
        except ValueError:
            return None
        ids, ends, flag = encode_with_byte_ends(tokenizer, text, tb, add_special_tokens=False)
        if len(ids) <= max_len:
            return ids, ends, text, msgs, spans, flag
        msgs = msgs[:-1]
        while msgs and msgs[-1].get("role") != "assistant":
            msgs = msgs[:-1]
    return None


def fit_reply(tokenizer, msgs: list[dict], max_len: int, tb, reason_target: bool = False) -> tuple | None:
    """Render a reply row within max_len tokens: the final assistant turn is
    the only target and is never dropped; leading turns after a system
    message are dropped, oldest first and up to the next user turn, until
    the render fits. Returns (ids, ends, text, messages, spans, flagged)
    or None when the last exchange alone does not fit or the template
    cannot locate the reply."""
    msgs = list(msgs)
    if not msgs or msgs[-1].get("role") != "assistant":
        return None
    while True:
        try:
            text, spans = render_row(tokenizer, msgs, open_tail=False, last_only=True, reason_target=reason_target)
        except ValueError:
            return None
        ids, ends, flag = encode_with_byte_ends(tokenizer, text, tb, add_special_tokens=False)
        if len(ids) <= max_len:
            return ids, ends, text, msgs, spans, flag
        head = 1 if msgs and msgs[0].get("role") == "system" else 0
        body = msgs[head:]
        cut = 1
        while cut < len(body) - 1 and body[cut].get("role") != "user":
            cut += 1
        if cut >= len(body) - 1:
            return None
        msgs = msgs[:head] + body[cut:]


def shared_boundaries_spans(t_ends: np.ndarray, s_ends: np.ndarray,
                            t_spans: list, s_spans: list) -> Alignment:
    """Shared boundaries between two renders of the same conversation whose
    frames differ: for each paired span, token ends at the same offset
    relative to the content start, from the content start through the
    content end (the position that predicts the end-of-turn marker), over
    positions whose successor is a target on both sides."""
    if len(t_spans) != len(s_spans):
        raise ValueError("teacher and student renders have different span counts")
    te = np.asarray(t_ends, dtype=np.int64)
    se = np.asarray(s_ends, dtype=np.int64)
    tm = target_mask(te, t_spans)
    sm = target_mask(se, s_spans)
    t_pos, s_pos, common = [], [], []
    for tsp, ssp in zip(t_spans, s_spans):
        tb0, tb1 = int(tsp[0]), int(tsp[1])
        sb0, sb1 = int(ssp[0]), int(ssp[1])
        if tb1 - tb0 != sb1 - sb0:
            raise ValueError("paired spans differ in content length")
        t_cand = {int(e - tb0): i for i, e in enumerate(te[:-1]) if tm[i] and tb0 <= e <= tb1}
        s_cand = {int(e - sb0): i for i, e in enumerate(se[:-1]) if sm[i] and sb0 <= e <= sb1}
        for r in sorted(set(t_cand) & set(s_cand)):
            t_pos.append(t_cand[r])
            s_pos.append(s_cand[r])
            common.append(r + tb0)
    return Alignment(t_pos=np.array(t_pos, dtype=np.int32), s_pos=np.array(s_pos, dtype=np.int32),
                     ends=np.array(common, dtype=np.int64), n_teacher=len(te), n_student=len(se))


def cut_windows(ids: np.ndarray, ws_start: np.ndarray, max_len: int,
                n_special_prefix: int, text: bytes | None = None,
                ends: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Windows [start, end) over a document's teacher tokens, at most
    max_len tokens each, cut at the last whitespace-initial token boundary.
    The first window keeps the special prefix (BOS); later windows start at
    a whitespace-initial token. A window with no whitespace-initial token
    in range is cut hard at max_len, backed off to a character boundary
    when the text bytes and the per-token end offsets are given (a
    byte-level tokenizer splits a multi-byte character over tokens)."""
    n = len(ids)
    out = []
    start = 0
    while start < n:
        end = min(start + max_len, n)
        if end < n:
            cut = end
            lo = start + n_special_prefix + 1 if start == 0 else start + 1
            while cut > lo and not ws_start[ids[cut]]:
                cut -= 1
            if cut > lo:
                end = cut
            elif text is not None and ends is not None:
                while end > lo and int(ends[end - 1]) < len(text) and (text[int(ends[end - 1])] & 0xC0) == 0x80:
                    end -= 1
        out.append((start, end))
        start = end
    return out


