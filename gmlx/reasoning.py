"""Streaming reasoning/thinking segmentation + display for the chat REPL.

Reasoning ("thinking") models interleave a private chain-of-thought with their
final answer using model-specific control markers, none of which a user wants
to read raw:

  * think-tag models (Qwen3 / DeepSeek-R1 / GLM): ``<think>`` ... ``</think>``.
    Many open the block *in the chat template's prompt*, so generation streams
    only the closing ``</think>`` - the opener is never emitted. Seed the filter
    with ``start_in_thinking=True`` (see ``thinking_budget.prompt_opens_thinking``)
    so the leading reasoning is still recognised.
  * harmony models (gpt-oss): ``<|channel|>analysis<|message|>`` ... ``<|end|>``
    for reasoning, ``<|start|>assistant<|channel|>final<|message|>`` ... for the
    answer (``commentary`` channels carry tool preludes - treated as reasoning).
  * Onyx ATEM (Muse Glimmer): the routing is a recipient in the message header,
    ``<|start|>assistant to=self<|message|>`` ... ``<|eom|>`` for reasoning and
    ``to=user`` (or a tool name) for everything else. The prompt ends at
    ``<|start|>assistant``, so the first header arrives without its opener -
    seed with ``start_in_header=True``.
  * Gemma-style ``<|channel>thought`` ... ``<channel|>`` (as detokenized).

``ReasoningFilter`` is a streaming state machine that strips the markers and
tags every span as reasoning vs answer; ``ReasoningPrinter`` renders it three
ways - ``show`` streams the reasoning dimmed inside a gutter-framed block that
closes with a ``thought for Xs * N tok`` payoff, ``hide`` collapses it to a
single live spinner that resolves to the same payoff, and ``raw`` passes
everything through verbatim. A Ctrl-O-style ``set_display`` flips show<->hide
live. Both classes are pure/testable - the filter is text-only, the printer
takes an injectable ``write`` sink and ``clock``.
"""

from __future__ import annotations

import os
import sys

from .render import _CONTROL_CHARS

# Span/marker actions.
_REASON = "reason"   # enter the chain-of-thought
_ANSWER = "answer"   # enter the final answer
_DROP = "drop"       # strip the marker, keep the current mode

# Control markers across the formats we've observed, with the mode each implies.
# Order is irrelevant (we sort longest-first at construction so the most specific
# marker wins at a shared position); keep it grouped by format for readability.
_MARKERS: tuple[tuple[str, str], ...] = (
    # harmony (gpt-oss): channel headers carry the routing in the marker
    # itself. Name-only forms catch annotated headers too
    # ("<|channel|>commentary to=functions.x <|constrain|>json<|message|>");
    # any "<|channel|>"-prefixed match swallows the rest of the header up to
    # "<|message|>" (see _swallow), and the bare "<|channel|>" catch-all
    # routes unknown channels to reasoning - never to user-visible content.
    ("<|channel|>analysis", _REASON),
    ("<|channel|>commentary", _REASON),
    ("<|channel|>final", _ANSWER),
    ("<|start|>assistant", _DROP),
    ("<|channel|>", _REASON),
    ("<|message|>", _DROP),
    ("<|constrain|>", _DROP),
    ("<|return|>", _DROP),
    ("<|start|>", _DROP),
    ("<|call|>", _DROP),
    ("<|end|>", _DROP),
    # Onyx ATEM (Muse Glimmer): no channel marker - the recipient lives in the
    # header that "<|start|>assistant" opens and "<|message|>" closes, so the
    # routing is decided in _close_header. "<|eom|>" ends one message of a
    # multi-message turn; the next header re-decides, and until it arrives the
    # safe assumption is answer.
    ("<|eom|>", _ANSWER),
    ("<|eot|>", _DROP),
    # gemma-style channel (as detokenized - note the lopsided pipes).
    ("<|channel>thought", _REASON),
    ("<channel|>", _ANSWER),
    # think-tag family (qwen3 / deepseek-r1 / glm / ...).
    ("<think>", _REASON),
    ("</think>", _ANSWER),
    # Hy3: think tags carry the ':opensource' suffix; at reasoning_effort
    # low/high the chat template pre-fills the open tag, so generation streams
    # only the close (seed with start_in_thinking, like <think>).
    ("<think:opensource>", _REASON),
    ("</think:opensource>", _ANSWER),
    # MiniMax-M3: the model emits the opener itself (legacy '<think>' tokens
    # exist in its vocab but the template only ever uses the mm spelling).
    ("<mm:think>", _REASON),
    ("</mm:think>", _ANSWER),
    # Kimi-K3 XTML sections: the prompt pre-opens the think section (seed with
    # start_in_thinking), generation streams the close, the response section,
    # then the message close + turn terminator. Attribute-carrying openers
    # (e.g. '<|open|>message role=...') only occur in the prompt, never in
    # generation, so literal matching is sufficient here.
    ("<|open|>think<|sep|>", _REASON),
    ("<|close|>think<|sep|>", _ANSWER),
    ("<|open|>response<|sep|>", _ANSWER),
    ("<|close|>response<|sep|>", _DROP),
    ("<|close|>message<|sep|>", _DROP),
    ("<|end_of_msg|>", _DROP),
)

# Markers that open a message header - text from here to the closing
# "<|message|>" is routing metadata, never a display span. Harmony puts the
# channel in the marker itself; ATEM puts the recipient in the header body.
_HEADER_PREFIXES = ("<|channel|>", "<|start|>assistant")

# ATEM's reasoning recipient. "self" is the only recipient that is not
# user-visible; "user" and tool namespaces are answer-side.
_SELF_RECIPIENT = "to=self"


class ReasoningFilter:
    """Strip reasoning control markers from a token stream and tag the rest.

    Feed it text as it arrives; ``feed`` returns a list of ``(text, mode)``
    spans where ``mode`` is ``"reason"`` or ``"answer"``. A trailing fragment
    that could still be the start of a marker is held back until the next feed,
    so a marker split across chunks still matches; ``flush`` releases whatever
    remains at end-of-stream (a partial marker there is just literal text).
    """

    def __init__(self, *, start_in_thinking: bool = False,
                 start_in_header: bool = False):
        self._markers = sorted(_MARKERS, key=lambda m: len(m[0]), reverse=True)
        self.mode = _REASON if start_in_thinking else _ANSWER
        self.buf = ""
        self._swallow = False  # inside a message header (drop text)
        self._swallow_budget = 0
        self._header = ""
        if start_in_header:
            self._open_header()

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.buf += text
        return self._consume(final=False)

    def flush(self) -> list[tuple[str, str]]:
        return self._consume(final=True)

    # -- internals -----------------------------------------------------------

    def _consume(self, *, final: bool) -> list[tuple[str, str]]:
        spans: list[tuple[str, str]] = []
        while self.buf:
            pos = self._next_marker_pos(final)
            if pos is None:
                self._emit(self.buf, spans)
                self.buf = ""
                break
            if pos > 0:
                self._emit(self.buf[:pos], spans)
                self.buf = self.buf[pos:]
                continue
            # buf now starts at a (possibly partial) marker.
            hit = self._full_marker_at0()
            if hit is None:
                if final:  # an unfinished marker at EOS is just text
                    self._emit(self.buf, spans)
                    self.buf = ""
                break  # otherwise wait for more input
            if not final and self._could_extend():
                break  # a longer marker might still complete - wait
            marker, action = hit
            self.buf = self.buf[len(marker):]
            if action != _DROP:
                self.mode = action
            # Message headers run "<|channel|>NAME [annotations]<|message|>"
            # (harmony) or "<|start|>assistant to=RECIPIENT<|message|>" (ATEM):
            # an opener marker starts the header, "<|message|>" closes it, and
            # the text in between is routing, never a display span.
            if marker.startswith(_HEADER_PREFIXES):
                self._open_header()
            elif marker == "<|message|>":
                self._close_header()
        return [s for s in spans if s[0]]

    def _open_header(self) -> None:
        self._swallow = True
        self._swallow_budget = 256
        self._header = ""

    def _close_header(self) -> None:
        """End the header and apply its routing. Only an explicit ``to=self``
        recipient moves the mode: a harmony header is empty (its channel marker
        already routed), and an ATEM answer/tool header must leave a mode that
        "<|eom|>" or the initial state already set."""
        self._swallow = False
        if _SELF_RECIPIENT in self._header:
            self.mode = _REASON
        self._header = ""

    def _emit(self, text: str, spans: list[tuple[str, str]]) -> None:
        """Append a display span, unless a message header is being swallowed.
        The budget bounds the swallow: a literal "<|channel|>" in a
        non-harmony reply (no "<|message|>" ever follows) must not eat the
        rest of the message."""
        if self._swallow:
            self._swallow_budget -= len(text)
            if self._swallow_budget >= 0:
                self._header += text
                return
            self._swallow = False
            self._header = ""
        spans.append((text, self.mode))

    def _next_marker_pos(self, final: bool) -> int | None:
        """Earliest index of a full marker, or - unless ``final`` - of a trailing
        fragment that could still grow into one. ``None`` if nothing marker-like."""
        best = -1
        for marker, _ in self._markers:
            j = self.buf.find(marker)
            if j != -1 and (best == -1 or j < best):
                best = j
        if not final:
            part = self._partial_tail_start()
            if part != -1 and (best == -1 or part < best):
                return part
        return None if best == -1 else best

    def _partial_tail_start(self) -> int:
        """Start index of the shortest trailing suffix that is a strict prefix of
        some marker (the fragment to hold back), or -1."""
        best = -1
        n = len(self.buf)
        for marker, _ in self._markers:
            kmax = min(n, len(marker) - 1)
            for k in range(kmax, 0, -1):
                if marker.startswith(self.buf[-k:]):
                    pos = n - k
                    if best == -1 or pos < best:
                        best = pos
                    break
        return best

    def _full_marker_at0(self) -> tuple[str, str] | None:
        for marker, action in self._markers:  # longest-first -> most specific wins
            if self.buf.startswith(marker):
                return marker, action
        return None

    def _could_extend(self) -> bool:
        """Could ``buf`` still grow into a marker longer than what it holds now?"""
        for marker, _ in self._markers:
            if len(marker) > len(self.buf) and marker.startswith(self.buf):
                return True
        return False


def prompt_opens_header(prompt) -> bool:
    """Whether a rendered ``prompt`` stops inside a message header, so the
    filter must start mid-header. Both harmony and ATEM generation prompts end
    at ``<|start|>assistant`` and leave the channel or recipient to the model.
    Tolerant of token-id prompts (False for non-strings)."""
    return (isinstance(prompt, str)
            and prompt.rstrip().endswith("<|start|>assistant"))


def split_harmony_reply(text: str, *,
                        start_in_header: bool = False) -> tuple[str | None, str]:
    """Split a complete harmony (gpt-oss) or ATEM reply into
    ``(reasoning, content)``.

    The serve path's stock splitter knows none of the harmony markers, so it
    returns the raw channel markup as content - which the model's own chat
    template rejects with a template error the moment a client sends the
    reply back as history. This runs the same state machine the chat REPL
    streams through, so both surfaces classify identically: analysis and
    commentary channels (tool preludes included) become reasoning, the final
    channel becomes content, and a reply truncated inside analysis returns
    all-reasoning with empty content (the convention the truncated-thinking
    handling already uses for think-tag models).

    ``start_in_header`` seeds the ATEM case, where the generation prompt ends
    mid-header at ``<|start|>assistant`` and the reply opens with ``
    to=self<|message|>``."""
    filt = ReasoningFilter(start_in_header=start_in_header)
    spans = filt.feed(text)
    spans += filt.flush()
    reasoning = "".join(t for t, m in spans if m == _REASON).strip()
    content = trim_content_ws("".join(t for t, m in spans if m == _ANSWER))
    return (reasoning or None, content)


def trim_content_ws(text: str) -> str:
    """Trim reply content: drop leading blank lines and trailing whitespace,
    keep the first content line's indent (a full strip eats verbatim-code
    indentation)."""
    lines = (text or "").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).rstrip()


_DIM = "\x1b[2m"
_RESET = "\x1b[0m"
_CLEAR_EOL = "\x1b[K"
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _fmt_elapsed(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{int(secs // 60)}m{int(secs % 60):02d}s"


class ReasoningPrinter:
    """Render ``(text, mode)`` spans to a writer, with timing + a live indicator.

    ``display``:
      * ``"show"``  - the reasoning streams dimmed inside a gutter-framed block
        that closes with a ``thought for Xs * N tok`` footer (the payoff).
      * ``"hide"``  - the reasoning is collapsed to a single live spinner line
        (``⠹ thinking... Xs * N tok``) that resolves to ``OK thought for Xs * N tok``
        when the answer starts; the body is never printed.
      * ``"raw"``   - everything written verbatim (no stripping, no styling).

    ``color`` toggles ANSI (and the animated spinner); ``write`` defaults to
    stdout; ``clock`` is injectable for tests. ``tick()`` should be called once
    per generated token so the spinner animates and reasoning tokens are counted;
    ``set_display()`` flips show<->hide live (e.g. a Ctrl-O keybinding).
    """

    def __init__(
        self,
        *,
        display: str = "show",
        color: bool = True,
        write=None,
        label: str = "thinking",
        clock=None,
        answer_sink=None,
        theme=None,
    ):
        self.display = display
        self.color = color
        self._w = write or (lambda s: (sys.stdout.write(s), sys.stdout.flush()))
        self._answer_sink = answer_sink
        self._theme = theme
        self.label = label
        import time

        self._clock = clock or time.monotonic
        self._cur: str | None = None      # last rendered kind
        self._saw_reason = False
        self._reason_done = False         # reasoning region finalized (footer/resolve shown)
        self._answer_lstrip = False
        self._reason_at_bol = False       # at the start of a gutter line (show mode)
        self._block: str | None = None    # None | "expanded" | "collapsed"
        self._t0: float | None = None     # clock at first reasoning token
        self._elapsed = 0.0
        self._tokens = 0
        self._spin = 0

    # -- public --------------------------------------------------------------

    def feed_spans(self, spans) -> None:
        for text, kind in spans:
            if self.display != "raw":    # raw = verbatim by contract
                # Reasoning text is model output: strip ESC/C0 controls so it
                # cannot inject ANSI through the styled gutter (same rule as
                # the answer renderer).
                text = _CONTROL_CHARS.sub("", text)
            self._render(text, kind)

    def tick(self) -> None:
        """One generated token elapsed. Counts reasoning tokens and, in collapsed
        mode, repaints the live spinner."""
        if self.display == "raw" or self._reason_done:
            return
        if self._cur == _ANSWER:
            return
        if self._saw_reason:
            self._tokens += 1
            if self.display == "hide" and self.color:
                self._draw_spinner()

    def set_display(self, display: str) -> None:
        """Switch show<->hide (raw is sticky - set it via the flag/command). Applies
        forward; already-streamed reasoning is left in place."""
        if display == self.display or display == "raw" or self.display == "raw":
            self.display = display
            return
        mid_reason = self._saw_reason and not self._reason_done
        if mid_reason and self.display == "show" and display == "hide":
            if not self._reason_at_bol:
                self._w("\n")  # end the open gutter line; spinner draws on next tick
            self._block = None
        elif mid_reason and self.display == "hide" and display == "show":
            if self._block == "collapsed" and self.color:
                self._w("\r" + _CLEAR_EOL)  # wipe the spinner line
            self._block = None  # _enter_reason reopens the expanded header
            self._cur = None
        self.display = display

    def close(self, canceled: bool = False) -> None:
        if self.display == "raw":
            return
        if self._saw_reason and not self._reason_done:
            # Reasoning never reached an answer (cancel, or a think-only reply):
            # still close it out with the payoff so the block isn't left dangling.
            self._finalize_reason(trailing="\n")
        elif self.color and self._cur == _REASON:
            self._w(_RESET)

    # -- internals -----------------------------------------------------------

    def _render(self, text: str, kind: str) -> None:
        if self.display == "raw":
            self._w(text)
            return
        if kind == _REASON:
            if self._reason_done:  # post-answer reasoning (rare) - drop it
                return
            self._saw_reason = True
            if self._t0 is None:
                self._t0 = self._clock()
            if self.display == "hide":
                return  # body suppressed; the spinner (tick) carries the signal
            self._emit_reason_body(text)
        else:
            if not self._reason_done:
                self._finalize_reason(trailing="\n\n")
            if self._answer_lstrip:
                text = text.lstrip("\n")
                if not text:
                    return
                self._answer_lstrip = False
            self._cur = _ANSWER
            (self._answer_sink or self._w)(text)

    def _emit_reason_body(self, text: str) -> None:
        if self._block != "expanded":
            self._w(self._dim(f"┌ {self.label}\n"))
            self._block = "expanded"
            self._reason_at_bol = True
            text = text.lstrip("\n")  # drop the leading newline after an open marker
        self._cur = _REASON
        if not text:
            return
        out = []
        for ch in text:
            if self._reason_at_bol:
                out.append("│ ")
                self._reason_at_bol = False
            out.append(ch)
            if ch == "\n":
                self._reason_at_bol = True
        self._w(self._dim("".join(out)))

    def _finalize_reason(self, *, trailing: str) -> None:
        """Close the reasoning region with the elapsed/token payoff, exactly once."""
        if self._reason_done:
            return
        self._reason_done = True
        if not self._saw_reason:
            return
        if self._t0 is not None:
            self._elapsed = self._clock() - self._t0
        payoff = f"thought for {_fmt_elapsed(self._elapsed)} · {self._tokens} tok"
        if self._block == "collapsed" and self.color:
            self._w("\r" + self._dim(f"✓ {payoff}") + _CLEAR_EOL + trailing)
        elif self._block == "expanded":
            if not self._reason_at_bol:
                self._w("\n")
            self._w(self._dim(f"└ {payoff}") + trailing)
        else:  # hide without color, or nothing drawn yet
            self._w(self._dim(payoff) + trailing)
        self._answer_lstrip = True

    def _draw_spinner(self) -> None:
        frame = _SPINNER[self._spin % len(_SPINNER)]
        self._spin += 1
        self._block = "collapsed"
        elapsed = self._clock() - self._t0 if self._t0 is not None else 0.0
        line = f"{frame} {self.label}… {_fmt_elapsed(elapsed)} · {self._tokens} tok"
        self._w("\r" + self._dim(line) + _CLEAR_EOL)

    def _dim(self, text: str) -> str:
        if not self.color:
            return text
        if self._theme is not None:
            return self._theme.paint("thinking", text)
        return f"{_DIM}{text}{_RESET}"


class StreamRenderer:
    """One-shot glue for the ``run`` CLI: :class:`ReasoningFilter` +
    :class:`ReasoningPrinter` behind a ``(write, close)`` pair, so the verbose
    generate paths stream a thinking model through the same show/hide styling
    the chat REPL uses. ``start_in_thinking`` seeds the pre-fill-template case
    (the prompt opened the think block, so the stream carries only the close
    marker); ``start_in_header`` the harmony/ATEM case (the prompt stopped
    mid-header)."""

    def __init__(self, display: str = "show", *,
                 start_in_thinking: bool = False,
                 start_in_header: bool = False, color: bool | None = None):
        self._filter = ReasoningFilter(start_in_thinking=start_in_thinking,
                                       start_in_header=start_in_header)
        self._printer = ReasoningPrinter(
            display=display, color=want_color() if color is None else color)

    def write(self, text: str) -> None:
        self._printer.feed_spans(self._filter.feed(text))
        self._printer.tick()

    def close(self) -> None:
        self._printer.feed_spans(self._filter.flush())
        self._printer.close()


def want_color(stream=None) -> bool:
    """ANSI is wanted when the stream is a TTY and ``NO_COLOR`` is unset."""
    stream = stream or sys.stdout
    try:
        tty = stream.isatty()
    except Exception:
        tty = False
    return bool(tty) and os.environ.get("NO_COLOR") is None


# -- provider-doc thinking switch spellings ------------------------------------

def thinking_flag(value) -> bool | None:
    """The z.ai / GLM API spelling of the thinking switch -
    ``{"type": "enabled"|"disabled"}`` - as a bool; ``None`` for anything
    else (including template variables that happen to be named thinking)."""
    if (isinstance(value, dict) and set(value) <= {"type"}
            and value.get("type") in ("enabled", "disabled")):
        return value["type"] == "enabled"
    return None


def normalize_template_kwargs(kwargs: dict) -> dict:
    """Translate ``thinking: {"type": ...}`` (the z.ai / GLM API request
    schema users copy from provider docs) into ``enable_thinking``, the
    variable chat templates actually read. Any other ``thinking`` value
    passes through untouched as a plain template variable."""
    flag = thinking_flag(kwargs.get("thinking"))
    if flag is None:
        return kwargs
    out = dict(kwargs)
    del out["thinking"]
    out.setdefault("enable_thinking", flag)
    return out


_TEMPLATE_DEFAULT_FLAG = "_kq_template_default_thinking"


def install_template_default_thinking() -> None:
    """Keep an absent ``enable_thinking`` meaning "the template's own default".

    mlx-vlm >= 0.6.15 ``get_chat_template`` injects ``enable_thinking=False``
    whenever the kwarg is missing and the tokenizer's ``apply_chat_template``
    accepts it - a ``**kwargs`` signature always does - so every default-on
    reasoning template renders the dead ``<think></think>`` prefill and the
    model stops thinking. Seed the kwarg with a Jinja ``Undefined``: the
    injection sees it as present and stands down, and the template evaluates
    ``enable_thinking is defined`` as false - same semantics as a truly
    absent variable. An explicit value passes through untouched. Idempotent;
    rebinds the module global both ``apply_chat_template`` call sites
    resolve at call time."""
    import importlib

    pu = importlib.import_module("mlx_vlm.prompt_utils")
    original = pu.get_chat_template
    if getattr(original, _TEMPLATE_DEFAULT_FLAG, False):
        return
    from jinja2 import Undefined

    undef = Undefined(name="enable_thinking")

    def get_chat_template(processor, messages, add_generation_prompt,
                          tokenize=False, **kwargs):
        if "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = undef
        return original(processor, messages, add_generation_prompt,
                        tokenize=tokenize, **kwargs)

    get_chat_template.__dict__[_TEMPLATE_DEFAULT_FLAG] = True
    pu.get_chat_template = get_chat_template


# Values accepted for the model-agnostic `thinking` control (CLI flag,
# config/profile key, z.ai-style request field).
_THINKING_LEVELS = {"on": "on", "off": "off", "adaptive": "adaptive",
                    "enabled": "on", "disabled": "off",
                    True: "on", False: "off"}


# Templates that grade reasoning depth under a name of their own. The control
# is spelled reasoning_effort throughout gmlx, so map it onto the template's
# spelling rather than emitting a variable the template ignores (Muse Glimmer's
# ATEM template reads reasoning_strength).
_EFFORT_ALIASES = ("reasoning_strength",)


def _effort_variable(template: str) -> str:
    """The reasoning-depth variable ``template`` reads, defaulting to the
    canonical ``reasoning_effort``."""
    if template and "reasoning_effort" not in template:
        for alias in _EFFORT_ALIASES:
            if alias in template:
                return alias
    return "reasoning_effort"


def map_thinking_controls(base: dict, thinking=None, reasoning_effort=None,
                          template: str = "", warn=None) -> dict:
    """Overlay the dedicated thinking controls onto ``base`` template kwargs,
    spelled as the variables ``template`` actually reads. ``base`` (an
    explicitly passed ``chat_template_kwargs`` dict) is never reinterpreted -
    its keys pass through verbatim; only the controls given as arguments are
    mapped, and a mapped switch overwrites a same-named base key (the
    dedicated control is the more specific ask).

    ``thinking`` (on/off/adaptive; enabled/disabled, plain bools, and the
    z.ai dict shape accepted) becomes MiniMax's three-state ``thinking_mode``
    where present, ``enable_thinking`` next (Qwen3.x, GLM, DeepSeek-V4
    alias), or Hy3's ``reasoning_effort: no_think`` dialect.
    ``reasoning_effort`` passes through under its own name (level names are
    the model's own; its template validates them), and an explicit level -
    argument or ``base`` key - wins over one a ``thinking`` switch would
    imply. gpt-oss-style templates grade effort but cannot disable
    reasoning; ``warn`` (an optional callable taking a message) hears about
    that and other unmappable controls. With no template text available the
    switch defaults to the ``enable_thinking`` spelling."""
    _warn = warn or (lambda msg: None)
    out = dict(base)
    if reasoning_effort is not None:
        name = _effort_variable(template)
        out[name] = reasoning_effort
        if template and name not in template:
            _warn("this model's chat template has no reasoning_effort "
                  "variable; reasoning_effort is likely a no-op")
    if thinking is None:
        return out
    if isinstance(thinking, dict):
        flag = thinking_flag(thinking)
        thinking = {True: "on", False: "off", None: thinking}[flag]
    mode = _THINKING_LEVELS.get(thinking) if not isinstance(thinking, dict) \
        else None
    if mode is None:
        _warn(f"unrecognized thinking value {thinking!r} "
              "(expected on/off/adaptive) - ignored")
        return out
    if "thinking_mode" in template:
        out["thinking_mode"] = {"on": "enabled", "off": "disabled",
                                "adaptive": "adaptive"}[mode]
    elif mode == "adaptive":
        _warn("adaptive is a MiniMax-style thinking_mode level; this model's "
              "chat template has no such variable - ignored")
    elif "enable_thinking" in template or not template:
        out["enable_thinking"] = mode == "on"
    elif "no_think" in template and "reasoning_effort" in template:
        out.setdefault("reasoning_effort",
                       "no_think" if mode == "off" else "high")
    elif "reasoning_effort" in template:
        _warn("this model has no thinking switch (reasoning always runs); "
              "use reasoning_effort low|medium|high to size it")
    elif any(alias in template for alias in _EFFORT_ALIASES):
        _warn("this model has no thinking switch (reasoning always runs); "
              "use reasoning_effort to size it")
    else:
        _warn("this model's chat template has no thinking switch; the "
              "thinking control is likely a no-op")
        out["enable_thinking"] = mode == "on"
    return out
