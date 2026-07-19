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
    # harmony (gpt-oss): channel headers carry the routing in the marker itself.
    ("<|channel|>analysis<|message|>", _REASON),
    ("<|channel|>commentary<|message|>", _REASON),
    ("<|channel|>final<|message|>", _ANSWER),
    ("<|start|>assistant", _DROP),
    ("<|channel|>", _DROP),
    ("<|message|>", _DROP),
    ("<|constrain|>", _DROP),
    ("<|return|>", _DROP),
    ("<|start|>", _DROP),
    ("<|call|>", _DROP),
    ("<|end|>", _DROP),
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
)


class ReasoningFilter:
    """Strip reasoning control markers from a token stream and tag the rest.

    Feed it text as it arrives; ``feed`` returns a list of ``(text, mode)``
    spans where ``mode`` is ``"reason"`` or ``"answer"``. A trailing fragment
    that could still be the start of a marker is held back until the next feed,
    so a marker split across chunks still matches; ``flush`` releases whatever
    remains at end-of-stream (a partial marker there is just literal text).
    """

    def __init__(self, *, start_in_thinking: bool = False):
        self._markers = sorted(_MARKERS, key=lambda m: len(m[0]), reverse=True)
        self.mode = _REASON if start_in_thinking else _ANSWER
        self.buf = ""

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
                spans.append((self.buf, self.mode))
                self.buf = ""
                break
            if pos > 0:
                spans.append((self.buf[:pos], self.mode))
                self.buf = self.buf[pos:]
                continue
            # buf now starts at a (possibly partial) marker.
            hit = self._full_marker_at0()
            if hit is None:
                if final:  # an unfinished marker at EOS is just text
                    spans.append((self.buf, self.mode))
                    self.buf = ""
                break  # otherwise wait for more input
            if not final and self._could_extend():
                break  # a longer marker might still complete - wait
            marker, action = hit
            self.buf = self.buf[len(marker):]
            if action != _DROP:
                self.mode = action
        return [s for s in spans if s[0]]

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


def want_color(stream=None) -> bool:
    """ANSI is wanted when the stream is a TTY and ``NO_COLOR`` is unset."""
    stream = stream or sys.stdout
    try:
        tty = stream.isatty()
    except Exception:
        tty = False
    return bool(tty) and os.environ.get("NO_COLOR") is None
