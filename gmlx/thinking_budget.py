"""Hard cap on reasoning ("thinking") tokens for the text generate path.

Thinking models open a reasoning block and reason until they emit a close
marker. Left uncapped the model can spend an unbounded number of tokens
thinking. This module provides an mlx-lm *logits processor* that counts
generated thinking tokens and, once a budget is exceeded, forces the close
sequence so the model proceeds to its answer. From then on the budget is
spent: reopened thinking blocks are closed on the spot (interleaved-thinking
models like MiniMax-M3 reopen legitimately - allowed, just not funded), and
a small EOS floor keeps the turn from ending before any answer exists.

The delimiters are not one fixed spelling. They are resolved per model from
mlx-lm's tokenizer inference, which covers ``<think>`` / ``</think>``,
``<longcat_think>``, and the multi-token channel format ``<|channel>thought`` /
``<channel|>``. A budget of 0 force-closes thinking as soon as it opens, which
is how the over-generation probe disables thinking in an injected critique.

Ported from mlx-vlm's ``ThinkingBudgetCriteria`` (a stopping-criteria callback
in mlx-vlm's own loop); reframed as a logits processor because gmlx's text
path drives generation through ``mlx_lm.generate`` / ``stream_generate``, whose
extension point is ``logits_processors``.
"""

from __future__ import annotations

import os
import signal
import sys

import mlx.core as mx


# Think-tag spellings scanned when the caller relies on the defaults. The
# suffixed pair is Hy3's (every control tag carries ':opensource'; its chat
# template pre-fills the open tag at reasoning_effort low/high); the mm pair
# is MiniMax-M3's. '<think>' cannot false-match inside '<think:opensource>'
# (the '>' differs) or '<mm:think>' (no leading '<'), so the pairs are
# disjoint and scanning all of them is safe.
_THINK_PAIRS = (("<think>", "</think>"),
                ("<think:opensource>", "</think:opensource>"),
                ("<mm:think>", "</mm:think>"),
                # Kimi-K3 XTML sections (multi-token markers; the prompt
                # pre-opens the think section).
                ("<|open|>think<|sep|>", "<|close|>think<|sep|>"),
                # Onyx ATEM (Muse Glimmer): the reasoning message is addressed
                # to self and ends at <|eom|>. Both literals appear verbatim in
                # the template source. The generation prompt stops one marker
                # short (at "<|start|>assistant"), so the block is not
                # prompt-opened and the model emits the whole header itself.
                ("<|start|>assistant to=self<|message|>", "<|eom|>"))

# Forced into the thinking block ahead of the first budget-triggered close.
# The model must see itself DECIDE to answer: a bare close tag cuts the
# thought mid-sentence, and the unresolved reasoning tends to resume untagged
# inside the answer (endless self-review). First person, decisive, final.
_BUDGET_WRAP_PHRASE = (
    "\n\nI've hit my thinking budget, so I'll stop reasoning here and "
    "write the complete final answer now.\n"
)

# Forced when the user presses ^T (finish thinking now) - same rationale,
# but the model should see the user's request, not a phantom budget.
_SKIP_WRAP_PHRASE = (
    "\n\nI've been asked to wrap up, so I'll stop reasoning here and "
    "write the complete final answer now.\n"
)


def _template_think_pair(tokenizer):
    """The ``_THINK_PAIRS`` spelling the model's chat template actually uses,
    or None. The template is authoritative over vocab probing: a vocab can
    carry legacy ``</think>`` entries while the template thinks in another
    spelling (MiniMax-M3 ships both ``</think>`` and ``</mm:think>`` as real
    single tokens but only ever emits the latter), and probing then resolves
    - and forces - a tag the model treats as ordinary text."""
    tmpl = getattr(tokenizer, "chat_template", None)
    if not isinstance(tmpl, str) or not tmpl:
        return None
    for pair in _THINK_PAIRS:
        if pair[0] in tmpl or pair[1] in tmpl:
            return pair
    # Kimi-K3 builds its markers by macro concatenation ('<|open|>' + tag +
    # '<|sep|>'), so the literal pair never appears in the template source;
    # detect the macro call on the 'think' tag instead.
    if "<|open|>" in tmpl and "open_tag('think" in tmpl:
        return ("<|open|>think<|sep|>", "<|close|>think<|sep|>")
    return None


def prompt_open_think_tag(prompt, start_token: str | None = None,
                          end_token: str | None = None, tokenizer=None):
    """The start marker of the thinking block a rendered ``prompt`` ends
    inside (its last start marker not yet closed by a later end marker), or
    None when the prompt is closed. The returned spelling is the model's own,
    so a raw printer can echo it to make the stream well-formed.

    A full configured pair (``start_token`` and ``end_token``, from a family
    card or the user's config) names the tags this model really emits, so it
    wins over the tokenizer probe below."""
    if not isinstance(prompt, str):
        return None
    if start_token and end_token:
        pairs = [(start_token, end_token)]
    else:
        pairs = [(start_token or "<think>", end_token or "</think>")]
        if tokenizer is not None:
            tpair = _template_think_pair(tokenizer)
            s = getattr(tokenizer, "think_start", None)
            e = getattr(tokenizer, "think_end", None)
            if tpair is not None:
                pairs = [tpair]
            elif s and e:
                pairs = [(s, e)]
    if pairs == [("<think>", "</think>")]:
        pairs = list(_THINK_PAIRS)
    for start, end in pairs:
        i = prompt.rfind(start)
        if i != -1 and i > prompt.rfind(end):
            return start
    return None


def prompt_opens_thinking(prompt, start_token: str | None = None,
                          end_token: str | None = None, tokenizer=None) -> bool:
    """Whether the rendered ``prompt`` ends inside an open thinking block.

    This is the right signal for seeding the budget processor's in-thinking state.
    A pre-fill model (GLM-style) opens the block in the prompt, so counting must
    start immediately; a generate model (Qwen3-style) leaves the prompt closed and
    emits the start marker itself, which the processor then detects. Keying off
    this instead of ``enable_thinking`` honors an explicit budget regardless of the
    flag (the flag only shapes the prompt, via the template). Pass ``tokenizer`` to
    use the model's detected markers instead of the defaults; with the defaults,
    every known think-tag spelling (``_THINK_PAIRS``) is scanned. A configured
    ``start_token``/``end_token`` pair wins over both."""
    return prompt_open_think_tag(
        prompt, start_token=start_token, end_token=end_token, tokenizer=tokenizer
    ) is not None


def _encode_ids(tokenizer, text: str):
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _last_token_id(tokenizer, text: str):
    """Resolve ``text`` to a single token id (its last, matching mlx-vlm)."""
    ids = _encode_ids(tokenizer, text)
    return int(ids[-1]) if ids is not None and len(ids) else None


def _single_token_id(tokenizer, text: str):
    """``text``'s id when it encodes to exactly one token (i.e. it is a real
    entry in the model's vocab, not a spelling split into BPE pieces)."""
    ids = _encode_ids(tokenizer, text)
    return int(ids[0]) if ids is not None and len(ids) == 1 else None


def _tag_pair_seqs(tokenizer, start_tag, end_tag):
    """``(start_seq, end_seq)`` for a known marker spelling, or None when the
    end tag does not encode at all (the caller then keeps probing). Single
    vocab entries are preferred over the BPE pieces of the same spelling;
    ``start_tag`` may be None, which yields a None start_seq."""
    if not end_tag:
        return None
    end_id = _single_token_id(tokenizer, end_tag)
    if end_id is not None:
        start_id = _single_token_id(tokenizer, start_tag) if start_tag else None
        return ((start_id,) if start_id is not None else None, (end_id,))
    end_ids = _encode_ids(tokenizer, end_tag)
    if not end_ids:
        return None
    start_ids = _encode_ids(tokenizer, start_tag) if start_tag else None
    return (tuple(start_ids) if start_ids else None, tuple(end_ids))


def _thinking_token_seqs(tokenizer, start_token=None, end_token=None):
    """``(start_seq, end_seq)`` token-id tuples for the model's thinking
    delimiters, from mlx-lm's ``TokenizerWrapper`` inference (covers ``<think>``,
    ``<longcat_think>``, and the ``<|channel>thought`` / ``<channel|>`` channel
    format). Bare tokenizers without the wrapper API are probed against every
    ``_THINK_PAIRS`` spelling: a pair whose tags are real single vocab tokens is
    the model's convention. The historic last-BPE-piece ``</think>`` probe stays
    as the final fallback -- it must not run first, because on a suffixed-tag
    model it resolves ``</think>`` to its trailing ``'>'`` piece, which then
    (dis)arms the processor on every ``>`` the model emits. ``end_seq`` is
    ``None`` when no thinking convention is detected. A chat template that
    names one of the ``_THINK_PAIRS`` spellings wins over everything except a
    configured ``end_token``: the template is the ground truth for which tags
    the model actually emits (see ``_template_think_pair``), and configured
    markers are the user's word for a model neither one recognizes."""
    seqs = _tag_pair_seqs(tokenizer, start_token, end_token)
    if seqs is not None:
        return seqs
    pair = _template_think_pair(tokenizer)
    if pair is not None:
        seqs = _tag_pair_seqs(tokenizer, *pair)
        if seqs is not None:
            return seqs
    end = getattr(tokenizer, "think_end_tokens", None)
    if end:
        start = getattr(tokenizer, "think_start_tokens", None)
        return (tuple(start) if start else None, tuple(end))
    for start_tag, end_tag in _THINK_PAIRS:
        end_id = _single_token_id(tokenizer, end_tag)
        if end_id is None:
            continue
        start_id = _single_token_id(tokenizer, start_tag)
        return ((start_id,) if start_id is not None else None, (end_id,))
    end_id = _last_token_id(tokenizer, "</think>")
    if end_id is None:
        return (None, None)
    start_id = _last_token_id(tokenizer, "<think>")
    if start_id == end_id:
        # Both tags collapsed to a shared trailing piece (a bare '>'): the
        # vocab has no think tokens at all, and arming on that piece would
        # trip on every '>' the model emits.
        return (None, None)
    return ((start_id,) if start_id is not None else None, (end_id,))


class ThinkingBudgetProcessor:
    """mlx-lm logits processor that caps generated thinking tokens.

    Called as ``processor(tokens, logits)`` where ``tokens`` is the *full*
    accumulated sequence (prompt + everything generated so far) and ``logits``
    is ``(1, vocab)`` for the next token. Only the generated suffix is counted:
    a length baseline taken on the first call skips the prompt. When the count
    of generated thinking tokens exceeds ``budget``, the next steps' logits are
    forced (one-hot) through ``forced_ids``; generation then continues normally.
    A natural ``end_seq`` before the budget disarms the processor.
    ``budget=None`` never trips on its own: the processor is armed only for an
    external ``request_close()`` (the ^T finish-thinking key), which closes the
    block through ``skip_ids`` and then behaves exactly like a spent budget.

    ``start_seq`` / ``end_seq`` are token-id tuples. They are multi-token for
    channel-style delimiters (``<|channel>thought`` / ``<channel|>``) and length
    one for ``<think>`` / ``</think>``; matching is on the token suffix.

    A *forced* close leaves the model at a seam it was never trained on.
    Interleaved-thinking models (MiniMax-M3) legitimately alternate think
    and answer segments, so banning either structural exit backfires: an
    outright reopen ban plus an EOS ban livelocks the model into escaping
    through stray close tags and restarting its answer forever. Instead,
    once the budget is spent the processor works *with* the format:

    * a reopened thinking block is closed on the spot (budget-0 semantics
      for the rest of the turn, via the terse ``reclose_ids`` - the wrap-up
      phrase belongs to the first close only) - reopening costs a few
      tokens, never uncapped thinking, and the empty block reads as
      "considered, moving on";
    * EOS is banned only as a floor: the first token after each forced
      close cannot be EOS (no answerless turns); the first real answer
      token lifts it, and answer boundaries are otherwise untouched;
    * three forced closes with no answer token in between mean the model
      is cycling open/close - the processor stands down (``done``) rather
      than fence with it.

    A natural close before the budget disarms everything, as ever.
    """

    def __init__(
        self,
        *,
        end_seq,
        forced_ids,
        budget,
        start_seq=None,
        start_in_thinking=True,
        eos_ids=None,
        reclose_ids=None,
        skip_ids=None,
    ):
        self.end_seq = tuple(end_seq)
        self.start_seq = tuple(start_seq) if start_seq else None
        self.forced_ids = list(forced_ids)
        self.reclose_ids = list(reclose_ids) if reclose_ids else self.forced_ids
        self.skip_ids = list(skip_ids) if skip_ids else self.forced_ids
        self.budget = None if budget is None else max(0, int(budget))
        self.in_thinking = bool(start_in_thinking)
        self.count = 0
        self._close_requested = False
        self._baseline = None  # len(tokens) at first call -> skip the prompt
        self._forcing = False
        self._forced_idx = 0
        self._force_seq = self.forced_ids
        self.done = False
        self.eos_ids = [int(t) for t in (eos_ids or [])]
        self._spent = False  # a forced close happened; budget-0 from here on
        self._answer_pending = False  # no answer token since the last close
        self._landing = False  # next end_seq match is our own forced close
        self._strikes = 0
        self._ban_rows = {}

    def _one_hot(self, logits, tok_id):
        mask = mx.arange(logits.shape[-1]) == tok_id
        row = mx.where(mask, 0.0, -1e9).astype(logits.dtype)
        return mx.broadcast_to(row, logits.shape)

    def _ban_row(self, ids, logits):
        row = self._ban_rows.get(ids)
        if row is None:
            ar = mx.arange(logits.shape[-1])
            hit = ar == ids[0]
            for t in ids[1:]:
                hit = hit | (ar == t)
            row = mx.where(hit, -1e9, 0.0)
            self._ban_rows[ids] = row
        return row.astype(logits.dtype)

    @staticmethod
    def _ends_with(tokens, seq):
        k = len(seq)
        if k == 0 or tokens.shape[0] < k:
            return False
        return [int(x) for x in tokens[-k:].tolist()] == list(seq)

    def _strike(self):
        self._strikes += 1
        if self._strikes >= 3:
            self.done = True

    def request_close(self):
        """Finish thinking now (^T): the next step force-closes the open
        thinking block exactly as if the budget had just been exceeded.
        Consumed as a no-op when the model is not thinking."""
        self._close_requested = True

    def __call__(self, tokens, logits):
        n = tokens.shape[0]
        if self._baseline is None:
            # First call: everything so far is the prompt; don't count it.
            self._baseline = n
        elif n > self._baseline and not self.done:
            # Exactly one new generated token per step after the first.
            self._baseline = n
            if not self._forcing:
                if self._ends_with(tokens, self.end_seq):
                    self.in_thinking = False
                    if self._landing:
                        self._landing = False  # our own forced close arriving
                    elif not self._spent:
                        self.done = True  # natural close - disarm
                    elif self._answer_pending:
                        self._strike()  # stray close with no answer between
                elif self.start_seq is not None and self._ends_with(
                    tokens, self.start_seq
                ):
                    self.in_thinking = True
                    self.count = 0
                    # A reopen after a forced close trips the forcing
                    # threshold immediately: closed on the spot.
                    if self._spent:
                        self._close_requested = True
                elif self.in_thinking:
                    self.count += 1
                elif self._answer_pending:
                    # A real answer token landed; the turn may end freely.
                    self._answer_pending = False
                    self._strikes = 0
        if self.done:
            return logits
        over = self.budget is not None and self.count > self.budget
        if not self._forcing and self.in_thinking and (
                over or self._close_requested):
            self._forcing = True
            self._forced_idx = 0
            # The first close carries a wrap-up phrase (the model must SEE
            # itself decide to answer, or the cut thought continues untagged
            # in the answer); spent-mode recloses are terse.
            if self._spent:
                self._force_seq = self.reclose_ids
            elif self._close_requested:
                self._force_seq = self.skip_ids
            else:
                self._force_seq = self.forced_ids
        self._close_requested = False
        if self._forcing:
            fid = self._force_seq[self._forced_idx]
            self._forced_idx += 1
            if self._forced_idx >= len(self._force_seq):
                self._forcing = False
                self.in_thinking = False
                self._spent = True
                self._landing = True
                if self._answer_pending:
                    self._strike()  # closed again without any answer between
                self._answer_pending = True
            return self._one_hot(logits, fid)
        if self._answer_pending and self.eos_ids:
            # Floor: the first token after a forced close cannot be EOS.
            return logits + self._ban_row(tuple(self.eos_ids), logits)
        return logits


def _eos_id_list(tokenizer) -> list[int]:
    """Sorted EOS ids; tolerates the attr being an int (raw GGUF-built
    tokenizers), a set/list (mlx-lm wrapper), or absent."""
    eos = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return sorted(int(t) for t in eos)


def think_tokenizer_for(processor):
    """A think-capable tokenizer for an mlx-vlm processor.

    The raw GGUF-built HF tokenizer carries no thinking markers (gemma4's
    channel format lives neither in the template text nor as single-token
    tag spellings), so resolve them by wrapping with mlx-lm's
    ``TokenizerWrapper`` inference - once, cached on the processor (the
    inference scans the vocab). Falls back to the raw tokenizer."""
    cached = getattr(processor, "_gmlx_think_tokenizer", None)
    if cached is not None:
        return cached
    tok = getattr(processor, "tokenizer", processor)
    wrapped = tok
    if not getattr(tok, "think_end_tokens", None):
        try:
            from mlx_lm.tokenizer_utils import TokenizerWrapper

            wrapped = TokenizerWrapper(
                tok, eos_token_ids=getattr(tok, "_gguf_eos_token_ids", None)
            )
        except Exception:  # noqa: BLE001 - best-effort; ^T just stays off
            wrapped = tok
    try:
        processor._gmlx_think_tokenizer = wrapped
    except Exception:  # noqa: BLE001 - unsettable processor: skip the cache
        pass
    return wrapped


def make_thinking_budget_processor(
    tokenizer, budget, *, start_in_thinking=True, verbose=False,
    eos_floor=True, interruptible=False, start_token=None, end_token=None,
):
    """Build a thinking-budget logits processor, or ``None`` if unsupported.

    ``interruptible`` builds the processor even with no budget (``None``): it
    then never trips on its own but stays armed for the ^T finish-thinking
    key (``request_close``). Without it, ``budget=None`` returns ``None`` as
    ever.

    Resolves the model's thinking delimiters via mlx-lm's tokenizer inference
    (``<think>``, ``<longcat_think>``, and the ``<|channel>thought`` /
    ``<channel|>`` channel format), so the cap is not tied to one spelling.
    ``start_token``/``end_token`` (a family card's or the user's configured
    markers) override that inference for a model it does not recognize.
    Returns ``None`` (warning when ``verbose``) when no thinking-end token is
    detected; the caller then generates uncapped. ``budget=0`` force-closes as
    soon as a thinking block opens.

    ``eos_floor`` arms the post-forced-close EOS floor (see
    ``ThinkingBudgetProcessor``): the first token after a forced close cannot
    be the tokenizer's EOS, so a mid-thought cut cannot end the turn with no
    answer at all. Pass False for callers that must observe the model's
    untouched stop behavior, e.g. the over-generation critique probe.

    When ``budget > 0`` the first forced close is prefixed with
    ``_BUDGET_WRAP_PHRASE`` inside the thinking block: a bare close ambushes
    the model mid-thought, and the unresolved reasoning then tends to
    continue *untagged* in the answer (observed as endless post-code
    self-review on MiniMax-M3). Seeing itself decide to finalize keeps the
    trajectory coherent. Spent-mode recloses stay terse (``\\n`` + close).
    """
    if budget is None and not interruptible:
        return None
    if budget is not None and budget < 0:
        return None
    if budget is None:
        # An implicit (^T-only) processor is best-effort: a tokenizer the
        # probe can't handle must not break plain generation.
        try:
            start_seq, end_seq = _thinking_token_seqs(
                tokenizer, start_token, end_token)
        except Exception:  # noqa: BLE001
            return None
    else:
        start_seq, end_seq = _thinking_token_seqs(
            tokenizer, start_token, end_token)
    if not end_seq:
        if verbose and budget is not None:
            print(
                "[thinking-budget] no thinking-end token detected; "
                "ignoring thinking budget"
            )
        return None
    nl_id = _last_token_id(tokenizer, "\n")
    reclose_ids = ([nl_id] if nl_id is not None else []) + list(end_seq)
    forced_ids = reclose_ids
    skip_ids = reclose_ids
    if budget is None or budget > 0:
        wrap = _encode_ids(tokenizer, _BUDGET_WRAP_PHRASE)
        if wrap:
            forced_ids = list(wrap) + list(end_seq)
        wrap = _encode_ids(tokenizer, _SKIP_WRAP_PHRASE)
        if wrap:
            skip_ids = list(wrap) + list(end_seq)
    if verbose and budget is not None:
        # Decode the actual forced ids first: the wrapper's think_end attr can
        # disagree with the resolved sequence (template-preferred spelling).
        try:
            end_str = tokenizer.decode(list(end_seq))
        except Exception:
            end_str = getattr(tokenizer, "think_end", None) or "</think>"
        print(
            f"[thinking-budget] capping thinking at ~{budget} tokens "
            f"(force-close {end_str!r})"
        )
    return ThinkingBudgetProcessor(
        end_seq=end_seq,
        forced_ids=forced_ids,
        budget=budget,
        start_seq=start_seq,
        start_in_thinking=start_in_thinking,
        eos_ids=_eos_id_list(tokenizer) if eos_floor else [],
        reclose_ids=reclose_ids,
        skip_ids=skip_ids,
    )


# --- ^T: finish thinking now -------------------------------------------------
#
# The chat REPL and `gmlx run` route Ctrl-T here. In a cooked (or cbreak) tty
# ^T is the BSD status character: the terminal driver turns it into SIGINFO,
# which install_finish_thinking_key catches; chat's mid-reply key listener
# also forwards a raw \x14 byte in case the status character is unset. Either
# way the in-flight generation's processor gets request_close() and the open
# thinking block is closed as if the budget had just run out.

_target = None  # the in-flight generation's processor, if any


def set_finish_key_target(processor) -> None:
    """Expose ``processor`` (may be None) to the ^T finish-thinking trigger
    for the duration of one generation; pair with clear_finish_key_target."""
    global _target
    _target = processor


def clear_finish_key_target() -> None:
    global _target
    _target = None


class FinishKeyUnsupported:
    """Armed as the ^T target on generation paths the key can't reach (the
    MTP walks expose no logits-processor seam): pressing it then explains
    itself once instead of silently doing nothing."""

    def __init__(self, why: str):
        self.why = why
        self._told = False

    def note(self) -> None:
        if not self._told:
            self._told = True
            print(f"\n[^T] finish-thinking isn't available {self.why}",
                  file=sys.stderr)


def finish_thinking_now() -> bool:
    """Ask the in-flight generation to close its thinking block. True when a
    generation was listening (the request may still be a no-op if the model
    is not currently thinking)."""
    p = _target
    if p is None:
        return False
    if isinstance(p, FinishKeyUnsupported):
        p.note()
        return False
    if p.done:
        return False
    p.request_close()
    return True


def _quiet_kernel_status() -> None:
    # Besides SIGINFO, ^T makes the kernel print a "load: ..." status line
    # straight onto the tty, mid-stream; NOKERNINFO turns that off. Python's
    # termios doesn't export the flag, so use the <sys/termios.h> value. The
    # shell restores its own tty state at the next prompt.
    try:
        import termios
        nokerninfo = getattr(termios, "NOKERNINFO", 0x02000000)
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
        attrs = termios.tcgetattr(fd)
        attrs[3] |= nokerninfo
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:  # noqa: BLE001 - cosmetic; never block generation
        pass


def install_finish_thinking_key() -> bool:
    """Route ^T (SIGINFO, Darwin/BSD) to ``finish_thinking_now``. Returns
    whether the handler is installed; False off-platform or off the main
    thread. Safe to call more than once."""
    if not hasattr(signal, "SIGINFO"):
        return False
    try:
        signal.signal(signal.SIGINFO, lambda signum, frame: finish_thinking_now())
    except ValueError:
        return False
    _quiet_kernel_status()
    return True
