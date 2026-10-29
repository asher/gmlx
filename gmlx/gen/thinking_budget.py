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

The MTP path has no per-step logits seam, so the ^T finish-thinking key is
served there by :class:`MTPFinishThinking` instead: the owned round loop
commits the forced close sequence directly as fully-accepted verify rounds.
"""

from __future__ import annotations

import functools
import os
import signal
import sys

import mlx.core as mx


# Harmony's analysis pair. Phi-3 also ends turns with <|end|>, so this pair
# is matched by its start tag only, and the single-token vocab probe skips
# it.
_HARMONY_PAIR = ("<|channel|>analysis<|message|>", "<|end|>")
_START_ONLY_PAIRS = (_HARMONY_PAIR,)

# Think-tag spellings scanned when the caller relies on the defaults. The
# suffixed pair is Hy3's (every control tag carries ':opensource'; its chat
# template pre-fills the open tag at reasoning_effort low/high); the mm pair
# is MiniMax-M3's. '<think>' cannot false-match inside '<think:opensource>'
# (the '>' differs) or '<mm:think>' (no leading '<'), so the pairs are
# disjoint and scanning all of them is safe.
_THINK_PAIRS = (("<think>", "</think>"),
                ("<think:opensource>", "</think:opensource>"),
                ("<think:6124c78e>", "</think:6124c78e>"),
                ("<mm:think>", "</mm:think>"),
                # Kimi-K3 XTML sections (multi-token markers; the prompt
                # pre-opens the think section).
                ("<|open|>think<|sep|>", "<|close|>think<|sep|>"),
                # Onyx ATEM (Muse Glimmer, Muse Spark): the reasoning message
                # is addressed to self and ends at <|eom|>. Both literals
                # appear verbatim in the template source. The generation
                # prompt stops one marker short (at "<|start|>assistant"), so
                # the block is not prompt-opened and the model emits the whole
                # header itself.
                ("<|start|>assistant to=self<|message|>", "<|eom|>"),
                # K2 Horizon: one pair per reasoning_effort (high, medium,
                # low). The template names all three and pre-opens the one
                # the effort picks, so the prompt decides the pair.
                ("<ifm|think>", "</ifm|think>"),
                ("<ifm|think_fast>", "</ifm|think_fast>"),
                ("<ifm|think_faster>", "</ifm|think_faster>"),
                # Harmony (gpt-oss, Beam): the analysis channel. Last, so a
                # real pair keeps priority. The family markers carry the full
                # "<|start|>assistant..." header.
                _HARMONY_PAIR)

# Other close tags a model ends a pair's block with. The K2 Horizon 7B ends
# its medium and low blocks with </ifm|think>; MoVA uses each pair's own
# tag. A close is detected on either, and a forced close writes the pair's
# own tag.
_END_ALIASES = {
    "</ifm|think_fast>": ("</ifm|think>",),
    "</ifm|think_faster>": ("</ifm|think>",),
}


def end_aliases(end_tag) -> tuple:
    """The other close tags that end ``end_tag``'s block."""
    return _END_ALIASES.get(end_tag, ()) if isinstance(end_tag, str) else ()


def normalize_end_aliases(text, end_tag):
    """``text`` with each alias close of ``end_tag`` spelled as ``end_tag``."""
    if not isinstance(text, str):
        return text
    for alias in end_aliases(end_tag):
        text = text.replace(alias, end_tag)
    return text


# Text that also ends a harmony analysis body, as in llama.cpp's harmony
# parsers: a new message header with no <|end|> first, and <|constrain|>.
# These end a block for the budget only, and only while it is open. They
# never rewrite text and never count as stray closes. Inside a header,
# <|constrain|> is the tool-call constraint, and a header is never inside
# an open block.
_SOFT_CLOSES = {
    _HARMONY_PAIR[0]: ("<|start|>assistant", "<|constrain|>"),
}


def soft_end_seqs(tokenizer, start_seq) -> tuple:
    """Token ids of the budget-only closes of the block ``start_seq`` opens."""
    if not start_seq:
        return ()
    try:
        text = tokenizer.decode(list(start_seq))
    except Exception:  # noqa: BLE001 - undecodable: no soft closes
        return ()
    for key, closes in _SOFT_CLOSES.items():
        if key in text:
            return tuple(s for s in (marker_ids(tokenizer, c) for c in closes)
                         if s)
    return ()


# Harmony answers on the final channel. When the request declares no tools,
# a forced close also opens that channel: after a cut analysis block, Beam
# otherwise calls a bash tool that was never declared.
_ANSWER_STEER = {
    _HARMONY_PAIR[0]: "<|start|>assistant<|channel|>final<|message|>",
}


def answer_steer_ids(tokenizer, start_seq) -> list:
    """Token ids a forced close of the block ``start_seq`` opens adds to
    lead into the answer, or [] for formats with no answer header."""
    if not start_seq:
        return []
    try:
        text = tokenizer.decode(list(start_seq))
    except Exception:  # noqa: BLE001 - undecodable: no steer
        return []
    for key, header in _ANSWER_STEER.items():
        if key in text:
            return list(marker_ids(tokenizer, header))
    return []


def _report_stand_down(forced_closes: int) -> None:
    print(f"[thinking] budget stood down after {forced_closes} forced closes",
          file=sys.stderr, flush=True)


def alias_end_seqs(tokenizer, end_seq) -> tuple:
    """Token ids of each alias close of the pair ``end_seq`` closes."""
    if not end_seq:
        return ()
    try:
        text = tokenizer.decode(list(end_seq))
    except Exception:  # noqa: BLE001 - undecodable: no aliases
        return ()
    return tuple(s for s in (marker_ids(tokenizer, a)
                             for a in end_aliases(text)) if s)

# Forced into the thinking block ahead of the first budget-triggered close.
# The model must see itself DECIDE to answer: a bare close tag cuts the
# thought mid-sentence, and the unresolved reasoning tends to resume untagged
# inside the answer (endless self-review). First person, decisive, final.
_BUDGET_WRAP_PHRASE = (
    "\n\nI've hit my thinking budget, so I'll stop reasoning here and "
    "write the complete final answer now.\n"
)
# the phrase a client finds at the end of a trace the budget cut
BUDGET_WRAP_PHRASE = _BUDGET_WRAP_PHRASE

# Forced when the user presses ^T (finish thinking now) - same rationale,
# but the model should see the user's request, not a phantom budget.
_SKIP_WRAP_PHRASE = (
    "\n\nI've been asked to wrap up, so I'll stop reasoning here and "
    "write the complete final answer now.\n"
)


def think_pairs_in(template) -> list:
    """Every ``_THINK_PAIRS`` spelling ``template`` names, in table order.
    Most templates name one; K2 Horizon names one per reasoning effort."""
    if not isinstance(template, str) or not template:
        return []
    pairs = [p for p in _THINK_PAIRS if p[0] in template
             or (p[1] in template and p not in _START_ONLY_PAIRS)]
    # Kimi-K3 builds its markers by macro concatenation ('<|open|>' + tag +
    # '<|sep|>'), so the literal pair never appears in the template source;
    # detect the macro call on the 'think' tag instead.
    if not pairs and "<|open|>" in template and "open_tag('think" in template:
        pairs = [("<|open|>think<|sep|>", "<|close|>think<|sep|>")]
    return pairs


def template_think_pairs(tokenizer) -> list:
    """:func:`think_pairs_in` for the tokenizer's chat template."""
    return think_pairs_in(getattr(tokenizer, "chat_template", None))


def _template_think_pair(tokenizer):
    """The ``_THINK_PAIRS`` spelling the model's chat template actually uses,
    or None. The template is authoritative over vocab probing: a vocab can
    carry legacy ``</think>`` entries while the template thinks in another
    spelling (MiniMax-M3 ships both ``</think>`` and ``</mm:think>`` as real
    single tokens but only ever emits the latter), and probing then resolves
    - and forces - a tag the model treats as ordinary text. A template that
    names several pairs gives the first; :func:`prompt_think_pair` picks the
    one a rendered prompt opens."""
    pairs = template_think_pairs(tokenizer)
    return pairs[0] if pairs else None


def open_think_pair(text, pairs):
    """The pair in ``pairs`` whose block ``text`` ends inside (its last start
    marker not yet closed by a later end marker), or None."""
    if not isinstance(text, str):
        return None
    for start, end in pairs:
        i = text.rfind(start)
        if i != -1 and i > text.rfind(end):
            return (start, end)
    return None


def vlm_budget_kwargs(budget, start, end, processor, prompt=None) -> dict:
    """The thinking kwargs for mlx-vlm's ``generate`` / ``stream_generate``.
    Its budget criteria acts only with ``enable_thinking`` and watches
    ``<think>`` by default. With a budget: the given markers, else the chat
    template's pair (the one ``prompt`` opens, else the first), and
    ``enable_thinking=True``, so an explicit budget holds whatever the
    template's switch says, as on the text path and serve. A template with
    no known pair (gemma-4's channel tags) leaves the criteria unarmed."""
    if budget is not None and not (start or end):
        tok = getattr(processor, "tokenizer", processor)
        pairs = think_pairs_in(getattr(processor, "chat_template", None)
                               or getattr(tok, "chat_template", None))
        if pairs:
            start, end = open_think_pair(prompt, pairs) or pairs[0]
    out = {}
    if budget is not None and (start or end):
        out["enable_thinking"] = True
    if start:
        out["thinking_start_token"] = start
    if end:
        out["thinking_end_token"] = end
    return out


def prompt_think_pair(prompt, tokenizer):
    """The template pair a rendered ``prompt`` ends inside, or None.
    ``prompt`` is the rendered text or its token ids, whose tail is decoded.
    With K2 Horizon this is the pair the request's reasoning effort chose."""
    pairs = template_think_pairs(tokenizer)
    if not pairs or prompt is None:
        return None
    if not isinstance(prompt, str):
        try:
            ids = prompt.tolist() if hasattr(prompt, "tolist") else list(prompt)
            while ids and isinstance(ids[0], list):
                ids = ids[-1]
            prompt = tokenizer.decode([int(t) for t in ids[-64:]])
        except Exception:  # noqa: BLE001 - no decodable tail: no pair
            return None
    return open_think_pair(prompt, pairs)


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
            tpairs = template_think_pairs(tokenizer)
            s = getattr(tokenizer, "think_start", None)
            e = getattr(tokenizer, "think_end", None)
            if tpairs:
                pairs = tpairs
            elif s and e:
                pairs = [(s, e)]
    if pairs == [("<think>", "</think>")]:
        pairs = list(_THINK_PAIRS)
    pair = open_think_pair(prompt, pairs)
    return pair[0] if pair else None


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


def _opens_marker(tail, seq) -> bool:
    """Whether ``tail`` ends partway into the multi-token marker ``seq``.
    The leading pieces of a start marker (ATEM's ``<|start|>assistant``)
    are not answer tokens yet."""
    if not seq or len(seq) < 2:
        return False
    tail = [int(t) for t in tail]
    return any(tail[-j:] == list(seq[:j]) for j in range(1, len(seq)))


def _encode_ids(tokenizer, text: str):
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def marker_ids(tokenizer, text) -> tuple:
    """Every token id of a marker. mlx-vlm keeps only the last one, which
    for ATEM's ``<|start|>assistant to=self<|message|>`` is the
    ``<|message|>`` that opens every message, answer and commentary too."""
    if not text:
        return ()
    return tuple(int(t) for t in _encode_ids(tokenizer, text))


def ends_with_seq(tail, seq) -> bool:
    return bool(seq) and len(tail) >= len(seq) and tail[-len(seq):] == list(seq)


def rfind_seq(toks, seq) -> int:
    """Start index of the last occurrence of ``seq`` in ``toks``, or -1."""
    k = len(seq)
    if not k:
        return -1
    last, head = seq[-1], list(seq)
    for i in range(len(toks) - 1, k - 2, -1):
        if toks[i] == last and toks[i - k + 1:i + 1] == head:
            return i - k + 1
    return -1


def prompt_opens_block(toks, start_seq, end_seq) -> bool:
    """Whether the token list ``toks`` ends inside a block: its last start
    marker comes after its last end marker. Markers match on all their
    tokens."""
    last_start = rfind_seq(toks, start_seq)
    return last_start >= 0 and last_start > rfind_seq(toks, end_seq)


def prompt_tail_ids(input_ids, n: int = 16) -> list:
    """The last ``n`` token ids of ``input_ids`` (an array or a list) as a
    list. ``n`` is longer than any thinking marker."""
    if input_ids is None:
        return []
    flat = input_ids.flatten() if hasattr(input_ids, "flatten") else input_ids
    tail = flat[-n:]
    return [int(t) for t in (tail.tolist() if hasattr(tail, "tolist") else tail)]


@functools.cache
def seq_thinking_budget_criteria_cls():
    """Subclass of mlx-vlm's ``ThinkingBudgetCriteria`` that matches each
    marker on all of its tokens. A marker that spans several tokens opens or
    closes a block only when the whole sequence arrives, and the forced
    close emits the whole end marker. ``prompt_ids`` gives the prompt's last
    tokens, so a start marker the generation prompt begins (ATEM's prompt
    ends ``<|start|>assistant``) completes on the generated rest.
    ``prompt_open_thinking`` is the in-thinking state at the start, kept
    apart from the stock ``enable_thinking`` gates. Single-token markers
    behave as the stock criteria. Built lazily so importing this module
    never imports mlx-vlm."""
    from gmlx._vendor.mlx_vlm.utils import ThinkingBudgetCriteria

    class SeqThinkingBudgetCriteria(ThinkingBudgetCriteria):
        def __init__(self, tokenizer, thinking_budget,
                     thinking_end_token="</think>", thinking_start_token=None,
                     *, prompt_open_thinking=False, prompt_ids=None,
                     tools=False, **kw):
            super().__init__(tokenizer, thinking_budget,
                             thinking_end_token=thinking_end_token,
                             thinking_start_token=thinking_start_token, **kw)
            self._start_seq = marker_ids(tokenizer, thinking_start_token)
            self._end_seq = marker_ids(tokenizer, thinking_end_token)
            self._end_seqs = (self._end_seq,) + tuple(
                s for s in (marker_ids(tokenizer, a)
                            for a in end_aliases(thinking_end_token)) if s)
            self._soft_end_seqs = soft_end_seqs(tokenizer, self._start_seq)
            self._forced_sequence = (self._forced_sequence[:-1]
                                     + list(self._end_seq))
            # Fed after the close lands, as our own tokens.
            self._steer = ([] if tools
                           else answer_steer_ids(tokenizer, self._start_seq))
            self._steer_left: list[int] = []
            self._steering = False
            self._tail_max = max(len(self._start_seq),
                                 *(len(s) for s in self._end_seqs),
                                 *(len(s) for s in self._soft_end_seqs))
            keep = self._tail_max - 1
            self._prompt_tail = ([int(t) for t in prompt_ids[-keep:]]
                                 if prompt_ids is not None and keep else [])
            self._tail = list(self._prompt_tail)
            self._prompt_open = bool(prompt_open_thinking)
            self.in_thinking = self._prompt_open
            self._reset_stand_down()

        def _reset_stand_down(self):
            # As in ThinkingBudgetProcessor: three forced closes with no
            # answer token between them stand the budget down.
            self.done = False
            self._spent = False
            self._answer_pending = False
            self._strikes = 0
            self._forced_closes = 0

        def reset_thinking_state(self):
            super().reset_thinking_state()
            self.in_thinking = self._prompt_open
            self._tail = list(self._prompt_tail)
            self._steer_left = []
            self._steering = False
            self._reset_stand_down()

        def _strike(self):
            self._strikes += 1
            if self._strikes >= 3:
                self.done = True
                self.budget_exceeded = False
                _report_stand_down(self._forced_closes)

        def _steer_next(self):
            if not self._steer_left:
                self._steering = False
                self.forced_token_id = None
                return None
            forced = self._steer_left.pop(0)
            self.forced_token_id = forced
            return forced

        def __call__(self, token_id):
            self._tail.append(int(token_id))
            del self._tail[:-self._tail_max]
            if self._steering:
                return self._steer_next()
            if self.done:
                self.forced_token_id = None
                return None
            if self.enable_thinking and ends_with_seq(self._tail, self._start_seq):
                self.in_thinking = True
                return None
            if any(ends_with_seq(self._tail, s) for s in self._end_seqs):
                # Also a commentary <|eom|> outside reasoning, where it
                # changes nothing.
                landing = self.budget_exceeded
                self.in_thinking = False
                self.budget_exceeded = False
                self._forced_index = 0
                if landing:
                    # A forced close just landed.
                    self._forced_closes += 1
                    self._spent = True
                    if self._answer_pending:
                        self._strike()
                    self._answer_pending = True
                    if self._steer:
                        self._steering = True
                        self._steer_left = list(self._steer)
                        return self._steer_next()
                elif self._spent and self._answer_pending:
                    self._strike()  # stray close with no answer between
                return None
            if self.in_thinking and not self.budget_exceeded and any(
                    ends_with_seq(self._tail, s) for s in self._soft_end_seqs):
                self.in_thinking = False
                return None
            if (not self.in_thinking and self._answer_pending
                    and not _opens_marker(self._tail, self._start_seq)):
                self._answer_pending = False
                self._strikes = 0
            if self.in_thinking:
                self.thinking_token_count += 1
                if self.thinking_token_count > self.thinking_budget:
                    self.budget_exceeded = True
            if (self.budget_exceeded
                    and self._forced_index < len(self._forced_sequence)):
                forced = self._forced_sequence[self._forced_index]
                self._forced_index += 1
                self.forced_token_id = forced
                return forced
            self.forced_token_id = None
            return None

    return SeqThinkingBudgetCriteria


# The final-answer headers of formats whose answer opens with a header of
# its own, keyed by the reasoning start marker. ATEM's reasoning closes on
# <|eom|>, which also closes commentary and tool calls, and the answer text
# starts only after "<|start|>assistant to=user<|message|>". The recipient
# is optional, as in llama.cpp's Muse parsers.
# Harmony's answer is the final channel, with the JSON-constrained variant
# llama.cpp's response-format rule accepts.
_ANSWER_HEADERS = {
    "<|start|>assistant to=self<|message|>": (
        "<|start|>assistant to=user<|message|>",
        "<|start|>assistant<|message|>"),
    "<|start|>assistant<|channel|>analysis<|message|>": (
        "<|channel|>final<|message|>",
        "<|channel|>final <|constrain|>json<|message|>"),
}


def answer_opener_seqs(tokenizer, start_token, end_token) -> tuple:
    """The token sequences after which the final answer starts: the
    format's answer headers, else the whole end marker."""
    texts = _ANSWER_HEADERS.get(start_token or "") or (end_token,)
    return tuple(s for s in (marker_ids(tokenizer, t) for t in texts) if s)


def has_answer_header(start_token) -> bool:
    """Whether the answer of this marker format opens with its own header,
    so a grammar must wait for it in every request, thinking or not."""
    return (start_token or "") in _ANSWER_HEADERS


class AnswerGate:
    """Finds where the final answer starts in a token stream, so a grammar
    constrains the answer only. The answer starts on the token after any
    sequence in ``seqs``, matched on all its tokens. ``tail`` holds the
    last tokens seen, so an opener that the prompt begins completes on
    generated tokens."""

    def __init__(self, seqs, tail=()):
        self.seqs = tuple(tuple(int(t) for t in s) for s in seqs if s)
        self.keep = max((len(s) for s in self.seqs), default=1)
        self.tail = [int(t) for t in tail][-self.keep:]

    def restart(self, tail=()) -> AnswerGate:
        """A fresh gate on the same openers, seeded with ``tail``."""
        return AnswerGate(self.seqs, tail)

    def opened(self) -> bool:
        """Whether the tokens seen end with an opener."""
        return any(ends_with_seq(self.tail, s) for s in self.seqs)

    def feed(self, tok) -> bool:
        """Take one token. True when it completes an opener."""
        self.tail.append(int(tok))
        del self.tail[:-self.keep]
        return self.opened()

    def advance(self, tokens) -> int | None:
        """Take ``tokens``. The index of the first one that completes an
        opener, or None."""
        hit = None
        for j, t in enumerate(tokens):
            if self.feed(t) and hit is None:
                hit = j
        return hit

    def find(self, tokens) -> int | None:
        """As ``advance``, without changing the gate."""
        return self.restart(self.tail).advance(tokens)


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
    end tag does not encode at all (the caller then keeps probing). Each tag
    keeps its whole encoding, so a start marker that spans several tokens
    (ATEM's ``<|start|>assistant to=self<|message|>`` is five) still opens
    a block when its end marker is one token. ``start_tag`` may be None,
    which yields a None start_seq."""
    if not end_tag:
        return None
    end_ids = _encode_ids(tokenizer, end_tag)
    if not end_ids:
        return None
    start_ids = _encode_ids(tokenizer, start_tag) if start_tag else None
    return (tuple(int(t) for t in start_ids) if start_ids else None,
            tuple(int(t) for t in end_ids))


def _thinking_token_seqs(tokenizer, start_token=None, end_token=None,
                         prompt=None):
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
    markers are the user's word for a model neither one recognizes.
    ``prompt`` (the rendered prompt, text or token ids) picks among the pairs
    a template names: the one the prompt opens wins."""
    seqs = _tag_pair_seqs(tokenizer, start_token, end_token)
    if seqs is not None:
        return seqs
    pair = prompt_think_pair(prompt, tokenizer) if end_token is None else None
    if pair is not None:
        seqs = _tag_pair_seqs(tokenizer, *pair)
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
        if (start_tag, end_tag) in _START_ONLY_PAIRS:
            continue
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

    Called as ``processor(tokens, logits)`` where ``tokens`` is the sequence
    mlx-lm accumulates (the prompt's last token, then everything generated so
    far) and ``logits`` is ``(1, vocab)`` for the next token. Only the
    generated suffix is counted: a length baseline taken on the first call
    skips the prompt. When the count of generated thinking tokens exceeds
    ``budget``, the next steps' logits are forced (one-hot) through
    ``forced_ids``; generation then continues normally.
    A natural ``end_seq`` before the budget disarms the processor.
    ``budget=None`` never trips on its own: the processor is armed only for an
    external ``request_close()`` (the ^T finish-thinking key), which closes the
    block through ``skip_ids`` and then behaves exactly like a spent budget.

    ``start_seq`` / ``end_seq`` are token-id tuples. They are multi-token for
    channel-style delimiters (``<|channel>thought`` / ``<channel|>``) and
    ATEM's reasoning header, and length one for ``<think>`` / ``</think>``.
    Matching is on the suffix of the tokens seen so far (see ``_seed_tail``
    for a start marker that the prompt begins).

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
        alt_end_seqs=(),
        soft_end_seqs=(),
    ):
        self.end_seq = tuple(end_seq)
        self.alt_end_seqs = tuple(tuple(s) for s in alt_end_seqs if s)
        self.soft_end_seqs = tuple(tuple(s) for s in soft_end_seqs if s)
        self.start_seq = tuple(start_seq) if start_seq else None
        self.forced_ids = list(forced_ids)
        self.reclose_ids = list(reclose_ids) if reclose_ids else self.forced_ids
        self.skip_ids = list(skip_ids) if skip_ids else self.forced_ids
        self.budget = None if budget is None else max(0, int(budget))
        self.in_thinking = bool(start_in_thinking)
        self.count = 0
        self._close_requested = False
        self._baseline = None  # len(tokens) at first call -> skip the prompt
        self._tail: list[int] = []  # last tokens seen, for marker matching
        self._tail_max = max(len(self.end_seq), len(self.start_seq or ()), 1,
                             *(len(s) for s in self.alt_end_seqs),
                             *(len(s) for s in self.soft_end_seqs))
        self._forcing = False
        self._forced_idx = 0
        self._force_seq = self.forced_ids
        self.done = False
        self.eos_ids = [int(t) for t in (eos_ids or [])]
        self._spent = False  # a forced close happened; budget-0 from here on
        self._answer_pending = False  # no answer token since the last close
        self._landing = False  # next end_seq match is our own forced close
        self._strikes = 0
        self._forced_closes = 0
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

    def _tail_is(self, seq) -> bool:
        k = len(seq)
        return k > 0 and len(self._tail) >= k and self._tail[-k:] == list(seq)

    def _tail_closes(self) -> bool:
        return self._tail_is(self.end_seq) or any(
            self._tail_is(s) for s in self.alt_end_seqs)

    def _seed_tail(self, tokens) -> None:
        """Seed the marker tail from the prompt tokens of the first call.
        mlx-lm hands over only the prompt's last token, so when those tokens
        end a proper prefix of a multi-token start marker (ATEM's prompt
        ends "<|start|>assistant"), the tail takes that whole prefix and the
        generated rest of the marker completes it."""
        seen = [int(t) for t in tokens[-self._tail_max:].tolist()]
        seq = self.start_seq or ()
        for j in range(len(seq) - 1, 0, -1):
            m = min(len(seen), j)
            if m and seen[-m:] == list(seq[j - m:j]):
                self._tail = list(seq[:j])
                return
        self._tail = seen

    def _strike(self):
        self._strikes += 1
        if self._strikes >= 3:
            self.done = True
            _report_stand_down(self._forced_closes)

    def _tail_soft_closes(self) -> bool:
        return any(self._tail_is(s) for s in self.soft_end_seqs)

    def _outside_block(self) -> bool:
        """Whether an end marker now closes no block this processor saw
        open. Muse Spark ends a user-visible commentary message with the
        same <|eom|> that ends reasoning, and that must not disarm."""
        return (not self.in_thinking and not self._spent
                and self.start_seq is not None)

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
            self._seed_tail(tokens)
        elif n > self._baseline and not self.done:
            # Exactly one new generated token per step after the first.
            self._baseline = n
            self._tail.append(int(tokens[-1].item()))
            del self._tail[:-self._tail_max]
            if self._landing:
                # The last token of our own forced close. It may end in an
                # answer header, which is no answer token.
                self._landing = False
            elif not self._forcing:
                if self._tail_closes() and not self._outside_block():
                    self.in_thinking = False
                    if not self._spent:
                        self.done = True  # natural close - disarm
                    elif self._answer_pending:
                        self._strike()  # stray close with no answer between
                elif self.in_thinking and self._tail_soft_closes():
                    self.in_thinking = False
                    if not self._spent:
                        self.done = True  # natural close - disarm
                elif self.start_seq is not None and self._tail_is(
                    self.start_seq
                ):
                    self.in_thinking = True
                    self.count = 0
                    # A reopen after a forced close trips the forcing
                    # threshold immediately: closed on the spot.
                    if self._spent:
                        self._close_requested = True
                elif self.in_thinking:
                    self.count += 1
                elif self._answer_pending and not _opens_marker(
                        self._tail, self.start_seq):
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
                self._forced_closes += 1
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
            from gmlx._vendor.mlx_lm.tokenizer_utils import TokenizerWrapper

            wrapped = TokenizerWrapper(
                tok, eos_token_ids=getattr(tok, "_gguf_eos_token_ids", None)
            )
        except Exception:  # noqa: BLE001 - best-effort; ^T just stays off
            wrapped = tok
    try:
        processor._gmlx_think_tokenizer = wrapped
    except Exception:  # noqa: BLE001, S110 - unsettable processor: skip the cache
        pass
    return wrapped


def _forced_close_ids(tokenizer, budget, end_seq, steer=()):
    """``(reclose_ids, forced_ids, skip_ids)`` for a resolved ``end_seq``:
    the terse newline+close, and the budget/^T variants carrying their wrap
    phrases when the budget funds any thinking at all (``None`` or > 0).
    Each ends with ``steer`` (``answer_steer_ids``) when given."""
    nl_id = _last_token_id(tokenizer, "\n")
    close = list(end_seq) + list(steer)
    reclose_ids = ([nl_id] if nl_id is not None else []) + close
    forced_ids = reclose_ids
    skip_ids = reclose_ids
    if budget is None or budget > 0:
        wrap = _encode_ids(tokenizer, _BUDGET_WRAP_PHRASE)
        if wrap:
            forced_ids = list(wrap) + close
        wrap = _encode_ids(tokenizer, _SKIP_WRAP_PHRASE)
        if wrap:
            skip_ids = list(wrap) + close
    return reclose_ids, forced_ids, skip_ids


def budget_close_tokens(tokenizer, *, start_token=None, end_token=None) -> int:
    """Tokens the first budget-triggered close spends inside the thinking
    block (the wrap phrase and the closing marker), or 0 when no
    thinking-end token resolves, since the budget is then ignored."""
    try:
        _, end_seq = _thinking_token_seqs(tokenizer, start_token, end_token)
    except Exception:  # noqa: BLE001 - a bare tokenizer with no think tokens
        return 0
    if not end_seq:
        return 0
    return len(_forced_close_ids(tokenizer, 1, end_seq)[1])


def _announce_budget(tokenizer, budget, end_seq) -> None:
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


def make_thinking_budget_processor(
    tokenizer, budget, *, start_in_thinking=True, verbose=False,
    eos_floor=True, interruptible=False, start_token=None, end_token=None,
    prompt=None, tools=False,
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
    ``prompt`` (the rendered prompt) picks the pair it opens when the
    template names several, so the close matches the open. ``tools`` says
    the request declares tools; without them a harmony close also opens
    the final channel (``answer_steer_ids``).
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
                tokenizer, start_token, end_token, prompt)
        except Exception:  # noqa: BLE001
            return None
    else:
        start_seq, end_seq = _thinking_token_seqs(
            tokenizer, start_token, end_token, prompt)
    if not end_seq:
        if verbose and budget is not None:
            print(
                "[thinking-budget] no thinking-end token detected; "
                "ignoring thinking budget"
            )
        return None
    steer = [] if tools else answer_steer_ids(tokenizer, start_seq)
    reclose_ids, forced_ids, skip_ids = _forced_close_ids(
        tokenizer, budget, end_seq, steer)
    if verbose and budget is not None:
        _announce_budget(tokenizer, budget, end_seq)
    return ThinkingBudgetProcessor(
        end_seq=end_seq,
        forced_ids=forced_ids,
        budget=budget,
        start_seq=start_seq,
        start_in_thinking=start_in_thinking,
        eos_ids=_eos_id_list(tokenizer) if eos_floor else [],
        reclose_ids=reclose_ids,
        skip_ids=skip_ids,
        alt_end_seqs=alias_end_seqs(tokenizer, end_seq),
        soft_end_seqs=soft_end_seqs(tokenizer, start_seq),
    )


class MTPFinishThinking:
    """Thinking-budget + ^T finish-thinking hook for the owned MTP round loop.

    The MTP walks verify a whole draft block per target forward, so the
    per-step logits seam :class:`ThinkingBudgetProcessor` forces through does
    not exist there. But the forced close is a *known* token sequence, so the
    round loop can commit it directly as a fully-accepted verify block. This
    hook carries the state that decision needs: the loop reports every emitted
    token through :meth:`observe` (tracking whether the model is inside a
    thinking block, and counting thinking tokens against ``budget``), and asks
    :meth:`take_forced` at each round boundary for a close sequence to inject.

    Duck-typed for :func:`finish_thinking_now` (``done`` + ``request_close``).
    Semantics mirror the processor: an exceeded budget forces the close (a
    round-boundary check, so the cap is ~budget, overshooting by at most one
    draft block); the first budget close carries the budget wrap phrase and a
    ^T close the ^T one; a natural close before any forced one disarms the
    hook; a reopen after a forced close is reclosed tersely; three closes
    with no answer token in between stand the hook down. ``budget=None``
    never trips on its own (^T-only).
    """

    def __init__(self, *, end_seq, skip_ids, reclose_ids, forced_ids=None,
                 budget=None, start_seq=None, start_in_thinking=False,
                 prompt_ids=None, alt_end_seqs=(), soft_end_seqs=()):
        self.end_seq = tuple(end_seq)
        self.alt_end_seqs = tuple(tuple(s) for s in alt_end_seqs if s)
        self.soft_end_seqs = tuple(tuple(s) for s in soft_end_seqs if s)
        self.start_seq = tuple(start_seq) if start_seq else None
        self.skip_ids = list(skip_ids)
        self.reclose_ids = list(reclose_ids)
        self.forced_ids = list(forced_ids) if forced_ids else self.skip_ids
        self.budget = None if budget is None else max(0, int(budget))
        self.in_thinking = bool(start_in_thinking)
        self.count = 0
        self.done = False
        self._close_requested = False
        self._spent = False
        self._answer_pending = False
        self._strikes = 0
        self._forced_closes = 0
        self._own = 0  # tokens of the injected close still to observe
        self._tail_max = max(len(self.end_seq), len(self.start_seq or ()), 1,
                             *(len(s) for s in self.alt_end_seqs),
                             *(len(s) for s in self.soft_end_seqs))
        # The prompt's last tokens, so a start marker that the generation
        # prompt begins (ATEM's "<|start|>assistant") completes on the
        # generated tokens. Never a whole marker, so the prompt itself
        # opens and closes nothing here.
        keep = self._tail_max - 1
        self._tail: list[int] = (
            [int(t) for t in prompt_ids[-keep:]]
            if prompt_ids is not None and keep else [])

    def request_close(self):
        """Finish thinking now (^T). Consumed as a no-op at the next round
        boundary when the model is not thinking."""
        self._close_requested = True

    def _outside_block(self) -> bool:
        """Whether an end marker now closes no block this hook saw open,
        as in :meth:`ThinkingBudgetProcessor._outside_block`."""
        return (not self.in_thinking and not self._spent
                and self.start_seq is not None)

    def _tail_is(self, seq) -> bool:
        k = len(seq)
        return k > 0 and len(self._tail) >= k and self._tail[-k:] == list(seq)

    def observe(self, tok) -> None:
        """Track one emitted token. The tokens of an injected close only
        extend the tail: take_forced has already closed the block, and a
        close that ends in an answer header writes no answer token."""
        self._tail.append(int(tok))
        if len(self._tail) > self._tail_max:
            del self._tail[: len(self._tail) - self._tail_max]
        if self._own:
            self._own -= 1
            return
        closes = self._tail_is(self.end_seq) or any(
            self._tail_is(s) for s in self.alt_end_seqs)
        if closes and not self._outside_block():
            self.in_thinking = False
            if not self._spent:
                self.done = True  # natural close - disarm, as on the plain path
        elif self.in_thinking and any(self._tail_is(s)
                                      for s in self.soft_end_seqs):
            self.in_thinking = False
            if not self._spent:
                self.done = True
        elif self.start_seq is not None and self._tail_is(self.start_seq):
            self.in_thinking = True
            self.count = 0
            # A reopen after a forced close is reclosed at the next round
            # boundary (budget-0 semantics, as in the processor).
            if self._spent:
                self._close_requested = True
        elif self.in_thinking:
            self.count += 1
        elif self._answer_pending and not _opens_marker(
                self._tail, self.start_seq):
            # A real answer token landed; stop counting closes as strikes.
            self._answer_pending = False
            self._strikes = 0

    def take_forced(self):
        """The close sequence to inject at this round boundary, or None.
        Consumes any pending ^T request either way."""
        requested = self._close_requested
        self._close_requested = False
        if self.done or not self.in_thinking:
            return None
        over = self.budget is not None and self.count > self.budget
        if not (requested or over):
            return None
        if self._answer_pending:
            # Closed again without any answer in between: the model is
            # cycling open/close - stand down rather than fence with it.
            self._strikes += 1
            if self._strikes >= 3:
                self.done = True
                _report_stand_down(self._forced_closes)
                return None
        if self._spent:
            seq = self.reclose_ids
        elif requested:
            seq = self.skip_ids
        else:
            seq = self.forced_ids
        self._spent = True
        self._answer_pending = True
        self._forced_closes += 1
        self.in_thinking = False
        self._own = len(seq)
        return list(seq)


def make_mtp_finish_hook(
    tokenizer, *, budget=None, start_in_thinking=False, verbose=False,
    start_token=None, end_token=None, prompt_ids=None, prompt=None,
    tools=False,
):
    """Build the thinking-budget/^T hook for the owned MTP round loop, or
    ``None`` when the model's thinking delimiters can't be resolved (the
    budget is then ignored - warning when ``verbose`` - and the key is a
    silent no-op, matching the plain path). ``budget=None`` arms ^T only.
    Best-effort by design: a tokenizer the probe can't handle must not break
    MTP decode. ``prompt_ids`` (the prompt's token ids, a list or a 1-D
    array) lets a start marker that the prompt begins complete on the
    generated tokens. ``prompt`` (the rendered text, else ``prompt_ids``)
    picks the pair it opens when the template names several. ``tools`` is
    as in ``make_thinking_budget_processor``."""
    if budget is not None and budget < 0:
        return None
    try:
        start_seq, end_seq = _thinking_token_seqs(
            tokenizer, start_token, end_token,
            prompt if prompt is not None else prompt_ids)
    except Exception:  # noqa: BLE001
        return None
    if not end_seq:
        if verbose and budget is not None:
            print(
                "[thinking-budget] no thinking-end token detected; "
                "ignoring thinking budget"
            )
        return None
    steer = [] if tools else answer_steer_ids(tokenizer, start_seq)
    reclose_ids, forced_ids, skip_ids = _forced_close_ids(
        tokenizer, budget, end_seq, steer)
    if verbose and budget is not None:
        _announce_budget(tokenizer, budget, end_seq)
    return MTPFinishThinking(
        end_seq=end_seq,
        skip_ids=skip_ids,
        reclose_ids=reclose_ids,
        forced_ids=forced_ids,
        budget=budget,
        start_seq=start_seq,
        start_in_thinking=start_in_thinking,
        prompt_ids=prompt_ids,
        alt_end_seqs=alias_end_seqs(tokenizer, end_seq),
        soft_end_seqs=soft_end_seqs(tokenizer, start_seq),
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
    stock mlx-vlm MTP round exposes no forced-close seam; the owned engine
    re-arms with a real :class:`MTPFinishThinking`): pressing it then
    explains itself once instead of silently doing nothing."""

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
    except Exception:  # noqa: BLE001, S110 - cosmetic; never block generation
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
