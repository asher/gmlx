"""Unit tests for the thinking-budget logits processor (CPU, no model load).

Drives the processor's state machine with synthetic ``(tokens, logits)`` calls
that mimic mlx-lm's contract: ``tokens`` is the full accumulated sequence
(prompt + generated), growing by one per step after the first; ``logits`` is
``(1, vocab)``. Covers single-token ``<think>`` delimiters and the multi-token
``<|channel>thought`` / ``<channel|>`` channel format.
"""

import mlx.core as mx

from gmlx.thinking_budget import (
    ThinkingBudgetProcessor,
    _thinking_token_seqs,
    make_thinking_budget_processor,
    prompt_opens_thinking,
)

NL, END, START = 10, 100, 99
VOCAB = 200


class _FakeTok:
    """Maps the control strings to single-token ids; '' otherwise. No wrapper
    think-token API, so the factory falls back to encode() probing."""

    _MAP = {"\n": [NL], "</think>": [END], "<think>": [START]}

    def encode(self, text, add_special_tokens=True):
        return self._MAP.get(text, [])


def _logits():
    return mx.zeros((1, VOCAB))


def _argmax_if_forced(out):
    """A forced step returns a one-hot (one id at 0.0, rest deeply negative).
    Post-close BAN rows (few deeply negative ids) read as forced here too;
    tests that care about bans use ``_banned_ids`` instead."""
    row = out[0]
    if float(mx.min(row).item()) < -1e8:
        return int(mx.argmax(row).item())
    return None


def _banned_ids(out):
    """Ids a ban row suppresses; empty for passthrough AND one-hot rows."""
    neg = [i for i, v in enumerate(out[0].tolist()) if v < -1e8]
    return set(neg) if len(neg) < VOCAB - 1 else set()


def _run(processor, generated, prompt=(1, 2, START)):
    """Replay a decode: ``prompt``, then each id in ``generated`` one at a time.
    Returns the forced id per call (or None when passed through unchanged)."""
    tokens = list(prompt)
    forced = [_argmax_if_forced(processor(mx.array(tokens), _logits()))]
    for tid in generated:
        tokens.append(tid)
        forced.append(_argmax_if_forced(processor(mx.array(tokens), _logits())))
    return forced


# --- single-token <think> delimiters ---------------------------------------

def test_forces_close_sequence_after_budget():
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=3, start_seq=(START,)
    )
    forced = _run(p, [50, 51, 52, 53, 54, 55, 56])
    assert NL in forced and END in forced
    nl_at = forced.index(NL)
    assert forced[nl_at + 1] == END  # newline immediately followed by close
    assert all(f is None for f in forced[:nl_at])
    assert p._spent and not p.in_thinking


def test_natural_close_disarms():
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=100, start_seq=(START,)
    )
    forced = _run(p, [50, 51, END, 60, 61, 62])
    assert all(f is None for f in forced)
    assert p.done and not p.in_thinking


def test_prompt_tokens_not_counted():
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=2, start_seq=(START,)
    )
    big_prompt = list(range(1, 50)) + [START]
    p(mx.array(big_prompt), _logits())
    assert p.count == 0  # baseline only; nothing counted


def test_caps_generated_think_when_not_started_in_block():
    # enable_thinking false so we don't start in a block, but the model emits
    # <think> anyway: the budget must still fire.
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=2, start_seq=(START,),
        start_in_thinking=False,
    )
    assert p.in_thinking is False
    forced = _run(p, [START, 50, 51, 52, 53, 54], prompt=(1, 2, 3))
    assert NL in forced and END in forced
    assert forced.index(NL) < forced.index(END)
    assert p._spent


def test_noop_when_never_thinking_and_not_started_in_block():
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=2, start_seq=(START,),
        start_in_thinking=False,
    )
    forced = _run(p, [50, 51, 52, 53, 54, 55, 56], prompt=(1, 2, 3))
    assert all(f is None for f in forced)
    assert not p.in_thinking


# --- spent-mode steering after a forced close --------------------------------

EOS = 150


def test_forced_close_eos_floor_then_turn_may_end():
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=1, start_seq=(START,),
        eos_ids=[EOS],
    )
    tokens = [1, 2, START]
    p(mx.array(tokens), _logits())  # baseline call (prompt only)
    out = None
    for tid in (50, 51, NL, END):  # two counted, then the forced NL+END land
        tokens.append(tid)
        out = p(mx.array(tokens), _logits())
    assert p._spent and not p.done
    # Floor: the first post-close token cannot be EOS...
    assert _banned_ids(out) == {EOS}
    # ...but one answer token later the turn may end (or not) freely - no
    # reopen ban, no EOS ban, nothing to livelock against.
    tokens.append(60)
    assert _banned_ids(p(mx.array(tokens), _logits())) == set()
    tokens.append(61)
    assert _banned_ids(p(mx.array(tokens), _logits())) == set()


def test_spent_reopen_is_instantly_closed():
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=1, start_seq=(START,),
        eos_ids=[EOS],
    )
    tokens = [1, 2, START]
    p(mx.array(tokens), _logits())
    for tid in (50, 51, NL, END, 60):  # forced close, then one answer token
        tokens.append(tid)
        p(mx.array(tokens), _logits())
    # Mid-answer reopen: allowed, but closed on the spot (budget-0), so the
    # interleaved format survives without funding more thinking.
    tokens.append(START)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) == NL
    tokens.append(NL)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) == END
    tokens.append(END)
    out = p(mx.array(tokens), _logits())
    assert not p.done and not p.in_thinking
    assert _banned_ids(out) == {EOS}  # fresh close, fresh floor
    tokens.append(61)
    assert _banned_ids(p(mx.array(tokens), _logits())) == set()


def test_open_close_cycling_strikes_out():
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=0, start_seq=(START,),
        start_in_thinking=True, eos_ids=[EOS],
    )
    tokens = [1, 2, START]
    p(mx.array(tokens), _logits())
    for tid in (50, NL, END):  # budget 0: first think token trips the close
        tokens.append(tid)
        p(mx.array(tokens), _logits())
    assert p._spent
    # Three reopen/close cycles with no answer token in between: the model
    # is cycling; the processor stands down instead of fencing with it.
    for _ in range(3):
        for tid in (START, NL, END):
            tokens.append(tid)
            p(mx.array(tokens), _logits())
    assert p.done
    tokens.append(START)  # no further intervention of any kind
    assert _banned_ids(p(mx.array(tokens), _logits())) == set()


def test_first_close_carries_wrap_phrase_recloses_terse():
    W1, W2 = 90, 91  # wrap-phrase tokens
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[W1, W2, END], budget=1,
        start_seq=(START,), reclose_ids=[NL, END],
    )
    tokens = [1, 2, START]
    p(mx.array(tokens), _logits())
    forced = []
    for tid in (50, 51, W1, W2, END):
        tokens.append(tid)
        forced.append(_argmax_if_forced(p(mx.array(tokens), _logits())))
    assert forced == [None, W1, W2, END, None]  # phrase, then the close
    tokens.append(60)  # an answer token lands
    p(mx.array(tokens), _logits())
    # Spent-mode reopen: the terse reclose, not the phrase again.
    tokens.append(START)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) == NL
    tokens.append(NL)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) == END


def test_factory_wrap_phrase_only_on_first_close_and_budget_gt0():
    from gmlx.thinking_budget import _BUDGET_WRAP_PHRASE

    class _PhraseTok(_FakeTok):
        _MAP = dict(_FakeTok._MAP, **{_BUDGET_WRAP_PHRASE: [90, 91]})

    p = make_thinking_budget_processor(_PhraseTok(), 8)
    assert p.forced_ids == [90, 91, END]
    assert p.reclose_ids == [NL, END]
    p0 = make_thinking_budget_processor(_PhraseTok(), 0)
    assert p0.forced_ids == [NL, END]  # nothing worth wrapping up


def test_natural_close_skips_spent_mode():
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=100, start_seq=(START,),
        eos_ids=[EOS],
    )
    forced = _run(p, [50, END, 60, 61, 62])
    assert all(f is None for f in forced)
    assert p.done and not p._spent


def test_factory_wires_eos_ids_and_floor():
    class _EosTok(_FakeTok):
        eos_token_ids = {77, 33}

    p = make_thinking_budget_processor(_EosTok(), 4)
    assert p.eos_ids == [33, 77]
    p0 = make_thinking_budget_processor(_EosTok(), 0, eos_floor=False)
    assert p0.eos_ids == []


# --- multi-token channel delimiters (<|channel>thought / <channel|>) --------

CH_OPEN, THOUGHT, CH_CLOSE = 120, 121, 122


def test_channel_budget_zero_force_closes_on_open():
    # budget=0: as soon as the channel thought opens and one token is generated,
    # the cap force-closes with the (single-token) <channel|> end marker.
    p = ThinkingBudgetProcessor(
        end_seq=(CH_CLOSE,), forced_ids=[NL, CH_CLOSE], budget=0,
        start_seq=(CH_OPEN, THOUGHT), start_in_thinking=False,
    )
    forced = _run(p, [CH_OPEN, THOUGHT, 50, 51, 52], prompt=(1, 2, 3))
    assert NL in forced and CH_CLOSE in forced
    assert forced.index(NL) + 1 == forced.index(CH_CLOSE)
    assert p._spent


def test_channel_natural_close_disarms():
    p = ThinkingBudgetProcessor(
        end_seq=(CH_CLOSE,), forced_ids=[NL, CH_CLOSE], budget=50,
        start_seq=(CH_OPEN, THOUGHT), start_in_thinking=False,
    )
    forced = _run(p, [CH_OPEN, THOUGHT, 50, 51, CH_CLOSE, 60], prompt=(1, 2, 3))
    assert all(f is None for f in forced)
    assert p.done and not p.in_thinking


def test_channel_partial_start_does_not_arm():
    # <|channel> alone (without the following 'thought' token) is not the start
    # of a thinking block, so the cap must not begin counting.
    p = ThinkingBudgetProcessor(
        end_seq=(CH_CLOSE,), forced_ids=[NL, CH_CLOSE], budget=0,
        start_seq=(CH_OPEN, THOUGHT), start_in_thinking=False,
    )
    forced = _run(p, [CH_OPEN, 55, 56, 57], prompt=(1, 2, 3))
    assert all(f is None for f in forced)
    assert not p.in_thinking


def test_channel_forced_close_floor_and_instant_reclose():
    p = ThinkingBudgetProcessor(
        end_seq=(CH_CLOSE,), forced_ids=[NL, CH_CLOSE], budget=0,
        start_seq=(CH_OPEN, THOUGHT), start_in_thinking=True,
        eos_ids=[EOS],
    )
    tokens = [1, 2]
    p(mx.array(tokens), _logits())  # baseline
    tokens.append(50)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) == NL
    tokens.append(NL)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) == CH_CLOSE
    tokens.append(CH_CLOSE)
    out = p(mx.array(tokens), _logits())
    assert p._spent and not p.done
    assert _banned_ids(out) == {EOS}  # first-token floor
    tokens.append(60)
    assert _banned_ids(p(mx.array(tokens), _logits())) == set()
    # Spent-mode instant reclose works on the multi-token open too.
    tokens.extend([CH_OPEN])
    p(mx.array(tokens), _logits())
    tokens.append(THOUGHT)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) == NL


# --- delimiter resolution ---------------------------------------------------


class _WrapperTok:
    """Stands in for mlx-lm's TokenizerWrapper think-token API."""

    def __init__(self, start, end):
        self.think_start_tokens = start
        self.think_end_tokens = end
        self.think_end = "<channel|>"

    def encode(self, text, add_special_tokens=True):
        return {"\n": [NL]}.get(text, [])


def test_thinking_token_seqs_prefers_wrapper_api():
    start, end = _thinking_token_seqs(
        _WrapperTok((CH_OPEN, THOUGHT), (CH_CLOSE,))
    )
    assert start == (CH_OPEN, THOUGHT) and end == (CH_CLOSE,)


def test_thinking_token_seqs_falls_back_to_think_probe():
    start, end = _thinking_token_seqs(_FakeTok())
    assert start == (START,) and end == (END,)


def test_thinking_token_seqs_hy3_suffixed_single_tokens():
    # Hy3: the suffixed tags are single vocab tokens; the bare spellings BPE-
    # split (ending in '>'). The probe must pick the suffixed pair, never the
    # trailing '>' piece of the bare fallback (which matches on every '>').
    GT, S_OPEN, S_CLOSE = 29, 120029, 120030

    class _Hy3Tok:
        _MAP = {
            "<think:opensource>": [S_OPEN],
            "</think:opensource>": [S_CLOSE],
            "<think>": [27, 37330, GT],
            "</think>": [27, 120039, GT],
        }

        def encode(self, text, add_special_tokens=True):
            return self._MAP.get(text, [])

    start, end = _thinking_token_seqs(_Hy3Tok())
    assert start == (S_OPEN,) and end == (S_CLOSE,)


def test_thinking_token_seqs_none_when_undetected():
    class _NoThink:
        def encode(self, text, add_special_tokens=True):
            return []

    assert _thinking_token_seqs(_NoThink()) == (None, None)


def test_factory_builds_channel_force_sequence():
    p = make_thinking_budget_processor(
        _WrapperTok((CH_OPEN, THOUGHT), (CH_CLOSE,)), 0, start_in_thinking=False
    )
    assert p is not None
    assert p.end_seq == (CH_CLOSE,) and p.start_seq == (CH_OPEN, THOUGHT)
    assert p.forced_ids == [NL, CH_CLOSE]


def test_factory_builds_think_force_sequence():
    p = make_thinking_budget_processor(_FakeTok(), 64)
    assert p is not None
    assert p.end_seq == (END,) and p.start_seq == (START,)
    assert p.forced_ids == [NL, END]
    assert p.in_thinking is True


def test_factory_returns_none_without_think_token():
    class _NoThink:
        def encode(self, text, add_special_tokens=True):
            return []

    assert make_thinking_budget_processor(_NoThink(), 64) is None


def test_negative_budget_is_noop():
    assert make_thinking_budget_processor(_FakeTok(), -1) is None
    assert make_thinking_budget_processor(_FakeTok(), None) is None


# --- prompt-open detection --------------------------------------------------

def test_prompt_opens_thinking():
    assert prompt_opens_thinking("...<|assistant|><think>") is True
    assert prompt_opens_thinking("...<|assistant|><think>\n") is True
    assert prompt_opens_thinking("...<think>\n\n</think>\n\n") is False
    assert prompt_opens_thinking("...<|im_start|>assistant\n") is False
    assert prompt_opens_thinking(None) is False
    assert prompt_opens_thinking([1, 2, 3]) is False


def test_prompt_opens_thinking_hy3_suffixed_tags():
    # Hy3 pre-fills '<think:opensource>' at reasoning_effort low/high; the
    # default scan must catch it (the bare '<think>' spelling cannot
    # substring-match the suffixed tag).
    assert prompt_opens_thinking(
        "...<|hy_Assistant:opensource|><think:opensource>") is True
    assert prompt_opens_thinking(
        "...<think:opensource></think:opensource>") is False  # no_think pre-fill
    # a closed suffixed block plus an open bare block still opens
    assert prompt_opens_thinking(
        "...<think:opensource>x</think:opensource>...<think>") is True


def test_prompt_opens_thinking_uses_tokenizer_markers():
    class _Marker:
        think_start = "<|channel>thought"
        think_end = "<channel|>"

    tok = _Marker()
    assert prompt_opens_thinking("x <|channel>thought reasoning", tokenizer=tok) is True
    assert prompt_opens_thinking(
        "x <|channel>thought done <channel|> answer", tokenizer=tok
    ) is False
    # the default <think> spelling must NOT match this model's prompt
    assert prompt_opens_thinking("x <|channel>thought reasoning") is False


def test_prompt_open_think_tag_returns_model_spelling():
    from gmlx.thinking_budget import prompt_open_think_tag

    assert prompt_open_think_tag("...<|assistant|><think>\n") == "<think>"
    assert prompt_open_think_tag(
        "...<|hy_Assistant:opensource|><think:opensource>"
    ) == "<think:opensource>"
    assert prompt_open_think_tag("...<think>x</think>\n") is None
    assert prompt_open_think_tag(None) is None

    class _Marker:
        think_start = "<|channel>thought"
        think_end = "<channel|>"

    assert prompt_open_think_tag(
        "x <|channel>thought reasoning", tokenizer=_Marker()
    ) == "<|channel>thought"
