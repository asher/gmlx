"""Unit tests for the thinking-budget logits processor (CPU, no model load).

Drives the processor's state machine with synthetic ``(tokens, logits)`` calls
that mimic mlx-lm's contract: ``tokens`` is the full accumulated sequence
(prompt + generated), growing by one per step after the first; ``logits`` is
``(1, vocab)``. Covers single-token ``<think>`` delimiters and the multi-token
``<|channel>thought`` / ``<channel|>`` channel format.
"""

import mlx.core as mx
import pytest

from gmlx.gen.thinking_budget import (
    MTPFinishThinking,
    ThinkingBudgetProcessor,
    _thinking_token_seqs,
    make_mtp_finish_hook,
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
    from gmlx.gen.thinking_budget import _BUDGET_WRAP_PHRASE

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


# --- ^T finish-thinking (request_close / uncapped interruptible mode) --------

SKIP = 7  # stand-in for the ^T wrap phrase's first token


def test_request_close_forces_skip_sequence_then_spent_mode():
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[SKIP, END], budget=None, start_seq=(START,),
        skip_ids=[NL, END], reclose_ids=[NL, END],
    )
    tokens = [1, 2, START]
    p(mx.array(tokens), _logits())
    for tid in range(30, 80):  # no budget: thinking runs free until ^T
        tokens.append(tid)
        assert _argmax_if_forced(p(mx.array(tokens), _logits())) is None
    p.request_close()
    tokens.append(80)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) == NL  # skip_ids,
    tokens.append(NL)                                               # not forced_ids
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) == END
    tokens.append(END)
    p(mx.array(tokens), _logits())
    assert p._spent and not p.in_thinking
    tokens.append(60)  # a real answer token lands
    p(mx.array(tokens), _logits())
    # A reopen after ^T is closed on the spot, budget-0 style.
    tokens.append(START)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) == NL


def test_request_close_consumed_when_not_thinking():
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=None, start_seq=(START,),
        start_in_thinking=False,
    )
    tokens = [1, 2, 3]
    p(mx.array(tokens), _logits())
    p.request_close()
    tokens.append(50)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) is None
    # The stale request must not close a block opened later.
    tokens.append(START)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) is None
    tokens.append(51)
    assert _argmax_if_forced(p(mx.array(tokens), _logits())) is None
    assert p.in_thinking


def test_factory_interruptible_without_budget():
    assert make_thinking_budget_processor(_FakeTok(), None) is None
    p = make_thinking_budget_processor(_FakeTok(), None, interruptible=True)
    assert p is not None and p.budget is None
    assert p.skip_ids[-1] == END


def test_finish_key_target_round_trip():
    from gmlx.gen.thinking_budget import (
        clear_finish_key_target,
        finish_thinking_now,
        set_finish_key_target,
    )

    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=None, start_seq=(START,)
    )
    assert finish_thinking_now() is False       # nothing armed
    set_finish_key_target(p)
    try:
        assert finish_thinking_now() is True
        assert p._close_requested
    finally:
        clear_finish_key_target()
    assert finish_thinking_now() is False


def test_fallback_rejects_shared_trailing_piece():
    # A vocab with no think tokens at all: both tag spellings BPE down to a
    # shared trailing '>' piece. Arming on that would trip on every '>'.
    class _NoThinkTok:
        def encode(self, text, add_special_tokens=True):
            return [7, 8, 42] if "think" in text else []

    assert _thinking_token_seqs(_NoThinkTok()) == (None, None)
    assert make_thinking_budget_processor(
        _NoThinkTok(), None, interruptible=True) is None


def test_factory_tolerates_int_eos_attr():
    # Raw GGUF-built tokenizers carry eos_token_ids as a bare int.
    class _IntEosTok(_FakeTok):
        eos_token_ids = 77

    p = make_thinking_budget_processor(_IntEosTok(), 4)
    assert p.eos_ids == [77]


def test_finish_key_unsupported_notes_once(capsys):
    from gmlx.gen.thinking_budget import (
        FinishKeyUnsupported,
        clear_finish_key_target,
        finish_thinking_now,
        set_finish_key_target,
    )

    set_finish_key_target(FinishKeyUnsupported("on the MTP path"))
    try:
        assert finish_thinking_now() is False
        assert finish_thinking_now() is False
    finally:
        clear_finish_key_target()
    err = capsys.readouterr().err
    assert err.count("finish-thinking isn't available on the MTP path") == 1


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


# --- configured markers (family card / config sampling keys) ----------------

CFG_OPEN, CFG_CLOSE = 77, 78


class _CfgTok:
    """Carries both the default spelling and a custom one as single tokens,
    so a configured pair has to beat the probe rather than be its only hit."""

    _MAP = {"\n": [NL], "</think>": [END], "<think>": [START],
            "<t>": [CFG_OPEN], "</t>": [CFG_CLOSE]}

    def encode(self, text, add_special_tokens=True):
        return self._MAP.get(text, [])


def test_configured_markers_win_over_the_probe():
    start, end = _thinking_token_seqs(
        _CfgTok(), start_token="<t>", end_token="</t>")
    assert start == (CFG_OPEN,) and end == (CFG_CLOSE,)
    p = make_thinking_budget_processor(
        _CfgTok(), 64, start_token="<t>", end_token="</t>")
    assert p.end_seq == (CFG_CLOSE,) and p.forced_ids == [NL, CFG_CLOSE]


def test_unencodable_configured_end_falls_back_to_the_probe():
    """A marker the vocab has no tokens for must not disarm the cap."""
    start, end = _thinking_token_seqs(
        _CfgTok(), start_token="<nope>", end_token="</nope>")
    assert start == (START,) and end == (END,)


def test_configured_markers_drive_prompt_open_detection():
    assert prompt_opens_thinking("...<|assistant|><t>", "<t>", "</t>") is True
    assert prompt_opens_thinking("...<t>x</t>\n", "<t>", "</t>") is False
    # the default probe does not know the spelling, so it reads as closed
    assert prompt_opens_thinking("...<|assistant|><t>") is False


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


def test_prompt_opens_thinking_kimi_k3_xtml():
    # K3's generation prompt ends '<|open|>message role="assistant"<|sep|>
    # <|open|>think<|sep|>'; the default scan must catch the XTML pair.
    assert prompt_opens_thinking(
        '...<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'
    ) is True
    # thinking=false pre-opens the response section instead.
    assert prompt_opens_thinking(
        '...<|open|>message role="assistant"<|sep|><|open|>response<|sep|>'
    ) is False
    # a closed think section in history does not count as open
    assert prompt_opens_thinking(
        "...<|open|>think<|sep|>x<|close|>think<|sep|>...") is False


def test_template_think_pair_kimi_k3_macro_concat():
    # The K3 jinja builds markers via open_tag('think') macro concatenation -
    # the literal '<|open|>think<|sep|>' never appears in the template source.
    from gmlx.gen.thinking_budget import _template_think_pair

    class _Tok:
        chat_template = (
            "{%- macro open_tag(tag, attrs=[]) -%}{{- '<|open|>' + tag -}}"
            "{{- '<|sep|>' -}}{%- endmacro -%}"
            "{{- open_tag('think' if thinking else 'response') -}}"
        )

    assert _template_think_pair(_Tok()) == (
        "<|open|>think<|sep|>", "<|close|>think<|sep|>")

    class _Plain:
        chat_template = "{{ '<think>' }}"

    assert _template_think_pair(_Plain()) == ("<think>", "</think>")


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
    from gmlx.gen.thinking_budget import prompt_open_think_tag

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


class _MuseTok:
    """Just the surface ``_template_think_pair`` reads."""
    chat_template = (
        "{%- if reasoning_strength %}Reasoning strength: {{ reasoning_strength }}"
        "{%- endif %}<|start|>assistant to=self<|message|>...<|eom|>"
    )


def test_muse_glimmer_think_pair_resolves_from_the_template():
    from gmlx.gen.thinking_budget import _template_think_pair

    assert _template_think_pair(_MuseTok()) == (
        "<|start|>assistant to=self<|message|>", "<|eom|>")


def test_muse_glimmer_generation_prompt_is_not_open_thinking():
    """The generation prompt stops at ``<|start|>assistant``; the reasoning
    header is only completed by what the model generates, so the budget must
    not start counting yet."""
    from gmlx.gen.thinking_budget import prompt_opens_thinking

    prompt = "<|start|>user<|message|>hi<|eot|><|start|>assistant"
    assert not prompt_opens_thinking(prompt, tokenizer=_MuseTok())


def test_muse_glimmer_open_reasoning_message_is_detected():
    from gmlx.gen.thinking_budget import prompt_opens_thinking

    prompt = "<|start|>assistant to=self<|message|>half a thought"
    assert prompt_opens_thinking(prompt, tokenizer=_MuseTok())


def test_muse_glimmer_closed_reasoning_message_is_not_open():
    from gmlx.gen.thinking_budget import prompt_opens_thinking

    prompt = "<|start|>assistant to=self<|message|>done<|eom|>"
    assert not prompt_opens_thinking(prompt, tokenizer=_MuseTok())


def test_muse_spark_think_pair_resolves_from_its_template():
    from pathlib import Path

    from gmlx.gen.thinking_budget import _template_think_pair

    class _SparkTok:
        chat_template = (Path(__file__).parents[1] / "fixtures"
                         / "muse_spark_template.jinja").read_text()

    assert _template_think_pair(_SparkTok()) == (
        "<|start|>assistant to=self<|message|>", "<|eom|>")


def test_commentary_close_before_reasoning_keeps_the_budget_armed():
    """Muse Spark can send a to=commentary message first. Its <|eom|> is the
    reasoning end marker too, but it closes no reasoning block, so the
    reasoning that follows is still capped."""
    p = ThinkingBudgetProcessor(
        end_seq=(END,), forced_ids=[NL, END], budget=2, start_seq=(START,),
        start_in_thinking=False,
    )
    forced = _run(p, [50, 51, END, START, 60, 61, 62, 63], prompt=(1, 2, 3))
    assert not p.done
    assert NL in forced and END in forced
    assert p._spent


def test_mtp_hook_commentary_close_before_reasoning_keeps_it_armed():
    h = _mtp_hook(start_in_thinking=False, budget=1)
    for t in (50, END, START, 60, 61):
        h.observe(t)
    assert not h.done and h.in_thinking
    assert h.take_forced() is not None


# --- MTPFinishThinking (^T on the owned MTP rounds) --------------------------

def _mtp_hook(**kw):
    kw.setdefault("end_seq", (END,))
    kw.setdefault("skip_ids", [7, 8, END])
    kw.setdefault("reclose_ids", [NL, END])
    kw.setdefault("start_seq", (START,))
    return MTPFinishThinking(**kw)


def test_mtp_hook_close_while_thinking():
    h = _mtp_hook(start_in_thinking=True)
    for t in (50, 51):
        h.observe(t)
    assert h.take_forced() is None  # nothing requested
    h.request_close()
    assert h.take_forced() == [7, 8, END]  # skip phrase on the first close
    for t in (7, 8, END):  # the round loop emits the forced ids
        h.observe(t)
    assert not h.in_thinking and h._spent


def test_mtp_hook_close_outside_thinking_is_noop():
    h = _mtp_hook(start_in_thinking=False)
    h.observe(50)
    h.request_close()
    assert h.take_forced() is None
    # The stale request must not close a block opened later.
    h.observe(START)
    assert h.in_thinking
    assert h.take_forced() is None


def test_mtp_hook_natural_close_then_request_is_noop():
    h = _mtp_hook(start_in_thinking=True)
    h.observe(50)
    h.observe(END)  # the model closed on its own
    h.request_close()
    assert h.take_forced() is None and not h._spent


def test_mtp_hook_reopen_recloses_tersely():
    h = _mtp_hook(start_in_thinking=True)
    h.request_close()
    assert h.take_forced() == [7, 8, END]
    for t in (7, 8, END):
        h.observe(t)
    h.observe(60)  # an answer token lands
    h.observe(START)  # the model reopens thinking
    assert h.take_forced() == [NL, END]  # terse reclose, no phrase
    for t in (NL, END):
        h.observe(t)
    assert not h.in_thinking


def test_mtp_hook_three_answerless_closes_stand_down():
    h = _mtp_hook(start_in_thinking=True)
    h.request_close()
    assert h.take_forced() == [7, 8, END]
    for t in (7, 8, END):
        h.observe(t)
    closes = 0
    for _ in range(4):  # open/close cycling with no answer between
        h.observe(START)
        forced = h.take_forced()
        if forced is None:
            break
        closes += 1
        for t in forced:
            h.observe(t)
    assert closes == 2 and h.done
    h.request_close()
    assert h.take_forced() is None


def test_mtp_hook_multi_token_end_seq_spans_rounds():
    h = MTPFinishThinking(
        end_seq=(98, END), skip_ids=[7, 98, END], reclose_ids=[98, END],
        start_seq=(START,), start_in_thinking=True)
    h.observe(50)
    h.observe(98)  # first half of the close marker (round boundary here)
    assert h.in_thinking
    h.observe(END)  # completed by the next round
    assert not h.in_thinking


def test_finish_key_targets_mtp_hook():
    from gmlx.gen.thinking_budget import (
        clear_finish_key_target,
        finish_thinking_now,
        set_finish_key_target,
    )

    h = _mtp_hook(start_in_thinking=True)
    set_finish_key_target(h)
    try:
        assert finish_thinking_now() is True
        assert h._close_requested
        h.done = True
        assert finish_thinking_now() is False
    finally:
        clear_finish_key_target()


def test_make_mtp_finish_hook_factory():
    h = make_mtp_finish_hook(_FakeTok(), start_in_thinking=True)
    assert h is not None and h.in_thinking
    assert h.end_seq == (END,)
    assert h.skip_ids[-1] == END and h.reclose_ids == [NL, END]

    class _NoThinkTok:
        def encode(self, text, add_special_tokens=True):
            return [7, 8, 42] if "think" in text else []

    assert make_mtp_finish_hook(_NoThinkTok()) is None


def test_mtp_hook_budget_trips_on_its_own():
    h = _mtp_hook(start_in_thinking=True, budget=3, forced_ids=[8, 9, END])
    for t in (50, 51, 52):
        h.observe(t)
    assert h.take_forced() is None  # count == budget: not over yet
    h.observe(53)
    assert h.take_forced() == [8, 9, END]  # budget phrase, not the ^T one
    for t in (8, 9, END):
        h.observe(t)
    assert not h.in_thinking and h._spent


def test_mtp_hook_natural_close_disarms_budget():
    h = _mtp_hook(start_in_thinking=True, budget=2)
    h.observe(50)
    h.observe(END)
    assert h.done
    h.observe(START)  # reopened blocks are free after a natural close
    for t in (60, 61, 62, 63):
        h.observe(t)
    assert h.take_forced() is None


def test_mtp_hook_reopen_resets_count():
    h = _mtp_hook(start_in_thinking=False, budget=3)
    h.observe(START)
    for t in (50, 51, 52):
        h.observe(t)
    assert h.take_forced() is None  # at budget, not over
    h.observe(53)
    assert h.take_forced() is not None  # over


def test_make_mtp_finish_hook_budget_variants():
    class _WrapTok(_FakeTok):
        def encode(self, text, add_special_tokens=True):
            ids = super().encode(text, add_special_tokens)
            if ids:
                return ids
            if "thinking budget" in text:
                return [70]
            if "asked to wrap" in text:
                return [71]
            return []

    h = make_mtp_finish_hook(_WrapTok(), budget=5, start_in_thinking=True)
    assert h.budget == 5 and h.in_thinking
    assert h.forced_ids == [70, END]  # budget wrap phrase
    assert h.skip_ids == [71, END]  # ^T wrap phrase
    assert h.reclose_ids == [NL, END]
    h0 = make_mtp_finish_hook(_WrapTok(), budget=0)
    assert h0.forced_ids == [NL, END] and h0.skip_ids == [NL, END]
    assert make_mtp_finish_hook(_WrapTok(), budget=-1) is None


# --- ATEM (Muse Spark, Muse Glimmer) with the real token ids -----------------
#
# The reasoning message opens with five tokens, <|start|> assistant " to"
# "=self" <|message|>, and closes with one, <|eom|>. A Spark commentary message
# ends with the same <|eom|>. The generation prompt stops after "assistant",
# and mlx-lm hands the processor only that last prompt token.

import atem_tokens as at  # noqa: E402

from gmlx.gen.thinking_budget import BUDGET_WRAP_PHRASE  # noqa: E402

_ATEM_PROMPT = "<|start|>user<|message|>hi<|eot|><|start|>assistant"
_REASON = " to=self<|message|>Thinking hard about it now.<|eom|>"
_COMMENTARY = ("<|start|>assistant to=commentary<|message|>"
               "Working on it.<|eom|>")
_ANSWER = "<|start|>assistant to=user<|message|>Hello world foo bar<|eot|>"
_ATEM_VOCAB = 202048


def _one_hot_id(out):
    """The forced id when ``out`` is a one-hot row, else None. An EOS ban
    row also holds very negative entries, so count the open entries."""
    row = out[0]
    n_open = int(mx.sum(row > -1e8).item())
    return int(mx.argmax(row).item()) if n_open == 1 else None


def _atem_run(p, script, after, *, close_at=None, tok=None):
    """Replay a decode the way mlx-lm drives a processor. The model writes
    ``script`` until the processor forces a close, then writes ``after``.
    ``close_at`` presses ^T after that many generated tokens. Returns the
    emitted text."""
    tok = tok or at.AtemTokenizer()
    tokens = [tok.encode(_ATEM_PROMPT)[-1]]
    src = iter(tok.encode(script))
    emitted = []
    logits = mx.zeros((1, _ATEM_VOCAB))
    out = p(mx.array(tokens), logits)
    while len(emitted) < 300:
        if close_at is not None and len(emitted) == close_at:
            p.request_close()
        forced = _one_hot_id(out)
        if forced is not None:
            if src is not None:
                src = None
                rest = iter(tok.encode(after))
            tid = forced
        else:
            tid = next(src if src is not None else rest, None)
            if tid is None:
                break
        emitted.append(tid)
        tokens.append(tid)
        out = p(mx.array(tokens), logits)
    return tok.decode(emitted)


def test_the_fixture_tokenizer_matches_the_real_one():
    tok = at.AtemTokenizer()
    for text, real in at.TABLE["encodings"].items():
        assert tok.encode(text) == real, text


@pytest.mark.parametrize("template", ["spark", "glimmer"])
def test_atem_markers_resolve_to_their_real_ids(template):
    start, end = _thinking_token_seqs(at.AtemTokenizer(template))
    assert start == (200022, 140680, 328, 19669, 200023)
    assert end == (200007,)


@pytest.mark.parametrize("template", ["spark", "glimmer"])
def test_atem_budget_caps_reasoning_only(template):
    tok = at.AtemTokenizer(template)
    p = make_thinking_budget_processor(
        tok, 3, start_in_thinking=prompt_opens_thinking(
            _ATEM_PROMPT, tokenizer=tok))
    rest = _COMMENTARY + _ANSWER
    text = _atem_run(p, _REASON + rest, rest, tok=tok)
    # Three reasoning tokens pass, the fourth trips the cap, and the close
    # is the wrap phrase and the whole end marker. Commentary and answer
    # then run untouched.
    assert text == (" to=self<|message|>Thinking hard about"
                    + BUDGET_WRAP_PHRASE + "<|eom|>" + rest)
    assert p._spent and not p.done


def test_atem_budget_is_spent_on_reasoning_after_commentary():
    """A commentary message first: its <|eom|> closes nothing, so the
    reasoning after it is still capped."""
    p = make_thinking_budget_processor(at.AtemTokenizer(), 3,
                                       start_in_thinking=False)
    head = " to=commentary<|message|>Hello world<|eom|><|start|>assistant"
    text = _atem_run(p, head + _REASON + _ANSWER, _ANSWER)
    assert text == (head + " to=self<|message|>Thinking hard about"
                    + BUDGET_WRAP_PHRASE + "<|eom|>" + _ANSWER)


def test_atem_reasoning_under_budget_closes_naturally():
    p = make_thinking_budget_processor(at.AtemTokenizer(), 100,
                                       start_in_thinking=False)
    script = _REASON + _COMMENTARY + _ANSWER
    assert _atem_run(p, script, "") == script
    assert p.done and not p._spent


def test_atem_finish_key_closes_the_reasoning():
    from gmlx.gen.thinking_budget import _SKIP_WRAP_PHRASE

    p = make_thinking_budget_processor(at.AtemTokenizer(), None,
                                       start_in_thinking=False,
                                       interruptible=True)
    rest = _COMMENTARY + _ANSWER
    # ^T after five tokens takes effect on the next step's logits.
    text = _atem_run(p, _REASON + rest, rest, close_at=5)
    assert text == (" to=self<|message|>Thinking hard" + _SKIP_WRAP_PHRASE
                    + "<|eom|>" + rest)


def test_atem_mtp_hook_counts_reasoning_only():
    tok = at.AtemTokenizer()
    h = make_mtp_finish_hook(tok, budget=3,
                             prompt_ids=tok.encode(_ATEM_PROMPT))
    for t in tok.encode(" to=self<|message|>Thinking hard"):
        h.observe(t)
    assert h.in_thinking and h.count == 3
    assert h.take_forced() is None
    h.observe(tok.encode(" about")[0])
    forced = h.take_forced()
    assert forced is not None and forced[-1] == 200007
    for t in forced:
        h.observe(t)
    for t in tok.encode(_COMMENTARY + _ANSWER):
        h.observe(t)
        assert h.take_forced() is None
    assert not h.in_thinking and not h.done


def test_atem_mtp_hook_commentary_close_keeps_it_armed():
    tok = at.AtemTokenizer()
    h = make_mtp_finish_hook(tok, budget=2,
                             prompt_ids=tok.encode(_ATEM_PROMPT))
    for t in tok.encode(" to=commentary<|message|>Hello world<|eom|>"
                        "<|start|>assistant to=self<|message|>Thinking hard"
                        " about"):
        h.observe(t)
    assert not h.done and h.in_thinking
    assert h.take_forced() is not None


# --- K2 Horizon: one think pair per reasoning effort ------------------------

_K2_EFFORTS = ("high", "medium", "low")


@pytest.mark.parametrize("effort", _K2_EFFORTS)
def test_k2_prompt_picks_its_effort_pair(effort):
    import k2_think_tokens as k2

    from gmlx.gen.thinking_budget import prompt_open_think_tag

    tok = k2.K2Tokenizer()
    prompt = k2.generation_prompt(effort)
    start, end = k2.PAIRS[effort]
    assert prompt_open_think_tag(prompt, tokenizer=tok) == start
    want = ((k2.PIECES[start],), (k2.PIECES[end],))
    assert _thinking_token_seqs(tok, prompt=prompt) == want
    assert _thinking_token_seqs(tok, prompt=tok.encode(prompt)) == want


def test_k2_without_a_prompt_takes_the_first_pair():
    import k2_think_tokens as k2

    _, end = _thinking_token_seqs(k2.K2Tokenizer())
    assert end == (k2.PIECES["</ifm|think>"],)


def _forced_run(processor, prompt_ids, steps, close_at=None):
    """Decode filler ids; collect every forced id. ``close_at`` requests ^T
    before that step."""
    tokens, forced = list(prompt_ids), []
    for i in range(steps):
        if i == close_at:
            processor.request_close()
        f = _one_hot_id(processor(mx.array(tokens), mx.zeros((1, 260000))))
        forced.append(f)
        tokens.append(f if f is not None else 1000 + ord("x"))
    return [f for f in forced if f is not None]


@pytest.mark.parametrize("effort", _K2_EFFORTS)
def test_k2_budget_close_uses_its_effort_tag(effort):
    import k2_think_tokens as k2

    tok = k2.K2Tokenizer()
    prompt = k2.generation_prompt(effort)
    p = make_thinking_budget_processor(
        tok, 2, start_in_thinking=prompt_opens_thinking(prompt, tokenizer=tok),
        prompt=prompt)
    forced = _forced_run(p, tok.encode(prompt), 200)
    assert forced[-1] == k2.PIECES[k2.PAIRS[effort][1]]


@pytest.mark.parametrize("effort", _K2_EFFORTS)
def test_k2_finish_key_closes_with_its_effort_tag(effort):
    import k2_think_tokens as k2

    tok = k2.K2Tokenizer()
    prompt = k2.generation_prompt(effort)
    p = make_thinking_budget_processor(
        tok, None, start_in_thinking=True, interruptible=True, prompt=prompt)
    forced = _forced_run(p, tok.encode(prompt), 200, close_at=3)
    assert forced[-1] == k2.PIECES[k2.PAIRS[effort][1]]


@pytest.mark.parametrize("effort", _K2_EFFORTS)
def test_k2_mtp_finish_key_closes_with_its_effort_tag(effort):
    import k2_think_tokens as k2

    tok = k2.K2Tokenizer()
    prompt = k2.generation_prompt(effort)
    close = k2.PIECES[k2.PAIRS[effort][1]]
    for kw in ({"prompt_ids": tok.encode(prompt)}, {"prompt": prompt}):
        hook = make_mtp_finish_hook(tok, start_in_thinking=True, **kw)
        hook.request_close()
        assert hook.take_forced()[-1] == close


# The 7B closes medium and low blocks with </ifm|think>, not the pair's own
# close tag.
@pytest.mark.parametrize("effort", ["medium", "low"])
def test_k2_alias_close_disarms_the_budget(effort):
    import k2_think_tokens as k2

    tok = k2.K2Tokenizer()
    prompt = k2.generation_prompt(effort)
    p = make_thinking_budget_processor(
        tok, 4, start_in_thinking=True, prompt=prompt)
    filler = 1000 + ord("x")
    tokens = tok.encode(prompt)
    script = [filler, filler, k2.PIECES["</ifm|think>"]] + [filler] * 20
    for t in script:
        assert _one_hot_id(p(mx.array(tokens), mx.zeros((1, 260000)))) is None
        tokens.append(t)
    assert p.done and not p.in_thinking


@pytest.mark.parametrize("effort", ["medium", "low"])
def test_k2_alias_close_disarms_the_mtp_hook(effort):
    import k2_think_tokens as k2

    tok = k2.K2Tokenizer()
    prompt = k2.generation_prompt(effort)
    hook = make_mtp_finish_hook(tok, budget=4, start_in_thinking=True,
                                prompt=prompt)
    for t in (1000, 1001, k2.PIECES["</ifm|think>"]):
        hook.observe(t)
    assert hook.done and not hook.in_thinking
    hook.request_close()
    assert hook.take_forced() is None


def test_k2_high_close_has_no_alias():
    import k2_think_tokens as k2

    tok = k2.K2Tokenizer()
    p = make_thinking_budget_processor(
        tok, 4, start_in_thinking=True, prompt=k2.generation_prompt("high"))
    assert p.alt_end_seqs == ()
