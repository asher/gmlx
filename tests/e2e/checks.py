"""Deterministic floor checks for server e2e responses - pure stdlib, no model.

These are the *floor*: every response must pass them regardless of the LLM judge.
They catch the failure modes that are cheap and unambiguous to detect mechanically
(broken transport, empty output, pathological repetition / looping, mojibake,
NaN/garbage), so the judge only has to rule on genuine semantic coherence.

Kept import-light (``re`` + ``collections`` only) so ``tests/test_e2e_checks.py``
can unit-test the detectors on the CPU without importing the heavy harness.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def __bool__(self) -> bool:           # so `if result:` reads as pass/fail
        return self.ok


# transport / schema
def check_http_ok(status: int, body) -> CheckResult:
    ok = status == 200 and isinstance(body, dict)
    return CheckResult("http_ok", ok,
                       "" if ok else f"status={status} body_type={type(body).__name__}")


def extract_chat_text(body: dict) -> Optional[str]:
    """The assistant message content from a /v1/chat/completions body, or None."""
    try:
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):           # some servers chunk content into parts
        content = "".join(
            p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else None


def check_chat_schema(body: dict) -> CheckResult:
    text = extract_chat_text(body)
    if text is None:
        return CheckResult("chat_schema", False, "no choices[0].message.content string")
    fr = (body.get("choices") or [{}])[0].get("finish_reason")
    if fr is None:
        return CheckResult("chat_schema", False, "missing finish_reason")
    return CheckResult("chat_schema", True, f"finish_reason={fr}")


def check_finish_reason(body: dict,
                        allowed=("stop", "length", "tool_calls", "end_turn")) -> CheckResult:
    fr = (body.get("choices") or [{}])[0].get("finish_reason")
    ok = fr in allowed
    return CheckResult("finish_reason", ok, f"finish_reason={fr!r}")


def check_usage(body: dict) -> CheckResult:
    """Token accounting present and non-degenerate (completion_tokens > 0)."""
    usage = body.get("usage") or {}
    ct = usage.get("completion_tokens")
    pt = usage.get("prompt_tokens")
    ok = isinstance(ct, int) and ct > 0 and isinstance(pt, int) and pt > 0
    return CheckResult("usage", ok, f"prompt={pt} completion={ct}")


# content sanity
def check_nonempty(text: Optional[str], min_chars: int = 1) -> CheckResult:
    n = len((text or "").strip())
    return CheckResult("nonempty", n >= min_chars, f"{n} chars")


_REPLACEMENT = "�"


def check_no_mojibake(text: str) -> CheckResult:
    """No Unicode replacement chars (a decoder/tokenizer corruption signal)."""
    n = (text or "").count(_REPLACEMENT)
    return CheckResult("no_mojibake", n == 0, f"{n} U+FFFD chars")


_NAN_RE = re.compile(r"\b(?:nan|-?inf|infinity)\b", re.IGNORECASE)


def check_no_nan_tokens(text: str) -> CheckResult:
    """Surfaced NaN/inf in text is a strong garbage signal. A couple of incidental
    'inf' substrings (e.g. 'information') won't match - we require the standalone
    word forms - but a flood (>=3) of literal nan/inf is flagged."""
    hits = _NAN_RE.findall(text or "")
    n = len(hits)
    return CheckResult("no_nan", n < 3, f"{n} nan/inf tokens")


# repetition / degeneration - the crux (degeneration usually shows at depth)
_CHAR_LOOP_RE = re.compile(r"(.{1,8}?)\1{9,}", re.DOTALL)   # a <=8-char unit x10+


def _max_consecutive_ngram_run(words, n: int) -> int:
    """Longest run of the SAME n-gram appearing back-to-back (step 1). A clean
    enumeration ('1 2 3 ...') never repeats an n-gram; a loop ('I can not I can
    not ...') repeats one many times in a row."""
    if len(words) < 2 * n:
        return 0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    best = run = 1
    for i in range(1, len(grams)):
        # back-to-back means the gram n positions earlier is identical
        if i >= n and grams[i] == grams[i - n]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def detect_repetition(text: str,
                      *,
                      min_words: int = 24,
                      consec_ngram_max: int = 4,
                      diversity_floor: float = 0.22,
                      top_word_frac_max: float = 0.45) -> CheckResult:
    """Heuristic degeneration detector. Flags (any of):

    * a short character/sub-token unit looping 10+ times (``ababab...``),
    * the same 3- or 4-gram repeating back-to-back > ``consec_ngram_max`` times,
    * very low lexical diversity once the text is long enough
      (``distinct/total < diversity_floor``),
    * a single word dominating (``top_word_frac > top_word_frac_max``).

    Tuned to pass legitimate enumerations/lists (high diversity) and short refrains
    (a chorus repeated 2-3x), and to fail true loops. Returns ok=True when clean.
    """
    t = text or ""
    m = _CHAR_LOOP_RE.search(t)
    if m and len(m.group(0)) >= 16:
        unit = m.group(1)
        return CheckResult("repetition", False,
                           f"char-loop unit={unit!r} x{len(m.group(0)) // max(len(unit),1)}")

    words = t.split()
    if len(words) < min_words:
        return CheckResult("repetition", True, f"short ({len(words)} words)")

    for n in (4, 3):
        run = _max_consecutive_ngram_run(words, n)
        if run > consec_ngram_max:
            return CheckResult("repetition", False,
                               f"{n}-gram repeated back-to-back x{run}")

    distinct = len(set(words)) / len(words)
    if distinct < diversity_floor:
        return CheckResult("repetition", False,
                           f"low diversity {distinct:.2f} over {len(words)} words")

    top, top_n = Counter(words).most_common(1)[0]
    frac = top_n / len(words)
    if frac > top_word_frac_max:
        return CheckResult("repetition", False,
                           f"word {top!r} is {frac:.0%} of output")

    return CheckResult("repetition", True,
                       f"diversity={distinct:.2f} top={frac:.0%}")


# anchored content (used sparingly - the judge is the primary semantic gate)
def check_contains(text: str, substrs, *, mode: str = "any",
                   ci: bool = True, name: str = "contains") -> CheckResult:
    hay = (text or "")
    if ci:
        hay = hay.lower()
    subs = [substrs] if isinstance(substrs, str) else list(substrs)
    norm = [s.lower() if ci else s for s in subs]
    present = [s for s, n in zip(subs, norm) if n in hay]
    ok = (len(present) == len(subs)) if mode == "all" else (len(present) > 0)
    return CheckResult(name, ok, f"matched {present or 'none'} of {subs} ({mode})")


def fraction_uppercase_letters(text: str) -> float:
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


# bundle: the floor every text response must clear
@dataclass
class FloorReport:
    results: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def failures(self) -> list:
        return [r for r in self.results if not r.ok]

    def add(self, r: CheckResult) -> "FloorReport":
        self.results.append(r)
        return self


def floor_text_checks(status: int, body: dict, *, min_chars: int = 1) -> FloorReport:
    """The standard floor for a non-streaming chat completion: transport, schema,
    finish_reason, usage, non-empty, mojibake, NaN, and repetition. The repetition
    + non-empty checks run on the extracted assistant text."""
    rep = FloorReport()
    rep.add(check_http_ok(status, body))
    if not rep.ok:                          # transport broken - nothing else to read
        return rep
    rep.add(check_chat_schema(body))
    rep.add(check_finish_reason(body))
    rep.add(check_usage(body))
    text = extract_chat_text(body) or ""
    rep.add(check_nonempty(text, min_chars=min_chars))
    rep.add(check_no_mojibake(text))
    rep.add(check_no_nan_tokens(text))
    rep.add(detect_repetition(text))
    return rep
