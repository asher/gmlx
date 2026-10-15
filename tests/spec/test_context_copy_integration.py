"""Context copy on a real MTP pair, one forced copy round at round ``AT``.

``integration``; skips unless ``KQUANT_TEST_MTP_GGUF`` points at a GGUF
with MTP support (``KQUANT_TEST_MTP_DRAFT_GGUF`` adds a companion drafter,
such as a DFlash 2 file). A copy of the drafter's draft length verifies at
the drafter's width, so the greedy output matches a run without copies. A
wider copy verifies on other kernels, whose rounding can flip a near tie
later on, but the drafter must still hold exactly the committed positions
after it. The last test runs the real prompt lookup under the identity cap.
"""

import os

import mlx.core as mx
import pytest

import gmlx.spec.speculative as spec

pytestmark = pytest.mark.integration

PROMPT = (
    "I am planning a two week trip through Japan in late autumn with my "
    "partner. We like food markets, quiet temples, hiking and small towns more "
    "than big shopping districts. Could you suggest a day by day itinerary, "
    "with the trains to take between cities, a rough budget per day, and a "
    "few tips on etiquette we should know before we go?")
N = 200
AT = 9
CODE = """
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda iv: iv[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [tuple(iv) for iv in merged]


def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) != 3:
        raise ValueError("bad line: " + line)
    name, start, end = parts
    return name.strip(), int(start), int(end)


def load_schedule(path):
    by_name = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            name, start, end = parse_line(line)
            by_name.setdefault(name, []).append((start, end))
    return {name: merge_intervals(ivs) for name, ivs in by_name.items()}


def busiest(schedule):
    best, best_total = None, -1
    for name, ivs in schedule.items():
        total = sum(end - start for start, end in ivs)
        if total > best_total:
            best, best_total = name, total
    return best, best_total
"""
# A near-verbatim edit: the reply repeats the prompt's code.
EDIT = ("Add the line `# Reviewed.` as the first line of this Python code. "
        "Change nothing else. Reply with the full code only.\n\n" + CODE)


@pytest.fixture(scope="module")
def pair():
    path = os.environ.get("KQUANT_TEST_MTP_GGUF")
    if not path or not os.path.isfile(os.path.expanduser(path)):
        pytest.skip("set KQUANT_TEST_MTP_GGUF to a GGUF with MTP support")
    draft = os.environ.get("KQUANT_TEST_MTP_DRAFT_GGUF")
    draft = os.path.expanduser(draft) if draft else None
    from gmlx.spec.mtp_load import load_mtp_model

    return load_mtp_model(os.path.expanduser(path), draft_gguf_path=draft,
                          verbose=False)


class _Forced:
    """One copy at drafter round ``AT``: the next ``k`` tokens of the
    reference, the first replaced when ``wrong``, clamped to the budget."""

    def __init__(self, ref, k, wrong):
        self.ref, self.k, self.wrong = ref, k, wrong
        self.seen = []
        self.calls = 0
        self.records = []
        self.rounds = self.copied = self.accepted = 0

    def extend(self, tokens):
        self.seen.extend(int(t) for t in tokens)

    def propose(self, shortest, budget):
        self.calls += 1
        if self.ref is None or self.calls != AT + 1:
            return []
        out = list(self.ref[len(self.seen):len(self.seen) + min(self.k, budget)])
        if self.wrong and out:
            out[0] = (out[0] + 1) % 1000 + 1000
        return out if len(out) >= max(shortest, 1) else []

    def drafted(self, tokens, seconds):
        pass

    def record(self, copied, accepted, seconds=None):
        self.records.append((copied, accepted))


def _generate(pair, monkeypatch, forced, widths, rounds=None,
              prompt=PROMPT, made=None):
    from gmlx.cache.kv_cache import make_prompt_cache

    model, drafter, _cfg, tok = pair
    monkeypatch.setenv("GMLX_CONTEXT_COPY", "1")
    if forced is not None:
        monkeypatch.setattr(spec, "PromptCopy", lambda p, **k: forced)
    else:
        from gmlx.spec.context_copy import PromptCopy as real

        def build(p, **k):
            made.append(real(p, **k))
            return made[-1]

        monkeypatch.setattr(spec, "PromptCopy", build)
    orig = spec._mtp_verify_target

    def verify(lm_, ids, *a, **k):
        widths.append(int(ids.shape[1]))
        return orig(lm_, ids, *a, **k)

    monkeypatch.setattr(spec, "_mtp_verify_target", verify)
    out = []
    if rounds is not None:
        orig_draft = drafter.draft_block

        def draft_block(*a, **k):
            rounds.append((len(out), int(drafter._cache[0].offset)))
            return orig_draft(*a, **k)

        drafter.draft_block = draft_block
    lm = getattr(model, "language_model", model)
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = mx.array(tok.encode(text), dtype=mx.int32)
    try:
        for t in spec.stream_speculative(
                model, drafter, ids, prompt_cache=make_prompt_cache(lm),
                max_tokens=N, sampler=None):
            out.append(int(t))
    finally:
        if rounds is not None:
            del drafter.draft_block     # the class method again
    mx.clear_cache()
    return out


_REF = {}


def _ref(pair, monkeypatch, prompt=PROMPT):
    """The copy-off output and the drafter's (committed, offset) per round."""
    if prompt not in _REF:
        rounds = []
        out = _generate(pair, monkeypatch, _Forced(None, 0, False), [],
                        rounds, prompt=prompt)
        _REF[prompt] = out, rounds
    return _REF[prompt]


def _offset_less_committed(rounds):
    # The last round can meet the end of the drafter's buffer.
    return {off - n for n, off in rounds[:-1]}


def _block(pair):
    return spec._resolve_block_total(pair[1], None)


@pytest.mark.parametrize("wrong", [True, False])
def test_a_copy_at_the_drafters_width_keeps_the_greedy_output(
        pair, monkeypatch, wrong):
    ref, _ = _ref(pair, monkeypatch)
    block = _block(pair)
    forced = _Forced(ref, block - 1, wrong)
    widths = []
    out = _generate(pair, monkeypatch, forced, widths)
    assert forced.records, "no copy round ran"
    assert max(widths) <= block
    assert out == ref


@pytest.mark.parametrize("wrong", [True, False])
def test_the_drafter_holds_the_committed_positions_after_a_wide_copy(
        pair, monkeypatch, wrong):
    ref, ref_rounds = _ref(pair, monkeypatch)
    forced = _Forced(ref, _block(pair) + 1, wrong)
    widths, rounds = [], []
    _generate(pair, monkeypatch, forced, widths, rounds)
    assert forced.records, "no copy round ran"
    assert max(widths) > _block(pair)
    want = _offset_less_committed(ref_rounds)
    assert len(want) == 1 and _offset_less_committed(rounds) == want


def test_the_identity_cap_keeps_the_output_end_to_end(pair, monkeypatch):
    """GMLX_CONTEXT_COPY_MAX at the drafter's draft length, with the real
    prompt lookup and copy lengths: copies run, and the output and the
    drafter's positions match a run without copies."""
    from gmlx.spec.context_copy import SHORTEST_USEFUL

    block = _block(pair)
    if block - 1 < SHORTEST_USEFUL:
        pytest.skip("this drafter's draft is too short to copy at its width")
    ref, ref_rounds = _ref(pair, monkeypatch, EDIT)
    monkeypatch.setenv("GMLX_CONTEXT_COPY_MAX", str(block - 1))
    # Drafter rounds feed no yield: a strong drafter on this text lifts the
    # floor of twice its yield past the cap, which would stop every copy.
    from gmlx.spec.context_copy import PromptCopy

    monkeypatch.setattr(PromptCopy, "drafted", lambda self, t, s: None)
    made, widths, rounds = [], [], []
    out = _generate(pair, monkeypatch, None, widths, rounds, prompt=EDIT,
                    made=made)
    assert made and made[0].rounds, "no copy round ran"
    assert max(widths) <= block
    assert out == ref
    assert _offset_less_committed(rounds) == _offset_less_committed(ref_rounds)
