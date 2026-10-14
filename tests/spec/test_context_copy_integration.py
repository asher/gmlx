"""Context copy on a real MTP pair, one forced copy round at round ``AT``.

``integration``; skips unless ``KQUANT_TEST_MTP_GGUF`` points at a GGUF
with MTP support (``KQUANT_TEST_MTP_DRAFT_GGUF`` adds a companion drafter,
such as a DFlash 2 file). A copy of the drafter's draft length verifies at
the drafter's width, so the greedy output matches a run without copies. A
wider copy verifies on other kernels, whose rounding can flip a near tie
later on, but the drafter must still hold exactly the committed positions
after it.
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


def _generate(pair, monkeypatch, forced, widths, rounds=None):
    from gmlx.cache.kv_cache import make_prompt_cache

    model, drafter, _cfg, tok = pair
    monkeypatch.setenv("GMLX_CONTEXT_COPY", "1")
    monkeypatch.setattr(spec, "PromptCopy", lambda p, **k: forced)
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

        monkeypatch.setattr(drafter, "draft_block", draft_block)
    lm = getattr(model, "language_model", model)
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = mx.array(tok.encode(text), dtype=mx.int32)
    for t in spec.stream_speculative(
            model, drafter, ids, prompt_cache=make_prompt_cache(lm),
            max_tokens=N, sampler=None):
        out.append(int(t))
    mx.clear_cache()
    return out


_REF = {}


def _ref(pair, monkeypatch):
    if "ref" not in _REF:
        _REF["ref"] = _generate(pair, monkeypatch, _Forced(None, 0, False), [])
    return _REF["ref"]


def _block(pair):
    return spec._resolve_block_total(pair[1], None)


@pytest.mark.parametrize("wrong", [True, False])
def test_a_copy_at_the_drafters_width_keeps_the_greedy_output(
        pair, monkeypatch, wrong):
    ref = _ref(pair, monkeypatch)
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
    ref = _ref(pair, monkeypatch)
    forced = _Forced(ref, _block(pair) + 1, wrong)
    widths, rounds = [], []
    _generate(pair, monkeypatch, forced, widths, rounds)
    assert forced.records, "no copy round ran"
    assert max(widths) > _block(pair)
    # offset less committed tokens stays fixed from round to round; the
    # last round can meet the end of the drafter's buffer
    assert len({off - n for n, off in rounds[:-1]}) == 1
