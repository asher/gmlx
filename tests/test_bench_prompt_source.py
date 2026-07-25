"""_ChatPromptSource slice determinism: the k-th prompt at a depth is a
pure function of (seed, depth, k), independent of which other depths the
process drew first (single-depth bisect cells must be content-identical to
the same depth inside a multi-depth sweep)."""

from gmlx.benchmarks import _ChatPromptSource


class _StubTok:
    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=True):
        out = []
        for m in msgs:
            c = m["content"]
            s = sum(map(ord, c)) % 997
            out.extend([(s + j) % 997 for j in range(max(1, len(c) // 4))])
        return out

    def encode(self, text):
        return [len(w) % 997 for w in text.split()]


def _convs(n=64):
    return [
        [
            {"role": "user", "content": f"question {i} " * 20},
            {"role": "assistant", "content": f"answer {i} " * 20},
        ]
        for i in range(n)
    ]


def test_depth_slice_independent_of_call_order():
    a = _ChatPromptSource(_convs(), _StubTok(), seed=42)
    b = _ChatPromptSource(_convs(), _StubTok(), seed=42)
    # a: sweep order; b: single-depth cells in reverse order
    a4, a16 = a.get(400), a.get(1600)
    b16, b4 = b.get(1600), b.get(400)
    assert a4 == b4
    assert a16 == b16


def test_repeat_calls_vary_but_reproducibly():
    a = _ChatPromptSource(_convs(), _StubTok(), seed=42)
    b = _ChatPromptSource(_convs(), _StubTok(), seed=42)
    a0, a1 = a.get(400), a.get(400)
    b0, b1 = b.get(400), b.get(400)
    assert a0 == b0 and a1 == b1
    assert a0 != a1  # runs > 1 still sees fresh content


def test_seed_changes_slice():
    a = _ChatPromptSource(_convs(), _StubTok(), seed=42)
    c = _ChatPromptSource(_convs(), _StubTok(), seed=7)
    assert a.get(400) != c.get(400)


def test_template_kwargs_reach_every_render():
    class _KwTok(_StubTok):
        def __init__(self):
            self.kws = []

        def apply_chat_template(self, msgs, add_generation_prompt=True,
                                tokenize=True, **kw):
            self.kws.append(kw)
            return super().apply_chat_template(
                msgs, add_generation_prompt, tokenize)

    tok = _KwTok()
    src = _ChatPromptSource(
        _convs(), tok, seed=42,
        template_kwargs={"reasoning_effort": "low"})
    src.get(400)
    assert tok.kws and all(
        kw == {"reasoning_effort": "low"} for kw in tok.kws)


class _StrictTok(_StubTok):
    """A Mistral/Llama-2-style template: roles must alternate, user first."""

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=True):
        if msgs and msgs[0]["role"] != "user":
            raise ValueError("Conversation roles must alternate user/assistant/...")
        for a, b in zip(msgs, msgs[1:]):
            if a["role"] == b["role"]:
                raise ValueError("Conversation roles must alternate")
        return super().apply_chat_template(msgs, add_generation_prompt, tokenize)


def test_trim_keeps_the_prompt_user_first_for_strict_templates():
    """The trim loop popped one message at a time, so an over-long prompt could
    be handed to apply_chat_template starting with an assistant turn - which
    Mistral/Llama-2 templates reject, crashing the bench run."""
    src = _ChatPromptSource(_convs(), _StrictTok(), seed=42)
    for target in (120, 200, 400, 1000):
        assert src.get(target)                # raised ValueError before the fix


def test_loaded_conversations_end_on_assistant(monkeypatch):
    """Conversations are concatenated to reach a depth, so one ending on a user
    turn would meet the next one's opening user turn - two user turns in a row,
    which alternation-enforcing templates reject."""
    import sys
    import types

    from gmlx import benchmarks

    rows = [
        {"messages": [{"role": "user", "content": "q1"},
                      {"role": "assistant", "content": "a1"},
                      {"role": "user", "content": "dangling"}]},
        {"messages": [{"role": "system", "content": "sys"},
                      {"role": "user", "content": "q2"},
                      {"role": "assistant", "content": "a2"}]},
    ] * 2

    class _DS(list):
        def select(self, idx):
            return self

    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda dataset_id, split: _DS(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake)

    convs = benchmarks._load_chat_dataset("stub/ds")
    assert len(convs) == 4
    for conv in convs:
        assert conv[0]["role"] == "user"
        assert conv[-1]["role"] == "assistant"
