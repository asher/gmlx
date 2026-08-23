"""Next-turn retirement keys: memo hops, LCP prediction, store decisions."""

import mlx.core as mx
import pytest
from mlx_vlm.models.cache import KVCache, RotatingKVCache

from gmlx import cache_snapshot as cs
from gmlx import retire_key as rk
from gmlx import speculative as sp


@pytest.fixture(autouse=True)
def _clean_memos():
    rk._reset_for_tests()
    yield
    rk._reset_for_tests()


def _fake_ctx(next_ids, media=False, decoded="hello"):
    class Tok:
        eos_token = "<eos>"

        def decode(self, ids, skip_special_tokens=False):
            return decoded

    class Proc:
        tokenizer = Tok()

    return {
        "messages": [{"role": "user", "content": "hi"}],
        "kw": {},
        "processor": Proc(),
        "config": {},
        "render": lambda p, c, msgs, **kw: "RENDER",
        "preprocess": lambda text: {"input_ids": [list(next_ids)]},
        "media": media,
    }


def _kv(tokens, heads=2, dim=4):
    c = KVCache()
    c.update_and_fetch(mx.zeros((1, heads, tokens, dim)),
                       mx.zeros((1, heads, tokens, dim)))
    return c


class _Manager:
    def __init__(self):
        self.exact = []

    def store_exact_cache(self, ids, snap, extra_hash=0):
        self.exact.append((list(ids), snap))
        return True

    def release(self, blocks):
        pass


# -- memo hops --

def test_memo_hop_and_lookup():
    rk.register_render("T", {"messages": [], "kw": {}})
    pre = object()
    rk.register_ids("T", [1, 2, 3], pre)
    ctx = rk.lookup_render_ctx([1, 2, 3])
    assert ctx is not None and ctx["preprocess"] is pre
    assert rk.lookup_render_ctx([1, 2]) is None


def test_ids_hop_requires_registered_text():
    rk.register_ids("never-rendered", [7, 8], object())
    assert rk.lookup_render_ctx([7, 8]) is None


def test_memo_caps():
    for i in range(rk._TEXT_MEMO_CAP + 4):
        rk.register_render(f"t{i}", {})
    assert len(rk._TEXT_MEMO) == rk._TEXT_MEMO_CAP
    rk.register_render("k", {})
    for i in range(rk._IDS_MEMO_CAP + 4):
        rk.register_ids("k", [i], object())
    assert len(rk._IDS_MEMO) == rk._IDS_MEMO_CAP


# -- LCP prediction --

def test_lcp_faithful_render_covers_sequence():
    seq = [1, 2, 3, 4]
    ctx = _fake_ctx(next_ids=[1, 2, 3, 4, 9])
    assert rk.next_turn_lcp(ctx, seq, [4]) == 4


def test_lcp_divergence_point():
    seq = [1, 2, 3, 4]
    ctx = _fake_ctx(next_ids=[1, 2, 9, 9])
    assert rk.next_turn_lcp(ctx, seq, [4]) == 2


def test_lcp_media_and_missing_ctx_return_none():
    assert rk.next_turn_lcp(None, [1], [1]) is None
    assert rk.next_turn_lcp(_fake_ctx([1], media=True), [1], [1]) is None


def test_lcp_prediction_failure_returns_none():
    ctx = _fake_ctx([1, 2])
    ctx["render"] = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError())
    assert rk.next_turn_lcp(ctx, [1, 2], [2]) is None


# -- prompt-stable LCP (render-stable turn boundaries) --

def test_prompt_stable_lcp_excludes_gen_tail():
    # Positions past the assistant-free re-render (the gen-prompt/think
    # tail) do not survive turn 2.
    ctx = _fake_ctx(next_ids=[1, 2, 3, 4])
    assert rk.prompt_stable_lcp(ctx, [1, 2, 3, 4, 90, 91]) == 4


def test_prompt_stable_lcp_renders_without_assistant_once():
    calls = []
    ctx = _fake_ctx(next_ids=[1, 2, 3])

    def render(p, c, msgs, **kw):
        calls.append((len(msgs), kw.get("add_generation_prompt")))
        return "RENDER"

    ctx["render"] = render
    assert rk.prompt_stable_lcp(ctx, [1, 2, 3, 9]) == 3
    assert rk.prompt_stable_lcp(ctx, [1, 2, 3, 9]) == 3   # ctx memo
    # One render only (the memo), no assistant appended: the toolless
    # branch appends the dummy-user demotion probe, so two messages.
    assert calls == [(2, False)]


def test_prompt_stable_lcp_failure_memoized():
    calls = []

    def boom(*a, **kw):
        calls.append(1)
        raise RuntimeError()

    ctx = _fake_ctx([1, 2])
    ctx["render"] = boom
    assert rk.prompt_stable_lcp(ctx, [1, 2]) is None
    assert rk.prompt_stable_lcp(ctx, [1, 2]) is None
    assert len(calls) == 1
    assert rk.prompt_stable_lcp(None, [1]) is None
    assert rk.prompt_stable_lcp(_fake_ctx([1], media=True), [1]) is None


def test_build_assistant_message_thinking_split():
    ctx = _fake_ctx([1])
    msg = rk.build_assistant_message(
        ctx, "<think>\nplan\n</think>\n\nanswer")
    assert msg["role"] == "assistant"
    assert msg["content"].strip() == "answer"
    assert "plan" in (msg.get("reasoning_content") or "")


def test_truncated_thinking_predicate():
    pairs = (("<think>", "</think>"),)
    f = rk.truncated_thinking
    assert f("cut off mid plan", pairs, "assistant\n<think>\n") is True
    assert f("plan</think>done", pairs, "assistant\n<think>\n") is False
    assert f("<think>plan", pairs, "assistant\n<think>\n") is False
    assert f("cut off", pairs, "assistant\n") is False
    assert f("cut off", pairs, "x<think>a</think>b") is False
    assert f("", pairs, "assistant\n<think>\n") is False
    assert f("cut off", pairs, None) is False


def test_build_assistant_message_truncated_thinking():
    # Prompt-opened think block, budget exhausted before the close marker:
    # the partial reasoning is reasoning, not content (server-shape mirror).
    ctx = _fake_ctx([1])
    ctx["_gen_prompt"] = "<|im_start|>assistant\n<think>\n"
    msg = rk.build_assistant_message(ctx, "half a plan, cut off")
    assert msg["content"] == ""
    assert msg["reasoning_content"] == "half a plan, cut off"


def test_build_assistant_message_no_open_block_stays_content():
    ctx = _fake_ctx([1])
    ctx["_gen_prompt"] = "<|im_start|>assistant\n"
    msg = rk.build_assistant_message(ctx, "just an answer")
    assert msg["content"] == "just an answer"
    assert "reasoning_content" not in msg


def test_build_assistant_message_harmony_channel_split():
    # Harmony templates raise on raw <|channel|> tags in content, which
    # made every next-turn render prediction fail and retirement store
    # full-depth unmatchable keys (gpt-oss cert: 406 ids_diverged).
    ctx = _fake_ctx([1])
    ctx["_gen_prompt"] = "<|start|>assistant"
    msg = rk.build_assistant_message(
        ctx, "<|channel|>analysis<|message|>weigh the options<|end|>"
             "<|start|>assistant<|channel|>final<|message|>the answer")
    assert msg["content"] == "the answer"
    assert msg["reasoning_content"] == "weigh the options"
    assert msg["thinking"] == "weigh the options"
    assert "<|channel|>" not in msg["content"]


def test_build_assistant_message_gemma_channel_split():
    ctx = _fake_ctx([1])
    ctx["_gen_prompt"] = "<|turn>model\n"
    msg = rk.build_assistant_message(
        ctx, "<|channel>thought\nponder the ask\n<channel|>the answer")
    assert msg["content"] == "the answer"
    assert msg["reasoning_content"] == "ponder the ask"


def test_channel_split_ignores_markerless_text():
    assert rk._channel_split("plain reply, no markers") == (None, None)
    r, c = rk._channel_split("<|channel|>final<|message|>answer only")
    assert r is None and c is None  # nothing classified as reasoning


def test_predict_render_gets_generation_prompt_off():
    seen = {}

    def render(p, c, msgs, **kw):
        seen.update(kw)
        seen["last"] = msgs[-1]
        return "X"

    ctx = _fake_ctx([1, 2, 3])
    ctx["render"] = render
    rk.predict_next_ids(ctx, {"role": "assistant", "content": "a"})
    assert seen["add_generation_prompt"] is False
    assert seen["last"]["role"] == "assistant"


# -- retirement_store max_len mechanics --

def test_block_store_truncates_ids(monkeypatch):
    seen = {}

    def harvest(manager, cache, row, ids, extra_hash=0):
        seen["ids"] = list(ids)
        return ["b"]

    import mlx_vlm.apc as apc
    monkeypatch.setattr(apc, "harvest_blocks_from_batch_cache", harvest)
    ok = cs.retirement_store(_Manager(), "block", [1, 2, 3, 4, 5], [_kv(5)],
                             max_len=3)
    assert ok and seen["ids"] == [1, 2, 3]


def test_exact_store_truncates_all_kv_snapshot():
    m = _Manager()
    ok = cs.retirement_store(m, "exact", [1, 2, 3, 4, 5, 6],
                             [_kv(6), _kv(6)], max_len=4)
    assert ok
    ids, snap = m.exact[0]
    assert ids == [1, 2, 3, 4]
    assert all(int(c.offset) == 4 and c.keys.shape[2] == 4 for c in snap)


def test_exact_store_untruncated_without_max_len():
    m = _Manager()
    ok = cs.retirement_store(m, "exact", [1, 2, 3, 4, 5, 6], [_kv(6)])
    assert ok and m.exact[0][0] == [1, 2, 3, 4, 5, 6]


def test_exact_store_skips_rotating_truncation():
    m = _Manager()
    rot = RotatingKVCache(max_size=4)
    rot.update_and_fetch(mx.zeros((1, 2, 6, 4)), mx.zeros((1, 2, 6, 4)))
    ok = cs.retirement_store(m, "exact", [1, 2, 3, 4, 5, 6],
                             [_kv(6), rot], max_len=4)
    assert not ok and not m.exact


def test_ckpt_store_skipped_on_truncation(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("ckpt_store must not run on a truncated key")

    monkeypatch.setattr(cs, "ckpt_store", boom)
    ok = cs.retirement_store(_Manager(), "ckpt", [1, 2, 3, 4], [_kv(4)],
                             max_len=3)
    assert not ok


# -- _retire_b1 decision wiring --

def _retire_setup(monkeypatch, lcp, mode="exact"):
    calls = {}

    def fake_store(manager, m, seq, cache, row=0, extra_hash=0,
                   max_len=None, decode_snaps=None):
        calls["store"] = (list(seq), max_len, m)
        return True

    monkeypatch.setattr(cs, "retirement_store", fake_store)
    monkeypatch.setattr(rk, "next_turn_lcp",
                        lambda ctx, seq, gen: lcp)

    class Model:
        _kq_apc_manager = _Manager()

    full = [1, 2, 3]
    gen = [4, 5]
    cache = [_kv(5)]
    retire = {"full_ids": full, "mode": mode, "extra_hash": 0,
              "render_ctx": {"messages": []}}
    return Model(), cache, gen, retire, calls


def test_retire_b1_full_match_stores_whole(monkeypatch):
    model, cache, gen, retire, calls = _retire_setup(monkeypatch, lcp=5)
    sp._retire_b1(model, cache, gen, retire)
    seq, max_len, _ = calls["store"]
    assert seq == [1, 2, 3, 4, 5] and max_len is None


def test_retire_b1_divergence_passes_max_len(monkeypatch):
    model, cache, gen, retire, calls = _retire_setup(monkeypatch, lcp=2)
    sp._retire_b1(model, cache, gen, retire)
    assert calls["store"][1] == 2


def test_retire_b1_truncation_skips_drafter_sidecar(monkeypatch):
    model, cache, gen, retire, calls = _retire_setup(monkeypatch, lcp=2)

    def boom(*a, **kw):
        raise AssertionError("sidecar must not store under a truncated key")

    monkeypatch.setattr(cs, "drafter_sidecar_store", boom)

    class Drafter:
        _kq_head_covered = True
        _kq_head_request = None

    sp._retire_b1(model, cache, gen, retire, drafter=Drafter())
    assert calls["store"][1] == 2


def test_retire_b1_kill_switch(monkeypatch):
    model, cache, gen, retire, calls = _retire_setup(monkeypatch, lcp=2)
    monkeypatch.setenv("GMLX_APC_RETIRE_LCP", "0")

    def boom(*a, **kw):
        raise AssertionError("lcp must not run with the kill switch set")

    monkeypatch.setattr(rk, "next_turn_lcp", boom)
    sp._retire_b1(model, cache, gen, retire)
    assert calls["store"][1] is None


def test_retire_b1_no_ctx_keeps_today(monkeypatch):
    model, cache, gen, retire, calls = _retire_setup(monkeypatch, lcp=None)
    retire["render_ctx"] = None
    sp._retire_b1(model, cache, gen, retire)
    assert calls["store"][1] is None


# -- mid-decode partial prediction: virtual thinking closer --


def test_virtually_finish_closes_open_thinking():
    # Prompt-opened block (qwen style: the start marker never appears in
    # the generated text). Default markers -- the request kwargs rarely
    # carry explicit thinking tokens.
    ctx = {"kw": {}, "_gen_prompt": "<|im_start|>assistant\n<think>\n"}
    assert rk._virtually_finish(ctx, "partial reasoning") == \
        "partial reasoning</think>"
    closed = "r\n</think>\n\nanswer"
    assert rk._virtually_finish(ctx, closed) == closed
    # A start marker inside the text triggers the closer even when the
    # prompt did not open the block.
    ctx2 = {"kw": {}, "_gen_prompt": "<|im_start|>assistant\n"}
    assert rk._virtually_finish(ctx2, "<think>partial") == \
        "<think>partial</think>"
    # Neither the text nor the prompt opens a block: untouched.
    assert rk._virtually_finish(ctx2, "plain content") == "plain content"
    # No-think prompts carry a closed empty block: untouched.
    ctx3 = {"kw": {},
            "_gen_prompt": "assistant\n<think>\n\n</think>\n\n"}
    assert rk._virtually_finish(ctx3, "plain content") == "plain content"


def test_gen_prompt_text_memoized():
    calls = []

    def render(proc, cfg, messages, **kw):
        calls.append(kw.get("add_generation_prompt"))
        return "<|im_start|>assistant\n<think>\n"

    ctx = {"render": render, "preprocess": lambda t: [1],
           "processor": None, "config": None, "messages": [], "kw": {}}
    assert rk._gen_prompt_text(ctx).endswith("<think>\n")
    assert rk._gen_prompt_text(ctx).endswith("<think>\n")
    assert calls == [True]
    # The memoized prompt feeds the closer.
    assert rk._virtually_finish(ctx, "deep partial") == \
        "deep partial</think>"


# -- system-prefix LCP (the sibling anchor offset) --

def _sys_ctx(next_ids, msgs):
    ctx = _fake_ctx(next_ids)
    ctx["messages"] = msgs
    return ctx


def _probe_ctx(msgs, template):
    """A ctx whose render/tokenize pair actually varies with the probe
    messages: ``template`` maps a message list to text; tokens are
    character codes, so the probe LCP is a plain string LCP."""
    ctx = _fake_ctx([0])
    ctx["messages"] = msgs
    ctx["render"] = lambda p, c, m, **kw: template(m)
    ctx["preprocess"] = lambda text: [ord(ch) for ch in text]
    return ctx


def _chatml(msgs):
    return "".join(f"<{m['role']}>{m['content']}</>" for m in msgs)


def _folding(msgs):
    """Gemma-style: system content folds into the first user turn; a
    system-only conversation renders to almost nothing."""
    sys_txt = "".join(m["content"] for m in msgs if m["role"] == "system")
    users = [m for m in msgs if m["role"] == "user"]
    if not users:
        return "<bos>"
    return ("<bos><user>" + sys_txt + "\n\n"
            + "".join(u["content"] for u in users) + "</>")


def test_system_prefix_lcp_probe_pair_explicit_system_block():
    msgs = [{"role": "system", "content": "policy"},
            {"role": "user", "content": "real question"}]
    ctx = _probe_ctx(msgs, _chatml)
    live = [ord(ch) for ch in _chatml(msgs)]
    # Probes diverge right after the shared "<system>policy</><user>"
    # header; the anchor offset includes the user header, which every
    # sibling also shares.
    assert rk.system_prefix_lcp(ctx, live) == len("<system>policy</><user>")


def test_system_prefix_lcp_probe_pair_folding_template():
    # The gemma shape: a system-only render measures nothing, but the
    # probe pair folds identically and splits at the real divergence.
    msgs = [{"role": "system", "content": "policy"},
            {"role": "user", "content": "real question"}]
    ctx = _probe_ctx(msgs, _folding)
    live = [ord(ch) for ch in _folding(msgs)]
    assert rk.system_prefix_lcp(ctx, live) == len("<bos><user>policy\n\n")


def test_system_prefix_lcp_probe_renders_lead_plus_dummy():
    calls = []
    ctx = _sys_ctx([1, 2, 3], [{"role": "system", "content": "a"},
                               {"role": "system", "content": "b"},
                               {"role": "user", "content": "u"}])
    inner = ctx["render"]
    ctx["render"] = (lambda p, c, msgs, **kw:
                     (calls.append(list(msgs)), inner(p, c, msgs, **kw))[1])
    assert rk.system_prefix_lcp(ctx, [1, 2, 3, 7, 8]) == 3
    lead = [{"role": "system", "content": "a"},
            {"role": "system", "content": "b"}]
    assert calls == [lead + [{"role": "user", "content": "0"}],
                     lead + [{"role": "user", "content": "1"}]]
    # Memoized: a second call does not re-render.
    assert rk.system_prefix_lcp(ctx, [1, 2, 3, 7, 8]) == 3
    assert len(calls) == 2


def test_system_prefix_lcp_clamped_by_live_prompt():
    # The live-prompt clamp guards against a template leaking user
    # content ahead of the divergence point.
    ctx = _sys_ctx([1, 2, 9, 9], [{"role": "system", "content": "s"},
                                  {"role": "user", "content": "u"}])
    assert rk.system_prefix_lcp(ctx, [1, 2, 3, 4, 5]) == 2


def test_system_prefix_lcp_requires_system_prefix_and_a_tail():
    # No leading system block: nothing siblings share by construction.
    ctx = _sys_ctx([1, 2], [{"role": "user", "content": "u"}])
    assert rk.system_prefix_lcp(ctx, [1, 2, 3]) is None
    # System-only prompt: the terminal checkpoint already covers it.
    ctx = _sys_ctx([1, 2], [{"role": "system", "content": "s"}])
    assert rk.system_prefix_lcp(ctx, [1, 2, 3]) is None
    # System block after the first non-system message does not count.
    ctx = _sys_ctx([1, 2], [{"role": "user", "content": "u"},
                            {"role": "system", "content": "s"}])
    assert rk.system_prefix_lcp(ctx, [1, 2, 3]) is None


def test_system_prefix_lcp_media_and_failure():
    assert rk.system_prefix_lcp(
        _sys_ctx([1], [{"role": "system", "content": "s"},
                       {"role": "user", "content": "u"}]) | {"media": True},
        [1, 2]) is None
    calls = []
    ctx = _sys_ctx([1, 2], [{"role": "system", "content": "s"},
                            {"role": "user", "content": "u"}])
    ctx["render"] = (lambda p, c, msgs, **kw:
                     (calls.append(1), (_ for _ in ()).throw(
                         ValueError("template refuses")))[1])
    assert rk.system_prefix_lcp(ctx, [1, 2]) is None
    # Failure memoized: no second render attempt.
    assert rk.system_prefix_lcp(ctx, [1, 2]) is None
    assert len(calls) == 1
