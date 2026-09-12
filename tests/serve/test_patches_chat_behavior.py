#!/usr/bin/env python3
"""Chat-behavior patches: thinking budget, template kwargs, role
normalization, harmony split - carved from test_server_patches.py."""
from __future__ import annotations

import importlib
import types

import pytest

pytest.importorskip("mlx_vlm")

import gmlx.serve.patches as sp  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import chat_behavior as sp_chat  # noqa: E402
import gmlx.serve.bridge_vlm as serving  # noqa: E402
from gmlx.config import ResolvedModel  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_UTILS = importlib.import_module("mlx_vlm.utils")
_PKG = importlib.import_module("mlx_vlm.server")

from test_server_patches import _FakeThinkTok  # noqa: E402


# 1b. thinking_budget enforcement fix (generate-<think> models)
def _drive(criteria, tokens):
    """Feed token ids; return the forced id (or None) the criteria emits each step."""
    return [criteria(t) for t in tokens]


def _armed(budget, prompt_open):
    cls = sp_chat._armed_thinking_budget_criteria_cls()
    return cls(tokenizer=_FakeThinkTok(), thinking_budget=budget,
               thinking_start_token="<think>", thinking_end_token="</think>",
               enable_thinking=True, prompt_open_thinking=prompt_open)


def test_armed_criteria_caps_generated_think():
    # prompt did NOT pre-fill <think> (Qwen3 case); the model generates it.
    c = _armed(3, prompt_open=False)
    assert c.in_thinking is False                       # not started in a block
    forced = _drive(c, [99, 1, 2, 3, 4, 5, 6])          # <think> then 6 words
    assert 10 in forced and 100 in forced              # forced \n then </think>
    assert forced.index(10) < forced.index(100)


def test_armed_criteria_caps_prefilled_think():
    # prompt pre-filled <think> (GLM-5.2 case): counting starts immediately.
    c = _armed(2, prompt_open=True)
    assert c.in_thinking is True
    forced = _drive(c, [1, 2, 3, 4, 5])
    assert 100 in forced                               # forced close


def test_armed_criteria_never_forces_non_thinking_answer():
    # thinking enabled + budget set, but the model never opens a <think> ->
    # no token is ever counted, so nothing is force-closed (no corruption).
    c = _armed(2, prompt_open=False)
    forced = _drive(c, [1, 2, 3, 4, 5, 6, 7, 8])
    assert all(f is None for f in forced)


def test_armed_criteria_reset_restores_prompt_seed():
    c = _armed(2, prompt_open=False)
    c.in_thinking = True
    c.reset_thinking_state()
    assert c.in_thinking is False                       # back to the prompt seed


def test_prompt_tail_opens_thinking_cases():
    pairs = (("<think>", "</think>"),)
    f = sp_chat._prompt_tail_opens_thinking
    assert f("", pairs) is False
    assert f("plain prompt, no markers", pairs) is False
    assert f("<|im_start|>assistant\n<think>\n", pairs) is True   # Qwen3.6 pre-fill
    assert f("x<think>\n\n</think>\n\n", pairs) is False          # thinking off
    assert f("a<think>x</think>b<think>", pairs) is True          # last pair open
    assert f(None, pairs) is False


def test_stream_thinking_seed_reseeds_from_prompt():
    rs = importlib.import_module("mlx_vlm.server.responses_state")
    cls = rs.ThinkingStreamState
    original_init = cls.__init__
    try:
        sp.install_stream_thinking_seed()
        assert getattr(cls.__init__, sp_chat._STREAM_SEED_FLAG, False)
        patched = cls.__init__
        sp.install_stream_thinking_seed()                    # idempotent
        assert cls.__init__ is patched
        tok = sp_chat._LAST_RENDERED_PROMPT.set("rendered, no thinking scaffold")
        try:
            st = cls(True)               # enable_thinking forced True (7b)
            assert st.in_thinking is False   # gemma-4 default-off: content mode
            sp_chat._LAST_RENDERED_PROMPT.set("<|im_start|>assistant\n<think>\n")
            st = cls(True)
            assert st.in_thinking is True    # Qwen3.6 pre-fill: reasoning first
            st = cls(False)
            assert st.in_thinking is True    # prompt truth beats the flag
            sp_chat._LAST_RENDERED_PROMPT.set(None)
            st = cls(True)
            assert st.in_thinking is True    # no render seen -> stock seed
        finally:
            sp_chat._LAST_RENDERED_PROMPT.reset(tok)
    finally:
        cls.__init__ = original_init


def test_stream_thinking_seed_wraps_render_binding():
    openai_mod = importlib.import_module("mlx_vlm.server.openai")
    rs = importlib.import_module("mlx_vlm.server.responses_state")
    original_fn = openai_mod.apply_chat_template
    original_init = rs.ThinkingStreamState.__init__
    try:
        openai_mod.apply_chat_template = lambda *a, **kw: "tail <think>\n"
        sp.install_stream_thinking_seed()
        wrapped = openai_mod.apply_chat_template
        assert getattr(wrapped, sp_chat._STREAM_SEED_FLAG, False)
        out = wrapped("processor", "config", [])
        assert out == "tail <think>\n"
        assert sp_chat._LAST_RENDERED_PROMPT.get() == out         # stashed
        sp_chat._LAST_RENDERED_PROMPT.set(None)
    finally:
        openai_mod.apply_chat_template = original_fn
        rs.ThinkingStreamState.__init__ = original_init


def test_nonstream_split_truncated_thinking_seeded_from_prompt():
    app_mod = importlib.import_module("mlx_vlm.server.app")
    sp.install_stream_thinking_seed()
    split = app_mod._split_thinking_text
    assert getattr(split, sp_chat._STREAM_SEED_FLAG, False)
    tok = sp_chat._LAST_RENDERED_PROMPT.set(None)
    try:
        # No stashed render (off-request callers): stock classification.
        assert split("half a plan, cut off") == (None, "half a plan, cut off")
        sp_chat._LAST_RENDERED_PROMPT.set("<|im_start|>assistant\n<think>\n")
        # Prompt-opened block, no marker in the truncated text: reasoning.
        assert split("half a plan, cut off") == ("half a plan, cut off", "")
        # A close marker in the text keeps the stock split untouched.
        r, c = split("plan\n</think>\n\nanswer")
        assert r == "plan" and c == "answer"
        # Prompt did not open a block: stock classification.
        sp_chat._LAST_RENDERED_PROMPT.set("<|im_start|>assistant\n")
        assert split("just an answer") == (None, "just an answer")
    finally:
        sp_chat._LAST_RENDERED_PROMPT.reset(tok)


def test_nonstream_split_strips_xtml_section_markers():
    """Kimi-K3: the response/message closers and turn terminator must not
    leak into chat content; other think spellings stay untouched."""
    app_mod = importlib.import_module("mlx_vlm.server.app")
    sp.install_stream_thinking_seed()
    split = app_mod._split_thinking_text
    tok = sp_chat._LAST_RENDERED_PROMPT.set(None)
    try:
        text = ("plan<|close|>think<|sep|><|open|>response<|sep|>"
                "2+2 equals 4.<|close|>response<|sep|>"
                "<|close|>message<|sep|><|end_of_msg|>")
        r, c = split(text, "<|open|>think<|sep|>", "<|close|>think<|sep|>")
        assert r == "plan"
        assert c == "2+2 equals 4."
        # Gate: a different think spelling passes content through unchanged.
        r, c = split("x</think>keep <|close|>message<|sep|> text",
                     "<think>", "</think>")
        assert c == "keep <|close|>message<|sep|> text"
    finally:
        sp_chat._LAST_RENDERED_PROMPT.reset(tok)


def test_nonstream_split_keeps_first_line_code_indent():
    """The stock splitter .strip()s content, which deletes the first line's
    leading indent from verbatim-code replies. The seeded splitter trims
    newlines only, for every marker branch and the no-marker fallthrough."""
    app_mod = importlib.import_module("mlx_vlm.server.app")
    rs = importlib.import_module("mlx_vlm.server.responses_state")
    sp.install_stream_thinking_seed()
    split = app_mod._split_thinking_text
    tok = sp_chat._LAST_RENDERED_PROMPT.set(None)
    try:
        # open+close pair
        r, c = split("<think>plan</think>\n\n    if x:\n        y()")
        assert r == "plan"
        assert c == "    if x:\n        y()"
        # close-only (prompt-opened block)
        r, c = split("plan\n</think>\n\n\tdoc = doc || document;")
        assert r == "plan"
        assert c == "\tdoc = doc || document;"
        # no marker at all
        assert split("    indented") == (None, "    indented")
        # whitespace-only content collapses to empty
        assert split("<think>plan</think>\n  \n") == ("plan", "")
        # XTML section strip keeps the indent too
        text = ("plan<|close|>think<|sep|><|open|>response<|sep|>"
                "    return 4;<|close|>response<|sep|><|end_of_msg|>")
        r, c = split(text, "<|open|>think<|sep|>", "<|close|>think<|sep|>")
        assert c == "    return 4;"
        # the /v1/responses module global got the same splitter
        assert rs._split_thinking is split
    finally:
        sp_chat._LAST_RENDERED_PROMPT.reset(tok)


def test_install_thinking_budget_fix_applies_and_idempotent():
    # Fail-loud guard: asserts the seam bound to the REAL mlx-vlm symbol, so a
    # rename of ResponseGenerator._make_thinking_budget_criteria turns into a CI
    # failure instead of a silent no-op.
    gen = importlib.import_module("mlx_vlm.server.generation")
    cls = gen.ResponseGenerator
    original = cls._make_thinking_budget_criteria
    try:
        sp.install_thinking_budget_fix()
        patched = cls._make_thinking_budget_criteria
        assert patched is not original                       # actually bound
        assert getattr(patched, sp_chat._TBUDGET_FLAG, False)
        sp.install_thinking_budget_fix()                     # idempotent
        assert cls._make_thinking_budget_criteria is patched
    finally:
        cls._make_thinking_budget_criteria = original


def test_make_criteria_honors_budget_when_enable_thinking_false():
    # A configured thinking_budget must arm even when enable_thinking is False:
    # a group/profile config may disable thinking, but the model can still emit
    # <think>, and an explicit budget must cap it. None budget still opts out.
    gen = importlib.import_module("mlx_vlm.server.generation")
    cls = gen.ResponseGenerator
    original = cls._make_thinking_budget_criteria
    try:
        sp.install_thinking_budget_fix()
        make = cls._make_thinking_budget_criteria
        me = types.SimpleNamespace(
            tokenizer=_FakeThinkTok(),
            _thinking_token_ids=lambda args: (99, 100),  # <think>=99, </think>=100
        )
        args = types.SimpleNamespace(
            thinking_budget=8, enable_thinking=False,
            thinking_start_token=None, thinking_end_token=None)
        # generate-style prompt (no open <think>): armed but not seeded in-block
        criteria = make(me, args, [1, 2, 3])
        assert criteria is not None                  # armed despite enable_thinking=False
        assert criteria.in_thinking is False
        # pre-fill prompt ending with an open <think> seeds in_thinking True, even
        # though enable_thinking is False - the cap must still fire (GLM-style).
        criteria2 = make(me, args, [1, 2, 99])
        assert criteria2 is not None and criteria2.in_thinking is True
        args.thinking_budget = None
        assert make(me, args, [1, 2, 3]) is None      # no budget -> still opts out
    finally:
        cls._make_thinking_budget_criteria = original


def test_seed_wrapper_survives_thinking_budget_fix():
    # Regression: the install order used to be seed -> tbfix, and tbfix rebinds
    # the criteria seam without delegating, so it clobbered the seed wrapper and
    # per-request seeds were dead on the serve path. The order is now tbfix ->
    # seed: one call must stash the seed AND run tbfix's construction, and a
    # tbfix re-install must see its flag through the wrapper and no-op.
    import gmlx.serve.seed_rows as sr
    gen = importlib.import_module("mlx_vlm.server.generation")
    ar = importlib.import_module("mlx_vlm.generate.ar")
    cls = gen.ResponseGenerator
    saved = (cls._make_thinking_budget_criteria, ar.BatchGenerator.insert,
             ar.GenerationBatch._step, ar.PromptProcessingBatch.generate,
             ar.SpeculativeGenerationBatch.next)
    sr._PENDING.clear()
    try:
        sp.install_thinking_budget_fix()
        sr.install_per_request_seed()
        crit = cls._make_thinking_budget_criteria
        assert getattr(crit, sr._INSTALLED_FLAG, False)      # seed outermost
        assert getattr(crit, sp_chat._TBUDGET_FLAG, False)   # tbfix flag carried
        sp.install_thinking_budget_fix()
        assert cls._make_thinking_budget_criteria is crit    # re-install no-ops
        me = types.SimpleNamespace(
            tokenizer=_FakeThinkTok(),
            _thinking_token_ids=lambda args: (99, 100))
        args = types.SimpleNamespace(
            seed=7, temperature=1.0, thinking_budget=8, enable_thinking=False,
            thinking_start_token=None, thinking_end_token=None)
        out = crit(me, args, [1, 2, 3])
        assert out is not None and out.in_thinking is False  # tbfix ran
        assert sr._PENDING == [7]                            # seed stashed
    finally:
        (cls._make_thinking_budget_criteria, ar.BatchGenerator.insert,
         ar.GenerationBatch._step, ar.PromptProcessingBatch.generate,
         ar.SpeculativeGenerationBatch.next) = saved
        sr._PENDING.clear()


# 7b. chat_template_kwargs passthrough
def _spec_ctkw(**ctkw):
    return ResolvedModel(id="m", path="/p", sampling={}, load={}, cache={},
                         system=None, speculative=False, mmproj=None,
                         draft_gguf=None, pin=False, ttl_s=None,
                         chat_template_kwargs=ctkw)


def test_merged_template_kwargs_request_wins_over_profile():
    spec = _spec_ctkw(preserve_thinking=True, foo="profile")
    request = types.SimpleNamespace(chat_template_kwargs={"foo": "request"})
    merged = sp_chat._merged_template_kwargs(request, spec)
    assert merged == {"preserve_thinking": True, "foo": "request"}


def test_merged_template_kwargs_each_side_alone_and_empty():
    # request only (single-model mode: no active spec)
    req = types.SimpleNamespace(chat_template_kwargs={"preserve_thinking": True})
    assert sp_chat._merged_template_kwargs(req, None) == {"preserve_thinking": True}
    # profile only (request carries nothing)
    spec = _spec_ctkw(preserve_thinking=False)
    assert sp_chat._merged_template_kwargs(types.SimpleNamespace(), spec) == {
        "preserve_thinking": False}
    # neither => {}
    assert sp_chat._merged_template_kwargs(types.SimpleNamespace(), None) == {}


def test_merged_template_kwargs_spec_thinking_controls_mapped():
    """Profile-level thinking/reasoning_effort are dedicated controls: mapped
    onto whatever switch the serving model's template reads."""
    spec = _spec_ctkw()
    spec.thinking = "off"
    req = types.SimpleNamespace()
    assert sp_chat._merged_template_kwargs(
        req, spec, "{% if enable_thinking %}...{% endif %}") == \
        {"enable_thinking": False}
    assert sp_chat._merged_template_kwargs(
        req, spec, "reasoning_effort in ['low','high','no_think']") == \
        {"reasoning_effort": "no_think"}
    spec.thinking = "adaptive"
    assert sp_chat._merged_template_kwargs(
        req, spec, 'thinking_mode == "adaptive"') == \
        {"thinking_mode": "adaptive"}
    spec.thinking = None
    spec.reasoning_effort = "high"
    assert sp_chat._merged_template_kwargs(
        req, spec, 'set reasoning_effort = "medium"') == \
        {"reasoning_effort": "high"}


def test_merged_template_kwargs_request_kwargs_beat_spec_controls():
    """A request's explicit chat_template_kwargs pass through verbatim and win
    over the profile's mapped controls."""
    spec = _spec_ctkw()
    spec.thinking = "off"
    spec.reasoning_effort = "low"
    req = types.SimpleNamespace(
        chat_template_kwargs={"enable_thinking": True})
    merged = sp_chat._merged_template_kwargs(
        req, spec, "{% if enable_thinking %}{% endif %} reasoning_effort")
    assert merged["enable_thinking"] is True
    assert merged["reasoning_effort"] == "low"


def test_install_chat_template_kwargs_forwards_into_to_template_kwargs():
    """End-to-end seam: the gen-args wrapper stashes the merged dict and the
    patched to_template_kwargs folds it into what mlx-vlm hands the template."""
    gen = importlib.import_module("mlx_vlm.server.generation")

    def stub(request, processor=None, tenant_id=None):
        return gen.GenerationArguments()

    _APP._build_gen_args = stub
    sp.install_gen_args_profile_injection()
    sp.install_chat_template_kwargs()
    fn = _APP._build_gen_args
    assert getattr(fn, sp_chat._CTKW_FLAG, False)        # stash carried on the chain

    spec = _spec_ctkw(preserve_thinking=True)
    tok = serving.set_active_spec(spec)
    try:
        req = types.SimpleNamespace(model_fields_set=set(),
                                    chat_template_kwargs={"foo": "bar"})
        args = _APP._build_gen_args(req)
    finally:
        serving.reset_active_spec(tok)
    kw = args.to_template_kwargs()
    assert kw["preserve_thinking"] is True          # from the profile
    assert kw["foo"] == "bar"                        # from the request
    # enable_thinking was not explicit (request/spec/env) -> dropped from the
    # template kwargs so the chat template's own default governs (b90aa60),
    # while the args flag stays True for the generation path.
    assert "enable_thinking" not in kw
    assert args.enable_thinking is True

    # explicitly set on the request -> preserved verbatim
    spec = _spec_ctkw()
    tok = serving.set_active_spec(spec)
    try:
        req = types.SimpleNamespace(model_fields_set={"enable_thinking"},
                                    chat_template_kwargs=None,
                                    enable_thinking=False)
        args = _APP._build_gen_args(req)
    finally:
        serving.reset_active_spec(tok)
    assert "enable_thinking" in args.to_template_kwargs()


def test_install_chat_template_kwargs_idempotent_and_noop_default():
    sp.install_chat_template_kwargs()
    gen = importlib.import_module("mlx_vlm.server.generation")
    first = gen.GenerationArguments.to_template_kwargs
    sp.install_chat_template_kwargs()
    assert gen.GenerationArguments.to_template_kwargs is first
    # a request/spec with no kwargs leaves to_template_kwargs untouched (stock keys)
    assert gen.GenerationArguments().to_template_kwargs() == {
        "enable_thinking": gen.GenerationArguments().enable_thinking}


_KIMI_TAIL = (
    "{%- if thinking is defined and thinking is false -%}<think></think>"
    "{%- else -%}<think>{%- endif -%}")


def test_request_thinking_off_maps_onto_kimi_bare_switch():
    """A plain --thinking value forwarded by the chat client (or any client's
    `thinking: "off"`) must reach the template as the model's own switch
    spelling - Kimi K2.x reads a bare `thinking` variable."""
    gen = importlib.import_module("mlx_vlm.server.generation")
    proc = types.SimpleNamespace(chat_template=_KIMI_TAIL)
    req = types.SimpleNamespace(model_fields_set=set(),
                                chat_template_kwargs=None, thinking="off")
    out = sp_chat._stash_template_kwargs(gen.GenerationArguments(), req, proc)
    assert out._kq_template_kwargs == {"thinking": False}
    assert out.enable_thinking is False
    assert out._kq_thinking_explicit is True

    req = types.SimpleNamespace(model_fields_set=set(),
                                chat_template_kwargs=None, thinking="on")
    out = sp_chat._stash_template_kwargs(gen.GenerationArguments(), req, proc)
    assert out._kq_template_kwargs == {"thinking": True}
    assert out.enable_thinking is True


def test_request_reasoning_effort_field_maps_onto_template():
    gen = importlib.import_module("mlx_vlm.server.generation")
    proc = types.SimpleNamespace(chat_template="reads reasoning_effort")
    req = types.SimpleNamespace(model_fields_set=set(),
                                chat_template_kwargs=None,
                                reasoning_effort="low")
    out = sp_chat._stash_template_kwargs(gen.GenerationArguments(), req, proc)
    assert out._kq_template_kwargs == {"reasoning_effort": "low"}
    assert out.enable_thinking is True             # effort alone: not a switch
    assert out._kq_thinking_explicit is False


class _RecordingTok:
    """A tokenizer stand-in whose **kwargs signature makes mlx-vlm's
    enable_thinking capability probe say yes (the 0.6.15 injection path)."""
    chat_template = "{{ messages }}"

    def __init__(self):
        self.kwargs = {}

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "rendered"


def test_template_default_thinking_blocks_0615_false_injection():
    """Regression: mlx-vlm >= 0.6.15 get_chat_template injects
    enable_thinking=False when the kwarg is absent, so served models with
    default-on reasoning templates rendered the dead think prefill and
    stopped thinking. The seeded Jinja Undefined must reach the tokenizer
    instead of False, and an explicit value must pass through verbatim."""
    import jinja2

    import gmlx.tui.reasoning as reasoning

    sp.install_chat_template_kwargs()      # installs the render guard too
    pu = importlib.import_module("mlx_vlm.prompt_utils")
    assert getattr(pu.get_chat_template, reasoning._TEMPLATE_DEFAULT_FLAG, False)

    tok = _RecordingTok()
    msgs = [{"role": "user", "content": "hi"}]
    assert pu.get_chat_template(tok, msgs, True) == "rendered"
    assert isinstance(tok.kwargs["enable_thinking"], jinja2.Undefined)

    pu.get_chat_template(tok, msgs, True, enable_thinking=False)
    assert tok.kwargs["enable_thinking"] is False
    pu.get_chat_template(tok, msgs, True, enable_thinking=True)
    assert tok.kwargs["enable_thinking"] is True

    # idempotent: a second install keeps the same wrapper
    fn = pu.get_chat_template
    reasoning.install_template_default_thinking()
    assert pu.get_chat_template is fn


# The old-style role dispatch from issue #66: no developer alias, else-raise.
_ROLE_RAISE_TMPL = ("{% if m.role == 'system' %}{% elif m.role == 'user' %}"
                    "{% else %}{{ raise_exception('Unexpected message role.') }}"
                    "{% endif %}")


def test_normalize_developer_roles():
    msgs = [{"role": "developer", "content": "terse"},
            {"role": "user", "content": "hi"}, "not-a-dict"]
    out = sp_chat._normalize_developer_roles(msgs, _ROLE_RAISE_TMPL)
    assert out[0] == {"role": "system", "content": "terse"}
    assert out[1]["role"] == "user" and out[2] == "not-a-dict"
    assert msgs[0]["role"] == "developer"          # input not mutated
    # a template that handles developer gets the messages verbatim
    assert sp_chat._normalize_developer_roles(
        msgs, "role == 'developer'") is msgs
    # nothing to rewrite -> same object
    plain = [{"role": "user", "content": "hi"}]
    assert sp_chat._normalize_developer_roles(plain, _ROLE_RAISE_TMPL) is plain
    assert sp_chat._normalize_developer_roles("prompt", _ROLE_RAISE_TMPL) == \
        "prompt"


def test_install_role_normalization_rewrites_before_render():
    """Issue #66: a developer-role request against a template without the
    alias must render as system instead of raising in the template."""
    class _Recorder:
        chat_template = _ROLE_RAISE_TMPL

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "rendered"

    sp.install_role_normalization()
    openai = importlib.import_module("mlx_vlm.server.openai")
    assert getattr(openai.apply_chat_template,
                   sp_chat._ROLE_NORM_FLAG, False)

    tok = _Recorder()
    out = openai.apply_chat_template(
        tok, {"model_type": "gguf-llama"},
        [{"role": "developer", "content": "terse"},
         {"role": "user", "content": "hi"}])
    assert out == "rendered"
    assert [m["role"] for m in tok.messages
            if isinstance(m, dict) and "role" in m][:2] == ["system", "user"]

    # idempotent
    fn = openai.apply_chat_template
    sp.install_role_normalization()
    assert openai.apply_chat_template is fn


def test_template_error_becomes_clean_400():
    """A raise_exception from the chat template answers 400 with the
    template's message; template bugs (subclasses) stay 500, clean body."""
    import jinja2
    from fastapi.testclient import TestClient

    app = _APP.app
    if not any(getattr(r, "path", None) == "/test/raise-template"
               for r in app.router.routes):
        @app.get("/test/raise-template")
        async def _raise_template():
            raise jinja2.exceptions.TemplateError("Unexpected message role.")

        @app.get("/test/raise-template-bug")
        async def _raise_template_bug():
            raise jinja2.exceptions.TemplateSyntaxError("bad", 1)

    sp.install_resolver_error_handlers()
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/test/raise-template")
    assert r.status_code == 400
    assert r.json()["error"] == {
        "type": "invalid_request_error",
        "message": "chat template rejected the conversation: "
                   "Unexpected message role."}
    r2 = client.get("/test/raise-template-bug")
    assert r2.status_code == 500
    assert r2.json()["error"]["type"] == "server_error"
    assert "chat template failed to render" in r2.json()["error"]["message"]


# harmony (gpt-oss) serve-side split
_HARMONY_PROMPT = ("<|start|>system<|message|>You are helpful.<|end|>"
                   "<|start|>user<|message|>hi<|end|><|start|>assistant")
_HARMONY_REPLY = ('<|channel|>analysis<|message|>User greets; keep it short.'
                  "<|end|><|start|>assistant<|channel|>final<|message|>"
                  "Hello! How can I help?")


def test_nonstream_split_harmony_reply():
    app_mod = importlib.import_module("mlx_vlm.server.app")
    sp.install_stream_thinking_seed()
    split = app_mod._split_thinking_text
    tok = sp_chat._LAST_RENDERED_PROMPT.set(None)
    try:
        r, c = split(_HARMONY_REPLY)
        assert r == "User greets; keep it short."
        assert c == "Hello! How can I help?"
        assert "<|" not in c and "<|" not in r
        # Length-capped inside analysis: all reasoning, empty content
        # (the truncated-thinking convention).
        r, c = split("<|channel|>analysis<|message|>Entry 55 reads 4")
        assert r == "Entry 55 reads 4"
        assert c == ""
        # Gemma's lopsided spelling must not take the harmony branch.
        r, c = split("<|channel>thought\nplan\n<channel|>Hi there.")
        assert c and "<|channel|>" not in c
    finally:
        sp_chat._LAST_RENDERED_PROMPT.reset(tok)


def test_stream_harmony_filter_routes_channels():
    rs = importlib.import_module("mlx_vlm.server.responses_state")
    cls = rs.ThinkingStreamState
    original_init = cls.__init__
    original_feed = cls.feed
    try:
        sp.install_stream_thinking_seed()
        tok = sp_chat._LAST_RENDERED_PROMPT.set(_HARMONY_PROMPT)
        try:
            st = cls(True)
            assert getattr(st, "_kq_harmony", None) is not None
            reasoning, content, closes = [], [], 0
            for i in range(0, len(_HARMONY_REPLY), 7):
                d = st.feed(_HARMONY_REPLY[i:i + 7])
                if d.reasoning:
                    reasoning.append(d.reasoning)
                if d.content:
                    content.append(d.content)
                closes += bool(d.thinking_closed)
            assert "".join(reasoning) == "User greets; keep it short."
            assert "".join(content) == "Hello! How can I help?"
            assert closes == 1
            # Non-harmony prompt: the stock state machine still drives.
            sp_chat._LAST_RENDERED_PROMPT.set("<|im_start|>assistant\n<think>\n")
            st = cls(True)
            assert getattr(st, "_kq_harmony", None) is None
            d = st.feed("plan</think>answer")
            assert d.reasoning == "plan" and d.content == "answer"
        finally:
            sp_chat._LAST_RENDERED_PROMPT.reset(tok)
    finally:
        cls.__init__ = original_init
        cls.feed = original_feed


def test_faithful_history_aliases_gpt_oss_thinking():
    from gmlx.serve.patches import render as sp_render

    def fake(processor, config, prompt, add_generation_prompt=True,
             return_messages=False, num_images=0, num_audios=0, **kwargs):
        return [dict(m) for m in prompt]

    # Exercise through the installer against a stub target module.
    target = types.SimpleNamespace(apply_chat_template=fake)
    orig_targets = sp_common._render_target_modules
    sp_common._render_target_modules = lambda: [target]
    try:
        sp_render.install_faithful_history()
    finally:
        sp_common._render_target_modules = orig_targets
    wrapped = target.apply_chat_template
    assert wrapped is not fake
    msgs = wrapped(
        "processor", {"model_type": "gpt_oss"},
        [{"role": "assistant", "content": "Hi.",
          "reasoning_content": "Short greeting."}],
        return_messages=True)
    assert msgs[0]["thinking"] == "Short greeting."
    # Explicit thinking key wins; non-gpt-oss untouched.
    msgs = wrapped(
        "processor", {"model_type": "gpt_oss"},
        [{"role": "assistant", "content": "Hi.", "thinking": "keep",
          "reasoning_content": "drop"}],
        return_messages=True)
    assert msgs[0]["thinking"] == "keep"
    msgs = wrapped(
        "processor", {"model_type": "qwen3"},
        [{"role": "assistant", "content": "Hi.",
          "reasoning_content": "r"}],
        return_messages=True)
    assert "thinking" not in msgs[0]
