"""Assistant prefill on /v1/chat/completions: ``continue_final_message`` and
``add_generation_prompt`` render the prompt llama-server renders for the same
messages (tests/fixtures/llama_continuation.json), refuse what llama-server
refuses, and refuse what gmlx cannot render the same way."""

from __future__ import annotations

import importlib
import json
import re
import types
from pathlib import Path

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

import gmlx.cache.retire_key as retire_key  # noqa: E402
import gmlx.serve.bridge_vlm as serving  # noqa: E402
import gmlx.serve.estimate as est  # noqa: E402
from gmlx.config import build_config  # noqa: E402
from gmlx.gen.continuation import Continuation  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import chat_behavior as sp_chat  # noqa: E402
from gmlx.serve.patches import prefill  # noqa: E402
from gmlx.serve.patches.apc import install_retire_render_capture  # noqa: E402
from gmlx.serve.patches.capacity_routes import install_estimate  # noqa: E402
from gmlx.serve.patches.render import (  # noqa: E402
    install_chat_image_placement,
    install_faithful_history,
)
from gmlx.upstream.patching import _walk, root  # noqa: E402

_OPENAI = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_PKG = importlib.import_module("gmlx._vendor.mlx_vlm.server")
_PROMPT_UTILS = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")
_RS = importlib.import_module("gmlx._vendor.mlx_vlm.server.responses_state")
_SCHEMAS = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")
_RUNTIME = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
_FIXTURES = Path(__file__).parents[1] / "fixtures"
_FIXTURE = json.loads((_FIXTURES / "llama_continuation.json").read_text())

# Fixture run label -> (model type, BOS text the tokenizer passes the
# template). Each is the type gmlx gives the text GGUF.
_MODELS = {
    "llama-server/qwen35_9b": ("qwen3_5", None),
    "llama-server/gemma4_12b": ("gemma4_text", "<bos>"),
    "llama-server/smollm2": ("llama", None),
}

_HISTORY = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"}]


class _Tokenizer:
    """Renders the template as an HF tokenizer does, with its special
    tokens in the template context."""

    def __init__(self, template, bos=None):
        self.chat_template = template
        self._special = {"bos_token": bos} if bos else {}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt,
            **{**self._special, **kw})
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


class _Processor:
    """A processor that forwards to its tokenizer, as the GGUF text
    processor does."""

    def __init__(self, template, bos=None):
        self._inner = _Tokenizer(template, bos)

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class _Gemma4Processor:
    """Renders as mlx-vlm's Gemma4Processor does: the stock rebuild once
    more, then the tokenizer."""

    apply_chat_template = Gemma4Processor.apply_chat_template
    _extract_media_from_messages = Gemma4Processor._extract_media_from_messages
    _as_list = staticmethod(Gemma4Processor._as_list)

    def __init__(self, template, bos=None):
        self.model_type = "gemma4_unified"
        self.tokenizer = _Tokenizer(template, bos)
        self.chat_template = template


def _processor_for(run, vision=False):
    """The processor and config of the text GGUF of ``run``, or with
    ``vision`` those of a Gemma 4 GGUF set up with ``mmproj:``."""
    row = next(r for r in _FIXTURE["rows"] if r["run"] == run)
    template = (_FIXTURES / row["template"]).read_text()
    model_type, bos = _MODELS[run]
    if vision:
        assert model_type == "gemma4_text"
        return _Gemma4Processor(template, bos), {"model_type": "gemma4_unified"}
    return _Processor(template, bos), {"model_type": model_type}


class _Tok:
    text, finish_reason, token_count = "ok", "stop", 1


class _RG:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        self.calls.append(prompt)
        return types.SimpleNamespace(prompt_tokens=1, uid=1), iter([_Tok()])


def _render_seams():
    state = _RS.ThinkingStreamState
    return ([(t, "apply_chat_template") for t in sp_common._render_target_modules()]
            + [(_OPENAI, "extract_text_from_content"),
               (_OPENAI, "_normalize_instruction_messages"),
               (_OPENAI, "get_cached_model"),
               (state, "__init__"), (state, "feed"),
               (_APP, "_split_thinking_text"), (_RS, "_split_thinking"),
               (_PKG, "apply_chat_template")])


def _install_render_wraps(with_prefill=True):
    """The render wraps in install_server_patches order."""
    install_faithful_history()
    install_chat_image_placement()
    if with_prefill:
        prefill.install_assistant_prefill()
    install_retire_render_capture()
    sp_chat.install_stream_thinking_seed()
    sp_chat.install_role_normalization()


@pytest.fixture
def chat(monkeypatch):
    """The chat route with the gmlx render wraps on a stub model. Returns
    ``post(run, body, status=200)``, which answers the rendered prompt."""
    saved = []
    for obj, attr in _render_seams():
        if hasattr(obj, attr):
            saved.append((obj, attr, getattr(obj, attr)))
            monkeypatch.setattr(obj, attr, getattr(obj, attr))
    rg = _RG()
    monkeypatch.setattr(_RUNTIME, "response_generator", rg, raising=False)
    # A render called in the test's own context sets the prompt the
    # splitters read; later tests on this worker must not see it.
    prompt_token = sp_chat._LAST_RENDERED_PROMPT.set(None)
    retire_key._reset_for_tests()
    sp_chat.install_chat_template_kwargs()
    install_estimate()
    _install_render_wraps()
    from fastapi.testclient import TestClient

    client = TestClient(_APP.app)
    loads = []

    def post(run, body, status=200, path="/v1/chat/completions",
             vision=False):
        processor, config = _processor_for(run, vision)
        monkeypatch.setattr(_OPENAI, "get_cached_model", lambda *a, **k: (
            loads.append(a) or (None, processor, config)))
        r = client.post(path, json={"model": "m", "max_tokens": 1, **body})
        assert r.status_code == status, r.text
        if status == 200:
            return rg.calls.pop()
        err = r.json()
        return err["error"]["message"] if "error" in err else str(err["detail"])

    def reinstall(with_prefill):
        for obj, attr, fn in saved:
            if attr == "apply_chat_template" and obj is not _PKG:
                setattr(obj, attr, fn)
        _install_render_wraps(with_prefill)

    post.loads = loads
    post.reinstall = reinstall
    yield post
    sp_chat._LAST_RENDERED_PROMPT.reset(prompt_token)
    retire_key._reset_for_tests()
    serving.clear_resolved_models()


def _expected(row, rendered):
    """llama-server's prompt, and gmlx's render with the BOS text cut the
    way llama.cpp cuts it when the vocab adds BOS."""
    bos = _MODELS[row["run"]][1]
    if row["bos_cut"]:
        assert rendered.startswith(bos)
        rendered = rendered[len(bos):]
    return row["prompt"], rendered


@pytest.mark.parametrize("row", _FIXTURE["rows"], ids=[r["id"] for r in _FIXTURE["rows"]])
def test_the_prompt_is_llama_servers(chat, row):
    want, got = _expected(row, chat(row["run"], row["body"]))
    assert got == want


def test_gemma_rows_cut_a_bos():
    assert {r["bos_cut"] for r in _FIXTURE["rows"]
            if r["run"] == "llama-server/gemma4_12b"} == {True}


def _prefill(final, cont=True, **extra):
    return {"messages": [*_HISTORY, final], "continue_final_message": cont,
            "add_generation_prompt": False, **extra}


_C = {"role": "assistant", "content": "Hello"}
_QWEN = "llama-server/qwen35_9b"


@pytest.mark.parametrize("body,message", [
    (_prefill(_C, cont="yes"), 'Invalid continue_final_message: expected a '
     'boolean, "content" or "reasoning_content"'),
    (_prefill(_C, cont=1), 'Invalid continue_final_message: expected a '
     'boolean, "content" or "reasoning_content"'),
    ({**_prefill(_C), "add_generation_prompt": "false"}, None),
    ({**_prefill(_C), "add_generation_prompt": True},
     "Cannot set both add_generation_prompt and continue_final_message to true."),
    ({"messages": [*_HISTORY, _C], "continue_final_message": True},
     "Cannot set both add_generation_prompt and continue_final_message to true."),
    (_prefill({"role": "assistant", "content": "x",
               "tool_calls": [{"id": "c", "type": "function", "function": {
                   "name": "f", "arguments": "{}"}}]}),
     "Cannot continue an assistant message that contains tool calls."),
    ({"messages": _HISTORY, "continue_final_message": True,
      "add_generation_prompt": False}, None),
    (_prefill({"role": "assistant", "content": ""}), None),
    (_prefill(_C, response_format={"type": "json_object"}), None),
])
def test_invalid_requests_get_a_400_before_any_model_loads(chat, body, message):
    got = chat(_QWEN, body, status=400)
    if message is not None:
        assert got.startswith(message)
    assert chat.loads == []


def _gen_args_refusal(body, processor=None):
    """The 400 the gen-args transform raises for ``body``, which the
    estimate reaches without the route wrap."""
    from fastapi import HTTPException

    req = _SCHEMAS.ChatRequest(**{"model": "m", **body})
    proc = processor or _processor_for(_QWEN)[0]
    with pytest.raises(HTTPException) as e:
        prefill._stash_continuation(types.SimpleNamespace(), req, proc)
    assert e.value.status_code == 400
    return str(e.value)


def test_the_gen_args_seam_runs_the_transform(chat):
    from fastapi import HTTPException

    req = _SCHEMAS.ChatRequest(model="m", **_prefill(
        {"role": "assistant", "content": None}))
    with pytest.raises(HTTPException) as e:
        _APP._build_gen_args(req, _processor_for(_QWEN)[0])
    assert (e.value.status_code, str(e.value)) == (400, prefill.EMPTY)
    req = _SCHEMAS.ChatRequest(model="m", **_prefill(_C))
    args = _APP._build_gen_args(req, _processor_for(_QWEN)[0])
    assert args._kq_continue.mode == "content" and args._kq_add_gen is False
    assert args.to_template_kwargs()["add_generation_prompt"] is False


@pytest.mark.parametrize("body,message", [
    (_prefill(_C, cont="yes"), prefill.INVALID_VALUE),
    ({**_prefill(_C), "add_generation_prompt": 0}, prefill.AGP_NOT_BOOL),
    ({"messages": [*_HISTORY, _C], "continue_final_message": True},
     prefill.BOTH_TRUE),
    ({"messages": _HISTORY, "continue_final_message": True,
      "add_generation_prompt": False}, prefill.NOT_ASSISTANT),
    (_prefill({"role": "assistant", "content": "x", "tool_calls": [{}]}),
     prefill.TOOL_CALLS),
    (_prefill({"role": "assistant", "content": None}), prefill.EMPTY),
    (_prefill(_C, response_format={"type": "json_schema", "json_schema": {
        "schema": {"type": "object"}}}), prefill.RESPONSE_FORMAT),
])
def test_the_gen_args_transform_repeats_the_request_rules(chat, body, message):
    assert _gen_args_refusal(body) == message


def test_a_text_format_schema_is_refused(chat):
    body = _prefill(_C, text={"format": {"type": "json_schema",
                                         "schema": {"type": "object"}}})
    assert _gen_args_refusal(body) == prefill.RESPONSE_FORMAT
    assert chat(_QWEN, body, status=400) == prefill.RESPONSE_FORMAT


def test_a_text_response_format_and_empty_tool_calls_pass(chat):
    row = next(r for r in _FIXTURE["rows"]
               if r["id"] == "qwen35_9b/think-off/content")
    body = {**row["body"], "response_format": {"type": "text"}}
    body["messages"] = [*body["messages"][:-1],
                        {**body["messages"][-1], "tool_calls": []}]
    assert chat(_QWEN, body) == row["prompt"]


class _OwnRenderer(_Processor):
    def apply_chat_template(self, messages, **kw):
        return "rendered in Python"


class _PythonTokenizer:
    def __init__(self, template):
        self.chat_template = template
        self.tokenizer = types.SimpleNamespace(_chat_template=lambda *a, **k: "")

    def __getattr__(self, name):
        raise AttributeError(name)


def _qwen_template():
    return (_FIXTURES / "qwen35_9b_template.jinja").read_text()


@pytest.mark.parametrize("processor", [
    _Processor(""),
    _Processor((_FIXTURES / "kimi_k3_template.jinja").read_text()),
    _Processor("<kimi-k3-native-python-chat-renderer>"),
    _Processor("plamo2vl"),
    _OwnRenderer(_qwen_template()),
    _Processor((_FIXTURES / "qwen3_template.jinja").read_text()),
], ids=["empty", "kimi-k3-handler", "kimi-k3-sentinel", "plamo2vl-sentinel",
        "own-renderer", "unlisted-autoparser"])
def test_templates_gmlx_cannot_render_get_a_400(chat, processor):
    assert _gen_args_refusal(_prefill(_C), processor) == prefill.UNSUPPORTED


def test_a_python_chat_template_gets_a_400(chat):
    proc = _Processor(_qwen_template())
    proc._inner._chat_template = lambda *a, **k: "python"
    assert _gen_args_refusal(_prefill(_C), proc) == prefill.UNSUPPORTED


_GEMMA_CONT_ROWS = [
    r["id"] for r in _FIXTURE["rows"]
    if r["run"] == "llama-server/gemma4_12b" and "continue_final_message" in r["body"]]


@pytest.mark.parametrize("rid", _GEMMA_CONT_ROWS)
def test_the_gemma4_processor_renders_llama_servers_prompt(chat, rid):
    row = _row(rid)
    processor = _Gemma4Processor(
        (_FIXTURES / row["template"]).read_text(), "<bos>")
    from gmlx.gen.continuation import handler_for, parse_continue, resolve

    msgs = [dict(m) for m in row["body"]["messages"]]
    cont = resolve(parse_continue(row["body"]["continue_final_message"]),
                   msgs[-1], handler_for(processor.chat_template))
    out = _OPENAI.apply_chat_template(
        processor, {"model_type": "gemma4_unified"}, msgs,
        add_generation_prompt=False, **{prefill.CONTINUE_KEY: cont},
        **row["body"]["chat_template_kwargs"])
    want, got = _expected(row, out)
    assert got == want


_GEMMA_ROWS = [r for r in _FIXTURE["rows"]
               if r["run"] == "llama-server/gemma4_12b"]


@pytest.mark.parametrize("row", _GEMMA_ROWS, ids=[r["id"] for r in _GEMMA_ROWS])
def test_a_gemma4_vision_entry_renders_llama_servers_prompt(chat, row):
    want, got = _expected(row, chat(row["run"], row["body"], vision=True))
    assert got == want


@pytest.mark.parametrize("messages", [
    [{"role": "developer", "content": "Be brief."},
     {"role": "user", "content": "Hi"}],
    [{"role": "system", "content": "Be brief."},
     {"role": "developer", "content": "Answer in French."},
     {"role": "user", "content": "Hi"}],
], ids=["leading", "after-system"])
def test_a_gemma4_vision_entry_renders_a_developer_message_as_text(chat,
                                                                  messages):
    run = "llama-server/gemma4_12b"
    body = {"messages": messages}
    got = chat(run, body, vision=True)
    assert " <turn|>" not in got
    assert got == chat(run, body)


# A 1x1 PNG.
_PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==")


def test_a_gemma4_vision_prefill_with_an_image_keeps_the_history(chat):
    run = "llama-server/gemma4_12b"
    history = [{"role": "system", "content": "Be brief."},
               {"role": "user", "content": [
                   {"type": "text", "text": "What is this?"},
                   {"type": "image_url", "image_url": {"url": _PNG}}]}]
    plain = chat(run, {"messages": history, "add_generation_prompt": False},
                 vision=True)
    got = chat(run, _prefill_of(history, {"role": "assistant",
                                          "content": "It is"}), vision=True)
    assert plain.count("<|image|>") == 1
    assert got.startswith(plain)
    assert got.endswith("It is")


def _prefill_of(history, final):
    return {"messages": [*history, final], "continue_final_message": True,
            "add_generation_prompt": False}


def test_template_for_reads_the_template_the_gemma4_processor_renders():
    processor, _ = _processor_for("llama-server/gemma4_12b", vision=True)
    assert prefill.template_for(processor) == processor.tokenizer.chat_template
    prefill._stash_continuation(
        types.SimpleNamespace(),
        _SCHEMAS.ChatRequest(model="m", **_prefill(_C)), processor)


def test_a_gemma4_processor_with_another_template_than_its_tokenizer_gets_a_400():
    processor, _ = _processor_for("llama-server/gemma4_12b", vision=True)
    processor.chat_template = processor.chat_template + "{# other #}"
    assert _gen_args_refusal(_prefill(_C), processor) == prefill.UNSUPPORTED


def test_a_render_error_in_a_prefill_is_not_a_template_refusal():
    """The real render error surfaces, as it does without prefill."""
    processor, config = _processor_for(_QWEN)
    cont = Continuation(mode="content", content="x", reasoning="",
                        handler="qwen3-coder")

    def fails(*a, **kw):
        raise ValueError("bad history")

    with pytest.raises(ValueError, match="bad history"):
        prefill.render_continued(fails, processor, config, [*_HISTORY, _C],
                                 **{prefill.CONTINUE_KEY: cont})


def test_a_render_that_is_not_text_gets_a_400():
    from fastapi import HTTPException

    processor, config = _processor_for(_QWEN)
    cont = Continuation(mode="content", content="x", reasoning="",
                        handler="qwen3-coder")
    with pytest.raises(HTTPException) as e:
        prefill.render_continued(lambda *a, **kw: {"role": "user"}, processor,
                                 config, [*_HISTORY, _C],
                                 **{prefill.CONTINUE_KEY: cont})
    assert (e.value.status_code, e.value.detail) == (400, prefill.UNSUPPORTED)


def test_a_handler_with_no_recipe_gets_a_400_before_any_render():
    from fastapi import HTTPException

    processor, config = _processor_for(_QWEN)
    cont = Continuation(mode="content", content="x", reasoning="",
                        handler="kimi-k3")
    calls = []
    with pytest.raises(HTTPException) as e:
        prefill.render_continued(lambda *a, **kw: calls.append(a) or "",
                                 processor, config, [*_HISTORY, _C],
                                 **{prefill.CONTINUE_KEY: cont})
    assert (e.value.status_code, e.value.detail) == (400, prefill.UNSUPPORTED)
    assert calls == []


def test_a_last_message_only_model_gets_a_400_not_a_500(chat, monkeypatch):
    processor = _Processor(_qwen_template())
    monkeypatch.setattr(_OPENAI, "get_cached_model", lambda *a, **k: (
        None, processor, {"model_type": "paligemma"}))
    from fastapi.testclient import TestClient

    r = TestClient(_APP.app).post("/v1/chat/completions", json={
        "model": "m", "max_tokens": 1, **_prefill(_C)})
    assert r.status_code == 400, r.text
    assert prefill.UNSUPPORTED in r.text


def test_a_gemma_legacy_template_refuses_tool_history(chat):
    src = (_FIXTURES / "gemma4_12b_template.jinja").read_text().replace(
        "{#- OpenAI Chat Completions:", "{#- Chat:")
    proc = _Processor(src, "<bos>")
    tool = [{"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "r"}, _C]
    body = {"messages": tool, "continue_final_message": True,
            "add_generation_prompt": False}
    assert _gen_args_refusal(body, proc) == prefill.UNSUPPORTED


def test_a_request_without_the_fields_renders_as_before(chat):
    body = {"messages": [*_HISTORY, _C],
            "chat_template_kwargs": {"enable_thinking": True}}
    with_prefill = chat(_QWEN, body)
    chat.reinstall(with_prefill=False)
    assert chat(_QWEN, body) == with_prefill
    assert with_prefill.endswith("<|im_start|>assistant\n<think>\n")


def test_a_request_without_the_fields_keeps_its_gen_args(chat):
    """The transform and the to_template_kwargs wrap leave such a request
    as the layers under them build it."""
    from gmlx.upstream.patching import under

    req = _SCHEMAS.ChatRequest(model="m", messages=[*_HISTORY, _C],
                               chat_template_kwargs={"enable_thinking": True})
    processor = _processor_for(_QWEN)[0]
    assert vars(prefill._stash_continuation(
        types.SimpleNamespace(), req, processor)) == {}
    args = _APP._build_gen_args(req, processor)
    assert not {"_kq_continue", "_kq_add_gen"} & set(vars(args))
    cls = importlib.import_module(
        "gmlx._vendor.mlx_vlm.server.generation").GenerationArguments
    stock = under(cls.to_template_kwargs, prefill._FLAG)
    assert stock is not cls.to_template_kwargs
    assert args.to_template_kwargs() == stock(args)


def test_the_install_places_prefill_inside_the_retire_capture(monkeypatch):
    """role(seed(retire(prefill(faithful(stock))))) on every render binding."""
    import gmlx.serve.patches as sp

    for obj, attr in _render_seams():
        if hasattr(obj, attr):
            monkeypatch.setattr(obj, attr, getattr(obj, attr))
    # Start from the stock render, whatever wraps earlier tests left.
    for target in sp_common._render_target_modules():
        monkeypatch.setattr(target, "apply_chat_template",
                            root(target.apply_chat_template))
    sp.install_server_patches(build_config({"server": {}}),
                              reload_fn=lambda *a, **k: None)
    flags = ["_kq_gguf_role_normalization", "_kq_gguf_stream_thinking_seed",
             "_kq_retire_capture", prefill._FLAG, "_kq_faithful_history"]
    for target in sp_common._render_target_modules():
        chain = list(_walk(target.apply_chat_template))
        order = [next(i for i, f in enumerate(chain) if getattr(f, flag, False))
                 for flag in flags]
        assert order == sorted(order), target


def test_the_route_check_runs_outside_the_dry_run_wrap(chat):
    body = {**_prefill(_C), "add_generation_prompt": True, "dry_run": True}
    assert chat(_QWEN, body, status=400).startswith("Cannot set both")


def _row(rid):
    return next(r for r in _FIXTURE["rows"] if r["id"] == rid)


def _ids(text):
    return list(text.encode())


def test_a_prefill_render_registers_a_reduced_retire_ctx(chat):
    row = _row("qwen35_9b/think-on/reasoning")
    prompt = chat(_QWEN, row["body"])
    ctx = dict(retire_key._TEXT_MEMO[prompt])
    assert ctx["prefill"] is True
    assert [m["role"] for m in ctx["messages"]] == ["system", "user"]
    assert prefill.CONTINUE_KEY not in ctx["kw"]
    assert "add_generation_prompt" not in ctx["kw"]
    ctx["preprocess"] = _ids
    assert retire_key.next_turn_lcp(ctx, _ids(prompt), [1]) is None
    assert retire_key.prompt_stable_lcp(ctx, _ids(prompt)) is None
    assert retire_key.predict_next_ids(ctx, None) is None

    normal = chat(_QWEN, {"messages": _HISTORY,
                          "chat_template_kwargs": {"enable_thinking": True}})
    nctx = dict(retire_key._TEXT_MEMO[normal])
    assert not nctx["prefill"]
    nctx["preprocess"] = _ids
    sys_prefix = retire_key.system_prefix_lcp(nctx, _ids(normal))
    assert sys_prefix and sys_prefix > 0
    assert retire_key.system_prefix_lcp(ctx, _ids(prompt)) == sys_prefix


def test_an_empty_reasoning_prefill_replaces_a_normal_render_of_its_text(chat):
    """A reasoning prefill with no reasoning renders the text of a normal
    thinking-on request, and the latest render of a text wins."""
    normal = chat(_QWEN, {"messages": _HISTORY,
                          "chat_template_kwargs": {"enable_thinking": True}})
    assert not retire_key._TEXT_MEMO[normal]["prefill"]
    body = _prefill({"role": "assistant", "content": "x"},
                    cont="reasoning_content",
                    chat_template_kwargs={"enable_thinking": True})
    assert chat(_QWEN, body) == normal
    assert retire_key._TEXT_MEMO[normal]["prefill"] is True


def test_add_generation_prompt_false_alone_registers_a_prefill_ctx(chat):
    row = _row("qwen35_9b/think-off/no-generation-prompt")
    prompt = chat(_QWEN, row["body"])
    ctx = retire_key._TEXT_MEMO[prompt]
    assert ctx["prefill"] is True and len(ctx["messages"]) == 2


@pytest.mark.parametrize("rid,in_thinking", [
    ("qwen35_9b/think-on/reasoning", True),
    ("qwen35_9b/think-on/content", False),
    ("gemma4_12b/think-on/reasoning", True),
    ("gemma4_12b/think-off/content", False),
])
def test_the_stream_splitter_starts_where_the_prefill_ends(chat, rid, in_thinking):
    row = _row(rid)
    processor, config = _processor_for(row["run"])
    msgs = [dict(m) for m in row["body"]["messages"]]
    cont_value = row["body"]["continue_final_message"]
    from gmlx.gen.continuation import handler_for, parse_continue, resolve

    cont = resolve(parse_continue(cont_value), msgs[-1],
                   handler_for(processor.chat_template))
    out = _OPENAI.apply_chat_template(
        processor, config, msgs, add_generation_prompt=False,
        **{prefill.CONTINUE_KEY: cont}, **row["body"]["chat_template_kwargs"])
    assert sp_chat._LAST_RENDERED_PROMPT.get() == out
    assert _expected(row, out)[1] == row["prompt"]
    assert _RS.ThinkingStreamState().in_thinking is in_thinking


@pytest.mark.parametrize("rid,start,end,opens", [
    ("qwen35_9b/think-on/reasoning", "<think>", "</think>", True),
    ("qwen35_9b/think-off/content", "<think>", "</think>", False),
    ("gemma4_12b/think-on/reasoning", "<|channel>thought", "<channel|>", True),
    ("gemma4_12b/think-on/content", "<|channel>thought", "<channel|>", False),
])
def test_a_reasoning_prefill_prompt_ends_inside_a_block(chat, rid, start, end,
                                                        opens):
    """The prompt gmlx renders ends inside a thinking block for a reasoning
    prefill, and outside one for a content prefill, so the budget criteria
    start in the right state."""
    from gmlx.gen.thinking_budget import prompt_opens_block

    row = _row(rid)
    prompt = _expected(row, chat(row["run"], row["body"]))[1]
    assert prompt_opens_block(_ids(prompt), _ids(start), _ids(end)) is opens


class _ByteTokenizer:
    """One id per byte, so a prompt's ids are its UTF-8 bytes."""

    def __init__(self, template):
        self.chat_template = template

    def encode(self, text, add_special_tokens=True):
        return list(text.encode())

    def decode(self, ids):
        return bytes(int(t) for t in ids).decode(errors="replace")


# Run -> the think markers the budget path reads from the request args.
# Qwen's are the server defaults. Gemma's must be configured, in the request
# or the server config: the plain criteria fall back to <think> without them.
_BUDGET_MARKERS = {
    _QWEN: (None, None, "<think>", "</think>"),
    "llama-server/gemma4_12b": ("<|channel>thought", "<channel|>",
                                "<|channel>thought", "<channel|>"),
}
_LONG_REASONING = "Count the letters first, then answer in one word."
_LONG_CONTENT = "Hello there, this answer is already well under way"


def _budget_prefill(chat, run, mode, budget):
    """The prefilled prompt's ids, the k reasoning tokens it ends with, the
    byte tokenizer, the gen args and the end marker ids. k is the byte count
    of the text after the open marker, and 0 for a content prefill."""
    last = ({"role": "assistant", "content": "",
             "reasoning_content": _LONG_REASONING} if mode == "reasoning"
            else {"role": "assistant", "content": _LONG_CONTENT})
    prompt = chat(run, {**_prefill(last),
                        "chat_template_kwargs": {"enable_thinking": True}})
    start_arg, end_arg, start, end = _BUDGET_MARKERS[run]
    if mode == "reasoning":
        tail = "\n" + _LONG_REASONING
        assert prompt.endswith(start + tail)
        k = len(tail.encode())
    else:
        assert prompt.endswith(_LONG_CONTENT)
        k = 0
    processor, _ = _processor_for(run)
    args = types.SimpleNamespace(
        thinking_budget=budget, thinking_start_token=start_arg,
        thinking_end_token=end_arg, thinking_budget_message=None,
        _kq_has_tools=False)
    return (_ids(prompt), k, _ByteTokenizer(processor.chat_template), args,
            _ids(end))


def _plain_criteria(tok, args, ids):
    import mlx.core as mx

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    sp_chat.install_thinking_budget_fix()
    rg = types.SimpleNamespace(tokenizer=tok)
    return gen.ResponseGenerator._make_thinking_budget_criteria(
        rg, args, mx.array([ids]))


def _mtp_hook(tok, args, ids):
    import mlx.core as mx

    from gmlx.serve.patches.mtp_thinking import _build_hook

    return _build_hook(types.SimpleNamespace(tokenizer=tok), args,
                       mx.array([ids]))


_THINK = ord("x")
_BUDGET_RUNS = list(_BUDGET_MARKERS)


@pytest.mark.parametrize("run", _BUDGET_RUNS)
@pytest.mark.parametrize("mode", ["reasoning", "content"])
def test_the_budget_count_of_a_prefilled_prompt(chat, run, mode):
    """prompt_reasoning_tokens on the prefilled ids gives the k tokens after
    the open marker for a reasoning prefill, k longer than the 16-id tail,
    and 0 for a content prefill."""
    from gmlx.gen.thinking_budget import prompt_reasoning_tokens

    ids, k, _tok, _args, end_ids = _budget_prefill(chat, run, mode, 0)
    start = _BUDGET_MARKERS[run][2]
    assert prompt_reasoning_tokens(ids, _ids(start), (end_ids,)) == k
    if mode == "reasoning":
        assert k > 16
    else:
        assert k == 0


@pytest.mark.parametrize("run", _BUDGET_RUNS)
@pytest.mark.parametrize("extra,through", [(2, 2), (0, 1)])
def test_a_prefill_budget_closes_on_the_plain_criteria(chat, run, extra,
                                                       through):
    """A budget of k + 2 on a reasoning prefill of k tokens forces the close
    after 2 generated tokens. A budget of k lets the one token prefill
    samples through, then forces the close."""
    ids, k, tok, args, end_ids = _budget_prefill(chat, run, "reasoning", 0)
    args.thinking_budget = k + extra
    crit = _plain_criteria(tok, args, ids)
    assert crit.thinking_token_count == k
    forced = None
    for _ in range(through):
        assert forced is None
        forced = crit(_THINK)
    assert forced is not None
    run_ids = [forced]
    while (forced := crit(forced)) is not None:
        run_ids.append(forced)
    assert run_ids[-len(end_ids):] == end_ids
    assert crit(_THINK) is None


@pytest.mark.parametrize("run", _BUDGET_RUNS)
@pytest.mark.parametrize("extra,through", [(2, 2), (0, 1)])
def test_a_prefill_budget_closes_on_the_mtp_hook(chat, run, extra, through):
    """The MTP hook seeds the same count. On rounds that emit one token,
    the loop observes it before it asks for a forced close, so the rows
    match the plain criteria."""
    ids, k, tok, args, end_ids = _budget_prefill(chat, run, "reasoning", 0)
    args.thinking_budget = k + extra
    hook = _mtp_hook(tok, args, ids)
    assert hook.count == k
    forced = None
    for _ in range(through):
        assert forced is None
        hook.observe(_THINK)
        forced = hook.take_forced()
    assert forced is not None
    assert forced[-len(end_ids):] == end_ids
    for t in forced:
        hook.observe(t)
    hook.observe(_THINK)
    assert hook.take_forced() is None


@pytest.mark.parametrize("run", _BUDGET_RUNS)
def test_a_prefill_budget_closes_after_an_mtp_block(chat, run):
    """The loop observes a whole accepted block before the round boundary.
    So with a budget of k + 2, a block of 4 thinking tokens runs 2 past the
    budget, and the close comes at the first boundary after it."""
    ids, k, tok, args, end_ids = _budget_prefill(chat, run, "reasoning", 0)
    args.thinking_budget = k + 2
    hook = _mtp_hook(tok, args, ids)
    hook.observe(_THINK)
    assert hook.take_forced() is None
    for _ in range(3):
        hook.observe(_THINK)
    assert hook.count == k + 4
    forced = hook.take_forced()
    assert forced is not None
    assert forced[-len(end_ids):] == end_ids


class _WrapperByteTokenizer(_ByteTokenizer):
    """A byte tokenizer that names gemma's think markers, as mlx-lm's
    TokenizerWrapper does for a vocab with the channel tokens."""

    think_start = "<|channel>thought"
    think_end = "<channel|>"
    think_start_tokens = tuple(think_start.encode())
    think_end_tokens = tuple(think_end.encode())


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "The plain criteria take the think markers from the args only, else "
    "<think>. Gemma names neither in its template's known pairs, so a "
    "gemma budget with no configured markers stays disarmed."))
def test_a_gemma_prefill_budget_arms_without_configured_markers(chat):
    run = "llama-server/gemma4_12b"
    ids, k, tok, args, _end = _budget_prefill(chat, run, "reasoning", 0)
    args.thinking_start_token = args.thinking_end_token = None
    args.thinking_budget = k
    crit = _plain_criteria(_WrapperByteTokenizer(tok.chat_template), args,
                           ids)
    assert crit.thinking_token_count == k


@pytest.mark.parametrize("run", _BUDGET_RUNS)
def test_a_content_prefill_does_not_arm_the_budget(chat, run):
    ids, _k, tok, args, _end = _budget_prefill(chat, run, "content", 1)
    crit = _plain_criteria(tok, args, ids)
    hook = _mtp_hook(tok, args, ids)
    for _ in range(3 * len(_LONG_CONTENT)):
        assert crit(_THINK) is None
        hook.observe(_THINK)
        assert hook.take_forced() is None
    assert crit.thinking_token_count == 0 and not crit.in_thinking
    assert hook.count == 0 and not hook.in_thinking


def test_the_render_key_never_reaches_the_stock_render(chat, monkeypatch):
    seen = []
    stock = _PROMPT_UTILS.get_chat_template

    def spy(processor, messages, add_generation_prompt, *a, **kw):
        seen.append(set(kw))
        return stock(processor, messages, add_generation_prompt, *a, **kw)

    monkeypatch.setattr(_PROMPT_UTILS, "get_chat_template", spy)
    chat(_QWEN, _row("qwen35_9b/think-on/reasoning")["body"])
    assert seen and not any(prefill.CONTINUE_KEY in k for k in seen)


def test_the_render_wrap_refuses_a_continuation_without_an_assistant(chat):
    processor, config = _processor_for(_QWEN)
    cont = Continuation(mode="content", content="x", reasoning="",
                        handler="qwen3-coder")
    with pytest.raises(RuntimeError):
        _OPENAI.apply_chat_template(processor, config, list(_HISTORY),
                                    **{prefill.CONTINUE_KEY: cont})


# The estimate renders a prefill body as the route does.
def _resident(monkeypatch, processor, config):
    import mlx.core as mx

    serving.register_resolved_models(build_config(
        {"models": {"m": {"path": "/abs/m.gguf"}}}))
    rg = types.SimpleNamespace(
        _preprocess_request=lambda prompt: {
            "input_ids": mx.array([_ids(prompt)])},
        kv_bits=None, apc_mode=None)
    entry = types.SimpleNamespace(
        response_generator=rg, apc_manager=None,
        model_cache={"model": None, "processor": processor, "config": config})
    # The serve fixture puts the pool back.
    _PKG._kq_residency_pool = types.SimpleNamespace(
        resident_entry=lambda p: entry)


@pytest.mark.parametrize("path", ["/v1/estimate", "/v1/chat/completions"])
def test_the_estimate_renders_the_prefill_as_the_route_does(chat, monkeypatch,
                                                            path):
    row = _row("qwen35_9b/think-on/reasoning-and-content")
    prompt = chat(_QWEN, row["body"])
    processor, config = _processor_for(_QWEN)
    _resident(monkeypatch, processor, config)
    seen = []
    stock = _PKG.apply_chat_template

    def spy(*a, **kw):
        seen.append(set(kw))
        return stock(*a, **kw)

    monkeypatch.setattr(_PKG, "apply_chat_template", spy)
    from fastapi.testclient import TestClient

    body = {"model": "m", **row["body"]}
    if path != "/v1/estimate":
        body["dry_run"] = True
    r = TestClient(_APP.app).post(path, json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["prompt_chars"] == len(prompt)
    assert out["prompt_tokens"] == len(_ids(prompt))
    assert seen and not any(prefill.CONTINUE_KEY in k for k in seen)


def test_the_estimate_answers_a_request_rule_with_its_message(chat, monkeypatch):
    processor, config = _processor_for(_QWEN)
    _resident(monkeypatch, processor, config)
    status, out = est.estimate_request({"model": "m", **_prefill(
        {"role": "assistant", "content": ""})})
    assert status == 400 and out["error"]["message"] == prefill.EMPTY


@pytest.mark.parametrize("body,message", [
    (_prefill(_C, cont="yes"), prefill.INVALID_VALUE),
    ({"messages": [*_HISTORY, _C], "continue_final_message": True},
     prefill.BOTH_TRUE),
    (_prefill({"role": "assistant", "content": ""}), prefill.EMPTY),
])
def test_the_estimate_of_a_model_not_resident_answers_a_request_rule(
        chat, body, message):
    serving.register_resolved_models(build_config(
        {"models": {"m": {"path": "/abs/m.gguf"}}}))
    _PKG._kq_residency_pool = types.SimpleNamespace(resident_entry=lambda p: None)
    status, out = est.estimate_request({"model": "m", **body})
    assert status == 400 and out["error"]["message"] == message


def test_every_template_kwargs_render_site_is_known():
    """Every caller that renders to_template_kwargs() output for a chat
    request pops the continuation through render_continued. A new caller
    must be added here and checked."""
    root = Path(prefill.__file__).resolve().parents[3]
    sites = set()
    for p in sorted((root / "gmlx").rglob("*.py")):
        for line in p.read_text().splitlines():
            if (re.search(r"\.to_template_kwargs\(\)", line)
                    and not line.lstrip().startswith("#")):
                sites.add(p.relative_to(root).as_posix())
    assert sites == {
        # Chat completions and Responses: renders through the wrapped
        # openai binding.
        "gmlx/_vendor/mlx_vlm/server/openai.py",
        # Messages: the wrapped anthropic binding; its request is never a
        # ChatRequest.
        "gmlx/_vendor/mlx_vlm/server/anthropic.py",
        # The estimate: render_continued over the package binding.
        "gmlx/serve/estimate.py",
        # The upsampler: the wrapped openai binding; its ChatRequest sets
        # neither field.
        "gmlx/serve/image_upsample.py",
    }
    src = (root / "gmlx/serve/estimate.py").read_text()
    assert "render_continued(pkg.apply_chat_template" in src


# Assistant aliases build their own prompt.
def _alias_route(monkeypatch):
    from gmlx.assistant import serve as aserve
    from fastapi.responses import JSONResponse

    turns = []

    async def _turn(state, alias_id, *args, **kwargs):
        turns.append(alias_id)
        return JSONResponse({"turn": alias_id})

    monkeypatch.setattr(aserve, "_assistant_completion", _turn)
    state = types.SimpleNamespace(
        aliases={"helper": (types.SimpleNamespace(model="m"), None)})
    aserve._wrap_chat_routes(_APP.app, state)
    return turns


@pytest.mark.parametrize("extra", [
    {"continue_final_message": True, "add_generation_prompt": False},
    {"add_generation_prompt": False},
    {"continue_final_message": None},
])
def test_an_assistant_alias_refuses_the_fields(chat, monkeypatch, extra):
    from fastapi.testclient import TestClient

    turns = _alias_route(monkeypatch)
    r = TestClient(_APP.app).post("/v1/chat/completions", json={
        "model": "helper", "messages": [*_HISTORY, _C], **extra})
    assert r.status_code == 400 and turns == []
    assert r.json()["error"]["message"] == (
        "continue_final_message and add_generation_prompt are not available "
        "for the assistant model 'helper', because its turns build their own "
        "prompt.")


def test_an_assistant_alias_with_tools_passes_the_fields(chat, monkeypatch):
    turns = _alias_route(monkeypatch)
    row = _row("qwen35_9b/think-off/content")
    tools = [{"type": "function", "function": {
        "name": "f", "parameters": {"type": "object", "properties": {}}}}]
    processor, config = _processor_for(_QWEN)
    monkeypatch.setattr(_OPENAI, "get_cached_model", lambda *a, **k: (
        None, processor, config))
    from fastapi.testclient import TestClient

    r = TestClient(_APP.app).post("/v1/chat/completions", json={
        "model": "helper", "max_tokens": 1, "tools": tools, **row["body"]})
    assert r.status_code == 200, r.text
    assert turns == []
    assert _RUNTIME.response_generator.calls[-1].endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\nHello")


@pytest.mark.parametrize("name", ["continue_final_message",
                                  "add_generation_prompt"])
def test_chat_template_kwargs_cannot_carry_the_fields(chat, name):
    body = {"messages": [*_HISTORY, _C], "chat_template_kwargs": {name: False}}
    assert name in chat(_QWEN, body, status=400)
