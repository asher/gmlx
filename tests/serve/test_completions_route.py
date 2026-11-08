#!/usr/bin/env python3
"""The gmlx /v1/completions route: prompt shapes and BOS, validation,
non-stream and SSE shapes, logprobs, echo, stop-sequence trimming, usage
accounting and the default max_tokens. CPU-only - the engine is a stub
ResponseGenerator whose tokenizer is a real GGUF one, no model load."""
from __future__ import annotations

import importlib
import json
import threading
import types

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import gmlx.serve.patches as sp  # noqa: E402
from gmlx._vendor.mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: E402
from gmlx.serve.bridge_vlm import _make_text_processor  # noqa: E402
from gmlx.upstream.single_bos_fix import install_single_bos_fix  # noqa: E402

_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_GEN = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
_DISPATCH = importlib.import_module("gmlx._vendor.mlx_vlm.generate.dispatch")
_UTILS = importlib.import_module("gmlx._vendor.mlx_vlm.utils")
_RUNTIME = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
_BOS = 0
_IMAGE = 200


def _gguf_tokenizer():
    from tests.upstream.test_single_bos_fix import _tokenizer

    return _tokenizer()


class _Tok:
    def __init__(self, text, finish=None, token=0, logprobs=None, top=None):
        self.text = text
        self.finish_reason = finish
        self.token_count = 1
        self.token = token
        self.logprobs = logprobs
        self.top_logprobs = top


class _FakeIter:
    def __init__(self, toks):
        self._it = iter(toks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        self.closed = True


class _FakeRG:
    """Stub ResponseGenerator: tokenizes with the engine's own preprocess
    on a GGUF tokenizer, records the prompt and args, yields fixed tokens."""

    _cpu_preprocess = _GEN.ResponseGenerator._cpu_preprocess
    _preprocess_request = _GEN.ResponseGenerator._preprocess_request

    def __init__(self, toks, model_type="llama", chat_template=None):
        self.toks = toks
        self.calls = []
        self.iters = []
        self.last_iter = None
        self.processor = _make_text_processor(TokenizerWrapper(_gguf_tokenizer()))
        self.processor.chat_template = chat_template
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(
            model_type=model_type, image_token_index=None,
            image_token_id=_IMAGE))
        self._tokenizer_lock = threading.Lock()
        self.fail_at = None

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise _GEN.PromptTooLongError("prompt is too long for the memory")
        self.calls.append((prompt, args))
        ctx = types.SimpleNamespace(prompt_tokens=len(prompt), uid=len(self.calls))
        self.last_iter = _FakeIter(self.toks)
        self.iters.append(self.last_iter)
        return ctx, self.last_iter


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    from gmlx.serve.patches import _common as sp_common
    saved_routes = sp_common._snapshot_routes(_APP.app)
    saved_gcm = _APP.get_cached_model
    saved_rg = _RUNTIME.response_generator
    for module in (_GEN, _DISPATCH):
        monkeypatch.setattr(module, "prepare_inputs", module.prepare_inputs)
    monkeypatch.setattr(_UTILS, "process_inputs", _UTILS.process_inputs)
    install_single_bos_fix()
    monkeypatch.delenv("TOP_LOGPROBS_K", raising=False)
    monkeypatch.setenv("GMLX_QUEUE_DEPTH_CAP", "0")
    yield
    sp_common._restore_routes(_APP.app, saved_routes)
    _APP.get_cached_model = saved_gcm
    _RUNTIME.response_generator = saved_rg


_LP_TOKS = [_Tok("Hello", token=40, logprobs=-0.5, top=[(40, -0.5), (41, -1.5), (42, -2.5)]),
            _Tok(" wor", token=41, logprobs=-1.0, top=[(43, -0.25), (41, -1.0), (44, -3.0)]),
            _Tok("ld END tail", "stop", token=1, logprobs=float("-inf"),
                 top=[(45, -0.1), (46, -0.2), (47, -0.3)])]


def _client(toks=None, **kw):
    from fastapi.testclient import TestClient

    rg = _FakeRG(toks if toks is not None
                 else [_Tok("Hello"), _Tok(" wor"), _Tok("ld END tail",
                                                         "stop")], **kw)
    _APP.get_cached_model = lambda mid, *a, **k: (None, None, None)
    _RUNTIME.response_generator = rg
    sp.install_completions_route()
    return TestClient(_APP.app), rg


def _encode(text, special=True):
    return list(_gguf_tokenizer().encode(text, add_special_tokens=special))


def test_route_registers_both_paths_idempotent():
    sp.install_completions_route()
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert "/v1/completions" in paths and "/completions" in paths
    n = len(_APP.app.router.routes)
    sp.install_completions_route()
    assert len(_APP.app.router.routes) == n


def test_non_stream_text_completion_shape():
    client, rg = _client()
    r = client.post("/v1/completions",
                    json={"model": "m", "prompt": "hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    choice = body["choices"][0]
    assert choice["text"] == "Hello world END tail"
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["logprobs"] is None
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 3,
                             "total_tokens": 6}
    # The raw prompt reaches the engine as its ids, with no chat template.
    assert rg.calls[0][0] == _encode("hi")
    assert rg.last_iter.closed


@pytest.mark.parametrize("model_type,template", [
    ("llama", None), ("gemma4", "t"), ("gemma4_text", None)])
def test_a_string_prompt_gets_one_bos(model_type, template):
    client, rg = _client(model_type=model_type, chat_template=template)
    r = client.post("/v1/completions", json={"model": "m", "prompt": "hi"})
    assert r.status_code == 200, r.text
    assert rg.calls[0][0] == [_BOS] + _encode("hi", special=False)


def test_add_special_tokens_false_gives_no_bos():
    client, rg = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "hi", "add_special_tokens": False})
    assert r.status_code == 200, r.text
    assert rg.calls[0][0] == _encode("hi", special=False)


@pytest.mark.parametrize("flag", [True, False, None])
def test_an_id_prompt_reaches_the_engine_as_sent(flag):
    client, rg = _client()
    body = {"model": "m", "prompt": [5, 6, 7]}
    if flag is not None:
        body["add_special_tokens"] = flag
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 200, r.text
    assert rg.calls[0][0] == [5, 6, 7]
    assert r.json()["usage"]["prompt_tokens"] == 3


def test_lists_of_prompts_are_one_choice_each():
    client, rg = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": [[5, 6], [7, 8, 9]]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [c["index"] for c in body["choices"]] == [0, 1]
    assert [c[0] for c in rg.calls] == [[5, 6], [7, 8, 9]]
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 6,
                             "total_tokens": 11}
    r = client.post("/v1/completions", json={"model": "m", "prompt": ["a", "b"]})
    assert r.status_code == 200, r.text
    assert [c[0] for c in rg.calls[2:]] == [_encode("a"), _encode("b")]
    assert len(r.json()["choices"]) == 2


def test_a_failed_row_cancels_the_rows_already_queued():
    client, rg = _client()
    rg.fail_at = 1
    r = client.post("/v1/completions", json={"model": "m", "prompt": [[5], [6]]})
    assert r.status_code == 400
    assert "too long" in r.json()["detail"]
    assert len(rg.iters) == 1 and rg.iters[0].closed


def test_non_stream_stop_trims_and_marks_stop():
    client, rg = _client()
    r = client.post("/v1/completions",
                    json={"model": "m", "prompt": "hi", "stop": "END"})
    choice = r.json()["choices"][0]
    assert choice["text"] == "Hello world "
    assert choice["finish_reason"] == "stop"


def test_non_stream_length_finish_reason():
    client, _ = _client([_Tok("a"), _Tok("b", "length")])
    r = client.post("/v1/completions", json={"model": "m", "prompt": "p"})
    assert r.json()["choices"][0]["finish_reason"] == "length"


def test_default_max_tokens_matches_chat_default():
    client, rg = _client()
    client.post("/v1/completions", json={"model": "m", "prompt": "p"})
    args = rg.calls[0][1]
    assert args.max_tokens == _GEN.get_server_max_tokens()
    # sampling params pass through _build_gen_args when set
    client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "max_tokens": 5, "temperature": 0.2,
        "top_p": 0.9, "seed": 42})
    args = rg.calls[1][1]
    assert (args.max_tokens, args.temperature, args.top_p, args.seed) == \
        (5, 0.2, 0.9, 42)
    # token logprobs are collected only when the request asks
    assert args.logprobs is False


@pytest.mark.parametrize("payload,fragment", [
    ({"prompt": ""}, "must not be empty"),
    ({"prompt": []}, "must not be empty"),
    ({"prompt": [[1], []]}, "must not be empty"),
    ({"prompt": ["a", ""]}, "must not be empty"),
    ({"prompt": [-1, 2]}, "out of range"),
    ({"prompt": [261]}, "token id 261 is out of range"),
    ({"prompt": [1, True]}, "not booleans"),
    ({"prompt": [1, 2.0]}, "not floats"),
    ({"prompt": [[[1]]]}, "nest at most two levels"),
    ({"prompt": [1, "a"]}, "mixes text and token ids"),
    ({"prompt": [[1], "a"]}, "mixes text and token ids"),
    ({"prompt": {"a": 1}}, "a list of token ids"),
    ({"prompt": [5, _IMAGE]}, "media placeholder"),
    ({"prompt": [[1]] * 65}, "65 prompts"),
    ({"prompt": "p", "logprobs": True}, "not a boolean"),
    ({"prompt": "p", "logprobs": -1}, "0 or more"),
    ({"prompt": "p", "add_special_tokens": "yes"}, "true or false"),
    ({"prompt": "p", "max_tokens": 0}, "needs 'echo'"),
    ({"prompt": "p", "stream": True, "echo": True, "logprobs": 0}, "does not stream"),
    ({"prompt": ["a", "b"], "stream": True}, "does not stream"),
    ({"prompt": "p", "n": 2}, "'n' greater than 1"),
    ({"prompt": "p", "suffix": "tail"}, "'suffix'"),
    ({"prompt": "p", "best_of": 4}, "'best_of' greater than 1"),
])
def test_rejected_shapes_get_400(payload, fragment):
    client, rg = _client()
    r = client.post("/v1/completions", json={"model": "m", **payload})
    assert r.status_code == 400, r.text
    assert fragment in r.json()["detail"]
    assert not rg.calls


def test_the_prompt_limit_is_the_queue_cap_when_on(monkeypatch):
    monkeypatch.setenv("GMLX_QUEUE_DEPTH_CAP", "3")
    client, _ = _client()
    r = client.post("/v1/completions", json={"model": "m", "prompt": [[1]] * 4})
    assert r.status_code == 400 and "above the limit 3" in r.json()["detail"]


def test_tolerated_singular_values_pass():
    client, _ = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "n": 1, "best_of": 1, "echo": False,
        "suffix": ""})
    assert r.status_code == 200


def test_logprobs_at_the_cap_pass_and_above_it_are_refused(monkeypatch):
    client, rg = _client(_LP_TOKS)
    r = client.post("/v1/completions", json={"model": "m", "prompt": "p",
                                             "logprobs": 20})
    assert r.status_code == 200, r.text
    assert (rg.calls[0][1].logprobs, rg.calls[0][1].top_logprobs) == (True, 20)
    r = client.post("/v1/completions", json={"model": "m", "prompt": "p",
                                             "logprobs": 21})
    assert r.status_code == 400
    assert r.json()["detail"] == ("logprobs is 21, above the server cap 20 "
                                  "(TOP_LOGPROBS_K)")
    monkeypatch.setenv("TOP_LOGPROBS_K", "2")
    r = client.post("/v1/completions", json={"model": "m", "prompt": "p",
                                             "logprobs": 3})
    assert r.status_code == 400 and "cap 2" in r.json()["detail"]


def test_the_logprobs_shape():
    client, rg = _client(_LP_TOKS)
    r = client.post("/v1/completions", json={"model": "m", "prompt": "p",
                                             "logprobs": 1})
    assert r.status_code == 200, r.text
    lp = r.json()["choices"][0]["logprobs"]
    tok = rg.processor.tokenizer
    names = [tok.decode([t]) for t in (40, 41, 1)]
    assert lp["tokens"] == names
    assert lp["token_logprobs"] == [-0.5, -1.0, -9999.0]
    # The top N plus the chosen token.
    assert lp["top_logprobs"][0] == {names[0]: -0.5}
    assert lp["top_logprobs"][1] == {names[1]: -1.0, tok.decode([43]): -0.25}
    assert lp["top_logprobs"][2] == {names[2]: -9999.0, tok.decode([45]): -0.1}
    assert lp["text_offset"] == [0, len(names[0]), len(names[0]) + len(names[1])]


def test_return_tokens_as_token_ids_names_every_token():
    client, _ = _client(_LP_TOKS)
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "logprobs": 0,
        "return_tokens_as_token_ids": True})
    lp = r.json()["choices"][0]["logprobs"]
    assert lp["tokens"] == ["token_id:40", "token_id:41", "token_id:1"]
    assert lp["top_logprobs"][1] == {"token_id:41": -1.0}
    assert lp["text_offset"] == [0, 11, 22]


def test_a_stop_string_trims_the_text_not_the_entries():
    client, _ = _client(_LP_TOKS)
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "logprobs": 0, "stop": "END"})
    choice = r.json()["choices"][0]
    assert choice["text"] == "Hello world "
    assert len(choice["logprobs"]["tokens"]) == 3


def test_echo_puts_the_prompt_before_the_completion():
    client, rg = _client()
    r = client.post("/v1/completions", json={"model": "m", "prompt": "hi",
                                             "echo": True})
    assert r.json()["choices"][0]["text"] == "hiHello world END tail"
    ids = _encode("Hello")
    r = client.post("/v1/completions", json={"model": "m", "prompt": ids,
                                             "echo": True})
    text = rg.processor.tokenizer.decode(ids)
    assert text.startswith("<s>")
    assert r.json()["choices"][0]["text"] == text + "Hello world END tail"


def test_max_tokens_0_with_echo_returns_the_prompt_alone():
    client, rg = _client()
    r = client.post("/v1/completions", json={"model": "m", "prompt": [5, 6],
                                             "echo": True, "max_tokens": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["text"] == rg.processor.tokenizer.decode([5, 6])
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 0,
                             "total_tokens": 2}
    assert not rg.calls


def _sse_events(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[len("data: "):]
        events.append("[DONE]" if payload == "[DONE]"
                      else json.loads(payload))
    return events


def test_stream_chunks_and_done():
    client, rg = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events if e != "[DONE]"]
    assert all(c["object"] == "text_completion" for c in chunks)
    text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
    assert text == "Hello world END tail"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert rg.last_iter.closed


def test_stream_echo_puts_the_prompt_in_the_first_chunk():
    client, _ = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "hi", "stream": True, "echo": True})
    chunks = [e for e in _sse_events(r.text) if e != "[DONE]"]
    assert chunks[0]["choices"][0]["text"] == "hi"
    text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
    assert text == "hiHello world END tail"


def test_stream_stop_trims_and_ends_early():
    client, rg = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "stream": True, "stop": ["END"]})
    events = _sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events if e != "[DONE]"]
    text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
    assert text == "Hello world "
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert rg.last_iter.closed


def test_stream_entries_follow_the_text_with_a_stop_string():
    client, _ = _client(_LP_TOKS)
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "stream": True, "logprobs": 0,
        "stop": ["END"], "return_tokens_as_token_ids": True})
    chunks = [e for e in _sse_events(r.text) if e != "[DONE]"]
    entries = [c["choices"][0]["logprobs"] for c in chunks]
    tokens = [t for e in entries if e for t in e["tokens"]]
    offsets = [o for e in entries if e for o in e["text_offset"]]
    assert tokens == ["token_id:40", "token_id:41", "token_id:1"]
    assert offsets == [0, 11, 22]
    for c, e in zip(chunks, entries):
        if c["choices"][0]["text"]:
            assert e["tokens"]


def test_stream_include_usage_emits_usage_chunk():
    client, _ = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "stream": True,
        "stream_options": {"include_usage": True}})
    events = _sse_events(r.text)
    usage_chunks = [e for e in events
                    if e != "[DONE]" and e.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"] == {
        "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert usage_chunks[0]["choices"] == []


def test_ignored_fields_warn_on_completions(caplog):
    import logging

    client, _ = _client()
    with caplog.at_level(logging.WARNING,
                         logger="gmlx.serve.patches.api_contract"):
        r = client.post("/v1/completions", json={
            "model": "m", "prompt": "p", "logprobs": 3, "user": "u",
            "return_tokens_as_token_ids": True, "add_special_tokens": True,
            "prompt_logprobs": 2})
    assert r.status_code == 200
    warns = [m for m in caplog.messages if "ignoring unsupported" in m]
    assert len(warns) == 1
    named = warns[0].split("parameter(s): ", 1)[1].split(" (", 1)[0]
    assert named.split(", ") == ["prompt_logprobs", "user"]


def test_prompt_too_long_maps_to_400():
    client, rg = _client()
    rg.fail_at = 0
    r = client.post("/v1/completions", json={"model": "m", "prompt": "p"})
    assert r.status_code == 400
    assert "too long" in r.json()["detail"]


def test_a_diffusion_model_refuses_logprobs(monkeypatch):
    import gmlx.gen.diffusion as diffusion

    monkeypatch.setattr(diffusion, "is_diffusion_model", lambda model: True)
    client, rg = _client()
    r = client.post("/v1/completions", json={"model": "m", "prompt": "p",
                                             "logprobs": 0})
    assert r.status_code == 400 and "diffusion" in r.json()["detail"]
    assert not rg.calls


# Scoring: echo with logprobs. The engine job is stubbed; the reader's
# entries are fixed, so these tests read the route, not the model.

class _Hold:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


def _entry(i, k):
    lp = -0.1 * i
    top = [[40 + j for j in range(k)], [lp - 0.01 * j for j in range(k)]] if k else [None, None]
    return (lp, top[0], top[1])


class _Reader:
    def __init__(self, refusal=None, raises=None):
        self.built = True
        self.refusal = refusal
        self.raises = raises
        self.scored = []

    def score(self, ids, k, rows_scope=None, should_stop=None):
        if self.raises is not None:
            raise self.raises
        self.scored.append(list(ids))
        yield
        return [_entry(i, k) for i in range(1, len(ids))]


@pytest.fixture
def scoring(monkeypatch):
    """The scoring route's engine side stubbed: a fixed reader, a
    run_on_engine that drives the job's steps on this thread, and a
    preflight that records its calls."""
    from gmlx.gen import score_reader
    from gmlx.serve import engine_jobs, mem_preflight

    state = types.SimpleNamespace(reader=_Reader(), runs=[], priced=[],
                                  kept=None)
    monkeypatch.setattr(score_reader, "reader_of", lambda rg: state.reader)
    monkeypatch.setattr(score_reader, "reader_for", lambda rg: state.kept)

    def run(rg, job, *, stop, stepwise=False, **kw):
        state.runs.append(kw)
        steps = job(rg, stop.is_set)
        try:
            while True:
                next(steps)
        except StopIteration as e:
            return e.value

    monkeypatch.setattr(engine_jobs, "run_on_engine", run)
    monkeypatch.setattr(mem_preflight, "preflight_scoring",
                        lambda rg, n, reader=None: state.priced.append(n))
    return state


def test_echo_with_logprobs_scores_the_prompt(scoring):
    client, rg = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": [5, 6, 7], "echo": True, "logprobs": 1,
        "max_tokens": 0, "return_tokens_as_token_ids": True})
    assert r.status_code == 200, r.text
    body = r.json()
    choice = body["choices"][0]
    lp = choice["logprobs"]
    assert lp["tokens"] == ["token_id:5", "token_id:6", "token_id:7"]
    assert lp["token_logprobs"] == [None, -0.1, -0.2]
    assert lp["top_logprobs"] == [None, {"token_id:6": -0.1, "token_id:40": -0.1},
                                  {"token_id:7": -0.2, "token_id:40": -0.2}]
    assert lp["text_offset"] == [0, 10, 20]
    assert choice["finish_reason"] == "length"
    assert choice["text"] == rg.processor.tokenizer.decode([5, 6, 7])
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 0,
                             "total_tokens": 3}
    assert scoring.reader.scored == [[5, 6, 7]]
    assert scoring.priced == [3]
    assert not rg.calls


def test_a_scored_string_prompt_names_its_tokens(scoring):
    client, rg = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "hi", "echo": True, "logprobs": 0,
        "max_tokens": 0})
    assert r.status_code == 200, r.text
    ids = [_BOS] + _encode("hi", special=False)
    assert scoring.reader.scored == [ids]
    lp = r.json()["choices"][0]["logprobs"]
    tok = rg.processor.tokenizer
    assert lp["tokens"] == [tok.decode([t]) for t in ids]
    assert lp["top_logprobs"][1] == {tok.decode([ids[1]]): -0.1}


def test_scoring_then_generating_continues_the_entries(scoring):
    client, rg = _client(_LP_TOKS)
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": [5, 6], "echo": True, "logprobs": 0,
        "max_tokens": 1, "return_tokens_as_token_ids": True})
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    lp = choice["logprobs"]
    assert lp["tokens"] == ["token_id:5", "token_id:6", "token_id:40",
                            "token_id:41", "token_id:1"]
    assert lp["token_logprobs"][:3] == [None, -0.1, -0.5]
    assert lp["text_offset"] == [0, 10, 20, 31, 42]
    assert choice["text"].startswith(rg.processor.tokenizer.decode([5, 6]))
    assert rg.calls[0][0] == [5, 6]


def test_a_list_of_prompts_is_scored_in_one_job(scoring):
    client, _ = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": [[5, 6], [7, 8, 9]], "echo": True,
        "logprobs": 0, "max_tokens": 0})
    assert r.status_code == 200, r.text
    assert len(scoring.runs) == 1
    assert scoring.reader.scored == [[5, 6], [7, 8, 9]]
    assert scoring.priced == [3]
    lps = [c["logprobs"]["token_logprobs"] for c in r.json()["choices"]]
    assert lps == [[None, -0.1], [None, -0.1, -0.2]]


def test_a_one_token_prompt_scores_nothing(scoring):
    client, _ = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": [5], "echo": True, "logprobs": 0,
        "max_tokens": 0})
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["logprobs"]["token_logprobs"] == [None]


def test_an_answer_over_the_entry_cap_is_refused(scoring, monkeypatch):
    from gmlx.serve.patches import completions

    monkeypatch.setattr(completions, "ENTRY_CAP", 7)
    client, _ = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": [5, 6, 7, 8], "echo": True, "logprobs": 1,
        "max_tokens": 0})
    assert r.status_code == 400
    assert "4 prompt positions" in r.json()["detail"]
    assert not scoring.runs
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": [5, 6, 7, 8], "echo": True, "logprobs": 0,
        "max_tokens": 0})
    assert r.status_code == 200, r.text


def test_the_logprobs_cap_holds_on_a_scoring_request(scoring):
    client, _ = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": [5, 6], "echo": True, "logprobs": 20,
        "max_tokens": 0})
    assert r.status_code == 200, r.text
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": [5, 6], "echo": True, "logprobs": 21,
        "max_tokens": 0})
    assert r.status_code == 400 and "cap 20" in r.json()["detail"]


def test_a_refused_model_gets_a_400_that_names_it(scoring):
    from gmlx.gen import score_reader

    scoring.reader = _Reader(raises=score_reader.ScoringRefused(
        score_reader.EMBEDS_DIFFER))
    client, _ = _client()
    body = {"model": "m", "prompt": [5, 6], "echo": True, "logprobs": 0,
            "max_tokens": 0}
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 400
    assert r.json()["detail"].startswith("model 'm' reads token ids")
    # A kept refusal answers before the request queues.
    scoring.kept = _Reader(refusal=score_reader.NO_COLUMNS)
    scoring.runs.clear()
    scoring.priced.clear()
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 400
    assert r.json()["detail"] == score_reader.NO_COLUMNS
    assert not scoring.runs and not scoring.priced


def test_scoring_failures_map_to_http_errors(scoring, monkeypatch):
    from gmlx.serve import engine_jobs, mem_preflight

    client, _ = _client()
    body = {"model": "m", "prompt": [5, 6], "echo": True, "logprobs": 0,
            "max_tokens": 0}

    def too_long(rg, n, reader=None):
        raise _GEN.PromptTooLongError("prompt is too long for the memory")

    monkeypatch.setattr(mem_preflight, "preflight_scoring", too_long)
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 400 and "too long" in r.json()["detail"]
    assert not scoring.runs
    monkeypatch.setattr(mem_preflight, "preflight_scoring",
                        lambda rg, n, reader=None: None)

    def timeout(rg, job, **kw):
        raise engine_jobs.JobTimeout("the scoring job did not finish")

    monkeypatch.setattr(engine_jobs, "run_on_engine", timeout)
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 504
    r = client.post("/v1/completions", json={**body, "lora": "x"})
    assert r.status_code == 400


def test_a_diffusion_model_refuses_scoring(scoring, monkeypatch):
    import gmlx.gen.diffusion as diffusion

    monkeypatch.setattr(diffusion, "is_diffusion_model", lambda model: True)
    client, _ = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": [5, 6], "echo": True, "logprobs": 0,
        "max_tokens": 0})
    assert r.status_code == 400 and "diffusion" in r.json()["detail"]
    assert not scoring.runs


def test_scoring_requests_take_turns_and_a_leaver_gets_499(scoring, monkeypatch):
    """Three scoring requests on one engine over one event loop. The first
    runs; the second waits on the loop, counted in the queue depth, with
    no job started; the third disconnects while waiting and gets 499; the
    second runs when the first returns. Each request holds its model from
    before the wait to the end, the leaver included."""
    import asyncio

    import httpx

    import gmlx.serve.queue_cap as qc
    from gmlx.serve import engine_jobs, job_route

    _client()
    rg = _RUNTIME.response_generator
    hold = _Hold()
    _RUNTIME.response_generator = types.SimpleNamespace(_rg=rg, _hold=hold)
    monkeypatch.setattr(qc, "_GEN_REF", lambda: None)
    started, release = [], {5: threading.Event(), 6: threading.Event()}
    first_running = threading.Event()

    def run(engine, job, *, stop, **kw):
        steps = job(engine, stop.is_set)
        first = None
        try:
            while True:
                next(steps)
                if first is None:
                    first = scoring.reader.scored[-1][0]
                    started.append(first)
                    if first == 5:
                        first_running.set()
                    release[first].wait()
        except StopIteration as e:
            return e.value

    monkeypatch.setattr(engine_jobs, "run_on_engine", run)
    queued: list = []

    class SignalTurn(job_route._Turn):
        def __init__(self):
            super().__init__()
            outer = self

            class Lock(asyncio.Lock):
                async def acquire(self):
                    if self.locked():
                        queued.append(outer.waiters)
                        signal.set()
                    return await super().acquire()

            self.lock = Lock()

    monkeypatch.setattr(job_route, "_Turn", SignalTurn)
    leave = {}

    async def watch(http_request, stop):
        gone = leave.get(http_request.headers.get("x-request"))
        try:
            if gone is not None:
                await gone.wait()
                stop.set()
                return
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass

    monkeypatch.setattr(job_route, "watch_disconnect", watch)
    signal = None

    async def main():
        nonlocal signal
        signal = asyncio.Event()
        leave["c"] = asyncio.Event()
        transport = httpx.ASGITransport(app=_APP.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            def post(first, name):
                return asyncio.create_task(client.post(
                    "/v1/completions", headers={"x-request": name}, json={
                        "model": "m", "prompt": [first, 7], "echo": True,
                        "logprobs": 0, "max_tokens": 0}))

            a = post(5, "a")
            running = asyncio.ensure_future(asyncio.to_thread(first_running.wait))
            await asyncio.wait({a, running}, return_when=asyncio.FIRST_COMPLETED)
            assert first_running.is_set(), "the first request ran no job"
            b = post(6, "b")
            await signal.wait()
            signal.clear()
            assert job_route.scoring_waiters(rg) == 1
            assert qc._waiting_depth(rg) == 1
            assert started == [5]
            c = post(6, "c")
            await signal.wait()
            assert job_route.scoring_waiters(rg) == 2
            assert hold.released == 0
            leave["c"].set()
            rc = await c
            assert rc.status_code == 499
            assert job_route.scoring_waiters(rg) == 1
            assert hold.released == 1
            release[5].set()
            release[6].set()
            ra, rb = await asyncio.gather(a, b)
            return ra.status_code, rb.status_code

    async def guarded():
        try:
            return await main()
        finally:
            # A failed check must not leave a thread waiting at shutdown.
            first_running.set()
            for event in release.values():
                event.set()

    codes = asyncio.run(guarded())
    assert codes == (200, 200)
    assert started == [5, 6]
    assert queued == [1, 2]
    assert hold.released == 3
    assert job_route._TURNS == {}
