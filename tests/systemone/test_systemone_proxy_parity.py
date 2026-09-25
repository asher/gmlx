"""gmlx's /v1/systemone bodies match the vLLM proxy's for the same reads.

Both sides share one word-level fake tokenizer and one read table: a pure
function of the prompt ids, the seed canvas and the position, so the
proxy's threaded reads and gmlx's batched reads get the same values. The
proxy runs through its /v1/systemone handler with its upstream calls
patched; gmlx runs through decide() and the contract helpers the route
uses. Bodies are compared as JSON with timing values and the engine name
removed."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
import types

import pytest

import _systemone_proxy_ref as proxy
from gmlx.systemone import (
    Limits,
    ReadResult,
    SampleRead,
    SlotRead,
    TemplateResolver,
    build_canvas,
    decide,
    jev_answers,
    jev_schema,
    jev_state,
    label_id_union,
    parse_seed,
    usage,
)

CANVAS = 64
VOCAB = 262144
MODEL = "dgemma-test"

_SPECIALS = ("<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>")
_TOKEN_RE = re.compile(
    "|".join(re.escape(s) for s in _SPECIALS) + r"|\n| ?\w+| ?[^\w\s]| "
)
_FIXED_IDS = {"<bos>": 2, "<|channel>": 100, "<channel|>": 101, "<|turn>": 105,
              "<turn|>": 106, "\n": 107, "<|think|>": 98}


class FakeTokenizer:
    """Word-level tokenizer with ids that depend only on the token text."""

    def __init__(self):
        self.text_of = {v: k for k, v in _FIXED_IDS.items()}
        self.lock = threading.Lock()

    def token_id(self, piece: str) -> int:
        tid = _FIXED_IDS.get(piece)
        if tid is None:
            tid = 1000 + int(hashlib.sha1(piece.encode()).hexdigest()[:12], 16)
        with self.lock:
            prev = self.text_of.setdefault(tid, piece)
        assert prev == piece, f"id collision: {prev!r} {piece!r}"
        return tid

    def encode(self, text, add_special_tokens=False):
        assert "".join(_TOKEN_RE.findall(text)) == text, text
        return [self.token_id(p) for p in _TOKEN_RE.findall(text)]

    def decode(self, ids):
        return "".join(self.text_of[int(i)] for i in ids)

    def render(self, messages, enable_thinking=False):
        out = "<bos>"
        for i, m in enumerate(messages):
            out += f"<|turn>{m['role']}\n"
            if i == 0 and m["role"] == "system" and enable_thinking:
                out += "<|think|>\n"
            out += m["content"] + "<turn|>\n"
        return out + "<|turn>model\n"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            enable_thinking=False):
        assert tokenize and add_generation_prompt
        return self.encode(self.render(messages, enable_thinking))


def _unit(*parts) -> float:
    """A value in [-1, 1) from the content of ``parts``."""
    h = hashlib.sha256(json.dumps(parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**63 - 1.0


class Table:
    """The read and thought table both sides answer from."""

    def __init__(self, tok, bias, noise, distractors):
        self.tok = tok
        self.bias = bias
        self.noise = noise
        self.distractors = [tok.token_id(d) for d in distractors]

    def top(self, prompt, canvas, pos, allowed, constrained):
        """The logprobs the engine reports at ``pos``: the whole allowed set
        normalized over it when constrained, else the argmax and the
        allowed ids with logprobs over the full candidate vocabulary."""
        vocab = list(allowed) if constrained else sorted(
            set(allowed) | set(self.distractors) | {canvas[pos]})
        key = hashlib.sha256(
            json.dumps([list(prompt), list(canvas), pos]).encode()).hexdigest()
        logits = {
            i: self.bias.get(self.tok.text_of.get(i), 0.0)
            + self.noise * _unit(key, i)
            for i in vocab
        }
        m = max(logits.values())
        lse = m + math.log(sum(math.exp(v - m) for v in logits.values()))
        lp = {i: v - lse for i, v in logits.items()}
        argmax = max(vocab, key=lambda i: lp[i])
        order = [argmax] + [i for i in allowed if i != argmax]
        return argmax, [(i, lp[i]) for i in order]

    def thought(self, prompt, budget):
        pool = self.tok.encode(" the user seems upset about a late refund")
        n = min(budget, 6)
        ids = [pool[int((_unit(list(prompt), k) + 1) * 1e6) % len(pool)]
               for k in range(n)]
        return ids, n < budget


class FakeCache:
    def __init__(self, ids):
        self.ids = list(ids)
        self.extends = 0

    def extend(self, ids):
        self.ids.extend(ids)
        self.extends += 1


class FakeEngine:
    """gmlx's ReadEngine over the table."""

    def __init__(self, table):
        self.table = table
        self.prefills = []

    def prefill(self, ids):
        self.prefills.append(list(ids))
        return FakeCache(ids)

    def read(self, prompt, req):
        allowed = label_id_union(req.slots)
        samples = []
        for seed in req.seeds:
            canvas = build_canvas(req.template, req.slots, req.width, seed, VOCAB)
            reads = []
            for s in req.slots:
                argmax, pairs = self.table.top(prompt.ids, canvas, s.pos, allowed,
                                               req.constrained)
                reads.append(SlotRead(argmax_id=argmax, top=dict(pairs)))
            samples.append(SampleRead(seed=seed, canvas_in=tuple(canvas),
                                      slots=tuple(reads)))
        return ReadResult(samples=tuple(samples), prompt_len=len(prompt.ids),
                          timing_ms={})

    def think(self, prompt_ids, budget, *, stop_id, canvas_width):
        ids, closed = self.table.thought(prompt_ids, budget)
        return ids, {"tokens": len(ids), "closed": closed, "ms": 0.0}


def _entry(tid, lp):
    return {"token": f"token_id:{tid}", "logprob": lp}


def _patch_upstream(monkeypatch, tok, table):
    """Answer the proxy's chat and completions calls from the table."""

    def rows(prompt, body):
        xargs = body["vllm_xargs"]
        canvas = xargs["diffusion_seed_canvas"]
        allowed = body["logprob_token_ids"]
        constrained = bool(xargs.get("diffusion_constrained"))
        return [table.top(prompt, canvas, pos, allowed, constrained)
                for pos in range(len(canvas))]

    def upstream_chat(body, timeout=600):
        thinking = body["chat_template_kwargs"]["enable_thinking"]
        prompt = tok.apply_chat_template(body["messages"], enable_thinking=thinking)
        content = [{"token": f"token_id:{a}", "logprob": pairs[0][1],
                    "top_logprobs": [_entry(i, v) for i, v in pairs]}
                   for a, pairs in rows(prompt, body)]
        return {"choices": [{"logprobs": {"content": content}}],
                "usage": {"prompt_tokens": len(prompt)}}

    def upstream_completions(body, timeout=600):
        prompt = body["prompt"]
        if "vllm_xargs" not in body:
            assert body["stop_token_ids"] == proxy.THOUGHT_CLOSE
            ids, closed = table.thought(prompt, body["max_tokens"])
            ids = ids + (proxy.THOUGHT_CLOSE if closed else [])
            return {"choices": [{"logprobs": {
                "tokens": [f"token_id:{i}" for i in ids]}}]}
        top = [{f"token_id:{i}": v for i, v in pairs}
               for _a, pairs in rows(prompt, body)]
        return {"choices": [{"logprobs": {"top_logprobs": top}}],
                "usage": {"prompt_tokens": len(prompt)}}

    monkeypatch.setattr(proxy, "upstream_chat", upstream_chat)
    monkeypatch.setattr(proxy, "upstream_completions", upstream_completions)


def _proxy_body(body):
    handler = proxy.Handler.__new__(proxy.Handler)
    out = {}

    def capture(code, obj):
        out["code"], out["obj"] = code, obj

    handler._json = capture
    handler._systemone(copy.deepcopy(body), [])
    assert out["code"] == 200, out["obj"]
    return out["obj"]


def _gmlx_body(body, tok, engine, constrained):
    """The body gmlx/serve/patches/systemone.py assembles, from the same
    calls."""
    body = copy.deepcopy(body)
    schema = jev_schema(body, Limits())
    state = jev_state(body)
    seed = parse_seed(body)
    resolver = TemplateResolver(lambda text: tok.encode(text, add_special_tokens=False),
                                CANVAS)

    def chat_ids(sys_text, thinking):
        msgs = [{"role": "system", "content": sys_text},
                {"role": "user", "content": state}]
        return tok.apply_chat_template(msgs, enable_thinking=thinking)

    result, completion_tokens = decide(
        schema, state, engine=engine, resolver=resolver, chat_ids=chat_ids,
        seed=seed, constrained=constrained, canvas_len=CANVAS, decode=tok.decode)
    diagnostics = result["diagnostics"]
    input_tokens = int(diagnostics.get("prompt_tokens") or 0)
    return {
        "model": MODEL,
        "answers": jev_answers(schema, result),
        "usage": usage(input_tokens, completion_tokens),
        "diagnostics": diagnostics,
    }


def _normalize(obj, engine):
    """The body as it goes on the wire, without timing values and with the
    engine name checked and dropped."""
    def walk(x):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k in ("ms", "total_ms"):
                    continue
                if k == "engine":
                    assert v == engine
                    continue
                out[k] = walk(v)
            return out
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return walk(json.loads(json.dumps(obj)))


# ----------------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------------

_TICKET = "Order 1182 arrived broken and I was charged twice. Fix this today."

_BASE_QS = {
    "urgent": {"type": "noul", "instructions": "Does this need a reply today?",
               "criteria": {"true": "a deadline or harm", "false": "no rush"}},
    "bucket": {"type": "choice", "instructions": "Which team owns it?",
               "criteria": {"billing": "charges and refunds",
                            "shipping": "delivery and damage", "other": None}},
    "tone": {"type": "score", "instructions": "How upset is the customer?",
             "criteria": ["calm", "annoyed", "furious"]},
}


def _many(n, prefix="q"):
    kinds = [
        {"type": "noul", "instructions": "Is it about money?"},
        {"type": "choice", "instructions": "Which channel?",
         "criteria": {"email": None, "phone": "a call", "chat": None}},
        {"type": "score", "instructions": "How clear is it?",
         "criteria": ["vague", "mixed", "clear", "exact"]},
    ]
    return {f"{prefix}{i}": dict(kinds[i % 3]) for i in range(n)}


def _chain():
    qs = copy.deepcopy(_BASE_QS)
    qs["bucket"]["depends_on"] = ["urgent"]
    qs["tone"]["depends_on"] = ["bucket"]
    return qs


def _ask_if():
    qs = copy.deepcopy(_BASE_QS)
    qs["bucket"]["ask_if"] = {"urgent": ["yes"]}
    qs["tone"]["depends_on"] = ["urgent"]
    return qs


# (name, body, bias by token text, noise amplitude)
CASES = [
    ("single_stage", {"state": _TICKET, "questions": _BASE_QS, "seed": 7},
     {" yes": 2.0, " B": 1.5, " 3": 1.0}, 2.0),
    ("state_object", {"state": {"text": _TICKET, "order": 1182},
                      "questions": _BASE_QS, "samples": 2},
     {" no": 1.0}, 1.5),
    ("chunked", {"state": _TICKET, "questions": _many(5), "chunk_rows": 12,
                 "samples": 2}, {" yes": 1.0}, 2.0),
    ("chunked_shared", {"state": _TICKET, "questions": _many(5), "chunk_rows": 12,
                        "chunk_prompt": "shared"}, {" A": 1.0}, 2.0),
    ("chunked_sequential", {"state": _TICKET, "questions": _many(5),
                            "chunk_rows": 12, "sequential": True, "samples": 3},
     {" 2": 1.0}, 2.0),
    ("alone", {"state": _TICKET, "questions": dict(
        _BASE_QS, extra={"type": "noul", "instructions": "Spam?", "alone": True})},
     {}, 2.0),
    ("depends_on_chain", {"state": _TICKET, "questions": _chain(), "samples": 3},
     {" yes": 1.0, " A": 1.0}, 2.0),
    ("ask_if_skip", {"state": _TICKET, "questions": _ask_if(), "samples": 2},
     {" no": 12.0}, 1.0),
    ("ask_if_asked", {"state": _TICKET, "questions": _ask_if(), "samples": 2},
     {" yes": 12.0}, 1.0),
    ("samples_4", {"state": _TICKET, "questions": _BASE_QS, "samples": 4, "seed": 3},
     {" yes": 0.5}, 2.0),
    ("samples_auto_extended", {"state": _TICKET, "questions": _BASE_QS,
                               "samples": "auto", "auto_max": 5}, {}, 1.0),
    ("samples_auto_not_extended", {"state": _TICKET, "samples": "auto",
                                   "questions": {"urgent": _BASE_QS["urgent"]}},
     {" yes": 30.0}, 0.1),
    ("steps_3", {"state": _TICKET, "questions": _BASE_QS, "steps": 3, "samples": 2},
     {" yes": 1.0}, 2.0),
    ("think", {"state": _TICKET, "questions": _BASE_QS, "think": 16, "samples": 2},
     {" yes": 1.0}, 2.0),
    ("think_unclosed", {"state": _TICKET, "questions": _BASE_QS, "think": 4},
     {}, 2.0),
    ("think_chained", {"state": _TICKET, "questions": _chain(), "think": 16,
                       "samples": 2}, {" A": 1.0}, 2.0),
    ("think_chunked", {"state": _TICKET, "questions": _many(4), "chunk_rows": 12,
                       "think": 8}, {}, 2.0),
    ("indexed", {"state": _TICKET, "questions": _many(12), "samples": 2},
     {"q0yes": 1.0}, 2.0),
    ("indexed_chained", {"state": _TICKET, "questions": dict(
        _many(11), late={"type": "noul", "instructions": "Escalate?",
                         "depends_on": ["q0", "q1"]})}, {}, 2.0),
]


@pytest.fixture
def tok(monkeypatch):
    t = FakeTokenizer()
    monkeypatch.setattr(proxy, "CANVAS_LEN", CANVAS)
    monkeypatch.setattr(proxy, "CANVAS_STEP", 16)
    monkeypatch.setattr(proxy, "_template_cache", {})
    proxy.init_tokenizer(t)
    return t


@pytest.mark.parametrize("constrained", [True, False],
                         ids=["constrained", "unconstrained"])
@pytest.mark.parametrize("name,body,bias,noise", CASES, ids=[c[0] for c in CASES])
def test_bodies_match_the_proxy(monkeypatch, capsys, tok, name, body, bias, noise,
                                constrained):
    monkeypatch.setattr(proxy, "ARGS", types.SimpleNamespace(
        model=MODEL, constrained=constrained, upstream="http://upstream.invalid"))
    table = Table(tok, bias, noise, distractors=[" the", " order", ":"])
    _patch_upstream(monkeypatch, tok, table)
    body = dict(body, model="jev-latest")

    want = _normalize(_proxy_body(body), "vllm")
    engine = FakeEngine(table)
    got = _normalize(_gmlx_body(body, tok, engine, constrained), "gmlx")
    assert got == want


def test_cases_cover_the_paths(monkeypatch, tok):
    """The table drives each named case down the path it is named for."""
    monkeypatch.setattr(proxy, "ARGS", types.SimpleNamespace(
        model=MODEL, constrained=True, upstream="http://upstream.invalid"))
    seen = {}
    for name, body, bias, noise in CASES:
        table = Table(tok, bias, noise, distractors=[" the", " order", ":"])
        engine = FakeEngine(table)
        seen[name] = (_gmlx_body(body, tok, engine, True), engine)

    def diag(name):
        return seen[name][0]["diagnostics"]

    assert len(diag("chunked")["chunks"]) > 1
    assert diag("chunked_sequential")["conditioning"] == "prefill"
    assert diag("depends_on_chain")["stages"] == [["urgent"], ["bucket"], ["tone"]]
    # later stages extend the first prompt instead of prefilling again
    assert len(seen["depends_on_chain"][1].prefills) == 1
    assert "bucket" in diag("ask_if_skip")["skipped"]
    assert seen["ask_if_skip"][0]["answers"]["bucket"] is None
    assert not diag("ask_if_asked")["skipped"]
    assert diag("samples_auto_extended")["samples"]["policy"]["extended"] is True
    assert diag("samples_auto_not_extended")["samples"]["policy"]["extended"] is False
    assert diag("samples_4")["samples"]["n"] == 4
    assert diag("think")["thought"]["closed"] is True
    assert diag("think_unclosed")["thought"]["closed"] is False
    assert isinstance(diag("think_chunked")["thought"], list)
    assert len(diag("think_chunked")["thought"]) == len(diag("think_chunked")["chunks"])
    assert diag("indexed")["questions"]["q0"]["pos"] >= 0
    assert len(diag("indexed_chained")["stages"]) == 2
