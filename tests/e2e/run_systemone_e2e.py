#!/usr/bin/env python3
"""Live check of POST /v1/systemone on a real DiffusionGemma GGUF.

Boots ``gmlx.serve.server`` from a one-model config with the default
``server.systemone`` settings and posts structured-decision requests over
HTTP. It checks the answer shapes per question type, that each answer is a
distribution, that the obvious answers win, that a replay with the same
seed returns the same answers and label logprobs, that images are refused,
that a chat request to the same model completes while a decision runs, that
``gmlx systemone`` prints the answers from the same server, and that a
served chat prompt starts with one BOS.

    python tests/e2e/run_systemone_e2e.py \
        --model ~/llm/gguf/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf

Each response body is written under ``--out`` for a slot-by-slot diff
against another implementation. Exit status 0 means every check passed.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_proc import ServerProc  # noqa: E402

_DEFAULT_GLOBS = (
    "~/llm/gguf/**/diffusiongemma*.gguf",
    "~/llm/gguf-test/**/diffusiongemma*.gguf",
)
_MODEL_ID = "dgemma"

TICKET = {
    "model": "jev-latest",
    "state": {"ticket": "Everything is down and we have a demo at noon."},
    "questions": {"urgent": {
        "type": "noul",
        "instructions": "Does the customer need a reply within the hour?"}},
}

TRIAGE = {
    "model": "jev-latest",
    "state": {"ticket": "I was charged twice for my March invoice. "
                        "Please refund the duplicate charge.",
              "customer": {"plan": "pro", "tenure_months": 30}},
    "questions": {
        "area": {"type": "choice", "instructions": "Which team owns the ticket?",
                 "criteria": {"billing": "invoices, charges and refunds",
                              "infra": "outages and latency",
                              "product": "features and bugs"}},
        "severity": {"type": "score", "instructions": "How severe is the problem?",
                     "criteria": ["low", "medium", "high"]},
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?",
                   "depends_on": ["area"], "ask_if": {"area": ["billing"]}},
        "outage": {"type": "noul", "instructions": "Is a service outage reported?",
                   "ask_if": {"area": ["infra"]}},
    },
}

# A read without a thought picks the 18th century here at 0.86, above the
# default think_threshold of 0.8, so the think "auto" case raises it to 0.9.
CENTURY = {
    "model": "jev-latest",
    "state": {"event": "the opening of the Suez Canal"},
    "questions": {"century": {
        "type": "choice", "instructions": "In which century did the event happen?",
        "criteria": {"16th": "1501-1600", "17th": "1601-1700", "18th": "1701-1800",
                     "19th": "1801-1900", "20th": "1901-2000"}}},
}

# Past ten questions the answer template writes each label right after its
# id, so the ids are numbered: a word id merges with the label into
# different tokens per label, which the template check refuses.
_FACTS = [
    ("q1", "Is the sky usually blue on a clear day?", True),
    ("q2", "Is fire cold?", False),
    ("q3", "Do fish live in water?", True),
    ("q4", "Can a stone speak?", False),
    ("q5", "Does the sun rise in the east?", True),
    ("q6", "Is ice hot?", False),
    ("q7", "Can most birds fly?", True),
    ("q8", "Is the moon made of cheese?", False),
    ("q9", "Is water wet?", True),
    ("q10", "Is snow black?", False),
    ("q11", "Do trees have leaves?", True),
    ("q12", "Do cats bark?", False),
]
FACTS = {
    "model": "jev-latest",
    "state": "General knowledge quiz.",
    "questions": {k: {"type": "noul", "instructions": q} for k, q, _ in _FACTS},
}
_WORDS = ["sky", "fire", "fish", "stone", "sun", "ice", "birds", "moon",
          "water", "snow", "trees", "cats"]
FACTS_WORD_IDS = {**FACTS, "questions": {
    w: q for w, q in zip(_WORDS, FACTS["questions"].values())}}


def find_default_model() -> str | None:
    for pattern in _DEFAULT_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pattern), recursive=True))
        if hits:
            return hits[0]
    return None


class Report:
    def __init__(self):
        self.failures = []
        self.timings = []

    def check(self, name: str, ok: bool, detail: str = ""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}".rstrip())
        if not ok:
            self.failures.append(name)


def post(base: str, body: dict, *, timeout: float = 900.0):
    t0 = time.perf_counter()
    r = requests.post(f"{base}/v1/systemone", json=body, timeout=timeout)
    ms = (time.perf_counter() - t0) * 1e3
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text}
    return r.status_code, data, ms


def check_shapes(rep: Report, name: str, body: dict, data: dict):
    answers = data.get("answers") or {}
    rep.check(f"{name}: every question answered",
              set(answers) == set(body["questions"]), f"{sorted(answers)}")
    for qid, a in answers.items():
        if a is None:
            continue
        kind = a.get("type")
        if kind == "noul":
            rep.check(f"{name}: {qid} noul in [0, 1]",
                      set(a) == {"type", "noul"} and 0.0 <= a["noul"] <= 1.0,
                      f"{a}")
            continue
        probs = a.get("probabilities") or {}
        total = sum(probs.values())
        keys = {"type", "choice", "probabilities", "confidence"} if kind == "choice" \
            else {"type", "score", "legend", "probabilities", "confidence"}
        rep.check(f"{name}: {qid} {kind} shape", set(a) == keys, f"{sorted(a)}")
        rep.check(f"{name}: {qid} probabilities sum to 1",
                  math.isclose(total, 1.0, abs_tol=1e-3), f"sum={total:.4f}")
    usage = data.get("usage") or {}
    rep.check(f"{name}: usage counts",
              usage.get("input_tokens", 0) > 0 and usage.get("output_tokens", 0) > 0,
              f"{usage}")


def dump(out_dir: str, name: str, body: dict, data: dict):
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump({"request": body, "response": data}, f, indent=2)


def run_decisions(rep: Report, base: str, out_dir: str):
    print("\n[decisions]")
    cases = [
        ("ticket", TICKET),
        ("ticket_samples4", {**TICKET, "samples": 4}),
        ("ticket_think64", {**TICKET, "think": 64}),
        ("triage_chain", TRIAGE),
        ("facts_indexed", FACTS),
        ("century_think_auto", {**CENTURY, "think": "auto", "think_threshold": 0.9,
                                "think_budget": 128}),
        ("ticket_think_auto", {**TICKET, "think": "auto"}),
    ]
    results = {}
    for name, body in cases:
        status, data, ms = post(base, body)
        dump(out_dir, name, body, data)
        rep.check(f"{name}: 200", status == 200, "" if status == 200 else f"{status} {data}")
        if status != 200:
            continue
        results[name] = data
        check_shapes(rep, name, body, data)
        diag = data.get("diagnostics") or {}
        timing = diag.get("timing") or {}
        rep.timings.append((name, ms, timing.get("reads"),
                            (data.get("usage") or {}).get("input_tokens")))

    t = results.get("ticket")
    if t:
        a = t["answers"]["urgent"]
        rep.check("ticket: urgent is yes", a["noul"] > 0.8, f"noul={a['noul']:.3f}")
        rep.check("ticket: model echoed", t.get("model") == _MODEL_ID, f"{t.get('model')}")
        policy = ((t.get("diagnostics") or {}).get("samples") or {}).get("policy") or {}
        rep.check("ticket: auto policy reported", "extended" in policy, f"{policy}")
    s4 = results.get("ticket_samples4")
    if s4:
        tops = ((s4.get("diagnostics") or {}).get("samples") or {}).get("tops") or []
        rep.check("ticket_samples4: four samples", len(tops) == 4, f"{len(tops)}")
    th = results.get("ticket_think64")
    if th:
        thought = (th.get("diagnostics") or {}).get("thought") or {}
        rep.check("ticket_think64: thought reported",
                  0 < int(thought.get("tokens") or 0) <= 64, f"{thought.get('tokens')}")
    tr = results.get("triage_chain")
    if tr:
        a = tr["answers"]
        rep.check("triage_chain: area billing", (a.get("area") or {}).get("choice") == "billing",
                  f"{a.get('area')}")
        rep.check("triage_chain: refund asked and yes",
                  a.get("refund") is not None and a["refund"]["noul"] > 0.5, f"{a.get('refund')}")
        rep.check("triage_chain: outage skipped", a.get("outage") is None, f"{a.get('outage')}")
        stages = (tr.get("diagnostics") or {}).get("stages") or []
        rep.check("triage_chain: two stages", len(stages) == 2, f"{stages}")
    ca = results.get("century_think_auto")
    if ca:
        auto = (ca.get("diagnostics") or {}).get("think_auto") or {}
        rep.check("century_think_auto: unsure, so the thought ran",
                  auto.get("thought") is True and auto.get("unsure") == ["century"]
                  and auto.get("budget") == 128,
                  f"{auto}")
        rep.check("century_think_auto: 19th century",
                  ca["answers"]["century"]["choice"] == "19th", f"{ca['answers']['century']}")
    ta = results.get("ticket_think_auto")
    if ta:
        auto = (ta.get("diagnostics") or {}).get("think_auto") or {}
        rep.check("ticket_think_auto: sure, so no thought",
                  auto.get("thought") is False and not (ta.get("diagnostics") or {}).get("thought"),
                  f"{auto}")
    fx = results.get("facts_indexed")
    if fx:
        wrong = [k for k, _, truth in _FACTS
                 if (fx["answers"][k]["noul"] > 0.5) != truth]
        rep.check("facts_indexed: obvious answers", not wrong, f"wrong={wrong}")


def run_determinism(rep: Report, base: str):
    print("\n[determinism]")
    body = {**TRIAGE, "seed": 7, "samples": 2}
    runs = [post(base, body) for _ in range(2)]
    ok = all(s == 200 for s, _, _ in runs)
    rep.check("replay: 200", ok)
    if not ok:
        return
    (_, a, _), (_, b, _) = runs
    rep.check("replay: answers identical", a["answers"] == b["answers"])
    tops_a = a["diagnostics"]["samples"]["tops"]
    tops_b = b["diagnostics"]["samples"]["tops"]
    rep.check("replay: label logprobs identical", tops_a == tops_b)
    status, c, _ = post(base, {**body, "seed": 8})
    rep.check("replay: another seed draws other slot tokens",
              status == 200 and c["diagnostics"]["samples"]["tops"] != tops_a)


def run_refusals(rep: Report, base: str):
    print("\n[refusals]")
    status, data, _ = post(base, {**TICKET, "images": ["data:image/png;base64,AAAA"]})
    rep.check("images: 400", status == 400, f"{status} {data}")
    status, data, _ = post(base, {**TICKET, "questions": {"x": {"type": "nope"}}})
    err = (data.get("error") or {}) if isinstance(data, dict) else {}
    rep.check("bad question: 422 validation_error",
              status == 422 and err.get("type") == "validation_error", f"{status} {data}")
    status, data, _ = post(base, FACTS_WORD_IDS)
    err = (data.get("error") or {}) if isinstance(data, dict) else {}
    rep.check("word ids past ten questions: 422 as in the vLLM example",
              status == 422 and "do not share one template slot" in str(err.get("message")),
              f"{status} {data}")


def run_concurrent_chat(rep: Report, base: str):
    print("\n[concurrent chat]")
    out = {}

    def decision():
        out["decision"] = post(base, {**TICKET, "think": 64, "samples": 4})

    def chat():
        t0 = time.perf_counter()
        r = requests.post(f"{base}/v1/chat/completions", timeout=900, json={
            "model": _MODEL_ID, "max_tokens": 32, "temperature": 0.0,
            "messages": [{"role": "user", "content": "Say hello in one word."}]})
        out["chat"] = (r.status_code, r.json(), (time.perf_counter() - t0) * 1e3)

    threads = [threading.Thread(target=decision), threading.Thread(target=chat)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    d_status, _, d_ms = out["decision"]
    c_status, c_body, c_ms = out["chat"]
    rep.check("concurrent: decision 200", d_status == 200, f"{d_ms:.0f} ms")
    content = ((c_body.get("choices") or [{}])[0].get("message") or {}).get("content") \
        if isinstance(c_body, dict) else None
    rep.check("concurrent: chat 200 with content", c_status == 200 and bool(content),
              f"{c_ms:.0f} ms {content!r}")
    return c_body


def run_single_bos(rep: Report, base: str, model_path: str):
    print("\n[single BOS]")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from gmlx.distill.tokens import tokenizer_from_gguf
    from gmlx.gen.generation import encode_prompt

    tok = tokenizer_from_gguf(model_path)
    msgs = [{"role": "user", "content": "Say hello in one word."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = list(encode_prompt(tok, text))
    r = requests.post(f"{base}/v1/chat/completions", timeout=900, json={
        "model": _MODEL_ID, "max_tokens": 8, "temperature": 0.0, "messages": msgs})
    usage = r.json().get("usage") or {}
    rep.check("served chat: one BOS", ids.count(2) == 1 and
              usage.get("prompt_tokens") == len(ids),
              f"prompt_tokens={usage.get('prompt_tokens')} single-BOS ids={len(ids)}")


def run_cli(rep: Report, base: str, python: str, out_dir: str):
    print("\n[gmlx systemone]")
    path = os.path.join(out_dir, "ticket_request.json")
    with open(path, "w") as f:
        json.dump(TICKET, f)
    r = subprocess.run([python, "-m", "gmlx", "systemone", path, "--url", base],
                       capture_output=True, text=True, timeout=900)
    line = r.stdout.strip()
    rep.check("cli: exit 0 and one answer line",
              r.returncode == 0 and line.startswith("urgent: "), f"{line!r} {r.stderr.strip()[-200:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=None, help="DiffusionGemma GGUF path")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=os.path.expanduser(
        "~/.local/state/claude-scratch/gmlx/systemone/e2e"))
    a = ap.parse_args()
    model = a.model or find_default_model()
    if not model:
        print("no DiffusionGemma GGUF found; pass --model", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    cfg = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    cfg.write(json.dumps({"models": {_MODEL_ID: {"path": os.path.expanduser(model)}},
                          "server": {"systemone": {"canvas": 64}}}))
    cfg.close()
    rep = Report()
    log = os.path.join(a.out, "server.log")
    sp = ServerProc(["--config", cfg.name, "--no-auth"], log_path=log, python=a.python)
    try:
        sp.start()
        sp.wait_ready(timeout=900)
        base = sp.base_url
        status, data, ms = post(base, TICKET)
        print(f"warm-up: {status} in {ms:.0f} ms (includes the load)")
        run_decisions(rep, base, a.out)
        run_determinism(rep, base)
        run_refusals(rep, base)
        run_concurrent_chat(rep, base)
        run_single_bos(rep, base, model)
        run_cli(rep, base, a.python, a.out)
    finally:
        sp.stop()
        os.unlink(cfg.name)
    with open(log) as f:
        text = f.read()
    rep.check("log: systemone lines", "systemone: " in text)
    rep.check("log: [req] lines for /v1/systemone", "/v1/systemone" in text)
    print("\n| case | wall ms | reads | input tokens |")
    print("|------|---------|-------|--------------|")
    for name, ms, reads, inp in rep.timings:
        print(f"| {name} | {ms:.0f} | {reads} | {inp} |")
    print(f"\nlog: {log}")
    if rep.failures:
        print(f"\nFAILED: {len(rep.failures)}: {rep.failures}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
