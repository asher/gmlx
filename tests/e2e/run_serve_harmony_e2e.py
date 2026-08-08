#!/usr/bin/env python3
"""Live serve-content contract for harmony (gpt-oss) models.

Boots a real ``gmlx.server`` on a harmony GGUF and asserts the response
contract the unit seams cannot prove end-to-end: ``content`` and
``reasoning_content`` carry no channel markup (non-stream and stream), a
``finish_reason=length`` reply truncated inside analysis returns empty
content with the partial reasoning, and a client that feeds a reply back as
history gets 200 - the exact request shape the leaked markup used to 500
(the model's own chat template rejects ``<|channel|>`` in a content field).

Red on main for every content/markup check; part of the pre-release live
gate alongside the APC engagement runs.

    python tests/e2e/run_serve_harmony_e2e.py --model ~/llm/gguf/.../gpt-oss-20b-MXFP4.gguf
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_proc import ServerProc         # noqa: E402

_DEFAULT_GLOBS = (
    "~/llm/gguf-test/**/gpt-oss*.gguf",
    "~/llm/gguf/**/gpt-oss*.gguf",
)


def find_default_model() -> str | None:
    for pattern in _DEFAULT_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pattern), recursive=True))
        if hits:
            return hits[0]
    return None


class Report:
    def __init__(self):
        self.failures = []

    def check(self, name: str, ok: bool, detail: str = ""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            self.failures.append(name)


def chat(base: str, mid: str, messages: list, *, max_tokens: int = 200,
         stream: bool = False):
    body = {"model": mid, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": stream}
    if not stream:
        r = requests.post(f"{base}/v1/chat/completions", json=body,
                          timeout=600)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {}
    r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=600,
                      stream=True)
    deltas = {"reasoning": "", "content": ""}
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            break
        d = json.loads(payload)["choices"][0].get("delta") or {}
        for k in ("reasoning", "reasoning_content"):
            if d.get(k):
                deltas["reasoning"] += d[k]
                break
        if d.get("content"):
            deltas["content"] += d["content"]
    return r.status_code, deltas


def msg_of(body: dict):
    m = body["choices"][0]["message"]
    return (str(m.get("content") or ""),
            str(m.get("reasoning_content") or m.get("reasoning") or ""),
            body["choices"][0].get("finish_reason"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", default=None,
                    help="harmony GGUF path (default: first gpt-oss under "
                         "~/llm/gguf-test or ~/llm/gguf)")
    ap.add_argument("--out", default=None, help="server log dir (default: cwd)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the serve subprocess")
    a = ap.parse_args()

    model = a.model or find_default_model()
    if not model or not os.path.exists(os.path.expanduser(model)):
        print("SKIP: no gpt-oss GGUF found (pass --model)")
        return 0
    model = os.path.expanduser(model)
    out = a.out or os.getcwd()
    os.makedirs(out, exist_ok=True)

    rep = Report()
    sp = ServerProc([model, "--no-auth"],
                    log_path=os.path.join(out, "serve-harmony.log"),
                    python=a.python)
    sp.start()
    try:
        sp.wait_ready(timeout=600)
        base = sp.base_url
        mids = requests.get(f"{base}/v1/models", timeout=30).json()
        mid = mids["data"][0]["id"]

        st, body = chat(base, mid, [
            {"role": "user",
             "content": "In one short sentence: why is the sky blue?"}])
        content, reasoning, fin = msg_of(body)
        rep.check("normal.status", st == 200, f"status {st}")
        rep.check("normal.content_clean",
                  bool(content) and "<|" not in content, repr(content[:70]))
        rep.check("normal.reasoning_clean", "<|" not in reasoning,
                  f"len={len(reasoning)}")

        st, body2 = chat(base, mid, [
            {"role": "user",
             "content": "Explain quantum entanglement carefully."}],
            max_tokens=8)
        c2, r2, fin2 = msg_of(body2)
        rep.check("capped.finish_length", st == 200 and fin2 == "length",
                  f"status {st} finish {fin2}")
        rep.check("capped.no_markup_in_content", "<|" not in c2, repr(c2[:70]))
        rep.check("capped.empty_content_with_reasoning",
                  c2 == "" and r2 != "",
                  f"content={c2!r:.50} reasoning_len={len(r2)}")

        # The old failure mode: a resent reply carrying markup made the
        # template raise -> 500. Both a full and a capped reply must render.
        st, body3 = chat(base, mid, [
            {"role": "user",
             "content": "In one short sentence: why is the sky blue?"},
            {"role": "assistant", "content": content,
             "reasoning_content": reasoning},
            {"role": "user", "content": "And why are sunsets red?"}])
        rep.check("turn2.status_200", st == 200, f"status {st}")
        if st == 200:
            c3, _, _ = msg_of(body3)
            rep.check("turn2.content_clean",
                      bool(c3) and "<|" not in c3, repr(c3[:70]))
        st, _ = chat(base, mid, [
            {"role": "user",
             "content": "Explain quantum entanglement carefully."},
            {"role": "assistant", "content": c2, "reasoning_content": r2},
            {"role": "user", "content": "Shorter please."}])
        rep.check("turn2_after_cap.status_200", st == 200, f"status {st}")

        st, deltas = chat(base, mid, [
            {"role": "user", "content": "Name three primary colors."}],
            stream=True)
        rep.check("stream.content_clean",
                  st == 200 and bool(deltas["content"])
                  and "<|" not in deltas["content"],
                  repr(deltas["content"][:70]))
        rep.check("stream.reasoning_clean", "<|" not in deltas["reasoning"],
                  f"len={len(deltas['reasoning'])}")
    finally:
        sp.stop()

    if rep.failures:
        print(f"FAILURES: {', '.join(rep.failures)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
