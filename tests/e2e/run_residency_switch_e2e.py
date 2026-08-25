#!/usr/bin/env python3
"""Two models that cannot both be resident, one server: the pool switches.

Boots a config with a preloaded primary (``defaults.model``) and a second
model whose weights do not fit next to it, then walks the switching
surface end to end and asserts what the capacity endpoints say at each
step:

* a request for the second model while the primary is held is refused
  cleanly by the load gate (no OOM, no crash, the primary keeps serving);
* an explicit ``POST /unload`` of the held primary succeeds, after which
  the second model loads by request;
* a burst on the second model fills the width; ``requests[]`` rows carry
  its id; a request for the first model while the second is busy is
  refused (busy entries are never evicted), and the live streams finish;
* once drained, a request for the first model evicts the idle second one
  and loads (the log records the eviction), and the reverse switch works
  the same way;
* ``/v1/estimate`` on the non-resident model answers ``resident: false``
  without loading it; ``/v1/capacity/plan`` names the resident model.

Usage: python tests/e2e/run_residency_switch_e2e.py \\
           --primary ID=PATH[:spec] --second ID=PATH[:spec|:draft=PATH]
Exit 0 on pass, 1 on any failed check, 2 when a model is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from client import Client  # noqa: E402
from server_proc import ServerProc  # noqa: E402

_results: list = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return bool(ok)


def parse_model(spec: str):
    mid, rest = spec.split("=", 1)
    path, extra = (rest.split(":", 1) + [""])[:2]
    return mid, os.path.expanduser(path), extra


def write_cfg(path: str, models: list, primary: str) -> None:
    with open(path, "w") as f:
        f.write("server:\n  cache:\n    enabled: true\n  defaults:\n"
                f"    model: {primary}\nmodels:\n")
        for mid, mpath, extra in models:
            f.write(f"  {mid}:\n    path: {mpath}\n")
            if extra == "spec":
                f.write("    speculative: true\n")
            elif extra.startswith("draft="):
                f.write(f"    draft_gguf: {os.path.expanduser(extra[6:])}\n")


def stream_one(base: str, model: str, idx: int, max_tokens: int, out: list) -> None:
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "temperature": 0.7, "seed": idx,
            "messages": [{"role": "user",
                          "content": f"Write a numbered list of {10 + idx} distinct "
                                     "animals, one per line, with a short fact about each."}]}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    status, chunks, err, t0 = 0, 0, None, time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            status = r.status
            for line in r:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    chunks += 1
    except urllib.error.HTTPError as e:
        status, err = e.code, e.read().decode()[:300]
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    out.append({"idx": idx, "model": model, "status": status, "chunks": chunks,
                "error": err, "wall": round(time.monotonic() - t0, 1)})


def resident_ids(c: Client) -> set:
    st, m = c.get("/v1/metrics")
    return {e["ids"][0] for e in m["server"].get("resident_models", []) if e.get("ids")}


def wait_drained(c: Client, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st, m = c.get("/v1/metrics")
        s = m["server"]
        if s.get("requests") == [] and (s["concurrency"].get("in_flight") or 0) == 0:
            return True
        time.sleep(0.5)
    return False


def chat(c: Client, model: str, max_tokens: int = 24, timeout: float = 1800.0):
    return c.post("/v1/chat/completions",
                  {"model": model, "max_tokens": max_tokens, "temperature": 0,
                   "messages": [{"role": "user", "content": "Name three planets."}]},
                  timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="ID=PATH[:spec] - preloaded and held")
    ap.add_argument("--second", required=True, help="ID=PATH[:spec|:draft=PATH]")
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--log", default="/tmp/gmlx-residency-switch-e2e.log")
    a = ap.parse_args()
    models = [parse_model(a.primary), parse_model(a.second)]
    for mid, path, extra in models:
        if not os.path.exists(path):
            print(f"model missing: {path}", file=sys.stderr)
            return 2
        if extra.startswith("draft=") and not os.path.exists(os.path.expanduser(extra[6:])):
            print(f"draft missing: {extra[6:]}", file=sys.stderr)
            return 2
    first, second = models[0][0], models[1][0]
    cfg_path = a.log + ".yaml"
    write_cfg(cfg_path, models, first)
    env = {"GMLX_DECODE_BATCH": str(a.width), "GMLX_QUEUE_DEPTH_CAP": str(a.cap)}
    t_start = time.monotonic()
    with ServerProc(["--config", cfg_path], env_extra=env, log_path=a.log) as srv:
        srv.wait_ready(timeout=900)
        base = srv.base_url
        c = Client(base)
        print(f"server {base}: primary {first}, second {second}")

        # ---- primary preloads (held); wait for it
        print("boot:")
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline and first not in resident_ids(c):
            time.sleep(2)
        check(f"{first} preloaded at boot", first in resident_ids(c), str(resident_ids(c)))
        st, body = chat(c, first)
        check(f"{first} answers", st == 200, json.dumps(body)[:160])
        st, est = c.post("/v1/estimate", {"model": second,
                                          "messages": [{"role": "user", "content": "hi"}]})
        check(f"estimate {second} -> resident false, no load", st == 200 and est.get("resident") is False
              and second not in resident_ids(c), json.dumps(est)[:160])

        # ---- second model cannot fit next to the held primary: clean refusal
        print("gate:")
        st, body = chat(c, second, timeout=900)
        alive = srv.proc.poll() is None
        err = (body or {}).get("error") if isinstance(body, dict) else None
        check(f"{second} request while {first} held -> 503 model_load_deferred (server alive)",
              st == 503 and alive and second not in resident_ids(c)
              and isinstance(err, dict) and err.get("type") == "model_load_deferred",
              f"status {st} alive {alive} body {json.dumps(body)[:200]}")
        st, body = chat(c, first)
        check(f"{first} still serves after the refusal", st == 200, json.dumps(body)[:120])
        st, hdr = c.get("/health?ready=1")
        check("ready=1 -> 200 (primary idle)", st == 200)

        # ---- explicit unload of the held primary, then the second loads
        print("unload primary, load second:")
        st, body = c.post("/unload", {"model": first})
        check(f"unload held primary {first} -> 200", st == 200 and first not in resident_ids(c),
              json.dumps(body)[:160])
        t0 = time.monotonic()
        st, body = chat(c, second, timeout=1800)
        load_s = round(time.monotonic() - t0, 1)
        check(f"{second} loads by request and answers", st == 200 and second in resident_ids(c),
              f"status {st} in {load_s}s; resident {resident_ids(c)}")
        st, est = c.post("/v1/estimate", {"model": second, "max_tokens": 64,
                                          "messages": [{"role": "user", "content": "Name three planets."}]})
        check(f"estimate {second} resident with pricing", st == 200 and est.get("resident") is True
              and isinstance(est.get("need_bytes"), int) and est.get("fits_drained") is True,
              json.dumps({k: est.get(k) for k in ("prompt_tokens", "need_bytes", "fits_now",
                                                  "fits_drained", "context_limit")}))
        st, plan = c.get(f"/v1/capacity/plan?width={a.width}&depth=4096")
        check(f"capacity plan names {second}", st == 200 and plan.get("model") == second,
              json.dumps(plan)[:200])

        # ---- burst on the second model; the first is refused while it is busy
        print(f"burst on {second}:")
        out: list = []
        ts = [threading.Thread(target=stream_one, args=(base, second, i, a.max_tokens, out), daemon=True)
              for i in range(a.streams)]
        for t in ts:
            t.start()
        samples, rows_second, max_dec, max_if = [], 0, 0, 0
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            st, m = c.get("/v1/metrics")
            s = m["server"]
            rows = s.get("requests") or []
            if rows:
                samples.append(s)
                rows_second += sum(1 for r in rows if r.get("model") == second)
                max_dec = max(max_dec, sum(1 for r in rows if r["state"] == "decode"))
                max_if = max(max_if, s["concurrency"].get("in_flight") or 0)
            time.sleep(0.25)
        check(f"requests[] rows carry {second}", rows_second > 0, f"{rows_second} rows / {len(samples)} samples")
        check("decode rows never exceed width", 0 < max_dec <= a.width, f"max decode {max_dec}")
        check("in_flight reached width", max_if >= a.width, f"max in_flight {max_if}")
        st, body = chat(c, first, timeout=900)
        alive = srv.proc.poll() is None
        err = (body or {}).get("error") if isinstance(body, dict) else None
        check(f"{first} request while {second} busy -> 503 model_load_deferred, server alive",
              st == 503 and alive and isinstance(err, dict) and err.get("type") == "model_load_deferred",
              f"status {st} body {json.dumps(body)[:200]}")
        for t in ts:
            t.join(timeout=1800)
        n200 = sum(1 for r in out if r["status"] == 200)
        n503 = sum(1 for r in out if r["status"] == 503)
        check(f"{second} burst streams completed (or capped 503)", n200 + n503 == a.streams
              and n200 >= a.width, str([(r["status"], r["chunks"]) for r in out]))
        check(f"{second} still the only resident", resident_ids(c) == {second}, str(resident_ids(c)))

        # ---- drained: the first model evicts the idle second and loads
        print("switch back:")
        check("drained", wait_drained(c, 60))
        t0 = time.monotonic()
        st, body = chat(c, first, timeout=1800)
        load_s = round(time.monotonic() - t0, 1)
        check(f"{first} evicts idle {second} and answers", st == 200 and resident_ids(c) == {first},
              f"status {st} in {load_s}s; resident {resident_ids(c)}")
        tail = srv.log_tail(400)
        check("log records the LRU eviction", "evicting LRU model" in tail)
        st, plan = c.get(f"/v1/capacity/plan?width={a.width}&depth=4096")
        check(f"capacity plan now names {first}", st == 200 and plan.get("model") == first,
              json.dumps(plan)[:160])

        # ---- and forward again
        t0 = time.monotonic()
        st, body = chat(c, second, timeout=1800)
        load_s = round(time.monotonic() - t0, 1)
        check(f"{second} evicts {first} and answers again", st == 200 and resident_ids(c) == {second},
              f"status {st} in {load_s}s; resident {resident_ids(c)}")
        st, m = c.get("/v1/metrics")
        s = m["server"]
        check("idle metrics sane after switching", s.get("requests") == []
              and (s["concurrency"].get("in_flight") or 0) == 0
              and len(s.get("resident_models", [])) == 1, json.dumps(s.get("concurrency")))
        st, hdr = c.get("/health?ready=1")
        check("ready=1 -> 200 at the end", st == 200)
        bad = [ln for ln in srv.log_tail(2000).splitlines()
               if "Traceback" in ln or " ERROR" in ln or "CRITICAL" in ln]
        check("no tracebacks / ERROR lines in the server log", not bad, "\n".join(bad[:3])[:400])

        failed = [n for n, ok, _ in _results if not ok]
        print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed "
              f"in {time.monotonic() - t_start:.0f}s")
        if failed:
            print("failed:", failed)
            print(srv.log_tail(60))
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
