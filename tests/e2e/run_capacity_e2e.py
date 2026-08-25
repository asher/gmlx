#!/usr/bin/env python3
"""Live check of the capacity-facing surface against a real server.

Launches ``gmlx.server`` on a tiny GGUF with a decode width of 2 and a
queue cap of 3, fires six concurrent streams at it, and asserts what the
new endpoints report while the batch is full and after it drains:

* ``GET /health`` stays liveness-only; ``?ready=1`` answers 200 at idle
  and 503 + ``Retry-After`` with a reason while the batch is full;
* ``/v1/metrics`` ``concurrency`` / ``queue`` / ``requests[]`` agree with
  the load (decode rows never exceed the width, queued rows carry a
  position, rows drain to empty);
* ``/metrics?format=prometheus`` renders and carries the live gauges;
* ``POST /v1/cache/reset`` scoped and unscoped, plus the 404s;
* ``POST /v1/estimate`` (and ``dry_run`` on chat) prices a resident prompt,
  reports a warm prefix after the same prompt has run, and answers
  ``resident: false`` for a model it would have to load;
* ``GET /v1/capacity/plan`` judges a fan-out; ``/v1/metrics`` ``rates``;
* ``/v1/models`` carries ``context_length``; ``POST /unload`` of the
  preloaded primary succeeds (200) and the next request reloads it.

Usage: python tests/e2e/run_capacity_e2e.py [--model PATH] [--streams N]
Exit 0 on pass, 1 on any failed check, 2 when the model is missing.
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

DEFAULT_MODEL = "~/llm/gguf/unsloth__Qwen3-0.6B-GGUF/Qwen3-0.6B-IQ4_XS.gguf"
WIDTH = 2
QUEUE_CAP = 3

# ~1.5k tokens: long enough for the prefix cache to hold whole blocks of it.
LONG_PROMPT = ("Summarize the following notes in two sentences.\n\n" + "\n".join(
    f"Note {i}: the {i}th sample was measured at {i * 7 % 101} units, logged by "
    f"station {i % 13}, and cross-checked against the reference on day {i % 29}."
    for i in range(1, 121)))

_results: list = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def get_raw(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def stream_one(base: str, model: str, idx: int, out: list) -> None:
    body = {"model": model, "stream": True, "max_tokens": 160,
            "temperature": 0.7, "seed": idx,
            "messages": [{"role": "user",
                          "content": f"Write a numbered list of {12 + idx} "
                                     "distinct animals, one per line, with "
                                     "a short fact about each."}]}
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    status, tokens, err = 0, 0, None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            status = r.status
            for line in r:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    tokens += 1
    except urllib.error.HTTPError as e:
        status, err = e.code, e.read().decode()[:200]
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    out.append({"idx": idx, "status": status, "chunks": tokens, "error": err})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--log", default="/tmp/gmlx-capacity-e2e.log")
    a = ap.parse_args()
    model_path = os.path.expanduser(a.model)
    if not os.path.exists(model_path):
        print(f"model missing: {model_path}", file=sys.stderr)
        return 2

    env = {"GMLX_DECODE_BATCH": str(WIDTH), "GMLX_QUEUE_DEPTH_CAP": str(QUEUE_CAP)}
    # A config launch so the prefix cache is on (a bare positional launch
    # serves without APC, and the tier / scoped-reset checks need it).
    cfg_path = a.log + ".yaml"
    with open(cfg_path, "w") as f:
        f.write("server:\n  cache:\n    enabled: true\nmodels:\n  tiny:\n"
                f"    path: {model_path}\n")
    with ServerProc(["--config", cfg_path], env_extra=env, log_path=a.log) as srv:
        srv.wait_ready()
        base = srv.base_url
        c = Client(base)
        st, models = c.get("/v1/models")
        mid = models["data"][0]["id"]
        print(f"server {base} model {mid}")

        # idle
        print("idle:")
        st, hdr, raw = get_raw(f"{base}/health")
        h = json.loads(raw)
        check("health liveness body unchanged", st == 200 and set(h) == {"status", "pid"}, raw)
        st, hdr, raw = get_raw(f"{base}/health?ready=1")
        h = json.loads(raw)
        check("ready=1 at idle -> 200 ready", st == 200 and h.get("ready") is True, raw)
        st, m = c.get("/v1/metrics")
        s = m["server"]
        conc, q = s.get("concurrency", {}), s.get("queue", {})
        check("concurrency.decode_batch == width", conc.get("decode_batch") == WIDTH, json.dumps(conc))
        check("concurrency.queue_cap == cap", conc.get("queue_cap") == QUEUE_CAP)
        check("idle in_flight/waiting == 0", conc.get("in_flight") == 0 and conc.get("waiting") == 0, json.dumps(conc))
        check("queue.eta_s == 0 idle", q.get("eta_s") == 0 and q.get("cap") == QUEUE_CAP, json.dumps(q))
        check("requests[] empty idle", s.get("requests") == [], json.dumps(s.get("requests"))[:200])

        # load
        print(f"load: {a.streams} concurrent streams")
        results: list = []
        threads = [threading.Thread(target=stream_one, args=(base, mid, i, results), daemon=True)
                   for i in range(a.streams)]
        for t in threads:
            t.start()
        samples: list = []
        not_ready = None
        t0 = time.monotonic()
        while any(t.is_alive() for t in threads) and time.monotonic() - t0 < 300:
            st, m = c.get("/v1/metrics", timeout=5)
            if st == 200:
                samples.append(m["server"])
            if not_ready is None:
                st, hdr, raw = get_raw(f"{base}/health?ready=1")
                if st == 503:
                    not_ready = (json.loads(raw), {k.lower(): v for k, v in hdr.items()})
            time.sleep(0.15)
        for t in threads:
            t.join(timeout=5)

        statuses = sorted(r["status"] for r in results)
        print(f"  stream results: {statuses}; errors: {[r['error'] for r in results if r['error']]}")
        n503 = sum(1 for r in results if r["status"] == 503)
        n200 = sum(1 for r in results if r["status"] == 200 and r["chunks"] > 0)
        check("streams completed or were capped with 503", n200 + n503 == a.streams and n200 >= WIDTH,
              f"200={n200} 503={n503}")

        rows_seen = [r for smp in samples for r in smp.get("requests", [])]
        states = {r["state"] for r in rows_seen}
        max_decode = max((sum(1 for r in smp.get("requests", []) if r["state"] == "decode")
                          for smp in samples), default=0)
        max_inflight = max((smp.get("concurrency", {}).get("in_flight") or 0 for smp in samples), default=0)
        max_waiting = max((smp.get("concurrency", {}).get("waiting") or 0 for smp in samples), default=0)
        check("requests[] rows observed under load", bool(rows_seen), f"{len(samples)} samples, {len(rows_seen)} rows")
        check("states within {queued,prefill,decode}", states <= {"queued", "prefill", "decode"}, str(states))
        check("decode rows never exceed width", 0 < max_decode <= WIDTH, f"max decode rows {max_decode}")
        check("queued rows seen with position", any(r["state"] == "queued" and isinstance(r.get("position"), int)
                                                    for r in rows_seen), f"max waiting {max_waiting}")
        dec = [r for r in rows_seen if r["state"] == "decode"]
        check("decode rows carry generated>0 and tok/s", any(r["generated"] > 0 and r.get("decode_tok_s") for r in dec))
        check("rows carry model id, prompt_tokens, max_tokens",
              all(r["model"] == mid and isinstance(r["prompt_tokens"], int) and r["max_tokens"] == 160
                  for r in dec), json.dumps(dec[:1])[:300])
        check("cache tier reported on decode rows", all(r["cache"]["tier"] in ("exact", "block", "miss") for r in dec),
              str({r["cache"]["tier"] for r in dec}))
        check("in_flight observed >= width", max_inflight >= WIDTH, f"max in_flight {max_inflight}")
        etas = [smp.get("queue", {}).get("eta_s") for smp in samples if (smp.get("queue", {}).get("waiting") or 0) > 0]
        check("queue.eta_s > 0 while waiting", all(isinstance(e, int) and e >= 2 for e in etas) and bool(etas),
              f"etas {sorted(set(etas))[:5]}")
        check("ready=1 -> 503 under load with reason + Retry-After",
              not_ready is not None and not_ready[0].get("reason") in ("busy", "queue", "pressure")
              and "retry-after" in not_ready[1], str(not_ready)[:200])

        # prometheus under/after load
        st, hdr, text = get_raw(f"{base}/metrics?format=prometheus")
        ctype = {k.lower(): v for k, v in hdr.items()}.get("content-type", "")
        check("prometheus renders", st == 200 and ctype.startswith("text/plain"), ctype)
        check("prometheus carries live gauges",
              all(k in text for k in ("gmlx_concurrency_in_flight", "gmlx_concurrency_decode_batch",
                                      "gmlx_queue_eta_s", "gmlx_requests_count", "gmlx_governor_band{")),
              text[:0])
        st, hdr, text2 = get_raw(f"{base}/v1/metrics", {"Accept": "text/plain"})
        check("Accept: text/plain negotiates", "gmlx_requests_count" in text2)
        st, m = c.get("/v1/metrics")
        check("JSON default unchanged", st == 200 and "server" in m)

        # drain
        deadline = time.monotonic() + 15
        drained = None
        while time.monotonic() < deadline:
            st, m = c.get("/v1/metrics")
            s = m["server"]
            if s.get("requests") == [] and s["concurrency"].get("in_flight") == 0:
                drained = s
                break
            time.sleep(0.25)
        check("requests[] and in_flight drain to 0", drained is not None,
              json.dumps((s.get("requests"), s.get("concurrency")))[:200])
        st, hdr, raw = get_raw(f"{base}/health?ready=1")
        check("ready=1 -> 200 after drain", st == 200)

        # estimate / plan / rates
        print("estimate:")
        long_msgs = [{"role": "user", "content": LONG_PROMPT}]
        st, body = c.post("/v1/chat/completions",
                          {"model": mid, "messages": long_msgs, "max_tokens": 8})
        check("long prompt completion (warms the cache)", st == 200, json.dumps(body)[:160])
        st, est = c.post("/v1/estimate", {"model": mid, "messages": long_msgs, "max_tokens": 64})
        check("estimate -> 200 resident", st == 200 and est.get("resident") is True, json.dumps(est)[:240])
        check("estimate prompt_tokens > 1000", isinstance(est.get("prompt_tokens"), int)
              and est["prompt_tokens"] > 1000, str(est.get("prompt_tokens")))
        check("estimate need_bytes / fits_now / fits_drained", isinstance(est.get("need_bytes"), int)
              and est["need_bytes"] > 0 and est.get("fits_now") is True and est.get("fits_drained") is True,
              json.dumps({k: est.get(k) for k in ("need_bytes", "avail_now_bytes", "avail_drained_bytes",
                                                  "fits_now", "fits_drained")}))
        check("estimate context_ok with limit", est.get("context_ok") is True
              and isinstance(est.get("context_limit"), int), str(est.get("context_limit")))
        check("estimate warm prefix after the same prompt ran",
              isinstance(est.get("warm_tokens"), int) and est["warm_tokens"] > 0
              and est.get("cache_tier") in ("block", "exact"),
              f"warm={est.get('warm_tokens')} tier={est.get('cache_tier')}")
        check("estimate est_ttft_s is a non-negative number",
              isinstance(est.get("est_ttft_s"), (int, float)) and est["est_ttft_s"] >= 0,
              str(est.get("est_ttft_s")))
        check("estimate carries band / in_flight / decode_batch",
              est.get("decode_batch") == WIDTH and est.get("in_flight") == 0 and "band" in est,
              json.dumps({k: est.get(k) for k in ("band", "in_flight", "waiting", "decode_batch")}))
        st, dry = c.post("/v1/chat/completions",
                         {"model": mid, "messages": long_msgs, "max_tokens": 64, "dry_run": True})
        check("chat dry_run -> same estimate shape", st == 200 and dry.get("resident") is True
              and dry.get("prompt_tokens") == est.get("prompt_tokens"), json.dumps(dry)[:160])
        st, body = c.post("/v1/estimate", {"model": "no-such-model", "messages": long_msgs})
        check("estimate unknown model -> 404", st == 404, json.dumps(body)[:120])
        st, body = c.post("/v1/estimate", {"model": mid})
        check("estimate without messages -> 400", st == 400, json.dumps(body)[:120])
        st, models_after = c.get("/v1/models")
        check("/v1/models still lists only the resident model",
              [x["id"] for x in models_after["data"]] == [mid])
        me = models_after["data"][0]
        check("/v1/models carries context_length", isinstance(me.get("context_length"), int)
              and me["context_length"] > 0, str(me.get("context_length")))
        check("/v1/models max_context_at_width_1 int or null",
              me.get("max_context_at_width_1") is None or isinstance(me["max_context_at_width_1"], int),
              str(me.get("max_context_at_width_1")))

        print("plan:")
        st, plan = c.get(f"/v1/capacity/plan?width={WIDTH}&depth=2048")
        check("plan -> 200 with verdict", st == 200 and "ok" in plan and "admit_now" in plan,
              json.dumps(plan)[:240])
        check("plan idle at width admits now", plan.get("admit_now") is True and plan.get("slots") == WIDTH,
              json.dumps({k: plan.get(k) for k in ("ok", "band", "slots", "admit_now", "reason")}))
        st, plan2 = c.get(f"/v1/capacity/plan?width={WIDTH + 4}&depth=2048")
        check("plan over the width -> not admitted, reason names slots",
              st == 200 and plan2.get("admit_now") is False and "slot" in str(plan2.get("reason")),
              json.dumps({k: plan2.get(k) for k in ("ok", "slots", "admit_now", "reason")}))
        st, plan3 = c.get("/v1/capacity/plan?width=abc&depth=1")
        check("plan bad ints -> 400", st == 400, json.dumps(plan3)[:120])
        st, m = c.get("/v1/metrics")
        rates = m["server"].get("rates") or {}
        check("metrics rates section present", set(rates) >= {"decode_tok_s", "decode_streams",
                                                                "prefill_tok_s_recent", "decode_tok_s_recent",
                                                                "decode_tok_s_lifetime"}, json.dumps(rates))
        check("rates idle: decode_tok_s 0 / streams 0, lifetime > 0",
              rates.get("decode_tok_s") == 0 and rates.get("decode_streams") == 0
              and (rates.get("decode_tok_s_lifetime") or 0) > 0, json.dumps(rates))
        check("rates recent prefill/decode > 0 after completions",
              (rates.get("prefill_tok_s_recent") or 0) > 0 and (rates.get("decode_tok_s_recent") or 0) > 0,
              json.dumps(rates))

        # cache reset
        print("cache reset:")
        st, body = c.post("/v1/cache/reset", {"model": mid})
        check("scoped reset -> cleared [id]", st == 200 and body.get("models") == [mid], json.dumps(body))
        st, body = c.post("/v1/cache/reset")
        check("unscoped reset -> cleared", st == 200 and body.get("status") in ("cleared", "no_cache"), json.dumps(body))
        st, body = c.post("/v1/cache/reset", {"model": "no-such-model"})
        check("unknown model -> 404", st == 404 and body.get("status") == "unknown_model", json.dumps(body))
        st, body = c.get("/v1/cache/stats")
        print(f"  cache stats after reset: {json.dumps(body)[:160]}")

        # explicit unload of the preloaded primary
        print("unload primary:")
        st, body = c.post("/unload", {"model": mid})
        check("unload preloaded primary -> 200", st == 200, json.dumps(body)[:160])
        st, models_u = c.get("/v1/models")
        check("primary no longer resident", st == 200
              and not any(x.get("resident") for x in models_u["data"]), json.dumps(models_u)[:200])
        st, est = c.post("/v1/estimate", {"model": mid, "messages": long_msgs})
        check("estimate on unloaded model -> resident false, no load",
              st == 200 and est.get("resident") is False and est.get("fits_now") is None,
              json.dumps(est)[:160])
        st, body = c.post("/v1/chat/completions",
                          {"model": mid, "messages": [{"role": "user", "content": "Say hi."}],
                           "max_tokens": 8}, timeout=300)
        check("request after unload reloads and answers", st == 200, json.dumps(body)[:160])
        st, models_r = c.get("/v1/models")
        check("primary resident again", any(x.get("resident") for x in models_r["data"]),
              json.dumps(models_r)[:200])
        st, body = c.post("/unload", {"model": mid})
        check("second unload (no hold) -> 200", st == 200, json.dumps(body)[:160])

        failed = [n for n, ok, _ in _results if not ok]
        print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
        if failed:
            print("failed:", failed)
            print(srv.log_tail(40))
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
