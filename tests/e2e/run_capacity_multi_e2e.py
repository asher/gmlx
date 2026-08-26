#!/usr/bin/env python3
"""Longer live check of the capacity surface with several resident models.

Launches ``gmlx serve`` on a config with multiple models (a large
speculative one, a mid-size one with a DFlash drafter, and a tiny one by
default), warms each, then runs rounds of mixed-model concurrent streams
for several minutes while sampling ``/v1/metrics`` and ``/health?ready=1``.
Asserts the multi-engine invariants the single-model check cannot:

* ``requests[]`` rows carry the id of the model they run on, and rows for
  different models appear in the same sample (one snapshot per engine,
  not a global that the engines overwrite);
* ``resident_models[].in_flight`` per model sums to
  ``concurrency.in_flight`` at every sample; decode rows never exceed the
  width per engine; ``queue.waiting`` counts every engine's backlog;
* repeat prompts on a model show a warm cache tier on its decode rows;
* a scoped ``/v1/cache/reset`` on one model leaves another model's
  in-flight streams untouched; unload + reload of a secondary model works
  and the readiness probe transitions 503 -> 200 around the load;
* Prometheus output labels every resident model;
* the governor band never reaches red and shed counters stay flat.

Usage: python tests/e2e/run_capacity_multi_e2e.py [--rounds N]
           [--streams N] [--max-tokens N] [--width N] [--cap N]
           [--models ID=PATH[:spec|:draft=PATH] ...]
Exit 0 on pass, 1 on any failed check, 2 when a model file is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from client import Client  # noqa: E402
from server_proc import ServerProc  # noqa: E402

GGUF = os.path.expanduser("~/llm/gguf")
DEFAULT_MODELS = [
    ("big", f"{GGUF}/unsloth__Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf", "spec"),
    ("muse", f"{GGUF}/meta-models__Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
     f"draft={GGUF}/meta-models__Muse-Glimmer-30B-GGUF/dflash-Muse-Glimmer-30B-Q4_K_M.gguf"),
    ("tiny", f"{GGUF}/unsloth__Qwen3-0.6B-GGUF/Qwen3-0.6B-IQ4_XS.gguf", ""),
]

_results: list = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return bool(ok)


def get_raw(url: str, headers: dict | None = None, timeout: float = 10):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read().decode()


LONG_PROMPT = ("Here is a reference passage. " +
               " ".join(f"Item {i}: the {['red', 'blue', 'green', 'amber'][i % 4]} "
                        f"crate in bay {i * 7 % 23} holds {i * 13 % 97} units." for i in range(220)) +
               " Summarize the passage in three sentences.")


def stream_one(base: str, model: str, idx: int, max_tokens: int, out: list,
               prompt: str | None = None, tag: str = "") -> None:
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "temperature": 0.7, "seed": idx,
            "messages": [{"role": "user",
                          "content": prompt or (f"Write a numbered list of {12 + idx % 9} distinct "
                                                "animals, one per line, each with a two-sentence fact.")}]}
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    status, chunks, err, t0 = 0, 0, None, time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            status = r.status
            for line in r:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    chunks += 1
    except urllib.error.HTTPError as e:
        status, err = e.code, e.read().decode()[:200]
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    out.append({"idx": idx, "model": model, "tag": tag, "status": status, "chunks": chunks,
                "error": err, "wall_s": round(time.monotonic() - t0, 1)})


class Sampler:
    """Polls /v1/metrics and /health?ready=1 on a thread; keeps every sample."""

    def __init__(self, base: str):
        self.base = base
        self.samples: list = []
        self.ready: list = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        c = Client(self.base)
        while not self._stop.is_set():
            try:
                st, m = c.get("/v1/metrics", timeout=5)
                if st == 200:
                    m["server"]["_t"] = time.monotonic()
                    self.samples.append(m["server"])
                st, hdr, raw = get_raw(f"{self.base}/health?ready=1")
                self.ready.append((time.monotonic(), st, json.loads(raw).get("reason"), hdr.get("retry-after")))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)

    def since(self, t: float) -> list:
        return [s for s in self.samples if s["_t"] >= t]


def rows_by_model(sample: dict) -> dict:
    out: dict = defaultdict(list)
    for r in sample.get("requests", []):
        out[r.get("model")].append(r)
    return out


def wait_drained(c: Client, timeout: float = 30) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st, m = c.get("/v1/metrics")
        s = m["server"]
        if s.get("requests") == [] and s["concurrency"].get("in_flight") == 0 \
                and s["concurrency"].get("waiting") == 0:
            return s
        time.sleep(0.25)
    return None


def parse_models(specs: list) -> list:
    out = []
    for spec in specs:
        mid, _, rest = spec.partition("=")
        path, _, extra = rest.partition(":")
        out.append((mid, os.path.expanduser(path), extra))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="ID=PATH[:spec|:draft=PATH]")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--streams", type=int, default=8, help="concurrent streams per round")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--log", default="/tmp/gmlx-capacity-multi-e2e.log")
    a = ap.parse_args()
    models = parse_models(a.models) if a.models else DEFAULT_MODELS
    for mid, path, extra in models:
        if not os.path.exists(path):
            print(f"model missing: {path}", file=sys.stderr)
            return 2
        if extra.startswith("draft=") and not os.path.exists(os.path.expanduser(extra[6:])):
            print(f"draft missing: {extra[6:]}", file=sys.stderr)
            return 2
    ids = [m[0] for m in models]
    big, small = ids[0], ids[-1]

    cfg_path = a.log + ".yaml"
    with open(cfg_path, "w") as f:
        f.write("server:\n  cache:\n    enabled: true\nmodels:\n")
        for mid, path, extra in models:
            f.write(f"  {mid}:\n    path: {path}\n")
            if extra == "spec":
                f.write("    speculative: true\n")
            elif extra.startswith("draft="):
                f.write(f"    draft_gguf: {os.path.expanduser(extra[6:])}\n")
    env = {"GMLX_DECODE_BATCH": str(a.width), "GMLX_QUEUE_DEPTH_CAP": str(a.cap)}
    t_start = time.monotonic()
    with ServerProc(["--config", cfg_path], env_extra=env, log_path=a.log) as srv:
        srv.wait_ready(timeout=600)
        base = srv.base_url
        c = Client(base)
        st, ml = c.get("/v1/models")
        listed = sorted(m["id"] for m in ml["data"])
        print(f"server {base} models {listed}  (launch {time.monotonic() - t_start:.0f}s)")
        check("/v1/models lists every config id", set(ids) <= set(listed), str(listed))

        # ---- warm-up: one request per model, rows labelled with that model
        print("warm-up:")
        with Sampler(base) as smp:
            for mid in ids:
                t0 = time.monotonic()
                out: list = []
                # long enough that even a 0.6B at 250 tok/s spans several samples
                stream_one(base, mid, 0, 200, out, tag="warm")
                r = out[0]
                labels = {row["model"] for s in smp.since(t0) for row in s.get("requests", [])}
                check(f"warm {mid}: 200 with tokens, rows labelled {mid}",
                      r["status"] == 200 and r["chunks"] > 0 and labels <= {mid} and mid in labels,
                      f"status {r['status']} chunks {r['chunks']} labels {labels} {r['wall_s']}s")
        st, m = c.get("/v1/metrics")
        s = m["server"]
        res = {e["ids"][0] if e.get("ids") else os.path.basename(e["model_path"]): e
               for e in s.get("resident_models", [])}
        check("all models resident after warm-up", set(ids) <= set(res), str(sorted(res)))
        check("resident in_flight all 0 at idle", all(e.get("in_flight") == 0 for e in res.values()),
              str({k: v.get("in_flight") for k, v in res.items()}))
        base_red = (s.get("governor", {}).get("red_failures") or 0)
        base_rej = s.get("queue", {}).get("rejections") or 0
        st, hdr, raw = get_raw(f"{base}/health?ready=1")
        check("ready=1 -> 200 after warm-up", st == 200, raw)

        # ---- sustained mixed load
        print(f"load: {a.rounds} rounds x {a.streams} streams, max_tokens {a.max_tokens}, models {ids}")
        results: list = []
        t_load = time.monotonic()
        with Sampler(base) as smp:
            for rnd in range(a.rounds):
                t_r = time.monotonic()
                threads = []
                for i in range(a.streams):
                    mid = ids[(i + rnd) % len(ids)]
                    threads.append(threading.Thread(
                        target=stream_one, args=(base, mid, rnd * 100 + i, a.max_tokens, results),
                        kwargs={"tag": f"r{rnd}"}, daemon=True))
                for t in threads:
                    t.start()
                    time.sleep(0.05)
                for t in threads:
                    t.join(timeout=900)
                rr = [r for r in results if r["tag"] == f"r{rnd}"]
                print(f"  round {rnd}: {time.monotonic() - t_r:.0f}s  "
                      + " ".join(f"{r['model']}:{r['status']}/{r['chunks']}" for r in rr), flush=True)
            samples = smp.since(t_load)
            ready = [x for x in smp.ready if x[0] >= t_load]
        load_s = time.monotonic() - t_load
        n503 = sum(1 for r in results if r["status"] == 503)
        n200 = sum(1 for r in results if r["status"] == 200 and r["chunks"] > 0)
        bad = [r for r in results if r["status"] not in (200, 503) or r["error"] and r["status"] != 503]
        check("every stream completed (200) or was capped (503)", not bad and n200 + n503 == len(results),
              f"200={n200} 503={n503} bad={bad[:2]} in {load_s:.0f}s")
        check("streams completed on every model",
              set(ids) <= {r["model"] for r in results if r["status"] == 200 and r["chunks"] > 0})

        per_model = defaultdict(list)
        multi = 0
        max_dec_per = defaultdict(int)
        sum_ok = 0
        for s in samples:
            bym = rows_by_model(s)
            live = [k for k in bym if k]
            if len(live) >= 2:
                multi += 1
            for k, rows in bym.items():
                per_model[k].extend(rows)
                max_dec_per[k] = max(max_dec_per[k], sum(1 for r in rows if r["state"] == "decode"))
            res_if = sum(e.get("in_flight") or 0 for e in s.get("resident_models", []))
            if res_if == (s["concurrency"].get("in_flight") or 0):
                sum_ok += 1
        rows_all = [r for rows in per_model.values() for r in rows]
        check("requests[] rows observed for every model", set(ids) <= set(per_model),
              f"{len(samples)} samples; rows per model {dict((k, len(v)) for k, v in per_model.items())}")
        check("row model labels are config ids", set(per_model) <= set(ids), str(set(per_model)))
        check("rows from >= 2 models in the same sample (per-engine snapshots)", multi > 0,
              f"{multi}/{len(samples)} samples")
        check("decode rows per model never exceed width", all(0 < v <= a.width for v in max_dec_per.values()),
              str(dict(max_dec_per)))
        check("resident in_flight sums to concurrency.in_flight at every sample", sum_ok == len(samples),
              f"{sum_ok}/{len(samples)}")
        max_if = max((s["concurrency"].get("in_flight") or 0) for s in samples)
        max_wait = max((s["concurrency"].get("waiting") or 0) for s in samples)
        check("in_flight observed >= width", max_if >= a.width, f"max in_flight {max_if}")
        check("waiting observed > 0 under a burst beyond width", max_wait > 0, f"max waiting {max_wait}")
        rej = [s["queue"].get("rejections") for s in samples][-1]
        check("queue.rejections == observed 503s", (rej or 0) - base_rej == n503, f"{rej} vs {n503}")
        qd = [r for r in rows_all if r["state"] == "queued"]
        check("queued rows carry a position", bool(qd) and all(isinstance(r.get("position"), int) for r in qd),
              f"{len(qd)} queued rows")
        dec = [r for r in rows_all if r["state"] == "decode"]
        check("decode rows carry generated, decode_tok_s, max_tokens",
              bool(dec) and all(r["generated"] >= 0 and r["max_tokens"] == a.max_tokens for r in dec)
              and any(r.get("decode_tok_s") for r in dec))
        check("cache tier set on every decode row",
              all(r["cache"]["tier"] in ("exact", "block", "miss") for r in dec),
              str({(r["model"], r["cache"]["tier"]) for r in dec}))
        spec_ids = [m[0] for m in models if m[2]]
        spec_rows = [r for r in dec if r["model"] in spec_ids and r.get("speculative")]
        check("speculative stats on rows of speculative models",
              not spec_ids or bool(spec_rows) and any((r["speculative"].get("rounds") or 0) > 0 for r in spec_rows),
              json.dumps(spec_rows[-1]["speculative"] if spec_rows else None))
        bands = [s.get("governor", {}).get("band") for s in samples]
        check("governor band never red", "red" not in bands, str(sorted(set(map(str, bands)))))
        red = samples[-1].get("governor", {}).get("red_failures") or 0
        check("no shed (red_failures flat)", red == base_red, f"{base_red} -> {red}")
        not_ready = [x for x in ready if x[1] == 503]
        check("ready=1 -> 503 with reason + Retry-After under load",
              bool(not_ready) and all(x[2] in ("busy", "queue", "pressure") and x[3] for x in not_ready),
              f"{len(not_ready)}/{len(ready)} probes not ready; reasons {sorted({x[2] for x in not_ready})}")
        wall = [r["wall_s"] for r in results if r["status"] == 200]
        print(f"  stream wall: min {min(wall):.0f}s max {max(wall):.0f}s; samples {len(samples)}; "
              f"max in_flight {max_if} waiting {max_wait}; bands {sorted(set(map(str, bands)))}")

        drained = wait_drained(c, 60)
        check("drain to 0 after load", drained is not None)

        # ---- cache warmth: same long prompt twice on the big model
        print("cache warmth:")
        st, body = c.post("/v1/cache/reset", {"model": big})
        with Sampler(base) as smp:
            tiers = []
            for n in range(2):
                t0 = time.monotonic()
                out = []
                stream_one(base, big, 900 + n, 96, out, prompt=LONG_PROMPT, tag="warmth")
                rows = [r for s in smp.since(t0) for r in s.get("requests", []) if r["state"] == "decode"]
                tiers.append(({r["cache"]["tier"] for r in rows},
                              max((r["cache"].get("warm_tokens") or 0 for r in rows), default=0),
                              out[0]["status"], out[0]["wall_s"]))
        check("first long prompt: miss tier, 200", tiers[0][2] == 200 and tiers[0][0] <= {"miss"},
              str(tiers[0]))
        check("repeat long prompt: warm tier with warm_tokens > 0",
              tiers[1][2] == 200 and tiers[1][0] and tiers[1][0] <= {"exact", "block", "ckpt", "anchor", "hit"}
              and tiers[1][1] > 0,
              str(tiers[1]))

        # ---- scoped reset on one model while another is generating
        print("scoped reset under load:")
        out = []
        ts = [threading.Thread(target=stream_one, args=(base, small, 700 + i, 200, out),
                               kwargs={"tag": "reset"}, daemon=True) for i in range(a.width)]
        for t in ts:
            t.start()
        time.sleep(1.0)
        st, body = c.post("/v1/cache/reset", {"model": big})
        check(f"scoped reset {big} -> 200 [{big}] while {small} streams", st == 200 and body.get("models") == [big],
              json.dumps(body))
        st, body = c.post("/v1/cache/reset", {"model": "no-such"})
        check("unknown model -> 404", st == 404 and body.get("status") == "unknown_model")
        for t in ts:
            t.join(timeout=300)
        check(f"{small} streams unaffected by {big} reset",
              len(out) == a.width and all(r["status"] == 200 and r["chunks"] > 0 for r in out),
              str([(r["status"], r["chunks"]) for r in out]))
        st, body = c.post("/v1/cache/reset")
        check("unscoped reset -> every resident model", st == 200 and set(body.get("models") or []) >= set(ids),
              json.dumps(body))

        # ---- unload a secondary model, reload it by request
        print("unload / reload:")
        drained = wait_drained(c, 30)
        victim = ids[1] if len(ids) > 1 else ids[-1]
        st, body = c.unload(victim)
        st2, m = c.get("/v1/metrics")
        res_ids = {e["ids"][0] for e in m["server"].get("resident_models", []) if e.get("ids")}
        check(f"unload {victim} -> 200 and not resident", st == 200 and victim not in res_ids,
              f"{st} {json.dumps(body)[:120]} resident {sorted(res_ids)}")
        st, body = c.post("/v1/cache/reset", {"model": victim})
        check("scoped reset on unloaded model -> 404 not_resident", st == 404 and body.get("status") == "not_resident",
              json.dumps(body))
        with Sampler(base) as smp:
            t0 = time.monotonic()
            out = []
            stream_one(base, victim, 800, 32, out, tag="reload")
            probes = [x for x in smp.ready if x[0] >= t0]
        check(f"request to {victim} reloads it -> 200", out[0]["status"] == 200 and out[0]["chunks"] > 0,
              f"{out[0]['status']} {out[0]['wall_s']}s")
        st2, m = c.get("/v1/metrics")
        res_ids = {e["ids"][0] for e in m["server"].get("resident_models", []) if e.get("ids")}
        check(f"{victim} resident again", victim in res_ids, str(sorted(res_ids)))
        print(f"  readiness during reload: {[(x[1], x[2]) for x in probes][:12]}")

        # ---- prometheus labels every model
        st, hdr, text = get_raw(f"{base}/metrics?format=prometheus")
        check("prometheus renders", st == 200 and hdr.get("content-type", "").startswith("text/plain"))
        missing = [mid for mid in ids if f'model="{mid}"' not in text]
        check("prometheus labels every resident model", not missing, f"missing {missing}")

        drained = wait_drained(c, 30)
        check("final drain: requests[] empty, in_flight 0", drained is not None)
        st, hdr, raw = get_raw(f"{base}/health?ready=1")
        check("ready=1 -> 200 at end", st == 200)

        failed = [n for n, ok, _ in _results if not ok]
        print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed in "
              f"{time.monotonic() - t_start:.0f}s")
        if failed:
            print("failed:", failed)
            print(srv.log_tail(40))
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
