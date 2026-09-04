#!/usr/bin/env python3
"""Serve-path crash hunt: seeded concurrent chaos against one live server.

Boots the target (APC armed, MTP on request) and runs N worker threads
firing a randomized mix of the seams that have produced crashes: client
aborts mid-stream and mid-prefill, tiny budgets at finish seams, exact
warm resends and sibling bursts through APC admission, growing session
chains with compaction-style rewrites, sampler and seed variety, and
per-request chat_template_kwargs flips. Content quality is not scored;
the pass criterion is purely mechanical: every request returns clean
HTTP, every stream parses, the server log stays free of exceptions, and
the process survives. Every request is journaled with its parameters so
a finding is replayable.

Usage: ./run_serve_stress_e2e.py --model M.gguf --speculative \
           --minutes 20 --clients 6 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_apc_depth_e2e import (  # noqa: E402
    deep_prefix,
    depth_env,
    model_id_of,
    serve_args,
)
from run_apc_disk_e2e import kv_engaged, kv_engagement  # noqa: E402
from server_proc import ServerProc  # noqa: E402

LOG_BAD = re.compile(
    r"Traceback|ERROR|CRITICAL|bad_cast|NotImplementedError|Segmentation|"
    r"Fatal|panic|assert", re.IGNORECASE)
# Lines the server emits at ERROR / WARNING level for CLIENT behavior the
# chaos deliberately causes (mid-stream disconnects); not findings.
# ``stream_closed_before_completion`` is mlx-vlm's record of a stream the
# client closed early ("Request failed: ... error=..." - the "error=" token
# is what LOG_BAD matches).
LOG_BENIGN = re.compile(
    r"ClientDisconnect|Broken pipe|Connection reset|ConnectionResetError|"
    r"stream_closed_before_completion")

PREFIX_SIZES = (800, 2000, 4000, 6000)


def build_prefixes():
    """Distinct deep prefixes: deep_prefix is deterministic, so salt each
    with a site tag line to keep APC chains distinct across sizes."""
    out = []
    for i, words in enumerate(PREFIX_SIZES):
        text, n = deep_prefix(words)
        out.append((f"Site tag: node-{i}.\n" + text, n))
    return out


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.actions = {}
        self.findings = []
        self.anomalies = []

    def count(self, action):
        with self.lock:
            self.actions[action] = self.actions.get(action, 0) + 1

    def finding(self, rec):
        with self.lock:
            self.findings.append(rec)
            print(f"  FINDING  {json.dumps(rec)[:300]}", flush=True)

    def anomaly(self, rec):
        with self.lock:
            self.anomalies.append(rec)


class Journal:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.fh = open(path, "w")

    def write(self, rec):
        with self.lock:
            self.fh.write(json.dumps(rec) + "\n")
            self.fh.flush()


def post_chat(base, body, timeout):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def plain_chat(base, body, timeout):
    """Non-streamed request. Returns (status, content, err)."""
    try:
        with post_chat(base, body, timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
            msg = payload["choices"][0]["message"]
            return resp.status, str(msg.get("content") or ""), None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:400], "http"
    except Exception as e:  # noqa: BLE001 - every failure is data here
        return -1, "", f"{type(e).__name__}: {e}"


def stream_chat(base, body, timeout, abort_after=None):
    """Streamed request. Returns (status, events, err). abort_after=0
    closes right after connect (mid-prefill cancel); N closes after N
    SSE events (mid-decode cancel)."""
    body = dict(body)
    body["stream"] = True
    events = 0
    try:
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if abort_after == 0:
                return resp.status, 0, None
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                json.loads(payload)  # malformed SSE JSON is a finding
                events += 1
                if abort_after is not None and events >= abort_after:
                    break
            return resp.status, events, None
    except urllib.error.HTTPError as e:
        return e.code, events, "http"
    except json.JSONDecodeError as e:
        return -2, events, f"sse-json: {e}"
    except Exception as e:  # noqa: BLE001
        return -1, events, f"{type(e).__name__}: {e}"


def worker(wid, args, base, mid, prefixes, stop_at, stats, journal, rng,
           warm_pool, warm_lock):
    sessions = []
    i = 0
    while time.monotonic() < stop_at:
        i += 1
        action = rng.choices(
            ["cold", "warm", "sibling", "session", "stream", "abort",
             "tiny", "sampled", "kwargs"],
            weights=[14, 10, 10, 16, 14, 16, 6, 8, 6])[0]
        stats.count(action)
        prefix, n = prefixes[rng.randrange(len(prefixes))]
        e = rng.randrange(1, n)
        q = (f"Question: what reading did entry {e} report and who logged "
             "it? Answer with the number and the name.")
        body = {"model": mid, "temperature": 0.0,
                "max_tokens": rng.choice([64, 128, 256, 512])}
        sys_msg = {"role": "system",
                   "content": "You are the maintenance log assistant."}
        stream = False
        abort_after = None

        if action == "cold":
            body["messages"] = [sys_msg,
                                {"role": "user", "content": prefix + q}]
            with warm_lock:
                warm_pool.append(body["messages"])
                del warm_pool[:-8]
        elif action == "warm":
            with warm_lock:
                msgs = rng.choice(warm_pool) if warm_pool else None
            if msgs is None:
                continue
            body["messages"] = msgs
        elif action == "sibling":
            tag = rng.randrange(1000)
            body["messages"] = [
                sys_msg,
                {"role": "user",
                 "content": prefix + f"Sibling {tag}. " + q}]
        elif action == "session":
            if sessions and (len(sessions) >= 3 or rng.random() < 0.7):
                s = rng.choice(sessions)
                if len(s) > 12 and rng.random() < 0.3:
                    # compaction-style rewrite: drop the middle
                    del s[2:len(s) - 2]
                    s.insert(2, {"role": "assistant",
                                 "content": "Summary: readings nominal."})
                s.append({"role": "user", "content": q})
                body["messages"] = list(s)
            else:
                s = [sys_msg, {"role": "user", "content": prefix + q}]
                sessions.append(s)
                body["messages"] = list(s)
                sessions[:] = sessions[-4:]
        elif action == "stream":
            stream = True
            body["messages"] = [sys_msg,
                                {"role": "user", "content": prefix + q}]
        elif action == "abort":
            stream = True
            abort_after = rng.choice([0, 1, 3, 8, 20, 40])
            body["messages"] = [sys_msg,
                                {"role": "user", "content": prefix + q}]
            body["max_tokens"] = 1024
        elif action == "tiny":
            body["max_tokens"] = rng.randrange(1, 5)
            body["messages"] = [sys_msg,
                                {"role": "user", "content": prefix + q}]
        elif action == "sampled":
            body.update(temperature=round(rng.uniform(0.5, 1.2), 2),
                        top_p=round(rng.uniform(0.8, 1.0), 2),
                        seed=rng.randrange(1 << 30))
            if rng.random() < 0.5:
                body["top_k"] = rng.choice([20, 40, 80])
            if rng.random() < 0.3:
                body["min_p"] = 0.05
            body["messages"] = [sys_msg,
                                {"role": "user", "content": prefix + q}]
        elif action == "kwargs":
            body["chat_template_kwargs"] = {
                "enable_thinking": rng.random() < 0.5}
            body["messages"] = [sys_msg,
                                {"role": "user", "content": prefix + q}]

        t0 = time.monotonic()
        if stream:
            st, events, err = stream_chat(base, body, args.request_timeout,
                                          abort_after=abort_after)
            wall = time.monotonic() - t0
            rec = {"wid": wid, "i": i, "action": action, "status": st,
                   "t": round(time.time(), 1),
                   "events": events, "wall": round(wall, 1), "err": err,
                   "abort_after": abort_after,
                   "max_tokens": body["max_tokens"]}
        else:
            st, content, err = plain_chat(base, body, args.request_timeout)
            wall = time.monotonic() - t0
            rec = {"wid": wid, "i": i, "action": action, "status": st,
                   "t": round(time.time(), 1),
                   "wall": round(wall, 1), "err": err,
                   "max_tokens": body["max_tokens"],
                   "n_msgs": len(body["messages"])}
            if st == 200 and not content and action != "tiny":
                stats.anomaly({**rec, "note": "empty content"})
        journal.write(rec)
        # An aborted stream reporting a transport error is the abort
        # itself; anything else non-200 is a finding.
        if action == "abort" and rec["err"] and rec["status"] in (-1, 200):
            continue
        if rec["status"] != 200 or rec["err"]:
            stats.finding(rec)


def scan_log(path, stats, seen):
    """Scan the server log for new exception lines; benign client-side
    disconnect noise is excluded."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return
    for ln, line in enumerate(lines):
        if ln in seen:
            continue
        if LOG_BAD.search(line) and not LOG_BENIGN.search(line):
            seen.add(ln)
            stats.finding({"server_log_line": ln + 1,
                           "text": line.strip()[:300]})
        elif LOG_BAD.search(line):
            seen.add(ln)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--speculative", action="store_true")
    ap.add_argument("--draft-gguf", default=None)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--clients", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--request-timeout", type=float, default=420.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--scheme", default=None,
                    help="KV_QUANT_SCHEME for the server (default fp16 KV)")
    ap.add_argument("--bits", default=None,
                    help="KV width: KV_BITS, or k6v5-style GMLX_KVARN_BITS")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    label = os.path.basename(args.out.rstrip("/"))
    log_path = os.path.join(args.out, "server.log")
    env = depth_env(os.path.join(args.out, "apc-disk"), 16, 2048, 16,
                    scheme=args.scheme, bits=args.bits)

    print(f"== {label}: building prefixes ==", flush=True)
    prefixes = build_prefixes()
    stats = Stats()
    journal = Journal(os.path.join(args.out, "journal.jsonl"))
    seen_log = set()

    with ServerProc(
        serve_args(args.model, args.speculative, args.draft_gguf),
        env_extra=env,
        log_path=log_path,
        python=args.python,
    ) as srv:
        srv.wait_ready(timeout=900)
        base = srv.base_url
        mid = model_id_of(base, srv)
        # One request before the workers: the server reports kv_quant
        # once the model is resident, and a scheme that fell back to fp16
        # would make the whole run measure the wrong cache.
        plain_chat(base, {"model": mid, "max_tokens": 4, "temperature": 0.0,
                          "messages": [{"role": "user", "content": "Say ok."}]},
                   args.request_timeout)
        kq = kv_engagement(base, mid)
        if not kv_engaged(kq, args.scheme, args.bits):
            # skip the chaos loop; the report still carries the finding
            stats.finding({"kv_not_engaged": kq})
            print(f"== {label}: KV scheme did not engage: {kq} ==", flush=True)
            args.minutes = 0.0
        print(f"== {label}: kv_quant {kq or 'fp16'} ==", flush=True)
        print(f"== {label}: chaos start, {args.clients} clients x "
              f"{args.minutes:.0f} min, seed {args.seed} ==", flush=True)
        stop_at = time.monotonic() + args.minutes * 60
        warm_pool, warm_lock = [], threading.Lock()
        threads = [
            threading.Thread(
                target=worker,
                args=(w, args, base, mid, prefixes, stop_at, stats, journal,
                      random.Random(args.seed * 1000 + w), warm_pool,
                      warm_lock),
                daemon=True)
            for w in range(args.clients)
        ]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            time.sleep(10)
            scan_log(log_path, stats, seen_log)
            if srv.proc.poll() is not None:
                stats.finding({"server_died": srv.proc.returncode})
                break
        for t in threads:
            t.join(timeout=args.request_timeout + 30)
        # settle, then final probe + log sweep
        time.sleep(3)
        st, content, err = plain_chat(
            base, {"model": mid, "max_tokens": 8, "temperature": 0.0,
                   "messages": [{"role": "user", "content": "Say ok."}]},
            60)
        if st != 200:
            stats.finding({"post_chaos_probe": st, "err": err})
        scan_log(log_path, stats, seen_log)
        if srv.proc.poll() is not None:
            stats.finding({"server_died": srv.proc.returncode})

    total = sum(stats.actions.values())
    report = {
        "label": label,
        "model": args.model,
        "speculative": args.speculative,
        "minutes": args.minutes,
        "clients": args.clients,
        "seed": args.seed,
        "kv_quant": kq,
        "requests": total,
        "actions": stats.actions,
        "findings": stats.findings,
        "anomalies": stats.anomalies[:50],
        "n_anomalies": len(stats.anomalies),
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"  requests {total}  actions {json.dumps(stats.actions)}")
    print(f"  anomalies {len(stats.anomalies)} (empty-content notes)")
    verdict = "PASS" if not stats.findings else \
        f"FAIL ({len(stats.findings)} findings)"
    print(f"== {label}: {verdict} -> {args.out}/report.json ==", flush=True)
    return 0 if not stats.findings else 1


if __name__ == "__main__":
    sys.exit(main())
