#!/usr/bin/env python3
"""Multi-cycle capacity soak: the failure classes the capacity surface has
produced, mixed under seeded chaos against one live server for a set
number of cycles, with the live-request metrics checked for invariants on
every tick.

Per client, a randomized mix of: cold deep-prefix prompts, exact warm
resends, sibling prefixes, growing session chains (answers appended
verbatim, compaction-style rewrites), streams aborted mid-prefill /
mid-decode, tiny budgets, sampler variety, thinking flips, dry-run
estimates (``/v1/estimate`` and ``dry_run`` on chat, including a prompt
far past the context and the continuation of an answered session, which
exact-tier models should report warm), capacity plan and readiness
probes, staggered same-client bursts past the queue cap (typed 503s; a
soak whose bursts never draw one is a finding), and - on a multi-model
config - requests for a model that is
not resident (load-by-request, typed 503 while the resident one is busy,
LRU switching once idle). A controller samples ``/v1/metrics`` every
second and, once per cycle, fires the operator actions: scoped and
unscoped cache resets under load, unload of an idle model, a Prometheus
render.

Pass criterion is mechanical: every response is either a success or one
of the contract's typed refusals (queue-cap 503 + Retry-After, 503
``model_load_deferred``, a governor shed), every stream parses, the
metrics invariants hold on every sample (including no stale waiting
census at idle), the server log carries no
exception outside the shed / client-abort classes, and the process
survives. Sheds are tallied, not failed: they are the governor's verdict
on the box, reported so the operator can judge the configuration.

Usage: python tests/e2e/run_capacity_soak_e2e.py \\
           --models ID=PATH[:spec|:draft=PATH] [ID2=PATH...] \\
           --cycles 4 --cycle-minutes 5 --clients 6 --out DIR
Exit 0 on pass, 1 on findings, 2 when a model file is missing.
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
from client import Client  # noqa: E402
from run_apc_depth_e2e import deep_prefix  # noqa: E402
from server_proc import ServerProc  # noqa: E402

LOG_BAD = re.compile(r"Traceback|ERROR|CRITICAL|bad_cast|NotImplementedError|"
                     r"Segmentation|Fatal|panic|assert", re.IGNORECASE)
# Client-caused and governor-caused lines: tallied, never findings.
LOG_BENIGN = re.compile(
    r"ClientDisconnect|Broken pipe|Connection reset|ConnectionResetError|"
    r"stream_closed_before_completion|shed under memory pressure|RowShedError|"
    r"governor red|governor orange|\[tick-guard\]|kernel floor|"
    r"request shed|row_failed|model load deferred|Error loading model in generation thread")
SHED_RE = re.compile(r"shed under memory pressure|governor (red|orange)")
PREFIX_SIZES = (600, 1500, 3000, 5000)
STATES = {"queued", "prefill", "decode"}
BANDS = {"green", "yellow", "orange", "red", None}


def build_prefixes():
    out = []
    for i, words in enumerate(PREFIX_SIZES):
        text, n = deep_prefix(words)
        out.append((f"Site tag: soak-{i}.\n" + text, n))
    return out


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.actions: dict = {}
        self.statuses: dict = {}
        self.findings: list = []
        self.sheds = 0
        self.typed: dict = {}
        self.walls: list = []
        self.samples = 0
        self.invariant_failures: list = []
        self.ghost_streak = 0

    def count(self, key, d=None):
        with self.lock:
            d = self.actions if d is None else d
            d[key] = d.get(key, 0) + 1

    def finding(self, rec):
        with self.lock:
            self.findings.append(rec)
        print(f"  FINDING  {json.dumps(rec)[:320]}", flush=True)

    def shed(self):
        with self.lock:
            self.sheds += 1


class Journal:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.fh = open(path, "w")

    def write(self, rec):
        with self.lock:
            self.fh.write(json.dumps(rec) + "\n")
            self.fh.flush()


def _post(base, path, body, timeout):
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def plain_chat(base, body, timeout):
    """(status, payload-or-text, err)."""
    try:
        with _post(base, "/v1/chat/completions", body, timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace")), None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw[:400]
        return e.code, payload, "http"
    except Exception as e:  # noqa: BLE001
        return -1, "", f"{type(e).__name__}: {e}"


def stream_chat(base, body, timeout, abort_after=None):
    """(status, events, terminal, err). terminal: 'done' | 'shed' | 'error:<msg>' | None."""
    body = dict(body)
    body["stream"] = True
    events, terminal = 0, None
    try:
        req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Accept": "text/event-stream"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if abort_after == 0:
                return resp.status, 0, None, None
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    terminal = terminal or "done"
                    break
                obj = json.loads(payload)                # malformed SSE is a finding
                if isinstance(obj, dict) and "error" in obj:
                    msg = obj["error"]
                    msg = msg.get("message") if isinstance(msg, dict) else str(msg)
                    terminal = "shed" if SHED_RE.search(str(msg)) else f"error:{str(msg)[:200]}"
                    break
                events += 1
                if abort_after is not None and events >= abort_after:
                    break
            return resp.status, events, terminal, None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw[:300]
        return e.code, events, payload, "http"
    except json.JSONDecodeError as e:
        return -2, events, None, f"sse-json: {e}"
    except Exception as e:  # noqa: BLE001
        return -1, events, None, f"{type(e).__name__}: {e}"


def classify(status, payload, headers=None):
    """'ok' | 'shed' | 'queue_cap' | 'load_deferred' | 'context' | 'finding'."""
    if status == 200:
        return "ok"
    msg, typ = "", ""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg, typ = str(err.get("message", "")), str(err.get("type", ""))
        elif err is not None:
            msg = str(err)
        else:
            msg = str(payload.get("detail") or payload.get("message") or "")
    else:
        msg = str(payload)
    if typ == "model_load_deferred":
        return "load_deferred"
    if status == 503 and ("queue" in msg.lower() or "capacity" in msg.lower()
                          or "busy" in msg.lower() or "Retry" in msg):
        return "queue_cap"
    if SHED_RE.search(msg) or "shed" in msg.lower():
        return "shed"
    if status == 400 and ("context" in msg.lower() or "too long" in msg.lower()
                          or "exceed" in msg.lower()):
        return "context"
    return "finding"


def worker(wid, args, base, ids, prefixes, stop_at, stats, journal, rng, shared):
    sessions: list = []
    i = 0
    multi = len(ids) > 1
    while time.monotonic() < stop_at:
        i += 1
        weights = {"cold": 12, "warm": 8, "sibling": 8, "session": 12, "stream": 12,
                   "abort": 12, "tiny": 4, "sampled": 6, "kwargs": 4, "estimate": 8,
                   "estimate_long": 3, "estimate_session": 4, "plan": 4, "ready": 4,
                   "burst": 5}
        if multi:
            weights["cross"] = 8
        action = rng.choices(list(weights), weights=list(weights.values()))[0]
        stats.count(action)
        # the model this client speaks to: the primary unless crossing
        mid = ids[0]
        if multi and action == "cross":
            mid = rng.choice(ids[1:])
        prefix, n = prefixes[rng.randrange(len(prefixes))]
        e = rng.randrange(1, n)
        q = (f"Question: what reading did entry {e} report and who logged it? "
             "Answer with the number and the name.")
        sys_msg = {"role": "system", "content": "You are the maintenance log assistant."}
        body = {"model": mid, "temperature": 0.0, "max_tokens": rng.choice([48, 96, 192, 384])}
        stream, abort_after = False, None
        sess = None                      # the session list a 200 answer extends
        if action == "estimate_session":
            long_sessions = [x for x in sessions if len(x) >= 3]
            if long_sessions:
                sess = rng.choice(long_sessions)
            else:
                action = "estimate"      # no answered session yet
                stats.count("estimate_session_fallback", stats.typed)
        rec = {"wid": wid, "i": i, "action": action, "model": mid, "t": round(time.time(), 1)}
        t0 = time.monotonic()

        if action in ("cold", "cross", "tiny", "sampled", "kwargs", "stream", "abort"):
            body["messages"] = [sys_msg, {"role": "user", "content": prefix + q}]
            if action == "cold":
                with shared["lock"]:
                    shared["warm"].append(body["messages"])
                    del shared["warm"][:-8]
            elif action == "tiny":
                body["max_tokens"] = rng.randrange(1, 5)
            elif action == "sampled":
                body.update(temperature=round(rng.uniform(0.5, 1.2), 2),
                            top_p=round(rng.uniform(0.8, 1.0), 2), seed=rng.randrange(1 << 30))
                if rng.random() < 0.5:
                    body["top_k"] = rng.choice([20, 40, 80])
            elif action == "kwargs":
                body["chat_template_kwargs"] = {"enable_thinking": rng.random() < 0.5}
            elif action == "stream":
                stream = True
            elif action == "abort":
                stream, abort_after = True, rng.choice([0, 1, 3, 8, 20, 40])
                body["max_tokens"] = 768
        elif action == "warm":
            with shared["lock"]:
                msgs = rng.choice(shared["warm"]) if shared["warm"] else None
            if msgs is None:
                continue
            body["messages"] = msgs
        elif action == "sibling":
            body["messages"] = [sys_msg, {"role": "user",
                                          "content": prefix + f"Sibling {rng.randrange(1000)}. " + q}]
        elif action == "session":
            # a growing chat: every answered turn is appended verbatim, so
            # the next turn is a continuation of a retired row (the shape
            # the exact tier serves); occasional compaction rewrites the
            # middle the way an agent harness does
            if sessions and (len(sessions) >= 3 or rng.random() < 0.7):
                s = rng.choice(sessions)
                if len(s) > 12 and rng.random() < 0.3:
                    del s[2:len(s) - 2]
                    s.insert(2, {"role": "assistant", "content": "Summary: readings nominal."})
                if s[-1].get("role") == "user":
                    s.pop()              # last turn got no answer; replace it
                s.append({"role": "user", "content": q})
                body["messages"] = list(s)
            else:
                s = [sys_msg, {"role": "user", "content": prefix + q}]
                sessions.append(s)
                body["messages"] = list(s)
                sessions[:] = sessions[-4:]
            sess = s
        elif action in ("estimate", "estimate_long", "estimate_session"):
            if action == "estimate_long":
                body["messages"] = [sys_msg, {"role": "user", "content": (prefix + "\n") * 40 + q}]
            elif action == "estimate_session":
                # continuation of an answered session: exact-tier models
                # (retired prompt+answer rows) should report a warm prefix
                body["messages"] = list(sess) + [{"role": "user", "content": q}]
            else:
                body["messages"] = [sys_msg, {"role": "user", "content": prefix + q}]
            if rng.random() < 0.5:
                st, payload, err = plain_chat(base, dict(body, dry_run=True), args.request_timeout)
                rec["via"] = "dry_run"
            else:
                try:
                    with _post(base, "/v1/estimate", body, args.request_timeout) as resp:
                        st, payload, err = resp.status, json.loads(resp.read().decode()), None
                except urllib.error.HTTPError as ex:
                    st, payload, err = ex.code, ex.read().decode()[:300], "http"
                except Exception as ex:  # noqa: BLE001
                    st, payload, err = -1, "", f"{type(ex).__name__}: {ex}"
                rec["via"] = "estimate"
            rec.update(status=st, err=err, wall=round(time.monotonic() - t0, 1))
            kind = classify(st, payload)
            if kind == "queue_cap":
                stats.count("queue_cap", stats.typed)
            elif st == 200 and isinstance(payload, dict):
                ok = ("resident" in payload and "prompt_tokens" in payload
                      and "fits_now" in payload and "context_ok" in payload)
                if payload.get("resident") and not isinstance(payload.get("prompt_tokens"), int):
                    ok = False
                if action == "estimate_long" and payload.get("resident") \
                        and payload.get("context_ok") is not False \
                        and payload.get("context_limit"):
                    if payload["prompt_tokens"] and payload["prompt_tokens"] > payload["context_limit"]:
                        ok = False
                if not ok:
                    stats.finding({**rec, "note": "estimate shape", "payload": str(payload)[:300]})
                rec["resident"] = payload.get("resident")
                rec["prompt_tokens"] = payload.get("prompt_tokens")
                rec["warm"] = payload.get("warm_tokens")
                rec["tier"] = payload.get("cache_tier")
                if action == "estimate_session":
                    stats.count("estimate_session_warm" if rec["warm"] else
                                "estimate_session_cold", stats.typed)
            else:
                stats.finding({**rec, "payload": str(payload)[:300]})
            journal.write(rec)
            stats.count(str(st), stats.statuses)
            continue
        elif action == "plan":
            w, d = rng.choice([1, 2, 4, 8]), rng.choice([1024, 4096, 16384, 65536, 200000])
            try:
                with urllib.request.urlopen(f"{base}/v1/capacity/plan?width={w}&depth={d}",
                                            timeout=30) as resp:
                    st, payload = resp.status, json.loads(resp.read().decode())
            except urllib.error.HTTPError as ex:
                st, payload = ex.code, ex.read().decode()[:200]
            except Exception as ex:  # noqa: BLE001
                st, payload = -1, f"{type(ex).__name__}: {ex}"
            rec.update(status=st, wall=round(time.monotonic() - t0, 1))
            ok = st == 200 and isinstance(payload, dict) and "admit_now" in payload \
                and payload.get("band") in BANDS and "reason" in payload
            if ok and payload.get("admit_now") and (payload.get("band") in ("orange", "red")
                                                    or (payload.get("slots") or 0) < w):
                ok = False
            if not ok:
                stats.finding({**rec, "note": "plan shape", "payload": str(payload)[:300]})
            journal.write(rec)
            continue
        elif action == "ready":
            try:
                with urllib.request.urlopen(f"{base}/health?ready=1", timeout=30) as resp:
                    st, payload = resp.status, json.loads(resp.read().decode())
            except urllib.error.HTTPError as ex:
                st, payload = ex.code, json.loads(ex.read().decode() or "{}")
            except Exception as ex:  # noqa: BLE001
                st, payload = -1, {"err": f"{type(ex).__name__}: {ex}"}
            rec.update(status=st, wall=round(time.monotonic() - t0, 1), reason=payload.get("reason"))
            ok = (st == 200 and payload.get("ready") is True) or \
                 (st == 503 and payload.get("reason") in ("busy", "queue", "pressure"))
            if not ok:
                stats.finding({**rec, "note": "ready shape", "payload": str(payload)[:200]})
            journal.write(rec)
            continue
        elif action == "burst":
            k = args.width + args.cap + 1          # past the queue cap: typed 503s expected
            outs: list = []

            def one(j, outs=outs):
                b = {"model": mid, "temperature": 0.0, "max_tokens": 64, "seed": j,
                     "messages": [sys_msg, {"role": "user", "content": prefix + f"Burst {j}. " + q}]}
                st_, ev_, term_, err_ = stream_chat(base, b, args.request_timeout)
                outs.append((st_, ev_, term_, err_))

            ts = [threading.Thread(target=one, args=(j,), daemon=True) for j in range(k)]
            for t in ts:
                t.start()
                time.sleep(0.2)          # let the queue build: the cap is judged at arrival
            for t in ts:
                t.join(timeout=args.request_timeout + 30)
            rec.update(wall=round(time.monotonic() - t0, 1), n=len(outs),
                       statuses=[o[0] for o in outs],
                       n503=sum(1 for o in outs if o[0] == 503))
            for st_, ev_, term_, err_ in outs:
                kind = classify(st_, term_ if st_ != 200 else None)
                if st_ == 200:
                    if term_ == "shed":
                        stats.shed()
                    elif isinstance(term_, str) and term_.startswith("error:"):
                        stats.finding({**rec, "note": "stream error", "terminal": term_})
                elif kind in ("queue_cap", "load_deferred"):
                    stats.count(kind, stats.typed)
                elif kind == "shed":
                    stats.shed()
                else:
                    stats.finding({**rec, "note": "burst member", "status": st_, "err": err_,
                                   "payload": str(term_)[:200]})
            journal.write(rec)
            continue

        if stream:
            st, events, terminal, err = stream_chat(base, body, args.request_timeout,
                                                    abort_after=abort_after)
            wall = time.monotonic() - t0
            rec.update(status=st, events=events, wall=round(wall, 1), err=err,
                       abort_after=abort_after, max_tokens=body["max_tokens"],
                       terminal=terminal if isinstance(terminal, str) else None)
            journal.write(rec)
            stats.count(str(st), stats.statuses)
            if st == 200:
                stats.walls.append(wall)
                if terminal == "shed":
                    stats.shed()
                elif isinstance(terminal, str) and terminal.startswith("error:"):
                    if action == "abort":
                        continue
                    stats.finding({**rec, "note": "stream error"})
                elif err and action != "abort":
                    stats.finding(rec)
                continue
            kind = classify(st, terminal)
        else:
            st, payload, err = plain_chat(base, body, args.request_timeout)
            wall = time.monotonic() - t0
            rec.update(status=st, wall=round(wall, 1), err=err, max_tokens=body["max_tokens"],
                       n_msgs=len(body["messages"]))
            if st == 200 and isinstance(payload, dict):
                try:
                    content = payload["choices"][0]["message"].get("content") or ""
                    rec["finish"] = payload["choices"][0].get("finish_reason")
                    rec["gen"] = (payload.get("usage") or {}).get("completion_tokens")
                    rec["cached"] = ((payload.get("usage") or {}).get("prompt_tokens_details")
                                     or {}).get("cached_tokens")
                    if not content and action != "tiny" and rec.get("finish") != "length":
                        rec["empty"] = True
                    if rec.get("cached"):
                        stats.count(f"{action}:cached", stats.typed)
                    if sess is not None and content:
                        sess.append({"role": "assistant", "content": content})
                except Exception:
                    stats.finding({**rec, "note": "response shape", "payload": str(payload)[:200]})
            journal.write(rec)
            stats.count(str(st), stats.statuses)
            if st == 200:
                stats.walls.append(wall)
                continue
            kind = classify(st, payload)
        if kind in ("queue_cap", "load_deferred"):
            stats.count(kind, stats.typed)
            if kind == "load_deferred" and not multi:
                stats.finding({**rec, "note": "load deferred on a single-model server"})
        elif kind == "shed":
            stats.shed()
        elif kind == "context":
            stats.count("context_refusal", stats.typed)
        elif action == "abort" and st in (-1, 200):
            continue
        else:
            stats.finding({**rec, "payload": str(payload if not stream else terminal)[:300]})


def check_sample(s: dict, ids: list, width: int, stats: Stats) -> None:
    """Metrics invariants on one /v1/metrics sample."""
    bad = []
    rows = s.get("requests") or []
    conc = s.get("concurrency") or {}
    by_model: dict = {}
    for r in rows:
        if r.get("state") not in STATES:
            bad.append(f"state {r.get('state')!r}")
        if r.get("model") not in ids:
            bad.append(f"row model {r.get('model')!r}")
        if r.get("state") == "queued" and not isinstance(r.get("position"), int):
            bad.append("queued row without position")
        if r.get("state") == "decode":
            by_model[r.get("model")] = by_model.get(r.get("model"), 0) + 1
    for m, n in by_model.items():
        if n > width:
            bad.append(f"{m}: {n} decode rows > width {width}")
    res = s.get("resident_models") or []
    res_if = sum(int(e.get("in_flight") or 0) for e in res)
    if isinstance(conc.get("in_flight"), int) and res_if != conc["in_flight"]:
        bad.append(f"resident in_flight {res_if} != concurrency.in_flight {conc['in_flight']}")
    if isinstance(conc.get("waiting"), int) and conc["waiting"] < 0:
        bad.append("negative waiting")
    # a waiting census with no live rows and nothing in flight is a stale
    # count (it makes readiness 503 and, with the cap, rejects at idle)
    if isinstance(conc.get("waiting"), int) and conc["waiting"] > 0 \
            and not rows and conc.get("in_flight") == 0:
        stats.ghost_streak += 1
        if stats.ghost_streak == 3:
            bad.append(f"waiting {conc['waiting']} with no rows and in_flight 0 for 3 samples")
    else:
        stats.ghost_streak = 0
    rates = s.get("rates") or {}
    if not {"decode_tok_s", "decode_streams"} <= set(rates):
        bad.append("rates missing")
    elif rates.get("decode_streams", 0) > width * max(1, len(res)):
        bad.append(f"rates.decode_streams {rates.get('decode_streams')} > width x models")
    gov = s.get("governor") or {}
    if gov.get("band") not in BANDS:
        bad.append(f"band {gov.get('band')!r}")
    if bad:
        stats.invariant_failures.append({"t": round(time.time(), 1), "bad": bad})
        if len(stats.invariant_failures) <= 10:
            stats.finding({"invariant": bad, "sample": json.dumps(
                {"concurrency": conc, "rows": [(r.get("model"), r.get("state")) for r in rows][:12],
                 "resident": [(e.get("ids"), e.get("in_flight")) for e in res]})[:300]})


def scan_log(path, stats, seen):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return
    for ln, line in enumerate(lines):
        if ln in seen:
            continue
        if LOG_BAD.search(line):
            seen.add(ln)
            if LOG_BENIGN.search(line):
                stats.count("log_benign", stats.typed)
            elif line.startswith(("  File", "    ", "Traceback")):
                # traceback body lines: attributed to the exception line that
                # follows / precedes; only the exception line itself is judged
                continue
            else:
                stats.finding({"server_log_line": ln + 1, "text": line.strip()[:300]})


def controller(args, base, c: Client, ids, stop_at, stats, journal, log_path, srv, rng):
    seen: set = set()
    tick = 0
    next_ops = time.monotonic() + 20
    last_exc_line = ""
    while time.monotonic() < stop_at:
        tick += 1
        t0 = time.monotonic()
        st, m = c.get("/v1/metrics", timeout=30)
        if st == 200 and isinstance(m, dict) and "server" in m:
            stats.samples += 1
            check_sample(m["server"], ids, args.width, stats)
        else:
            stats.finding({"metrics_status": st, "body": str(m)[:200]})
        if tick % 15 == 0:
            st, body = c.get("/metrics?format=prometheus", timeout=30)
            if st != 200 or "gmlx_requests_count" not in str(body):
                stats.finding({"prometheus_status": st, "body": str(body)[:200]})
        if time.monotonic() >= next_ops:
            next_ops = time.monotonic() + args.ops_interval
            op = rng.choice(["reset_scoped", "reset_all", "unload_idle", "stats"])
            stats.count("op:" + op)
            if op == "reset_scoped":
                st, body = c.post("/v1/cache/reset", {"model": rng.choice(ids)}, timeout=60)
                if st not in (200, 404):
                    stats.finding({"op": op, "status": st, "body": str(body)[:200]})
            elif op == "reset_all":
                st, body = c.post("/v1/cache/reset", timeout=60)
                if st != 200:
                    stats.finding({"op": op, "status": st, "body": str(body)[:200]})
            elif op == "unload_idle":
                st, m = c.get("/v1/metrics", timeout=30)
                res = (m.get("server", {}).get("resident_models") if isinstance(m, dict) else None) or []
                idle = [e["ids"][0] for e in res if e.get("ids") and not e.get("in_flight")]
                if len(ids) > 1 and idle:
                    victim = rng.choice(idle)
                    st, body = c.post("/unload", {"model": victim}, timeout=120)
                    if st not in (200, 409):
                        stats.finding({"op": op, "model": victim, "status": st, "body": str(body)[:200]})
                    journal.write({"op": op, "model": victim, "status": st, "t": round(time.time(), 1)})
            elif op == "stats":
                st, body = c.get("/v1/cache/stats", timeout=30)
                if st != 200:
                    stats.finding({"op": op, "status": st, "body": str(body)[:200]})
        scan_log(log_path, stats, seen)
        if srv.proc.poll() is not None:
            stats.finding({"server_died": srv.proc.returncode})
            return
        # exception-line pairing: the exception name line right after a
        # traceback is judged by scan_log; keep the last one for the report
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith(("Traceback",)):
                        last_exc_line = line.strip()[:120]
        except OSError:
            pass
        time.sleep(max(0.0, 1.0 - (time.monotonic() - t0)))
    stats.last_exc = last_exc_line


def parse_model(spec: str):
    mid, rest = spec.split("=", 1)
    path, extra = (rest.split(":", 1) + [""])[:2]
    return mid, os.path.expanduser(path), extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True, help="ID=PATH[:spec|:draft=PATH]")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--cycle-minutes", type=float, default=5.0)
    ap.add_argument("--clients", type=int, default=6)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ops-interval", type=float, default=45.0, help="seconds between operator ops")
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = [parse_model(m) for m in a.models]
    for mid, path, extra in models:
        if not os.path.exists(path):
            print(f"model missing: {path}", file=sys.stderr)
            return 2
        if extra.startswith("draft=") and not os.path.exists(os.path.expanduser(extra[6:])):
            print(f"draft missing: {extra[6:]}", file=sys.stderr)
            return 2
    ids = [m[0] for m in models]
    os.makedirs(a.out, exist_ok=True)
    label = os.path.basename(a.out.rstrip("/"))
    log_path = os.path.join(a.out, "server.log")
    cfg_path = os.path.join(a.out, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write("server:\n  cache:\n    enabled: true\n  defaults:\n"
                f"    model: {ids[0]}\nmodels:\n")
        for mid, path, extra in models:
            f.write(f"  {mid}:\n    path: {path}\n")
            if extra == "spec":
                f.write("    speculative: true\n")
            elif extra.startswith("draft="):
                f.write(f"    draft_gguf: {os.path.expanduser(extra[6:])}\n")
    env = {"GMLX_DECODE_BATCH": str(a.width), "GMLX_QUEUE_DEPTH_CAP": str(a.cap)}

    print(f"== {label}: building prefixes ==", flush=True)
    prefixes = build_prefixes()
    stats = Stats()
    journal = Journal(os.path.join(a.out, "journal.jsonl"))
    shared = {"lock": threading.Lock(), "warm": []}
    t_start = time.monotonic()
    with ServerProc(["--config", cfg_path], env_extra=env, log_path=log_path) as srv:
        srv.wait_ready(timeout=900)
        base = srv.base_url
        c = Client(base)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            st, m = c.get("/v1/metrics", timeout=30)
            res = (m.get("server", {}).get("resident_models") if isinstance(m, dict) else None) or []
            if any(e.get("ids") and e["ids"][0] == ids[0] for e in res):
                break
            time.sleep(2)
        print(f"== {label}: {ids} on {base}; {a.cycles} cycles x {a.cycle_minutes:.0f} min, "
              f"{a.clients} clients, width {a.width}, cap {a.cap}, seed {a.seed} ==", flush=True)
        for cycle in range(1, a.cycles + 1):
            stop_at = time.monotonic() + a.cycle_minutes * 60
            rng_c = random.Random(a.seed * 7919 + cycle)
            threads = [threading.Thread(
                target=worker, args=(w, a, base, ids, prefixes, stop_at, stats, journal,
                                     random.Random(a.seed * 1000 + cycle * 100 + w), shared),
                daemon=True) for w in range(a.clients)]
            ctl = threading.Thread(target=controller, args=(a, base, c, ids, stop_at, stats,
                                                            journal, log_path, srv, rng_c),
                                   daemon=True)
            for t in threads:
                t.start()
            ctl.start()
            for t in threads:
                t.join(timeout=a.cycle_minutes * 60 + a.request_timeout + 60)
            ctl.join(timeout=120)
            if srv.proc.poll() is not None:
                break
            # cycle boundary: drain, then a resident-set churn on multi-model
            # configs (evict-and-reload of the primary), and a scoped reset
            drained = None
            for _ in range(120):
                st, m = c.get("/v1/metrics", timeout=30)
                s = m.get("server", {}) if isinstance(m, dict) else {}
                if s.get("requests") == [] and (s.get("concurrency", {}).get("in_flight") or 0) == 0:
                    drained = s
                    break
                time.sleep(1)
            if drained is None:
                stats.finding({"cycle": cycle, "note": "did not drain within 120 s"})
            st, body = c.post("/v1/cache/reset", {"model": ids[0]}, timeout=60)
            if st != 200:
                stats.finding({"cycle": cycle, "op": "reset_primary", "status": st})
            if len(ids) > 1:
                st, body = c.post("/unload", {"model": ids[0]}, timeout=120)
                stats.count("op:cycle_unload_primary")
                if st != 200:
                    stats.finding({"cycle": cycle, "op": "unload_primary", "status": st,
                                   "body": str(body)[:200]})
                st, body = plain_chat(base, {"model": ids[0], "max_tokens": 8,
                                             "messages": [{"role": "user", "content": "Say ok."}]},
                                      a.request_timeout)
                kind = classify(st, body)
                if kind not in ("ok", "shed", "load_deferred"):
                    stats.finding({"cycle": cycle, "op": "reload_primary", "status": st,
                                   "body": str(body)[:200]})
            n_ok = stats.statuses.get("200", 0)
            print(f"  cycle {cycle}/{a.cycles}: {sum(stats.actions.values())} actions, "
                  f"{n_ok} ok, {stats.sheds} sheds, typed {stats.typed}, "
                  f"{len(stats.findings)} findings, {stats.samples} samples", flush=True)
        time.sleep(3)
        st, body, err = plain_chat(base, {"model": ids[0], "max_tokens": 8, "temperature": 0.0,
                                          "messages": [{"role": "user", "content": "Say ok."}]}, 300)
        if classify(st, body) not in ("ok", "shed"):
            stats.finding({"post_soak_probe": st, "err": err, "body": str(body)[:200]})
        scan_log(log_path, stats, set())
        if srv.proc.poll() is not None:
            stats.finding({"server_died": srv.proc.returncode})

    if stats.actions.get("burst", 0) >= 3 and not stats.typed.get("queue_cap"):
        stats.finding({"note": "queue cap never fired",
                       "bursts": stats.actions["burst"], "k": a.width + a.cap + 1})
    walls = sorted(stats.walls)
    report = {
        "label": label, "models": a.models, "cycles": a.cycles, "cycle_minutes": a.cycle_minutes,
        "clients": a.clients, "width": a.width, "cap": a.cap, "seed": a.seed,
        "elapsed_s": round(time.monotonic() - t_start), "actions": stats.actions,
        "statuses": stats.statuses, "typed": stats.typed, "sheds": stats.sheds,
        "samples": stats.samples, "invariant_failures": len(stats.invariant_failures),
        "wall_p50": walls[len(walls) // 2] if walls else None,
        "wall_p95": walls[int(len(walls) * 0.95)] if walls else None,
        "findings": stats.findings,
    }
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"  actions {json.dumps(stats.actions)}")
    print(f"  statuses {json.dumps(stats.statuses)}  typed {json.dumps(stats.typed)}  sheds {stats.sheds}")
    print(f"  samples {stats.samples}  invariant failures {len(stats.invariant_failures)}  "
          f"wall p50/p95 {report['wall_p50']}/{report['wall_p95']}")
    verdict = "PASS" if not stats.findings else f"FAIL ({len(stats.findings)} findings)"
    print(f"== {label}: {verdict} -> {a.out}/report.json ==", flush=True)
    return 0 if not stats.findings else 1


if __name__ == "__main__":
    sys.exit(main())
