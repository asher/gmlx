#!/usr/bin/env python3
"""Live checks of a streamed MoE beside the rest of the box.

Five phases, each on its own server, all on an over-RAM model served
with ``stream: experts``:

* ``cycles``: load, unload and reload the model three times. Wired
  memory returns near its pre-load value after every unload, the arena
  and the priced footprint come back at the same size, and the first
  ``[stream] memory budget:`` line agrees with the fit planner.
* ``warmth``: the arena hit rate rises over a series of requests as the
  arena fills. Then another process wires memory while a stream
  decodes: the arena steps down without a collapse, no row is shed,
  and the arena regrows after the release when the box has the room.
* ``occupied``: another process holds memory before the server boots.
  The arena sizes smaller by about that amount, the kernel floor does
  not trip, and decode runs.
* ``coresident``: the streamed model and a dense model share the
  residency budget with a capped arena. Alternating requests evict
  neither, and ``resident_bytes`` is the sum of the priced footprints.
* ``coload``: the streamed model loads while the dense model decodes,
  the dense model is requested while the streamed model decodes, and
  again during a streamed reload. Every stream completes. No shed, no
  Metal out-of-memory error, and the load lowers the wired limit the
  dense generator raised.

Usage: python tests/e2e/run_stream_e2e.py [--phases cycles warmth occupied coresident coload]
           [--model PATH] [--dense PATH] [--arena-gb N] [--hog-gb N]
Run on an idle box. Each phase costs 5 to 15 minutes on Kimi-K2.7 UD-Q2.
Exit 0 on pass, 1 on any failed check, 2 when a model file is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from client import Client  # noqa: E402
from run_capacity_multi_e2e import Sampler  # noqa: E402
from server_proc import ServerProc  # noqa: E402

GGUF = os.path.expanduser("~/llm/gguf")
MODEL = (f"{GGUF}/unsloth__Kimi-K2.7-Code-GGUF/UD-Q2_K_XL/"
         "Kimi-K2.7-Code-UD-Q2_K_XL-00001-of-00008.gguf")
DENSE = f"{GGUF}/unsloth__Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf"
GIB = 1 << 30
PHASES = ("cycles", "warmth", "occupied", "coresident", "coload")
LONG_PROMPT = "Count from 1 to 5000, one number per line, no other text."
_results: list = []
_BUDGET = re.compile(r"memory budget: ceiling ([\d.]+) GB = every-token ([\d.]+) "
                     r"\+ arena ([\d.]+) \+ ring ([\d.]+) \+ kv room ([\d.]+) "
                     r"\(.*\) \+ floor ([\d.]+)")


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
          flush=True)
    return bool(ok)


def gb(v) -> str:
    return "n/a" if v is None else f"{v / 1e9:.1f} GB"


def wired_bytes() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"page size of (\d+) bytes", out)
    page = int(m.group(1)) if m else 16384
    for ln in out.splitlines():
        if ln.startswith("Pages wired down"):
            return int(ln.split()[-1].rstrip(".")) * page
    return 0


def rss_bytes(pid: int) -> int:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
    try:
        return int(out.strip()) * 1024
    except ValueError:
        return 0


def count_in_log(path: str, needle: str) -> int:
    try:
        with open(path, errors="replace") as f:
            return f.read().count(needle)
    except OSError:
        return -1


def wait_budget_lines(path: str, n: int, timeout: float = 120.0) -> list:
    """The install prints the budget line after residency flips."""
    deadline = time.monotonic() + timeout
    while True:
        lines = budget_lines(path)
        if len(lines) >= n or time.monotonic() >= deadline:
            return lines
        time.sleep(2)


def footprint(pid: int) -> dict:
    """``vmmap --summary`` numbers: the physical footprint and the resident
    mapped-file bytes. Both empty when vmmap is unavailable."""
    out = {}
    try:
        txt = subprocess.run(["vmmap", "--summary", str(pid)], capture_output=True,
                             text=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return out
    m = re.search(r"Physical footprint:\s+([\d.]+)([KMG])", txt)
    if m:
        out["footprint"] = float(m.group(1)) * {"K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
    m = re.search(r"^mapped file\s+[\d.]+[KMG]\s+([\d.]+)([KMG])", txt, re.M)
    if m:
        out["mapped_file"] = float(m.group(1)) * {"K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
    return out


def budget_lines(path: str) -> list:
    out = []
    try:
        with open(path, errors="replace") as f:
            for ln in f:
                m = _BUDGET.search(ln)
                if m:
                    c, e, ar, rg, r, fl = (float(x) * 1e9 for x in m.groups())
                    out.append({"ceiling": c, "every": e, "arena": ar,
                                "ring": rg, "room": r, "floor": fl})
    except OSError:
        pass
    return out


def shed_count(path: str) -> int:
    return count_in_log(path, "RowShedError") + count_in_log(path, "shed under memory pressure")


def planner(path: str):
    """``(model, box)`` from the fit planner."""
    from gmlx.stream import plan

    return plan.plan_path(path)


def ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 128 * GIB


def kernel_reclaimable() -> int | None:
    """The box's reclaimable pages now. The governor's field only moves
    on a decode tick, so an idle read of it is stale."""
    from gmlx.serve.kernel_vm import reclaimable_bytes

    return reclaimable_bytes()


def floor_bytes(gov: dict) -> float | None:
    note = str(gov.get("kernel_floor") or "")
    m = re.match(r"([\d.]+) GB", note)
    return float(m.group(1)) * 1e9 if m else None


def state(c: Client) -> dict:
    st, m = c.get("/v1/metrics", timeout=20)
    return m["server"] if st == 200 else {}


def resident(c: Client) -> dict:
    """model id -> resident_models entry."""
    out = {}
    for e in state(c).get("resident_models", []):
        for i in e.get("ids") or []:
            out[i] = e
    return out


def wait_resident(c: Client, mid: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mid in resident(c):
            return True
        time.sleep(2)
    return False


def chat(c: Client, mid: str, max_tokens: int = 8, timeout: float = 1800.0):
    return c.post("/v1/chat/completions",
                  {"model": mid, "max_tokens": max_tokens, "temperature": 0,
                   "messages": [{"role": "user", "content": "Name three planets."}]},
                  timeout=timeout)


def stream_timed(base: str, mid: str, max_tokens: int, seed: int = 0,
                 timeout: float = 1800.0, first_token: threading.Event | None = None,
                 prompt: str | None = None) -> dict:
    """One SSE stream; decode tok/s from the first to the last chunk."""
    if prompt is None:
        prompt = (f"Write a numbered list of {12 + seed % 9} distinct "
                  "animals, one per line, each with a two-sentence fact.")
    body = {"model": mid, "stream": True, "max_tokens": max_tokens,
            "temperature": 0.7, "seed": seed,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    status, n, first, last, err = 0, 0, None, None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = r.status
            for line in r:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    n += 1
                    last = time.monotonic()
                    if first is None:
                        first = last
                        if first_token is not None:
                            first_token.set()
    except urllib.error.HTTPError as e:
        status, err = e.code, e.read().decode()[:200]
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    if first_token is not None:
        first_token.set()
    dec = (n - 1) / (last - first) if n > 1 and last > first else 0.0
    return {"status": status, "tokens": n, "error": err,
            "ttft_s": None if first is None else round(first - t0, 1),
            "decode_tok_s": round(dec, 2), "wall_s": round(time.monotonic() - t0, 1),
            "end": time.monotonic()}


def stream_in_thread(base: str, mid: str, max_tokens: int, seed: int = 0,
                     prompt: str | None = None):
    """Start a long stream; returns (thread, result dict, first-token event)."""
    out: dict = {}
    first = threading.Event()
    t = threading.Thread(
        target=lambda: out.update(stream_timed(base, mid, max_tokens, seed=seed,
                                               first_token=first, prompt=prompt)),
        daemon=True)
    t.start()
    return t, out, first


class Hog:
    """Another process that wires ``gb`` GiB for the block's duration."""

    def __init__(self, gb: float):
        self.gb = gb
        self.proc = None
        self.mode = ""

    def __enter__(self) -> "Hog":
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "memhog.py"), "--gb", str(self.gb)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        line = self.proc.stdout.readline()
        if not line.startswith("wired"):
            self.release()
            raise RuntimeError(f"memhog did not wire: {line!r}")
        self.mode = line.split()[2]
        return self

    def release(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __exit__(self, *a) -> None:
        self.release()


def write_cfg(path: str, models: list) -> str:
    with open(path, "w") as f:
        f.write("server:\n  cache:\n    enabled: true\nmodels:\n")
        for mid, p, stream in models:
            f.write(f"  {mid}:\n    path: {p}\n")
            if stream:
                f.write("    stream: experts\n")
    return path


def governor_calm(name: str, s: dict, log: str, red0: int = 0,
                  allow_floor: bool = False) -> None:
    g = s.get("governor") or {}
    reds = (g.get("red_failures") or 0) - red0
    floors = g.get("kernel_floor_reds") or 0
    what = "no shed" if allow_floor else "no shed, no kernel-floor trip"
    check(f"{name}: {what}",
          reds == 0 and (allow_floor or floors == 0) and shed_count(log) == 0,
          f"band {g.get('band')} red_failures +{reds} kernel_floor_reds {floors} "
          f"shed lines {shed_count(log)} last_action {g.get('last_action')!r}")


def no_tracebacks(srv: ServerProc) -> None:
    bad = [ln for ln in srv.log_tail(4000).splitlines()
           if "Traceback" in ln or " ERROR" in ln or "CRITICAL" in ln]
    check("no tracebacks / ERROR lines in the server log", not bad, "\n".join(bad[:3])[:400])


# ---- phases ----------------------------------------------------------------

def phase_cycles(a) -> None:
    log = f"{a.log_prefix}-cycles.log"
    cfg = write_cfg(log + ".yaml", [("m", a.model, True)])
    w_base = wired_bytes()
    print(f"cycles: wired before boot {gb(w_base)}")
    first: dict = {}
    with ServerProc(["--config", cfg], log_path=log) as srv:
        srv.wait_ready(timeout=600)
        c = Client(srv.base_url, timeout=1800)
        for i in range(1, a.cycles + 1):
            t0 = time.monotonic()
            if i > 1:
                st, body = chat(c, "m")
                check(f"cycle {i}: request reloads the model", st == 200, json.dumps(body)[:120])
            ok = wait_resident(c, "m", a.load_window)
            check(f"cycle {i}: resident within {a.load_window:.0f}s", ok,
                  f"{time.monotonic() - t0:.0f}s")
            if i == 1:
                st, body = chat(c, "m")
                check(f"cycle {i}: request answers", st == 200, json.dumps(body)[:120])
            s = state(c)
            mem = s.get("memory") or {}
            nominal = mem.get("arena_nominal_bytes") or 0
            arena = mem.get("arena_bytes") or 0
            ent = resident(c).get("m") or {}
            fp = ent.get("footprint_bytes") or 0
            rb = (s.get("residency") or {}).get("resident_bytes")
            w_loaded = wired_bytes()
            if not first:
                first = {"nominal": nominal, "fp": fp, "wired": w_loaded - arena}
            check(f"cycle {i}: one live arena, sized like the first load",
                  0 < arena <= nominal and abs(nominal - first["nominal"]) <= 0.15 * first["nominal"],
                  f"arena {gb(arena)} of nominal {gb(nominal)}; first {gb(first['nominal'])}")
            check(f"cycle {i}: footprint priced like the first load, resident_bytes equal",
                  fp > 0 and abs(fp - first["fp"]) <= 0.15 * first["fp"] and rb == fp,
                  f"footprint {gb(fp)} resident_bytes {gb(rb)}; first {gb(first['fp'])}")
            # the arena's ladder step differs between loads, so compare
            # the wired bytes outside the arena
            check(f"cycle {i}: wired memory outside the arena like the first load",
                  abs((w_loaded - arena) - first["wired"]) <= 6e9,
                  f"{gb(w_loaded)} wired - arena {gb(arena)} = {gb(w_loaded - arena)}; "
                  f"first {gb(first['wired'])}")
            st, body = c.post("/unload", {"model": "m"})
            time.sleep(6)
            s2 = state(c)
            mem2 = s2.get("memory") or {}
            rb2 = (s2.get("residency") or {}).get("resident_bytes") or 0
            w_un = wired_bytes()
            check(f"cycle {i}: unload -> 200, no live arena, resident_bytes 0",
                  st == 200 and "arena_bytes" not in mem2 and rb2 == 0 and "m" not in resident(c),
                  f"status {st} memory keys {sorted(mem2)} resident_bytes {gb(rb2)}")
            check(f"cycle {i}: wired memory back near the pre-load value",
                  w_un <= w_base + 6e9, f"{gb(w_un)} vs base {gb(w_base)}")
            rss = rss_bytes(srv.proc.pid)
            fpt = footprint(srv.proc.pid)
            first.setdefault("rss", rss)
            first.setdefault("footprint", fpt.get("footprint", 0))
            check(f"cycle {i}: server process holds no more after the unload than after the first",
                  rss <= first["rss"] + 3e9 and fpt.get("footprint", 0) <= first["footprint"] + 3e9,
                  f"rss {gb(rss)} first {gb(first['rss'])}; footprint {gb(fpt.get('footprint'))} "
                  f"first {gb(first['footprint'])}; mapped file {gb(fpt.get('mapped_file'))}; "
                  f"mlx active {gb(mem2.get('active_bytes'))} cache {gb(mem2.get('cache_bytes'))}")
        n_budget = count_in_log(log, "[stream] memory budget:")
        n_hit = count_in_log(log, "arena hit rate:")
        check(f"log carries {a.cycles} budget lines and {a.cycles} hit-rate summaries",
              n_budget == a.cycles and n_hit == a.cycles, f"budget {n_budget} hit-rate {n_hit}")

        # the planner against the first live load
        lines = wait_budget_lines(log, 1)
        model, box = planner(a.model)
        b = lines[0] if lines else {}
        check("planner ceiling matches the live budget line",
              bool(b) and abs(b["ceiling"] - box.ceiling_bytes) <= 0.02 * box.ceiling_bytes,
              f"live {gb(b.get('ceiling'))} planner {gb(box.ceiling_bytes)}")
        check("planner KV room matches the live budget line",
              bool(b) and abs(b["room"] - box.room.bytes) <= 0.03 * box.room.bytes,
              f"live {gb(b.get('room'))} planner {gb(box.room.bytes)}")
        check("planner ring matches the live budget line",
              bool(b) and abs(b["ring"] - model.ring_bytes) <= 0.02 * model.ring_bytes + 1e6,
              f"live {gb(b.get('ring'))} planner {gb(model.ring_bytes)}")
        from gmlx.stream.budget import host_floor_bytes
        floor = host_floor_bytes(ram_bytes())
        check("live floor matches the host floor",
              bool(b) and abs(b["floor"] - floor) <= 0.1e9,
              f"live {gb(b.get('floor'))} floor {gb(floor)}")
        check("live arena within the planner's arena (the clamp only shrinks it)",
              bool(b) and b["arena"] <= box.arena_bytes * 1.02 + 1e9
              and b["arena"] >= box.arena_bytes - a.plan_slack_gb * 1e9,
              f"live {gb(b.get('arena'))} planner {gb(box.arena_bytes)}; every-token "
              f"live {gb(b.get('every'))} planner {gb(model.every_token_bytes)}")
        no_tracebacks(srv)


def phase_warmth(a) -> None:
    log = f"{a.log_prefix}-warmth.log"
    cfg = write_cfg(log + ".yaml", [("m", a.model, True)])
    with ServerProc(["--config", cfg], log_path=log) as srv:
        srv.wait_ready(timeout=600)
        base = srv.base_url
        c = Client(base, timeout=1800)
        check("warmth: resident", wait_resident(c, "m", a.load_window))

        print("warmth: warm series")
        series = []
        for i in range(a.warm_requests):
            m0 = state(c).get("memory") or {}
            r = stream_timed(base, "m", a.max_tokens, seed=i)
            m1 = state(c).get("memory") or {}
            looks = (m1.get("arena_lookups") or 0) - (m0.get("arena_lookups") or 0)
            hits = (m1.get("arena_hits") or 0) - (m0.get("arena_hits") or 0)
            rate = hits / looks if looks else 0.0
            series.append((r, rate))
            check(f"warm {i}: 200 with tokens",
                  r["status"] == 200 and r["tokens"] >= a.max_tokens // 2,
                  f"{r['tokens']} tokens, ttft {r['ttft_s']}s, decode {r['decode_tok_s']} tok/s, "
                  f"hit rate {100 * rate:.1f}% ({hits}/{looks}), arena {gb(m1.get('arena_bytes'))}")
        (r_first, rate_first), (r_last, rate_last) = series[0], series[-1]
        check("arena hit rate rises from the first request to the last",
              rate_last > rate_first + 0.05,
              f"{100 * rate_first:.1f}% -> {100 * rate_last:.1f}%")
        check("warm decode is not slower than the cold first request",
              r_last["decode_tok_s"] >= 0.9 * r_first["decode_tok_s"],
              f"{r_first['decode_tok_s']} -> {r_last['decode_tok_s']} tok/s")
        warm_tok_s = max(r["decode_tok_s"] for r, _ in series)

        # a transient: another process wires memory while a stream decodes
        s0 = state(c)
        mem0, g0 = s0.get("memory") or {}, s0.get("governor") or {}
        nominal = mem0.get("arena_nominal_bytes") or 0
        arena0 = mem0.get("arena_bytes") or 0
        recl0 = kernel_reclaimable()
        floor = floor_bytes(g0)
        red0 = g0.get("red_failures") or 0
        if recl0 is None or floor is None:
            hog_gb = a.hog_gb
        else:
            hog_gb = min(a.hog_max_gb, max(4.0, (recl0 - floor + 2e9) / GIB))
        print(f"warmth: transient of {hog_gb:.1f} GiB; arena {gb(arena0)} of {gb(nominal)}, "
              f"reclaimable {gb(recl0)}, floor {gb(floor)}")
        first_tok = threading.Event()
        out: list = []
        t = threading.Thread(
            target=lambda: out.append(stream_timed(base, "m", a.transient_tokens, seed=7,
                                                   first_token=first_tok)),
            daemon=True)
        with Sampler(base) as smp:
            t.start()
            first_tok.wait(timeout=900)
            time.sleep(5)
            t_hog = time.monotonic()
            with Hog(hog_gb) as hog:
                print(f"warmth: hog wired ({hog.mode}) at +{time.monotonic() - t_hog:.0f}s")
                t.join(timeout=a.hold_s)
                hold = smp.since(t_hog)
            t_rel = time.monotonic()
            t.join(timeout=1800)
        r_hold = out[0] if out else {"status": 0, "tokens": 0, "decode_tok_s": 0}
        arenas = [x["memory"].get("arena_bytes") for x in hold if x.get("memory")]
        arenas = [v for v in arenas if v]
        recls = [(x.get("governor") or {}).get("kernel_reclaimable_bytes") for x in hold]
        recls = [v for v in recls if v is not None]
        min_arena = min(arenas) if arenas else None
        min_recl = min(recls) if recls else None
        s1 = state(c)
        g1 = s1.get("governor") or {}
        levels = count_in_log(log, "memory pressure (level")
        check("transient: stream decoding through it completes",
              r_hold["status"] == 200 and r_hold["tokens"] >= a.transient_tokens // 2,
              f"{r_hold['tokens']} tokens, decode {r_hold['decode_tok_s']} tok/s")
        if floor is not None:
            check("transient: reclaimable fell under the kernel floor",
                  min_recl is not None and min_recl < floor,
                  f"min reclaimable {gb(min_recl)} floor {gb(floor)}")
        check("transient: arena stepped down",
              min_arena is not None and min_arena < arena0 - 0.1 * nominal,
              f"arena {gb(arena0)} -> min {gb(min_arena)} (nominal {gb(nominal)}); "
              f"{levels} pressure-level lines")
        check("transient: no collapse (arena kept at least 45% of nominal)",
              min_arena is not None and min_arena >= 0.45 * nominal,
              f"min {gb(min_arena)} of {gb(nominal)}")
        freed = re.search(r"kernel floor reclaim freed ([\d.]+) GB", str(g1.get("last_action")))
        actions = {str((x.get("governor") or {}).get("last_action")) for x in hold}
        freed_gb = max([float(m.group(1)) for m in
                        (re.search(r"freed ([\d.]+) GB", s) for s in actions) if m] or [0.0])
        reclaims = count_in_log(log, "cache cleared, now")
        collapses = count_in_log(log, "and collapsing")
        check("transient: a kernel-floor reclaim takes about one arena step, not all",
              freed_gb <= 0.25 * nominal / 1e9 + 6.0,
              f"freed {freed_gb:.1f} GB; one step {0.25 * nominal / 1e9:.1f} GB; "
              f"last_action {g1.get('last_action')!r}" + ("" if freed else " (no floor reclaim)"))
        governor_calm("transient", s1, log, red0, allow_floor=True)
        check("transient: the floor reclaims once or twice and never reads a collapse",
              1 <= reclaims <= 2 and collapses == 0,
              f"{reclaims} reclaim lines, {collapses} collapse lines")

        # release: regrow on the next decode when the box has the room
        time.sleep(10)
        s2 = state(c)
        recl1 = kernel_reclaimable()
        arena1 = (s2.get("memory") or {}).get("arena_bytes") or 0
        need = 0.25 * nominal
        from gmlx.load.loader import _ram_floor_bytes
        gate = need + _ram_floor_bytes(ram_bytes()) + (floor or 0)
        expected = recl1 is not None and recl1 >= gate
        print(f"warmth: released at +{time.monotonic() - t_rel:.0f}s; arena {gb(arena1)}, "
              f"reclaimable {gb(recl1)}, regrow gate {gb(gate)} -> "
              f"{'regrow expected' if expected else 'regrow gated'}")
        with Sampler(base) as smp:
            t_re = time.monotonic()
            r_re = stream_timed(base, "m", a.regrow_tokens, seed=11)
            after = smp.since(t_re)
        arenas2 = [x["memory"].get("arena_bytes") for x in after if x.get("memory")]
        max_arena = max([v for v in arenas2 if v] or [0])
        regrow_lines = count_in_log(log, "regrowing toward")
        check("release: stream completes",
              r_re["status"] == 200 and r_re["tokens"] >= a.regrow_tokens // 2,
              f"{r_re['tokens']} tokens, decode {r_re['decode_tok_s']} tok/s")
        if expected:
            check("release: arena regrows on the next decode",
                  regrow_lines >= 1 and max_arena > arena1 + 0.1 * nominal,
                  f"{regrow_lines} regrow lines; arena {gb(arena1)} -> max {gb(max_arena)}")
        else:
            check("release: regrow gated by the reclaimable rule (no room for a step)",
                  regrow_lines == 0,
                  f"reclaimable {gb(recl1)} < gate {gb(gate)}; {regrow_lines} regrow lines")
        check("release: decode continues at 60% of warm speed or better",
              r_re["decode_tok_s"] >= 0.6 * warm_tok_s,
              f"{r_re['decode_tok_s']} vs warm {warm_tok_s} tok/s")
        governor_calm("release", state(c), log, red0, allow_floor=True)
        no_tracebacks(srv)


def phase_occupied(a) -> None:
    log = f"{a.log_prefix}-occupied.log"
    cfg = write_cfg(log + ".yaml", [("m", a.model, True)])
    _model, box = planner(a.model)
    hog_bytes = a.hog_gb * GIB
    with Hog(a.hog_gb) as hog:
        print(f"occupied: {a.hog_gb:.0f} GiB wired ({hog.mode}) before boot; "
              f"planner arena {gb(box.arena_bytes)}")
        with ServerProc(["--config", cfg], log_path=log) as srv:
            srv.wait_ready(timeout=600)
            c = Client(srv.base_url, timeout=1800)
            check("occupied: resident", wait_resident(c, "m", a.load_window))
            r = stream_timed(srv.base_url, "m", a.max_tokens, seed=3)
            check("occupied: stream completes",
                  r["status"] == 200 and r["tokens"] >= a.max_tokens // 2,
                  f"{r['tokens']} tokens, decode {r['decode_tok_s']} tok/s")
            # the budget line prints at the streaming install, on the first request
            lines = wait_budget_lines(log, 1)
            arena = lines[0]["arena"] if lines else 0
            check("occupied: arena sized down by about the occupied memory",
                  0 < arena <= box.arena_bytes - hog_bytes + a.plan_slack_gb * 1e9,
                  f"live {gb(arena)} planner {gb(box.arena_bytes)} hog {gb(hog_bytes)}")
            check("occupied: arena still worth streaming", arena >= 15e9, gb(arena))
            s = state(c)
            mem = s.get("memory") or {}
            check("occupied: arena live after the stream",
                  (mem.get("arena_bytes") or 0) >= 0.45 * (mem.get("arena_nominal_bytes") or 1),
                  f"{gb(mem.get('arena_bytes'))} of {gb(mem.get('arena_nominal_bytes'))}; "
                  f"{count_in_log(log, 'memory pressure (level')} pressure-level lines")
            governor_calm("occupied", s, log)
            no_tracebacks(srv)


def phase_coresident(a) -> None:
    log = f"{a.log_prefix}-coresident.log"
    cfg = write_cfg(log + ".yaml", [("m", a.model, True), ("d", a.dense, False)])
    env = {"GMLX_DECODE_ARENA_GB": str(a.arena_gb)}
    dense_size = os.path.getsize(a.dense)
    with ServerProc(["--config", cfg], env_extra=env, log_path=log) as srv:
        srv.wait_ready(timeout=600)
        base = srv.base_url
        c = Client(base, timeout=1800)
        st, body = chat(c, "m")
        check("coresident: streamed model loads by request",
              st == 200 and "m" in resident(c), json.dumps(body)[:120])
        st, body = chat(c, "d")
        check("coresident: dense model loads beside it",
              st == 200 and set(resident(c)) == {"m", "d"}, str(sorted(resident(c))))
        for rnd in range(a.rounds):
            for mid in ("m", "d"):
                r = stream_timed(base, mid, a.max_tokens, seed=rnd * 2)
                check(f"round {rnd} {mid}: 200, both still resident",
                      r["status"] == 200 and r["tokens"] > 0 and set(resident(c)) == {"m", "d"},
                      f"{r['tokens']} tokens, decode {r['decode_tok_s']} tok/s; "
                      f"resident {sorted(resident(c))}")
        s = state(c)
        ents = resident(c)
        fp = {k: (e.get("footprint_bytes") or 0) for k, e in ents.items()}
        res = s.get("residency") or {}
        rb, budget = res.get("resident_bytes"), res.get("budget_bytes")
        lines = wait_budget_lines(log, 1)
        every = lines[0]["every"] if lines else 0
        check("coresident: no LRU eviction during the alternation",
              count_in_log(log, "evicting LRU model") == 0)
        check("coresident: resident_bytes is the sum of the priced footprints, under budget",
              rb == sum(fp.values()) and rb is not None and budget is not None and rb <= budget,
              f"resident {gb(rb)} = {' + '.join(f'{k} {gb(v)}' for k, v in fp.items())}; "
              f"budget {gb(budget)}")
        check("coresident: streamed footprint = every-token + the capped arena",
              abs(fp.get("m", 0) - (every + a.arena_gb * GIB)) <= 2.5e9,
              f"footprint {gb(fp.get('m'))} every-token {gb(every)} arena {a.arena_gb} GiB")
        check("coresident: dense footprint = file size",
              abs(fp.get("d", 0) - dense_size) <= 0.02 * dense_size,
              f"footprint {gb(fp.get('d'))} file {gb(dense_size)}")
        check("coresident: streaming entry priced once in the log",
              count_in_log(log, "streaming entry priced at") == 1)
        governor_calm("coresident", s, log)
        no_tracebacks(srv)


def phase_coload(a) -> None:
    log = f"{a.log_prefix}-coload.log"
    cfg = write_cfg(log + ".yaml", [("m", a.model, True), ("d", a.dense, False)])
    env = {"GMLX_DECODE_ARENA_GB": str(a.arena_gb)}
    lowered = "[stream] wired limit lowered from"
    with ServerProc(["--config", cfg], env_extra=env, log_path=log) as srv:
        srv.wait_ready(timeout=600)
        base = srv.base_url
        c = Client(base, timeout=1800)
        st, body = chat(c, "d")
        check("coload: dense model loads by request",
              st == 200 and "d" in resident(c), json.dumps(body)[:120])
        # The streamed model loads while the dense model decodes. The dense
        # generator raised the MLX wired limit; the load lowers it first.
        t, out, first = stream_in_thread(base, "d", a.coload_tokens, seed=1, prompt=LONG_PROMPT)
        check("coload: dense stream started", first.wait(600) and not out)
        st, body = chat(c, "m", timeout=a.load_window)
        t_answer = time.monotonic()
        t.join()
        check("coload: streamed model loads beside the busy dense model and answers",
              st == 200 and set(resident(c)) == {"m", "d"},
              f"status {st}; resident {sorted(resident(c))}; {json.dumps(body)[:100]}")
        check("coload: the dense stream was still decoding at that answer",
              out.get("end", 0) > t_answer,
              f"dense stream ended {out.get('end', 0) - t_answer:+.0f} s from the answer "
              f"({out.get('tokens')} tokens in {out.get('wall_s')} s); raise --coload-tokens")
        check("coload: the dense stream completed, 200, no shed",
              out.get("status") == 200 and (out.get("tokens") or 0) > 0 and shed_count(log) == 0,
              f"{out.get('status')} {out.get('tokens')} tokens {out.get('error')!r} "
              f"shed lines {shed_count(log)}")
        check("coload: the load lowered the raised wired limit",
              count_in_log(log, lowered) == 1, f"{count_in_log(log, lowered)} lines")
        governor_calm("coload: streamed load beside the busy dense model", state(c), log)
        # The dense model is requested while the streamed model decodes.
        t, out, first = stream_in_thread(base, "m", max(24, a.coload_tokens // 10), seed=2,
                                         prompt=LONG_PROMPT)
        check("coload: streamed stream started", first.wait(a.load_window) and not out)
        st, body = chat(c, "d", timeout=a.load_window)
        t_answer = time.monotonic()
        err = body.get("error") if isinstance(body, dict) else None
        typed = st == 503 and isinstance(err, dict) and err.get("type") == "model_load_deferred"
        t.join()
        check("coload: dense request beside the busy stream answers or defers typed",
              (st == 200 and set(resident(c)) == {"m", "d"}) or typed,
              f"status {st}; resident {sorted(resident(c))}; {json.dumps(body)[:160]}")
        check("coload: the streamed stream was still decoding at that answer",
              out.get("end", 0) > t_answer,
              f"stream ended {out.get('end', 0) - t_answer:+.0f} s from the answer "
              f"({out.get('tokens')} tokens in {out.get('wall_s')} s)")
        check("coload: the streamed stream completed, 200, no shed",
              out.get("status") == 200 and (out.get("tokens") or 0) > 0 and shed_count(log) == 0,
              f"{out.get('status')} {out.get('tokens')} tokens {out.get('error')!r} "
              f"shed lines {shed_count(log)}")
        # A dense request lands during the streamed model's reload.
        st, body = c.post("/unload", {"model": "m"})
        check("coload: unload the streamed model", st == 200 and set(resident(c)) == {"d"},
              json.dumps(body)[:120])
        t, out, first = stream_in_thread(base, "m", 8, seed=3)
        time.sleep(3)
        t0 = time.monotonic()
        st, body = chat(c, "d", timeout=a.load_window)
        wall = time.monotonic() - t0
        t.join()
        check("coload: dense request during the streamed reload answers, both complete",
              st == 200 and out.get("status") == 200 and set(resident(c)) == {"m", "d"}
              and shed_count(log) == 0,
              f"dense {st} after {wall:.0f} s, streamed {out.get('status')} "
              f"{out.get('tokens')} tokens ttft {out.get('ttft_s')} s; resident {sorted(resident(c))}")
        check("coload: no LRU eviction", count_in_log(log, "evicting LRU model") == 0)
        governor_calm("coload: end", state(c), log)
        no_tracebacks(srv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="*", default=list(PHASES), choices=PHASES)
    ap.add_argument("--model", default=MODEL, help="over-RAM MoE served with stream: experts")
    ap.add_argument("--dense", default=DENSE,
                    help="dense model for the coresident and coload phases")
    ap.add_argument("--arena-gb", type=float, default=56.0,
                    help="GMLX_DECODE_ARENA_GB for the coresident and coload phases")
    ap.add_argument("--coload-tokens", type=int, default=1600,
                    help="dense stream length that must outlast the streamed load")
    ap.add_argument("--hog-gb", type=float, default=30.0,
                    help="GiB another process wires for the occupied phase")
    ap.add_argument("--hog-max-gb", type=float, default=60.0,
                    help="cap on the warmth transient (sized from the live reclaimable)")
    ap.add_argument("--hold-s", type=float, default=90.0, help="transient hold, seconds")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--warm-requests", type=int, default=4)
    ap.add_argument("--transient-tokens", type=int, default=160)
    ap.add_argument("--regrow-tokens", type=int, default=400)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--load-window", type=float, default=1800.0)
    ap.add_argument("--plan-slack-gb", type=float, default=10.0,
                    help="how far under the planner's arena the live clamp may land")
    ap.add_argument("--log-prefix",
                    default=os.path.expanduser("~/.cache/gmlx/e2e/stream"),
                    help="server log path prefix (not a temp dir: a run that "
                         "panics the box must leave its logs)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log_prefix), exist_ok=True)
    needs_dense = {"coresident", "coload"} & set(a.phases)
    for p in (a.model, a.dense if needs_dense else a.model):
        if not os.path.exists(p):
            print(f"model missing: {p}", file=sys.stderr)
            return 2
    t_start = time.monotonic()
    for ph in a.phases:
        t0 = time.monotonic()
        print(f"\n=== phase {ph} ===", flush=True)
        try:
            globals()[f"phase_{ph}"](a)
        except Exception as e:  # noqa: BLE001
            check(f"{ph}: phase ran to the end", False, f"{type(e).__name__}: {e}")
        print(f"=== phase {ph} done in {time.monotonic() - t0:.0f}s ===", flush=True)
    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed "
          f"in {time.monotonic() - t_start:.0f}s")
    if failed:
        print("failed:", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
