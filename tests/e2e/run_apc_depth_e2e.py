#!/usr/bin/env python3
"""APC depth e2e: multi-thousand-token reuse over real HTTP on real models.

The shallow twin (run_apc_disk_e2e.py) proves the disk tier's lifecycle on a
0.6B model with a ~250-token prefix. This harness proves reuse where it
actually matters: deep prefixes on the archs each APC tier serves - the ckpt
tier (GDN hybrids, SWA models) above all, since serve-path reuse on those
archs is exactly what the apc-overhaul branch exists to fix. `--tier`
resolves which counters must move (tier_keys in run_apc_disk_e2e); the
harness is scheme-agnostic - fp16 KV by default, KV-quant vars only when
`--scheme`/`--bits-*` are passed.

Correctness is asserted, not assumed: the prefix is a synthetic log with
computable per-entry facts (entry_facts), and every cache-served answer
must retrieve the cold-proven probe facts or match the cold answer
byte-for-byte. Reuse floors come from the ckpt cursor's own schedule
arithmetic (the expected_* helpers mirror _ckpt_cursor_init and
_ckpt_turn_boundaries in gmlx/spec_engine.py): an identical resend must
adopt the N-1 replay boundary, turns the render-stable grid floor, a
divergent suffix the interval grid floor. The content field must stay
markup-free everywhere.

Phases against one model:

  populate      cold server, deep-prefix request -> tier stores and disk
                write-through; ckpt additionally proves mid-prefill interval
                boundaries advanced across the > interval prompt
  warm          identical replay -> matched climbs to the N-1 replay
                boundary, wall time collapses vs cold, facts still right
  divergent     same prefix, different question -> ckpt adopts at least the
                interval grid floor (the class the p=N-only store orphaned)
  turns         a real conversation through the real chat template: each
                turn adopts a render-stable boundary at or past the unit
                grid below the prefix, prompts grow monotonically -> the
                render_ctx production path all unit tests fake; with
                --template-kwargs, one extra turn re-renders under those
                kwargs and must still adopt the interval grid
  concurrent    N deep clients sharing the prefix plus one short unrelated
                client -> a ragged mixed warm/cold batch; every deep reply
                must retrieve the proven probe readings, the short row
                must answer correctly (the batch pad-corruption witness)
  restart       fresh process, same APC_DISK_PATH -> skeletons re-index,
                replay repairs from disk; then a divergent suffix and a
                follow-up turn against the restarted process (crash-freedom
                and correctness; adoption depth is layout-dependent, noted)
  reset         /v1/cache/reset clears memory, not disk -> replay hits disk
  churn         distinct mid-size prefixes cycle records through the ckpt
                LRU, then the original replay must still serve correctly
                with no exception-declines
  burst         (block/exact) K siblings sharing a cold user-turn prefix
                fire at once; on the block tier the fresh gate must hold
                the followers for the leader's stores so they admit warm,
                on the exact tier the numbers are noted (user-turn stores
                land at retirement, past the hold ceiling)
  queue-cap     (only with --queue-cap) the main servers boot with
                GMLX_QUEUE_DEPTH_CAP=1; a flood past the decode batch
                must draw immediate 503s carrying the Retry-After
                contract while the rest of the flood serves, the
                rejections counter must move, and a retry after the
                drain must succeed (Unit 5's wire contract, which unit
                tests fake)
  session       (only with --session N) an agent-shaped conversation on a
                dedicated server: N turns of growing history with streamed
                replies, tool-call/tool-role messages, a mid-stream client
                abort, sampled turns, and a compaction rewrite of the
                middle of the history; every turn must keep adopting near
                the previous prompt's grid floor while retirement clones
                churn the record LRU
  tripwire      (ckpt) third server with replay/turn boundaries disabled:
                identical resends are refused by the p-bound, and the
                missed-adoption tripwire must count them and log its
                one-time warning
  bitrate-b     (only with --bits-b) fresh server at a second KV width on
                the SAME disk root -> no cross-adoption, own namespace warms

All requests carry a system message unless --no-system (the system-render
path is part of the shared prefix, so every reuse floor exercises it).
--system-words N swaps the one-liner for an agent-shaped system prompt of
N words of tool and policy blocks, the shape the system-prompt anchor
serves: the shared prefix then ends at the system boundary instead of
sitting in the user turn, and the divergent request must restore it.
--speculative adds MTP to every server (spec-path ckpt arming + sidecar).
`--warm-factor`/`--restart-factor` loosen the wall-clock collapse thresholds
(ckpt restores clone >100 MB of GDN state; default 0.6x cold). Wall-clock
checks assume an otherwise idle machine: the harness notes the load average
at start and warns; --require-idle makes a loaded machine a hard failure.

Not ``test_``-prefixed: needs the GPU and a large local GGUF
(tests/test_e2e_harness_smoke.py pins imports and --help). Run directly::

    python tests/e2e/run_apc_depth_e2e.py --model ~/llm/gguf/.../model.gguf \
        --tier ckpt --out ./depth-out
    python tests/e2e/run_apc_depth_e2e.py --model ... --speculative --out ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import Client  # noqa: E402
from server_proc import ServerProc  # noqa: E402
from run_apc_disk_e2e import (  # noqa: E402
    apc_env,
    model_id_of,
    shard_files,
    stats,
    tier_keys,
    wait_disk_drained,
)

# Mirrors of the ckpt cursor's schedule arithmetic (gmlx/spec_engine.py):
# boundaries sit on unit = lcm(prefill_step 2048, block 16); interval
# points land every GMLX_APC_CKPT_INTERVAL (default 4096) snapped to that
# grid; the replay boundary is N-1; turn boundaries are the unit-grid
# point at or below p_stable (plus p_stable exactly on rotating layouts).
CKPT_UNIT = 2048
CKPT_INTERVAL = 4096
# Rendered-template slack: p_stable / the terminal guard sit within this
# many tokens of the measured prompt length (gen-prompt/think tail).
RENDER_SLACK = 64
# A divergent question replaces roughly this much prompt tail (question
# tokens + template close); boundaries below survive the divergence.
QUESTION_SLACK = 256

_TOPICS = (
    "coolant loop",
    "telemetry bus",
    "actuator array",
    "power rail",
    "sensor mesh",
    "attitude gyro",
    "thermal shroud",
    "antenna gimbal",
    "battery bank",
    "solar wing",
    "reaction wheel",
    "star tracker",
)

SYSTEM_MSG = (
    "You are the flight operations duty assistant. Answer strictly from "
    "the maintenance log you are given."
)


def agent_system(target_words: int) -> str:
    """Coding-agent-shaped system prompt: identity plus numbered tool and
    policy blocks, deterministic and non-repetitive (~1.3 tok/word).

    The shape is what the system-prompt anchor keys on. Agent fan-out
    shares thousands of tokens of system prompt and tool schemas and
    diverges at the first user message, so the reusable prefix ends at
    the system boundary. A harness carrying its shared text in the user
    turn arms no anchor at all.
    """
    out = [SYSTEM_MSG, "\n\nTools and policies:\n"]
    words = len(SYSTEM_MSG.split())
    i = 0
    while words < target_words:
        t = _TOPICS[i % len(_TOPICS)]
        s = (
            f"Tool {i} ({t}_probe): reads the {t} channel and returns "
            f"{(i * 7) % 97} fields; call it at most {i % 4 + 1} times "
            f"per turn, never before the {_TOPICS[(i + 3) % len(_TOPICS)]} "
            f"check, and log the result under key K{(i * 31) % 4093}. "
            f"Policy {i}: when the {t} reading drifts past "
            f"{(i * 13) % 977}, escalate to the duty engineer and cite "
            f"entry {max(0, i - 4)}.\n"
        )
        out.append(s)
        words += len(s.split())
        i += 1
    return "".join(out)


# Session-phase system message: tool output is part of the conversation,
# so the strict answer-from-the-log framing would invite refusals.
SESSION_SYSTEM = (
    "You are the flight operations duty assistant. Use the maintenance "
    "log and any diagnostic tool output in this conversation."
)
SESSION_REPLY_TOKENS = 512


def entry_facts(i: int) -> dict:
    """Ground truth for log entry i (mirrors deep_prefix's formulas)."""
    return {
        "topic": _TOPICS[i % len(_TOPICS)],
        "reading": (i * 7) % 97,
        "technician": "ana" if i % 2 else "bo",
    }


def deep_prefix(target_words: int, header: str = "") -> tuple:
    """Deterministic, non-repetitive briefing text (~1.3 tok/word). Content
    varies per sentence so the APC block chain is non-degenerate, and every
    entry's facts are computable (entry_facts) so answers can be verified.
    ``header`` prepends distinguishing text (churn phase: distinct chains).
    Returns (text, n_entries)."""
    out = [
        header,
        "You are the flight systems engineer on duty. Read the maintenance "
        "log below carefully; every entry matters.\n\nMaintenance log:\n",
    ]
    words = 20
    i = 0
    while words < target_words:
        f = entry_facts(i)
        s = (
            f"Entry {i}: the {f['topic']} reported a reading of "
            f"{f['reading']} units during pass {i % 13}, drifting "
            f"{'up' if i % 3 else 'down'} by {(i * 5) % 11} since entry "
            f"{max(0, i - 4)}; technician {f['technician']} logged it as "
            f"{'nominal' if i % 5 else 'watch'}.\n"
        )
        out.append(s)
        words += len(s.split())
        i += 1
    out.append("\nInstruction: answer using only the log above, concisely.\n")
    return "".join(out), i


def dump_value(j: int, k: int) -> int:
    """Ground truth for channel k of diagnostic dump j. Two digits, so the
    witness regex cannot match a stray pass or drift digit."""
    return (j * 13 + k * 7) % 89 + 10


def tool_dump(j: int) -> str:
    lines = [f"Diagnostic dump {j}:"]
    for k in range(40):
        lines.append(f"Channel C{j}.{k}: {dump_value(j, k)} units")
    return "\n".join(lines) + "\n"


def expected_populate_stores(ptok: int) -> int:
    """Minimum distinct ckpt store positions the cursor schedules for a
    cold ptok-token prompt: the interval points strictly below the
    terminal, the terminal itself, and the N-1 replay boundary when it
    does not coincide with a grid point. Conservative by RENDER_SLACK
    (the terminal guard sits a template tail below ptok)."""
    terminal = ((ptok - RENDER_SLACK) // CKPT_UNIT) * CKPT_UNIT
    if terminal <= 0:
        return 1
    grid = {b for b in range(CKPT_INTERVAL, terminal, CKPT_INTERVAL)}
    grid.add(terminal)
    replay = ptok - 1
    extra = 1 if replay - max(grid) > RENDER_SLACK else 0
    return len(grid) + extra


def turn_grid_floor(prefix_tok: int) -> int:
    """Deepest unit-grid boundary every layout stores below p_stable: the
    render-stable turn floor (rotating layouts additionally store
    p_stable exactly, so real adoption may run deeper)."""
    return ((prefix_tok - RENDER_SLACK) // CKPT_UNIT) * CKPT_UNIT


def divergent_grid_floor(prefix_tok: int) -> int:
    """Deepest interval boundary that survives replacing the question
    tail: the divergent-suffix floor."""
    return ((prefix_tok - QUESTION_SLACK) // CKPT_INTERVAL) * CKPT_INTERVAL


def http_chat(
    base: str,
    mid: str,
    messages: list,
    *,
    max_tokens: int = 32,
    timeout: float = 900.0,
    **extra,
):
    t0 = time.monotonic()
    st, body = Client(base, timeout=timeout).chat(
        mid, messages, max_tokens=max_tokens, temperature=0.0, **extra
    )
    wall = time.monotonic() - t0
    text = ""
    content = ""
    ptok = 0
    if isinstance(body, dict):
        try:
            msg = body["choices"][0]["message"]
            # text includes reasoning: a fact retrieved mid-think is a
            # valid witness. content is what a client feeds back as history.
            content = str(msg.get("content") or "")
            reasoning = str(msg.get("reasoning_content") or msg.get("reasoning") or "")
            text = reasoning + content
        except (KeyError, IndexError, TypeError):
            pass
        ptok = int((body.get("usage") or {}).get("prompt_tokens", 0) or 0)
    return st, text, content, ptok, wall


def raw_chat(
    base: str,
    mid: str,
    messages: list,
    *,
    max_tokens: int = 24,
    timeout: float = 900.0,
):
    """One chat POST that surfaces the raw HTTP contract: (status,
    headers, parsed JSON body). A rejection returns its error body and
    headers instead of raising (queue-cap phase reads Retry-After)."""
    body = {
        "model": mid,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8", "replace")
            return resp.status, resp.headers, json.loads(payload or "{}")
    except urllib.error.HTTPError as e:
        try:
            parsed = json.loads(e.read().decode("utf-8", "replace") or "{}")
        except json.JSONDecodeError:
            parsed = {}
        return e.code, e.headers, parsed
    except Exception as e:  # noqa: BLE001 - report, don't raise
        return -1, {}, {"error": {"message": f"{type(e).__name__}: {e}"}}


def sse_chat(
    base: str,
    mid: str,
    messages: list,
    *,
    max_tokens: int = SESSION_REPLY_TOKENS,
    timeout: float = 900.0,
    abort_after: int = None,
    **extra,
):
    """Streamed chat completion over SSE. Returns (status, text, content,
    events, wall); text includes reasoning deltas. ``abort_after`` closes
    the connection after that many events: a client cancel mid-decode."""
    body = {
        "model": mid,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    body.update(extra)
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    t0 = time.monotonic()
    content, reasoning, events = [], [], 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta") or {}
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                events += 1
                r = delta.get("reasoning_content") or delta.get("reasoning")
                if r:
                    reasoning.append(str(r))
                if delta.get("content"):
                    content.append(str(delta["content"]))
                if abort_after is not None and events >= abort_after:
                    break
    except urllib.error.HTTPError as e:
        return e.code, "", "", 0, time.monotonic() - t0
    except Exception as e:  # noqa: BLE001 - report, don't raise
        return -1, f"{type(e).__name__}: {e}", "", 0, time.monotonic() - t0
    c = "".join(content)
    return status, "".join(reasoning) + c, c, events, time.monotonic() - t0


class Report:
    def __init__(self):
        self.rows = []
        self.failures = []

    def check(self, name: str, ok: bool, detail: str = ""):
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            self.failures.append(name)

    def note(self, name: str, detail: str):
        self.rows.append({"note": name, "detail": detail})
        print(f"  note  {name}  {detail}")


def depth_env(
    disk_root: str,
    block_size: int,
    num_blocks: int,
    disk_gb: int,
    *,
    scheme: str = None,
    bits: str = None,
) -> dict:
    """Server env for one depth run: APC vars always, KV-quant vars only on
    request - fp16 KV is the default and the acceptance shape."""
    env = apc_env(disk_root, block_size, num_blocks)
    env["APC_DISK_MAX_GB"] = str(disk_gb)
    if scheme:
        env["KV_QUANT_SCHEME"] = scheme
    if bits:
        if str(bits).startswith("k"):
            env["GMLX_KVARN_BITS"] = str(bits)  # mixed form, e.g. k6v5
        else:
            env["KV_BITS"] = str(bits)
    return env


def serve_args(model_path: str, speculative: bool, draft_gguf: str = None) -> list:
    args = [model_path, "--no-auth"]
    if speculative:
        args.append("--speculative")
    if draft_gguf:
        args += ["--draft-gguf", draft_gguf]
    return args


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="GGUF path")
    ap.add_argument("--label", default=None, help="report label (default: basename)")
    ap.add_argument(
        "--tier",
        choices=["block", "exact", "ckpt"],
        default="ckpt",
        help="APC tier the model routes to; resolves which counters must move",
    )
    ap.add_argument(
        "--prefix-words",
        type=int,
        default=6000,
        help="deep-prefix size in words (~1.3 tok/word; default 6000)",
    )
    ap.add_argument("--scheme", default=None, help="KV_QUANT_SCHEME (default: fp16 KV)")
    ap.add_argument("--bits-a", default=None, help="primary KV width (needs --scheme)")
    ap.add_argument(
        "--bits-b",
        default=None,
        help="second width: enables the bitrate-b isolation phase (kNvM form ok)",
    )
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument(
        "--session",
        type=int,
        default=0,
        metavar="N",
        help="agent-session phase on a dedicated server: N growing turns "
        "with streamed replies, tool messages, a mid-stream abort, sampled "
        "turns, and a compaction rewrite (0 disables; the abort/sampled/"
        "compaction turns need N >= 10; 14 is the validated shape)",
    )
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument(
        "--churn",
        type=int,
        default=5,
        help="distinct mid-size prefixes cycled through the record LRU after "
        "reset (0 disables; GDN eviction pressure needs more, see note)",
    )
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument(
        "--num-blocks",
        type=int,
        default=2048,
        help="in-memory pool blocks (production default; ckpt chains must fit "
        "the whole prefix - a small pool that forces disk write-through only "
        "suits --tier block)",
    )
    ap.add_argument("--disk-gb", type=int, default=16)
    ap.add_argument(
        "--warm-factor",
        type=float,
        default=0.6,
        help="warm replay must run under this fraction of cold wall",
    )
    ap.add_argument(
        "--restart-factor",
        type=float,
        default=0.6,
        help="post-restart replay threshold (ckpt clones >100 MB GDN state)",
    )
    ap.add_argument(
        "--template-kwargs",
        default=None,
        help="JSON chat_template_kwargs for one extra render-variant turn "
        "(e.g. '{\"enable_thinking\": false}'); model-specific, off by default",
    )
    ap.add_argument(
        "--no-system",
        action="store_true",
        help="omit the system message (for templates that reject the role)",
    )
    ap.add_argument(
        "--system-words",
        type=int,
        default=0,
        metavar="N",
        help="agent-shaped system prompt of N words of tool and policy "
        "blocks (~1.3 tok/word) shared by every request, instead of the "
        "one-line default; the shape the system-prompt anchor serves",
    )
    ap.add_argument(
        "--queue-cap",
        action="store_true",
        help="queue-cap phase: boot the main servers with "
        "GMLX_QUEUE_DEPTH_CAP=1 and flood past the decode batch; tail "
        "arrivals must draw an immediate 503 with the Retry-After "
        "contract and the server must recover",
    )
    ap.add_argument(
        "--no-tripwire",
        action="store_true",
        help="skip the missed-adoption tripwire phase (boots a third server)",
    )
    ap.add_argument(
        "--require-idle",
        action="store_true",
        help="hard-fail instead of warning when the machine is loaded at "
        "start (wall-clock checks assume an idle machine)",
    )
    ap.add_argument("--speculative", action="store_true")
    ap.add_argument(
        "--draft-gguf",
        default=None,
        help="companion drafter GGUF (assistant-shape MTP targets, e.g. "
        "gemma4; implies --speculative on the server side)",
    )
    ap.add_argument("--out", required=True, help="artifact dir (logs, report.json)")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    label = a.label or os.path.basename(a.model)
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    disk_root = os.path.join(out_dir, "apc-disk")
    os.makedirs(disk_root, exist_ok=True)
    rep = Report()

    # log every request/reply to transcript.jsonl for failure diagnosis
    tpath = os.path.join(out_dir, "transcript.jsonl")
    tlock = threading.Lock()

    def chat(base, mid, messages, **kw):  # noqa: F811 - logged shadow
        st, text, content, ptok, wall = http_chat(base, mid, messages, **kw)
        rec = {
            "status": st,
            "ptok": ptok,
            "wall": round(wall, 1),
            "q": str(messages[-1].get("content"))[-160:],
            "text": text[:600],
        }
        with tlock:
            with open(tpath, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return st, text, content, ptok, wall

    def schat(base, mid, messages, **kw):
        st, text, content, events, wall = sse_chat(base, mid, messages, **kw)
        rec = {
            "status": st,
            "stream_events": events,
            "wall": round(wall, 1),
            "q": str(messages[-1].get("content"))[-160:],
            "text": text[:600],
        }
        with tlock:
            with open(tpath, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return st, text, content, events, wall

    K = tier_keys(a.tier)
    ck = a.tier == "ckpt"
    template_kwargs = json.loads(a.template_kwargs) if a.template_kwargs else None
    prefix, n_entries = deep_prefix(a.prefix_words)

    # Probes sit at 1/4 and 1/2 log depth. Deep single-entry retrieval
    # is borderline for small models, so the witness calibrates on the
    # cold reply: only facts it provably retrieved are required later.
    # Readings below 10 collide with pass/drift digits and are skipped.
    def pick_probe(start: int) -> int:
        e = start
        while entry_facts(e)["reading"] < 10:
            e += 1
        return e

    e1 = pick_probe(n_entries // 4)
    e2 = pick_probe(n_entries // 2)
    f1, f2 = entry_facts(e1), entry_facts(e2)
    while f1["reading"] == f2["reading"]:  # only if depth spans ~97 entries
        e2 = pick_probe(e2 + 1)
        f2 = entry_facts(e2)
    q = (
        f"Question: how many units did entries {e1} and {e2} each report, "
        f"and which technician logged entry {e2}? "
        "Answer with the two numbers and the name."
    )
    probes = [
        (f"reading[{e1}]", re.compile(rf"\b{f1['reading']}\b")),
        (f"reading[{e2}]", re.compile(rf"\b{f2['reading']}\b")),
        (f"tech[{e2}]", re.compile(rf"\b{f2['technician']}\b", re.IGNORECASE)),
    ]
    witness = []  # calibrated against the cold answer in populate

    def facts_ok(text: str) -> bool:
        return bool(witness) and all(rx.search(text) for _, rx in witness)

    def served_ok(text: str, cold: str) -> bool:
        # Divergence from cold, not wrongness, is the corruption signal:
        # a warm answer passes on the calibrated facts or on byte-identity
        # with the cold answer.
        return facts_ok(text) or text == cold

    def content_clean(content: str) -> bool:
        return "<|" not in content

    facts_hint = (
        f"entry {e1} -> {f1['reading']} units, entry {e2} -> "
        f"{f2['reading']} units by {f2['technician']}"
    )

    system_msg = agent_system(a.system_words) if a.system_words > 0 \
        else SYSTEM_MSG

    def mk_msgs(user_content: str, system: str | None = None) -> list:
        system = system_msg if system is None else system
        msgs = [] if a.no_system else [{"role": "system", "content": system}]
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def tick(key: str) -> int:
        return int(stats(base).get(key, 0) or 0)

    def reuse_tick() -> int:
        # Total reused prefix tokens: APC adoption plus spec prefix-cache
        # restores. The MTP serve path's first reuse layer never moves the
        # manager's matched counter, but a warm turn is warm either way.
        s = stats(base)
        return int(s.get(K["matched"], 0) or 0) + int(
            s.get("spec_prefix_hit_tokens", 0) or 0)

    def settle(key: str, timeout: float = 15.0) -> int:
        # retirement stores are async; wait for the counter to hold still
        last = tick(key)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            cur = tick(key)
            if cur == last:
                return cur
            last = cur
        return last

    print(f"== APC depth e2e: {label} ==")
    print(
        f"   tier {a.tier} | ~{a.prefix_words} words | scheme={a.scheme or 'fp16'} | "
        f"speculative={a.speculative} | out {out_dir}"
    )

    # wall-clock gates assume an idle machine; --require-idle enforces it
    la1 = os.getloadavg()[0]
    cores = os.cpu_count() or 1
    loaded = la1 / cores > 0.35
    rep.note(
        "env.load",
        f"loadavg1 {la1:.1f} on {cores} cores"
        + (" -- loaded, wall checks unreliable" if loaded else ""),
    )
    if a.require_idle and loaded:
        rep.check("env.idle", False, f"loadavg1 {la1:.1f}/{cores} cores > 0.35")
        print("== aborting before any server boot (--require-idle) ==")
        return 1

    env_a = depth_env(
        disk_root,
        a.block_size,
        a.num_blocks,
        a.disk_gb,
        scheme=a.scheme,
        bits=a.bits_a,
    )
    if a.queue_cap:
        # cap 1 = one waiter past the decode batch. Unreachable by every
        # other phase: their concurrency stays far below the batch size.
        env_a["GMLX_QUEUE_DEPTH_CAP"] = "1"
    t_load0 = time.monotonic()
    with ServerProc(
        serve_args(a.model, a.speculative, a.draft_gguf),
        env_extra=env_a,
        log_path=os.path.join(out_dir, "server-a.log"),
        python=a.python,
    ) as srv:
        srv.wait_ready(timeout=900)
        rep.note("load-a", f"{time.monotonic() - t_load0:.0f}s")
        base = srv.base_url
        mid = model_id_of(base, srv)
        # absorb first-request one-time costs (kernel warmup) so wall-time
        # comparisons below measure caching, not JIT. The warmup stays off
        # the shared chain: carrying the run's system prompt behind a
        # two-word user turn would put its own guard-column store at the
        # system boundary, which then serves every later sibling and hides
        # whether the anchor works at all.
        chat(base, mid, [{"role": "user", "content": "Say ok."}],
             max_tokens=4)

        # -- populate ------------------------------------------------------
        st, text, content, ptok, cold_wall = chat(
            base, mid, mk_msgs(prefix + q), max_tokens=256
        )
        cold_text = text
        settle(K["stores"])
        s = stats(base)
        rep.check("populate.status", st == 200, f"prompt {ptok} tok, {cold_wall:.1f}s")
        rep.check("populate.answer", len(text) > 0, repr(text[:70]))
        # facts the cold answer retrieves become the required witness;
        # at least one exact reading must land
        witness[:] = [(n, rx) for n, rx in probes if rx.search(text)]
        rep.check(
            "populate.facts",
            any(n.startswith("reading") for n, _ in witness),
            f"cold answer retrieves {[n for n, _ in witness]} "
            f"of {[n for n, _ in probes]} ({facts_hint})",
        )
        rep.check("populate.content_clean", content_clean(content), repr(content[:70]))
        rep.check(
            "populate.stores",
            int(s.get(K["stores"], 0)) > 0,
            f"{K['stores']}={s.get(K['stores'])}",
        )
        if ck:
            # the cursor's own schedule sets the store floor
            floor = expected_populate_stores(ptok)
            rep.check(
                "populate.ckpt_interval",
                int(s.get("ckpt_stores", 0)) >= floor,
                f"ckpt_stores={s.get('ckpt_stores')} >= {floor} "
                f"for a {ptok}-tok prompt (interval {CKPT_INTERVAL})",
            )
        drained = wait_disk_drained(base, min_files=1, timeout=120)
        rep.check(
            "populate.disk_write",
            int(drained.get("disk_files", 0)) > 0,
            f"disk_files={drained.get('disk_files')} "
            f"disk_writes={drained.get('disk_writes')}",
        )
        prefix_tok = ptok

        # -- warm ----------------------------------------------------------
        # An identical replay must adopt the deepest boundary below the
        # resend, not a shallower interval point.
        m0 = tick(K["matched"])
        st, text, content, ptok, warm_wall = chat(
            base, mid, mk_msgs(prefix + q), max_tokens=256
        )
        dm = tick(K["matched"]) - m0
        if ck:
            warm_floor = prefix_tok - 8  # N-1 replay, small tokenizer slack
        elif a.tier == "exact":
            warm_floor = prefix_tok - RENDER_SLACK
        else:
            warm_floor = prefix_tok - 2 * a.block_size
        rep.check(
            "warm.status", st == 200, f"{warm_wall:.1f}s vs cold {cold_wall:.1f}s"
        )
        rep.check(
            "warm.matched",
            warm_floor <= dm <= prefix_tok,
            f"{K['matched']} +{dm}, expected [{warm_floor}, {prefix_tok}]",
        )
        rep.check(
            "warm.faster",
            warm_wall < a.warm_factor * cold_wall,
            f"{warm_wall:.1f}s < {a.warm_factor}x{cold_wall:.1f}s",
        )
        # the adopted-cache answer must retrieve the same record-region facts
        rep.check("warm.served_ok", served_ok(text, cold_text), facts_hint)
        rep.check("warm.content_clean", content_clean(content), repr(content[:70]))
        rep.note("warm.same_as_cold", str(text == cold_text))

        # -- divergent suffix ----------------------------------------------
        # Same prefix, different question: ckpt must adopt a boundary below
        # the divergence point. Exact misses by design.
        m0 = tick(K["matched"])
        st, _, _, _, div_wall = chat(
            base,
            mid,
            mk_msgs(prefix + "Question: which subsystem appears in entry 7?"),
        )
        dm = tick(K["matched"]) - m0
        div_floor = max(1, divergent_grid_floor(prefix_tok))
        if ck:
            rep.check(
                "divergent.ckpt_boundary_adopted",
                dm >= div_floor,
                f"{K['matched']} +{dm} >= grid floor {div_floor}, {div_wall:.1f}s",
            )
        else:
            rep.note(
                "divergent-suffix",
                f"status {st}, matched +{dm} "
                f"({'0 expected' if a.tier == 'exact' else 'grid reuse expected'}), "
                f"{div_wall:.1f}s",
            )

        # -- anchor (only with --system-words) ------------------------------
        # A shared agent-shaped system prompt makes the divergent request a
        # sibling: on the ckpt and exact tiers it must restore the system
        # block from the anchor stored during the cold request; the block
        # tier serves the same prefix from the block chain with no anchor.
        # Without --system-words the divergence sits a few tokens in and no
        # anchor arms at all, which is why this gate is opt-in.
        if a.system_words > 0 and not a.no_system:
            st_all = stats(base)
            # ~1.3 tok/word, less the template scaffolding and the grid or
            # chunk snap the ckpt tier applies below the boundary.
            anchor_floor = int(a.system_words * 0.6)
            rep.check(
                "anchor.divergent_adopted",
                dm >= anchor_floor,
                f"divergent matched +{dm} >= system-anchor floor "
                f"{anchor_floor} ({a.system_words} system words)",
            )
            if a.tier == "block":
                # Pure-KV models never enter the anchor code: the stock
                # block tier already serves siblings at block granularity.
                rep.note(
                    "anchor.scope",
                    "block tier serves the system prefix from the block "
                    "chain; no anchor arms",
                )
            else:
                anchored = int(st_all.get("anchor_stores", 0) or 0) + int(
                    st_all.get("ckpt_stores", 0) or 0
                )
                rep.check(
                    "anchor.armed",
                    anchored > 0,
                    f"anchor_stores={st_all.get('anchor_stores', 0)} "
                    f"ckpt_stores={st_all.get('ckpt_stores', 0)}",
                )
                served = int(st_all.get("anchor_hits", 0) or 0) + int(
                    st_all.get("ckpt_hits", 0) or 0
                )
                rep.check(
                    "anchor.served",
                    served > 0,
                    f"anchor_hits={st_all.get('anchor_hits', 0)} "
                    f"ckpt_hits={st_all.get('ckpt_hits', 0)}",
                )

        # -- turns ---------------------------------------------------------
        # A real conversation through the real chat template: the
        # production render_ctx path unit tests fake with synthetic ids.
        def history_safe(reply: str) -> str:
            # cascade guard: a leak already failed content_clean; keep
            # later turns renderable
            if "<|" not in reply:
                return reply
            seg = reply.rsplit("<|message|>", 1)[-1]
            return re.sub(r"<\|[^|>]*\|>", " ", seg).strip() or "(elided)"

        history = []
        last_answer = content
        prev_ptok = 0
        turn_qs = [
            "Follow-up: is the reading in entry 12 above or below 50 units?",
            "Follow-up: name any entry logged as watch, and by whom.",
            "Follow-up: which pass number shows up most often in the log?",
        ]
        turns_ok, monotone, turns_clean = True, True, True
        turn_dms = []
        # ground truth for turn 1's question: entry 12 reads (12*7)%97 = 84
        for t_i in range(a.turns):
            history.append((turn_qs[t_i % len(turn_qs)], history_safe(last_answer)))
            msgs = mk_msgs(prefix + q)
            for uq, at in history:
                msgs.append({"role": "assistant", "content": at})
                msgs.append({"role": "user", "content": uq})
            m0 = reuse_tick()
            st, turn_text, last_answer, ptok, wall = chat(
                base, mid, msgs, max_tokens=96
            )
            dm = reuse_tick() - m0
            turn_dms.append(dm)
            turns_ok &= st == 200 and len(turn_text) > 0
            turns_clean &= content_clean(last_answer)
            monotone &= ptok > prev_ptok
            prev_ptok = ptok
            extra = ""
            if t_i == 0:
                extra = f", says-above={'above' in turn_text.lower()}"
            rep.note(
                f"turn{t_i + 1}",
                f"status {st}, prompt {ptok} tok, matched +{dm}, {wall:.1f}s{extra}",
            )
        rep.check("turns.status", turns_ok, f"{a.turns} turns")
        rep.check(
            "turns.content_clean",
            turns_clean,
            "no channel markup in any turn's content field",
        )
        rep.check("turns.growing", monotone, "templated prompt grows per turn")
        if a.tier != "exact":
            # every turn must reach the unit-grid floor; rot layouts may
            # adopt the exact p_stable and run deeper
            t_floor = (
                max(1, turn_grid_floor(prefix_tok))
                if ck
                else max(
                    1, ((prefix_tok - QUESTION_SLACK) // a.block_size) * a.block_size
                )
            )
            rep.check(
                "turns.adopted",
                min(turn_dms) >= t_floor,
                f"min +{min(turn_dms)} >= floor {t_floor} "
                f"of ~{prefix_tok} across {a.turns} turns",
            )

        # -- render-variant turn (only with --template-kwargs) -------------
        # Changed kwargs can break the exact render-stable boundary; the
        # interval grid below the shared prefix must still adopt.
        if template_kwargs is not None:
            msgs = mk_msgs(prefix + q)
            for uq, at in history:
                msgs.append({"role": "assistant", "content": at})
                msgs.append({"role": "user", "content": uq})
            m0 = reuse_tick()
            st, vtext, vcontent, ptok, wall = chat(
                base,
                mid,
                msgs,
                max_tokens=96,
                chat_template_kwargs=template_kwargs,
            )
            dm = reuse_tick() - m0
            rep.check(
                "variant.status",
                st == 200 and len(vtext) > 0,
                f"kwargs {a.template_kwargs}, prompt {ptok} tok, {wall:.1f}s",
            )
            rep.check(
                "variant.content_clean", content_clean(vcontent), repr(vcontent[:70])
            )
            if ck:
                rep.check(
                    "variant.adopted",
                    dm >= div_floor,
                    f"matched +{dm} >= grid floor {div_floor} under "
                    "changed render kwargs",
                )
            else:
                rep.note("variant.matched", f"+{dm}")

        # -- concurrent ----------------------------------------------------
        # One short client rides the deep burst: the ragged mixed
        # warm/cold batch where stale pad masks corrupt the short row.
        deep_wordings = [
            f"Priority check {i}: what reading did entry {e1} report, and "
            f"what reading did entry {e2} report? Answer with both numbers."
            for i in range(a.concurrency)
        ]

        def fire(i):
            return chat(base, mid, mk_msgs(prefix + deep_wordings[i]), max_tokens=256)

        def fire_short():
            # plain request: the log-focused system message would make the
            # model deliberate the instruction hierarchy instead of answering
            return chat(
                base,
                mid,
                [
                    {
                        "role": "user",
                        "content": "What is 2 plus 2? Answer with just the number.",
                    }
                ],
                max_tokens=160,
            )

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=a.concurrency + 1) as ex:
            futs = [ex.submit(fire, i) for i in range(a.concurrency)]
            fut_short = ex.submit(fire_short)
            res = [f.result() for f in futs]
            short_res = fut_short.result()
        rep.check(
            "concurrent.status",
            all(st == 200 and text for st, text, _, _, _ in res)
            and short_res[0] == 200,
            f"{a.concurrency}+1 clients, {time.monotonic() - t0:.1f}s wall",
        )
        rep.check(
            # technician probes excluded: the burst questions ask for
            # readings only
            "concurrent.coherent",
            all(
                all(rx.search(text) for n, rx in witness if n.startswith("reading"))
                for _, text, _, _, _ in res
            ),
            "each deep reply retrieves the calibrated readings "
            + str([n for n, _ in witness if n.startswith("reading")]),
        )
        rep.check(
            "concurrent.short_row_intact",
            bool(re.search(r"\b4\b", short_res[1])),
            f"short unrelated row answers 4: {short_res[1]!r:.60}",
        )
        if ck:
            rep.note(
                "concurrent.ckpt_scope",
                "ckpt retirement is B=1 by design; the burst asserts "
                "formation correctness, not per-row stores",
            )

        # -- decline/adoption hygiene ---------------------------------------
        settle(K["stores"])
        pre_restart = stats(base)
        if ck:
            declines = pre_restart.get("ckpt_declines") or {}
            n_declined = sum(int(v) for k, v in declines.items() if k != "buffered")
            n_stores = int(pre_restart.get("ckpt_stores", 0) or 0)
            rep.check(
                # a mass-decline regression declines more than it stores;
                # 'buffered' is by design under --speculative on rot layouts
                "hygiene.declines_bounded",
                n_declined <= max(4, n_stores),
                f"non-buffered declines {n_declined} <= stores {n_stores} "
                f"({json.dumps(declines)})",
            )
            rep.check(
                # no request in this run should be refused a prefix-matching
                # record; the tripwire phase proves the counter itself works
                "hygiene.no_missed_adoptions",
                int(pre_restart.get("ckpt_missed_adoptions", 0) or 0) == 0,
                f"ckpt_missed_adoptions={pre_restart.get('ckpt_missed_adoptions')}",
            )

        counter_keys = [
            K["stores"],
            K["hits"],
            K["matched"],
            "disk_writes",
            "disk_files",
            "lookups_hit",
            "matched_tokens",
            "disk_hits",
        ]
        if ck:
            counter_keys += ["ckpt_declines", "ckpt_missed_adoptions"]
        rep.note(
            "counters-a",
            json.dumps({k: pre_restart.get(k) for k in counter_keys}),
        )
        wait_disk_drained(base, min_files=1, timeout=120)
    n_shards = len(shard_files(disk_root))
    rep.check("shards.on_disk", n_shards > 0, f"{n_shards} safetensors shards")

    # -- restart -----------------------------------------------------------
    with ServerProc(
        serve_args(a.model, a.speculative, a.draft_gguf),
        env_extra=env_a,
        log_path=os.path.join(out_dir, "server-a2.log"),
        python=a.python,
    ) as srv:
        srv.wait_ready(timeout=900)
        base = srv.base_url
        mid = model_id_of(base, srv)
        chat(base, mid, mk_msgs("Say ok."), max_tokens=4)
        st, text, r_content, ptok, disk_wall = chat(
            base, mid, mk_msgs(prefix + q), max_tokens=256
        )
        s = stats(base)
        rep.check(
            "restart.disk_hit",
            int(s.get("disk_hits", 0) or 0) > 0,
            f"disk_hits={s.get('disk_hits')}, {disk_wall:.1f}s "
            f"vs cold {cold_wall:.1f}s",
        )
        if ck:
            # the fresh process rebuilt a record from the disk skeleton
            rep.check(
                "restart.skeleton_repair",
                int(s.get("ckpt_hits", 0) or 0) > 0,
                f"ckpt_hits={s.get('ckpt_hits')} "
                f"ckpt_matched_tokens={s.get('ckpt_matched_tokens')}",
            )
        # this KV crossed a process boundary through safetensors shards
        rep.check("restart.served_ok", served_ok(text, cold_text), facts_hint)
        rep.note("restart.same_as_cold", str(text == cold_text))
        rep.check(
            "restart.faster",
            disk_wall < a.restart_factor * cold_wall,
            f"{disk_wall:.1f}s < {a.restart_factor}x{cold_wall:.1f}s",
        )
        rep.check(
            "restart.indexed",
            int(s.get(K["indexed"], 0) or 0) > 0,
            f"{K['indexed']}={s.get(K['indexed'])}",
        )

        # -- divergent + turn against the restarted process ----------------
        # The repaired index must survive a divergent suffix and a
        # follow-up turn. Adoption depth is layout-dependent (grid
        # boundaries persist on arr layouts, window chains are
        # memory-only), so matched is reported, not asserted.
        m0 = tick(K["matched"])
        st, rd_text, rd_content, _, rd_wall = chat(
            base,
            mid,
            mk_msgs(
                prefix + f"Question: which technician logged entry {e2}, "
                f"and what reading did entry {e1} report? "
                "Answer with the name and the number."
            ),
            max_tokens=256,
        )
        dm = tick(K["matched"]) - m0
        rd_req = [
            (n, rx)
            for n, rx in witness
            if n == f"reading[{e1}]" or n.startswith("tech")
        ]
        rep.check(
            # falls back to status plus non-empty when neither asked-for
            # probe was cold-proven
            "restart.divergent_ok",
            st == 200
            and (
                all(rx.search(rd_text) for _, rx in rd_req)
                if rd_req
                else len(rd_text) > 0
            ),
            f"expects calibrated {[n for n, _ in rd_req]} "
            f"({f1['reading']} / {f2['technician']}), matched +{dm}, {rd_wall:.1f}s",
        )
        m0 = tick(K["matched"])
        msgs = mk_msgs(prefix + q)
        msgs.append({"role": "assistant", "content": history_safe(r_content)})
        msgs.append(
            {
                "role": "user",
                "content": f"Follow-up: which technician logged entry {e2}?",
            }
        )
        st, rt_text, rt_content, _, rt_wall = chat(base, mid, msgs, max_tokens=192)
        dm = tick(K["matched"]) - m0
        rt_req = [rx for n, rx in witness if n.startswith("tech")]
        rep.check(
            "restart.turn_ok",
            st == 200
            and (
                all(rx.search(rt_text) for rx in rt_req) if rt_req else len(rt_text) > 0
            ),
            f"expects {f2['technician']!r} (if cold-proven), "
            f"matched +{dm}, {rt_wall:.1f}s",
        )
        rep.check(
            "restart.content_clean",
            content_clean(r_content)
            and content_clean(rd_content)
            and content_clean(rt_content),
            "post-restart content fields markup-free",
        )

        # -- reset ---------------------------------------------------------
        st_reset, _ = Client(base).post("/v1/cache/reset")
        h0 = tick("disk_hits")
        st, text, _, ptok, wall = chat(base, mid, mk_msgs(prefix + q), max_tokens=256)
        h1 = tick("disk_hits")
        rep.check(
            "reset.disk_survives",
            st_reset == 200 and st == 200 and h1 > h0,
            f"disk_hits {h0} -> {h1}, {wall:.1f}s",
        )
        rep.check("reset.served_ok", served_ok(text, cold_text), facts_hint)

        # -- churn ---------------------------------------------------------
        # Distinct mid-size prefixes cycle records through the ckpt LRU,
        # then the original replay must still serve correctly with no
        # exception-declines. Forcing real GDN eviction against the 4 GiB
        # budget needs --churn raised.
        if a.churn > 0:
            churn_text, n_churn = deep_prefix(1800)
            ec = n_churn // 3
            churn_ok = True
            for j in range(a.churn):
                cst, ctext, _, _, _ = chat(
                    base,
                    mid,
                    mk_msgs(
                        f"Archive {j} of prior mission logs follows.\n"
                        + churn_text
                        + f"Question: how many units did entry {ec} report?"
                    ),
                    max_tokens=64,
                )
                churn_ok &= cst == 200 and len(ctext) > 0
            settle(K["stores"])
            d0 = stats(base)
            m0 = tick(K["matched"])
            st, text, _, _, wall = chat(base, mid, mk_msgs(prefix + q), max_tokens=256)
            dm = tick(K["matched"]) - m0
            rep.check("churn.status", churn_ok, f"{a.churn} distinct prefixes")
            rep.check(
                "churn.replay_ok",
                st == 200 and served_ok(text, cold_text),
                f"matched +{dm}, {wall:.1f}s",
            )
            if ck:
                declines = stats(base).get("ckpt_declines") or {}
                rep.check(
                    "churn.no_exception_declines",
                    int(declines.get("exception", 0) or 0) == 0,
                    json.dumps(declines),
                )
                rep.note(
                    "churn.stores",
                    f"ckpt_stores {d0.get('ckpt_stores')} at churn end, "
                    f"replay matched +{dm}",
                )

        # -- sibling burst -------------------------------------------------
        # Unit 4 fresh gate: K siblings sharing a cold user-turn prefix
        # arrive together. On the block tier the followers must end warm
        # (held for the leader's post-prefill stores). The exact tier
        # stores user-turn prefixes only at retirement, beyond the hold
        # ceiling, so its numbers are reported, not gated. Ckpt-tier
        # models admit one row at a time; no co-admission window exists.
        if a.tier == "ckpt":
            rep.note(
                "burst.scope",
                "ckpt tier admits B=1; no co-admission window to gate",
            )
        else:
            burst_text, n_burst = deep_prefix(
                a.prefix_words // 2, header="Sibling-burst variant log.\n"
            )
            eb = n_burst // 2
            fb = entry_facts(eb)
            plug_text, n_plug = deep_prefix(2000, header="Burst plug log.\n")
            m0 = tick(K["matched"])
            fr0 = ((Client(base).metrics()[1] or {}).get("server")
                   or {}).get("freshness") or {}

            def fire_plug():
                # unrelated prefill in flight while the siblings land, so
                # they queue up and co-admit in one formation tick -- the
                # exact window the fresh gate exists for
                return chat(
                    base,
                    mid,
                    mk_msgs(plug_text + "Say ok."),
                    max_tokens=16,
                )

            def fire_sib(i):
                return chat(
                    base,
                    mid,
                    mk_msgs(
                        burst_text
                        + f"Sibling {i}: how many units did entry {eb} "
                        "report? Answer with just the number."
                    ),
                    max_tokens=64,
                )

            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=a.concurrency + 1) as ex:
                fut_plug = ex.submit(fire_plug)
                time.sleep(0.25)
                futs = [ex.submit(fire_sib, i) for i in range(a.concurrency)]
                bres = [f.result() for f in futs]
                plug_res = fut_plug.result()
            b_wall = time.monotonic() - t0
            settle(K["stores"])
            dm = tick(K["matched"]) - m0
            fr1 = ((Client(base).metrics()[1] or {}).get("server")
                   or {}).get("freshness") or {}
            d_holds = int(fr1.get("holds", 0) or 0) - int(
                fr0.get("holds", 0) or 0
            )
            rep.check(
                "burst.status",
                all(st == 200 and text for st, text, _, _, _ in bres)
                and plug_res[0] == 200,
                f"{a.concurrency} cold siblings behind a plug prefill, "
                f"{b_wall:.1f}s wall",
            )
            rep.check(
                "burst.coherent",
                all(
                    re.search(rf"\b{fb['reading']}\b", text)
                    for _, text, _, _, _ in bres
                ),
                f"entry {eb} -> {fb['reading']} units in every reply",
            )
            floor = int(0.6 * (a.prefix_words // 2)) * (a.concurrency - 1)
            if a.tier == "block":
                rep.check(
                    "burst.gate_held",
                    d_holds >= 1,
                    f"fresh holds +{d_holds} (>=1: the siblings "
                    "co-admitted and a follower was deferred)",
                )
                rep.check(
                    "burst.followers_warm",
                    dm >= floor,
                    f"matched +{dm} >= floor {floor} "
                    f"({a.concurrency - 1} followers), fresh holds "
                    f"+{d_holds}",
                )
            else:
                rep.note(
                    "burst.followers",
                    f"matched +{dm} (floor would be {floor}), fresh "
                    f"holds +{d_holds}; exact tier stores user-turn "
                    "prefixes at retirement, past the hold ceiling",
                )
            rep.note(
                "burst.fresh_holds",
                f"+{d_holds} (last reason "
                f"{str(fr1.get('last_hold_reason'))[:80]!r})",
            )

        # -- queue cap -----------------------------------------------------
        # Unit 5's wire contract. The servers run at cap 1, and depth
        # counts past the decode slots (completion batch), so the flood
        # must exceed the batch; tail arrivals then land while the queue
        # is deep and must be rejected at the socket, never mid-stream.
        if a.queue_cap:
            flood = 44
            q0 = ((Client(base).metrics()[1] or {}).get("server")
                  or {}).get("queue") or {}

            def fire_q(i):
                return raw_chat(
                    base,
                    mid,
                    mk_msgs(
                        f"Flood {i}: what is 2 plus {i % 7}? "
                        "Answer with just the number.",
                        system=SYSTEM_MSG,
                    ),
                    max_tokens=24,
                )

            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=flood) as ex:
                qres = list(ex.map(fire_q, range(flood)))
            q_wall = time.monotonic() - t0
            n200 = sum(1 for st, _, _ in qres if st == 200)
            rejects = [(h, b) for st, h, b in qres if st == 503]
            q1 = ((Client(base).metrics()[1] or {}).get("server")
                  or {}).get("queue") or {}
            d_rej = int(q1.get("rejections", 0) or 0) - int(
                q0.get("rejections", 0) or 0
            )
            rep.check(
                "queuecap.rejected",
                len(rejects) >= 1,
                f"{len(rejects)} of {flood} drew 503 at cap 1, "
                f"{q_wall:.1f}s wall",
            )
            rep.check(
                "queuecap.no_other_errors",
                n200 + len(rejects) == flood,
                f"{n200} served + {len(rejects)} rejected == {flood}",
            )

            def retry_ok(h) -> bool:
                try:
                    return 2 <= int(h.get("Retry-After", "")) <= 60
                except (TypeError, ValueError):
                    return False

            first_retry = (
                rejects[0][0].get("Retry-After") if rejects else None
            )
            rep.check(
                "queuecap.retry_after",
                bool(rejects) and all(retry_ok(h) for h, _ in rejects),
                f"every 503 carries Retry-After in [2, 60] "
                f"(first: {first_retry}s)",
            )
            err0 = (rejects[0][1].get("error") or {}) if rejects else {}
            rep.check(
                "queuecap.body_contract",
                bool(rejects)
                and all(
                    (b.get("error") or {}).get("type") == "server_overloaded"
                    and int((b.get("error") or {}).get("queue_cap", 0) or 0)
                    == 1
                    and int((b.get("error") or {}).get("queue_depth", 0) or 0)
                    >= 1
                    for _, b in rejects
                ),
                "type/queue_cap/queue_depth on every 503 "
                f"(first depth {err0.get('queue_depth')})",
            )
            rep.check(
                "queuecap.metrics_counted",
                d_rej >= len(rejects),
                f"queue.rejections +{d_rej} >= {len(rejects)} observed 503s",
            )
            st, text, _, _, _ = chat(
                base,
                mid,
                mk_msgs(
                    "What is 2 plus 2? Answer with just the number.",
                    system=SYSTEM_MSG,
                ),
                max_tokens=8,
            )
            rep.check(
                "queuecap.recovers",
                st == 200 and "4" in text,
                f"post-drain retry status {st}: {text[:40]!r}",
            )

    # -- bitrate-b isolation ------------------------------------------------
    # Only meaningful when a second KV width is requested: the namespaces
    # must not cross-adopt. Skipped entirely on the fp16 acceptance shape.
    if a.bits_b:
        env_b = depth_env(
            disk_root,
            a.block_size,
            a.num_blocks,
            a.disk_gb,
            scheme=a.scheme,
            bits=a.bits_b,
        )
        with ServerProc(
            serve_args(a.model, a.speculative, a.draft_gguf),
            env_extra=env_b,
            log_path=os.path.join(out_dir, "server-b.log"),
            python=a.python,
        ) as srv:
            srv.wait_ready(timeout=900)
            base = srv.base_url
            mid = model_id_of(base, srv)
            chat(base, mid, mk_msgs("Say ok."), max_tokens=4)
            st, text, _, ptok, wall = chat(
                base, mid, mk_msgs(prefix + q), max_tokens=256
            )
            settle(K["stores"])
            s = stats(base)
            rep.check(
                "bitrate_b.status",
                st == 200 and len(text) > 0,
                f"bits {a.bits_b}, {wall:.1f}s",
            )
            rep.note("bitrate_b.facts", f"{facts_ok(text)} ({facts_hint})")
            b_cold_text = text
            rep.check(
                # disk_hits stays 0 and matched stays trivial (a stray BOS-level
                # token can tick the counter without any real adoption)
                "bitrate_b.no_cross_adoption",
                int(s.get("disk_hits", 0) or 0) == 0
                and int(s.get("matched_tokens", 0) or 0) < 64,
                f"disk_hits={s.get('disk_hits')} matched={s.get('matched_tokens')}",
            )
            rep.check(
                "bitrate_b.stores",
                int(s.get(K["stores"], 0) or 0) > 0,
                f"{K['stores']}={s.get(K['stores'])}",
            )
            m0 = tick(K["matched"])
            st, text, _, ptok, wall2 = chat(
                base, mid, mk_msgs(prefix + q), max_tokens=256
            )
            dm = tick(K["matched"]) - m0
            rep.check(
                "bitrate_b.warm",
                st == 200 and dm > 0.5 * prefix_tok,
                f"matched +{dm}, {wall2:.1f}s",
            )
            rep.check(
                "bitrate_b.warm_served_ok", served_ok(text, b_cold_text), facts_hint
            )

    # -- session -------------------------------------------------------------
    # An agent-shaped conversation on a dedicated server: growing history
    # with streamed replies, tool messages, a mid-stream abort, sampled
    # turns, and a compaction rewrite. Retirement clones churn the record
    # LRU as depth grows; each turn adopts from the newest records, so the
    # grid floor must survive the eviction pressure.
    sess_final_ptok = None
    if a.session > 0:
        n_turns = a.session
        events_on = n_turns >= 10
        abort_t = (n_turns // 2) | 1
        sampled_t = abort_t + 2
        comp_t = n_turns - 2
        if not events_on:
            rep.note(
                "session.events",
                f"{n_turns} turns < 10: abort/sampled/compaction disabled",
            )
        sess_disk = os.path.join(out_dir, "apc-disk-session")
        os.makedirs(sess_disk, exist_ok=True)
        env_s = depth_env(
            sess_disk,
            a.block_size,
            a.num_blocks,
            a.disk_gb,
            scheme=a.scheme,
            bits=a.bits_a,
        )
        with ServerProc(
            serve_args(a.model, a.speculative, a.draft_gguf),
            env_extra=env_s,
            log_path=os.path.join(out_dir, "server-s.log"),
            python=a.python,
        ) as srv:
            srv.wait_ready(timeout=900)
            base = srv.base_url
            mid = model_id_of(base, srv)
            chat(base, mid, mk_msgs("Say ok."), max_tokens=4)

            root = (
                [] if a.no_system else [{"role": "system", "content": SESSION_SYSTEM}]
            )
            root = root + [{"role": "user", "content": prefix + q}]
            st, text, content, ptok, wall = chat(
                base, mid, root, max_tokens=SESSION_REPLY_TOKENS
            )
            settle(K["stores"])
            # the session runs under its own system message, so the log
            # witness recalibrates on this server's cold answer
            sess_witness = [(n, rx) for n, rx in probes if rx.search(text)]
            rep.check(
                "session.populate",
                st == 200 and len(text) > 0,
                f"prompt {ptok} tok, {wall:.1f}s, cold answer retrieves "
                f"{[n for n, _ in sess_witness]}",
            )

            def sess_facts(txt: str) -> bool:
                if not sess_witness:
                    return len(txt) > 0
                return all(rx.search(txt) for _, rx in sess_witness)

            history = [
                {"role": "assistant", "content": history_safe(content) or "(ok)"}
            ]
            known_ptok = ptok
            root_ptok = ptok
            dump_id = 0
            live_dumps = []
            odd_flip = False
            tool_state = "untried"
            turns_ok = clean_ok = mono_ok = floors_ok = True
            log_ok = near_ok = True
            near_n = far_n = far_hits = 0
            abort_ok = None
            comp_row = None
            floor_rows = []
            for t in range(2, n_turns + 1):
                is_abort = events_on and t == abort_t
                is_sampled = events_on and t == sampled_t
                is_comp = events_on and t == comp_t
                if is_comp:
                    # replace the middle of the history with a summary;
                    # divergence drops below every per-turn record, and the
                    # refused records must be re-stored, not crashed on
                    summary = (
                        "Summary of the session so far: diagnostics ran "
                        f"clean on probes 1 through {dump_id}; all readings "
                        "matched the maintenance log."
                    )
                    msgs = root + [
                        {"role": "assistant", "content": summary},
                        {"role": "user", "content": q},
                    ]
                    m0 = reuse_tick()
                    st, text, content, ptok_t, wall = chat(
                        base, mid, msgs, max_tokens=SESSION_REPLY_TOKENS
                    )
                    dm = reuse_tick() - m0
                    turns_ok &= st == 200 and len(text) > 0
                    clean_ok &= content_clean(content)
                    comp_row = (st, sess_facts(text), dm, wall)
                    history = msgs[len(root) :] + [
                        {
                            "role": "assistant",
                            "content": history_safe(content) or "(ok)",
                        }
                    ]
                    if ptok_t:
                        known_ptok = ptok_t
                    live_dumps = []
                    rep.note(
                        f"session.t{t}",
                        f"compaction status {st}, prompt {ptok_t} tok, "
                        f"matched +{dm}, {wall:.1f}s",
                    )
                    continue
                stream = t % 2 == 1
                extra = {}
                expect = None
                hist = history
                block = []
                if is_abort:
                    kind = "abort"
                    q_t = (
                        "Walk through the log's overall trends in as much "
                        "detail as possible."
                    )
                elif is_sampled:
                    kind = "sampled"
                    q_t = (
                        "Describe any notable anomalies across the log and "
                        "the diagnostic dumps so far."
                    )
                    extra = {"temperature": 0.7, "top_p": 0.9}
                elif t % 2 == 0:
                    kind = "tool"
                    dump_id += 1
                    j = dump_id
                    kq = (3 * t) % 40
                    q_t = (
                        f"In diagnostic dump {j}, what value does channel "
                        f"C{j}.{kq} show? Answer with the number, then "
                        "describe that channel in detail."
                    )
                    call = {
                        "id": f"call_{j}",
                        "type": "function",
                        "function": {
                            "name": "run_diagnostics",
                            "arguments": json.dumps({"probe": j}),
                        },
                    }
                    dump = tool_dump(j)
                    if tool_state == "unsupported":
                        block = [{"role": "user", "content": dump + "\n" + q_t}]
                    else:
                        hist = history[:-1] + [{**history[-1], "tool_calls": [call]}]
                        block = [
                            {
                                "role": "tool",
                                "tool_call_id": f"call_{j}",
                                "name": "run_diagnostics",
                                "content": dump,
                            },
                            {"role": "user", "content": q_t},
                        ]
                    expect = re.compile(rf"\b{dump_value(j, kq)}\b")
                else:
                    odd_flip = not odd_flip
                    if odd_flip or not live_dumps:
                        kind = "log"
                        q_t = q
                    else:
                        kind = "far"
                        j0 = live_dumps[0]
                        kq = (3 * t) % 40
                        q_t = (
                            f"Back in diagnostic dump {j0}, what value did "
                            f"channel C{j0}.{kq} show? Answer with the number."
                        )
                        expect = re.compile(rf"\b{dump_value(j0, kq)}\b")
                if not block:
                    block = [{"role": "user", "content": q_t}]
                msgs = root + hist + block
                # floor: the unit grid below the last known prompt length,
                # one unit of slack for render-shape key divergence
                floor = None
                if ck:
                    floor = max(0, turn_grid_floor(known_ptok) - CKPT_UNIT)
                elif a.tier == "block":
                    floor = max(
                        0,
                        ((root_ptok - QUESTION_SLACK) // a.block_size) * a.block_size,
                    )
                m0 = reuse_tick()
                if stream:
                    st, text, content, events, wall = schat(
                        base,
                        mid,
                        msgs,
                        max_tokens=1024 if is_abort else SESSION_REPLY_TOKENS,
                        abort_after=24 if is_abort else None,
                        **extra,
                    )
                    ptok_t = 0
                else:
                    st, text, content, ptok_t, wall = chat(
                        base, mid, msgs, max_tokens=SESSION_REPLY_TOKENS, **extra
                    )
                    events = 0
                if kind == "tool" and tool_state == "untried":
                    if st == 200:
                        tool_state = "ok"
                    else:
                        # template rejected the tool shape; resend the dump
                        # as plain user text and stay in that mode
                        tool_state = "unsupported"
                        hist = history
                        block = [{"role": "user", "content": dump + "\n" + q_t}]
                        msgs = root + hist + block
                        m0 = reuse_tick()
                        st, text, content, ptok_t, wall = chat(
                            base, mid, msgs, max_tokens=SESSION_REPLY_TOKENS
                        )
                dm = reuse_tick() - m0
                if is_abort:
                    abort_ok = st == 200 and events >= 24
                    settle(K["stores"])
                else:
                    turns_ok &= st == 200 and len(text) > 0
                    clean_ok &= content_clean(content)
                    if st == 200:
                        history = (
                            hist
                            + block
                            + [
                                {
                                    "role": "assistant",
                                    "content": history_safe(content) or "(ok)",
                                }
                            ]
                        )
                        if kind == "tool":
                            live_dumps.append(dump_id)
                    if ptok_t:
                        mono_ok &= ptok_t > known_ptok
                        known_ptok = ptok_t
                if floor is not None:
                    ok_f = dm >= floor
                    floors_ok &= ok_f
                    floor_rows.append(f"t{t}{'' if ok_f else '(FAIL)'}:+{dm}/{floor}")
                if kind == "log":
                    log_ok &= sess_facts(text)
                elif kind == "tool":
                    near_n += 1
                    near_ok &= bool(expect.search(text))
                elif kind == "far":
                    far_n += 1
                    far_hits += bool(expect.search(text))
                shape = f"stream {events}ev" if stream else f"prompt {ptok_t} tok"
                rep.note(
                    f"session.t{t}",
                    f"{kind} status {st}, {shape}, matched +{dm}, {wall:.1f}s",
                )
            sess_final_ptok = known_ptok
            settle(K["stores"])
            s1 = stats(base)
            rep.check(
                "session.turns",
                turns_ok,
                f"{n_turns} turns, final prompt {known_ptok} tok",
            )
            rep.check(
                "session.content_clean",
                clean_ok,
                "content fields markup-free across the session",
            )
            rep.check("session.growing", mono_ok, "prompt grows at full-history turns")
            if a.tier == "exact":
                rep.note("session.adopted", " ".join(floor_rows) or "n/a")
            else:
                rep.check("session.adopted", floors_ok, " ".join(floor_rows))
            rep.check(
                "session.log_facts",
                log_ok,
                f"log probes retrieve {[n for n, _ in sess_witness]}",
            )
            if near_n:
                rep.check(
                    "session.dump_facts",
                    near_ok,
                    f"{near_n} fresh-dump probes answered",
                )
            rep.note(
                "session.tool_shape",
                "tool_calls + tool role rendered"
                if tool_state == "ok"
                else f"tool messages sent as plain text ({tool_state})",
            )
            if far_n:
                rep.note(
                    "session.far_dumps",
                    f"{far_hits}/{far_n} distant-dump retrievals",
                )
            if events_on:
                rep.check(
                    "session.abort",
                    bool(abort_ok),
                    f"stream at t{abort_t} aborted after ~24 events; "
                    "recovery is covered by session.turns and .adopted",
                )
                st_c, facts_c, dm_c, wall_c = comp_row
                rep.check(
                    "session.compaction",
                    st_c == 200 and facts_c,
                    f"rewrite served with the calibrated facts, matched "
                    f"+{dm_c} (depth noted: pre-rewrite records may be "
                    f"evicted), {wall_c:.1f}s",
                )
            if ck:
                declines = s1.get("ckpt_declines") or {}
                rep.check(
                    "session.no_exception_declines",
                    int(declines.get("exception", 0) or 0) == 0,
                    json.dumps(declines),
                )
                rep.check(
                    "session.no_missed_adoptions",
                    int(s1.get("ckpt_missed_adoptions", 0) or 0) == 0,
                    f"ckpt_missed_adoptions={s1.get('ckpt_missed_adoptions')}",
                )
                rep.note(
                    "session.counters",
                    json.dumps(
                        {
                            k: s1.get(k)
                            for k in (
                                "ckpt_stores",
                                "ckpt_hits",
                                "ckpt_matched_tokens",
                                "ckpt_pool_evictions",
                                "matched_tokens",
                                "disk_writes",
                            )
                        }
                    ),
                )

    # -- tripwire ------------------------------------------------------------
    # With replay and turn boundaries off, an identical resend is refused
    # by the p < len(query) bound and the missed-adoption warning must
    # fire. The prompt must tokenize under unit plus guard (no adoptable
    # terminal boundary) yet past the rotating window (windows over ~1300
    # tokens need --no-tripwire). Disk stays off so a retire skeleton
    # cannot serve the resend, and the system message stays short even
    # under --system-words: an agent-shaped system block arms the anchor,
    # which then serves the resends exactly as designed and leaves this
    # phase with nothing to refuse.
    if ck and not a.no_tripwire:
        trip_prefix, n_trip = deep_prefix(1050)
        trip_q = (
            f"Question: which technician logged entry {n_trip // 2}? "
            "Answer with the name."
        )
        trip_env = dict(env_a)
        trip_env.pop("APC_DISK_PATH", None)
        trip_env.pop("APC_DISK_MAX_GB", None)
        trip_env["GMLX_APC_CKPT_REPLAY"] = "0"
        trip_env["GMLX_APC_CKPT_TURN"] = "0"
        trip_env["GMLX_APC_CKPT_TRIPWIRE"] = "2"
        trip_log = os.path.join(out_dir, "server-c.log")
        with ServerProc(
            serve_args(a.model, a.speculative, a.draft_gguf),
            env_extra=trip_env,
            log_path=trip_log,
            python=a.python,
        ) as srv:
            srv.wait_ready(timeout=900)
            base = srv.base_url
            mid = model_id_of(base, srv)
            sts = []
            for _ in range(4):  # populate + 3 refused resends
                st, _, _, _, _ = chat(
                    base, mid,
                    mk_msgs(trip_prefix + trip_q, system=SYSTEM_MSG),
                    max_tokens=32,
                )
                sts.append(st)
            settle("ckpt_stores")
            s = stats(base)
            rep.check(
                "tripwire.setup",
                all(st == 200 for st in sts)
                and int(s.get("ckpt_stores", 0) or 0) > 0
                and int(s.get("ckpt_hits", 0) or 0) == 0,
                f"stores={s.get('ckpt_stores')} hits={s.get('ckpt_hits')} "
                "(p=N/retire records only, resends must be refused)",
            )
            rep.check(
                "tripwire.missed_counted",
                int(s.get("ckpt_missed_adoptions", 0) or 0) >= 3,
                f"ckpt_missed_adoptions={s.get('ckpt_missed_adoptions')} "
                ">= 3 refused resends",
            )
        with open(trip_log, encoding="utf-8", errors="replace") as f:
            log_text = f.read()
        rep.check(
            "tripwire.warning_logged",
            "APC ckpt tripwire" in log_text and "prefix-matching record" in log_text,
            "one-time missed-adoption warning in server-c.log",
        )

    report = {
        "label": label,
        "model": a.model,
        "tier": a.tier,
        "scheme": a.scheme,
        "bits": [a.bits_a, a.bits_b] if (a.bits_a or a.bits_b) else None,
        "prefix_tokens": prefix_tok,
        "session_turns": a.session or None,
        "session_final_ptok": sess_final_ptok,
        "speculative": a.speculative,
        "system_message": not a.no_system,
        "system_words": a.system_words or None,
        "template_kwargs": template_kwargs,
        "cold_wall_s": round(cold_wall, 1),
        "warm_wall_s": round(warm_wall, 1),
        "disk_wall_s": round(disk_wall, 1),
        "shards": n_shards,
        "rows": rep.rows,
        "failures": rep.failures,
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"== {label}: {'PASS' if not rep.failures else 'FAIL'} "
        f"({len(rep.failures)} failures) -> {out_dir}/report.json =="
    )
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
