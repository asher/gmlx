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

Correctness is asserted, not assumed: the prefix is a synthetic log whose
every entry is computable (entry_facts), the probe question targets entries
at 1/4 and 1/2 depth, and every cache-served answer (warm, restart, reset)
must retrieve those facts or reproduce the cold answer byte-for-byte.

Phases against one model:

  populate      cold server, deep-prefix request -> tier stores and disk
                write-through; ckpt additionally proves mid-prefill interval
                boundaries advanced across the > interval prompt
  warm          identical replay -> tier matched counter climbs by roughly
                the prefix, wall time collapses vs cold, facts still right
  divergent     same prefix, different question -> ckpt adopts a grid/render-
                stable boundary (the class the p=N-only store orphaned)
  turns         a real conversation through the real chat template: each
                turn adopts a render-stable boundary (matched grows), prompts
                grow monotonically -> the render_ctx production path all
                unit tests fake
  concurrent    N clients sharing the deep prefix; ckpt formation is B=1 by
                design, so the burst asserts formation correctness
  restart       fresh process, same APC_DISK_PATH -> skeletons re-index,
                replay repairs from disk (disk_hits, ckpt_hits move) fast
  reset         /v1/cache/reset clears memory, not disk -> replay hits disk
  bitrate-b     (only with --bits-b) fresh server at a second KV width on the
                SAME disk root -> no cross-adoption, own namespace warms

--speculative adds MTP to every server (spec-path ckpt arming + sidecar).
`--warm-factor`/`--restart-factor` loosen the wall-clock collapse thresholds
(ckpt restores clone >100 MB of GDN state; default 0.6x cold).

Not ``test_``-prefixed: needs the GPU and a large local GGUF. Run directly::

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
import time
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

# GMLX_APC_CKPT_INTERVAL default: mid-prefill ckpt boundaries land on this grid.
CKPT_INTERVAL = 4096

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


def entry_facts(i: int) -> dict:
    """Ground truth for log entry i (mirrors deep_prefix's formulas)."""
    return {
        "topic": _TOPICS[i % len(_TOPICS)],
        "reading": (i * 7) % 97,
        "technician": "ana" if i % 2 else "bo",
    }


def deep_prefix(target_words: int) -> tuple:
    """Deterministic, non-repetitive briefing text (~1.3 tok/word). Content
    varies per sentence so the APC block chain is non-degenerate, and every
    entry's facts are computable (entry_facts) so answers can be verified.
    Returns (text, n_entries)."""
    out = [
        "You are the flight systems engineer on duty. Read the maintenance "
        "log below carefully; every entry matters.\n\nMaintenance log:\n"
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


def chat(
    base: str, mid: str, messages: list, *, max_tokens: int = 32, timeout: float = 900.0
):
    t0 = time.monotonic()
    st, body = Client(base, timeout=timeout).chat(
        mid, messages, max_tokens=max_tokens, temperature=0.0
    )
    wall = time.monotonic() - t0
    text = ""
    content = ""
    ptok = 0
    if isinstance(body, dict):
        try:
            msg = body["choices"][0]["message"]
            # text includes any reasoning stream: a fact retrieved
            # mid-think is just as valid a non-corruption witness as one
            # in the answer. content is the clean field alone -- what a
            # real client feeds back as history (reasoning re-sent as
            # content breaks channel-tagged templates like gpt-oss).
            content = str(msg.get("content") or "")
            text = "".join(
                str(msg.get(k) or "")
                for k in ("reasoning_content", "reasoning", "content")
            )
        except (KeyError, IndexError, TypeError):
            pass
        ptok = int((body.get("usage") or {}).get("prompt_tokens", 0) or 0)
    return st, text, content, ptok, wall


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
    disk_root: str, block_size: int, num_blocks: int, disk_gb: int,
    *, scheme: str = None, bits: str = None,
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


def serve_args(model_path: str, speculative: bool) -> list:
    args = [model_path, "--no-auth"]
    if speculative:
        args.append("--speculative")
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
    ap.add_argument("--concurrency", type=int, default=3)
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
    ap.add_argument("--speculative", action="store_true")
    ap.add_argument("--out", required=True, help="artifact dir (logs, report.json)")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    label = a.label or os.path.basename(a.model)
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    disk_root = os.path.join(out_dir, "apc-disk")
    os.makedirs(disk_root, exist_ok=True)
    rep = Report()
    K = tier_keys(a.tier)
    ck = a.tier == "ckpt"
    prefix, n_entries = deep_prefix(a.prefix_words)
    # Correctness probes live at 1/4 and 1/2 log depth: deep inside the
    # cache-served region, past any fp16 sink and clear of the tail - a
    # restored cache that decoded to garbage cannot answer them.
    e1, e2 = n_entries // 4, n_entries // 2
    f1, f2 = entry_facts(e1), entry_facts(e2)
    q = (
        f"Question: how many units did entry {e1} report, and which "
        f"technician logged entry {e2}? Answer with the number and the name."
    )

    def facts_ok(text: str) -> bool:
        return bool(
            re.search(rf"\b{f1['reading']}\b", text)
            and re.search(rf"\b{f2['technician']}\b", text, re.IGNORECASE)
        )

    def served_ok(text: str, cold: str) -> bool:
        # A cache-served answer is sound if it retrieves the facts, or if
        # it reproduces the cold answer byte-for-byte (fidelity fallback:
        # a model that misreads the log cold must misread it identically
        # warm - divergence, not wrongness, is the corruption signal).
        return facts_ok(text) or text == cold

    facts_hint = (
        f"entry {e1} -> {f1['reading']} units, entry {e2} -> {f2['technician']}"
    )

    def tick(key: str) -> int:
        return int(stats(base).get(key, 0) or 0)

    print(f"== APC depth e2e: {label} ==")
    print(
        f"   tier {a.tier} | ~{a.prefix_words} words | scheme={a.scheme or 'fp16'} | "
        f"speculative={a.speculative} | out {out_dir}"
    )

    env_a = depth_env(
        disk_root, a.block_size, a.num_blocks, a.disk_gb,
        scheme=a.scheme, bits=a.bits_a,
    )
    t_load0 = time.monotonic()
    with ServerProc(
        serve_args(a.model, a.speculative),
        env_extra=env_a,
        log_path=os.path.join(out_dir, "server-a.log"),
        python=a.python,
    ) as srv:
        srv.wait_ready(timeout=900)
        rep.note("load-a", f"{time.monotonic() - t_load0:.0f}s")
        base = srv.base_url
        mid = model_id_of(base, srv)
        # absorb first-request one-time costs (kernel warmup) so wall-time
        # comparisons below measure caching, not JIT
        chat(base, mid, [{"role": "user", "content": "Say ok."}], max_tokens=4)

        # -- populate ------------------------------------------------------
        st, text, content, ptok, cold_wall = chat(
            base, mid, [{"role": "user", "content": prefix + q}], max_tokens=256
        )
        cold_text = text
        s = stats(base)
        rep.check("populate.status", st == 200, f"prompt {ptok} tok, {cold_wall:.1f}s")
        rep.check("populate.answer", len(text) > 0, repr(text[:70]))
        rep.check("populate.facts", facts_ok(text), facts_hint)
        rep.check(
            "populate.stores",
            int(s.get(K["stores"], 0)) > 0,
            f"{K['stores']}={s.get(K['stores'])}",
        )
        if ck:
            # mid-prefill boundaries: a > interval prompt must land interval
            # grid stores, not just the terminal/replay pair
            floor = ptok // CKPT_INTERVAL + 1
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
        # Identical replay: ckpt adopts the N-1 replay record, exact needs a
        # verbatim stored prefix, block walks the grid - all three must
        # produce a near-full matched prefix on their own tier counter.
        m0 = tick(K["matched"])
        st, text, content, ptok, warm_wall = chat(
            base, mid, [{"role": "user", "content": prefix + q}], max_tokens=256
        )
        dm = tick(K["matched"]) - m0
        rep.check(
            "warm.status", st == 200, f"{warm_wall:.1f}s vs cold {cold_wall:.1f}s"
        )
        rep.check(
            "warm.matched",
            dm > 0.5 * prefix_tok,
            f"{K['matched']} +{dm} of ~{prefix_tok}",
        )
        rep.check(
            "warm.faster",
            warm_wall < a.warm_factor * cold_wall,
            f"{warm_wall:.1f}s < {a.warm_factor}x{cold_wall:.1f}s",
        )
        # the adopted-cache answer must retrieve the same record-region facts
        rep.check("warm.served_ok", served_ok(text, cold_text), facts_hint)
        rep.note("warm.same_as_cold", str(text == cold_text))

        # -- divergent suffix ----------------------------------------------
        # Same prefix, different question: the ckpt tier must adopt a grid /
        # render-stable boundary below the divergence point (the class bug 1
        # orphaned when only p=N records existed). Exact tier misses by
        # design; block tier reuses the shared grid.
        m0 = tick(K["matched"])
        st, _, _, _, div_wall = chat(
            base,
            mid,
            [
                {
                    "role": "user",
                    "content": prefix + "Question: which subsystem appears in entry 7?",
                }
            ],
        )
        dm = tick(K["matched"]) - m0
        if ck:
            rep.check(
                "divergent.ckpt_boundary_adopted",
                dm > 0,
                f"{K['matched']} +{dm}, {div_wall:.1f}s",
            )
        else:
            rep.note(
                "divergent-suffix",
                f"status {st}, matched +{dm} "
                f"({'0 expected' if a.tier == 'exact' else 'grid reuse expected'}), "
                f"{div_wall:.1f}s",
            )

        # -- turns ---------------------------------------------------------
        # A real conversation through the real chat template: this is the
        # production render_ctx path (think-stripping and all) that every
        # unit test fakes with synthetic ids.
        def history_safe(reply: str) -> str:
            # A length-capped reasoning reply can leak channel markup into
            # the content field (server parser gap, noted in the report);
            # templates reject it, so feed back plain text only.
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
        turns_ok, monotone = True, True
        turn_dms = []
        # ground truth for turn 1's question: entry 12 reads (12*7)%97 = 84
        for t_i in range(a.turns):
            history.append((turn_qs[t_i % len(turn_qs)],
                            history_safe(last_answer)))
            msgs = [{"role": "user", "content": prefix + q}]
            for uq, at in history:
                msgs.append({"role": "assistant", "content": at})
                msgs.append({"role": "user", "content": uq})
            m0 = tick(K["matched"])
            st, turn_text, last_answer, ptok, wall = chat(
                base, mid, msgs, max_tokens=96)
            dm = tick(K["matched"]) - m0
            turn_dms.append(dm)
            turns_ok &= st == 200 and len(turn_text) > 0
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
        rep.check("turns.growing", monotone, "templated prompt grows per turn")
        if a.tier != "exact":
            # every turn must adopt a stored boundary at least as deep as the
            # ckpt grid floor / block grid below the shared prefix
            rep.check(
                "turns.adopted",
                min(turn_dms) > 0.4 * prefix_tok,
                f"min +{min(turn_dms)} of ~{prefix_tok} across {a.turns} turns",
            )

        # -- concurrent ----------------------------------------------------
        def fire(i):
            return chat(
                base,
                mid,
                [
                    {
                        "role": "user",
                        "content": prefix
                        + f"Question {i}: list {i + 2} subsystem names "
                        "from the log.",
                    }
                ],
                max_tokens=96,
            )

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            res = list(ex.map(fire, range(a.concurrency)))
        rep.check(
            "concurrent.status",
            all(st == 200 and text for st, text, _, _, _ in res),
            f"{a.concurrency} clients, {time.monotonic() - t0:.1f}s wall",
        )
        rep.check(
            # every batched reply must name real subsystems from the log:
            # a corrupted batch row degenerates to text that names none
            "concurrent.coherent",
            all(any(t in text.lower() for t in _TOPICS) for _, text, _, _, _ in res),
            "each reply names >=1 log subsystem",
        )
        if ck:
            rep.note(
                "concurrent.ckpt_scope",
                "ckpt retirement is B=1 by design; the burst asserts "
                "formation correctness, not per-row stores",
            )

        counter_keys = [
            K["stores"], K["hits"], K["matched"],
            "disk_writes", "disk_files", "lookups_hit", "matched_tokens",
            "disk_hits",
        ]
        if ck:
            counter_keys.append("ckpt_declines")
        pre_restart = stats(base)
        rep.note(
            "counters-a",
            json.dumps({k: pre_restart.get(k) for k in counter_keys}),
        )
        wait_disk_drained(base, min_files=1, timeout=120)
    n_shards = len(shard_files(disk_root))
    rep.check("shards.on_disk", n_shards > 0, f"{n_shards} safetensors shards")

    # -- restart -----------------------------------------------------------
    with ServerProc(
        serve_args(a.model, a.speculative),
        env_extra=env_a,
        log_path=os.path.join(out_dir, "server-a2.log"),
        python=a.python,
    ) as srv:
        srv.wait_ready(timeout=900)
        base = srv.base_url
        mid = model_id_of(base, srv)
        chat(base, mid, [{"role": "user", "content": "Say ok."}], max_tokens=4)
        st, text, _, ptok, disk_wall = chat(
            base, mid, [{"role": "user", "content": prefix + q}], max_tokens=256
        )
        s = stats(base)
        rep.check(
            "restart.disk_hit",
            int(s.get("disk_hits", 0) or 0) > 0,
            f"disk_hits={s.get('disk_hits')}, {disk_wall:.1f}s "
            f"vs cold {cold_wall:.1f}s",
        )
        if ck:
            # skeleton repair: the fresh process rebuilt a usable record from
            # the on-disk skeleton and the replay adopted it on the ckpt tier
            rep.check(
                "restart.skeleton_repair",
                int(s.get("ckpt_hits", 0) or 0) > 0,
                f"ckpt_hits={s.get('ckpt_hits')} "
                f"ckpt_matched_tokens={s.get('ckpt_matched_tokens')}",
            )
        # the strongest corruption probe in the file: this KV crossed a
        # process boundary through safetensors shards before answering
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

        # -- reset ---------------------------------------------------------
        st_reset, _ = Client(base).post("/v1/cache/reset")
        h0 = tick("disk_hits")
        st, text, _, ptok, wall = chat(
            base, mid, [{"role": "user", "content": prefix + q}], max_tokens=256
        )
        h1 = tick("disk_hits")
        rep.check(
            "reset.disk_survives",
            st_reset == 200 and st == 200 and h1 > h0,
            f"disk_hits {h0} -> {h1}, {wall:.1f}s",
        )
        rep.check("reset.served_ok", served_ok(text, cold_text), facts_hint)

    # -- bitrate-b isolation ------------------------------------------------
    # Only meaningful when a second KV width is requested: the namespaces
    # must not cross-adopt. Skipped entirely on the fp16 acceptance shape.
    if a.bits_b:
        env_b = depth_env(
            disk_root, a.block_size, a.num_blocks, a.disk_gb,
            scheme=a.scheme, bits=a.bits_b,
        )
        with ServerProc(
            serve_args(a.model, a.speculative),
            env_extra=env_b,
            log_path=os.path.join(out_dir, "server-b.log"),
            python=a.python,
        ) as srv:
            srv.wait_ready(timeout=900)
            base = srv.base_url
            mid = model_id_of(base, srv)
            chat(base, mid, [{"role": "user", "content": "Say ok."}], max_tokens=4)
            st, text, _, ptok, wall = chat(
                base, mid, [{"role": "user", "content": prefix + q}], max_tokens=256
            )
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
                base, mid, [{"role": "user", "content": prefix + q}], max_tokens=256
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

    report = {
        "label": label,
        "model": a.model,
        "tier": a.tier,
        "scheme": a.scheme,
        "bits": [a.bits_a, a.bits_b] if (a.bits_a or a.bits_b) else None,
        "prefix_tokens": prefix_tok,
        "speculative": a.speculative,
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
