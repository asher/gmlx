#!/usr/bin/env python3
"""End-to-end test for the disk-backed Automatic Prefix Cache (APC) over real HTTP.

Exercises the SSD prompt-cache tier the way a user gets it - `APC_ENABLED=1` plus
`APC_DISK_PATH=<dir>` - across real `gmlx serve` subprocesses, and proves the
cache **survives a server restart** plus the properties that make that useful:

  populate    a cold server writes prompt blocks through to disk (shards on the FS,
              disk_writes > 0) and the first request is an all-miss;
  warm        replaying the same prompt on the same server hits the in-memory tier
              (lookups_hit / matched_tokens climb);
  batching    several concurrent clients sharing one long prefix all succeed AND
              reuse the same cached prefix - APC is shared across the continuous-
              batching engine, not per-connection;
  restart     a fresh server process with the same APC_DISK_PATH + model re-indexes
              the shards at load and the same prompt now hits the DISK tier
              (disk_hits > 0) - the cache outlived the process;
  reset       POST /v1/cache/reset clears the in-memory pool but NOT the disk tier,
              so the next replay hits disk again;
  isolation   a different model (different namespace) sharing the same APC_DISK_PATH
              does NOT see the first model's blocks (disk_hits == 0) and writes its
              own namespace subdirectory.

Both servers run as real subprocesses (`gmlx.server`), so argparse, the
monkeypatch wiring, uvicorn, the residency pool, and mlx-vlm's APC all get
exercised end to end.

Not ``test_``-prefixed, so pytest skips it - it needs the GPU and a base GGUF. Run
it directly with the project interpreter::

    python tests/e2e/run_apc_disk_e2e.py
    python tests/e2e/run_apc_disk_e2e.py --concurrency 6
    python tests/e2e/run_apc_disk_e2e.py --keep --out ./apc-e2e-out
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import Client                  # noqa: E402
from models import ModelRegistry           # noqa: E402
from server_proc import ServerProc         # noqa: E402

# A long, deterministic shared prefix. With APC_BLOCK_SIZE=16 this is many full
# blocks, so a single request seals a dozen-plus blocks to disk and a replay has a
# large matched-token prefix. Distinct sentences (not repeated padding) so the block
# chain is non-degenerate.
PREFIX = (
    "You are a meticulous systems engineer. Read the following background carefully "
    "and keep every detail in mind when you answer.\n\n"
    "Background. Apple silicon uses a unified memory architecture: the CPU, GPU, and "
    "Neural Engine share one pool of high-bandwidth memory, so a tensor produced on "
    "the GPU is visible to the CPU without a copy. The M5 Max in this machine has a "
    "wide memory bus and a large last-level cache. Quantized model weights are stored "
    "as K-quant blocks; at decode time each block is dequantized on the fly inside a "
    "Metal kernel, which keeps the working set small and the memory traffic bounded. "
    "A prompt cache stores the key/value tensors for a prefix of tokens so that a "
    "later request sharing that prefix can skip recomputing it. When the prefix cache "
    "is backed by an SSD tier, those key/value blocks are written to disk as shards "
    "and can be restored after the server process restarts, which matters for a long "
    "coding session whose system prompt never changes between turns.\n\n"
    "Instruction. Use only the background above. Be concise and precise.\n"
)
# Fixed suffix for the single-prompt (cold / warm / restart / reset) path: identical
# every time so the prefix hashes match across processes.
FIXED_Q = "Question: in one sentence, what is unified memory?"


def chat(base_url: str, model_id: str, prompt: str, *, max_tokens: int = 8,
         timeout: float = 600.0):
    status, body = Client(base_url, timeout=timeout).chat(
        model_id, [{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0)
    return status, body


def stats(base_url: str) -> dict:
    status, body = Client(base_url).cache_stats()
    return body if isinstance(body, dict) else {"_raw": body, "_status": status}


def model_id_of(base_url: str, proc: ServerProc) -> str:
    status, body = Client(base_url).models()
    if status != 200 or not (body or {}).get("data"):
        raise RuntimeError(f"/v1/models failed ({status}): {body}\n" + proc.log_tail())
    return body["data"][0]["id"]


def shard_files(disk_root: str) -> list:
    return sorted(str(p) for p in Path(disk_root).rglob("*.safetensors"))


def ns_dirs(disk_root: str) -> list:
    """Namespace subdirectories that actually hold shards (one per served model)."""
    root = Path(disk_root)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and any(d.glob("*.safetensors")))


def wait_disk_drained(base_url: str, *, min_files: int = 1, timeout: float = 30.0,
                      poll: float = 0.4) -> dict:
    """Poll cache stats until the async disk writer has flushed >= ``min_files``
    shards (so a subsequent restart reads a fully-written tier). Returns the last
    stats seen."""
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        last = stats(base_url)
        if int(last.get("disk_files", 0) or 0) >= min_files:
            return last
        time.sleep(poll)
    return last


def apc_env(disk_root: str, block_size: int, num_blocks: int) -> dict:
    # A small in-memory pool forces the SSD tier to engage deterministically: a
    # concurrent batch overflows it, so prefix blocks evict + write through to disk
    # within the session (with the default large pool the disk tier still persists,
    # but only flushes lazily/at teardown - harder to observe in a fast test).
    return {
        "APC_ENABLED": "1",
        "APC_DISK_PATH": disk_root,
        "APC_BLOCK_SIZE": str(block_size),
        "APC_NUM_BLOCKS": str(num_blocks),
        "APC_DISK_MAX_GB": "4",
    }


# ---- phases -----------------------------------------------------------------

def run_batch(base, mid, concurrency, *, suffix_tag):
    """Fire ``concurrency`` requests that share PREFIX but carry distinct suffixes,
    concurrently. Returns (statuses, wall_s, serial_sum_s)."""
    prompts = [PREFIX + f"\n{suffix_tag} {i}: list {i + 2} facts from the background."
               for i in range(concurrency)]

    def _fire(p):
        t0 = time.monotonic()
        st, _bd = chat(base, mid, p, max_tokens=24)
        return st, time.monotonic() - t0

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        res = list(ex.map(_fire, prompts))
    return [st for st, _ in res], time.monotonic() - t0, sum(dt for _, dt in res)


def phase_sequential(python, model_path, disk_root, log_dir, block_size,
                     num_blocks) -> dict:
    """The headline single-user case: PURELY SEQUENTIAL lone requests - never two at
    once - must populate the SSD tier. The server harvests prefill K/V into prefix
    blocks at the end of each prefill; a lone request takes a fast path that builds a
    plain KVCache, which the stock harvest could not read (it sliced a batched cache
    by ``_idx``), so a solo coding session stored nothing. With the lone-harvest fix a
    single request must store + write through to disk, a later turn must reuse the
    shared prefix, and the tier must survive a restart - all with no concurrency."""
    out = {}
    turn1 = PREFIX + FIXED_Q
    turn2 = (PREFIX + FIXED_Q + "\nAnswer: one shared CPU/GPU memory pool.\n"
             "Follow-up: why does that help the prompt cache?")
    with ServerProc([model_path],
                    env_extra=apc_env(disk_root, block_size, num_blocks),
                    log_path=os.path.join(log_dir, "serve-seq.log"),
                    python=python) as proc:
        proc.wait_ready()
        base, mid = proc.base_url, model_id_of(proc.base_url, proc)
        _ms, m_body = Client(base).metrics()
        srv = (m_body or {}).get("server") if isinstance(m_body, dict) else {}
        out["continuous_batching_enabled"] = bool((srv or {}).get(
            "continuous_batching_enabled"))
        # turn 1 - a single, lone request (no concurrency). It must harvest its
        # prefix to memory AND write through to the SSD tier.
        s1, _ = chat(base, mid, turn1)
        out["turn1_status"] = s1
        drained = wait_disk_drained(base, min_files=1)
        out["turn1_stores"] = int(drained.get("stores", 0))
        out["turn1_disk_writes"] = int(drained.get("disk_writes", 0))
        out["turn1_disk_files"] = int(drained.get("disk_files", 0))
        # turn 2 - a later lone request that shares turn 1's prefix -> reuse it
        before = stats(base)
        s2, _ = chat(base, mid, turn2)
        after = stats(base)
        out["turn2_status"] = s2
        out["turn2_matched_delta"] = (int(after.get("matched_tokens", 0))
                                      - int(before.get("matched_tokens", 0)))
        out["turn2_hit_delta"] = (int(after.get("lookups_hit", 0))
                                  - int(before.get("lookups_hit", 0))
                                  + int(after.get("disk_hits", 0))
                                  - int(before.get("disk_hits", 0)))
    out["shards_after_stop"] = len(shard_files(disk_root))
    # restart - a fresh process replays turn 1; it must restore from the disk tier
    # the sequential traffic alone wrote.
    with ServerProc([model_path],
                    env_extra=apc_env(disk_root, block_size, num_blocks),
                    log_path=os.path.join(log_dir, "serve-seq2.log"),
                    python=python) as proc:
        proc.wait_ready()
        base, mid = proc.base_url, model_id_of(proc.base_url, proc)
        before = stats(base)
        s3, _ = chat(base, mid, turn1)
        after = stats(base)
        out["restart_status"] = s3
        out["restart_disk_hit_delta"] = (int(after.get("disk_hits", 0))
                                         - int(before.get("disk_hits", 0)))
        out["restart_matched"] = int(after.get("matched_tokens", 0))
    return out


def phase_populate_warm_batch(python, model_path, disk_root, log_dir, block_size,
                              num_blocks, concurrency) -> dict:
    """Populate the cache with a concurrent multi-client batch (the store path is
    triggered by batched contention, not by a lone sequential request), confirm the
    SSD tier was written through, then prove a within-session prefix replay reuses it.
    Also the multi-client / batching answer: the concurrent clients all succeed and
    their shared prefix is what gets persisted."""
    out = {"concurrency": concurrency}
    with ServerProc([model_path],
                    env_extra=apc_env(disk_root, block_size, num_blocks),
                    log_path=os.path.join(log_dir, "serve-A1.log"),
                    python=python) as proc:
        proc.wait_ready()
        base, mid = proc.base_url, model_id_of(proc.base_url, proc)

        # continuous batching must be on for the batching claim to mean anything
        _ms, m_body = Client(base).metrics()
        srv = (m_body or {}).get("server") if isinstance(m_body, dict) else {}
        out["continuous_batching_enabled"] = bool((srv or {}).get(
            "continuous_batching_enabled"))

        # 1) one lone request first: with the lone-harvest fix this already stores
        #    its prefix (the dedicated sequential phase asserts that in isolation);
        #    here it just primes before the concurrent batch below.
        s, _ = chat(base, mid, PREFIX + FIXED_Q)
        out["cold_status"] = s
        out["cold"] = stats(base)

        # 2) batching / multi-client: concurrent clients sharing PREFIX. This is the
        #    store trigger - the engine commits the shared prefix to APC and writes it
        #    through to the SSD tier.
        statuses, wall, serial = run_batch(base, mid, concurrency, suffix_tag="Q")
        out["batch_statuses"] = statuses
        out["batch_all_ok"] = all(st == 200 for st in statuses)
        out["batch_wall_s"] = round(wall, 2)
        out["batch_serial_estimate_s"] = round(serial, 2)
        post = wait_disk_drained(base, min_files=1)
        out["post_batch"] = post
        out["shards_on_fs_live"] = len(shard_files(disk_root))

        # 3) within-session warm replay: now that the batch populated PREFIX, a single
        #    replay reuses it (in-memory and/or disk tier) - the cache works mid-session.
        before = stats(base)
        s3, _ = chat(base, mid, PREFIX + FIXED_Q)
        after = stats(base)
        out["warm_status"] = s3
        out["warm_hit_delta"] = (int(after.get("lookups_hit", 0))
                                 - int(before.get("lookups_hit", 0)))
        out["warm_matched_delta"] = (int(after.get("matched_tokens", 0))
                                     - int(before.get("matched_tokens", 0)))
        out["warm_disk_hit_delta"] = (int(after.get("disk_hits", 0))
                                      - int(before.get("disk_hits", 0)))
    # server stopped; the persisted tier should remain on disk
    out["shards_on_fs_after_stop"] = len(shard_files(disk_root))
    return out


def phase_restart_and_reset(python, model_path, disk_root, log_dir, block_size,
                            num_blocks) -> dict:
    """Fresh process, same APC_DISK_PATH + model: the same prompt must hit the DISK
    tier (cache survived the restart), and an in-memory reset must NOT wipe disk."""
    out = {}
    with ServerProc([model_path],
                    env_extra=apc_env(disk_root, block_size, num_blocks),
                    log_path=os.path.join(log_dir, "serve-A2.log"),
                    python=python) as proc:
        proc.wait_ready()
        base, mid = proc.base_url, model_id_of(proc.base_url, proc)

        # replay the exact cold prompt -> should restore from disk
        s, _ = chat(base, mid, PREFIX + FIXED_Q)
        out["replay_status"] = s
        after = stats(base)
        out["after_restart"] = after
        out["disk_hits_after_restart"] = int(after.get("disk_hits", 0))
        out["disk_blocks_indexed"] = int(after.get("disk_blocks_indexed", 0))
        out["matched_tokens_after_restart"] = int(after.get("matched_tokens", 0))

        # in-memory reset, then replay again: disk tier must persist through it
        rs, _ = Client(base).cache_reset()
        out["reset_status"] = rs
        before2 = stats(base)
        s2, _ = chat(base, mid, PREFIX + FIXED_Q)
        out["replay2_status"] = s2
        after2 = stats(base)
        out["disk_hit_delta_after_reset"] = (int(after2.get("disk_hits", 0))
                                             - int(before2.get("disk_hits", 0)))
        out["after_reset_replay"] = after2
    return out


def phase_isolation(python, model_path, disk_root, log_dir, block_size, num_blocks,
                    concurrency) -> dict:
    """A different model shares the same APC_DISK_PATH. Its first replay of the exact
    same prefix must NOT read the first model's shards (separate namespace -> disk
    miss), and its own batch writes its own namespace subdirectory."""
    out = {}
    ns_before = ns_dirs(disk_root)
    with ServerProc([model_path],
                    env_extra=apc_env(disk_root, block_size, num_blocks),
                    log_path=os.path.join(log_dir, "serve-B.log"),
                    python=python) as proc:
        proc.wait_ready()
        base, mid = proc.base_url, model_id_of(proc.base_url, proc)
        # B's first request replays A's exact prefix: its namespace is empty, so it
        # cannot read A's shards -> a clean disk miss (isolation).
        s, _ = chat(base, mid, PREFIX + FIXED_Q)
        out["status"] = s
        out["cold_disk_hits"] = int(stats(base).get("disk_hits", 0))
        # B's own batch populates B's namespace
        out["batch_statuses"], _w, _s = run_batch(base, mid, concurrency,
                                                   suffix_tag="B")
        wait_disk_drained(base, min_files=1)
        out["disk_writes"] = int(stats(base).get("disk_writes", 0))
    ns_after = ns_dirs(disk_root)
    out["ns_before"] = ns_before
    out["ns_after"] = ns_after
    out["new_namespace"] = len(ns_after) > len(ns_before)
    return out


# ---- grading ----------------------------------------------------------------

def grade(seq, pop, restart, iso, out_dir: Path) -> int:
    checks = {}
    # SEQUENTIAL single-user populate (the headline: lone requests must store)
    checks["sequential turn-1 200"] = seq.get("turn1_status") == 200
    checks["SEQUENTIAL lone request stores (stores>0)"] = seq.get("turn1_stores", 0) > 0
    checks["SEQUENTIAL lone request writes disk (disk_writes>0)"] = (
        seq.get("turn1_disk_writes", 0) > 0)
    checks["SEQUENTIAL disk files present (disk_files>0)"] = (
        seq.get("turn1_disk_files", 0) > 0)
    checks["SEQUENTIAL follow-up reuses prefix"] = (
        seq.get("turn2_matched_delta", 0) > 0 or seq.get("turn2_hit_delta", 0) > 0)
    checks["SEQUENTIAL shards persist after stop"] = seq.get("shards_after_stop", 0) > 0
    checks["SEQUENTIAL survives restart (disk hit)"] = (
        seq.get("restart_disk_hit_delta", 0) > 0)
    # batching / multi-client populate (the store trigger)
    checks["cold request 200"] = pop.get("cold_status") == 200
    checks["continuous batching enabled"] = bool(pop.get("continuous_batching_enabled"))
    checks["all concurrent clients 200"] = bool(pop.get("batch_all_ok"))
    pb = pop.get("post_batch", {})
    checks["batch populates the cache (stores>0)"] = int(pb.get("stores", 0)) > 0
    checks["batch writes through to disk (disk_writes>0)"] = int(
        pb.get("disk_writes", 0)) > 0
    checks["disk files present (disk_files>0)"] = int(pb.get("disk_files", 0)) > 0
    checks["disk blocks indexed (>0)"] = int(pb.get("disk_blocks_indexed", 0)) > 0
    checks["shards on FS (live)"] = pop.get("shards_on_fs_live", 0) > 0
    # within-session prefix reuse, after the batch populated it
    checks["within-session prefix reuse (matched up)"] = (
        pop.get("warm_matched_delta", 0) > 0)
    checks["within-session hit registered"] = (
        pop.get("warm_hit_delta", 0) > 0 or pop.get("warm_disk_hit_delta", 0) > 0)
    # shards survive the process exit (on disk after the server stopped)
    checks["shards persist after server stop"] = pop.get("shards_on_fs_after_stop", 0) > 0
    # restart survival (the headline)
    checks["replay after restart 200"] = restart.get("replay_status") == 200
    checks["DISK HIT after restart (survived)"] = (
        restart.get("disk_hits_after_restart", 0) > 0)
    checks["shards re-indexed on startup"] = restart.get("disk_blocks_indexed", 0) > 0
    checks["restart restored matched tokens"] = (
        restart.get("matched_tokens_after_restart", 0) > 0)
    # reset keeps disk
    checks["reset keeps disk tier (hit after reset)"] = (
        restart.get("disk_hit_delta_after_reset", 0) > 0)
    # namespace isolation (skipped cleanly if no second model)
    if iso is not None:
        checks["other model: no cross-namespace disk hit"] = (
            iso.get("cold_disk_hits", 1) == 0)
        checks["other model writes its own namespace"] = bool(iso.get("new_namespace"))

    lines = ["# APC disk-tier e2e\n"]
    lines.append("## sequential single-user (no concurrency)")
    lines.append(f"- continuous batching: {seq.get('continuous_batching_enabled')}")
    lines.append(f"- lone turn 1: stores={seq.get('turn1_stores')} "
                 f"disk_writes={seq.get('turn1_disk_writes')} "
                 f"disk_files={seq.get('turn1_disk_files')}")
    lines.append(f"- lone turn 2 (shares turn-1 prefix): "
                 f"matched_tokens +{seq.get('turn2_matched_delta')} "
                 f"hits +{seq.get('turn2_hit_delta')}")
    lines.append(f"- shards on FS after stop: {seq.get('shards_after_stop')}")
    lines.append(f"- restart replay: disk_hits +{seq.get('restart_disk_hit_delta')} "
                 f"matched_tokens={seq.get('restart_matched')}\n")
    lines.append("## batching / multi-client populate")
    lines.append(f"- continuous batching: {pop.get('continuous_batching_enabled')}")
    lines.append(f"- cold (lone request): disk_writes="
                 f"{pop.get('cold',{}).get('disk_writes')} "
                 f"stores={pop.get('cold',{}).get('stores')} "
                 f"(lone requests populate too - see the sequential phase)")
    lines.append(f"- batch x{pop.get('concurrency')}: statuses={pop.get('batch_statuses')} "
                 f"wall={pop.get('batch_wall_s')}s vs serial-sum="
                 f"{pop.get('batch_serial_estimate_s')}s")
    lines.append(f"- after batch: stores={pb.get('stores')} disk_writes="
                 f"{pb.get('disk_writes')} disk_files={pb.get('disk_files')} "
                 f"disk_blocks_indexed={pb.get('disk_blocks_indexed')} "
                 f"(shards on FS={pop.get('shards_on_fs_live')})")
    lines.append(f"- within-session replay: lookups_hit +{pop.get('warm_hit_delta')} "
                 f"matched_tokens +{pop.get('warm_matched_delta')} "
                 f"disk_hits +{pop.get('warm_disk_hit_delta')}")
    lines.append(f"- shards on FS after server stop: "
                 f"{pop.get('shards_on_fs_after_stop')}\n")
    lines.append("## restart + reset")
    lines.append(f"- after restart: disk_hits={restart.get('disk_hits_after_restart')} "
                 f"disk_blocks_indexed={restart.get('disk_blocks_indexed')} "
                 f"matched_tokens={restart.get('matched_tokens_after_restart')}")
    lines.append(f"- disk hit delta after in-memory reset: "
                 f"{restart.get('disk_hit_delta_after_reset')}\n")
    if iso is not None:
        lines.append("## namespace isolation")
        lines.append(f"- other-model first-replay disk_hits={iso.get('cold_disk_hits')} "
                     f"(must be 0), own disk_writes={iso.get('disk_writes')}")
        lines.append(f"- namespaces: {iso.get('ns_before')} -> {iso.get('ns_after')}\n")
    else:
        lines.append("## namespace isolation\n- SKIPPED (no second model present)\n")

    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report)
    print("\n" + report)

    print("=" * 64)
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("=" * 64)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-root", default=ModelRegistry.root,
                    help="root for the base GGUFs (default ~/llm/gguf)")
    ap.add_argument("--model-a-handle", default="qwen3_0_6b_q8",
                    help="registry handle for the primary model (default qwen3_0_6b_q8)")
    ap.add_argument("--model-b-handle", default="qwen3_0_6b_q4",
                    help="registry handle for the isolation model (default qwen3_0_6b_q4)")
    ap.add_argument("--block-size", type=int, default=16,
                    help="APC_BLOCK_SIZE (default 16; smaller => more blocks per prompt)")
    ap.add_argument("--num-blocks", type=int, default=16,
                    help="APC_NUM_BLOCKS in-memory pool (default 16, small so the "
                         "batch overflows it and the SSD tier engages deterministically)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="concurrent clients for the batching phase (default 4)")
    ap.add_argument("--out", default=None, help="artifact dir (default: temp, removed)")
    ap.add_argument("--keep", action="store_true", help="keep artifacts + disk cache")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the serve subprocesses")
    a = ap.parse_args()

    reg = ModelRegistry(root=a.models_root)
    model_a = reg.find(a.model_a_handle)
    if model_a is None:
        print(f"SKIP: model handle {a.model_a_handle!r} not found under {a.models_root}.")
        reg.print_bootstrap([a.model_a_handle])
        return 0
    model_b = reg.find(a.model_b_handle)
    if model_b is None or os.path.abspath(model_b) == os.path.abspath(model_a):
        print(f"NOTE: isolation model {a.model_b_handle!r} unavailable/duplicate - "
              f"skipping the namespace-isolation phase.")
        model_b = None

    import tempfile
    tmp = a.out or tempfile.mkdtemp(prefix="gguf-apc-e2e-")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    disk_root = os.path.join(tmp, "apc-disk")
    disk_root_seq = os.path.join(tmp, "apc-disk-seq")
    log_dir = os.path.join(tmp, "logs")
    Path(disk_root).mkdir(parents=True, exist_ok=True)
    Path(disk_root_seq).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(f"[apc-e2e] model A = {model_a}")
    print(f"[apc-e2e] model B = {model_b or '(none)'}")
    print(f"[apc-e2e] APC_DISK_PATH = {disk_root}")
    print(f"[apc-e2e] artifacts = {tmp}")

    rc = 1
    try:
        print("\n[phase] sequential single-user (no concurrency) ...")
        seq = phase_sequential(a.python, model_a, disk_root_seq, log_dir,
                               a.block_size, a.num_blocks)
        print("[phase] populate + warm + batching (server A1) ...")
        pop = phase_populate_warm_batch(a.python, model_a, disk_root, log_dir,
                                        a.block_size, a.num_blocks, a.concurrency)
        print("[phase] restart + reset (server A2, fresh process) ...")
        restart = phase_restart_and_reset(a.python, model_a, disk_root, log_dir,
                                          a.block_size, a.num_blocks)
        iso = None
        if model_b is not None:
            print("[phase] namespace isolation (server B, other model) ...")
            iso = phase_isolation(a.python, model_b, disk_root, log_dir,
                                  a.block_size, a.num_blocks, a.concurrency)
        rc = grade(seq, pop, restart, iso, Path(tmp))
    finally:
        if not a.keep and not a.out:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"[apc-e2e] artifacts kept under {tmp}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
