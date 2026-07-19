#!/usr/bin/env python3
"""Competitive serve-path benchmark: gmlx vs stock llama.cpp (llama-server).

Brings up each OpenAI-compatible server one at a time (SEQUENTIAL: load A ->
measure -> unload -> load B -> measure -> unload), with matched, cache-disabled,
forced-length configuration, and reports prefill + decode-at-depth + light
concurrent throughput through the /v1 API. MTP-aware: each model is measured both
with and without speculative decoding where a drafter is supplied.

Fairness guards baked in:
  * SAME GGUF feeds both runtimes (gmlx is zero-conversion), so the only
    variable is the runtime.
  * Caches OFF both sides (gmlx single positional + APC disabled; llama
    --no-cache-prompt --cache-reuse 0) AND every request carries a unique leading
    nonce, so no prefix/KV cache can ever hit.
  * Forced-length decode both sides (gmlx serve --ignore-eos, llama-server
    --ignore-eos) so decode_tps is measured over an identical, fixed token window.
  * Equalized sampling: the IDENTICAL sampling body is sent to both runtimes
    (temperature/top_p[/top_k/min_p/repetition_penalty/seed]); neither relies on
    its own default sampler. DEFAULT = Qwen3.x recommended thinking-mode sampling
    (temp 0.6, top_p 0.95, top_k 20, min_p 0) -- the realistic deploy regime, NOT
    greedy (temp 0 distorts MTP acceptance dynamics). Override via --config.
  * Prompts matched to the target's tuning, because MTP acceptance is corpus-
    dependent. INSTRUCT/chat targets: --chat-dataset (real multi-turn chat sent as
    a messages array => server applies the chat template => draft head on-distribution;
    chained to any depth). BASE/continuation targets: --corpus (raw text sliced to
    depth). Neither => an embedded tiled passage (unrepresentative; MTP runs only).
    The corpus used is stamped into the report (md header + JSON meta).
  * Thermal fairness: alternate which runtime runs first each round, cooldown
    between server blocks, and report the MEDIAN across rounds (alternation makes
    that order-unbiased). "Both throttled" stays fair because the throttle is
    shared symmetrically. pmset -g therm is snapshotted into the JSON per round.
  * Token counts re-tokenized with each model's own tokenizer (never the server's
    self-reported usage).

Run in a venv with gmlx installed (`pip install gmlx`; --chat-dataset also
needs `datasets`):
  ./serve-bench.py --config example.json
Always --dry-run first (prints every server command line, launches nothing).

--vs ds4 swaps the comparison arm to dwarfstar's ds4-server (DeepSeek V4
GGUFs; same OpenAI chat API, all metrics client-side). ds4 notes: forced
length needs the bundled ignore_eos patch (patches/ds4-ignore-eos.patch); config
"thinking": false equalizes sampling (ds4 overrides sampling while thinking);
MTP arms (mtp.ds4 {gguf, margin?}) speculate only under greedy sampling, so
run them separately with temperature 0; one Metal worker serializes requests
(c>1 = queueing, not batching).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests

GMLX_PORT = 8080
LLAMA_PORT = 8081
DS4_PORT = 8082
RUNTIMES = ("gmlx", "llama")   # default pair; --vs ds4 swaps the comparison arm

# Embedded fallback corpus (generic prose). Point --corpus at a real text file
# (e.g. a fineweb-edu sample) for representative MTP acceptance; this fallback
# tiles, which makes acceptance optimistic.
_EMBEDDED_CORPUS = (
    "The study of language begins with the observation that meaning is carried "
    "not by single words alone but by the patterns in which they are arranged. "
    "A reader encountering an unfamiliar passage relies on context, on the shape "
    "of sentences, and on expectations built from everything read before. "
    "Machines that model language do something similar: they predict the next "
    "token from the tokens that came before, and the quality of that prediction "
    "depends on how much of the surrounding structure the model can hold in view. "
    "Longer context lets a model track an argument across paragraphs, recall a "
    "name introduced pages earlier, and keep a consistent tone. Shorter context "
    "forces it to lean on local cues, which works for ordinary prose but fails "
    "when the thread of meaning stretches far. Researchers measure these effects "
    "with careful experiments, holding constant everything but the one variable "
    "under test, because a fair comparison is the only kind that teaches anything. "
    "When two systems are compared, the conditions must match: the same inputs, "
    "the same limits, the same way of counting what each produced. Otherwise the "
    "numbers describe the setup, not the systems, and the conclusion dissolves "
    "under scrutiny. Good measurement is patient, repeated, and skeptical of its "
    "own first results."
).split()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    gguf: str                      # GGUF path; SAME file feeds both runtimes
    tokenizer: str                 # HF id or local path; scores this model
    # MTP plumbing per runtime; absent -> baseline arm only.
    mtp_gmlx: dict = field(default_factory=dict)   # {} | {speculative:true} | {draft_gguf:path}
    mtp_llama: dict = field(default_factory=dict)      # {} | {draft_gguf} | {spec_type, draft_min_p}
    mtp_ds4: dict = field(default_factory=dict)        # {} | {gguf:path, margin?}
    kv_bits: int = None                                # quantized KV, symmetric: gmlx KV_BITS + llama -ctk/-ctv

    def arms(self, draft_depths) -> list:
        """List of (label, draft_n). Baseline is (label, None); each MTP draft
        depth N gets its own arm so the speculation depth is swept and equalized
        with llama's --spec-draft-n-max."""
        out = [("baseline", None)]
        if self.mtp_gmlx or self.mtp_llama or self.mtp_ds4:
            out += [(f"mtp@{n}", n) for n in draft_depths]
        return out


def arms_for(spec, cfg):
    out = spec.arms(cfg["draft_depths"])
    if cfg.get("mtp_only"):
        out = [(lbl, n) for lbl, n in out if n is not None]   # drop baseline arm
    return out


def runtime_ctx(runtime, max_conc, cfg):
    """Server -c context allocation. Headroom over the target depth: prompts are
    sized by OUR tokenizer, but the server re-tokenizes with the GGUF's tokenizer
    and can count ~10-15% more on the same text (plus the chat-template wrapper).
    Too tight a -c makes the server reject the request ("exceeds context size")
    before prefill. 1.35x + slack absorbs the mismatch. llama-server: -c is total
    across -np slots, so multiply by max_conc. ds4-server: one Metal worker
    serializes requests through a single session, so -c is per-sequence only.
    gmlx sizes KV dynamically and ignores the value."""
    per_seq = int(max(cfg["depths"]) * 1.35) + cfg["max_tokens"] + 1024
    # llama_ctx_cap clamps the per-sequence -c (e.g. to n_ctx_train) so a very
    # deep target does not push -c past the trained context and silently trigger
    # RoPE scaling. The prompt still fits when cap >= depth + max_tokens.
    cap = cfg.get("llama_ctx_cap")
    if cap:
        per_seq = min(per_seq, int(cap))
    return per_seq if runtime == "ds4" else max_conc * per_seq


def _expand(p):
    return os.path.abspath(os.path.expanduser(p)) if p else p


def load_specs(args) -> tuple:
    """Returns (list[ModelSpec], run-config dict)."""
    cfg = {
        "depths": [0, 1024, 4096, 8192],
        "concurrency": [1, 2, 3],
        "draft_depths": [2, 3],   # block size 1 dropped: weakest MTP point + mlx-vlm bs<=1 early-stop bug
        "max_tokens": 192,
        "rounds": 2,
        "cooldown": 15.0,
        # Optional active thermal gate: after the fixed cooldown, poll die temp
        # and keep waiting (up to cool_max_wait s) until the max SoC die sensor
        # is <= cool_to_c. None => fixed cooldown only (legacy behavior). This is
        # how "GPU <= 50C before every arm launch" is enforced across a long run.
        "cool_to_c": None,
        "cool_max_wait": 180.0,
        # Clamp llama-server -c (None = use the 1.35x-headroom formula). Set to a
        # model's n_ctx_train for very deep cells so -c never exceeds the trained
        # context (avoids silent RoPE scaling / non-representative deep numbers).
        "llama_ctx_cap": None,
        "warmup": 1,
        # DEFAULT = Qwen3.x recommended thinking-mode sampling (the realistic
        # deploy operating point). Greedy (temp 0) is NOT how these models are
        # served and it distorts MTP: speculative acceptance dynamics differ under
        # sampling (a sampling draft + the rejection-rule accept, like llama's
        # top_k draft), so benching greedy measures the wrong regime. Sent
        # IDENTICALLY to both runtimes (fairness). Override via --config sampling.
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "min_p": 0.0, "seed": 1234},
        # Thinking protocol. null (prod default) => model/template default --
        # thinking ON for hybrid-reasoning models. MTP acceptance is strongly
        # content-sensitive (think-mode text drafts much better than no-think
        # answers; see sweep-history 2026-07-16), so never compare MTP cells
        # across thinking modes. ds4 arms: ds4-server, while thinking, OVERRIDES
        # request sampling to DeepSeek's recommended values (temp 1.0 / top_p
        # 1.0 / top_k 0 / min_p 0.05); the PREFERRED ds4 baseline A/B is
        # thinking-on with sampling set to exactly those values on both arms
        # (gmlx honors them as sent; arms agree). false => thinking off both
        # sides (gmlx env + ds4 body key), sampling honored as sent -- the
        # sampling-equalized alternative, ds4 configs only. ds4 MTP arms
        # additionally require greedy sampling (server gate).
        "thinking": None,
    }
    specs = []
    if args.config:
        with open(args.config) as fh:
            raw = json.load(fh)
        # The chat corpus can live in the config so a sweep is self-contained
        # and reproducible; an explicit CLI flag still wins.
        if args.chat_dataset is None:
            args.chat_dataset = raw.get("chat_dataset")
        if args.chat_split is None:
            args.chat_split = raw.get("chat_split")
        if args.chat_max_convs is None:
            args.chat_max_convs = raw.get("chat_max_convs")
        for k in ("depths", "concurrency", "draft_depths", "max_tokens", "rounds",
                  "cooldown", "cool_to_c", "cool_max_wait", "llama_ctx_cap",
                  "warmup", "sampling", "thinking"):
            if k in raw:
                cfg[k] = raw[k]
        for m in raw["models"]:
            specs.append(ModelSpec(
                name=m["name"], gguf=_expand(m["gguf"]),
                tokenizer=m.get("tokenizer") or args.tokenizer,
                mtp_gmlx=(m.get("mtp", {}) or {}).get("gmlx", {}) or {},
                mtp_llama=(m.get("mtp", {}) or {}).get("llama", {}) or {},
                mtp_ds4=(m.get("mtp", {}) or {}).get("ds4", {}) or {},
                kv_bits=m.get("kv_bits")))
    else:
        for g in args.gguf:
            g = _expand(g)
            specs.append(ModelSpec(name=os.path.basename(g), gguf=g,
                                   tokenizer=args.tokenizer))
    if args.chat_split is None:
        args.chat_split = "train_sft"
    if args.chat_max_convs is None:
        args.chat_max_convs = 8000
    # CLI overrides
    if args.depths:
        cfg["depths"] = [int(x) for x in args.depths.split(",")]
    if args.concurrency:
        cfg["concurrency"] = [int(x) for x in args.concurrency.split(",")]
    if args.draft_depths:
        cfg["draft_depths"] = [int(x) for x in args.draft_depths.split(",")]
    if args.max_tokens is not None:
        cfg["max_tokens"] = args.max_tokens
    if args.rounds is not None:
        cfg["rounds"] = args.rounds
    if args.cooldown is not None:
        cfg["cooldown"] = args.cooldown
    if args.warmup is not None:
        cfg["warmup"] = args.warmup
    if args.mtp_only:
        cfg["mtp_only"] = True
    if cfg["thinking"] is False and any(len(arms_for(s, cfg)) > 1 for s in specs):
        print("[warn] thinking:false with MTP arm(s): acceptance is content-"
              "sensitive (no-think text drafts markedly worse than think-mode; "
              "prod default is thinking ON). Keep false only for ds4-arm "
              "fairness, and never compare MTP cells across the two modes.")
    if cfg["warmup"] == 0 and max(cfg["depths"]) >= 17000 and args.requests < 3:
        print("[warn] boost-clock hazard: warmup=0 with deep depths and "
              "requests<3. The first big prefill after a cool/idle window "
              "rides DVFS boost clocks (~+8% on gmlx; llama's slower deep "
              "prefill self-sustains). Booked standard = sustained clocks: "
              "set warmup>=1 (the max-depth warmup burns the boost window) "
              "or requests>=3 (median lands on sustained samples). "
              "requests=1 books PURE boost -- label it 'burst', never book "
              "or A/B it against sustained cells.")
    return specs, cfg


# ---------------------------------------------------------------------------
# prompt synthesis (realistic text + unique nonce; record the REAL token count)
# ---------------------------------------------------------------------------
def count_prompt_tokens(tok, prompt):
    """Token length of the content string via a raw encode. We deliberately do
    NOT use apply_chat_template here: some tokenizers (e.g. gemma-4 mlx dirs)
    return a degenerate constant length from apply_chat_template regardless of
    content, which would make depth-sizing loop forever. The server applies its
    own chat template at serve time; that wrapper is a small fixed offset that is
    symmetric across both runtimes, so raw content length is the right, robust
    basis for depth bucketing. NOT thread-safe; call under the lock."""
    return len(tok.encode(prompt))


def _build_tokenizer(spec):
    """Tokenizer used to size prompts to a token depth and to re-count output
    tokens (never the server's self-reported usage). tokenizer="gguf" (or a
    .gguf path) synthesizes the exact tokenizer gmlx serves from the GGUF
    metadata -- no gated HF download, identical to what the runtime tokenizes
    with. Otherwise tokenizer is an HF id or local dir loaded via AutoTokenizer."""
    t = spec.tokenizer
    if t and t.lower() != "gguf" and not t.endswith(".gguf"):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(t)
    src = spec.gguf if (not t or t.lower() == "gguf") else _expand(t)
    from gmlx.loader import load_gguf_wire_bytes
    from gmlx.tokenizer import load_tokenizer_from_gguf
    arrays, kquant_meta, arch_meta, meta, _shapes = load_gguf_wire_bytes(src, zero_copy=True)
    del arrays, kquant_meta            # release mmap views; only meta+arch needed
    return load_tokenizer_from_gguf(meta, arch_meta)


# ---------------------------------------------------------------------------
# chat-corpus prompts (INSTRUCT-target MTP: real multi-turn, chat-templated)
# ---------------------------------------------------------------------------
# Raw web text (make_prompt / --corpus) is the WRONG corpus class for instruct/
# chat targets: it is off-distribution for the tuned model AND carries no chat
# template, so it understates MTP draft-acceptance. For instruct targets the
# corpus must be real conversations sent as an OpenAI `messages` array so the
# server applies the model's chat template, keeping the draft head on-distribution
# at every depth. Multi-turn conversations also accumulate (chaining when one is
# short) to any prefill depth -- unlike single-turn benchmark prompts that cap
# at ~1-2k tokens.
class ChatCorpus:
    """Normalized multi-turn conversations. .convs is a list of conversations,
    each a list of {"role": "user"|"assistant", "content": str} starting with a
    user turn. .source is a provenance label for the report."""
    def __init__(self, convs, source):
        self.convs = convs
        self.source = source

    def __len__(self):
        return len(self.convs)


def _truncate_to_tokens(tok, text, target_tokens, lock):
    """Trim text to about target_tokens by dropping trailing words (approximate;
    exact token slicing is unnecessary for depth bucketing)."""
    words = text.split()
    if not words or target_tokens <= 0:
        return text
    with lock:
        tk = count_prompt_tokens(tok, text)
    if tk <= target_tokens:
        return text
    keep = max(4, int(len(words) * target_tokens / max(tk, 1)))
    return " ".join(words[:keep])


def _chat_turn_queue(chat, rng):
    """An endless alternating turn stream, chaining randomly-ordered conversations
    so any prefill depth is reachable with real chat turns. Yields (role, content)."""
    n = len(chat.convs)
    order = list(range(n))
    rng.shuffle(order)
    i = 0
    while True:
        for t in chat.convs[order[i % n]]:
            yield t["role"], t["content"]
        i += 1
        if i % n == 0:
            rng.shuffle(order)


def make_chat_prompt(tok, target_tokens, chat, rng, lock):
    """A unique multi-turn chat prompt of about target_tokens content tokens: real
    user/assistant turns accumulated (chaining conversations when one is too short)
    and ending on a user turn, so the target model generates the next assistant
    reply with the draft head on-distribution. A random leading nonce on the first
    user turn defeats prefix caches. Returns (messages, token_count); messages is
    an OpenAI chat list the server templates -- the template overhead is symmetric
    across runtimes, so raw content length is the depth basis (as in make_prompt)."""
    nonce = f"[ref-{rng.randrange(1 << 40):010x}] "
    gen = _chat_turn_queue(chat, rng)

    def next_user_content():
        for role, content in gen:
            if role == "user":
                return content
        return "Continue."

    # depth ~0: a single short real user turn.
    if target_tokens <= 8:
        content = nonce + next_user_content()
        with lock:
            n = count_prompt_tokens(tok, content)
        return [{"role": "user", "content": content}], n

    tol = max(16, target_tokens // 50)
    # Phase 1: collect strictly-alternating turns (user, assistant, ...) until a
    # user turn pushes us to the depth target, or a safety cap on turn count.
    raw, want, total = [], "user", 0
    for role, content in gen:
        if role != want:
            continue
        with lock:
            tk = count_prompt_tokens(tok, content)
        raw.append([role, content, tk])
        total += tk
        want = "assistant" if want == "user" else "user"
        if (total >= target_tokens and want == "assistant") or len(raw) >= 4096:
            break
    # Phase 2: must end on a user turn (the one the model answers).
    while raw and raw[-1][0] != "user":
        total -= raw[-1][2]
        raw.pop()
    if not raw:
        content = nonce + next_user_content()
        with lock:
            n = count_prompt_tokens(tok, content)
        return [{"role": "user", "content": content}], n
    # unique nonce on the first turn
    raw[0][1] = nonce + raw[0][1]
    with lock:
        new_first = count_prompt_tokens(tok, raw[0][1])
    total += new_first - raw[0][2]
    raw[0][2] = new_first
    # truncate the final user turn so the prompt lands near the depth target
    if total > target_tokens + tol:
        keep = max(8, raw[-1][2] - (total - (target_tokens + tol // 2)))
        raw[-1][1] = _truncate_to_tokens(tok, raw[-1][1], keep, lock)
        with lock:
            raw[-1][2] = count_prompt_tokens(tok, raw[-1][1])
    msgs = [{"role": r[0], "content": r[1]} for r in raw]
    return msgs, sum(r[2] for r in raw)


def load_chat_corpus(name, split, max_convs):
    """Load a multi-turn chat dataset into normalized ChatCorpus conversations.
    Accepts the OpenAI `messages` schema (HuggingFaceH4/ultrachat_200k, smoltalk,
    tulu-3, ...) and the ShareGPT `conversations` schema ({from, value}). Forces
    offline use of the local HF cache."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    ds = ds.select(range(min(len(ds), max(max_convs * 3, 2000))))
    role_map = {"human": "user", "user": "user", "gpt": "assistant",
                "assistant": "assistant", "chatgpt": "assistant", "bard": "assistant",
                "system": None, "tool": None}
    convs = []
    for row in ds:
        turns = row.get("messages")
        if not turns and row.get("conversations"):
            turns = [{"role": role_map.get((t.get("from") or "").lower()),
                      "content": t.get("value") or ""} for t in row["conversations"]]
        if not turns:
            continue
        norm = []
        for t in turns:
            role = role_map.get((t.get("role") or "").lower(), t.get("role"))
            content = (t.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                continue
            if norm and norm[-1]["role"] == role:
                norm[-1]["content"] += "\n\n" + content      # merge consecutive same-role
            else:
                norm.append({"role": role, "content": content})
        while norm and norm[0]["role"] != "user":
            norm.pop(0)
        if norm:
            convs.append(norm)
        if len(convs) >= max_convs:
            break
    if len(convs) < 4:
        sys.exit(f"[chat] {name}:{split} yielded {len(convs)} usable conversations (need >=4)")
    return ChatCorpus(convs, f"{name}:{split}")


def make_prompt(tok, target_tokens, corpus, rng, lock):
    """A unique prompt of about target_tokens tokens: a random leading nonce
    (defeats every prefix cache) + realistic text sliced from the corpus + a
    continuation cue. Returns (prompt, real_token_count) where the count lands in
    [target, target + ~2%]. Converging on the TOKEN target (not the word count)
    matters: corpora run ~1.0-1.3 tokens/word, so seeding nwords=target would
    overshoot the depth by that ratio (e.g. +13%) and quietly prefill more than
    the labelled depth on both runtimes. A ChatCorpus routes to make_chat_prompt
    (multi-turn messages) -- the right corpus class for instruct targets."""
    if isinstance(corpus, ChatCorpus):
        return make_chat_prompt(tok, target_tokens, corpus, rng, lock)
    nonce = f"[ref-{rng.randrange(1 << 40):010x}]"
    cue = "\n\nContinue the passage at length:"
    if target_tokens <= 8:
        prompt = nonce + cue
        with lock:
            return prompt, count_prompt_tokens(tok, prompt)
    off = rng.randrange(len(corpus))

    def build(nwords):
        parts = [nonce] + [corpus[(off + i) % len(corpus)] for i in range(nwords)]
        p = " ".join(parts) + cue
        with lock:
            return p, count_prompt_tokens(tok, p)

    # Measure tokens/word from the build itself and proportionally correct.
    # Bounded iterations so a misbehaving tokenizer can never cause a spin.
    tol = max(16, target_tokens // 50)
    nwords = target_tokens
    prompt, n = build(nwords)
    for _ in range(8):
        if target_tokens <= n <= target_tokens + tol:
            break
        ratio = n / max(nwords, 1)               # measured tokens per word
        nwords = max(8, round((target_tokens + tol // 2) / ratio))
        prompt, n = build(nwords)
    return prompt, n


# ---------------------------------------------------------------------------
# one streamed request (lifted from bench-openai-serving.py)
# ---------------------------------------------------------------------------
def build_body(model_id, prompt, max_tokens, sampling, stream, extra=None):
    # /v1/chat/completions on both runtimes: the mlx-vlm server exposes only the
    # chat endpoint (no legacy /v1/completions), and chat is the representative
    # serve path. `prompt` is a single content string (one user turn) OR a prebuilt
    # OpenAI `messages` list (multi-turn chat-corpus mode). Same messages + same
    # model => same chat template both sides, so any template overhead is symmetric.
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": sampling.get("temperature", 0.6),
        "top_p": sampling.get("top_p", 0.95),
    }
    # Equalized extras: sent IDENTICALLY to both runtimes. ignore_eos is NOT here
    # -- it is a server launch flag on both sides (forced-length), so no body field
    # can be silently honored by one runtime and dropped by the other.
    for k in ("top_k", "min_p", "repetition_penalty", "seed"):
        v = sampling.get(k)
        if v is not None:
            body[k] = v
    # Per-runtime body extras (ds4 arm only): its ignore-EOS lever is a request
    # key (bundled patch; the other runtimes take a launch flag), and thinking-off
    # must be requested per-call ("thinking": false) where gmlx uses env.
    # Same MEANING both sides, different spelling -- fairness is preserved.
    if extra:
        body.update(extra)
    return body


def run_one(base_url, model_id, prompt, max_tokens, sampling, timeout, extra=None):
    url = base_url.rstrip("/") + "/chat/completions"
    body = build_body(model_id, prompt, max_tokens, sampling, stream=True, extra=extra)
    text_parts, chunk_times = [], []
    t_first = None
    t0 = time.perf_counter()
    try:
        with requests.post(url, json=body, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                try:
                    d = obj["choices"][0].get("delta") or {}
                    # Count reasoning_content too: gemma-4 (and other thinking
                    # templates) stream generated tokens under reasoning_content
                    # with content=null in llama-server's OpenAI layer. For a
                    # forced-length throughput benchmark a thinking token is one
                    # forward pass like any other, so it counts as decode work.
                    delta = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                except (KeyError, IndexError, TypeError):
                    delta = None
                if delta:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    chunk_times.append(now)
                    text_parts.append(delta)
    except Exception as e:  # noqa: BLE001 - network/HTTP failures are reported
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "t0": t0}
    t_end = time.perf_counter()
    if t_first is None:
        return {"ok": False, "error": "no content tokens streamed", "t0": t0}
    return {"ok": True, "t0": t0, "t_first": t_first, "t_end": t_end,
            "text": "".join(text_parts)}


def llama_timings(base_url, model_id, prompt, max_tokens, sampling, timeout):
    """One non-streaming request to harvest llama-server's `timings` (prompt/
    predicted per_second, and draft_n / draft_n_accepted for MTP acceptance).
    gmlx exposes none, so this is a llama-only cross-check annotation."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = build_body(model_id, prompt, max_tokens, sampling, stream=False)
    try:
        r = requests.post(url, json=body, timeout=timeout)
        r.raise_for_status()
        return (r.json() or {}).get("timings")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def score_sample(s, prompt_tokens, tok, lock):
    with lock:
        out_tok = len(tok.encode(s["text"])) if s["text"] else 0
    ttft = s["t_first"] - s["t0"]
    decode_time = s["t_end"] - s["t_first"]
    return {
        "ok": True,
        "prompt_tokens": prompt_tokens,
        "output_tokens": out_tok,
        "ttft_ms": ttft * 1e3,
        "prefill_tps": prompt_tokens / ttft if ttft > 0 else None,
        # decode rate excludes the first token (it lands at TTFT):
        "decode_tps": (out_tok - 1) / decode_time if out_tok > 1 and decode_time > 0 else None,
        "tpot_ms": (decode_time / (out_tok - 1)) * 1e3 if out_tok > 1 and decode_time > 0 else None,
        "t0": s["t0"], "t_end": s["t_end"],
    }


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def run_batch(base_url, model_id, prompts, max_tokens, sampling, concurrency,
              timeout, tok, lock, extra=None):
    """Fire `concurrency` requests at once; return (scored, agg_throughput)."""
    scored, raw = [], []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(run_one, base_url, model_id, p, max_tokens, sampling,
                          timeout, extra): pt for (p, pt) in prompts}
        for fut in as_completed(futs):
            pt = futs[fut]
            s = fut.result()
            if s.get("ok"):
                m = score_sample(s, pt, tok, lock)
                scored.append(m)
                raw.append((s["t0"], s["t_end"], m["output_tokens"]))
            else:
                scored.append(s)
    agg = None
    if len(raw) > 1:
        wall = max(e for _, e, _ in raw) - min(t for t, _, _ in raw)
        total_out = sum(o for _, _, o in raw)
        agg = total_out / wall if wall > 0 else None
    return scored, agg


# ---------------------------------------------------------------------------
# server lifecycle
# ---------------------------------------------------------------------------
def _gmlx_baseline_config(spec, outdir):
    """Baseline must serve with MTP OFF. gmlx auto-enables MTP when the GGUF
    carries a native nextn head (qwen3.5/3.6), and single-positional serve has no
    --no-mtp flag -- so the documented off-switch is a config model with
    `speculative: false` (resolve_speculative honors it as an explicit opt-out).
    Single-positional is itself wrapped as a one-model config, so this is the same
    infrastructure minus the auto-enable."""
    os.makedirs(outdir, exist_ok=True)
    cfg = os.path.join(outdir, f"_baseline-{spec.name}.yaml")
    with open(cfg, "w") as f:
        f.write(f"server:\n  host: 127.0.0.1\n  port: {GMLX_PORT}\n"
                f"models:\n  {spec.name}:\n    path: {spec.gguf}\n"
                f"    speculative: false\n")
    return cfg


def gmlx_cmd(args, spec, draft_n, ctx=None):
    if draft_n is None:
        # baseline: force MTP OFF (native-head GGUFs auto-enable it) via config.
        cfg = _gmlx_baseline_config(spec, args.outdir)
        return [args.gmlx_bin, "serve", "--config", cfg,
                "--ignore-eos", "-f", "--no-menubar"]
    cmd = [args.gmlx_bin, "serve", spec.gguf,
           "--host", "127.0.0.1", "--port", str(GMLX_PORT),
           "--ignore-eos", "-f", "--no-menubar"]
    m = spec.mtp_gmlx
    if not m:
        return None        # no gmlx-side MTP config -> skip this cell
    if m.get("draft_gguf"):
        cmd += ["--draft-gguf", _expand(m["draft_gguf"])]   # two-GGUF (gemma)
    else:
        cmd += ["--speculative"]                            # native-head (nextn)
    # draft_n is the DRAFT-TOKEN count (== llama --spec-draft-n-max). gmlx's
    # --draft-block-size counts the anchor too (block = 1 anchor + drafts), so a
    # block of N drafts only N-1 tokens. Add 1 to equalize lookahead depth with
    # llama; feeding the same integer to both undershoots gmlx by one draft.
    cmd += ["--draft-block-size", str(draft_n + 1)]
    return cmd


def llama_cmd(args, spec, draft_n, ctx):
    cmd = [args.llama_server_bin, "-m", spec.gguf,
           "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
           "-ngl", "99", "-fa", "on", "-c", str(ctx),
           # Disable ALL prompt caching for parity with gmlx (no SSD/prompt
           # cache). --no-cache-prompt + --cache-reuse 0 are not enough on recent
           # llama.cpp: the in-RAM prompt cache stays on until --cache-ram 0.
           "--no-cache-prompt", "--cache-reuse", "0", "--cache-ram", "0",
           "--ignore-eos", "-np", str(args._max_conc), "-cb"]
    if spec.kv_bits:
        # Symmetric with gmlx KV_BITS. -fa on (above) is required for a
        # quantized V cache. Map bit-width -> llama cache type.
        t = {8: "q8_0", 5: "q5_1", 4: "q4_1"}.get(spec.kv_bits)
        if t:
            cmd += ["-ctk", t, "-ctv", t]
    if draft_n is not None:
        m = spec.mtp_llama
        if not m:
            return None        # no llama-side MTP config -> skip this cell (reported)
        cmd += ["--spec-type", m.get("spec_type", "draft-mtp")]
        if m.get("draft_gguf"):
            cmd += ["--spec-draft-model", _expand(m["draft_gguf"])]   # two-GGUF (gemma)
        elif m.get("spec_default", True):
            # native-head MTP. NB: --spec-default ALSO registers an n-gram speculator
            # (ngram-mod, up to 64 free tokens) at HIGHER priority than the MTP head,
            # so llama's measured speedup blends two speculators. Set
            # mtp.llama.spec_default=false to isolate the native nextn head: draft-mtp
            # stays enabled via --spec-type + the auto-discovered .mtp sibling.
            cmd += ["--spec-default"]                                 # native head (+ngram)
        cmd += ["--spec-draft-n-max", str(draft_n)]
        if m.get("draft_min_p") is not None:
            # llama-server spells this --draft-p-min (alias --spec-draft-p-min);
            # --draft-min-p does not exist and aborts startup.
            cmd += ["--draft-p-min", str(m["draft_min_p"])]
    return cmd


def ds4_cmd(args, spec, draft_n, ctx):
    """ds4-server (dwarfstar) comparison arm. Cache fairness: the disk KV cache
    only exists behind --kv-disk-dir (not passed) and the in-RAM live-KV prefix
    match cannot hit across our unique-nonce prompts. Forced length comes from
    the ignore_eos request key (bundled patch patches/ds4-ignore-eos.patch -- stock
    ds4 stops at EOS and would break the fixed decode window). --warm-weights
    pre-touches weights at load; warmup requests are excluded from stats on all
    runtimes anyway. kv_bits has no ds4 mapping (its MLA latent cache layout is
    fixed) and is ignored on this arm."""
    cmd = [args.ds4_bin, "-m", spec.gguf, "--metal",
           "--host", "127.0.0.1", "--port", str(DS4_PORT),
           "-c", str(ctx), "--warm-weights"]
    if draft_n is not None:
        m = spec.mtp_ds4
        if not m or not m.get("gguf"):
            return None        # no ds4-side MTP config -> skip this cell (reported)
        if draft_n < 2:
            return None        # ds4 spec gate is mtp_draft_tokens > 1; N=1 never speculates
        cmd += ["--mtp", _expand(m["gguf"]), "--mtp-draft", str(draft_n)]
        if m.get("margin") is not None:
            # Fast-accept margin heuristic (default 3.0). Not strict rejection
            # sampling; DS4_MTP_STRICT=1 env forces exact verify.
            cmd += ["--mtp-margin", str(m["margin"])]
    return cmd


# Runtime -> server command builder. A builder returns None to skip a cell
# (e.g. no MTP config for that runtime); the skip is recorded in the report.
BUILDERS = {"gmlx": gmlx_cmd, "llama": llama_cmd, "ds4": ds4_cmd}


def server_env(runtime):
    env = dict(os.environ)
    if runtime == "gmlx":
        env["GMLX_IGNORE_EOS"] = "1"   # belt + suspenders alongside the flag
        env["APC_ENABLED"] = "false"       # no prompt/disk cache
        env["GMLX_SPEC_APC"] = "0"     # no spec-path prefix cache either (its
                                           # stores pin KV+hidden across requests)
        env["PYTHONUNBUFFERED"] = "1"      # flush [req] log lines live (probe parses them)
        env["MLX_VLM_ENABLE_THINKING"] = "1"
    return env


def _parse_round_log(path):
    """Parse GMLX_ROUND_LOG and return (draft_n, draft_n_accepted)."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None, None
    total_drafted = 0
    total_accepted = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        try:
            accepted = int(parts[-2])
            bs = int(parts[-1])
            total_drafted += bs - 1
            total_accepted += accepted
        except (ValueError, IndexError):
            continue
    if total_drafted == 0:
        return None, None
    return total_drafted, total_accepted


_DS4_SPEC_KV = re.compile(r"drafted=(\d+).*?accepted=(\d+)")


def _parse_ds4_spec_log(path):
    """Parse ds4-server stderr (DS4_MTP_SPEC_LOG=1) and return
    (draft_n, draft_n_accepted). Lines: 'ds4: mtp spec seq accept drafted=D
    accepted=A' / '... seq partial drafted=D verified=V accepted=A' / '... seq
    miss at=I ... drafted=D accepted=A' / '... miss first draft=T'. accepted=
    counts the anchor token too, so accepted drafts = A - 1; a 'miss first'
    cycle proposed 1 draft and accepted 0."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None, None
    total_drafted = 0
    total_accepted = 0
    for line in lines:
        if "ds4: mtp spec" not in line:
            continue
        if "miss first" in line:
            total_drafted += 1
            continue
        m = _DS4_SPEC_KV.search(line)
        if m:
            total_drafted += int(m.group(1))
            total_accepted += max(int(m.group(2)) - 1, 0)
    if total_drafted == 0:
        return None, None
    return total_drafted, total_accepted


def _port(runtime):
    return {"gmlx": GMLX_PORT, "llama": LLAMA_PORT, "ds4": DS4_PORT}[runtime]


def base_url(runtime):
    return f"http://127.0.0.1:{_port(runtime)}/v1"


def wait_ready(runtime, proc, timeout):
    """Poll /v1/models until 200; return the first model id (the request `model`
    field). Fails fast if the process dies during load."""
    url = base_url(runtime).rstrip("/") + "/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{runtime} server exited (code {proc.returncode}) "
                               f"before becoming ready")
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                data = (r.json() or {}).get("data") or []
                return data[0]["id"] if data else "model"
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"{runtime} server not ready within {timeout}s")


def _port_free(port):
    """True if nothing is already LISTENing on 127.0.0.1:port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", int(port))) != 0


def _assert_model_match(runtime, model_id, spec):
    """Guard against a stale server squatting the target port. The served model
    MUST be the one we launched: llama reports the gguf path as its id, gmlx
    reports the config model name. Anything matching neither means we connected
    to someone else's server (e.g. a leftover from a prior run) -> abort loudly
    rather than silently benchmark the wrong model."""
    mid = str(model_id)
    base = os.path.basename(spec.gguf)
    stem = os.path.splitext(base)[0]
    # Normalize separators before comparing: a positionally-served GGUF reports
    # an id derived from general.name with separators folded to '-' (e.g.
    # 'Qwen_Qwen3.6-27B' -> 'qwen-qwen3.6-27b'), while the config name / file
    # stem keep '_' -- a raw prefix test then spuriously fails at the '_' vs '-'.
    # gmlx serve (config-UX profiles) also reports a family-profile id that
    # is a prefix of the full name (e.g. 'qwen3.6-27b' for 'Qwen3.6-27B-...-Q6_K').
    # spawn() already asserts the port was free, so a separator-agnostic prefix
    # match after our own launch is still our server.
    def _n(s):
        return re.sub(r"[^a-z0-9.]+", "-", str(s).lower()).strip("-")
    # mid may be a bare id (gmlx config name, or a general.name-derived id
    # for a positionally-served GGUF) OR a full gguf path (llama reports the
    # path). Compare the id, its basename, and its stem against the config name
    # / file base / stem, all separator-agnostic. A general.name id can also be
    # a prefix of the file stem (family profile, quant suffix dropped).
    mids = {_n(mid), _n(os.path.basename(mid)),
            _n(os.path.splitext(os.path.basename(mid))[0])}
    tgts = {_n(spec.name), _n(base), _n(stem)}
    ok = bool(mids & tgts) or any(
        len(m) >= 8 and any(t.startswith(m) or m.startswith(t) for t in tgts)
        for m in mids)
    # ds4-server reports a fixed alias id regardless of the loaded GGUF filename.
    if runtime == "ds4" and _n(mid).startswith("deepseek-v4"):
        ok = True
    if not ok:
        # spawn() already asserted the port was free before we launched, so this
        # is almost certainly our own server reporting an id shape the matcher
        # does not recognize (not a squatter). Warn loudly with the id so the
        # matcher can be refined -- but do NOT abort and forfeit the arm.
        print(f"    [warn] {runtime} served id={mid!r} did not match "
              f"{spec.name!r} / {base!r}; port was verified free at launch, "
              f"proceeding (measuring our own server).")


def spawn(runtime, cmd, startup_timeout, log_path, extra_env=None):
    port = _port(runtime)
    if not _port_free(port):
        raise RuntimeError(
            f"{runtime} target port {port} is already in use -- a stale server "
            f"is squatting it (its /v1/models would be measured instead of the "
            f"one we launch). Free it or pass --{runtime}-port <free-port>.")
    log = open(log_path, "w")
    env = server_env(runtime)
    if extra_env:
        env.update(extra_env)
    # ds4-server resolves its metal/*.metal kernel sources relative to CWD, so
    # it must run from its own binary dir (else "metal backend unavailable").
    cwd = os.path.dirname(os.path.abspath(cmd[0])) if runtime == "ds4" else None
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env, cwd=cwd)
    try:
        model_id = wait_ready(runtime, proc, startup_timeout)
    except Exception:
        teardown(proc)
        log.close()
        raise
    return proc, model_id, log


def teardown(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


_DIE_BIN = None   # compiled gputemp path, cached per process ("" = tried, absent)


def _die_temp_bin():
    """Compile the no-sudo Apple Silicon die-temp reader (gputemp.c, sibling file)
    once and cache the binary. Returns the path, or None if clang/source absent."""
    global _DIE_BIN
    if _DIE_BIN is not None:
        return _DIE_BIN or None
    import shutil
    import tempfile
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gputemp.c")
    clang = shutil.which("clang")
    if not (clang and os.path.exists(src)):
        _DIE_BIN = ""
        return None
    out = os.path.join(tempfile.gettempdir(), "gputemp-bin")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        rc = subprocess.run([clang, "-O2", "-o", out, src, "-framework", "IOKit",
                             "-framework", "CoreFoundation"],
                            capture_output=True).returncode
        if rc != 0:
            _DIE_BIN = ""
            return None
    _DIE_BIN = out
    return out


def die_temp():
    """(mean, max) SoC die temp in Celsius from the PMU tdie sensors, or None."""
    b = _die_temp_bin()
    if not b:
        return None
    try:
        out = subprocess.run([b], capture_output=True, text=True, timeout=5).stdout
        parts = out.split()                      # "DIE 62.4 MAX 66.1 (n=42)"
        if len(parts) >= 4 and parts[0] == "DIE":
            return float(parts[1]), float(parts[3])
    except Exception:
        pass
    return None


def thermal_snapshot():
    """SoC die temperature (no sudo) as the thermal annotation. Apple Silicon
    exposes no no-sudo throttle counter (pmset CPU_Speed_Limit is Intel-only and
    absent here), so die temp is the usable steady-state / soak signal; falls back
    to the pmset line, then None. Annotation only, never a gate."""
    dt = die_temp()
    if dt is not None:
        return f"die={dt[0]:.1f}C max={dt[1]:.1f}C"
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                             text=True, timeout=5).stdout
        for line in out.splitlines():
            if "CPU_Speed_Limit" in line:
                return line.strip()
    except Exception:
        pass
    return None


# Hot-start cool-gate timeouts recorded this run: stamped into the MD header + JSON
# meta so a thermal lane shift is visible in the receipts, not just the server log.
_COOL_TIMEOUTS = []


def _cooldown(cfg):
    """Inter-server-block cooldown: fixed sleep, then (if cool_to_c is set) poll
    the SoC die temp until the max sensor is <= cool_to_c or cool_max_wait s
    elapse. Keeps every arm launching from a comparable thermal floor over a long
    run. No-op beyond the fixed sleep when cool_to_c is None or the sensor is
    unavailable."""
    time.sleep(cfg["cooldown"])
    target = cfg.get("cool_to_c")
    if not target:
        return
    cap = cfg.get("cool_max_wait", 180.0)
    waited = 0.0
    while waited < cap:
        dt = die_temp()
        if dt is None:                 # sensor unavailable -> don't block
            return
        if dt[1] <= target:
            return
        print(f"    [cool] die max {dt[1]:.1f}C > {target:.0f}C; "
              f"waiting ({waited:.0f}/{cap:.0f}s)")
        time.sleep(10.0)
        waited += 10.0
    dt = die_temp()
    if dt is not None and dt[1] > target:
        _COOL_TIMEOUTS.append({"die_c": round(dt[1], 1), "target_c": target,
                               "waited_s": cap})
        print(f"    [warn] cool gate timed out at {dt[1]:.1f}C (target {target:.0f}C, "
              f"waited {cap:.0f}s) -- HOT START, thermal lane may shift")
    elif dt is not None:
        print(f"    [cool] cap reached at die max {dt[1]:.1f}C (target {target:.0f}C)")


def diagnose_failure(outdir, name, runtime, arm_label):
    """Best-effort: tail the server log for a recognizable error line so a failed
    arm reports WHY (port clash, OOM, bad flag, ctx overflow) instead of a bare
    exception string. Returns None if nothing telling is found."""
    log_path = os.path.join(outdir, f"server-{name}-{runtime}-{arm_label}.log")
    try:
        with open(log_path) as fh:
            tail = fh.readlines()[-60:]
    except OSError:
        return None
    keys = ("error", "failed", "exception", "traceback", "out of memory",
            "exceeds", "abort", "assert", "cannot", "refused", "address already in use")
    for line in reversed(tail):
        if any(k in line.lower() for k in keys):
            return line.strip()[:300]
    return None


# ---------------------------------------------------------------------------
# one server block: bring up a (runtime, model, arm), sweep, tear down
# ---------------------------------------------------------------------------
def measure_block(args, cfg, spec, runtime, arm_label, draft_n, tok, lock, rng,
                  results, outdir, flush_cb=None, round_idx=0):
    if runtime == "ds4" and draft_n is not None and \
            (cfg["sampling"].get("temperature") or 0) > 0:
        # ds4 speculates only when temperature <= 0 (server-side gate). Running
        # this arm anyway would silently measure baseline decode labeled mtp@N.
        print(f"    [skip] ds4/{arm_label}: ds4 MTP requires greedy sampling "
              f"(set sampling temperature 0 for the MTP run)")
        results.setdefault((spec.name, runtime, arm_label), {})["_skipped"] = \
            "ds4 mtp requires greedy sampling (temp 0)"
        return
    ctx = runtime_ctx(runtime, args._max_conc, cfg)
    cmd = BUILDERS[runtime](args, spec, draft_n, ctx)
    if cmd is None:
        print(f"    [skip] {runtime}/{arm_label}: no MTP config for this runtime")
        results.setdefault((spec.name, runtime, arm_label), {})["_skipped"] = "no mtp config"
        return

    extra_env = None
    round_log_path = None
    if runtime == "gmlx" and draft_n is not None:
        round_log_path = os.path.join(outdir, f"round-log-{spec.name}-{arm_label}-r{round_idx}.tsv")
        extra_env = {"GMLX_ROUND_LOG": round_log_path}
    if runtime == "gmlx" and spec.kv_bits:
        # Quantized KV: mlx-vlm's server reads KV_BITS at model-build time
        # (serve has no --kv-bits flag; the config-file path sets this same env).
        extra_env = dict(extra_env or {})
        extra_env["KV_BITS"] = str(spec.kv_bits)
    if runtime == "gmlx":
        # Server-side "wait for next token" guard must not give up before the
        # client budget does (deep dense prefill can exceed mlx-vlm's default).
        extra_env = dict(extra_env or {})
        extra_env["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] = str(args.timeout)
    if runtime == "gmlx" and cfg.get("thinking") is False:
        # Thinking-off protocol: overrides server_env's default "1".
        extra_env = dict(extra_env or {})
        extra_env["MLX_VLM_ENABLE_THINKING"] = "0"
    if runtime == "ds4" and draft_n is not None:
        # Emit per-cycle draft/accept stderr lines; parsed after the sweep.
        extra_env = dict(extra_env or {})
        extra_env["DS4_MTP_SPEC_LOG"] = "1"

    # ds4 request-body extras: forced length (bundled ignore_eos patch) always;
    # thinking off per protocol (its env-free lever is the body key).
    body_extra = None
    if runtime == "ds4":
        body_extra = {"ignore_eos": True}
        if cfg.get("thinking") is False:
            body_extra["thinking"] = False

    log_path = os.path.join(outdir, f"server-{spec.name}-{runtime}-{arm_label}.log")
    print(f"    [up] {runtime}/{arm_label}: {' '.join(cmd[:6])} ...")
    t_load = time.perf_counter()
    proc, model_id, log = spawn(runtime, cmd, args.startup_timeout, log_path,
                                extra_env=extra_env)
    print(f"    [ready] {runtime}/{arm_label} model_id={model_id} "
          f"({time.perf_counter() - t_load:.1f}s)")
    try:
        _assert_model_match(runtime, model_id, spec)  # reject a stale-port squatter
        # cold-end equilibration: one unmeasured throwaway request per restart
        # ramps GPU DVFS off idle clock so the FIRST measured cell (esp. shallow
        # depths) is clock-equilibrated, not cold-start-depressed. Analog of the
        # cool-below-Nc rule; excluded from stats. Fixed small prompt so it stays
        # cheap regardless of the config's max depth (unlike the warmup below).
        if args.ramp_tokens > 0:
            rp = [make_prompt(tok, args.ramp_tokens, args._corpus, rng, lock)]
            t_ramp = time.perf_counter()
            run_batch(base_url(runtime), model_id, rp, min(cfg["max_tokens"], 48),
                      cfg["sampling"], 1, args.timeout, tok, lock, extra=body_extra)
            print(f"    [ramp] {args.ramp_tokens}-tok clock warmup "
                  f"({time.perf_counter() - t_ramp:.1f}s)")
        # warmup (triggers gmlx lazy load; excluded from stats)
        for _ in range(cfg["warmup"]):
            wp = [make_prompt(tok, max(cfg["depths"]), args._corpus, rng, lock)]
            run_batch(base_url(runtime), model_id, wp, cfg["max_tokens"],
                      cfg["sampling"], 1, args.timeout, tok, lock,
                      extra=body_extra)

        for depth in cfg["depths"]:
            for c in cfg["concurrency"]:
                # Deterministic per-cell RNG: same (round, spec, arm, depth, c)
                # yields the same prompts for BOTH runtimes, eliminating
                # prompt-selection bias in acceptance comparisons.
                cell_key = f"{args.seed}:{round_idx}:{spec.name}:{arm_label}:{depth}:{c}"
                cell_seed = int(hashlib.md5(cell_key.encode()).hexdigest()[:8], 16)
                cell_rng = random.Random(cell_seed)
                prompts = [make_prompt(tok, depth, args._corpus, cell_rng, lock)
                           for _ in range(c * args.requests)]
                scored, agg = run_batch(base_url(runtime), model_id, prompts,
                                        cfg["max_tokens"], cfg["sampling"], c,
                                        args.timeout, tok, lock, extra=body_extra)
                key = (spec.name, runtime, arm_label, depth, c)
                results.setdefault(key, {"samples": [], "agg": []})
                results[key]["samples"].extend(s for s in scored if s.get("ok"))
                if agg is not None:
                    results[key]["agg"].append(agg)
                ok = sum(1 for s in scored if s.get("ok"))
                dec = _med([s.get("decode_tps") for s in scored if s.get("ok")])
                pre = _med([s.get("prefill_tps") for s in scored if s.get("ok")])
                print(f"      d={depth:<6} c={c}  ok={ok}/{len(prompts)}  "
                      f"prefill={pre:.0f} dec={dec:.1f} tps"
                      if dec and pre else
                      f"      d={depth:<6} c={c}  ok={ok}/{len(prompts)}")
                if flush_cb:
                    flush_cb()           # persist after every cell (crash-safe)

        # llama-only MTP acceptance cross-check (deepest depth, c=1)
        if runtime == "llama" and draft_n is not None:
            p, _pt = make_prompt(tok, max(cfg["depths"]), args._corpus, rng, lock)
            tm = llama_timings(base_url(runtime), model_id, p, cfg["max_tokens"],
                               cfg["sampling"], args.timeout)
            if tm:
                results.setdefault((spec.name, runtime, arm_label), {})["timings"] = tm
                dn, da = tm.get("draft_n"), tm.get("draft_n_accepted")
                if dn:
                    print(f"      [llama timings] accept={da}/{dn} "
                          f"({100.0 * (da or 0) / dn:.0f}%)")

        # gmlx MTP acceptance from round log
        if round_log_path is not None:
            dn, da = _parse_round_log(round_log_path)
            if dn:
                acc = results.setdefault((spec.name, runtime, arm_label), {})
                acc.setdefault("round_log_accept", []).append({"draft_n": dn, "draft_n_accepted": da})
                print(f"      [gmlx round log] accept={da}/{dn} "
                      f"({100.0 * da / dn:.0f}%)")
                if flush_cb:
                    flush_cb()

        # ds4 MTP acceptance from the server's DS4_MTP_SPEC_LOG stderr lines
        if runtime == "ds4" and draft_n is not None:
            dn, da = _parse_ds4_spec_log(log_path)
            if dn:
                acc = results.setdefault((spec.name, runtime, arm_label), {})
                acc.setdefault("round_log_accept", []).append({"draft_n": dn, "draft_n_accepted": da})
                print(f"      [ds4 spec log] accept={da}/{dn} "
                      f"({100.0 * da / dn:.0f}%)")
                if flush_cb:
                    flush_cb()
    finally:
        teardown(proc)
        log.close()
        print(f"    [down] {runtime}/{arm_label}; cooldown {cfg['cooldown']}s"
              + (f" +cool<={cfg['cool_to_c']:.0f}C" if cfg.get("cool_to_c") else ""))
        _cooldown(cfg)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _cv(xs):
    """Coefficient of variation (%) across pooled samples -- the thermal/run-to-run
    spread. None if fewer than 2 samples."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = statistics.mean(xs)
    return 100.0 * statistics.pstdev(xs) / m if m else None


def cell(results, name, runtime, arm, depth, c):
    r = results.get((name, runtime, arm, depth, c))
    if not r or not r["samples"]:
        return None
    return {
        "n": len(r["samples"]),
        "prompt_tokens": _med([s["prompt_tokens"] for s in r["samples"]]),
        "output_tokens": _med([s["output_tokens"] for s in r["samples"]]),
        "prefill_tps": _med([s["prefill_tps"] for s in r["samples"]]),
        "decode_tps": _med([s["decode_tps"] for s in r["samples"]]),
        "decode_cv": _cv([s["decode_tps"] for s in r["samples"]]),
        "tpot_ms": _med([s["tpot_ms"] for s in r["samples"]]),
        "agg_tps": _med(r["agg"]) if r["agg"] else None,
    }


def f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def fdec(d):
    """decode_tps as 'median (+-CV%)' so thermal/run spread is visible."""
    v, cv = d.get("decode_tps"), d.get("decode_cv")
    if v is None:
        return "-"
    return f"{v:.1f}" + (f" (+-{cv:.0f}%)" if cv is not None else "")


def ratio(a, b):
    return f"{a / b:.2f}x" if (a and b) else "-"


def render_md(meta, specs, cfg, results):
    rt_a, rt_b = meta.get("runtime_pair") or list(RUNTIMES)
    L = [f"# Serve-path benchmark: {rt_a} vs {rt_b} - {meta['timestamp']}", ""]
    L += [
        "- SAME GGUF both runtimes; caches OFF; forced-length (--ignore-eos / "
        "ds4 ignore_eos patch); equalized sampling.",
        f"- sampling: `{json.dumps(cfg['sampling'])}` | thinking={cfg.get('thinking')} "
        f"| max_tokens={cfg['max_tokens']} "
        f"| requests/cell={meta['requests']} x {cfg['rounds']} rounds (alternating order) "
        f"| warmup={cfg['warmup']}",
        "- prefill_tps = prompt_tokens / TTFT ; decode_tps = (out-1)/(end-TTFT) ; "
        "agg_tps = sum(out)/batch_wall. Token counts re-tokenized per model.",
        "- decode_tps shown as median (+-CV%): the CV is the spread across all pooled "
        "samples (requests x rounds), i.e. run-to-run + thermal variance. High CV => "
        "treat the median with caution; raise rounds. Full per-sample data in the JSON.",
    ]
    _cts = meta.get("cool_gate_timeouts") or []
    if _cts:
        _hot = max(c["die_c"] for c in _cts)
        L.append(f"- **[warn] cool gate timed out {len(_cts)}x** (die up to {_hot:.1f}C "
                 f"vs target {_cts[0]['target_c']:.0f}C): one or more arms had a HOT "
                 f"START -- prefill/decode for the affected block(s) may be "
                 f"thermal-lane-shifted; treat those cells with caution.")
    if "llama" in (rt_a, rt_b):
        L.append(f"- llama-server: `{meta['llama_server_bin']}`")
    if "ds4" in (rt_a, rt_b):
        L.append(f"- ds4-server: `{meta.get('ds4_bin', '?')}` "
                 f"@ {meta.get('ds4_commit', '?')} (+ ignore_eos patch) | "
                 f"thinking={cfg.get('thinking')}")
        L.append("- ds4 caveats: one Metal worker serializes requests (c>1 measures "
                 "queueing, not batching); MTP arms run only under greedy sampling "
                 "(ds4 spec gate) with margin-based fast-accept (--mtp-margin, "
                 "default 3.0; DS4_MTP_STRICT=1 for exact verify).")
    L += [
        f"- corpus: `{meta.get('corpus', '?')}` (chat:* = real multi-turn instruct "
        "prompts, chat-templated => representative instruct MTP acceptance; corpus:* = "
        "raw text, for base/continuation targets; embedded-tiled = unrepresentative "
        "fallback)",
        "",
    ]
    for spec in specs:
        L.append(f"## {spec.name}")
        L.append(f"`{spec.gguf}`  tokenizer=`{spec.tokenizer}`")
        L.append("")
        # decode + prefill at c=1, per arm, with A/B (runtime pair)
        L.append("### single-stream (c=1): prefill_tps / decode_tps / TPOT ms")
        hdr = ["depth", "arm",
               f"{rt_a} prefill", f"{rt_b} prefill", "prefill A/B",
               f"{rt_a} decode", f"{rt_b} decode", "decode A/B",
               f"{rt_a} TPOT", f"{rt_b} TPOT"]
        L.append("| " + " | ".join(hdr) + " |")
        L.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for depth in cfg["depths"]:
            for arm_label, _n in arms_for(spec, cfg):
                gm = cell(results, spec.name, rt_a, arm_label, depth, 1)
                ll = cell(results, spec.name, rt_b, arm_label, depth, 1)
                if gm is None and ll is None:
                    continue
                gm = gm or {}
                ll = ll or {}
                L.append("| " + " | ".join([
                    str(depth), arm_label,
                    f(gm.get("prefill_tps"), 0), f(ll.get("prefill_tps"), 0),
                    ratio(gm.get("prefill_tps"), ll.get("prefill_tps")),
                    fdec(gm), fdec(ll),
                    ratio(gm.get("decode_tps"), ll.get("decode_tps")),
                    f(gm.get("tpot_ms")), f(ll.get("tpot_ms")),
                ]) + " |")
        L.append("")
        # MTP speedup per runtime, swept over draft depth N (decode, c=1)
        mtp_arms = [(lbl, n) for lbl, n in arms_for(spec, cfg) if n is not None]
        if mtp_arms:
            L.append("### MTP speedup (decode_tps at draft-depth N / baseline, c=1)")
            L.append(f"| depth | draft N | {rt_a} | {rt_b} |")
            L.append("|---|---|---|---|")
            for depth in cfg["depths"]:
                for lbl, n in mtp_arms:
                    row = [str(depth), str(n)]
                    for rt in (rt_a, rt_b):
                        b = cell(results, spec.name, rt, "baseline", depth, 1)
                        m = cell(results, spec.name, rt, lbl, depth, 1)
                        row.append(ratio(m.get("decode_tps") if m else None,
                                         b.get("decode_tps") if b else None))
                    L.append("| " + " | ".join(row) + " |")
            sk = {k: v for k, v in results.items()
                  if len(k) == 3 and k[0] == spec.name and v.get("_skipped")}
            for (_n2, rt, al), v in sk.items():
                L.append(f"\n_note: {rt}/{al} skipped ({v['_skipped']})._")
            fl = {k: v for k, v in results.items()
                  if len(k) == 3 and k[0] == spec.name and v.get("_failed")}
            for (_n4, rt, al), v in fl.items():
                L.append(f"\n_note: {rt}/{al} unavailable -- {v['_failed']}._")
            # draft acceptance, where harvested (both runtimes)
            for (_n3, rt, al), v in {k: v for k, v in results.items()
                                     if len(k) == 3 and k[0] == spec.name
                                     and v.get("timings")}.items():
                tm = v["timings"]
                dn, da = tm.get("draft_n"), tm.get("draft_n_accepted")
                if dn:
                    L.append(f"\n_{rt}/{al} draft acceptance: {da}/{dn} "
                             f"({100.0 * (da or 0) / dn:.0f}%)._")
            for (_n3, rt, al), v in {k: v for k, v in results.items()
                                     if len(k) == 3 and k[0] == spec.name
                                     and v.get("round_log_accept")}.items():
                entries = v["round_log_accept"]
                dn = sum(e["draft_n"] for e in entries)
                da = sum(e["draft_n_accepted"] for e in entries)
                if dn:
                    L.append(f"\n_{rt}/{al} draft acceptance: {da}/{dn} "
                             f"({100.0 * da / dn:.0f}%)._")
            L.append("")
        # concurrent aggregate throughput
        concs = [c for c in cfg["concurrency"] if c > 1]
        if concs:
            L.append("### concurrent aggregate throughput (agg_tps)")
            hdr = ["depth", "arm", "runtime"] + [f"c={c}" for c in concs]
            L.append("| " + " | ".join(hdr) + " |")
            L.append("|" + "|".join(["---"] * len(hdr)) + "|")
            for depth in cfg["depths"]:
                for arm_label, _n in arms_for(spec, cfg):
                    for rt in (rt_a, rt_b):
                        cells = [cell(results, spec.name, rt, arm_label, depth, c)
                                 for c in concs]
                        if all(c is None for c in cells):
                            continue
                        row = [str(depth), arm_label, rt]
                        row += [f((c or {}).get("agg_tps"), 0) for c in cells]
                        L.append("| " + " | ".join(row) + " |")
            L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def render_svgs(args, specs, runtimes, json_path):
    """Render the chart set the gmlx docs publish (per-model panels; fleet-ratio
    + mtp-lift when the run has the arms for them) via the sibling plot-bench.py,
    light + dark themes. Non-fatal: a chart without enough data is skipped."""
    pb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot-bench.py")
    if not os.path.exists(pb):
        print("[svg] plot-bench.py not found next to this script; skipping charts")
        return
    # sizes match the published docs/benchmarks.md chart set
    jobs = [("panels", [json_path, "--model", s.name,
                        "--width", "960", "--height", "720"],
             os.path.join(args.outdir, s.name + "-panels%s.svg")) for s in specs]
    if "gmlx" in runtimes and args.vs in runtimes:
        jobs.append(("fleet-ratio", [json_path, "--vs", args.vs,
                                     "--width", "960", "--height", "620"],
                     os.path.join(args.outdir, "fleet-ratio%s.svg")))
    if "gmlx" in runtimes and any(s.mtp_gmlx or s.mtp_llama or s.mtp_ds4
                                  for s in specs):
        extra = ["--with-ref"] if "llama" in runtimes else []
        jobs.append(("mtp-lift", [json_path, "--width", "960",
                                  "--height", "560"] + extra,
                     os.path.join(args.outdir, "mtp-lift%s.svg")))
    for chart, argv, outfmt in jobs:
        for theme, sfx in (("light", ""), ("dark", "-dark")):
            out = outfmt % sfx
            r = subprocess.run([sys.executable, pb, chart, *argv, "--drop-depth",
                                "0", "--theme", theme, "--out", out],
                               capture_output=True, text=True)
            if r.returncode == 0:
                print(f"wrote {out}")
            else:
                err = (r.stderr or r.stdout).strip().splitlines()
                print(f"[svg] {chart}{sfx}: skipped "
                      f"({err[-1] if err else 'render error'})")


def write_outputs(outdir, stamp, meta, cfg, specs, results):
    """Render + persist the md/json report. Called incrementally after every
    cell so a kill mid-run still leaves the latest complete-so-far report on
    disk (the timestamped pair is rewritten in place, not appended)."""
    md = render_md(meta, specs, cfg, results)
    md_path = os.path.join(outdir, f"serve-bench-{stamp}.md")
    json_path = os.path.join(outdir, f"serve-bench-{stamp}.json")
    tmp = json_path + ".tmp"
    dump = {"meta": meta, "config": cfg,
            "models": [{"name": s.name, "gguf": s.gguf, "tokenizer": s.tokenizer,
                        "arms": [a for a, _ in arms_for(s, cfg)]} for s in specs],
            "results": {"|".join(map(str, k)): v for k, v in results.items()}}
    with open(md_path, "w") as fh:
        fh.write(md)
    with open(tmp, "w") as fh:          # atomic-ish: full write then rename
        json.dump(dump, fh, indent=2, default=str)
    os.replace(tmp, json_path)
    return md_path, json_path, md


# ===========================================================================
# Step-0 cold concurrent-prefill probe
# ===========================================================================
# Goal: SIZE the concurrency prefill cliff and ATTRIBUTE it -- throughput ceiling
# (the GPU forward itself can't batch-prefill C ragged rows efficiently) vs TTFT
# starvation (the forward is fine but rows wait in the scheduler/queue). It does
# NOT decide whether to bother; it decides seam-patch (B/C admission policy) vs
# fusion (A unified prefill+decode forward). Method, per (runtime, depth, c):
#   * TRUE-simultaneous submit: C requests released from one barrier, so every
#     TTFT and the aggregate prefill window share one origin (a ThreadPoolExecutor
#     smears the start by thread-spawn latency; round 1's "admission lottery" noise
#     came from staggered arrival -- the barrier removes it).
#   * AGGREGATE prefill throughput = sum(prompt_tokens) / (last first-token - submit).
#     Single-stream prefill falls ~1/C with C by physics; aggregate is the honest
#     "how fast did the engine chew C*depth cold tokens" and its SCALING vs C (and
#     vs llama) is the cliff.
#   * Three attribution signals: (1) aggregate scaling efficiency, (2) per-stream
#     TTFT SKEW (max/min) -- uniform => ceiling, skewed => starvation, (3) the
#     SERVER-measured forward-region prefill rate (gmlx [req] line / llama
#     `prompt eval time`) vs the CLIENT TTFT -- if server stays fast while client
#     craters, the deficit is queue/scheduling not the GPU forward.
# Caches OFF (baseline arm forces MTP off too), unique-nonce prompts => no APC hit,
# so this isolates COLD prefill (APC reduces volume, never per-cold-token efficiency).

# regime -> decode tokens. prefill_only isolates the batched-prefill ceiling;
# with_decode reproduces round 1's simultaneous prefill+decode (its prefill cols
# should match prefill_only -- a consistency check -- and it yields aggregate
# decode scaling as a bonus). prefill_only uses a TINY 4-token decode budget (not
# 1): TTFT = prefill-completion is unaffected by the 4-token tail, but llama-server
# streams NO content chunk at max_tokens=1 (it hits the limit on the role chunk and
# sends finish with no content) -> the client records no first token. 4 makes every
# runtime stream a first token reliably. Selectable via cfg["_probe_regimes"].
def probe_regimes(cfg):
    allr = [("prefill_only", 4), ("with_decode", cfg["max_tokens"])]
    sel = cfg.get("_probe_regimes")
    return [r for r in allr if (not sel or r[0] in sel)]


# decode is logged as `-t/s` when gen<=1 (no decode rate to report -- the
# prefill_only regime), so the decode field must accept `-`, else those lines
# (whose prefill rate is exactly the server-forward signal we need) never match.
_GM_REQ = re.compile(
    r"prompt=(\d+)\s+gen=(\d+)\s+ttft=([\d.]+)s\s+prefill=(-|[\d.]+)t/s\s+decode=(-|[\d.]+)t/s")


def _opt_float(s):
    return None if s == "-" else float(s)


def parse_server_reqs(text, runtime):
    """Per-request SERVER-measured rates from freshly-appended server-log text.
    gmlx emits a single `[req] ... prefill=Xt/s decode=Yt/s` line per request;
    llama-server emits `prompt eval time = ... ( Z tokens per second)` (prefill)
    and a separate `eval time = ...` (decode). Returns list of dicts with
    prefill_tps and (where present) decode_tps."""
    if runtime == "gmlx":
        return [{"prompt": int(p), "gen": int(g), "ttft": float(tt),
                 "prefill_tps": _opt_float(pf), "decode_tps": _opt_float(dc)}
                for p, g, tt, pf, dc in _GM_REQ.findall(text)]
    out = []
    for line in text.splitlines():
        m = re.search(r"([\d.]+)\s+tokens per second", line)
        if not m:
            continue
        rate = float(m.group(1))
        low = line.lower()
        if "prompt eval time" in low:
            out.append({"prefill_tps": rate, "decode_tps": None})
        elif "eval time" in low:           # decode line; attach to last prefill
            if out and out[-1].get("decode_tps") is None:
                out[-1]["decode_tps"] = rate
            else:
                out.append({"prefill_tps": None, "decode_tps": rate})
    return out


def fire_simultaneous(base_url, model_id, prompts, max_tokens, sampling, timeout):
    """Release all C requests from one barrier. Returns (oks, t_submit): oks =
    [(run_one_dict, prompt_tokens), ...] for streams that produced tokens;
    t_submit = the shared release instant (earliest per-request t0)."""
    n = len(prompts)
    barrier = threading.Barrier(n)
    out = [None] * n

    def worker(i, p):
        try:
            barrier.wait(timeout=120)
        except threading.BrokenBarrierError:
            out[i] = {"ok": False, "error": "barrier broken"}
            return
        out[i] = run_one(base_url, model_id, p, max_tokens, sampling, timeout)

    threads = [threading.Thread(target=worker, args=(i, p), daemon=True)
               for i, (p, _pt) in enumerate(prompts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    oks = [(o, pt) for o, (_p, pt) in zip(out, prompts) if o and o.get("ok")]
    if not oks:
        return [], None
    return oks, min(o["t0"] for o, _ in oks)


def score_probe_cell(oks, t_submit, srv, dec, tok, lock):
    """One simultaneous batch -> aggregate prefill / TTFT-fairness / decode +
    server-side rates. TTFTs share the t_submit origin."""
    base = {"n": len(oks), "agg_prefill_tps": None, "per_stream_prefill_tps": None,
            "ttft_p50_ms": None, "ttft_min_ms": None, "ttft_max_ms": None,
            "ttft_skew": None, "agg_decode_tps": None,
            "srv_prefill_tps": None, "srv_decode_tps": None}
    if srv:        # server rates survive even if client streams all failed
        base["srv_prefill_tps"] = _med([s["prefill_tps"] for s in srv if s.get("prefill_tps")])
        base["srv_decode_tps"] = _med([s["decode_tps"] for s in srv if s.get("decode_tps")])
    if not oks or t_submit is None:
        return base
    recs = []
    for o, pt in oks:
        with lock:
            out_tok = len(tok.encode(o["text"])) if o.get("text") else 0
        recs.append({"pt": pt, "ttft": o["t_first"] - t_submit,
                     "t_first": o["t_first"], "t_end": o["t_end"], "out": out_tok})
    ttfts = [r["ttft"] for r in recs if r["ttft"] > 0]
    if ttfts:
        window = max(ttfts)                                # last row to finish prefill
        base["agg_prefill_tps"] = sum(r["pt"] for r in recs) / window if window > 0 else None
        base["per_stream_prefill_tps"] = _med([r["pt"] / r["ttft"] for r in recs if r["ttft"] > 0])
        base["ttft_p50_ms"] = statistics.median(ttfts) * 1e3
        base["ttft_min_ms"] = min(ttfts) * 1e3
        base["ttft_max_ms"] = max(ttfts) * 1e3
        base["ttft_skew"] = max(ttfts) / min(ttfts) if min(ttfts) > 0 else None
    if dec > 1:
        # aggregate decode across the batch: total decoded tokens (excl the first,
        # which lands at TTFT) over the wall from first-token to last-end.
        first = min(r["t_first"] for r in recs)
        last = max(r["t_end"] for r in recs)
        tot = sum(max(r["out"] - 1, 0) for r in recs)
        base["agg_decode_tps"] = tot / (last - first) if last > first else None
    return base


def measure_probe_block(args, cfg, spec, runtime, tok, lock, rng, results, outdir,
                        rnd, flush_cb=None):
    """One baseline server (MTP off) per (runtime, round); sweep depth x regime x c
    with simultaneous submit, capturing client + server-side rates."""
    ctx = runtime_ctx(runtime, args._max_conc, cfg)
    cmd = BUILDERS[runtime](args, spec, None, ctx)     # baseline => cold prefill only
    log_path = os.path.join(outdir, f"probe-server-{spec.name}-{runtime}-r{rnd + 1}.log")
    print(f"    [up] {runtime}: {' '.join(cmd[:6])} ...")
    t_load = time.perf_counter()
    extra_env = ({"MLX_VLM_TOKEN_QUEUE_TIMEOUT": str(args.timeout)}
                 if runtime == "gmlx" else None)
    proc, model_id, log = spawn(runtime, cmd, args.startup_timeout, log_path,
                                extra_env=extra_env)
    print(f"    [ready] {runtime} model_id={model_id} ({time.perf_counter() - t_load:.1f}s)")
    reader = open(log_path)                            # tails server [req] lines by offset
    try:
        wp = [make_prompt(tok, max(cfg["depths"]), args._corpus, rng, lock)]
        fire_simultaneous(base_url(runtime), model_id, wp, 8, cfg["sampling"], args.timeout)
        time.sleep(0.4)
        reader.read()                                 # discard warmup server lines
        for depth in cfg["depths"]:
            for regime, dec in probe_regimes(cfg):
                for c in cfg["concurrency"]:
                    reader.read()                     # advance to current EOF
                    prompts = [make_prompt(tok, depth, args._corpus, rng, lock)
                               for _ in range(c)]
                    oks, t_submit = fire_simultaneous(base_url(runtime), model_id,
                                                      prompts, dec, cfg["sampling"],
                                                      args.timeout)
                    time.sleep(0.4)                   # let server flush this cell's [req] lines
                    srv = parse_server_reqs(reader.read(), runtime)
                    m = score_probe_cell(oks, t_submit, srv, dec, tok, lock)
                    key = (spec.name, runtime, regime, depth, c)
                    results.setdefault(key, {"rounds": []})["rounds"].append(m)
                    print(f"      {regime:<12} d={depth:<6} c={c}  ok={m['n']}/{c}  "
                          f"agg_pre={f(m['agg_prefill_tps'], 0)} "
                          f"ttft_p50={f(m['ttft_p50_ms'], 0)}ms skew={f(m['ttft_skew'], 2)} "
                          f"srv_pre={f(m['srv_prefill_tps'], 0)}"
                          + (f" agg_dec={f(m['agg_decode_tps'], 1)}" if dec > 1 else ""))
                    if flush_cb:
                        flush_cb()
    finally:
        reader.close()
        teardown(proc)
        log.close()
        print(f"    [down] {runtime}; cooldown {cfg['cooldown']}s"
              + (f" +cool<={cfg['cool_to_c']:.0f}C" if cfg.get("cool_to_c") else ""))
        _cooldown(cfg)


def probe_cell(results, name, runtime, regime, depth, c):
    """Median across rounds for one probe cell."""
    r = results.get((name, runtime, regime, depth, c))
    if not r or not r["rounds"]:
        return None
    rs = r["rounds"]
    keys = ("agg_prefill_tps", "per_stream_prefill_tps", "ttft_p50_ms", "ttft_skew",
            "agg_decode_tps", "srv_prefill_tps", "srv_decode_tps")
    out = {k: _med([x.get(k) for x in rs]) for k in keys}
    out["rounds"] = len(rs)
    return out


def _eff(agg_c, agg_1, c):
    """Aggregate-prefill scaling efficiency: agg(c) / (c*agg(1)). 1.0 = perfect
    batch scaling; ~1/c = fully serialized (aggregate flat, no batching benefit)."""
    if not agg_c or not agg_1 or c < 1:
        return None
    return agg_c / (c * agg_1)


def render_probe_md(meta, specs, cfg, results):
    regimes = probe_regimes(cfg)
    L = [f"# Step-0 cold concurrent-prefill probe - {meta['timestamp']}", ""]
    L += [
        "- TRUE-simultaneous submit (one barrier); APC/caches OFF; unique-nonce prompts "
        "(no prefix hit); baseline arm (MTP off). SAME GGUF both runtimes.",
        f"- depths={cfg['depths']} concurrency={cfg['concurrency']} "
        f"rounds={cfg['rounds']} (alternating order) | with_decode max_tokens={cfg['max_tokens']}",
        "- agg_prefill = sum(prompt_tokens)/(last first-token - submit). "
        "eff = agg(c)/(c*agg(1)): 1.0=perfect batch scaling, ~1/c=serialized.",
        "- TTFT skew = max/min TTFT in the batch (uniform=>ceiling, skewed=>starvation). "
        "srv_prefill = server-measured forward-region rate (client-vs-server gap => queue/scheduling).",
        "",
    ]
    for spec in specs:
        L.append(f"## {spec.name}")
        L.append(f"`{spec.gguf}`  tokenizer=`{spec.tokenizer}`")
        L.append("")
        # 1) aggregate-prefill head-to-head (the cliff)
        L.append("### Aggregate cold-prefill throughput: gmlx vs llama (prefill_only)")
        hdr = ["depth", "c", "gm agg", "ll agg", "agg A/B",
               "gm eff", "ll eff", "gm srv_pre", "ll srv_pre",
               "gm ttft_skew", "ll ttft_skew"]
        L.append("| " + " | ".join(hdr) + " |")
        L.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for depth in cfg["depths"]:
            g1 = probe_cell(results, spec.name, "gmlx", "prefill_only", depth, cfg["concurrency"][0])
            l1 = probe_cell(results, spec.name, "llama", "prefill_only", depth, cfg["concurrency"][0])
            ga1 = (g1 or {}).get("agg_prefill_tps")
            la1 = (l1 or {}).get("agg_prefill_tps")
            for c in cfg["concurrency"]:
                gm = probe_cell(results, spec.name, "gmlx", "prefill_only", depth, c) or {}
                ll = probe_cell(results, spec.name, "llama", "prefill_only", depth, c) or {}
                if not gm and not ll:
                    continue
                L.append("| " + " | ".join([
                    str(depth), str(c),
                    f(gm.get("agg_prefill_tps"), 0), f(ll.get("agg_prefill_tps"), 0),
                    ratio(gm.get("agg_prefill_tps"), ll.get("agg_prefill_tps")),
                    f(_eff(gm.get("agg_prefill_tps"), ga1, c), 2),
                    f(_eff(ll.get("agg_prefill_tps"), la1, c), 2),
                    f(gm.get("srv_prefill_tps"), 0), f(ll.get("srv_prefill_tps"), 0),
                    f(gm.get("ttft_skew"), 2), f(ll.get("ttft_skew"), 2),
                ]) + " |")
        L.append("")
        # 2) attribution at the cliff bottom (deepest depth, max c)
        L.append("### Attribution (deepest depth, max concurrency)")
        depth = cfg["depths"][-1]
        c = cfg["concurrency"][-1]
        gm = probe_cell(results, spec.name, "gmlx", "prefill_only", depth, c) or {}
        ll = probe_cell(results, spec.name, "llama", "prefill_only", depth, c) or {}
        g1 = probe_cell(results, spec.name, "gmlx", "prefill_only", depth, cfg["concurrency"][0]) or {}
        ab_client = (gm.get("agg_prefill_tps") and ll.get("agg_prefill_tps")
                     and gm["agg_prefill_tps"] / ll["agg_prefill_tps"])
        ab_server = (gm.get("srv_prefill_tps") and ll.get("srv_prefill_tps")
                     and gm["srv_prefill_tps"] / ll["srv_prefill_tps"])
        skew = gm.get("ttft_skew")
        L.append(f"- at d={depth} c={c}: client agg A/B = {f(ab_client, 2)}, "
                 f"server-forward A/B = {f(ab_server, 2)}, gm TTFT skew = {f(skew, 2)}, "
                 f"gm eff = {f(_eff(gm.get('agg_prefill_tps'), g1.get('agg_prefill_tps'), c), 2)}")
        verdict = _probe_verdict(ab_client, ab_server, skew)
        L.append(f"- **heuristic verdict: {verdict}**")
        L.append("  - throughput ceiling (A/D, fusion) if client and server A/B both fall together "
                 "with low skew; queue/scheduling (B/C, seam) if server stays competitive while "
                 "client falls, or skew is high.")
        L.append("")
        # 3) aggregate decode under concurrency (bonus, with_decode)
        if "with_decode" in [r for r, _ in regimes] and any(r > 1 for r in cfg["concurrency"]):
            L.append("### Aggregate decode throughput under concurrency (with_decode)")
            hdr = ["depth", "c", "gm agg_dec", "ll agg_dec", "dec A/B"]
            L.append("| " + " | ".join(hdr) + " |")
            L.append("|" + "|".join(["---"] * len(hdr)) + "|")
            for depth in cfg["depths"]:
                for c in cfg["concurrency"]:
                    gm = probe_cell(results, spec.name, "gmlx", "with_decode", depth, c) or {}
                    ll = probe_cell(results, spec.name, "llama", "with_decode", depth, c) or {}
                    if not gm.get("agg_decode_tps") and not ll.get("agg_decode_tps"):
                        continue
                    L.append("| " + " | ".join([
                        str(depth), str(c),
                        f(gm.get("agg_decode_tps"), 1), f(ll.get("agg_decode_tps"), 1),
                        ratio(gm.get("agg_decode_tps"), ll.get("agg_decode_tps")),
                    ]) + " |")
            L.append("")
    return "\n".join(L)


def _probe_verdict(ab_client, ab_server, skew):
    # NOTE: this attribution presumes a CLIENT-side deficit exists to explain. It
    # only discriminates ceiling-vs-starvation once gm is actually losing on the
    # client aggregate; if gm wins/ties there is nothing to attribute, so check
    # that FIRST (else intra-batch finish-order skew gets mislabelled "starvation").
    # It also does NOT cover prefill-arriving-during-decode (the continuous-arrival
    # scenario) -- both probe regimes use simultaneous submit, so a "no cliff" here
    # only clears the simultaneous-cold-prefill case; see the late-arrival test.
    if ab_client is None:
        return "insufficient data"
    if ab_client >= 0.95:
        return ("NO CLIFF in simultaneous cold prefill (gm wins/ties client agg) -- "
                "throughput-ceiling A falsified here; prefill-into-decode (B) still untested")
    if skew and skew > 2.0:
        return "TTFT STARVATION (high skew) -> seam patch B/C (admission/priority)"
    if ab_server and ab_client and ab_server > 1.15 * ab_client:
        return ("QUEUE/SCHEDULING (server forward competitive, client TTFT lags) "
                "-> seam patch B/C")
    if ab_client < 0.9:
        return ("THROUGHPUT CEILING (forward batch-prefill deficit, low skew) "
                "-> fusion A / padding D likely needed")
    return "NO MATERIAL CLIFF at this point (client A/B ~>=0.9) -- re-check deeper/wider"


def write_probe_outputs(outdir, stamp, meta, cfg, specs, results):
    md = render_probe_md(meta, specs, cfg, results)
    md_path = os.path.join(outdir, f"cold-prefill-probe-{stamp}.md")
    json_path = os.path.join(outdir, f"cold-prefill-probe-{stamp}.json")
    tmp = json_path + ".tmp"
    dump = {"meta": meta, "config": cfg,
            "models": [{"name": s.name, "gguf": s.gguf, "tokenizer": s.tokenizer}
                       for s in specs],
            "results": {"|".join(map(str, k)): v for k, v in results.items()}}
    with open(md_path, "w") as fh:
        fh.write(md)
    with open(tmp, "w") as fh:
        json.dump(dump, fh, indent=2, default=str)
    os.replace(tmp, json_path)
    return md_path, json_path, md


def run_probe(args, specs, cfg, runtimes, toks, lock, rng):
    os.makedirs(args.outdir, exist_ok=True)
    results = {}
    thermals = []
    stamp = time.strftime("%Y%m%d-%H%M%S")

    def _meta(partial):
        return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "partial": partial,
                "mode": "cold-prefill-probe", "llama_server_bin": args.llama_server_bin,
                "runtimes": runtimes, "thermals": thermals}

    def _flush():
        write_probe_outputs(args.outdir, stamp, _meta(True), cfg, specs, results)

    for rnd in range(cfg["rounds"]):
        order = list(runtimes)
        if rnd % 2 == 1:
            order = order[::-1]
        therm = thermal_snapshot()
        thermals.append({"round": rnd + 1, "order": order, "therm": therm})
        print(f"\n=== probe round {rnd + 1}/{cfg['rounds']} | order {order} | {therm} ===")
        for runtime in order:
            for spec in specs:
                print(f"  -- {spec.name} / {runtime}")
                try:
                    measure_probe_block(args, cfg, spec, runtime, toks[spec.name],
                                        lock, rng, results, args.outdir, rnd, flush_cb=_flush)
                except Exception as e:  # noqa: BLE001 - one bad arm shouldn't kill the run
                    reason = (diagnose_failure(args.outdir, spec.name, runtime, f"r{rnd + 1}")
                              or str(e))
                    print(f"    [error] {spec.name}/{runtime}: {reason}")
                    results.setdefault((spec.name, runtime, "_failed"), {})["reason"] = reason
                _flush()

    md_path, json_path, md = write_probe_outputs(
        args.outdir, stamp, _meta(False), cfg, specs, results)
    print("\n" + md)
    print(f"\nwrote {md_path}\nwrote {json_path}")


# ---------------------------------------------------------------------------
def main():
    global GMLX_PORT, LLAMA_PORT, DS4_PORT   # rebound from --*-port flags
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", nargs="*", help="GGUF path(s) for a quick baseline-only "
                    "run (use --config for MTP arms / per-model tokenizers)")
    ap.add_argument("--config", help="JSON: {depths, concurrency, max_tokens, rounds, "
                    "cooldown, sampling, models:[{name,gguf,tokenizer,mtp:{...}}]}")
    ap.add_argument("--tokenizer", help="default HF id / path when a model omits one")
    ap.add_argument("--corpus", help="text file for realistic prompts (else embedded). "
                    "Raw text is for BASE/continuation targets; instruct targets want "
                    "--chat-dataset instead.")
    ap.add_argument("--chat-dataset", help="HF multi-turn chat dataset for INSTRUCT-target "
                    "MTP prompts (real conversations sent as messages, chat-templated "
                    "server-side => draft head on-distribution; raw --corpus understates "
                    "acceptance). e.g. HuggingFaceH4/ultrachat_200k. Overrides --corpus "
                    "and the chat_dataset config key.")
    ap.add_argument("--chat-split", default=None,
                    help="split for --chat-dataset (default train_sft; "
                    "config key chat_split)")
    ap.add_argument("--chat-max-convs", type=int, default=None,
                    help="conversations to load for --chat-dataset (default 8000; "
                    "config key chat_max_convs)")
    ap.add_argument("--depths", help="comma list, e.g. 0,1024,4096,8192")
    ap.add_argument("--concurrency", help="comma list, e.g. 1,2,3")
    ap.add_argument("--draft-depths", help="MTP draft depths to sweep, e.g. 1,2,3 "
                    "(both runtimes set to each N)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--cooldown", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--ramp-tokens", type=int, default=256,
                    help="post-restart clock-ramp prompt tokens, unmeasured (0=off)")
    ap.add_argument("--requests", type=int, default=4, help="recorded requests/cell/round "
                    "(per concurrent slot)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-request "
                    "read timeout (between-bytes on a stream). gmlx emits "
                    "SSE keepalives that reset it; llama-server does not, so "
                    "deep dense prefills need the full headroom")
    ap.add_argument("--startup-timeout", type=float, default=240.0,
                    help="seconds to wait for a server to load + answer /v1/models")
    ap.add_argument("--vs", choices=("llama", "ds4"), default="llama",
                    help="comparison arm vs gmlx (default llama; ds4 = "
                         "dwarfstar ds4-server)")
    ap.add_argument("--gmlx-bin", default=shutil.which("gmlx") or "gmlx",
                    help="gmlx executable (default: from PATH)")
    ap.add_argument("--llama-server-bin",
                    default=shutil.which("llama-server") or "llama-server",
                    help="llama-server executable (default: from PATH)")
    ap.add_argument("--ds4-bin", default=shutil.which("ds4-server") or "ds4-server",
                    help="ds4-server executable (default: from PATH)")
    ap.add_argument("--gmlx-port", type=int, default=GMLX_PORT,
                    help="port for the gmlx server (override to dodge a "
                         "stale server already on the default)")
    ap.add_argument("--llama-port", type=int, default=LLAMA_PORT,
                    help="port for llama-server (override to dodge a stale server)")
    ap.add_argument("--ds4-port", type=int, default=DS4_PORT,
                    help="port for ds4-server (override to dodge a stale server)")
    ap.add_argument("--only", help="comma subset of runtimes: gmlx,llama,ds4")
    ap.add_argument("--mtp-only", action="store_true",
                    help="skip the baseline arm; measure only the MTP arms")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--no-svg", action="store_true",
                    help="skip SVG chart rendering after the run")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the full plan + every server command line, launch nothing")
    ap.add_argument("--probe", action="store_true",
                    help="Step-0 cold concurrent-prefill probe (true-simultaneous submit, "
                    "aggregate throughput + TTFT-fairness + server-rate attribution; baseline "
                    "arm only). Defaults to deep depths + c=1..4 unless --depths/--concurrency given.")
    ap.add_argument("--probe-regimes", help="comma subset of probe regimes to run: "
                    "prefill_only,with_decode (default both)")
    args = ap.parse_args()

    GMLX_PORT = args.gmlx_port   # honor --*-port overrides everywhere
    LLAMA_PORT = args.llama_port
    DS4_PORT = args.ds4_port

    try:
        sys.stdout.reconfigure(line_buffering=True)   # flush progress even when piped to a file
    except (AttributeError, ValueError):
        pass

    if not args.config and not args.gguf:
        sys.exit("give GGUF path(s) or --config")
    specs, cfg = load_specs(args)
    if args.probe:
        # Probe targets COLD prefill at depth under concurrency; the full-bench
        # config's shallow depths / c<=3 don't apply. Use deep defaults + c=1..4
        # unless the CLI explicitly overrides (config values are ignored here).
        if not args.depths:
            cfg["depths"] = [4096, 16384]
        if not args.concurrency:
            cfg["concurrency"] = [1, 2, 3, 4]
        if args.probe_regimes:
            cfg["_probe_regimes"] = [r.strip() for r in args.probe_regimes.split(",")]
    pair = ["gmlx", args.vs]
    runtimes = ([r.strip() for r in args.only.split(",")] if args.only else list(pair))
    for r in runtimes:
        if r not in BUILDERS:
            sys.exit(f"unknown runtime {r!r} in --only (choices: {sorted(BUILDERS)})")
    if args.probe and "ds4" in runtimes:
        sys.exit("--probe is llama-vs-gmlx only (it parses llama-server "
                 "stdout rates); drop --vs ds4 / --only ds4")
    if "ds4" in runtimes:
        if cfg.get("thinking") is not False and \
                cfg["sampling"].get("temperature") not in (None, 1.0):
            print("[warn] ds4 thinking mode is ON (thinking != false): it will "
                  "OVERRIDE the requested sampling to temp 1.0 / top_p 1.0 / "
                  "top_k 0 / min_p 0.05 while gmlx honors "
                  f"{cfg['sampling']} -- set \"thinking\": false or match "
                  "sampling to ds4's forced defaults for a fair A/B")
        if any(s.kv_bits for s in specs):
            print("[warn] kv_bits has no ds4 mapping (fixed MLA latent cache); "
                  "the gmlx arm would quantize KV while ds4 does not -- "
                  "unset kv_bits for a fair A/B")
    args._max_conc = max(cfg["concurrency"])

    for spec in specs:
        if not spec.tokenizer:
            sys.exit(f"model {spec.name}: no tokenizer (set per-model or --tokenizer)")

    if args.dry_run and args.probe:
        print("=== COLD-PREFILL PROBE PLAN (dry-run, nothing launched) ===")
        print(f"runtimes: {runtimes}  rounds: {cfg['rounds']} (alternating order)")
        print(f"depths: {cfg['depths']}  concurrency: {cfg['concurrency']} (true-simultaneous)")
        print(f"regimes: {[r for r, _ in probe_regimes(cfg)]}  "
              f"with_decode max_tokens: {cfg['max_tokens']}")
        print(f"sampling (identical both sides): {cfg['sampling']}")
        _cor = (f"chat:{args.chat_dataset}:{args.chat_split}" if args.chat_dataset
                else f"corpus:{args.corpus}" if args.corpus
                else "embedded-tiled (unrepresentative; smoke tests only)")
        print(f"corpus: {_cor}")
        print(f"llama -c (ctx across {args._max_conc} slots): "
              f"{runtime_ctx('llama', args._max_conc, cfg)}")
        ncells = len(cfg['depths']) * len(probe_regimes(cfg)) * len(cfg['concurrency'])
        print(f"cells/runtime/round: {ncells}  | server launches: "
              f"{cfg['rounds'] * len(runtimes)} (baseline/MTP-off, APC off)")
        for spec in specs:
            print(f"\n# {spec.name}  ({spec.gguf})  tok={spec.tokenizer}")
            for rt in runtimes:
                cmd = BUILDERS[rt](args, spec, None,
                                   runtime_ctx(rt, args._max_conc, cfg))
                print(f"  {rt}/baseline: " + (" ".join(cmd) if cmd else "SKIP"))
        print("\nMetrics: agg cold-prefill throughput + scaling eff + TTFT skew + "
              "server-forward rate -> ceiling-vs-starvation attribution.")
        return

    if args.dry_run:
        print("=== PLAN (dry-run, nothing launched) ===")
        print(f"runtimes: {runtimes}  rounds: {cfg['rounds']} (alternating order)")
        print(f"depths: {cfg['depths']}  concurrency: {cfg['concurrency']}  "
              f"draft_depths: {cfg['draft_depths']}  "
              f"max_tokens: {cfg['max_tokens']}  requests/cell/round: {args.requests}")
        print(f"sampling (identical both sides): {cfg['sampling']}")
        _cor = (f"chat:{args.chat_dataset}:{args.chat_split}" if args.chat_dataset
                else f"corpus:{args.corpus}" if args.corpus
                else "embedded-tiled (unrepresentative; smoke tests only)")
        print(f"corpus: {_cor}")
        for rt in runtimes:
            if rt == "llama":
                print(f"llama -c (ctx across {args._max_conc} slots): "
                      f"{runtime_ctx(rt, args._max_conc, cfg)}")
            elif rt == "ds4":
                print(f"ds4 -c (per-sequence; requests serialized): "
                      f"{runtime_ctx(rt, args._max_conc, cfg)}  "
                      f"thinking={cfg.get('thinking')}")
        for spec in specs:
            arms = arms_for(spec, cfg)
            print(f"\n# {spec.name}  ({spec.gguf})  tok={spec.tokenizer}  "
                  f"arms={[a for a, _ in arms]}")
            for rt in runtimes:
                for arm_label, draft_n in arms:
                    cmd = BUILDERS[rt](args, spec, draft_n,
                                       runtime_ctx(rt, args._max_conc, cfg))
                    print(f"  {rt}/{arm_label}: " + (" ".join(cmd) if cmd
                                                     else "SKIP (no MTP config)"))
        print("\nForced-length both sides (--ignore-eos / ds4 ignore_eos patch); "
              "caches off; unique nonce prompts.")
        return

    # preflight binaries
    if "gmlx" in runtimes and not shutil.which(args.gmlx_bin):
        sys.exit(f"gmlx not found ({args.gmlx_bin}); pip install gmlx "
                 f"or pass --gmlx-bin")
    if "llama" in runtimes and not shutil.which(args.llama_server_bin):
        sys.exit(f"llama-server not found ({args.llama_server_bin})\n"
                 f"  build llama.cpp: cmake -B build -DGGML_METAL=ON "
                 f"-DCMAKE_BUILD_TYPE=Release && cmake --build build -j "
                 f"--target llama-server; then pass --llama-server-bin")
    if "ds4" in runtimes and not shutil.which(args.ds4_bin):
        sys.exit(f"ds4-server not found ({args.ds4_bin})\n"
                 f"  build ds4-server with patches/ds4-ignore-eos.patch "
                 f"applied, then pass --ds4-bin")

    # corpus: chat dataset (instruct MTP) > raw text file > embedded fallback
    if args.chat_dataset:
        args._corpus = load_chat_corpus(args.chat_dataset, args.chat_split,
                                        args.chat_max_convs)
        print(f"[chat] {args._corpus.source} -> {len(args._corpus)} conversations "
              f"(multi-turn, chat-templated; instruct-target MTP corpus)")
    elif args.corpus:
        with open(args.corpus) as fh:
            args._corpus = fh.read().split()
        if len(args._corpus) < 64:
            sys.exit("--corpus too small")
    else:
        args._corpus = _EMBEDDED_CORPUS
        print("[warn] using embedded corpus (tiled); pass --chat-dataset (instruct MTP) "
              "or --corpus (raw text) for representative acceptance")

    toks = {spec.name: _build_tokenizer(spec) for spec in specs}
    lock = threading.Lock()
    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    if args.probe:
        run_probe(args, specs, cfg, runtimes, toks, lock, rng)
        return

    results = {}
    thermals = []
    stamp = time.strftime("%Y%m%d-%H%M%S")   # fixed up front; one output pair, rewritten in place

    if isinstance(args._corpus, ChatCorpus):
        corpus_label = f"chat:{args._corpus.source}"
    elif args.corpus:
        corpus_label = f"corpus:{os.path.basename(args.corpus)}"
    else:
        corpus_label = "embedded-tiled"

    ds4_commit = None
    if "ds4" in runtimes:
        try:
            ds4_commit = subprocess.run(
                ["git", "-C", os.path.dirname(args.ds4_bin), "rev-parse",
                 "--short", "HEAD"], capture_output=True, text=True,
                timeout=5).stdout.strip() or None
        except Exception:
            pass

    def _meta(partial):
        return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "partial": partial,
                "requests": args.requests, "llama_server_bin": args.llama_server_bin,
                "ds4_bin": args.ds4_bin if "ds4" in runtimes else None,
                "ds4_commit": ds4_commit, "thinking": cfg.get("thinking"),
                "runtime_pair": pair,
                "runtimes": runtimes, "thermals": thermals, "corpus": corpus_label,
                "cool_gate_timeouts": list(_COOL_TIMEOUTS),
                # provenance for the MTP-fairness levers: gmlx owned single-stream
                # serve round (now the package default, no env gate) + per-model llama
                # native-head isolation (no --spec-default).
                "gmlx_owned_mtp_round": "package-default",
                "llama_spec_default": {s.name: (s.mtp_llama.get("spec_default", True)
                                                if s.mtp_llama else None)
                                       for s in specs}}

    def _flush():
        write_outputs(args.outdir, stamp, _meta(True), cfg, specs, results)

    for rnd in range(cfg["rounds"]):
        order = list(runtimes)
        if rnd % 2 == 1:
            order = order[::-1]                 # alternate runtime order per round
        therm = thermal_snapshot()
        thermals.append({"round": rnd + 1, "order": order, "therm": therm})
        print(f"\n=== round {rnd + 1}/{cfg['rounds']} | order {order} | {therm} ===")
        for runtime in order:
            for spec in specs:
                for arm_label, draft_n in arms_for(spec, cfg):
                    print(f"  -- {spec.name} / {runtime} / {arm_label}")
                    try:
                        measure_block(args, cfg, spec, runtime, arm_label, draft_n,
                                      toks[spec.name], lock, rng, results, args.outdir,
                                      flush_cb=_flush, round_idx=rnd)
                    except Exception as e:  # noqa: BLE001 - one bad cell shouldn't kill the run
                        reason = diagnose_failure(args.outdir, spec.name, runtime,
                                                  arm_label) or str(e)
                        print(f"    [error] {spec.name}/{runtime}/{arm_label}: {reason}")
                        results.setdefault((spec.name, runtime, arm_label), {})["_failed"] = reason
                        results.setdefault((spec.name, runtime, arm_label), {})["_error"] = str(e)
                    _flush()                    # persist after every arm too

    md_path, json_path, md = write_outputs(
        args.outdir, stamp, _meta(False), cfg, specs, results)
    print("\n" + md)
    print(f"\nwrote {md_path}\nwrote {json_path}")
    if not args.no_svg:
        render_svgs(args, specs, runtimes, json_path)


if __name__ == "__main__":
    main()
