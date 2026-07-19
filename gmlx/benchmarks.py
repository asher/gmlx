"""Throughput benchmarks driven through the deployed generation paths.

``bench`` measures prefill/decode tok/s per prompt length; ``bench_tg_depth``
measures decode tok/s at each context depth, with an optional MTP speculative
A/B (accept rate + speedup) per depth. ``_ChatPromptSource`` feeds both from a
HuggingFace chat dataset so accept rates are content-comparable across runs.
"""

from __future__ import annotations

import os
import time

import mlx.core as mx
import mlx.nn as nn

from .generation import (
    _PREFILL_CHUNK,
    _chunked_prefill_cache,
    generate_speculative,
    generate_speculative_owned,
)
from . import loader


def _synth_prompt_ids(tokenizer, n: int) -> list[int]:
    """Filler token ids of exactly length ``n`` for benchmarking."""
    seed = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    if not seed:
        seed = [1, 2, 3, 4, 5]
    out: list[int] = []
    while len(out) < n:
        out.extend(seed)
    return out[:n]


# Chat-dataset prompt source for bench_tg_depth
def _load_chat_dataset(dataset_id: str, split: str = "train_sft",
                       max_convs: int = 4000) -> list[list[dict]]:
    """Load a multi-turn chat dataset into normalized conversations.

    Each conversation is a list of {role, content} dicts starting with a user
    turn. Accepts OpenAI ``messages`` schema (ultrachat_200k, smoltalk, tulu-3)
    and ShareGPT ``conversations`` schema ({from, value}).
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    ds = ds.select(range(min(len(ds), max(max_convs * 3, 2000))))
    role_map = {
        "human": "user", "user": "user",
        "gpt": "assistant", "assistant": "assistant",
        "chatgpt": "assistant", "bard": "assistant",
        "system": None, "tool": None,
    }
    convs: list[list[dict]] = []
    for row in ds:
        turns = row.get("messages")
        if not turns and row.get("conversations"):
            turns = [{"role": role_map.get((t.get("from") or "").lower()),
                      "content": t.get("value") or ""}
                     for t in row["conversations"]]
        if not turns:
            continue
        norm: list[dict] = []
        for t in turns:
            role = role_map.get((t.get("role") or "").lower(), t.get("role"))
            content = (t.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                continue
            if norm and norm[-1]["role"] == role:
                norm[-1]["content"] += "\n\n" + content
            else:
                norm.append({"role": role, "content": content})
        while norm and norm[0]["role"] != "user":
            norm.pop(0)
        # Every conversation starts with user and ends with assistant, so
        # concatenating them keeps roles alternating across the seams.
        while norm and norm[-1]["role"] != "assistant":
            norm.pop()
        if norm:
            convs.append(norm)
        if len(convs) >= max_convs:
            break
    if len(convs) < 4:
        raise ValueError(
            f"chat dataset {dataset_id}:{split} yielded {len(convs)} "
            f"usable conversations (need >=4)"
        )
    return convs


class _ChatPromptSource:
    """Yields chat-templated token-id prompts of a target length from a
    HuggingFace chat dataset. The k-th prompt at a given target length is a
    pure function of ``(seed, target_len, k)``: a depth draws the same
    conversation slice whether it is benched alone or inside any
    ``--bench-depths`` sweep, so accept rates stay content-comparable
    across protocols and across days. Vary ``seed`` (``--bench-chat-seed``)
    to re-check a close decision against a different slice."""

    def __init__(self, convs: list[list[dict]], tokenizer, seed: int = 42):
        self._convs = convs
        self._tokenizer = tokenizer
        self._seed = seed
        self._len_calls: dict[int, int] = {}

    def _order_for(self, target_len: int) -> list[int]:
        import random
        k = self._len_calls.get(target_len, 0)
        self._len_calls[target_len] = k + 1
        rng = random.Random(f"{self._seed}:{int(target_len)}:{k}")
        order = list(range(len(self._convs)))
        rng.shuffle(order)
        return order

    def get(self, target_len: int) -> list[int]:
        """Return chat-templated token ids of approximately ``target_len``."""
        tok = self._tokenizer
        order = self._order_for(target_len)
        msgs: list[dict] = []
        total = 0
        safety = 0
        pos = 0
        while total < target_len and safety < 8192:
            conv = self._convs[order[pos % len(order)]]
            pos += 1
            for t in conv:
                msgs.append(t)
                safety += 1
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True
            )
            total = len(ids)

        while msgs and msgs[-1]["role"] != "user":
            msgs.pop()
        if not msgs:
            msgs = [{"role": "user", "content": "Hello."}]

        while len(msgs) > 1:
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True
            )
            if len(ids) <= target_len:
                break
            # Front-trim to the next user turn: a list left starting with an
            # assistant turn is rejected by alternation-enforcing templates
            # (Mistral / Llama-2 raise inside apply_chat_template, above).
            msgs.pop(0)
            while len(msgs) > 1 and msgs[0]["role"] != "user":
                msgs.pop(0)
        else:
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True
            )

        if len(ids) < target_len:
            pad = _synth_prompt_ids(tok, target_len - len(ids))
            ids = pad + ids
        return ids


def bench(
    model,
    tokenizer,
    lengths=(512, 4096, 16384),
    *,
    decode_tokens: int = 32,
    runs: int = 2,
    warmup: bool = True,
    prefill_step_size: int | None = None,
) -> dict[int, dict[str, float]]:
    """Measure prefill and decode throughput (tokens/sec) per prompt length.

    Drives the real generation path (``mlx_lm.stream_generate``): prefill is
    chunked (reported as ``prompt_tps``) and decode runs through the async
    one-step-ahead pipeline (``generation_tps``) - so the numbers match deployed
    throughput. A single un-chunked forward understates prefill ~3x and an eager
    per-token loop understates decode ~10-14%; both are avoided here.
    ``prefill_step_size`` follows the deployed default too: explicit value >
    streaming-mode 8192 > mlx-lm's own 2048.

    Returns ``{length: {"prefill_tps": float, "decode_tps": float}}`` using the
    best (max-tps) run per length to suppress pipeline-warmup noise.
    """
    import mlx_lm

    step, defaulted = loader._resolve_prefill_step(model, prefill_step_size)
    if defaulted:
        print(f"[bench] streaming model: prefill chunk size defaults to {step}")
    pf_kwargs = {} if step is None else {"prefill_step_size": step}

    if warmup:
        wn = min(max(lengths), 256)
        for _ in mlx_lm.stream_generate(
            model, tokenizer, _synth_prompt_ids(tokenizer, wn), max_tokens=4
        ):
            pass

    results: dict[int, dict[str, float]] = {}
    for L in lengths:
        prefill_tps: list[float] = []
        decode_tps: list[float] = []
        for _ in range(runs):
            prompt = _synth_prompt_ids(tokenizer, L)
            final = None
            for resp in mlx_lm.stream_generate(
                model, tokenizer, prompt, max_tokens=decode_tokens, **pf_kwargs
            ):
                final = resp
            prefill_tps.append(final.prompt_tps)
            decode_tps.append(final.generation_tps)
        results[L] = {"prefill_tps": max(prefill_tps), "decode_tps": max(decode_tps)}
    return results


class _RawLogitsLM(nn.Module):
    """mlx-lm-compatible view of an owned-engine MTP target: unwraps
    ``LanguageModelOutput.logits`` so ``mlx_lm.stream_generate`` can drive
    the same model for the plain (no-MTP) baseline in ``bench_tg_depth``."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def make_cache(self):
        return self.lm.make_cache()

    def __call__(self, inputs, cache=None, **kwargs):
        out = self.lm(inputs, cache=cache, **kwargs)
        return out.logits if hasattr(out, "logits") else out


def _bench_ar_tps(
    model, tokenizer, seed_ids, *, decode_tokens: int,
    prefill_chunk: int | None = None,
):
    """Plain (no-MTP) AR prefill/decode tok/s via mlx-vlm's engine - the A/B
    baseline for ``generate_speculative`` on the same mlx-vlm target. (mlx-lm's
    stream_generate can't consume the target's ``LanguageModelOutput``.) Prefill
    is split off at the first-token boundary; ``decode_tps`` is the steady decode
    window only, so it's directly comparable to the speculative ``decode_tps``.
    Returns ``(prefill_tps, decode_tps)``.
    """
    from mlx_lm.sample_utils import make_sampler
    from mlx_vlm.generate.ar import generate_step

    input_ids = mx.array(seed_ids, dtype=mx.int32)[None]
    sampler = make_sampler(temp=0.0)
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    n = 0
    tic = time.perf_counter()
    prefill_s = None
    # Chunk the target prefill exactly like generate_speculative: hand
    # generate_step only the trailing token so no single expert-gather forward
    # exceeds the safe width, and the A/B baseline pays the same (deployed)
    # chunked-prefill cost. The pre-prefill wall time lands inside the
    # first-token window, so prefill_tps stays correct.
    chunk = prefill_chunk or _PREFILL_CHUNK
    feed, prompt_cache = input_ids, None
    if input_ids.shape[1] > chunk:
        prompt_cache = _chunked_prefill_cache(
            model.language_model, input_ids, chunk
        )
        feed = input_ids[:, -1:]
    for tok, _lp in generate_step(
        feed,
        model,
        None,
        None,
        prompt_cache=prompt_cache,
        max_tokens=decode_tokens,
        temperature=0.0,
        sampler=sampler,
        draft_model=None,
    ):
        if prefill_s is None:
            prefill_s = time.perf_counter() - tic
            tic = time.perf_counter()
        tok = int(tok[0]) if isinstance(tok, list) else int(tok)
        if tok in eos_ids:
            break
        n += 1
        if n >= decode_tokens:
            break
    decode_s = time.perf_counter() - tic
    prefill_tps = (len(seed_ids) / prefill_s) if prefill_s and prefill_s > 0 else 0.0
    decode_tps = (n / decode_s) if decode_s > 0 else 0.0
    return prefill_tps, decode_tps


def bench_tg_depth(
    model,
    tokenizer,
    depths=(0, 4096, 16384),
    *,
    decode_tokens: int = 128,
    runs: int = 2,
    warmup: bool = True,
    drafter=None,
    draft_block_size: int | None = None,
    prefill_step_size: int | None = None,
    temp: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 0,
    min_p: float = 0.05,
    prompt_source: "_ChatPromptSource | None" = None,
) -> dict[int, dict[str, float]]:
    """Token-generation throughput measured at each context depth.

    For each depth ``D``: prefill to ``D`` (the cache setup is untimed beyond the
    reported ``prefill_tps``), then time a ``decode_tokens``-long decode run from
    that depth - so ``tg`` (decode tok/s) is reported as its own column per depth
    rather than coupled to a single prompt length. ``depths`` may include ``0``
    (decode from a 1-token prefill, i.e. shallow).

    When ``drafter`` is given, each depth additionally runs MTP speculative decode
    from a depth-``D`` prompt and reports ``accept_rate``, ``mean_accept_len``,
    ``spec_tps`` and the ``speedup`` over the plain decode at that depth.

    Returns ``{depth: {prefill_tps, tg_tps[, accept_rate, mean_accept_len,
    spec_tps, speedup]}}`` (best run per depth).
    """
    import mlx_lm

    def _seed_len(D: int) -> int:
        return max(1, int(D))

    # Same prefill-width policy as deployed generation (explicit > streaming
    # 8192 > stock). The mlx-lm path takes it as a stream_generate kwarg; the
    # drafter A/B baseline chunks through _bench_ar_tps(prefill_chunk=...).
    step, defaulted = loader._resolve_prefill_step(model, prefill_step_size)
    if defaulted:
        print(f"[bench] streaming model: prefill chunk size defaults to {step}")
    pf_kwargs = {} if step is None else {"prefill_step_size": step}

    def _get_ids(n: int) -> list[int]:
        if prompt_source is not None:
            return prompt_source.get(n)
        return _synth_prompt_ids(tokenizer, n)

    # Owned-engine targets are mlx-lm-style models (own make_cache; logits
    # unboxed by _RawLogitsLM): their honest plain baseline is the deployed
    # mlx-lm path, not mlx-vlm's engine (which loses ~30% at depth and would
    # flatter the speculative speedup).
    owned = drafter is not None and getattr(
        drafter, "requires_owned_engine", False
    )
    plain_lm = model
    if owned and hasattr(model, "language_model"):
        plain_lm = _RawLogitsLM(model.language_model)

    if warmup:
        if drafter is not None and not owned:
            _bench_ar_tps(
                model, tokenizer, _synth_prompt_ids(tokenizer, 64), decode_tokens=4
            )
        else:
            for _ in mlx_lm.stream_generate(
                plain_lm, tokenizer, _synth_prompt_ids(tokenizer, 64), max_tokens=4
            ):
                pass

    results: dict[int, dict[str, float]] = {}
    for D in depths:
        # Buffer-cache hygiene: each depth (and each arm) allocates its own
        # shape-set, and the cached pools of prior sets push total resident
        # past the wired limit -- decode then unwires mid-process and later
        # depths read progressively slower.
        mx.clear_cache()
        prefill_tps: list[float] = []
        tg_tps: list[float] = []
        seed = _seed_len(D)
        # One prompt per depth, shared by the plain and speculative arms:
        # spec round time is content-sensitive beyond the accept rate, so
        # arms measured on different prompts make the speedup column noisy.
        seed_ids = _get_ids(seed)
        for _ in range(runs):
            if drafter is not None and not owned:
                # baseline = same mlx-vlm target, no drafter (apples-to-apples)
                p_tps, d_tps = _bench_ar_tps(
                    model, tokenizer, seed_ids, decode_tokens=decode_tokens,
                    prefill_chunk=step,
                )
            else:
                final = None
                for resp in mlx_lm.stream_generate(
                    plain_lm, tokenizer, seed_ids, max_tokens=decode_tokens,
                    **pf_kwargs
                ):
                    final = resp
                p_tps, d_tps = final.prompt_tps, final.generation_tps
            prefill_tps.append(p_tps)
            tg_tps.append(d_tps)

        row = {"prefill_tps": max(prefill_tps), "tg_tps": max(tg_tps)}

        if drafter is not None:
            mx.clear_cache()
            # Owned round is the serve default; the bench matches it so the
            # A/B measures the production path. GMLX_OWNED_ROUND=0 opts
            # back to mlx-vlm's generate_speculative.
            _spec_fn = (
                generate_speculative
                if os.environ.get("GMLX_OWNED_ROUND") == "0"
                else generate_speculative_owned
            )
            best = None
            for _ in range(runs):
                stats = _spec_fn(
                    model,
                    drafter,
                    tokenizer,
                    seed_ids,
                    max_tokens=decode_tokens,
                    temp=temp,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    draft_block_size=draft_block_size,
                    apply_chat_template=False,
                    verbose=False,
                )
                if best is None or stats["decode_tps"] > best["decode_tps"]:
                    best = stats
            row["accept_rate"] = best["accept_rate"]
            row["mean_accept_len"] = best["mean_accept_len"]
            row["rounds"] = best.get("rounds", 0)
            row["draft_n"] = best.get("draft_n", 0)
            row["draft_n_accepted"] = best.get("draft_n_accepted", 0)
            row["spec_tps"] = best["decode_tps"]
            row["speedup"] = (
                best["decode_tps"] / row["tg_tps"] if row["tg_tps"] > 0 else 0.0
            )

        results[D] = row
    return results
