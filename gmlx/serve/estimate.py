"""Queryable admission: dry-run estimates, the capacity plan, and rates.

The batch is a governed frontier (width x depth, bounded by memory and
the governor band) rather than a fixed slot count, and the memory
preflight (``mem_preflight``) already tokenizes a prompt and prices its
KV against the drained working set before admitting or refusing it.
This module exposes that computation - and the capacity table's policy -
as questions a client can ask before sending:

* :func:`estimate_request` - the dry-run. Same body as
  ``/v1/chat/completions``; renders the chat template, tokenizes, prices
  the KV the way the preflight does, asks the resident model's APC how
  much of the prefix is already warm, and answers ``fits_now`` /
  ``fits_drained`` with the numbers, plus an ETA to first token. Never
  loads a model: a non-resident model answers ``resident: false``.
* :func:`capacity_plan` - "can I run ``width`` streams at ``depth``
  tokens each, and may I start them now?" - the policy of the pi plan
  (section 5.4) evaluated where the numbers live: the capacity table for
  the geometry, the governor band and the live concurrency for the
  timing.
* :func:`rates_view` - the ``server.rates`` section of ``/v1/metrics``:
  the live aggregate decode rate (the sum over decoding rows), and the
  recent per-request prefill / decode rates the ETA is computed from.

Everything here is read-only and best-effort: a probe that fails leaves
its field ``None`` rather than failing the request.
"""

from __future__ import annotations

import importlib
import logging
import os
import time

_log = logging.getLogger(__name__)

# How many recently completed requests feed the recent-rate means.
_RECENT_N = 8


# --- rates --------------------------------------------------------------

def _recent_rates(metrics) -> dict:
    """Mean prefill / decode tok/s over the last ``_RECENT_N`` completed
    requests (only those that reported a rate)."""
    out = {"prefill_tok_s": None, "decode_tok_s": None, "n": 0}
    try:
        recent = list(getattr(metrics, "_recent", ()) or ())[-_RECENT_N:]
    except Exception:
        return out
    pre = [float(p.get("prefill_tok_s") or 0) for p in recent
           if isinstance(p, dict) and (p.get("prefill_tok_s") or 0) > 0]
    dec = [float(p.get("decode_tok_s") or 0) for p in recent
           if isinstance(p, dict) and (p.get("decode_tok_s") or 0) > 0]
    out["n"] = len(recent)
    if pre:
        out["prefill_tok_s"] = round(sum(pre) / len(pre), 1)
    if dec:
        out["decode_tok_s"] = round(sum(dec) / len(dec), 1)
    return out


def rates_view() -> dict:
    """``server.rates``: ``decode_tok_s`` (aggregate over the rows decoding
    right now), ``decode_streams``, the recent per-request means
    ``prefill_tok_s_recent`` / ``decode_tok_s_recent``, and the lifetime
    mean ``decode_tok_s_lifetime``."""
    out = {"decode_tok_s": 0.0, "decode_streams": 0,
           "prefill_tok_s_recent": None, "decode_tok_s_recent": None,
           "decode_tok_s_lifetime": None}
    try:
        from .live_requests import live_requests_view

        rows = [r for r in live_requests_view() if r.get("state") == "decode"]
        out["decode_streams"] = len(rows)
        out["decode_tok_s"] = round(sum(float(r.get("decode_tok_s") or 0)
                                        for r in rows), 1)
    except Exception:
        pass
    try:
        runtime = importlib.import_module("mlx_vlm.server.runtime").runtime
        metrics = getattr(runtime, "metrics", None)
        recent = _recent_rates(metrics)
        out["prefill_tok_s_recent"] = recent["prefill_tok_s"]
        out["decode_tok_s_recent"] = recent["decode_tok_s"]
        gen = float(getattr(metrics, "_generated_tokens_total", 0) or 0)
        secs = float(getattr(metrics, "_decode_time_total_s", 0.0) or 0.0)
        if gen > 0 and secs > 0:
            out["decode_tok_s_lifetime"] = round(gen / secs, 1)
    except Exception:
        pass
    return out


# --- capacity plan --------------------------------------------------------

def _table_ctx_at_width(t: dict, width: int) -> int | None:
    """Max context at ``width`` from the table: the entry for the smallest
    tabulated width >= ``width`` (conservative between rows). None past
    the widest row."""
    ctx = t.get("max_ctx") or {}
    widths = sorted(int(w) for w in ctx)
    for w in widths:
        if w >= width:
            return int(ctx[w])
    return None


def _table_width_at_depth(t: dict, depth: int) -> int:
    ctx = t.get("max_ctx") or {}
    best = 0
    for w in sorted(int(w) for w in ctx):
        if int(ctx[w]) >= depth:
            best = w
    return best


def capacity_plan(width: int, depth: int) -> dict:
    """The policy for a fan-out of ``width`` streams at ``depth`` tokens.

    ``ok`` is the geometry (the table says ``width`` streams fit at
    ``depth``; None without a table or with overcommit armed).
    ``admit_now`` is the timing on top of it: governor not orange/red,
    nothing waiting, and at least ``width`` free decode slots (one under
    yellow). ``reason`` names the first condition that fails."""
    from .governor import governor_stats
    from .queue_cap import concurrency_stats

    width = max(1, int(width))
    depth = max(0, int(depth))
    out = {"width": width, "depth": depth, "ok": None,
           "max_context_at_width": None, "max_width_at_depth": None,
           "model": None, "band": None, "decode_batch": None,
           "in_flight": None, "waiting": None, "slots": None,
           "admit_now": False, "reason": None}
    try:
        from . import capacity as _cap

        t = _cap.get_table()
        if t is not None and not _cap.overcommit():
            out["max_context_at_width"] = _table_ctx_at_width(t, width)
            out["max_width_at_depth"] = _table_width_at_depth(t, depth)
            out["ok"] = (out["max_context_at_width"] is not None
                         and out["max_context_at_width"] >= depth)
            out["model"] = os.path.basename(str(t.get("path") or "")) or None
            try:
                from . import server_bridge_vlm as serving

                ids = getattr(serving, "_PATH_TO_IDS", {}).get(str(t.get("path"))) or []
                if ids:
                    out["model"] = ids[0]
            except Exception:
                pass
    except Exception:
        _log.debug("capacity plan: table read failed", exc_info=True)
    try:
        band = governor_stats().get("band")
        out["band"] = band
        conc = concurrency_stats()
        out["decode_batch"] = conc.get("decode_batch")
        out["in_flight"] = conc.get("in_flight")
        out["waiting"] = conc.get("waiting")
        dbatch = conc.get("decode_batch") or 0
        in_flight = conc.get("in_flight") or 0
        slots = max(0, int(dbatch) - int(in_flight))
        if band == "yellow":
            slots = min(slots, 1)
        out["slots"] = slots
    except Exception:
        _log.debug("capacity plan: concurrency read failed", exc_info=True)
        slots = None
    # The band the payload reports is the band the reason is judged on,
    # even when the concurrency read failed after it was set.
    band = out["band"]

    if out["ok"] is False:
        out["reason"] = (f"depth {depth} exceeds max context "
                         f"{out['max_context_at_width']} at width {width}")
    elif band in ("orange", "red"):
        out["reason"] = f"governor {band}"
    elif (out["waiting"] or 0) > 0:
        out["reason"] = f"{out['waiting']} waiting for a slot"
    elif slots is not None and slots < width:
        out["reason"] = f"{slots} free slot(s) of {out['decode_batch']}, need {width}"
    else:
        out["admit_now"] = True
        out["reason"] = "ok" if out["ok"] else "ok (no capacity table; geometry unchecked)"
    return out


# --- dry-run admission ----------------------------------------------------

def _normalize_messages(messages) -> tuple:
    """The chat handler's message normalization, text only: returns
    ``(messages, media_count)``. Media (images / audio / video) is counted
    and skipped - the preflight does not price it either."""
    from mlx_vlm.prompt_utils import extract_text_from_content

    out, media = [], 0
    for m in messages or []:
        if not isinstance(m, dict):
            m = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        msg = {"role": m.get("role")}
        content = m.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in (
                        "input_image", "image_url", "input_audio",
                        "input_video", "video_url", "video"):
                    media += 1
            msg["content"] = extract_text_from_content(content)
        else:
            msg["content"] = content
        for k in ("tool_calls", "tool_call_id", "name"):
            if m.get(k) is not None:
                msg[k] = m[k]
        if m.get("reasoning_content") is not None:
            msg["reasoning_content"] = m["reasoning_content"]
            msg["reasoning"] = m["reasoning_content"]
        out.append(msg)
    return out, media


def _warm_tokens(manager, ids: list, extra_hash: int, model=None) -> tuple:
    """``(warm_tokens, tier)``: how much of ``ids`` the APC already holds,
    from the memory block chain, the exact-cache index and (for models on
    the checkpoint tier) the pinned checkpoint records. No probe assembles
    or restores a cache; matched blocks are released at once. The exact
    index probe does open the header of every on-disk exact shard it
    considers (``estimate_ms`` carries that cost). ``tier`` names the
    deepest tier holding the prefix, which is the warm count's source,
    not necessarily the tier the real request would restore from (the
    runtime picks by its own precedence)."""
    best, tier = 0, None
    if manager is None or len(ids) < 2:
        return best, tier
    if model is not None:
        try:
            from .cache_snapshot import ckpt_peek
            from .spec_engine import _ckpt_layout_for

            n = ckpt_peek(manager, ids, extra_hash=extra_hash,
                          layout=_ckpt_layout_for(model, int(manager.block_size)))
            if n > best:
                best, tier = int(n), "ckpt"
        except Exception:
            pass
    try:
        blocks, n = manager.lookup_prefix(ids, extra_hash=extra_hash)
        try:
            if n and n > best:
                best, tier = int(n), "block"
        finally:
            if blocks:
                manager.release(blocks)
    except Exception:
        pass
    try:
        hit = manager.find_exact_prefix(ids, extra_hash=extra_hash)
        if hit:                                  # (cache_hash, prefix_len): disk shards
            n = int(hit[1]) if isinstance(hit, (tuple, list)) else int(hit)
            if n > best:
                best, tier = n, "exact"
    except Exception:
        pass
    n = _exact_peek(manager, ids, extra_hash)
    if n > best:
        best, tier = n, "exact"
    return best, tier


def _exact_peek(manager, ids: list, extra_hash: int) -> int:
    """Deepest in-memory exact-tier entry that is a strict prefix of
    ``ids`` at the same salt (``lookup_exact_cache``'s rule, without the
    clone). The exact tier stores a finished row whole (prompt plus the
    answer), so it serves the next turn of a conversation, never a
    verbatim resend of the same prompt."""
    cache = getattr(manager, "_exact_cache", None)
    if not cache or len(ids) < 2:
        return 0
    tid = tuple(int(t) for t in ids)
    max_len = len(tid) - 1
    best = 0
    try:
        with manager.lock:
            for entry in list(cache.values()):
                toks = getattr(entry, "token_ids", None) or ()
                n = len(toks)
                if (getattr(entry, "extra_hash", 0) != extra_hash or n <= best
                        or n > max_len or tid[:n] != tuple(toks)):
                    continue
                best = n
    except Exception:
        return 0
    return best


def _queue_wait_s(metrics, waiting: int) -> float:
    """Unclamped drain estimate for ``waiting`` requests (the Retry-After
    formula without its 2-60 s clamp); 0 with nothing waiting."""
    if not waiting or waiting <= 0:
        return 0.0
    try:
        done = int(getattr(metrics, "_requests_completed", 0) or 0)
        toks = int(getattr(metrics, "_completion_tokens_total", 0) or 0)
        gen = int(getattr(metrics, "_generated_tokens_total", 0) or 0)
        secs = float(getattr(metrics, "_decode_time_total_s", 0.0) or 0.0)
        if done > 0 and toks > 0 and gen > 0 and secs > 0:
            return waiting * (toks / done) / (gen / secs)
    except Exception:
        pass
    return 5.0 * waiting


def estimate_request(body: dict, *, tenant_id=None) -> tuple:
    """Dry-run a chat-completions ``body``. Returns ``(status, payload)``:
    404 for an unknown model, 400 for a body without messages, else 200
    with the estimate (``resident: false`` and null fits for a model that
    is not loaded; ``media: true`` and null fits when the request carries
    images / audio / video)."""
    from . import server_bridge_vlm as serving

    t0 = time.perf_counter()
    model_id = body.get("model") if isinstance(body, dict) else None
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list) or not messages:
        return 400, {"error": {"message": "messages required",
                               "type": "invalid_request_error"}}
    try:
        path, spec = serving.resolve_request_model(model_id)
    except Exception as e:  # unknown / missing / no default
        return 404, {"error": {"message": str(e), "type": "not_found_error",
                               "model": model_id}}
    mid = getattr(spec, "id", None) or model_id
    out = {"model": mid, "resident": False, "media": False,
           "prompt_tokens": None, "prompt_chars": None, "max_tokens": None,
           "warm_tokens": 0, "cache_tier": None,
           "need_bytes": None, "need_prompt_bytes": None,
           "avail_now_bytes": None, "avail_drained_bytes": None,
           "fits_now": None, "fits_drained": None, "context_ok": None,
           "context_limit": None, "context_limit_source": None, "est_ttft_s": None,
           "band": None, "waiting": None, "in_flight": None,
           "decode_batch": None, "estimate_ms": None}
    try:
        from .governor import governor_stats
        from .queue_cap import concurrency_stats

        out["band"] = governor_stats().get("band")
        conc = concurrency_stats()
        out["waiting"], out["in_flight"] = conc.get("waiting"), conc.get("in_flight")
        out["decode_batch"] = conc.get("decode_batch")
    except Exception:
        pass

    pkg = importlib.import_module("mlx_vlm.server")
    pool = getattr(pkg, "_kq_residency_pool", None)
    entry = pool.resident_entry(path) if pool is not None else None
    if entry is None or entry.response_generator is None:
        out["hint"] = "model not resident; the real request loads it first"
        out["estimate_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return 200, out
    out["resident"] = True

    rg = entry.response_generator
    mc = entry.model_cache or {}
    model, processor, config = mc.get("model"), mc.get("processor"), mc.get("config")

    msgs, media = _normalize_messages(messages)
    out["media"] = media > 0
    try:
        tok = serving.set_active_spec(spec)    # the gen-args seam reads it
    except Exception:
        tok = None
    try:
        return _estimate_bound(body, out, t0, path, pkg, rg, model, processor,
                               config, entry, msgs, media, tenant_id)
    finally:
        if tok is not None:
            try:
                serving.reset_active_spec(tok)
            except Exception:
                pass


def _estimate_bound(body, out, t0, path, pkg, rg, model, processor, config,
                    entry, msgs, media, tenant_id):
    """The resident half of :func:`estimate_request`, run with the request
    spec bound to the serving seam (the caller resets it)."""
    try:
        from mlx_vlm.server import openai as _oa
        from mlx_vlm.server.app import _build_gen_args
        from mlx_vlm.server.schemas import ChatRequest

        req = ChatRequest(**body)
        msgs, tools, tool_choice = _oa._prepare_chat_tool_choice(
            msgs, body.get("tools"), body.get("tool_choice"))
        gen_args = _build_gen_args(req, processor, tenant_id=tenant_id)
        template_kwargs = gen_args.to_template_kwargs()
        if tool_choice is not None:
            template_kwargs["tool_choice"] = tool_choice
        prompt = pkg.apply_chat_template(processor, config, msgs, num_images=0,
                                         tools=tools or None, **template_kwargs)
        pinned = int(getattr(gen_args, "max_tokens", 0) or 0) \
            if "max_tokens" in body else 0
    except Exception as e:
        _log.debug("estimate: prompt render failed", exc_info=True)
        return 400, {"error": {"message": f"cannot render prompt: {e}",
                               "type": "invalid_request_error"}}
    out["prompt_chars"] = len(prompt) if isinstance(prompt, str) else None
    out["max_tokens"] = pinned or None

    if media:
        out["hint"] = "media requests are not estimated (text prompt only)"
        out["estimate_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return 200, out

    try:
        from mlx_vlm.server.generation import _count_prompt_tokens

        raw = rg._preprocess_request(prompt)
        tokens = int(_count_prompt_tokens(raw))
        out["prompt_tokens"] = tokens
        ids_arr = raw.get("input_ids") if isinstance(raw, dict) else None
        ids = [int(t) for t in ids_arr.reshape(-1).tolist()] if ids_arr is not None else []
    except Exception:
        _log.debug("estimate: tokenize failed", exc_info=True)
        tokens, ids = 0, []

    # configured context limit (the same check the real request runs)
    try:
        from mlx_vlm.server.generation import (PromptTooLongError,
                                               _check_configured_context_budget,
                                               get_configured_context_limit)

        limit = get_configured_context_limit()
        if limit is not None:
            out["context_limit"], out["context_limit_source"] = int(limit), "configured"
            try:
                _check_configured_context_budget(tokens, pinned)
                out["context_ok"] = True
            except PromptTooLongError:
                out["context_ok"] = False
        else:
            # nothing configured: the server refuses nothing, so judge
            # against the GGUF's trained context (what the model was built
            # for), and say which limit the verdict used
            from . import capacity as _capacity

            trained = _capacity.trained_context_length(path)
            if trained:
                out["context_limit"], out["context_limit_source"] = int(trained), "trained"
                out["context_ok"] = bool(tokens + max(0, pinned) <= int(trained))
    except Exception:
        pass

    # warm prefix
    try:
        extra_hash = 0
        if getattr(rg, "apc_mode", None) is not None:
            from mlx_vlm import apc as _apc

            extra_hash = _apc.semantic_extra_hash(
                tenant=tenant_id, image_hash=0, media={},
                model=getattr(model, "language_model", model),
                processor=processor)
        warm, tier = _warm_tokens(entry.apc_manager, ids, extra_hash, model)
        out["warm_tokens"], out["cache_tier"] = warm, tier
    except Exception:
        _log.debug("estimate: apc probe failed", exc_info=True)

    # memory, exactly as the preflight prices it
    try:
        from .mem_preflight import (_need_bytes, available_drained_bytes,
                                    kv_layer_costs)
        from .prefill_decay import headroom_bytes

        bits = getattr(rg, "kv_bits", None)
        bpe = bits / 8.0 if isinstance(bits, int) and bits > 0 else 2.0
        costs = kv_layer_costs(model, bpe)
        if costs and tokens > 0:
            need_prompt = _need_bytes(model, costs, tokens)
            need = _need_bytes(model, costs, tokens, pinned) if pinned else need_prompt
            out["need_prompt_bytes"] = int(need_prompt)
            out["need_bytes"] = int(need)
            drained = available_drained_bytes()
            if drained is not None:
                out["avail_drained_bytes"] = int(drained)
                out["fits_drained"] = bool(need <= drained)
            now = headroom_bytes()
            if now is not None:
                out["avail_now_bytes"] = int(now)
                out["fits_now"] = bool(need <= now)
    except Exception:
        _log.debug("estimate: memory pricing failed", exc_info=True)

    # ETA: queue drain + cold-suffix prefill at the recent prefill rate
    try:
        runtime = importlib.import_module("mlx_vlm.server.runtime").runtime
        metrics = getattr(runtime, "metrics", None)
        pre = _recent_rates(metrics)["prefill_tok_s"]
        if tokens > 0 and pre:
            suffix = max(0, tokens - int(out["warm_tokens"] or 0))
            eta = _queue_wait_s(metrics, out.get("waiting") or 0) + suffix / pre
            out["est_ttft_s"] = round(eta, 2)
    except Exception:
        pass
    out["estimate_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return 200, out
