"""Startup fit check for the one-shot tools (run, chat, bench).

Weights plus projected KV plus the prefill score transient against the
device working set, computed from the GGUF header alone, before the
multi-GB load starts. A configuration that cannot fit refuses with the
numbers (what it needs, what the box has, and the largest context that
would fit) instead of dying mid-load or mid-run on a raw allocator
message. The run-loop guard extends the same numbers to a catchable
allocator error mid-run.

One formula, not two: KV pricing comes from mem_preflight (the serve
preflight's own admit-side model), the config comes from the loader's
config synthesis, and weights are the on-disk shard bytes the
zero-copy load will map. The transient term is the stock prefill score
peak (heads x chunk x (context + chunk) x 2 bytes), the measured
serve-peak class.

Estimates err on the permit side: unknown geometry, an unsupported
arch, or any estimator failure skips the check silently. Expert
streaming skips too (resident bytes are policy there, not file size).

Knobs:
    GMLX_TOOL_PREFLIGHT=0   kill switch
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

_log = logging.getLogger(__name__)

GB = 1e9
_SCORE_CHUNK = 2048  # stock prefill step; decay only shrinks it


def _enabled() -> bool:
    return os.environ.get("GMLX_TOOL_PREFLIGHT", "1") != "0"


def _shards(gguf_path: str) -> list[str]:
    from gmlx.load.preflight import shard_names

    d, base = os.path.split(gguf_path)
    try:
        names = shard_names(base)
    except ValueError:
        return [gguf_path]
    paths = [os.path.join(d, n) for n in names]
    return paths if all(os.path.exists(p) for p in paths) else [gguf_path]


def _synth_config(gguf_path: str) -> dict | None:
    """The config synthesized from the header: metadata from the first
    shard, tensor shapes from every shard (a synth that derives MLA head
    dims from a projection tensor must see the shard that holds it)."""
    from gmlx.load.config_synth import synthesize_config
    from gmlx.load.headerscan import scan_gguf

    kv = None
    shapes = {}
    for p in _shards(gguf_path):
        scan = scan_gguf(p, include_tensors=True)
        if kv is None:
            kv = scan.kv
        shapes.update({t.name: list(t.shape) for t in scan.tensors})
    return synthesize_config(kv, shapes)


def working_set_bytes() -> float | None:
    import mlx.core as mx

    try:
        ws = float(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        return None
    return ws if ws > 0 else None


def _kv_costs(cfg: dict):
    from gmlx.serve.mem_preflight import kv_layer_costs

    return kv_layer_costs(SimpleNamespace(config=cfg))


def _need_at(weights: float, costs, heads: int | None, ctx: int) -> float:
    from gmlx.serve.mem_preflight import prompt_kv_bytes

    kv = prompt_kv_bytes(costs, ctx)
    transient = 0.0
    if heads:
        transient = heads * _SCORE_CHUNK * (ctx + _SCORE_CHUNK) * 2.0
    return weights + kv + transient


def estimate(gguf_path: str, ctx_tokens: int | None = None) -> dict | None:
    """Fit estimate for serving ``ctx_tokens`` of context (default: the
    model's trained context) on this box, or None when any input cannot
    be read (the caller must treat None as permit)."""
    try:
        shards = _shards(gguf_path)
        weights = float(sum(os.path.getsize(p) for p in shards))
        cfg = _synth_config(shards[0])
        costs = _kv_costs(cfg)
        ws = working_set_bytes()
        if not (cfg and costs and ws):
            return None
        from gmlx.serve.mem_preflight import _get, _lm_config

        heads = _get(_lm_config(SimpleNamespace(config=cfg)),
                     "num_attention_heads")
        heads = heads if isinstance(heads, int) and heads > 0 else None
        if ctx_tokens:
            ctx = int(ctx_tokens)
        else:
            # No pinned context: price a floor working session, not the
            # trained maximum (refuse the definitely-unfittable only).
            trained = cfg.get("max_position_embeddings")
            ctx = min(4096, trained) if isinstance(trained, int) \
                and trained > 0 else 4096
        need = _need_at(weights, costs, heads, ctx)
        lo, hi = 0, 1 << 22
        while lo < hi:  # largest context that fits, admit-side
            mid = (lo + hi + 1) // 2
            if _need_at(weights, costs, heads, mid) <= ws:
                lo = mid
            else:
                hi = mid - 1
        return {
            "path": gguf_path,
            "weights": weights,
            "ctx": ctx,
            "need": need,
            "working_set": ws,
            "fits": need <= ws,
            "largest_fit_ctx": lo,
        }
    except Exception:
        _log.debug("tool preflight estimate skipped", exc_info=True)
        return None


def refusal_text(est: dict) -> str:
    largest = est["largest_fit_ctx"]
    fit = (f"largest fitting context is ~{largest}"
           if largest > 0 else
           "the weights alone exceed the working set")
    return (f"cannot fit: {est['ctx']}-token context needs "
            f"~{est['need'] / GB:.1f} GB "
            f"(weights {est['weights'] / GB:.1f} GB), this box's GPU "
            f"working set is {est['working_set'] / GB:.1f} GB; {fit}. "
            "For MoE models --stream-experts serves the experts from "
            "disk; GMLX_TOOL_PREFLIGHT=0 overrides.")


def check_or_exit(gguf_path: str, ctx_tokens: int | None = None,
                  streaming: bool = False):
    """Refuse (SystemExit 2, numbers on stderr) when the configuration
    definitively cannot fit; otherwise return the estimate (or None
    when skipped)."""
    if not _enabled() or streaming:
        return None
    est = estimate(gguf_path, ctx_tokens)
    if est is not None and not est["fits"]:
        print(f"error: {refusal_text(est)}", file=sys.stderr)
        raise SystemExit(2)
    return est


_ALLOC_MARKS = ("metal::malloc", "Insufficient Memory",
                "Command buffer execution failed", "kIOGPUCommand")


def guard_run(fn, est: dict | None = None):
    """Run ``fn()``; a catchable allocator or command-buffer error
    prints the same needs-versus-has numbers instead of a raw C++
    message and exits 2. Everything else propagates."""
    try:
        return fn()
    except RuntimeError as e:
        if not any(m in str(e) for m in _ALLOC_MARKS):
            raise
        import mlx.core as mx

        try:
            active = mx.get_active_memory() / GB
            cache = mx.get_cache_memory() / GB
            state = f"active {active:.1f} GB, cache {cache:.1f} GB"
        except Exception:
            state = "counters unavailable"
        ws = working_set_bytes()
        have = f"{ws / GB:.1f}" if ws else "?"
        need = (f"~{est['need'] / GB:.1f} GB projected for "
                f"{est['ctx']}-token context, " if est else "")
        print(f"error: out of GPU memory mid-run: {need}working set "
              f"{have} GB, {state}.\n  {str(e).splitlines()[0][:200]}",
              file=sys.stderr)
        raise SystemExit(2) from e
