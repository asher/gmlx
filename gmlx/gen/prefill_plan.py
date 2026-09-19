"""Chunk plan for a full-position teacher pass: how many tokens the trunk
runs at once and how many positions the reduction head takes per
sub-chunk.

A pass that scores every position of long rows has two memory limits. The
trunk is bounded by attention activations and the room the model leaves,
which gmlx already measures for chunked prefill, and for a streamed
mixture-of-experts teacher a wider trunk also divides the expert bytes
read per token. The head is bounded by the full-width logits and their
float32 reductions, at a fixed number of bytes per vocabulary element,
and takes the largest halving tier of 4096 positions that fits under a
cap in decimal gigabytes.
"""
from __future__ import annotations

GB = 1e9
# Live bytes per vocabulary element in the head's reduction: float32
# log-softmax, bf16 logits, a full-width index and one float32 temporary,
# budgeted with a margin; ``--floor`` keeps a second temporary alive.
BYTES_PER_V_ELEMENT = 16
BYTES_PER_V_ELEMENT_FLOOR = 20


def head_step(V: int, cap_gb: float, bytes_per_v: float, max_step: int = 4096) -> int:
    """The largest halving tier of ``max_step`` with
    ``step * V * bytes_per_v <= cap_gb * 1e9``, never below 1."""
    cap = cap_gb * GB
    step = max_step
    while step > 1 and step * V * bytes_per_v > cap:
        step //= 2
    return step


def prefill_plan(model, *, V: int, cap_gb: float = 4.0, floor: bool = False,
                 streaming: bool | None = None, requested_trunk: int | None = None,
                 target_streaming_trunk: int = 8192, dense_trunk: int = 512,
                 bytes_per_v: float | None = None, config: dict | None = None) -> dict:
    """The trunk chunk ``trunk`` (tokens, a multiple of 512) and the head
    sub-chunk ``step`` for one teacher, with the inputs to both.

    With ``model`` None the plan is the arithmetic alone. Otherwise the
    trunk starts from gmlx's prefill step for the model, is capped by the
    chunked-prefill depth ceiling for the row length, and never widens past
    ``dense_trunk`` for a resident teacher or ``target_streaming_trunk`` for
    a streamed one. ``requested_trunk`` overrides that target. Headroom is
    sampled after the buffer cache is cleared and reported, not enforced.
    ``config`` supplies the attention head count (under ``text_config`` when
    present) and defaults to the model's own ``args``."""
    if bytes_per_v is None:
        bytes_per_v = BYTES_PER_V_ELEMENT_FLOOR if floor else BYTES_PER_V_ELEMENT
    step = head_step(V, cap_gb, bytes_per_v)
    plan = {"V": V, "cap_gb": cap_gb, "bytes_per_v": bytes_per_v, "step": step,
            "streaming": bool(streaming), "trunk_base": None, "trunk": None,
            "headroom_gb": None}
    if model is None:
        plan["trunk"] = requested_trunk or (target_streaming_trunk if streaming else dense_trunk)
        return plan
    import mlx.core as mx

    from gmlx.gen import prefill_decay
    from gmlx.stream.expert_streaming import _resolve_prefill_step, moe_streaming_active

    if streaming is None:
        streaming = bool(moe_streaming_active(model))
    plan["streaming"] = streaming
    base = _resolve_prefill_step(model, requested_trunk)[0] or 2048
    plan["trunk_base"] = int(base)
    mx.clear_cache()
    hb = prefill_decay.headroom_bytes()
    plan["headroom_gb"] = None if hb is None else hb / GB
    cfg = config
    if cfg is None:
        cfg = getattr(model, "args", None)
        cfg = getattr(cfg, "__dict__", cfg) if cfg is not None else {}
    cfg = cfg.get("text_config", cfg) if isinstance(cfg, dict) else {}
    heads = int(cfg.get("num_attention_heads", 32)) if isinstance(cfg, dict) else 32
    target = requested_trunk or (target_streaming_trunk if streaming else dense_trunk)
    t = min(int(target), int(base)) if streaming else int(target)
    t = min(t, int(prefill_decay.decayed_step(max(t, 64), 0, heads)))
    plan["trunk"] = max(512, (t // 512) * 512)
    return plan
