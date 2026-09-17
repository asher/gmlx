"""Arm the DeepSeek-V4.1 prefill tail.

Once the model knows where the prompt ends, the layers past the last
kv-source layer skip the prompt rows no later layer can reach through
its sliding window (``DeepseekV41Model._tail_plan``). The end is marked
on the first prompt-cache entry before each prefill; the model clears
it when a chunk reaches it.
"""
from __future__ import annotations

import contextlib

from gmlx.gen.prefill_decay import kv_depth

_ATTR = "_gmlx_prefill_end"
_FLAG = "_gmlx_prefill_tail_wrapped"


def tail_model(model) -> bool:
    """True for a DeepSeek-V4.1 model or container."""
    for m in (model, getattr(model, "language_model", None)):
        if m is not None and type(m).__module__.startswith(
            "gmlx.models.deepseek_v41"
        ):
            return True
    return False


def arm_prefill_tail(prompt_cache, n_prompt: int) -> int | None:
    """Mark the prompt end (current depth plus ``n_prompt`` rows) on the
    first cache entry. Returns the end, or None when nothing was armed."""
    if not prompt_cache or n_prompt <= 0:
        return None
    end = kv_depth(prompt_cache) + int(n_prompt)
    try:
        setattr(prompt_cache[0], _ATTR, end)
    except AttributeError:
        return None
    return end


def arm_batch(batch) -> int | None:
    """Arm a one-row PromptProcessingBatch for its next prompt_step: the
    prompt end, or the next APC checkpoint column when that comes first
    (the checkpoint state must hold every row)."""
    embeds = getattr(batch, "_inputs_embeds", None)
    cache = getattr(batch, "prompt_cache", None)
    if embeds is None or not cache or embeds.shape[0] != 1:
        return None
    if getattr(batch, "draft_model", None) is not None:
        return None
    remaining = int(embeds.shape[1])
    col = None
    next_col = getattr(batch, "_next_apc_checkpoint_column", None)
    if callable(next_col):
        with contextlib.suppress(Exception):
            col = next_col()
    if col is not None:
        done = int(getattr(batch, "_processed_prompt_columns", 0) or 0)
        remaining = min(remaining, max(0, int(col) - done))
    return arm_prefill_tail(cache, remaining)


def install_prefill_tail() -> bool:
    """Wrap PromptProcessingBatch.prompt_step to arm the tail before each
    chunk (serve path). Idempotent."""
    try:
        from mlx_vlm.generate.ar import PromptProcessingBatch
    except ImportError:
        return False
    if getattr(PromptProcessingBatch, _FLAG, False):
        return True
    orig_step = PromptProcessingBatch.prompt_step

    def _armed_prompt_step(self):
        arm_batch(self)
        return orig_step(self)

    PromptProcessingBatch.prompt_step = _armed_prompt_step
    setattr(PromptProcessingBatch, _FLAG, True)
    return True
