"""The MLX cache clear after a prefill chunk, held until the GPU has work.

A chunk's scratch buffers sit in MLX's cache when the chunk ends, and
releasing them takes the host several milliseconds. Cleared at the chunk's
end, that time passes with the GPU idle. Deferred, the clear runs right
after the next forward's first submit (a next chunk), or at the end of the
first decode round (the last chunk), while the GPU runs what was submitted.
The buffers are held for those few milliseconds more and no longer.

``GMLX_SPEC_CLEAR_AFTER_SUBMIT=0`` clears at the chunk's end.
"""
import mlx.core as mx

from gmlx.envflags import env_bool

_ENABLED = None
_pending = False


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = env_bool("GMLX_SPEC_CLEAR_AFTER_SUBMIT", True)
    return _ENABLED


def clear_now() -> None:
    """Book what the process holds, clear the cache, and settle a deferred
    clear."""
    global _pending
    from gmlx.serve.capacity import note_own_before_clear

    _pending = False
    note_own_before_clear()
    mx.clear_cache()


def at_chunk_end() -> None:
    """A prefill chunk has ended: clear now, or leave it to the next
    submit."""
    global _pending
    if enabled():
        _pending = True
    else:
        clear_now()


def settled() -> None:
    """The cache was cleared by other code: nothing is left to run."""
    global _pending
    _pending = False


def pending() -> bool:
    return _pending


def run() -> None:
    """Run a deferred clear, if one is waiting."""
    if _pending:
        clear_now()
