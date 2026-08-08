"""Ragged right-padded prefill mask fix for ArraysCache state models.

Both upstream batch engines right-pad ragged prefill rows and declare the
layout with ``ArraysCache.prepare(lengths=...)``: mlx-lm's ``BatchGenerator``
for every ragged multi-prompt insert, mlx-vlm's ``PromptProcessingBatch`` for
mixed warm/cold APC prefill. But by then ``ArraysCache.merge`` (all-empty
branch) or the engine's ``_make_cache`` has already seeded
``left_padding = [0] * B``, ``prepare`` leaves it in place, and ``make_mask``
prefers ``left_padding`` over ``lengths`` - producing an all-True mask for a
right-padded batch. The pad garbage then enters the conv window and recurrent
state of every short row; attention layers are unaffected
(``BatchKVCache.prepare`` folds right padding into its offsets), so only
state-cache archs corrupt (falcon-h1, granite/nemotron mamba2 hybrids, pure
mamba2): falcon-h1's shortest row diverged at its first decode token with a
rank-14807 token at 13 nats.

The fix clears ``left_padding`` when ``prepare`` declares a lengths-based
layout: ``lengths`` is the mask source for right padding, and ``finalize()``
already resets both fields after the padded prefill window. Patched on both
class origins (distinct classes on mlx-vlm >= 0.6.4). Verified live: with
this patch falcon-h1 ragged batched decode is token-identical to
single-stream on every row.
"""

from __future__ import annotations

_PATCH_FLAG = "_kq_lengths_clears_left_padding"

_installed = False


def install_arrays_cache_fix() -> bool:
    """Make ``ArraysCache.prepare(lengths=...)`` authoritative for the mask.

    Idempotent; patches every loaded origin of the class.
    """
    global _installed
    if _installed:
        return True
    from .cache_compat import cache_types

    for cls in cache_types("ArraysCache"):
        orig = cls.prepare
        if getattr(orig, _PATCH_FLAG, False):
            continue

        def prepare(self, lengths=None, _orig=orig, **kwargs):
            _orig(self, lengths=lengths, **kwargs)
            if lengths is not None:
                self.left_padding = None

        prepare.__dict__[_PATCH_FLAG] = True
        cls.prepare = prepare
    _installed = True
    return True
