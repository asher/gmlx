"""Recurrent-layer cache for the vendored qwen3.5 text model.

mlx-vlm 0.7.3's ``ArraysCache`` reads ``left_padding`` and ``lengths``
through an advance offset, so every read after ``advance`` returns a new
array. The vendored model memoizes host copies of both keyed on array
identity, and would miss that memo on every read. ``Qwen35ArraysCache``
rebinds both arrays at ``advance``, as mlx-vlm 0.6.15 did, and keeps the
0.7.3 base for everything else: the APC capability (resolved through the
MRO), prefix snapshots, merge and extract, and the speculation
transaction, whose abort restores the arrays it saved because they are
never mutated in place. The host-side bound that ``mask_excludes_nothing``
keeps moves along with each rebinding, the restore of a speculative commit
or abort included, so it is read once per batch.
"""

from gmlx.cache.kv_cache import ArraysCache


class Qwen35ArraysCache(ArraysCache):
    def _settle(self):
        if self._left_padding_advance:
            old, steps = self._left_padding, self._left_padding_advance
            self._left_padding = old - steps
            self._left_padding_advance = 0
            memo = getattr(self, "_left_padding_max", None)
            if memo is not None and memo[0] is old:
                self._left_padding_max = (self._left_padding, memo[1] - steps)
            self._record_settled(0, old, self._left_padding, steps)
        if self._lengths_advance:
            old, steps = self._lengths, self._lengths_advance
            self._lengths = old - steps
            self._lengths_advance = 0
            memo = getattr(self, "_lengths_min", None)
            if memo is not None and memo[0] is old:
                self._lengths_min = (self._lengths, memo[1] - steps)
            self._record_settled(2, old, self._lengths, steps)

    def _record_settled(self, slot, old, new, steps):
        """Keep, for the active speculative block, how far the array is
        past the one the block saved, so a bound first read inside the
        block still reaches the saved array when the block ends."""
        block = self._speculation
        if block is None:
            return
        settled = block["settled"]
        if block["initial_metadata"][slot] is old:
            settled[slot] = (new, steps)
        elif slot in settled and settled[slot][0] is old:
            settled[slot] = (new, settled[slot][1] + steps)
        else:
            settled.pop(slot, None)

    def advance(self, N):
        super().advance(N)
        self._settle()

    def _restore_speculative_metadata(self, lengths, initial_metadata):
        super()._restore_speculative_metadata(lengths, initial_metadata)
        self._settle()
