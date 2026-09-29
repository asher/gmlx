"""Recurrent-layer cache for the vendored qwen3.5 text model.

mlx-vlm 0.7.3's ``ArraysCache`` reads ``left_padding`` and ``lengths``
through an advance offset, so every read after ``advance`` returns a new
array. The vendored model memoizes host copies of both keyed on array
identity, and would miss that memo on every read. ``Qwen35ArraysCache``
rebinds both arrays at ``advance``, as mlx-vlm 0.6.15 did, and keeps the
0.7.3 base for everything else: the APC capability (resolved through the
MRO), prefix snapshots, merge and extract, and the speculation
transaction, whose abort restores the arrays it saved because they are
never mutated in place.
"""

from gmlx.cache.kv_cache import ArraysCache


class Qwen35ArraysCache(ArraysCache):
    def _settle(self):
        if self._left_padding_advance:
            self._left_padding = self._left_padding - self._left_padding_advance
            self._left_padding_advance = 0
        if self._lengths_advance:
            self._lengths = self._lengths - self._lengths_advance
            self._lengths_advance = 0

    def advance(self, N):
        super().advance(N)
        self._settle()

    def _restore_speculative_metadata(self, lengths, initial_metadata):
        super()._restore_speculative_metadata(lengths, initial_metadata)
        self._settle()
