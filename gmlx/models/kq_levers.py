# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""kq's cache of KQ_* lever reads, for the models that record kq graphs.

Direct replay of a record needs the cache on. Each call that turns it on
also drops the Metal calls every record of the process has learned, so
the models turn it on through ``lever_cache_on``, once a process.

``RecordTable`` holds a layer's records and releases them when it goes.
"""

# Whether a model turned the cache on.
_ON = [False]


def lever_cache_on() -> None:
    """Turn kq's lever cache on, unless a model already did."""
    if not _ON[0]:
        import mlx_kquant as kq

        kq.env_cache(True)
        _ON[0] = True


class RecordTable(dict):
    """A layer's kq records by layout. ``release`` releases the records
    the table holds and empties it. It also runs when the table goes (its
    layer let go, or the table replaced), since a record holds the
    layer's arrays as constants: without it, a dropped model's records
    keep those arrays in kq's table for the life of the process."""

    __slots__ = ("release",)

    def __init__(self, release):
        super().__init__()
        self.release = release

    def __del__(self):
        try:
            self.release(self)
        except Exception:  # noqa: S110 (kq may be gone at exit)
            pass
