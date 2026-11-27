# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""kq's cache of KQ_* lever reads, for the models that record kq graphs.

Direct replay of a record needs the cache on. Each call that turns it on
also drops the Metal calls every record of the process has learned, so
the models turn it on through ``lever_cache_on``, once a process.
"""

# Whether a model turned the cache on.
_ON = [False]


def lever_cache_on() -> None:
    """Turn kq's lever cache on, unless a model already did."""
    if not _ON[0]:
        import mlx_kquant as kq

        kq.env_cache(True)
        _ON[0] = True
