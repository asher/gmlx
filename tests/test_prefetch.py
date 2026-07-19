#!/usr/bin/env python3
"""ExpertPrefetcher fd lifetime + F_RDADVISE range packing. Pure file logic -
the advisory fcntl is faked, no model, no GPU."""
from __future__ import annotations

import struct

from gmlx import prefetch as pf


def _make(tmp_path, offsets_for_file):
    p = tmp_path / "w.gguf"
    p.write_bytes(b"x")
    return pf.ExpertPrefetcher({0: [(str(p), off, n) for off, n in
                                    offsets_for_file]})


def test_advise_chunks_ranges_past_int32(tmp_path, monkeypatch):
    # radvisory.ra_count is a signed int32: a >=2 GB tensor must be advised in
    # chunks, not dropped via an uncaught struct.error in the worker thread.
    calls = []
    monkeypatch.setattr(pf.fcntl, "fcntl",
                        lambda fd, op, arg: calls.append(struct.unpack("=qi4x", arg)))
    three_gb = 3 * (1 << 30)
    fetcher = _make(tmp_path, [(64, three_gb)])
    try:
        fetcher._advise(0)
    finally:
        fetcher.close()
    assert [n for _, n in calls] == [1 << 30] * 3
    assert calls[0][0] == 64 and calls[1][0] == 64 + (1 << 30)
    assert sum(n for _, n in calls) == three_gb
    assert all(n < (1 << 31) for _, n in calls)


def test_expert_offset_map_matches_ggufreader(tmp_path):
    # expert_offset_map builds from headerscan; a GGUFReader-derived reference
    # is the parity oracle (absolute offsets, nbytes, expert count, kind).
    import numpy as np
    from gguf import GGUFReader, GGUFWriter

    p = tmp_path / "m.gguf"
    w = GGUFWriter(str(p), "llama")
    w.add_uint32("llama.block_count", 2)
    for li in range(2):
        for kind in ("gate", "up", "down"):
            w.add_tensor(f"blk.{li}.ffn_{kind}_exps.weight",
                         np.zeros((4, 8, 16), dtype=np.float16))
    w.add_tensor("blk.0.ffn_gate_shexp.weight",       # decoy: shared expert
                 np.zeros((8, 16), dtype=np.float16))
    w.add_tensor("token_embd.weight",                 # decoy: 2-dim
                 np.zeros((32, 16), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    ref: dict = {}
    for t in GGUFReader(str(p)).tensors:
        m = pf._EXPS_RE.fullmatch(t.name)
        if m:
            n_exp = int(t.shape[-1]) if len(t.shape) == 3 else 0
            if n_exp and int(t.n_bytes) % n_exp:
                n_exp = 0
            ref.setdefault(int(m.group(1)), []).append(
                (str(p), int(t.data_offset), int(t.n_bytes), n_exp,
                 m.group(2)))

    got = pf.expert_offset_map(str(p))
    assert {k: sorted(v) for k, v in got.items()} == \
           {k: sorted(v) for k, v in ref.items()}
    assert set(got) == {0, 1} and all(len(v) == 3 for v in got.values())
    assert all(r[3] == 4 for v in got.values() for r in v)


def test_close_is_idempotent(tmp_path):
    fetcher = _make(tmp_path, [(0, 128)])
    fd = next(iter(fetcher._fds.values()))
    fetcher.close()
    fetcher.close()          # second close (teardown + GC) must not raise
    import os
    import pytest
    with pytest.raises(OSError):
        os.fstat(fd)         # fd really was closed
