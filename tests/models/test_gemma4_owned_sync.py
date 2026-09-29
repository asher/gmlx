"""The owned Gemma 4 text model takes no host sync to build its masks or
to rotate by an int offset.

Stock ``_make_masks`` probes the sliding-layer cache offset with
``.item()`` on every chunk wider than one token, a host sync once per
verify round and once per batched prefill chunk. Stock attention wraps
the cache offset in ``mx.array`` on every layer, a device upload per
layer per step. The owned copy compares an int offset on the host,
assumes a prefix for an array offset, and passes an int offset to rope
as is. An array offset is still copied, since ``update_and_fetch``
advances ``cache.offset`` in place between the key rope and the query
rope.

Each owned test runs in process. Each stock half runs in a child that
applies no loader install (gemma4_probes.py), so it sees the mlx-vlm
0.7.3 bodies, and shows what the owned copy removes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from gmlx.models.gemma4 import language as owned

import gemma4_probes as probes

DRIVER = Path(__file__).with_name("gemma4_probes.py")


@pytest.fixture(scope="module")
def stock() -> dict:
    r = subprocess.run([sys.executable, str(DRIVER), "--cpu"],
                       capture_output=True, text=True, env=dict(os.environ),
                       timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout)


def test_cache_has_prefix_decision():
    class C:
        pass

    c = C()
    c.offset = 0
    assert owned._cache_has_prefix(c) is False
    c.offset = 7
    assert owned._cache_has_prefix(c) is True
    c.offset = mx.array([0, 0])  # never probed: a prefix is assumed
    assert owned._cache_has_prefix(c) is True
    c.offset = mx.array(3)
    assert owned._cache_has_prefix(c) is True


def test_owned_masks_take_no_host_sync():
    assert probes.mask_syncs(owned) == {"int": 0, "array": 0}


def test_stock_masks_sync_once_per_chunk(stock):
    assert stock["mask_syncs"] == {"int": 1, "array": 1}


def test_owned_mask_decisions_match_stock_int_offsets(stock):
    assert stock["mask_decisions"] == {
        "prefill": True, "verify": True, "decode": True}


def test_owned_rope_takes_an_int_offset_as_is():
    got = probes.rope_offsets(owned)
    assert got["int"] == [["int", 9], ["int", 9]]
    assert got["array_copied"] == [True, True]


def test_stock_rope_takes_an_int_offset_as_an_array(stock):
    got = stock["rope_offsets"]
    assert got["int"] == [["array", 9], ["array", 9]]
    assert got["array_copied"] == [True, True]
