"""A model that a module or session fixture shares keeps its feeder pools
for every test that uses it. The suite's teardown (tests/conftest.py)
shuts down only the pools a test created. A stopped pool would leave the
next test's prefill waiting out the stage timeout and its demand reads
quarantined as wedged. The fixture's patches live in this file alone,
since a module fixture holds them until the module ends."""

from __future__ import annotations

import numpy as np
import pytest

from test_decode_feeder import _fake_arena_alloc, _make_feeder, _make_fixture


@pytest.fixture(scope="module")
def shared_feeders(tmp_path_factory):
    import mlx_kquant as kq

    import gmlx.stream.prefill_feeder as pfm

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kq, "arena_alloc", _fake_arena_alloc, raising=False)
        mp.setattr(pfm, "lock_pages", lambda mv: None)
        mp.setattr(pfm, "_STAGE_TIMEOUT_S", 3.0)
        mp.setenv("GMLX_DECODE_READ_TIMEOUT", "3")
        offsets, modules = _make_fixture(tmp_path_factory.mktemp("pf"), 2)
        prefill = pfm.PrefillFeeder(offsets, modules)
        decode, _ = _make_feeder(
            mp, tmp_path_factory.mktemp("df"), slots_per_layer=4)
        yield prefill, modules, decode
        prefill.close()
        decode.close()


@pytest.mark.parametrize("run", [1, 2])
def test_a_shared_model_keeps_its_pools_for_the_next_test(
        shared_feeders, run):
    prefill, modules, decode = shared_feeders
    for li in (0, 1):
        with prefill.prefill_call(modules[li][0], li):
            pass
    assert not prefill._wedged
    # A new expert each run misses the arena and reads on the read pool.
    assert decode.stage(0, np.array([[run]], dtype=np.uint32)) is not None
    assert decode._wedges == 0
