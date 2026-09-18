"""Streamed expert stacks drop their GPU mapping once the feeders own them."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.stream.prefetch import expert_offset_map
from gmlx.stream.stack_unmap import (
    STAMP,
    placeholder,
    stacks_remapped,
    unmap_stacks,
)

gguf = pytest.importorskip("gguf")
kq = pytest.importorskip("mlx_kquant")

E, N, K = 4, 8, 16
KINDS = ("gate", "up", "down")


@pytest.fixture
def stack_gguf(tmp_path):
    """One MoE layer's three F16 expert stacks."""
    rng = np.random.default_rng(3)
    raw = {k: rng.standard_normal((E, N, K)).astype(np.float16) for k in KINDS}
    path = str(tmp_path / "s.gguf")
    w = gguf.GGUFWriter(path, "llama")
    for k, a in raw.items():
        w.add_tensor(f"blk.0.ffn_{k}_exps.weight", a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, raw


class _Proj:
    def __init__(self, w):
        self.weight = w


class _Mod:
    def __init__(self, arrays):
        for k in KINDS:
            setattr(self, f"{k}_proj", _Proj(arrays[f"blk.0.ffn_{k}_exps.weight"]))


def _weights(mod):
    return {k: getattr(mod, f"{k}_proj").weight for k in KINDS}


def test_placeholder_keeps_shape_dtype_and_nbytes_over_one_row():
    w = mx.ones((E, N, K), dtype=mx.float16)
    mx.eval(w)
    before = mx.get_active_memory()
    p = placeholder(w)
    assert p.shape == w.shape and p.dtype == w.dtype and p.nbytes == w.nbytes
    assert mx.get_active_memory() - before < w.nbytes
    assert float(mx.abs(p).sum()) == 0.0


def test_unmap_and_remap_for_a_call(stack_gguf):
    path, raw = stack_gguf
    mod = _Mod(kq.load_gguf(path, True)[0])
    for k in KINDS:
        assert np.array_equal(np.array(_weights(mod)[k]), raw[k])
    ranges = expert_offset_map(path)[0]
    layers = {0: {kind: (mod, p, off, nb) for p, off, nb, _n, kind in ranges}}

    n_layers, nbytes = unmap_stacks(layers)
    assert (n_layers, nbytes) == (1, 3 * E * N * K * 2)
    assert set(getattr(mod, STAMP)) == set(KINDS)
    for k, w in _weights(mod).items():
        assert w.shape == (E, N, K) and w.dtype == mx.float16
        assert w.nbytes == raw[k].nbytes
        assert float(mx.abs(w).sum()) == 0.0

    # The no-feeder fallback sees the file's bytes again for its call.
    with stacks_remapped(mod):
        for k, w in _weights(mod).items():
            assert np.array_equal(np.array(w), raw[k])
    for w in _weights(mod).values():
        assert float(mx.abs(w).sum()) == 0.0

    # A module whose stacks were never unmapped: a no-op.
    with stacks_remapped(object()):
        pass


def test_unmap_refuses_an_offset_with_no_tensor(stack_gguf):
    path, _raw = stack_gguf
    mod = _Mod(kq.load_gguf(path, True)[0])
    with pytest.raises(RuntimeError, match="no tensor at offset"):
        unmap_stacks({0: {"gate": (mod, path, 7, 16)}})
