"""Streamed expert stacks drop their GPU mapping once the feeders own them."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.load.transforms import split_fused_gate_up_kquant
from gmlx.stream.feeder_common import FeederRange, feeder_ranges
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


def _layer(mod, path):
    """Layer 0 as the prefill feeder holds it."""
    return {0: {r.kind: (mod, r.path, r.off, r.length * r.n_exp, r)
                for r in feeder_ranges(expert_offset_map(path)[0])}}


def test_placeholder_keeps_shape_dtype_and_nbytes_over_one_row():
    # 32 MiB over a 512 KiB row: a small buffer MLX allocates on its own
    # during the call stays far under the bound.
    w = mx.ones((64, 512, 512), dtype=mx.float16)
    mx.eval(w)
    before = mx.get_active_memory()
    p = placeholder(w)
    assert p.shape == w.shape and p.dtype == w.dtype and p.nbytes == w.nbytes
    assert mx.get_active_memory() - before < w.nbytes // 4
    assert float(mx.abs(p).sum()) == 0.0


def test_unmap_and_remap_for_a_call(stack_gguf):
    path, raw = stack_gguf
    mod = _Mod(kq.load_gguf(path, True)[0])
    for k in KINDS:
        assert np.array_equal(np.array(_weights(mod)[k]), raw[k])
    n_layers, nbytes = unmap_stacks(_layer(mod, path))
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
    rng = FeederRange(path, 7, 16, 16, 1, "gate", 7, None)
    with pytest.raises(RuntimeError, match="no tensor at offset"):
        unmap_stacks({0: {"gate": (mod, path, 7, 16, rng)}})


def test_unmap_and_remap_a_fused_gate_up_layer(tmp_path):
    rng = np.random.default_rng(4)
    gate_up = rng.standard_normal((E, 2 * N, K)).astype(np.float16)
    down = rng.standard_normal((E, N, K)).astype(np.float16)
    path = str(tmp_path / "f.gguf")
    w = gguf.GGUFWriter(path, "gemma4")
    w.add_tensor("blk.0.ffn_gate_up_exps.weight", gate_up)
    w.add_tensor("blk.0.ffn_down_exps.weight", down)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    raw = {"gate": gate_up[:, :N], "up": gate_up[:, N:], "down": down}

    # The module holds the loader's split, as a Gemma 4 load does.
    arrays = kq.load_gguf(path, True)[0]
    gate, up = split_fused_gate_up_kquant(
        arrays["blk.0.ffn_gate_up_exps.weight"])
    mod = _Mod({"blk.0.ffn_gate_exps.weight": gate,
                "blk.0.ffn_up_exps.weight": up,
                "blk.0.ffn_down_exps.weight":
                    arrays["blk.0.ffn_down_exps.weight"]})
    del arrays

    assert unmap_stacks(_layer(mod, path)) == (1, 3 * E * N * K * 2)
    files = getattr(mod, STAMP)
    assert files["gate"][1] == files["up"][1] == "blk.0.ffn_gate_up_exps.weight"
    assert (files["gate"][3], files["up"][3], files["down"][3]) == (0, 1, None)

    # The fallback sees each half's bytes; the halves are views, so only
    # the values are compared.
    with stacks_remapped(mod):
        for k, w in _weights(mod).items():
            assert w.shape == (E, N, K)
            assert np.array_equal(np.array(w), raw[k])
    for w in _weights(mod).values():
        assert float(mx.abs(w).sum()) == 0.0
