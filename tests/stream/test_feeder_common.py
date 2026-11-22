"""The feeders' view of a layer's expert ranges: plain stacks pass through,
and a fused gate_up stack splits into its gate and up halves."""
from __future__ import annotations

import numpy as np
import pytest

from gmlx.stream.feeder_common import FeederRange, feeder_ranges, kind_bytes

E = 8


def test_plain_ranges_pass_through():
    raw = [("p", 100, 800, E, "gate"), ("p", 900, 800, E, "up"),
           ("p", 1700, 400, E, "down")]
    assert feeder_ranges(raw) == [
        FeederRange("p", 100, 100, 100, E, "gate", 100, None),
        FeederRange("p", 900, 100, 100, E, "up", 900, None),
        FeederRange("p", 1700, 50, 50, E, "down", 1700, None),
    ]
    assert kind_bytes(raw) == {"gate": 800, "up": 800, "down": 400}


def test_a_fused_range_splits_at_each_experts_midpoint():
    raw = [("p", 100, 1600, E, "gate_up"), ("p", 1700, 400, E, "down")]
    assert feeder_ranges(raw) == [
        FeederRange("p", 100, 200, 100, E, "gate", 100, 0),
        FeederRange("p", 200, 200, 100, E, "up", 100, 1),
        FeederRange("p", 1700, 50, 50, E, "down", 1700, None),
    ]
    assert kind_bytes(raw) == {"gate": 800, "up": 800, "down": 400}


def test_layers_the_feeders_cannot_cover():
    down = ("p", 1700, 400, E, "down")
    # No expert count: no per-expert offsets, but the size is still known.
    flat = [("p", 100, 1600, 0, "gate_up"), ("p", 1700, 400, 0, "down")]
    assert feeder_ranges(flat) is None
    assert kind_bytes(flat) == {"gate": 800, "up": 800, "down": 400}
    # Mixed expert counts.
    assert feeder_ranges([("p", 100, 1600, E, "gate_up"),
                          ("p", 1700, 400, E // 2, "down")]) is None
    # A fused stack beside a separate gate, and a layer missing a kind.
    both = [("p", 100, 1600, E, "gate_up"), ("p", 0, 800, E, "gate"), down]
    assert feeder_ranges(both) is None and kind_bytes(both) is None
    gate_down = [("p", 0, 800, E, "gate"), down]
    assert feeder_ranges(gate_down) is None
    assert kind_bytes(gate_down) is None


def test_up_down_stacks_with_no_gate_pass_through():
    # SwitchMLP layers (nemotron_h): fc1 is the up stack, fc2 the down.
    raw = [("p", 0, 800, E, "up"), ("p", 900, 400, E, "down")]
    assert feeder_ranges(raw) == [
        FeederRange("p", 0, 100, 100, E, "up", 0, None),
        FeederRange("p", 900, 50, 50, E, "down", 900, None),
    ]
    assert kind_bytes(raw) == {"up": 800, "down": 400}


def test_the_refusal_names_why_each_layer_was_skipped():
    from gmlx.stream.feeder_common import no_layers_error, ranges_or_reason

    gate_down = [("p", 0, 800, E, "gate"), ("p", 900, 400, E, "down")]
    assert ranges_or_reason(gate_down) == (
        None, "expert stacks down/gate, not gate/up/down or up/down")
    flat = [("p", 0, 1600, 0, "gate_up"), ("p", 1700, 400, 0, "down")]
    assert ranges_or_reason(flat)[1] == (
        "an expert stack with no per-expert layout")
    err = no_layers_error({"expert stacks with a bias": 24,
                           "expert stacks down/up, not gate/up/down": 1})
    assert str(err) == ("no layers the feeder can stream: expert stacks "
                        "with a bias (24 layers); expert stacks down/up, "
                        "not gate/up/down (1 layer)")


def _bf16_bits(shape, seed=0):
    """Random BF16 values as their uint16 bit patterns (numpy has no
    bfloat16): the top half of float32."""
    f = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)


def test_typed_views_land_bf16_bytes_on_the_weight_geometry():
    import mlx.core as mx
    import mlx_kquant as kq

    from gmlx.stream.feeder_common import slot_view, typed_view

    shape = (2, 3, 64)
    bits = _bf16_bits(shape)
    raw = bits.tobytes()

    # A plain uint8 arena (the decode feeder's slots).
    arr, mv = kq.arena_alloc([len(raw)])
    mv[: len(raw)] = raw
    v = typed_view(arr, shape, mx.bfloat16)
    assert v.dtype == mx.bfloat16 and v.shape == shape
    assert np.array_equal(np.array(v.view(mx.uint16)), bits)
    # A uint8 dtype, or none, keeps the raw byte geometry.
    assert typed_view(arr[:96], (2, 48), mx.uint8).shape == (2, 48)
    assert typed_view(arr[:96], (2, 48)).dtype == mx.uint8

    # The prefill ring's slot view, on a plain and on a wide arena.
    for itemsize in (1, 8):
        arr, mv = kq.arena_alloc([len(raw) // itemsize], itemsize=itemsize)
        mv[: len(raw)] = raw
        v = slot_view(arr, len(raw), shape, mx.bfloat16)
        assert v.dtype == mx.bfloat16 and v.shape == shape
        assert np.array_equal(np.array(v.view(mx.uint16)), bits)


def test_verify_zero_copy_reads_a_bf16_stack(tmp_path):
    import os

    gguf = pytest.importorskip("gguf")
    kq = pytest.importorskip("mlx_kquant")

    from gmlx.stream.feeder_common import verify_zero_copy

    bits = _bf16_bits((4, 8, 16), seed=1)
    path = str(tmp_path / "bf16.gguf")
    w = gguf.GGUFWriter(path, "llama")
    w.add_tensor("blk.0.ffn_up_exps.weight", bits,
                 raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    t = gguf.GGUFReader(path).tensors[0]
    off = int(t.data_offset)

    class _Proj:
        weight = kq.load_gguf(path, True)[0]["blk.0.ffn_up_exps.weight"]

    class _Mod:
        up_proj = _Proj()

    assert _Proj.weight.dtype.size == 2  # a native BF16 stack, not bytes
    fd = os.open(path, os.O_RDONLY)
    try:
        verify_zero_copy(0, [("up", _Mod(), path, off, 8 * 16 * 2)],
                         {path: fd})
        with pytest.raises(RuntimeError, match="not a zero-copy view"):
            verify_zero_copy(0, [("up", _Mod(), path, off + 2, 8 * 16 * 2)],
                             {path: fd})
    finally:
        os.close(fd)
