#!/usr/bin/env python3
"""Wire ranges the weight pin must not wire.

The pin mlocks a tensor's GGUF bytes so the runtime's zero-copy view of
them stays resident. A tensor the load pipeline converts has no such view
- an F32 lm head becomes its own bf16 array - so wiring the F32 range
holds memory nothing reads again, and on a streaming model that memory is
the decode arena's.

The match is on element count and width, not tensor names, so these cover
a head under any name and a tied head as well as the conventional one.

Unpinning alone frees nothing: the arena sizer charges the unpinned share
of the non-expert set against the same budget, so it must also stop
charging the dead wire.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from gmlx.load.loader import _decode_arena_bytes
from gmlx.stream.pin_weights import cast_copies

VOCAB, HIDDEN = 120832, 6144
HEAD_N = VOCAB * HIDDEN


def _scan(*specs):
    """specs: (name, type_name, elements)."""
    width = {"F32": 4, "F16": 2, "BF16": 2}
    return SimpleNamespace(tensors=[
        SimpleNamespace(name=n, type_name=t, nbytes=e * width.get(t, 1))
        for n, t, e in specs])


def _model(params):
    return SimpleNamespace(parameters=lambda: params)


def _arr(n, dtype=mx.bfloat16):
    return mx.zeros((n,), dtype=dtype)


def _run(params, *specs):
    with mock.patch("gmlx.load.headerscan.scan_gguf",
                    return_value=_scan(*specs)):
        return cast_copies(_model(params), "model.gguf")


def test_an_f32_head_cast_to_bf16_is_excluded():
    got = _run({"lm_head": {"weight": _arr(HEAD_N)}},
               ("output.weight", "F32", HEAD_N))
    assert got.names == frozenset({"output.weight"})


def test_it_finds_the_head_under_any_name():
    """No name list: an arch that calls its head something else still hits."""
    got = _run({"decoder": {"proj_out": {"w": _arr(HEAD_N)}}},
               ("some.other.head", "F32", HEAD_N))
    assert got.names == frozenset({"some.other.head"})


def test_a_head_the_runtime_views_in_place_stays_pinned():
    got = _run({"lm_head": {"weight": _arr(HEAD_N, mx.float32)}},
               ("output.weight", "F32", HEAD_N))
    assert got.names == frozenset()


def test_a_quantized_tensor_is_never_considered():
    """Wire bytes are the array; there is nothing to convert."""
    got = _run({"model": {"embed_tokens": {"weight": _arr(HEAD_N)}}},
               ("token_embd.weight", "Q4_K", HEAD_N))
    assert got.names == frozenset()


def test_a_tensor_with_no_matching_parameter_stays_pinned():
    got = _run({"lm_head": {"weight": _arr(64)}},
               ("output.weight", "F32", HEAD_N))
    assert got.names == frozenset()


def test_a_small_float_tensor_is_left_alone():
    """Norms and biases are not worth an element-count collision."""
    got = _run({"norm": {"weight": _arr(4096)}},
               ("output_norm.weight", "F32", 4096))
    assert got.names == frozenset()


def test_the_env_gate_disables_it():
    with mock.patch.dict("os.environ", {"GMLX_PIN_CAST_EXCLUDE": "0"}):
        got = _run({"lm_head": {"weight": _arr(HEAD_N)}},
                   ("output.weight", "F32", HEAD_N))
    assert got.names == frozenset()


def test_no_model_or_no_path_excludes_nothing():
    assert cast_copies(None, "model.gguf").names == frozenset()
    assert cast_copies(_model({}), None).names == frozenset()


def test_dead_bytes_are_the_wire_minus_the_copy():
    got = _run({"lm_head": {"weight": _arr(HEAD_N)}},
               ("output.weight", "F32", HEAD_N))
    assert got.dead_bytes == HEAD_N * 4 - HEAD_N * 2


def test_the_arena_gains_what_the_dead_wire_gave_up(monkeypatch):
    """Without this the sizer re-charges every byte the pin released."""
    import gmlx.load.loader as loader

    monkeypatch.delenv("GMLX_DECODE_ARENA_GB", raising=False)
    monkeypatch.setattr(loader, "_available_ram_bytes", lambda *a, **k: None)
    monkeypatch.setattr(mx, "device_info",
                        lambda: {"memory_size": 137 * 10**9})
    offsets = {0: [(0, 0, 200 << 30)]}
    kw = dict(total_bytes=229 << 30, offsets=offsets, budget=96 << 30,
              pinned_bytes=20 << 30)
    base = _decode_arena_bytes(**kw)
    with_cast = _decode_arena_bytes(**kw, cast_dead_bytes=3 << 30)
    assert with_cast - base == 3 << 30


def test_an_unreadable_gguf_excludes_nothing():
    with mock.patch("gmlx.load.headerscan.scan_gguf",
                    side_effect=OSError("truncated")):
        assert cast_copies(
            _model({"lm_head": {"weight": _arr(HEAD_N)}}),
            "model.gguf").names == frozenset()
