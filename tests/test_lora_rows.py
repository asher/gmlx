#!/usr/bin/env python3
"""Per-row LoRA scale channel (``lora_rows``): static vs rows mode, the dense
wrapper's per-row factor and its dtype rule, and the request -> uid -> row
handoff through the engine's ``_make_logits_processors`` / ``insert`` pair.
CPU-only (tiny float Linears, fake engine objects)."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx import lora_rows
import gmlx.load.modules as modules


@pytest.fixture(autouse=True)
def _static_mode():
    lora_rows.configure("static", 1)
    yield
    lora_rows.configure("static", 1)


def _wrap(in_dims=4, out_dims=3, rank=2, scale=0.5, dtype=mx.float32):
    mx.random.seed(0)
    base = nn.Linear(in_dims, out_dims, bias=False)
    base.weight = mx.random.normal((out_dims, in_dims)).astype(dtype)
    a = mx.random.normal((rank, in_dims)).astype(dtype)
    b = mx.random.normal((out_dims, rank)).astype(dtype)
    return modules.LoRAKQuantLinear(base, a, b, scale), base, a, b


def test_static_mode_none_is_scale_one():
    wrap, base, a, b = _wrap()
    x = mx.random.normal((3, 2, 4))
    ref = base(x) + 0.5 * ((x @ a.T) @ b.T)
    assert lora_rows.row_scales(0) is None
    # f32 GEMM is TF32 on M5; the kernel path accumulates exactly
    assert mx.allclose(wrap(x), ref, rtol=1e-2, atol=1e-2)


def test_per_row_scales_equal_per_row_solo_forwards():
    wrap, base, a, b = _wrap()
    x = mx.random.normal((4, 2, 4))
    s = [0.0, 1.0, 0.25, 2.0]
    lora_rows.set_rows(s)
    y = wrap(x)
    lora_rows.clear_rows()
    for i, si in enumerate(s):
        ref = base(x[i:i + 1]) + (0.5 * si) * ((x[i:i + 1] @ a.T) @ b.T)
        assert mx.allclose(y[i:i + 1], ref, rtol=1e-2, atol=1e-2), i


def test_zero_scale_row_is_bitexact_base():
    wrap, base, _a, _b = _wrap(dtype=mx.float16)
    x = mx.random.normal((2, 3, 4)).astype(mx.float16)
    lora_rows.set_rows([0.0, 1.0])
    y = wrap(x)
    assert mx.array_equal(y[0], base(x[0:1])[0])
    assert not mx.array_equal(y[1], base(x[1:2])[0])


def test_delta_product_forms_at_adapter_dtype_not_f32():
    # bf16 base, F16 factors, float32 row vector: the factor is cast to the
    # delta dtype before the product (no f32 promotion of the delta).
    wrap, _base, a, _b = _wrap(dtype=mx.float16)
    wrap.base.weight = wrap.base.weight.astype(mx.bfloat16)
    x = mx.random.normal((2, 3, 4)).astype(mx.bfloat16)
    lora_rows.set_rows([1.0, 0.5])
    z = (x.astype(a.dtype) @ a.T)
    assert z.dtype == mx.float16
    f = (0.5 * lora_rows.row_scales(0)).astype(z.dtype)
    assert (z[..., :1] * f.reshape(2, 1, 1)).dtype == mx.float16
    y = wrap(x)
    assert y.dtype == mx.bfloat16


def test_rows_mode_unpublished_raises():
    lora_rows.configure("rows", 1)
    wrap, _base, _a, _b = _wrap()
    with pytest.raises(lora_rows.LoraRowsError):
        wrap(mx.zeros((1, 2, 4)))


def test_rows_mode_row_count_mismatch_raises():
    lora_rows.configure("rows", 1)
    wrap, _base, _a, _b = _wrap()
    lora_rows.set_rows([1.0, 1.0, 1.0])
    with pytest.raises(lora_rows.LoraRowsError):
        wrap(mx.zeros((2, 1, 4)))


def test_published_clears_after_forward_in_rows_mode():
    lora_rows.configure("rows", 1)
    lora_rows.register_uid(7, (1.0,))
    with lora_rows.published([7, 8]):
        v = lora_rows.row_scales(0)
        assert v.tolist() == [1.0, 0.0]      # 8 unregistered = bare row
    with pytest.raises(lora_rows.LoraRowsError):
        lora_rows.row_scales(0)


def test_published_is_noop_in_static_mode():
    with lora_rows.published([1, 2]):
        assert lora_rows.row_scales(0) is None


def test_slot_column_selection():
    lora_rows.configure("rows", 2)
    lora_rows.set_rows([[1.0, 0.0], [0.0, 0.5]])
    assert lora_rows.row_scales(0).tolist() == [1.0, 0.0]
    assert lora_rows.row_scales(1).tolist() == [0.0, 0.5]


def test_request_scales_from_spec():
    assert lora_rows.request_scales(None) == (0.0,)
    assert lora_rows.request_scales(SimpleNamespace(adapter=None)) == (0.0,)
    assert lora_rows.request_scales(SimpleNamespace(adapter="x.gguf")) == (1.0,)


# request -> uid handoff through the patched engine hooks

@pytest.fixture
def engine():
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.generate import ar as _ar
    from mlx_vlm.server.generation import ResponseGenerator
    lora_rows.install_row_channel()
    return _ar, ResponseGenerator


def _fake_batch_gen():
    return SimpleNamespace(max_tokens=8, logits_processors=None,
                           _unprocessed_sequences=[], uid_count=100)


def _fake_args(scales):
    return SimpleNamespace(
        logit_bias=None, repetition_penalty=None, repetition_context_size=20,
        presence_penalty=None, presence_context_size=20,
        frequency_penalty=None, frequency_context_size=20,
        logits_processors=None, enable_thinking=False, _kq_lora=scales)


def test_insert_binds_request_tuple_to_uid_and_leaves_kwargs_untouched(engine):
    _ar, RG = engine
    lora_rows.configure("rows", 1)
    bg = _fake_batch_gen()
    kw = {"inputs_embeds": None, "_apc_tenant": "t"}
    procs = RG._make_logits_processors(SimpleNamespace(), _fake_args((1.0,)), None)
    uids = _ar.BatchGenerator.insert(bg, [[1, 2, 3]], prompt_kwargs=[kw],
                                     logits_processors=[procs])
    assert lora_rows.scales_for(uids[0]) == (1.0,)
    assert kw == {"inputs_embeds": None, "_apc_tenant": "t"}    # untouched
    assert "_kq_lora" not in bg._unprocessed_sequences[0][3]   # never reaches forward
    # second insert of the same request object still yields its slot
    RG._make_logits_processors(SimpleNamespace(), _fake_args((1.0,)), None)
    uids2 = _ar.BatchGenerator.insert(bg, [[1, 2, 3]], prompt_kwargs=[kw])
    assert lora_rows.scales_for(uids2[0]) == (1.0,)


def test_insert_without_hooks_in_rows_mode_raises(engine):
    _ar, _RG = engine
    lora_rows.configure("rows", 1)
    with pytest.raises(lora_rows.LoraRowsError):
        _ar.BatchGenerator.insert(_fake_batch_gen(), [[1]])


def test_bare_request_registers_zero_row(engine):
    _ar, RG = engine
    lora_rows.configure("rows", 1)
    bg = _fake_batch_gen()
    RG._make_logits_processors(SimpleNamespace(), _fake_args((0.0,)), None)
    uids = _ar.BatchGenerator.insert(bg, [[1]])
    assert lora_rows.vector_for_uids(uids).tolist() == [[0.0]]


# chat /adapter rides the static-mode channel for the single chat row

def test_chat_adapter_command_publishes_single_row_scale(capsys):
    chat = pytest.importorskip("gmlx.tui.chat")
    state = chat.ChatState()
    chat._handle_slash("/adapter off", state)
    assert lora_rows.row_scales(0).tolist() == [0.0]
    chat._handle_slash("/adapter 0.5", state)
    assert lora_rows.row_scales(0).tolist() == [0.5]
    chat._handle_slash("/adapter on", state)
    assert lora_rows.row_scales(0) is None            # static: scale 1.0
    chat._handle_slash("/adapter", state)
    assert "adapter scale: 1" in capsys.readouterr().out
