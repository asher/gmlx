#!/usr/bin/env python3
"""Per-row LoRA scale channel (``lora_rows``): static vs rows mode, the dense
wrapper's per-row factor and its dtype rule, the per-thread published
vector, the uid lookups bound per generator, row retirement, and the
borrowed-encoder scope. CPU-only (tiny float Linears, fake engine
objects)."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
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


@pytest.mark.parametrize("published, shape", [
    ([1.0, 1.0, 1.0], (2, 1, 4)),
    # An input with a batch dimension whose row total is a multiple of the
    # published count is still a mismatch.
    ([1.0, 0.0, 0.5], (2, 3, 4)),
    ([1.0, 0.0], (4, 3, 4)),
    ([1.0, 0.0], (1, 4, 4)),
    # A flat input must hold a multiple of the published count.
    ([1.0, 0.0], (3, 4)),
])
def test_rows_mode_row_count_mismatch_raises(published, shape):
    lora_rows.configure("rows", 1)
    wrap, _base, _a, _b = _wrap()
    lora_rows.set_rows(published)
    with pytest.raises(lora_rows.LoraRowsError,
                       match=f"has {len(published)} rows, forward has"):
        wrap(mx.zeros(shape))


def test_flat_inputs_of_two_sizes_share_one_publish():
    """One publish serves flat (B * tokens, D) inputs of any multiple of
    its row count: each gets its own factor, one value per input row in
    (B, tokens) order."""
    lora_rows.configure("rows", 1)
    wrap, base, a, b = _wrap()
    lora_rows.set_rows([2.0, 0.0])
    for tokens in (2, 3, 1, 2):
        x = mx.random.normal((2 * tokens, 4))
        y = wrap(x)
        assert y.shape == (2 * tokens, 3)
        on, off = x[:tokens], x[tokens:]
        ref = base(on) + (0.5 * 2.0) * ((on @ a.T) @ b.T)
        assert mx.allclose(y[:tokens], ref, rtol=1e-2, atol=1e-2), tokens
        assert mx.allclose(y[tokens:], base(off), rtol=1e-2,
                           atol=1e-2), tokens
        delta = mx.abs((0.5 * 2.0) * ((off @ a.T) @ b.T)).max().item()
        assert delta > 0.1, "the bare rows would show a delta"


def test_published_clears_after_forward_in_rows_mode():
    lora_rows.configure("rows", 1)
    with lora_rows.bind(lora_rows.UidRegistry()):
        lora_rows.register_uid(7, (1.0,))
        lora_rows.register_uid(8, None)
        with lora_rows.published([7, 8]):
            v = lora_rows.row_scales(0)
            assert v.tolist() == [1.0, 0.0]
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


def test_all_off_rows_run_the_base_alone(monkeypatch):
    lora_rows.configure("rows", 2)
    wrap, base, _a, _b = _wrap(dtype=mx.float16)
    wrap.add_slot(mx.ones((2, 4), mx.float16), mx.ones((3, 2), mx.float16), 1.0, 1)
    reg = lora_rows.UidRegistry()
    reg.register(1, (0.0, 0.0))
    reg.register(2, (0.0, 1.0))
    x = mx.random.normal((2, 3, 4)).astype(mx.float16)
    called = []
    real = lora_rows.dense_row_factor
    monkeypatch.setattr(lora_rows, "dense_row_factor",
                        lambda slot, *a: called.append(slot) or real(slot, *a))
    monkeypatch.setattr(lora_rows, "dense_input_rows",
                        lambda slot, *a: pytest.fail("epilogue rows read"))
    with lora_rows.bind(reg), lora_rows.published([1, 1]):
        assert lora_rows.all_off((0, 1))
        assert mx.array_equal(wrap(x), base(x))
    assert called == []
    with lora_rows.bind(reg), lora_rows.published([1, 2]):
        assert lora_rows.slot_off(0) and not lora_rows.slot_off(1)
        y = wrap(x)
    assert called == [1]
    assert mx.array_equal(y[0], base(x[0:1])[0])
    assert not mx.array_equal(y[1], base(x[1:2])[0])


def test_a_direct_publish_never_counts_as_off():
    lora_rows.configure("rows", 1)
    lora_rows.set_rows([0.0])
    assert not lora_rows.slot_off(0)


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


# uid lookups: per generator, per thread, retirement, padding


@pytest.fixture
def ar():
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    from gmlx._vendor.mlx_vlm.generate import ar as _ar
    lora_rows.install_row_channel()
    return _ar


def _row_lora(scales, salt):
    from gmlx.serve.lora_request import RowLora
    return RowLora(tuple(scales), np.asarray(scales, np.float32), salt, "m")


def _settings(scales, salt=0):
    from gmlx.serve.row_sampling import RowSettings
    return RowSettings(lora=_row_lora(scales, salt))


def _fake_gen(**kw):
    gen = SimpleNamespace(max_tokens=8, logits_processors=None,
                          _unprocessed_sequences=[], uid_count=0,
                          _prompt_batch=None, _generation_batch=None)
    gen.__dict__.update(kw)
    return gen


def _step_vector(gen, uids):
    """What a publish over ``uids`` reads inside ``gen``'s step."""
    seen = {}

    def step(self, **kwargs):
        with lora_rows.published(uids):
            seen["v"] = np.array(lora_rows._local.current).tolist()
            seen["salt"] = [lora_rows.salt_for(u) for u in uids]
        return [], []

    lora_rows.bound_step(step)(gen)
    return seen


def test_two_generators_keep_their_own_uid_rows(ar):
    """Uids count from 0 in each generator: two resident entries both hand
    out uid 0, and each step reads its own generator's row."""
    lora_rows.configure("rows", 2)
    a, b = _fake_gen(), _fake_gen()
    ar.BatchGenerator.insert(a, [[1, 2]], row_settings=[_settings((1.0, 0.0), 11)])
    ar.BatchGenerator.insert(b, [[1, 2]], row_settings=[_settings((0.0, 1.0), 22)])
    got_a, got_b = _step_vector(a, [0]), _step_vector(b, [0])
    assert got_a == {"v": [[1.0, 0.0]], "salt": [11]}
    assert got_b == {"v": [[0.0, 1.0]], "salt": [22]}


def test_the_step_binds_the_lookup_only_for_its_length(ar):
    from gmlx.upstream.patching import wrapped_with

    assert wrapped_with(ar.BatchGenerator._next, lora_rows._ROWS_STEP_FLAG)
    lora_rows.configure("rows", 1)
    gen = _fake_gen()
    ar.BatchGenerator.insert(gen, [[1]], row_settings=[_settings((0.5,), 3)])
    assert _step_vector(gen, [0])["v"] == [[0.5]]
    assert lora_rows.bound() is None
    with pytest.raises(lora_rows.LoraRowsError):
        lora_rows.publish_uids([0])          # outside the step: no lookup


def test_insert_in_rows_mode_needs_a_row_lora(ar):
    lora_rows.configure("rows", 1)
    with pytest.raises(lora_rows.LoraRowsError):
        ar.BatchGenerator.insert(_fake_gen(), [[1]])
    from gmlx.serve.row_sampling import RowSettings
    with pytest.raises(lora_rows.LoraRowsError):
        ar.BatchGenerator.insert(_fake_gen(), [[1]], row_settings=[RowSettings()])
    with pytest.raises(lora_rows.LoraRowsError, match="unsalted"):
        ar.BatchGenerator.insert(_fake_gen(), [[1]],
                                 row_settings=[_settings((1.0,), None)])


def test_insert_salts_adapted_rows_only(ar):
    lora_rows.configure("rows", 2)
    gen = _fake_gen(apc_manager=object(), _apc_extra_hash=lambda kw: 1234)
    kws = [{"_apc_semantic_hash": 99}, {"_apc_semantic_hash": 99}, {}]
    ar.BatchGenerator.insert(
        gen, [[1, 2], [1, 2], [1, 2]], prompt_kwargs=kws,
        row_settings=[_settings((0.0, 0.0), 0), _settings((1.0, 0.0), 77),
                      _settings((0.0, 1.0), 88)])
    assert kws[0]["_apc_semantic_hash"] == 99
    assert kws[1]["_apc_semantic_hash"] == 99 ^ 77
    assert kws[2]["_apc_semantic_hash"] == 1234 ^ 88


def test_published_vector_is_per_thread():
    """Each engine publishes on its own thread: another thread sees no
    vector, and its clear leaves this thread's in place."""
    lora_rows.configure("rows", 1)
    lora_rows.set_rows([0.5])
    seen = {}

    def other():
        seen["other"] = lora_rows._local.current
        lora_rows.set_rows(None)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert "other" in seen and seen["other"] is None
    assert lora_rows.row_scales(0).tolist() == [0.5]
    lora_rows.clear_rows()


def test_a_finished_uid_still_in_a_round_reads_as_zeros(ar):
    lora_rows.configure("rows", 1)
    gen = _fake_gen()
    ar.BatchGenerator.insert(gen, [[1], [2]],
                             row_settings=[_settings((1.0,), 5), _settings((1.0,), 5)])
    ar.BatchGenerator._retire_row(gen, 0)
    view = lora_rows._GeneratorRows(gen)
    with lora_rows.bind(view), lora_rows.published([0, 1]):
        assert lora_rows.row_scales(0).tolist() == [0.0, 1.0]
    with lora_rows.bind(view), pytest.raises(lora_rows.LoraRowsError):
        lora_rows.publish_uids([7])                    # never stored


def test_remove_keeps_the_row_and_the_store_prunes_only_dead_uids(ar, monkeypatch):
    from gmlx.serve import row_sampling
    monkeypatch.setattr(row_sampling, "MAX_ROWS", 8)
    lora_rows.configure("rows", 1)
    gen = _fake_gen()
    ar.BatchGenerator.insert(gen, [[1]], row_settings=[_settings((1.0,), 5)])
    gen._generation_batch = SimpleNamespace(uids=[0])          # uid 0 is live
    for _ in range(12):
        (uid,) = ar.BatchGenerator.insert(gen, [[1]],
                                          row_settings=[_settings((0.5,), 6)])
        gen._unprocessed_sequences.clear()
        ar.BatchGenerator._retire_row(gen, uid)
        gen._row_settings[uid] = _settings((0.5,), 6)          # finished, not popped
    assert 0 in gen._row_settings
    assert len(gen._row_settings) <= 9
    assert len(gen._retired_rows) <= 9
    with lora_rows.bind(lora_rows._GeneratorRows(gen)), lora_rows.published([0]):
        assert lora_rows.row_scales(0).tolist() == [1.0]


def test_rows_mode_publish_needs_a_lookup_and_a_known_uid():
    lora_rows.configure("rows", 1)
    with pytest.raises(lora_rows.LoraRowsError, match="no row lookup"):
        lora_rows.publish_uids([1])
    with lora_rows.bind(lora_rows.UidRegistry()):
        with pytest.raises(lora_rows.LoraRowsError, match="no LoRA scales"):
            lora_rows.publish_uids([1])
    lora_rows.publish_uids([])                     # nothing to read
    lora_rows.clear_rows()


def test_rows_pad_on_read_and_stay_as_stored():
    lora_rows.configure("rows", 1)
    reg = lora_rows.UidRegistry()
    with lora_rows.bind(reg):
        lora_rows.register_uid(3, (1.0,))
        lora_rows.ensure_rows(5)
        with lora_rows.published([3]):
            assert np.array(lora_rows._local.current).tolist() == [[1.0, 0, 0, 0, 0]]
        assert lora_rows.scales_for(3) == (1.0, 0.0, 0.0, 0.0, 0.0)
    assert reg.row(3).row.tolist() == [1.0]


@pytest.mark.parametrize("b,s", [(1, 1), (3, 4), (8, 32)])
def test_numpy_rows_match_the_per_slot_loops(b, s):
    rng = np.random.default_rng(b * 100 + s)
    lora_rows.configure("rows", s)
    reg = lora_rows.UidRegistry()
    rows = []
    for uid in range(b):
        row = [float(v) if rng.random() < 0.5 else 0.0 for v in rng.random(s)]
        rows.append(row)
        reg.register(uid, row)
    with lora_rows.bind(reg), lora_rows.published(list(range(b))):
        got = np.array(lora_rows._local.current).tolist()
        off = lora_rows._local.off
    want_off = tuple(all(r[k] == 0.0 for r in rows) for k in range(s))
    assert np.allclose(got, rows) and off == want_off


def test_publish_tuple_marks_zero_slots_off():
    lora_rows.configure("rows", 3)
    lora_rows.publish_tuple((0.5, 0.0), 2)
    assert np.array(lora_rows._local.current).tolist() == [[0.5, 0, 0], [0.5, 0, 0]]
    assert not lora_rows.slot_off(0) and lora_rows.slot_off(1) and lora_rows.slot_off(2)
    lora_rows.clear_rows()


def test_row_salt_names_files_and_strengths_not_slots():
    ida = SimpleNamespace(sha256="a" * 64)
    idb = SimpleNamespace(sha256="b" * 64)
    assert lora_rows.row_salt((ida, idb), (0.0, 0.0)) == 0
    one = lora_rows.row_salt((ida, idb), (0.5, 1.0))
    assert lora_rows.row_salt((idb, ida), (1.0, 0.5)) == one        # slot order
    assert lora_rows.row_salt((ida, idb, ida), (0.5, 1.0, 0.0)) == one   # padding
    assert lora_rows.row_salt((ida, idb), (0.5, 0.5)) != one
    other = SimpleNamespace(sha256="c" * 64)
    assert lora_rows.row_salt((other, idb), (0.5, 1.0)) != one      # another file


# the borrowed encoder runs with every slot off


def test_adapters_off_runs_the_base_and_restores_the_outer_state():
    lora_rows.configure("rows", 2)
    wrap, base, _a, _b = _wrap()
    wrap.add_slot(mx.ones((2, 4)), mx.ones((3, 2)), 1.0, 1)
    x = mx.random.normal((2, 3, 4))
    with pytest.raises(lora_rows.LoraRowsError):
        wrap(x)                                   # nothing published
    lora_rows.set_rows([[1.0, 0.0], [0.0, 1.0]])
    outer = (lora_rows._local.current, lora_rows._local.off, lora_rows._local.derived)
    lora_rows.row_factor(0, mx.float32, (2, 1, 1))
    assert outer[2]
    with lora_rows.adapters_off():
        assert lora_rows.all_off((0, 1))
        assert mx.array_equal(wrap(x), base(x))
        with lora_rows.adapters_off():
            assert lora_rows.slot_off(1)
        lora_rows.set_rows([[1.0, 1.0]])
        lora_rows.set_rows(None)
    assert lora_rows._local.current is outer[0]
    assert lora_rows._local.off is outer[1]
    assert lora_rows._local.derived is outer[2] and outer[2]
    lora_rows.clear_rows()
