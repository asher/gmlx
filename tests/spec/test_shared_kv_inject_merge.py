"""Regression for the gemma shared-KV spec-injection crash (2026-07-23
bench, second layer under the RotatingKVCache lift fix): _drain_injections
batch-concatenated the injected stream's prefill shared-KV onto the engine
dict with no seq alignment -- windowed shared layers cap S per prefill, so
K_old (B, H, S_old, D) vs K_new (1, H, S_inj, D) died in mx.concatenate.
The concat also never reached the drafter (set_shared_kv normalizes into a
copy), whose stored view, position, and kv_valid_len stayed at the old
width. _merge_injected_shared_kv widens the dict in the drafter's
normalized layout (front-aligned rows, zero tail) and the drain site
re-calls set_shared_kv with the widened per-row arrays."""

import mlx.core as mx

from gmlx.spec.speculative import _merge_injected_shared_kv, _pad_shared_kv_seq

H, D = 2, 4


def _kv(batch, seq, fill):
    K = mx.full((batch, H, seq, D), float(fill))
    return K, K + 0.5


class _FakeDrafter:
    def __init__(self, shared_kv=None):
        self._shared_kv = shared_kv


def _widths(shared_kv):
    return {k: (v[0].shape, v[1].shape) for k, v in shared_kv.items()}


def test_ragged_seq_is_tail_padded_and_rows_appended():
    engine = {"full_attention": _kv(2, 6, 1.0)}
    drafter = _FakeDrafter({"full_attention": _kv(2, 6, 2.0)})
    inj = {"full_attention": _kv(1, 9, 3.0)}
    _merge_injected_shared_kv(drafter, engine, inj, n_old=2, n_new=1)
    K, V = engine["full_attention"]
    assert K.shape == (3, H, 9, D) and V.shape == (3, H, 9, D)
    # live drafter view preferred over engine snapshot for old rows
    assert mx.all(K[:2, :, :6, :] == 2.0).item()
    assert mx.all(V[:2, :, :6, :] == 2.5).item()
    # old rows tail-padded with zeros to the injected row's seq
    assert mx.all(K[:2, :, 6:, :] == 0.0).item()
    # injected row appended intact
    assert mx.all(K[2, :, :, :] == 3.0).item()


def test_shorter_injected_row_is_tail_padded():
    engine = {"full_attention": _kv(2, 10, 1.0)}
    drafter = _FakeDrafter({"full_attention": _kv(2, 10, 2.0)})
    inj = {"full_attention": _kv(1, 4, 3.0)}
    _merge_injected_shared_kv(drafter, engine, inj, n_old=2, n_new=1)
    K, _ = engine["full_attention"]
    assert K.shape == (3, H, 10, D)
    assert mx.all(K[2, :, :4, :] == 3.0).item()
    assert mx.all(K[2, :, 4:, :] == 0.0).item()


def test_stale_live_width_falls_back_to_engine_dict():
    # drafter view still at a retired-rows width; engine dict matches
    engine = {"full_attention": _kv(2, 6, 1.0)}
    drafter = _FakeDrafter({"full_attention": _kv(5, 6, 2.0)})
    inj = {"full_attention": _kv(1, 6, 3.0)}
    _merge_injected_shared_kv(drafter, engine, inj, n_old=2, n_new=1)
    K, _ = engine["full_attention"]
    assert K.shape == (3, H, 6, D)
    assert mx.all(K[:2] == 1.0).item()
    assert mx.all(K[2] == 3.0).item()


def test_all_finished_adoption_zero_row_base():
    # adoption path: engine dict filtered to 0 rows, drafter view stale
    K0, V0 = _kv(4, 6, 9.0)
    empty = mx.array([], dtype=mx.int32)
    engine = {"full_attention": (K0[empty], V0[empty])}
    drafter = _FakeDrafter({"full_attention": _kv(4, 6, 9.0)})
    inj = {"full_attention": _kv(2, 7, 3.0)}
    _merge_injected_shared_kv(drafter, engine, inj, n_old=0, n_new=2)
    K, _ = engine["full_attention"]
    assert K.shape == (2, H, 7, D)
    assert mx.all(K == 3.0).item()


def test_missing_capture_key_zero_fills_rows():
    engine = {
        "full_attention": _kv(2, 6, 1.0),
        "sliding_attention": _kv(2, 5, 1.0),
    }
    drafter = _FakeDrafter(None)
    inj = {"full_attention": _kv(1, 6, 3.0)}
    _merge_injected_shared_kv(drafter, engine, inj, n_old=2, n_new=1)
    for key, (K, V) in engine.items():
        assert K.shape[0] == 3, key
    K, V = engine["sliding_attention"]
    assert mx.all(K[2] == 0.0).item() and mx.all(V[2] == 0.0).item()


def test_no_width_match_zero_fills_old_rows():
    # neither live nor engine matches n_old: shape-safe zeros for old
    # rows beat a width crash at the next draft_block
    engine = {"full_attention": _kv(5, 6, 1.0)}
    drafter = _FakeDrafter({"full_attention": _kv(4, 6, 2.0)})
    inj = {"full_attention": _kv(1, 6, 3.0)}
    _merge_injected_shared_kv(drafter, engine, inj, n_old=2, n_new=1)
    K, _ = engine["full_attention"]
    assert K.shape == (3, H, 6, D)
    assert mx.all(K[:2] == 0.0).item()
    assert mx.all(K[2] == 3.0).item()


def test_two_entry_drain_composes_via_engine_dict():
    # entry 1 widens the engine dict in place; entry 2's live view is
    # stale, so the width check routes to the engine dict
    engine = {"full_attention": _kv(2, 6, 1.0)}
    live = {"full_attention": _kv(2, 6, 2.0)}
    drafter = _FakeDrafter(live)
    _merge_injected_shared_kv(
        drafter, engine, {"full_attention": _kv(1, 8, 3.0)}, 2, 1)
    _merge_injected_shared_kv(
        drafter, engine, {"full_attention": _kv(1, 4, 4.0)}, 3, 1)
    K, _ = engine["full_attention"]
    assert K.shape == (4, H, 8, D)
    assert mx.all(K[:2, :, :6, :] == 2.0).item()
    assert mx.all(K[2, :, :8, :] == 3.0).item()
    assert mx.all(K[3, :, :4, :] == 4.0).item()
    assert mx.all(K[3, :, 4:, :] == 0.0).item()


def test_dtype_preserved():
    K = mx.full((2, H, 6, D), 1.0, dtype=mx.bfloat16)
    engine = {"full_attention": (K, K)}
    drafter = _FakeDrafter(None)
    inj_K = mx.full((1, H, 9, D), 3.0, dtype=mx.bfloat16)
    _merge_injected_shared_kv(
        drafter, engine, {"full_attention": (inj_K, inj_K)}, 2, 1)
    K_m, V_m = engine["full_attention"]
    assert K_m.dtype == mx.bfloat16 and V_m.dtype == mx.bfloat16


def test_pad_noop_returns_same_array():
    K, _ = _kv(2, 6, 1.0)
    assert _pad_shared_kv_seq(K, 6) is K
    padded = _pad_shared_kv_seq(K, 8)
    assert padded.shape == (2, H, 8, D)
    assert mx.all(padded[:, :, 6:, :] == 0.0).item()
