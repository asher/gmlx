import pytest
from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache

from gmlx.cache.kv_policy import (kv_line, kvarn_bytes_per_element,
                                  kvarn_fixed_tokens,
                                  packed_bytes_per_element,
                                  resolve_kv_quant_policy)


def _dense(n):
    return [KVCache() for _ in range(n)]


def _resolve(stack, **kw):
    kw.setdefault("kv_bits", 8)
    return resolve_kv_quant_policy(stack, **kw)


def test_dense_full_with_carveout():
    p = _resolve(_dense(28))
    assert p.verdict == "full"
    assert p.n_quant == 27 and p.n_held == 1
    assert not p.per_layer[-1].quantize
    assert p.bytes_per_element_vector()[-1] == 2.0
    assert p.bytes_per_element_vector()[0] == packed_bytes_per_element(8, 64)


def test_shallow_stack_quantizes_all():
    p = _resolve(_dense(2))
    assert p.verdict == "full" and p.n_quant == 2 and p.n_held == 0


def test_hybrid_partial():
    # qwen3.5 shape: GDN state layers between attn layers.
    stack = [KVCache(), ArraysCache(1), KVCache(), ArraysCache(1), KVCache()]
    p = _resolve(stack)
    assert p.verdict == "partial"
    assert p.n_quant == 2 and p.n_held == 1 and p.n_state == 2


def test_window_layers_stay_fp16():
    # gemma shape: the carve-out is by index, so a kv layer in the
    # last slot is held even with windows between.
    stack = [RotatingKVCache(512), KVCache(), RotatingKVCache(512), KVCache()]
    p = _resolve(stack)
    assert p.verdict == "partial"
    assert p.n_window == 2 and p.n_quant == 1 and p.n_held == 1


def test_rotating_only_natural_drops():
    p = _resolve([RotatingKVCache(512)] * 4)
    assert p.verdict == "dropped"
    assert "no quantizable layers" in p.reason


def test_rotating_under_max_kv_size_errors():
    # --kv-bits + --max-kv-size on a model without make_cache.
    p = _resolve([RotatingKVCache(512)] * 4, max_kv_size=512)
    assert p.verdict == "error"
    assert "max_kv_size" in p.reason


def test_mtp_batched_drops_single_quantizes():
    stack = _dense(8)
    single = _resolve(stack, mtp=True, mode="single")
    batched = _resolve(stack, mtp=True, mode="batched")
    assert single.verdict == "full"
    assert batched.verdict == "dropped"
    assert all(b == 2.0 for b in batched.bytes_per_element_vector())


def test_non_mtp_batched_same_as_single():
    stack = _dense(8)
    a = _resolve(stack, mode="single")
    b = _resolve(stack, mode="batched")
    assert (a.verdict, a.n_quant) == (b.verdict, b.n_quant)


def test_bad_bits_error():
    for bad in (5, 3.5, 16):
        p = _resolve(_dense(4), kv_bits=bad)
        assert p.verdict == "error" and "kv_bits" in p.reason


def test_bad_group_error():
    p = _resolve(_dense(4), kv_group_size=48)
    assert p.verdict == "error" and "kv_group_size" in p.reason


def test_head_dim_group_mismatch_error():
    p = _resolve(_dense(4), kv_group_size=128, head_dim=96)
    assert p.verdict == "error" and "head_dim" in p.reason


def test_turbo_scheme_error():
    p = _resolve(_dense(4), scheme="turboquant")
    assert p.verdict == "error" and "turboquant" in p.reason
    assert _resolve(_dense(4), scheme="uniform").verdict == "full"


def test_split_bits_error():
    assert _resolve(_dense(4), key_bits=8).verdict == "error"
    assert _resolve(_dense(4), value_bits=4).verdict == "error"


def test_start_not_honored_batched():
    p = _resolve(_dense(4), quantized_kv_start=100, mode="batched")
    assert not p.start_honored
    assert "quantized_kv_start" in p.summary()
    assert _resolve(_dense(4), quantized_kv_start=100).start_honored


def test_packed_bpe_values():
    assert packed_bytes_per_element(8, 64) == 1.0625
    assert packed_bytes_per_element(4, 32) == 0.625


def test_kv_line_shape():
    line = kv_line("qwen3-0.6b", _resolve(_dense(28)))
    assert line.startswith("[kv] qwen3-0.6b: kv_bits=8 group=64 -> ")
    assert "quantized 27/28 attn layers (1 held fp16)" in line
    dropped = kv_line("m", _resolve(_dense(4), mtp=True, mode="batched"))
    assert "-> dropped:" in dropped


# -- kvarn scheme ------------------------------------------------------------


@pytest.fixture
def _kvarn_ops(monkeypatch):
    """kvarn eligibility without the Metal kernels: the resolver reads the
    ops probe, and every row below is pure policy arithmetic."""
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    for k in ("GMLX_KVARN", "GMLX_KVARN_BITS"):
        monkeypatch.delenv(k, raising=False)


def _kvarn(stack, **kw):
    kw.setdefault("kv_bits", 6)
    kw.setdefault("scheme", "kvarn")
    return resolve_kv_quant_policy(stack, **kw)


def test_kvarn_dense_carveout(_kvarn_ops):
    p = _kvarn(_dense(28))
    assert p.verdict == "full" and p.scheme == "kvarn"
    assert p.n_quant == 27 and p.n_held == 1
    assert p.bytes_per_element_vector()[0] == kvarn_bytes_per_element(6)
    assert p.bytes_per_element_vector()[-1] == 2.0
    # The fp16 sink/horizon/tail buffers are priced as a resident region,
    # not folded into the per-token cost.
    assert p.per_layer[0].regions == ((kvarn_fixed_tokens(1024), 2.0),)
    assert p.per_layer[-1].regions == ()


def test_kvarn_record_bpe_beats_kv8():
    assert kvarn_bytes_per_element(6) == 6 / 8 + 3 * 2.0 / 128
    assert kvarn_bytes_per_element(6, 5) == 11 / 16 + 3 * 2.0 / 128
    assert kvarn_bytes_per_element(6) < packed_bytes_per_element(8, 64)


def test_kvarn_fixed_region_geometry():
    # sink stage (128 + one spare group) + horizon group + tail + slack.
    assert kvarn_fixed_tokens(1024) == 128 + 128 + 128 + 1024 + 256
    assert kvarn_fixed_tokens(0) == 128 + 128 + 128 + 1


def test_kvarn_split_widths(_kvarn_ops):
    p = _kvarn(_dense(2), kv_bits=6, value_bits=5)
    assert p.verdict == "full" and p.value_bits == 5
    assert kv_line("m", p).startswith("[kv] m: kvarn k6 v5 tail=1024 -> ")


def test_kvarn_width_and_tail_errors(_kvarn_ops):
    assert _kvarn(_dense(2), kv_bits=7).verdict == "error"
    p = _kvarn(_dense(2), tail_tokens=100)
    assert p.verdict == "error" and "multiple of 128" in p.reason
    # 5-bit is a kvarn width and not an affine one
    assert _kvarn(_dense(2), kv_bits=5).verdict == "full"
    assert _resolve(_dense(2), kv_bits=5).verdict == "error"


def test_kvarn_head_dim_and_model_declines(_kvarn_ops):
    p = _kvarn(_dense(2), head_dim=64)
    assert p.verdict == "dropped" and "128/256/512" in p.reason
    p = _kvarn(_dense(2), scheme_reason="MLA latent KV cache")
    assert p.verdict == "dropped" and p.reason == "MLA latent KV cache"


def test_kvarn_optout_layer_stays_fp16(_kvarn_ops):
    class _Optout(KVCache):
        kv_quant_unsupported = True

    stack = [KVCache(), _Optout(), KVCache(), KVCache()]
    p = _kvarn(stack)
    assert p.verdict == "partial"
    assert [pl.quantize for pl in p.per_layer] == [True, False, True, False]
    assert p.per_layer[1].kind == "optout"


def test_kvarn_windows_need_the_matching_window(_kvarn_ops):
    # A model's own SWA stack keeps its windows fp16 ...
    stack = [KVCache(), RotatingKVCache(max_size=512)]
    assert _kvarn(stack).n_quant == 1
    # ... and a --max-kv-size stack converts the windows built for it.
    rot = [RotatingKVCache(max_size=4096), RotatingKVCache(max_size=4096)]
    p = _kvarn(rot, rotating_window=4096)
    assert p.verdict == "full" and p.n_quant == 2
    # a window built for some other size is not kvarn's to take
    other = [RotatingKVCache(max_size=8192)]
    assert _kvarn(other, rotating_window=4096).verdict == "dropped"


def test_kvarn_window_floor(_kvarn_ops):
    p = _kvarn(_dense(2), rotating_window=512)
    assert p.verdict == "error" and "window floor" in p.reason


def test_kvarn_batched_verdicts(_kvarn_ops):
    """The three rows serve prices from: B>1 without MTP quantizes and is
    priced with the kvarn cost; B>1 with MTP drops to fp16."""
    single = _kvarn(_dense(4), mode="single")
    assert single.verdict == "full" and single.n_quant == 3
    batched = _kvarn(_dense(4), mode="batched")
    assert batched.verdict == "full" and batched.n_quant == 3
    assert batched.bytes_per_element_vector()[0] == kvarn_bytes_per_element(6)
    mtp_batched = _kvarn(_dense(4), mode="batched", mtp=True)
    assert mtp_batched.verdict == "dropped"
    assert all(b == 2.0 for b in mtp_batched.bytes_per_element_vector())
