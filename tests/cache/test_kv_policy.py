from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache

from gmlx.cache.kv_policy import (kv_line, packed_bytes_per_element,
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
    # gemma shape: the carve-out is by index, so a KV layer in the last
    # slot is held even when windows sit between (2-of-3 globals census).
    stack = [RotatingKVCache(512), KVCache(), RotatingKVCache(512), KVCache()]
    p = _resolve(stack)
    assert p.verdict == "partial"
    assert p.n_window == 2 and p.n_quant == 1 and p.n_held == 1


def test_rotating_only_natural_drops():
    p = _resolve([RotatingKVCache(512)] * 4)
    assert p.verdict == "dropped"
    assert "no quantizable layers" in p.reason


def test_rotating_under_max_kv_size_errors():
    # Defect 4: --kv-bits + --max-kv-size on a no-make_cache model.
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
