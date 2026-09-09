"""The bench KV arm resolves the scheme before the width."""

import gmlx.gen.benchmarks as bm


def test_kvarn_arm_engages_without_a_width(monkeypatch):
    import gmlx.gen.generation as gen

    seen = []

    def fake_setup(model, kv_bits, kv_tail_tokens, max_kv_size, out=None,
                   quantized_kv_start=0, **kw):
        seen.append((kv_bits, kv_tail_tokens, quantized_kv_start))
        return ["cache"]

    monkeypatch.setattr(gen, "setup_kvarn_cache", fake_setup)
    kwargs, factory = bm._bench_kv_arm(
        object(), None, 64, kv_quant_scheme="kvarn", kv_tail_tokens=512)
    assert kwargs == {} and factory is not None
    assert factory() == ["cache"]
    assert seen == [(None, 512, 0)] * 2


def test_affine_arm_needs_a_width():
    assert bm._bench_kv_arm(object(), None, 64) == ({}, None)
