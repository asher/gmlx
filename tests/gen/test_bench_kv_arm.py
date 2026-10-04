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


def test_unset_scheme_picks_per_model(kvarn_ops_ok, monkeypatch, capsys):
    from types import SimpleNamespace

    from mlx_vlm.models.cache import ArraysCache, KVCache

    import gmlx.gen.generation as gen

    monkeypatch.setattr(gen, "setup_kvarn_cache",
                        lambda *a, **kw: ["kvarn"])
    hybrid = SimpleNamespace(
        args=SimpleNamespace(head_dim=128),
        make_cache=lambda: [KVCache(), ArraysCache(1)] * 2)
    kwargs, factory = bm._bench_kv_arm(hybrid, 8, 64)
    assert factory() == ["kvarn"]
    assert "auto picked kvarn" in capsys.readouterr().err
