"""Serve-side KV policy resolve at model load."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_vlm")

from mlx_vlm.models.cache import KVCache  # noqa: E402

import gmlx.serve.kv_policy as skv  # noqa: E402


def _model(layers=4):
    lm = SimpleNamespace(
        layers=[None] * layers,
        make_cache=lambda: [KVCache() for _ in range(layers)],
        config=SimpleNamespace(num_hidden_layers=layers,
                               num_attention_heads=8,
                               num_key_value_heads=8, head_dim=64),
    )
    lm.language_model = lm
    return lm


def _rg(**over):
    kw = dict(model=_model(), kv_bits=8.0, kv_group_size=64,
              quantized_kv_start=0, kv_quant_scheme="uniform",
              kv_key_bits=None, kv_value_bits=None, kv_key_scheme=None,
              kv_value_scheme=None, draft_model_path=None)
    kw.update(over)
    return SimpleNamespace(**kw)


def test_resolve_stamps_both_modes(monkeypatch):
    monkeypatch.setenv("KV_BITS", "8")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    rg = _rg()
    pol = skv.resolve_for_load(rg, "m")
    assert getattr(rg, skv.RG_ATTR) is pol
    assert pol.single.verdict == "full" and pol.batched.verdict == "full"
    assert pol.pricing_vector() == [1.0625] * 3 + [2.0]


def test_mtp_batched_dropped(monkeypatch):
    monkeypatch.setenv("KV_BITS", "8")
    monkeypatch.setenv("MLX_VLM_GGUF_SPECULATIVE", "1")
    pol = skv.resolve_for_load(_rg(), "m")
    assert pol.single.verdict == "full"
    assert pol.batched.verdict == "dropped"
    assert pol.pricing_vector() == [2.0] * 4


def test_error_verdict_raises(monkeypatch):
    monkeypatch.setenv("KV_BITS", "8")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    with pytest.raises(skv.KvPolicyError, match="turboquant"):
        skv.resolve_for_load(_rg(kv_quant_scheme="turboquant"), "m")
    with pytest.raises(skv.KvPolicyError, match="split"):
        skv.resolve_for_load(_rg(kv_key_bits=8.0), "m")


def test_qat_drop_surfaces(monkeypatch):
    # Upstream drops KV_BITS for qat ids. The policy reports the drop.
    monkeypatch.setenv("KV_BITS", "8")
    pol = skv.resolve_for_load(_rg(kv_bits=None), "m-qat")
    assert pol.single.verdict == "dropped"
    assert "qat" in pol.single.reason


def test_kv_off_returns_none(monkeypatch):
    monkeypatch.delenv("KV_BITS", raising=False)
    assert skv.resolve_for_load(_rg(kv_bits=None), "m") is None


def test_to_json_shapes(monkeypatch):
    monkeypatch.setenv("KV_BITS", "8")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    j = skv.resolve_for_load(_rg(), "m").to_json()
    assert j == {"bits": 8, "group_size": 64, "layers_quantized": 3,
                 "layers_fp16": 1, "verdict": "full",
                 "verdict_batched": "full"}
    monkeypatch.setenv("MLX_VLM_GGUF_SPECULATIVE", "1")
    j = skv.resolve_for_load(_rg(), "m").to_json()
    assert j["verdict"] == "full"
    assert j["verdict_batched"] == "dropped"
    assert "batched_reason" in j
