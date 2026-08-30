"""Memory preflight: geometry estimator and the decision table."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_vlm")

import gmlx.serve.mem_preflight as mp  # noqa: E402
from mlx_vlm.server.generation import PromptTooLongError  # noqa: E402


def _model(**cfg):
    return SimpleNamespace(config=SimpleNamespace(**cfg))


DENSE = _model(num_hidden_layers=4, num_attention_heads=8,
               num_key_value_heads=8, head_dim=64)
GQA = _model(num_hidden_layers=4, num_attention_heads=8,
             num_key_value_heads=2, head_dim=64)
MLA = _model(num_hidden_layers=4, kv_lora_rank=512, qk_rope_head_dim=64)
SWA = _model(num_hidden_layers=6, num_attention_heads=8,
             num_key_value_heads=2, head_dim=64, sliding_window=128,
             sliding_window_pattern=3)
TYPED = _model(num_hidden_layers=2, num_attention_heads=8,
               num_key_value_heads=2, head_dim=64, sliding_window=128,
               layer_types=["sliding_attention", "full_attention"])


def test_dense_cost():
    costs = mp.kv_layer_costs(DENSE)
    # 2 (K+V) x 8 heads x 64 dim x 2 bytes = 2048 B/token/layer
    assert costs == [(None, 2048.0)] * 4
    assert mp.prompt_kv_bytes(costs, 1000) == 4 * 2048.0 * 1000


def test_gqa_cost_scales_with_kv_heads():
    assert mp.kv_layer_costs(GQA) == [(None, 512.0)] * 4


def test_head_dim_derived_from_hidden_size():
    m = _model(num_hidden_layers=1, num_attention_heads=8,
               hidden_size=512)
    assert mp.kv_layer_costs(m) == [(None, 2 * 8 * 64 * 2.0)]


def test_mla_prices_the_latent():
    costs = mp.kv_layer_costs(MLA)
    assert costs == [(None, (512 + 64) * 2.0)] * 4


def test_sliding_pattern_caps_layers():
    costs = mp.kv_layer_costs(SWA)
    # pattern 3: layers 3 and 6 are global, the rest cap at the window
    assert [w for w, _ in costs] == [128, 128, None, 128, 128, None]
    per = 512.0
    assert mp.prompt_kv_bytes(costs, 1000) == per * (4 * 128 + 2 * 1000)


def test_layer_types_win_over_pattern():
    assert [w for w, _ in mp.kv_layer_costs(TYPED)] == [128, None]


def test_kv_bits_lower_the_cost():
    assert mp.kv_layer_costs(DENSE, bytes_per_elem=1.0)[0][1] == 1024.0


def test_unprobeable_geometry_is_none():
    assert mp.kv_layer_costs(SimpleNamespace(config=None)) is None
    assert mp.kv_layer_costs(_model(num_hidden_layers=4)) is None


def _rg(model, kv_bits=None, tokens=None):
    rg = SimpleNamespace(model=model, kv_bits=kv_bits)
    calls = []

    def _pre(prompt, *a, **k):
        calls.append(prompt)
        return {"input_ids": [0] * (tokens or len(prompt))}
    rg._preprocess_request = _pre
    rg._pre_calls = calls
    return rg


@pytest.fixture
def tight(monkeypatch):
    """A box where 100k tokens of DENSE KV (0.8 GB) does not fit."""
    monkeypatch.setattr(mp, "available_drained_bytes", lambda: 0.5e9)
    monkeypatch.setattr(
        "gmlx.gen.prefill_decay.score_transient_bytes",
        lambda model, pc, depth: 0.0)


def test_prompt_impossible_rejects(tight):
    rg = _rg(DENSE, tokens=100_000)
    with pytest.raises(PromptTooLongError) as e:
        mp.preflight_prompt_memory(rg, "x" * 200_000)
    assert "cannot fit" in str(e.value)
    assert "prompt_tokens=100000" in str(e.value)


def test_big_but_possible_admits(tight):
    rg = _rg(DENSE, tokens=50_000)  # 0.4 GB < 0.5 GB
    mp.preflight_prompt_memory(rg, "x" * 200_000)
    assert rg._pre_calls  # char bound failed, token count decided


def test_small_prompt_never_tokenizes(tight):
    rg = _rg(DENSE)
    mp.preflight_prompt_memory(rg, "short prompt")
    assert not rg._pre_calls


def test_pinned_max_impossible_rejects(tight, monkeypatch):
    monkeypatch.setattr(
        "mlx_vlm.server.generation.get_server_max_tokens", lambda: 512)
    rg = _rg(DENSE, tokens=30_000)  # 0.24 GB prompt
    args = SimpleNamespace(max_tokens=40_000)  # +0.32 GB pinned
    with pytest.raises(PromptTooLongError) as e:
        mp.preflight_prompt_memory(rg, "x" * 60_000, args=args)
    assert "max_tokens" in str(e.value)


def test_default_max_never_generation_rejects(tight, monkeypatch):
    monkeypatch.setattr(
        "mlx_vlm.server.generation.get_server_max_tokens", lambda: 40_000)
    rg = _rg(DENSE, tokens=30_000)
    args = SimpleNamespace(max_tokens=40_000)  # equals server default
    mp.preflight_prompt_memory(rg, "x" * 60_000, args=args)


def test_kill_switch(tight, monkeypatch):
    monkeypatch.setenv("GMLX_PREFLIGHT_MEM", "0")
    rg = _rg(DENSE, tokens=100_000)
    mp.preflight_prompt_memory(rg, "x" * 200_000)


def test_media_requests_skip(tight):
    rg = _rg(DENSE, tokens=100_000)
    mp.preflight_prompt_memory(rg, "x" * 200_000, images=["img"])


def test_no_model_skips(tight):
    rg = _rg(None)
    mp.preflight_prompt_memory(rg, "x" * 200_000)


def test_probe_failure_admits(tight, monkeypatch):
    rg = _rg(DENSE)

    def _boom(prompt, *a, **k):
        raise RuntimeError("tokenizer broke")

    rg._preprocess_request = _boom
    mp.preflight_prompt_memory(rg, "x" * 200_000)


def _stamp_policy(rg, layers=4, mtp=False):
    from mlx_vlm.models.cache import KVCache

    from gmlx.cache.kv_policy import resolve_kv_quant_policy
    from gmlx.serve.kv_policy import RG_ATTR, ServeKvPolicy

    kw = dict(kv_bits=8, kv_group_size=64, mtp=mtp)
    setattr(rg, RG_ATTR, ServeKvPolicy(
        resolve_kv_quant_policy(
            [KVCache() for _ in range(layers)], mode="single", **kw),
        resolve_kv_quant_policy(
            [KVCache() for _ in range(layers)], mode="batched", **kw)))
    return rg


def test_kv_policy_shrinks_the_estimate(tight):
    # 75k DENSE tokens: fp16 needs 0.61 GB (rejected), the resolved
    # 8-bit policy prices 3 packed + 1 held layers at 0.40 GB (admitted).
    rg = _rg(DENSE, kv_bits=8.0, tokens=75_000)
    with pytest.raises(PromptTooLongError):
        mp.preflight_prompt_memory(rg, "x" * 200_000)
    mp.preflight_prompt_memory(_stamp_policy(rg), "x" * 200_000)


def test_kv_bits_without_policy_prices_fp16():
    # rg.kv_bits is a float upstream; without a resolved policy the
    # pricing must not guess a packed rate.
    costs = mp._policy_costs(_rg(DENSE, kv_bits=8.0), DENSE)
    assert costs == [(None, 2048.0)] * 4


def test_policy_costs_price_per_layer():
    rg = _stamp_policy(_rg(DENSE, kv_bits=8.0))
    costs = mp._policy_costs(rg, DENSE)
    packed = 2 * 8 * 64 * 1.0625
    assert costs == [(None, packed)] * 3 + [(None, 2048.0)]


def test_mtp_policy_batched_prices_fp16():
    # Engagement is batch-dependent: MTP quantizes at B=1 and swaps to
    # fp16 when batched, so admission prices the batched mode.
    rg = _stamp_policy(_rg(DENSE, kv_bits=8.0), mtp=True)
    assert mp._policy_costs(rg, DENSE) == [(None, 2048.0)] * 4


def test_policy_layer_count_mismatch_falls_back():
    rg = _stamp_policy(_rg(DENSE, kv_bits=8.0), layers=6)
    assert mp._policy_costs(rg, DENSE) == [(None, 2048.0)] * 4


def test_install_wraps_both_and_is_idempotent(monkeypatch):
    from mlx_vlm.server.generation import ResponseGenerator as RG

    saved_gen, saved_val = RG.generate, RG.validate_context_budget
    try:
        mp.install_memory_preflight()
        g1, v1 = RG.generate, RG.validate_context_budget
        assert g1 is not saved_gen and v1 is not saved_val
        mp.install_memory_preflight()
        assert RG.generate is g1 and RG.validate_context_budget is v1
    finally:
        RG.generate, RG.validate_context_budget = saved_gen, saved_val
