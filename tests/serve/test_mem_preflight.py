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
    # 75k dense tokens: fp16 pricing rejects, the resolved 8-bit
    # policy admits.
    rg = _rg(DENSE, kv_bits=8.0, tokens=75_000)
    with pytest.raises(PromptTooLongError):
        mp.preflight_prompt_memory(rg, "x" * 200_000)
    mp.preflight_prompt_memory(_stamp_policy(rg), "x" * 200_000)


def test_kv_bits_without_policy_prices_fp16():
    # rg.kv_bits is a float upstream. Without a resolved policy the
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


# Recurrent-state geometry. The three byte figures are the cache arrays
# measured on the loaded models (fp32 state, bf16 conv tails).
GDN_CFG = dict(model_type="qwen3_5", num_hidden_layers=32,
               num_attention_heads=16, num_key_value_heads=4, head_dim=256,
               full_attention_interval=4, linear_num_value_heads=32,
               linear_num_key_heads=16, linear_key_head_dim=128,
               linear_value_head_dim=128, linear_conv_kernel_dim=4)
GDN_STATE = 2097152 + 49152          # Qwen3.5-9B
FALCON_CFG = dict(model_type="falcon_h1", num_hidden_layers=36,
                  num_attention_heads=8, num_key_value_heads=2, head_dim=64,
                  mamba_n_heads=24, mamba_d_head=64, mamba_d_state=128,
                  mamba_d_conv=4, mamba_n_groups=1)
FALCON_STATE = 786432 + 10752        # Falcon-H1-0.5B
NEMO_PATTERN = list("MEME*EMEM*EE")
NEMO_CFG = dict(model_type="nemotron_h", num_hidden_layers=12,
                num_attention_heads=32, num_key_value_heads=2, head_dim=128,
                mamba_num_heads=64, mamba_head_dim=64, ssm_state_size=128,
                conv_kernel=4, n_groups=8, hybrid_override_pattern=NEMO_PATTERN)
NEMO_STATE = 2097152 + 36864         # Nemotron-3.5-Lightning
KDA_CFG = dict(model_type="kimi_k3", num_hidden_layers=4,
               num_attention_heads=4, kv_lora_rank=512, qk_rope_head_dim=64,
               kda_head_dim=128, ssm_conv_kernel=4,
               layer_types=["linear_attention", "linear_attention",
                            "linear_attention", "full_attention"])
KDA_STATE = 3 * 3 * 512 * 2 + 4 * 128 * 128 * 4


def test_fixed_rows_charge_once():
    assert mp.span_tokens(mp.FixedRows(1), 0) == 0
    assert mp.span_tokens(mp.FixedRows(1), 1) == 1
    assert mp.span_tokens(mp.FixedRows(1), 100_000) == 1
    costs = [(None, 8.0), (mp.FixedRows(1), 1e6), (128, 2.0)]
    assert mp.per_token_bytes(costs) == 10.0
    assert mp.prompt_kv_bytes(costs, 1000) == 8000.0 + 1e6 + 256.0


def test_recurrent_state_bytes_per_family():
    assert mp.recurrent_state_bytes(SimpleNamespace(**GDN_CFG)) == GDN_STATE
    assert mp.recurrent_state_bytes(SimpleNamespace(**FALCON_CFG)) == FALCON_STATE
    assert mp.recurrent_state_bytes(SimpleNamespace(**NEMO_CFG)) == NEMO_STATE
    assert mp.recurrent_state_bytes(SimpleNamespace(**KDA_CFG)) == KDA_STATE
    assert mp.recurrent_state_bytes(DENSE.config) is None


def test_gdn_layers_hold_state_and_only_attention_grows():
    costs = mp.kv_layer_costs(_model(**GDN_CFG))
    attn = [c for c in costs if c[0] is None]
    state = [c for c in costs if isinstance(c[0], mp.FixedRows)]
    assert len(attn) == 8 and attn[0] == (None, 2 * 4 * 256 * 2.0)
    assert len(state) == 24 and state[0] == (mp.FixedRows(1), GDN_STATE)
    # 32k of context: 1.05 GiB, where uniform growth said 4 GiB
    assert mp.prompt_kv_bytes(costs, 32768) == (
        8 * 4096 * 32768 + 24 * GDN_STATE)
    assert mp.per_token_bytes(costs) == 8 * 4096


def test_nemotron_pattern_skips_mlp_blocks():
    geo = mp.config_geometry(SimpleNamespace(**NEMO_CFG))
    # M and * blocks own a cache; E blocks own none
    assert len(geo) == 6
    assert [g.attn for g in geo] == [False, False, True, False, False, True]
    assert all(g.state == NEMO_STATE for g in geo if not g.attn)
    costs = mp.kv_layer_costs(_model(**NEMO_CFG))
    assert mp.per_token_bytes(costs) == 2 * (2 * 2 * 128 * 2.0)


def test_falcon_layers_grow_and_hold_state():
    costs = mp.kv_layer_costs(_model(**FALCON_CFG))
    assert len(costs) == 72
    assert costs[0] == (None, 2 * 2 * 64 * 2.0)
    assert costs[1] == (mp.FixedRows(1), FALCON_STATE)


def test_kda_layer_types_with_mla_latent():
    costs = mp.kv_layer_costs(_model(**KDA_CFG))
    assert costs == [(mp.FixedRows(1), KDA_STATE)] * 3 + [(None, 576 * 2.0)]


def test_unsized_recurrent_family_prices_growth():
    # layer_types names recurrent blocks but no family key sizes them
    m = _model(num_hidden_layers=2, num_attention_heads=8,
               num_key_value_heads=2, head_dim=64,
               layer_types=["mamba", "attention"])
    assert mp.kv_layer_costs(m) == [(None, 512.0)] * 2


def test_stack_geometry_reads_the_caches():
    from mlx_vlm.models.cache import (ArraysCache, CacheList, KVCache,
                                      RotatingKVCache)

    stack = [ArraysCache(size=2), KVCache(), RotatingKVCache(max_size=128),
             CacheList(ArraysCache(size=2), KVCache())]
    geo = mp.stack_geometry(_model(**GDN_CFG), stack)
    assert geo == [mp.LayerGeometry(False, None, GDN_STATE),
                   mp.LayerGeometry(True, None, 0.0),
                   mp.LayerGeometry(True, 128, 0.0),
                   mp.LayerGeometry(True, None, GDN_STATE)]


def test_policy_costs_follow_the_stack_not_the_config():
    # nemotron_h: 12 config layers, a 4-entry stack. The policy's
    # per-layer vector is stack-indexed and must price by stack index.
    from mlx_vlm.models.cache import ArraysCache, KVCache

    from gmlx.cache.kv_policy import resolve_kv_quant_policy
    from gmlx.serve.kv_policy import RG_ATTR, ServeKvPolicy

    def make():
        return [ArraysCache(size=2), KVCache(), KVCache(), KVCache()]

    model = SimpleNamespace(config=SimpleNamespace(**NEMO_CFG), make_cache=make)
    rg = _rg(model, kv_bits=8.0)
    kw = dict(kv_bits=8, kv_group_size=64)
    setattr(rg, RG_ATTR, ServeKvPolicy(
        resolve_kv_quant_policy(make(), mode="single", **kw),
        resolve_kv_quant_policy(make(), mode="batched", **kw)))
    costs = mp._policy_costs(rg, model)
    per_tok = 2 * 2 * 128
    assert costs == [(mp.FixedRows(1), NEMO_STATE),
                     (None, per_tok * 1.0625), (None, per_tok * 1.0625),
                     (None, per_tok * 2.0)]
    # no policy: the stack still decides what grows
    assert mp._policy_costs(_rg(model), model) == [
        (mp.FixedRows(1), NEMO_STATE)] + [(None, per_tok * 2.0)] * 3
