"""Serve-side KV policy resolve at model load."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_vlm")

from mlx_vlm.models.cache import KVCache  # noqa: E402

import gmlx.serve.kv_policy as skv  # noqa: E402
from gmlx.cache.kv_policy import kv_line  # noqa: E402


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


def test_kvarn_resolve_prices_record_and_regions(monkeypatch):
    from gmlx.cache import kvarn_sdpa
    from gmlx.cache.kv_policy import kvarn_fixed_tokens

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    monkeypatch.setenv("KV_BITS", "6")
    monkeypatch.setenv("KV_TAIL_TOKENS", "1024")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    model = _model()
    model.config.head_dim = 128
    pol = skv.resolve_for_load(_rg(model=model, kv_bits=6.0,
                                   kv_quant_scheme="kvarn"), "m")
    assert pol.single.verdict == "full"
    assert pol.pricing_vector() == [0.796875] * 3 + [2.0]
    rows = kvarn_fixed_tokens(1024)
    assert pol.region_vector() == [((rows, 2.0),)] * 3 + [()]
    assert pol.step_vector() == [4096] * 3 + [0]
    j = pol.to_json()
    assert j["scheme"] == "kvarn" and j["value_bits"] == 6
    assert j["tail_tokens"] == 1024


def test_kvarn_notes_only_an_explicit_start_offset(monkeypatch):
    from gmlx.cache import kvarn_sdpa
    from gmlx.cache.kv_policy import kv_line

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    monkeypatch.delenv("QUANTIZED_KV_START", raising=False)
    monkeypatch.setenv("KV_BITS", "6")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    model = _model()
    model.config.head_dim = 128
    # upstream's default rides on rg whether or not anyone asked
    rg = _rg(model=model, kv_bits=None, kv_quant_scheme="kvarn",
             quantized_kv_start=5000)
    pol = skv.resolve_for_load(rg, "m")
    assert pol.single.verdict == "full" and pol.single.start_honored
    assert "not honored" not in kv_line("m", pol.single)
    monkeypatch.setenv("QUANTIZED_KV_START", "512")
    pol = skv.resolve_for_load(_rg(model=model, kv_bits=None,
                                   kv_quant_scheme="kvarn",
                                   quantized_kv_start=512), "m")
    assert not pol.single.start_honored
    assert "quantized_kv_start=512 not honored" in kv_line("m", pol.single)


def test_malformed_tail_fails_the_boot(monkeypatch):
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    monkeypatch.setenv("KV_BITS", "6")
    monkeypatch.setenv("KV_TAIL_TOKENS", "lots")
    model = _model()
    model.config.head_dim = 128
    rg = _rg(model=model, kv_bits=None, kv_quant_scheme="kvarn")
    with pytest.raises(skv.KvPolicyError, match="KV_TAIL_TOKENS='lots'"):
        skv.resolve_for_load(rg, "m")


def test_affine_start_reads_the_load_window(monkeypatch):
    # upstream froze rg.quantized_kv_start from the process env at server
    # start; the per-model window must reach the generator.
    monkeypatch.setenv("KV_BITS", "8")
    monkeypatch.setenv("QUANTIZED_KV_START", "512")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    rg = _rg(quantized_kv_start=5000)
    pol = skv.resolve_for_load(rg, "m")
    assert rg.quantized_kv_start == 512
    assert pol.single.quantized_kv_start == 512
    monkeypatch.setenv("QUANTIZED_KV_START", "soon")
    with pytest.raises(skv.KvPolicyError, match="QUANTIZED_KV_START"):
        skv.resolve_for_load(_rg(), "m")


def test_kvarn_split_widths_follow_the_key_value_config(monkeypatch):
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    monkeypatch.setenv("KV_BITS", "6")
    monkeypatch.setenv("GMLX_KVARN_BITS", "k4v4")
    model = _model()
    model.config.head_dim = 128
    rg = _rg(model=model, kv_bits=None, kv_quant_scheme="kvarn",
             kv_key_bits=8.0, kv_value_bits=5.0)
    pol = skv.resolve_for_load(rg, "m")
    assert (pol.single.bits, pol.single.value_bits) == (8, 5)
    assert "kvarn k8 v5" in skv.kv_line("m", pol.single)


def test_kvarn_admission_charges_the_fp16_buffers(monkeypatch):
    """The record bpe alone underprices a kvarn layer: the fp16 sink,
    horizon and tail are resident from the first token."""
    import gmlx.serve.mem_preflight as mp
    from gmlx.cache import kvarn_sdpa
    from gmlx.cache.kv_policy import kvarn_fixed_tokens

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    monkeypatch.setenv("KV_BITS", "6")
    monkeypatch.setenv("KV_TAIL_TOKENS", "1024")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    model = _model()
    model.config.head_dim = 128
    rg = _rg(model=model, kv_bits=6.0, kv_quant_scheme="kvarn")
    skv.resolve_for_load(rg, "m")
    costs = mp._policy_costs(rg, model)
    elems = 2 * 8 * 128
    rows = kvarn_fixed_tokens(1024)
    assert costs[:4] == [(4096, elems * 0.796875)] * 3 + [(None, elems * 2.0)]
    assert all(isinstance(w, mp.StepTokens) for w, _ in costs[:3])
    assert costs[4:] == [(rows, elems * 2.0)] * 3
    assert all(isinstance(w, mp.FixedRows) for w, _ in costs[4:])
    # One token allocates a full code slab per layer plus the fp16 rows.
    assert mp.prompt_kv_bytes(costs, 1) == (
        3 * elems * 0.796875 * 4096 + elems * 2.0 + 3 * elems * 2.0 * rows)
    assert mp.prompt_kv_bytes(costs, 8192) > mp.prompt_kv_bytes(costs[:4], 8192)


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
    rg = _rg(kv_bits=None)
    assert skv.resolve_for_load(rg, "m") is None
    # /v1/models reports no policy, but the model carries an explicit
    # off stamp so request-time readers never inherit another model's
    # boot env.
    assert not hasattr(rg, skv.RG_ATTR)
    stamped = getattr(rg.model, skv.RG_ATTR)
    assert stamped.single.verdict == "off" and stamped.single.bits is None
    assert "-> off" in kv_line("m", stamped.single)


def test_to_json_shapes(monkeypatch):
    monkeypatch.setenv("KV_BITS", "8")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    j = skv.resolve_for_load(_rg(), "m").to_json()
    assert j == {"scheme": "uniform", "bits": 8, "group_size": 64,
                 "layers_quantized": 3,
                 "layers_fp16": 1, "verdict": "full",
                 "verdict_batched": "full"}
    monkeypatch.setenv("MLX_VLM_GGUF_SPECULATIVE", "1")
    j = skv.resolve_for_load(_rg(), "m").to_json()
    assert j["verdict"] == "full"
    assert j["verdict_batched"] == "dropped"
    assert "batched_reason" in j


def test_load_window_scheme_wins_over_the_frozen_generator(monkeypatch):
    """Upstream freezes the scheme from the process env at server start,
    so a per-model load: key reaches only the env window. Reading it here
    is what makes kv_quant_scheme work per model -- and rg must be
    corrected, since upstream's batch build gates on the attribute."""
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    monkeypatch.setenv("KV_BITS", "6")
    monkeypatch.setenv("KV_QUANT_SCHEME", "kvarn")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    model = _model()
    model.config.head_dim = 128
    rg = _rg(model=model, kv_bits=6.0, kv_quant_scheme="uniform")
    pol = skv.resolve_for_load(rg, "m")
    assert pol.single.scheme == "kvarn"
    assert rg.kv_quant_scheme == "kvarn"
    assert pol.to_json()["scheme"] == "kvarn"


def test_kvarn_engages_without_kv_bits(monkeypatch):
    # The scheme alone requests kvarn at its default width; upstream's
    # kv_bits parse (and its qat drop) does not gate it.
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    monkeypatch.delenv("KV_BITS", raising=False)
    monkeypatch.setenv("KV_QUANT_SCHEME", "kvarn")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    model = _model()
    model.config.head_dim = 128
    rg = _rg(model=model, kv_bits=None, kv_quant_scheme="kvarn")
    pol = skv.resolve_for_load(rg, "m-qat")
    assert pol is not None and getattr(rg, skv.RG_ATTR) is pol
    assert pol.single.scheme == "kvarn" and pol.single.bits == 6
    assert pol.single.verdict in ("full", "partial")
    # affine keeps the off-when-unset contract
    monkeypatch.delenv("KV_QUANT_SCHEME")
    assert skv.resolve_for_load(_rg(kv_bits=None), "m") is None


def test_generator_scheme_is_kept_without_an_env_window(monkeypatch):
    monkeypatch.delenv("KV_QUANT_SCHEME", raising=False)
    monkeypatch.setenv("KV_BITS", "8")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    rg = _rg()
    assert skv.resolve_for_load(rg, "m").single.scheme == "uniform"
    assert rg.kv_quant_scheme == "uniform"
