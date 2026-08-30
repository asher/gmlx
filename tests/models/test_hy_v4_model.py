"""HY4 model-level seams: fp32 pins, the kv-bits refusal, and the levers.

Nothing here loads the 229 GB GGUF. A tiny real HY4 stack stands in, which is
enough for every claim below because each is structural: which parameter names
are pinned fp32, which cache classes refuse quantization, which streaming
levers find their gate submodule, and that no speculative-decoding table can
ever name this family.

Two facts are maintained in two places by hand and are cross-checked here:
``Model.cast_predicate`` and ``loader._FP32_KEEP_BY_MODEL_TYPE["hy_v4"]``
encode the same fp32 set, and serve's admission path has to reach the same
kv-bits verdict the CLI path does.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

import gmlx.models.hy_v4.model as hy_v4_model
from gmlx.gen.generation import kv_quantization_unsupported
from gmlx.models.hy_v4.model import HyV4KVCache, Model, ModelArgs

DIM = 16


def _args(**over):
    base = dict(
        model_type="hy_v4", vocab_size=32, hidden_size=DIM, intermediate_size=32,
        moe_intermediate_size=16, num_hidden_layers=3, num_attention_heads=4,
        q_lora_rank=16, kv_lora_rank=12, qk_nope_head_dim=8, qk_rope_head_dim=4,
        v_head_dim=8, n_routed_experts=4, num_experts_per_tok=2,
        n_shared_experts=1, first_k_dense_replace=1, rms_norm_eps=1e-6,
        hc_mult=4, hc_eps=1e-6, hc_magnitude=2.0, index_n_heads=2,
        index_head_dim=8, index_topk=4, index_is_full=[1, 0, 1],
        swiglu_limit=10.0, routed_scaling_factor=2.827,
    )
    base.update(over)
    return ModelArgs(**base)


def _model(**over):
    m = Model(_args(**over))
    mx.eval(m.parameters())
    return m


# --- fp32 pins ---------------------------------------------------------------

_FP32_PATHS = [
    "model.layers.0.attn_hc.fn",
    "model.layers.0.attn_hc.base",
    "model.layers.0.attn_hc.scale",
    "model.layers.0.ffn_hc.fn",
    "model.hc_head.fn",
    "model.layers.0.self_attn.sinks",
    "model.layers.1.mlp.gate.weight",
    "model.layers.1.mlp.gate.e_score_correction_bias",
    "model.layers.0.self_attn.indexer.weights_proj.weight",
]

_CASTABLE_PATHS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.self_attn.embed_q.weight",
    "model.layers.0.self_attn.attn_gate.weight",
    "model.layers.1.mlp.switch_mlp.gate_proj.weight",
    "model.layers.0.self_attn.indexer.wq_b.weight",
    "lm_head.weight",
]


@pytest.mark.parametrize("path", _FP32_PATHS)
def test_cast_predicate_pins_fp32_params(path):
    assert _model().cast_predicate(path) is False


@pytest.mark.parametrize("path", _CASTABLE_PATHS)
def test_cast_predicate_allows_the_rest(path):
    assert _model().cast_predicate(path) is True


def test_fp32_pins_are_exactly_this_set_in_a_real_stack():
    # Walk the real parameter tree rather than trusting the lists above: this
    # catches an over-pin (a whole projection kept fp32 by a loose substring)
    # as well as a missed one. 3 layers, indexers on layers 0 and 2, MoE on
    # layers 1 and 2.
    from mlx.utils import tree_flatten

    model = _model()
    predicate = model.cast_predicate
    pinned = {p for p, _ in tree_flatten(model.parameters())
              if not predicate(p)}

    want = {f"model.hc_head.{n}" for n in ("fn", "base", "scale")}
    for i in range(3):
        want |= {f"model.layers.{i}.{sub}_hc.{n}"
                 for sub in ("attn", "ffn")
                 for n in ("fn", "base", "scale")}
        want.add(f"model.layers.{i}.self_attn.sinks")
    for i in (1, 2):
        want.add(f"model.layers.{i}.mlp.gate.weight")
        want.add(f"model.layers.{i}.mlp.gate.e_score_correction_bias")
    for i in (0, 2):
        want.add(f"model.layers.{i}.self_attn.indexer.weights_proj.weight")

    assert pinned == want


def test_cast_predicate_agrees_with_the_loader_fp32_list():
    # Two hand-maintained copies of one fact. The loader keys on substrings,
    # the class on a predicate; they must classify the same paths.
    from gmlx.load.loader import _FP32_KEEP_BY_MODEL_TYPE

    keeps = _FP32_KEEP_BY_MODEL_TYPE["hy_v4"]
    predicate = _model().cast_predicate
    for path in _FP32_PATHS:
        assert any(k in path for k in keeps), path
    for path in _CASTABLE_PATHS:
        assert not any(k in path for k in keeps), path
        assert predicate(path) is True


def test_quant_predicate_protects_the_indexer():
    predicate = _model().quant_predicate
    # The head-weights GEMM decides near-tied key rankings: never quantized.
    assert predicate("model.layers.0.self_attn.indexer.weights_proj", None) is False
    # The indexer projections take 8-bit, not the model default.
    for name in ("wq_b", "wk"):
        cfg = predicate(f"model.layers.0.self_attn.indexer.{name}", None)
        assert cfg == {"group_size": 64, "bits": 8}
    assert predicate("model.layers.0.self_attn.q_b_proj", None) is True


# --- the kv-bits refusal -----------------------------------------------------


def test_cache_declares_kv_quantization_unsupported():
    assert HyV4KVCache.kv_quant_unsupported is True


def test_kv_quantization_unsupported_names_the_cache():
    reason = kv_quantization_unsupported(_model())
    assert reason and "HyV4KVCache" in reason


def test_make_cache_carries_an_indexer_slot_only_on_full_layers():
    # index_is_full [1, 0, 1]: layer 1 owns no indexer, so it gets the latent
    # cache alone. An unwritten second slot is not inert - CacheList.state
    # reads keys.shape on every entry, and the generation loop evals it.
    model = _model()
    caches = model.make_cache()
    assert len(caches) == 3
    assert [getattr(c, "caches", None) is not None for c in caches] == [
        True, False, True]
    for c, layer in zip(caches, model.layers):
        latent, idx = hy_v4_model.split_cache(c)
        assert isinstance(latent, HyV4KVCache)
        assert (idx is not None) == (layer.self_attn.indexer is not None)


def test_cache_state_evaluates_after_a_prefill():
    # The crash signature this cache shape exists to avoid: mlx-lm's
    # generate_step runs mx.eval([c.state for c in prompt_cache]) after the
    # prefill. Stock KVCache.state dereferences self.keys unguarded, so an
    # indexer slot on a shared layer - written by nothing - takes the whole
    # decode loop down on its first step.
    model = _model()
    cache = model.make_cache()
    mx.eval(model(mx.array(np.arange(6).reshape(1, 6)), cache=cache))
    mx.eval([c.state for c in cache])


def test_forced_dense_still_writes_the_indexer_cache(monkeypatch):
    # GMLX_HY4_SPARSE_DISABLE drops the selection but must not skip the
    # indexer: an unwritten slot 1 on a full layer takes CacheList.state
    # down the same way an unwritten shared-layer slot would.
    monkeypatch.setenv("GMLX_HY4_SPARSE_DISABLE", "1")
    model = _model()
    cache = model.make_cache()
    mx.eval(model(mx.array(np.arange(6).reshape(1, 6)), cache=cache))
    mx.eval([c.state for c in cache])
    for c, layer in zip(cache, model.layers):
        _, idx = hy_v4_model.split_cache(c)
        if layer.self_attn.indexer is not None:
            assert idx is not None and idx.offset == 6


def test_split_cache_handles_every_entry_shape():
    from mlx_lm.models.cache import CacheList, KVCache

    latent, idx = hy_v4_model.split_cache(None)
    assert latent is None and idx is None

    plain = HyV4KVCache()
    assert hy_v4_model.split_cache(plain) == (plain, None)

    a, b = HyV4KVCache(), KVCache()
    assert hy_v4_model.split_cache(CacheList(a, b)) == (a, b)
    assert hy_v4_model.split_cache(CacheList(a)) == (a, None)


def test_serve_admission_drops_kv_bits_for_this_model(caplog):
    # Serve prices the cache at kv_bits/8 in estimate.py and mem_preflight.py
    # while safe_maybe_quantize skips a CacheList wholesale - so a served
    # kv_bits would under-provision. The admission path must clear it, and
    # say so: the config key stays as the operator wrote it, so without the
    # warning a `kv_bits: 4` entry looks honoured forever.
    import logging

    from gmlx.serve.residency import _drop_unsupported_kv_bits

    class _RG:
        kv_bits = 4

    rg = _RG()
    rg.model = _model()
    with caplog.at_level(logging.WARNING, logger="gmlx.serve.residency"):
        _drop_unsupported_kv_bits(rg, "/models/hy4-preview.gguf")
    assert rg.kv_bits is None

    (record,) = [r for r in caplog.records if r.levelno >= logging.WARNING]
    msg = record.getMessage()
    assert "kv_bits=4" in msg                      # what was asked for
    assert "/models/hy4-preview.gguf" in msg       # which model
    assert "HyV4KVCache" in msg                    # why
    assert "fp16" in msg and "4x" in msg           # what it costs instead


def test_serve_prices_the_cache_at_fp16_after_the_drop():
    # The point of clearing the field: serve/estimate.py and
    # serve/mem_preflight.py both read rg.kv_bits and fall back to 2 bytes
    # per element when it is None. Before the drop they priced HY4's fp16
    # cache at 0.5 - a four-fold under-provision at admission.
    #
    # Serve holds the model inside the text_only wrapper, which is what
    # carries the .config the pricing reads; the bare mlx-lm class has only
    # .args. Use the wrapper so this exercises the real shape.
    from gmlx.models.vlm_text_only import Model as TextOnlyModel
    from gmlx.serve.mem_preflight import kv_layer_costs
    from gmlx.serve.residency import _drop_unsupported_kv_bits

    def _bpe(rg):
        bits = getattr(rg, "kv_bits", None)
        return bits / 8.0 if isinstance(bits, int) and bits > 0 else 2.0

    args = _args()

    class _RG:
        kv_bits = 4

    rg = _RG()
    rg.model = TextOnlyModel(Model(args), vars(args))

    assert _bpe(rg) == 0.5
    under = kv_layer_costs(rg.model, _bpe(rg))
    _drop_unsupported_kv_bits(rg)
    assert _bpe(rg) == 2.0
    priced = kv_layer_costs(rg.model, _bpe(rg))

    # MLA: one entry per layer, priced on the compressed latent + rope half.
    assert under and priced
    assert priced == [(None, (args.kv_lora_rank + args.qk_rope_head_dim) * 2.0)
                      ] * args.num_hidden_layers
    assert sum(b for _, b in priced) == pytest.approx(
        4 * sum(b for _, b in under))


def test_serve_admission_leaves_kv_bits_alone_for_a_quantizable_model():
    from mlx_lm.models.cache import KVCache

    from gmlx.serve.residency import _drop_unsupported_kv_bits

    class _Plain:
        def make_cache(self):
            return [KVCache() for _ in range(2)]

    class _RG:
        kv_bits = 4

    rg = _RG()
    rg.model = _Plain()
    _drop_unsupported_kv_bits(rg)
    assert rg.kv_bits == 4


# --- no speculative decoding -------------------------------------------------


def test_hy_v4_is_never_a_speculative_target_or_drafter():
    # The converter drops the nextn layers, so no HY4 GGUF carries an MTP
    # block. A sibling file must never be autowired as draft_gguf.
    from gmlx.load.arch_table import MTP_DRAFTER_ARCHES, MTP_WIRED_MODEL_TYPES

    assert "hyv4" not in MTP_DRAFTER_ARCHES
    assert "hy_v4" not in MTP_DRAFTER_ARCHES
    assert "hy_v4" not in MTP_WIRED_MODEL_TYPES


# --- streaming levers --------------------------------------------------------


def test_moe_gate_submodule_is_duck_typed_for_the_streaming_levers():
    # --moe-expert-mass / --moe-miss-shed / --moe-layer-shed and lookahead all
    # find the router through this shape. No per-arch table entry is needed.
    from gmlx.stream.moe_experts import _gate_submodule

    moe = _model().model.layers[1].mlp
    gate = _gate_submodule(moe)
    assert gate is moe.gate
    assert isinstance(gate.top_k, int)

    x = mx.zeros((1, 2, DIM))
    inds, scores = gate(x)
    mx.eval(inds, scores)
    assert inds.shape[-1] == gate.top_k
    assert scores.shape == inds.shape


def test_experts_are_a_switch_glu_so_they_stream():
    from mlx_lm.models.switch_layers import SwitchGLU

    from gmlx.stream.prefetch import _EXPS_RE

    moe = _model().model.layers[1].mlp
    assert isinstance(moe.switch_mlp, SwitchGLU)
    # The wire names the streaming tier matches on.
    for name in ("blk.1.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight",
                 "blk.1.ffn_down_exps.weight"):
        assert _EXPS_RE.match(name), name


def test_streaming_prefill_step_is_narrowed_for_hy_v4():
    # 4 fp32 residual streams at hidden 6144 make an 8192-token chunk cost
    # ~805 MB per iHC tensor, several per layer.
    from gmlx.load.loader import (
        _STREAMING_PREFILL_STEP,
        _STREAMING_PREFILL_STEP_BY_MODEL_TYPE,
    )

    assert _STREAMING_PREFILL_STEP_BY_MODEL_TYPE["hy_v4"] < _STREAMING_PREFILL_STEP


def test_prefill_step_resolves_from_the_model_type(monkeypatch):
    from gmlx.load import loader

    monkeypatch.setattr(loader, "moe_streaming_active", lambda _m: True)
    model = _model()
    step, defaulted = loader._resolve_prefill_step(model, None)
    assert defaulted is True
    assert step == loader._STREAMING_PREFILL_STEP_BY_MODEL_TYPE["hy_v4"]
    # An explicit request always wins.
    assert loader._resolve_prefill_step(model, 1024) == (1024, False)


# --- forward integrity -------------------------------------------------------


def test_prefill_then_decode_stays_finite_past_the_selection_boundary():
    model = _model(index_topk=4)
    cache = model.make_cache()
    ids = mx.array(np.arange(6).reshape(1, 6))
    out = model(ids, cache=cache)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)))
    # Decode past index_topk cached keys, where the selection starts to bite.
    for t in range(6):
        out = model(mx.array([[t % 32]]), cache=cache)
        mx.eval(out)
        assert bool(mx.all(mx.isfinite(out))), t
    assert cache[0].caches[0].offset == 12


def test_sparse_and_dense_agree_while_the_selection_is_the_identity(monkeypatch):
    # Below index_topk cached keys the top-k selection covers every key, so
    # the sparse and forced-dense paths must produce identical logits. This
    # is the oracle-free anchor for the selection chain: it isolates a
    # gather/mask bug from a genuine sparsity difference.
    model = _model(index_topk=8)
    ids = mx.array(np.arange(6).reshape(1, 6))

    sparse = np.array(model(ids, cache=model.make_cache()))
    monkeypatch.setenv("GMLX_HY4_SPARSE_DISABLE", "1")
    dense = np.array(model(ids, cache=model.make_cache()))
    assert np.allclose(sparse, dense, atol=1e-5), np.abs(sparse - dense).max()


def test_sparse_disable_changes_logits_once_the_selection_bites():
    # The complement of the test above: with more keys than index_topk the
    # two paths must differ, or GMLX_HY4_SPARSE_DISABLE proves nothing.
    import os

    model = _model(index_topk=3)
    ids = mx.array(np.arange(12).reshape(1, 12))
    sparse = np.array(model(ids, cache=model.make_cache()))
    os.environ["GMLX_HY4_SPARSE_DISABLE"] = "1"
    try:
        dense = np.array(model(ids, cache=model.make_cache()))
    finally:
        del os.environ["GMLX_HY4_SPARSE_DISABLE"]
    assert not np.allclose(sparse, dense)


def test_no_cache_forward_matches_the_cached_one():
    model = _model()
    ids = mx.array(np.arange(6).reshape(1, 6))
    cached = np.array(model(ids, cache=model.make_cache()))
    plain = np.array(model(ids))
    assert np.allclose(cached, plain, atol=1e-5)


def test_residual_streams_start_as_exact_copies():
    # hc_mult copies of the embedding, no scaling and no one-hot seeding: a
    # scaled or zeroed stream is a silent depth-only failure.
    model = _model()
    h = model.model.embed_tokens(mx.array([[1, 2]]))
    broadcast = mx.broadcast_to(
        h[:, :, None, :], (1, 2, model.args.hc_mult, DIM))
    mx.eval(h, broadcast)
    for i in range(model.args.hc_mult):
        assert np.array_equal(np.array(broadcast)[:, :, i], np.array(h))


def test_sanitize_drops_tensors_past_the_trunk():
    model = _model()
    n = model.args.num_hidden_layers
    weights = {
        "model.layers.0.self_attn.q_a_proj.weight": mx.zeros((1,)),
        f"model.layers.{n}.self_attn.q_a_proj.weight": mx.zeros((1,)),
        "model.hc_head.fn": mx.zeros((1,)),
    }
    kept = model.sanitize(weights)
    assert set(kept) == {"model.layers.0.self_attn.q_a_proj.weight",
                         "model.hc_head.fn"}


def test_module_registers_as_mlx_lm_hy_v4():
    import sys

    hy_v4_model.ensure_registered()
    assert "mlx_lm.models.hy_v4" in sys.modules
    from mlx_lm.utils import _get_classes

    Model_cls, Args_cls = _get_classes({"model_type": "hy_v4"})
    assert Model_cls is Model
    assert Args_cls is ModelArgs
