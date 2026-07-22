"""Regression tests for the glm-dsa / DeepSeek-V3.2 sparse-attention indexer
correctness patches in ``loader`` - the root cause of GLM-5.2's "starts strong,
degrades at depth". CPU/model-free: builds a tiny random-weight stock
``glm_moe_dsa`` model and exercises the patches directly.

* ``_patch_dsv32_indexer_rope`` pins each indexer's RoPE to the stock + HF
  convention - **interleaved** (``traditional=True``), the same geometry as the
  main DeepSeek attention - and sets ``k_norm`` eps to HF's ``1e-6`` (mlx-lm
  inherits nn.LayerNorm's 1e-5). (An earlier build hypothesised a NeoX /
  half-split indexer rope; that was falsified - keeping it interleaved is what
  matches HF + llama.cpp and holds token-exact at depth. See dsv32_patches.py.)
* ``_patch_dsv32_indexer_fp32`` recomputes the top-k selection in fp32 (matching
  llama.cpp's fp32 indexer accumulation) and force-keeps the attention-sink +
  recent local window (the "sink drop" depth fix; ``GMLX_DSV32_SINK/_LOCAL``).
* ``_patch_dsv32_moe_gate_fp32`` upcasts the MoE router selection to fp32.

Each patch is checked for install/flag, kill-switch, idempotence, and numeric
fidelity against the stock forward.
"""

import contextlib
import os

import mlx.core as mx
import mlx.nn as nn
import pytest

import gmlx.dsv32_patches as dsv32_patches
from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v32 import Indexer, MoEGate
from mlx_lm.models.glm_moe_dsa import Model, ModelArgs


@pytest.fixture(autouse=True)
def _cpu_device():
    # CPU numerics by design; restore so the flip never leaks to other files.
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)

_BASE = 10000.0


@contextlib.contextmanager
def _env(**kw):
    """Set env vars for the block, restoring prior values on exit."""
    prev = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _tiny_args(index_topk: int = 4) -> ModelArgs:
    return ModelArgs.from_dict(
        {
            "model_type": "glm_moe_dsa",
            "vocab_size": 128,
            "hidden_size": 32,
            "index_head_dim": 8,
            "index_n_heads": 2,
            "index_topk": index_topk,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "routed_scaling_factor": 1.0,
            "kv_lora_rank": 16,
            "q_lora_rank": 24,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
            "qk_nope_head_dim": 8,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": True,
            "n_group": 1,
            "topk_group": 1,
            "num_experts_per_tok": 2,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": _BASE},
            "attention_bias": False,
        }
    )


def _build() -> Model:
    m = Model(_tiny_args())
    m.eval()
    mx.eval(m.parameters())
    return m


def _indexers(model: Model):
    return [m for m in model.modules() if isinstance(m, Indexer)]


def test_patch_flags_and_keeps_indexer_rope_interleaved():
    model = _build()
    idxs = _indexers(model)
    assert idxs

    dsv32_patches._patch_dsv32_indexer_rope(model)

    rope_dim = idxs[0].rope_head_dim
    head_dim = idxs[0].head_dim
    x = mx.random.normal((1, idxs[0].n_heads, 3, head_dim))
    for offset in (0, 7, 64):
        # Reference: stock / HF interleaved (GPT-J) RoPE on the first rope_dim dims.
        interleaved = nn.RoPE(rope_dim, traditional=True, base=_BASE)(x, offset=offset)
        neox = nn.RoPE(rope_dim, traditional=False, base=_BASE)(x, offset=offset)
        for ix in idxs:
            assert getattr(ix, "_dsv32_indexer_rope_fixed", False)
            got = ix.rope(x, offset=offset)
            assert float(mx.max(mx.abs(got - interleaved)).item()) < 1e-4
        if offset > 0:
            # ... and is genuinely interleaved, NOT the (falsified) NeoX layout.
            assert float(mx.max(mx.abs(got - neox)).item()) > 1e-2


def test_patch_is_idempotent():
    model = _build()
    dsv32_patches._patch_dsv32_indexer_rope(model)
    ropes = [id(ix.rope) for ix in _indexers(model)]
    dsv32_patches._patch_dsv32_indexer_rope(model)  # second call must not rebuild
    assert [id(ix.rope) for ix in _indexers(model)] == ropes


def test_kill_switch_skips_patch():
    model = _build()
    prev = os.environ.get("GMLX_DSV32_INDEXER_ROPE")
    os.environ["GMLX_DSV32_INDEXER_ROPE"] = "0"
    try:
        dsv32_patches._patch_dsv32_indexer_rope(model)
    finally:
        if prev is None:
            os.environ.pop("GMLX_DSV32_INDEXER_ROPE", None)
        else:
            os.environ["GMLX_DSV32_INDEXER_ROPE"] = prev
    assert not any(
        getattr(ix, "_dsv32_indexer_rope_fixed", False) for ix in _indexers(model)
    )


def test_rope_patch_also_sets_knorm_eps_to_1e6():
    model = _build()
    dsv32_patches._patch_dsv32_indexer_rope(model)
    for ix in _indexers(model):
        assert abs(ix.k_norm.eps - 1e-6) < 1e-12


def _indexer_topk(ix, seq_len: int, fp32: bool):
    """Run one indexer forward over ``seq_len`` tokens (> index_topk so the sparse
    branch fires) and return the last token's sorted top-k, in either the fp32
    override (fp32=True) or the stock path (fp32=False)."""
    ix._dsv32_indexer_fp32 = fp32
    x = mx.random.normal((1, seq_len, ix.dim), key=mx.random.key(3))
    qr = mx.random.normal((1, seq_len, ix.q_lora_rank), key=mx.random.key(4))
    sel = ix(x, qr, mask=None, cache=KVCache())
    return None if sel is None else mx.sort(sel[0, 0, -1]).tolist()


def test_fp32_patch_installs_and_flags_every_indexer():
    model = _build()
    n = len(_indexers(model))
    assert n > 0

    dsv32_patches._patch_dsv32_indexer_fp32(model)

    assert dsv32_patches._INDEXER_FP32_PATCH.installed
    assert Indexer.__call__ is dsv32_patches._dsv32_indexer_fp32_call
    assert sum(getattr(ix, "_dsv32_indexer_fp32", False) for ix in _indexers(model)) == n


def test_fp32_override_matches_stock_when_model_is_fp32():
    # With the sink/local force-keep disabled, the override differs from stock only
    # by dtype - a no-op on an fp32 model - so it must reproduce the stock selection
    # exactly. Guards the hand-copied indexer forward against mlx-lm drift / edits.
    model = _build()  # CPU default dtype is fp32
    with _env(GMLX_DSV32_SINK=0, GMLX_DSV32_LOCAL=0):
        dsv32_patches._patch_dsv32_indexer_fp32(model)
    ix = _indexers(model)[0]
    seq_len = ix.index_topk + 6
    stock = _indexer_topk(ix, seq_len, fp32=False)
    fixed = _indexer_topk(ix, seq_len, fp32=True)
    assert stock is not None and fixed is not None
    assert stock == fixed


def test_fp32_override_force_keeps_sink_and_local():
    # The override force-keeps the attention-sink (first `sink` keys) + a recent
    # local window (last `local` keys, query-relative) so the top-k never drops
    # them - the "sink drop" depth fix. With sink=1, local=2 the forced set is
    # smaller than index_topk, so every forced key must land in the selection.
    model = _build()
    with _env(GMLX_DSV32_SINK=1, GMLX_DSV32_LOCAL=2):
        dsv32_patches._patch_dsv32_indexer_fp32(model)
    ix = _indexers(model)[0]
    ix._dsv32_indexer_fp32 = True
    seq_len = ix.index_topk + 6
    x = mx.random.normal((1, seq_len, ix.dim), key=mx.random.key(3))
    qr = mx.random.normal((1, seq_len, ix.q_lora_rank), key=mx.random.key(4))
    sel = ix(x, qr, mask=None, cache=KVCache())
    assert sel is not None
    last = set(sel[0, 0, -1].tolist())
    forced = {0, seq_len - 2, seq_len - 1}  # sink {0} + local window {seq-2, seq-1}
    assert forced <= last


def test_fp32_kill_switch_skips_patch():
    model = _build()
    prev = os.environ.get("GMLX_DSV32_INDEXER_FP32")
    os.environ["GMLX_DSV32_INDEXER_FP32"] = "0"
    try:
        dsv32_patches._patch_dsv32_indexer_fp32(model)
    finally:
        if prev is None:
            os.environ.pop("GMLX_DSV32_INDEXER_FP32", None)
        else:
            os.environ["GMLX_DSV32_INDEXER_FP32"] = prev
    assert not any(
        getattr(ix, "_dsv32_indexer_fp32", False) for ix in _indexers(model)
    )


def _gates(model: Model):
    return [m for m in model.modules() if isinstance(m, MoEGate)]


def test_gate_fp32_patch_installs_and_flags_every_gate():
    model = _build()
    n = len(_gates(model))
    assert n > 0

    dsv32_patches._patch_dsv32_moe_gate_fp32(model)

    assert dsv32_patches._GATE_FP32_PATCH.installed
    assert MoEGate.__call__ is dsv32_patches._dsv32_moe_gate_fp32_call
    assert sum(getattr(g, "_dsv32_gate_fp32", False) for g in _gates(model)) == n


def test_gate_fp32_override_matches_stock_when_model_is_fp32():
    # On a fp32 model the upcast is a no-op -> override must reproduce the stock
    # routing exactly (guards the MoEGate reimplementation). Randomise the gate
    # weight/bias (MoEGate inits them to zeros) so routing is non-trivial.
    model = _build()
    dsv32_patches._patch_dsv32_moe_gate_fp32(model)
    g = _gates(model)[0]
    g.weight = mx.random.normal(g.weight.shape, key=mx.random.key(6))
    g.e_score_correction_bias = mx.random.normal(
        g.e_score_correction_bias.shape, key=mx.random.key(7)
    )
    x = mx.random.normal((1, 5, g.weight.shape[1]), key=mx.random.key(8))

    g._dsv32_gate_fp32 = False
    si, ss = g(x)
    g._dsv32_gate_fp32 = True
    fi, fs = g(x)
    assert bool(mx.array_equal(si, fi))
    assert float(mx.max(mx.abs(ss - fs)).item()) < 1e-5


def test_gate_fp32_kill_switch_skips_patch():
    model = _build()
    prev = os.environ.get("GMLX_DSV32_GATE_FP32")
    os.environ["GMLX_DSV32_GATE_FP32"] = "0"
    try:
        dsv32_patches._patch_dsv32_moe_gate_fp32(model)
    finally:
        if prev is None:
            os.environ.pop("GMLX_DSV32_GATE_FP32", None)
        else:
            os.environ["GMLX_DSV32_GATE_FP32"] = prev
    assert not any(getattr(g, "_dsv32_gate_fp32", False) for g in _gates(model))


def _moes(model: Model):
    from mlx_lm.models.deepseek_v32 import DeepseekV32MoE

    return [m for m in model.modules() if isinstance(m, DeepseekV32MoE)]


def test_moe_scores_patch_installs_and_flags_every_block():
    from mlx_lm.models.deepseek_v32 import DeepseekV32MoE

    model = _build()
    n = len(_moes(model))
    assert n > 0

    dsv32_patches._patch_dsv32_moe_scores(model)

    assert dsv32_patches._MOE_SCORES_PATCH.installed
    assert DeepseekV32MoE.__call__ is dsv32_patches._dsv32_moe_scores_call
    assert sum(
        getattr(b, "_dsv32_moe_scores", False) for b in _moes(model)) == n


def test_moe_scores_seam_matches_stock_and_feeds_sink():
    # Without a scores-taking switch the patched forward reproduces the stock
    # forward; a switch advertising _kq_scores_sink (the streaming offload
    # wrapper) receives the gate scores, and its unmixed return keeps the
    # stock python-side sum, so the output is unchanged either way.
    model = _build()
    dsv32_patches._patch_dsv32_moe_scores(model)
    blk = _moes(model)[0]
    g = blk.gate
    g.weight = mx.random.normal(g.weight.shape, key=mx.random.key(9))
    g.e_score_correction_bias = mx.random.normal(
        g.e_score_correction_bias.shape, key=mx.random.key(10))
    mx.eval(model.parameters())
    x = mx.random.normal((1, 3, g.weight.shape[1]), key=mx.random.key(11))

    blk._dsv32_moe_scores = False
    ref = blk(x)
    blk._dsv32_moe_scores = True
    assert bool(mx.allclose(blk(x), ref, atol=1e-6))

    seen = []
    inner = blk.switch_mlp

    class _Sink(nn.Module):
        _kq_scores_sink = True

        def __call__(self, x, inds, scores=None):
            seen.append(None if scores is None else mx.array(scores))
            return inner(x, inds)

    blk.switch_mlp = _Sink()
    out = blk(x)
    assert seen and seen[-1] is not None
    assert seen[-1].shape == (1, 3, blk.num_experts_per_tok)
    assert bool(mx.allclose(out, ref, atol=1e-6))


def test_moe_scores_kill_switch_skips_patch():
    model = _build()
    prev = os.environ.get("GMLX_DSV32_MOE_MIX")
    os.environ["GMLX_DSV32_MOE_MIX"] = "0"
    try:
        dsv32_patches._patch_dsv32_moe_scores(model)
    finally:
        if prev is None:
            os.environ.pop("GMLX_DSV32_MOE_MIX", None)
        else:
            os.environ["GMLX_DSV32_MOE_MIX"] = prev
    assert not any(
        getattr(b, "_dsv32_moe_scores", False) for b in _moes(model))
