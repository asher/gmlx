# SPDX-License-Identifier: MIT
"""K2 Horizon model against a NumPy reference of llama.cpp's graph
(src/models/k2-horizon.cpp at 7182ef71e): group RMSNorm, NeoX rope, the
log2(1 + 2^g) output gate, MoVA value routing and the sigmoid MoE with a
shared expert."""

from __future__ import annotations

import math

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.models.k2_horizon.model import (
    K2HorizonMoE,
    K2HorizonMoEGate,
    Model,
    ModelArgs,
    ValueExperts,
)


def _tiny_args(mova: bool = True, **over) -> ModelArgs:
    kw = dict(
        model_type="k2_horizon",
        hidden_size=64,
        num_hidden_layers=3,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=40,
        rms_norm_eps=1e-6,
        norm_groups=4,
        rope_theta=1e7,
        max_position_embeddings=4096,
    )
    if mova:
        kw.update(
            norm_groups=2,
            attn_output_gate=True,
            num_experts=8,
            num_experts_per_tok=3,
            moe_intermediate_size=24,
            shared_expert_intermediate_size=24,
            first_k_dense_replace=1,
            num_value_experts=6,
            num_value_experts_per_tok=2,
            scoring_func="sigmoid",
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
        )
    kw.update(over)
    return ModelArgs(**kw)


def _randomize(module, seed: int):
    rng = np.random.default_rng(seed)
    upd = []
    for k, v in tree_flatten(module.parameters()):
        shape = v.shape
        if "norm" in k or k.endswith("layernorm.weight"):
            a = 1.0 + 0.3 * rng.standard_normal(shape)
        elif k.endswith(("e_score_correction_bias", "selection_bias")):
            a = 0.3 * rng.standard_normal(shape)
        else:
            a = rng.standard_normal(shape) / np.sqrt(shape[-1])
        upd.append((k, mx.array(a.astype(np.float32))))
    module.update(tree_unflatten(upd))
    mx.eval(module.parameters())
    return module


def _np(x) -> np.ndarray:
    return np.array(x.astype(mx.float32), dtype=np.float64)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _group_rms(x, w, groups, eps):
    shape = x.shape
    y = x.reshape(*shape[:-1], groups, shape[-1] // groups)
    y = y / np.sqrt((y * y).mean(-1, keepdims=True) + eps)
    return y.reshape(shape) * w


def _rope_neox(x, theta, dims=None):
    # x: [T, H, D]; NeoX pairs dim i with dim i + dims/2.
    T, _, D = x.shape
    dims = dims or D
    half = dims // 2
    freqs = theta ** (-np.arange(half) * 2.0 / dims)
    ang = np.arange(T)[:, None] * freqs[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:dims]
    out = x.copy()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:dims] = x2 * cos + x1 * sin
    return out


def _route(logits, bias, k, norm, scale):
    scores = 1.0 / (1.0 + np.exp(-logits))
    choice = scores + bias
    inds = np.argsort(-choice, axis=-1, kind="stable")[..., :k]
    w = np.take_along_axis(scores, inds, axis=-1)
    if norm:
        w = w / np.maximum(w.sum(-1, keepdims=True), 6.103515625e-5)
    if scale not in (0.0, 1.0):
        w = w * scale
    return inds, w


def _ref_values(ve, x, args):
    W = _np(ve.v_router.weight)
    inds, w = _route(x @ W.T, _np(ve.selection_bias),
                     args.num_value_experts_per_tok, args.norm_topk_prob,
                     args.routed_scaling_factor)
    E = _np(ve.v_experts.weight)  # [n, out, in]
    out = np.zeros((x.shape[0], E.shape[1]))
    for t in range(x.shape[0]):
        for j in range(inds.shape[1]):
            out[t] += w[t, j] * _silu(E[inds[t, j]] @ x[t])
    return out


def _ref_mlp(m, x):
    g = x @ _np(m.gate_proj.weight).T
    u = x @ _np(m.up_proj.weight).T
    return (_silu(g) * u) @ _np(m.down_proj.weight).T


def _ref_moe(moe, x, args):
    inds, w = _route(x @ _np(moe.gate.weight).T,
                     _np(moe.gate.e_score_correction_bias),
                     args.num_experts_per_tok, args.norm_topk_prob,
                     args.routed_scaling_factor)
    sw = moe.switch_mlp
    G, U, D = (_np(sw.gate_proj.weight), _np(sw.up_proj.weight),
               _np(sw.down_proj.weight))
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(inds.shape[1]):
            e = inds[t, j]
            h = _silu(G[e] @ x[t]) * (U[e] @ x[t])
            out[t] += w[t, j] * (D[e] @ h)
    if "shared_experts" in moe:
        out = out + _ref_mlp(moe.shared_experts, x)
    return out


def _ref_attn(attn, x, args):
    T = x.shape[0]
    H, KV, D = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    q = (x @ _np(attn.q_proj.weight).T).reshape(T, H, D)
    k = (x @ _np(attn.k_proj.weight).T).reshape(T, KV, D)
    if "value_experts" in attn:
        v = _ref_values(attn.value_experts, x, args).reshape(T, KV, D)
    else:
        v = (x @ _np(attn.v_proj.weight).T).reshape(T, KV, D)
    if "q_norm" in attn:
        qw = _np(attn.q_norm.weight).reshape(H, D)
        kw = _np(attn.k_norm.weight).reshape(KV, D)
        q = q / np.sqrt((q * q).mean(-1, keepdims=True) + args.rms_norm_eps) * qw
        k = k / np.sqrt((k * k).mean(-1, keepdims=True) + args.rms_norm_eps) * kw
    q = _rope_neox(q, args.rope_theta, args.rope_dim)
    k = _rope_neox(k, args.rope_theta, args.rope_dim)
    rep = H // KV
    k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
    s = np.einsum("thd,shd->hts", q, k) / math.sqrt(D)
    s = s + np.triu(np.full((T, T), -np.inf), 1)[None]
    p = np.exp(s - s.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    o = np.einsum("hts,shd->thd", p, v).reshape(T, H * D)
    if "gate_proj" in attn:
        g = x @ _np(attn.gate_proj.weight).T
        o = o * np.log2(1.0 + np.exp2(g))
    return o @ _np(attn.o_proj.weight).T


def _ref_model(model, tokens, args):
    eps, groups = args.rms_norm_eps, args.norm_groups
    h = _np(model.model.embed_tokens.weight)[tokens]
    for layer in model.layers:
        a = _group_rms(h, _np(layer.input_layernorm.weight), groups, eps)
        h = h + _ref_attn(layer.self_attn, a, args)
        f = _group_rms(h, _np(layer.post_attention_layernorm.weight), groups, eps)
        if isinstance(layer.mlp, K2HorizonMoE):
            h = h + _ref_moe(layer.mlp, f, args)
        else:
            h = h + _ref_mlp(layer.mlp, f)
    h = _group_rms(h, _np(model.model.norm.weight), groups, eps)
    return h @ _np(model.lm_head.weight).T


def _rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


@pytest.mark.parametrize("args", [
    _tiny_args(mova=False),
    _tiny_args(mova=True),
    _tiny_args(mova=True, qk_norm=True),
    _tiny_args(mova=False, rope_dim=8),
    _tiny_args(mova=True, norm_topk_prob=False, routed_scaling_factor=1.0),
], ids=["dense", "mova", "mova-qk-norm", "dense-partial-rope", "mova-no-norm"])
def test_full_model_matches_reference(args):
    model = _randomize(Model(args), seed=3)
    tokens = np.array([1, 5, 9, 2, 33, 7, 11])
    got = _np(model(mx.array(tokens[None]))[0])
    ref = _ref_model(model, tokens, args)
    assert _rel(got, ref) < 2e-4


def test_layer_kinds_follow_the_dense_lead():
    model = Model(_tiny_args())
    kinds = [(type(ly.mlp).__name__, "value_experts" in ly.self_attn)
             for ly in model.layers]
    assert kinds == [("MLP", False), ("K2HorizonMoE", True),
                     ("K2HorizonMoE", True)]
    assert all("gate_proj" in ly.self_attn for ly in model.layers)


def test_cached_decode_matches_full_forward():
    args = _tiny_args()
    model = _randomize(Model(args), seed=5)
    tokens = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
    full = model(tokens)
    cache = model.make_cache()
    steps = [model(tokens[:, :5], cache=cache)[:, -1]]
    for i in range(5, 8):
        steps.append(model(tokens[:, i:i + 1], cache=cache)[:, -1])
    got = mx.stack(steps[1:], axis=1)
    assert _rel(_np(got), _np(full[:, 5:])) < 1e-4


def test_value_bias_moves_selection_but_not_weights():
    args = _tiny_args()
    ve = _randomize(ValueExperts(args), seed=7)
    x = mx.array(np.random.default_rng(0).standard_normal((1, 4, 64)),
                 dtype=mx.float32)
    base = ve(x)
    # A large bias on expert 0 forces it into every selection; its weight
    # still comes from the unbiased score.
    bias = np.zeros(args.num_value_experts, dtype=np.float32)
    bias[0] = 10.0
    ve.selection_bias = mx.array(bias)
    forced = _np(ve(x))
    ref = _ref_values(ve, _np(x[0]), args)
    assert _rel(forced[0], ref) < 1e-5
    assert _rel(forced, _np(base)) > 1e-3


def test_route_weight_sum_is_clamped_like_llama_cpp():
    args = _tiny_args(num_value_experts_per_tok=1)
    ve = ValueExperts(args)
    ve.v_router.weight = mx.full(ve.v_router.weight.shape, -30.0)
    ve.selection_bias = mx.zeros((args.num_value_experts,))
    x = mx.ones((1, 1, 64))
    from gmlx.models.k2_horizon.model import _route
    _, w = _route(ve.v_router(x), ve.selection_bias, 1, "sigmoid", True, 1.0)
    # sigmoid(-1920) underflows to 0, and the clamped sum keeps the
    # weight finite instead of 0/0.
    assert np.isfinite(_np(w)).all()


def test_gate_route_weights_match_the_forward():
    args = _tiny_args()
    gate = _randomize(K2HorizonMoEGate(args), seed=9)
    x = mx.array(np.random.default_rng(1).standard_normal((2, 3, 64)),
                 dtype=mx.float32)
    inds, w = gate(x)
    assert _rel(_np(gate._kq_route_weights(x, inds)), _np(w)) < 1e-6


def test_route_replay_supports_every_moe_layer():
    from gmlx.stream.moe_experts import _gate_submodule, _hook_target
    from gmlx.stream.moe_routes import moe_layers, replay_unsupported

    model = Model(_tiny_args())
    assert replay_unsupported(model) == []
    assert moe_layers(model) == [1, 2]
    for layer in model.layers[1:]:
        assert _gate_submodule(layer.mlp) is layer.mlp.gate
        assert _hook_target(layer.mlp) is layer.mlp.gate
        assert _gate_submodule(layer.self_attn) is None


def test_cast_predicate_matches_the_loader_keep_list():
    from gmlx.load.loader import _FP32_KEEP_BY_MODEL_TYPE

    keep = _FP32_KEEP_BY_MODEL_TYPE["k2_horizon"]
    model = Model(_tiny_args())
    pred = model.cast_predicate
    pinned = set()
    for path, _ in tree_flatten(model.parameters()):
        assert pred(path) == (not any(s in path for s in keep)), path
        if not pred(path):
            pinned.add(path.split(".", 3)[-1])
    # The dense layer's mlp.gate_proj and the quantized value router stay
    # on the activation dtype.
    assert pinned == {
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "self_attn.value_experts.selection_bias",
    }


_PER_LAYER_DENSE = [
    "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_output.weight", "attn_gate.weight", "ffn_norm.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
]
_PER_LAYER_MOVA = [
    "attn_norm.weight", "attn_q.weight", "attn_k.weight",
    "attn_v_gate.weight", "attn_v_gate.bias", "attn_v_exps.weight",
    "attn_output.weight", "attn_gate.weight", "ffn_norm.weight",
    "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight",
    "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight", "ffn_down_shexp.weight",
]


@pytest.mark.parametrize("qk_norm", [False, True])
def test_remap_covers_every_wire_tensor_onto_real_params(qk_norm):
    """Every tensor llama.cpp's k2-horizon loader creates maps onto a
    parameter of the model, and every parameter is loaded."""
    from gmlx.load.remap import RemapDecision, parse_gguf_name

    args = _tiny_args(qk_norm=qk_norm)
    params = {k for k, _ in tree_flatten(Model(args).parameters())}
    extra = ["attn_q_norm.weight", "attn_k_norm.weight"] if qk_norm else []
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for i in range(args.num_hidden_layers):
        per = _PER_LAYER_DENSE if i < args.first_k_dense_replace \
            else _PER_LAYER_MOVA
        names += [f"blk.{i}.{t}" for t in per + extra]
    mapped = set()
    for name in names:
        dec = parse_gguf_name("k2-horizon", name)
        assert dec.kind == RemapDecision.KIND_MAP, (name, dec.reason)
        assert dec.transform == "passthrough", name
        assert dec.hf_name in params, (name, dec.hf_name)
        mapped.add(dec.hf_name)
    assert mapped == params


def test_unknown_tensor_is_not_claimed():
    from gmlx.load.remap import RemapDecision, parse_gguf_name

    dec = parse_gguf_name("k2-horizon", "blk.0.attn_sinks.weight")
    assert dec.kind != RemapDecision.KIND_MAP


def test_synth_config_builds_this_model():
    from gmlx.load.config_synth import synthesize_config
    from test_config_synth import _K2_HORIZON_MOVA_SHAPES, _k2_horizon_mova_meta

    c = synthesize_config(_k2_horizon_mova_meta(),
                          tensor_shapes=_K2_HORIZON_MOVA_SHAPES)
    model = Model(ModelArgs.from_dict(c))
    ve = model.layers[1].self_attn.value_experts
    assert ve.v_experts.weight.shape == (6, 32, 64)
    assert ve.v_router.weight.shape == (6, 64)
    assert model.layers[1].mlp.switch_mlp.gate_proj.weight.shape == (8, 24, 64)
    assert model.layers[1].mlp.shared_experts.gate_proj.weight.shape == (24, 64)


def _loader_shaped():
    from tune.tiny_train_archs import build

    model, config, _ = build("k2_horizon")
    return model, config


def test_loader_shaped_dtypes_and_fused_moe(monkeypatch):
    """As the loader leaves it: the FFN router and both selection biases are
    fp32, the value router is a K-quant linear (Q4_K on the wire), the
    value experts gather K-quant, and every MoE layer's expert container
    gets a bf16 input the fused decode gate accepts."""
    from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

    import gmlx.load.modules as modules
    from gmlx.load.dtypes import activation_dtype

    if activation_dtype() != mx.bfloat16:
        pytest.skip("the loader picks float16 activations on this device")
    model, _ = _loader_shaped()
    moe = model.layers[1]
    ve = moe.self_attn.value_experts
    assert isinstance(ve.v_router, KQuantLinear)
    assert isinstance(ve.v_experts, KQuantSwitchLinear)
    assert ve.selection_bias.dtype == mx.float32
    assert moe.mlp.gate.weight.dtype == mx.float32
    assert moe.mlp.gate.e_score_correction_bias.dtype == mx.float32
    assert model.layers[0].mlp.gate_proj.weight.dtype != mx.float32

    glu_cls = type(moe.mlp.switch_mlp)
    if not hasattr(glu_cls, "_fused_ok"):
        pytest.skip("fused MoE glu not installable in this build")
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)
    orig = glu_cls._fused_ok
    verdicts = []

    def record(self, x, indices):
        verdicts.append((x.dtype, orig(self, x, indices)))
        return False  # the stock path runs on the CPU

    monkeypatch.setattr(glu_cls, "_fused_ok", record)
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache=cache)
    verdicts.clear()
    out = model(mx.array([[4]]), cache=cache)
    mx.eval(out)
    assert out.dtype == mx.bfloat16
    assert len(verdicts) == 2
    assert all(v == (mx.bfloat16, True) for v in verdicts), verdicts


def test_expert_streaming_wraps_the_ffn_experts_only(monkeypatch):
    """Streaming wraps each MoE layer's SwitchGLU; the MoVA value experts
    stay resident, and the top-k override reaches the FFN gate only."""
    from mlx_kquant.nn import KQuantSwitchLinear

    from gmlx.stream.expert_streaming import (
        install_expert_streaming,
        install_moe_experts_override,
    )

    model, _ = _loader_shaped()
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024})
    tokens = mx.array([[1, 2, 3, 4, 5]])
    ref = _np(model(tokens))
    install_expert_streaming(model)
    for layer in model.layers[1:]:
        assert type(layer.mlp.switch_mlp).__name__.endswith("_CPUOffload")
        assert type(layer.self_attn.value_experts.v_experts) is KQuantSwitchLinear
    # The streamed experts run on the CPU stream, about one bf16 ulp off.
    assert _rel(_np(model(tokens)), ref) < 3e-2

    assert install_moe_experts_override(model, 1) == 2
    for layer in model.layers[1:]:
        assert layer.mlp.gate.top_k == 1
        assert layer.self_attn.value_experts.top_k == 2


def test_lookahead_router_replica_matches_the_gate():
    from gmlx.stream.lookahead import _router_fn_for

    args = _tiny_args()
    moe = _randomize(K2HorizonMoE(args), seed=13)
    select = _router_fn_for(moe)
    assert select is not None
    x = mx.array(np.random.default_rng(2).standard_normal((1, 1, 64)),
                 dtype=mx.float32)
    inds, w = moe.gate(x)
    ids, mass = select(x)
    assert sorted(_np(ids).ravel().tolist()) == sorted(_np(inds).ravel().tolist())
    assert (np.diff(_np(mass).ravel()) <= 0).all()


_ON_GPU = mx.default_device() == mx.gpu


@pytest.mark.skipif(not _ON_GPU, reason="the kq router kernel is Metal-only")
@pytest.mark.parametrize("n, k, norm, scale", [
    (100, 8, True, 2.5), (64, 4, True, 2.5), (64, 4, False, 1.0)])
def test_fused_router_matches_the_eager_select(n, k, norm, scale):
    from gmlx.models.k2_horizon.model import _kq_router_ok, _route

    if not _kq_router_ok(1, k):
        pytest.skip("this kquant build has no sigmoid router arm")
    rng = np.random.default_rng(n + k)
    logits = mx.array(rng.standard_normal((1, 1, n)) * 3, dtype=mx.float32)
    bias = mx.array(rng.standard_normal(n) * 0.3, dtype=mx.float32)
    fi, fw = _route(logits, bias, k, "sigmoid", norm, scale, fused=True)
    ei, ew = _route(logits, bias, k, "sigmoid", norm, scale, fused=False)
    fo, eo = np.argsort(_np(fi).ravel()), np.argsort(_np(ei).ravel())
    assert (_np(fi).ravel()[fo] == _np(ei).ravel()[eo]).all()
    np.testing.assert_allclose(_np(fw).ravel()[fo], _np(ew).ravel()[eo],
                               rtol=1e-5)


@pytest.mark.skipif(not _ON_GPU, reason="the fused MoE kernels are Metal-only")
def test_shared_expert_fold_matches_the_unfolded_block():
    model, _ = _loader_shaped()
    moe = model.layers[1].mlp
    se = getattr(moe.switch_mlp, "_kq_shexp_mod", None)
    if se is None:
        pytest.skip("this kquant build does not fold the shared expert")
    assert se is moe.shared_experts
    x = mx.array(np.random.default_rng(4).standard_normal((1, 1, 256)),
                 dtype=mx.bfloat16)
    folded = _np(moe(x))
    object.__setattr__(moe.switch_mlp, "_kq_shexp_mod", None)
    plain = _np(moe(x))
    assert _rel(folded, plain) < 2e-2


def test_router_kernel_is_never_called_off_the_gpu(monkeypatch):
    import mlx_kquant as kq

    from gmlx.models.k2_horizon import model as k2

    def boom(*a, **k):
        raise AssertionError("kq.moe_router_topk called off the GPU")

    monkeypatch.setattr(kq, "moe_router_topk", boom, raising=False)
    monkeypatch.setitem(k2._KQ_ROUTER_STATE, "ok", True)
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        logits = mx.zeros((1, 1, 64))
        inds, _ = k2._route(logits, None, 4, "sigmoid", True, 2.5)
        mx.eval(inds)
    finally:
        mx.set_default_device(prev)
