"""The chunked gated delta rule against mlx-lm's per-token scan: outputs,
final state and gradients, with GQA, padding masks, an initial state, a
row length that is not a chunk multiple, and near-zero decays. Then the
training wrapper and the owned Qwen3.5 forward's route to it. CPU."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

# the module attributes, not import-time bindings: gmlx rebinds them to the
# tiled K->V head mapping once a qwen35-family GGUF has loaded in the process
from mlx_lm.models import gated_delta as gd

import gmlx.tune.gdn as tg


@pytest.fixture(autouse=True)
def _cpu():
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(dev)


def _inputs(B=2, T=150, Hk=2, Hv=4, Dk=16, Dv=24, seed=0, tiny_decay=False):
    r = np.random.default_rng(seed)
    q = r.standard_normal((B, T, Hk, Dk)).astype(np.float32) * 0.3
    k = r.standard_normal((B, T, Hk, Dk)).astype(np.float32) * 0.3
    v = r.standard_normal((B, T, Hv, Dv)).astype(np.float32)
    g = r.uniform(0.6, 1.0, (B, T, Hv)).astype(np.float32)
    if tiny_decay:
        g[:, ::7] = 1e-30
    beta = r.uniform(0.05, 0.95, (B, T, Hv)).astype(np.float32)
    return [mx.array(x) for x in (q, k, v, g, beta)]


@pytest.mark.parametrize("T,chunk", [(150, 64), (64, 64), (200, 32), (5, 64)])
def test_chunk_matches_scan_outputs_and_state(T, chunk):
    q, k, v, g, beta = _inputs(T=T)
    y0, s0 = gd.gated_delta_ops(q, k, v, g, beta)
    y1, s1 = tg.gated_delta_chunk(q, k, v, g, beta, chunk=chunk)
    mx.eval(y0, s0, y1, s1)
    assert np.allclose(np.array(y0), np.array(y1), atol=2e-4, rtol=1e-3)
    assert np.allclose(np.array(s0), np.array(s1), atol=2e-4, rtol=1e-3)


def test_chunk_with_mask_and_initial_state():
    q, k, v, g, beta = _inputs(T=100, seed=3)
    mask = np.ones((2, 100), dtype=bool)
    mask[0, 70:] = False
    mask[1, 90:] = False
    mask = mx.array(mask)
    r = np.random.default_rng(9)
    state = mx.array(r.standard_normal((2, 4, 24, 16)).astype(np.float32) * 0.1)
    y0, s0 = gd.gated_delta_ops(q, k, v, g, beta, state, mask)
    y1, s1 = tg.gated_delta_chunk(q, k, v, g, beta, state, mask, chunk=32)
    mx.eval(y0, s0, y1, s1)
    m = np.array(mask)
    assert np.allclose(np.array(y0)[m], np.array(y1)[m], atol=2e-4, rtol=1e-3)
    assert np.allclose(np.array(s0), np.array(s1), atol=2e-4, rtol=1e-3)
    assert np.all(np.array(y1)[~m] == 0)


def test_chunk_tiny_decays_stay_finite_and_match():
    q, k, v, g, beta = _inputs(T=96, seed=5, tiny_decay=True)
    y0, s0 = gd.gated_delta_ops(q, k, v, g, beta)
    y1, s1 = tg.gated_delta_chunk(q, k, v, g, beta, chunk=32)
    mx.eval(y0, s0, y1, s1)
    assert np.isfinite(np.array(y1)).all()
    assert np.allclose(np.array(y0), np.array(y1), atol=2e-4, rtol=1e-3)


def test_chunk_gradients_match_scan():
    q, k, v, g, beta = _inputs(T=70, Hk=4, Hv=4, seed=11)
    r = np.random.default_rng(1)
    w = mx.array(r.standard_normal((2, 70, 4, 24)).astype(np.float32))
    mask = np.ones((2, 70), dtype=bool)
    mask[1, 60:] = False
    mask = mx.array(mask)
    # the scan writes an output at padded positions (from a state it then
    # discards); the chunked path writes zero, so the loss reads real ones
    wm = w * mask[..., None, None]

    def loss_scan(q, k, v, g, beta):
        y, s = gd.gated_delta_ops(q, k, v, g, beta, None, mask)
        return (y * wm).sum() + (s * s).sum() * 0.01

    def loss_chunk(q, k, v, g, beta):
        y, s = tg.gated_delta_chunk(q, k, v, g, beta, None, mask, chunk=32)
        return (y * wm).sum() + (s * s).sum() * 0.01

    g0 = mx.grad(loss_scan, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    g1 = mx.grad(loss_chunk, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    mx.eval(g0, g1)
    m = np.array(mask)
    for name, a, b in zip("qkvgb", g0, g1):
        a, b = np.array(a)[m], np.array(b)[m]
        scale = max(np.abs(a).max(), 1e-6)
        assert np.allclose(a, b, atol=2e-3 * scale, rtol=2e-3), name


def _update_inputs(T=40, seed=2):
    q, k, v, _g, _beta = _inputs(T=T, Hk=2, Hv=4, seed=seed)
    r = np.random.default_rng(4)
    a = mx.array(r.standard_normal((2, T, 4)).astype(np.float32))
    b = mx.array(r.standard_normal((2, T, 4)).astype(np.float32))
    A_log = mx.array(r.uniform(0.0, 2.0, (4,)).astype(np.float32))
    dt_bias = mx.array(r.standard_normal((4,)).astype(np.float32))
    return q, k, v, a, b, A_log, dt_bias


def test_update_wrapper_matches_gated_delta_update():
    q, k, v, a, b, A_log, dt_bias = _update_inputs()
    y0, s0 = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=False)
    y1, s1 = tg.gated_delta_update_chunked(q, k, v, a, b, A_log, dt_bias, chunk=16)
    mx.eval(y0, s0, y1, s1)
    assert np.allclose(np.array(y0), np.array(y1), atol=2e-4, rtol=1e-3)
    assert np.allclose(np.array(s0), np.array(s1), atol=2e-4, rtol=1e-3)


def test_training_update_checkpointed_matches_loop_with_gradients():
    q, k, v, a, b, A_log, dt_bias = _update_inputs(T=48, seed=6)
    mask = np.ones((2, 48), dtype=bool)
    mask[0, 40:] = False
    mask = mx.array(mask)
    r = np.random.default_rng(3)
    w = mx.array(r.standard_normal((2, 48, 4, 24)).astype(np.float32)) * mask[..., None, None]

    def loss_loop(q, k, v, a, b, A_log, dt_bias):
        y, s = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, None, mask, use_kernel=False)
        return (y * w).sum() + (s * s).sum() * 0.01

    def loss_train(q, k, v, a, b, A_log, dt_bias):
        y, s = tg.training_gated_delta_update(q, k, v, a, b, A_log, dt_bias, None, mask, chunk=16)
        return (y * w).sum() + (s * s).sum() * 0.01

    args = (q, k, v, a, b, A_log, dt_bias)
    l0, g0 = mx.value_and_grad(loss_loop, argnums=tuple(range(7)))(*args)
    l1, g1 = mx.value_and_grad(loss_train, argnums=tuple(range(7)))(*args)
    mx.eval(l0, g0, l1, g1)
    assert abs(float(l0) - float(l1)) <= 2e-3 * max(abs(float(l0)), 1.0)
    m = np.array(mask)
    for name, x, y in zip("qkvabAd", g0, g1):
        x, y = np.array(x), np.array(y)
        if x.ndim >= 3:
            x, y = x[m], y[m]
        scale = max(np.abs(x).max(), 1e-6)
        assert np.allclose(x, y, atol=3e-3 * scale, rtol=3e-3), name


def test_switch_and_vector_gating_route_to_the_loop(monkeypatch):
    q, k, v, a, b, A_log, dt_bias = _update_inputs()
    assert tg.chunked_gdn_active(a)
    monkeypatch.setenv("GMLX_TRAIN_GDN_CHUNK", "0")
    assert not tg.chunked_gdn_active(a)
    monkeypatch.delenv("GMLX_TRAIN_GDN_CHUNK")
    assert not tg.chunked_gdn_active(a[..., None])
    with pytest.raises(ValueError):
        tg.gated_delta_chunk(q, k, v, mx.ones(q.shape), mx.ones(a.shape))
    calls = []
    orig = tg.gated_delta_update_chunked

    def spy(*args, **kw):
        calls.append(1)
        return orig(*args, **kw)

    monkeypatch.setattr(tg, "gated_delta_update_chunked", spy)
    y, _ = tg.training_gated_delta_update(q, k, v, a, b, A_log, dt_bias)
    mx.eval(y)
    assert calls == [1]
    monkeypatch.setenv("GMLX_TRAIN_GDN_CHUNK", "0")
    y, _ = tg.training_gated_delta_update(q, k, v, a, b, A_log, dt_bias)
    mx.eval(y)
    assert calls == [1]


def test_owned_qwen35_training_forward_routes_to_the_chunked_scan(monkeypatch):
    pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    from mlx_vlm.models.qwen3_5.config import TextConfig

    import gmlx.models.qwen35.gdn as qgdn

    cfg = TextConfig(
        model_type="qwen3_5", hidden_size=64, intermediate_size=128,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_hidden_layers=4, num_attention_heads=4,
        rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=2,
        max_position_embeddings=2048, tie_word_embeddings=True, head_dim=32,
        rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                         "rope_theta": 100000, "partial_rotary_factor": 0.25},
        full_attention_interval=4,
    )
    mx.random.seed(5)
    mod = qgdn.OwnedQwen3_5GatedDeltaNet(cfg)
    mx.eval(mod.parameters())
    x = mx.random.normal((2, 37, 64))
    mod.eval()
    y_eval = mod(x)
    calls = []
    orig = qgdn.training_gated_delta_update

    def spy(*args, **kw):
        calls.append(1)
        return orig(*args, **kw)

    monkeypatch.setattr(qgdn, "training_gated_delta_update", spy)
    mod.train()
    y_train = mod(x)
    mx.eval(y_eval, y_train)
    assert calls == [1]
    assert np.allclose(np.array(y_eval), np.array(y_train), atol=2e-3, rtol=2e-3)


def test_chunk_follows_the_tiled_head_mapping(monkeypatch):
    """gmlx's loader switches mlx-lm's gated delta to the GGUF tiled K->V
    head mapping for qwen35-family models; the chunked scan must pair
    heads the same way or every value head reads the wrong key head."""
    q, k, v, g, beta = _inputs(T=48, Hk=2, Hv=4, seed=21)
    # both references expand the heads by hand, so the mapping is explicit
    # whichever one the process's gated delta module carries
    qg, kg = mx.repeat(q, 2, -2), mx.repeat(k, 2, -2)
    grouped_ref, _ = gd.gated_delta_ops(qg, kg, v, g, beta)
    qt, kt = mx.tile(q, [1, 1, 2, 1]), mx.tile(k, [1, 1, 2, 1])
    tiled_ref, _ = gd.gated_delta_ops(qt, kt, v, g, beta)
    mx.eval(grouped_ref, tiled_ref)
    assert not np.allclose(np.array(grouped_ref), np.array(tiled_ref), atol=1e-3)
    y_explicit, _ = tg.gated_delta_chunk(q, k, v, g, beta, chunk=16, tiled=True)
    monkeypatch.setattr(gd, "_gmlx_tiled_v_patched", True, raising=False)
    assert tg.tiled_heads()
    y_flag, _ = tg.gated_delta_chunk(q, k, v, g, beta, chunk=16)
    mx.eval(y_explicit, y_flag)
    assert np.allclose(np.array(tiled_ref), np.array(y_explicit), atol=2e-4, rtol=1e-3)
    assert np.allclose(np.array(tiled_ref), np.array(y_flag), atol=2e-4, rtol=1e-3)
    monkeypatch.setattr(gd, "_gmlx_tiled_v_patched", False, raising=False)
    y_grouped, _ = tg.gated_delta_chunk(q, k, v, g, beta, chunk=16)
    mx.eval(y_grouped)
    assert np.allclose(np.array(grouped_ref), np.array(y_grouped), atol=2e-4, rtol=1e-3)
