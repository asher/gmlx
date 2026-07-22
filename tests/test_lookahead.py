"""Lookahead predictor + recall probe (gmlx.lookahead), and the offload
wrapper's lookahead seam."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.loader import install_expert_streaming
from gmlx.lookahead import (
    LookaheadProbe,
    _gate_module_select,
    _norm_gains,
    _router_fn_for,
    _sigmoid_bias_select,
    install_lookahead,
)
from mlx_lm.models.switch_layers import SwitchGLU

DIM, HID, NE, K = 16, 32, 8, 3


class _FakeGate(nn.Module):
    """DeepSeek-shaped gate submodule: returns (inds, weights), top-k by
    plain dot-product score, selection order deliberately unsorted."""

    def __init__(self):
        super().__init__()
        self.top_k = K
        self.weight = mx.random.normal((NE, DIM))

    def __call__(self, x):
        scores = x @ self.weight.T
        inds = mx.argpartition(-scores, kth=K - 1, axis=-1)[..., :K]
        return inds, mx.take_along_axis(scores, inds, axis=-1)


class _FakeMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _FakeGate()
        self.switch_mlp = SwitchGLU(DIM, HID, NE)

    def __call__(self, x):
        inds, w = self.gate(x)
        y = self.switch_mlp(x, inds)
        return (y * w[..., None]).sum(axis=-2)


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = nn.RMSNorm(DIM)
        self.mlp = _FakeMoE()


class _FakeModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = [_FakeLayer() for _ in range(n_layers)]


def _streaming_model(monkeypatch, n_layers=2):
    mx.random.seed(7)
    model = _FakeModel(n_layers)
    mx.eval(model.parameters())
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)
    for layer in model.layers:
        assert type(layer.mlp.switch_mlp).__name__.endswith("_CPUOffload")
    return model


def test_lookahead_family_default(monkeypatch):
    """glm_moe_dsa / deepseek_v32 default lookahead OFF (replica-router sync
    tax measured above its stall savings there); everything else defaults ON;
    an explicit GMLX_DECODE_LOOKAHEAD always wins."""
    from gmlx.envflags import env_bool
    from gmlx.loader import _lookahead_default

    m = _FakeModel(1)
    assert _lookahead_default(m)  # no model_type: on
    for fam in ("glm_moe_dsa", "deepseek_v32"):
        m.model_type = fam
        assert not _lookahead_default(m), fam
    for fam in ("minimax_m3", "hunyuan_v3_moe", "qwen3_moe"):
        m.model_type = fam
        assert _lookahead_default(m), fam

    m.model_type = "glm_moe_dsa"
    monkeypatch.setenv("GMLX_DECODE_LOOKAHEAD", "1")
    assert env_bool("GMLX_DECODE_LOOKAHEAD", _lookahead_default(m))
    m.model_type = "qwen3_moe"
    monkeypatch.setenv("GMLX_DECODE_LOOKAHEAD", "0")
    assert not env_bool("GMLX_DECODE_LOOKAHEAD", _lookahead_default(m))


def test_norm_gains_plain_rmsnorm():
    norm = nn.RMSNorm(DIM)
    norm.weight = mx.arange(1, DIM + 1).astype(mx.float32)
    g = _norm_gains(norm)
    assert g is not None
    assert mx.allclose(g, norm.weight, atol=1e-5)


def test_gate_module_select_ranks_by_weight():
    class _Stub(nn.Module):
        top_k = 3

        def __call__(self, x):
            return (
                mx.array([[4, 9, 2]]),
                mx.array([[0.1, 0.7, 0.2]]),
            )

    ids = np.array(_gate_module_select(_Stub())(mx.zeros((1, DIM))))
    assert ids.tolist() == [[9, 2, 4]]


def test_sigmoid_bias_select_matches_stock_selection():
    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(DIM, NE, bias=False)
            self.e_score_correction_bias = mx.random.normal((NE,))
            self.num_experts_per_tok = K

    mx.random.seed(3)
    blk = _Block()
    x = mx.random.normal((1, 1, DIM))
    ids = np.array(_sigmoid_bias_select(blk)(x)).reshape(-1)
    # Reference: the stock forward's selection seam.
    choice = mx.sigmoid(blk.gate(x.astype(mx.float32)))
    choice = np.array(choice + blk.e_score_correction_bias).reshape(-1)
    ref = np.argsort(-choice)[:K]
    assert ids.tolist() == ref.tolist()


def test_router_fn_for_unknown_block_is_none():
    class _Odd(nn.Module):
        pass

    assert _router_fn_for(_Odd()) is None


def test_probe_recall_math():
    probe = LookaheadProbe()
    probe.note(3, {"raw": np.array([[5, 1, 2, 9]])})
    probe.actual(3, np.array([[1, 2, 7, 8]]))
    cells = probe._recall[(3, "raw")]
    assert cells[1] == [0.0, 4.0]  # pred[5] misses; |actual| = 4
    assert cells[2] == [1.0, 4.0]  # {5,1}: one hit
    assert cells[4] == [2.0, 4.0]  # {5,1,2,9}: two hits
    # Previous-token baseline on the same layer.
    probe.actual(3, np.array([[2, 7, 9, 9]]))  # unique {2,7,9}; prev {1,2,7,8}
    assert probe._prev_recall == [2.0, 3.0]
    # A prediction for a layer that never reports actuals stays pending.
    probe.note(9, {"raw": np.array([[1, 2, 3, 4]])})
    assert 9 in probe._pending


def test_probe_multirow_and_shape_mismatch():
    probe = LookaheadProbe()
    probe.note(1, {"raw": np.array([[1, 2], [3, 4]])})
    probe.actual(1, np.array([[1, 5], [3, 4]]))
    cells = probe._recall[(1, "raw")]
    assert cells[2] == [3.0, 4.0]  # row0: 1 hit of 2; row1: 2 of 2
    # Row-count mismatch (draft-length change) is skipped, not misaligned.
    probe.note(1, {"raw": np.array([[1, 2]])})
    probe.actual(1, np.array([[1, 2], [1, 2]]))
    assert cells[2] == [3.0, 4.0]


def test_install_and_probe_end_to_end(monkeypatch):
    model = _streaming_model(monkeypatch)
    monkeypatch.setenv("GMLX_DECODE_LOOKAHEAD_PROBE", "1")
    n = install_lookahead(model, model.layers, probe=True)
    assert n == 1
    glu0 = model.layers[0].mlp.switch_mlp
    glu1 = model.layers[1].mlp.switch_mlp
    la0 = getattr(glu0, "_kq_lookahead", None)
    la1 = getattr(glu1, "_kq_lookahead", None)
    assert la0 is not None and la0.predictor is not None
    assert la0.predictor.dst_li == 1 and la0.predictor._ratio is not None
    # Last MoE layer: no next layer to predict, but still anchors actuals.
    assert la1 is not None and la1.predictor is None
    assert la0.probe is la1.probe

    x = mx.random.normal((1, 1, DIM))
    ref0 = mx.array(_FakeMoE.__call__.__get__(model.layers[0].mlp)(x))
    mx.eval(ref0)
    out0 = model.layers[0].mlp(x)
    mx.eval(out0)
    # The hook only observes: layer output is numerically untouched.
    assert mx.allclose(ref0, out0, atol=1e-6, rtol=1e-6)
    probe = la0.probe
    assert la0.predictor.dst_li in probe._pending
    model.layers[1].mlp(x)
    assert la0.predictor.dst_li not in probe._pending
    assert probe._recall  # comparison recorded
    # The prediction equals layer 1's router run on the ratio-scaled input:
    # with identity norms (fresh RMSNorm weights are ones) that is exactly
    # layer 1's own routing on the same x - recall@K must be perfect.
    cells = probe._recall[(1, "ratio")]
    assert cells[K] == [float(K), float(K)]


def test_dead_predictor_disables_cleanly(monkeypatch):
    model = _streaming_model(monkeypatch)
    install_lookahead(model, model.layers, probe=True)
    glu0 = model.layers[0].mlp.switch_mlp
    la0 = glu0._kq_lookahead
    monkeypatch.setattr(
        la0.predictor, "_router_fn", lambda x: (_ for _ in ()).throw(TypeError("boom"))
    )
    x = mx.random.normal((1, 1, DIM))
    out = model.layers[0].mlp(x)  # must not raise
    mx.eval(out)
    assert la0.predictor.dead
    out = model.layers[0].mlp(x)  # dead predictor: plain path, still fine
    mx.eval(out)


def test_probe_report_smoke(capsys):
    probe = LookaheadProbe()
    probe.note(2, {"raw": np.array([[1, 2, 3, 4]])})
    probe.actual(2, np.array([[1, 2, 9, 9]]))
    probe.report()
    out = capsys.readouterr().out
    assert "router recall probe" in out and "raw" in out
    probe.report()  # idempotent
    assert capsys.readouterr().out == ""


def test_wrapper_prestages_next_layer(monkeypatch):
    """With prefetch enabled the offload wrapper hands the materialized
    prediction to the decode feeder's prestage for the NEXT MoE layer,
    after this layer's stage() has returned."""
    from contextlib import contextmanager

    model = _streaming_model(monkeypatch)
    install_lookahead(model, model.layers, probe=False, prefetch=True)
    glu0 = model.layers[0].mlp.switch_mlp
    assert glu0._kq_lookahead.prefetch

    class _FakeDF:
        calls = []

        def covers(self, li):
            return True

        def stage(self, li, ids):
            self.calls.append(("stage", li, ids.copy()))
            return ids.astype(np.uint32)

        def prestage(self, li, pred):
            self.calls.append(("prestage", li, pred.copy()))

        def wedged_at(self, li):
            return False

        @contextmanager
        def swapped(self, li):
            yield

    df = _FakeDF()
    object.__setattr__(glu0, "_kq_decode_feeder", df)
    object.__setattr__(glu0, "_kq_li", 0)
    x = mx.random.normal((1, 1, DIM))
    out = model.layers[0].mlp(x)
    mx.eval(out)
    kinds = [c[0] for c in df.calls]
    assert kinds == ["stage", "prestage"]  # prestage after demand joined
    assert df.calls[1][1] == 1  # targets the NEXT MoE layer
    pred = df.calls[1][2]
    assert pred.shape[-1] == K and pred.min() >= 0 and pred.max() < NE


def test_rank_gate_trims_and_recovers():
    """Per-rank hit EMAs start optimistic, trim the unreliable tail below
    min_p, and re-qualify ranks that start hitting again (the full width
    keeps being observed while gated)."""
    from gmlx.lookahead import RankGate

    g = RankGate(0.5)
    assert g.k(1, 3) == 3  # warm start: submit everything
    pred = np.array([[5, 6, 7]])
    for _ in range(200):  # rank 0 always routed, ranks 1-2 never
        g.note(1, pred)
        g.observe(1, np.array([[5, 0, 1]]))
    assert g.k(1, 3) == 1
    for _ in range(200):  # rank 1 starts landing: re-qualifies
        g.note(1, pred)
        g.observe(1, np.array([[5, 6, 1]]))
    assert g.k(1, 3) == 2


def test_wrapper_gate_trims_prestage(monkeypatch):
    """A destination layer whose tail ranks measure unreliable gets a
    narrower prestage; a fully unreliable one gets none at all."""
    from contextlib import contextmanager

    model = _streaming_model(monkeypatch)
    install_lookahead(model, model.layers, probe=False, prefetch=True)
    glu0 = model.layers[0].mlp.switch_mlp
    gate = glu0._kq_lookahead.gate
    assert gate is not None

    class _FakeDF:
        calls = []

        def covers(self, li):
            return True

        def stage(self, li, ids):
            return ids.astype(np.uint32)

        def prestage(self, li, pred):
            self.calls.append(pred.copy())

        def wedged_at(self, li):
            return False

        @contextmanager
        def swapped(self, li):
            yield

    df = _FakeDF()
    object.__setattr__(glu0, "_kq_decode_feeder", df)
    object.__setattr__(glu0, "_kq_li", 0)
    x = mx.random.normal((1, 1, DIM))
    gate._ema[1] = np.array([0.9, 0.2, 0.1])  # only rank 0 reliable
    mx.eval(model.layers[0].mlp(x))
    assert df.calls[-1].shape[-1] == 1
    gate._ema[1] = np.array([0.1, 0.1, 0.1])  # nothing reliable
    n = len(df.calls)
    mx.eval(model.layers[0].mlp(x))
    assert len(df.calls) == n  # no prestage submitted at all


class _FakeRouterMoE(nn.Module):
    """hy_v3-shaped block: the DeepSeek-shaped gate submodule sits at
    ``router`` (no ``gate`` attribute at all)."""

    def __init__(self):
        super().__init__()
        self.router = _FakeGate()
        self.switch_mlp = SwitchGLU(DIM, HID, NE)

    def __call__(self, x):
        inds, w = self.router(x)
        y = self.switch_mlp(x, inds)
        return (y * w[..., None]).sum(axis=-2)


def test_router_attr_gate_drives_predictor(monkeypatch):
    """A gate submodule named ``router`` (hy_v3) is recognized and wires
    the same layer-pair predictor a ``gate``-named submodule gets."""
    mx.random.seed(7)
    model = _FakeModel(2)
    for layer in model.layers:
        layer.mlp = _FakeRouterMoE()
    mx.eval(model.parameters())
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)
    assert _router_fn_for(model.layers[1].mlp) is not None
    assert install_lookahead(model, model.layers, probe=True) == 1
    la0 = model.layers[0].mlp.switch_mlp._kq_lookahead
    assert la0.predictor is not None and la0.predictor.dst_li == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
