"""A MoE block whose gate and up expert stacks use different codecs (Hy4:
an iq2_xxs gate with an stq1_0 up) takes the fused gathers where the kq
build lists the pair, and hands the kernel the up codec. CPU-safe: the
kernel calls and the build's pair list are faked, so the tests run on a kq
without the pair kernels too."""

import mlx.core as mx
import mlx.nn as nn
import pytest

import gmlx.load.modules as modules
from gmlx.load.modules import install_fused_moe_glu
from gmlx.models.deepseek_v4 import model as v4

D, INTER, E, TOP_K = 256, 256, 4, 2


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _block(gate="iq2_xxs", up="stq1_0", down="iq3_xxs"):
    from mlx_kquant.nn import KQuantSwitchLinear

    args = v4.ModelArgs(
        hidden_size=D, moe_intermediate_size=INTER, n_routed_experts=E,
        num_experts_per_tok=TOP_K, n_shared_experts=1, num_hidden_layers=4,
        vocab_size=64, swiglu_limit=10.0)
    block = v4.DeepseekV4MoE(args, 3)
    for name, codec, (o, i) in (("gate_proj", gate, (INTER, D)),
                                ("up_proj", up, (INTER, D)),
                                ("down_proj", down, (D, INTER))):
        setattr(block.switch_mlp, name,
                KQuantSwitchLinear(E, o, i, False, codec))
    block.eval()
    return block


_LISTED = {("iq2_xxs", "stq1_0"), ("stq1_0", "iq2_xxs"), ("q3_k", "iq3_s")}


@pytest.fixture
def listed(monkeypatch):
    """A kq build that lists the pairs in _LISTED."""
    base = modules._FusedMoeCaps

    class _Caps(base):
        def __init__(self):
            super().__init__()
            self._glu_pair = (lambda g, u, *a, **kw: (g, u) in _LISTED)

    monkeypatch.setattr(modules, "_FusedMoeCaps", _Caps)
    return _Caps


def test_a_listed_pair_installs_fused(listed):
    blk = _block()
    assert install_fused_moe_glu(_Shell(blk)) == 1
    assert hasattr(blk.switch_mlp, "_fused_h")


@pytest.mark.parametrize("why", ["unlisted", "old_kq"])
def test_a_pair_without_a_kernel_stays_unfused(why, monkeypatch):
    if why == "unlisted":
        blk = _block(gate="iq3_xxs", up="q4_k")
    else:
        blk = _block()
        base = modules._FusedMoeCaps

        class _Caps(base):
            def __init__(self):
                super().__init__()
                self._glu_pair = None    # a kq build before the pairs

        monkeypatch.setattr(modules, "_FusedMoeCaps", _Caps)
    assert install_fused_moe_glu(_Shell(blk)) == 0


@pytest.mark.parametrize("up", ["stq1_0", "iq2_xxs"])
def test_the_up_codec_reaches_the_kernel_only_when_it_differs(
        up, listed, monkeypatch):
    import mlx_kquant as kq

    blk = _block(up=up)
    install_fused_moe_glu(_Shell(blk))
    seen = {}

    def fake_glu(x, gw, uw, ktype, idx, **kw):
        seen["ktype"], seen["kw"] = ktype, kw
        return mx.zeros((x.shape[0], idx.shape[1], INTER), x.dtype)

    monkeypatch.setattr(kq, "moe_glu_gather_kq", fake_glu)
    blk.switch_mlp._fused_h(mx.zeros((1, D)), mx.array([[0, 1]]))
    assert seen["ktype"] == "iq2_xxs"
    if up == "iq2_xxs":
        assert "up_kquant_type" not in seen["kw"]
    else:
        assert seen["kw"]["up_kquant_type"] == "stq1_0"


def test_a_split_pair_never_counts_on_the_qf_kernels(listed, monkeypatch):
    caps = modules._FusedMoeCaps()
    monkeypatch.setattr(caps, "qf_codec", lambda codec: True)
    same = _block(up="iq2_xxs").switch_mlp
    split = _block().switch_mlp
    assert modules._kq_row_floor(same, TOP_K, D, caps) == 1
    assert modules._kq_row_floor(split, TOP_K, D, caps) > 1


def test_a_split_pair_of_one_shape_never_takes_the_concat(listed):
    """q3_k and iq3_s blocks are both 110 bytes per 256 weights, so the
    stacks match in shape; the concat would decode up in the gate codec."""
    mlp = _block(gate="q3_k", up="iq3_s").switch_mlp
    assert mlp.gate_proj.weight.shape == mlp.up_proj.weight.shape
    budget = 1 << 40
    assert modules._install_gateup_concat(mlp, budget) == budget
    assert not getattr(mlp, "_kq_gate_up_pending", False)
