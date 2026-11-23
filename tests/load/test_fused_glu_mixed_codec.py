"""A MoE block whose gate and up expert stacks use different codecs (Hy4:
an iq2_xxs gate with an stq1_0 up) takes the fused gathers where the kq
build lists the pair, and hands the kernel the up codec. CPU-safe: the
kernel calls are faked."""

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


def _pair_listed():
    return modules._FusedMoeCaps().glu_pair_ok("iq2_xxs", "stq1_0")


needs_pair = pytest.mark.skipif(
    not _pair_listed(),
    reason="installed mlx-kquant has no iq2_xxs gate with stq1_0 up kernel")


@needs_pair
def test_a_listed_pair_installs_fused():
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


@needs_pair
@pytest.mark.parametrize("up", ["stq1_0", "iq2_xxs"])
def test_the_up_codec_reaches_the_kernel_only_when_it_differs(
        up, monkeypatch):
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


@needs_pair
def test_a_split_pair_never_counts_on_the_qf_kernels(monkeypatch):
    caps = modules._FusedMoeCaps()
    monkeypatch.setattr(caps, "qf_codec", lambda codec: True)
    same = _block(up="iq2_xxs").switch_mlp
    split = _block().switch_mlp
    assert modules._kq_row_floor(same, TOP_K, D, caps) == 1
    assert modules._kq_row_floor(split, TOP_K, D, caps) > 1
