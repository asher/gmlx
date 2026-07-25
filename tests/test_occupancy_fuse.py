"""occupancy_fuse install-time wiring: eligibility gates and env kill.

Numerics and perf are certified against a real q6_k llama model by the
lab cert probe (bit-equal B=2 logits/tokens, +2.6%/+4.8% at B=8/16);
these tests pin the cheap invariants that need no model.
"""

from __future__ import annotations

import mlx.nn as nn

from gmlx import occupancy_fuse as of


class _PlainAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.rope = lambda x, offset=0: x


_PlainAttention.__name__ = "Attention"
_PlainAttention.__module__ = "mlx_lm.models.llama"


class _PlainMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)


_PlainMLP.__name__ = "MLP"
_PlainMLP.__module__ = "mlx_lm.models.llama"


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _PlainAttention()
        self.mlp = _PlainMLP()


def test_plain_linear_not_eligible():
    # right names/modules but nn.Linear leaves: no swap, no crash
    assert of.install_occupancy_fuse(_Model()) == 0


def test_env_kill_disables_install(monkeypatch):
    monkeypatch.setenv("GMLX_OCCUPANCY_FUSE", "0")
    assert of.install_occupancy_fuse(_Model()) == 0


def test_wrong_arch_not_eligible():
    m = _Model()
    type(m.attn).__module__ = "mlx_lm.models.qwen3"
    type(m.mlp).__module__ = "mlx_lm.models.qwen3"
    try:
        assert of.install_occupancy_fuse(m) == 0
    finally:
        type(m.attn).__module__ = "mlx_lm.models.llama"
        type(m.mlp).__module__ = "mlx_lm.models.llama"


def test_same_codec_gate_rejects_non_kquant():
    m = _Model()
    projs = [m.attn.q_proj, m.attn.k_proj, m.attn.v_proj]
    assert not of._same_codec_kquant(projs)
