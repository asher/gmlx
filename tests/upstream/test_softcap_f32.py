"""gemma-4 final-logit softcap runs in float32 on the mlx-lm and mlx-vlm
seams. Stock computes tanh(x/cap)*cap at the activation dtype, so bfloat16
logits round by about cap*eps. The input is bfloat16 by design."""
import mlx.core as mx
import pytest
from gmlx._vendor.mlx_lm.models import gemma4_text as lm_g4

import gmlx.upstream.softcap_f32 as sc

vlm_g4 = pytest.importorskip("gmlx._vendor.mlx_vlm.models.gemma4.language")

CAP = 30.0


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setenv("GMLX_G4_SOFTCAP_F32", "1")
    monkeypatch.setattr(lm_g4, "logit_softcap", lm_g4.logit_softcap)
    monkeypatch.setattr(vlm_g4, "logit_softcap", vlm_g4.logit_softcap)
    monkeypatch.setattr(sc, "_installed", False)
    assert sc.install_gemma4_softcap_f32()


def _logits():
    mx.random.seed(3)
    return (mx.random.normal((2, 4096)) * 12.0).astype(mx.bfloat16)


def _ref(x):
    return mx.tanh(x.astype(mx.float32) / CAP) * CAP


@pytest.mark.parametrize("mod", [lm_g4, vlm_g4], ids=["gmlx._vendor.mlx_lm", "gmlx._vendor.mlx_vlm"])
def test_softcap_matches_float32_reference(installed, mod):
    x = _logits()
    out = mod.logit_softcap(CAP, x)
    err = mx.abs(out.astype(mx.float32) - _ref(x)).max().item()
    assert err < 1e-5, err
    assert out.dtype == mx.float32

