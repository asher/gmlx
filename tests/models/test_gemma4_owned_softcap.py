"""The owned Gemma 4 final-logit softcap runs in float32.

Stock computes tanh(x / cap) * cap at the activation dtype, in mlx-vlm and
in mlx-lm, so each bfloat16 logit rounds by about cap * eps: about 0.12
nats at a cap of 30, enough to flip a near tie. The owned softcap computes
in float32 and returns float32 logits. The input is bfloat16 by design.
The stock halves run in a child that applies no loader install
(gemma4_probes.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gmlx.models.gemma4 import language as owned

import gemma4_probes as probes

DRIVER = Path(__file__).with_name("gemma4_probes.py")


@pytest.fixture(scope="module")
def stock() -> dict:
    r = subprocess.run([sys.executable, str(DRIVER), "--cpu"],
                       capture_output=True, text=True, env=dict(os.environ),
                       timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout)


def test_owned_softcap_matches_the_float32_reference():
    got = probes.softcap(owned.logit_softcap)
    assert got["dtype"] == "mlx.core.float32"
    assert got["err"] < 1e-5, got


@pytest.mark.parametrize("package", ["gmlx._vendor.mlx_vlm", "gmlx._vendor.mlx_lm"])
def test_stock_softcap_rounds_at_bfloat16(stock, package):
    got = stock["softcap"][package]
    assert got["dtype"] == "mlx.core.bfloat16"
    assert got["err"] > 1e-5, got


def test_a_bfloat16_owned_model_returns_float32_logits():
    assert probes.model_logits_dtype(owned) == "mlx.core.float32"


def test_a_bfloat16_stock_model_returns_bfloat16_logits(stock):
    assert stock["model_logits_dtype"] == "mlx.core.bfloat16"
