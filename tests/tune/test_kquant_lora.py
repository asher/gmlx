# vendor_upstream: skip
"""LoRA on a K-quant base builds the vendored tuner's classes through
gmlx.tune.kquant_lora, whatever another package set on the K-quant
classes, and changes no object of a stock package or of mlx-kquant."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq
import pytest
from mlx_kquant.nn import KQuantEmbedding, KQuantLinear, KQuantSwitchLinear

from gmlx._vendor.mlx_lm.tuner.lora import (
    LoRAEmbedding,
    LoRALinear,
    LoRASwitchLinear,
)
from gmlx._vendor.mlx_lm.tuner.utils import linear_to_lora_layers
from gmlx.tune.lora import LORA_KEYS, prepare_lora_student

_KQ = (KQuantLinear, KQuantSwitchLinear, KQuantEmbedding)
_CONFIG = {"rank": 2, "scale": 1.0, "dropout": 0.0}


def _wire(rows, cols):
    return kq.quantize(mx.random.normal((rows, cols)) * 0.1, "q8_0")[0]


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = KQuantLinear(64, 32, False, "q8_0")
        self.proj.weight = _wire(32, 64)
        self.experts = KQuantSwitchLinear(2, 32, 64, False, "q8_0")
        self.experts.weight = _wire(2 * 32, 64).reshape(2, 32, -1)
        self.table = KQuantEmbedding(16, 64, "q8_0")
        self.table.weight = _wire(16, 64)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Layer(), _Layer()]


def _adapt():
    mx.random.seed(0)
    model = _Model()
    bases = {k: m for k, m in model.named_modules() if isinstance(m, _KQ)}
    linear_to_lora_layers(model, 2, dict(_CONFIG))
    return model, bases


def _check_adapted(model, bases):
    want = {KQuantLinear: LoRALinear, KQuantSwitchLinear: LoRASwitchLinear,
            KQuantEmbedding: LoRAEmbedding}
    modules = dict(model.named_modules())
    assert len(bases) == 6
    for path, base in bases.items():
        lora = modules[path]
        assert type(lora) is want[type(base)], path
        assert getattr(lora, "linear", getattr(lora, "embedding", None)) is base
    x = mx.random.normal((1, 3, 64))
    lora, base = modules["layers.0.proj"], bases["layers.0.proj"]
    assert lora.lora_a.shape == (64, 2)
    # lora_b starts at zero, so the adapted layer computes the base.
    assert mx.allclose(lora(x), base(x)).item()


def test_linear_to_lora_layers_wraps_each_kquant_class():
    _check_adapted(*_adapt())


def test_a_foreign_to_lora_on_the_kquant_classes_is_not_consulted(monkeypatch):
    def foreign(self, r, scale, dropout):
        raise AssertionError("a foreign to_lora ran")

    for cls in _KQ:
        monkeypatch.setattr(cls, "to_lora", foreign, raising=False)
    _check_adapted(*_adapt())


def test_prepare_lora_student_adapts_a_kquant_gguf(tmp_path):
    """The train keys (the tuner's defaults) and the distill keys both adapt
    the Q8_0 projections of a tiny GGUF."""
    import gmlx
    from test_loader_api import _mint_tiny_llama

    path = str(tmp_path / "tiny-q8.gguf")
    _mint_tiny_llama(path, q8_0=True)
    for keys in (None, LORA_KEYS):
        model, _, _ = gmlx.load_model(path)
        n = prepare_lora_student(model, rank=2, scale=1.0, keys=keys)
        adapted = [m for _, m in model.named_modules()
                   if isinstance(m, LoRALinear)]
        assert n == len(adapted) > 0
        assert all(isinstance(m.linear, KQuantLinear) for m in adapted)
        mx.eval(model(mx.array([[1, 2, 3]])))


_NO_STOCK_MUTATION = r"""
import json
import sys

sys.path[:0] = [sys.argv[1]]
try:
    import mlx_lm.models.cache as lm_cache
    import mlx_lm.tuner.lora as lm_lora
    import mlx_vlm.models.cache as vlm_cache
except ImportError as e:
    print(json.dumps({"skip": f"stock packages not installed: {e}"}))
    raise SystemExit(0)

from mlx_kquant.nn import KQuantEmbedding, KQuantLinear, KQuantSwitchLinear


def snapshot():
    held = {}
    for label, module in (("mlx_lm.models.cache", lm_cache),
                          ("mlx_vlm.models.cache", vlm_cache),
                          ("mlx_lm.tuner.lora", lm_lora)):
        held[label] = dict(vars(module))
        for name, value in vars(module).items():
            if isinstance(value, type) and value.__module__ == module.__name__:
                held[f"{label}.{name}"] = dict(vars(value))
    for cls in (KQuantLinear, KQuantSwitchLinear, KQuantEmbedding):
        held[f"mlx_kquant.nn.{cls.__name__}"] = dict(vars(cls))
    return held


def stock_modules():
    return sorted(m for m in sys.modules
                  if m.partition(".")[0] in ("mlx_lm", "mlx_vlm"))


before, loaded = snapshot(), stock_modules()

import mlx.core as mx
import gmlx
from gmlx.tune.lora import LORA_KEYS, prepare_lora_student
from test_loader_api import _mint_tiny_llama

path = sys.argv[2]
_mint_tiny_llama(path, q8_0=True)
adapted = []
for keys in (None, LORA_KEYS):
    model, _, _ = gmlx.load_model(path)
    adapted.append(prepare_lora_student(model, rank=2, scale=1.0, keys=keys))
    mx.eval(model(mx.array([[1, 2, 3]])))

after = snapshot()
changed = []
for label in sorted(set(before) | set(after)):
    a, b = before.get(label, {}), after.get(label, {})
    for name in sorted(set(a) | set(b)):
        if a.get(name, KeyError) is not b.get(name, KeyError):
            changed.append(f"{label}: {name}")
print(json.dumps({"adapted": adapted, "changed": changed,
                  "new_stock_modules": sorted(set(stock_modules()) - set(loaded))}))
"""


def test_lora_preparation_changes_no_stock_or_kquant_object(tmp_path):
    """Stock mlx-lm and mlx-vlm load first, then gmlx loads a tiny K-quant
    GGUF and prepares a LoRA student with the train and the distill keys.
    Every attribute of the stock cache and LoRA modules and classes, and of
    the K-quant classes, is the same object afterwards. Runs in a child,
    since the session hides the stock packages."""
    load_tests = str(Path(__file__).resolve().parents[1] / "load")
    r = subprocess.run(
        [sys.executable, "-c", _NO_STOCK_MUTATION, load_tests,
         str(tmp_path / "tiny-q8.gguf")],
        capture_output=True, text=True, timeout=600, env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    if "skip" in out:
        pytest.skip(out["skip"])
    assert all(n > 0 for n in out["adapted"]), out["adapted"]
    assert out["changed"] == []
    assert out["new_stock_modules"] == []
