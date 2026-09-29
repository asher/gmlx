"""gmlx's KVCache subclasses stay out of batching and KV quantization.

MSAKVCache, QSAKVCache and HyV4KVCache carry state a plain KVCache row
does not, and a layer held at fp16 hides ``to_quantized``. Each must meet
the mlx-vlm batch constructor, the three arms of mlx-vlm's
``maybe_quantize_kv_cache`` and gmlx's safe KV quantizer the way it does
when the classes derive from mlx-lm's KVCache: the batch constructor
raises, the hybrid and TurboQuant arms pass the layer through, and the
uniform arm and the safe path call ``to_quantized``, which MSA and QSA
refuse. No ``extract`` is visible on any of them. A plain KVCache beside
each one shows the arm ran. The checks run in a child, so each arm is the
stock function with no gmlx replacement installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

CHILD = r'''
import importlib
import json
import types

import mlx.core as mx

mx.set_default_device(mx.cpu)

# importlib: the mlx_vlm and mlx_lm packages export a generate function
# that shadows the submodule attribute.
ar = importlib.import_module("mlx_vlm.generate.ar")
common = importlib.import_module("mlx_vlm.generate.common")
from mlx_lm.models.cache import KVCache
from mlx_vlm.models import cache as vlm_cache

from gmlx.cache.kv_policy import hold_fp16
from gmlx.models.hy_v4.model import HyV4KVCache
from gmlx.models.minimax_m3 import MSAKVCache
from gmlx.models.qwen4_exp.model import QSAKVCache

stock_quantize = common.maybe_quantize_kv_cache
SUBJECTS = {
    "MSAKVCache": MSAKVCache,
    "QSAKVCache": QSAKVCache,
    "HyV4KVCache": HyV4KVCache,
    "Fp16HeldKVCache": lambda: hold_fp16(KVCache()),
}


def filled(make):
    c = make()
    k = mx.zeros((1, 2, 4, 64))
    c.update_and_fetch(k, k)
    return c


def describe(value, subject):
    return {"type": type(value).__name__, "same": value is subject,
            "bits": getattr(value, "bits", None)}


def outcome(fn):
    try:
        return {"ok": fn()}
    except Exception as e:
        return {"raises": type(e).__name__, "msg": str(e)}


def batch(make):
    model = types.SimpleNamespace(make_cache=lambda: [make()], layers=[None])
    return outcome(lambda: [type(c).__name__
                            for c in ar._make_cache(model, [0, 1])])


def quantize(make, fn, **kwargs):
    control, subject = vlm_cache.KVCache(), filled(make)
    stack = [control, subject]
    out = outcome(lambda: fn(stack, 0, 64, **kwargs))
    out["control"] = describe(stack[0], control)
    out["subject"] = describe(stack[1], subject)
    return out


ARMS = {
    "uniform": dict(kv_bits=8),
    "hybrid": dict(kv_bits=4, kv_key_scheme="turboquant",
                   kv_value_scheme="uniform"),
    "turboquant": dict(kv_bits=4, kv_quant_scheme="turboquant"),
}

out = {"control batch": batch(vlm_cache.KVCache)}
for name, make in SUBJECTS.items():
    row = {"batch": batch(make),
           "extract": callable(getattr(filled(make), "extract", None))}
    for arm, kwargs in ARMS.items():
        row[arm] = quantize(make, stock_quantize, **kwargs)
    out[name] = row

from gmlx.cache.apc_pooling import install_safe_kv_quantization

install_safe_kv_quantization()
lm_generate = importlib.import_module("mlx_lm.generate")
for name, make in SUBJECTS.items():
    out[name]["safe"] = quantize(make, lm_generate.maybe_quantize_kv_cache,
                                 kv_bits=8)
print(json.dumps(out))
'''

SUBJECTS = ("MSAKVCache", "QSAKVCache", "HyV4KVCache", "Fp16HeldKVCache")
# What to_quantized does for each: MSA and QSA refuse, HyV4 converts, and
# the held layer has no to_quantized to call.
CONVERTS = {"MSAKVCache": "raises", "QSAKVCache": "raises",
            "HyV4KVCache": "converts", "Fp16HeldKVCache": "skipped"}
CONTROL = {"uniform": "QuantizedKVCache", "hybrid": "HybridQuantKVCache",
           "turboquant": "TurboQuantKVCache", "safe": "QuantizedKVCache"}


@pytest.fixture(scope="module")
def child():
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True,
                       text=True, env=dict(os.environ), timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_a_plain_kvcache_batches(child):
    assert child["control batch"] == {"ok": ["BatchKVCache"]}


@pytest.mark.parametrize("name", SUBJECTS)
def test_batch_constructor_refuses(child, name):
    got = child[name]["batch"]
    assert got["raises"] == "ValueError"
    assert name in got["msg"] and "does not yet support batching" in got["msg"]


@pytest.mark.parametrize("name", SUBJECTS)
def test_no_extract(child, name):
    assert child[name]["extract"] is False


@pytest.mark.parametrize("arm", ["hybrid", "turboquant"])
@pytest.mark.parametrize("name", SUBJECTS)
def test_hybrid_and_turboquant_arms_pass_the_layer_through(child, name, arm):
    got = child[name][arm]
    assert "raises" not in got, got
    assert got["subject"]["same"], got
    assert got["control"]["type"] == CONTROL[arm]


@pytest.mark.parametrize("arm", ["uniform", "safe"])
@pytest.mark.parametrize("name", SUBJECTS)
def test_uniform_arm_and_safe_path_call_to_quantized(child, name, arm):
    got = child[name][arm]
    assert got["control"]["type"] == CONTROL[arm]
    if CONVERTS[name] == "raises":
        assert got["raises"] == "NotImplementedError"
        assert name in got["msg"]
        assert got["subject"]["same"]
    elif CONVERTS[name] == "converts":
        assert "raises" not in got, got
        assert got["subject"] == {"type": "QuantizedKVCache", "same": False,
                                  "bits": 8}
    else:
        assert "raises" not in got, got
        assert got["subject"]["same"]
