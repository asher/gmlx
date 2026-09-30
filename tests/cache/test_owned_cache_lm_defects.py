"""mlx-lm cache defects the owned classes fix by taking the mlx-vlm body.

gmlx binds the mlx_lm cache names to its own classes, so each mlx-lm
defect below goes away in a process that imports gmlx. One scenario
script runs twice: in a child that never imports gmlx, where it must
still show the stock defect, and in a child that imports gmlx first,
where the same mlx_lm names must give the fixed result.

- ChunkedKVCache.maybe_trim_front trims on the padded buffer width and
  drops valid tokens.
- ConcatenateKVCache.trim leaves the trimmed keys in the buffer.
- BatchRotatingKVCache takes a one-token update in place while prepare
  lengths are set, and raises.
- KVCache, RotatingKVCache and BatchKVCache ``state`` read ``keys.shape``
  on an empty cache.
- The quantized SDPA puts a 4D batch mask against 5D grouped scores at
  B > 1: it raises when B differs from the kv head count and misapplies
  the mask when they match. The case runs through mlx_lm's llama
  attention.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_SCENARIO = r'''
import importlib
import json
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)
if sys.argv[1] == "owned":
    import gmlx  # noqa: F401
C = importlib.import_module("gmlx._vendor.mlx_lm.models.cache")
llama = importlib.import_module("gmlx._vendor.mlx_lm.models.llama")
out = {"module": C.KVCache.__module__,
       "gmlx": any(m == "gmlx" for m in sys.modules)}


def cols(k, row=0):
    return [int(x) for x in (k[row, 0, :, 0] - 1).tolist()]


def kv(c, rows=1):
    k = mx.array(c, dtype=mx.float32)[None, None, :, None] + 1
    k = mx.broadcast_to(k, (rows, 1, len(c), 2))
    return k, k


def run(fn):
    try:
        return {"ok": fn()}
    except Exception as e:
        return {"raises": type(e).__name__, "msg": str(e)[:200]}


def chunked():
    c = C.ChunkedKVCache(chunk_size=4)
    c.update_and_fetch(*kv(list(range(6))))
    c.maybe_trim_front()
    live = c.offset - c.start_position
    return [c.start_position, cols(c.keys[..., :max(live, 0), :])]


def concat_trim():
    c = C.ConcatenateKVCache()
    c.update_and_fetch(*kv(list(range(5))))
    c.trim(2)
    k, _ = c.update_and_fetch(*kv([9]))
    return cols(k)


def rotating_lengths():
    import numpy as np

    c = C.BatchRotatingKVCache(8, [0, 0])
    c.prepare(lengths=[4, 2], right_padding=[0, 2])
    c.update_and_fetch(*kv([0, 1, 2], 2))
    c.update_and_fetch(*kv([3], 2))
    c.finalize()
    m = c.make_mask(1)
    k, _ = c.update_and_fetch(*kv([50], 2))
    full = np.array(mx.broadcast_to(m, (2, 1, 1, k.shape[2])))
    return [sorted(int(x) for x, v in zip(cols(k, b), full[b, 0, 0]) if v)
            for b in range(2)]


def none_state(make):
    def fn():
        return [x is None for x in make().state]
    return fn


def qsdpa(nkv):
    def fn():
        args = llama.ModelArgs(
            model_type="llama", hidden_size=512, num_hidden_layers=1,
            intermediate_size=64, num_attention_heads=8, rms_norm_eps=1e-5,
            vocab_size=32, num_key_value_heads=nkv)
        mx.random.seed(3)
        attn = llama.Attention(args)
        mx.eval(attn.parameters())
        x = mx.random.normal((2, 40, 512)) * 0.5
        y = mx.random.normal((2, 1, 512)) * 0.5
        pads = mx.array([7, 0])
        errs = []
        ref_c, q_c = C.KVCache(), C.KVCache()
        attn(x, mask="causal", cache=ref_c)
        attn(x, mask="causal", cache=q_c)
        q_c = q_c.to_quantized(group_size=64, bits=8)
        pos = mx.arange(41)[None, None, None, :]
        mask = pos >= pads[:, None, None, None]
        ref = attn(y, mask=mask, cache=ref_c)
        got = attn(y, mask=mask, cache=q_c)
        return mx.abs(got - ref).max().item()
    return fn


out["chunked"] = run(chunked)
out["concat_trim"] = run(concat_trim)
out["rotating_lengths"] = run(rotating_lengths)
out["none_state"] = {
    "KVCache": run(none_state(C.KVCache)),
    "RotatingKVCache": run(none_state(lambda: C.RotatingKVCache(8))),
    "BatchKVCache": run(none_state(lambda: C.BatchKVCache([0, 0]))),
}
out["qsdpa"] = {str(n): run(qsdpa(n)) for n in (4, 2)}
print(json.dumps(out))
'''

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _run(mode):
    env = dict(os.environ)
    cwd = None
    if mode == "stock":
        keep = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                if p and os.path.abspath(p) != _ROOT]
        env["PYTHONPATH"] = os.pathsep.join(keep)
        cwd = "/"
    r = subprocess.run([sys.executable, "-c", _SCENARIO, mode],
                       capture_output=True, text=True, timeout=600, env=env,
                       cwd=cwd)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def stock():
    out = _run("stock")
    assert out["module"] == "gmlx._vendor.mlx_lm.models.cache" and not out["gmlx"]
    return out


@pytest.fixture(scope="module")
def owned():
    out = _run("owned")
    assert out["module"] == "gmlx.cache.kv_cache" and out["gmlx"]
    return out


def test_chunked_trim_front_keeps_the_last_chunk_of_valid_tokens(stock, owned):
    assert stock["chunked"]["ok"][0] != 2
    assert owned["chunked"] == {"ok": [2, [2, 3, 4, 5]]}


def test_concatenate_trim_drops_the_trimmed_keys(stock, owned):
    assert stock["concat_trim"] == {"ok": [0, 1, 2, 3, 4, 9]}
    assert owned["concat_trim"] == {"ok": [0, 1, 2, 9]}


def test_batch_rotating_one_token_update_with_lengths(stock, owned):
    assert stock["rotating_lengths"]["raises"] == "RuntimeError"
    assert owned["rotating_lengths"] == {
        "ok": [[0, 1, 2, 3, 50], [0, 1, 50]]}


@pytest.mark.parametrize("cls", ["KVCache", "RotatingKVCache",
                                 "BatchKVCache"])
def test_state_of_an_empty_cache_is_none(stock, owned, cls):
    assert stock["none_state"][cls]["raises"] == "AttributeError"
    assert owned["none_state"][cls]["ok"][:2] == [True, True]


def test_quantized_sdpa_batch_mask_through_mlx_lm_llama(stock, owned):
    assert stock["qsdpa"]["4"]["raises"] == "ValueError"
    assert stock["qsdpa"]["2"]["ok"] > 0.01
    for nkv in ("4", "2"):
        assert owned["qsdpa"][nkv]["ok"] < 5e-3, nkv
