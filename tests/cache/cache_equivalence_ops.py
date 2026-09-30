"""Run one fixed sequence of cache and SDPA operations and print the results.

``python cache_equivalence_ops.py owned|vlm|lm``. ``owned`` drives
gmlx.cache.kv_cache and gmlx.cache.attention; ``vlm`` and ``lm`` drive the
cache and base modules of the copied mlx-vlm and mlx-lm. The stock results
in golden/cache_equivalence.json come from ``vlm`` and ``lm`` with the
module names reversed to the stock packages (record_stock_goldens.py).
Each result, a raised exception included, is flattened to JSON twice:
whole, and reduced to the protocol attributes the packages share, so the
owned run can be compared exactly with mlx-vlm and, where the bodies
agree, with mlx-lm. Everything runs on the CPU device.
"""

import base64
import copy
import hashlib
import json
import sys
import types

import mlx.core as mx
import numpy as np

H, D = 2, 32
SCALE = D**-0.5
# Attributes both packages give every cache; the reduced form keeps only these.
PROTOCOL = ("keys", "values", "offset", "_idx", "left_padding", "max_size",
            "keep", "step", "bits", "group_size", "cache", "caches",
            "chunk_size", "start_position")
MODULE_TEXT = ("gmlx.cache.kv_cache", "gmlx._vendor.mlx_vlm.models.cache",
               "gmlx._vendor.mlx_lm.models.cache", "gmlx.cache.attention",
               "gmlx._vendor.mlx_vlm.models.base", "gmlx._vendor.mlx_lm.models.base")


def load(mode):
    if mode == "owned":
        import gmlx.cache.attention as attention
        import gmlx.cache.kv_cache as cache
    elif mode == "vlm":
        import gmlx._vendor.mlx_vlm.models.base as attention
        import gmlx._vendor.mlx_vlm.models.cache as cache
    else:
        import gmlx._vendor.mlx_lm.models.base as attention
        import gmlx._vendor.mlx_lm.models.cache as cache
    return cache, attention


def rnd(shape, seed, dtype=mx.float32):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(dtype)


def kv(B, S, seed, dtype=mx.float32, heads=H, dim=D):
    return (rnd((B, heads, S, dim), seed, dtype),
            rnd((B, heads, S, dim), seed + 1000, dtype))


# Groups whose float results come from matmul and softmax. They record the
# values, which another machine's BLAS may round differently in the last
# bits, and compare within a tolerance per dtype: (rtol, atol), two ulps.
VALUE_GROUPS = ("sdpa",)
TOLERANCE = {"mlx.core.float16": (2e-3, 1e-3), "mlx.core.bfloat16": (1.6e-2, 8e-3),
             "mlx.core.float32": (1e-5, 1e-6)}


def flat(v, full=True, depth=0, values=False):
    if isinstance(v, mx.array):
        if values and str(v.dtype) in TOLERANCE:
            bits = v.view(mx.uint16) if v.dtype == mx.bfloat16 else v
            return {"array": str(v.dtype), "shape": list(v.shape),
                    "b64": base64.b64encode(np.array(bits).tobytes()).decode()}
        # Exact: the bytes of the float32 widening, or of the array itself.
        a = v.astype(mx.float32) if v.dtype in (mx.float16, mx.bfloat16) else v
        data = np.array(a)
        return {"array": str(v.dtype), "shape": list(v.shape),
                "sha256": hashlib.sha256(data.tobytes()).hexdigest()[:24]}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, mx.Dtype):
        return {"dtype": str(v)}
    if isinstance(v, (list, tuple)):
        return {type(v).__name__: [flat(x, full, depth, values) for x in v]}
    if isinstance(v, dict):
        return {"dict": {str(k): flat(x, full, depth, values)
                         for k, x in v.items()}}
    if isinstance(v, type):
        return {"type": v.__name__}
    if hasattr(v, "__dict__") and depth < 4:
        attrs = dict(vars(v))
        if not full:
            # Read through getattr: a field can be a property in one package
            # and a plain attribute in the other.
            attrs = {}
            for k in PROTOCOL:
                try:
                    x = getattr(v, k)
                except Exception:
                    continue
                if not callable(x) or isinstance(x, mx.array):
                    attrs[k] = x
        return {"obj": type(v).__name__,
                "vars": {k: flat(x, full, depth + 1, values)
                         for k, x in attrs.items()}}
    return {"other": type(v).__name__}


def _decode(d):
    raw = base64.b64decode(d["b64"])
    if d["array"] == "mlx.core.bfloat16":
        a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        a = np.frombuffer(raw, np.dtype(d["array"].rsplit(".", 1)[1]))
    return a.astype(np.float32).reshape(d["shape"])


def _close(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) == set(y) == {"array", "shape", "b64"}:
            if (x["array"], x["shape"]) != (y["array"], y["shape"]):
                return False
            rtol, atol = TOLERANCE[x["array"]]
            return bool(np.allclose(_decode(x), _decode(y), rtol=rtol,
                                    atol=atol, equal_nan=True))
        return x.keys() == y.keys() and all(_close(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(map(_close, x, y))
    return x == y


def same(a: str, b: str) -> bool:
    """Two flattened results agree: exactly, or recorded values within the
    tolerance of their dtype."""
    return a == b or _close(json.loads(a), json.loads(b))


def norm(msg):
    for name in MODULE_TEXT:
        msg = msg.replace(name, "MODULE")
    return msg


class Recorder:
    def __init__(self):
        self.rows = []
        self.n = {}

    def _add(self, group, cls, method, key, value, err=None):
        n = self.n[group] = self.n.get(group, 0) + 1
        if err is not None:
            out = {"raises": type(err).__name__, "msg": norm(str(err))}
            full = reduced = json.dumps(out, sort_keys=True)
        else:
            values = group in VALUE_GROUPS
            full = json.dumps(flat(value, values=values), sort_keys=True)
            reduced = json.dumps(flat(value, False, values=values),
                                 sort_keys=True)
        self.rows.append({"key": f"{group}/{n:03d} {key}", "group": group,
                          "cls": cls, "method": method, "full": full,
                          "reduced": reduced})

    def step(self, group, cls, method, fn, key=""):
        try:
            value = fn()
        except Exception as e:  # recorded, the comparison reads it
            self._add(group, cls, method, key or method, None, e)
            return None
        self._add(group, cls, method, key or method, value)
        return value

    def view(self, group, cls, cache, key="view"):
        self._add(group, cls, "__dict__", key, cache)


def attr(obj, name):
    return lambda: getattr(obj, name)


def call(obj, name):
    return lambda: getattr(obj, name)()


def fresh(make):
    try:
        return make()
    except Exception:
        return None


# make_mask as the base create_attention_mask calls it, and bare.
MASKS_1 = tuple(((n,), {"return_array": ra, "window_size": ws})
                for n, ra, ws in ((1, False, None), (3, False, None),
                                  (3, True, None), (3, False, 4), (1, False, 4),
                                  (1, False, 2)))
MASKS_B = tuple(((n,), {"return_array": ra, "window_size": ws})
                for n, ra, ws in ((1, False, None), (2, False, None),
                                  (2, True, None), (1, False, 4), (2, False, 4)))


# Single-row caches.

def single_row(r, C, group, make):
    c = fresh(make)
    if c is None:
        r.step(group, "", "__init__", make)
        return
    cls = type(c).__name__
    s = lambda method, fn, key="": r.step(group, cls, method, fn, key)  # noqa: E731
    s("empty", call(c, "empty"))
    s("is_trimmable", call(c, "is_trimmable"))
    k, v = kv(1, 5, 1)
    s("update_and_fetch", lambda: c.update_and_fetch(k, v), "width 5")
    r.view(group, cls, c)
    for i in range(6):
        k1, v1 = kv(1, 1, 10 + i)
        s("update_and_fetch", lambda k1=k1, v1=v1: c.update_and_fetch(k1, v1),
          f"width 1 step {i}")
    r.view(group, cls, c)
    for args, kw in MASKS_1:
        s("make_mask", lambda args=args, kw=kw: c.make_mask(*args, **kw),
          f"make_mask {args} {sorted(kw.items())}")
    s("size", call(c, "size"))
    s("nbytes", attr(c, "nbytes"))
    s("is_trimmable", call(c, "is_trimmable"))
    k3, v3 = kv(1, 3, 20)
    s("update_and_fetch", lambda: c.update_and_fetch(k3, v3), "width 3")
    r.view(group, cls, c)
    s("trim", lambda: c.trim(2))
    r.view(group, cls, c)
    k1, v1 = kv(1, 1, 30)
    s("update_and_fetch", lambda: c.update_and_fetch(k1, v1), "after trim")
    r.view(group, cls, c)
    state = s("state", attr(c, "state"))
    meta = s("meta_state", attr(c, "meta_state"))
    s("from_state", lambda: type(c).from_state(c.state, c.meta_state))

    def set_state():
        d = make()
        d.state = c.state
        d.meta_state = c.meta_state
        return d

    s("state.setter", set_state)
    del state, meta
    s("size", call(c, "size"))
    s("empty", call(c, "empty"))
    s("nbytes", attr(c, "nbytes"))
    s("memory_profile", lambda: c.memory_profile(16))
    s("to_quantized", lambda: c.to_quantized(group_size=32, bits=4))
    s("extract", lambda: c.extract(0))

    def merged():
        other = make()
        k2, v2 = kv(1, 3, 40)
        other.update_and_fetch(k2, v2)
        return type(c).merge([copy.deepcopy(c), other])

    s("merge", merged)
    s("deepcopy", lambda: copy.deepcopy(c))
    s("prefix_cache_snapshot", lambda: c.prefix_cache_snapshot())

    def restored():
        d = make()
        d.prefix_cache_restore(c.prefix_cache_snapshot())
        return d

    s("prefix_cache_restore", restored)

    def reserved():
        d = copy.deepcopy(c)
        d.prefix_cache_reserve(40)
        return d

    s("prefix_cache_reserve", reserved)
    s("prefix_cache_merge",
      lambda: c.prefix_cache_merge([copy.deepcopy(c), copy.deepcopy(c)], [4, 6]))
    empty = make()
    s("state", attr(empty, "state"), "state of an empty cache")
    s("nbytes", attr(empty, "nbytes"), "nbytes of an empty cache")
    s("trim", lambda: empty.trim(1), "trim of an empty cache")


def chunked(r, C, group):
    c = fresh(lambda: C.ChunkedKVCache(chunk_size=4))
    cls = "ChunkedKVCache"
    s = lambda method, fn, key="": r.step(group, cls, method, fn, key)  # noqa: E731
    if c is None:
        return
    for i, width in enumerate((5, 1, 1, 3, 1)):
        k, v = kv(1, width, 50 + i)
        s("update_and_fetch", lambda k=k, v=v: c.update_and_fetch(k, v),
          f"width {width}")
        r.view(group, cls, c)
    s("trim", lambda: c.trim(2))
    for key in ("state", "meta_state"):
        s(key, attr(c, key))
    s("from_state", lambda: type(c).from_state(c.state, c.meta_state))
    s("size", call(c, "size"))
    s("nbytes", attr(c, "nbytes"))
    s("is_trimmable", call(c, "is_trimmable"))
    s("empty", call(c, "empty"))
    s("memory_profile", lambda: c.memory_profile(16))
    s("prefix_cache_snapshot", call(c, "prefix_cache_snapshot"))
    # The front trim last: mlx-lm's body leaves different state behind.
    s("maybe_trim_front", call(c, "maybe_trim_front"))
    r.view(group, cls, c)
    for i in range(3):
        k, v = kv(1, 1, 60 + i)
        s("update_and_fetch", lambda k=k, v=v: c.update_and_fetch(k, v),
          f"after front trim {i}")
        s("maybe_trim_front", call(c, "maybe_trim_front"), f"front trim {i}")
        r.view(group, cls, c)
    s("state", attr(c, "state"), "state after front trims")


def simple(r, C, group):
    c = fresh(lambda: C.SimpleKVCache())
    if c is None:
        return
    cls = "SimpleKVCache"
    s = lambda method, fn, key="": r.step(group, cls, method, fn, key)  # noqa: E731
    k, v = kv(1, 4, 70)
    s("update_and_fetch", lambda: c.update_and_fetch(k, v))
    k, v = kv(1, 1, 71)
    s("update", lambda: c.update(k, v))
    s("fetch", call(c, "fetch"))
    s("state", attr(c, "state"))
    s("meta_state", attr(c, "meta_state"))
    s("from_state", lambda: type(c).from_state(c.state, c.meta_state))
    s("prefix_cache_snapshot", call(c, "prefix_cache_snapshot"))
    s("empty", call(c, "empty"))
    s("nbytes", attr(c, "nbytes"))


def static_prefix(r, C, group):
    c = fresh(lambda: C.StaticPrefixKVCache(16, step=8))
    if c is None:
        return
    cls = "StaticPrefixKVCache"
    s = lambda method, fn, key="": r.step(group, cls, method, fn, key)  # noqa: E731
    for i, width in enumerate((5, 1, 12, 1)):
        k, v = kv(1, width, 80 + i)
        s("update_and_fetch", lambda k=k, v=v: c.update_and_fetch(k, v),
          f"width {width}")
        r.view(group, cls, c)
    s("decoder_state", attr(c, "decoder_state"))
    s("memory_profile", lambda: c.memory_profile(8))
    ro = s("from_prefix", lambda: type(c).from_prefix(c))
    if ro is not None:
        k, v = kv(1, 2, 90)
        s("update_and_fetch", lambda: ro.update_and_fetch(k, v), "read only")
        s("memory_profile", lambda: ro.memory_profile(8), "read only")
    s("trim", lambda: c.trim(3))
    s("state", attr(c, "state"))
    s("meta_state", attr(c, "meta_state"))
    s("from_state", lambda: type(c).from_state(c.state, c.meta_state))
    s("nbytes", attr(c, "nbytes"))


# Batch caches.

def batch(r, C, group, make, make_single):
    c = fresh(make)
    if c is None:
        r.step(group, "", "__init__", make)
        return
    cls = type(c).__name__
    s = lambda method, fn, key="": r.step(group, cls, method, fn, key)  # noqa: E731
    s("empty", call(c, "empty"))
    s("prepare", lambda: c.prepare(lengths=[3, 4], right_padding=[1, 0]))
    k, v = kv(2, 4, 100)
    s("update_and_fetch", lambda: c.update_and_fetch(k, v), "width 4")
    r.view(group, cls, c)
    s("finalize", call(c, "finalize"))
    r.view(group, cls, c)
    for i in range(6):
        k1, v1 = kv(2, 1, 110 + i)
        s("update_and_fetch", lambda k1=k1, v1=v1: c.update_and_fetch(k1, v1),
          f"width 1 step {i}")
    r.view(group, cls, c)
    for args, kw in MASKS_B:
        s("make_mask", lambda args=args, kw=kw: c.make_mask(*args, **kw),
          f"make_mask {args} {sorted(kw.items())}")
    k2, v2 = kv(2, 2, 120)
    s("update_and_fetch", lambda: c.update_and_fetch(k2, v2), "width 2")
    s("trim", lambda: c.trim(1))
    r.view(group, cls, c)
    k1, v1 = kv(2, 1, 130)
    s("update_and_fetch", lambda: c.update_and_fetch(k1, v1), "after trim")
    s("is_trimmable", call(c, "is_trimmable"))
    s("size", call(c, "size"))
    s("empty", call(c, "empty"))
    s("nbytes", attr(c, "nbytes"))
    s("batch_size", attr(c, "batch_size"))
    s("is_single_row", call(c, "is_single_row"))
    s("state", attr(c, "state"))
    s("meta_state", attr(c, "meta_state"))
    s("from_state", lambda: type(c).from_state(c.state, c.meta_state))
    s("extract", lambda: c.extract(0), "extract 0")
    s("extract", lambda: c.extract(1), "extract 1")

    def filtered():
        d = copy.deepcopy(c)
        d.filter(mx.array([1]))
        return d

    s("filter", filtered)

    def extended():
        d = copy.deepcopy(c)
        e = make()
        k3, v3 = kv(2, 3, 140)
        e.update_and_fetch(k3, v3)
        d.extend(e)
        return d

    s("extend", extended)

    def merged():
        rows = []
        for i, width in enumerate((3, 5)):
            one = make_single()
            k4, v4 = kv(1, width, 150 + i)
            one.update_and_fetch(k4, v4)
            rows.append(one)
        return type(c).merge(rows)

    s("merge", merged)
    s("to_quantized", lambda: c.to_quantized(group_size=32, bits=4))
    s("dequantize_for_apc", lambda: c.dequantize_for_apc())
    s("deepcopy", lambda: copy.deepcopy(c))
    empty = make()
    s("state", attr(empty, "state"), "state of an empty cache")
    s("finalize", call(empty, "finalize"), "finalize of an empty cache")
    s("trim", lambda: empty.trim(1), "trim of an empty cache")


def arrays(r, C, group):
    cls = "ArraysCache"
    s = lambda method, fn, key="": r.step(group, cls, method, fn, key)  # noqa: E731
    c = C.ArraysCache(size=2, left_padding=[1, 0])
    s("empty", call(c, "empty"))
    s("__setitem__", lambda: c.__setitem__(0, rnd((2, 3, 8), 200)))
    s("__setitem__", lambda: c.__setitem__(1, rnd((2, 4), 201)))
    s("__getitem__", lambda: c[0])
    s("make_mask", lambda: c.make_mask(3))
    s("left_padding", attr(c, "left_padding"))
    s("lengths", attr(c, "lengths"))
    s("state", attr(c, "state"))
    s("meta_state", attr(c, "meta_state"))
    s("from_state", lambda: type(c).from_state(c.state, c.meta_state))
    s("batch_size", attr(c, "batch_size"))
    s("nbytes", attr(c, "nbytes"))
    s("extract", lambda: c.extract(1))

    def filtered():
        d = copy.deepcopy(c)
        d.filter(mx.array([1]))
        return d

    s("filter", filtered)

    def extended():
        d = copy.deepcopy(c)
        e = C.ArraysCache(size=2)
        e[0] = rnd((1, 3, 8), 202)
        e[1] = rnd((1, 4), 203)
        d.extend(e)
        return d

    s("extend", extended)

    def merged():
        rows = []
        for i in range(2):
            one = C.ArraysCache(size=2)
            one[0] = rnd((1, 3, 8), 210 + i)
            if i:
                one[1] = rnd((1, 4), 212)
            rows.append(one)
        return type(c).merge(rows)

    s("merge", merged)
    s("merge", lambda: type(c).merge([C.ArraysCache(size=2)] * 2), "merge of empties")
    s("prepare", lambda: c.prepare(lengths=[2, 3]))
    s("make_mask", lambda: c.make_mask(3), "make_mask after prepare")
    s("__getitem__", lambda: c[1], "slot 1")
    s("memory_profile", lambda: c.memory_profile(16))
    s("prefix_cache_snapshot", call(c, "prefix_cache_snapshot"))
    s("deepcopy", lambda: copy.deepcopy(c))
    # advance last: mlx-lm's body subtracts the padding in place.
    s("advance", lambda: c.advance(2))
    s("left_padding", attr(c, "left_padding"), "left_padding after advance")
    s("lengths", attr(c, "lengths"), "lengths after advance")
    s("make_mask", lambda: c.make_mask(3), "make_mask after advance")
    r.view(group, cls, c)
    s("state", attr(c, "state"), "state after advance")
    s("extract", lambda: c.extract(0), "extract after advance")
    s("finalize", call(c, "finalize"))
    r.view(group, cls, c)
    s("empty", call(c, "empty"), "empty after finalize")


def cache_list(r, C, group):
    cls = "CacheList"
    s = lambda method, fn, key="": r.step(group, cls, method, fn, key)  # noqa: E731
    c = C.CacheList(C.KVCache(), C.RotatingKVCache(max_size=8))
    for i, width in enumerate((5, 1, 4)):
        k, v = kv(1, width, 300 + i)
        s("update_and_fetch", lambda k=k, v=v: [
            sub.update_and_fetch(k, v) for sub in c.caches], f"width {width}")
    s("__getitem__", lambda: c[1])
    s("is_trimmable", call(c, "is_trimmable"))
    s("trim", lambda: c.trim(1))
    s("size", call(c, "size"))
    s("empty", call(c, "empty"))
    s("nbytes", attr(c, "nbytes"))
    s("state", attr(c, "state"))
    s("meta_state", attr(c, "meta_state"))
    s("from_state", lambda: type(c).from_state(c.state, c.meta_state))
    s("deepcopy", lambda: copy.deepcopy(c))

    b = C.CacheList(C.BatchKVCache([1, 0]), C.BatchRotatingKVCache(8, [1, 0]))
    s("prepare", lambda: b.prepare(lengths=[3, 4], right_padding=[1, 0]),
      "batch prepare")
    k, v = kv(2, 4, 310)
    s("update_and_fetch", lambda: [sub.update_and_fetch(k, v) for sub in b.caches],
      "batch width 4")
    s("finalize", call(b, "finalize"), "batch finalize")
    r.view(group, cls, b)
    s("extract", lambda: b.extract(1))

    def filtered():
        d = copy.deepcopy(b)
        d.filter(mx.array([0]))
        return d

    s("filter", filtered)

    def extended():
        d = copy.deepcopy(b)
        e = C.CacheList(C.BatchKVCache([0, 0]), C.BatchRotatingKVCache(8, [0, 0]))
        k2, v2 = kv(2, 2, 320)
        for sub in e.caches:
            sub.update_and_fetch(k2, v2)
        d.extend(e)
        return d

    s("extend", extended)

    def merged():
        rows = []
        for i, width in enumerate((2, 3)):
            one = C.CacheList(C.KVCache(), C.RotatingKVCache(max_size=8))
            k3, v3 = kv(1, width, 330 + i)
            for sub in one.caches:
                sub.update_and_fetch(k3, v3)
            rows.append(one)
        return type(c).merge(rows)

    s("merge", merged)
    s("state", attr(b, "state"), "batch state")
    s("meta_state", attr(b, "meta_state"), "batch meta_state")
    s("from_state", lambda: type(b).from_state(b.state, b.meta_state),
      "batch from_state")


def pooling(r, C, group):
    if not hasattr(C, "PoolingCache"):
        return
    cls = "PoolingCache"
    s = lambda method, fn, key="": r.step(group, cls, method, fn, key)  # noqa: E731
    p = C.PoolingCache(ratio=4)
    s("accumulate_windows",
      lambda: p.accumulate_windows(rnd((1, 6, 8), 400), rnd((1, 6, 8), 401), 0),
      "prompt 6")
    s("update_and_fetch", lambda: p.update_and_fetch(rnd((1, 1, 8), 402)))
    for i in range(3):
        s("accumulate_windows", lambda i=i: p.accumulate_windows(
            rnd((1, 1, 8), 410 + i), rnd((1, 1, 8), 420 + i), 6 + i),
          f"decode {i}")
    s("update_and_fetch", lambda: p.update_and_fetch(rnd((1, 0, 8), 403)), "width 0")
    s("make_mask", lambda: p.make_mask(3, offset=5))
    s("make_mask", lambda: p.make_mask(1), "make_mask 1")
    r.view(group, cls, p)
    s("state", attr(p, "state"))
    s("meta_state", attr(p, "meta_state"))
    s("from_state", lambda: type(p).from_state(p.state, p.meta_state))
    s("offset", attr(p, "offset"))
    s("size", call(p, "size"))
    s("empty", call(p, "empty"))
    s("is_trimmable", call(p, "is_trimmable"))
    s("nbytes", attr(p, "nbytes"))
    s("memory_profile", lambda: p.memory_profile(16))
    s("prefix_cache_snapshot", call(p, "prefix_cache_snapshot"))

    def speculate():
        q = copy.deepcopy(p)
        gen = q.start_speculation(2)
        for i in range(2):
            q.accumulate_windows(rnd((1, 1, 8), 430 + i), rnd((1, 1, 8), 440 + i),
                                 9 + i)
        q.commit_speculation([1], gen)
        return q

    s("commit_speculation", speculate)

    def aborted():
        q = copy.deepcopy(p)
        gen = q.start_speculation(1)
        q.accumulate_windows(rnd((1, 1, 8), 450), rnd((1, 1, 8), 451), 9)
        q.abort_speculation(gen)
        return q

    s("abort_speculation", aborted)
    s("trim", lambda: copy.deepcopy(p).trim(1))

    def two():
        rows = []
        for i, width in enumerate((5, 9)):
            one = C.PoolingCache(ratio=4)
            one.accumulate_windows(rnd((1, width, 8), 460 + i),
                                   rnd((1, width, 8), 470 + i), 0)
            one.update_and_fetch(rnd((1, width // 4, 8), 480 + i))
            rows.append(one)
        return rows

    b = s("merge", lambda: type(p).merge(two()))
    if b is None:
        return
    cls = "BatchPoolingCache"
    s = lambda method, fn, key="": r.step(group, cls, method, fn, key)  # noqa: E731
    r.view(group, cls, b)
    s("prepare", lambda: b.prepare(lengths=[1, 1]))
    s("accumulate_windows", lambda: b.accumulate_windows(
        rnd((2, 1, 8), 490), rnd((2, 1, 8), 491), 9))
    s("update_and_fetch", lambda: b.update_and_fetch(rnd((2, 1, 8), 492)))
    s("finalize", call(b, "finalize"))
    s("make_mask", lambda: b.make_mask(1))
    s("make_mask", lambda: b.make_mask(2, offset=3), "make_mask 2")
    s("offset", attr(b, "offset"))
    s("state", attr(b, "state"))
    s("meta_state", attr(b, "meta_state"))
    s("extract", lambda: b.extract(1))

    def filtered():
        d = copy.deepcopy(b)
        d.filter(mx.array([1]))
        return d

    s("filter", filtered)
    s("extend", lambda: copy.deepcopy(b).extend(copy.deepcopy(b)))
    r.view(group, cls, b)
    s("size", call(b, "size"))
    s("empty", call(b, "empty"))
    s("nbytes", attr(b, "nbytes"))
    s("trim", lambda: copy.deepcopy(b).trim(1))
    s("empty", C.BatchPoolingCache(4, [0, 1]).empty, "empty batch")
    s("prepare", lambda: C.BatchPoolingCache(4, [0, 1]).prepare(left_padding=[1, 0]),
      "prepare left padding")


def functions(r, C, group):
    s = lambda method, fn, key="": r.step(group, "", method, fn, key)  # noqa: E731
    ccm = C.create_causal_mask
    s("create_causal_mask", lambda: ccm(4))
    s("create_causal_mask", lambda: ccm(4, 3), "offset")
    s("create_causal_mask", lambda: ccm(4, 3, window_size=2), "window")
    s("create_causal_mask", lambda: ccm(3, 2, right_padding=mx.array([1, 0])),
      "right padding")
    s("create_causal_mask", lambda: ccm(3, 2, left_padding=mx.array([0, 2])),
      "left padding")
    cam = C.create_attention_mask
    for args in ((1, 0, False, None), (4, 0, False, None), (4, 2, True, None),
                 (4, 2, False, 3), (1, 5, False, 3)):
        s("create_attention_mask.cache", lambda args=args: cam(*args), str(args))
    x = mx.arange(24).reshape(2, 3, 4)
    s("dynamic_roll", lambda: C.dynamic_roll(x, mx.array([[1], [2]]), axis=2))
    layers = types.SimpleNamespace(layers=[0, 0, 0])
    s("make_prompt_cache", lambda: C.make_prompt_cache(layers))
    s("make_prompt_cache", lambda: C.make_prompt_cache(layers, max_kv_size=16),
      "max_kv_size")
    own = types.SimpleNamespace(layers=[0], make_cache=lambda: ["own"])
    s("make_prompt_cache", lambda: C.make_prompt_cache(own), "model make_cache")
    s("should_quantize_kv_layer", lambda: [
        [C.should_quantize_kv_layer(i, n) for i in range(n)] for n in (1, 2, 3, 5)])

    def nbytes():
        a = C.KVCache()
        a.update_and_fetch(*kv(1, 3, 500))
        lst = C.CacheList(C.KVCache(), C.RotatingKVCache(max_size=4))
        return C.cache_nbytes([a, [lst, None], (a,)])

    s("cache_nbytes", nbytes)
    s("CacheMemory", lambda: C.CacheMemory(source_bytes=10, fixed_bytes=4))

    def footprint():
        a = C.KVCache()
        a.update_and_fetch(*kv(1, 3, 501))
        prof = a.memory_profile(3)
        return prof, prof.footprint(64)

    s("CacheMemory.footprint", footprint)


def sdpa(r, C, A, group):
    s = lambda method, fn, key="": r.step(group, "", method, fn, key)  # noqa: E731
    for dtype in (mx.float16, mx.bfloat16):
        dn = str(dtype).split(".")[-1]
        for L, K in ((1, 8), (4, 8)):
            q = rnd((1, 4, L, D), 600 + L, dtype)
            k, v = kv(1, K, 610 + L, dtype)
            bool_mask = mx.arange(K - L, K)[:, None] >= mx.arange(K)[None]
            add_mask = mx.where(bool_mask, 0.0, -1e4).astype(dtype)[None, None]
            masks = (("none", None), ("causal", "causal"), ("bool", bool_mask),
                     ("additive", add_mask))
            plain = C.KVCache()
            for mname, mask in masks:
                s("sdpa", lambda q=q, k=k, v=v, mask=mask: A.scaled_dot_product_attention(
                    q, k, v, cache=plain, scale=SCALE, mask=mask),
                  f"{dn} L{L} {mname}")
                sinks = rnd((4,), 620, dtype)
                s("sdpa", lambda q=q, k=k, v=v, mask=mask, sinks=sinks:
                  A.scaled_dot_product_attention(
                      q, k, v, cache=plain, scale=SCALE, mask=mask, sinks=sinks),
                  f"{dn} L{L} {mname} sinks")
            for bits in (4, 8):
                qc = C.QuantizedKVCache(group_size=32, bits=bits)
                qk, qv = qc.update_and_fetch(k, v)
                for mname, mask in masks:
                    s("quantized_sdpa", lambda q=q, qk=qk, qv=qv, qc=qc, mask=mask:
                      A.scaled_dot_product_attention(
                          q, qk, qv, cache=qc, scale=SCALE, mask=mask),
                      f"{dn} L{L} bits{bits} {mname}")
                s("quantized_sdpa", lambda q=q, qk=qk, qv=qv, qc=qc:
                  A.scaled_dot_product_attention(
                      q, qk, qv, cache=qc, scale=SCALE, mask=None,
                      sinks=mx.zeros((4,), dtype)),
                  f"{dn} L{L} bits{bits} sinks")
                s("quantized_sdpa", lambda q=q, qk=qk, qv=qv, bits=bits:
                  A.quantized_scaled_dot_product_attention(
                      q, qk, qv, scale=SCALE, mask="causal", group_size=32,
                      bits=bits), f"{dn} L{L} bits{bits} direct")
    # Two rows with GQA and a 4D pad mask: mlx-lm broadcasts it wrongly.
    q = rnd((2, 4, 3, D), 630, mx.float16)
    k, v = kv(2, 6, 631, mx.float16)
    qc = C.QuantizedKVCache(group_size=32, bits=8)
    qk, qv = qc.update_and_fetch(k, v)
    pad = (mx.arange(6)[None] >= mx.array([0, 2])[:, None])[:, None, None, :]
    s("quantized_sdpa.b2", lambda: A.scaled_dot_product_attention(
        q, qk, qv, cache=qc, scale=SCALE, mask=pad))

    h1, h4 = mx.zeros((1, 1, 8)), mx.zeros((1, 4, 8))
    filled = C.KVCache()
    filled.update_and_fetch(*kv(1, 5, 640))
    rot = C.RotatingKVCache(max_size=4)
    rot.update_and_fetch(*kv(1, 6, 641))
    for hname, h in (("h1", h1), ("h4", h4)):
        for cname, c in (("none", None), ("kv", filled), ("rotating", rot)):
            for kw in ({}, {"return_array": True}, {"window_size": 2}):
                s("create_attention_mask.base",
                  lambda h=h, c=c, kw=kw: A.create_attention_mask(h, c, **kw),
                  f"{hname} {cname} {sorted(kw.items())}")
    ac = C.ArraysCache(size=2, left_padding=[1, 0])
    s("create_ssm_mask", lambda: A.create_ssm_mask(mx.zeros((2, 3, 8)), ac))
    s("create_ssm_mask", lambda: A.create_ssm_mask(h4, None), "none")
    s("align_attention_mask_to_scores", lambda: A.align_attention_mask_to_scores(
        pad, mx.zeros((2, 2, 2, 3, 6))))


GROUPS = (
    "KVCache", "RotatingKVCache", "RotatingKVCache keep", "ConcatenateKVCache",
    "QuantizedKVCache 4", "QuantizedKVCache 8", "BufferedRotatingKVCache",
    "ChunkedKVCache", "SimpleKVCache", "StaticPrefixKVCache", "BatchKVCache",
    "BatchRotatingKVCache", "BatchQuantizedKVCache", "ArraysCache",
    "CacheList", "PoolingCache", "functions", "sdpa",
)


def run(C, A):
    r = Recorder()
    single_row(r, C, "KVCache", lambda: C.KVCache())
    single_row(r, C, "RotatingKVCache", lambda: C.RotatingKVCache(max_size=8))
    single_row(r, C, "RotatingKVCache keep",
               lambda: C.RotatingKVCache(max_size=8, keep=2))
    single_row(r, C, "ConcatenateKVCache", lambda: C.ConcatenateKVCache())
    single_row(r, C, "QuantizedKVCache 4",
               lambda: C.QuantizedKVCache(group_size=32, bits=4))
    single_row(r, C, "QuantizedKVCache 8",
               lambda: C.QuantizedKVCache(group_size=32, bits=8))
    single_row(r, C, "BufferedRotatingKVCache",
               lambda: C.BufferedRotatingKVCache(8, buffer_size=4))
    chunked(r, C, "ChunkedKVCache")
    simple(r, C, "SimpleKVCache")
    static_prefix(r, C, "StaticPrefixKVCache")
    batch(r, C, "BatchKVCache", lambda: C.BatchKVCache([1, 0]),
          lambda: C.KVCache())
    batch(r, C, "BatchRotatingKVCache", lambda: C.BatchRotatingKVCache(8, [1, 0]),
          lambda: C.RotatingKVCache(max_size=8))
    batch(r, C, "BatchQuantizedKVCache",
          lambda: C.BatchQuantizedKVCache([1, 0], group_size=32, bits=4),
          lambda: C.QuantizedKVCache(group_size=32, bits=4))
    arrays(r, C, "ArraysCache")
    cache_list(r, C, "CacheList")
    pooling(r, C, "PoolingCache")
    functions(r, C, "functions")
    sdpa(r, C, A, "sdpa")
    return r.rows


def main(mode):
    mx.set_default_device(mx.cpu)
    C, A = load(mode)
    rows = run(C, A)
    loaded = sorted(m for m in ("gmlx", "gmlx._vendor.mlx_vlm", "gmlx._vendor.mlx_lm")
                    if any(k == m or k.startswith(m + ".") for k in sys.modules))
    print(json.dumps({"mode": mode, "loaded": loaded,
                      "cache_module": C.__name__, "rows": rows}))


if __name__ == "__main__":
    main(sys.argv[1])
