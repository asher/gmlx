"""KVarN KV cache: variance-normalized quantized KV storage (BeeLlama wire).

KVarNKVCache stores the KV history in three regions, all rotated to the
WHT domain except the tail:

  sink    first sink_cap tokens, fp16 rotated stage rows, never quantized
  records full 128-token groups after the sink, quantized eagerly at seal
          time through kq.kvarn_quantize (codes uint32 + fp16 axes)
  live    the incomplete trailing group, fp16 rotated stage rows

Two fp16 side buffers make quantization reversible near the frontier: a
horizon copy of the last sealed group's rotated rows (MTP rollback across
one seal), and an original-domain tail of the last tail_cap tokens (the
precision-tail overlay served at full fidelity by the SDPA route, and the
trim substrate for deeper chat rewinds).

update_and_fetch returns KVarNView handles, not arrays: the kvarn SDPA
route (kvarn_sdpa.py) reads the cache regions directly, and any code path
that tries to use the views as arrays fails loudly instead of silently
attending to garbage. The class deliberately exposes neither ``bits`` nor
``to_quantized`` so no affine quantized-KV path ever claims it.

Trim never dequantizes: n tokens off the end succeed when the new
frontier lands in the live stage, in the horizon group, or inside tail
coverage (tail rows re-rotate to rebuild the live group bit-identically);
anything deeper returns 0 and the caller falls back to its rebuild path.

Head dim 128, single stream (B=1), group size 128. All widths in
{2, 3, 4, 5, 6, 8}, K and V independently.
"""

from __future__ import annotations

import sys

import mlx.core as mx

try:
    import mlx_kquant as kq
except ImportError:  # pragma: no cover - mlx_kquant always present in practice
    kq = None

GROUP = 128
HEAD_DIM = 128
KVARN_BITS = (2, 3, 4, 5, 6, 8)


class KVarNView:
    """Handle returned by KVarNKVCache.update_and_fetch. The kvarn SDPA
    route unwraps it; array use means the route was bypassed."""

    __slots__ = ("cache", "side")

    def __init__(self, cache, side):
        self.cache = cache
        self.side = side

    def _bypass(self, what):
        raise RuntimeError(
            f"[kvarn] {what} on a KVarN KV view: this model's attention "
            "bypassed the kvarn SDPA route. The arch is not kvarn-capable; "
            "rerun without --kv-quant-scheme kvarn."
        )

    def __getitem__(self, idx):
        self._bypass("indexing")

    def __len__(self):
        self._bypass("len()")

    def __iter__(self):
        self._bypass("iteration")

    def __array__(self, *a, **k):
        self._bypass("array conversion")

    def __repr__(self):
        return f"KVarNView({self.side}, offset={self.cache.offset})"


def _base_cache():
    from mlx_lm.models.cache import _BaseCache

    return _BaseCache


class KVarNKVCache(_base_cache()):
    kv_quant_scheme = "kvarn"
    kvarn_layout_version = 1
    gcap_step = 32
    tail_slack = 256

    def __init__(self, k_bits=6, v_bits=6, tail_tokens=1024, sink_tokens=GROUP):
        for bits in (k_bits, v_bits):
            if bits not in KVARN_BITS:
                raise ValueError(
                    f"[kvarn] bits must be one of {KVARN_BITS}, got {bits}."
                )
        if sink_tokens < GROUP or sink_tokens % GROUP:
            raise ValueError("[kvarn] sink_tokens must be a positive multiple of 128.")
        if tail_tokens < 0 or tail_tokens % GROUP:
            raise ValueError(
                "[kvarn] tail_tokens must be a multiple of 128 (0 disables)."
            )
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.sink_cap = sink_tokens
        self.tail_cap = tail_tokens
        self.offset = 0
        self.n_sealed = 0
        self.horizon_valid = False
        self.tail_start = 0
        self.tail_end = 0
        self.codes_k = None
        self.axes_k = None
        self.codes_v = None
        self.axes_v = None
        self.stage_k = None
        self.stage_v = None
        self.horizon_k = None
        self.horizon_v = None
        self.tail_k = None
        self.tail_v = None

    # -- derived watermarks -------------------------------------------------

    @property
    def sink_used(self):
        return min(self.offset, self.sink_cap)

    @property
    def live_len(self):
        return self.offset - self.sink_used - GROUP * self.n_sealed

    @property
    def tail_len(self):
        return self.tail_end - self.tail_start

    def _allocated(self):
        return self.stage_k is not None and self.stage_k.shape[-1] == HEAD_DIM

    # -- lifecycle ----------------------------------------------------------

    def _alloc(self, h):
        s_rows = self.sink_cap + GROUP
        self.stage_k = mx.zeros((1, h, s_rows, HEAD_DIM), mx.float16)
        self.stage_v = mx.zeros((1, h, s_rows, HEAD_DIM), mx.float16)
        self.horizon_k = mx.zeros((1, h, GROUP, HEAD_DIM), mx.float16)
        self.horizon_v = mx.zeros((1, h, GROUP, HEAD_DIM), mx.float16)
        t_rows = self.tail_cap + self.tail_slack if self.tail_cap else 1
        self.tail_k = mx.zeros((1, h, t_rows, HEAD_DIM), mx.float16)
        self.tail_v = mx.zeros((1, h, t_rows, HEAD_DIM), mx.float16)
        g = self.gcap_step
        self.codes_k = mx.zeros((1, h, g, 512 * self.k_bits), mx.uint32)
        self.codes_v = mx.zeros((1, h, g, 512 * self.v_bits), mx.uint32)
        self.axes_k = mx.zeros((1, h, g, 3, HEAD_DIM), mx.float16)
        self.axes_v = mx.zeros((1, h, g, 3, HEAD_DIM), mx.float16)

    def _ensure_gcap(self, groups):
        gcap = self.codes_k.shape[2]
        if groups <= gcap:
            return
        add = (groups - gcap + self.gcap_step - 1) // self.gcap_step
        add *= self.gcap_step

        def grow(x):
            pad = mx.zeros(x.shape[:2] + (add,) + x.shape[3:], x.dtype)
            return mx.concatenate([x, pad], axis=2)

        self.codes_k, self.codes_v = grow(self.codes_k), grow(self.codes_v)
        self.axes_k, self.axes_v = grow(self.axes_k), grow(self.axes_v)

    def update_and_fetch(self, keys, values):
        if keys.ndim != 4 or keys.shape[-1] != HEAD_DIM or values.shape[-1] != HEAD_DIM:
            raise ValueError(
                "[kvarn] KVarNKVCache requires head_dim 128 K and V, got "
                f"K {tuple(keys.shape)} V {tuple(values.shape)}."
            )
        if keys.shape[0] != 1:
            raise ValueError("[kvarn] KVarNKVCache is single-stream (B=1).")
        if keys.dtype != mx.float16:
            keys = keys.astype(mx.float16)
        if values.dtype != mx.float16:
            values = values.astype(mx.float16)
        if not self._allocated() or self.stage_k.shape[1] != keys.shape[1]:
            if self.offset:
                raise RuntimeError("[kvarn] cache head count changed mid-stream.")
            self._alloc(keys.shape[1])
        self._write_tail(keys, values)
        rk = kq.kvarn_rotate(keys)
        rv = kq.kvarn_rotate(values)
        self._append_rotated(rk, rv)
        self.offset += keys.shape[2]
        return KVarNView(self, "k"), KVarNView(self, "v")

    def _write_tail(self, keys, values):
        if not self.tail_cap:
            return
        n = keys.shape[2]
        if n >= self.tail_cap:
            self.tail_k[:, :, : self.tail_cap] = keys[:, :, -self.tail_cap :]
            self.tail_v[:, :, : self.tail_cap] = values[:, :, -self.tail_cap :]
            self.tail_start, self.tail_end = 0, self.tail_cap
            return
        if self.tail_end + n > self.tail_k.shape[2]:
            keep = min(self.tail_len, self.tail_cap - n)
            if keep:
                src = slice(self.tail_end - keep, self.tail_end)
                self.tail_k[:, :, :keep] = self.tail_k[:, :, src]
                self.tail_v[:, :, :keep] = self.tail_v[:, :, src]
            self.tail_start, self.tail_end = 0, keep
        e = self.tail_end
        self.tail_k[:, :, e : e + n] = keys
        self.tail_v[:, :, e : e + n] = values
        self.tail_end = e + n
        self.tail_start = max(self.tail_start, self.tail_end - self.tail_cap)

    def _append_rotated(self, rk, rv):
        pos = self.offset
        n = rk.shape[2]
        a = 0
        if pos < self.sink_cap:
            t = min(n, self.sink_cap - pos)
            self.stage_k[:, :, pos : pos + t] = rk[:, :, :t]
            self.stage_v[:, :, pos : pos + t] = rv[:, :, :t]
            a = t
        while a < n:
            live = (pos + a) - self.sink_cap - GROUP * self.n_sealed
            if live > 0:
                t = min(GROUP - live, n - a)
                s = self.sink_cap + live
                self.stage_k[:, :, s : s + t] = rk[:, :, a : a + t]
                self.stage_v[:, :, s : s + t] = rv[:, :, a : a + t]
                a += t
                if live + t == GROUP:
                    s0 = self.sink_cap
                    self._seal(
                        self.stage_k[:, :, s0 : s0 + GROUP],
                        self.stage_v[:, :, s0 : s0 + GROUP],
                    )
                continue
            m = (n - a) // GROUP
            if m:
                self._seal(rk[:, :, a : a + m * GROUP], rv[:, :, a : a + m * GROUP])
                a += m * GROUP
            rem = n - a
            if rem:
                s0 = self.sink_cap
                self.stage_k[:, :, s0 : s0 + rem] = rk[:, :, a:]
                self.stage_v[:, :, s0 : s0 + rem] = rv[:, :, a:]
                a = n

    def _seal(self, rk_block, rv_block):
        """Quantize one or more complete rotated groups into records and
        refresh the horizon with the last of them."""
        m = rk_block.shape[2] // GROUP
        g = self.n_sealed
        self._ensure_gcap(g + m)
        ck, ak = kq.kvarn_quantize(rk_block, self.k_bits, "k")
        cv, av = kq.kvarn_quantize(rv_block, self.v_bits, "v")
        self.codes_k[:, :, g : g + m] = ck
        self.axes_k[:, :, g : g + m] = ak
        self.codes_v[:, :, g : g + m] = cv
        self.axes_v[:, :, g : g + m] = av
        self.horizon_k = rk_block[:, :, -GROUP:]
        self.horizon_v = rv_block[:, :, -GROUP:]
        self.horizon_valid = True
        self.n_sealed = g + m

    # -- attention-side accessors -------------------------------------------

    def materialize(self, dtype=mx.float16):
        """Full rotated-domain K/V at the given dtype (prefill route and
        parity references)."""
        outs = []
        for side in ("k", "v"):
            stage = getattr(self, f"stage_{side}")
            parts = [stage[:, :, : self.sink_used].astype(dtype)]
            if self.n_sealed:
                parts.append(
                    kq.kvarn_dequant(
                        getattr(self, f"codes_{side}")[:, :, : self.n_sealed],
                        getattr(self, f"axes_{side}")[:, :, : self.n_sealed],
                        self.k_bits if side == "k" else self.v_bits,
                        side,
                        dtype=dtype,
                    )
                )
            if self.live_len:
                s0 = self.sink_cap
                parts.append(stage[:, :, s0 : s0 + self.live_len].astype(dtype))
            outs.append(mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0])
        return outs[0], outs[1]

    def tail_slices(self, n_tokens):
        """Original-domain fp16 rows for the last n_tokens tokens."""
        if n_tokens > self.tail_len:
            raise ValueError("[kvarn] tail request beyond coverage.")
        s = slice(self.tail_end - n_tokens, self.tail_end)
        return self.tail_k[:, :, s], self.tail_v[:, :, s]

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    # -- trim ---------------------------------------------------------------

    def is_trimmable(self):
        return True

    def _trim_plan(self, n):
        """How to serve a trim of n tokens, or None when it would need
        dequantizing history that only exists as records."""
        new_off = self.offset - n
        if new_off <= self.sink_cap:
            return ("sink",)
        body = new_off - self.sink_cap
        g = body // GROUP
        live = body % GROUP
        if g == self.n_sealed:
            return ("live",)
        if g == self.n_sealed - 1 and self.horizon_valid:
            return ("horizon", live)
        cover0 = self.offset - self.tail_len
        if self.sink_cap + g * GROUP >= cover0:
            return ("tail", g, live, cover0)
        return None

    def _can_trim(self, n):
        n = min(int(n), self.offset)
        return n <= 0 or self._trim_plan(n) is not None

    def trim(self, n):
        n = min(int(n), self.offset)
        if n <= 0:
            return 0
        plan = self._trim_plan(n)
        if plan is None:
            return 0
        new_off = self.offset - n
        kind = plan[0]
        if kind == "sink":
            self.n_sealed = 0
            self.horizon_valid = False
        elif kind == "horizon":
            live = plan[1]
            s0 = self.sink_cap
            if live:
                self.stage_k[:, :, s0 : s0 + live] = self.horizon_k[:, :, :live]
                self.stage_v[:, :, s0 : s0 + live] = self.horizon_v[:, :, :live]
            self.n_sealed -= 1
            self.horizon_valid = False
        elif kind == "tail":
            g, live, cover0 = plan[1:]
            if live:
                a = self.sink_cap + g * GROUP - cover0
                tk = self.tail_k[:, :, self.tail_start + a : self.tail_start + a + live]
                tv = self.tail_v[:, :, self.tail_start + a : self.tail_start + a + live]
                s0 = self.sink_cap
                self.stage_k[:, :, s0 : s0 + live] = kq.kvarn_rotate(tk)
                self.stage_v[:, :, s0 : s0 + live] = kq.kvarn_rotate(tv)
            self.n_sealed = g
            self.horizon_valid = False
        self.tail_end = max(self.tail_start, self.tail_end - n)
        self.offset = new_off
        return n

    # -- serialization ------------------------------------------------------

    _STATE_FIELDS = (
        "codes_k",
        "axes_k",
        "codes_v",
        "axes_v",
        "stage_k",
        "stage_v",
        "horizon_k",
        "horizon_v",
        "tail_k",
        "tail_v",
    )

    @property
    def state(self):
        if not self._allocated():
            # Fixed-arity placeholder so an empty cache still round-trips
            # (safetensors rejects zero-size arrays). meta_state's
            # allocated flag tells the setter to discard it.
            z16 = mx.zeros((1, 1, 1, 1), mx.float16)
            z32 = mx.zeros((1, 1, 1, 1), mx.uint32)
            return (z32, z16, z32, z16, z16, z16, z16, z16, z16, z16)
        return tuple(getattr(self, f) for f in self._STATE_FIELDS)

    @state.setter
    def state(self, v):
        for f, a in zip(self._STATE_FIELDS, v, strict=True):
            setattr(self, f, a)

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.kvarn_layout_version,
                    1 if self._allocated() else 0,
                    self.offset,
                    self.n_sealed,
                    self.k_bits,
                    self.v_bits,
                    self.sink_cap,
                    self.tail_cap,
                    self.tail_start,
                    self.tail_end,
                    1 if self.horizon_valid else 0,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        (
            version,
            allocated,
            self.offset,
            self.n_sealed,
            self.k_bits,
            self.v_bits,
            self.sink_cap,
            self.tail_cap,
            self.tail_start,
            self.tail_end,
            horizon_valid,
        ) = map(int, v)
        if version != self.kvarn_layout_version:
            raise ValueError(
                f"[kvarn] cache layout version {version} does not match this "
                f"build ({self.kvarn_layout_version}); refusing to restore."
            )
        self.horizon_valid = bool(horizon_valid)
        if not allocated:
            for f in self._STATE_FIELDS:
                setattr(self, f, None)

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        if not self._allocated():
            return 0
        return sum(getattr(self, f).nbytes for f in self._STATE_FIELDS)

    # -- conversion ---------------------------------------------------------

    @classmethod
    def from_cache(cls, cache, k_bits=6, v_bits=6, tail_tokens=1024, sink_tokens=GROUP):
        """Bulk-convert a plain KV cache's history (bit-identical to having
        accumulated the same tokens incrementally)."""
        out = cls(
            k_bits=k_bits,
            v_bits=v_bits,
            tail_tokens=tail_tokens,
            sink_tokens=sink_tokens,
        )
        off = int(getattr(cache, "offset", 0))
        if off:
            keys, values = cache.state
            out.update_and_fetch(keys[:, :, :off], values[:, :, :off])
        return out


def ensure_registered():
    """Graft KVarNKVCache onto both cache namespaces so snapshot and
    prompt-cache restores resolve it by name. Upstream wins if the name
    ever appears there."""
    import mlx_lm.models.cache as lm_cache

    vlm_cache = sys.modules.get("mlx_vlm.models.cache")
    for mod in (lm_cache, vlm_cache):
        if mod is not None and not hasattr(mod, "KVarNKVCache"):
            mod.KVarNKVCache = KVarNKVCache


def convert_prompt_cache(
    prompt_cache, k_bits=6, v_bits=6, tail_tokens=1024, sink_tokens=GROUP
):
    """Replace every plain KVCache entry with a KVarNKVCache (converting
    any existing history). Returns the number of layers converted; other
    cache kinds are left untouched."""
    from .cache_compat import cache_types

    ensure_registered()
    kv_types = cache_types("KVCache")
    n = 0
    for i, c in enumerate(prompt_cache):
        if type(c) in kv_types:
            prompt_cache[i] = KVarNKVCache.from_cache(
                c,
                k_bits=k_bits,
                v_bits=v_bits,
                tail_tokens=tail_tokens,
                sink_tokens=sink_tokens,
            )
            n += 1
    return n
