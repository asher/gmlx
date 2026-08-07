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

    def _raw_single(self):
        """Recovered original-domain K/V [1, H, offset, D]: body un-rotated
        from the materialized regions (WHT is self-inverse; one fp16
        rounding), the last tail_len tokens taken verbatim from the raw
        tail rows. The BatchKVarNKVCache._raw_rows twin."""
        mk, mv = self.materialize()
        rk = kq.kvarn_rotate(mk)
        rv = kq.kvarn_rotate(mv)
        t = self.tail_len
        if t:
            tk, tv = self.tail_slices(t)
            rk[:, :, self.offset - t :] = tk
            rv[:, :, self.offset - t :] = tv
        return rk, rv

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


class BatchKVarNKVCache(_base_cache()):
    """Batched KVarN KV cache: the same region layout with a leading batch
    axis, a uniform buffer watermark ``_idx`` and per-row ``left_padding``
    (the BatchKVCache model). Pad rows hold whatever K/V the model produced
    for pad positions; they seal into records like real tokens and are
    excluded at attention time by per-row starts, keeping the region map
    uniform across rows -- the invariant the fused kernel's single-n layout
    requires. ``filter`` therefore never compacts left padding (a non-group
    shift would re-index every sealed record), and ``extend`` realigns the
    shorter side by rebuilding its rows from recovered raw K/V (one extra
    quantization pass on the admitted rows). No horizon and no trim:
    batched MTP declines kvarn, matching BatchQuantizedKVCache's
    ``is_trimmable() -> False``."""

    kv_quant_scheme = "kvarn"
    kvarn_layout_version = 1
    gcap_step = 32
    tail_slack = 256

    def __init__(
        self, left_padding, k_bits=6, v_bits=6, tail_tokens=1024, sink_tokens=GROUP
    ):
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
        self.left_padding = mx.array(left_padding).astype(mx.int32)
        self._idx = 0
        self.n_sealed = 0
        self.tail_start = 0
        self.tail_end = 0
        self._right_padding = None
        self.codes_k = None
        self.axes_k = None
        self.codes_v = None
        self.axes_v = None
        self.stage_k = None
        self.stage_v = None
        self.tail_k = None
        self.tail_v = None

    # -- derived watermarks -------------------------------------------------

    @property
    def offset(self):
        return self._idx - self.left_padding

    @property
    def sink_used(self):
        return min(self._idx, self.sink_cap)

    @property
    def live_len(self):
        return self._idx - self.sink_used - GROUP * self.n_sealed

    @property
    def tail_len(self):
        return self.tail_end - self.tail_start

    def _allocated(self):
        return self.stage_k is not None and self.stage_k.shape[-1] == HEAD_DIM

    # -- lifecycle ----------------------------------------------------------

    def _alloc(self, b, h):
        s_rows = self.sink_cap + GROUP
        self.stage_k = mx.zeros((b, h, s_rows, HEAD_DIM), mx.float16)
        self.stage_v = mx.zeros((b, h, s_rows, HEAD_DIM), mx.float16)
        t_rows = self.tail_cap + self.tail_slack if self.tail_cap else 1
        self.tail_k = mx.zeros((b, h, t_rows, HEAD_DIM), mx.float16)
        self.tail_v = mx.zeros((b, h, t_rows, HEAD_DIM), mx.float16)
        g = self.gcap_step
        self.codes_k = mx.zeros((b, h, g, 512 * self.k_bits), mx.uint32)
        self.codes_v = mx.zeros((b, h, g, 512 * self.v_bits), mx.uint32)
        self.axes_k = mx.zeros((b, h, g, 3, HEAD_DIM), mx.float16)
        self.axes_v = mx.zeros((b, h, g, 3, HEAD_DIM), mx.float16)

    _ensure_gcap = KVarNKVCache._ensure_gcap
    _write_tail = KVarNKVCache._write_tail
    tail_slices = KVarNKVCache.tail_slices
    materialize = KVarNKVCache.materialize

    def update_and_fetch(self, keys, values):
        if keys.ndim != 4 or keys.shape[-1] != HEAD_DIM or values.shape[-1] != HEAD_DIM:
            raise ValueError(
                "[kvarn] BatchKVarNKVCache requires head_dim 128 K and V, got "
                f"K {tuple(keys.shape)} V {tuple(values.shape)}."
            )
        if keys.shape[0] != self.left_padding.shape[0]:
            raise ValueError(
                f"[kvarn] batch size {keys.shape[0]} does not match "
                f"left_padding ({self.left_padding.shape[0]} rows)."
            )
        if keys.dtype != mx.float16:
            keys = keys.astype(mx.float16)
        if values.dtype != mx.float16:
            values = values.astype(mx.float16)
        if not self._allocated() or self.stage_k.shape[1] != keys.shape[1]:
            if self._idx:
                raise RuntimeError("[kvarn] cache head count changed mid-stream.")
            self._alloc(keys.shape[0], keys.shape[1])
        self._write_tail(keys, values)
        rk = kq.kvarn_rotate(keys)
        rv = kq.kvarn_rotate(values)
        self._append_rotated(rk, rv)
        self._idx += keys.shape[2]
        return KVarNView(self, "k"), KVarNView(self, "v")

    def _append_rotated(self, rk, rv):
        pos = self._idx
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
        m = rk_block.shape[2] // GROUP
        g = self.n_sealed
        self._ensure_gcap(g + m)
        ck, ak = kq.kvarn_quantize(rk_block, self.k_bits, "k")
        cv, av = kq.kvarn_quantize(rv_block, self.v_bits, "v")
        self.codes_k[:, :, g : g + m] = ck
        self.axes_k[:, :, g : g + m] = ak
        self.codes_v[:, :, g : g + m] = cv
        self.axes_v[:, :, g : g + m] = av
        self.n_sealed = g + m

    # -- batch ops ----------------------------------------------------------

    _ARRAY_FIELDS = (
        "codes_k",
        "axes_k",
        "codes_v",
        "axes_v",
        "stage_k",
        "stage_v",
        "tail_k",
        "tail_v",
    )

    def filter(self, batch_indices):
        for f in self._ARRAY_FIELDS:
            a = getattr(self, f)
            if a is not None:
                setattr(self, f, a[batch_indices])
        self.left_padding = self.left_padding[batch_indices]

    def _normalize_tail(self):
        """Move the tail ring window to [0, tail_len) so two caches at the
        same _idx concatenate row-wise."""
        t = self.tail_len
        if not t or self.tail_start == 0:
            self.tail_start, self.tail_end = 0, t
            return
        s = slice(self.tail_start, self.tail_end)
        self.tail_k[:, :, :t] = self.tail_k[:, :, s]
        self.tail_v[:, :, :t] = self.tail_v[:, :, s]
        self.tail_start, self.tail_end = 0, t

    def _raw_rows(self):
        """Recovered original-domain K/V [B, H, _idx, D]: body un-rotated
        from the materialized regions (WHT is self-inverse; one fp16
        rounding), the last tail_len tokens taken verbatim from the raw
        tail rows."""
        mk, mv = self.materialize()
        rk = kq.kvarn_rotate(mk)
        rv = kq.kvarn_rotate(mv)
        t = self.tail_len
        if t:
            tk, tv = self.tail_slices(t)
            rk[:, :, self._idx - t :] = tk
            rv[:, :, self._idx - t :] = tv
        return rk, rv

    def _realigned(self, new_idx):
        """A fresh cache holding the same rows right-justified to new_idx
        (rows re-quantize once at the new group alignment)."""
        delta = new_idx - self._idx
        out = BatchKVarNKVCache(
            self.left_padding + delta,
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            tail_tokens=self.tail_cap,
            sink_tokens=self.sink_cap,
        )
        rk, rv = self._raw_rows()
        b, h = rk.shape[0], rk.shape[1]
        pad_k = mx.zeros((b, h, delta, HEAD_DIM), mx.float16)
        pad_v = mx.zeros((b, h, delta, HEAD_DIM), mx.float16)
        out.update_and_fetch(
            mx.concatenate([pad_k, rk], axis=2),
            mx.concatenate([pad_v, rv], axis=2),
        )
        return out

    def _compatible(self, other):
        return type(other) is BatchKVarNKVCache and (
            self.k_bits,
            self.v_bits,
            self.sink_cap,
            self.tail_cap,
        ) == (other.k_bits, other.v_bits, other.sink_cap, other.tail_cap)

    def extend(self, other):
        if not self._compatible(other):
            raise ValueError(
                f"[kvarn] cannot extend with {type(other).__name__} "
                "(mismatched class or kvarn parameters)."
            )
        if self._idx == 0 and other._idx == 0:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            return
        if self._idx == 0 or other._idx == 0:
            raise ValueError(
                "[kvarn] cannot extend an empty batch with a filled one "
                "(serve always prefills before admission)."
            )
        target = max(self._idx, other._idx)
        a = self if self._idx == target else self._realigned(target)
        b = other if other._idx == target else other._realigned(target)
        a._normalize_tail()
        b._normalize_tail()
        for f in self._ARRAY_FIELDS:
            xa, xb = getattr(a, f), getattr(b, f)
            gcap = max(xa.shape[2], xb.shape[2])
            grown = []
            for x in (xa, xb):
                if x.shape[2] < gcap:
                    pad = mx.zeros(
                        x.shape[:2] + (gcap - x.shape[2],) + x.shape[3:], x.dtype
                    )
                    x = mx.concatenate([x, pad], axis=2)
                grown.append(x)
            setattr(self, f, mx.concatenate(grown, axis=0))
        self.left_padding = mx.concatenate([a.left_padding, b.left_padding])
        self._idx = target
        self.n_sealed = a.n_sealed
        self.tail_start, self.tail_end = a.tail_start, a.tail_end

    def extract(self, idx):
        """Row ``idx`` as a single-stream KVarNKVCache with the left padding
        stripped (the BatchKVCache.extract model). A pad-free row adopts
        copies of its region buffers bit-exactly (horizon stays unarmed);
        a padded row rebuilds from recovered raw K/V, one re-quantization
        pass, the extend-admission cost."""
        pad = int(self.left_padding[idx].item())
        out = KVarNKVCache(
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            tail_tokens=self.tail_cap,
            sink_tokens=self.sink_cap,
        )
        if self._idx - pad <= 0:
            return out
        if pad == 0:
            h = self.stage_k.shape[1]
            for f in self._ARRAY_FIELDS:
                setattr(out, f, mx.contiguous(getattr(self, f)[idx : idx + 1]))
            out.horizon_k = mx.zeros((1, h, GROUP, HEAD_DIM), mx.float16)
            out.horizon_v = mx.zeros((1, h, GROUP, HEAD_DIM), mx.float16)
            out.offset = self._idx
            out.n_sealed = self.n_sealed
            out.tail_start, out.tail_end = self.tail_start, self.tail_end
            return out
        state = tuple(getattr(self, f)[idx : idx + 1] for f in self._ARRAY_FIELDS)
        row = type(self).from_state(
            state + (self.left_padding[idx : idx + 1],), self.meta_state
        )
        rk, rv = row._raw_rows()
        out.update_and_fetch(rk[:, :, pad:], rv[:, :, pad:])
        return out

    @classmethod
    def merge(cls, caches):
        """Batch single-stream KVarNKVCache rows, right-justified (the
        BatchKVCache.merge model). One filled row adopts copies of its
        region buffers bit-exactly; multi-row merges rebuild from recovered
        raw K/V (one re-quantization pass per row)."""
        if not caches:
            raise ValueError("[kvarn] merge requires at least one cache.")
        first = caches[0]
        params = (
            (first.k_bits, first.v_bits, first.sink_cap, first.tail_cap)
            if isinstance(first, KVarNKVCache)
            else None
        )
        for c in caches:
            if (
                not isinstance(c, KVarNKVCache)
                or (c.k_bits, c.v_bits, c.sink_cap, c.tail_cap) != params
            ):
                raise ValueError(
                    "[kvarn] merge requires KVarNKVCache rows with matching "
                    "kvarn parameters."
                )
        lengths = [c.offset for c in caches]
        max_len = max(lengths)
        out = cls(
            [max_len - n for n in lengths],
            k_bits=first.k_bits,
            v_bits=first.v_bits,
            tail_tokens=first.tail_cap,
            sink_tokens=first.sink_cap,
        )
        if max_len == 0:
            return out
        if len(caches) == 1:
            for f in cls._ARRAY_FIELDS:
                setattr(out, f, mx.contiguous(getattr(first, f)))
            out._idx = first.offset
            out.n_sealed = first.n_sealed
            out.tail_start, out.tail_end = first.tail_start, first.tail_end
            return out
        h = next(c.stage_k.shape[1] for c in caches if c.offset)
        slab_k = mx.zeros((len(caches), h, max_len, HEAD_DIM), mx.float16)
        slab_v = mx.zeros((len(caches), h, max_len, HEAD_DIM), mx.float16)
        for i, c in enumerate(caches):
            if not c.offset:
                continue
            rk, rv = c._raw_single()
            slab_k[i : i + 1, :, max_len - c.offset :] = rk
            slab_v[i : i + 1, :, max_len - c.offset :] = rv
        out.update_and_fetch(slab_k, slab_v)
        return out

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        del lengths
        if left_padding is not None:
            if self._idx:
                raise ValueError(
                    "[kvarn] left padding can only be added to an empty cache."
                )
            self.left_padding = self.left_padding + mx.array(left_padding).astype(
                mx.int32
            )
        if right_padding is not None:
            self._right_padding = right_padding

    def finalize(self):
        # All-zero right padding (a lone warm row) needs no roll; anything
        # real would shift sealed records per row, which kvarn cannot do.
        rp = getattr(self, "_right_padding", None)
        if rp is not None and any(rp):
            raise RuntimeError(
                "[kvarn] mixed warm/cold prefill (right padding) is not "
                "supported under kvarn KV."
            )
        self._right_padding = None

    def make_mask(self, N, return_array=False, **kwargs):
        del return_array
        from mlx_vlm.models.cache import create_causal_mask

        mask = create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )
        if N == 1 and not any(v is not None for v in kwargs.values()):
            from .quantized_sdpa_fix import _register_starts

            _register_starts(mask, self.left_padding)
        return mask

    def is_trimmable(self):
        return False

    def trim(self, n):
        return 0

    # -- serialization ------------------------------------------------------

    @property
    def state(self):
        if not self._allocated():
            z16 = mx.zeros((1, 1, 1, 1), mx.float16)
            z32 = mx.zeros((1, 1, 1, 1), mx.uint32)
            return (z32, z16, z32, z16, z16, z16, z16, z16, self.left_padding)
        return tuple(getattr(self, f) for f in self._ARRAY_FIELDS) + (
            self.left_padding,
        )

    @state.setter
    def state(self, v):
        for f, a in zip(self._ARRAY_FIELDS + ("left_padding",), v, strict=True):
            setattr(self, f, a)

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.kvarn_layout_version,
                    1 if self._allocated() else 0,
                    self._idx,
                    self.n_sealed,
                    self.k_bits,
                    self.v_bits,
                    self.sink_cap,
                    self.tail_cap,
                    self.tail_start,
                    self.tail_end,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        (
            version,
            allocated,
            self._idx,
            self.n_sealed,
            self.k_bits,
            self.v_bits,
            self.sink_cap,
            self.tail_cap,
            self.tail_start,
            self.tail_end,
        ) = map(int, v)
        if version != self.kvarn_layout_version:
            raise ValueError(
                f"[kvarn] cache layout version {version} does not match this "
                f"build ({self.kvarn_layout_version}); refusing to restore."
            )
        if not allocated:
            for f in self._ARRAY_FIELDS:
                setattr(self, f, None)

    def size(self):
        return self._idx

    def empty(self):
        return self._idx == 0

    @property
    def nbytes(self):
        if not self._allocated():
            return 0
        return sum(getattr(self, f).nbytes for f in self._ARRAY_FIELDS)


def ensure_registered():
    """Graft the kvarn cache classes onto both cache namespaces so snapshot
    and prompt-cache restores resolve them by name. Upstream wins if the
    names ever appear there."""
    import mlx_lm.models.cache as lm_cache

    vlm_cache = sys.modules.get("mlx_vlm.models.cache")
    for mod in (lm_cache, vlm_cache):
        if mod is None:
            continue
        if not hasattr(mod, "KVarNKVCache"):
            mod.KVarNKVCache = KVarNKVCache
        if not hasattr(mod, "BatchKVarNKVCache"):
            mod.BatchKVarNKVCache = BatchKVarNKVCache


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
