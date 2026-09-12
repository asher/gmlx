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

Trim dequantizes only as a last resort: n tokens off the end are exact
when the new frontier lands in the live stage, in the horizon group, or
inside tail coverage (tail rows re-rotate to rebuild the live group
bit-identically). Deeper than that, the frontier group's records
dequantize into the live stage, one extra quantization round trip on
those rows. A rotating window returns 0 when the frontier falls in
evicted history, and the caller falls back to its rebuild path.

Head dim 128, 256 or 512, single stream (B=1), group size 128. Heads
wider than 128 quantize as D/128 independent 128-dim slices per group,
slice-minor in the codes with one axes triplet per slice (the kq wire).
All widths in {2, 3, 4, 5, 6, 8}, K and V independently.
KVarNRotatingKVCache bounds the record region to a --max-kv-size window
by compacting the oldest sealed groups away at update entry.
"""

from __future__ import annotations

import logging
import sys

import mlx.core as mx

try:
    import mlx_kquant as kq
except ImportError:  # pragma: no cover - mlx_kquant always present in practice
    kq = None

GROUP = 128
HEAD_DIMS = (128, 256, 512)
KVARN_BITS = (2, 3, 4, 5, 6, 8)
KVARN_DEFAULT_TAIL = 1024

_log = logging.getLogger(__name__)

_META_ARITY_MSG = (
    "[kvarn] cache metadata arity does not match this cache class; the "
    "state was written by a different kvarn cache class or layout."
)


def kvarn_head_dims(model):
    """Candidate attention head dims, {-1} for MLA latents, empty when
    unknown. Mixed-dim archs (gemma-4: sliding head_dim + global
    global_head_dim) contribute every dim; any supported one suffices since
    conversion is per-layer and update_and_fetch validates per instance."""
    for holder in (model, getattr(model, "language_model", None)):
        args = getattr(holder, "args", None) or getattr(holder, "config", None)
        if args is None:
            continue
        if (getattr(args, "kv_lora_rank", None)
                or getattr(args, "compress_ratios", None)):
            # deepseek_v4 stores a compressed latent per layer and has no
            # kv_lora_rank field.
            return {-1}
        dims = set()
        for key in ("head_dim", "global_head_dim"):
            hd = getattr(args, key, None)
            if hd:
                dims.add(int(hd))
        if dims:
            return dims
        hs = getattr(args, "hidden_size", None)
        nh = getattr(args, "num_attention_heads", None)
        if hs and nh:
            return {int(hs) // int(nh)}
    return set()


def kvarn_resolve_kwargs(model, kv_bits=None, value_bits=None, tail_tokens=None,
                         rotating_window=None, key_bits=None):
    """resolve_kv_quant_policy kwargs for the kvarn scheme on ``model``.
    The one place widths, tail, window and the model-shape decline are
    assembled: CLI, chat, serve load, the serve batch seam and the MTP
    spec path all resolve through it. ``key_bits``/``value_bits`` (serve's
    KV_KEY_BITS/KV_VALUE_BITS) win over the GMLX_KVARN_BITS split, which
    wins over ``kv_bits`` for both sides."""
    k_bits, v_bits = kvarn_widths(kv_bits)
    if key_bits is not None:
        k_bits = int(key_bits)
    if value_bits is not None:
        v_bits = int(value_bits)
    from gmlx.cache.kvarn_sdpa import kvarn_row_ends_ok

    return dict(
        scheme="kvarn",
        kv_bits=k_bits,
        value_bits=v_bits,
        tail_tokens=KVARN_DEFAULT_TAIL if tail_tokens is None else int(tail_tokens),
        rotating_window=rotating_window,
        scheme_reason=kvarn_unsupported(model),
        row_ends_ok=kvarn_row_ends_ok(),
    )


def kvarn_unsupported(model) -> str | None:
    """Reason string when the kvarn scheme cannot serve this model, else
    None. Coverage is partial by design (sliding windows and recurrent
    layers stay fp16); the reason fires only when zero layers are
    eligible. Model shape only -- the policy resolver owns the width,
    tail and window checks."""
    from gmlx.envflags import env_bool

    if not env_bool("GMLX_KVARN", True):
        return "disabled (GMLX_KVARN=0)"
    from gmlx.cache.kvarn_sdpa import kvarn_ops_missing

    reason = kvarn_ops_missing()
    if reason:
        return reason
    dims = kvarn_head_dims(model)
    if dims == {-1}:
        return "MLA latent KV cache (K and V share storage)"
    if not dims & set(HEAD_DIMS):
        shown = "/".join(str(d) for d in sorted(dims)) or "unknown"
        return f"head_dim {shown} (kvarn supports 128/256/512)"
    from gmlx.models.gemma4.owned import is_owned_language_model

    holders = (model, getattr(model, "language_model", None))
    if any(is_owned_language_model(h) for h in holders if h is not None):
        # The owned tree's attention calls an import-time sdpa alias the
        # kvarn sweep never rebinds; unreachable today (the shared-KV
        # drafter gate declines first), declined here so a future drafter
        # change fails informatively at setup, not mid-forward.
        return "gemma-4 owned (MTP) tree"
    return None


def kvarn_widths(kv_bits):
    """K/V bit widths for kvarn: kv_bits (default 6) for both sides, or a
    GMLX_KVARN_BITS=k6v5 style pair."""
    import os
    import re

    env = os.environ.get("GMLX_KVARN_BITS", "").strip().lower()
    if env:
        m = re.fullmatch(r"k(\d+)v(\d+)", env)
        if m:
            return int(m.group(1)), int(m.group(2))
        _log.warning("GMLX_KVARN_BITS=%r is not of the form k6v5; ignored", env)
    bits = int(kv_bits) if kv_bits is not None else 6
    return bits, bits


def kvarn_mtp_window_decline(caches) -> str | None:
    """Why kvarn declines this MTP target stack, or None. A mixed
    kvarn/rotating stack has never been validated through MTP rollback
    (no eligible SWA+MTP checkpoint exists to validate it on), so a
    sliding-window layer declines the whole stack. Serve, run and chat
    share the rule."""
    from gmlx.cache.compat import cache_types

    rotating = cache_types("RotatingKVCache") + cache_types("BatchRotatingKVCache")
    if any(isinstance(c, rotating) for c in caches):
        return "sliding-window cache stack cannot quantize"
    return None


def parse_tail_tokens(raw) -> int:
    """KV_TAIL_TOKENS text as the tail width; empty means the default.
    Malformed text raises ValueError (the resolver checks the value)."""
    text = "" if raw is None else str(raw).strip()
    if not text:
        return KVARN_DEFAULT_TAIL
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"KV_TAIL_TOKENS={text!r} is not an integer") from None


def kvarn_rotating_window(model, max_kv_size):
    """The rotating window kvarn must honor, or None. --max-kv-size only
    manufactures RotatingKVCache stacks for models without their own
    make_cache; models with one ignore the flag."""
    if max_kv_size is None or callable(getattr(model, "make_cache", None)):
        return None
    return int(max_kv_size)


def _quantize_head(x, bits, side):
    """Quantize rotated [B, H, T, D] groups into the flat slice-minor wire:
    codes [B, H, G, (D/128) * 512 * bits], axes [B, H, G, 3 * (D/128), 128]."""
    b, h, t, d = x.shape
    sl = d // GROUP
    if sl == 1:
        return kq.kvarn_quantize(x, bits, side)
    xs = x.reshape(b, h, t, sl, GROUP).transpose(0, 1, 3, 2, 4)
    c, a = kq.kvarn_quantize(xs, bits, side)
    g = c.shape[3]
    c = c.transpose(0, 1, 3, 2, 4).reshape(b, h, g, sl * 512 * bits)
    a = a.transpose(0, 1, 3, 2, 4, 5).reshape(b, h, g, 3 * sl, GROUP)
    return c, a


def _dequant_head(codes, axes, bits, side, d, dtype):
    """Invert _quantize_head's layout back to rotated [B, H, G * 128, D]."""
    b, h, g = codes.shape[:3]
    sl = d // GROUP
    if sl == 1:
        return kq.kvarn_dequant(codes, axes, bits, side, dtype=dtype)
    c = codes.reshape(b, h, g, sl, 512 * bits).transpose(0, 1, 3, 2, 4)
    a = axes.reshape(b, h, g, sl, 3, GROUP).transpose(0, 1, 3, 2, 4, 5)
    out = kq.kvarn_dequant(c, a, bits, side, dtype=dtype)
    return out.transpose(0, 1, 3, 2, 4).reshape(b, h, g * GROUP, d)


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


class _KVarNStorage(_base_cache()):
    """Region storage shared by the single-stream and batched caches:
    geometry validation, the fp16 stage and tail buffers, the record
    slabs, the append/seal walk and the attention-side accessors.
    Subclasses supply the write position ``_pos``, ``_alloc`` and the
    horizon policy."""

    kv_quant_scheme = "kvarn"
    kvarn_layout_version = 1
    gcap_step = 32
    tail_slack = 256

    def _init_geometry(self, k_bits, v_bits, tail_tokens, sink_tokens):
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
        self.n_sealed = 0
        self.tail_start = 0
        self.tail_end = 0
        self.codes_k = None
        self.axes_k = None
        self.codes_v = None
        self.axes_v = None
        self.stage_k = None
        self.stage_v = None
        self.tail_k = None
        self.tail_v = None

    @property
    def _pos(self):
        """Buffer write position: keys present in the region map."""
        raise NotImplementedError

    # -- derived watermarks -------------------------------------------------

    @property
    def sink_used(self):
        return min(self._pos, self.sink_cap)

    @property
    def live_len(self):
        return self._pos - self.sink_used - GROUP * self.n_sealed

    @property
    def tail_len(self):
        return self.tail_end - self.tail_start

    def _allocated(self):
        # Real stages are >= 128 wide; empty-state placeholders are 1.
        return self.stage_k is not None and self.stage_k.shape[-1] >= GROUP

    @property
    def head_dim(self):
        return self.stage_k.shape[-1] if self._allocated() else None

    # -- lifecycle ----------------------------------------------------------

    def _check_kv(self, keys, values):
        if (
            keys.ndim != 4
            or keys.shape[-1] not in HEAD_DIMS
            or values.shape[-1] != keys.shape[-1]
        ):
            raise ValueError(
                f"[kvarn] {type(self).__name__} requires head_dim in {HEAD_DIMS} "
                f"with matching K and V, got K {tuple(keys.shape)} "
                f"V {tuple(values.shape)}."
            )

    def _alloc_regions(self, b, h, d):
        sl = d // GROUP
        s_rows = self.sink_cap + GROUP
        self.stage_k = mx.zeros((b, h, s_rows, d), mx.float16)
        self.stage_v = mx.zeros((b, h, s_rows, d), mx.float16)
        t_rows = self.tail_cap + self.tail_slack if self.tail_cap else 1
        self.tail_k = mx.zeros((b, h, t_rows, d), mx.float16)
        self.tail_v = mx.zeros((b, h, t_rows, d), mx.float16)
        g = self._initial_gcap()
        self.codes_k = mx.zeros((b, h, g, sl * 512 * self.k_bits), mx.uint32)
        self.codes_v = mx.zeros((b, h, g, sl * 512 * self.v_bits), mx.uint32)
        self.axes_k = mx.zeros((b, h, g, 3 * sl, GROUP), mx.float16)
        self.axes_v = mx.zeros((b, h, g, 3 * sl, GROUP), mx.float16)

    def _initial_gcap(self):
        return self.gcap_step

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

    def _ensure_tail_rows(self):
        rows = self.tail_cap + self.tail_slack
        have = self.tail_k.shape[2]
        if have >= rows:
            return
        pad = self.tail_k.shape[:2] + (rows - have,) + self.tail_k.shape[3:]
        self.tail_k = mx.concatenate([self.tail_k, mx.zeros(pad, mx.float16)], 2)
        self.tail_v = mx.concatenate([self.tail_v, mx.zeros(pad, mx.float16)], 2)

    def _content(self, f):
        """Field ``f`` cut to what a restore needs: the sealed groups, the
        tail ring through tail_end, a valid horizon. Growth slack is not
        carried; the writers regrow it. The stage keeps its rows (the
        fused kernel reads sink_cap from its shape)."""
        a = getattr(self, f)
        if f.startswith(("codes", "axes")):
            return a[:, :, : max(1, self.n_sealed)]
        if f.startswith("tail"):
            return a[:, :, : max(1, self.tail_end)]
        if f.startswith("horizon") and not getattr(self, "horizon_valid", False):
            return a[:, :, :1]
        return a

    def _ingest(self, keys, values):
        """Write the tail, rotate, and append; the caller advances its
        own position counter."""
        if keys.dtype != mx.float16:
            keys = keys.astype(mx.float16)
        if values.dtype != mx.float16:
            values = values.astype(mx.float16)
        self._write_tail(keys, values)
        self._append_rotated(kq.kvarn_rotate(keys), kq.kvarn_rotate(values))
        return KVarNView(self, "k"), KVarNView(self, "v")

    def _write_tail(self, keys, values):
        if not self.tail_cap:
            return
        self._ensure_tail_rows()
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
        pos = self._pos
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
        """Quantize one or more complete rotated groups into records."""
        m = rk_block.shape[2] // GROUP
        g = self.n_sealed
        self._ensure_gcap(g + m)
        ck, ak = _quantize_head(rk_block, self.k_bits, "k")
        cv, av = _quantize_head(rv_block, self.v_bits, "v")
        self.codes_k[:, :, g : g + m] = ck
        self.axes_k[:, :, g : g + m] = ak
        self.codes_v[:, :, g : g + m] = cv
        self.axes_v[:, :, g : g + m] = av
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
                    _dequant_head(
                        getattr(self, f"codes_{side}")[:, :, : self.n_sealed],
                        getattr(self, f"axes_{side}")[:, :, : self.n_sealed],
                        self.k_bits if side == "k" else self.v_bits,
                        side,
                        self.head_dim,
                        dtype,
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


class KVarNKVCache(_KVarNStorage):
    # Tokens dropped from the record region by the rotating subclass. A
    # class attribute so from_state-restored instances (cls.__new__, no
    # __init__) resolve 0; all base arithmetic is identity at 0.
    evicted = 0

    def __init__(self, k_bits=6, v_bits=6, tail_tokens=1024, sink_tokens=GROUP):
        self._init_geometry(k_bits, v_bits, tail_tokens, sink_tokens)
        self.offset = 0
        self.horizon_valid = False
        self.horizon_k = None
        self.horizon_v = None

    @property
    def visible(self):
        """Keys present (offset counts every token ever appended)."""
        return self.offset - self.evicted

    _pos = visible

    def _alloc(self, h, d):
        self._alloc_regions(1, h, d)
        self.horizon_k = mx.zeros((1, h, GROUP, d), mx.float16)
        self.horizon_v = mx.zeros((1, h, GROUP, d), mx.float16)

    def update_and_fetch(self, keys, values):
        self._check_kv(keys, values)
        if keys.shape[0] != 1:
            raise ValueError("[kvarn] KVarNKVCache is single-stream (B=1).")
        if not self._allocated() or self.stage_k.shape[1] != keys.shape[1]:
            if self.offset:
                raise RuntimeError("[kvarn] cache head count changed mid-stream.")
            self._alloc(keys.shape[1], keys.shape[-1])
        views = self._ingest(keys, values)
        self.offset += keys.shape[2]
        return views

    def _seal(self, rk_block, rv_block):
        """Seal, then refresh the horizon with the last group."""
        super()._seal(rk_block, rv_block)
        # copy, not view: on the bulk path rk_block slices a per-call
        # slab, and a view would pin the whole slab in the horizon
        self.horizon_k = mx.contiguous(rk_block[:, :, -GROUP:])
        self.horizon_v = mx.contiguous(rv_block[:, :, -GROUP:])
        self.horizon_valid = True

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.visible, **kwargs)

    # -- trim ---------------------------------------------------------------

    def is_trimmable(self):
        return True

    def _trim_plan(self, n):
        """How to serve a trim of n tokens, or None when the frontier
        falls in evicted history. Exact plans first; the records plan
        dequantizes the frontier group's rows and is the fallback."""
        new_off = self.offset - n
        if new_off <= self.sink_cap:
            return ("sink",)
        body = new_off - self.sink_cap - self.evicted
        if body < 0:
            # Frontier inside evicted history: nothing to reopen from.
            return None
        g = body // GROUP
        live = body % GROUP
        if g == self.n_sealed:
            return ("live",)
        if g == self.n_sealed - 1 and self.horizon_valid:
            return ("horizon", live)
        cover0 = self.offset - self.tail_len
        if self.sink_cap + self.evicted + g * GROUP >= cover0:
            return ("tail", g, live, cover0)
        return ("records", g, live)

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
            self.evicted = 0
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
                a = self.sink_cap + self.evicted + g * GROUP - cover0
                tk = self.tail_k[:, :, self.tail_start + a : self.tail_start + a + live]
                tv = self.tail_v[:, :, self.tail_start + a : self.tail_start + a + live]
                s0 = self.sink_cap
                self.stage_k[:, :, s0 : s0 + live] = kq.kvarn_rotate(tk)
                self.stage_v[:, :, s0 : s0 + live] = kq.kvarn_rotate(tv)
            self.n_sealed = g
            self.horizon_valid = False
        elif kind == "records":
            g, live = plan[1:]
            if live:
                d = self.head_dim
                s0 = self.sink_cap
                for side, codes, axes, bits, stage in (
                    ("k", self.codes_k, self.axes_k, self.k_bits, self.stage_k),
                    ("v", self.codes_v, self.axes_v, self.v_bits, self.stage_v),
                ):
                    rows = _dequant_head(codes[:, :, g : g + 1], axes[:, :, g : g + 1],
                                         bits, side, d, stage.dtype)
                    stage[:, :, s0 : s0 + live] = rows[:, :, :live]
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
        """Content only (see _content): a clone or disk record carries no
        growth slack, and a restored cache regrows on its first write."""
        if not self._allocated():
            # Fixed-arity placeholder so an empty cache still round-trips
            # (safetensors rejects zero-size arrays). meta_state's
            # allocated flag tells the setter to discard it.
            z16 = mx.zeros((1, 1, 1, 1), mx.float16)
            z32 = mx.zeros((1, 1, 1, 1), mx.uint32)
            return (z32, z16, z32, z16, z16, z16, z16, z16, z16, z16)
        return tuple(self._content(f) for f in self._STATE_FIELDS)

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
        if len(v) != 11:
            raise ValueError(_META_ARITY_MSG)
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
        return self.visible

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
            rk[:, :, self.visible - t :] = tk
            rv[:, :, self.visible - t :] = tv
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


class KVarNRotatingKVCache(KVarNKVCache):
    """Bounded-window kvarn cache for --max-kv-size (CLI run/chat).

    At most ``g_max = (max_size - sink_cap) // 128`` sealed groups stay
    resident (max_size rounds down to whole groups); older groups are
    dropped by compacting the record slabs at update entry, before the
    append, so the first row of an incoming chunk still sees at least
    max_size context and visibility never changes between mask creation
    and attention within a call. Keys hold post-RoPE absolute positions,
    so dropping leading groups is numerically exact; ``offset`` stays the
    absolute token count (RoPE, chat checkpoints) while ``visible`` feeds
    the kernel.

    Divergence from mlx-lm's RotatingKVCache: the window is a memory
    bound with a context floor, not a per-query attention span, so only
    string masks are served (see make_mask) and no keep parameter exists
    (sink_cap plays that role). Trims that would reopen evicted history
    return 0 and the caller rebuilds.
    """

    evict_slack = 4  # groups of hysteresis between compactions

    def __init__(
        self, max_size, k_bits=6, v_bits=6, tail_tokens=1024, sink_tokens=GROUP
    ):
        super().__init__(
            k_bits=k_bits,
            v_bits=v_bits,
            tail_tokens=tail_tokens,
            sink_tokens=sink_tokens,
        )
        floor = self.sink_cap + max(self.tail_cap, GROUP) + GROUP
        if int(max_size) < floor:
            raise ValueError(
                f"[kvarn] max_size {max_size} is below the kvarn window "
                f"floor {floor} (sink {self.sink_cap} + tail "
                f"{max(self.tail_cap, GROUP)} + {GROUP})."
            )
        self.max_size = int(max_size)
        self.evicted = 0

    @property
    def g_max(self):
        return (self.max_size - self.sink_cap) // GROUP

    def _initial_gcap(self):
        # +1: decode steady state peaks one seal past the compaction
        # trigger (entry leaves g_max + slack, the append seals one more).
        return min(self.gcap_step, self.g_max + self.evict_slack + 1)

    def _ensure_gcap(self, groups):
        # Grow to exactly what the call needs: the window re-compacts to
        # g_max, so step-chunk growth would durably overshoot small
        # windows. Wide prefill chunks push past the steady capacity
        # transiently; the next compaction restores it.
        gcap = self.codes_k.shape[2]
        if groups <= gcap:
            return
        add = groups - gcap

        def grow(x):
            pad = mx.zeros(x.shape[:2] + (add,) + x.shape[3:], x.dtype)
            return mx.concatenate([x, pad], axis=2)

        self.codes_k, self.codes_v = grow(self.codes_k), grow(self.codes_v)
        self.axes_k, self.axes_v = grow(self.axes_k), grow(self.axes_v)

    def _compact(self):
        d = self.n_sealed - self.g_max
        if d <= self.evict_slack:
            return
        # Steady capacity: g_max resident plus the slack and the one seal
        # an append adds past the trigger. A wide prefill's overshoot
        # goes here.
        cap = self.g_max + self.evict_slack + 1
        fields = ("codes_k", "axes_k", "codes_v", "axes_v")
        slabs = []
        for f in fields:
            x = getattr(self, f)
            pad = mx.zeros(x.shape[:2] + (cap - self.g_max,) + x.shape[3:], x.dtype)
            slabs.append(mx.concatenate([x[:, :, d : self.n_sealed], pad], axis=2))
        # One eval: the compaction's transient is these four copies, not
        # the whole pending forward graph four times over.
        mx.eval(*slabs)
        for f, slab in zip(fields, slabs):
            setattr(self, f, slab)
        self.n_sealed = self.g_max
        self.evicted += d * GROUP

    def update_and_fetch(self, keys, values):
        self._compact()
        return super().update_and_fetch(keys, values)

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        if return_array or window_size is not None:
            # No correct array is producible: masks are built before layer
            # 0's update, so an array would size to pre-compaction width.
            raise ValueError(
                "[kvarn] rotating kvarn serves string masks only; "
                "array or windowed masks cannot span an eviction."
            )
        return "causal" if N > 1 else None

    @property
    def meta_state(self):
        return KVarNKVCache.meta_state.fget(self) + (
            str(self.max_size),
            str(self.evicted),
        )

    @meta_state.setter
    def meta_state(self, v):
        if len(v) != 13:
            raise ValueError(_META_ARITY_MSG)
        KVarNKVCache.meta_state.fset(self, v[:11])
        self.max_size = int(v[11])
        self.evicted = int(v[12])

    @classmethod
    def from_cache(cls, cache, k_bits=6, v_bits=6, tail_tokens=1024, sink_tokens=GROUP):
        """Convert a stock RotatingKVCache (window taken from the source).
        Pre-wrap sources only: a wrapped ring buffer is no longer in
        temporal order."""
        max_size = int(getattr(cache, "max_size", 0) or 0)
        off = int(getattr(cache, "offset", 0) or 0)
        idx = int(getattr(cache, "_idx", off) or 0)
        if max(off, idx) >= max_size:
            raise ValueError(
                "[kvarn] cannot convert a wrapped rotating cache (its "
                "buffer is no longer in temporal order)."
            )
        out = cls(
            max_size,
            k_bits=k_bits,
            v_bits=v_bits,
            tail_tokens=tail_tokens,
            sink_tokens=sink_tokens,
        )
        if off:
            keys, values = cache.state
            out.update_and_fetch(keys[:, :, :off], values[:, :, :off])
        return out


class BatchKVarNKVCache(_KVarNStorage):
    """Batched KVarN KV cache: the region layout with a leading batch axis
    and a geometry per row. Row b keeps its own physical ``starts[b]`` (the
    first real key, fixed when the row is ingested: 0 for a row that arrived
    alone, its pad for a row from a left-padded batch prefill) and
    ``ends[b]`` (keys present). Attention is permutation-invariant over
    keys, so rows never realign: a speculative rollback lowers each row's
    end by its own rejected count and the next append writes each row at
    its own end. Records stay put and group boundaries stay slot-indexed
    and shared; a row seals its live group when its own end crosses a
    128-token boundary. Pad rows hold whatever K/V the model produced for
    pad positions and are excluded at attention time by the row's start.

    The scalar watermarks (``_pos``, ``n_sealed``, ``sink_used``,
    ``live_len``, ``tail_end``) describe the longest row, which is what
    sizes the buffers; everything per row reads the lists. Starts and ends
    are Python ints, and the int32 mirrors the kernels read are rebuilt
    after a mutation, never on read.

    The rest of the stack (fp16 BatchKVCache layers, recurrent state) keeps
    upstream's right-justified geometry and the shared mask must describe
    it, so this class also keeps a shadow of it: ``_idx`` and
    ``left_padding`` follow BatchKVCache's arithmetic exactly (append adds,
    trim subtracts, finalize moves right padding into left_padding, extend
    right-aligns to the larger _idx, filter compacts by the minimum pad),
    with ``_idx - left_padding[b] == ends[b] - starts[b]`` on every row.
    ``make_mask`` builds from the shadow; the kvarn route ignores the
    mask's geometry and reads the physical starts and ends off the cache.

    Rollback: ``trim(n)`` applies the single-stream plan per row (live,
    horizon, tail, records), and ``prepare(right_padding=...)`` followed by
    ``finalize()`` trims each row by its own count. The fp16 tail is one
    ring shared by the rows with a window per row; ring rows [0, tail_ends
    [b]) are that row's most recent tokens, so its coverage is min(
    tail_ends[b], tail_cap) and never falls short of its content up to
    tail_cap, which the batched route relies on (its body leg stops
    tail_cap short of every row's end)."""

    ragged_trim = True
    batch_state_version = 2

    _ARRAY_FIELDS = (
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

    def __init__(
        self, left_padding, k_bits=6, v_bits=6, tail_tokens=1024, sink_tokens=GROUP
    ):
        self._init_geometry(k_bits, v_bits, tail_tokens, sink_tokens)
        self.horizon_k = None
        self.horizon_v = None
        self._idx = 0
        self._right_padding = None
        self._reset_rows(_int_list(left_padding))

    def _reset_rows(self, pads):
        """The empty state for these pads: starts at the pads, nothing
        present, no tail coverage, no horizon."""
        b = len(pads)
        self.starts = list(pads)
        self.ends = [0] * b
        self.tail_ends = [0] * b
        self.horizon_valid = [False] * b
        self._set_pads(pads)

    def _set_pads(self, pads):
        self._pads = [int(p) for p in pads]
        self.left_padding = mx.array(self._pads, dtype=mx.int32)
        self._touch()

    # -- geometry -----------------------------------------------------------

    @property
    def offset(self):
        """Shadow visible length per row (the BatchKVCache reading)."""
        return self._idx - self.left_padding

    @property
    def _pos(self):
        return max(self.ends) if self.ends else 0

    def n_sealed_row(self, b):
        return max(0, self.ends[b] - self.sink_cap) // GROUP

    def live_len_row(self, b):
        e = self.ends[b]
        return e - min(e, self.sink_cap) - GROUP * self.n_sealed_row(b)

    def tail_start_row(self, b):
        return max(0, self.tail_ends[b] - self.tail_cap)

    def tail_len_row(self, b):
        return min(self.tail_ends[b], self.tail_cap)

    # _init_geometry writes the single-stream scalars once before any row
    # exists; on the batch class they are derived from the lists.
    def _derived_scalar(name, fn):
        def get(self):
            return fn(self)

        def set_(self, v):
            if v:
                raise AttributeError(
                    f"[kvarn] {name} is derived per row on the batch cache"
                )

        return property(get, set_)

    n_sealed = _derived_scalar(
        "n_sealed",
        lambda self: max((self.n_sealed_row(b) for b in range(len(self.ends))),
                         default=0),
    )
    tail_end = _derived_scalar(
        "tail_end", lambda self: max(self.tail_ends, default=0)
    )
    tail_start = _derived_scalar(
        "tail_start",
        lambda self: min((self.tail_start_row(b) for b in range(len(self.ends))),
                         default=0),
    )
    del _derived_scalar

    @property
    def tail_len(self):
        """Tail coverage of the best-covered row."""
        return max((self.tail_len_row(b) for b in range(len(self.ends))),
                   default=0)

    def _touch(self):
        self._mirrors = {}

    def _mirror(self, name, values):
        m = self.__dict__.get("_mirrors")
        if m is None:
            m = self._mirrors = {}
        a = m.get(name)
        if a is None:
            a = m[name] = mx.array(list(values), dtype=mx.int32)
        return a

    @property
    def starts_mx(self):
        return self._mirror("starts", self.starts)

    @property
    def ends_mx(self):
        return self._mirror("ends", self.ends)

    @property
    def tail_ends_mx(self):
        return self._mirror("tail_ends", self.tail_ends)

    @property
    def tail_leg_starts_mx(self):
        """Ring row where each row's tail leg begins: the window's last
        min(ends - starts, tail_cap) rows, so a row shorter than the tail
        attends its whole content there."""
        return self._mirror(
            "tail_leg_starts",
            [te - min(max(e - s, 0), self.tail_cap)
             for s, e, te in zip(self.starts, self.ends, self.tail_ends)],
        )

    # -- lifecycle ----------------------------------------------------------

    def _alloc(self, b, h, d):
        self._alloc_regions(b, h, d)
        self.horizon_k = mx.zeros((b, h, GROUP, d), mx.float16)
        self.horizon_v = mx.zeros((b, h, GROUP, d), mx.float16)

    def _ensure_horizon(self):
        """A full horizon buffer (restores and lone-row merges carry a
        one-row placeholder while no row's horizon is valid)."""
        if self.horizon_k.shape[2] != GROUP:
            b, h, _, d = self.horizon_k.shape
            self.horizon_k = mx.zeros((b, h, GROUP, d), mx.float16)
            self.horizon_v = mx.zeros((b, h, GROUP, d), mx.float16)

    def update_and_fetch(self, keys, values):
        self._check_kv(keys, values)
        if keys.shape[0] != len(self.ends):
            raise ValueError(
                f"[kvarn] batch size {keys.shape[0]} does not match "
                f"left_padding ({len(self.ends)} rows)."
            )
        if not self._allocated() or self.stage_k.shape[1] != keys.shape[1]:
            if self._pos:
                raise RuntimeError("[kvarn] cache head count changed mid-stream.")
            self._alloc(keys.shape[0], keys.shape[1], keys.shape[-1])
        views = self._ingest(keys, values)
        t = keys.shape[2]
        self.ends = [e + t for e in self.ends]
        self._idx += t
        self._touch()
        return views

    # -- append -------------------------------------------------------------

    def _write_tail(self, keys, values):
        if not self.tail_cap:
            return
        self._ensure_tail_rows()
        n = keys.shape[2]
        b = len(self.ends)
        rows = self.tail_k.shape[2]
        if n >= self.tail_cap:
            # A slab restarts every window with as much history as the
            # ring holds: the slack past tail_cap is what a rollback
            # rewinds into.
            w = min(n, rows)
            self.tail_k[:, :, :w] = keys[:, :, -w:]
            self.tail_v[:, :, :w] = values[:, :, -w:]
            self.tail_ends = [w] * b
            return
        if len(set(self.tail_ends)) == 1:
            e = self.tail_ends[0]
            if e + n > rows:
                # Keep every row the ring holds: a rollback rewinds the
                # window into that history.
                keep = min(e, rows - n)
                src = slice(e - keep, e)
                self.tail_k[:, :, :keep] = self.tail_k[:, :, src]
                self.tail_v[:, :, :keep] = self.tail_v[:, :, src]
                e = keep
            self.tail_k[:, :, e : e + n] = keys
            self.tail_v[:, :, e : e + n] = values
            self.tail_ends = [e + n] * b
            return
        for i in range(b):
            e = self.tail_ends[i]
            if e + n > rows:
                keep = min(e, rows - n)
                src = slice(e - keep, e)
                self.tail_k[i : i + 1, :, :keep] = self.tail_k[i : i + 1, :, src]
                self.tail_v[i : i + 1, :, :keep] = self.tail_v[i : i + 1, :, src]
                self.tail_ends[i] = keep
        idx = mx.array(self.tail_ends, dtype=mx.int32)[:, None] + mx.arange(
            n, dtype=mx.int32
        )[None, :]
        idx = idx[:, None, :, None]
        self.tail_k = mx.put_along_axis(self.tail_k, idx, keys, axis=2)
        self.tail_v = mx.put_along_axis(self.tail_v, idx, values, axis=2)
        self.tail_ends = [e + n for e in self.tail_ends]

    def _append_rotated(self, rk, rv):
        pos = list(self.ends)
        if len(set(pos)) == 1:
            self._append_uniform(rk, rv, pos[0])
            return
        n = rk.shape[2]
        for c0 in range(0, n, GROUP):
            c1 = min(n, c0 + GROUP)
            self._append_ragged(rk[:, :, c0:c1], rv[:, :, c0:c1], pos)
            pos = [p + (c1 - c0) for p in pos]

    def _append_uniform(self, rk, rv, pos):
        """Every row at ``pos``: the single-stream walk over shared slots."""
        n = rk.shape[2]
        sc = self.sink_cap
        a = 0
        if pos < sc:
            t = min(n, sc - pos)
            self.stage_k[:, :, pos : pos + t] = rk[:, :, :t]
            self.stage_v[:, :, pos : pos + t] = rv[:, :, :t]
            a = t
        g = max(0, pos - sc) // GROUP
        while a < n:
            live = (pos + a) - sc - GROUP * g
            if live > 0:
                t = min(GROUP - live, n - a)
                s = sc + live
                self.stage_k[:, :, s : s + t] = rk[:, :, a : a + t]
                self.stage_v[:, :, s : s + t] = rv[:, :, a : a + t]
                a += t
                if live + t == GROUP:
                    self._seal_all(
                        g,
                        self.stage_k[:, :, sc : sc + GROUP],
                        self.stage_v[:, :, sc : sc + GROUP],
                    )
                    g += 1
                continue
            m = (n - a) // GROUP
            if m:
                self._seal_all(g, rk[:, :, a : a + m * GROUP], rv[:, :, a : a + m * GROUP])
                g += m
                a += m * GROUP
            rem = n - a
            if rem:
                self.stage_k[:, :, sc : sc + rem] = rk[:, :, a:]
                self.stage_v[:, :, sc : sc + rem] = rv[:, :, a:]
                a = n

    def _append_ragged(self, rk, rv, pos):
        """At most one group of tokens with the rows at their own ends.
        Rows whose live group fills inside the chunk seal first (the old
        stage rows plus the new tokens up to the boundary, one quantize
        call for all of them), then every token lands in its slot: below
        the sink cap directly, else at the live slot of its position."""
        n = rk.shape[2]
        sc = self.sink_cap
        crossing = []
        for b, p in enumerate(pos):
            body0 = p - sc
            if body0 < 0:
                continue
            live = body0 % GROUP
            if live + n >= GROUP:
                crossing.append((b, GROUP - live - 1, body0 // GROUP, live))
        if crossing:
            blk_k, blk_v = [], []
            for b, fill_t, _, live in crossing:
                pk = [self.stage_k[b : b + 1, :, sc : sc + live]] if live else []
                pv = [self.stage_v[b : b + 1, :, sc : sc + live]] if live else []
                pk.append(rk[b : b + 1, :, : fill_t + 1])
                pv.append(rv[b : b + 1, :, : fill_t + 1])
                blk_k.append(mx.concatenate(pk, axis=2) if len(pk) > 1 else pk[0])
                blk_v.append(mx.concatenate(pv, axis=2) if len(pv) > 1 else pv[0])
            self._seal_rows(
                [c[0] for c in crossing],
                [c[2] for c in crossing],
                mx.concatenate(blk_k, axis=0) if len(blk_k) > 1 else blk_k[0],
                mx.concatenate(blk_v, axis=0) if len(blk_v) > 1 else blk_v[0],
            )
        p = mx.array(pos, dtype=mx.int32)[:, None] + mx.arange(n, dtype=mx.int32)[None, :]
        slot = mx.where(p < sc, p, sc + (p - sc) % GROUP)[:, None, :, None]
        self.stage_k = mx.put_along_axis(
            self.stage_k, slot, rk.astype(mx.float16), axis=2
        )
        self.stage_v = mx.put_along_axis(
            self.stage_v, slot, rv.astype(mx.float16), axis=2
        )

    def _seal(self, rk_block, rv_block):
        raise NotImplementedError("[kvarn] the batch cache seals per row")

    def _seal_all(self, g, rk_block, rv_block):
        """Seal complete groups at group index ``g`` on every row."""
        m = rk_block.shape[2] // GROUP
        self._ensure_gcap(g + m)
        ck, ak = _quantize_head(rk_block, self.k_bits, "k")
        cv, av = _quantize_head(rv_block, self.v_bits, "v")
        self.codes_k[:, :, g : g + m] = ck
        self.axes_k[:, :, g : g + m] = ak
        self.codes_v[:, :, g : g + m] = cv
        self.axes_v[:, :, g : g + m] = av
        self.horizon_k = mx.contiguous(rk_block[:, :, -GROUP:])
        self.horizon_v = mx.contiguous(rv_block[:, :, -GROUP:])
        self.horizon_valid = [True] * len(self.ends)

    def _seal_rows(self, rows, groups, blk_k, blk_v):
        """Seal one group per listed row at that row's group index."""
        self._ensure_gcap(max(groups) + 1)
        self._ensure_horizon()
        ck, ak = _quantize_head(blk_k, self.k_bits, "k")
        cv, av = _quantize_head(blk_v, self.v_bits, "v")
        r = mx.array(rows, dtype=mx.int32)
        g = mx.array(groups, dtype=mx.int32)
        self.codes_k[r, :, g] = ck[:, :, 0]
        self.axes_k[r, :, g] = ak[:, :, 0]
        self.codes_v[r, :, g] = cv[:, :, 0]
        self.axes_v[r, :, g] = av[:, :, 0]
        self.horizon_k[r] = blk_k
        self.horizon_v[r] = blk_v
        for b in rows:
            self.horizon_valid[b] = True

    # -- attention-side accessors -------------------------------------------

    def _content(self, f):
        a = getattr(self, f)
        if f.startswith(("codes", "axes")):
            return a[:, :, : max(1, self.n_sealed)]
        if f.startswith("tail"):
            return a[:, :, : max(1, self.tail_end)]
        if f.startswith("horizon") and not any(self.horizon_valid):
            return a[:, :, :1]
        return a

    def _content_row(self, f, b):
        """Field ``f`` cut to row ``b``'s needs (the single-stream cut)."""
        a = getattr(self, f)[b : b + 1]
        if f.startswith(("codes", "axes")):
            return a[:, :, : max(1, self.n_sealed_row(b))]
        if f.startswith("tail"):
            return a[:, :, : max(1, self.tail_ends[b])]
        if f.startswith("horizon") and not self.horizon_valid[b]:
            return a[:, :, :1]
        return a

    def materialize(self, dtype=mx.float16):
        """Rotated-domain K/V [B, H, max(ends), D]: every row's sink rows,
        its records and its live rows at their physical positions. Past a
        row's end the columns hold whatever the buffers hold; callers mask
        by the row's start and end."""
        n = self._pos
        sc = self.sink_cap
        sink = min(n, sc)
        b = len(self.ends)
        d = self.head_dim
        h = self.stage_k.shape[1]
        gmax = self.n_sealed
        per_row = [self.n_sealed_row(i) for i in range(b)]
        ragged = gmax and any(g != gmax for g in per_row)
        live = n - sink - GROUP * gmax
        outs = []
        for side in ("k", "v"):
            stage = getattr(self, f"stage_{side}")
            parts = [stage[:, :, :sink].astype(dtype)]
            if gmax:
                rec = _dequant_head(
                    getattr(self, f"codes_{side}")[:, :, :gmax],
                    getattr(self, f"axes_{side}")[:, :, :gmax],
                    self.k_bits if side == "k" else self.v_bits,
                    side,
                    d,
                    dtype,
                )
                if ragged:
                    # A row with fewer sealed groups reads its live stage
                    # rows where the longer rows read records.
                    rec5 = rec.reshape(b, h, gmax, GROUP, d)
                    live5 = stage[:, :, sc : sc + GROUP].astype(dtype)[:, :, None]
                    gidx = mx.arange(gmax)[None, None, :, None, None]
                    have = mx.array(per_row)[:, None, None, None, None]
                    rec = mx.where(gidx < have, rec5, live5).reshape(
                        b, h, gmax * GROUP, d
                    )
                parts.append(rec)
            if live:
                parts.append(stage[:, :, sc : sc + live].astype(dtype))
            outs.append(mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0])
        return outs[0], outs[1]

    def _materialize_row(self, b, dtype=mx.float16):
        """Row ``b`` alone, exact: [1, H, ends[b], D]."""
        e = self.ends[b]
        sc = self.sink_cap
        g = self.n_sealed_row(b)
        live = self.live_len_row(b)
        outs = []
        for side in ("k", "v"):
            stage = getattr(self, f"stage_{side}")[b : b + 1]
            parts = [stage[:, :, : min(e, sc)].astype(dtype)]
            if g:
                parts.append(
                    _dequant_head(
                        getattr(self, f"codes_{side}")[b : b + 1, :, :g],
                        getattr(self, f"axes_{side}")[b : b + 1, :, :g],
                        self.k_bits if side == "k" else self.v_bits,
                        side,
                        self.head_dim,
                        dtype,
                    )
                )
            if live:
                parts.append(stage[:, :, sc : sc + live].astype(dtype))
            outs.append(mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0])
        return outs[0], outs[1]

    def tail_slices(self, n_tokens):
        """Original-domain fp16 rows for the last n_tokens tokens of every
        row (each row's own window)."""
        if any(n_tokens > self.tail_len_row(b) for b in range(len(self.ends))):
            raise ValueError("[kvarn] tail request beyond coverage.")
        if len(set(self.tail_ends)) == 1:
            e = self.tail_ends[0]
            s = slice(e - n_tokens, e)
            return self.tail_k[:, :, s], self.tail_v[:, :, s]
        idx = (self.tail_ends_mx - n_tokens)[:, None] + mx.arange(
            n_tokens, dtype=mx.int32
        )[None, :]
        idx = idx[:, None, :, None]
        return (
            mx.take_along_axis(self.tail_k, idx, axis=2),
            mx.take_along_axis(self.tail_v, idx, axis=2),
        )

    def _raw_row(self, b):
        """Row ``b``'s recovered original-domain K/V [1, H, ends[b], D]:
        the body un-rotated from its regions (WHT is self-inverse; one fp16
        rounding), its tail window verbatim."""
        mk, mv = self._materialize_row(b)
        rk = kq.kvarn_rotate(mk)
        rv = kq.kvarn_rotate(mv)
        t = self.tail_len_row(b)
        if t:
            e = self.ends[b]
            te = self.tail_ends[b]
            rk[:, :, e - t :] = self.tail_k[b : b + 1, :, te - t : te]
            rv[:, :, e - t :] = self.tail_v[b : b + 1, :, te - t : te]
        return rk, rv

    def _raw_rows(self):
        """Every row's recovered K/V at its physical positions, zero past
        its end: [B, H, max(ends), D]. The KVarNKVCache._raw_single twin."""
        n = self._pos
        outs_k, outs_v = [], []
        for b in range(len(self.ends)):
            rk, rv = self._raw_row(b)
            pad = n - rk.shape[2]
            if pad:
                z = mx.zeros(rk.shape[:2] + (pad,) + rk.shape[3:], rk.dtype)
                rk = mx.concatenate([rk, z], axis=2)
                rv = mx.concatenate([rv, z], axis=2)
            outs_k.append(rk)
            outs_v.append(rv)
        return mx.concatenate(outs_k, axis=0), mx.concatenate(outs_v, axis=0)

    # -- rollback -----------------------------------------------------------

    def is_trimmable(self):
        return True

    def _can_trim(self, n):
        """A uniform trim of n is served whenever every row holds n keys
        past its start: nothing is ever evicted, so the records plan
        always exists."""
        n = min(int(n), self._idx)
        if n <= 0:
            return True
        return bool(self.ends) and n <= min(
            e - s for s, e in zip(self.starts, self.ends)
        )

    def trim(self, n):
        n = min(int(n), self._idx)
        if n <= 0 or not self._can_trim(n):
            return 0
        for b in range(len(self.ends)):
            self._trim_row(b, n)
        self._idx -= n
        self._touch()
        return n

    def _trim_row(self, b, n):
        """The single-stream trim plan on one row: exact from the live
        stage, the horizon or the tail ring (its whole history, slack
        included), else the frontier group's records dequantize into the
        live stage."""
        e = self.ends[b]
        new_end = e - n
        sc = self.sink_cap
        if new_end > sc:
            body = new_end - sc
            g = body // GROUP
            live = body % GROUP
            have = self.n_sealed_row(b)
            if g == have:
                pass
            elif g == have - 1 and self.horizon_valid[b]:
                if live:
                    self.stage_k[b : b + 1, :, sc : sc + live] = self.horizon_k[
                        b : b + 1, :, :live
                    ]
                    self.stage_v[b : b + 1, :, sc : sc + live] = self.horizon_v[
                        b : b + 1, :, :live
                    ]
                self.horizon_valid[b] = False
            else:
                # ring row 0 holds position cover0
                cover0 = e - self.tail_ends[b]
                if sc + g * GROUP >= cover0:
                    if live:
                        a = sc + g * GROUP - cover0
                        tk = self.tail_k[b : b + 1, :, a : a + live]
                        tv = self.tail_v[b : b + 1, :, a : a + live]
                        self.stage_k[b : b + 1, :, sc : sc + live] = kq.kvarn_rotate(tk)
                        self.stage_v[b : b + 1, :, sc : sc + live] = kq.kvarn_rotate(tv)
                elif live:
                    d = self.head_dim
                    for side, codes, axes, bits, stage in (
                        ("k", self.codes_k, self.axes_k, self.k_bits, self.stage_k),
                        ("v", self.codes_v, self.axes_v, self.v_bits, self.stage_v),
                    ):
                        rows = _dequant_head(
                            codes[b : b + 1, :, g : g + 1],
                            axes[b : b + 1, :, g : g + 1],
                            bits,
                            side,
                            d,
                            stage.dtype,
                        )
                        stage[b : b + 1, :, sc : sc + live] = rows[:, :, :live]
                self.horizon_valid[b] = False
        else:
            self.horizon_valid[b] = False
        self.tail_ends[b] = max(0, self.tail_ends[b] - n)
        self.ends[b] = new_end
        self._top_up_tail(b)

    def _row_span(self, b, p0, p1):
        """Row ``b``'s rotated-domain K/V for positions [p0, p1): the sink
        stage, only the record groups the span touches, and the live
        stage. Bounded by the span, not the row."""
        sc = self.sink_cap
        have = self.n_sealed_row(b)
        rec_end = sc + have * GROUP
        d = self.head_dim
        outs = []
        for side in ("k", "v"):
            stage = getattr(self, f"stage_{side}")[b : b + 1]
            parts = []
            a0 = p0
            if a0 < sc:
                a1 = min(p1, sc)
                parts.append(stage[:, :, a0:a1])
                a0 = a1
            if a0 < p1 and a0 < rec_end:
                a1 = min(p1, rec_end)
                g0 = (a0 - sc) // GROUP
                g1 = (a1 - sc + GROUP - 1) // GROUP
                rows = _dequant_head(
                    getattr(self, f"codes_{side}")[b : b + 1, :, g0:g1],
                    getattr(self, f"axes_{side}")[b : b + 1, :, g0:g1],
                    self.k_bits if side == "k" else self.v_bits,
                    side,
                    d,
                    mx.float16,
                )
                off = a0 - (sc + g0 * GROUP)
                parts.append(rows[:, :, off : off + (a1 - a0)])
                a0 = a1
            if a0 < p1:
                parts.append(stage[:, :, sc + (a0 - rec_end) : sc + (p1 - rec_end)])
            outs.append(mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0])
        return outs[0], outs[1]

    def _top_up_tail(self, b):
        """Restore row ``b``'s tail coverage to min(length, tail_cap) rows
        when a trim rewound its window past the ring's history (a trim
        wider than the slack, or a single-stream row adopted right after
        its own normalization; never a verify round). The missing
        positions come from the body once, un-rotated, and go in front of
        the window, so the tail leg reads for them what the body leg
        would have."""
        if not self.tail_cap:
            return
        s, e, te = self.starts[b], self.ends[b], self.tail_ends[b]
        need = min(max(e - s, 0), self.tail_cap) - te
        if need <= 0:
            return
        if te + need > self.tail_k.shape[2]:
            self._ensure_tail_rows()
        rk, rv = self._row_span(b, e - te - need, e - te)
        if te:
            self.tail_k[b : b + 1, :, need : need + te] = self.tail_k[b : b + 1, :, :te]
            self.tail_v[b : b + 1, :, need : need + te] = self.tail_v[b : b + 1, :, :te]
        self.tail_k[b : b + 1, :, :need] = kq.kvarn_rotate(rk)
        self.tail_v[b : b + 1, :, :need] = kq.kvarn_rotate(rv)
        self.tail_ends[b] = te + need
        self._touch()

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        del lengths
        if left_padding is not None:
            if self._idx:
                raise ValueError(
                    "[kvarn] left padding can only be added to an empty cache."
                )
            add = _int_list(left_padding)
            self.starts = [s + a for s, a in zip(self.starts, add, strict=True)]
            self._set_pads([p + a for p, a in zip(self._pads, add, strict=True)])
        if right_padding is not None:
            self._right_padding = _int_list(right_padding)

    def finalize(self):
        """Apply the right padding as per-row trims (the ragged step of a
        speculative rollback) and move it into the shadow left padding, as
        BatchKVCache.finalize does with its roll."""
        rp = self._right_padding
        self._right_padding = None
        if rp is None or not any(rp):
            return
        if len(rp) != len(self.ends):
            raise ValueError(
                f"[kvarn] right padding has {len(rp)} entries for "
                f"{len(self.ends)} rows."
            )
        for b, n in enumerate(rp):
            if n < 0 or n > self.ends[b] - self.starts[b]:
                raise RuntimeError(
                    f"[kvarn] right padding {n} exceeds row {b}'s "
                    f"{self.ends[b] - self.starts[b]} keys."
                )
        for b, n in enumerate(rp):
            if n:
                self._trim_row(b, n)
        self._set_pads([p + n for p, n in zip(self._pads, rp, strict=True)])

    # -- batch ops ----------------------------------------------------------

    def filter(self, batch_indices):
        idx = _int_list(batch_indices)
        if not idx:
            # No survivors: the empty state at watermark 0, so the rows
            # extend() adopts next carry no shadow padding (the spec
            # loop's all-rows-finished adoption; a one-row batch decodes
            # at the watermark's position).
            for f in self._ARRAY_FIELDS:
                setattr(self, f, None)
            self._idx = 0
            self._right_padding = None
            self._reset_rows([])
            return
        sel = mx.array(idx, dtype=mx.int32)
        for f in self._ARRAY_FIELDS:
            a = getattr(self, f)
            if a is not None:
                setattr(self, f, a[sel])
        self.starts = [self.starts[i] for i in idx]
        self.ends = [self.ends[i] for i in idx]
        self.tail_ends = [self.tail_ends[i] for i in idx]
        self.horizon_valid = [self.horizon_valid[i] for i in idx]
        if self._right_padding is not None:
            self._right_padding = [self._right_padding[i] for i in idx]
        pads = [self._pads[i] for i in idx]
        # The shadow shifts left by the minimum padding, as
        # BatchKVCache.filter does; the physical rows have nothing to
        # compact.
        m = min(pads) if pads else 0
        if m > 0:
            self._idx -= m
            pads = [p - m for p in pads]
        if self._idx == 0 and self._allocated():
            # Every surviving row is empty: back to the empty state.
            for f in self._ARRAY_FIELDS:
                setattr(self, f, None)
            self._reset_rows(pads)
            return
        self._set_pads(pads)

    def _compatible(self, other):
        return type(other) is BatchKVarNKVCache and (
            self.k_bits,
            self.v_bits,
            self.sink_cap,
            self.tail_cap,
        ) == (other.k_bits, other.v_bits, other.sink_cap, other.tail_cap)

    def extend(self, other):
        """Append ``other``'s rows: the buffers concatenate as they are (a
        row keeps its physical geometry) and only the shadow right-aligns
        to the larger _idx."""
        if not self._compatible(other):
            raise ValueError(
                f"[kvarn] cannot extend with {type(other).__name__} "
                "(mismatched class or kvarn parameters)."
            )
        if not self._pads:
            # An emptied batch adopts the incoming rows as they are.
            for f in self._ARRAY_FIELDS:
                setattr(self, f, getattr(other, f))
            self.starts = list(other.starts)
            self.ends = list(other.ends)
            self.tail_ends = list(other.tail_ends)
            self.horizon_valid = list(other.horizon_valid)
            self._right_padding = None
            self._idx = other._idx
            self._set_pads(other._pads)
            return
        target = max(self._idx, other._idx)
        pads = [p + target - self._idx for p in self._pads] + [
            p + target - other._idx for p in other._pads
        ]
        if not self._allocated() and not other._allocated():
            self.starts += list(other.starts)
            self.ends += list(other.ends)
            self.tail_ends += list(other.tail_ends)
            self.horizon_valid += list(other.horizon_valid)
            self._idx = target
            self._set_pads(pads)
            return
        if not (self._allocated() and other._allocated()):
            raise ValueError(
                "[kvarn] cannot extend an empty batch with a filled one "
                "(serve always prefills before admission)."
            )
        for f in self._ARRAY_FIELDS:
            xa, xb = getattr(self, f), getattr(other, f)
            rows = max(xa.shape[2], xb.shape[2])
            grown = []
            for x in (xa, xb):
                if x.shape[2] < rows:
                    pad = mx.zeros(
                        x.shape[:2] + (rows - x.shape[2],) + x.shape[3:], x.dtype
                    )
                    x = mx.concatenate([x, pad], axis=2)
                grown.append(x)
            setattr(self, f, mx.concatenate(grown, axis=0))
        self.starts += list(other.starts)
        self.ends += list(other.ends)
        self.tail_ends += list(other.tail_ends)
        self.horizon_valid += list(other.horizon_valid)
        self._right_padding = None
        self._idx = target
        self._set_pads(pads)

    def extract(self, idx):
        """Row ``idx`` as a single-stream KVarNKVCache with its pad
        stripped (the BatchKVCache.extract model). A row starting at 0
        adopts copies of its buffers bit-exactly, horizon included; a
        padded row rebuilds from its recovered raw K/V, one
        re-quantization pass."""
        s, e = self.starts[idx], self.ends[idx]
        out = KVarNKVCache(
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            tail_tokens=self.tail_cap,
            sink_tokens=self.sink_cap,
        )
        if e - s <= 0:
            return out
        if s == 0:
            for f in self._ARRAY_FIELDS:
                setattr(out, f, mx.contiguous(self._content_row(f, idx)))
            out.offset = e
            out.n_sealed = self.n_sealed_row(idx)
            out.tail_start, out.tail_end = self.tail_start_row(idx), self.tail_ends[idx]
            out.horizon_valid = self.horizon_valid[idx]
            return out
        rk, rv = self._raw_row(idx)
        out.update_and_fetch(rk[:, :, s:], rv[:, :, s:])
        return out

    @classmethod
    def merge(cls, caches):
        """Batch single-stream KVarNKVCache rows (the BatchKVCache.merge
        model: the shadow right-justifies them). One filled row adopts
        copies of its buffers and horizon bit-exactly; when its tail rows
        stop short of tail_cap (a single-stream trim right after a tail
        normalization) the few missing rows are topped up from its body.
        Multi-row merges rebuild from recovered raw K/V, one
        re-quantization pass per row.

        The rotating subclass is refused by exact type: its offset counts
        evicted tokens the buffers no longer hold, so a merge would set the
        row's end too high and attend the sink twice."""
        if not caches:
            raise ValueError("[kvarn] merge requires at least one cache.")
        first = caches[0]
        params = (
            (first.k_bits, first.v_bits, first.sink_cap, first.tail_cap)
            if type(first) is KVarNKVCache
            else None
        )
        for c in caches:
            if (
                type(c) is not KVarNKVCache
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
                setattr(out, f, mx.contiguous(first._content(f)))
            out.starts = [0]
            out.ends = [first.offset]
            out.tail_ends = [first.tail_end]
            out.horizon_valid = [bool(first.horizon_valid)]
            out._idx = first.offset
            out._set_pads([0])
            out._top_up_tail(0)
            return out
        h, d = next(
            (c.stage_k.shape[1], c.stage_k.shape[-1]) for c in caches if c.offset
        )
        slab_k = mx.zeros((len(caches), h, max_len, d), mx.float16)
        slab_v = mx.zeros((len(caches), h, max_len, d), mx.float16)
        for i, c in enumerate(caches):
            if not c.offset:
                continue
            rk, rv = c._raw_single()
            slab_k[i : i + 1, :, max_len - c.offset :] = rk
            slab_v[i : i + 1, :, max_len - c.offset :] = rv
        out.update_and_fetch(slab_k, slab_v)
        return out

    def make_mask(self, N, return_array=False, **kwargs):
        """The stack's shared mask, in the shadow geometry the fp16 layers
        hold. A plain call registers this cache's physical geometry after
        the pending update for the mask memo; the kvarn route itself reads
        the geometry off the cache."""
        del return_array
        from mlx_vlm.models.cache import create_causal_mask

        mask = create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )
        if not any(v is not None for v in kwargs.values()):
            from gmlx.upstream.quantized_sdpa_fix import _register_starts

            _register_starts(
                mask, self.starts_mx, self._mirror(f"ends+{N}", [e + N for e in self.ends])
            )
        return mask

    # -- serialization ------------------------------------------------------

    _VECTOR_FIELDS = ("left_padding", "starts", "ends", "tail_ends", "horizon_valid")

    def _vectors(self):
        return (
            self.left_padding,
            self.starts_mx,
            self.ends_mx,
            self.tail_ends_mx,
            mx.array([1 if v else 0 for v in self.horizon_valid], dtype=mx.int32),
        )

    @property
    def state(self):
        """Content only (see _content) plus the per-row geometry as int32
        vectors; a restore rebuilds the lists once."""
        if not self._allocated():
            z16 = mx.zeros((1, 1, 1, 1), mx.float16)
            z32 = mx.zeros((1, 1, 1, 1), mx.uint32)
            arrays = (z32, z16, z32, z16, z16, z16, z16, z16, z16, z16)
        else:
            arrays = tuple(self._content(f) for f in self._ARRAY_FIELDS)
        return arrays + self._vectors()

    @state.setter
    def state(self, v):
        n = len(self._ARRAY_FIELDS)
        if len(v) != n + len(self._VECTOR_FIELDS):
            raise ValueError(_META_ARITY_MSG)
        for f, a in zip(self._ARRAY_FIELDS, v[:n], strict=True):
            setattr(self, f, a)
        pads, starts, ends, tail_ends, horizon_valid = (
            [int(x) for x in a.tolist()] for a in v[n:]
        )
        self.starts = starts
        self.ends = ends
        self.tail_ends = tail_ends
        self.horizon_valid = [bool(x) for x in horizon_valid]
        self._right_padding = None
        self._set_pads(pads)

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.kvarn_layout_version,
                    self.batch_state_version,
                    1 if self._allocated() else 0,
                    self._idx,
                    self.k_bits,
                    self.v_bits,
                    self.sink_cap,
                    self.tail_cap,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        if len(v) != 8:
            raise ValueError(_META_ARITY_MSG)
        (
            version,
            batch_version,
            allocated,
            self._idx,
            self.k_bits,
            self.v_bits,
            self.sink_cap,
            self.tail_cap,
        ) = map(int, v)
        if version != self.kvarn_layout_version:
            raise ValueError(
                f"[kvarn] cache layout version {version} does not match this "
                f"build ({self.kvarn_layout_version}); refusing to restore."
            )
        if batch_version != self.batch_state_version:
            raise ValueError(
                f"[kvarn] batch cache state version {batch_version} does not "
                f"match this build ({self.batch_state_version}); refusing to "
                "restore."
            )
        if not allocated:
            for f in self._ARRAY_FIELDS:
                setattr(self, f, None)

    def size(self):
        return self._pos

    def empty(self):
        return self._idx == 0

    @property
    def nbytes(self):
        if not self._allocated():
            return 0
        return sum(getattr(self, f).nbytes for f in self._ARRAY_FIELDS)


def _int_list(values):
    """Python ints from a list, tuple or array (one sync for an array)."""
    if isinstance(values, mx.array):
        return [int(x) for x in values.tolist()]
    return [int(x) for x in values]


def kvarn_batch_row(batch, cache):
    """A one-row BatchKVarNKVCache with ``batch``'s kvarn parameters
    holding ``cache``'s history, for admission into ``batch``. A
    single-stream kvarn cache with the same parameters adopts its buffers
    bit-exactly; an fp16 row (a warm hit from the exact tier, a row
    prefilled fp16) or a kvarn row at other widths ingests its recovered
    K/V into a fresh row, one quantization pass."""
    if callable(getattr(cache, "extract", None)) and hasattr(cache, "left_padding"):
        cache = cache.extract(0)
    params = (batch.k_bits, batch.v_bits, batch.sink_cap, batch.tail_cap)
    if type(cache) is KVarNKVCache:
        if (cache.k_bits, cache.v_bits, cache.sink_cap, cache.tail_cap) == params:
            return BatchKVarNKVCache.merge([cache])
        keys, values = cache._raw_single()
    else:
        off = int(getattr(cache, "offset", 0) or 0)
        keys, values = cache.state
        keys, values = keys[:, :, :off], values[:, :, :off]
    row = KVarNKVCache(
        k_bits=batch.k_bits,
        v_bits=batch.v_bits,
        tail_tokens=batch.tail_cap,
        sink_tokens=batch.sink_cap,
    )
    if keys.shape[2]:
        row.update_and_fetch(keys, values)
    return BatchKVarNKVCache.merge([row])

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
        if not hasattr(mod, "KVarNRotatingKVCache"):
            mod.KVarNRotatingKVCache = KVarNRotatingKVCache
        if not hasattr(mod, "BatchKVarNKVCache"):
            mod.BatchKVarNKVCache = BatchKVarNKVCache


def convertible_kv_types():
    """Cache classes the kvarn policy converts to KVarNKVCache.
    ChunkedKVCache is included deliberately: serve's batch build maps it
    to BatchKVCache and converts that, so converting it here keeps the
    CLI and serve paths agreeing on llama4-shaped stacks."""
    from .compat import cache_types

    return cache_types("KVCache") + cache_types("ChunkedKVCache")
