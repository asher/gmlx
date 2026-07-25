"""Occupancy-driven projection fusion for dense mlx-lm archs.

Covers llama (Attention/MLP) and qwen3_5 text (full-attention layers +
Qwen3NextMLP; GDN layers are untouched). qwen3_5's q_proj carries the
output gate (split post-matmul) and q/k RMS-norms run after the split,
so the fused path replays the exact stock op sequence on slices of one
matmul output. SDPA resolves from the defining module at call time so
seam patches (qwen35_verify_fold) apply to the fused path too.

Second lever, same mechanism on the other axis: at B >= 12 the fused
MLP also splits down_proj into two half-K matmuls plus an add
(GMLX_SPLITK_MIN_B, 0 kills). The long-K single launch craters at the
M=12 route cliff (q6_k [4096x14336] serial 224 -> 124 GB/s) while two
overlapping halves hold 165; below the cliff the mv route wins whole,
so the split never runs there. The add costs one extra bf16 rounding,
so this path is allclose-not-bit-identical to stock; B = 1 and all
single-stream traffic never reach it. Down halves are extra resident
copies (originals stay for prefill), like the fused wires.

Batched decode runs the layer chain serially, so each projection executes
alone; the small-M qmm tiles put up few threadgroups (llama-8B kv
[1024x4096] = 16, q/o [4096x4096] = 64) and underfill the chip. Measured
in the serial regime on q6_k at M=16: q+k+v separate = 316 us (the kv
matmul alone runs at 33.7 GB/s), fused [6144x4096] = 126 us; gate+up
fused wins 1.17x. Fusing is an occupancy lever here, not a launch-count
lever: the whole non-matmul dispatch soup costs ~1.2 ms per step flat.

Mechanics follow gmlx.qkv_fuse (gpt-oss): class-swap eligible modules
onto a subclass whose batched-decode path (L == 1, B >= 2) runs ONE
matmul over row-concatenated wire bytes, split afterwards. B == 1
stays stock: the M=1 matvec grids already fill the chip and the
fused-output slicing costs a consistent ~0.7% there. K-quant wire
rows are independent, so concatenation is bit-exact per output row (the
q6_k scales tensor is a placeholder; the block scales live in the wire).
Rope runs on q and k separately AFTER the split so the fused path makes
the exact same rope calls as stock (bit-comparable arm to arm; the
slices are views, the extra call is cheap). Certifying this path is
what surfaced the upstream int-offset batched rope bug -- see
rope_batch_fix, which must be installed for any B >= 2 plain-KVCache
chain to be correct at all. Originals stay resident for prefill
and the parameter tree (one extra copy of the fused projections).
Disable with GMLX_OCCUPANCY_FUSE=0 (read per call; A/B safe).
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx

try:
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantLinear
except ImportError:  # pragma: no cover - kq is a hard dep of the loader
    kq = None
    KQuantLinear = ()


def _on() -> bool:
    return os.environ.get("GMLX_OCCUPANCY_FUSE", "1") != "0"


def _splitk_min_b() -> int:
    # Width where the down_proj K-split engages. The serial-probe band
    # (q6_k [4096x14336]): whole craters 224 -> 124 GB/s at the M=12
    # route cliff while two K-halves + add hold 165 (+33%); at M<=10
    # the split is neutral, at M=2-8 the mv route wins whole. 0 kills.
    try:
        return int(os.environ.get("GMLX_SPLITK_MIN_B", "12"))
    except ValueError:
        return 12


def _same_codec_kquant(projs) -> bool:
    if not all(isinstance(p, KQuantLinear) for p in projs):
        return False
    if any("bias" in p for p in projs):
        return False
    return len({p.kquant_type for p in projs}) == 1


def _fuse_wires(projs):
    w = mx.concatenate([p["weight"] for p in projs], axis=0)
    mx.eval(w)
    return w


def _sdpa_of(cls):
    # resolve from the defining module at call time so seam patches
    # (e.g. qwen35_verify_fold) cover the fused path too
    return sys.modules[cls.__module__].scaled_dot_product_attention


def _attention_eligible(m) -> bool:
    if type(m).__name__ != "Attention":
        return False
    if not type(m).__module__.endswith("models.llama"):
        return False
    projs = [getattr(m, p, None) for p in ("q_proj", "k_proj", "v_proj")]
    if not _same_codec_kquant(projs):
        return False
    if len({p.weight.shape[1] for p in projs}) != 1:
        return False
    return hasattr(m, "rope")


_QWEN_MODULES = ("models.qwen3_5", "models.qwen3_next")


def _qwen35_attention_eligible(m) -> bool:
    # the qwen3_5 hybrid reuses qwen3_next classes; both share the
    # gated-q + post-split-norm flow the fused path replays
    if type(m).__name__ not in ("Attention", "Qwen3NextAttention"):
        return False
    if not type(m).__module__.endswith(_QWEN_MODULES):
        return False
    projs = [getattr(m, p, None) for p in ("q_proj", "k_proj", "v_proj")]
    if not _same_codec_kquant(projs):
        return False
    if len({p.weight.shape[1] for p in projs}) != 1:
        return False
    return all(hasattr(m, a) for a in ("rope", "q_norm", "k_norm"))


def _mlp_eligible(m) -> bool:
    if type(m).__name__ == "MLP":
        if not type(m).__module__.endswith("models.llama"):
            return False
    elif type(m).__name__ == "Qwen3NextMLP":
        if not type(m).__module__.endswith(_QWEN_MODULES):
            return False
    else:
        return False
    projs = [getattr(m, p, None) for p in ("gate_proj", "up_proj")]
    if not _same_codec_kquant(projs):
        return False
    return len({p.weight.shape[1] for p in projs}) == 1


def _make_fused_attention(base_cls):
    class _FusedAttention(base_cls):
        def __call__(self, x, mask=None, cache=None):
            if (
                not _on()
                or cache is None
                or x.ndim != 3
                or x.shape[1] != 1
                or x.shape[0] < 2
                or self._kq_fuse_off
            ):
                return super().__call__(x, mask=mask, cache=cache)
            w = self._kq_wqkv
            if w is None:
                w = self._kq_build_fused()
                if w is None:
                    return super().__call__(x, mask=mask, cache=cache)

            B, L, _ = x.shape
            n_q, n_kv = self.n_heads, self.n_kv_heads
            D = self.head_dim
            qkv = kq.quantized_matmul(
                x, w, self.q_proj["scales"], self.q_proj.kquant_type,
                transpose=True,
            )
            queries = qkv[..., : n_q * D]
            queries = queries.reshape(B, L, n_q, D).transpose(0, 2, 1, 3)
            keys = qkv[..., n_q * D: (n_q + n_kv) * D]
            keys = keys.reshape(B, L, n_kv, D).transpose(0, 2, 1, 3)
            values = qkv[..., (n_q + n_kv) * D:]
            values = values.reshape(B, L, n_kv, D).transpose(0, 2, 1, 3)

            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)

            output = _sdpa_of(base_cls)(
                queries, keys, values, cache=cache, scale=self.scale,
                mask=mask,
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)

        def _kq_build_fused(self):
            projs = (self.q_proj, self.k_proj, self.v_proj)
            D = self.head_dim
            if (
                projs[0].weight.shape[0] != self.n_heads * D
                or projs[1].weight.shape[0] != self.n_kv_heads * D
                or projs[1].weight.shape[0] != projs[2].weight.shape[0]
            ):
                object.__setattr__(self, "_kq_fuse_off", True)
                return None
            w = _fuse_wires(projs)
            object.__setattr__(self, "_kq_wqkv", w)
            return w

    _FusedAttention.__name__ = "_FusedAttention"
    return _FusedAttention


def _make_fused_qwen35_attention(base_cls):
    class _FusedQwen35Attention(base_cls):
        def __call__(self, x, mask=None, cache=None):
            if (
                not _on()
                or cache is None
                or x.ndim != 3
                or x.shape[1] != 1
                or x.shape[0] < 2
                or self._kq_fuse_off
            ):
                return super().__call__(x, mask=mask, cache=cache)
            w = self._kq_wqkv
            if w is None:
                w = self._kq_build_fused()
                if w is None:
                    return super().__call__(x, mask=mask, cache=cache)

            B, L, _ = x.shape
            n_q = self.num_attention_heads
            n_kv = self.num_key_value_heads
            qr, kr = self._kq_q_rows, self._kq_k_rows
            qkv = kq.quantized_matmul(
                x, w, self.q_proj["scales"], self.q_proj.kquant_type,
                transpose=True,
            )
            queries, gate = mx.split(
                qkv[..., :qr].reshape(B, L, n_q, -1), 2, axis=-1)
            gate = gate.reshape(B, L, -1)
            keys = qkv[..., qr: qr + kr]
            values = qkv[..., qr + kr:]

            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(B, L, n_kv, -1)).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, n_kv, -1).transpose(0, 2, 1, 3)

            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)

            output = _sdpa_of(base_cls)(
                queries, keys, values, cache=cache, scale=self.scale,
                mask=mask,
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output * mx.sigmoid(gate))

        def _kq_build_fused(self):
            projs = (self.q_proj, self.k_proj, self.v_proj)
            qr = projs[0].weight.shape[0]
            kr = projs[1].weight.shape[0]
            # q carries queries + gate (2x per head); k/v rows must match
            if (
                qr % (2 * self.num_attention_heads) != 0
                or kr != projs[2].weight.shape[0]
                or kr % self.num_key_value_heads != 0
            ):
                object.__setattr__(self, "_kq_fuse_off", True)
                return None
            w = _fuse_wires(projs)
            object.__setattr__(self, "_kq_q_rows", qr)
            object.__setattr__(self, "_kq_k_rows", kr)
            object.__setattr__(self, "_kq_wqkv", w)
            return w

    _FusedQwen35Attention.__name__ = "_FusedQwen35Attention"
    return _FusedQwen35Attention


def _make_fused_mlp(base_cls):
    from mlx_lm.models.activations import swiglu

    class _FusedMLP(base_cls):
        def __call__(self, x):
            if (
                not _on()
                or x.ndim != 3
                or x.shape[1] != 1
                or x.shape[0] < 2
                or self._kq_fuse_off
            ):
                return super().__call__(x)
            w = self._kq_wgu
            if w is None:
                w = self._kq_build_fused()
                if w is None:
                    return super().__call__(x)
            gu = kq.quantized_matmul(
                x, w, self.gate_proj["scales"], self.gate_proj.kquant_type,
                transpose=True,
            )
            n = self.gate_proj.weight.shape[0]
            h = swiglu(gu[..., :n], gu[..., n:])

            minb = _splitk_min_b()
            dn = self._kq_wdn
            if minb <= 0 or h.shape[0] < minb or dn is None:
                return self.down_proj(h)
            # down_proj K-split: two half-K matmuls overlap where the
            # single long-K launch underfills the chip. One extra bf16
            # rounding on the add (not bit-identical to stock). The
            # halves MUST be materialized dense: a [B, 1, K/2] view
            # passes the kq last-two-dims contiguity check through its
            # size-1 axis, routes as M=1, and re-reads the weight per
            # batch row (measured +42% step at B=16 without this).
            w_lo, w_hi, k_half = dn
            d = self.down_proj
            h_lo = mx.contiguous(h[..., :k_half])
            h_hi = mx.contiguous(h[..., k_half:])
            lo = kq.quantized_matmul(
                h_lo, w_lo, d["scales"], d.kquant_type, transpose=True)
            hi = kq.quantized_matmul(
                h_hi, w_hi, d["scales"], d.kquant_type, transpose=True)
            return lo + hi

        def _kq_build_fused(self):
            projs = (self.gate_proj, self.up_proj)
            if projs[0].weight.shape != projs[1].weight.shape:
                object.__setattr__(self, "_kq_fuse_off", True)
                return None
            w = _fuse_wires(projs)
            object.__setattr__(self, "_kq_wgu", w)
            d = self.down_proj
            k = projs[0].weight.shape[0]
            wire_cols = d.weight.shape[1] if isinstance(d, KQuantLinear) \
                else 0
            if (
                isinstance(d, KQuantLinear)
                and "bias" not in d
                and wire_cols % 2 == 0
                and k % 512 == 0
            ):
                half = wire_cols // 2
                w_lo = mx.contiguous(d["weight"][:, :half])
                w_hi = mx.contiguous(d["weight"][:, half:])
                mx.eval(w_lo, w_hi)
                object.__setattr__(
                    self, "_kq_wdn", (w_lo, w_hi, k // 2))
            return w

    _FusedMLP.__name__ = "_FusedMLP"
    return _FusedMLP


def install_occupancy_fuse(model) -> int:
    """Class-swap eligible llama Attention/MLP modules onto fused-decode
    subclasses. Fused wires concatenate lazily on the first fused call.
    Returns instances swapped; 0 when disabled at install or kq absent."""
    if not _on() or kq is None:
        return 0
    classes: dict = {}
    n = 0
    for _, m in model.named_modules():
        maker = None
        slot = None
        if _attention_eligible(m):
            maker, slot = _make_fused_attention, "_kq_wqkv"
        elif _qwen35_attention_eligible(m):
            maker, slot = _make_fused_qwen35_attention, "_kq_wqkv"
        elif _mlp_eligible(m):
            maker, slot = _make_fused_mlp, "_kq_wgu"
        if maker is None:
            continue
        base = type(m)
        sub = classes.get(base)
        if sub is None:
            sub = maker(base)
            classes[base] = sub
        m.__class__ = sub
        object.__setattr__(m, slot, None)
        if slot == "_kq_wgu":
            object.__setattr__(m, "_kq_wdn", None)
        object.__setattr__(m, "_kq_fuse_off", False)
        n += 1
    return n
