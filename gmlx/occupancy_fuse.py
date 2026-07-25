"""Occupancy-driven projection fusion for dense mlx-lm archs (llama first).

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

import mlx.core as mx

try:
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantLinear
except ImportError:  # pragma: no cover - kq is a hard dep of the loader
    kq = None
    KQuantLinear = ()


def _on() -> bool:
    return os.environ.get("GMLX_OCCUPANCY_FUSE", "1") != "0"


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


def _mlp_eligible(m) -> bool:
    if type(m).__name__ != "MLP":
        return False
    if not type(m).__module__.endswith("models.llama"):
        return False
    projs = [getattr(m, p, None) for p in ("gate_proj", "up_proj")]
    if not _same_codec_kquant(projs):
        return False
    return len({p.weight.shape[1] for p in projs}) == 1


def _make_fused_attention(base_cls):
    from mlx_lm.models.base import scaled_dot_product_attention

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

            output = scaled_dot_product_attention(
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
            return self.down_proj(swiglu(gu[..., :n], gu[..., n:]))

        def _kq_build_fused(self):
            projs = (self.gate_proj, self.up_proj)
            if projs[0].weight.shape != projs[1].weight.shape:
                object.__setattr__(self, "_kq_fuse_off", True)
                return None
            w = _fuse_wires(projs)
            object.__setattr__(self, "_kq_wgu", w)
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
        object.__setattr__(m, "_kq_fuse_off", False)
        n += 1
    return n
