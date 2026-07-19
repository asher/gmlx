"""Fused QKV decode projection for gpt-oss (q8_0 wire + bias).

At decode (q_len == 1, single row), the three per-projection matvecs
(gpt-oss-20b: q [4096, 2880], k [512, 2880], v [512, 2880]) collapse into
ONE ``kq.quantized_matmul_qmv_bias`` dispatch over the row-concatenated
wire bytes, and the two rope dispatches into ONE over the concatenated
q+k head axis. Bit-identical to the unfused path: a q8_0 matvec treats
output rows independently (same K loop, same per-row accumulation order),
and rope is per-head elementwise -- concatenation changes neither.

Why it wins: the decode step is a serial layer chain, and the k/v matvecs
are skinny (1.5 MB reads) -- latency-bound, not bandwidth-bound, so each
extra dispatch is pure fixed cost. Measured on gpt-oss-20b at d512 (M5
Max): 7.06 -> 6.73 ms/step pipelined decode, token-identical output.

The fused weight/bias are built lazily on the first eligible decode call
(install time predates ``load_weights``) and kept out of the parameter
tree. Cost: one extra resident copy of the attention projections (~677 MB
on gpt-oss-20b; the originals still serve prefill). Disable with
``GMLX_FUSED_QKV=0``.
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

_ENABLED = os.environ.get("GMLX_FUSED_QKV", "1") != "0"


def _eligible(m) -> bool:
    if type(m).__name__ != "AttentionBlock":
        return False
    if "gpt_oss" not in type(m).__module__:
        return False
    projs = [getattr(m, p, None) for p in ("q_proj", "k_proj", "v_proj")]
    if not all(isinstance(p, KQuantLinear) for p in projs):
        return False
    if any(p.kquant_type != "q8_0" or "bias" not in p for p in projs):
        return False
    if len({p.weight.shape[1] for p in projs}) != 1:
        return False
    return hasattr(m, "rope") and hasattr(m, "sinks")


def _make_fused(base_cls):
    from mlx_lm.models.base import scaled_dot_product_attention

    class _FusedQKVAttention(base_cls):
        def __call__(self, x, mask, cache=None):
            # Fused path: single-row decode with a cache. Everything else
            # (prefill, batch, speculative widths) is the stock forward.
            if (
                not _ENABLED
                or cache is None
                or x.shape[1] != 1
                or x.size != x.shape[-1]
                or self._kq_qkv_off
            ):
                return super().__call__(x, mask, cache)
            w = self._kq_wqkv
            if w is None:
                w = self._kq_build_fused()
                if w is None:  # post-load shapes disagree; disable for good
                    return super().__call__(x, mask, cache)

            B, L, _ = x.shape
            D = self.head_dim
            n_q = self.num_attention_heads
            n_kv = self.num_key_value_heads

            qkv = kq.quantized_matmul_qmv_bias(
                x, w, self.q_proj["scales"], self._kq_bqkv, "q8_0")
            qk = qkv[..., : (n_q + n_kv) * D]
            qk = qk.reshape(B, L, n_q + n_kv, D).swapaxes(1, 2)
            v = qkv[..., (n_q + n_kv) * D :]
            v = v.reshape(B, L, n_kv, D).swapaxes(1, 2)

            qk = self.rope(qk, offset=cache.offset)
            q, k = qk[:, :n_q], qk[:, n_q:]
            k, v = cache.update_and_fetch(k, v)

            v_hat = scaled_dot_product_attention(
                q, k, v, cache, self.sm_scale, mask=mask, sinks=self.sinks)
            return self.o_proj(v_hat.swapaxes(1, 2).reshape(B, L, -1))

        def _kq_build_fused(self):
            projs = (self.q_proj, self.k_proj, self.v_proj)
            rows = [p["weight"] for p in projs]
            biases = [p["bias"] for p in projs]
            if (
                len({r.shape[1] for r in rows}) != 1
                or len({b.dtype for b in biases}) != 1
                or rows[0].shape[0] != self.num_attention_heads * self.head_dim
                or rows[1].shape[0] != self.num_key_value_heads * self.head_dim
                or rows[1].shape[0] != rows[2].shape[0]
            ):
                object.__setattr__(self, "_kq_qkv_off", True)
                return None
            w = mx.concatenate(rows, axis=0)
            b = mx.concatenate(biases, axis=0)
            mx.eval(w, b)
            object.__setattr__(self, "_kq_wqkv", w)
            object.__setattr__(self, "_kq_bqkv", b)
            return w

    _FusedQKVAttention.__name__ = "_FusedQKVAttention"
    return _FusedQKVAttention


def install_fused_qkv(model) -> int:
    """Class-swap eligible gpt-oss ``AttentionBlock``s onto the fused-QKV
    decode subclass. Weights concatenate lazily on the first fused call.
    Returns the number of instances swapped; 0 when disabled or kq absent."""
    if not _ENABLED or kq is None or not hasattr(kq, "quantized_matmul_qmv_bias"):
        return 0
    classes: dict = {}
    n = 0
    for _, m in model.named_modules():
        if not _eligible(m):
            continue
        base = type(m)
        sub = classes.get(base)
        if sub is None:
            sub = _make_fused(base)
            classes[base] = sub
        m.__class__ = sub
        object.__setattr__(m, "_kq_wqkv", None)
        object.__setattr__(m, "_kq_bqkv", None)
        object.__setattr__(m, "_kq_qkv_off", False)
        n += 1
    return n
