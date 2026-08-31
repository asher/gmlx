"""Owned GatedDeltaNet forward for qwen3.5/3.6 MTP targets.

Subclasses mlx-vlm's ``Qwen3_5GatedDeltaNet``; the owned constructors
build it directly and ``prepare_gdn`` arms the fused routes after
weights load (``rebind_gdn`` arms stock-built trees in tests). Three
routes: fused decode (S=1, no sink), fused verify (sink set, S>1), and
an owned unfused chain for the rest, which sends the scan through
mlx-lm's gated_delta (tiled under the GGUF K->V fixup) and projections
through ``qwen35_verify_linear``.

Owned loads never touch ``mlx_vlm.models.qwen3_5.gated_delta``; the
rebind and class patches install only for the ``GMLX_QWEN_OWNED=0``
fallback. ``GMLX_FUSED_GDN=0`` disables the fused routes.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.qwen3_5 import language as _L

import gmlx.upstream.gdn_patches as _gp
from gmlx.envflags import env_bool
from gmlx.load.loadlog import verbose_print
from .owned import _qwen3_5_advance_left_padding_info, _qwen3_5_advance_lengths_info
from .verify_linear import verify_linear, verify_linears


_QWEN_GDN_FAMILY = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")


def owned_gdn_active(model_type) -> bool:
    """Whether this load takes the owned GatedDeltaNet (same switch as
    the owned model-level forward)."""
    return model_type in _QWEN_GDN_FAMILY and env_bool("GMLX_QWEN_OWNED", True)


def stock_gdn_fallback(model_type) -> bool:
    """True when a GDN-family model runs the bare-stock text fallback
    (GMLX_QWEN_OWNED=0). The fallback has no verify patches and must
    not receive quantized KV tuples."""
    return (model_type in _QWEN_GDN_FAMILY
            and not env_bool("GMLX_QWEN_OWNED", True))


def _gdn_update_with_states_tiled(q, k, v, a, b, A_log, dt_bias, state, mask):
    """Sink-shaped scan, calling the tiled state-capturing ops directly
    (what upstream's ``gated_delta_update_with_states`` resolves to
    after the rebind patch)."""
    from mlx_lm.models.gated_delta import compute_g

    g = compute_g(A_log, a, dt_bias)
    beta = mx.sigmoid(b)
    if state is None:
        B = q.shape[0]
        Hv, Dv = v.shape[-2:]
        Dk = q.shape[-1]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    return _gp._tiled_gd_with_states_ops(q, k, v, g, beta, state, mask)


def _owned_gdn_unfused(self, inputs, mask, cache, gdn_sink, target_verify):
    """The stock forward chain under owned control flow.

    Body mirrors upstream's ``Qwen3_5GatedDeltaNet.__call__`` with three
    substitutions: the scan goes to mlx-lm's gated_delta (tiled under
    the GGUF fixup) instead of the vlm module globals, the sink-shaped
    scan goes to the tiled state-capturing ops, and the memo advance
    uses the owned copies.
    """
    from mlx_lm.models import gated_delta as _gd

    B, S, _ = inputs.shape

    mixed_qkv, z, b, a = verify_linears(
        (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a),
        inputs,
        target_verify,
    )

    z = z.reshape(B, S, -1, self.head_v_dim)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
        if conv_state.shape[0] != B:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
    else:
        conv_state = mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim),
            dtype=inputs.dtype,
        )

    if mask is not None:
        if mask.shape[0] != B:
            mask = None
        else:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    if cache is not None:
        n_keep = self.conv_kernel_size - 1
        if getattr(cache, "lengths", None) is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    if gdn_sink is not None:
        conv_out = nn.silu(self._causal_conv1d_verify(conv_input, S))
    elif (
        S == 1
        and conv_input.shape[1] == self.conv_kernel_size
        and self.conv1d.weight.dtype in (mx.bfloat16, mx.float16)
    ):
        conv_out = nn.silu(self._causal_conv1d_decode(conv_input))
    else:
        conv_out = nn.silu(self.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]

    state = cache[1] if cache else None
    if state is not None and state.shape[0] != B:
        state = None
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    initial_state = state
    if gdn_sink is not None:
        out, state, intermediate_states = _gdn_update_with_states_tiled(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask
        )
    else:
        out, state = _gd.gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        intermediate_states = None

    if gdn_sink is not None:
        gdn_sink.append(
            (
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                initial_state,
                mask,
                conv_input,
                self.conv_kernel_size,
                intermediate_states,
            )
        )

    if cache is not None:
        cache[1] = state
        if hasattr(cache, "advance"):
            cache.advance(S)
            _qwen3_5_advance_left_padding_info(cache, S)
            _qwen3_5_advance_lengths_info(cache, S)

    out = self.norm(out, z)
    return verify_linear(
        self.out_proj, out.reshape(B, S, -1), target_verify
    )


class OwnedQwen3_5GatedDeltaNet(_L.Qwen3_5GatedDeltaNet):
    """Built by the owned constructors (``prepare_gdn`` arms the fused
    routes after weights load); ``rebind_gdn`` remains for arming a
    stock-built module tree in tests."""

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        gdn_sink: Optional[list] = None,
        target_verify: bool = False,
    ) -> mx.array:
        B, S, _ = inputs.shape
        target_verify = target_verify or gdn_sink is not None
        Dv = self.head_v_dim
        fused = getattr(self, "_gdn_owned_fused", False)
        if (
            S == 1
            and gdn_sink is None
            and fused
            and _gp.gpu_active()
            and _gp._gdn_fused_decode_kernel is not None
            and (mask is None or not isinstance(mask, mx.array))
            and cache is not None
            and cache[1] is not None
            and Dv % _gp.gdn_sg(B) == 0
            and self.head_k_dim % 32 == 0
        ):
            return _gp._gdn_fused_decode_body(
                self, inputs, cache, vlm_cache_advance=True
            )
        if (
            fused
            and _gp.gpu_active()
            and gdn_sink is not None
            and S > 1
            and _gp._gdn_fused_verify_kernel is not None
            and Dv % _gp.gdn_sg(B) == 0
            and self.head_k_dim % 32 == 0
        ):
            return _gp._gdn_fused_verify_body(
                self, inputs, mask, cache, gdn_sink
            )
        return _owned_gdn_unfused(
            self, inputs, mask, cache, gdn_sink, target_verify
        )


def prepare_gdn(model) -> int:
    """Arm the fused routes on every owned GatedDeltaNet in ``model``.

    Runs after weights load (the b/a concatenation reads loaded
    weights); an unarmed tree decodes on the unfused chain. Returns the
    armed count; raises on zero, which means the install site gated on
    the wrong predicate for a stock-built target.
    """
    fused = env_bool("GMLX_FUSED_GDN", True)
    cat_ba = env_bool("GMLX_GDN_BA_CAT", True)
    lm = getattr(model, "language_model", model)
    n = 0
    n_ba = 0
    for m in lm.modules():
        if isinstance(m, OwnedQwen3_5GatedDeltaNet):
            m._gdn_owned_fused = fused
            n += 1
            if fused and cat_ba and getattr(m, "_gdn_ba_weight", None) is None:
                if _gp._gdn_try_cat_ba(m):
                    n_ba += 1
    if n == 0:
        raise RuntimeError(
            "prepare_gdn armed 0 layers: no owned GatedDeltaNet in this "
            "tree, so the install site gated on the wrong predicate for "
            "a stock-built target"
        )
    ba = f", b/a matvecs concatenated on {n_ba}" if n_ba else ""
    verbose_print(f"[build] gated_delta: owned forward on {n} layers{ba}")
    return n


def rebind_gdn(model) -> int:
    """Rebind every stock GatedDeltaNet in ``model`` to the owned class
    and arm it. Real loads build owned classes at construction; this is
    the install path for stock-built toy trees in tests. Returns the
    rebind count.
    """
    lm = getattr(model, "language_model", model)
    n = 0
    for m in lm.modules():
        if type(m) is _L.Qwen3_5GatedDeltaNet:
            m.__class__ = OwnedQwen3_5GatedDeltaNet
            n += 1
    prepare_gdn(model)
    return n
