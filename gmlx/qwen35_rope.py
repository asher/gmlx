"""Owned MRoPE apply chain for qwen3.5/3.6 MTP targets.

In-tree copy of the rope_utils closure the qwen3.5 interleaved rotary
reaches: the fused Metal MRoPE kernel and its compiled/custom-vjp
wrappers, the cos/sin frequency chain, and
``apply_multimodal_rotary_pos_emb``. The owned attention and model
forwards call the two free-function mirrors at the bottom
(``rope_cos_sin``, ``rope_apply_rotary``) on the stock rotary
submodule -- construction stays upstream scaffold; the forward math
runs from this module.

The upstream bodies are verbatim copies, source-equality-tested
against the pinned mlx-vlm release every run (see
tests/test_qwen35_rope.py). The two sectioned-style helpers aliased
below are dead branches for the interleaved style qwen3.5 pins; they
are imported rather than copied so the verbatim bodies that mention
them stay byte-identical.
"""

from functools import lru_cache
from typing import Optional, Sequence

import mlx.core as mx

from mlx_vlm.models.rope_utils import (  # sectioned-style branches only
    _maybe_fast_precomputed_rotary,
    _section_cos_sin,
)

__all__ = ["rope_cos_sin", "rope_apply_rotary", "apply_multimodal_rotary_pos_emb"]


# --- verbatim upstream copies (rope_utils.py) ---

_HAS_METAL = mx.metal.is_available()
_HALF_SPLIT = "half_split"
_EVEN_ODD = "even_odd"
_HALF_COS = "half"
_FULL_COS = "full"

def _interleaved_position_selector(mrope_section: Sequence[int], freq_dim: int):
    selector = [0] * freq_dim
    for dim, offset in enumerate((1, 2), start=1):
        for idx in range(offset, min(mrope_section[dim] * 3, freq_dim), 3):
            selector[idx] = dim
    return mx.array(selector, dtype=mx.int32)


def _chunked_position_selector(mrope_section: Sequence[int], freq_dim: int):
    selector = [0] * freq_dim
    offset = mrope_section[0]
    for dim, length in enumerate(mrope_section[1:], start=1):
        for idx in range(offset, min(offset + length, freq_dim)):
            selector[idx] = dim
        offset += length
    return mx.array(selector, dtype=mx.int32)


@mx.compile
def _selected_mrope_freqs(position_ids, inv_freq, position_selector):
    positions = mx.take(position_ids, position_selector, axis=0).transpose(1, 2, 0)
    return positions.astype(mx.float32) * inv_freq


def mrope_position_selector(style: str, mrope_section: Sequence[int], freq_dim: int):
    if style == "interleaved":
        return _interleaved_position_selector(mrope_section, freq_dim)
    return _chunked_position_selector(mrope_section, freq_dim)


def _selects_frequency_by_position(style: str):
    return style in {"chunked", "interleaved", "split_select"}


def _is_sectioned_style(style: str):
    return style in {"sectioned_half_split", "sectioned_even_odd"}


def _has_mrope_apply_selector(style: str):
    return _selects_frequency_by_position(style) or _is_sectioned_style(style)


def _uses_even_odd_pairing(style: str):
    return style in {"sectioned_even_odd", "split_select", "ernie_3d"}


def _needs_even_odd_layout(style: str):
    return style in {"sectioned_even_odd", "split_select"}


def _pairing_for_style(style: str):
    if _uses_even_odd_pairing(style):
        return _EVEN_ODD
    return _HALF_SPLIT

@lru_cache(maxsize=None)
def _mrope_apply_kernel(
    rotary_dim: int,
    position_ndim: int,
    pairing: str,
):
    if not _HAS_METAL:
        return None

    if position_ndim == 2:
        position_expr = "position_ids[b * q_len + t]"
        selector_source = ""
    else:
        position_expr = "position_ids[(axis * q_bsz + b) * q_len + t]"
        selector_source = "int axis = int(position_selector[freq_idx]);"

    if pairing == _EVEN_ODD:
        pair_source = f"""
        int freq_idx = slot;
        int d = freq_idx * 2;
        int pair_d = d + 1;
        {selector_source}
        float pos = static_cast<float>({position_expr});
        float angle = pos * static_cast<float>(inv_freq[freq_idx]);
        float c = metal::cos(angle);
        float s = metal::sin(angle);
        """
    else:
        pair_source = f"""
        int freq_idx = slot;
        int d = freq_idx;
        int pair_d = d + half_dim;
        {selector_source}
        float pos = static_cast<float>({position_expr});
        float angle = pos * static_cast<float>(inv_freq[freq_idx]);
        float c = metal::cos(angle);
        float s = metal::sin(angle);
        """

    source = f"""
        uint elem = thread_position_in_grid.x;

        const int half_dim = {rotary_dim // 2};
        const int q_bsz = x_shape[0];
        const int q_heads = x_shape[1];
        const int q_len = x_shape[2];
        const int q_dim = x_shape[3];
        const int slots = half_dim + q_dim - {rotary_dim};
        const int work_size = q_bsz * q_heads * q_len * slots;

        if (elem >= uint(work_size)) {{
            return;
        }}

        int local = int(elem);
        int slot = local % slots;
        int tmp = local / slots;
        int t = tmp % q_len;
        tmp = tmp / q_len;
        int h = tmp % q_heads;
        int b = tmp / q_heads;
        int base = ((b * q_heads + h) * q_len + t) * q_dim;

        if (slot >= half_dim) {{
            int pass_d = {rotary_dim} + slot - half_dim;
            int pass_idx = base + pass_d;
            x_out[pass_idx] = x[pass_idx];
            return;
        }}

        {pair_source}

        int idx = base + d;
        float xv = static_cast<float>(x[idx]);
        float xp = static_cast<float>(x[base + pair_d]);
        x_out[idx] = static_cast<T>(xv * c - xp * s);
        x_out[base + pair_d] = static_cast<T>(xp * c + xv * s);
    """

    return mx.fast.metal_kernel(
        name=f"mrope_apply_{pairing}_{rotary_dim}_{position_ndim}d",
        input_names=["x", "position_ids", "inv_freq", "position_selector"],
        output_names=["x_out"],
        source=source,
    )


def _mrope_apply_cos_sin(x, position_ids, inv_freq, position_selector, pairing):
    """Pure-MLX cos/sin matching the fused MRoPE kernel's angle computation.

    Mirrors ``_mrope_apply_kernel``: ``angle = pos * inv_freq[freq_idx]`` with
    ``pos`` selected per-axis (via ``position_selector``) for 3D position ids.
    Returns cos/sin already laid out for the pairing so a plain rotate matches
    the kernel's element-wise pair rotation.
    """
    half_dim = inv_freq.shape[0]
    if position_ids.ndim == 2:
        # (b, t, half) angles; the same scalar position feeds every freq.
        angle = position_ids.astype(mx.float32)[..., None] * inv_freq
    else:
        # (axis, b, t) -> select the axis feeding each freq, giving (b, t, half).
        positions = mx.take(position_ids, position_selector, axis=0)
        angle = positions.transpose(1, 2, 0).astype(mx.float32) * inv_freq
    cos = mx.cos(angle)[:, None, :, :]
    sin = mx.sin(angle)[:, None, :, :]
    if pairing == _EVEN_ODD:
        cos = mx.repeat(cos, repeats=2, axis=-1)
        sin = mx.repeat(sin, repeats=2, axis=-1)
    else:
        cos = mx.concatenate([cos, cos], axis=-1)
        sin = mx.concatenate([sin, sin], axis=-1)
    return cos, sin, half_dim


def _mrope_apply(q, k, position_ids, inv_freq, position_selector, pairing):
    """Differentiable pure-MLX equivalent of the fused MRoPE kernel apply."""
    cos, sin, _ = _mrope_apply_cos_sin(
        q, position_ids, inv_freq, position_selector, pairing
    )
    rotate_fn = rotate_half_even_odd if pairing == _EVEN_ODD else rotate_half
    return _apply_rotary_embedding(
        q, k, cos, sin, rotate_fn, cast_output=True, compute_dtype=mx.float32
    )


def _fast_mrope_apply(
    kernel,
    q,
    k,
    position_ids,
    inv_freq,
    position_selector,
    pairing=_HALF_SPLIT,
):
    def apply_one(x):
        half_dim = inv_freq.shape[0]
        slots = half_dim + x.shape[-1] - half_dim * 2
        work_size = x.shape[0] * x.shape[1] * x.shape[2] * slots
        (out,) = kernel(
            inputs=[x, position_ids, inv_freq, position_selector],
            template=[("T", x.dtype)],
            grid=(work_size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )
        return out

    # Wrap the kernel forward in a custom_function so value_and_grad (training)
    # can differentiate through it: a raw CustomKernel has no VJP, so route the
    # gradient through the pure-MLX equivalent. position_ids/inv_freq/
    # position_selector are position constants (zero cotangent).
    @mx.custom_function
    def apply(q, k, position_ids, inv_freq, position_selector):
        return apply_one(q), apply_one(k)

    @apply.vjp
    def apply_vjp(primals, cotangents, _output):
        q, k, position_ids, inv_freq, position_selector = primals
        _, (dq, dk) = mx.vjp(
            lambda q, k: list(
                _mrope_apply(q, k, position_ids, inv_freq, position_selector, pairing)
            ),
            [q, k],
            list(cotangents),
        )
        return (
            dq,
            dk,
            mx.zeros_like(position_ids),
            mx.zeros_like(inv_freq),
            mx.zeros_like(position_selector),
        )

    return apply(q, k, position_ids, inv_freq, position_selector)

@lru_cache(maxsize=None)
def _compiled_mrope_apply(rotary_dim: int, position_ndim: int, pairing: str):
    kernel = _mrope_apply_kernel(rotary_dim, position_ndim, pairing)
    if kernel is None:
        return None

    @mx.compile
    def apply(q, k, position_ids, inv_freq, position_selector):
        return _fast_mrope_apply(
            kernel,
            q,
            k,
            position_ids,
            inv_freq,
            position_selector,
            pairing,
        )

    return apply


def get_mrope_section(
    *,
    rope_scaling: Optional[dict] = None,
    rope_parameters: Optional[dict] = None,
    default: Sequence[int] = (24, 20, 20),
):
    rope_scaling = rope_scaling or {}
    rope_parameters = rope_parameters or {}
    return list(
        rope_parameters.get("mrope_section")
        or rope_scaling.get("mrope_section")
        or default
    )


def compute_inv_freq(dim: int, base: float):
    return 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

@mx.compile
def _apply_selected_mrope_frequency_layout(freqs, position_selector):
    indices = mx.broadcast_to(
        position_selector[None, None, None, :],
        (1, freqs.shape[1], freqs.shape[2], freqs.shape[3]),
    )
    return mx.take_along_axis(freqs, indices, axis=0)[0]

def apply_mrope_frequency_layout(
    freqs,
    mrope_section: Sequence[int],
    *,
    style: str = "interleaved",
):
    mrope_section = list(mrope_section)

    if _selects_frequency_by_position(style):
        position_selector = mrope_position_selector(
            style,
            mrope_section,
            freqs.shape[-1],
        )
        return _apply_selected_mrope_frequency_layout(freqs, position_selector)
    return freqs


def compute_mrope_frequencies(
    position_ids,
    inv_freq,
    mrope_section: Sequence[int],
    *,
    style: str = "interleaved",
    position_selector=None,
):
    if position_ids.ndim == 2:
        # Text-only positions use the same scalar position for every MRoPE axis,
        # so chunked/interleaved layout selection collapses to the same angles.
        return position_ids.astype(mx.float32)[..., None] * inv_freq

    # Fast path
    if _selects_frequency_by_position(style):
        if position_selector is None:
            position_selector = mrope_position_selector(
                style,
                mrope_section,
                inv_freq.shape[0],
            )
        return _selected_mrope_freqs(position_ids, inv_freq, position_selector)

    # Slow path
    freqs = position_ids.astype(mx.float32)[..., None] * inv_freq
    return apply_mrope_frequency_layout(freqs, mrope_section, style=style)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def rotate_half_even_odd(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return mx.flatten(mx.stack([-x2, x1], axis=-1), start_axis=-2, end_axis=-1)


def _apply_rotary_embedding(
    q,
    k,
    cos,
    sin,
    rotate_fn,
    *,
    cast_output: bool = True,
    compute_dtype=None,
):
    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    if compute_dtype is not None:
        q_rot = q_rot.astype(compute_dtype)
        k_rot = k_rot.astype(compute_dtype)

    q_embed = (q_rot * cos) + (rotate_fn(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_fn(k_rot) * sin)

    if cast_output:
        q_embed = q_embed.astype(q.dtype)
        k_embed = k_embed.astype(k.dtype)

    if q_pass.shape[-1] == 0 and k_pass.shape[-1] == 0:
        return q_embed, k_embed

    return (
        mx.concatenate([q_embed, q_pass], axis=-1),
        mx.concatenate([k_embed, k_pass], axis=-1),
    )

@mx.compile
def _apply_interleaved_rotary_pos_emb_axis1(q, k, cos, sin):
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)

    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = q_embed.astype(q.dtype)
    k_embed = k_embed.astype(k.dtype)

    return (
        mx.concatenate([q_embed, q_pass], axis=-1),
        mx.concatenate([k_embed, k_pass], axis=-1),
    )

def apply_multimodal_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    *,
    mrope_section: Optional[Sequence[int]] = None,
    unsqueeze_dim: int = 1,
    style: str = "interleaved",
    cast_output: bool = True,
):
    if style == "interleaved" and unsqueeze_dim == 1 and cast_output:
        return _apply_interleaved_rotary_pos_emb_axis1(q, k, cos, sin)

    if _is_sectioned_style(style):
        if mrope_section is None:
            raise ValueError("mrope_section is required for sectioned MRoPE")
        fast = _maybe_fast_precomputed_rotary(
            q,
            k,
            cos,
            sin,
            pairing=_pairing_for_style(style),
            mrope_section=mrope_section,
            unsqueeze_dim=unsqueeze_dim,
            cast_output=cast_output,
        )
        if fast is not None:
            return fast
        cos, sin = _section_cos_sin(cos, sin, mrope_section)
    else:
        cos = mx.expand_dims(cos, axis=unsqueeze_dim)
        sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    if _needs_even_odd_layout(style):
        cos = mx.repeat(cos[..., : cos.shape[-1] // 2], repeats=2, axis=-1)
        sin = mx.repeat(sin[..., : sin.shape[-1] // 2], repeats=2, axis=-1)
        rotate_fn = rotate_half_even_odd
    else:
        rotate_fn = rotate_half

    return _apply_rotary_embedding(
        q,
        k,
        cos,
        sin,
        rotate_fn,
        cast_output=cast_output,
    )


# --- free-function mirrors of the MRoPERotaryEmbedding forwards ---


def rope_cos_sin(rotary_emb, x, position_ids):
    """Mirror of ``MRoPERotaryEmbedding.__call__`` on the stock rotary
    submodule, resolving the frequency chain to the owned copies."""
    freqs = compute_mrope_frequencies(
        position_ids,
        rotary_emb.inv_freq,
        rotary_emb.mrope_section,
        style=rotary_emb.style,
        position_selector=rotary_emb.position_selector,
    )
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos = mx.cos(emb) * rotary_emb.attention_scaling
    sin = mx.sin(emb) * rotary_emb.attention_scaling

    if rotary_emb.cast_output:
        return cos.astype(x.dtype), sin.astype(x.dtype)
    return cos, sin


def rope_apply_rotary(
    rotary_emb,
    q,
    k,
    position_ids,
    *,
    unsqueeze_dim: int = 1,
    cast_output: bool = True,
):
    """Mirror of ``MRoPERotaryEmbedding.apply_rotary`` with the compiled
    apply built from the owned kernel factory. The per-ndim memo lives
    on the instance under an owned attr so a stock-compiled entry can
    never masquerade as the owned path."""
    if (
        rotary_emb.fused_apply
        and unsqueeze_dim == 1
        and position_ids.ndim in (2, 3)
        and q.ndim == 4
        and k.ndim == 4
    ):
        memo = getattr(rotary_emb, "_gmlx_compiled_apply", None)
        if memo is None:
            memo = {}
            rotary_emb._gmlx_compiled_apply = memo
        compiled_apply = memo.get(position_ids.ndim)
        if compiled_apply is None:
            compiled_apply = _compiled_mrope_apply(
                rotary_emb.dim, position_ids.ndim, rotary_emb.pairing
            )
            if compiled_apply is not None:
                memo[position_ids.ndim] = compiled_apply

        if compiled_apply is not None:
            return compiled_apply(
                q,
                k,
                position_ids,
                rotary_emb.inv_freq,
                rotary_emb.position_selector,
            )

    cos, sin = rope_cos_sin(rotary_emb, k, position_ids)
    return apply_multimodal_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
        mrope_section=rotary_emb.mrope_section,
        unsqueeze_dim=unsqueeze_dim,
        style=rotary_emb.style,
        cast_output=cast_output,
    )
