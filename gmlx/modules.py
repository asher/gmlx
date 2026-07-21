"""Leaf-swap installer + the GGUF-runtime-specific kquant / native-fp / LoRA modules.

The canonical kquant ``nn.Module`` subclasses (``KQuantLinear`` /
``KQuantEmbedding`` / ``KQuantSwitchLinear`` / ``KQuantMultiLinear``) live in
``mlx_kquant.nn`` and are imported here. Each stores the GGUF wire bytes directly
as a ``uint8`` ``weight`` and dispatches through the standalone ``mlx_kquant``
extension (``kq.dequantize`` / ``kq.quantized_matmul`` / ``kq.gather_qmm``) on a
stock, unmodified ``mlx`` wheel. This module keeps the pieces specific to the GGUF
runtime: the micro-scaling float experts (``NativeFPSwitchLinear``), the live
GGUF-LoRA wrapper (``LoRAKQuantLinear`` / ``install_lora_adapter``), and the
leaf-swap installer.

``install_kquant_modules`` is the module-swap seam: it walks the constructed
mlx-lm model's leaf modules and replaces each ``Linear`` / ``Embedding`` /
``SwitchLinear`` whose ``<path>.weight`` carries a codec with the matching kquant
(or native-fp) equivalent. It is arch-generic - driven entirely by codec strings -
so generalizing architecture coverage never touches this file.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from mlx_lm.models.switch_layers import SwitchLinear

try:  # absorbed-MLA module (deepseek_v3 family); absent on older mlx-lm.
    from mlx_lm.models.mla import MultiLinear
except Exception:  # pragma: no cover - depends on installed mlx-lm version
    MultiLinear = None

from mlx_kquant.nn import (
    KQuantEmbedding,
    KQuantLinear,
    KQuantMultiLinear,
    KQuantSwitchLinear,
)

from .native_fp import NATIVE_FP_CODECS, NATIVE_FP_GEOMETRY
from .envflags import env_int
from .transforms import qk_permute_wire

_SWITCH_TYPES = None


def switch_layer_types() -> tuple[tuple[type, ...], tuple[type, ...]]:
    """(SwitchLinear origins, SwitchGLU origins) for identity checks.

    mlx-vlm 0.6.4 vendored mlx-lm's switch_layers module, so models built
    from mlx-vlm instantiate its copies of these classes - functionally
    identical but distinct types. Every isinstance/type check on MoE expert
    layers must accept either origin. Lazy: importing mlx_vlm here would pull
    the whole VLM stack into plain text-model loads.
    """
    global _SWITCH_TYPES
    if _SWITCH_TYPES is None:
        from mlx_lm.models.switch_layers import SwitchGLU
        lin, glu = (SwitchLinear,), (SwitchGLU,)
        try:
            from mlx_vlm.models import switch_layers as _vlm_sl
            lin += (_vlm_sl.SwitchLinear,)
            glu += (_vlm_sl.SwitchGLU,)
        except ImportError:  # mlx-vlm <= 0.6.3 (uses mlx-lm's) or absent
            pass
        _SWITCH_TYPES = (lin, glu)
    return _SWITCH_TYPES


class NativeFPSwitchLinear(nn.Module):
    """MoE expert linear backed by an MLX native-fp codec (mxfp4 / nvfp4).

    Counterpart to ``KQuantSwitchLinear`` but for the micro-scaling float
    formats: stores MLX's native packed ``weight`` (uint32) + ``scales`` (uint8)
    and dispatches through the stock ``mx.gather_qmm(..., mode=<codec>)`` kernel
    - no kquant extension call. The GGUF wire bytes are de-interleaved into this
    layout at load (see ``native_fp.repack_native_fp_weights``).
    """

    def __init__(self, num_experts: int, output_dims: int, input_dims: int,
                 bias: bool, codec: str):
        super().__init__()
        gs, bits, _, _ = NATIVE_FP_GEOMETRY[codec]
        self.group_size = gs
        self.bits = bits
        self.mode = codec
        packed_per_row = input_dims * bits // 32       # uint32 words per row
        scales_per_row = input_dims // gs              # one scale per group
        self.weight = mx.zeros((num_experts, output_dims, packed_per_row),
                               dtype=mx.uint32)
        self.scales = mx.zeros((num_experts, output_dims, scales_per_row),
                               dtype=mx.uint8)
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x, self["weight"], self["scales"],
            rhs_indices=indices, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
            sorted_indices=sorted_indices)
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def _extra_repr(self):
        n, m, w = self.weight.shape
        return (f"num_experts={n}, output_dims={m}, packed_per_row={w}, "
                f"mode={self.mode}")


# Decode routes through kq.moe_glu_gather / kq.gather_qmv_bias when True (see
# install_fused_moe_glu); module-level so A/B tools can flip it per call.
_FUSED_MOE_ENABLED = True

# Block-level fusion (shared expert folded into the gather + routing mix
# folded into the down projection) on top of _FUSED_MOE_ENABLED; separate so
# A/B tools can isolate the block-level win from the plain fused gathers.
_FUSED_MOE_BLOCK_ENABLED = True

# Router fusion (one matvec on the concatenated router + shared-gate weight,
# then kq.moe_router_topk for softmax/top-k/norm/sigmoid in one dispatch) on
# top of _FUSED_MOE_BLOCK_ENABLED; separate for A/B attribution.
_FUSED_MOE_ROUTER_ENABLED = True

# Decoder-layer glue fusion (fused residual+rmsnorm dispatch cuts around the
# gemma-4 layer body) on top of _FUSED_MOE_ENABLED; separate for A/B
# attribution.
_FUSED_MOE_GLUE_ENABLED = True


def _kq_fused_device_ok(*mods) -> bool:
    """The fused kq decode kernels are Metal-only. A fused branch must fall
    back to the stock path (which has CPU implementations) when the call has
    to honor a CPU placement: any of ``mods`` is pinned to the CPU stream by
    the streaming offload (``_kq_cpu_only``), or the call is already running
    with a CPU default device (``--stream-cpu``, or inside the offload
    wrapper's ``mx.stream(mx.cpu)`` scope). Without this, ``--stream-cpu``
    decode on an over-RAM model dies at the first token with "[mlx_kquant.*] has no
    CPU implementation" and in-RAM placement overrides silently run on the
    GPU."""
    for m in mods:
        if getattr(m, "_kq_cpu_only", False):
            return False
    return mx.default_device() != mx.cpu


# Fused-MoE decode: capability probe


class _FusedMoeCaps:
    """One install_fused_moe_glu run's view of the fused kq ops: which ops
    the installed mlx-kquant build ships, which codecs they cover, and the
    install-time env gates. Factories bake a probe into each fused subclass,
    so a subclass never outlives the capabilities it was built against."""

    def __init__(self):
        import mlx_kquant as kq

        self.kq = kq
        self.enabled = os.environ.get("GMLX_FUSED_MOE", "1") != "0"
        self.has_base = (hasattr(kq, "moe_glu_gather")
                         and hasattr(kq, "gather_qmv_bias"))
        self.has_kq_fused = hasattr(kq, "moe_glu_gather_kq")
        _, self.glu_types = switch_layer_types()

        codecs = (
            "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0",
            "q4_0", "q4_1", "q5_0", "q5_1",
            "iq4_nl", "iq4_xs", "iq3_s", "iq3_xxs",
            "iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m",
        )
        # Native-fp wire codecs joined the fused family with the capability
        # query itself; sniff it so older kq builds keep them unfused.
        kq_has_glu = getattr(kq, "codec_has_moe_glu", None)
        if kq_has_glu is not None:
            codecs += tuple(c for c in ("mxfp4", "nvfp4") if kq_has_glu(c))
        self.kq_fused_codecs = codecs
        # Shared-expert codecs allowed to differ from the expert stacks (the
        # only mixed combos with kernels); needs the shexp_kquant_type-aware
        # ops.
        mix_doc = getattr(
            getattr(kq, "gather_qmv_mix_kq", None), "__doc__", "") or ""
        self.shexp_upcast = (
            ("q6_k", "q8_0") if "shexp_kquant_type" in mix_doc else ())
        # Same-release proxy for the K % 256 dispatch fallback (kq routes
        # tuned q8_0 with K % 256 != 0 to the generic q8_0_ext kernels);
        # older kq builds need the strict 256 geometry.
        self.has_mix_ns = hasattr(kq, "gather_qmv_mix_ns_kq")
        # Fused residual+rmsnorm glue kernels (gemma-4 decoder-layer seam).
        self.has_norm_fused = (
            hasattr(kq, "add_rmsnorm")
            and hasattr(kq, "rmsnorm_multi3")
            and hasattr(kq, "rmsnorm2_add")
        )
        # Packed-mxfp4 down gather with the routing mix + bias folded in
        # (gpt-oss MLPBlock-level fusion).
        self.has_mix_bias = (
            self.has_base and hasattr(kq, "gather_qmv_mix_bias"))
        # silu_limit (deepseek-v4 LimitedSwiGLU) shipped with a `limit` kwarg
        # on moe_glu_gather_kq; the docstring mention is the same-release
        # sniff the mixed-shexp combos use (older kq builds simply keep V4
        # unfused). swiglu_clamp (gpt-oss clamped SwiGLU) shipped with the
        # biased kernels (gate_bias/up_bias/alpha kwargs); same sniff.
        glu_doc = (getattr(kq.moe_glu_gather_kq, "__doc__", "") or ""
                   if self.has_kq_fused else "")
        self.has_silu_limit = self.has_kq_fused and "silu_limit" in glu_doc
        self.has_swiglu_clamp = (
            self.has_kq_fused and "swiglu_clamp" in glu_doc)
        self.router_ok = (
            os.environ.get("GMLX_FUSED_MOE_ROUTER", "1") != "0"
            and hasattr(kq, "moe_router_topk")
        )
        router_doc = (
            getattr(getattr(kq, "moe_router_topk", None), "__doc__", "") or "")
        self.router_pes_ok = self.router_ok and "per_expert_scale" in router_doc
        # quoted: the pre-sigmoid docstring already says "sigmoid" for the
        # shared-gate column
        self.router_sigmoid_ok = self.router_ok and '"sigmoid"' in router_doc
        self.block_env_on = (
            os.environ.get("GMLX_FUSED_MOE_BLOCK", "1") != "0")
        self.glue_env_on = (
            os.environ.get("GMLX_FUSED_MOE_GLUE", "1") != "0")
        # Glue-kernel bitmask: bit0 = post-attn add_rmsnorm, bit1 =
        # rmsnorm_multi3, bit2 = rmsnorm2_add, bit3 = final
        # add_rmsnorm(+scale). Unset bits use the stock composition of that
        # pattern inside the otherwise-fused layer body. Default 0: at B=1
        # decode every fused glue kernel measured a small E2E loss on
        # gemma-a4b despite winning isolated (in-situ exec of
        # one-threadgroup-per-row kernels runs slower than the stock chains;
        # busy-fraction rises while tok/s drops) -- the win is the layer
        # restructure itself. Revisit bits at t > 1 (MTP verify, batched
        # decode), where per-row threadgroups fill the GPU.
        self.glue_parts = env_int("GMLX_FUSED_MOE_GLUE_PARTS", 0)


def _swap_class(m, cache, factory, caps) -> None:
    """Swap ``m`` onto the fused subclass of its class, creating it on first
    sight. ``cache`` is per install call: each subclass bakes in that
    install's capability probe, and class identity stays one layer deep
    (``type(m)`` of an already-swapped instance is never a fusable base)."""
    base = type(m)
    sub = cache.get(base)
    if sub is None:
        sub = factory(base, caps)
        cache[base] = sub
    m.__class__ = sub


def _kq_wire_k(w, codec):
    """Logical K of a wire-byte tensor's last dim under `codec`
    (-1 when the byte width is not a whole number of blocks)."""
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    wpb, bpb = GGML_QUANT_SIZES[GGMLQuantizationType[codec.upper()]]
    return (w.shape[-1] // bpb) * wpb if w.shape[-1] % bpb == 0 else -1


def _kq_dense(proj, codecs):
    return (isinstance(proj, KQuantLinear)
            and proj.kquant_type in codecs and "bias" not in proj)


def _kq_act_kind(m, caps):
    """Fused-kernel act string for a SwitchGLU's activation, else None.
    Only plain act(gate) * up semantics map onto the kernel epilogue."""
    _has_silu_limit = caps.has_silu_limit
    _has_swiglu_clamp = caps.has_swiglu_clamp
    act = getattr(m, "activation", None)
    if act is None:
        return None
    name, mod = type(act).__name__, type(act).__module__
    if name == "SwiGLU" and "switch_layers" in mod:
        return "silu"
    if name == "SwiGLU" and "gpt_oss" in mod:
        # clamped SwiGLU: biases fold into the kernel, so it needs the
        # biased fused family (mxfp4/nvfp4 only, checked in _eligible_kq)
        if _has_swiglu_clamp and os.environ.get(
                "GMLX_FUSED_MOE_SWIGLU_CLAMP", "1") != "0":
            return "swiglu_clamp"
        return None
    if name == "GeGLU" and "gemma4" in mod:
        return "gelu"  # nn.gelu_approx == the kernel's tanh-approx gelu
    if name == "LimitedSwiGLU" and "deepseek_v4" in mod:
        limit = float(getattr(act, "limit", 0.0) or 0.0)
        if limit <= 0.0:
            return "silu"  # limit<=0 degenerates to plain SwiGLU
        if _has_silu_limit and os.environ.get(
                "GMLX_FUSED_MOE_SILU_LIMIT", "1") != "0":
            return "silu_limit"
        return None
    return None


def _build_router_cat(m):
    """One [E + 1, D] float weight = router rows + the shared-gate row,
    so routing is a single matvec + kq.moe_router_topk. Built lazily on
    the block's first fused call (install time still has load_weights
    placeholders) and kept out of the parameter tree (raw attribute) --
    it duplicates two tiny weights. Returns False when ineligible."""
    gw = getattr(getattr(m, "gate", None), "weight", None)
    sgw = getattr(getattr(m, "shared_expert_gate", None), "weight", None)
    if (
        not isinstance(gw, mx.array) or not isinstance(sgw, mx.array)
        or gw.ndim != 2 or sgw.ndim != 2 or sgw.shape[0] != 1
        or gw.shape[1] != sgw.shape[1]
        or not mx.issubdtype(gw.dtype, mx.floating)
        or not mx.issubdtype(sgw.dtype, mx.floating)
        or gw.shape[0] > 1024 or m.top_k > 16
    ):
        return False
    cat = mx.concatenate([gw, sgw.astype(gw.dtype)], axis=0)
    mx.eval(cat)
    return cat


# Regime 1: gpt-oss packed-mxfp4 SwitchGLU (biased NativeFPSwitchLinear)


def _eligible_nativefp(m, caps):
    _GLU_TYPES = caps.glu_types
    if type(m) not in _GLU_TYPES:  # skip already-wrapped instances
        return False
    act = getattr(m, "activation", None)
    if act is None or type(act).__name__ != "SwiGLU":
        return False
    if "gpt_oss" not in type(act).__module__:
        return False  # only the clamped-SwiGLU semantics are fused
    for proj in (m.gate_proj, m.up_proj, m.down_proj):
        if not isinstance(proj, NativeFPSwitchLinear):
            return False
        if proj.mode != "mxfp4" or "bias" not in proj:
            return False
    return True


def _make_fused_nativefp(base_cls, caps):
    kq = caps.kq

    class _FusedNativeFPSwitchGLU(base_cls):
        def __call__(self, x, indices):
            if (
                _FUSED_MOE_ENABLED
                and indices.size < 64
                and x.dtype in (mx.bfloat16, mx.float16)
                and not self.training
                and _kq_fused_device_ok(self)
            ):
                d_in = x.shape[-1]
                t = x.size // d_in
                k = indices.shape[-1]
                gate, up, down = (
                    self.gate_proj, self.up_proj, self.down_proj)
                idx = indices.reshape(t, k)
                h = kq.moe_glu_gather(
                    x.reshape(t, d_in),
                    gate.weight, gate.scales, gate.bias,
                    up.weight, up.scales, up.bias,
                    idx, self._kq_glu_alpha, self._kq_glu_limit)
                y = kq.gather_qmv_bias(
                    h, down.weight, down.scales, down.bias, idx)
                return y.reshape(*indices.shape, y.shape[-1])
            return super().__call__(x, indices)

    _FusedNativeFPSwitchGLU.__name__ = "_FusedNativeFPSwitchGLU"
    return _FusedNativeFPSwitchGLU


def _install_nativefp_fusion(model, caps) -> int:
    classes: dict = {}
    n = 0
    for _, m in model.named_modules():
        if not _eligible_nativefp(m, caps):
            continue
        # gpt-oss clamped-SwiGLU constants (swiglu() defaults in mlx-lm).
        object.__setattr__(m, "_kq_glu_alpha", 1.702)
        object.__setattr__(m, "_kq_glu_limit", 7.0)
        _swap_class(m, classes, _make_fused_nativefp, caps)
        n += 1
    return n


# Regime 1b: gpt-oss MLPBlock (router + packed-mxfp4 SwitchGLU + mix)


def _eligible_gptoss_mlp(m, caps):
    """gpt-oss ``MLPBlock``: router Linear + packed-mxfp4 SwitchGLU +
    softmax-weighted (y * w).sum(-2) mix. The mix and expert bias fold into
    the down gather (kq.gather_qmv_mix_bias) and the top-k/softmax epilogue
    into kq.moe_router_topk; gate/up ride kq.moe_glu_gather as in the plain
    SwitchGLU swap. Already-swapped instances fail the module check (the
    fused subclass lives in gmlx.modules)."""
    if not caps.has_mix_bias or not caps.router_ok:
        return False
    if type(m).__name__ != "MLPBlock" or "gpt_oss" not in type(m).__module__:
        return False
    sw = getattr(m, "experts", None)
    if sw is None or not isinstance(sw, caps.glu_types):  # subclasses OK
        return False
    act = getattr(sw, "activation", None)
    if act is None or type(act).__name__ != "SwiGLU":
        return False
    if "gpt_oss" not in type(act).__module__:
        return False
    for proj in (sw.gate_proj, sw.up_proj, sw.down_proj):
        if not isinstance(proj, NativeFPSwitchLinear):
            return False
        if proj.mode != "mxfp4" or "bias" not in proj:
            return False
    if not isinstance(getattr(m, "router", None), nn.Linear):
        return False
    k = getattr(m, "num_experts_per_tok", 0)
    e = getattr(m, "num_local_experts", 0)
    return 0 < k <= 16 and k <= e <= 1024


def _make_fused_gptoss_mlp(base_cls, caps):
    kq = caps.kq

    class _FusedGptOssMLPBlock(base_cls):
        """gpt-oss fused MLPBlock decode: router matvec stays stock; the
        top-k + softmax epilogue collapses into kq.moe_router_topk
        (norm_topk_prob softmax over the selected raw logits == the model's
        softmax(mlx_topk(g))), and the routing mix + expert bias fold into
        the down gather in f32 (kq.gather_qmv_mix_bias). The f32 mix sits
        closer to the f32 reference than stock's bf16 per-slot rounding, so
        token tie-flips are possible."""

        def __call__(self, x):
            sw = self.experts
            d = x.shape[-1]
            t = x.size // d
            k = self.num_experts_per_tok
            if not (
                _FUSED_MOE_ENABLED
                and _FUSED_MOE_BLOCK_ENABLED
                and _FUSED_MOE_ROUTER_ENABLED
                and t * k < 64
                and x.dtype in (mx.bfloat16, mx.float16)
                and not self.training
                and self.sharding_group is None
                and _kq_fused_device_ok(self, sw)
            ):
                return super().__call__(x)
            xt = x.reshape(t, d)
            g = self.router(xt)
            idx, scores = kq.moe_router_topk(
                g, k, norm_topk_prob=True, shared_gate=False)
            h = kq.moe_glu_gather(
                xt,
                sw.gate_proj.weight, sw.gate_proj.scales, sw.gate_proj.bias,
                sw.up_proj.weight, sw.up_proj.scales, sw.up_proj.bias,
                idx,
                getattr(sw, "_kq_glu_alpha", 1.702),
                getattr(sw, "_kq_glu_limit", 7.0))
            y = kq.gather_qmv_mix_bias(
                h, sw.down_proj.weight, sw.down_proj.scales,
                sw.down_proj.bias, idx, scores)
            return y.reshape(*x.shape[:-1], y.shape[-1])

    _FusedGptOssMLPBlock.__name__ = "_FusedGptOssMLPBlock"
    return _FusedGptOssMLPBlock


def _install_gptoss_block_fusion(model, caps) -> int:
    if not caps.block_env_on:
        return 0
    classes: dict = {}
    n = 0
    for _, m in model.named_modules():
        if not _eligible_gptoss_mlp(m, caps):
            continue
        _swap_class(m, classes, _make_fused_gptoss_mlp, caps)
        n += 1
    return n


# Regime 2: K-quant SwitchGLU (per-codec fused act(gate) * up)


def _eligible_kquant_glu(m, caps):
    _has_kq_fused = caps.has_kq_fused
    _GLU_TYPES = caps.glu_types
    _KQ_FUSED_CODECS = caps.kq_fused_codecs
    _has_mix_ns = caps.has_mix_ns
    if not _has_kq_fused or type(m) not in _GLU_TYPES:
        return False
    act_kind = _kq_act_kind(m, caps)
    if act_kind is None:
        return False
    # swiglu_clamp folds the expert biases into the kernels (all three
    # projections biased, gpt-oss shape); every other act is bias-free.
    want_bias = act_kind == "swiglu_clamp"
    for proj in (m.gate_proj, m.up_proj, m.down_proj):
        if not isinstance(proj, KQuantSwitchLinear):
            return False
        if proj.kquant_type not in _KQ_FUSED_CODECS:
            return False
        if ("bias" in proj) != want_bias:
            return False
        if want_bias and proj.kquant_type not in ("mxfp4", "nvfp4"):
            return False  # biased kernels exist for the fp4 codecs only
        # wire bytes must be whole blocks (any K % block, e.g. gemma's
        # inter=704 down; the q8_0 K % 256 gap dispatches q8_0_ext) and
        # the kernel grid needs N % 8
        k_wire = _kq_wire_k(proj.weight, proj.kquant_type)
        if k_wire <= 0 or (not _has_mix_ns and k_wire % 256):
            return False
        if proj.weight.shape[1] % 8:
            return False
    # one dispatch fuses gate+up: they must share a codec (down may differ)
    return m.gate_proj.kquant_type == m.up_proj.kquant_type


def _make_fused_kquant(base_cls, caps):
    kq = caps.kq
    _has_mix_ns = caps.has_mix_ns

    class _FusedKQuantSwitchGLU(base_cls):
        """K-quant fused decode path: act(gate) * up SwitchGLU whose
        projections are un-biased KQuantSwitchLinear wire-byte stacks
        (silu = qwen-style SwiGLU, gelu = gemma-style GeGLU). The fused
        epilogue runs in f32 (the stock path rounds gate/up to the
        activation dtype first), so outputs sit closer to the f32
        reference than stock; down projection is bit-exact vs the stock
        gather. Geometry (wire-K / N % 8) is validated at install time.

        With `scores`, the routed weighted sum folds into the down gather
        (gather_qmv_mix_ns_kq, f32 accumulate) and the output comes back
        already mixed over the expert axis; the ineligible path applies
        the same sum python-side.

        With `scores` and a `_kq_shexp_mod` stamp (install_hyv3_shexp_fold),
        the shared expert rides the gathers as one extra slot with mix
        weight 1. Contract with the stamping caller: a mixed return
        consumed the shared expert; the stamped fallback returns unmixed
        (skipping the epilogue below) and the caller mixes + adds it."""

        _kq_mix_scores = _has_mix_ns

        def __call__(self, x, indices, scores=None):
            if (
                _FUSED_MOE_ENABLED
                and indices.size < 64
                and x.dtype in (mx.bfloat16, mx.float16)
                and not self.training
                and _kq_fused_device_ok(self)
            ):
                d_in = x.shape[-1]
                t = x.size // d_in
                k = indices.shape[-1]
                gate, up, down = (
                    self.gate_proj, self.up_proj, self.down_proj)
                idx = indices.reshape(t, k)
                act = getattr(self, "_kq_glu_act", "silu")
                # extra kwargs only where the act needs them so silu/gelu
                # calls stay byte-identical against older kq builds.
                akw = {"act": act}
                if act == "silu_limit":
                    akw["limit"] = self._kq_glu_limit
                elif act == "swiglu_clamp":
                    akw["limit"] = self._kq_glu_limit
                    akw["alpha"] = self._kq_glu_alpha
                    akw["gate_bias"] = self._kq_gb32
                    akw["up_bias"] = self._kq_ub32
                se = (getattr(self, "_kq_shexp_mod", None)
                      if scores is not None else None)
                if se is not None:
                    skw = ({"shexp_kquant_type": se.gate_proj.kquant_type}
                           if se.gate_proj.kquant_type != gate.kquant_type
                           else {})
                    h = kq.moe_glu_gather_shexp_kq(
                        x.reshape(t, d_in), gate.weight, up.weight,
                        se.gate_proj.weight, se.up_proj.weight,
                        gate.kquant_type, idx, **akw, **skw)
                    sc = scores.reshape(t, k)
                    sc = mx.concatenate(
                        [sc, mx.ones((t, 1), dtype=sc.dtype)], axis=-1)
                    skw = ({"shexp_kquant_type": se.down_proj.kquant_type}
                           if se.down_proj.kquant_type != down.kquant_type
                           else {})
                    y = kq.gather_qmv_mix_kq(
                        h, down.weight, se.down_proj.weight,
                        down.kquant_type, idx, sc, **skw)
                    return y.reshape(*indices.shape[:-1], y.shape[-1])
                h = kq.moe_glu_gather_kq(
                    x.reshape(t, d_in),
                    gate.weight, up.weight,
                    gate.kquant_type, idx, **akw)
                if (scores is not None and _has_mix_ns
                        and "bias" not in down):
                    y = kq.gather_qmv_mix_ns_kq(
                        h, down.weight, down.kquant_type, idx,
                        scores.reshape(t, k))
                    return y.reshape(*indices.shape[:-1], y.shape[-1])
                dkw = {"bias": self._kq_db32} if "bias" in down else {}
                y = kq.gather_qmv_kq(
                    h, down.weight, down.kquant_type, idx, **dkw)
                y = y.reshape(*indices.shape, y.shape[-1])
            else:
                y = super().__call__(x, indices)
            if scores is not None and y.ndim == scores.ndim + 1:
                if getattr(self, "_kq_shexp_mod", None) is not None:
                    return y  # unmixed: caller mixes + adds its shexp
                y = (y * scores[..., None].astype(y.dtype)).sum(-2)
            return y

    _FusedKQuantSwitchGLU.__name__ = "_FusedKQuantSwitchGLU"
    return _FusedKQuantSwitchGLU


def _install_kquant_glu_fusion(model, caps) -> int:
    classes: dict = {}
    n = 0
    for _, m in model.named_modules():
        if not _eligible_kquant_glu(m, caps):
            continue
        act_kind = _kq_act_kind(m, caps)
        object.__setattr__(m, "_kq_glu_act", act_kind)
        if act_kind == "silu_limit":
            object.__setattr__(
                m, "_kq_glu_limit", float(m.activation.limit))
        elif act_kind == "swiglu_clamp":
            # gpt-oss constants (swiglu() defaults in mlx-lm).
            object.__setattr__(m, "_kq_glu_alpha", 1.702)
            object.__setattr__(m, "_kq_glu_limit", 7.0)
            # f32 bias copies cached once: the ops astype-and-copy the
            # per-expert biases on every call otherwise (~26 MB and a
            # few percent of decode at gpt-oss size).
            for attr, proj in (("_kq_gb32", m.gate_proj),
                               ("_kq_ub32", m.up_proj),
                               ("_kq_db32", m.down_proj)):
                object.__setattr__(
                    m, attr, proj.bias.astype(mx.float32))
            mx.eval(m._kq_gb32, m._kq_ub32, m._kq_db32)
        _swap_class(m, classes, _make_fused_kquant, caps)
        n += 1
    return n


# Regime 3: qwen3-next-shaped MoE block (router + SwitchGLU + shared expert)


def _eligible_kq_block(m, caps):
    """qwen3-next-shaped MoE block: router + SwitchGLU + shared expert
    with a sigmoid gate. Fusable when the shared expert's projections are
    K-quant rows shape-matched to the expert stacks (same codec, or a
    q6_k/q8_0 upcast), so it rides the gather as one extra slot and the
    routing mix folds into the down projection (kq.moe_glu_gather_shexp_kq
    + kq.gather_qmv_mix_kq)."""
    kq = caps.kq
    _has_kq_fused = caps.has_kq_fused
    _GLU_TYPES = caps.glu_types
    _KQ_FUSED_CODECS = caps.kq_fused_codecs
    _KQ_SHEXP_UPCAST = caps.shexp_upcast
    if type(m).__name__ == "_FusedKQuantMoeBlock":
        return False  # already swapped; keep class identity one layer deep
    if not _has_kq_fused or not hasattr(kq, "moe_glu_gather_shexp_kq"):
        return False
    for attr in ("gate", "switch_mlp", "shared_expert",
                 "shared_expert_gate", "top_k", "norm_topk_prob"):
        if not hasattr(m, attr):
            return False
    sw = m.switch_mlp
    if not isinstance(sw, _GLU_TYPES):  # subclasses (incl. fused) OK
        return False
    act = getattr(sw, "activation", None)
    if act is None or type(act).__name__ != "SwiGLU":
        return False
    if "switch_layers" not in type(act).__module__:
        return False
    for proj in (sw.gate_proj, sw.up_proj, sw.down_proj):
        if not isinstance(proj, KQuantSwitchLinear):
            return False
        if proj.kquant_type not in _KQ_FUSED_CODECS or "bias" in proj:
            return False
    if sw.gate_proj.kquant_type != sw.up_proj.kquant_type:
        return False
    # kernel geometry: K of both matvecs % 256, N (= down K) % 256
    d_in = _kq_wire_k(sw.gate_proj.weight, sw.gate_proj.kquant_type)
    inter = sw.gate_proj.weight.shape[1]
    if d_in <= 0 or d_in % 256 or inter % 256:
        return False
    se = m.shared_expert
    for attr in ("gate_proj", "up_proj", "down_proj"):
        if not hasattr(se, attr):
            return False
    # plain silu(gate) * up shared expert only (qwen3-next MLP shape)
    if hasattr(se, "activation"):
        return False
    # the GLU gather runs both shexp slots with one codec
    if se.gate_proj.kquant_type != se.up_proj.kquant_type:
        return False
    for proj, stack in ((se.gate_proj, sw.gate_proj),
                        (se.up_proj, sw.up_proj),
                        (se.down_proj, sw.down_proj)):
        if not _kq_dense(proj, (stack.kquant_type,) + _KQ_SHEXP_UPCAST):
            return False
        # shape-matched in each side's own codec wire bytes
        if proj.weight.shape[0] != stack.weight.shape[1]:
            return False
        if (_kq_wire_k(proj.weight, proj.kquant_type)
                != _kq_wire_k(stack.weight, stack.kquant_type)):
            return False
    return True


def _make_fused_block(base_cls, caps):
    kq = caps.kq

    class _FusedKQuantMoeBlock(base_cls):
        """Whole-MoE-block fused decode: router glue stays stock; the
        shared expert rides the GLU gather as slot R and the routing mix
        (scores-weighted sum + sigmoid-gated shared add) runs inside the
        down gather in f32 -- closer to the f32 reference than stock's
        bf16 per-slot rounding, so token tie-flips are possible."""

        def __call__(self, x):
            d = x.shape[-1]
            t = x.size // d
            expert_ctl = (
                getattr(self, "_kq_expert_mass", None) is not None
                or getattr(self, "_kq_expert_probe", None) is not None
            )
            if not (
                _FUSED_MOE_ENABLED
                and _FUSED_MOE_BLOCK_ENABLED
                and t * self.top_k < 64
                and x.dtype in (mx.bfloat16, mx.float16)
                and not self.training
                and getattr(self, "sharding_group", None) is None
                and _kq_fused_device_ok(self, self.switch_mlp)
            ):
                if expert_ctl:
                    # Stock forward with the fan-out hook at the selection
                    # seam (the eligibility check asserted this shape).
                    from .moe_experts import qwen3_next_moe_forward

                    return qwen3_next_moe_forward(self, x)
                return super().__call__(x)
            xf = x.reshape(t, d)
            rc = getattr(self, "_kq_router_cat", None)
            if (rc is None and _FUSED_MOE_ROUTER_ENABLED
                    and getattr(self, "_kq_router_want", False)):
                # built on first use: at install time the router weights
                # are still load_weights placeholders
                rc = _build_router_cat(self)
                object.__setattr__(self, "_kq_router_cat", rc)
            if (rc is not None and rc is not False
                    and _FUSED_MOE_ROUTER_ENABLED):
                logits = xf.astype(rc.dtype) @ rc.T
                inds, sc = kq.moe_router_topk(
                    logits, self.top_k, bool(self.norm_topk_prob))
            else:
                gates = mx.softmax(
                    self.gate(xf), axis=-1, precise=True)
                k = self.top_k
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                if self.norm_topk_prob:
                    scores = scores / scores.sum(axis=-1, keepdims=True)
                shared_g = mx.sigmoid(self.shared_expert_gate(xf))
                sc = mx.concatenate(
                    [scores, shared_g.astype(scores.dtype)], axis=-1)
            if expert_ctl:
                # Adaptive fan-out on the routed slots only; the trailing
                # shared-gate mix weight rides along untouched.
                from .moe_experts import _apply_expert_controls

                k = self.top_k
                inds, routed = _apply_expert_controls(
                    self, inds, sc[..., :k])
                sc = mx.concatenate([routed, sc[..., k:]], axis=-1)
            sw, se = self.switch_mlp, self.shared_expert
            skw = {}
            if se.gate_proj.kquant_type != sw.gate_proj.kquant_type:
                skw = {"shexp_kquant_type": se.gate_proj.kquant_type}
            h = kq.moe_glu_gather_shexp_kq(
                xf, sw.gate_proj.weight, sw.up_proj.weight,
                se.gate_proj.weight, se.up_proj.weight,
                sw.gate_proj.kquant_type, inds, act="silu", **skw)
            skw = {}
            if se.down_proj.kquant_type != sw.down_proj.kquant_type:
                skw = {"shexp_kquant_type": se.down_proj.kquant_type}
            y = kq.gather_qmv_mix_kq(
                h, sw.down_proj.weight, se.down_proj.weight,
                sw.down_proj.kquant_type, inds, sc, **skw)
            return y.reshape(x.shape)

    _FusedKQuantMoeBlock.__name__ = "_FusedKQuantMoeBlock"
    return _FusedKQuantMoeBlock


def _install_kq_block_fusion(model, caps) -> int:
    if not caps.block_env_on:
        return 0
    classes: dict = {}
    n = 0
    for _, m in model.named_modules():
        if not _eligible_kq_block(m, caps):
            continue
        _swap_class(m, classes, _make_fused_block, caps)
        object.__setattr__(m, "_kq_router_want", caps.router_ok)
        n += 1
    return n


# Regime 3b: hy_v3 MoE (sigmoid+bias router, ungated shared expert).
# Stamps the fused SwitchGLU instead of swapping the block class, so the
# expert-streaming offload wrapper stays in the call path.


def _eligible_hyv3_shexp(m, caps):
    """hy_v3 MoE whose ungated shared expert can ride the fused SwitchGLU
    gathers as one extra slot with mix weight 1 (moe_glu_gather_shexp_kq +
    gather_qmv_mix_kq): K-quant projections shape-matched to the expert
    stacks (same codec, or a q6_k/q8_0 upcast), SwitchGLU already swapped
    to the scores-taking fused class."""
    kq = caps.kq
    _KQ_SHEXP_UPCAST = caps.shexp_upcast
    if not hasattr(kq, "moe_glu_gather_shexp_kq"):
        return False
    if not hasattr(kq, "gather_qmv_mix_kq"):
        return False
    for attr in ("router", "switch_mlp", "shared_mlp", "fp32_combine"):
        if not hasattr(m, attr):
            return False
    if hasattr(m, "shared_expert_gate"):  # qwen3-next shape: regime 3
        return False
    se = m.shared_mlp
    if se is None:
        return False
    sw = m.switch_mlp
    if not getattr(sw, "_kq_mix_scores", False):
        return False
    if getattr(sw, "_kq_glu_act", None) != "silu":
        return False
    # kernel geometry: K of both matvecs % 256, N (= down K) % 256
    d_in = _kq_wire_k(sw.gate_proj.weight, sw.gate_proj.kquant_type)
    inter = sw.gate_proj.weight.shape[1]
    if d_in <= 0 or d_in % 256 or inter % 256:
        return False
    for attr in ("gate_proj", "up_proj", "down_proj"):
        if not hasattr(se, attr):
            return False
    # plain silu(gate) * up shared expert only (hy_v3 MLP shape)
    if hasattr(se, "activation"):
        return False
    # the GLU gather runs both shexp slots with one codec
    if se.gate_proj.kquant_type != se.up_proj.kquant_type:
        return False
    for proj, stack in ((se.gate_proj, sw.gate_proj),
                        (se.up_proj, sw.up_proj),
                        (se.down_proj, sw.down_proj)):
        if not _kq_dense(proj, (stack.kquant_type,) + _KQ_SHEXP_UPCAST):
            return False
        # shape-matched in each side's own codec wire bytes
        if proj.weight.shape[0] != stack.weight.shape[1]:
            return False
        if (_kq_wire_k(proj.weight, proj.kquant_type)
                != _kq_wire_k(stack.weight, stack.kquant_type)):
            return False
    return True


def install_hyv3_shexp_fold(model) -> int:
    """Stamp eligible hy_v3 MoE blocks' fused SwitchGLUs with their shared
    expert (``_kq_shexp_mod``, a module ref: install runs before
    load_weights). Call after install_fused_moe_glu. Disable with
    GMLX_FUSED_MOE_BLOCK=0. Returns the number of blocks stamped."""
    caps = _FusedMoeCaps()
    if not caps.enabled or not caps.has_base or not caps.block_env_on:
        return 0
    n = 0
    for _, m in model.named_modules():
        if not _eligible_hyv3_shexp(m, caps):
            continue
        object.__setattr__(m.switch_mlp, "_kq_shexp_mod", m.shared_mlp)
        n += 1
    return n


def _eligible_hyv3_router(m, caps):
    """hy_v3 MoEGate: sigmoid scoring + selection bias + renorm + scale is
    exactly kq.moe_router_topk(scoring="sigmoid")."""
    if not caps.router_sigmoid_ok:
        return False
    if type(m).__name__ != "MoEGate":
        return False
    if "hy_v3" not in type(m).__module__:
        return False
    for attr in ("gate", "expert_bias", "top_k", "norm_topk_prob",
                 "routed_scaling_factor"):
        if not hasattr(m, attr):
            return False
    w = getattr(m.gate, "weight", None)
    if w is None or w.ndim != 2:
        return False
    return w.shape[0] <= 1024 and int(m.top_k) <= 16


def _make_fused_hyv3_router(base_cls, caps):
    kq = caps.kq

    class _FusedHyV3Router(base_cls):
        """hy_v3 router epilogue in one dispatch
        (kq.moe_router_topk(scoring="sigmoid")): selection ranked by
        sigmoid(logits) + expert_bias, emitted scores unbiased,
        renormalized with the 1e-20 guard, scaled by
        routed_scaling_factor. Decode-shaped calls only; prefill keeps
        the stock compiled epilogue."""

        def __call__(self, x):
            d = x.shape[-1]
            t = x.size // d
            if not (
                _FUSED_MOE_ENABLED
                and _FUSED_MOE_ROUTER_ENABLED
                and t < 64
                and not self.training
                and _kq_fused_device_ok(self)
            ):
                return super().__call__(x)
            b32 = getattr(self, "_kq_bias32", None)
            if b32 is None:
                # built on first use: at install time expert_bias is a
                # load_weights placeholder
                b32 = self.expert_bias.astype(mx.float32)
                mx.eval(b32)
                object.__setattr__(self, "_kq_bias32", b32)
            k = int(self.top_k)
            inds, sc = kq.moe_router_topk(
                self.gate(x).reshape(t, -1),
                k,
                bool(self.norm_topk_prob) and k > 1,
                shared_gate=False,
                bias=b32,
                scoring="sigmoid",
                scale=float(self.routed_scaling_factor),
            )
            shp = x.shape[:-1]
            return inds.reshape(*shp, k), sc.reshape(*shp, k)

    _FusedHyV3Router.__name__ = "_FusedHyV3Router"
    return _FusedHyV3Router


def install_hyv3_router_fuse(model) -> int:
    """Class-swap eligible hy_v3 MoEGates onto the fused sigmoid-router
    epilogue. Disable with GMLX_FUSED_MOE_ROUTER=0. Returns the number of
    routers swapped."""
    caps = _FusedMoeCaps()
    if not caps.enabled or not caps.has_base:
        return 0
    classes: dict = {}
    n = 0
    for _, m in model.named_modules():
        if not _eligible_hyv3_router(m, caps):
            continue
        _swap_class(m, classes, _make_fused_hyv3_router, caps)
        n += 1
    return n


# Regime 4: gemma-4 MoE (Router + Experts modules, GeGLU, no shared expert)


def _eligible_gemma_experts(m, caps):
    """gemma4 ``Experts``: a GeGLU SwitchGLU plus a score-weighted sum,
    no shared expert. The routing mix folds into the down gather
    (kq.gather_qmv_mix_ns_kq); gate/up stay stock, so only the down
    projection needs a fused-codec stack."""
    _has_kq_fused = caps.has_kq_fused
    _GLU_TYPES = caps.glu_types
    _KQ_FUSED_CODECS = caps.kq_fused_codecs
    _has_mix_ns = caps.has_mix_ns
    if not _has_kq_fused or not _has_mix_ns:
        return False
    if type(m).__name__ != "Experts" or "gemma4" not in type(m).__module__:
        return False
    sw = getattr(m, "switch_glu", None)
    if not isinstance(sw, _GLU_TYPES):  # subclasses (incl. fused) OK
        return False
    if _kq_act_kind(sw, caps) != "gelu":
        return False
    down = sw.down_proj
    if not isinstance(down, KQuantSwitchLinear):
        return False
    if down.kquant_type not in _KQ_FUSED_CODECS or "bias" in down:
        return False
    if _kq_wire_k(down.weight, down.kquant_type) <= 0:
        return False
    return down.weight.shape[1] % 8 == 0


def _make_fused_gemma_experts(base_cls, caps):
    kq = caps.kq
    _KQ_FUSED_CODECS = caps.kq_fused_codecs

    class _FusedKQuantGemmaExperts(base_cls):
        """gemma4 fused Experts decode: gate/up/activation stay on the
        stock gathers (the fused GLU gather wins in isolation but loses
        E2E to in-step scheduling on this small-expert geometry); the
        score-weighted (w * y).sum(-2) mix folds into the down gather in
        f32 (kq.gather_qmv_mix_ns_kq). The f32 mix sits closer to the
        f32 reference than stock's bf16 per-slot rounding, so token
        tie-flips are possible.

        Exception: inside the wide-NX window (few enough output rows
        that mlx-kquant picks the 16-lane GLU kernels, t * k * N / 8 <
        2048 threadgroups) the fused GLU gather flips to an E2E win
        where the stock gathers hit the tuned q6_k/q8_0 kernel ceiling
        (Q6_K +7.4% d0 / +5.7% @65k, token-exact) -- those codecs route
        gate/up through kq.moe_glu_gather_kq by default. Other codecs
        measure as a wash (q4_k -0.4 +/- 1% over three runs) and stay
        stock unless GMLX_GEMMA_FUSED_GLU=1 forces all eligible
        codecs; =0 kills. Outside the window (MTP verify t >= 3) the
        NX = 8 inversion stands and gate/up stay stock everywhere. The
        variable is read per call so A/B harnesses can flip arms
        in-process."""

        def _kq_glu_ok(self, d):
            ok = getattr(self, "_kq_glu_ok_cached", None)
            if ok is None:
                sw = self.switch_glu
                g, u = sw.gate_proj, sw.up_proj
                ok = (
                    isinstance(g, KQuantSwitchLinear)
                    and isinstance(u, KQuantSwitchLinear)
                    and g.kquant_type == u.kquant_type
                    and g.kquant_type in _KQ_FUSED_CODECS
                    and "bias" not in g and "bias" not in u
                    and g.weight.shape[1] % 8 == 0
                    and d >= 512  # wide kernels want K/16 >= 32 chunks
                    and _kq_wire_k(g.weight, g.kquant_type) == d
                )
                self._kq_glu_ok_cached = ok
            return ok

        def __call__(self, x, top_k_indices, top_k_weights):
            d = x.shape[-1]
            t = x.size // d
            k = top_k_indices.shape[-1]
            if not (
                _FUSED_MOE_ENABLED
                and _FUSED_MOE_BLOCK_ENABLED
                and t * k < 64
                and x.dtype in (mx.bfloat16, mx.float16)
                and not self.training
                and _kq_fused_device_ok(self, self.switch_glu)
            ):
                return super().__call__(x, top_k_indices, top_k_weights)
            sw = self.switch_glu
            idx = top_k_indices.reshape(t, k)
            env = os.environ.get("GMLX_GEMMA_FUSED_GLU", "")
            if (
                (env == "1" or (env != "0" and
                                sw.gate_proj.kquant_type
                                in ("q6_k", "q8_0")))
                and t * k * sw.gate_proj.weight.shape[1] // 8 < 2048
                and self._kq_glu_ok(d)
            ):
                h = kq.moe_glu_gather_kq(
                    x.reshape(t, d),
                    sw.gate_proj.weight, sw.up_proj.weight,
                    sw.gate_proj.kquant_type, idx, act="gelu")
            else:
                xe = mx.expand_dims(x.reshape(t, d), (-2, -3))
                up = sw.up_proj(xe, idx)
                gate = sw.gate_proj(xe, idx)
                h = sw.activation(up, gate).squeeze(-2)
            y = kq.gather_qmv_mix_ns_kq(
                h, sw.down_proj.weight, sw.down_proj.kquant_type,
                idx, top_k_weights.reshape(t, k))
            return y.reshape(*x.shape[:-1], y.shape[-1])

    _FusedKQuantGemmaExperts.__name__ = "_FusedKQuantGemmaExperts"
    return _FusedKQuantGemmaExperts


def _eligible_gemma_router(m, caps):
    """gemma4 ``Router``: rms_norm + proj stay stock; the argpartition /
    take_along_axis / softmax / per-expert-scale epilogue collapses into
    kq.moe_router_topk(shared_gate=False, per_expert_scale=...)."""
    _router_pes_ok = caps.router_pes_ok
    if not _router_pes_ok:
        return False
    if type(m).__name__ != "Router" or "gemma4" not in type(m).__module__:
        return False
    cfg = getattr(m, "config", None)
    if cfg is None:
        return False
    for attr in ("proj", "scale", "eps", "_root_size"):
        if not hasattr(m, attr):
            return False
    if "per_expert_scale" not in m:
        return False
    e = getattr(cfg, "num_experts", None) or 0
    k = getattr(cfg, "top_k_experts", None) or 0
    return 0 < k <= 16 and k <= e <= 1024


def _make_fused_gemma_router(base_cls, caps):
    kq = caps.kq

    class _FusedKQuantGemmaRouter(base_cls):
        """gemma4 fused Router decode: gemma's softmax over the selected
        raw logits equals the full softmax renormalized over the picked
        experts, so norm_topk_prob=True reproduces it exactly; the
        per-expert scale applies to the picked scores in-kernel."""

        def __call__(self, x):
            k = self.config.top_k_experts
            d = x.shape[-1]
            t = x.size // d
            if not (
                _FUSED_MOE_ENABLED
                and _FUSED_MOE_ROUTER_ENABLED
                and t * k < 64
                and not self.training
                and _kq_fused_device_ok(self)
            ):
                return super().__call__(x)
            xn = mx.fast.rms_norm(
                x, self.scale * self._root_size, self.eps)
            inds, sc = kq.moe_router_topk(
                self.proj(xn).reshape(t, -1), k, True,
                shared_gate=False,
                per_expert_scale=self.per_expert_scale)
            shp = x.shape[:-1]
            return inds.reshape(*shp, k), sc.reshape(*shp, k)

    _FusedKQuantGemmaRouter.__name__ = "_FusedKQuantGemmaRouter"
    return _FusedKQuantGemmaRouter


def _eligible_gemma_layer(m, caps):
    """gemma4 ``DecoderLayer``: swap the layer body for the restructured
    form (cached premultiplied router norm weight + in-layer
    kq.moe_router_topk epilogue; optional kq glue kernels via
    _glue_parts). MoE layers need the fused router epilogue -- the
    router's norm runs on the cached weight in the layer body."""
    _has_norm_fused = caps.has_norm_fused
    _router_pes_ok = caps.router_pes_ok
    if not _has_norm_fused:
        return False
    if (type(m).__name__ != "DecoderLayer"
            or "gemma4" not in type(m).__module__):
        return False
    for attr in ("self_attn", "mlp", "input_layernorm"):
        if getattr(m, attr, None) is None:
            return False
    for nm in ("post_attention_layernorm", "pre_feedforward_layernorm",
               "post_feedforward_layernorm"):
        norm = getattr(m, nm, None)
        if (norm is None or getattr(norm, "weight", None) is None
                or not hasattr(norm, "eps")):
            return False
    if not getattr(m, "enable_moe", False):
        return True
    if not _router_pes_ok:
        return False
    for nm in ("post_feedforward_layernorm_1",
               "post_feedforward_layernorm_2",
               "pre_feedforward_layernorm_2"):
        norm = getattr(m, nm, None)
        if (norm is None or getattr(norm, "weight", None) is None
                or not hasattr(norm, "eps")):
            return False
    r = getattr(m, "router", None)
    if r is None or getattr(m, "experts", None) is None:
        return False
    for attr in ("proj", "scale", "eps", "_root_size"):
        if not hasattr(r, attr):
            return False
    if "per_expert_scale" not in r:
        return False
    cfg = getattr(r, "config", None)
    k = getattr(cfg, "top_k_experts", None) or 0
    e = getattr(cfg, "num_experts", None) or 0
    if not (0 < k <= 16 and k <= e <= 1024):
        return False
    # Shared-reduction kernels need matching epsilons.
    if (m.pre_feedforward_layernorm.eps != r.eps
            or m.pre_feedforward_layernorm.eps
            != m.pre_feedforward_layernorm_2.eps):
        return False
    return (m.post_feedforward_layernorm_1.eps
            == m.post_feedforward_layernorm_2.eps)


def _make_fused_gemma_layer(base_cls, caps):
    kq = caps.kq
    _glue_parts = caps.glue_parts

    class _FusedKQuantGemmaDecoderLayer(base_cls):
        """gemma4 fused decoder-layer body. The measured win is the
        RESTRUCTURE: the router's rms_norm weight is premultiplied by
        root_size once and cached (kills a per-call scale mul), and the
        router epilogue runs in-layer via kq.moe_router_topk on that
        cached weight. The kq glue kernels (_glue_parts bits) are
        optional: at B=1 decode each measured a small E2E loss vs the
        stock composition, so they default off; when enabled, glue math
        runs in f32 with one round per tensor, so token tie-flips vs
        stock are possible. The router weight is cached on first fused
        call -- install time is before load_weights, when tensors are
        placeholders."""

        def _kq_glue_setup(self, dt):
            norms = [
                self.post_attention_layernorm,
                self.pre_feedforward_layernorm,
                self.post_feedforward_layernorm,
            ]
            moe = getattr(self, "enable_moe", False)
            if moe:
                norms += [
                    self.post_feedforward_layernorm_1,
                    self.post_feedforward_layernorm_2,
                    self.pre_feedforward_layernorm_2,
                ]
            for norm in norms:
                w = norm.weight
                if w.ndim != 1 or w.dtype != dt:
                    return False
            ls = getattr(self, "layer_scalar", None)
            if ls is not None and (ls.size != 1 or ls.dtype != dt):
                return False
            glue = {"scale": ls}
            if moe:
                r = self.router
                rw = (r.scale * r._root_size).astype(dt)
                if rw.ndim != 1:
                    return False
                mx.eval(rw)
                glue["router_w"] = rw
            return glue

        def __call__(self, x, mask=None, cache=None,
                     per_layer_input=None, shared_kv=None, offset=None):
            d = x.shape[-1]
            t = x.size // d
            glue = getattr(self, "_kq_glue", None)
            if (
                glue is False
                or not _FUSED_MOE_ENABLED
                or not _FUSED_MOE_GLUE_ENABLED
                or per_layer_input is not None
                or t >= 64
                or x.dtype not in (mx.bfloat16, mx.float16)
                or self.training
                or not _kq_fused_device_ok(self)
            ):
                return super().__call__(
                    x, mask, cache, per_layer_input=per_layer_input,
                    shared_kv=shared_kv, offset=offset)
            if glue is None:
                glue = self._kq_glue_setup(x.dtype)
                object.__setattr__(self, "_kq_glue", glue)
                if glue is False:
                    return super().__call__(
                        x, mask, cache, per_layer_input=per_layer_input,
                        shared_kv=shared_kv, offset=offset)

            h, shared_kv, offset = self.self_attn(
                self.input_layernorm(x), mask, cache,
                shared_kv=shared_kv, offset=offset)
            pan = self.post_attention_layernorm
            if _glue_parts & 1:
                res = kq.add_rmsnorm(h, x, pan.weight, pan.eps)
            else:
                res = x + pan(h)

            if getattr(self, "enable_moe", False):
                if _glue_parts & 2:
                    h1_in, xn, h2_in = kq.rmsnorm_multi3(
                        res,
                        self.pre_feedforward_layernorm.weight,
                        glue["router_w"],
                        self.pre_feedforward_layernorm_2.weight,
                        self.pre_feedforward_layernorm.eps)
                else:
                    h1_in = self.pre_feedforward_layernorm(res)
                    xn = mx.fast.rms_norm(
                        res, glue["router_w"],
                        self.pre_feedforward_layernorm.eps)
                    h2_in = self.pre_feedforward_layernorm_2(res)
                h1 = self.mlp(h1_in)
                r = self.router
                k = r.config.top_k_experts
                inds, sc = kq.moe_router_topk(
                    r.proj(xn).reshape(t, -1), k, True,
                    shared_gate=False,
                    per_expert_scale=r.per_expert_scale)
                shp = x.shape[:-1]
                h2 = self.experts(
                    h2_in, inds.reshape(*shp, k), sc.reshape(*shp, k))
                if h2.dtype != x.dtype:  # stock-experts arm mixes in f32
                    h2 = h2.astype(x.dtype)
                if _glue_parts & 4:
                    h = kq.rmsnorm2_add(
                        h1, self.post_feedforward_layernorm_1.weight,
                        h2, self.post_feedforward_layernorm_2.weight,
                        self.post_feedforward_layernorm_1.eps)
                else:
                    h = (self.post_feedforward_layernorm_1(h1)
                         + self.post_feedforward_layernorm_2(h2))
            else:
                h = self.mlp(self.pre_feedforward_layernorm(res))

            pfn = self.post_feedforward_layernorm
            if _glue_parts & 8:
                out = kq.add_rmsnorm(
                    h, res, pfn.weight, pfn.eps, scale=glue["scale"])
            else:
                out = res + pfn(h)
                if glue["scale"] is not None:
                    out = out * glue["scale"]
            return out, shared_kv, offset

    _FusedKQuantGemmaDecoderLayer.__name__ = "_FusedKQuantGemmaDecoderLayer"
    return _FusedKQuantGemmaDecoderLayer


def _install_gemma_fusion(model, caps) -> int:
    classes: dict = {}
    n = 0
    for _, m in model.named_modules():
        if caps.block_env_on and _eligible_gemma_experts(m, caps):
            _swap_class(m, classes, _make_fused_gemma_experts, caps)
        elif _eligible_gemma_router(m, caps):
            _swap_class(m, classes, _make_fused_gemma_router, caps)
        elif caps.glue_env_on and _eligible_gemma_layer(m, caps):
            _swap_class(m, classes, _make_fused_gemma_layer, caps)
        else:
            continue
        n += 1
    return n


def install_fused_moe_glu(model) -> int:
    """Fuse the mxfp4 MoE decode path: class-swap each ``SwitchGLU`` whose
    three projections are packed-mxfp4 ``NativeFPSwitchLinear`` with bias and
    whose activation is the clamped SwiGLU (gpt-oss), so decode runs
    ``kq.moe_glu_gather`` (gate + up + biases + activation, one dispatch) +
    ``kq.gather_qmv_bias`` (down + bias) instead of three stock gathers plus
    ~9 elementwise kernels per MoE layer. Beyond the kernel-count win, the kq
    matvecs run any K % 32 == 0 at full speed -- stock's fast gather needs
    K % 512 == 0, which e.g. gpt-oss's K=2880 fails, so its every expert
    gather takes the guarded-slow variant.

    gpt-oss ``MLPBlock`` additionally gets block-level fusion when the kq
    build ships gather_qmv_mix_bias: the top-k/softmax router epilogue
    collapses into kq.moe_router_topk (gated by GMLX_FUSED_MOE_ROUTER) and
    the softmax-weighted mix + expert bias fold into the down gather
    (kq.gather_qmv_mix_bias, f32 accumulate; gated by
    GMLX_FUSED_MOE_BLOCK).

    K-quant SwitchGLUs (plain silu(gate) * up on un-biased KQuantSwitchLinear
    stacks, e.g. qwen3.6-MoE) get the same treatment via kq.moe_glu_gather_kq
    + kq.gather_qmv_kq, per-codec (full GGUF codec matrix; gate and up must
    share a codec, down may differ). gpt-oss under native-fp wire mode
    (biased mxfp4/nvfp4 KQuantSwitchLinear stacks) rides the same ops with
    act "swiglu_clamp" -- the expert biases fold into the kernels.

    qwen3-next-shaped MoE blocks (router + SwitchGLU + sigmoid-gated shared
    expert) additionally get whole-block fusion when the shared expert is
    shape-matched to the expert stacks -- same codec, or a q6_k/q8_0 upcast
    (UD-style): the shared expert rides the GLU gather as one extra slot and
    the routing mix folds into the down gather (kq.moe_glu_gather_shexp_kq +
    kq.gather_qmv_mix_kq). Disable just this level with
    GMLX_FUSED_MOE_BLOCK=0.

    hy_v3 MoE blocks (sigmoid+bias router, ungated shared expert) get the
    same shared-expert ride-along via install_hyv3_shexp_fold, a separate
    call that stamps the fused SwitchGLU instead of swapping the block.

    gemma-4 MoE layers (separate Router + Experts modules, GeGLU, no shared
    expert): the score-weighted (w * y).sum(-2) mix folds into the down
    gather (kq.gather_qmv_mix_ns_kq; gate/up stay stock -- the fused GLU
    gather loses E2E to in-step scheduling at this small-expert geometry;
    gated by GMLX_FUSED_MOE_BLOCK), and the router epilogue collapses
    into kq.moe_router_topk(shared_gate=False, per_expert_scale=...) (gated
    by GMLX_FUSED_MOE_ROUTER). The whole DecoderLayer body additionally
    swaps to a restructured form that caches the router's premultiplied
    norm weight and runs the router epilogue in-layer (gated by
    GMLX_FUSED_MOE_GLUE; kq residual+rmsnorm glue kernels are optional
    via GMLX_FUSED_MOE_GLUE_PARTS and default off -- they lose E2E at
    B=1 despite isolated wins).

    Prefill / sorted calls (indices.size >= 64, mirroring SwitchGLU's own sort
    gate) fall through to the stock path. Disable with GMLX_FUSED_MOE=0.
    Returns the number of instances swapped."""
    caps = _FusedMoeCaps()
    if not caps.enabled or not caps.has_base:
        return 0
    return (
        _install_nativefp_fusion(model, caps)
        + _install_gptoss_block_fusion(model, caps)
        + _install_kquant_glu_fusion(model, caps)
        + _install_kq_block_fusion(model, caps)
        + _install_gemma_fusion(model, caps)
    )


class LoRAKQuantLinear(nn.Module):
    """A base Linear (kquant or float) with a LoRA delta applied live.

    Forward: ``base(x) + scale * (x @ A.T) @ B.T``, where ``A`` is ``(rank, in)``
    and ``B`` is ``(out, rank)`` - the PEFT / llama.cpp GGUF orientation, so
    ``delta_W = scale * B @ A`` has the base weight's ``(out, in)`` shape. The base
    weight is left untouched (no merge, no requantization): the served model is the
    base's existing K-quant error plus the adapter's *exact* full-precision delta.

    The delta is computed in the adapter's (float) precision for accuracy, then
    cast back to the base output dtype so the residual stream dtype is unchanged.
    """

    def __init__(self, base: nn.Module, a: mx.array, b: mx.array, scale: float):
        super().__init__()
        self.base = base
        self.lora_a = a            # (rank, in)
        self.lora_b = b            # (out, rank)
        self.scale = float(scale)
        self.freeze()

    def __call__(self, x, *args, **kwargs):
        y = self.base(x, *args, **kwargs)
        z = (x @ self["lora_a"].T) @ self["lora_b"].T
        return y + (self.scale * z).astype(y.dtype)

    def _extra_repr(self):
        rank = self.lora_b.shape[-1]
        return f"rank={rank}, scale={self.scale:g}, base={type(self.base).__name__}"


def install_lora_adapter(model: nn.Module, plan,
                         *, n_head: int | None = None,
                         n_head_kv: int | None = None) -> int:
    """Wrap each Linear named by a LoRA ``plan`` with its live adapter delta.

    A second leaf-swap pass (mirrors :func:`install_kquant_modules`): walks
    ``model.leaf_modules()`` and replaces each targeted ``KQuantLinear`` / float
    ``nn.Linear`` with a :class:`LoRAKQuantLinear`. Returns the count wrapped.

    ``qk_permute`` targets (llama-family q/k) need the base head counts to bring
    the adapter delta into mlx-lm's de-permuted attention layout: the permute
    reorders the output rows of ``B`` (``n_head`` for ``q_proj``, ``n_head_kv``
    for ``k_proj``). Missing head counts on such a target raise rather than apply
    an un-permuted delta onto a de-permuted base (which would corrupt attention).

    Two loud failures, never a silent drop (a dropped adapter weight serves a
    subtly wrong model): a target that names an unwrappable module (MoE-expert /
    embedding adapters are deferred) and a target with no matching module at all.
    """
    targets = dict(plan.modules)   # module_path -> LoraModule
    wrapped: set[str] = set()

    def _wrap(path: str, module):
        lm = targets.get(path)
        if lm is None:
            return module
        if not isinstance(module, (KQuantLinear, nn.Linear)):
            raise NotImplementedError(
                f"LoRA target {path!r} is a {type(module).__name__}, not a "
                f"wrappable Linear (MoE-expert / embedding LoRA is deferred)")
        b = lm.b
        if lm.transform == "qk_permute":
            nh = n_head_kv if path.endswith("k_proj") else n_head
            if nh is None:
                # Never fall back to n_head for a k_proj: on a GQA base the
                # wrong head count permutes the delta into garbage attention.
                raise ValueError(
                    f"LoRA target {path!r} needs a qk_permute but the base head "
                    f"counts (n_head / n_head_kv) were not provided")
            b = qk_permute_wire(b, nh)
        elif lm.transform != "passthrough":
            raise NotImplementedError(
                f"LoRA target {path!r} has unsupported transform "
                f"{lm.transform!r}")
        wrapped.add(path)
        return LoRAKQuantLinear(module, lm.a, b, lm.scale)

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_wrap, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)

    missing = set(targets) - wrapped
    if missing:
        raise ValueError(
            f"LoRA adapter targets {sorted(missing)} have no matching module in "
            f"the loaded model - adapter/base mismatch (never silently skipped)")
    return len(wrapped)


def install_kquant_modules(model: nn.Module,
                           hf_kquant_meta: dict[str, str],
                           native_fp_wire: bool = False) -> int:
    """Swap leaf modules for kquant equivalents.

    Walks ``model.leaf_modules()``; replaces each ``Linear`` / ``Embedding`` /
    ``SwitchLinear`` whose ``<path>.weight`` is in ``hf_kquant_meta`` with the
    matching ``KQuant*`` module (codec from the meta). Returns the count of
    replacements made.

    ``native_fp_wire`` routes mxfp4/nvfp4 experts to ``KQuantSwitchLinear``
    (zero-copy GGUF wire bytes, streamable) instead of ``NativeFPSwitchLinear``
    (MLX packed layout, needs the eager repack).

    A codec'd leaf whose module class matches no branch raises: the packed
    wire bytes would otherwise be loaded into a stock float module and fail
    later as an opaque shape error inside its matmul. The usual cause is an
    upstream release moving/renaming a module class (e.g. mlx-vlm 0.6.4
    vendoring mlx-lm's switch_layers).
    """
    _switch_linear_types, _ = switch_layer_types()
    n_replaced = 0
    unmatched: list[str] = []

    def _replace(path: str, module):
        nonlocal n_replaced
        weight_key = f"{path}.weight"
        codec = hf_kquant_meta.get(weight_key)
        if codec is None:
            return module
        if codec in NATIVE_FP_CODECS:
            # Micro-scaling float codec. Only the SwitchGLU expert path is
            # exercised today (gpt-oss, DS-V4). A native-fp Linear / Embedding
            # would need its own module (packed) or its own wire Metal leaves;
            # fail loud rather than mis-dispatch onto the kquant path.
            if isinstance(module, _switch_linear_types):
                n_experts, out_dims, in_dims = module.weight.shape
                bias = "bias" in module
                n_replaced += 1
                if native_fp_wire:
                    return KQuantSwitchLinear(n_experts, out_dims, in_dims,
                                              bias, codec)
                return NativeFPSwitchLinear(n_experts, out_dims, in_dims, bias,
                                            codec)
            raise NotImplementedError(
                f"native-fp codec {codec!r} on {type(module).__name__} at "
                f"{path!r} - only SwitchLinear (MoE experts) is wired so far")
        if isinstance(module, nn.Linear):
            out_dims, in_dims = module.weight.shape
            bias = "bias" in module
            n_replaced += 1
            return KQuantLinear(in_dims, out_dims, bias, codec)
        if isinstance(module, nn.Embedding):
            num_emb, dims = module.weight.shape
            n_replaced += 1
            return KQuantEmbedding(num_emb, dims, codec)
        if isinstance(module, _switch_linear_types):
            n_experts, out_dims, in_dims = module.weight.shape
            bias = "bias" in module
            n_replaced += 1
            return KQuantSwitchLinear(n_experts, out_dims, in_dims, bias, codec)
        if MultiLinear is not None and isinstance(module, MultiLinear):
            num_heads, out_dims, in_dims = module.weight.shape
            n_replaced += 1
            return KQuantMultiLinear(in_dims, out_dims, num_heads, codec)
        if isinstance(module, (KQuantLinear, KQuantEmbedding,
                               KQuantSwitchLinear, KQuantMultiLinear,
                               NativeFPSwitchLinear)):
            return module  # already swapped (repeat install)
        unmatched.append(
            f"{path} ({type(module).__module__}.{type(module).__name__}, "
            f"codec {codec!r})")
        return module

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_replace, leaves, is_leaf=nn.Module.is_module)
    if unmatched:
        head = ", ".join(unmatched[:3])
        more = f" (+{len(unmatched) - 3} more)" if len(unmatched) > 3 else ""
        raise ValueError(
            f"{len(unmatched)} quantized weights have no recognized module "
            f"class to attach to: {head}{more} - upstream module class not "
            f"recognized (mlx-lm/mlx-vlm version change?)")
    model.update_modules(leaves)
    return n_replaced
