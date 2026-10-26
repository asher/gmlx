"""Per-tensor weight scales (``<weight>.scale`` in a GGUF).

llama.cpp multiplies a matmul's product by the weight's ``.scale`` tensor
before the bias (NVFP4 exports carry it as the global "scale2"). Remap
names it ``<module>.out_scale``; a stacked expert module has one value per
expert. ``install_out_scales`` gives each such module an ``out_scale``
parameter. An nvfp4 KQuantLinear or KQuantSwitchLinear keeps its class:
mlx-kquant applies the scale inside its kernels, and the fused paths pass
it to their ops. Any other module gets a subclass that applies the scale
after its own ``__call__``, and the fusions that read wire bytes directly
skip such a module (``wrapped_scale``). The subclass keeps the scale as
``post_scale``: an mlx-kquant module applies an ``out_scale`` itself, and
refuses one on a codec other than nvfp4.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import mlx.nn as nn

from gmlx.envflags import env_bool

_SCALED: dict[type, type] = {}

# The kq ops gmlx calls on a scaled module, with the scale argument each
# documents.
_KQ_SCALE_ARGS = (
    ("quantized_matmul", "out_scale"),
    ("quantized_matmul_multi", "out_scales"),
    ("gather_qmm", "out_scale"),
    ("gather_qmm_seg", "out_scale"),
    ("gather_qmv_kq", "out_scale"),
    ("gather_qmv_mix_ns_kq", "out_scale"),
    ("gather_qmv_mix_kq", "shexp_scale"),
    ("moe_glu_gather_kq", "gate_scale"),
    ("moe_glu_gather_shexp_kq", "shexp_gate_scale"),
)


@functools.cache
def _kq_takes_scales() -> bool:
    try:
        import mlx_kquant as kq
        from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear
    except ImportError:
        return False
    if any("out_scale" not in (c.__doc__ or "")
           for c in (KQuantLinear, KQuantSwitchLinear)):
        return False
    return all(arg in (getattr(getattr(kq, op, None), "__doc__", "") or "")
               for op, arg in _KQ_SCALE_ARGS)


def kq_applies_out_scale() -> bool:
    """Whether nvfp4 weight scales go to the mlx-kquant kernels: the
    installed build takes them on every op gmlx calls, and
    ``GMLX_KQ_OUT_SCALE`` is not 0."""
    return env_bool("GMLX_KQ_OUT_SCALE", True) and _kq_takes_scales()


def _kq_native(m) -> bool:
    """``m`` is a kq module whose own call passes its ``out_scale`` to the
    kernel."""
    from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

    from .hadamard_modules import HadamardKQuantLinear

    return (type(m) in (KQuantLinear, KQuantSwitchLinear, HadamardKQuantLinear)
            and m.kquant_type == "nvfp4")


def _rescale(m, y, args, kwargs):
    """``s * (W x) + b`` from the module's ``W x + b``. A per-expert scale
    or bias is read at the call's expert ids."""
    s = m["post_scale"]
    b = m["bias"] if "bias" in m else None
    idx = None
    if s.size > 1 or (b is not None and b.ndim > 1):
        idx = args[0] if args else kwargs["indices"]
    if s.size > 1:
        s = s.reshape(-1)[idx]
        s = s.reshape(s.shape + (1,) * (y.ndim - s.ndim))
    s = s.astype(y.dtype)
    if b is None:
        return y * s
    if b.ndim > 1:
        b = mx.expand_dims(b[idx], -2)
    b = b.astype(y.dtype)
    return (y - b) * s + b


def _scaled_class(cls: type) -> type:
    sub = _SCALED.get(cls)
    if sub is None:
        def __call__(self, x, *args, **kwargs):
            if x.dtype == mx.float16:
                # A small scale leaves an unscaled product past the float16
                # range; bfloat16 holds it.
                y = cls.__call__(self, x.astype(mx.bfloat16), *args, **kwargs)
                return _rescale(self, y, args, kwargs).astype(mx.float16)
            y = cls.__call__(self, x, *args, **kwargs)
            return _rescale(self, y, args, kwargs)

        sub = type(f"OutScaled{cls.__name__}", (cls,), {
            "__call__": __call__, "__module__": __name__})
        _SCALED[cls] = sub
    return sub


def weight_scale(module):
    """The module's weight scale, applied in the kernels or by the
    subclass, or None."""
    s = getattr(module, "out_scale", None)
    return getattr(module, "post_scale", None) if s is None else s


def has_out_scale(module) -> bool:
    return isinstance(module, nn.Module) and weight_scale(module) is not None


def wrapped_scale(module) -> bool:
    """Whether ``module`` applies its weight scale after its own call, so a
    path that reads its wire bytes instead of calling it would drop the
    scale."""
    return bool(_SCALED) and isinstance(module, tuple(_SCALED.values()))


def cat_out_scale(mods):
    """The weight scale of a module whose rows are the rows of ``mods`` in
    order: each one's scale over its rows as one float32 vector, ones for a
    module without one. None when none has a scale."""
    scales = [m.get("out_scale") for m in mods]
    if all(s is None for s in scales):
        return None
    return mx.concatenate([
        mx.broadcast_to(mx.ones((1,), dtype=mx.float32) if s is None else s,
                        (m["weight"].shape[0],))
        for m, s in zip(mods, scales)])


def set_out_scale(m, scale) -> None:
    """Give ``m`` a frozen ``out_scale``; no-op for None."""
    if scale is not None:
        m.out_scale = scale
        m.freeze(recurse=False, keys=["out_scale"])


def _scale_shape(m, key: str, size: int) -> tuple:
    if size == 1:
        return (1,)
    w = m.get("weight")
    if isinstance(w, mx.array) and w.ndim == 3 and w.shape[0] == size:
        return (size, 1)
    raise NotImplementedError(
        f"{key}: {size} weight scale values, but only an expert stack of "
        f"{size} experts takes one per expert")


def install_out_scales(model: nn.Module, hf_weights: dict,
                       hf_kquant_meta: dict | None = None) -> int:
    """Give every module with a ``<path>.out_scale`` weight its scale: an
    ``out_scale`` parameter that mlx-kquant applies, or the scaling
    subclass and a ``post_scale`` parameter (see the module docstring). Runs after ``install_kquant_modules`` and the
    Hadamard swap, before the weights load and the fusions install. An
    ``lm_head`` scale applies only to an NVFP4 head, as in llama.cpp.
    Returns the count."""
    suffix = ".out_scale"
    paths = {k[:-len(suffix)] for k in hf_weights if k.endswith(suffix)}
    if not paths:
        return 0
    meta = hf_kquant_meta or {}
    native = kq_applies_out_scale()
    n = 0
    for path, m in model.named_modules():
        if path not in paths:
            continue
        paths.discard(path)
        key = path + suffix
        scale = hf_weights[key]
        if path.split(".")[-1] == "lm_head" and meta.get(
                path + ".weight") != "nvfp4":
            del hf_weights[key]
            continue
        shape = _scale_shape(m, key, int(scale.size))
        scale = scale.astype(mx.float32).reshape(shape)
        name = "out_scale"
        if not (native and _kq_native(m)):
            if not wrapped_scale(m):
                m.__class__ = _scaled_class(type(m))
            name = "post_scale"
            del hf_weights[key]
        hf_weights[f"{path}.{name}"] = scale
        setattr(m, name, mx.ones(shape, dtype=mx.float32))
        m.freeze(recurse=False, keys=[name])
        n += 1
    return n
