"""Per-tensor weight scales (``<weight>.scale`` in a GGUF).

llama.cpp multiplies a matmul's output by the weight's ``.scale`` tensor
when the file has one (NVFP4 exports carry it as the global "scale2").
Remap names it ``<module>.out_scale``. ``install_out_scales`` swaps each
such module's class for a subclass that applies the scale after its own
``__call__``, so every caller that runs the module gets it. Fusions that
read a projection's wire bytes directly skip a module with ``out_scale``,
as they skip one with a bias.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

_SCALED: dict[type, type] = {}


def _scaled_class(cls: type) -> type:
    sub = _SCALED.get(cls)
    if sub is None:
        def __call__(self, x, *args, **kwargs):
            if x.dtype == mx.float16:
                # A small scale leaves an unscaled product past the float16
                # range; bfloat16 holds it.
                y = cls.__call__(self, x.astype(mx.bfloat16), *args, **kwargs)
                y = y * self["out_scale"].astype(y.dtype)
                return y.astype(mx.float16)
            y = cls.__call__(self, x, *args, **kwargs)
            return y * self["out_scale"].astype(y.dtype)

        sub = type(f"OutScaled{cls.__name__}", (cls,), {
            "__call__": __call__, "__module__": __name__})
        _SCALED[cls] = sub
    return sub


def has_out_scale(module) -> bool:
    return isinstance(module, nn.Module) and "out_scale" in module


def install_out_scales(model: nn.Module, hf_weights: dict,
                       hf_kquant_meta: dict | None = None) -> int:
    """Give every module with a ``<path>.out_scale`` weight an ``out_scale``
    parameter and the scaling subclass. Runs after ``install_kquant_modules``
    and the Hadamard swap, before the weights load. An ``lm_head`` scale
    applies only to an NVFP4 head, as in llama.cpp. Returns the count."""
    suffix = ".out_scale"
    paths = {k[:-len(suffix)] for k in hf_weights if k.endswith(suffix)}
    if not paths:
        return 0
    meta = hf_kquant_meta or {}
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
        if int(scale.size) != 1:
            raise NotImplementedError(
                f"{key}: per-expert weight scales ({int(scale.size)} values) "
                f"are not supported yet")
        if type(m) not in _SCALED.values():
            m.__class__ = _scaled_class(type(m))
        m.out_scale = mx.ones((1,), dtype=mx.float32)
        m.freeze(recurse=False, keys=["out_scale"])
        n += 1
    return n
