"""Replace the rope modules of a class-built model with the resolved scaling.

The model classes read rope scaling from their config in their own ways, and
some not at all. After the build, one pass replaces every rope module of an
architecture in ``rope_scaling.SWAP_ARCHES`` with a ``ScaledRoPE`` built from
the scaling the config synthesizer resolved, so the table and the magnitude
are llama.cpp's whatever the class did. Attention code that calls
``self.rope`` (the stock classes and the fused patches alike) picks it up.
Sliding-window layers take the sliding scaling (``RopeScaling.for_sliding``).
A plain rope leaves the model untouched.
"""

from __future__ import annotations

import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import loadlog
from . import rope_scaling as rs

_SLIDING_TYPES = frozenset({"sliding_attention", "sliding", "swa"})
_LAYER_RE = re.compile(r"^(.*?layers)\.(\d+)\.")


class ScaledRoPE(nn.Module):
    """``mx.fast.rope`` on a precomputed table, with the cos and sin
    magnitude llama.cpp's rope_yarn applies. The magnitude scales the
    rotated dims before the rotation; the dims past ``dims`` pass through."""

    def __init__(self, dims: int, traditional: bool, freqs: mx.array,
                 mscale: float):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.mscale = mscale
        self._freqs = freqs

    def __call__(self, x, offset=0):
        m = self.mscale
        if m != 1.0:
            d = self.dims
            if x.shape[-1] == d:
                x = x * m
            else:
                x = mx.concatenate([x[..., :d] * m, x[..., d:]], axis=-1)
        return mx.fast.rope(x, self.dims, traditional=self.traditional,
                            base=None, scale=1.0, offset=offset,
                            freqs=self._freqs)


def _swappable(r) -> bool:
    """A rope module whose table the swap rebuilds: plain, yarn, linear, and
    the rope_freqs factored rope. Others (NoPE, longrope, own classes) stay."""
    from gmlx._vendor.mlx_lm.models.rope_utils import YarnRoPE
    from gmlx._vendor.mlx_vlm.models.rope_utils import YarnRoPE as VlmYarnRoPE

    return (type(r) in (nn.RoPE, YarnRoPE, VlmYarnRoPE)
            or type(r).__name__ == "_FactoredRoPE")


def _is_mrope(r) -> bool:
    from gmlx._vendor.mlx_vlm.models.rope_utils import MRoPERotaryEmbedding

    return isinstance(r, MRoPERotaryEmbedding)


def _scale_mrope(r, scaling: rs.RopeScaling, base: float) -> None:
    """Multi-axis rope (the Qwen 3.5 family): the kernel and the cos/sin
    path both read ``inv_freq``, and ``attention_scaling`` is the cos and
    sin magnitude, which the fused apply takes beside the kernel. The
    sections pick a position axis per pair and leave the table alone."""
    inv = rs.inv_freq(int(r.dim), base, scaling)
    r._inv_freq = mx.array(inv, dtype=mx.float32)
    r.attention_scaling = scaling.mscale
    r._gmlx_rope_scaled = True
    mx.eval(r._inv_freq)


def _rope_like(r) -> bool:
    tn = type(r).__name__.lower()
    return ("rope" in tn or "rotary" in tn) and "nope" not in tn


def _text_config(config) -> dict:
    tc = config.get("text_config") if isinstance(config, dict) else None
    if isinstance(tc, dict) and rs.CONFIG_KEY in tc:
        return tc
    return config


def _sliding_layers(config: dict, model) -> tuple[str | None, set[int]]:
    """The layer list path holding the trunk and the indices of its sliding
    layers, from the config's ``layer_types``."""
    types = config.get("layer_types") or []
    sliding = {i for i, t in enumerate(types) if t in _SLIDING_TYPES}
    if not sliding:
        return None, set()
    counts: dict[str, int] = {}
    for name, _ in model.named_modules():
        m = _LAYER_RE.match(name + ".")
        if m:
            counts[m.group(1)] = max(counts.get(m.group(1), 0),
                                     int(m.group(2)) + 1)
    trunk = [p for p, n in counts.items() if n == len(types)]
    return (min(trunk, key=len) if trunk else None), sliding


def _module_base(r) -> float | None:
    b = getattr(r, "base", None)
    return None if b is None else float(b)


def _scaled(r, scaling: rs.RopeScaling, base: float) -> ScaledRoPE:
    dims = int(r.dims)
    inv = rs.inv_freq(dims, base, scaling).astype(np.float64)
    factors = getattr(r, "_factors", None)
    if factors is not None:
        # ggml rope with freq_factors divides the extrapolated angle before
        # the yarn blend, which scales the blended angle the same way.
        inv = inv / np.asarray(factors.astype(mx.float32), dtype=np.float64)
    freqs = mx.array(1.0 / inv, dtype=mx.float32)
    return ScaledRoPE(dims, bool(r.traditional), freqs, scaling.mscale)


def apply(model, config) -> int:
    """Swap the rope modules of ``model`` for the scaling its synthesized
    ``config`` carries. Returns the number of modules swapped; 0 and no
    change when the rope is plain or the architecture is not swapped.

    Raises NotImplementedError when a rope module of a type the swap cannot
    rebuild remains, rather than run with a rope the scaling does not
    describe. A second call changes nothing.
    """
    cfg = _text_config(config)
    scaling = rs.from_config(cfg)
    if not scaling.active or scaling.arch not in rs.SWAP_ARCHES:
        return 0
    trunk, sliding = _sliding_layers(cfg, model)
    sliding_scaling = scaling.for_sliding()
    n = 0
    unswapped = []
    for name, mod in list(model.named_modules()):
        for attr in ("rope", "rotary_emb"):
            r = getattr(mod, attr, None)
            if r is None or not isinstance(r, nn.Module):
                continue
            path = f"{name}.{attr}" if name else attr
            if isinstance(r, ScaledRoPE) or getattr(r, "_gmlx_rope_scaled",
                                                    False):
                continue
            mrope = _is_mrope(r)
            if not mrope and not _swappable(r):
                if _rope_like(r):
                    unswapped.append(f"{path} ({type(r).__name__})")
                continue
            m = _LAYER_RE.match(path)
            is_swa = (trunk is not None and m is not None
                      and m.group(1) == trunk and int(m.group(2)) in sliding)
            mb = _module_base(r)
            if is_swa:
                s = sliding_scaling
                base = mb or scaling.freq_base_train or rs._DEFAULT_FREQ_BASE
            else:
                s = scaling
                base = (scaling.freq_base if scaling.freq_base_override
                        else mb or scaling.freq_base_train
                        or rs._DEFAULT_FREQ_BASE)
            if mrope:
                _scale_mrope(r, s, base)
            else:
                setattr(mod, attr, _scaled(r, s, base))
            n += 1
    if unswapped:
        raise NotImplementedError(
            f"the rope scaling options are not supported for this "
            f"{scaling.arch} model: its rope modules "
            f"{', '.join(unswapped[:4])} have no scaled form")
    if n:
        loadlog.verbose_print(
            f"[rope] {scaling.describe()} on {n} rope modules"
            + (f" ({len(sliding)} sliding layers)" if sliding else ""))
    return n
