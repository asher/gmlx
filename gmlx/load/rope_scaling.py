"""Rope scaling from the GGUF header and the user override, by llama.cpp's rules.

One resolver for every architecture. ``resolve`` reads the ``<arch>.rope.*``
keys and the run options and returns a ``RopeScaling``; ``inv_freq`` and
``freqs`` turn it into the per-pair frequency table, ``mscale`` gives the
factor llama.cpp multiplies cos and sin by, and ``effective_context`` the
context length every consumer reads.

The rules follow llama.cpp's context setup: each option wins over its header
key, a header with no type reads as linear with factor 1, type none forces
the frequency scale to 1, the original context falls back to the trained
context, and the extrapolation mix is 1 for yarn and 0 otherwise. The yarn
beta, extrapolation, attention-factor and log-multiplier keys are read only
for the architectures llama.cpp reads them for, so the same GGUF gives the
same numbers on both runtimes.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

import numpy as np

from .gguf_meta import read_float, read_int, read_string

SCALING_TYPES = ("none", "linear", "yarn", "longrope")
# The types a user may ask for, as llama.cpp's --rope-scaling accepts.
USER_SCALING_TYPES = ("none", "linear", "yarn")

# Header keys llama.cpp reads for some architectures only.
_BETA_ARCHES = frozenset({"mistral3", "grok", "k2-horizon"})
_EXT_ATTN_ARCHES = frozenset({"grok"})
# log multiplier: the deepseek converters write 0.1 * mscale_all_dim.
_LOG_MUL_DIV10_ARCHES = frozenset({"deepseek2", "deepseek32"})
_LOG_MUL_RAW_ARCHES = frozenset({"mistral3"})

_DEFAULT_FREQ_BASE = 10000.0
# Architectures whose sliding-window layers keep the trained frequency scale
# (rope_freq_scale_train_swa = rope_freq_scale_train in their llama.cpp
# loaders); every other one rotates them with scale 1.
_SWA_TRAIN_SCALE_ARCHES = frozenset({"gpt-oss", "gemma2"})
_DEFAULT_BETA_FAST = 32.0
_DEFAULT_BETA_SLOW = 1.0


@dataclass(frozen=True)
class RopeOverride:
    """Rope options from the command line or the serve config. ``None``
    means not given, so the header key or the default applies."""

    rope_scaling: str | None = None
    rope_scale: float | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None
    yarn_orig_ctx: int | None = None
    yarn_ext_factor: float | None = None
    yarn_attn_factor: float | None = None
    yarn_beta_fast: float | None = None
    yarn_beta_slow: float | None = None

    def __post_init__(self):
        t = self.rope_scaling
        if t is not None and t not in USER_SCALING_TYPES:
            raise ValueError(
                f"rope scaling type {t!r} is not one of "
                f"{', '.join(USER_SCALING_TYPES)}")
        for name in ("rope_scale", "rope_freq_scale", "rope_freq_base"):
            v = getattr(self, name)
            if v is not None and not (v > 0 and math.isfinite(v)):
                raise ValueError(f"{name.replace('_', '-')} must be a "
                                 f"positive number, got {v}")
        if (self.rope_scale is not None and self.rope_freq_scale is not None
                and not math.isclose(1.0 / self.rope_scale,
                                     self.rope_freq_scale, rel_tol=1e-6)):
            raise ValueError(
                "rope-scale and rope-freq-scale set the same value and "
                "disagree; give one of them")
        if self.yarn_orig_ctx is not None and self.yarn_orig_ctx <= 0:
            raise ValueError("yarn-orig-ctx must be a positive integer")
        for name in ("yarn_beta_fast", "yarn_beta_slow", "yarn_ext_factor",
                     "yarn_attn_factor"):
            v = getattr(self, name)
            if v is not None and not (v >= 0 and math.isfinite(v)):
                raise ValueError(f"{name.replace('_', '-')} must be a "
                                 f"non-negative number, got {v}")

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))

    @property
    def freq_scale(self) -> float | None:
        """The frequency scale the options ask for, from either spelling."""
        if self.rope_freq_scale is not None:
            return float(np.float32(self.rope_freq_scale))
        if self.rope_scale is not None:
            return float(np.float32(1.0) / np.float32(self.rope_scale))
        return None

    # The serve config spells the keys without the flag prefixes.
    _CONFIG_KEYS = {
        "type": "rope_scaling",
        "factor": "rope_scale",
        "freq_base": "rope_freq_base",
        "freq_scale": "rope_freq_scale",
        "original_context": "yarn_orig_ctx",
        "ext_factor": "yarn_ext_factor",
        "attn_factor": "yarn_attn_factor",
        "beta_fast": "yarn_beta_fast",
        "beta_slow": "yarn_beta_slow",
    }

    @classmethod
    def from_config(cls, block: Mapping[str, Any] | None) -> "RopeOverride":
        """Build from a serve config ``rope_scaling`` mapping. Unknown keys
        raise, naming the accepted ones."""
        if not block:
            return cls()
        if not isinstance(block, Mapping):
            raise ValueError("rope_scaling must be a mapping of "
                             f"{', '.join(cls._CONFIG_KEYS)}")
        kw: dict[str, Any] = {}
        for key, value in block.items():
            name = cls._CONFIG_KEYS.get(key)
            if name is None:
                raise ValueError(
                    f"rope_scaling: unknown key {key!r} (accepted: "
                    f"{', '.join(cls._CONFIG_KEYS)})")
            if value is None:
                continue
            if name == "rope_scaling":
                kw[name] = str(value)
            elif name == "yarn_orig_ctx":
                kw[name] = int(value)
            else:
                kw[name] = float(value)
        return cls(**kw)

    @classmethod
    def from_args(cls, args) -> "RopeOverride":
        """Build from an argparse namespace that carries the rope flags."""
        return cls(**{f.name: getattr(args, f.name, None) for f in fields(cls)})


NO_OVERRIDE = RopeOverride()


@dataclass(frozen=True)
class RopeScaling:
    """The resolved rope scaling for one model.

    ``attn_factor`` is the value llama.cpp passes to its rope kernel;
    ``mscale`` is what cos and sin are multiplied by at a site that rotates
    with ``freq_scale``. ``freq_base`` is None when neither the header nor
    the options name one, so the model keeps its own base.
    """

    type: str = "none"
    freq_scale: float = 1.0
    freq_scale_train: float = 1.0
    original_context: int = 0
    ext_factor: float = 0.0
    attn_factor: float = 1.0
    beta_fast: float = _DEFAULT_BETA_FAST
    beta_slow: float = _DEFAULT_BETA_SLOW
    freq_base: float | None = None
    freq_base_override: bool = False
    alpha: float | None = None
    # The GGUF architecture, the header base, and the frequency scale of
    # sliding-window layers, which llama.cpp rotates with the trained scale
    # of the architecture rather than the run options.
    arch: str = ""
    freq_base_train: float | None = None
    freq_scale_swa: float = 1.0
    # True when the run options set any rope key; False for a scaling the
    # GGUF header alone asks for.
    from_options: bool = False

    @property
    def factor(self) -> float:
        return 1.0 / self.freq_scale

    @property
    def mscale(self) -> float:
        """llama.cpp's cos and sin multiplier at a site rotating with
        ``freq_scale`` (ggml rope_yarn)."""
        m = np.float32(self.attn_factor)
        if self.ext_factor != 0.0:
            m = m * (np.float32(1.0) + np.float32(0.1)
                     * np.log(np.float32(1.0) / np.float32(self.freq_scale)))
        return float(np.float32(m))

    def for_sliding(self) -> "RopeScaling":
        """The scaling of a sliding-window layer: llama.cpp's
        get_rope_freq_scale and get_rope_freq_base pick the trained sliding
        values there (llama-model.cpp), while the extrapolation mix, the
        attention factor and the betas stay the run's."""
        return replace(self, freq_scale=self.freq_scale_swa,
                       freq_base=self.freq_base_train,
                       freq_base_override=False)

    @property
    def scales_frequencies(self) -> bool:
        """True when the table differs from the plain rope table."""
        return self.freq_scale != 1.0

    @property
    def active(self) -> bool:
        """True when a rope site must change: the table, the magnitude, or
        a base the options set."""
        return (self.scales_frequencies or self.mscale != 1.0
                or self.freq_base_override)

    @property
    def user_set(self) -> bool:
        """True when the run options changed the scaling (the
        context-length rule and the logs key off it)."""
        return self.freq_scale != self.freq_scale_train

    def identity(self) -> tuple | None:
        """A hashable key for caches that outlive one model: None when the
        rope is plain, so default keys stay as they were."""
        if not self.active:
            return None
        return (self.type, self.freq_scale, self.original_context,
                self.ext_factor, self.attn_factor, self.beta_fast,
                self.beta_slow, self.freq_base if self.freq_base_override
                else None)

    def describe(self) -> str:
        if not self.active:
            return "none"
        parts = [self.type, f"factor {self.factor:g}"]
        if self.type == "yarn":
            parts.append(f"original context {self.original_context}")
        m = self.mscale
        if m != 1.0:
            parts.append(f"attention scale {m:.4f}")
        if self.freq_base_override and self.freq_base is not None:
            parts.append(f"freq base {self.freq_base:g}")
        return ", ".join(parts)


NO_SCALING = RopeScaling()


def _f32(x) -> np.float32:
    return np.float32(x)


def _get_mscale(scale: np.float32, mscale: np.float32) -> np.float32:
    if scale <= 1.0:
        return _f32(1.0)
    return _f32(0.1) * mscale * _f32(np.log(scale)) + _f32(1.0)


def resolve(meta, arch: str, override: RopeOverride | None = None, *,
            n_ctx_train: int | None = None) -> RopeScaling:
    """Resolve the rope scaling for ``arch`` from the GGUF header ``meta``
    (a decoded KV dict or a reader) and the run options ``override``.

    ``n_ctx_train`` defaults to ``<arch>.context_length``.
    """
    ov = override or NO_OVERRIDE
    k = f"{arch}.rope."

    if n_ctx_train is None:
        n_ctx_train = read_int(meta, f"{arch}.context_length") or 0

    header_type = read_string(meta, k + "scaling.type") or "linear"
    if header_type not in SCALING_TYPES:
        raise ValueError(f"{arch}: unknown rope scaling type {header_type!r} "
                         f"in the GGUF header")
    factor = read_float(meta, k + "scaling.factor")
    if factor is None:
        factor = read_float(meta, k + "scale_linear")
    freq_scale_train = 1.0 if not factor else float(_f32(1.0) / _f32(factor))

    orig_hdr = read_int(meta, k + "scaling.original_context_length")
    base_hdr = read_float(meta, k + "freq_base")
    rope_attn_factor = read_float(meta, k + "scaling.attn_factor")
    rope_attn_factor = 1.0 if rope_attn_factor is None else rope_attn_factor
    alpha = read_float(meta, k + "scaling.alpha")

    beta_fast_hdr = beta_slow_hdr = None
    if arch in _BETA_ARCHES:
        beta_fast_hdr = read_float(meta, k + "scaling.yarn_beta_fast")
        beta_slow_hdr = read_float(meta, k + "scaling.yarn_beta_slow")
    ext_hdr = attn_hdr = None
    if arch in _EXT_ATTN_ARCHES:
        ext_hdr = read_float(meta, k + "scaling.yarn_ext_factor")
        attn_hdr = read_float(meta, k + "scaling.yarn_attn_factor")
    log_mul = 0.0
    if arch in _LOG_MUL_DIV10_ARCHES or arch in _LOG_MUL_RAW_ARCHES:
        lm = read_float(meta, k + "scaling.yarn_log_multiplier")
        if lm is not None:
            log_mul = (float(_f32(lm) / _f32(0.1))
                       if arch in _LOG_MUL_DIV10_ARCHES else lm)

    # The cparams block.
    ext_factor = ov.yarn_ext_factor if ov.yarn_ext_factor is not None else (
        ext_hdr if ext_hdr is not None else -1.0)
    attn_factor = ov.yarn_attn_factor if ov.yarn_attn_factor is not None else (
        attn_hdr if attn_hdr is not None else 1.0)
    beta_fast = ov.yarn_beta_fast if ov.yarn_beta_fast is not None else (
        beta_fast_hdr if beta_fast_hdr is not None else _DEFAULT_BETA_FAST)
    beta_slow = ov.yarn_beta_slow if ov.yarn_beta_slow is not None else (
        beta_slow_hdr if beta_slow_hdr is not None else _DEFAULT_BETA_SLOW)

    freq_base = ov.rope_freq_base if ov.rope_freq_base is not None else base_hdr
    freq_scale = ov.freq_scale if ov.freq_scale is not None else freq_scale_train
    original_context = (ov.yarn_orig_ctx if ov.yarn_orig_ctx
                        else orig_hdr if orig_hdr else n_ctx_train)

    scaling_type = ov.rope_scaling if ov.rope_scaling is not None else header_type
    if scaling_type == "none":
        freq_scale = 1.0
    if ext_factor < 0.0:
        ext_factor = 1.0 if scaling_type == "yarn" else 0.0

    af = _f32(attn_factor)
    if ext_factor != 0.0:
        factor_f = _f32(1.0) / _f32(freq_scale)
        if log_mul != 0.0:
            mscale = _f32(1.0)
            mscale_all = _f32(log_mul)
            if arch == "deepseek2" and mscale_all != 1.0:
                mscale = mscale_all
            af = _get_mscale(factor_f, mscale) / _get_mscale(factor_f, mscale_all)
        else:
            af = _get_mscale(factor_f, _f32(1.0))
        af = af * (_f32(1.0) / (_f32(1.0) + _f32(0.1) * _f32(np.log(factor_f))))
    af = af * _f32(rope_attn_factor)

    return RopeScaling(
        type=scaling_type,
        freq_scale=float(freq_scale),
        freq_scale_train=float(freq_scale_train),
        original_context=int(original_context),
        ext_factor=float(ext_factor),
        attn_factor=float(_f32(af)),
        beta_fast=float(beta_fast),
        beta_slow=float(beta_slow),
        freq_base=None if freq_base is None else float(freq_base),
        freq_base_override=ov.rope_freq_base is not None,
        alpha=alpha,
        arch=arch,
        freq_base_train=None if base_hdr is None else float(base_hdr),
        freq_scale_swa=(float(freq_scale_train)
                        if arch in _SWA_TRAIN_SCALE_ARCHES else 1.0),
        from_options=not ov.is_empty(),
    )


def with_base(scaling: RopeScaling, base: float) -> RopeScaling:
    """The same scaling with ``base`` as its frequency base."""
    return replace(scaling, freq_base=float(base))


def corr_dims(dims: int, base: float, scaling: RopeScaling) -> tuple[float, float]:
    """ggml_rope_yarn_corr_dims: the pair indices the ramp runs between."""
    def corr_dim(n_rot):
        return (_f32(dims) * _f32(np.log(_f32(scaling.original_context)
                                         / (_f32(n_rot) * _f32(2) * _f32(math.pi))))
                / (_f32(2) * _f32(np.log(_f32(base)))))
    start = float(np.floor(corr_dim(scaling.beta_fast)))
    end = float(np.ceil(corr_dim(scaling.beta_slow)))
    return max(0.0, start), min(float(dims - 1), end)


def inv_freq(dims: int, base: float, scaling: RopeScaling = NO_SCALING,
             *, pairs: int | None = None) -> np.ndarray:
    """Per-pair angle per position, float32, as ggml rope_yarn rotates.

    ``dims`` is the rotary width the exponents divide by; ``pairs`` limits
    the table (defaults to ``dims // 2``)."""
    n = dims // 2 if pairs is None else pairs
    j = np.arange(n, dtype=np.float64)
    theta_extrap = np.power(float(base), -2.0 * j / dims).astype(np.float32)
    fs = _f32(scaling.freq_scale)
    theta = fs * theta_extrap
    if scaling.ext_factor != 0.0 and scaling.original_context > 0:
        lo, hi = corr_dims(dims, base, scaling)
        y = (j.astype(np.float32) - _f32(lo)) / max(_f32(0.001), _f32(hi - lo))
        ramp = _f32(1) - np.minimum(_f32(1), np.maximum(_f32(0), y))
        mix = (ramp * _f32(scaling.ext_factor)).astype(np.float32)
        theta = theta * (_f32(1) - mix) + theta_extrap * mix
    return theta.astype(np.float32)


def freqs(dims: int, base: float, scaling: RopeScaling = NO_SCALING,
          *, pairs: int | None = None):
    """The table in the ``mx.fast.rope(freqs=)`` convention (the angle is
    position over freqs), as an ``mx.array``."""
    import mlx.core as mx

    return mx.array(1.0 / inv_freq(dims, base, scaling, pairs=pairs)
                    .astype(np.float64), dtype=mx.float32)


def effective_context(trained: int | None, scaling: RopeScaling) -> int | None:
    """The context length to report and clamp to. A YaRN factor set by the
    run options extends it to ``original_context / freq_scale``, as
    llama.cpp re-adjusts its trained context."""
    if trained is None:
        return None
    if (scaling.type == "yarn" and scaling.user_set
            and scaling.original_context > 0):
        return int(scaling.original_context / np.float32(scaling.freq_scale))
    return trained


# The options of the load in progress. The load entry points set it around
# their body, so the config synthesizers they reach (target, VLM tower, MTP
# head) all see the same options without a parameter at every hop. A drafter
# with its own GGUF resolves from its own header and does not read it.
_ACTIVE: contextvars.ContextVar[RopeOverride] = contextvars.ContextVar(
    "gmlx_rope_override", default=NO_OVERRIDE)


@contextlib.contextmanager
def override_scope(override: RopeOverride | None):
    token = _ACTIVE.set(override or NO_OVERRIDE)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_override() -> RopeOverride:
    return _ACTIVE.get()


def scoped(fn):
    """Run a load entry point under the ``rope_override`` keyword it was
    called with. Without one, an outer scope stays in force."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ov = kwargs.get("rope_override")
        if ov is None:
            return fn(*args, **kwargs)
        with override_scope(ov):
            return fn(*args, **kwargs)
    return wrapper


# Class-built architectures whose rope modules the loader replaces with the
# resolved table after the build (gmlx.load.rope_swap). Their header scaling
# goes through the swap too, so the model class's own reading of it no
# longer decides the rope.
SWAP_ARCHES = frozenset({
    "llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "glm4", "glm4moe",
    "mistral3", "gpt-oss", "seed_oss", "smollm3", "granite", "qwen3next",
    "minimax-m2", "qwen35", "qwen35moe", "qwen3vl", "qwen3vlmoe",
    "hy_v3", "hyv4", "muse-glimmer", "minimax-m3", "falcon-h1",
    "granitehybrid", "ernie4_5-moe", "gemma", "gemma2", "gemma3", "gemma4",
})

# Architectures whose every rope site applies the run options. A load of any
# other architecture with options refuses before it reads the weights,
# rather than running with a rope the options did not ask for.
WIRED_ARCHES: set[str] = {"qwen4exp", "deepseek4", "deepseek41",
                          *SWAP_ARCHES}


def require_wired(arch: str, override: RopeOverride | None) -> None:
    if override is None or override.is_empty() or arch in WIRED_ARCHES:
        return
    raise NotImplementedError(
        f"the rope scaling options are not supported for the {arch} "
        f"architecture; supported: {', '.join(sorted(WIRED_ARCHES)) or 'none'}")


# Config keys the synthesizers add. ``max_position_embeddings`` carries the
# effective context, as llama.cpp rewrites its trained context; the trained
# value stays under its own key.
CONFIG_KEY = "gmlx_rope_scaling"
TRAINED_CONTEXT_KEY = "gmlx_trained_context"


def to_config(config: dict, scaling: RopeScaling) -> None:
    """Record ``scaling`` in a synthesized config and move the context
    length to the effective one. A plain rope leaves the config as is."""
    if not scaling.active:
        return
    config[CONFIG_KEY] = asdict(scaling)
    trained = config.get("max_position_embeddings")
    if trained is not None:
        eff = effective_context(int(trained), scaling)
        if eff != trained:
            config[TRAINED_CONTEXT_KEY] = int(trained)
            config["max_position_embeddings"] = eff


def from_config(config) -> RopeScaling:
    """The scaling a synthesized config carries; plain when it has none."""
    raw = (config.get(CONFIG_KEY) if isinstance(config, dict)
           else getattr(config, CONFIG_KEY, None))
    if not raw:
        return NO_SCALING
    return RopeScaling(**raw)


def cache_suffix(override: RopeOverride | None) -> str:
    """A stable suffix for the name of a cache that outlives one model
    instance (the APC disk tier): empty without options, so default names
    stay as they were. KV made under one scaling is wrong under another."""
    if override is None or override.is_empty():
        return ""
    import hashlib

    text = repr(tuple((f.name, getattr(override, f.name))
                      for f in fields(override)))
    return "#rope-" + hashlib.sha256(text.encode()).hexdigest()[:12]
