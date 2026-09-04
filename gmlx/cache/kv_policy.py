"""KV quantization policy resolver.

One verdict per (constructed cache stack, kv params, batch mode). Every
entry point that owns a stack (serve residency, CLI run, chat REPL, MTP
spec-cache build) resolves here and acts on the verdict, so engagement,
logging, and memory pricing all read the same object.
"""

from dataclasses import dataclass

VALID_BITS = (2, 3, 4, 6, 8)
VALID_GROUPS = (32, 64, 128)
FP16_BPE = 2.0

# Schemes: uniform = affine per-group scale/bias, kvarn =
# variance-normalized plus Hadamard rotation (gmlx/cache/kvarn_cache.py).
SCHEMES = ("uniform", "kvarn")

# kvarn record geometry. Mirrors the cache module's constants, which stay
# authoritative; kv_policy keeps them local so the resolver imports no
# kernel-backed module until a kvarn boot actually needs one.
KVARN_GROUP = 128
KVARN_SINK = 128
KVARN_WIDTHS = (2, 3, 4, 5, 6, 8)

# Layer kinds: kv = growing attention KV (quantizable), window =
# size-capped sliding window, state = recurrent state, pool = packs at
# rest via quantize_storage, optout = declares kv_quant_unsupported,
# other = unrecognized (held fp16).
_KINDS = ("kv", "window", "state", "pool", "optout", "other")


def packed_bytes_per_element(bits: int, group_size: int) -> float:
    """Affine-packed KV cost: payload plus one fp16 scale and bias per
    group. 8-bit g64 = 1.0625, 4-bit g32 = 0.625."""
    return bits / 8.0 + 4.0 / group_size


def kvarn_bytes_per_element(bits: int, value_bits=None) -> float:
    """kvarn record cost: codes at the mean K/V width plus one fp16
    Sinkhorn axes triplet per 128-token group. 6-bit = 0.796875.

    The fp16 sink, horizon and tail are fixed-size regions, not a
    per-token cost; kvarn_fixed_tokens prices them separately.
    """
    v = bits if value_bits is None else value_bits
    return (bits + v) / 16.0 + 3.0 * FP16_BPE / KVARN_GROUP


def kvarn_fixed_tokens(tail_tokens) -> int:
    """fp16 rows a kvarn layer holds whatever the context length: the sink
    stage with its spare group, the horizon group, and the tail buffer with
    its slack. Priced as a windowed region, so a short context pays for
    what it actually allocates."""
    from gmlx.cache.kvarn_cache import GROUP, KVarNKVCache

    tail = int(tail_tokens or 0)
    rows = KVARN_SINK + GROUP + GROUP
    return rows + (tail + KVarNKVCache.tail_slack if tail else 1)


@dataclass(frozen=True)
class KvLayerPlan:
    kind: str
    quantize: bool
    bytes_per_element: float
    pools: int = 0              # quantizable pools under the layer, any kind
    # Resident regions beyond the per-token cost, as (tokens, bpe) pairs
    # charged over the first `tokens` tokens. kvarn's fp16 buffers.
    regions: tuple = ()


@dataclass(frozen=True)
class KvQuantPolicy:
    verdict: str                # full | partial | dropped | error | off
    reason: str | None
    bits: int | None
    group_size: int | None
    mode: str                   # single | batched
    per_layer: tuple = ()
    quantized_kv_start: int = 0
    start_honored: bool = True
    scheme: str = "uniform"
    value_bits: int | None = None   # kvarn value width; None = same as bits
    tail_tokens: int | None = None  # kvarn fp16 precision tail
    rotating_window: int | None = None
    pool_bits: int | None = None    # affine width pools pack at, None = none

    @property
    def n_quant(self):
        return sum(1 for p in self.per_layer if p.quantize)

    @property
    def n_held(self):
        return sum(1 for p in self.per_layer
                   if p.kind == "kv" and not p.quantize)

    @property
    def n_window(self):
        return sum(1 for p in self.per_layer if p.kind == "window")

    @property
    def n_state(self):
        return sum(1 for p in self.per_layer if p.kind == "state")

    @property
    def n_pool(self):
        return sum(p.pools for p in self.per_layer)

    def bytes_per_element_vector(self) -> list:
        return [p.bytes_per_element for p in self.per_layer]

    def regions_vector(self) -> list:
        return [p.regions for p in self.per_layer]

    @property
    def width_label(self) -> str:
        """The width clause of the [kv] line. Affine wording is pinned by
        tests and the e2e checks; kvarn renders its own."""
        if self.scheme != "uniform":
            v = self.bits if self.value_bits is None else self.value_bits
            width = (f"{self.scheme}{self.bits}" if v == self.bits
                     else f"{self.scheme} k{self.bits} v{v}")
            if self.tail_tokens is not None:
                width += f" tail={self.tail_tokens}"
            if self.rotating_window:
                width += f" window={self.rotating_window}"
            return width
        head = f"kv_bits={self.bits}"
        if self.group_size:
            head += f" group={self.group_size}"
        return head

    def summary(self) -> str:
        """The clause after the arrow in the canonical [kv] line."""
        if self.verdict == "error":
            return f"error: {self.reason}"
        if self.verdict == "dropped":
            return f"dropped: {self.reason}"
        if self.verdict == "off":
            return "off"
        n = len(self.per_layer)
        parts = [f"quantized {self.n_quant}/{n} attn layers"]
        held = []
        if self.n_held:
            held.append(f"{self.n_held} held fp16")
        if self.n_window:
            held.append(f"{self.n_window} sliding-window fp16")
        if self.n_state:
            held.append(f"{self.n_state} recurrent-state fp16")
        if held:
            parts.append(f"({', '.join(held)})")
        out = " ".join(parts)
        if self.n_pool:
            # pools lead: quantized 0/N alone reads as a no-op
            out = f"{self.n_pool} pooled at rest; " + out
        if self.quantized_kv_start and not self.start_honored:
            out += (f"; quantized_kv_start={self.quantized_kv_start} "
                    "not honored (batch caches quantize from token 0)")
        return out


def _error(reason, bits, group, mode, stack, **kw):
    per = tuple(KvLayerPlan("other", False, FP16_BPE) for _ in stack)
    return KvQuantPolicy("error", reason, bits, group, mode, per, **kw)


def dropped_policy(reason, bits, group, mode, **kw) -> KvQuantPolicy:
    """A dropped verdict for paths where no stack was constructed."""
    return KvQuantPolicy("dropped", reason, bits, group, mode, (), **kw)


def off_policy(mode) -> KvQuantPolicy:
    """The verdict stamped on a model loaded without KV quantization, so
    request-time readers can tell "off" from "never resolved"."""
    return KvQuantPolicy("off", "not requested", None, None, mode, ())


def _dropped(reason, bits, group, mode, stack, kinds=None, **kw):
    per = tuple(
        KvLayerPlan(kinds[i] if kinds else "other", False, FP16_BPE)
        for i in range(len(stack)))
    return KvQuantPolicy("dropped", reason, bits, group, mode, per, **kw)


def _classify(c, types):
    if hasattr(c, "quantize_storage"):
        # An opted-out pool classifies as state, so a list of only
        # opted-out pools gets no pool verdict.
        return "pool" if getattr(c, "quantizable", True) else "state"
    if getattr(c, "kv_quant_unsupported", False):
        return "optout"
    if getattr(c, "kv_quant_scheme", None) == "kvarn":
        # An already-converted kvarn layer, rotating subclass included.
        # No live path re-resolves over one today (every call site
        # resolves a freshly built stack); without this arm a future one
        # would report "no quantizable layers", since the kvarn classes
        # derive from the base cache, not KVCache.
        return "kv"
    if isinstance(c, types["window"]):
        return "window"
    if isinstance(c, types["kv"]):
        return "kv"
    if isinstance(c, types["state"]):
        return "state"
    inner = getattr(c, "caches", None)
    if inner is not None:
        subs = {_classify(s, types) for s in inner}
        # A kv member rules a mixed list. A pool member marks the layer
        # poolable. Side caches stay fp16 and are not priced.
        if "optout" in subs:
            return "optout"
        if "kv" in subs:
            return "kv"
        if "pool" in subs:
            return "pool"
        if subs <= {"window"}:
            return "window"
        return "state"
    return "other"


def _cache_kind_types():
    from gmlx.cache.compat import cache_types

    def _get(*names):
        out = []
        for n in names:
            try:
                out.extend(cache_types(n))
            except AttributeError:
                pass
        return tuple(out)

    return {
        "kv": _get("KVCache", "ChunkedKVCache", "SimpleKVCache",
                   "BatchKVCache", "QuantizedKVCache",
                   "BatchQuantizedKVCache"),
        "window": _get("RotatingKVCache", "BatchRotatingKVCache",
                       "BufferedRotatingKVCache"),
        "state": _get("ArraysCache", "MambaCache", "ConcatenateKVCache"),
    }


def resolve_kv_quant_policy(stack, *, kv_bits, kv_group_size=64,
                            quantized_kv_start=0, scheme=None,
                            key_bits=None, value_bits=None,
                            mode="single", mtp=False, max_kv_size=None,
                            head_dim=None, can_quantize_kv=True,
                            no_kv_reason=None, tail_tokens=None,
                            rotating_window=None,
                            scheme_reason=None) -> KvQuantPolicy:
    """Resolve the KV quantization policy for one constructed cache stack.

    stack must be the stack the engine will run, built with the same
    args. A bare make_cache() probe misses what max_kv_size builds.
    mode is the batch axis: MTP quantizes at B=1 and runs fp16 KV when
    batched. Layer rule: quantize growing KV layers except the last
    layer of a deep stack. Recurrent state and opt-outs stay fp16, and
    so do windows unless the scheme owns them. Pools pack at rest.
    can_quantize_kv=False limits engagement to pooled packing and
    no_kv_reason names why.

    scheme picks the packing: "uniform" is affine (bits/group_size),
    "kvarn" is variance-normalized plus rotation (bits/value_bits split
    widths, tail_tokens, and rotating_window for the windows it owns).
    scheme_reason carries a model-shaped decline the caller resolved.
    """
    stack = list(stack or [])
    if kv_bits is None:
        raise ValueError("resolve_kv_quant_policy needs kv_bits set")
    group = kv_group_size
    scheme = (scheme or "uniform").lower()
    if scheme not in SCHEMES:
        return _error(
            f"kv_quant_scheme {scheme!r} unsupported (choose from "
            f"{', '.join(SCHEMES)})",
            None, group, mode, stack)
    if key_bits is not None:
        return _error("split key/value KV bits are unsupported",
                      None, group, mode, stack)
    if scheme == "kvarn":
        return _resolve_kvarn(
            stack, kv_bits=kv_bits, value_bits=value_bits, mode=mode,
            mtp=mtp, head_dim=head_dim, tail_tokens=tail_tokens,
            rotating_window=rotating_window, group=group,
            quantized_kv_start=quantized_kv_start,
            can_quantize_kv=can_quantize_kv, no_kv_reason=no_kv_reason,
            scheme_reason=scheme_reason)

    fb = float(kv_bits)
    if fb != int(fb) or int(fb) not in VALID_BITS:
        return _error(
            f"kv_bits {kv_bits} unsupported (affine widths: "
            f"{', '.join(map(str, VALID_BITS))})",
            None, group, mode, stack)
    bits = int(fb)
    if group not in VALID_GROUPS:
        return _error(
            f"kv_group_size {group} unsupported (choose from "
            f"{', '.join(map(str, VALID_GROUPS))})",
            bits, None, mode, stack)
    if head_dim is not None and head_dim % group:
        return _error(
            f"head_dim {head_dim} not divisible by kv_group_size {group}",
            bits, group, mode, stack)
    if value_bits is not None:
        return _error(
            "split key/value KV bits are unsupported",
            bits, group, mode, stack)
    if not stack:
        return _error("empty cache stack", bits, group, mode, stack)

    types = _cache_kind_types()
    kinds = [_classify(c, types) for c in stack]

    if mtp and mode == "batched":
        return _dropped(
            "MTP batch rollback cannot trim packed KV; fp16 when batched",
            bits, group, mode, stack, kinds)

    n = len(kinds)
    quantizable = [k for k in kinds
                   if (k == "kv" and can_quantize_kv) or k == "pool"]
    if not quantizable:
        if max_kv_size is not None and "window" in kinds:
            return _error(
                "max_kv_size builds sliding-window caches that cannot "
                "quantize; drop kv_bits or max_kv_size",
                bits, group, mode, stack)
        return _dropped(
            no_kv_reason if (not can_quantize_kv and no_kv_reason)
            else "cache stack has no quantizable layers",
            bits, group, mode, stack, kinds)

    from mlx_vlm.models.cache import should_quantize_kv_layer

    packed = packed_bytes_per_element(bits, group)
    per = []
    for i, kind in enumerate(kinds):
        pools = sum(1 for _ in _quantizable_pools(stack[i]))
        if (kind == "kv" and can_quantize_kv
                and should_quantize_kv_layer(i, n)):
            per.append(KvLayerPlan("kv", True, packed, pools))
        elif kind == "pool":
            per.append(KvLayerPlan("pool", False, packed, pools))
        else:
            per.append(KvLayerPlan(kind, False, FP16_BPE, pools))

    hetero = any(k in ("window", "state", "optout", "other") for k in kinds)
    verdict = "partial" if hetero else "full"
    # Batch caches quantize at construction. Only the single-stream
    # converter honors a nonzero start.
    honored = mode == "single" or not quantized_kv_start
    return KvQuantPolicy(verdict, None, bits, group, mode, tuple(per),
                         quantized_kv_start=int(quantized_kv_start or 0),
                         start_honored=honored, pool_bits=bits)


def _kvarn_owns(c, kind, rotating_window, batched=False):
    """True when kvarn takes this layer. Rotating layers convert only for
    the window the caller built them at: a model's own SWA stack keeps its
    windows fp16, while a --max-kv-size stack on a make_cache-less model
    converts throughout. In batched mode any growing KV layer counts --
    serve's batch seam constructs BatchKVarNKVCache itself, so the class
    on the probe stack says nothing about what it will build."""
    from gmlx.cache.compat import cache_types
    from gmlx.cache.kvarn_cache import convertible_kv_types

    if kind == "kv":
        # A CacheList layer classifies "kv" through its members, but the
        # batch seam has no kvarn container for it.
        leaf = getattr(c, "caches", None) is None
        return ((batched and leaf)
                or type(c) in convertible_kv_types()
                or getattr(c, "kv_quant_scheme", None) == "kvarn")
    if kind != "window" or not rotating_window:
        return False
    return (type(c) in cache_types("RotatingKVCache")
            and int(getattr(c, "max_size", 0) or 0) == int(rotating_window))


def _resolve_kvarn(stack, *, kv_bits, value_bits, mode, mtp, head_dim,
                   tail_tokens, rotating_window, group,
                   quantized_kv_start, can_quantize_kv, no_kv_reason,
                   scheme_reason) -> KvQuantPolicy:
    """The kvarn arm of resolve_kv_quant_policy."""
    from gmlx.cache.kvarn_cache import HEAD_DIMS, ensure_registered

    k_bits = int(kv_bits) if kv_bits is not None else 6
    v_bits = k_bits if value_bits is None else int(value_bits)
    tail = 1024 if tail_tokens is None else int(tail_tokens)
    extra = dict(scheme="kvarn", value_bits=v_bits, tail_tokens=tail,
                 rotating_window=rotating_window)

    def err(reason):
        return _error(reason, k_bits, None, mode, stack, **extra)

    if k_bits not in KVARN_WIDTHS or v_bits not in KVARN_WIDTHS:
        return err("kvarn bits must be one of "
                   f"{', '.join(map(str, KVARN_WIDTHS))}")
    if tail < 0 or tail % KVARN_GROUP:
        return err(f"kv_tail_tokens must be a multiple of {KVARN_GROUP} "
                   "(0 disables)")
    if rotating_window is not None:
        floor = KVARN_GROUP + max(tail, KVARN_GROUP) + KVARN_GROUP
        if int(rotating_window) < floor:
            return err(
                f"max_kv_size {rotating_window} is below the kvarn window "
                f"floor ({floor} = sink {KVARN_GROUP} + tail "
                f"{max(tail, KVARN_GROUP)} + {KVARN_GROUP}); raise it or "
                "lower kv_tail_tokens")
    if not stack:
        return err("empty cache stack")

    def drop(reason, kinds=None):
        return _dropped(reason, k_bits, None, mode, stack, kinds, **extra)

    if scheme_reason:
        return drop(scheme_reason)
    if head_dim is not None and int(head_dim) not in HEAD_DIMS:
        return drop(f"head_dim {head_dim} (kvarn supports "
                    f"{'/'.join(map(str, HEAD_DIMS))})")

    ensure_registered()
    types = _cache_kind_types()
    kinds = [_classify(c, types) for c in stack]

    if mtp and mode == "batched":
        return drop(
            "MTP batch rollback cannot trim packed KV; fp16 when batched",
            kinds)

    n = len(kinds)
    owns = [can_quantize_kv and _kvarn_owns(stack[i], kinds[i],
                                            rotating_window,
                                            mode == "batched")
            for i in range(n)]
    # kvarn packs records, not pool storage; a pool arms only when the
    # requested width is also a valid affine width.
    pool_bits = k_bits if k_bits in VALID_BITS else None
    if not any(owns) and not (pool_bits and "pool" in kinds):
        return drop(
            no_kv_reason if (not can_quantize_kv and no_kv_reason)
            else "cache stack has no kvarn-convertible layers", kinds)

    from mlx_vlm.models.cache import should_quantize_kv_layer

    record = kvarn_bytes_per_element(k_bits, v_bits)
    regions = ((kvarn_fixed_tokens(tail), FP16_BPE),)
    per = []
    for i, kind in enumerate(kinds):
        pools = (sum(1 for _ in _quantizable_pools(stack[i]))
                 if pool_bits else 0)
        if owns[i] and should_quantize_kv_layer(i, n):
            # A converted window is a kvarn record like any other; the
            # summary counts it as quantized, not as held fp16.
            per.append(KvLayerPlan("kv", True, record, pools, regions))
        elif kind == "pool":
            per.append(KvLayerPlan("pool", False,
                                   packed_bytes_per_element(k_bits, group)
                                   if pool_bits else FP16_BPE, pools))
        else:
            per.append(KvLayerPlan(kind, False, FP16_BPE, pools))

    # Same reading as the affine arm: the carve-out alone is still a full
    # application. Only a layer the scheme cannot take makes it partial.
    hetero = any(not owns[i] and kinds[i] != "pool" for i in range(n))
    honored = mode == "single" or not quantized_kv_start
    return KvQuantPolicy("partial" if hetero else "full", None, k_bits,
                         None, mode, tuple(per),
                         quantized_kv_start=int(quantized_kv_start or 0),
                         start_honored=honored, pool_bits=pool_bits,
                         **extra)


def kv_line(model_id, policy: KvQuantPolicy) -> str:
    """The canonical engagement line, identical across serve, run, chat.
    ``model_id`` may be None on single-model CLI paths."""
    head = f"[kv] {model_id}: " if model_id else "[kv] "
    return f"{head}{policy.width_label} -> {policy.summary()}"


def resolve_and_report(stack, *, model_id=None, **kwargs) -> KvQuantPolicy:
    """Resolve, print the canonical [kv] line to stderr, exit 2 on error.
    The caller handles a dropped verdict."""
    import sys

    policy = resolve_kv_quant_policy(stack, **kwargs)
    print(kv_line(model_id, policy), file=sys.stderr)
    if policy.verdict == "error":
        raise SystemExit(2)
    return policy


_HELD_CLASSES: dict = {}


def _held_to_quantized(self):
    raise AttributeError("held fp16 by kv policy")


def hold_fp16(c):
    """Rebind a fresh cache to a subclass that hides to_quantized, so
    hasattr-gated converters leave the layer fp16. Held layers must not
    cross the serve APC rebind: rebind_to_runtime_origin passes them
    through and APC silently skips the stack."""
    if not hasattr(c, "to_quantized"):
        return c
    cls = type(c)
    held = _HELD_CLASSES.get(cls)
    if held is None:
        held = type("Fp16Held" + cls.__name__, (cls,),
                    {"to_quantized": property(_held_to_quantized)})
        _HELD_CLASSES[cls] = held
    out = held.__new__(held)
    out.__dict__.update(c.__dict__)
    return out


def _quantizable_pools(c):
    """Yield the quantizable pools under c, recursing into cache lists.
    Opted-out pools are skipped."""
    inner = getattr(c, "caches", None)
    if inner is not None:
        for s in inner:
            yield from _quantizable_pools(s)
    elif hasattr(c, "quantize_storage") and getattr(c, "quantizable", True):
        yield c


def _arm_pools(c, bits, group) -> int:
    n = 0
    for p in _quantizable_pools(c):
        p.quantize_storage(group_size=group, bits=bits)
        n += 1
    return n


def _convert_leaf(c, kind, policy: KvQuantPolicy):
    if policy.scheme != "kvarn":
        return c.to_quantized(group_size=policy.group_size,
                              bits=policy.bits)
    from gmlx.cache.kvarn_cache import KVarNKVCache, KVarNRotatingKVCache

    from gmlx.cache.kvarn_sdpa import install_kvarn_sdpa

    v = policy.bits if policy.value_bits is None else policy.value_bits
    cls = KVarNRotatingKVCache if kind == "window" else KVarNKVCache
    out = cls.from_cache(c, k_bits=policy.bits, v_bits=v,
                         tail_tokens=policy.tail_tokens or 0)
    install_kvarn_sdpa()
    return out


def quantize_kv_members(c, policy: KvQuantPolicy):
    """Convert the growing KV caches under c per the policy's scheme,
    descending into cache lists. Returns (cache, n). State and opt-out
    members stay fp16, and so do windows the scheme does not own."""
    types = _cache_kind_types()
    kind = _classify(c, types)
    if kind not in ("kv", "window"):
        return c, 0
    inner = getattr(c, "caches", None)
    if inner is None:
        if getattr(c, "kv_quant_scheme", None) is not None:
            return c, 0        # already converted
        if policy.scheme == "kvarn":
            if not _kvarn_owns(c, kind, policy.rotating_window):
                return c, 0
        elif kind != "kv":
            return c, 0
        return _convert_leaf(c, kind, policy), 1
    subs, n = [], 0
    for s in inner:
        sub, k = quantize_kv_members(s, policy)
        subs.append(sub)
        n += k
    c.caches = tuple(subs)
    return c, n


def arm_stack(stack, policy: KvQuantPolicy, hold=True) -> int:
    """Conform a freshly built stack to the policy in place. Returns the
    number of pools armed. Held fp16 layers lose to_quantized so
    converters skip them. Quantizable pools arm at-rest packing on every
    layer, not only pool-kind ones: a kv member rules a mixed list, so
    its pools would otherwise stay fp16. hold=False arms pools only.
    Holds stay top-level: converters iterate only the top of the
    stack. Under a non-affine scheme nothing duck-types to_quantized, so
    holds are skipped: rebinding the class there would only hide the
    layer from the scheme's own converter."""
    if policy.verdict not in ("full", "partial"):
        return 0
    armed = 0
    for i, plan in enumerate(policy.per_layer):
        if policy.pool_bits:
            armed += _arm_pools(stack[i], policy.pool_bits,
                                policy.group_size or 64)
        if hold and policy.scheme == "uniform" and not plan.quantize:
            stack[i] = hold_fp16(stack[i])
    return armed


def quantize_stack(stack, policy: KvQuantPolicy) -> tuple:
    """Arm pools and convert the planned layers now, in place. Returns
    (pools armed, caches converted). Paths with a later kv-kwargs
    converter use arm_stack with holds instead."""
    if policy.verdict not in ("full", "partial"):
        return 0, 0
    armed = arm_stack(stack, policy, hold=False)
    n = 0
    for i, plan in enumerate(policy.per_layer):
        if plan.quantize:
            stack[i], k = quantize_kv_members(stack[i], policy)
            n += k
    return armed, n
