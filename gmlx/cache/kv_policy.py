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

# Layer kinds: kv = growing attention KV (quantizable), window =
# size-capped sliding window, state = recurrent state, pool = packs at
# rest via quantize_storage, optout = declares kv_quant_unsupported,
# other = unrecognized (held fp16).
_KINDS = ("kv", "window", "state", "pool", "optout", "other")


def packed_bytes_per_element(bits: int, group_size: int) -> float:
    """Affine-packed KV cost: payload plus one fp16 scale and bias per
    group. 8-bit g64 = 1.0625, 4-bit g32 = 0.625."""
    return bits / 8.0 + 4.0 / group_size


@dataclass(frozen=True)
class KvLayerPlan:
    kind: str
    quantize: bool
    bytes_per_element: float


@dataclass(frozen=True)
class KvQuantPolicy:
    verdict: str                # full | partial | dropped | error
    reason: str | None
    bits: int | None
    group_size: int | None
    mode: str                   # single | batched
    per_layer: tuple = ()
    quantized_kv_start: int = 0
    start_honored: bool = True

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
        return sum(1 for p in self.per_layer if p.kind == "pool")

    def bytes_per_element_vector(self) -> list:
        return [p.bytes_per_element for p in self.per_layer]

    def summary(self) -> str:
        """The clause after the arrow in the canonical [kv] line."""
        if self.verdict == "error":
            return f"error: {self.reason}"
        if self.verdict == "dropped":
            return f"dropped: {self.reason}"
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
            # pool packing is the engagement on pool-bearing stacks; a
            # quantized-0/N headline would read as a no-op
            out = f"{self.n_pool} pooled at rest; " + out
        if self.quantized_kv_start and not self.start_honored:
            out += (f"; quantized_kv_start={self.quantized_kv_start} "
                    "not honored (batch caches quantize from token 0)")
        return out


def _error(reason, bits, group, mode, stack):
    per = tuple(KvLayerPlan("other", False, FP16_BPE) for _ in stack)
    return KvQuantPolicy("error", reason, bits, group, mode, per)


def dropped_policy(reason, bits, group, mode) -> KvQuantPolicy:
    """A bare dropped verdict for paths where no stack was constructed
    (upstream ate the flag before a stack could exist)."""
    return KvQuantPolicy("dropped", reason, bits, group, mode, ())


def _dropped(reason, bits, group, mode, stack, kinds=None):
    per = tuple(
        KvLayerPlan(kinds[i] if kinds else "other", False, FP16_BPE)
        for i in range(len(stack)))
    return KvQuantPolicy("dropped", reason, bits, group, mode, per)


def _classify(c, types):
    if hasattr(c, "quantize_storage"):
        # An opted-out pool (dsv4 indexer: the score kernel reads it in
        # full every step) rides fp16 as plain state, so a list holding
        # only opted-out pools never claims a pool verdict.
        return "pool" if getattr(c, "quantizable", True) else "state"
    if getattr(c, "kv_quant_unsupported", False):
        return "optout"
    if isinstance(c, types["window"]):
        return "window"
    if isinstance(c, types["kv"]):
        return "kv"
    if isinstance(c, types["state"]):
        return "state"
    inner = getattr(c, "caches", None)
    if inner is not None:
        subs = {_classify(s, types) for s in inner}
        # A list with any quantizable KV counts as kv; the side caches
        # ride along fp16 (their cost is in the kv estimate's noise).
        # A pool member (dsv4: CacheList(Rotating, PoolingCache, ...))
        # makes the layer poolable; arming walks back in and packs only
        # the quantizable pools.
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
                            no_kv_reason=None) -> KvQuantPolicy:
    """Resolve the KV quantization policy for one constructed cache stack.

    ``stack`` must be the real stack the engine will run (same
    construction args, max_kv_size included): probing a bare make_cache()
    misses stacks the args reshape. ``mode`` is the batch axis: MTP
    quantizes at B=1 and runs fp16 KV when batched, so engagement and
    memory pricing differ per mode. Reference layer policy is the serve
    rule: quantize growing KV-class layers except the last layer of a
    deep stack (should_quantize_kv_layer); windows, recurrent state, and
    opt-outs stay fp16; pooled caches pack at rest.
    ``can_quantize_kv=False`` models an engine with no KV converter in
    its loop (the CLI MTP rounds): only pooled packing engages, and
    ``no_kv_reason`` names why.
    """
    stack = list(stack or [])
    if kv_bits is None:
        raise ValueError("resolve_kv_quant_policy needs kv_bits set")
    group = kv_group_size

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
    if scheme not in (None, "uniform"):
        return _error(
            f"kv_quant_scheme {scheme!r} unsupported (only 'uniform' "
            "affine is certified)",
            bits, group, mode, stack)
    if key_bits is not None or value_bits is not None:
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
        if (kind == "kv" and can_quantize_kv
                and should_quantize_kv_layer(i, n)):
            per.append(KvLayerPlan("kv", True, packed))
        elif kind == "pool":
            per.append(KvLayerPlan("pool", False, packed))
        else:
            per.append(KvLayerPlan(kind, False, FP16_BPE))

    hetero = any(k in ("window", "state", "optout", "other") for k in kinds)
    verdict = "partial" if hetero else "full"
    # Batch caches quantize at construction; a nonzero start is only
    # honored by the single-stream converter.
    honored = mode == "single" or not quantized_kv_start
    return KvQuantPolicy(verdict, None, bits, group, mode, tuple(per),
                         quantized_kv_start=int(quantized_kv_start or 0),
                         start_honored=honored)


def kv_line(model_id, policy: KvQuantPolicy) -> str:
    """The canonical engagement line, identical across serve, run, chat.
    ``model_id`` may be None on single-model CLI paths."""
    head = f"[kv] {model_id}: " if model_id else "[kv] "
    head += f"kv_bits={policy.bits}"
    if policy.group_size:
        head += f" group={policy.group_size}"
    return f"{head} -> {policy.summary()}"


_HELD_CLASSES: dict = {}


def _held_to_quantized(self):
    raise AttributeError("held fp16 by kv policy")


def hold_fp16(c):
    """Rebind a FRESH cache to a subclass whose to_quantized is hidden,
    so stock hasattr-gated converters leave the layer fp16 (mlx-lm
    maybe_quantize_kv_cache and the mlx-vlm equivalents).

    Held subclasses live in this module, so compat's
    rebind_to_runtime_origin (gated on mlx-lm-origin class identity)
    passes them through unswapped. Accepted: arm_stack serves CLI paths
    whose stacks never cross the serve APC rebind; a held layer reaching
    that seam would quietly skip APC for its stack."""
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


def _arm_pools(c, bits, group) -> int:
    """Arm at-rest packing on every quantizable pool under ``c``,
    recursing into CacheList (dsv4 nests pools beside a window). The
    quantizable gate keeps opted-out pools (dsv4 indexer) fp16."""
    inner = getattr(c, "caches", None)
    if inner is not None:
        return sum(_arm_pools(s, bits, group) for s in inner)
    if hasattr(c, "quantize_storage") and getattr(c, "quantizable", True):
        c.quantize_storage(group_size=group, bits=bits)
        return 1
    return 0


def arm_stack(stack, policy: KvQuantPolicy, hold=True):
    """Conform a freshly built stack to the policy in place: fp16 layers
    lose their to_quantized (converters skip them, including rotating
    windows whose to_quantized raises NYI), pooled layers arm at-rest
    packing (recursing into CacheList for dsv4-shaped layers).
    ``hold=False`` arms pools only, for engines that run no converter.
    Holds stay top-level: mlx-lm's converter iterates only the top of
    the stack and CacheList exposes no to_quantized, so nested members
    never meet a converter."""
    if policy.verdict not in ("full", "partial"):
        return stack
    for i, plan in enumerate(policy.per_layer):
        if plan.kind == "pool":
            _arm_pools(stack[i], policy.bits, policy.group_size)
        if hold and not plan.quantize:
            stack[i] = hold_fp16(stack[i])
    return stack
