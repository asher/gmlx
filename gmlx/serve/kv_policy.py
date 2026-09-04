"""Serve-side KV quantization policy: resolved once per model at load.

The residency build calls resolve_for_load inside the per-model env
window. The result rides on the entry and the ResponseGenerator, so
the memory preflight, /v1/models, and the engagement log read the same
object. An error verdict fails residency with the reason.
"""

import logging
import os
from dataclasses import dataclass

from gmlx.cache.kv_policy import (KvQuantPolicy, dropped_policy, kv_line,
                                  off_policy, resolve_kv_quant_policy)

_log = logging.getLogger(__name__)

RG_ATTR = "_gmlx_kv_policy"


class KvPolicyError(RuntimeError):
    """kv quantization config cannot run. The model fails residency."""


@dataclass(frozen=True)
class ServeKvPolicy:
    single: KvQuantPolicy
    batched: KvQuantPolicy

    def pricing_vector(self):
        """Per-layer bytes-per-element for admission (batched mode)."""
        return self.batched.bytes_per_element_vector()

    def region_vector(self):
        """Per-layer fixed resident regions for admission (batched mode)."""
        return self.batched.regions_vector()

    def step_vector(self):
        """Per-layer record slab sizes for admission (batched mode)."""
        return self.batched.steps_vector()

    def to_json(self) -> dict:
        """The /v1/models kv_quant field. verdict_batched is separate:
        MTP models quantize at B=1 and run fp16 when batched."""
        s, b = self.single, self.batched
        out = {
            "scheme": s.scheme,
            "bits": s.bits,
            "group_size": s.group_size,
            "layers_quantized": s.n_quant,
            "layers_fp16": len(s.per_layer) - s.n_quant,
            "verdict": s.verdict,
            "verdict_batched": b.verdict,
        }
        if s.scheme != "uniform":
            # bits is the key width under a split-width scheme, so the
            # value width is always reported beside it.
            out["value_bits"] = (s.bits if s.value_bits is None
                                 else s.value_bits)
            out["tail_tokens"] = s.tail_tokens
        if s.reason:
            out["reason"] = s.reason
        if b.verdict != s.verdict and b.reason:
            out["batched_reason"] = b.reason
        return out


def _probe_stack(model):
    lm = getattr(model, "language_model", None) or model
    make = getattr(lm, "make_cache", None)
    if callable(make):
        return make()
    from mlx_vlm.models.cache import KVCache

    return [KVCache() for _ in lm.layers]


def _config_head_dim(model):
    from .mem_preflight import _get, _lm_config

    c = _lm_config(model)
    head_dim = _get(c, "head_dim")
    if not head_dim:
        heads = _get(c, "num_attention_heads")
        hidden = _get(c, "hidden_size")
        if heads and hidden:
            head_dim = hidden // heads
    return head_dim if isinstance(head_dim, int) and head_dim > 0 else None


def _serve_tail_tokens() -> int:
    """The kvarn precision tail from the per-model load window
    (KV_TAIL_TOKENS); upstream carries no attribute for it."""
    from gmlx.cache.kvarn_cache import KVARN_DEFAULT_TAIL

    val = os.environ.get("KV_TAIL_TOKENS")
    try:
        return KVARN_DEFAULT_TAIL if val in (None, "") else int(val)
    except (TypeError, ValueError):
        return KVARN_DEFAULT_TAIL


def _load_window_scheme(rg) -> str:
    """The scheme this model loads under, and rg agrees with it on return.

    Upstream freezes runtime.config.kv_quant_scheme from the process env
    at server start, and app.py's ``cfg.kv_quant_scheme or
    get_kv_quant_scheme()`` cannot fall through it -- the default is the
    non-empty string "uniform". So a per-model ``load:`` key never
    reaches the generator on its own. The env window is the per-model
    truth, exactly as it is for KV_BITS; rg is corrected here because
    upstream's batch construction gates ``_make_cache`` on the attribute,
    not on the policy.
    """
    scheme = (os.environ.get("KV_QUANT_SCHEME")
              or getattr(rg, "kv_quant_scheme", None)
              or "uniform").strip().lower()
    if getattr(rg, "kv_quant_scheme", None) != scheme:
        try:
            rg.kv_quant_scheme = scheme
        except Exception:
            _log.warning("[kv] cannot set kv_quant_scheme on the generator; "
                         "batch caches will build %r",
                         getattr(rg, "kv_quant_scheme", None), exc_info=True)
    return scheme


def resolve_for_load(rg, model_id: str):
    """Resolve both batch modes for a freshly built ResponseGenerator.

    Returns the ServeKvPolicy (also stamped on rg), or None when kv
    quantization is not requested. Raises KvPolicyError on an error
    verdict. Must run inside the model's env window: KV_BITS in
    os.environ distinguishes off from upstream's silent qat drop.
    """
    requested = os.environ.get("KV_BITS")
    try:
        req_val = float(requested) if requested else 0.0
    except ValueError:
        # Upstream parses the same var, so a live rg.kv_bits never
        # coexists with an unparseable KV_BITS.
        raise KvPolicyError(
            f"[kv] {model_id}: KV_BITS={requested!r} is not a number")
    bits = getattr(rg, "kv_bits", None)
    scheme = _load_window_scheme(rg)
    if bits is None and scheme != "kvarn":
        if req_val:
            # get_quantized_kv_bits drops the flag for "qat" model ids.
            reason = ("model id marked quantization-aware (qat); "
                      "upstream drops KV quantization")
            b = int(req_val)
            pol = ServeKvPolicy(
                dropped_policy(reason, b, rg.kv_group_size, "single"),
                dropped_policy(reason, b, rg.kv_group_size, "batched"))
            setattr(rg, RG_ATTR, pol)
            # Warm merges read the model stamp and must stay float
            # here: upstream dropped the flag and live caches run fp16.
            _stamp_model(rg, pol, model_id)
            _log.warning(kv_line(model_id, pol.single))
            return pol
        # Off is stamped too: request-time readers (the B=1 MTP spec
        # cache) must not fall back to another model's boot env.
        _stamp_model(rg, ServeKvPolicy(off_policy("single"),
                                       off_policy("batched")), model_id)
        return None

    mtp = bool(getattr(rg, "draft_model_path", None)
               or os.environ.get("MLX_VLM_GGUF_SPECULATIVE") == "1")
    stack = _probe_stack(rg.model)
    kw = dict(
        kv_bits=bits,
        kv_group_size=getattr(rg, "kv_group_size", 64),
        quantized_kv_start=getattr(rg, "quantized_kv_start", 0),
        scheme=scheme,
        key_bits=getattr(rg, "kv_key_bits", None),
        value_bits=getattr(rg, "kv_value_bits", None),
        mtp=mtp,
        head_dim=_config_head_dim(rg.model),
    )
    if scheme == "kvarn":
        # kvarn owns its widths: KV_BITS from the window (default 6),
        # independent of upstream's kv_bits parse and its qat drop;
        # GMLX_KVARN_BITS may split them. rotating_window stays None:
        # serve's MAX_KV_SIZE only caps the request context budget, it
        # never builds a rotating stack.
        from gmlx.cache.kvarn_cache import kvarn_resolve_kwargs

        kw["key_bits"] = None
        kw.update(kvarn_resolve_kwargs(
            rg.model, int(req_val) if req_val else None,
            tail_tokens=_serve_tail_tokens()))
        # rg carries upstream's 5000 default, which affine honors and
        # kvarn never can; only an explicit request is worth a "not
        # honored" note on the line.
        kw["quantized_kv_start"] = int(
            os.environ.get("QUANTIZED_KV_START") or 0)
    pol = ServeKvPolicy(
        resolve_kv_quant_policy(stack, mode="single", **kw),
        resolve_kv_quant_policy(_probe_stack(rg.model), mode="batched",
                                **kw),
    )
    if pol.single.verdict == "error" or pol.batched.verdict == "error":
        bad = pol.single if pol.single.verdict == "error" else pol.batched
        raise KvPolicyError(kv_line(model_id, bad))
    setattr(rg, RG_ATTR, pol)
    _stamp_model(rg, pol, model_id)
    _log.info(kv_line(model_id, pol.single))
    if pol.batched.verdict != pol.single.verdict:
        _log.info(kv_line(model_id, pol.batched)
                  + " (when batched)")
    return pol


def _stamp_model(rg, pol, model_id):
    """The batch worker reads the policy off batch.model. The residency
    proxy does not cross threads. Model and language model both carry
    it: the cache builders disagree on which one they are handed."""
    try:
        setattr(rg.model, RG_ATTR, pol)
        lm = getattr(rg.model, "language_model", None)
        if lm is not None:
            setattr(lm, RG_ATTR, pol)
    except Exception:
        _log.warning("[kv] %s: model stamp failed; warm merges see no "
                     "policy (stay fp16)", model_id, exc_info=True)


def pricing_vector(rg, num_layers: int):
    """The admission bytes-per-element vector for rg, or None for
    uniform pricing. Length-checked against the config layer count."""
    pol = getattr(rg, RG_ATTR, None)
    if pol is None:
        return None
    vec = pol.pricing_vector()
    return vec if len(vec) == num_layers else None


def region_vector(rg, num_layers: int):
    """The admission fixed-region vector for rg, or None when the scheme
    holds no fixed buffers. Length-checked like pricing_vector."""
    pol = getattr(rg, RG_ATTR, None)
    if pol is None:
        return None
    vec = pol.region_vector()
    if len(vec) != num_layers or not any(vec):
        return None
    return vec


def step_vector(rg, num_layers: int):
    """The admission slab-step vector for rg, or None when the scheme
    grows per token. Length-checked like pricing_vector."""
    pol = getattr(rg, RG_ATTR, None)
    if pol is None:
        return None
    vec = pol.step_vector()
    if len(vec) != num_layers or not any(vec):
        return None
    return vec
