"""Serve-side KV quantization policy: resolved once per model at load.

The residency build calls resolve_for_load inside the per-model env
window; the result rides on the entry and the ResponseGenerator so the
memory preflight, /v1/models, and the engagement log all read the same
object. An error verdict raises, failing residency with the reason.
"""

import logging
import os
from dataclasses import dataclass

from gmlx.cache.kv_policy import (KvQuantPolicy, dropped_policy, kv_line,
                                  resolve_kv_quant_policy)

_log = logging.getLogger(__name__)

RG_ATTR = "_gmlx_kv_policy"


class KvPolicyError(RuntimeError):
    """kv quantization config cannot run; the model fails residency."""


@dataclass(frozen=True)
class ServeKvPolicy:
    single: KvQuantPolicy
    batched: KvQuantPolicy

    def pricing_vector(self):
        """Per-layer bytes-per-element for admission: the batched mode,
        the state a concurrent server actually runs in."""
        return self.batched.bytes_per_element_vector()

    def to_json(self) -> dict:
        """The /v1/models kv_quant field. verdict_batched exists because
        MTP models quantize at B=1 and run fp16 when batched; a single
        field would assert a number true only while the server is idle."""
        s, b = self.single, self.batched
        out = {
            "bits": s.bits,
            "group_size": s.group_size,
            "layers_quantized": s.n_quant,
            "layers_fp16": len(s.per_layer) - s.n_quant,
            "verdict": s.verdict,
            "verdict_batched": b.verdict,
        }
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
        # Refuse the load instead of raising bare mid-build; upstream
        # parses the same var, so a live rg.kv_bits never coexists with
        # an unparseable KV_BITS.
        raise KvPolicyError(
            f"[kv] {model_id}: KV_BITS={requested!r} is not a number")
    bits = getattr(rg, "kv_bits", None)
    if bits is None:
        if req_val:
            # get_quantized_kv_bits drops the flag for "qat" model ids.
            reason = ("model id marked quantization-aware (qat); "
                      "upstream drops KV quantization")
            b = int(req_val)
            pol = ServeKvPolicy(
                dropped_policy(reason, b, rg.kv_group_size, "single"),
                dropped_policy(reason, b, rg.kv_group_size, "batched"))
            setattr(rg, RG_ATTR, pol)
            # Model stamp too: warm APC merges must stay float here even
            # though KV_BITS sits in the environment (upstream dropped
            # the flag, so live caches run fp16).
            try:
                setattr(rg.model, RG_ATTR, pol)
            except Exception:
                pass
            _log.warning(kv_line(model_id, pol.single))
            return pol
        return None

    mtp = bool(getattr(rg, "draft_model_path", None)
               or os.environ.get("MLX_VLM_GGUF_SPECULATIVE") == "1")
    stack = _probe_stack(rg.model)
    kw = dict(
        kv_bits=bits,
        kv_group_size=getattr(rg, "kv_group_size", 64),
        quantized_kv_start=getattr(rg, "quantized_kv_start", 0),
        scheme=getattr(rg, "kv_quant_scheme", None),
        key_bits=getattr(rg, "kv_key_bits", None),
        value_bits=getattr(rg, "kv_value_bits", None),
        mtp=mtp,
        head_dim=_config_head_dim(rg.model),
    )
    pol = ServeKvPolicy(
        resolve_kv_quant_policy(stack, mode="single", **kw),
        resolve_kv_quant_policy(_probe_stack(rg.model), mode="batched",
                                **kw),
    )
    if pol.single.verdict == "error" or pol.batched.verdict == "error":
        bad = pol.single if pol.single.verdict == "error" else pol.batched
        raise KvPolicyError(kv_line(model_id, bad))
    setattr(rg, RG_ATTR, pol)
    # Also stamp the model: the batch worker thread reads the policy off
    # batch.model (residency's context-var proxy does not cross threads,
    # see engine._install_apc_manager_stash).
    try:
        setattr(rg.model, RG_ATTR, pol)
    except Exception:
        pass
    _log.info(kv_line(model_id, pol.single))
    if pol.batched.verdict != pol.single.verdict:
        _log.info(kv_line(model_id, pol.batched)
                  + " (when batched)")
    return pol


def pricing_vector(rg, num_layers: int):
    """The admission bytes-per-element vector for rg, or None to fall
    back to uniform pricing. Length-checked against the config layer
    count so a stack/config mismatch never misprices."""
    pol = getattr(rg, RG_ATTR, None)
    if pol is None:
        return None
    vec = pol.pricing_vector()
    return vec if len(vec) == num_layers else None
