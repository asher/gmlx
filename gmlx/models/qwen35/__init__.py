"""Qwen3.5 family: attention, GDN, and verify paths."""

# The copied package's language module is an alias of the owned one, and
# the owned one imports the copied package's config. The copied package
# goes first so that its container never sees a half-built language module.
import gmlx._vendor.mlx_vlm.models.qwen3_5  # noqa: F401
import gmlx.gen.prefill_decay as _prefill_decay

from .gdn import _QWEN_GDN_FAMILY


def _prefill_score_profile(model, prompt_cache):
    # Prefill attention is stock SDPA with a causal or array mask. The MTP
    # drafter shares the model type and keeps the dense model.
    from .language import LanguageModel

    if not isinstance(getattr(model, "language_model", model), LanguageModel):
        return None
    return _prefill_decay.fused_attention_profile(model, prompt_cache)


for _mt in _QWEN_GDN_FAMILY:
    _prefill_decay.register_score_profile(_mt, _prefill_score_profile)
