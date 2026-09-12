"""MTP target wrapper and capability resolver.

The hook tables the mlx-vlm MTP engine probes on a target's
``language_model``, the per-arch target classes
(``_mtp_target_classes`` / ``_vlm_spec_language_model``), and
``MTPTextTarget``, the wrapper that exposes an mlx-vlm text
``LanguageModel`` to the MTP engine and the drafter bind walk.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from gmlx.envflags import env_bool

from . import loadlog


_MTP_TARGET_HOOKS = (
    "rollback_speculative_cache",
    "speculative_logits_from_hidden",
    "speculative_argmax_from_hidden",
    "speculative_verify_logits",
    "speculative_verify_hidden",
)
_MTP_TARGET_HOOKS_BY_TYPE = {
    "gemma4_text": (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_draft_hidden",
    ),
    # DeepseekV4SpecLM (vendored mlx-lm class, not mlx-vlm): no
    # speculative_verify_logits -- verify goes through verify_hidden and the
    # walk computes logits/argmax from the raw 4D hidden.
    "deepseek_v4": (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_argmax_from_hidden",
        "speculative_verify_hidden",
    ),
    # HyV3SpecLM (vendored mlx-lm class): same lean set as deepseek_v4.
    "hy_v3": (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_argmax_from_hidden",
        "speculative_verify_hidden",
    ),
    # MuseGlimmerSpecLM (vendored mlx-lm class): same lean set as deepseek_v4.
    "muse_glimmer": (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_argmax_from_hidden",
        "speculative_verify_hidden",
    ),
    # Qwen4ExpSpecLM (vendored mlx-lm class): same lean set as deepseek_v4.
    "qwen4_exp": (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_argmax_from_hidden",
        "speculative_verify_hidden",
    ),
    # Glm5NextSpecLM (vendored mlx-lm class): same lean set as deepseek_v4.
    "glm5_next": (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_argmax_from_hidden",
        "speculative_verify_hidden",
    ),
    # NemotronHSpecLM (stock mlx-lm class + hooks): same lean set.
    "nemotron_h": (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_argmax_from_hidden",
        "speculative_verify_hidden",
    ),
}


class MTPTextTarget(nn.Module):
    """Expose an mlx-vlm text ``LanguageModel`` as ``.language_model``.

    The MTP engine reaches the target through ``model.language_model`` (for the
    ``speculative_*`` hooks + ``hidden_states``), and the drafter's ``bind``
    walks ``.language_model.model.embed_tokens``. This is deliberately not the
    serving ``TextOnlyModel`` wrapper, whose ``.language_model`` is a logits-only
    adapter with none of those hooks.
    """

    def __init__(self, language_model, config: dict):
        super().__init__()
        self.language_model = language_model
        self.config = config

    def make_cache(self):
        return self.language_model.make_cache()

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        """Text-only embedding lookup the MTP engine calls on the top-level
        model (``mlx_vlm.generate.ar.generate_step``). Mirrors the qwen3.5 VLM
        ``Model``'s text-only branch - a GGUF text target has no vision tower -
        returning an ``InputEmbeddingsFeatures`` whose ``inputs_embeds`` is the
        token embedding. Clears ``_position_ids`` so mrope falls back to the
        plain text positions."""
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        self.language_model._position_ids = None
        embeds = self.language_model.model.embed_tokens(input_ids)
        # gemma4 scales token embeddings by sqrt(hidden) in its input_ids path
        # (Gemma4Model.__call__), but the inputs_embeds path does not - so a
        # target fed embeds must get them pre-scaled. qwen has no such scale.
        if self.config.get("model_type") == "gemma4_text":
            embeds = embeds * self.language_model.model.embed_scale
        return InputEmbeddingsFeatures(inputs_embeds=embeds)

    def __call__(self, *args, **kwargs):
        out = self.language_model(*args, **kwargs)
        if isinstance(out, mx.array):
            # mlx-lm-style SpecLMs (deepseek_v4) return raw logits on the
            # plain path; mlx-vlm's AR engine expects ``outputs.logits``.
            from mlx_vlm.models.base import LanguageModelOutput

            return LanguageModelOutput(logits=out)
        return out


def _mtp_target_classes(model_type: str):
    """Capability-resolver row ``(arch, speculative) -> (LanguageModel, build)``.

    ``build(config_dict) -> language_model`` encapsulates the per-arch
    constructor signature: qwen3.5/3.6 take ``(TextConfig, ModelConfig)`` (the
    mrope ``get_rope_index`` reads ``vision_config`` even for text, so we pass a
    default one - the text forward never touches the tower); gemma4 takes a bare
    ``TextConfig``. The stock mlx-lm classes gmlx builds for the plain text
    capability ship none of the ``speculative_*`` hooks, which is why MTP
    escalates to mlx-vlm here. Extend with new rows (deepseek_v4, ...) as drafters
    land, together with ``_vlm_spec_language_model`` when the arch also has a
    VLM shape (mmproj) so the two paths stay in lockstep.
    """
    import importlib

    if model_type in ("qwen3_5", "qwen3_5_moe"):
        sub = model_type
        lang = importlib.import_module(f"mlx_vlm.models.{sub}.language")
        cfg = importlib.import_module(f"mlx_vlm.models.{sub}.config")

        # Owned model-level forward by default: the model-level control
        # flow (masks, left-padding walk, batched-padded prefill, capture)
        # runs from gmlx code on a subclass; layers stay stock so the
        # fused-kernel seams keep engaging. GMLX_QWEN_OWNED=0 reverts to
        # the stock class wholesale.
        language_model_cls = lang.LanguageModel
        if env_bool("GMLX_QWEN_OWNED", True):
            import gmlx.models.qwen35.owned as qwen35_owned

            language_model_cls = qwen35_owned.language_model_class(sub)

        def build(config):
            text_config = cfg.TextConfig.from_dict(config)
            model_config = cfg.ModelConfig.from_dict(
                {
                    "model_type": sub,
                    "text_config": dict(config),
                    "vision_config": {},
                    "vocab_size": config.get("vocab_size"),
                }
            )
            return language_model_cls(text_config, model_config)

        return language_model_cls, build
    if model_type == "gemma4_text":
        lang = importlib.import_module("mlx_vlm.models.gemma4.language")
        cfg = importlib.import_module("mlx_vlm.models.gemma4.config")

        # Owned mask builder + attention by default: the nosync offset
        # semantics and the hd512 row-route dispatch run from gmlx
        # subclasses; layers stay stock so the fused-MoE swap keeps
        # engaging. GMLX_GEMMA_OWNED=0 reverts to the stock class (the
        # patch regime still installs and covers it).
        language_model_cls = lang.LanguageModel
        if env_bool("GMLX_GEMMA_OWNED", True):
            import gmlx.models.gemma4.owned as gemma4_owned

            language_model_cls = gemma4_owned.OwnedGemma4LanguageModel

        def build(config):
            return language_model_cls(cfg.TextConfig.from_dict(config))

        return language_model_cls, build
    if model_type == "deepseek_v4":
        # Vendored mlx-lm-class target (no mlx-vlm counterpart): the SpecLM
        # subclass carries the speculative_* hooks + rotating-undo arming.
        import gmlx.models.deepseek_v4.mtp as deepseek_v4_mtp
        from gmlx.models.deepseek_v4.model import ModelArgs, ensure_registered

        ensure_registered()

        def build(config):
            return deepseek_v4_mtp.DeepseekV4SpecLM(ModelArgs.from_dict(config))

        return deepseek_v4_mtp.DeepseekV4SpecLM, build
    if model_type == "hy_v3":
        import gmlx.models.hy_v3.mtp as hy_v3_mtp
        import gmlx.models.hy_v3.tools as hy_v3_tools
        from gmlx.models.hy_v3.model import ModelArgs, ensure_registered

        ensure_registered()
        hy_v3_tools.ensure_registered()

        def build(config):
            return hy_v3_mtp.HyV3SpecLM(ModelArgs.from_dict(config))

        return hy_v3_mtp.HyV3SpecLM, build
    if model_type == "qwen4_exp":
        import gmlx.models.qwen4_exp.mtp as qwen4_exp_mtp
        from gmlx.models.qwen4_exp.model import ModelArgs, ensure_registered

        ensure_registered()

        def build(config):
            return qwen4_exp_mtp.Qwen4ExpSpecLM(ModelArgs.from_dict(config))

        return qwen4_exp_mtp.Qwen4ExpSpecLM, build
    if model_type == "muse_glimmer":
        import gmlx.models.muse_glimmer.mtp as muse_glimmer_mtp
        import gmlx.models.muse_glimmer.tools as muse_glimmer_tools
        from gmlx.models.muse_glimmer.model import ModelArgs, ensure_registered

        ensure_registered()
        muse_glimmer_tools.ensure_registered()

        def build(config):
            return muse_glimmer_mtp.MuseGlimmerSpecLM(ModelArgs.from_dict(config))

        return muse_glimmer_mtp.MuseGlimmerSpecLM, build
    if model_type == "glm5_next":
        import gmlx.models.glm5_next.mtp as glm5_next_mtp
        from gmlx.models.glm5_next.model import ModelArgs, ensure_registered

        ensure_registered()

        def build(config):
            return glm5_next_mtp.Glm5NextSpecLM(ModelArgs.from_dict(config))

        return glm5_next_mtp.Glm5NextSpecLM, build
    if model_type == "nemotron_h":
        import gmlx.models.nemotron_h.mtp as nemotron_h_mtp
        from mlx_lm.models.nemotron_h import ModelArgs

        def build(config):
            return nemotron_h_mtp.NemotronHSpecLM(ModelArgs.from_dict(config))

        return nemotron_h_mtp.NemotronHSpecLM, build
    from .arch_table import MTP_WIRED_MODEL_TYPES

    raise NotImplementedError(
        f"MTP target class for model_type {model_type!r} not wired "
        f"(supported: {' / '.join(sorted(MTP_WIRED_MODEL_TYPES))})"
    )


# VLM model_types whose spec-capability row lives under a different text
# model_type in the tables above (hook sets + target classes).
_VLM_SPEC_MODEL_TYPE_ALIASES = {
    "gemma4": "gemma4_text",
    "gemma4_unified": "gemma4_text",
    # The Vision-Exp container's language model carries the deepseek_v4
    # text hooks (DeepseekV4SpecHooks).
    "deepseek_v4_vl": "deepseek_v4",
}


def _spec_hook_key(model_type: str) -> str:
    """The `_MTP_TARGET_HOOKS_BY_TYPE` / target-class key for a VLM
    model_type."""
    return _VLM_SPEC_MODEL_TYPE_ALIASES.get(model_type, model_type)


def _vlm_spec_language_model(model_type: str):
    """Spec-capable ``language_model`` row for a VLM model_type, or None.

    ``build(model_config) -> language_model`` constructs from the VLM's REAL
    mlx-vlm ModelConfig (real vision_config, so mrope ``get_rope_index`` on
    image turns stays correct - unlike ``_mtp_target_classes``'s text build,
    which passes an empty one). None means the built ``.language_model``
    already carries the hooks (muse_glimmer / glm5_next / qwen4_exp wire
    their mixins in their own vlm_model), or the arch has no spec support
    (the per-arch hook check downstream fails loud). The env gates
    (GMLX_QWEN_OWNED / GMLX_GEMMA_OWNED) are consulted at call time via
    ``_mtp_target_classes``, so =0 resolves to the stock class and the swap
    no-ops against the already-stock tree."""
    if model_type in ("qwen3_5", "qwen3_5_moe"):
        cls, _ = _mtp_target_classes(model_type)

        def build(model_config):
            return cls(model_config.text_config, model_config)

        return cls, build
    if model_type in ("gemma4", "gemma4_unified"):
        cls, _ = _mtp_target_classes("gemma4_text")

        def build(model_config):
            return cls(model_config.text_config)

        return cls, build
    return None


def _ensure_argmax_hook(language_model) -> None:
    """Batched greedy verify walk: under greedy the engine takes the
    per-position deferred walk (one CPU<->GPU sync per draft position) unless
    the target exposes speculative_argmax_from_hidden, which lets it argmax
    all block+1 verify positions in a single op (zero per-position syncs ->
    _speculative_walk). gemma4's LanguageModel ships only
    speculative_logits_from_hidden, so synthesize the argmax wrapper from it -
    lossless (same tokens), just fewer syncs (~+8% decode on a small target
    whose round is sync-bound). Only ever fires for the gemma4 row; every
    other type's hook table already lists speculative_argmax_from_hidden."""
    if not hasattr(language_model, "speculative_argmax_from_hidden") and hasattr(
        language_model, "speculative_logits_from_hidden"
    ):
        _lm = language_model
        _lm.speculative_argmax_from_hidden = lambda hidden: mx.argmax(
            _lm.speculative_logits_from_hidden(hidden), axis=-1
        )


def _build_mtp_target(config_dict: dict):
    """Build the MTP target as an mlx-vlm text ``LanguageModel`` (seam 1)."""
    config = dict(config_dict)
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    model_type = config.get("model_type", "")
    LanguageModel, build = _mtp_target_classes(model_type)
    hooks = _MTP_TARGET_HOOKS_BY_TYPE.get(model_type, _MTP_TARGET_HOOKS)
    missing = [h for h in hooks if not hasattr(LanguageModel, h)]
    if missing:
        raise RuntimeError(
            f"mlx-vlm {model_type} LanguageModel missing MTP hooks {missing} "
            f"- version drift; pin mlx-vlm or update the hook set"
        )
    language_model = build(config)
    _ensure_argmax_hook(language_model)
    wrapper = MTPTextTarget(language_model, config)
    loadlog.verbose_print(
        f"[build] {model_type} -> {type(language_model).__name__} (MTP target wrapper)"
    )
    return wrapper, config
