"""Serve GGUF K-quant models through mlx-vlm's continuous-batching server.

mlx-vlm's FastAPI server (OpenAI / Responses / Anthropic surfaces, tool-call
parsing, APC prefix reuse, and a ``BatchGenerator`` continuous-batching engine)
loads checkpoints from a path via a single module-level loader,
``server.generation.load_model_resources``. This module bridges a GGUF path into
that server by replacing only that load step:

* :func:`load_serveable_model` loads the GGUF with
  :func:`gmlx.load_model`, wraps the resulting stock model in
  mlx-vlm's ``text_only.Model`` so it exposes the ``get_input_embeddings`` /
  ``language_model`` interface the engine consumes, and adapts the synthesized
  tokenizer into the processor shape the server expects (a callable
  ``.tokenizer`` carrying a ``StoppingCriteria`` plus a copy-safe streaming
  ``.detokenizer``). With an associated float ``mmproj`` GGUF it instead loads
  a two-GGUF VLM via :func:`gmlx.vlm.load_vlm_model` - a real mlx-vlm
  vision/audio model whose synthesized processor is already engine-ready, so the
  multimodal request path (``prepare_inputs`` -> ``get_input_embeddings`` ->
  ``BatchGenerator``) runs stock.
* :func:`install_gguf_server_bridge` patches ``load_model_resources`` so a
  ``*.gguf`` model path is loaded this way; every other path falls through to
  mlx-vlm's original loader untouched. The mmproj for a VLM is supplied via
  :func:`register_gguf_vlm` (or the ``MLX_VLM_GGUF_MMPROJ`` env fallback); a GGUF
  is served with MTP speculative decoding via :func:`register_gguf_mtp` (or the
  ``MLX_VLM_GGUF_SPECULATIVE`` env fallback).

Speculative (MTP) serving needs two things the plain bridge can't express through
the ``(model, processor, config)`` load triple: the target must be the
speculative-capable ``LanguageModel`` (not the logits-only ``text_only.Model``),
and the drafter the loader builds beside it has no slot in that triple. So the
MTP path returns the real target wrapper as the model and stashes the drafter by
path; :func:`install_gguf_server_bridge` additionally patches
``speculative.drafters.load_drafter`` so the engine's stock drafter-load step
picks up that in-memory drafter instead of reloading one from disk (a disk reload
would rebuild affine-quantized leaves and dequantize the K-quant drafter).

Nothing else changes. The ``ResponseGenerator``, its worker loop,
embedding/prefill path, sampler, tool plumbing and the MTP round engine are all
stock and protocol-agnostic - there is deliberately no GGUF-specific generator.
This mirrors ``server_bridge_lm.py`` (which bridges GGUF into ``mlx_lm.server`` by
patching ``ModelProvider._load``); here the batched engine is used, not the
sequential one.
"""

from __future__ import annotations

import json
import os
import sys
from contextvars import ContextVar

from mlx_vlm import tokenizer_utils as _mlxvlm_tok
from mlx_vlm.models.text_only import Model as TextOnlyModel
from mlx_vlm.utils import StoppingCriteria

from .loader import load_model


def _is_gguf(path) -> bool:
    return isinstance(path, str) and path.endswith(".gguf")


# VLM (two-GGUF) association
#
# A VLM is two GGUFs - the K-quant LLM GGUF plus a float ``mmproj`` GGUF - but
# the server's load seam (``load_model_resources``) only ever sees the *model*
# path. This registry lets a caller associate an mmproj (and an optional
# processor/config override) with an LLM GGUF path ahead of the load, so the
# bridge can recover it. Keyed by absolute path; a single-model launch can skip
# registration and set ``MLX_VLM_GGUF_MMPROJ`` instead.

_GGUF_VLM_REGISTRY: dict[str, dict] = {}


def register_gguf_vlm(
    gguf_path: str, mmproj_path: str, *, hf_source: str | None = None
) -> None:
    """Associate a float ``mmproj`` GGUF with an LLM GGUF so the server serves it
    as a VLM. ``hf_source`` is an optional processor/config override (rarely
    needed - the processor is normally synthesized from the two GGUFs)."""
    _GGUF_VLM_REGISTRY[os.path.abspath(gguf_path)] = {
        "mmproj_path": os.path.abspath(mmproj_path),
        "hf_source": hf_source,
    }


def _resolve_vlm_spec(model_path: str) -> dict | None:
    """The ``{mmproj_path, hf_source}`` for a GGUF model path, or ``None`` for a
    plain text GGUF. Explicit :func:`register_gguf_vlm` wins; otherwise the
    ``MLX_VLM_GGUF_MMPROJ`` / ``MLX_VLM_GGUF_HF_SOURCE`` env vars provide a
    single-model launch fallback."""
    spec = _GGUF_VLM_REGISTRY.get(os.path.abspath(model_path))
    if spec is not None:
        return spec
    mmproj = os.environ.get("MLX_VLM_GGUF_MMPROJ")
    if mmproj:
        return {
            "mmproj_path": mmproj,
            "hf_source": os.environ.get("MLX_VLM_GGUF_HF_SOURCE"),
        }
    return None


# MTP (speculative decoding) association
#
# Like the VLM case, the server's load seam only ever sees the *model* path, so a
# caller marks a GGUF for speculative (MTP) serving ahead of the load. Native-head
# MTP (qwen3.5/3.6) needs no companion file - the drafter lives in the target
# GGUF's own MTP block; the assistant shape (gemma4) carries a separate drafter
# GGUF. A single-model launch can skip registration and set
# ``MLX_VLM_GGUF_SPECULATIVE=1`` (plus ``MLX_VLM_GGUF_DRAFT`` for the assistant
# shape) instead.

_GGUF_MTP_REGISTRY: dict[str, dict] = {}

# model-abspath -> (drafter, kind). Populated while loading an MTP target;
# consumed by the drafter-load patch so the in-memory drafter the loader built is
# injected into the engine without a disk round-trip (which would rebuild
# affine-quantized leaves and dequantize the K-quant drafter).
_MTP_DRAFTER_STASH: dict[str, tuple] = {}


def register_gguf_mtp(gguf_path: str, *, draft_gguf_path: str | None = None) -> None:
    """Mark a GGUF for speculative (MTP) serving. ``draft_gguf_path`` is the
    companion drafter GGUF for the assistant shape (gemma4); leave it ``None`` for
    native-head targets (qwen3.5/3.6) whose drafter ships inside the target GGUF.
    """
    _GGUF_MTP_REGISTRY[os.path.abspath(gguf_path)] = {
        "draft_gguf_path": (
            os.path.abspath(draft_gguf_path) if draft_gguf_path else None
        ),
    }


def drop_mtp_stash(gguf_path: str) -> None:
    """Drop the stashed in-memory drafter for ``gguf_path``. Called by the
    residency pool's teardown so an unconsumed stash entry (e.g. a build that
    failed between stash and drafter-load) can't keep drafter weights referenced
    after the owning model is evicted."""
    _MTP_DRAFTER_STASH.pop(os.path.abspath(gguf_path), None)


def _resolve_mtp_spec(model_path: str) -> dict | None:
    """The ``{draft_gguf_path}`` for a GGUF marked for MTP serving, or ``None`` for
    a non-speculative load.

    Precedence: an explicit *falsey* ``MLX_VLM_GGUF_SPECULATIVE`` wins over
    everything - the residency env window sets it per build so a non-speculative
    id forces a plain load even when a *sibling* id registered the same GGUF for
    MTP (one GGUF backing a spec id + a lossless-oracle id). The model builds in
    the engine's worker thread, so this process-global env - not a request-thread
    ContextVar - is the per-build signal that reaches the bridge. Otherwise an
    explicit :func:`register_gguf_mtp` wins; otherwise a truthy
    ``MLX_VLM_GGUF_SPECULATIVE`` enables it for a single-model launch, with
    ``MLX_VLM_GGUF_DRAFT`` supplying the assistant-shape companion."""
    env = os.environ.get("MLX_VLM_GGUF_SPECULATIVE", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return None
    spec = _GGUF_MTP_REGISTRY.get(os.path.abspath(model_path))
    if spec is not None:
        return spec
    if env in ("1", "true", "yes", "on"):
        draft = os.environ.get("MLX_VLM_GGUF_DRAFT")
        return {"draft_gguf_path": os.path.abspath(draft) if draft else None}
    return None


def _build_detokenizer(backend):
    """Pick the mlx-vlm streaming detokenizer matching the tokenizer's decoder.

    Mirrors mlx-vlm's ``load_tokenizer`` inference (SPM / SPM-no-space / BPE /
    Naive) but reads the decoder description from the in-memory tokenizer - a
    synthesized GGUF tokenizer has no ``tokenizer.json`` on disk. mlx-vlm's
    detokenizers define ``__copy__`` so the server can copy one per request;
    mlx-lm's Naive detokenizer (what the loader attaches) is *not* copy-safe, so
    we never reuse it. Any inference failure falls back to mlx-vlm's Naive
    detokenizer, which is correct for every tokenizer.
    """
    naive = _mlxvlm_tok.NaiveStreamingDetokenizer
    try:
        raw = getattr(backend, "backend_tokenizer", None)
        decoder = json.loads(raw.to_str()).get("decoder") if raw is not None else None
        if decoder is not None:
            if _mlxvlm_tok._is_spm_decoder(decoder):
                return _mlxvlm_tok.SPMStreamingDetokenizer(backend)
            if _mlxvlm_tok._is_spm_decoder_no_space(decoder):
                return _mlxvlm_tok.SPMStreamingDetokenizer(backend, trim_space=False)
            if _mlxvlm_tok._is_bpe_decoder(decoder):
                return _mlxvlm_tok.BPEStreamingDetokenizer(backend)
    except Exception:
        pass  # unprobeable tokenizer json -> naive detokenizer
    return naive(backend)


class _GgufServerProcessor:
    """Present a synthesized GGUF tokenizer in the processor shape mlx-vlm wants.

    mlx-vlm's server and batching engine treat the processor as an HF-style
    object: both derive the working tokenizer as ``processor.tokenizer if
    hasattr(processor, "tokenizer") else processor``, call it like an HF
    tokenizer (``tokenizer(prompts, return_tensors="mlx")`` inside
    ``prepare_inputs``), read the stop criteria off it, copy
    ``processor.detokenizer`` once per request, and read ``chat_template`` off
    the processor. mlx-lm's ``TokenizerWrapper`` is not itself callable, exposes
    the callable backend as ``._tokenizer``, and rebuilds a *non-copy-safe*
    detokenizer through a class property. This adapter surfaces the callable
    backend as ``.tokenizer``, holds a copy-safe mlx-vlm ``.detokenizer``, and
    delegates everything else (``chat_template``, ``apply_chat_template``,
    ``encode``/``decode`` ...) to the original tokenizer wrapper.
    """

    def __init__(self, wrapper, backend, detokenizer):
        self._wrapper = wrapper
        self.tokenizer = backend
        self.detokenizer = detokenizer

    def __getattr__(self, name):
        # Only reached for attributes not set on the instance; guard the backing
        # field so a premature lookup can't recurse forever.
        if name == "_wrapper":
            raise AttributeError(name)
        return getattr(self._wrapper, name)


def _as_dict(config) -> dict:
    if isinstance(config, dict):
        return config
    for attr in ("to_dict", "__dict__"):
        value = getattr(config, attr, None)
        if callable(value):
            return dict(value())
        if isinstance(value, dict):
            return dict(value)
    return {}


class _AttrDict(dict):
    """A dict whose keys are also reachable as attributes.

    The batched server reads a model's config *both* ways: dict-style inside the
    MTP target's ``get_input_embeddings`` (``config.get(...)``) and
    attribute-style in the request preprocessor (``model.config.model_type``). A
    plain dict supports only the former; mlx-vlm's ``AttributeConfig`` only the
    latter. This bridges both, so one object can back the model's ``.config`` and
    the loader's returned config alike (as the text path's ``AttributeConfig``
    does)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _ensure_inner_config(model, cfg) -> None:
    """Mirror a model's config onto its inner ``.language_model``.

    The batched engine binds its ``BatchGenerator`` to ``model.language_model``
    (the inner ``text_only.LanguageModel``), which carries ``model_type`` but no
    ``.config`` - yet the engine's APC prefix picker reads
    ``self.model.config.image_token_id`` for every sequence, so an enabled prompt
    cache crashes a text GGUF with ``'LanguageModel' object has no attribute
    'config'``. Attach the config to the inner LM (as a plain instance attribute,
    not through the Module parameter tree) so APC and any ``model.config`` reader
    resolves. No-op when the inner LM already carries a config (real VLM language
    models do)."""
    lm = getattr(model, "language_model", None)
    if lm is not None and getattr(lm, "config", None) is None:
        try:
            object.__setattr__(lm, "config", cfg)
        except (AttributeError, TypeError):
            pass                               # slotted/exotic inner model - skip


# Token-embedding attribute names mlx-vlm's text_only.LanguageModel probes for.
_EMBED_ATTR_NAMES = (
    "embed_tokens", "wte", "embeddings", "tok_embeddings", "word_embeddings",
)


def _find_token_embedding(raw_model):
    """Locate the token-embedding module on a stock mlx-lm model.

    mlx-vlm's ``text_only.LanguageModel._token_embedding`` probes only the top
    level and one ``.model`` hop - the canonical mlx-lm dense layout
    (``Model.model.embed_tokens``: llama / qwen2 / qwen3 / gemma / ...). The
    hybrid qwen3_5/3.6 *dense* family keeps a VLM-shaped
    ``Model.language_model.model.embed_tokens`` nesting even for its text
    checkpoint (the family has VL variants), one hop deeper than that probe - so
    the batched engine's GPU-embed step (``get_input_embeddings`` ->
    ``input_embeds`` -> the probe) raises. Walk the known nestings here and
    return the embedding module, or ``None`` if none is reachable.
    """
    lang = getattr(raw_model, "language_model", None)
    hops = (
        raw_model,
        getattr(raw_model, "model", None),
        lang,
        getattr(lang, "model", None),
    )
    for hop in hops:
        if hop is None:
            continue
        for name in _EMBED_ATTR_NAMES:
            emb = getattr(hop, name, None)
            if emb is not None:
                return emb
    return None


def _ensure_text_embedding_probe(model, raw_model) -> None:
    """Make ``model.language_model`` expose its token embedding to the batched
    engine's GPU-embed probe.

    No-op when mlx-vlm's own shallow ``_token_embedding`` already resolves
    (top-level or single ``.model`` hop); only the deeper-nested families (hybrid
    qwen3_5/3.6 dense) get a one-instance override returning the embedding
    :func:`_find_token_embedding` located. The override is a plain instance
    attribute (via ``object.__setattr__``, like :func:`_ensure_inner_config`), so
    it shadows the class method without registering a duplicate leaf in the
    Module parameter tree. Fail fast at *load* if no embedding is reachable, so a
    new arch surfaces a clear error here instead of an opaque worker-thread crash
    mid-generation."""
    lm = getattr(model, "language_model", None)
    if lm is None:
        return
    try:
        if lm._token_embedding() is not None:
            return                             # stock probe already reaches it
    except Exception:                          # noqa: BLE001 - exotic wrapper
        pass
    emb = _find_token_embedding(raw_model)
    if emb is None:
        raise RuntimeError(
            f"served text model {type(raw_model).__name__!r} exposes no token "
            f"embedding the batched engine can reach (probed {_EMBED_ATTR_NAMES} "
            f"on self / .model / .language_model / .language_model.model) - wire "
            f"its layout into gmlx.server_bridge_vlm._find_token_embedding")
    object.__setattr__(lm, "_token_embedding", lambda: emb)


def _load_serveable_vlm(
    gguf_path: str, mmproj_path: str, *, hf_source: str | None = None
) -> tuple[object, object, object]:
    """Load a two-GGUF VLM (K-quant LLM + float mmproj) for the batched engine.

    Unlike the text path there is no adapter and no ``text_only`` wrap:
    :func:`gmlx.vlm.load_vlm_model` returns the *real* mlx-vlm vision/audio
    ``Model`` (already exposing ``get_input_embeddings`` and a ``config`` with
    ``model_type`` / ``image_token_id``) plus a GGUF-synthesized processor that
    already carries the full contract the engine reads: a callable
    ``PreTrainedTokenizerFast`` at ``.tokenizer`` (with ``stopping_criteria`` +
    ``eos_token_ids``), a copy-safe streaming ``.detokenizer``, and the
    ``image_processor`` / ``feature_extractor`` ``prepare_inputs`` needs. Only
    the return order differs (``load_vlm_model`` yields ``(model, config,
    processor)``; the server wants ``(model, processor, config)``).
    """
    from .vlm import load_vlm_model

    model, _config_dict, processor = load_vlm_model(
        gguf_path, mmproj_path, hf_source=hf_source, verbose=False
    )
    # Return the model's own dataclass config (what stock load_model_resources
    # returns as the 3rd element), not the synthesized dict.
    return model, processor, model.config


def _make_text_processor(tokenizer) -> "_GgufServerProcessor":
    """Present a synthesized GGUF ``TokenizerWrapper`` in the processor shape the
    engine reads. Shared by the plain-text and speculative (MTP) text paths - both
    attach EOS stop criteria to the callable backend and a copy-safe streaming
    detokenizer. (The VLM path gets an engine-ready processor from the loader and
    does not use this.)"""
    backend = getattr(tokenizer, "_tokenizer", tokenizer)
    eos = getattr(tokenizer, "eos_token_ids", None) or getattr(
        tokenizer, "eos_token_id", None
    )
    # StoppingCriteria.add_eos_token_ids() mutates this list in place, so it must
    # be a list - the synthesized tokenizer exposes eos_token_ids as a set. The
    # engine reads the criteria off ``processor.tokenizer`` (= the backend).
    if isinstance(eos, (set, tuple)):
        eos = list(eos)
    backend.stopping_criteria = StoppingCriteria(eos, backend)
    return _GgufServerProcessor(tokenizer, backend, _build_detokenizer(backend))


def _load_serveable_mtp(
    gguf_path: str, *, draft_gguf_path: str | None = None,
    chat_template: str | None = None,
) -> tuple[object, object, object]:
    """Load an MTP (speculative) text target + drafter for the batched engine.

    Returns ``(model, processor, config)`` like the other paths, with two
    differences the server's drafter ingress needs:

    * ``model`` is the MTP target wrapper whose ``.language_model`` is the real
      speculative-capable ``LanguageModel`` (carrying the verify hooks the engine
      reaches through ``model.language_model``), not the logits-only
      ``text_only.Model`` the plain-text path uses.
    * the drafter the loader built alongside the target has no slot in the
      ``(model, processor, config)`` triple, so it is stashed by absolute path;
      the drafter-load patch (:func:`_install_drafter_injection`) hands it to the
      engine when it asks to load a drafter for this GGUF.
    """
    from .mtp_load import load_mtp_model

    model, drafter, _config, tokenizer = load_mtp_model(
        gguf_path, draft_gguf_path=draft_gguf_path,
        chat_template=chat_template, verbose=False
    )
    # MTPTextTarget.config is a plain dict (the CLI path only reads it
    # dict-style). The server also reads it attribute-style
    # (``self.model.config.model_type`` in the request preprocessor), so promote
    # it in place to an attribute-readable dict and hand the same object back as
    # the loader's config - matching the text path, whose model.config is the
    # returned config.
    model.config = _AttrDict(model.config)
    _ensure_inner_config(model, model.config)
    # Runtime-origin cache identities for the apc/ar isinstance gates; a
    # vlm-native LanguageModel target makes this a no-op, an mlx-lm-style
    # target (deepseek_v4 et al) needs it since mlx-vlm 0.6.4.
    from .cache_compat import ensure_runtime_origin_make_cache
    ensure_runtime_origin_make_cache(model.language_model)
    _MTP_DRAFTER_STASH[os.path.abspath(gguf_path)] = (drafter, "mtp")
    processor = _make_text_processor(tokenizer)
    return model, processor, model.config


def _load_serveable_vlm_mtp(
    gguf_path: str, mmproj_path: str, *, hf_source: str | None = None,
    draft_gguf_path: str | None = None, chat_template: str | None = None,
) -> tuple[object, object, object]:
    """Load a VLM (K-quant LLM + float mmproj) with an MTP drafter - the serve form
    of the run/chat ``load_vlm_mtp_model`` path. A text-only request runs through the
    engine's speculative rounds (which touch ``model.language_model`` + caches); an
    image/audio request prefills the vision/audio embeddings into the KV cache and
    verify still runs token-only over that cache (greedy verify stays lossless), so
    one resident model serves both. Combines the two seams the other paths use:

    * the real mlx-vlm vision/audio ``Model`` + its engine-ready processor, exactly
      like :func:`_load_serveable_vlm` (so image/audio ``prepare_inputs`` works), and
    * the drafter stashed by absolute path for the engine's drafter-load step, exactly
      like :func:`_load_serveable_mtp`.

    The drafter is the gemma4 assistant (``draft_gguf_path``) or the qwen3.5/3.6
    native head; ``load_vlm_mtp_model`` picks by ``draft_gguf_path``.
    """
    from .mtp_load import load_vlm_mtp_model

    model, drafter, _config, _tokenizer, processor = load_vlm_mtp_model(
        gguf_path, mmproj_path, hf_source=hf_source,
        draft_gguf_path=draft_gguf_path, chat_template=chat_template,
        verbose=False,
    )
    _MTP_DRAFTER_STASH[os.path.abspath(gguf_path)] = (drafter, "mtp")
    # The VLM model's own dataclass config is attribute-readable already (unlike the
    # text MTP wrapper's dict), so hand it back directly like _load_serveable_vlm.
    return model, processor, model.config


def _apply_gguf_adapter(raw_model, config, adapter_gguf: str,
                        base_gguf_path: str | None = None) -> int:
    """Wrap the base text model's Linear leaves with a GGUF LoRA adapter - live, no
    merge (base stays K-quant; the adapter rides alongside in full precision).

    Applies to the *raw* mlx-lm model, whose leaf paths are the HF names the adapter's
    GGUF-base-name remap targets - the same keys :func:`install_kquant_modules` swapped.
    Head counts (for the llama-family q/k de-permute of the adapter's ``B``) come from
    the synthesized config; a qk_permute target without them raises in the installer.
    ``base_gguf_path`` supplies the base's GGUF arch so an adapter trained for a
    different family fails with the clean arch-mismatch message up front, instead
    of the structural missing-targets raise from :func:`install_lora_adapter`."""
    from .adapter import apply_gguf_adapter

    base_arch = None
    if base_gguf_path:
        from .discovery import header_meta
        base_arch = (header_meta(base_gguf_path) or {}).get("arch")
    return apply_gguf_adapter(raw_model, config, adapter_gguf,
                              base_arch=base_arch)


def load_serveable_model(
    gguf_path: str,
    *,
    mmproj_path: str | None = None,
    hf_source: str | None = None,
    speculative: bool = False,
    draft_gguf_path: str | None = None,
    chat_template: str | None = None,
    adapter_gguf: str | None = None,
    stream=None,
    moe_experts: int | None = None,
    moe_expert_mass: float | None = None,
    moe_miss_shed: float | None = None,
    moe_layer_shed: float | None = None,
    feeder_prefill: bool | None = None,
    feeder_decode: bool | None = None,
) -> tuple[object, object, object]:
    """Load a GGUF model into the form mlx-vlm's server expects.

    Drop-in replacement for ``server.generation.load_model_resources``: returns
    ``(model, processor, config)``. Four exclusive shapes:

    * ``mmproj_path`` + ``speculative`` -> a VLM with an MTP drafter via
      :func:`_load_serveable_vlm_mtp` (text-only requests speculate, media requests
      take the VLM forward; drafter stashed for the engine's drafter-load step).
    * ``mmproj_path`` -> a two-GGUF VLM via :func:`_load_serveable_vlm` (a real
      mlx-vlm vision/audio model).
    * ``speculative`` -> an MTP target + drafter via :func:`_load_serveable_mtp`
      (the speculative-capable ``LanguageModel`` wrapper; drafter stashed for the
      engine's drafter-load step).
    * otherwise -> a text GGUF whose stock model is wrapped in ``text_only.Model``
      (for the ``get_input_embeddings`` / ``language_model`` interface) and whose
      synthesized tokenizer is presented through :class:`_GgufServerProcessor`.

    ``chat_template`` (a config profile's resolved override - inline Jinja or a
    ``.jinja``/``.txt`` path) is baked into the synthesized tokenizer for the text
    and MTP paths. The VLM path keeps its mmproj-synthesized processor template (a
    template override there would need processor plumbing - not yet wired).

    ``adapter_gguf`` (a GGUF LoRA adapter) is applied live over the loaded base on the
    text path (no merge, no requant). The VLM and speculative/MTP paths don't yet wire
    adapter apply, so an adapter on those raises rather than silently dropping it.

    ``stream`` selects the text-path execution placement: ``"experts"`` streams
    only the routed-expert stacks from disk while the every-token layers + KV
    cache stay on GPU; ``"cpu"`` runs the whole model on the CPU device (all
    weights streamed through the page cache). Like the adapter, the VLM and
    MTP paths raise rather than silently dropping it.

    The lossy MoE levers (config ``moe_experts: K`` / ``moe_expert_mass: P`` /
    ``moe_miss_shed: P`` / ``moe_layer_shed: P``) install their filters/hooks
    over the streamed layers after the placement. They ride on ``stream`` -
    without a placement each is announced as ignored (there are no streamed
    experts to filter).
    """
    def _reject_unwired(base_kind: str) -> None:
        # Raising beats silently dropping the option on bases that don't
        # wire it yet.
        if adapter_gguf is not None:
            raise NotImplementedError(
                f"live GGUF LoRA on a {base_kind} base is not wired yet; "
                f"adapter={adapter_gguf!r}")
        if stream:
            raise NotImplementedError(
                f"stream placement on a {base_kind} base is not wired yet; "
                f"stream={stream!r}")

    if not stream:
        for key, val in (("moe_experts", moe_experts),
                         ("moe_expert_mass", moe_expert_mass),
                         ("moe_miss_shed", moe_miss_shed),
                         ("moe_layer_shed", moe_layer_shed)):
            if val is not None:
                print(
                    f"[stream] {key} ignored: needs stream: experts|cpu "
                    "(it only applies to streamed MoE layers)"
                )
        moe_experts = moe_expert_mass = None
        moe_miss_shed = moe_layer_shed = None
    if mmproj_path is not None and speculative:
        # VLM x MTP: text-only requests speculate; image/audio requests prefill media
        # into the KV and decode normally (verify is token-only over that cache).
        _reject_unwired("VLM x MTP")
        return _load_serveable_vlm_mtp(
            gguf_path, mmproj_path, hf_source=hf_source,
            draft_gguf_path=draft_gguf_path, chat_template=chat_template)

    if mmproj_path is not None:
        _reject_unwired("VLM")
        return _load_serveable_vlm(gguf_path, mmproj_path, hf_source=hf_source)

    if speculative:
        _reject_unwired("speculative/MTP")
        return _load_serveable_mtp(
            gguf_path, draft_gguf_path=draft_gguf_path,
            chat_template=chat_template)

    raw_model, config, tokenizer = load_model(
        gguf_path, chat_template=chat_template, verbose=False)

    from .diffusion import is_diffusion_model

    if is_diffusion_model(raw_model):
        # A DiffusionGemma checkpoint is already a real mlx-vlm Model; the
        # server's native diffusion lane drives it directly. Don't wrap it in
        # text_only.Model or flatten its nested ModelConfig (the denoiser reads
        # ``model.config.text_config.vocab_size`` attribute-style) - hand back the
        # model + its own config, with the engine-ready text processor.
        _reject_unwired("diffusion")
        return raw_model, _make_text_processor(tokenizer), raw_model.config

    if adapter_gguf is not None:
        _apply_gguf_adapter(raw_model, config, adapter_gguf,
                            base_gguf_path=gguf_path)
    if stream == "cpu":
        # Whole model on the CPU device (process-global). Intended for a single
        # over-RAM positional model; mixing with GPU-resident models in one
        # config-mode server is unsupported.
        from .loader import configure_stream_cpu
        configure_stream_cpu(
            raw_model, gguf_path=gguf_path,
            feeder_prefill=feeder_prefill, feeder_decode=feeder_decode)
    elif stream:  # "experts": routed experts stream; rest of model + KV on GPU
        from .loader import install_expert_streaming
        install_expert_streaming(
            raw_model, gguf_path=gguf_path,
            feeder_prefill=feeder_prefill, feeder_decode=feeder_decode)
    if moe_experts is not None:
        from .loader import install_moe_experts_override
        install_moe_experts_override(raw_model, moe_experts)
    if moe_expert_mass is not None:
        from .moe_experts import install_moe_expert_mass
        install_moe_expert_mass(raw_model, moe_expert_mass)
    if moe_miss_shed is not None:
        from .moe_experts import install_moe_miss_shed
        install_moe_miss_shed(raw_model, moe_miss_shed)
    if moe_layer_shed is not None:
        from .moe_experts import install_moe_layer_shed
        install_moe_layer_shed(raw_model, moe_layer_shed)
    processor = _make_text_processor(tokenizer)
    # mlx-lm-arch caches must carry the vlm runtime's class identities or
    # apc/ar isinstance-gates (own classes since mlx-vlm 0.6.4) resolve
    # model_apc_mode to None and APC silently disengages for this model.
    from .cache_compat import ensure_runtime_origin_make_cache
    ensure_runtime_origin_make_cache(raw_model)
    model = TextOnlyModel(raw_model, config=_as_dict(config))
    _ensure_inner_config(model, _AttrDict(_as_dict(config)))
    _ensure_text_embedding_probe(model, raw_model)
    return model, processor, model.config


_BRIDGE_FLAG = "_kq_gguf_server_bridge_installed"
_DRAFTER_PATCH_FLAG = "_kq_gguf_drafter_injection_installed"


def _install_drafter_injection() -> None:
    """Hand the engine the in-memory MTP drafter the loader built.

    Idempotent. The server's worker loads a drafter only when
    ``MLX_VLM_DRAFT_MODEL`` is set, via ``speculative.drafters.load_drafter(path,
    kind)`` (resolved at call time from the module). This patches that loader so a
    path present in :data:`_MTP_DRAFTER_STASH` returns the prebuilt
    ``(drafter, "mtp")`` pair; any other path falls through to the stock disk
    loader untouched. Pairing this in-memory drafter with the target keeps the
    K-quant drafter leaves intact (a disk reload would rebuild affine-quantized
    leaves and dequantize them)."""
    import importlib

    drafters = importlib.import_module("mlx_vlm.speculative.drafters")
    if getattr(drafters, _DRAFTER_PATCH_FLAG, False):
        return

    original = drafters.load_drafter

    def load_drafter(path_or_repo, kind=None, **kwargs):
        result = None
        if isinstance(path_or_repo, str):
            # Pop on consume: the engine keeps the drafter alive on its
            # ResponseGenerator; a lingering stash entry would pin the drafter
            # weights across evictions/reloads.
            stash = _MTP_DRAFTER_STASH.pop(os.path.abspath(path_or_repo), None)
            if stash is not None:
                result = stash
        if result is None:
            result = original(path_or_repo, kind=kind, **kwargs)
        _apply_draft_block_size_override(result)
        return result

    drafters.load_drafter = load_drafter
    setattr(drafters, _DRAFTER_PATCH_FLAG, True)


def _apply_draft_block_size_override(result) -> None:
    """Honor GMLX_DRAFT_BLOCK_SIZE (serve --draft-block-size): set the loaded
    drafter's config block size so the engine drafts N tokens/round. _dflash_block_total
    reads config.block_size when no explicit override is passed, so this covers both
    native-head (nextn) and two-GGUF assistant drafters. Best-effort; a frozen config
    or unset env is a no-op."""
    raw = os.environ.get("GMLX_DRAFT_BLOCK_SIZE", "").strip()
    if not raw:
        return
    try:
        n = int(raw)
    except ValueError:
        return
    if n <= 0:
        return
    drafter = result[0] if isinstance(result, tuple) else result
    cfg = getattr(drafter, "config", None)
    if cfg is None:
        return
    try:
        cfg.block_size = n
        if hasattr(cfg, "runtime_block_size"):
            cfg.runtime_block_size = n
    except Exception:
        pass  # frozen/odd config object -> keep the drafter's own default


def install_gguf_server_bridge() -> None:
    """Route ``*.gguf`` model paths in mlx-vlm's server through gmlx.

    Idempotent. Patches the single module-level loader
    ``server.generation.load_model_resources`` so a ``*.gguf`` model path is
    loaded via :func:`load_serveable_model` (text, VLM, or MTP per its
    registration); every other path uses mlx-vlm's original loader untouched.
    Also installs :func:`_install_drafter_injection` so a speculative load's
    in-memory drafter reaches the engine. Must be called before a model is loaded.
    """
    import importlib

    generation = importlib.import_module("mlx_vlm.server.generation")
    if getattr(generation, _BRIDGE_FLAG, False):
        return

    original = generation.load_model_resources

    def load_model_resources(model_path, adapter_path=None):
        # A prior speculative GGUF build's drafter env must never leak into this
        # load: the engine's load_drafter block fires whenever MLX_VLM_DRAFT_MODEL
        # is set, so a stale value would hand the K-quant MTP drafter to an
        # unrelated model (including a plain HF one on the fall-through path).
        # Pop unconditionally before branch dispatch; the MTP branch re-sets it
        # for its own build (builds are serialized by the residency build lock).
        os.environ.pop("MLX_VLM_DRAFT_MODEL", None)
        os.environ.pop("MLX_VLM_DRAFT_KIND", None)
        if _is_gguf(model_path):
            if adapter_path is not None:
                # The LoRA/DoRA adapter-apply path on a GGUF base is not built yet.
                # Surface it here rather than silently dropping the adapter:
                # this is the seam where that work threads adapter_path into
                # load_serveable_model and applies it to the loaded model.
                raise NotImplementedError(
                    f"adapter inference on a GGUF base is not supported yet "
                    f"(adapter_path={adapter_path!r})")
            # The config-resolved spec for the model now being built carries the
            # profile's chat-template override, if any. Read it from the build-spec
            # channel, not the request-thread ``_active_spec`` ContextVar: this bridge
            # runs in the engine's generation worker thread, where that ContextVar is
            # invisible. Residency publishes the build spec under its build lock around
            # the blocking stock load. In single-model mode there is no spec, so the
            # GGUF's own template is kept.
            spec = get_build_spec()
            chat_template = getattr(spec, "chat_template", None)
            # The GGUF LoRA adapter (config `adapter:` / `--adapter`) rides the same
            # build-spec channel as chat_template - set by residency under its build
            # lock, read here in the worker thread. It is applied to the loaded base in
            # load_serveable_model (text path; VLM/MTP+adapter raise there, never a
            # silent drop). Distinct from the stock ``adapter_path`` param above, which
            # is mlx-vlm's HF-PEFT mechanism (unsupported on a GGUF base).
            adapter_gguf = getattr(spec, "adapter", None)
            # The stream placement rides the same build-spec channel (config
            # `stream:` / `serve --stream-experts` / `--stream-cpu`): it is
            # applied to the loaded base in load_serveable_model (text path;
            # VLM/MTP raise).
            # The feeder overrides (config `prefill_feeder:`/`decode_feeder:` /
            # the paired serve flags) and the lossy MoE levers (config
            # `moe_experts:`/`moe_expert_mass:`/`moe_miss_shed:`/
            # `moe_layer_shed:` / the paired serve flags) ride along; None
            # keeps the loader default / trained fan-out.
            stream = getattr(spec, "stream", None)
            feeders = dict(
                moe_experts=getattr(spec, "moe_experts", None),
                moe_expert_mass=getattr(spec, "moe_expert_mass", None),
                moe_miss_shed=getattr(spec, "moe_miss_shed", None),
                moe_layer_shed=getattr(spec, "moe_layer_shed", None),
                feeder_prefill=getattr(spec, "prefill_feeder", None),
                feeder_decode=getattr(spec, "decode_feeder", None),
            )
            # Path registry + the per-build MLX_VLM_GGUF_SPECULATIVE env window
            # (set by residency) decide MTP vs plain - both process-global, so they
            # reach this bridge in the engine's generation worker thread, unlike the
            # request-thread spec ContextVar. A non-speculative id over an
            # MTP-registered GGUF (the lossless-oracle case) resolves to None here.
            vlm = _resolve_vlm_spec(model_path)
            mtp = _resolve_mtp_spec(model_path)
            if vlm is not None and mtp is not None:
                # VLM x MTP: a resident VLM whose text-only requests speculate. Arm
                # the drafter-load block (keyed to this GGUF -> the patched
                # load_drafter returns the in-memory drafter stashed at build) exactly
                # like the text MTP branch, but also pass the mmproj so image/audio
                # requests keep the full VLM forward.
                os.environ["MLX_VLM_DRAFT_MODEL"] = model_path
                os.environ["MLX_VLM_DRAFT_KIND"] = "mtp"
                return load_serveable_model(
                    model_path,
                    mmproj_path=vlm["mmproj_path"],
                    hf_source=vlm.get("hf_source"),
                    speculative=True,
                    draft_gguf_path=mtp.get("draft_gguf_path"),
                    chat_template=chat_template,
                    adapter_gguf=adapter_gguf,
                    stream=stream,
                    **feeders,
                )
            if vlm is not None:
                return load_serveable_model(
                    model_path,
                    mmproj_path=vlm["mmproj_path"],
                    hf_source=vlm.get("hf_source"),
                    adapter_gguf=adapter_gguf,
                    stream=stream,
                    **feeders,
                )
            if mtp is not None:
                # Trigger the stock drafter-load block (it runs only when
                # MLX_VLM_DRAFT_MODEL is set) and key it to this GGUF so the
                # patched load_drafter returns the in-memory drafter stashed
                # while the target loads just below.
                os.environ["MLX_VLM_DRAFT_MODEL"] = model_path
                os.environ["MLX_VLM_DRAFT_KIND"] = "mtp"
                return load_serveable_model(
                    model_path,
                    speculative=True,
                    draft_gguf_path=mtp.get("draft_gguf_path"),
                    chat_template=chat_template,
                    adapter_gguf=adapter_gguf,
                    stream=stream,
                    **feeders,
                )
            # Plain text load (stale drafter env already popped above).
            return load_serveable_model(model_path, chat_template=chat_template,
                                        adapter_gguf=adapter_gguf,
                                        stream=stream, **feeders)
        return original(model_path, adapter_path)

    generation.load_model_resources = load_model_resources
    setattr(generation, _BRIDGE_FLAG, True)
    _install_drafter_injection()


# Friendly-id resolution layer
#
# The bridge above is keyed by GGUF *path*. A config-driven server instead
# addresses models by a friendly *id* (``/v1/models`` id == request ``model``),
# optionally with an inline ``id@profile`` sampling/load override. This layer maps
# id -> concrete path + a fully-merged :class:`config.ResolvedModel`, registers each
# model's path-keyed companion spec (mmproj / drafter) so the load bridge still
# finds it, and exposes the resolved spec for *this* request through a ContextVar
# (mirroring residency's ``_active_entry`` discipline).

_RESOLVED_MODELS: dict[str, "object"] = {}     # id -> ResolvedModel
_PATH_TO_IDS: dict[str, list[str]] = {}        # abspath -> [id, ...]
_SERVER_CFG = None                             # the live ServerCfg (for re-resolve)
# The ResolvedModel for the request in flight - set at the residency seam, read at
# the gen-args seam. A ContextVar so concurrent requests don't see each other's.
_active_spec: ContextVar = ContextVar("_kq_active_spec", default=None)
# The ResolvedModel for the model *currently being built*. Distinct from
# ``_active_spec`` because the load bridge (``load_model_resources``) runs in the
# engine's generation worker thread, where a request-thread ContextVar is invisible.
# Residency sets this under its build lock around the blocking stock load (which the
# request thread waits on), so the bridge - and the profile chat-template it carries -
# crosses the thread boundary. A plain global is safe precisely because the build lock
# serialises builds: only one model is ever mid-build at a time.
_current_build_spec = None


class ModelNotFound(KeyError):
    """An addressed model id isn't configured. Carries the available ids so the
    HTTP layer can return a helpful 404 (never an HF fetch)."""

    def __init__(self, model_id: str, available):
        self.model_id = model_id
        self.available = sorted(available)
        self.message = f"unknown model id {model_id!r}; available: {self.available}"
        super().__init__(self.message)

    def __str__(self):
        # KeyError.__str__ is repr() - that would double-quote the message in
        # every HTTP error body built from str(exc).
        return self.message


class ModelFileMissing(ValueError):
    """An addressed model id is configured, but its GGUF is gone from disk
    (these entries are skipped at registration with a warning). Carries the id
    and the resolver's detail; the message - which becomes a 404 body - names
    the fix but not the detail, which lists local directories that must not
    reach API clients (it stays available on ``.detail`` for logs)."""

    def __init__(self, model_id: str, detail: str):
        self.model_id = model_id
        self.detail = detail
        super().__init__(
            f"model {model_id!r} is configured but its file is missing on "
            f"disk; restore the file, or run `gmlx sync-models` "
            f"to reconcile the config")


class UnknownProfile(ValueError):
    """A request named a profile that isn't defined (inline ``@profile`` or the
    ``profile`` field). Carries the valid profile names for a 400 body."""

    def __init__(self, profile: str, available):
        self.profile = profile
        self.available = sorted(available)
        super().__init__(
            f"unknown profile {profile!r}; available: {self.available}")


class NoModelSpecified(ValueError):
    """A request omitted ``model`` and there's no ``server.defaults.model`` and
    more than one configured model - so the target is ambiguous."""

    def __init__(self, available):
        self.available = sorted(available)
        super().__init__(
            "no model in request and no server.defaults.model set; specify one "
            f"of: {self.available}")


def _fill_families(cfg) -> None:
    """Fill each model's sampling ``family`` from its GGUF header, in place on
    the live cfg - so request-time re-resolution through
    :func:`config.resolve_model` sees it with no signature changes. The logic
    (and its silent-miss / kill-switch behaviour) lives in
    :func:`discovery.fill_families`, shared with the run/chat config overlay."""
    from .discovery import fill_families
    fill_families(cfg)


def _residency_build_lock():
    """The residency pool's build lock, or a no-op context when the pool is
    not installed (CLI paths). Registry rebuilds must not interleave with an
    in-flight model build: clearing _MTP_DRAFTER_STASH / the VLM/MTP
    registries mid-build strands the drafter the build is about to consume."""
    import contextlib
    pool = getattr(sys.modules.get("mlx_vlm.server"), "_kq_residency_pool", None)
    lock = getattr(pool, "_build_lock", None)
    return lock if lock is not None else contextlib.nullcontext()


def register_resolved_models(cfg) -> None:
    """Populate the id->spec tables from a :class:`config.ServerCfg` and register
    each model's path-keyed companion (mmproj / drafter) so the load bridge finds
    it at build time. Idempotent / reload-safe: the id tables and the path-keyed
    companion registries are rebuilt each call, so a reload that removes/adds an
    mmproj or drafter takes effect on the next cold load (a stale registry entry
    would otherwise keep loading the old VLM/speculative shape). The drafter
    stash goes with them - any live engine already owns its drafter.

    A model whose file is gone from disk is skipped with a warning instead of
    failing the whole server: registration runs at startup and on config
    reload, and one deleted GGUF must not take a multi-model server down. The
    entry stays in ``cfg.models``, so it disappears from ``/v1/models`` but
    self-heals on the next reload once the file is back.
    """
    global _SERVER_CFG
    _SERVER_CFG = cfg
    _fill_families(cfg)
    with _residency_build_lock():
        _register_resolved_models_locked(cfg)


def _clear_registries() -> None:
    _RESOLVED_MODELS.clear()
    _PATH_TO_IDS.clear()
    _GGUF_VLM_REGISTRY.clear()
    _GGUF_MTP_REGISTRY.clear()
    _MTP_DRAFTER_STASH.clear()


def _register_one(mid: str, rm) -> None:
    """Enter one resolved model in the id tables + companion registries."""
    _RESOLVED_MODELS[mid] = rm
    _PATH_TO_IDS.setdefault(rm.path, []).append(mid)
    # Companions are model-level (not profile-level), so they're stable per
    # path even when two ids back one GGUF under different load profiles.
    if rm.mmproj:
        register_gguf_vlm(rm.path, rm.mmproj)
    if rm.speculative:
        register_gguf_mtp(rm.path, draft_gguf_path=rm.draft_gguf)


def _register_resolved_models_locked(cfg) -> None:
    from .config import MissingModelFile, resolve_model

    _clear_registries()
    for root in getattr(cfg, "model_dirs", None) or []:
        expanded = os.path.expanduser(os.path.expandvars(root))
        if not os.path.isdir(expanded):
            print(f"[server] model_dirs root missing: {root!r} (a relative "
                  "root resolves against the server's own cwd; use an "
                  "absolute path, or restore/remount the directory)",
                  file=sys.stderr)
    skipped = []
    for mid in cfg.models:
        try:
            rm = resolve_model(mid, cfg)        # default-profile (no request) view
        except MissingModelFile as e:           # disk state, not config shape
            skipped.append((mid, e))
            continue
        _register_one(mid, rm)
    for mid, e in skipped:
        print(f"[server] skipping model {mid!r}: {e}", file=sys.stderr)
    if skipped:
        print(f"[server] {len(skipped)} of {len(cfg.models)} configured "
              "model(s) missing on disk; run `gmlx sync-models` to reconcile "
              "the config (drops gone entries, registers new files)",
              file=sys.stderr)


def reregister_missing_models() -> bool:
    """Re-try config entries that failed registration (file was missing). A
    restored file already self-heals on request (resolve_request_model
    re-resolves), but ``/v1/models`` reads the registration table - without
    this, a healed model serves fine while staying invisible to clients that
    enumerate models. Cheap when nothing is missing (a set-membership pass);
    one resolve attempt per still-missing id otherwise. True if any healed."""
    from .config import ConfigError, resolve_model

    cfg = _SERVER_CFG
    if cfg is None:
        return False
    healed = False
    for mid in cfg.models:
        if mid in _RESOLVED_MODELS:
            continue
        try:
            rm = resolve_model(mid, cfg)
        except ConfigError:
            continue                      # still missing (or still malformed)
        _register_one(mid, rm)
        print(f"[server] model {mid!r} is back on disk; re-registered",
              file=sys.stderr)
        healed = True
    return healed


def clear_resolved_models() -> None:
    """Drop the id tables, companion registries, drafter stash + config reference
    (test isolation / pre-reload)."""
    global _SERVER_CFG
    _SERVER_CFG = None
    _clear_registries()


def server_config():
    """The live :class:`config.ServerCfg`, or ``None`` if none is registered."""
    return _SERVER_CFG


def resolved_models() -> dict:
    """``{id: ResolvedModel}`` for every configured model (default-profile view).
    What ``/v1/models`` lists - never the HF cache."""
    return dict(_RESOLVED_MODELS)


def aliases() -> dict:
    """``{alias_name: (target_id, target_profile|None)}`` for every configured alias.
    Surfaced in ``/v1/models`` as pickable preset entries (an alias that carries a
    profile is a sampling preset clients can select without ``@profile`` syntax)."""
    cfg = _SERVER_CFG
    if cfg is None:
        return {}
    from .config import profile_names, split_address
    known = profile_names(cfg)
    return {name: split_address(target, known)
            for name, target in cfg.aliases.items()}


def default_model_id() -> str | None:
    """A best-effort default model id for clients - the ``default`` marker in
    ``/v1/models`` and what ``launch`` writes as a harness default. Unambiguous picks
    only: ``server.defaults.model``, else a lone pinned model, else the sole model,
    else ``None`` (let the client choose). Distinct from :func:`_default_model_id`,
    the *strict* empty-``model`` request fallback that 400s on ambiguity."""
    cfg = _SERVER_CFG
    if cfg is None:
        return None
    if cfg.defaults.model:
        return cfg.defaults.model
    pinned = [mid for mid, m in cfg.models.items() if m.pin]
    if len(pinned) == 1:
        return pinned[0]
    if len(cfg.models) == 1:
        return next(iter(cfg.models))
    return None


def split_profile_address(model_field: str) -> tuple[str, str | None]:
    """Split ``id@profile`` (or ``alias@profile``) on the **last** ``@``. The right
    side is taken as a profile only if it names a known one, so hf ``org/model@rev``
    and ids/aliases containing ``@``-like text stay intact. ``(head, profile|None)``."""
    if _SERVER_CFG is None:
        return model_field, None
    from .config import profile_names, split_address
    return split_address(model_field, profile_names(_SERVER_CFG))


def _default_model_id() -> str:
    cfg = _SERVER_CFG
    if cfg.defaults.model:
        if cfg.defaults.model not in cfg.models:
            raise ModelNotFound(cfg.defaults.model, cfg.models)
        return cfg.defaults.model
    if len(cfg.models) == 1:
        return next(iter(cfg.models))
    raise NoModelSpecified(cfg.models)


def resolve_request_model(model_field: str | None, *,
                          profile_field: str | None = None):
    """Resolve a request ``model`` field to ``(abspath, ResolvedModel)``.

    Handles the empty field (-> ``server.defaults.model`` / the sole model), an
    **alias** (a friendly name / profile preset, expanded to its target id + baked
    profile), an inline ``id@profile`` / ``alias@profile`` override, and an explicit
    ``profile`` request field. Profile precedence: inline ``@profile`` > request
    ``profile`` field > an alias's baked profile. An unknown id raises
    :class:`ModelNotFound`, a configured id whose file is gone
    :class:`ModelFileMissing`, and an unknown profile :class:`UnknownProfile` -
    **never** an HF fetch. Re-resolves through :func:`config.resolve_model` so the
    chosen profile reshapes the merged sampling/load spec (and a skipped model
    self-heals here the moment its file is back)."""
    from .config import MissingModelFile, profile_names, resolve_model, split_address

    cfg = _SERVER_CFG
    if cfg is None:
        raise RuntimeError(
            "no config registered; call register_resolved_models() first")

    known = profile_names(cfg)
    raw = (model_field or "").strip()
    if not raw:
        model_id = _default_model_id()
        request_profile = profile_field
    else:
        head, inline_profile = split_profile_address(raw)
        base_profile = None
        if head in cfg.aliases:                       # expand alias -> id (+ profile)
            head, base_profile = split_address(cfg.aliases[head], known)
        model_id = head
        # inline @profile > request `profile` field > alias's baked profile
        request_profile = inline_profile or profile_field or base_profile

    if request_profile is not None and request_profile not in known:
        raise UnknownProfile(request_profile, known)
    if model_id not in cfg.models:
        # `<id|alias>@suffix` with a real head but an unknown suffix isn't split off
        # as a profile (last-@ rule keeps hf `org/model@rev` intact), so it lands
        # here whole - surface it as the unknown profile it really is.
        if "@" in raw:
            h, t = raw.rsplit("@", 1)
            if h in cfg.models or h in cfg.aliases:
                raise UnknownProfile(t, known)
        raise ModelNotFound(model_id, cfg.models)

    try:
        rm = resolve_model(model_id, cfg, request_profile=request_profile)
    except MissingModelFile as e:
        raise ModelFileMissing(model_id, str(e)) from None
    return rm.path, rm


def set_active_spec(spec):
    """Bind the ResolvedModel for the in-flight request; returns a reset token."""
    return _active_spec.set(spec)


def get_active_spec():
    """The ResolvedModel bound for this request, or ``None``."""
    return _active_spec.get()


def reset_active_spec(token) -> None:
    _active_spec.reset(token)


# The request body's `profile` field, captured at the route wrapper (see
# server_patches.install_request_profile_capture) so the residency seam - which
# only receives the model string - can hand it to resolve_request_model. A
# ContextVar: it crosses `asyncio.to_thread` (contextvars are copied), so the
# chat load-offload pre-warm resolves with the same profile; preload / TTL /
# warm threads have no request context and see the default None.
_request_profile: ContextVar = ContextVar("_kq_request_profile", default=None)


def set_request_profile(profile):
    """Bind the in-flight request's body ``profile``; returns a reset token."""
    return _request_profile.set(profile)


def get_request_profile():
    """The body ``profile`` for this request, or ``None``."""
    return _request_profile.get()


def reset_request_profile(token) -> None:
    _request_profile.reset(token)


def set_build_spec(spec) -> None:
    """Publish the ResolvedModel for the model now being built so the load bridge,
    running in the engine's generation worker thread, can read its profile overrides
    (e.g. ``chat_template``). Residency calls this under its build lock around the
    blocking stock load; pass ``None`` to clear it afterward."""
    global _current_build_spec
    _current_build_spec = spec


def get_build_spec():
    """The ResolvedModel for the in-flight build, or ``None`` (single-model mode, or
    no build in progress). Safe to read from any thread - set under the build lock."""
    return _current_build_spec
