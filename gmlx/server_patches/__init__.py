"""Config-driven HTTP-surface patches over mlx-vlm's FastAPI server.

mlx-vlm's model-management surface assumes a single HF-resolved model. For a
GGUF-only, multi-model, config-driven server that surface is wrong in five places;
this module installs late-bound monkeypatches over each seam (the same no-fork
pattern as :mod:`server_bridge_vlm` / :mod:`residency`):

* **Sampling-profile injection** - a request's unset sampling fields take their
  values from the resolved profile (``serving.get_active_spec()``), not mlx-vlm's
  hardcoded schema defaults. ``_build_gen_args`` reads temperature/top_p/top_k/
  min_p/penalties with plain ``getattr`` against pydantic defaults, so a profile
  can only win by post-processing the built args per key, honouring
  ``request.model_fields_set`` (a client-set field always wins).
* **``/v1/models`` override** - lists the configured/discovered ids (with
  resident / pinned / capability markers), never ``scan_cache_dir()``.
* **HF-download gate** - a non-local, non-GGUF id with HF disabled raises instead
  of triggering a snapshot download; with ``hf_cache`` on, HF resolves from the
  local cache only (``HF_HUB_OFFLINE``), never the network.
* **Runtime-snapshot enrichment** - ``/health`` and ``/v1/metrics`` gain a
  ``resident_models[]`` view of the whole residency pool (idle/ttl/footprint/busy).
* **Pool-aware ``/unload``** - an optional ``{"model": "<id>"}`` evicts that one
  resident model; no body clears all. Plus a ``/v1/reload`` hook.
* **Vanilla streaming chunks** - streaming chat-completion chunks serialize with
  ``exclude_none`` so they drop mlx-vlm's non-standard ``timings: null`` (and the
  other null fields). The null ``timings`` otherwise crashes Open WebUI's stream
  relay (``{}.update(None)`` -> ``TypeError`` -> every content chunk dropped -> blank
  message).
* **XTC sampling** - ``xtc_probability`` / ``xtc_threshold`` (request extras or
  profile sampling) become a per-request logits processor; the batch engine's
  own sampler only knows temperature/top_p.
* **OpenAI ``stop``** - the chat-completions routes gain OpenAI-API stop
  sequences (mlx-vlm only implements them on the Anthropic endpoint).
* **API contract** - one warning line per request naming set-but-unread
  parameters (per-dialect allowlists), server-side ``tool_choice: "none"``
  enforcement, and a warning when a forced tool call parsed zero calls.
* **``/v1/completions``** - a minimal classic text-completions route (single
  string prompt, ``n=1``, SSE streaming, stop sequences); mlx-vlm serves only
  the chat-shaped routes.
* **SSE keepalive** - the streaming routes emit SSE comment lines while the
  engine is silent (a deep-context dense prefill can run >10 minutes before the
  first token), so clients with a between-bytes read timeout don't tear the
  socket down mid-prefill.
* **Off-loop model load** - the chat routes pre-warm the request's model on a
  worker thread before the stock handler runs. mlx-vlm calls ``get_cached_model``
  synchronously inside the async handler, so a cold load / model swap would block
  the single event loop (``/health`` and every sibling request stall until it
  finishes); pre-warming keeps the loop free and makes the handler's own call a
  cache hit. Pairs with the background preload warm (``spawn_preload_warm``),
  which moves the startup preload off the blocking lifespan path.
* **Hardening** - optional API-key auth (every route but ``/health``), a
  DNS-rebinding Host guard on loopback binds, credential-less CORS, and a
  ``/health`` body trimmed to liveness (no filesystem paths).

The numerics, batching, and protocol handlers stay stock.
"""

from __future__ import annotations

import importlib
import os

from ._common import _PATCH_FLAG
from .apc import install_apc_batched_store_eval, install_apc_lone_harvest
from .api_contract import install_api_contract
from .chat_behavior import (
    install_chat_template_kwargs,
    install_ignore_eos,
    install_openai_stop_sequences,
    install_stream_thinking_seed,
    install_thinking_budget_fix,
    install_vanilla_stream_chunks,
)
from .completions import install_completions_route
from .hardening import (
    disable_credentialed_cors,
    install_api_key_auth,
    install_health_liveness_override,
    install_json_content_type_tolerance,
    install_loopback_host_guard,
)
from .observability import install_request_timing_log, uvicorn_log_config
from .request_flow import (
    install_chat_load_offload,
    install_optional_request_model,
    install_request_profile_capture,
    install_sse_keepalive,
)
from .routes import (
    HFAccessDisabled,
    install_audio_speech_route,
    install_audio_transcription_route,
    install_audio_translation_route,
    install_audio_voices_route,
    install_auto_docs_removal,
    install_embeddings_route,
    install_hf_download_gate,
    install_keep_route,
    install_models_endpoint_override,
    install_pool_aware_unload,
    install_reload_route,
    install_rerank_route,
    install_resolver_error_handlers,
    install_runtime_snapshot_enrichment,
    spawn_preload_warm,
)
from .sampling import (
    install_fast_sampler,
    install_gen_args_profile_injection,
    install_xtc_sampling,
)

__all__ = [
    "HFAccessDisabled",
    "disable_credentialed_cors",
    "install_apc_batched_store_eval",
    "install_apc_lone_harvest",
    "install_api_contract",
    "install_api_key_auth",
    "install_audio_speech_route",
    "install_audio_transcription_route",
    "install_audio_translation_route",
    "install_audio_voices_route",
    "install_auto_docs_removal",
    "install_chat_load_offload",
    "install_chat_template_kwargs",
    "install_completions_route",
    "install_embeddings_route",
    "install_fast_sampler",
    "install_gen_args_profile_injection",
    "install_health_liveness_override",
    "install_hf_download_gate",
    "install_ignore_eos",
    "install_json_content_type_tolerance",
    "install_keep_route",
    "install_loopback_host_guard",
    "install_models_endpoint_override",
    "install_openai_stop_sequences",
    "install_optional_request_model",
    "install_pool_aware_unload",
    "install_reload_route",
    "install_request_profile_capture",
    "install_request_timing_log",
    "install_rerank_route",
    "install_resolver_error_handlers",
    "install_runtime_snapshot_enrichment",
    "install_server_patches",
    "install_sse_keepalive",
    "install_stream_thinking_seed",
    "install_thinking_budget_fix",
    "install_vanilla_stream_chunks",
    "install_xtc_sampling",
    "spawn_preload_warm",
    "uvicorn_log_config",
]


def install_server_patches(cfg, *, reload_fn=None) -> None:
    """Install the full config-driven HTTP surface for a registered ``ServerCfg``.
    Call after register_resolved_models + the bridge/residency installs, before
    ``uvicorn.run``."""
    from ..config import LOOPBACK_HOSTS

    install_api_key_auth(getattr(cfg, "api_key", None))
    install_json_content_type_tolerance()
    if getattr(cfg, "host", None) in LOOPBACK_HOSTS:
        install_loopback_host_guard(cfg.host)
    disable_credentialed_cors()
    install_health_liveness_override()
    install_gen_args_profile_injection()
    install_vanilla_stream_chunks()
    install_xtc_sampling()
    if os.environ.get("GMLX_STEP_LOG"):
        from ..step_timing import install_step_timing
        install_step_timing()
    if os.environ.get("GMLX_DISABLE_FAST_SAMPLER") != "1":
        install_fast_sampler()
    from .. import spec_engine
    spec_engine.install_full_prompt_mtp_prefill()
    spec_engine.install_owned_spec_engine()
    spec_engine.install_continuous_batch_admission()
    spec_engine.install_spec_kv_quant()
    from ..apc_pooling import (
        install_pooled_prompt_kv_quant,
        install_pooling_apc_support,
        install_safe_kv_quantization,
    )
    install_pooling_apc_support()
    install_safe_kv_quantization()
    install_pooled_prompt_kv_quant()
    install_chat_template_kwargs()
    install_thinking_budget_fix()
    install_stream_thinking_seed()
    install_openai_stop_sequences()
    install_api_contract()
    # Before the load-offload / profile-capture / keepalive wrappers so they
    # wrap the completions route too.
    install_completions_route()
    install_chat_load_offload()
    install_optional_request_model()
    install_request_profile_capture()
    install_models_endpoint_override(
        stt_model=getattr(cfg, "stt", None),
        tts_model=getattr(cfg, "tts", None),
        embeddings_model=getattr(cfg, "embeddings", None),
        rerank_model=getattr(cfg, "rerank", None),
        model_dirs=getattr(cfg, "model_dirs", ()) or ())
    install_auto_docs_removal()
    install_hf_download_gate(bool(getattr(cfg, "hf_cache", False)))
    install_runtime_snapshot_enrichment()
    install_pool_aware_unload()
    install_apc_lone_harvest()
    install_apc_batched_store_eval()
    install_keep_route()
    install_reload_route(reload_fn)
    install_audio_transcription_route(getattr(cfg, "stt", None))
    install_audio_translation_route(getattr(cfg, "stt", None))
    install_audio_speech_route(getattr(cfg, "tts", None))
    install_audio_voices_route(getattr(cfg, "tts", None))
    install_embeddings_route(getattr(cfg, "embeddings", None))
    install_rerank_route(getattr(cfg, "rerank", None))
    install_resolver_error_handlers()
    install_request_timing_log()
    # Last: the assistant chat wrapper must be outermost (alias ids never
    # reach the model resolver) and wrap the models override above.
    from ..assistant_serve import install_assistant_serve
    install_assistant_serve(cfg)
    # True last: the keepalive body wrapper must be outermost on every
    # streaming route, including the assistant-re-registered chat routes.
    install_sse_keepalive()
    app = importlib.import_module("mlx_vlm.server.app")
    setattr(app, _PATCH_FLAG, True)
