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
* **Faithful history render** - the chat routes' ``apply_chat_template``
  keeps message keys the per-model rebuild does not produce (notably
  ``reasoning_content``), so ``preserve_thinking`` reaches the chat template
  on non-tool turns instead of being silently dropped for model types in
  mlx-vlm's ``MODEL_CONFIG`` table.
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
from .apc import (
    install_apc_lone_harvest,
    install_retire_render_capture,
)
from .api_contract import install_api_contract
from .chat_behavior import (
    install_chat_template_kwargs,
    install_ignore_eos,
    install_openai_stop_sequences,
    install_role_normalization,
    install_stream_thinking_seed,
    install_stream_timings,
    install_thinking_budget_fix,
    install_vanilla_stream_chunks,
)
from .capacity_routes import (
    install_capacity_plan,
    install_estimate,
    install_health_readiness,
    install_metrics_prometheus,
    install_scoped_cache_reset,
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
from .render import install_faithful_history
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
    "install_apc_lone_harvest",
    "install_retire_render_capture",
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
    "install_faithful_history",
    "install_fast_sampler",
    "install_gen_args_profile_injection",
    "install_health_liveness_override",
    "install_health_readiness",
    "install_hf_download_gate",
    "install_ignore_eos",
    "install_json_content_type_tolerance",
    "install_keep_route",
    "install_loopback_host_guard",
    "install_metrics_prometheus",
    "install_models_endpoint_override",
    "install_openai_stop_sequences",
    "install_optional_request_model",
    "install_pool_aware_unload",
    "install_reload_route",
    "install_request_profile_capture",
    "install_request_timing_log",
    "install_role_normalization",
    "install_rerank_route",
    "install_scoped_cache_reset",
    "install_capacity_plan",
    "install_estimate",
    "install_resolver_error_handlers",
    "install_runtime_snapshot_enrichment",
    "install_server_patches",
    "install_sse_keepalive",
    "install_stream_thinking_seed",
    "install_stream_timings",
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
    from gmlx.config import LOOPBACK_HOSTS

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
    from ..seed_rows import install_per_request_seed
    install_per_request_seed()
    import gmlx.spec.engine as spec_engine
    spec_engine.install_full_prompt_mtp_prefill()
    spec_engine.install_owned_spec_engine()
    spec_engine.install_continuous_batch_admission()
    spec_engine.install_spec_kv_quant()
    from ..batch_sched import install_decode_priority_sched
    install_decode_priority_sched()
    from gmlx.cache.apc_pooling import (
        install_batched_cachelist_admission,
        install_pooled_prefill_batch_gate,
        install_pooled_prompt_kv_quant,
        install_pooling_apc_support,
        install_safe_kv_quantization,
    )
    install_pooling_apc_support()
    install_safe_kv_quantization()
    install_pooled_prompt_kv_quant()
    install_pooled_prefill_batch_gate()
    # Before the model loads, so the cascade stamp wrapper (installed at load
    # time) wraps this and both survive.
    install_batched_cachelist_admission()
    # After the pacer above so this wrapper runs outside it: a declined
    # tick hides the pending list before the pacer looks, the pacer sees
    # no prefill work, and decode runs unpaced while admission waits.
    from ..admit_gate import install_admit_headroom_gate
    install_admit_headroom_gate()
    # After the headroom gate so a truncated tick still projects memory
    # for the rows it admits.
    from gmlx.cache.fresh_gate import install_fresh_admission_gate
    install_fresh_admission_gate()
    from gmlx.cache.kvarn_serve import install_kvarn_serve
    install_kvarn_serve()
    from gmlx.cache.kvarn_apc import install_kvarn_apc
    install_kvarn_apc()
    install_chat_template_kwargs()
    install_thinking_budget_fix()
    install_stream_timings()
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
    # After the liveness override (replaces its handler) and the residency
    # pool install (the scoped reset walks the pool's entries).
    install_health_readiness()
    install_metrics_prometheus()
    install_scoped_cache_reset()
    install_capacity_plan()
    install_estimate()
    # Render-wrap nesting per target: seed(retire(faithful(orig))).
    # Faithful innermost so the retire capture memoizes it and predictions
    # see the render the server produces. Seed outermost for two reasons:
    # the faithful wrap's inner return_messages=True call would feed an
    # inner stash the message list ("no prompt"), and the captured render
    # must exclude the stash so off-request predictions never write the
    # request contextvar the thinking splitters read.
    install_faithful_history()
    install_apc_lone_harvest()
    install_retire_render_capture()
    install_stream_thinking_seed()
    # Outermost render wrap: the retire capture below it must memoize the
    # normalized messages, so the next-turn prediction renders identically.
    install_role_normalization()
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
    from ..queue_cap import install_queue_depth_cap
    install_queue_depth_cap()
    from ..mem_preflight import install_memory_preflight
    install_memory_preflight()
    # Late so the trace brackets the full tick including pacing and
    # admission work.
    from gmlx.serve.memtrace import install_serve_memtrace
    install_serve_memtrace()
    # Outside the trace (band decisions and shed work show up as tick
    # time, which the trace should attribute) and inside the tick guard
    # (a memory error a governor action itself trips must be contained
    # like any other).
    from ..governor import install_governor
    install_governor()
    # The rows the guard or the governor fail permanently must reach
    # their handlers (typed error + close on the response queue).
    from .row_failed import install_row_failed_bridge
    install_row_failed_bridge()
    # Outside the row-failed bridge (reads ``active`` after the bridge has
    # popped shed rows, so a shed uid never lingers in the live view).
    from ..live_requests import install_live_requests
    install_live_requests()
    # The true last BatchGenerator._next wrapper (outermost): the tick
    # guard must see every error the tick can raise, and its recovery
    # time must not pollute the trace's tick bracket.
    from ..tick_guard import install_tick_guard
    install_tick_guard()
    # Last: the assistant chat wrapper must be outermost (alias ids never
    # reach the model resolver) and wrap the models override above.
    from gmlx.assistant.serve import install_assistant_serve
    install_assistant_serve(cfg)
    # True last: the keepalive body wrapper must be outermost on every
    # streaming route, including the assistant-re-registered chat routes.
    install_sse_keepalive()
    app = importlib.import_module("mlx_vlm.server.app")
    setattr(app, _PATCH_FLAG, True)
