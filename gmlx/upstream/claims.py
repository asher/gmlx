"""The claims table: the names in the copied mlx-vlm and mlx-lm that gmlx
rebinds when it installs, or builds on through a subclass.

gmlx copies mlx-vlm and mlx-lm whole into ``gmlx._vendor``. Much gmlx
behavior still lives in installers that rebind a name of the copy at load
or serve time: the claims. Each row names one claimed name and the gmlx
code that claims it. A fold moves a claim's behavior into the copy at the
definition, marked ``# gmlx:``, and deletes the installer, its install call
and its row.

A row's ``how`` says what holds once the installers ran:

- ``wrap``: the name leads to a gmlx object, a wrapper or a replacement,
  and its wrap chain holds code from each module ``wrappers`` lists.
- ``route``: the name keeps the copy's endpoint, and each POST route in
  ``paths`` serves a gmlx endpoint in its place.
- ``app``: the app serves at least one gmlx endpoint.
- ``subclass``: nothing is installed. A gmlx class derives from the class
  that holds the name and relies on it. ``by`` names that class when it is
  defined at module level.

A ``wrap`` or ``route`` row is installed by the serve startup, or by the
installer ``install`` names, which runs with no arguments. An installer
that takes the loaded model gets an empty module: it installs its class
dispatch and flags no instance.

``check_claims()`` holds that every name exists in the copy and that each
named installer and subclass exists. With ``installed=True`` it also holds
that every claim took. ``python -m gmlx.upstream.claims --installed`` runs
the serve startup and every installer the table names first, so it needs a
fresh process.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import os
import sys
from dataclasses import dataclass

_GMLX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COPY_DIR = os.path.join(_GMLX_DIR, "_vendor")
_COPY_ROOTS = ("gmlx._vendor.mlx_vlm.", "gmlx._vendor.mlx_lm.")
# The root packages are modules of the copy too: their bindings re-export.
_COPY_PACKAGES = ("gmlx._vendor.mlx_vlm", "gmlx._vendor.mlx_lm")
_HOWS = ("wrap", "route", "app", "subclass")


@dataclass(frozen=True)
class Claim:
    module: str      # a module of the copy
    attr: str        # dotted attribute path in that module
    # The gmlx code that claims it, for the failure message. A wrap row
    # names only code in its chain or its installer; a comment above the
    # row names other code that relies on the name.
    used_by: str
    how: str = "wrap"
    install: str = ""  # "module:function"; "" is the serve startup
    paths: tuple[str, ...] = ()  # the POST routes of a "route" claim
    by: str = ""     # "module:Class" of a "subclass" claim's gmlx class
    # The gmlx modules whose code the wrap chain of a "wrap" claim holds
    # once the installers ran, one entry per installer that wraps the name.
    wrappers: tuple[str, ...] = ()


_G4_NOSYNC = "gmlx.models.gemma4.sync:install_gemma4_nosync"
_APC_MANAGER = "gmlx.cache.apc_manager:GmlxAPCManager"
# Load-time installers the serve startup does not run.
_CASCADE = "gmlx.upstream.cascade_sdpa:install_cascade_stamp"

CLAIMS: tuple[Claim, ...] = (
    # --- batched serve scheduler ---
    # memtrace brackets each tick outermost only when GMLX_SERVE_MEMSTATS
    # is set, so the installed check does not see it.
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._next",
          "batch_sched.install_decode_priority_sched (decode-first tick, "
          "prompt-arm structure, _prompt_time_counter contract); "
          "admit_gate.install_admit_headroom_gate (admission-arm gating "
          "via the pending-list stash); "
          "mtp_prefill._install_l0_tenant_rows (publishes the generator's "
          "tenant rows for the step; the prompt batch is built and first "
          "stepped inside _next)",
          wrappers=("gmlx.cache.fresh_gate", "gmlx.serve.admit_gate",
                    "gmlx.serve.batch_sched", "gmlx.serve.governor",
                    "gmlx.serve.queue_cap", "gmlx.serve.tick_guard",
                    "gmlx.spec.mtp_prefill")),
    # batch_sched's arrival merge relies on insert appending to or
    # rebinding _unprocessed_sequences.
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator.insert",
          "lora_rows (per-row prompt_kwargs carry the APC "
          "salt, positional order prompts, max_tokens, prompt_kwargs); "
          "seed_rows (returned uids); mtp_prefill._install_l0_tenant_rows "
          "(each row's _apc_tenant kwarg, None included, on the generator "
          "by returned uid)",
          wrappers=("gmlx.lora_rows", "gmlx.serve.seed_rows",
                    "gmlx.spec.mtp_prefill")),
    # --- gemma4 host-sync-free masks and offsets. Owned gemma4_text MTP
    # trees shadow these by subclass override and mirror the same bodies
    # (gemma4.owned._owned_make_masks and _owned_attention_call); the
    # claims stay live for trees built from the copy. ---
    Claim("gmlx._vendor.mlx_vlm.models.gemma4.language",
          "Gemma4TextModel._make_masks",
          "gemma4.sync.install_gemma4_nosync (body carried and "
          "mirror-tested, offset probe; the carried body, or the stock one "
          "under GMLX_G4_NOSYNC=0, runs under the image mask builder)",
          install=_G4_NOSYNC,
          wrappers=("gmlx.models.gemma4.sync",)),
    # A correctness fix installed by the same call, ahead of its kill switch.
    Claim("gmlx._vendor.mlx_vlm.models.gemma4.language",
          "LanguageModel.chunked_prefill_policy",
          "gemma4.sync._install_image_chunking_policy (wrap: under "
          "bidirectional image attention answers as for a causal model, "
          "since the prefill loops keep image blocks whole; the owned MTP "
          "target inherits it)",
          install=_G4_NOSYNC,
          wrappers=("gmlx.models.gemma4.sync",)),
    Claim("gmlx._vendor.mlx_vlm.models.gemma4_unified.gemma4_unified",
          "Model._should_disable_chunked_prefill",
          "gemma4.sync._install_image_chunking_policy (wrap: never "
          "declines, so the per-request stamp stays off)",
          install=_G4_NOSYNC,
          wrappers=("gmlx.models.gemma4.sync",)),
    Claim("gmlx._vendor.mlx_vlm.models.gemma4.language", "Attention.__call__",
          "gemma4.sync.install_gemma4_nosync (body carried, offset wrap)",
          install=_G4_NOSYNC,
          wrappers=("gmlx.models.gemma4.sync",)),
    Claim("gmlx._vendor.mlx_vlm.models.gemma4.language", "LanguageModel.__call__",
          "gemma4 verifier gate (speculative_verify routing); "
          "gemma4.owned inherits it",
          how="subclass", by="gmlx.models.gemma4.owned:OwnedGemma4LanguageModel"),
    Claim("gmlx._vendor.mlx_vlm.models.gemma4.language", "Gemma4TextModel.__call__",
          "gemma4.owned inherits it (owned masks and attention take the "
          "per-layer mask, shared KV and offset it passes, including the "
          "logits_to_keep trim before the KV-shared layers)",
          how="subclass", by="gmlx.models.gemma4.owned:OwnedGemma4TextModel"),
    # --- prompt batches, speculative and AR batch engine ---
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "PromptProcessingBatch.__init__",
          "cascade_sdpa.install_cascade_stamp (token-prefix stamp at "
          "batch formation; rows and caches coexist in matching order "
          "here); kvarn_serve._install_ppb_fastpath (B=1 fast path skips "
          "_make_cache; kvarn rebuilds the empty cache); "
          "mtp_prefill.install_full_prompt_mtp_prefill (prefill-step "
          "restore, stock-path ckpt arming, packed warm_cache kwarg "
          "recovered to fp16 before init); "
          "apc_pooling.install_pooled_prompt_kv_quant",
          install=_CASCADE,
          wrappers=("gmlx.cache.apc_pooling", "gmlx.cache.kvarn_serve",
                    "gmlx.spec.mtp_prefill", "gmlx.upstream.cascade_sdpa")),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "_extend_cache",
          "cascade_sdpa.install_cascade_stamp (stamp carry across the "
          "B=1-to-batch merge lift on admission); "
          "apc_pooling.install_batched_cachelist_admission (promotion "
          "test looks through CacheList so an already-batched one is not "
          "merged twice)",
          install=_CASCADE,
          wrappers=("gmlx.cache.apc_pooling", "gmlx.upstream.cascade_sdpa")),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator.__init__",
          "spec.engine._install_apc_manager_stash; kvarn_serve APC gate; "
          "apc_pooling.install_pooled_prefill_batch_gate (prompt batches "
          "stay B=1 on pooling-cache models; to_batch_cache has no pooled "
          "arm)",
          wrappers=("gmlx.cache.apc_pooling", "gmlx.cache.kvarn_serve",
                    "gmlx.spec.engine")),
    # cache.refusals relies on it too: to_batch is tried before the
    # isinstance chain.
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "_make_cache",
          "kvarn_serve._install_make_cache (batch kvarn KV construction)",
          wrappers=("gmlx.cache.kvarn_serve",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar",
          "PromptProcessingBatch._store_apc_exact_checkpoints",
          "ckpt._install_ckpt_checkpoint_store (ckpt cursor rides the "
          "stock store)",
          wrappers=("gmlx.spec.ckpt",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "PromptProcessingBatch.prompt_progress",
          "ckpt._install_restored_prompt_progress (a gmlx-owned restore "
          "reported as the row's cached tokens)",
          wrappers=("gmlx.spec.ckpt",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "GenerationBatch._step",
          "ckpt._install_plain_ckpt_decode (token accounting and "
          "decode-time snapshots)",
          wrappers=("gmlx.lora_rows", "gmlx.serve.cb_phase",
                    "gmlx.serve.seed_rows", "gmlx.spec.ckpt")),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "GenerationBatch.filter",
          "ckpt._install_plain_ckpt_decode (B=1 retirement at row exit)",
          wrappers=("gmlx.spec.ckpt",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "PromptProcessingBatch.prompt_step",
          "mtp_prefill.install_full_prompt_mtp_prefill; "
          "media_spans.install_span_aware_prompt_step (the block step; a "
          "chunk past right-padded row ends rides the _suffix_lens end "
          "clip and the _speculative_prefill.append output)",
          install="gmlx.gen.media_spans:install_span_aware_prompt_step",
          wrappers=("gmlx.gen.media_spans", "gmlx.lora_rows",
                    "gmlx.serve.cb_phase", "gmlx.spec.mtp_prefill")),
    Claim("gmlx._vendor.mlx_vlm.generate.ar",
          "PromptProcessingBatch.needs_processing",
          "prefill_decay.install_prefill_decay (a batch that hands "
          "generate() its last step's columns reads the decayed step)",
          install="gmlx.gen.prefill_decay:install_prefill_decay",
          wrappers=("gmlx.gen.prefill_decay",)),
    # stream_generate calls the binding dispatch imported from ar, so
    # media_spans wraps that one; cb_phase wraps ar's own, which the CLI
    # generate path imports by name.
    Claim("gmlx._vendor.mlx_vlm.generate.dispatch", "generate_step",
          "media_spans.install_block_aware_generate_step (block starts "
          "ride prompt_cache_checkpoint_lengths through a pass-through "
          "prompt_cache_checkpoint; _TailForward reads the chunk call's "
          "inputs, inputs_embeds and n_to_process keywords, and the final "
          "_step call's positional ids, from the model's language_model)",
          install="gmlx.gen.media_spans:install_block_aware_generate_step",
          wrappers=("gmlx.gen.media_spans",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "generate_step",
          "cb_phase.install_cb_phase_steps (phased steps around the loop)",
          wrappers=("gmlx.serve.cb_phase",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "PromptProcessingBatch.generate",
          "mtp_prefill.install_full_prompt_mtp_prefill; "
          "seed_rows.install_per_request_seed (row-uid publish); "
          "patches.mtp_thinking (thinking-hook transport); "
          "patches.spec_processors (reads each row's processors and "
          "prompt tokens before the prefill clears them, outermost)",
          wrappers=("gmlx.lora_rows", "gmlx.serve.cb_phase",
                    "gmlx.serve.patches.mtp_thinking",
                    "gmlx.serve.patches.spec_processors", "gmlx.serve.seed_rows",
                    "gmlx.spec.mtp_prefill")),
    Claim("gmlx._vendor.mlx_vlm.generate.ar",
          "SpeculativeGenerationBatch._start_rounds",
          "patches.spec_processors (stashes the row processors on the first "
          "cache entry before the rounds generator starts)",
          wrappers=("gmlx.serve.patches.spec_processors",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "SpeculativeGenerationBatch.next",
          "admission.install_continuous_batch_admission",
          wrappers=("gmlx.serve.cb_phase", "gmlx.serve.seed_rows",
                    "gmlx.spec.admission")),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "SpeculativeGenerationBatch.filter",
          "admission._filter_with_release (per-row release rides the "
          "mark-finished contract; the rounds generator sheds via "
          "stop_check)",
          wrappers=("gmlx.spec.admission",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "SpeculativeGenerationBatch.__len__",
          "admission._len_with_promotion (patched semantics are D-3's "
          "hazard; decision modules count rows via _orig_len only)",
          wrappers=("gmlx.spec.admission",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "run_speculative_server_rounds",
          "spec.engine.install_owned_spec_engine",
          wrappers=("gmlx.spec.engine",)),
    Claim("gmlx._vendor.mlx_vlm.speculative.utils", "make_speculative_prompt_cache",
          "spec.engine.install_owned_b1_spec_cache (B=1 MTP single-stream "
          "caches); kv_quant.install_spec_kv_quant (B=1 KV_BITS/kvarn)",
          wrappers=("gmlx.spec.engine", "gmlx.spec.kv_quant")),
    # spec.engine._bind_l1_view binds these three, unbound, as its L1 APC
    # helpers.
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._apc_pick_for",
          "ckpt._install_exact_anchor_pick (the anchor LRU inside the stock "
          "admission pick)",
          wrappers=("gmlx.spec.ckpt",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._apc_extra_hash",
          "media_salt.install_media_salt (a MediaSalt passes whole)",
          wrappers=("gmlx.cache.media_salt",)),
    # Its guard reads self.model's config, the language model under serve,
    # which load.vlm._mirror_media_token_ids fills.
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._apc_media_token_ids",
          "apc_media.install_causal_serve_media (CausalMedia for qwen3.5)",
          wrappers=("gmlx.cache.apc_media",)),
    # --- server engine ---
    Claim("gmlx._vendor.mlx_vlm.server.generation", "load_model_resources",
          "bridge_vlm (GGUF model resource loader)",
          wrappers=("gmlx.serve.bridge_vlm",)),
    Claim("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._make_sampler",
          "patches.sampling.install_fast_sampler",
          wrappers=("gmlx.serve.patches.sampling",)),
    Claim("gmlx._vendor.mlx_vlm.server.generation", "_check_configured_context_budget",
          "patches.api_contract.install_context_overflow_wording (replaced; "
          "same check, overflow text clients match)",
          wrappers=("gmlx.serve.patches.api_contract",)),
    Claim("gmlx._vendor.mlx_vlm.server.generation", "get_configured_context_limit",
          "patches.api_contract.install_context_overflow_wording (wrapped; a "
          "per-model max_kv_size wins over the process-wide limit)",
          wrappers=("gmlx.serve.patches.api_contract",)),
    # Callers look these up at call time, and the server package re-exports
    # them, so the installer rebinds both bindings.
    Claim("gmlx._vendor.mlx_vlm.server", "_check_configured_context_budget",
          "patches.api_contract.install_context_overflow_wording (the "
          "package binding, replaced as in the generation module)",
          wrappers=("gmlx.serve.patches.api_contract",)),
    Claim("gmlx._vendor.mlx_vlm.server", "get_configured_context_limit",
          "patches.api_contract.install_context_overflow_wording (the "
          "package binding, wrapped as in the generation module)",
          wrappers=("gmlx.serve.patches.api_contract",)),
    Claim("gmlx._vendor.mlx_vlm.server.anthropic", "get_cached_model",
          "patches.api_contract.install_context_overflow_wording (wrapped: "
          "the lookup's HTTP errors, such as the unknown-model 404, pass "
          "the Messages catch-all and reach the app's handler)",
          wrappers=("gmlx.serve.patches.api_contract",)),
    Claim("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator.generate",
          "mem_preflight.install_memory_preflight; "
          "patches.mtp_thinking (thinking budget deferral); "
          "patches.spec_processors (moves an MTP model's logits_processors "
          "aside past the speculative refusal, and back on return); "
          "media_salt.install_media_salt (a request scope: the salt is "
          "hash_image_payload(pixel_values=raw['pixel_values']) after "
          "_preprocess_request, then semantic_extra_hash, before the "
          "request queues)",
          wrappers=("gmlx.cache.media_salt", "gmlx.serve.mem_preflight",
                    "gmlx.serve.patches.mtp_thinking",
                    "gmlx.serve.patches.spec_processors")),
    Claim("gmlx._vendor.mlx_vlm.server.generation",
          "ResponseGenerator._make_thinking_budget_criteria",
          "chat_behavior.install_thinking_budget_fix (rebind); "
          "patches.mtp_thinking (restore and hook attach; runtime "
          "chain mtp, then tbfix)",
          wrappers=("gmlx.serve.patches.chat_behavior",
                    "gmlx.serve.patches.mtp_thinking",
                    "gmlx.serve.patches.sampling")),
    Claim("gmlx._vendor.mlx_vlm.server.generation",
          "ResponseGenerator._make_logits_processors",
          "seed_rows.install_per_request_seed; "
          "lora_rows.install_row_channel (engine-thread handoff to the "
          "insert that takes the result as an argument); "
          "patches.spec_processors (puts the deferred processors back "
          "before the row's processors are built)",
          wrappers=("gmlx.lora_rows", "gmlx.serve.patches.spec_processors",
                    "gmlx.serve.seed_rows")),
    Claim("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._step",
          "patches.row_failed (permanently failed rows delivered "
          "to their response queues on the engine thread)",
          wrappers=("gmlx.serve.live_requests",
                    "gmlx.serve.patches.row_failed")),
    Claim("gmlx._vendor.mlx_vlm.server.generation",
          "GenerationArguments.to_template_kwargs",
          "patches.chat_behavior (chat_template_kwargs transform)",
          wrappers=("gmlx.serve.patches.chat_behavior",)),
    Claim("gmlx._vendor.mlx_vlm.server.generation",
          "ResponseGenerator._preprocess_request",
          "patches.apc.install_retire_render_capture (ids hop and "
          "tokenize path for the next-turn retirement key); "
          "media_salt.install_media_salt (hands the request's inputs to "
          "its image hash)",
          wrappers=("gmlx.cache.media_salt", "gmlx.serve.patches.apc")),
    Claim("gmlx._vendor.mlx_vlm.server.generation", "prepare_inputs",
          "single_bos_fix (module attr: a prompt that opens with BOS is "
          "encoded without special tokens); media_sinks (the request's "
          "media decode as one pass under the pixel caps)",
          wrappers=("gmlx.serve.media_sinks", "gmlx.upstream.single_bos_fix")),
    # --- routes and their render bindings. The route modules copy their
    # bindings from _protocol_deps members at route registration. ---
    Claim("gmlx._vendor.mlx_vlm.server.openai", "apply_chat_template",
          "patches.apc.install_retire_render_capture (render-context "
          "memo, module attr); render.install_faithful_history (inner "
          "key-merge wrap)",
          wrappers=("gmlx.serve.patches.apc",
                    "gmlx.serve.patches.chat_behavior",
                    "gmlx.serve.patches.render")),
    # estimate._estimate_bound follows the chat route's merge.
    Claim("gmlx._vendor.mlx_vlm.server.openai", "_normalize_instruction_messages",
          "patches.chat_behavior.install_role_normalization (wrapped: a call "
          "with the messages alone, the chat completions one, keeps "
          "developer messages in place)",
          wrappers=("gmlx.serve.patches.chat_behavior",)),
    Claim("gmlx._vendor.mlx_vlm.server.openai", "chat_completions_endpoint",
          "render.install_chat_image_placement (route wrap: a user "
          "message's content with an image part passes through the "
          "route's text extraction call, which keeps it); the route stack "
          "of patches and the assistant aliases",
          how="route", paths=("/chat/completions", "/v1/chat/completions")),
    Claim("gmlx._vendor.mlx_vlm.server.openai", "extract_text_from_content",
          "render.install_chat_image_placement (wrapped: marked content "
          "returns its text parts and one image marker per image, which "
          "apply_chat_template counts per message)",
          wrappers=("gmlx.serve.patches.render",)),
    Claim("gmlx._vendor.mlx_vlm.server.anthropic", "_preflight_stream_context_budget",
          "patches.api_contract.install_context_overflow_wording (wrapped: the "
          "overflow 400 is recorded for the endpoint wrapper)",
          wrappers=("gmlx.serve.patches.api_contract",)),
    Claim("gmlx._vendor.mlx_vlm.server.anthropic", "anthropic_messages_endpoint",
          "patches.api_contract.install_context_overflow_wording (the streaming "
          "preflight's 400 reaches the catch-all 500; the wrapper answers "
          "400); the route stack of patches and the assistant "
          "aliases",
          how="route", paths=("/messages", "/v1/messages")),
    Claim("gmlx._vendor.mlx_vlm.server.anthropic", "apply_chat_template",
          "patches.render, patches.apc and patches.chat_behavior (faithful "
          "history, retire capture, thinking seed and developer-role "
          "normalization wrap every render binding _render_target_modules "
          "captures)",
          wrappers=("gmlx.serve.patches.apc",
                    "gmlx.serve.patches.chat_behavior",
                    "gmlx.serve.patches.render")),
    # The Messages route's conversion step: gmlx claims it and calls the
    # stock helpers as the stock body does.
    Claim("gmlx._vendor.mlx_vlm.server.anthropic", "_anthropic_messages_to_internal",
          "messages_route.install_messages_route (replaced; the stock "
          "body, with a user message that holds an image as its text runs "
          "and image markers in block order, and its tool results ahead "
          "of its text)",
          wrappers=("gmlx.serve.patches.messages_route",)),
    Claim("gmlx._vendor.mlx_vlm.server.openai", "responses_endpoint",
          "render.install_faithful_history (renders the messages and "
          "image list _response_items_to_chat builds, in the order it "
          "builds them); the route stack of patches and the "
          "assistant aliases",
          how="route", paths=("/responses", "/v1/responses")),
    Claim("gmlx._vendor.mlx_vlm.server.app", "_protocol_deps.apply_chat_template",
          "patches.render, patches.apc and patches.chat_behavior (gmlx wraps "
          "the member beside both route-module copies)",
          wrappers=("gmlx.serve.patches.apc",
                    "gmlx.serve.patches.chat_behavior",
                    "gmlx.serve.patches.render")),
    Claim("gmlx._vendor.mlx_vlm.server.app", "_protocol_deps.build_gen_args",
          "patches._common._install_gen_args_transform (each "
          "transform wraps the member the routes call)",
          wrappers=("gmlx.serve.patches._common",)),
    Claim("gmlx._vendor.mlx_vlm.prompt_utils", "apply_chat_template",
          "patches.render.install_faithful_history "
          "(return_messages contract: one rebuilt message per readable "
          "input item; tool passthrough branch); render._images_in_place "
          "(each user message gets as many images as it holds markers, "
          "and the rest go to the last user message); render._TOOL_IMAGES "
          "(a tool message's image markers count toward the rest, so the "
          "wrapper leaves them out of num_images); "
          "render._install_nested_in_place (wrapped: the same, for the "
          "rebuild a processor runs inside a render that keeps them)",
          wrappers=("gmlx.serve.patches.render",)),
    Claim("gmlx._vendor.mlx_vlm.prompt_utils", "get_chat_template",
          "patches.render.install_faithful_history (owned render "
          "tail mirrors the stock function's); "
          "reasoning.install_template_default_thinking (absent "
          "enable_thinking seeded as Jinja Undefined so the 0.6.15 False "
          "injection stands down); system_content_fix (a text-only "
          "system or developer message reaches the template as a "
          "string); render._install_nested_in_place (wrapped: within an "
          "in-place render, the images of a nested rebuild go back in "
          "place)",
          wrappers=("gmlx.serve.patches.render", "gmlx.tui.reasoning",
                    "gmlx.upstream.system_content_fix")),
    Claim("gmlx._vendor.mlx_vlm.server", "_infer_tool_parser_from_processor",
          "patches.apc.install_retire_render_capture (wrapped: "
          "records the request's tool_parser override for the retirement "
          "key; the "
          "_protocol_deps lambda looks the package attribute up per call)",
          wrappers=("gmlx.serve.patches.apc",)),
    Claim("gmlx._vendor.mlx_vlm.server.app", "_split_thinking_text",
          "patches.chat_behavior.install_stream_thinking_seed "
          "(non-stream truncated-thinking classification, module attr "
          "looked up per call by app._split_thinking)",
          wrappers=("gmlx.serve.patches.chat_behavior",)),
    Claim("gmlx._vendor.mlx_vlm.server.runtime", "runtime",
          "residency._RuntimeProxy wrap",
          wrappers=("gmlx.serve.residency",)),
    Claim("gmlx._vendor.mlx_vlm.server.app", "app",
          "patches route surgery; serve.server uvicorn root",
          how="app"),
    Claim("gmlx._vendor.mlx_vlm.speculative.drafters", "load_drafter",
          "bridge_vlm (in-memory MTP drafter injection); media_sinks (loads "
          "only the drafter the server's own build chose)",
          wrappers=("gmlx.serve.bridge_vlm", "gmlx.serve.media_sinks")),
    # --- media checks: media_sinks puts them in every function that opens
    # request media, loads a model or creates a folder, and in each module
    # that holds its own reference to one. ---
    Claim("gmlx._vendor.mlx_vlm.utils", "load_image",
          "media_sinks (data: URI, media folder file or checked URL, under "
          "the byte and pixel caps)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.utils", "load_audio",
          "media_sinks (data: URI, media folder file or checked URL, under "
          "the byte cap)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.utils", "load_video",
          "media_sinks (data: URI, streamed media folder file or checked "
          "URL)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.utils", "prepare_inputs",
          "media_sinks (one media decode pass under the total pixel cap)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm", "prepare_inputs",
          "media_sinks (the package binding, one media decode pass)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "prepare_inputs",
          "media_sinks (one media decode pass)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.server.embeddings", "prepare_inputs",
          "media_sinks (one media decode pass)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.server.reranking", "prepare_inputs",
          "media_sinks (one media decode pass)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.server.embeddings", "load_image",
          "media_sinks (the checked image loader)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.server.reranking", "load_image",
          "media_sinks (the checked image loader)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.server.reranking", "load_video",
          "media_sinks (the checked video loader)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.generate.video", "resolve_video_inputs",
          "media_sinks (the video frames join the request's decode pass)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.server.openai", "resolve_video_inputs",
          "media_sinks (the video frames join the request's decode pass)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.server.audio", "audio_read",
          "media_sinks (the checked mlx_audio reader the transcription "
          "module copied at import)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.utils", "load",
          "media_sinks (loads only a configured model path)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.server.generation", "load",
          "media_sinks (loads only a configured model path)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.apc", "DiskBlockStore.__init__",
          "media_sinks (stores only under the APC disk path the server's "
          "own build chose)",
          wrappers=("gmlx.serve.media_sinks",)),
    Claim("gmlx._vendor.mlx_vlm.server.openai", "generate_image",
          "patches.media_gate (refuses an output path, a model to load by "
          "path, a size or output_resolution over the pixel cap, and a "
          "reference image that is not inline data)",
          wrappers=("gmlx.serve.patches.media_gate",)),
    Claim("gmlx._vendor.mlx_vlm.server.openai", "edit_image",
          "patches.media_gate (the same checks, on the edit's reference "
          "images too)",
          wrappers=("gmlx.serve.patches.media_gate",)),
    Claim("gmlx._vendor.mlx_vlm.utils", "get_model_path",
          "patches.routes (HF download gate)",
          wrappers=("gmlx.serve.patches.routes",)),
    Claim("gmlx._vendor.mlx_vlm.utils", "StoppingCriteria.__call__",
          "patches.chat_behavior.install_ignore_eos (ignore-EOS)",
          wrappers=("gmlx.serve.patches.chat_behavior",)),
    Claim("gmlx._vendor.mlx_vlm.utils", "process_inputs",
          "single_bos_fix (module attr: the flag reaches only a processor "
          "that names it; inside a generation call that drops the "
          "duplicate BOS, forwarded to a known type through **kwargs)",
          wrappers=("gmlx.upstream.single_bos_fix",)),
    Claim("gmlx._vendor.mlx_vlm.generate.dispatch", "prepare_inputs",
          "single_bos_fix (module attr, generate path); media_sinks (one "
          "media decode pass)",
          wrappers=("gmlx.serve.media_sinks", "gmlx.upstream.single_bos_fix")),
    Claim("gmlx._vendor.mlx_vlm.server.anthropic", "anthropic_count_tokens_endpoint",
          "single_bos_fix (the fallback count calls the prepare_inputs "
          "binding with prompts= and no add_special_tokens, after "
          "get_cached_model filled the model cache); the assistant "
          "aliases' route guard",
          how="route",
          paths=("/messages/count_tokens", "/v1/messages/count_tokens")),
    Claim("gmlx._vendor.mlx_vlm.server.anthropic", "prepare_inputs",
          "single_bos_fix (module attr: the count adds special tokens as "
          "generation does, and drops a duplicate BOS); media_sinks (one "
          "media decode pass)",
          wrappers=("gmlx.serve.media_sinks", "gmlx.upstream.single_bos_fix")),
    Claim("gmlx._vendor.mlx_vlm.server.openai", "responses_input_tokens_endpoint",
          "single_bos_fix (the fallback count calls the prepare_inputs "
          "binding with prompts= and no add_special_tokens, after "
          "get_cached_model filled the model cache); the assistant "
          "aliases' route guard",
          how="route",
          paths=("/responses/input_tokens", "/v1/responses/input_tokens")),
    Claim("gmlx._vendor.mlx_vlm.server.openai", "prepare_inputs",
          "single_bos_fix (module attr: the count adds special tokens as "
          "generation does, and drops a duplicate BOS); media_sinks (one "
          "media decode pass)",
          wrappers=("gmlx.serve.media_sinks", "gmlx.upstream.single_bos_fix")),
    # single_bos_fix's count reads the processor and config this stores
    # in the runtime's model cache.
    Claim("gmlx._vendor.mlx_vlm.server.app", "get_cached_model",
          "residency pool resolver",
          wrappers=("gmlx.serve.residency",)),
    # --- APC internals (lone-harvest patch, gmlx manager subclass,
    # apc_pooling) ---
    Claim("gmlx._vendor.mlx_vlm.apc", "harvest_blocks_from_batch_cache",
          "patches.apc.install_apc_lone_harvest",
          wrappers=("gmlx.serve.patches.apc",)),
    Claim("gmlx._vendor.mlx_vlm.apc", "DiskBlockStore._snapshot_exact_cache_entry",
          "apc_pooling, apc_qsa and kvarn_apc disk arms (each wraps the "
          "method it found at install time)",
          wrappers=("gmlx.cache.apc_pooling", "gmlx.cache.apc_qsa",
                    "gmlx.cache.kvarn_apc")),
    Claim("gmlx._vendor.mlx_vlm.apc", "DiskBlockStore._load_exact_cache_entry",
          "apc_pooling, apc_qsa and kvarn_apc disk arms (each wraps the "
          "method it found at install time)",
          wrappers=("gmlx.cache.apc_pooling", "gmlx.cache.apc_qsa",
                    "gmlx.cache.kvarn_apc")),
    Claim("gmlx._vendor.mlx_vlm.apc", "APCManager.store_kv_blocks",
          "apc_manager.GmlxAPCManager.store_kv_blocks (mirror: re-sync "
          "the copy on change; each block hashes under the salt at its "
          "end)",
          how="subclass", by=_APC_MANAGER),
    Claim("gmlx._vendor.mlx_vlm.apc", "APCManager._make_room",
          "apc_manager.GmlxAPCManager._prefill_reserve_bytes (reads 0 "
          "after autosize; a renamed reserve field would escape the "
          "property), GmlxAPCManager._memory_headroom (the "
          "memory_evictions count and the evict, clear, read-again order) "
          "and GmlxAPCManager._make_room (scopes the headroom marks to "
          "the call)",
          how="subclass", by=_APC_MANAGER),
    Claim("gmlx._vendor.mlx_vlm.apc", "APCManager._memory_headroom",
          "apc_manager.GmlxAPCManager._memory_headroom (drops a cached "
          "free-RAM reading after an eviction or a cache clear inside "
          "_make_room)",
          how="subclass", by=_APC_MANAGER),
    Claim("gmlx._vendor.mlx_vlm.apc", "APCManager.prepare_prefill",
          "apc_manager.GmlxAPCManager._prefill_reserve_bytes (disk flush "
          "stays stock; the reserve it sets reads 0 after autosize)",
          how="subclass", by=_APC_MANAGER),
    Claim("gmlx._vendor.mlx_vlm.apc", "_free_ram_bytes",
          "apc_manager._install_fork_free_free_ram (disk-restore gates "
          "and the _make_room headroom read, rebound to the in-process "
          "mach read, else /usr/bin/vm_stat behind a cache; stock forks "
          "the vm_stat on PATH)",
          install="gmlx.cache.apc_manager:_install_fork_free_free_ram",
          wrappers=("gmlx.cache.apc_manager",)),
    Claim("gmlx._vendor.mlx_vlm.apc_adapters", "clone_cache_entry",
          "apc_pooling and kvarn_apc (wrapped: pooling and kvarn clone "
          "arms; CacheList members recurse through the module global)",
          wrappers=("gmlx.cache.apc_pooling", "gmlx.cache.kvarn_apc")),
    Claim("gmlx._vendor.mlx_vlm.apc_adapters", "merge_cache_entries",
          "apc_pooling and kvarn_apc (wrapped: pooling and kvarn merge "
          "arms)",
          wrappers=("gmlx.cache.apc_pooling", "gmlx.cache.kvarn_apc")),
    Claim("gmlx._vendor.mlx_vlm.apc_adapters", "build_prefix_cache_plan",
          "kvarn_apc plan wrap (a stamped kvarn model pages nothing, so "
          "its strategy is checkpoint)",
          wrappers=("gmlx.cache.kvarn_apc",)),
    Claim("gmlx._vendor.mlx_vlm.apc_coordinator", "build_prefix_cache_plan",
          "kvarn_apc plan wrap (the coordinator's own binding, rebound "
          "beside the adapter's)",
          wrappers=("gmlx.cache.kvarn_apc",)),
    Claim("gmlx._vendor.mlx_vlm.apc", "_read_safetensors_tensor",
          "apc_pooling (disk-tier zero-width spill; wrapped reader)",
          wrappers=("gmlx.cache.apc_pooling",)),
    # load.vlm._mirror_media_token_ids puts gemma-4 container ids onto a
    # bidirectional text config for it.
    Claim("gmlx._vendor.mlx_vlm.apc", "multimodal_token_ids_from_config",
          "apc_media.install_media_token_ids (config media_token_ids fold "
          "for expanded image blocks)",
          wrappers=("gmlx.cache.apc_media",)),
    Claim("gmlx._vendor.mlx_vlm.apc", "media_safe_prefix_min",
          "apc_media.install_media_token_ids (zero floor for "
          "WholeMediaBlocks)",
          wrappers=("gmlx.cache.apc_media",)),
    Claim("gmlx._vendor.mlx_vlm.apc", "prefix_leaves_text_only_suffix",
          "apc_media.install_media_token_ids (a WholeMediaBlocks prefix "
          "ends outside every run)",
          wrappers=("gmlx.cache.apc_media",)),
    Claim("gmlx._vendor.mlx_vlm.apc", "adjust_prefix_to_text_suffix_boundary",
          "apc_media.install_media_token_ids (a WholeMediaBlocks column "
          "inside a run moves to its end)",
          wrappers=("gmlx.cache.apc_media",)),
    Claim("gmlx._vendor.mlx_vlm.apc", "hash_image_payload",
          "media_salt.install_media_salt (a request's images hash once, "
          "per image, into spans; a list of images of different sizes "
          "hashes by content, not as 0)",
          wrappers=("gmlx.cache.media_salt",)),
    Claim("gmlx._vendor.mlx_vlm.apc", "semantic_extra_hash",
          "media_salt.install_media_salt (spans and the text salt make "
          "the MediaSalt)",
          wrappers=("gmlx.cache.media_salt",)),
    Claim("gmlx._vendor.mlx_vlm.apc", "APCManager.lookup_exact_cache",
          "apc_manager.GmlxAPCManager.lookup_exact_cache (one call per "
          "MediaSalt window: min exclusive, a positive max inclusive)",
          how="subclass", by=_APC_MANAGER),
    Claim("gmlx._vendor.mlx_vlm.apc", "APCManager.store_exact_cache",
          "apc_manager.GmlxAPCManager.store_exact_cache (keys on the salt "
          "at the entry length)",
          how="subclass", by=_APC_MANAGER),
    Claim("gmlx._vendor.mlx_vlm.apc", "APCManager.lookup_prefix",
          "apc_manager.GmlxAPCManager.lookup_prefix (mirror for a "
          "MediaSalt: the stock walk over block_chain, one hit or miss; "
          "re-sync on change); fresh_gate._covered_len (the same walk)",
          how="subclass", by=_APC_MANAGER),
    Claim("gmlx._vendor.mlx_vlm.apc", "APCManager.lookup_prefix_disk_cache",
          "apc_manager.GmlxAPCManager.lookup_prefix_disk_cache (mirror "
          "for a MediaSalt: the stock restore over block_chain, each "
          "shard block's salt checked at its end; re-sync on change)",
          how="subclass", by=_APC_MANAGER),
    # residency and the spec.engine L1 probe read it.
    Claim("gmlx._vendor.mlx_vlm.apc", "model_apc_mode",
          "kvarn_apc wraps it (a stamped model answers from the plan, as "
          "the coordinator does)",
          wrappers=("gmlx.cache.kvarn_apc",)),
    # cache.refusals relies on its hybrid and TurboQuant arms skipping
    # preserve_auxiliary_kv_state.
    Claim("gmlx._vendor.mlx_vlm.generate", "maybe_quantize_kv_cache",
          "apc_pooling (rotating-safe KV-quant replacement)",
          wrappers=("gmlx.cache.apc_pooling",)),
    # --- switch layers. gmlx builds its fused and block-routed classes on
    # both packages' classes at load time. ---
    Claim("gmlx._vendor.mlx_vlm.models.switch_layers", "SwitchLinear.__call__",
          "modules.switch_layer_types (dual-origin leaf swap)",
          how="subclass"),
    Claim("gmlx._vendor.mlx_vlm.models.switch_layers", "SwitchGLU.__call__",
          "modules.switch_layer_types (dual-origin fused GLU); "
          "modules.install_switch_glu_block_routing (a block of 2 to "
          "DECODE_BLOCK_SIZE positions loops per token when the exact "
          "kernels decline)",
          how="subclass"),
    Claim("gmlx._vendor.mlx_vlm.models.switch_layers", "SwitchGLU._combine",
          "modules._BlockRoutedSwitchGLU (block call, then the stock "
          "weights, shared and residual combine)",
          how="subclass"),
    Claim("gmlx._vendor.mlx_lm.models.switch_layers", "SwitchLinear.__call__",
          "modules leaf swap and fused subclass signature contract",
          how="subclass"),
    Claim("gmlx._vendor.mlx_lm.models.switch_layers", "SwitchGLU.__call__",
          "modules.install_fused_moe_glu; loader.install_expert_streaming",
          how="subclass"),
    # --- wired limit. The stream and CPU device modes replace it with a
    # quiet context, and systemone.engine.engine_scope enters it. ---
    Claim("gmlx._vendor.mlx_lm.generate", "wired_limit",
          "stream.wired_limit._install_wired_limit_warn_once (replaced: "
          "the near-budget warning prints once per process)",
          install="gmlx.stream.wired_limit:_install_wired_limit_warn_once",
          wrappers=("gmlx.stream.wired_limit",)),
    # --- engine jobs ---
    Claim("gmlx._vendor.mlx_vlm.server.generation",
          "ResponseGenerator._generate_diffusion",
          "engine_jobs.install_engine_jobs (wrapped: a queued job runs in "
          "place of a generation)",
          wrappers=("gmlx.serve.engine_jobs",)),
    Claim("gmlx._vendor.mlx_vlm.server.generation",
          "ResponseGenerator._collect_pending_requests",
          "engine_jobs.install_engine_jobs (wrapped: step jobs run "
          "between batch steps; active=True never blocks, capacity 0 "
          "reads nothing)",
          wrappers=("gmlx.serve.engine_jobs",)),
    # --- mlx-lm ---
    Claim("gmlx._vendor.mlx_lm.server", "ModelProvider._load",
          "bridge_lm (GGUF ModelProvider)",
          install="gmlx.serve.bridge_lm:install_gguf_bridge",
          wrappers=("gmlx.serve.bridge_lm",)),
    Claim("gmlx._vendor.mlx_lm.generate", "maybe_quantize_kv_cache",
          "apc_pooling",
          wrappers=("gmlx.cache.apc_pooling",)),
    Claim("gmlx._vendor.mlx_lm.models.hunyuan", "MoeBlock",
          "loader._patch_hunyuan_norm_topk",
          how="subclass"),
    Claim("gmlx._vendor.mlx_lm.models.deepseek_v32", "MoEGate.__call__",
          "dsv32_patches._patch_dsv32_moe_gate_fp32 (fp32 router "
          "selection for the flagged instances)",
          install="gmlx.upstream.dsv32_patches:_patch_dsv32_moe_gate_fp32",
          wrappers=("gmlx.upstream.dsv32_patches",)),
    # kimi_k3.KimiK3DeltaAttention calls it for the CPU and fallback scan.
    Claim("gmlx._vendor.mlx_lm.models.gated_delta", "gated_delta_ops",
          "gdn_patches._patch_gated_delta_tiled_v (the GGUF tiled K to V "
          "head mapping)",
          install="gmlx.upstream.gdn_patches:_patch_gated_delta_tiled_v",
          wrappers=("gmlx.upstream.gdn_patches",)),
    # --- phased command buffer steps around mlx-lm's generation loops ---
    Claim("gmlx._vendor.mlx_lm.generate", "generate_step",
          "cb_phase.install_cb_phase_steps (phased steps around the loop)",
          wrappers=("gmlx.serve.cb_phase",)),
    Claim("gmlx._vendor.mlx_lm.generate", "speculative_generate_step",
          "cb_phase.install_cb_phase_steps (phased steps around the loop)",
          wrappers=("gmlx.serve.cb_phase",)),
    # --- speculative batches: the checkpoint tier, admission and the owned
    # speculative caches ---
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "GenerationBatch.extend",
          "spec.ckpt (the plain decode checkpoint tier follows the rows an "
          "extend adds)",
          wrappers=("gmlx.spec.ckpt",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "SpeculativeGenerationBatch.extend",
          "admission.install_continuous_batch_admission (replaced: an "
          "extend buffers the batch for promotion, where the copy's raises)",
          wrappers=("gmlx.spec.admission",)),
    Claim("gmlx._vendor.mlx_vlm.generate.ar", "make_speculative_prompt_cache",
          "spec.engine.install_owned_b1_spec_cache and "
          "kv_quant.install_spec_kv_quant (the ar module's own binding, "
          "rebound with the speculative.utils one)",
          wrappers=("gmlx.spec.engine", "gmlx.spec.kv_quant")),
    # --- KV quantization: the common module's binding ---
    Claim("gmlx._vendor.mlx_vlm.generate.common", "maybe_quantize_kv_cache",
          "apc_pooling.install_safe_kv_quantization (the common module's "
          "binding, replaced with the package one)",
          wrappers=("gmlx.cache.apc_pooling",)),
    # --- residency pool: the package and app bindings ---
    Claim("gmlx._vendor.mlx_vlm.server", "get_cached_model",
          "residency.install_gguf_residency_pool (the package binding that "
          "the protocol handlers resolve per request)",
          wrappers=("gmlx.serve.residency",)),
    Claim("gmlx._vendor.mlx_vlm.server", "unload_model_sync",
          "residency.install_gguf_residency_pool (the package binding)",
          wrappers=("gmlx.serve.residency",)),
    Claim("gmlx._vendor.mlx_vlm.server.app", "unload_model_sync",
          "residency.install_gguf_residency_pool (the app binding)",
          wrappers=("gmlx.serve.residency",)),
    # --- chat behavior: stream thinking seed, stop sequences, stream
    # timings ---
    Claim("gmlx._vendor.mlx_vlm.server.responses_state",
          "ThinkingStreamState.__init__",
          "patches.chat_behavior.install_stream_thinking_seed (the stream "
          "starts in reasoning mode only when the rendered prompt opens a "
          "thinking block)",
          wrappers=("gmlx.serve.patches.chat_behavior",)),
    Claim("gmlx._vendor.mlx_vlm.server.responses_state",
          "ThinkingStreamState.feed",
          "patches.chat_behavior.install_stream_thinking_seed",
          wrappers=("gmlx.serve.patches.chat_behavior",)),
    Claim("gmlx._vendor.mlx_vlm.server.responses_state", "_split_thinking",
          "patches.chat_behavior.install_stream_thinking_seed (replaced: "
          "the split keeps the content's whitespace)",
          wrappers=("gmlx.serve.patches.chat_behavior",)),
    Claim("gmlx._vendor.mlx_vlm.server.anthropic", "_apply_stop_sequences",
          "patches.chat_behavior.install_openai_stop_sequences (replaced)",
          wrappers=("gmlx.serve.patches.chat_behavior",)),
    Claim("gmlx._vendor.mlx_vlm.server.generation",
          "GenerationMetrics.record_chunk",
          "patches.chat_behavior.install_stream_timings (wrapped: the "
          "exact output token count a timings chunk carries)",
          wrappers=("gmlx.serve.patches.chat_behavior",)),
    # --- generation arguments: the per-handler bindings ---
    Claim("gmlx._vendor.mlx_vlm.server.app", "_build_gen_args",
          "patches._common._install_gen_args_transform (wrapped: each "
          "installed transform applies to the generation arguments)",
          wrappers=("gmlx.serve.patches._common",)),
    Claim("gmlx._vendor.mlx_vlm.server.openai", "_build_gen_args",
          "patches._common._install_gen_args_transform (the openai "
          "binding of the app's wrapper)",
          wrappers=("gmlx.serve.patches._common",)),
    Claim("gmlx._vendor.mlx_vlm.server.anthropic", "_build_gen_args",
          "patches._common._install_gen_args_transform (the anthropic "
          "binding of the app's wrapper)",
          wrappers=("gmlx.serve.patches._common",)),
    # --- runtime snapshot ---
    Claim("gmlx._vendor.mlx_vlm.server.app", "_server_runtime_snapshot",
          "patches.routes.install_runtime_snapshot_enrichment",
          wrappers=("gmlx.serve.patches.routes",)),
    # --- memory preflight: the context budget check ---
    Claim("gmlx._vendor.mlx_vlm.server.generation",
          "ResponseGenerator.validate_context_budget",
          "mem_preflight.install_memory_preflight (wrapped: the media "
          "check and the prompt memory preflight run with the budget "
          "check)",
          wrappers=("gmlx.serve.mem_preflight",)),
)


def _key(claim: Claim) -> str:
    return f"{claim.module}:{claim.attr}"


def _resolve(module: str, attr: str):
    """('ok', obj), ('missing module', None) or ('missing attribute', None).
    A loaded module comes from ``sys.modules``: a package can shadow a
    submodule attribute of the same name."""
    try:
        obj = sys.modules.get(module) or importlib.import_module(module)
    except ImportError:
        return "missing module", None
    for part in attr.split(".") if attr else ():
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return "missing attribute", None
    return "ok", obj


def _resolve_spec(spec: str):
    module, _, attr = spec.partition(":")
    return _resolve(module, attr)


def _module_origin(name: str) -> str:
    if name.startswith("gmlx._vendor."):
        return "copy"
    if name == "gmlx" or name.startswith("gmlx."):
        return "gmlx"
    return "other"


def _code_origin(obj) -> str | None:
    """Where the code of ``obj`` was written, or None when it has none."""
    code = getattr(obj, "__code__", None)
    if code is None:
        return None
    path = os.path.abspath(code.co_filename)
    if path.startswith(_COPY_DIR + os.sep):
        return "copy"
    if path.startswith(_GMLX_DIR + os.sep):
        return "gmlx"
    return "other"


def _origin(obj) -> str:
    """Where ``obj`` was written: "gmlx", "copy" or "other". A wrapper that
    a decorator written elsewhere made (contextlib's, functools') is looked
    through to the function it wraps, and a partial to its function. An
    object with no code of its own goes by the module of its class, or of
    itself when it is a class."""

    def stop(f):
        return isinstance(f, functools.partial) or _code_origin(f) != "other"

    for _ in range(64):
        if isinstance(obj, functools.partial):
            obj = obj.func
            continue
        try:
            obj = inspect.unwrap(obj, stop=stop)
        except ValueError:  # a cycle of wrappers
            return "other"
        if not isinstance(obj, functools.partial):
            break
    origin = _code_origin(obj)
    if origin is None:
        cls = obj if isinstance(obj, type) else type(obj)
        return _module_origin(getattr(cls, "__module__", "") or "")
    return origin


def _model_params(fn) -> list[str] | None:
    """The names of ``fn``'s required parameters, or None when one of them
    is not the loaded model."""
    names = [p.name for p in inspect.signature(fn).parameters.values()
             if p.default is p.empty
             and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return names if all(n == "model" for n in names) else None


def _row_problems(claim: Claim) -> list[str]:
    """What is wrong with the row itself, before anything is installed."""
    key, where = _key(claim), f"(claimed by {claim.used_by})"
    out = []
    if claim.how not in _HOWS:
        out.append(f"{key}: how is {claim.how!r}, not one of {_HOWS}")
    if (claim.how == "route") != bool(claim.paths):
        out.append(f"{key}: paths go with how='route' only")
    if claim.by and claim.how != "subclass":
        out.append(f"{key}: by goes with how='subclass' only")
    if (claim.how == "wrap") != bool(claim.wrappers):
        out.append(f"{key}: wrappers go with how='wrap', which needs them")
    if [m for m in claim.wrappers if _module_origin(m) != "gmlx"]:
        out.append(f"{key}: wrappers name gmlx modules outside the copy")
    if claim.install:
        if claim.how != "wrap":
            out.append(f"{key}: install goes with how='wrap' only")
        status, fn = _resolve_spec(claim.install)
        if status != "ok":
            out.append(f"{key}: installer {claim.install} {status}; a fold "
                       f"that deleted it deletes this row too {where}")
        elif not callable(fn) or _origin(fn) != "gmlx":
            out.append(f"{key}: installer {claim.install} is not a gmlx "
                       f"function {where}")
        elif _model_params(fn) is None:
            out.append(f"{key}: installer {claim.install} takes arguments "
                       f"other than the model {where}")
    return out


def _subclass_problems(claim: Claim, obj) -> list[str]:
    if not claim.by:
        return []
    key, where = _key(claim), f"(claimed by {claim.used_by})"
    status, cls = _resolve_spec(claim.by)
    if status != "ok" or not isinstance(cls, type):
        return [f"{key}: subclass {claim.by} is not a class ({status}) "
                f"{where}"]
    head, _, _ = claim.attr.rpartition(".")
    owner = _resolve(claim.module, head)[1] if head else obj
    if not isinstance(owner, type) or not issubclass(cls, owner):
        return [f"{key}: {claim.by} does not derive from the copy's class "
                f"{where}"]
    return []


def _app():
    return importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app


def _installed_problems(claim: Claim, obj) -> list[str]:
    key, where = _key(claim), f"(claimed by {claim.used_by})"
    by = claim.install or "the serve startup"
    if claim.how == "wrap":
        if _origin(obj) != "gmlx":
            return [f"{key}: still the copy's after install; {by} did not "
                    f"claim it {where}"]
        have = chain_modules(obj)
        return [f"{key}: no wrapper from {m} in its chain after install; "
                f"{by} did not claim it {where}"
                for m in claim.wrappers if m not in have]
    from gmlx.serve.patches._common import _find_route, _iter_route_lists

    if claim.how == "route":
        out = []
        for path in claim.paths:
            route = _find_route(_app(), path, "POST")
            if route is None:
                out.append(f"{key}: no POST {path} route {where}")
            elif _origin(route.endpoint) != "gmlx":
                out.append(f"{key}: POST {path} still serves the copy's "
                           f"endpoint after install {where}")
        return out
    if claim.how == "app":
        if not any(_origin(getattr(r, "endpoint", None)) == "gmlx"
                   for routes in _iter_route_lists(obj) for r in routes):
            return [f"{key}: serves no gmlx endpoint after install {where}"]
    return []


def check_claims(installed: bool = False) -> list[str]:
    """Problems with the table against the copy; [] when every row holds.
    ``installed``: the installers ran in this process, so each ``wrap``,
    ``route`` and ``app`` claim must have taken."""
    problems, seen = [], set()
    for claim in CLAIMS:
        key = _key(claim)
        if key in seen:
            problems.append(f"{key}: two rows; merge them")
        seen.add(key)
        if not (claim.module in _COPY_PACKAGES
                or claim.module.startswith(_COPY_ROOTS)):
            problems.append(f"{key}: not a module of the copy")
            continue
        problems += _row_problems(claim)
        status, obj = _resolve(claim.module, claim.attr)
        if status != "ok":
            problems.append(f"{key}: {status} in the copy (claimed by "
                            f"{claim.used_by})")
            continue
        if claim.how == "subclass":
            problems += _subclass_problems(claim, obj)
        elif installed:
            problems += _installed_problems(claim, obj)
    return problems


def _code_module(path: str) -> str:
    """The module name of a gmlx source file."""
    rel = os.path.relpath(path, os.path.dirname(_GMLX_DIR))
    name = os.path.splitext(rel)[0].replace(os.sep, ".")
    return name.removesuffix(".__init__")


def _held(fn) -> list:
    """The functions an unlinked gmlx wrapper holds in its closure or its
    defaults: the one it wraps among them."""
    out = list(getattr(fn, "__defaults__", None) or ())
    out += list((getattr(fn, "__kwdefaults__", None) or {}).values())
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            out.append(cell.cell_contents)
        except ValueError:  # an empty cell
            pass
    return [f for f in out if callable(f) and _code_origin(
        getattr(f, "__func__", f)) in ("gmlx", "copy")]


def chain_modules(obj) -> set[str]:
    """The gmlx modules outside the copy whose code sits in the wrap chain
    over ``obj``: each wrapper, through ``__wrapped__`` and ``link``, down
    to the function at the bottom. A gmlx wrapper with neither link is
    followed through the functions it holds. An object with no code counts
    by the module of its class."""
    from gmlx.upstream.patching import _LINKS, _MAX_LAYERS

    out: set[str] = set()
    seen: set[int] = set()
    todo = [obj]
    while todo and len(seen) < _MAX_LAYERS:
        f = todo.pop()
        while isinstance(f, functools.partial):
            f = f.func
        if f is None or id(f) in seen:
            continue
        seen.add(id(f))
        links = [getattr(f, name, None) for name in _LINKS]
        todo += links
        code = (getattr(f, "__code__", None)
                or getattr(getattr(f, "__func__", None), "__code__", None))
        if code is None:
            cls = f if isinstance(f, type) else type(f)
            name = getattr(cls, "__module__", "") or ""
        else:
            path = os.path.abspath(code.co_filename)
            name = (_code_module(path)
                    if path.startswith(_GMLX_DIR + os.sep) else "")
            if _module_origin(name) == "gmlx" and not any(links):
                todo += _held(getattr(f, "__func__", f))
        if _module_origin(name) == "gmlx":
            out.add(name)
    return out


def _serve_startup() -> None:
    """The installs ``gmlx serve`` runs before it binds a port, for one
    GGUF model and one assistant alias. No model loads."""
    import gmlx.serve.bridge_vlm as bridge
    from gmlx.config import build_config
    from gmlx.serve import patches
    from gmlx.serve.cb_phase import install_cb_phase_flips, install_cb_phase_steps
    from gmlx.serve.residency import install_gguf_residency_pool
    from gmlx.serve.server import _import_serving

    _import_serving()
    cfg = build_config({
        "server": {"host": "127.0.0.1", "port": 8080,
                   "assistants": {"claims-helper": {"model": "claims"}}},
        "models": {"claims": {"path": "/claims.gguf"}}})
    bridge.register_resolved_models(cfg)
    bridge.install_gguf_server_bridge()
    install_gguf_residency_pool(budget_bytes=1 << 30)
    patches.install_server_patches(cfg, reload_fn=None)
    patches.install_ignore_eos()
    install_cb_phase_flips()
    install_cb_phase_steps()


def install_claims() -> None:
    """Run the serve startup, then every installer the table names. The
    names they rebind stay rebound for the rest of the process."""
    import mlx.nn as nn

    _serve_startup()
    for spec in dict.fromkeys(c.install for c in CLAIMS if c.install):
        status, fn = _resolve_spec(spec)
        params = _model_params(fn) if status == "ok" and callable(fn) else None
        if params is None:
            continue  # check_claims names it
        fn(*[nn.Module() for _ in params])


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Check the claims table against the copied mlx-vlm "
                    "and mlx-lm.")
    p.add_argument("--installed", action="store_true",
                   help="run the serve startup and every installer the "
                        "table names first, then check that each claim "
                        "took (rebinds names in this process)")
    args = p.parse_args(argv)
    if args.installed:
        install_claims()
    problems = check_claims(installed=args.installed)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"{len(CLAIMS)} claims OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
