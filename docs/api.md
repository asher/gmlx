# HTTP API

The endpoints `gmlx serve` exposes, how a request names a model, and which
request fields each protocol honors. For the YAML that configures the server
see [server-config.md](server-config.md).

## Addressing a model in a request

```jsonc
{"model": "qwen3.6-27b"}                                 // the model's configured profile (else family base)
{"model": "qwen3.6-27b@coding"}                          // a built-in intent, resolved per the model's family
{"model": "qwen3.6-27b@qwen-coder"}                      // inline user profile (beats model.profile)
{"model": "coder"}                                        // an alias (here: qwen3.6-27b@qwen-coder)
{"model": "qwen3.6-27b", "profile": "coding",            // profile field (needs extra_body in the OpenAI SDK)
 "temperature": 0.1}                                      //   + an explicit field that wins over the profile
```

The `@profile` suffix is split on the last `@`, and only treated as a profile
when it names a known one (a user profile or, with `family_defaults` on, a
built-in intent), so an hf-style `org/model@rev` or an id containing
`@`-like text stays intact. An unknown id returns 404 (listing available
ids); an unknown profile returns 400 (listing valid profiles, built-ins
included). An empty/missing `model` uses `server.defaults.model`, else the
sole model, else 400.

The same addressing works in the CLI: `gmlx run <id-or-path>@coding`,
`gmlx run <id> --profile coding`, and identically for `chat`. A bare-path
`run`/`chat` (no config) still gets its family's base defaults. An explicit
sampling flag always wins, and `--no-family-defaults` (or
`GMLX_NO_FAMILY_DEFAULTS=1`) opts a run out entirely.

---

## Operational endpoints

The standard generation and listing routes work: `/v1/chat/completions`,
`/v1/messages` (Anthropic), `/v1/responses`, `/v1/models`, `/health`. gmlx
adds a minimal classic `POST /v1/completions` text route (single string
prompt, `n=1`; see the table). Added to those: `POST /unload`,
`POST /v1/reload`, and `POST /v1/audio/transcriptions` /
`POST /v1/audio/speech` / `POST /v1/embeddings` when STT / TTS / embeddings
is enabled. These are the config-server overrides and additions:

| Endpoint | Behaviour |
|----------|-----------|
| `GET /v1/models`, `GET /models` | Configured/discovered ids + alias presets, each with `resident` / `pinned` / `speculative` / `vlm` / `profile` / `default` markers, plus `context_length` (the GGUF's trained context) and `max_context_at_width_1` (what the capacity table says fits at width 1; only for the model the table was derived from, else `null`). `gmlx launch pi` writes the smaller of the two as pi's `contextWindow`. Never the HF cache. |
| `GET /health` | Liveness only: `{"status": "healthy", "pid": N}`, no model/adapter paths, no context fields. The one route api-key auth exempts, so the liveness body deliberately leaks nothing. `?ready=1` adds a readiness verdict - deliberately a little more than liveness (a coarse busy/not-busy and a throughput-derived wait hint, which is all a keyless caller can learn; the numbers behind it stay on the authed metrics): 200 with `"ready": true`, or 503 with `"ready": false`, a one-word `reason` (`pressure` when the governor is orange/red, `queue` when requests are waiting for a slot, `busy` when every resident model's engine is at its decode width) and a `Retry-After` header from the same drain estimate a queue-cap 503 carries. Still keyless; see [Capacity and live-request metrics](#capacity-and-live-request-metrics). |
| `GET /v1/metrics`, `GET /metrics` | The stock runtime snapshot (under its `server` key) enriched with `resident_models[]`, the memory governor, the capacity table, `concurrency`, `queue`, and the live `requests[]` view; see [Capacity and live-request metrics](#capacity-and-live-request-metrics). Authed like every other endpoint; what `gmlx ps` reads. `?format=prometheus` (or `Accept: text/plain` / an OpenMetrics `Accept`) renders the same snapshot as Prometheus text (`gmlx_*` gauges and `*_total` counters; lists of models become `model`-labelled series, the capacity tables `width`/`depth`-labelled ones, `governor.band` a `band`-labelled indicator plus `gmlx_governor_band_level` 0-3). |
| `POST /v1/completions` | Classic OpenAI text completions, no chat template applied. Scope: a single string `prompt`, `n=1`; supports `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `stream` (SSE with a final `[DONE]`), plus `profile` and the other shared sampling extras. List / token-array prompts, `n > 1`, `echo`, `suffix`, and `best_of > 1` are rejected with a 400. |
| `POST /v1/messages/count_tokens` | (stock) Anthropic-style token counting: same body shape as `/v1/messages`, returns `{"input_tokens": N}` after applying the chat template. |
| `GET /v1/cache/stats` | Automatic Prefix Cache statistics, or `{"enabled": false}` when APC is off. On checkpoint-tier models (hybrid/SWA archs) the snapshot carries the `ckpt_*` counter family beside the stock fields; see [Checkpoint-tier counters](internals/prompt-cache.md#checkpoint-tier-counters). |
| `POST /v1/cache/reset` | Clears the Automatic Prefix Cache. With no body, every resident model's (the stock handler reached only the request context's manager); `{"model": "<id>"}` clears one resident model's. Returns `{"enabled", "status": "cleared" \| "no_cache", "models": [ids cleared]}`; 404 `unknown_model` / `not_resident` for a bad id. The disk tier's files are untouched, as before. |
| `POST /v1/images/generations`, `POST /v1/images/edits` | (stock mlx-vlm routes) not usable with GGUF text models - they require an MLX image-generation checkpoint, which gmlx does not serve; a request against a configured GGUF fails with an error. |
| `POST /unload` | `{"model": "<id>"}` evicts just that resident entry (also clearing any keep mark); an empty body clears the whole pool. An explicit unload outranks the preload's lifetime hold (that hold guards against implicit eviction, not the operator), so the preloaded primary unloads too; a model with in-flight streams still answers 409. A model unloaded this way and later reloaded by request has no lifetime hold - it is TTL/LRU-managed like any other until a `/v1/reload` with `preload` re-pins it. |
| `POST /v1/estimate` | Dry-run admission: the same body as `/v1/chat/completions`, answered with the numbers instead of a generation (`prompt_tokens`, `warm_tokens`, `need_bytes`, `fits_now`, `fits_drained`, `est_ttft_s`, ...). `"dry_run": true` on `/v1/chat/completions` returns the same estimate (that form goes through the queue cap; this route does not). See [Capacity and live-request metrics](#capacity-and-live-request-metrics). |
| `GET /v1/capacity/plan?width=W&depth=D` | The fan-out policy: can `W` streams run at `D` tokens each (`ok`, from the capacity table), and may they start now (`admit_now`, from the governor band, the waiting census and the free decode slots), with `reason`. |
| `POST /v1/keep` | `{"model": "<id>", "warm": true}` keeps a model resident through the idle-TTL reaper (still LRU-evictable; the keep tier above), and by default background-loads it so it is hot before the first request. `gmlx launch --model <id>` fires this once the server is up, and a `gmlx talk` / menu-bar voice session holds its model this way for the session's lifetime. `{"model": "<id>", "keep": false}` releases the hold without evicting; `/unload` releases and evicts. |
| `POST /v1/reload` | (config mode only) re-reads the config and re-registers models, keeping warm entries whose load signature is unchanged. In non-config modes it returns 200 with `{"status": "unsupported"}`. `SIGHUP` triggers the same reload; see [Reloading the config](server-config.md#reloading-the-config). |
| `POST /v1/audio/transcriptions` | (only with `server.stt:` / `--stt`) OpenAI-compatible speech-to-text; see below. |
| `POST /v1/audio/translations` | (only with `server.stt:` / `--stt`) OpenAI-compatible audio translation: any-language audio to English, same model; see below. |
| `POST /v1/audio/speech` | (only with `server.tts:` / `--tts`) OpenAI-compatible text-to-speech; see below. |
| `POST /v1/embeddings` | (only with `server.embeddings:` / `--embeddings`) OpenAI-compatible text embeddings; see below. |
| `POST /v1/rerank` | (only with `server.rerank:` / `--rerank`) Cohere/Jina-shaped reranking; see below. Also served at `/rerank`. |

### Capacity and live-request metrics

`GET /v1/metrics` carries, under `server`, everything a load balancer or a
harness that fans out subagents needs to size its work to the server. All
of it is read-only and cheap (the poll is on the request log's silent
list); the keyless `GET /health?ready=1` gives the coarse yes/no without
the key.

| Section | Fields | Meaning |
|---|---|---|
| `concurrency` | `decode_batch`, `queue_cap`, `in_flight`, `waiting` | The effective decode width (`GMLX_DECODE_BATCH`, bounded by the capacity frontier), the waiting-queue cap, streams generating now (each resident entry's `in_flight`: its busy refcount minus the process-lifetime hold the primary preload keeps, so an idle server reads 0), and requests waiting for a slot (every resident engine's server queue plus its unadmitted prompts, summed). Each resident model decodes on its own engine with its own width, so `in_flight` is server-wide while `resident_models[].in_flight` is the per-model number to compare against `decode_batch`. |
| `queue` | `waiting`, `cap`, `eta_s`, `rejections`, `last_reject_reason` | The waiting census again, the cap it is judged against, and the drain estimate in seconds a client would get as `Retry-After` right now (`0` with nothing waiting; the same formula: waiting x mean tokens per request / aggregate decode rate, clamped 2-60 s). |
| `requests[]` | `id`, `uid`, `model`, `state`, `position`, `prompt_tokens`, `generated`, `max_tokens`, `elapsed_s`, `ttft_s`, `decode_tok_s`, `cache {tier, warm_tokens}`, `speculative {rounds, accepted, drafted, accept_rate}` | One row per request the serve path knows about, queued rows first in queue order. `state` is `queued` (server queue or engine-side unadmitted; `position` is the place in line), `prefill`, or `decode`. `cache.tier` is the prefix-cache hit the row got (`exact`, `block`, or gmlx's own `ckpt` / `anchor` restores; `miss`; `hit` when only the warm-token count is known) and `warm_tokens` how many prompt tokens it reused. `speculative` is the drafter's acceptance since the row started (exact at batch width 1, shared across the batch otherwise) and `null` without a drafter. Rows come from each engine's tick, refreshed at most four times a second per engine and merged across resident models (`position` is the place in that model's queue); an idle engine contributes nothing. Drafted models (`draft_gguf` / `speculative: true`) report rows like any other; their `speculative` numbers are the drafter's per-generation round tally, which is exact at batch width 1 and shared across a wider speculative batch. |
| `governor` | `band`, counters | The memory governor's band and shed history; see the `GMLX_GOVERNOR` / `GMLX_GOV_*` rows in [cli.md](env-vars.md). |
| `memory` | `active_bytes`, `cache_bytes`, `headroom_bytes`, `arena_bytes`, `arena_nominal_bytes`, `kv_room_bytes`, `arena_hits`, `arena_lookups` | MLX's active and cached bytes and the headroom the admission gate reads. The arena fields appear for a streaming model: the decode arena's bytes now, its sized capacity, and the KV room the arena leaves under the governor ceiling (`GMLX_STREAM_KV_CTX`). A gap between the first two is a pressure or governor shrink, or a lend to the prefill ring. `arena_hits` over `arena_lookups` is the arena hit rate since load. It rises as the arena warms. |
| `capacity` | `max_ctx` by width, `max_width_at_depth`, byte budgets | The boot capacity table (`GMLX_OVERCOMMIT=1` disables its ceilings); absent for an HF fall-through load. Priced per cache entry: growing attention KV at the resolved `kv_bits` width, sliding windows at their cap, the fixed recurrent state of hybrid models (gated DeltaNet, Mamba2, KDA) once per sequence. |
| `rates` | `decode_tok_s`, `decode_streams`, `prefill_tok_s_recent`, `decode_tok_s_recent`, `decode_tok_s_lifetime` | The aggregate decode rate right now (the sum over the rows in `requests[]` that are decoding) and how many streams it is spread over; the mean prefill and per-stream decode rates over the last eight completed requests (what the dry-run's `est_ttft_s` is computed from); the lifetime mean decode rate. |

The sections are independent: a server without a capacity table (an HF
fall-through load) omits `capacity` and everything else still appears;
any probe failure inside a section leaves that section's live fields
`null` rather than failing the snapshot.

**Asking before sending.** Two routes turn the same numbers into
answers a dispatcher can act on without a refused request:

- `POST /v1/estimate` (or `"dry_run": true` on `/v1/chat/completions`)
  takes a chat-completions body and returns, for a resident model:
  `prompt_tokens` (the rendered prompt, tokenized the way the request
  would be), `warm_tokens` and `cache_tier` (how much of the prefix the
  prefix cache already holds, and the deepest tier holding it: the block
  chain, the exact index, or a pinned checkpoint record on `ckpt`-tier
  models - the request itself restores by the runtime's own precedence; which
  server holds your prefix, and how much of it, is the routing signal
  across machines), `need_bytes` (the prompt's KV plus the prefill
  transient, plus `max_tokens` when the body pins one - exactly what the
  memory preflight prices), `avail_now_bytes` / `fits_now` (against the
  live headroom) and `avail_drained_bytes` / `fits_drained` (against the
  working set with the batch drained: the preflight's own refusal line),
  `context_ok` against `context_limit` (`context_limit_source` says
  whether that is the configured `max_kv_size` or, with nothing
  configured, the GGUF's trained context), and `est_ttft_s`
  (queue drain plus the cold suffix at the recent prefill rate). A model
  that is not resident answers `resident: false` with null fits - the
  dry-run never loads a model. Requests carrying images / audio / video
  render but are not priced (`media: true`), matching the preflight.
- A chat request for a model the load gate cannot admit right now (its
  weights would fit the box, but not next to what is resident and
  pinned or busy, or not without pushing the kernel under the governor's
  reclaimable floor while other processes hold the rest) answers `503`
  with `{"error": {"type": "model_load_deferred", ...}}`, the gate's
  numbers in the message, and `Retry-After`. The gate judges the load
  against the serve ceiling (working set less margin and kernel reserve,
  the same ceiling request admission uses) and against the kernel's own
  reclaimable count, so a load that would Metal-OOM in the weight warm
  is refused before it starts. Memory the kernel is still returning (an
  unload or eviction a moment earlier) is waited for, up to 3 s, before
  a load is deferred. Explicitly `POST /unload` the resident model, or
  retry once its streams drain and the pool can evict it.
- `GET /v1/capacity/plan?width=W&depth=D` evaluates the fan-out policy
  where the numbers live: `ok` when the capacity table holds `W` streams
  at `D` tokens each (`max_context_at_width` is read at the smallest
  tabulated width >= `W`, so it is conservative between rows), and
  `admit_now` when, on top of that, the governor is not orange/red,
  nothing is waiting, and at least `W` decode slots are free (one under
  yellow). `reason` names the first condition that fails. Without a
  table (an HF fall-through load, or `GMLX_OVERCOMMIT=1`) `ok` is null
  and only the timing is judged.

The Prometheus rendering (`?format=prometheus`) flattens these to
`gmlx_concurrency_in_flight`, `gmlx_queue_eta_s`,
`gmlx_governor_band{band="green"} 1`, `gmlx_capacity_max_ctx{width="8"}`,
`gmlx_resident_models_busy{model="<id>",profile="default"}` (`model`
is the configured `id[@profile]` whose request built the entry; the
`profile` label - adapter basename, model kind and/or a short hash of
the load signature, `default` for a bare single-model launch - keeps
two entries backing one GGUF as distinct series; both are fixed for the
entry's lifetime) and so on; `requests[]` is high-cardinality and contributes only
`gmlx_requests_count`.

## API capabilities

The protocol surface is inherited from mlx-vlm (gmlx swaps the model layer,
not the handlers), so the engine's request features work unchanged on GGUF
models.

The context window comes from the GGUF's own metadata; there is no
server-side override or per-request context knob. (`GMLX_ROPE_FACTORS`
exists as an expert escape hatch for models with mis-declared RoPE scaling -
see the environment-variable appendix in [cli.md](cli.md).)

A few request features worth knowing about:

### Tool / function calling

OpenAI `tools` + `tool_calls` on `/v1/chat/completions`, and Anthropic
`tools` / `tool_use` blocks on `/v1/messages`. The
tool-call parser is inferred from the model's chat template
(`mlx_lm.tool_parsers`, plus mlx-vlm's own additions, e.g. gemma4's
`<|tool_call>` format): a model whose template defines a tool-call syntax
gets parsing automatically, nothing to configure. Streaming works too: a
parsed call ends the stream with `finish_reason: "tool_calls"`.

`tool_choice` is honest but limited:

- `"none"` (and Anthropic `{"type": "none"}`) is enforced server-side: the
  tools are stripped before the chat template runs, so the model never sees
  them and cannot emit a call.
- `"auto"` is the default behaviour - the model decides.
- `"required"` and named-function forms (Anthropic `{"type": "any"}` /
  `{"type": "tool", ...}`) are forwarded to the chat template as a plain
  template variable and honored only if that template implements them; there
  is no grammar-level enforcement, so clients must not rely on them for
  routing. When a forced call was requested and the output parsed zero tool
  calls, the server logs one warning naming the mismatch.

This is the client-side-loop shape: the server parses the calls, the client
executes them and sends results back. To run the loop server-side instead --
config-allowlisted MCP tools, no client loop at all -- serve an assistant id
(`server.assistants:`, [assistant.md](assistant.md#served-assistants)).

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.6-27b",
  "messages": [{"role": "user", "content": "Weather in Paris?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
}'
```

### Parameter support

The request schemas accept unknown fields, so nothing is ever rejected just
for being present - but a parameter the server accepts and then never reads
is worse than an error. Two guardrails:

- **honored** parameters change the response; **ignored** parameters are
  accepted and skipped. Every ignored parameter a request sets draws one
  `warning` line in the server log naming it.
- The table is kept in lockstep with the allowlists the warning uses
  (`gmlx/serve/patches/api_contract.py`); a test cross-checks the two, so
  they cannot drift.

The standard sampling parameters (`max_tokens` / `max_output_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `top_n_sigma`, `p_less`,
`typical_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`
and their `*_context_size` companions, `enable_thinking`,
`thinking_budget`, and the OpenAI standard `reasoning` /
`reasoning_effort` controls) are honored on all three generation
dialects. The rest:

| Parameter | `/v1/chat/completions` | `/v1/responses` | `/v1/messages` | Notes |
|-----------|------------------------|-----------------|----------------|-------|
| `n` | ignored | ignored | ignored | always one choice; `n > 1` on `/v1/completions` is a 400 |
| `user` | ignored | ignored | ignored | no per-user accounting |
| `parallel_tool_calls` | ignored | ignored | ignored | the template decides how many calls to emit |
| `tool_choice` | none/auto enforced | template-dependent | none/auto enforced | `required`/named forms are template-dependent; see above |
| `metadata` | ignored | ignored | ignored | accepted for Anthropic compatibility, never read |
| `output_config` | ignored | ignored | honored | Anthropic `json_schema` format maps onto structured output |
| `logit_bias` | honored | honored | honored | token-id keyed |
| `seed` | honored | honored | honored | per-request sampling seed |
| `presence_penalty` | honored | honored | honored | |
| `frequency_penalty` | honored | honored | honored | |
| `stream_options` | honored | ignored | ignored | `include_usage` adds the final usage chunk (chat + `/v1/completions`) |
| `timings_per_token` | honored | ignored | ignored | streamed chat chunks carry `timings.predicted_n`, the exact cumulative output-token count (llama.cpp convention) |
| `response_format` | honored | honored | honored | `json_schema` / `json_object`; unknown types are rejected (see below) |
| `logprobs` | honored | ignored | ignored | chat-only; `/v1/completions` never returns logprobs |
| `top_logprobs` | honored | ignored | ignored | capped by `TOP_LOGPROBS_K` (below) |
| `stop` | honored | ignored | ignored | chat + `/v1/completions`; Anthropic uses `stop_sequences` |
| `stop_sequences` | ignored | ignored | honored | the Anthropic-native spelling |
| `chat_template_kwargs` | honored | honored | honored | extra template variables, request wins over profile |
| `profile` | honored | honored | honored | sampling/system profile by name ([profiles](server-config.md#profiles-sampling-profiles-and-built-in-intents)) |
| `xtc_probability` | honored | honored | honored | XTC sampling (with `xtc_threshold`) |

`echo`, `suffix`, `best_of > 1`, and list / token-array prompts on
`/v1/completions` are **rejected** with a 400 and a message naming the
limit.

### Structured output

`response_format: {"type": "json_schema", ...}` gives grammar-constrained
decoding: the model cannot emit tokens that violate the schema. The backing
engine is [llguidance](https://github.com/guidance-ai/llguidance), a declared
mlx-vlm dependency installed with the base package; nothing separate to
install. The Anthropic endpoint maps an `output_config` of type `json_schema`
to the same machinery. Two honest caveats:

- `"type": "json_schema"` constrains to your schema; `"json_object"` is
  accepted and constrained to a permissive object grammar (valid JSON, no
  particular shape). Unknown types are rejected with
  `Unsupported response_format type`.
- Not available on speculative/MTP models. The engine rejects per-request
  logits processors there (the same restriction as XTC above), so such a
  request errors.

A malformed schema (or one llguidance can't compile) is rejected with a `400`
before generation, not a `500`. The first structured request per model pays a
one-time llguidance tokenizer build (~1.5 s on a 150k vocab), cached for the
process lifetime.

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.6-27b",
  "messages": [{"role": "user", "content": "Name a city and its population."}],
  "response_format": {"type": "json_schema", "json_schema": {"schema": {
    "type": "object",
    "properties": {"city": {"type": "string"},
                   "population": {"type": "integer"}},
    "required": ["city", "population"]}}}
}'
```

### Logprobs

`logprobs: true` returns each generated token's logprob; `top_logprobs: N`
asks for the N most likely alternatives per token. The alternatives are
capped by a server-side env var, `TOP_LOGPROBS_K` (0-20; 20 is OpenAI's own
cap), whose default is `0`. A request with `top_logprobs` still succeeds, but
the alternatives lists stay empty until the server is started with the cap
raised. It is an mlx-vlm engine env var (there is no config key):

```sh
TOP_LOGPROBS_K=5 gmlx serve --config ~/.config/gmlx/gmlx.yaml
```

### Vision messages

OpenAI `image_url` content parts work against a VLM entry: a model configured
with `mmproj:` (or served with `--mmproj`). The url can be an `http(s)://`
URL or a base64 `data:` URI (`"url": "data:image/jpeg;base64,<...>"`).

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "gemma-e4b-vlm",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
  ]}]
}'
```

---
