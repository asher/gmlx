# HTTP API

`gmlx serve` speaks the OpenAI and Anthropic chat protocols, plus endpoints
for capacity, residency and the prompt cache that a client or a load
balancer can read. A request names a model by id, profile or alias, each
protocol honors its own set of request fields, and the server refuses what
cannot fit with a typed error. The YAML that configures the server is in
[Configuration](config.md).

- [Addressing a model in a request](#addressing-a-model-in-a-request)
- [Endpoints](#endpoints)
- [Capacity and live-request metrics](#capacity-and-live-request-metrics)
- [API capabilities](#api-capabilities)
- [Limits and back-pressure](#limits-and-back-pressure)
- [Hugging Face policy](#hugging-face-policy)

## Addressing a model in a request

```jsonc
{"model": "qwen3.8-27b-ud-q6"}                                 // the model's configured profile, else the family base
{"model": "qwen3.8-27b-ud-q6@coding"}                          // a built-in intent, resolved for the model's family
{"model": "qwen3.8-27b-ud-q6@qwen-coder"}                      // an inline user profile, which overrides model.profile
{"model": "coder"}                                        // an alias, here for qwen3.8-27b-ud-q6@qwen-coder
{"model": "qwen3.8-27b-ud-q6", "profile": "coding",            // the profile field, via extra_body in the OpenAI SDK
 "temperature": 0.1}                                      //   plus an explicit field that overrides the profile
```

The `@profile` suffix is split on the last `@` and only treated as a profile
when it names a known one, so an id that itself contains an `@` stays
intact. An unknown id returns 404 of type `model_not_found`, with the
served ids in `available_models`. An unknown profile returns 400 of type
`unknown_profile`, which lists the valid ones. With `model` empty, the server
uses its default model, else the sole model, else returns 400. The same
addressing works on the CLI, as `gmlx run <id-or-path>@coding` or
`--profile coding`.

## Endpoints

Beside the OpenAI and Anthropic generation routes, the server has routes
for metrics, capacity, residency, the prompt cache and the optional
services. Every route except `/health` needs the API key when one is set.
mlx-vlm's image generation routes are present but cannot serve a GGUF
model.

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | It serves OpenAI chat completions and is the primary route. |
| `POST /v1/responses` | It serves OpenAI Responses. |
| `POST /v1/messages` | It serves Anthropic Messages, and `/v1/messages/count_tokens` counts a body's input tokens. |
| `POST /v1/completions` | It serves classic text completions, with a single string prompt, a single choice and no chat template. |
| `GET /v1/models` | It lists configured ids and aliases with their markers, also at `/models`. |
| `GET /health` | It reports liveness, and `?ready=1` adds a readiness verdict. |
| `GET /v1/metrics` | It returns the runtime snapshot, also at `/metrics`, or Prometheus text with `?format=prometheus`. |
| `POST /v1/estimate` | It runs a dry-run admission check for a chat body. |
| `GET /v1/capacity/plan` | It says whether `width` streams fit at `depth` tokens each, and whether they may start now. |
| `GET /v1/cache/stats` | It returns prompt cache statistics, or `{"enabled": false}`. |
| `POST /v1/cache/reset` | It clears the prompt cache for all resident models, or for one with `{"model": "<id>"}`. |
| `POST /unload` | It evicts the resident model that `{"model": "<id>"}` names, or every idle model when the body is empty. The route answers 409 while that model streams. |
| `POST /v1/keep` | With `{"model": "<id>"}`, it keeps that model resident past the idle timeout and warms it. `"warm": false` skips the warm-up, and `"keep": false` releases it. |
| `POST /v1/reload` | It re-reads the config and re-registers models, keeping entries whose load parameters are unchanged. |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | It transcribes or translates speech, with `stt` configured as in [Speech, embeddings and rerank](services.md). |
| `POST /v1/audio/speech` | It turns text into speech, with `tts` configured. |
| `POST /v1/embeddings` | It returns text embeddings, with `embeddings` configured. |
| `POST /v1/rerank` | It reranks documents, with `rerank` configured, also at `/rerank`. |
| `POST /v1/systemone` | It answers a fixed question set about a state, also at `/systemone`. See [Structured decisions](decisions.md). |
| `POST /v1/prewarm` | It reads a decision state ahead of its questions, also at `/prewarm`. See [Repeated states](decisions.md#repeated-states). |

`GET /v1/models` lists configured and discovered ids plus alias presets,
and never the Hugging Face cache. Each entry carries `resident`, `pinned`,
`speculative`, `vlm`, `profile` and `default` markers and two context
figures.

`context_length` is the GGUF's trained window, or the model's
[`max_kv_size`](config.md#loadmax_kv_size) when that is smaller.
`max_context_at_width_1` is how much of the window fits in memory for a
single stream. It is `null` for every model except the one that the boot
capacity table was built for, and for that model too when there is no
table or `GMLX_OVERCOMMIT=1` is set. A harness sizes its context window
from the smaller of the two, or from `context_length` when
`max_context_at_width_1` is `null`.

A resident model with KV quantization configured adds a `kv_quant` object
with `scheme`, `bits`, `group_size`, `layers_quantized`, `layers_fp16` and a
`verdict` of `full`, `partial` or `dropped`. Under
[kvarn](glossary.md#kvarn) it also carries `value_bits` and `tail_tokens`.
The object also carries `verdict_batched`, which differs from `verdict`
when the model runs fp16 KV while batched, as a speculative model does
under `uniform`.

`GET /health` returns only `{"status": "healthy", "pid": N}`. Adding
`?ready=1` gives a coarse readiness verdict, either 200 with `"ready": true`
or 503 with a one-word `reason` and a `Retry-After` header. The reason is
`pressure` when the [governor](glossary.md#governor) is orange or red,
`queue` when requests are waiting and `busy` when all engines are at their
decode width.

`POST /v1/completions` honors the standard sampling parameters, `seed`,
`stop`, `stream`, `stream_options` and `profile`. List or token-array prompts, `n > 1`, `echo`,
`suffix` and `best_of > 1` are rejected with a 400.

The residency routes act on what
[Memory and residency](config.md#memory-and-residency) configures.
`/unload` also unloads a pinned model, which stays pinned when the next
request loads it again. `/v1/keep` is what `gmlx launch --model` and voice
sessions call, and a kept model stays LRU-evictable under memory pressure.
`/v1/reload` returns `{"status": "unsupported"}` outside config mode, as
[Changing the file](config.md#changing-the-file) explains.

## Capacity and live-request metrics

`GET /v1/metrics` carries, under `server`, what a load balancer or a harness
that fans out subagents needs to size its work. The sections are
independent, so a probe failure inside one leaves its live fields `null`
instead of failing the snapshot.

| Section | Fields | Meaning |
|---|---|---|
| `concurrency` | `decode_batch`, `queue_cap`, `in_flight`, `waiting` | They give the decode width, the queue cap, the streams generating now and the requests waiting for a slot. |
| `queue` | `waiting`, `cap`, `eta_s`, `rejections`, `last_reject_reason` | They give the waiting count, its cap, the drain estimate that `Retry-After` would carry now, the queue-cap rejection count and the last rejection's reason. |
| `requests[]` | Each request has one row. | Queued rows come first. `state` is `queued`, `prefill` or `decode`. |
| `resident_models[]` | Each model has `in_flight`, `pinned`, `kept` and bytes. | Each model decodes on a separate engine, so compare each model's `in_flight` with `decode_batch`. |
| `governor` | `band`, counters | They give the memory governor's band and shed history. |
| `memory` | `active_bytes`, `cache_bytes`, `headroom_bytes`, arena fields | They hold MLX's active and cached bytes, the measured free working set, and a streamed model's [arena](glossary.md#arena) bytes, capacity and hit rate. |
| `capacity` | `max_ctx` by width, `max_width_at_depth`, byte budgets | It holds the boot capacity table, which is absent for a non-GGUF model that the mlx-vlm model loader loads. |
| `rates` | `decode_tok_s`, `decode_streams`, `prefill_tok_s_recent`, `decode_tok_s_recent`, `decode_tok_s_lifetime` | They give the aggregate decode rate now and its stream count, the recent means over the last eight requests, and the lifetime mean. |

A request row carries `id`, `model`, `state`, `position`, `prompt_tokens`,
`generated`, `max_tokens`, `elapsed_s`, `ttft_s` and `decode_tok_s`, and
two structured fields. `cache` holds the tier its prefix hit, one of
`exact`, `block`, `ckpt`, `anchor` and `miss`, or `hit` when only the reused
token count is known, plus the `warm_tokens` it reused. `speculative` holds
the [drafter's](glossary.md#drafter) rounds, drafted and accepted counts
and accept rate, exact at batch width 1 and shared across a wider batch, or
`null` without a drafter. Rows refresh at most four times a second on each engine.

Two routes use the figures that `/v1/metrics` reports to tell a dispatcher
whether to proceed before sending a request. `POST /v1/estimate` takes a chat-completions body and
returns an admission estimate for a resident model. `prompt_tokens`,
`warm_tokens` and `cache_tier` say how much of the prefix the cache already
holds and on which tier, which is the routing signal across machines.

`need_bytes` is the prefill transient plus the
KV for the prompt's tokens and, when the body pins `max_tokens`, for that
many more. `fits_now` and
`fits_drained` judge that against the current free memory and the drained
working set, while `context_ok` judges it against `context_limit`.

`est_ttft_s` estimates the time to first token. The dry run never loads a
model, so a model that is not resident answers `resident: false`. A media
request is rendered but not estimated. `"dry_run": true` on
`/v1/chat/completions` returns the same estimate instead of generating.

`GET /v1/capacity/plan?width=W&depth=D` answers `ok` when the capacity
table holds `W` streams at `D` tokens each, reading the table conservatively
at the smallest tabulated width at or above `W`. It answers `admit_now` when
the geometry does not fail, the governor is not orange or red, nothing is
waiting and at least `W` decode slots are free. Without a capacity table,
or with `GMLX_OVERCOMMIT=1`, `ok` is `null` and `admit_now` judges only the
timing. Under a yellow band only one free slot is counted, so a wider
fan-out waits for the band to clear. `reason` names the first condition
that fails.

The Prometheus rendering of `/v1/metrics` flattens its sections to gauges
such as `gmlx_concurrency_in_flight`, `gmlx_queue_eta_s`,
`gmlx_governor_band` with a `band` label and `gmlx_capacity_max_ctx` with a
`width` label. Per-model series carry a `model` label, and entries that carry
a profile add a `profile` label. `requests[]` is high-cardinality and contributes only its count.

## API capabilities

The protocol surface follows mlx-vlm's. gmlx patches mlx-vlm's handlers so that
upstream request features work on GGUF models, and it adds the
`/v1/completions` route.

### Tool calling

`/v1/chat/completions` takes OpenAI `tools` and answers with `tool_calls`,
and `/v1/messages` takes Anthropic `tools` and answers with `tool_use`
blocks. The parser is inferred from the model's chat template, so a model
whose template defines a tool-call syntax gets parsing with nothing to
configure. Streaming works too, and a parsed call ends the stream with
`finish_reason: "tool_calls"`.

`tool_choice` is enforced where it can be:

| Value | Behavior |
|-------|----------|
| `none` | The server enforces it by stripping the tools before the template runs, so the model cannot emit a call. |
| `auto` | This is the default, and the model decides whether to call a tool. |
| `required` and named-function forms | They reach the template as a variable and work only if the template implements them. A forced call that produces no call logs a warning. |

The server only parses the calls, and the client runs them and sends the
results back. To have the server run config-allowlisted MCP tools itself,
serve an [assistant id](assistant.md#served-assistants).

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.8-27b-ud-q6",
  "messages": [{"role": "user", "content": "Weather in Paris?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
}'
```

### Parameter support

The request schemas accept unknown fields, so nothing is rejected for being
present. An honored parameter changes the response. An ignored one is
accepted and skipped, and a request that sets any produces one warning line
in the server log naming them all. `/v1/systemone` has its own fields, and
[Choosing a model](decisions.md#choosing-a-model) lists which of them each
model reads.

The generation routes honor the standard sampling parameters.
They are `max_tokens` and `max_output_tokens`, `temperature`, `top_p`,
`top_k`, `min_p`, `top_n_sigma`, `p_less`, `typical_p`, `repetition_penalty`,
`presence_penalty`, `frequency_penalty` and their `*_context_size` companions,
`enable_thinking`, `thinking_budget` and the OpenAI `reasoning` and
`reasoning_effort` controls. Support for the other parameters differs by
route:

| Parameter | `/v1/chat/completions` | `/v1/responses` | `/v1/messages` | Notes |
|-----------|------------------------|-----------------|----------------|-------|
| `max_completion_tokens` | Honored | Ignored | Ignored | It is OpenAI's current name for the chat output cap, and it wins over `max_tokens` and a profile value. |
| `n` | Ignored | Ignored | Ignored | The server always returns a single choice. |
| `user` | Ignored | Ignored | Ignored | The server keeps no per-user accounting. |
| `parallel_tool_calls` | Ignored | Ignored | Ignored | The template decides how many calls to emit. |
| `tool_choice` | None/auto enforced | None/auto enforced | None/auto enforced | `required` and named forms depend on the template, as [Tool calling](#tool-calling) describes. |
| `metadata` | Ignored | Ignored | Ignored | The server accepts it for Anthropic compatibility and never reads it. |
| `output_config` | Ignored | Ignored | Honored | Anthropic `json_schema` format maps onto structured output. |
| `logit_bias` | Honored | Honored | Honored | Its keys are token ids. |
| `seed` | Honored | Honored | Honored | It sets the sampling seed for one request. |
| `stream_options` | Honored | Ignored | Ignored | `include_usage` adds the final usage chunk on chat and `/v1/completions`. |
| `timings_per_token` | Honored | Ignored | Ignored | Streamed chat chunks carry `timings.predicted_n`, the exact cumulative output-token count, following llama.cpp. |
| `response_format` | Honored | Honored | Honored | It takes `json_schema` or `json_object`. Unknown types are rejected, as [Structured output](#structured-output) explains. |
| `logprobs` | Honored | Ignored | Ignored | Only chat returns logprobs, and `/v1/completions` never does. |
| `top_logprobs` | Honored | Ignored | Ignored | `TOP_LOGPROBS_K` caps it, as [Logprobs](#logprobs) explains. |
| `stop` | Honored | Ignored | Ignored | Chat and `/v1/completions` honor it, and Anthropic uses `stop_sequences`. |
| `stop_sequences` | Ignored | Ignored | Honored | It is the Anthropic spelling of `stop`. |
| `chat_template_kwargs` | Honored | Honored | Honored | It passes extra template variables, and the request's values override the profile's. |
| `profile` | Honored | Honored | Honored | It selects a sampling and system [profile](config.md#profiles) by name. |
| `xtc_probability` | Honored | Honored | Honored | It turns on XTC sampling, together with `xtc_threshold`. |

### Structured output

`response_format: {"type": "json_schema", ...}` gives grammar-constrained
decoding, where the model cannot emit tokens that violate the schema.
Enforcement comes from [llguidance](https://github.com/guidance-ai/llguidance),
which the pinned mlx-vlm release depends on, so it is present in every
install. On the Anthropic endpoint an `output_config`
of type `json_schema` maps to the same engine. `"json_object"` is accepted
and constrained to a permissive object grammar, and unknown types are
rejected.

A malformed schema is rejected with a 400 before generation. The first
structured request on a model runs a one-time tokenizer build of about
1.5 s, cached for the process lifetime. A speculative model cannot take
request-level logits processors, so a structured request to one errors.

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.8-27b-ud-q6",
  "messages": [{"role": "user", "content": "Name a city and its population."}],
  "response_format": {"type": "json_schema", "json_schema": {"schema": {
    "type": "object",
    "properties": {"city": {"type": "string"},
                   "population": {"type": "integer"}},
    "required": ["city", "population"]}}}
}'
```

### Logprobs

`logprobs: true` returns each generated token's logprob, and `top_logprobs: N`
asks for the N most likely alternatives for each token. The alternatives are
capped by the server-side `TOP_LOGPROBS_K` variable, 0 to 20, with a default
of 0, so until the server is started with the cap raised the lists stay
empty. No config key sets it, so set it in the server's environment:

```sh
TOP_LOGPROBS_K=5 gmlx serve --config ~/.config/gmlx/gmlx.yaml
```

### Vision messages

OpenAI `image_url` content parts work against a model configured with
`mmproj:`. The image can be an `http(s)://` URL or a base64 `data:` URI.

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "gemma-e4b-vlm",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
  ]}]
}'
```

## Limits and back-pressure

A model's context window comes from its GGUF metadata. A request cannot
change it, and the [`max_kv_size`](config.md#loadmax_kv_size) load key can
only lower it.

| Condition | Response | Switch |
|-----------|----------|--------|
| The prompt plus `max_tokens` exceeds the context budget. | The server answers 400 with both token counts and the budget. | [`max_kv_size`](config.md#loadmax_kv_size) |
| The prompt cannot fit in memory. | The server answers 400 with the estimated need and the available budget. | `GMLX_PREFLIGHT_MEM=0` |
| More requests are waiting than the queue cap. | The server answers 503 of type `server_overloaded`, with `Retry-After` set to the estimated drain time of 2 to 60 seconds. | `GMLX_QUEUE_DEPTH_CAP` |
| A model cannot load beside the resident models that are pinned or busy. | The server answers 503 of type `model_load_deferred`, with the load gate's numbers in the message and `Retry-After`. | `GMLX_OVERCOMMIT=1` |
| Memory runs out while a request streams. | The [governor](glossary.md#governor) ends the largest request with an error of type `server_overloaded_shed` and `finish_reason` `shed`. | `GMLX_GOVERNOR=0` |
| A streaming request is silent, as during a long prefill. | The server sends periodic SSE comment lines, so that read timeouts do not drop the connection. | `GMLX_SSE_KEEPALIVE_S` |

The preflight uses the same estimate as `POST /v1/estimate`, which
[Capacity and live-request metrics](#capacity-and-live-request-metrics)
describes. The load gate judges a model's weights against the memory left
beside the resident models that are pinned or busy, and against the
governor's floor. When a load would leave less than the floor, the gate
first waits up to 3 seconds for memory that the kernel is still returning
from a recent unload.

Each switch variable, and the decode batch width that sets the queue cap,
is listed among the
[server environment variables](env-vars.md#server), except `GMLX_GOVERNOR`,
which is among the [runtime environment variables](env-vars.md#runtime).

Both 400s for a request that does not fit start their message with `prompt
is too long`, and the budget error also says the request `exceeds the context
window`. Agent clients that recognize those words, pi among them, compact the
conversation and retry instead of stopping at the error.

## Hugging Face policy

A request never makes the server download the model it names. A
generation request whose `model` is not a configured id gets the 404 that
[Addressing a model in a request](#addressing-a-model-in-a-request)
describes, whatever the id looks like. The mlx-vlm model loader, which runs
after the id lookup, is gated as well. Anything that reaches it with a repo id instead
of a GGUF or local path gets a 403 of type `hf_access_disabled`, so no
route can fetch a chat model. The service models that the config names
download in the background at start, as
[Speech, embeddings and rerank](services.md) describes.

An `hf:` ref in `models:` resolves from the local Hugging Face cache and
never from the network, so a repo that another tool downloaded can be
served without copying the file. `server.hf_cache: true` runs the Hugging
Face libraries offline and lets a repo id that reaches the mlx-vlm model
loader resolve from that cache instead of getting the 403. It never changes what
a request may name.
