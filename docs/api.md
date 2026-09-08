# HTTP API

The endpoints `gmlx serve` exposes, how a request names a model, and which
request fields each protocol honors. It is for anyone writing a client or
putting the server behind a load balancer. The YAML that configures the
server is in [server-config.md](server-config.md).

- [Addressing a model in a request](#addressing-a-model-in-a-request)
- [Endpoints](#endpoints)
- [Capacity and live-request metrics](#capacity-and-live-request-metrics)
- [API capabilities](#api-capabilities)
- [Limits and back-pressure](#limits-and-back-pressure)
- [Hugging Face policy](#hugging-face-policy)

To call the API with tools and structured output:

1. Serve a model that supports tool calling and note its id from `gmlx list`.
2. Send `tools` on `/v1/chat/completions` as in [Tool calling](#tool-calling)
   and execute the calls the reply carries.
3. Add `response_format` as in [Structured output](#structured-output) when
   the reply must be JSON of a known shape.

## Addressing a model in a request

```jsonc
{"model": "qwen3.6-27b"}                                 // the model's configured profile (else family base)
{"model": "qwen3.6-27b@coding"}                          // a built-in intent, resolved per the model's family
{"model": "qwen3.6-27b@qwen-coder"}                      // inline user profile (beats model.profile)
{"model": "coder"}                                        // an alias (here: qwen3.6-27b@qwen-coder)
{"model": "qwen3.6-27b", "profile": "coding",            // profile field (needs extra_body in the OpenAI SDK)
 "temperature": 0.1}                                      //   + an explicit field that overrides the profile
```

The `@profile` suffix is split on the last `@` and only treated as a profile
when it names a known one, so an id containing `@`-like text stays intact.
An unknown id returns 404 listing the available ids, and an unknown profile
returns 400 listing the valid ones. An empty `model` uses the server's
default model, else the sole model, else 400. The same addressing works on
the CLI: `gmlx run <id-or-path>@coding`, or `--profile coding`.

## Endpoints

The standard generation and listing routes work, and gmlx adds operational
ones. Every route except `/health` requires the API key when one is set.

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | OpenAI chat completions, the primary route |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages; `/v1/messages/count_tokens` counts a body's input tokens |
| `POST /v1/completions` | classic text completions: one string prompt, one choice, no chat template |
| `GET /v1/models` | configured ids and aliases with their markers; also at `/models` |
| `GET /health` | liveness; `?ready=1` adds a readiness verdict |
| `GET /v1/metrics` | the runtime snapshot, see below; also at `/metrics`, and as Prometheus text with `?format=prometheus` |
| `POST /v1/estimate` | dry-run admission for a chat body, see below |
| `GET /v1/capacity/plan` | can `width` streams run at `depth` tokens each, and may they start now |
| `GET /v1/cache/stats` | prompt cache statistics, or `{"enabled": false}` |
| `POST /v1/cache/reset` | clear the prompt cache for every resident model, or one with `{"model": "<id>"}` |
| `POST /unload` | evict one resident model with `{"model": "<id>"}`, or all with an empty body; 409 while streams are in flight |
| `POST /v1/keep` | keep a model resident through the idle timeout with `{"model": "<id>", "warm": true}`; `"keep": false` releases |
| `POST /v1/reload` | re-read the config and re-register models, keeping entries whose load parameters are unchanged; same as SIGHUP |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | speech-to-text, with `stt` configured ([services.md](services.md)) |
| `POST /v1/audio/speech` | text-to-speech, with `tts` configured |
| `POST /v1/embeddings` | text embeddings, with `embeddings` configured |
| `POST /v1/rerank` | reranking, with `rerank` configured; also at `/rerank` |

`GET /v1/models` lists configured and discovered ids plus alias presets, each
with `resident`, `pinned`, `speculative`, `vlm`, `profile` and `default`
markers, the GGUF's trained `context_length`, and `max_context_at_width_1`
from the capacity table for the model it was derived from. The Hugging Face
cache is never listed.

`GET /health` returns only `{"status": "healthy", "pid": N}` and is the one
route the API key exempts. `?ready=1` adds a coarse readiness verdict: 200
with `"ready": true`, or 503 with a one-word `reason` (`pressure` when the
governor is orange or red, `queue` when requests are waiting, `busy` when
every engine is at its decode width) and a `Retry-After` header.

`POST /v1/completions` supports `max_tokens`, `temperature`, `top_p`, `seed`,
`stop`, `stream` and `profile`. List or token-array prompts, `n > 1`, `echo`,
`suffix` and `best_of > 1` are rejected with a 400. The stock image
generation routes are not usable with GGUF models.

An explicit `/unload` overrides the preload's lifetime hold, so a preloaded
model unloads too, and a model reloaded by request afterward is managed like
any other until a reload with `preload` re-pins it. `/v1/keep` is what
`gmlx launch --model` and voice sessions call; the kept model stays
LRU-evictable under memory pressure. `/v1/reload` returns
`{"status": "unsupported"}` outside config mode
([Reloading the config](server-config.md#reloading-the-config)).

## Capacity and live-request metrics

`GET /v1/metrics` carries, under `server`, what a load balancer or a harness
that fans out subagents needs to size its work. All of it is read-only and
fast to read. The keyless `GET /health?ready=1` gives the coarse yes or no without
the key.

| Section | Fields | Meaning |
|---|---|---|
| `concurrency` | `decode_batch`, `queue_cap`, `in_flight`, `waiting` | the decode width, the waiting-queue cap, streams generating now, and requests waiting for a slot, summed across resident models |
| `queue` | `waiting`, `cap`, `eta_s`, `rejections`, `last_reject_reason` | the waiting count, the cap it is judged against, and the drain estimate a client would receive as `Retry-After` now |
| `requests[]` | `id`, `model`, `state`, `position`, `prompt_tokens`, `generated`, `max_tokens`, `elapsed_s`, `ttft_s`, `decode_tok_s`, `cache`, `speculative` | one row per request, queued rows first; `state` is `queued`, `prefill` or `decode` |
| `resident_models[]` | per model: `in_flight`, `pinned`, `kept`, bytes | the per-model number to compare against `decode_batch`, since each model decodes on its own engine |
| `governor` | `band`, counters | the memory governor's band and shed history |
| `memory` | `active_bytes`, `cache_bytes`, `headroom_bytes`, arena fields | MLX's active and cached bytes, the free memory the admission gate reads, and for a streamed model the arena's bytes, capacity and hit rate |
| `capacity` | `max_ctx` by width, `max_width_at_depth`, byte budgets | the boot capacity table; absent for a Hugging Face fall-through load |
| `rates` | `decode_tok_s`, `decode_streams`, `prefill_tok_s_recent`, `decode_tok_s_recent`, `decode_tok_s_lifetime` | the aggregate decode rate now and its stream count, the recent means over the last eight requests, and the lifetime mean |

A request row's `cache` holds the tier its prefix hit (`exact`, `block`,
`ckpt`, `anchor`, `miss`) and the `warm_tokens` it reused. Its `speculative`
field holds the drafter's rounds, drafted and accepted counts and accept
rate, exact at batch width 1 and shared across a wider batch, or `null`
without a drafter. Rows refresh at most four times a second per engine. The
sections are independent: a probe failure inside one leaves its live fields
`null` rather than failing the snapshot.

Two routes use the same numbers to tell a dispatcher whether to proceed before
sending a request.

`POST /v1/estimate` takes a chat-completions body and returns, for a resident
model, `prompt_tokens`, `warm_tokens` and `cache_tier` (how much of the prefix
the cache already holds and on which tier, which is the routing signal across
machines), `need_bytes` (the prompt's KV plus the prefill transient, plus
`max_tokens` when the body pins one), `fits_now` and `fits_drained` against
the current free memory and the drained working set, `context_ok` against
`context_limit`, and `est_ttft_s`. A model that is not resident answers
`resident: false`; the dry run never loads a model. Media requests render but
are not estimated. `"dry_run": true` on `/v1/chat/completions` returns the same
estimate through the queue cap.

`GET /v1/capacity/plan?width=W&depth=D` answers `ok` when the capacity table
holds `W` streams at `D` tokens each, read conservatively at the smallest
tabulated width at or above `W`, and `admit_now` when additionally the
governor is not orange or red, nothing is waiting, and at least `W` decode
slots are free. `reason` names the first condition that fails.

A chat request for a model the load gate cannot admit right now, because its
weights would not fit beside what is resident and busy or would push the
kernel under the governor's floor, answers 503 with an error of type
`model_load_deferred`, the gate's numbers in the message, and `Retry-After`.
Memory the kernel is still returning from an unload shortly before is
waited for, up to 3 seconds, before a load is deferred.

The Prometheus rendering flattens these to gauges such as
`gmlx_concurrency_in_flight`, `gmlx_queue_eta_s`, `gmlx_governor_band` with a
`band` label, `gmlx_capacity_max_ctx` with a `width` label, and per-model
series with `model` and `profile` labels. `requests[]` is high-cardinality
and contributes only its count.

## API capabilities

The protocol surface is inherited from mlx-vlm, since gmlx swaps the model
layer rather than the handlers, so the engine's request features work
unchanged on GGUF models. The context window comes from the GGUF's own
metadata; there is no server-side override or per-request context setting.

### Tool calling

OpenAI `tools` and `tool_calls` on `/v1/chat/completions`, and Anthropic
`tools` and `tool_use` blocks on `/v1/messages`. The parser is inferred from
the model's chat template, so a model whose template defines a tool-call
syntax gets parsing with nothing to configure. Streaming works too: a parsed
call ends the stream with `finish_reason: "tool_calls"`.

`tool_choice` is enforced where it can be:

| Value | Behavior |
|-------|----------|
| `none` | enforced: the tools are stripped before the template runs, so the model cannot emit a call |
| `auto` | the default; the model decides |
| `required` and named-function forms | forwarded to the template as a variable and honored only if the template implements them; the server logs a warning when a forced call produced no call |

This is the client-side-loop shape: the server parses the calls, the client
executes them and sends results back. To run the loop server-side with
config-allowlisted MCP tools, serve an
[assistant id](assistant.md#served-assistants).

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

The request schemas accept unknown fields, so nothing is rejected for being
present. A honored parameter changes the response; an ignored one is
accepted and skipped, and every ignored parameter a request sets produces one
warning line in the server log naming it. The table is cross-checked by a
test against the allowlists the warning uses.

The standard sampling parameters (`max_tokens` and `max_output_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `top_n_sigma`, `p_less`,
`typical_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`
and their `*_context_size` companions, `enable_thinking`,
`thinking_budget`, and the OpenAI `reasoning` and `reasoning_effort`
controls) are honored on all three generation dialects. The rest:

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
| `chat_template_kwargs` | honored | honored | honored | extra template variables, request overrides profile |
| `profile` | honored | honored | honored | sampling/system profile by name ([profiles](server-config.md#profiles)) |
| `xtc_probability` | honored | honored | honored | XTC sampling (with `xtc_threshold`) |

`echo`, `suffix`, `best_of > 1`, and list or token-array prompts on
`/v1/completions` are rejected with a 400 naming the limit.

### Structured output

`response_format: {"type": "json_schema", ...}` gives grammar-constrained
decoding: the model cannot emit tokens that violate the schema. The backing
engine is [llguidance](https://github.com/guidance-ai/llguidance), installed
with the base package. The Anthropic endpoint maps an `output_config` of type
`json_schema` to the same engine. `"json_object"` is accepted and
constrained to a permissive object grammar; unknown types are rejected.

A malformed schema is rejected with a 400 before generation. The first
structured request per model runs a one-time tokenizer build of about 1.5 s,
cached for the process lifetime. Structured output is not available on
speculative models, since the engine rejects per-request logits processors
there, and such a request errors.

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

`logprobs: true` returns each generated token's logprob, and `top_logprobs: N`
asks for the N most likely alternatives per token. The alternatives are
capped by the server-side `TOP_LOGPROBS_K` variable, 0 to 20, whose default
is 0, so the lists stay empty until the server is started with the cap
raised. It is an engine variable with no config key:

```sh
TOP_LOGPROBS_K=5 gmlx serve --config ~/.config/gmlx/gmlx.yaml
```

### Vision messages

OpenAI `image_url` content parts work against a model configured with
`mmproj:`. The url can be an `http(s)://` URL or a base64 `data:` URI.

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

| Condition | Response | Switch |
|-----------|----------|--------|
| the prompt alone cannot fit in memory | 400 with the estimated need and the available budget | `GMLX_PREFLIGHT_MEM=0` |
| more requests waiting than the queue cap | 503 with `Retry-After` set to the estimated drain time, 2 to 60 seconds | `GMLX_QUEUE_DEPTH_CAP` |
| a model cannot be loaded beside what is resident | 503 of type `model_load_deferred` with `Retry-After` | |
| a streaming request is silent, as during a long prefill | an SSE comment line every 15 seconds so read timeouts do not drop the connection | `GMLX_SSE_KEEPALIVE_S` |

The memory preflight estimates prompt KV at the model's per-token size plus the
prefill transient, against the working set with the batch drained.
`max_tokens` counts only when the request pins it explicitly, and media
requests are not estimated. Decode concurrency defaults to 8 requests per
batch step, set by `GMLX_DECODE_BATCH`; past that width aggregate gains
shrink while every stream slows. The switches are documented under
[Server](env-vars.md#server).

## Hugging Face policy

By default the server makes no Hugging Face access. A request `model` that is
neither a local GGUF nor a configured id is refused with a 403 of type
`hf_access_disabled` rather than triggering a download. With
`server.hf_cache: true` a named repo id resolves from the local cache only,
still never the network, and cached models are addressable only when named
in `models:`.
