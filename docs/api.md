# HTTP API

`gmlx serve` answers the OpenAI and Anthropic APIs on one port, so most
clients and SDKs work against it unchanged:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.8-27b-ud-q6",
  "messages": [{"role": "user", "content": "Explain entropy in one paragraph."}]
}'
```

Add `"stream": true` to get the reply as server-sent events. The YAML that
sets up the server is in [Configuration](config.md).

- [Naming a model](#naming-a-model)
- [Endpoints](#endpoints)
- [Request features](#request-features)
  - [Tool calling](#tool-calling)
  - [Parameter support](#parameter-support)
  - [Structured output](#structured-output)
  - [Logprobs](#logprobs)
  - [Media in requests](#media-in-requests)
- [Limits and back-pressure](#limits-and-back-pressure)
- [Hugging Face policy](#hugging-face-policy)

## Naming a model

```jsonc
{"model": "qwen3.8-27b-ud-q6"}                       // the model's configured profile, else the family base
{"model": "qwen3.8-27b-ud-q6@coding"}                // a built-in intent for the model's family
{"model": "qwen3.8-27b-ud-q6@qwen-coder"}            // a profile of your own
{"model": "coder"}                                   // an alias, here for qwen3.8-27b-ud-q6@qwen-coder
{"model": "qwen3.8-27b-ud-q6", "profile": "coding"}  // the profile field, via extra_body in the OpenAI SDK
```

A field in the request, such as `temperature`, overrides the profile. An
empty `model` uses the server's default model, or its only model. An unknown id gets a 404 of
type `model_not_found` that lists the served ids, and an unknown profile
gets a 400 of type `unknown_profile`.

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | OpenAI chat completions, the main route |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages. `/v1/messages/count_tokens` counts a body's input tokens. |
| `POST /v1/completions` | Plain text completion: one string prompt, one choice, no chat template |
| `GET /v1/models` | Configured ids and aliases, with markers and [context figures](capacity.md#context-windows-in-the-model-list) |
| `GET /health` | Liveness, with no API key. `?ready=1` adds readiness, as [Capacity and metrics](capacity.md#readiness-checks) shows. |
| `GET /v1/metrics` | Live snapshot, or Prometheus text with `?format=prometheus` |
| `POST /v1/estimate` | Dry-run admission check for a chat body |
| `GET /v1/capacity/plan` | Whether `width` streams of `depth` tokens fit, and can start now |
| `GET /v1/cache/stats` | Prompt cache statistics, or `{"enabled": false}` |
| `POST /v1/cache/reset` | Clear the prompt cache, for every model or for `{"model": "<id>"}` |
| `POST /unload` | Unload `{"model": "<id>"}`, or every idle model with no body. 409 while that model streams. |
| `POST /v1/keep` | Load and warm `{"model": "<id>"}`, and keep it past its idle timeout. `"warm": false` skips the load. `"keep": false` releases it. |
| `POST /v1/reload` | Re-read the config file and keep models whose load settings did not change |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | Speech to text, with `stt` set up as in [Speech, embeddings and rerank](services.md) |
| `POST /v1/audio/speech` | Text to speech, with `tts` set up. `GET /v1/audio/voices` lists the voices. |
| `POST /v1/embeddings` | Text embeddings, with `embeddings` set up |
| `POST /v1/rerank` | Document reranking, with `rerank` set up |
| `POST /v1/systemone` | Probabilities for a fixed set of questions, as [Structured decisions](decisions.md) shows |
| `POST /v1/prewarm` | Read a decision state ahead of its questions, as [Repeated states](decisions-reference.md#repeated-states) shows |

[Capacity and metrics](capacity.md) covers `/v1/metrics`, `/v1/estimate`
and `/v1/capacity/plan`.

`GET /v1/models` lists the configured ids, discovered ids and aliases, never
the Hugging Face cache. Each entry carries the markers `resident`, `pinned`,
`speculative`, `vlm`, `profile` and `default`.

`/v1/keep` answers `{"status": "kept", "model": "<id>", "warming": true}`
and loads the model in the background. A kept model is not pinned, so the
server can still unload it to make room, as
[Keep, pin and idle](glossary.md#keep-pin-and-idle) describes.

Every route except `/health` needs the API key when the server has one. Most
routes also answer without the `/v1` prefix. Every route takes a JSON body,
except the two audio upload routes, which take a form.

`/v1/completions` honors the sampling parameters, `seed`, `stop`, `stream`,
`stream_options` and `profile`. It refuses list prompts, `n` above 1,
`echo`, `suffix` and `best_of` above 1. It never returns logprobs, and each
choice carries `"logprobs": null`.

## Request features

### Tool calling

`/v1/chat/completions` takes OpenAI `tools` and answers with `tool_calls`.
`/v1/messages` takes Anthropic `tools` and answers with `tool_use` blocks.
The server reads the call syntax from the model's chat template, so there
is nothing to set up. The client runs the tools and sends the results back.

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

`tool_choice: "none"` removes the tools before the template runs, and
`"auto"`, the default, lets the model decide. `"required"` and a named
function work only when the chat template supports them, and the server
logs a warning when a forced call does not come. To have the server run
MCP tools itself, serve an [assistant](assistant.md#served-assistants).

### Parameter support

Every route accepts unknown fields. An honored field changes the response.
An ignored one is skipped, and the server log names it in one warning
line. `/v1/systemone` has its own fields, and
[Fields by model](decisions-reference.md#fields-by-model) lists them.

All generation routes honor `max_tokens` and `max_output_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `top_n_sigma`, `p_less`,
`typical_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`
and their `*_context_size` companions, `enable_thinking`,
`thinking_budget`, and the OpenAI `reasoning` and `reasoning_effort`
controls. The three chat routes ignore `n`, `user`, `parallel_tool_calls`
and `metadata`: they return one choice, and the template decides how many
tool calls to make. The other fields differ by route:

| Parameter | `/v1/chat/completions` | `/v1/responses` | `/v1/messages` | Notes |
|-----------|------------------------|-----------------|----------------|-------|
| `max_completion_tokens` | Honored | Ignored | Ignored | Wins over `max_tokens` and a profile value |
| `tool_choice` | None/auto enforced | None/auto enforced | None/auto enforced | Other values depend on the template, as [Tool calling](#tool-calling) says |
| `output_config` | Ignored | Ignored | Honored | Anthropic `json_schema` format, mapped onto structured output |
| `logit_bias` | Honored | Honored | Honored | Keyed by token id |
| `seed` | Honored | Honored | Honored | Sampling seed for one request |
| `stream_options` | Honored | Ignored | Ignored | `include_usage` adds a final usage chunk |
| `timings_per_token` | Honored | Ignored | Ignored | Streamed chunks carry `timings.predicted_n`, the output-token count so far, as in llama.cpp |
| `response_format` | Honored | Honored | Honored | `json_schema` or `json_object`, as [Structured output](#structured-output) shows |
| `logprobs` | Honored | Ignored | Ignored | Token logprobs, as [Logprobs](#logprobs) shows |
| `top_logprobs` | Honored | Ignored | Ignored | Capped by `TOP_LOGPROBS_K` |
| `stop` | Honored | Ignored | Ignored | Anthropic clients use `stop_sequences` |
| `stop_sequences` | Ignored | Ignored | Honored | Anthropic spelling of `stop` |
| `chat_template_kwargs` | Honored | Honored | Honored | Template variables that override the profile's. `chat_template` itself gets a 400. |
| `profile` | Honored | Honored | Honored | A [profile](config.md#profiles) by name |
| `xtc_probability` | Honored | Honored | Honored | XTC sampling, with `xtc_threshold` |

### Structured output

`response_format` with a JSON schema constrains the output, so the model
cannot write tokens that break the schema:

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

`"json_object"` allows any JSON object. An unknown type or a malformed
schema gets a 400. On `/v1/messages`, an `output_config` of type
`json_schema` works the same way. A speculative model refuses structured
output and XTC sampling.

### Logprobs

`logprobs: true` returns each token's logprob, and `top_logprobs: N` asks
for the N most likely alternatives. The server caps the alternatives with
the `TOP_LOGPROBS_K` environment variable, 0 to 20, which is 0 by default.
Raise it when you start the server, or the lists stay empty:

```sh
TOP_LOGPROBS_K=5 gmlx serve --config ~/.config/gmlx/gmlx.yaml
```

### Media in requests

A model set up with `mmproj:` reads OpenAI `image_url` parts:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "gemma-e4b-vlm",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
  ]}]
}'
```

Send an image, audio clip or video in one of two ways:

- Inline, as a base64 `data:` URI, an Anthropic `base64` image source or
  base64 `input_audio`. This works everywhere, and it is the only form a
  [container session](container-security.md#what-the-client-reaches-on-the-server)
  can send.
- As a file in the server's media folder, `~/.cache/gmlx/media`, by its
  absolute path or a `file:` URL.

Any other path or URL gets a 400 that tells you how to copy the file into
the media folder. [`server.media_urls`](config.md#servermedia_urls) allows
URLs from public hosts. Each item holds at most 32 MiB. Images can be PNG,
JPEG, WebP, GIF, BMP or TIFF, and videos MP4, QuickTime, Matroska, WebM or
AVI.

## Limits and back-pressure

A model's context window comes from its GGUF.
[`max_kv_size`](config.md#loadmax_kv_size) can lower it, and a request
cannot change it.

| Condition | Response | Switch |
|-----------|----------|--------|
| The prompt plus `max_tokens` exceeds the context window | 400 with both token counts and the window | [`max_kv_size`](config.md#loadmax_kv_size) |
| The prompt does not fit in memory | 400 with the estimated need and the budget | `GMLX_PREFLIGHT_MEM=0` |
| More requests wait than the queue cap | 503 `server_overloaded`, with `Retry-After` of 2 to 60 seconds | `GMLX_QUEUE_DEPTH_CAP` |
| A model cannot load beside the pinned or busy models | 503 `model_load_deferred`, with `Retry-After` | `GMLX_OVERCOMMIT=1` |
| Memory runs out while a request streams | The [governor](glossary.md#governor) ends the largest request: `server_overloaded_shed`, `finish_reason` `shed` | `GMLX_GOVERNOR=0` |
| A body over 64 MiB, or an audio form over 1024 MiB | 413 before the body is read | None |
| From a [launch session](container-security.md#what-the-client-reaches-on-the-server), a body over 32 MiB, or 64 MiB for audio | 413 before the body is read | None |
| More than 64 media items, more than 256 Mi pixels in all, or audio over 128 Mi samples or with a sample rate over 384 kHz | 400 before decoding | None |

Both 400s for a prompt that does not fit start with `prompt is too long`,
so agent clients such as pi compact the conversation and retry.
[`POST /v1/estimate`](capacity.md#estimate-one-request) runs the same check
before you send. The switch variables are in
[server environment variables](env-vars.md#server), and `GMLX_GOVERNOR` is
in [runtime environment variables](env-vars.md#runtime).

## Hugging Face policy

A request never makes the server download a model. A `model` that is not a
configured id gets a 404. A load that would still need a download gets a
403 of type `hf_access_disabled`. To serve that model, add it to `models:`,
or set `server.hf_cache: true` so that its repo id resolves from the local
Hugging Face cache.

An `hf:` ref in `models:` also resolves from the local cache, never the
network, so you can serve a file that another tool downloaded. The service
models in the config download at start, as
[Speech, embeddings and rerank](services.md) describes.
