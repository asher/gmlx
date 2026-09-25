# HTTP API

This page is for anyone writing a client against `gmlx serve` or putting it
behind a load balancer. It covers the endpoints, how a request names a
model, the metrics a dispatcher reads, which request fields each protocol
honors, and what the server refuses. The YAML that configures the server is
in [server-config.md](server-config.md).

- [Addressing a model in a request](#addressing-a-model-in-a-request)
- [Endpoints](#endpoints)
- [Capacity and live-request metrics](#capacity-and-live-request-metrics)
- [API capabilities](#api-capabilities)
- [Structured decisions](#structured-decisions)
- [Limits and back-pressure](#limits-and-back-pressure)
- [Hugging Face policy](#hugging-face-policy)

Tool calling, structured output, logprobs and vision messages each have a
subsection under API capabilities.

## Addressing a model in a request

```jsonc
{"model": "qwen3.6-27b"}                                 // the model's configured profile, else the family base
{"model": "qwen3.6-27b@coding"}                          // a built-in intent, resolved for the model's family
{"model": "qwen3.6-27b@qwen-coder"}                      // an inline user profile, which overrides model.profile
{"model": "coder"}                                        // an alias, here for qwen3.6-27b@qwen-coder
{"model": "qwen3.6-27b", "profile": "coding",            // the profile field, via extra_body in the OpenAI SDK
 "temperature": 0.1}                                      //   plus an explicit field that overrides the profile
```

The `@profile` suffix is split on the last `@` and only treated as a profile
when it names a known one, so an id that itself contains an `@` stays
intact. An unknown id returns 404 listing the available ids, and an unknown
profile returns 400 listing the valid ones. With `model` empty, the server
uses its default model, else the sole model, else returns 400. The same
addressing works on the CLI, as `gmlx run <id-or-path>@coding` or
`--profile coding`.

## Endpoints

The standard generation and listing routes work. gmlx adds operational ones.
All routes except `/health` require the API key when one is set.

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | OpenAI chat completions, the primary route |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages. `/v1/messages/count_tokens` counts a body's input tokens |
| `POST /v1/completions` | classic text completions, with a single string prompt, a single choice and no chat template |
| `GET /v1/models` | configured ids and aliases with their markers, also at `/models` |
| `GET /health` | liveness. `?ready=1` adds a readiness verdict |
| `GET /v1/metrics` | the runtime snapshot, also at `/metrics` and as Prometheus text with `?format=prometheus` |
| `POST /v1/estimate` | dry-run admission for a chat body |
| `GET /v1/capacity/plan` | can `width` streams run at `depth` tokens each, and may they start now |
| `GET /v1/cache/stats` | prompt cache statistics, or `{"enabled": false}` |
| `POST /v1/cache/reset` | clear the prompt cache for all resident models, or one with `{"model": "<id>"}` |
| `POST /unload` | evict a resident model with `{"model": "<id>"}`, or all with an empty body. 409 while streams are in flight |
| `POST /v1/keep` | keep a model resident through the idle timeout with `{"model": "<id>", "warm": true}`. `"keep": false` releases it |
| `POST /v1/reload` | re-read the config and re-register models, keeping entries whose load parameters are unchanged. Same as SIGHUP |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | speech-to-text, with `stt` configured as in [services.md](services.md) |
| `POST /v1/audio/speech` | text-to-speech, with `tts` configured |
| `POST /v1/embeddings` | text embeddings, with `embeddings` configured |
| `POST /v1/rerank` | reranking, with `rerank` configured, also at `/rerank` |
| `POST /v1/systemone` | answers to a fixed question set about a state from a DiffusionGemma model, also at `/systemone` |

`GET /v1/models` lists configured and discovered ids plus alias presets.
Each entry carries `resident`, `pinned`, `speculative`, `vlm`, `profile` and
`default` markers and two context figures. `context_length` is the GGUF's
trained window, or the model's [`max_kv_size`](server-config.md#load-keys)
when that is smaller. `max_context_at_width_1` is how much of the window
fits in memory for a single stream. A harness sizes its context window from
the smaller of the two. A
resident model with KV quantization configured adds a `kv_quant` object:
`scheme`, `bits`, `group_size`, `layers_quantized`, `layers_fp16`, a
`verdict` of `full`, `partial`, `dropped` or `error`, and `verdict_batched`
for a speculative model, which differs from `verdict` under `uniform`, or
under kvarn on an mlx-kquant older than 0.4.9, where the model runs fp16 KV
while batched. Under [kvarn](glossary.md) it also carries `value_bits` and
`tail_tokens`. The
Hugging Face cache is never listed.

`GET /health` returns only `{"status": "healthy", "pid": N}`. Adding
`?ready=1` gives a coarse readiness verdict, either 200 with `"ready": true`
or 503 with a one-word `reason` and a `Retry-After` header. The reason is
`pressure` when the [governor](glossary.md) is orange or red, `queue` when
requests are waiting and `busy` when all engines are at their decode width.

`POST /v1/completions` supports `max_tokens`, `temperature`, `top_p`, `seed`,
`stop`, `stream` and `profile`. List or token-array prompts, `n > 1`, `echo`,
`suffix` and `best_of > 1` are rejected with a 400. mlx-vlm's image
generation routes are present but cannot serve a GGUF model.

The residency routes act on what
[Memory and residency](server-config.md#memory-and-residency) configures.
`/unload` outranks the hold a preloaded model has for the process lifetime,
so it unloads too, and until the next reload it is then managed like any
other model. `/v1/keep` is what `gmlx launch --model` and voice sessions
call, and a kept model stays LRU-evictable under memory pressure.
`/v1/reload` returns `{"status": "unsupported"}` outside config mode, as
[Reloading the config](server-config.md#reloading-the-config) explains.

## Capacity and live-request metrics

`GET /v1/metrics` carries, under `server`, what a load balancer or a harness
that fans out subagents needs to size its work. The sections are
independent, so a probe failure inside one leaves its live fields `null`
instead of failing the snapshot.

| Section | Fields | Meaning |
|---|---|---|
| `concurrency` | `decode_batch`, `queue_cap`, `in_flight`, `waiting` | the decode width, the queue cap, streams generating now and requests waiting for a slot |
| `queue` | `waiting`, `cap`, `eta_s`, `rejections`, `last_reject_reason` | the waiting count, the cap it is judged against and the drain estimate a client would receive as `Retry-After` now |
| `requests[]` | one row for each request | queued rows first. `state` is `queued`, `prefill` or `decode`, and the other fields are listed below the table |
| `resident_models[]` | for each model, `in_flight`, `pinned`, `kept` and bytes | the number for each model to compare against `decode_batch`, since each model decodes on a separate engine |
| `governor` | `band`, counters | the memory governor's band and shed history |
| `memory` | `active_bytes`, `cache_bytes`, `headroom_bytes`, arena fields | MLX's active and cached bytes, the free memory the admission gate reads, and for a streamed model the [arena's](glossary.md) bytes, capacity and hit rate |
| `capacity` | `max_ctx` by width, `max_width_at_depth`, byte budgets | the boot capacity table, absent for a Hugging Face fall-through load |
| `rates` | `decode_tok_s`, `decode_streams`, `prefill_tok_s_recent`, `decode_tok_s_recent`, `decode_tok_s_lifetime` | the aggregate decode rate now and its stream count, the recent means over the last eight requests, and the lifetime mean |

A request row carries `id`, `model`, `state`, `position`, `prompt_tokens`,
`generated`, `max_tokens`, `elapsed_s`, `ttft_s` and `decode_tok_s`, and
two structured fields. `cache` holds the tier its prefix hit, one of
`exact`, `block`, `ckpt`, `anchor` and `miss`, or `hit` when only the reused
token count is known, plus the `warm_tokens` it reused. `speculative` holds
the [drafter's](glossary.md) rounds, drafted and accepted counts and accept
rate, exact at batch width 1 and shared across a wider batch, or `null`
without a drafter. Rows refresh at most four times a second on each engine.

Two routes use the same numbers to tell a dispatcher whether to proceed before
sending a request.

`POST /v1/estimate` takes a chat-completions body and returns an admission
estimate for a resident model. `prompt_tokens`, `warm_tokens` and `cache_tier`
say how much of the prefix the cache already holds and on which tier, which is
the routing signal across machines. `need_bytes` is the prompt's KV plus the
prefill transient, plus `max_tokens` when the body pins one. `fits_now` and
`fits_drained` judge that against the current free memory and the drained
working set, while `context_ok` judges it against `context_limit`.
`est_ttft_s` estimates the time to first token. The dry run never loads a
model, so a model that is not resident answers `resident: false`. A media
request is rendered but not estimated. `"dry_run": true` on
`/v1/chat/completions` returns the same estimate through the queue cap.

`GET /v1/capacity/plan?width=W&depth=D` answers `ok` when the capacity
table holds `W` streams at `D` tokens each, reading the table conservatively
at the smallest tabulated width at or above `W`. It answers `admit_now` when
in addition the governor is not orange or red, nothing is waiting and at
least `W` decode slots are free. Under a yellow band only one free slot is
counted, so a wider fan-out waits for the band to clear. `reason` names the
first condition that fails.

The Prometheus rendering flattens these to gauges such as
`gmlx_concurrency_in_flight`, `gmlx_queue_eta_s`, `gmlx_governor_band` with a
`band` label, `gmlx_capacity_max_ctx` with a `width` label and per-model
series with a `model` label, plus a `profile` label on entries that carry a
profile. `requests[]` is high-cardinality and contributes only its count.

## API capabilities

The protocol surface follows mlx-vlm's, with gmlx patches on the handlers
(around twenty route-level patches, including the added `/v1/completions`
route), so upstream request features work on GGUF models. Context windows
come from the GGUF's metadata, with no server-side override or
request-level context setting.

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
| `none` | enforced. The tools are stripped before the template runs, so the model cannot emit a call |
| `auto` | the default. The model decides |
| `required` and named-function forms | forwarded to the template as a variable and honored only if the template implements them. The server logs a warning when a forced call produced no call |

This is the client-side loop, where the server parses the calls and the
client executes them and sends results back. To run the loop server-side
with config-allowlisted MCP tools, serve an
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
present. An honored parameter changes the response. An ignored one is
accepted and skipped, and a request that sets any produces one warning line
in the server log naming them all. `/v1/systemone` has its own fields,
listed under [Structured decisions](#structured-decisions).

The standard sampling parameters are honored on all three generation dialects.
They are `max_tokens` and `max_output_tokens`, `temperature`, `top_p`,
`top_k`, `min_p`, `top_n_sigma`, `p_less`, `typical_p`, `repetition_penalty`,
`presence_penalty`, `frequency_penalty` and their `*_context_size` companions,
`enable_thinking`, `thinking_budget` and the OpenAI `reasoning` and
`reasoning_effort` controls. The rest:

| Parameter | `/v1/chat/completions` | `/v1/responses` | `/v1/messages` | Notes |
|-----------|------------------------|-----------------|----------------|-------|
| `max_completion_tokens` | honored | ignored | ignored | OpenAI's current name for the chat output cap. It wins over `max_tokens` and a profile value |
| `n` | ignored | ignored | ignored | always a single choice. `n > 1` on `/v1/completions` is a 400 |
| `user` | ignored | ignored | ignored | no per-user accounting |
| `parallel_tool_calls` | ignored | ignored | ignored | the template decides how many calls to emit |
| `tool_choice` | none/auto enforced | template-dependent | none/auto enforced | `required` and named forms are template-dependent, as the previous table says |
| `metadata` | ignored | ignored | ignored | accepted for Anthropic compatibility, never read |
| `output_config` | ignored | ignored | honored | Anthropic `json_schema` format maps onto structured output |
| `logit_bias` | honored | honored | honored | token-id keyed |
| `seed` | honored | honored | honored | per-request sampling seed |
| `presence_penalty` | honored | honored | honored | |
| `frequency_penalty` | honored | honored | honored | |
| `stream_options` | honored | ignored | ignored | `include_usage` adds the final usage chunk on chat and `/v1/completions` |
| `timings_per_token` | honored | ignored | ignored | streamed chat chunks carry `timings.predicted_n`, the exact cumulative output-token count, following llama.cpp |
| `response_format` | honored | honored | honored | `json_schema` or `json_object`. Unknown types are rejected, as Structured output explains |
| `logprobs` | honored | ignored | ignored | chat only. `/v1/completions` never returns logprobs |
| `top_logprobs` | honored | ignored | ignored | capped by `TOP_LOGPROBS_K`, as Logprobs explains |
| `stop` | honored | ignored | ignored | chat and `/v1/completions`. Anthropic uses `stop_sequences` |
| `stop_sequences` | ignored | ignored | honored | the Anthropic-native spelling |
| `chat_template_kwargs` | honored | honored | honored | extra template variables, request overrides profile |
| `profile` | honored | honored | honored | a sampling and system [profile](server-config.md#profiles) by name |
| `xtc_probability` | honored | honored | honored | XTC sampling, with `xtc_threshold` |

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
asks for the N most likely alternatives for each token. The alternatives are
capped by the server-side `TOP_LOGPROBS_K` variable, 0 to 20, with a default
of 0, so until the server is started with the cap raised the lists stay
empty. It is an engine variable with no config key:

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

## Structured decisions

`POST /v1/systemone` answers a fixed set of questions about a state with a
DiffusionGemma model, and returns a probability distribution for each
question instead of generated text. The request and answer shapes are those
of the Jev decision API, so a client written for that service works against
the server. The route is also at `/systemone`.

The questions become the system prompt and the state becomes the user
message. The model's canvas is seeded with an answer template of one line
per question, and one denoise step gives the distribution over each
question's labels. That is a [structured read](glossary.md), and
[structured-reads.md](internals/structured-reads.md) describes the
mechanism. A decision is deterministic for a given request and `seed`.

### Request body

| Field | Default | Meaning |
|-------|---------|---------|
| `model` | the fallback | a configured id, alias or `id@profile`. A name that resolves to nothing, or no name, takes the fallback below |
| `state` | required | what the questions are about, as a string or any JSON value, which is sent as its JSON text |
| `questions` | required | a map of question id to question, with the fields in the next table |
| `seed` | `42` | the base seed for the random tokens placed at the label positions. Each sample offsets it |
| `profile` | none | the [profile](server-config.md#profiles) to resolve `model` with |
| `images` | none | refused with a 400, since the model is text-only |

The fallback is [`server.systemone.model`](server-config.md#structured-decisions)
when set, else the default model, else the only configured model. Jev
clients send `"model": "jev-latest"`, which reaches the fallback this way. A
name that resolves to nothing when no fallback exists gets the usual 404.

A question has these fields. An id may not contain a colon or a newline.
Past ten questions the answer template writes each label directly after
its id, so ids such as `q1` work there and a word id can be refused with a
422.

| Field | Default | Meaning |
|-------|---------|---------|
| `type` | required | `noul` for yes or no, `choice` for one of several options, or `score` for one of ordered levels |
| `instructions` | empty | the question as the model reads it |
| `criteria` | none | `noul`: an object with `true` and `false` descriptions. `choice`: option name to description. `score`: a list of level names in order |
| `depends_on` | none | question ids answered in an earlier stage, whose answers this question's read sees |
| `ask_if` | none | question id to a list of its answers. The question is asked only when that answer is among them, else it answers `null` |
| `alone` | `false` | read this question in a read of its own |

A choice or score takes 2 to 26 alternatives. The request may also carry
these fields, which the vLLM implementation of the route defines:

| Field | Default | Meaning |
|-------|---------|---------|
| `instructions` | none | text added to the system prompt ahead of the questions |
| `samples` | `"auto"` | how many reads with different random label tokens are averaged. `"auto"` reads once and adds more when an answer is uncertain |
| `auto_max` | `4` | the sample count `"auto"` extends to |
| `auto_threshold` | `0.1` | the entropy in nats at a label position above which `"auto"` extends |
| `steps` | `1` | denoise steps per read, clamped to 1 through 8 |
| `think` | `0` | a thought budget in tokens, 0 to 4096. The model writes a thought first, and the reads see it |
| `ask` | every question | the ids to answer, which must include everything they depend on |
| `chunk_rows` | the canvas | the most canvas tokens one read's answer template may take, at least 8. A larger stage is split into chunks |
| `chunk_prompt` | `"own"` | `"own"` gives each chunk a system prompt with its questions only, and `"shared"` gives every chunk all of them |
| `sequential` | `false` | read the chunks in order on one prompt, each seeing the answers before it |

A request with more questions than `server.systemone.max_questions` gets
a 422, and `samples` and `auto_max` are lowered to
`server.systemone.max_samples`.

### Answers

`answers` maps each question id to one of these shapes, or to `null` for a
question that `ask_if` skipped:

```jsonc
{"type": "noul", "noul": 0.93}                  // the probability of yes
{"type": "choice", "choice": "infra",
 "probabilities": {"infra": 0.81, "billing": 0.19}, "confidence": 0.81}
{"type": "score", "score": 1.72, "legend": {"0": "low", "1": "mid", "2": "high"},
 "probabilities": {"0": 0.07, "1": 0.14, "2": 0.79}, "confidence": 0.79}
```

`confidence` is the probability of the chosen option or level. A score is
the expected level index, counted from 0, and `legend` names each index.
`usage.input_tokens` is the longest prompt a read ran on, and
`usage.output_tokens` counts the answer template tokens and the thought
tokens. `diagnostics` shows how the decision ran: the `stages`, `chunks`
and `skipped` questions, the `thought`, each sample's label logprobs under
`samples.tops`, the entropies under `questions`, and `timing`.

### Status codes

| Status | When |
|--------|------|
| 400 | the body is not a JSON object, it carries images, the model is not DiffusionGemma, or the prompt does not fit the context or memory budget |
| 404 | `model` names nothing and no fallback exists |
| 422 | a question, `state` or `seed` is invalid. The error type is `validation_error` |
| 503 | the queue cap or a deferred load, as under [Limits and back-pressure](#limits-and-back-pressure) |
| 504 | the decision ran past [`token_queue_timeout_s`](server-config.md#scheduling). The error type is `timeout` |
| 500 | the engine failed. The error type is `server_error` |

A decision holds the model from its first read to its last, so a chat
request to the same model waits behind it, as it waits behind any
DiffusionGemma generation. The context budget is checked against the
largest prompt the decision can build before the decision is queued.

```sh
curl localhost:8080/v1/systemone -d '{
  "model": "jev-latest",
  "state": {"ticket": "Everything is down and we have a demo at noon."},
  "questions": {"urgent": {"type": "noul",
    "instructions": "Does the customer need a reply within the hour?"}}
}'
```

## Limits and back-pressure

| Condition | Response | Switch |
|-----------|----------|--------|
| the prompt plus `max_tokens` exceeds the context budget | 400 with both token counts and the budget | [`max_kv_size`](server-config.md#load-keys) |
| the prompt alone cannot fit in memory | 400 with the estimated need and the available budget | `GMLX_PREFLIGHT_MEM=0` |
| more requests waiting than the queue cap | 503 with `Retry-After` set to the estimated drain time, 2 to 60 seconds | `GMLX_QUEUE_DEPTH_CAP` |
| a model cannot be loaded beside what is resident and busy | 503 of type `model_load_deferred`, with the gate's numbers in the message and `Retry-After` | |
| a streaming request is silent, as during a long prefill | an SSE comment line every 15 seconds so read timeouts do not drop the connection | `GMLX_SSE_KEEPALIVE_S` |

The preflight estimates the prompt's KV cache from the model's per-token KV
size, plus the prefill transient, against the working set with the batch
drained. `max_tokens` counts only when the request pins it
explicitly, and media requests are not estimated. The load gate judges a
model's weights against what is resident and busy and against the
governor's floor, after waiting up to 3 seconds for memory the kernel is
still returning from a recent unload. The environment variables in the last
column, and the decode batch width the queue cap is derived from, are
documented under [Server](env-vars.md#server).

Both 400s for a request that does not fit start their message with `prompt
is too long`, and the budget error also says the request `exceeds the context
window`. Agent clients that recognize those words, pi among them, compact the
conversation and retry instead of stopping at the error.

## Hugging Face policy

The server never downloads on a request. A generation request whose
`model` is not a configured id gets the 404 described under Addressing a
model, whatever the id looks like. Below that resolver, the stock model
loader is gated as well: anything that reaches it with a repo id rather
than a GGUF or local path is refused with a 403 of type
`hf_access_disabled`, so no route can trigger a fetch.

`server.hf_cache: true` changes what the config may reference, not what a
request may name. With it on, `hf:` refs in `models:` and the `gmlx init
--from-hf-cache` scan resolve from the local Hugging Face cache, still never
the network, so a repo that was downloaded by another tool can be served
without copying the file.
