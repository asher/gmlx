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
  - [Adapters](#adapters)
  - [Thinking budget](#thinking-budget)
  - [Structured output](#structured-output)
  - [Logprobs](#logprobs)
  - [Text completions](#text-completions)
  - [Tokens](#tokens)
  - [Assistant prefill](#assistant-prefill)
  - [Media in requests](#media-in-requests)
  - [Word timestamps in chat](#word-timestamps-in-chat)
  - [Transcription and translation](#transcription-and-translation)
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
gets a 400 of type `unknown_profile`. A chat route that names one of the
[image models](config.md#image-models) gets a 400 of type
`wrong_model_kind`, and so does an image route that names a chat model.

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | OpenAI chat completions, the main route |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages. `/v1/messages/count_tokens` counts a body's input tokens. |
| `POST /v1/completions` | Text completion with no chat template: string or token-id prompts, lists of prompts, echo and logprobs, as [Text completions](#text-completions) shows |
| `POST /tokenize`, `/detokenize` | A prompt or chat messages as token ids, and token ids as text, as [Tokens](#tokens) shows |
| `GET /v1/models` | Configured ids and aliases, with markers and [context figures](capacity.md#context-windows-in-the-model-list) |
| `GET /health` | Liveness, with no API key. `?ready=1` adds readiness, as [Capacity and metrics](capacity.md#readiness-checks) shows. |
| `GET /v1/metrics` | Live snapshot, or Prometheus text with `?format=prometheus` |
| `POST /v1/estimate` | Dry-run admission check for a chat body |
| `GET /v1/capacity/plan` | Whether `width` streams of `depth` tokens fit, and can start now |
| `GET /v1/cache/stats` | Prompt cache statistics, or `{"enabled": false}` |
| `POST /v1/cache/reset` | Clear the prompt cache, for every model or for `{"model": "<id>"}` |
| `POST /unload` | Unload `{"model": "<id>"}`, a chat or an image model, or every idle model with no body. 409 while that model streams. |
| `POST /v1/keep` | Load and warm `{"model": "<id>"}`, a chat or an image model, and keep it past its idle timeout. `"warm": false` skips the load. `"keep": false` releases it. |
| `POST /v1/reload` | Re-read the config file and keep models whose load settings did not change |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | Speech to text, with `stt` set up as in [Speech, embeddings and rerank](services.md), or a served model that hears audio |
| `POST /v1/audio/speech` | Text to speech, with `tts` set up, or a served model that speaks. `GET /v1/audio/voices` lists the voices. |
| `POST /v1/embeddings` | Text embeddings, with `embeddings` set up |
| `POST /v1/rerank` | Document reranking, with `rerank` set up |
| `POST /v1/images/generations`, `/v1/images/edits`, `/v1/images/variations` | Image generation, with [image models](config.md#image-models) set up, as the [Image API](image-api.md) shows |
| `POST /v1/videos`, `GET` and `DELETE /v1/videos/{id}`, `GET /v1/videos/{id}/content` | Video jobs, with [video models](config.md#video-models) set up, as the [Video API](video-api.md) shows. These answer only with the `/v1` prefix. |
| `POST /v1/systemone` | Probabilities for a fixed set of questions, as [Structured decisions](decisions.md) shows |
| `POST /v1/prewarm` | Read a decision state ahead of its questions, as [Repeated states](decisions-reference.md#repeated-states) shows |

[Capacity and metrics](capacity.md) covers `/v1/metrics`, `/v1/estimate`
and `/v1/capacity/plan`.

`GET /v1/models` lists the configured ids, discovered ids and aliases, never
the Hugging Face cache. Each entry carries the markers `resident`, `pinned`,
`speculative`, `vlm`, `profile` and `default`. A chat entry also lists its
`adapters`, each with the `index`, file `name`, default `strength` and the
`sha256` of the loaded file, as [Adapters](#adapters) describes.

A chat entry's `modalities` names what the model reads and writes: `text`,
`vision`, `audio` for audio input, and `speech` for speech output. An
entry with `speech` also lists the model's `voices`, the default first.

Each [image model](config.md#image-models) is listed too, with the fields
that [Routes](image-api.md#routes) describes.

`/v1/keep` answers `{"status": "kept", "model": "<id>", "warming": true}`
and loads the model in the background. A kept model is not pinned, so the
server can still unload it to make room, as
[Keep, pin and idle](glossary.md#keep-pin-and-idle) describes.

Every route except `/health` and the image files of
[URL responses](image-api.md#url-responses) needs the API key when the server
has one. Most routes also answer without the `/v1` prefix. Every route takes
a JSON body, except the two audio upload routes and image variations, which
take a form, and image edits and `POST /v1/videos`, which take a form or a
JSON body.

[Text completions](#text-completions) covers `/v1/completions`.

## Request features

### Tool calling

`/v1/chat/completions` takes OpenAI `tools` and answers with `tool_calls`.
`/v1/messages` takes Anthropic `tools` and answers with `tool_use` blocks.
The server reads the call syntax from the model's chat template, so there
is nothing to set up. A request can name the parser with `tool_parser`
instead. The client runs the tools and sends the results back.

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
`thinking_budget`, `thinking_budget_message`, llama-server's
`reasoning_budget_tokens` and `reasoning_budget_message`, and the OpenAI
`reasoning` and `reasoning_effort` controls.
[Thinking budget](#thinking-budget) shows the budget fields. The three chat
routes ignore `n`, `user`, `parallel_tool_calls` and `metadata`: they return
one choice, and the template decides how many tool calls to make.
`presence_penalty` and `frequency_penalty` count every token generated so
far and never the prompt, as vLLM and SGLang do. `presence_context_size` or
`frequency_context_size` set to N counts only the last N generated tokens,
and 0 or no value counts all of them. The other fields differ by route:

| Parameter | `/v1/chat/completions` | `/v1/responses` | `/v1/messages` | Notes |
|-----------|------------------------|-----------------|----------------|-------|
| `max_completion_tokens` | Honored | Ignored | Ignored | Wins over `max_tokens` and a profile value |
| `tool_choice` | None/auto enforced | None/auto enforced | None/auto enforced | Other values depend on the template, as [Tool calling](#tool-calling) says |
| `tool_parser` | Honored | Honored | Honored | Names the tool-call parser in place of template inference. An unknown name fails the request. |
| `output_config` | Ignored | Ignored | Honored | Anthropic `json_schema` format, mapped onto structured output |
| `logit_bias` | Honored | Honored | Honored | Keyed by token id |
| `seed` | Honored | Honored | Honored | Sampling seed for one request |
| `stream_options` | Honored | Ignored | Ignored | `include_usage` adds a final usage chunk |
| `timings_per_token` | Honored | Ignored | Ignored | Streamed chunks carry `timings.predicted_n`, the output-token count so far, as in llama.cpp |
| `response_format` | Honored | Honored | Honored | `json_schema` or `json_object`, as [Structured output](#structured-output) shows |
| `logprobs` | Honored | Ignored | Ignored | Token logprobs, as [Logprobs](#logprobs) shows |
| `top_logprobs` | Honored | Ignored | Ignored | 0 to 100, capped by `TOP_LOGPROBS_K` |
| `return_tokens_as_token_ids` | Honored | Ignored | Ignored | Names each logprob token as `token_id:<id>` |
| `return_token_ids` | Honored | Ignored | Ignored | Returns the prompt's token ids and the reply's, as [Tokens](#tokens) shows |
| `continue_final_message` | Honored | Ignored | Ignored | Continues the last assistant message, as [Assistant prefill](#assistant-prefill) shows |
| `add_generation_prompt` | Honored | Ignored | Ignored | `false` renders the messages with no generation prompt |
| `stop` | Honored | Ignored | Ignored | Anthropic clients use `stop_sequences` |
| `stop_sequences` | Ignored | Ignored | Honored | Anthropic spelling of `stop` |
| `chat_template_kwargs` | Honored | Honored | Honored | Template variables that override the profile's. `chat_template` itself gets a 400. |
| `profile` | Honored | Honored | Honored | A [profile](config.md#profiles) by name |
| `xtc_probability` | Honored | Honored | Honored | XTC sampling, with `xtc_threshold` |
| `adapters` | Honored | Honored | Honored | Adapter strengths for the request, by path or file name, as [Adapters](#adapters) shows |
| `lora` | Honored | Honored | Honored | Adapter strengths by index, in the llama-server form, as [Adapters](#adapters) shows |

`/v1/completions` honors the sampling parameters, `seed`, `profile`,
`adapters` and `lora`, and these fields:

| Parameter | Notes |
|-----------|-------|
| `logprobs` | An integer: the alternatives at each position. Above `TOP_LOGPROBS_K`, a 400. `true` gets a 400. |
| `echo` | Puts the prompt before the completion. With `logprobs`, also scores the prompt, as [Text completions](#text-completions) shows. |
| `return_tokens_as_token_ids` | Names each logprob token as `token_id:<id>` |
| `add_special_tokens` | Honored. Default true. A string prompt is tokenized with the flag; a token-id prompt ignores it. |
| `stop` | Trims the text at the first stop string. The logprob entries keep every generated token. |
| `stream_options` | `include_usage` adds a final usage chunk |

### Adapters

A model entry with [`adapters`](config.md#modelsadapters) takes a strength
for each of them in each request. `adapters` names an adapter by its path
in the config or its file name, and the adapters it does not list keep
their default. `lora` names an adapter by its index in `/v1/models`, and
the adapters it does not list run at 0:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3-0.6b-mix",
  "lora": [{"id": 1, "scale": 0.5}],
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

The chat, completions, Responses and Messages routes take both fields, and
so do `/v1/systemone` and `/v1/prewarm`. A request that sets both, or names
an adapter its model does not load, gets a 400.
[Mixing adapters per request](lora.md#mixing-adapters-per-request) lists the
rules.

### Thinking budget

`thinking_budget` caps the reasoning tokens of one request. When the budget
runs out, the server closes the thinking block, and the model writes its
answer. `thinking_budget_message` sets text that the server puts in the
thinking block before the close, so the model reads that it must answer now:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.8-27b-ud-q6",
  "messages": [{"role": "user", "content": "Plan three days in Kyoto."}],
  "thinking_budget": 256,
  "thinking_budget_message": "Time is up, so I will answer now."
}'
```

| Field | Default | Meaning |
|-------|---------|---------|
| `thinking_budget` | No budget | Reasoning tokens the model may write. `0` closes the block at once, and `-1` takes the server's budget. |
| `thinking_budget_message` | None | Text the server writes in the thinking block, then the end marker, when the budget runs out |
| `reasoning_budget_tokens` | None | llama-server's name for `thinking_budget` |
| `reasoning_budget_message` | None | llama-server's name for `thinking_budget_message` |

The fields work on `/v1/chat/completions`, `/v1/responses`, `/v1/messages`
and `/v1/completions`. On `/v1/messages`, `thinking.budget_tokens` also
sets the budget. A request that sets both names of one field, such as
`thinking_budget` and `reasoning_budget_tokens`, gets a 400.

A budget of N lets N reasoning tokens through, as llama-server does.
Reasoning that the prompt already holds after its last start marker counts
too. `-1` takes the budget of the model's profile or overrides, of
`gmlx serve --thinking-budget`, or of `MLX_VLM_THINKING_BUDGET`, in that
order.

Special tokens in the message are read as tokens. Without a message, the
server writes a newline before the end marker, and on a model with a
built-in drafter a budget above 0 writes a short sentence that says the
budget ran out. The message and the end marker are part of
`reasoning_content`, and they count against `max_tokens`.

The built-in drafters, the MTP heads, DFlash, DFlash2, DSpark and
LiLiCorr, honor the budget and the message on a request that runs alone.
The close can come up to one draft round late. A request that runs in a
batch with other requests, or that is paused and resumed, drops both. A
drafter that `MLX_VLM_DRAFT_MODEL` loads with a draft kind other than `mtp`
answers a request's budget with a 400.

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
`json_schema` works the same way. On a thinking model the schema applies to
the answer after the thinking. When the prompt neither opens nor closes a
thinking block, the reply can open one first, and the schema applies after
the block ends.

A model with MTP speculative decoding keeps speculating under a schema, and
its output is the same as without speculation. The first structured request
on a model takes about 1.5 s longer, once per process.

### Logprobs

`logprobs: true` returns the logprob of each token in the reply, and
`top_logprobs: N` asks for the N most likely alternatives at each position.
The server caps the alternatives with the `TOP_LOGPROBS_K` environment
variable, 0 to 100, which is 20 by default. Chat clamps a larger request to
the cap, and `/v1/completions` answers 400. To allow more, raise the cap
when you start the server:

```sh
TOP_LOGPROBS_K=50 gmlx serve --config ~/.config/gmlx/gmlx.yaml
```

The values are the model's own distribution: the log-softmax of its raw
logits. Logit bias, penalties, XTC, structured output and the profile's
sampling values change which token the server samples, but not the
logprobs. A token that the thinking budget forces in reports the logprob
that the model gave it. The token that ends the reply gets its entry too.
A logprob below -9999 comes back as -9999.0.

`bytes` holds the token's raw bytes from the vocabulary. A multi-byte
character can split across tokens, and each of those tokens returns only
its part. A special token, such as the token that ends the turn, returns
the UTF-8 of its token text, for example `<|im_end|>`.
`return_tokens_as_token_ids: true` replaces each `token` string, in the
entries and in their alternatives, with `token_id:<id>`. The `bytes` do not
change.

With `logprobs: true` and no `top_logprobs`, a reply has no alternatives, as
in the OpenAI API.

A model with a [speculative drafter](speculative-decoding.md), such as an
MTP head or a DFlash, DSpark or LiLiCorr drafter, returns logprobs while it
speculates. They come from the target model's pass over the drafted tokens,
and agree with a run without speculation to within bf16 rounding. A
diffusion model refuses a logprob request with a 400.

While a request with logprobs is in a batch, each step of that batch takes
longer, because the server also computes the alternatives.

`/v1/completions` also scores a prompt: with `echo` and `logprobs`, it
returns the logprob of each prompt token given the tokens before it, as
[Text completions](#text-completions) shows. A scored value is a prefill
value. It is close to the logprob of the same token while the model
generated it, but not equal: the two passes round differently in bf16.

### Text completions

`/v1/completions` sends a prompt to the model with no chat template. A
prompt is a string, a list of strings, a list of token ids, or a list of
lists of token ids. Each prompt of a list gets its own choice, with its
`index`, and `usage` adds up the prompts. A list holds at most as many
prompts as the queue cap, `GMLX_QUEUE_DEPTH_CAP`, or 64 when the cap is
off.

The server tokenizes a string prompt with the model's tokenizer and adds
the model's start token when the model has one. `add_special_tokens: false`
leaves the start token out. A token-id prompt goes to the model as sent,
with no start token. Each id must be in the vocabulary and must not be a
media placeholder, or the request gets a 400.

`logprobs: N` returns the OpenAI completions shape: `tokens`,
`token_logprobs`, `top_logprobs` and `text_offset`. Each `top_logprobs`
entry holds the N most likely tokens and the chosen token, so it can hold
N + 1 keys. Two tokens with the same text share one key, so an entry can
also hold fewer. `return_tokens_as_token_ids: true` names each token
`token_id:<id>`, which keeps every key apart. `text_offset` adds up the
length of each token's name.

`echo: true` puts the prompt before the completion. A string prompt comes
back as sent, and a token-id prompt as the tokenizer decodes it, special
tokens included.

`echo` with `logprobs` scores the prompt. The entries start with the
prompt tokens: the first has `null` in `token_logprobs` and `top_logprobs`,
and entry i holds the logprob of prompt token i given the tokens before
it. `max_tokens: 0` returns the prompt entries alone, with
`finish_reason: "length"` and 0 completion tokens. `max_tokens: 0` needs
`echo`. With `max_tokens` of 1 or more, the completion follows as usual,
and the server reads the prompt twice: once to score it and once to
generate.

```sh
curl localhost:8080/v1/completions -d '{
  "model": "qwen3.8-27b-ud-q6",
  "prompt": "The capital of France is Paris.",
  "echo": true, "logprobs": 0, "max_tokens": 0
}'
```

A scoring request:

- Does not stream. `stream` with `echo` and `logprobs` gets a 400.
- Holds at most 1,048,576 entries: the prompt tokens times `logprobs` + 1.
  A larger request gets a 400.
- Runs on a cache of its own at full precision, whatever `KV_BITS` sets.
  It does not read or fill the [prompt cache](prompt-cache.md).
- Runs one at a time on each model, between the steps of the chat
  requests. The others wait in the order they came, and count toward the
  queue cap.
- Gets a 400 on a model that reads token ids and embeddings differently,
  or that does not return logits for each prompt token. The message names
  the model.

The route refuses `n` and `best_of` above 1, `suffix`, a list of prompts
with `stream`, and `logprobs: true`. vLLM's `prompt_logprobs` is ignored,
and the server log names it.

### Tokens

Chat `return_token_ids: true` returns the token ids the model read and
wrote: `prompt_token_ids` at the top level, and `token_ids` in each choice.
With `logprobs`, each id has one logprob entry, in the same order. A
stream sends `prompt_token_ids` in its first chunk and `token_ids` in each
chunk. The prompt ids of a request with images, audio or video include
media placeholders, which a token-id prompt cannot send back. A diffusion
model refuses the field with a 400.

`POST /tokenize` takes vLLM's body and returns
`{"tokens", "count", "max_model_len"}`:

- `prompt`, a string, tokenizes as `/v1/completions` tokenizes it, with
  `add_special_tokens` (default true).
- `messages` renders and tokenizes as `/v1/chat/completions` does, with
  `add_generation_prompt`, `continue_final_message`, `chat_template_kwargs`
  and `tools`. Messages with media get a 400.
- `return_token_strs: true` adds `token_strs`, each token's vocabulary
  entry.

`max_model_len` is the configured context limit, or the model's trained
context. `POST /detokenize` takes `{"tokens": [ids]}` and returns
`{"prompt": text}`.

To score a reply token by token at its exact tokens, generate it on chat
with `return_token_ids: true`. Keep `prompt_token_ids` as P and the
choice's `token_ids` as R. For a reply that gmlx did not generate,
`/tokenize` with the `messages` gives P. Then score P and R in one request:

```sh
curl localhost:8080/v1/completions -d '{
  "model": "qwen3.8-27b-ud-q6",
  "prompt": [P..., R...],
  "echo": true, "logprobs": 0, "max_tokens": 0,
  "return_tokens_as_token_ids": true
}'
```

`token_logprobs[len(P) + i]` is the logprob of `R[i]` given P and the reply
tokens before it. The model reads the ids it generated, not a new
tokenization of the reply's text. With `KV_BITS`, the reply was generated
on a quantized cache and the score runs on a full-precision one, so the
two values can differ more than bf16 rounding.

### Assistant prefill

To have the model continue a reply that you started, end `messages` with
the partial assistant message. Then send `continue_final_message: true`
with `add_generation_prompt: false`, as for llama-server:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.8-27b-ud-q6",
  "messages": [{"role": "user", "content": "Name three rivers."},
               {"role": "assistant", "content": "1. The Nile\n2."}],
  "continue_final_message": true,
  "add_generation_prompt": false
}'
```

The reply holds only the new text, in `content` or `reasoning_content`.
`usage.prompt_tokens` counts the prefill, and `max_tokens`, `logprobs` and
`stop` apply to the new tokens only.

A continued message is text, and the server tokenizes it again. Its tokens
can differ from the ones the model generated, for example at a run of
spaces or newlines. To continue from exact tokens, send them as a token-id
prompt, as [Tokens](#tokens) shows.

| Field | Default | Meaning |
|-------|---------|---------|
| `continue_final_message` | `false` | `true` continues the content, or the reasoning when the message has only `reasoning_content`. A part name picks it. |
| `add_generation_prompt` | `true` | `false` renders the messages with no generation prompt. Send it with `continue_final_message`. |

A part name is `"content"` or `"reasoning_content"`. `"content"` closes the
reasoning before the content. `add_generation_prompt: false` alone renders
the messages as they are, so a last assistant message stays a finished
turn.

These chat templates can continue a message, in the prompt that
llama-server renders for the same messages:

- Qwen3.5 and later Qwen templates, and the other templates that write tool
  calls in the same `<tool_call>` and `<function=...>` form.
- Gemma 4.
- SmolLM2-135M-Instruct.

A request to any other model gets a 400. So does a last message with
`tool_calls`, or `response_format` with `continue_final_message`. An
[assistant](assistant.md#served-assistants) without `tools` refuses both
fields, because its turns build their own prompt.

[Assistant prefill](migrating.md#assistant-prefill) in the migration guide
lists how gmlx differs from llama-server here.

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
AVI. The server reads videos with ffmpeg, which picks and decodes frames as
the models' reference code does, or with OpenCV from the
[`video` extra](installation.md#optional-features) when ffmpeg is missing.

A text model that reads audio through a speech companion, such as
[Granite Switch](granite-switch.md), takes `input_audio` parts on
`/v1/chat/completions` and `/v1/responses`. The server transcribes each
clip and puts the transcript where the clip was in the message. Without a
companion, a request with audio gets a 400. `/v1/estimate` reports such a
request as `media: true`, because the transcript length is unknown until
the clip is transcribed.

A model that hears audio through its own speech encoder, such as
[Nemotron 3.5 Transcribe](nemotron-transcribe.md), takes `input_audio`
parts on `/v1/chat/completions` and `/v1/responses`, up to six clips and
3600 seconds of audio in one request. The model reads each clip where its
message put it.

Other models with an audio tower take audio on `/v1/chat/completions`
only. `/v1/responses` answers a request with audio for them with a 400.

A model with speech output, such as [LFM3-Omni-2B](lfm3.md), returns
audio on `/v1/chat/completions` when the request sets
`"modalities": ["text", "audio"]`, as
[Speech on the server](lfm3.md#speech-on-the-server) shows. On any other
model, such a request gets a 400.

### Word timestamps in chat

A chat completion on a model that hears audio through its own speech
encoder can return the time of each word of its reply. Set
`mm_processor_kwargs` in the request:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "nemotron-3.5-transcribe@instruct",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only."},
    {"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}
  ]}],
  "mm_processor_kwargs": {"capture_ctc_timestamps": true}
}'
```

`"timestamp_granularities": ["word"]` asks for the same. The request must
hold exactly one clip, and `n` above 1 gets a 400. The response adds a
`ctc_timestamps` object, in the shape that NVIDIA's NeMo and vLLM servers
use:

```jsonc
"ctc_timestamps": {"words": [{"word": "He", "start": 0.36, "end": 0.524, "speaker": "0"}, ...],
                   "diarization": [{"speaker": 0, "start": 0.39, "end": 4.98}, ...],
                   "speaker_tag_to_diarization_speaker": {"0": 0, "1": 1}, "error": null}
```

| Field | Meaning |
|-------|---------|
| `words` | Each word of the reply, with `start` and `end` in seconds, and `speaker`, the number of its `<spk:N>` tag as a string |
| `diarization` | The stretches of speech of each speaker that the encoder heard, by the encoder's own speaker number |
| `speaker_tag_to_diarization_speaker` | The encoder speaker that each speaker tag matched |
| `error` | Why `words` is empty, else `null`. The response is still a 200. |

With `"stream": true`, the text deltas stream as the model writes them.
The final choice chunk waits until the words are timed, and carries
`finish_reason` and `ctc_timestamps` together. The usage chunk follows
when `stream_options.include_usage` is set, and then `[DONE]`.

### Transcription and translation

`/v1/audio/transcriptions` and `/v1/audio/translations` take a multipart
form with the recording in `file`, as OpenAI's routes do.
[Speech-to-text](services.md#speech-to-text) lists the fields that every
speech model takes. A served model that hears audio, such as
[Nemotron 3.5 Transcribe](nemotron-transcribe.md#transcription-and-translation-routes),
adds speakers and word times:

| Name | Default | Meaning |
|------|---------|---------|
| `diarization` | `false` | `true` labels the speakers in `verbose_json`, `srt`, `vtt` and `diarized_json`. With `json` or `text`, a 400. |
| `response_format=diarized_json` | Off | OpenAI's diarized transcript: one segment per speaker turn, with speakers `A`, `B` and on |
| `timestamp_granularities[]` | None | `word` adds `words` to `verbose_json`. With `diarized_json`, a 400. |

When `server.stt` names such a model, the model name
`gpt-4o-transcribe-diarize` goes to it, as `whisper-1` does. When the aligner cannot time a transcript, the
response is still a 200 with the text. `verbose_json` and `diarized_json`
then carry empty `words` and `segments` and an `alignment_error` field
that names the reason.

Transcriptions and translations do not stream. The server answers when
the whole recording is done.

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
| A body over 64 MiB, or an audio form over 1024 MiB, to a route other than the image API, `POST /v1/videos` and the step routes | 413 before the body is read | None |
| A body to the [image API](image-api.md), or a JSON body to [`POST /v1/videos`](video-api.md), whose read and parse need more than the memory free | 413 before the body is parsed, an image form's files are read, or an image decodes, with the body's size or the image's pixels and the memory free | `GMLX_OVERCOMMIT=1` |
| A JSON body to the [image API](image-api.md) or [`POST /v1/videos`](video-api.md) that fits the memory free alone, but not beside the other bodies being read | 503 with `Retry-After`, before the body is parsed, with what those bodies hold | `GMLX_OVERCOMMIT=1` |
| An [image API](image-api.md) form, or a [`POST /v1/videos`](video-api.md) form's text, that fits the memory free alone but not beside other bodies being read | 503 with `Retry-After`, with what those bodies hold | `GMLX_OVERCOMMIT=1` |
| The text fields of an [image API](image-api.md) or [`POST /v1/videos`](video-api.md) form past the memory free | 413 once the text read passes the memory free, with the memory free | `GMLX_OVERCOMMIT=1` |
| The files of a [`POST /v1/videos`](video-api.md) form larger than the room left in the video job store | 413 before the files are read, with their size and the room | [`video.store_max_bytes`](config.md#video-jobs) |
| A JSON body or a form to the [image API](image-api.md) or [`POST /v1/videos`](video-api.md) that sends nothing for 60 s before its end | 408 | None |
| From a [launch session](container-security.md#what-the-client-reaches-on-the-server), a body over 32 MiB, or 64 MiB for audio | 413 before the body is read | None |
| More than 64 media items, more than 256 Mi pixels in all, or audio over 128 Mi samples or with a sample rate over 384 kHz | 400 before decoding | None |

The read and the parse of a body hold more than its bytes, so its price
counts what they hold at their peak. A JSON body is read into one buffer
and parsed once, which holds about twice its size. A data URL that the
route decodes adds its payload and the bytes it decodes to, and twice those
bytes for an image that an edit or a variation checks with Pillow. Each
value the body parses to adds 80 bytes, counted from its commas, colons
and opening brackets and braces before the parse. Each string counts at the
width of its own characters: a byte a character when it is plain ASCII,
up to four when it is not or holds a `\u` escape, so a prompt that is not
ASCII leaves an inline image at a byte a character. The text fields of a
form are priced one at a time as they arrive, each at about twice its
size when it is plain ASCII and up to about seven times when it is not,
and an image form adds the largest of its files, which the route reads
whole.
With `GMLX_OVERCOMMIT=1`, or with no model pool or memory budget, the
server reads the bodies it prices from free memory with no bound, and a
video form's files still take the job store's room. The step routes that the [ComfyUI nodes](comfyui.md) call
take a body of up to one 1 GiB tensor and its header, and price the work
it asks for from the memory free. The nodes send a video's frames in
windows that fit.

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
