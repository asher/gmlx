# Configuration keys

These are the keys of `gmlx.yaml`, grouped by the top-level block that
holds them. What the file is for, where gmlx finds it and how to create it
are in [Configuration file](config.md).

- [server](#server)
- [models](#models)
- [aliases](#aliases)
- [profiles](#profiles)
- [rules](#rules)
- [discover](#discover)
- [Precedence](#precedence)
- [Sampling, load and cache keys](#sampling-load-and-cache-keys)
- [Family defaults](#family-defaults)
- [Complete example](#complete-example)

## server

Settings of the server process. Every key has a default, and a file with
no `server` block runs a server on `127.0.0.1:8080` with the prompt cache
off.

```yaml
server:
  host: 127.0.0.1
  port: 8080
  model_dirs: [~/models]
  cache: {enabled: true, disk: false}
  defaults:
    ttl_s: 900
    model: qwen3.6-27b
```

### Bind and auth

| Key | Default | Meaning |
|-----|---------|---------|
| `host` | `127.0.0.1` | bind address. A non-loopback address needs `api_key` or `no_auth` |
| `port` | `8080` | bind port |
| `api_key` | `null` | key required on every endpoint except `/health` |
| `no_auth` | `false` | allow a non-loopback bind with no key, for auth handled by a proxy in front of the server |

Clients send the key as `Authorization: Bearer <key>` or `x-api-key: <key>`.
The server reads its key only from this file, with no flag or environment
variable, which keeps the key out of process listings and shell history.
The client commands `ps`, `status`, `launch` and `menubar` take `--api-key`
to present it. `/health` and CORS preflight `OPTIONS` requests pass
without a key.

A loopback server refuses a request whose `Host` header is not a loopback
name with 403, which blocks DNS rebinding. It answers CORS with `*` and no
credentials, so a web page cannot reuse a session. Clients never notice
either check, because the key travels in a header.

### Paths

| Key | Default | Meaning |
|-----|---------|---------|
| `model_dirs` | `[]` | folders searched for a relative `path`, `mmproj`, `draft_gguf` or `adapter`, and the default scan target |
| `hf_cache` | `false` | let a model `path` be an `hf:<org>/<repo>/<file.gguf>[@rev]` reference, resolved from the local Hugging Face cache |

`model_dirs` entries expand `~` and `$VAR`. A relative path is searched in
each folder in order, and a miss fails the load with the folders it
searched.

The server never downloads from Hugging Face. With `hf_cache: true`, the
Hugging Face libraries run offline and an `hf:` path resolves from the
local cache only, which makes the entry portable across machines that
share the cache. `gmlx init --from-hf-cache` and
`gmlx sync-models --from-hf-cache` write such entries and set the key. How
the server treats a request that names a Hugging Face model is in the
[HTTP API](api.md#hugging-face-policy).

### Memory and residency

| Key | Default | Meaning |
|-----|---------|---------|
| `budget_gb` | `null` | memory budget for loaded models in GB. `null` is 0.8 times the GPU working set that macOS recommends |
| `max_models` | `null` | the most models loaded at once, checked after the budget |
| `cache_limit_gb` | `null` | MLX buffer cache limit in GiB. `null` picks 4 to 12 GiB from the memory left over, and a negative value never limits it |
| `defaults.ttl_s` | `900` | seconds a model that is not pinned stays loaded without a request. `null` or `0` never unloads |
| `defaults.preload` | `null` | more models to load at start, `all` or a list of ids |
| `defaults.model` | `null` | the model for a request that names none. With a single model, that model |
| `defaults.profile` | `null` | a profile for every model, below rules and the model's own profile |

The server opens its port at once and loads its first model in the
background. That model is a pinned one, else `defaults.model`, else the
only model in the file, and it stays loaded for the life of the process.
The ids in `defaults.preload` load after it, one at a time, and unload
like any idle model. A request that arrives during a load waits for it.
How the budget is shared is in
[Which models stay loaded](config.md#which-models-stay-loaded).

The buffer cache holds GPU buffers that MLX keeps for reuse. At deep
context it can grow to tens of GB, so the server limits it when the
largest model leaves little memory spare. When to change the limit is in
[Performance tuning](performance.md#the-mlx-buffer-cache-at-deep-context).

### Scheduling

| Key | Default | Meaning |
|-----|---------|---------|
| `prefill_step_size` | `null`, meaning 2048 | tokens per prefill chunk. A lower value lowers the memory peak of a long prompt and slows prefill |
| `dtype` | `null`, meaning `auto` | activation type, `auto`, `bfloat16` or `float16`. `auto` picks `float16` on M1 and M2, which lack bfloat16 arithmetic |
| `decode_prefill_ratio` | `null`, meaning `auto` | how prefill of a new request shares GPU time with requests that are generating. A number fixes the share and `0` turns pacing off |
| `prefill_tick_ms` | `null`, meaning 500 | time budget per prefill chunk while other requests generate. Chunks halve to fit, and `0` never halves them |
| `token_queue_timeout_s` | `null`, meaning 1800 | seconds to wait for the next token before the request fails. `0` waits forever. On `/v1/systemone` it limits a whole decision |

These keys apply to every model, and each has a
[`gmlx serve` flag](cli.md#gmlx-serve).
`dtype` sets the width of the unquantized weights, the activations and
the KV cache, and never changes the weights on disk.

`decode_prefill_ratio` and `prefill_tick_ms` matter only when a request
arrives while others are generating. Their effect is measured in
[Performance tuning](performance.md#serving-concurrent-requests).

A request that times out is cancelled and recorded as `last_error` in
`/v1/metrics`, and a streaming client receives a final error event. The
usual cause is a long prompt on a model larger than memory, which can take
minutes to produce its first token.

### Features

| Key | Default | Meaning |
|-----|---------|---------|
| `family_defaults` | `true` | apply the family sampling defaults and the built-in intents |
| `stochastic_mtp` | `false` | accept sampled speculative tokens by rejection sampling. Acceptance rises, and output is no longer token-identical |
| `gpu_keepwarm` | `false` | keep the GPU clock up between tokens for every model. Streamed models with a decode feeder do this anyway |
| `menubar` | `true` | let a background `serve` start the macOS menu bar app |

`stochastic_mtp` keeps the sampling distribution exact, and greedy
requests are unaffected. A reload does not change it. The gain is in
[Performance tuning](performance.md#stochastic-acceptance), and the GPU
clock heartbeat is described in
[Models larger than memory](streaming.md#the-lossless-settings).

### Services

| Key | Default | Meaning |
|-----|---------|---------|
| `stt` | `null` | the speech-to-text model, as an alias such as `whisper-turbo`, a repo id, a folder, or `true` |
| `tts` | `null` | the text-to-speech model, as an alias such as `kokoro`, a repo id, a folder, or `true` |
| `embeddings` | `null` | the embeddings model, as a GGUF path, `hf:` reference, alias, or `true` |
| `rerank` | `null` | the reranker, as a Qwen3-Reranker GGUF path, `hf:` reference, alias, or `true` |
| `systemone` | `{}` | settings of `/v1/systemone`, listed under [Structured decisions](#structured-decisions) |
| `assistants` | `{}` | assistants served as models, each with a `model` and optional `memory` and `mcp` |
| `assistant_allow_remote` | `false` | allow served assistants on a non-loopback bind |

Each service adds an endpoint, and
[Speech, embeddings and rerank](services.md) lists the models each one
accepts. A service whose model is missing is turned off with a warning,
and the server still starts.

A served assistant runs its tools on the server host. A non-loopback
server with assistants therefore refuses to start unless
`assistant_allow_remote` is set, and each assistant it exposes must then
name an `mcp` scope. See [Assistant](assistant.md#served-assistants).

### Structured decisions

`server.systemone` sets how `POST /v1/systemone` answers. The endpoint and
its requests are described in [Structured decisions](decisions.md).

| Key | Default | Meaning |
|-----|---------|---------|
| `model` | `null` | the model id or alias, optionally `id@profile`, for a request whose `model` is absent or names nothing |
| `canvas` | `64` | the most tokens one read's canvas holds, a positive multiple of 16, capped at the model's own canvas length |
| `constrained` | `true` | unembed only the read's label tokens. With `false`, the full vocabulary gives the same one-step label probabilities but other entropies and multi-step reads |
| `max_questions` | `64` | the most questions one request may ask. A request with more gets a 422 |
| `max_samples` | `32` | the cap on a request's `samples` and `auto_max`. A larger value is lowered to it |
| `think` | `0` | the thought budget for a request without `think`, 0 to 4096, or `"auto"` to think only when an answer is unsure |
| `think_threshold` | `0.8` | the confidence below which `"auto"` thinks, for a request without `think_threshold` |
| `think_budget` | `64` | the thought budget of `"auto"`, for a request without `think_budget` |

An unknown key here, or a `model` that is not a configured id or alias,
fails the load. A reload applies new values to the next request.

### Cache

`server.cache` holds the prompt cache settings for every model, and a
profile or a model entry can change them. The keys are listed under
[Cache keys](#cache-keys).

## models

One entry per model. The key of the entry is the model's id.

```yaml
models:
  qwen3.6-27b:                       # a GGUF with its own MTP head
    path: Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q4_K_S.gguf
    profile: qwen-creative
    speculative: true
    overrides: {sampling: {max_tokens: 2048}}
    pin: true
  gemma-31b-mtp:                     # a separate drafter GGUF
    path: google_gemma-4-31B-it-Q6_K_L.gguf
    draft_gguf: gemma-4-31B-it-assistant.Q8_0.gguf
  gemma-e4b-vlm:                     # a vision model
    path: gemma-4-E4B-it-Q6_K.gguf
    mmproj: mmproj-gemma-4-E4B-it-bf16.gguf
  qwen3.6-27b-pure:                  # a change to one intent for this model
    path: Qwen3.6-27B-Q4_K-pure/Qwen3.6-27B-Q4_K.gguf
    family: qwen3.6
    profiles:
      coding: {sampling: {min_p: 0.05}}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `path` | required | the GGUF, absolute, relative to `model_dirs`, or an `hf:` reference |
| `profile` | `null` | the profile used when the request names none |
| `family` | detected | the family for sampling defaults, instead of the one read from the header |
| `profiles` | `{}` | changes to a named profile or intent for this model only |
| `overrides` | `{}` | settings above every profile. Takes `sampling`, `load`, `cache`, `system`, `chat_template`, `chat_template_kwargs`, `thinking` and `reasoning_effort` |
| `mmproj` | `null` | the vision or audio projector GGUF that makes this a multimodal model |
| `draft_gguf` | `null` | a separate [drafter](glossary.md) GGUF. Implies `speculative` |
| `speculative` | `false` | speculate with the GGUF's own MTP head or with `draft_gguf`. Only a `discover` scan turns this on by itself |
| `native_mtp` | `false` | use the GGUF's own head even when `draft_gguf` is set |
| `speculative_width_cap` | `null` | speculate only while at most this many requests generate together |
| `adapter` | `null` | a GGUF LoRA adapter applied at load. See [LoRA adapters](lora.md#serving-one-base-with-many-adapters) |
| `stream` | `null` | `experts` streams the routed experts from disk, `cpu` runs the whole model on the CPU |
| `moe_experts` | `null` | a fixed expert count per token on a streamed model. Lossy |
| `moe_expert_mass` | `null` | keep the smallest expert set that covers this share of gate mass. Lossy |
| `moe_miss_shed` | `null` | drop experts that miss the decode [arena](glossary.md), down to this share of gate mass. Lossy |
| `moe_layer_shed` | `null` | skip a streamed layer's routed experts with this probability. Lossy |
| `moe_prestage` | `ranked` | `keepers` limits prestaging to the experts that `moe_miss_shed` keeps |
| `prefill_feeder` | `true` | stage expert prefill directly from the GGUF on a streamed model |
| `decode_feeder` | `true` under `stream: experts` | decode from a wired expert arena that keeps the most used experts |
| `stream_fast_disk` | `auto` | prefetch policy of streamed decode, `auto`, `on` or `off`. `auto` tests the drive |
| `pin` | `false` | never unload this model |
| `ttl_s` | `server.defaults.ttl_s` | seconds without a request before this model unloads |

The `moe_*`, feeder and `stream_fast_disk` keys need `stream`, and
[Models larger than memory](streaming.md) explains how to choose them. The
load refuses two combinations: a speculative model with `stream`, and a
multimodal model with `stream: cpu`. The key `cpu_moe` is still read as
`stream`, with a warning.

An entry whose file is missing is skipped with a warning. It drops out of
`/v1/models` and a request for it gets 404, while the server keeps
running. `gmlx sync-models` removes such entries.

### speculative_width_cap

Speculation and batching compete for memory bandwidth. Checking a draft
widens each request's weight reads, which costs little with one request
and a lot with several. `null` takes the drafter's measured default: no
cap for a native head on a dense Qwen model, `2` for the Gemma assistant
drafter and for families without a measurement, and `1` for every
mixture-of-experts model and for the drafters that handle a single
sequence, such as Hy3, DeepSeek-V4, Muse, Qwen3.8-Flash-Next and
GLM5-next. `0` removes the cap, except on a single-sequence drafter,
which stays at `1` whatever the value.

A batch that grows past the cap switches to plain decode with the drafter
still loaded, and speculates again once it shrinks. The mechanism is in
[Speculative batching](internals/speculative-batching.md)
and the measurements behind the defaults in
[Performance tuning](performance.md#mtp-speculative-decoding).

## aliases

A map from a second name to a model id, optionally with a profile.

```yaml
aliases:
  fast:  gemma-e4b-vlm
  coder: qwen3.6-27b@qwen-coder
```

`/v1/models` lists each alias as its own entry with `alias_of` set, which
lets a client that picks models from a menu choose a profile too. An alias
may not contain `@` or match a model id, and its target must exist.

## profiles

A profile is a named set of settings that a model entry, a rule or a
request selects. Profiles come in two kinds. Built-in intents ship with
gmlx and need no configuration, and user profiles are defined in this
block.

### Built-in intents

Model publishers recommend sampling values for each model family, and
some publish more than one operating point. gmlx reads each model's
family from its GGUF header, uses the recommended values as the family
defaults, and offers the other operating points as intents: `@coding`,
`@instruct`, `@creative`, and `@reasoning-low` through `@reasoning-xhigh`.

A request selects an intent with a suffix on the model id, such as
`qwen3.6-27b@coding`, or with a `profile` field. `gmlx run` and
`gmlx chat` take `--profile coding`. An intent that a family does not
define gives the family defaults, never another family's values. The
values are listed under [Family defaults](#family-defaults).

The detected family is cached in `~/.cache/gmlx/header-meta.json`. A
`family` key on a model entry replaces it, and
`server.family_defaults: false` turns off both the family defaults and
the intents.

### User profiles

```yaml
profiles:
  brief:
    sampling: {max_tokens: 512}
  qwen-coder:
    extends: brief
    system: "You are a terse senior engineer."
    sampling: {temperature: 0.2, repetition_penalty: 1.05}
    load: {kv_bits: 8, kv_group_size: 64}
    chat_template: ./templates/qwen-tools.jinja
```

| Key | Default | Meaning |
|-----|---------|---------|
| `extends` | `null` | a profile or intent applied first. This profile's own keys win |
| `sampling` | `{}` | request defaults, see [Sampling keys](#sampling-keys) |
| `load` | `{}` | how the model is built, such as KV cache quantization, see [Load keys](#load-keys) |
| `cache` | `{}` | prompt cache settings, see [Cache keys](#cache-keys) |
| `system` | `null` | a system prompt used when the request has none |
| `chat_template` | `null` | inline Jinja or a `.jinja` or `.txt` path that replaces the GGUF's template |
| `chat_template_kwargs` | `{}` | variables passed to the template on each request |
| `thinking` | `null` | `on`, `off` or `adaptive`, mapped to the model's own template variable |
| `reasoning_effort` | `null` | a level the model's template accepts, such as `low`, `medium` or `high` |

An `extends` cycle or an unknown parent fails the load.

`chat_template` applies at load, so two ids with different templates are
two loaded copies of the model. A multimodal model keeps the template its
`mmproj` provides.

`chat_template_kwargs` sets flags that a template exposes. The most useful
is `preserve_thinking` on Qwen3.6 and recent Gemma templates, which keeps
earlier `<think>` blocks in the prompt. Agents work better when the model
sees its earlier reasoning, so set it in a profile for agent use. Keys in
the request win.

```yaml
profiles:
  agent:
    chat_template_kwargs: {preserve_thinking: true}
```

`thinking` and `reasoning_effort` let one profile control reasoning on
models whose templates name the control differently:

| Family | Template variable | Values |
|--------|-------------------|--------|
| Qwen3.x, GLM | `enable_thinking` | `true`, `false` |
| MiniMax-M3 | `thinking_mode` | three states, so `adaptive` is accepted |
| Kimi K2.x | `thinking` | `true`, `false` |
| Hy3 | `reasoning_effort` | levels including `no_think` |
| gpt-oss | `reasoning_effort` | `low`, `medium`, `high`. Reasoning cannot be turned off |

On each request, gmlx maps the two keys to the variable name the model's
template uses. An explicit `chat_template_kwargs` entry passes through
unchanged and wins over the profile. On `run` and `chat` the same
controls are `--thinking` and `--reasoning-effort`.

A request can set reasoning too. Its `enable_thinking` wins, then its own
`thinking` or `reasoning_effort`, then the profile, then the template's
default. A `thinking` field also accepts the z.ai form,
`{"type": "enabled"}` or `{"type": "disabled"}`. Its optional
`clear_thinking` key reaches a template that reads `clear_thinking`, and
reaches a template that reads `preserve_thinking` inverted, so
`clear_thinking: false` keeps earlier reasoning in the prompt.

```yaml
profiles:
  quick:
    thinking: off
  deep:
    reasoning_effort: high
```

### Overriding a built-in

You can change an intent at three levels:

1. A user profile named after an intent, such as `coding`, replaces that
   intent for every model.
2. A profile with `extends: coding` starts from the intent as each family
   defines it and changes only the keys it sets.
3. A model's `profiles` block changes what a profile means for that model
   only.

```yaml
profiles:
  coding:
    sampling: {temperature: 0.4, min_p: 0.05}
  my-coding:
    extends: coding
    load: {kv_bits: 8}
```

## rules

A list of patterns, each giving a profile to the model ids it matches.
The first match wins, and patterns use shell wildcards, not regular
expressions.

```yaml
rules:
  - {match: "*coder*",   profile: qwen-coder}
  - {match: "qwen3.6-*", profile: qwen-creative}
```

## discover

Folders that the server scans at each start. Every GGUF found joins the
models without an entry of its own. The scan reads only the headers.

```yaml
discover:
  - dir: null                  # null scans server.model_dirs
    recursive: true
    pair_mmproj: true
    speculative: auto
```

| Key | Default | Meaning |
|-----|---------|---------|
| `dir` | `null` | the folder. `null` scans every `model_dirs` folder |
| `recursive` | `false` | scan subfolders too |
| `pair_mmproj` | `true` | pair a sibling `mmproj*.gguf` with the model it matches |
| `speculative` | `auto` | `auto` and `true` turn on speculation for models with an MTP head, `false` never does |

A drafter GGUF in the same folder pairs with a model as its `draft_gguf`
when their architecture, hidden size and file name agree. A drafter whose
header names its base model pairs only with that model, wherever it is.
Streamed models get no drafter.

The scan names each model after its file. It removes the shard suffix,
markers such as `mmproj`, `assistant`, `draft` and `mtp`, and imatrix
tags, and appends the quantization in short form, such as `-q4`. When two
files would get the same name, both get the full quantization, such as
`-q4-k-m` and `-q4-k-s`, and a remaining clash gets a number. The server
prints the resulting names at start.

## Precedence

The settings of a request come from these layers, lowest first. Each layer
fills only what the layers above it leave unset, so a profile's
`temperature` applies to a request that sends none and loses to one that
does.

| Layer | Set where | Wins over |
|-------|-----------|-----------|
| family defaults | built in, per detected family | nothing |
| server default profile | `server.defaults.profile` | family defaults |
| rule profile | the first matching `rules` entry | server default profile |
| model profile | `models.<id>.profile`, or `@profile` on the request | rule profile |
| per-model profile change | `models.<id>.profiles.<name>` | the selected profile |
| model overrides | `models.<id>.overrides` | every profile |
| request fields | the request body | everything |

A request's `@profile` replaces the model's own profile. It may name a
user profile or an intent, and an unknown name gets a 400. A per-model
profile change applies only when its name is the selected profile.

A profile's `system` prompt applies only to a request without a system
message. Its `chat_template` applies at load, so a request cannot change
it.

## Sampling, load and cache keys

These keys go inside `sampling`, `load` and `cache` in a profile or in a
model's `overrides`. The cache keys also go in `server.cache`.

### Sampling keys

The request fields that the engine honors. A field in the request wins
over every profile.

| Key | Default | Meaning |
|-----|---------|---------|
| `temperature` | family defaults | sampling temperature |
| `top_p` | family defaults | nucleus probability. `0` turns the filter off |
| `top_k` | family defaults | candidate count. `0` turns the filter off |
| `min_p` | family defaults | minimum probability relative to the most likely token. `0` turns it off |
| `max_tokens` | unlimited | the most tokens to generate |
| `seed` | `null` | a sampling seed for the request |
| `repetition_penalty` | `null` | penalty on tokens from the last `repetition_context_size` tokens |
| `repetition_context_size` | `20` | window of the repetition penalty |
| `presence_penalty` | `null` | penalty on any token already generated |
| `frequency_penalty` | `null` | penalty that grows with how often a token was generated |
| `enable_thinking` | template default | whether the template opens a thinking block |
| `thinking_budget` | unlimited | reasoning tokens allowed before the engine closes the thinking block |
| `thinking_start_token` | `<think>` | the model's opening reasoning marker |
| `thinking_end_token` | `</think>` | the model's closing reasoning marker |
| `stop` | `null` | a string or list of stop sequences, chat completions only |
| `xtc_probability` | `null` | XTC sampling probability. Not available on speculative models |
| `xtc_threshold` | `null` | XTC sampling threshold |

When only `top_p` is set, the nucleus is limited to the 1024 most likely
tokens so that the sort stays batched.

A `seed` makes one request's sampling repeatable without changing the
other requests in its batch. Two runs give the same tokens only when the
batch and the speculation setting are also the same, because a different
batch shape changes the logits slightly.

`thinking_budget` counts reasoning tokens from the moment a thinking block
opens, whether the template or the model opens it, and closes the block
at the limit. On a speculative model the close lands at the end of a
draft round, so the limit can be exceeded by one draft block. A request
that decodes in a batch, or resumes after preemption, runs without the
limit, and a speculative model with a separate drafter refuses the key.
Family defaults set `thinking_start_token` and `thinking_end_token` for
models whose markers are not `<think>`. The three keys apply to `run` and
`chat` as well.

A stop sequence can end generation in the middle of a token, and ends the
stream with `finish_reason: "stop"`. The Anthropic endpoint uses its own
`stop_sequences`. XTC never removes newline or end-of-sequence tokens.

### Load keys

These keys change how a model is built. Two ids that differ in a load key
are two loaded copies of the model. Each key also has an environment
variable that sets it for every model the process loads, listed in
[Environment variables](env-vars.md#load-and-cache-keys), and a
`gmlx serve model.gguf` start takes each key as a flag of the same name.

| Key | Default | Meaning |
|-----|---------|---------|
| `kv_bits` | `null` | quantize the KV cache to 2, 3, 4, 6 or 8 bits affine, or to 2, 3, 4, 5, 6 or 8 under [kvarn](glossary.md), default 6 |
| `kv_group_size` | `64` | affine quantization group size |
| `kv_quant_scheme` | `uniform` | `uniform` for affine or `kvarn` for variance-normalized. Any other value is refused at parse |
| `kv_tail_tokens` | `1024` | under kvarn, the newest tokens kept fp16. A multiple of 128 |
| `max_kv_size` | `null` | cap the request context budget at this many tokens |
| `quantized_kv_start` | `0` | tokens kept unquantized at the start of the cache. Not applied under kvarn |

With `kv_bits` set, the server decides for each layer whether its cache
quantizes, and logs the result in a `[kv]` line. Ordinary attention
layers quantize, except the last layer of a deep stack, while
sliding-window and recurrent state stay fp16. `/v1/models` reports the
result per loaded model as a `kv_quant` object whose `verdict` is `full`,
`partial`, `dropped` or `error`, as described in the
[HTTP API](api.md#endpoints). The load fails with `error` for a width
outside the scheme's list, a `kv_tail_tokens` that is not a multiple of
128, or different key and value widths under `uniform`. Under `uniform`,
speculative models quantize only while they serve one request.

`kv_quant_scheme: kvarn` uses the same layer rules. A model where no layer
converts runs fp16 KV and logs why, and never falls back to affine without
saying so. Speculative models keep kvarn at any batch size with
mlx-kquant 0.4.9 or later, and older versions fall back to fp16 while
batched. Which architectures convert is in
[Performance tuning](performance.md#kv-cache-quantization).

On the server, `max_kv_size` only caps the context budget of a request and
never builds a rotating window as it does on `run` and `chat`. The fp16
sink and tail buffers of kvarn count against the memory budget from the
first token.

`prefill_step_size` and `dtype` are not load keys. They apply to the
whole server, under [Scheduling](#scheduling).

### Cache keys

Settings of the prompt cache, which mlx-vlm calls APC, and of its
optional SSD tier. What the cache restores for each architecture is in
[Performance tuning](performance.md#the-prompt-cache).

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | turn the prompt cache on |
| `block_size` | `16` | tokens per cache block |
| `num_blocks` | `2048` | blocks in the shared pool, 32k tokens at the default block size |
| `exact_entries` | `4` | whole-prompt entries kept for hybrid and recurrent models |
| `hash` | `fast` | the block hash |
| `disk` | `false` | `true` turns on the SSD tier at `~/.cache/gmlx/apc`. A mapping sets the `disk.` keys |

| `disk.` key | Default | Meaning |
|-------------|---------|---------|
| `path` | `null` | the SSD tier folder. Setting it turns the tier on |
| `max_gb` | `null` | limit per namespace, so the most it uses is this times the model count |
| `workers` | `null` | writer threads |
| `read_mode` | `null` | how entries are read back |
| `namespace` | model path | the partition on disk. The default keeps models apart |

The pool holds `num_blocks` times `block_size` tokens for all cached
prompts together, and one long request can fill it. A full pool drops
its oldest prompts. Size it as the expected prompt length times the
number of conversations, divided by `block_size`. Sliding-window models
also use about `window / block_size` blocks per checkpoint. More blocks
cost only metadata until they fill.

Hybrid and recurrent models keep whole-prompt copies instead of blocks,
and `exact_entries` counts them. Each entry is a full copy of the cache,
so a higher value uses more memory. The default of 4 lets a third
conversation start without evicting the first.

## Family defaults

The sampling defaults of each family and the intents it defines, as
`gmlx profiles` prints them. The second column lists the GGUF
architectures of the family, and each intent is shown with the family
values it keeps. `gmlx profiles <id>` shows one model the same way. The
reasoning level has three names: `reasoning_effort` for gpt-oss, Hy3 and
Hy4, `thinking_effort` for Kimi, and `reasoning_strength` for Muse. Each
value comes from the model card cited in `gmlx/gen/profiles.py`.

| family | GGUF arches | base (general use) | family intents |
|--------|-------------|--------------------|----------------|
| `qwen3.6` | `qwen35`, `qwen35moe`, `qwen3next`, `qwen4exp` | temperature=1.0 top_p=0.95 top_k=20 min_p=0.0 | `@coding`: temperature=0.6 top_p=0.95 top_k=20 min_p=0.0; `@instruct`: temperature=0.7 top_p=0.8 top_k=20 min_p=0.0 presence_penalty=1.5 enable_thinking=False |
| `qwen3` | `qwen3`, `qwen3moe`, `qwen3vlmoe` | temperature=0.6 top_p=0.95 top_k=20 min_p=0.0 | `@instruct`: temperature=0.7 top_p=0.8 top_k=20 min_p=0.0 enable_thinking=False |
| `qwen2.5` | `qwen2`, `qwen2moe` | temperature=0.7 top_p=0.8 top_k=20 repetition_penalty=1.05 | - |
| `gemma` | `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4`, `diffusion-gemma` | temperature=1.0 top_p=0.95 top_k=64 | - |
| `gpt-oss` | `gpt-oss` | temperature=1.0 top_p=1.0 | `@reasoning-high`: temperature=1.0 top_p=1.0 reasoning_effort=high; `@reasoning-low`: temperature=1.0 top_p=1.0 reasoning_effort=low; `@reasoning-medium`: temperature=1.0 top_p=1.0 reasoning_effort=medium |
| `glm` | `glm4`, `glm4moe`, `glm-dsa`, `glm5next` | temperature=1.0 top_p=0.95 | - |
| `deepseek` | `deepseek2`, `deepseek4` | temperature=0.6 top_p=0.95 | - |
| `deepseek41` | `deepseek41` | temperature=1.0 top_p=0.95 | `@reasoning-high`: temperature=1.0 top_p=0.95 reasoning_effort=high; `@reasoning-low`: temperature=1.0 top_p=0.95 reasoning_effort=low; `@reasoning-max`: temperature=1.0 top_p=0.95 reasoning_effort=max |
| `minimax` | `minimax-m2`, `minimax-m3` | temperature=1.0 top_p=0.95 top_k=40 | - |
| `nemotron` | `nemotron_h_moe` | temperature=1.0 top_p=0.95 | - |
| `hunyuan` | `hunyuan-moe` | temperature=0.7 top_p=0.8 top_k=20 repetition_penalty=1.05 | - |
| `hy3` | `hy_v3` | temperature=0.9 thinking_start_token=<think:opensource> thinking_end_token=</think:opensource> | `@reasoning-high`: temperature=0.9 thinking_start_token=<think:opensource> thinking_end_token=</think:opensource> reasoning_effort=high; `@reasoning-low`: temperature=0.9 thinking_start_token=<think:opensource> thinking_end_token=</think:opensource> reasoning_effort=low |
| `hy4` | `hyv4` | temperature=0.9 top_p=1.0 thinking_start_token=<think:6124c78e> thinking_end_token=</think:6124c78e> | `@reasoning-high`: temperature=0.9 top_p=1.0 thinking_start_token=<think:6124c78e> thinking_end_token=</think:6124c78e> reasoning_effort=high; `@reasoning-low`: temperature=0.9 top_p=1.0 thinking_start_token=<think:6124c78e> thinking_end_token=</think:6124c78e> reasoning_effort=low |
| `kimi` | `kimi-k3` | temperature=1.0 top_p=0.95 thinking_start_token=<|open|>think<|sep|> thinking_end_token=<|close|>think<|sep|> | `@reasoning-high`: temperature=1.0 top_p=0.95 thinking_start_token=<|open|>think<|sep|> thinking_end_token=<|close|>think<|sep|> thinking_effort=high; `@reasoning-low`: temperature=1.0 top_p=0.95 thinking_start_token=<|open|>think<|sep|> thinking_end_token=<|close|>think<|sep|> thinking_effort=low; `@reasoning-max`: temperature=1.0 top_p=0.95 thinking_start_token=<|open|>think<|sep|> thinking_end_token=<|close|>think<|sep|> thinking_effort=max |
| `kimi-k2` | `deepseek2 named (?i)\bkimi` | temperature=1.0 top_p=0.95 | - |
| `muse` | `muse-glimmer` | temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=<|start|>assistant to=self<|message|> thinking_end_token=<|eom|> | `@reasoning-high`: temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=<|start|>assistant to=self<|message|> thinking_end_token=<|eom|> reasoning_strength=high; `@reasoning-low`: temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=<|start|>assistant to=self<|message|> thinking_end_token=<|eom|> reasoning_strength=low; `@reasoning-medium`: temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=<|start|>assistant to=self<|message|> thinking_end_token=<|eom|> reasoning_strength=medium; `@reasoning-xhigh`: temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=<|start|>assistant to=self<|message|> thinking_end_token=<|eom|> reasoning_strength=xhigh |
| `llama` | `llama`, `smollm3` | temperature=0.6 top_p=0.9 | - |
| `mistral` | `mistral3` | temperature=0.15 | - |
| `default` | (anything else) | temperature=0.7 top_p=0.95 | `@coding`: temperature=0.3 top_p=0.95; `@creative`: temperature=1.0 top_p=0.95 min_p=0.05; `@instruct`: temperature=0.7 top_p=0.95 |

The `default` row applies to architectures that no family claims.

## Complete example

Every block in one file that loads without errors.

```yaml
# doctest: build
server:
  host: 127.0.0.1
  port: 8080
  api_key: null
  no_auth: false
  model_dirs: [~/models]
  budget_gb: 96
  max_models: null
  hf_cache: false
  cache:
    enabled: true
    block_size: null
    num_blocks: null
    hash: fast
    disk:
      path: ~/.cache/gmlx/apc
      max_gb: 200
      workers: 2
      read_mode: direct
  family_defaults: true
  assistants:
    helper: {model: qwen3.6-27b, memory: false, mcp: null}
  assistant_allow_remote: false
  defaults:
    profile: null
    ttl_s: 900
    model: qwen3.6-27b
    preload: null
profiles:
  brief:
    sampling: {max_tokens: 1024}
  qwen-coder:
    extends: coding                # builds on the built-in intent
    system: "You are a terse senior engineer."
    sampling: {temperature: 0.2, top_p: 0.9, repetition_penalty: 1.05}
    load: {kv_bits: 8, kv_group_size: 64}
    chat_template: "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
  qwen-creative:
    extends: creative
    sampling: {top_p: 0.98}
  reasoning:
    # limits thinking for the models that use this profile
    sampling: {enable_thinking: true, thinking_budget: 1024}
rules:
  - {match: "*coder*", profile: qwen-coder}
models:
  qwen3.6-27b:
    path: Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q4_K_S.gguf
    profile: qwen-creative
    speculative: true
    overrides: {sampling: {max_tokens: 2048}}
    pin: true
  gemma-31b-mtp:
    path: google_gemma-4-31B-it-Q6_K_L.gguf
    draft_gguf: gemma-4-31B-it-assistant.Q8_0.gguf
    speculative: true
  gemma-e4b-vlm:
    path: gemma-4-E4B-it-Q6_K.gguf
    mmproj: mmproj-gemma-4-E4B-it-bf16.gguf
  qwen3.6-35b-a3:
    path: Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf
    family: qwen3.6                # replaces header detection, rarely needed
    profile: reasoning
    profiles:
      coding: {sampling: {min_p: 0.05}}              # changes @coding for this model
    overrides: {sampling: {thinking_budget: 2048}}   # wins over the profile
aliases:
  fast: gemma-e4b-vlm
  coder: qwen3.6-27b@qwen-coder
assistant:
  max_tool_rounds: 8
  tool_timeout_s: 60
  mcp:
    - {name: clock, command: [uvx, mcp-server-time]}
  memory: {enabled: true}
discover:
  - {dir: null, recursive: true, pair_mmproj: true, speculative: auto}
```

The smallest useful file is one model with a path:

```yaml
# doctest: build
models:
  my-model:
    path: ./my-model-Q4_K_M.gguf
```
