# Server configuration

The YAML file that drives `gmlx serve`: every key, what it defaults to, and
how the layers combine into the settings a request runs with. It is written
for anyone running the server; the HTTP surface is in [api.md](api.md) and the
flags in [cli.md](cli.md#gmlx-serve).

- [Quick start](#quick-start)
- [Where the config lives](#where-the-config-lives)
- [Precedence](#precedence)
- [server](#server)
- [profiles](#profiles)
- [rules](#rules)
- [models](#models)
- [aliases](#aliases)
- [discover](#discover)
- [Param key reference](#param-key-reference)
- [Complete example](#complete-example)
- [Residency](#residency)
- [Reloading the config](#reloading-the-config)
- [Family defaults](#family-defaults)

## Quick start

There are three ways to start the server. Each one produces the same kind of
server; they differ in where the list of models comes from.

```sh
gmlx serve model-Q4_K_M.gguf        # one file, no config
gmlx init --models-dir ~/models     # scan a folder, write ~/.config/gmlx/gmlx.yaml
gmlx serve                          # a bare start finds that file
gmlx serve --models-dir ~/models    # scan a folder every start, no file written
```

A request names a model by the id `gmlx init` printed, and `gmlx list` shows
the ids again later. Auto-named ids carry the quant tag, so a folder with two
quants of one model gets `qwen3.6-27b-q4` and `qwen3.6-27b-q6`.

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.6-27b",
  "messages": [{"role": "user", "content": "Explain entropy."}]
}'
```

Sampling defaults come from the model's family card, so a fresh config needs
no sampling section. Append `@coding`, `@instruct`, `@creative` or
`@reasoning-low`, `@reasoning-medium`, `@reasoning-high`, `@reasoning-max` to
any id to switch operating point, as in `"model": "qwen3.6-27b@coding"`. The
[Family defaults](#family-defaults) table at the end lists what each intent
sets, and `gmlx profiles` prints it live.

To expose the server on the LAN with a key, set `host: 0.0.0.0` and
`api_key` under `server`, restart, and have clients send the key as a bearer
token. The [bind and auth](#bind-and-auth) table has the details.

## Where the config lives

A bare `gmlx serve` searches three locations and takes the first that exists.
A project-local file is searched first so a repository can override your
user-level config.

### Default config locations

1. `./gmlx.yaml`
2. `~/.config/gmlx/gmlx.yaml`, where `gmlx init` writes
3. `~/.gmlx.yaml`, the legacy dotfile

`gmlx init --out FILE` writes somewhere else, and `gmlx serve --config FILE`
reads from anywhere. With no config found, a bare start scans the current
directory and prints a hint to run `init`.

`gmlx serve --print-config` resolves the config for the start mode you give
it, prints every key with its effective default as YAML, and exits. It is the
quickest way to see the whole schema.

An unknown key at the top level or inside `server`, `profiles`, `models`,
`rules` or `discover` fails the load. A typo like `pinned:` for `pin:` is
caught before the server starts. Unknown keys inside `sampling:`, `load:` and
`cache:` warn instead, because those namespaces pass through to the engine.

A config is a mapping with these top-level keys, all optional:

| Key | Contains |
|-----|----------|
| `server` | bind, auth, paths, memory, scheduling, services |
| `profiles` | reusable bundles of sampling, load, cache and prompt settings |
| `rules` | glob patterns that attach a profile to model ids |
| `models` | the served models, one entry per id |
| `aliases` | friendly names for a model or a model with a profile |
| `discover` | directory scans that add models without listing them |
| `talk` | voice chat settings, see [talk.md](talk.md) |
| `assistant` | the tool-loop assistant, see [assistant.md](assistant.md) |
| `theme`, `themes` | chat REPL colors, see [chat.md](chat.md) |

## Precedence

A request's effective settings are merged from these layers, lowest first.
Each layer only fills what the layers above it leave unset, so a profile's
`temperature` applies to a request that omits `temperature` and yields to
one that sends it.

| Layer | Set where | Wins over |
|-------|-----------|-----------|
| family base | built in, per detected family | nothing |
| server default profile | `server.defaults.profile` | family base |
| rule profile | first matching `rules` entry | server default |
| model profile | `models.<id>.profile`, or `@profile` on the request | rule profile |
| per-model tweak | `models.<id>.profiles.<name>` | the selected profile |
| model overrides | `models.<id>.overrides` | everything configured |
| request fields | the request body | everything |

An `@profile` on the request replaces the model's configured profile rather
than stacking on it. The name may be a user profile or a built-in intent, and
an unknown name returns 400. A per-model tweak applies only when its name is
the profile selected for the request.

Two settings follow the chain but bind at different times. A profile's
`system` is injected only when the request carries no system message. A
profile's `chat_template` is baked into the tokenizer at load, so two ids on
the same file with different templates are two resident copies, and a
request cannot change it.

Some settings can also be set by a `serve` flag or an environment variable.
When more than one is set, this is the order:

| Source | When it is read | Applies to |
|--------|-----------------|------------|
| `serve` flag | at start | this server process |
| config key | at start and on reload | this server process |
| environment variable | at start, some per request | every gmlx process in that shell |

The flag wins, then the config key, then the environment. The one exception
is the MLX buffer-cache limit, where the environment variable wins over
`server.cache_limit_gb` so a benchmark can pin it without editing the file.
Every variable is listed in [env-vars.md](env-vars.md).

## server

Settings for the process itself. Everything here has a default, so a config
with no `server:` block runs a loopback server on port 8080 with the prompt
cache off.

```yaml
server:
  host: 127.0.0.1
  port: 8080
  api_key: null
  model_dirs: [~/models]
  budget_gb: null
  cache: {enabled: true, disk: false}
  defaults:
    ttl_s: 900
    model: qwen3.6-27b
```

### Bind and auth

| Key | Default | Meaning |
|-----|---------|---------|
| `host` | `127.0.0.1` | bind address; a non-loopback address needs `api_key` or `no_auth` |
| `port` | `8080` | bind port |
| `api_key` | `null` | key required on every endpoint except `/health` |
| `no_auth` | `false` | allow a non-loopback bind with no key, for auth handled in front |

With `api_key` set, clients send `Authorization: Bearer <key>` or
`x-api-key: <key>`. The config is the only place the server reads its key
from; there is no `serve --api-key` flag and no environment fallback. The
client tools `ps`, `status`, `launch` and `menubar` take `--api-key` to
present one. `/health` stays open and returns liveness only, so probes and
`launch` keep working against a keyed server. `OPTIONS` requests are also
exempt, because browsers send CORS preflights without credentials.

Two guards protect a local server from browser-borne attacks. On a loopback
bind, a request whose `Host` header is not a loopback name gets 403, which
defeats DNS rebinding. CORS answers with a literal `*` and no credentials, so
a page cannot ride a cookie session; auth here is header based, so real
clients are unaffected.

### Paths

| Key | Default | Meaning |
|-----|---------|---------|
| `model_dirs` | `[]` | roots for relative `path`, `mmproj`, `draft_gguf` and `adapter` values, and the default scan target |
| `hf_cache` | `false` | let named Hugging Face ids and `hf:` paths resolve from the local cache |

Paths in `model_dirs` expand `~` and `$VAR`. A relative path on a model is
searched against the roots in order, and a miss fails the load naming the
roots searched. With one root, every model entry can be a bare filename.

By default the server never touches Hugging Face. A request for a model that
is neither a local file nor a configured id gets 403 rather than a download
([api.md](api.md#hugging-face-policy)). With `hf_cache: true`, `HF_HUB_OFFLINE`
is set and a named repo id resolves from the local cache only. A model `path`
may then be a portable `hf:<org>/<repo>/<file.gguf>[@rev]` reference, which
resolves on any machine that shares the cache. `gmlx init --from-hf-cache`
and `gmlx sync-models --from-hf-cache` write those entries and set the key.
The cache is never listed in `/v1/models`; a cached model is addressable only
when named under `models`.

### Memory and residency

| Key | Default | Meaning |
|-----|---------|---------|
| `budget_gb` | `null` | resident weight budget in GB; `null` is 0.8x the GPU recommended working set |
| `max_models` | `null` | secondary cap on how many models stay resident |
| `cache_limit_gb` | `null` | MLX buffer-cache cap in GiB; `null` sizes it at 4 to 12 GiB from the working-set slack, negative never bounds it |
| `defaults.ttl_s` | `900` | seconds a non-pinned model sits idle before it unloads; `null` or `0` never |
| `defaults.preload` | `null` | models to load at startup: `all` or a list of ids |
| `defaults.model` | `null` | the model used when a request omits `model`; with one model, that model |
| `defaults.profile` | `null` | a profile applied to every model as the lowest configured layer |

How pinned, kept and idle models share the budget is in
[Residency](#residency). The buffer cache is the pool of freed GPU buffers
MLX keeps for reuse; left unbounded it can hold tens of GB the kernel counts
as wired, so the server always bounds it. [performance.md](performance.md#the-mlx-buffer-cache-at-deep-context)
explains when to change the cap.

### Scheduling

| Key | Default | Meaning |
|-----|---------|---------|
| `prefill_step_size` | `null` (2048) | prefill chunk size in tokens for every model; lower caps peak memory on long prompts |
| `dtype` | `null` (`auto`) | activation width for every model: `auto`, `bfloat16` or `float16`; `auto` picks float16 on M1 and M2 |
| `decode_prefill_ratio` | `null` (`auto`) | how admission prefills share GPU time with live decode; a number pins a static share, `0` alternates |
| `prefill_tick_ms` | `null` (500) | wall-clock budget per prefill chunk while streams decode; chunks halve to fit, `0` never |
| `token_queue_timeout_s` | `null` (1800) | seconds to wait for the next token before failing the request; `0` waits forever |

These are server-wide because the engine reads them per request, after each
model's load has finished. The same values are available as `serve` flags
and, for a live A/B, as environment variables.

`prefill_step_size` bounds the working memory of one request, which scales
with chunk size times context depth. Lower it to fit deep prompts on a big
model, at some prefill throughput cost.

`dtype` covers the non-quantized parameters, the dequantized embedding table
and every activation between quantized matmuls; the KV cache follows it.
Weights on disk are untouched. `auto` reads the GPU generation and gives
float16 on M1 and M2, which have no native bfloat16 arithmetic, and bfloat16
elsewhere.

`decode_prefill_ratio` matters only when requests arrive while others are
decoding. Stock scheduling runs one decode step per prefill chunk, which at
deep context stalls every live stream for seconds per admission. `auto`
paces only when a live stream would otherwise drop below half its rate. A
number such as `1.0` admits a chunk only after the decode batch has received
that multiple of the chunk's GPU time, and `0` restores stock scheduling.
`prefill_tick_ms` bounds the other half of the problem, the length of a
single chunk, by halving chunks until their predicted time fits the budget.
Both are inert when nothing is decoding. The measured effects are in
[performance.md](performance.md#serving-concurrent-requests).

`token_queue_timeout_s` fires mainly on a long prefill that has not produced
its first token, such as a big prompt on an over-RAM model. A timed-out
request is cancelled, logged as failed, and recorded as `last_error` in
`/v1/metrics`. A streaming client receives a final error event.

### Features

| Key | Default | Meaning |
|-----|---------|---------|
| `family_defaults` | `true` | apply the family model-card sampling and the built-in intents |
| `stochastic_mtp` | `false` | accept sampled speculative tokens by rejection sampling, raising acceptance at the cost of token-identical output |
| `gpu_keepwarm` | `false` | hold GPU clocks up while a streamed model decodes; idle costs nothing |
| `menubar` | `true` | let a background `serve` raise the macOS menu bar app |

`stochastic_mtp` keeps the exact sampling distribution but is not
token-identical to a non-speculative run; greedy requests are unaffected.
It is applied at startup, so a reload does not change it. The measured gain
is in [performance.md](performance.md#stochastic-acceptance-opt-in).
`gpu_keepwarm` is described in [streaming.md](streaming.md#gpu-keep-warm---gpu-keepwarm).

### Services

| Key | Default | Meaning |
|-----|---------|---------|
| `stt` | `null` | speech-to-text model: an alias such as `whisper-turbo`, a repo id, a directory, or `true` |
| `tts` | `null` | text-to-speech model: an alias such as `kokoro`, a repo id, a directory, or `true` |
| `embeddings` | `null` | embeddings model: a GGUF path, `hf:` ref, alias, or `true` |
| `rerank` | `null` | reranker: a Qwen3-Reranker GGUF path, `hf:` ref, alias, or `true` |
| `assistants` | `{}` | served assistant ids, each naming a `model` and optional `memory` and `mcp` |
| `assistant_allow_remote` | `false` | required to serve assistants on a non-loopback bind |

Each service adds an endpoint; the models, the extras they need and the
request shapes are in [services.md](services.md). A service whose file is
missing is disabled with a warning rather than failing startup.

Served assistants are pseudo-models that answer through the built-in tool
loop on the server, so a thin client gets tools without a loop of its own.
Their tools run on the server host, so a non-loopback bind with assistants
configured refuses to start unless `assistant_allow_remote` is set, and a
remote-exposed assistant must declare its own `mcp` scope. The contract is in
[assistant.md](assistant.md#served-assistants).

### Cache

`server.cache` is the base of the prompt cache settings; profiles and models
override it. `gmlx init` writes `cache: {enabled: true, disk: false}`, and a
config with no `cache` block leaves the prompt cache off. The keys are in
[Cache keys](#cache-keys).

## profiles

A profile is a named bundle of settings that a rule, a model or a request can
select. Two kinds exist. The built-in intents ship in code and need no
config; user profiles are written here and may extend an intent or another
profile.

### Built-in intents

Model vendors publish recommended sampling per family, and the numbers
differ: the Gemma card warns against low temperature, Qwen3.6 publishes three
operating points, and gpt-oss wants `top_k` off. gmlx detects each model's
family from its GGUF header and uses the card's values as the lowest layer.
The card's other operating points become intents, addressable on every model
as `id@coding`, as a request `profile` field, or as `gmlx run --profile
coding`. An intent the family has no value for resolves to the family base,
never to another family's number. The full table is in
[Family defaults](#family-defaults).

Detection reads only the header and caches the result in
`~/.cache/gmlx/header-meta.json`. A `family` key on a model entry overrides
it, and `server.family_defaults: false` removes the base layer and the intent
names entirely.

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
| `extends` | `null` | a profile or intent applied first; this profile overrides what it sets |
| `sampling` | `{}` | request defaults, see [Sampling keys](#sampling-keys) |
| `load` | `{}` | model build settings such as KV quantization, see [Load keys](#load-keys) |
| `cache` | `{}` | prompt cache settings, see [Cache keys](#cache-keys) |
| `system` | `null` | a system prompt used when the request has none |
| `chat_template` | `null` | inline Jinja or a `.jinja` or `.txt` path replacing the GGUF's template |
| `chat_template_kwargs` | `{}` | variables passed to the template on every request |
| `thinking` | `null` | `on`, `off` or `adaptive`, mapped to the model's own template variable |
| `reasoning_effort` | `null` | a level the model's template accepts, such as `low`, `medium`, `high` |

A cycle in `extends` or an unknown parent fails the load.

`chat_template` applies to text models and to speculative models; a VLM
keeps the template its mmproj synthesizes. Because it is baked in at load, it
is part of the resident identity (see [Residency](#residency)).

`chat_template_kwargs` is for template flags a model exposes. The one that
matters most is `preserve_thinking` on Qwen3.6 and recent Gemma templates,
which keeps prior-turn `<think>` blocks in the rendered prompt. Agent loops
depend on the model seeing its own prior reasoning, so set it there. A
request may send the same field, and request keys win.

```yaml
profiles:
  agent:
    chat_template_kwargs: {preserve_thinking: true}
```

`thinking` and `reasoning_effort` exist because templates spell the two
reasoning controls differently:

| Family | Template variable | Values |
|--------|-------------------|--------|
| Qwen3.x, GLM | `enable_thinking` | `true`, `false` |
| MiniMax-M3 | `thinking_mode` | three states, so `adaptive` is accepted |
| Kimi K2.x | `thinking` | `true`, `false` |
| Hy3 | `reasoning_effort` | levels including `no_think` |
| gpt-oss | `reasoning_effort` | `low`, `medium`, `high`; cannot disable reasoning |

The two keys are mapped onto the serving model's spelling per request by
inspecting its template, so one profile applies across models. An explicit
`chat_template_kwargs` entry passes through verbatim and wins. The same
controls are `--thinking` and `--reasoning-effort` on `run` and `chat`.

```yaml
profiles:
  quick:
    thinking: off
  deep:
    reasoning_effort: high
```

### Overriding a built-in

Three levels, from global to surgical. A user profile named after an intent
replaces that intent everywhere. A profile with `extends: coding` inherits
the intent as resolved for each family and overrides only what it sets. A
model's own `profiles` block reshapes what a named profile means for that one
model (see [models](#models)).

```yaml
profiles:
  coding:
    sampling: {temperature: 0.4, min_p: 0.05}
  my-coding:
    extends: coding
    load: {kv_bits: 8}
```

## rules

A list of glob patterns, each attaching a profile to the model ids it
matches. The first match wins, and the pattern is `fnmatch` syntax, not a
regular expression. A rule sits below a model's own `profile` and above the
server default.

```yaml
rules:
  - {match: "*coder*",   profile: qwen-coder}
  - {match: "qwen3.6-*", profile: qwen-creative}
```

## models

One entry per served model. The key is the id everywhere: in `/v1/models`,
in the request `model` field, and on the command line.

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
  qwen3.6-27b-pure:                  # plain text, with a per-model tweak
    path: Qwen3.6-27B-Q4_K-pure/Qwen3.6-27B-Q4_K.gguf
    family: qwen3.6
    profiles:
      coding: {sampling: {min_p: 0.05}}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `path` | required | the GGUF, absolute, relative to `model_dirs`, or an `hf:` reference |
| `profile` | `null` | the profile selected when the request names none |
| `family` | detected | override the family read from the header |
| `profiles` | `{}` | per-model tweaks keyed by profile or intent name |
| `overrides` | `{}` | settings above every profile: `sampling`, `load`, `cache`, `system`, `chat_template`, `chat_template_kwargs`, `thinking`, `reasoning_effort` |
| `mmproj` | `null` | the vision or audio projector GGUF that makes this a multimodal model |
| `draft_gguf` | `null` | a separate drafter GGUF; implies `speculative` |
| `speculative` | `false` | speculative decoding with the GGUF's own MTP head or `draft_gguf` |
| `native_mtp` | `false` | prefer the GGUF's own head when `draft_gguf` is also set |
| `speculative_width_cap` | `null` | speculate only while at most this many requests decode together |
| `adapter` | `null` | a GGUF LoRA adapter applied at load, see [lora.md](lora.md#serving-one-base-with-many-adapters) |
| `stream` | `null` | `experts` streams the routed experts from disk, `cpu` runs the whole model on the CPU |
| `moe_experts` | `null` | fixed experts per token on a streamed model, lossy |
| `moe_expert_mass` | `null` | keep the smallest expert set covering this share of gate mass, lossy |
| `moe_miss_shed` | `null` | drop experts that would miss the decode arena down to this share of gate mass, lossy |
| `moe_layer_shed` | `null` | skip a streamed layer's routed experts with this probability, lossy |
| `moe_prestage` | `ranked` | `keepers` filters prestage predictions through the miss-shed policy |
| `prefill_feeder` | `true` | stage expert prefill straight from the GGUF on a streamed model |
| `decode_feeder` | `true` under `stream: experts` | decode from a wired, popularity-managed expert arena |
| `stream_fast_disk` | `auto` | the streamed-decode prefetch recipe: `auto` probes the drive, `on`, `off` |
| `pin` | `false` | never unload this model |
| `ttl_s` | `server.defaults.ttl_s` | idle seconds before this model unloads |

The streaming keys and the four lossy levers are explained in
[streaming.md](streaming.md), including how to size a lever before serving
with it. Each requires `stream`, is validated at load, and forks a resident
copy (see [Residency](#residency)). `stream: cpu` moves the whole process to
the CPU device, so it suits a single-model server. A speculative entry
refuses both `stream` values, and a VLM refuses `stream: cpu`. The old key
`cpu_moe` still parses as an alias for `stream` and warns.

An entry whose file has gone from disk is skipped with a warning, drops out
of `/v1/models`, and returns 404 when requested; the server keeps running.
Restore the file, or run `gmlx sync-models` to drop dead entries and register
new files in one pass. A malformed entry still fails the load.

#### speculative_width_cap

Speculation and batching compete for the same bandwidth: verifying a draft
widens each request's weight reads, which is nearly free with one stream and
costly with several. Each drafter carries a measured default, and this key
overrides it. `null` takes the default: uncapped for a native head on a dense
Qwen target, `2` for the Gemma assistant drafter, `1` for any routed-expert
target and for the Hy3 and DeepSeek-V4 drafters. `0` turns the cap off, and
`N` speculates only while at most N requests decode together. A drafter that
handles one sequence clamps any larger value.

A batch that grows past the cap converts to plain decode with the drafter
left loaded, and re-arms once it drains back. The transitions are described
in [internals/speculative-batching.md](internals/speculative-batching.md) and
the numbers behind the defaults in
[performance.md](performance.md#mtp-speculative-decoding).

## aliases

A map from a friendly name to a model id, or to an id with a profile. An
alias is listed in `/v1/models` as its own entry marked `alias_of`, which is
the only way a menu-driven client can pick a profile preset.

```yaml
aliases:
  fast:  gemma-e4b-vlm
  coder: qwen3.6-27b@qwen-coder
```

An alias may not contain `@` or collide with a model id, and its target must
exist. All of that is checked at load.

## discover

An opt-in scan that registers every GGUF under a directory without listing
them. The scan reads headers only, so it costs no tensor I/O.

```yaml
discover:
  - dir: null                  # null scans server.model_dirs
    recursive: true
    pair_mmproj: true
    speculative: auto
```

| Key | Default | Meaning |
|-----|---------|---------|
| `dir` | `null` | the directory; `null` scans every `model_dirs` root |
| `recursive` | `false` | descend into subdirectories |
| `pair_mmproj` | `true` | pair a sibling `mmproj*.gguf` with the model it matches |
| `speculative` | `auto` | `auto` and `true` enable speculation on models with an MTP head, `false` never |

A sibling assistant drafter pairs in as `draft_gguf` when its architecture,
hidden size and filename agree with the target. A drafter whose header names
its base model pairs with that model wherever it was found, and never with a
different one. A streamed model gets no drafter.

Ids derive from filenames: the shard suffix, the quant tag and the kind
markers `mmproj`, `assistant`, `draft` and `mtp` are stripped, and the rest
is slugified. On a collision the quant tag is appended rather than dropping
an entry. The id table prints at start.

## Param key reference

The keys accepted inside `sampling:`, `load:` and `cache:`, which appear
under `profiles`, under a model's `overrides` and, for cache, under `server`.

### Sampling keys

The request fields the engine honors, carried into generation. A request
that sets a field wins over any profile.

| Key | Default | Meaning |
|-----|---------|---------|
| `temperature` | family base | sampling temperature |
| `top_p` | family base | nucleus probability; `0` disables the filter |
| `top_k` | family base | candidate count; `0` disables the filter |
| `min_p` | family base | minimum probability relative to the best token; `0` disables |
| `max_tokens` | unlimited | generation cap |
| `seed` | `null` | per-request sampling seed |
| `repetition_penalty` | `null` | penalty over the last `repetition_context_size` tokens |
| `repetition_context_size` | `20` | window for the repetition penalty |
| `presence_penalty` | `null` | penalty on any token already generated |
| `frequency_penalty` | `null` | penalty scaled by how often a token was generated |
| `enable_thinking` | template default | whether the template opens a thinking block |
| `thinking_budget` | unlimited | reasoning tokens before the engine forces the block closed |
| `thinking_start_token` | `<think>` | the model's opening reasoning marker |
| `thinking_end_token` | `</think>` | the model's closing reasoning marker |
| `stop` | `null` | a string or list of stop sequences, chat completions only |
| `xtc_probability` | `null` | XTC sampling probability; not available on speculative models |
| `xtc_threshold` | `null` | XTC sampling threshold |

`top_p: 0` and `min_p: 0` mean disabled, as on the stock server. When only
`top_p` is set the nucleus is bounded to the top 1024 candidates so the sort
stays batched.

`seed` gives a deterministic sampling stream for that request, in-batch:
seeded rows draw from their own key stream while unseeded rows in the same
batch keep the shared one. Output is bitwise identical only under the same
batch composition and speculation setting, because batched matmul reduction
order shifts logits at float tolerance.

`thinking_budget` arms whenever it is set and acts once the model opens a
thinking block, whether the template pre-fills it or the model generates it.
On speculative models the close lands at a round boundary, so the cap can
overshoot by one draft block, and it is dropped for requests that decode in
a batch or after a preempt. Speculative models with a separate drafter
reject the key. The two token keys override the markers everywhere the
server needs them, and a family card sets them when a model spells its
markers differently. All three apply to `run` and `chat` through the config
overlay.

`stop` sequences trim mid-token safely and end the stream with
`finish_reason: "stop"`. The Anthropic endpoint keeps its own
`stop_sequences`. XTC excludes newline and end-of-sequence tokens, as
`gmlx run` does.

### Load keys

Applied at model build through a per-model environment window, so each model
loads with its own values. Each maps to an mlx-vlm environment variable
listed in [env-vars.md](env-vars.md#load-and-cache-keys).

| Key | Default | Meaning |
|-----|---------|---------|
| `kv_bits` | `null` | quantize the KV cache to 2, 3, 4, 6 or 8 bits |
| `kv_group_size` | `64` | quantization group size |
| `kv_quant_scheme` | `uniform` | the only accepted scheme |
| `max_kv_size` | `null` | cap the KV cache at this many tokens |
| `quantized_kv_start` | `0` | tokens kept unquantized at the start of the cache |

With `kv_bits` set, the policy is applied layer by layer at load and logged
as one `[kv]` line: growing attention KV quantizes except the last layer of a
deep stack, sliding windows and recurrent state stay fp16, and pooled caches
pack at rest. `/v1/models` reports the outcome per resident model as
`kv_quant` with a verdict of `full`, `partial`, `dropped` or `error`.
Speculative models quantize at batch size 1 and run fp16 KV while batched,
reported as `verdict_batched`.

`prefill_step_size` and `dtype` are not load keys, because the engine reads
them per request after the load window has closed. Set them under
[server](#scheduling).

### Cache keys

The prompt cache, which mlx-vlm calls APC, and its optional SSD tier. The
keys are server-level and overridable per profile and per model. The
environment variable behind each key is listed in
[env-vars.md](env-vars.md#load-and-cache-keys).

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | turn the prompt cache on |
| `block_size` | `16` | tokens per cache block |
| `num_blocks` | `2048` | blocks in the shared pool, 32k tokens at the default block size |
| `exact_entries` | `4` | whole-prefix entries kept for hybrid and recurrent models |
| `hash` | `fast` | the block hash |
| `disk` | `false` | `true` enables the SSD tier at `~/.cache/gmlx/apc`; a mapping sets the keys below |

| `disk.` key | Default | Meaning |
|-------------|---------|---------|
| `path` | `null` | the SSD tier directory; setting it enables the tier |
| `max_gb` | `null` | cap per namespace, so worst-case use is this times the model count |
| `workers` | `null` | writer threads |
| `read_mode` | `null` | how entries are read back |
| `namespace` | model path | the on-disk partition; the default keeps models apart |

The shared pool holds `num_blocks` times `block_size` tokens for every cached
prefix together, and one long request can fill it. A full pool evicts the
oldest prefixes rather than failing, so reuse depth shrinks to what fits.
Size it at expected prompt tokens times concurrent conversations, divided by
`block_size`. Sliding-window models also spend about `window / block_size`
blocks per checkpoint, so budget extra there. Raising the count costs only
metadata until stores land.

`exact_entries` sizes the in-memory pool hybrid and recurrent architectures
use instead of blocks. Each entry is a full prompt-cache clone, so raising it
trades memory for reuse; the default of 4 keeps a third conversation from
evicting the first.

What the cache restores per architecture is in
[performance.md](performance.md#the-prompt-cache).

## Complete example

Every key above in one config that builds cleanly. The test suite runs it
through the real loader.

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
    extends: coding                # compose over the built-in intent
    system: "You are a terse senior engineer."
    sampling: {temperature: 0.2, top_p: 0.9, repetition_penalty: 1.05}
    load: {kv_bits: 8, kv_group_size: 64}
    chat_template: "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
  qwen-creative:
    extends: creative
    sampling: {top_p: 0.98}
  reasoning:
    # cap thinking for this group; unlimited elsewhere
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
    family: qwen3.6                # override header detection (rarely needed)
    profile: reasoning
    profiles:
      coding: {sampling: {min_p: 0.05}}              # reshape @coding for this model
    overrides: {sampling: {thinking_budget: 2048}}   # per-model cap wins over profile
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

The smallest useful config is a model with a path:

```yaml
# doctest: build
models:
  my-model:
    path: ./my-model-Q4_K_M.gguf
```

## Residency

Several models stay resident at once: pinned models plus a pool bounded by
`server.budget_gb`. Each model's footprint is its GGUF size, because the
weights map from the file without a copy. Four states govern the pool.

| State | Set by | Unloads when |
|-------|--------|--------------|
| pinned | `pin: true`, `--pin` | never |
| kept | `POST /v1/keep`, `launch --model`, a talk session | the budget is full and it is least recently used |
| idle | any request | `ttl_s` seconds pass with no request, or the budget needs the room |
| preloaded | `defaults.preload` | as idle, after the first request |

The idle reaper only tears down a model whose batch worker has drained, so a
model is never unloaded mid-generation. A kept model holds a coding session's
model through idle gaps without blocking the budget; `{"keep": false}`
releases it and `POST /unload` evicts it.

The same GGUF served under two ids is one resident copy unless a setting
that changes how the model is loaded differs: `load` keys, `mmproj`,
`draft_gguf`, `speculative`, `adapter`, `chat_template`, `stream` and the
streaming levers. Sampling, `system` and `ttl_s` never fork a copy. How a
streamed model is priced against the budget is in
[streaming.md](streaming.md#residency-of-a-streamed-model).

## Reloading the config

A server started from a config file re-reads it on `POST /v1/reload` or
`SIGHUP`. The file is parsed again, the model registry rebuilt, and resident
models whose load settings are unchanged stay loaded. A config with no
models serves, so you can start empty and add models by reloading.

`gmlx init` and `gmlx sync-models` reload for you: after rewriting the
config, each signals a running server started from that file, unless you
pass `--no-reload`. A single-model server records no config path and is
never signalled.

Adding or removing `mmproj`, `draft_gguf` or `speculative` on a model takes
effect on that model's next cold load. A resident copy keeps its shape until
it is evicted or unloaded.

## Family defaults

The built-in sampling per family and the intents each family defines. The
first column is the family, the second the GGUF architectures it covers, and
the last the intents, each with what it changes from the base. `gmlx
profiles` prints the same table live, and `gmlx profiles <id>` shows one
model fully resolved.

| family | GGUF arches | base (general use) | family intents |
|--------|-------------|--------------------|----------------|
| `qwen3.6` | `qwen35`, `qwen35moe`, `qwen3next`, `qwen4exp` | temperature=1.0 top_p=0.95 top_k=20 min_p=0.0 | `@coding`: temperature=0.6 top_p=0.95 top_k=20 min_p=0.0; `@instruct`: temperature=0.7 top_p=0.8 top_k=20 min_p=0.0 presence_penalty=1.5 enable_thinking=False |
| `qwen3` | `qwen3`, `qwen3moe`, `qwen3vlmoe` | temperature=0.6 top_p=0.95 top_k=20 min_p=0.0 | `@instruct`: temperature=0.7 top_p=0.8 top_k=20 min_p=0.0 enable_thinking=False |
| `qwen2.5` | `qwen2`, `qwen2moe` | temperature=0.7 top_p=0.8 top_k=20 repetition_penalty=1.05 | - |
| `gemma` | `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4`, `diffusion-gemma` | temperature=1.0 top_p=0.95 top_k=64 | - |
| `gpt-oss` | `gpt-oss` | temperature=1.0 top_p=1.0 | `@reasoning-high`: temperature=1.0 top_p=1.0 reasoning_effort=high; `@reasoning-low`: temperature=1.0 top_p=1.0 reasoning_effort=low; `@reasoning-medium`: temperature=1.0 top_p=1.0 reasoning_effort=medium |
| `glm` | `glm4`, `glm4moe`, `glm-dsa`, `glm5next` | temperature=1.0 top_p=0.95 | - |
| `deepseek` | `deepseek2`, `deepseek4` | temperature=0.6 top_p=0.95 | - |
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

The `default` row is the fallback for unknown architectures. On gpt-oss the
reasoning intents set `reasoning_effort`, which the harmony template renders;
Kimi's set `thinking_effort` with `low`, `high` and `max` and no medium
point; Muse's set `reasoning_strength`. The Qwen `@instruct` intents also set
`enable_thinking: false`. The Kimi and Muse rows pin their thinking markers
because those models spell them differently. The values are cited to the
model cards in `gmlx/gen/profiles.py`, and the test suite asserts this table
matches the code.
