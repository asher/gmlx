# Server configuration

The YAML file that configures `gmlx serve`: each key, what it defaults to,
and how the layers combine into the settings a request runs with. The HTTP
surface is in [api.md](api.md) and the flags in [cli.md](cli.md#gmlx-serve).

- [Quick start](#quick-start)
- [Config file locations](#config-file-locations)
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

There are three ways to start the server, and they differ only in where the
list of models comes from.

```sh
gmlx serve model-Q4_K_M.gguf        # one file, no config
gmlx init --models-dir ~/models     # scan a folder, write ~/.config/gmlx/gmlx.yaml
gmlx serve                          # a bare start finds that file
gmlx serve --models-dir ~/models    # scan a folder every start, no file written
```

A request names a model by its id, which `gmlx list` prints. Where the id
comes from depends on the start mode. A single positional file is served as
its filename with the quant tag removed, so `model-Q4_K_M.gguf` becomes
`model`. A scan, whether by `gmlx init`, `--models-dir` or a
[discover](#discover) block, keeps the tag in compact form, so a folder with
two quants of a model gets `qwen3.6-27b-q4` and `qwen3.6-27b-q6`. Ids that
`init` wrote can be renamed in the file.

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.6-27b",
  "messages": [{"role": "user", "content": "Explain entropy."}]
}'
```

Sampling defaults come from the model's family card, so a fresh config needs
no sampling section. Append an intent such as `@coding`, `@instruct`,
`@creative` or a `@reasoning-` level from `low` to `xhigh` to any id to
switch operating point, as in `"model": "qwen3.6-27b@coding"`. The
[Family defaults](#family-defaults) table at the end lists the intents each
family has and what they set, and `gmlx profiles` prints the same table from
the running code.

To expose the server on the LAN with a key, set `host: 0.0.0.0` and `api_key`
under `server`, restart and have clients send the key as a bearer token. The
[bind and auth](#bind-and-auth) table has the details.

## Config file locations

A bare `gmlx serve` searches three locations and takes the first that
exists. The project-local file comes first, so a repository can override
your user-level config.

### Default config locations

1. `./gmlx.yaml`
2. `~/.config/gmlx/gmlx.yaml`, where `gmlx init` writes
3. `~/.gmlx.yaml`, the legacy dotfile

`gmlx init --out FILE` writes somewhere else and `gmlx serve --config FILE`
reads from anywhere. When no config is found, a bare start scans the current
directory and prints a hint to run `init`.

The quickest way to see the whole schema is `gmlx serve --print-config`,
which resolves the config for the start mode you give it, prints each key
with its effective default as YAML and exits.

An unknown key at the top level or inside `server`, `profiles`, `models`,
`rules` or `discover` fails the load, so a typo like `pinned:` for `pin:` is
caught before the server starts. Unknown keys inside `sampling:`, `load:` and
`cache:` only warn, because those namespaces pass through to the engine.

A config is a mapping with these top-level keys, all optional:

| Key | Contains |
|-----|----------|
| `server` | bind, auth, paths, memory, scheduling, services |
| `profiles` | reusable bundles of sampling, load, cache and prompt settings |
| `rules` | glob patterns that attach a profile to model ids |
| `models` | the served models, one entry per id |
| `aliases` | short names for a model or a model with a profile |
| `discover` | directory scans that add models without listing them |
| `talk` | voice chat settings, see [talk.md](talk.md) |
| `assistant` | the tool-loop assistant, see [assistant.md](assistant.md) |
| `theme`, `themes` | chat REPL colors, see [chat.md](chat.md) |

## Precedence

A request's effective settings are merged from these layers, lowest first,
and each layer only fills what the layers above it leave unset. A profile's
`temperature` therefore applies to a request that omits `temperature` and is
overridden by one that sends it.

| Layer | Set where | Wins over |
|-------|-----------|-----------|
| family base | built in, per detected family | nothing |
| server default profile | `server.defaults.profile` | family base |
| rule profile | first matching `rules` entry | server default |
| model profile | `models.<id>.profile`, or `@profile` on the request | rule profile |
| per-model tweak | `models.<id>.profiles.<name>` | the selected profile |
| model overrides | `models.<id>.overrides` | everything configured |
| request fields | the request body | everything |

An `@profile` on the request replaces the model's configured profile instead
of combining with it. The name may be a user profile or a built-in intent,
and an unknown name returns 400. A per-model tweak applies only when its name
is the profile selected for the request.

Two settings follow the precedence order but bind at different times. A
profile's `system` is injected only when the request carries no system
message, whereas its `chat_template` is applied to the tokenizer at load, so
two ids on the same file with different templates are two
[resident](glossary.md) copies and a request cannot change the template.

Some settings can also be set by a `serve` flag or an environment variable.
When more than one is set, this is the order:

| Source | When it is read | Applies to |
|--------|-----------------|------------|
| `serve` flag | at start | this server process |
| config key | at start and on reload | this server process |
| environment variable | at start, some on each request | each gmlx process in that shell |

The one exception is the MLX buffer-cache limit, where the environment
variable overrides `server.cache_limit_gb` so that a benchmark can pin it
without editing the file. All the variables are listed in
[env-vars.md](env-vars.md).

## server

Settings for the process itself. Everything here has a default, and a config
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
| `host` | `127.0.0.1` | bind address. A non-loopback address needs `api_key` or `no_auth` |
| `port` | `8080` | bind port |
| `api_key` | `null` | key required on all endpoints except `/health` |
| `no_auth` | `false` | allow a non-loopback bind with no key, for auth handled by a proxy in front of the server |

With `api_key` set, clients send `Authorization: Bearer <key>` or
`x-api-key: <key>`. The server reads its key from the config and nowhere
else. There is no `serve --api-key` flag and no environment fallback, which
keeps the key out of process listings and shell history. The client tools
`ps`, `status`, `launch` and `menubar` take `--api-key` so they can present
one. Two kinds of request skip the check. `/health` stays open and returns
liveness only, so probes and `launch` keep working against a keyed server,
and `OPTIONS` requests pass because browsers send CORS preflights without
credentials.

A loopback server also gets two protections against browser-borne attacks.
A request whose `Host` header is not a loopback name is refused with 403,
which defeats DNS rebinding. CORS answers with a literal `*` and no
credentials, so a page cannot reuse a cookie session. Real clients never
notice either check, because auth is header based.

### Paths

| Key | Default | Meaning |
|-----|---------|---------|
| `model_dirs` | `[]` | roots for relative `path`, `mmproj`, `draft_gguf` and `adapter` values, and the default scan target |
| `hf_cache` | `false` | let named Hugging Face ids and `hf:` paths resolve from the local cache |

Paths in `model_dirs` expand `~` and `$VAR`. A relative path on a model is
searched against the roots in order, and a miss fails the load naming the
roots searched. With a single root, each model entry can be a bare filename.

The server never contacts Hugging Face on a request. A request names a
configured id or gets 404, and the stock loader beneath the resolver is
gated as well, as [api.md](api.md#hugging-face-policy) describes.
`hf_cache: true` changes what the config may reference rather than what a
request may name. With it on, the Hugging Face libraries run in offline
mode and a model `path` may be a portable
`hf:<org>/<repo>/<file.gguf>[@rev]` reference, resolved from the local
cache and never the network, that works on any machine sharing that cache.
Both `gmlx init --from-hf-cache` and `gmlx sync-models --from-hf-cache`
write such entries and set the key.

### Memory and residency

| Key | Default | Meaning |
|-----|---------|---------|
| `budget_gb` | `null` | resident weight budget in GB. `null` is 0.8x the GPU recommended working set |
| `max_models` | `null` | secondary cap on how many models stay resident |
| `cache_limit_gb` | `null` | MLX buffer-cache cap in GiB. `null` sizes it at 4 to 12 GiB from the working-set margin, and negative never bounds it |
| `defaults.ttl_s` | `900` | seconds a non-pinned model is idle before it unloads. `null` or `0` never unloads |
| `defaults.preload` | `null` | further models to warm at startup, `all` or a list of ids |
| `defaults.model` | `null` | the model used when a request omits `model`. With a single model, that model |
| `defaults.profile` | `null` | a profile applied to all models as the lowest configured layer |

The server binds its port as soon as it starts and loads its first model in
a background thread. That model is a pinned one, else `defaults.model`, else
the sole entry when the config holds exactly one, and it stays resident for
the life of the process. Any further ids in `defaults.preload` warm after
it, one at a time, and those remain evictable like any idle model. A
request that arrives while a load is running waits for the rest of it.
With nothing to preload, the first request carries the whole load.

How pinned, kept and idle models share the budget is in
[Residency](#residency). The buffer cache is the pool of freed GPU buffers
that MLX keeps for reuse. Left unbounded at deep context it can hold tens of
GB that the kernel counts as wired, so the server caps it whenever the
largest configured model leaves little working-set slack, and leaves it
alone otherwise. When to change that is in
[performance.md](performance.md#the-mlx-buffer-cache-at-deep-context).

### Scheduling

| Key | Default | Meaning |
|-----|---------|---------|
| `prefill_step_size` | `null`, meaning 2048 | prefill chunk size in tokens for all models. Lower caps peak memory on long prompts |
| `dtype` | `null`, meaning `auto` | activation width for all models, `auto`, `bfloat16` or `float16`. `auto` picks float16 on M1 and M2 |
| `decode_prefill_ratio` | `null`, meaning `auto` | how admission prefills share GPU time with live decode. A number pins a static share, and `0` is stock scheduling |
| `prefill_tick_ms` | `null`, meaning 500 | wall-clock budget for each prefill chunk while streams decode. Chunks halve to fit, and `0` never halves |
| `token_queue_timeout_s` | `null`, meaning 1800 | seconds to wait for the next token before failing the request. `0` waits forever |

These are server-wide because the engine reads them on each request, after
the model's load has finished. They are also available as `serve` flags and,
for a live A/B, as environment variables.

The working memory of a request scales with chunk size times context
depth, so lowering `prefill_step_size` fits deep prompts on a big model, at
lower prefill throughput.

What `dtype` sets is the width of the non-quantized parameters, the
dequantized embedding table and the activations between quantized matmuls,
and the KV cache follows it. Weights on disk are untouched. `auto` reads the
GPU generation and gives float16 on M1 and M2, which have no native bfloat16
arithmetic, and bfloat16 elsewhere.

The two pacing keys, `decode_prefill_ratio` and `prefill_tick_ms`, matter
only when a request arrives while others are decoding. Stock scheduling
runs one decode step after each prefill chunk, and at deep context that
stalls the live streams for seconds on each admission. Two things set the
length of a stall: how often prefill chunks get the GPU, and how long each
chunk runs. The ratio
governs the first. `auto` paces admission only when a live stream would
otherwise drop below half its rate, a number such as `1.0` admits a chunk
only after the decode batch has received that multiple of the chunk's GPU
time, and `0` restores stock scheduling. The tick governs the second by
halving chunks until their predicted time fits the budget. The measured
effects are in [performance.md](performance.md#serving-concurrent-requests).

The request timeout, `token_queue_timeout_s`, triggers mainly on a long
prefill that has not produced its first token, such as a big prompt on an
over-RAM model. A
timed-out request is cancelled, logged as failed and recorded as `last_error`
in `/v1/metrics`, and streaming clients receive a final error event.

### Features

| Key | Default | Meaning |
|-----|---------|---------|
| `family_defaults` | `true` | apply the family model-card sampling and the built-in intents |
| `stochastic_mtp` | `false` | accept sampled speculative tokens by rejection sampling, raising acceptance but giving up token-identical output |
| `gpu_keepwarm` | `false` | force the GPU clock heartbeat on for every model. Streamed loads with a decode feeder run it anyway |
| `menubar` | `true` | let a background `serve` raise the macOS menu bar app |

`stochastic_mtp` keeps the exact sampling distribution but is not
token-identical to a non-speculative run, and greedy requests are
unaffected. It is applied at startup, so a reload does not change it. The
measured gain is in [performance.md](performance.md#stochastic-acceptance).
What the keep-warm heartbeat does, why it only helps a streamed model and
the variable that turns it off there are under
[streaming.md](streaming.md#the-lossless-settings).

### Services

| Key | Default | Meaning |
|-----|---------|---------|
| `stt` | `null` | the speech-to-text model, as an alias such as `whisper-turbo`, a repo id, a directory, or `true` |
| `tts` | `null` | the text-to-speech model, as an alias such as `kokoro`, a repo id, a directory, or `true` |
| `embeddings` | `null` | the embeddings model, as a GGUF path, `hf:` ref, alias, or `true` |
| `rerank` | `null` | the reranker, as a Qwen3-Reranker GGUF path, `hf:` ref, alias, or `true` |
| `assistants` | `{}` | served assistant ids, each naming a `model` and optional `memory` and `mcp` |
| `assistant_allow_remote` | `false` | required to serve assistants on a non-loopback bind |

Each service adds an endpoint. [services.md](services.md) lists the models
each accepts, the extras it needs and the request shape. A service whose
file is missing is disabled with a warning instead of failing startup.

Served assistants are pseudo-models that answer through the built-in tool
loop on the server, so a thin client gets tools without a loop of its own.
Because their tools run on the server host, a non-loopback bind with
assistants configured refuses to start unless `assistant_allow_remote` is
set, and even then a remote-exposed assistant must declare an `mcp` scope.
The contract is in [assistant.md](assistant.md#served-assistants).

### Cache

`server.cache` is the base of the prompt cache settings, which profiles and
models can override. `gmlx init` writes `cache: {enabled: true, disk:
false}`. The keys are in [Cache keys](#cache-keys).

## profiles

A profile is a named bundle of settings that a rule, a model or a request
can select. There are two kinds: the built-in intents ship in the package
and need no config, while user profiles are written in this block and may
extend an intent or another profile.

### Built-in intents

Model vendors publish recommended sampling for each family, and the numbers
differ: Gemma's card warns against low temperature, Qwen3.6 publishes three
operating points and gpt-oss recommends `top_k` off. gmlx detects each
model's family from its GGUF header and uses the card's values as the lowest
layer. A card's other operating points become intents, addressable on any
model as `id@coding`, as a request `profile` field, or as
`gmlx run --profile coding`. An intent the family has no value for resolves
to the family base, never to another family's number. The full table is in
[Family defaults](#family-defaults).

Detection reads only the header and caches the result in
`~/.cache/gmlx/header-meta.json`. A `family` key on a model entry overrides
it, and `server.family_defaults: false` removes the base layer and the
intent names entirely.

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
| `extends` | `null` | a profile or intent applied first. This profile overrides what it sets |
| `sampling` | `{}` | request defaults, see [Sampling keys](#sampling-keys) |
| `load` | `{}` | model build settings such as KV quantization, see [Load keys](#load-keys) |
| `cache` | `{}` | prompt cache settings, see [Cache keys](#cache-keys) |
| `system` | `null` | a system prompt used when the request has none |
| `chat_template` | `null` | inline Jinja or a `.jinja` or `.txt` path replacing the GGUF's template |
| `chat_template_kwargs` | `{}` | variables passed to the template on each request |
| `thinking` | `null` | `on`, `off` or `adaptive`, mapped to the model's own template variable |
| `reasoning_effort` | `null` | a level the model's template accepts, such as `low`, `medium`, `high` |

A cycle in `extends` or an unknown parent fails the load.

`chat_template` applies to text models and to speculative models, while a
VLM keeps the template its mmproj synthesizes. Because the template is
applied at load, it is part of the resident identity described under
[Residency](#residency).

`chat_template_kwargs` is for template flags a model exposes. The one that
matters most is `preserve_thinking` on Qwen3.6 and recent Gemma templates,
which keeps prior-turn `<think>` blocks in the rendered prompt. Agent loops
depend on the model seeing its prior reasoning, so a profile for agents
should set it, as the example below does. A request may send the field as
well, and request keys win.

```yaml
profiles:
  agent:
    chat_template_kwargs: {preserve_thinking: true}
```

`thinking` and `reasoning_effort` exist because templates name the two
reasoning controls differently:

| Family | Template variable | Values |
|--------|-------------------|--------|
| Qwen3.x, GLM | `enable_thinking` | `true`, `false` |
| MiniMax-M3 | `thinking_mode` | three states, so `adaptive` is accepted |
| Kimi K2.x | `thinking` | `true`, `false` |
| Hy3 | `reasoning_effort` | levels including `no_think` |
| gpt-oss | `reasoning_effort` | `low`, `medium`, `high`. Reasoning cannot be disabled |

The two keys are mapped onto the serving model's variable name on each
request by inspecting its template, which lets a single profile apply across
models. An explicit `chat_template_kwargs` entry passes through verbatim and
overrides the profile. On `run` and `chat` the same controls are `--thinking`
and `--reasoning-effort`.

```yaml
profiles:
  quick:
    thinking: off
  deep:
    reasoning_effort: high
```

### Overriding a built-in

A built-in intent can be changed at three levels, from global to most
specific:

1. A user profile named after an intent, such as `coding`, replaces that
   intent everywhere.
2. A profile with `extends: coding` inherits the intent as resolved for each
   family and overrides only what it sets.
3. A model's `profiles` block changes what a named profile means for that
   model alone, as described under [models](#models).

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
matches. The first match is used, and patterns are `fnmatch` syntax rather
than regular expressions. A rule ranks below a model's own `profile` and
above the server default.

```yaml
rules:
  - {match: "*coder*",   profile: qwen-coder}
  - {match: "qwen3.6-*", profile: qwen-creative}
```

## models

An entry for each served model. The key is the id everywhere, in `/v1/models`,
in the request `model` field and on the command line.

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
| `overrides` | `{}` | settings above all profiles. Accepts `sampling`, `load`, `cache`, `system`, `chat_template`, `chat_template_kwargs`, `thinking` and `reasoning_effort` |
| `mmproj` | `null` | the vision or audio projector GGUF that makes this a multimodal model |
| `draft_gguf` | `null` | a separate [drafter](glossary.md) GGUF, which implies `speculative` |
| `speculative` | `false` | speculate with the GGUF's own MTP head or `draft_gguf`. Only a `discover` scan enables this on its own |
| `native_mtp` | `false` | prefer the GGUF's own head when `draft_gguf` is also set |
| `speculative_width_cap` | `null` | speculate only while at most this many requests decode together |
| `adapter` | `null` | a GGUF LoRA adapter applied at load, see [lora.md](lora.md#serving-one-base-with-many-adapters) |
| `stream` | `null` | `experts` streams the routed experts from disk, `cpu` runs the whole model on the CPU |
| `moe_experts` | `null` | a fixed expert count for each token on a streamed model, lossy |
| `moe_expert_mass` | `null` | keep the smallest expert set covering this share of gate mass, lossy |
| `moe_miss_shed` | `null` | drop experts that would miss the decode [arena](glossary.md) down to this share of gate mass, lossy |
| `moe_layer_shed` | `null` | skip a streamed layer's routed experts with this probability, lossy |
| `moe_prestage` | `ranked` | `keepers` filters prestage predictions through the miss-shed policy, so it needs `moe_miss_shed` |
| `prefill_feeder` | `true` | stage expert prefill directly from the GGUF on a streamed model |
| `decode_feeder` | `true` under `stream: experts` | decode from a wired, popularity-managed expert arena |
| `stream_fast_disk` | `auto` | the streamed-decode prefetch policy, `auto`, `on` or `off`. `auto` probes the drive |
| `pin` | `false` | never unload this model |
| `ttl_s` | `server.defaults.ttl_s` | idle seconds before this model unloads |

The streaming keys are explained in [streaming.md](streaming.md), including
how to size a lossy setting before serving with it. All of the `moe_*`,
feeder and `stream_fast_disk` keys require `stream`, and each combination
of them is a separate resident copy as described under
[Residency](#residency). Two combinations are refused at load: a
speculative entry with either `stream` value, and a VLM with `stream: cpu`,
which moves the whole process to the CPU device and suits a single-model
text server. The old key `cpu_moe` still parses as an alias for `stream`,
with a warning.

An entry whose file is missing from disk is skipped with a warning, drops
out of `/v1/models` and returns 404 when requested, while the server keeps
running. Restore the file, or run `gmlx sync-models` to drop entries with
missing files and register new files in a single pass. A malformed entry
still fails the load.

#### speculative_width_cap

Speculation and batching compete for the same bandwidth. Verifying a draft
widens each request's weight reads, and that costs little with a single
stream and a lot with several. Each drafter carries a measured default and
this key overrides it. `null` takes the default: uncapped for a native head
on a dense Qwen target, `2` for the Gemma assistant drafter and for any
family without a measured value, and `1` for every routed-expert target and
for the drafters that handle a single sequence, such as Hy3, DeepSeek-V4,
Muse, Qwen3.8-Flash-Next and GLM5-next. `0` turns the cap off, and `N`
speculates only while at most N requests decode together. A single-sequence
drafter clamps any larger value.

A batch that grows past the cap converts to plain decode with the drafter
left loaded, and re-arms once it drains back. The transitions are described
in [internals/speculative-batching.md](internals/speculative-batching.md) and
the numbers behind the defaults in
[performance.md](performance.md#mtp-speculative-decoding).

## aliases

A map from a short name to a model id, or to an id with a profile. An
alias is listed in `/v1/models` as a separate entry marked `alias_of`, which
is the only way a menu-driven client can pick a profile preset.

```yaml
aliases:
  fast:  gemma-e4b-vlm
  coder: qwen3.6-27b@qwen-coder
```

An alias may not contain `@` or collide with a model id, and its target must
exist. All of that is checked at load.

## discover

An opt-in scan that registers each GGUF under a directory without listing
them. The scan reads headers only and does no tensor I/O.

```yaml
discover:
  - dir: null                  # null scans server.model_dirs
    recursive: true
    pair_mmproj: true
    speculative: auto
```

| Key | Default | Meaning |
|-----|---------|---------|
| `dir` | `null` | the directory. `null` scans all `model_dirs` roots |
| `recursive` | `false` | descend into subdirectories |
| `pair_mmproj` | `true` | pair a sibling `mmproj*.gguf` with the model it matches |
| `speculative` | `auto` | `auto` and `true` enable speculation on models with an MTP head, `false` never |

A sibling assistant drafter pairs in as `draft_gguf` when its architecture,
hidden size and filename agree with the target, and when a drafter's header
names its base model it pairs with that model wherever it was found, never
with a different one. Streamed models get no drafter.

Ids derive from filenames. The shard suffix, kind markers such as `mmproj`,
`assistant`, `draft` and `mtp`, and imatrix provenance tags are stripped,
what remains is slugified, and the quant tag is appended in compact form,
`-q4` or `-iq2`. When two files would share a compact tag, both get the full
codec instead, `-q4-k-m` and `-q4-k-s`, and a genuine clash gets a numeric
suffix. The id table prints at start.

## Param key reference

The keys accepted inside `sampling:`, `load:` and `cache:`, which appear
under `profiles`, under a model's `overrides` and, for cache, under `server`.

### Sampling keys

The request fields the engine honors, carried into generation. A request
that sets a field overrides any profile.

| Key | Default | Meaning |
|-----|---------|---------|
| `temperature` | family base | sampling temperature |
| `top_p` | family base | nucleus probability. `0` disables the filter |
| `top_k` | family base | candidate count. `0` disables the filter |
| `min_p` | family base | minimum probability relative to the best token. `0` disables |
| `max_tokens` | unlimited | generation cap |
| `seed` | `null` | a sampling seed for the request |
| `repetition_penalty` | `null` | penalty over the last `repetition_context_size` tokens |
| `repetition_context_size` | `20` | window for the repetition penalty |
| `presence_penalty` | `null` | penalty on any token already generated |
| `frequency_penalty` | `null` | penalty scaled by how often a token was generated |
| `enable_thinking` | template default | whether the template opens a thinking block |
| `thinking_budget` | unlimited | reasoning tokens before the engine forces the block closed |
| `thinking_start_token` | `<think>` | the model's opening reasoning marker |
| `thinking_end_token` | `</think>` | the model's closing reasoning marker |
| `stop` | `null` | a string or list of stop sequences, chat completions only |
| `xtc_probability` | `null` | XTC sampling probability. Not available on speculative models |
| `xtc_threshold` | `null` | XTC sampling threshold |

`top_p: 0` and `min_p: 0` mean disabled, as on the stock server. When only
`top_p` is set, the nucleus is bounded to the top 1024 candidates so that the
sort stays batched.

A `seed` makes a request's sampling repeatable without affecting the other
requests in its batch. Two runs give the same tokens only when the batch
composition and the speculation setting are also the same, because a
different batch shape changes the logits at floating-point tolerance.

The budget counts reasoning tokens from the moment the model opens a
thinking block, whether the template pre-fills the opener or the model
generates it, and forces the block closed at the cap. Two serving cases
loosen it. On a speculative model the close lands on a round boundary, so
the cap can overshoot by one draft block, and a request that decodes in a
batch, or resumes after a preempt, runs without the cap. A speculative model
with a separate drafter rejects the key. `thinking_start_token` and
`thinking_end_token` tell the server what the model's markers look like, and
family cards set them for the models whose markers differ from `<think>`.
All three keys also apply to `run` and `chat`.

Stop sequences trim mid-token safely and end the stream with
`finish_reason: "stop"`, and the Anthropic endpoint keeps its own
`stop_sequences`. XTC excludes newline and end-of-sequence tokens, as
`gmlx run` does.

### Load keys

These change how a model is built, so each model can have its own values
and two ids that differ in a load key are two resident copies. Each key has
a matching mlx-vlm environment variable, listed in
[env-vars.md](env-vars.md#load-and-cache-keys), which sets the same thing
for every model the process loads, and a positional `gmlx serve model.gguf`
takes each key as a flag of the same name, listed in [cli.md](cli.md#gmlx-serve).

| Key | Default | Meaning |
|-----|---------|---------|
| `kv_bits` | `null` | quantize the KV cache to 2, 3, 4, 6 or 8 bits affine, or to 2, 3, 4, 5, 6 or 8 under [kvarn](glossary.md), default 6 |
| `kv_group_size` | `64` | affine quantization group size |
| `kv_quant_scheme` | `uniform` | `uniform` for affine or `kvarn` for variance-normalized. Any other value is refused at parse |
| `kv_tail_tokens` | `1024` | under kvarn, the newest tokens kept fp16. A multiple of 128 |
| `max_kv_size` | `null` | cap the request context budget at this many tokens |
| `quantized_kv_start` | `0` | tokens kept unquantized at the start of the cache. Not applied under kvarn |

With `kv_bits` set, the server decides layer by layer which caches to
quantize and logs the result as a `[kv]` line. Ordinary attention layers
quantize, apart from the last layer of a deep stack, while sliding-window and
recurrent state stay fp16. `/v1/models` reports the outcome for each
resident model as a `kv_quant` object whose `verdict` is `full`, `partial`,
`dropped` or `error`, as [api.md](api.md#endpoints) describes. A load fails
with `error` for a width outside the scheme's list, a `kv_tail_tokens` that
is not a multiple of 128, or split key and value widths under `uniform`.
Under `uniform`, speculative models quantize at batch size 1 and run fp16 KV
while batched.

`kv_quant_scheme: kvarn` uses the same layer policy with `kv_bits` picking
the width and `kv_tail_tokens` the fp16 tail. Which layers convert and which
architectures decline is in
[performance.md](performance.md#kv-cache-quantization), and a model where no
layer converts runs fp16 KV with a logged reason, never a silent affine
fallback. Speculative models keep their kvarn records at any batch width
when mlx-kquant 0.4.9 or later is installed; older kernels fall back to fp16
while batched. Split key and value widths are a server-wide environment setting,
listed in [env-vars.md](env-vars.md#load-and-cache-keys). Two things differ
from `run` and `chat`. On the server `max_kv_size` only caps the request
context budget and never builds a rotating window, and kvarn's fp16 sink and
tail buffers count against the memory budget from the first token.

`prefill_step_size` and `dtype` are not load keys. They are server-wide
[scheduling](#scheduling) settings.

### Cache keys

The prompt cache, which mlx-vlm calls APC, and its optional SSD tier. These
keys are server-level, and a profile or a model can override them. The
environment variable behind each key is listed in
[env-vars.md](env-vars.md#load-and-cache-keys).

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | turn the prompt cache on |
| `block_size` | `16` | tokens in each cache block |
| `num_blocks` | `2048` | blocks in the shared pool, 32k tokens at the default block size |
| `exact_entries` | `4` | whole-prefix entries kept for hybrid and recurrent models |
| `hash` | `fast` | the block hash |
| `disk` | `false` | `true` enables the SSD tier at `~/.cache/gmlx/apc`. A mapping sets the `disk.` keys |

| `disk.` key | Default | Meaning |
|-------------|---------|---------|
| `path` | `null` | the SSD tier directory. Setting it enables the tier |
| `max_gb` | `null` | cap for each namespace, so worst-case use is this times the model count |
| `workers` | `null` | writer threads |
| `read_mode` | `null` | how entries are read back |
| `namespace` | model path | the on-disk partition. The default keeps models apart |

The shared pool holds `num_blocks` times `block_size` tokens for all cached
prefixes together, and a single long request can fill it. When full, the
pool evicts the oldest prefixes instead of failing, so reuse depth shrinks to
what fits. Size it at expected prompt tokens times concurrent conversations,
divided by `block_size`, and allow extra for sliding-window models, which
also use about `window / block_size` blocks for each checkpoint. Raising the
count uses only metadata memory until blocks are stored.

`exact_entries` sizes the in-memory pool that hybrid and recurrent
architectures use instead of blocks. Each entry is a full prompt-cache clone,
so raising it trades memory for reuse. The default of 4 keeps a third
conversation from evicting the first.

What the cache restores for each architecture is in
[performance.md](performance.md#the-prompt-cache).

## Complete example

All the keys above in a single config that loads without errors.

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
    # cap thinking for this group, unlimited elsewhere
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
    family: qwen3.6                # override header detection, rarely needed
    profile: reasoning
    profiles:
      coding: {sampling: {min_p: 0.05}}              # reshape @coding for this model
    overrides: {sampling: {thinking_budget: 2048}}   # per-model cap overrides profile
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

Several models stay resident at once, the pinned models plus a pool bounded
by `server.budget_gb`. Each model's memory use is its GGUF size, because the
weights map from the file without a copy. Four states govern the pool.

| State | Set by | Unloads when |
|-------|--------|--------------|
| pinned | `pin: true`, `--pin` | never |
| kept | `POST /v1/keep`, `launch --model`, a talk session | the budget is full and it is least recently used |
| idle | any request | `ttl_s` seconds pass with no request, or the budget needs the room |
| preloaded | `defaults.preload` | as idle |

A model is never unloaded mid-generation, because the idle timer waits for its
last request to finish. Keeping is what `gmlx launch --model` asks for, so
that a coding session's model survives the idle periods between turns
without reserving budget. `{"keep": false}` on the same endpoint releases
it and `POST /unload` evicts it.

A GGUF served under two ids is a single resident copy unless a setting that
changes how the model is loaded differs. Those settings are the `load` keys,
`mmproj`, `draft_gguf`, `speculative`, `adapter`, `chat_template`, `stream`
and the streaming settings. Sampling, `system` and `ttl_s` never create a
second copy. How a streamed model is counted against the budget is in
[streaming.md](streaming.md#residency-of-a-streamed-model).

## Reloading the config

A server started from a config file re-reads it on `POST /v1/reload` or
`SIGHUP`. The file is parsed again and the model registry rebuilt. Resident
models whose load settings are unchanged stay loaded. A config with no models
serves, which lets you start empty and add models by reloading.

The commands that rewrite the config, `init`, `sync-models`, `pull` and
`rm`, signal a running server started from that file unless you pass
`--no-reload`. A single-model server records no config path and is never
signalled.

Adding or removing `mmproj`, `draft_gguf` or `speculative` on a model takes
effect on that model's next cold load. A resident copy keeps its load
settings until it is evicted or unloaded.

## Family defaults

The built-in sampling for each family and the intents each family defines,
as `gmlx profiles` prints them. Column one is the family, column two the
GGUF architectures it covers, and each intent is shown fully resolved, with
its base values repeated. `gmlx profiles <id>` shows a single model the same
way. The reasoning level goes under three names, `reasoning_effort` for
gpt-oss, Hy3 and Hy4, `thinking_effort` for Kimi and `reasoning_strength`
for Muse, and the models whose thinking markers are not `<think>` carry them
here as well. Each value is cited to its model card in
`gmlx/gen/profiles.py`.

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

The `default` row is the fallback for architectures no family claims.
