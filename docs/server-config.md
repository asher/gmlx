# Configuration keys

This page lists every key that `gmlx.yaml` accepts, under the block that
holds it. Each key has its own heading with its full path, so you can
search for it or link to it. What the file is for, and how models, profiles and requests combine, is
in [Configuration file](config.md).

A path such as `models.*.pin` means the `pin` key of any entry under
`models`. The keys of `sampling`, `load` and `cache` go wherever a profile
or a model entry accepts those blocks, so they are listed once under their
own headings.

- [models](#models)
- [aliases](#aliases)
- [profiles](#profiles)
- [sampling](#sampling)
- [load](#load)
- [cache](#cache)
- [rules](#rules)
- [discover](#discover)
- [server](#server)
- [talk](#talk)
- [assistant](#assistant)
- [theme and themes](#theme-and-themes)
- [Complete example](#complete-example)

## models

One entry per model. The key of each entry is the model's id, which
requests, `/v1/models` and the command line use to name it. An id cannot
contain `@`.

```yaml
models:
  qwen3.6-27b:
    path: Qwen3.6-27B-Q4_K_S.gguf
    speculative: true
    pin: true
  gemma-12b:
    path: gemma-4-12b-it-Q4_K_M.gguf
    mmproj: mmproj-gemma-4-12b-it-bf16.gguf
    profile: coding
```

An entry whose file is missing is skipped with a warning. It drops out of
`/v1/models` and a request for it gets 404, while the server keeps
running. `gmlx sync-models` removes such entries.

### `models.*.path`

The GGUF file of the model. It can be absolute, relative to a folder in
[`server.model_dirs`](#servermodel_dirs), or an
`hf:<org>/<repo>/<file.gguf>[@rev]` reference when
[`server.hf_cache`](#serverhf_cache) is on. For a model split into
shards, name the first shard. This key is required.

### `models.*.profile`

The profile a request gets when it names none. It may be a profile under
[`profiles`](#profiles) or a built-in intent such as `coding`. A request
that adds `@name` to the model id uses that profile instead. The default
is no profile.

### `models.*.profiles`

Changes to a profile for this model only, keyed by the profile or intent
name. Each value takes the same keys as a profile and applies only when
its name is the profile selected for the request. The default is none.

```yaml
models:
  qwen3.6-27b:
    path: Qwen3.6-27B-Q4_K_S.gguf
    profiles:
      coding: {sampling: {min_p: 0.05}}
```

### `models.*.overrides`

Settings that win over every profile, for this model only. It takes the
blocks `sampling`, `load` and `cache` and the keys `system`,
`chat_template`, `chat_template_kwargs`, `thinking` and
`reasoning_effort`. Fields in the request still win over it. The default
is none.

### `models.*.family`

The family whose [defaults](family-defaults.md) the model starts from,
instead of the family detected from the GGUF header. Set it only when
detection picks the wrong family. The default is the detected family.

### `models.*.mmproj`

The vision or audio projector GGUF that makes this a multimodal model, as
[Vision and audio](vlm.md) describes. A relative path is searched in
`server.model_dirs`. The default is none.

### `models.*.speculative`

Speculative decoding with the GGUF's own MTP head, or with
[`draft_gguf`](#modelsdraft_gguf) when that is set. Output stays
token-identical to plain decoding. `gmlx init` and a
[`discover`](#discover) scan set it for models with a head. The default
is `false`.

### `models.*.draft_gguf`

A separate [drafter](glossary.md) GGUF that proposes tokens for this
model. Setting it turns on `speculative`. A drafter next to its model is
paired automatically by `gmlx init` and `discover`. The default is none.

### `models.*.native_mtp`

Draft with the GGUF's own MTP head even when `draft_gguf` is set. The
default is `false`.

### `models.*.speculative_width_cap`

The largest number of requests that may generate together while the model
still speculates. Checking a draft widens every request's weight reads,
which costs little for one request and a lot for several, so past the cap
the batch switches to plain decoding and speculates again once it shrinks.

`null` takes the drafter's measured default. That is no cap for a native
head on a dense Qwen model, `2` for the Gemma assistant drafter and for
families without a measurement, and `1` for every mixture-of-experts model
and for the drafters that handle one sequence at a time, such as Hy3,
DeepSeek-V4, Muse, Qwen3.8-Flash-Next and GLM5-next. `0` removes the cap,
except on a single-sequence drafter, which stays at `1` whatever the
value. The default is `null`.

The switch is described in
[Speculative batching](internals/speculative-batching.md), and the
measurements behind the defaults in
[Performance tuning](performance.md#mtp-speculative-decoding).

### `models.*.adapter`

A GGUF LoRA adapter applied at load. Two ids with the same `path` and
different adapters share the base weights, as
[LoRA adapters](lora.md#serving-one-base-with-many-adapters) describes.
The default is none.

### `models.*.pin`

Keep the model loaded for the life of the server. A pinned model is the
first to load at start and is never unloaded. The default is `false`.

### `models.*.ttl_s`

Seconds without a request before this model unloads, in place of
[`server.defaults.ttl_s`](#serverdefaultsttl_s). `null` or `0` never
unloads it. The default is the server-wide value.

### `models.*.stream`

Run a model that does not fit in memory. `experts` streams the routed
experts of a mixture-of-experts model from disk and keeps the rest of the
model on the GPU. `cpu` runs the whole model on the CPU. The load refuses
`stream` on a speculative model, and `stream: cpu` on a multimodal model.
The old key `cpu_moe` is read as `stream`, with a warning. The default is
no streaming.

[Models larger than memory](streaming.md) explains how to choose between
the two and how to size the lossy keys below. Every key from here to
`stream_fast_disk` needs `stream`.

### `models.*.moe_experts`

A fixed number of experts per token on a streamed model, below the number
the model was trained with. A positive integer. This changes the output.
The default is the model's own count.

### `models.*.moe_expert_mass`

Keep, for each token, the smallest set of experts whose gate weights add
up to this share of the total. A number above 0 and at most 1. This
changes the output. The default is off.

### `models.*.moe_miss_shed`

Drop the experts of a token that are not in the decode
[arena](glossary.md), as long as the kept experts still cover this share
of the gate weight. A number above 0 and at most 1. This changes the
output. The default is off.

### `models.*.moe_layer_shed`

Skip the routed experts of a streamed layer with this probability. A
number between 0 and 1, exclusive. This changes the output. The default is
off.

### `models.*.moe_prestage`

Which experts are read ahead of need. `ranked` reads the predicted
experts, and `keepers` reads only those that `moe_miss_shed` would keep,
so it needs that key. The default is `ranked`.

### `models.*.prefill_feeder`

Read expert weights for prefill directly from the GGUF on a streamed
model. The default is `true`.

### `models.*.decode_feeder`

Decode from a wired arena that keeps the most used experts in memory. The
default is `true` under `stream: experts`.

### `models.*.stream_fast_disk`

How streamed decoding reads ahead. `auto` tests the drive, and `on` and
`off` force the choice. The default is `auto`.

## aliases

Second names for models. Each maps a name to a model id, optionally with a
profile after `@`.

```yaml
aliases:
  fast: gemma-12b
  coder: qwen3.6-27b@coding
```

`/v1/models` lists each alias as its own entry with `alias_of` set, which
lets a client that picks models from a menu choose a profile too. An alias
may not contain `@` or match a model id, and its target must exist.

## profiles

Named sets of settings that a model entry, a rule or a request selects.
Each key under `profiles` is a profile name. Besides your own profiles,
every model has the built-in intents `@coding`, `@instruct`, `@creative`,
`@reasoning-low`, `@reasoning-medium`, `@reasoning-high`,
`@reasoning-max` and `@reasoning-xhigh`, each with the values that the
model's family publishes, as listed in [Family defaults](family-defaults.md).
A key that names a profile, such as `extends`, names an intent without
the `@`. A profile of your own with an intent's name replaces the intent.

```yaml
profiles:
  review:
    extends: coding
    system: "You are a terse code reviewer."
    sampling: {max_tokens: 2048}
    load: {kv_bits: 8}
  agent:
    chat_template_kwargs: {preserve_thinking: true}
```

An unknown profile name in a model entry, a rule or `extends` fails the
load, and so does a cycle of `extends`.

### `profiles.*.extends`

A profile or intent applied first, whose keys this profile then changes.
With an intent, each model starts from the intent as its own family
defines it. The default is none.

### `profiles.*.sampling`

Default values for the request fields listed under [sampling](#sampling).
A field that the request sends wins. The default is none.

### `profiles.*.load`

How the model is built, with the keys listed under [load](#load). Two ids
that differ in a load key are two loaded copies of the model. The default
is none.

### `profiles.*.cache`

Prompt cache settings, with the keys listed under [cache](#cache). They
change [`server.cache`](#servercache) for the models that use this
profile. The default is none.

### `profiles.*.system`

A system prompt, used only when the request has no system message. The
default is none.

### `profiles.*.chat_template`

A chat template that replaces the one in the GGUF, as inline Jinja or as
a path to a `.jinja` or `.txt` file. The template applies at load, so two
ids with different templates are two loaded copies, and a multimodal
model keeps the template of its `mmproj`. The default is the GGUF's
template.

### `profiles.*.chat_template_kwargs`

Variables passed to the chat template on each request. The most useful is
`preserve_thinking` on Qwen3.6 and recent Gemma templates, which keeps
earlier `<think>` blocks in the prompt so that an agent sees its earlier
reasoning. Keys the request sends win. The default is none.

### `profiles.*.thinking`

Turn reasoning `on`, `off` or `adaptive`, on any model. gmlx maps the
value to the variable that the model's template reads, so one profile
works across families:

| Family | Template variable | Values |
|--------|-------------------|--------|
| Qwen3.x, GLM | `enable_thinking` | `true`, `false` |
| MiniMax-M3 | `thinking_mode` | three states, so `adaptive` is accepted |
| Kimi K2.x | `thinking` | `true`, `false` |
| Hy3 | `reasoning_effort` | levels including `no_think` |
| gpt-oss | `reasoning_effort` | `low`, `medium`, `high`. Reasoning cannot be turned off |

A request's `enable_thinking` wins, then its `thinking` or
`reasoning_effort` field, then the profile, then the template's default.
A request's `thinking` also accepts the z.ai form, `{"type": "enabled"}`
or `{"type": "disabled"}`, whose optional `clear_thinking: false` keeps
earlier reasoning in the prompt. An entry in `chat_template_kwargs` passes
through unchanged and wins over this key. `gmlx run` and `gmlx chat` take
`--thinking`. The default is the template's own.

### `profiles.*.reasoning_effort`

A reasoning level that the model's template accepts, such as `low`,
`medium` or `high`, mapped to the template's variable in the same way as
`thinking`. `gmlx run` and `gmlx chat` take `--reasoning-effort`. The
default is the template's own.

## sampling

The keys of a `sampling` block, in a profile or in a model's `overrides`.
Each is also a request field of the same name, and a field in the request
wins. The defaults marked "family" come from the model's
[family defaults](family-defaults.md).

```yaml
profiles:
  precise:
    sampling:
      temperature: 0.2
      top_p: 0.9
      max_tokens: 4096
      stop: ["</answer>"]
```

### `sampling.temperature`

How random the choice of each token is. `0` always takes the most likely
token. `temp` is accepted as another name. The default is family.

### `sampling.top_p`

Sample only from the most likely tokens whose probabilities add up to
this value. `0` turns the filter off. When `top_p` is the only filter, it
considers at most the 1024 most likely tokens, so that sorting stays
batched. The default is family.

### `sampling.top_k`

Sample only from this many of the most likely tokens. `0` turns the
filter off. The default is family.

### `sampling.min_p`

Drop tokens whose probability is below this share of the most likely
token's. `0` turns the filter off. The default is family.

### `sampling.max_tokens`

The most tokens to generate. The default is no limit.

### `sampling.seed`

A seed that makes the request's sampling repeatable, without changing the
other requests in its batch. Two runs give the same tokens only when their
batches and the speculation setting also match, because a different batch
shape changes the logits slightly. The default is none.

### `sampling.stop`

A string or a list of strings that end generation when generated. A stop
string can end generation in the middle of a token, and the response then
has `finish_reason: "stop"`. It applies to chat completions, and the
Anthropic endpoint uses its own `stop_sequences`. The default is none.

### `sampling.repetition_penalty`

Lower the chance of tokens that appeared in the last
`repetition_context_size` tokens. `1` has no effect. The default is off.

### `sampling.repetition_context_size`

How many recent tokens `repetition_penalty` looks at. The default is `20`.

### `sampling.presence_penalty`

Lower the chance of every token that was already generated, by the same
amount. The default is off.

### `sampling.frequency_penalty`

Lower the chance of each generated token in proportion to how often it
was generated. The default is off.

### `sampling.xtc_probability`

The probability of applying XTC sampling to a token, which removes the
most likely candidates to vary the text. XTC never removes newline or
end-of-sequence tokens, and it is not available on speculative models.
The default is off.

### `sampling.xtc_threshold`

The probability above which XTC removes a candidate. The default is none.

### `sampling.enable_thinking`

Whether the chat template opens a thinking block. The
[`thinking`](#profilesthinking) key of a profile sets this on any
family. The default is the template's own.

### `sampling.thinking_budget`

The most reasoning tokens before gmlx closes the thinking block. The count
starts when a thinking block opens, whether the template or the model
opens it. On a speculative model the block closes at the end of a draft
round, so the budget can be exceeded by one round. A request that runs in
a batch, or resumes after preemption, has no budget, and a speculative
model with a separate drafter refuses the key. `gmlx run` and `gmlx chat`
honor it too. The default is no limit.

### `sampling.thinking_start_token`

The text that opens the model's reasoning. Family defaults set it for
models that do not use `<think>`. The default is `<think>`.

### `sampling.thinking_end_token`

The text that closes the model's reasoning. The default is `</think>`.

## load

The keys of a `load` block, which change how a model is built. Two ids
that differ in a load key are two loaded copies of the model. Each key
also has a [`gmlx serve` flag](cli.md#gmlx-serve) for a server started
with one GGUF, and an [environment variable](env-vars.md#load-and-cache-keys)
that sets it for every model in the process.

`prefill_step_size` and `dtype` are not load keys. They apply to the whole
server, under [Scheduling](#scheduling).

### `load.kv_bits`

Quantize the KV cache to this many bits, which lets a long context use
less memory. `uniform` accepts 2, 3, 4, 6 and 8, and `kvarn` accepts 2, 3,
4, 5, 6 and 8. The server decides for each layer whether its cache
quantizes, and logs the result in a `[kv]` line. Ordinary attention
layers quantize, except the last layer of a deep stack, and sliding-window
and recurrent state stay fp16. `/v1/models` reports the result for each
loaded model as a `kv_quant` object, as the [HTTP API](api.md#endpoints)
describes. The default is no quantization, and with `kvarn` it is `6`.

### `load.kv_quant_scheme`

How the KV cache quantizes. `uniform` is affine quantization. `kvarn`
normalizes the variance first and keeps the newest tokens in fp16. A model
where no layer converts under `kvarn` runs fp16 and logs why, and never
falls back to affine. Under `uniform`, speculative models quantize only
while they serve one request. Under `kvarn` they stay quantized at any
batch size with mlx-kquant 0.4.9 or later. Which architectures convert is
in [Performance tuning](performance.md#kv-cache-quantization). The default
is `uniform`.

### `load.kv_group_size`

The group size of `uniform` quantization. The default is `64`.

### `load.kv_tail_tokens`

Under `kvarn`, how many of the newest tokens stay fp16. A multiple of 128.
The default is `1024`.

### `load.quantized_kv_start`

How many tokens at the start of the cache stay unquantized. `kvarn`
ignores it. The default is `0`.

### `load.max_kv_size`

The largest context a request may use on this model, in tokens. On the
server this is a budget only. `gmlx run` and `gmlx chat` also make the
cache a rotating window of this size. The default is no cap.

## cache

The keys of a `cache` block, which set the prompt cache. The cache keeps
the computed prefix of recent prompts, so a request that starts the same
way skips that part of prefill. [`server.cache`](#servercache) sets them
for every model, and a profile or a model's `overrides` changes them. What
the cache restores for each architecture is in
[Performance tuning](performance.md#the-prompt-cache).

```yaml
server:
  cache:
    enabled: true
    disk: {path: ~/.cache/gmlx/apc, max_gb: 100}
```

### `cache.enabled`

Turn the prompt cache on. `gmlx init` writes `true`. The default is
`false`.

### `cache.block_size`

Tokens per cache block. When unset, the server picks the smallest of 16,
32, 64, 128 and 256 that lets the pool fill its memory share without
passing the Metal limit on the number of buffers. The default is chosen
per model.

### `cache.num_blocks`

Blocks in the pool that all cached prompts share. When unset, the pool
starts at 2048 blocks and grows after the model loads until it could fill
half of the memory left over, within the Metal limit on the number of
buffers. Blocks cost memory only once they hold a prompt. A full pool drops the oldest prompts. Set a number to fix the size.
The default is chosen per model.

### `cache.exact_entries`

How many whole-prompt copies of the cache to keep for hybrid and recurrent
models, which cannot cache in blocks. Each copy is a full cache, so a
higher value uses more memory. With the default, a third conversation can
start without evicting the first. The default is `4`.

### `cache.hash`

How blocks are identified. `fast` uses Python's hash, which is stable
only within one process. `sha256` is stable across processes and costs
more per token. The default is `fast`.

### `cache.disk`

A second cache tier on the SSD, so that a prompt survives the model
unloading or the server restarting. `true` puts the tier at
`~/.cache/gmlx/apc`, and a mapping sets the keys below. The default is
`false`.

### `cache.disk.path`

The folder of the SSD tier. Setting it turns the tier on. The default is
`~/.cache/gmlx/apc` when `disk` is `true`.

### `cache.disk.max_gb`

The most space each namespace may use, in GB. With the default namespace
each model has its own, so the total can reach this times the number of
models. The default is no limit.

### `cache.disk.namespace`

The partition on disk that a model reads and writes. The default is the
model path, which gives each model file its own partition.

### `cache.disk.workers`

Threads that write entries to disk. The default is `1`.

### `cache.disk.read_mode`

How entries are read back. `direct` reads the needed bytes with plain file
reads, and `mmap` maps the files. The default is `direct`.

## rules

A list of patterns, each giving a profile to the model ids it matches. The
first match wins. A rule ranks below a model's own `profile` and above
[`server.defaults.profile`](#serverdefaultsprofile).

```yaml
rules:
  - {match: "*coder*", profile: review}
  - {match: "qwen3.6-*", profile: agent}
```

### `rules[].match`

A pattern with shell wildcards such as `*`, not a regular expression.
This key is required.

### `rules[].profile`

The profile for the matching ids. This key is required.

## discover

Folders that the server scans at each start. Every GGUF found joins the
models without an entry of its own, named after its file. The scan reads
only the headers.

```yaml
discover:
  - dir: ~/models
    recursive: true
```

The name of each model comes from its file. The scan removes the shard
suffix, markers such as `mmproj`, `assistant`, `draft` and `mtp`, and
imatrix tags, and adds the quantization in short form, such as `-q4`.
When two files would get the same name, both get the full quantization,
such as `-q4-k-m` and `-q4-k-s`, and a remaining clash gets a number. The
server prints the names at start.

### `discover[].dir`

The folder to scan. `null` scans every folder in `server.model_dirs`. The
default is `null`.

### `discover[].recursive`

Scan subfolders too. The default is `false`.

### `discover[].pair_mmproj`

Pair each `mmproj*.gguf` with the model in the same folder that it
matches. The default is `true`.

### `discover[].speculative`

Turn on speculative decoding for the models found. `auto` and `true` turn
it on for models with an MTP head, and `false` never does. A drafter GGUF
in the same folder becomes a model's `draft_gguf` when their architecture,
hidden size and file name agree, and a drafter whose header names its base
model pairs only with that model. Streamed models get no drafter. The default is
`auto`.

## server

Settings of the server process. Every key has a default, and a file with
no `server` block runs a server on `127.0.0.1:8080` with the prompt cache
off.

```yaml
server:
  host: 127.0.0.1
  port: 8080
  model_dirs: [~/models]
  cache: {enabled: true}
  defaults:
    ttl_s: 900
    model: qwen3.6-27b
```

### Bind and auth

#### `server.host`

The address the server listens on. An address other than loopback needs
`api_key` or `no_auth`. The default is `127.0.0.1`.

#### `server.port`

The port the server listens on. The default is `8080`.

#### `server.api_key`

A key that every request must present, as `Authorization: Bearer <key>`
or `x-api-key: <key>`. `/health` and CORS preflight requests pass without
it. The server reads its key only from this file, never from a flag or
the environment, so the key stays out of process listings and shell
history. The client commands `ps`, `status`, `launch` and `menubar` take
`--api-key` to present it. The default is no key.

A loopback server refuses a request whose `Host` header is not a loopback
name, which blocks DNS rebinding, and answers CORS with `*` and no
credentials.

#### `server.no_auth`

Allow an address other than loopback without a key, when a proxy in
front of the server handles authentication. The default is `false`.

### Paths

#### `server.model_dirs`

Folders searched, in order, for a relative `path`, `mmproj`, `draft_gguf`
or `adapter`, and scanned by a [`discover`](#discover) entry with no
`dir`. Entries expand `~` and `$VAR`. A path found in none of them fails
the load with the folders searched. `gmlx pull` downloads into the first
one. The default is none.

#### `server.hf_cache`

Let a model `path` be an `hf:<org>/<repo>/<file.gguf>[@rev]` reference,
resolved from the local Hugging Face cache. The Hugging Face libraries
then run offline, and the server never downloads. `gmlx init
--from-hf-cache` and `gmlx sync-models --from-hf-cache` write such
entries and set this key. How the server treats a request that names a
Hugging Face model is in the [HTTP API](api.md#hugging-face-policy). The
default is `false`.

### Memory and residency

How models share memory is in
[Which models stay loaded](config.md#which-models-stay-loaded).

#### `server.budget_gb`

The memory, in GB, that loaded models may use together. The default is
0.8 times the GPU working set that macOS recommends.

#### `server.max_models`

The most models loaded at once, checked after the budget. The default is
no limit.

#### `server.cache_limit_gb`

The limit, in GiB, of the MLX buffer cache, which keeps freed GPU buffers
for reuse. At deep context it can grow to tens of GB. When unset, the
server sets a limit of 4 to 12 GiB when the largest model leaves little
memory spare, and none otherwise. A negative value never limits it. When
to change it is in
[Performance tuning](performance.md#the-mlx-buffer-cache-at-deep-context).
The default is chosen at start.

#### `server.defaults.model`

The model for a request that names none. With a single model in the
file, that model. It is also the first model the server loads when no
model is pinned. The default is none.

#### `server.defaults.ttl_s`

Seconds without a request before a model that is not pinned unloads.
`null` or `0` never unloads. The default is `900`.

#### `server.defaults.preload`

More models to load at start, after the first, one at a time. `all`
loads every model. Preloaded models unload like any idle model. The
default is none.

#### `server.defaults.profile`

A profile for every model, below rules and the model's own profile. The
default is none.

### Scheduling

These keys apply to every model, and each has a
[`gmlx serve` flag](cli.md#gmlx-serve).

#### `server.prefill_step_size`

Tokens per prefill chunk. A lower value lowers the memory peak of a long
prompt and slows prefill. The default is `2048`.

#### `server.dtype`

The type of the activations, the unquantized weights and the KV cache.
`auto`, `bfloat16` or `float16`. `auto` picks `float16` on M1 and M2,
which lack bfloat16 arithmetic, and `bfloat16` on later chips. The weights
on disk never change. The default is `auto`.

#### `server.decode_prefill_ratio`

How the prefill of a new request shares the GPU with requests that are
generating. `auto` slows prefill only when a generating request would
drop below half its speed. A number such as `1.0` makes each prefill
chunk wait until the generating requests have had that multiple of the
chunk's GPU time, and `0` runs one prefill chunk per generation step. The effect is measured in
[Performance tuning](performance.md#serving-concurrent-requests). The
default is `auto`.

#### `server.prefill_tick_ms`

The time, in milliseconds, one prefill chunk may take while other
requests generate. Chunks halve until they fit, and `0` never halves
them. The default is `500`.

#### `server.token_queue_timeout_s`

Seconds to wait for the next token before the request fails. `0` waits
forever. A request that times out is cancelled and recorded as
`last_error` in `/v1/metrics`, and a streaming client receives a final
error event. The usual cause is a long prompt on a model larger than
memory. On `/v1/systemone` it limits a whole decision. The default is
`1800`.

### Features

#### `server.family_defaults`

Start each request from its model's [family defaults](family-defaults.md),
and offer the built-in intents. `false` turns off both. The default is
`true`.

#### `server.stochastic_mtp`

Accept speculative tokens by rejection sampling, which accepts more of
them. The sampling distribution stays exact, but output is no longer
token-identical to plain decoding. Greedy requests are unaffected, and a
reload does not change the key. The gain is in
[Performance tuning](performance.md#stochastic-acceptance). The default is
`false`.

#### `server.gpu_keepwarm`

Keep the GPU clock up between tokens for every model. Streamed models with
a decode feeder do this already, as
[Models larger than memory](streaming.md#the-lossless-settings) describes.
The default is `false`.

#### `server.menubar`

Let a server started in the background open the macOS
[menu bar app](menubar.md). The default is `true`.

#### `server.cache`

The prompt cache settings for every model, with the keys listed under
[cache](#cache). A profile or a model's `overrides` changes them. The
default is the cache off.

### Services

Each service adds an endpoint, and
[Speech, embeddings and rerank](services.md) lists the models each one
accepts. A service whose model is missing is turned off with a warning,
and the server still starts.

#### `server.stt`

The speech-to-text model, as an alias such as `whisper-turbo`, a Hugging
Face repo id, a folder, or `true` for the default. The default is none.

#### `server.tts`

The text-to-speech model, as an alias such as `kokoro`, a Hugging Face
repo id, a folder, or `true` for the default. The default is none.

#### `server.embeddings`

The embeddings model, as a GGUF path, an `hf:` reference, an alias, or
`true` for the default. The default is none.

#### `server.rerank`

The reranker, as a Qwen3-Reranker GGUF path, an `hf:` reference, an alias,
or `true` for the default. The default is none.

### Structured decisions

The settings of `POST /v1/systemone`, which [Structured decisions](decisions.md)
describes. An unknown key, or a `model` that is not a configured id or
alias, fails the load. A reload applies new values to the next request.

#### `server.systemone.model`

The model id or alias, optionally with `@profile`, for a request whose
`model` is absent or names nothing configured. The default is none.

#### `server.systemone.canvas`

The most tokens one read's canvas holds. A positive multiple of 16,
capped at the model's own canvas length. The default is `64`.

#### `server.systemone.constrained`

Compute probabilities only over the answer tokens of a read. With `false`
the full vocabulary gives the same one-step answer probabilities, and
different entropies and multi-step reads. The default is `true`.

#### `server.systemone.max_questions`

The most questions one request may ask. A request with more gets a 422.
The default is `64`.

#### `server.systemone.max_samples`

The most `samples` and `auto_max` a request may ask for. A larger value is
lowered to this one. The default is `32`.

#### `server.systemone.think`

The thought budget, in tokens, for a request that does not set `think`.
From 0 to 4096, or `"auto"` to think only when an answer is unsure. The
default is `0`.

#### `server.systemone.think_threshold`

The confidence below which `"auto"` thinks, for a request that does not
set it. The default is `0.8`.

#### `server.systemone.think_budget`

The thought budget of `"auto"`, for a request that does not set it. The
default is `64`.

### Served assistants

Assistants served as models, which run the tool loop of the
[assistant](#assistant) on the server. A client names an assistant as its
model and gets tools without a loop of its own, as
[Assistant](assistant.md#served-assistants) describes.

```yaml
server:
  assistants:
    helper:
      model: qwen3.6-27b
      mcp: []
```

#### `server.assistants.*.model`

The configured model that answers for this assistant. This key is
required.

#### `server.assistants.*.memory`

Give this assistant long-term memory, in one store shared by all its
clients. The default is `false`.

#### `server.assistants.*.mcp`

The tool servers of this assistant, with the keys of
[`assistant.mcp`](#assistantmcp). `null` uses `assistant.mcp`, and `[]`
gives it no tools. An assistant on a server beyond loopback must set this
key. The default is `null`.

#### `server.assistant_allow_remote`

Allow served assistants on an address other than loopback. Their tools
run on the server host, so the server otherwise refuses to start with
them. The default is `false`.

## talk

Settings of the voice client, [`gmlx talk`](talk.md), and of voice
sessions in the menu bar app. Most keys have a
[`gmlx talk` flag](cli.md#gmlx-talk), which wins over the file.

```yaml
talk:
  model: qwen3.6-27b@instruct
  voice: af_heart
  mode: vad
  vad: {silence_ms: 450}
```

### `talk.model`

The model to talk to, as an id or alias, optionally with `@profile`. The
default is the server's default model.

### `talk.voice`

The voice, as a Kokoro preset or a Qwen3-TTS speaker name. The default is
the server's default voice.

### `talk.speed`

How fast replies are spoken, as a multiple of normal speed. The default is
`1.0`.

### `talk.system`

The spoken persona. When the key is absent, a prompt that asks for
speakable text without markdown is used. `null` or `""` sends no system
prompt at all, which is not the same as leaving the key out. The default
is the speakable-text prompt.

### `talk.language`

A language hint for speech recognition, such as `en`. The default is
automatic detection.

### `talk.max_tokens`

The most tokens in a spoken reply. The default is no limit.

### `talk.mode`

How listening starts. `wake` waits for the wake phrase, `vad` starts on
any speech, `ptt` uses Space as push-to-talk, and `text` takes typed
prompts and still speaks the replies. The default is `wake`.

### `talk.wake_word`

The wake phrase, any text. The default is `hey assistant`.

### `talk.wake_threshold`

How confident detection must be before the wake phrase counts, from 0 to
1. A higher value gives fewer false wakes. The default is `0.3`.

### `talk.push_to_talk_modifier`

The key held with Space for the menu bar hotkey. One of `globe`,
`right-command`, `right-option` and `control`. There is no flag for this
key. The default is `globe`.

### `talk.input_device`

The microphone, as part of a device name or an index. `gmlx talk` lists
devices with `/devices`. The default is the system input.

### `talk.output_device`

The speaker, as part of a device name or an index. The default is the
system output.

### `talk.chime`

Play a sound on wake and at the end of a turn. The default is `true`.

### `talk.brain`

What answers. `chat` is the plain model, and `assistant` adds the tools
and memory of the [assistant](#assistant) block. The default is `chat`.

### `talk.vad.threshold`

The speech probability above which audio counts as speech, from 0 to 1.
The default is `0.6`.

### `talk.vad.silence_ms`

The pause, in milliseconds, that ends an utterance. A shorter pause
answers sooner and cuts off more sentences. The default is `550`.

### `talk.vad.min_speech_ms`

The shortest utterance, in milliseconds, that is kept. Shorter sounds are
dropped as noise. The default is `300`.

### `talk.vad.pre_roll_ms`

Audio kept from before speech starts, in milliseconds. There is no flag
for this key. The default is `400`.

## assistant

Settings of the built-in assistant, which adds tools and long-term memory
to a model. `gmlx chat --assistant`, `gmlx talk` with `brain: assistant`,
and [served assistants](#served-assistants) use it. It does not affect the
coding agents that `gmlx launch` connects. [Assistant](assistant.md)
describes how the tool loop and memory work.

```yaml
assistant:
  max_tool_rounds: 8
  mcp:
    - name: files
      command: [npx, -y, "@modelcontextprotocol/server-filesystem", "~/notes"]
    - name: search
      url: http://127.0.0.1:8931/mcp
  memory:
    top_k: 6
```

### `assistant.max_tool_rounds`

The most rounds of tool calls in one turn, after which the model must
answer. At least 1. The default is `8`.

### `assistant.tool_timeout_s`

Seconds one tool call may take. At least 1. The default is `60`.

### `assistant.mcp`

The [MCP](glossary.md) servers that provide tools, as a list. Each entry
has a `name` and exactly one of `command` and `url`. A server that fails
to start gives a warning, and the assistant runs without its tools. The
default is none.

### `assistant.mcp[].name`

A name for the server, unique in the list. When two servers offer a tool
with the same name, each tool gets its server's name as a prefix. This key
is required.

### `assistant.mcp[].command`

The command that starts a server over stdio, as a list of arguments or a
string split like a shell command line. The server's log goes to
`~/.cache/gmlx/mcp-<name>.log`.

### `assistant.mcp[].url`

The address of a server over streamable HTTP.

### `assistant.mcp[].env`

Environment variables for a stdio server. The server gets only `HOME`,
`PATH`, `SHELL`, `TERM`, `USER` and `LOGNAME` from your environment, so a
token it needs must be set here. The default is none.

### `assistant.memory.enabled`

Remember facts across conversations. Memory needs
[`server.embeddings`](#serverembeddings), and without it the assistant
runs with a warning and no memory. The default is `true`.

### `assistant.memory.path`

The memory database file. The default is
`~/.local/share/gmlx/assistant-memory.db`.

### `assistant.memory.top_k`

How many remembered facts are added to each turn. At least 1. The default
is `4`.

### `assistant.memory.extract`

Store each exchange as short facts that the model extracts. `false` stores
the exchanges as they are. The default is `true`.

### `assistant.memory.ttl_days`

Forget facts older than this many days, at start. The default is never.

### `assistant.memory.max_items`

The most facts stored. When full, the oldest facts that were never
recalled go first. At least 1. The default is `20000`.

## theme and themes

Colors of [`gmlx chat`](chat.md).

```yaml
theme: my-black
themes:
  my-black:
    extends: dark
    heading: {bold: true, rgb: "#88c0d0"}
```

### `theme`

The theme each chat starts with, a built-in name or one from `themes`.
`--theme` and `/theme` change it. The default is `dark`.

### `themes`

Themes of your own, keyed by name. A theme with a built-in's name
replaces it. Each theme sets styles for kinds of text, such as `heading`
and `thinking`, and takes `extends` for the slots it leaves out. The
slots and style keys are listed under [Themes](chat.md#themes). The
default is none.

## Complete example

A file with every block, which loads without errors.

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
