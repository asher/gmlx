# Configuration

`gmlx.yaml` lists the models gmlx knows by name, the settings each model
runs with, and the server's own settings. `gmlx serve` reads it at start.
The other commands read it to turn a model name into a GGUF file, so a name
that works in a request also works with `gmlx run` and `gmlx chat`. Every key has a default, so a file needs only the keys whose
defaults you want to change.

## Create the file

`gmlx init` scans a folder of GGUF files and writes a config with one entry
for each model it finds:

```sh
gmlx init --models-dir ~/models
```

The file goes to `~/.config/gmlx/gmlx.yaml`. With no flags on a terminal,
`init` asks its questions as a wizard instead. The scan reads only the GGUF
headers, so it takes seconds even for a large folder. The result holds the
server settings you chose and the models it found:

```yaml
# gmlx configuration, written by `gmlx init`.
# Every key, with its default: https://asher.github.io/gmlx/config.html
# The settings a server runs with: gmlx serve --print-config
# The sampling each model starts from: gmlx profiles

server:
  host: 127.0.0.1
  port: 8080
  model_dirs:
    - ~/models
  defaults:
    ttl_s: 900
  cache:
    enabled: true

models:
  # sampling (gemma): t=1 top_p=0.95 top_k=64
  gemma-4-12b-it-q4:
    path: gemma-4-12b-it-Q4_K_M.gguf
  # sampling (qwen3.6): t=1 top_p=0.95 top_k=20
  qwen3.8-27b-ud-q6:
    path: Qwen3.8-27B-UD-Q6_K.gguf
    speculative: true
```

Above each model, a comment shows the sampling values it starts from,
which come from its [family defaults](family-defaults.md) and from the
GGUF itself. Start the server with `gmlx serve`, which finds the file on
its own.


## Where gmlx looks

A command that needs the file uses the first one it finds:

1. `~/.config/gmlx/gmlx.yaml`, where `gmlx init` writes.
2. `~/.gmlx.yaml`.

gmlx never reads a `gmlx.yaml` in the current directory. A config file can
name commands that the server runs on your Mac. A cloned repository can
hold such a file, and so can a folder that a
[container](launch-container.md) client writes.

When `./gmlx.yaml` exists and neither file above does, commands print
`gmlx no longer reads ./gmlx.yaml. Move it to ~/.config/gmlx/gmlx.yaml to use it.`
The line stops once you move the file.

The commands that take `--config FILE`, such as `gmlx serve`, read that file
instead of searching. When no file exists, `gmlx serve` does not start and
says to run `gmlx init`.

A command reads only the first file it finds and takes nothing from the
others. A new `~/.config/gmlx/gmlx.yaml` that holds only a `launch` block
makes the server stop reading `~/.gmlx.yaml` and lose its models, so add
the block to the file you already have.

To see the configuration a server would run with, including every default
you did not set, run `gmlx serve --print-config`. It prints the result as
YAML and exits without loading a model.


## What the file contains

The top level of the file is a YAML mapping of these blocks. Every block
is optional, and a server needs models from `models` or from a `discover`
scan.

| Block | What it does |
|-------|--------------|
| [`models`](#models) | It lists the models gmlx can run, one entry each, with its GGUF file and its own settings. |
| [`aliases`](#aliases) | It gives models extra names, such as `coder` for `qwen3.8-27b-ud-q6@coding`. |
| [`profiles`](#profiles) | It defines named sets of sampling, loading and prompt settings that any model can use. |
| [`rules`](#rules) | It gives a profile to every model whose id matches a pattern, such as all ids that contain `coder`. |
| [`discover`](#model-discovery) | It names folders that the server scans at each start, adding every GGUF it finds without an entry. |
| [`server`](#server) | It sets where the server listens, its API key, the model folders, how much memory models may use, and optional services. |
| [`talk`](#voice) | It sets the voice client's model, voice, wake phrase and listening thresholds. |
| [`assistant`](#assistant) | It gives the built-in assistant its tool servers and long-term memory. |
| [`launch`](#launch) | It sets how `gmlx launch` runs clients in containers. |
| [`theme`, `themes`](#chat-themes) | They set the colors of the terminal chat. |

A key is named by its full path, such as `server.port`. A path such as
`models.*.pin` means the `pin` key of any entry under `models`. After you
edit the file, a running server reads it again as
[Changing the file](#changing-the-file) describes.


## Models

Each entry under `models` is one model gmlx can run. The entry's key is
the model's id. A request names the model by its id in the `model`
field, `/v1/models` lists it, and `gmlx run` and `gmlx chat` take it on
the command line. `gmlx init` names each model after its file with the
quantization in short form, such as `qwen3.8-27b-ud-q6`, and you can rename
an entry freely. An id cannot contain `@`, because `@` separates an id
from a profile name.

```yaml
models:
  qwen3.8-27b-ud-q6:
    path: Qwen3.8-27B-UD-Q6_K.gguf
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

This is the model's GGUF file. The path can be absolute, relative
to a folder in [`server.model_dirs`](#servermodel_dirs), or an
`hf:<org>/<repo>/<file.gguf>[@rev]` reference. Such a reference resolves
from the local Hugging Face cache or from the `gmlx pull` folders under
`model_dirs`. This key is required.

A relative path that no `model_dirs` folder holds is taken from the folder
the server runs in, which is the config file's folder for a server that
gmlx starts in the background. For a model split into shards, name the
first shard.

### `models.*.profile`

A request that names no profile gets this one. It can be a profile under
[`profiles`](#profiles) or a built-in intent such as `coding`. A request
that adds `@name` to the model id uses that profile instead. The default
is no profile.

### `models.*.profiles`

This key changes a profile for this model only. Its entries are keyed by
the profile or intent name. Each value takes the same keys as
[`overrides`](#modelsoverrides) and applies only when its key names the
request's profile. The default is none.

```yaml
models:
  qwen3.8-27b-ud-q6:
    path: Qwen3.8-27B-UD-Q6_K.gguf
    profiles:
      coding: {sampling: {min_p: 0.05}}
```

### `models.*.overrides`

These settings win over every profile, for this model only. The key takes
the blocks `sampling`, `load` and `cache` and the keys `system`,
`chat_template`, `chat_template_kwargs`, `thinking` and
`reasoning_effort`. Fields in the request still win over it. The default
is none.

### `models.*.family`

The model starts from the [family defaults](family-defaults.md) for the
family named here, instead of the family detected from the GGUF header. Set it only when
detection picks the wrong family. The default is the detected family.

### `models.*.mmproj`

This vision or audio projector GGUF makes the model multimodal, as
[Vision and audio](vlm.md) describes. A relative path is searched in
`server.model_dirs`. The default is none.

### `models.*.speculative`

With `true`, the model uses speculative decoding with the GGUF's own MTP
head, or with [`draft_gguf`](#modelsdraft_gguf) when that is set. The
output stays token-identical to plain decoding. `gmlx init` and a
[`discover`](#model-discovery) scan set it for models with a head. The
default is `false`.

### `models.*.draft_gguf`

This separate [drafter](glossary.md#drafter) GGUF proposes tokens for the
model, and setting it turns on `speculative`. `gmlx init` and `discover`
pair a drafter with the model next to it. The default is none.

### `models.*.native_mtp`

With `true`, the model drafts with the GGUF's own MTP head even when
`draft_gguf` is set. The default is `false`.

### `models.*.speculative_width_cap`

The model speculates only while at most this many requests generate
together. Past the cap, the batch switches to plain decoding, and it
speculates again once the batch shrinks.
[Several requests at once](speculative-decoding.md#several-requests-at-once)
explains why speculation costs more as the batch grows.

`null` takes the drafter's default. A native head on a dense Qwen model
has no cap, and the Gemma assistant drafter and every family without a
cap of its own get `2`. Every mixture-of-experts model gets `1`, as do the
drafters that handle one sequence at a time. Those are the MTP heads of
Hy3, DeepSeek-V4, Muse Glimmer, Qwen3.8-Flash-Next and GLM-5.3-Flash, and
every DFlash drafter. `0` removes the cap, except on a single-sequence
drafter, which stays at `1` whatever the value. The default is `null`.

[Speculative batching](internals/speculative-batching.md) describes how a
batch switches between the two kinds of decoding.

### `models.*.adapter`

This GGUF LoRA adapter is applied at load. Two ids with the same `path`
and different adapters share the base weights, as
[LoRA adapters](lora.md#serving-one-base-with-many-adapters) describes.
The default is none.

### `models.*.pin`

With `true`, the model loads at start and stays loaded. Pinned models load
in file order, before the models in
[`server.defaults.preload`](#serverdefaultspreload). Neither the idle
timeout nor the budget unloads a pinned model. Only `POST /unload` does.
The default is `false`.

### `models.*.ttl_s`

This model unloads after this many seconds without a request, in place of
[`server.defaults.ttl_s`](#serverdefaultsttl_s). `null` or `0` turns off
the timeout. The default is the server-wide value.

### `models.*.stream`

This key runs a model that does not fit in memory. `experts` streams the
routed experts of a mixture-of-experts model from disk and keeps the rest
of the model on the GPU. `cpu` runs the whole model on the CPU. A
speculative model with `stream` loads streamed without speculation, and
the load refuses `stream: cpu` on a multimodal model. The old key `cpu_moe` is read as `stream`, with a
warning. The default is no streaming.

[Models larger than memory](streaming.md) explains how to choose between
the two and how to size the lossy `moe_*` keys. The `moe_*` keys, the two
feeder keys and `stream_fast_disk` apply only to a model with `stream`
set.

### `models.*.moe_experts`

A streamed model uses this fixed number of experts per token, below the
number the model was trained with. The value is a positive integer, and it
changes the output. The default is the model's own count.

### `models.*.moe_expert_mass`

For each token, the model keeps the smallest set of experts whose gate
weights add up to this share of the total. The value is above 0 and at
most 1, and it changes the output. The default is off.

### `models.*.moe_miss_shed`

The model drops a token's experts that are not in the decode
[arena](glossary.md#arena), as long as the kept experts still cover this
share of the gate weight. The value is above 0 and at most 1, and it changes
the output. The default is off.

### `models.*.moe_layer_shed`

The model skips a streamed layer's routed experts with this
probability. The value is between 0 and 1, exclusive, and it changes the
output. The default is off.

### `models.*.moe_prestage`

This key selects which experts are read before they are needed. `ranked` reads the
predicted experts. `keepers` reads only those that `moe_miss_shed` would
keep, so it needs that key. The default is `ranked`.

### `models.*.prefill_feeder`

With `true`, prefill on a streamed model reads expert weights directly
from the GGUF. The default is `true`.

### `models.*.decode_feeder`

With `true`, decoding uses a wired arena that keeps the most used experts
in memory. The default is `true` under `stream: experts`.

### `models.*.stream_fast_disk`

This key sets how streamed decoding reads ahead. `auto` tests the drive,
and `on` and `off` force the choice. The default is `auto`.


## Aliases

An alias gives a model a second name, and can fix a profile to it. The
server lists each alias in `/v1/models` as its own entry with `alias_of`
set, so a client that picks models from a menu can choose a model and a
profile together.

```yaml
aliases:
  fast: gemma-12b
  coder: qwen3.8-27b-ud-q6@coding
```

Each alias must not contain `@` or match a model id, and its target must
exist.


## Profiles

A profile is a named set of settings that any model can use. Every model
starts from its [family defaults](family-defaults.md), the sampling values
that its publisher recommends, so a new file needs no profiles at all.

Some publishers also recommend values for a task, such as a lower
temperature for code. gmlx offers these as built-in profiles called
intents. They are `@coding`, `@instruct`, `@creative`,
`@reasoning-low`, `@reasoning-medium`, `@reasoning-high`,
`@reasoning-max` and `@reasoning-xhigh`. A request selects one by adding
it to the model id:

```sh
gmlx run qwen3.8-27b-ud-q6@coding --prompt "Write a binary search in Go."
```

Profiles of your own go under `profiles`. A profile can hold
[sampling defaults](#sampling) such as
[`temperature`](#samplingtemperature), [load settings](#model-loading)
such as [KV cache quantization](#loadkv_bits), and
[prompt cache](#prompt-cache) settings. It can also hold a
[`system`](#profilessystem) prompt, a
[`chat_template`](#profileschat_template), and the reasoning controls
[`thinking`](#profilesthinking) and
[`reasoning_effort`](#profilesreasoning_effort).

A profile can start from an intent with [`extends`](#profilesextends) and
change only what it sets. A key that names a profile, such as `extends`,
names an intent without the `@`. A profile of your own with an intent's
name replaces that intent.

```yaml
profiles:
  review:
    extends: coding
    system: "You are a terse code reviewer."
    sampling:
      temperature: 0.3
      max_tokens: 2048
  agent:
    chat_template_kwargs: {preserve_thinking: true}
```

### How a request gets its settings

A model gets a profile from its own entry, from a matching rule, from the
server default, or from the request. When a setting comes from more than
one place, a later layer wins over an earlier one:

| Layer | Where it is set |
|-------|-----------------|
| Family defaults | They are built in for each detected family. |
| Server default profile | [`server.defaults.profile`](#serverdefaultsprofile) sets it. |
| Rule profile | The first matching entry in [`rules`](#rules) sets it. |
| Model profile | [`models.*.profile`](#modelsprofile) sets it, or `@name` on the request. |
| Per-model profile change | [`models.*.profiles`](#modelsprofiles) sets it. |
| Model overrides | [`models.*.overrides`](#modelsoverrides) sets them. |
| Request fields | The request body sets them. |

A request's `@name` replaces the model's own profile rather than
adding to it, and an unknown name gets a 400. `gmlx profiles <id>`
prints the sampling values a model resolves to under each of its
profiles, with every layer applied.

An unknown profile name in a model entry, a rule or `extends` fails the
load, and so does a cycle of `extends`.

### `profiles.*.extends`

The profile starts from this profile or intent, and each key that the
profile sets replaces the inherited value. With
an intent, each model starts from the intent as its own family defines
it. The default is none.

### `profiles.*.sampling`

These are default values for the request fields listed under
[Sampling](#sampling). A field that the request sends wins. The default is
none.

### `profiles.*.load`

These keys change how the model is built, as listed under
[Model loading](#model-loading). The default is none.

### `profiles.*.cache`

These prompt cache settings, with the keys listed under
[Prompt cache](#prompt-cache), change [`server.cache`](#servercache) for
the models that use this profile. The default is none.

### `profiles.*.system`

This system prompt applies only when the request has no system message.
The default is none.

### `profiles.*.chat_template`

This chat template replaces the one in the GGUF. It is inline Jinja or a
path to a `.jinja` or `.txt` file. The template applies at load, so two
ids with different templates are two loaded copies, and a multimodal
model keeps the template that it loads with its `mmproj`. The default is the GGUF's
template.

### `profiles.*.chat_template_kwargs`

The chat template receives these variables on each request. For example,
`preserve_thinking` on the Qwen3.6 and Gemma 4 templates keeps earlier
`<think>` blocks in the prompt, so that an agent sees its earlier reasoning.
Keys that the request sends win.

A key that names a parameter of the template call, such as `chat_template`
or `tokenize`, is dropped with a warning when the config loads. A request
that sends one gets a 400. Set a template of your own with `chat_template`.
The default is none.

### `profiles.*.thinking`

This key turns reasoning `on`, `off` or `adaptive` on any model. gmlx maps
the value to the variable that the model's template reads, so one profile
works across families.

| Family | Template variable | Values |
|--------|-------------------|--------|
| Qwen3.x, GLM | `enable_thinking` | It takes `true` or `false`. |
| MiniMax-M3 | `thinking_mode` | It has three states, so `adaptive` is accepted. |
| Kimi K2.x | `thinking` | It takes `true` or `false`. |
| Hy3 | `reasoning_effort` | Its levels include `no_think`. |
| gpt-oss | `reasoning_effort` | It takes `low`, `medium` or `high`, and reasoning cannot be turned off. |

A request's `enable_thinking` wins, then its `thinking` or
`reasoning_effort` field, then the profile, then the template's default.
An entry in the request's `chat_template_kwargs` wins over this key, and
this key wins over the same entry in the profile's own
`chat_template_kwargs`.

The z.ai form, `{"type": "enabled"}` or `{"type": "disabled"}`, is also
accepted in a request's `thinking`, and its optional
`clear_thinking: false` keeps earlier reasoning in the prompt. `gmlx run`
and `gmlx chat` take `--thinking`. The default is the template's own.

### `profiles.*.reasoning_effort`

This key sets a reasoning level that the model's template accepts, such
as `low`, `medium` or `high`. gmlx maps it to the template's variable in
the same way as `thinking`. `gmlx run` and `gmlx chat` take
`--reasoning-effort`. The default is the template's own.


## Rules

Rules give a profile to every model whose id matches a pattern, so a group
of models can share settings without repeating them in each entry. The
first rule that matches wins. A rule ranks below a model's own `profile`
and above [`server.defaults.profile`](#serverdefaultsprofile).

```yaml
rules:
  - {match: "*coder*", profile: review}
  - {match: "qwen3.8-*", profile: agent}
```

### `rules[].match`

The rule applies to the model ids that match this pattern. The pattern
uses shell wildcards such as `*`, and it is not a regular expression.
This key is required.

### `rules[].profile`

The models that match get this profile. This key is required.


## Sampling

A `sampling` block sets defaults for the request fields that control
generation, such as temperature and length. It goes in a profile or in a
model's `overrides`. Each key is also a request field of the same name,
and a field in the request wins. A default taken from the model's
[family defaults](family-defaults.md) is the value that `gmlx profiles`
prints for that model.

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

This key sets how random the choice of each token is, and `0` always
takes the most likely token. The key also accepts the name `temp`. The
default comes from the family defaults.

### `sampling.top_p`

The model samples only from the most likely tokens whose probabilities
add up to this value, and `0` turns off the filter. When `top_p` is the
only filter, it considers at most the 1024 most likely tokens, so a very
flat distribution loses its tail past that rank. The default comes from
the family defaults.

### `sampling.top_k`

The model samples only from this many of the most likely tokens, and `0`
turns off the filter. The default comes from the family defaults.

### `sampling.min_p`

The model drops each token whose probability is below this share of the
most likely token's probability, and `0` turns off the filter. The
default comes from the family defaults.

### `sampling.max_tokens`

Generation stops after this many tokens. The default is no limit, so
generation runs until the model stops or the context fills.
[`gmlx serve --max-tokens`](cli.md#gmlx-serve) sets a cap for server
requests that set none of their own.

### `sampling.seed`

This seed makes the request's sampling repeatable, without changing
the other requests in its batch. Two runs give the same tokens only when
their batches and the speculation setting also match, because a
different batch shape changes the logits slightly. The default is none.

### `sampling.stop`

Generation ends when it produces this string, or any string in this list.
A stop string can end generation in the middle of a token, and the
response then has `finish_reason: "stop"`. The key applies to chat
completions and `/v1/completions`, and the Anthropic endpoint uses its
own `stop_sequences`.
The default is none.

### `sampling.repetition_penalty`

This penalty lowers the chance of the tokens that appeared in the last
`repetition_context_size` tokens, and `1` has no effect. The default is
off.

### `sampling.repetition_context_size`

`repetition_penalty` looks at this many recent tokens. The default is
`20`.

### `sampling.presence_penalty`

This penalty lowers the chance of every token that was already generated,
by the same amount. The default is off.

### `sampling.frequency_penalty`

This penalty lowers the chance of each generated token in proportion to
how often it was generated. The default is off.

### `sampling.xtc_probability`

XTC sampling applies to a token with this probability. XTC removes the
most likely candidates to vary the text. It never removes newline or
end-of-sequence tokens, and it is not available on speculative models.
The default is off.

### `sampling.xtc_threshold`

XTC removes a candidate whose probability is above this value. The
default is `0.0`.

### `sampling.enable_thinking`

This key sets whether the chat template opens a thinking block. The
[`thinking`](#profilesthinking) key of a profile sets it on any family.
The default is the template's own.

### `sampling.thinking_budget`

gmlx closes the thinking block after this many reasoning tokens. The
count starts when a thinking block opens, whether the template or the
model opens it. On a model with a native MTP head, the block closes at the
end of a draft round, so the budget can be exceeded by one round. Such a
model drops the budget of a request that runs in a batch or resumes after
preemption. A model with a separate drafter refuses the key. `gmlx run`
and `gmlx chat` honor it too. The default is no limit.

### `sampling.thinking_start_token`

This text opens the model's reasoning. Family defaults set it for models
that do not use `<think>`. The default is `<think>`.

### `sampling.thinking_end_token`

This text closes the model's reasoning. The default is `</think>`.


## Model loading

A `load` block changes how a model is built, for example to quantize its
KV cache so that a long context uses less memory. It goes in a profile or
in a model's `overrides`. Two ids that differ in a load key are two
loaded copies of the model. Each key also has a
[`gmlx serve` flag](cli.md#gmlx-serve) for a server started with one GGUF,
and an [environment variable](env-vars.md#load-and-cache-keys) that sets
it for every model in the process.

```yaml
profiles:
  long-context:
    load: {kv_bits: 6, kv_quant_scheme: kvarn}
```

The keys `prefill_step_size` and `dtype` are not load keys. They apply to
the whole server, under [Scheduling](#scheduling).

### `load.kv_bits`

The KV cache quantizes to this many bits, which lets a long context use
less memory. `uniform` accepts 2, 3, 4, 6 and 8, and `kvarn` accepts 2,
3, 4, 5, 6 and 8. The server decides for each layer whether its cache
quantizes, as
[Which layers quantize](kv-quantization.md#which-layers-quantize)
describes. The default is no quantization, and with `kvarn` it is `6`.

### `load.kv_quant_scheme`

This key selects how the KV cache quantizes. `uniform` is affine
quantization. `kvarn` normalizes the variance first and keeps the newest
tokens in fp16, and a model that `kvarn` declines runs fp16 instead of
affine. [KV cache quantization](kv-quantization.md) lists which
architectures convert and how each scheme behaves with speculative
decoding. The default is `uniform`.

### `load.kv_group_size`

`uniform` quantization uses groups of this size. The default is `64`.

### `load.kv_tail_tokens`

Under `kvarn`, this many of the newest tokens stay fp16. The value must be
a multiple of 128. The default is `1024`.

### `load.quantized_kv_start`

The cache stays fp16 until it holds this many tokens, and then all of it
is quantized. Batched requests and `kvarn` quantize from the first token.
The default is `0`.

### `load.max_kv_size`

A request may use at most this many tokens of context on this model. On
the server this is a budget only. `gmlx run` and `gmlx chat` also make
the cache a rotating window of this size. The default is no cap.


## Prompt cache

The prompt cache keeps the computed start of recent prompts, so a request
that begins the same way skips that part of prefill. This matters most
for agents and long chats, which resend the same history on every turn.
[`server.cache`](#servercache) sets the cache for every model, and a
`cache` block in a profile or a model's `overrides` changes it. What the
cache restores for each architecture is in
[Prompt cache](prompt-cache.md).

```yaml
server:
  cache:
    enabled: true
    disk: {path: ~/.cache/gmlx/apc, max_gb: 100}
```

### `cache.enabled`

With `true`, the prompt cache is on. `gmlx init` writes `true`. The
default is `false`.

### `cache.block_size`

Each cache block holds this many tokens. The default, chosen for each
model, is the smallest of 16, 32, 64, 128 and 256 that lets the pool fill
its memory share without passing the Metal limit on the number of
buffers.

### `cache.num_blocks`

The pool that all cached prompts share holds this many blocks. Blocks
cost memory only once they hold a prompt, and a full pool drops the oldest
prompts. By default the pool starts at 2048 blocks and grows after the
model loads until it could fill half of the memory left over, within the
Metal limit on the number of buffers.

### `cache.exact_entries`

Hybrid and recurrent models, which cannot cache in blocks, keep this many
whole-prompt copies of the cache. Each copy is a full cache, so a higher
value uses more memory, and a new prompt past the limit evicts an older
copy. The default is `4`.

### `cache.hash`

This key selects how blocks are identified. `fast` uses Python's hash,
which is stable only within one process. `sha256` is stable across
processes and costs more per token. The default is `fast`.

### `cache.disk`

This key adds a second cache tier on the SSD, so that a prompt survives
the model unloading or the server restarting. `true` puts the tier at
the default [`path`](#cachediskpath), and a mapping sets the
`cache.disk.*` keys. The default is `false`.

### `cache.disk.path`

The SSD tier is stored in this folder, and setting the key turns the tier
on. The default is `$XDG_CACHE_HOME/gmlx/apc`, or `~/.cache/gmlx/apc`
when `XDG_CACHE_HOME` is not set.

### `cache.disk.max_gb`

Each namespace may use at most this much space, in GB. With the default
namespace each model has its own, so the total can reach this value times
the number of models. The default is no limit.

### `cache.disk.namespace`

A model reads and writes this partition on disk. The default is the model
path, which gives each model file its own partition.

### `cache.disk.workers`

This many threads write entries to disk. The default is `1`.

### `cache.disk.read_mode`

This key selects how entries are read back. `direct` reads the needed
bytes with plain file reads, and `mmap` maps the files. The default is
`direct`.


## Model discovery

The `discover` block lists folders that the server scans at each start
and reload. Every GGUF that it finds without an entry under `models`
becomes a model, so new files appear after a restart or a reload.
`gmlx run`, `gmlx list` and `gmlx rm` scan the same folders. The scan
reads only the GGUF headers. To write entries once instead, run `gmlx init` or
`gmlx sync-models`.

```yaml
discover:
  - dir: ~/models
    recursive: true
```

Each model is named after its file. The scan removes the shard
suffix, markers such as `mmproj`, `assistant`, `draft` and `mtp`, and
imatrix tags, and adds the quantization in short form, such as `-q4`.
When two files would get the same name, both get the full quantization,
such as `-q4-k-m` and `-q4-k-s`, and a remaining clash gets a number. The
server prints the names in a `[server] discover:` line.

### `discover[].dir`

The scan reads this folder. `null` scans every folder in
`server.model_dirs`. The default is `null`.

### `discover[].recursive`

With `true`, the scan also reads subfolders. The default is `false`.

### `discover[].pair_mmproj`

With `true`, each `mmproj*.gguf` pairs with the model in the same folder
that it matches. The default is `true`.

### `discover[].speculative`

This key turns on speculative decoding for the models found. `auto` and
`true` turn it on for models with an MTP head, and `false` never does. A
drafter GGUF in the same folder becomes a model's `draft_gguf` when its
architecture can draft for the model, their hidden sizes agree, and their
file names match. When only one model in the folder qualifies, it gets
the drafter whatever its file name. A drafter whose header names its base
model pairs with that model in any scanned folder, and streamed models
get no drafter. The default is `auto`.


## Server

The `server` block sets up the server process. It says where the server
listens, who may call it, where models are found, how much memory they
may use, how requests share the GPU, and which extra services run beside
the models. A file with no `server` block runs a server on
`127.0.0.1:8080` with the prompt cache off.

```yaml
server:
  host: 127.0.0.1
  port: 8080
  model_dirs: [~/models]
  cache: {enabled: true}
  defaults:
    ttl_s: 900
    model: qwen3.8-27b-ud-q6
```

### Address and authentication

#### `server.host`

The server listens on this address. An address other than loopback needs
[`api_key`](#serverapi_key) or [`no_auth`](#serverno_auth). The default is
`127.0.0.1`.

#### `server.port`

The server listens on this port. The default is `8080`.

#### `server.api_key`

Every request must present this key, as `Authorization: Bearer <key>` or
`x-api-key: <key>`. Only `/health` and CORS preflight requests pass
without it. The server reads its key only from this file, never from a
flag or the environment, so the key stays out of process listings and
shell history. The client commands `chat`, `talk`, `ps`, `systemone`,
`launch` and `launch menubar` take `--api-key` to present it. The default
is no key.

A loopback server refuses a request whose `Host` header is not a loopback
name, which blocks DNS rebinding. Every server refuses, with status 403, a
request from a page whose origin [`cors_origins`](#servercors_origins) does
not allow. Its CORS answers name the page's own origin, never `*`, and
allow no credentials.

#### `server.no_auth`

With `true`, the server accepts an address other than loopback without a
key. Use it when a proxy in front of the server handles authentication.
The default is `false`.

#### `server.media_urls`

With `true`, a request may name an image, audio or video by an `http(s)://`
URL, and the server fetches it from the Mac. The server refuses a host with
any address that is not public, such as one on the Mac or the local
network. It checks each redirect the same way and uses no proxy.

A fetch stops when it takes more than 60 seconds or passes the size limit
of [Media in requests](api.md#media-in-requests). The server reads this key
when it starts, so run `gmlx restart` after you change it.

Any client that reaches the server's port can then make the Mac send
requests to any public host, and the URL can carry data out. A client in a
launch container uses a [session socket](glossary.md#session-socket)
instead, and the socket refuses a URL in the media parts of a request, as
[Container security](container-security.md#what-the-client-reaches-on-the-server)
describes.

A server that listens beyond loopback with no key is the exception, because
the container also reaches its port. Set [`server.api_key`](#serverapi_key)
on such a server. This key does not change which files a request can name.
The default is `false`, which takes no URLs.

#### `server.cors_origins`

Pages from these origins may call the server. Pages on a loopback address,
such as `http://localhost:3000` or `http://127.0.0.1:5173`, may always call
it. So may desktop apps built on Electron, Tauri or VS Code webviews, which
send an origin with the scheme `app`, `file`, `tauri`, `vscode-file` or
`vscode-webview`, since a web page cannot send one.

An entry is a scheme, a host and an optional port, such as
`https://chat.example.com` or `http://192.168.1.20:3000`, with nothing
after them. For an app with another scheme, the entry is the scheme and a
name, such as `capacitor://localhost`.

A browser extension's entry is the origin that the refusal in the server
log names, such as `chrome-extension://<id>` or `moz-extension://<uuid>`.
[Troubleshooting](troubleshooting.md#a-web-page-or-browser-extension-gets-403-or-a-cors-error)
says where each browser shows the ID. A wildcard entry lets every extension
of one browser call the server:

- `chrome-extension://*` for Chrome, Edge and other Chromium browsers.
- `moz-extension://*` for Firefox.
- `safari-web-extension://*` for Safari, which changes an extension's ID
  each time it starts, so a Safari extension needs this entry.

A browser sends the page's origin with each request, and the server answers
any other origin with status 403, even when the request carries a valid
[`api_key`](#serverapi_key). The page cannot read that answer, so its
browser console shows only a CORS error. Programs that send no origin, such
as curl and most API clients, are not affected.

The server log shows a line for each refused origin, at most once a minute.
The line names the entry to add, or says why an origin such as `null`
cannot be listed. Look there first when a page or extension fails.

Listing an origin gives every page on it whatever the server offers,
including the MCP tools of [served assistants](#served-assistants), which
run on the Mac. A wildcard entry gives the same to every extension you
install in that browser, now or later. When the browser keeps an
extension's ID, list that extension's own origin instead.

Neither `*` nor `null` names one origin, so the server refuses both entries.
Apart from the three wildcards above, an entry with a `*` is refused too,
and the error says what to write instead.

The server reads the list when it starts, and logs each listed origin that
is not a loopback one. A reload does not apply a change to the list, so run
`gmlx restart` after you edit it. The default is no origins.

### Model folders

#### `server.model_dirs`

The server searches these folders, in order, for a relative `path`,
`mmproj`, `draft_gguf` or `adapter`. A [`discover`](#model-discovery)
entry with no `dir` scans them. Entries expand `~` and `$VAR`. A path
found in none of the folders fails the load, and the error lists the
folders searched. `gmlx pull` downloads into the first one. The default is
no folders.

#### `server.hf_cache`

With `true`, the Hugging Face libraries run offline, and the server never
downloads. An `hf:` model [`path`](#modelspath) works whatever this key's
value is. `gmlx init --from-hf-cache` and
`gmlx sync-models --from-hf-cache` write such entries and set this key,
and `sync-models` then also scans the cache. The
[HTTP API](api.md#hugging-face-policy) describes how the server treats a
request that names a Hugging Face model. The default is `false`.

### Services

Each service adds an endpoint.
[Speech, embeddings and rerank](services.md) lists the models that each
one accepts and tells what happens when a model file is missing.

#### `server.stt`

This key names the speech-to-text model, as an alias such as
`whisper-turbo`, a Hugging Face repo id or a folder. `true` selects the
default model. The default is none.

#### `server.tts`

This key names the text-to-speech model, as an alias such as `kokoro`, a
Hugging Face repo id or a folder. `true` selects the default model. The
default is none.

#### `server.embeddings`

This key names the embeddings model, as a GGUF path, an `hf:` reference,
an alias, or the repo id or local folder of an mlx-embeddings model. `true` selects the default model. The default is none.

#### `server.rerank`

This key names the reranker, as a Qwen3-Reranker GGUF path, an `hf:`
reference or an alias. `true` selects the default model. The default is
none.

### Memory and residency

The server keeps several models in memory at once, up to the budget in
`server.budget_gb`. A model takes about the size of its GGUF file in
memory, because the weights map from the file without a copy. How a
streamed model counts against the budget is in
[Models larger than memory](streaming.md#residency-of-a-streamed-model).

| State | Set by | When it unloads |
|-------|--------|-----------------|
| Pinned | `pin: true`, `--pin` | It unloads only on `POST /unload`. |
| Kept | `POST /v1/keep`, `gmlx launch --model`, a talk session | It unloads when the budget is full and it is the least recently used. |
| Idle | Any request | It unloads after `ttl_s` seconds without a request, or when the budget needs the room. |
| Preloaded | `server.defaults.preload` | It unloads like an idle model. |

A model is never unloaded during a generation. Keeping is what
`gmlx launch --model` asks for, so a coding session's model survives
the pauses between turns without holding budget permanently. A request
to `POST /v1/keep` with `{"keep": false}` releases a model, and
`POST /unload` unloads it at once.

Two ids that point to the same GGUF share one loaded copy, unless a
setting that changes how the model is loaded differs between them. Those
settings are the `load` and `cache` keys, `mmproj`, `draft_gguf`,
`speculative`, `speculative_width_cap`, `chat_template`, `stream` and the
streaming keys. Ids that differ only in `adapter` share one copy, as
[LoRA adapters](lora.md#serving-one-base-with-many-adapters) describes.
Sampling, `system` and `ttl_s` never cause a second copy.

#### `server.budget_gb`

Loaded models may use this much memory together, in GB. The default is
0.8 times the GPU working set that macOS recommends.

#### `server.max_models`

The server loads at most this many models at once, and checks this limit
after the budget. The default is no limit.

#### `server.cache_limit_gb`

This key limits the MLX buffer cache, in GiB. MLX keeps freed GPU memory in
this cache for reuse instead of returning it to macOS, and at deep context
it can grow to tens of GB. `0` turns the buffer cache off,
and a negative value removes the limit.
[The MLX buffer cache](memory.md#the-mlx-buffer-cache) explains when to
change it. By default the server sets a limit from 4 to 12 GiB at start.
It is a quarter of the memory that the weights leave free when they take
more than 60 percent of the GPU working set, and 5 percent of the working
set otherwise.

#### `server.defaults.model`

A request that names no model goes to this model. It is also the first
model that the server loads when no model is pinned. The default is the
only model when the file has one, and none otherwise.

#### `server.defaults.ttl_s`

A model that is not pinned unloads after this many seconds without a
request. `null` or `0` turns off the timeout. The default is `900`.

#### `server.defaults.preload`

The server loads these models at start, one at a time, after the pinned
models, or after the default model when none is pinned. `all` loads every
model. The default is none.

#### `server.defaults.profile`

Every model gets this profile, which ranks below rules and the model's own
profile. The default is none.

### Scheduling

These keys apply to every model. Each one except
`server.token_queue_timeout_s` also has a
[`gmlx serve` flag](cli.md#gmlx-serve).

#### `server.prefill_step_size`

Prefill processes the prompt in chunks of this many tokens. A lower value
lowers a long prompt's memory peak and slows prefill. The default is
`2048`.

#### `server.dtype`

The activations, the unquantized weights and the KV cache use this type.
The value is `auto`, `bfloat16` or `float16`. `auto` picks `float16` on
M1 and M2, which lack bfloat16 arithmetic, and `bfloat16` on later chips.
The weights on disk never change. The default is `auto`.

#### `server.decode_prefill_ratio`

This key sets how a new request's prefill shares the GPU with
requests that are generating. With `auto`, prefill slows only when a
generating request would drop below half its speed. A number such as
`1.0` makes each prefill chunk wait until the generating requests have
had that multiple of the chunk's GPU time. With `0`, one prefill chunk
runs per generation step. [Concurrent requests](concurrency.md)
describes the effect. The default is `auto`.

#### `server.prefill_tick_ms`

One prefill chunk may take this many milliseconds while other requests
generate. A longer chunk is halved until it fits, and `0` turns off the
halving. The default is `500`.

#### `server.token_queue_timeout_s`

A request fails when its next token takes longer than this many seconds,
and `0` waits forever. The server cancels a request that times out and
records it as `last_error` in `/v1/metrics`, and a streaming client
receives a final error event. The usual cause is a long prompt on a model
larger than memory. On `/v1/systemone` the limit applies to a whole
decision. The default is `1800`.

### Features

#### `server.family_defaults`

With `true`, each request starts from its model's
[family defaults](family-defaults.md), and the built-in intents are
available. `false` turns off both. The default is `true`.

#### `server.stochastic_mtp`

With `true`, the server accepts speculative tokens by rejection sampling,
which accepts more of them. The sampling distribution stays exact, but the
output is no longer token-identical to plain decoding. Greedy requests do
not change, and a reload does not apply a new value of this key.
[Stochastic acceptance](speculative-decoding.md#stochastic-acceptance)
describes it. The default is `false`.

#### `server.gpu_keepwarm`

This key controls GPU keep-warm, which keeps the GPU clock up between
tokens for a streamed model with a decode feeder, as
[Models larger than memory](streaming.md#the-lossless-settings) describes.
No other model is affected. `false` turns it off and `true` turns it on,
overriding `GMLX_GPU_KEEPWARM`. The default is none, which leaves
keep-warm on.

#### `server.menubar`

With `true`, a server started in the background can open the macOS
[menu bar app](menubar.md). The default is `true`.

#### `server.cache`

These are the prompt cache settings for every model, with the keys listed
under [Prompt cache](#prompt-cache). A profile or a model's `overrides`
changes them. The default is the cache turned off.

### Structured decisions

These keys set up `POST /v1/systemone`, which
[Structured decisions](decisions.md) describes. `canvas`, `constrained`
and the three think keys apply to DiffusionGemma, and the
[letter readout](glossary.md#letter-readout) on other models ignores them.
An unknown key, or a `model` that is not a configured id or alias, fails
the load. A reload applies new values to the next request.

#### `server.systemone.model`

A request whose `model` is absent or names nothing configured uses this
model id or alias, optionally with `@profile`. The default is none.

#### `server.systemone.canvas`

One [structured read](glossary.md#structured-read) fills a
[canvas](glossary.md#canvas) of at most this many tokens. The value must be a
positive multiple of 16, and the model's own canvas length caps it. The
default is `64`.

#### `server.systemone.constrained`

With `true`, a read computes probabilities only over its answer tokens.
With `false`, the full vocabulary gives the same one-step answer
probabilities, but different entropies and multi-step reads. The default
is `true`.

#### `server.systemone.max_questions`

One request may ask at most this many questions, and a request with more
gets a 422. The default is `64`.

#### `server.systemone.max_samples`

A request may ask for at most this many `samples` and `auto_max`, and a
larger value is lowered to this one. On the letter readout, the limit
applies to the option orders that `samples` sets. The default is `32`.

#### `server.systemone.think`

A request that does not set `think` gets this thought budget, in tokens.
The value is from 0 to 4096, or `"auto"` to think only when an answer is
uncertain. The default is `0`.

#### `server.systemone.think_threshold`

When `think` is `"auto"`, the model thinks when the probability of an
answer's chosen label is below this value. That probability is not the
answer's `confidence`. A request can set its own. The default is `0.8`.

#### `server.systemone.think_budget`

When `think` is `"auto"`, a thought gets this budget in tokens. A
request can set its own. The default is `64`.

### Served assistants

The server can itself run the [assistant's](#assistant) tool loop and
offer each assistant as a model. A client names an assistant as its
model and gets tools without a loop of its own, as
[Assistant](assistant.md#served-assistants) describes.

```yaml
server:
  assistants:
    helper:
      model: qwen3.8-27b-ud-q6
      mcp: []
```

#### `server.assistants.*.model`

This configured model answers for the assistant. This key is required.

#### `server.assistants.*.memory`

With `true`, the assistant has long-term memory, in one store that all
its clients share. The store is `assistant-<id>.db` beside the default
memory file, and it ignores `assistant.memory.enabled` and
`assistant.memory.path`. A turn from a
[container mode](launch-container.md) client never reads or adds to the
memory. The default is `false`.

#### `server.assistants.*.mcp`

These are the assistant's tool servers, with the same keys as
[`assistant.mcp`](#assistantmcp). `null` uses `assistant.mcp`, and `[]`
gives the assistant no tools. With
[`server.assistant_allow_remote`](#serverassistant_allow_remote) set and a
non-empty `assistant.mcp`, every assistant must set this key. The default
is `null`.

#### `server.assistant_allow_remote`

With `true`, served assistants may run on a server with an address other
than loopback. Their tools run on the server host, so the server otherwise
refuses to start with them. The default is `false`.


## Voice

The `talk` block sets up the voice client [`gmlx talk`](talk.md) and the
menu bar app's voice sessions. It chooses which model answers, the
voice, how listening starts and when an utterance ends. Most keys also
have a [`gmlx talk` flag](cli.md#gmlx-talk), which wins over the file.

```yaml
talk:
  model: qwen3.8-27b-ud-q6@instruct
  voice: af_heart
  mode: vad
  vad: {silence_ms: 450}
```

### `talk.model`

The voice client talks to this model, given as an id or alias and
optionally `@profile`. The default is the server's default model.

### `talk.voice`

Replies use this voice, which is a Kokoro preset or a Qwen3-TTS speaker
name. The default is the server's default voice.

### `talk.speed`

Replies are spoken at this multiple of normal speed, from 0.25 to 4. The
default is `1.0`.

### `talk.system`

This system prompt sets the spoken persona. `null` or `""` sends no
system prompt at all, which is not the same as leaving the key out. The
default is a prompt that asks for speakable text without markdown.

### `talk.language`

Speech recognition gets this language hint, such as `en`. The default is
automatic detection.

### `talk.max_tokens`

A spoken reply has at most this many tokens. The default is no limit.

### `talk.mode`

This key sets how listening starts. `wake` waits for the wake phrase,
`vad` starts on any speech, and `ptt` uses Space as push-to-talk. `text`
takes typed prompts and still speaks the replies. The default is `wake`.

### `talk.wake_word`

Listening starts on this wake phrase, which can be any text. The default
is `hey assistant`.

### `talk.wake_threshold`

Detection must reach this confidence, from 0 to 1, before the wake phrase
counts. A higher value gives fewer false wakes. The default is `0.3`.

### `talk.push_to_talk_modifier`

The menu bar hotkey is Space, pressed while holding this modifier. The
value is `globe`, `right-command`, `right-option` or `control`. There is
no flag for this key. The default is `globe`.

### `talk.input_device`

This key selects the microphone, by part of a device name or by an index.
`gmlx talk` lists devices with `/devices`. The default is the system
input.

### `talk.output_device`

This key selects the speaker, by part of a device name or by an index. The
default is the system output.

### `talk.chime`

With `true`, a sound plays on wake and at the end of a turn. The default
is `true`.

### `talk.brain`

This key selects what answers. `chat` is the plain model, and `assistant`
adds the [assistant](#assistant) block's tools and memory. The
default is `chat`.

### `talk.vad.threshold`

Audio counts as speech when its speech probability, from 0 to 1, is above
this value. The default is `0.6`.

### `talk.vad.silence_ms`

A pause of this many milliseconds ends an utterance. A lower value
gives an answer sooner and cuts off more sentences. The default is `550`.

### `talk.vad.min_speech_ms`

An utterance shorter than this many milliseconds is dropped as noise. The
default is `300`.

### `talk.vad.pre_roll_ms`

The client keeps this many milliseconds of audio from before speech
starts. There is no flag for this key. The default is `400`.


## Assistant

The `assistant` block gives the built-in assistant its tools and its
memory. `gmlx chat --assistant`, `gmlx talk` with `brain: assistant`, and
[served assistants](#served-assistants) use it. It does not affect the
coding agents that `gmlx launch` connects. How the tool loop and memory
work is in [Assistant](assistant.md).

```yaml
assistant:
  max_tool_rounds: 8
  mcp:
    - name: files
      command: [npx, -y, "@modelcontextprotocol/server-filesystem", "/Users/me/notes"]
    - name: search
      url: http://127.0.0.1:8931/mcp
  memory:
    top_k: 6
```

### `assistant.max_tool_rounds`

The model may make at most this many rounds of tool calls in one turn,
and then it must answer. The value is at least 1. The default is `8`.

### `assistant.tool_timeout_s`

One tool call may take at most this many seconds. The value is at least
1. The default is `60`.

### `assistant.mcp`

These [MCP](glossary.md#mcp) servers provide the tools, as a list. Each
entry has a `name` and exactly one of `command` and `url`. The default is
none.

### `assistant.mcp[].name`

This name identifies the server, and it must be unique in the list. When
a tool has the same name as a tool from a server earlier in the list, the
later tool gets its server's name and an underscore as a prefix. This key
is required.

### `assistant.mcp[].command`

This command starts a server over stdio. It is a list of arguments, or a
string that is split like a shell command line. The server's log goes to
`~/.cache/gmlx/mcp-<name>.log`. The default is none.

### `assistant.mcp[].url`

The assistant reaches a server over streamable HTTP at this address. The
default is none.

### `assistant.mcp[].env`

A stdio server gets these environment variables. From your environment,
the server gets only `HOME`, `PATH`, `SHELL`, `TERM`, `USER` and
`LOGNAME`, so a token that it needs must be set here. The default is
none.

### `assistant.memory.enabled`

With `true`, `gmlx chat --assistant` and `gmlx talk` remember facts
across conversations. Memory needs
[`server.embeddings`](#serverembeddings), as
[Memory](assistant.md#memory) describes. The default is `true`.

### `assistant.memory.path`

The memory is stored in this database file. The default is
`$XDG_DATA_HOME/gmlx/assistant-memory.db`, which is
`~/.local/share/gmlx/assistant-memory.db` when `XDG_DATA_HOME` is not
set.

### `assistant.memory.top_k`

This many remembered facts are added to each turn. The value is at least
1. The default is `4`.

### `assistant.memory.extract`

With `true`, the model extracts short facts from each exchange, and the
assistant stores the facts. With `false`, the assistant stores the exchanges as
they are. The default is `true`.

### `assistant.memory.ttl_days`

At start, the assistant forgets facts older than this many days. The
default is to keep facts forever.

### `assistant.memory.max_items`

The store holds at most this many facts, and [Memory](assistant.md#memory)
describes which facts go first. The value is at least 1. The default is `20000`.


## Launch

The `launch` block sets how [`gmlx launch`](launch.md) runs clients in
[container mode](launch-container.md). Launch reads it from the first of
`~/.config/gmlx/gmlx.yaml` and `~/.gmlx.yaml` that exists, the same file the
server reads, as [Where gmlx looks](#where-gmlx-looks) describes.

A `launch` block in a file that `--config` names is never read by launch,
and the server takes no setting from the block.

A malformed block stops a container launch with a message that names the
file, and `--no-container` then runs the client on the Mac. The server
ignores a malformed `launch` block with one warning and loads the rest of
the file.

A launch with no container flag stops too, unless the malformed block
clearly leaves that client off. That means every `enabled` on the way to
the client is absent or `false`, and no key on the way is unknown. Such a
launch runs the client on the Mac with one notice. Fix the block, or pass
`--no-container`.

A file that does not parse, or a misspelled top-level key that holds a
`container` block, such as `lauch:`, stops every launch without
`--no-container`, since launch cannot tell whether container mode is on.
The server refuses the whole file for an unknown top-level key, so correct
the spelling before the server starts.

The keys under `launch.container` apply to every client. Each of them also
goes under `launch.container.clients.<client>` for one client, where the
client's value wins over the global one and the lists of the two levels add
up, with volume names that depend on the level. Six more keys exist only
for one client, and they follow the shared keys.

This block turns on container mode with more memory, gives Claude Code a
volume and a seed, and runs Open WebUI from its official image with the
served assistant `home`:

```yaml
# doctest: build
launch:
  container:
    enabled: true
    memory: 6G
    clients:
      claude-code:
        volumes: [claude-pg:/var/lib/postgresql:8G]
        seed: [~/.claude/CLAUDE.md]
      open-webui:
        image: ghcr.io/open-webui/open-webui:main
        command: image
        assistants: [home]
```

### `launch.container.enabled`

With `true`, `gmlx launch` runs the client in a container. `--container`
and `--no-container` override it for one launch. The default is `false`.

### `launch.container.mount_cwd`

With `true`, the current folder is shared read-write at the same path, and
the client starts there. `--mount-cwd` and `--no-mount-cwd` override it for
one launch. When neither level sets it, the folder is shared for every
client except `open-webui` and `elia`.

### `launch.container.mounts`

Each entry `PATH[:DST][:ro]` shares another folder at `DST`, or at the same
path without one, and `:ro` makes the share read-only. `--mount` adds
entries for one launch. The default is no extra folders.

### `launch.container.volumes`

Each entry `NAME:/path[:SIZE]` mounts the named volume at the path, and a
missing volume is created with `SIZE`, such as `8G`. The name starts with a
letter or digit and holds only letters, digits, `_`, `.` and `-`. One name
cannot appear at two paths or with two sizes. The default size is `32G`.

An entry under `launch.container.clients.<client>` gets a volume for each
project, named `NAME-` and 8 hex digits of the project, and the `default`
project uses the name as written. An entry at the global level keeps its
name in every session, so every client and project shares its data. A name
has at most 200 characters.
[Volumes](launch-container.md#volumes) describes how they behave.

### `launch.container.forward`

Each port in this list, from 1 to 65535, reaches the same port on the
Mac's `127.0.0.1` from the container's own `127.0.0.1`. The default is no
ports.

### `launch.container.network`

With `default`, the container has internet access. With `none`, it reaches
only the gmlx server and the forwarded ports. `--network` overrides it for
one launch. The default is `default`.

### `launch.container.cpus`

The container gets this many CPUs. The default is `4`.

### `launch.container.memory`

The container gets this much memory, such as `4G` or `6144M`, and its
virtual machine takes 128 MB more. [Limits](container-security.md#limits)
describes how it counts against the model server. The default is `4G`.

### `launch.container.ssh_agent`

With `true`, the client can use the SSH agent that `SSH_AUTH_SOCK` names
when the session starts. The full path of an agent socket gives the client
that agent instead. For 1Password, that path is
`~/Library/Group Containers/2BUA8C4S2C.com.1password/t/agent.sock`.

The client can sign with every key loaded in the agent, so it can push to
any repository those keys reach, as
[Access you turn on](container-security.md#access-you-turn-on) describes.

Launch refuses a path that is not a socket you own. It also refuses a path
in a shared folder or a private home, and a path whose symbolic links lead
through one, also when `SSH_AUTH_SOCK` names it. A client could leave a
link to another agent there. Shared folders are the ones this session
shares and the ones an earlier session shared read-write. Keep the socket
and its links out of them.

Before the session starts, launch runs `ssh-add -l` and prints a line when
the agent holds no keys or does not answer. With `true` and no
`SSH_AUTH_SOCK`, it prints a line and the container gets no agent. A copy
that joins a running session, and `--shell` in one, use the agent of the
session. The default is `false`.

### `launch.container.env`

Each entry `NAME` passes that variable from your environment into the
container, and `NAME=VALUE` sets it. Values never appear on a command line.
An entry `NAME` for a variable that launch sets for the client, such as
`ANTHROPIC_MODEL`, keeps launch's value. `CLAUDE_CODE_MAX_CONTEXT_TOKENS`
follows the rule in [claude-code](launch.md#claude-code).

`HOME`, `TERM`, `COLORTERM`, `LANG`, `TZ`, `PATH` and `SSH_AUTH_SOCK` are
refused, because launch sets them itself or keeps the image's own. The
default is no variables.

### `launch.container.open_browser`

With `true`, launch opens a browser app such as Open WebUI in the Mac's
browser once the app answers. With `false`, it prints the app's address at
that point instead. Launch never opens a port that served the pages of
another project, as [Browser apps](launch-container.md#browser-apps)
describes. The default is `true`.

### `launch.container.clipboard`

With `images`, clients in the container can paste images from the Mac
clipboard, as [Clipboard images](launch-container.md#clipboard-images)
describes. A bare `off` also works. The default is `off`.

### `launch.container.clients`

This mapping holds the settings of single clients, keyed by `claude-code`,
`opencode`, `pi`, `omp`, `hermes`, `goose`, `aichat`, `elia`, `open-webui`
or `dsh`. Each takes the shared keys above and the keys below.

### `launch.container.clients.*.image`

The client runs this image, a local tag or a registry reference, instead
of the one gmlx builds. It cannot be combined with `build` or `packages`.
[A ready-made image](container-images.md#a-ready-made-image) describes it.
The default is the image gmlx builds.

### `launch.container.clients.*.build`

Launch builds the client's image from this Containerfile, or from a folder
that holds a `Containerfile` or `Dockerfile`. The path must be absolute or
start with `~`, and it cannot be combined with `image`.
[Your own Containerfile](container-images.md#your-own-containerfile)
describes it. The default is the image gmlx builds.

### `launch.container.clients.*.command`

A list of strings replaces the client's own command, and the word `image`
runs the image's own ENTRYPOINT and CMD, as
[The command that runs](launch-container.md#the-command-that-runs)
describes. The default is the client's own command.

### `launch.container.clients.*.packages`

These Debian packages are added to the image that gmlx builds for the
client, and the next launch builds the image again with them:

```yaml
# doctest: build
launch:
  container:
    clients:
      claude-code:
        packages: [make, python3, postgresql-client]
```

With `build`, they apply only when the Containerfile starts from the
client's own `:base`, and launch refuses them otherwise. The default is no
packages.

### `launch.container.clients.*.seed`

Each file or folder named here is copied into the private home at the
same path relative to your home folder:

```yaml
# doctest: build
launch:
  container:
    clients:
      claude-code:
        seed: [~/.claude/CLAUDE.md, ~/.claude/commands]
```

A seed is copied again when it changes on the Mac while the copy does not,
and `--reseed` copies it again in any case. Its real path must lie inside
your home folder and outside credential folders, as
[The private home](launch-container.md#the-private-home) describes with
the copy's limits. The default is no files.

### `launch.container.clients.*.assistants`

Each name in this list is a [served assistant](#served-assistants) that
the client can use, and the server answers the client as if the others did
not exist. Their tools run on the Mac, so list assistants only for a chat
client such as `open-webui`, as
[What the client reaches on the server](container-security.md#what-the-client-reaches-on-the-server)
explains.

An assistant's [memory](#serverassistantsmemory) is off for the client's
turns. The default is no assistants.

## Chat themes

The `theme` and `themes` keys set the colors of [`gmlx chat`](chat.md).

```yaml
theme: my-black
themes:
  my-black:
    extends: dark
    heading: {bold: true, rgb: "#88c0d0"}
```

### `theme`

Each chat starts with this theme, which is a built-in name or one from
`themes`. `--theme` and `/theme` change it. The default is `dark`.

### `themes`

These are themes of your own, keyed by name. A theme with the name of a
built-in theme replaces it. Each theme sets styles for kinds of text,
such as `heading` and `thinking`, and takes `extends` for the slots that
it leaves out. [Themes](chat.md#themes) lists the slots and style keys.
The default is none.


## Changing the file

gmlx checks the whole file when it reads it. An unknown key fails the load
with the name of the key, so a typo such as `pinned:` for `pin:` is caught
before the server starts. In `sampling`, `load` and `cache`, an unknown
key causes only a warning. The load does not check the keys inside
`chat_template_kwargs` or `themes`, and `gmlx chat` warns about a bad
theme when it starts.

A running server reads its file again on `POST /v1/reload` or on `SIGHUP`.
Models that are already loaded stay loaded when their load settings did
not change. A change to a load setting, such as `mmproj` or `speculative`,
applies the next time that model loads.

Several commands change the file for you, and they keep your comments and
formatting:

| Command | Change |
|---------|--------|
| `gmlx sync-models` | It adds new GGUFs from `model_dirs` and removes the entries whose file is gone. |
| `gmlx pull` | It adds the entry of each GGUF it downloads. |
| `gmlx rm` | It deletes a model's files and its entry. |

Each of them, and `gmlx init`, tells a running server to reload. Pass
`--no-reload` to `init`, `sync-models` or `rm` to prevent that.
`gmlx pull --no-register` leaves the config file and the running server
unchanged.

A server started without a config file, from a GGUF path or
`--models-dir`, has no file to read again. These commands do not signal
it. It ignores a `SIGHUP` with a log line and
answers `/v1/reload` with status 501.


## Flags and environment variables

Some settings can also be set by a `gmlx serve` flag or by an environment
variable. When a setting has more than one source, this order decides:

1. The `serve` flag wins.
2. The key in the file comes next.
3. The environment variable comes last.

Two variables are the exception. `GMLX_CACHE_LIMIT_GB` wins over
`server.cache_limit_gb`, and `GMLX_MTP_WIDTH_CAP` wins over each model's
`speculative_width_cap`, so that a benchmark can fix either without
changing the file. The flags are in the [CLI reference](cli.md#gmlx-serve)
and the variables in [Environment variables](env-vars.md).


## Complete example

This file combines the common blocks, and it loads without errors.

```yaml
# doctest: build
server:
  host: 127.0.0.1
  port: 8080
  api_key: null
  no_auth: false
  media_urls: false
  cors_origins: []
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
    helper: {model: qwen3.8-27b-ud-q6, memory: false, mcp: null}
  assistant_allow_remote: false
  defaults:
    profile: null
    ttl_s: 900
    model: qwen3.8-27b-ud-q6
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
  qwen3.8-27b-ud-q6:
    path: unsloth__Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
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
  coder: qwen3.8-27b-ud-q6@qwen-coder
assistant:
  max_tool_rounds: 8
  tool_timeout_s: 60
  mcp:
    - {name: clock, command: [uvx, mcp-server-time]}
  memory: {enabled: true}
discover:
  - {dir: null, recursive: true, pair_mmproj: true, speculative: auto}
```

The smallest useful file has one model with a path:

```yaml
# doctest: build
models:
  my-model:
    path: ./my-model-Q4_K_M.gguf
```
