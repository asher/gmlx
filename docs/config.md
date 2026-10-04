# Configuration

`gmlx.yaml` names the models gmlx can run, the settings each one runs with,
and the server's own settings. Every key has a default, so a file holds only
the keys you want to change. A model name from the file works in a request,
in `gmlx run` and in `gmlx chat`.

- [Create the file](#create-the-file)
- [Where gmlx looks](#where-gmlx-looks)
- [What the file contains](#what-the-file-contains)
- [Models](#models), [Aliases](#aliases), [Profiles](#profiles),
  [Rules](#rules)
- [Sampling](#sampling), [Model loading](#model-loading),
  [Prompt cache](#prompt-cache)
- [Model discovery](#model-discovery)
- [Server](#server)
- [Voice](#voice), [Assistant](#assistant), [Launch](#launch),
  [Chat themes](#chat-themes)
- [Changing the file](#changing-the-file)
- [Flags and environment variables](#flags-and-environment-variables)
- [Complete example](#complete-example)

## Create the file

`gmlx init` scans a folder of GGUF files and writes a config with one entry
for each model it finds:

```sh
gmlx init --models-dir ~/models
```

The file goes to `~/.config/gmlx/gmlx.yaml`. With no flags on a terminal,
`init` asks its questions as a wizard instead. The result holds the server
settings you chose and the models it found:

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

The comment above each model shows the sampling values it starts from,
taken from its [family defaults](family-defaults.md) and the GGUF. Start the
server with `gmlx serve`, which finds the file on its own.

## Where gmlx looks

A command that needs the file reads the first one it finds:

1. `~/.config/gmlx/gmlx.yaml`, where `gmlx init` writes.
2. `~/.gmlx.yaml`.

Pass `--config FILE` to read another file. gmlx never reads a `gmlx.yaml`
in the current folder, because a config can name commands that the server
runs.

gmlx reads only the first file it finds. If you already have a
`~/.gmlx.yaml`, a new `~/.config/gmlx/gmlx.yaml` hides it completely, so
add new blocks to the file you already have.

To see every setting a server would run with, including the defaults you
did not set, run `gmlx serve --print-config`. It prints YAML and exits
without loading a model.

## What the file contains

Every block is optional. A server needs models, from `models` or from a
`discover` scan.

| Block | What it sets |
|-------|--------------|
| [`models`](#models) | The models gmlx can run, each with its GGUF file and settings |
| [`aliases`](#aliases) | Extra model names, such as `coder` for `qwen3.8-27b-ud-q6@coding` |
| [`profiles`](#profiles) | Named sets of sampling, loading and prompt settings that any model can use |
| [`rules`](#rules) | A profile for every model whose id matches a pattern |
| [`discover`](#model-discovery) | Folders scanned at each start for GGUFs that have no entry |
| [`server`](#server) | Address, API key, model folders, memory budget and extra services |
| [`talk`](#voice) | The voice client's model, voice, wake phrase and listening thresholds |
| [`assistant`](#assistant) | Tool servers and long-term memory of the built-in assistant |
| [`launch`](#launch) | How `gmlx launch` runs clients in containers |
| [`theme`, `themes`](#chat-themes) | Colors of the terminal chat |

A key is named by its full path, such as `server.port`. `models.*.pin`
means the `pin` key of any entry under `models`.

## Models

Each entry under `models` is one model. The entry's key is the model's id,
which a request puts in its `model` field and which `gmlx run` and
`gmlx chat` take on the command line. You can rename an entry freely, but
an id cannot contain `@`, which separates an id from a profile name.

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

An entry whose file is missing is skipped with a warning, and the server
keeps running. `gmlx sync-models` removes such entries.

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="modelspath"></a>`path` | required | The GGUF file: absolute, relative to a [`server.model_dirs`](#servermodel_dirs) folder, or `hf:<org>/<repo>/<file.gguf>[@rev]` |
| <a id="modelsprofile"></a>`profile` | none | Profile for requests that name none: one under [`profiles`](#profiles) or an intent such as `coding` |
| <a id="modelsprofiles"></a>`profiles` | none | Changes to a profile for this model only, keyed by profile or intent name. Each value takes the keys of `overrides`. |
| <a id="modelsoverrides"></a>`overrides` | none | Settings that win over every profile, for this model only. Takes every profile key except `extends`. |
| <a id="modelsfamily"></a>`family` | detected | The [family defaults](family-defaults.md) to start from. Set it only when detection picks the wrong family. |
| <a id="modelsmmproj"></a>`mmproj` | none | Vision or audio projector GGUF that makes the model [multimodal](vlm.md) |
| <a id="modelsspeculative"></a>`speculative` | `false` | Speculative decoding with the GGUF's MTP head, or with `draft_gguf`. The output stays token-identical. |
| <a id="modelsdraft_gguf"></a>`draft_gguf` | none | A separate [drafter](glossary.md#drafter) GGUF. Setting it turns on `speculative`. |
| <a id="modelsnative_mtp"></a>`native_mtp` | `false` | Draft with the GGUF's own MTP head even when `draft_gguf` is set |
| <a id="modelsspeculative_width_cap"></a>`speculative_width_cap` | `null` | Speculate only while at most this many requests generate together. `null` takes the drafter's default, `0` removes the cap. |
| <a id="modelsadapter"></a>`adapter` | none | [LoRA adapter](lora.md) GGUF applied at load. Ids with the same `path` and different adapters share the base weights. |
| <a id="modelspin"></a>`pin` | `false` | Load at start and stay loaded. Only `POST /unload` unloads a pinned model. |
| <a id="modelsttl_s"></a>`ttl_s` | `server.defaults.ttl_s` | Seconds without a request before the model unloads. `null` or `0` turns off the timeout. |
| <a id="modelsstream"></a>`stream` | none | Run a model larger than memory. `experts` streams the routed experts from disk, `cpu` runs the whole model on the CPU. |

Each drafter has its own default `speculative_width_cap`, which
[Several requests at once](speculative-decoding.md#several-requests-at-once)
lists.

### Streaming keys

These keys apply only to a model with `stream` set.
[Models larger than memory](streaming.md) explains how to choose a
placement and how to size the `moe_*` keys, which all change the output.

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="modelsmoe_experts"></a>`moe_experts` | the model's count | Fixed number of experts per token, below the number the model was trained with |
| <a id="modelsmoe_expert_mass"></a>`moe_expert_mass` | off | Keep the smallest set of experts whose gate weights reach this share, above 0 and at most 1 |
| <a id="modelsmoe_miss_shed"></a>`moe_miss_shed` | off | Drop experts outside the decode [arena](glossary.md#arena) while the kept ones still cover this share of the gate weight |
| <a id="modelsmoe_layer_shed"></a>`moe_layer_shed` | off | Probability, between 0 and 1, of skipping a streamed layer's routed experts |
| <a id="modelsmoe_prestage"></a>`moe_prestage` | `ranked` | Experts read ahead: `ranked` reads the predicted ones, `keepers` only those that `moe_miss_shed` keeps, so it needs `moe_miss_shed` |
| <a id="modelsprefill_feeder"></a>`prefill_feeder` | `true` | Prefill reads expert weights directly from the GGUF |
| <a id="modelsdecode_feeder"></a>`decode_feeder` | `true` with `stream: experts` | Decoding keeps the most used experts in a wired arena |
| <a id="modelsstream_fast_disk"></a>`stream_fast_disk` | `auto` | Read-ahead for streamed decoding. `auto` tests the drive, `on` and `off` force the choice. |

## Aliases

An alias gives a model a second name, and can fix a profile to it. The
server lists each alias in `/v1/models` as its own entry, so a client that
picks models from a menu can choose a model and a profile together.

```yaml
aliases:
  fast: gemma-12b
  coder: qwen3.8-27b-ud-q6@coding
```

An alias cannot contain `@` or match a model id.

## Profiles

A profile is a named set of settings that any model can use. Every model
starts from its [family defaults](family-defaults.md), the sampling values
its publisher recommends, so a new file needs no profiles at all.

Some publishers also recommend values for a task, such as a lower
temperature for code. gmlx offers these as built-in profiles called
intents: `@coding`, `@instruct`, `@creative`, `@reasoning-low`,
`@reasoning-medium`, `@reasoning-high`, `@reasoning-max` and
`@reasoning-xhigh`. Add one to the model id to use it:

```sh
gmlx run qwen3.8-27b-ud-q6@coding --prompt "Write a binary search in Go."
```

Your own profiles go under `profiles`. A profile can start from an intent
with `extends` and change only what it sets. In the file, you write an
intent without the `@`, as in `extends: coding`. A profile with an
intent's name replaces that intent.

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

Settings come in layers, and a later layer wins over an earlier one:

| Layer | Set by |
|-------|--------|
| Family defaults | Built in, for each detected family |
| Server default profile | [`server.defaults.profile`](#serverdefaultsprofile) |
| Rule profile | The first matching entry in [`rules`](#rules) |
| Model profile | [`models.*.profile`](#modelsprofile), or `@name` on the request |
| Per-model profile change | [`models.*.profiles`](#modelsprofiles) |
| Model overrides | [`models.*.overrides`](#modelsoverrides) |
| Request fields | The request body |

A request's `@name` replaces the model's own profile rather than adding to
it. `gmlx profiles <id>` prints the sampling values a model resolves to
under each of its profiles, with every layer applied.

### Profile keys

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="profilesextends"></a>`extends` | none | Profile or intent to start from. Each key this profile sets replaces the inherited value. |
| <a id="profilessampling"></a>`sampling` | none | Defaults for the [sampling](#sampling) request fields. A field the request sends wins. |
| <a id="profilesload"></a>`load` | none | [Model loading](#model-loading) keys |
| <a id="profilescache"></a>`cache` | none | [Prompt cache](#prompt-cache) keys that change [`server.cache`](#servercache) for this profile's models |
| <a id="profilessystem"></a>`system` | none | System prompt, used only when the request has no system message |
| <a id="profileschat_template"></a>`chat_template` | the GGUF's | Inline Jinja, or a path to a `.jinja` or `.txt` file, that replaces the GGUF's chat template |
| <a id="profileschat_template_kwargs"></a>`chat_template_kwargs` | none | Variables passed to the chat template on each request. Keys the request sends win. |
| <a id="profilesthinking"></a>`thinking` | the template's | Reasoning `on`, `off` or `adaptive`, on any family |
| <a id="profilesreasoning_effort"></a>`reasoning_effort` | the template's | A reasoning level the template accepts, such as `low`, `medium` or `high` |

`preserve_thinking: true` in `chat_template_kwargs` keeps earlier `<think>`
blocks in the prompt on the Qwen3.6 and Gemma 4 templates, so an agent sees
its earlier reasoning. A template applies at load, so two ids with different
`chat_template` values are two loaded copies.

`thinking` and `reasoning_effort` map to the variable that each model's
template reads, so one profile works across families:

| Family | Template variable | Values |
|--------|-------------------|--------|
| Qwen3.x, GLM | `enable_thinking` | `true` or `false` |
| MiniMax-M3 | `thinking_mode` | Three states, so `adaptive` works |
| Kimi K2.x | `thinking` | `true` or `false` |
| Hy3 | `reasoning_effort` | Levels that include `no_think` |
| gpt-oss | `reasoning_effort` | `low`, `medium` or `high`; reasoning cannot be turned off |

A request's `enable_thinking`, `thinking` or `reasoning_effort` field wins
over the profile. `gmlx run` and `gmlx chat` take `--thinking` and
`--reasoning-effort`.

## Rules

Rules give a profile to every model whose id matches a pattern, so a group
of models can share settings. The first rule that matches wins. A rule
ranks below a model's own `profile` and above
[`server.defaults.profile`](#serverdefaultsprofile).

```yaml
rules:
  - {match: "*coder*", profile: review}
  - {match: "qwen3.8-*", profile: agent}
```

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="rulesmatch"></a>`match` | required | Model ids the rule applies to, as a shell wildcard pattern such as `qwen3.8-*`, not a regular expression |
| <a id="rulesprofile"></a>`profile` | required | The profile the matching models get |

## Sampling

A `sampling` block sets defaults for the request fields that control
generation. It goes in a profile or in a model's `overrides`. Each key is
also a request field of the same name, and the request's value wins. A
default of "family" comes from the [family defaults](family-defaults.md),
and `gmlx profiles <id>` prints it.

```yaml
profiles:
  precise:
    sampling:
      temperature: 0.2
      top_p: 0.9
      max_tokens: 4096
      stop: ["</answer>"]
```

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="samplingtemperature"></a>`temperature` | family | How random each token choice is. `0` always takes the most likely token. Also accepted as `temp`. |
| <a id="samplingtop_p"></a>`top_p` | family | Sample only from the most likely tokens whose probabilities add up to this value. `0` turns it off. |
| <a id="samplingtop_k"></a>`top_k` | family | Sample only from this many of the most likely tokens. `0` turns it off. |
| <a id="samplingmin_p"></a>`min_p` | family | Drop tokens below this share of the most likely token's probability. `0` turns it off. |
| <a id="samplingmax_tokens"></a>`max_tokens` | no limit | Stop after this many tokens. [`gmlx serve --max-tokens`](cli.md#gmlx-serve) caps requests that set none. |
| <a id="samplingseed"></a>`seed` | none | Repeatable sampling for this request, when its batch and the speculation setting also match |
| <a id="samplingstop"></a>`stop` | none | A string, or a list of strings, that ends generation. The Anthropic endpoint uses its own `stop_sequences`. |
| <a id="samplingrepetition_penalty"></a>`repetition_penalty` | off | Lowers the chance of tokens seen in the last `repetition_context_size` tokens. `1` has no effect. |
| <a id="samplingrepetition_context_size"></a>`repetition_context_size` | `20` | Recent tokens that `repetition_penalty` looks at |
| <a id="samplingpresence_penalty"></a>`presence_penalty` | off | Lowers the chance of every token already generated, by the same amount |
| <a id="samplingfrequency_penalty"></a>`frequency_penalty` | off | Lowers the chance of each token in proportion to how often it was generated |
| <a id="samplingxtc_probability"></a>`xtc_probability` | off | Chance that XTC removes the most likely candidates, to vary the text. Not available on speculative models. |
| <a id="samplingxtc_threshold"></a>`xtc_threshold` | `0.0` | XTC removes candidates whose probability is above this value |
| <a id="samplingenable_thinking"></a>`enable_thinking` | the template's | Whether the chat template opens a thinking block. A profile's [`thinking`](#profilesthinking) sets it on any family. |
| <a id="samplingthinking_budget"></a>`thinking_budget` | no limit | Close the thinking block after this many reasoning tokens. A model with a separate drafter refuses it. |
| <a id="samplingthinking_start_token"></a>`thinking_start_token` | `<think>` | Text that opens the model's reasoning. Family defaults set it for models that use other text. |
| <a id="samplingthinking_end_token"></a>`thinking_end_token` | `</think>` | Text that closes the model's reasoning |

## Model loading

A `load` block changes how a model is built, for example to quantize its KV
cache so that a long context uses less memory. It goes in a profile or in a
model's `overrides`. Two ids that differ in a load key are two loaded
copies. When you serve a single GGUF with no config, each key is also a
[`gmlx serve` flag](cli.md#gmlx-serve), such as `--kv-bits`.

```yaml
profiles:
  long-context:
    load: {kv_bits: 6, kv_quant_scheme: kvarn}
```

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="loadkv_bits"></a>`kv_bits` | none, `6` with `kvarn` | Bits per KV cache value. `uniform` takes 2, 3, 4, 6 or 8, and `kvarn` takes 2, 3, 4, 5, 6 or 8. |
| <a id="loadkv_quant_scheme"></a>`kv_quant_scheme` | `uniform` | `uniform` (affine) or `kvarn`, which [KV cache quantization](kv-quantization.md) compares |
| <a id="loadkv_group_size"></a>`kv_group_size` | `64` | Group size of `uniform` quantization |
| <a id="loadkv_tail_tokens"></a>`kv_tail_tokens` | `1024` | Newest tokens that stay fp16 under `kvarn`, as a multiple of 128 |
| <a id="loadquantized_kv_start"></a>`quantized_kv_start` | `0` | Tokens the cache holds in fp16 before it quantizes. Batched requests and `kvarn` quantize from the first token. |
| <a id="loadmax_kv_size"></a>`max_kv_size` | no cap | Most context tokens a request may use. `gmlx run` and `gmlx chat` also make the cache a rotating window of this size. |

`prefill_step_size` and `dtype` apply to the whole server, under
[Scheduling](#scheduling).

## Prompt cache

The prompt cache keeps the computed start of recent prompts, so a request
that begins the same way skips that part of prefill. Agents and long chats
gain the most, because they resend the same history every turn.
[`server.cache`](#servercache) sets it for every model, and a `cache` block
in a profile or a model's `overrides` changes it.
[Prompt cache](prompt-cache.md) says what it restores for each architecture.

```yaml
server:
  cache:
    enabled: true
    disk: {path: ~/.cache/gmlx/apc, max_gb: 100}
```

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="cacheenabled"></a>`enabled` | `false` | Turns the prompt cache on. `gmlx init` writes `true`. |
| <a id="cacheblock_size"></a>`block_size` | chosen per model | Tokens per cache block: 16, 32, 64, 128 or 256 |
| <a id="cachenum_blocks"></a>`num_blocks` | grows to fit | Blocks in the pool that all cached prompts share. A full pool drops the oldest prompts. |
| <a id="cacheexact_entries"></a>`exact_entries` | `4` | Whole-prompt copies kept for hybrid and recurrent models. Each copy is a full cache. |
| <a id="cachehash"></a>`hash` | `fast` | How blocks are identified. `sha256` is stable across processes and costs more per token. |
| <a id="cachedisk"></a>`disk` | `false` | A second tier on the SSD, which survives unloads and restarts. `true` uses the default `path`. |
| <a id="cachediskpath"></a>`disk.path` | `~/.cache/gmlx/apc` | Folder of the SSD tier, under `$XDG_CACHE_HOME` when that is set. Setting it turns the tier on. |
| <a id="cachediskmax_gb"></a>`disk.max_gb` | no limit | Space each namespace may use, in GB. By default each model has its own namespace. |
| <a id="cachedisknamespace"></a>`disk.namespace` | the model path | The partition on disk that the model reads and writes |
| <a id="cachediskworkers"></a>`disk.workers` | `1` | Threads that write entries to disk |
| <a id="cachediskread_mode"></a>`disk.read_mode` | `direct` | How entries are read back: `direct` with plain file reads, `mmap` by mapping the files |

## Model discovery

The `discover` block lists folders that the server scans at each start and
reload. Every GGUF without an entry under `models` becomes a model, named
after its file, so a new file appears after a restart or a reload.
`gmlx run`, `gmlx list` and `gmlx rm` scan the same folders. To write the
entries into the file once instead, run `gmlx init` or `gmlx sync-models`.

```yaml
discover:
  - dir: ~/models
    recursive: true
```

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="discoverdir"></a>`dir` | `null` | The folder to scan. `null` scans every folder in `server.model_dirs`. |
| <a id="discoverrecursive"></a>`recursive` | `false` | Also scan subfolders |
| <a id="discoverpair_mmproj"></a>`pair_mmproj` | `true` | Pair each `mmproj*.gguf` with the model in the same folder that it matches |
| <a id="discoverspeculative"></a>`speculative` | `auto` | `auto` and `true` turn on speculative decoding for models with an MTP head, and pair drafters. `false` never does. |

## Server

The `server` block sets where the server listens, who may call it, where
models are found, how much memory they may use, and which extra services
run. With no `server` block, the server listens on `127.0.0.1:8080` with
the prompt cache off. The keys below are under `server`, as in the
[complete example](#complete-example).

### Address and authentication

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="serverhost"></a>`host` | `127.0.0.1` | The address to listen on. Any address other than loopback needs `api_key` or `no_auth`. |
| <a id="serverport"></a>`port` | `8080` | The port to listen on |
| <a id="serverapi_key"></a>`api_key` | none | Key every request must send, as `Authorization: Bearer <key>` or `x-api-key: <key>`. Read only from this file. |
| <a id="serverno_auth"></a>`no_auth` | `false` | Accept an address other than loopback without a key, for a proxy that handles authentication |
| <a id="servermedia_urls"></a>`media_urls` | `false` | Let a request name an image, audio or video by an `http(s)://` URL, which the server fetches |
| <a id="servercors_origins"></a>`cors_origins` | none | Web page origins that may call the server, such as `https://chat.example.com` or `http://192.168.1.20:3000` |

The client commands `chat`, `talk`, `ps`, `systemone`, `launch` and
`launch menubar` take `--api-key` to send the key. A change to `api_key`,
`media_urls` or `cors_origins` applies after `gmlx restart`.

With `media_urls` on, any client that reaches the port can make the Mac
fetch from any public host. The server still refuses hosts on the Mac and
the local network.

Pages on `localhost`, `127.0.0.1` and `[::1]`, and desktop apps built on
Electron, Tauri or VS Code webviews, may always call the server. curl sends
no origin and is not affected. For a browser extension, add the origin that
the server log names, such as `chrome-extension://<id>`. These wildcards
allow every extension of one browser:

- `chrome-extension://*` for Chrome, Edge and other Chromium browsers
- `moz-extension://*` for Firefox
- `safari-web-extension://*` for Safari, which changes an extension's ID at
  every start

A listed origin can use the tools of [served assistants](#served-assistants),
which run on the Mac, so prefer one extension's own origin to a wildcard.

### Model folders

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="servermodel_dirs"></a>`model_dirs` | none | Folders searched in order for a relative `path`, `mmproj`, `draft_gguf` or `adapter`. `gmlx pull` downloads into the first. |
| <a id="serverhf_cache"></a>`hf_cache` | `false` | Run the Hugging Face libraries offline, so the server never downloads. `gmlx init --from-hf-cache` sets it. |

### Services

Each service adds an endpoint. `true` picks the default model.
[Speech, embeddings and rerank](services.md) lists the models each one
accepts.

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="serverstt"></a>`stt` | none | Speech-to-text model: an alias such as `whisper-turbo`, a Hugging Face repo id or a folder |
| <a id="servertts"></a>`tts` | none | Text-to-speech model: an alias such as `kokoro`, a Hugging Face repo id or a folder |
| <a id="serverembeddings"></a>`embeddings` | none | Embeddings model: a GGUF path, an `hf:` reference, an alias, or an mlx-embeddings repo id or folder |
| <a id="serverrerank"></a>`rerank` | none | Reranker: a Qwen3-Reranker GGUF path, an `hf:` reference or an alias |

### Memory and residency

The server keeps several models loaded up to `budget_gb`, and unloads a
model after `ttl_s` idle seconds or when the budget needs room.
[Memory](memory.md) explains which models stay loaded and when two ids
share one loaded copy.

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="serverbudget_gb"></a>`budget_gb` | 0.8 of the GPU working set | Memory that loaded models may use together, in GB |
| <a id="servermax_models"></a>`max_models` | no limit | Most models loaded at once, checked after the budget |
| <a id="servercache_limit_gb"></a>`cache_limit_gb` | 4 to 12 GiB, set at start | Limit of the MLX buffer cache, in GiB. `0` turns the cache off, a negative value removes the limit. |
| <a id="serverdefaultsmodel"></a>`defaults.model` | the only model, else none | Model for requests that name none, and the first one loaded when no model is pinned |
| <a id="serverdefaultsttl_s"></a>`defaults.ttl_s` | `900` | Seconds without a request before a model that is not pinned unloads. `null` or `0` turns off the timeout. |
| <a id="serverdefaultspreload"></a>`defaults.preload` | none | Models loaded at start, one at a time, after the pinned models. `all` loads every model. |
| <a id="serverdefaultsprofile"></a>`defaults.profile` | none | Profile for every model, below rules and the model's own profile |

[The MLX buffer cache](memory.md#the-mlx-buffer-cache) says when to change
`cache_limit_gb`.

### Scheduling

These keys apply to every model. Each one except `token_queue_timeout_s`
also has a [`gmlx serve` flag](cli.md#gmlx-serve).

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="serverprefill_step_size"></a>`prefill_step_size` | `2048` | Prompt chunk size in tokens. A lower value lowers a long prompt's memory peak and slows prefill. |
| <a id="serverdtype"></a>`dtype` | `auto` | Type of activations, unquantized weights and KV cache: `bfloat16`, `float16`, or `auto` (`float16` on M1 and M2) |
| <a id="serverdecode_prefill_ratio"></a>`decode_prefill_ratio` | `auto` | How a new prompt's prefill shares the GPU with generating requests. `auto` keeps generation above half its speed. |
| <a id="serverprefill_tick_ms"></a>`prefill_tick_ms` | `500` | Longest prefill chunk, in ms, while other requests generate. A longer chunk is halved. `0` turns off the halving. |
| <a id="servertoken_queue_timeout_s"></a>`token_queue_timeout_s` | `1800` | Seconds a request may wait for its next token before it fails. `0` waits forever. |

`decode_prefill_ratio` also takes a number, such as `1.0`, which makes each
prefill chunk wait until the generating requests have had that multiple of
its GPU time. `0` runs one chunk per generation step.
[Concurrent requests](concurrency.md) shows the effect.

### Features

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="serverfamily_defaults"></a>`family_defaults` | `true` | Start each request from its [family defaults](family-defaults.md) and offer the intents. `false` turns off both. |
| <a id="serverstochastic_mtp"></a>`stochastic_mtp` | `false` | Accept more speculative tokens by rejection sampling. The output is no longer token-identical. A reload does not apply it. |
| <a id="servergpu_keepwarm"></a>`gpu_keepwarm` | on | Keep the GPU clock up between tokens of a streamed model with a decode feeder. `false` turns it off. |
| <a id="servermenubar"></a>`menubar` | `true` | Let a server started in the background open the [menu bar app](menubar.md) |
| <a id="servercache"></a>`cache` | off | [Prompt cache](#prompt-cache) settings for every model |

[Stochastic acceptance](speculative-decoding.md#stochastic-acceptance)
describes `stochastic_mtp`.

### Structured decisions

These keys set up `POST /v1/systemone`, which
[Structured decisions](decisions.md) describes. `canvas`, `constrained` and
the three think keys apply only to DiffusionGemma. A reload applies new
values to the next request.

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="serversystemonemodel"></a>`systemone.model` | none | Model id or alias, with an optional `@profile`, for a request that names no configured model |
| <a id="serversystemonecanvas"></a>`systemone.canvas` | `64` | Most tokens one [structured read](glossary.md#structured-read) fills, as a positive multiple of 16 |
| <a id="serversystemoneconstrained"></a>`systemone.constrained` | `true` | Compute probabilities over the answer tokens only. `false` uses the full vocabulary. |
| <a id="serversystemonemax_questions"></a>`systemone.max_questions` | `64` | Most questions in one request |
| <a id="serversystemonemax_samples"></a>`systemone.max_samples` | `32` | Most `samples` and `auto_max` a request may ask for. A larger value is lowered to this one. |
| <a id="serversystemonethink"></a>`systemone.think` | `0` | Thought budget in tokens, from 0 to 4096, for a request that sets none. `"auto"` thinks only when an answer is uncertain. |
| <a id="serversystemonethink_threshold"></a>`systemone.think_threshold` | `0.8` | Under `"auto"`, think when the chosen label's probability is below this value |
| <a id="serversystemonethink_budget"></a>`systemone.think_budget` | `64` | Under `"auto"`, the thought budget in tokens |

### Served assistants

The server can run the [assistant's](#assistant) tool loop itself and offer
each assistant as a model. A client that names the assistant as its model
gets tools without a loop of its own, as
[Assistant](assistant.md#served-assistants) describes.

```yaml
server:
  assistants:
    helper:
      model: qwen3.8-27b-ud-q6
      mcp: []
```

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="serverassistantsmodel"></a>`assistants.*.model` | required | The configured model that answers for the assistant |
| <a id="serverassistantsmemory"></a>`assistants.*.memory` | `false` | Long-term memory, in one store that all the assistant's clients share |
| <a id="serverassistantsmcp"></a>`assistants.*.mcp` | `null` | Tool servers, with the keys of [`assistant.mcp`](#assistantmcp). `null` uses `assistant.mcp`, `[]` gives no tools. |
| <a id="serverassistant_allow_remote"></a>`assistant_allow_remote` | `false` | Allow served assistants on an address other than loopback. Their tools run on the server's Mac. |

## Voice

The `talk` block sets up the voice client [`gmlx talk`](talk.md) and the
menu bar app's voice sessions. Most keys also have a
[`gmlx talk` flag](cli.md#gmlx-talk), which wins over the file.

```yaml
talk:
  model: qwen3.8-27b-ud-q6@instruct
  voice: af_heart
  mode: vad
  vad: {silence_ms: 450}
```

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="talkmodel"></a>`model` | the server's default | Model id or alias to talk to, with an optional `@profile` |
| <a id="talkvoice"></a>`voice` | the server's default | A Kokoro preset or a Qwen3-TTS speaker name |
| <a id="talkspeed"></a>`speed` | `1.0` | Speaking speed, from 0.25 to 4 |
| <a id="talksystem"></a>`system` | a prompt for speakable text | System prompt for the spoken persona. `null` or `""` sends none. |
| <a id="talklanguage"></a>`language` | detected | Language hint for speech recognition, such as `en` |
| <a id="talkmax_tokens"></a>`max_tokens` | no limit | Most tokens in a spoken reply |
| <a id="talkmode"></a>`mode` | `wake` | How listening starts: `wake` on the wake phrase, `vad` on any speech, `ptt` on Space, `text` from typed prompts |
| <a id="talkwake_word"></a>`wake_word` | `hey assistant` | The wake phrase, any English text. It needs no training. |
| <a id="talkwake_threshold"></a>`wake_threshold` | `0.3` | Confidence, from 0 to 1, that the wake phrase needs. A higher value gives fewer false wakes. |
| <a id="talkpush_to_talk_modifier"></a>`push_to_talk_modifier` | `globe` | Modifier held with Space for the menu bar hotkey: `globe`, `right-command`, `right-option` or `control` |
| <a id="talkinput_device"></a>`input_device` | the system input | Microphone, by part of its name or by index. `/devices` in `gmlx talk` lists them. |
| <a id="talkoutput_device"></a>`output_device` | the system output | Speaker, by part of its name or by index |
| <a id="talkchime"></a>`chime` | `true` | Play a sound on wake and at the end of a turn |
| <a id="talkbrain"></a>`brain` | `chat` | What answers: `chat` is the plain model, `assistant` adds the [assistant](#assistant) block's tools and memory |
| <a id="talkvadthreshold"></a>`vad.threshold` | `0.6` | Speech probability, from 0 to 1, above which audio counts as speech |
| <a id="talkvadsilence_ms"></a>`vad.silence_ms` | `550` | Pause in ms that ends an utterance. Lower answers sooner and cuts off more sentences. |
| <a id="talkvadmin_speech_ms"></a>`vad.min_speech_ms` | `300` | Utterances shorter than this many ms are dropped as noise |
| <a id="talkvadpre_roll_ms"></a>`vad.pre_roll_ms` | `400` | Audio in ms kept from before speech starts |

## Assistant

The `assistant` block gives the built-in assistant its tools and memory.
`gmlx chat --assistant`, `gmlx talk` with `brain: assistant`, and
[served assistants](#served-assistants) use it. The coding agents that
`gmlx launch` connects do not. [Assistant](assistant.md) explains the tool
loop and memory.

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

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="assistantmax_tool_rounds"></a>`max_tool_rounds` | `8` | Most rounds of tool calls in one turn before the model must answer |
| <a id="assistanttool_timeout_s"></a>`tool_timeout_s` | `60` | Most seconds one tool call may take |
| <a id="assistantmcp"></a>`mcp` | none | [MCP](glossary.md#mcp) servers that provide the tools. Each entry has a `name` and one of `command` and `url`. |
| <a id="assistantmcpname"></a>`mcp[].name` | required | A name for the server, unique in the list |
| <a id="assistantmcpcommand"></a>`mcp[].command` | none | Command that starts a server over stdio, as a list of arguments or a shell-style string |
| <a id="assistantmcpurl"></a>`mcp[].url` | none | Address of a server over streamable HTTP |
| <a id="assistantmcpenv"></a>`mcp[].env` | none | Environment variables for a stdio server, such as the token it needs |
| <a id="assistantmemoryenabled"></a>`memory.enabled` | `true` | Remember facts across conversations. Needs [`server.embeddings`](#serverembeddings). |
| <a id="assistantmemorypath"></a>`memory.path` | `~/.local/share/gmlx/assistant-memory.db` | Database file of the memory, under `$XDG_DATA_HOME` when that is set |
| <a id="assistantmemorytop_k"></a>`memory.top_k` | `4` | Remembered facts added to each turn |
| <a id="assistantmemoryextract"></a>`memory.extract` | `true` | Store short facts that the model extracts. `false` stores each exchange as it is. |
| <a id="assistantmemoryttl_days"></a>`memory.ttl_days` | forever | Facts older than this many days are forgotten at start |
| <a id="assistantmemorymax_items"></a>`memory.max_items` | `20000` | Most facts the store holds |

A stdio server gets only `HOME`, `PATH`, `SHELL`, `TERM`, `USER` and
`LOGNAME` from your environment, so put anything else it needs in `env`.
Its log goes to `~/.cache/gmlx/mcp-<name>.log`.

## Launch

The `launch` block configures [container mode](launch-container.md) and
defines [custom agents](launch-agents.md). `launch` reads it only from the
config file in your home folder, `~/.config/gmlx/gmlx.yaml` or
`~/.gmlx.yaml`, never from a file that `--config` names. The server ignores
the block.

Keys under `launch.container` apply to every client. Each one also works
under `launch.container.clients.<client>` for a single client, where the
client's value wins and lists from both levels add up. Six keys exist only
under a client. Agents take most of these keys, plus five of their own.

This block turns on container mode with more memory, gives Claude Code a
volume and a seed, runs Open WebUI from its official image with the served
assistant `home`, and defines the agent `research-bot`:

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
        image: ghcr.io/open-webui/open-webui:v0.11.4
        command: image
        assistants: [home]
  agents:
    research-bot:
      runtime: python
      command: [research-bot]
```

### `launch.container`

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="launchcontainerenabled"></a>`enabled` | `false` | Run clients in a container. `--container` and `--no-container` override it. |
| <a id="launchcontainermount_cwd"></a>`mount_cwd` | `true`, `false` for `open-webui` and `elia` | Share the current folder read-write at the same path, and start the client there. `--mount-cwd` and `--no-mount-cwd` override it. |
| <a id="launchcontainermounts"></a>`mounts` | none | More folders to share, each `PATH[:DST][:ro]`. `PATH` is a full path or starts with `~`. `--mount` adds entries. |
| <a id="launchcontainervolumes"></a>`volumes` | none | Named volumes, each `NAME:/path[:SIZE]`, created with `SIZE` when missing. The default size is `32G`. |
| <a id="launchcontainerforward"></a>`forward` | none | Ports the container reaches on its own `127.0.0.1`, which lead to the same ports on the Mac's `127.0.0.1` |
| <a id="launchcontainernetwork"></a>`network` | `default` | `default` reaches the internet and your local network. `none` reaches only the gmlx server and forwarded ports. `--network` overrides it. |
| <a id="launchcontainercpus"></a>`cpus` | `4` | CPUs the container gets |
| <a id="launchcontainermemory"></a>`memory` | `4G` | Memory the container gets, such as `6144M`. It counts against the memory the model server can use. |
| <a id="launchcontainerssh_agent"></a>`ssh_agent` | `false` | `true` lets the client use the SSH agent in `SSH_AUTH_SOCK`. A socket path gives it that agent instead. |
| <a id="launchcontainerenv"></a>`env` | none | Variables for the container: `NAME` passes yours in, `NAME=VALUE` sets one. An entry can replace a client setting such as `OPENAI_API_KEY`. |
| <a id="launchcontaineropen_browser"></a>`open_browser` | `true` | Open a [browser app](launch-container.md#browser-apps) in the Mac's browser once it answers. `false` prints the address. |
| <a id="launchcontainerpaste_copy_max"></a>`paste_copy_max` | `1G` | Largest pasted file that `launch` copies from another disk into the private home. A file on the same disk is cloned, with no limit. |

A volume under a client is separate for each project, while a volume at
the top level is shared by every project and client. With `ssh_agent`, the
client can sign with every key in the agent. To use 1Password's agent, give
its socket path,
`~/Library/Group Containers/2BUA8C4S2C.com.1password/t/agent.sock`.

More on each key:

- [Volumes](container-access.md#volumes),
  [Forwarded ports](container-access.md#forwarded-ports) and
  [Pasting files and images](container-access.md#pasting-files-and-images)
- [Limits](container-security.md#limits) for `memory`
- [Access you turn on](container-security.md#access-you-turn-on) for
  `ssh_agent`, `forward`, `env` and `network`

### `launch.container.clients`

Settings for single clients, keyed by `claude-code`, `opencode`, `pi`,
`omp`, `hermes`, `goose`, `aichat`, `elia`, `open-webui` or `dsh`. Each
client takes the keys above and these six:

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="launchcontainerclientsimage"></a>`image` | the image gmlx builds | Run this image, a local tag or a registry reference. Not with `build` or `packages`. |
| <a id="launchcontainerclientsbuild"></a>`build` | the image gmlx builds | Build the image from this Containerfile, or from a folder with a `Containerfile` or `Dockerfile`. A full path or one that starts with `~`. |
| <a id="launchcontainerclientscommand"></a>`command` | the client's own | A list replaces the client's command. `image` runs the image's own ENTRYPOINT and CMD. |
| <a id="launchcontainerclientspackages"></a>`packages` | none | Debian packages added to the image gmlx builds. With `build`, only when the Containerfile starts from the client's `:base`. |
| <a id="launchcontainerclientsseed"></a>`seed` | none | Files and folders copied from your home folder into the private home, at the same path |
| <a id="launchcontainerclientsassistants"></a>`assistants` | none | [Served assistants](#served-assistants) the client can use. The others stay hidden from it. |

[Container images](container-images.md) covers `image`, `build` and
`command`. Here a client gets extra packages and two seeds:

```yaml
# doctest: build
launch:
  container:
    clients:
      claude-code:
        packages: [make, python3, postgresql-client]
        seed: [~/.claude/CLAUDE.md, ~/.claude/commands]
```

[Seeds](container-access.md#seeds) says when a seed is copied.
`--seed-instructions` also seeds the client's instruction and skill files,
which [Instructions and skills](container-access.md#instructions-and-skills)
lists. A served assistant's tools run on the Mac, so list assistants only
for chat apps where you write the messages, as
[What the client reaches on the server](container-security.md#what-the-client-reaches-on-the-server)
explains.

### `launch.agents`

Each entry under `launch.agents` defines a [custom agent](launch-agents.md),
such as `research-bot` in the example above. A name starts with a
lowercase letter, holds lowercase letters, digits, `-` and `_`, has at most
32 characters, and is not a client's name or `menubar`.

An agent needs `command` and one of `runtime`, `image` or `build`. It also
takes every key of `launch.container` except `enabled` and `clients`, and
the client keys `image`, `build`, `seed` and `assistants`. The table lists
`command`, which an agent always needs, and the five keys only agents take:

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="launchagentscommand"></a>`command` | required | A list that starts the agent, or `image` for the image's own ENTRYPOINT and CMD. With `runtime`, the list runs in the project's environment. |
| <a id="launchagentsruntime"></a>`runtime` | none | `python` installs the project's dependencies with uv before the command runs. Without it, the command runs as it is in the image. |
| <a id="launchagentssource"></a>`source` | the current folder | The project folder that uv installs, shared read-only. A full path or one that starts with `~`. Only with `runtime`. |
| <a id="launchagentsapi"></a>`api` | `openai` | `openai` sets the `OPENAI_` variables, `anthropic` the `ANTHROPIC_` ones, and `none` neither |
| <a id="launchagentsmodel"></a>`model` | the server's default | The served model in `GMLX_MODEL`. `--model` overrides it. |
| <a id="launchagentsweb_port"></a>`web_port` | none | Port of the agent's web app in the container, which makes it a browser app. Without it, the agent runs in the terminal. |

[Custom agents](launch-agents.md) covers
[dependencies at run time](launch-agents.md#dependencies-at-run-time),
[the source folder](launch-agents.md#the-source-folder),
[what the agent gets](launch-agents.md#what-the-agent-gets) and
[a browser interface](launch-agents.md#a-browser-interface).

## Chat themes

The `theme` and `themes` keys set the colors of [`gmlx chat`](chat.md).

```yaml
theme: my-black
themes:
  my-black:
    extends: dark
    heading: {bold: true, rgb: "#88c0d0"}
```

| Key | Default | Meaning |
|-----|---------|---------|
| <a id="theme"></a>`theme` | `dark` | Theme each chat starts with: a built-in name or one from `themes`. `--theme` and `/theme` change it. |
| <a id="themes"></a>`themes` | none | Your own themes, keyed by name. A theme with a built-in name replaces it. |

Each theme sets styles for kinds of text, such as `heading` and `thinking`,
and takes `extends` for the ones it leaves out. [Themes](chat.md#themes)
lists them.

## Changing the file

A running server reads its file again on `POST /v1/reload` or `SIGHUP`, and
the commands that edit the file send the reload for you:

```sh
curl -X POST http://127.0.0.1:8080/v1/reload   # with api_key set, add -H "Authorization: Bearer <key>"
gmlx restart                                   # applies everything, such as a new api_key
```

Loaded models stay loaded through a reload. A change to a model's load
settings, such as `mmproj` or `speculative`, applies the next time it loads.

| Command | What it changes |
|---------|-----------------|
| `gmlx sync-models` | Adds new GGUFs from `model_dirs` and removes entries whose file is gone |
| `gmlx pull` | Adds an entry for each GGUF it downloads |
| `gmlx rm` | Deletes a model's files and its entry |

These commands keep your comments and formatting. Pass `--no-reload` to
`init`, `sync-models` or `rm` to leave a running server alone, and
`--no-register` to `gmlx pull` to leave the file alone.

## Flags and environment variables

Some settings also have a `gmlx serve` flag or an environment variable.
The flag wins, then the key in the file, then the variable. Two variables
win over the file: `GMLX_CACHE_LIMIT_GB` over `server.cache_limit_gb`, and
`GMLX_MTP_WIDTH_CAP` over each model's `speculative_width_cap`. The flags
are in the [CLI reference](cli.md#gmlx-serve) and the variables in
[Environment variables](env-vars.md).

## Complete example

This file uses the blocks most setups need:

```yaml
# doctest: build
server:
  model_dirs: [~/models]
  budget_gb: 96
  cache:
    enabled: true
    disk: {path: ~/.cache/gmlx/apc, max_gb: 200}
  defaults:
    model: qwen3.8-27b-ud-q6
    ttl_s: 900
profiles:
  review:
    extends: coding
    system: "You are a terse senior engineer."
    sampling: {temperature: 0.2, max_tokens: 4096}
  long-context:
    load: {kv_bits: 8}
rules:
  - {match: "*coder*", profile: review}
models:
  qwen3.8-27b-ud-q6:
    path: Qwen3.8-27B-UD-Q6_K.gguf
    speculative: true
    pin: true
  gemma-31b:
    path: google_gemma-4-31B-it-Q6_K_L.gguf
    draft_gguf: gemma-4-31B-it-assistant.Q8_0.gguf
  gemma-e4b:
    path: gemma-4-E4B-it-Q6_K.gguf
    mmproj: mmproj-gemma-4-E4B-it-bf16.gguf
    profile: long-context
aliases:
  coder: qwen3.8-27b-ud-q6@review
assistant:
  mcp:
    - {name: clock, command: [uvx, mcp-server-time]}
```

The smallest useful file has one model with a path:

```yaml
# doctest: build
models:
  my-model:
    path: ~/models/my-model-Q4_K_M.gguf
```
