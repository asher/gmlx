# Configuration file

`gmlx.yaml` lists the models gmlx knows by name, the settings each model
runs with, and the settings of the server itself. `gmlx serve` reads it at
start, and the other commands read it to turn a model name into a GGUF
file, so a name that works in a request also works with `gmlx run` and
`gmlx chat`. Every key is described in
[Configuration keys](server-config.md).

- [Create the file](#create-the-file)
- [Where gmlx looks](#where-gmlx-looks)
- [What the file contains](#what-the-file-contains)
- [Models and their names](#models-and-their-names)
- [How a request gets its settings](#how-a-request-gets-its-settings)
- [Which models stay loaded](#which-models-stay-loaded)
- [Changing the file](#changing-the-file)
- [Flags and environment variables](#flags-and-environment-variables)

## Create the file

`gmlx init` scans a folder of GGUF files and writes a config with one entry
for each model it finds:

```sh
gmlx init --models-dir ~/models
```

The file goes to `~/.config/gmlx/gmlx.yaml`. With no flags on a terminal,
`init` asks its questions as a wizard instead. The scan reads only the GGUF
headers, so it takes seconds even for a large folder. This is the part of
the result that does the work, with the comments removed:

```yaml
server:
  host: 127.0.0.1
  port: 8080
  model_dirs:
    - ~/models
  hf_cache: false
  defaults:
    ttl_s: 900
  cache:
    enabled: true
    disk: false

models:
  gemma-4-12b-it-q4:
    path: gemma-4-12b-it-Q4_K_M.gguf
  qwen3.6-27b-q4:
    path: Qwen3.6-27B-Q4_K_S.gguf
```

The file that `init` writes also carries every other option as a commented
example with its default, so you can turn a setting on by removing the `#`.
Start the server with `gmlx serve`, which finds the file on its own.

## Where gmlx looks

A command that needs the file uses the first of these that exists:

1. `./gmlx.yaml` in the current directory
2. `~/.config/gmlx/gmlx.yaml`, where `gmlx init` writes
3. `~/.gmlx.yaml`

The file in the current directory comes first, so a project can carry its
own models and settings. Pass `--config FILE` to read a different file.
Without any file, `gmlx serve` scans the current directory for GGUFs and
prints a hint to run `init`.

To see the configuration a server would run with, including every default
you did not set, run `gmlx serve --print-config`. It prints the result as
YAML and exits without loading a model.

## What the file contains

The file is a YAML mapping, and every top-level key is optional. A server
needs only `models`, and everything else has a default.

| Key | Holds | Read by |
|-----|-------|---------|
| `server` | the address, API key, model folders, memory budget and optional services | `serve`, `launch`, `pull`, `menubar` |
| `models` | one entry per model, with its GGUF path and settings | every command that takes a model name |
| `aliases` | other names for a model, optionally with a profile | every command that takes a model name |
| `profiles` | named sets of sampling, loading and prompt settings | every command that takes a model name |
| `rules` | name patterns that give the matching models a profile | every command that takes a model name |
| `discover` | folders scanned at each server start, whose GGUFs join `models` | `serve` |
| `talk` | voice chat settings | `talk`, `menubar` |
| `assistant` | tools and memory for the built-in assistant | `chat --assistant`, `talk`, `serve` |
| `theme`, `themes` | chat colors | `chat` |

[Configuration keys](server-config.md) covers `server`,
`models`, `aliases`, `profiles`, `rules` and `discover`. The `talk` keys
are in [Voice](talk.md), the `assistant` keys in [Assistant](assistant.md)
and the theme keys in [Chat](chat.md).

## Models and their names

Each entry under `models` gives a model its id, which is the name used
everywhere: in the `model` field of a request, in `/v1/models`, and on the
command line of `gmlx run` and `gmlx chat`.

```yaml
models:
  qwen3.6-27b:
    path: Qwen3.6-27B-Q4_K_S.gguf
    speculative: true
  gemma-12b:
    path: gemma-4-12b-it-Q4_K_M.gguf
    mmproj: mmproj-gemma-4-12b-it-bf16.gguf
```

A relative `path` is looked up in each folder of `server.model_dirs` in
order. `init` names each model after its file with the quantization in
short form, such as `qwen3.6-27b-q4`, and you can rename an entry freely.
An id cannot contain `@`, because `@` separates an id from a profile name.

An alias gives a model a second name, and can fix a profile to it. The
server lists each alias in `/v1/models`, which lets a client that picks
from a menu choose a model and profile together:

```yaml
aliases:
  fast: gemma-12b
  coder: qwen3.6-27b@coding
```

## How a request gets its settings

Every model starts from its [family defaults](family-defaults.md), the
sampling values that its publisher recommends, so a new file needs no
sampling settings. Some publishers also recommend other values for a task,
such as a lower temperature for code. gmlx offers these as built-in
profiles called intents, which a request selects by adding `@name` to the
model id:

```sh
gmlx run qwen3.6-27b@coding "Write a binary search in Go."
```

A profile of your own is a named set of settings under `profiles`. It can
hold four kinds of setting:

- [`sampling`](server-config.md#sampling): defaults for request fields
  such as [`temperature`](server-config.md#samplingtemperature) and
  [`max_tokens`](server-config.md#samplingmax_tokens)
- [`load`](server-config.md#load): how the model is built, such as
  [KV cache quantization](server-config.md#loadkv_bits)
- [`cache`](server-config.md#cache): the prompt cache
- prompt settings: a [`system`](server-config.md#profilessystem)
  prompt, a [`chat_template`](server-config.md#profileschat_template)
  and the reasoning controls
  [`thinking`](server-config.md#profilesthinking) and
  [`reasoning_effort`](server-config.md#profilesreasoning_effort)

A profile can start from an intent with
[`extends`](server-config.md#profilesextends) and change only what it
sets:

```yaml
profiles:
  review:
    extends: coding
    system: "You are a terse code reviewer."
    sampling:
      temperature: 0.3
      max_tokens: 2048
```

A model gets a profile from its own entry, from a matching rule, from the
server default, or from the request. When a setting comes from more than
one place, the layer lower in this table wins:

| Layer | Set where |
|-------|-----------|
| family defaults | built in, per detected family |
| server default profile | [`server.defaults.profile`](server-config.md#serverdefaultsprofile) |
| rule profile | the first matching [`rules`](server-config.md#rules) entry |
| model profile | [`models.*.profile`](server-config.md#modelsprofile), or `@name` on the request |
| per-model profile change | [`models.*.profiles`](server-config.md#modelsprofiles) |
| model overrides | [`models.*.overrides`](server-config.md#modelsoverrides) |
| request fields | the request body |

A request's `@name` replaces the model's own profile rather than adding to
it, and an unknown name gets a 400. A profile's `system` prompt applies
only to a request without a system message. Its `chat_template` applies
when the model loads, so a request cannot change it.

`gmlx profiles <id>` prints the sampling values a model resolves to under
each of its profiles, with every layer applied.

## Which models stay loaded

The server keeps several models in memory at once, up to the budget in
`server.budget_gb`. The default budget is 0.8 times the working set that
macOS recommends for the GPU. A model uses as much memory as its GGUF file
is large, because the weights map from the file without a copy.

| State | Set by | Unloads when |
|-------|--------|--------------|
| pinned | `pin: true`, `--pin` | never |
| kept | `POST /v1/keep`, `gmlx launch --model`, a talk session | the budget is full and it is the least recently used |
| idle | any request | `ttl_s` seconds pass with no request, or the budget needs the room |
| preloaded | `server.defaults.preload` | as idle |

A model is never unloaded during a generation. Keeping is what
`gmlx launch --model` asks for, so the model of a coding session survives
the pauses between turns without holding budget permanently. A request
with `{"keep": false}` to the same endpoint releases a model, and
`POST /unload` unloads it at once.

Two ids that point to the same GGUF share one loaded copy, unless a
setting that changes how the model is loaded differs between them. Those
settings are the `load` keys, `mmproj`, `draft_gguf`, `speculative`,
`adapter`, `chat_template`, `stream` and the streaming keys. Sampling,
`system` and `ttl_s` never cause a second copy. A streamed model is
counted against the budget as described in
[Models larger than memory](streaming.md#residency-of-a-streamed-model).

## Changing the file

gmlx checks the whole file when it reads it. An unknown key fails the load
with the name of the key, so a typo such as `pinned:` for `pin:` is caught
before the server starts. The one exception is the contents of `sampling`,
`load` and `cache`, where an unknown key causes only a warning.

A running server reads its file again on `POST /v1/reload` or on `SIGHUP`.
Models that are already loaded stay loaded when their load settings did
not change. A change to a load setting, such as `mmproj` or `speculative`,
applies the next time that model loads.

Several commands change the file for you, and they keep your comments and
formatting:

| Command | Change |
|---------|--------|
| `gmlx sync-models` | adds new GGUFs from `model_dirs` and removes entries whose file is gone |
| `gmlx pull` | downloads a GGUF into the first folder of `model_dirs` and adds its entry |
| `gmlx rm` | deletes a model's files and its entry |

Each of them, and `gmlx init`, tells a running server to reload. Pass
`--no-reload` to prevent that. A server started with a GGUF path instead
of a config file has no file to read again, so it ignores the signal and
answers `/v1/reload` as unsupported.

## Flags and environment variables

Some settings can also be set by a `gmlx serve` flag or by an environment
variable. When a setting has more than one source, the first of these
wins:

1. the `serve` flag
2. the key in the file
3. the environment variable

The MLX buffer cache limit is the exception. Its environment variable wins
over `server.cache_limit_gb`, so that a benchmark can fix the limit without
changing the file. The flags are in the [CLI reference](cli.md#gmlx-serve)
and the variables in [Environment variables](env-vars.md).
