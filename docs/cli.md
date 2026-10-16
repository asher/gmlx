# CLI reference

The `gmlx` command has one verb per task, each with its own flags. Run
`gmlx <verb> --help` or `gmlx help <verb>` for a verb's options, and
`--help-all` on `run` and `chat` for the full set. The guides linked from
each verb explain when to use a flag.

```sh
gmlx --version        # also -V or gmlx version
gmlx help serve       # the same as gmlx serve --help
```

| Verb | Does |
|------|------|
| [`gmlx init`](#gmlx-init) | Write a starter server config from the GGUFs on disk. |
| [`gmlx serve`](#gmlx-serve) | Run the OpenAI and Anthropic compatible server. |
| [`gmlx stop`](#gmlx-stop) | Stop a background server. |
| [`gmlx status`](#gmlx-status) | Show a background server's pid, uptime and URL. |
| [`gmlx restart`](#gmlx-restart) | Stop and relaunch a background server. |
| [`gmlx logs`](#gmlx-logs) | Print or follow a background server's log. |
| [`gmlx service`](#gmlx-service) | Install the server and menu bar as a login item. |
| [`gmlx list`](#gmlx-list) | List the models a config defines. |
| [`gmlx run`](#gmlx-run) | Generate from, benchmark or inspect one GGUF. |
| [`gmlx chat`](#gmlx-chat) | Chat with a model in the terminal. |
| [`gmlx launch`](#gmlx-launch) | Configure a coding agent or chat app to use the server. |
| [`gmlx pull`](#gmlx-pull) | Check a remote GGUF and download it. |
| [`gmlx validate`](#gmlx-validate) | Check that a local or remote GGUF will load. |
| [`gmlx rm`](#gmlx-rm) | Delete a model's files and config entry. |
| [`gmlx sync-models`](#gmlx-sync-models) | Reconcile a config with the files on disk. |
| [`gmlx ps`](#gmlx-ps) | Show the models resident in a running server. |
| [`gmlx systemone`](#gmlx-systemone) | Answer a structured-decision request. |
| [`gmlx profiles`](#gmlx-profiles) | Show the family sampling defaults and intents. |
| [`gmlx talk`](#gmlx-talk) | Voice chat with a served model. |
| [`gmlx train`](#gmlx-train) | Train a LoRA adapter on a GGUF base. |
| [`gmlx distill`](#gmlx-distill) | Distill a teacher GGUF into a student adapter offline. |
| [`gmlx image`](#gmlx-image) | Generate images from GGUF diffusion models. |
| [`gmlx comfyui`](#gmlx-comfyui) | Put the gmlx node pack into ComfyUI, or remove it. |
| [`gmlx doctor`](#gmlx-doctor) | Check the runtime, config, models and services. |
| [`gmlx completion`](#gmlx-completion) | Print a shell completion script. |

`gmlx ls` is an alias for `gmlx list`. Sampling flags you leave unset take
the model's [family defaults](family-defaults.md). "The first default
location" means the first file in the order that
[Where gmlx looks](config.md#where-gmlx-looks) gives. How flags, config keys
and [environment variables](env-vars.md) combine is in
[Flags and environment variables](config.md#flags-and-environment-variables).

## gmlx init

`gmlx init` scans your model directories and writes a starter config. Run
bare on a terminal, it opens a wizard that lets you rename models, set a
default and aliases, and enable the prompt cache and the speech, embedding
and rerank services. With flags it writes the file without asking.

```sh
gmlx init                                  # the wizard
gmlx init --models-dir ~/models            # flag-driven, writes ~/.config/gmlx/gmlx.yaml
gmlx init --models-dir ~/models -r --out ~/configs/studio.yaml
gmlx init --from-hf-cache                  # models already in the Hugging Face cache
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--models-dir DIR` | Required unless `--from-hf-cache` | Scan this directory. Repeat the flag for more directories. |
| `--from-hf-cache`, `--hf-cache` | Off | Also scan the local Hugging Face cache and write portable `hf:` entries. |
| `-r`, `--recursive`, `--no-recursive` | Shallow | Descend into subdirectories. |
| `--out FILE` | `~/.config/gmlx/gmlx.yaml` | Write the config to this file. |
| `--force` | Off | Overwrite an existing file. |
| `-i`, `--interactive` | On a terminal with no other flags | Run the wizard even with flags, which pre-fill its answers. |
| `--no-interactive` | Off | Never run the wizard. |
| `--disk-cache [GB]` | On in the wizard, else off | Enable the on-disk prompt cache with this cap for each model, 50 GB when the flag is bare. |
| `--with-stt [MODEL]` | Off | Configure speech-to-text, with `whisper-turbo` when the flag is bare. |
| `--with-tts [MODEL]` | Off | Configure text-to-speech, with `kokoro` when the flag is bare. |
| `--with-embeddings [MODEL]` | Off | Configure embeddings, with `qwen3-embed-0.6b` when the flag is bare. |
| `--with-rerank [MODEL]` | Off | Configure reranking, with `qwen3-rerank-0.6b` when the flag is bare. |
| `--install`, `--no-install` | Ask in the wizard, else off | Install the extras the chosen services need, or never offer to. |
| `--default-model ID` | None | A request that omits a model uses this one. |
| `--port N` | `8080` | Write this port into the config. |
| `--idle-ttl SECONDS` | `900` | A model unloads after this many idle seconds, and `none` keeps models resident. |
| `--request-timeout DURATION` | Unset, and the server applies `30m` | Fail the request when no token arrives for this long, such as `10m` or `1h`. `none` waits forever. |
| `--no-reload` | Off | Do not signal a running server to re-read the file. |

Model ids carry the quant in compact form, such as `qwen3-0.6b-q4`. The
walkthrough is in the [Quickstart](quickstart.md#serving-models), and
[Configuration](config.md) describes the file.

## gmlx serve

`gmlx serve` runs the server. It detaches by default and returns once the
server is ready, so the same shell can run `gmlx launch` next.
`--foreground` keeps it attached. A background server on a macOS desktop
also raises the [menu bar app](menubar.md).

```sh
gmlx serve                                  # the config in the default location
gmlx serve --config ~/configs/studio.yaml
gmlx serve --models-dir ~/models --recursive
gmlx serve model-Q4_K_M.gguf                # one model, id from the filename
gmlx serve model.gguf --mmproj mmproj.gguf  # one vision model
```

A bare `gmlx serve` needs a config in a
[default location](config.md#where-gmlx-looks), or it tells you to run
`gmlx init` or to name a GGUF.

These flags say where the models come from:

| Flag | Default | Meaning |
|------|---------|---------|
| `model`, positional | None | Serve this GGUF, pinned, with an id derived from the filename. |
| `--config FILE` | The first default location | Serve a YAML config. |
| `--models-dir DIR` | None | Serve a scan of this directory. Repeat the flag for more directories. |
| `-r`, `--recursive`, `--no-recursive` | Shallow | Descend when scanning. |
| `--hf-cache`, `--from-hf-cache` | Off | Let Hugging Face ids resolve from the local cache, never the network. |
| `--print-config` | Off | Print the resolved config as YAML and exit. |

These flags control the process:

| Flag | Default | Meaning |
|------|---------|---------|
| `--host ADDR` | Config or `127.0.0.1` | Bind to this address. A non-loopback bind needs `server.api_key` or `--no-auth`. |
| `--port N` | Config or `8080` | Bind to this port. |
| `--no-auth` | Off | Allow a non-loopback bind with no key. |
| `-f`, `--foreground` | Off | Stay attached to the terminal. |
| `--no-menubar` | Off | Do not raise the menu bar app. |
| `--log FILE` | `~/.cache/gmlx/server-<host>-<port>.log` | Write the background log here. Each start rotates the last one to `.1`. |
| `--log-level LEVEL` | `info` | Set the log level to `critical`, `error`, `warning`, `info`, `debug` or `trace`. |
| `--start-timeout S` | `40` | A background start waits this many seconds for readiness before it returns. |

`serve` has no `--api-key` flag. The key lives in the config, as
[Address and authentication](config.md#address-and-authentication)
describes.

These flags set memory and scheduling. Most are also `server` keys in the
config:

| Flag | Default | Meaning |
|------|---------|---------|
| `--budget-gb F` | 0.8x the GPU working set | Keep the weights of all [resident](glossary.md#resident) models within this many GB. |
| `--max-models N` | None | Keep at most this many models resident. |
| `--pin ID_OR_PATH` | None | Load this model at start and never evict it. Repeat the flag for more models. |
| `--max-tokens N` | Until the model stops | Cap a completion at this many tokens when the request sets no cap. |
| `--no-family-defaults` | Off | Do not seed each family's model-card sampling under profiles and requests. |
| `--prefill-step-size N` | `2048` | Prefill in chunks of this many tokens. A lower value caps peak memory. |
| `--dtype {auto,bfloat16,float16}` | `auto` | Set the activation width. `auto` picks float16 on M1 and M2. |
| `--decode-prefill-ratio R` | `auto` | Make each prefill chunk wait until decoding streams have had this multiple of its GPU time. `0` restores stock scheduling. |
| `--prefill-tick-ms MS` | `500` | Give each prefill chunk this wall-clock budget while streams decode. `0` never halves a chunk. |
| `--speculative-width-cap N` | Each drafter's default | Speculate on at most N requests at a time, for every model. Wins over each model's `speculative_width_cap`. `0` removes the cap. |
| `--ignore-eos` | Off | Decode each request to its output cap. |

These flags apply to a positional GGUF only. In config mode the same
settings are per-model keys under [models](config.md#models), the KV flags
are the [`load` keys](config.md#model-loading), and the sampling flags are
the [`sampling` keys](config.md#sampling):

| Flag | Default | Meaning |
|------|---------|---------|
| `--mmproj PATH` | None | Load this projector GGUF to make the model multimodal. |
| `--adapter PATH` | None | Apply this GGUF LoRA adapter at load, for text only. |
| `--profile NAME` | None | Apply a built-in intent such as `coding` or `reasoning-high`, resolved for the model's family. |
| `--system-prompt STR` | None | Use this system prompt when the request has none. |
| `--thinking-budget N` | Unlimited | Cap reasoning tokens for each request. `0` closes thinking at once. |
| `--kv-quant-scheme {auto,uniform,kvarn}` | `auto` | Pick the [scheme](kv-quantization.md#the-scheme-gmlx-picks) for the model, affine, or kvarn, where `--kv-bits` defaults to 6. |
| `--max-kv-size N` | None | Cap the request context budget at N tokens. |

These flags also work as under [run](#gmlx-run): `--hf-source`,
`--chat-template`, `--thinking`, `--reasoning-effort`,
`--chat-template-config`, `--kv-bits`, `--kv-group-size`,
`--kv-tail-tokens` and `--quantized-kv-start`.

The sampling flags work as under [run](#gmlx-run) too, as defaults for a
request that omits the field: `--temp`, `--top-p`, `--top-k`, `--min-p`,
`--seed`, `--repetition-penalty`, `--repetition-context-size`,
`--presence-penalty`, `--frequency-penalty`, `--stop`, `--xtc-probability`,
`--xtc-threshold`, `--thinking-start-token` and `--thinking-end-token`. A
request that sends the field wins, so `--temp 0` does not pin a client that
sends its own temperature.

`--speculative` turns on speculative decoding for a positional model, with
the model's own MTP head or `--draft-gguf`. A config `discover` scan turns
it on by itself. `--draft-gguf`, `--native-mtp`, `--draft-block-size` and
`--stochastic-mtp` work as under [run](#speculative-decoding-flags), and the
first two imply `--speculative`. Under serve, `--draft-block-size` and
`--stochastic-mtp` apply to every model the server loads.

`--stream-experts`, `--stream-cpu`, `--stream-fast-disk`,
`--prefill-feeder`, `--no-prefill-feeder`, `--decode-feeder`,
`--no-decode-feeder`, `--gpu-keepwarm`, `--no-gpu-keepwarm`,
`--moe-experts`, `--moe-expert-mass`, `--moe-miss-shed`, `--moe-layer-shed`
and `--moe-prestage` stream a positional model bigger than memory. They
work as under [run](#streaming-flags), and `--gpu-keepwarm` applies to the
whole server.

`--rope-scaling`, `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`,
`--yarn-orig-ctx`, `--yarn-ext-factor`, `--yarn-attn-factor`,
`--yarn-beta-slow` and `--yarn-beta-fast` work as under
[run](#rope-scaling-flags) for a positional model. In config mode they are
the model's [`rope_scaling`](config.md#rope-scaling) block, and serve
refuses them on the command line.

These flags enable the services in
[Speech, embeddings and rerank](services.md). Each is also a `server` key:

| Flag | Default | Meaning |
|------|---------|---------|
| `--stt [MODEL]` | Off | Serve speech-to-text at `POST /v1/audio/transcriptions`, with `whisper-turbo` when the flag is bare. |
| `--tts [MODEL]` | Off | Serve text-to-speech at `POST /v1/audio/speech`, with `kokoro` when the flag is bare. |
| `--embeddings [MODEL]` | Off | Serve embeddings at `POST /v1/embeddings`, with `qwen3-embed-0.6b` when the flag is bare. |
| `--rerank [MODEL]` | Off | Serve reranking at `POST /v1/rerank`, with `qwen3-rerank-0.6b` when the flag is bare. |

Each completed request logs a line with the endpoint, model, token counts,
timing and the MLX memory in use. A finish reason other than `stop` adds
`finish=<reason>`:

```text
[req] 2026-06-15 16:07:42 /chat/completions qwen3-0.6b prompt=19 gen=3 ttft=0.47s prefill=45t/s decode=172.6t/s total=0.51s active=1.3G cache=0.2G
```

## gmlx stop

`gmlx stop` asks a background server to stop. A server that is still
running after `--timeout` seconds is killed, which cuts off any generation
in progress.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Select the server by host when several run in the background. |
| `--port P` | The managed server | Select the server by port. |
| `--timeout S` | `15` | Force the server to stop after this many seconds. |
| `--stale` | Off | Clear the records of servers that have exited, and stop nothing. |

## gmlx status

`gmlx status` prints a background server's pid, uptime, URL, log path and
how it is managed. It needs no API key. After the servers, it lists your
[container sessions](container-sessions.md#see-and-stop-sessions).

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Select the server by host. |
| `--port P` | The managed server | Select the server by port. |
| `--json` | Off | Emit JSON, without the container sessions. |

## gmlx restart

`gmlx restart` stops the server and starts it again with the same
arguments, from any directory. It checks the config and the model files
first, and if one is missing it leaves the server running.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Select the server by host. |
| `--port P` | The managed server | Select the server by port. |
| `--timeout S` | `15` | Force the old server to stop after this many seconds. |
| `--start-timeout S` | `40` | Wait this many seconds for the new server to become ready. |

## gmlx logs

`gmlx logs` prints the tail of a background server's log, without the menu
bar's health polls.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Select the server by host. |
| `--port P` | The managed server | Select the server by port. |
| `-n`, `--lines N` | `40` | Print this many lines. |
| `-f`, `--follow` | Off | Keep printing as the log grows. |
| `--clear` | Off | Truncate the log and exit. |

## gmlx service

`gmlx service` installs a launchd login item on macOS. By default the item
is the menu bar app, which starts the server at login. A server run this
way gets its own identity, so macOS asks for permissions on behalf of gmlx
instead of your terminal. `install` takes the `serve` flags.

```sh
gmlx service install --config ~/.config/gmlx/gmlx.yaml
gmlx service status
gmlx service uninstall
```

| Subcommand | Flags | Meaning |
|------------|-------|---------|
| `install` | The `serve` flags, `--no-autostart`, `--headless` and `--keepalive` | Register the login item and start now. |
| `status` | `--host H`, `--port P` | Print the launchd state. |
| `uninstall` | `--host H`, `--port P` | Remove the headless agent of the port and the menu bar's login item. |

| Flag | Default | Meaning |
|------|---------|---------|
| `--no-autostart` | Off | Install the menu bar item without starting the server at login. |
| `--headless` | Off | Install a server-only agent with no menu bar, for machines without a desktop session. |
| `--keepalive`, `--no-keepalive` | On | With `--headless`, restart the server when it crashes. |

To stop a headless server, run `gmlx service uninstall`, not `gmlx stop`.
For the menu bar side, read [Menu bar app](menubar.md).

## gmlx list

`gmlx list` lists the model ids a request can address. Discovered models
are tagged, aliases follow, and the default model is marked with `*`. The
KIND column reads `chat` or `image`. An image model's size counts its DiT,
VAE and vision tower, and its text encoder unless it shares a chat model's,
which that model's row counts.

```sh
gmlx list
gmlx list --image --edit
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | The first default location | Read this config. |
| `-v`, `--paths` | Off | Also show each model's GGUF path. |
| `--json` | Off | Emit JSON. |
| `--image` | Off | List only the [image models](images.md). |
| `--family NAME` | All | List only the image models of this family, such as `z-image`, `qwen-image-2.1` or `flux-3-klein`. |
| `--edit` | Off | List only the image models that edit from reference images: a family that takes them, with the encoder's vision tower where it needs one. |
| `--shared-encoder` | Off | List only the image models whose text encoder is a chat model they share. |

## gmlx run

`gmlx run` loads a GGUF and generates a completion, runs a benchmark, or
prints the load plan.

```sh
gmlx run model.gguf --prompt "Explain entropy." --max-tokens 128
gmlx run model.gguf --bench 512,4096,16384 --bench-runs 3
gmlx run model.gguf --bench-depths 0,4096,16384,32768
gmlx run model.gguf --report-only
gmlx run coder --prompt "Refactor this loop."    # a config id, with its settings
```

The positional argument is a path, or a model id or alias from your server
config. A config id brings its path, sampling, system prompt, template,
adapter, drafter and streaming settings, and flags you pass still win.

These flags control generation:

| Flag | Default | Meaning |
|------|---------|---------|
| `gguf`, positional | Required | Load this GGUF, which may be sharded, or this config id. |
| `--prompt STR` | `Hello, world!` | Generate from this prompt. |
| `--prompt-file PATH` | None | Read the prompt from a file. |
| `--system-prompt STR` | None | Pass this system message to the chat template. |
| `--max-tokens N` | Until the model stops | Stop after this many tokens. |
| `--temp F` | Family default | Set the sampling temperature. `0` is greedy. |
| `--top-p F`, `--top-k N`, `--min-p F` | Family default | Set the sampling filters. `0` disables `--top-k`. |
| `--repetition-penalty F` | `0` | Set the repetition penalty. `0` disables it. |
| `--repetition-context-size N` | `20` | The repetition penalty looks back over this many tokens. |
| `--presence-penalty F`, `--frequency-penalty F` | `0` | Set the presence and frequency penalties. |
| `--xtc-probability F`, `--xtc-threshold F` | `0` | Set XTC sampling, which works on the text path only. |
| `--logit-bias JSON` | None | Add these biases to the logits, given as a map from token id to bias. |
| `--stop STR` | None | Stop at this sequence. Repeat the flag for more sequences. |
| `--seed N` | None | Seed the sampler. |
| `--reasoning {show,hide,raw}` | `show` | `show` styles the thinking, `hide` prints only the answer, and `raw` passes everything through. |
| `--thinking {on,off,adaptive}` | Template default | Turn reasoning on, off or adaptive through the model's template variable. |
| `--reasoning-effort LEVEL` | Template default | Set the reasoning depth on models that support levels. |
| `--thinking-budget N` | Unlimited | Cap reasoning tokens. |
| `--thinking-start-token STR`, `--thinking-end-token STR` | Detected | Set the model's reasoning markers when detection fails. |
| `--chat-template-config JSON` | None | Pass extra template variables, such as `'{"enable_thinking": false}'`. |
| `--profile NAME` | None | Apply a built-in intent or, with a config, a user profile, as `@NAME` on the positional argument does. |
| `--no-family-defaults` | Off | Do not apply the family's sampling defaults on a bare path. |
| `--config FILE` | The first default location | Resolve an id against this config. |
| `-v`, `--verbose` | Off | Print full load diagnostics instead of the spinner. |

These flags control memory. [KV cache quantization](kv-quantization.md)
covers kvarn and the models it declines, and
[Settings that limit memory](memory.md#settings-that-limit-memory) covers
the window sizes:

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-kv-size N` | None | Cap the KV cache with a rotating window. Works with kvarn but not with affine `--kv-bits`, so `auto` picks kvarn. |
| `--kv-bits N` | Off | Quantize the KV cache to 2, 3, 4, 6 or 8 bits affine, or to 2, 3, 4, 5, 6 or 8 under kvarn, where it defaults to 6. |
| `--kv-group-size N` | `64` | Set the affine quantization group size. |
| `--kv-quant-scheme {auto,uniform,kvarn}` | `auto` | Pick the [scheme](kv-quantization.md#the-scheme-gmlx-picks) for the model, affine, or kvarn. |
| `--kv-tail-tokens N` | `1024` | Under kvarn, the newest N tokens stay fp16. N is a multiple of 128, and `0` disables the tail. |
| `--quantized-kv-start N` | `0` | Keep the cache in fp16 until it holds this many tokens, then quantize all of it. Not under kvarn. |
| `--prefill-step-size N` | `2048`, `8192` when streaming, `4096` for HY4 when streaming | Prefill in chunks of this many tokens. |
| `--dtype {auto,bfloat16,float16}` | `auto` | Set the activation width. `auto` picks float16 on M1 and M2. |

These flags control loading:

| Flag | Default | Meaning |
|------|---------|---------|
| `--arch NAME` | Detected | Override architecture detection. |
| `--hf-source ID_OR_DIR` | None | Take the config, processor and template from this repo or directory. |
| `--chat-template STR_OR_PATH` | The GGUF's | Replace the chat template with inline Jinja or a `.jinja` or `.txt` file. |
| `--no-chat-template` | Off | Pass the prompt verbatim. |
| `--no-remap` | Off | Keep raw GGUF tensor names. |
| `--no-zero-copy` | Off | Copy tensors out of the mmap instead of viewing them. |
| `--adapter PATH` | None | Apply this GGUF LoRA adapter at load, for text only. |

These flags control multimodal input, as [Vision and audio](vlm.md)
describes. Under `--mmproj`, the bench and report flags and `--stream-cpu`
are refused:

| Flag | Default | Meaning |
|------|---------|---------|
| `--mmproj PATH` | None | Load this projector GGUF. |
| `--image PATH_OR_URL` | None | Prepend these comma-separated images. |
| `--audio PATH_OR_URL` | None | Prepend these comma-separated audio files. The model needs an audio tower. |
| `--resize-shape N_OR_WxH` | Model default | Resize images before encoding. |

These flags inspect and benchmark:

| Flag | Default | Meaning |
|------|---------|---------|
| `--report-only` | Off | Print the load plan and the rendered prompt without building the model. |
| `--bench LIST` | None | Time prefill and decode in tok/s at these comma-separated prompt lengths. |
| `--bench-depths LIST` | None | Time decode at these context depths. |
| `--bench-runs N` | `2` | Time each length this many times and report the best. |
| `--bench-decode-tokens N` | `32`, `128` for depths | Decode this many tokens in each run. |
| `--bench-temp T` | `0` | Sample speculative bench runs at this temperature. |
| `--bench-chat-dataset DATASET` | Synthetic | Take bench prompts from this Hugging Face chat dataset, given as `id` or `id:split`. |

### Speculative decoding flags

[Speculative decoding](speculative-decoding.md) describes the drafters.
Speculation drops, with a warning, a flag it cannot honor, as
[Settings that speculation drops](speculative-decoding.md#settings-that-speculation-drops)
lists.

| Flag | Default | Meaning |
|------|---------|---------|
| `--speculative`, `--mtp` | Auto for an MTP head or a DeepSeek-V4 companion, off under `--stream-experts` | Force speculation on. |
| `--no-speculative`, `--no-mtp` | Off | Force plain decoding, which honors every flag. |
| `--draft-gguf PATH` | None, or the companion beside a DeepSeek-V4 file | Draft with this separate [drafter](glossary.md#drafter) GGUF, which implies `--speculative`. |
| `--native-mtp` | Off | Draft with the model's own head, even when a drafter is set. Forces speculation on. |
| `--draft-block-size N` | Drafter default | Set the block size of each round, which drafts N-1 tokens and checks them in one N-token target pass. |
| `--stochastic-mtp` | Off | Accept sampled drafts by rejection sampling, which accepts more but is not token-identical. |

### Streaming flags

These flags run a model bigger than memory, as
[Models larger than memory](streaming.md) describes.

| Flag | Default | Meaning |
|------|---------|---------|
| `--stream-experts` | Off | Stream the routed experts and lookup tables from disk, even on a model that fits. Over the memory budget, a table streams first. Attention and KV stay on GPU. |
| `--stream-cpu` | Off | Run the whole model on the CPU device from the page cache. |
| `--stream-fast-disk {auto,on,off}` | `auto` | Set the prefetch policy under `--stream-experts`. `auto` measures the drive at load. |
| `--prefill-feeder`, `--no-prefill-feeder` | On | Stage expert prefill directly from the GGUF. |
| `--decode-feeder`, `--no-decode-feeder` | On under `--stream-experts` | Decode from a wired expert [arena](glossary.md#arena) that keeps the experts the router picks most often. |
| `--gpu-keepwarm`, `--no-gpu-keepwarm` | On with the decode feeder | Keep GPU clocks high while a streamed model decodes, or turn that off. |
| `--moe-experts K` | Trained | Cap the router at K experts for each token, which is lossy. |
| `--moe-expert-mass P` | Off | Keep the smallest expert set covering share P of gate mass, which is lossy. |
| `--moe-expert-probe` | Off | Run lossless and print how many experts each token needed at candidate P values. `run` and `chat` only. |
| `--moe-miss-shed P` | Off | Drop experts that would miss the arena down to share P, which is lossy. |
| `--moe-layer-shed P` | Off | Skip a streamed layer's experts with probability P, which is lossy. |
| `--moe-prestage {ranked,keepers}` | `ranked` | `keepers` filters prestage predictions through the miss-shed policy, so it needs `--moe-miss-shed`. |

### Rope scaling flags

These flags run a model past its trained context. They take llama.cpp's
names and meanings, so a llama.cpp command line ports unchanged, and each
one wins over its key in the GGUF header:

| Flag | Default | Meaning |
|------|---------|---------|
| `--rope-scaling {none,linear,yarn}` | The header's, else `linear` | Pick the rope frequency scaling method. `none` turns the header's scaling off. |
| `--rope-scale N` | The header's, else 1 | Expand the context by a factor of N. |
| `--rope-freq-scale N` | The header's, else 1 | Expand the context by a factor of 1/N, the same value as `--rope-scale` spelled the other way. |
| `--rope-freq-base N` | The header's | Set the rope base frequency. |
| `--yarn-orig-ctx N` | The header's, else the trained context | Set the context the model was trained at. |
| `--yarn-ext-factor N` | 1 under yarn, else 0 | Set the extrapolation mix. `0` is full interpolation. |
| `--yarn-attn-factor N` | 1 | Scale the attention magnitude when `--yarn-ext-factor` is 0. |
| `--yarn-beta-fast N` | A mistral3 header's, else 32 | Set the low correction dimension. |
| `--yarn-beta-slow N` | A mistral3 header's, else 1 | Set the high correction dimension. |

```sh
gmlx run model.gguf --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144
```

A YaRN factor from these flags extends the reported context to the original
context times the factor. A linear factor does not: requests still stop at
the trained context. A constant factor can cost quality on short prompts,
so pick the smallest factor that fits your context. A model whose rope gmlx
cannot scale yet refuses the flags at load, and so does a load with
`--hf-source`, whose config does not take them.

## gmlx chat

`gmlx chat` is an interactive chat in the terminal. Locally the model loads
once and each turn prefills only the new message. When the config's server
is running, a bare `gmlx chat` or one naming a served id becomes a client of
that server instead of loading a second copy. The commands, sessions,
rendering and themes are in [Chat](chat.md).

```sh
gmlx chat model.gguf --temp 0.7 --system-prompt "You are terse."
gmlx chat                          # the running server's default model
gmlx chat --assistant              # the tool-loop assistant on the server
```

These flags say where the model runs:

| Flag | Default | Meaning |
|------|---------|---------|
| `gguf`, positional | Server default | Chat with this GGUF, config id or served id. |
| `--server` | Auto when the server is running | Run as a plain client of the server. |
| `--assistant` | Off | Chat through the server's tool-loop assistant, with MCP tools and memory. |
| `--local` | Off | Load in-process even when the server is running. |
| `--base-url URL` | The managed server | Connect to this server, implying `--server`. |
| `--host H`, `--port P`, `--api-key KEY` | The managed server | Select the server and give its key, implying `--server`. |
| `--no-start` | Off | Never start the server, implying `--server`. |
| `--start-timeout S` | `180` | Wait this many seconds for an auto-start. |
| `--config FILE` | The first default location | Resolve an id against this config. |
| `--profile NAME` | None | Apply a built-in intent or user profile. |
| `--no-family-defaults` | Off | Do not apply the family's sampling defaults on a bare path. |

Slash commands in the chat change the system prompt, sampling, penalties,
`--reasoning`, `--thinking` and `--thinking-budget` as you go. The other
flags last for the whole chat.

These flags control display and sessions:

| Flag | Default | Meaning |
|------|---------|---------|
| `--render {auto,plain,lite,rich}` | `auto` | Pick how replies render their Markdown. |
| `--theme NAME` | The config's `theme:`, else `dark` | Set the color theme. |
| `--colorblind` | Off | Use colorblind-friendly accents on any theme. |
| `--no-history` | Off | Do not read or write the prompt history file. |
| `--no-autosave` | Off | Do not save the session after each turn. |
| `--resume [NAME]` | Off | Resume a saved session, this model's latest when the flag is bare. |
| `-v`, `--verbose` | Off | Print full load diagnostics. |

These [`gmlx run`](#gmlx-run) flags mean the same as they do there:

| Group | Flags shared with `run` |
|-------|-------------------------|
| Generation | `--system-prompt`, `--max-tokens`, `--temp`, `--top-p`, `--top-k`, `--min-p`, `--seed`, `--stop`, `--logit-bias`, `--xtc-probability`, `--xtc-threshold` |
| Penalties | `--repetition-penalty`, `--repetition-context-size`, `--presence-penalty`, `--frequency-penalty` |
| Reasoning | `--reasoning`, `--thinking`, `--reasoning-effort`, `--thinking-budget`, `--thinking-start-token`, `--thinking-end-token` |
| Loading | `--arch`, `--hf-source`, `--chat-template`, `--no-chat-template`, `--chat-template-config`, `--no-remap`, `--no-zero-copy`, `--adapter` |
| Memory | `--max-kv-size`, `--kv-bits`, `--kv-group-size`, `--kv-quant-scheme`, `--kv-tail-tokens`, `--quantized-kv-start`, `--prefill-step-size`, `--dtype` |
| Multimodal | `--mmproj`, `--resize-shape` |
| Speculation | `--speculative`, `--mtp`, `--no-speculative`, `--no-mtp`, `--draft-gguf`, `--native-mtp`, `--draft-block-size`, `--stochastic-mtp` |
| Streaming | `--stream-experts`, `--stream-cpu`, `--stream-fast-disk` |
| Streaming feeders | `--prefill-feeder`, `--no-prefill-feeder`, `--decode-feeder`, `--no-decode-feeder`, `--gpu-keepwarm`, `--no-gpu-keepwarm` |
| Lossy streaming | `--moe-experts`, `--moe-expert-mass`, `--moe-expert-probe`, `--moe-miss-shed`, `--moe-layer-shed`, `--moe-prestage` |
| Rope scaling | `--rope-scaling`, `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale` |
| YaRN | `--yarn-orig-ctx`, `--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-slow`, `--yarn-beta-fast` |

A server client refuses `--adapter`, `--mmproj`, `--chat-template`,
`--no-chat-template`, `--chat-template-config` and the rope scaling flags,
and ignores the other
loading, memory, speculation and streaming flags. A base model with no
chat template needs `--chat-template` or `--no-chat-template`.

## gmlx launch

`gmlx launch` points a coding agent or chat app at a gmlx server, starts
the server if none is reachable, and runs the tool. It never installs the
tool on the Mac, and [container mode](launch-container.md) installs it in
the container's image.

[Agents and chat apps](launch.md) describes each client, and
`gmlx launch CLIENT --help` ends with its install command. A
[custom agent](launch-agents.md) launches by its name and always runs in a
container. `gmlx launch --help` lists your agents.

```sh
gmlx launch opencode
gmlx launch pi --model qwen3.8-27b-ud-q6@coding
gmlx launch claude-code --container --model qwen3.8-27b-ud-q6
gmlx launch open-webui
gmlx launch omp --config-only
gmlx launch claude-code -- --continue
```

| Flag | Default | Meaning |
|------|---------|---------|
| `client`, positional | None | Launch a [client](launch.md#the-clients), such as `opencode`, `open-webui` or `comfyui`, `menubar`, or a [custom agent](launch-agents.md). |
| `--model ID[@profile]` | An agent's `model`, else the server's default | Point the tool at this served model, which the server keeps loaded through its idle timeout. |
| `--base-url URL` | None | Connect to this server, which is never auto-started. |
| `--host H`, `--port P` | The managed server | Select the server. |
| `--api-key KEY` | The running server's `server.api_key` | Write this key to the tool's native config field. Without a key, tools that require one get the provider id. |
| `--provider-id NAME` | `gmlx` | Write this provider id into the tool's config. |
| `--config-path PATH` | The client's location | Write the tool config here, as [How a launch works](launch.md#how-a-launch-works) lists. hermes and container mode refuse it. |
| `--config-only` | Off | Write the config and print the run command without running it. In container mode, print the `container run` command. |
| `--no-start` | Off | Never start a server. |
| `--start-timeout S` | `0`, no limit | Cap the auto-start wait. |
| `--no-keep` | Off | Let `--model`, or an agent's `model` setting, unload while idle. |
| `--dsh-profile NAME` | `gmlx` | Boot this dsh profile with the gmlx overlay, as [dsh](launch.md#dsh) describes. |
| `--comfyui-dir DIR` | The folder used last, else `~/ComfyUI`, else the ComfyUI found | Put the gmlx nodes into the ComfyUI in this folder and run it on its `.venv` or `venv`, as [ComfyUI](comfyui.md) describes. |
| `--container`, `--no-container` | The config's `enabled` | Run the client in an Apple container, or on the Mac, as [Container mode](launch-container.md) describes. |
| `--mount PATH[:DST][:ro]` | None | Share another folder with the container, beside the configured [mounts](config.md#launchcontainermounts). Repeatable. |
| `--mount-cwd`, `--no-mount-cwd` | The config's [`mount_cwd`](config.md#launchcontainermount_cwd) | Share the current folder with the container, or not. |
| `--image REF` | The configured image | Run this image in the container, as [A ready-made image](container-images.md#a-ready-made-image) describes. |
| `--rebuild` | Off | Rebuild the client's image, or pull an `image:` reference again. |
| `--reseed` | Off | Copy each [seed](config.md#launchcontainerclientsseed) into the [private home](glossary.md#private-home) again, over its old copy. |
| `--seed-instructions` | Off | Also seed the client's instruction and skill files, as [Instructions and skills](container-access.md#instructions-and-skills) lists. |
| `--network {default,none}` | The config's [`network`](config.md#launchcontainernetwork) | Set the container's network for this launch. |
| `--shell` | Off | Open a shell instead of the client, in the project's running session if any. |
| `--remove-home` | Off | Ask, then remove the project's private home and the volumes only it uses, and start nothing. |
| `--detach` | Off | Start the session of Open WebUI, a dsh web profile or a custom agent in the background. |
| `--stop` | Off | End the project's session, and start nothing. |
| `--list` | Off | List the sessions of every client and agent, or of the one named, and start nothing. |
| `--forget-share PATH` | None | Remove `PATH` and each folder in it from the [share history](container-security.md#the-share-history), and start nothing. |
| `-- ARGS` | None | Pass the arguments after `--` to the client, after the arguments `launch` adds. |

The flags from `--mount` to `--stop` work only in a container, and each
turns on container mode by itself. [Container sessions](container-sessions.md)
covers `--detach`, `--stop`, `--list` and `--shell`. Once the client runs,
`launch` exits with the client's status. Its own codes are in
[Launch exit codes](#launch-exit-codes).

### launch menubar

`gmlx launch menubar` runs the macOS menu bar app directly, although a
background `serve` starts it automatically. What it shows is in
[Menu bar app](menubar.md).

| Flag | Default | Meaning |
|------|---------|---------|
| `-f`, `--foreground` | Off | Run the event loop in this process. |
| `--stop` | Off | Quit a detached menu bar app. |
| `--url URL` | The managed server | Track the server at this URL. |
| `--host H`, `--port P` | The managed server | Track the server at this host and port. |
| `--api-key KEY` | The managed server's | Send this key to a keyed server whose config the app cannot read. |
| `--interval S` | `4` | Poll the server at this interval in seconds. |

## gmlx pull

`gmlx pull` checks a remote GGUF's header and, when it will load, downloads
all its shards into your model directory. A file saved under a
`model_dirs` root is added to the config, and a running server reloads it:
one started from that config, or one in the foreground at the config's
address.

```sh
gmlx pull hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
gmlx pull hf:org/repo/model.gguf --to ~/models
gmlx pull hf:org/gemma-3-27b-GGUF/gemma-3-27b-Q4_K_M.gguf mmproj-F16.gguf
gmlx pull z-image-turbo
gmlx pull qwen-image --quant Q5_K_M
```

An image model name, `z-image-turbo`, `qwen-image`, `qwen-image-turbo`,
`krea-2-turbo`, `flux-3-klein` or `flux-3-klein-base`, pulls its DiT, text
encoder and VAE, checks the VAE
against its published sha256, packs it to GGUF, and adds an `image_models`
entry. The quants default to
the ones for this Mac's memory, and an encoder or VAE already on disk is
used again when its weights match.
[Models and quants](images.md#models-and-quants) has the quants for each Mac.

Of several encoders on disk, `pull` takes the quant for this Mac first,
then a chat model, then the file nearest the DiT. A chat model taken as
the encoder is shared by its id, so the server holds its weights once. For
Qwen-Image 2.1 that chat model needs its `mmproj`. Without one, the entry
names the encoder file and the vision tower that `pull` adds.

| Flag | Default | Meaning |
|------|---------|---------|
| `REF`, positional | Required | Download these `hf:<org>/<repo>/<file.gguf>[@rev]` references or URLs, or one image model name. Later bare filenames resolve in the first ref's repo. |
| `--to DIR`, `--out DIR` | The first `model_dirs` root | Download into this directory instead, with no repo subfolder. A file outside every `model_dirs` root is not added to the config. |
| `--config FILE` | The first default location | Read `model_dirs` from this config. |
| `--force` | Off | Download even when the header check, the disk-space check or the chip check of an image model fails. |
| `--no-register` | Off | Do not add the file to the config. |
| `--hf-source ID` | None | Treat the architecture as loadable with this config override. |
| `--max-mb N` | `128` | Cap the header range read. |
| `--json` | Off | Emit each verdict as JSON before downloading. |
| `--quant QUANT` | The one for this Mac's memory | Image models: download the DiT at this quant. |
| `--encoder-quant QUANT` | The one for this Mac's memory | Image models: download the text encoder at this quant. Without it, a matching encoder on disk is used at any quant. |
| `--only PARTS` | All three | Image models: pull only these parts, a comma list of `dit`, `encoder` and `vae`. |
| `--keep-source` | Off | Image models: keep the safetensors VAE after packing it. |

Inside a `model_dirs` root, downloads go under `<org>__<repo>/`. An
interrupted `pull` resumes on the next run. A gated or private repo needs a
token in `HF_TOKEN`, or one stored by `hf auth login`. The retry settings
are in [Environment variables](env-vars.md).

## gmlx validate

`gmlx validate` reports from the header alone whether a GGUF will load. A
remote file is range-read, so the check downloads a few megabytes. The
report names the architecture, the quant codecs, the total size, whether it
fits this Mac's RAM and, for a MoE model, the streaming plan.

```sh
gmlx validate ~/models/Qwen3.8-27B-UD-Q6_K.gguf
gmlx validate hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
gmlx validate hf:unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF     # lists the quants
gmlx validate https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q6_K.gguf
```

| Ref form | Example |
|----------|---------|
| Local path | `~/models/model.gguf` |
| `hf:` file | `hf:org/repo/path/file.gguf`. An `@<revision>` suffix is optional. |
| `hf:` folder | `hf:org/repo/UD-Q5_K_M`. A single model inside resolves, and several are listed. |
| `hf:` repo | `hf:org/repo`. Each quant is listed as a complete ref. |
| Hugging Face page | A `blob`, `tree` or `resolve` link is rewritten to the file or folder. |
| Direct URL | `https://host/path/file.gguf` |

| Flag | Default | Meaning |
|------|---------|---------|
| `ref`, positional | Required | Check this file, folder, repo or URL. |
| `--arch NAME` | Detected | Override architecture detection. |
| `--hf-source ID` | None | Treat the architecture as loadable with this config override. |
| `--max-mb N` | `128` | Cap the header range read. |
| `--json` | Off | Emit the verdict as JSON. |

## gmlx rm

`gmlx rm` deletes a model's GGUF files, partial downloads and companions,
and removes its entry, aliases and default from the config. A file another
model still uses is kept. `rm` prints the plan and asks before it deletes
anything.

An image model id removes its DiT, VAE, text encoder and vision tower. A text encoder
that is a chat model, or that another image model uses, is kept, and `rm`
refuses a chat model that an image model uses as its encoder.

```sh
gmlx rm old-model
gmlx rm old-model --yes
gmlx rm old-model --keep-files
```

| Flag | Default | Meaning |
|------|---------|---------|
| `ID`, positional | Required | Remove the model with this id, alias or discovered id. |
| `--config FILE` | The first default location | Read this config. |
| `--keep-files` | Off | Remove only the config entry. |
| `--yes` | Off | Skip the confirmation. Required without a terminal. |
| `--json` | Off | Emit the result as JSON. Needs `--yes`. |
| `--no-reload` | Off | Do not signal a running server to re-read the file. |

## gmlx sync-models

`gmlx sync-models` rescans the model directories and updates the `models`
block to match disk. Existing entries keep their comments and edits,
entries whose file is gone are dropped, and new files are added. Run it
after adding files by hand or pulling with `--no-register`. An image DiT
found with its text encoder and VAE is added to `image_models`.

```sh
gmlx sync-models
gmlx sync-models --from-hf-cache
gmlx sync-models --dry-run
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | The first default location | Read this config. |
| `--models-dir DIR` | The config's `model_dirs` | Scan this directory. Repeat the flag for more directories. |
| `--from-hf-cache`, `--hf-cache` | The config's `hf_cache` | Also reconcile the Hugging Face cache. |
| `-r`, `--recursive`, `--no-recursive` | Deep | Descend into subdirectories. |
| `--dry-run` | Off | Print the plan without writing. |
| `--no-reload` | Off | Do not signal a running server to re-read the file. |

An entry on an unmounted drive is kept and reported, not dropped.

A scan writes `stream: experts` for a MoE model whose weights are larger
than 90% of the GPU working set, with a comment that says so. Each sync
checks that model again and drops the key once the model fits, such as
after you raise the GPU wired limit. A `stream` key you wrote stays.

## gmlx ps

`gmlx ps` shows the models resident in a running server, with the id,
kind (`chat` or `image`), size, idle time, TTL, pinned and kept state,
and path of each.

| Flag | Default | Meaning |
|------|---------|---------|
| `--url URL` | The managed server | Query the server at this base URL. |
| `--host H`, `--port P` | The managed server | Select the server. |
| `--api-key KEY` | The `GMLX_API_KEY` variable | Send this key to a keyed server. |
| `--json` | Off | Emit JSON. |
| `--image` | Off | Show only the resident image models. |

## gmlx systemone

`gmlx systemone` sends a JSON file holding a `/v1/systemone` request body
to a running server and prints one line per question. The body is
described in [Structured decisions](decisions.md). With `--model` it loads
the GGUF itself and answers offline, with no server.

```sh
gmlx systemone ticket.json
gmlx systemone ticket.json --model diffusiongemma-Q4_K_M.gguf
gmlx systemone ticket.json --model OpenJev-Q4_K_M.gguf
```

A yes or no answer prints as its probability, a choice as the chosen
option with its [confidence](decisions.md#reading-the-answers), and a
score as the expected level with its confidence. A question skipped by
`ask_if` prints `skipped`.

| Flag | Default | Meaning |
|------|---------|---------|
| `REQUEST.json`, positional | Required | Send the request body in this file. |
| `--url URL` | The managed server | Query the server at this base URL. |
| `--host H`, `--port P` | The managed server | Select the server. |
| `--api-key KEY` | The `GMLX_API_KEY` variable | Send this key to a keyed server. |
| `--model GGUF` | None | Answer offline on this GGUF path or configured model id. Not with `--url`, `--host` or `--port`. |
| `--config FILE` | The first default location | Take an offline run's model ids and `server.systemone` settings from this config. Needs `--model`. |
| `--seed N` | The request's `seed` | Replace the request's seed. |
| `--json` | Off | Print the whole response body as JSON. |

## gmlx profiles

`gmlx profiles` prints the family sampling table with its intents, then the
config's user profiles and each model's family. With a model id it prints
that model's resolved sampling for its base and each intent.

```sh
gmlx profiles
gmlx profiles qwen3.8-27b-ud-q6
```

| Flag | Default | Meaning |
|------|---------|---------|
| `model`, positional | None | Resolve this model id or alias. |
| `--config FILE` | The first default location | Read this config. |
| `--json` | Off | Emit JSON. |

## gmlx talk

`gmlx talk` holds a voice chat with a served model. Say the wake phrase,
speak, and the reply streams back as speech. The server needs `stt` and
`tts` configured. Setup and the in-session keys are in
[Voice chat](talk.md).

```sh
gmlx talk
gmlx talk qwen3 --voice bf_emma
gmlx talk --mode vad
gmlx talk --once
```

Most flags override the key of the same name in the
[`talk` block](config.md#voice).

| Flag | Default | Meaning |
|------|---------|---------|
| `model`, positional | The server's default model | Talk to this served model, with an optional `@profile`. |
| `--mode {wake,vad,ptt,text}` | `wake` | Set how a turn starts. |
| `--once` | Off | Hold one exchange without the wake gate, then exit. |
| `--wake-word PHRASE` | `hey assistant` | Wake on this English phrase, which needs no training. |
| `--wake-threshold X` | `0.3` | Set the wake sensitivity from 0 to 1. A higher value means fewer false wakes. |
| `--vad-threshold X` | `0.6` | A frame counts as speech above this probability. |
| `--vad-silence-ms MS` | `550` | This much trailing silence ends an utterance. |
| `--min-speech-ms MS` | `300` | Discard utterances shorter than this. |
| `--voice NAME` | The server's default | Speak with this TTS voice. |
| `--list-voices` | Off | List the server's voices and exit. |
| `--speed X` | `1.0` | Set the speech speed, from 0.25 to 4. |
| `--no-chime` | Off | Disable the wake and idle sounds. |
| `--input-device D`, `--output-device D` | System default | Pick the audio devices by name substring or index. |
| `--list-devices` | Off | List audio devices and exit. |
| `--system TEXT` | The talk default | Set the spoken persona. |
| `--language L` | Detected | Pass this language hint to Whisper. |
| `--max-tokens N` | Until the model stops | Cap each reply at this many tokens. |
| `--brain {chat,assistant}` | `chat` | Use plain chat, or the assistant with tools and memory. |
| `--base-url URL` | The managed server | Connect to this server, which also runs the speech services. |
| `--host H`, `--port P`, `--api-key KEY` | The managed server | Select the server and give its key. |
| `--no-start` | Off | Never start the server. |
| `--start-timeout S` | `180` | Wait this many seconds for an auto-start. |
| `--config PATH` | The first default location | Read the `talk` block from this YAML file. |

## gmlx train

`gmlx train` trains a LoRA adapter on a quantized GGUF base and writes it
as a GGUF adapter. The base stays quantized, so a model that does not fit
in fp16 can still be fine-tuned. The walkthrough is in
[LoRA adapters](lora.md).

```sh
gmlx train base-Q8_0.gguf --data ./my-data --adapter-out my-lora.gguf
gmlx run base-Q8_0.gguf --adapter my-lora.gguf --prompt "..."
```

| Flag | Default | Meaning |
|------|---------|---------|
| `model`, positional | Required | Train on this base GGUF or config id. |
| `--data PATH_OR_ID` | Required | Train on a directory with `train.jsonl` and `valid.jsonl`, or on a Hugging Face dataset id. |
| `--adapter-out PATH` | Required | Write the adapter here. |
| `--config FILE` | The first default location | Resolve an id against this config. |
| `--iters N` | `150` | Train for this many iterations. |
| `--batch-size N` | `4` | Train on batches of this size. |
| `--num-layers N` | `8` | Adapt this many top transformer layers. |
| `--rank N` | `8` | Set the LoRA rank. |
| `--scale F` | `20.0` | Set the LoRA scale. Alpha is scale times rank. |
| `--dropout F` | `0.0` | Set the LoRA dropout, below 1. |
| `--learning-rate F` | `1e-4` | Set the Adam learning rate. |
| `--max-seq-length N` | `2048` | Cap training sequences at this many tokens. |
| `--val-batches N` | `25` | Use this many validation batches in each evaluation. |
| `--steps-per-report N` | `10` | Report the training loss every N steps. |
| `--steps-per-eval N` | `200` | Validate every N steps. |
| `--seed N` | `0` | Seed the random number generator. |
| `--hf-source ID` | None | Fall back to this repo for the tokenizer and config, which is rarely needed. |
| `--grad-checkpoint` | Off | Recompute activations in the backward pass, trading time for memory. Not with `--dropout` above 0. |

The data can be chat messages, prompt and completion pairs, or plain text,
in the formats mlx-lm's trainer accepts.

## gmlx distill

`gmlx distill` trains a LoRA adapter for a small GGUF on a larger model's
outputs. [Distillation](distill.md) is the guide,
[Distillation walkthrough](distill-walkthrough.md) runs one task end to end,
and every action's flags are in the
[Distillation reference](distill-reference.md).

## gmlx image

`gmlx image` generates images from GGUF diffusion models, which
[Image generation](images.md) introduces. Its actions follow.

- `generate` samples images from a DiT GGUF with its text encoder and VAE.
- `bench` times whole images and their phases, with a JSON record.
- `pack-vae` writes a lossless VAE GGUF from a safetensors VAE.

```sh
gmlx image generate z-image-turbo --prompt "a red bicycle" --seed 1 --output bicycle.png
gmlx image generate z-image-turbo --prompt "a red bicycle" --init-image sketch.png --strength 0.6
gmlx image generate qwen-image-turbo --prompt "make it a watercolor" --ref photo.jpg
gmlx image pack-vae vae/diffusion_pytorch_model.safetensors --out z-image-vae-BF16.gguf
gmlx image generate z-image-turbo-Q4_K_M.gguf --encoder Qwen3-4B-Q4_K_M.gguf \
    --vae z-image-vae-BF16.gguf --prompt "a red bicycle" --seed 1 --output bicycle.png
gmlx image bench z-image-turbo-Q4_K_M.gguf --encoder Qwen3-4B-Q4_K_M.gguf \
    --vae z-image-vae-BF16.gguf --sizes 1024x1024,2048x2048 --runs 3 --json
```

Every action exits 0 on success, 1 on a refusal or a runtime failure, 2 on
a usage error and 130 on an interrupt.

### image generate

`generate` loads the DiT, the text encoder and the VAE, encodes the prompt
once and samples each image in turn, at the seed plus the image's index.
Before anything loads, it checks from the file headers that the request
fits in memory. When it does not, `generate` refuses with
[`image request does not fit`](troubleshooting.md#image-request-does-not-fit).

To fit, `generate` runs the two halves of a guided step one at a time,
releases the text encoder after encoding and decodes in tiles. Last, it
loads the text encoder alone, encodes every prompt and reference, and
releases it before the DiT loads, which gives the same image.

The variant of the DiT sets the steps, guidance and size that the flags
leave out, and a Z-Image DiT of an unknown variant needs `--steps` and
`--guidance`. The latents equal those of the Python `sample` call with the
noise of the same seed.

Each family takes the text encoder its DiT was trained with and its own
VAE. A file of another model in either place is refused by name before
the weights load.

| Family | `--encoder` | `--vae` | Defaults |
|--------|-------------|---------|----------|
| Z-Image | Qwen3-4B | The FLUX.1 autoencoder | The variant's steps and guidance. |
| Qwen-Image 2.1 | Qwen3-VL-8B, its text GGUF | The Qwen-Image 2.1 autoencoder | 40 steps and guidance 1. Turbo: 8 steps and guidance 1. |
| Krea 2 | Qwen3-VL-4B-Instruct, its text GGUF | The Qwen-Image autoencoder | Turbo: 8 steps and guidance 1. |
| FLUX.3 klein | Qwen3-VL-4B-Instruct, its text GGUF | The FLUX.2 autoencoder | 16 steps and guidance 1. Base: 50 steps and guidance 3.5. |

FLUX.3 klein Base shares the distilled model's architecture, encoder and
VAE, and gmlx tells the two apart by their tensors. The distilled model
runs any step count on its own schedule.

Qwen-Image 2.1 Turbo shares the base model's architecture, encoder and VAE.
gmlx tells the two apart by their weights. Turbo samples on the eight sigmas
it was distilled to, and refuses any other step count, on the command
line, in a request and in a config's `defaults.steps`. A Qwen-Image 2.1 file
that matches neither, such as a fine-tune, samples as the base model, and
the load line names its variant as unknown.

Qwen-Image 2.1 encodes the prompt in the template of its reference
pipeline, so a prompt gives the same conditioning as it does there.

A guidance other than 1 runs the DiT twice in each step, on the prompt and on
the negative prompt. With no `--negative-prompt` it guides against the
empty prompt, which is encoded like any other prompt.

`--ref` gives Qwen-Image 2.1 a reference image to edit from. Repeat the
flag for several, up to 10, and the prompt can name them in that order, as
the first picture and the second. Each reference is resized to about
`--ref-size` pixels on a side at its own aspect ratio, with each side a
multiple of 32, and a reference keeps its alpha channel. The text encoder
reads the references through its vision tower, which `--encoder-mmproj`
loads from the encoder's mmproj GGUF, and the VAE encodes them. `--size`
sets the size of the output, which defaults to the entry's `defaults.size`
or the variant's, and not to a reference's. The fit check counts the references, and a request that does
not fit is refused before any model loads. Z-Image and Krea 2 take no
reference images, and `--ref` exits 2 on them.

`--ref` gives FLUX.3 klein up to 10 reference images too. Its text encoder
reads the prompt alone, so it needs no `--encoder-mmproj`, and the VAE
encodes the references for the DiT. A single reference is scaled down to
at most the area of 2048x2048 pixels and padded out to sides that are
multiples of 16. Of several, each is scaled down to at most the area of
1024x1024 and cropped to sides that are multiples of 16. Without `--size`
the output has the first reference's size, and the output of a single
reference is cropped back to the scaled image. FLUX.3 klein sizes its
references by this rule, and `--ref-size` exits 2 on it.

`--adapter` applies a LoRA safetensors file to the DiT, live beside the
quantized weights, so the file is read once and nothing is merged. The
kohya, diffusers, PEFT and OneTrainer key spellings are read. The rank and
alpha in a diffusers file's PEFT header scale each pair as the diffusers
loader scales it, rank-stabilized LoRA included. ComfyUI does not read that
header, so such a file runs at another strength there, and gmlx warns of
it. The strength follows the path after a colon and defaults to 1, and it
may be negative. Repeat the flag to stack adapters, whose deltas add. A
LoHa, LoKr, DoRA, full-weight or Tucker LoCon file is refused by name from
its header, before any model loads. So is a file for another model family or one with keys
that match no module of the DiT, which its GGUF header names. An adapter's
text encoder pairs are applied to the text encoder at the same strength.

A PNG holds a text chunk keyed `gmlx` with the prompt, seed, steps, size,
sampler, scheduler, guidance, adapters, file names and gmlx version, unless
`--no-metadata` is set. Each adapter is recorded with its file name, its
strength, the `sha256` of the file, and `rows_sha256`, a digest of the
LoRA pairs gmlx applied and the layers it applied them to. Each reference
is recorded with its file name, the size it ran at and the `sha256` of the
file. The size is that of the image written. An image cropped back from the
canvas of one reference also records `canvas`, the size it was sampled at,
and `crop`, the left, top, right and bottom of the image in it.

| Flag | Default | Meaning |
|------|---------|---------|
| `MODEL`, positional | Required | Sample with this DiT GGUF, or with the `image_models` entry of this id, which supplies the encoder, mmproj, VAE and adapters. |
| `--config FILE` | The first default location | Read the `image_models` entry that `MODEL` names from this config. |
| `--variant NAME` | From the weights, or the entry's `variant` | Run the DiT as this variant, such as `turbo`, when its weights match no known variant. A Krea 2 file of unknown weights needs `--variant turbo`. |
| `--encoder GGUF` | Required with a DiT GGUF | Encode the prompt with this text encoder. |
| `--encoder-mmproj GGUF` | None | Read the `--ref` images with this mmproj GGUF of the text encoder. Qwen-Image 2.1 needs it with `--ref`, and it is not loaded without it. |
| `--vae PATH` | Required with a DiT GGUF | Decode with this VAE, a GGUF or safetensors file. |
| `--prompt TEXT` | Required, or `--prompt-file` | Generate from this prompt. |
| `--prompt-file FILE` | None | Read the prompt from this file. |
| `--negative-prompt TEXT` | Empty | Guide away from this prompt, or from the empty prompt without it. It is unused when `--guidance` is 1. |
| `--size WxH` | The init image's, else a FLUX.3 klein reference's, else the entry's `defaults.size`, else the variant's | Set the width and height, each a multiple of 32 on Qwen-Image 2.1 and of 16 on the others. Another size exits 2 and names the multiple. |
| `--steps N` | The variant's | Sample for N steps. |
| `--seed N` | Random, and printed | Seed the first image. |
| `--guidance CFG` | The variant's | Set classifier-free guidance. 1 turns it off. |
| `--sampler NAME` | `euler` | Sample with `euler` or `uni_pc_bh2`. |
| `--init-image PATH` | None | Start from this image, resized to `--size` (img2img). |
| `--mask PATH` | None | Change only where this mask is white (inpainting). It needs `--init-image`. |
| `--strength F` | `0.6`, or `1` with `--mask` | Set how far img2img moves from the init image, from 0.01 to 1. |
| `--ref PATH` | None | Edit from this reference image, on Qwen-Image 2.1 and FLUX.3 klein. Repeat the flag for up to 10 images, in the order the prompt names them. |
| `--ref-size N` | `1024` | On Qwen-Image 2.1, resize each reference to about N x N pixels at its own aspect ratio, from 256 to 1024. It needs `--ref`. |
| `--count N` | `1` | Generate N images. |
| `--output PATH` | `./gmlx-<time>-<seed>.<ext>` | Write to this file, directory or template with `{seed}` and `{index}`. |
| `--format png/jpeg/webp` | From `--output`, else `png` | Encode the images in this format. |
| `--quality N` | `90` | Set the jpeg and webp quality, from 1 to 100. |
| `--no-metadata` | Off | Leave out the PNG text chunk. |
| `--background opaque/transparent` | `opaque` | `transparent` writes the alpha channel that Qwen-Image 2.1 decodes, as a png or a webp. It exits 2 with jpeg, and on a model that decodes no alpha channel. |
| `--save-latent PATH` | None | Also write the final latents and seeds to this `.npz` file. |
| `--adapter PATH[:STRENGTH]` | None | Apply this LoRA safetensors file at this strength, 1 when left out. Repeat the flag to stack adapters. |
| `--encoder-release auto/always/never` | `auto` | Release the text encoder after encoding when the memory plan needs it, always, or never. `never` exits 2 when the plan needs it. |
| `--json` | Off | Print a JSON record on stdout, and progress on stderr. |

With several images, a `--output` file gains `-<seed>` before its
extension.

Under a `--ref-size` of 320, a reference that is not square can come out
under the 256 x 256 pixels the text encoder takes. The command then exits 2
and names the `--ref-size` that image needs.

### image bench

`bench` loads the model once and, for each size, runs `--warmup` images
untimed and then `--runs` timed ones. Each run starts with empty
conditioning caches and times the prompt encoding, every step, the decode
and the whole image, with the peak of MLX memory. The summary gives the
load time of each part, the first step apart from the median and mean of
the rest, and the medians of the other phases.

| Flag | Default | Meaning |
|------|---------|---------|
| `MODEL`, positional | Required | Time this DiT GGUF, or the `image_models` entry of this id. |
| `--config FILE` | The first default location | Read the `image_models` entry that `MODEL` names from this config. |
| `--variant NAME` | From the weights, or the entry's `variant` | Run the DiT as this variant, such as `turbo`, when its weights match no known variant. A Krea 2 file of unknown weights needs `--variant turbo`. |
| `--encoder GGUF` | Required with a DiT GGUF | Encode the prompt with this text encoder. |
| `--vae PATH` | Required with a DiT GGUF | Decode with this VAE. |
| `--adapter PATH[:STRENGTH]` | None | Time with this LoRA safetensors file applied, as in `generate`. Repeat the flag to stack adapters. |
| `--prompt TEXT` | A fixed bench prompt | Generate from this prompt. |
| `--sizes LIST` | The entry's `defaults.size`, else the variant's size | Time each of these comma-separated sizes, such as `1024x1024,2048x2048`. |
| `--steps N` | The variant's | Sample for N steps. |
| `--guidance CFG` | The variant's | Set classifier-free guidance. |
| `--sampler NAME` | `euler` | Sample with `euler` or `uni_pc_bh2`. |
| `--runs N` | `2` | Time N runs per size. |
| `--warmup N` | `1` | Run N untimed images per size first. |
| `--seed N` | `0` | Seed every run. |
| `--label TEXT` | None | Record this label with the results. |
| `--via-steps` | Off | Run each step through the [image step routes](internals/image-steps.md) on a loopback port of this process. Not with `--adapter`. |
| `--json` | Off | Print one JSON record per size on stdout, and the summaries on stderr. |
| `--json-out FILE` | None | Write one JSON record per size to this file. |

A record has schema `gmlx.image.bench/1`. It holds the gmlx and MLX
versions, the machine's chip, memory and OS, the model's files with their
sizes, the settings, every run, the summary, and the transport, which is
`direct` for a run in the same process.

With `--via-steps` the transport is `steps`, and the record holds the
fields that [Timing the step routes](internals/image-steps.md#timing-the-step-routes)
describes.

### image pack-vae

`pack-vae` copies the tensors of a safetensors VAE into a GGUF under their
official names, in torch layout and their source dtype, so the file is
lossless and other GGUF readers can use it. It then reads the GGUF back
and compares every tensor bit for bit before the file replaces `--out`.

It packs the VAE of each image family: the FLUX.1 autoencoder for
`z_image`, the Qwen-Image 2.1 autoencoder for `qwen_image21`, whose
official file is F32 and stays F32, the Qwen-Image autoencoder for
`krea2`, whose file is BF16, and the FLUX.2 autoencoder for `flux3_klein`,
whose Comfy-Org `flux2-vae.safetensors` is F32 and needs no config. It
packs from that file's layout or from the original one of BFL's
`ae.safetensors`. The family follows from the tensor names and
shapes. A file that fills none of them is refused with what it holds, for
example another family's VAE under `--family`, or a video VAE of the same
layout.

A Qwen-Image 2.1 file may leave out its `time_conv` tensors, which one
image never reads. The places where the video layout halves time come
from the official config, so a `time_conv` tensor anywhere else is
refused by name. So is a `--source-config` whose `temperal_downsample`,
`latents_mean` or `latents_std` differs from the official one.

A VAE config names its Autoencoder class in `_class_name`, as diffusers
writes it. A `config.json` beside `SRC` that is not one, such as the
pipeline config beside BFL's `ae.safetensors`, is left out with a note,
and a `--source-config` that is not one is refused.

| Flag | Default | Meaning |
|------|---------|---------|
| `SRC`, positional | Required | Pack this safetensors VAE. |
| `--out FILE.gguf` | Required | Write the GGUF here. |
| `--dtype f32/f16/bf16` | The source dtype | Store every tensor in this dtype. A narrower dtype is not lossless, and the file records that. |
| `--family NAME` | The family the file fills | Pack the VAE of this image family, such as `z-image`, `qwen-image-2.1`, `krea-2` or `flux-3-klein`. |
| `--source-repo ID` | `unknown` | Record this repository as the source. |
| `--source-revision REV` | `unknown` | Record this revision of the source repository. |
| `--source-config FILE` | `config.json` beside `SRC`, when it is a VAE config | Record this VAE config. |
| `--force` | Off | Replace `--out` when it exists. |
| `--json` | Off | Print a JSON record of the packed file. |

## gmlx comfyui

`gmlx comfyui` puts the gmlx node pack into ComfyUI, as
[ComfyUI](comfyui.md) describes. Each action takes its own flags.

```sh
gmlx comfyui install
gmlx comfyui install --comfyui-dir ~/src/ComfyUI
gmlx comfyui status
gmlx comfyui uninstall
```

`install` finds ComfyUI, refuses one older than the pack needs, copies the
pack to a folder gmlx owns and links it into ComfyUI's `custom_nodes`. Run
again, it refreshes the copy and the links it made before.

| Flag | Default | Meaning |
|------|---------|---------|
| `--comfyui-dir DIR` | The folders installed before, else the ComfyUI found | Install into this ComfyUI: its folder, the folder above its `custom_nodes`, or a Comfy Desktop install folder. |
| `--link-source` | Off | Link to the pack in the gmlx source tree instead of a copy, for work on the pack. |
| `--model ID` | The config's default image model | The image model a new loader node starts on. |

`uninstall` removes the links, the copy and the record. With
`--comfyui-dir DIR` it removes the link from that ComfyUI only. `status`
prints the copy, the gmlx command the nodes run and each ComfyUI with its
version; `--json` prints the same as JSON.

## gmlx doctor

`gmlx doctor` checks what a working setup needs and prints a PASS, WARN,
FAIL or SKIP line for each check, with the fix named. It covers the
runtime, the config, each model and service, background servers, login
items, container mode, optional extras, the Hugging Face token, RAM and
disk space. Its `image` check reads the headers of each image model's DiT,
text encoder, vision tower and VAE, and fails one whose parts are not of
one family. Once `gmlx comfyui install` has run, its `comfyui` check reads
each ComfyUI folder's link to the gmlx nodes and its version against the
range the nodes were tested on. No check uses the network.

```sh
gmlx doctor
gmlx doctor --deep
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | The first default location | Read this config. |
| `--deep` | Off | Also read each configured chat model's header. The `image` check reads the image models' headers on every run. |
| `--json` | Off | Emit JSON. |

## gmlx completion

`gmlx completion` prints a completion script for zsh, bash or fish. It
completes verbs and their actions, flags, model ids from your config,
client names for `launch`, and the host, port and URL of your background
servers.

```sh
eval "$(gmlx completion zsh)"      # ~/.zshrc
eval "$(gmlx completion bash)"     # ~/.bashrc
gmlx completion fish | source      # ~/.config/fish/config.fish
```

With these lines the shell asks the installed gmlx at each start, so the
script needs no update after an upgrade. A script saved to a file does.

| Flag | Default | Meaning |
|------|---------|---------|
| `shell`, positional | None | Print the script for `zsh`, `bash` or `fish`. Without a shell, it prints the help with the install lines. |

## Exit codes

Every verb exits 0 on success and 2 when a flag or argument is wrong.
These verbs also use other codes:

| Verb | Codes |
|------|-------|
| `serve` | 2 when a bare `serve` finds no config. |
| `status` | 0 when a server is running, 3 when none is. |
| `restart` | 1 when a file the server needs is gone or its config does not load. The old server keeps running. |
| `comfyui` | 1 for no ComfyUI, a `--comfyui-dir` that is not one, one too old, a `status` problem or no install, or a settings file that is not a JSON object. 2 for several. |
| `list` | 2 when no config was found or it failed to load. |
| `run` | 1 when the model fails to load or to generate, 2 when a flag, id or file is wrong and nothing has loaded yet, 130 when interrupted. |
| `validate` | 0 when the file will load or the quants are listed, 1 when it will not load, 2 when the reference cannot be read. |
| `rm` | 1 when you declined or a file could not be deleted, 2 for an unknown id, a bad config, or a missing `--yes`. |
| `ps` | 1 when the server answered with an error or is not gmlx, 3 when no server was reachable. |
| `systemone` | 1 when the file, the request or the model was refused, or no server was reachable. |
| `doctor` | 1 when a check failed, 2 when the `--config` file does not exist. |
| `launch` | The client's own status once it runs, else a code from the next table. |

### Launch exit codes

Before the client runs, `gmlx launch` exits with one of these codes, which
follow sysexits(3) where one fits. Each code apart from 130 comes with a
message that names the cause and the next step. A script can retry after
75.

| Code | Meaning |
|------|---------|
| 0 | `--config-only`, `--remove-home`, `--stop`, `--list` or `--forget-share` finished, or `--detach` started the session. |
| 1 | Launch refused, such as a folder it will not share, a flag that does not fit this launch or a ComfyUI too old for the gmlx nodes. |
| 2 | A flag is unknown, abbreviated or refused, the command names a client it should not or none where it needs one, or several ComfyUI installs fit. |
| 69 | Something launch needs is missing or does not answer: Apple container, its kernel, the client or ComfyUI's environment, the server or a model. |
| 75 | Something is busy: a project another launch uses, a volume or port in use, or a server that is starting or full. |
| 78 | No gmlx config exists, or the config or its [`launch`](config.md#launch) block does not load. |
| 125 | The container could not open its connections to the Mac, such as when the image already uses a port that launch forwards. |
| 126 | The client's command is in the image but cannot run. |
| 127 | The image has no client command, or no shell for `--shell`. |
| 130 | Ctrl-C arrived before the client started. |
| 128 + N | Signal N arrived before the client started. |
