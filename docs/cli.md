# CLI reference

The `gmlx` command has one verb per task, each with its own flags. Run
`gmlx <verb> --help` for a verb's options, and `--help-all` on `run` and
`chat` for the full set. The guides linked from each verb explain when to
use a flag.

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
| `--disk-cache [GB]` | Off | Enable the on-disk prompt cache with this cap for each model, 50 GB when the flag is bare. |
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
| `--kv-quant-scheme {uniform,kvarn}` | `uniform` | Pick affine or [kvarn](kv-quantization.md) quantization. Under kvarn `--kv-bits` defaults to 6. |
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

These flags turn on speculative decoding for a positional model:

| Flag | Default | Meaning |
|------|---------|---------|
| `--speculative` | Off | Speculate with the model's own MTP head or `--draft-gguf`. A config `discover` scan enables it on its own. |
| `--speculative-width-cap N` | Drafter default | Speculate only while at most N requests decode together. `0` removes the cap. |

`--draft-gguf`, `--native-mtp`, `--draft-block-size` and `--stochastic-mtp`
work as under [run](#speculative-decoding-flags), and the first two imply
`--speculative`. Under serve, `--stochastic-mtp` applies to the whole
server.

`--stream-experts`, `--stream-cpu`, `--stream-fast-disk`,
`--prefill-feeder`, `--no-prefill-feeder`, `--decode-feeder`,
`--no-decode-feeder`, `--gpu-keepwarm`, `--no-gpu-keepwarm`,
`--moe-experts`, `--moe-expert-mass`, `--moe-miss-shed`, `--moe-layer-shed`
and `--moe-prestage` stream a positional model bigger than memory. They
work as under [run](#streaming-flags), and `--gpu-keepwarm` applies to the
whole server.

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

`gmlx stop` stops a background server. Generation in progress ends when
the timeout runs out.

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
are tagged, aliases follow, and the default model is marked with `*`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | The first default location | Read this config. |
| `-v`, `--paths` | Off | Also show each model's GGUF path. |
| `--json` | Off | Emit JSON. |

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
| `--max-kv-size N` | None | Cap the KV cache with a rotating window. Works with kvarn but not with affine `--kv-bits`. |
| `--kv-bits N` | Off | Quantize the KV cache to 2, 3, 4, 6 or 8 bits affine, or to 2, 3, 4, 5, 6 or 8 under kvarn, where it defaults to 6. |
| `--kv-group-size N` | `64` | Set the affine quantization group size. |
| `--kv-quant-scheme {uniform,kvarn}` | `uniform` | Pick affine or kvarn quantization. |
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
| `--stream-experts` | Off | Stream the routed experts from disk. Attention and the KV cache stay on GPU. |
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

A server client refuses `--adapter`, `--mmproj`, `--chat-template`,
`--no-chat-template` and `--chat-template-config`, and ignores the other
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

`launch` refuses an abbreviated flag such as `--cont`.

| Flag | Default | Meaning |
|------|---------|---------|
| `client`, positional | None | Launch `claude-code`, `opencode`, `pi`, `omp`, `hermes`, `goose`, `aichat`, `elia`, `open-webui`, `dsh`, `menubar` or a [custom agent](launch-agents.md). |
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
`model_dirs` root is added to the config, and a running server reloads it.

```sh
gmlx pull hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
gmlx pull hf:org/repo/model.gguf --to ~/models
gmlx pull hf:org/gemma-3-27b-GGUF/gemma-3-27b-Q4_K_M.gguf mmproj-F16.gguf
```

| Flag | Default | Meaning |
|------|---------|---------|
| `REF`, positional | Required | Download these `hf:<org>/<repo>/<file.gguf>[@rev]` references or URLs. Later bare filenames resolve in the first ref's repo. |
| `--to DIR`, `--out DIR` | The first `model_dirs` root | Download into this directory instead, with no repo subfolder. |
| `--config FILE` | The first default location | Read `model_dirs` from this config. |
| `--force` | Off | Download even when the header check or the disk-space check fails. |
| `--no-register` | Off | Do not add the file to the config. |
| `--hf-source ID` | None | Treat the architecture as loadable with this config override. |
| `--max-mb N` | `128` | Cap the header range read. |
| `--json` | Off | Emit each verdict as JSON before downloading. |

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
after adding files by hand or pulling with `--no-register`.

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

## gmlx ps

`gmlx ps` shows the models resident in a running server, with the id,
size, idle time, TTL, pinned and kept state, and path of each.

| Flag | Default | Meaning |
|------|---------|---------|
| `--url URL` | The managed server | Query the server at this base URL. |
| `--host H`, `--port P` | The managed server | Select the server. |
| `--api-key KEY` | The `GMLX_API_KEY` variable | Send this key to a keyed server. |
| `--json` | Off | Emit JSON. |

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
outputs. The walkthrough is [Distillation](distill.md), and every action's
flags are in the [Distillation reference](distill-reference.md).

## gmlx doctor

`gmlx doctor` checks what a working setup needs and prints a PASS, WARN,
FAIL or SKIP line for each check, with the fix named. It covers the
runtime, the config, each model and service, background servers, login
items, container mode, optional extras, the Hugging Face token, RAM and
disk space. No check uses the network.

```sh
gmlx doctor
gmlx doctor --deep
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | The first default location | Read this config. |
| `--deep` | Off | Also read each configured model's header. |
| `--json` | Off | Emit JSON. |

## gmlx completion

`gmlx completion` prints a completion script for zsh, bash or fish. It
completes verbs, flags, model ids from your config, client names for
`launch`, and the host, port and URL of your background servers.

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
| `list` | 2 when no config was found or it failed to load. |
| `run` | 1 when the file cannot load, 2 on a usage or file error, 130 when interrupted. |
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
| 1 | Launch refused, such as a folder it will not share or a flag that does not fit this launch. |
| 2 | A flag is unknown, abbreviated or refused, or the command names a client it should not, or none where it needs one. |
| 69 | Something launch needs is missing or does not answer: Apple container, its kernel, the client, the server or a model. |
| 75 | Something is busy: a project another launch uses, a volume or port in use, or a server that is starting or full. |
| 78 | No gmlx config exists, or the config or its [`launch`](config.md#launch) block does not load. |
| 125 | The container could not open its connections to the Mac, such as when the image already uses a port that launch forwards. |
| 126 | The client's command is in the image but cannot run. |
| 127 | The image has no client command, or no shell for `--shell`. |
| 130 | Ctrl-C arrived before the client started. |
| 128 + N | Signal N arrived before the client started. |
