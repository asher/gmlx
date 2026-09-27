# CLI reference

The `gmlx` command has one verb per task, such as `serve`, `run` or
`pull`, each with its own flags, defaults and exit codes. The guides linked
from each verb explain when to use a flag.

`gmlx` has these verbs:

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

`gmlx --version` prints the version, and a verb's options come from
`gmlx help <verb>` or `gmlx <verb> --help`. `run` and `chat` also take
`--help-all` for their full flag set. `gmlx ls` is an alias for `gmlx list`.

Many settings exist as a flag, a config key and an environment variable.
How they combine is in
[Flags and environment variables](config.md#flags-and-environment-variables),
and the variables are in [Environment variables](env-vars.md).
Sampling flags you leave unset take the model's
[family defaults](family-defaults.md). A `--config` default of the first
default location means the first file that exists in the order that
[Where gmlx looks](config.md#where-gmlx-looks) gives.

## gmlx init

`gmlx init` scans your model directories and writes a starter config. Run
bare on a terminal, it opens a wizard that lets you rename models, set a
default and aliases, and enable the prompt cache and the speech, embedding
and rerank services. With flags it writes the file without asking.

```sh
gmlx init                                  # the wizard
gmlx init --models-dir ~/models            # flag-driven, writes ~/.config/gmlx/gmlx.yaml
gmlx init --models-dir ~/models -r --out ./gmlx.yaml
gmlx init --from-hf-cache                  # models already in the Hugging Face cache
```

These flags control `gmlx init`:

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

Auto-named ids carry the quant in compact form, such as `qwen3-0.6b-q4`, and
fall back to the full codec when two quants would collide. An empty directory
is accepted and produces a valid config with no models. When a server is
already running the config you rewrote, `init` signals it to reload. The
walkthrough is in the [Quickstart](quickstart.md#serving-models), and
[Configuration](config.md) describes the file that `init` writes.

## gmlx serve

`gmlx serve` runs the server. It detaches by default and returns when the
server is ready or `--start-timeout` runs out, so the same shell can run
`gmlx launch` next. `--foreground` keeps it attached
instead. A background server keeps a runfile and a log under
`~/.cache/gmlx/` and, on a macOS desktop session, raises the [menu bar
app](menubar.md).

```sh
gmlx serve                                  # the config in the default location
gmlx serve --config ./gmlx.yaml
gmlx serve --models-dir ~/models --recursive
gmlx serve model-Q4_K_M.gguf                # one model, id from the filename
gmlx serve model.gguf --mmproj mmproj.gguf  # one vision model
```

These flags say where the models come from:

| Flag | Default | Meaning |
|------|---------|---------|
| `model`, positional | None | Serve this GGUF, pinned, with an id derived from the filename. |
| `--config FILE` | The first default location | Serve a YAML config. |
| `--models-dir DIR` | None | Serve a scan of this directory. Repeat the flag for more directories. |
| `-r`, `--recursive`, `--no-recursive` | Shallow | Descend when scanning. |
| `--hf-cache`, `--from-hf-cache` | Off | Let Hugging Face ids resolve from the local cache, never the network. |
| `--print-config` | Off | Print the resolved config as YAML and exit. |

These flags control the process and its lifecycle:

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

These flags set memory and scheduling. Most are also `server` keys in the
config, where the same settings apply to a config-mode server:

| Flag | Default | Meaning |
|------|---------|---------|
| `--budget-gb F` | 0.8x the GPU working set | Keep the weights of all [resident](glossary.md#resident) models within this many GB. |
| `--max-models N` | None | Keep at most this many models resident. |
| `--pin ID_OR_PATH` | None | Load this model at start and never evict it. Repeat the flag for more models. |
| `--max-tokens N` | Until the model stops | Cap a completion at this many tokens when the request sets no cap. Without the flag, the cap is the room left in the context. |
| `--no-family-defaults` | Off | Do not seed each family's model-card sampling under profiles and requests. In config mode a reload restores `server.family_defaults`. |
| `--prefill-step-size N` | `2048` | Prefill in chunks of this many tokens. A lower value caps peak memory. |
| `--dtype {auto,bfloat16,float16}` | `auto` | Set the activation width. `auto` picks float16 on M1 and M2. |
| `--decode-prefill-ratio R` | `auto` | Make each prefill chunk wait until decoding streams have had this multiple of its GPU time. `0` restores stock scheduling. |
| `--prefill-tick-ms MS` | `500` | Give each prefill chunk this wall-clock budget while streams decode. `0` never halves a chunk. |
| `--ignore-eos` | Off | Decode each request to its output cap. |

These settings apply to a positional GGUF only. In config mode the same
things are per-model keys under [models](config.md#models):

| Flag | Default | Meaning |
|------|---------|---------|
| `--mmproj PATH` | None | Load this projector GGUF to make the model multimodal. |
| `--hf-source REPO` | None | Take a vision model's processor and config from this repo, which is rarely needed. |
| `--adapter PATH` | None | Apply this GGUF LoRA adapter at load, for text only. |
| `--chat-template STR_OR_PATH` | The GGUF's | Replace the chat template with inline Jinja or a `.jinja` or `.txt` file. |
| `--thinking {on,off,adaptive}` | Template default | Turn reasoning on, off or adaptive through the model's template variable. |
| `--thinking-budget N` | Unlimited | Cap reasoning tokens for each request. `0` closes thinking at once. |
| `--reasoning-effort LEVEL` | Template default | Set the reasoning level on models whose template grades thinking, such as `low`, `medium` or `high`. |
| `--profile NAME` | None | Apply a built-in intent such as `coding` or `reasoning-high`, resolved for the model's family. An unknown name is refused at start. |
| `--system-prompt STR` | None | Use this system prompt when the request has none. |
| `--chat-template-config JSON` | None | Pass this JSON object of extra chat-template variables through verbatim. |
| `--kv-bits N` | Off | Quantize the KV cache to 2, 3, 4, 6 or 8 bits affine, or to 2, 3, 4, 5, 6 or 8 under kvarn. |
| `--kv-group-size N` | `64` | Set the affine quantization group size. |
| `--kv-quant-scheme {uniform,kvarn}` | `uniform` | Pick affine or [kvarn](glossary.md#kvarn) quantization. Under kvarn `--kv-bits` defaults to 6. |
| `--kv-tail-tokens N` | `1024` | Under kvarn, keep this many newest tokens fp16, a multiple of 128. |
| `--max-kv-size N` | None | Cap the request context budget at N tokens. |
| `--quantized-kv-start N` | `0` | Keep the cache in fp16 until it holds this many tokens, then quantize all of it. Batches and kvarn quantize from the first token. |

The KV flags are the config's [`load` keys](config.md#model-loading).
`--kv-quant-scheme kvarn` on a positional model is the same as
`load: {kv_quant_scheme: kvarn}` on a config model, priced and reported the
same way.

These flags set a positional model's sampling defaults on top of the
family defaults that `gmlx profiles` prints. They are the
config's [`sampling` keys](config.md#sampling). A default applies to a
request that omits the field, and a request that sends the field wins, so
`--temp 0` does not pin a client that sends its own temperature.

| Flag | Default | Meaning |
|------|---------|---------|
| `--temp T` | Family base | Set the sampling temperature. |
| `--top-p P` | Family base | Set the nucleus probability. `0` disables the filter. |
| `--top-k N` | Family base | Keep this many candidate tokens. `0` disables the filter. |
| `--min-p P` | Family base | Drop tokens less likely than this share of the best token. `0` disables the filter. |
| `--seed N` | None | Seed every request that sends no seed of its own. |
| `--repetition-penalty X` | None | Penalize tokens repeated within the last `--repetition-context-size` tokens. |
| `--repetition-context-size N` | `20` | The repetition penalty looks back over this many tokens. |
| `--presence-penalty X` | None | Penalize any token already generated. |
| `--frequency-penalty X` | None | Penalize a token by how often it was generated. |
| `--stop STR` | None | Stop chat completions at this sequence. Repeat the flag for more sequences. |
| `--xtc-probability P` | None | Set the XTC sampling probability, which speculative models do not support. |
| `--xtc-threshold T` | None | Set the XTC sampling threshold. |
| `--thinking-start-token STR` | `<think>` | Set the model's opening reasoning marker. |
| `--thinking-end-token STR` | `</think>` | Set the model's closing reasoning marker. |

These flags control speculative decoding:

| Flag | Default | Meaning |
|------|---------|---------|
| `--speculative` | Off | Speculate with the model's own MTP head or `--draft-gguf`. A config `discover` scan enables it on its own. |
| `--draft-gguf PATH` | None | Draft with this separate [drafter](glossary.md#drafter) GGUF, which implies `--speculative`. |
| `--native-mtp` | Off | Draft with the model's own head, even when `--draft-gguf` is set. The flag implies `--speculative`. |
| `--draft-block-size N` | Drafter default | Set the block size of each round, which drafts N-1 tokens and checks them in one N-token target pass. |
| `--speculative-width-cap N` | Drafter default | Speculate only while at most N requests decode together. `0` removes the cap. |
| `--stochastic-mtp` | Off | Accept sampled drafts by rejection sampling, which accepts more but is not token-identical. |

These flags stream a model bigger than memory, as
[Models larger than memory](streaming.md) explains:

| Flag | Default | Meaning |
|------|---------|---------|
| `--stream-experts` | Off | Stream the routed experts from disk. Attention and the KV cache stay on GPU. |
| `--stream-cpu` | Off | Run the whole model on the CPU device from the page cache. |
| `--stream-fast-disk {auto,on,off}` | `auto` | Set the streamed-decode prefetch policy under `--stream-experts`. `auto` probes the drive. |
| `--prefill-feeder`, `--no-prefill-feeder` | On | Stage expert prefill directly from the GGUF. |
| `--decode-feeder`, `--no-decode-feeder` | On under `--stream-experts` | Decode from a wired expert [arena](glossary.md#arena) that keeps the experts the router picks most often. |
| `--gpu-keepwarm`, `--no-gpu-keepwarm` | On with the decode feeder | Keep GPU clocks high while a streamed model decodes, or turn that off. |
| `--moe-experts K` | Trained | Cap the router at K experts for each token, which is lossy. |
| `--moe-expert-mass P` | Off | Keep the smallest expert set covering share P of gate mass, which is lossy. |
| `--moe-miss-shed P` | Off | Drop experts that would miss the arena down to share P, which is lossy. |
| `--moe-layer-shed P` | Off | Skip a streamed layer's experts with probability P, which is lossy. |
| `--moe-prestage {ranked,keepers}` | `ranked` | `keepers` filters prestage predictions through the miss-shed policy, so it needs `--moe-miss-shed`. |

These flags enable services. Each is also a `server` key and is described
in [Speech, embeddings and rerank](services.md):

| Flag | Default | Meaning |
|------|---------|---------|
| `--stt [MODEL]` | Off | Serve speech-to-text at `POST /v1/audio/transcriptions`, with `whisper-turbo` when the flag is bare. |
| `--tts [MODEL]` | Off | Serve text-to-speech at `POST /v1/audio/speech`, with `kokoro` when the flag is bare. |
| `--embeddings [MODEL]` | Off | Serve embeddings at `POST /v1/embeddings`, with `qwen3-embed-0.6b` when the flag is bare. No extra is needed. |
| `--rerank [MODEL]` | Off | Serve reranking at `POST /v1/rerank`, with `qwen3-rerank-0.6b` when the flag is bare. No extra is needed. |

`serve` has no `--api-key` flag, because the key lives in the config.
[Address and authentication](config.md#address-and-authentication) gives
the reasons.

Each completed request logs a line with the endpoint, model, token counts,
timing and the MLX memory in use afterwards. A finish reason other than
`stop` adds `finish=<reason>` before the memory fields:

```text
[req] 2026-06-15 16:07:42 /chat/completions qwen3-0.6b prompt=19 gen=3 ttft=0.47s prefill=45t/s decode=172.6t/s total=0.51s active=1.3G cache=0.2G
```

## gmlx stop

`gmlx stop` stops a background server with SIGTERM to the process group,
then SIGKILL after the timeout. Before it signals, it checks that the pid
belongs to the gmlx server, and it clears and reports any stale runfiles
found during the check.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Select the server by host when several run in the background. |
| `--port P` | The managed server | Select the server by port. |
| `--timeout S` | `15` | Send SIGKILL after this many seconds, which ends any in-flight generation. |
| `--stale` | Off | Clear runfiles whose server has exited and signal nothing. |

## gmlx status

`gmlx status` prints a background server's pid, uptime, URL, log path and
how it is managed. It uses `/health`, so it needs no API key. Stale runfiles
are listed with the reason and their age.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Select the server by host. |
| `--port P` | The managed server | Select the server by port. |
| `--json` | Off | Emit JSON. |

The command exits 0 when a server is running and 3 when none is.

## gmlx restart

`gmlx restart` stops the server and relaunches it with the arguments
recorded in its runfile, from any directory.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Select the server by host. |
| `--port P` | The managed server | Select the server by port. |
| `--timeout S` | `15` | Send SIGKILL after this many seconds during the stop. |
| `--start-timeout S` | `40` | Wait this many seconds for the new process to become ready. |

## gmlx logs

`gmlx logs` prints the tail of a background server's log. The menu bar's
health polls are filtered out of it.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Select the server by host. |
| `--port P` | The managed server | Select the server by port. |
| `-n`, `--lines N` | `40` | Print this many lines. |
| `-f`, `--follow` | Off | Keep printing as the log grows. |
| `--clear` | Off | Truncate the log and exit. |

## gmlx service

`gmlx service` installs a launchd login item on macOS. By default the item
is the menu bar app, which starts the server at login unless
`--no-autostart` is set. A server run this way gets its own identity, so
macOS attributes permission prompts to gmlx instead of to your terminal.
`install` takes the `serve` flags, and the server it starts now is the one
it starts again at each login.

```sh
gmlx service install --config ~/.config/gmlx/gmlx.yaml
gmlx service status
gmlx service uninstall
```

| Subcommand | Flags | Meaning |
|------------|-------|---------|
| `install` | The `serve` flags, `--no-autostart`, `--headless` and `--keepalive` | Register the login item and start now. |
| `status` | `--host H`, `--port P` | Print the launchd state. |
| `uninstall` | `--host H`, `--port P` | Unload and remove the item. |

| Flag | Default | Meaning |
|------|---------|---------|
| `--no-autostart` | Off | Install the menu bar item without starting the server at login. |
| `--headless` | Off | Install a server-only agent with no menu bar, for machines without a desktop session. |
| `--keepalive`, `--no-keepalive` | On | With `--headless`, restart the server when it crashes. |

The server stays an ordinary background process, and one you stop stays
stopped until the next login. Stopping a headless server takes
`service uninstall` rather than `stop`, and the two modes cannot share a
host and port. For the menu bar side, read [Menu bar app](menubar.md).

## gmlx list

`gmlx list` lists the models a config defines, which is the set of ids a
request can address rather than the files on disk. Discovered models are
tagged, aliases follow, and the default model is marked with `*`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | The first default location | Read this config. |
| `-v`, `--paths` | Off | Also show each model's GGUF path. |
| `--json` | Off | Emit JSON. |

Exit code 2 means no config was found or it failed to load. When none was
found, the message names `gmlx init`.

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

The positional argument is a path, or a model id or alias from your server config
when it is not a file. A config id supplies its path, sampling, system
prompt, template, adapter, drafter and streaming placement, although flags
you pass still win. An id with an unknown profile fails and lists the valid
ones.

These flags control generation:

| Flag | Default | Meaning |
|------|---------|---------|
| `gguf`, positional | Required | Load this GGUF, which may be sharded, or this config id. |
| `--prompt STR` | `Hello, world!` | Generate from this prompt. |
| `--prompt-file PATH` | None | Read the prompt from a file. |
| `--system-prompt STR` | None | Pass this system message to the chat template. |
| `--max-tokens N` | Until the model stops | Stop after this many tokens. |
| `--temp F` | Family default | Set the sampling temperature. `0` is greedy. |
| `--top-p F` | Family default | Set the nucleus probability. |
| `--top-k N` | Family default | Keep this many candidate tokens. `0` disables the filter. |
| `--min-p F` | Family default | Drop tokens less likely than this share of the best token. |
| `--repetition-penalty F` | `0` | Set the repetition penalty. `0` disables it. |
| `--repetition-context-size N` | `20` | The repetition penalty looks back over this many tokens. |
| `--presence-penalty F` | `0` | Set the presence penalty. |
| `--frequency-penalty F` | `0` | Set the frequency penalty. |
| `--xtc-probability F`, `--xtc-threshold F` | `0` | Set XTC sampling, which works on the text path only. |
| `--logit-bias JSON` | None | Add these biases to the logits, given as a map from token id to bias. |
| `--stop STR` | None | Stop at this sequence. Repeat the flag for more sequences. |
| `--seed N` | None | Seed the sampler. |
| `--reasoning {show,hide,raw}` | `show` | `show` styles the thinking and strips its markers, `hide` prints only the answer, and `raw` passes everything through. |
| `--thinking {on,off,adaptive}` | Template default | Turn reasoning on, off or adaptive through the model's template variable. |
| `--reasoning-effort LEVEL` | Template default | Set the reasoning depth on models that support levels. |
| `--thinking-budget N` | Unlimited | Cap reasoning tokens. |
| `--thinking-start-token STR`, `--thinking-end-token STR` | Detected | Set the model's reasoning markers when detection fails. |
| `--chat-template-config JSON` | None | Pass extra template variables, such as `'{"enable_thinking": false}'`. |
| `-v`, `--verbose` | Off | Print full load diagnostics instead of the spinner. |

These flags pick profiles and family defaults:

| Flag | Default | Meaning |
|------|---------|---------|
| `--profile NAME` | None | Apply a built-in intent or, with a config, a user profile, as `@NAME` on the positional argument does. |
| `--no-family-defaults` | Off | Do not apply the family's sampling defaults on a bare path. |
| `--config FILE` | The first default location | Resolve an id against this config. |

These flags control memory:

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-kv-size N` | None | Cap the KV cache with a rotating window. The flag combines with [kvarn](glossary.md#kvarn) but not with affine `--kv-bits`. |
| `--kv-bits N` | Off | Quantize the KV cache to 2, 3, 4, 6 or 8 bits affine, or to 2, 3, 4, 5, 6 or 8 under kvarn, where it defaults to 6. |
| `--kv-group-size N` | `64` | Set the affine quantization group size. |
| `--kv-quant-scheme {uniform,kvarn}` | `uniform` | Pick affine or `kvarn`, the variance-normalized quantization that [KV cache quantization](kv-quantization.md) describes. |
| `--kv-tail-tokens N` | `1024` | Under kvarn, the newest N tokens stay fp16. N is a multiple of 128, and `0` disables the tail. |
| `--quantized-kv-start N` | `0` | Keep the cache in fp16 until it holds this many tokens, then quantize all of it. The flag does not apply under kvarn. |
| `--prefill-step-size N` | `2048`, `8192` when streaming, `4096` for HY4 when streaming | Prefill in chunks of this many tokens. |
| `--dtype {auto,bfloat16,float16}` | `auto` | Set the activation width. `auto` picks float16 on M1 and M2. |

A `--kv-bits` value outside the scheme's list exits 2, and so does a
`--max-kv-size` window too small for kvarn's block layout.
[Settings that limit memory](memory.md#settings-that-limit-memory) gives
the smallest window that kvarn accepts, and
[KV cache quantization](kv-quantization.md) describes the models kvarn
declines. A declined model prints the reason and runs fp16
KV, and the VLM path under `--mmproj` declines kvarn the same way.

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

These flags control multimodal input. [Vision and audio](vlm.md) describes them:

| Flag | Default | Meaning |
|------|---------|---------|
| `--mmproj PATH` | None | Load this projector GGUF. |
| `--image PATH_OR_URL` | None | Prepend these comma-separated images. |
| `--audio PATH_OR_URL` | None | Prepend these comma-separated audio files. The model needs an audio tower. |
| `--resize-shape N_OR_WxH` | Model default | Resize images before encoding. |

Under `--mmproj`, the run ignores `--stop` and the XTC flags with a warning.
The bench and report flags and `--stream-cpu` exit with an error, while
`--stream-experts` still works.

These flags control speculative decoding, which
[Speculative decoding](speculative-decoding.md) describes:

| Flag | Default | Meaning |
|------|---------|---------|
| `--speculative`, `--mtp` | Auto for an MTP head or a DeepSeek-V4 companion, off under `--stream-experts` | Force speculation on. |
| `--no-speculative`, `--no-mtp` | Off | Force speculation off. |
| `--draft-gguf PATH` | None, or the companion beside a DeepSeek-V4 file | Draft with this separate drafter GGUF, which implies `--speculative`. |
| `--native-mtp` | Off | Draft with the model's own head, even when a drafter is set. The flag forces speculation on, and a GGUF with no head exits 2. |
| `--draft-block-size N` | Drafter default | Set the block size of each round, which drafts N-1 tokens and checks them in one N-token target pass. |
| `--stochastic-mtp` | Off | Accept sampled drafts by rejection sampling, which accepts more but is not token-identical. |

Speculation drops, with a warning, each flag it cannot honor, and `--no-mtp`
switches to plain decoding, which honors every flag.
[Settings that speculation drops](speculative-decoding.md#settings-that-speculation-drops)
lists those flags.

KV quantization works on the speculative path and quantizes the same layers
`serve` would. The exceptions are kvarn on a sliding-window stack and on an
architecture whose drafter reads the target KV. There kvarn declines, and
the `[kv]` line gives the reason.

These flags stream a model bigger than memory, which
[Models larger than memory](streaming.md) describes:

| Flag | Default | Meaning |
|------|---------|---------|
| `--stream-experts` | Off | Stream the routed experts from disk. Attention and the KV cache stay on GPU. |
| `--stream-cpu` | Off | Run the whole model on the CPU device from the page cache. |
| `--stream-fast-disk {auto,on,off}` | `auto` | Set the prefetch policy. `auto` measures the drive at load. |
| `--prefill-feeder`, `--no-prefill-feeder` | On | Stage expert prefill directly from the GGUF. |
| `--decode-feeder`, `--no-decode-feeder` | On under `--stream-experts` | Decode from a wired expert arena that keeps the experts the router picks most often. |
| `--gpu-keepwarm`, `--no-gpu-keepwarm` | On with the decode feeder | Keep GPU clocks high while a streamed model decodes, or turn that off. |
| `--moe-experts K` | Trained | Cap the router at K experts for each token, which is lossy. |
| `--moe-expert-mass P` | Off | Keep the smallest expert set covering share P of gate mass, which is lossy. |
| `--moe-expert-probe` | Off | Run lossless and print how many experts each token needed at candidate P values. |
| `--moe-miss-shed P` | Off | Drop experts that would miss the arena down to share P, which is lossy. |
| `--moe-layer-shed P` | Off | Skip a streamed layer's experts with probability P, which is lossy. |
| `--moe-prestage {ranked,keepers}` | `ranked` | `keepers` filters prestage predictions through the miss-shed policy, so it needs `--moe-miss-shed`. |

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

The command exits 0 on success, 1 when the file cannot load, 2 on a usage
or file error, and 130 when interrupted.

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

Slash commands in the chat change the system prompt, the sampling and
penalty settings, `--reasoning`, `--thinking` and `--thinking-budget`. The
other flags last for the whole chat.

These flags control generation:

| Flag | Default | Meaning |
|------|---------|---------|
| `--system-prompt STR` | None | Send this system message on the first turn and after each reset. |
| `--max-tokens N` | Until the model stops | Cap each reply at this many tokens. |
| `--temp F`, `--top-p F`, `--top-k N`, `--min-p F` | Family default | Set the sampling temperature and filters. |
| `--repetition-penalty F`, `--presence-penalty F`, `--frequency-penalty F` | `0` | Set the repetition, presence and frequency penalties. |
| `--repetition-context-size N` | `20` | The repetition penalty looks back over this many tokens. |
| `--xtc-probability F`, `--xtc-threshold F` | `0` | Set XTC sampling. |
| `--logit-bias JSON` | None | Add these biases to the logits, given as a map from token id to bias. |
| `--stop STR` | None | Stop at this sequence. Repeat the flag for more sequences. |
| `--seed N` | None | Seed the sampler. |
| `--reasoning {show,hide,raw}` | `show` | `show` styles the thinking and strips its markers, `hide` prints only the answer, and `raw` passes everything through. |
| `--thinking {on,off,adaptive}`, `--reasoning-effort LEVEL` | Template default | Set the reasoning switch and depth. |
| `--thinking-budget N` | Unlimited | Cap reasoning tokens. |
| `--thinking-start-token STR`, `--thinking-end-token STR` | Detected | Set the model's reasoning markers when detection fails. |
| `--chat-template-config JSON` | None | Pass extra template variables. |

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

A local load also takes these [`gmlx run`](#gmlx-run) flags, which mean
the same as they do there:

| Group | Flags shared with `run` |
|-------|-------------------------|
| Loading | `--arch`, `--hf-source`, `--chat-template`, `--no-chat-template`, `--no-remap`, `--no-zero-copy`, `--adapter` |
| Memory | `--max-kv-size`, `--kv-bits`, `--kv-group-size`, `--kv-quant-scheme`, `--kv-tail-tokens`, `--quantized-kv-start`, `--prefill-step-size`, `--dtype` |
| Multimodal | `--mmproj`, `--resize-shape` |
| Speculation | `--speculative`, `--mtp`, `--no-speculative`, `--no-mtp`, `--draft-gguf`, `--native-mtp`, `--draft-block-size`, `--stochastic-mtp` |
| Streaming | `--stream-experts`, `--stream-cpu`, `--stream-fast-disk` |
| Streaming feeders | `--prefill-feeder`, `--no-prefill-feeder`, `--decode-feeder`, `--no-decode-feeder`, `--gpu-keepwarm`, `--no-gpu-keepwarm` |
| Lossy streaming | `--moe-experts`, `--moe-expert-mass`, `--moe-expert-probe`, `--moe-miss-shed`, `--moe-layer-shed`, `--moe-prestage` |

When the chat is a server client, `--adapter`, `--mmproj`,
`--chat-template`, `--no-chat-template` and `--chat-template-config` exit
with code 2, and the other shared `run` flags do not apply. A base model with no
chat template refuses to start until you pass one with `--chat-template` or
send turns verbatim with `--no-chat-template`.

## gmlx launch

`gmlx launch` writes an external tool's configuration to point at a gmlx
server, starts the server if none is reachable, and runs the tool. It never
installs the tool. `opencode`, `hermes`, `aichat`, `elia` and `dsh` get a
configuration of their own under `~/.config/gmlx`. `pi`, `omp` and `goose`
get a provider merged into their own files, and `claude-code` and
`open-webui` get environment variables only. [Agents and chat apps](launch.md) describes each client.

```sh
gmlx launch opencode
gmlx launch pi --model qwen3.8-27b-ud-q6@coding
gmlx launch claude-code --model qwen3.8-27b-ud-q6
gmlx launch open-webui
gmlx launch dsh --model qwen3.8-27b-ud-q6
gmlx launch omp --config-only
```

These flags control `gmlx launch`:

| Flag | Default | Meaning |
|------|---------|---------|
| `client`, positional | None | Launch `claude-code`, `opencode`, `pi`, `omp`, `hermes`, `goose`, `aichat`, `elia`, `open-webui`, `dsh` or `menubar`. Without one, it prints the help. |
| `--model ID[@profile]` | The server's default | Point the tool at this served model, which the server keeps loaded through its idle timeout. |
| `--base-url URL` | None | Connect to this server, which is never auto-started. |
| `--host H`, `--port P` | The managed server | Select the server. |
| `--api-key KEY` | The config's `server.api_key` | Write this key to the tool's native config field. Without a key, tools that require one get the provider id. |
| `--provider-id NAME` | `gmlx` | Write this provider id into the tool's config. |
| `--config-path PATH` | The client's location | Write the tool config to this file or directory, depending on the client. [How a launch works](launch.md#how-a-launch-works) lists each one. |
| `--config-only` | Off | Write the config and print the run command without running it. |
| `--no-start` | Off | Never start a server. |
| `--start-timeout S` | `0`, no limit | Cap the auto-start wait. |
| `--no-keep` | Off | Do not keep `--model` resident. |
| `--dsh-profile NAME` | `gmlx` | Boot this dsh profile with the gmlx overlay, for dsh only, as [dsh](launch.md#dsh) describes. |

Once the tool starts, it replaces gmlx, so the exit status is the tool's
own. Before that, the command exits 0 after `--config-only` and 1 when the
server is unreachable, has died or has timed out, or when the tool is not
installed. It
exits 2 when the config is missing or malformed, and 130 when interrupted
during the start wait.

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
all its shards into your model directory as plain files. A file saved under
a `model_dirs` root is registered in the config immediately, and any running
server is signalled to reload the config.

```sh
gmlx pull hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
gmlx pull hf:org/repo/model.gguf --to ~/models
gmlx pull hf:org/gemma-3-27b-GGUF/gemma-3-27b-Q4_K_M.gguf mmproj-F16.gguf
```

These flags control `gmlx pull`:

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

Inside a `model_dirs` root, downloads nest under `<org>__<repo>/` so that a
model's siblings stay together. Before it downloads the first byte, `pull`
checks that the volume has space for every shard. It also notes, without
refusing, a model that will not fit this Mac's RAM.

A stalled or dropped read retries
with backoff from the `.part` file, and `GMLX_PULL_RETRIES` and
`GMLX_PULL_TIMEOUT` tune the retries. An interrupted `pull` resumes from
the `.part` file on the next run. A gated or private repo needs a token in
`HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`, or one stored by `hf auth login`.

## gmlx validate

`gmlx validate` reports whether a GGUF will load, from the header alone. A
remote reference is range-read, so the check reads a few megabytes rather
than the whole file. The report names the architecture, the quant codecs,
the total size across shards, whether it fits this Mac's RAM and, for a MoE
model, the streaming plan.

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

A split model is checked across all its shards, because a codec used by a
single tensor can appear only in a later shard. Projector GGUFs are
recognized as companions and are not checked as models.

The command exits 0 when the file will load or when it lists the quants
of a folder or repo, 1 when the file will not load, and 2 when the
reference cannot be resolved or read.

## gmlx rm

`gmlx rm` deletes a model's GGUF files, its partial-download files and its
companions, and removes the entry from the config. A file another model
still references is kept. Aliases to the removed id are dropped, and if the
removed id was the default model, the default is cleared. Before anything
is deleted, the plan is printed and confirmed.

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
| `--yes` | Off | Skip the confirmation. The flag is required without a terminal. |
| `--json` | Off | Emit the result as JSON. The flag needs `--yes`. |
| `--no-reload` | Off | Do not signal a running server to re-read the file. |

The command exits 0 when the model was removed, 1 when you declined or a
file could not be deleted, and 2 for an unknown id or a missing or invalid
config. It also exits 2 without `--yes` when there is no terminal or
`--json` is set.

## gmlx sync-models

`gmlx sync-models` rescans the model directories and updates the `models`
block to match disk. Existing entries keep their comments and edits, entries
whose file is gone are dropped, and new files are added, with a sibling
drafter pairing into the model it serves. Run it after adding files by hand
or pulling with `--no-register`. The scan descends into subdirectories by
default because `pull` nests its downloads.

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

An entry that could not be checked, because its root is unmounted or the
cache is unreadable, is kept and reported instead of dropped.

## gmlx ps

`gmlx ps` shows the models resident in a running server from its
`/v1/metrics` snapshot, with the id, size, idle time, TTL, pinned and kept
state, and path of each.

| Flag | Default | Meaning |
|------|---------|---------|
| `--url URL` | The managed server | Query the server at this base URL. |
| `--host H`, `--port P` | The managed server | Select the server. |
| `--api-key KEY` | The `GMLX_API_KEY` variable | Send this key to a keyed server. |
| `--json` | Off | Emit JSON. |

The command exits 0 when it listed the models, 1 when the server answered
with an error or is not gmlx, and 3 when no server was reachable.

## gmlx systemone

`gmlx systemone` sends a JSON file holding a `/v1/systemone` request body
to a running server and prints one line per question. The body is described in
[Structured decisions](decisions.md). With `--model` the verb loads the GGUF
itself and answers offline, with no server. DiffusionGemma answers with a
[structured read](glossary.md#structured-read), and any other text model
with the [letter readout](glossary.md#letter-readout).

```sh
gmlx systemone ticket.json
gmlx systemone ticket.json --model diffusiongemma-Q4_K_M.gguf
gmlx systemone ticket.json --model OpenJev-Q4_K_M.gguf
```

A yes or no answer prints as its probability, a choice as the chosen
option with its [confidence](decisions.md#reading-the-answers), and a
score as the expected level with its confidence. A question skipped by
`ask_if` prints `skipped`.

These flags control `gmlx systemone`:

| Flag | Default | Meaning |
|------|---------|---------|
| `REQUEST.json`, positional | Required | Send the request body in this file. |
| `--url URL` | The managed server | Query the server at this base URL. |
| `--host H`, `--port P` | The managed server | Select the server. |
| `--api-key KEY` | The `GMLX_API_KEY` variable | Send this key to a keyed server. |
| `--model GGUF` | None | Answer offline on this GGUF path or configured model id. It cannot be combined with `--url`, `--host` or `--port`. |
| `--config FILE` | The first default location | Take an offline run's model ids and `server.systemone` settings from this config. The flag needs `--model`. |
| `--seed N` | The request's `seed` | Replace the request's seed on both paths. |
| `--json` | Off | Print the whole response body as JSON. |

The command exits 0 when the request was answered, 2 on a usage error, and
1 when the file, the request or the model was refused or no server was
reachable.

## gmlx profiles

`gmlx profiles` prints the family sampling table with its intents, then the
config's user profiles and each model's family. With a model id it prints
that model's resolved sampling for its base and each intent, plus the layers
that produced it. Without a config it prints the family table alone.

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
speak, and the reply streams back as speech. It is a client of the server's
speech and chat endpoints, so the server needs `stt` and `tts` configured.
Setup, the config block and the in-session keys are in [Voice
chat](talk.md).

```sh
gmlx talk
gmlx talk qwen3 --voice bf_emma
gmlx talk --mode vad
gmlx talk --once
```

These flags control `gmlx talk`. Most of them override the key of the same
name in the [`talk` block](config.md#voice), and a flag's default applies
when neither is set.

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
as a GGUF adapter. The base stays quantized throughout, so a model that
does not fit in fp16 can still be fine-tuned. For the walkthrough, read
[LoRA adapters](lora.md).

```sh
gmlx train base-Q8_0.gguf --data ./my-data --adapter-out my-lora.gguf
gmlx run base-Q8_0.gguf --adapter my-lora.gguf --prompt "..."
```

These flags control `gmlx train`:

| Flag | Default | Meaning |
|------|---------|---------|
| `model`, positional | Required | Train on this base GGUF or config id. |
| `--data PATH_OR_ID` | Required | Train on a directory with `train.jsonl` and `valid.jsonl`, or on a Hugging Face dataset id. |
| `--adapter-out PATH` | Required | Write the adapter here. A module the adapter cannot hold is refused before the first step. |
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
| `--grad-checkpoint` | Off | Recompute each layer's activations in the backward pass, trading time for memory. It is refused with `--dropout` above 0 and on Kimi K3 and DeepSeek-V4.1. |

The data can be chat messages, prompt and completion pairs, or plain text,
in the formats mlx-lm's trainer accepts.

## gmlx distill

`gmlx distill` trains a LoRA adapter for a small GGUF on a larger one's
outputs in six actions plus one check. The teacher and the student may
use different tokenizers, and the walkthrough is [Distillation](distill.md).

- `gen` runs a teacher through `gmlx serve` over a prompt set and writes
  its replies as a corpus.
- `filter` drops the generated rows a student should not learn from.
- `cache` runs a teacher GGUF over a corpus once and stores its most
  likely next tokens and their log-probabilities at every position.
- `align` maps that cache onto a student tokenizer and writes a view, the
  positions and values the student trains to match.
- `train` fits a LoRA adapter on a GGUF student against the view.
- `eval` scores the student with and without the adapter.
- `census` compares two reply caches of the same replies, one made with a
  context the student never sees, and measures how much that context moves
  the teacher.

```sh
gmlx distill gen --teacher teacher-Q6_K.gguf --prompts prompts.jsonl --out replies.jsonl
gmlx distill filter --in replies.jsonl --out corpus.jsonl
gmlx distill cache --teacher teacher-Q6_K.gguf --corpus corpus.jsonl --frame reply --out cache/
gmlx distill align --cache cache/ --student student-Q4_K_M.gguf --out view/
gmlx distill train --view view/ --student student-Q4_K_M.gguf --adapter-out student-distill.gguf --iters 2000
gmlx distill eval --student student-Q4_K_M.gguf --adapter student-distill.gguf --before \
    --slice prose=heldout.txt --md eval.md --json eval.json
```

Every size flag is in decimal GB, 1e9 bytes. Each action checks its
output paths before any model loads. It exits 0 on success, and 2 on a
refused input or setting, a missing required flag or input file, or an
output path it cannot write. Some actions add codes of their own:

- `gen` exits 1 when some requests failed and their prompts remain to be
  rerun, and 2 when its server fails to start.
- `filter` exits 2 when its `--verify` command fails.
- `cache` exits 2 when a shard cannot be written, and keeps the verified
  shards. It exits 3 when its memory probe misses twice or a `--routes`
  recording does not match its rows, and 4 when the validator fails on
  what it wrote. `cache --validate` exits 1 on a problem.
- `align` exits 3 when the own-group check refuses the pair, and writes no
  view.

### distill gen

`gen` reads a prompt file with one `{"id", "messages", "context"}` object
per line, whose messages end on a user turn. A row's context, or the file given by
`--context`, goes in front of the last user turn for the teacher, and
either one must hold text. A row that took a context is written with the
teacher's list under `messages` and the prompt as given under
`student_messages`. A row without one carries `messages` alone.

Prompt ids already in the output are skipped, so a run resumes where it
stopped. A resume checks that each skipped id still names the prompt it
answered and refuses when one differs or is gone, since ids taken from
line numbers shift when a line is inserted. An interrupt cancels the
queued requests and stops the server, and the run ends when the requests
in flight have failed or returned.

Beside the output, `<out>.gen.json` holds the settings a resume must
match and is written before the first request. When a run ends, the
sidecar gains a `run` block with the reply and token totals read from the output
rows, the wall time summed over the runs that ended, and this run's
failed requests and aggregate token rate. An interrupted run leaves the
block as it found it, and a rerun that finds every prompt answered
writes the block from the rows when the sidecar has none.

A `--base-url` server that lists several models serves the run with the
one named like `--teacher`, and gen refuses when none or several match.

These flags control `gmlx distill gen`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--out PATH` | Required | Write the corpus jsonl here, with `<out>.gen.json` beside it. |
| `--prompts PATH` | None | Read prompt rows that end on a user turn from this jsonl. |
| `--corpus PATH_OR_ID` | None | Build continuation prompts from this text corpus instead of `--prompts`. |
| `--teacher GGUF` | None | Serve this GGUF for the run, the teacher or, for a measurement, the student. `--model` is the same flag. |
| `--base-url URL` | None | Use this running server's `/v1` base instead of serving `--teacher`. With `--thinking-budget` the close is sized for a server that runs a drafter. |
| `--host HOST` | `127.0.0.1` | Bind the served teacher to this host. |
| `--port N` | `8093` | Serve the teacher on this port. |
| `--text-key KEY` | `text` | With `--corpus`, read text from this column of a jsonl or dataset row. |
| `--hf-split NAME` | `train` | With `--corpus`, read this split of a Hugging Face dataset. |
| `--prefix-chars N` | `1500` | With `--corpus`, quote this many characters of each document in the user turn, cut at a space. |
| `--min-chars N` | `2000` | With `--corpus`, skip documents shorter than this. |
| `--docs N` | All | With `--corpus`, build this many prompts. |
| `--instruction TEXT` | `Continue the following text.` | With `--corpus`, place this user turn before the prefix. |
| `--chat-template-kwargs JSON` | None | Pass this JSON object of template variables to every teacher render, through serve's `--chat-template-config`. A thinking key is refused. |
| `--context FILE` | None | The teacher reads this text for every prompt without its own context field. A blank file is refused. |
| `--context-format FMT` | `{context}\n\n{prompt}` | Combine the context and the last user turn with this format, which must place both fields. |
| `--thinking` | Off | Turn thinking on and keep the reasoning trace as `reasoning_content` on the reply. Without it, gen turns thinking off. |
| `--thinking-budget N` | None | With `--thinking`, cap the reasoning trace at N tokens per request, and mark the replies it cut for `filter`. |
| `--tokenizer GGUF_OR_DIR` | `--teacher` | Count the reasoning trace against the budget with this tokenizer when `--base-url` is given. |
| `--serve-arg ARG` | None | Pass this argument to `gmlx serve`. The flag repeats, and a resume checks it. Flags that alter the prompt or thinking, and a drafter with a budget, are refused. |
| `--startup-timeout S` | `900` | Wait this many seconds for the served teacher. |
| `--concurrency N` | `8` | Keep this many requests in flight. |
| `--max-tokens N` | `1024` | Give each request this answer budget. With `--thinking-budget` the trace gets its own budget plus the forced close. Without it, the trace shares this budget. |
| `--temperature F` | `0.7` | Set the sampling temperature. |
| `--top-p F` | `0.9` | Keep the most likely tokens whose probabilities add to this. |
| `--top-k N` | The server's | Keep this many candidate tokens. |
| `--min-p F` | The server's | Drop tokens less likely than this share of the best token. |
| `--seed N` | `1` | Use this base seed, to which each request adds its prompt index. |
| `--timeout S` | `1800` | Fail a request after this many seconds. |
| `--report-every N` | `50` | Print a progress line every N replies. |

`--serve-arg` refuses `--thinking`, `--thinking-budget`, `--chat-template`,
`--chat-template-config`, `--reasoning-effort`, `--system-prompt` and
`--profile`, in any spelling serve accepts. Each changes what the teacher
is prompted with, and the rows would not record it. `--native-mtp`,
`--speculative` and `--draft-gguf` are refused beside `--thinking-budget`
too, because a server that runs a drafter does not hold each request to the
budget.

Set the thinking switch and budget with gen's own flags, template variables
with `--chat-template-kwargs`, sampling with gen's sampling flags, and a
system prompt as a system turn in the prompt rows. A template override has
no gen form, since `cache` renders the rows with the teacher's own
template.

### distill filter

`filter` runs its checks in a fixed order, and the first failure names the
reason. The reason is one of `length`, `budget`, `empty`, `marker`,
`repeat`, `ascii`, `tokens` and `verify`, each defined in
[Round one trains on the teacher's replies](distill.md#round-one-trains-on-the-teachers-replies)
in the distillation guide.

`--context` rebuilds every kept row with the context on the teacher's side
and the prompt as given under `student_messages`. The rebuilt rows prepare
a second round from replies a student wrote without the context. The flag
refuses a row that already carries `student_messages`, since that row was
generated with a context.

These flags control `gmlx distill filter`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--in PATH` | Required | Read this generated corpus jsonl. The flag repeats, and inputs join in order. Sidecars that disagree, inputs filtered differently or shared row ids are refused. |
| `--out PATH` | Required | Write the filtered corpus here, with `<out>.gen.json` beside it. |
| `--report JSON` | None | Write the kept and dropped counts here. |
| `--rejects PATH` | None | Write one `{id, reason}` line per dropped row here, with the checker's word under `detail`. |
| `--min-words N` | `16` | Drop replies whose answer, trace excluded, has fewer than this many units. A unit is a word or one ideograph or kana character. `--min-tokens` is the same flag. |
| `--ngram N` | `8` | The repetition check uses n-grams of this size, in the units of `--min-words`. |
| `--max-repeat F` | `0.2` | Drop replies whose repeated n-grams exceed this fraction. |
| `--max-trace-repeat F` | `0.5` | Drop replies whose reasoning trace's repeated n-grams exceed this fraction. |
| `--max-line-repeats N` | `2` | Drop replies with a line repeated more than this many times in a row, skipping lines without a letter or digit. |
| `--max-non-ascii F` | Off | Drop replies whose non-ASCII character fraction exceeds this. |
| `--max-reply-tokens N` | Off | Drop replies longer than this many tokens, reasoning trace included. |
| `--keep-budget-hit` | Off | Keep replies whose thinking budget cut the reasoning trace. |
| `--verify CMD` | None | Run this shell command as your checker, which reads the rows that passed the earlier checks as jsonl on stdin and prints `ok` or a reason word per row. |
| `--context FILE` | None | Put this text on the teacher's side of every kept row. A blank file is refused. |
| `--context-format FMT` | `{context}\n\n{prompt}` | Combine the context and the last user turn with this format, which must place both fields. |

### distill cache

`cache` runs the teacher pass. `--messages-key` picks which list of a row
the teacher reads, and `--student-messages-key` only names the list the
student's render reads later, in `align` and `eval`.

Reply and reply-think rows whose final turn has no content, such as a
tool call, have nothing to target and are dropped, counted in the `[cache] frame` line. That line also counts the
reply-think rows whose reasoning trace the teacher's template does not
render, which train on the reply alone, and a reply-think pass in which
no row keeps its trace is refused.

A corpus written by `gen` renders with the thinking switch and the
`--chat-template-kwargs` its `.gen.json` sidecar records, mapped onto the
variables the teacher's template reads. `--frame-kwargs` adds to them,
and a value that contradicts one is refused. A template that prints the
date, as Llama 3's and gpt-oss's do, renders the day the cache was first
started, and a resume and the student's render in `align` keep that day.

These flags control `gmlx distill cache`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--teacher GGUF` | Required unless `--validate` | Run this teacher GGUF, which may be sharded. |
| `--corpus PATH_OR_ID` | Required unless `--validate` | Read this jsonl file, directory of text files or Hugging Face dataset id, `id[@config]`. |
| `--out DIR` | Required unless `--validate` | Write the cache to this directory. |
| `--validate DIR` | None | Validate an existing cache and exit without loading a teacher. |
| `--top-k N` | `256` | Keep this many log-probabilities per position. |
| `--max-len N` | `2048` | Fill each window with this many teacher tokens, start token included, at least 2. The continue frame and closing tail count toward it and must leave at least 8. |
| `--max-disk-gb F` | None | Refuse when the size estimate exceeds this. |
| `--cache-limit-gb F` | `8.0` | Cap the MLX buffer cache at this many GB during the pass. |
| `--logits-cap-gb F` | `4.0` | Size the head sub-chunk to fit this memory cap. |
| `--floor` | Off | Also store `floor_kld`, the KL against the f16-rounded top-k. |
| `--rows-per-shard N` | `64` | Write this many rows per shard file. |
| `--trunk N` | `512`, or `8192` streaming | Run the trunk in chunks of this many tokens, with rows stacked on the batch axis. |
| `--resume` | Off | Continue after the last verified shard, refused when the corpus, teacher, template, HF source or row options changed. A finished cache is validated, not redone. |
| `--max-rows N` | None | Stop after this many rows. |
| `--max-tokens N` | None | Stop after this many teacher tokens. |
| `--limit-docs N` | None | Read at most this many documents. |
| `--text-key KEY` | `text` | Read text from this column of a jsonl or dataset row. |
| `--hf-split NAME` | `train` | Read this split of a Hugging Face dataset id. |
| `--source TAG` | `human`, or `synthetic` with a generator sidecar | Write this source tag on every row. |
| `--frame KIND` | `none` | Place the targets in the chat template by frame, `none`, `continue`, `chat`, `reply` or `reply-think`. `reply-think` starts at the final turn's reasoning trace. |
| `--per-turn` | Off | With the chat or reply frame, write one reply row per assistant turn. |
| `--student-messages-key KEY` | `student_messages` | Name the corpus key that holds the student's own message list on reply rows. |
| `--frame-instruction TEXT` | `Continue the following text.` | Use this user turn for the continue frame. |
| `--messages-key KEY` | `messages` | Read the conversation from this column for the chat and reply frames. |
| `--close-final-windows` | Off | With the continue frame, close the last window of a document with the turn-end marker. |
| `--frame-kwargs JSON` | None | Pass these chat-template kwargs, an object or a file, to every teacher render, beside those a `gen` sidecar records. |
| `--hf-source ID` | None | Replace the config synthesized from the GGUF with this Hugging Face repo's config.json. The tokenizer always comes from the GGUF. |
| `--no-require-feeder` | Off | Run a streaming teacher without the prefill feeder. |
| `--no-wired-limit` | Off | Leave the wired limit where it is for a teacher that fits in memory. |
| `--stream-experts` | Off | Force expert streaming on a MoE teacher that would fit in memory. |
| `--expert-bytes-gb F` | The streamed expert bytes | Report this many expert bytes read per forward pass in the read-traffic report. Use `0` for a resident teacher. |
| `--routes` | Off | On a MoE teacher, store every layer's top-k expert ids per position for replay by `eval`. A gate that cannot replay is refused. |
| `--hidden` | Off | Also store a seeded random sketch of the teacher's final hidden state per position, for `train --hs`. |
| `--hidden-dim N` | `256` | Set the width of the hidden sketch. |
| `--hidden-seed N` | `1` | Seed the sketch matrix. |
| `--cpu` | Off | Run on the CPU device. |

### distill align

`align` maps a teacher cache onto the student's tokenizer and writes the
view that `train` reads.

These flags control `gmlx distill align`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--cache DIR` | Required | Read this cache directory. |
| `--student GGUF_OR_DIR` | Required | Align to this student GGUF, or to an MLX checkpoint directory for its tokenizer. |
| `--out DIR` | Required | Write the view to this directory. An earlier view there is replaced once every check has passed. |
| `--tables DIR` | None | Reuse the tokenizer tables of this earlier view directory when the pair matches. |
| `--kprime N` | The maximum seen | Keep at most this many distinct student-token groups per boundary. The identity path ignores it, since K' = K there. |
| `--materialize` | Off | Also write the batch tensors as view shards. |
| `--max-disk-gb F` | None | Refuse to materialize past this size. |
| `--force` | Off | Keep a view the own-group check would refuse. |
| `--val-fraction F` | `0.02` | Hold this fraction of rows for validation, whole documents at a time. A cache of two or more rows holds at least one. |
| `--seed N` | `1` | Seed the validation split. |
| `--w-mid F` | `0.5` | Weight an intra-word shared boundary by this much. |
| `--gamma F` | `0.001` | In the chunk term (ALM), drop chunks whose teacher boundary mass is under this positive value. |
| `--tau-alm F` | `1.0` | Set the positive temperature of the chunk term (ALM). |
| `--T-dk F` | `1.0` | Set the positive temperature that the KL term's group softmaxes use, under every `--loss` form. |
| `--max-chunk-len N` | `8` | Cap ALM chunks at this many tokens on either side, at least 1. |
| `--frame-kwargs JSON` | None | Pass these chat-template kwargs to every student render and store them in the view, over those the cache recorded and its `gen` thinking switch. |
| `--cpu` | Off | Run on the CPU device. |

### distill train

`train` fits the LoRA adapter against one or more views and saves
checkpoints as it goes.

These flags control `gmlx distill train`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--view DIR` | Required | Train on this view directory. Repeat the flag to mix views aligned alike over one tokenizer pair. |
| `--student GGUF` | Required | Train this student GGUF, which may be sharded. |
| `--adapter-out PATH` | Required | Write the GGUF adapter here. An unwritable path is refused before the load, and a module the adapter cannot hold before the first step. |
| `--iters N` | Required | Train for this many steps. |
| `--lora-rank N` | `16` | Set the LoRA rank. |
| `--lora-scale F` | `2.0` | Apply this nonzero LoRA multiplier directly. |
| `--lora-alpha F` | None | Set the nonzero LoRA multiplier as alpha over rank, instead of `--lora-scale`. |
| `--lora-dropout F` | `0.0` | Set the LoRA dropout, below 1, with one mask per step that `--grad-checkpoint` replays. |
| `--grad-checkpoint` | Off | Recompute each layer's activations in the backward pass. It is refused on Kimi K3 and DeepSeek-V4.1. |
| `--lr F` | `1e-4` | Set the peak learning rate. |
| `--batch-size N` | `8` | Train on this many rows per step. |
| `--warmup F` | `0.05` | Warm up for this fraction of the steps, at least one step and never the last, then decay by cosine. `0` starts at the peak rate. |
| `--weight-decay F` | `0` | Set the AdamW weight decay. |
| `--clip F` | `1.0` | Clip the gradient norm at this value. `0` turns clipping off. |
| `--seed N` | `1` | Seed the data order and the LoRA init. |
| `--loss MODE` | `bucketed` | Pick the sparse KL variant, `bucketed`, `paper` or `renorm`. |
| `--dk F` | `1` | Weight the bucketed KL term by this much. |
| `--alm F` | `1`, `0` when `align` took the identity path | Weight the chunk term (ALM) by this much. |
| `--ce F` | `0` | Weight the cross-entropy term by this much. |
| `--T-dk F` | The view's | Override the view's T_dk. |
| `--tau-alm F` | The view's | Override the view's tau_alm. |
| `--gamma F` | The view's | Override the view's gamma, refused when it differs on a materialized view (its chunks are cut by `align`). |
| `--chunk N` | `512` | Run the head in chunks of this many positions. |
| `--hs F` | `0` | Weight the hidden-state term, a learned map from the student's final hidden state to the cache's sketch at every boundary. |
| `--hs-loss MODE` | `cosine` | Compare hidden states by `cosine`, or by `mse` on unit vectors. |
| `--ckpt-dir DIR` | `./ckpt` | Write checkpoints to this directory. A fresh run refuses one that holds an earlier run's checkpoints. |
| `--resume` | Off | Resume from `--ckpt-dir`, refused when none exists or when the views, student, training settings or gmlx's validation leave-out rule differ from that run. |
| `--save-every N` | `200` | Save a checkpoint every N steps. |
| `--val-every N` | `200` | Validate every N steps. |
| `--val-batches N` | `16` | Score this many validation batches per pass, from one seeded draw across the val rows of every view. |
| `--report-every N` | `10` | Report the train loss every N steps. |
| `--report JSON` | None | Write the run log here. |
| `--hf-source ID` | None | Replace the config synthesized from the GGUF with this Hugging Face repo's config.json. The tokenizer always comes from the GGUF. |
| `--no-wired-limit` | Off | Leave the wired limit where it is. |
| `--cache-limit-gb F` | `8.0` | Cap the MLX buffer cache at this many GB. |
| `--cpu` | Off | Run on the CPU device. |

### distill eval

`eval` scores the student on held-out text, tasks and chat sets, with or
without the adapter. Its task files are jsonl. `arc_easy.jsonl` and
`hellaswag.jsonl` hold `{id, query, choices, gold}` rows, `gsm8k.jsonl`
holds `{id, question, answer}` rows, and `gsm8k_shots.jsonl` holds the
worked examples shown before each question.

These flags control `gmlx distill eval`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--student GGUF` | Required | Score this student GGUF. |
| `--adapter GGUF` | None | Apply this GGUF adapter. |
| `--md PATH` | Required | Write the Markdown report here. |
| `--json PATH` | Required | Write the JSON report here. |
| `--cache DIR` | None | Check the slices for overlap against this cache's corpus. |
| `--slice NAME=PATH` | None | Score this held-out text slice. Repeat the flag for more slices. |
| `--teacher-bpb JSON` | None | Show these teacher bits per byte beside the student's, from a `{slice: bpb}` map or an earlier eval report. |
| `--tasks-dir DIR` | `.` | Read the four task files from this directory. |
| `--tasks LIST` | None | Run these comma-separated tasks, from `arc_easy`, `hellaswag` and `gsm8k`. |
| `--task-limit N` | All | Score this many items per task. |
| `--gsm8k-max-tokens N` | `384` | Give each GSM8K item this generation budget. |
| `--before` | Off | Also score with the adapter disabled in process. The flag needs `--adapter`. |
| `--chat-slice NAME=PATH` | None | Score this jsonl of `{messages}` conversations on their assistant turns, `student_messages` first. The flag repeats. |
| `--chat-sanity PATH` | None | Score this jsonl of `{id, messages, kind}` prompts, where `kind` is `task` or `refuse`, for template compliance and drift from an earlier report's replies. |
| `--chat-max-tokens N` | `256` | Give each chat sanity reply this token budget. |
| `--chat-refs JSON` | None | Anchor the drift score on the replies of this earlier eval report. `--before` overrides it and anchors on the adapter-off replies. |
| `--chat-max-len N` | `2048` | Score chat and reply rows of up to this many student tokens. A longer row loses turns until it fits, or is dropped. |
| `--chat-per-turn` | Off | Score every assistant turn as its own row. |
| `--reply-slice NAME=PATH` | None | Score this jsonl of conversations on the final reply. The flag repeats. |
| `--reply-think` | Off | Reply slices target the final turn from its reasoning trace onward. |
| `--reply-positions JSON` | None | Restrict the reply slices to this census JSON's `high_delta` maps. A file naming none of their rows, or with a frame other than `--reply-think`'s, is refused. |
| `--kld-cache DIR` | None | Score sparse KL against this same-vocabulary cache, which is refused on another tokenizer or a cached id beyond the student's head. |
| `--kld-rows N` | All | Score this many rows of the KL cache, spread over its length order. |
| `--frame-kwargs JSON` | None | Pass these chat-template kwargs to every render. |
| `--max-len N` | `512` | Measure bits per byte in windows of this many tokens. |
| `--bpb-prefix TEXT` | None | Place this text before every window (`\n`, `\t`, `\r` and `\\` decoded), or `@continue` or `@model` for that frame's template prefix. Another `@` exits 2. |
| `--batch-size N` | `8` | Score this many windows per batch. |
| `--cache-limit-gb F` | `4.0` | Cap the MLX buffer cache at this many GB. |
| `--decontam-threshold F` | `0.01` | Void a slice's gate when more than this fraction of its windows is found in the corpus. |
| `--hf-source ID` | None | Replace the config synthesized from the GGUF with this Hugging Face repo's config.json. The tokenizer always comes from the GGUF. |
| `--cpu` | Off | Run on the CPU device. |

### distill census

`census` pairs the reply rows of a cache made without a context with the
same rows in one or more caches made with one, and runs on the CPU. It
reports how much more likely the context makes each token the teacher
wrote, and the distance between the two stored top-k distributions with
everything outside the top-k pooled. With several contexts it also reports
the part no single adapter can learn. Every `--with` cache then decides
which rows pair and which positions count, while the effect, the histogram
and the positions map come from the first `--with` cache.

The action exits 2 when a cache has no manifest
or one it cannot read, or when no rows pair. It also exits 2 when a
`--with` cache was made with another teacher, tokenizer, top-k or head
width than `--without`, and when a reply-think cache records no
`content_start`, which older caches lack. A `--corpus`
that names no file, or holds a line that is not a JSON object, exits 2 as
well.

These flags control `gmlx distill census`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--without DIR` | Required | Read the same replies without the context from this cache. |
| `--with DIR` | Required | Read a cache made with a context. Repeat the flag for more contexts. |
| `--out JSON` | Required | Write the census JSON here. |
| `--md PATH` | None | Write a Markdown summary here. |
| `--corpus JSONL` | None | Key `high_delta` by row id, using the corpus jsonl the caches were made from. |
| `--delta-threshold F` | `1.0` | A position is high-delta when the context adds this many nats at the token the teacher wrote. |
| `--pair-by MODE` | `line` | Pair rows across caches by corpus `line` or by the full `doc` id. |
| `--max-rows N` | All | Measure this many paired rows. |

## gmlx doctor

`gmlx doctor` checks what a working setup needs and prints a PASS, WARN,
FAIL or SKIP line for each check, with the fix named. No check accesses the
network. The checks cover the macOS version, the runtime and kernels, the
config, and the files of each configured model and service. They also cover background
servers, the login items and the launcher that background starts use,
optional extras, ffmpeg, MCP tools, and assistants served on a non-loopback
address. The last checks
are the Hugging Face token, RAM against each model's size, and disk space.

```sh
gmlx doctor
gmlx doctor --deep
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | The first default location | Read this config. |
| `--deep` | Off | Also read each configured model's header. |
| `--json` | Off | Emit JSON. |

The command exits 0 when no check failed, 1 when a check failed, and 2 on a
usage error or a `--config` file that does not exist.

## gmlx completion

`gmlx completion` prints a completion script for zsh, bash or fish. The
script is a shim that asks the installed `gmlx` for candidates on each
tab. It completes verbs, each verb's flags, model ids from your config and
client names for `launch`, plus the host, port and URL of servers you have
backgrounded. The script needs no regeneration after an upgrade.

```sh
eval "$(gmlx completion zsh)"      # ~/.zshrc
eval "$(gmlx completion bash)"     # ~/.bashrc
gmlx completion fish | source      # ~/.config/fish/config.fish
```

| Flag | Default | Meaning |
|------|---------|---------|
| `shell`, positional | None | Print the script for `zsh`, `bash` or `fish`. Without a shell, it prints the help with the install lines. |
