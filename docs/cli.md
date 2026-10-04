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
gmlx init --models-dir ~/models -r --out ~/configs/studio.yaml
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
gmlx serve --config ~/configs/studio.yaml
gmlx serve --models-dir ~/models --recursive
gmlx serve model-Q4_K_M.gguf                # one model, id from the filename
gmlx serve model.gguf --mmproj mmproj.gguf  # one vision model
```

A bare `gmlx serve` needs a config in a
[default location](config.md#where-gmlx-looks). Without one, it exits with
status 2 and says to run `gmlx init` or to name a GGUF.

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

After the servers, it prints a line for each
[container session](container-sessions.md#see-and-stop-sessions), except
with `--json`. `gmlx launch --list` shows more. The command exits 0 when a
server is running and 3 when none is.

## gmlx restart

`gmlx restart` stops the server and relaunches it with the arguments
recorded in its runfile, from any directory. Before it stops the server, it
loads the server's config file and checks that the GGUF and the `--mmproj`,
`--draft-gguf` and `--adapter` files on its command line still exist.

When one is gone or the config does not load, restart prints the error,
leaves the server running, and exits with status 1. Fix the file and run
`gmlx restart` again.

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
| `uninstall` | `--host H`, `--port P` | Unload and remove the headless agent of the port and the menu bar's login item. |

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
installs the tool on the Mac, and [container mode](launch-container.md)
installs it in the container's image.

[Agents and chat apps](launch.md) describes each client, and
`gmlx launch CLIENT --help` ends with its install command. A
[custom agent](launch-agents.md) launches by its name and always runs in a
container. `gmlx launch --help` lists your agents.

```sh
gmlx launch opencode
gmlx launch pi --model qwen3.8-27b-ud-q6@coding
gmlx launch claude-code --model qwen3.8-27b-ud-q6
gmlx launch claude-code --container --model qwen3.8-27b-ud-q6
gmlx launch open-webui
gmlx launch dsh --model qwen3.8-27b-ud-q6
gmlx launch omp --config-only
gmlx launch claude-code -- --continue
```

These flags control `gmlx launch`, which refuses an abbreviated flag such
as `--cont`:

| Flag | Default | Meaning |
|------|---------|---------|
| `client`, positional | None | Launch `claude-code`, `opencode`, `pi`, `omp`, `hermes`, `goose`, `aichat`, `elia`, `open-webui`, `dsh`, `menubar` or a [custom agent](launch-agents.md). |
| `--model ID[@profile]` | An agent's `model`, else the server's default | Point the tool at this served model, which the server keeps loaded through its idle timeout. |
| `--base-url URL` | None | Connect to this server, which is never auto-started. |
| `--host H`, `--port P` | The managed server | Select the server. |
| `--api-key KEY` | The running server's `server.api_key` | Write this key to the tool's native config field. Without a key, tools that require one get the provider id. |
| `--provider-id NAME` | `gmlx` | Write this provider id into the tool's config. |
| `--config-path PATH` | The client's location | Write the tool config to this file or directory, as [How a launch works](launch.md#how-a-launch-works) lists. hermes and container mode refuse it. |
| `--config-only` | Off | Write the config and print the run command without running it. In container mode, print the `container run` command. |
| `--no-start` | Off | Never start a server. |
| `--start-timeout S` | `0`, no limit | Cap the auto-start wait. |
| `--no-keep` | Off | Let `--model`, or an agent's `model` setting, unload while idle. |
| `--dsh-profile NAME` | `gmlx` | Boot this dsh profile with the gmlx overlay, for dsh only, as [dsh](launch.md#dsh) describes. |
| `--container`, `--no-container` | The config's `enabled` | Run the client in an Apple container, or on the Mac, as [Container mode](launch-container.md) describes. |
| `--mount PATH[:DST][:ro]` | None | Share another folder with the container, in addition to the configured [mounts](config.md#launchcontainermounts). Repeatable. |
| `--mount-cwd`, `--no-mount-cwd` | The config's [`mount_cwd`](config.md#launchcontainermount_cwd) | Share the current folder with the container, or not. |
| `--image REF` | The configured image | Run this image in the container, as [A ready-made image](container-images.md#a-ready-made-image) describes. |
| `--rebuild` | Off | Rebuild the client's image, or pull an `image:` reference again. |
| `--reseed` | Off | Copy each [seed](config.md#launchcontainerclientsseed) into the [private home](glossary.md#private-home) again, over its old copy. A dry run only names them. |
| `--seed-instructions` | Off | Also seed the instruction and skill files of the client, as [Instructions and skills](container-access.md#instructions-and-skills) lists. |
| `--network {default,none}` | The config's [`network`](config.md#launchcontainernetwork) | Set the container's network for this launch. |
| `--shell` | Off | Open a shell instead of the client, in the project's running session if any. See [Container sessions](container-sessions.md#open-a-shell-in-the-container). |
| `--remove-home` | Off | Ask, then remove the project's private home and the volumes only it uses, free its browser port and start nothing. `--mount` and `--mount-cwd` pick the project. |
| `--detach` | Off | Start the session of Open WebUI, a dsh web profile or a custom agent in the background, and return once it runs. |
| `--stop` | Off | End the session of the project that `--mount` and `--mount-cwd` pick, or the one a launch from here would join, and start nothing. |
| `--list` | Off | List the sessions of every client and agent, or of the one named, and start nothing. |
| `--forget-share PATH` | None | Remove `PATH` and each folder in it from the [share history](container-security.md#the-share-history), and start nothing. |
| `-- ARGS` | None | Pass the arguments after `--` to the client, after the arguments `launch` adds. |

The flags from `--mount` to `--stop` work only in a container, and each
turns on container mode by itself. `--detach`, `--stop`, `--list` and
`--remove-home` go one at a time, and `launch` says when a combination does
not fit. [Container sessions](container-sessions.md) covers `--detach`,
`--stop`, `--list` and `--shell`.

### Exit codes

Once the client runs, `gmlx launch` exits with the client's own status,
also in a container. With `--detach`, it exits 0 once the session runs.

Before the client runs, `launch` exits with one of these codes, which follow
sysexits(3) where one fits:

| Code | Meaning |
|------|---------|
| 0 | `--config-only`, `--remove-home`, `--stop`, `--list` or `--forget-share` did its work or found nothing to do, or launch printed its help. |
| 1 | Launch refused for a reason no other code covers, such as a folder it will not share or a flag that does not fit this launch. |
| 2 | A flag is unknown or abbreviated, a pair of flags is refused, `--detach`, `--stop` or `--remove-home` names no client, or `--forget-share` names one. |
| 69 | Something launch needs is missing or does not answer, such as Apple container or its Linux kernel, the client on the Mac, the server, or a model. |
| 75 | Something is busy, such as a project that another launch uses, a session that outlasts `--stop`, a volume or port in use, or a server that starts or is full. |
| 78 | No gmlx config exists, or the config or its [`launch`](config.md#launch) block does not load. `--list` then still lists the sessions and exits 0. |
| 125 | The container could not start its connections to the Mac, such as when a program in the image already uses a port that launch forwards. |
| 126 | The session found the client's command in the image but cannot run it. |
| 127 | The session found no client command, or no shell for `--shell`, in the image. |
| 130 | Ctrl-C arrived before the client started, such as at the kernel question, during the kernel download or image build, or while `--detach` waited. |
| 128 + N | Signal N arrived after launch began to set up the session and before the client started, or ended the launch that `--detach` started. |

Each of these codes, apart from 130, comes with a message that names the
cause and the next step. A script can retry after 75 and should report the
message for any other code.

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

`gmlx distill` trains a LoRA adapter for a small GGUF on a larger model's
outputs. The walkthrough is [Distillation](distill.md), and every action's
flags are in the [Distillation reference](distill-reference.md).

## gmlx doctor

`gmlx doctor` checks what a working setup needs and prints a PASS, WARN,
FAIL or SKIP line for each check, with the fix named. No check accesses the
network.

The checks cover the macOS version, the runtime and kernels, the config,
and the files of each configured model and service. They also cover
background servers, the login items and the launcher that background starts
use, the Apple container service, and the disk space container mode takes.

Later checks cover optional extras, ffmpeg, MCP tools, and assistants served
on a non-loopback address. The last checks are the Hugging Face token, RAM
against each model's size, and disk space.

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
backgrounded.

```sh
eval "$(gmlx completion zsh)"      # ~/.zshrc
eval "$(gmlx completion bash)"     # ~/.bashrc
gmlx completion fish | source      # ~/.config/fish/config.fish
```

With these lines, the shell loads the script of the installed gmlx at each
start, so it needs no update after an upgrade. A script saved to a file
needs writing again after an upgrade.

| Flag | Default | Meaning |
|------|---------|---------|
| `shell`, positional | None | Print the script for `zsh`, `bash` or `fish`. Without a shell, it prints the help with the install lines. |
