# CLI reference

The `gmlx` command has one verb per task, and each verb's section gives
its flags, defaults and exit codes. A section says what a flag does, while
the guides it links explain when to use it.

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
| [`gmlx systemone`](#gmlx-systemone) | Answer a structured-decision request with a DiffusionGemma model. |
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
[family defaults](family-defaults.md).

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

| Flag | Default | Meaning |
|------|---------|---------|
| `--models-dir DIR` | Required unless `--from-hf-cache` | Directory to scan, repeatable. |
| `--from-hf-cache`, `--hf-cache` | Off | Also scan the local Hugging Face cache and write portable `hf:` entries. |
| `-r`, `--recursive`, `--no-recursive` | Shallow | Descend into subdirectories. |
| `--out FILE` | `~/.config/gmlx/gmlx.yaml` | Where to write. |
| `--force` | Off | Overwrite an existing file. |
| `-i`, `--interactive` | On a terminal | Run the wizard even with flags, which pre-fill its answers. |
| `--no-interactive` | Off | Never run the wizard. |
| `--disk-cache [GB]` | Off | Enable the on-disk prompt cache, capped for each model. Bare is 50 GB. |
| `--with-stt [MODEL]` | Off | Configure speech-to-text. Bare is `whisper-turbo`. |
| `--with-tts [MODEL]` | Off | Configure text-to-speech. Bare is `kokoro`. |
| `--with-embeddings [MODEL]` | Off | Configure embeddings. Bare is `qwen3-embed-0.6b`. |
| `--with-rerank [MODEL]` | Off | Configure reranking. Bare is `qwen3-rerank-0.6b`. |
| `--install`, `--no-install` | Ask | Install the extras the chosen services need, or never offer to. |
| `--default-model ID` | None | The model used when a request omits one. |
| `--port N` | `8080` | The port to write. |
| `--idle-ttl SECONDS` | `900` | Idle seconds before a model unloads. `none` keeps models resident. |
| `--request-timeout DURATION` | Unset, and the server applies `30m` | Fail the request when no token arrives for this long, such as `10m` or `1h`. `none` waits forever. |
| `--no-reload` | Off | Do not signal a running server to re-read the file. |

Auto-named ids carry the quant in compact form, such as `qwen3-0.6b-q4`, and
fall back to the full codec when two quants would collide. An empty directory
is accepted and produces a valid config with no models. When a server is
already running the config you rewrote, `init` signals it to reload. The
walkthrough is in the [Quickstart](quickstart.md#serving-models)
and the file it writes is described in [Configuration](config.md).

## gmlx serve

`gmlx serve` runs the server. It detaches by default and returns at once, so
the same shell can run `gmlx launch` next. `--foreground` keeps it attached
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
| `model`, positional | None | A GGUF to serve, pinned, with the id derived from the filename. |
| `--config FILE` | The first default location | Serve a YAML config. |
| `--models-dir DIR` | None | Serve a scan of a directory, repeatable. |
| `-r`, `--recursive`, `--no-recursive` | Shallow | Descend when scanning. |
| `--hf-cache`, `--from-hf-cache` | Off | Let Hugging Face ids resolve from the local cache, never the network. |
| `--print-config` | Off | Print the resolved config as YAML and exit. |

These flags control the process and its lifecycle:

| Flag | Default | Meaning |
|------|---------|---------|
| `--host ADDR` | Config or `127.0.0.1` | Bind address. A non-loopback bind needs `server.api_key` or `--no-auth`. |
| `--port N` | Config or `8080` | Bind port. |
| `--no-auth` | Off | Allow a non-loopback bind with no key, for auth handled by a proxy in front of the server. |
| `-f`, `--foreground` | Off | Stay attached to the terminal. |
| `--no-menubar` | Off | Do not raise the menu bar app. |
| `--log FILE` | `~/.cache/gmlx/server-<host>-<port>.log` | The background log. Each start rotates the last one to `.1`. |
| `--log-level LEVEL` | `info` | `critical`, `error`, `warning`, `info`, `debug` or `trace`. |
| `--start-timeout S` | `40` | Seconds a background start waits for readiness before returning. |

These flags set memory and scheduling. Most are also `server` keys in the
config, where the same settings apply to a config-mode server:

| Flag | Default | Meaning |
|------|---------|---------|
| `--budget-gb F` | 0.8x the GPU working set | [Resident](glossary.md#resident) weight budget across all models. |
| `--max-models N` | None | Cap on resident models. |
| `--pin ID_OR_PATH` | None | Never evict this model, repeatable. |
| `--max-tokens N` | None | Default completion cap. |
| `--no-family-defaults` | Off | Do not seed each family's model-card sampling under profiles and requests. In config mode a reload restores `server.family_defaults`. |
| `--prefill-step-size N` | `2048` | Prefill chunk size in tokens. Lower caps peak memory. |
| `--dtype {auto,bfloat16,float16}` | `auto` | Activation width. `auto` is float16 on M1 and M2. |
| `--decode-prefill-ratio R` | `auto` | GPU-time share prefill gets while streams decode. `0` is stock scheduling. |
| `--prefill-tick-ms MS` | `500` | Wall-clock budget for each prefill chunk while streams decode. `0` never halves. |
| `--ignore-eos` | Off | Decode each request to `max_tokens`, for throughput benchmarks. |

These settings apply to a positional GGUF only. In config mode the same
things are per-model keys under [models](config.md#models):

| Flag | Default | Meaning |
|------|---------|---------|
| `--mmproj PATH` | None | The projector GGUF that makes the model multimodal. |
| `--hf-source REPO` | None | Processor and config override for a vision model, rarely needed. |
| `--adapter PATH` | None | A GGUF LoRA adapter applied at load, text only. |
| `--chat-template STR_OR_PATH` | The GGUF's | Inline Jinja or a `.jinja` or `.txt` file. |
| `--thinking {on,off,adaptive}` | Template default | The reasoning switch, mapped to the model's template variable. |
| `--thinking-budget N` | Unlimited | Cap reasoning tokens for each request. `0` closes thinking at once. |
| `--reasoning-effort LEVEL` | Template default | The reasoning level for models whose template grades thinking, such as `low`, `medium` or `high`. |
| `--profile NAME` | None | A built-in intent such as `coding` or `reasoning-high`, resolved for the model's family. `gmlx profiles` lists them. |
| `--system-prompt STR` | None | A system prompt used when the request has none. |
| `--chat-template-config JSON` | None | Extra chat-template variables, a JSON object passed through verbatim. |
| `--kv-bits N` | Off | Quantize the KV cache to N bits. 2, 3, 4, 6 or 8 affine, or 2, 3, 4, 5, 6 or 8 under kvarn. |
| `--kv-group-size N` | `64` | Affine quantization group size. |
| `--kv-quant-scheme {uniform,kvarn}` | `uniform` | Affine or [kvarn](glossary.md#kvarn). Under kvarn `--kv-bits` defaults to 6. |
| `--kv-tail-tokens N` | `1024` | Under kvarn, the newest tokens kept fp16. A multiple of 128. |
| `--max-kv-size N` | None | Cap the request context budget at N tokens. |
| `--quantized-kv-start N` | `0` | Tokens kept unquantized at the start of the cache. Not applied under kvarn. |

The KV flags are the [`load` keys](config.md#model-loading) of the config.
`--kv-quant-scheme kvarn` on a positional model is the same as
`load: {kv_quant_scheme: kvarn}` on a config model, priced and reported the
same way.

These flags set a positional model's sampling defaults, the
[`sampling` keys](config.md#sampling) of the config. A default applies to a
request that omits the field, and a request that sends the field wins, so
`--temp 0` does not pin a client that sends its own temperature. They sit on
top of the family defaults `gmlx profiles` prints, and an unknown `--profile`
is refused at start:

| Flag | Default | Meaning |
|------|---------|---------|
| `--temp T` | Family base | Sampling temperature. |
| `--top-p P` | Family base | Nucleus probability. `0` disables the filter. |
| `--top-k N` | Family base | Candidate count. `0` disables the filter. |
| `--min-p P` | Family base | Minimum probability relative to the best token. `0` disables. |
| `--seed N` | None | A sampling seed for every request that sends none. |
| `--repetition-penalty X` | None | Penalty over the last `--repetition-context-size` tokens. |
| `--repetition-context-size N` | `20` | Window for the repetition penalty. |
| `--presence-penalty X` | None | Penalty on any token already generated. |
| `--frequency-penalty X` | None | Penalty scaled by how often a token was generated. |
| `--stop STR` | None | A stop sequence, repeatable. Chat completions only. |
| `--xtc-probability P` | None | XTC sampling probability. Not available on speculative models. |
| `--xtc-threshold T` | None | XTC sampling threshold. |
| `--thinking-start-token STR` | `<think>` | The model's opening reasoning marker. |
| `--thinking-end-token STR` | `</think>` | The model's closing reasoning marker. |

These flags control speculative decoding:

| Flag | Default | Meaning |
|------|---------|---------|
| `--speculative` | Off | Speculate with the model's own MTP head or `--draft-gguf`. A config `discover` scan enables it on its own. |
| `--draft-gguf PATH` | None | A separate [drafter](glossary.md#drafter) GGUF, which implies `--speculative`. |
| `--native-mtp` | Off | Prefer the model's own head when `--draft-gguf` is also set. |
| `--draft-block-size N` | Drafter default | Block size of each round, which drafts N-1 tokens and checks them in one N-token target pass. |
| `--speculative-width-cap N` | Drafter default | Speculate only while at most N requests decode together. `0` uncapped. |
| `--stochastic-mtp` | Off | Accept sampled drafts by rejection sampling. More accepted, not token-identical. |

These flags stream a model bigger than memory, as
[Models larger than memory](streaming.md) explains:

| Flag | Default | Meaning |
|------|---------|---------|
| `--stream-experts` | Off | Stream the routed experts from disk. Attention and the KV cache stay on GPU. |
| `--stream-cpu` | Off | Run the whole model on the CPU device from the page cache. |
| `--stream-fast-disk {auto,on,off}` | `auto` | The streamed-decode prefetch policy under `--stream-experts`. `auto` probes the drive. |
| `--prefill-feeder`, `--no-prefill-feeder` | On | Stage expert prefill directly from the GGUF. |
| `--decode-feeder`, `--no-decode-feeder` | On under `--stream-experts` | Decode from a wired, popularity-managed expert [arena](glossary.md#arena). |
| `--gpu-keepwarm` | On for streamed loads | Keep GPU clocks high while a streamed model decodes. |
| `--moe-experts K` | Trained | Cap the router at K experts for each token, lossy. |
| `--moe-expert-mass P` | Off | Keep the smallest expert set covering share P of gate mass, lossy. |
| `--moe-miss-shed P` | Off | Drop experts that would miss the arena down to share P, lossy. |
| `--moe-layer-shed P` | Off | Skip a streamed layer's experts with probability P, lossy. |
| `--moe-prestage {ranked,keepers}` | `ranked` | `keepers` filters prestage predictions through the miss-shed policy, so it needs `--moe-miss-shed`. |

These flags enable services. Each is also a `server` key and is described
in [Speech, embeddings and rerank](services.md):

| Flag | Default | Meaning |
|------|---------|---------|
| `--stt [MODEL]` | Off | Speech-to-text at `POST /v1/audio/transcriptions`. Bare is `whisper-turbo`. |
| `--tts [MODEL]` | Off | Text-to-speech at `POST /v1/audio/speech`. Bare is `kokoro`. |
| `--embeddings [MODEL]` | Off | Embeddings at `POST /v1/embeddings`. Bare is `qwen3-embed-0.6b`. No extra is needed. |
| `--rerank [MODEL]` | Off | Reranking at `POST /v1/rerank`. Bare is `qwen3-rerank-0.6b`. No extra is needed. |

`serve` has no `--api-key` flag, because the key lives in the config.
[Address and authentication](config.md#address-and-authentication) gives
the reasons.

Each completed request logs a line with the endpoint, model, token counts
and timing:

```text
[req] 2026-06-15 16:07:42 /chat/completions qwen3-0.6b prompt=19 gen=3 ttft=0.47s prefill=45t/s decode=172.6t/s total=0.51s
```

## gmlx stop

`gmlx stop` stops a background server with SIGTERM to the process group,
then SIGKILL after the timeout. Before it signals, it checks that the pid
belongs to the gmlx server, and it clears and reports any stale runfiles
found during the check.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Which server, when several are backgrounded. |
| `--port P` | The managed server | Which server. |
| `--timeout S` | `15` | Seconds before SIGKILL, which ends any in-flight generation. |
| `--stale` | Off | Clear runfiles whose server has exited and signal nothing. |

## gmlx status

`gmlx status` prints a background server's pid, uptime, URL, log path and
how it is managed. It uses `/health`, so it needs no API key. Stale runfiles
are listed with the reason and their age.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Which server. |
| `--port P` | The managed server | Which server. |
| `--json` | Off | Emit JSON. |

The command exits 0 when a server is running and 3 when none is.

## gmlx restart

`gmlx restart` stops the server and relaunches it with the arguments
recorded in its runfile, from any directory.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Which server. |
| `--port P` | The managed server | Which server. |
| `--timeout S` | `15` | Seconds before SIGKILL during the stop. |
| `--start-timeout S` | `40` | Readiness wait for the new process. |

## gmlx logs

`gmlx logs` prints the tail of a background server's log. The menu bar's
health polls are filtered out of it.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | The managed server | Which server. |
| `--port P` | The managed server | Which server. |
| `-n`, `--lines N` | `40` | Lines to print. |
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
| `install` | The `serve` flags plus the table below | Register the login item and start now. |
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
| `--config FILE` | The first default location | Which config. |
| `-v`, `--paths` | Off | Also show each model's GGUF path. |
| `--json` | Off | Emit JSON. |

Exit code 2 means no config was found. The message names `gmlx init`.

## gmlx run

`gmlx run` loads a GGUF and generates a completion, runs a benchmark, or
prints the load plan. `--help` shows the common flags, and `--help-all`
shows every flag.

```sh
gmlx run model.gguf --prompt "Explain entropy." --max-tokens 128
gmlx run model.gguf --bench 512,4096,16384 --bench-runs 3
gmlx run model.gguf --bench-depths 0,4096,16384,32768
gmlx run model.gguf --report-only
gmlx run coder --prompt "Refactor this loop."    # a config id, with its settings
```

The positional is a path, or a model id or alias from your server config
when it is not a file. A config id supplies its path, sampling, system
prompt, template, adapter, drafter and streaming placement, although flags
you pass still win. An id with an unknown profile fails and lists the valid
ones.

These flags control generation:

| Flag | Default | Meaning |
|------|---------|---------|
| `gguf`, positional | Required | The GGUF, which may be sharded, or a config id. |
| `--prompt STR` | `Hello, world!` | The prompt. |
| `--prompt-file PATH` | None | Read the prompt from a file. |
| `--system-prompt STR` | None | A system message for the chat template. |
| `--max-tokens N` | Until the model stops | Generation cap. |
| `--temp F` | Family default | Temperature. `0` is greedy. |
| `--top-p F` | Family default | Nucleus probability. |
| `--top-k N` | Family default | Candidate count. `0` disables. |
| `--min-p F` | Family default | Minimum probability relative to the best token. |
| `--repetition-penalty F` | `0` | Repetition penalty. `0` disables. |
| `--repetition-context-size N` | `20` | Tokens the repetition penalty looks back over. |
| `--presence-penalty F` | `0` | Presence penalty. |
| `--frequency-penalty F` | `0` | Frequency penalty. |
| `--xtc-probability F`, `--xtc-threshold F` | `0` | XTC sampling, text path only. |
| `--logit-bias JSON` | None | Token id to bias map. |
| `--stop STR` | None | A stop sequence, repeatable. |
| `--seed N` | None | Sampling seed. |
| `--reasoning {show,hide,raw}` | `show` | `show` styles the thinking and strips its markers, `hide` prints only the answer, `raw` passes everything through. |
| `--thinking {on,off,adaptive}` | Template default | The reasoning switch, mapped to the model's template variable. |
| `--reasoning-effort LEVEL` | Template default | Reasoning depth on models that support levels. |
| `--thinking-budget N` | Unlimited | Cap reasoning tokens. |
| `--thinking-start-token STR`, `--thinking-end-token STR` | Detected | The model's reasoning markers when detection fails. |
| `--chat-template-config JSON` | None | Extra template variables, such as `'{"enable_thinking": false}'`. |
| `-v`, `--verbose` | Off | Full load diagnostics instead of the spinner. |

These flags pick profiles and family defaults:

| Flag | Default | Meaning |
|------|---------|---------|
| `--profile NAME` | None | A built-in intent or, with a config, a user profile. Equivalent to `@NAME` on the positional. |
| `--no-family-defaults` | Off | Do not apply the family's sampling defaults on a bare path. |
| `--config FILE` | The first default location | The config an id is resolved against. |

These flags control memory:

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-kv-size N` | None | Cap the KV cache with a rotating window. Combines with [kvarn](glossary.md#kvarn) but not with affine `--kv-bits`. |
| `--kv-bits N` | Off | Quantize the KV cache to 2, 3, 4, 6 or 8 bits affine, or to 2, 3, 4, 5, 6 or 8 under kvarn, default 6. |
| `--kv-group-size N` | `64` | Affine quantization group size. |
| `--kv-quant-scheme {uniform,kvarn}` | `uniform` | `kvarn` is variance-normalized quantization, [KV cache quantization](kv-quantization.md). |
| `--kv-tail-tokens N` | `1024` | Under kvarn, the newest N tokens stay fp16. A multiple of 128. `0` disables. |
| `--quantized-kv-start N` | `0` | Tokens kept unquantized at the start of the cache. Not applied under kvarn. |
| `--prefill-step-size N` | `2048`, `8192` when streaming | Prefill chunk size. |
| `--dtype {auto,bfloat16,float16}` | `auto` | Activation width. `auto` is float16 on M1 and M2. |

A width outside the scheme's list exits 2, and so does a `--max-kv-size`
window too small for kvarn's block layout, which
[KV cache quantization](kv-quantization.md) describes along with
the models kvarn declines. A declined model prints the reason and runs fp16
KV, and the VLM media path always keeps fp16.

These flags control loading:

| Flag | Default | Meaning |
|------|---------|---------|
| `--arch NAME` | Detected | Override architecture detection. |
| `--hf-source ID_OR_DIR` | None | Take the config, processor and template from this repo or directory. |
| `--chat-template STR_OR_PATH` | The GGUF's | Inline Jinja or a `.jinja` or `.txt` file. |
| `--no-chat-template` | Off | Pass the prompt verbatim, for base models. |
| `--no-remap` | Off | Keep raw GGUF tensor names. |
| `--no-zero-copy` | Off | Copy tensors out of the mmap instead of viewing them. |
| `--adapter PATH` | None | A GGUF LoRA adapter applied at load, text only. |

These flags are multimodal. [Vision and audio](vlm.md) describes them:

| Flag | Default | Meaning |
|------|---------|---------|
| `--mmproj PATH` | None | The projector GGUF. |
| `--image PATH_OR_URL` | None | Images to prepend, comma separated. |
| `--audio PATH_OR_URL` | None | Audio to prepend, comma separated. Needs an audio tower. |
| `--resize-shape N_OR_WxH` | Model default | Resize images before encoding. |

With `--mmproj`, `--stop` and the XTC flags are ignored with a warning,
and the bench, report and streaming flags exit with an error.

These flags control speculative decoding, which
[Speculative decoding](speculative-decoding.md) describes:

| Flag | Default | Meaning |
|------|---------|---------|
| `--speculative`, `--mtp` | Auto for models with an MTP head | Force speculation on. |
| `--no-speculative`, `--no-mtp` | Off | Force it off. |
| `--draft-gguf PATH` | Detected sibling | A separate drafter GGUF, which implies `--speculative`. |
| `--native-mtp` | Off | Prefer the model's own head when a drafter is also present. |
| `--draft-block-size N` | Drafter default | Block size of each round, which drafts N-1 tokens and checks them in one N-token target pass. |
| `--stochastic-mtp` | Off | Accept sampled drafts by rejection sampling. More accepted, not token-identical. |

Speculation honors `--temp`, `--top-p`, `--top-k`, `--min-p` and
`--system-prompt`. A flag it cannot honor, such as `--stop`, a penalty,
`--logit-bias` or `--max-kv-size`, is dropped with a warning, and
`--no-mtp` decodes on the plain path, which honors everything. KV
quantization works on the speculative path and quantizes the same layers
`serve` would. The exceptions are kvarn on a sliding-window stack and on an
architecture whose drafter reads the target KV. There kvarn declines, and
the `[kv]` line gives the reason.

These flags stream a model bigger than memory, which
[Models larger than memory](streaming.md) describes:

| Flag | Default | Meaning |
|------|---------|---------|
| `--stream-experts` | Off | Stream the routed experts from disk. Attention and the KV cache stay on GPU. |
| `--stream-cpu` | Off | Run the whole model on the CPU device from the page cache. |
| `--stream-fast-disk {auto,on,off}` | `auto` | The prefetch policy. `auto` measures the drive at load. |
| `--prefill-feeder`, `--no-prefill-feeder` | On | Stage expert prefill directly from the GGUF. |
| `--decode-feeder`, `--no-decode-feeder` | On under `--stream-experts` | Decode from a wired, popularity-managed expert arena. |
| `--gpu-keepwarm` | On for streamed loads | Keep GPU clocks high while decoding. |
| `--moe-experts K` | Trained | Cap the router at K experts for each token, lossy. |
| `--moe-expert-mass P` | Off | Keep the smallest expert set covering share P of gate mass, lossy. |
| `--moe-expert-probe` | Off | Run lossless and print how many experts each token needed at candidate P values. |
| `--moe-miss-shed P` | Off | Drop experts that would miss the arena down to share P, lossy. |
| `--moe-layer-shed P` | Off | Skip a streamed layer's experts with probability P, lossy. |
| `--moe-prestage {ranked,keepers}` | `ranked` | `keepers` filters prestage predictions through the miss-shed policy, so it needs `--moe-miss-shed`. |

These flags inspect and benchmark:

| Flag | Default | Meaning |
|------|---------|---------|
| `--report-only` | Off | Print the load plan and the rendered prompt without building the model. |
| `--bench LIST` | None | Prompt lengths to time, comma separated. Prints prefill and decode tok/s. |
| `--bench-depths LIST` | None | Context depths to time decode at. |
| `--bench-runs N` | `2` | Timed runs at each length. The best is reported. |
| `--bench-decode-tokens N` | `32`, `128` for depths | Decode tokens in each run. |
| `--bench-temp T` | `0` | Temperature for speculative bench runs. |
| `--bench-chat-dataset DATASET` | Synthetic | A Hugging Face chat dataset for bench prompts, `id` or `id:split`. |

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
| `gguf`, positional | Server default | A GGUF, a config id, or a served id. |
| `--server` | Auto when the server is running | A plain client of the server. |
| `--assistant` | Off | Chat through the server's tool-loop assistant, with MCP tools and memory. |
| `--local` | Off | Load in-process even when the server is running. |
| `--base-url URL` | The managed server | An explicit server. |
| `--host H`, `--port P`, `--api-key KEY` | The managed server | The server to target and its key. |
| `--no-start` | Off | Never start the server. |
| `--start-timeout S` | `180` | How long an auto-start may take. |
| `--config FILE` | The first default location | The config an id is resolved against. |
| `--profile NAME` | None | A built-in intent or user profile. |
| `--no-family-defaults` | Off | Do not apply the family's sampling defaults on a bare path. |

These flags control generation. All of them can be changed during the chat:

| Flag | Default | Meaning |
|------|---------|---------|
| `--system-prompt STR` | None | The system message, sent on the first turn and after each reset. |
| `--max-tokens N` | Until the model stops | Cap on each reply. |
| `--temp F`, `--top-p F`, `--top-k N`, `--min-p F` | Family default | Sampling |
| `--repetition-penalty F`, `--presence-penalty F`, `--frequency-penalty F` | `0` | Penalties |
| `--repetition-context-size N` | `20` | Tokens the repetition penalty looks back over. |
| `--xtc-probability F`, `--xtc-threshold F` | `0` | XTC sampling. |
| `--logit-bias JSON` | None | Token id to bias map. |
| `--stop STR` | None | A stop sequence, repeatable. |
| `--seed N` | None | Sampling seed. |
| `--reasoning {show,hide,raw}` | `show` | `show` styles the thinking and strips its markers, `hide` prints only the answer, `raw` passes everything through. |
| `--thinking {on,off,adaptive}`, `--reasoning-effort LEVEL` | Template default | The reasoning switch and depth. |
| `--thinking-budget N` | Unlimited | Cap reasoning tokens. |
| `--thinking-start-token STR`, `--thinking-end-token STR` | Detected | The model's reasoning markers when detection fails. |
| `--chat-template-config JSON` | None | Extra template variables. |

These flags control display and sessions:

| Flag | Default | Meaning |
|------|---------|---------|
| `--render {auto,plain,lite,rich}` | `auto` | Markdown rendering of replies. |
| `--theme NAME` | The config's `theme:`, else `dark` | Color theme. |
| `--colorblind` | Off | Colorblind-friendly accents on any theme. |
| `--no-history` | Off | Do not read or write the prompt history file. |
| `--no-autosave` | Off | Do not save the session after each turn. |
| `--resume [NAME]` | Off | Resume a saved session. Bare is this model's latest. |
| `-v`, `--verbose` | Off | Full load diagnostics. |

A local load also takes these [`gmlx run`](#gmlx-run) flags, which mean
the same as they do there:

| Group | Flags shared with `run` |
|-------|-------------------------|
| loading | `--arch`, `--hf-source`, `--chat-template`, `--no-chat-template`, `--no-remap`, `--no-zero-copy`, `--adapter` |
| memory | `--max-kv-size`, `--kv-bits`, `--kv-group-size`, `--kv-quant-scheme`, `--kv-tail-tokens`, `--quantized-kv-start`, `--prefill-step-size`, `--dtype` |
| multimodal | `--mmproj`, `--resize-shape` |
| speculation | `--speculative`, `--mtp`, `--no-speculative`, `--no-mtp`, `--draft-gguf`, `--native-mtp`, `--draft-block-size`, `--stochastic-mtp` |
| streaming | `--stream-experts`, `--stream-cpu`, `--stream-fast-disk`, `--prefill-feeder`, `--no-prefill-feeder`, `--decode-feeder`, `--no-decode-feeder`, `--gpu-keepwarm` |
| lossy streaming | `--moe-experts`, `--moe-expert-mass`, `--moe-expert-probe`, `--moe-miss-shed`, `--moe-layer-shed`, `--moe-prestage` |

None of them applies when the chat is a server client. A base model with no
chat template refuses to start until you pass one with `--chat-template` or
send turns verbatim with `--no-chat-template`.

## gmlx launch

`gmlx launch` writes an external tool's configuration to point at a gmlx
server, starts the server if none is reachable, and runs the tool. It never
installs the tool. Most clients get a configuration of their own under
`~/.config/gmlx`, while pi, omp and goose get a provider merged into their
own files. [Agents and chat apps](launch.md) describes each client.

```sh
gmlx launch opencode
gmlx launch pi --model qwen3.8-27b-ud-q6@coding
gmlx launch claude-code --model qwen3.8-27b-ud-q6
gmlx launch open-webui
gmlx launch dsh --model qwen3.8-27b-ud-q6
gmlx launch omp --config-only
```

| Flag | Default | Meaning |
|------|---------|---------|
| `client`, positional | Required | `claude-code`, `opencode`, `pi`, `omp`, `hermes`, `goose`, `aichat`, `elia`, `open-webui`, `dsh` or `menubar`. |
| `--model ID[@profile]` | The server's default | The served model the tool uses, which the server keeps loaded through its idle timeout. |
| `--base-url URL` | None | An explicit server, never auto-started. |
| `--host H`, `--port P` | The managed server | The server to target. |
| `--api-key KEY` | A placeholder | The key, written to the tool's native config field. Without one, tools that require a key get the provider id. |
| `--provider-id NAME` | `gmlx` | The provider id written into the tool's config. |
| `--config-path PATH` | The client's location | Where the tool config is written, a file or a directory depending on the client, as [How a launch works](launch.md#how-a-launch-works) lists. |
| `--config-only` | Off | Write the config and print the run command without running it. |
| `--no-start` | Off | Never start a server. |
| `--start-timeout S` | `0`, no limit | Cap the auto-start wait. |
| `--no-keep` | Off | Do not keep `--model` resident. |
| `--dsh-profile NAME` | `gmlx` | dsh only: the dsh profile to boot with the gmlx overlay, as [dsh](launch.md#dsh) describes. |

The command exits 0 when the tool ran or the server is ready, 1 when the
server is unreachable or died, 2 when the config is missing or malformed,
and 130 when interrupted during the start wait.

### launch menubar

`gmlx launch menubar` runs the macOS menu bar app directly, although a
background `serve` starts it automatically. What it shows is in
[Menu bar app](menubar.md).

| Flag | Default | Meaning |
|------|---------|---------|
| `-f`, `--foreground` | Off | Run the event loop in this process. |
| `--stop` | Off | Quit a detached menu bar app. |
| `--url URL` | The managed server | The server to track. |
| `--host H`, `--port P` | The managed server | The server to track. |
| `--api-key KEY` | The managed server's | The key for a keyed server the app cannot read the config of. |
| `--interval S` | `4` | Poll interval in seconds. |

## gmlx pull

`gmlx pull` checks a remote GGUF's header and, when it will load, downloads
all its shards into your model library as plain files. A file saved under a
`model_dirs` root is registered in the config immediately, and any running
server is signalled to reload it.

```sh
gmlx pull hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
gmlx pull hf:org/repo/model.gguf --to ~/models
gmlx pull hf:org/gemma-3-27b-GGUF/gemma-3-27b-Q4_K_M.gguf mmproj-F16.gguf
```

| Flag | Default | Meaning |
|------|---------|---------|
| `refs`, positional | Required | `hf:<org>/<repo>/<file.gguf>[@rev]` or a URL. Later bare filenames resolve in the first ref's repo. |
| `--to DIR`, `--out DIR` | The first `model_dirs` root | Download into this directory instead, with no repo subfolder. |
| `--config FILE` | The first default location | The config to read `model_dirs` from. |
| `--force` | Off | Download even when the header check or the disk-space check fails. |
| `--no-register` | Off | Do not add the file to the config. |
| `--hf-source ID` | None | Treat the architecture as loadable with this config override. |
| `--max-mb N` | `128` | Cap the header range read. |
| `--json` | Off | Emit each verdict as JSON before downloading. |

Inside a `model_dirs` root, downloads nest under `<org>__<repo>/` so that a
model's siblings stay together. Before the first byte, `pull` checks that
the volume has space for every shard, and it notes, without refusing, a
model that will not fit this Mac's RAM. A stalled or dropped read retries
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
| local path | `~/models/model.gguf` |
| `hf:` file | `hf:org/repo/path/file.gguf`, optionally `@<revision>` |
| `hf:` folder | `hf:org/repo/UD-Q5_K_M`. A single model inside resolves. Several are listed |
| `hf:` repo | `hf:org/repo`. Each quant is listed as a complete ref |
| Hugging Face page | A `blob`, `tree` or `resolve` link, rewritten to the file or folder |
| direct URL | `https://host/path/file.gguf` |

| Flag | Default | Meaning |
|------|---------|---------|
| `ref`, positional | Required | The file, folder, repo or URL. |
| `--arch NAME` | Detected | Override architecture detection. |
| `--hf-source ID` | None | Treat the architecture as loadable with this config override. |
| `--max-mb N` | `128` | Cap the header range read. |
| `--json` | Off | Emit the verdict as JSON. |

A split model is checked across all its shards, because a codec used by a
single tensor can appear only in a later shard. Projector GGUFs are
recognized as companions and are not checked as models.

The command exits 0 when the file will load, 1 when it will not, and 2
when the reference cannot be resolved or read.

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
| `ID`, positional | Required | A model id, alias, or discovered model's id. |
| `--config FILE` | The first default location | Which config. |
| `--keep-files` | Off | Remove only the config entry. |
| `--yes` | Off | Skip the confirmation. Required without a terminal. |
| `--json` | Off | Emit the result as JSON. Needs `--yes`. |
| `--no-reload` | Off | Do not signal a running server to re-read the file. |

The command exits 0 when the model was removed, 1 when you declined or a
file could not be deleted, and 2 for an unknown id or a missing config.

## gmlx sync-models

`gmlx sync-models` rescans the model directories and updates the `models`
block to match disk. Existing entries keep their comments and edits, entries
whose file is gone are dropped, and new files are added, with a sibling
drafter pairing into the model it serves. Run it after adding files to the
directory or pulling them.

```sh
gmlx sync-models
gmlx sync-models --from-hf-cache
gmlx sync-models --dry-run
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | The first default location | Which config. |
| `--models-dir DIR` | The config's `model_dirs` | Directories to scan, repeatable. |
| `--from-hf-cache`, `--hf-cache` | The config's `hf_cache` | Also reconcile the Hugging Face cache. |
| `-r`, `--recursive`, `--no-recursive` | Deep | Descend into subdirectories. Deep because `pull` nests. |
| `--dry-run` | Off | Print the plan without writing. |
| `--no-reload` | Off | Do not signal a running server to re-read the file. |

An entry that could not be checked, because its root is unmounted or the
cache is unreadable, is kept and reported instead of dropped.

## gmlx ps

`gmlx ps` shows the models resident in a running server from its
`/v1/metrics` snapshot, with the id, size, idle time, TTL, pinned state and
path of each.

| Flag | Default | Meaning |
|------|---------|---------|
| `--url URL` | The managed server | The server's base URL. |
| `--host H`, `--port P` | The managed server | The server to target. |
| `--api-key KEY` | The `GMLX_API_KEY` variable | The key for a keyed server. |
| `--json` | Off | Emit JSON. |

The command exits 0 when it listed the models, 1 when the server answered
with an error or is not gmlx, and 3 when no server was reachable.

## gmlx systemone

`gmlx systemone` sends a JSON file holding a `/v1/systemone` request body
to a running server and prints one line per question. The body is described in
[Structured decisions](decisions.md). With `--model` the verb loads the GGUF
itself and answers offline, with no server.

```sh
gmlx systemone ticket.json
gmlx systemone ticket.json --model diffusiongemma-Q4_K_M.gguf
```

A yes or no answer prints as its probability, a choice as the chosen
option with its confidence, and a score as the expected level with its
confidence. A question skipped by `ask_if` prints `skipped`.

| Flag | Default | Meaning |
|------|---------|---------|
| `REQUEST.json`, positional | None | The file holding the request body. |
| `--url URL` | The managed server | The server's base URL. |
| `--host H`, `--port P` | The managed server | The server to target. |
| `--api-key KEY` | The `GMLX_API_KEY` variable | The key for a keyed server. |
| `--model GGUF` | None | Answer offline on this GGUF path or configured model id. Not combined with `--url`, `--host` or `--port`. |
| `--config FILE` | The first default location | The config whose model ids and `server.systemone` settings an offline run uses. Needs `--model`. |
| `--seed N` | The request's `seed` | Replaces the request's seed on both paths. |
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
| `id`, positional | None | A model id or alias to resolve. |
| `--config FILE` | The first default location | Which config. |
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

Most flags override the key of the same name in the
[`talk` block](config.md#voice), and a flag's default applies when neither
is set.

| Flag | Default | Meaning |
|------|---------|---------|
| `model`, positional | The server's default model | The served model, with an optional `@profile`. |
| `--mode {wake,vad,ptt,text}` | `wake` | How a turn starts. |
| `--once` | Off | One exchange without the wake gate, then exit. |
| `--wake-word PHRASE` | `hey assistant` | Any phrase, no training. |
| `--wake-threshold X` | `0.3` | Higher means fewer false wakes. |
| `--vad-threshold X` | `0.6` | Speech probability above which a frame is speech. |
| `--vad-silence-ms MS` | `550` | Trailing silence that ends an utterance. |
| `--min-speech-ms MS` | `300` | Shorter utterances are discarded. |
| `--voice NAME` | The server's default | The TTS voice. |
| `--list-voices` | Off | List the server's voices and exit. |
| `--speed X` | `1.0` | Speech speed, 0.25 to 4. |
| `--no-chime` | Off | Disable the wake and idle sounds. |
| `--input-device D`, `--output-device D` | System default | Audio devices by name substring or index. |
| `--list-devices` | Off | List audio devices and exit. |
| `--system TEXT` | The talk default | The spoken persona. |
| `--language L` | Detected | A Whisper language hint. |
| `--max-tokens N` | Until the model stops | Reply cap. |
| `--brain {chat,assistant}` | `chat` | Plain chat, or the assistant with tools and memory. |
| `--base-url URL` | The managed server | An explicit server, which also runs the speech services. |
| `--host H`, `--port P`, `--api-key KEY` | The managed server | The server to target and its key. |
| `--no-start` | Off | Never start the server. |
| `--start-timeout S` | `180` | How long an auto-start may take. |
| `--config PATH` | The first default location | The YAML with the `talk` block. |

## gmlx train

`gmlx train` trains a LoRA adapter on a quantized GGUF base and writes it
as a GGUF adapter. The base stays quantized throughout, so a model that
does not fit in fp16 can still be fine-tuned. For the walkthrough, read
[LoRA adapters](lora.md).

```sh
gmlx train base-Q8_0.gguf --data ./my-data --adapter-out my-lora.gguf
gmlx run base-Q8_0.gguf --adapter my-lora.gguf --prompt "..."
```

| Flag | Default | Meaning |
|------|---------|---------|
| `model`, positional | Required | The base GGUF, or a config id. |
| `--data PATH_OR_ID` | Required | A directory with `train.jsonl` and `valid.jsonl`, or a Hugging Face dataset id. |
| `--adapter-out PATH` | Required | Where to write the adapter. A module the adapter cannot hold is refused before the first step. |
| `--config FILE` | The first default location | The config an id is resolved against. |
| `--iters N` | `150` | Training iterations. |
| `--batch-size N` | `4` | Batch size. |
| `--num-layers N` | `8` | Top transformer layers to adapt. |
| `--rank N` | `8` | LoRA rank. |
| `--scale F` | `20.0` | LoRA scale. Alpha is scale times rank. |
| `--dropout F` | `0.0` | LoRA dropout, below 1. |
| `--learning-rate F` | `1e-4` | Adam learning rate. |
| `--max-seq-length N` | `2048` | Longest training sequence. |
| `--val-batches N` | `25` | Validation batches in each evaluation. |
| `--steps-per-report N` | `10` | Training-loss report interval. |
| `--steps-per-eval N` | `200` | Validation interval. |
| `--seed N` | `0` | RNG seed. |
| `--hf-source ID` | None | Tokenizer and config fallback, rarely needed. |
| `--grad-checkpoint` | Off | Recompute each layer's activations in the backward pass, trading time for memory. Refused with `--dropout` above 0 and on Kimi K3 and DeepSeek-V4.1. |

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
- `train` fits a LoRA adapter on a K-quant GGUF student against the view.
- `eval` scores the student with and without the adapter.
- `census` checks, from two reply caches of the same replies, how much
  a context the student never sees, the document in the guide, moves the
  teacher.

```sh
gmlx distill gen --teacher teacher-Q6_K.gguf --prompts prompts.jsonl --out replies.jsonl
gmlx distill filter --in replies.jsonl --out corpus.jsonl
gmlx distill cache --teacher teacher-Q6_K.gguf --corpus corpus.jsonl --frame reply --out cache/
gmlx distill align --cache cache/ --student student-Q4_K_M.gguf --out view/
gmlx distill train --view view/ --student student-Q4_K_M.gguf --adapter-out student-distill.gguf --iters 2000
gmlx distill eval --student student-Q4_K_M.gguf --adapter student-distill.gguf --before \
    --slice prose=heldout.txt --md eval.md --json eval.json
```

Every size flag is in decimal GB (1e9 bytes). Each action exits 0 on
success. A refused input or setting exits 2, and so does a missing
required flag, with argparse's usage message, a missing input file, and
`filter` when its `--verify` command fails. Every output path is checked
before any model loads, and one the action cannot write exits 2 as well.
`cache` exits 2 when a shard cannot be written and keeps the verified
shards. `gen` exits 1 when some requests failed and their prompts remain
to be rerun, and 2 when its server fails to start. `align` exits 3 when
the own-group check refuses the pair, and writes no view. `cache` exits
3 when its memory probe misses twice or a `--routes` recording does not
match its rows, and 4 when the validator fails on what it wrote.
`cache --validate` exits 1 on a problem.

### distill gen

The prompt file holds one `{"id", "messages", "context"}` object per line
whose messages end on a user turn. A row's context, or the file given by
`--context`, goes in front of the last user turn for the teacher, and
either one must hold text. A row that took a context is written with the
teacher's list under `messages` and the prompt as given under
`student_messages`, and a row without one
carries `messages` alone. Prompt ids already in the output are skipped,
so a run resumes where it stopped. A resume checks that each skipped id
still names the prompt it answered and refuses when one differs or is
gone, since ids taken from line numbers shift when a line is inserted.
An interrupt cancels the queued requests and stops the server, and the
run ends when the requests in flight have failed or returned.

Beside the output, `<out>.gen.json` holds the settings a resume must
match and is written before the first request. When a run ends it
gains a `run` block with the reply and token totals read from the output
rows, the wall time summed over the runs that ended, and this run's
failed requests and aggregate token rate. An interrupted run leaves the
block as it found it, and a rerun that finds every prompt answered
writes the block from the rows when the sidecar has none. A `--base-url`
server that lists several models serves the run with the one named like
`--teacher`, and gen refuses when none or several match.

| Flag | Default | Meaning |
|------|---------|---------|
| `--out PATH` | Required | Corpus jsonl to write, with `<out>.gen.json` beside it. |
| `--prompts PATH` | None | A jsonl of prompt rows ending on a user turn. |
| `--corpus PATH_OR_ID` | None | A text corpus to build continuation prompts from, instead of `--prompts`. |
| `--teacher GGUF` | None | GGUF served for the run, the teacher or, for a measurement, the student. `--model` is the same flag. |
| `--base-url URL` | None | A running server's `/v1` base, instead of serving `--teacher`. With `--thinking-budget` the close is sized for a drafted server. |
| `--host HOST` | `127.0.0.1` | Bind host of the served teacher. |
| `--port N` | `8093` | Port of the served teacher. |
| `--text-key KEY` | `text` | With `--corpus`, text column of a jsonl or dataset row. |
| `--hf-split NAME` | `train` | With `--corpus`, dataset split for a Hugging Face id. |
| `--prefix-chars N` | `1500` | With `--corpus`, document prefix quoted in the user turn, cut at a space. |
| `--min-chars N` | `2000` | With `--corpus`, skip documents shorter than this. |
| `--docs N` | All | With `--corpus`, prompts to build. |
| `--instruction TEXT` | `Continue the following text.` | With `--corpus`, the user turn placed before the prefix. |
| `--chat-template-kwargs JSON` | None | Chat-template variables for every teacher render, as a JSON object passed to serve as `--chat-template-config`. A thinking key is refused. |
| `--context FILE` | None | Text the teacher reads for every prompt without its own context field, refused when blank. |
| `--context-format FMT` | `{context}\n\n{prompt}` | How the context and the last user turn combine, must place both fields. |
| `--thinking` | Off | Thinking on, reasoning trace kept as `reasoning_content` on the reply, off sends the server's thinking switch off. |
| `--thinking-budget N` | None | With `--thinking`, cap the reasoning trace at N tokens per request, and mark the replies it cut for `filter`. |
| `--tokenizer GGUF_OR_DIR` | `--teacher` | Tokenizer that counts the reasoning trace against the budget when `--base-url` is given. |
| `--serve-arg ARG` | None | Extra `gmlx serve` argument, repeatable, compared on a resume. A flag that changes the prompt or thinking is refused, as is a drafter with `--thinking-budget`. |
| `--startup-timeout S` | `900` | Seconds to wait for the served teacher. |
| `--concurrency N` | `8` | Requests in flight. |
| `--max-tokens N` | `1024` | Answer budget per request. With `--thinking-budget` the trace gets its own budget plus the forced close on top, without one the trace shares this budget. |
| `--temperature F` | `0.7` | Sampling temperature. |
| `--top-p F` | `0.9` | Keep the most likely tokens whose probabilities add to this. |
| `--top-k N` | The server's | Sampler top-k cutoff. |
| `--min-p F` | The server's | Minimum-probability cutoff. |
| `--seed N` | `1` | Base seed, and each request uses it plus the prompt index. |
| `--timeout S` | `1800` | Per-request timeout. |
| `--report-every N` | `50` | Progress line interval in replies. |

`--serve-arg` refuses `--thinking`, `--thinking-budget`, `--chat-template`,
`--chat-template-config`, `--reasoning-effort`, `--system-prompt` and
`--profile`, in any spelling serve accepts. Each changes what the teacher
is prompted with, and the rows would not record it. Set the thinking
switch and budget with gen's own flags, template variables with
`--chat-template-kwargs`, sampling with gen's sampling flags, and a
system prompt as a system turn in the prompt rows. A template override
has no gen form, since `cache` renders the rows with the teacher's own
template. `--native-mtp`, `--speculative` and `--draft-gguf` are refused
beside `--thinking-budget` too, because a drafted server does not hold each
request to the budget.

### distill filter

Checks run in a fixed order and the first failure names the reason, one of
`length`, `budget`, `empty`, `marker`, `repeat`, `ascii`, `tokens` and
`verify`, each defined in [Round
one](distill.md#round-one-trains-on-the-teachers-replies) of the guide. The
checker, the `--verify` command, reads the surviving rows as jsonl on stdin
and prints one line per row, `ok` or a reason word. `--context` rebuilds
every kept row with the context on the teacher's side and the prompt as
given under `student_messages`, which prepares a second round from replies a
student wrote without it. It refuses a row that already carries
`student_messages`, since that row was generated with a context.

| Flag | Default | Meaning |
|------|---------|---------|
| `--in PATH` | Required | Generated corpus jsonl, repeatable, concatenated in order, refused when the sidecars disagree, the inputs were filtered differently or two rows share an id. |
| `--out PATH` | Required | Filtered corpus to write, with `<out>.gen.json` beside it. |
| `--report JSON` | None | Write the kept and dropped counts here. |
| `--rejects PATH` | None | Write one `{id, reason}` line per dropped row here, with the checker's word under `detail`. |
| `--min-words N` | `16` | Drop replies whose answer has fewer units, a unit being a word or one ideograph or kana character, trace not counted. `--min-tokens` is the same flag. |
| `--ngram N` | `8` | N-gram size of the repetition check, in the units of `--min-words`. |
| `--max-repeat F` | `0.2` | Drop replies whose repeated n-grams exceed this fraction. |
| `--max-trace-repeat F` | `0.5` | Drop replies whose reasoning trace's repeated n-grams exceed this fraction. |
| `--max-line-repeats N` | `2` | Drop replies with a line repeated more than this many times in a row, lines without a letter or digit skipped. |
| `--max-non-ascii F` | Off | Drop replies whose non-ASCII character fraction exceeds this. |
| `--max-reply-tokens N` | Off | Drop replies longer than this many tokens, reasoning trace included. |
| `--keep-budget-hit` | Off | Keep replies whose thinking budget cut the reasoning trace. |
| `--verify CMD` | None | Shell command of your checker, which reads the survivors on stdin and prints `ok` or a reason per row. |
| `--context FILE` | None | Put this text on the teacher's side of every kept row, refused when blank. |
| `--context-format FMT` | `{context}\n\n{prompt}` | How the context and the last user turn combine, must place both fields. |

### distill cache

`cache` takes the flags of the teacher pass, listed in the order `--help`
prints them. `--messages-key` picks which list of a row the teacher reads, and
`--student-messages-key` only names the list the student's render reads
later, in `align` and `eval`. A reply or reply-think row whose final
turn has no content, such as a tool call, has nothing to target and is
dropped, counted in the `[cache] frame` line. That line also counts the
reply-think rows whose reasoning trace the teacher's template does not
render, which train on the reply alone, and a reply-think pass in which
no row keeps its trace is refused.

A corpus written by `gen` renders with the thinking switch and the
`--chat-template-kwargs` its `.gen.json` sidecar records, mapped onto the
variables the teacher's template reads. `--frame-kwargs` adds to them,
and a value that contradicts one is refused. A template that prints the
date, as Llama 3's and gpt-oss's do, renders the day the cache was first
started, and a resume and the student's render in `align` keep that day.

| Flag | Default | Meaning |
|------|---------|---------|
| `--teacher GGUF` | Required unless `--validate` | The teacher GGUF, sharded ok. |
| `--corpus PATH_OR_ID` | Required unless `--validate` | A jsonl file, a directory of text files, or a Hugging Face dataset id, `id[@config]`. |
| `--out DIR` | Required unless `--validate` | The cache directory to write. |
| `--validate DIR` | None | Validate an existing cache and exit, no teacher load. |
| `--top-k N` | `256` | Log-probabilities kept per position. |
| `--max-len N` | `2048` | Teacher tokens per window, including the start token, at least 2. The continue frame and closing tail count toward it and must leave the window at least 8. |
| `--max-disk-gb F` | None | Refuse when the size estimate exceeds this. |
| `--cache-limit-gb F` | `8.0` | MLX buffer cache cap during the pass. |
| `--logits-cap-gb F` | `4.0` | Memory cap that sizes the head sub-chunk. |
| `--floor` | Off | Also store `floor_kld`, the KL against the f16-rounded top-k. |
| `--rows-per-shard N` | `64` | Rows per shard file. |
| `--trunk N` | `512`, or `8192` streaming | Trunk chunk in tokens, rows stacked on the batch axis. |
| `--resume` | Off | Continue after the last verified shard, refused when the corpus, teacher, template, HF source or row options changed. A finished cache is validated, not redone. |
| `--max-rows N` | None | Stop after this many rows. |
| `--max-tokens N` | None | Stop after this many teacher tokens. |
| `--limit-docs N` | None | Read at most this many documents. |
| `--text-key KEY` | `text` | Text column of a jsonl or dataset row. |
| `--hf-split NAME` | `train` | Dataset split for a Hugging Face id. |
| `--source TAG` | `human`, or `synthetic` with a generator sidecar | Source tag written on every row. |
| `--frame KIND` | `none` | `none`, `continue`, `chat`, `reply` or `reply-think`: where the targets sit in the chat template, `reply-think` from the final turn's reasoning trace on. |
| `--per-turn` | Off | With the chat or reply frame, one reply row per assistant turn. |
| `--student-messages-key KEY` | `student_messages` | Corpus key of the student's own message list on reply rows. |
| `--frame-instruction TEXT` | `Continue the following text.` | User turn for the continue frame. |
| `--messages-key KEY` | `messages` | Conversation column for the chat and reply frames. |
| `--close-final-windows` | Off | With the continue frame, close the last window of a document with the turn-end marker. |
| `--frame-kwargs JSON` | None | Chat-template kwargs for every teacher render, an object or a file, added to those a `gen` sidecar records. |
| `--hf-source ID` | None | Hugging Face repo id whose config.json replaces the one synthesized from the GGUF. The tokenizer always comes from the GGUF. |
| `--no-require-feeder` | Off | Run a streaming teacher without the prefill feeder. |
| `--no-wired-limit` | Off | Leave the wired limit where it is for a teacher that fits in memory. |
| `--stream-experts` | Off | Force expert streaming on a MoE teacher that would fit in memory. |
| `--expert-bytes-gb F` | The streamed expert bytes | Expert bytes read per forward pass, for the read-traffic report. `0` for a resident teacher. |
| `--routes` | Off | MoE teachers: store every layer's top-k expert ids per position for replay by `eval`, refused when a gate cannot replay. |
| `--hidden` | Off | Also store a seeded random sketch of the teacher's final hidden state per position, for `train --hs`. |
| `--hidden-dim N` | `256` | Width of the hidden sketch. |
| `--hidden-seed N` | `1` | Seed of the sketch matrix. |
| `--cpu` | Off | Run on the CPU device, for smoke tests. |

### distill align

`align` takes these flags, listed in the order `--help` prints them.

| Flag | Default | Meaning |
|------|---------|---------|
| `--cache DIR` | Required | The cache directory. |
| `--student GGUF_OR_DIR` | Required | The student GGUF, or an MLX checkpoint directory for its tokenizer. |
| `--out DIR` | Required | The view directory to write. An earlier view there is replaced once every check has passed. |
| `--tables DIR` | None | An earlier view directory whose tokenizer tables are reused when the pair matches. |
| `--kprime N` | The maximum seen | Cap on distinct student-token groups kept per boundary. Ignored on the identity path, where K' = K. |
| `--materialize` | Off | Also write the batch tensors as view shards. |
| `--max-disk-gb F` | None | Refuse to materialize past this size. |
| `--force` | Off | Keep a view the own-group check would refuse. |
| `--val-fraction F` | `0.02` | Fraction of rows held for validation, whole documents at a time and at least one row of a cache with two. A one-document cache splits it. |
| `--seed N` | `1` | Seed of the validation split. |
| `--w-mid F` | `0.5` | Weight of an intra-word shared boundary. |
| `--gamma F` | `0.001` | Drop chunks of the chunk term (ALM) whose teacher boundary mass is below this, positive. |
| `--tau-alm F` | `1.0` | Temperature on the chunk term (ALM), positive. |
| `--T-dk F` | `1.0` | Temperature on the group softmaxes of the KL term, positive, under every `--loss` form. |
| `--max-chunk-len N` | `8` | Longest ALM chunk in tokens on either side, at least 1. |
| `--frame-kwargs JSON` | None | Chat-template kwargs for every student render, stored in the view, over those the cache recorded and its `gen` thinking switch. |
| `--cpu` | Off | Run on the CPU device, for smoke tests. |

### distill train

`train` takes these flags, in `--help` order.

| Flag | Default | Meaning |
|------|---------|---------|
| `--view DIR` | Required | A view directory, repeatable to mix views aligned alike over one tokenizer pair. |
| `--student GGUF` | Required | The student GGUF, sharded ok. |
| `--adapter-out PATH` | Required | Where to write the GGUF adapter. A path that cannot be written is refused before the load, a module the adapter cannot hold before the first step. |
| `--iters N` | Required | Training steps. |
| `--lora-rank N` | `16` | LoRA rank. |
| `--lora-scale F` | `2.0` | LoRA multiplier applied directly, nonzero. |
| `--lora-alpha F` | None | LoRA multiplier as alpha over rank, nonzero, instead of `--lora-scale`. |
| `--lora-dropout F` | `0.0` | LoRA dropout, below 1, one mask per step, replayed by `--grad-checkpoint`. |
| `--grad-checkpoint` | Off | Recompute each layer's activations in the backward pass. Refused on Kimi K3 and DeepSeek-V4.1. |
| `--lr F` | `1e-4` | Peak learning rate. |
| `--batch-size N` | `8` | Rows per step. |
| `--warmup F` | `0.05` | Warmup as a fraction of the steps, at least one step and never the last, then cosine decay. `0` starts at the peak rate. |
| `--weight-decay F` | `0` | AdamW weight decay. |
| `--clip F` | `1.0` | Gradient norm clip, `0` turns clipping off. |
| `--seed N` | `1` | Data order and LoRA init. |
| `--loss MODE` | `bucketed` | `bucketed`, `paper` or `renorm`: the sparse KL variant. |
| `--dk F` | `1` | Weight of the bucketed KL term. |
| `--alm F` | `1`, `0` when `align` took the identity path | Weight of the chunk term (ALM). |
| `--ce F` | `0` | Weight of the cross-entropy term. |
| `--T-dk F` | The view's | Override the view's T_dk. |
| `--tau-alm F` | The view's | Override the view's tau_alm. |
| `--gamma F` | The view's | Override the view's gamma, refused when it differs on a materialized view (its chunks are cut by `align`). |
| `--chunk N` | `512` | Positions per head chunk. |
| `--hs F` | `0` | Weight of the hidden-state term, a learned map from the student's final hidden state to the cache's sketch at every boundary. |
| `--hs-loss MODE` | `cosine` | `cosine` or `mse` on unit vectors. |
| `--ckpt-dir DIR` | `./ckpt` | Checkpoint directory. A fresh run refuses one that holds an earlier run's checkpoints. |
| `--resume` | Off | Resume from `--ckpt-dir`, refused when none exists or when the views, student, training settings or gmlx's validation leave-out rule differ from that run. |
| `--save-every N` | `200` | Checkpoint interval in steps. |
| `--val-every N` | `200` | Validation interval in steps. |
| `--val-batches N` | `16` | Validation batches per pass, one seeded draw across the val rows of every view. |
| `--report-every N` | `10` | Train-loss report interval. |
| `--report JSON` | None | Write the run log here. |
| `--hf-source ID` | None | Hugging Face repo id whose config.json replaces the one synthesized from the GGUF. The tokenizer always comes from the GGUF. |
| `--no-wired-limit` | Off | Leave the wired limit where it is. |
| `--cache-limit-gb F` | `8.0` | MLX buffer cache cap. |
| `--cpu` | Off | Run on the CPU device, for smoke tests. |

### distill eval

`eval` takes these flags, in `--help` order. Its task files are jsonl
files. `arc_easy.jsonl` and `hellaswag.jsonl` hold
`{id, query, choices, gold}` rows, `gsm8k.jsonl` holds
`{id, question, answer}` rows, and `gsm8k_shots.jsonl` holds the worked
examples shown before each question.

| Flag | Default | Meaning |
|------|---------|---------|
| `--student GGUF` | Required | The student GGUF. |
| `--adapter GGUF` | None | The GGUF adapter to apply. |
| `--md PATH` | Required | The Markdown report to write. |
| `--json PATH` | Required | The JSON report to write. |
| `--cache DIR` | None | Cache whose corpus the slices are checked against for overlap. |
| `--slice NAME=PATH` | None | A held-out text slice, repeatable. |
| `--teacher-bpb JSON` | None | Teacher bits per byte per slice, a `{slice: bpb}` map or an earlier eval report, shown beside the student's. |
| `--tasks-dir DIR` | `.` | Directory of the four task files. |
| `--tasks LIST` | None | Comma list of `arc_easy`, `hellaswag`, `gsm8k`. |
| `--task-limit N` | All | Items per task. |
| `--gsm8k-max-tokens N` | `384` | Generation budget per GSM8K item. |
| `--before` | Off | Also score with the adapter disabled in process, needs `--adapter`. |
| `--chat-slice NAME=PATH` | None | A jsonl of `{messages}` conversations scored on their assistant turns, `student_messages` first, repeatable. |
| `--chat-sanity PATH` | None | A jsonl of `{id, messages, kind}` chat prompts, `kind` being `task` or `refuse`, scored for template compliance and drift from an earlier report's replies. |
| `--chat-max-tokens N` | `256` | Reply budget for the chat sanity set. |
| `--chat-refs JSON` | None | An earlier eval report whose replies anchor the drift score. Ignored with `--before`, which anchors on the adapter-off replies. |
| `--chat-max-len N` | `2048` | Longest chat or reply row scored, in student tokens. A longer row loses turns until it fits, or is dropped. |
| `--chat-per-turn` | Off | Score every assistant turn as its own row. |
| `--reply-slice NAME=PATH` | None | A jsonl of conversations scored on the final reply, repeatable. |
| `--reply-think` | Off | Reply slices target the final turn from its reasoning trace onward. |
| `--reply-positions JSON` | None | A `distill census` JSON whose `high_delta` maps restrict the reply slices, refused when it names none of their rows or its frame differs from `--reply-think`. |
| `--kld-cache DIR` | None | Same-vocabulary cache to score sparse KL against, refused on another tokenizer or a cached id beyond the student's head. |
| `--kld-rows N` | All | Rows of the KL cache to score, spread over its length order. |
| `--frame-kwargs JSON` | None | Chat-template kwargs for every render. |
| `--max-len N` | `512` | Window length for bits per byte. |
| `--bpb-prefix TEXT` | None | Text placed before every window (`\n`, `\t`, `\r` and `\\` decoded), or `@continue` or `@model` for that frame's template prefix. Another `@` exits 2. |
| `--batch-size N` | `8` | Windows per batch. |
| `--cache-limit-gb F` | `4.0` | MLX buffer cache cap. |
| `--decontam-threshold F` | `0.01` | Slice window fraction found in the corpus above which its gate is void. |
| `--hf-source ID` | None | Hugging Face repo id whose config.json replaces the one synthesized from the GGUF. The tokenizer always comes from the GGUF. |
| `--cpu` | Off | Run on the CPU device, for smoke tests. |

### distill census

`census` pairs the reply rows of a cache made without a context with the
same rows in one or more caches made with one, and runs on the CPU. It
reports how much more likely the context makes each token the teacher
wrote, and the distance between the two stored top-k distributions with
everything outside the top-k pooled. With several contexts it also reports
the part no single adapter can learn.

With several `--with` caches, every cache decides which rows pair and
which positions count, while the effect, the histogram and the positions
map come from the first. The action exits 2 when a cache has no manifest
or one it cannot read, or when no rows pair. It also exits 2 when a
`--with` cache was made with another teacher, tokenizer, top-k or head
width than `--without`, and when a reply-think cache records no
`content_start`, as one written before rows carried it does. A `--corpus`
that names no file, or holds a line that is not a JSON object, exits 2 as
well.

| Flag | Default | Meaning |
|------|---------|---------|
| `--without DIR` | Required | Cache of the same replies read without the context. |
| `--with DIR` | Required | Cache with a context, repeatable. |
| `--out JSON` | Required | The census JSON to write. |
| `--md PATH` | None | Markdown summary to write. |
| `--corpus JSONL` | None | The corpus jsonl the caches were made from, so `high_delta` is keyed by row id. |
| `--delta-threshold F` | `1.0` | Nats gained at the token the teacher wrote that make a position high-delta. |
| `--pair-by MODE` | `line` | Pair rows across caches by corpus `line` or by the full `doc` id. |
| `--max-rows N` | All | Paired rows to measure. |

## gmlx doctor

`gmlx doctor` checks what a working setup needs and prints a PASS, WARN or
FAIL line for each check, with the fix named. No check accesses the
network. The checks cover the runtime and kernels, the config, and the
files of each configured model and service. They also cover background
servers, the login items and the launcher that background starts use,
optional extras, ffmpeg, MCP tools and assistant exposure. The last checks
are the Hugging Face token, RAM against each model's size, and disk space.

```sh
gmlx doctor
gmlx doctor --deep
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | The first default location | Which config. |
| `--deep` | Off | Also read each configured model's header. |
| `--json` | Off | Emit JSON. |

The command exits 0 when no check failed, 1 when a check failed, and 2 on a
usage error.

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

| Flag | Default | Meaning |
|------|---------|---------|
| `shell`, positional | None | `zsh`, `bash` or `fish`. Bare prints the help with the install lines. |

No regeneration is needed after an upgrade.
