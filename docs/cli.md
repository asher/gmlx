# CLI reference

Every verb of the `gmlx` command with its flags, defaults and exit codes. It
is the place to look up what a flag does; the guides linked from each section
explain when to use it.

| Verb | Does |
|------|------|
| [`gmlx init`](#gmlx-init) | write a starter server config from the GGUFs on disk |
| [`gmlx serve`](#gmlx-serve) | run the OpenAI and Anthropic compatible server |
| [`gmlx stop`](#gmlx-stop) | stop a background server |
| [`gmlx status`](#gmlx-status) | show a background server's pid, uptime and URL |
| [`gmlx restart`](#gmlx-restart) | stop and relaunch a background server |
| [`gmlx logs`](#gmlx-logs) | print or follow a background server's log |
| [`gmlx service`](#gmlx-service) | install the server and menu bar as a login item |
| [`gmlx list`](#gmlx-list) | list the models a config defines |
| [`gmlx run`](#gmlx-run) | generate from, benchmark or inspect one GGUF |
| [`gmlx chat`](#gmlx-chat) | chat with a model in the terminal |
| [`gmlx launch`](#gmlx-launch) | configure a coding agent or chat app to use the server |
| [`gmlx pull`](#gmlx-pull) | check a remote GGUF and download it |
| [`gmlx validate`](#gmlx-validate) | check that a local or remote GGUF will load |
| [`gmlx rm`](#gmlx-rm) | delete a model's files and config entry |
| [`gmlx sync-models`](#gmlx-sync-models) | reconcile a config with the files on disk |
| [`gmlx ps`](#gmlx-ps) | show the models resident in a running server |
| [`gmlx profiles`](#gmlx-profiles) | show the family sampling defaults and intents |
| [`gmlx talk`](#gmlx-talk) | voice chat with a served model |
| [`gmlx train`](#gmlx-train) | train a LoRA adapter on a GGUF base |
| [`gmlx doctor`](#gmlx-doctor) | check the runtime, config, models and services |
| [`gmlx completion`](#gmlx-completion) | print a shell completion script |

`gmlx --version` prints the version. `gmlx help <verb>` and `gmlx <verb>
--help` print a verb's options, and `run` and `chat` also take `--help-all`
for their full flag set. `gmlx ls` is an alias for `gmlx list`.

Settings that exist as a flag, a config key and an environment variable are
resolved flag first, then config key, then environment. The config keys are
in [server-config.md](server-config.md) and the variables in
[env-vars.md](env-vars.md). Sampling flags you leave unset take the model's
family defaults, listed in
[server-config.md](server-config.md#family-defaults).

## gmlx init

Scans your model directories and writes a starter config. Run bare on a
terminal it opens a wizard that lets you rename models, set a default and
aliases, and enable the prompt cache and the speech, embedding and rerank
services. With flags it writes the file without asking.

```sh
gmlx init                                  # the wizard
gmlx init --models-dir ~/models            # flag-driven, writes ~/.config/gmlx/gmlx.yaml
gmlx init --models-dir ~/models -r --out ./gmlx.yaml
gmlx init --from-hf-cache                  # models already in the Hugging Face cache
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--models-dir DIR` | required unless `--from-hf-cache` | directory to scan, repeatable |
| `--from-hf-cache`, `--hf-cache` | off | also scan the local Hugging Face cache and write portable `hf:` entries |
| `-r`, `--recursive`, `--no-recursive` | shallow | descend into subdirectories |
| `--out FILE` | `~/.config/gmlx/gmlx.yaml` | where to write |
| `--force` | off | overwrite an existing file |
| `-i`, `--interactive` | on a terminal | run the wizard even with flags, which pre-fill its answers |
| `--no-interactive` | off | never run the wizard |
| `--disk-cache [GB]` | off | enable the on-disk prompt cache, capped per model; bare is 50 GB |
| `--with-stt [MODEL]` | off | configure speech-to-text; bare is `whisper-turbo` |
| `--with-tts [MODEL]` | off | configure text-to-speech; bare is `kokoro` |
| `--with-embeddings [MODEL]` | off | configure embeddings; bare is `qwen3-embed-0.6b` |
| `--with-rerank [MODEL]` | off | configure reranking; bare is `qwen3-rerank-0.6b` |
| `--install`, `--no-install` | ask | install the extras the chosen services need, or never offer to |
| `--default-model ID` | none | the model used when a request omits one |
| `--port N` | `8080` | the port to write |
| `--idle-ttl SECONDS` | `900` | idle seconds before a model unloads; `none` keeps models resident |
| `--request-timeout DURATION` | `30m` in the written file | fail the request when no token arrives for this long, such as `10m` or `1h`; `none` waits forever |
| `--no-reload` | off | do not signal a running server to re-read the file |

Auto-named ids carry the quant in compact form, such as `qwen3-0.6b-q4`,
and fall back to the full codec when two quants would collide. An empty
directory is accepted; the result is a valid config with no models. When a
server is already running the config you rewrote, `init` signals it to
reload. See [getting-started.md](getting-started.md#set-up-the-server) for
the walkthrough and [server-config.md](server-config.md) for the file it
writes.

## gmlx serve

Runs the server. It detaches by default and returns at once, so the same
shell can then run `gmlx launch`; pass `--foreground` to stay attached. A
background server keeps a runfile and a log under `~/.cache/gmlx/` and, on a
macOS desktop session, raises the [menu bar app](menubar.md).

```sh
gmlx serve                                  # the config in the default location
gmlx serve --config ./gmlx.yaml
gmlx serve --models-dir ~/models --recursive
gmlx serve model-Q4_K_M.gguf                # one model, id from the filename
gmlx serve model.gguf --mmproj mmproj.gguf  # one vision model
```

The flags that end in "single model" apply to a positional GGUF only. In
config mode the same setting is a per-model key in
[server-config.md](server-config.md#models).

Where the models come from:

| Flag | Default | Meaning |
|------|---------|---------|
| `model` (positional) | none | one GGUF to serve, pinned, with the id derived from the filename |
| `--config FILE` | the first default location | serve a YAML config |
| `--models-dir DIR` | none | serve a scan of a directory, repeatable |
| `-r`, `--recursive`, `--no-recursive` | shallow | descend when scanning |
| `--hf-cache`, `--from-hf-cache` | off | let Hugging Face ids resolve from the local cache; never the network |
| `--print-config` | off | print the resolved config as YAML and exit |

Process and lifecycle:

| Flag | Default | Meaning |
|------|---------|---------|
| `--host ADDR` | config or `127.0.0.1` | bind address; a non-loopback bind needs `server.api_key` or `--no-auth` |
| `--port N` | config or `8080` | bind port |
| `--no-auth` | off | allow a non-loopback bind with no key, for auth handled by a proxy in front of the server |
| `-f`, `--foreground` | off | stay attached to the terminal |
| `--no-menubar` | off | do not raise the menu bar app |
| `--log FILE` | `~/.cache/gmlx/server-<host>-<port>.log` | the background log; each start rotates the last one to `.1` |
| `--log-level LEVEL` | `info` | `critical`, `error`, `warning`, `info`, `debug` or `trace` |
| `--start-timeout S` | `40` | seconds a background start waits for readiness before returning |

Memory and scheduling, each also a `server` key:

| Flag | Default | Meaning |
|------|---------|---------|
| `--budget-gb F` | 0.8x the GPU working set | resident weight budget across all models |
| `--max-models N` | none | cap on resident models |
| `--pin ID_OR_PATH` | none | never evict this model, repeatable |
| `--max-tokens N` | none | default completion cap |
| `--prefill-step-size N` | `2048` | prefill chunk size in tokens; lower caps peak memory |
| `--dtype {auto,bfloat16,float16}` | `auto` | activation width; `auto` is float16 on M1 and M2 |
| `--decode-prefill-ratio R` | `auto` | GPU-time share prefill gets while streams decode; `0` is stock scheduling |
| `--prefill-tick-ms MS` | `500` | wall-clock budget per prefill chunk while streams decode; `0` never halves |
| `--ignore-eos` | off | decode every request to `max_tokens`, for throughput benchmarks |

Single-model settings, each also a per-model key:

| Flag | Default | Meaning |
|------|---------|---------|
| `--mmproj PATH` | none | the projector GGUF that makes the model multimodal |
| `--hf-source REPO` | none | processor and config override for a vision model, rarely needed |
| `--adapter PATH` | none | a GGUF LoRA adapter applied at load; text only |
| `--chat-template STR_OR_PATH` | the GGUF's | inline Jinja or a `.jinja` or `.txt` file |
| `--thinking {on,off,adaptive}` | template default | the reasoning switch, mapped to the model's template variable |
| `--thinking-budget N` | unlimited | cap reasoning tokens per request; `0` closes thinking at once |

Speculative decoding:

| Flag | Default | Meaning |
|------|---------|---------|
| `--speculative` | auto | speculate with the model's own MTP head or `--draft-gguf` |
| `--draft-gguf PATH` | none | a separate drafter GGUF; implies `--speculative` |
| `--native-mtp` | off | prefer the model's own head when `--draft-gguf` is also set |
| `--draft-block-size N` | drafter default | draft tokens per round |
| `--speculative-width-cap N` | per drafter | speculate only while at most N requests decode together; `0` uncapped |
| `--stochastic-mtp` | off | accept sampled drafts by rejection sampling; more accepted, not token-identical |

Streaming a model bigger than memory, explained in
[streaming.md](streaming.md):

| Flag | Default | Meaning |
|------|---------|---------|
| `--stream-experts` | off | stream the routed experts from disk; attention and KV cache stay on GPU |
| `--stream-cpu` | off | run the whole model on the CPU device from the page cache |
| `--prefill-feeder`, `--no-prefill-feeder` | on | stage expert prefill directly from the GGUF |
| `--decode-feeder`, `--no-decode-feeder` | on under `--stream-experts` | decode from a wired, popularity-managed expert arena |
| `--gpu-keepwarm` | on for streamed loads | keep GPU clocks high while a streamed model decodes |
| `--moe-experts K` | trained | cap the router at K experts per token, lossy |
| `--moe-expert-mass P` | off | keep the smallest expert set covering share P of gate mass, lossy |
| `--moe-miss-shed P` | off | drop experts that would miss the arena down to share P, lossy |
| `--moe-layer-shed P` | off | skip a streamed layer's experts with probability P, lossy |
| `--moe-prestage {ranked,keepers}` | `ranked` | `keepers` filters prestage predictions through the miss-shed policy |

Services, each also a `server` key and described in
[services.md](services.md):

| Flag | Default | Meaning |
|------|---------|---------|
| `--stt [MODEL]` | off | speech-to-text at `POST /v1/audio/transcriptions`; bare is `whisper-turbo` |
| `--tts [MODEL]` | off | text-to-speech at `POST /v1/audio/speech`; bare is `kokoro` |
| `--embeddings [MODEL]` | off | embeddings at `POST /v1/embeddings`; bare is `qwen3-embed-0.6b`, no extra needed |
| `--rerank [MODEL]` | off | reranking at `POST /v1/rerank`; bare is `qwen3-rerank-0.6b`, no extra needed |

The API key is read from `server.api_key` in the config and nowhere else, so
the lifecycle tools and the menu bar can read the same file. Each completed
request logs one line with the endpoint, model, token counts and timing:

```text
[req] 2026-06-15 16:07:42 /chat/completions qwen3-0.6b prompt=19 gen=3 ttft=0.47s prefill=45t/s decode=172.6t/s total=0.51s
```

## gmlx stop

Stops a background server: SIGTERM to the process group, then SIGKILL after
the timeout. The pid is checked to be ours before signalling, and stale
runfiles found during the check are cleared and reported.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | the managed server | which server, when several are backgrounded |
| `--port P` | the managed server | which server |
| `--timeout S` | `15` | seconds before SIGKILL, which ends any in-flight generation |
| `--stale` | off | clear runfiles whose server has exited and signal nothing |

## gmlx status

Prints a background server's pid, uptime, URL, log path and how it is
managed. It uses `/health`, so it needs no API key. Stale runfiles are listed
with the reason and their age.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | the managed server | which server |
| `--port P` | the managed server | which server |
| `--json` | off | emit JSON |

## gmlx restart

Stops the server and relaunches it with the arguments recorded in its
runfile, from any directory.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | the managed server | which server |
| `--port P` | the managed server | which server |
| `--timeout S` | `15` | seconds before SIGKILL during the stop |
| `--start-timeout S` | `40` | readiness wait for the new process |

## gmlx logs

Prints the tail of a background server's log. The menu bar's health polls
are filtered out of it.

| Flag | Default | Meaning |
|------|---------|---------|
| `--host H` | the managed server | which server |
| `--port P` | the managed server | which server |
| `-n`, `--lines N` | `40` | lines to print |
| `-f`, `--follow` | off | keep printing as the log grows |
| `--clear` | off | truncate the log and exit |

## gmlx service

Installs a launchd login item on macOS. By default the item is the menu bar
app, which starts the server at login unless `--no-autostart` is set, and which makes
macOS attribute permission prompts to gmlx rather than to your terminal.
`install` takes every `serve` flag, and the server it starts now is the one
it starts again at each login.

```sh
gmlx service install --config ~/.config/gmlx/gmlx.yaml
gmlx service status
gmlx service uninstall
```

| Subcommand | Flags | Meaning |
|------------|-------|---------|
| `install` | the `serve` flags plus the table below | register the login item and start now |
| `status` | `--host H`, `--port P` | print the launchd state |
| `uninstall` | `--host H`, `--port P` | unload and remove the item |

| Flag | Default | Meaning |
|------|---------|---------|
| `--no-autostart` | off | install the menu bar item without starting the server at login |
| `--headless` | off | install a server-only agent with no menu bar, for machines without a desktop session |
| `--keepalive`, `--no-keepalive` | on | with `--headless`, restart the server when it crashes |

The server stays an ordinary background process, so a server you stop stays
stopped until the next login. A headless server is stopped with `service
uninstall` rather than `stop`, and the two modes cannot share a host and
port. The menu bar side is described in [menubar.md](menubar.md).

## gmlx list

Lists the models a config defines, which is the set of ids a request can
address, not the files on disk. Discovered models are tagged, aliases follow,
and the default model is marked with `*`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | the first default location | which config |
| `-v`, `--paths` | off | also show each model's GGUF path |
| `--json` | off | emit JSON |

Exit code 2 means no config was found; the message names `gmlx init`.

## gmlx run

Loads one GGUF and generates a completion, runs a benchmark, or prints the
load plan. `--help` shows the common flags and `--help-all` every flag in
the tables below.

```sh
gmlx run model.gguf --prompt "Explain entropy." --max-tokens 128
gmlx run model.gguf --bench 512,4096,16384 --bench-runs 3
gmlx run model.gguf --bench-depths 0,4096,16384,32768
gmlx run model.gguf --report-only
gmlx run coder --prompt "Refactor this loop."    # a config id, with its settings
```

The positional is a path, or a model id or alias from your server config
when it is not a file. A config id supplies its path, sampling, system prompt,
template, adapter, drafter and streaming placement; flags you pass
still win. An id with an unknown profile fails listing the valid ones.

Generation:

| Flag | Default | Meaning |
|------|---------|---------|
| `gguf` (positional) | required | the GGUF, which may be sharded, or a config id |
| `--prompt STR` | `Hello, world!` | the prompt |
| `--prompt-file PATH` | none | read the prompt from a file |
| `--system-prompt STR` | none | a system message for the chat template |
| `--max-tokens N` | until the model stops | generation cap |
| `--temp F` | family default | temperature; `0` is greedy |
| `--top-p F` | family default | nucleus probability |
| `--top-k N` | family default | candidate count; `0` disables |
| `--min-p F` | family default | minimum probability relative to the best token |
| `--repetition-penalty F` | `0` | repetition penalty; `0` disables |
| `--repetition-context-size N` | `20` | tokens the repetition penalty looks back over |
| `--presence-penalty F` | `0` | presence penalty |
| `--frequency-penalty F` | `0` | frequency penalty |
| `--xtc-probability F`, `--xtc-threshold F` | `0` | XTC sampling, text path only |
| `--logit-bias JSON` | none | token id to bias map |
| `--stop STR` | none | a stop sequence, repeatable |
| `--seed N` | none | sampling seed |
| `--reasoning {show,hide,raw}` | `show` | `show` styles the thinking and strips its markers, `hide` prints only the answer, `raw` passes everything through |
| `--thinking {on,off,adaptive}` | template default | the reasoning switch, mapped to the model's template variable |
| `--reasoning-effort LEVEL` | template default | reasoning depth on models that support levels |
| `--thinking-budget N` | unlimited | cap reasoning tokens |
| `--thinking-start-token STR`, `--thinking-end-token STR` | detected | the model's reasoning markers when detection fails |
| `--chat-template-config JSON` | none | extra template variables, such as `'{"enable_thinking": false}'` |
| `-v`, `--verbose` | off | full load diagnostics instead of the spinner |

Profiles and family defaults:

| Flag | Default | Meaning |
|------|---------|---------|
| `--profile NAME` | none | a built-in intent or, with a config, a user profile; same as `@NAME` on the positional |
| `--no-family-defaults` | off | do not apply the family's sampling defaults on a bare path |
| `--config FILE` | the first default location | the config an id is resolved against |

Memory:

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-kv-size N` | none | cap the KV cache; a rotating cache is used above it. Under kvarn the window quantizes; under affine it is refused with `--kv-bits` |
| `--kv-bits N` | off | quantize the KV cache: 2, 3, 4, 6 or 8 bits affine; 2, 3, 4, 5, 6 or 8 under kvarn, default 6 |
| `--kv-group-size N` | `64` | affine quantization group size |
| `--kv-quant-scheme {uniform,kvarn}` | `uniform` | `kvarn` is variance-normalized quantization, [performance.md](performance.md#kv-cache-quantization) |
| `--kv-tail-tokens N` | `1024` | under kvarn, the newest N tokens stay fp16; a multiple of 128, `0` disables |
| `--quantized-kv-start N` | `0` | tokens kept unquantized at the start of the cache; not applied under kvarn |
| `--prefill-step-size N` | `2048`, `8192` when streaming | prefill chunk size |
| `--dtype {auto,bfloat16,float16}` | `auto` | activation width; `auto` is float16 on M1 and M2 |

Under kvarn the first 128 tokens and the newest `--kv-tail-tokens` tokens stay
fp16. A `--max-kv-size` window must hold that sink, the tail and one 128-token
record: 384 tokens at tail 0 and 1280 at the default tail, or `run` exits 2. A
width outside the scheme's list exits 2. A model the scheme declines prints the
reason and runs fp16 KV. The VLM media path always keeps fp16.

Loading:

| Flag | Default | Meaning |
|------|---------|---------|
| `--arch NAME` | detected | override architecture detection |
| `--hf-source ID_OR_DIR` | none | take the config, processor and template from this repo or directory |
| `--chat-template STR_OR_PATH` | the GGUF's | inline Jinja or a `.jinja` or `.txt` file |
| `--no-chat-template` | off | pass the prompt verbatim, for base models |
| `--no-remap` | off | keep raw GGUF tensor names |
| `--no-zero-copy` | off | copy tensors out of the mmap instead of viewing them |
| `--adapter PATH` | none | a GGUF LoRA adapter applied at load; text only |

Multimodal, described in [vlm.md](vlm.md):

| Flag | Default | Meaning |
|------|---------|---------|
| `--mmproj PATH` | none | the projector GGUF |
| `--image PATH_OR_URL` | none | images to prepend, comma separated |
| `--audio PATH_OR_URL` | none | audio to prepend, comma separated; needs an audio tower |
| `--resize-shape N_OR_WxH` | model default | resize images before encoding |

`--stop` and the XTC flags are ignored with `--mmproj`, with a warning. The
bench, report and streaming flags error with it.

Speculative decoding, described in
[performance.md](performance.md#mtp-speculative-decoding):

| Flag | Default | Meaning |
|------|---------|---------|
| `--speculative`, `--mtp` | auto for models with an MTP head | force speculation on |
| `--no-speculative`, `--no-mtp` | off | force it off |
| `--draft-gguf PATH` | detected sibling | a separate drafter GGUF; implies `--speculative` |
| `--native-mtp` | off | prefer the model's own head when a drafter is also present |
| `--draft-block-size N` | drafter default | draft tokens per round |
| `--stochastic-mtp` | off | accept sampled drafts by rejection sampling; more accepted, not token-identical |

Speculation honors `--temp`, `--top-p`, `--top-k`, `--min-p` and
`--system-prompt`. A flag it cannot honor, such as `--stop`, a penalty,
`--logit-bias` or `--max-kv-size`, is dropped with a warning; pass
`--no-mtp` to decode on the plain path, which honors every flag. `--kv-bits`
and `--kv-quant-scheme kvarn` apply on the MTP path and quantize the same
layers `serve` does; kvarn also declines a sliding-window stack under MTP and
an architecture whose drafter reads the target KV. The `[kv]` line gives the
reason.

Streaming a model bigger than memory, described in
[streaming.md](streaming.md):

| Flag | Default | Meaning |
|------|---------|---------|
| `--stream-experts` | off | stream the routed experts from disk; attention and KV cache stay on GPU |
| `--stream-cpu` | off | run the whole model on the CPU device from the page cache |
| `--stream-fast-disk {auto,on,off}` | `auto` | the prefetch policy; `auto` measures the drive at load |
| `--prefill-feeder`, `--no-prefill-feeder` | on | stage expert prefill directly from the GGUF |
| `--decode-feeder`, `--no-decode-feeder` | on under `--stream-experts` | decode from a wired, popularity-managed expert arena |
| `--gpu-keepwarm` | on for streamed loads | keep GPU clocks high while decoding |
| `--moe-experts K` | trained | cap the router at K experts per token, lossy |
| `--moe-expert-mass P` | off | keep the smallest expert set covering share P of gate mass, lossy |
| `--moe-expert-probe` | off | run lossless and print how many experts each token needed at candidate P values |
| `--moe-miss-shed P` | off | drop experts that would miss the arena down to share P, lossy |
| `--moe-layer-shed P` | off | skip a streamed layer's experts with probability P, lossy |
| `--moe-prestage {ranked,keepers}` | `ranked` | `keepers` filters prestage predictions through the miss-shed policy |

Inspecting and benchmarking:

| Flag | Default | Meaning |
|------|---------|---------|
| `--report-only` | off | print the load plan and the rendered prompt without building the model |
| `--bench LIST` | none | prompt lengths to time, comma separated; prints prefill and decode tok/s |
| `--bench-depths LIST` | none | context depths to time decode at |
| `--bench-runs N` | `2` | timed runs per length; the best is reported |
| `--bench-decode-tokens N` | `32`, `128` for depths | decode tokens per run |
| `--bench-temp T` | `0` | temperature for speculative bench runs |
| `--bench-chat-dataset DATASET` | synthetic | a Hugging Face chat dataset for bench prompts, `id` or `id:split` |

Exit codes: 0 success, 1 the file cannot load, 2 a usage or file error,
130 interrupted.

## gmlx chat

An interactive chat in the terminal. Locally the model loads once and each
turn prefills only the new message. When the config's server is running, a
bare `gmlx chat` or one naming a served id becomes a server client instead
of loading a second copy. The commands, sessions, rendering and themes are
in [chat.md](chat.md).

```sh
gmlx chat model.gguf --temp 0.7 --system-prompt "You are terse."
gmlx chat                          # the running server's default model
gmlx chat --assistant              # the tool-loop assistant on the server
```

Where the model runs:

| Flag | Default | Meaning |
|------|---------|---------|
| `gguf` (positional) | server default | a GGUF, a config id, or a served id |
| `--server` | auto when the server is running | a plain client of the server |
| `--assistant` | off | chat through the server's tool-loop assistant, with MCP tools and memory |
| `--local` | off | load in-process even when the server is running |
| `--base-url URL` | the managed server | an explicit server |
| `--host H`, `--port P`, `--api-key KEY` | the managed server | the server to target and its key |
| `--no-start` | off | never start the server |
| `--start-timeout S` | `180` | how long an auto-start may take |
| `--config FILE` | the first default location | the config an id is resolved against |
| `--profile NAME` | none | a built-in intent or user profile |
| `--no-family-defaults` | off | do not apply the family's sampling defaults on a bare path |

Generation, all adjustable during the chat:

| Flag | Default | Meaning |
|------|---------|---------|
| `--system-prompt STR` | none | the system message, sent on the first turn and after each reset |
| `--max-tokens N` | until the model stops | per-reply cap |
| `--temp F`, `--top-p F`, `--top-k N`, `--min-p F` | family default | sampling |
| `--repetition-penalty F`, `--presence-penalty F`, `--frequency-penalty F` | `0` | penalties |
| `--repetition-context-size N` | `20` | tokens the repetition penalty looks back over |
| `--xtc-probability F`, `--xtc-threshold F` | `0` | XTC sampling |
| `--logit-bias JSON` | none | token id to bias map |
| `--stop STR` | none | a stop sequence, repeatable |
| `--seed N` | none | sampling seed |
| `--reasoning {show,hide,raw}` | `show` | `show` styles the thinking and strips its markers, `hide` prints only the answer, `raw` passes everything through |
| `--thinking {on,off,adaptive}`, `--reasoning-effort LEVEL` | template default | the reasoning switch and depth |
| `--thinking-budget N` | unlimited | cap reasoning tokens |
| `--thinking-start-token STR`, `--thinking-end-token STR` | detected | the model's reasoning markers when detection fails |
| `--chat-template-config JSON` | none | extra template variables |

Display and sessions:

| Flag | Default | Meaning |
|------|---------|---------|
| `--render {auto,plain,lite,rich}` | `auto` | markdown rendering of replies |
| `--theme NAME` | `dark` | color theme |
| `--colorblind` | off | colorblind-friendly accents on any theme |
| `--no-history` | off | do not read or write the prompt history file |
| `--no-autosave` | off | do not save the session after each turn |
| `--resume [NAME]` | off | resume a saved session; bare is this model's latest |
| `-v`, `--verbose` | off | full load diagnostics |

Loading, memory, multimodal, speculation and streaming take the same flags
as [`gmlx run`](#gmlx-run): `--arch`, `--hf-source`, `--chat-template`,
`--no-chat-template`, `--no-remap`, `--no-zero-copy`, `--adapter`,
`--max-kv-size`, `--kv-bits`, `--kv-group-size`, `--kv-quant-scheme`,
`--kv-tail-tokens`, `--quantized-kv-start`,
`--prefill-step-size`, `--dtype`, `--mmproj`, `--resize-shape`,
`--speculative`, `--mtp`, `--no-speculative`, `--no-mtp`, `--draft-gguf`,
`--native-mtp`, `--draft-block-size`, `--stochastic-mtp`, `--stream-experts`,
`--stream-cpu`, `--stream-fast-disk`, `--prefill-feeder`,
`--no-prefill-feeder`, `--decode-feeder`, `--no-decode-feeder`,
`--gpu-keepwarm`, `--moe-experts`, `--moe-expert-mass`, `--moe-expert-probe`,
`--moe-miss-shed`, `--moe-layer-shed` and `--moe-prestage`. Local-load flags
do not apply in the server modes. A base model with no chat template refuses
to start; pass one with `--chat-template` or send turns verbatim with
`--no-chat-template`.

## gmlx launch

Writes an external tool's native config to point at a gmlx server, starts
the server if none is reachable, and runs the tool. It never modifies your
dotfiles and never installs the tool. The clients and their quirks are in
[launch.md](launch.md).

```sh
gmlx launch opencode
gmlx launch pi --model qwen3.6-27b@coding
gmlx launch claude-code --model qwen3.6-27b
gmlx launch open-webui
gmlx launch omp --config-only
```

| Flag | Default | Meaning |
|------|---------|---------|
| `client` (positional) | required | `claude-code`, `opencode`, `pi`, `omp`, `hermes`, `goose`, `aichat`, `elia`, `open-webui` or `menubar` |
| `--model ID[@profile]` | the server's default | the served model the tool uses, kept resident while it runs |
| `--base-url URL` | none | an explicit server, never auto-started |
| `--host H`, `--port P` | the managed server | the server to target |
| `--api-key KEY` | none | the key, written to the tool's native config field |
| `--provider-id NAME` | `gmlx` | the provider id written into the tool's config |
| `--config-path PATH` | under `~/.config/gmlx` | where the tool config is written |
| `--config-only` | off | write the config and print the run command without running it |
| `--no-start` | off | never start a server |
| `--start-timeout S` | unbounded | cap the auto-start wait |
| `--no-keep` | off | do not keep `--model` resident |

Exit codes: 0 the tool ran or the server is ready, 1 the server is unreachable or
died, 2 no config or a malformed one, 130 interrupted during the start wait.

### launch menubar

`gmlx launch menubar` runs the macOS menu bar app directly; a background
`serve` starts it automatically. What it shows is in [menubar.md](menubar.md).

| Flag | Default | Meaning |
|------|---------|---------|
| `-f`, `--foreground` | off | run the event loop in this process |
| `--stop` | off | quit a detached menu bar app |
| `--url URL` | the managed server | the server to track |
| `--host H`, `--port P` | the managed server | the server to track |
| `--api-key KEY` | the managed server's | the key for a keyed server the app cannot read the config of |
| `--interval S` | `4` | poll interval in seconds |

## gmlx pull

Checks a remote GGUF's header and, when it will load, downloads every shard
into your model library as plain files. A file saved under a
`model_dirs` root is registered in the config immediately, and a running
server is signalled to reload it.

```sh
gmlx pull hf:unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf
gmlx pull hf:org/repo/model.gguf --to ~/models
gmlx pull hf:org/gemma-3-27b-GGUF/gemma-3-27b-Q4_K_M.gguf mmproj-F16.gguf
```

| Flag | Default | Meaning |
|------|---------|---------|
| `refs` (positional) | required | `hf:<org>/<repo>/<file.gguf>[@rev]` or a URL; later bare filenames resolve in the first ref's repo |
| `--to DIR`, `--out DIR` | the first `model_dirs` root | download here, without the `<org>__<repo>` nesting |
| `--config FILE` | the first default location | the config to read `model_dirs` from |
| `--force` | off | download even when the header check or the disk-space check fails |
| `--no-register` | off | do not add the file to the config |
| `--hf-source ID` | none | treat the architecture as loadable with this config override |
| `--max-mb N` | `128` | cap the header range read |
| `--json` | off | emit each verdict as JSON before downloading |

Downloads nest under `<dir>/<org>__<repo>/` so a model's siblings stay
together. An interrupted download resumes from its `.part` file. Before
starting, `pull` checks that the volume has space for every shard. Set
`HF_TOKEN` for gated repositories. A model that will not fit this Mac's RAM
still downloads, with a note.

## gmlx validate

Reports whether a GGUF will load, from the header alone. A remote reference
is range-read, so the check reads a few megabytes rather than the whole file.
The report names the architecture, the quant codecs, the total size across
shards, whether it fits this Mac's RAM, and for a MoE model the streaming
plan.

```sh
gmlx validate ~/models/Qwen3.6-27B-Q4_K_S.gguf
gmlx validate hf:unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf
gmlx validate hf:unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF     # lists every quant
gmlx validate https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/blob/main/Qwen3.6-27B-Q4_K_S.gguf
```

| Ref form | Example |
|----------|---------|
| local path | `~/models/model.gguf` |
| `hf:` file | `hf:org/repo/path/file.gguf`, optionally `@<revision>` |
| `hf:` folder | `hf:org/repo/UD-Q5_K_M`; one model inside resolves, several are listed |
| `hf:` repo | `hf:org/repo`; every quant listed as a complete ref |
| Hugging Face page | a `blob`, `tree` or `resolve` link, rewritten to the file or folder |
| direct URL | `https://host/path/file.gguf` |

| Flag | Default | Meaning |
|------|---------|---------|
| `ref` (positional) | required | the file, folder, repo or URL |
| `--arch NAME` | detected | override architecture detection |
| `--hf-source ID` | none | treat the architecture as loadable with this config override |
| `--max-mb N` | `128` | cap the header range read |
| `--json` | off | emit the verdict as JSON |

A split model is checked across every shard, because a codec used by one
tensor can appear only in a later shard. A projector GGUF is recognized as a
companion rather than checked as a model. Exit codes: 0 loadable, 1 not, 2
the reference could not be resolved or read.

## gmlx rm

Deletes a model's GGUF files, its partial-download files and its companions, and
removes its entry from the config. A file another model still references is
kept. Aliases to the removed id are dropped and the default model is cleared
when it named it. The plan is printed and confirmed before anything is
deleted.

```sh
gmlx rm old-model
gmlx rm old-model --yes
gmlx rm old-model --keep-files
```

| Flag | Default | Meaning |
|------|---------|---------|
| `ID` (positional) | required | a model id, alias, or discovered model's id |
| `--config FILE` | the first default location | which config |
| `--keep-files` | off | remove only the config entry |
| `--yes` | off | skip the confirmation; required without a terminal |
| `--json` | off | emit the result as JSON; needs `--yes` |
| `--no-reload` | off | do not signal a running server to re-read the file |

Exit codes: 0 removed, 1 declined or a file could not be deleted, 2 an
unknown id or no config.

## gmlx sync-models

Rescans the model directories and updates the `models` block to match disk:
existing entries keep their comments and edits, entries whose file is gone
are dropped, and new files are added. A sibling drafter pairs into the model
it serves. Run it after adding files to the directory or pulling them.

```sh
gmlx sync-models
gmlx sync-models --from-hf-cache
gmlx sync-models --dry-run
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | the first default location | which config |
| `--models-dir DIR` | the config's `model_dirs` | directories to scan, repeatable |
| `--from-hf-cache`, `--hf-cache` | the config's `hf_cache` | also reconcile the Hugging Face cache |
| `-r`, `--recursive`, `--no-recursive` | deep | descend into subdirectories; deep because `pull` nests |
| `--dry-run` | off | print the plan without writing |
| `--no-reload` | off | do not signal a running server to re-read the file |

An entry that could not be checked, because its root is unmounted or the
cache is unreadable, is kept and reported rather than dropped.

## gmlx ps

Shows the models resident in a running server from its `/v1/metrics`
snapshot: id, size, idle time, TTL, pinned, and the path. A keyed server
needs the key.

| Flag | Default | Meaning |
|------|---------|---------|
| `--url URL` | the managed server | the server's base URL |
| `--host H`, `--port P` | the managed server | the server to target |
| `--api-key KEY` | `GMLX_API_KEY` | the key for a keyed server |
| `--json` | off | emit JSON |

Exit code 1 means no server was reachable.

## gmlx profiles

Prints the family sampling table with its intents, then the config's user
profiles and each model's family. With a model id it prints that model's
resolved sampling for its base and every intent, and the layers that produced
it. It works with no config at all.

```sh
gmlx profiles
gmlx profiles qwen3.6-27b
```

| Flag | Default | Meaning |
|------|---------|---------|
| `id` (positional) | none | a model id or alias to resolve |
| `--config FILE` | the first default location | which config |
| `--json` | off | emit JSON |

## gmlx talk

Voice chat with a served model: say the wake phrase, speak, and the reply
streams back as speech. It is a client of the server's speech and chat
endpoints, so the server needs `stt` and `tts` configured. Setup, the
config block and the in-session keys are in [talk.md](talk.md).

```sh
gmlx talk
gmlx talk qwen3 --voice bf_emma
gmlx talk --mode vad
gmlx talk --once
```

| Flag | Default | Meaning |
|------|---------|---------|
| `model` (positional) | `talk.model`, else the server default | the served model, with an optional `@profile` |
| `--mode {wake,vad,ptt,text}` | `wake` | how a turn starts |
| `--once` | off | one exchange without the wake gate, then exit |
| `--wake-word PHRASE` | `hey assistant` | any phrase, no training |
| `--wake-threshold X` | `0.3` | higher means fewer false wakes |
| `--vad-threshold X` | `0.6` | speech probability above which a frame is speech |
| `--vad-silence-ms MS` | `550` | trailing silence that ends an utterance |
| `--min-speech-ms MS` | `300` | shorter utterances are discarded |
| `--voice NAME` | server default | the TTS voice |
| `--list-voices` | off | list the server's voices and exit |
| `--speed X` | `1.0` | speech speed, 0.25 to 4 |
| `--no-chime` | off | disable the wake and idle sounds |
| `--input-device D`, `--output-device D` | system default | audio devices by name substring or index |
| `--list-devices` | off | list audio devices and exit |
| `--system TEXT` | the talk default | the spoken persona |
| `--language L` | detected | a Whisper language hint |
| `--max-tokens N` | until the model stops | reply cap |
| `--brain {chat,assistant}` | `talk.brain`, else `chat` | plain chat, or the assistant with tools and memory |
| `--base-url URL` | the managed server | an explicit server, which also runs the speech services |
| `--host H`, `--port P`, `--api-key KEY` | the managed server | the server to target and its key |
| `--no-start` | off | never start the server |
| `--start-timeout S` | `180` | how long an auto-start may take |
| `--config PATH` | the first default location | the YAML with the `talk` block |

## gmlx train

Trains a LoRA adapter on a quantized GGUF base and writes it as a GGUF
adapter. The base stays quantized throughout, so a model that does not fit
in fp16 can still be fine-tuned. The walkthrough is in [lora.md](lora.md).

```sh
gmlx train base-Q8_0.gguf --data ./my-data --adapter-out my-lora.gguf
gmlx run base-Q8_0.gguf --adapter my-lora.gguf --prompt "..."
```

| Flag | Default | Meaning |
|------|---------|---------|
| `model` (positional) | required | the base GGUF, or a config id |
| `--data PATH_OR_ID` | required | a directory with `train.jsonl` and `valid.jsonl`, or a Hugging Face dataset id |
| `--adapter-out PATH` | required | where to write the adapter |
| `--config FILE` | the first default location | the config an id is resolved against |
| `--iters N` | `150` | training iterations |
| `--batch-size N` | `4` | batch size |
| `--num-layers N` | `8` | top transformer layers to adapt |
| `--rank N` | `8` | LoRA rank |
| `--scale F` | `20.0` | LoRA scale; alpha is scale times rank |
| `--dropout F` | `0.0` | LoRA dropout |
| `--learning-rate F` | `1e-4` | Adam learning rate |
| `--max-seq-length N` | `2048` | longest training sequence |
| `--val-batches N` | `25` | validation batches per evaluation |
| `--steps-per-report N` | `10` | training-loss report interval |
| `--steps-per-eval N` | `200` | validation interval |
| `--seed N` | `0` | RNG seed |
| `--hf-source ID` | none | tokenizer and config fallback, rarely needed |

The data can be chat messages, prompt and completion pairs, or plain text,
in the formats mlx-lm's trainer accepts.

## gmlx doctor

Checks everything a working setup needs and prints one PASS, WARN or FAIL
line per check with the fix named. It covers the runtime and kernels, the
config, every configured model's files, background servers, RAM against each
model's size, disk space and the Hugging Face token. It never accesses the
network.

```sh
gmlx doctor
gmlx doctor --deep
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config FILE` | the first default location | which config |
| `--deep` | off | also read every configured model's header |
| `--json` | off | emit JSON |

Exit codes: 0 nothing failed, 1 a check failed, 2 a usage error.

## gmlx completion

Prints a completion script for zsh, bash or fish. The script is a shim that
queries the installed `gmlx` for candidates on every tab, so it completes verbs,
each verb's flags, model ids from your config, client names for `launch`,
and the host, port and URL of servers you have backgrounded.

```sh
eval "$(gmlx completion zsh)"      # ~/.zshrc
eval "$(gmlx completion bash)"     # ~/.bashrc
gmlx completion fish | source      # ~/.config/fish/config.fish
```

| Flag | Default | Meaning |
|------|---------|---------|
| `shell` (positional) | required | `zsh`, `bash` or `fish` |

No regeneration is needed after an upgrade.
