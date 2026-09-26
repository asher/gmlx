# Troubleshooting

Most failures in a new setup have a known cause and a fix, grouped by the
step where they appear. [Logs and files](#logs-and-files) says where gmlx
writes its logs, runfiles, caches and sessions.

Run [gmlx doctor](cli.md#gmlx-doctor) first. It checks each part of a
working setup, and it names the fix for each check that fails. For a coding agent
or chat app that does not connect, read its entry under
[The clients](launch.md#the-clients).

- [Installing](#installing)
- [Downloading and loading models](#downloading-and-loading-models)
- [Starting the server](#starting-the-server)
- [Requests](#requests)
- [Memory](#memory)
- [Voice](#voice)
- [Distillation](#distillation)
- [Logs and files](#logs-and-files)

## Installing

### The install fails on macOS before 26.2

gmlx needs macOS 26.2 or newer, because mlx-kquant's Metal kernels are built
for that version. On an earlier version, the install fails or the
kernels cannot run. Update macOS in System Settings, then install gmlx
again.

### `gmlx: command not found` in a new terminal

The command worked in an earlier terminal, but a new one says `command not
found: gmlx`. This happens only with the [pip](installation.md#pip) route,
where gmlx lives in a Python venv that each new terminal starts with
inactive. Run `source <install dir>/.venv/bin/activate` to get the command
back. A background server or the menu bar app keeps running either way. An
install through Homebrew or `uv tool install` stays on PATH in every
terminal.

### A feature says its extra is not installed

Voice chat stops with `voice chat requires the optional talk extra`, or
speech on the server or the assistant's MCP tools report a missing extra in
the same way. These features come in
[optional extras](installation.md#optional-features). The message gives the
install command for your kind of install, and the feature works once that
command finishes.

## Downloading and loading models

### A download was interrupted or the disk filled

`gmlx pull` stopped partway, or refused to start with `error: not enough
disk space`. A dropped connection retries by itself with backoff, from the
bytes already on disk. Raise
[`GMLX_PULL_RETRIES`](env-vars.md#commands) for a flaky host, and
`GMLX_PULL_TIMEOUT` for a slow one. An interrupted pull resumes when you run
the same command again, because the bytes so far stay in a `.part` file
beside the destination.

The disk check
names how much space the file needs and how much is free. Free some space,
pass `--to DIR` for another volume, or pass `--force` to skip the check.

### A pull reports a stale partial download

The pull stops with `stale partial download: <file>.part has N bytes but the
remote file is M`. The repo replaced the file after the earlier attempt, so
the partial bytes belong to an older version. Delete the `.part` file it
names, and the next pull fetches that file from the start.

### A gated or private repo will not download

`gmlx validate` or `gmlx pull` gets a 401 or 403 from Hugging Face. gmlx
sends the Hugging Face token that the
[command environment variables](env-vars.md#commands)
describe. Accept the repo's terms on its Hugging Face page, and check
that this token has access to it.

### A load says the file is incomplete or truncated

A load stops with `incomplete split GGUF: N/M shard(s) missing`, or with
`truncated GGUF` and the size the file should have. The download did not
finish. The same `gmlx pull` fetches a missing shard, but it skips a file
that already exists, so delete a truncated file before you pull it again.

### The architecture is not supported

A load stops with `GGUF architecture 'X' is not supported`. gmlx maps each
model architecture to code that runs it, and this file's architecture has
no mapping. [Supported architectures](arch-coverage.md) lists the ones that
do. `gmlx validate` shows the architecture before you download.

### A file refuses to load with an unsupported codec

`validate`, `pull` or a load names a tensor type with no kernel, which
`validate` marks `<- no kernel`. The K-quant, legacy, IQ, MXFP4 and NVFP4
types all have kernels, as do the ternary `STQ1_0` and `PTQ1_0`
types and the 2-bit `PQ2_0` type. The usual cause is the plain ternary `TQ1_0` or `TQ2_0` type.

Pick another quant from the same repo. `gmlx validate hf:<org>/<repo>`
lists the files so that you can choose without downloading. A uniform
K-quant also [decodes fastest](performance.md#choosing-a-quant-for-speed).

### A Hadamard-folded file refuses to load

A load stops with `is Hadamard-folded` and names the fold version. Such a
file stores its weights under a rotation that gmlx undoes at run time only
for fold version 1 on the `qwen35` architecture, which covers the dense
Qwen3.5, 3.6 and 3.8 models. Any other folded file is refused before a tensor is read, and so
is a folded drafter.

`gmlx validate` prints a `weights: Hadamard-folded` line for such a file.
Read that line before you download, because the `loadable` verdict below it
does not cover the fold. Pick an unfolded quant of the same model instead.
The file contract is in [Hadamard-folded GGUFs](internals/hadamard-fold.md).

## Starting the server

### `gmlx serve` finds no config

Started in the background with no config in the
[default locations](config.md#where-gmlx-looks), `gmlx serve` prints
`note: no config found` and serves a discovery scan of the directory you
started it from, which may hold no GGUFs. Run
[`gmlx init --models-dir DIR`](config.md#create-the-file) to write a
config, or pass `--models-dir DIR`. `gmlx sync-models` has no such fallback
and stops with `no config found in the default locations`.

### `gmlx status` reports 0 models served

The server is up, but every request gets a 404. Either the config lists no
models, or every entry was skipped at startup. The log shows a
`[server] skipping model` line for each entry whose file is missing, and a
`[server] model_dirs root missing` line for each folder in
[`server.model_dirs`](config.md#servermodel_dirs) that does not exist. A
relative folder resolves against the directory the server started from, so
use absolute paths.

### Binding another address refuses to start

`gmlx serve --host 0.0.0.0` stops with `binding 0.0.0.0 exposes this server
beyond localhost`. A server reachable from the network needs a key. Set
[`server.api_key`](config.md#serverapi_key), or pass `--no-auth` when a
reverse proxy in front handles authentication.

A config with [served assistants](assistant.md#served-assistants) also
stops with `exposes the assistant tool loop beyond localhost`, because
anyone with the key could run tools on the Mac. Bind a loopback address,
remove `server.assistants`, or set
[`server.assistant_allow_remote`](config.md#serverassistant_allow_remote)
if you accept that risk.

### Port 8080 is already in use

`serve` cannot bind its port, or requests reach some other program. `gmlx
status` shows whether a gmlx server already holds the port. If one does,
run `gmlx stop`, or `gmlx restart` after a config change. A server that
`gmlx service install` set up comes back at login, so remove it with `gmlx
service uninstall`. For another program, `lsof -i :8080` names it. Free the
port, or serve on another one with `--port 8081`.

### `gmlx status` says the source changed on disk

`gmlx status` prints `source changed on disk since this server started`
under the status line.
gmlx was upgraded, or its checkout switched, while the server kept running
the old code. Requests can fail with import errors until you run `gmlx
restart`.

### The first request after startup is slow

The server answered at once, but the first reply took many seconds, because
that request loaded the model. At startup the server loads every pinned
model, else [`server.defaults.model`](config.md#serverdefaultsmodel), else
the only configured model. With several models and neither setting, pin
the ones to keep loaded, or list the ones to load in
[`server.defaults.preload`](config.md#serverdefaultspreload). A slow first
turn on a long prompt is prefill instead, which the
[prompt cache](prompt-cache.md) shortens on later turns.

## Requests

### A request names a model the server does not have

The server answers 404 of type `model_not_found`, with the ids it serves in
`available_models`, and it never downloads on a request. Use an id from
`gmlx list`, or fetch the model with `gmlx pull`, which registers it when it
lands under a `model_dirs` folder. A file saved elsewhere with `--to` needs
`gmlx sync-models --models-dir DIR` or a [`models`](config.md#models)
entry.

A suffix such as `@coding` that names no intent or profile gets 400 of type
`unknown_profile`, which lists the valid names. A 403 of type
`hf_access_disabled` is rare, and
[Hugging Face policy](api.md#hugging-face-policy) explains when it happens.

### A configured model gets 404 model_file_missing

An id from your config is not listed, or a request for it gets 404 of type
`model_file_missing`, and the log shows `[server] skipping model '<id>'`.
The GGUF was deleted, moved or renamed, so the server skipped it and kept
serving the rest. Restore the file and the id works again with no restart.
If the file is gone for good, `gmlx sync-models` removes its entry.

A missing [`server.embeddings`](config.md#serverembeddings) or
[`server.rerank`](config.md#serverrerank) file disables that service with a
warning, and its route answers a plain 404 with no error type. That
service stays off until `gmlx restart`, even after the file comes back. A
service file set by an absolute path is the exception. When that file is
missing, its route answers 404 of type `model_file_missing` until the file
is back, and then works with no
restart.

### A model answers as if the message were empty

Replies ignore what you sent, the log shows `the chat template of <file>
drops message text` at load, and a request with a system prompt can fail
with 500. For Qwen3.5 and other model types that also accept images, the
server passes each message to the chat template as a list of parts. Some
fine-tunes ship an older template that renders only plain strings, so every
message arrives empty.

Replace the template with one that renders lists, usually the base model's
template. Pass the file with `--chat-template` on
[gmlx serve](cli.md#gmlx-serve), or set
[`chat_template`](config.md#profileschat_template) in a profile or in the
model's `overrides`.

### Requests get 503

A 503 carries a `Retry-After` header, and its error type says why:

| Type | Cause | What to do |
|------|-------|------------|
| `server_overloaded` | More requests are waiting than the queue cap. | Retry after the given delay, or raise `GMLX_QUEUE_DEPTH_CAP`. |
| `model_load_deferred` | The model cannot load beside the resident models that are busy. | Retry after the delay, or lower what stays resident. |

[Limits and back-pressure](api.md#limits-and-back-pressure) has every limit
a request can hit, including the 400 for a prompt that cannot fit.

### A streamed reply ends with server_overloaded_shed

A streaming reply stops early with an error of type
`server_overloaded_shed`, code `row_shed` and `finish_reason` `shed`. The
memory governor ran out of other ways to free memory, so it shed this
request to keep the others running. Send it again, and read
[Memory](#memory) if it happens often.

## Memory

### The Mac swaps, or a load or reply fails for memory

The whole Mac slows while a model runs, or a command stops with an error.
On `run` and `chat`, a context that cannot fit is refused before the load
with `cannot fit:` and the numbers. A context that grows past memory
during the reply stops with `out of GPU memory mid-run`. Both show how much
the model needs and what the GPU may use.

In each case, the weights plus the KV cache need more memory than the GPU
may use. [Memory and the KV cache](memory.md) shows how to estimate both
and which settings reduce them. The usual fixes are a
[quantized KV cache](kv-quantization.md), a smaller context, a smaller
quant, or [streaming](streaming.md) for a MoE model larger than memory. On a
server with several models, lower
[`server.budget_gb`](config.md#serverbudget_gb) or
[`server.max_models`](config.md#servermax_models). To try anyway, set
[`GMLX_TOOL_PREFLIGHT=0`](env-vars.md#commands), which skips the refusal on
`run` and `chat`.

## Voice

### The mic never works in talk

`gmlx talk` runs but never hears you, and macOS never showed a permission
prompt. macOS grants the microphone to the terminal app that you ran `talk`
from. In System Settings, open Privacy and Security, then Microphone, and
turn on your terminal, whether Terminal, iTerm2 or your editor. If you
dismissed the prompt long ago, turn the entry off and on to get a new one.
`gmlx talk --list-devices` shows whether any input device is visible.

### Transcription or speech fails because ffmpeg is not found

`/v1/audio/transcriptions`, or speech in mp3, flac or opus, answers 500,
and the request line in the server log names the cause. Transcription logs
`audio decoding needs ffmpeg on PATH`, and speech logs `ffmpeg not found`.
Whisper decodes its input through ffmpeg, and speech needs it for every
format except wav and pcm.
Run `brew install ffmpeg`, then `gmlx restart`.

## Distillation

### A distill step fails or the adapter learns nothing

`gmlx distill filter` drops most rows, `align` warns or refuses, the census
effect is small, `train` runs out of memory, or the served adapter scores
near zero. [When something goes wrong](distill.md#when-something-goes-wrong)
in the distillation guide covers each case, and
[gmlx distill](cli.md#gmlx-distill) lists every action's exit codes.

## Logs and files

### Where the logs are

`gmlx logs -n 100` prints the last lines of the background server's log,
and `-f` follows it. Each finished request logs a line with the model, the
token counts and the timing, which is often enough to see what was slow.
`gmlx status` reports the process, and `gmlx ps` lists the resident models.
`gmlx serve --print-config` prints the full config the server would run
with.

### Where files are on disk

gmlx writes to these places. Paths under `~/.cache` and `~/.local/share`
follow `XDG_CACHE_HOME` and `XDG_DATA_HOME` when they are set.

| Path | Contents |
|------|----------|
| `./gmlx.yaml`, `~/.config/gmlx/gmlx.yaml`, `~/.gmlx.yaml` | These hold the config, as [Where gmlx looks](config.md#where-gmlx-looks) describes. |
| `~/.config/gmlx/` | `gmlx launch` writes injected clients' configs here. |
| `~/.pi/agent/`, `~/.omp/agent/`, `~/.config/goose/config.yaml` | `gmlx launch` merges its settings into these files, as [The clients](launch.md#the-clients) describes. To remove gmlx, delete those settings and keep the files. |
| `~/.cache/gmlx/` | It holds server runfiles and logs, chat input history and the GGUF header cache. |
| `~/.cache/gmlx/apc/` | The prompt cache is stored here when the disk tier is on and has no `path` of its own. |
| `~/.cache/gmlx/talk/` | The first `talk` fetches the wake-word and voice-activity models here. |
| `~/.cache/huggingface/` | `hf:` references resolve from these files. |
| `~/.local/share/gmlx/chats/` | Saved chat sessions are kept here. |
| `~/.local/share/gmlx/assistant-memory.db` | It holds the assistant's memory, with `assistant-<id>.db` beside it for each served assistant. |
| `~/Library/Application Support/gmlx/` | The menu bar runs from an app bundle that gmlx writes here. |
| `~/Library/LaunchAgents/com.gmlx.*.plist` | `gmlx service install` writes its login items here. |
| `~/.open-webui/` | Open WebUI keeps its chat history here. |
| Your model folders | `pull` downloads GGUFs into them. |

[Removing gmlx](installation.md#removing-gmlx) gives the steps that remove
gmlx and these files.
