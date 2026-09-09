# Troubleshooting

This page is for when something breaks. It lists the failures common in new
setups, each with the symptom, the cause and the fix, plus where the logs and
files are. A coding agent or chat app that will not connect is covered
with its client in [launch.md](launch.md#the-clients).

Start with [gmlx doctor](cli.md#gmlx-doctor), which checks the runtime,
config, model paths, background server and optional services in one pass,
and names the fix for anything it flags.

| Symptom | Section |
|---------|---------|
| the install fails on macOS before 26.2 | [The install fails compiling the Metal kernels](#the-install-fails-compiling-the-metal-kernels) |
| the command is not found in a new terminal | [gmlx: command not found in a new terminal](#gmlx-command-not-found-in-a-new-terminal) |
| a download stopped or the disk filled | [A download was interrupted or the disk filled](#a-download-was-interrupted-or-the-disk-filled) |
| a load or validate names an unsupported codec | [A file refuses to load with an unsupported codec](#a-file-refuses-to-load-with-an-unsupported-codec) |
| a configured model is not listed | [A configured model is missing from /v1/models](#a-configured-model-is-missing-from-v1models) |
| transcription or talk complains about ffmpeg | [Whisper fails because ffmpeg is not found](#whisper-fails-because-ffmpeg-is-not-found) |
| talk never receives mic input | [The mic never works in talk](#the-mic-never-works-in-talk) |
| serve cannot bind its port | [Port 8080 is already in use](#port-8080-is-already-in-use) |
| the first request takes many seconds | [The first request after startup is slow](#the-first-request-after-startup-is-slow) |
| a request gets 403 hf_access_disabled | [Requests fail with 403 hf_access_disabled](#requests-fail-with-403-hf_access_disabled) |
| Hugging Face answers 401 or 403 | [A gated or private repo will not download](#a-gated-or-private-repo-will-not-download) |
| the machine swaps or the server exits | [Memory pressure, swapping, or a server that exits](#memory-pressure-swapping-or-a-server-that-exits) |
| you need the logs and the resolved config | [Where the logs are](#where-the-logs-are) |
| you need to find or remove what gmlx wrote | [Where files are on disk](#where-files-are-on-disk) |

## The install fails compiling the Metal kernels

On macOS versions before 26.2, `pip install` fails partway through
building `mlx-kquant`. Typical messages are a compiler or SDK error, or
`cannot execute tool 'metal'`.

On older macOS the kernels build from source, and that build needs full
Xcode: the C++ parts compile with the Command Line Tools, but the Metal
shaders compile with `xcrun metal`, which the Command Line Tools do not
include. Install Xcode, select it with
`sudo xcode-select -s /Applications/Xcode.app` and re-run the pip install.
Recent Xcode versions fetch the Metal toolchain as a separate download, so
run `xcodebuild -downloadComponent MetalToolchain` once. If the build still
fails after a macOS upgrade, update Xcode so its SDK matches and try again.
On macOS 26.2 and newer none of this applies, because the kernels install
as a prebuilt wheel.

## `gmlx: command not found` in a new terminal

`gmlx` worked in an earlier terminal, but a new terminal says `command not
found: gmlx`. That leaves `gmlx doctor` unavailable too.

Nothing is broken. This happens with the plain-venv install route, where
gmlx is installed in the Python venv you chose and each new terminal starts
with that venv inactive. Run `source <install dir>/.venv/bin/activate`,
using the directory from the [install step](getting-started.md#install),
and the command is available again. A background server or menu-bar app
keeps running either way, since only the terminal command needs the venv.
An install via `uv tool install` or pipx stays on PATH in all terminals and
never has this problem.

## A download was interrupted or the disk filled

`gmlx pull` stopped mid-download, or refused to start with
`error: not enough disk space`.

Interrupted downloads resume: re-run the same `pull` and it continues from
where it stopped, shard by shard for sharded files. The disk-space refusal
is a preflight check that names how much the file needs and how much is
free, so free some space, pass `--to DIR` on another volume, or `--force`
to skip the check. A download that failed mid-write for another reason,
such as a network drop or a Hugging Face error, is also safe to re-run.

## A file refuses to load with an unsupported codec

`validate`, `pull`, or a load fails and names a tensor codec.

The file uses a tensor type with no kernel. That is rare, because the
K-quant, legacy and IQ families all have kernels, as does the
structured-ternary `STQ1_0`, and the usual culprits are the plain ternary
`TQ1_0` and `TQ2_0` types. The refusal names the unsupported codec and what
is supported. Pick a different quant from the same repo.
`gmlx validate hf:<org>/<repo>` lists the variants so you can choose
without downloading, and a uniform K-quant file also
[decodes fastest](performance.md#choosing-a-quant-for-speed).

## A configured model is missing from /v1/models

An id from your config is not listed, or requesting it returns a 404 with type
`model_file_missing`. `gmlx logs` shows `[server] skipping model '<id>'` at
the last startup or reload.

The entry's GGUF is gone from disk, deleted, moved or renamed, so the
server skipped it and kept serving everything else. Restore the file and
the server recovers with no restart: requests for the id work again and it
re-appears in `/v1/models`. If the file is permanently gone,
`gmlx sync-models` reconciles the config in one pass, removing entries for
missing files, registering new files and preserving your comments and
hand-edits. A missing `server.embeddings` or `server.rerank` GGUF is
handled the same way. The service is disabled with a warning and dropped
from `/v1/models`, and chat keeps serving.

## Whisper fails because ffmpeg is not found

`/v1/audio/transcriptions` errors, or `gmlx talk` fails its capability check.
The message mentions ffmpeg.

Whisper decodes input audio through ffmpeg, and TTS needs it for non-wav
output formats. Run `brew install ffmpeg`, then restart the server.

## The mic never works in talk

`gmlx talk` runs but never receives your speech. No macOS permission prompt
ever appeared.

macOS grants microphone access to each app through TCC, keyed to the
terminal you ran `talk` from. Check System Settings, Privacy and Security,
Microphone, and enable your terminal, whether Terminal.app, iTerm2 or your
IDE. If the prompt was dismissed long ago, toggling the entry off and on
triggers a new prompt. `gmlx talk --list-devices` shows whether an input
device is visible at all.

## Port 8080 is already in use

`serve` fails to bind, or requests reach some other process.

`gmlx status` shows whether a managed gmlx server already holds the port.
If so, run `gmlx stop`, or `gmlx restart` after a config change. A
launchd-managed server comes back at login, so stop that one with
`gmlx service uninstall`. If the process is not gmlx, `lsof -i :8080`
names it, and you can either free the port or serve on another with
`--port 8081`.

## The first request after startup is slow

The server answered immediately, but the first chat completion took many
seconds.

Nothing was preloaded, so the first request carried the whole model load.
The server begins loading a model the moment it starts when the config pins
one, names one in `server.defaults.model` or holds exactly one, and
`server.defaults.preload` warms further ids after it. The port answers while
that load runs, so an early request waits only for what is left of it. The
keys are under [Memory and residency](server-config.md#memory-and-residency).
A slow first turn on a very long prompt is a different case: that is prefill
rather than loading, and the [prompt cache](performance.md#the-prompt-cache)
covers it.

## Requests fail with 403 hf_access_disabled

An API request names a model and gets a 403 with `hf_access_disabled`.

The request's `model` is neither a configured id nor a local file, and
this server never downloads on a request. Use an id from `gmlx list`, or
fetch the model with `gmlx pull`, which registers it in the config when it
lands under a `model_dirs` root. A file saved elsewhere with `--to` needs
`gmlx sync-models` or a `models` entry.

## A gated or private repo will not download

`validate` or `pull` gets a 401 or 403 from Hugging Face.

Set `HF_TOKEN` in the environment to a token with access to the repo, then
rerun.

## Memory pressure, swapping, or a server that exits

The whole machine becomes slow while a model runs, or loads abort.

The weights plus KV cache exceed available RAM. Check the arithmetic in
[performance.md](performance.md#memory-and-the-kv-cache), then either
quantize the KV cache or pick a smaller quant. On `run` and `chat` the KV
flags are `--kv-bits 8` and `--max-kv-size`. On the server the same
settings are the `kv_bits` and `max_kv_size`
[load keys](server-config.md#load-keys) under a profile or a model. For an
over-budget MoE model use `--stream-cpu`, or `--stream-experts` for
long-context work with a quantized KV cache, as [streaming.md](streaming.md)
describes. On a multi-model server, lower `--budget-gb` or `--max-models` so
that residency stays under the limit.

## Where the logs are

`gmlx logs -n 100` prints the managed server's log and `-f` follows it,
with the files under `~/.cache/gmlx/`. Each completed request logs a line
with the model, token counts and timing, which is usually enough to see
what was slow. `gmlx status` reports the process, `gmlx ps` the resident
models and `gmlx serve
--print-config` the fully resolved config the server would run with.

## Where files are on disk

| Path | Contents |
|------|----------|
| `~/.config/gmlx/gmlx.yaml` | your config |
| `~/.config/gmlx/` | client configs written by `gmlx launch` |
| `~/.cache/gmlx/` | server runfiles and logs, chat history, the GGUF header cache |
| `~/.cache/gmlx/apc/` | the on-disk prompt cache, when enabled |
| `~/.cache/gmlx/talk/` | wake-word and voice-activity models, fetched on the first `talk` |
| `~/.local/share/gmlx/chats/` | saved chat sessions |
| `~/.local/share/gmlx/assistant-memory.db` | the assistant's long-term memory, with `assistant-<id>.db` beside it for served assistants |
| `~/Library/Application Support/gmlx/` | the menu bar app bundle |
| `~/Library/LaunchAgents/com.gmlx.*.plist` | the login items written by `gmlx service install` |
| `~/.open-webui/` | Open WebUI's chat history |
| your model directories | the GGUFs, where `pull` writes |

To remove gmlx completely, first run `gmlx service uninstall` if you
installed the login item, then delete the directories in the table and the
models you pulled, and uninstall the `gmlx` and `mlx-kquant` packages the
way you installed them.
