# Troubleshooting

This page is for when something breaks. It lists the failures common in new setups,
each with the symptom, the cause and the fix, plus where the logs and files
are. Client-launch problems, such as a tool that will not connect, are under
[launch.md](launch.md#troubleshooting).

Start with `gmlx doctor`. It checks the runtime, config, model paths,
background server and optional services in one pass, and names the fix for
anything it flags ([gmlx doctor](cli.md#gmlx-doctor)).

| Symptom | Section |
|---------|---------|
| the install fails on macOS before 26 | [The install fails compiling the Metal kernels](#the-install-fails-compiling-the-metal-kernels) |
| the command is not found in a new terminal | [gmlx: command not found in a new terminal](#gmlx-command-not-found-in-a-new-terminal) |
| a download stopped or the disk filled | [A download was interrupted or the disk filled](#a-download-was-interrupted-or-the-disk-filled) |
| a load or validate names an unsupported codec | [A file refuses to load: unsupported codec](#a-file-refuses-to-load-unsupported-codec) |
| a configured model is not listed | [A configured model is missing from /v1/models](#a-configured-model-is-missing-from-v1models) |
| transcription or talk complains about ffmpeg | [Whisper fails: ffmpeg not found](#whisper-fails-ffmpeg-not-found) |
| talk never receives mic input | [The mic never works in talk](#the-mic-never-works-in-talk) |
| serve cannot bind its port | [Port 8080 is already in use](#port-8080-is-already-in-use) |
| the first request takes many seconds | [The first request after startup is slow](#the-first-request-after-startup-is-slow) |
| a request gets 403 hf_access_disabled | [Requests fail with 403 hf_access_disabled](#requests-fail-with-403-hf_access_disabled) |
| Hugging Face answers 401 or 403 | [A gated or private repo will not download](#a-gated-or-private-repo-will-not-download) |
| the machine swaps or the server exits | [Memory pressure: swapping, an unresponsive machine, or a server that exits](#memory-pressure-swapping-an-unresponsive-machine-or-a-server-that-exits) |
| you need the logs and the resolved config | [Where the logs are](#where-the-logs-are) |
| you need to find or remove what gmlx wrote | [Where files are on disk](#where-files-are-on-disk) |

## The install fails compiling the Metal kernels

Symptom: on macOS versions before 26, `pip install` fails partway through
building `mlx-kquant`. Typical messages are a compiler or SDK error, or
`cannot execute tool 'metal'`.

On older macOS the kernels build from source, and the build needs full
Xcode. The C++ parts compile with the Command Line Tools, but the Metal
shaders compile with `xcrun metal`, which the Command Line Tools do not
include. Install Xcode, select it
(`sudo xcode-select -s /Applications/Xcode.app`), and re-run the pip
install. Recent Xcode versions fetch the Metal toolchain as a separate
download: run `xcodebuild -downloadComponent MetalToolchain` once. If the
build still fails after a macOS upgrade, update Xcode so its SDK matches,
and try again. On macOS 26.2 and newer none of this applies: the kernels
install as a prebuilt wheel.

## `gmlx: command not found` in a new terminal

Symptom: `gmlx` worked in an earlier terminal; a new terminal says
`command not found: gmlx` (so `gmlx doctor` is unavailable too).

Nothing is broken. This happens with the plain-venv install route: gmlx
is installed in the Python venv you chose, and each new terminal starts
with that venv inactive. Run `source <install dir>/.venv/bin/activate` (the
directory from the [install step](getting-started.md#install)) and the
command is available again. A background server or menu-bar app keeps running either
way. Only the terminal command needs the venv. An install via `uv tool
install` or pipx stays on PATH in every terminal and never has this problem.

## A download was interrupted or the disk filled

Symptom: `gmlx pull` stopped mid-download, or refused to start with
`error: not enough disk space`.

Interrupted downloads resume: re-run the same `pull` and it continues from
where it stopped (sharded files resume per shard). The disk-space refusal is a
preflight: it names how much the file needs and how much is free. Free space,
pass `--to DIR` on another volume, or `--force` to skip the check. A download
that failed mid-write for another reason (network drop, Hugging Face error) is also
safe to re-run.

## A file refuses to load: unsupported codec

Symptom: `validate`, `pull`, or a load fails naming a tensor codec.

The K-quant, legacy, and IQ families all have kernels in gmlx, as does the
structured-ternary `STQ1_0`, so this is rare: it means the file uses an uncommon
type with none (the plain ternary `TQ1_0`/`TQ2_0` types, for instance). The refusal names the unsupported codec and what is
supported. Pick a different quant from the same repo;
`gmlx validate hf:<org>/<repo>` lists every variant so you can choose without
downloading. Uniform K-quant files also decode fastest
([performance.md](performance.md#choosing-a-quant-for-speed)).

## A configured model is missing from /v1/models

Symptom: an id from your config is not listed, or requesting it returns a 404
with type `model_file_missing`. `gmlx logs` shows
`[server] skipping model '<id>'` at the last startup or reload.

The entry's GGUF is gone from disk (deleted, moved, or renamed), so the server
skipped it and kept serving everything else. Restore the file and the server recovers
with no restart: requests for the id work again and it re-appears in
`/v1/models`. If the file is permanently
gone, `gmlx sync-models` reconciles the config in one pass: entries for missing files
are removed, new files are registered, and your comments and hand-edits are preserved. A missing
`server.embeddings` or `server.rerank` GGUF behaves the same way: the service
is disabled with a log warning and de-listed from `/v1/models` while chat
keeps serving.

## Whisper fails: ffmpeg not found

Symptom: `/v1/audio/transcriptions` errors, or `gmlx talk` fails its capability
check, mentioning ffmpeg.

Whisper decodes input audio through ffmpeg, and TTS needs it for non-wav output
formats. `brew install ffmpeg`, then restart the server.

## The mic never works in talk

Symptom: `gmlx talk` runs but never receives your speech, and no macOS permission prompt ever
appeared.

macOS grants microphone access per app (TCC), keyed to the terminal you ran `talk`
from. Check System Settings, Privacy and Security, Microphone, and enable your
terminal (Terminal.app, iTerm2, or your IDE). If the prompt was dismissed long ago,
toggling the entry off and on triggers a new prompt. `gmlx talk --list-devices` shows
whether an input device is visible at all.

## Port 8080 is already in use

Symptom: `serve` fails to bind, or requests reach some other process.

`gmlx status` shows whether a managed gmlx server already holds the port. If
so, `gmlx stop` (or `gmlx restart` after a config change; a launchd-managed
server stops with `gmlx service uninstall`). If that process is not gmlx,
`lsof -i :8080` names it. Either free the port or serve on another port with
`--port 8081`.

## The first request after startup is slow

Symptom: the server answered immediately, but the first chat completion took many
seconds.

Nothing was preloaded, so the first request included the model load. Set
`server.defaults.model: <id>` in the config, pin one, or pass `--model` to
`gmlx launch`. The
auto-start path then loads the weights before binding the port, and the first turn
does not wait for a load. Distinct from this: the first turn on a very long prompt is prefill, not
loading; see [performance.md](performance.md#the-prompt-cache).

## Requests fail with 403 hf_access_disabled

Symptom: an API request names a model and gets a 403 with `hf_access_disabled`.

The request's `model` is neither a configured id nor a local file, and this server
never downloads on a request. Use an id from `gmlx list`, or add the model to the
config (`gmlx pull` it, then `gmlx sync-models`).

## A gated or private repo will not download

Symptom: `validate` or `pull` gets a 401 or 403 from Hugging Face.

Set `HF_TOKEN` in the environment to a token with access to the repo, then rerun.

## Memory pressure: swapping, an unresponsive machine, or a server that exits

Symptom: the whole machine becomes slow while a model runs, or loads abort.

The weights plus KV cache exceed available RAM. Check the arithmetic in
[performance.md](performance.md#memory-and-the-kv-cache): quantize the KV cache
(`--kv-bits 8`), cap it (`--max-kv-size`), pick a smaller quant, or for over-budget
MoE models use `--stream-cpu` (or `--stream-experts` for long-context work with a
quantized KV cache; see
[streaming.md](streaming.md)). On the
multi-model server, lower `--budget-gb`
or `--max-models` so residency stays under the limit.

## Where the logs are

`gmlx logs -n 100` prints the managed server's log (`-f` follows). The files are
under `~/.cache/gmlx/`. Each completed request logs one line with the model,
token counts, and timing, which is usually enough to see what was slow.
`gmlx status` reports the process, `gmlx ps` the resident models, and
`gmlx serve --print-config` the fully resolved config the server would run with.

## Where files are on disk

| Path | Contents |
|------|----------|
| `~/.config/gmlx/gmlx.yaml` | your config |
| `~/.config/gmlx/` | client configs written by `gmlx launch` |
| `~/.cache/gmlx/` | server runfiles and logs, chat history, the GGUF header cache |
| `~/.cache/gmlx/apc/` | the on-disk prompt cache, when enabled |
| `~/.cache/gmlx/talk/` | wake-word and voice-activity models, fetched on the first `talk` |
| `~/.local/share/gmlx/chats/` | saved chat sessions |
| `~/.local/share/gmlx/assistant-memory.db` | the assistant's long-term memory; served assistants get `assistant-<id>.db` beside it |
| `~/Library/Application Support/gmlx/` | the menu bar app bundle |
| `~/Library/LaunchAgents/com.gmlx.*.plist` | the login items written by `gmlx service install` |
| `~/.open-webui/` | Open WebUI's chat history |
| your model directories | the GGUFs; `pull` writes here |

To remove gmlx completely, run `gmlx service uninstall` if you installed the
login item, delete the directories above and the models you pulled, and
uninstall the `gmlx` and `mlx-kquant` packages the way you installed them.
