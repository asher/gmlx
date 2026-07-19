# Troubleshooting

Start with `gmlx doctor`: it checks the runtime, config, model paths,
background server, and optional services in one pass, and names the fix for
anything it flags ([reference](cli.md#gmlx-doctor)).

The rest of this page covers the failures new setups actually hit, with the
diagnostic and the fix. Client-launch problems (a tool will not connect or
install) live in [launch.md](launch.md#troubleshooting); everything below is
the runtime and server.

## The install fails compiling the Metal kernels

Symptom: on macOS versions before 26, `pip install` fails partway through
building `mlx-kquant`, usually with a compiler or SDK error.

On older macOS the kernels build from source, which needs Apple's Xcode
Command Line Tools. If the build cannot find a compiler, install them
(`xcode-select --install`) and re-run the pip install. If you upgraded macOS
recently, the tools may be stale for the new SDK: reinstall them
(`sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install`)
and try again. On macOS 26 and newer none of this applies - the kernels
arrive as a prebuilt wheel.

## `gmlx: command not found` in a new terminal

Symptom: `gmlx` worked yesterday; a fresh terminal says
`command not found: gmlx` (so `gmlx doctor` is unavailable too).

Nothing is broken - gmlx lives in the Python venv you installed it into, and
each new terminal starts with that venv inactive. Run
`source <install dir>/.venv/bin/activate` (the directory from the
[install step](getting-started.md#install)) and the command is back. A
background server or menu-bar app keeps running either way; only the terminal
command needs the venv.

## A download was interrupted or the disk filled

Symptom: `gmlx pull` stopped mid-download, or refused to start with
`error: not enough disk space`.

Interrupted downloads resume: re-run the same `pull` and it continues from
where it stopped (sharded files resume per shard). The disk-space refusal is a
preflight - it names how much the file needs and how much is free; free space,
pass `--to DIR` on another volume, or `--force` to skip the check. A download
that failed mid-write for another reason (network drop, HF hiccup) is also
safe to re-run.

## A file refuses to load: unsupported codec

Symptom: `validate`, `pull`, or a load fails naming a tensor codec.

The K-quant, legacy, and IQ families all have kernels here, so this is rare: it
means the file uses an exotic type with none (the ternary `TQ1_0`/`TQ2_0`
types, for instance). The refusal names the offending codec and what is
supported. Pick a different quant from the same repo;
`gmlx validate hf:<org>/<repo>` lists every variant so you can choose without
downloading. Uniform K-quant files also decode fastest
([performance.md](performance.md#choosing-a-quant-for-speed)).

## A configured model is missing from /v1/models

Symptom: an id from your config is not listed, or requesting it returns a 404
with type `model_file_missing`; `gmlx logs` shows
`[server] skipping model '<id>'` at the last startup or reload.

The entry's GGUF is gone from disk (deleted, moved, or renamed), so the server
skipped it and kept serving everything else. Restore the file and it heals
with no restart: requests for the id work again and it re-appears in
`/v1/models`. If the file is gone for
good, `gmlx sync-models` reconciles the config in one pass: dead entries
drop, new files register, your comments and hand-edits survive. A missing
`server.embeddings` / `server.rerank` GGUF behaves the same way - the service
is disabled with a log warning and de-listed from `/v1/models` while chat
keeps serving.

## Whisper fails: ffmpeg not found

Symptom: `/v1/audio/transcriptions` errors, or `gmlx talk` fails its capability
check, mentioning ffmpeg.

Whisper decodes input audio through ffmpeg, and TTS needs it for non-wav output
formats. `brew install ffmpeg`, then restart the server.

## The mic never works in talk

Symptom: `gmlx talk` runs but never hears you, and no macOS permission prompt ever
appeared.

macOS grants microphone access per app (TCC), keyed to the terminal you ran `talk`
from. Check System Settings, Privacy and Security, Microphone, and enable your
terminal (Terminal.app, iTerm2, or your IDE). If the prompt was dismissed long ago,
toggling the entry off and on forces a fresh one. `gmlx talk --list-devices` shows
whether an input device is visible at all.

## Port 8080 is already in use

Symptom: `serve` fails to bind, or requests reach some other process.

`gmlx status` shows whether a managed gmlx server already holds the port; if
so, `gmlx stop` (or `gmlx restart` after a config change; a launchd-managed
server stops with `gmlx service uninstall`). If the holder is not gmlx,
`lsof -i :8080` names it. Either free the port or serve elsewhere with
`--port 8081`.

## The first request after startup is slow

Symptom: the server answered immediately, but the first chat completion took many
seconds.

Nothing was preloaded, so the first request paid the model load. Set
`server.defaults.model: <id>` in the config, pin one, or pass `--model` to
`gmlx launch`; the
auto-start path then loads the weights before binding the port, and the first turn
is hot. Distinct from this: the first turn on a very long prompt is prefill, not
loading; see [performance.md](performance.md#the-prompt-cache).

## Requests fail with 403 hf_access_disabled

Symptom: an API request names a model and gets a 403 with `hf_access_disabled`.

The request's `model` is neither a configured id nor a local file, and this server
never downloads on a request. Use an id from `gmlx list`, or add the model to the
config (`gmlx pull` it, then `gmlx sync-models`).

## A gated or private repo will not download

Symptom: `validate` or `pull` gets a 401 or 403 from Hugging Face.

Set `HF_TOKEN` in the environment to a token with access to the repo, then rerun.

## Memory pressure: swapping, beachballs, or a dying server

Symptom: the whole machine turns sluggish while a model runs, or loads abort.

The weights plus KV cache exceed comfortable RAM. Check the arithmetic in
[getting-started.md](getting-started.md#will-it-fit): quantize the KV cache
(`--kv-bits 8`), cap it (`--max-kv-size`), pick a smaller quant, or for over-budget
MoE models use `--stream-cpu` (or `--stream-experts` for long-context work with a
quantized KV cache; see
[performance.md](performance.md#bigger-than-memory-moe-offload)). On the
multi-model server, lower `--budget-gb`
or `--max-models` so residency stops short of the ceiling.

## Where the evidence lives

`gmlx logs -n 100` prints the managed server's log (`-f` follows); the files sit
under `~/.cache/gmlx/`. Each completed request logs one line with the model,
token counts, and timing, which is usually enough to see what was slow.
`gmlx status` reports the process, `gmlx ps` the resident models, and
`gmlx serve --print-config` the fully resolved config the server would run with.
