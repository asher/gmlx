# Troubleshooting

Run [gmlx doctor](cli.md#gmlx-doctor) first. It checks each part of a
working setup and names the fix for each check that fails. For a coding
agent or chat app that does not connect, read its entry under
[The clients](launch.md#the-clients).

- [Installing](#installing)
- [Downloading and loading models](#downloading-and-loading-models)
- [Starting the server](#starting-the-server)
- [Requests](#requests)
- [Memory](#memory)
- [Container mode](#container-mode)
- [Custom agents](#custom-agents)
- [Voice](#voice)
- [Distillation](#distillation)
- [Logs and files](#logs-and-files)

## Installing

### The install fails on macOS before 26.2

The mlx-kquant kernels need macOS 26.2 or newer, and `gmlx doctor` warns on
an older version. Update macOS in System Settings, then install gmlx again.

### `gmlx: command not found` in a new terminal

A [pip](installation.md#pip) install lives in a Python venv, and each new
terminal starts with that venv inactive. Activate it again:

```sh
source <install dir>/.venv/bin/activate
```

Homebrew and `uv tool install` keep `gmlx` on PATH in every terminal.

## Downloading and loading models

### `not enough disk space`, or a download stopped partway

Run the same `gmlx pull` again. It resumes from the bytes already on disk.

- Not enough disk space: free some, pass `--to DIR` for another volume, or
  pass `--force` to skip the check.
- `stale partial download`: the repo replaced the file. Delete the `.part`
  file that the message names, and pull again.
- A flaky or slow host: raise `GMLX_PULL_RETRIES` or `GMLX_PULL_TIMEOUT`
  ([Command environment variables](env-vars.md#commands)).
- A load says `incomplete split GGUF` or `truncated GGUF`: pull again. A
  pull skips a file that exists, so delete a truncated file first.

### A gated or private repo will not download

Hugging Face answers 401 or 403. Accept the repo's terms on its Hugging Face
page, and check that your token has access to it. gmlx sends `HF_TOKEN`, or
the token that `hf auth login` stored.

### `GGUF architecture 'X' is not supported`

gmlx has no code for this model architecture.
[Supported architectures](arch-coverage.md) lists the ones it runs, and
`gmlx validate` shows a file's architecture before you download it.

### A tensor type is marked `<- no kernel`

The file uses a tensor type that gmlx cannot run, usually the plain ternary
`TQ1_0` or `TQ2_0`. Pick another quant from the same repo.
`gmlx validate hf:<org>/<repo>` lists the files without downloading, and a
uniform K-quant [decodes fastest](performance.md#choosing-a-quant-for-speed).

### A Hadamard-folded file refuses to load

gmlx runs a Hadamard-folded file only as a dense Qwen3.5, 3.6 or 3.8 model
at fold version 1, never as a drafter. Pick an unfolded quant. Before you
download, look for a `weights: Hadamard-folded` line in `gmlx validate`,
because its `loadable` verdict does not cover the fold.

## Starting the server

### `gmlx` no longer reads `./gmlx.yaml`

Move the file to `~/.config/gmlx/gmlx.yaml`. gmlx does not read a config
from the current folder, because a client in a container could write one
there. When a login item still points at the old file, the `login start`
row of `gmlx doctor` gives the commands that fix it.

### `gmlx status` reports 0 models served

The config lists no models, or the server skipped every entry. `gmlx logs`
shows a line for each skip:

- `[server] skipping model`: the entry's file is missing.
- `[server] model_dirs root missing`: a folder in
  [`server.model_dirs`](config.md#servermodel_dirs) does not exist. Use
  absolute paths.

### `binding 0.0.0.0 exposes this server beyond localhost`

A server that the network can reach needs a key in
[`server.api_key`](config.md#serverapi_key). With
[served assistants](assistant.md#served-assistants), anyone with that key
could also run tools on the Mac, so gmlx refuses unless you set
[`server.assistant_allow_remote`](config.md#serverassistant_allow_remote).

### Port 8080 is already in use

`gmlx status` shows whether a gmlx server holds the port, and `gmlx stop`
stops it. A server from `gmlx service install` starts again at login until
`gmlx service uninstall`. For another program, `lsof -i :8080` names it, or
serve on another port with `--port 8081`.

### The first request after startup is slow

That request loaded the model. To load a model at startup,
[pin](config.md#modelspin) it or list it in
[`server.defaults.preload`](config.md#serverdefaultspreload). A slow first
turn on a long prompt is prefill instead, which the
[prompt cache](prompt-cache.md) shortens on later turns.

## Requests

### 404 `model_not_found`

The server never downloads a model on a request. Use an id from
`gmlx list`, which the error also gives in `available_models`, or fetch the
model with `gmlx pull`. A file that `--to` saved outside your model folders
needs `gmlx sync-models --models-dir DIR` or a [`models`](config.md#models)
entry.

A 400 `unknown_profile` means that a suffix such as `@coding` names no
intent or profile. The error lists the valid names.

### 404 `model_file_missing`

The model's GGUF was deleted, moved or renamed, and the log shows
`[server] skipping model '<id>'`. Restore the file, and the id works again
with no restart. When the file is gone for good, `gmlx sync-models` removes
its entry.

A missing [`server.embeddings`](config.md#serverembeddings) or
[`server.rerank`](config.md#serverrerank) file turns that service off. Run
`gmlx restart` once the file is back.

### `the chat template of <file> drops message text`

Replies ignore what you sent. For models that also take images, the server
sends each message as a list of parts, and this template renders only plain
strings. Replace it, usually with the base model's template, through
`--chat-template` on [gmlx serve](cli.md#gmlx-serve) or
[`chat_template`](config.md#profileschat_template) in a profile or in the
model's `overrides`.

### Requests get 503

A 503 carries a `Retry-After` header, and its error type says why:

| Type | Cause | What to do |
|------|-------|------------|
| `server_overloaded` | More requests wait than the queue cap allows. | Retry after the delay, or raise `GMLX_QUEUE_DEPTH_CAP`. |
| `model_load_deferred` | The model cannot load beside the busy resident models. | Retry after the delay, or keep fewer models resident. |

[Limits and back-pressure](api.md#limits-and-back-pressure) lists every
limit a request can hit.

### A request with media gets 400 or 413

The server takes media only inline or from its media folder, as
[Media in requests](api.md#media-in-requests) describes. `gmlx logs` shows
the message, with the command that copies a file into the media folder.

### `server_overloaded_shed`

The memory governor stopped this request to keep the other requests
running. Send it again, and read [Memory](#memory) if it happens often.

### A web page or browser extension gets 403 or a CORS error

The page's origin is not in
[`server.cors_origins`](config.md#servercors_origins). `gmlx logs` names the
origin. Add it to `server.cors_origins` and run `gmlx restart`.

An extension's origin holds an ID that the browser gives it:

- Chrome and Edge show it on the extension's card in `chrome://extensions`
  or `edge://extensions`, with Developer mode on.
- Firefox shows it as the Internal UUID in `about:debugging`, under This
  Firefox.
- Safari changes it at every start, so list `safari-web-extension://*`.

Do not list a website's origin for an extension, because every page on that
site could then call the server. A page opened from disk sends
`Origin: null`, so serve it from `http://localhost:<port>` instead.

## Memory

### `cannot fit:`, `out of GPU memory mid-run`, or the Mac swaps

The weights plus the KV cache need more memory than the GPU may use.
[Memory and the KV cache](memory.md) shows how to estimate both. To fix it:

- Use a [quantized KV cache](kv-quantization.md).
- Use a smaller context or a smaller quant.
- [Stream](streaming.md) a MoE model that is larger than memory.
- On a server with several models, lower
  [`server.budget_gb`](config.md#serverbudget_gb) or
  [`server.max_models`](config.md#servermax_models).

To try anyway on `run` and `chat`, set
[`GMLX_TOOL_PREFLIGHT=0`](env-vars.md#commands).

## Container mode

### A launch stops on a malformed launch block

The `launch` block of your config does not load, and the message names the
file and the problem. Fix the block, or pass `--no-container` to run the
client on the Mac meanwhile. The server ignores a broken `launch` block and
loads the rest of the file.

### Launch refuses a mount through a symbolic link

A `--mount` or [`mounts`](config.md#launchcontainermounts) path is, or passes
through, a symbolic link. Write the real path that the message gives
instead, for example `/private/tmp/x` for `/tmp/x`.

### No Mac port is free for a browser app

`no Mac port from 3100 to 3199 is free` means other projects keep every
browser app port. The message names projects you used longest ago, each with
the `--remove-home` command that frees its port. Run one, or stop another
program that uses a port, and launch again.

### Launch says Apple container has no Linux kernel

Apple container needs a Linux kernel, about 700 MB, before any container can
start. A launch in a terminal asks to download it. A launch from a script
has no terminal to ask in, so it stops and names one command to run:

- `container system start --enable-kernel-install` when the container
  service is not running;
- `container system kernel set --recommended` when it runs.

Run it, then launch again. If you answered no, or the download failed, the
next launch in a terminal asks again.

### A container command gave no answer

The container service is stuck. Run `container system stop`, then
`container system start`, and try again.

### A leftover container of another session keeps running

A launch that was killed left its container running, and the message gives
the `container stop` command. The container holds its memory until it
stops, so run that command. `gmlx launch --list` lists these containers too.

### A volume is in use

[A volume serves one container at a time](container-access.md#volumes).
Stop the other session first. To give each project its own volume, list the
volume under the client rather than directly under `launch.container`.

### Apple's image builder cannot start without Rosetta

Apple container starts its image builder with Rosetta by default, and this
Mac has no Rosetta. `launch` turns that setting off in
`~/.config/container/config.toml`. Run `container system stop` so the
service reads the change, and launch again.

### Launch refuses to build while the builder forwards your SSH agent

The image builder was started with your SSH agent, which every build could
use. Run `container builder stop`, and the next launch starts a builder
without it.

### The image build fails

One step of the image build failed, and the build output above the message
shows which. For a package in
[`packages`](config.md#launchcontainerclientspackages), check that Debian 13
has a package by that name. For a step in your own Containerfile, fix the
file. Otherwise launch again with `--rebuild`.

### The image build cannot reach the network

A VPN that sends all traffic through its tunnel is the usual cause. The Mac
stays online, but containers get no connection out. Disconnect the VPN, or
turn on its setting that allows local network access, and launch again. A
running client also has no internet while the VPN is on, but it still
reaches the gmlx server.

### An image has no `linux/arm64` variant

Container mode runs only Linux on arm64 images. Use an arm64 or
multi-platform tag, or build one with
[`build`](config.md#launchcontainerclientsbuild).

### A command is not in the image

The image lacks the client or the command in
[`command`](config.md#launchcontainerclientscommand). Install it in the
image, as [Custom container images](container-images.md) shows, or fix the
`command` list. Other messages name a specific problem:

| Message | Fix |
|---------|-----|
| `has no execute bit` | Add `RUN chmod 755 <file>` to the Containerfile. |
| `names X in its #! line, which is not in the image` | Install the interpreter, or change the script's `#!` line. |
| `ends in a carriage return` | Convert the script to Unix line endings, for example with `dos2unix`. |
| `env receives it as one command name` | Write `#!/usr/bin/env -S tool --flag`. |
| `Exec format error` | Install an arm64 build, or add a `#!` line to the script. |
| `its #! interpreter or its program loader is not in the image` | The program is built for another system, such as glibc in a musl image. |

### Launch will not follow a file in the private home

The client can put links in its [private home](container-access.md#the-private-home),
so `launch` never follows one there. Delete the path the message names, or
remove the home with `gmlx launch <client> --remove-home`, and launch again.

### A pasted file path reaches the client unchanged

`launch` left the path as it was. The session log,
`~/.cache/gmlx/launch/last-<client>-<project>.log`, has a `paste:` line with
the reason. Folders, links, files in protected folders and paths inside
other text stay as they are, as
[Pasting files and images](container-access.md#pasting-files-and-images)
lists. If macOS denied your terminal app access to the file, allow it in
Privacy & Security in System Settings and paste again.

### The clipboard image paste fails in a container

The client read the clipboard without a press of its paste key first. Press
Ctrl-V, or Alt-V in hermes, in the session's terminal, and the client can
read one image.

### A container launch waits with no output

macOS is asking whether the container may read a protected folder, such as
`~/Documents`. Look for the prompt behind other windows, or launch from a
folder outside the protected places.

### A request from a container gets 403 peer_not_allowed

An Apple container localhost domain sends container traffic to the Mac's
`127.0.0.1`, and the server refuses it. Launch sessions need no such domain,
so remove it with `sudo container system dns delete <domain>`.

### The Mac runs out of file handles

A session that reads a very large shared folder holds a Mac file handle for
each file. Stop the session to release them, and share a narrower folder.

### Postgres refuses the data folder on a share

Every file in a share belongs to root in the container, which Postgres does
not accept. Put the data on a volume, as the
[Postgres](container-recipes.md#postgres) recipe shows.

### Packages installed in `--shell` are gone at the next launch

Only the project folder, the private home and volumes are kept. Add the
package to the image with [`packages`](container-images.md#extra-packages).

## Custom agents

### An agent name is refused

The name breaks a rule of [`launch.agents`](config.md#launchagents), such as
being a client's name or longer than 32 characters. The message names the
rule. Rename the agent.

### An agent cannot start because the container has no command

The `command` does not match a script that uv installed. Make
[`command`](config.md#launchagentscommand) match an entry in the project's
`[project.scripts]`. If the project is not a package, uv installs no
scripts, so add a `[build-system]` table, as `uv init --package` writes it.

### A source elsewhere fails with No module named

The project is not a package, so Python cannot import it from another
folder. Add a `[build-system]` table and a `[project.scripts]` entry, as
`uv init --package` writes them, or launch from the project folder.

### A read-only source fails with Read-only file system

The build backend, usually setuptools, writes an `.egg-info` folder into
the read-only source. Switch the project's `[build-system]` to hatchling or
uv_build, or launch from the source folder.

### uv says the lockfile needs to be updated

The [source folder](launch-agents.md#the-source-folder) is read-only, so uv
cannot update `uv.lock`. Launch the agent once from its source folder, where
uv can write the lock, then launch it as before.

### An agent's first launch fails under network none

uv cannot install the environment without a network. Launch once with
`--network default` in each project folder, then turn the network off. See
[Offline launches](launch-agents.md#offline-launches).

## Voice

### The mic never works in talk

macOS gives the microphone to the terminal app that you run `gmlx talk`
from. In System Settings, open Privacy and Security, then Microphone, and
turn on your terminal app. If you dismissed the prompt long ago, turn the
entry off and on again. `gmlx talk --list-devices` shows whether any input
device is visible.

### Transcription or speech fails because ffmpeg is not found

Transcription, or speech in mp3, flac or opus, answers 500. Install ffmpeg,
and send the request again with no restart:

```sh
brew install ffmpeg
```

The server never runs an ffmpeg from a folder that a container session
shares or shared read-write, and the log names each folder it skips. Once
you trust a folder that only an earlier session shared, run
`gmlx launch --forget-share PATH`, as
[The share history](container-security.md#the-share-history) explains.

## Distillation

### A distill step fails or the adapter learns nothing

[Distillation reports and troubleshooting](distill-troubleshooting.md)
covers each step, and [gmlx distill](distill-reference.md) lists the exit
codes.

## Logs and files

### Where the logs are

```sh
gmlx logs -f                # Follows the background server's log.
gmlx serve --print-config   # Prints the full config the server would run with.
```

Each finished request logs a line with the model, the token counts and the
timing. `gmlx ps` lists the resident models.

### Where files are on disk

Paths under `~/.cache` and `~/.local/share` follow `XDG_CACHE_HOME` and
`XDG_DATA_HOME` when they are set.

| Path | Contents |
|------|----------|
| `~/.config/gmlx/gmlx.yaml`, `~/.gmlx.yaml` | The config, as [Where gmlx looks](config.md#where-gmlx-looks) describes. |
| `~/.config/gmlx/` | Client configs that `gmlx launch` writes. |
| `~/.pi/agent/`, `~/.omp/agent/`, `~/.config/goose/config.yaml`, `~/.hermes/config.yaml` | Client files that `gmlx launch` adds its settings to. Delete those settings to remove gmlx. |
| `~/.cache/gmlx/` | Server runfiles and logs, chat input history and the GGUF header cache. |
| `~/.cache/gmlx/apc/` | The prompt cache, when the disk tier is on and has no `path` of its own. |
| `~/.cache/gmlx/media/` | Media files that requests name, as [Media in requests](api.md#media-in-requests) describes. |
| `~/.cache/gmlx/talk/` | The wake-word and voice-activity models of `gmlx talk`. |
| `~/.cache/huggingface/` | The files that `hf:` references resolve from. |
| `~/.local/share/gmlx/chats/` | Saved chat sessions. |
| `~/.local/share/gmlx/assistant-memory.db` | The assistant's memory, with an `assistant-<id>.db` beside it for each served assistant. |
| `~/Library/Application Support/gmlx/` | The menu bar app bundle. |
| `~/Library/LaunchAgents/com.gmlx.*.plist` | Login items from `gmlx service install`. |
| `~/.open-webui/` | Open WebUI chat history. |
| `~/.local/share/gmlx/launch/` | Container mode's private homes, browser app ports, locks and records. |
| `~/.cache/gmlx/launch/` | Container mode's session logs, detached session output and session folders. |
| Your model folders | The GGUFs that `pull` downloads. |

[Removing gmlx](installation.md#removing-gmlx) gives the steps that remove
gmlx and these files.
