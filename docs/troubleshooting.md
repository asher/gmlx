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
- [Container mode](#container-mode)
- [Voice](#voice)
- [Distillation](#distillation)
- [Logs and files](#logs-and-files)

## Installing

### The install fails on macOS before 26.2

gmlx needs macOS 26.2 or newer, because mlx-kquant's Metal kernels are built
for that version. On an earlier version, the install fails or the
kernels cannot run. `gmlx doctor` prints the macOS version and warns
below 26.2. Update macOS in System Settings, then install gmlx again.

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

`gmlx serve` stops with `No gmlx config yet.` when no config exists in the
[default locations](config.md#where-gmlx-looks) and the command names no
GGUF, `--config` or `--models-dir`. `gmlx sync-models` also needs a config,
and stops with `no config found in the default locations`. Run
[`gmlx init`](config.md#create-the-file) to write `~/.config/gmlx/gmlx.yaml`,
or serve one model with `gmlx serve <file.gguf>`.

A login item that starts `gmlx serve` with no config exits at login in the
same way. The `login start` row of `gmlx doctor` names the item. Run
`gmlx init` to give it a config, or remove the item with the
`gmlx service uninstall` command that the row gives. That command also
removes the menu bar's login item.

After `gmlx init`, a headless item stays stopped until the next login. To
start it at once, run the `launchctl kickstart` command that the row gives,
or log out and log in again.

### `gmlx` no longer reads `./gmlx.yaml`

A command run from a folder that holds a `gmlx.yaml`, with no config in your
home folder, prints
`gmlx no longer reads ./gmlx.yaml. Move it to ~/.config/gmlx/gmlx.yaml to use it.`
A file in a project folder can name commands the server runs, and a client
in a container can write one, so gmlx does not read it. Move the file to
`~/.config/gmlx/gmlx.yaml`, and the line stops.

Login items that [`gmlx service install`](cli.md#gmlx-service) set up from
that folder keep the old relative path, so the server does not start at
login. A headless item logs `A login start cannot find a relative --config`
and stays stopped. The `login start` row of `gmlx doctor` warns about it and
gives the steps that point the login item at the moved file.

For the menu bar's item, the steps are `gmlx stop`, then
`gmlx service install`. A headless item needs only
`gmlx service install --headless`, since `gmlx stop` refuses a server that
launchd runs. The row adds the item's `--port` when it is not 8080, and its
`--host` when it is not 127.0.0.1. Run the steps as the row writes them.

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

### A request with media gets 400 or 413

The server takes an image, audio or video only as inline data or from its
media folder, as [Media in requests](api.md#media-in-requests) describes.
It answers 400 to a file path, a URL or an image that it cannot read, and
413 to a body over its limit. The message names the cause. When a client
hides the message, `gmlx logs` shows it after
`[server] refused a request with status` and the status.

Send the media inline, or copy the file into the media folder with the
command that the message gives. A route of a service that is not
configured gets 404 with a line of the same kind, which names the config
key to set.

### A streamed reply ends with server_overloaded_shed

A streaming reply stops early with an error of type
`server_overloaded_shed`, code `row_shed` and `finish_reason` `shed`. The
memory governor ran out of other ways to free memory, so it shed this
request to keep the others running. Send it again, and read
[Memory](#memory) if it happens often.

### A web page or browser extension gets 403 or a CORS error

The server refused the request with status 403 and the error type
`origin_not_allowed`, because its origin is not in
[`server.cors_origins`](config.md#servercors_origins). A web page cannot
read that answer, so the browser console shows a CORS error instead. An
extension with permission for the server's address sees the 403 itself.

`gmlx logs` shows a line `[server] refused a request with status 403` that
names the origin and the entry to add. Add the entry it names to
`server.cors_origins`, then run `gmlx restart`. An extension's origin holds
an ID that the browser gives it, and each browser shows that ID in its own
place:

- Chrome shows it on the extension's card in `chrome://extensions` when
  Developer mode is on, and Edge does the same in `edge://extensions`.
- Firefox shows it as the Internal UUID in `about:debugging`, under This
  Firefox.
- Safari gives the extension a new ID each time Safari starts, so list
  `safari-web-extension://*` instead, as `server.cors_origins` describes.

When the refused origin is a website while you use an extension, the
extension makes the call from a script inside that page, and the request
carries the page's origin. Do not list that site, since every page on it
could then call the server.

A page opened from disk sends `Origin: null`, which no entry can allow.
Serve the page from a loopback address instead, such as
`http://localhost:8000`, since loopback pages need no entry.

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

## Container mode

### Launch says the container service is not running

The first start of the container service asks whether to install a Linux
kernel. A container launch from a script or another program has no
terminal for that question, so it exits 69 and names the command. Run
`container system start` once in a terminal, answer its question about the
kernel, and launch again. Later launches start a stopped service
themselves, with or without a terminal, such as after a Mac restart.

### Launch says Apple container has no Linux kernel

The container service runs, but its first start ended without a kernel,
after a no at the kernel question, a failed download or a Ctrl-C. No
container can start, so launch stops, also in a dry run, and `gmlx doctor`
reports `the container service runs with no Linux kernel`. Run
`container system kernel set --recommended`, and launch again.

A first start that fails before the service answers stops with
`The container service does not answer` instead. Read the service log with
`container system logs`, fix the cause, and launch again.

### A container launch waits with no output

macOS is asking whether the container runtime may read a
[protected folder](launch-container.md#shares) that the session shares,
and the container waits for the answer. Look for the prompt behind other
windows, or launch from a project folder outside the protected places.

### A command is not in the image

A launch that stops with `is not on the image's PATH` names the command
and the search path it used. The image lacks the client or the command in
[`command`](config.md#launchcontainerclientscommand). Install it in the
image, as [Custom container images](container-images.md) shows, or fix the
`command` list.

The message `has no execute bit` means that launch found the file but
cannot run it. Add `RUN chmod 755` for that file to the Containerfile, as the
[start script example](container-images.md#starting-services-with-the-client)
does.

A launch that stops with `names X in its #! line, which is not in the
image` found a script whose interpreter, such as `python3`, is missing.
Install the interpreter in the image, or change the script's `#!` line. The
message `ends in a carriage return` means the script has Windows line
endings. Convert it to Unix line endings, for example with `dos2unix`.

Without `-S`, env receives everything after its name as one command name,
so `#!/usr/bin/env tool --flag` looks for a command called `tool --flag`.
The message `env receives it as one command name` reports this case. Write
`#!/usr/bin/env -S tool --flag` instead.

Launch does not check an `env -S` line that sets `PATH=`, changes folder
with `-C`, uses `-P`, or names the command through a variable such as
`${TOOLDIR}/tool`. A mistake in such a line shows only when the session
starts the script, with the error that the container's own exec gives. Run
the script once from `--shell` to see that error.

When the session itself cannot start the command, it exits 127 or 126 with
the same messages. Exit 126 with `its #! interpreter or its program loader
is not in the image` means the file is a program for another system, such
as a build for glibc in a musl image. Exit 126 with `Exec format error`
means the file is an x86_64 build or a script with no `#!` line. Install an
arm64 build, or add a `#!` line to the script.

### A request from a container gets 403 peer_not_allowed

The server on a loopback address refused a request that came from another
address through a redirect. A localhost domain of Apple container adds
such a redirect, and its 403 of type `peer_not_allowed` names the address.

Launch and `gmlx doctor` warn while such a domain exists, as
[Access you turn on](container-security.md#access-you-turn-on) explains. A
client in a launch session needs no domain, so remove it with
`sudo container system dns delete <domain>`.

When the 403 says that the server `cannot read the address that this
request came from`, the connection closed before the server could read its
address. A container client that resets its connection can cause this.
Send the request again from the Mac.

### A container command gave no answer

A launch or `gmlx doctor` that reports `gave no answer` found the
container service stuck. Run `container system stop` and then
`container system start`, and try again.

### The image build fails

A launch that stops with `the build of the <client> image failed` ran the
build of the image that gmlx makes for the client. With
[`build`](config.md#launchcontainerclientsbuild), the message names your
Containerfile instead, as `the build of <path> failed`. In both cases one
step of the build exited with an error, and the build output above the
message shows the step and its error.

When that step installs a Debian package from
[`packages`](config.md#launchcontainerclientspackages), check the package
name, since Debian bookworm may not have it. Fix or remove the entry and
launch again. When the step is in your own Containerfile, fix it in that
file. For any other step, launch again with `--rebuild`, which builds the
image without its cache.

### The image build needs Rosetta

A launch that stops with `the image builder needs Rosetta` started Apple's
image builder, which uses Rosetta, on a Mac where Rosetta is not installed.
macOS asks once whether to install it, and this message follows a refusal.
Run `softwareupdate --install-rosetta --agree-to-license`, and launch
again.

### The image build cannot reach the network

A launch that stops with `the image build could not reach the network from
the container` found that npm, apt, pip or curl in the build could not look
up a host. A VPN that routes all traffic through its tunnel is the usual
cause. The Mac stays online and still pulls images, but a container gets no
connection out.

Disconnect the VPN, or turn on its setting that allows local network
access, and launch again. While such a VPN is connected, a running client
has no internet either, so its web fetches fail. Its connection to the gmlx
server does not use the network and keeps working.

### An image has no `linux/arm64` variant

Launch names the platforms the image has, and
[container mode](launch-container.md#the-image) runs only Linux on arm64.
Use an arm64 or multi-platform tag of the image, or build one with
[`build`](config.md#launchcontainerclientsbuild).

### No Mac port is free for a browser app

A launch that stops with `no Mac port from 3100 to 3199 is free` found each
port of the [browser apps](launch-container.md#browser-apps) kept by
another project or used by another program. When other projects keep the
ports, the message names up to three projects used longest ago, each with
the step that removes its private home and so frees its port.

Run each step where the message says. A step with `--mount .` runs in the
project's folder, the step for the `default` project of dsh runs in `/`, and
the step for Open WebUI runs in any folder. For a project whose folder no
longer exists, the step is `rm -rf` of the project's folder under
`~/.local/share/gmlx/launch`, because launch finds a project by its folder.

A message that names no project means that other programs use the ports.
Stop one of them, then launch again. `gmlx doctor` lists the private homes,
newest first, with the folder, size and last use of each.

When a launch stops with `cannot listen on [::1]:P` and
`another program answers on`, a program took the port after launch chose
it. The message names the address that answered. Launch again, and the app
moves to another port.

### Launch will not follow a file in the private home

A launch that stops with a message that names a path in the
[private home](launch-container.md#the-private-home) found a file there
that launch will not read or replace, such as a symbolic link or a file
larger than 16 MiB. The client in the container owns that folder and can put
links there, so launch never follows one.

Delete the path the message names, or remove the client's home for that
project with `gmlx launch <client> --remove-home`, and launch again. A file
that only the git identity needs, such as `.gitconfig`, gives a warning
instead and the launch goes on.

### Launch refuses a mount through a symbolic link

A `--mount` or a [`mounts`](config.md#launchcontainermounts) entry whose
path is or passes through a symbolic link stops the launch, as
[Shares](launch-container.md#shares) explains. When you made the link
yourself, write the real path that the message gives instead.

### Launch refuses to build while the builder forwards your SSH agent

The image builder was started with SSH forwarding, so any Containerfile it
builds could use every key in your Mac's SSH agent. Launch never builds on
such a builder. Run `container builder stop`, and the next launch starts a
builder without the agent.

### A leftover container of another session keeps running

A killed launch of another client or project left its container behind,
and launch prints `still running` with a `container stop` command. A
container whose name starts with `gmlx-check-` is left from the check of an
image. The container holds its memory until it stops, so run that command.
`gmlx doctor` lists these containers too.

### A volume is in use

Another session or container has the volume attached, and
[one volume serves one container](launch-container.md#volumes) at a time.
Stop that session first, or list the volume under the client rather than
directly under `launch.container`, so that each project gets its own.

### The Mac runs out of file handles

A session that reads a very large shared tree holds a Mac file handle for
each file, as [Limits](container-security.md#limits) explains. Stop the
session to release the handles, and share a narrower folder next time.
`gmlx doctor` reports the count.

### Packages installed in `--shell` are gone at the next launch

A package that `apt-get` or `npm install -g` installs lands outside the
private home, so the session discards it, as
[What persists](container-images.md#what-persists) explains. Add it to the
image with [`packages`](container-images.md#extra-packages) instead.

### Postgres refuses the data folder on a share

Postgres in the container cannot keep its data in a share. Put the data on
a volume, as [Postgres](container-images.md#postgres) explains and shows.

## Voice

### The mic never works in talk

`gmlx talk` runs but never hears you, and macOS never showed a permission
prompt. macOS grants the microphone to the terminal app that you ran `talk`
from. In System Settings, open Privacy and Security, then Microphone, and
turn on your terminal, whether Terminal, iTerm2 or your editor. If you
dismissed the prompt long ago, turn the entry off and on to get a new one.
`gmlx talk --list-devices` shows whether any input device is visible.

### Transcription or speech fails because ffmpeg is not found

`/v1/audio/transcriptions`, or speech in mp3, flac or opus, answers 500.
The server log then says `The gmlx server finds no ffmpeg on its PATH`, or
`The gmlx server will not run` with the path of an ffmpeg and the reason.
The server looks for ffmpeg as
[How the services run](services.md#how-the-services-run) describes.

For a missing ffmpeg, run `brew install ffmpeg`, or start the server from a
shell whose `PATH` holds your ffmpeg. The server refuses an ffmpeg in a
folder that a container session shares or shared read-write, or a link that
leads there, such as one in `~/bin`, so remove that file. A line of the log
names each `PATH` entry that the server skips and the reason.

Send the request again after the install or the removal. The server looks
for ffmpeg at each request, so it needs no restart. A new `PATH` reaches the
server only when it starts again. `gmlx doctor` reports FAIL for ffmpeg
until it finds one.

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
| `~/.config/gmlx/gmlx.yaml`, `~/.gmlx.yaml` | These hold the config, as [Where gmlx looks](config.md#where-gmlx-looks) describes. |
| `~/.config/gmlx/` | `gmlx launch` writes injected clients' configs here. |
| `~/.pi/agent/`, `~/.omp/agent/`, `~/.config/goose/config.yaml`, `~/.hermes/config.yaml` | `gmlx launch` merges its settings into these files, as [The clients](launch.md#the-clients) describes. Delete those settings to remove gmlx. |
| `~/.cache/gmlx/` | It holds server runfiles and logs, chat input history and the GGUF header cache. |
| `~/.cache/gmlx/apc/` | The prompt cache is stored here when the disk tier is on and has no `path` of its own. |
| `~/.cache/gmlx/media/` | The server opens the media files a request names from here, as [Media in requests](api.md#media-in-requests) describes. |
| `~/.cache/gmlx/talk/` | The first `talk` fetches the wake-word and voice-activity models here. |
| `~/.cache/huggingface/` | `hf:` references resolve from these files. |
| `~/.local/share/gmlx/chats/` | Saved chat sessions are kept here. |
| `~/.local/share/gmlx/assistant-memory.db` | It holds the assistant's memory, with `assistant-<id>.db` beside it for each served assistant. |
| `~/Library/Application Support/gmlx/` | The menu bar runs from an app bundle that gmlx writes here. |
| `~/Library/LaunchAgents/com.gmlx.*.plist` | `gmlx service install` writes its login items here. |
| `~/.open-webui/` | Open WebUI keeps its chat history here. |
| `~/.local/share/gmlx/launch/` | Container mode keeps the private homes, the ports of browser apps, and its locks and records here. |
| `~/.cache/gmlx/launch/` | Container mode keeps the session logs and session folders here. |
| Your model folders | `pull` downloads GGUFs into them. |

[Removing gmlx](installation.md#removing-gmlx) gives the steps that remove
gmlx and these files.
