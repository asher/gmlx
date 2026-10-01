# Container security

This page describes what a client in [container mode](launch-container.md)
can still reach on the Mac and on the gmlx server, and the limits a session
runs under. Read it before you share a folder read-write or turn on an
option that gives the client more access.

- [Shares that lead back to the Mac](#shares-that-lead-back-to-the-mac)
- [Your terminal](#your-terminal)
- [Browser app pages](#browser-app-pages)
- [Access you turn on](#access-you-turn-on)
- [What the client reaches on the server](#what-the-client-reaches-on-the-server)
- [Limits](#limits)

## Shares that lead back to the Mac

The container limits what the client can reach. It does not limit what the
client does in the folders you share, and a read-write share leads back to
the Mac in these ways:

- Files the client writes in a share run on the Mac when you use them.
  Examples are `.git/hooks`, `.git/config`, `.envrc`, the scripts in
  `package.json`, and a project's `.claude/settings.json`,
  `.claude/settings.local.json` and `.mcp.json`, which Claude Code on the
  Mac reads. Read what the client changed before you run the project on the
  Mac, including files that git ignores, such as `.venv` and `__pycache__`,
  since `git diff` does not show them.
- A `gmlx.yaml` the client writes in a share takes effect only when you
  pass it with `--config`. It can then change where the server listens,
  turn off its key or add a tool server command that the server runs on the
  Mac, so read it before you use it.
- When the server's config file or a model folder it scans is in a
  read-write share, the client can change what the server loads, and
  launch prints a warning. The same applies to a file that the config
  names, such as a model, a chat template file, the local model of a speech
  or embedding service, or a tool server's program. Move it out of the
  share, or share it read-only with `--mount PATH:ro`. When the running
  server has no config file, or an older gmlx started it, launch cannot
  check it and prints a line that names the fix.
- A server config that sets [`server.api_key`](config.md#serverapi_key)
  gives that key to the client in any share, also a read-only one, and
  launch warns. With the key, the client can call every route of the server
  wherever it reaches the server's port. Move such a config out of every
  share.
- A client's [`build`](config.md#launchcontainerclientsbuild) folder runs
  its code at the next build, with internet access even under
  `network: none`, so launch keeps it out of every read-write share, as
  [Your own Containerfile](container-images.md#your-own-containerfile)
  describes.
- The Python environment that gmlx runs from holds code that the Mac runs
  at the next `gmlx` command. Launch therefore refuses a read-write share
  that holds or lies in it, and it warns for a share that holds an editable
  checkout of gmlx. Launch from a folder that holds neither.
- Launch warns when `PATH` or `PYTHONPATH` has an empty or relative entry,
  or an entry in a read-write share. A program or a `gmlx` package that the
  client writes there would then run on the Mac in place of yours, so
  remove the entry. An empty entry is what
  `export PYTHONPATH="$PYTHONPATH:/x"` leaves when the variable was unset.
  Launch itself refuses a `container` program that it finds in a
  read-write share or a private home.

Launch checks every shared folder again right before the container starts,
and it stops when one has changed, such as a folder that another session's
client replaced with a link. A change after that check still reaches the
container, so share only folders that no other session can write.

A private home and a client's volumes lead from one session to the next in
the same way. What the client wrote there, such as hooks, a `.bashrc`, a
`.gitconfig` or a dsh profile, is in place when the next session of the
same project starts. The `default` project carries it into every launch
that uses that project, whatever folders those launches share. After a
client you do not trust has run, remove its home with `--remove-home`.

## Your terminal

The client runs in the terminal that you launched it from, as a program on
a remote host does over ssh. Launch passes the client's output to your
terminal unchanged, so the client can use any feature that your terminal
offers to programs.

One such feature is OSC 52, an escape sequence that writes the Mac
clipboard. kitty, Ghostty, WezTerm and Alacritty accept it with their
default settings and ask nothing, so a client can replace what you copied
with a command that you then paste into a Mac shell. Terminal.app ignores
OSC 52, and iTerm2 accepts it only when you allow clipboard access in its
settings. Turn the write off before you run a client you do not trust:

| Terminal | Setting that stops the write |
|----------|------------------------------|
| Ghostty | `clipboard-write = deny` |
| kitty | Remove `write-clipboard` from `clipboard_control`. |
| Alacritty | `terminal.osc52 = "Disabled"` |
| WezTerm | It has none, so run container sessions in another terminal. |

Other features need a setting or a click. kitty's remote control, which can
type into your other windows and start programs, works only with
`allow_remote_control` on. iTerm2 asks before a file download or upload,
and clipboard reads through OSC 52 are off or ask first in each of these
terminals. Leave remote control off, and answer no to a prompt that appears
while a session runs.

The client can also draw text that looks like a line from launch or like
your shell's prompt. Before you type a password in a terminal that ran a
session, make sure that the session ended, for example with `container ls`
in another terminal.

## Browser app pages

A [browser app](launch-container.md#browser-apps) page is code that the
container serves, and it runs in your Mac browser with a `127.0.0.1`
origin. It can send requests to the other services on the Mac's loopback
address, and read the answers of those that allow loopback pages. It can
read their cookies that are not `HttpOnly`, since cookies do not keep ports
apart, and reach the internet through the browser, also under
`network: none`. Set
[`open_browser: false`](config.md#launchcontaineropen_browser) for a client
you do not trust.

While the session is open, the gmlx server refuses the requests that a page
on the web port sends to its TCP port, so the page reaches the server only
through the session socket. The refusal lasts 15 minutes after the session
ends, also across a server restart, and its 403 message says to close the
app's browser tabs. After that, the page reaches the TCP port like any
local page, so close the app's tabs when the session ends.

A page cannot open a session of its own, because the server answers 404 to
a session request that a page sends. Another gmlx server on the Mac answers
the page as it answers any local page, so set a
[`server.api_key`](config.md#serverapi_key) on any other server you run.

## Access you turn on

Only the variables in [`env`](config.md#launchcontainerenv) and the ones the
client's configuration needs reach the container.

[`ssh_agent`](config.md#launchcontainerssh_agent) lets the client use the
SSH agent on the Mac while the session runs. The client can sign with every
key loaded in that agent, and it can also remove keys from the agent. Load
only the keys that the task uses.

A deploy key in the [private home](launch-container.md#ssh-in-the-container)
reaches only its own repository, but the client can copy it and use it after
the session. Prefer a deploy key when the work touches a single repository.

Each [forwarded port](launch-container.md#forwarded-ports) gives the client
a Mac service with the rights of a local user. With
[clipboard images](launch-container.md#clipboard-images) on, the client can
read the clipboard image at any time during the session.

The container reaches the internet and your local network unless you set
[`network: none`](config.md#launchcontainernetwork). On the default
network, a Mac service that listens on all addresses is reachable from the
container, and so is any device on your network. The connection to the
server needs no sudo, changes no network setting and raises no firewall
prompt, and the server sees `Host: 127.0.0.1:<port>` on every request.

A localhost domain of Apple container, which
`sudo container system dns create <domain> --localhost <ip>` adds, sends
every container to the Mac's loopback address on every port. The gmlx
server and a browser app's web port refuse such a connection, but other
local services may accept it. Launch and `gmlx doctor` warn while one
exists, so remove it with `sudo container system dns delete <domain>`
unless you need it.

## What the client reaches on the server

The client reaches only the inference routes of the gmlx server, and only
the served assistants you list for it. It connects through a
[session socket](glossary.md#session-socket), which the server opens for
that session alone, not through the server's port. The socket needs no
key, so the client's configuration holds the placeholder key
`gmlx-container-session` and never the server's key.

Those routes are the model list, chat, text completions, responses and
messages with their token counts, embeddings, rerank, speech and its voice
list, transcription, translation, image generation and image edits, and
`systemone`, plus `/health`. Every other route answers 404, so the client
cannot unload or keep models, reload the server's configuration or open
another socket.

Served assistants stay hidden from the client unless its
[`assistants`](config.md#launchcontainerclientsassistants) key lists them.
A request that names any other assistant gets the answer for an unknown
model, and the model list leaves it out. Launch prints one line for each
assistant the client can use, with the tool servers it calls:

```text
[launch] open-webui can use assistant home, whose tools run on the Mac: web, files
```

An assistant's tools run on the Mac with your rights, outside the
container, and the messages the client sends decide which tools it calls.
In a chat app such as Open WebUI, you write those messages, so the risk is
modest. A coding agent also sends text from the files, command output and
web pages it reads, and any of them can carry instructions for the tools.
Give a coding agent no assistants.

The server keeps the prompts of a session in its caches apart from those of
other clients and other projects, under a key that the client cannot
choose. Sessions of one client in one project share them, so a new session
reuses the prompts of the last. A session gets 400 for `dry_run`, which
reports the shared cache, and a served assistant's
[memory](config.md#serverassistantsmemory) is off for its turns.

The server checks every media part of a request through the socket, and it
refuses a file path or a URL there, even a file in the server's
[media folder](api.md#media-in-requests) or a URL with
[`server.media_urls`](config.md#servermedia_urls) on. The code that reads
the media does not check the session, so a reference outside those parts
could still reach it. On a server that container clients use, keep
`server.media_urls` off, and keep in the media folder only files that a
client may read.

The server decodes the client's media on the Mac, so a flaw in a decoder
runs with your rights. Images go through Pillow, and audio in WAV, MP3 or
FLAC through miniaudio, in the server process. Audio in M4A, Ogg, Opus or
WebM goes to ffmpeg and ffprobe, and with
[`server.stt`](config.md#serverstt) set, the transcription routes run
ffmpeg on every upload. A video goes to the FFmpeg that OpenCV bundles, in
the server process. Upgrade gmlx for new Pillow and OpenCV releases, and
run `brew upgrade ffmpeg`, which does not update the copy in OpenCV.

These limits apply to a plain http server on this Mac, which is a server
whose host resolves only to loopback addresses or the Mac's own. With
`--base-url` naming another host or an https URL, launch opens no socket and
prints a line saying so. The client then gets the key you pass with
`--api-key`, and it can do all that key allows on that server.

A local server that offers no session sockets refuses container mode, since
launch cannot limit it. When that server is gmlx, the message says to run
`gmlx restart`, so that it runs the installed version.

The socket ends with the session. When the server restarts during a
session, launch asks it for a new socket, with the same assistants, within
about 2 seconds. When the server gives none, such as after a restart with
another API key, launch asks again at each new connection of the client,
and it prints the reason after the client exits.

## Limits

Each file the container reads in a share holds one file handle on the Mac
until the container stops. A client that reads a very large tree, such as a
home folder full of projects, can reach the Mac's limit for one process,
and two such sessions can reach the limit of the whole Mac. The two limits
differ from Mac to Mac, and `sysctl kern.maxfilesperproc kern.maxfiles`
prints them. Share narrow folders.

[`gmlx doctor`](cli.md#gmlx-doctor) reports the open file count and both
limits, and it warns when more than half of the Mac's limit is open while a
session runs. Stop the session to release the handles.

The container's memory counts against the model server's memory until the
container stops, even when the client inside frees it.
[`memory`](config.md#launchcontainermemory) sets its size, and launch warns
once for each size above a quarter of the Mac's memory. When other launch
containers already run, launch prints the memory that all of them and the
new one will hold, against the Mac's.

Requests take server memory too. A session sends at most 16 requests at
once, each with a body of at most 32 MiB, and the server holds several
times that size while it reads and decodes a body. The server keeps at most
32 sessions open. Leave a few GiB free beside the model when a client you
do not trust runs.

The private home and the read-write shares have no size limit, so a client
can fill the Mac's disk. A volume stops at its size, and launch warns when
the volumes could outgrow the free space. Watch the free space while a
client works unattended.

A configured model that fails to load answers with its load error, which
can name the model's path on the Mac. Keep model paths free of names that
you would not show the client.
