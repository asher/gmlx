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
  launch prints a warning. A voice session of the menu bar also reads such
  a config when it starts, with its `talk` block and the assistant's tool
  servers. Move the file out of the share, or share it read-only with
  `--mount PATH:ro`.
- A file that the config names gets the same warning in such a share.
  Examples are a model, a chat template file, the local model of a speech
  or embedding service, and a tool server's program or a path in its
  arguments.
- The server can reach its config through a link in a read-write share, or
  in a folder that an earlier session shared read-write. The client can
  then choose the file that the server reads, so launch does not read the
  config, and it warns. Start the server with `--config` and a path that
  does not go through the link.
- The commands that write the config, `gmlx init`, `gmlx pull`,
  `gmlx sync-models`, `gmlx rm` and the menu bar's Edit config, refuse such
  a link, or one in a private home, when it leads out of that folder. Remove
  the link if you did not make it, or give a path that does not go through
  it. Edit config then shows no text and names the refusal, and its Save
  and Open in Editor stay refused.
- When the running server has no config file, or an older gmlx recorded
  its config by a relative path, launch cannot check it and prints a line
  that names the fix.
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
  that holds or lies in it, or that holds a link on the way to it, such as
  a project's `.venv` that leads there. The same refusal covers the `gmlx`
  program that you ran or that `PATH` finds, and the Python that the login
  agents, the menu bar and the server run.
- Launch warns for a read-write share that holds the gmlx package or the
  Python installation that gmlx's environment comes from. It also warns for
  another editable checkout in that environment, such as a package that you
  installed with `pip install -e`. Share such a folder read-only with `:ro`,
  or launch from a folder that holds none of them.
- Launch runs `git` and `ssh-add` only from `/opt/homebrew/bin`,
  `/usr/local/bin`, `/usr/bin` and `/bin`, and for `/usr/bin/git` it runs
  the git of the developer folder that `xcode-select -p` names. A
  read-write share that holds such a program, or a folder searched before
  it, is refused. So is a share that holds or lies in the developer folder,
  or in the installation that the program comes from, such as
  `/opt/homebrew`.
- Launch warns when `PATH` or `PYTHONPATH` has an empty or relative entry,
  or an entry that lies in a read-write share or leads through a link in
  one. A program or a `gmlx` package that the client writes there would
  then run on the Mac in place of yours, so remove the entry. An empty
  entry is what `export PYTHONPATH="$PYTHONPATH:/x"` leaves when the
  variable was unset.
- An active Python environment in a read-write share gets its own step in
  that warning. What the client changes there stays after the session and
  runs when you use the environment or activate it again. Keep the
  environment outside the share, or share the project read-only.
- The server and the menu bar that a container launch starts get no
  `PATH` entry that a client can write. A server or menu bar that you start
  yourself, also through a `gmlx launch` that runs the client on the Mac,
  keeps your shell's `PATH`. The menu bar runs its own programs, such as
  `open` and `launchctl`, by their full paths.
- The server never runs a program from a folder that a client can write,
  whatever its `PATH`. It skips such folders when it looks for ffmpeg,
  ffprobe and the command of a tool server. It refuses such a program when
  the config names it by its full path or when a link leads there.
  `gmlx chat --assistant` and `gmlx talk` do the same for their tool
  servers, and [How the services run](services.md#how-the-services-run)
  describes the search.
- A tool server gets a `PATH` without those folders, so a program that it
  runs by name, such as the `node` of an `npx` server, does not come from
  them. gmlx never starts a tool server in a folder that a client can
  write, because npx and `python -m` load code from the folder that they
  run in. Start gmlx in another folder, such as your home folder.
- Before each tool call, gmlx checks the program, the working folder and
  the `PATH` of the running tool server again. When a later session shares
  one of them, gmlx stops the tool server and starts it again without that
  folder. When the share holds the program or the working folder, gmlx
  refuses the call. The log of the tool server,
  `~/.cache/gmlx/mcp-<name>.log`, names each stop.
- Launch refuses a `container`, `git` or `ssh-add` program in a read-write
  share, in a private home or in a folder that an earlier session shared
  read-write. It opens a browser app with `/usr/bin/open`, never with a
  program that `PATH` finds.

The checks of the server compare paths, so they cannot see a hard link.
conda and pnpm link one file into several environments that way. A program
on `PATH` that is a hard link of a file in a share passes the checks, and a
write through the share changes it. A tool server that runs a program by
name between tool calls keeps its old `PATH` until the next call.

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

Volumes listed directly under `launch.container` keep their names in every
project and client, as [Volumes](launch-container.md#volumes) describes.
What one client writes there reaches every other client and project that
mounts it, so keep such a volume for data that no client can misuse.

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

When a session ends, launch drops the input that waits in the terminal,
which holds your terminal's answers to the client's last queries. An answer
that arrives after launch exits, or while launch runs in the background,
still reaches your shell.

## Browser app pages

A [browser app](launch-container.md#browser-apps) page is code that the
container serves, and it runs in your Mac browser at `http://[::1]` on the
project's port. It can send requests to the other services on the Mac's
loopback address, and read the answers of those that allow loopback pages.
It can reach the internet through the browser too, even under
`network: none`.

For the browser, `[::1]` is not the same site as `127.0.0.1` or `localhost`,
so the page gets none of the cookies of the apps there, such as host-mode
dsh and Open WebUI. A request that the page sends to such an app, such as an
image, a fetch or a form, carries none of its cookies with `SameSite=Lax` or
`SameSite=Strict`.

Any page can still show another app in a frame, unless that app forbids it
with `X-Frame-Options` or `frame-ancestors`, which dsh does not send. What
`[::1]` changes is that the frame carries no cookies. Browsers send no Lax
or Strict cookie into a frame of another site, and Safari sends no cookie
there at all, so a host-mode dsh in a frame loads signed out.

A page can also open another app in a new window, or send the browser
there. That navigation carries the app's Lax cookies and a cookie with no
`SameSite`, but not its Strict ones. So host-mode Open WebUI, whose sign-in
cookie is Lax, opens signed in, and host-mode dsh opens signed out. The page
cannot read either new page.

Container apps of different projects stay the same site on `[::1]`, and gmlx
accepts this limit. The browser keeps cookies by host name, not by port, so
a request to one container app carries the cookies of every container app.
The app's server in its container receives them, such as the dsh sign-in of
another project or the `token` cookie of Open WebUI in a container.

The container cannot use such a cookie against those apps. The web ports
listen on the Mac's `::1` only, and a forwarded port leads to the Mac's
`127.0.0.1`. The app in another container listens on that container's own
`127.0.0.1`, unless you pass another `--host` after `--`.

For a client you do not trust, set
[`open_browser: false`](config.md#launchcontaineropen_browser), and open its
apps in a browser profile of their own, where its pages find no cookies of
your other apps.

A page can also leave a service worker, stored data and cached files at its
address, which stay after the session ends. Each project gets a
[port of its own](launch-container.md#browser-apps), so the pages of
another project do not reach them. After a session of a client you do not
trust, close the app's tabs and windows, then clear the site data of
`http://[::1]:<port>`. The app answers at no other address.

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

Open WebUI that `gmlx launch` runs on the Mac listens on `127.0.0.1` only.
With `gmlx launch open-webui -- --host 0.0.0.0` it listens on every
address, and other computers and the containers on the default network
reach it. Under `WEBUI_AUTH=false`, or before its first account exists, a
container can then make itself the admin of that Open WebUI.

A localhost domain of Apple container, which
`sudo container system dns create <domain> --localhost <ip>` adds, sends
every container to the Mac's `127.0.0.1` on every port. The gmlx server
refuses such a connection, and the web port of a browser app listens on
`::1`, which the domain does not reach.

Other local services may accept it, so launch and `gmlx doctor` warn while
such a domain exists. Remove it with
`sudo container system dns delete <domain>` unless you need it.

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

Every media part of a request through the socket is checked, and the server
refuses a file path or a URL there, even a file in the server's
[media folder](api.md#media-in-requests) or a URL with
[`server.media_urls`](config.md#servermedia_urls) on.

Keep `server.media_urls` off on a server that container clients use, and
keep in the media folder only files that a client may read. The code that
reads the media does not check the session, so a reference outside those
parts could still reach it.

The server decodes the client's media on the Mac, so a flaw in a decoder
runs with your rights. Images go through Pillow, and audio in WAV, MP3 or
FLAC through miniaudio, in the server process. A video goes to the FFmpeg
that OpenCV bundles, also in the server process.

Audio in M4A, Ogg, Opus or WebM goes to
[ffmpeg and ffprobe](services.md#how-the-services-run), and with
[`server.stt`](config.md#serverstt) set, every upload to the transcription
routes goes to ffmpeg.

Keep these decoders up to date. `brew upgrade gmlx` brings the Pillow and
OpenCV versions tested with each release. `uv tool upgrade gmlx` brings the
newest versions that gmlx's requirements allow, also when gmlx itself has no
new release.

With pip, `pip install -U gmlx` keeps the installed Pillow and OpenCV while
they still meet its requirements, so also run
`pip install -U pillow opencv-python`. `brew upgrade ffmpeg` updates ffmpeg,
but not the copy of FFmpeg in OpenCV.

These limits apply to a plain http server on this Mac, which is a server
whose host resolves only to loopback addresses or the Mac's own. With
`--base-url` naming another host or an https URL, launch opens no socket and
prints a line saying so. The client then gets the key you pass with
`--api-key`, and it can do all that key allows on that server.

A server on this Mac that listens on all addresses or a network address
and needs no key is open to the container too. The client reaches every
route of that server at the Mac's address on the container network, beside
its session socket. Launch and `gmlx doctor` warn about such a server, so
set [`server.api_key`](config.md#serverapi_key) on it.

Launch gives that warning also for a server that you name by `127.0.0.1`,
because it reads the address that a server in the background listens on
from its [runfile](glossary.md#runfile). A server that `gmlx serve -f` runs
in the foreground has no runfile, so check its `--host` yourself.

A local server that offers no session sockets refuses container mode, since
launch cannot limit it. When that server is gmlx, the message says to run
`gmlx restart`, so that it runs the installed version.

The socket ends with the session. When the server restarts during a
session, launch asks it for a new socket, with the same assistants, within
about 2 seconds. When the server gives none, such as after a restart with
another API key, launch keeps asking, also at each new connection of the
client, as [Limits](#limits) describes. It prints the reason after the
client exits.

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
[`memory`](config.md#launchcontainermemory) sets its size, and its virtual
machine holds 128 MB more.

Launch prints a note once for each size that, with those 128 MB, is above a
quarter of the Mac's memory. When other launch containers already run,
launch prints the memory that all of them and the new one will hold, against
the Mac's.

Requests take server memory too. A session sends at most 16 requests at
once, each with a body of at most 32 MiB, and the server holds several
times that size while it reads and decodes a body. Leave a few GiB free
beside the model when a client you do not trust runs.

Each running launch keeps a session open on the server. Past 32 sessions, a
new launch closes the oldest session that has no open connection, and the
launch of that session opens a new one. When every session has an open
connection, the new launch stops with a message that says to wait. A launch
whose socket is gone asks for a new one every 2 seconds until it gets one.

The private home and the read-write shares have no size limit, so a client
can fill the Mac's disk. A volume stops at its size, and launch warns when
the volumes could outgrow the free space. Watch the free space while a
client works unattended.

A configured model that fails to load answers with its load error, which
can name the model's path on the Mac. Keep model paths free of names that
you would not show the client.
