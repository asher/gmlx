# Container security

This page describes what a client in [container mode](launch-container.md)
can still reach on the Mac and on the gmlx server, and the limits a session
runs under. Read it before you share a folder read-write, run a client on
code you do not trust, or turn on an option that gives the client more
access.

- [A session for code you do not trust](#a-session-for-code-you-do-not-trust)
- [Shares that lead back to the Mac](#shares-that-lead-back-to-the-mac)
- [Your terminal](#your-terminal)
- [Browser app pages](#browser-app-pages)
- [Access you turn on](#access-you-turn-on)
- [What the client reaches on the server](#what-the-client-reaches-on-the-server)
- [Custom agents](#custom-agents)
- [Limits](#limits)

## A session for code you do not trust

These steps run a coding agent on a repository that you do not trust, and
bring out only the changes that you read:

1. Turn off the clipboard write of your terminal, as
   [Your terminal](#your-terminal) shows. Then clone the repository into a
   separate folder:

   ```sh
   git clone https://example.com/them/project ~/review/project
   cd ~/review/project
   ```

2. Install the project's dependencies from a shell in the container, with
   the project's install command:

   ```sh
   gmlx launch claude-code --container --shell -- -c "npm ci"
   ```

3. Start the agent without the network:

   ```sh
   gmlx launch claude-code --container --network none
   ```

4. Take the changes out as a patch, read it, and apply it to your own
   clone:

   ```sh
   gmlx launch claude-code --container --shell -- -c \
     "git add -A && git diff --cached --binary > changes.patch"
   git -C ~/src/project apply ~/review/project/changes.patch
   ```

5. Remove the private home with `gmlx launch claude-code --remove-home` in
   the folder, then delete the folder.

The install in step 2 runs the project's install scripts in the container
with the network, so do it before the agent starts. Under `network: none`,
the agent reaches only the gmlx server and the forwarded ports. It cannot
send the code anywhere or download tools.

The agent can write files that git and other tools on the Mac run, as
[Shares that lead back to the Mac](#shares-that-lead-back-to-the-mac)
lists. So run no git command and none of the project's scripts in that
folder on the Mac. The patch is text that you can read in full, and
`git apply` writes only the files that it names. Give the agent no
[assistants](#what-the-client-reaches-on-the-server), whose tools run on
the Mac.

## Shares that lead back to the Mac

The container limits what the client can reach. It does not limit what the
client does in the folders you share, and a read-write share leads back to
the Mac through files that the Mac reads or runs later. Some of these you
check yourself, and `launch` and the server guard the rest.

`launch` checks every shared folder again right before the container
starts, and it stops when one has changed, for example a folder that
another session's client replaced with a link. A change after that check
still reaches the container, so share only folders that no other session
can write.

### Files you read before you use them

- Files the client writes in a share run on the Mac when you use them.
  Examples are `.git/hooks`, `.git/config`, `.envrc`, the scripts in
  `package.json`, and a project's `.claude/settings.json`,
  `.claude/settings.local.json` and `.mcp.json`, which Claude Code on the
  Mac reads. Read what the client changed before you run the project on the
  Mac, including files that git ignores, such as `.venv` and `__pycache__`,
  since `git diff` does not show them.
- For a Python project, the files to read include `pyproject.toml`,
  `uv.lock`, `uv.toml`, `.python-version` and build backend code such as
  `setup.py` or a `backend-path` folder. uv runs them on the Mac when you
  run or lock the project there.
- A `gmlx.yaml` the client writes in a share takes effect only when you
  pass it with `--config`. It can then change where the server listens,
  turn off its key or add a tool server command that the server runs on the
  Mac, so read it before you use it.

A private home and a client's volumes lead from one session to the next in
the same way. What the client wrote there, such as hooks, a `.bashrc`, a
`.gitconfig` or a dsh profile, is in place when the next session of the
same project starts. The `default` project carries it into every launch
that uses that project, whatever folders those launches share. After a
client you do not trust has run, remove its home with `--remove-home`.

Volumes listed directly under `launch.container` keep their names in every
project and client. What one client writes there reaches every other
client and project that mounts it, so keep these volumes for data that no
client can misuse.

### The server's config and models

- When the server's config file or a model folder it scans is in a
  read-write share, the client can change what the server loads, and
  `launch` warns. A voice session of the menu bar also reads that config
  when it starts, with its `talk` block and the assistant's tool servers.
  Move the file out of the share, or share it read-only with
  `--mount PATH:ro`.
- A file that the config names gets the same warning in a read-write share.
  Examples are a model, a chat template file, the local model of a speech
  or embedding service, and a tool server's program or a path in its
  arguments.
- The server can reach its config through a link in a read-write share, or
  in a folder that an earlier session shared read-write. The client can
  then choose the file that the server reads, so `launch` does not read the
  config, and it warns. Start the server with `--config` and a path that
  does not go through the link.
- The commands that write the config, `gmlx init`, `gmlx pull`,
  `gmlx sync-models`, `gmlx rm` and the menu bar's Edit config, refuse a
  link in these folders, or in a private home, when it leads out of that
  folder. Remove
  the link if you did not make it, or give a path that does not go through
  it. Edit config then shows no text and names the refusal, and its Save
  and Open in Editor stay refused.
- When the running server has no config file, or an older gmlx recorded
  its config by a relative path, `launch` cannot check it and names the
  fix.
- A server config that sets [`server.api_key`](config.md#serverapi_key)
  gives that key to the client in any share, even a read-only one, and
  `launch` warns. With the key, the client can call every route of the
  server wherever it reaches the server's port. Move that config out of
  every share.

### Programs that the Mac runs

- A client's [`build`](config.md#launchcontainerclientsbuild) folder runs
  its code at the next build, with internet access even under
  `network: none`, so `launch` keeps it out of every read-write share by
  the rules of
  [Your own Containerfile](container-images.md#your-own-containerfile).
- The Python environment that gmlx runs from holds code that the Mac runs
  at the next `gmlx` command. `launch` therefore refuses a read-write share
  that holds or lies in it, or that holds a link on the way to it, such as
  a project's `.venv` that leads there. The same refusal covers the `gmlx`
  program that you ran or that `PATH` finds, the gmlx app whose Python the
  login agents and the menu bar run, and the copy of Python that the server
  runs as.
- `launch` warns for a read-write share that holds the gmlx package or the
  Python installation that gmlx's environment comes from. It also warns for
  another editable checkout in that environment, such as a package that you
  installed with `pip install -e`. Share that folder read-only with `:ro`,
  or launch from a folder that holds none of them.
- `launch` runs `git` and `ssh-add` only from `/opt/homebrew/bin`,
  `/usr/local/bin`, `/usr/bin` and `/bin`, and for `/usr/bin/git` it runs
  the git of the developer folder that `xcode-select -p` names. A
  read-write share that holds one of these programs, or a folder searched
  before it, is refused. So is a share that holds or lies in the developer folder or
  the installation that the program comes from, such as `/opt/homebrew`.
- `launch` refuses a `container` program in a read-write share, in a
  private home or in a folder that an earlier session shared read-write.
  It opens a browser app with `/usr/bin/open`, never with a program that
  `PATH` finds.
- `launch` warns when `PATH` or `PYTHONPATH` has an empty or relative
  entry, or an entry that lies in a read-write share or leads through a
  link in one. A program or a `gmlx` package that the client writes there
  would then run on the Mac in place of yours, so remove the entry. An
  empty entry is what `export PYTHONPATH="$PYTHONPATH:/x"` leaves when the
  variable was unset.
- An active Python environment in a read-write share gets a separate step
  in that warning. What the client changes there stays after the session and
  runs when you use the environment or activate it again. Keep the
  environment outside the share, or share the project read-only.
- The server and the menu bar that a container launch starts get no
  `PATH` entry that a client can write. A server or menu bar that you start
  yourself, also through a `gmlx launch` that runs the client on the Mac,
  keeps your shell's `PATH`. The menu bar runs programs such as `open` and
  `launchctl` by their full paths.
- The server never runs a program from a folder that a client can write,
  whatever its `PATH`. These folders are the private homes, and each
  folder that a session shares read-write now or
  [shared read-write earlier](cli.md#the-share-history). The server skips
  them when it looks for ffmpeg, ffprobe and the command of a tool server.
  It refuses a program in one of them when the config names it by its full
  path, when its path passes through a link in one of them, or when a link
  leads there. It also refuses a program from a Homebrew
  installation, `/opt/homebrew` or `/usr/local`, that holds a shared
  folder. `gmlx chat --assistant` and `gmlx talk` do the same for their
  tool servers, and
  [How the services run](services.md#how-the-services-run) describes the
  search.
- A tool server gets a `PATH` without those folders, so a program that it
  runs by name, such as the `node` of an `npx` server, does not come from
  them. gmlx never starts a tool server in a folder that a client can
  write, because npx and `python -m` load code from the folder that they
  run in. Start gmlx in another folder, such as your home folder.
- Before each tool call, gmlx checks the program, the working folder and
  the `PATH` of the running tool server again. When a later session shares
  a folder that holds one of them, gmlx stops the tool server and starts it
  again without that folder. When it cannot, as when the share holds the
  working folder, gmlx refuses the call. The log of the tool server,
  `~/.cache/gmlx/mcp-<name>.log`, names each stop.

The checks of the server compare paths, so they cannot see a hard link.
conda and pnpm link one file into several environments that way. A program
on `PATH` that is a hard link of a file in a share passes the checks, and a
write through the share changes it. A tool server that runs a program by
name between tool calls keeps its old `PATH` until the next call.

## Your terminal

The client runs in the terminal that you launched it from, as a program on
a remote host does over ssh. `launch` passes the client's output to your
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

The client can also draw text that looks like a line from `launch` or like
your shell's prompt. Before you type a password in a terminal that ran a
session, make sure that the session ended, for example with `container ls`
in another terminal.

When a session ends, `launch` drops the input that waits in the terminal,
which holds your terminal's answers to the client's last queries.
`--detach` shows the client's start output in your terminal until the
session runs, and it drops that input too when it stops waiting. An answer
that arrives after `launch` exits, or while you run `launch` as a
background job of the shell, still reaches the shell.

## Browser app pages

A [browser app](launch-container.md#browser-apps) page is code that the
container serves, and it runs in your Mac browser at `http://[::1]` on the
project's port. It can send requests to the other services on the Mac's
loopback address, and read the answers of those that allow loopback pages.
It can reach the internet through the browser too, even under
`network: none`.

For the browser, `[::1]` is not the same site as `127.0.0.1` or
`localhost`, so the page gets none of the cookies of the apps there,
host-mode dsh and Open WebUI among them. A request that the page sends to
one of these apps, whether an image, a fetch or a form POST, carries none of its
cookies with `SameSite=Lax` or `SameSite=Strict`.

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

Container apps of different projects share one site on `[::1]`, and gmlx
does not separate them. The browser keeps cookies by host name, not by
port, so a request to one container app carries the cookies of every
container app. A page of one project can therefore send requests to the app
of another project with that app's cookies, its Strict ones included. The
page cannot read the answers unless that app allows the page's origin.

Each app's server in its container receives these cookies too, such as the
dsh sign-in of another project or the `token` cookie of Open WebUI in a
container. The container cannot use these cookies against those apps. The
web ports listen on the Mac's `::1` only, and a forwarded port leads to the
Mac's `127.0.0.1`. The app in another container listens on `127.0.0.1` in
that container, unless you pass another `--host` after `--`.

For a client you do not trust, set
[`open_browser: false`](config.md#launchcontaineropen_browser), and open its
apps in a separate browser profile, where its pages find no cookies of
your other apps.

A page can also leave a service worker, stored data and cached files at its
address, which stay after the session ends. Each project gets a
[separate port](launch-container.md#browser-apps), so the pages of
another project do not reach them.

Pages that are still open keep running and can store data again. After a
session of a client you do not trust, and before a port that served another
project opens a new app, close each tab and window of that address and each
window its pages opened, or quit the browser. Then clear the site data of
`http://[::1]:<port>`, the only address at which the app answers.

While the session is open, the gmlx server refuses the requests that a page
on the web port sends to its TCP port, so the page reaches the server only
through the session socket. The refusal lasts 15 minutes after the session
ends, even across a server restart, and its 403 message says to close the
app's browser tabs. After that, the page reaches the TCP port like any
local page, so close the app's tabs when the session ends.

A page cannot open a session, because the server answers 404 to
a session request that a page sends. Another gmlx server on the Mac answers
the page as it answers any local page, so set a
[`server.api_key`](config.md#serverapi_key) on any other server you run.

## Access you turn on

Only the variables in [`env`](config.md#launchcontainerenv) and the ones the
client's configuration needs reach the container.

[`ssh_agent`](config.md#launchcontainerssh_agent) lets the client use the
SSH agent on the Mac while the session runs. The client can sign with every
key loaded in that agent, so it can push to any repository those keys
reach, and it can also remove keys from the agent. Load only the keys that
the task uses.

A [deploy key](launch-container.md#ssh-in-the-container) in the private
home reaches only one repository, but the client can copy it and use it
after the session. Prefer a deploy key when the work touches a single
repository.

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
refuses these connections, and the web port of a browser app listens on
`::1`, which the domain does not reach.

Other local services may accept it, so `launch` and `gmlx doctor` warn
while a localhost domain exists. Remove it with
`sudo container system dns delete <domain>` unless you need it.

## What the client reaches on the server

The client reaches only the inference routes of the gmlx server, and only
the served assistants you list for it. It connects through a
[session socket](glossary.md#session-socket), which the server opens for
that session alone, not through the server's port. The socket needs no
key, so the client's configuration holds the placeholder key
`gmlx-container-session` and never the server's key.

The socket and the limits in this section apply to a plain http server on
this Mac, which is a server whose host resolves only to loopback addresses
or to the addresses of the Mac. With `--base-url` naming another host or an https URL,
`launch` opens no socket and says so. The client then gets the key you
pass with `--api-key`, and it can do all that key allows on that server.

Those routes are the model list, chat, text completions, responses and
messages with their token counts, embeddings, rerank, speech and its voice
list, transcription, translation, image generation and image edits, and
`systemone`, plus `/health`. Every other route answers 404, so the client
cannot unload or keep models, reload the server's configuration or open
another socket.

Served assistants stay hidden from the client unless its
[`assistants`](config.md#launchcontainerclientsassistants) key lists them.
A request that names any other assistant gets the answer for an unknown
model, and the model list leaves it out. `launch` names each assistant
that the client can use, with the tool servers it calls:

```text
[launch] open-webui can use assistant home, whose tools run on the Mac: web, files
```

An assistant's tools run on the Mac with your rights, outside the
container, and the messages the client sends decide which tools it calls.
In a chat app such as Open WebUI, you write those messages yourself, so the
risk is smaller. A coding agent also sends text from the files, command
output and web pages it reads, and any of them can carry instructions for
the tools. Give a coding agent no assistants, and give it tools through a
[tool server in the container](container-recipes.md#tool-servers) instead.

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
routes goes to ffmpeg. Keep these decoders up to date, as
[Upgrading](installation.md#upgrading) describes.

A server on this Mac that listens on all addresses or a network address
and needs no key is open to the container too. The client reaches every
route of that server at the Mac's address on the container network, beside
its session socket. `launch` and `gmlx doctor` warn about that server, so
set [`server.api_key`](config.md#serverapi_key) on it.

`launch` gives that warning even for a server that you name by
`127.0.0.1`, because it reads the address that a server in the background
listens on from its [runfile](glossary.md#runfile). A server that
`gmlx serve -f` runs in the foreground has no runfile, so check its
`--host` yourself.

A local server that offers no session sockets refuses container mode, since
`launch` cannot limit it. When that server is gmlx, the message says to run
`gmlx restart`, so that it runs the installed version.

The socket ends with the session. When the server restarts during a
session, `launch` asks it for a new socket, with the same assistants,
within about 2 seconds. When the server gives none, for example after a
restart with another API key, `launch` keeps asking at each new connection
of the client and every 2 seconds, and it prints the reason after the
client exits.

## Custom agents

A [custom agent](launch-agents.md) runs inside the boundary of every
container session, which is its virtual machine, its shares, its private
home, its volumes, the session socket, the forwarded ports and the web port
of a browser app. These points add to the rest of this page for agents:

- A runtime agent installs its dependencies when the session starts, and
  the code of every dependency runs inside that boundary. `launch` never
  builds an image from the agent's project folder.
- Network is all or nothing. `network: default` gives the agent the
  internet, and `network: none` leaves it the server and the forwarded
  ports.
- The agent, and every package it installs, can read each variable you
  pass through `env`, such as a search API key. Pass only the keys the
  agent needs.
- An `env` entry `OPENAI_API_KEY=...` replaces the session placeholder,
  and the agent then holds that key.
- A read-only `source` protects the source folder only. The environment
  on the dependency volume is writable, stays across launches and
  `--rebuild`, and runs at the next launch. Delete the volume with
  `--remove-home` after an agent you do not trust has run.

## Limits

Each file the container reads in a share holds one file handle on the Mac
until the container stops. A client that reads a very large tree, such as a
home folder full of projects, can reach the Mac's limit for one process,
and two of these sessions can reach the limit of the whole Mac. The two limits
differ from Mac to Mac, and `sysctl kern.maxfilesperproc kern.maxfiles`
prints them. Share narrow folders.

[`gmlx doctor`](cli.md#gmlx-doctor) reports the open file count and both
limits, and it warns when more than half of the Mac's limit is open while a
session runs. Stop the session to release the handles.

The container's memory counts against the model server's memory until the
container stops, even when the client inside frees it.
[`memory`](config.md#launchcontainermemory) sets its size, and its virtual
machine holds 128 MB more.

`launch` notes once for each size that, with those 128 MB, is above a
quarter of the Mac's memory. When other launch containers already run,
`launch` shows the memory that all of them and the new one will hold,
against the Mac's.

Requests take server memory too. A session sends at most 16 requests at
once, each with a body of at most 32 MiB, or 64 MiB for an audio upload,
as the [HTTP API](api.md#limits-and-back-pressure) lists. The server holds
about five times the size of a body while it reads and checks it, so 16
requests hold about 3 GiB.

A small media file can decode to much more than its size. The server limits
what the media of one request decode to, as the same list shows, and it
decodes the media of one request at a time, which takes at most about
2.5 GiB. Transcriptions also run one at a time, and the longest clip takes
about 6 GiB while the speech model reads it. In all, a client can make the
server hold about 11 GiB beside the model, or about 6 GiB on a server
without `server.stt`. Leave that much memory free when a client you do not
trust runs.

Each running launch keeps a session open on the server. Past 32 sessions, a
new launch closes the oldest session that has no open connection, and the
launch of that session opens a new one. When every session has an open
connection, the new launch stops with a message that says to wait. A launch
whose socket is gone asks for a new one every 2 seconds until it gets one.

The private home and the read-write shares have no size limit, so a client
can fill the Mac's disk. A volume stops at its size, and `launch` warns
when the volumes could outgrow the free space. The output file of a
detached session stays under 64 MiB of client output, but the output
before the session starts has no limit. Watch the free space while a
client works unattended.

A configured model that fails to load answers with its load error, which
can name the model's path on the Mac. Keep model paths free of names that
you would not show the client.
