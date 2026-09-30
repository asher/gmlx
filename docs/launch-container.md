# Container mode

Container mode runs a client from [`gmlx launch`](launch.md) inside an Apple
container, a small Linux virtual machine that sees only the folders you
share with it. This page covers what the container sees, how a session
runs, and what the isolation protects against.

The model server stays on the Mac, because a Linux guest has no Metal GPU.
Only the client moves into the virtual machine. That is the part that runs
shell commands and edits files, so it is the part worth isolating.

- [Turning on container mode](#turning-on-container-mode)
- [What the client sees](#what-the-client-sees)
- [Security model](#security-model)
- [The image](#the-image)
- [The clients](#the-clients)
- [Browser apps](#browser-apps)
- [The shell](#the-shell)
- [Volumes](#volumes)
- [Forwarded ports](#forwarded-ports)
- [Clipboard images](#clipboard-images)
- [Sessions, signals and exit codes](#sessions-signals-and-exit-codes)
- [The dry run](#the-dry-run)
- [What does not work in a container](#what-does-not-work-in-a-container)
- [Limits](#limits)
- [Removing container data](#removing-container-data)

## Turning on container mode

Container mode needs Apple container 1.4 or newer, which
[Installation](installation.md#apple-container) covers. Turn it on for one
run with `--container`, from the project folder the client should see:

```sh
cd ~/src/my-project
gmlx launch claude-code --container --model qwen3.8-27b-ud-q6
gmlx launch pi --container -- --continue
```

To make it the default, set
[`launch.container.enabled`](config.md#launchcontainerenabled) for every
client or for one client. `--no-container` then runs a client on the Mac for
one launch. Each flag that only makes sense in a container turns on
container mode by itself. These flags are `--mount`, `--mount-cwd`,
`--no-mount-cwd`, `--image`, `--rebuild`, `--reseed`, `--network` and
`--shell`, and the [CLI reference](cli.md#gmlx-launch) describes each one.

Launch reads its container settings only from the config file in your
home folder, as [Launch](config.md#launch) explains.

The first container launch takes a few minutes. Launch numbers the steps
it runs, so that a slow download does not look like a hang. Launch checks
the shares and the image settings before these steps, so a mistake in them
never waits behind a download. A busy web port, a problem in the image
itself and a link in the client's private home stop the launch only after
these steps.

1. The container service starts. Its first start asks to install a Linux
   kernel and downloads about 700 MB. When the service is stopped and the
   terminal is not interactive, launch prints `container system start` and
   exits 1 instead.
2. Launch builds the client's image, which takes a minute or two and
   downloads the Node base image once.
3. The client starts.

Later launches start the virtual machine in about a second.

## What the client sees

The client sees the folders you share, its
[private home](glossary.md#private-home), its volumes, the inference routes
of the gmlx server, the forwarded ports and, when you turn it on, images
from the Mac clipboard. It does not see the rest of your files. Your
keychain, SSH keys, other projects and `gmlx.yaml` stay out of reach unless
you share them. The [security model](#security-model) lists the ways a
session can still reach the Mac.

### Shares

By default launch shares the current folder, read-write, at the same path
in the container, and the client starts there. Open WebUI and elia share
nothing by default.
[`launch.container.mount_cwd`](config.md#launchcontainermount_cwd) sets the
share in the config, and `--no-mount-cwd` turns it off for one launch.

Launch refuses to share these folders by default, and it asks you to launch
from a project folder instead:

- Your home folder, any folder that holds it, and system folders such as
  `/`, `/Users`, `/Volumes`, `/tmp` and `/Applications`.
- The temporary folders of macOS, `$TMPDIR` and everything under
  `/private/var/folders`, which hold the temporary files of every program.
  A scratch project under `/private/tmp` can be shared.
- Credential folders such as `~/.ssh`, `~/.aws`, `~/.config/gh` and
  `~/.config/gmlx`, folders whose files the Mac runs, such as
  `~/Library/LaunchAgents`, `~/.local/bin` and `/opt/homebrew`, and any
  folder that holds one of these or lies inside one.
- gmlx's own data, under `~/.cache/gmlx` and `~/.local/share/gmlx`.

Launch compares these paths in the form macOS itself gives a folder, so
`/System/Volumes/Data/Users/you` counts as `/Users/you`. On a volume that
ignores the case of names, as APFS does by default, launch also ignores
case, so `~/.SSH` counts as `~/.ssh`.

`--mount PATH[:DST][:ro]` and
[`launch.container.mounts`](config.md#launchcontainermounts) share more
folders. A mount of a credential folder is honored, and launch prints a
warning that names it. A mount that holds or lies in gmlx's own data,
settings or server state, such as `~/.cache`, `~/.config/gmlx` or
`~/.local/share/gmlx`, is always refused, because a client could then
choose the config file that later launches read. Only folders can be
shared, not single files, and a path that contains `,` or
`=` is refused. A mount of the current folder at its own path replaces the
default share, so `--mount .:ro` shares the project read-only.

A mount path that is a symbolic link, or that passes through one, is
refused, and the message gives the real path to write instead. A client in
an earlier, wider share could have replaced a folder with a link, so launch
never follows one in a mount. This also refuses a path such as `/tmp/x`,
which you write as `/private/tmp/x`.

Files the client writes in a share appear on the Mac with your user as
their owner, and modes and symbolic links are kept. Inside the container
every file of a share appears to belong to root, and the client runs as
root.

A file lock taken in the container does not block the Mac, and a lock
taken on the Mac does not block the container. Use a database or a build
folder that relies on locks from one side at a time, either the Mac or the
container.

macOS guards `~/Desktop`, `~/Documents`, `~/Downloads`, iCloud Drive and
`/Volumes`. For a share in one of them, launch prints a notice, because
macOS may ask once whether the container runtime can read the folder. The
container waits until you answer.

### The private home

Each client gets one persistent home of its own at
`~/.local/share/gmlx/launch/<client>/home`, which appears at the same path
in the container. The client's settings, sessions and caches live there,
so they survive from one launch to the next. Your own `~/.claude`, `~/.pi`
and other client folders are never shared, and launch writes the client's
configuration into the private home instead.

The client can change anything in its private home, so launch never
follows a symbolic link there when it reads or writes the configuration.
It stops the launch instead, as
[the troubleshooting entry](troubleshooting.md#launch-will-not-follow-a-file-in-the-private-home)
describes.

A fresh private home starts empty, so a client such as Claude Code shows
its first-run steps once. Launch copies your git `user.name` and
`user.email` into the private home's `.gitconfig` when they are missing
there, so commits made in the container carry your name.

[`seed`](config.md#launchcontainerclientsseed) copies chosen files or
folders from your home into the private home. Launch copies each seed once
and records it outside the private home, so a client that deletes its copy
does not get a new one. `--reseed` copies every seed again and replaces the
copies. Launch checks a seed only when it copies it. It refuses a seed
whose real path lies outside your home folder, in a credential folder or in
gmlx's own data. It also refuses a seed in a folder that a session shares
or once shared read-write, when its real path leads out of that folder,
since a client may have replaced it with a link. Launch remembers the 500
folders it shared read-write most recently for this check and for the
build and git checks. A seed copied through a
link you made yourself prints a line with the path the link leads to.

Launch never follows a symbolic link while it copies, skips named pipes,
sockets and devices, and stops at 64 MiB, 10,000 files and folders, or 64
folders deep in one seed. The client reads every seeded file, so never seed
a sign-in token. Launch warns when it copies a file that can hold one, such
as `~/.claude.json`, `~/.gitconfig`, `~/.local/share/opencode/auth.json` or
`~/.config/goose/secrets.yaml`.

A seeded settings file can hold settings that only work on the Mac. A
`.gitconfig` with `credential.helper = osxkeychain` or commit signing, or a
Claude Code `settings.json` with hooks or a status line that run Mac
commands, fails in the container. Seed a copy without those settings.

### Git in a worktree

Git in the container works when you launch from the root of a repository.
A linked worktree or a submodule keeps its git folder outside its own
folder, so launch also shares that git folder and prints a line that names
its repository. The git folder is read-write, or read-only when the share
that holds the repository root is read-only, as with `--mount .:ro`.

Launch shares that git folder only when it names the project back. For a
worktree, the repository's entry for it names the project's `.git` file,
by an absolute path or by one relative to the entry. For a submodule, its
`core.worktree` names the project folder. The git folder must also be
named like one: `.git`, a name that ends in `.git`, `.bare`, or a folder
inside a `.git/modules` folder, and it must not hold the project. For a git
folder with any other name, such as a bare clone without the `.git`
suffix, share it with `--mount`.

The client can edit the `.git` file in the share, but not the repository
outside it, so it cannot use this to reach another repository. An earlier
launch that shared both the project and that repository read-write gave its
client both, and the client could then have made the project look like a
worktree git made. In that case launch shares the git folder only when the
project's `.git` file named it the first time launch shared that git folder.
Otherwise it prints a line with the `--mount` that shares the git folder,
so check the worktrees with `git worktree list` first.

Launch compares the recorded path as git wrote it and follows no symbolic
link in it, because the client can place a link in any folder it could
write, in this launch or an earlier one. Git records real paths, so every
worktree and submodule that git creates passes. A worktree that was moved
by hand and is reached through a link does not, and launch then does not
share its git folder and prints a note that names the repository. When that
folder really is a worktree of that repository, run `git worktree repair`
in it to record its real path.

After you delete a worktree by hand, run `git worktree prune`. Its stale
entry still names the old path, and a `.git` file placed there later would
count as that worktree.

The refusals for the current folder also apply to that git folder, and to
the repository that holds it. A worktree of a dotfiles repository in your
home folder therefore gets no git folder. In each case that shares nothing,
launch prints a note, and `--mount` shares the folder when you intend it.
Launched from a subfolder of a repository, the client sees only that
subfolder, and launch notes that git needs the repository root.

## Security model

The container limits what the client can reach. It does not limit what the
client does in the folders you share, and a read-write share leads back to
the Mac in these ways:

- Files the client writes in a read-write share run on the Mac when you
  use them. Examples are `.git/hooks`, `.git/config`, `.envrc` and the
  scripts in `package.json`.
- gmlx never reads a `gmlx.yaml` from the current folder by itself. A
  `gmlx.yaml` the client writes in a share takes effect only when you pass
  it with `--config`. It can then change where the server listens, turn off
  its key, or add a tool server command that the server runs on the Mac, so
  read it before you use it.
- The server reads its configuration and scans its model folders again
  when it reloads or restarts, and it reads a model file each time it
  loads that model. When that configuration, a model folder the server
  scans, or a model file it lists is inside a read-write share, the client
  can change what the server loads. A server that gmlx starts runs in its
  config file's folder, never in the folder you launch from, so a relative
  path in the config resolves beside the file. Launch prints a warning
  when it finds any of these in a share. A config in a folder that a session
  shares or once shared read-write, which leads out of that folder through a
  link, gets a warning and is not read. Launch prints a line instead when
  the running server has no config file, or one from an older gmlx that it
  cannot locate, since it cannot check that server. Start the server with
  `--config`, or run `gmlx restart`.
- A [`build`](config.md#launchcontainerclientsbuild) folder that the client
  could write at any time runs the client's code at the next build, with
  internet access even under `network: none`. Launch refuses to share any
  client's build folder read-write, refuses to build from a folder any
  launch shared read-write, and names the files that changed when it builds
  again. It never gives a build your SSH agent, and it
  refuses to build while the image builder forwards the agent.
- A `gmlx` package the client writes in a share does not run in the
  processes gmlx starts. They run with Python's `-P` and without the empty
  or relative entries of `PYTHONPATH`, each of which puts the current
  folder on the import path. A `gmlx` command you run yourself keeps your
  `PYTHONPATH`, so launch warns when it has such an entry. An empty entry is
  what `export PYTHONPATH="$PYTHONPATH:/x"` leaves when the variable was
  unset.
- The client reads every file you [seed](#the-private-home), so a seeded
  token is the client's token.

Other access is opt-in:

- Only the variables in [`env`](config.md#launchcontainerenv) and the ones
  the client's configuration needs reach the container.
- [`ssh_agent`](config.md#launchcontainerssh_agent) lets the client sign
  with every key loaded in your Mac's SSH agent. For example, it can push
  to any repository those keys reach, so load only the keys the task needs.
- Each forwarded port gives the client that Mac service with the rights of
  a local user. Homebrew's Postgres and Redis accept local connections
  without a password, so set a password or a limited role before you
  forward one. Never forward a browser's remote debugging port, such as
  9222, because that gives the client your logged-in browser.

The container reaches the internet and your local network unless you set
[`network: none`](config.md#launchcontainernetwork). On the default
network, a Mac service that listens on all addresses is reachable from the
container, and so is any device on your network. The connection to the
server needs no sudo, changes no network setting and raises no firewall
prompt, and the server sees `Host: 127.0.0.1:<port>` on every request.

Launch shows every control character in a name or message it prints as a
`\xNN` escape, so a name the client chose cannot move the cursor, rewrite
earlier lines or set your terminal's clipboard.

### What the client reaches on the server

A client reaches a gmlx server on the Mac through a socket that the server
opens for its session, not through the server's port. The socket needs no
key, so the client's configuration holds the placeholder key
`gmlx-container-session` and never the server's key. The socket serves only
the routes that clients use for inference: the model list, chat, text
completions, responses, messages, embeddings, rerank, speech, transcription,
images and `systemone`, plus `/health`. Every other route answers 404, so
the client cannot unload or keep models, reload the server's configuration
or open another socket.

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

A request through the socket takes an image, audio or video only as inline
data. The client therefore cannot make the server read a Mac file, even one
in the server's [media folder](api.md#media-in-requests), or fetch a URL,
even with [`server.media_urls`](config.md#servermedia_urls) on.

With [`server.stt`](config.md#serverstt) set, the transcription and
translation routes write the client's upload to a temporary file and run
ffmpeg on it. ffmpeg on the Mac therefore parses bytes the client chose, so
keep it up to date, or leave `server.stt` unset on a server that container
clients use.

The socket serves at most 16 connections at a time and answers 503 to the
next one without reading its body. Its body limits are smaller than those
of the TCP port, as [Limits and back-pressure](api.md#limits-and-back-pressure)
lists. A client in the container therefore cannot make the server hold more
than 16 request bodies at a time.

A [browser app](#browser-apps) gives the client a page in your browser at
`http://127.0.0.1:<web port>`, and that page runs on the Mac. The app's own
server reaches the gmlx server through the socket, so while the session is
open the server refuses every request from a local page on the web port
that comes in on its TCP port. The page therefore cannot go around the
socket's limits. Another gmlx server on the Mac does not know about the
session, and it answers the page as it answers any local page, so set a
[`server.api_key`](config.md#serverapi_key) on any other server you run.

These limits apply to a plain http server on this Mac. Launch resolves the
server's host and counts it as local when every address is a loopback
address or one of the Mac's own, however the URL spells it. With
`--base-url` naming another host or an https URL, launch opens no socket
and prints a line saying so. The client then gets the key you pass with
`--api-key`, and it can do all that key allows on that server. A local
server that offers no session sockets refuses container mode, since launch
cannot limit it. When that server is gmlx, the message says to run
`gmlx restart` so that it runs the installed version.

The socket ends with the session, and the server removes every session
socket when it starts and when it stops. When the server restarts during a
session, launch asks it for a new socket, with the same assistants, at the
client's next request. When the server gives none, such as after a restart
with another API key, the session log names the reason, and launch prints
it after the client exits. Launch relays the client only to a socket that only
your user can open, in a session folder of a server on that port, so a
program that answers at the server's address cannot point the client at
another socket on the Mac.

## The image

With no settings, launch builds one image per client from a recipe that
ships with gmlx. The image starts from Debian with Node.js and adds git,
ripgrep, curl, an SSH client and the client itself, installed from its
official source. The recipe pins the base image by digest and each client
at one version, with a checksum on each download. The image is built once,
and built again when a gmlx upgrade changes the recipe or when its
`packages` change, so a newer client arrives with a gmlx release.

`--rebuild` builds the image again without its cache, which picks up new
Debian packages. Launch prints the image's age at each start, and it
suggests `--rebuild` when the image is older than 30 days.

[Custom container images](container-images.md) covers adding packages,
your own Containerfile and ready-made images. Any image works when it is
for Linux on arm64 and contains the command that runs. Launch refuses an
image for another architecture. It also checks an image of your own once
for each command. The check finds the command on the image's `PATH`,
confirms its execute bit, and confirms that the interpreter in its `#!`
line is in the image. A command that fails the check stops the launch
with exit 1 before the session starts, and under `--shell` it only prints
a warning. A program built for another system passes the check and fails
when the session starts.

### The command that runs

The container runs the client's command, followed by the arguments after
`--`. [`command`](config.md#launchcontainerclientscommand) changes it:

- A list replaces the client's command. Launch still writes the client's
  configuration, and the arguments after `--` follow the list.
- The word `image` runs the image's own ENTRYPOINT and CMD. The arguments
  after `--` replace CMD, and the container starts in the image's working
  folder when it sets one.

## The clients

Each client gets the same configuration as on the Mac, written into its
private home. A client that merges into its own files, such as pi or
hermes, changes only the copies in the private home. Two clients change
further in a container:

| Client | In a container |
|--------|----------------|
| `claude-code` | Launch sets `IS_SANDBOX=1`, so `-- --dangerously-skip-permissions` works as root, and turns off its auto-updater. |
| `dsh` | A `--dsh-profile` must exist in the private home. Only `gmlx` and `web` run as web apps. `acp`, `sdk` and `sdk-minimal` need `--no-container`. |

Under `network: none`, Claude Code also gets
`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, so it stops trying to reach
Anthropic's servers.

## Browser apps

Open WebUI and the dsh web profiles serve a web app. The app listens on the
container's own `127.0.0.1`, and launch relays it to the same port on the
Mac. Open WebUI uses port 3000 and dsh uses 3080, or the next port when the
gmlx server already uses it. The port on the Mac accepts connections only
from the Mac itself.

Launch opens the browser once the app answers. For Open WebUI it waits up
to five minutes and then prints the address it waited for. dsh puts a
login token in its address, so launch reads the address from dsh's own
output and opens that. dsh's default workspace is in its private home, so
use Add workspace in the app to open the shared project folder.

A custom `command` for a browser app must listen on `127.0.0.1:$PORT`.
Launch sets `HOST` and `PORT` in the container for that. The official Open
WebUI image works with `command: image`, as its start script reads both
variables:

```yaml
launch:
  container:
    clients:
      open-webui:
        image: ghcr.io/open-webui/open-webui:main
        command: image
```

Launch then keeps Open WebUI's secret key in the private home, so a browser
login survives the next launch.

## The shell

`--shell` opens a shell in the same image, with the same shares, private
home and server connection as the client, instead of starting the client.
It runs `bash`, or `sh` when the image has no `bash`, and the arguments
after `--` go to the shell:

```sh
gmlx launch claude-code --shell
gmlx launch claude-code --shell -- -c "npm test"
```

While a session of that client runs, `--shell` opens a shell in its
container instead, to look at what the agent is doing. The shell starts in
the folder you run it from, when a share of the session holds it, and in
the session's working folder otherwise. It ends when the session ends. Only
`--container` and the arguments after `--` apply to a shell in a running
session.

An image with no shell at all makes `--shell` exit 127 with a message.
Anything you install from the shell is gone when the session ends, and
[Custom container images](container-images.md) covers keeping it.

## Volumes

A share keeps no file owners in the container, as [Shares](#shares)
describes, and it is slower than a local disk for work with many small
files. A named volume is a disk image that the container mounts as its own
disk, so ownership, file modes and locks work as on Linux. Databases and
large caches belong on a volume.

[`volumes`](config.md#launchcontainervolumes) entries take the form
`NAME:/path[:SIZE]`:

```yaml
launch:
  container:
    clients:
      claude-code:
        volumes: [claude-pg:/var/lib/postgresql:8G]
```

Launch creates a missing volume with the size in its entry, or the default
size that [`volumes`](config.md#launchcontainervolumes) gives. The disk
image on the Mac grows only as the container writes, up to that limit, and
the size is fixed when the volume is created. Launch prints one line per
volume with its limit and the space it takes on the Mac, and it warns when
the Mac disk has less free space than the volumes could still use. A
volume created with a different size gets a line that says how to recreate
it, which deletes its data.

The volume's root holds a `lost+found` folder, so put data in a subfolder.
Two containers never mount one volume at the same time. Launch refuses a
session whose volume another session or another container is using, so two
clients that list the same volume do not run side by side.

`container system df` shows the space that all volumes and images take
on the Mac. Deleting files in a volume does not free that space. To give it
back, trim the volume from a container that no session is using:

```sh
container run --rm --cap-add CAP_SYS_ADMIN \
  --mount type=volume,source=claude-pg,target=/v \
  docker.io/library/debian:bookworm-slim fstrim -v /v
```

A volume writes its data to the Mac with ordinary file syncs, not full
disk flushes. A database on a volume can lose its most recent commits if
the Mac loses power. Launch never deletes a volume, and
`container volume delete NAME` removes one with its data.

## Forwarded ports

[`forward`](config.md#launchcontainerforward) gives the client a service
that runs on the Mac, such as a database. Each port P in the list appears
on the container's own `127.0.0.1:P`, the same address as on the Mac, and
it works under `network: none` too. Launch connects to the Mac's
`127.0.0.1:P` only, never to `::1`, so a service that listens only on `::1`
cannot be forwarded. When the Mac service is down, the connection closes
at once and launch logs one line.

A forward of the gmlx server's port is refused, since the container already
reaches the server, and so is a forward of a browser app's web port. A
program in the container cannot listen on a forwarded port, so run a
service either in the container on a volume or on the Mac with a forward,
not both.

## Clipboard images

Clients on Linux paste an image by running `xclip`, `xsel` or `wl-paste`,
which find no clipboard in a container. With
[`clipboard: images`](config.md#launchcontainerclipboard), launch puts
replacements for those three commands first on the client's `PATH`, and
they read images from the Mac clipboard:

```yaml
launch:
  container:
    clipboard: images
```

Paste with the client's own key for images, such as Ctrl-V in Claude Code.
Cmd-V pastes only text into a terminal.

The container gets images only. It cannot read clipboard text, and it
cannot write the Mac clipboard at all. An image arrives as PNG. A PNG over
20 MB is refused, and so is an image in another type over 64 MB, before
launch converts it. Each image read adds a line to the session log, and so
do the first request for the clipboard's image types and every hundredth
one after it.

The setting is off by default, because the client can read the clipboard
image at any time during the session, not only when you paste. With it off,
a paste in the client fails or finds the image's own clipboard tools. To
hand over one image, save it into the shared folder instead.

macOS can deny an app access to the clipboard. The replacement commands
then fail with a message that names the setting to change, Paste from
Other Apps under Privacy & Security in System Settings.

## Sessions, signals and exit codes

One container session of each client runs at a time, and different clients
run side by side. A second launch of a running client is refused with the
name of its container and the `--shell` command. A session named
`gmlx-<client>-<6 characters>` prints its image, shares, volumes and
forwarded ports when it starts, and then prints nothing of its own.

Ctrl-C reaches the client as it does on the Mac. When launch itself is
stopped, it stops the container, and a second stop ends it at once. After
the client exits, launch removes the container and its session files. A
container that is still there afterwards gets a line with its
`container delete --force` command. A session whose launch was killed is
cleaned up by the next launch of that client, and a launch of any other
client prints a line with its `container stop` command, since the leftover
virtual machine holds memory until it stops.

The exit code is the client's own. Three codes come from the container
before the client starts, and each prints a one-line message that names
the cause:

| Code | Meaning |
|------|---------|
| 125 | The relay inside the container could not start, for example because a port it needs is in use. |
| 126 | The command is in the image but cannot run, such as a file without its execute bit, a script whose `#!` interpreter is missing, or a program for another system. |
| 127 | The command, or a shell for `--shell`, is not in the image. |

Launch exits 1 when it refuses a session, such as for a folder it will not
share or a volume in use. The session log is
`~/.cache/gmlx/launch/last-<client>.log`, and the private home holds
`.gmlx-entry.log` for errors inside the container. A line the container
causes, such as a refused connection, appears at most once a minute for
each kind, with a count of the ones in between. Each image the clipboard
sends gets its own line.

## The dry run

`--config-only` in container mode writes the client's configuration into
the private home and prints the `container run` command that a session
would use. The variables launch sets itself, such as `HOME`, `TERM`, `LANG`
and `IS_SANDBOX`, appear with their values. The client's own settings and
your [`env`](config.md#launchcontainerenv) entries appear by name only,
because they can hold keys. The dry run builds nothing, pulls nothing and
starts no container. It reports whether the image and volumes exist yet,
and whether the server offers session sockets. Use it to inspect a
session. The printed command cannot run by itself, because the connection
to the server exists only while launch supervises the session.

## What does not work in a container

Some client features call into the Mac and stop working in a container:

- Pasting clipboard images, unless you turn on
  [clipboard images](#clipboard-images).
- Opening a browser or a URL from the client. The client prints the link
  in the terminal instead.
- Notifications, sounds and hooks that run Mac commands such as
  `osascript` or `afplay`.
- Credentials kept in the Keychain, such as `gh` logins or git's
  `osxkeychain` helper. Pass a token through
  [`env`](config.md#launchcontainerenv) instead.

## Limits

Each file the container reads in a share holds one file handle on the Mac
until the container stops. A client that reads a very large tree, such as a
home folder full of projects, can reach the Mac's per-process limit of
about 245,000, and two such sessions can reach the limit of the whole Mac.
Share narrow folders. [`gmlx doctor`](cli.md#gmlx-doctor) reports the open
file count, and it warns when more than half of the Mac's limit is open
while a session runs.

The container's memory counts against the model server's memory until the
container stops, even when the client inside frees it.
[`memory`](config.md#launchcontainermemory) sets its size, and launch warns
when you give the container more than a quarter of the Mac's memory.

Work with many small files, such as `npm install`, runs slower in a share
than on a volume.

Launch closes a relayed connection, such as one to a forwarded port, when
no data moves in either direction for 30 seconds after it opens. A
connection from the container to the server must instead send a whole
request head in those 30 seconds. After that first data, or that request
head, the connection has no time limit, so a streamed answer is never cut.
When the client ends its half, the connection waits up to one hour with
no data for the answer, so a slow answer that does not stream still
arrives. Once the other end has ended its half, the connection closes after
30 seconds with no data.

Each listener accepts at most 200 new connections a second. Idle
connections and bursts of connections therefore cannot use up the file
handles of the server that other clients share. A forwarded port also
holds at most 32 connections at a time, so the client cannot take every
connection that a Mac service such as Postgres allows.

Launch checks every shared folder again just before the container starts,
and it stops when one has changed, such as a folder that another session's
client replaced with a link. A change after that check still reaches the
container, so share only folders that no other session can write.

## Removing container data

Uninstalling gmlx leaves container data in place, and each kind is removed
separately. Apple container keeps its images, volumes and Linux kernel in
`~/Library/Application Support/com.apple.container`. `gmlx doctor` reports
the space that volumes, private homes and images take:

| Data | How to remove it |
|------|------------------|
| A private home | Delete `~/.local/share/gmlx/launch/<client>/home` to reset that client. |
| Volumes | Run `container volume delete NAME` for each volume, which deletes its data. |
| Images | Run `container image delete` on the `gmlx.invalid/launch-*` entries and the `@sha256:` entries of your `image` references, then `container image prune`. |
| The guest program | Delete `~/.local/share/gmlx/launch/runtime`. |
| Apple container from Homebrew | Run `container system stop` and `brew uninstall container`, then delete that folder. |
| Apple container from Apple's installer | Run `container system stop`, then `uninstall-container.sh -d`, which also deletes that folder. |
