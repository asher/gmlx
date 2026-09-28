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
`--no-mount-cwd`, `--image`, `--rebuild`, `--network` and `--shell`, and
the [CLI reference](cli.md#gmlx-launch) describes each one.

Launch reads its container settings only from the config file in your
home folder, as [Launch](config.md#launch) explains.

The first container launch takes a few minutes. Launch numbers the steps
it runs, so that a slow download does not look like a hang. Every refusal
that needs neither the service nor the image, such as a folder launch will
not share, comes before these steps, so such a mistake never waits behind a
download.

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
[private home](glossary.md#private-home), its volumes, the
gmlx server, the forwarded ports and, when you turn it on, images from the
Mac clipboard. It does not see the rest of your files. Your keychain, SSH
keys, other projects and `gmlx.yaml` stay out of reach unless you share
them. The [security model](#security-model) lists the ways a session can
still reach the Mac.

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
- Credential folders such as `~/.ssh`, `~/.aws`, `~/.config/gh` and
  `~/.config/gmlx`, and any folder that holds one or lies inside one.
- gmlx's own data, under `~/.cache/gmlx` and `~/.local/share/gmlx`.

On a volume that ignores the case of names, as APFS does by default, launch
compares these paths the same way, so `~/.SSH` counts as `~/.ssh`.

`--mount PATH[:DST][:ro]` and
[`launch.container.mounts`](config.md#launchcontainermounts) share more
folders. A mount can name any folder. A mount of a credential folder is
honored, and launch prints a warning that names it. Only folders can be
shared, not single files, and a path that contains `,` or `=` is refused.
A mount of the current folder at its own path replaces the default share,
so `--mount .:ro` shares the project read-only.

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
its first-run steps once. [`seed`](config.md#launchcontainerclientsseed)
copies chosen files from your home into it. Launch also copies your git
`user.name` and `user.email` into the private home's `.gitconfig` when they
are missing there, so commits made in the container carry your name.

A seeded settings file can hold settings that only work on the Mac. A
`.gitconfig` with `credential.helper = osxkeychain` or commit signing, or a
Claude Code `settings.json` with hooks or a status line that run Mac
commands, fails in the container. Seed a copy without those settings.

### Git in a worktree

Git in the container works when you launch from the root of a repository.
A linked worktree or a submodule keeps its git folder outside its own
folder, so launch also shares that git folder, read-write, and prints a
line saying so.

Launch shares that git folder only when it names the project back. For a
worktree, the repository's entry for it names the project's `.git` file,
by an absolute path or by one relative to the entry. For a submodule, its
`core.worktree` names the project folder. The client can edit the `.git`
file in the share, but not the repository outside it, so it cannot use
this to reach another repository. Launch refuses the git folder when the
recorded path passes through a symbolic link in a share or a private home,
because the client can place such a link. After you delete a worktree by
hand, run `git worktree prune`, because its stale entry still names the old
path, and a new folder there would count as that worktree.

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
- A `./gmlx.yaml` the client writes in a share is read by the next
  `gmlx serve` from that folder. It can change where the server listens,
  turn off its key, or add a tool server command that the server runs on
  the Mac.
- The client can make the running server read its configuration again
  through the API. When that configuration, a model folder the server
  scans, or a model file it lists is inside a read-write share, the client
  can make the server load files of its choosing. Launch prints a warning
  when it finds any of these in a share.

The client gets the server's API key, which also unloads models and
reloads the server's configuration, not only chat. It can therefore
unload models that other clients are using.

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

The container has internet access unless you set
[`network: none`](config.md#launchcontainernetwork). On the default
network, a Mac service that listens on all addresses is reachable from the
container. The connection to the server needs no sudo, changes no network
setting and raises no firewall prompt, and the server sees
`Host: 127.0.0.1:<port>` on every request.

## The image

With no settings, launch builds one image per client from a recipe that
ships with gmlx. The image starts from Debian with Node.js and adds git,
ripgrep, curl, an SSH client and the client itself, installed from its
official source. It is built once, and built again when a gmlx upgrade
changes the recipe or when its `packages` change.

`--rebuild` builds the image again with fresh downloads, which also
updates the client. Launch prints the image's age at each start, and it
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
| `dsh` | A `--dsh-profile` must exist in the private home. The stdio profiles `acp`, `sdk` and `sdk-minimal` run only with `--no-container`. |

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
it works under `network: none` too. When the Mac service is down, the
connection closes at once and launch logs one line.

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
cannot write the Mac clipboard at all. An image arrives as PNG, and an
image of more than 20 MB is refused. Each image read adds a line to the
session log.

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
session whose launch was killed is cleaned up by the next launch of that
client, and a launch of any other client prints a line with its
`container stop` command, since the leftover virtual machine holds memory
until it stops.

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
`.gmlx-entry.log` for errors inside the container.

## The dry run

`--config-only` in container mode writes the client's configuration into
the private home and prints the `container run` command that a session
would use. The variables launch sets itself, such as `HOME`, `TERM`, `LANG`
and `IS_SANDBOX`, appear with their values. The client's own settings and
your [`env`](config.md#launchcontainerenv) entries appear by name only,
because they can hold keys. The dry run builds nothing, pulls
nothing and starts no container, and it reports whether the image and
volumes exist yet. Use it to inspect a session. The printed command cannot
run by itself, because the connection to the server exists only while
launch supervises the session.

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
request head in those 30 seconds. Once data flows, the connection has no
time limit, so a streamed answer is never cut. Idle connections therefore
cannot use up the file handles of the server that other clients share.

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
