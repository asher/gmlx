# Container mode

Container mode runs a client from [`gmlx launch`](launch.md) inside an Apple
container, a small Linux virtual machine that sees only the folders you
share with it. This page covers the first launch, what the container sees,
how sessions run and how to remove what container mode keeps on disk.

The model server stays on the Mac, because a Linux virtual machine has no
Metal GPU. Only the client moves into the container. That is the part that
runs shell commands and edits files, so it is the part worth isolating.

- [The first launch](#the-first-launch)
- [Turning on container mode](#turning-on-container-mode)
- [What does not work in a container](#what-does-not-work-in-a-container)
- [What the client sees](#what-the-client-sees)
- [Projects and sessions](#projects-and-sessions)
- [The image](#the-image)
- [The clients](#the-clients)
- [Browser apps](#browser-apps)
- [The shell](#the-shell)
- [Volumes](#volumes)
- [Forwarded ports](#forwarded-ports)
- [Clipboard images](#clipboard-images)
- [Signals, cleanup and logs](#signals-cleanup-and-logs)
- [The dry run](#the-dry-run)
- [Security](#security)
- [Removing container data](#removing-container-data)

## The first launch

Container mode needs Apple container 1.4 or newer, which
[Installation](installation.md#apple-container) covers. Run the client with
`--container` from the project folder that it should see:

```sh
cd ~/src/my-project
gmlx launch claude-code --container --model qwen3.8-27b-ud-q6
```

The first container launch takes a few minutes. Launch first checks the
shares, the image settings and the server, so a mistake in them never waits
behind a download. It then prints a numbered line, such as `step 2 of 3`,
for each of these steps that it runs:

- The container service starts for the first time. It asks to install a
  Linux kernel and downloads about 700 MB, so it needs a terminal. Without
  one, launch prints `container system start` and stops.
- Launch builds the client's image. The build downloads the Node base image
  once and takes a few minutes, and longer for hermes, elia and open-webui,
  which install Python packages. Behind a VPN that routes all traffic it
  fails, as
  [The image build cannot reach the network](troubleshooting.md#the-image-build-cannot-reach-the-network)
  explains.
- The client starts.

Before the client starts, launch prints what the session holds. One line
names the image, the client version in it and how long ago launch built it.
Each share, volume and forwarded port gets a line, and a share's line says
whether the client can change it. The client then takes over the terminal,
as it does on the Mac.

A busy web port, a problem in the image itself and a link in the client's
[private home](glossary.md#private-home) stop the launch only after these
steps. Later launches skip the first two steps and start the virtual
machine in about a second. After a Mac restart, launch starts the stopped
container service with one line and no question.

The image holds the client and a few common tools, as
[The image](#the-image) lists. Add the tools your project needs, such as
`python3`, `make` or `cargo`, with
[`packages`](container-images.md#extra-packages) before an agent runs your
tests.

## Turning on container mode

To make container mode the default, set
[`launch.container.enabled`](config.md#launchcontainerenabled) for every
client or for one client, and `--no-container` then runs a client on the
Mac for one launch. Launch reads its container settings only from the
config file in your home folder, as [Launch](config.md#launch) explains.

Each flag that only makes sense in a container, such as `--mount` or
`--shell`, turns on container mode by itself. The
[CLI reference](cli.md#gmlx-launch) lists these flags.

A container gets 4 CPUs and 4G of memory unless
[`cpus`](config.md#launchcontainercpus) and
[`memory`](config.md#launchcontainermemory) say otherwise.

By default the container can reach the internet and your local network.
[`network: none`](config.md#launchcontainernetwork), or `--network none` for
one launch, leaves it only the gmlx server and the
[forwarded ports](#forwarded-ports).

## What does not work in a container

Some features stop working in a container, most of them because they call
into the Mac:

- Pasting clipboard images, unless you turn on
  [clipboard images](#clipboard-images).
- Opening a browser or a URL from the client. The client prints the link
  in the terminal instead.
- Notifications, sounds and hooks that run Mac commands such as
  `osascript` or `afplay`.
- Credentials kept in the Keychain, such as `gh` logins or git's
  `osxkeychain` helper. Pass a token through
  [`env`](config.md#launchcontainerenv) instead.
- SSH keys. The client runs as root, and `ssh` looks for keys in root's own
  home folder in the container, not in the private home or a share. Turn on
  [`ssh_agent`](config.md#launchcontainerssh_agent) to let the client sign
  with the keys in your Mac's SSH agent.
- File locks between the virtual machine and anything outside it. A lock
  taken in the container blocks neither the Mac nor another container, so
  two programs on different sides can write one file at once. Two sessions
  therefore never share one private home, where the client's databases and
  settings could be corrupted. Another launch in a folder the running
  session shares joins it instead, and another project gets a home of its
  own.

## What the client sees

The client sees the folders you share, its
[private home](#the-private-home), its volumes, the gmlx server's
inference routes, the forwarded ports and, when you turn it on, images from
the Mac clipboard.

Your keychain, other projects and `gmlx.yaml` stay out of reach unless you
share them, and SSH keys need `ssh_agent`, as
[What does not work](#what-does-not-work-in-a-container) explains.
[Container security](container-security.md) describes the ways a client can
still reach the Mac.

### Shares

By default launch shares the current folder, read-write, at the same path
in the container, and the client starts there. Open WebUI and elia share
nothing by default.
[`launch.container.mount_cwd`](config.md#launchcontainermount_cwd) sets the
share in the config, and `--no-mount-cwd` turns it off for one launch.

Launch never shares the current folder by itself when it is one of these
folders. It asks you to launch from a project folder, or to pass
`--no-mount-cwd`, instead:

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

`--mount PATH[:DST][:ro]` and
[`launch.container.mounts`](config.md#launchcontainermounts) share more
folders, and the list above does not apply to them. A mount of a system
folder such as `/Applications` is shared as you wrote it. A mount of a
folder that holds credentials or files the Mac runs gets a warning that
names it.

A mount that holds or lies in gmlx's own data, settings or server state,
such as your home folder or `~/.config/gmlx`, is always refused. A client
could otherwise choose the config file that later launches read.

Only folders can be shared, not single files, and a path that contains `,`
or `=` is refused. A mount of the current folder at its own path replaces
the default share, so `--mount .:ro` shares the project read-only.

A mount path that is a symbolic link, or that passes through one, is
refused, because a client in an earlier, wider share could have replaced a
folder with a link. When you made the link yourself, write the real path
that the message gives instead, such as `/private/tmp/x` for `/tmp/x`.

Files the client writes in a share appear on the Mac with your user as
their owner, and modes and symbolic links are kept. Inside the container
every file of a share appears to belong to root, and the client runs as
root. A program that checks who owns its files, such as Postgres, therefore
cannot keep its data in a share, as [Postgres](container-images.md#postgres)
explains.

A file lock in a share works on one side only, as
[What does not work](#what-does-not-work-in-a-container) explains. Use a
database or a build folder that relies on locks from one side at a time,
either the Mac or the container.

macOS guards `~/Desktop`, `~/Documents`, `~/Downloads`, iCloud Drive and
`/Volumes`. The first share in one of them gets a notice, because macOS may
ask once whether the container runtime can read the folder. The container
waits until you answer.

### The private home

Each client gets a persistent home for each project at
`~/.local/share/gmlx/launch/<client>/projects/<project>/home`, which
appears at the same path in the container. The client's settings, sessions
and caches live there, so they survive from one launch to the next. Your
own `~/.claude`, `~/.pi` and other client folders are never shared, and
launch writes the client's configuration into the private home instead.

The client can change anything in its private home, so launch never
follows a symbolic link there when it reads or writes the configuration.
It stops the launch instead, as
[Launch will not follow a file in the private home](troubleshooting.md#launch-will-not-follow-a-file-in-the-private-home)
describes.

A new private home gets the client's configuration and the seeds. The
client's history there starts empty, and its history on the Mac, such as
Claude Code conversations or Open WebUI chats, stays on the Mac. The first
launch with a new home prints a line that says so.

For Claude Code, launch also marks the first-run steps of a new home as
done, with the theme from your Mac's `~/.claude.json` when it sets one.
Claude Code still asks whether to trust the project folder.

Launch copies your git `user.name` and `user.email` into the private home's
`.gitconfig` when they are missing there, so commits made in the container
carry your name.

[`seed`](config.md#launchcontainerclientsseed) copies chosen files or
folders from your home into the private home, with one line for each copy.
Launch records each seed outside the private home, so a client that
deletes its copy does not get a new one.

When you change a seed on the Mac, the next launch copies it again, unless
the client changed its copy too. Launch then keeps the copy and prints one
line that names `--reseed`, which copies every seed again and replaces the
copies.

The client reads every seeded file, so never seed a sign-in token. Launch
warns when it copies a file that can hold one, such as `~/.claude.json`,
`~/.gitconfig`, `~/.local/share/opencode/auth.json` or
`~/.config/goose/secrets.yaml`.

Launch refuses a seed whose real path lies outside your home folder, in a
credential folder or in gmlx's own data. It also refuses a seed that a link
leads out of a folder that a session shares or once shared read-write. A
seed that you reach through a link of your own gets a line with the path
the link leads to, and one seed copies at most 64 MiB and 10,000 files.

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

Launch shares that git folder only when its records name the project, and it
follows no symbolic link in them. Otherwise it prints a note that names the
fix, such as a `--mount` for the folder, `git worktree repair` for a
worktree moved by hand, or `git worktree list` to confirm that a worktree is
yours.

After you delete a worktree by hand, run `git worktree prune` in its
repository. The stale entry still names the old path, and a folder placed
there later would count as that worktree.

The rules for the current folder also apply to that git folder, and to the
repository that holds it. A worktree of a dotfiles repository in your home
folder therefore gets no git folder. Launched from a subfolder of a
repository, the client sees only that subfolder, and launch notes that git
needs the repository root.

## Projects and sessions

A session runs one client for one project in a virtual machine of its own.
Launch names the project after the current folder that the session shares:
the folder's name and 8 hex digits of a hash of its real path, such as
`my-project-1a2b3c4d`. A session that shares no current folder, such as
Open WebUI, elia or a launch with `--no-mount-cwd`, belongs to the
`default` project, and so does every [browser app](#browser-apps) session.

Sessions of different projects or different clients run side by side, and
each holds memory of its own, as [Limits](container-security.md#limits)
describes. Each project keeps its [private home](#the-private-home), so
`--continue`, history and the tools a client installs in its home stay with
the project.

Another launch of the same client joins the running session, instead of
starting a second virtual machine, when it runs in a folder that a share of
the session holds, such as the project folder or a subfolder. The launch
prints `joining the running <client> session for <folder>` and runs another
copy of the client in the same container, in the current folder, with its
own arguments after `--`.

When the shares of several sessions hold the folder, the session with the
longest share path takes the launch. A subfolder that no running session
shares is a project of its own. A launch from a folder that holds a running
session's project, such as its parent folder, starts a session of its own
that shares the same files from a second virtual machine. Launch from the
project folder instead, or wait until the other session has ended.

The copies share the private home inside one virtual machine, where file
locks work, as two copies share a home on the Mac. Two virtual machines
never share a home, for the reason
[What does not work](#what-does-not-work-in-a-container) gives.

A joining launch ignores the flags that chose the session's server and
model, such as `--model` or `--port`, with a note. It refuses a flag that
shapes a new session, such as `--mount` or `--image`, and a dsh profile
other than the running one. A second launch of a browser app prints the
address of the running app and opens it, unless
[`open_browser`](config.md#launchcontaineropen_browser) is `false`.

The session lasts until its last copy exits, and the terminal that started
it stays with it. When the first copy exits while others still run, that
terminal says the session stays open while they run, and it waits. A
Ctrl-C there asks for a second one, which ends the session and stops the
other copies. A launch that tries to join while the session ends stops with
a message, so launch again once the session has stopped.

A home stays until you remove it. [`gmlx doctor`](cli.md#gmlx-doctor) lists
each private home with its project folder, its size and its last use, and
[Removing container data](#removing-container-data) shows how to remove
one.

An older gmlx kept one home per client at
`~/.local/share/gmlx/launch/<client>/home`. The first launch of that client
in a project that has no home yet takes that home over, with its seed
record, and prints a line that says so. Other projects start with a new
home, and nothing is deleted.

## The image

With no settings, launch builds one image per client from a recipe that
ships with gmlx. The image starts from Debian with Node.js and adds git,
ripgrep, curl, an SSH client and the client itself, installed from its
official source. The recipe pins the base image by digest and each client
at one version, and checks each release download against its checksum.

The image is built once, and built again when a gmlx upgrade changes that
client's part of the recipe or the part every client shares, or when its
`packages` change. The line `rebuilding because` names the change. A newer
version of the client therefore arrives with a gmlx release, and
[A newer client](container-images.md#a-newer-client) shows how to run one
sooner.

`--rebuild` builds the image again without its cache, which picks up new
Debian packages. Each start names the image, the client version in it and
how long ago launch built it, or pulled it for an
[`image`](config.md#launchcontainerclientsimage) reference. After 30 days,
a note suggests `--rebuild` once a day, and the rebuild or a new pull ends
it.

When you remove a [`build`](config.md#launchcontainerclientsbuild) setting
or change an [`image`](config.md#launchcontainerclientsimage) reference, the
next launch deletes the images of the old setting. It keeps an image that a
running container uses and an image reference that launch did not pull.

[Custom container images](container-images.md) covers adding packages, your
own Containerfile and ready-made images. Any image works when it is for
Linux on arm64 and contains the command that runs, and launch refuses an
image for another architecture.

Launch checks an image of your own once for each command. A command that is
missing or cannot run stops the launch before the session starts, and under
`--shell` the check only warns. A program built for another system passes
the check and fails when the session starts, as
[A command is not in the image](troubleshooting.md#a-command-is-not-in-the-image)
describes.

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
| `dsh` | The `gmlx` and `web` profiles run as [browser apps](#browser-apps), and `headless` and your own profiles run in the terminal. |

Under `network: none`, Claude Code also gets
`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, so it stops trying to reach
Anthropic's servers.

The `acp`, `sdk` and `sdk-minimal` profiles of dsh serve a program over
stdio, so they need `--no-container`. A profile of your own must already
exist under `~/.dsh/profiles` in the private home, so create it from the
shell that `gmlx launch dsh --shell` opens.

## Browser apps

Open WebUI and the dsh web profiles open in your Mac browser, at port 3000
for Open WebUI and 3080 for dsh, or the next port when the gmlx server
already uses that one. The app listens on the container's own `127.0.0.1`,
and launch forwards it to the same port on the Mac, which accepts
connections only from the Mac itself.

Launch opens the browser when the app answers, which can take half a minute.
For Open WebUI it waits up to five minutes and then prints the address it
waited for. With
[`open_browser: false`](config.md#launchcontaineropen_browser), launch
prints the address when the app answers instead.

dsh puts a login token in its address, so launch reads the address from
dsh's own output and opens that. dsh's default workspace is in its private
home, so use Add workspace in the app to open the shared project folder.

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

While a session of that client runs for the project, or shares the current
folder, `--shell` [joins it](#projects-and-sessions) with a shell instead,
to look at what the agent is doing. The shell starts in the current folder
when a share of the session holds it, and in the session's working folder
otherwise. Like any joined copy, the shell keeps the session open until it
exits.

An image with no shell at all makes `--shell` stop with a message, so add a
shell to an image of your own to use it. What you install from the shell
outside the private home is gone when the session ends, as
[What persists](container-images.md#what-persists) explains.

## Volumes

A named volume is a disk of the container's own, for databases and large
caches. File owners, modes and locks work on it as on Linux, which a
[share](#shares) does not give, and work with many small files, such as
`npm install`, runs faster on a volume than in a share.

[`volumes`](config.md#launchcontainervolumes) entries take the form
`NAME:/path[:SIZE]`:

```yaml
launch:
  container:
    clients:
      claude-code:
        volumes: [claude-pg:/var/lib/postgresql:8G]
```

An entry under one client gets a volume for each
[project](#projects-and-sessions), named after the entry and the project,
such as `claude-pg-5e6f7a8b`, so a database belongs to one project. The
`default` project, and a project whose home came from an older gmlx, use
the name as written. An entry directly under `launch.container` keeps its
name in every project and client.

Launch creates a missing volume with the size in its entry, or the default
size that [`volumes`](config.md#launchcontainervolumes) gives. The disk
image on the Mac grows only as the container writes, up to that limit, and
the size is fixed when the volume is created.

Each session prints one line per volume with its name, its limit and the
space it takes on the Mac. Launch warns when the Mac disk has less free
space than the volumes could still use.

A volume created with a different size gets a warning with the command
that deletes it, and the next launch creates it again with the configured
size. Deleting a volume deletes its data, so the warning prints once for
each volume and size, and a volume you keep stays as it is.

The volume's root holds a `lost+found` folder, so put data in a subfolder.
Two containers never mount one volume at the same time. Launch refuses a
session whose volume another session or another container is using, so two
sessions that use the same volume do not run side by side.

`container system df` shows the space that all volumes and images take
on the Mac. Deleting files in a volume does not free that space. To give it
back, trim the volume from a container that no session is using:

```sh
container run --rm --cap-add CAP_SYS_ADMIN \
  --mount type=volume,source=claude-pg-5e6f7a8b,target=/v \
  docker.io/library/debian:bookworm-slim fstrim -v /v
```

A volume writes its data to the Mac with ordinary file syncs, not full
disk flushes. A database on a volume can lose its most recent commits if
the Mac loses power. Launch never deletes a volume, and
`container volume delete NAME` removes one with its data.

## Forwarded ports

[`forward`](config.md#launchcontainerforward) gives the client a service
that runs on the Mac, such as a database. Each port P in the list appears on
the container's own `127.0.0.1:P`, the same address as on the Mac, and it
works under `network: none` too.

Launch connects to the Mac's `127.0.0.1:P` only, never to `::1`, so a
service that listens only on `::1` cannot be forwarded. When the Mac service
is down, the connection closes at once and launch logs one line. A
forwarded port holds at most 32 open connections, and another connection
waits until one closes, so keep a client's connection pool at 32 or fewer.

A forwarded port gives the client that service with the rights of a local
user. Homebrew's Postgres and Redis accept local connections without a
password, so set a password or a limited role before you forward one. Never
forward a browser's remote debugging port, such as 9222, because that
gives the client your logged-in browser.

Launch refuses a forward of the gmlx server's port, since the container
already reaches the server, and a forward of a browser app's web port. A
program in the container cannot listen on a forwarded port, so run a service
either in the container on a volume or on the Mac with a forward, not both.
Two sessions can forward the same port, and each reaches the Mac service
with connections of its own.

## Clipboard images

With [`clipboard: images`](config.md#launchcontainerclipboard), the client
can paste images from the Mac clipboard. Clients on Linux paste an image by
running `xclip`, `xsel` or `wl-paste`, so launch puts replacements for those
three commands first on the client's `PATH`, and they read the Mac
clipboard:

```yaml
launch:
  container:
    clipboard: images
```

Paste with the client's own key for images, such as Ctrl-V in Claude Code.
Cmd-V pastes only text into a terminal.

The container gets images only. It cannot read clipboard text, and it
cannot write the Mac clipboard at all. An image arrives as PNG. A PNG over
20 MiB is refused, and so is an image in another type over 64 MiB, before
launch converts it. Each image the client reads adds a line to the session
log.

Clipboard images stay off by default, because the client can read the
clipboard image at any time during the session, not only when you paste.
With the setting off, a paste in the client fails or finds the image's own
clipboard tools. To hand over one image, save it into the shared folder
instead.

macOS can deny an app access to the clipboard. The replacement commands
then fail with a message that names the setting to change, Paste from
Other Apps under Privacy & Security in System Settings.

## Signals, cleanup and logs

A session is named `gmlx-<client>-<6 characters>`. Ctrl-C reaches the
client as it does on the Mac. When launch itself is stopped, it stops the
container, and a second stop ends the container at once.

After the last copy of the client exits, launch removes the container and
its session files. A container that is still there afterwards gets a line
with its `container delete --force` command. A session whose launch was
killed is cleaned up by the next launch of that client in that project. Any
other launch prints a line with the leftover container's `container stop`
command, since that virtual machine holds memory until it stops.

The launch that started the session exits with the first copy's exit
code, and [Exit codes](cli.md#exit-codes) lists the codes that launch and
the container return instead.

Launch writes the session log to
`~/.cache/gmlx/launch/last-<client>-<project>.log`, and the private home
holds `.gmlx-entry.log` for errors inside the container.

## The dry run

`--config-only` in container mode writes the client's configuration into
the private home and prints the `container run` command that a session
would use. The variables launch sets itself, such as `HOME`, `TERM`, `LANG`
and `IS_SANDBOX`, appear with their values. The client's own settings and
your [`env`](config.md#launchcontainerenv) entries appear by name only,
because they can hold keys.

The dry run builds nothing, pulls nothing and starts no container. It
reports whether the image and volumes exist yet, and whether the server
offers [session sockets](glossary.md#session-socket). The printed command
cannot run by itself, because the connection to the server exists only
while launch supervises the session.

When no server answers, the dry run still shows the image, the shares and
the volumes, and it says why it cannot show the client's configuration and
the command.

## Security

The container limits what the client can reach, not what it does in the
folders you share. A read-write share leads back to the Mac through files
the Mac runs later, such as `.git/hooks`, and through the server's config
and model folders. On the server, the client reaches only the inference
routes. [Container security](container-security.md) describes each of these
paths, the access you can turn on and the limits of a session.

## Removing container data

Uninstalling gmlx leaves container data in place, and each kind is removed
separately. Apple container keeps its images, volumes and Linux kernel in
`~/Library/Application Support/com.apple.container`. `gmlx doctor` reports
the space that volumes, private homes and images take, and it names the
images that no setting uses with the command that deletes them:

| Data | How to remove it |
|------|------------------|
| A private home | Run `gmlx launch <client> --remove-home` from the project folder, or delete its folder under `~/.local/share/gmlx/launch/<client>/projects`. |
| Volumes | Run `container volume delete NAME` for each volume, which deletes its data. |
| Images | Run `container image delete` on the `gmlx.invalid/launch-*` entries and the `@sha256:` entries of your `image` references, then `container image prune`. |
| The program launch runs in each container | Delete `~/.local/share/gmlx/launch/runtime`. |
| Apple container from Homebrew | Run `container system stop` and `brew uninstall container`, then delete `~/Library/Application Support/com.apple.container`. |
| Apple container from Apple's installer | Run `container system stop`, then `uninstall-container.sh -d`, which also deletes `~/Library/Application Support/com.apple.container`. |
