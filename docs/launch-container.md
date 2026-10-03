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
- [Sessions in the background](#sessions-in-the-background)
- [The shell](#the-shell)
- [Volumes](#volumes)
- [Forwarded ports](#forwarded-ports)
- [Clipboard images](#clipboard-images)
- [Signals, cleanup and logs](#signals-cleanup-and-logs)
- [The dry run](#the-dry-run)
- [Security](#security)
- [Removing container data](#removing-container-data)

## The first launch

Container mode needs Apple container 1.5 or newer, which
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
  one, launch prints `container system start` and stops. After a no or a
  failed download, launch names the command that installs the kernel, and
  after a Ctrl-C the next launch names it. When the service fails before
  it answers, launch names `container system logs` instead.
- Launch builds the client's image. The build downloads the Node base image
  once and takes a few minutes, and longer for hermes, elia and open-webui,
  which install Python packages. macOS can ask once to install Rosetta,
  which Apple's image builder uses, so accept it. Behind a VPN that routes
  all traffic the build fails, as
  [The image build cannot reach the network](troubleshooting.md#the-image-build-cannot-reach-the-network)
  explains.
- The client starts.

Before the client starts, launch prints what the session holds. One line
names the image, the client version in it and how long ago launch built it.
Each share, volume and forwarded port gets a line, and a share's line says
whether the client can change it. The client then takes over the terminal,
as it does on the Mac.

A problem in the image itself and a link in the client's
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
- The SSH keys in your Mac's `~/.ssh`. Turn on
  [`ssh_agent`](config.md#launchcontainerssh_agent) to let the client sign
  with the keys in an SSH agent on the Mac, or keep a key for the project
  in the private home, as [SSH in the container](#ssh-in-the-container)
  describes.
- File locks between the virtual machine and anything outside it. A lock
  taken in the container blocks neither the Mac nor another container, so
  two programs on different sides can write one file at once. Two sessions
  therefore never share one private home, where the client's databases and
  settings could be corrupted. Another launch of the client in the same
  project joins the running session instead, and another project gets a
  home of its own.

## What the client sees

The client sees the folders you share, its
[private home](#the-private-home), its volumes, the gmlx server's
inference routes, the forwarded ports and, when you turn it on, images from
the Mac clipboard.

Your keychain and other projects stay out of reach unless you share them.
Launch refuses a share of gmlx's own settings, and it warns for a
read-write share that holds the server's config file. Your Mac's SSH keys
need `ssh_agent`, as [What does not work](#what-does-not-work-in-a-container)
explains. [Container security](container-security.md) describes the ways a
client can still reach the Mac.

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
  `~/Library/LaunchAgents`, `~/.local/bin`, `~/.local/share/claude` and
  `/opt/homebrew`, and any folder that holds one of these or lies inside
  one.
- The folders of settings whose commands the Mac runs, such as `~/.vim`,
  `~/.emacs.d`, `~/.config/fish` and `~/.config/nvim`.
- The folders where clients keep their settings and history on the Mac,
  such as `~/.claude`, `~/.pi`, `~/.config/opencode` and
  `~/.cache/opencode`.
- The paths that variables such as `CLAUDE_CONFIG_DIR`, `GNUPGHOME`,
  `GH_CONFIG_DIR`, `ZDOTDIR` or `XDG_CONFIG_HOME` move these folders to.
- A folder that holds the real file of a settings link, such as
  `~/.gitconfig`, `~/.zshrc` or `~/.tmux.conf`, or of a link in one of these
  folders, such as `~/.ssh/config`. A dotfiles folder is the usual case. The
  message names the link, also when one of these folders is a link itself
  or lies in a linked folder, such as a `~/.config` that leads elsewhere.
- gmlx's own data, under `~/.cache/gmlx` and `~/.local/share/gmlx`.

For a project folder that such links lead into, the step in the message is
a read-only share with `--no-mount-cwd --mount PATH:ro`, or the removal of
each link, and it names up to three links. A link to credentials or to a
sign-in token, such as `~/.claude/.credentials.json`, gets no read-only
step, because the client could still read the file.

A link in `~/.local/bin`, `~/bin`, `~/.cargo/bin` or `$CARGO_HOME/bin` that
leads into a project, such as `~/.local/bin/mytool` that leads to
`~/src/mytool/mytool.py`, runs only when you run its name. So launch shares
that project, with a warning that names the link. The warning says to
remove the link, or to share the project read-only.

`--mount PATH[:DST][:ro]` and
[`launch.container.mounts`](config.md#launchcontainermounts) share more
folders, and the list above does not apply to them. A mount of a system
folder such as `/Applications` is shared as you wrote it.

Such a mount gets a warning that names what it holds when it holds
credentials, settings or files that the Mac runs, a client's own settings
or the temporary files of macOS. A read-write mount that holds a link to
one of these gets the warning too.

A mount that holds or lies in gmlx's own data, settings or server state,
such as your home folder or `~/.config/gmlx`, is always refused. So is a
read-write mount that holds a link on the way to one of them. A mount of
`$TMPDIR`, or of a folder in which gmlx keeps the session sockets of its
servers, is refused too. A client could otherwise choose the config file
that later launches read, or replace a socket.

Launch also refuses a read-write share that holds or lies in a program that
the Mac runs for gmlx, also as the default share. Examples are gmlx's Python
environment and the `git` that launch runs. A share that holds an editable
checkout in that environment, such as gmlx's own, only warns.
[Container security](container-security.md#shares-that-lead-back-to-the-mac)
lists these programs and the reason.

In the container, a mount cannot use `/proc`, `/sys`, `/dev`, `/opt/gmlx`,
`/var/host-services` or `/run/gmlx-session`, since the container needs
these paths for itself. A mount inside one of them, or at a folder that
holds one, such as `/run`, is refused too. Most images link `/var/run` to
`/run`, so a mount at `/var/run`, or at or inside `/var/run/gmlx-session`,
is refused as well.

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
and caches live there, so they survive from one launch to the next. Launch
writes the client's configuration into the private home, and your own
`~/.claude`, `~/.pi` and other client folders stay on the Mac.

The client can change anything in its private home, so launch never
follows a symbolic link there when it reads or writes the configuration.
It stops the launch instead, as
[Launch will not follow a file in the private home](troubleshooting.md#launch-will-not-follow-a-file-in-the-private-home)
describes.

A new private home gets the client's configuration and the seeds. The
client's history there starts empty, and its history on the Mac, such as
Claude Code conversations or Open WebUI chats, stays on the Mac. Launches
with a new home print a line that says so, until a session in that home
starts.

For Claude Code, launch also marks the first-run steps of a new home as
done, with the theme from your Mac's `~/.claude.json` when it sets one.
Claude Code still asks whether to trust the project folder.

Launch copies your git `user.name` and `user.email` into the private home's
`.gitconfig` when they are missing there, so commits made in the container
carry your name. When you change one on the Mac, the next launch updates the
copy and prints a line, unless you set another value in the container.

When launch finds no git that runs, it prints a line once a day. Examples
are `/usr/bin/git` without the command line tools, and a git outside the
folders that [Container security](container-security.md#shares-that-lead-back-to-the-mac)
names. The session then gets no git name and email, and no git folder for
a linked worktree. Run `xcode-select --install`, or `brew install git`.

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

### SSH in the container

`ssh` in the container uses the `.ssh` folder of the private home, because
launch links `/root/.ssh` to it when the session starts. The hosts you
accept, the keys and the `config` file therefore stay with the client and
the project. In each private home, ssh asks about a new host once.

Launch makes no link in two cases, and ssh then uses the `/root/.ssh` that
it finds. An image that has its own `/root/.ssh` keeps it. A
[share](#shares) or a [volume](#volumes) at `/root` gets no link, since the
link would stay there after the session ends.

The client can read and copy any key in the private home. Keep only a
deploy key for the project's repository there, with mode 600, and delete
it from the repository's deploy keys when you no longer need it.

With [`ssh_agent`](config.md#launchcontainerssh_agent), no key enters the
virtual machine, because an agent on the Mac signs for the client.
[Access you turn on](container-security.md#access-you-turn-on) compares the
risks of a key in the private home and of an agent.

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

A read-write `--mount` of exactly that git folder vouches for the worktree,
and launch records it. Later launches from the worktree then share the git
folder with no note. A read-only mount, or a share of a wider folder,
records nothing.

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
The project is the current folder that the session shares, and launch names
it after that folder, with 16 hex digits of a hash of its real path, such
as `my-project-1a2b3c4d5e6f7a8b`. When a `--mount` or a
`mounts` entry shares the current folder, the project is the folder of that
share, the longest when several hold the current folder.

Launches whose shares do not hold the current folder belong to the
`default` project. Examples are elia, and a launch with `--no-mount-cwd`
and no mount that holds the current folder. Open WebUI always belongs to
the `default` project, because it keeps one store of chats. Every other
client, dsh in the browser included, gets a project for each folder.

Sessions of different projects or different clients run side by side, and
each holds memory of its own, as [Limits](container-security.md#limits)
describes. Each project keeps its [private home](#the-private-home), so
`--continue`, history and the tools a client installs in its home stay with
the project.

Another launch of the same client in the project joins the running session,
instead of starting a second virtual machine. So does a launch from a
folder inside the session's project folder or one of its read-write shares,
and the session with the longest such folder takes the launch.

A launch that shares no folder by default, such as elia or one under
`mount_cwd: false`, joins in the same way. When a session of the client's
`default` project runs, such a launch joins that session instead. With
`--no-mount-cwd`, or a `--mount` of another folder, such a launch stays in
the `default` project.

The joining launch prints `joining the running <client> session for <folder>`
and runs another copy of the client in the same container, in the current
folder, with its own arguments after `--`.

A launch can join from a folder that the session does not share, as in the
`default` project. It then prints a line that names the folders the session
shares, and the copy starts in the session's working folder.

While a session is still starting or is ending, it stops a launch with a
message that says when to try again. A session is starting while its launch
starts the container service, prepares the image and boots the virtual
machine, and it is ending once its container stops. Another command that
holds the project, such as a `--remove-home` that waits for its answer,
stops a launch the same way.

When `container ls` fails, launch cannot tell the state of such a session,
so it stops. Its message says to try again once `container ls` works, or to
restart the container service, which also stops that session.

A launch from a folder that a session shares read-only, outside its project
folder, does not join that session, since a copy there could not change the
files. It starts a session of its own, and so does a launch from a folder
that holds a running session's project, such as its parent folder, or a
launch of another client in the project.

Two virtual machines then share the same files, and file locks do not reach
from one to the other, so launch prints a warning that names the other
session, also one that is still starting. Launch the same client from the
project folder instead, or wait until the other session has ended.

The copies share the private home inside one virtual machine, where file
locks work, as two copies share a home on the Mac. Two virtual machines
never share a home, for the reason
[What does not work](#what-does-not-work-in-a-container) gives.

A joining launch ignores the flags that chose the session's server and
model, such as `--model` or `--port`, with a note. A `--mount` joins when
the session already has that share, with the same folder, path and mode, so
the command that started a session joins it again.

Launch refuses any other flag that shapes a new session, such as `--image`,
and a dsh profile other than the running one. The refusal of a flag says how
to join, such as without the flag, and for `--mount` it lists the session's
shares in the form that joins. To use such a flag, end the session and
launch again.

The session lasts until its last copy exits, and the terminal that started
it stays with it. When the first copy exits while others still run, that
terminal says the session stays open while they run, and it waits. A
Ctrl-C there asks for a second one, which ends the session.

When the session ends, or launch stops its container, the other copies get
SIGHUP, as from a closed terminal. A copy that exits within 5 seconds prints
`the session ended in another terminal, so this copy of <command> stopped`.
Here `<command>` is the program that the copy runs, such as `claude` or
`bash`. A copy still running then stops with the container. Another Ctrl-C
in the first terminal stops them at once.

Closing the window of a joined copy, or stopping its launch, ends only that
copy. The copy gets SIGHUP, as a client on the Mac does, and it is killed
when it has not exited 10 seconds later.

A home stays until you remove it. [`gmlx doctor`](cli.md#gmlx-doctor) lists
each private home with its project folder, its size and its last use, and
[Removing container data](#removing-container-data) shows how to remove
one.

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

`--rebuild` builds the image again without its cache, which brings the
current Debian updates for every package in it. A newer Node.js arrives
with a gmlx release, since the recipe pins the base image.

Each start names the image, the client version in it and how long ago
launch built it, or pulled it for an
[`image`](config.md#launchcontainerclientsimage) reference. After 30 days, a
note suggests `--rebuild` once a day, and the rebuild or a new pull ends it.

When you remove a [`build`](config.md#launchcontainerclientsbuild) setting
or change an [`image`](config.md#launchcontainerclientsimage) reference, the
next launch deletes the images of the old setting. It keeps an image that a
running container uses, and an image reference that was in the image store
before launch pulled it, also after `--rebuild`.

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
- The word `image` runs the image's own ENTRYPOINT and CMD, where an
  ENTRYPOINT of `[""]` counts as none. The arguments after `--` replace
  CMD, and the container starts in the image's working folder when it sets
  one.

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
stdio, so they run only on the Mac with `--no-container --config-only`,
which prints the command to give that program. A profile of your own must already
exist under `~/.dsh/profiles` in the private home, so create it from the
shell that `gmlx launch dsh --shell` opens.

A program of your own runs in a container the same way, as a
[custom agent](launch-agents.md) defined under `launch.agents`, with the
server's address, a key and a model in environment variables.

## Browser apps

Open WebUI, the dsh web profiles and
[custom agents with a browser interface](launch-agents.md#a-browser-interface)
open in your Mac browser at `http://[::1]:<port>/`, the IPv6 loopback
address of the Mac. Each project gets a port of its own from 3100 to 3199,
which launch uses only in container mode, and Open WebUI keeps one port for
its one store of chats. The app listens on the container's own
`127.0.0.1`, and launch forwards it to the project's port at `::1`, which
accepts connections only from the Mac itself.

Only that address reaches the app. `http://localhost:<port>` gets a
`421 Misdirected Request` page with the text
`This app answers only at http://[::1]:<port>/. Open that address.`, and
`http://127.0.0.1:<port>` does not connect. The session log records such a
refusal, and [Browser app pages](container-security.md#browser-app-pages)
explains why the app uses `[::1]`.

The browser keeps saved data by address, and both apps tie their sign-ins
to the address, so a project keeps its port from one launch to the next
while it has a private home. When that port is not free, the app moves to
another one with a line that says so, and the app can ask you to sign in
again. The [dry run](#the-dry-run) names the port and records nothing.

A port that served the pages of another project goes to this one only when
no other port is free, because those pages can have left a service worker
and stored data there. Launch then prints what to do first, and it does
not open the browser.

Pages that are still open keep running and can store data again. So close
each tab and window of the address that launch names, and each window that
its pages opened, or quit the browser. Then clear the site data of that
address before you open the app.

When no port is free, launch stops. When other projects keep the ports, its
message tells you how to remove the private home of a project, which frees
its port, as
[No Mac port is free for a browser app](troubleshooting.md#no-mac-port-is-free-for-a-browser-app)
describes.

Launch opens the browser when the app answers, which can take half a minute.
For Open WebUI and custom agents, it waits up to five minutes and then
prints the address it waited for. With
[`open_browser: false`](config.md#launchcontaineropen_browser), launch
prints the address when the app answers instead.

A second launch of the app in the project prints the address of the running
app. It opens the address too, unless `open_browser` is `false` or the port
served the pages of another project.

dsh puts a login token in its address, so launch reads the address from
dsh's own output. dsh prints it with `127.0.0.1`, where the Mac does not
serve the app, so launch opens it with `[::1]` in its place and prints the
line `dsh answers on this Mac at http://[::1]:<port>/?token=...`. dsh's
default workspace is in its private home, so use Add workspace in the app to
open the shared project folder.

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

## Sessions in the background

`--detach` starts the session of Open WebUI, a dsh web profile or a custom
agent in the background and returns once the session runs. Until then,
launch shows the session's output. It then prints the app's address, the
file that takes the rest of the output, and the commands that list and end
the session:

```sh
gmlx launch open-webui --detach
gmlx launch --list
gmlx launch open-webui --stop
```

A detached session has no terminal, so launch refuses `--detach` for a
client that needs one, such as pi or claude-code, and with `--shell`. An
agent without a browser interface gets empty input, and a program that
reads its input gets end of file at once. The first start of the container
service asks whether to install a Linux kernel, so `--detach` also stops
when the service has never started. Run `container system start` once in a
terminal, then launch again.

All output of the session, from launch and from the client, goes to
`~/.cache/gmlx/launch/output-<client>-<project>.log`. The next detached
launch of the project empties that file. When the client's output in it
passes 64 MiB, launch empties the file and starts it with a line that
counts how often it did so, and the newest output stays. The output before the session starts, such as an
image build, does not count toward that limit. A launch that fails before
its session runs prints its message in your terminal and exits with its
own code, as [Exit codes](cli.md#exit-codes) lists.

Launch waits up to 2 minutes after the container starts for it to run, and
for a browser app up to 5.5 minutes after the start for the app to answer.
When that time ends, launch stops waiting, exits 0 and the session goes on.
A Ctrl-C ends the wait the same way, with exit code 130. In both cases
launch prints the path of the output file, and `gmlx launch --list` shows
whether the session runs. When the session ends during the wait, launch
says so and exits with the session's exit code.

A second `--detach` of a browser app in the project prints the address of
the running app, as any second launch does. For an agent without a browser
interface, a second `--detach` is refused, since a copy that joins the
session needs a terminal. Launch without `--detach` to join it. While the
first session still starts or ends, a second launch exits 75 and asks you
to try again. The same refusal comes while another command holds the
project with no session recorded, such as a `--remove-home` that waits for
its answer or a `--config-only` run.

`gmlx launch --list` prints a table of the sessions that start, run or end,
and `gmlx launch <name> --list` limits it to one client or agent. Each row
shows the project folder, the state, whether `--detach` started the
session, the browser app's address and when its launch started. The
address leaves out its query, which holds dsh's login token, and a second
launch of dsh in the project opens the whole address.

Below the table, launch prints the output file of each detached session
and the command that ends each session. A container left over from a
launch that is gone gets its `container stop` command. So does the session
of an agent that is no longer in `launch.agents`, since `gmlx launch`
refuses that name for anything but `--list`. `--stop` reads the launch
settings, so when they do not load, `--list` prints their error first. Each
session then gets its `--stop` command, to run once the settings load, and
its `container stop` command when it has a container.
[`gmlx status`](cli.md#gmlx-status) prints a line for each session too.

`--list` and `gmlx status` wait up to 5 seconds for the container service.
When it does not answer, a session whose state launch cannot tell shows as
`unknown`, a container left over is not listed, and launch prints the
error. `gmlx status` leaves the error out when it lists no session, unless
the service gave no answer in time, since a stopped service runs no
container.

`--stop` ends the session of the current project, whether `--detach`
started it or not. `--mount-cwd`, `--no-mount-cwd` and `--mount` choose
the project as they do for a launch, so the stop command that `--detach`
prints repeats the ones you passed. From a folder inside a session's
project folder, `--stop` ends the session that a launch from there would
join. It sends SIGTERM to the launch that runs the session, which stops the
container as closing its window does, and waits up to a minute for that
launch to exit. The project keeps its private home, volumes and port.

When the launch of a session is gone, `--stop` stops and deletes the
container itself, and a container that does not stop gets its
`container stop` command and exit code 75. With no session in the project,
`--stop` exits 0 and names the command that ends each other session of the
client. A session that has not ended after a minute gets exit code 75 with
the command that ends it. That is `container stop`, or `kill -KILL` with
the launch's process ID while the session has no container yet.

A launch of the project that has not recorded its session yet, such as the
launch in the background just after `--detach` started it, makes `--stop`
exit 75 with a request to try again. A `--remove-home` that waits for its
answer and a `--config-only` run do the same. To find a session, `--stop`
waits up to 5 seconds for the container list. A session that a record names
needs no list, since `--stop` ends it through its launch. When the list
fails, no record names a session and the service has not stopped, `--stop`
cannot tell whether a container is left over, so it ends nothing and prints
the error. The exit code is 69 when the service did not answer in time, and
1 for another error.

## The shell

`--shell` opens a shell in the same image, with the same shares, private
home and server connection as the client, instead of starting the client.
It runs `bash`, or `sh` when the image has no `bash`, and the arguments
after `--` go to the shell:

```sh
gmlx launch claude-code --shell
gmlx launch claude-code --shell -- -c "npm test"
```

While a session of that client runs for the project, `--shell`
[joins it](#projects-and-sessions) with a shell instead, to look at what
the agent is doing. The shell starts in the current folder when a share of
the session holds it, and in the session's working folder otherwise. Like
any joined copy, the shell keeps the session open until it
exits.

For a browser app, `--shell` starts the session with a shell and no app.
Launch prints the address of the app and the command that starts it, such
as `dsh ... --port 3100` with the project's port, so run that command in
the shell. With `command: image`, the printed command first changes to the
image's working folder, such as `cd /app/backend && bash start.sh` for the
official Open WebUI image. For a runtime agent, the command starts with
`uv run`, which brings the agent's environment up to date first, and names
the script by its full path in the guest. When a link lies on the path to
the script, the guest can reach another file than the Mac sees, so the
command keeps the script's name as the agent's `command` gives it.

The first start of dsh makes its profile from the `web` template, and a
line says to leave out `--from-default-profile web` after that. dsh then
prints its address with a login token and `127.0.0.1`, so open that address
with `[::1]` in place of `127.0.0.1`. While the shell runs, a second launch
prints the start command again, and `gmlx launch <client> --shell` opens
another shell in the session.

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
`default` project uses the name as written. An entry directly under
`launch.container` keeps its name in every project and client.

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
  docker.io/library/debian:trixie-slim fstrim -v /v
```

A volume writes its data to the Mac with ordinary file syncs, not full
disk flushes. A database on a volume can lose its most recent commits if
the Mac loses power. Launch deletes a volume in one case, when
`--remove-home` asks about a custom agent's
[dependency volume](launch-agents.md#the-dependency-volume) and you answer
yes. `container volume delete NAME` removes any volume with its data.

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

The replacement commands pass images only. They never read clipboard text
or write the Mac clipboard. Each image the client reads adds a line to the
session log. Your terminal can still let the client write the clipboard, as
[Your terminal](container-security.md#your-terminal) explains.

An image arrives as PNG. The clipboard can hold one image in several types,
and launch passes the first type that gives a PNG of at most 20 MiB. A type
over 64 MiB is not converted. When no type gives such a PNG, the paste
fails with a message that says why.

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
client as it does on the Mac. When the launch that started the session is
stopped, it stops the container, and the client gets up to 10 seconds to
exit. A second stop kills the container, at the latest when those 10
seconds end. A joining launch that is stopped ends only its own copy, as
[Projects and sessions](#projects-and-sessions) describes.

When a stop arrives while the virtual machine starts, it waits until the
container runs, for up to a minute, and then acts. A third stop, for a
container service that no longer answers, kills `container run` and puts
the terminal settings back as they were.

While launch prepares the image, a Ctrl-C, SIGTERM or SIGHUP ends the
launch once its clean-up is done. Launch ignores a second one, so that the
clean-up, such as the stop of the image builder, finishes. A second Ctrl-C
also prints a line that says to press Ctrl-C again to stop at once. A
third one ends the clean-up too. The exit code is 128 plus the signal
number, or 130 for Ctrl-C.

Closing the window of a launch stops that launch. To keep a browser app or
a custom agent running after you close the window, start it with
`--detach`, and end it later with `--stop`, as
[Sessions in the background](#sessions-in-the-background) describes.

Ctrl-Z cannot suspend a client in the container. The client goes on
running, and the first Ctrl-Z prints a line that says so. While the first
terminal waits for joined copies, Ctrl-Z there prints how to end the
session, and the wait goes on. Quit the client instead when you need the
terminal.

The dsh web profiles run with no terminal in the container, so Ctrl-Z
suspends launch itself. The page then cannot reach dsh, and dsh cannot
reach the server, until you run `fg`.

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
offers [session sockets](glossary.md#session-socket). For a browser app, it
names the Mac port that the app would take, and it records no port. The
printed command cannot run by itself, because the connection to the server
exists only while launch supervises the session.

When no server answers and launch cannot start one, the dry run still shows
the image, the shares and the volumes. It also says why it cannot show the
client's configuration and the command.

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
| A private home | Run `gmlx launch <client> --remove-home` in the project folder, also after you delete its folder under `~/.local/share/gmlx/launch/<client>/projects` by hand. |
| A custom agent's home and dependency volume | Run `gmlx launch <agent> --remove-home` in the project folder, which asks about both, as [Sessions and data](launch-agents.md#sessions-and-data) describes. |
| A custom agent's images | The `gmlx.invalid/launch-agent-*-build` and `gmlx.invalid/launch-runtime-python` images, which the Images row below covers. |
| What browser app pages left | Close the app's tabs and windows, then clear the site data that `--remove-home` names. After you delete all launch data, do so for `[::1]` ports 3100 to 3199. |
| Volumes | Run `container volume delete NAME` for each volume, which deletes its data. |
| Images | Run `container image delete` on the `gmlx.invalid/launch-*` images and unused `image` references with their `@sha256:` entries, then `container image prune`. |
| The image builder and its cache | Run `container builder stop`, then `container builder delete`. |
| The program launch runs in each container | Delete `~/.local/share/gmlx/launch/runtime`. |
| Session logs, output files and session folders | Delete `~/.cache/gmlx/launch` while no session runs. |
| Apple container from Homebrew | Run `container system stop` and `brew uninstall container`, then delete `~/Library/Application Support/com.apple.container`. |
| Apple container from Apple's installer | Run `container system stop`, then `uninstall-container.sh -d`, which also deletes `~/Library/Application Support/com.apple.container`. |
