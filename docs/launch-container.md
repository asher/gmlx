# Container mode

Container mode runs a client from [`gmlx launch`](launch.md) inside an Apple
container, a small Linux virtual machine that sees only the folders you
share with it. This page covers the first launch, what the client sees in
the container, how sessions run and how to remove what container mode
keeps on disk.

The model server stays on the Mac, because a Linux virtual machine has no
Metal GPU. Only the client moves into the container, since the client is
the program that runs shell commands and edits your files.

- [The first launch](#the-first-launch)
- [Turning on container mode](#turning-on-container-mode)
- [What does not work in a container](#what-does-not-work-in-a-container)
- [What the client sees](#what-the-client-sees)
- [Shares](#shares)
- [The private home](#the-private-home)
- [Volumes](#volumes)
- [Forwarded ports](#forwarded-ports)
- [Clipboard images](#clipboard-images)
- [The image](#the-image)
- [Clients in a container](#clients-in-a-container)
- [Browser apps](#browser-apps)
- [Projects and sessions](#projects-and-sessions)
- [Sessions in the background](#sessions-in-the-background)
- [Listing and ending sessions](#listing-and-ending-sessions)
- [The shell](#the-shell)
- [Signals, cleanup and logs](#signals-cleanup-and-logs)
- [The dry run](#the-dry-run)
- [Removing container data](#removing-container-data)

## The first launch

Container mode needs Apple container 1.5 or newer, which
[Installation](installation.md#apple-container) covers. Run the client with
`--container` from the project folder that it should see:

```sh
cd ~/src/my-project
gmlx launch claude-code --container --model qwen3.8-27b-ud-q6
```

The first container launch takes a few minutes. `launch` first checks the
shares, the image settings and the server, so a mistake in them never waits
behind a download. It then prints a numbered line, `step N of M`, for each
of these steps that it runs:

- Apple container gets its Linux kernel. `launch` asks whether to download
  it, about 700 MB once, and then starts the container service with the
  kernel. This step comes before `launch` starts the gmlx server, so the
  question does not wait behind a model load.
- `launch` builds the client's image. The build downloads the Node base
  image once and takes a few minutes, and longer for hermes, elia and
  open-webui, which install Python packages. Behind a VPN that routes all
  traffic, the build fails until you follow
  [The image build cannot reach the network](troubleshooting.md#the-image-build-cannot-reach-the-network).
- The client starts.

The kernel question needs a terminal. A no starts the service with no
kernel and stops the launch, and the next launch asks again. The next
launch also asks after a Ctrl-C, a failed download or
`brew services start container`, which all leave the service running with
no kernel. A launch with no terminal stops and names the one command that
downloads the kernel, as
[Apple container has no Linux kernel](troubleshooting.md#launch-says-apple-container-has-no-linux-kernel)
explains.

The images that `launch` builds are for arm64 and do not need Rosetta. On
a Mac without Rosetta, `launch` sets `rosetta = false` under `[build]` in
`~/.config/container/config.toml`, the settings file of Apple container,
before it starts the container service. Apple's image builder then starts
without Rosetta.

Before the client starts, `launch` lists what the session holds. One line
names the image, the client version in it and the age of the image. Each
share, volume and forwarded port gets a line, and a share's line says
whether the client can change it. The client then takes over the terminal,
as it does on the Mac. Wait for the client's prompt before you type, since
Apple container drops what you type before the container starts.

A problem in the image itself and a link in the client's
[private home](glossary.md#private-home) stop the launch only after these
steps. Later launches skip the first two steps and start the virtual
machine in about a second. After a Mac restart, `launch` starts the stopped
container service with one line and no question.

The image holds the client and a few common tools, which
[The image](#the-image) lists. Before an agent runs your tests, add the
tools your project needs, for example `python3`, `make` or `cargo`, with
[`packages`](container-images.md#extra-packages).

## Turning on container mode

To make container mode the default, set
[`launch.container.enabled`](config.md#launchcontainerenabled) for every
client or for one client. `--no-container` then runs a client on the Mac
for one launch. `launch` reads its container settings only from the
[config file in your home folder](config.md#launch).

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
- Credentials kept in the Keychain, for example `gh` logins or git's
  `osxkeychain` helper. Pass a token through
  [`env`](config.md#launchcontainerenv) instead.
- The SSH keys in your Mac's `~/.ssh`. Turn on
  [`ssh_agent`](config.md#launchcontainerssh_agent) to let the client sign
  with the keys in an SSH agent on the Mac, or keep a key for the project
  in the private home. See [SSH in the container](#ssh-in-the-container).
- File locks between the virtual machine and anything outside it. A lock
  taken in the container blocks neither the Mac nor another container, so
  two programs on different sides can write one file at once. Two sessions
  therefore never share one private home, where the client's databases and
  settings could be corrupted. Another launch of the client in the same
  project joins the running session instead, and another project gets a
  home of its own.
- The dependencies that the Mac built in a shared project, for example
  `.venv` or a `node_modules` with native modules. They are builds for
  macOS, so the container cannot run them, and an install in the container
  replaces them with Linux builds that the Mac cannot run. After an install
  in the container, install the dependencies again on the Mac before you
  run the project there.
- The Mac's GPU for the client's own code. PyTorch, MLX or any other code
  that runs in the container uses the CPU. The model itself runs on the
  gmlx server, which uses the GPU.

## What the client sees

The client sees the folders you [share](#shares), its
[private home](#the-private-home), its [volumes](#volumes), the gmlx
server's inference routes, the [forwarded ports](#forwarded-ports) and,
when you turn it on, [images from the Mac clipboard](#clipboard-images).
Your keychain and other projects stay out of reach unless you share them.

What the client writes in a share, its private home or a volume stays after
the session. Everything else it writes in the container is gone when the
session ends. A package that `apt-get install` or `npm install -g` adds
from `--shell` lands in the image's own folders, so it is gone at the next
session. To keep a tool from one session to the next, put it in the image
with one of the methods in [Custom container images](container-images.md).

The container limits what the client can reach. It does not limit what the
client does in the folders you share, and some of those changes run on the
Mac later. [Container security](container-security.md) describes each way
back to the Mac, and it opens with the steps for a session on code you do
not trust.

## Shares

By default `launch` shares the current folder, read-write, at the same path
in the container, and the client starts there. Open WebUI and elia share
nothing by default.
[`launch.container.mount_cwd`](config.md#launchcontainermount_cwd) sets the
share in the config, and `--no-mount-cwd` turns it off for one launch.

### Folders launch does not share

`launch` never shares one of these folders as the current folder. It asks
you to launch from a project folder, or to pass `--no-mount-cwd`, instead:

- Your home folder, any folder that holds it, and the system folders `/`,
  `/Users`, `/Volumes`, `/private`, `/tmp`, `/var`, `/opt`, `/usr`,
  `/Library`, `/System`, `/Applications`, `/private/tmp`, `/private/var`,
  `/System/Volumes` and `/System/Volumes/Data`.
- The temporary folders of macOS, `$TMPDIR` and everything under
  `/private/var/folders`, which hold the temporary files of every program.
  A scratch project under `/private/tmp` can be shared.
- The credentials in `~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.azure`,
  `~/.config/gcloud`, `~/.kube`, `~/.docker`, `~/.password-store`,
  `~/Library/Keychains`, `~/.netrc`, `~/.config/gh`, `~/.npmrc`,
  `~/.git-credentials`, `~/.config/gmlx`, `~/.cache/huggingface` and
  `~/.codex`, and any folder that holds one of them or lies inside one.
- The files that the Mac runs in `~/Library/LaunchAgents`, `~/.config/git`,
  `~/.local/bin`, `~/bin`, `~/Library/Application Support`, `~/.cargo`,
  `/opt/homebrew`, `/usr/local` and `~/.local/share/claude`, and any folder
  that holds one of them or lies inside one.
- The settings folders of the shells and editors, `~/.config/fish`,
  `~/.vim`, `~/.config/vim`, `~/.config/nvim`, `~/.local/share/nvim`,
  `~/.config/tmux`, `~/.emacs.d` and `~/.config/emacs`, whose commands the
  Mac runs.
- The folders where the clients keep their settings and history on the
  Mac, which are `~/.claude`, `~/.pi`, `~/.omp`, `~/.hermes`,
  `~/.open-webui`, `~/.dsh`, `~/.config/goose`, `~/.config/opencode`,
  `~/.local/share/opencode`, `~/.cache/opencode`, `~/.opencode`,
  `~/.config/elia`, `~/.config/aichat` and
  `~/Library/Application Support/aichat`.
- The paths that `CLAUDE_CONFIG_DIR`, `GNUPGHOME`, `GH_CONFIG_DIR`,
  `ZDOTDIR`, `XDG_CONFIG_HOME` and similar variables move these folders to.
- A folder that holds the real file of a settings link, such as
  `~/.gitconfig`, `~/.zshrc` or `~/.tmux.conf`, or of a link in one of these
  folders, such as `~/.ssh/config`. A dotfiles folder is the usual case.
  The message names the link even when one of these folders is itself a
  link or lies in a linked folder, for example a `~/.config` that leads
  elsewhere.
- gmlx's own data, under `~/.cache/gmlx` and `~/.local/share/gmlx`.

For a project folder that these links lead into, the message offers a
read-only share with `--no-mount-cwd --mount PATH:ro`, or the removal of
each link, and it names up to three links. A link to credentials or to a
sign-in token, for example `~/.claude/.credentials.json`, gets no read-only
step, because the client could still read the file.

A link in `~/.local/bin`, `~/bin`, `~/.cargo/bin` or `$CARGO_HOME/bin` that
leads into a project, such as `~/.local/bin/mytool` that leads to
`~/src/mytool/mytool.py`, runs only when you run its name. So `launch`
shares that project, with a warning that names the link. The warning says
to remove the link, or to share the project read-only.

### More shares

`--mount PATH[:DST][:ro]` and
[`launch.container.mounts`](config.md#launchcontainermounts) share more
folders, and the list above does not apply to them. A mount of a system
folder such as `/Applications` is shared as you wrote it. A mount that holds
credentials, settings or files that the Mac runs, a client's own settings
or the temporary files of macOS gets a warning that names what it holds. A
read-write mount that holds a link to one of these gets the warning too.

`launch` refuses these mounts:

- A mount that holds or lies in gmlx's own data, settings or server state,
  for example your home folder or `~/.config/gmlx`, and a read-write mount
  that holds a link on the way to one of them. A client could otherwise
  choose the config file that later launches read.
- A mount of `$TMPDIR`, or of a folder in which gmlx keeps the session
  sockets of its servers, where a client could replace a socket.
- A read-write mount, the default share included, that holds or lies in a
  program that the Mac runs for gmlx. Examples are gmlx's Python
  environment and the `git` that `launch` runs. A share that holds an
  editable checkout in that environment only warns, and
  [Shares that lead back to the Mac](container-security.md#shares-that-lead-back-to-the-mac)
  lists these programs.
- A mount at `/proc`, `/sys`, `/dev`, `/opt/gmlx`, `/var/host-services` or
  `/run/gmlx-session` in the container, inside one of them, or at a folder
  that holds one, such as `/run`. The container needs these paths for
  itself. Most images link `/var/run` to `/run`, so the same rule covers
  `/var/run` and the paths at or inside `/var/run/gmlx-session`.
- A mount path that is a symbolic link, or that passes through one,
  because a client in an earlier, wider share could have replaced a folder
  with a link. When you made the link yourself, write the real path that
  the message gives, for example `/private/tmp/x` for `/tmp/x`.
- A single file, and a path that contains `,` or `=`. Only folders can be
  shared.

A mount of the current folder at its own path replaces the default share,
so `--mount .:ro` shares the project read-only.

### Files in a share

Files the client writes in a share appear on the Mac with your user as
their owner, and modes and symbolic links are kept. Inside the container
every file of a share appears to belong to root, and the client runs as
root. A program that checks who owns its files, such as Postgres, therefore
cannot keep its data in a share. The [Postgres](container-recipes.md#postgres)
recipe puts it on a volume.

A file lock in a share works on one side only, as
[What does not work in a container](#what-does-not-work-in-a-container)
says. Use a database or a build folder that relies on locks from one side
at a time, either the Mac or the container.

macOS guards `~/Desktop`, `~/Documents`, `~/Downloads`, iCloud Drive and
`/Volumes`. The first share in one of them gets a notice, because macOS may
ask once whether the container runtime can read the folder. The container
waits until you answer.

### Git in a worktree

Git in the container works when you launch from the root of a repository.
A linked worktree or a submodule keeps its git folder outside its own
folder, so `launch` also shares that git folder and names its repository.
The git folder is read-write, or read-only when the share that holds the
repository root is read-only, as with `--mount .:ro`.

`launch` shares that git folder only when its records name the project, and
it follows no symbolic link in them. Otherwise it prints a note with the
fix, for example a `--mount` for the folder, `git worktree repair` for a
worktree moved by hand, or `git worktree list` to confirm that a worktree is
yours.

A read-write `--mount` of exactly that git folder tells `launch` that the
worktree is yours, and `launch` records it. Later launches from the
worktree then share the git folder with no note. A read-only mount, or a
share of a wider folder, records nothing.

After you delete a worktree by hand, run `git worktree prune` in its
repository. The stale entry still names the old path, and a folder placed
there later would count as that worktree.

The rules for the current folder also apply to that git folder, and to the
repository that holds it. A worktree of a dotfiles repository in your home
folder therefore gets no git folder. Launched from a subfolder of a
repository, the client sees only that subfolder, and `launch` notes that
git needs the repository root.

## The private home

Each client gets a persistent home for each project at
`~/.local/share/gmlx/launch/<client>/projects/<project>/home`, which
appears at the same path in the container. The client's settings, sessions
and caches live there, so they survive from one launch to the next.
`launch` writes the client's configuration into the private home, and your
own `~/.claude`, `~/.pi` and other client folders stay on the Mac.

The client can change anything in its private home, so `launch` never
follows a symbolic link there when it reads or writes the configuration.
It stops the launch instead, and
[Launch will not follow a file in the private home](troubleshooting.md#launch-will-not-follow-a-file-in-the-private-home)
gives the fix.

A new private home gets the client's configuration and the seeds. The
client's history there starts empty, and its history on the Mac, such as
Claude Code conversations or Open WebUI chats, stays on the Mac. Until a
session in a new home starts, each launch says that its history starts
empty.

For Claude Code, `launch` also marks the first-run steps of a new home as
done, with the theme from your Mac's `~/.claude.json` when it sets one.
Claude Code still asks whether to trust the project folder.

`launch` copies your git `user.name` and `user.email` into the private
home's `.gitconfig` when they are missing there, so commits made in the
container carry your name. When you change one on the Mac, the next launch
updates the copy and says so, unless you set another value in the
container.

When `launch` finds no git that runs, it warns once a day. Examples are
`/usr/bin/git` without the command line tools, and a git outside the
folders that
[Container security](container-security.md#shares-that-lead-back-to-the-mac)
names. The session then gets no git name and email, and no git folder for
a linked worktree. Run `xcode-select --install`, or `brew install git`.

A home stays until you remove it. [`gmlx doctor`](cli.md#gmlx-doctor) lists
each private home with its project folder, its size and its last use, and
[Removing container data](#removing-container-data) shows how to remove
one.

### Seeds

[`seed`](config.md#launchcontainerclientsseed) copies chosen files or
folders from your home into the private home, and `launch` names each
copy. `launch` records each seed outside the private home, so a client that
deletes its copy does not get a new one. To copy the instructions and
skills that the client reads, use `--seed-instructions`, as
[Instructions and skills](#instructions-and-skills) shows.

When you change a seed on the Mac, the next launch copies it again, unless
the client changed its copy too. `launch` then keeps the copy and suggests
`--reseed`, which copies every seed again and replaces the copies.

The client reads every seeded file, so never seed a sign-in token. `launch`
warns when it copies a file that can hold one. These files are
`~/.claude.json`, `~/.claude/.credentials.json`, `~/.gitconfig`,
`~/.local/share/opencode/auth.json`, `~/.config/goose/secrets.yaml`,
`~/.pi/agent/auth.json`, `~/.omp/agent/agent.db`, `~/.hermes/auth.json` and
`~/.hermes/.env`.

`launch` refuses a seed whose real path lies outside your home folder, in a
credential folder or in gmlx's own data. It also refuses a seed that a link
leads out of a folder that a session shares or once shared read-write. For
a seed that you reach through a link of your own, `launch` names the path
the link leads to. One seed copies at most 64 MiB and 10,000 files.

A seeded settings file can hold settings that only work on the Mac. A
`.gitconfig` with `credential.helper = osxkeychain` or commit signing, or a
Claude Code `settings.json` with hooks or a status line that run Mac
commands, fails in the container. Seed a copy without those settings.

### Instructions and skills

A launch with `--seed-instructions` adds the client's global instruction,
skill, command and subagent files to its seeds, so the client in the
container follows the same instructions as on the Mac. `launch` copies only
the files that exist in your home folder, and they then behave as any other
seed. The lists hold no sign-in tokens and no settings files, and `launch`
writes the client's settings itself.

These are the files and folders that each client reads, from your home
folder:

- claude-code: `~/.claude/CLAUDE.md`, and the folders `rules`, `skills`,
  `commands`, `agents` and `output-styles` in `~/.claude`.
- opencode: `~/.config/opencode/AGENTS.md`, `~/.claude/CLAUDE.md`, the
  folders `skill`, `skills`, `command`, `commands`, `agent`, `agents`,
  `mode` and `modes` in `~/.config/opencode`, `~/.claude/skills` and
  `~/.agents/skills`.
- pi: `AGENTS.md`, `CLAUDE.md`, `SYSTEM.md`, `APPEND_SYSTEM.md` and the
  folders `skills` and `prompts` in `~/.pi/agent`, and `~/.agents/skills`.
- omp: `AGENTS.md`, `SYSTEM.md`, `RULES.md` and the folders `rules`,
  `instructions`, `skills`, `commands`, `prompts` and `agents` in
  `~/.omp/agent`, and `AGENTS.md` and the folders `skills`, `rules`,
  `prompts` and `commands` in `~/.agents`.
- hermes: `~/.hermes/SOUL.md` and `~/.hermes/skills`.
- goose: `.goosehints`, `AGENTS.md` and the folders `skills`, `agents` and
  `recipes` in `~/.config/goose`, `AGENTS.md` and the folders `skills`,
  `agents` and `recipes` in `~/.agents`, and `~/.claude/skills` and
  `~/.claude/agents`.
- dsh: `~/.dsh/AGENTS.md`, `~/.dsh/skills` and `~/.agents/skills`.
- aichat: `~/.config/aichat/roles` and `~/.config/aichat/macros`.
- elia and open-webui: none.

A custom agent is not a client, so `--seed-instructions` refuses it. List
the files that an agent needs in `launch.agents.<name>.seed`, as
[`seed`](config.md#launchcontainerclientsseed) describes.

### SSH in the container

`ssh` in the container uses the `.ssh` folder of the private home, because
`launch` links `/root/.ssh` to it when the session starts. The hosts you
accept, the keys and the `config` file therefore stay with the client and
the project. In each private home, ssh asks about a new host once.

`launch` makes no link in two cases, and ssh then uses the `/root/.ssh`
that it finds. An image that has its own `/root/.ssh` keeps it. A
[share](#shares) or a [volume](#volumes) at `/root` gets no link, since the
link would stay there after the session ends.

The client can read and copy any key in the private home. Keep only a
deploy key for the project's repository there, with mode 600, and delete
it from the repository's deploy keys when you no longer need it.

With [`ssh_agent`](config.md#launchcontainerssh_agent), no key enters the
virtual machine, because an agent on the Mac signs for the client.
[Access you turn on](container-security.md#access-you-turn-on) compares the
risks of a key in the private home and of an agent.

## Volumes

A named volume is a separate disk for the container, for databases and
large caches. File owners, modes and locks work on it as on Linux, which a
[share](#shares) does not give, and work with many small files, like
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
[project](#projects-and-sessions), named after the entry and a hash of the
project, for example `claude-pg-9c0d1e2f`, so a database belongs to one
project. The `default` project uses the name as written. An entry directly
under `launch.container` keeps its name in every project and client.

`launch` creates a missing volume with the size in its entry, or the
default size that [`volumes`](config.md#launchcontainervolumes) gives. The
disk image on the Mac grows only as the container writes, up to that limit,
and the size is fixed when the volume is created.

Each session lists every volume with its name, its limit and the space it
takes on the Mac. `launch` warns when the Mac disk has less free space than
the volumes could still use.

A volume created with a different size gets a warning with the command
that deletes it, and the next launch creates it again with the configured
size. Deleting a volume deletes its data, so the warning comes once for
each volume and size, and a volume you keep stays as it is.

The volume's root holds a `lost+found` folder, so put data in a subfolder.
Two containers never mount one volume at the same time. `launch` refuses a
session whose volume another session or another container is using.

`container system df` shows the space that all volumes and images take
on the Mac. Deleting files in a volume does not free that space. To give it
back, trim the volume from a container that no session is using. Take the
volume's name from the session's volume line or from `container volume ls`:

```sh
container run --rm --cap-add CAP_SYS_ADMIN \
  --mount type=volume,source=claude-pg-9c0d1e2f,target=/v \
  docker.io/library/debian:trixie-slim fstrim -v /v
```

A volume writes its data to the Mac with ordinary file syncs, not full
disk flushes. A database on a volume can lose its most recent commits if
the Mac loses power. `launch` deletes a volume in one case, when
`--remove-home` asks about a custom agent's
[dependency volume](launch-agents.md#the-dependency-volume) and you answer
yes. `container volume delete NAME` removes any volume with its data.

## Forwarded ports

[`forward`](config.md#launchcontainerforward) gives the client a service
that runs on the Mac, such as a database. Each port P in the list appears at
`127.0.0.1:P` in the container, the same address as on the Mac, and it works
under `network: none` too.

`launch` connects to the Mac's `127.0.0.1:P` only, never to `::1`, so a
service that listens only on `::1` cannot be forwarded. When the Mac service
is down, the connection closes at once and `launch` logs it. A forwarded
port holds at most 32 open connections, and another connection waits until
one closes, so keep a client's connection pool at 32 or fewer.

A forwarded port gives the client that service with the rights of a local
user. Homebrew's Postgres and Redis accept local connections without a
password, so set a password or a limited role before you forward one. Never
forward a browser's remote debugging port, such as 9222, because that
gives the client your logged-in browser.

`launch` refuses a forward of the gmlx server's port, since the container
already reaches the server, and a forward of a browser app's web port. A
program in the container cannot listen on a forwarded port, so run a service
either in the container on a volume or on the Mac with a forward, not both.
Two sessions can forward the same port, and each opens separate connections
to the Mac service.

## Clipboard images

With [`clipboard: images`](config.md#launchcontainerclipboard), the client
can paste images from the Mac clipboard. Clients on Linux paste an image by
running `xclip`, `xsel` or `wl-paste`, so `launch` puts replacements for
those three commands first on the client's `PATH`, and they read the Mac
clipboard:

```yaml
launch:
  container:
    clipboard: images
```

Paste with the client's key for images, which is Ctrl-V in Claude Code.
Cmd-V pastes only text into a terminal.

The replacement commands pass images only. They never read clipboard text
or write the Mac clipboard, and the session log records each image that the
client reads. Your terminal can still let the client write the clipboard,
which [Your terminal](container-security.md#your-terminal) shows how to
turn off.

An image arrives as PNG. The clipboard can hold one image in several types,
and `launch` passes the first type that gives a PNG of at most 20 MiB. A
type over 64 MiB is not converted. When no type gives a PNG within these
limits, the paste fails with a message that says why.

Clipboard images stay off by default, because the client can read the
clipboard image at any time during the session, not only when you paste.
With the setting off, a paste in the client fails, or it uses the clipboard
tools that the image holds. To hand over one image, save it into the shared folder
instead.

macOS can deny an app access to the clipboard. The replacement commands
then fail with a message that names the setting to change, Paste from
Other Apps under Privacy & Security in System Settings.

## The image

With no settings, `launch` builds one image per client from a recipe that
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
`launch` built it, or pulled it for an
[`image`](config.md#launchcontainerclientsimage) reference. After 30 days, a
note suggests `--rebuild` once a day, and the rebuild or a new pull ends it.

When you remove a [`build`](config.md#launchcontainerclientsbuild) setting
or change an [`image`](config.md#launchcontainerclientsimage) reference, the
next launch deletes the images of the old setting. It keeps an image that a
running container uses, and an image reference that was in the image store
before `launch` pulled it, even after `--rebuild`.

[Custom container images](container-images.md) covers extra packages, your
own Containerfile, ready-made images, what an image needs and the command
that runs in it.

## Clients in a container

Each client gets the same configuration as on the Mac, written into its
private home. A client that merges into its own files, such as pi or
hermes, changes only the copies in the private home. Three clients change
further in a container:

| Client | In a container |
|--------|----------------|
| `claude-code` | `launch` sets `IS_SANDBOX=1`, so `-- --dangerously-skip-permissions` works as root, and turns off the auto-updater. |
| `open-webui` | The app opens as a [browser app](#browser-apps), and `CORS_ALLOW_ORIGIN` names its `http://[::1]:<port>` address. |
| `dsh` | The `gmlx` and `web` profiles run as browser apps, and `headless` and your own profiles run in the terminal. |

Under `network: none`, Claude Code also gets
`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, so it stops trying to reach
Anthropic's servers. A `CLAUDE_CODE_MAX_CONTEXT_TOKENS` that you set comes
from [`env`](config.md#launchcontainerenv). The [dry run](#the-dry-run)
shows the value that Claude Code gets on a separate line, because the
command it prints names the client's variables without their values.

Variables that you export on the Mac reach the container only through an
[`env`](config.md#launchcontainerenv) entry. In the `env` of
`launch.container.clients.open-webui`, write `WEBUI_AUTH=false` and
`CORS_ALLOW_ORIGIN=<addresses>` with their values, because a bare
`CORS_ALLOW_ORIGIN` entry keeps the value that `launch` sets. Open WebUI
keeps its chats in `~/.open-webui` in the private home, so the chats of the
app on the Mac do not appear in the container.

The `acp`, `sdk` and `sdk-minimal` profiles of dsh run only on the Mac
with `--no-container --config-only`, as [dsh](launch.md#dsh) explains. A
profile that you made must already exist under `~/.dsh/profiles` in the
private home, so create it from the shell that `gmlx launch dsh --shell`
opens. dsh's default workspace is in its private home, so use Add
workspace in the app to open the shared project folder.

[Container recipes](container-recipes.md) adds pi packages, dsh plugins and
tool servers to a client, and web search to Open WebUI. A
[custom agent](launch-agents.md) runs a program of your own in a container.

## Browser apps

Open WebUI, the dsh web profiles and
[custom agents with a browser interface](launch-agents.md#a-browser-interface)
open in your Mac browser at `http://[::1]:<port>/`, the IPv6 loopback
address of the Mac. Each project gets a separate port from 3100 to 3199,
which `launch` uses only in container mode, and Open WebUI keeps one port
for its one store of chats.

The app listens on `127.0.0.1` in the container, and `launch` forwards it
to the project's port at `::1`, which accepts connections only from the Mac
itself.

Only that address reaches the app. `http://localhost:<port>` gets a
`421 Misdirected Request` page with the text
`This app answers only at http://[::1]:<port>/. Open that address.`, and
`http://127.0.0.1:<port>` does not connect. The session log records each
refusal, and [Browser app pages](container-security.md#browser-app-pages)
gives the reason for `[::1]`.

The browser keeps saved data by address, and Open WebUI and dsh tie their
sign-ins to the address. A project therefore keeps its port from one
launch to the next while it has a private home. When that port is not
free, the app moves to another one, `launch` says so, and the app can ask
you to sign in again. The [dry run](#the-dry-run) names the port and
records nothing.

A port that served the pages of another project goes to this one only when
no other port is free, because those pages can have left a service worker
and stored data there. `launch` then prints what to do first and does not
open the browser. [Browser app pages](container-security.md#browser-app-pages)
explains the clean-up.

When no port is free, `launch` stops. When other projects hold the ports,
the message says how to remove the private home of one of them, which
frees its port. See
[No Mac port is free for a browser app](troubleshooting.md#no-mac-port-is-free-for-a-browser-app).

`launch` opens the browser when the app answers, which can take half a
minute. For Open WebUI and custom agents, it waits up to 5 minutes and then
prints the address it waited for. With
[`open_browser: false`](config.md#launchcontaineropen_browser), `launch`
prints the address when the app answers instead.

A second launch of the app in the project prints the address of the running
app. It opens the address too, unless `open_browser` is `false` or the port
served the pages of another project.

dsh puts a login token in its address, so `launch` reads the address from
the output of dsh. dsh prints it with `127.0.0.1`, where the Mac does not
serve the app, so `launch` opens it with `[::1]` in its place and prints
the line `dsh answers on this Mac at http://[::1]:<port>/?token=...`.

A custom `command` for a browser app must listen on `127.0.0.1:$PORT`, and
`launch` sets `HOST` and `PORT` in the container for that. The official
Open WebUI image works with `command: image`, as its start script reads
both variables:

```yaml
launch:
  container:
    clients:
      open-webui:
        image: ghcr.io/open-webui/open-webui:v0.11.4
        command: image
```

`launch` then keeps Open WebUI's secret key in the private home, so a
browser login survives the next launch.

## Projects and sessions

A session runs one client for one project in a separate virtual machine.
The project is the current folder that the session shares, and `launch`
names it after that folder, with 16 hex digits of a hash of its real path,
such as `my-project-1a2b3c4d5e6f7a8b`. When a `--mount` or a `mounts`
entry shares the current folder, the project is the folder of that share,
the longest when several hold the current folder.

Launches whose shares do not hold the current folder belong to the
`default` project. Examples are elia, and a launch with `--no-mount-cwd`
and no mount that holds the current folder. Open WebUI always belongs to
the `default` project, because it keeps one store of chats. Every other
client, dsh in the browser included, gets a project for each folder.

Sessions of different projects or different clients run side by side, and
each one holds memory, which [Limits](container-security.md#limits)
counts. Each project keeps its [private home](#the-private-home), so
`--continue`, history and the tools a client installs in its home stay with
the project.

### Joining a running session

Another launch of the same client in the project joins the running session,
instead of starting a second virtual machine. A launch from a folder inside
the session's project folder or one of its read-write shares joins too,
and the session with the longest such folder takes the launch.

Launches of elia and Open WebUI, and launches under `mount_cwd: false`,
share no folder by default. When a session of the client's `default`
project runs, one of these launches joins that session. With
`--no-mount-cwd`, or a `--mount` of another folder, a launch also stays in
the `default` project.

The joining launch prints `joining the running <client> session for <folder>`
and runs another copy of the client in the same container, in the current
folder, with the arguments that it gets after `--`. From a folder that the session
does not share, as in the `default` project, the copy starts in the
session's working folder, and `launch` names the folders that the session
shares.

While a session is still starting or is ending, a launch in its project
stops and says when to try again. A session is starting while its launch
starts the container service, prepares the image and boots the virtual
machine, and it is ending once its container stops. Another command that
holds the project, such as a `--remove-home` that waits for its answer or a
`--config-only` run, stops a launch the same way.

When `container ls` fails, `launch` cannot tell whether a session starts
or ends, so it stops. Its message says to try again once `container ls`
works, or to restart the container service, which also stops that session.

A launch from a folder that a session shares read-only, outside its project
folder, does not join that session, since a copy there could not change the
files. It starts a separate session. A launch from a folder that holds a
running session's project, such as its parent folder, or a launch of
another client in the project, starts a separate session too.

Two virtual machines then share the same files, and file locks do not reach
from one to the other, so `launch` warns and names the other session, even
one that is still starting. Launch the same client from the project folder
instead, or wait until the other session has ended.

The copies share the private home inside one virtual machine, where file
locks work, as two copies share a home on the Mac. Two virtual machines
never share a home, for the reason that
[What does not work in a container](#what-does-not-work-in-a-container)
gives.

A joining launch ignores the flags that chose the session's server and
model, such as `--model` or `--port`, with a note. A `--mount` joins when
the session already has that share, with the same folder, path and mode, so
the command that started a session joins it again.

`launch` refuses the other flags that shape a new session, which are
`--image`, `--rebuild`, `--reseed`, `--seed-instructions`, `--network`,
`--config-only` and `--provider-id`. It also refuses a dsh profile other
than the running one. The refusal says to leave out the flag to join, and
for `--mount` it lists the session's shares in the form that joins. To use
one of these flags, end the session and launch again.

### When a session ends

The session lasts until its last copy exits, and the terminal that started
it stays with it. When the first copy exits while others still run, that
terminal says the session stays open while they run, and it waits. A
Ctrl-C there asks for a second one, which ends the session.

When the session ends, or `launch` stops its container, the other copies
get SIGHUP, as from a closed terminal. A copy that exits within 5 seconds
prints
`the session ended in another terminal, so this copy of <command> stopped`.
Here `<command>` is the program that the copy runs, for example `claude` or
`bash`. A copy still running then stops with the container. Another Ctrl-C
in the first terminal stops them at once.

Closing the window of a joined copy, or stopping its launch, ends only that
copy. The copy gets SIGHUP, as a client on the Mac does, and it is killed
when it has not exited 10 seconds later.

## Sessions in the background

`--detach` starts the session of Open WebUI, a dsh web profile or a custom
agent in the background and returns once the session runs. Until then,
`launch` shows the session's output. It then prints the app's address, the
file that takes the rest of the output, and the commands that list and end
the session:

```sh
gmlx launch open-webui --detach
gmlx launch --list
gmlx launch open-webui --stop
```

A detached session has no terminal, so `launch` refuses `--detach` with
`--shell` and for a client that needs a terminal, such as pi or
claude-code. To give Claude Code a task in the background, define it as a
custom agent, as
[Coding agents in the background](launch-agents.md#coding-agents-in-the-background)
shows. An agent without a browser interface gets empty input, and a program
that reads its input gets end of file at once.

When Apple container has no Linux kernel, `--detach` asks the kernel
question in your terminal and shows the download there, and then starts the
session in the background. With no terminal, `--detach` stops and names the
command that downloads the kernel.

All output of the session, from `launch` and from the client, goes to
`~/.cache/gmlx/launch/output-<client>-<project>.log`. The next detached
launch of the project empties that file. When the client's output in it
passes 64 MiB, `launch` empties the file and starts it with a line that
counts how often it did so, and the newest output stays. The image build
and other output before the session starts do not count toward that limit.

After the container starts, `--detach` waits up to 2 minutes for it to run,
and for a browser app up to 5 minutes and 30 seconds for the app to answer.
When the time ends, or at a Ctrl-C, `launch` stops waiting and the session
goes on. `launch` then prints the path of the output file, and
`gmlx launch --list` shows whether the session runs.

When the session ends during the wait, `launch` says so and exits with the
session's exit code. A launch that fails before its session runs prints its
message in your terminal. [Exit codes](cli.md#exit-codes) lists the code of
each case.

A second `--detach` of a browser app in the project prints the address of
the running app, as any second launch does. For an agent without a browser
interface, a second `--detach` is refused, since a copy that joins the
session needs a terminal, so launch without `--detach` to join it. While
the first session still starts or ends, the rule in
[Joining a running session](#joining-a-running-session) applies.

## Listing and ending sessions

`gmlx launch --list` prints a table of the sessions that start, run or end,
and `gmlx launch <name> --list` limits it to one client or agent. Each row
shows the project folder, the state, whether `--detach` started the
session, the browser app's address and when its launch started. The
address leaves out its query, which holds dsh's login token, and a second
launch of dsh in the project opens the whole address.

Below the table, `--list` names the output file of each detached session
and the command that ends each session. A container left over from a
launch that is gone gets its `container stop` command. `gmlx launch --list`
shows every such container, also one of an agent whose home is gone and
one of an image check, whose target shows as `image check`. So does the session
of an agent that is no longer in `launch.agents`, since `gmlx launch`
refuses that name for anything but `--list`.

`--stop` reads the launch settings, so when they do not load, `--list`
prints their error first. Each session then gets its `--stop` command, to
run once the settings load, and its `container stop` command when it has a
container. [`gmlx status`](cli.md#gmlx-status) prints a line for each
session too.

`--list` and `gmlx status` wait up to 5 seconds for the container service.
When it does not answer, a session whose state `launch` cannot tell shows
as `unknown`, a container left over is not listed, and the error is
printed. `gmlx status` leaves the error out when it lists no session,
unless the service gave no answer in time, since a stopped service runs no
container.

`--stop` ends the session of the current project, whether `--detach`
started it or not. `--mount-cwd`, `--no-mount-cwd` and `--mount` choose
the project as they do for a launch, so the stop command that `--detach`
prints repeats the ones you passed. From a folder inside a session's
project folder, `--stop` ends the session that a launch from there would
join. The project keeps its private home, volumes and port.

To end a session, `--stop` sends SIGTERM to the launch that runs it, which
stops the container as closing its window does, and waits up to a minute
for that launch to exit. When the launch of a session is gone, `--stop`
stops and deletes the container itself. With no session in the project,
`--stop` names the command that ends each other session of the client.

A session that has not ended after a minute, or a container that does not
stop, gets the command that ends it. That is `container stop`, or
`kill -KILL` with the launch's process ID while the session has no
container yet. A project that another launch holds with no session
recorded yet, as in the moment after `--detach` starts its launch in the
background, makes `--stop` ask you to try again.

To find a session, `--stop` waits up to 5 seconds for the container list. A
session that a record names needs no list, since `--stop` ends it through
its launch. When the list fails, no record names a session and the service
has not stopped, `--stop` cannot tell whether a container is left over, so
it ends nothing and prints the error. [Exit codes](cli.md#exit-codes) lists
the code for each of these outcomes.

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
[joins it](#joining-a-running-session) with a shell instead, to look at
what the agent is doing. The shell starts in the current folder when a
share of the session holds it, and in the session's working folder
otherwise. Like any joined copy, the shell keeps the session open until it
exits.

For a browser app, `--shell` starts the session with a shell and no app.
`launch` prints the address of the app and the command that starts it, such
as `dsh ... --port 3100` with the project's port, so run that command in
the shell.

With `command: image`, the printed command first changes to the image's
working folder, such as `cd /app/backend && bash start.sh` for the official
Open WebUI image. For a runtime agent, the printed command starts with
`uv run`, as [Sessions and data](launch-agents.md#sessions-and-data)
describes.

The first start of dsh makes its profile from the `web` template, and a
line says to leave out `--from-default-profile web` after that. dsh then
prints its address with `127.0.0.1`, so open that address with `[::1]` in
its place. While the shell runs, a second launch prints the start command
again, and `gmlx launch <client> --shell` opens another shell in the
session.

An image with no shell at all makes `--shell` stop with a message, so add a
shell to a custom image to use it. What you install from the shell
outside the private home is gone when the session ends, as
[What the client sees](#what-the-client-sees) explains.

## Signals, cleanup and logs

A session is named `gmlx-<client>-` followed by 6 hex digits. Ctrl-C
reaches the client as it does on the Mac. When the launch that started the
session is stopped, it stops the container, and the client gets up to 10
seconds to exit. A second stop kills the container, at the latest when
those 10 seconds end.

When a stop arrives while the virtual machine starts, it waits until the
container runs, for up to a minute, and then acts. A third stop, for a
container service that no longer answers, kills `container run` and puts
the terminal settings back as they were.

While `launch` prepares the image, a Ctrl-C, SIGTERM or SIGHUP ends the
launch once its clean-up is done. `launch` ignores a second one, so that
its clean-up, which can include the stop of the image builder, finishes. A
second Ctrl-C also says to press Ctrl-C again to stop at once, and a third
one ends the clean-up too.

Closing the window of the launch that started a session stops the session.
A browser app or a custom agent that should keep running without a window
starts with [`--detach`](#sessions-in-the-background).

Ctrl-Z cannot suspend a client in the container. The client goes on
running, and the first Ctrl-Z says so. While the first terminal waits for
joined copies, Ctrl-Z there prints how to end the session, and the wait
goes on. Quit the client instead when you need the terminal.

The dsh web profiles run with no terminal in the container, so Ctrl-Z
suspends `launch` itself. The page then cannot reach dsh, and dsh cannot
reach the server, until you run `fg`.

After the last copy of the client exits, `launch` removes the container and
its session files, and prints the `container delete --force` command for a
container that is still there. The next launch of that client in that
project cleans up a session whose launch was killed. Any other launch names
the leftover container with its `container stop` command, since that
virtual machine holds memory until it stops.

The launch that started the session exits with the first copy's exit
code. [Exit codes](cli.md#exit-codes) lists the codes that `launch` and the
container return instead, a signal's code among them.

`launch` writes the session log to
`~/.cache/gmlx/launch/last-<client>-<project>.log`, and the private home
holds `.gmlx-entry.log` for errors inside the container.

## The dry run

`--config-only` in container mode writes the client's configuration into
the private home, copies each seed that has no copy there yet, and prints
the `container run` command that a session would use. It copies no seed
again, even with `--reseed`. The variables `launch` sets itself, such as
`HOME`, `TERM`, `LANG` and `IS_SANDBOX`, appear with their values. The
client's settings and your [`env`](config.md#launchcontainerenv)
entries appear by name only, because they can hold keys.

The dry run builds nothing, pulls nothing and starts no container. It
reports whether the image and volumes exist yet, and whether the server
offers [session sockets](glossary.md#session-socket). For a browser app, it
names the Mac port that the app would take, and it records no port. The
printed command cannot run by itself, because the connection to the server
exists only while `launch` supervises the session.

When no server answers and `launch` cannot start one, the dry run still
shows the image, the shares and the volumes, and says why it cannot show
the client's configuration and the command.

## Removing container data

Uninstalling gmlx leaves container data in place, and each kind is removed
separately. Apple container keeps its images, volumes and Linux kernel in
`~/Library/Application Support/com.apple.container`. `gmlx doctor` reports
the space that volumes, private homes and images take. It names the
command that deletes each image and volume that no setting or private home
uses, each stopped container that a killed launch left, and the home of an
agent that is no longer in `launch.agents`:

| Data | How to remove it |
|------|------------------|
| A private home | Run `gmlx launch <client> --remove-home` in the project folder, even after you delete its folder under `~/.local/share/gmlx/launch/<client>/projects` by hand. |
| A custom agent's home and dependency volume | Run `gmlx launch <agent> --remove-home` in the project folder, which asks about both. See [Sessions and data](launch-agents.md#sessions-and-data). |
| A custom agent's images | The `gmlx.invalid/launch-agent-*-build` and `gmlx.invalid/launch-runtime-python` images, which the Images row below covers. |
| What browser app pages left | Close the app's tabs and windows, then clear the site data that `--remove-home` names. After you delete all launch data, do so for `[::1]` ports 3100 to 3199. |
| Volumes | Run `container volume delete NAME` for each volume, which deletes its data. `--remove-home` names the volumes of the project that it keeps. |
| Images | Run `container image delete` on the `gmlx.invalid/launch-*` images and unused `image` references with their `@sha256:` entries, then `container image prune`. |
| The image builder and its cache | Run `container builder stop`, then `container builder delete`. |
| The program `launch` runs in each container | Delete `~/.local/share/gmlx/launch/runtime`. |
| Session logs, output files and session folders | Delete `~/.cache/gmlx/launch` while no session runs. |
| Apple container from Homebrew | Run `container system stop` and `brew uninstall container`, then delete `~/Library/Application Support/com.apple.container`. |
| Apple container from Apple's installer | Run `container system stop`, then `uninstall-container.sh -d`, which also deletes `~/Library/Application Support/com.apple.container`. |
