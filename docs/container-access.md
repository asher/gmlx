# Container access

A client in a container sees only what you give it. This page covers each
way to give it more: shared folders, its private home, volumes, services on
the Mac, pasted files and SSH.

- [Shares](#shares)
- [The private home](#the-private-home)
- [Volumes](#volumes)
- [Forwarded ports](#forwarded-ports)
- [Pasting files and images](#pasting-files-and-images)
- [Instructions and skills](#instructions-and-skills)

## Shares

`launch` shares the current folder, read-write, at the same path, and the
client starts there. To share it read-only, launch with `--mount .:ro`.

Share more folders with `--mount PATH[:DST][:ro]` for one launch, or with
[`mounts`](config.md#launchcontainermounts) in the config:

```yaml
launch:
  container:
    mounts: [~/src/shared-lib:ro]
```

`launch` will not share your home folder, a system folder or a folder that
holds credentials or settings, such as `~/.ssh` or `~/.claude`. Its message
says why and what to do instead.
[What launch never shares](container-security.md#what-launch-never-shares)
lists these folders.

Inside the container, every file in a share belongs to root, and the client
runs as root. Programs that check who owns their files, such as Postgres,
need a [volume](#volumes) instead. On the Mac, files the client writes belong
to you.

macOS may ask once whether the container can read a folder in `~/Desktop`,
`~/Documents`, `~/Downloads` or iCloud Drive. The launch waits until you
answer.

### Git in the container

Launch from the root of a repository to use git in the container. From a
subfolder, git finds no repository.

Linked worktrees and submodules keep their git folder outside the project.
`launch` shares that folder too. When it cannot tell that the worktree is
yours, it prints a note with the fix.

## The private home

Each client gets a home of its own for each project. It keeps the client's
settings, logins, history and caches from one launch to the next, and your
Mac's `~/.claude`, `~/.pi` and other client folders are never touched.

A new home starts with an empty history. Conversations and chats from the
Mac stay on the Mac. `launch` copies your git name and email into the home,
so commits in the container carry your name.

The homes live in `~/.local/share/gmlx/launch/<client>/projects/`.
[`gmlx doctor`](cli.md#gmlx-doctor) lists them with their project folder,
size and last use, and [Removing container data](launch-container.md#removing-container-data)
shows how to remove one.

### Seeds

A seed copies a file or folder from your Mac home into the private home,
such as a settings file the client should start with:

```yaml
launch:
  container:
    clients:
      claude-code:
        seed: [~/.claude/settings.json]
```

When you change a seed on the Mac, the next launch copies it again, unless
the client changed its copy too. `--reseed` replaces every copy.

To bring along the instructions, skills and commands the client reads on the
Mac, launch once with `--seed-instructions`.
[Instructions and skills](#instructions-and-skills) lists the files for each
client.

The client can read every seeded file, so never seed a login token.
`launch` warns when a seed can hold one. A seeded settings file with
Mac-only settings, such as hooks that run Mac commands, fails in the
container, so seed a copy without them.

### SSH in the container

`ssh` in the container uses the `.ssh` folder in the private home, never
your Mac's keys. You have two choices:

- Turn on [`ssh_agent`](config.md#launchcontainerssh_agent). An SSH agent
  on the Mac signs for the client, and no key enters the container.
- Put a deploy key for the project's repository in the private home's
  `.ssh`, with mode 600. The client can read and copy it.

[Access you turn on](container-security.md#access-you-turn-on) compares the
two.

## Volumes

A volume is a separate disk for the container. Use one for a database or a
large cache: file owners and locks work as on Linux, and `npm install`
runs faster there than in a share.

```yaml
launch:
  container:
    clients:
      claude-code:
        volumes: [claude-pg:/var/lib/postgresql:8G]
```

Each entry takes the form `NAME:/path[:SIZE]`. An entry under a client gets a
separate volume for each project, so each project has its own database. An
entry directly under `launch.container` is shared by every project and
client.

The disk image on the Mac grows as the container writes, up to the size in
the entry. Keep data in a subfolder, because the volume's root holds a
`lost+found` folder. Each session lists its volumes and how much Mac disk
they use.

Deleting files in a volume does not give the space back to the Mac. To free
it, trim the volume while no session uses it. The session's volume line
gives the name:

```sh
container run --rm --cap-add CAP_SYS_ADMIN \
  --mount type=volume,source=claude-pg-9c0d1e2f,target=/v \
  docker.io/library/debian:trixie-slim fstrim -v /v
```

## Forwarded ports

[`forward`](config.md#launchcontainerforward) gives the client a service
that runs on the Mac, such as a database. Each port appears at the same
address in the container, `127.0.0.1:<port>`, also with `network: none`.

The client gets that service with your rights as a local user. Homebrew's
Postgres and Redis accept local connections without a password, so set a
password or a limited role before you forward one. Never forward a browser's
debugging port, such as 9222: that gives the client your logged-in browser.

## Pasting files and images

Drag a file onto the terminal, or copy it in Finder and paste it with Cmd-V.
The terminal types the file's Mac path, which the client in the container
cannot open. `launch` copies the file into the client's private home and
gives the client that path instead. This works for any file, such as a
screenshot, a PDF or a log, in every client.

A few pastes keep their Mac path:

- folders and links;
- files the container already sees, such as files in the project folder;
- files in folders `launch` never shares, such as `~/.ssh`;
- a path inside other text, such as a log line that names a file.

A file on the Mac's main disk appears at once and takes no extra space. A
file on another disk is copied, up to
[`paste_copy_max`](config.md#launchcontainerpaste_copy_max), which is 1 GiB
by default.

To paste an image from the clipboard, use the client's own image paste key,
which is Ctrl-V in Claude Code, opencode, pi, omp and hermes. Each press lets
the client read one image from the Mac clipboard. Ctrl-V reads only images,
so paste a file copied in Finder with Cmd-V.

## Instructions and skills

`--seed-instructions` copies these files and folders from your Mac home into
the private home, when they exist. It never copies login tokens or settings
files.

| Client | Files and folders |
|--------|-------------------|
| `claude-code` | `~/.claude/CLAUDE.md`, and `rules`, `skills`, `commands`, `agents` and `output-styles` in `~/.claude` |
| `opencode` | `~/.config/opencode/AGENTS.md`, its `skill(s)`, `command(s)`, `agent(s)` and `mode(s)` folders, `~/.claude/CLAUDE.md`, `~/.claude/skills`, `~/.agents/skills` |
| `pi` | `AGENTS.md`, `CLAUDE.md`, `SYSTEM.md`, `APPEND_SYSTEM.md`, `skills` and `prompts` in `~/.pi/agent`, and `~/.agents/skills` |
| `omp` | `AGENTS.md`, `SYSTEM.md`, `RULES.md` and its folders in `~/.omp/agent`, and `AGENTS.md` and its folders in `~/.agents` |
| `hermes` | `~/.hermes/SOUL.md` and `~/.hermes/skills` |
| `goose` | `.goosehints`, `AGENTS.md`, `skills`, `agents` and `recipes` in `~/.config/goose` and `~/.agents`, and `~/.claude/skills` and `~/.claude/agents` |
| `dsh` | `~/.dsh/AGENTS.md`, `~/.dsh/skills` and `~/.agents/skills` |
| `aichat` | `~/.config/aichat/roles` and `~/.config/aichat/macros` |

elia and open-webui read no such files. A custom agent lists the files it
needs in [`seed`](config.md#launchcontainerclientsseed).
