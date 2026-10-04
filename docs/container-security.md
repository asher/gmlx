# Container security

A container keeps a client away from the rest of your Mac, but not from the
folders you share with it. This page covers what a client in
[container mode](launch-container.md) can still reach, what gmlx guards for
you, and how to run a client on code you do not trust.

- [What the container protects](#what-the-container-protects)
- [A session for code you do not trust](#a-session-for-code-you-do-not-trust)
- [Shares that lead back to the Mac](#shares-that-lead-back-to-the-mac)
- [The share history](#the-share-history)
- [Your terminal](#your-terminal)
- [Browser app pages](#browser-app-pages)
- [Access you turn on](#access-you-turn-on)
- [What the client reaches on the server](#what-the-client-reaches-on-the-server)
- [Custom agents](#custom-agents)
- [Limits](#limits)
- [What launch never shares](#what-launch-never-shares)

## What the container protects

The client cannot reach your home folder, keychain, SSH keys, other
projects, or the gmlx server's admin routes. It reaches the model through a
connection that serves inference only and never holds the server's key.

The client can still change anything in a read-write share, and some of
those files run on the Mac later: git hooks, `package.json` scripts, a
project's `.claude/settings.json`. The container cannot stop that. Read what
the client changed before you run the project on the Mac.

## A session for code you do not trust

These steps run a coding agent on a repository you do not trust, and bring
out only the changes you have read:

1. Turn off your terminal's clipboard write, as
   [Your terminal](#your-terminal) shows.

2. Clone the repository into a separate folder:

   ```sh
   git clone https://example.com/them/project ~/review/project
   cd ~/review/project
   ```

3. Install its dependencies from a shell in the container, while it still
   has the network:

   ```sh
   gmlx launch claude-code --container --shell -- -c "npm ci"
   ```

4. Start the agent without the network:

   ```sh
   gmlx launch claude-code --container --network none
   ```

5. Take the changes out as a patch, read it, and apply it to your own clone:

   ```sh
   gmlx launch claude-code --container --shell -- -c \
     "git add -A && git diff --cached --binary > changes.patch"
   git -C ~/src/project apply ~/review/project/changes.patch
   ```

6. Remove the private home with `gmlx launch claude-code --remove-home` in
   the folder, then delete the folder.

Do not run git or the project's scripts in `~/review/project` on the Mac,
because the agent may have changed the files they run. The patch is plain
text you can read in full. Give the agent no
[assistants](#what-the-client-reaches-on-the-server), because their tools
run on the Mac.

## Shares that lead back to the Mac

Read these before you use them on the Mac, after a client has worked in a
read-write share:

- Files that tools run: `.git/hooks`, `.git/config`, `.envrc`, the scripts
  in `package.json`, and Claude Code's `.claude/settings.json`,
  `.claude/settings.local.json` and `.mcp.json`.
- For Python: `pyproject.toml`, `uv.lock`, `uv.toml`, `.python-version` and
  build code such as `setup.py`. uv runs them when you sync the project.
- Files git ignores, such as `.venv` and `__pycache__`. `git diff` does not
  show them.
- A `gmlx.yaml` in the share. It takes effect only when you pass it with
  `--config`, but it can then change the server's address or key, or add a
  tool server command.

The private home and the client's volumes carry changes into the next
session of the same project, such as a changed `.bashrc` or `.gitconfig`.
After a client you do not trust has run, remove its home with
`--remove-home`.

gmlx guards the rest, and its messages name the folder and the fix:

- `launch` refuses a read-write share that holds gmlx's Python environment
  or the `git` and `ssh-add` it runs.
- `launch` warns when a read-write share holds the server's config file, a
  model folder, or a folder on your `PATH` or `PYTHONPATH`. Share those
  folders read-only with `:ro`, or move the file out.
- The server never runs ffmpeg or a tool server from a folder a client can
  write. That covers private homes and every folder a session shares
  read-write, now or earlier, as [The share history](#the-share-history)
  explains.
- A client's [`build`](config.md#launchcontainerclientsbuild) folder must be
  outside every read-write share, because its code runs at the next image
  build with internet access.

gmlx checks programs by path, so it cannot see a hard link. Share a conda or
pnpm project read-only when its files are hard-linked to programs the Mac
runs.

## The share history

A client can leave changed files and links in a shared folder after the
session ends. So gmlx remembers every folder a session shared read-write,
and keeps refusing to run tool servers and ffmpeg from it.

You notice this when a tool server that used to work stops, sometimes weeks
later. The message names the folder and the way back:

```text
gmlx will not run ~/src/tools/.venv/bin/search-server, because it lies in ~/src/tools, a folder that a container session shared read-write.
When you trust the files in ~/src/tools again, remove it from the share history with gmlx launch --forget-share ~/src/tools.
```

Read what the clients changed in the folder first, as
[Shares that lead back to the Mac](#shares-that-lead-back-to-the-mac)
lists. Then run the command:

```sh
gmlx launch --forget-share ~/src/tools
```

The history is in `~/.local/share/gmlx/launch/shared.json`. A later session
that shares the folder read-write adds it again.

## Your terminal

The client writes straight to your terminal, as a program does over ssh.
Some terminals let a program replace your clipboard with an escape sequence
(OSC 52), with no prompt. A client could then put a command on the clipboard
that you later paste into a Mac shell. Turn this off before you run a client
you do not trust:

| Terminal | Setting that stops the write |
|----------|------------------------------|
| Ghostty | `clipboard-write = deny` |
| kitty | Remove `write-clipboard` from `clipboard_control`. |
| Alacritty | `terminal.osc52 = "Disabled"` |
| WezTerm | It has none, so run container sessions in another terminal. |

Terminal.app ignores these writes, and iTerm2 allows them only when you turn
on clipboard access in its settings. Leave kitty's remote control off.

A client can also draw text that looks like your shell prompt. Before you
type a password in a terminal that ran a session, make sure the session has
ended.

## Browser app pages

A [browser app](launch-container.md#browser-apps) page comes from the
container but runs in your Mac browser. It opens at `http://[::1]:<port>`,
which the browser treats as a separate site from `localhost` and
`127.0.0.1`. So the page gets no cookies of the apps you run there, such as
Open WebUI or dsh on the Mac.

The page can still send requests to services on the Mac, and reach the
internet through the browser, even with `network: none`. For a client you do
not trust:

1. Set [`open_browser: false`](config.md#launchcontaineropen_browser) and
   open its apps in a separate browser profile.
2. When the session ends, close the app's tabs and windows.
3. Clear the site data for its `http://[::1]:<port>` address.

Container apps of different projects share the `[::1]` site. A page of one
project can send requests to another project's app with that app's cookies,
but it cannot read the answers.

## Access you turn on

Each of these gives the client more than the default:

- [`ssh_agent`](config.md#launchcontainerssh_agent) lets the client sign
  with every key loaded in your SSH agent, so it can push to any repository
  those keys reach. Load only the keys the task needs. A deploy key in the
  [private home](container-access.md#ssh-in-the-container) reaches one
  repository, but the client can copy it.
- A [forwarded port](container-access.md#forwarded-ports) gives the client
  a Mac service with your rights as a local user.
- [`env`](config.md#launchcontainerenv) passes variables into the
  container, where any program can read them.
- The default network reaches the internet, your local network and Mac
  services that listen on all addresses. Set
  [`network: none`](config.md#launchcontainernetwork) when the client does
  not need it.

Two kinds of access work only when you act. The client reads an image from
the Mac clipboard only after you press its paste key, one image per press,
and never reads clipboard text. A
[pasted file](container-access.md#pasting-files-and-images) is copied into
the private home, where the client can keep it.

## What the client reaches on the server

The client talks to the gmlx server through a session socket that serves
only the inference routes: models, chat, completions, responses, messages,
embeddings, rerank, speech, transcription, images, `systemone` and
`/health`. Every other route answers 404, so the client cannot unload models
or change the server's settings. Its configuration holds the placeholder key
`gmlx-container-session`, never the server's key.

The client sees only the [served assistants](config.md#served-assistants)
you list in its [`assistants`](config.md#launchcontainerclientsassistants)
key. An assistant's tools run on the Mac with your rights, and the messages
the client sends decide which tools it calls. That is fine for a chat app,
where you write the messages. A coding agent sends text from files and web
pages it reads, and that text can carry instructions. Give coding agents no
assistants, and give them a
[tool server in the container](container-recipes.md#tool-servers) instead.

The server refuses file paths and URLs in the media of a session's requests.
It decodes images, audio and video on the Mac, with Pillow, miniaudio and
ffmpeg, so keep them up to date, as [Upgrading](installation.md#upgrading)
describes.

Another server on the Mac that listens on all addresses with no key is open
to the container too. `launch` and `gmlx doctor` warn about it. Set
[`server.api_key`](config.md#serverapi_key) on it.

## Custom agents

A [custom agent](launch-agents.md) runs inside the same boundary as any
client. A few points add to the rest of this page:

- A runtime agent installs its dependencies when the session starts, and
  their code runs inside the container.
- The agent and every package it installs can read the variables in `env`,
  such as a search API key. Pass only the keys it needs.
- A read-only `source` protects only the source folder. The dependency
  volume is writable and runs at the next launch. Remove it with
  `--remove-home` after an agent you do not trust has run.

## Limits

- Memory: the container's memory counts against the model server's until
  the container stops. `launch` notes when a container would take more than
  a quarter of the Mac's memory. A client's requests can also make the
  server hold up to about 11 GiB beside the model, or about 6 GiB without
  `server.stt`.
- Disk: the private home and the shares have no size limit, so a client can
  fill the Mac's disk. Watch the free space while a client works unattended.
- Open files: each file the container reads in a share holds a file handle
  on the Mac. Share narrow folders, not a home folder full of projects.
  [`gmlx doctor`](cli.md#gmlx-doctor) warns when the Mac runs low.

## What launch never shares

`launch` refuses to share these folders as the project folder. A `--mount`
of most of them only warns, but a mount of your home folder, `$TMPDIR` or
gmlx's own folders is refused too.

| Kind | Folders |
|------|---------|
| Home and system | your home folder and any folder that holds it, `/`, `/Users`, `/Volumes`, `/private`, `/tmp`, `/var`, `/opt`, `/usr`, `/Library`, `/System`, `/Applications` |
| Temporary files | `$TMPDIR` and `/private/var/folders`. A scratch project under `/private/tmp` is fine. |
| Credentials | `~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.azure`, `~/.config/gcloud`, `~/.kube`, `~/.docker`, `~/.password-store`, `~/Library/Keychains` |
| More credentials | `~/.netrc`, `~/.config/gh`, `~/.npmrc`, `~/.git-credentials`, `~/.config/gmlx`, `~/.cache/huggingface`, `~/.codex` |
| Programs the Mac runs | `~/Library/LaunchAgents`, `~/.config/git`, `~/.local/bin`, `~/bin`, `~/Library/Application Support`, `~/.cargo`, `/opt/homebrew`, `/usr/local` |
| More programs | `~/.local/share/claude` |
| Shell and editor settings | `~/.config/fish`, `~/.vim`, `~/.config/vim`, `~/.config/nvim`, `~/.local/share/nvim`, `~/.config/tmux`, `~/.emacs.d`, `~/.config/emacs` |
| Client settings | `~/.claude`, `~/.pi`, `~/.omp`, `~/.hermes`, `~/.open-webui`, `~/.dsh`, `~/.config/goose`, `~/.config/elia`, `~/.config/aichat` |
| More client settings | `~/.config/opencode`, `~/.local/share/opencode`, `~/.cache/opencode`, `~/.opencode`, `~/Library/Application Support/aichat` |
| gmlx data | `~/.cache/gmlx` and `~/.local/share/gmlx` |

Apart from the home and system folders, the rule also covers folders
inside these and folders that hold them. It follows variables such as
`CLAUDE_CONFIG_DIR` or `XDG_CONFIG_HOME` that move a folder, and covers a
dotfiles folder that links such as `~/.zshrc` lead into. The message names
the reason and offers a read-only share when that is safe.
