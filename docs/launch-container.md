# Container mode

Container mode runs a client from [`gmlx launch`](launch.md) inside an Apple
container, a small Linux virtual machine that sees only the folders you
share with it. The client can run shell commands and edit files without
reaching the rest of your Mac. The model server stays on the Mac, where it
has the GPU.

- [The first launch](#the-first-launch)
- [Turning on container mode](#turning-on-container-mode)
- [What the client sees](#what-the-client-sees)
- [What does not work in a container](#what-does-not-work-in-a-container)
- [Browser apps](#browser-apps)
- [Clients in a container](#clients-in-a-container)
- [The dry run](#the-dry-run)
- [Removing container data](#removing-container-data)

[Container access](container-access.md) covers shares, volumes and pasted
files, and [Container sessions](container-sessions.md) covers second
terminals, background apps and stopping sessions.

## The first launch

Install Apple container 1.5 or newer, as
[Installation](installation.md#apple-container) shows. Then run the client
with `--container` from the project folder it should work in:

```sh
cd ~/src/my-project
gmlx launch claude-code --container --model qwen3.8-27b-ud-q6
```

The first launch takes a few minutes:

1. Apple container needs a Linux kernel. `launch` asks before it downloads
   it, about 700 MB, once.
2. `launch` builds the client's image. This takes a few minutes, and longer
   for hermes, elia and open-webui.
3. The client starts and takes over the terminal. Wait for its prompt before
   you type, because the container drops input that arrives before it starts.

Later launches start in about a second. If a step fails, the message says
what to do, and [Troubleshooting](troubleshooting.md#container-mode) covers
the common causes.

The image holds the client, git, ripgrep, curl and ssh. Add the tools your
project needs, such as `python3`, `make` or `cargo`, with
[`packages`](container-images.md#extra-packages) before you let an agent run
your tests.

## Turning on container mode

To run clients in a container by default, set
[`launch.container.enabled`](config.md#launchcontainerenabled) in your
config file:

```yaml
launch:
  container:
    enabled: true
```

`--no-container` then runs a client on the Mac for one launch. Flags that
only make sense in a container, such as `--mount` or `--shell`, turn on
container mode by themselves.

A container gets 4 CPUs and 4G of memory unless
[`cpus`](config.md#launchcontainercpus) and
[`memory`](config.md#launchcontainermemory) say otherwise. It can reach the
internet and your local network. Set
[`network: none`](config.md#launchcontainernetwork), or pass
`--network none`, to leave it only the gmlx server and the
[forwarded ports](container-access.md#forwarded-ports).

## What the client sees

Inside the container, the client sees:

- the project folder, read-write, at the same path as on the Mac;
- its [private home](container-access.md#the-private-home), which keeps its settings and
  history;
- the gmlx server, for inference only;
- any extra [shares](container-access.md#shares), [volumes](container-access.md#volumes) and
  [forwarded ports](container-access.md#forwarded-ports) you set up;
- files you [paste or drag](container-access.md#pasting-files-and-images) into the terminal.

Your home folder, keychain, SSH keys and other projects stay out of reach.

What the client writes in a share, its private home or a volume is kept.
Anything else it writes, such as a package it installs with `apt-get`, is
gone when the session ends. To keep a tool, add it to the
[image](container-images.md).

The container limits what the client can reach, not what it writes in the
project. [Container security](container-security.md) covers the files in a
project that the Mac runs later, and what to do with code you do not trust.

## What does not work in a container

Most of these call into the Mac:

- Opening a browser or a URL. The client prints the link instead.
- Notifications, sounds, and hooks that run Mac commands such as
  `osascript`.
- Credentials in the Keychain, such as `gh` logins. Pass a token with
  [`env`](config.md#launchcontainerenv) instead.
- Your Mac's `~/.ssh` keys. See [SSH in the container](container-access.md#ssh-in-the-container).
- The Mac's GPU. Code you run in the container uses the CPU. The model still
  runs on the GPU, on the gmlx server.
- Dependencies built on the Mac, such as `.venv` or `node_modules` with
  native modules. The container needs Linux builds, and an install there
  replaces the Mac builds. Install again on the Mac before you run the
  project there.
- File locks between the container and the Mac. Run a database or a build
  that relies on locks from one side at a time.
- Internet access while a VPN sends all Mac traffic through its tunnel. The
  client still reaches the gmlx server. See
  [The image build cannot reach the network](troubleshooting.md#the-image-build-cannot-reach-the-network).

## Browser apps

Open WebUI, the dsh web profiles and
[custom agents with a browser interface](launch-agents.md#a-browser-interface)
open in your Mac browser at `http://[::1]:<port>/`. `launch` opens the page
when the app is ready, which can take half a minute. Set
[`open_browser: false`](config.md#launchcontaineropen_browser) to get the
address printed instead.

Use the `[::1]` address. `localhost` and `127.0.0.1` do not reach the app.
This keeps container apps away from the cookies of apps on `localhost`, as
[Browser app pages](container-security.md#browser-app-pages) explains.

Each project gets its own port from 3100 to 3199 and keeps it, so your
logins and browser data stay with the project. Open WebUI uses one port for
all projects, because it keeps one store of chats.

To run the official Open WebUI image instead of the one gmlx builds, let the
image's own start command run:

```yaml
launch:
  container:
    clients:
      open-webui:
        image: ghcr.io/open-webui/open-webui:v0.11.4
        command: image
```

Your own `command` for a browser app must listen on `127.0.0.1:$PORT`.
`launch` sets `HOST` and `PORT` in the container.

## Clients in a container

Each client gets the same configuration as on the Mac, written into its
private home. Three clients change further:

| Client | In a container |
|--------|----------------|
| `claude-code` | `-- --dangerously-skip-permissions` works, and the auto-updater is off. |
| `open-webui` | Opens as a [browser app](#browser-apps). Its chats are separate from the app on the Mac. |
| `dsh` | The `gmlx` and `web` profiles open as browser apps. `headless` and your own profiles run in the terminal. |

Environment variables you export on the Mac do not reach the container. Add
the ones a client needs, such as `CLAUDE_CODE_MAX_CONTEXT_TOKENS` or Open
WebUI's `WEBUI_AUTH`, to [`env`](config.md#launchcontainerenv).

[Container recipes](container-recipes.md) adds pi packages, dsh plugins,
tool servers and web search. To run a program of your own in a container,
see [Custom agents](launch-agents.md).

## The dry run

`--config-only` shows what a launch would do without starting it:

```sh
gmlx launch claude-code --container --config-only
```

It writes the client's configuration into the private home and prints the
`container run` command, the image, the shares, the volumes and, for a
browser app, the port. It builds nothing and starts no container. Values
that can hold keys appear by name only.

## Removing container data

Uninstalling gmlx leaves container data in place. `gmlx doctor` shows how
much space each kind takes and names the command that removes what nothing
uses any more.

| Data | How to remove it |
|------|------------------|
| A project's private home and volumes | Run `gmlx launch <client> --remove-home` in the project folder. It asks first. |
| A custom agent's home | Run `gmlx launch <agent> --remove-home` in the project folder. |
| Other volumes | Run `container volume delete NAME`, which deletes the data. |
| Images | Run `container image delete` on the `gmlx.invalid/launch-*` images, then `container image prune`. |
| The image builder | Run `container builder stop`, then `container builder delete`. |
| Logs and session files | Delete `~/.cache/gmlx/launch` while no session runs. |
| Apple container | Run `container system stop` and `brew uninstall container`, then delete `~/Library/Application Support/com.apple.container`. |

After you remove a browser app's home, clear the site data for its
`[::1]` address in your browser. `--remove-home` names the address.
