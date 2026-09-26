# Installation

gmlx runs on Apple Silicon Macs with macOS 26.2 or newer. It installs with
Homebrew, or with uv or pip when you want to choose its optional features.
Intel Macs and Linux are not supported.

## Requirements

- gmlx needs an Apple Silicon Mac. The model you can run depends on its
  memory, as [Choosing a model](quickstart.md#choosing-a-model) shows.
- It needs macOS 26.2 or newer, because the Metal kernels of
  [mlx-kquant](https://github.com/asher/mlx-kquant) are built for it and
  install prebuilt.
- The models need several GB of disk space each.

## Homebrew

Install with Homebrew:

```sh
brew install asher/gmlx/gmlx
```

The formula installs gmlx with every optional feature, and ffmpeg for
voice and audio. It installs the exact dependency versions that were
tested with its release, in an environment of its own, so there is no
Python setup to do.

## uv

To choose the optional features yourself, install with
[uv](https://docs.astral.sh/uv/):

```sh
brew install uv
uv tool install "gmlx[all]"
brew install ffmpeg
```

uv puts the `gmlx` command on your PATH in an isolated environment, and it
downloads a suitable Python when your system has none. ffmpeg is needed
only for voice and for audio that is not WAV.

## pip

To use gmlx from a Python environment that you manage, for example to call
its [Python API](python.md), install it with pip in that environment. It
needs Python 3.11 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install "gmlx[all]"
```

The `gmlx` command then exists only while the environment is active. If a
new terminal cannot find `gmlx`, activate the environment again, as
[Troubleshooting](troubleshooting.md#gmlx-command-not-found-in-a-new-terminal)
describes.

## Optional features

The core install serves models, loads vision models, computes embeddings
and runs the menu bar app. Each extra adds one of the other features, and
the Homebrew formula includes all of them.

| Extra | Adds |
|-------|------|
| `chat` | Line editing, history and rich rendering in `gmlx chat`. |
| `stt` | Speech-to-text on the server, with mlx-whisper. |
| `tts` | Text-to-speech on the server, with the Kokoro phoneme front end. |
| `talk` | The voice client, [`gmlx talk`](talk.md), with `stt` and `tts`. |
| `assistant` | MCP tools for the built-in [assistant](assistant.md). |
| `all` | Every extra in this list. |

To add an extra to a uv install later, name every extra you want in one
command, because uv replaces the install with exactly what the command
lists:

```sh
uv tool install --force "gmlx[chat,talk]"
```

With pip, run `pip install "gmlx[talk]"` in the same environment, which
keeps the extras already there. When you turn on speech in
[`gmlx init`](config.md#create-the-file), it offers to install the extras
that speech needs. A message that says a feature is not installed also
gives the command for your kind of install.

## Tab completion

gmlx completes its commands, flags, your model ids and the ports of running
servers. For zsh, add this line to `~/.zshrc`:

```sh
eval "$(gmlx completion zsh)"
```

For bash, add `eval "$(gmlx completion bash)"` to `~/.bashrc`. For fish,
add `gmlx completion fish | source` to `~/.config/fish/config.fish`.

## Upgrading

Upgrade with the tool that installed gmlx:

| Install | Upgrade command |
|---------|-----------------|
| Homebrew | `brew upgrade gmlx` |
| uv | `uv tool upgrade gmlx` |
| pip | `pip install -U gmlx` in its environment |

A server that is running during an upgrade keeps the old code until you run
`gmlx restart`. A server installed as a login item with `--headless` is
restarted by launchd instead, and `gmlx restart` prints the `launchctl`
command that does it.

## Removing gmlx

To remove gmlx completely:

1. If you installed the login item, run `gmlx service uninstall`.
2. Uninstall the package with the tool that installed it, which is
   `brew uninstall gmlx`, `uv tool uninstall gmlx`, or
   `pip uninstall gmlx mlx-kquant` in its environment.
3. Delete the files that gmlx wrote, which
   [Where files are on disk](troubleshooting.md#where-files-are-on-disk)
   lists, and the models you downloaded.

Steps 1 and 2 leave your configuration, caches and models in place, so a
later install finds them again.
