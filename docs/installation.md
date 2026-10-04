# Installation

gmlx needs an Apple Silicon Mac with macOS 26.2 or newer. It does not run on
Intel Macs or Linux. Install it with Homebrew:

```sh
brew install asher/gmlx/gmlx
```

The formula installs gmlx with every optional feature, plus ffmpeg for voice
and audio, in an environment of its own. There is no Python setup to do.

Each model takes several GB of disk space, and the model you can run depends
on your Mac's memory, as [Choosing a model](quickstart.md#choosing-a-model)
shows. Next, follow the [Quickstart](quickstart.md).

- [uv](#uv)
- [pip](#pip)
- [Optional features](#optional-features)
- [Apple container](#apple-container)
- [Tab completion](#tab-completion)
- [Upgrading](#upgrading)
- [Removing gmlx](#removing-gmlx)

## uv

To install without Homebrew, use [uv](https://docs.astral.sh/uv/). This
installs every optional feature, and [Optional features](#optional-features)
shows how to choose them:

```sh
brew install uv
uv tool install "gmlx[all]"
brew install ffmpeg
```

uv puts the `gmlx` command on your PATH and downloads a Python when your
system has none. Voice, speech-to-text on the server, and speech output in
a format other than WAV or PCM need ffmpeg.

## pip

To call the [Python API](python.md) from an environment you manage, install
with pip. It needs Python 3.11 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install "gmlx[all]"
```

The `gmlx` command exists only while the environment is active. If a new
terminal cannot find it, see
[Troubleshooting](troubleshooting.md#gmlx-command-not-found-in-a-new-terminal).

## Optional features

The core install serves models, loads vision models, computes embeddings and
runs the menu bar app. The extras add the rest:

| Extra | Adds |
|-------|------|
| `chat` | Completion menu, toolbar and rich markdown rendering in `gmlx chat` |
| `stt` | Speech-to-text on the server, with mlx-whisper |
| `tts` | Text-to-speech on the server, with mlx-audio, for Kokoro, Qwen3-TTS and other speech models |
| `talk` | The voice client [`gmlx talk`](talk.md), with `stt` and `tts` |
| `assistant` | MCP tools for the built-in [assistant](assistant.md) |
| `all` | Every extra above |

To add or remove extras with uv, install again and list every extra you
want, because uv keeps only the ones you name:

```sh
uv tool install --force "gmlx[chat,talk]"
```

With pip, `pip install "gmlx[talk]"` keeps the extras you already have. When
a feature is not installed, gmlx prints the command that adds it.

## Apple container

[Container mode](launch-container.md) needs Apple container 1.5 or newer:

```sh
brew install container
```

Apple also publishes a signed installer on its
[releases page](https://github.com/apple/container/releases). Make your
first container launch from a terminal, because it starts the container
service and asks to download its Linux kernel. `gmlx doctor` reports the
installed version.

## Tab completion

gmlx completes commands, flags, model ids and the ports of running servers.
For zsh, add this line to `~/.zshrc`:

```sh
eval "$(gmlx completion zsh)"
```

For bash, add `eval "$(gmlx completion bash)"` to `~/.bashrc`. For fish, add
`gmlx completion fish | source` to `~/.config/fish/config.fish`.

## Upgrading

| Install | Upgrade command |
|---------|-----------------|
| Homebrew | `brew upgrade gmlx` |
| uv | `uv tool upgrade gmlx` |
| pip | `pip install -U gmlx pillow opencv-python` in its environment |

The pip command also names pillow and opencv-python, the image and video
decoders that the server runs on files clients send, because
`pip install -U gmlx` keeps their installed versions.

Before you upgrade, read the Changed, Removed and Security sections of each
newer release in the
[changelog](https://github.com/asher/gmlx/blob/main/CHANGELOG.md). They list
what may need a change to your config or scripts.

A running server keeps the old version until you run `gmlx restart`. Update
ffmpeg with `brew upgrade ffmpeg`.

## Removing gmlx

1. If you used container mode, remove its images, volumes and private homes
   as [Removing container data](launch-container.md#removing-container-data)
   describes. Do this first, because those steps run gmlx commands.
2. If you installed the login item, run `gmlx service uninstall`.
3. Uninstall the package: `brew uninstall gmlx`, `uv tool uninstall gmlx`,
   or `pip uninstall gmlx mlx-kquant` in its environment.
4. Delete the files that gmlx wrote, listed in
   [Where files are on disk](troubleshooting.md#where-files-are-on-disk),
   and the models you downloaded.

Steps 2 and 3 leave your configuration, caches and models in place, so a
later install finds them again.
