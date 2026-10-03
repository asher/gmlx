# Installation

gmlx installs with Homebrew, or with uv or pip when you want to choose its
optional features.

## Requirements

- gmlx needs an Apple Silicon Mac, and it does not run on Intel Macs or
  Linux. The model you can run depends on the Mac's memory, as
  [Choosing a model](quickstart.md#choosing-a-model) shows.
- It needs macOS 26.2 or newer, because the
  [mlx-kquant](https://github.com/asher/mlx-kquant) Metal kernels are built
  for it and install prebuilt.
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

The uv tool puts the `gmlx` command on your PATH in an isolated
environment, and it downloads a suitable Python when your system has none.
Voice, speech-to-text on the server, and speech output in a format other
than WAV or PCM need ffmpeg. The server finds it on its `PATH` or in the
Homebrew folder, as
[How the services run](services.md#how-the-services-run) explains.

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

| Extra | What it adds |
|-------|--------------|
| `chat` | It adds a completion menu, a toolbar and rich markdown rendering to `gmlx chat`. |
| `stt` | It adds speech-to-text to the server, with mlx-whisper. |
| `tts` | It adds text-to-speech to the server, with the Kokoro phoneme front end. |
| `talk` | It adds the voice client, [`gmlx talk`](talk.md), and includes `stt` and `tts`. |
| `assistant` | It adds MCP tools to the built-in [assistant](assistant.md). |
| `all` | It includes every extra in this list. |

To add an extra to a uv install later, name every extra you want in one
command, because uv replaces the install with exactly what the command
lists:

```sh
uv tool install --force "gmlx[chat,talk]"
```

With pip, run `pip install "gmlx[talk]"` in the same environment. The
command keeps the extras that are already there. When you turn on speech in
[`gmlx init`](config.md#create-the-file), it offers to install the extras
that speech needs. A message that says a feature is not installed also
gives the command for your kind of install.

## Apple container

[Container mode](launch-container.md) runs a client from `gmlx launch` in
an Apple container, and it needs Apple container 1.5 or newer. Install it
with Homebrew:

```sh
brew install container
```

Apple also publishes a signed installer on its
[releases page](https://github.com/apple/container/releases). Upgrade a
Homebrew install with `brew upgrade container`. To upgrade the installer's
install, run `container system stop`, then the `update-container.sh` script
that the installer puts beside the `container` program.

`launch` runs the first `container` program on your PATH. When that program
is older than 1.5, `launch` stops with a message that names it and gives the
upgrade step for its install. When a newer program comes later on PATH, the
message says to put its folder first or to remove the older install.

On the first container launch, the container service starts and asks once
to install a Linux kernel, as
[The first launch](launch-container.md#the-first-launch) describes.
`gmlx doctor` reports the version and whether the service runs. It reports
FAIL for an old version while container mode is on.

The Homebrew, uv and pip installs include the program that container mode
runs inside the container. A git checkout of gmlx needs it built first,
with the Rust toolchain that
[CONTRIBUTING.md](https://github.com/asher/gmlx/blob/main/CONTRIBUTING.md)
names.

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

Before you upgrade, read the Changed, Removed and Security sections of each
newer release in the
[changelog](https://github.com/asher/gmlx/blob/main/CHANGELOG.md), which
list what may need a change to your config or scripts.

The upgrade also brings fixes for the media decoders that the server runs
on the files clients send. `brew upgrade gmlx` brings the Pillow and OpenCV
versions tested with each release, and `uv tool upgrade gmlx` brings the
newest versions that gmlx's requirements allow, even when gmlx itself has
no new release. `pip install -U gmlx` keeps the installed Pillow and OpenCV
while they still meet its requirements, so also run
`pip install -U pillow opencv-python`. `brew upgrade ffmpeg` updates
ffmpeg, but not the copy of FFmpeg in OpenCV.

A server that is running during an upgrade keeps the old code until you run
`gmlx restart`. A server installed as a login item with `--headless` is
restarted by launchd instead, and `gmlx restart` prints the `launchctl`
command that does it.

## Removing gmlx

To remove gmlx completely:

1. If you used container mode, remove its images, volumes and private
   homes as
   [Removing container data](launch-container.md#removing-container-data)
   describes. Do this first, because those steps run gmlx commands.
2. If you installed the login item, run `gmlx service uninstall`.
3. Uninstall the package with the tool that installed it, which is
   `brew uninstall gmlx`, `uv tool uninstall gmlx`, or
   `pip uninstall gmlx mlx-kquant` in its environment.
4. Delete the files that gmlx wrote, which
   [Where files are on disk](troubleshooting.md#where-files-are-on-disk)
   lists, and the models you downloaded.

Steps 2 and 3 leave your configuration, caches and models in place, so a
later install finds them again.
