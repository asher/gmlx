# Getting started

This guide takes five steps in order: install gmlx, generate once, pick a
model that suits your Mac, set up the server and connect a client to it. Each
step ends in something usable, so you can stop wherever your needs are met.

- [What you need](#what-you-need)
- [Install](#install)
- [First generation](#first-generation)
- [Pick a model for your Mac](#pick-a-model-for-your-mac)
- [Set up the server](#set-up-the-server)
- [Talk to it over HTTP](#talk-to-it-over-http)
- [Connect a client](#connect-a-client)
- [Chat in your browser](#chat-in-your-browser)
- [Voice, login items and what comes
  next](#voice-login-items-and-what-comes-next)

## What you need

- An Apple Silicon Mac, any M-series chip.
- macOS 26.2 or newer, recommended. The Metal kernels then install as a
  prebuilt wheel. On older versions the install compiles them, which needs
  full Xcode with its Metal toolchain. The Command Line Tools alone are not
  enough. Recent Xcode fetches the toolchain with
  `xcodebuild -downloadComponent MetalToolchain`.
- Python 3.11 or newer. Installing with uv or pipx fetches one for you.
- Disk space for models, plus [Homebrew](https://brew.sh) if you want voice.

[Pick a model for your Mac](#pick-a-model-for-your-mac) has suggestions
for each machine size.

## Install

```sh
uv tool install "gmlx[all]"      # or: pip install "gmlx[all]"
brew install ffmpeg              # voice and non-wav audio only
```

uv puts the `gmlx` command on your PATH in an isolated environment and fetches
a suitable Python for it, and pipx behaves the same way. Install
[uv](https://docs.astral.sh/uv/) with `brew install uv`, and upgrade later
with `uv tool upgrade gmlx`. The pip form installs into a venv you manage
yourself, where the command exists only while that venv is active, which is
why a `command not found: gmlx` in a new terminal usually means the venv is
not active.

`[all]` turns on all optional features. The core install already serves, loads
vision models, embeds and runs the menu bar, which leaves few extras:

| Extra | Adds |
|-------|------|
| `chat` | line editing, history and rich rendering in `gmlx chat` |
| `stt` | server speech-to-text with mlx-whisper |
| `tts` | server text-to-speech with the Kokoro phoneme front end |
| `talk` | voice chat, including `stt` and `tts` |
| `assistant` | MCP tools for the built-in [assistant](assistant.md) |
| `all` | everything above |

`gmlx[chat]` is the smaller choice, omitting only voice and the assistant.
To add an extra later, run the install command again with the new extra, in
the same form you used the first time. `gmlx init` offers to install the
extra for each service you turn on, and any "not installed" message names
the command as well. ffmpeg is the only dependency that no Python installer
supplies. It decodes audio uploads and encodes mp3, flac and opus.

Tab completion needs this line in `~/.zshrc`, with bash and fish variants
available:

```sh
eval "$(gmlx completion zsh)"
```

It completes verbs, flags, your model ids and the ports of running servers.

## First generation

Download a small model into the current directory and run it:

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to .
gmlx run Qwen3-0.6B-Q4_K_M.gguf --prompt "Explain entropy in one paragraph."
```

You get a one-line load summary, the reply, and a closing line with prompt
and generation speed in tokens per second. At 0.4 GB this model is here to
verify the install, not for its quality.

The same file also runs as an interactive chat:

```sh
gmlx chat Qwen3-0.6B-Q4_K_M.gguf
```

The conversation keeps its KV cache between turns, so each turn reads only
the new message. Type `/help` inside for the commands, Esc to cancel a reply
and `/exit` to quit. [chat.md](chat.md) describes the rest.

Sampling defaults come from the model family's card, so a bare `run` or
`chat` already uses the settings the model's authors recommend. Intents such
as `@coding` and `@creative` switch to the card's other operating points on
any model, as in `gmlx run model.gguf@creative`, and `gmlx profiles` prints
the whole table.

## Pick a model for your Mac

The suffix on a GGUF name is its [quant](glossary.md), roughly the bits per
weight, so a Q4 file is smaller than the Q6 of the same model and slightly
lossier. The suggestions below are instruct models that leave memory for
the KV cache at everyday context lengths.

| Mac RAM | Suggestion | Notes |
|---------|------------|-------|
| 16 GB | Qwen3-4B, Q4_K_M, 2.5 GB | fast, capable small model |
| 32 GB | Qwen3.5-9B, Q6_K, 8 GB | has a draft head, so speculative decoding is automatic |
| 64 GB | Qwen3.6-27B, Q6_K, 23 GB | strong general model and the recommended tool-calling model |
| 96 GB and up | Qwen3.6-35B-A3B, Q6_K, or gpt-oss-120b, 63 GB | MoE models: big-model quality at small-model decode speed |

A running model needs memory for its weights, roughly the file size, and
for the KV cache, which grows with the conversation and in a long session
can reach the size of the weights. The per-token arithmetic, the families
that use less memory than it suggests, and the `--kv-bits` and
`--kv-quant-scheme` flags are all in
[performance.md](performance.md#memory-and-the-kv-cache). A MoE model larger
than RAM can still run by streaming its experts from disk, and
[streaming.md](streaming.md) has the fit calculation for that case.

Two commands save time and disk.

```sh
gmlx validate hf:unsloth/Qwen3.6-27B-GGUF          # lists every quant with a fits verdict
gmlx pull hf:unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q6_K.gguf --to ~/models
```

`validate` downloads only the header, a few megabytes, and names the codec
when a file cannot load. `--to` is needed only until a config exists, after
which a bare `pull` writes files to your model directory and registers them.
Set `HF_TOKEN` for gated repositories. An existing LM Studio model library
serves as-is, and [migrating.md](migrating.md) lists what transfers from
llama.cpp and Ollama.

## Set up the server

`gmlx init` writes the config that all the other commands read. Run with no
arguments it opens a wizard, which scans your model folders, lets you rename
ids and set a default, offers the on-disk prompt cache and the speech,
embedding and rerank services, and asks about idle unload. It previews the
file before writing it to `~/.config/gmlx/gmlx.yaml`. Each choice has a flag,
so `gmlx init --models-dir ~/models` writes the config with no questions.

```sh
gmlx init
gmlx serve            # finds the config, detaches, returns
gmlx status           # pid, uptime, url
gmlx ps               # which models are resident
gmlx logs -n 20 -f    # follow the log
gmlx stop
```

The file has a `server` block, a `models` block with an entry for each model,
and optional `profiles`, `rules` and `aliases`, with optional keys appearing
as commented hints with their defaults. [server-config.md](server-config.md)
is the reference. `serve` runs in the background so you keep your shell, and
on a macOS desktop it also starts a small [menu bar app](menubar.md) showing
what is resident.

## Talk to it over HTTP

The server implements the OpenAI, Anthropic and OpenAI Responses APIs on a
single port. Set the `model` field to the id that `init` printed, quant tag
included, such as `qwen3-0.6b-q4`. If you have forgotten them, `gmlx list`
shows the ids.

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3-0.6b-q4",
  "messages": [{"role": "user", "content": "Explain entropy in one paragraph."}]
}'
```

Add `"stream": true` for server-sent events, or append `@coding` or another
intent to the id to switch sampling for that request. From Python the
standard OpenAI client works unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
reply = client.chat.completions.create(
    model="qwen3-0.6b-q4",
    messages=[{"role": "user", "content": "Explain entropy in one paragraph."}],
)
print(reply.choices[0].message.content)
```

Tool calling, structured output, logprobs and vision messages all work over
this API. [api.md](api.md) has the endpoints and request formats.

## Connect a client

`gmlx launch` configures an external tool to use your server and starts the
server first if it is not running:

```sh
gmlx launch pi --model qwen3.6-27b
```

This adds a provider block for your server to pi's own settings, keeping the
providers already there, waits for the model to load and then runs `pi`. The
same one-liner works for Claude Code, opencode, omp, hermes, goose, aichat,
elia and Open WebUI, and [launch.md](launch.md) covers each of them.

## Chat in your browser

Open WebUI is a ChatGPT-style browser app that can use your server. It is a
separate program with its own install, but once it is there `gmlx launch
open-webui` starts your server if needed, configures the app to use it and
prints the URL. Chat works at once, and document upload and voice become
available when the server also runs embeddings and speech.
[launch.md](launch.md#open-webui) has the install and the single-user setup.

## Voice, login items and what comes next

With the `talk` extra and speech services configured, `gmlx talk` is a
hands-free voice loop: you say the wake phrase and ask, and the reply is
spoken as it streams. [talk.md](talk.md) covers it. `gmlx service install`
keeps the server and the menu bar running from login, as
[menubar.md](menubar.md) describes.

When something fails, `gmlx doctor` checks the runtime, config, model paths
and services in one pass, and [troubleshooting.md](troubleshooting.md) lists
the failures common in new setups, where each file is on disk and how to
remove gmlx completely. [README.md](README.md) indexes the rest of the docs.
