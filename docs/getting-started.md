# Getting started

This guide takes you from nothing to a running model: install, a first
generation, a model that suits your Mac, the server, and a client connected
to it. Each step ends in something you can use, so stop wherever your needs
are met.

- [What you need](#what-you-need)
- [Install](#install)
- [First generation](#first-generation)
- [Pick a model for your Mac](#pick-a-model-for-your-mac)
- [Set up the server](#set-up-the-server)
- [Talk to it over HTTP](#talk-to-it-over-http)
- [Connect a client](#connect-a-client)
- [Chat in your browser](#chat-in-your-browser)
- [Voice, login items and what comes next](#voice-login-items-and-what-comes-next)

## What you need

- An Apple Silicon Mac, any M-series chip.
- macOS 26 or newer, recommended. The Metal kernels then install as a
  prebuilt wheel. On older versions the install compiles them, which needs
  full Xcode with its Metal toolchain; the Command Line Tools alone are not
  enough, and recent Xcode fetches the toolchain with
  `xcodebuild -downloadComponent MetalToolchain`.
- Python 3.11 or newer. Installing with uv or pipx fetches one for you.
- Disk space for models, and [Homebrew](https://brew.sh) if you want voice.

A running model needs memory for two things: the weights, roughly the GGUF
file size, and the KV cache, which grows with the length of the
conversation. [Pick a model for your Mac](#pick-a-model-for-your-mac) has
suggestions per machine size.

## Install

```sh
uv tool install "gmlx[all]"      # or: pip install "gmlx[all]"
brew install ffmpeg              # voice and non-wav audio only
```

With uv the `gmlx` command lands on your PATH in every terminal, in its own
environment, with a suitable Python fetched for it.
[uv](https://docs.astral.sh/uv/) itself is `brew install uv`, and pipx
behaves the same way. Upgrade later with `uv tool upgrade gmlx`. The pip
form installs into a venv you manage, and the command then exists only while
that venv is active; a `command not found: gmlx` in a new terminal means
only that.

`[all]` turns on every optional feature. The core install already serves,
loads vision models, embeds and runs the menu bar, so the extras are few:

| Extra | Adds |
|-------|------|
| `chat` | line editing, history and rich rendering in `gmlx chat` |
| `stt` | server speech-to-text with mlx-whisper |
| `tts` | server text-to-speech with the Kokoro phoneme front end |
| `talk` | voice chat, including `stt` and `tts` |
| `assistant` | MCP tools for the built-in [assistant](assistant.md) |
| `all` | everything above |

`gmlx[chat]` is the common smaller choice, giving up only voice and the
assistant. To add an extra later, run the install command again with the new
extra, in the same form you used the first time:

```sh
uv tool install "gmlx[all]"      # or: pip install "gmlx[all]"
```

`gmlx init` offers to do this for the services you turn on, and every "not
installed" message names the command. ffmpeg is the one dependency no Python
installer supplies; it decodes audio uploads and encodes mp3, flac and opus.

Tab completion is worth one line in `~/.zshrc`: `eval "$(gmlx completion
zsh)"`, with bash and fish variants. It completes verbs, flags, your model
ids and the ports of running servers.

## First generation

Download a small model into the current directory and run it:

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to .
gmlx run Qwen3-0.6B-Q4_K_M.gguf --prompt "Explain entropy in one paragraph."
```

You see a one-line load summary, the reply, and a closing line with prompt
and generation speed in tokens per second. This model is 0.4 GB and exists
to prove the pipeline, not to impress.

Chat is the same file, interactive:

```sh
gmlx chat Qwen3-0.6B-Q4_K_M.gguf
```

The conversation keeps its KV cache between turns, so each turn reads only
the new message. Type `/help` inside for the commands, Esc cancels a reply,
and `/exit` quits. The commands are described in [chat.md](chat.md).

Sampling defaults come from the model family's card, so a bare `run` or
`chat` already uses the settings the model's authors recommend. Intents
such as `@coding` and `@creative` switch to the card's other operating
points on any model: `gmlx run model.gguf@creative`. `gmlx profiles` prints
the table.

## Pick a model for your Mac

The suffix on a GGUF name says how many bits each weight keeps. Q4 files are
smaller and slightly lossier, Q6 and Q8 bigger and closer to the original.
These suggestions are instruct models that leave room for the KV cache at
everyday context lengths.

| Mac RAM | Suggestion | Notes |
|---------|------------|-------|
| 16 GB | Qwen3-4B, Q4_K_M, 2.5 GB | fast, capable small model |
| 32 GB | Qwen3.5-9B, Q6_K, 8 GB | has a draft head, so speculative decoding is automatic |
| 64 GB | Qwen3.6-27B, Q6_K, 23 GB | strong general model and the tool-calling pick |
| 96 GB and up | Qwen3.6-35B-A3B, Q6_K, or gpt-oss-120b, 63 GB | MoE models: big-model quality at small-model decode cost |

A long session can make the cache rival the weights. The per-token
arithmetic, the families that are cheaper than it suggests, and the
`--kv-bits` lever are in
[performance.md](performance.md#memory-and-the-kv-cache). A MoE model larger
than RAM can still run by streaming its experts from disk; the fit rule is
different and [streaming.md](streaming.md) has it.

Two habits save time and disk:

```sh
gmlx validate hf:unsloth/Qwen3.6-27B-GGUF          # lists every quant with a fits verdict
gmlx pull hf:unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q6_K.gguf --to ~/models
```

`validate` reads only the header, so it costs a few megabytes rather than a
download, and it names the codec when a file cannot load. `--to` is needed
only until a config exists; after that a bare `pull` lands files in your
model directory and registers them. Set `HF_TOKEN` for gated repositories.
A model library from LM Studio serves as-is, and
[migrating.md](migrating.md) maps what carries over from llama.cpp and
Ollama.

## Set up the server

`gmlx init` writes the one config every other command reads. Run bare it
opens a wizard that scans your model folders, lets you rename ids and set a
default, offers the on-disk prompt cache and the speech, embedding and
rerank services, asks about idle unload, and previews the file before
writing it to `~/.config/gmlx/gmlx.yaml`. Every choice has a flag, so
`gmlx init --models-dir ~/models` scaffolds with no questions.

```sh
gmlx init
gmlx serve            # finds the config, detaches, returns
gmlx status           # pid, uptime, url
gmlx ps               # which models are resident
gmlx logs -n 20 -f    # follow the log
gmlx stop
```

The file has a `server` block, a `models` block with one entry per model,
and optional `profiles`, `rules` and `aliases`. Every optional key appears
as a commented hint with its default. The reference is
[server-config.md](server-config.md). `serve` runs in the background so you
keep your shell, and on a macOS desktop it raises a small
[menu bar app](menubar.md) showing what is resident.

## Talk to it over HTTP

The server speaks the OpenAI API, and the Anthropic and OpenAI Responses
APIs on the same port. The `model` field is the id `init` printed, which
carries the quant tag, such as `qwen3-0.6b-q4`; `gmlx list` shows them.

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3-0.6b-q4",
  "messages": [{"role": "user", "content": "Explain entropy in one paragraph."}]
}'
```

Add `"stream": true` for server-sent events, and `@coding` or another intent
to the id to switch sampling per request. From Python the standard OpenAI
client works unchanged:

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
this API. [api.md](api.md) has the endpoints and request shapes.

## Connect a client

`gmlx launch` points an external tool at your server and starts the server
first if it is down:

```sh
gmlx launch claude-code --model qwen3.6-27b
```

It exports the base URL and model Claude Code needs, waits for the model to
load, and runs `claude`. Your own configuration files are never touched. The
same one-liner works for opencode, pi, omp, hermes, goose, aichat, elia and
Open WebUI; [launch.md](launch.md) covers each.

## Chat in your browser

Open WebUI gives you a ChatGPT-style browser app on top of your server. It
is a separate program with its own install, and `gmlx launch open-webui`
starts your server if needed, wires the app to it, and prints the URL. Chat
works at once, and document upload and voice light up when the server also
runs embeddings and speech. [launch.md](launch.md#open-webui) has the
install and the single-user setup.

## Voice, login items and what comes next

With the `talk` extra and speech services configured, `gmlx talk` is a
hands-free voice loop: say the wake phrase, ask, and the reply is spoken as
it streams ([talk.md](talk.md)). `gmlx service install` keeps the server and
the menu bar running from login ([menubar.md](menubar.md)).

When something misbehaves, `gmlx doctor` checks the runtime, config, model
paths and services in one pass, and [troubleshooting.md](troubleshooting.md)
lists the failures new setups hit, where each file lives on disk, and how to
remove gmlx completely. The rest of the docs are indexed in
[README.md](README.md).
