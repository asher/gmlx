# Getting started

This guide takes you from nothing to a running model: install, download a GGUF (the
single-file model format the open-model ecosystem publishes on Hugging Face), chat
with it, stand up the server, and connect a client. Each step ends in something you
can use, so stop wherever your needs are met.

## What you need

- An Apple Silicon Mac (any M-series chip).
- Python 3.11 or newer.
- Disk space for models.
- On macOS versions before 26: the Xcode Command Line Tools
  (`xcode-select --install`), because the install compiles the Metal kernels
  from source there.
- Optional, for voice and the browser chat app: [Homebrew](https://brew.sh),
  used below to install `ffmpeg` and `pipx`.

gmlx installs like a Python developer tool - a terminal, `git`, and `pip` are
assumed throughout.

A running model costs memory in two parts: the weights, roughly the GGUF file size,
and the KV cache, which grows with context length.
[Pick a model for your Mac](#pick-a-model-for-your-mac) has per-machine suggestions
and the arithmetic for estimating the cache.

## Install

```sh
mkdir ~/gmlx && cd ~/gmlx
python3 -m venv .venv && source .venv/bin/activate
pip install "gmlx[chat]"
```

That is the whole install. One venv habit to know: the `gmlx` command exists
only while the venv is active. In every new terminal, run
`source ~/gmlx/.venv/bin/activate` first - if you ever see
`command not found: gmlx`, that is all that happened.

The `mlx-kquant` dependency (the Metal kernels) arrives
as a prebuilt wheel from PyPI on macOS 26 and newer; on older macOS versions pip
builds it from source, which needs the Xcode Command Line Tools
(`xcode-select --install`) and takes a few minutes.

The `[chat]` extra upgrades the chat REPL's line editor. Other extras, all optional:

| Extra | Adds | Needed for |
|-------|------|------------|
| `chat` | prompt_toolkit line editor, rich | nicer `gmlx chat` |
| `stt` | mlx-whisper (plus ffmpeg on PATH) | server speech-to-text |
| `tts` | misaki[en] phoneme front-end (plus ffmpeg for non-wav) | server text-to-speech |
| `talk` | client audio + wake word, includes stt and tts | `gmlx talk` voice chat |
| `assistant` | the MCP SDK | tools for the built-in [assistant](assistant.md) (talk, `chat --assistant`, served assistants) |
| `all` | chat + talk + assistant in one | every optional feature |

The `tts`, `talk`, and `all` extras need Python 3.11-3.13 (on 3.14 the Kokoro
voice is unavailable until spacy ships wheels - the `qwen3-tts` model
still works). Everything else runs on any supported Python.

`vlm` and `embeddings` exist as empty back-compat extras; multimodal loading and the
embeddings endpoint are part of the core install. mlx-audio itself already arrives
with the core install; the `tts` extra pins it and adds the
misaki[en] grapheme-to-phoneme front-end the default Kokoro voice needs.

Tab completion is worth the one line: add `eval "$(gmlx completion zsh)"` to
`~/.zshrc` (there are `bash` and `fish` variants). It completes verbs, flags, your
config's model ids, and the host and port of any running server.

The installed command is `gmlx`.

## First generation in two minutes

Download a small model into the current directory and run it:

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to .
gmlx run Qwen3-0.6B-Q4_K_M.gguf --prompt "Explain entropy in one paragraph."
```

You should see a short load report (architecture, codecs, load time), the model's
reply, and a closing line with prompt and generation speeds in tokens per second.
This model is only 0.4 GB; it exists to prove the pipeline, not to impress.

Chat is the same file, interactive:

```sh
gmlx chat Qwen3-0.6B-Q4_K_M.gguf
```

Multi-turn, with the KV cache kept between turns. Type `/help` inside the REPL for
the runtime commands (sampling changes, `/undo`, `/retry`, sessions, staging a file
or shell output into the next message). Esc cancels a reply mid-stream;
`/exit` (or Ctrl-D) quits.

Sampling defaults come from the model family's card, so a bare `run` or `chat` is
already using the settings the model's authors recommend. `gmlx profiles` prints
the table. Two words you will meet for adjusting them: built-in *intents*
(`@coding`, `@creative`, ...) work on any model with no config, while *profiles*
are your own named setting bundles, defined later in a server config's
`profiles:` block; both are addressed the same way, `model@NAME` or
`--profile NAME`.

## Pick a model for your Mac

Suggestions by machine memory, all instruct models that load end to end here.
A quick key to the quant names you will see everywhere: the Q-number is roughly
bits per weight, so Q4 files are smaller and slightly lossier, Q6/Q8 bigger and
closer to the original; when and why it matters is in
[performance.md](performance.md#choosing-a-quant-for-speed).

| Mac RAM | Suggestion | Notes |
|---------|-----------|-------|
| 16 GB | Qwen3-4B (Q4_K_M, ~2.5 GB) | fast, capable small model |
| 32 GB | Qwen3.5-9B (Q6_K, ~8 GB) | native MTP head: speculative decoding is automatic |
| 64 GB | Qwen3.6-27B (Q6_K, ~23 GB) | strong general model; also the tool-calling pick |
| 96+ GB | Qwen3.6-35B-A3B (Q6_K) or gpt-oss-120b (MXFP4, ~63 GB) | MoE models: big-model quality, small-model decode cost |

These sizes leave room for the KV cache at everyday context lengths;
[Will it fit?](#will-it-fit) below has the per-token arithmetic and the
`--kv-bits` lever for long sessions.

Two habits save time and disk:

```sh
# check a file will load BEFORE downloading gigabytes (reads only the header)
gmlx validate hf:unsloth/Qwen3.6-27B-GGUF

# then pull the variant you picked (sharded downloads resume if interrupted)
gmlx pull hf:unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q6_K.gguf --to ~/models
```

`--to` says where the file lands; it is required until a config exists (next
section), after which bare `pull` lands files in your model directory, registers
them in the config, and a running server serves them immediately.
`gmlx sync-models` reconciles in bulk after hand-moving or deleting files. These
are multi-gigabyte downloads - minutes to an hour depending on your connection -
with progress, rate, and resume built in.

`validate` accepts a repo, a folder, a pasted browser link, or an exact file. Given
a repo it lists every quant variant as a ready-to-paste ref. K-quant, legacy, and
IQ files all load; in the rare case a file uses a codec with no kernel (the
ternary TQ types, for instance), the verdict names it so you can pick another
variant. Uniform K-quant files also decode
faster; see [performance.md](performance.md#choosing-a-quant-for-speed).

Set `HF_TOKEN` for gated or private repos. If you already have a model library
from LM Studio, it serves as-is (the files are plain GGUFs):
`gmlx init --models-dir ~/.lmstudio/models` in the next section picks it up, and
the wizard offers the directory on its own. Coming from llama.cpp or Ollama?
[migrating.md](migrating.md) maps what carries over.

### Will it fit?

Weights cost about the file size. For a standard dense model, the KV cache costs:

```text
bytes per token = 2 (K and V) x layers x kv_heads x head_dim x 2 (bf16)
```

An 8B-class model (32 layers, 8 KV heads, head dim 128) uses 128 KB per token of
context, so a 32k-token session adds 4 GB on top of the weights. A 32B-class dense
model (64 layers, same heads) uses 256 KB per token: 8 GB at 32k. The layer and head
counts are in the GGUF metadata, and the model card lists them too.

If weights plus cache crowd your RAM, quantize the cache: `--kv-bits 8` roughly
halves it at nearly no quality cost, and `--kv-bits 4` roughly quarters it with a
small cost at long range. In server configs the same knob is the `kv_bits` load key.

Several families are much cheaper than the formula suggests. Sliding-window layers
(gemma) stop growing at the window size, hybrid linear-attention models (Qwen3.5 and
3.6, Falcon-H1, Granite 4.x) keep a small fixed state on most layers, and MLA models
(DeepSeek) store a compressed cache. Worked numbers:
[performance.md](performance.md#memory-and-the-kv-cache).

## Set up the server

`gmlx init` scaffolds the config. Run bare in a terminal, it opens a guided wizard
that walks through:

1. Model directories to scan (recursive optional), plus your local Hugging Face
   cache if it holds GGUFs.
2. The discovered models: rename ids, drop entries, mark a default, add aliases.
3. A per-family sampling summary, with an optional default intent per family.
4. The on-disk prompt cache (recommended for coding agents) and its size cap.
5. Optional services: speech-to-text, text-to-speech, embeddings, reranking. If you
   configure both STT and TTS, a voice-chat step follows.
6. Idle unload (how long an unused model stays resident) and a request timeout.
7. Where to write the file, with a preview before anything is saved.

Prefer flags? `gmlx init --models-dir ~/models` scaffolds non-interactively; every
wizard choice has a flag equivalent (`--with-stt`, `--disk-cache`, `--default-model`,
and so on).

The config lands at `~/.config/gmlx/gmlx.yaml`. It is one YAML file with a
`server:` block (port, model directories, services), a `models:` block (one entry
per model: path, optional per-model settings), and optional `profiles:`, `rules:`,
`aliases:`, and `talk:` blocks. Every optional key appears as a commented hint with
its default, so the file documents itself. The full reference is
[server-config.md](server-config.md).

Then:

```sh
gmlx serve            # finds the config, detaches, returns immediately
gmlx status           # pid, uptime, url
gmlx ps               # which models are resident right now
gmlx logs -n 20 -f    # follow the server log
gmlx stop             # shut it down
```

`serve` runs in the background by default so you keep your shell (pass `-f` to stay
attached). On a macOS GUI session it also raises a small menu-bar monitor showing
the resident models, with unload, restart, log, and config-editing controls.

## Talk to it over HTTP

The server speaks the OpenAI API (plus Anthropic and OpenAI Responses on the same
port). The `model` field is whatever id `init` printed for your file - auto-named
ids carry the quant tag (`qwen3-0.6b-q4`, `qwen3.6-27b-q6`); `gmlx list` shows
them, and an unknown id gets a 404 listing the valid ones. With the small model
from the walkthrough above:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3-0.6b-q4",
  "messages": [{"role": "user", "content": "Explain entropy in one paragraph."}]
}'
```

Add `"stream": true` for server-sent events. The `model` field takes any id from
your config, an alias, or `id@intent` to switch the sampling operating point per
request (`qwen3.6-27b@coding`, `@instruct`, `@creative`, and friends; `gmlx
profiles` lists them).

From Python, the standard OpenAI client works unchanged
(`pip install openai` - it is not a gmlx dependency):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
reply = client.chat.completions.create(
    model="qwen3-0.6b-q4",
    messages=[{"role": "user", "content": "Explain entropy in one paragraph."}],
)
print(reply.choices[0].message.content)
```

Tool calling, structured output (`response_format: json_schema`), logprobs, and
vision messages all work over this API; see
[server-config.md](server-config.md#api-capabilities).

## Connect a client

`gmlx launch` points an external tool at your server and starts the server first if
it is down. For example, Claude Code on a local model:

```sh
gmlx launch claude-code --model qwen3.6-27b
```

launch exports the environment Claude Code needs (its Anthropic-API base URL and
model), starts your server if necessary with a spinner while the model loads, and
execs `claude`. Your `~/.claude` configuration is never touched. The same one-liner
works for opencode, pi, omp, hermes, goose, the aichat and elia chat clients, and
the Open WebUI browser app. Per-client details: [launch.md](launch.md).

## Chat in your browser

If you would rather click than type in a terminal, Open WebUI gives you a
ChatGPT-style browser app on top of your server. It is a separate program with
its own install (it needs Python 3.11 or 3.12):

```sh
brew install pipx           # once, if you don't have pipx
pipx install open-webui --python python3.12
gmlx launch open-webui
```

`launch` starts your gmlx server if needed, wires Open WebUI to it, runs it on
port 3000, and prints the URL to open. Chat works immediately; if your server
also runs embeddings, speech-to-text, or text-to-speech, document upload and
voice light up too. Details, including a no-login single-user setup:
[launch.md](launch.md#open-webui).

## Optional: voice

With the `[talk]` extra installed and STT + TTS configured (the init wizard offers
both), `gmlx talk` runs a hands-free voice loop against your server: say the wake
phrase, ask, and the reply is spoken back as it streams. The built-in
[assistant](assistant.md) (`talk.brain: assistant`) adds MCP tools and long-term
memory. Setup and worked examples: [talk.md](talk.md).

## Run it at login

```sh
gmlx service install    # menu bar at login; it starts the server when needed
gmlx service status
gmlx service uninstall  # remove the login item
```

macOS only. `service install` accepts the same options as `serve`. The menu
bar runs as the launchd agent (so its permission prompts attribute to gmlx)
and starts the recorded server once per login; `--no-autostart` leaves the
server to its Start menu item, and `--headless` installs a server-only agent
for GUI-less machines (that one restarts on crash and is stopped with
`service uninstall`).

## Where things live on disk

| Path | Contents |
|------|----------|
| `~/.config/gmlx/gmlx.yaml` | your config |
| `~/.config/gmlx/` | client configs written by `gmlx launch` |
| `~/.cache/gmlx/` | server runfiles and logs, chat line-editor history, GGUF header cache |
| `~/.cache/gmlx/apc/` | the on-disk prompt cache, when enabled |
| `~/.cache/gmlx/talk/` | wake-word and voice-activity models (a few MB, first `talk` run) |
| `~/.local/share/gmlx/chats/` | saved chat sessions (`--resume`, `/save`) |
| `~/.local/share/gmlx/assistant-memory.db` | the assistant's long-term memory (served assistants get their own `assistant-<id>.db` beside it) |
| `~/Library/Application Support/gmlx/` | the menu-bar app bundle (created on first `serve`/`service install` on macOS) |
| `~/Library/LaunchAgents/com.gmlx.*.plist` | the login items written by `gmlx service install` |
| `~/.open-webui/` | Open WebUI's chat history, if you use `gmlx launch open-webui` |
| your model directories | the GGUFs themselves; `pull` writes here |

To remove gmlx completely: `gmlx service uninstall` if installed (that removes
the LaunchAgents), delete the directories above, delete the models you pulled,
and `pip uninstall gmlx mlx-kquant`.

## When something goes wrong

The common first-run problems, each with a fix in
[troubleshooting.md](troubleshooting.md): the install fails building the Metal
kernels on an older macOS, `gmlx` is "command not found" in a new terminal (the
venv is not active), a download was interrupted or the disk filled, a file
refuses to load because of IQ codecs, Whisper complains about ffmpeg, the mic
permission prompt never appeared, port 8080 is already in use, or the first
request after a cold start is slow because the model was loading. For anything
else, `gmlx doctor` checks the runtime, config, model paths, and services in one
pass, and `gmlx logs` shows what the server was doing.
