# Connect coding agents and chat apps

This guide covers configuring an external tool to use your server. The tool
can be a coding harness, an agent runtime, a terminal chat client or the Open
WebUI browser app. `gmlx launch <client>` probes the server, writes the tool's
native configuration without modifying your dotfiles and runs the tool. If no
server is answering, it starts one first.

```sh
gmlx launch pi --model qwen3.6-27b            # pi on a local model
gmlx launch opencode                          # uses the server's default model
gmlx launch open-webui                        # browser chat app on :3000
```

The tool itself is never installed for you. When its binary is not on PATH,
`gmlx launch` prints an install hint and exits, and you rerun it after
installing. The flag table and exit codes are under
[gmlx launch](cli.md#gmlx-launch).

- [How a launch works](#how-a-launch-works)
- [Starting the server automatically](#starting-the-server-automatically)
- [Choosing the model](#choosing-the-model)
- [Authentication](#authentication)
- [The clients](#the-clients)

## How a launch works

A launch has three steps. It probes `/health` and `/v1/models`, and the
served ids, aliases and default-model marker it reads there become the
choices a menu-driven tool offers. It then writes the tool's configuration
in the style the table below gives for that client. Finally it execs the
tool, which replaces the launch process already connected to your server.
`--config-only` stops after the second step and prints the run command
instead, for inspection or scripting.

The three configuration styles differ in what they touch. Injection writes
a config under `~/.config/gmlx/` and points the tool at it through an
environment variable the tool honors, so the tool's own config is never
read or written. Merge adds a provider block to the tool's file, preserves
the providers already there and refuses to overwrite a file it cannot
parse. Environment passes everything in the exec environment with no file
at all.

| Client | What it is | Style | Where the config goes |
|--------|------------|-------|-----------------------|
| claude-code | Anthropic Claude Code | environment | `ANTHROPIC_*` variables |
| opencode | coding harness | injection | `~/.config/gmlx/opencode.json` via `OPENCODE_CONFIG` |
| pi | coding harness | merge | `~/.pi/agent/models.json` and `settings.json` |
| omp | oh-my-pi, a coding harness | merge | `~/.omp/agent/models.yml` and `config.yml` |
| hermes | NousResearch hermes-agent | injection | `~/.config/gmlx/hermes-config.yaml` via `HERMES_CONFIG` |
| goose | Block's agent runtime | merge plus environment | `~/.config/goose/config.yaml` |
| aichat | terminal chat client with tools | injection | `~/.config/gmlx/aichat/` via `AICHAT_CONFIG_DIR` |
| elia | terminal chat TUI | injection | `~/.config/gmlx/elia-xdg` via `XDG_CONFIG_HOME` |
| open-webui | browser chat app | environment | `OPENAI_API_BASE_URL` and related variables |

## Starting the server automatically

If no server answers, a background server is started from the first config
in a [default location](server-config.md#default-config-locations) and polled
until it responds. With no config anywhere, the command prints `gmlx init`
guidance and exits with code 2. Nothing is started when you pass
`--base-url`: the launch uses that address as given and reads no config, so
a project-local config cannot redirect the session or supply an unexpected
key.

The server binds its port as soon as it starts, and a model the config
marks for preloading begins loading in the background at the same moment.
Which model that is, and the `defaults.preload` key that warms more, is
under [Memory and residency](server-config.md#memory-and-residency). A
first turn that arrives before the load finishes waits for the rest of it.
With nothing to preload, the first request carries the whole load, and the
launch prints a note saying so. Passing `--model` avoids that wait, because
the keep request it sends starts loading the model before the tool has asked
for anything.

There is no fixed timeout. Only the server process exiting counts as a
failure, and Ctrl-C stops waiting while the server keeps starting in the
background. `--start-timeout SECONDS` caps the wait for non-interactive use,
and `--no-start` turns auto-start off entirely, so the command fails when the
server is not running.

## Choosing the model

`--model ID` picks which served model the tool uses, and without it the
tool gets the server's default. An `id@profile` form such as
`--model qwen3.6-27b@coding` runs all requests from the tool at that
profile's sampling. The id is validated against the served list. Three
clients, claude-code, goose and hermes, cannot start without a model, so
for them either pass `--model` or set `server.defaults.model`.

When you pass `--model`, the server is also asked to keep that model
resident through the idle timeout, so a long session's model is not
unloaded while you work. This is not a pin, and under memory pressure the
pool can still evict it. `gmlx ps` shows the model as kept, `POST /unload`
releases it, and `--no-keep` opts out.

To give a coding agent the coding profile and keep it resident:

1. Serve a model with tool-calling support. Qwen3.6-27B is the recommended
   size.
2. Run `gmlx launch pi --model qwen3.6-27b@coding`.
3. Leave the session open. The model stays kept until it ends.

## Authentication

`--api-key KEY` passes the key the server runs with, and each tool receives
it in its native setting. A tool that connects but gets 401 on completions
was launched without the key, so rerun with `--api-key`.

| Client | Where the key goes |
|--------|--------------------|
| opencode | `options.apiKey` in the injected config |
| pi | `apiKey` in the merged provider block |
| omp | nowhere. omp's provider registry has no API-key slot, so a note is printed and you configure omp's auth by hand |
| hermes | `providers.custom.api_key` in the injected config |
| goose | `OPENAI_API_KEY` in the exec environment only, never the YAML |
| claude-code | `ANTHROPIC_AUTH_TOKEN` in the exec environment |
| aichat, elia | `api_key` in the injected config |
| open-webui | `OPENAI_API_KEY` in the exec environment only |

## The clients

### claude-code

Claude Code uses the server's Anthropic API, so the launch exports
`ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL` and
`ANTHROPIC_AUTH_TOKEN`, with a placeholder token when the server has no
auth. An inherited `ANTHROPIC_API_KEY` is dropped so that the injected token
takes effect, and `~/.claude` is never modified.

Claude Code is prefill-heavy. It sends a very long system prompt and often
rewrites its request prefix through compaction and tool results, so prompt
processing dominates turn latency. Serve with the
[prompt cache](performance.md#the-prompt-cache) on, and prefer a model and
machine with strong prefill throughput.

### opencode, pi and omp

These three coding harnesses differ only in where the default model lands:
opencode takes it in the injected file's top-level `model` key, pi as
`defaultProvider` and `defaultModel` in the merged files, and omp as
`modelRoles.default`.

### hermes

The injected file is your `~/.hermes/config.yaml` merged with the gmlx
provider block, passed through `HERMES_CONFIG` plus `CUSTOM_BASE_URL`.
hermes refuses at startup any model with less than 64k context, and the
window it sees comes from the GGUF metadata, so a context-length error from
hermes means the served model's trained context is too short. Give it a
model of at least 64k tokens.

### goose

The provider, model and server address are merged into goose's file as
`GOOSE_PROVIDER`, `GOOSE_MODEL`, `OPENAI_HOST` and `OPENAI_BASE_PATH`, and
the same four are exported as environment variables, which take precedence
in goose. A later bare `goose` therefore still finds the server.

### aichat

Each served id is flagged as supporting function calling, which lets aichat's
tools and agents work against the server's tool-call surface. Tool execution
still needs aichat's `llm-functions` installed.

### elia

Each served id becomes an OpenAI-compatible litellm model. elia 1.x or
newer is required, and an older elia starts but lists no local models, so
upgrade with `pipx upgrade elia-chat` if that happens.

### open-webui

Open WebUI is a browser chat app and a web server of its own, so this
launch starts a second service instead of a terminal client. Install it
separately with `pipx install open-webui --python python3.12`. It needs
Python 3.11 or 3.12, and a failure to install or start is usually pipx
holding the wrong Python.

The launch exports the base URL and key, disables the Ollama API and sets
`DATA_DIR`. It runs the app on port 3000, or on 3001 when the gmlx server
itself holds 3000, and prints the URL. Chat history is stored in
`~/.open-webui` on the host, or at `--config-path`. Add `WEBUI_AUTH=false`
to its environment for a no-login single-user setup on a fresh data
directory.

The configuration depends on which services the server reports:

| Server runs | Open WebUI gets |
|-------------|-----------------|
| chat only | chat, with its document embedder pointed at this server so it starts without downloading one |
| `embeddings` | document RAG ([rag.md](rag.md)) |
| `rerank` as well | hybrid search with the external reranker at `/v1/rerank` |
| `stt` and `tts` | audio engines at the server's `/v1/audio/*` endpoints. The voice rule is in [services.md](services.md#text-to-speech-tts) |
