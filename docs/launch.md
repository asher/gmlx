# Connect coding agents and chat apps

This guide is for configuring an external tool to use your server: a coding
harness, an agent runtime, a terminal chat client or the Open WebUI browser
app. `gmlx launch <client>` probes the server, writes the tool's native
configuration without modifying your own dotfiles, and runs the tool. If no
server is answering, it starts one first.

```sh
gmlx launch pi --model qwen3.6-27b            # pi on a local model
gmlx launch opencode                          # uses the server's default model
gmlx launch open-webui                        # browser chat app on :3000
```

launch never installs the tool itself. If the binary is not on PATH, it
prints an install hint and exits. The flag table and exit codes are under
[gmlx launch](cli.md#gmlx-launch).

- [How a launch works](#how-a-launch-works)
- [Starting the server automatically](#starting-the-server-automatically)
- [Choosing the model](#choosing-the-model)
- [Authentication](#authentication)
- [The clients](#the-clients)
- [Troubleshooting](#troubleshooting)

## How a launch works

1. Probe. launch checks `/health` and `/v1/models`. Served ids, aliases and
   the default-model marker come from `/v1/models`, so they are selectable
   inside menu-driven tools.
2. Configure. Each client is configured in one of three styles, listed in the
   table below. Injection writes a config under `~/.config/gmlx/` and points
   the tool at it through the tool's own mechanism, so your own config for
   that tool is never read or written. Merge adds a provider block to the
   tool's own file, preserving existing providers, and refuses to overwrite a
   file it cannot parse. Environment passes everything in the exec
   environment with no file at all.
3. Exec. The tool replaces the launch process, connected to your server.

`--config-only` writes the configuration and prints the run command instead
of running the tool, for inspection or scripting.

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

If no server answers, launch starts one in the background from the first
config in a [default location](server-config.md#default-config-locations)
and polls until it responds. When that config preloads a model, because it
pins one, names a default, or holds exactly one, the server loads the
weights before binding its port, and the spinner names the model while it
loads. When nothing is preloaded, the server answers in about a second and
the model loads on the first request, which makes that first turn slower.

There is no fixed timeout: only the server process exiting is a failure.
Ctrl-C stops waiting while the server keeps starting. With no config
anywhere, launch prints `gmlx init` guidance and starts nothing.

- `--no-start` never auto-starts, and launch errors if the server is not running.
- `--start-timeout SECONDS` caps the wait, for non-interactive use.
- An explicit `--base-url` is never auto-started and reads no config, so a
  unexpected project-local config cannot redirect the session or supply an
  unexpected key.

## Choosing the model

`--model ID` picks which served model the tool uses; without it, the tool
gets the server's default. An `id@profile` form such as
`--model qwen3.6-27b@coding` runs every request from the tool at that
profile's sampling, and the id is validated against the served list.

When you pass `--model`, launch also asks the server to keep that model
resident through the idle timeout, so a long session's model is not unloaded
during the session. This is not a pin: under memory pressure the pool can still evict
it. `gmlx ps` shows the model as kept, `POST /unload` releases it, and
`--no-keep` opts out.

To give a coding agent the coding intent and keep it resident:

1. Serve a model with tool-calling support; Qwen3.6-27B is the recommended size.
2. Run `gmlx launch pi --model qwen3.6-27b@coding`.
3. Leave the session open; the model stays kept until it ends.

## Authentication

`--api-key KEY` passes the key the server runs with, and launch writes it
to each tool's native setting:

| Client | Where the key goes |
|--------|--------------------|
| opencode | `options.apiKey` in the injected config |
| pi | `apiKey` in the merged provider block |
| omp | no API-key slot; configure its auth manually, launch prints a note |
| hermes | `providers.custom.api_key` in the injected config |
| goose | `OPENAI_API_KEY` in the exec environment only, never the YAML |
| claude-code | `ANTHROPIC_AUTH_TOKEN` in the exec environment |
| aichat, elia | `api_key` in the injected config |
| open-webui | `OPENAI_API_KEY` in the exec environment only |

## The clients

### claude-code

Uses the server's Anthropic API. launch exports
`ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_FAST_MODEL`,
and `ANTHROPIC_AUTH_TOKEN` (a placeholder when the server has no auth). A
model is required: pass `--model` or set `server.defaults.model`. An
inherited `ANTHROPIC_API_KEY` is dropped so the injected token takes effect, and
`~/.claude` is never modified.

Claude Code is prefill-heavy: it sends a very long system prompt and often
rewrites its request prefix through compaction and tool results, so turn
latency is dominated by prompt processing. Serve with the
[prompt cache](performance.md#the-prompt-cache) on and prefer a model and
machine with strong prefill throughput.

### opencode

The default model is written to the injected file's top-level `model` key.

### pi

launch sets `defaultProvider` and `defaultModel` in the merged files.

### omp

launch sets `modelRoles.default`. omp's provider registry has no API-key
slot.

### hermes

The injected file is your own `~/.hermes/config.yaml` merged with the gmlx
provider block, injected with `HERMES_CONFIG` plus `CUSTOM_BASE_URL`. A model
is required. hermes refuses models with less than 64k context at startup, and
the window comes from the GGUF metadata, so give it a model whose trained
context is at least 64k tokens.

### goose

launch merges the non-secret pointer keys into goose's own file and also
exports them as environment variables, which take precedence in goose. A
model is required.

### aichat

Every served id is flagged as supporting function calling, so aichat's tools
and agents work against the server's tool-call surface. Tool execution still
needs aichat's `llm-functions` installed.

### elia

Each served id becomes an OpenAI-compatible litellm model. Requires elia 1.x
or newer; upgrade with `pipx upgrade elia-chat`.

### open-webui

Open WebUI is a browser chat app and itself a web server, so this launch
starts a second service rather than a terminal client. Install it
separately with `pipx install open-webui --python python3.12`; it needs
Python 3.11 or 3.12.

launch exports the base URL and key, disables the Ollama API, sets
`DATA_DIR`, runs it on port 3000 (3001 if the gmlx server holds 3000), and
prints the URL. Chat history is stored in `~/.open-webui` on the host, or at
`--config-path`. Add `WEBUI_AUTH=false` to its environment for a no-login
single-user setup on a fresh data directory.

The configuration depends on which services the server reports:

| Server runs | Open WebUI gets |
|-------------|-----------------|
| chat only | chat, with its document embedder pointed at this server so it starts without downloading one |
| `embeddings` | document RAG ([rag.md](rag.md)) |
| `rerank` as well | hybrid search with the external reranker at `/v1/rerank` |
| `stt` and `tts` | audio engines at the server's `/v1/audio/*` endpoints, with a Kokoro voice as the default |

## Troubleshooting

- `command not found` for the tool: launch does not install clients. Follow
  the install hint it prints, then rerun.
- Exit code 2 and an init hint: no config exists in a default location. Run
  `gmlx init`, or pass `--base-url` for an already-running server.
- The tool connects but completions fail with 401: the server has an API
  key. Rerun launch with `--api-key` and the same key.
- hermes exits at startup with a context-length error: the served
  model's window is under 64k tokens. Pick a larger-context model.
- elia starts but shows no local models: the installed elia is older than
  1.x.
- open-webui fails to install or start: check the pipx Python version.
- The first turn is slow: nothing was preloaded. Set `server.defaults.model`
  or pass `--model` so the auto-start loads it before binding the port.
