# Connect coding agents and chat apps

`gmlx launch <client>` points an external tool at a gmlx server and runs it:
a coding harness (pi, opencode, omp, claude-code), an agent runtime (hermes, goose),
a terminal chat client (aichat, elia), or the Open WebUI browser app. It probes the
server, writes the tool's native configuration without touching your own dotfiles,
and execs the tool. If no server is answering, it starts one in the background first.

```sh
gmlx launch pi --model qwen3.6-27b            # pi on a local model
gmlx launch opencode                          # uses the server's default model
gmlx launch open-webui                        # browser chat app on :3000
```

launch never installs the tool itself. If the binary is not on PATH, it prints an
install hint and exits.

## How a launch works

1. Probe. launch checks `/health` and `/v1/models` on the target server. Aliases and
   the default-model marker come straight from `/v1/models`, so served ids and
   profile presets are pickable inside menu-driven tools.
2. Configure. There are three styles, chosen per client:
   - Injection style (opencode, hermes, aichat, elia). launch writes a config under
     the gmlx namespace, `~/.config/gmlx/` by default, and points the tool at
     it through the tool's own mechanism (an environment variable such as
     `OPENCODE_CONFIG`). Your own config for that tool is never read or written.
   - Merge style (pi, omp, goose). These tools have no config-injection mechanism, so
     launch merges a provider block into the tool's own file. Existing providers,
     roles, and extensions are preserved, and launch refuses to overwrite a file it
     cannot parse.
   - Environment style (claude-code, open-webui). No config file at all; everything
     travels in the exec environment.
3. Exec. The tool replaces the launch process, connected to your server.

`--config-only` writes the configuration and prints the run command instead of
exec'ing, which is useful for inspection or for wiring the tool into scripts.

## Starting the server automatically

If no server answers at the target endpoint, launch starts one in the background from
the first config found in a default location (`./gmlx.yaml`,
`~/.config/gmlx/gmlx.yaml`, `~/.gmlx.yaml`), then polls with a spinner
until it responds.

When that config preloads a model (it pins one, `server.defaults.model` names
one, or contains exactly one), the server loads those weights before binding
its port. The spinner
names the model and its on-disk size while it loads, and the model is hot for the
tool's first turn. When nothing is preloaded, the server answers in about a second
and the model loads lazily on the first request, which makes that first turn slower.

There is no fixed timeout; only the server process dying is a hard failure. Press
Ctrl-C to stop waiting (the server keeps starting in the background). With no config
anywhere, launch prints `gmlx init` guidance and starts nothing.

- `--no-start` never auto-starts; launch errors if the server is down.
- `--start-timeout SECONDS` caps the readiness wait (default `0`, meaning wait as
  long as the server process lives). Set it for non-interactive use where Ctrl-C is
  not available.
- An explicit `--base-url` is never auto-started: launch either reaches that server
  or errors, and it reads no config for a server that is already answering. This
  keeps a stray project-local `./gmlx.yaml` from silently redirecting the session
  or supplying an unexpected API key.

## Choosing the model

`--model ID` picks which served model the tool uses. Without it, the tool gets the
server's default-marked model. The id half is validated against the served list; an
`id@profile` form (`--model qwen3.6-27b@coding`) runs every request from the tool at
that profile's sampling, validated by the server.

When you pass `--model`, launch also asks the server to keep that model resident
through its idle-TTL reaper, so a long coding session's model is not idle-unloaded
mid-use (which would force a cold reload on the next turn). This is not a full pin:
under memory pressure the pool can still evict it. The request is fire-and-forget;
the server warm-loads in the background while the tool execs. `gmlx ps` shows the
model as `kept`, and `POST /unload {model}` releases it. Pass `--no-keep` to opt out.

## Authentication

`--api-key KEY` passes the same key the server runs with (its `server.api_key`).
launch carries it into each tool's configuration in that tool's native slot:

| Client | Where the key goes |
|--------|--------------------|
| opencode | `options.apiKey` in the injected config |
| pi | `apiKey` in the merged provider block |
| omp | no API-key slot in its provider registry; launch prints a note to configure auth manually |
| hermes | `providers.custom.api_key` in the injected config |
| goose | `OPENAI_API_KEY` in the exec environment only, never written to its `config.yaml` (which may hold a real OpenAI credential) |
| claude-code | `ANTHROPIC_AUTH_TOKEN` in the exec environment |
| aichat, elia | `api_key` in the injected config |
| open-webui | `OPENAI_API_KEY` in the exec environment only |

## Flags

| Flag | Meaning |
|------|---------|
| `--model ID[@profile]` | Served model (and optional profile) the tool should use. Default: the server's default-marked model. |
| `--base-url URL` | Target an explicit server. Never auto-started. |
| `--host H` / `--port P` | Target host/port for the managed server (default `127.0.0.1:8080`). |
| `--api-key KEY` | Key for a key-protected server, plumbed per the table above. |
| `--provider-id NAME` | Provider id written into the tool's config. |
| `--config-path PATH` | Where the tool config is written (default under `~/.config/gmlx`). For open-webui this picks its `DATA_DIR` instead. |
| `--config-only` | Write the config and print the run command; do not exec. |
| `--no-start` | Never auto-start a server. |
| `--start-timeout S` | Cap the auto-start readiness wait (default `0` = unbounded). |
| `--no-keep` | Do not ask the server to keep `--model` resident. |

Exit codes: `0` tool launched (or server started and ready); `1` server down with
`--no-start` or `--base-url`, a launchd server mid-restart, or the auto-started
process died; `2` no config or a malformed config; `130` Ctrl-C during the start wait.

## The clients

### claude-code

Anthropic Claude Code, driven over the server's Anthropic surface (`/v1/messages`).
Pure environment injection; it never touches `~/.claude`. The `claude` binary must be
on PATH. launch exports:

- `ANTHROPIC_BASE_URL`: the server root. Claude Code appends `/v1/messages` itself.
- `ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_FAST_MODEL`: the model to use. Required:
  pass `--model` or set `server.defaults.model` in the server config.
- `ANTHROPIC_AUTH_TOKEN`: the `--api-key` value, or a placeholder when the server has
  no auth. An inherited `ANTHROPIC_API_KEY` is dropped from the exec environment so
  the injected token wins.

Claude Code on local models is prefill-heavy: it sends a very long system prompt
(tens of thousands of tokens) and frequently rewrites its request prefix through
context compaction and tool-result injection, so KV-prefix reuse across requests is
limited and turn latency is dominated by prompt processing. Serve with the prompt
cache enabled (`cache:` in the config, see
[server-config.md](server-config.md#cache-keys-cache)) to soften repeated prefixes,
and prefer a model and machine with strong prefill throughput.

### opencode

Injection style: launch writes `~/.config/gmlx/opencode.json` and points opencode
at it via `OPENCODE_CONFIG`. Your own opencode config is untouched. The default model
lands in the top-level `model` key.

### pi

Merge style: launch merges the provider into `~/.pi/agent/models.json` and
`~/.pi/agent/settings.json`, setting `defaultProvider` and `defaultModel`. Other
providers in the files are preserved.

### omp (oh-my-pi)

Merge style: launch merges into `~/.omp/agent/models.yml` and
`~/.omp/agent/config.yml`, setting `modelRoles.default`. omp's provider registry has
no API-key slot, so on a key-protected server configure its auth manually (launch
prints a note).

### hermes

NousResearch hermes-agent. Injection style: launch writes
`~/.config/gmlx/hermes-config.yaml` (your `~/.hermes/config.yaml` merged with the
gmlx provider block) and injects it via `HERMES_CONFIG` plus `CUSTOM_BASE_URL`.
Your own file is untouched. A model is required (`inference.model`).

hermes refuses models with less than 64k context at startup, so serve it a model
whose context window is at least 64k tokens. The window comes from the model's GGUF
metadata and is not configurable server-side; the `max_kv_size` load key only caps
the rolling KV cache, it does not change the advertised window.

### goose

Block's goose. Merge style: launch merges the non-secret pointer keys into
`~/.config/goose/config.yaml` (existing keys preserved) and also exports the same
values as environment variables at exec, which take precedence in goose. A model is
required (`GOOSE_MODEL`). The API key travels as `OPENAI_API_KEY` in the exec
environment only; it is never written into the YAML, where it could clobber a real
OpenAI credential.

### aichat

sigoden/aichat, a chat-focused terminal REPL with tools and agents (not a coding
harness). Injection style: launch writes a `config.yaml` with an `openai-compatible`
client under `~/.config/gmlx/aichat/` and injects it via `AICHAT_CONFIG_DIR`;
your own `~/.config/aichat` is untouched. Every served id is flagged
`supports_function_calling`, so aichat's tools and agents work against the server's
tool-call surface. Actual tool execution still needs aichat's `llm-functions`
installed.

### elia

darrenburns/elia, a chat TUI (not a coding harness). Injection style: launch writes a
fresh `config.toml` under `~/.config/gmlx/elia-xdg` and injects it via
`XDG_CONFIG_HOME`; your own `~/.config/elia` is untouched. Each served id becomes an
OpenAI-compatible litellm model. Requires elia 1.x or newer (older builds ignore
custom endpoints); upgrade with `pipx upgrade elia-chat`.

### open-webui

Open WebUI is a browser chat app, itself a web server, so this launch starts a second
service rather than a terminal client. Install it separately:
`pipx install open-webui --python python3.12` (it needs Python 3.11 or 3.12, not
3.13).

Environment style, no config file: launch exports `OPENAI_API_BASE_URL`,
`OPENAI_API_KEY`, `ENABLE_OLLAMA_API=false`, and `DATA_DIR`, runs it on port 3000
(passed as `serve --port`, since `open-webui serve` ignores the `PORT` variable and
would otherwise collide with the gmlx server on 8080; if the gmlx server itself is
bound to 3000, Open WebUI bumps to 3001), and prints the URL to
open. Chat history and its sqlite database land at `DATA_DIR` on the host filesystem
(default `~/.open-webui`; `--config-path` overrides), not in a Docker volume.

launch points Open WebUI's document-RAG embedder back at the gmlx server
(`RAG_EMBEDDING_ENGINE=openai`) instead of the default local HuggingFace download, so
it boots cleanly without fetching or caching an embedder. Chat works immediately; run
the server with `--embeddings` and document RAG works too (see [rag.md](rag.md)). If the
server also advertises a reranker (started with `--rerank`), launch turns on Open WebUI's
hybrid search and points its external reranker at the server's `/v1/rerank`
(`RAG_RERANKING_ENGINE=external`); reranking only runs under hybrid search.

Audio is capability-gated: when the server advertises STT and TTS in `/v1/models`
(run it with `--stt` / `--tts`), launch also wires Open WebUI's audio engines
(`AUDIO_STT_ENGINE=openai`, `AUDIO_TTS_ENGINE=openai`) at the server's `/v1/audio/*`
endpoints, defaulting the read-aloud voice to a Kokoro-valid one (`af_heart`). A
chat-only server keeps Open WebUI's built-in browser audio untouched.

Add `WEBUI_AUTH=false` to its environment for a no-login single-user setup (fresh
`DATA_DIR` only).

## The menu bar app: `launch menubar`

`gmlx launch menubar` puts a small macOS status-bar item up for a backgrounded
server: up/down state (the dot fills in while requests are generating or queued),
the resident models (size, default/pinned/kept markers, an eviction countdown on
idle ones; click one to unload it), reload-config, restart, stop, copy-URL, and
open-logs, all over the existing HTTP endpoints. If a tracked server dies, it
posts a macOS notification (an intentional stop or restart does not). A background
`gmlx serve` raises it for you on a macOS GUI session, so you rarely run it by hand.
Disable that with `--no-menubar` or `server.menubar: false`. To keep it (and the
server) across reboots, install it as a login item with `gmlx service install`
([getting-started](getting-started.md#run-it-at-login)) - that also makes macOS
permission prompts attribute to gmlx instead of your terminal.

"Edit config" opens the server's YAML in a floating editor panel. Validate runs
the draft through the server's own config parser, so the verdict - a typo'd key,
a bad value, a model path that doesn't resolve - is exactly what `gmlx serve`
would say, caught before the server ever sees the file. Save writes atomically
and refuses (once) if the file changed on disk while you were editing; Save &
Reload validates first, then saves and triggers the running server's config
reload in one step. Open in Editor hands the file to your default text editor
instead. The item is there whenever the bar knows a config: the tracked
server's own file or, with everything stopped, the default config location -
fixing the config is usually why you are there.

Like `serve`, it detaches by default; pass `-f` / `--foreground` to run the event
loop in place, and `--stop` to quit a detached monitor from the CLI. One menu bar per machine, deduplicated via a pidfile: a second `serve`
on any port, or a manual `launch menubar`, is a no-op. With no explicit target it
tracks the primary server (the single managed one, else `127.0.0.1:8080`), following
it as servers come and go; pass `--url`, `--host`, or `--port` to pin it to one.

It reads the API key from the managed server's own `server.api_key`, or takes
`--api-key` for a server whose config it cannot see. A key-protected server it has no
key for shows as up with a key-required note, never as down. macOS only (it needs
`rumps`, a default dependency there); on Linux or over SSH it prints a one-line
notice. `--interval S` sets the poll interval (default 4 seconds).

When the tracked server also advertises STT and TTS, the menu gains a voice-chat
item that runs a full talk session inside the menu bar app, no terminal needed. See
[talk.md](talk.md#menu-bar-voice-sessions).

## Troubleshooting

- "command not found" for the tool: launch does not install clients. Follow the
  install hint it prints, then rerun.
- Exit code 2 and an init hint: no config exists in a default location. Run
  `gmlx init`, or pass `--base-url` at an already-running server.
- The tool connects but completions fail with 401: the server has `server.api_key`
  set. Rerun launch with `--api-key <the same key>`.
- hermes exits at startup complaining about context length: the served model's
  window is under 64k tokens. Pick a larger-context model.
- elia starts but shows no local models: the installed elia is older than 1.x.
  `pipx upgrade elia-chat`.
- open-webui fails to install or start: check the pipx Python version; it must be
  3.11 or 3.12.
- First turn is slow: nothing was preloaded, so the model loaded on the first
  request. Set `server.defaults.model` in the config, or pass `--model`, so the
  auto-start loads it before binding the port.
