# Agents and chat apps

`gmlx launch` connects a coding agent, a chat app or an agent runtime to
your server. It writes the tool's configuration so that the tool uses the
server, starts the server when none is running, and then runs the tool:

```sh
gmlx launch pi --model qwen3.8-27b-ud-q6    # the pi coding agent on a local model
gmlx launch opencode                         # the server's default model
gmlx launch open-webui                       # a chat app in the browser
```

gmlx never installs the tool itself. When the tool is not on your PATH,
`launch` prints how to install it and exits. The flags and exit codes are
in the [CLI reference](cli.md#gmlx-launch).

## How a launch works

A launch has three steps:

1. It asks the server for its models, aliases and default model. A tool
   that offers a model menu gets these as its choices.
2. It writes the tool's configuration by injection, merge or environment
   variables.
3. It runs the tool in its own place, already connected to the server.

`--config-only` stops after the second step and prints the command that
would run the tool, for inspection or for a script. It also skips the check
for the tool on your PATH. Output from `launch` itself starts with
`[launch]`.

The three styles differ in which files they touch:

- Injection writes a configuration file under `~/.config/gmlx/` and points
  the tool at it through an environment variable or an option that the tool
  reads. The tool's own configuration file is never written.
- Merge adds a provider for the server to the tool's own configuration
  file and keeps the providers already there. It refuses to change a file
  that it cannot parse.
- Environment passes every setting in environment variables, with no file.

Each client uses one or two of these styles:

| Client | What it is | Style | Where the configuration goes |
|--------|------------|-------|------------------------------|
| `claude-code` | Anthropic's Claude Code | Environment | `ANTHROPIC_*` variables |
| `opencode` | A coding agent | Injection | `~/.config/gmlx/opencode.json`, through `OPENCODE_CONFIG` |
| `pi` | A coding agent | Merge | `~/.pi/agent/models.json` and `settings.json` |
| `omp` | oh-my-pi, a coding agent | Merge | `~/.omp/agent/models.yml` and `config.yml` |
| `hermes` | NousResearch hermes-agent | Injection | `~/.config/gmlx/hermes-config.yaml`, through `HERMES_CONFIG` |
| `goose` | Block's agent runtime | Merge and environment | `~/.config/goose/config.yaml` |
| `aichat` | A terminal chat client with tools | Injection | `~/.config/gmlx/aichat/`, through `AICHAT_CONFIG_DIR` |
| `elia` | A terminal chat app | Injection | `~/.config/gmlx/elia-xdg`, through `XDG_CONFIG_HOME` |
| `open-webui` | A chat app for the browser | Environment | `OPENAI_API_BASE_URL` and related variables |
| `dsh` | DeepSeek Harness, an agent app for the browser | Injection | `~/.config/gmlx/dsh/gmlx.cordis.yml`, through `--patch` |

`--config-path` moves the written configuration to the path you give,
which takes the place of the location that the table lists for the client.
It names a file for opencode, hermes and dsh, and a directory for pi, omp
and aichat. For goose it names the `config.yaml`, for elia the
`XDG_CONFIG_HOME` directory, and for Open WebUI the data directory.

## Starting the server

When no server answers, `launch` starts one in the background from the
first configuration file in the
[places gmlx looks](config.md#where-gmlx-looks), and it waits until the
server responds. On a Mac desktop that also opens the
[menu bar app](menubar.md), unless
[`server.menubar`](config.md#servermenubar) is `false`. With no
configuration file anywhere, `launch` says to run `gmlx init` and exits
with code 2.

The wait has no fixed limit. Only the server process exiting counts as a
failure. Ctrl-C stops the wait, and the server keeps starting in the
background. `--start-timeout SECONDS` limits the wait for scripts, and
`--no-start` turns off starting entirely, so the command fails when no
server runs.

`--base-url URL` names a server explicitly. `launch` then never starts one
and reads no configuration file, so a configuration file in the current
folder cannot redirect the session or supply a key.

The first request to a model that is not loaded waits for the load. Which
models the server loads at start is under
[Memory and residency](config.md#memory-and-residency).

## Choosing the model

`--model ID` selects the model that the tool uses, and without it the tool
gets the server's default model. An id with a profile, such as
`--model qwen3.8-27b-ud-q6@coding`, applies that profile to every request
from the tool. `launch` checks the id against the models that the server
lists. The `claude-code`, `dsh`, `goose` and `hermes` clients cannot start
without a model, so pass `--model` for them or set
[`server.defaults.model`](config.md#serverdefaultsmodel). `dsh` also starts
when the server has exactly one chat model.

With `--model`, `launch` also asks the server to load the model and keep it
loaded through the idle timeout, so that the model is not unloaded between
the turns of a long session. The keep lasts after the tool exits,
until `POST /unload` or a request to `POST /v1/keep` with
`{"keep": false}` releases it, or the server restarts. A kept model can
still be unloaded when the memory budget needs the room, as
[Memory and residency](config.md#memory-and-residency) describes. `--no-keep`
skips the request.

## Authentication

A server with an [API key](config.md#serverapi_key) refuses a launch
without the key before the tool starts, and `launch` says to pass
`--api-key`. When `launch` starts the server itself from a configuration
file, it takes the key from that file. Each tool gets the key in its own
setting:

| Client | Where the key goes |
|--------|--------------------|
| `opencode` | `options.apiKey` in the injected file. |
| `pi` | `apiKey` in the merged provider. |
| `omp` | Nowhere, because omp has no setting for it. `launch` prints a note, and you set up omp's authentication yourself. |
| `hermes` | `providers.custom.api_key` in the injected file. |
| `goose` | `OPENAI_API_KEY` in the environment only, never in the file. |
| `claude-code` | `ANTHROPIC_AUTH_TOKEN` in the environment. |
| `aichat`, `elia` | `api_key` in the injected file. |
| `open-webui` | `OPENAI_API_KEY` in the environment only. |
| `dsh` | `GMLX_API_KEY` in the environment only. |

Without a key on the server, each tool still gets a placeholder key,
because some tools refuse to run without one.

## The clients

### claude-code

Claude Code uses the server's Anthropic API. `launch` sets
`ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL` and
`ANTHROPIC_AUTH_TOKEN`, and it removes an inherited `ANTHROPIC_API_KEY` so
that the token takes effect. It does not change `~/.claude`.

Its system prompt is very long, and it often rewrites the start of its
requests, so processing the prompt takes most of the time of a turn.
Turn on the [prompt cache](config.md#prompt-cache), and prefer a model and
a Mac with fast prefill.

### opencode, pi and omp

These three coding agents take the default model in different places.
opencode takes it in the `model` key of the injected file, pi as
`defaultProvider` and `defaultModel` in its merged files, and omp as
`modelRoles.default`. For pi, `launch` also sets the context window and
output limit of each model from the server's model list.

### hermes

`launch` reads your `~/.hermes/config.yaml`, adds the gmlx provider, and
writes the result to the injected file, which it passes through
`HERMES_CONFIG` and `CUSTOM_BASE_URL`. Your own file is not changed. hermes
refuses a model with less than 64K tokens of context, so give it a model
trained for at least that length.

### goose

`launch` merges `GOOSE_PROVIDER`, `GOOSE_MODEL`, `OPENAI_HOST` and
`OPENAI_BASE_PATH` into the goose file and also sets them, with
`OPENAI_API_KEY`, in the environment, where they take precedence. It runs
`goose session`. A later `goose` with no launch still finds the server
through the file.

### aichat

Each served model is marked as supporting function calls, so aichat's
tools and agents work with the server. Running tools also needs aichat's
`llm-functions`.

### elia

Each served model becomes an OpenAI-compatible model in elia, and `launch`
starts elia on the selected model. elia 1.x or newer is required. An older
elia starts but lists no local models, so upgrade it with
`pipx upgrade elia-chat`.

### open-webui

Open WebUI is a chat app that runs its own web server, so this launch
starts a second service. Install it first with
`pipx install open-webui --python python3.12`, because it needs Python 3.11
or 3.12.

`launch` sets the server address and key, turns off Open WebUI's Ollama
connection, and sets its data directory. The app runs on port 3000, or on
3001 when the gmlx server uses 3000, and `launch` prints its address. Chat
history is stored in `~/.open-webui`, or in the folder that
`--config-path` names. For a single user with no login, add
`WEBUI_AUTH=false` to its environment before the first launch on a new
data directory.

Open WebUI gets a feature for each service that the server runs, as
[Speech, embeddings and rerank](services.md) describes:

| Server runs | Open WebUI gets |
|-------------|-----------------|
| Chat models only | Chat. Its document embedder points at the server, so it starts without downloading one. |
| `embeddings` | Document search, as [RAG pipelines](rag.md) describes. |
| `rerank` | Hybrid search with the server's reranker at `/v1/rerank`. |
| `stt` | Speech input through `/v1/audio/transcriptions`. |
| `tts` | Spoken replies through `/v1/audio/speech`. |

### dsh

DeepSeek Harness is an agent app that runs in the browser. Install version
0.1.7 or newer with `npm install -g @deepseek-ai/dsh@next`. `launch`
refuses an older version and prints that command.

`launch` runs dsh with a profile of its own, `gmlx`, under
`$DSH_HOME/profiles/`, where `DSH_HOME` defaults to `~/.dsh`. The first
launch creates the profile from dsh's `web` template, and later launches
reuse it. Your other dsh profiles are not changed. A `gmlx` folder there
without a `package.json` is refused, so remove or rename it first.

The providers, the default model and the title and compaction settings go
in `~/.config/gmlx/dsh/gmlx.cordis.yml`, which `launch` passes to dsh with
`--patch`. dsh applies that file above its own settings and never saves
it, so the web app cannot save a different default model or an edit to the
gmlx providers. Run `launch` again with `--model` to change the default.

Two entries in the file point at the server. Under `gmlx (local)`, the
server and its profiles decide whether a model thinks. Under
`gmlx (thinking off)`, the same models answer without thinking, and dsh
writes its session titles there with the default model.

Port 3080 serves the web app, or 3081 when the gmlx server uses 3080, and
the launch opens a browser. The app starts in
`~/Documents/deepseek-harness/default-workspace`, not in the folder you
launch from, and Add workspace in the app opens a project folder.

dsh compacts a conversation by itself only when the model's context is
large enough for its default headroom, and `launch` prints a note when it
is not. Otherwise a conversation still compacts when the server reports
that a request no longer fits, as
[Limits and back-pressure](api.md#limits-and-back-pressure) describes.

The `headless` profile of dsh works with the same file. This command answers
one task about the current folder and exits:

```sh
GMLX_API_KEY=gmlx dsh --profile headless \
  --patch ~/.config/gmlx/dsh/gmlx.cordis.yml "run the tests"
```

`--dsh-profile NAME` starts another dsh profile with the same file, for
example a terminal profile you built with `dsh plugin`. dsh creates its
own shipped profiles on first use, and any other profile must exist before
the launch. The `acp`, `sdk` and `sdk-minimal` profiles serve a program
over stdio, so `launch` runs them only with `--config-only` and prints the
command to give that program. `launch` refuses the `desktop` profile.

dsh's `web_search` tool uses DeepSeek's search service and needs
`DEEPSEEK_API_KEY`. dsh sends a conversation to DeepSeek only with feedback
that you submit, and `DSH_TELEMETRY_MODE=DISABLED` turns that off.
