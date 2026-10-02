# Agents and chat apps

`gmlx launch` connects a coding agent, a chat app or an agent runtime to
your server. It writes the tool's configuration so that the tool uses the
server, starts the server when none is running, and then runs the tool:

```sh
gmlx launch pi --model qwen3.8-27b-ud-q6    # the pi coding agent on a local model
gmlx launch opencode                         # the server's default model
gmlx launch open-webui                       # a chat app in the browser
```

gmlx never installs the tool on your Mac. When the tool is not on your
PATH, `launch` prints its install command and exits before it starts the
server, and `gmlx launch CLIENT --help` prints the same command. In
[container mode](launch-container.md), gmlx installs the tool in the
container's image instead. The flags and exit codes are in the
[CLI reference](cli.md#gmlx-launch).

- [How a launch works](#how-a-launch-works)
- [Starting the server](#starting-the-server)
- [Choosing the model](#choosing-the-model)
- [Authentication](#authentication)
- [The clients](#the-clients)

## How a launch works

A launch has three steps:

1. It asks the server for its models, aliases and default model. A tool
   that offers a model menu gets these as its choices.
2. It writes the tool's configuration by injection, merge or environment
   variables.
3. It replaces itself with the tool, which starts already connected to the
   server.

`--config-only` stops after the second step and prints the command that
would run the tool, with each value quoted for the shell, for inspection or
for a script. It also skips the check for the tool on your PATH. Output
from `launch` itself starts with `[launch]`.

The three styles differ in which files they touch:

- Injection writes a configuration file under `~/.config/gmlx/` and points
  the tool at it through an environment variable or an option that the tool
  reads. The tool's own configuration file is never written.
- Merge adds a provider for the server to the tool's own configuration
  file and keeps the providers already there. It refuses to change a file
  that it cannot parse, a file larger than 256 KiB, or a file whose
  providers entry is not a mapping.
- Environment passes every setting in environment variables, with no file.

Each file `launch` writes replaces the old one in a single step, so a
failed launch never leaves it half written. A new file gets mode 600, so
only you can read it, and a rewritten file keeps the old file's mode. A
file that holds a key therefore stays private.

A file that is a symbolic link is written through when the link stays
inside your home folder, which keeps a link into a dotfiles repository
working. `launch` refuses a link that points elsewhere, and a path that
leads through a [private home](glossary.md#private-home) or out of a folder
that a container session shared read-write. It checks every file of the
client before it changes one, so a refusal leaves them all as they were.

Each client uses one or two of these styles:

| Client | What it is | Style | Where the configuration goes |
|--------|------------|-------|------------------------------|
| `claude-code` | It is Anthropic's Claude Code. | Environment | The configuration goes in `ANTHROPIC_*` variables. |
| `opencode` | It is a coding agent. | Injection | The configuration goes in `~/.config/gmlx/opencode.json`, through `OPENCODE_CONFIG`. |
| `pi` | It is a coding agent. | Merge | The configuration goes in `~/.pi/agent/models.json` and `settings.json`. |
| `omp` | It is oh-my-pi, a coding agent. | Merge | The configuration goes in `~/.omp/agent/models.yml` and `config.yml`. |
| `hermes` | It is NousResearch hermes-agent. | Merge | The configuration goes in `~/.hermes/config.yaml`, after a backup of the previous file. |
| `goose` | It is Block's agent runtime. | Merge and environment | The configuration goes in `~/.config/goose/config.yaml`. |
| `aichat` | It is a terminal chat client with tools. | Injection | The configuration goes in `~/.config/gmlx/aichat/`, through `AICHAT_CONFIG_DIR`. |
| `elia` | It is a terminal chat app. | Injection | The configuration goes in `~/.config/gmlx/elia-xdg`, through `XDG_CONFIG_HOME`. |
| `open-webui` | It is a chat app for the browser. | Environment | The configuration goes in `OPENAI_API_BASE_URL` and related variables. |
| `dsh` | It is DeepSeek Harness, an agent app for the browser. | Injection | The configuration goes in `~/.config/gmlx/dsh/gmlx.cordis.yml`, through `--patch`. |

With `--container`, the third step starts the tool in an Apple container
instead, a Linux virtual machine that sees only the folders you share. The
configuration then goes into the tool's
[private home](glossary.md#private-home) rather than the places the table
lists, as [Container mode](launch-container.md) describes.

`gmlx launch <name>` also runs a program of your own against the server. A
[custom agent](launch-agents.md) is defined under `launch.agents`, runs
only in a container, and gets the server's address, a key and a model in
environment variables instead of a written configuration.

`--config-path` moves the written configuration to the path you give,
which takes the place of the location that the table lists for the client.
It names a file for opencode and dsh, and a directory for pi, omp and
aichat. For goose it names the `config.yaml`, for elia the
`XDG_CONFIG_HOME` directory, and for Open WebUI the data directory.

hermes refuses `--config-path` before the server starts, because hermes
reads only `$HERMES_HOME/config.yaml`, so set `HERMES_HOME` to use another
folder. Container mode refuses it for every client, because the
configuration goes into the private home.

## Starting the server

When no server answers, `launch` starts one in the background from the
first configuration file in the
[places gmlx looks](config.md#where-gmlx-looks), and it waits until the
server responds.

On a Mac desktop, starting the server also opens the
[menu bar app](menubar.md), unless
[`server.menubar`](config.md#servermenubar) is `false`. With no configuration
file anywhere, `launch` says to run `gmlx init` and exits with the
configuration error code in [Exit codes](cli.md#exit-codes).

The wait has no fixed limit, and only the server process exiting counts as
a failure. Ctrl-C stops the wait, and the server keeps starting in the
background. `--start-timeout SECONDS` limits the wait for scripts, and
`--no-start` turns off starting entirely, so the command fails when no
server runs.

`--base-url URL` names a server explicitly. `launch` then never starts one
and reads no configuration file.

The first request to a model that is not loaded waits for the load. Which
models the server loads at start is under
[Memory and residency](config.md#memory-and-residency).

## Choosing the model

`--model ID` selects the model that the tool uses. Without it, the tool gets
the model that the server marks as its default, which
[`server.defaults.model`](config.md#serverdefaultsmodel) sets. An id with a
profile, such as `--model qwen3.8-27b-ud-q6@coding`, applies that profile
to every request from the tool.

`launch` checks the id against the models that the server lists. In
container mode it checks before it builds or pulls the image. When the
server has no models, `launch` stops and says to download one with
`gmlx pull`.

The `claude-code`, `dsh`, `goose` and `hermes` clients need a default
model, so `launch` refuses them when the server marks none and `--model`
names none. `dsh` also takes the server's only chat model.

With `--model`, `launch` also asks the server to load the model and keep it
loaded through the idle timeout, so that the model is not unloaded between
the turns of a long session. The keep lasts after the tool exits,
until `POST /unload` or a request to `POST /v1/keep` with
`{"keep": false}` releases it, or the server restarts. A kept model can
still be unloaded when the memory budget needs the room, as
[Memory and residency](config.md#memory-and-residency) describes. `--no-keep`
skips the request, and so does `--config-only`.

## Authentication

A server with an [API key](config.md#serverapi_key) refuses a launch
without the key before the tool starts, and `launch` says to pass
`--api-key`. Without that flag, `launch` takes the key from a configuration
file.

For a server that is running, `launch` reads the config file that the
server started with, such as the file that `gmlx serve --config FILE`
names. Otherwise it reads the first file in the
[places gmlx looks](config.md#where-gmlx-looks). A server that `launch`
starts gets the key of the file it starts from. A server named with
`--base-url` gets no key from a file.

Each tool gets the key in its own setting:

| Client | Where the key goes |
|--------|--------------------|
| `opencode` | The key goes in `options.apiKey` in the injected file. |
| `pi` | The key goes in `apiKey` in the merged provider. |
| `omp` | The key goes nowhere, because omp has no setting for it. `launch` prints a note, and you set up omp's authentication yourself. |
| `hermes` | The key goes in `model.api_key` and `providers.custom.api_key` in the merged file. |
| `goose` | The key goes in `OPENAI_API_KEY` in the environment only, never in the file. |
| `claude-code` | The key goes in `ANTHROPIC_AUTH_TOKEN` in the environment. |
| `aichat`, `elia` | The key goes in `api_key` in the injected file. |
| `open-webui` | The key goes in `OPENAI_API_KEY` in the environment only. |
| `dsh` | The key goes in `GMLX_API_KEY` in the environment only. |

Without a key on the server, a tool that needs a key still gets a
placeholder key, because it refuses to run without one. The opencode, omp
and aichat configurations get no key. In container mode with a server on
the Mac, a tool never gets the server's key, as
[What the client reaches on the server](container-security.md#what-the-client-reaches-on-the-server)
describes.

## The clients

### claude-code

Claude Code uses the server's Anthropic API. `launch` sets
`ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL` and
`ANTHROPIC_AUTH_TOKEN`, and it removes an inherited `ANTHROPIC_API_KEY` so
that the token takes effect. It does not change `~/.claude`.

`launch` also sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the model's context
window from the server's model list, so that Claude Code compacts a
conversation before it outgrows the model. A whole number of your own from
1 to that window stays. A larger value, or one that is not a whole number,
gets the window, and `launch` prints a line that names the window. The line
repeats your value when it is at most 20 printable characters.

For a `--model id@profile` that the server does not list, `launch` uses the
base model's window only when no profile in the chain sets `load` or
`cache`, since those can change the window. That check also covers the
profile that the `@profile` replaces, such as the model's own `profile`,
and the model's `profiles` tweaks for both.

When `launch` cannot read the server's config, only a built-in intent such
as `coding` keeps the window. That happens for a server that `gmlx serve -f`
runs, a server named by `--base-url`, or a server that an older gmlx started
without a record of its config's full path. For any other profile,
`launch` sets no window and prints a line that says why, so set the
variable yourself to the profile's window.

In container mode, a value of your own comes from
[`launch.container.env`](config.md#launchcontainerenv). The
[dry run](launch-container.md#the-dry-run) shows the variable's value in a
line of its own, because the command it prints names the client's
variables without their values.

Its system prompt is very long, and it often rewrites the start of its
requests, so processing the prompt takes most of a turn's time.
Turn on the [prompt cache](config.md#prompt-cache), and prefer a model and
a Mac with fast prefill. `launch` prints a note when the configuration of
the running server leaves the cache off for the model.

### opencode, pi and omp

These three coding agents take the default model in different places.
opencode takes it in the `model` key of the injected file, pi as
`defaultProvider` and `defaultModel` in its merged files, and omp as
`modelRoles.default`. For pi, `launch` also sets each model's context window
and output limit from the server's model list.

### hermes

`launch` merges the gmlx provider into hermes's own `config.yaml`, in
`$HERMES_HOME` or `~/.hermes`, and keeps every other setting. hermes reads
its settings from no other file, and it sends an API key to a local server
only from that file. A launch that would change nothing writes nothing.

Before it changes the file, `launch` copies it to a new
`config.yaml.gmlx-<date>-<time>`, in UTC, with `-<n>` added when that name
is taken, and prints the copy's path. The copy goes beside the file that
`config.yaml` leads to when it is a link. `launch` keeps the three copies
written last and deletes only older files named that way. The rewritten file
keeps its settings, but not its comments or layout, which the copy keeps.

hermes refuses a model with less than 64K tokens of context. When the
server reports a smaller context window for the default model, `launch`
prints a note, and `--model` then selects a model with more.

### goose

The goose file gets `GOOSE_PROVIDER`, `GOOSE_MODEL`, `OPENAI_HOST` and
`OPENAI_BASE_PATH`, which `launch` also sets, with `OPENAI_API_KEY`, in
the environment, where they take precedence. `launch` then runs
`goose session`. A later `goose` with no launch still finds the server
through the file.

### aichat

Each served model is marked as supporting function calls, so aichat's
tools and agents work with the server. Running tools also needs aichat's
`llm-functions`.

### elia

elia lists each served model as an OpenAI-compatible model, and `launch`
starts it on the selected model. elia 1.x or newer is required. An older
elia starts but lists no local models, so upgrade it with
`uv tool upgrade elia-chat`.

### open-webui

Open WebUI is a chat app that runs its own web server, so this launch
starts a second service. Install it first with
`uv tool install --python 3.12 open-webui`, because it needs Python 3.11
or 3.12.

`launch` sets the server address and key, turns off Open WebUI's Ollama
connection, and sets its data directory. The app runs on port 3000, or on
3001 when the gmlx server uses 3000, and `launch` prints its address. In
container mode, it opens at `http://[::1]:<port>` with a port from 3100 to
3199, as [Browser apps](launch-container.md#browser-apps) describes.

The app listens on `127.0.0.1` only, because `launch` runs
`open-webui serve --host 127.0.0.1`. `gmlx launch open-webui -- --host 0.0.0.0`
makes it listen on every address, where other computers and containers
reach it, as [Access you turn on](container-security.md#access-you-turn-on)
describes.

`launch` also sets `CORS_ALLOW_ORIGIN` to the app's own address, which is
`http://localhost:<port>;http://127.0.0.1:<port>`, or `http://[::1]:<port>`
in container mode. Other web pages then cannot read Open WebUI's answers or
send it JSON calls, such as the call that makes a Function, which runs
Python. A plain form or link from a page of the same site still reaches it,
but that page cannot read the answer.

Open WebUI's live updates accept only these addresses too. To open the app
at another address, such as through a reverse proxy, export
`CORS_ALLOW_ORIGIN` with every address of the app, split by `;`, and your
value replaces launch's value. In container mode, set it as
`CORS_ALLOW_ORIGIN=...` in the [`env`](config.md#launchcontainerenv) of
`launch.container.clients.open-webui` instead.

The app keeps its chat history in `~/.open-webui`, or in the folder that
`--config-path` names. In container mode that folder is `~/.open-webui` in
the [private home](glossary.md#private-home), so the history of the app on
the Mac does not appear there.

Open WebUI asks for a login unless `WEBUI_AUTH=false` is set before the
first account exists. On the first launch with a new data directory,
`launch` prints how to set it: in the environment of
`gmlx launch open-webui` on the Mac, or in the
[`env`](config.md#launchcontainerenv) of
`launch.container.clients.open-webui` in container mode.

Open WebUI gets a feature for each service that the server runs, as
[Speech, embeddings and rerank](services.md) describes:

| Server runs | Open WebUI gets |
|-------------|-----------------|
| Chat models only | It gets chat. Its document embedder points at the server, so it starts without downloading one. |
| `embeddings` | It gets document search, as [RAG pipelines](rag.md) describes. |
| `rerank` | It gets hybrid search with the server's reranker at `/v1/rerank`. |
| `stt` | It gets speech input through `/v1/audio/transcriptions`. |
| `tts` | It gets spoken replies through `/v1/audio/speech`. |

### dsh

DeepSeek Harness is an agent app that runs in the browser. Install version
0.1.7 or newer with `npm install -g @deepseek-ai/dsh@next`. `launch`
refuses an older version before it starts the server, and prints that
command.

`launch` runs dsh with a profile of its own, `gmlx`, under
`$DSH_HOME/profiles/`, where `DSH_HOME` defaults to `~/.dsh`. The first
launch creates the profile from dsh's `web` template, and later launches
reuse it. Your other dsh profiles are not changed. A `gmlx` folder there
without a `package.json` is refused, so remove or rename it first.

The providers, the default model and the title and compaction settings go
in `~/.config/gmlx/dsh/gmlx.cordis.yml`, which `launch` passes to dsh with
`--patch`. The settings in that file override dsh's own settings, and dsh
never saves the file, so the web app cannot save a different default model or an edit to the
gmlx providers. Run `launch` again with `--model` to change the default.

Two entries in the file point at the server. Under `gmlx (local)`, the
server and its profiles decide whether a model thinks. Under
`gmlx (thinking off)`, the same models answer without thinking, and dsh
writes its session titles there with the default model.

Port 3080 serves the web app, or 3081 when the gmlx server uses 3080, and
the launch opens a browser. The app starts in
`~/Documents/deepseek-harness/default-workspace`, not in the folder you
launch from, and Add workspace opens a project folder. In container mode, it
opens at `http://[::1]:<port>` with a port from 3100 to 3199, as
[Browser apps](launch-container.md#browser-apps) describes.

The dsh web app compacts a conversation by itself only when the model's
context is large enough for dsh's default headroom, and `launch` prints a
note when it is not. Otherwise a conversation still compacts when the
server reports that a request no longer fits, as
[Limits and back-pressure](api.md#limits-and-back-pressure) describes. The
headless and acp profiles use compaction settings from the file, sized to
each model's context.

dsh's `headless` profile works with the same file. This command answers
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
that you submit, and `DSH_TELEMETRY_MODE=DISABLED` stops dsh from
sending the conversation.
