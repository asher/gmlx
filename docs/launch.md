# Agents and chat apps

`gmlx launch` connects a coding agent or a chat app to your gmlx server. It
sets the tool up to use the server, starts the server if it is not running,
and runs the tool:

```sh
gmlx launch pi --model qwen3.8-27b-ud-q6    # the pi coding agent on a local model
gmlx launch opencode                         # the server's default model
gmlx launch open-webui                       # a chat app in the browser
```

gmlx does not install the tools on your Mac. When a tool is missing,
`launch` prints its install command. To skip the install, run the tool in a
[container](launch-container.md), where gmlx installs it for you. The flags
are in the [CLI reference](cli.md#gmlx-launch).

- [How a launch works](#how-a-launch-works)
- [Containers and custom agents](#containers-and-custom-agents)
- [Starting the server](#starting-the-server)
- [Choosing the model](#choosing-the-model)
- [Authentication](#authentication)
- [The clients](#the-clients)

## How a launch works

`launch` asks the server for its models, writes the tool's settings, and
then runs the tool, already connected. Each tool keeps its settings in a
different place:

| Client | What it is | Where `launch` puts the settings |
|--------|------------|----------------------------------|
| `claude-code` | Anthropic's coding agent | `ANTHROPIC_*` environment variables |
| `opencode` | A coding agent | `~/.config/gmlx/opencode.json`, through `OPENCODE_CONFIG` |
| `pi` | A coding agent | A provider added to `~/.pi/agent/models.json` and `settings.json` |
| `omp` | oh-my-pi, a coding agent | A provider added to `~/.omp/agent/models.yml` and `config.yml` |
| `hermes` | NousResearch's hermes-agent | A provider added to `~/.hermes/config.yaml` |
| `goose` | Block's agent runtime | `~/.config/goose/config.yaml` and environment variables |
| `aichat` | A terminal chat client with tools | `~/.config/gmlx/aichat/`, through `AICHAT_CONFIG_DIR` |
| `elia` | A terminal chat app | `~/.config/gmlx/elia-xdg`, through `XDG_CONFIG_HOME` |
| `open-webui` | A chat app in the browser | `OPENAI_API_BASE_URL` and related variables |
| `dsh` | DeepSeek Harness, an agent app in the browser | `~/.config/gmlx/dsh/gmlx.cordis.yml`, through `--patch` |

Where `launch` adds a provider to a tool's own file, it keeps everything
else in the file. It refuses to change a file it cannot parse, and never
leaves a file half written.

A new file gets mode 600, so only you can read the key in it, and a
rewritten file keeps its mode. When a file is a symbolic link into your home
folder, such as a link into a dotfiles repository, `launch` writes through
the link. It refuses a link that points outside your home folder.

`--config-path PATH` writes the settings somewhere else. What the path names
depends on the client:

| Client | `--config-path` names |
|--------|-----------------------|
| `opencode`, `dsh` | The settings file |
| `pi`, `omp`, `aichat` | The settings folder |
| `goose` | goose's `config.yaml` |
| `elia` | The `XDG_CONFIG_HOME` folder |
| `open-webui` | Open WebUI's data folder, in place of `~/.open-webui` |
| `hermes` | Refused. Set `HERMES_HOME` to use another folder. |
| `claude-code` | Nothing, because Claude Code gets only environment variables |

Container mode refuses `--config-path`, because the settings go into the
tool's private home.

To see what a launch would do without running the tool, add
`--config-only`. It writes the settings and prints the command.

## Containers and custom agents

Add `--container` to run the tool in an Apple container that sees only the
project folder. The settings then go into the tool's private home in the
container, and your own files are not touched.

- [Container mode](launch-container.md) covers the first launch and what
  works in a container.
- [Custom agents](launch-agents.md) runs a program of your own in a
  container against the server.
- [Container security](container-security.md) covers running a tool on
  code you do not trust.

## Starting the server

When no server answers, `launch` starts one in the background from your
[config file](config.md#where-gmlx-looks), and waits until it is ready. On a
Mac desktop, this also opens the [menu bar app](menubar.md). Ctrl-C stops
the wait, and the server keeps starting.

`--no-start` fails instead of starting a server, and `--base-url URL`
connects to a server you name, such as one on another Mac.

## Choosing the model

`--model` picks the model. Without it, the tool gets the server's default
model, which [`server.defaults.model`](config.md#serverdefaultsmodel) sets.
A profile works too, as in `--model qwen3.8-27b-ud-q6@coding`.

With `--model`, the server also keeps the model loaded through its idle
timeout, so it is not unloaded between the turns of a long session. The
model stays kept after the tool exits, until `POST /unload`, a
`POST /v1/keep` request with `{"keep": false}`, or a server restart.
`--no-keep` turns this off, and `--config-only` never keeps a model.

`claude-code`, `dsh`, `goose` and `hermes` need a model, so they need
`--model` when the server has no default.

## Authentication

When the server has an [API key](config.md#serverapi_key), `launch` reads it
from the server's config file and gives it to the tool. `--api-key` passes
another one. A tool that needs a key gets a placeholder when the server has
none.

The key goes into the tool's own key setting, or into an environment
variable for `claude-code` (`ANTHROPIC_AUTH_TOKEN`), `goose`, `open-webui`
and `dsh`. omp has no key setting, so `launch` prints a note and you set it
up in omp yourself.

In a container, a tool never gets the server's key, as
[What the client reaches on the server](container-security.md#what-the-client-reaches-on-the-server)
explains.

## The clients

### claude-code

Claude Code uses the server's Anthropic API. `launch` sets
`ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL` and
`ANTHROPIC_AUTH_TOKEN`, and leaves `~/.claude` alone. Arguments after `--`
go to Claude Code, so `gmlx launch claude-code -- --continue` resumes your
last conversation.

`launch` also sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the model's context
window, so Claude Code compacts a conversation before it outgrows the model.
A smaller value that you set yourself stays.

Claude Code sends a very long prompt and often changes its start, so most of
a turn goes to reading the prompt. Turn on the
[prompt cache](config.md#prompt-cache), and prefer a model with fast
prefill. `launch` prints a note when the cache is off for the model.

### opencode, pi and omp

These coding agents get the server's models with the chosen model as the
default. pi also gets each model's context window and output limit.

### hermes

`launch` adds the gmlx provider to hermes's `config.yaml`, in `$HERMES_HOME`
or `~/.hermes`. Before it changes the file, it saves a copy beside it,
named `config.yaml.gmlx-<date>-<time>`, and prints its path. It keeps the
last three copies. The rewritten file keeps your settings but not your
comments, which the copy keeps.

hermes refuses a model with less than 64K tokens of context, and `launch`
warns when the model is smaller.

### goose

`launch` writes the server into goose's `config.yaml` and runs
`goose session`. A later `goose` without `launch` still finds the server.

### aichat

aichat gets every served model with function calls turned on, so its tools
and agents work. Running tools also needs aichat's `llm-functions`.

### elia

elia lists each served model, and starts on the chosen one. It needs elia
1.x or newer. An older elia starts with no local models, so upgrade it with
`uv tool upgrade elia-chat`.

### open-webui

Open WebUI is a chat app with its own web server. Install it with
`uv tool install --python 3.12 open-webui`, because it needs Python 3.11 or
3.12. `launch` starts it on port 3000, or 3001 when gmlx uses 3000, and
prints its address. Your chats are kept in `~/.open-webui`.

Open WebUI asks you to create an account unless `WEBUI_AUTH=false` is set
before the first account exists. The first launch prints how to set it.

The app listens only on `127.0.0.1`, and accepts requests only from its own
address. To open it at another address, such as behind a reverse proxy,
export `CORS_ALLOW_ORIGIN` with every address of the app, separated by `;`.

Open WebUI gets a feature for each service the gmlx server runs, as
[Speech, embeddings and rerank](services.md) describes:

| Server runs | Open WebUI gets |
|-------------|-----------------|
| Chat models only | Chat. Its document embedder points at the server, so it starts without downloading one. |
| `embeddings` | Document search, as [RAG pipelines](rag.md) describes. |
| `rerank` | Hybrid search with the server's reranker. |
| `stt` | Speech input. |
| `tts` | Spoken replies. |

### dsh

DeepSeek Harness is an agent app that runs in the browser. Install version
0.1.7 or newer with `npm install -g @deepseek-ai/dsh@next`.

`launch` runs dsh with its own profile, `gmlx`, which the first launch makes
from dsh's `web` template. Your other dsh profiles are not changed. The app
opens on port 3080, or 3081 when gmlx uses 3080. It starts in
`~/Documents/deepseek-harness/default-workspace`, so use Add workspace to
open a project folder.

The gmlx providers and the default model live in
`~/.config/gmlx/dsh/gmlx.cordis.yml`, which dsh reads but never saves. To
change the default model, launch again with `--model`. The provider
`gmlx (thinking off)` serves the same models without thinking.

The same file works with dsh's other profiles. This command runs one task in
the current folder and exits:

```sh
GMLX_API_KEY=gmlx dsh --profile headless \
  --patch ~/.config/gmlx/dsh/gmlx.cordis.yml "run the tests"
```

`--dsh-profile NAME` launches another profile. The `acp`, `sdk` and
`sdk-minimal` profiles serve another program, so `launch` only prints their
command, with `--config-only`.

dsh's `web_search` tool uses DeepSeek's search service and needs
`DEEPSEEK_API_KEY`. Set `DSH_TELEMETRY_MODE=DISABLED` to stop dsh from
sending feedback conversations to DeepSeek.
