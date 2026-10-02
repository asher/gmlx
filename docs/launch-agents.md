# Custom agents

A custom agent is a program of your own that `gmlx launch` runs in an Apple
container against your server, defined under `launch.agents` in your gmlx
config. This page covers defining one, what it gets from launch, how its
dependencies are installed, and how its sessions and data behave.

An agent runs only in a container, as [Container mode](launch-container.md)
describes for the built-in clients, and launch writes no configuration file
for it. It gets the server's address, a key and a model in environment
variables, and the rest is your program.

- [Defining an agent](#defining-an-agent)
- [What the agent gets](#what-the-agent-gets)
- [Dependencies at run time](#dependencies-at-run-time)
- [Your own image](#your-own-image)
- [A browser interface](#a-browser-interface)
- [Sessions and data](#sessions-and-data)
- [Security](#security)
- [A LangChain recipe](#a-langchain-recipe)

## Defining an agent

Each key under `launch.agents` names an agent, and `gmlx launch <name>`
runs it. The smallest agent names a Python project in the current folder
and the command that starts it:

```yaml
launch:
  agents:
    research-bot:
      runtime: python
      command: [python, -m, research_bot]
```

```sh
cd ~/src/research-bot
gmlx launch research-bot
```

`runtime: python` installs the project's dependencies with uv when the
session starts, as [Dependencies at run time](#dependencies-at-run-time)
describes. An agent that brings its own image sets
[`image`](config.md#launchcontainerclientsimage) or
[`build`](config.md#launchcontainerclientsbuild) instead, as
[Your own image](#your-own-image) describes. `command` is always required.

A name starts with a lowercase letter and holds lowercase letters, digits
and single `-` or `_` separators, at most 32 characters. It cannot be the
name of a built-in client or `menubar`. The agent takes every key of
[`launch.container`](config.md#launch) apart from `enabled` and `packages`,
such as `mounts`, `volumes`, `network` or `env`, and the values under
`launch.container` apply to it as they do to a client. Five keys exist
only for an agent, `runtime`, `source`, `api`, `model` and `web_port`, and
the [configuration reference](config.md#launchagents) lists them with
`command`.

Launch reads agents only from the config file in your home folder, as
[Launch](config.md#launch) explains, and `gmlx launch --help` lists the
configured agents. `--no-container` does not apply to an agent, and
neither does `--provider-id`, since launch writes no provider entry for
it. Every other launch flag works as it does for a client, and the
arguments after `--` follow the command.

## What the agent gets

Launch sets these variables in the container. An
[`env`](config.md#launchcontainerenv) entry `NAME=VALUE` replaces launch's
value of a server, key or model variable, and a bare `NAME` entry does not.
Launch's `HOST`, `PORT` and `UV_` values always apply, and an `env` entry
for one of them prints a line that says it has no effect:

| Variable | Value |
|----------|-------|
| `GMLX_BASE_URL` | The server's `/v1` address, such as `http://127.0.0.1:8080/v1`. |
| `GMLX_API_KEY` | The key the agent sends. In a session it is the placeholder `gmlx-container-session`. |
| `GMLX_MODEL` | The model, from `--model`, then `model`, then the server's default. Absent when none of them names one. |
| `OPENAI_BASE_URL`, `OPENAI_API_BASE` | The same as `GMLX_BASE_URL`, with `api: openai`, the default. |
| `OPENAI_API_KEY` | The same as `GMLX_API_KEY`, with `api: openai`. |
| `ANTHROPIC_BASE_URL` | The server's root address, without `/v1`, with `api: anthropic`. |
| `ANTHROPIC_API_KEY` | The same as `GMLX_API_KEY`, with `api: anthropic`. |
| `ANTHROPIC_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL` | The same as `GMLX_MODEL`, with `api: anthropic`. |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | The model's context window, with `api: anthropic`, by the rule in [claude-code](launch.md#claude-code). |
| `HOST`, `PORT` | `127.0.0.1` and the `web_port`, for an agent with a [browser interface](#a-browser-interface). |

[`api`](config.md#launchagentsapi) chooses the second group. `openai` suits
the OpenAI SDK, langchain-openai and most frameworks, `anthropic` the
Anthropic SDKs, langchain-anthropic and the Claude Agent SDK, and `none` a
program that reads only the `GMLX_` variables. The Anthropic SDKs send
`ANTHROPIC_API_KEY` as `x-api-key`, which the server accepts.

The agent reaches the server through a
[session socket](glossary.md#session-socket), so it sees only the
inference routes, and the served assistants its
[`assistants`](config.md#launchcontainerclientsassistants) list names, as
[What the client reaches on the server](container-security.md#what-the-client-reaches-on-the-server)
describes. A session sends at most 16 requests at once, as
[Limits](container-security.md#limits) says, so keep a framework's
concurrency at 16 or fewer. When no model is chosen, launch prints a line
that says the agent gets no `GMLX_MODEL`, and the agent then has to name
one itself.

## Dependencies at run time

With [`runtime: python`](config.md#launchagentsruntime), the agent runs in
an image that gmlx builds from its shipped recipe, Debian with Python 3.11
and uv. Each time a session starts, `uv sync` brings the environment in
line with the project's `uv.lock`, and the command then runs in that
environment, with the environment's `bin` folder first on `PATH`. Both
`[python, -m, research_bot]` and a script from the project's
`[project.scripts]`, such as `[research-bot]`, work, and a `.py` file runs
with the environment's Python. A command that is not on `PATH` in the
container stops the launch with a line that names it. An `env` entry
`UV_NO_SYNC=1` skips the sync, as it does for `uv run`.

The first launch of any runtime agent builds that image once, which
downloads uv, about 19 MB, and takes about three minutes. On a Mac where no
client image was built yet, the build also downloads the Node base image of
about 80 MB first, as [The first launch](launch-container.md#the-first-launch)
describes. The first launch of an agent in each project folder then
installs its dependencies, which takes about a minute for a LangChain
project, and the lines that uv prints follow launch's own. Later launches
from the same folder find the environment in place, and uv prints two lines
as it checks it. A project whose `requires-python` Debian's Python does not
meet gets a Python that uv downloads on that first launch, 30 to 90 MB, and
keeps for later launches. `uv init` writes the version of the Python on your Mac into
`requires-python`, so most new projects get that download.

### The source folder

The project is the current folder, which launch shares read-write by
default, so uv can update `uv.lock` there as you develop. An agent that
lives elsewhere names its folder with
[`source`](config.md#launchagentssource), as a full path or one that starts
with `~`:

```yaml
launch:
  agents:
    reviewer:
      runtime: python
      source: ~/src/reviewer
      command: [reviewer]
```

A launch from a folder that holds no `pyproject.toml`, with no `source`,
has no project to install and fails at once with uv's message
``No `pyproject.toml` found in current directory or any parent directory``.
Launch from the project folder, or set `source`. A single script with
inline metadata, which the paragraphs below describe, needs no project.

A relative path as the command, such as `agent.py` or `bin/start`, is
looked up in the working folder and then in the project folder, which is
the `source` when one is set. So a script that lives in the source runs
from any folder.

Launch shares a `source` outside the shared folders read-only at its own
path, and prints a line that names it as the source folder. uv then uses
the `uv.lock` in it as it is, and a lock that is missing or out of date
stops the launch with uv's own message, as
[uv says the lockfile needs to be updated](troubleshooting.md#uv-says-the-lockfile-needs-to-be-updated)
describes. A `source` inside a folder the session already shares takes that
share's mode and gets no share of its own. A `source` that is a symbolic
link, or that lies in a folder launch never shares, gets the same refusal
as a [share](launch-container.md#shares) would.

A project run from another folder has two rules. It must be a package, as
`uv init --package` makes it, with a `[build-system]` table, so that its
module imports from any working folder. A plain `uv init` project is not
installed, and `python -m research_bot` from another folder fails with
`No module named`. Its build backend must write nothing into the source
when it builds the project. hatchling and uv_build write nothing.
setuptools writes a `.egg-info` folder into the source and fails against a
read-only share with `Read-only file system`.

A single script with inline metadata, the `# /// script` block of PEP 723,
runs with the script as the command, as `[agent.py]`. uv installs the
dependencies that the block names into an environment of the script's own,
and the script runs there. The form `[python, agent.py]` ignores the block.
Under a read-only source, a script without a lockfile installs anyway, and
launch prints a line that says so. `uv lock --script agent.py` writes the
lockfile beside the script.

Refresh a lock in the container rather than on the Mac. A project with
dynamic metadata runs its build backend to lock, which can be code from the
folder the agent edits. Launch the agent once from its source folder, where
the source is the read-write working folder and uv updates the lock, or
open `--shell` from that folder and run `uv lock` there.

### The dependency volume

uv keeps the environment, its cache and any downloaded Python on a volume
at `/opt/agent` in the container, named `gmlx-agent-<name>-uv`, or
`gmlx-agent-<name>-uv-<8 hex digits>` for a project other than `default`,
as [Volumes](launch-container.md#volumes) names a client's. Launch creates
it with the default size of 32G and the `gmlx.launch=1` label, and one
session uses it at a time.

The shared current folder chooses the project, as
[Projects and sessions](launch-container.md#projects-and-sessions)
describes, so an agent with a `source` gets a volume, and installs again,
in each folder you launch it from.

To keep one volume, launch with `--no-mount-cwd` from a folder that no
`--mount` or `mounts` entry holds. The session then belongs to the
`default` project. From outside the source, the agent starts in that
project's private home without the current folder. From inside the source,
it starts in the source, which stays read-only. Use this for an agent that
works only from its source.

The launch from the source folder without the flag, which
[The source folder](#the-source-folder) gives for a lock refresh, belongs
to the source folder's own project. It gets a volume of its own and
installs the dependencies once more.

A [`volumes`](config.md#launchcontainervolumes) entry of the agent at
`/opt/agent` takes the place of that volume, which is how you set its size
or its name. No share or `--mount` may use `/opt/agent` or a path inside
it, and no global `volumes` entry may either. The environment is writable
by the agent and stays from one launch to the next, so a package the agent
changes there stays changed until the volume is deleted, which
[Sessions and data](#sessions-and-data) covers.

The environment and the cache share one disk, so uv links files between
them instead of copying, and the many small files of an environment live
on an ext4 disk rather than in a share. `--shell` gets the same `UV_`
variables as the agent, so `uv run python` in the shell uses the agent's
environment from any folder.

### Offline launches

Under [`network: none`](config.md#launchcontainernetwork), launch sets
`UV_OFFLINE=1`, so uv starts from the synced environment without a network
and never waits for one. An environment that was never synced then fails
at once with uv's message. Launch the agent once with the network in each
project folder before you turn it off, as
[An agent's first launch fails under network none](troubleshooting.md#an-agents-first-launch-fails-under-network-none)
describes. The agent still reaches the server through its socket.

## Your own image

[`image`](config.md#launchcontainerclientsimage) and
[`build`](config.md#launchcontainerclientsbuild) work as they do for a
client. Launch pulls or builds the image, builds it again when its
Containerfile or context changes, runs it by digest and checks the command
once in it, as [Custom container images](container-images.md) describes.
An agent's Containerfile may start from any client's `:base` image, such
as `gmlx.invalid/launch-claude-code:base` to build on the Claude Code
image.

`runtime: python` with `image` or `build` runs the same uv steps in that
image, which must provide `uv` and `sh`, and `grep` for a `.py` command.
That is how a runtime agent gets system packages. Start from the runtime
base and install them:

```dockerfile
FROM gmlx.invalid/launch-runtime-python:base
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
```

```yaml
launch:
  agents:
    transcriber:
      runtime: python
      build: ~/containers/transcriber
      command: [transcriber]
```

An image without `uv` or `sh` stops the launch at the one-time check, with
a message that names the runtime key. Without `runtime`, the command runs as written
in the image, and `command: image` runs the image's ENTRYPOINT and CMD, as
[The command that runs](launch-container.md#the-command-that-runs)
describes.

The build-folder rules apply to agents in both directions. A launch of
any target refuses a read-write share that holds or lies in an agent's
`build` folder, and an agent launch refuses one that holds a client's.
Launch also refuses to build from a folder that a session shares
read-write now or shared read-write earlier. So build a packaged agent
from a folder that no launch has shared read-write, such as a fresh clone
that you have reviewed.

## A browser interface

An agent with [`web_port`](config.md#launchagentsweb_port) is a browser
app. Launch forwards that port on the Mac to the container, opens
`http://127.0.0.1:<web_port>/` when the app answers, and prints the
address instead with
[`open_browser: false`](config.md#launchcontaineropen_browser), as
[Browser apps](launch-container.md#browser-apps) describes:

```yaml
launch:
  agents:
    dashboard:
      runtime: python
      command: [streamlit, run, app.py, --server.address, "127.0.0.1",
                --server.port, "8501", --server.headless, "true"]
      web_port: 8501
```

The command must listen on `127.0.0.1` at `web_port` inside the container.
Launch sets `HOST` and `PORT` for programs that read them, and a command
list runs with no shell, so a `$PORT` in the list stays literal. Write the
port itself, as the example does.

Launch refuses a `web_port` equal to the gmlx server's port, and a
`forward` entry equal to the `web_port`. One session of the agent runs at
a time, since the port is one, and a launch from another project names the
running session. A running session of any other target that holds the
same port stops the launch with its name too, and the fix is to end it or
to give one of them another port.

## Sessions and data

Agents follow the rules of the clients in
[Projects and sessions](launch-container.md#projects-and-sessions). The
shared current folder keys the project, one session runs per project, a
second launch from the same project joins it, and `--shell` opens a shell
in it. The container is named `gmlx-agent-<name>-` followed by six hex
digits, and the session's log is
`~/.cache/gmlx/launch/last-agent-<name>-<project>.log`.

Each project gets a [private home](glossary.md#private-home) at
`~/.local/share/gmlx/launch/agent-<name>/projects/<project>/home`, which
starts empty, since launch writes no configuration for an agent. The
dependency volume is named per project, as
[The dependency volume](#the-dependency-volume) says, and an image built
from `build` is tagged `gmlx.invalid/launch-agent-<name>-build`.

`gmlx launch <name> --remove-home` from the project folder asks one
question that names the project's private home and, for a runtime agent,
its dependency volume with the space it takes on the Mac. A yes removes
the home and deletes the volume. The question names only the volume launch
created, never one you configured at `/opt/agent`. When the home is gone
and the volume remains, the question names the volume alone. Without a
terminal, the message gives the `rm -rf` and `container volume delete`
commands instead. [Removing container data](launch-container.md#removing-container-data)
lists the other data, and `gmlx doctor` reports agent homes under the
agent's name, with the images that no setting uses.

## Security

An agent runs inside the boundary of every container session, which is
its virtual machine, its shares, its private home, its volumes, the session
socket and the forwarded ports. [Container security](container-security.md)
describes what still leads back to the Mac, and these points add to it for
agents:

- The run-time install runs every dependency's code inside that boundary.
  Launch never builds an image from the agent's project folder, and the
  build-folder rules cover agents in both directions.
- Network is all or nothing. `network: default` gives the agent the
  internet, and `network: none` leaves it the server and the forwarded
  ports.
- The agent, and every package it installs, can read each variable you
  pass through `env`, such as a search API key. Pass only the keys the
  agent needs.
- An `env` entry `OPENAI_API_KEY=...` replaces the session placeholder,
  and the agent then holds that key.
- A read-only `source` protects the source folder only. The environment
  on the dependency volume is writable, stays across launches and
  `--rebuild`, and runs at the next launch. Delete the volume with
  `--remove-home` after an agent you do not trust has run.
- When the source is the working folder, the agent can edit
  `pyproject.toml`, `uv.lock`, `uv.toml`, `.python-version` and build
  backend code such as `setup.py`, which run on the Mac when you run the
  project or `uv lock` there. Read what changed first.

## A LangChain recipe

The recipe runs a LangChain agent on your server in five steps.

1. Make a packaged project and add the libraries:

   ```sh
   uv init --package research-bot
   cd research-bot
   uv add langchain langchain-openai
   ```

2. Write the agent in `src/research_bot/__init__.py`. It reads the model
   from `GMLX_MODEL`, and langchain-openai reads the address and the key
   from `OPENAI_API_BASE` and `OPENAI_API_KEY`. Leave the sampling to the
   server's [family defaults](family-defaults.md), since a fixed
   `temperature=0` makes a thinking model repeat itself, and cap the
   answer with `max_tokens`, which a thinking model spends on its
   reasoning first:

   ```python
   import os

   from langchain_core.messages import HumanMessage
   from langchain_openai import ChatOpenAI


   def main():
       llm = ChatOpenAI(model=os.environ["GMLX_MODEL"], max_tokens=2048)
       reply = llm.invoke([HumanMessage("Name three uses of a local model.")])
       print(reply.content)
   ```

3. Define the agent in `~/.config/gmlx/gmlx.yaml`:

   ```yaml
   launch:
     agents:
       research-bot:
         runtime: python
         command: [research-bot]
   ```

4. Launch it from the project folder. The first launch builds the runtime
   image once and installs the project, and later launches start at once.
   The `@instruct` intent turns thinking off for a Qwen model, so the
   answer comes in a few seconds rather than after its reasoning:

   ```sh
   gmlx launch research-bot --model qwen3.8-27b-ud-q6@instruct
   ```

5. Read [What the agent gets](#what-the-agent-gets) for the other
   variables, such as the Anthropic set for `api: anthropic`.
