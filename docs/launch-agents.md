# Custom agents

A custom agent is your own program that `gmlx launch` runs in an Apple
container against your server, defined under `launch.agents` in your gmlx
config. This page covers defining one, what it gets from `launch`, how its
dependencies are installed and how its sessions behave, and it ends with
four complete agents to start from.

An agent runs only in a container, as [Container mode](launch-container.md)
describes for the built-in clients, and `launch` writes no configuration
file for it. It gets the server's address, a key and a model in environment
variables, and the rest is your program.

- [Defining an agent](#defining-an-agent)
- [What the agent gets](#what-the-agent-gets)
- [Dependencies at run time](#dependencies-at-run-time)
- [Your own image](#your-own-image)
- [A browser interface](#a-browser-interface)
- [Sessions and data](#sessions-and-data)
- [A LangChain agent](#a-langchain-agent)
- [A chat app in the browser](#a-chat-app-in-the-browser)
- [A notebook server](#a-notebook-server)
- [Coding agents in the background](#coding-agents-in-the-background)

## Defining an agent

Each key under `launch.agents` names an agent, and `gmlx launch <name>`
runs it. The smallest agent names a Python project in the current folder
and the command that starts it, here a script that the project's
`[project.scripts]` table defines:

```yaml
launch:
  agents:
    research-bot:
      runtime: python
      command: [research-bot]
```

```sh
cd ~/src/research-bot
gmlx launch research-bot
```

`runtime: python` installs the project's dependencies with uv when the
session starts, which [Dependencies at run time](#dependencies-at-run-time)
covers. An agent that brings its own image sets
[`image`](config.md#launchcontainerclientsimage) or
[`build`](config.md#launchcontainerclientsbuild) instead, and
[Your own image](#your-own-image) covers that. `command` is always
required.

An agent takes the keys that the reference entry
[`launch.agents`](config.md#launchagents) lists, together with the rules
for an agent's name. The values under `launch.container` apply to an agent
as they do to a client.

`launch` reads agents only from the
[config file in your home folder](config.md#launch), and
`gmlx launch --help` lists the configured agents. The launch flags work as
they do for a client, and the arguments after `--` follow the command.
[`gmlx launch`](cli.md#gmlx-launch) refuses `--no-container`,
`--config-path` and `--provider-id` for an agent, because an agent runs
only in a container and gets no configuration file.

## What the agent gets

`launch` sets these variables in the container. An
[`env`](config.md#launchcontainerenv) entry can replace the server, key and
model variables, by the rules of that key:

| Variable | Value |
|----------|-------|
| `GMLX_BASE_URL` | The server's `/v1` address, such as `http://127.0.0.1:8080/v1`. |
| `GMLX_API_KEY` | The key the agent sends. Through a session socket it is the placeholder `gmlx-container-session`. |
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
inference routes and the served assistants that its
[`assistants`](config.md#launchcontainerclientsassistants) list names. A
session sends at most 16 requests at once, so keep a framework's
concurrency at 16 or fewer. When no model is chosen, `launch` warns that the
agent gets no `GMLX_MODEL`, and the agent then has to name one itself.

An agent runs inside the same boundary as a client.
[Container security](container-security.md) describes what still leads
back to the Mac, and its [Custom agents](container-security.md#custom-agents)
section adds what the run-time install, `env` and the dependency volume
change for an agent.

## Dependencies at run time

With [`runtime: python`](config.md#launchagentsruntime), the agent runs in
an image that gmlx builds from its shipped recipe. That image starts from
the same base as every image that gmlx builds, Debian 13 with Node.js, and
adds Python 3.13 and uv. Each time a session starts, `uv sync` brings the
environment in line with the project's `uv.lock`. The command then runs in that
environment, with the environment's `bin` folder first on `PATH`.

A script from the project's `[project.scripts]`, such as `[research-bot]`,
works as a command, and so does `[python, -m, research_bot]` for a module
or a package with a `__main__.py`. A `.py` file runs with the environment's
Python. A command that is not on `PATH` in the container stops the launch
with a message that names it. An `env` entry `UV_NO_SYNC=1` skips the sync,
as it does for `uv run`.

The first launch of any runtime agent builds that image once, which
downloads uv, about 19 MB, and takes about three minutes. On a Mac where
gmlx has built no client image yet, the build first downloads the shared
base image, about 80 MB. The first launch of an agent in each project
folder then installs its dependencies, which takes about a minute for a
LangChain project, with uv's output after the lines of `launch`. Later
launches from the same folder find the environment in place.

The environment uses Debian's Python 3.13 when the project allows it. A
project whose `requires-python` excludes 3.13, or whose `.python-version`
names another version, gets a Python that uv downloads on the first launch,
30 to 90 MB, and keeps for later launches. `uv init` writes the minor
version of the Python on your Mac into both files, so a project made with a
Python other than 3.13 gets that download.

### The source folder

The project is the current folder, which `launch` shares read-write by
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

`launch` shares a `source` outside the shared folders read-only at the same
path, and names it as the source folder. uv then uses the `uv.lock` in it
as it is. A lock that is missing or out of date stops the launch with the
message that
[troubleshooting](troubleshooting.md#uv-says-the-lockfile-needs-to-be-updated)
shows.

A `source` inside a folder the session already shares takes that share's
mode and gets no separate share. A `source` that is a symbolic link, or
that is inside a folder `launch` never shares, gets the same refusal as a
[share](launch-container.md#shares) would.

A relative path as the command, such as `agent.py` or `bin/start`, is
looked up in the working folder and then in the project folder, which is
the `source` when one is set. So a script that lives in the source runs
from any folder.

A project run from another folder must be a package, as
`uv init --package` makes it, with a `[build-system]` table, so that its
module imports from any working folder. Its build backend must also write
nothing into the source when it builds the project. hatchling and uv_build
write nothing, while setuptools writes a `.egg-info` folder.

When the project is not a package, `python -m` cannot find its module from
another folder, and the launch stops with `No module named`. A build
backend that writes into a read-only source stops the launch with
`Read-only file system`. Troubleshooting gives the fix for
[the missing module](troubleshooting.md#a-source-elsewhere-fails-with-no-module-named)
and for
[the read-only source](troubleshooting.md#a-read-only-source-fails-with-read-only-file-system).

A launch with no `source` from a folder without a `pyproject.toml` has no
project to install, so uv stops it at once with ``No `pyproject.toml` found
in current directory or any parent directory``. Launch from the project
folder, or set `source`. A [single script](#a-single-script) with inline
metadata needs no project.

Refresh a lock in the container rather than on the Mac. A project with
dynamic metadata runs its build backend to lock, which can be code from the
folder the agent edits. To refresh it, launch the agent once from its
source folder, where the source is the read-write working folder and uv
updates the lock. You can also open `--shell` from that folder and run
`uv lock` there.

### A single script

A single script with inline metadata, the `# /// script` block of PEP 723,
runs with the script as the command, as `[agent.py]`. uv installs the
dependencies that the block names into a separate environment for the
script, and the script runs there. The form `[python, agent.py]` ignores
the block.

Under a read-only source, a script without a lockfile installs anyway,
with a warning. `uv lock --script agent.py` writes the lockfile beside the
script.

### The dependency volume

uv keeps the environment, its cache and any downloaded Python on a volume
at `/opt/agent` in the container. Its name is `gmlx-agent-<name>-uv`, or
`gmlx-agent-<name>-uv-<8 hex digits>` for a project other than `default`,
like a client's [volumes](launch-container.md#volumes). `launch` creates it
with the default size of 32G and the `gmlx.launch=1` label, and one session
uses it at a time.

The shared current folder chooses the
[project](launch-container.md#projects-and-sessions), so an agent with a
`source` gets a separate volume, and installs again, in each folder you
launch it from.

To use one volume from every folder, launch with `--no-mount-cwd`, from a
folder that no `--mount` or `mounts` entry shares. The session then belongs
to the `default` project and uses its volume. The agent starts in its
source when you launch from inside the source, and in the private home
otherwise. The source stays read-only in both cases.

The lock refresh in [The source folder](#the-source-folder) runs without
that flag. It belongs to the project of the source folder, so it installs
the dependencies once more, into a separate volume.

A [`volumes`](config.md#launchcontainervolumes) entry of the agent at
`/opt/agent` takes the place of that volume, which is how you set its size
or its name. No share or `--mount` may use `/opt/agent` or a path inside
it, and no global `volumes` entry may either. The environment is writable
by the agent and stays from one launch to the next, so a package the agent
changes there stays changed until the volume is deleted.

`uv run python` in a `--shell` session uses the agent's environment from
any folder, because the shell gets the same `UV_` variables as the agent.

### Offline launches

Under [`network: none`](config.md#launchcontainernetwork), `launch` sets
`UV_OFFLINE=1`, so uv starts from the synced environment without a network
and never waits for one. An environment that was never synced then fails
at once, as
[troubleshooting](troubleshooting.md#an-agents-first-launch-fails-under-network-none)
describes. So launch the agent once with the network in each project folder
before you turn it off. The agent still reaches the server through its
socket.

## Your own image

[`image`](config.md#launchcontainerclientsimage) and
[`build`](config.md#launchcontainerclientsbuild) work as they do for a
client. `launch` pulls or builds the image, builds it again when its
Containerfile or context changes, runs it by digest and checks the command
once in it.

An agent's Containerfile may start from any client's `:base` image, such as
`gmlx.invalid/launch-claude-code:base` to build on the Claude Code image.
Its `build` folder follows the rules of
[Your own Containerfile](container-images.md#your-own-containerfile), so
build a packaged agent from a fresh clone that you have reviewed.

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
a message that names the runtime key. Without `runtime`, the command runs
as written in the image, and `command: image` runs the image's ENTRYPOINT
and CMD, by the rules of
[The command that runs](container-images.md#the-command-that-runs).

## A browser interface

An agent with [`web_port`](config.md#launchagentsweb_port) is a
[browser app](launch-container.md#browser-apps), served as Open WebUI and
dsh are. The Mac reaches the app at `http://[::1]:<port>/` on a port that
belongs to the project, which `launch` opens when the app answers, or
prints with [`open_browser: false`](config.md#launchcontaineropen_browser).
[A chat app in the browser](#a-chat-app-in-the-browser) and
[A notebook server](#a-notebook-server) are complete examples.

The command must listen on `127.0.0.1` at `web_port` inside the container.
`launch` sets `HOST` and `PORT` for programs that read them, but a command
list runs with no shell, so a `$PORT` in the list stays literal. Write the
port itself, as the examples do.

`launch` refuses a `web_port` equal to the gmlx server's port, and a
`forward` entry equal to the `web_port`, since the container reaches those
ports at the same numbers. Two agents can use the same `web_port`, and
sessions of one agent in two projects run at once, because each gets a
separate port on the Mac.

## Sessions and data

Agents follow the rules of the clients in
[Projects and sessions](launch-container.md#projects-and-sessions). The
shared current folder chooses the project, one session runs per project, a
second launch from the same project joins it, and `--shell` opens a shell
in it. The container is named `gmlx-agent-<name>-` followed by 6 hex
digits, and the session's log is
`~/.cache/gmlx/launch/last-agent-<name>-<project>.log`.

For an agent with a browser interface, `--shell` starts the session with a
shell and prints the command that starts the app. For a runtime agent, that
command starts with `uv run`, which brings the environment up to date
first, and it names the script by its full path in the container. When a
symbolic link lies on that path, the printed command names the script as
`command` does.

`gmlx launch <name> --detach` runs the session in the background with no
terminal, and `gmlx launch <name> --stop` ends it. See
[Sessions in the background](launch-container.md#sessions-in-the-background).
An agent that reads its input then gets end of file, so give it a browser
interface or a task in its arguments, or run it in a terminal without
`--detach`.

When an agent leaves `launch.agents` while its session runs,
`gmlx launch <name> --list` still names its container, with the
`container stop` command that ends it.

Each project gets a [private home](glossary.md#private-home) at
`~/.local/share/gmlx/launch/agent-<name>/projects/<project>/home`, with no
client configuration. `launch` copies your
[seeds](launch-container.md#seeds) and your git name and email into it. The
dependency volume is named per project, and an image built from `build` is
tagged `gmlx.invalid/launch-agent-<name>-build`.

`gmlx launch <name> --remove-home` from the project folder asks one
question that names the project's private home and each volume that only
this project uses, with the space it takes on the Mac. These are the
dependency volume, also after `runtime` leaves the agent's settings, and
the volumes that the agent's `volumes` entries name for the project. A yes
removes the home and deletes the volumes. When the home is gone and
volumes remain, the question names the volumes alone.

For an agent with a browser interface, the removal also frees the
project's Mac port and names the address whose site data to clear. Without
a terminal, `--remove-home` removes nothing, and its message gives the
`rm -rf` and `container volume delete` commands instead.
[Removing container data](launch-container.md#removing-container-data)
lists the other data. `gmlx doctor` reports agent homes under the agent's
name, and it names the images and volumes that no setting uses, with the
command that deletes each one.

## A LangChain agent

These steps make a LangChain program that asks your server one question
and prints the answer:

1. Make a packaged project for the Python of the runtime image, and add
   the library:

   ```sh
   cd ~/src
   uv init --package --python 3.13 research-bot
   cd research-bot
   uv add langchain-openai
   ```

2. Write the agent in `src/research_bot/__init__.py`:

   ```python
   import os

   from langchain_core.messages import HumanMessage
   from langchain_openai import ChatOpenAI


   def main():
       llm = ChatOpenAI(model=os.environ["GMLX_MODEL"], max_tokens=2048)
       reply = llm.invoke([HumanMessage("Name three uses of a local model.")])
       print(reply.content)
   ```

3. Add the agent to your gmlx config file, the one that
   [Where gmlx looks](config.md#where-gmlx-looks) names:

   ```yaml
   launch:
     agents:
       research-bot:
         runtime: python
         command: [research-bot]
   ```

4. Launch it from the project folder:

   ```sh
   gmlx launch research-bot --model qwen3.8-27b-ud-q6@instruct
   ```

The program reads the model from `GMLX_MODEL`, and langchain-openai reads
the address and the key from `OPENAI_API_BASE` and `OPENAI_API_KEY`. It
leaves the sampling to the server's [family defaults](family-defaults.md),
since a fixed `temperature=0` makes a thinking model repeat itself, and it
caps the answer with `max_tokens`, which a thinking model spends on its
reasoning first.

The first launch builds the runtime image and installs the project, and
later launches start at once. The `@instruct` intent turns thinking off for
a Qwen model, so the answer comes in a few seconds rather than after its
reasoning. For `api: anthropic` and the other variables, see
[What the agent gets](#what-the-agent-gets).

## A chat app in the browser

These steps run a Streamlit chat page in the container, which you open in
your Mac browser:

1. Make a project and add the libraries:

   ```sh
   cd ~/src
   uv init --package --python 3.13 chat-desk
   cd chat-desk
   uv add streamlit openai
   ```

2. Write `app.py` in the project folder:

   ```python
   import os

   import streamlit as st
   from openai import OpenAI

   client = OpenAI()
   model = os.environ["GMLX_MODEL"]

   st.title("Chat desk")
   if "messages" not in st.session_state:
       st.session_state.messages = []
   for message in st.session_state.messages:
       st.chat_message(message["role"]).write(message["content"])
   if prompt := st.chat_input("Ask the local model"):
       st.session_state.messages.append({"role": "user", "content": prompt})
       st.chat_message("user").write(prompt)
       stream = client.chat.completions.create(
           model=model, messages=st.session_state.messages, stream=True)
       reply = st.chat_message("assistant").write_stream(stream)
       st.session_state.messages.append({"role": "assistant", "content": reply})
   ```

3. Add the agent to your gmlx config file, with the port that Streamlit
   listens on in the container:

   ```yaml
   launch:
     agents:
       chat-desk:
         runtime: python
         command: [streamlit, run, app.py, --server.address, "127.0.0.1",
                   --server.port, "8501", --server.headless, "true"]
         web_port: 8501
   ```

4. Launch it from the project folder with `gmlx launch chat-desk`, or with
   `gmlx launch chat-desk --detach` to keep it running without a terminal.

The OpenAI client reads the address and the key from `OPENAI_BASE_URL` and
`OPENAI_API_KEY`. `launch` opens the page at `http://[::1]:<port>/` once
Streamlit answers, and `gmlx launch chat-desk --stop` ends a detached
session. `--server.headless true` keeps Streamlit from asking for an email
address at its first start, which a session in the background cannot
answer.

A thinking model thinks before the first word of its answer appears, and
the page shows only the answer. Launch with an `@instruct` model, as the
[LangChain agent](#a-langchain-agent) does, for a model that answers at
once.

## A notebook server

These steps run JupyterLab in the container, with notebooks that call the
model:

1. Make a project with JupyterLab and the OpenAI library:

   ```sh
   cd ~/src
   uv init --package --python 3.13 lab
   cd lab
   uv add jupyterlab openai
   ```

2. Make a token for the page with `openssl rand -hex 16`, and
   add the agent to your gmlx config file with it:

   ```yaml
   launch:
     agents:
       lab:
         runtime: python
         command: [jupyter, lab, --ip, "127.0.0.1", --port, "8888",
                   --no-browser, --allow-root]
         web_port: 8888
         env: [JUPYTER_TOKEN=<your token>]
   ```

3. Launch it from the project folder with `gmlx launch lab --detach`. Open
   the address that `launch` prints, and sign in with the token.

4. In a notebook, call the model:

   ```python
   import os

   from openai import OpenAI

   reply = OpenAI().chat.completions.create(
       model=os.environ["GMLX_MODEL"],
       messages=[{"role": "user", "content": "Name three uses of a local model."}])
   print(reply.choices[0].message.content)
   ```

5. End the session with `gmlx launch lab --stop` from the project folder.

The kernel has the agent's variables, so the OpenAI client finds the
server. The session runs as root, and Jupyter refuses to start as root
without `--allow-root`. Every program on the Mac can open the address of
the page, and a notebook runs any code in the container, so keep the token
secret.

The notebooks are saved in the shared project folder. Code in a notebook
runs on the CPU of the container, and only the model runs on the GPU.

## Coding agents in the background

These steps give Claude Code one task at a time in the background, each in
a separate git worktree, and you merge the branches that it commits:

1. Make the folder `~/containers/claude-bg` with a Containerfile of one
   line:

   ```dockerfile
   FROM gmlx.invalid/launch-claude-code:base
   ```

2. Add the agent to your gmlx config file:

   ```yaml
   launch:
     agents:
       fixer:
         build: ~/containers/claude-bg
         api: anthropic
         command: [claude, -p, --dangerously-skip-permissions]
         env: [IS_SANDBOX=1, DISABLE_AUTOUPDATER=1]
   ```

3. Make a worktree for the task, and start the agent in it with the task
   after `--`:

   ```sh
   cd ~/src/app
   git worktree add ../app-mul -b add-mul
   cd ../app-mul
   gmlx launch fixer --detach -- "Add a mul function to calc.py, then commit it."
   ```

4. Follow the sessions with `gmlx launch --list`, which names the output
   file of each one.

5. When the session has ended, read the branch, merge it, and remove the
   private home and the worktree:

   ```sh
   cd ~/src/app
   git diff main add-mul
   git merge add-mul
   (cd ../app-mul && gmlx launch fixer --remove-home)
   git worktree remove ../app-mul
   ```

An agent needs `runtime`, `image` or `build`, and the one-line
Containerfile gives it the image of Claude Code that gmlx builds. With
`api: anthropic`, Claude Code gets the server's address, key and model.
`-p` runs the task that follows `--`, and the session ends when Claude Code
has finished it. `launch` sets `IS_SANDBOX=1` and turns off the
auto-updater only for the built-in client, so the agent sets both in `env`.

Each worktree is a separate project, so tasks in two worktrees run at once,
each with a separate session and private home. The session shares the
repository's [git folder](launch-container.md#git-in-a-worktree)
read-write, so the agent commits to the branch of its worktree with your
git name. Run `--remove-home` before you remove the worktree, because
`launch` finds the private home by the project's folder.

With `--dangerously-skip-permissions`, Claude Code runs every command
without asking, inside the container.

Read each branch before you merge it or run its code on the Mac, since a
branch can change
[files that the Mac runs](container-security.md#shares-that-lead-back-to-the-mac).
On a repository that you do not trust, follow
[A session for code you do not trust](container-security.md#a-session-for-code-you-do-not-trust)
instead.
