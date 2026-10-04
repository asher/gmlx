# Custom agents

A custom agent is a program of your own that `gmlx launch` runs in a
container against your gmlx server. You define it in your config, and
`gmlx launch <name>` starts it with the server's address, a key and a model
in its environment. [Agent examples](agent-examples.md) has a chat page, a
notebook server and coding agents that run in the background.

- [Your first agent](#your-first-agent)
- [What the agent gets](#what-the-agent-gets)
- [Dependencies at run time](#dependencies-at-run-time)
- [Your own image](#your-own-image)
- [A browser interface](#a-browser-interface)
- [Sessions and data](#sessions-and-data)

## Your first agent

These steps make a LangChain program that asks your server one question and
prints the answer:

1. Make a Python project and add the library:

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

3. Add the agent to your [gmlx config file](config.md#where-gmlx-looks):

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

The first launch builds the agent image and installs the project, which
takes a few minutes. Later launches start at once. `@instruct` turns off
thinking for a Qwen model, so the answer comes in a few seconds.

`runtime: python` tells `launch` to install the project's dependencies with
uv, and `command` is the script that `[project.scripts]` defines.
[`launch.agents`](config.md#launchagents) lists every key an agent takes,
and `gmlx launch --help` lists your agents.

## What the agent gets

`launch` sets these variables in the container:

| Variable | Value |
|----------|-------|
| `GMLX_BASE_URL` | The server's `/v1` address, such as `http://127.0.0.1:8080/v1`. |
| `GMLX_API_KEY` | The key the agent sends, the placeholder `gmlx-container-session`. |
| `GMLX_MODEL` | The model, from `--model`, then the agent's `model`, then the server's default. |
| `OPENAI_BASE_URL`, `OPENAI_API_BASE` | The same as `GMLX_BASE_URL`, with `api: openai`, the default. |
| `OPENAI_API_KEY` | The same as `GMLX_API_KEY`, with `api: openai`. |
| `ANTHROPIC_BASE_URL` | The server's root address, without `/v1`, with `api: anthropic`. |
| `ANTHROPIC_API_KEY` | The same as `GMLX_API_KEY`, with `api: anthropic`. |
| `ANTHROPIC_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL` | The same as `GMLX_MODEL`, with `api: anthropic`. |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | The model's context window, with `api: anthropic`. |
| `HOST`, `PORT` | `127.0.0.1` and the `web_port`, for an agent with a [browser interface](#a-browser-interface). |

Most frameworks, including the OpenAI SDK and LangChain, read the `OPENAI_`
variables with no code of your own. Set [`api: anthropic`](config.md#launchagentsapi)
for the Anthropic SDKs and the Claude Agent SDK.

The agent reaches only the server's inference routes, as
[What the client reaches on the server](container-security.md#what-the-client-reaches-on-the-server)
describes. Keep a framework's concurrency at 16 requests or fewer.

## Dependencies at run time

With [`runtime: python`](config.md#launchagentsruntime), the agent runs in an
image with Python 3.13 and uv. Each session starts with `uv sync`, then runs
the command in the project's environment. A command can be a script from
`[project.scripts]`, `[python, -m, mypackage]`, or a `.py` file.

The first runtime agent on a Mac builds that image once, in about three
minutes. The first launch in each project folder then installs the
dependencies, about a minute for a LangChain project.

Create projects with `--python 3.13`, as the steps above do. `uv init` writes
your Mac's Python version into the project, and any version other than 3.13
makes uv download another Python on the first launch.

### The source folder

By default the project is the folder you launch from, shared read-write so
uv can update `uv.lock`. An agent that lives elsewhere names its folder with
[`source`](config.md#launchagentssource):

```yaml
launch:
  agents:
    reviewer:
      runtime: python
      source: ~/src/reviewer
      command: [reviewer]
```

`launch` shares that folder read-only, so the agent can work in another
project without changing its own code. Three things follow:

- The `uv.lock` must be up to date. To refresh it, launch the agent once
  from the source folder.
- The project must be a package, as `uv init --package` makes it, so its
  module imports from any folder.
- Its build backend must not write into the source. hatchling and uv_build
  are fine, while setuptools is not.

Troubleshooting covers each error:
[the lockfile](troubleshooting.md#uv-says-the-lockfile-needs-to-be-updated),
[`No module named`](troubleshooting.md#a-source-elsewhere-fails-with-no-module-named)
and
[`Read-only file system`](troubleshooting.md#a-read-only-source-fails-with-read-only-file-system).

### A single script

A script with an inline `# /// script` dependency block runs with the script
as the command, as `[agent.py]`. uv installs what the block names. Write the
lockfile beside it with `uv lock --script agent.py`.

### The dependency volume

uv keeps the environment on a [volume](container-access.md#volumes), one per
project. An agent with a `source` therefore installs again in each folder
you launch it from. To share one environment from every folder, launch with
`--no-mount-cwd`.

To set the volume's size, add a `volumes` entry for `/opt/agent` to the
agent. The environment stays from one launch to the next, including anything
the agent changes in it, until `--remove-home` deletes the volume.

### Offline launches

With [`network: none`](config.md#launchcontainernetwork), uv cannot
download anything. Launch the agent once with the network in each project
folder before you turn it off, as
[troubleshooting](troubleshooting.md#an-agents-first-launch-fails-under-network-none)
explains.

## Your own image

[`image`](config.md#launchcontainerclientsimage) and
[`build`](config.md#launchcontainerclientsbuild) work for an agent as they do
for a client, as [Custom container images](container-images.md) shows. An
agent's Containerfile can start from any client's image, such as
`gmlx.invalid/launch-claude-code:base`.

To add system packages to a runtime agent, start from the runtime base and
keep `runtime: python`:

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

Without `runtime`, the command runs in the image as written.

## A browser interface

An agent with [`web_port`](config.md#launchagentsweb_port) is a
[browser app](launch-container.md#browser-apps). `launch` opens it in your
Mac browser at `http://[::1]:<port>/` once it answers.

The command must listen on `127.0.0.1` at the `web_port` inside the
container. A command list runs without a shell, so write the port number in
the list rather than `$PORT`.
[A chat app in the browser](agent-examples.md#a-chat-app-in-the-browser)
and [A notebook server](agent-examples.md#a-notebook-server) show complete
setups.

## Sessions and data

Agents follow the same session rules as clients, as
[Container sessions](container-sessions.md) describes. Each project folder
gets its own session and private home, a second launch joins the running
session, and `--shell` opens a shell in it.

`--detach` runs an agent in the background, and `--stop` ends it. An agent
in the background gets no input, so give it a browser interface or a task in
its arguments.

`gmlx launch <name> --remove-home` in the project folder removes the
project's private home and the agent's volumes, after one question.
