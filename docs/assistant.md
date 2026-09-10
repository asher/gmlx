# The assistant

This guide is for giving a model tools and long-term memory. gmlx has a
built-in assistant that runs a tool loop around the server's chat
completions and keeps a memory of what you tell it. Configure it once in a
top-level `assistant:` block and it is available from three surfaces.

| Surface | How | Tools run |
|---------|-----|-----------|
| voice | `gmlx talk` with `talk.brain: assistant`, described in [talk.md](talk.md#the-assistant-by-voice) | on your machine, as you |
| text | `gmlx chat --assistant` | on your machine, as you |
| API | served assistant ids under `server.assistants:` | on the server host |

It is built for short tasks: looking something up, chaining a few tool
calls, writing a note or remembering a fact. Each turn ends when the model
answers, and nothing keeps working in the background afterward. The coding
agents that `gmlx launch` connects have loops of their own and use this
server for inference only.

Tool support needs the MCP SDK, which the `assistant` extra installs:

```sh
uv tool install "gmlx[assistant]"     # or: pip install "gmlx[assistant]"
```

- [The tool loop](#the-tool-loop)
- [The assistant block](#the-assistant-block)
- [Tool examples](#tool-examples)
- [Memory](#memory)
- [Text chat](#text-chat)
- [Served assistants](#served-assistants)
- [Security](#security)

## The tool loop

A turn is the standard OpenAI tool loop, run as a client of the server's
chat completions endpoint. Your text goes out with the configured tools
attached. If the model answers with tool calls instead of prose, the
assistant executes them, sends the results back and asks again, until the
model answers in prose. A failed call comes back to the model as an error
string it can retry, and completed rounds stay in the history so later turns
can build on them.

The loop is bounded by `max_tool_rounds`. Once that many rounds have called
tools, a final request goes out with no tools attached, which forces an
answer from a model that would otherwise keep calling them.

Each round is an ordinary chat-completion request, so whatever the server
applies to a request, such as sampling profiles, speculative decoding and
the prompt cache, applies here too. A round costs a model turn plus the
tool call, which makes multi-tool answers slower than plain chat.

Tool calling needs a model that is competent at it. On a 48 GB or larger
machine the Qwen3.6-27B class is a good fit. On 32 GB the smallest usable
model is Qwen3.5-9B, at the cost of more tool-call errors.

## The assistant block

The full block, with defaults:

```yaml
assistant:                    # the built-in tool-loop assistant used by talk,
                              # chat --assistant, and server.assistants
  max_tool_rounds: 8          # tool-call rounds per turn, then it must answer
  tool_timeout_s: 60          # per tool invocation
  mcp:                        # tool servers (Model Context Protocol)
    - name: files             # stdio, command is the argv to spawn
      command: [npx, -y, "@modelcontextprotocol/server-filesystem", "~/notes"]
    - name: search            # or streamable HTTP
      url: http://127.0.0.1:8931/mcp
  memory:
    enabled: true             # long-term memory (needs server embeddings:)
    path: null                # default ~/.local/share/gmlx/assistant-memory.db
    top_k: 4                  # memories injected per turn
    extract: true             # distill turns into facts (false = raw transcripts)
    ttl_days: null            # expire older memories (null = keep forever)
    max_items: 20000          # store cap, evicts least-recalled oldest first
```

[MCP](glossary.md), the Model Context Protocol, is the standard way for a
model to call tools provided by separate programs. Each `mcp:` entry is
either a stdio server, where `command` is the argv to spawn plus an optional
`env` map, or a streamable-HTTP endpoint given as `url`. When two servers
offer a tool of the same name, the tool gets the server's name as a prefix.
An MCP server that fails to start produces a warning, and the loop runs
without it. A missing `assistant` extra is reported the same way, with an
install hint. Each stdio server's stderr goes to its own log at
`~/.cache/gmlx/mcp-<name>.log`.

A stdio tool server starts with the MCP SDK's default environment, which
on macOS is `HOME`, `PATH`, `SHELL`, `TERM`, `USER` and `LOGNAME`, plus
whatever `env:` adds. Nothing else from your shell is inherited, so a token
set in your environment never reaches third-party tool code unless you pass
it.

## Tool examples

Any stdio or streamable-HTTP MCP server is configured the same way. The
examples below are each a complete `assistant:` block, and the first two run
with no account or key.

### Web search without an API key

DuckDuckGo's search endpoint is keyless. This community server wraps it with a
search tool plus a page fetcher:

```yaml
assistant:
  mcp:
    - name: web
      command: [uvx, duckduckgo-mcp-server]
```

### Web search, self-hosted

If you already run [SearXNG](https://docs.searxng.org), point the assistant
at it for aggregated results that stay local. The instance must allow the
JSON format, so add `json` to `search.formats` in its settings.

```yaml
assistant:
  mcp:
    - name: search
      command: [npx, -y, mcp-searxng]
      env: {SEARXNG_URL: "http://127.0.0.1:8888"}
```

### Web search with an API key

Brave's official server returns richer results on a free-tier key, which
reaches the server through `env:`:

```yaml
assistant:
  mcp:
    - name: brave
      command: [npx, -y, "@brave/brave-search-mcp-server"]
      env: {BRAVE_API_KEY: your-key-here}
```

### Document retrieval as a tool

Qdrant's official server in embedded local mode gives the assistant store
and find tools over a vector collection on disk, with no database process
to run. The tool server embeds documents itself with a separate small model,
so this needs no `server.embeddings:`.

```yaml
assistant:
  mcp:
    - name: docs
      command: [uvx, mcp-server-qdrant]
      env: {QDRANT_LOCAL_PATH: ~/vectors, COLLECTION_NAME: notes}
```

This is different from memory, described next. Memory is automatic and
personal: distilled facts about you, recalled on every turn. A retrieval
tool is explicit and works on documents, and the model decides when to
search what you loaded. For document RAG in a chat UI instead of through the
assistant, see [rag.md](rag.md).

## Memory

Memory is a local retrieval store over the server's endpoints. Facts are
embedded through `/v1/embeddings` into a sqlite file. On each turn the
closest facts are recalled, reordered by `/v1/rerank` when that service is
configured, and injected as transient context, so the chat history never
grows because of them. Without `server.embeddings:` the assistant still
runs, memoryless, after a warning.

What gets stored is an extracted fact such as "sister Ana, birthday March
12", not a transcript. After each turn a background request asks the chat
model to reduce the exchange to at most three durable facts. Small talk
yields none and stores nothing, and a new fact that restates an existing
one replaces it. `extract: false` stores raw exchanges instead. Rows older
than `ttl_days` expire at startup, and when the store reaches `max_items`
the oldest never-recalled rows are evicted first.

The store is shared between the voice and text surfaces, so what you tell
the assistant in `gmlx talk` it remembers in `gmlx chat --assistant`. Inside
either, `/memory` lists the stored facts with their ids, `/memory forget ID`
removes one and `/memory clear yes` removes them all. The file sits at
`~/.local/share/gmlx/assistant-memory.db`. A served assistant that enables
memory gets a separate store for each id, beside it.

## Text chat

`gmlx chat --assistant` runs the chat client against the assistant on the
managed server, starting the server if it is down, instead of loading a
model in-process. The positional argument is a served model id, or is
omitted for the server's default. A file path is refused, since the server
owns the model. Tool activity appears as transient status lines while the
answer streams.

```sh
gmlx chat --assistant                 # server default model
gmlx chat qwen3.6-27b --assistant     # a specific served model
```

The terminal experience is the one [chat.md](chat.md) describes. Rendering,
themes, history, sessions and `/system` all work, `/retry` and `/undo`
rewind whole tool rounds, and `/memory` is added. Sampling flags you set
are forwarded to the server on each round. Flags that only apply to a local
load fall into two groups. `--adapter`, `--mmproj` and the `--chat-template`
flags are refused, because honoring them would need a different model than
the one the server holds. The loading, memory, speculation and streaming
flags are ignored with a note, since the server already decided them.
`/image`, `/audio` and the thinking budget are not available in this mode.

## Served assistants

`server.assistants:` exposes pseudo-model ids that run the tool loop
server-side behind plain chat completions. Thin clients such as curl, Open
WebUI or a phone app pick the assistant id as their model and get MCP tools
with no client-side loop:

```yaml
server:
  model_dirs: [~/models]
  assistants:
    helper:              # the served id, listed in /v1/models
      model: qwen3.6-27b # required. The configured model that answers
      memory: false      # off by default. true gives this id one store,
                         #   assistant-helper.db, shared by all its clients
      mcp: null          # null inherits assistant.mcp below. A list scopes
                         #   this id's tools, and [] gives it none
  assistant_allow_remote: false   # see the Security section

models:
  qwen3.6-27b:
    path: Qwen3.6-27B-Q6_K.gguf

assistant:
  mcp:
    - name: clock
      command: [uvx, mcp-server-time]
```

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "helper",
  "messages": [{"role": "user", "content": "What time is it?"}]
}'
```

How a request to `/v1/chat/completions` is routed depends on its `model`
and whether it carries tools of its own. An empty `tools` list counts as
none.

| Request | Behavior |
|---------|----------|
| assistant id, no `tools` | the server runs the loop and returns the prose answer under the assistant id. A stream stays alive with SSE comments naming each tool |
| assistant id, non-empty `tools` | the client is running a loop of its own, so the id is rewritten to the underlying model and the request passes through untouched |
| any other id | untouched |
| assistant id on `/v1/responses` or `/v1/messages` | 400, assistant models are chat-completions only |

Reported usage sums completion tokens across all rounds, and prompt tokens
are the final round's. `stop` sequences forward to each round. A server
runs at most 4 assistant turns at once, and a request over the cap gets an
immediate 429. A streaming turn can be cancelled by client disconnect, which
takes effect at the next delta or tool boundary. A non-streaming turn runs
to completion.

## Security

`gmlx talk` and `gmlx chat --assistant` execute tools on your local
machine, as you, which is the trust domain of any local CLI tool. A served
assistant executes tools on the server host, reachable by anything that can
reach the server. On the default loopback bind that is still your machine,
but beyond loopback it means anyone with the API key can trigger tool
execution on the host. The protections are:

- Tools come only from the config. A request cannot supply MCP servers or
  redefine tools. A `tools` array in the request switches the server loop off.
- A non-loopback bind with `server.assistants` configured refuses to start
  unless `server.assistant_allow_remote: true`, in addition to the rule that a
  non-loopback bind requires an API key.
- Even then, a remote-exposed assistant must declare an explicit `mcp:`
  list, with `[]` for a tool-less loop, instead of inheriting the shared
  one. Each assistant's tool exposure is therefore a decision you make in
  the config, never a default.
- `gmlx doctor` warns when assistants are served beyond loopback and names
  each one's tool scope.

Memory on a served assistant is one store shared by every client of that
id, which suits a personal server and not a multi-user one.
