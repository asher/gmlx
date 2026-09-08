# The assistant

This guide is for giving a model tools and long-term memory. gmlx has a
built-in assistant, a bounded tool loop with memory, wrapped around the
server's own chat completions. Configure it once in a top-level `assistant:`
block and it is available from three surfaces.

| Surface | How | Tools run |
|---------|-----|-----------|
| voice | `gmlx talk` with `talk.brain: assistant` ([talk.md](talk.md#the-assistant-by-voice)) | on your machine, as you |
| text | `gmlx chat --assistant` | on your machine, as you |
| API | served assistant ids under `server.assistants:` | on the server host |

It is a lightweight assistant, not an autonomous agent. Each turn runs a
bounded tool loop and ends when the model answers; nothing keeps working in
the background afterward. It is built for short tasks: look something
up, chain a few tool calls, write a note, remember a fact. The coding agents
`gmlx launch` connects have their own loops and use this server for
inference only.

- [The tool loop](#the-tool-loop)
- [The assistant block](#the-assistant-block)
- [Tool examples](#tool-examples)
- [Memory](#memory)
- [Text chat](#text-chat)
- [Served assistants](#served-assistants)
- [Security](#security)

Tool support needs the MCP SDK:

```sh
uv tool install "gmlx[assistant]"     # or: pip install "gmlx[assistant]"
```

## The tool loop

A turn is the standard OpenAI tool loop, run as a client of the server's own
chat completions endpoint. The user text goes out with the configured tools
attached. If the model answers with tool calls instead of prose, the
assistant executes them and sends the results back, and this repeats until
the model answers. Up to `max_tool_rounds` rounds may call tools, then one
final tool-less request forces an answer, so a model that keeps calling tools cannot loop
forever. A failed call comes back to the model as an error string it can
retry, and completed rounds stay in the history so later turns can build on
them.

Because every round is an ordinary chat-completion request, whatever the
server applies per request, such as sampling profiles, speculative decoding
and the prompt cache, applies to assistant rounds too. Each round costs a
model turn plus the tool call, so multi-tool answers are slower than plain
chat.

Tool calling needs a model that is competent at it. The Qwen3.6-27B class is
suitable on a 48 GB or larger machine; Qwen3.5-9B is the smallest usable model on
32 GB, with more tool-call errors.

## The assistant block

The full block, with defaults:

```yaml
assistant:                    # the built-in tool-loop assistant used by talk,
                              # chat --assistant, and server.assistants
  max_tool_rounds: 8          # tool-call rounds per turn, then it must answer
  tool_timeout_s: 60          # per tool invocation
  mcp:                        # tool servers (Model Context Protocol)
    - name: files             # stdio: command is the argv to spawn
      command: [npx, -y, "@modelcontextprotocol/server-filesystem", "~/notes"]
    - name: search            # or streamable HTTP
      url: http://127.0.0.1:8931/mcp
  memory:
    enabled: true             # long-term memory (needs server embeddings:)
    path: null                # default ~/.local/share/gmlx/assistant-memory.db
    top_k: 4                  # memories injected per turn
    extract: true             # distill turns into facts (false = raw transcripts)
    ttl_days: null            # expire older memories (null = keep forever)
    max_items: 20000          # store cap; evicts least-recalled oldest first
```

[MCP](glossary.md), the Model Context Protocol, is the standard way for a
model to call tools provided by separate programs. Each `mcp:` entry is
either a stdio server, where `command` is the argv to spawn plus an optional
`env` map, or a streamable-HTTP endpoint given as `url`. Tool-name
collisions across servers get a server-name prefix. An MCP server that fails
to start produces a warning rather than blocking the loop, and a missing
`assistant` extra does the same with an install hint. A stdio server's
stderr goes to a per-server log at `~/.cache/gmlx/mcp-<name>.log`.

A stdio tool server runs with a minimal environment of `HOME`, `PATH`,
`SHELL`, `TERM`, `USER` and `LOGNAME`. `env:` adds to that, and nothing else
from your shell is inherited, so a token set in your environment never
reaches third-party tool code unless you pass it.

## Tool examples

Any stdio or streamable-HTTP MCP server is configured the same way. These need no
further setup.

### Web search without an API key

DuckDuckGo's search endpoint is keyless, and this community server wraps it
with a search tool plus a page fetcher:

```yaml
assistant:
  mcp:
    - name: web
      command: [uvx, duckduckgo-mcp-server]
```

### Web search, self-hosted

If you already run [SearXNG](https://docs.searxng.org), point the assistant
at it for aggregated results that stay local. The instance must allow the
JSON format: add `json` to `search.formats` in its settings.

```yaml
assistant:
  mcp:
    - name: search
      command: [npx, -y, mcp-searxng]
      env: {SEARXNG_URL: "http://127.0.0.1:8888"}
```

### Web search with an API key

Brave's official server returns richer results on a free-tier key, and shows
how `env:` is used:

```yaml
assistant:
  mcp:
    - name: brave
      command: [npx, -y, "@brave/brave-search-mcp-server"]
      env: {BRAVE_API_KEY: your-key-here}
```

### Document retrieval as a tool

Qdrant's official server in embedded local mode gives the assistant store and
find tools over a vector collection on disk, with no database process to run.
Embedding happens inside the tool server with its own small model:

```yaml
assistant:
  mcp:
    - name: docs
      command: [uvx, mcp-server-qdrant]
      env: {QDRANT_LOCAL_PATH: ~/vectors, COLLECTION_NAME: notes}
```

Memory, below, is automatic and personal: distilled facts about you, recalled
every turn. A retrieval tool is explicit and operates on documents: the model
decides when to search over what you loaded. For document RAG in a chat UI
rather than through the assistant, see [rag.md](rag.md).

## Memory

Memory is a local retrieval store over the server's own endpoints. Facts are
embedded through `/v1/embeddings` into a sqlite file, and every turn recalls
the closest ones, reordered by `/v1/rerank` when configured, and injects them
as transient context that never lengthens the chat history. Without
`server.embeddings:` the assistant still runs, memoryless, after one warning.

What gets stored is an extracted fact ("sister Ana, birthday March 12"), not a
transcript. After each turn a background request asks the chat model to reduce
the exchange to at most three durable facts, or none, so small talk
stores nothing, and a new fact that restates an existing one replaces it.
`extract: false` stores raw exchanges instead. `ttl_days` expires old rows at
startup, and `max_items` caps the store, evicting the never-recalled oldest
rows first.

The store is shared between the voice and text surfaces, so what you tell
the assistant in `gmlx talk` it remembers in `gmlx chat --assistant`. Inside
either, `/memory` lists the stored facts with their ids, `/memory forget ID`
removes one, and `/memory clear yes` removes them all. The file sits at
`~/.local/share/gmlx/assistant-memory.db`. Served assistants that enable
memory get their own per-id store beside it.

## Text chat

`gmlx chat --assistant` switches the chat client's turn engine from a local
model load to the assistant on the managed server, auto-started if down. The
positional argument is a served model id, or omitted for the server's
default. A file path is refused, since the server owns the model. Tool
activity appears as transient status lines while the answer streams.

```sh
gmlx chat --assistant                 # server default model
gmlx chat qwen3.6-27b --assistant     # a specific served model
```

The terminal experience is unchanged: rendering, themes, history, sessions,
`/system`, `/retry` and `/undo`, which rewind whole tool rounds, plus
`/memory`. Sampling flags forward to the server per round once you set
them, and flags that only make sense for a local load are rejected or
ignored with a note. `/image`, `/audio` and the thinking budget are not
available in this mode. The client is described in [chat.md](chat.md).

## Served assistants

`server.assistants:` exposes pseudo-model ids that run the tool loop
server-side behind plain chat completions. Thin clients such as curl, Open
WebUI or a phone app pick the assistant id as their model and get MCP tools
with no client-side loop:

```yaml
server:
  model_dirs: [~/models]
  assistants:
    helper:              # served id: shows up in /v1/models
      model: qwen3.6-27b # required: the underlying configured model
      memory: false      # off by default; true = one shared store for every
                         #   client of this id, in its own assistant-helper.db
      mcp: null          # null = inherit assistant.mcp below; or an explicit
                         #   list to scope this id's tools ([] = tool-less)
  assistant_allow_remote: false   # see Security below

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

The routing contract for a request to `/v1/chat/completions`:

| Request | Behavior |
|---------|----------|
| assistant id, no `tools` (an empty list counts) | the server runs the loop and returns the prose answer under the assistant id; a stream stays alive with SSE comments naming each tool |
| assistant id, non-empty `tools` | the client is running its own loop: the id is rewritten to the underlying model and the request passes through untouched |
| any other id | untouched |
| assistant id on `/v1/responses` or `/v1/messages` | 400, assistant models are chat-completions only |

Reported usage sums completion tokens across all rounds, and prompt tokens
are the final round's. Concurrent assistant turns are capped at 4 per server,
and a request over the cap gets an immediate 429. A `stop` sequence forwards
to every round, a non-streaming turn cannot be cancelled by client disconnect,
and a stream-path cancel takes effect at the next delta or tool boundary.

## Security

`gmlx talk` and `gmlx chat --assistant` execute tools on your local machine,
as you, which is the trust domain of any local CLI tool. A served
assistant executes tools on the server host, reachable by anything that can
reach the server. On the default loopback bind that is still your machine;
beyond loopback it means anyone with the API key can trigger tool execution
on the host. The protections are:

- Tools come only from the config. A request cannot supply MCP servers or
  redefine tools, and its own `tools` array switches the server loop off.
- A non-loopback bind with `server.assistants` configured refuses to start
  unless `server.assistant_allow_remote: true`, in addition to the rule that a
  non-loopback bind requires an API key.
- Even then, a remote-exposed assistant must declare an explicit `mcp:` list
  (`[]` for a tool-less loop) rather than inherit the shared one, so exposing
  tools to the network is a deliberate, per-assistant decision.
- `gmlx doctor` warns when assistants are served beyond loopback and names
  each one's tool scope.

Memory on a served assistant is one shared store across every client of that
id, acceptable for a personal server and unsuitable for anything multi-user, which is
why it defaults off.
