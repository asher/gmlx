# Assistant

The built-in assistant gives a model tools and a long-term memory. Name the
tools in the `assistant` block of the [configuration file](config.md), and
chat with it:

```yaml
assistant:
  mcp:
    - name: web
      command: [uvx, duckduckgo-mcp-server]
```

```sh
gmlx chat --assistant
```

This gives the model web search with no API key. Tools need the `assistant`
extra, which [Optional features](installation.md#optional-features) shows
how to install. Memory needs an embeddings service, as [Memory](#memory)
describes.

The same `assistant` block works in three places:

| Where | How to start it | Where tools run |
|-------|-----------------|-----------------|
| Text | `gmlx chat --assistant` | On your Mac, as you |
| Voice | `gmlx talk` with `talk.brain: assistant`, as [The assistant by voice](talk.md#the-assistant-by-voice) shows | On your Mac, as you |
| API | A served assistant id from `server.assistants` | On the server host |

The assistant suits short tasks: looking something up, chaining a few tool
calls, writing a note or remembering a fact. A turn ends when the model
answers. For longer coding work, connect a coding agent with
[`gmlx launch`](launch.md).

- [Tools](#tools)
- [Tool examples](#tool-examples)
- [Memory](#memory)
- [Text chat](#text-chat)
- [Served assistants](#served-assistants)
- [Security](#security)

## Tools

Tools come from [MCP](glossary.md#mcp) servers: separate programs that offer
tools to a model. Each entry in [`assistant.mcp`](config.md#assistantmcp)
runs a local `command` or connects to a `url`:

```yaml
assistant:
  mcp:
    - name: files
      command: [npx, -y, "@modelcontextprotocol/server-filesystem", "/Users/me/notes"]
    - name: search
      url: http://127.0.0.1:8931/mcp
```

When the model asks for a tool, the assistant runs it, sends the result back
and asks the model again, until the model answers in text. Each round is an
ordinary chat request, so an answer that needs several tools takes longer
than a plain reply. Models that handle tool calls well, such as
Qwen3.8-27B, make fewer mistakes than small ones.

Two settings limit a turn:

| Key | Default | Meaning |
|-----|---------|---------|
| [`assistant.max_tool_rounds`](config.md#assistantmax_tool_rounds) | `8` | Rounds of tool calls before the model must answer without tools. |
| [`assistant.tool_timeout_s`](config.md#assistanttool_timeout_s) | `60` | Seconds before a tool call returns a timeout error to the model. |

A command server gets only a few variables from your environment, so pass
any token it needs in [`env`](config.md#assistantmcpenv). `~` is not
expanded in `command` or `env`, so write full paths. `gmlx doctor` checks
that each command server's program is found.

## Tool examples

Each example is a complete `assistant` block. The block at the top of this
page adds DuckDuckGo search, which also fetches pages and needs no key.

### Web search on your own SearXNG

A [SearXNG](https://docs.searxng.org) instance that you run keeps your
queries local. Add `json` to `search.formats` in its settings:

```yaml
assistant:
  mcp:
    - name: search
      command: [npx, -y, mcp-searxng]
      env: {SEARXNG_URL: "http://127.0.0.1:8888"}
```

### Web search with an API key

Brave's official server gives richer results with a free API key:

```yaml
assistant:
  mcp:
    - name: brave
      command: [npx, -y, "@brave/brave-search-mcp-server"]
      env: {BRAVE_API_KEY: your-key-here}
```

### Document search as a tool

The official Qdrant server, in local mode, stores and finds documents in a
folder on disk. It embeds them with its own small model, so it does not
need `server.embeddings`:

```yaml
assistant:
  mcp:
    - name: docs
      command: [uvx, mcp-server-qdrant]
      env: {QDRANT_LOCAL_PATH: /Users/me/vectors, COLLECTION_NAME: notes}
```

Unlike memory, which recalls facts about you on every turn, a search tool
works on documents you load, and the model decides when to use it. For
document search in a chat app, see [RAG pipelines](rag.md).

## Memory

The assistant remembers facts across conversations. After each turn, it
asks the model for up to three lasting facts from the exchange, such as
"sister Ana, birthday March 12". Small talk stores nothing. Before each
turn, it finds the stored facts closest to your message and adds them to
that request only.

Memory needs [`server.embeddings`](config.md#serverembeddings). When the
server also runs a [reranker](services.md#reranking), it picks the best
facts when many match. Without embeddings, the assistant prints a
`memory disabled` warning and runs without memory.

`gmlx talk` and `gmlx chat --assistant` share one store, so a fact you say
by voice is known in text chat too. Manage it from either client:

| Command | Effect |
|---------|--------|
| `/memory` | Lists the newest 20 facts with their ids |
| `/memory forget ID` | Removes one fact |
| `/memory clear yes` | Removes every fact |

The store is `~/.local/share/gmlx/assistant-memory.db` by default, and
[`assistant.memory.path`](config.md#assistantmemorypath) moves it.
[`top_k`](config.md#assistantmemorytop_k) sets how many facts a turn gets,
4 by default. [`ttl_days`](config.md#assistantmemoryttl_days) and
[`max_items`](config.md#assistantmemorymax_items) limit the store's age and
size.

## Text chat

`gmlx chat --assistant` sends each turn through the server and starts the
server if it is down. Name a served model id, or none for the server's
default model:

```sh
gmlx chat --assistant
gmlx chat qwen3.8-27b-ud-q6 --assistant
```

Chat works as it does [with a server model](chat.md#where-the-model-runs).
While the model uses a tool, a status line shows `[assistant] using NAME...`.
`/retry` and `/undo` go back over whole tool rounds, and `/model <id>`
switches models and keeps the conversation.

## Served assistants

A served assistant is a model id that runs the tool loop in the server.
Clients that cannot run a tool loop, such as curl, Open WebUI or a phone
app, choose the id as their model and get the tools. The
[`server.assistants`](config.md#served-assistants) block defines each id:

```yaml
server:
  model_dirs: [~/models]
  assistants:
    helper:                      # The served id, listed in /v1/models.
      model: qwen3.8-27b-ud-q6   # Required. The configured model that answers.
      memory: false              # With true, the id gets its own store.
      mcp: null                  # null uses assistant.mcp. [] gives no tools.

models:                          # The entry that gmlx pull wrote.
  qwen3.8-27b-ud-q6:
    path: unsloth__Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf

assistant:
  mcp:
    - name: clock
      command: [uvx, mcp-server-time]
```

```sh
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "helper",
  "messages": [{"role": "user", "content": "What time is it?"}]
}'
```

At start, the server prints a line such as
`[server] assistant 'helper' -> qwen3.8-27b-ud-q6  tools: ...` for each id.

- Assistant ids work only on `/v1/chat/completions`.
- A request that sends its own `tools` goes to the underlying model
  unchanged, so a client with its own tool loop keeps it.
- `max_tokens` limits each round, and defaults to 4096.
- A streaming reply carries a comment line such as `: assistant using NAME`
  for each tool call.

With `memory: true`, each id has one store that every client of the id
shares. That suits a personal server, not one with several users.

## Security

`gmlx talk` and `gmlx chat --assistant` run tools on your Mac, as you. A
served assistant runs tools on the server host for anyone who can reach the
server. On the default loopback address, that is only your Mac.

- Tools come only from the configuration file. A request cannot add MCP
  servers or change tools.
- On an address other than loopback, a server with `server.assistants`
  refuses to start unless
  [`server.assistant_allow_remote`](config.md#serverassistant_allow_remote)
  is `true`. It also needs an API key, unless
  [`server.no_auth`](config.md#serverno_auth) is set.
- With `assistant_allow_remote: true` and tools in `assistant.mcp`, each
  assistant must list its own `mcp`, with `[]` for no tools, so you choose
  each one's tools.
- `gmlx doctor` warns when assistants are served beyond loopback, and says
  which tools each one has.
