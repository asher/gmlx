# Assistant

The built-in assistant gives a served model tools and a long-term memory.
It runs a tool loop over MCP tools, keeps facts in a memory store, and
works in three places from one `assistant` block of the configuration
file:

| Surface | How to start it | Where tools run |
|---------|-----------------|-----------------|
| Voice | `gmlx talk` with `talk.brain: assistant`, as [The assistant by voice](talk.md#the-assistant-by-voice) shows. | On your Mac, as you. |
| Text | `gmlx chat --assistant`. | On your Mac, as you. |
| API | A served assistant id under `server.assistants`. | On the server host. |

Short tasks suit the assistant, such as looking something up, chaining a
few tool calls, writing a note or remembering a fact. A turn ends when the
model answers, and no work continues afterwards. The coding agents that
`gmlx launch` connects have loops of their own and use the server only for
the model. Tools need the `assistant` extra, which
[Optional features](installation.md#optional-features) shows how to
install.

- [The tool loop](#the-tool-loop)
- [Tools](#tools)
- [Tool examples](#tool-examples)
- [Memory](#memory)
- [Text chat](#text-chat)
- [Served assistants](#served-assistants)
- [Security](#security)

## The tool loop

A turn is the standard OpenAI tool loop, sent to the chat completions
endpoint of the server. Your message goes out with the tools attached.
When the model answers with tool calls instead of text, the assistant runs
them, sends the results back and asks again. This repeats until the model
answers in text. A failed call returns to the model as an error message,
so the model can try again. Tool calls run one at a time.

[`assistant.max_tool_rounds`](config.md#assistantmax_tool_rounds) limits
the rounds. After that many rounds of tool calls, the assistant sends a
last request with no tools, so the model must answer. A tool call that
takes longer than
[`assistant.tool_timeout_s`](config.md#assistanttool_timeout_s) returns a
timeout error to the model, and the tool keeps running in the background.

Each round is an ordinary chat completion request, so the profiles,
speculative decoding and prompt cache of the server apply to it. A round
costs a model reply plus the tool call, so an answer that needs several
tools takes longer than a plain reply. Choose a model that handles tool
calls well, such as Qwen3.8-27B, because smaller models make more mistakes
in their tool calls.

## Tools

Tools come from [MCP](glossary.md#mcp) servers, which are separate programs
that offer tools to a model through the Model Context Protocol. The
[`assistant.mcp`](config.md#assistantmcp) list names them. This block
connects one server that runs as a local command and one that answers over
HTTP:

```yaml
assistant:
  mcp:
    - name: files
      command: [npx, -y, "@modelcontextprotocol/server-filesystem", "/Users/me/notes"]
    - name: search
      url: http://127.0.0.1:8931/mcp
```

A server that fails to start gives a warning, and the assistant runs
without its tools. A server that has not connected after 20 seconds also
gives a warning, which names its log file. A command server gets only a
few variables from your environment, so pass a token that it needs with
[`env`](config.md#assistantmcpenv). `~` is not expanded in the command or
in `env` values, so write full paths. An HTTP server gets no custom
headers. `gmlx doctor` checks that the programs of the command servers
exist.

## Tool examples

Every stdio or streamable HTTP MCP server is configured in the same way.
Each example is a complete `assistant` block, and the first two need no
account or key.

### Web search without an API key

This community server searches with DuckDuckGo, which needs no key, and
also fetches pages:

```yaml
assistant:
  mcp:
    - name: web
      command: [uvx, duckduckgo-mcp-server]
```

### Web search on your own SearXNG

A [SearXNG](https://docs.searxng.org) instance that you run gives results
from several search engines and keeps the queries local. The instance must
allow the JSON format, so add `json` to `search.formats` in its settings:

```yaml
assistant:
  mcp:
    - name: search
      command: [npx, -y, mcp-searxng]
      env: {SEARXNG_URL: "http://127.0.0.1:8888"}
```

### Web search with an API key

Brave's official server gives richer results with a free API key, which
reaches it through `env`:

```yaml
assistant:
  mcp:
    - name: brave
      command: [npx, -y, "@brave/brave-search-mcp-server"]
      env: {BRAVE_API_KEY: your-key-here}
```

### Document search as a tool

The official Qdrant server, in local mode, gives the assistant tools to
store and find documents in a vector collection on disk, with no database
process to run. It embeds documents with its own small model, so it does
not need `server.embeddings`:

```yaml
assistant:
  mcp:
    - name: docs
      command: [uvx, mcp-server-qdrant]
      env: {QDRANT_LOCAL_PATH: /Users/me/vectors, COLLECTION_NAME: notes}
```

A search tool differs from memory. Memory is automatic and holds short
facts about you, recalled on every turn. A search tool works on documents
that you load, and the model decides when to use it. For document search
in a chat app instead, see [RAG pipelines](rag.md).

## Memory

The assistant remembers facts across conversations. After each turn, a
background request asks the chat model to reduce the exchange to at most
three lasting facts, such as "sister Ana, birthday March 12". Small talk
gives none, and nothing is stored. A new fact that repeats a stored one
closely replaces it. With
[`assistant.memory.extract: false`](config.md#assistantmemoryextract), the
assistant stores a short form of each exchange instead. When extraction
fails, it stores the exchange and prints one warning. Cancelled turns and
very short messages are not stored.

Before each turn, the assistant finds the facts closest to your message
through the `/v1/embeddings` endpoint of the server. It keeps up to
[`top_k`](config.md#assistantmemorytop_k) of them, and when more facts
match, the `/v1/rerank` endpoint chooses the best ones if the server runs
a reranker. The facts go into that request only, so the conversation does
not grow because of them.

Memory therefore needs [`server.embeddings`](config.md#serverembeddings).
Without it, the assistant prints a `memory disabled` warning the first
time that it uses memory, and continues without memory. Facts older than
[`ttl_days`](config.md#assistantmemoryttl_days) are removed at startup.
When the store holds more than
[`max_items`](config.md#assistantmemorymax_items) facts, those recalled
the fewest times go first, and the oldest go first among equals.

`gmlx talk` and `gmlx chat --assistant` share one store, so a fact that
you tell the assistant by voice is known in text chat too. The store is
`$XDG_DATA_HOME/gmlx/assistant-memory.db`, which is
`~/.local/share/gmlx/assistant-memory.db` by default, and
[`assistant.memory.path`](config.md#assistantmemorypath) moves it. In
either client, these commands manage the store:

| Command | Effect |
|---------|--------|
| `/memory` | Lists the newest 20 facts, with their ids. |
| `/memory forget ID` | Removes one fact. |
| `/memory clear yes` | Removes every fact. Without `yes`, it asks you to confirm. |

## Text chat

`gmlx chat --assistant` sends each turn to the assistant through the
server, and starts the server if it is down. Name a served model id, or no
model for the default model of the server. A file path is refused, because
the server owns the model:

```sh
gmlx chat --assistant
gmlx chat qwen3.8-27b-ud-q6 --assistant
```

The client works as [Chat](chat.md) describes, with the server-mode limits
listed in [Where the model runs](chat.md#where-the-model-runs). `/retry`
and `/undo` go back over whole tool rounds, and `/memory` is added. While
the model uses a tool, a status line shows `[assistant] using NAME...`.
The sampling that you set is sent with every round, as are `--thinking` and
`--reasoning-effort`. `/model <id>` switches to another served model and
keeps the conversation.

## Served assistants

A served assistant is a model id that runs the tool loop in the server.
Clients that cannot run a loop, such as curl, Open WebUI or a phone app,
choose the id as their model and get the tools. The
[`server.assistants`](config.md#served-assistants) block defines each id:

```yaml
server:
  model_dirs: [~/models]
  assistants:
    helper:                      # The served id, listed in /v1/models.
      model: qwen3.8-27b-ud-q6   # Required. The configured model that answers.
      memory: false              # With true, the id gets its own store.
      mcp: null                  # null uses assistant.mcp. [] gives no tools.
  assistant_allow_remote: false  # See Security.

models:
  qwen3.8-27b-ud-q6:
    path: Qwen3.8-27B-UD-Q6_K.gguf

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

At startup, the server connects the tools and prints a line such as
`[server] assistant 'helper' -> qwen3.8-27b-ud-q6  tools: ...` for each
id. `/v1/models` lists each id with `"assistant": true` and `alias_of`,
which names the model. An id may not contain `@` or match a model id or
alias. The server routes a request to `/v1/chat/completions` by its
`model` and by whether it carries tools of its own, where an empty `tools`
list counts as none:

| Request | Result |
|---------|--------|
| An assistant id with no `tools` | The server runs the loop and returns the answer under the assistant id. |
| An assistant id with `tools` | The client runs its own loop, so the server sends the request to the underlying model unchanged. |
| Any other id | The request is not changed. |
| An assistant id on `/v1/responses` or `/v1/messages` | A 400, because assistants work only through chat completions. |

The server builds each turn from the messages that the client sends, so
tool calls from earlier turns are not kept. The last message must be a
user message with text only, and earlier system messages are kept. The
answer joins the text of every round. A stream carries the reasoning of
the model and a comment line such as `: assistant using NAME` for each
tool, which also keeps the connection alive.

Every round uses the sampling fields and `stop` of the request, but not
`response_format`, because it would stop the model from calling tools.
`max_tokens` limits each round and defaults to 4096, and
`max_completion_tokens` wins over it. The reported usage adds up the
completion tokens of all rounds and gives the prompt tokens of the last
round.

A round that fails returns a 502 with the code `assistant_upstream_error`,
or an error object in a stream. The server runs at most 4 assistant turns
at a time, and a request over that limit gets an immediate 429. When a
streaming client disconnects, the turn stops at the next reply chunk or
tool call. A request without streaming runs to the end.

Memory on a served assistant is one store for each id, in
`assistant-<id>.db` beside the default memory file, and every client of
the id shares it. The store uses the `top_k`, `extract`, `ttl_days` and
`max_items` settings, and ignores `assistant.memory.enabled` and
`assistant.memory.path`. A shared store suits a personal server, not one
with several users.

## Security

`gmlx talk` and `gmlx chat --assistant` run tools on your Mac, as you,
like any other command-line tool. A served assistant runs tools on the
server host, for anyone who can reach the server. On the default loopback
address, that is still only your Mac. On any other address, anyone with
the API key can make the host run tools.

These rules limit what a served assistant can do:

- Tools come only from the configuration file. A request cannot add MCP
  servers or change tools, and a request with its own `tools` bypasses the
  loop.
- On an address other than loopback, a server with `server.assistants`
  refuses to start unless
  [`server.assistant_allow_remote`](config.md#serverassistant_allow_remote)
  is `true`. Such an address also needs an API key, unless
  [`server.no_auth`](config.md#serverno_auth) is set.
- With `assistant_allow_remote: true` and a non-empty `assistant.mcp`,
  each assistant must list its own `mcp`, with `[]` for no tools. The
  tools of each assistant are therefore a choice that you make in the
  file.
- `gmlx doctor` warns when assistants are served beyond loopback, and
  names the tools of each one.
