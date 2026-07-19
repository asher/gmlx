# The assistant: tools and memory

gmlx has a built-in assistant: a tool loop with long-term memory, wrapped
around the server's own chat completions. Configure it once in a top-level
`assistant:` block and it is available from three surfaces:

- **Voice**: `gmlx talk` with `talk.brain: assistant`
  ([talk.md](talk.md#worked-example-the-assistant) has the worked example).
- **Text**: `gmlx chat --assistant`, the same engine in the chat REPL.
- **API**: served assistant ids (`server.assistants:`) that run the loop
  server-side for any OpenAI client.

It is deliberately a lightweight assistant, not an autonomous agent. Each turn
runs a bounded tool loop and ends when the model answers; nothing keeps working
in the background afterward. It is built for errand-sized tasks: look something
up, chain a few tool calls, write a note, remember a fact. The external coding
agents `gmlx launch` connects (pi, opencode, and friends) are a different
thing entirely -- they bring their own loops and tools and use this server for
inference only.

Tool support needs the MCP SDK:

```sh
pip install 'gmlx[assistant]'
```

## The tool loop

A turn is the standard OpenAI tool loop, run as a client of the server's own
`/v1/chat/completions`: the user text goes out with the configured tools
attached; if the model answers with tool calls instead of prose, the assistant
executes them and sends the results back; repeat until the model answers. Up
to `max_tool_rounds` rounds may call tools, then one final tool-less request
forces an answer, so a tool-happy model cannot loop forever. A failed call
(unknown tool, malformed arguments, timeout) comes back to the model as an
error string it can retry. Completed tool rounds stay in the conversation
history, so later rounds and later turns can build on earlier results.

Because every round is an ordinary chat-completion request, whatever the
server applies per request -- sampling profiles, speculative decoding, the
prompt cache -- applies to assistant rounds too. Each tool round costs a model
turn plus the tool call, so multi-tool answers are noticeably slower than
plain chat.

Tool calling needs a model that is competent at it. The Qwen3.6-27B class is a
good fit on a 48 GB or larger machine; Qwen3.5-9B is a workable floor on
32 GB, with more tool fumbles.

## The `assistant:` block

The full block, with defaults:

```yaml
assistant:                    # the built-in tool-loop assistant used by talk,
                              # chat --assistant, and server.assistants -- not
                              # the coding agents `gmlx launch` connects
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

Each `mcp:` entry is a stdio server (`command` is the argv to spawn, plus an
optional `env` map) or a streamable-HTTP endpoint (`url`) -- exactly one of
the two. Tool-name collisions across servers get a server-name prefix. An MCP
server that fails to come up degrades to a warning rather than blocking the
loop, and a missing `[assistant]` extra does the same with an install hint. A
stdio server's stderr goes to a per-server log in the cache directory
(`~/.cache/gmlx/mcp-<name>.log`), never into the REPL or voice UI.

## Tool recipes

Any stdio or streamable-HTTP MCP server plugs in the same way; these are
starting points that need no further setup (third-party package names, current
as of this writing).

**Web search, no API key.** DuckDuckGo's search endpoint is keyless, and this
community server wraps it with a `search` tool plus a page fetcher, so the
assistant can look something up and then read the page it found:

```yaml
assistant:
  mcp:
    - name: web
      command: [uvx, duckduckgo-mcp-server]
```

**Web search, self-hosted.** If you already run [SearXNG](https://docs.searxng.org),
point the assistant at it: aggregated results from many engines, still keyless
and fully local, and none of the bot-detection fragility of scraping a public
endpoint. (The instance must allow the JSON format: add `json` to
`search.formats` in its `settings.yml`.)

```yaml
assistant:
  mcp:
    - name: search
      command: [npx, -y, mcp-searxng]
      env: {SEARXNG_URL: "http://127.0.0.1:8888"}
```

**Web search with an API key.** Brave's official server returns richer
results (web, news, images) on a free-tier key, and shows where `env:` fits:

```yaml
assistant:
  mcp:
    - name: brave
      command: [npx, -y, "@brave/brave-search-mcp-server"]
      env: {BRAVE_API_KEY: your-key-here}
```

A stdio tool server runs with a minimal environment (`HOME`, `PATH`, `SHELL`,
`TERM`, `USER`, `LOGNAME`); `env:` adds to that, and nothing else from this
process's environment is inherited. Pass a secret a server needs explicitly, as
above -- an `HF_TOKEN` or `OPENAI_API_KEY` sitting in your shell never reaches
third-party tool code.

**Document retrieval (RAG as a tool).** Qdrant's official server in embedded
local mode gives the assistant `qdrant-store` and `qdrant-find` over a vector
collection on disk -- no database process to run. Ask the assistant to store
passages and it can retrieve them semantically later; embedding happens
inside the tool server with its own small local model, independent of this
server's `/v1/embeddings`:

```yaml
assistant:
  mcp:
    - name: docs
      command: [uvx, mcp-server-qdrant]
      env: {QDRANT_LOCAL_PATH: ~/vectors, COLLECTION_NAME: notes}
```

The division of labor: [memory](#memory) below is automatic and personal --
distilled facts about you, recalled every turn. A retrieval tool is deliberate
and document-shaped -- the model decides when to search, and over what you
loaded. For document RAG in a chat UI rather than through the assistant (Open
WebUI uploads against this server's own embeddings), see [rag.md](rag.md).

## Memory

Memory is a local retrieval store over the server's own endpoints: facts are
embedded via `/v1/embeddings` into a sqlite file, and every turn recalls the
closest ones (reordered by `/v1/rerank` when configured) and injects them as
transient context. They never bloat the rolling chat history. Without
`server.embeddings:` the assistant still runs, just memoryless, after one
warning.

What gets stored is a distilled fact ("sister Ana, birthday March 12"), not a
transcript: after each turn, a background request asks the chat model to boil
the exchange down to at most three durable facts, or none, so small talk
leaves no residue. A new fact that restates an existing one replaces it.
`extract: false` stores raw user and assistant exchanges instead. `ttl_days`
expires old rows at startup; `max_items` caps the store, evicting the
never-recalled oldest rows first.

The store is shared by design between the voice and text surfaces: what you
tell the assistant in `gmlx talk` it remembers in `gmlx chat --assistant`,
and vice versa. Inside either, `/memory` lists the stored facts with their
ids, `/memory forget ID` removes one, and `/memory clear yes` removes them
all. The file itself sits at `~/.local/share/gmlx/assistant-memory.db` if
you want to look deeper. Served assistants that enable memory get their own
per-id store (`assistant-<id>.db`), never this one.

## Text chat: `gmlx chat --assistant`

`--assistant` switches the chat REPL's turn engine from a local model load to
the assistant on the managed server (auto-started if down, like `talk`). The
positional argument is a served model id, or omitted for the server's default
model; a file path is refused, since the server owns the model. Tool activity
appears as transient status lines while the answer streams.

```sh
gmlx chat --assistant                 # server default model
gmlx chat qwen3.6-27b --assistant     # a specific served model
```

The terminal experience is unchanged: markdown rendering, reasoning display,
themes, history, sessions and `--resume`, `/system`, `/retry` and `/undo` (a
retry or undo rewinds whole tool rounds, so the history never holds a
half-finished tool exchange), plus `/memory` as above. Sampling flags and
their `/commands` forward to the server per round once you touch them;
untouched knobs stay on the server's own defaults. Flags that only make sense
for a local load are rejected (`--adapter`, `--mmproj`, chat-template flags)
or ignored with a printed note (loading, KV-cache, speculative/MTP, CPU
placement); `/image`, `/audio` and the thinking budget are not available
in this mode. `--base-url`, `--api-key`, `--no-start`, and `--start-timeout`
target a remote or already-running server, exactly as in `talk`.

## Served assistants

`server.assistants:` exposes pseudo-model ids that run the tool loop
server-side, behind plain chat completions. Thin clients -- curl, Open WebUI,
a phone app, a shell script -- pick the assistant id as their model and get
MCP tools with no client-side loop at all:

```yaml
server:
  model_dirs: [~/models]
  assistants:
    helper:              # served id: shows up in /v1/models
      model: qwen3.6-27b # required: the underlying configured model
      memory: false      # off by default; true = ONE shared store for every
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

The routing contract, per request to `/v1/chat/completions`:

- **Assistant id, no `tools` in the request** (an empty `tools: []` counts):
  the server runs the loop and returns only the prose answer, under the
  assistant id. On a streaming request, tool rounds keep the stream alive
  with SSE comments naming the tool in use; the final round's deltas stream
  as normal chunks.
- **Assistant id, non-empty `tools`**: the client is running its own loop.
  The model id is rewritten to the underlying model and the request passes
  through untouched -- the config's tools are not offered, and the client
  gets its `tool_calls` back as usual. Client loops always win; the server
  never double-wraps one.
- **Any other model id**: untouched.
- `/v1/responses` and `/v1/messages` reject assistant ids with a 400
  ("assistant models are chat-completions only"); `/v1/models` lists them.

Reported usage sums `completion_tokens` across all rounds; `prompt_tokens` is
the final round's. Concurrent assistant turns are capped (currently 4 per
server); a request over the cap gets an immediate 429 rather than queueing.
Three known limits: a `stop` sequence forwards to every round and can in
principle truncate an intermediate tool round; a non-streaming assistant turn
cannot be cancelled by client disconnect (it runs its rounds to completion);
stream-path cancellation lands at the next delta or tool boundary, so a
cancel mid-tool can take up to `tool_timeout_s` to bite.

## Security

The three surfaces run tools in two different places, and the configuration
is deliberately strict about the difference:

- `gmlx talk` and `gmlx chat --assistant` execute MCP tools **on the
  machine you are sitting at, as you**. That is the trust domain of any local
  CLI tool.
- A served assistant executes MCP tools **on the server host**, reachable by
  anything that can reach the server. On the default loopback bind that is
  still just your machine. Beyond loopback it means anyone holding the API
  key can drive tool execution on the server host.

The gates that follow from this:

- Tools come only from the config. There is no way for a request to supply
  MCP servers or redefine tools; a request's own `tools` array switches the
  server loop off entirely (passthrough, above).
- A non-loopback bind with `server.assistants` configured refuses to start
  unless `server.assistant_allow_remote: true` -- on top of the existing rule
  that a non-loopback bind requires an `api_key`.
- Even with `assistant_allow_remote: true`, a remote-exposed assistant may
  not silently inherit the shared `assistant.mcp` list: each one must declare
  an explicit `mcp:` list (`[]` for a tool-less loop), so exposing tools to
  the network is always a deliberate, per-assistant decision.
- `gmlx doctor` warns when assistants are served beyond loopback and names
  each one's tool scope, so the exposure is visible at a glance.

Memory on a served assistant is one shared store across every client of that
id -- fine for a personal server, wrong for anything multi-user, which is why
it defaults off.
