# Serving architecture

How the gmlx server serves a loaded GGUF as a continuously batched HTTP
server, for contributors. This page covers the implementation, while the
config surface is documented in [server-config.md](../server-config.md) and
the endpoints in [api.md](../api.md).

The server is stock mlx-vlm, with its app, batching engine and protocol
handlers untouched, and gmlx reaches into it through patched seams. Loads
route to the gmlx loader, which reads GGUF bytes through mlx-kquant's C++
reader and swaps model leaves for K-quant kernels, and the stock engine then
executes those kernels in its own forward pass. There is no engine fork.
The seam inventory, and why each one is fragile, is in
[upstream-upgrades.md](upstream-upgrades.md).

## From file to response

```mermaid
flowchart TD
    subgraph DISK["GGUF on disk"]
        LLM["text LLM GGUF<br/>K-quant: Q4_K / Q6_K / MXFP4 ..."]
        MM["mmproj GGUF<br/>float or Q8_0, VLM only"]
    end

    subgraph LOAD["Loader: gmlx.load_model"]
        direction TB
        PARSE["parse file bytes + GGUF->HF name remap"]
        SYNTH["config + tokenizer synth<br/>including the chat template"]
        BUILD["build stock model class"]
        KQ["install K-quant leaves<br/>KQuantLinear, gather_qmm, KQuantMultiLinear"]
        PARSE --> SYNTH --> BUILD --> KQ
    end
    LLM --> PARSE
    MM -. VLM .-> PARSE

    subgraph ADAPT["Model adapter + residency"]
        direction TB
        WRAP["text -> gmlx vendored text_only.Model<br/>(VLM -> mlx_vlm vision/audio model class)"]
        STOP["attach StoppingCriteria to tokenizer"]
        REG["multi-model residency pool<br/>pinned + LRU, one process wired_limit"]
        WRAP --> STOP --> REG
    end
    KQ --> WRAP

    subgraph ENGINE["Engine: mlx_vlm.generate.ar.BatchGenerator"]
        direction TB
        EMB["precompute inputs_embeds<br/>get_input_embeddings(input_ids)"]
        BG["continuous batching (embeds-in)<br/>insert(inputs_embeds) -> next()"]
        KVC["BatchKVCache (ragged, left-pad)"]
        APC["prompt cache<br/>exact / checkpoint / block tiers"]
        SAMP["sampler / logits procs / stop"]
        MTP["speculative draft / MTP"]
        EMB --> BG
        BG --- KVC
        BG --- APC
        BG --- SAMP
        BG -.- MTP
    end
    REG --> EMB

    subgraph PROTO["HTTP: mlx-vlm FastAPI"]
        direction TB
        TOOLS["tool-call extractor<br/>mlx_lm.tool_parsers (from chat template)"]
        ANTH["/v1/messages (Anthropic)"]
        OAI["/v1/chat/completions (OpenAI)"]
        RESP["/v1/responses (OpenAI Responses)"]
        SSE["streaming SSE formatters"]
        TOOLS --> ANTH & OAI & RESP
        ANTH --- SSE
        OAI --- SSE
        RESP --- SSE
    end
    BG --> TOOLS

    CC["Anthropic-API client<br/>such as Claude Code via ANTHROPIC_BASE_URL"]
    SDK["OpenAI SDK / curl / apps"]
    ANTH --> CC
    OAI --> SDK
    RESP --> SDK
```

## What the diagram leaves out

The loader's output is a model, config and tokenizer triple with no
safetensors round-trip. The text-only wrapper in the adapter stage is
gmlx's own copy of the class mlx-vlm removed in 0.6.15, vendored in
`gmlx/models/vlm_text_only.py`. Keeping it in-tree holds the embedding and
language-model interface the engine expects steady across upstream
releases.

The prompt cache picks its tier per architecture, as
[prompt-cache.md](prompt-cache.md) describes, and the verify round that
speculative decoding runs is gmlx's own, which is what keeps the prompt
cache usable under a drafter ([speculative-batching.md](speculative-batching.md)).

Two things happen before a request reaches the engine. Its sampling
parameters resolve through the config precedence chain, from the family's
model-card defaults up to the request's own fields
([Precedence](../server-config.md#precedence)). And a request to a served
assistant id never reaches the HTTP layer as itself: the tool loop runs on a
worker thread and each round re-enters the server as an ordinary loopback
client ([served assistants](../assistant.md#served-assistants)).

## The request path through the seams

```mermaid
sequenceDiagram
  participant C as Client
  participant A as mlx-vlm app
  participant R as residency pool
  participant S as serving (resolver + bridge)
  participant L as loader + kquant swap
  participant E as BatchGenerator (kq.* kernels)
  C->>A: POST /v1/chat (model "id@profile")
  A->>R: get_cached_model(id)   [patched]
  R->>S: resolve_request_model(id@profile)
  S-->>R: abspath + ResolvedModel, sets _active_spec
  alt resident (cache key includes the load parameters)
    R-->>A: model, processor, config
  else cold build
    R->>R: set load-param + APC env window
    R->>S: load_model_resources(path)   [patched]
    S->>L: load_model / load_mtp_model / load_vlm
    L-->>S: model (leaves = kq.* modules)
    S-->>R: model, processor, config
  end
  A->>A: _build_gen_args seeds sampling from the active profile   [patched]
  A->>E: generate(...)
  E-->>C: stream tokens
```

On this path the patched seams are the residency lookup, the load call and
the generation argument builder. Everything between them is stock.
