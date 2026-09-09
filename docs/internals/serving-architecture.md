# Serving architecture

How the gmlx server serves a loaded GGUF as a continuously batched HTTP
server, for contributors. This page covers the implementation. The config
surface is documented in [server-config.md](../server-config.md) and the
endpoints in [api.md](../api.md).

gmlx is a thin patch layer over stock mlx-vlm. It installs late-bound patches
over a small set of mlx-vlm seams and leaves the stock app, batching engine
and protocol handlers untouched. Loads route to the gmlx loader, which reads
GGUF bytes through mlx-kquant's C++ reader and swaps model leaves for K-quant
kernels. The stock engine then executes those kernels in its own forward
pass. There is no engine fork.

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
        WRAP["text -> mlx_vlm text_only.Model<br/>(VLM -> mlx_vlm vision/audio model class)"]
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

## Components

The loader, `gmlx.load_model`, parses the GGUF bytes and remaps tensor names
to the Hugging Face layout. It synthesizes the config and tokenizer, including
the chat template, builds the stock model class and swaps the quantized leaves
for K-quant modules. A VLM adds a second file containing the vision or audio
tower. The output is a model, config and tokenizer triple with no safetensors
round-trip.

An adapter wraps a text model in mlx-vlm's text-only model class, which
exposes the embedding and language-model interface the engine expects. It also
attaches stopping criteria to the tokenizer. VLM models are wrapped in their
mlx-vlm class instead. Wrapped models are held in a residency pool of pinned
and LRU entries that owns the single process-wide wired limit.

The engine is mlx-vlm's batch generator. It runs continuous batching over a
ragged KV cache, given embeddings that the request path precomputes. Prefix
reuse is the prompt cache manager, which picks a tier per architecture
([prompt-cache.md](prompt-cache.md)). Speculative decoding runs gmlx's own
verify round, which keeps the prompt cache available under a drafter
([speculative-batching.md](speculative-batching.md)).

Above the engine is mlx-vlm's FastAPI app, which serves OpenAI chat
completions, OpenAI Responses and Anthropic Messages, each with streaming.
Tool calls are extracted from the raw token stream by mlx-lm's tool parsers,
selected from the model's chat template and re-emitted in each protocol's
format. Each request's sampling parameters resolve through the config
precedence chain before generation, from the family's model-card defaults up
to the request's own fields ([Precedence](../server-config.md#precedence)).
Served assistant ids are handled in front of this layer. A request to one runs
the tool loop on a worker thread. Each round re-enters the server as an
ordinary loopback client ([served
assistants](../assistant.md#served-assistants)).

Clients are anything that implements either API. Pointing `ANTHROPIC_BASE_URL`
at the server lets Anthropic-API tools such as Claude Code use a local
model.

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

The patched seams are the residency lookup, the load call and the generation
argument builder. Everything between them is stock. The seam inventory and the
procedure for moving it to a new upstream release are in
[upstream-upgrades.md](upstream-upgrades.md).
