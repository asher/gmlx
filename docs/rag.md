# A local RAG stack

Retrieval-augmented generation needs two services next to the chat model: an
embedder to index documents and queries into vectors, and (optionally but usefully)
a reranker to re-score the vector search's shortlist. The gmlx server provides
both as OpenAI- and Cohere-compatible endpoints, from GGUF models, on the same port
as chat. Nothing in the stack touches a cloud API.

This guide stands the stack up and wires Open WebUI to it. If you are building your
own pipeline instead, the two curl sections show the exact request and response
shapes.

## Enable the endpoints

```sh
gmlx init --with-embeddings --with-rerank    # or take both steps in the wizard
gmlx serve
```

That writes the two keys into the config's `server:` block with the default models:

```yaml
server:
  embeddings: qwen3-embed-0.6b     # POST /v1/embeddings
  rerank: qwen3-rerank-0.6b        # POST /v1/rerank
```

The GGUF defaults shown above (`qwen3-embed-0.6b`, `qwen3-rerank-0.6b`, about 0.6 GB
each) resolve from your local Hugging Face cache only -- fetch them first with
`gmlx pull` (or `gmlx sync-models`); a server that starts without them in cache
disables the endpoint until the file is present. The one exception is the
mlx-embeddings safetensors encoders (`embeddinggemma`, `arctic-l`, `nomic-embed`,
`bge-m3`): those download once on a cache miss and load locally from then on (rerank
is GGUF-only, so it never auto-downloads). Both endpoints are pre-warmed in the
background at startup and cached in-process, separate from the chat residency pool,
so a re-index and chat never evict each other. Requests for either run in a worker
thread that interleaves with, rather than blocking, batched chat decode.

## Choosing the models

The default embedder, `qwen3-embed-0.6b`, is a Qwen3-Embedding GGUF run as a
decoder-LM embedder: last-token pooling over the ordinary Qwen3 trunk, so it loads
like any other GGUF and carries the model's full 32k context. Long documents embed
without truncation, which BERT-window encoders cannot offer. Step up to
`qwen3-embed-4b` or `-8b` for better retrieval at a bigger index, or point the key
at any local or `hf:` GGUF ref. Encoder options exist too: `embeddinggemma-gguf`
(EmbeddingGemma from a GGUF) and several mlx-embeddings safetensors encoders
(`embeddinggemma`, `arctic-l`, `nomic-embed`, `bge-m3`). The full alias tables with
dimensions and context windows:
[server-config.md](server-config.md#text-embeddings-embeddings).

The reranker is a Qwen3-Reranker GGUF: a causal LM fine-tuned to answer yes or no to
"does this document satisfy this query", scored as the probability of yes. Aliases
`qwen3-rerank-0.6b` / `-4b` / `-8b`, or any GGUF ref.

## The embeddings endpoint

OpenAI shape. `input` is a string or list of strings; vectors come back
L2-normalized:

```sh
curl localhost:8080/v1/embeddings -H 'content-type: application/json' \
  -d '{"model": "text-embedding-3-small", "input": ["hello", "world"]}'
```

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.0789, ...]}
  ],
  "model": "text-embedding-3-small",
  "usage": {"prompt_tokens": 2, "total_tokens": 2}
}
```

The conventional OpenAI model names (`text-embedding-3-small`, `-3-large`,
`-ada-002`) all map to the configured model, and you can omit the field. Any other
requested model is refused rather than silently substituted. The response `model`
echoes the name you requested (or `text-embedding-3-small` when you omit it), not the
configured id. An optional `encoding_format` selects `float` (default) or `base64`
vectors.

## The rerank endpoint

Cohere/Jina shape on `POST /v1/rerank` (also `/rerank`). Send the query and the
candidate documents; get back indices sorted best-first with relevance scores:

```sh
curl localhost:8080/v1/rerank -H 'content-type: application/json' \
  -d '{
    "query": "how do I cancel my subscription?",
    "documents": ["Billing FAQ ...", "Setup guide ...", "Refund policy ..."],
    "top_n": 2
  }'
```

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.93, "document": {"text": "Billing FAQ ..."}},
    {"index": 2, "relevance_score": 0.71, "document": {"text": "Refund policy ..."}}
  ],
  "model": "reranker",
  "usage": {"total_tokens": 41}
}
```

`documents` entries may also be `{"text": ...}` objects. `return_documents` **defaults
to true**, so each result echoes its document text back (as above); set
`return_documents: false` to return indices and scores only. The response `model`
echoes the name you requested, or `reranker` when you omit it. An optional
`instruction` overrides the default query instruction. Scoring is one model forward
per document, so keep the candidate list to a vector search's shortlist (tens, not
thousands).

## Wire up Open WebUI

`gmlx launch open-webui` does the wiring for you: it points Open WebUI's
document-RAG embedder at this server, points the external reranker here too when
the server advertises one, and enables the hybrid search mode that reranking
requires. Upload documents in Open WebUI and query them; indexing and retrieval all
happen against your server. Details of exactly which variables it sets:
[launch.md](launch.md#open-webui).

## Other consumers

Any OpenAI-compatible RAG framework points here the same way: base URL
`http://127.0.0.1:8080/v1`, any API key unless the server sets one. The built-in
assistant's long-term memory is a built-in consumer: it embeds remembered facts
through this same `/v1/embeddings` endpoint and reorders recall through
`/v1/rerank` when configured ([assistant.md](assistant.md#memory)). To give the
assistant retrieval over your own documents as a tool it can call, add a
vector-store MCP server ([assistant.md](assistant.md#tool-recipes)).
