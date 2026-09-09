# A local RAG stack

This guide is for building retrieval-augmented generation on the server. RAG
needs two services beside the chat model. An embedder indexes documents and
queries as vectors, and an optional reranker re-scores the vector search's
shortlist. The server provides both as OpenAI- and Cohere-compatible
endpoints, from GGUF models, on the same port as chat.

## Enable the endpoints

```sh
gmlx init --with-embeddings --with-rerank    # or take both steps in the wizard
gmlx serve
```

That writes the two keys into the config with the default models:

```yaml
server:
  embeddings: qwen3-embed-0.6b     # POST /v1/embeddings
  rerank: qwen3-rerank-0.6b        # POST /v1/rerank
```

The default GGUFs, about 0.6 GB each, resolve from your local Hugging Face
cache only, so fetch them first with `gmlx pull`. A server that starts
without them disables the endpoint until the file is present. Both services
load in the background at startup. They sit outside the chat residency
pool, so a re-index and chat never evict each other, and they run in a
worker thread that interleaves with batched chat decode.

## Choosing the models

The default embedder is a Qwen3-Embedding GGUF run as a decoder embedder
with last-token pooling. It loads like any other GGUF and carries the
model's full 32k context, so long documents embed without truncation. Use
`qwen3-embed-4b` or `-8b` for better retrieval at a bigger index, or point
the key at any local or `hf:` GGUF. Encoder options exist too, including
EmbeddingGemma from a GGUF and several safetensors encoders that download
once on a cache miss. The alias tables with dimensions and context windows
are under [Text embeddings](services.md#text-embeddings-embeddings).

The reranker is a Qwen3-Reranker GGUF, a causal model fine-tuned to answer
yes or no to "does this document satisfy this query". The score is the
probability of yes. Aliases are `qwen3-rerank-0.6b`, `-4b` and `-8b`, or
any GGUF ref. [Reranking](services.md#reranking-rerank) has the details.

## The embeddings endpoint

The endpoint has the OpenAI shape. `input` is a string or a list of
strings, and vectors come back L2-normalized:

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

The conventional OpenAI model names all map to the configured model, and the
field can be omitted. Any other requested model is refused, never silently
substituted. The response echoes the name you requested. An optional
`encoding_format` selects `float`, the default, or `base64`.

## The rerank endpoint

`POST /v1/rerank`, also served at `/rerank`, has the Cohere and Jina
shape. Send the query and the candidate documents, and get back indices
sorted best-first with relevance scores:

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

`documents` entries may also be `{"text": ...}` objects. `return_documents`
defaults to true, so each result echoes its document. Set it false for
indices and scores only. An optional `instruction` overrides the default
query instruction. Scoring runs a model forward for each document, so keep
the candidate list to a vector search's shortlist of tens, not thousands.

## Configure Open WebUI

`gmlx launch open-webui` points Open WebUI's document embedder at this
server, points its external reranker here when the server advertises one,
and enables the hybrid search mode that reranking requires. Upload documents
in Open WebUI and query them. Indexing and retrieval happen against your
server. The variables it sets are listed under
[open-webui](launch.md#open-webui).

## Other consumers

Any OpenAI-compatible RAG framework connects in the same way, with base
URL `http://127.0.0.1:8080/v1` and any API key unless the server sets one.
The built-in assistant's [long-term memory](assistant.md#memory) embeds
remembered facts through this endpoint and reorders recall through
`/v1/rerank`. To give the assistant retrieval over your documents as a tool
it can call, add a vector-store MCP server from the
[tool examples](assistant.md#tool-examples).
