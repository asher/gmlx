# A local RAG stack

This guide is for building retrieval-augmented generation on the server.
RAG needs two services beside the chat model: an embedder that indexes
documents and queries as vectors, and an optional reranker that re-scores
the vector search's shortlist. The server provides both as OpenAI- and
Cohere-compatible endpoints, from GGUF models, on the same port as chat.

## Enable the endpoints

```sh
gmlx init --models-dir ~/models --with-embeddings --with-rerank
gmlx serve
```

The two `--with-*` flags add the services to a scaffolded config, which
needs a models directory, or the interactive `gmlx init` offers both as
steps. Either way the config gains the two keys with the default models:

```yaml
server:
  embeddings: qwen3-embed-0.6b     # POST /v1/embeddings
  rerank: qwen3-rerank-0.6b        # POST /v1/rerank
```

The default GGUFs, about 0.6 GB each, resolve from your local Hugging Face
cache only, so fetch them first with `gmlx pull`. Both services load in the
background at startup and sit outside the chat residency pool, which
[services.md](services.md) describes along with the other behavior the
four services share.

## Choosing the models

The default embedder, `qwen3-embed-0.6b`, embeds a document in one pass
over the default model's 32k context, so long documents are not truncated.
`qwen3-embed-4b` and `-8b` retrieve better at the cost of a bigger index,
any local or `hf:` GGUF can be named instead, and there are encoder options
for particular languages or sizes. The reranker is `qwen3-rerank-0.6b` by
default, with `-4b` and `-8b` above it. The alias tables, dimensions,
context windows and how each model scores are under
[Text embeddings](services.md#text-embeddings-embeddings) and
[Reranking](services.md#reranking-rerank).

## The embeddings endpoint

The endpoint has the OpenAI shape, where `input` is a string or a list of
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

The response echoes the `model` name you sent, which can be any of the
conventional OpenAI embedding names or omitted. An optional
`encoding_format` selects `float`, the default, or `base64`.

## The rerank endpoint

`POST /v1/rerank`, also served at `/rerank`, has the Cohere and Jina shape.
Send the query and the candidate documents to get back indices sorted
best-first with relevance scores:

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

The optional fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `documents[]` | | strings, or `{"text": ...}` objects |
| `top_n` | all | how many results to return |
| `return_documents` | `true` | echo each result's document. Set it false for indices and scores only |
| `instruction` | the model's | replaces the query instruction the reranker is prompted with |

Scoring runs a model forward for each document, so keep the candidate list
to a vector search's shortlist of tens, not thousands.

## Configure Open WebUI

`gmlx launch open-webui` points Open WebUI's document embedder at this
server, points its external reranker here when the server advertises one
and enables the hybrid search mode that reranking requires. Upload
documents in Open WebUI and query them, and indexing and retrieval happen
against your server. The variables it sets are listed under
[open-webui](launch.md#open-webui).

## Other consumers

Any OpenAI-compatible RAG framework connects in the same way, with base
URL `http://127.0.0.1:8080/v1` and any API key unless the server sets one.
The built-in assistant's [long-term memory](assistant.md#memory) embeds
remembered facts through this endpoint and reorders recall through
`/v1/rerank`. To give the assistant retrieval over your documents as a tool
it can call, add a vector-store MCP server from the
[tool examples](assistant.md#tool-examples).
