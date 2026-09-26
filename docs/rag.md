# RAG pipelines

Retrieval-augmented generation, or RAG, answers questions from your own
documents. An embedder turns the documents and each question into vectors,
a search finds the passages closest to the question, a reranker puts the
best of them first, and the chat model answers from those passages. The
server provides the embedder and the reranker on the same port as its chat
models.

## Setting up the services

Add an embedder and a reranker to the `server` block of the
[configuration file](config.md):

```yaml
server:
  embeddings: qwen3-embed-0.6b
  rerank: qwen3-rerank-0.6b
```

The server never downloads these GGUF models, so download them before it
starts:

```sh
gmlx pull hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf
gmlx pull hf:mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf
gmlx restart
```

A service whose model is missing when the server starts stays off until
the next restart, and the server prints a warning about it. To create a new
configuration file with both services, run
`gmlx init --models-dir ~/models --with-embeddings --with-rerank`, or answer
the questions of the `gmlx init` wizard. The larger models, the other kinds
of embedder and the request fields are in
[Speech, embeddings and rerank](services.md#embeddings).

## A pipeline in Python

This example indexes three passages, finds the best one for a question, and
answers from it. It uses the OpenAI client for embeddings and chat, and
httpx, which the OpenAI client installs, for reranking:

```python
import httpx
from openai import OpenAI

base = "http://127.0.0.1:8080/v1"
client = OpenAI(base_url=base, api_key="none")

docs = [
    "To cancel a subscription, open Billing and choose Plans.",
    "Invoices are emailed on the first day of each month.",
    "Install the desktop app from the Downloads page.",
]
question = "How do I cancel my plan?"

def embed(texts):
    reply = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in reply.data]

doc_vectors = embed(docs)
query_vector = embed([question])[0]

# The vectors have length 1, so the dot product is the cosine similarity.
scores = [sum(q * d for q, d in zip(query_vector, v)) for v in doc_vectors]
best = sorted(range(len(docs)), key=scores.__getitem__, reverse=True)[:2]
shortlist = [docs[i] for i in best]

ranked = httpx.post(f"{base}/rerank",
                    json={"query": question, "documents": shortlist, "top_n": 1}).json()
passage = shortlist[ranked["results"][0]["index"]]

answer = client.chat.completions.create(
    model="qwen3.8-27b-ud-q6",
    messages=[{"role": "system", "content": f"Answer from this passage:\n{passage}"},
              {"role": "user", "content": question}],
)
print(answer.choices[0].message.content)
```

A real index keeps the document vectors in a vector database instead of a
list, and embeds each document once. On a server with an
[API key](config.md#serverapi_key), pass the key to the OpenAI client and
as an `Authorization: Bearer` header to httpx.

## Open WebUI

`gmlx launch open-webui` points Open WebUI's document embedder at the
server. When the server runs a reranker, it also turns on Open WebUI's
hybrid search and points its external reranker at the server. Documents
that you upload in Open WebUI are then indexed and searched on your
server. [Open WebUI](launch.md#open-webui) lists what the launch sets.

## Other tools

Any RAG framework that supports an OpenAI-compatible embeddings endpoint
works with the base URL `http://127.0.0.1:8080/v1`, and any API key unless
the server has one. The built-in [assistant](assistant.md) uses the same
endpoints for its long-term memory. To give the assistant a search over
your documents as a tool, add a vector store as an MCP server, as
[Tool examples](assistant.md#tool-examples) shows.
