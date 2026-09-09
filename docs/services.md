# Services

The server can host speech-to-text, text-to-speech, embeddings and
reranking services beside chat models. This page is the reference for each
service's config value, aliases and endpoint. [rag.md](rag.md) and
[talk.md](talk.md) show them in use, and `gmlx launch open-webui` wires all
four into Open WebUI, as [launch.md](launch.md#open-webui) describes.

The four services share their runtime behaviour. Each configured model is
warmed in the background at startup and then cached in-process, and if the
warm-up fails the first request loads it. None of them counts against
`budget_gb` or lives in the chat residency pool, so a RAG re-index and chat
never evict each other. Requests run serialized with each other in a worker
thread for that service, which interleaves with batched LLM decode.

A request's `model` field may be omitted, may be `default`, or may be one
of the names OpenAI clients conventionally send, all of which map to the
configured model. Any other name is refused with a 400, which keeps clients
from making the server download arbitrary repos. `/v1/models` advertises
each service that is on under the first name in its row.

| Service | Names accepted in `model` |
|---------|---------------------------|
| speech-to-text | `whisper-1` |
| text-to-speech | `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` |
| embeddings | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |
| rerank | advertised as `reranker`. Any name is accepted and echoed back, since one reranker is served |

## Speech-to-text (`stt:`)

`server.stt:`, or `--stt` in any serve mode, adds an OpenAI-compatible
`POST /v1/audio/transcriptions` endpoint backed by
[mlx-whisper](https://pypi.org/project/mlx-whisper/). It needs the `stt`
extra and `ffmpeg` on PATH for audio decoding:

```sh
uv tool install "gmlx[stt]"      # or: pip install "gmlx[stt]"
brew install ffmpeg
```

Whisper checkpoints are not GGUFs, since whisper.cpp uses a separate ggml
container, so the server loads them in MLX format.

```yaml
server:
  stt: whisper-turbo    # or: whisper-turbo-q4 | an HF repo id | a local model dir | true
```

The value is an alias, an HF repo in MLX-whisper format, or a local
converted model directory, and `true` means the default alias. To pick a
precision the alias does not offer, name the exact repo.
`stt: mlx-community/whisper-large-v3-turbo` is the fp16 turbo.

| Alias | Repo | Notes |
|-------|------|-------|
| `whisper-turbo` | `mlx-community/whisper-large-v3-turbo` | the default. large-v3 quality at about 6x the speed, fp16, about 1.6 GB |
| `whisper-turbo-q4` | `mlx-community/whisper-large-v3-turbo-q4` | 4-bit, about 600 MB |
| `whisper-large` | `mlx-community/whisper-large-v3-mlx` | full large-v3 |
| `whisper-medium`, `-small`, `-base`, `-tiny` | `mlx-community/whisper-<size>-mlx` | smaller and faster |

The configured model is fetched from Hugging Face on first use when it is
not already local. Naming it in the config is the opt-in, and the
no-download policy for chat models is unchanged.

Requests follow the OpenAI shape, `multipart/form-data` with `file` plus the
optional fields `model`, `language`, `prompt`, `temperature` and
`response_format`. `response_format` takes `json`, `text`, `verbose_json`,
`srt` or `vtt`:

```sh
curl localhost:8080/v1/audio/transcriptions -F file=@clip.ogg -F model=whisper-1
```

The same `stt:` model also serves `POST /v1/audio/translations`, Whisper's
built-in translate task, which takes audio in any language and returns
English text. The request is the same multipart form without `language`,
which the OpenAI translations endpoint does not take:

```sh
curl localhost:8080/v1/audio/translations -F file=@japanese.ogg -F model=whisper-1
```

## Text-to-speech (`tts:`)

`server.tts:`, or `--tts` in any serve mode, adds an OpenAI-compatible `POST
/v1/audio/speech` endpoint backed by
[mlx-audio](https://pypi.org/project/mlx-audio/). It needs the `tts` extra,
plus `ffmpeg` on PATH for formats other than WAV, which encodes through
miniaudio:

```sh
uv tool install "gmlx[tts]"      # or: pip install "gmlx[tts]"
brew install ffmpeg
```

Like Whisper, TTS checkpoints are not GGUFs and load in MLX format.

```yaml
server:
  tts: kokoro    # or: kokoro-8bit | qwen3-tts | an HF repo id | a local model dir | true
```

The value is an alias, an HF repo in MLX-audio format, or a local
converted model directory, and `true` means the default alias.

| Alias | Repo | Notes |
|-------|------|-------|
| `kokoro` | `mlx-community/Kokoro-82M-bf16` | the default. 82M parameters, 24 kHz, Apache license, 54 voices |
| `kokoro-8bit`, `kokoro-4bit` | `mlx-community/Kokoro-82M-8bit`, `-4bit` | smaller |
| `qwen3-tts` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | larger, multilingual, named voices |
| `qwen3-tts-small` | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | smaller Qwen3-TTS |

Requests follow the OpenAI shape, a JSON body with `input` plus the
optional fields `model`, `voice`, `speed` from 0.25 to 4.0 and
`response_format`, where `response_format` takes `mp3`, the default, `wav`,
`flac`, `opus` or `pcm`. `voice` defaults to Kokoro's `af_heart`:

```sh
curl localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"tts-1","input":"Hello from MLX.","voice":"af_heart"}' -o out.mp3
```

When TTS is configured the server also answers `GET /v1/audio/voices` with
the configured model's voice names, which is where a client such as
`gmlx talk` fills its `/voice` list. Kokoro-style repos enumerate their
`voices/` directory once the model is local, qwen3-tts models return their
named-speaker set and unknown models return an empty list:

```sh
curl localhost:8080/v1/audio/voices
# {"model": "mlx-community/Kokoro-82M-bf16", "voices": ["af_alloy", ...], "default": "af_heart"}
```

A client that gets a 404 here, from an older server or one without TTS,
passes voice names through unchecked. Open WebUI's default voice is an
OpenAI name that Kokoro rejects, so the launch pins `af_heart`. A TTS model
other than Kokoro needs one of its own voices set instead, as
[launch.md](launch.md#open-webui) explains.

## Text embeddings (`embeddings:`)

`server.embeddings:`, or `--embeddings` in any serve mode, adds an
OpenAI-compatible `POST /v1/embeddings` endpoint, so that Open WebUI and
other OpenAI clients have a local RAG embedder. No extra is needed for any
backend.

The form of the value picks one of three backends.

- A GGUF decoder embedder. The value is a `*.gguf` path, an
  `hf:<org>/<repo>/<file>.gguf` ref, or a `qwen3-embed-*` alias for
  Qwen3-Embedding `0.6b`, `4b` or `8b`. These are the Qwen3 dense decoder
  trunk plus last-token pooling and an L2 norm. The runtime loads them like
  any other GGUF. They carry the model's full 32k to 40k context, so long
  documents embed without truncation.
- A GGUF encoder. `embeddinggemma-gguf` runs an EmbeddingGemma GGUF as a
  bidirectional sentence encoder, with mean pooling and a dense head, on
  this runtime's loader.
- A safetensors encoder through
  [mlx-embeddings](https://pypi.org/project/mlx-embeddings/), a core
  dependency. This covers the Gemma3, XLM-RoBERTa and ModernBERT encoders
  in the second table.

```yaml
server:
  embeddings: qwen3-embed-0.6b                  # the default, a GGUF decoder embedder
  # embeddings: hf:Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q6_K.gguf   # a specific quant
  # embeddings: ~/models/Qwen3-Embedding-4B.Q6_K.gguf   # a local GGUF
  # embeddings: embeddinggemma-gguf             # EmbeddingGemma encoder from a GGUF
  # embeddings: embeddinggemma                  # EmbeddingGemma encoder from safetensors
```

The value is a GGUF ref, an alias, an HF repo in MLX-embeddings format, or
a local converted model directory, and `true` means the default alias. A
bare alias resolves to the default quant shown in the tables, while the
`gmlx init` wizard offers a follow-up to pick another quant and writes its
concrete ref. A GGUF ref, whether written out or reached through an alias,
resolves from the local Hugging Face cache only and never the network, so
fetch it with `gmlx pull` first. A miss at startup disables the endpoint
until the file is present. The safetensors encoders download once on a
cache miss, like the speech models.

GGUF embedders, where dim is the vector width and ctx the largest input in
tokens:

| Alias | Repo and default quant | dim / ctx | Notes |
|-------|------------------------|-----------|-------|
| `qwen3-embed-0.6b` | `Qwen/Qwen3-Embedding-0.6B-GGUF`, Q8_0 | 1024 / 32k | the default. Small, fast and multilingual, about 0.6 GB |
| `qwen3-embed-4b` | `Qwen/Qwen3-Embedding-4B-GGUF`, Q8_0 | 2560 / 40k | higher retrieval quality, about 4.3 GB |
| `qwen3-embed-8b` | `Qwen/Qwen3-Embedding-8B-GGUF`, Q8_0 | 4096 / 40k | highest quality of the family, about 8 GB, with the largest index |
| `embeddinggemma-gguf` | `ggml-org/embeddinggemma-300M-GGUF`, Q8_0 | 768 / 2k | encoder, a small multilingual model from Google, about 0.3 GB |

The bare `qwen3-embed` is a back-compat alias for `qwen3-embed-0.6b`.

Safetensors encoders, default quant `8bit`:

| Alias | Repo | dim / ctx | Notes |
|-------|------|-----------|-------|
| `embeddinggemma` | `mlx-community/embeddinggemma-300m-8bit` | 768 / 2k | a small, high-quality multilingual model from Google, about 0.3 GB |
| `arctic-l` | `mlx-community/snowflake-arctic-embed-l-v2.0-8bit` | 1024 / 8k | multilingual long-context XLM-RoBERTa |
| `nomic-embed` | `mlx-community/nomicai-modernbert-embed-base-8bit` | 768 / 8k | widely used long-context English ModernBERT |
| `bge-m3` | `mlx-community/bge-m3-mlx-8bit` | 1024 / 8k | multilingual long-context XLM-RoBERTa |

Pick from the GGUF tier unless you want one of the encoders in
particular, for its size or its language coverage.

Requests follow the OpenAI shape, a JSON body with `input` as a string or a
list of strings, plus the optional fields `model` and `encoding_format`,
where `encoding_format` takes `float`, the default, or `base64`:

```sh
curl localhost:8080/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"text-embedding-3-small","input":["hello","world"]}'
```

Vectors are L2-normalized, mean-pooled by the encoder backends and
last-token pooled by the GGUF decoder backend.

## Reranking (`rerank:`)

`server.rerank:`, or `--rerank` in any serve mode, adds a Cohere- and
Jina-shaped `POST /v1/rerank`, also served at `/rerank`. Reranking is the
second RAG stage: a vector search returns a coarse top-N, the reranker
re-scores those documents jointly against the query, and the highest-scored
few are sent to the model. Open WebUI calls it as an external reranker.

The model is a Qwen3-Reranker GGUF, a Qwen3 causal LM fine-tuned to answer yes
or no to whether a document satisfies a query. It loads like any other GGUF,
with no extra needed. Its relevance score is the probability it assigns to yes
over no, computed as `sigmoid(yes - no)`. BGE and Jina BERT cross-encoders,
which llama.cpp reranks through a classifier head, are not mlx-lm
architectures and are out of scope.

```yaml
server:
  rerank: qwen3-rerank-0.6b                      # the default, a Qwen3-Reranker GGUF
  # rerank: hf:mradermacher/Qwen3-Reranker-4B-GGUF/Qwen3-Reranker-4B.Q6_K.gguf   # a specific quant
  # rerank: ~/models/Qwen3-Reranker-4B.Q6_K.gguf   # a local GGUF
```

The value is a `qwen3-rerank-*` alias for `0.6b`, `4b` or `8b` at the default
quant `Q8_0`, a `*.gguf` path, or an `hf:<org>/<repo>/<file>.gguf` ref, and
it resolves from the local cache only, like a GGUF embedder. Although the
reranker is independent of the embedder, `gmlx init` defaults its quant to
the quant chosen for the embedder.

Requests have the Cohere and Jina shape. They take `query`, `documents` as
strings or `{"text": ...}` objects and the optional fields `top_n`,
`instruction` and `return_documents`. The response holds `results` sorted
best-first, each with `index` and `relevance_score`, plus `model` and
`usage`:

```sh
curl localhost:8080/v1/rerank -H 'content-type: application/json' \
  -d '{"query":"how do I cancel?","documents":["Billing FAQ ...","Setup guide ..."]}'
```

Scoring runs a model forward for each document, so keep the candidate
list to a vector search's shortlist of tens, not thousands.
