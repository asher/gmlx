# Services

The speech-to-text, text-to-speech, embeddings and reranking services the
server can host beside chat models. Each has a config block and an endpoint.

## Speech-to-text (`stt:`)

Setting `server.stt:` (or passing `--stt` in any serve mode) adds an
OpenAI-compatible `POST /v1/audio/transcriptions` endpoint backed by
[mlx-whisper](https://pypi.org/project/mlx-whisper/). It needs the optional
extra, `pip install 'gmlx[stt]'`, plus `ffmpeg` on PATH for audio
decoding. Whisper checkpoints are not GGUFs (whisper.cpp uses its own ggml
container), so this is the one model kind the server loads in MLX format
rather than from a GGUF.

```yaml
server:
  stt: whisper-turbo    # or: whisper-turbo-q4 | an HF repo id | a local model dir | true
```

The value is an alias, an HF repo in MLX-whisper format, or a local converted
model directory (`true` means the default alias). Aliases:

| Alias | Repo | Notes |
|-------|------|-------|
| `whisper-turbo` | `mlx-community/whisper-large-v3-turbo` | default; large-v3 quality, ~6x faster, fp16 ~1.6 GB |
| `whisper-turbo-q4` | `mlx-community/whisper-large-v3-turbo-q4` | 4-bit, ~600 MB |
| `whisper-large` | `mlx-community/whisper-large-v3-mlx` | full large-v3 |
| `whisper-medium` / `-small` / `-base` / `-tiny` | `mlx-community/whisper-<size>-mlx` (tiny: `whisper-tiny`) | smaller/faster |

Want full precision instead of a q4 (or vice versa)? Name the exact repo:
`stt: mlx-community/whisper-large-v3-turbo` is the fp16 turbo.

The Whisper model is pre-warmed in the background at startup (best-effort,
falling back to a lazy first-request load), then cached in-process. It is
tiny next to a resident LLM and does not count against `budget_gb`. Requests
follow the OpenAI shape (`multipart/form-data` with `file`, plus optional
`model`, `language`, `prompt`, `temperature`,
`response_format: json|text|verbose_json|srt|vtt`):

```sh
curl localhost:8080/v1/audio/transcriptions -F file=@clip.ogg -F model=whisper-1
```

Send `model=whisper-1` (or omit it); the conventional OpenAI name maps to the
configured model, and `/v1/models` advertises a `whisper-1` entry when STT is
on. Any other requested model is refused: clients can't make the server pull
arbitrary repos. Note that the configured `stt:` model itself is fetched from
Hugging Face on first use when it isn't already local; naming it in the
config is the opt-in. The LLM-side no-HF policy (below) is unchanged.
Transcriptions run serialized with each other in a worker thread,
interleaving with (not blocking) batched LLM decode.

The same `stt:` model also serves `POST /v1/audio/translations`: Whisper's
built-in `translate` task, which takes any-language audio and returns English
text. Same multipart request (minus `language`, which the OpenAI translations
endpoint doesn't take):

```sh
curl localhost:8080/v1/audio/translations -F file=@japanese.ogg -F model=whisper-1
```

---

## Text-to-speech (`tts:`)

Setting `server.tts:` (or passing `--tts` in any serve mode) adds an
OpenAI-compatible `POST /v1/audio/speech` endpoint backed by
[mlx-audio](https://pypi.org/project/mlx-audio/). It needs the optional
extra, `pip install 'gmlx[tts]'`, plus `ffmpeg` on PATH for non-WAV
formats (WAV encodes via miniaudio). Like Whisper, TTS checkpoints are not
GGUFs, so this is loaded in MLX format rather than from a GGUF.

```yaml
server:
  tts: kokoro    # or: kokoro-8bit | qwen3-tts | an HF repo id | a local model dir | true
```

The value is an alias, an HF repo in MLX-audio format, or a local converted
model directory (`true` means the default alias). Aliases:

| Alias | Repo | Notes |
|-------|------|-------|
| `kokoro` | `mlx-community/Kokoro-82M-bf16` | default; 82M, ~24 kHz, Apache, 54 voices |
| `kokoro-8bit` / `kokoro-4bit` | `mlx-community/Kokoro-82M-8bit` / `-4bit` | smaller |
| `qwen3-tts` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | larger, multilingual, named voices |
| `qwen3-tts-small` | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | smaller Qwen3-TTS |

The model is pre-warmed in the background at startup (best-effort, falling
back to a lazy first-request load), then cached in-process. Requests follow
the OpenAI shape (a JSON body with `input`, plus optional `model`, `voice`,
`speed` (0.25-4.0), and `response_format: mp3|wav|flac|opus|pcm`, default
`mp3`):

```sh
curl localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"tts-1","input":"Hello from MLX.","voice":"af_heart"}' -o out.mp3
```

Send `model=tts-1` (or `tts-1-hd`, or omit it); the conventional OpenAI names
map to the configured model, and `/v1/models` advertises a `tts-1` entry when
TTS is on. Any other requested model is refused. `voice` defaults to Kokoro's
`af_heart`. Synthesis runs serialized with each other in a worker thread,
interleaving with (not blocking) batched LLM decode.
`gmlx launch open-webui` wires the chat app's read-aloud (and mic STT) to
this endpoint when the server advertises it. The voice is pinned to
`af_heart` because Open WebUI's default (`alloy`) is an OpenAI voice Kokoro
rejects; a non-Kokoro `--tts` model needs `AUDIO_TTS_VOICE` overridden to one
of its own voices.

When TTS is configured the server also answers `GET /v1/audio/voices` with
the configured model's voice names (Kokoro-style repos enumerate their
`voices/` directory once the model is local; qwen3-tts models return their
named-speaker set; unknown models return an empty list):

```sh
curl localhost:8080/v1/audio/voices
# {"model": "mlx-community/Kokoro-82M-bf16", "voices": ["af_alloy", ...], "default": "af_heart"}
```

This is what `gmlx talk`'s `/voice` command lists from; clients treat a 404
(older server, or no TTS) as "no listing" and pass voice names through blind.

---

## Text embeddings (`embeddings:`)

Setting `server.embeddings:` (or passing `--embeddings` in any serve mode)
adds an OpenAI-compatible `POST /v1/embeddings` endpoint. The point of the
endpoint is to give Open WebUI (and other OpenAI clients) a local RAG
embedder: `gmlx launch open-webui` points the chat app's document-RAG here,
so nothing is downloaded from HuggingFace at its boot.

Three backends, chosen by what the value points at (none needs an optional
extra):

- GGUF decoder-LM embedder: a `*.gguf` path, an
  `hf:<org>/<repo>/<file>.gguf` ref, or a `qwen3-embed-*` alias
  (Qwen3-Embedding `0.6b`/`4b`/`8b`). These are the Qwen3 dense decoder
  trunk plus last-token (EOS) pooling and an L2-norm, so the runtime loads
  them like any other GGUF. They carry the model's full (32k-40k) context,
  so long documents embed without truncation.
- GGUF encoder: `embeddinggemma-gguf`, an EmbeddingGemma GGUF run as a
  bidirectional sentence encoder (mean-pool + dense head) on this runtime's
  own loader.
- mlx-embeddings safetensors encoder: Gemma3 / XLM-RoBERTa / ModernBERT
  encoders via
  [mlx-embeddings](https://pypi.org/project/mlx-embeddings/), a core
  dependency. Aliases below.

```yaml
server:
  embeddings: qwen3-embed-0.6b                  # the default: a GGUF decoder-LM embedder
  # embeddings: hf:Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q6_K.gguf   # a specific rung
  # embeddings: ~/models/Qwen3-Embedding-4B.Q6_K.gguf   # a local GGUF
  # embeddings: embeddinggemma-gguf             # EmbeddingGemma encoder from a GGUF
  # embeddings: embeddinggemma                  # EmbeddingGemma encoder from safetensors
```

The value is a GGUF ref, an alias, an HF repo in MLX-embeddings format, or a
local converted model directory (`true` means the default alias). A bare
alias resolves to the default rung shown below; the `gmlx init` wizard
offers a quant follow-up to pick another rung and writes its concrete ref.
Presets:

GGUF embedders (dim = vector width, ctx = max input tokens):

| Alias | Repo (default rung) | dim / ctx | Notes |
|-------|--------------------|-----------|-------|
| `qwen3-embed-0.6b` | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0) | 1024 / 32k | default; decoder-LM, small/fast/multilingual; ~0.6 GB |
| `qwen3-embed-4b` | `Qwen/Qwen3-Embedding-4B-GGUF` (Q8_0) | 2560 / 40k | decoder-LM, higher retrieval quality; ~4.3 GB |
| `qwen3-embed-8b` | `Qwen/Qwen3-Embedding-8B-GGUF` (Q8_0) | 4096 / 40k | decoder-LM, best of the family; ~8 GB, largest index |
| `embeddinggemma-gguf` | `ggml-org/embeddinggemma-300M-GGUF` (Q8_0) | 768 / 2k | encoder (mean-pool + dense head), tiny multilingual (Google); ~0.3 GB |

(The bare `qwen3-embed` is a back-compat alias for `qwen3-embed-0.6b`.)

mlx-embeddings safetensors encoders (default rung `8bit`):

| Alias | Repo | dim / ctx | Notes |
|-------|------|-----------|-------|
| `embeddinggemma` | `mlx-community/embeddinggemma-300m-8bit` | 768 / 2k | tiny, strong multilingual (Google); ~0.3 GB |
| `arctic-l` | `mlx-community/snowflake-arctic-embed-l-v2.0-8bit` | 1024 / 8k | multilingual long-context (XLM-RoBERTa) |
| `nomic-embed` | `mlx-community/nomicai-modernbert-embed-base-8bit` | 768 / 8k | popular long-context English (ModernBERT) |
| `bge-m3` | `mlx-community/bge-m3-mlx-8bit` | 1024 / 8k | multilingual long-context (XLM-RoBERTa) |

Picking by family: the GGUF Qwen3-Embedding tier carries the longest context.
`0.6b` is the best size/quality trade for most RAG; step up to `4b`/`8b` for
higher retrieval quality at a larger index and more RAM. The encoder tier is
for when you specifically want one of those models (e.g. `embeddinggemma`
for a tiny multilingual footprint).

The model is pre-warmed in the background at startup (best-effort, falling
back to a lazy first-request load), then cached in-process, kept separate
from the chat residency pool so a RAG re-index and chat never evict each
other. Requests follow the OpenAI shape (a JSON body with `input`, a string
or list of strings, plus optional `model` and
`encoding_format: float|base64`, default `float`):

```sh
curl localhost:8080/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"text-embedding-3-small","input":["hello","world"]}'
```

Send `model=text-embedding-3-small` (or `-3-large` / `-ada-002`, or omit it);
the conventional OpenAI names map to the configured model, and `/v1/models`
advertises a `text-embedding-3-small` entry when embeddings are on. Any other
requested model is refused. Vectors are L2-normalized (mean-pooled by the
encoder backend, last-token/EOS pooled by the GGUF decoder-LM backend).
Embedding passes run serialized with each other in a worker thread,
interleaving with (not blocking) batched LLM decode. To point Open WebUI's
RAG here: `RAG_EMBEDDING_ENGINE=openai`, `RAG_OPENAI_API_BASE_URL=<server>/v1`,
`RAG_EMBEDDING_MODEL=text-embedding-3-small`, which is exactly what
`gmlx launch open-webui` sets for you.

## Reranking (`rerank:`)

Setting `server.rerank:` (or `--rerank` in any serve mode) adds a
Cohere/Jina-shaped `POST /v1/rerank` (also `/rerank`): the second RAG stage.
A vector search returns a coarse top-N, the reranker re-scores those
documents jointly against the query, and the best few go to the model. Open
WebUI calls it as an external reranker.

The model is a Qwen3-Reranker GGUF: a Qwen3 causal LM fine-tuned to answer
"yes" or "no" to whether a document satisfies a query, so the runtime loads
it like any other GGUF, no extra needed. The relevance score is the
probability it assigns "yes" over "no" (`sigmoid(yes - no)`). (BGE/Jina BERT
cross-encoders, which llama.cpp reranks via a classifier head, are not
mlx-lm arches and are out of scope.)

```yaml
server:
  rerank: qwen3-rerank-0.6b                      # the default: a Qwen3-Reranker GGUF
  # rerank: hf:mradermacher/Qwen3-Reranker-4B-GGUF/Qwen3-Reranker-4B.Q6_K.gguf   # a specific rung
  # rerank: ~/models/Qwen3-Reranker-4B.Q6_K.gguf   # a local GGUF
```

Its value is a `qwen3-rerank-*` alias (`0.6b`/`4b`/`8b`, default rung
`Q8_0`), a `*.gguf` path, or an `hf:<org>/<repo>/<file>.gguf` ref. The
reranker is independent of the embedder, but `gmlx init` defaults its quant
to the embedder's chosen rung. The request is the Cohere/Jina shape (`query`,
`documents` as strings or `{"text": ...}` objects, and optional `top_n`,
`instruction`, `return_documents`); the response is `results` (sorted
best-first, each with `index` + `relevance_score`), plus `model` and `usage`:

```sh
curl localhost:8080/v1/rerank -H 'content-type: application/json' \
  -d '{"query":"how do I cancel?","documents":["Billing FAQ ...","Setup guide ..."]}'
```

The server serves one configured reranker (the request `model` is echoed but
never selects a different one). Scoring is one model forward per document,
serialized in a worker thread, interleaving with batched LLM decode.
`gmlx launch open-webui` points Open WebUI's external reranker here
automatically when the server advertises `rerank` via `/v1/models` (it sets
`RAG_RERANKING_ENGINE=external`, `RAG_EXTERNAL_RERANKER_URL=<server>/v1/rerank`,
`RAG_RERANKING_MODEL=reranker`, and enables hybrid search, which is when
reranking runs).

---
