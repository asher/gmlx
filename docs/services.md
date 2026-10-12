# Speech, embeddings and rerank

Beside its chat models, the gmlx server can run four services on the same
port: speech-to-text, text-to-speech, embeddings and reranking. The voice
client [`gmlx talk`](talk.md), [RAG pipelines](rag.md) and apps such as
[Open WebUI](launch.md#open-webui) use them.

To turn a service on, name its model in the server block of the
[configuration file](config.md#services), then
[reload](config.md#changing-the-file) the server or run `gmlx restart`:

```yaml
server:
  stt: whisper-turbo
  tts: kokoro
  embeddings: qwen3-embed-0.6b
  rerank: qwen3-rerank-0.6b
```

`true` selects each service's default model, the one shown here. The
embeddings and rerank models here are GGUF files that the server never
downloads, so pull them first, as [Embeddings](#embeddings) and
[Reranking](#reranking) show.

- [Speech-to-text](#speech-to-text)
- [Text-to-speech](#text-to-speech)
- [Embeddings](#embeddings)
- [Reranking](#reranking)
- [How the services run](#how-the-services-run)

## Speech-to-text

Speech-to-text turns a recording into text with
[mlx-whisper](https://pypi.org/project/mlx-whisper/). It needs the `stt`
extra, which [Optional features](installation.md#optional-features)
installs, and ffmpeg (`brew install ffmpeg`). Set `server.stt`, restart, and
send a file:

```sh
curl localhost:8080/v1/audio/transcriptions -F file=@clip.ogg -F model=whisper-1
```

`/v1/audio/translations` takes speech in any language and returns English
text:

```sh
curl localhost:8080/v1/audio/translations -F file=@japanese.ogg -F model=whisper-1
```

The optional form fields are `language` (transcriptions only), `prompt`,
`temperature` (0 by default) and `response_format`: `json`, `text`,
`verbose_json`, `srt` or `vtt`. A recording longer than the
[audio limit](api.md#limits-and-back-pressure) gets a 400, so split it.

`server.stt` takes one of these aliases, a Hugging Face repository in
MLX-Whisper format, or a local folder with a converted model:

| Alias | Repository | Notes |
|-------|------------|-------|
| `whisper-turbo` | `mlx-community/whisper-large-v3-turbo` | Default. Large-v3 quality at about six times the speed. |
| `whisper-turbo-q4` | `mlx-community/whisper-large-v3-turbo-q4` | 4-bit turbo, about 600 MB |
| `whisper-large` | `mlx-community/whisper-large-v3-mlx` | Full large-v3 |
| `whisper-medium` | `mlx-community/whisper-medium-mlx` | Smaller and faster than large-v3 |
| `whisper-small` | `mlx-community/whisper-small-mlx` | Smaller and faster than medium |
| `whisper-base` | `mlx-community/whisper-base-mlx` | Smaller and faster than small |
| `whisper-tiny` | `mlx-community/whisper-tiny` | Smallest and fastest |

## Text-to-speech

Text-to-speech reads text aloud with
[mlx-audio](https://pypi.org/project/mlx-audio/). It needs the `tts` extra.
Set `server.tts`, restart, and send a line:

```sh
curl localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model": "tts-1", "input": "Hello from MLX.", "voice": "af_heart"}' -o out.mp3
```

The optional fields are `voice`, `speed` (0.25 to 4.0) and
`response_format`: `mp3` (the default), `wav`, `flac`, `opus` or `pcm`.
Every format except `wav` and `pcm` needs ffmpeg. The server removes
markdown and emoji from `input` before it speaks.

`voice` takes one voice name, or several joined by commas. Without it,
Kokoro speaks as `af_heart`. To list the model's voices:

```sh
curl localhost:8080/v1/audio/voices
# {"model": "mlx-community/Kokoro-82M-bf16", "voices": ["af_alloy", ...], "default": "af_heart"}
```

`server.tts` takes one of these aliases, a Hugging Face repository in
MLX-Audio format, or a local folder with a converted model:

| Alias | Repository | Notes |
|-------|------------|-------|
| `kokoro` | `mlx-community/Kokoro-82M-bf16` | Default. Small English model with many preset voices. |
| `kokoro-8bit` | `mlx-community/Kokoro-82M-8bit` | Smaller Kokoro |
| `kokoro-4bit` | `mlx-community/Kokoro-82M-4bit` | Smallest Kokoro |
| `qwen3-tts` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | Larger multilingual model with named voices |
| `qwen3-tts-small` | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | Smaller Qwen3-TTS |

Kokoro downloads a small English language model the first time it speaks,
so a server without internet access needs one run with access first.

A Sesame model, such as `mlx-community/csm-1b`, speaks the preset voices in
its repository, such as `conversational_a`. Set `server.tts` to the
repository, not a local folder. Sesame also needs files from two other
repositories in the Hugging Face cache:

```sh
hf download kyutai/moshiko-pytorch-bf16 tokenizer-e351c8d8-checkpoint125.safetensors
hf download unsloth/Llama-3.2-1B --include "*.json"
```

`gmlx launch open-webui` sets Open WebUI's voice to `af_heart`. With another
speech model, export `AUDIO_TTS_VOICE` with one of its voices before the
launch.

## Embeddings

Embeddings turn text into vectors for search. They need no extra. The
default model is a GGUF that the server never downloads, so pull it first:

```sh
gmlx pull hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf
```

Set `server.embeddings`, restart, and send text:

```sh
curl localhost:8080/v1/embeddings -H 'content-type: application/json' \
  -d '{"model": "text-embedding-3-small", "input": ["hello", "world"]}'
```

`input` is a string or a list of strings. `encoding_format` is `float` (the
default) or `base64`. Vectors come back normalized to length 1. Input longer
than the model's context is cut to fit, and the OpenAI `dimensions` field is
ignored.

`server.embeddings` takes a GGUF alias, a `*.gguf` path or an
`hf:<org>/<repo>/<file>.gguf` reference. The dimension is the width of each
vector, and the context is the most input tokens the model reads:

| Alias | Repository, Q8_0 | Dimension | Context | Notes |
|-------|------------------|-----------|---------|-------|
| `qwen3-embed-0.6b` | `Qwen/Qwen3-Embedding-0.6B-GGUF` | 1024 | 32K | Default. Small, fast and multilingual, about 0.6 GB. |
| `qwen3-embed-4b` | `Qwen/Qwen3-Embedding-4B-GGUF` | 2560 | 32K | Better retrieval, about 4.3 GB |
| `qwen3-embed-8b` | `Qwen/Qwen3-Embedding-8B-GGUF` | 4096 | 32K | Best retrieval of the family and the largest index, about 8 GB |
| `embeddinggemma-gguf` | `ggml-org/embeddinggemma-300M-GGUF` | 768 | 2K | Small multilingual encoder from Google, about 0.3 GB |

`gmlx init` can pick another quant and write its full reference.

The server also runs safetensors encoders, from an alias, a Hugging Face
repository or a local folder. An MLX checkpoint, plain or quantized, runs on
the encoders built into gmlx, which cover EmbeddingGemma, BERT,
XLM-RoBERTa, ModernBERT and Qwen3-Embedding. These download once when they
are not in the cache:

| Alias | Repository | Dimension | Context | Notes |
|-------|------------|-----------|---------|-------|
| `embeddinggemma` | `mlx-community/embeddinggemma-300m-8bit` | 768 | 2K | Small multilingual model from Google, about 0.3 GB |
| `arctic-l` | `mlx-community/snowflake-arctic-embed-l-v2.0-8bit` | 1024 | 8K | Multilingual, long inputs |
| `nomic-embed` | `mlx-community/nomicai-modernbert-embed-base-8bit` | 768 | 8K | English, long inputs |
| `bge-m3` | `mlx-community/bge-m3-mlx-8bit` | 1024 | 8K | Multilingual, long inputs |

Choose a GGUF model unless you want an encoder for its size or languages.

The server checks the model at startup, from a safetensors encoder's
`config.json` or from a GGUF's header, and a model that fails the check
does not load. A safetensors encoder fails for another model type, and for
a Gemma 3 or Qwen3 chat model, which shares its model type with
EmbeddingGemma or Qwen3-Embedding. It also fails when its `config.json` or
`modules.json` cannot be read or is not valid JSON. A GGUF is an embeddings
model when it is of an encoder architecture, sets `<arch>.pooling_type` to 1
or more, or has `embed` or `embedding` as a word of its file name, as the
Qwen3-Embedding GGUFs do and a chat model does not. Any other GGUF fails.
So does a reranker, by its file name or a pooling type of 4, a GGUF of a
pooling type over 4, and a GGUF of an encoder architecture that gmlx cannot
run, which is every encoder architecture but EmbeddingGemma's. A GGUF of
EmbeddingGemma's architecture fails when it sets a pooling type other than
1, since that encoder takes the mean. A GGUF that sets
`<arch>.attention.causal` to false fails unless it is of the llama,
mistral3, qwen2 or qwen3 architecture and has no sliding window. The server
still starts, and the server log and every `/v1/embeddings` response give
the reason with a 501 status.

A Qwen3 checkpoint is Qwen3-Embedding when its `modules.json` lists a
sentence-transformers Pooling module, or when its repository, folder name or
model card names Qwen3-Embedding, as for a conversion without `modules.json`.

An encoder pools its vectors as the sentence-transformers configuration in
its folder says. When the folder has none, BGE-M3, Arctic-Embed, the English
and Chinese BGE v1 and v1.5 models, mxbai-embed-large-v1 and
mxbai-embed-2d-large-v1 use the first token, as their upstream
configurations do. The server knows these models by the name in
`config.json` or in the model card, or by the repository or folder name.
Qwen3-Embedding uses the last token, and the other models use the mean over
the tokens, as sentence-transformers does. The server log names the pooling
it chose.

A GGUF embedder that sets `<arch>.pooling_type` to 1 takes the mean over
the tokens, one that sets 2 the first token, and one that sets 3 the last
token. One that sets no pooling type, or sets 0, uses the last token and
appends an EOS to an input that does not end in one, as Qwen3-Embedding
does. One that sets `<arch>.attention.causal` to false, as Nemotron-3-Embed
does, runs without the causal mask, so each token attends to the whole
input.

EmbeddingGemma never runs in float16, which its model card rules out. It
runs in bfloat16, or in float32 where [`server.dtype`](config.md#serverdtype)
gives float16.

A cached encoder loads with no network access when its own files or its
name settle how it pools. Otherwise the server fetches the model card and
the sentence-transformers module files that the cached copy lacks, at the
commit of the copy, so the copy stays the one in use. Each file gets one
try, and the fetch stops after 5 seconds in all. Once the Hub has answered
for each file, or the fetch has timed out before the Hub answered any
request, the server leaves a mark in the cache,
`models--<org>--<name>/.gmlx-fetched/<commit>`, and does not fetch for that
copy again. To fetch again, delete the mark, which the log names when a
file is not fetched, and restart the server. A fetch that finds no
connection, or that times out after the Hub answered a request, as on a
slow connection, leaves no mark. The files it fetched stay, and the next
start asks for the others. With `HF_HUB_OFFLINE=1`, or in a cache the
server cannot write, the server fetches nothing.

## Reranking

A reranker scores documents against a query more accurately than a vector
search, so a RAG pipeline runs it on the short list that the search
returns. The model is a Qwen3-Reranker GGUF, which the server never
downloads:

```sh
gmlx pull hf:mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf
```

Set `server.rerank`, restart, and rank some documents:

```sh
curl localhost:8080/v1/rerank -H 'content-type: application/json' \
  -d '{"query": "how do I cancel?", "documents": ["Billing FAQ ...", "Setup guide ..."]}'
```

The request and response have the shape that Cohere and Jina use, and
`/rerank` works too. The response holds `results`, best first, each with
its `index` and `relevance_score`. The score is the probability that the
model gives to yes over no, from 0 to 1, so you can drop documents below a
threshold. The fields are:

| Field | Default | Meaning |
|-------|---------|---------|
| `query` | Required | The text to rank against |
| `documents` | Required | Strings or `{"text": ...}` objects |
| `top_n` | Every document | How many results to return. `top_k` is another name for it. |
| `return_documents` | `true` | With `false`, each result has only its index and score. |
| `instruction` | "Given a web search query, retrieve relevant passages that answer the query" | The task the reranker reads with the query |

`server.rerank` takes `qwen3-rerank-0.6b`, `qwen3-rerank-4b` or
`qwen3-rerank-8b` (each at Q8_0), a `*.gguf` path, or an `hf:` reference.
The model reads each document separately, up to 8192 tokens with the query,
so send tens of documents, not thousands.

## How the services run

The server loads each service model in the background at start, and
again at a reload that changes it. Service
models do not count against [`server.budget_gb`](config.md#serverbudget_gb)
and are never unloaded for chat models, so indexing and chatting do not
push each other out. Each service runs its requests one at a time, so a
long batch of embeddings holds up the next embeddings request.

A request may leave out `model`, or send a name that OpenAI clients use.
`/v1/models` lists each running service under the first of its names:

| Service | Names accepted in `model` |
|---------|---------------------------|
| Speech-to-text | `whisper-1` |
| Text-to-speech | `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` |
| Embeddings | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |
| Reranking | `reranker`, or any name |

A request never makes the server download a model. The models you name in
the config download at start, except GGUF models, which you pull, as
[Hugging Face policy](api.md#hugging-face-policy) explains.

A service that is not configured answers 404 with the key to set. When the
extra of a speech service in the config is missing, the server starts
without that service and logs a warning, which `gmlx serve` repeats, and
its routes answer 404 with the command that installs the extra. A reload of
the config turns the service off the same way, and the server stays up.
With `--stt` or `--tts`, a missing extra stops the server from starting. When an embeddings or rerank model
file is missing, the server runs without that service. Pull the file, then
reload the server or run `gmlx restart`.

The server finds ffmpeg on its `PATH` or in the Homebrew and system
folders, so a server started by a login item finds it too. `gmlx doctor`
reports FAIL when a speech service is configured and ffmpeg is missing.
