# Speech, embeddings and rerank

Beside its chat models, the server can run services for speech-to-text,
text-to-speech, embeddings and reranking on the same port. The speech and
embeddings services add OpenAI-compatible endpoints, and the reranker adds
the endpoint shape that Cohere and Jina use. The voice client
[`gmlx talk`](talk.md), [RAG pipelines](rag.md) and apps such as
[Open WebUI](launch.md#open-webui) can use them. A service starts when its
key in the [configuration file](config.md#services) names a model:

```yaml
server:
  stt: whisper-turbo
  tts: kokoro
  embeddings: qwen3-embed-0.6b
  rerank: qwen3-rerank-0.6b
```

- [How the services run](#how-the-services-run)
- [Speech-to-text](#speech-to-text)
- [Text-to-speech](#text-to-speech)
- [Embeddings](#embeddings)
- [Reranking](#reranking)

## How the services run

The server loads each configured service model in the background at
start, and the first request loads it if that load failed. Service models
do not count against [`server.budget_gb`](config.md#serverbudget_gb) and
are never unloaded to make room for chat models, so indexing documents and
chatting do not push each other out. One service's requests run one at a
time, alongside chat model generation.

A request can leave out its `model` field, set it to `default`, or send a
name that OpenAI clients commonly send. All of these reach the configured
model, as does the configured model's own alias, repository id or path.
The speech and embeddings services answer any other name with a 400, so a
client cannot make the server download a model. `/v1/models` lists each
running service under the first name in its row:

| Service | Names accepted in `model` |
|---------|---------------------------|
| Speech-to-text | `whisper-1` |
| Text-to-speech | `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` |
| Embeddings | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |
| Reranking | `reranker`. Any name is accepted and returned in the response, since the server runs one reranker. |

The routes of a service that is not configured answer 404, with a message
that names the service's key. Set the key in the config file and run
`gmlx restart` to start the service.

Speech-to-text and text-to-speech need the `stt` and `tts` extras, which
[Optional features](installation.md#optional-features) describes. When a
speech service is configured and its extra is missing, the server refuses
to start.

Speech-to-text, speech in a format other than WAV or PCM, and some chat
audio need ffmpeg. Install it with `brew install ffmpeg`. The server looks
for it on its `PATH`, then in the Homebrew and system folders, so a server
that a login item starts finds it too.

The server skips relative `PATH` entries and never runs a program from a
folder that a [container](launch-container.md) client can write. Its log
names the ffmpeg it uses and any entry it skips. `gmlx doctor` reports FAIL
when a speech service is configured and no ffmpeg is found.

When an embeddings or rerank service's model file is missing, the server
starts without that service and prints a warning. For an absolute path, the
endpoint returns a 404 until the file is back, and then it works with no
restart.

A service set by an alias, an `hf:` reference or a relative path stays off.
Its routes answer 404 with a message that says the model was missing. A
reload does not start the service, so run `gmlx restart` once the file is
back.

## Speech-to-text

`server.stt` adds `POST /v1/audio/transcriptions`, which turns speech into
text with [mlx-whisper](https://pypi.org/project/mlx-whisper/). Whisper
models are not GGUF files, so the server loads them in MLX format. The
value is an alias, a Hugging Face repository in MLX-Whisper format, or a
local folder with a converted model, and `true` selects `whisper-turbo`.
The aliases are these:

| Alias | Repository | Notes |
|-------|------------|-------|
| `whisper-turbo` | `mlx-community/whisper-large-v3-turbo` | It is the default, and it gives large-v3 quality at about six times the speed. |
| `whisper-turbo-q4` | `mlx-community/whisper-large-v3-turbo-q4` | It is a 4-bit version of the turbo model, about 600 MB. |
| `whisper-large` | `mlx-community/whisper-large-v3-mlx` | It is the full large-v3 model. |
| `whisper-medium` | `mlx-community/whisper-medium-mlx` | It is smaller and faster than large-v3. |
| `whisper-small` | `mlx-community/whisper-small-mlx` | It is smaller and faster than medium. |
| `whisper-base` | `mlx-community/whisper-base-mlx` | It is smaller and faster than small. |
| `whisper-tiny` | `mlx-community/whisper-tiny` | It is the smallest and fastest. |

When the configured model is not already local, the server downloads it
from Hugging Face in the background at start. Naming it in the configuration file allows
that download, and chat models are still never downloaded.

A request is a `multipart/form-data` upload with `file`, and the optional
fields `model`, `language`, `prompt`, `temperature` and `response_format`.
`response_format` is `json`, `text`, `verbose_json`, `srt` or `vtt`, and
`temperature` defaults to 0. Decoding the upload needs
[ffmpeg](#how-the-services-run). A recording longer than the
[audio limit](api.md#limits-and-back-pressure) gets a 400, so split such a
recording and send each part in its own request. This request transcribes a
clip:

```sh
curl localhost:8080/v1/audio/transcriptions -F file=@clip.ogg -F model=whisper-1
```

The same model also serves `POST /v1/audio/translations`, which takes
speech in any language and returns English text. The request is the same
upload without `language`:

```sh
curl localhost:8080/v1/audio/translations -F file=@japanese.ogg -F model=whisper-1
```

## Text-to-speech

`server.tts` adds `POST /v1/audio/speech`, which turns text into speech with
[mlx-audio](https://pypi.org/project/mlx-audio/). Speech models also load in
MLX format. The value is an alias, a Hugging Face repository in MLX-Audio
format, or a local folder with a converted model. `true` selects `kokoro`,
the first of these aliases:

| Alias | Repository | Notes |
|-------|------------|-------|
| `kokoro` | `mlx-community/Kokoro-82M-bf16` | It is the default, a small English model with many preset voices. |
| `kokoro-8bit` | `mlx-community/Kokoro-82M-8bit` | It is a smaller Kokoro. |
| `kokoro-4bit` | `mlx-community/Kokoro-82M-4bit` | It is the smallest Kokoro. |
| `qwen3-tts` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | It is a larger multilingual model with named voices. |
| `qwen3-tts-small` | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | It is a smaller Qwen3-TTS. |

Kokoro also downloads a small English language model from Hugging Face the
first time it speaks, so a server without internet access needs that
download done beforehand.

A request is a JSON body with `input`, and the optional fields `model`,
`voice`, `speed` and `response_format`. `speed` is from 0.25 to 4.0.
`response_format` is `mp3`, the default, or `wav`, `flac`, `opus` or `pcm`.
Every format except `wav` and `pcm` needs [ffmpeg](#how-the-services-run).

`voice` is a voice name, or several joined by commas. A value with a dot or
a slash gets a 400, because it would name a file. Without `voice`, Kokoro
speaks as `af_heart`, and other models use their own default voice. This
request speaks one line as `af_heart`:

```sh
curl localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model": "tts-1", "input": "Hello from MLX.", "voice": "af_heart"}' -o out.mp3
```

Before speaking, the server removes markdown, emoji and control characters
from `input`, and it turns dashes into pauses. Input with nothing left to
speak gets a 400.

`GET /v1/audio/voices` lists the configured model's voices, which is
what `gmlx talk` shows for `/voice`. Kokoro lists the voices in its model
folder, a Qwen3-TTS model lists its named speakers, and another model
lists none:

```sh
curl localhost:8080/v1/audio/voices
# {"model": "mlx-community/Kokoro-82M-bf16", "voices": ["af_alloy", ...], "default": "af_heart"}
```

A Sesame model, such as `mlx-community/csm-1b`, speaks the preset voices in
its repository, such as `conversational_a` and `conversational_b`. Set
`server.tts` to the repository, not a local folder, to get them. A voice it
does not have gets a 400 that lists the ones it has.

Sesame also needs files from two other repositories. Download them into the
Hugging Face cache first:

```sh
hf download kyutai/moshiko-pytorch-bf16 tokenizer-e351c8d8-checkpoint125.safetensors
hf download unsloth/Llama-3.2-1B --include "*.json"
```

Open WebUI always sends a voice, and its default voice is an OpenAI name
that Kokoro does not have. `gmlx launch open-webui` therefore sets
`AUDIO_TTS_VOICE` to `af_heart`. With another speech model, export
`AUDIO_TTS_VOICE` with one of the model's own voices before the launch,
and the launch keeps your value.

## Embeddings

`server.embeddings` adds `POST /v1/embeddings`, which turns text into
vectors for search. No extra is needed. The value selects one of three
kinds of model:

- A GGUF embedder: The value is a `*.gguf` path, an
  `hf:<org>/<repo>/<file>.gguf` reference, or a `qwen3-embed-*` alias.
  These are Qwen3-Embedding models, loaded like any other GGUF.
- A GGUF encoder: `embeddinggemma-gguf` runs an EmbeddingGemma GGUF.
- A safetensors encoder: An alias, a Hugging Face repository or a local
  folder runs through
  [mlx-embeddings](https://pypi.org/project/mlx-embeddings/).

`true` selects `qwen3-embed-0.6b`, and `qwen3-embed` without a size is
another name for it. These are the GGUF models, where the
dimension is the width of each vector and the context is the most tokens of
input that the model reads:

| Alias | Repository and default quant | Dimension | Context | Notes |
|-------|------------------------------|-----------|---------|-------|
| `qwen3-embed-0.6b` | `Qwen/Qwen3-Embedding-0.6B-GGUF`, Q8_0 | 1024 | 32K | It is the default, a small, fast multilingual model of about 0.6 GB. |
| `qwen3-embed-4b` | `Qwen/Qwen3-Embedding-4B-GGUF`, Q8_0 | 2560 | 32K | It retrieves better and takes about 4.3 GB. |
| `qwen3-embed-8b` | `Qwen/Qwen3-Embedding-8B-GGUF`, Q8_0 | 4096 | 32K | It retrieves best of the family, takes about 8 GB and makes the largest index. |
| `embeddinggemma-gguf` | `ggml-org/embeddinggemma-300M-GGUF`, Q8_0 | 768 | 2K | It is a small multilingual encoder from Google, about 0.3 GB. |

These are the safetensors encoders, each at its 8-bit default:

| Alias | Repository | Dimension | Context | Notes |
|-------|------------|-----------|---------|-------|
| `embeddinggemma` | `mlx-community/embeddinggemma-300m-8bit` | 768 | 2K | It is a small multilingual model from Google, about 0.3 GB. |
| `arctic-l` | `mlx-community/snowflake-arctic-embed-l-v2.0-8bit` | 1024 | 8K | It is a multilingual model for long inputs. |
| `nomic-embed` | `mlx-community/nomicai-modernbert-embed-base-8bit` | 768 | 8K | It is a widely used English model for long inputs. |
| `bge-m3` | `mlx-community/bge-m3-mlx-8bit` | 1024 | 8K | It is a multilingual model for long inputs. |

Choose a GGUF model unless you want one of the encoders for its size or its
languages. Like the speech models, a safetensors encoder downloads once
when it is not in the cache.

A GGUF reference, written out or reached through an alias, is never
downloaded by the server. The server finds it in your local Hugging Face
cache, or in a folder of [`server.model_dirs`](config.md#servermodel_dirs)
where `gmlx pull` saves it. `gmlx init` can choose a quant other than the
default and write its full reference. Download the default model before you
start the server:

```sh
gmlx pull hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf
```

A request is a JSON body with `input`, which is a string or a list of
strings. The optional fields are `model` and `encoding_format`, which is
`float` by default or `base64`. Every model returns vectors normalized to
length 1, and input longer than the model's context is cut to fit without
an error. The server refuses input given as token ids, and it ignores the
OpenAI `dimensions` field:

```sh
curl localhost:8080/v1/embeddings -H 'content-type: application/json' \
  -d '{"model": "text-embedding-3-small", "input": ["hello", "world"]}'
```

## Reranking

`server.rerank` adds `POST /v1/rerank`, which is also served at `/rerank`.
A reranker scores documents against a query more accurately than a vector
search, so a RAG pipeline uses it on the short list that the search
returns. The request and response have the shape that Cohere and Jina use.

The model is a Qwen3-Reranker GGUF, which judges whether a document answers
a query. Its score is the probability it gives to yes over no. The value is
a `qwen3-rerank-0.6b`, `qwen3-rerank-4b` or `qwen3-rerank-8b` alias at the
Q8_0 quant, a `*.gguf` path, or an `hf:` reference, and `true` selects
`qwen3-rerank-0.6b`. Like a GGUF embedder, it is never downloaded by the
server:

```sh
gmlx pull hf:mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf
```

A request is a JSON body with these fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `query` | Required | The documents are ranked against this text. |
| `documents` | Required | These documents are ranked, given as strings or `{"text": ...}` objects. |
| `top_n` | Every document | The response holds this many results, a positive integer. `top_k` is another name for it. |
| `return_documents` | `true` | With `false`, each result has only its index and score. |
| `instruction` | "Given a web search query, retrieve relevant passages that answer the query" | The reranker reads this instruction with the query. |

```sh
curl localhost:8080/v1/rerank -H 'content-type: application/json' \
  -d '{"query": "how do I cancel?", "documents": ["Billing FAQ ...", "Setup guide ..."]}'
```

The response holds `model`, `usage` and `results`, best first, each with
its `index` and `relevance_score`. The model reads each document
separately, up to 8192 tokens with the query, so send the short list from a
vector search, tens of documents rather than thousands.
