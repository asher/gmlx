# Migrating from other tools

gmlx runs the same GGUF files as llama.cpp, LM Studio and Ollama, so a model
needs no conversion. Most of those tools' commands, flags and settings have
a gmlx equivalent, and an LM Studio library runs without a new download.

## llama.cpp

`gmlx chat model.gguf` is the equivalent of `llama-cli -m model.gguf`, and
`gmlx serve model.gguf` is the equivalent of `llama-server -m model.gguf`,
on the same default port 8080. `gmlx run model.gguf --prompt "..."`
generates one reply and exits.

Most flags have a direct equivalent:

| llama.cpp | gmlx | Notes |
|-----------|------|-------|
| `-m model.gguf` | The model as the first argument | For a model split into shards, give the first shard. |
| `-n N` | `--max-tokens N` | Both generate until end-of-sequence by default. |
| `--temp`, `--top-k`, `--top-p`, `--min-p` | The same names | The defaults come from the model's [family defaults](family-defaults.md). |
| `-c N` | `--max-kv-size N` | The context length comes from the GGUF, and this flag limits it. On `run` and `chat`, the cache also becomes a rotating window. |
| `--rope-scaling`, `--yarn-*` | None | gmlx reads the scaling from the GGUF, and there is no override. |
| `-ngl N` | None | Every layer runs on the GPU. A model larger than memory runs with `--stream-experts` or `--stream-cpu`. |
| `--cache-type-k q8_0`, `--cache-type-v q8_0` | `--kv-bits 8` | `--kv-group-size` sets the group size. |
| `--draft-model`, `--spec-draft-n-max M` | `--draft-gguf`, `--draft-block-size M+1` | The block also counts the token that the drafts extend. A model with an MTP head needs no drafter. |
| `--chat-template` | `--chat-template` | The value is a template or the path of one. |
| `--ignore-eos` | `--ignore-eos` on `serve` | The flag has the same effect, for benchmarks at a fixed length. |
| `--api-key K` | [`server.api_key`](config.md#serverapi_key) | The key is set only in the configuration file, so it never shows in process listings or shell history. |
| `--parallel N` | None | The server batches concurrent requests with no setting, and `--budget-gb` limits the memory of loaded models. |
| `--lora FILE` | `--adapter FILE` | [LoRA adapters](lora.md) covers adapter files in both directions. |

The server answers the OpenAI, Anthropic Messages and OpenAI Responses APIs
on one port, and the [HTTP API](api.md) lists the fields that each one
accepts.

## Ollama

Ollama stores its models as blobs named by their hash, not as `.gguf`
files, and `gmlx init` finds only `.gguf` files. Download the models that
you use again with `gmlx pull`. `gmlx validate hf:<org>/<repo>` lists the
files of a repository first.

gmlx does not implement the Ollama API. A client that can use an
OpenAI-compatible endpoint works with no changes, and an app built for
Ollama needs its OpenAI mode, pointed at port 8080.

A Modelfile's settings map to the [configuration file](config.md):

| Modelfile | gmlx |
|-----------|------|
| `PARAMETER num_predict` | [`sampling.max_tokens`](config.md#samplingmax_tokens) |
| `PARAMETER temperature`, `top_k`, `top_p`, `min_p`, `seed`, `stop` | The [sampling](config.md#sampling) keys of the same names |
| `PARAMETER repeat_penalty`, `repeat_last_n` | [`sampling.repetition_penalty`](config.md#samplingrepetition_penalty), [`sampling.repetition_context_size`](config.md#samplingrepetition_context_size) |
| `PARAMETER num_ctx` | [`load.max_kv_size`](config.md#loadmax_kv_size) |
| `SYSTEM` | [`system`](config.md#profilessystem) |

Each of these goes in a [profile](config.md#profiles) or in a model
entry's [`overrides`](config.md#modelsoverrides). A Modelfile
`TEMPLATE` does not carry over, because gmlx uses the Jinja chat template
in the GGUF.

Ollama's keep-alive setting corresponds to the server's idle timeout. A
model unloads after [`ttl_s`](config.md#serverdefaultsttl_s) seconds
without a request, or when the memory budget needs the room that the model
takes. [`pin`](config.md#modelspin) keeps a model loaded.
[Memory and residency](config.md#memory-and-residency) explains the rules.

## LM Studio

LM Studio stores plain GGUF files, so your library runs as it is:

```sh
gmlx init --models-dir ~/.lmstudio/models -r
```

The `-r` flag scans the folders inside the library too. When you have no
`~/models` or `~/gguf` folder, the `gmlx init` wizard offers the LM Studio
folder by itself. You can then rename the models and set a default model in
the [configuration file](config.md).

A client that used LM Studio's OpenAI-compatible endpoint works with
gmlx after you change its port from 1234 to 8080.
