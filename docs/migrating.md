# Migrating from llama.cpp, Ollama or LM Studio

gmlx runs the same GGUF files those tools use, and the models themselves
are reused with no conversion. This page maps the rest. It lists what transfers
directly, what has a different name, and what is different by design.

## Coming from llama.cpp

`gmlx run model.gguf` is the equivalent of `llama-cli -m model.gguf`, and
`gmlx serve model.gguf` is the equivalent of `llama-server`, on the same
default port 8080.

| llama.cpp | gmlx | Notes |
|-----------|------|-------|
| `-m model.gguf` | positional `model.gguf` | for sharded files, point at the first shard |
| `-n N` | `--max-tokens N` | default is until end-of-sequence on both |
| `--temp`, `--top-k`, `--top-p`, `--min-p` | same names | defaults come from each model family's card, so bare `run` and `chat` are already tuned |
| `-c N` | none | the window comes from the GGUF metadata, and `--max-kv-size N` bounds cache memory with a rotating cache |
| `--rope-scaling`, `--yarn-*` | none | read from metadata, with an expert override in [debug-switches.md](internals/debug-switches.md) |
| `-ngl` | none needed | everything runs on the GPU, and `--stream-experts` and `--stream-cpu` are the over-RAM MoE placements in [streaming.md](streaming.md) |
| `--cache-type-k/-v q8_0` | `--kv-bits 8` | same purpose, with `--kv-group-size` |
| `--draft-model`, `--spec-draft-n-max` | `--draft-gguf`, `--draft-block-size` | models with a native head need no companion drafter |
| `--chat-template` | `--chat-template STR_OR_PATH` | per model in server configs under `overrides` |
| `--ignore-eos` | `--ignore-eos` | same benchmarking semantics |
| `--api-key K` | `server.api_key` in the config | config-only, so the key never appears in process listings or shell history |
| `--parallel N` | none | continuous batching admits requests automatically, and `--budget-gb` bounds residency |
| `--lora adapter` | `--adapter adapter.gguf` | [lora.md](lora.md) covers adapter interoperation in both directions |

`/v1/completions` accepts a single string prompt and returns a single
choice, and `/v1/chat/completions` is the primary route. Anthropic Messages and
OpenAI Responses run on the same port, as [api.md](api.md) describes.

## Coming from Ollama

Any GGUF file on disk can be reused. Ollama's model store and API do not.

- Ollama's library is stored as sha-named blobs, not `.gguf` files, and it
  cannot be used directly. Re-download the models you use with `gmlx pull`.
  `gmlx validate hf:<org>/<repo>` lists the available quants first.
- gmlx implements the OpenAI, Anthropic and OpenAI Responses APIs, not the
  Ollama API. Clients configured for an OpenAI-compatible endpoint work
  unchanged. Ollama-native integrations need their OpenAI mode, pointed at
  port 8080.
- Modelfile parameters map onto the config in
  [server-config.md](server-config.md). `num_predict` is the server's
  `max_tokens` default, sampling keys are set for each model or in
  `profiles:`, and `SYSTEM` becomes `system:`.
- Keep-alive and unload behavior is the
  [residency system](server-config.md#residency), with an idle timeout, LRU
  eviction under a byte budget, and `pin` for always-resident models.

## Coming from LM Studio

Your existing library serves as it is, since the files are plain GGUFs:

```sh
gmlx init --models-dir ~/.lmstudio/models -r
```

The init wizard also offers the LM Studio directory unprompted when it
exists. Ids, sampling profiles and a default model can then be adjusted in
the YAML file. The local server API matches LM Studio's, plus Anthropic
Messages on the same port.
