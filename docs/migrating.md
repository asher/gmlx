# Migrating from llama.cpp, Ollama or LM Studio

gmlx runs the same GGUF files those tools use, so the models themselves move
with no conversion. This page maps the rest: what carries over directly, what
has a different name, and what works differently on purpose.

## Coming from llama.cpp

`gmlx run model.gguf` is the equivalent of `llama-cli -m model.gguf`, and
`gmlx serve model.gguf` of `llama-server`. The default port is the same 8080.

| llama.cpp | gmlx | Notes |
|-----------|------|-------|
| `-m model.gguf` | positional `model.gguf` | sharded files: point at the first shard |
| `-n N` | `--max-tokens N` | default is until end-of-sequence on both |
| `--temp`, `--top-k`, `--top-p`, `--min-p` | same names | defaults come from each model family's card, so bare `run` and `chat` are already tuned |
| `-c N` | none | the window comes from the GGUF metadata; `--max-kv-size N` bounds cache memory with a rotating cache instead |
| `--rope-scaling`, `--yarn-*` | none | metadata-driven; an expert escape hatch exists in [debug-switches.md](internals/debug-switches.md) |
| `-ngl` | none needed | everything runs on the GPU; `--stream-experts` and `--stream-cpu` are the over-RAM MoE placements ([streaming.md](streaming.md)) |
| `--cache-type-k/-v q8_0` | `--kv-bits 8` | same purpose, with `--kv-group-size` |
| `--draft-model`, `--spec-draft-n-max` | `--draft-gguf`, `--draft-block-size` | models with a native head need no companion drafter |
| `--chat-template` | `--chat-template STR_OR_PATH` | per model in server configs under `overrides` |
| `--ignore-eos` | `--ignore-eos` | same benchmarking semantics |
| `--api-key K` | `server.api_key` in the config | config-only, so the key never lands in process listings or shell history |
| `--parallel N` | none | continuous batching admits requests automatically; residency is bounded by `--budget-gb` instead |
| `--lora adapter` | `--adapter adapter.gguf` | llama.cpp-format adapters interoperate in both directions ([lora.md](lora.md)) |

`/v1/completions` is served with a minimal surface of one string prompt and
one choice. `/v1/chat/completions` is the primary route, and Anthropic
Messages and OpenAI Responses run on the same port ([api.md](api.md)).

## Coming from Ollama

Any GGUF you can point at carries over. Ollama's model store and API do not.

- Ollama's library lives as sha-named blobs, not `.gguf` files, so it cannot
  be pointed at directly. Re-download the models you use with `gmlx pull`;
  `gmlx validate hf:<org>/<repo>` lists every variant first.
- gmlx speaks the OpenAI, Anthropic and OpenAI Responses APIs, not the Ollama
  API. Clients configured for an OpenAI-compatible endpoint work unchanged;
  Ollama-native integrations need their OpenAI mode, pointed at port 8080.
- Modelfile parameters map onto the config: `num_predict` is the server's
  `max_tokens` default, sampling keys live per model or in `profiles:`, and
  `SYSTEM` becomes `system:` ([server-config.md](server-config.md)).
- Keep-alive and unload behavior is the residency system: idle timeout, LRU
  under a byte budget, and `pin` for always-resident models
  ([Residency](server-config.md#residency)).

## Coming from LM Studio

Your existing library serves as it is, since the files are plain GGUFs:

```sh
gmlx init --models-dir ~/.lmstudio/models -r
```

The init wizard also offers the LM Studio directory on its own when it
exists. Ids, sampling profiles and a default model are then yours to adjust
in one YAML file. The local server surface is the same shape LM Studio's is,
plus Anthropic Messages on the same port.
