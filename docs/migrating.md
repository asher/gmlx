# Migrating from llama.cpp, Ollama or LM Studio

gmlx runs the same GGUF files those tools use, so the models themselves are
reused with no conversion. This page maps the rest: what transfers
directly, what has a different name and what is different by design.

## Coming from llama.cpp

`gmlx run model.gguf` is the equivalent of `llama-cli -m model.gguf`, and
`gmlx serve model.gguf` of `llama-server`, on the same default port 8080.

| llama.cpp | gmlx | Notes |
|-----------|------|-------|
| `-m model.gguf` | positional `model.gguf` | for sharded files, point at the first shard |
| `-n N` | `--max-tokens N` | default is until end-of-sequence on both |
| `--temp`, `--top-k`, `--top-p`, `--min-p` | same names | defaults come from each model family's card, so bare `run` and `chat` are already tuned |
| `-c N` | none | the window comes from the GGUF metadata, and `--max-kv-size N` bounds cache memory with a rotating cache |
| `--rope-scaling`, `--yarn-*` | none | read from the GGUF metadata, with no override |
| `-ngl` | none needed | everything runs on the GPU, and `--stream-experts` and `--stream-cpu` are the over-RAM MoE placements in [streaming.md](streaming.md) |
| `--cache-type-k/-v q8_0` | `--kv-bits 8` | same purpose, with `--kv-group-size` |
| `--draft-model`, `--spec-draft-n-max` | `--draft-gguf`, `--draft-block-size` | models with a native head need no companion drafter |
| `--chat-template` | `--chat-template STR_OR_PATH` | per model in server configs under `overrides` |
| `--ignore-eos` | `--ignore-eos` | same benchmarking semantics |
| `--api-key K` | `server.api_key` in the config | config-only, so the key never appears in process listings or shell history |
| `--parallel N` | none | continuous batching admits requests automatically, and `--budget-gb` bounds residency |
| `--lora adapter` | `--adapter adapter.gguf` | [lora.md](lora.md) covers adapter interoperation in both directions |

The server speaks the OpenAI, Anthropic Messages and OpenAI Responses APIs
on one port, and [api.md](api.md) lists what each honors.

## Coming from Ollama

Any GGUF file on disk can be reused, but Ollama's model store and API do
not carry over.

- Ollama's library is stored as sha-named blobs rather than `.gguf` files,
  and the blobs cannot be served directly. Re-download the models you use
  with `gmlx pull`.
  `gmlx validate hf:<org>/<repo>` lists the available quants first.
- The Ollama API is not implemented. Clients configured for an
  OpenAI-compatible endpoint work unchanged, and Ollama-native integrations
  need their OpenAI mode, pointed at port 8080.
- Modelfile parameters map onto the config in
  [server-config.md](server-config.md). `num_predict` becomes the
  `max_tokens` sampling key, set for a model or in `profiles:` like the
  other sampling keys, and `SYSTEM` becomes `system:`.
- Keep-alive and unload behavior is the [residency
  system](server-config.md#residency), with an idle timeout, LRU eviction
  under a byte budget and `pin` for always-resident models.

## Coming from LM Studio

Your existing library serves as it is, since the files are plain GGUFs:

```sh
gmlx init --models-dir ~/.lmstudio/models -r
```

The init wizard also offers the LM Studio directory unprompted when it
exists, and ids, sampling profiles and a default model can then be adjusted
in the YAML file. Clients that used LM Studio's OpenAI-compatible endpoint
work against this server unchanged, and Anthropic Messages is available on
the same port.
