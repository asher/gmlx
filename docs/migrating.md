# Migrating from llama.cpp, Ollama, or LM Studio

gmlx runs the same GGUF files those tools use, so the models themselves move
with no conversion. This page maps the rest: what carries over directly, what
has a different name, and what works differently on purpose.

## Coming from llama.cpp / llama-server

Your GGUFs work as-is: `gmlx run model.gguf` is the moral equivalent of
`llama-cli -m model.gguf`, and `gmlx serve model.gguf` of `llama-server`.
The default port is the same 8080, and generation runs until the model stops
by default, like `-n -1`.

Common flag equivalents:

| llama.cpp | gmlx | Notes |
|-----------|------|-------|
| `-m model.gguf` | positional `model.gguf` | sharded files: point at the first shard |
| `-n N` | `--max-tokens N` | default is until-EOS on both |
| `--temp`, `--top-k`, `--top-p`, `--min-p` | same names | defaults come from each model family's card, so bare `run`/`chat` is already tuned |
| `-c N` (context size) | none | the window comes from the GGUF's own metadata; `--max-kv-size N` bounds cache memory with a rotating cache instead |
| `--rope-scaling` / `--yarn-*` | none | metadata-driven; `GMLX_ROPE_FACTORS` exists as an expert escape hatch ([cli.md](cli.md#environment-variables)) |
| `-ngl` (GPU layers) | none needed | everything runs on the GPU; `--stream-experts` / `--stream-cpu` are the deliberate over-RAM MoE placements ([streaming.md](streaming.md)) |
| `--cache-type-k/-v q8_0` | `--kv-bits 8` (+ `--kv-group-size`) | same purpose, mlx-lm quantized KV cache |
| `--draft-model`, `--spec-draft-n-max` | `--draft-gguf`, `--draft-block-size` | native-MTP models (Qwen3.5/3.6, Hy3, DS-V4) need no companion drafter at all |
| `--chat-template` | `--chat-template STR\|PATH` | per-model in server configs (`overrides: {chat_template: ...}`) |
| `--ignore-eos` | `--ignore-eos` | same benchmarking semantics |
| `--api-key K` | `server.api_key` in the YAML config | config-only by design, so the key never lands in `ps` output or shell history |
| `--parallel N` | none | continuous batching admits requests automatically; residency is bounded by `--budget-gb` instead |
| `--lora adapter` | `--adapter adapter.gguf` | llama.cpp-format adapter GGUFs interop in both directions ([lora.md](lora.md)) |

`/v1/completions` is served with a minimal surface (single string prompt,
one choice); `/v1/chat/completions` is the primary route, and Anthropic
Messages and OpenAI Responses run on the same port. Per-request details:
[server-config.md](server-config.md#api-capabilities).

## Coming from Ollama

What carries over: any GGUF you can point at. What does not: Ollama's model
store and API.

- Ollama's library lives as sha-named blobs, not `.gguf` files, so it cannot
  be pointed at directly; re-download the models you use with `gmlx pull`
  (`gmlx validate hf:<org>/<repo>` lists every variant first).
- gmlx speaks the OpenAI, Anthropic, and OpenAI Responses APIs, not the
  Ollama API (`/api/generate`, `/api/chat`). Clients configured for an
  OpenAI-compatible endpoint work unchanged; Ollama-native integrations need
  their OpenAI mode, pointed at port 8080 (not 11434).
- Modelfile parameters map onto the YAML config: `num_predict` is the server
  `--max-tokens` default, sampling knobs live per model or in `profiles:`
  blocks, and `SYSTEM` becomes `system:` ([server-config.md](server-config.md)).
- Keep-alive/unload behavior is the residency system: idle TTL, LRU under a
  byte budget, `--pin` for always-resident models.

## Coming from LM Studio

Your existing library serves as-is - the files are plain GGUFs:

```sh
gmlx init --models-dir ~/.lmstudio/models -r
```

The init wizard also offers the LM Studio directory on its own when it
exists. Ids, sampling profiles, and a default model are then yours to adjust
in one YAML file; the local server surface (OpenAI API, `/v1/models`) is the
same shape LM Studio's is, plus Anthropic Messages on the same port.

## Why serve from gmlx

Beyond the kernel-level speed on K-quants ([performance.md](performance.md)):
warm config reload (SIGHUP / `POST /v1/reload`) without dropping residents;
residency controls (byte budget, pinning, idle unload, keep tiers);
loopback-by-default binding that refuses a wide bind without a key;
cross-request prompt caching with an optional SSD tier; MTP speculative
decoding on served models; and one-command client hookups
(`gmlx launch claude-code`, `open-webui`, ...) that never touch your
dotfiles ([launch.md](launch.md)).
