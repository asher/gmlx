# Environment variables

Most settings are a flag or a config key, and that is the usual way to set
them. Use an environment variable for a setting that has neither, or for a
quick A/B run without editing the config:

```sh
GMLX_PULL_RETRIES=30 gmlx pull hf:org/repo/model.gguf
GMLX_STREAM_KV_CTX=16384 gmlx validate model-00001-of-00004.gguf
```

An exported variable applies to every model the process loads and to every
server started from that shell. When a flag or config key also sets the
value, the flag wins, then the key, then the variable, except for the two
variables that
[Flags and environment variables](config.md#flags-and-environment-variables)
names. Tuning and debug variables are in
[Debug switches](internals/debug-switches.md). Any variable on neither page
is internal and may change between releases.

- [Commands](#commands)
- [Server](#server)
- [Runtime](#runtime)
- [Load and cache keys](#load-and-cache-keys)

## Commands

| Name | Default | Meaning |
|------|---------|---------|
| `GMLX_API_KEY` | Unset | Key that `ps` and `systemone` send when `--api-key` is not passed. The server reads its own key only from `server.api_key`. |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | The `hf auth login` token | Hugging Face token for `pull` and `validate`, checked in that order. Needed for gated or private repos. |
| `GMLX_PULL_RETRIES` | `10` | Failed attempts in a row that `pull` accepts on one file. An attempt that moves bytes resets the count, and `0` fails at once. |
| `GMLX_PULL_TIMEOUT` | `60` | Socket timeout of `pull` in seconds, which also bounds one stalled read. |
| `GMLX_TOOL_PREFLIGHT` | `1` | `0` skips the fit checks that `run`, `chat` and `image generate` make before a load. `run` and `chat` refuse with `cannot fit:`. |
| `GMLX_OVERCOMMIT` | `0` | `1` skips the memory checks before a load in `run`, `chat`, `image generate` and `video`. Past RAM, `run`, `chat` and the upsampler still stream or refuse. |
| `GMLX_NO_FAMILY_DEFAULTS` | Unset | Any value turns off the family sampling defaults on bare-path `run` and `chat`, as `--no-family-defaults` does. |
| `GMLX_BASE_URL` | The settings file's `base_url`, else the address `gmlx status` reports, else `http://127.0.0.1:8080` | Server root that the [ComfyUI node pack](comfyui.md) calls. `gmlx launch comfyui` sets it, and the pack also reads `GMLX_API_KEY`. |
| `GMLX_COMFY_SETTINGS` | `~/.config/gmlx/comfyui.json` | The ComfyUI node pack's settings: `base_url`, `api_key` and the install record. ComfyUI must see the same value, which Comfy Desktop cannot get from a shell. |
| `GMLX_DEFAULT_IMAGE_MODEL` | The model `gmlx comfyui install` recorded, else `z-image-turbo` | Image model that the ComfyUI node pack selects first. `gmlx launch comfyui` sets it. |
| `XDG_CACHE_HOME` | `~/.cache` | Root of the `gmlx/` cache directory, which [Where files are on disk](troubleshooting.md#where-files-are-on-disk) lists. |
| `XDG_DATA_HOME` | `~/.local/share` | Root of the `gmlx/` data directory, with saved chat sessions, the assistant's memory and the copy of the ComfyUI node pack. |

## Server

| Name | Default | Meaning |
|------|---------|---------|
| `MLX_VLM_RESIDENT_BUDGET_GB` | Unset | GB of resident weights when neither `--budget-gb` nor `server.budget_gb` sets a budget. |
| `MLX_VLM_MAX_RESIDENT_MODELS` | Unset | Most models resident at once when neither `--max-models` nor `server.max_models` sets a cap. |
| `MLX_VLM_PINNED_MODELS` | Unset | Comma-separated model paths to pin, added to `--pin` and `pin: true` entries. |
| `MLX_VLM_TOKEN_QUEUE_TIMEOUT` | `1800` | Seconds without a next token before a request fails. `server.token_queue_timeout_s` sets the same limit and wins. |
| `TOP_LOGPROBS_K` | `0` | Most `top_logprobs` a request may get, up to 20, as [Logprobs](api.md#logprobs) describes. |
| `GMLX_QUEUE_DEPTH_CAP` | Twice the decode batch | Waiting requests the server admits before it answers 503. `0` disables the cap. |
| `GMLX_PREFLIGHT_MEM` | `1` | `0` turns off the memory check that answers 400 when a prompt cannot fit. |
| `GMLX_OVERCOMMIT` | `0` | `1` skips the memory checks before a load, so a MoE [keeps its experts in memory](streaming.md#when-a-moe-streams-its-experts), except as that page says. |
| `GMLX_MTP_WIDTH_CAP` | Each drafter's default | Speculative width cap for every model, over each model's `speculative_width_cap`. `0` removes the cap. |
| `GMLX_CONTEXT_COPY` | Unset | `1` turns on [context copy](speculative-decoding.md#context-copy) for every model, `0` turns it off. Unset, each model's `context_copy` decides. |
| `GMLX_CONTEXT_COPY_MAX` | `24` | Longest context copy in tokens, up to 31. Under 4 turns copying off. A block drafter's draft length keeps the output token-identical to a run without copies. |
| `GMLX_IMAGE_CHAT_SHARE` | `0.5` | Share of the GPU that chat replies get while an image runs, `0` to `0.9`, or `off`, as [Image serving](internals/image-serving.md) describes. |

## Runtime

| Name | Default | Meaning |
|------|---------|---------|
| `GMLX_CACHE_LIMIT_GB` | Sized from the weights | MLX buffer cache limit of `serve`, `run` and `chat` in GiB, over `server.cache_limit_gb` in `serve`. `0` disables it, and `unlimited` lifts the limit. |
| `GMLX_DECODE_ARENA_GB` | What the memory ceiling leaves | Size of a streamed model's expert [arena](glossary.md#arena) in GiB. Lower it to keep a second model loaded. |
| `GMLX_GOVERNOR` | `1` | `0` turns off the [governor](glossary.md#governor) that sheds requests when memory runs out. |
| `GMLX_SPARSE_ATTN` | `0` | `1` turns on top-k sparse attention for deep decode, which is lossy. |
| `GMLX_SPARSE_K` | `2048` | Tokens that sparse attention reads in each step. |
| `GMLX_SPARSE_MIN_S` | `8192` | Context depth in tokens where sparse attention starts. |
| `GMLX_STREAM_KV_CTX` | `32768`, at most the trained context | Tokens of KV cache that a streamed model keeps room for. A lower value leaves more memory for the expert arena. |
| `GMLX_KVARN_BITS` | Unset | Separate kvarn widths for keys and values, such as `k6v5`, over `kv_bits`. |
| `GMLX_IMAGE_ALLOW_FP16` | `0` | `1` lets `gmlx pull`, `gmlx image` and `gmlx serve` run Z-Image in float16 on an M1 or M2, with its VAE in float32. Its images differ from an M3's. |
| `GMLX_NORM_DTYPE` | `auto` | Width of a language model's norm weights. `auto` keeps each model's default, `fp32` keeps them float32, and `activation` casts them to bf16 (fp16 on M1, M2). |
| `GMLX_QWEN35_VISION_FP32` | `0` | `1` loads the vision encoder of a Qwen3.5 or Qwen3.6 pair in float32. Image features are more exact, and an image request's first token comes about 12% later. |

## Load and cache keys

gmlx sets these variables for each model from the `load` and `cache`
blocks of the config, which [Model loading](config.md#model-loading) and
[Prompt cache](config.md#prompt-cache) describe. Set the config key
instead. `PREFILL_STEP_SIZE` is set the same way for the whole server.

| Variable | Config key |
|----------|------------|
| `KV_BITS` | `load.kv_bits` |
| `KV_GROUP_SIZE` | `load.kv_group_size` |
| `KV_QUANT_SCHEME` | `load.kv_quant_scheme` |
| `KV_TAIL_TOKENS` | `load.kv_tail_tokens` |
| `MAX_KV_SIZE` | `load.max_kv_size` |
| `QUANTIZED_KV_START` | `load.quantized_kv_start` |
| `APC_ENABLED` | `cache.enabled` |
| `APC_BLOCK_SIZE` | `cache.block_size` |
| `APC_NUM_BLOCKS` | `cache.num_blocks` |
| `APC_EXACT_CACHE_ENTRIES` | `cache.exact_entries` |
| `APC_HASH` | `cache.hash` |
| `APC_DISK_PATH` | `cache.disk.path` |
| `APC_DISK_MAX_GB` | `cache.disk.max_gb` |
| `APC_DISK_WORKERS` | `cache.disk.workers` |
| `APC_DISK_READ_MODE` | `cache.disk.read_mode` |
| `APC_DISK_NAMESPACE` | `cache.disk.namespace` |
| `PREFILL_STEP_SIZE` | `server.prefill_step_size` |
