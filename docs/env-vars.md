# Environment variables

Most settings are a flag or a config key, and that is the usual way to set
them. Use an environment variable for a setting that has neither, or for a
quick A/B run without editing the config:

```sh
GMLX_PULL_RETRIES=30 gmlx pull hf:org/repo/model.gguf
MLX_VLM_RESIDENT_BUDGET_GB=48 gmlx serve
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
| `GMLX_TOOL_PREFLIGHT` | `1` | `0` skips the fit check that `run` and `chat` make before a load, the check behind `cannot fit:` refusals. |
| `GMLX_NO_FAMILY_DEFAULTS` | Unset | Any value turns off the family sampling defaults on bare-path `run` and `chat`, as `--no-family-defaults` does. |
| `XDG_CACHE_HOME` | `~/.cache` | Root of the `gmlx/` cache directory, which [Where files are on disk](troubleshooting.md#where-files-are-on-disk) lists. |
| `XDG_DATA_HOME` | `~/.local/share` | Root of the `gmlx/` data directory, with saved chat sessions and the assistant's memory. |

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
| `GMLX_OVERCOMMIT` | `0` | `1` loads a model even when it does not fit beside the resident models. |

## Runtime

| Name | Default | Meaning |
|------|---------|---------|
| `GMLX_CACHE_LIMIT_GB` | Sized from the weights | MLX buffer cache limit of `serve` in GiB, over `server.cache_limit_gb`. `0` disables it, and `unlimited` lifts the limit. |
| `GMLX_DECODE_ARENA_GB` | What the memory ceiling leaves | Size of a streamed model's expert [arena](glossary.md#arena) in GiB. Lower it to keep a second model loaded. |
| `GMLX_GOVERNOR` | `1` | `0` turns off the [governor](glossary.md#governor) that sheds requests when memory runs out. |
| `GMLX_SPARSE_ATTN` | `0` | `1` turns on top-k sparse attention for deep decode, which is lossy. |
| `GMLX_KVARN_BITS` | Unset | Separate kvarn widths for keys and values, such as `k6v5`, over `kv_bits`. |

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
