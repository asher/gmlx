# Environment variables

Every environment variable a user can set, in one place. Most settings are
also reachable as a flag or a config key; when the same setting is set more
than one way the precedence is flag, then config key, then environment
variable. The named exception is `GMLX_CACHE_LIMIT_GB`, which overrides
`server.cache_limit_gb` so a benchmark run can pin the MLX buffer-cache limit
without editing the config.

Anything not listed here or in
[internals/debug-switches.md](internals/debug-switches.md) is internal and may
change meaning or disappear between releases.

## Load and cache keys

These are upstream mlx-vlm variables. gmlx sets them from the
`load:` and `cache:` blocks of the config
([server-config.md](server-config.md#param-key-reference)), per model, so a
config key is the normal way to set them. Exporting one applies it to every
model the process loads.

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

`KV_KEY_BITS` and `KV_VALUE_BITS` set split key and value widths for kvarn
KV server-wide; they have no config key and override `GMLX_KVARN_BITS`.
`PREFILL_STEP_SIZE` is likewise mlx-vlm's own variable for the prefill chunk
size; prefer `--prefill-step-size` or `server.prefill_step_size`.
`TOP_LOGPROBS_K` caps the `top_logprobs` a request may ask for
([api.md](api.md#logprobs)).

## Residency

| Variable | Meaning |
|----------|---------|
| `MLX_VLM_RESIDENT_BUDGET_GB` | Resident weight-byte budget in GB. Consulted when neither `--budget-gb` nor `server.budget_gb` sets one; otherwise the flag/config value is used. |
| `MLX_VLM_MAX_RESIDENT_MODELS` | Secondary cap on resident model count. Same precedence: a fallback below `--max-models` / `server.max_models`. |
| `MLX_VLM_PINNED_MODELS` | Comma-separated model paths to pin; unioned with `--pin` and `pin: true` entries, so it always adds pins. |
| `MLX_VLM_RESIDENT_TTL_DISABLE` | `1`/`true`/`yes`/`on` disables the idle-TTL reaper entirely (LRU-under-pressure still applies). |
| `MLX_VLM_RESIDENT_TTL_TICK` | Reaper wake-up interval in seconds (default `30`). |
| `MLX_VLM_TOKEN_QUEUE_TIMEOUT` | Seconds a queued request waits for a decode slot before the server returns 503. `server.token_queue_timeout_s` sets the same limit from the config. |

## Server

These change how `gmlx serve` schedules and admits requests. Each has a
config key or flag that is the normal way to set it; the variable exists so
a live server can be reconfigured for an A/B without a restart.

| Variable | Meaning |
|----------|---------|
| `GMLX_DECODE_PREFILL_RATIO` | The `server.decode_prefill_ratio` value, read per scheduler tick. |
| `GMLX_DECODE_PREFILL_FLOOR` | The decode-rate floor `auto` pacing protects, as a share of a stream's batched rate (default `0.5`). |
| `GMLX_PREFILL_TICK_MS` | The `server.prefill_tick_ms` value, read per chunk. |
| `GMLX_PREFILL_MIN_STEP` | Smallest chunk the tick budget may halve down to, in tokens. |
| `GMLX_DECODE_BATCH` | Requests that decode together in one step (default `8`; `0` restores the upstream 32). |
| `GMLX_QUEUE_DEPTH_CAP` | Waiting requests admitted before the server answers 503 (default 2x the decode batch; `0` disables). |
| `GMLX_SSE_KEEPALIVE_S` | Seconds between SSE keepalive comments while a stream is silent (default `15`; `0` disables). |
| `GMLX_PREFLIGHT_MEM=0` | Disable the memory preflight that answers 400 when a prompt cannot fit. |
| `GMLX_FAITHFUL_HISTORY=0` | Restore mlx-vlm's stock chat-history rebuild, which drops `reasoning_content` from plain assistant turns. |
| `GMLX_MTP_PREEMPT=0` | Keep a speculating stream from converting to plain decode when a batch grows past the width cap. |
| `GMLX_MTP_RESUME=0` | Keep a gated batch plain instead of re-arming speculation when it drains. |
| `GMLX_DECODE_FAST_DISK` | The `stream_fast_disk` policy: `auto`, `on` or `off` (same as `--stream-fast-disk`). |

## Runtime

Streaming, memory-governor and kernel-route switches. The mechanisms are
explained in [streaming.md](streaming.md) and
[performance.md](performance.md); the rows here say what each variable
changes.

| Variable | Meaning |
|----------|---------|
| `GMLX_STREAM_GPU_TOKENS` | Expert calls with at least this many tokens run on the GPU stream during streamed prefill (default `32`; `0` keeps every expert call on the CPU). |
| `GMLX_STREAM_PREFETCH=0` | Disable sequential expert prefetch on streamed models. Default on: prefill-sized expert calls advise the kernel two layers ahead. |
| `GMLX_DECODE_ARENA_GB` | Decode arena size override in GB. Default: what the memory limit leaves after the every-token weights, KV room and prefill ring. |
| `GMLX_DECODE_ARENA_RAM_FRAC` | Cap the arena size limit at a fraction of physical RAM. No default. |
| `GMLX_DECODE_ARENA_FORCE=1` | Honor an oversized `GMLX_DECODE_ARENA_GB` instead of clamping it to the host floor. |
| `GMLX_STREAM_KV_CTX` | Tokens of KV cache the arena leaves room for (default `32768`, capped at the trained context). Raise it for deep prompts. |
| `GMLX_STREAM_KV_WIDTH` | Concurrent streams the KV room is sized for (default `1`). Each uses arena slots. |
| `GMLX_KVARN_BITS` | Split key and value widths for kvarn KV in `k6v5` form; overrides the width `kv_bits` gives both. A value not of that form is ignored with a warning. |
| `GMLX_DECODE_KV_RESERVE_GB` | Replace the estimated KV room with a flat reserve in GB; also the fallback of `8` when the KV size cannot be computed from the header. |
| `GMLX_PREFILL_NOCACHE=0` | Route prefill ring reads through the page cache again. Default off, since a ring pass reads each expert once. |
| `GMLX_ARENA_STAGE_MAX_TOKENS` | Largest expert call served router-aware instead of by whole-layer staging (default `64`). |
| `GMLX_ARENA_SPLIT_MAX_TOKENS` | Largest expert call the arena serves by token-splitting when its routed set exceeds the arena (default `256`; `0` disables). |
| `GMLX_DECODE_PRESSURE=0` | Keep the arena at its sized capacity under memory pressure. Default on: it shrinks, keeping its most routed experts, and regrows when pressure clears. |
| `GMLX_DECODE_RAM_FLOOR_GB` | Host floor kept free for the rest of the machine when the arena is sized. Default 5% of RAM, at least `4`. |
| `GMLX_DECODE_PAGECACHE_GB` | Page-cache reserve added to the host floor (default `2.5`). Buffered read throughput drops sharply when the page cache has too little memory. |
| `GMLX_PIN_WEIGHTS=0` | Do not lock the every-token weights of a streamed model in memory. Default on, skipped with a printed reason above 60% of RAM. |
| `GMLX_GPU_RESIDENT=0` | Skip wiring the every-token weights into the Metal residency set on streamed models. |
| `GMLX_STREAM_PLE=0` | Disable the streamable lookup-table tier; `1` forces the table onto the CPU stream on a model that fits, for measurement. |
| `GMLX_GPU_KEEPWARM=0` | Disable GPU keep-warm, which is on by default for streamed models. |
| `GMLX_KEEPWARM_IDLE_S` | Seconds without streamed decode before the keep-warm heartbeat pauses (default `1`; `0` runs continuously). |
| `GMLX_DECODE_LOOKAHEAD=0` | Disable lookahead expert prestage on the decode feeder. |
| `GMLX_DECODE_LOOKAHEAD_K` | Ranked predictions considered per call (default `6`). |
| `GMLX_DECODE_LOOKAHEAD_WORKERS` | Size of the dedicated prestage read pool (default `6`). |
| `GMLX_DECODE_LOOKAHEAD_NORM` | Prediction input: `ratio` (default) or `raw`, which skips the norm-gain rescale. |
| `GMLX_DECODE_LOOKAHEAD_MIN_P` | Per-rank reliability floor below which a prediction rank stops being submitted (default `0.5`). |
| `GMLX_DECODE_LOOKAHEAD_CANCEL=0` | Let unrouted predictions read to completion instead of cancelling the unstarted ones. |
| `GMLX_DECODE_LOOKAHEAD_IOPOL=0` | Run the prestage read pool at default disk priority instead of the utility tier. |
| `GMLX_GOVERNOR=0` | Disable the runtime memory governor. Its band, shed counters and floor show at `/v1/metrics`. |
| `GMLX_GOV_KERNEL_FLOOR_GB` | Reclaimable-pages floor in GB below which the governor goes red and reclaims caches. Default the lower of `4` and 10% of RAM; `0` disables. |
| `GMLX_GOV_RESERVE_GB` | RAM left to the kernel and other processes when the server's memory limit is computed. Default the larger of `8` and 10% of RAM. |
| `GMLX_CACHE_LIMIT_GB` | MLX buffer-cache limit for `serve` in GiB; overrides `server.cache_limit_gb`. Negative, `off`, `none` or `unlimited` unbounds it; `0` disables caching. |
| `GMLX_NATIVE_FP` | Layout for MXFP4 and NVFP4 expert tensors: `wire` (zero-copy file bytes), `packed` (repack at load), or `auto` (wire when streaming or near the wired budget). |
| `GMLX_CASCADE_SDPA=0` | Disable the shared-prefix cascade decode route, which reads a shared prefix once per step for the whole batch. |
| `GMLX_CASCADE_MIN_P` | Smallest shared-prefix length in tokens the cascade route handles (default `1024`). |
| `GMLX_SPARSE_ATTN=1` | Enable top-k sparse attention for deep decode (lossy, default off). |
| `GMLX_SPARSE_K` | Sparse-attention kept-token budget (default `2048`). |
| `GMLX_SPARSE_MIN_S` | Depth in tokens where sparse attention begins (default `8192`). |
| `GMLX_NO_FAMILY_DEFAULTS` | Disable the family model-card sampling defaults on bare-path `run` and `chat`; same as `--no-family-defaults`. |
| `GMLX_DRAFT_BLOCK_SIZE` | Draft tokens per speculative round for `serve`; same as `--draft-block-size`. |
| `GMLX_MTP_WIDTH_CAP` | Speculate only while at most this many requests decode together (`0` uncapped). Overrides every model's `speculative_width_cap`; read per round. |
| `GMLX_IGNORE_EOS=1` | Never stop on end-of-sequence in `serve`; same as `--ignore-eos`, for forced-length benchmarking. |
| `GMLX_API_KEY` | Client-side default key for `ps` when `--api-key` is not passed. The server reads its key only from `server.api_key`. |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Hugging Face auth for `validate` and `pull` on gated or private repos. |
| `XDG_CACHE_HOME` | Where `chat` keeps its prompt history and backgrounded servers keep their runfiles and logs, under a `gmlx/` subdirectory. |
