# Environment variables

Every environment variable a user can set, in one place. Most settings are
also reachable as a flag or a config key; when the same setting is set more
than one way the precedence is flag, then config key, then environment
variable. The named exception is `GMLX_CACHE_LIMIT_GB`, which wins over
`server.cache_limit_gb` so a benchmark run can pin the MLX buffer-cache limit
without editing the config.

Anything not listed here or in
[internals/debug-switches.md](internals/debug-switches.md) is internal and may
change meaning or disappear between releases.

## Load and cache keys

These are upstream mlx-vlm variables. gmlx sets them for you from the
`load:` and `cache:` blocks of the config
([server-config.md](server-config.md#param-key-reference)), per model, so a
config key is the normal way to set them. Exporting one applies it to every
model the process loads.

| Variable | Config key |
|----------|------------|
| `KV_BITS` | `load.kv_bits` |
| `KV_GROUP_SIZE` | `load.kv_group_size` |
| `KV_QUANT_SCHEME` | `load.kv_quant_scheme` |
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

`PREFILL_STEP_SIZE` is likewise mlx-vlm's own variable for the prefill chunk
size; prefer `--prefill-step-size` or `server.prefill_step_size`.
`TOP_LOGPROBS_K` caps the `top_logprobs` a request may ask for
([api.md](api.md#logprobs)).

## Residency

| Variable | Meaning |
|----------|---------|
| `MLX_VLM_RESIDENT_BUDGET_GB` | Resident weight-byte budget in GB. Consulted when neither `--budget-gb` nor `server.budget_gb` sets one; otherwise the flag/config value wins. |
| `MLX_VLM_MAX_RESIDENT_MODELS` | Secondary cap on resident model count. Same precedence: a fallback below `--max-models` / `server.max_models`. |
| `MLX_VLM_PINNED_MODELS` | Comma-separated model paths to pin; unioned with `--pin` and `pin: true` entries, so it always adds pins. |
| `MLX_VLM_RESIDENT_TTL_DISABLE` | `1`/`true`/`yes`/`on` disables the idle-TTL reaper entirely (LRU-under-pressure still applies). |
| `MLX_VLM_RESIDENT_TTL_TICK` | Reaper wake-up interval in seconds (default `30`). |
| `MLX_VLM_TOKEN_QUEUE_TIMEOUT` | Seconds a queued request waits for a decode slot before the server returns 503. `server.token_queue_timeout_s` sets the same limit from the config. |

## Server

These change how `gmlx serve` schedules and admits requests. Each has a
config key or flag that is the normal way to set it; the variable exists so
a live server can be re-gated for an A/B without a restart.

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
| `GMLX_DECODE_FAST_DISK` | The `stream_fast_disk` recipe: `auto`, `on` or `off` (same as `--stream-fast-disk`). |

## Runtime

| Variable | Meaning |
|----------|---------|
| `GMLX_STREAM_GPU_TOKENS` | `--stream-experts` prefill staging threshold: offloaded expert calls with at least this many tokens run on the GPU stream (same zero-copy buffers; prefill is a GEMM workload the CPU loses badly). Default `32`; `0` keeps every expert call on CPU (conservative for models far larger than RAM). |
| `GMLX_STREAM_PREFETCH=0` | Disable sequential expert prefetch for streaming-mode (over-wired-budget) `--stream-cpu` / `--stream-experts` models. Default on: prefill-sized expert calls advise the kernel (`F_RDADVISE`) two layers ahead and pace the lazy graph per layer, reading expert stacks at sequential bandwidth instead of demand-faulting. |
| `GMLX_DECODE_ARENA_GB` | Decode-feeder arena size override in GB (see `--decode-feeder`). Default: what the memory ceiling leaves after the every-token weights, the KV room and the prefill ring, clamped to the memory reclaimable at load. The ceiling is the serve governor's: Metal's recommended working set less the margin, never closer to physical RAM than `GMLX_GOV_RESERVE_GB` (see [streaming.md](streaming.md#how-big-a-model-can-this-box-stream)). The prefill ring keeps its own room under the ceiling, and a host floor (`GMLX_DECODE_RAM_FLOOR_GB`) stays free under it for the rest of the box. `GMLX_DECODE_ARENA_RAM_FRAC` caps the ceiling at a fraction of physical RAM when set. It has no default. |
| `GMLX_STREAM_KV_CTX` | Tokens of KV cache the decode arena leaves room for (default `32768`, capped at the trained context). The cost per token comes from the GGUF header. The room also holds the prefill score transient and the admission reserve. At decode it is the serve governor's headroom. Raise it for deep prompts. Raise `GMLX_STREAM_KV_WIDTH` (default `1`) for concurrent streams. Both cost arena slots. The load log prints the budget as `[stream] memory budget:`. |
| `GMLX_PREFILL_NOCACHE=0` | Prefill ring reads go through the page cache again. Default off: a ring pass reads every routed expert once, and through the cache it evicts the rest of the box for pages it never reads again. |
| `GMLX_DECODE_KV_RESERVE_GB` | Replace the priced KV room with a flat reserve in GB. Also the fallback (`8`) when the header cannot be priced. |
| `GMLX_ARENA_STAGE_MAX_TOKENS` | Largest expert call served router-aware (decode-feeder arena or partial ring staging) instead of whole-layer staging. Default `64`; above it a chunk routes to nearly every expert anyway. |
| `GMLX_ARENA_SPLIT_MAX_TOKENS` | Largest expert call the decode arena serves by token-splitting when its routed union exceeds the arena's slots (a chat-turn prefill after decode, or a wide speculative verify batch). Halves recurse until each piece fits, keeping reads on the arena's read pool instead of the CPU page-cache gather. Default `256`; `0` disables. |
| `GMLX_DECODE_PRESSURE` | Set `0` to keep the decode-feeder arena at its sized capacity regardless of system memory pressure. Default on: the arena shrinks (keeping its most popular experts) when the kernel reports pressure and regrows once pressure clears and reclaimable RAM returns. |
| `GMLX_GOVERNOR=0` | Disable the runtime memory governor (default on). Band and shed counters, the kernel reclaimable sample, and the armed floor show at `/v1/metrics`. |
| `GMLX_GOV_KERNEL_FLOOR_GB` | Kernel reclaimable floor in GB (free + purgeable + speculative + file-backed pages, read from the kernel every governed tick). Below it the governor goes red at once. A first dip reclaims the deficit plus half a floor from the registered caches (a decode arena steps down by a quarter). A collapse reclaims all of them. Then it clears the MLX buffer cache, and fails the largest request if that did not clear the floor. Prefix-cache block stores stop at the same floor while a governor is installed. Default: the lower of `4` and 10% of RAM; `0` disables. This is the counter that predicts a free-page freeze; MLX cannot see it because its buffer cache reads as free inside the process and wired to the kernel. |
| `GMLX_GOV_RESERVE_GB` | Bytes left to the kernel and every other process: the ceiling on the server's tracked memory is the lower of Metal's recommended working set less the margin and physical RAM minus this reserve. Default `max(8, 10% of RAM)`. |
| `GMLX_PIN_WEIGHTS=0` | Do not lock the every-token weights of a streamed model in memory. Default on, skipped with a printed reason when that set would exceed 60% of RAM. |
| `GMLX_STREAM_PLE=0` | Disable the streamable lookup-table tier on architectures that declare one; `1` forces the table onto the CPU stream on a model that fits, for overhead measurement. |
| `GMLX_GPU_KEEPWARM=0` | Disable GPU keep-warm (default on for streamed installs; see `--gpu-keepwarm`). |
| `GMLX_GPU_RESIDENT=0` | Skip wiring the every-token (non-expert) weights into the Metal residency set on streamed installs (default on: command buffers otherwise re-wire those pages on every use). |
| `GMLX_KEEPWARM_IDLE_S` | Seconds without streamed-decode activity before the keep-warm heartbeat parks (default `1`; `0` beats continuously). |
| `GMLX_DECODE_LOOKAHEAD=0` | Disable lookahead expert prestage on the decode feeder. Default on: each MoE layer runs the *next* MoE layer's router on its own input (the residual moves little between adjacent sublayers, so recall is far above previous-token reuse) and pre-reads the predicted arena misses while the current layer computes. Lossless - predictions move bytes, never routing. `GMLX_DECODE_LOOKAHEAD_K` caps the ranked predictions considered per call (default `6`); `GMLX_DECODE_LOOKAHEAD_WORKERS` sizes the dedicated read pool (default `6`); `GMLX_DECODE_LOOKAHEAD_NORM` picks the prediction input (`ratio` default, `raw` skips the norm-gain rescale); `GMLX_DECODE_LOOKAHEAD_MIN_P` sets the per-rank reliability floor below which a prediction rank stops being submitted (default `0.5`); `GMLX_DECODE_LOOKAHEAD_CANCEL=0` keeps unrouted predictions reading to completion instead of cancelling the unstarted ones at settle; `GMLX_DECODE_LOOKAHEAD_IOPOL=0` runs the read pool at default disk-I/O priority instead of the utility tier. |
| `GMLX_DECODE_RAM_FLOOR_GB` | Host floor in GB kept free for the rest of the box when the decode arena is sized: under the memory ceiling, against the RAM reclaimable at load, and on every pressure-driven regrow. Default: 5% of RAM, at least `4`. `GMLX_DECODE_PAGECACHE_GB` adds to it. |
| `GMLX_DECODE_PAGECACHE_GB` | Page-cache reserve inside the decode-arena RAM floor (default `2.5`). Buffered read paths (prefill feeder, CPU-mmap fallback) collapse when the cache is starved; the floor also clamps an oversized `GMLX_DECODE_ARENA_GB` (`GMLX_DECODE_ARENA_FORCE=1` restores the unclamped override). |
| `GMLX_CACHE_LIMIT_GB` | MLX buffer-cache limit for `serve`, in GiB (wins over `server.cache_limit_gb`). Negative or `off`/`none`/`unlimited` forces an unbounded cache; `0` disables buffer caching. See [performance.md](performance.md). |
| `GMLX_NATIVE_FP` | Layout for MXFP4/NVFP4 expert tensors: `wire` (zero-copy GGUF wire bytes, loads in seconds), `packed` (eager repack into MLX's layout), or the default `auto` (wire when a streaming placement is requested or the file nears the wired budget). See [streaming.md](streaming.md). |
| `GMLX_CASCADE_SDPA=0` | Disable the shared-prefix cascade decode route. Default on: concurrent streams that share a prompt-cached prefix read it once per step for the whole batch instead of once per stream. Exact; see [performance.md](performance.md#serving-concurrent-requests). |
| `GMLX_CASCADE_MIN_P` | Smallest shared-prefix token length the cascade route claims (default `1024`). |
| `GMLX_SPARSE_ATTN=1` | Enable top-k sparse attention for deep decode (lossy, default off): past `GMLX_SPARSE_MIN_S` tokens each step attends only the best-scoring KV pages within the `GMLX_SPARSE_K` budget, making attention cost depth-flat. See [performance.md](performance.md#sparse-attention-at-depth). |
| `GMLX_SPARSE_K` | Sparse-attention kept-token budget (default `2048`). |
| `GMLX_SPARSE_MIN_S` | Depth in tokens where sparse attention engages (default `8192`). |
| `GMLX_NO_FAMILY_DEFAULTS` | Disable the family model-card sampling defaults on bare-path `run` / `chat` (same as `--no-family-defaults`). |
| `GMLX_DRAFT_BLOCK_SIZE` | MTP draft tokens per round for `serve` (same as `--draft-block-size`). |
| `GMLX_MTP_WIDTH_CAP` | `serve`: run MTP only while at most this many requests decode together (`0` = uncapped; same as `--speculative-width-cap`). Overrides every model's `speculative_width_cap`; read per round, so a live server can be re-gated for an A/B. Drafters limited to one sequence clamp it. |
| `GMLX_IGNORE_EOS=1` | `serve`: never stop on EOS; decode every request to `max_tokens` (same as `--ignore-eos`; forced-length throughput benchmarking). |
| `GMLX_API_KEY` | Client-side default key for `ps` (sent to `/v1/metrics`) when `--api-key` isn't passed. Not a `serve` source; the server reads its key only from `server.api_key` in the config. |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face auth for `validate` / `pull` on gated or private repos. |
| `XDG_CACHE_HOME` | Where `chat` keeps its prompt history (`$XDG_CACHE_HOME/gmlx/chat_history[.ptk]`), and where backgrounded servers keep their runfiles and logs (`$XDG_CACHE_HOME/gmlx/`). |
