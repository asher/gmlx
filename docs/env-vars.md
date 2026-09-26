# Environment variables

Environment variables let a benchmark or an A/B run change one setting
without editing the config. Most of them are also a flag or a config key,
which is the usual way to set them, because an exported variable applies
to every model the process loads and to every server started from that
shell.

When more than one source sets a value, the flag wins, then the config
key, then the variable, except for the two variables that
[Flags and environment variables](config.md#flags-and-environment-variables)
names. Variables that appear neither here nor under
[Debug switches](internals/debug-switches.md) are internal and may change
meaning or disappear between releases.

- [Load and cache keys](#load-and-cache-keys)
- [Residency](#residency)
- [Server](#server)
- [Runtime](#runtime)
- [Commands](#commands)

## Load and cache keys

gmlx sets these variables for each model from the `load` and `cache`
blocks of the config, which [Model loading](config.md#model-loading) and
[Prompt cache](config.md#prompt-cache) describe. All of them except
`KV_TAIL_TOKENS` are upstream mlx-vlm variables.

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

Four more upstream variables apply to the whole server instead of one
model. `KV_KEY_BITS` and `KV_VALUE_BITS` give [kvarn](glossary.md#kvarn)
keys and values different widths, and they override `GMLX_KVARN_BITS`.
`PREFILL_STEP_SIZE` is the prefill chunk size, which `--prefill-step-size`
and `server.prefill_step_size` also set. `TOP_LOGPROBS_K` caps the
`top_logprobs` that a request may ask for, as [Logprobs](api.md#logprobs)
describes.

## Residency

| Variable | Meaning |
|----------|---------|
| `MLX_VLM_RESIDENT_BUDGET_GB` | The server keeps resident weights within this many GB when neither `--budget-gb` nor `server.budget_gb` sets a budget. |
| `MLX_VLM_MAX_RESIDENT_MODELS` | The server keeps at most this many models resident when neither `--max-models` nor `server.max_models` sets a cap. |
| `MLX_VLM_PINNED_MODELS` | The server pins these comma-separated model paths beside `--pin` and `pin: true` entries, so the variable only adds pins. |
| `MLX_VLM_RESIDENT_TTL_DISABLE` | `1`, `true`, `yes` or `on` disables the idle-TTL reaper entirely. LRU eviction under pressure still applies. |
| `MLX_VLM_RESIDENT_TTL_TICK` | The idle reaper wakes up at this interval in seconds. The default is `30`. |
| `MLX_VLM_TOKEN_QUEUE_TIMEOUT` | A request fails after this many seconds without a next token. `server.token_queue_timeout_s` sets the same limit, and `1800` applies when neither is set. |

## Server

The server variables change how `gmlx serve` schedules and admits
requests and how it drafts and stops replies. A variable read per tick or
per chunk takes effect on a running server.

| Variable | Meaning |
|----------|---------|
| `GMLX_DECODE_PREFILL_RATIO` | It sets `server.decode_prefill_ratio`, and the scheduler reads it each tick. |
| `GMLX_DECODE_PREFILL_FLOOR` | `auto` pacing protects this decode-rate floor, given as a share of a stream's batched rate. The default is `0.5`. |
| `GMLX_PREFILL_TICK_MS` | It sets `server.prefill_tick_ms`, and the server reads it each chunk. |
| `GMLX_PREFILL_MIN_STEP` | The tick budget may halve a chunk down to this many tokens. The default is `256`. |
| `GMLX_DECODE_BATCH` | This many requests decode together in a step. The default is `8`, or the capacity table's widest fit when that is smaller. `0` restores the upstream 32. |
| `GMLX_QUEUE_DEPTH_CAP` | The server admits this many waiting requests before it answers 503. The default is twice the decode batch. `0` disables the cap. |
| `GMLX_SSE_KEEPALIVE_S` | The server sends an SSE keepalive comment at this interval in seconds while a stream is silent. The default is `15`, and `0` disables them. |
| `GMLX_PREFLIGHT_MEM=0` | Disable the memory preflight that answers 400 when a prompt cannot fit. |
| `GMLX_OVERCOMMIT=1` | Skip the load gate and the ceilings that the capacity table sets, so a model loads even when it does not fit beside the resident models. |
| `GMLX_FAITHFUL_HISTORY=0` | Restore mlx-vlm's stock chat-history rebuild, which drops `reasoning_content` from plain assistant turns. |
| `GMLX_MTP_PREEMPT=0` | Make queued requests wait for a lone speculating request to finish, instead of moving it onto the batch loop so they can join. |
| `GMLX_MTP_RESUME=0` | Keep speculation off on a batch that grew past the width cap, instead of turning it back on when the batch shrinks back within the cap. |
| `GMLX_DRAFT_BLOCK_SIZE` | It sets the block size of each speculative round for `serve`, as `--draft-block-size` does. A round drafts one token fewer. |
| `GMLX_MTP_WIDTH_CAP` | It replaces `speculative_width_cap` for every model and is read each round. `0` removes the cap, but a single-sequence drafter stays at `1`. |
| `GMLX_IGNORE_EOS=1` | Never stop on end-of-sequence in `serve`, as `--ignore-eos` does. |

## Runtime

The runtime variables switch streaming, the memory governor and kernel
routes. [Models larger than memory](streaming.md) and
[Performance tuning](performance.md) explain the mechanisms.

| Variable | Meaning |
|----------|---------|
| `GMLX_STREAM_GPU_TOKENS` | On a model with `stream` set that fits in memory, expert calls of this many tokens or more run on the GPU. The default is `1`, and `0` keeps them on the CPU. |
| `GMLX_STREAM_PREFETCH=0` | Disable sequential expert prefetch on streamed models. By default, prefill-sized expert calls read the experts of the next two layers into the page cache. |
| `GMLX_STREAM_CACHE_GB` | A streamed model keeps this much MLX buffer cache, in GiB. The default is the KV room, or `4` when no KV room is reserved. |
| `GMLX_STREAM_ALLOC_LIMITS=0` | Keep the MLX allocator's default memory and cache limits on a streamed model. Every cache miss then purges the whole buffer cache. |
| `GMLX_DECODE_FAST_DISK` | It sets the `stream_fast_disk` policy to `auto`, `on` or `off`, as `--stream-fast-disk` does. |
| `GMLX_DECODE_SEED=0` | Start the decode arena empty instead of seeding it from the prefill ring with the prompt's most routed experts. |
| `GMLX_DECODE_ASYNC_GATHER=0` | Keep each streamed layer's expert gather in the next layer's eval instead of submitting it as soon as it is built. |
| `GMLX_DECODE_ARENA_GB` | It overrides the decode arena size, in GiB. The default is what the memory ceiling leaves after the every-token weights, KV room, prefill ring and host floor. |
| `GMLX_DECODE_ARENA_RAM_FRAC` | Cap the arena size limit at a fraction of physical RAM. It is unset by default. |
| `GMLX_BATCH_INVARIANT=1` | Run small float `nn.Linear` layers on a row-count-invariant kernel. Raw-array routers, router calls under 64 routed rows and training are not covered. |
| `GMLX_BATCH_INVARIANT_MAX_OUT` | The batch-invariant kernel takes float projections of up to this many outputs. The default is `512`. |
| `GMLX_DECODE_ARENA_FORCE=1` | Honor an oversized `GMLX_DECODE_ARENA_GB` instead of clamping it to the reclaimable RAM less the host floor. |
| `GMLX_STREAM_KV_CTX` | The [KV room](streaming.md#what-changes-the-limit) holds this many tokens of KV cache. The default is `32768`, capped at the trained context. |
| `GMLX_STREAM_KV_WIDTH` | The KV room is sized for this many concurrent streams. The default is `1`, and each extra stream takes its KV room out of the arena. |
| `GMLX_KVARN_BITS` | It gives kvarn keys and values separate widths in `k6v5` form and overrides `kv_bits`. A value not of that form is ignored with a warning. |
| `GMLX_DECODE_KV_RESERVE_GB` | Replace the estimated KV room with a flat reserve in GiB. The fallback is `8` when the KV size cannot be computed from the header. |
| `GMLX_PREFILL_NOCACHE=0` | Route prefill ring reads through the page cache. By default the ring bypasses it, as [The lossless settings](streaming.md#the-lossless-settings) explains. |
| `GMLX_PREFILL_RING_SLOTS` | The prefill ring holds this many layer slots. Each extra slot stages one more layer ahead and takes one layer's expert bytes from the arena. The default is `2`. |
| `GMLX_ARENA_STAGE_MAX_TOKENS` | The arena serves expert calls of up to this many tokens by reading only their routed experts, instead of by whole-layer staging. The default is `64`. |
| `GMLX_ARENA_SPLIT_MAX_TOKENS` | The arena splits expert calls of up to this many tokens when their routed set exceeds the arena. The default is `256`, and `0` disables splitting. |
| `GMLX_DECODE_PRESSURE=0` | Keep the arena at its sized capacity under memory pressure. By default it shrinks, keeping its most routed experts, then regrows when pressure clears. |
| `GMLX_DECODE_RAM_FLOOR_GB` | Arena sizing keeps this many GiB free for the rest of the machine. The default is 5% of RAM, and at least `4`. |
| `GMLX_DECODE_PAGECACHE_GB` | Arena sizing adds this page-cache reserve in GiB to the host floor. The default is `2.5`. |
| `GMLX_PIN_WEIGHTS=0` | Do not lock the every-token weights of a streamed model in memory. Pinning is on by default. |
| `GMLX_GPU_RESIDENT=0` | Skip wiring the every-token weights and the decode arena into the Metal residency set on streamed models. |
| `GMLX_STREAM_UNMAP_STACKS=0` | Keep the expert stacks' Metal buffers after the prefill and decode feeders take a layer. By default the buffers are dropped at that point. |
| `GMLX_STREAM_PREFILL_TAIL_MERGE=0` | Keep the streamed prefill chunk exact. By default a tail under an eighth of the chunk folds into the chunks before it. |
| `GMLX_STREAM_PLE=0` | Disable the streamable lookup-table tier. `1` forces the tables to stream even when the model fits, and `--stream-cpu` forces them too. |
| `GMLX_TABLE_MAX_BUFFER` | A lookup table may hold this many bytes in one GPU buffer. The default is the device limit. A table past it is read from the GGUF row by row. |
| `GMLX_TABLE_PREAD_WORKERS` | Each file-backed lookup table uses this many reader threads. The default is `32`. |
| `GMLX_GPU_KEEPWARM=0` | Disable GPU keep-warm, which is on by default for streamed models that use the decode feeder. |
| `GMLX_KEEPWARM_IDLE_S` | The keep-warm heartbeat pauses after this many seconds without streamed decode. The default is `1`, and `0` runs continuously. |
| `GMLX_DECODE_LOOKAHEAD=0` | Disable lookahead expert prestage on the decode feeder. |
| `GMLX_DECODE_LOOKAHEAD_K` | The decode feeder considers this many ranked predictions on each call. The default is `6`. |
| `GMLX_DECODE_LOOKAHEAD_WORKERS` | The dedicated prestage read pool has this many threads. The default is `6`. |
| `GMLX_DECODE_LOOKAHEAD_NORM` | It selects the prediction input, `ratio` or `raw`. `raw` skips the norm-gain rescale. The default is `ratio`. |
| `GMLX_DECODE_LOOKAHEAD_MIN_P` | A prediction rank stops being submitted when its reliability falls under this floor. The default is `0.5`. |
| `GMLX_DECODE_LOOKAHEAD_CANCEL=0` | Let unrouted predictions read to completion instead of cancelling the unstarted ones. |
| `GMLX_DECODE_LOOKAHEAD_IOPOL=0` | Run the prestage read pool at default disk priority. By default it uses utility I/O priority on a disk that the fast-disk policy did not call fast. |
| `GMLX_GOVERNOR=0` | Disable the runtime memory governor. Its band, shed counters and floor show at `/v1/metrics`. |
| `GMLX_GOV_KERNEL_FLOOR_GB` | Under this many GB of reclaimable pages, the governor goes red and reclaims caches. The default is the lower of `4` and 10% of RAM, and `0` disables the floor. |
| `GMLX_GOV_RESERVE_GB` | The server leaves this much RAM to the kernel and other processes when it computes its memory limit. The default is the larger of `8` and 10% of RAM. |
| `GMLX_CACHE_LIMIT_GB` | It overrides `server.cache_limit_gb` for `serve`, in GiB. `0` disables caching, and a negative value, `off`, `none` or `unlimited` lifts the limit. |
| `GMLX_NATIVE_FP` | It sets the MXFP4 and NVFP4 layout. `wire` keeps the file bytes, `packed` repacks them, and `auto` picks wire when streaming or past 90% of the working set. |
| `GMLX_CASCADE_SDPA=0` | Disable the shared-prefix cascade decode route, which reads a shared prefix once per step for the whole batch. |
| `GMLX_CASCADE_MIN_P` | The cascade route handles shared prefixes of at least this many tokens. The default is `1024`. |
| `GMLX_SPARSE_ATTN=1` | Enable top-k sparse attention for deep decode. It is lossy and off by default. |
| `GMLX_SPARSE_K` | Sparse attention keeps this many tokens. The default is `2048`. |
| `GMLX_SPARSE_MIN_S` | Sparse attention begins at this depth in tokens. The default is `8192`. |
| `GMLX_DRAFT_HEAD` | With `q8` or `q4`, an MTP or DFlash 2 drafter drafts through a `q8_0` or `q4_0` copy of a float target head, and `f16` uses the head as is. The default is `q8`. |

## Commands

The command variables change the defaults of `run`, `chat`, `ps`,
`systemone`, `pull` and `validate`, and the directories where gmlx keeps
its files.

| Variable | Meaning |
|----------|---------|
| `GMLX_TOOL_PREFLIGHT=0` | Skip the fit check that `run` and `chat` make from the header before a load, and that `cannot fit:` refusals name. |
| `GMLX_NO_FAMILY_DEFAULTS` | Disable the family model-card sampling defaults on bare-path `run` and `chat`, as `--no-family-defaults` does. |
| `GMLX_API_KEY` | `ps` and `systemone` send this key when `--api-key` is not passed. The server reads its own key only from `server.api_key`. |
| `GMLX_PULL_RETRIES` | `pull` accepts this many consecutive failed attempts on one file. An attempt that moves bytes resets the count, and `0` fails at once. The default is `10`. |
| `GMLX_PULL_TIMEOUT` | `pull` uses this socket timeout in seconds, which also bounds one stalled read. The default is `60`. |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | `validate` and `pull` send this Hugging Face token, checking the two in that order. Without either, they use the token that `hf auth login` stored. |
| `XDG_CACHE_HOME` | It sets the root of the `gmlx/` cache directory, which [Where files are on disk](troubleshooting.md#where-files-are-on-disk) lists. The default is `~/.cache`. |
| `XDG_DATA_HOME` | It sets the root of the `gmlx/` data directory, which holds saved chat sessions and the assistant's memory. The default is `~/.local/share`. |
