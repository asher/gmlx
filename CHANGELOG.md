# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Streamed-expert decode probes the drive at load and takes a faster
  prefetch recipe when it has bandwidth to spare. With the wider arena
  below, GLM-5.3-Flash UD-Q4_K_XL decode goes 7.93 to 10.42 tok/s on an
  M5 Max with no configuration. `--stream-fast-disk`, the
  `stream_fast_disk` config key or `GMLX_DECODE_FAST_DISK` force the choice.
- `GMLX_DECODE_PRESTAGE_EVICT`, `GMLX_DECODE_SETTLE` and
  `GMLX_DECODE_DECAY_EVERY` select the individual prefetch policies.

### Fixed

- A model addressed by config id or alias now gets the same command-buffer
  lift as a typed `--stream-experts`, so streamed decode no longer runs 40%
  slower from the config than from the bare GGUF path (10.3 vs 7.2 tok/s on
  GLM-5.3-Flash UD-Q4_K_XL). `serve` was never affected.

### Changed

- The decode arena sizes to 0.7 of physical RAM rather than 0.6, which on a
  128 GB machine is 90.7 GB rather than 78.6 and is worth 12% of decode.
  The live reclaimable ceiling and the system floor are unchanged, and
  `GMLX_DECODE_ARENA_RAM_FRAC` still overrides.

## [0.4.7] - 2026-08-30

### Fixed

- MTP with `kv_bits` no longer runs an fp16 cache on architectures whose
  layers are cache lists (glm5_next): nested KV members now convert, as
  `/v1/models` and admission already reported, and pooled members pack
  at rest as well. The `[kv]` line counts every pool packed.
- Hybrid sliding-window models no longer drop `kv_bits` on the MTP path:
  windows stay fp16 and full-attention layers quantize.
- Serve output with `kv_bits` set no longer corrupts into repeating
  garbage on long prompts (issue #104): a host-side mlx 0.32.1 bug made
  dequantize read the wrong buffers for strided cache slices; gmlx now
  hands it contiguous operands (fixed upstream in mlx 0.32.2).
- `GMLX_QWEN_OWNED=0` with `--kv-bits` and MTP no longer crashes every
  request; the spec path declines quantization on the stock fallback.
- Concurrent MTP requests with `kv_bits` no longer fail at batch
  formation; batched MTP runs fp16 KV (rollback cannot trim packed KV)
  and single-stream keeps the quantized cache.
- `--kv-bits` with `--max-kv-size` is refused at start instead of
  crashing mid-generation with "RotatingKVCache Quantization NYI".
- `/v1/metrics` server snapshot reports `apc.enabled` true whenever the
  residency pool holds a live APC manager; it said false until a request
  had touched the entry (the proxy resolves per request context).

### Changed

- KV quantization resolves one per-stack policy everywhere (serve,
  `run`, `chat`, MTP): attention KV quantizes (minus the last layer of
  a deep stack); windows and recurrent state stay fp16; pooled caches
  pack at rest. One `[kv]` line and `/v1/models` `kv_quant` report the
  verdict; admission pricing, the deepseek-v4 pooled path, and APC
  warm merges all follow the same policy (fp16 under MTP).

## [0.4.6] - 2026-08-29

### Added

- Qwen3.8-Flash-Next: when only its 27-54 GB n-gram table pushes the model
  past the wired budget, that table now streams instead of the experts, so
  the experts run fully on GPU (Q4 build: 45.9 tok/s decode, bit-identical
  output). Models too big even post-table stream table and experts together
  (Q6 build: short-context decode 8.4 -> 12.6-13.4 tok/s, wired peak
  106 -> 54 GB). `GMLX_STREAM_PLE` overrides; `GMLX_STREAM_PLE_COMPOSE=0`
  keeps the table resident when the experts stream.
- `chat` shows a spinner from send to first token.
- STQ1_0 GGUFs (llama.cpp PR #22836, GGML type 43 -- unmerged upstream, the
  id may shift) load end-to-end. Headerscan carries a fallback table for
  type ids newer than the installed gguf-py, preflight accepts the codec,
  discovery tags `-STQ1_0` filenames, and remote and local classification
  agree on it. Requires mlx-kquant >= 0.4.4, the release carrying the
  stq1_0 kernels.
- STQ1_0 GGUFs (llama.cpp PR #22836, GGML type 43; unmerged, the id may
  shift) load end-to-end. Requires mlx-kquant >= 0.4.4.

### Changed

- `gmlx validate` and the manage verbs classify local GGUFs via headerscan
  instead of gguf-py's `GGUFReader`, so a type id newer than the installed
  gguf-py gets a verdict instead of "cannot read as GGUF". The refusal
  message now also names mxfp4/nvfp4.

### Fixed

- Streaming Qwen3.8-Flash-Next no longer pins the n-gram table or sizes
  the decode arena around it (Q6 build default decode 3.4 -> 8.5 tok/s
  with the table resident).
- Streaming both the table and the experts credits both against tracked
  memory (the second credit was silently dropped), so serve headroom no
  longer understates free memory in that placement.
- Lookahead prestage reaches fused MoE decode blocks again; it stays off
  on Qwen3.8-Flash-Next, where it measured slower.

## [0.4.5] - 2026-08-29

### Added

- MTP drafter seeding streams into prefill: the long first-token stall on
  deep prompts is gone (worst inter-token gap at 67k depth drops from
  ~850ms to ~150ms; decode there gains ~13%). GMLX_MTP_SEED_STREAM=0
  restores the deferred seeding pass.
- The ^T finish-thinking key now works on the MTP path in `run` and `chat`:
  the owned round loop commits the forced close sequence as fully-accepted
  verify rounds at the next round boundary, so an open thinking block wraps
  up and the answer starts without leaving speculative decoding. `gmlx run`
  MTP now routes to the owned engine by default like `chat`, serve, and
  bench (`GMLX_OWNED_ROUND=0` opts back to the stock round, where ^T still
  prints its notice).
- `--thinking-budget` now applies on the MTP path in `run` and `chat`,
  through the same forced-close rounds as ^T (the stock MTP round still
  drops it with a notice).
- The server honors `thinking_budget` on MTP-drafted models (request field
  or profile/model config key), instead of rejecting those requests.
  Enforced by the owned round loop for a request decoding alone; requests
  batched with others drop the budget with a log note. See the behavior
  matrix in docs/server-config.md. Non-MTP speculative models keep the
  stock rejection.
- Model ids on one GGUF that differ only in `adapter:` now share a single
  resident model: base and every adapter load once, requests to any of the
  ids batch together, and each row applies only its own adapter. Single-model
  `serve --adapter` also registers the bare base as `<id>-base`.
- GGUF LoRA adapters apply on MoE expert targets, under `--stream-experts`,
  and with MTP speculative decoding; the delta runs inside the mlx-kquant
  matmul ops when the build has the LoRA epilogue, so an adapter costs about
  1-2% of decode.
- Managed servers detect when the gmlx install changed on disk after they
  started (a pip upgrade, or a checkout switch under an editable install):
  the runfile records a source fingerprint at boot, `gmlx status` and
  harness connects (`chat`, `launch`) flag the stale server with a
  `gmlx restart` hint, and a lazy import that fails inside the serve load
  path reports the condition instead of a bare "No module named" 500.
- `gmlx chat --server`: `/model <id>` switches the served model mid
  conversation with the transcript kept (`/model` lists the served ids,
  tab completes them), so a base and its adapters can be compared in one
  session.

### Fixed

- Prompt caching works on qwen4exp (Qwen3.8-Flash-Next): every APC tier now
  restores the QSA indexer state, and repeated prompts produce identical output.
- Metal buffer leaks: glm5next decode no longer hits the 499k resource limit
  on long generations, and streamed serve releases arena/weight residency on
  feeder close and model eviction.
- Per-request `seed` works on the server again: the thinking-budget patch
  installed after the seed wrapper on the same seam and clobbered it, so
  request seeds were silently ignored.

## [0.4.4] - 2026-08-27

### Added

- `glm5next` (GLM-5.3-Flash 320B-A18B, llama.cpp PR 27754) loads: hybrid
  KDA linear attention + NoPE MLA with a pooled DSA sparse indexer
  (top-512 key pools at depth), sigmoid MoE with clamped SwiGLU, 4-stream
  sinkhorn hyper-connections, and the glm4 BPE pretokenizer with
  `ignore_merges`. Long prompts stream the absorbed MLA attention in
  online-softmax tiles and gather the sparse-selected pool union per query
  block instead of masking the full key set.
- glm5next MTP speculative decoding from the GGUF's native NextN block:
  the drafter's DSA layer rides the trunk caches, verify rollback trims
  the latent-KV/pool caches and replays the KDA recurrent state from the
  recorded pre-verify sink.
- glm5next VLM pairing (`--mmproj`): the GLM-OCR ViT + conv-downsample
  projector load onto a vendored tower; images preprocess with the
  align-28 canvas search (16..8000 token budget), soft tokens splice at
  the `<|image|>` placeholders, and text-only requests keep MTP.
- VLM loads (--mmproj) compose with every speculative-decode form the text
  path supports: companion drafters (DFlash2, qwen4exp-mtp) load against a
  multimodal target, and companion-only families autodetect their drafter.
- nemotron_h_moe (Nemotron-3.5-Lightning) MTP speculative decoding from the
  in-file NextN head (auto-enabled) or the llama.cpp `mtp-*.gguf` sidecar
  (`--draft-gguf`, autodetected next to the target). Greedy decoding stays
  token-identical to plain decode: the verify walk steps Mamba2, attention
  and the MoE router gate per position, prefill matches mlx-lm's last-token
  split, and dense verify matmuls take the bit-exact kquant route
  (`--stochastic-mtp` keeps the faster non-exact route).

### Changed

- qwen4exp prefill rewritten around the sparse boundary: split-regime QSA
  dispatch (causal prefix, gathered ragged span, block-sparse tail) plus a
  ragged-length branch so serve one-shot prompts skip the dense token mask.
- qwen4exp hyper-connection prefill epilogues run as single-pass kernels
  (norm, mix+inject, combine, inject GEMV); `GMLX_Q4_HC_PREFILL_KERN=0`
  restores the bit-exact eager path.
- qwen4exp defaults to an 8192-token prefill chunk when the block-sparse
  kernels are armed (`GMLX_Q4_PREFILL_STEP` overrides).

### Fixed

- qwen3.5/3.6/3.8 image turns degraded to repetitive text with an early
  stop: mlx-vlm 0.6.15 vendored its qwen3_5 gated_delta functions, so plain
  VLM loads ran the GGUF's tiled V heads through grouped K-to-V kernels.
  Plain loads now rebind the vendored module like the MTP paths do.
- qwen4exp VLM conversations could fail with a broadcast_shapes error when
  the cache grew after a trim landed mid-step: the QSA position buffer was
  sized against its untruncated width and fell behind the key stream.
- `gmlx chat`/`serve` on nemotron_h_moe failed with "exposes no token
  embedding the batched engine can reach": the embedding probe now reaches
  the `Model.backbone.embeddings` nesting.
- Nemotron-3.5-Lightning built its trunk with the NextN/MTP block as a
  53rd layer, silently degrading all output; the trunk now excludes NextN
  layers, and their tensors are stripped from the trunk remap.
- `mtp-*.gguf` sidecars (same arch and metadata as their base model) were
  discovered as servable models; they now classify as drafters.
- qwen4exp ran the whole residual stream in fp32: the router's fp32 scores
  promoted each layer's MoE output and every downstream elementwise chain.
  Prefill is ~35% faster and decode ~20% faster after the dtype returns to
  the activation width.
- qwen4exp gate+up expert concat could freeze constructor placeholder zeros
  when weights were installed after the module was built, corrupting MoE
  outputs on some load orders.
- Serve governor: a kernel-floor breach zeroed the Metal cache and kept
  re-triggering itself, pinning throughput low until restart. The throttle
  now clamps the cache to a small budget (`GMLX_GOV_THROTTLE_CACHE_GB`) and
  the floor default drops to 4 GB (min with 10% of RAM) for small machines.

## [0.4.3] - 2026-08-26

### Added

- `qwen4exp` (Qwen3.8-Flash-Next, llama.cpp PR 27742) loads: the qwen3.5
  hybrid plus hyper-connections, QSA sparse attention and PLE n-gram hash
  embeddings (the 320M-row table stays zero-copy, rows gathered per token).
  Fused GDN decode kernel with the sigmoid output gate; fused MoE gathers
  at the 640-wide experts.
- qwen4exp MTP speculative decoding from a companion drafter GGUF (arch
  `qwen4exp-mtp`, built from the HF `mtp.*` tensors; autodetected next to
  the target or passed with `--draft-gguf`). Full-prompt teacher-forced
  seeding; verify rollback rewinds the GDN, PLE and QSA cache state to the
  accepted prefix.
- qwen4exp VLM pairing: the Qwen3-VL mmproj (`qwen3vl_merger`) loads onto a
  vendored wrapper reusing the qwen3.5 vision tower; interleaved mrope
  position ids thread through the attention and the QSA indexer (block keys
  roped at their cached positions), text-only turns keep the fast rope path.

## [0.4.2] - 2026-08-25

### Added

- `gmlx serve model.gguf --thinking on|off|adaptive` and `--thinking-budget N`
  set the reasoning switch and thinking-token cap for a single positional
  model without a config file (the same `thinking:` / sampling
  `thinking_budget:` keys a config sets).
- `/v1/metrics`: live per-request rows, `concurrency` and `rates`
  sections, waiting depth and in-flight counts.
- `GET /health?ready=1`: keyless readiness; 503 + `Retry-After` with
  reason `pressure`, `queue`, or `busy`.
- `GET /metrics?format=prometheus`: Prometheus rendering of the snapshot.
- `POST /v1/cache/reset` takes `{"model": "<id>"}`; with no body it clears
  every resident model's cache.
- `POST /v1/estimate` (or `"dry_run": true` on chat completions): memory
  preflight for a prompt without loading or generating.
- `GET /v1/capacity/plan?width=W&depth=D`: fan-out policy from the
  capacity table, governor band, and free slots.
- `/v1/models`: `context_length` and `max_context_at_width_1`; `gmlx
  launch pi` writes them as pi's `contextWindow` / `maxTokens`.

### Fixed

- Model loads are gated on the serve ceiling and the kernel floor (a
  co-load could Metal-OOM); memory still being freed is waited for.
- A deferred load answers `503 model_load_deferred` + `Retry-After`, not
  a 500.
- Speculative decode retires finished rows into the prefix cache at any
  batch width.
- Speculative decode at width > 1 on sliding-window models crashed on the
  first mid-decode admission.
- `GMLX_QUEUE_DEPTH_CAP` now fires under the residency pool.

### Changed

- `POST /unload` of the preloaded primary succeeds; in-flight streams
  still 409.
- Prefix cache key salt no longer hashes the embedding matrix per request;
  older on-disk prefix caches miss once and are rewritten.
- Stale runfiles no longer cause a running server refusal, with cleanup support.

## [0.4.1] - 2026-08-24

### Fixed

- Requests using the OpenAI `developer` role failed with a 500 on models
  whose chat template predates the alias ("Unexpected message role.", #66);
  the server now rewrites `developer` to `system` before render for
  templates that do not handle it.
- A chat template that rejects a conversation (`raise_exception`: misplaced
  system message, malformed tool call) now answers a clean 400 carrying the
  template's own message instead of a 500 traceback.

## [0.4.0] - 2026-08-24

### Added

- `/thinking [on|off|adaptive|default]` in chat flips the model's reasoning
  switch mid-session, in both server and local modes.
- Server-mode chat now forwards `--thinking` and `--reasoning-effort` (they
  were silently dropped); the server also accepts plain
  `thinking: on|off|adaptive` request values and a top-level
  `reasoning_effort` field.
- The `thinking` control maps onto Kimi K2.x's bare `thinking` template
  variable, so `thinking: off` works for Kimi K2.7.
- The config thinking keys (`thinking_budget`, `thinking_start_token`,
  `thinking_end_token`) now apply to `run` and `chat`, not just `serve`,
  with matching `--thinking-start-token`/`--thinking-end-token` flags.
- Server and assistant chat render the model's chain-of-thought through the
  standard reasoning display (it used to collapse into a "thinking..."
  status line); `serve --assistant` streams it as `reasoning` deltas.
- The server now governs memory pressure at runtime instead of dying on it:
  under pressure it stops admitting, throttles allocation, shrinks prefill
  chunks and speculative width, evicts idle caches, and as a last resort
  retires or fails the largest request with a clear error
  (`GMLX_GOVERNOR=0` disables; band and shed counters at `/v1/metrics`).
- A request shed under memory pressure now gets a typed answer instead of a
  hung stream: a 500 with the numbers before first byte, or a terminal SSE
  event with `finish_reason: shed` mid-stream.
- Cancelling one request in a speculative batch now frees its memory right
  away instead of holding it until the whole batch finishes.
- The server derives a capacity table at model load (max context by batch
  width, single-buffer and buffer-count ceilings) and logs it; a model that
  cannot hold any context refuses at boot with the numbers instead of
  aborting later (`GMLX_OVERCOMMIT=1` overrides). Decode concurrency is
  bounded by the derived width; the table shows at `/v1/metrics`.
- Loading or swapping a model that cannot fit the measured free working
  set now fails with the numbers instead of taking down the server.
- Sibling requests that arrive together no longer each prefill the shared
  prefix cold: the server admits the first one, waits for its stores, and
  starts the rest warm (`GMLX_APC_FRESH_WAIT_MS`, `0` disables).
- Requests beyond the queue cap now get an immediate 503 with Retry-After
  instead of queueing toward the timeout (`GMLX_QUEUE_DEPTH_CAP`).
- A prompt that cannot fit in memory now gets a 400 with the numbers before
  the stream opens, instead of dying mid-stream (`GMLX_PREFLIGHT_MEM=0`
  disables).
- `seed` is now honored per request inside a batch. Before, only the first
  request's seed took effect and it colored every row.
- Short prompts on sliding-window models now cache their block-aligned
  prefix at retirement instead of nothing, so an immediate follow-up
  turn starts warm.
- Decode concurrency is now a control (`GMLX_DECODE_BATCH`, default 8).
  The server always decoded up to 32 requests together, which slows every
  stream past the width where total throughput stops growing.
- DFlash 2 drafters (Inco AI / z-lab) for Qwen3.8-27B and Muse-Glimmer-30B:
  `--draft-gguf <DFlash2>.gguf`, or `gmlx discover` pairs a drafter with the
  base model its header declares, across directories. A DFlash 2 pairing
  replaces a DFlash v1 sibling on Muse Glimmer. Exact-match acceptance, so
  greedy output stays token-identical; `--stochastic-mtp` applies as well.
- `--native-mtp` (run/chat/serve) and the per-model `native_mtp` config key
  draft with the GGUF's own MTP head when a companion drafter is configured;
  a configured companion otherwise wins over the head.

### Changed

- Muse Glimmer's DFlash drafter runs on gmlx's own DFlash base. Its block
  attention now applies the reference sliding-window mask over the
  drafter's committed positions (a full ring trims the oldest keys per block
  row); output is unchanged until the ring fills.

### Fixed

- Hybrid prompt-cache retirement predicted the next turn with reasoning
  echoed back and stored a dead chain on keep-mode templates (deepseek4);
  it now echoes content only unless `preserve_thinking` is set.
- Exact-tier prompt-cache hits under serve `kv_bits` crashed the batch
  update path; the warm merge now re-quantizes to the live KV policy.
- Same-stream snapshots of rotating layers (exact retirement, drafter
  sidecars) reordered the ring and shifted MoE logits on the resumed turn;
  they now copy it bitwise.
- The memory governor read green while the box ran out of free pages (MLX
  buffer cache counted as free); a 128 GB box froze under a long-context
  batch. It now goes red below a kernel reclaimable floor
  (`GMLX_GOV_KERNEL_FLOOR_GB`, default min(8, 10% of RAM)) and block stores
  stop at it.
- The governor ceiling was 95% of the Metal working set, leaving the kernel
  14 GB on a 128 GB box; it is now capped at RAM minus a reserve
  (`GMLX_GOV_RESERVE_GB`, default max(8, 10% of RAM)).
- The MLX buffer cache is always bounded now (4-12 GiB); MLX's default let
  it hold 27-48 GB of freed buffers on an 8B model.
- Shell completion offered nothing for `gmlx launch <harness> --model`;
  it now lists the config's model ids and aliases.
- Served and VLM-local models stopped thinking by default on mlx-vlm 0.6.15,
  which injects `enable_thinking=false` into every render where the kwarg is
  absent; absent now means the template's own default again.
- Text-only served models build mlx-vlm cache classes again, keeping prompt
  caching engaged (the gmlx.cache crash fix built mlx-lm classes the APC
  engine's checks ignore).
- serve: evicting a streamed MoE model left its weights resident, so every
  later load deferred with a negative free working set until restart.
- Plain text models (qwen3, llama) failed every request with a missing
  gmlx.cache import.
- The exit-segfault guards (process exit and engine-thread park) armed only
  on mlx 0.32.1 exactly; newer mlx wheels carry the same unfixed upstream bug
  and ran unguarded. The guards now arm on every mlx from 0.32.1 on.
- Sampled speculative verify and the server's unfiltered sampler computed
  logprob math at the activation dtype; float16 rows reached categorical
  unwidened. Both now widen to float32 first (greedy paths unchanged).
- MiniMax MSA indexer projections stay F32 under float16 activations; the
  bfloat16 exact-bit narrowing does not exist at float16.
- Non-streamed replies lost the leading indentation of their first content
  line: the reasoning splitters (ATEM/harmony and the think-tag path all
  served models use) stripped all content whitespace. Verbatim code answers
  came back misindented on line one. Content now keeps first-line indent
  and still drops leading blank lines and trailing whitespace.
- gemma-4 applied the final logit softcap at the activation dtype, rounding
  every logit by up to ~0.12 nats at bfloat16 and flipping near-tie top-1
  picks. The softcap now computes in float32 and emits float32 logits, like
  muse-glimmer. GMLX_G4_SOFTCAP_F32=0 restores the old behavior.
- `--stochastic-mtp` with a block drafter (Muse Glimmer DFlash) stashed one
  proposal row per block and misaligned the walk; DFlash 2 records every
  draft row, and a DFlash v1 drafter (independent rows) keeps exact-match
  acceptance with a log line instead.
- serve: two model ids over one GGUF that differ in drafter (a companion
  `draft_gguf` and a `native_mtp: true` sibling) both loaded whichever was
  registered last; each build now carries its own drafter.
- The owned Qwen3.5 forward dispatched its Metal kernels (fused MRoPE,
  the bf16 verify GEMVs, the fused GDN bodies) whatever the default
  device; stock MLX raises on the CPU device (`KQUANT_FORCE_CPU=1`). The
  pure-MLX routes now take over there.

## [0.3.2] - 2026-08-13

### Changed

- `init`, `sync-models`, and `pull` now pair a sibling drafter (gemma4
  assistant, DSpark, DFlash) into the model it serves as that model's
  `draft_gguf`, which turns speculative decoding on. `sync-models` also adds
  the key to a model the config already carries. Before, you added it by hand.
- Muse-Glimmer DFlash drafts the block its checkpoint was trained for, which
  is 16 tokens per round on the current one, instead of 2. On a machine with
  no NAX tile a verify step costs about the same for 16 rows as for 8, so a
  full block accepts more tokens for the same forward.

### Fixed

- `--draft-block-size N` could only lower the draft depth. Each drafter froze
  its trained depth to the depth it loaded with, so a deeper request did
  nothing at all, and said nothing. A drafter now carries its trained depth
  apart from the depth it drafts by default, a request up to that depth is
  honored, and a deeper one clamps and warns. `gmlx run`, `gmlx chat` and
  `gmlx serve` (`--draft-block-size`, `GMLX_DRAFT_BLOCK_SIZE`) all resolve the
  depth the same way now.

## [0.3.1] - 2026-08-12

### Added

- Moonshot Kimi-K2.5 / K2.7 support and tool parser.
- Kimi K2.x vision (`--mmproj`, projector `kimik25`). The MoonViT tower and
  the patch-merge projector remap onto mlx-vlm's `kimi_k25`. llama.cpp's
  converter writes the vision Q/K in the split 2-D RoPE layout, and the
  remap puts them back into MoonViT's interleaved layout.
- You can now use `--stream-experts` with `--mmproj` with the cli run/chat
  commands. Server support for vision + streaming still to come.
- The server accepts `stream: experts` on a VLM entry, so you can serve an
  over-RAM VLM. gmlx puts the placement on the text tower, and the vision
  tower stays resident. The server refuses `stream: cpu` on a VLM entry.
  Send only one request at a time to a streamed entry (see streaming.md).

### Changed

- M1 and M2 now run the model graph in float16 instead of bfloat16. Those
  GPUs have no native bfloat16 arithmetic, so the compiler expanded every
  bfloat16 operation into a costly software sequence.
- mlx-kquant floor raised to 0.3.12: IQ4 perf improvements
- DeepSeek-V4 decodes slightly faster: the compressor emit-path fp8 round-trip
  runs as one mlx-kquant dispatch instead of a 13-dispatch graph, on
  every layer that completes a pool window. Bit-identical;
  GMLX_EMIT_QAT_FUSED=0 restores the graph.

### Fixed

- deepseek2 / glm-dsa yarn scaling. llama.cpp writes
  `rope.scaling.yarn_log_multiplier` as `0.1 * mscale_all_dim`, and gmlx
  read it back as `mscale_all_dim` itself. Thus every yarn-scaled GGUF on
  those arches (DeepSeek-V3/R1, Kimi-K2.x, GLM-5.2) ran with an attention
  scale approximately 1.85x too flat. Decoding stayed fluent, but it
  ignored the context and degenerated into repetition. `beta_fast` and
  `beta_slow` now pass through as well.
- Streaming decode-feeder token-split corruption. When gmlx staged one
  split piece, it could overwrite arena slots that an earlier piece's lazy
  gather still referenced. This garbled short-prompt prefills on
  low-residency models. Each piece now executes before the next piece
  stages.
- `--dtype auto|bfloat16|float16` (env `GMLX_ACTIVATION_DTYPE`, which serve
  reads too) sets the activation dtype for the model graph.
- serve: `--dtype` and `server.dtype` set the same knob for every model the
  server loads. An unrecognized value is a config error rather than a silent
  fall back to the default.
- Prompt cache: hybrid-arch models keep an anchor checkpoint at the end
  of the system prompt, so parallel requests sharing a system prompt and
  tool schemas (subagent fan-out) start warm instead of re-prefilling
  the shared prefix after the conversation deepens. GMLX_APC_CKPT_SYS=0
  disables.
- Prompt cache: exact-tier models (deepseek-v4-class pooling stacks) get
  the same system-prompt anchor as a whole-prefix snapshot in its own
  LRU, so sibling fan-out stays warm there too. GMLX_APC_ANCHOR_ENTRIES
  and GMLX_APC_ANCHOR_BUDGET_MB bound it; GMLX_APC_CKPT_SYS=0 disables.

## [0.3.0] - 2026-08-09

### Added

- Muse Glimmer support (GGUF arch `muse-glimmer`, Meta Muse Glimmer 30B):
  a vendored text decoder with sandwich norms at two epsilons, an
  attention output gate, and NoPE on the full-attention layers with RoPE
  only on the 2048-window sliding ones. Vision rides the `muse-glimmer`
  mmproj through a vendored ViT and a GGUF-only image processor. The ATEM
  reasoning channel and its XML tool calls are wired through chat, serve,
  thinking budgets, and the muse profile family (reasoning strength
  low/medium/high/xhigh). `--draft-gguf` loads the DFlash drafter for
  speculative decoding.
- serve --speculative: a request arriving while one stream decodes with
  MTP no longer waits for it to finish; the stream converts to shared
  batch decode and speculation resumes once the batch drains back under
  the width cap. GMLX_MTP_PREEMPT / GMLX_MTP_RESUME disable each half.
- gmlx chat --server: chat against a running server as a plain client,
  without the assistant's tools and memory (no background requests).
  Engages automatically when the config's server is already up and
  serves the requested model; --local pins the in-process load. An
  explicit GGUF path always loads the file on disk.
- decode_prefill_ratio accepts "auto" and it is the new default: live
  streams keep at least half their decode rate while deep prompts admit,
  and pacing stands down wherever it would not help (simultaneous
  bursts, cheap chunks, stuck queues). A numeric ratio pins the previous
  static behavior; GMLX_DECODE_PREFILL_AUTO=0 reverts on a live server.
- GMLX_SERVE_MEMSTATS=path.jsonl writes a per-tick serve memory trace:
  MLX counters, free-headroom estimate, and per-owner cache byte
  attribution with allocation shapes marked on change, for diagnosing
  serve memory growth under load.
- Serve admission is gated on projected memory headroom: a request whose
  measured KV and prefill-transient projection does not fit is kept
  queued and retried each tick instead of committing memory the box does
  not have. Requests are never failed by the gate, an idle server always
  admits, and a request deferred past GMLX_ADMIT_DEFER_MAX_S (default
  60s) is admitted anyway with a loud log. GMLX_ADMIT_HEADROOM=0
  disables.
- /v1/metrics reports residency budget vs resident bytes, live
  active/cache/headroom memory, and admission deferral counters.

### Changed

- mlx-kquant floor raised to 0.3.11: MoE prefill gather runs 12-28%
  faster per call at chat-chunk widths, lifting serve prefill 20-43%
  shallow and 8-14% deep on many-expert models.
- Bench chart value axes clamp to the data range when a zero anchor
  would waste the panel height on empty space; nearby engine lines
  now read as visually distinct.
- benchmarks.md tracks builds and measured date per model, and merged
  results carry the newest contributing run date: one model rebenched
  on newer releases no longer implies the rest was remeasured.
- DeepSeek-V4-Flash IQ2_XXS rebenched on gmlx 0.2.2 + mlx-kquant
  0.3.11 vs ds4-server b030961 (2026-08-05): prefill 1.11-1.86x and
  decode 1.05-1.59x across the full d512-500k ladder.
- DeepSeek-V4 single-token decode runs its hyper-connection glue as four
  native mlx-kquant ops instead of about 176 python kernel launches per
  step. `GMLX_HC_M1_FUSED=0` and `GMLX_HC_KQ=0` restore the previous
  routes for A/Bs.
- DeepSeek-V4 hyper-connections at speculative verify widths run as one
  fused kernel per call.
- The hc_expand between attention and the ffn hyper-connection fuses into
  the following collapse kernel, and dense small-N projections (router
  gate, indexer weights, hyper head) route through mlx-kquant's
  skinny_matmul at speculative verify widths when available
  (GMLX_KQ_SKINNY=0 opts out).
- The server now sizes command buffers per phase, coarse while decoding
  and fine during deep prefill, rather than one cap for both.
  `GMLX_CB_PHASE=0` restores the single cap.

### Fixed

- serve --speculative no longer serializes concurrent requests: MTP
  loads counted the model weights twice in the admission headroom
  estimate, holding it negative for the server's lifetime.
- Chat's bottom toolbar no longer clips the live tok/s readout on
  narrow terminals; sampling knobs are dropped first instead.
- The serve free-headroom estimate went negative on models whose load
  materializes weights into MLX-tracked memory (the same bytes counted
  twice); the loader now registers only the truly untracked mmap
  remainder, measured against the load's active-memory delta.
- DeepSeek V4 serves concurrent requests: multi-row prompt batches on
  pooling-cache models failed before prefill, and admission re-merged
  already-batched caches, killing every request in flight above c=1.
- GGUFs that quantize the MoE router gate (some community DeepSeek quants;
  llama.cpp's own quantize leaves it F32) now load: small quantized tensors
  on raw-array modules are dequantized to f32 at load instead of erroring.
- MTP prefill steps now use fine prefill caps to avoid a transient memory
  spike that could trigger a Metal OOM.
- Prompts short enough to skip chunked prefill also use fine prefill caps;
  they previously ran under decode caps, and a burst of 1-2k-token requests
  could crash the server with a Metal OOM.

## [0.2.2] - 2026-08-06

### Fixed

- Fixed a DeepSeek-V4 indexer crash on non-NAX (pre-M5) GPUs once the
  context passed ~8k tokens.
- The large-model memory warning now prints once per chat session, not
  every turn.

## [0.2.1] - 2026-08-05

### Added

- `--moe-prestage keepers`: lookahead reads skip experts the miss-shed
  policy would drop and stage predicted keepers demand-grade, turning
  their synchronous demand stalls into reads that overlap compute. Also
  a per-model server config key (`moe_prestage: keepers`) and serve flag.
- Ctrl-T during a `chat` or `run` reply closes the model's open thinking
  block early (as if the thinking budget had just run out) so the answer
  starts now.
- DeepSeek-V4 speculative decoding now accepts llama.cpp dflash draft GGUFs
  in addition to DwarfStar's.

### Fixed

- Server: two model ids over the same GGUF now share one resident copy when
  they differ only in spelling a default explicitly (`moe_prestage: ranked`,
  `prefill_feeder: true`, or any streaming lever on a model with no `stream`).
- `--over-generation` runs now render thinking in the verbose stream like
  normal runs (the probe path bypassed the styled emitter).
- CLI polish: `--help`/`--help-all` now win even after a value-taking flag,
  the short `-h` pages gained the typical sampling/config/streaming flags,
  tab completion offers flag values (choices, themes, profiles).
- Chat: cancelling or stopping mid-round no longer leaves rejected draft KV 
  in the persistent chat cache.

## [0.2.0] - 2026-08-02

### Added

- Kimi-K3 support (GGUF arch `kimi-k3`, llama.cpp PR #26185 / unsloth
  conversions): vendored KDA + nope-only MLA + latent-MoE model, kimi-k2
  pre-tokenizer, config synthesis from the GGUF header, and XTML thinking
  wired through chat, serve, thinking budgets, and the kimi profile family
  (thinking effort low/high/max).
- Larger-than-RAM MoE decode: expert streaming composes with an
  every-token-weight RAM pin, a GPU-resident Metal residency set for
  weights and arena (`GMLX_GPU_RESIDENT`), a wired-budget handoff between
  the prefill ring and the decode arena, wide layer slots for >2 GiB
  expert stacks, and an MLX command-buffer cap lift at CLI entry.
  Kimi-K3 UD-IQ2_XXS (662 GB) decodes at ~1.4 tok/s on a 128 GB box with
  bit-identical output.
- `GMLX_LOG_ROUTED`: routed-expert-id trace from the decode feeder, for
  offline arena replay and sizing experiments.
- Speculative decoding for the GA DeepSeek-V4-Flash 0731 checkpoint via its
  DSpark draft model (three chained draft blocks replacing the legacy
  single-block MTP head). The drafter loads from a sidecar GGUF discovered
  next to the target (preferred over legacy `deepseek4_mtp_support`
  sidecars; `--draft-gguf` selects one explicitly) - measured +20% decode
  over plain on UD-Q2_K_XL at ~0.99 acceptance. `GMLX_DSPARK_CONF` moves
  the confidence gate (default 0.9, the model's own calibration) and
  `GMLX_DSPARK_ROWS` the draft rows per block (default 3, the measured
  optimum: fewer noise rows sharpen the non-causal block and keep verify
  widths on the compiled attention paths).
- `scripts/convert_dspark_sidecar.py` builds that sidecar from the three
  draft shards of the upstream release (~10 GiB read instead of the full
  167 GB checkpoint): expert tensors repack bit-exact into GGML MXFP4,
  dense tensors encode Q8_0. `--experts-codec q3_k` (or `q4_k`/`q2_k`)
  trades a little acceptance for a smaller file, with a round-trip RMS
  report against the upstream weights.
- Draft-model sidecars whose codecs are all fp4-wire capable load in wire
  mode by default: the DSpark MXFP4 experts stay zero-copy and file-backed
  instead of pinning ~9.6 GiB of anonymous memory (an explicit
  `GMLX_NATIVE_FP` setting still wins).
- Shared-prefix cascade decode for concurrent serving: streams that share a
  prefix (same system prompt, warm history, cold or cached alike) read it
  once per step for the whole batch instead of once per stream - measured
  1.4x aggregate decode at four streams on a 12k system prompt. Exact and
  on by default; `GMLX_CASCADE_SDPA=0` disables. Speculative-verify rounds
  cascade too: two streams on a 13k prompt with the gemma assistant
  drafter decode 1.37x faster aggregate, and native-MTP models (qwen)
  get the same verify-round win. Composes with quantized KV
  (`kv_bits: 8`).
- Opt-in sparse attention for deep contexts: `GMLX_SPARSE_ATTN=1` decodes
  past `GMLX_SPARSE_MIN_S` (default 8192) tokens attending only the
  top-scoring KV pages within a `GMLX_SPARSE_K` (default 2048) token budget.
  Attention cost stops growing with depth - 1.40x end-to-end decode at 32k,
  quality loss on the order of Q6 quantization noise. Engages only on
  quality-gated architectures (llama-family full attention); measured SWA
  hybrids (gemma-4) are excluded and always run full attention.
- Sliding-window models (gemma-4) join the hybrid prompt-cache checkpoint
  tier, and prefill now checkpoints at intervals (`GMLX_APC_CKPT_INTERVAL`,
  default 4096 tokens): a burst of long shared-prefix requests converts to
  one cold prefill plus warm tails instead of N cold prefills.
- The checkpoint tier serves the plain (non-speculative) path too: lookup,
  interval checkpoints, and retirement no longer require an MTP drafter.
- On hybrid (GDN and sliding-window) models, generated tokens survive
  retirement even when the next turn's re-rendered history diverges
  (thinking strip, tool-call re-serialization): decode-time snapshots
  retain the reply up to the divergence point (`GMLX_APC_DECODE_CKPT`,
  default 512 generated tokens; `0` off).
- `run`/`chat --thinking on|off|adaptive` and `--reasoning-effort LEVEL`:
  first-class switches for thinking models. The chat template picks the
  spelling: MiniMax's three-state `thinking_mode`
  (enabled/disabled/adaptive) where present, `enable_thinking` next
  (Qwen3.x, GLM, DeepSeek-V4 alias), the Hy3 `reasoning_effort: no_think`
  dialect where neither exists, and a pointer at the effort levels on
  gpt-oss (which cannot disable reasoning). `--reasoning-effort` passes
  through verbatim - level names are the model's own (gpt-oss/GLM
  low|medium|high, Hy3 no_think|low|high, DeepSeek-V4 max) - and warns
  when the template has no such variable.
- The z.ai / GLM API spelling of the thinking switch -
  `thinking: {"type": "enabled"|"disabled"}` - now works everywhere the
  template kwargs do: `--chat-template-config`, config/profile
  `chat_template_kwargs`, and as a top-level request field on the server
  (the request field maps onto the serving model's own thinking switch).
- Config profiles and model overrides accept `thinking:` and
  `reasoning_effort:` keys, applied per model in whatever spelling its
  chat template reads; explicit `chat_template_kwargs` stay verbatim.

### Fixed

- `--moe-miss-shed` no longer crashes chat prefill on kimi-k3 (a
  concatenate rank mismatch): shed is decode-only and now skips the
  single-token leaves of an arena token split.
- `gmlx chat` no longer rejects `--stream-experts` on an MTP/speculative
  base.
- Chat markdown rendering no longer reverts a block to raw text when it
  grows taller than the terminal: the rendered top scrolls into scrollback
  and only the last screenful stays live-repainted, so long fenced code
  blocks keep their formatting end to end.
- The decode-arena reclaimable-RAM estimate now counts the whole
  file-backed page cache (droppable without swap), not just the inactive
  queue - a machine full of hot GGUF cache no longer clamps
  `GMLX_DECODE_ARENA_GB` as if it were out of RAM.
- Sampling embedded in the GGUF header (`general.sampling.*`) is now
  honored: it refines the arch-family defaults for both configured models
  and bare-path runs (profiles, overrides, and explicit flags still win).
- `gmlx init` / `sync-models` no longer scaffold `speculative: true` for
  models whose architecture has no MTP target class (they failed at load),
  and models too large for RAM get `stream: experts` (MoE) or a `stream:
  cpu` hint (dense) instead of an entry that cannot load.
- Kimi-K3 supports the router-side MoE levers: --moe-expert-probe,
  --moe-expert-mass and --moe-experts previously printed "no supported
  offloaded MoE block" and silently did nothing (the block's inline
  plain-Linear router was invisible to the installers).
- Kimi-K3 now thinks by default in chat and serve: templates that gate
  thinking on an XTML channel (<|open|>think<|sep|>) were missed by
  mlx-lm's vocab-pair detection, which then forced enable_thinking=False
  and the model answered without reasoning.
- Freed decode-arena and prefill-ring buffers are returned to the OS
  instead of idling in MLX's freed-buffer cache, where the kernel
  compressed them and ran the box to the free-page floor mid-prefill.
- Auto arena sizing credits the prefill ring's bytes now that the two
  time-share the wired budget: a large pinned model no longer sizes its
  decode arena to zero and decodes on the page-cache path with most of
  RAM wired (measured driving the box to the free-page floor).
- Long streamed-expert follow-up turns no longer breach the wired cap:
  rebuilding the prefill ring after decode now borrows the ring's footprint
  from the decode arena (hot experts kept), and the next decode returns it.
  Previously the ring re-allocated on top of the fully wired arena.
- Streamed-expert chat turns no longer stall at the decode-to-prefill
  transition: an expert call routing more distinct experts than the decode
  arena holds is now token-split and served from the arena's read pool
  instead of a CPU page-cache gather (measured 8.5x on a second-turn
  prefill; GMLX_ARENA_SPLIT_MAX_TOKENS caps, 0 disables).
- Non-stream replies that hit `max_tokens` inside a think block now return
  the partial reasoning as `reasoning_content` with empty content, matching
  the streaming path (the raw reasoning previously leaked into `content`).
- `preserve_thinking` now works on plain chat turns: the server keeps
  `reasoning_content` on assistant history messages instead of dropping it
  before the template renders (`GMLX_FAITHFUL_HISTORY=0` restores stock).
- Multi-turn prefix-cache entries from finished server requests now match
  the follow-up turn on thinking and tool-calling models (entries are keyed
  on what the chat template will actually replay; `GMLX_APC_RETIRE_LCP=0`
  restores the old keys).
- Checkpoint-tier memory and disk are bounded: record payload rides a byte
  budget (`GMLX_APC_CKPT_BUDGET_MB`, default 4096), drafter sidecars too
  (`GMLX_SPEC_APC_SIDECAR_BUDGET_MB`, default 512), and interval
  checkpoints on GDN models no longer write recurrent state to disk.
- A finished request's retirement-key memo no longer pins the response
  generator (and its model weights) past a pool unload.
- Quantized-KV (`kv_bits`) prefill no longer runs 1.6-1.9x slower than
  fp16: prefill-width attention now takes the fused flash path
  (`GMLX_KV8_PREFILL_FLASH=0` restores the old behavior).

### Changed

- The DSpark draft-model module (`deepseek_v4_dspark.py`) and the sidecar
  converter are MIT licensed (LICENSE-MIT, SPDX headers); the rest of gmlx
  stays BUSL-1.1.
- The mlx-lm-style model modules (kimi-k3, minimax-m3, hy-v3) and the
  kimi-k3 tests are MIT licensed (LICENSE-MIT, SPDX headers); the rest of
  gmlx stays BUSL-1.1.
- The mlx-kquant floor is 0.3.9 (kimi-k3 zero-copy load and residency ops).
- Informational `[stream]` banners (streaming summary, feeder/arena sizes,
  keep-warm, lookahead, MoE lever confirmations, runtime stats) and the
  `[prefill]` chunk-size note only print under `--verbose`; warnings
  (ignored flags, fallbacks, wedged reads, clamps) stay visible.
- Scaffolded config comments now sit on their own line above live keys
  instead of trailing them (no more wrapped lines on narrow terminals); the
  redundant mmproj `# VLM companion` comment is gone.
- Streamed installs default the every-token-weight residency set and the
  keep-warm heartbeat on; GMLX_GPU_RESIDENT=0 / GMLX_GPU_KEEPWARM=0 disable.
- glm-dsa/DeepSeek-V3.2 sparse decode uses the stock top-k gather again
  (O(index_topk) per step); the mask-path workaround is now opt-in via
  `GMLX_DSV32_MASK_DECODE=1`.
- The server's APC prompt-cache manager is now gmlx-owned, built at model
  load; config `cache.enabled` and a plain `APC_ENABLED=1` both keep working.
- gemma-4 batched decode runs the global (hd512) layers as one ragged
  kernel call per layer instead of a per-row loop (needs mlx-kquant with
  `sdpa_decode_gqa` starts; older wheels keep the loop).
- gemma4 text MTP targets now build gmlx-owned mask/attention classes at
  construction (`GMLX_GEMMA_OWNED=0` reverts to the stock classes plus the
  patch regime; numerics identical either way).
- Quantized-KV (`kv_bits: 8`) decode routes through the fused mlx_kquant
  kernel when available, ~1.4x per attention call at depth
  (`GMLX_QSDPA_KQ=0` restores the stock path).

## [0.1.4] - 2026-07-27

### Added

- `gmlx run --reasoning {show,hide,raw}`: `run` now streams a thinking
  model through the same styled reasoning display as chat (default `show`),
  on the plain, MTP-speculative, and VLM text paths alike.
- Size and RAM-fit verdicts: `validate` and `pull` report a model's total
  size (all shards) and whether it fits this Mac's RAM - comfortable, tight
  (KV-cache headroom), or over (with the `--stream-experts` pointer). Repo
  listings gain per-variant size and fit columns; a tight/over pull prints
  an advisory note but still downloads. `gmlx doctor` gains a `memory` row
  that warns on configured larger-than-RAM models not set to stream.
- `--help-all` on `run` and `chat`: the default `--help` now shows the
  everyday flags with a pointer at the rest; `--help-all` prints the full
  reference. Shell completion still offers every flag.
- A glossary in getting-started.md (GGUF, quant, KV cache, prefill/decode,
  MoE, MTP, ...), a pointer to it from the README opening, and a row for it
  in the docs index.

### Fixed

- A request admitted into a running speculative decode batch was truncated
  at its batchmates' `max_tokens` instead of its own.
- left-padded single-row batches attended their pad tokens as content:
  the stock B=1 batch-cache shortcut extracts a row cache that drops
  `left_padding` while recursing with the unsliced input. The owned
  forward takes the direct masked path instead and matches an unpadded
  reference exactly; the stock defect is pinned by an anti-identity
  test so an upstream fix surfaces.
- Extras now install correctly when gmlx itself was installed with `uv tool`
  or pipx. Those environments are owned by their installer - uv's has no pip
  at all - so `gmlx init`'s "install it now?" failed, and every "not
  installed" message named a `pip install` command that could not work.
  `gmlx.extras` detects the route and issues `uv tool install --force`
  (merging the extras and interpreter already recorded, so nothing is
  dropped), `pipx inject`, or `pip install` as appropriate.
- `gmlx[tts]` on its own could not synthesize: the vendored misaki G2P
  imported torch at module scope, and torch is not a dependency of any gmlx
  extra - it only ever arrived through `gmlx[stt]`'s mlx-whisper. The import
  now happens inside the one class that uses it, which Kokoro never
  constructs, so the voice stack no longer pulls torch at all.
- First speech synthesis no longer shells out to pip. The vendored G2P called
  `spacy.cli.download` for its English pipeline; with no pip in the
  environment spacy answered by exiting the process with no diagnostic. The
  pipeline is fetched from its Hugging Face mirror at a pinned revision
  instead, and an already-installed pipeline still takes precedence.
- The `tts`, `talk`, and `all` extras work on Python 3.14. spacy carried a
  `python_version < '3.14'` marker from when it had no cp314 wheels; it has
  shipped them since, and the stale marker had turned into a silent failure -
  spacy was dropped from the install while the G2P still imported it.
- `gmlx doctor`'s extras check now detects a voice stack that is missing
  spacy, which previously reported as fully installed.

### Changed

- qwen3.5/3.6 text MTP targets run owned gmlx forwards instead of
  runtime patches on the stock mlx-vlm classes, parity-tested against
  the pinned release (greedy tokens identical on every route).
  Multimodal MTP targets are still built stock and keep the patched
  regime; `GMLX_QWEN_OWNED=0` reverts text loads to genuinely stock
  classes for debugging.
- Install docs lead with `uv tool install "gmlx[all]"` / pipx: one command,
  on PATH in every terminal, every optional feature on, no follow-up
  installs. Smaller extra sets and the plain-venv route stay documented.
- `mlx-kquant>=0.3.7`, the first release with a cp314 wheel. Below it a
  Python 3.14 install - what uv and pipx select by default - silently
  compiled the Metal kernels from source.
- Python 3.14 is a supported release: the CI matrix runs it alongside
  3.11-3.13, and it carries the trove classifier. It is what the documented
  `uv tool` and pipx installs select, so it was already the default target.
- The quiet-load `[load]` summary now reads family | size | dominant quant
  | elapsed; the full codec histogram moved to `--verbose`.
- Bare `gmlx serve` with no config says so in the foreground before
  detaching; the ready banner and `gmlx status` count the served models and
  call out a server serving 0 models (with the init/pull/restart recipe)
  instead of reporting a bare "healthy".
- `gmlx chat` with no model now explains the four ways to get one instead
  of dumping the usage block.

## [0.1.2] - 2026-07-26

### Added

- Prefill ticks: while streams are decoding, each admission prefill chunk is
  halved until its predicted wall time fits a stall budget
  (`server.prefill_tick_ms` / `--prefill-tick-ms` / `GMLX_PREFILL_TICK_MS`,
  default 500 ms; 0 = full chunks), bounding the per-chunk decode hitch that
  pacing's average-share arithmetic cannot. Inert with no live decode, so
  single-stream TTFT is unchanged. Measured (27B dense, 14k context, four
  streams): worst inter-token gap 3.6 s -> 109 ms, per-stream decode +80%
  at the contended cell, aggregate throughput and single-stream rates
  unchanged.
- batched decode fuses the attention and MLP projections at load on
  llama and qwen3.5/3.6 text (full-attention layers; GDN layers
  untouched): q/k/v (and gate/up) K-quant wires row-concatenate into
  one matmul per group on the single-position B>=2 path, filling the
  GPU where the separate small-M launches underfill it (llama-8B k/v
  put up 16 threadgroups each). Certified bit-exact against stock on
  Dolphin3-8B and Qwen3.5-9B Q6_K (B=2 greedy logits and tokens
  identical, ragged BatchGenerator tokens identical); B=1 takes the
  stock path exactly (`GMLX_OCCUPANCY_FUSE=0` reverts, read per call).
- at batch width 12 and above the fused MLP also splits down_proj into
  two half-K matmuls plus an add: the single long-K launch craters at
  the M=12 kernel-route cliff (q6_k [4096x14336] serial 224 -> 124
  GB/s) while two overlapping halves hold 165. The add costs one bf16
  rounding, so this path is allclose-not-bit-identical to stock
  (certified: B=12 greedy tokens identical, logits within 1 ulp);
  single-stream traffic never reaches it. Combined step-time win at
  B=16 d64: -11.0% on Dolphin3-8B, -7.1% on Qwen3.5-9B
  (`GMLX_SPLITK_MIN_B` tunes the width, `0` kills the split).
- gemma-4 concurrent decode and speculative verify keep the global layers
  on fused attention: head_dim-512 batched calls at decode width (one
  position) and MTP verify width (2-8 positions) route each stream through
  the single-stream kernels (left padding via per-row K/V tail slices;
  verify blocks via end-aligned causal on the slice) instead of the stock
  materialized fallback, on both load paths (text-only GGUFs via the
  mlx-lm text module, multimodal via the mlx-vlm language module).
  Certified in-process on gemma-4-31b Q6_K: whole-step -8.6% at four
  streams on mixed 8k-14k contexts (tail slices also skip the padded
  prefix short rows would otherwise re-read); -3.4% at two streams at
  uniform 14k; speculative verify -17.5% per verify call at two streams
  with a 3-token block at 10-14k depth (`GMLX_G4_BATCHED_SDPA=0`
  reverts).

### Fixed

- worked around an upstream Metal kernel bug in `mx.fast.rope` (stock
  mlx 0.31.2): with a scalar offset and a single-position batched
  input (B, \*, 1, D) with B > 1, every batch row past the first is
  rotated from out-of-bounds memory (allocator-dependent garbage; the
  CPU path is correct). Scalar means fewer offset entries than batch
  rows: a plain int, a 0-d array (the `mx.array(cache.offset)` wrap in
  mlx-lm's gemma4_text produces one), or a size-1 array. Batched
  serving was never affected -- BatchKVCache passes size-B per-row
  offset arrays, which are correct -- but any plain-KVCache batched
  decode (raw chains, external harnesses) silently corrupted rows past
  the first. The fix wraps `mx.fast.rope` and expands any offset
  carrying fewer entries than the batch dimension to a per-row int32
  array, covering every arch including direct `mx.fast.rope` call
  sites (`GMLX_ROPE_BATCH_FIX=0` reverts; per-variant subprocess
  tripwire tests flag when an mlx upgrade fixes the kernel so the
  workaround can be dropped -- subprocesses because an in-process A/B
  is primed by the fix's own expanded-offset buffer and reads clean).
- the qwen3.5/3.6 batched-verify SDPA seam no longer forwards to the
  stock per-pad-group gather loop whenever the cache carries a
  left-padding attribute: it now bails only on real padding, and when
  padding is real it answers with one SDPA call under a combined
  left-pad + causal boolean mask (bit-exact against the stock loop on
  the bf16 serve path; `GMLX_VERIFY_RAGGED_MASK=0` restores the
  forward). Live verify traffic reaches this seam with the pad
  attribute already cleared and an upstream-built array mask, so this
  hardens the seam against flows that pass pads at verify width rather
  than changing current serve numbers; a one-shot
  `[verify] ragged-pads route:` stderr note reports whenever either
  branch actually fires.
- quantized KV cache (`kv_bits`) no longer crashes concurrent serving on
  grouped-query models: upstream's quantized SDPA applies the batched
  left-pad mask to 5D grouped scores and every B>1 masked call raised a
  broadcast error (single-stream was unaffected). gmlx inserts the missing
  mask axis at both upstream seams (`GMLX_QSDPA_MASK_FIX=0` reverts).

### Changed

- mlx-kquant floor raised to 0.3.6, which ships the M-banded NAX qmm
  routing (BM=32/64-db/128 tiles behind a measured per-codec policy).
  Older wheels still run correctly, but the batched-decode rates
  certified on this release, and the batch-width floor of the down_proj
  split, are calibrated against 0.3.6's routes.
- Batched decode on gated-delta hybrids (qwen3.6 dense) speeds up ~8%: the two
  tiny per-layer decay/gate projections (b/a) are concatenated at load and
  routed through the M-stationary head kernel at batch width 2-8, instead of
  falling onto a full GEMM tile per matvec (~70 us wall each for ~1 MB of
  weights). Token-identical; single-stream decode and prefill unaffected.
  `GMLX_GDN_BA_CAT=0` restores the separate matvecs.
- multimodal gemma-4 decode/verify no longer host-syncs per step: the
  sliding-mask cache probe compares int offsets host-side (array offsets skip
  the probe), and per-layer int rope offsets pass through without a device
  wrap. Array rope offsets keep upstream's snapshot copy: the cache update
  between the key rope and the query rope advances its offset array with an
  in-place `+=` (mx arrays mutate through every handle under augmented
  assignment), so an aliased offset rotates queries one position ahead of
  keys on every batched decode step. A serve-level gate certification caught
  width-cap-gated B>=3 gemma decode degenerating into repetition loops from
  exactly that skew (server-side throughput looked healthy; only the content
  was wrong). Token outputs are bit-identical on the int paths;
  `GMLX_G4_NOSYNC=0` restores upstream behavior. Text-only gemma-4 loads build from the mlx-lm text module, which
  has no per-step host syncs to begin with.
- gemma-4 MTP targets no longer install the three qwen3_5 verify levers
  (module-scoped no-ops on gemma4); the tied quantized head already serves
  verify logits directly.

## [0.1.1] - 2026-07-24

### Added

- Decode-priority prefill pacing: live decode batches keep a share of GPU time
  while admission prefills run (`server.decode_prefill_ratio` /
  `--decode-prefill-ratio`, default 1.0; 0 = stock). Concurrent decode at depth
  no longer stalls behind other requests' prefills: c2 per-stream +77% at 14k,
  ~5x at 50k, c1 unaffected.
- Speculative decoding backs off under concurrency: MTP runs while the live
  decode batch is at or under a per-model width cap, wider batches finish in
  plain decode with the drafter loaded (`models[].speculative_width_cap` /
  `--speculative-width-cap`; per-family default, 0 = uncapped). Mixture-of-experts
  targets default to a single stream and are detected from the loaded model, so
  the default reaches new MoE architectures unaided.
- Unified ragged-plan batched decode for the qwen3_5 family: concurrent streams
  at different depths stay on the fused ragged-attention kernel instead of a
  per-row fallback loop.
- `detect_arch` and `load_tokenizer_from_gguf` promoted to the stable public
  API (`gmlx.__all__`): synthesize the HF tokenizer from GGUF metadata without
  loading the model.
- Lossy MoE decode levers for streamed models: `--moe-miss-shed` drops
  non-resident experts while keeping a configured share of routing mass, and
  `--moe-layer-shed` probabilistically skips routed MoE paths per token.
- GPU keep-warm for streamed decode: `--gpu-keepwarm` (config
  `server.gpu_keepwarm`) holds the GPU clock up between tokens.
- Decode lookahead depth: `GMLX_DECODE_LOOKAHEAD_DEPTH` extends expert
  prestage prediction up to three MoE layers ahead, gated independently per
  layer and depth.
- Hunyuan 3 MoE fusion: routing-scores fold and shared-expert ride-along on
  the fused streaming path.
- MiniMax-M3 streaming enables miss-shed.
- Streaming MoE guide (`docs/streaming.md`) and a `bench/plot-bench.py
  batch-scaling` chart grammar for scheduler A/Bs.
- Chat CLI adds user customizable themes

### Changed

- `--stream-experts` now composes with MTP speculative decoding. Auto-MTP defers
  under `--stream-experts`; explicit `--speculative` opts in.
- Lookahead prestage defaults off for the `glm_moe_dsa` and `deepseek_v32`
  families.
- Feeder and lookahead end-of-run stats print only at `-v` on run and chat.
- `gmlx talk` takes the model as a positional (like `chat`/`run`, replacing
  `--model`) with tab completion, `max_tokens` now unset by default.

### Fixed

- Speculative serving: admitting a request into a live batch could kill every
  request in flight in three ways. A rope-deltas broadcast error on the
  qwen3.5/3.6 prompt path, `'RotatingKVCache' object has no attribute 'rotated'`
  on sliding-window models, and a shared-KV shape mismatch with the gemma
  assistant drafter. Deltas are now zero-padded to the live width, rotating
  caches are lifted to the batch class before the join, and injected rows are
  aligned into the drafter's view at the new width.
- Chat died at the first token with `There is no Stream(gpu, N) in current
  thread` on models carrying precomputed RoPE frequencies (Gemma 4 and other
  scaled-RoPE families).
- Menu bar pid tracking
- MoE expert controls (`--moe-experts` mass, probe, fixed-k, lookahead)
  silently no-opped on Hunyuan 3, whose gate submodule is named `router`.
- Streamed GLM decode could return corrupted output: under async pipelining,
  prestage evictions could overwrite arena slots a still-executing gather was
  reading. Layer outputs are now evaluated before any arena mutation.
- Loading an MTP model for streaming no longer wires the resident buffer set,
  which marched wired memory through the free-page floor on over-RAM targets.
- Miss-shed no longer costs a second per-layer host sync, and streamed decode
  uses the fused GLU pair instead of falling back to the stock triple-gather.

## [0.1.0] - 2026-07-19

First public release: a local inference platform for Apple Silicon that runs
GGUF models natively on MLX with no conversion, built on the companion
[mlx-kquant](https://github.com/asher/mlx-kquant) project's Metal kernels.
The feature surface at this release is documented in the
[README](README.md) and [docs/](docs/README.md); changes are recorded here
from the next release on.

## [0.0.1] - 2026-07-19

Initial public release: packaging and release-pipeline validation.
