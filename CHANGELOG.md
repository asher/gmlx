# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- glm-dsa/DeepSeek-V3.2 sparse decode uses the stock top-k gather again
  (O(index_topk) per step); the mask-path workaround is now opt-in via
  `GMLX_DSV32_MASK_DECODE=1`.

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
