# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Lossy MoE decode levers for streamed models: `--moe-miss-shed` drops
  non-resident experts while keeping a share of routing mass, `--moe-layer-shed`
  skips routed MoE paths per token. Never on by default.
- GPU keep-warm for streamed decode (`--gpu-keepwarm`) holds the GPU clock up
  between tokens; decode-gated, so an idle server pays no power cost. Measured
  +45% on GLM and +32% on Hunyuan 3.
- Decode lookahead depth (`GMLX_DECODE_LOOKAHEAD_DEPTH`): expert prestage
  prediction up to three MoE layers ahead.
- Hunyuan 3 MoE fusion (routing-scores fold, shared-expert ride-along) and
  MiniMax-M3 streaming via normalized routing weights through the mix seam.
- `detect_arch` and `load_tokenizer_from_gguf` promoted to the stable public API:
  synthesize the HF tokenizer from GGUF metadata without loading the model.
- Streaming guide (`docs/streaming.md`) and a `bench/plot-bench.py batch-scaling`
  chart grammar for scheduler A/Bs.

### Changed

- `--stream-experts` now composes with MTP speculative decoding. Auto-MTP defers
  under `--stream-experts`; explicit `--speculative` opts in.
- Lookahead prestage defaults off for the `glm_moe_dsa` and `deepseek_v32`
  families; feeder and lookahead end-of-run stats print only at `-v`.

### Fixed

- Speculative serving: admitting a request into a live batch could kill every
  request in flight, in three ways - a rope-deltas broadcast error on the
  qwen3.5/3.6 prompt path, `'RotatingKVCache' object has no attribute 'rotated'`
  on sliding-window models, and a shared-KV shape mismatch with the gemma
  assistant drafter. Deltas are zero-padded to the live width, rotating caches
  are lifted to the batch class before the join, and injected rows are aligned
  into the drafter's view at the new width.
- Streamed GLM decode could return corrupted output: prestage evictions could
  overwrite arena slots a still-executing gather was reading. Layer outputs are
  now evaluated before any arena mutation.
- MoE expert controls silently no-opped on Hunyuan 3, whose gate submodule is
  named `router`.
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
