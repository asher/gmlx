# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Decode-priority prefill pacing for the batched server: live decode
  batches keep a configurable GPU-time share while admission prefills run
  (`server.decode_prefill_ratio` / `--decode-prefill-ratio` /
  `GMLX_DECODE_PREFILL_RATIO`, default 1.0; 0 = stock). Deep-context
  concurrent decode no longer stalls behind other requests' prefills
  (measured: c2 per-stream decode +77% at 14k depth, ~5x at 50k; c1
  unaffected).
- Unified ragged-plan batched decode for the qwen3_5 family: concurrent
  streams at different context depths stay on the fused ragged-attention
  kernel instead of a per-row fallback loop
  (`GMLX_RAGGED_UNIFIED_PLAN=0` disables).

## [0.1.0] - 2026-07-19

First public release: a local inference platform for Apple Silicon that runs
GGUF models natively on MLX with no conversion, built on the companion
[mlx-kquant](https://github.com/asher/mlx-kquant) project's Metal kernels.
The feature surface at this release is documented in the
[README](README.md) and [docs/](docs/README.md); changes are recorded here
from the next release on.

## [0.0.1] - 2026-07-19

Initial public release: packaging and release-pipeline validation.
