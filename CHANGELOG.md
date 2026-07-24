# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `detect_arch` and `load_tokenizer_from_gguf` promoted to the stable public
  API (`gmlx.__all__`): synthesize the HF tokenizer from GGUF metadata without
  loading the model. Consumed by mlx-kld's `[gguf]` scoring extra for
  tokenizer parity checks ahead of the model load.
- Lossy MoE decode levers for streamed models: `--moe-miss-shed` drops
  non-resident experts while keeping a configured share of routing mass, and
  `--moe-layer-shed` probabilistically skips routed MoE paths per token. Both
  trade output quality for throughput, are never on by default, and are wired
  through config, serve, and the run/chat overlays.
- GPU keep-warm for streamed decode: `--gpu-keepwarm` (config
  `server.gpu_keepwarm`) holds the GPU clock up between tokens. The heartbeat
  is decode-gated, so an idle server pays no power cost. Measured +45% on GLM
  and +32% on Hunyuan 3 production configs.
- Decode lookahead depth: `GMLX_DECODE_LOOKAHEAD_DEPTH` extends expert
  prestage prediction up to three MoE layers ahead, gated independently per
  layer and depth.
- Hunyuan 3 MoE fusion: routing-scores fold and shared-expert ride-along on
  the fused streaming path.
- MiniMax-M3 streaming: normalized routing weights pass through the mix seam,
  enabling miss-shed on streamed M3 with stock base models.
- Performance guide: lossy-lever decision procedure with measured settings,
  sampling-regime certification guidance, and the GPU keep-warm section.

### Changed

- `--stream-experts` now composes with MTP speculative decoding. Auto-MTP
  defers under `--stream-experts`; explicit `--speculative` opts in.
- Lookahead prestage defaults off for the `glm_moe_dsa` and `deepseek_v32`
  families.
- Feeder and lookahead end-of-run stats print only at `-v` on run and chat.

### Fixed

- MoE expert controls (`--moe-experts` mass, probe, fixed-k, lookahead)
  silently no-opped on Hunyuan 3, whose gate submodule is named `router`.
- Streamed GLM decode could return corrupted output: under async pipelining,
  prestage evictions could overwrite arena slots a still-executing gather was
  reading. Layer outputs are now evaluated before any arena mutation.
- Loading an MTP model for streaming no longer wires the resident buffer set,
  which marched wired memory through the free-page floor on over-RAM targets.
- Miss-shed no longer costs a second per-layer host sync; routing scores fold
  into the router eval, recovering the shed's IO win at high hit rates.
- Streamed decode uses the fused GLU pair; arena-backed calls were falling
  back to the stock triple-gather on every token.

## [0.1.0] - 2026-07-19

First public release: a local inference platform for Apple Silicon that runs
GGUF models natively on MLX with no conversion, built on the companion
[mlx-kquant](https://github.com/asher/mlx-kquant) project's Metal kernels.
The feature surface at this release is documented in the
[README](README.md) and [docs/](docs/README.md); changes are recorded here
from the next release on.

## [0.0.1] - 2026-07-19

Initial public release: packaging and release-pipeline validation.
