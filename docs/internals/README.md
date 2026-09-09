# Internals

How gmlx works internally, for contributors. Users start at
[../README.md](../README.md).

| Page | Contains |
|------|----------|
| [serving-architecture.md](serving-architecture.md) | how the loader, engine, batching and HTTP layers compose, plus the config-server call graph |
| [speculative-batching.md](speculative-batching.md) | how speculative decoding and continuous batching run together |
| [prompt-cache.md](prompt-cache.md) | prompt cache tiers per architecture, reuse counters, environment switches |
| [adding-architectures.md](adding-architectures.md) | what adding a model family involves and the acceptance gate |
| [testing.md](testing.md) | test tiers, GPU-gated invocations, the end-to-end harnesses |
| [upstream-upgrades.md](upstream-upgrades.md) | bumping the pinned mlx-vlm, mlx-lm and mlx versions |
| [streaming-measurements.md](streaming-measurements.md) | the measurements behind the streaming guide's lossless and lossy settings tables |
| [debug-switches.md](debug-switches.md) | environment variables for isolating defects |
