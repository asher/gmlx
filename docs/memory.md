# Memory and the KV cache

A model needs memory for its weights and for the KV cache of its context.
This page explains how to estimate both, which settings limit the KV
cache, and how gmlx manages the memory that macOS lets the GPU use.

- [The weights and the KV cache](#the-weights-and-the-kv-cache)
- [Settings that limit memory](#settings-that-limit-memory)
- [The GPU memory limit](#the-gpu-memory-limit)
- [The MLX buffer cache](#the-mlx-buffer-cache)

## The weights and the KV cache

The weights take about the size of the GGUF file in memory. The KV cache
of a standard dense model takes this much for each token of context:

```text
bytes per token = 2 (K and V) x layers x kv_heads x head_dim x 2 (bf16)
```

An 8B model with 32 layers, 8 KV heads and a head dimension of 128 uses
128 KB for each token, which is 4 GB for a 32K-token context. A 32B dense
model with 64 layers uses 256 KB for each token, or 8 GB at 32K. For
long-context agent work, the KV cache can be as large as the weights.

Many families use much less than the formula gives. Sliding-window
layers, as in gemma, stop growing at the size of the window. Hybrid
models, such as Qwen3.5, Qwen3.6, Qwen3.8, Granite 4 and Nemotron-H, keep
a small fixed state on most layers and a full KV cache only on their few
attention layers. Qwen3.6-27B, with 16 attention layers of 64, therefore
uses 2.1 GB at 32K, against 8 GB for a dense model of the same depth.
[MLA](glossary.md) models, such as the DeepSeek family, store a compressed
cache. The capacity planner of the server counts all of these.

To see the numbers of a running server, read the `memory` and `capacity`
sections of `GET /v1/metrics`. `POST /v1/estimate` estimates whether a
request fits before you send it. [Capacity and live-request
metrics](api.md#capacity-and-live-request-metrics) describes both.

## Settings that limit memory

These settings reduce memory, the cheapest first:

- A quantized KV cache. `--kv-bits 8` about halves the KV cache with
  almost no quality cost, and `--kv-quant-scheme kvarn` keeps that quality
  at 6 bits in less memory. [KV cache quantization](kv-quantization.md)
  describes both schemes and the models that gain from them. On the
  server, they are the [load keys](config.md#model-loading).
- A context limit. On `run` and `chat`, `--max-kv-size` keeps a rolling
  window of the most recent tokens and drops the oldest ones. Under kvarn,
  the window is quantized when it is at least the kvarn minimum, and a
  smaller window exits with an error. Plain `--kv-bits` cannot quantize a
  rolling window, so the command refuses that combination. On the server,
  [`max_kv_size`](config.md#loadmax_kv_size) limits the context of a
  request and keeps no rolling window.
- A smaller prefill chunk. `--prefill-step-size` makes the 2048-token
  chunk smaller, which lowers the memory peak of a long prompt and slows
  prefill.

## The GPU memory limit

macOS limits how much memory the GPU may wire to a share of RAM that
depends on the machine. The server keeps its loaded models within
[`server.budget_gb`](config.md#serverbudget_gb), and gmlx streams a MoE
model that does not fit, as [Models larger than memory](streaming.md)
describes. If one dense model and its cache sit right at the limit on a
Mac with a lot of RAM, `sudo sysctl iogpu.wired_limit_mb=<MB>` raises the
limit at your own risk until the next restart. Leave several GB for
macOS.

## The MLX buffer cache

MLX keeps freed GPU buffers in a pool, so that it can reuse them. This
buffer cache is separate from the KV cache and the prompt cache, and its
limit never removes their contents. At deep contexts, a model near the
size of RAM leaves prefill buffers of several GB in this pool. Without a
limit, the pool can use up the free memory of the Mac and freeze it
instead of failing with an error, because MLX counts the pool as free
while macOS counts it as wired.

The server therefore always limits the pool, and logs the limit on a
`[serve] MLX cache limit:` line. When the largest configured model takes
more than 60% of the GPU working set, the limit is a quarter of the memory
that is left. Otherwise, it is 5% of the working set. Either way, the
limit is between 4 and 12 GiB. The memory governor also checks the free
memory of macOS on every tick.

[`server.cache_limit_gb`](config.md#servercache_limit_gb) sets the limit
in GiB, which benchmarks should do so that runs compare. `0` turns the
buffer cache off, and a negative value removes the limit.
[`GMLX_CACHE_LIMIT_GB`](env-vars.md#runtime) sets the limit through the
environment.
