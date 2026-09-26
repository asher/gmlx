# Performance tuning

A local model's speed on Apple Silicon depends mostly on memory bandwidth,
the quant and the depth of the context. gmlx has features that trade among
speed, memory and exact output, each with a page of its own, and a benchmark
mode that measures your own setup.

- [What determines speed](#what-determines-speed)
- [The features](#the-features)
- [Measuring](#measuring)
- [Choosing a quant for speed](#choosing-a-quant-for-speed)
- [Sparse attention at depth](#sparse-attention-at-depth)

## What determines speed

Decoding one stream is limited by memory bandwidth. Each token reads the
model's active weights once, so tokens per second is about the
bandwidth divided by the active bytes. Smaller quants therefore decode
faster when nothing else limits them. A MoE model reads only the routed
experts for a token, so it decodes at the speed of a small model with the
quality of a large one.

Prefill, which reads the prompt, is limited by GPU compute instead, and
gains more from compute and batching than from small weights. At long
contexts, more of the time goes to the KV cache and attention than to the
weights.

## The features

Each gmlx feature targets one of these limits, at a cost in memory, speed
or exact output.

| Feature | What it gains | What it costs | Page |
|---------|---------------|---------------|------|
| A uniform K-quant file | It decodes faster than a mixed file of the same model. | It can take more memory than a mixed low-bit build. | [Choosing a quant for speed](#choosing-a-quant-for-speed) |
| Speculative decoding | The model decodes faster with the same output. | The drafter takes memory, and the gain shrinks with many streams. | [Speculative decoding](speculative-decoding.md) |
| The prompt cache | The server skips prefill for the start of a prompt that it has seen before. | The entries take memory or SSD space. | [Prompt cache](prompt-cache.md) |
| Batched serving and admission pacing | Several clients get more total throughput, and streams stay steady while a long prompt arrives. | A new request starts a little later. | [Concurrent requests](concurrency.md) |
| A quantized KV cache | The KV cache takes a half to a quarter of the memory. | Quality drops a little at 4 bits, and the model accepts fewer drafts. | [KV cache quantization](kv-quantization.md) |
| Sparse attention | The cost of attention stops growing past 8K tokens. | It changes the output, is opt-in, and applies to the `llama` family only. | [Sparse attention at depth](#sparse-attention-at-depth) |
| Streaming | A MoE model larger than RAM can run. | Decoding runs at a few tokens per second. | [Models larger than memory](streaming.md) |

To see how much memory a model and its context take, and which settings
limit it, read [Memory and the KV cache](memory.md).

## Measuring

```sh
# Prefill and decode speed at several prompt lengths.
gmlx run model.gguf --bench "128,512,2048" --bench-runs 3

# Decode speed at depth, timed after 0, 4096 and 16384 tokens of context.
gmlx run model.gguf --bench-depths "0,4096,16384"
```

`--bench-runs 3` reports the best run at each length. A laptop slows down
under sustained load, and the best run is the one that throttling affected
least. Measure long-running work at the slower sustained speed, with a
warmup of minutes rather than tokens. When you compare two settings one
after the other, let the Mac cool between them, because the second run
starts on a hotter chip.

Prefill speed at a 512-token prompt is the usual benchmark figure, but it
describes short prompts only. If your real work is a coding agent with a
30K-token prompt, compare engines and models at that depth.

When a speed at depth is unexpected, check first which attention kernel
runs. Deep decoding and speculative verification should run on fused
routes, and a one-time warning appears when a verify call does not. The
switches that log the routes are in
[Debug switches](internals/debug-switches.md).

Absolute speeds scale with the chip's memory bandwidth. A Pro chip has
about half the bandwidth of a Max, and a base chip a quarter to a fifth,
while the ratios between models and quants stay the same.

The kernels were tuned on M3 and M5 GPUs, and M5 and later chips also get
a path through the GPU's neural accelerators. M1 and M2 run the same
kernels, but they have not been tuned or compared with llama.cpp.
[Benchmarks](benchmarks.md) compares gmlx with llama.cpp on the same
files, across many models and up to 200K tokens of context.

## Choosing a quant for speed

The file that you download matters as much as any setting. Community
GGUFs come in two styles. In a uniform file, such as `Q6_K` or `Q4_K_M`,
all quantized tensors use one K-quant codec. A mixed file, such as
Unsloth's `UD-*` builds, raises some tensors to Q8_0 or float to keep
quality at a low average size. K-quants reach about half the KL divergence
of MLX's native affine quantization at the same size, as the
[mlx-kquant KLD table](https://github.com/asher/mlx-kquant#why) shows.

A mixed file decodes slower. When one stream decodes, the slowest matrix
product of each layer sets the time per token, and the raised Q8_0 and
float tensors run slower than the K-quant kernels. The gap is largest on
dense models, and [Benchmarks](benchmarks.md) has examples. Follow these
rules:

- If a uniform Q6_K fits in memory, prefer it over a mixed UD-Q4_K_XL. It
  decodes faster and is more accurate, and the mixed file's only advantage is
  that it uses less memory.
- Check before you download. `gmlx validate <ref>` lists the codecs in a
  file, and a file with one K-quant codec throughout is the fastest.
- Use a mixed low-bit build only when memory requires it.

## Sparse attention at depth

At a deep context, decode attention reads the whole KV cache for each
token, and from about 8K tokens that read is a large part of each step.
[`GMLX_SPARSE_ATTN=1`](env-vars.md#runtime) switches decoding past that
depth to top-k sparse attention. The runtime keeps a small index over the
cache, and each step reads only the best-scoring pages within a fixed
token budget, plus the first tokens and the most recent pages. The cost of
attention then stops growing with depth.

Sparse attention changes the output, so it is off by default. It applies
only to the `llama` family, where its quality was measured, and there the
model still finds a single fact placed deep in the context. The token budget and the depth
where it starts are the `GMLX_SPARSE_K` and `GMLX_SPARSE_MIN_S` variables.
A quantized KV cache and speculative verification always use full
attention.
