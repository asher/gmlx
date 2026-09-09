# Performance

What makes a local model fast on Apple Silicon, what each gmlx setting gains,
and how to measure your own setup. It is written for anyone choosing a quant,
turning on speculative decoding, or serving more than one client.

- [What determines speed](#what-determines-speed)
- [The settings](#the-settings)
- [Measuring](#measuring)
- [Reference numbers](#reference-numbers)
- [Choosing a quant for speed](#choosing-a-quant-for-speed)
- [MTP speculative decoding](#mtp-speculative-decoding)
- [The prompt cache](#the-prompt-cache)
- [Serving concurrent requests](#serving-concurrent-requests)
- [Sparse attention at depth](#sparse-attention-at-depth)
- [Memory and the KV cache](#memory-and-the-kv-cache)

## What determines speed

Single-stream decode is memory-bandwidth-bound: each token reads the model's
active weights once, so tokens per second is roughly bandwidth divided by
active bytes. Three things follow. Smaller quants decode faster when nothing
else limits them. MoE models read only the routed experts for a token, so
they decode at the speed of small models with the quality of large ones. And
a chip with more memory bandwidth is faster in proportion, independent of
anything gmlx does.

Prefill, the prompt processing, is compute-bound instead, and benefits from
GPU compute and batching more than from small weights. Long-context work
shifts time from weights to the KV cache and attention. Each setting in the
next section addresses one of these regimes.

## The settings

| Setting | What it gains | Cost | Where to set it |
|-------|--------------|------|-----------------|
| a uniform K-quant file | up to 64% faster decode than a mixed file of the same model | none | the file you download, see [Choosing a quant](#choosing-a-quant-for-speed) |
| speculative decoding | 1.1x to 2x decode, identical output | memory for the drafter, less gain at high concurrency | automatic on models with a head, otherwise `--draft-gguf` or `speculative:` |
| the prompt cache | skips prefill for any prefix seen before | RAM or SSD for the entries | on by default, and `cache:` keys add the SSD tier |
| a quantized KV cache | halves to quarters cache memory | a small quality cost at 4 bits, and a cost to speculation described under MTP | `--kv-bits` or `--kv-quant-scheme kvarn`, or the load keys `kv_bits` and `kv_quant_scheme` |
| admission pacing | keeps live streams decoding while a long prompt is admitted | slower admission of the new request | `server.decode_prefill_ratio`, default `auto` |
| sparse attention | depth-flat attention cost past 8k tokens | lossy, opt-in, llama-family only | `GMLX_SPARSE_ATTN=1` |
| streaming | runs a MoE bigger than RAM | single-digit tokens per second | [streaming.md](streaming.md) |

Flags are documented in [cli.md](cli.md), config keys in
[server-config.md](server-config.md) and env switches in
[env-vars.md](env-vars.md).

## Measuring

```sh
# prefill + decode throughput at several prompt lengths
gmlx run model.gguf --bench "128,512,2048" --bench-runs 3

# decode speed AT depth: how fast is token 16,001?
gmlx run model.gguf --bench-depths "0,4096,16384"
```

`--bench-runs 3` reports the best run at each length, which matters on a
laptop because sustained runs throttle, and the best run is the one a
thermally degraded repeat cannot lower. How long a rested machine holds its
boost clocks depends on the chassis: a 14-inch M5 Max ran roughly twenty
minutes of streamed MoE decode before settling about 20% lower, and the
16-inch chassis holds boost longer. A chat-length session on a rested
machine therefore runs at the faster rate the whole time, while anything
long-running should be measured at the sustained rate, with warmup sized in
minutes of decode rather than tokens. In back-to-back comparisons the second
arm runs on a hotter chip, so let the machine cool between arms.

Prefill throughput at a 512-token prompt is the conventional benchmark
figure, but it is a short-context number. If your real workload is a coding
agent with a 30k-token prompt, compare engines and models at that depth.

When a depth number is unexpected, check which attention kernel is running
before anything else. Deep decode and speculative verify should run on fused
routes, and a one-shot warning fires when a verify-shaped call does not. The
route log and SDPA trace switches are in
[internals/debug-switches.md](internals/debug-switches.md).

## Reference numbers

Measured on a 14-inch M5 Max MacBook Pro with 128 GB, at 512-token
prompts, as medians of repeated runs:

| Model | File | Decode | Decode (MTP) | llama.cpp decode (spec) | Prefill | llama.cpp prefill |
|-------|------|--------|--------------|-------------------------|---------|-------------------|
| gemma-4-12B-it (dense) | Q6_K | ~44 tok/s | ~72 tok/s | ~54 tok/s | ~850 tok/s | ~730 tok/s |
| Qwen3.5-9B (dense) | Q6_K | ~70 tok/s | ~112 tok/s | ~76 tok/s | ~1600 tok/s | ~1140 tok/s |

Against llama.cpp on the same GGUF, prefill is faster on all models at all
depths measured. Above 4k tokens decode is faster too, and the gap grows as
the context deepens. The full tables to
200k tokens, the charts, the methodology and the weight provenance are in
[benchmarks.md](benchmarks.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/perf/fleet-ratio-dark.svg">
  <img src="assets/perf/fleet-ratio.svg" alt="gmlx vs llama.cpp throughput speedup across the fleet">
</picture>

Your absolute numbers scale with your chip's memory bandwidth. A Pro-tier
chip has about half the bandwidth of a Max, and a base M-series chip a
quarter to a fifth, but the ratios between models and quants stay the same.

Unless a table names another machine, the numbers and llama.cpp comparisons
in these docs come from an M5 Max with a 40-core GPU. The kernels target
the matrix hardware in M3 and later GPUs, and MoE prefill and expert-gather
batches route through a kernel validated on an M3 Max. M1 and M2 run the
standard kernel paths, but they have not been a tuning focus or benchmarked
against llama.cpp, so if you run the bench commands on one, an issue with
your numbers is welcome.

## Choosing a quant for speed

The file you download matters as much as any runtime flag. Community GGUFs
come in two styles: in a uniform file, such as `Q6_K` or `Q4_K_M`, all
quantized tensors use a single K-quant codec, while a mixed file, such as
Unsloth's `UD-*` builds, promotes some tensors to Q8_0 or float to preserve
quality at low average bits. K-quants themselves reach roughly half the KL
divergence of MLX's native affine quantization at equal bitrate, as
[mlx-kquant's KLD table](https://github.com/asher/mlx-kquant#why) shows.

Mixed files decode measurably slower, because at single-stream decode each
layer's slowest matmul sets the token time, and the promoted float and Q8_0
tensors run slower than the K-quant kernels. On an M5 Max, switching from
the mixed UD build to a uniform Q6_K of the same model sped
decode up 64% on the dense Qwen3.6-27B and 15% on the MoE Qwen3.6-35B-A3B,
with output quality equal or better.

The practical rules:

- If a flat Q6_K fits your memory, prefer it over a mixed UD-Q4_K_XL. It is
  faster to decode and more accurate. The mixed file's only advantage is
  lower memory use.
- Check before downloading. `gmlx validate <ref>` lists the codecs in the
  file. Prefer files that are a single K-quant codec end to end.
- Use mixed low-bit builds only when memory requires them, not as a default.

## MTP speculative decoding

Models that include a native multi-token-prediction head, which the dense
Qwen3.5, 3.6 and 3.8 models do, get [speculative decoding](glossary.md)
automatically on `run` and `chat`. The head drafts tokens ahead and the base
model verifies them, so the output is exactly what the base model would have
produced, only faster when drafts are accepted. `--no-mtp` turns it off.
gemma-4, Muse Glimmer and Qwen3.8-Flash-Next use the two-file form instead:
a small companion drafter GGUF, passed with `--draft-gguf` or, for the last
two, found on its own when it sits beside the target. On the server the
`speculative:` config key enables either form. A configured companion takes
precedence over a native head, and `--native-mtp`, or the per-model
`native_mtp: true`, forces the head.

Gains depend on acceptance rate and context depth. Speculation roughly
doubles dense-model decode at short context, and still gives 1.4x to 1.8x
through 110k tokens and 1.2x to 1.4x at 200k. MoE models gain less, and on
some of them the gain becomes a loss at depth, so benchmark before enabling
it there. The per-model speedup curves are in [benchmarks.md](benchmarks.md).
Predictable text such as code accepts more drafts than freeform prose, so
measure your own model and workload:

```sh
gmlx run model.gguf --bench-depths "0,4096" --speculative     # accept rate + speedup
```

Speculation and batching compete for the same bandwidth. Verifying a draft
widens each request's weight reads, which costs little while one stream
decodes and much more once several do, so the server applies a per-model
width cap: speculation runs while the live batch is narrow, the batch decodes
plain past the cap, and speculation resumes once it drains. The default cap
depends on the drafter and on whether the target routes experts, and a
per-model `speculative_width_cap` key overrides it, as described in
[server-config.md](server-config.md#speculative_width_cap). For the
transition mechanics, read
[internals/speculative-batching.md](internals/speculative-batching.md).

Quantizing the KV cache shifts the target's verify logits away from the draft
head and reduces accepted drafts, by about a third at 4 bits. Keep the KV
cache in full precision when speculation is on if you can, and prefer 8 bits
if memory forces quantization.

### DFlash 2 drafters

[DFlash 2](https://inco.ai/blog/dflash2/) is a block-diffusion drafter with
checkpoints for Qwen3.8-27B and Muse-Glimmer-30B. One drafter forward
proposes a whole block of tokens and the target verifies the block in one
pass, so a round costs one small forward plus one verify instead of a verify
for each drafted token.

Pair it with `--draft-gguf`, or let `gmlx discover` do it, since a DFlash 2
header declares its base model and discovery can pair it across directories.
The drafted depth defaults to the checkpoint's trained block, 8 on Qwen3.8
and 16 on Muse Glimmer, and `--draft-block-size` lowers it. Because the
drafter is single-stream, the server width cap is 1.

Acceptance is exact-match by default, so greedy output is token-identical to
plain decoding and sampled output follows the target's sampler, and
`--stochastic-mtp` applies to DFlash 2 as well. On Qwen3.8-27B at Q6 the
drafter roughly tripled decode speed over plain decoding and was about 1.5x
the native head. The runs are in [benchmarks.md](benchmarks.md).

### Stochastic acceptance

By default a draft is accepted only when it matches the token the base model
would emit, which keeps output token-identical. At temperature above zero
that is also a limit, because a draft cannot match a sampled token more often
than the target's probabilities allow. `--stochastic-mtp`, or
`stochastic_mtp: true` in the server config, removes the limit with rejection
sampling: drafts are sampled and accepted with probability `min(1, p/q)`,
which preserves the sampling distribution exactly. The output remains a true
sample from the distribution plain decoding samples from, although the tokens
are no longer bit-identical to a non-speculative run. Greedy requests are
unaffected.

Measured gains range from a few points of acceptance on a Q6 dense model to
around 14 points on a low-bit MoE quant, because the lower the trunk
precision and the flatter the text, the more acceptance exact-match forgoes.
Turn it on when you sample and want throughput.

## The prompt cache

The server keeps a cross-request [prompt cache](glossary.md), so a request
whose prefix was seen before skips prefill for the cached span. Resending a
32k-token prefix reduces tens of seconds of prompt processing to a sub-second
time-to-first-token, which makes this the largest speedup for agent
workloads, since they resend a large, mostly stable system prompt each turn.

What reuse to expect depends on the model family, because recurrent state
cannot be rolled back:

| Family | Tier | Identical resend | Next turn | Branch or regenerate |
|--------|------|------------------|-----------|----------------------|
| dense and plain-KV MoE | block | full | full | full, at 16-token block granularity |
| GDN hybrids such as Qwen3.5 and 3.6 | checkpoint | all but the last token | to the turn boundary snapped to a 2048-token grid, about 90% at a 9k history | to the deepest checkpoint below the divergence |
| sliding-window models such as gemma-4 and gpt-oss | checkpoint | as GDN | to within a few tokens of the divergence once the prefix clears the window | as GDN |
| pure-recurrent and CacheList, such as falcon-h1 and deepseek4 | exact | full | full, since each turn extends the stored sequence verbatim | none. An edited history prefills from scratch |

Sliding-window models under `--speculative` keep no record of generated
tokens, so their next-turn reuse comes from the prefill boundaries alone and
the reply is re-prefilled. Under kvarn KV, dense models move to the exact
tier, for the reason
[internals/prompt-cache.md](internals/prompt-cache.md#under-kvarn-kv) gives.

The optional SSD tier persists entries across restarts and holds more
entries than RAM would. Turn it on with `gmlx init --disk-cache` or the
`cache:` block in the config, whose keys are in
[server-config.md](server-config.md#cache-keys). Entries are evicted by size
budget, and hit and store counts are reported on `GET /v1/metrics`.

Thinking templates that strip prior-turn `<think>` blocks from the
re-rendered history diverge right after the assistant header, so a
full-length entry for the reply can never match. The server keys the stored
entry on the predicted next-turn render instead, and what follows the
divergence re-prefills, as it does on every server. That is a property of the
template.

A warm hit restores more than the prompt's KV. A finished request's whole
sequence, prompt plus reply, is stored back so the next turn of the
conversation warm-starts past it, and on a speculative model the drafter's
own KV is saved beside the target's so acceptance is not degraded while it
rebuilds. The cache layers behind this, their counters and the triage
switches are in [internals/prompt-cache.md](internals/prompt-cache.md).

## Serving concurrent requests

The server decodes all active streams as a single batch. Because decode is
bandwidth-bound and the batched step reads the weights once for all streams,
aggregate throughput rises with client count while each stream loses less
than its proportional share. On an M5 Max with Qwen3.6-35B-A3B Q6_K, three
streams give 1.3x to 1.7x the aggregate of one, and the ratio falls as
context grows, because attention work is done separately for each stream.

What needs managing is admission, because a new request's prompt must
prefill while existing streams are mid-decode. Prefill runs in 2048-token
chunks, and at depth a chunk costs hundreds of decode steps of GPU time, so a
scheduler that alternates one decode step with one chunk lets a long
admission stall live streams. When chunks are short, on the other hand,
pacing only delays admission and narrows the decode batch. Two settings cover
the two symptoms:

| Symptom | Setting | Default | Effect |
|---------|---------|---------|--------|
| a stalled decode batch | `server.decode_prefill_ratio` | `auto` | paces admissions only when a decoding stream would otherwise fall below half its batched rate. A number pins the ratio, and `0` restores strict alternation |
| a stream that pauses | `server.prefill_tick_ms` | `500` | halves each chunk until its predicted wall time fits the budget. Set `0` for batch jobs where only aggregate throughput matters |

Paced admission bounds a waiter's time-to-first-token at twice its unpaced
prefill, and prefill runs at full speed whenever nothing is decoding, which
leaves single-client serving unaffected under any setting. In the serve
benchmarks a second client arriving at 14k tokens slowed the live stream to
4% of its decode rate under strict alternation, whereas paced it kept 80%,
with the second client's time-to-first-token unchanged. Both keys are
documented under [Scheduling](server-config.md#scheduling), and both are read
live, so a running server can be retuned.

Pacing decides how admissions share GPU time, and the speculative width cap
decides which decode mode each batch runs in. Larger than either is the
prompt cache: a warm prefix skips its prefill entirely, so an agent session
that resends a cached history is admitted at almost no cost.

Concurrent streams often share a prefix, whether a common system prompt or
histories restored from the cache. The server detects the sharing from the
streams' token ids and decodes such batches through a cascade kernel that
reads the prefix once for the whole batch. Four streams on a 12k-token system
prompt decode about 1.4x faster aggregate, and the gain grows with prefix
length and stream count. It is exact and on by default, with switches under
the `GMLX_CASCADE_SDPA` rows in [env-vars.md](env-vars.md#runtime).

## Sparse attention at depth

At deep context, decode attention reads the whole KV cache for each token,
and from about 8k tokens that read is a large share of the step.
`GMLX_SPARSE_ATTN=1` switches decode past that depth to top-k sparse
attention: the runtime keeps a small
index over the cache, and each step attends only the best-scoring pages
within a fixed token budget, plus the attention sink and the most recent
pages, so attention cost stops growing with depth.

This is lossy, which is why it is opt-in. On a Llama-3.1-8B Q6_K at 32k
depth with the default 2048-token budget, the divergence from full attention
is of the same order as the Q6 quantization noise. Decode runs at 1.4x
single-stream and 1.8x aggregate at three streams on a shared prompt.
Needle lookups deep in the context keep working.

The route engages only on architectures whose quality has been measured,
because the property it depends on is architectural: full-attention stacks
concentrate decode attention into a small key set, and the sliding-window
hybrids measured do not. gemma-4 is excluded and runs full attention whatever
the switch. The budget and engagement depth are the `GMLX_SPARSE_K` and
`GMLX_SPARSE_MIN_S` rows in [env-vars.md](env-vars.md#runtime). Quantized KV
caches and speculative verify steps always run full attention.

## Memory and the KV cache

Weights use about the GGUF file size of memory. For a standard dense model
the KV cache takes:

```text
bytes per token = 2 (K and V) x layers x kv_heads x head_dim x 2 (bf16)
```

An 8B-class model with 32 layers, 8 KV heads and head dim 128 uses 128 KB for
each token, which is 4 GB for a 32k-token session, and a 32B-class dense
model with 64 layers uses 256 KB for each token, or 8 GB at 32k. Long-context
agent work can make the cache as large as the weights.

Several families use much less memory than the formula gives. Sliding-window
layers, as in gemma, stop growing at the window size. Hybrid models such as
Qwen3.5 and 3.6, Falcon-H1, Granite 4.x and Nemotron-H keep a small fixed
state on most layers and use full KV only on their few attention layers, so
Qwen3.6-27B, with 16 attention layers of 64, uses 2.1 GB at 32k where a dense
64-layer model would use 8.4 GB. MLA models such as the DeepSeek family store
a compressed cache. The capacity planner accounts for all of these.

Settings, lowest cost first:

- A quantized KV cache. `--kv-bits 8` roughly halves the cache at nearly no
  quality cost. `--kv-quant-scheme kvarn` gives that fidelity at 6 bits in
  about half the fp16 cache and stays usable down to 4. [KV cache
  quantization](#kv-cache-quantization) says what each scheme does, which
  models gain from it and what the fidelity data shows. Server-side these
  are the [load keys](server-config.md#load-keys).
- `--max-kv-size` caps the cache as a rolling window, dropping the oldest
  context. On `run` and `chat` the window quantizes under kvarn once the cap
  is at least the kvarn minimum, as [cli.md](cli.md#gmlx-run) describes.
  Plain `--kv-bits` cannot quantize a rotating window and is refused at
  start. The server's `max_kv_size` only caps the request context budget and
  builds no rotating window.
- `--prefill-step-size` shrinks the 2048-token prefill chunk to cap peak
  memory further, at some prefill-throughput cost.

macOS caps how much RAM the GPU may wire at a machine-dependent majority
share of total memory. gmlx handles the over-budget MoE case itself, as
[streaming.md](streaming.md) describes, and the server budgets resident
weights with `budget_gb`. If a single dense model plus cache sits right at
the cap on a high-RAM Mac, the limit can be raised at your own risk with
`sudo sysctl iogpu.wired_limit_mb=<MB>`, which resets at reboot. Leave the OS
several GB unallocated.

### KV cache quantization

Two schemes shrink the cache, and a single policy decides both layer by
layer, prints a `[kv]` line and reports the result as `kv_quant` on
`GET /v1/models`.

Affine quantization, `--kv-bits N` or the `kv_bits` load key, is mlx-lm's
QuantizedKVCache. Each token's K and V rows are split into groups of
`--kv-group-size` values, 64 by default, and each group stores N-bit codes
plus an fp16 scale and bias. Widths are 2, 3, 4, 6 and 8, and
`--quantized-kv-start` keeps the first stretch of the context in fp16.

KVarN, `--kv-quant-scheme kvarn` or `kv_quant_scheme: kvarn`, is
variance-normalized quantization. `--kv-bits` picks the width, 6 by default
from 2, 3, 4, 5, 6 and 8, and the `GMLX_KVARN_BITS` row in
[env-vars.md](env-vars.md#runtime) splits the key and value widths. Each
128-value slice of a head is rotated by a Hadamard transform, which spreads
outlier channels over the whole slice. K and V are stored in 128-token
records. Before rounding, each record is scaled along both axes by 16
alternating row and column normalizations in log space, a Sinkhorn iteration,
after which no token and no channel dominates the code range. Three fp16 axis
vectors in each record undo the scaling on read. The first 128 tokens, the
attention sink, stay fp16, as do the newest `--kv-tail-tokens` tokens, 1024 by
default. A record is sealed once the tail has moved past it. Decode and MTP
verify read the records in the mlx-kquant kernels and merge the fp16 tail
through a single softmax. The prompt cache stores records on its exact and
checkpoint tiers.

Which layers quantize is decided by cache shape, not model name. Growing
attention KV quantizes, except the last layer of a deep stack, which stays
fp16 under either scheme. Recurrent state and sliding windows stay fp16, with
a single exception: a `--max-kv-size` window on `run` and `chat` quantizes
under kvarn and is refused under affine. kvarn accepts head_dim 128, 256 and
512 only, which leaves head_dim-64 layers, as in gpt-oss and falcon-h1,
affine only. MLA architectures such as deepseek4, glm5_next and kimi-k3 keep
K and V in a single latent store, which kvarn declines and affine packs when
stored. The VLM media path keeps fp16. A declined model prints the reason and
runs fp16.

Quantization saves memory in proportion to how much of the cache grows with
context, and loses fidelity in proportion to how many layers it touches. The
cache shape decides both:

| Cache shape | Families | fp16 cache at 32k | What to use |
|---|---|---|---|
| dense full attention on all layers | llama, Mistral, Qwen3 dense | 4 to 8 GB for an 8B to 32B model | `--kv-bits 8`, or kvarn at 6 for equal fidelity in less memory. kvarn at 4 when memory is the limit |
| hybrid recurrent, one attention layer in four | Qwen3.5/3.6/3.8, Nemotron-H, Falcon-H1, Granite 4 | about 2 GB at 27B plus a fixed recurrent state | only when the context budget is the limit, 64k and up. The fidelity cost is small, since three layers in four never quantize |
| sliding-window mix | gemma-4 | the window layers stop growing at the window | a small saving, since only the global layers quantize |
| MLA latent | DeepSeek-V4, GLM-5.3, Kimi-K2 and K3 | already compressed by the architecture | affine only. It packs the latent pools when stored, and kvarn declines |
| head_dim 64 | gpt-oss | small per token | affine only |

`/status` reports the live cache size and `POST /v1/estimate` estimates a
load in advance. Both are in
[api.md](api.md#capacity-and-live-request-metrics).

The fidelity measure is teacher-forced logit KLD against an fp16 cache on
wikitext, from `scripts/kld_harness.py`, on two legs: one scores chunked
prefill logits, the other scores decode token by token from full prefill
depth. KLD is in nats, and lower is better. The median is the typical
position, decode p99 is the worst hundredth, where a quantizer's outliers
show, and top-1 is the share of generated positions whose argmax matches the
fp16 cache.

<!-- kld-tables -->
Qwen3.5-9B Q4_K_M, 16k context, head_dim 256, 7 of 32 layers quantized:

| cache | prefill median | decode median | decode p99 | decode top-1 |
|---|---|---|---|---|
| affine 2 | 0.02778 | 0.02745 | 0.5596 | 89.0% |
| kvarn 2 | 0.01503 | 0.00612 | 0.2301 | 94.7% |
| affine 3 | 0.00648 | 0.00606 | 0.1057 | 94.3% |
| kvarn 3 | 0.00291 | 0.00139 | 0.0313 | 97.4% |
| affine 4 | 0.00190 | 0.00183 | 0.0276 | 96.9% |
| kvarn 4 | 0.00117 | 0.00060 | 0.0071 | 98.3% |
| kvarn 5 | 0.00055 | 0.00029 | 0.0042 | 98.2% |
| kvarn k6 v5 | 0.00046 | 0.00028 | 0.0039 | 98.6% |
| affine 6 | 0.00046 | 0.00038 | 0.0045 | 98.7% |
| kvarn 6 | 0.00036 | 0.00027 | 0.0036 | 98.7% |
| affine 8 | 0.00029 | 0.00020 | 0.0027 | 98.7% |
| kvarn 8 | 0.00027 | 0.00020 | 0.0030 | 99.2% |

Qwen3.8-27B Q6_K_XL, 16k context, head_dim 256, 15 of 65 layers quantized:

| cache | prefill median | decode median | decode p99 | decode top-1 |
|---|---|---|---|---|
| affine 2 | 0.01975 | 0.02314 | 0.4697 | 90.3% |
| kvarn 2 | 0.01009 | 0.00491 | 0.1280 | 95.1% |
| affine 3 | 0.00383 | 0.00419 | 0.1034 | 96.1% |
| kvarn 3 | 0.00212 | 0.00113 | 0.0318 | 97.3% |
| affine 4 | 0.00138 | 0.00136 | 0.0268 | 97.5% |
| kvarn 4 | 0.00084 | 0.00045 | 0.0078 | 97.4% |
| kvarn 5 | 0.00041 | 0.00025 | 0.0039 | 98.4% |
| kvarn k6 v5 | 0.00034 | 0.00019 | 0.0039 | 98.5% |
| affine 6 | 0.00033 | 0.00030 | 0.0055 | 98.7% |
| kvarn 6 | 0.00027 | 0.00019 | 0.0030 | 98.8% |
| affine 8 | 0.00023 | 0.00019 | 0.0032 | 98.7% |
| kvarn 8 | 0.00021 | 0.00015 | 0.0037 | 98.9% |

Qwen3.8-27B Q6_K_XL, 32k context:

| cache | prefill median | decode median | decode p99 | decode top-1 |
|---|---|---|---|---|
| affine 4 | 0.00162 | 0.00205 | 0.0176 | 97.6% |
| kvarn 4 | 0.00104 | 0.00070 | 0.0068 | 98.1% |
| affine 6 | 0.00039 | 0.00049 | 0.0037 | 98.5% |
| kvarn 6 | 0.00033 | 0.00032 | 0.0027 | 99.4% |
| affine 8 | 0.00027 | 0.00030 | 0.0038 | 98.8% |
| kvarn 8 | 0.00026 | 0.00028 | 0.0026 | 99.0% |

Nemotron-3.5-Lightning-30B-A3B, 16k context, Mamba2 hybrid, head_dim 128:

| cache | prefill median | decode median | decode p99 | decode top-1 |
|---|---|---|---|---|
| kvarn 4 | 0.00270 | 0.00163 | 0.0508 | 98.1% |
| kvarn 6 | 0.00125 | 0.00103 | 0.0266 | 98.8% |
| affine 8 | 0.00123 | 0.00094 | 0.0335 | 98.2% |
| kvarn 8 | 0.00111 | 0.00095 | 0.0298 | 98.6% |
<!-- /kld-tables -->

At a matched width kvarn beats the affine cache on both legs at each width
below 8, by 3 to 5x on the decode median at 2 to 4 bits, and the two converge
at 8. kvarn at 6 bits falls between the affine 6 and 8 bit caches on the 9B
and matches affine 8 on the 27B, in three quarters of the 8-bit record's
bytes, although at 32k its decode median can trail affine 8 by a few percent
while its p99 and top-1 stay ahead. The split width k6 v5 keeps kvarn 6's
median with kvarn 5's p99 and top-1, for the bytes in between. Widths 2 and 3
are for experiments. The decode leg is the one a long generation accumulates.

Top-1 is closest to what a greedy or low-temperature user sees, the share of
tokens that come out identical. Median KLD measures how far the whole
next-token distribution moved, which is what sampling at temperature draws
from, and it keeps scoring positions whose argmax never changed. The p99
bounds the outliers, and those matter because a single badly wrong position
can change a reasoning chain or a tool call, and a long generation feeds its
own errors back in. Over a thousand tokens, a cache with a lower p99
therefore drifts less even when its median is not the lowest, so when two
caches differ by a few percent on one measure, take the one with the lower
p99 and the higher top-1. A width that gains 2x or more on the median gains
on all measures. The ranking does not follow width across schemes: kvarn at
2 bits matches affine at 3 on decode median and is within a point of it on
top-1, ahead on the 9B and behind on Qwen3.8-27B, and at 4 bits kvarn cuts
affine 4's decode median to a third while matching or beating it on top-1.
Because the corpus is wikitext under teacher forcing, these tables rank
caches against each other and do not predict a task score.

TurboQuant, mlx-vlm's scheme, is not offered. The harness has a `turboN`
arm, and measured on the same models and legs it falls between the other
two: ahead of affine at 2 and 3 bits, level at 4, behind at 6 and 8, and
behind kvarn at each width on all measures. At mlx-vlm's recommended 3.5-bit
setting its decode median is 2.5x kvarn 3's, at 4 bits about 3x kvarn 4's,
and at 6 bits 2x kvarn 6's, with equal or lower top-1 at each.

Decode speed depends on how much of a step the KV read is. On GDN hybrids
and gemma-4 all three caches are within run-to-run spread, but a KV-bound
dense stack is different: on Qwen3-0.6B Q8 with 27 of 28 layers quantized,
kvarn 6 decodes at 0.81x fp16 and 0.69x affine 8 at 16k, and at 0.98x and
0.75x at 32k, with prefill within 10% of both. Native MTP composes at batch
size 1, where verify rounds attend the records on the matrix-unit verify
kernels at about a decode step's cost: at head_dim 256 with 8 queries for
each KV head at 16k, that is 1.0 ms for each layer and round against 0.9 for
decode. Choose kvarn for memory and fidelity, or affine for peak decode speed
on a KV-bound dense model. The debug switches are in
[internals/debug-switches.md](internals/debug-switches.md).

KVarN is the method of Muller, Bich, Boretti, Chang, Zhuang and Cavigelli at
Huawei, "KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error
Accumulation in Reasoning Tasks",
[arXiv:2606.03458](https://arxiv.org/abs/2606.03458). Their reference vLLM
implementation is at [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN)
under Apache-2.0, and no code from it is used here. gmlx's cache is an MLX
implementation that follows the record format of
[beellama.cpp](https://github.com/Anbeeld/beellama.cpp) by Anbeeld, under
MIT, the llama.cpp fork that first brought the method to GGUF inference. The
`kvarnN` width names, the fp16 precision tail and `--kv-tail-tokens` are
beellama's, and the mlx-kquant kernels are checked against fixtures generated
from its CPU reference. Notices are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
TurboQuant is the scheme of Zandieh, Daliri, Hadian and Mirrokni,
[arXiv:2504.19874](https://arxiv.org/abs/2504.19874), shipped by mlx-vlm under
its own `turboquant` scheme name, which gmlx does not accept.

### The MLX buffer cache at deep context

MLX keeps freed GPU buffers in a wired reuse pool. Deep-context serving of a
near-RAM-size model retains multi-gigabyte prefill transients in that pool,
and the accumulated wired memory can exhaust free pages. What follows is a
system freeze rather than an error, because the process reads the pool as
free while the kernel counts it as wired. For that reason the server always
bounds the pool and logs a `[serve] MLX cache limit:` line. When the biggest
configured model uses more than about 60% of the working set, the bound is a
quarter of the remaining slack, and otherwise it is 5% of the working set,
clamped either way to 4 to 12 GiB. A second check comes from the runtime
governor, which samples the kernel's reclaimable pages each tick.

Override it with `server.cache_limit_gb` when needed. A value pins the limit
in GiB, which benchmarks should do for reproducibility, a negative value
forces an unbounded pool, and `0` disables buffer caching. The env form is
the `GMLX_CACHE_LIMIT_GB` row in [env-vars.md](env-vars.md#runtime).
