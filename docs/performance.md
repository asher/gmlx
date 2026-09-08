# Performance

What makes a local model fast on Apple Silicon, what each gmlx lever buys, and
how to measure your own setup. It is for anyone choosing a quant, turning on
speculative decoding, or serving more than one client.

- [What determines speed](#what-determines-speed)
- [The levers](#the-levers)
- [Measuring](#measuring)
- [Reference numbers](#reference-numbers)
- [Choosing a quant for speed](#choosing-a-quant-for-speed)
- [MTP speculative decoding](#mtp-speculative-decoding)
- [The prompt cache](#the-prompt-cache)
- [Serving concurrent requests](#serving-concurrent-requests)
- [Sparse attention at depth](#sparse-attention-at-depth)
- [Memory and the KV cache](#memory-and-the-kv-cache)

## What determines speed

Single-stream decode is memory-bandwidth-bound: every token reads the model's active
weights once, so tokens per second is roughly bandwidth divided by active bytes.
That has three practical consequences. Smaller quants decode faster when nothing
else gets in the way. MoE models decode like small models (only the routed experts
are read per token) while answering like big ones. And chips with more memory
bandwidth are faster in proportion, independent of anything gmlx does.

Prefill (prompt processing) is compute-bound instead, so it rewards the GPU and
batching rather than small weights. Long-context work shifts time from weights to
the KV cache and attention. The levers below each attack one of these regimes.

## The levers

| Lever | What it buys | Cost | Where to set it |
|-------|--------------|------|-----------------|
| a uniform K-quant file | up to 64% faster decode than a mixed file of the same model | none | the file you download, see [Choosing a quant](#choosing-a-quant-for-speed) |
| speculative decoding | 1.1x to 2x decode, identical output | memory for the drafter, less lift at high concurrency | automatic on models with a head; `--draft-gguf` or `speculative:` otherwise |
| the prompt cache | skips prefill for any prefix seen before | RAM or SSD for the entries | on by default; `cache:` keys add the SSD tier |
| a quantized KV cache | halves or quarters cache memory | fewer accepted drafts with speculation on, small quality cost at 4 bits | `--kv-bits`, load key `kv_bits` |
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

`--bench-runs 3` reports the best run per length, which matters on laptops:
sustained runs throttle, and the best run keeps a thermally degraded repeat
from dragging the number. Back-to-back comparisons still hand the second arm
a hotter chip, so let the machine cool between arms you intend to compare.

The cool-box transient runs long and depends on the chassis. A rested 14-inch
M5 Max held full boost clocks for roughly twenty minutes of streamed MoE
decode before settling into a sustained rate about 20% lower, and the 16-inch
chassis holds boost longer. Book sustained numbers for anything long-running
and size warmup in minutes of decode rather than tokens. A chat-length session
on a rested machine runs at the faster rate the whole time.

Prefill throughput at a 512-token prompt is the conventional benchmark figure,
and it is a short-context number. If your real workload is a coding agent with
a 30k-token prompt, compare engines and models at that depth.

When a depth number looks wrong, check which attention kernel is running
before anything else. The route log and SDPA trace switches are in
[internals/debug-switches.md](internals/debug-switches.md); deep decode and
speculative verify should land on fused routes, and a one-shot warning fires
when a verify-shaped call does not.

## Reference numbers

Measured on a 14-inch M5 Max MacBook Pro (128 GB), 512-token prompts,
medians of repeated runs:

| Model | File | Decode | Decode (MTP) | llama.cpp decode (spec) | Prefill | llama.cpp prefill |
|-------|------|--------|--------------|-------------------------|---------|-------------------|
| gemma-4-12B-it (dense) | Q6_K | ~44 tok/s | ~72 tok/s | ~54 tok/s | ~850 tok/s | ~730 tok/s |
| Qwen3.5-9B (dense) | Q6_K | ~70 tok/s | ~112 tok/s | ~76 tok/s | ~1600 tok/s | ~1140 tok/s |

Against llama.cpp on the same GGUF, prefill is faster on every model at every
depth measured and decode is faster in every cell but one at 512 tokens,
widening as the KV cache deepens. The full tables to 200k tokens, the charts,
the methodology and the weight provenance are in
[benchmarks.md](benchmarks.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/perf/fleet-ratio-dark.svg">
  <img src="assets/perf/fleet-ratio.svg" alt="gmlx vs llama.cpp throughput speedup across the fleet">
</picture>

Your absolute numbers scale with your chip's memory bandwidth. A Pro-tier chip
has about half the bandwidth of a Max and a base M-series chip a quarter to a
fifth. The ratios between models and quants hold.

Every number and every llama.cpp comparison in these docs was measured on an
M5 Max with a 40-core GPU. The kernels target the matrix hardware in M3 and
later GPUs, and MoE prefill and expert-gather batches route through a kernel
validated on an M3 Max. M1 and M2 run the standard kernel paths and have not
been a tuning focus or benchmarked against llama.cpp. If you run the bench
commands on one, an issue with your numbers is welcome.

## Choosing a quant for speed

The file you download matters as much as any runtime flag. Community GGUFs come in
two styles: uniform files, where every quantized tensor uses one K-quant codec
(`Q6_K`, `Q4_K_M`), and mixed files (Unsloth's `UD-*` builds and similar), which
promote some tensors to Q8_0 or float to protect quality at low average bits.
(For how K-quants themselves compare with MLX's native affine quantization,
roughly half the KL divergence at equal bitrate, see
[mlx-kquant's KLD table](https://github.com/asher/mlx-kquant#why).)

Mixed files cost real decode speed. At single-stream decode, every layer's slowest
matmul gates the token, and the promoted float and Q8_0 tensors run below the
K-quant kernels' pace. In our M5 Max measurements, switching from the mixed UD build
to a uniform Q6_K of the same model sped decode up 64% on Qwen3.6-27B (dense) and
15% on Qwen3.6-35B-A3B (MoE), with equal or better output quality.

The practical rules:

- If a flat Q6_K fits your memory, prefer it over a mixed UD-Q4_K_XL: it is faster
  to decode and more accurate. The mixed file's advantage is footprint only.
- Check before downloading: `gmlx validate <ref>` lists every codec in the file.
  Prefer files that are one K-quant codec end to end.
- Reach for mixed low-bit builds when memory forces the choice, not as a default.

## MTP speculative decoding

Models that ship a native multi-token-prediction head (Qwen3.5, 3.6 and 3.8)
get [speculative decoding](glossary.md) automatically on `run` and `chat`: the
head drafts tokens ahead and the base model verifies them. Output is exactly
what the base model would have produced, faster when drafts are accepted.
`--no-mtp` turns it off. gemma-4 and Muse Glimmer take the two-file shape
instead, a small companion drafter GGUF via `--draft-gguf`. On the server it
is the `speculative:` config key. A configured companion wins over a native
head; `--native-mtp`, or the per-model `native_mtp: true`, forces the head.

Gains depend on acceptance rate and context depth. Speculation roughly doubles
dense-model decode at short context, still delivers 1.4x to 1.8x through 110k
tokens, and holds 1.2x to 1.4x at 200k. MoE models gain less, and the lift can
invert at depth on some of them, so benchmark before enabling it there. The
per-model lift curves are in [benchmarks.md](benchmarks.md). Predictable text
such as code accepts more drafts than freeform prose. Measure your own model
and workload:

```sh
gmlx run model.gguf --bench-depths "0,4096" --speculative     # accept rate + speedup
```

Speculation and batching compete for the same bandwidth. Verifying a draft
widens each request's weight reads, which is nearly free while one stream
decodes and costly once several do. The server handles this with a per-model
width cap: speculation runs while the live batch is narrow, the batch decodes
plain past the cap, and speculation resumes once it drains. The default cap
depends on the drafter and on whether the target routes experts, and the
per-model `speculative_width_cap` key overrides it; see
[server-config.md](server-config.md#speculative_width_cap). The transition
mechanics are in
[internals/speculative-batching.md](internals/speculative-batching.md).

Quantizing the KV cache shifts the target's verify logits away from the draft
head and costs accepted drafts, about a third fewer at 4 bits. Keep the KV
cache in full precision when speculation is on if you can, and prefer 8 bits
if memory forces quantization.

### DFlash 2 drafters

[DFlash 2](https://inco.ai/blog/dflash2/) is a block-diffusion drafter with
checkpoints for Qwen3.8-27B and Muse-Glimmer-30B. One drafter forward
proposes a whole block of tokens and the target verifies the block in one
pass, so a round costs one small forward plus one verify instead of one verify
per drafted token.

Pair it with `--draft-gguf`, or let `gmlx discover` do it: a DFlash 2 header
declares its base model, so discovery pairs it across directories. The drafted
depth defaults to the checkpoint's trained block, 8 on Qwen3.8 and 16 on Muse
Glimmer, and `--draft-block-size` lowers it. The drafter is single-stream, so
the server width cap is 1.

Acceptance is exact-match by default, so greedy output is token-identical to
plain decoding and sampled output follows the target's sampler.
`--stochastic-mtp` applies to DFlash 2 as well. On Qwen3.8-27B at Q6 the
drafter roughly tripled decode over plain and added half again over the native
head; the runs are in [benchmarks.md](benchmarks.md).

### Stochastic acceptance

By default a draft is accepted only when it matches the token the base model
would emit, which keeps output token-identical. At temperature above zero that
is also a ceiling: a draft cannot match a sampled token more often than the
target's own probabilities allow. `--stochastic-mtp`, or `stochastic_mtp: true`
in the server config, lifts the ceiling with rejection sampling: drafts are
sampled and accepted with probability `min(1, p/q)`. This preserves the
sampling distribution exactly, so output remains a true sample from what
plain decoding samples from, but tokens are no longer bit-identical to a
non-speculative run. Greedy requests are unaffected.

Measured gains run from a few points of acceptance on a Q6 dense model to
around 14 points on a low-bit MoE quant. The lower the trunk precision and the
flatter the text, the more exact-match leaves on the table. Turn it on when
you sample and want throughput.

## The prompt cache

The server keeps a cross-request [prompt cache](glossary.md): a request whose
prefix was seen before skips prefill for the cached span. A repeated 32k-token
prefix turns tens of seconds of prompt processing into a sub-second
time-to-first-token. This is the single biggest lever for agent workloads,
which resend a large, mostly stable system prompt every turn.

What reuse to expect depends on the model family, because recurrent state
cannot rewind:

| Family | Tier | Identical resend | Next turn | Branch or regenerate |
|--------|------|------------------|-----------|----------------------|
| dense and plain-KV MoE | block | full | full | full, at 16-token block granularity |
| GDN hybrids (Qwen3.5, 3.6) | checkpoint | all but the last token | to the turn boundary snapped to a 2048-token grid, about 90% at a 9k history | to the deepest checkpoint below the divergence |
| sliding-window models (gemma-4, gpt-oss) | checkpoint | as GDN | to within a few tokens of the divergence once the prefix clears the window | as GDN |
| pure-recurrent and CacheList (falcon-h1, deepseek4) | exact | full | full, since each turn extends the stored sequence verbatim | none; an edited history prefills cold |

Sliding-window models under `--speculative` keep no record of generated
tokens, so their next-turn reuse comes from the prefill boundaries alone and
the reply re-prefills.

The optional SSD tier (`gmlx init --disk-cache`, or the `cache:` block in the
config) persists entries across restarts and holds more than RAM comfortably
would. Entries are evicted by size budget. The keys are in
[server-config.md](server-config.md#cache-keys), and hit and store counts
surface on `GET /v1/metrics`.

Thinking templates that strip prior-turn `<think>` blocks from the re-rendered
history diverge right after the assistant header, so a full-length entry for
the reply can never match. The server keys the stored entry on the predicted
next-turn render instead, and what lies past the divergence re-prefills, as it
does on every server. That is a template property.

### What a warm hit restores

A speculative request uses the same pools the plain path uses, plus two
speculative-only layers, all on by default:

- Prefix layer. An in-memory LRU of post-prefill KV and hidden state, so a
  request sharing a prefix with an earlier one skips that prefill even with
  `cache:` off.
- Shared pools. With `cache.enabled`, the lookup ladder of exact, block and
  disk fills the prompt cache before prefill, including warm restarts from the
  SSD tier.
- Retirement. At request finish the whole sequence, prompt plus reply, is
  stored back, so the next turn of a conversation warm-starts past the whole
  of this one.
- Drafter sidecar. A native MTP head keeps its own KV, and a warm target with
  a cold drafter decodes at degraded acceptance until it catches up. A small
  sidecar entry saves the drafter's KV beside the target's so a warm hit
  restores both.
- Checkpoints. Hybrid models save restore points piecewise along a prefill and
  while generating, plus targeted ones at the end of the system prompt, one
  token before the prompt end, and at the predicted next-turn boundary. The
  system-prompt one is what lets parallel agents sharing a prompt restore
  from it. Prompt prefill on these models runs one request at a time.

The design and the triage switches are in
[internals/prompt-cache.md](internals/prompt-cache.md).

## Serving concurrent requests

The server decodes all active streams as one batch. Decode is
bandwidth-bound and the batched step reads the weights once for every
stream, so aggregate throughput rises with client count while each stream
gives up less than its proportional share. On an M5 Max with Qwen3.6-35B-A3B
Q6_K, three streams deliver 1.3x to 1.7x the aggregate of one, the ratio
narrowing as context deepens because attention work is per-stream.

What needs managing is admission: a new request's prompt must prefill while
existing streams are mid-decode. Prefill runs in 2048-token chunks, and at
depth a chunk costs hundreds of decode steps of GPU time, so a scheduler that
alternates one decode step with one chunk lets a long admission starve live
streams. When chunks are cheap, pacing only delays admission and narrows the
decode batch. Two settings cover the two symptoms:

| Symptom | Setting | Default | Effect |
|---------|---------|---------|--------|
| a starved decode batch | `server.decode_prefill_ratio` | `auto` | paces admissions only when an already-decoding stream would otherwise fall below half its batched rate; a number pins the ratio, `0` restores strict alternation |
| a stuttering stream | `server.prefill_tick_ms` | `500` | halves each chunk until its predicted wall time fits the budget; `0` for batch jobs that only care about aggregate throughput |

Paced admission bounds every waiter's time-to-first-token at twice its
unpaced prefill, and prefill runs at full speed whenever nothing is decoding,
so single-client serving is unaffected under every setting. In our serve
benchmarks a second client arriving at 14k tokens froze the live stream to 4%
of its decode rate under strict alternation; paced, it kept 80% with the
second client's time-to-first-token unchanged. Both keys are documented under
[Scheduling](server-config.md#scheduling), and both are read live so a running
server can be retuned.

Pacing decides how admissions share GPU time; the speculative width cap
decides which decode mode each batch runs in. And the prompt cache is the
strongest admission lever of all: a warm prefix skips its prefill outright,
so agent sessions that resend a cached history admit almost for free.

Concurrent streams often share a prefix, whether the same system prompt or
histories restored from the cache. The server detects the sharing from the
streams' token ids and decodes such batches through a cascade kernel that
reads the prefix once for the whole batch. Four streams on a 12k-token system
prompt decode about 1.4x faster aggregate, and the win grows with prefix
length and stream count. It is exact and on by default; its switches are the
`GMLX_CASCADE_SDPA` rows in [env-vars.md](env-vars.md#runtime).

## Sparse attention at depth

At deep context, decode attention reads the whole KV cache every token, and
past roughly 16k tokens it dominates the step. `GMLX_SPARSE_ATTN=1` switches
deep decode to top-k sparse attention: the runtime keeps a small index over
the cache and each step attends only the best-scoring pages within a fixed
token budget, plus the attention sink and the most recent pages. Attention
cost stops growing with depth.

This is lossy, which is why it is opt-in. On a Llama-3.1-8B Q6_K at 32k depth
with the default 2048-token budget, the divergence from full attention is the
same order as the Q6 quantization noise, at 1.4x single-stream decode and 1.8x
aggregate at three streams on a shared prompt. Needle lookups deep in the
context keep working.

The route engages only on architectures whose quality has been measured,
because the property it trades on is architectural: full-attention stacks
concentrate decode attention into a small key set, and the sliding-window
hybrids measured do not. gemma-4 is excluded and runs full attention whatever
the switch. The budget and engagement depth are the `GMLX_SPARSE_K` and
`GMLX_SPARSE_MIN_S` rows in [env-vars.md](env-vars.md#runtime). Quantized KV
caches and speculative verify steps always run full attention.

## Memory and the KV cache

Weights cost about the GGUF file size. The KV cache, for a standard dense model:

```text
bytes per token = 2 (K and V) x layers x kv_heads x head_dim x 2 (bf16)
```

An 8B-class model (32 layers, 8 KV heads, head dim 128) pays 128 KB per token: 4 GB
for a 32k-token session. A 32B-class dense model (64 layers) pays 256 KB per token:
8 GB at 32k. Long-context agent work can make the cache rival the weights.

Several families are much cheaper than the formula. Sliding-window layers
(gemma) stop growing at the window size. Hybrid models (Qwen3.5/3.6, Falcon-H1,
Granite 4.x, Nemotron-H) keep a small fixed state on most layers and pay full
KV only on their few attention layers: Qwen3.6-27B, with 16 attention layers
of 64, pays 2.1 GB at 32k rather than the 8.4 GB a dense 64-layer model would.
MLA models (DeepSeek-family) store a compressed cache. The capacity planner
prices all of these.

Levers, cheapest first:

- `--kv-bits 8` roughly halves the cache at nearly no quality cost, and
  `--kv-bits 4` roughly quarters it with a small cost at long range.
  `--quantized-kv-start` keeps the first stretch of context in full precision.
  Server-side these are the [load keys](server-config.md#load-keys). With
  speculation on, quantized KV also costs draft acceptance.
- `--max-kv-size` caps the cache as a rolling window, trading away the oldest
  context.
- `--prefill-step-size` shrinks the 2048-token prefill chunk to cap peak
  memory further, at some prefill-throughput cost.

macOS caps how much RAM the GPU may wire at a machine-dependent majority share
of total memory. gmlx handles the over-budget MoE case itself
([streaming.md](streaming.md)), and the server budgets resident weights with
`budget_gb`. If a single dense model plus cache sits right at the cap on a
high-RAM Mac, the ceiling can be raised at your own risk with
`sudo sysctl iogpu.wired_limit_mb=<MB>`. It resets at reboot; leave the OS
several GB of headroom.

### The MLX buffer cache at deep context

MLX keeps freed GPU buffers in a wired reuse pool. Deep-context serving of a
near-RAM-size model retains multi-gigabyte prefill transients in that pool,
and the accumulated wired footprint can exhaust free pages. The failure is a
system freeze, not an error, because the process reads the pool as free while
the kernel counts it as wired. The server therefore always bounds the pool and
logs one `[serve] MLX cache limit:` line: a quarter of the remaining slack when
the biggest configured model uses more than about 60% of the working set,
otherwise 5% of the working set, clamped to 4 to 12 GiB. The runtime governor
backs this up by sampling the kernel's reclaimable pages every tick.

Override it with `server.cache_limit_gb` when needed: a value pins the limit
in GiB, which benchmarks should do for reproducibility, a negative value
forces an unbounded pool, and `0` disables buffer caching. The env form is the
`GMLX_CACHE_LIMIT_GB` row in [env-vars.md](env-vars.md#runtime).
