# Performance

What makes a local model fast on Apple Silicon, what each gmlx lever buys, and
how to run your own benchmarks.

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

## Measuring

```sh
# prefill + decode throughput at several prompt lengths
gmlx run model.gguf --bench "128,512,2048" --bench-runs 3

# decode speed AT depth: how fast is token 16,001?
gmlx run model.gguf --bench-depths "0,4096,16384"
```

`--bench-runs 3` reports the best (max-tps) run per length, which matters on
laptops: sustained runs throttle, and picking the best run keeps a thermally
degraded repeat from dragging the number. Back-to-back A/B comparisons still
hand the second arm a hotter chip - let the machine cool between arms you
intend to compare.

The cool-box transient runs long, and it is chassis-dependent. A rested
14-inch M5 Max held full boost clocks for roughly twenty minutes of
streamed MoE decode before heat-soaking into a sustained rate about 20%
lower. The 16-inch chassis cools better and holds boost longer still.
Both regimes are real. Book sustained numbers for anything long-running,
and size A/B warmup in minutes of decode rather than tokens. A
chat-length session on a rested machine, though, genuinely runs at the
faster rate the whole time.

A note on `pp512`-style numbers: prefill throughput at a 512-token prompt is the
conventional benchmark figure, and it is a short-context number. If your real
workload is a coding agent with a 30k-token prompt, compare engines and models at
that depth, not at 512.

When a depth number looks wrong, check which attention kernel is actually
running before anything else. `GMLX_ROUTE_LOG=1` prints per-route SDPA call
counts at process exit; `GMLX_SDPA_DEBUG=1` traces the first deep calls
live. Deep decode and speculative verify should land on fused routes
(`gqa_decode`, `fa_decode`, `fa_verify`, `verify_gemm`, `sdpa_vector`). `stock`
at depth means the shape missed every eligibility gate and is paying for
materialized attention scores. A one-shot warning fires automatically when a
verify-shaped call does this. These flags work in the server process too, which
is where serve-path numbers must be measured (`GMLX_ROUND_PROFILE=1` +
`GMLX_ROUND_LOG=path` writes a per-round phase log there).

## Reference numbers

Measured on a 14-inch M5 Max MacBook Pro (128 GB), 512-token prompts,
medians of repeated runs:

| Model | File | Decode | Decode (MTP) | llama.cpp decode (spec) | Prefill | llama.cpp prefill |
|-------|------|--------|--------------|-------------------------|---------|-------------------|
| gemma-4-12B-it (dense) | Q6_K | ~44 tok/s | ~72 tok/s | ~54 tok/s | ~850 tok/s | ~730 tok/s |
| Qwen3.5-9B (dense) | Q6_K | ~70 tok/s | ~112 tok/s | ~76 tok/s | ~1600 tok/s | ~1140 tok/s |

These are the 512-token-depth medians from the July 2026 fleet round. The
per-model tables in [benchmarks.md](benchmarks.md) carry the same runs to
200k tokens with run-to-run ranges.

Against llama.cpp on the same GGUF, in the July 2026 fleet round: prefill is
faster on every model at every depth measured, 1.2-1.6x at short context and
2-4x past 100k tokens. Decode is faster in every cell but one at 512 tokens
(0.95x on gpt-oss-20b) and widens as the KV cache deepens. With speculative
decoding active on both engines, decode runs 1.1-2x ahead at every depth.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/perf/fleet-ratio-dark.svg">
  <img src="assets/perf/fleet-ratio.svg" alt="gmlx vs llama.cpp throughput speedup across the fleet">
</picture>

Per-model charts, the full tables, methodology, and exact weight provenance
are in [benchmarks.md](benchmarks.md). Your absolute numbers scale with your
chip's memory bandwidth. As a rough guide, a Pro-tier chip has about half the
bandwidth of a Max and a base M-series chip a quarter to a fifth, so scale
the table accordingly. The ratios between models and quants hold.

A scope note on hardware: every number and every llama.cpp comparison in this
guide was measured on an M5 Max (40-core GPU, 128 GB). The kernels are written
for the matrix hardware in recent Apple GPU generations (M3 and later), and
tuning is no longer M5-only. MoE prefill and expert-gather batches route
through a kernel tuned and validated on an M3 Max (128 GB), with the M5's
tensor units still taking over where the hardware has them. M1 and M2 run the
standard kernel paths but have not been a tuning focus, and we have not
benchmarked them against llama.cpp. The bandwidth scaling above transfers
gmlx's own numbers between chips. The comparative claims are measured on
M5-family hardware. If you run the bench commands on an M1 or M2, an issue
with your numbers is welcome.

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
get speculative decoding automatically on `run` and `chat`: the head drafts
tokens ahead and the base model verifies them. Output is exactly what the base
model would have produced, just faster when drafts are accepted. `--no-mtp`
turns it off. gemma-4 and Muse Glimmer take the two-file shape instead: a small
companion drafter GGUF via `--draft-gguf`. On the server it is the
`speculative:` config key. A configured companion wins over a native head;
`--native-mtp` (or the per-model `native_mtp: true` key) forces the head.

Gains depend on acceptance rate and context depth. In our serve benchmarks (M5
Max, the same server with MTP off as the baseline), speculation roughly
doubles dense-model decode at short context (1.9-2.1x on Qwen3.6-27B and, via
the companion drafter, gemma-4-12B/31B), still delivers 1.4-1.8x from 17k
through 110k, and holds 1.2-1.4x at 200k. MoE models gain less (1.1-1.3x) and
the lift can invert at depth (gemma-4-26B-A4B drops to 0.86x past 100k), so
benchmark before enabling it there. The verify pass runs on kernels built for
it, which is what keeps the speedup alive at depth. The per-model lift curves
are charted in [benchmarks.md](benchmarks.md). Predictable text (code,
structured output) accepts more drafts than freeform prose. Measure your model
and workload:

```sh
gmlx run model.gguf --bench-depths "0,4096" --speculative     # accept rate + speedup
```

One interaction to know about: speculation and batching compete for the same
bandwidth. Verifying a draft widens each request's weight reads, which is
nearly free while one stream decodes and costly once several do, so the lift
falls as concurrency rises. The server handles this for you with a per-model
batch-width cap: speculation runs while the live batch is narrow, the batch
decodes plain past the cap, and speculation resumes once it drains back
under it. A lone speculating stream likewise yields to arriving requests
instead of making them wait. The transition mechanics are in
[speculative-batching.md](speculative-batching.md).

Where the trade turns depends on the drafter and on whether the target routes
experts. A native head verified by a dense hybrid-attention target keeps
winning to the widest batch we measured, so it defaults to uncapped. A
separate-model drafter pays a full small-model forward per drafted token and
that cost grows with the batch, so the gemma assistant shape defaults to a cap
of 2 on a dense target. Routed-expert targets default to 1 and speculate only
while a single stream decodes: verification multiplies the union of experts
each drafted position touches, and both MoE families measured lose the trade
at width 2. That one is decided by inspecting the loaded model for stacked
expert layers rather than by a table of architectures, so a new MoE model
inherits it on arrival. Two drafters (hy3, deepseek4) can draft only a single
sequence, and are capped at 1 for that reason instead.

Per-model `speculative_width_cap` overrides the default, and
`GMLX_MTP_WIDTH_CAP=0` turns the cap off for a measurement run. See
[server-config.md](server-config.md#speculative_width_cap).

A second interaction: quantizing the KV cache (`--kv-bits`) shifts the target
model's verify logits away from the draft head and costs accepted drafts --
about a third fewer at 4-bit in our measurements (9B hybrid, temperature 0.6),
which can outweigh the memory saved. Keep the KV cache in full precision when
speculation is on if you can. If memory forces quantization, prefer 8-bit,
which perturbs the distribution far less.

### DFlash 2 drafters

[DFlash 2](https://inco.ai/blog/dflash2/) is a block-diffusion drafter from
Inco AI / z-lab with checkpoints for Qwen3.8-27B and Muse-Glimmer-30B (the
llama.cpp conversions are the `dflash` GGUFs with a candidate selector). One
drafter forward proposes a whole block of tokens: the drafter reads five of
the target's residual streams, denoises a block of mask tokens in a single
pass, and a selector walks one path through the top-16 candidates of every
block position. The target verifies the block in one forward, so a round costs
one small forward plus one verify instead of one verify per drafted token.

Pairing: `gmlx run target.gguf --draft-gguf <DFlash2>.gguf`, or let discovery
do it. A DFlash 2 header declares its base model, so `gmlx discover` pairs it
with that model across directories (the `<publisher>__<repo>/` layout keeps
drafter and target apart); a DFlash 2 drafter also replaces a same-directory
DFlash v1 pairing on Muse Glimmer. The drafted depth defaults to the
checkpoint's trained block (8 on Qwen3.8, 16 on Muse Glimmer);
`--draft-block-size` lowers it. The drafter is single-stream (server width
cap 1).

Acceptance is exact-match by default, so greedy output is token-identical to
plain decoding and sampled output follows the target's sampler. On the sampled
path each draft row is drawn with the target's temperature, top-p and min-p
over the selector's candidates (top-k is bounded by the 16 candidates); the
reference implementation applies temperature only. `--stochastic-mtp` applies
to DFlash 2 as well: the drafter draws each row from the sharpened proposal
and records it, and the p/q walk accepts against it (Qwen3.8 at temperature
1.0: mean accept 4.2 exact-match, 5.0 stochastic). A DFlash v1 drafter
proposes independent rows per block, which is not a proposal the walk can
accept against (forced, Muse v1 fell from 6.7 to 1.7 accepted per round), so
`--stochastic-mtp` keeps exact-match there and says so in the log.

Measured (M3 Max, greedy, a gsm8k prompt, 300 tokens): Qwen3.8-27B
UD-Q6_K_XL decodes at 14.1 tok/s plain, 31.0 tok/s on its native head (mean
accept 1.9) and 47.5 tok/s with the DFlash 2 drafter (mean accept 4.8 at
block 8, 69% of drafts accepted; a block-8 round is 107 ms of verify).
Muse-Glimmer-30B Q6_K_L goes from a mean accept of 5.1 with its DFlash v1
drafter to 6.5 with DFlash 2 (49 rounds to 40 for the same 300 tokens, about
25% more tok/s). llama.cpp's DFlash 2 implementation on the same GGUF pair
and prompt produces the identical greedy text and accepts 66.6% of 7 drafts
per round to gmlx's 68.6%.

Muse Glimmer's DFlash v1 drafter runs on the same code. Its block attends the
drafter's last `sliding_window - 1` committed positions; the mask trims the
ring's rollback slack and, once the ring is full, block row `i` drops the
oldest `i` keys exactly as the reference does.

### Stochastic acceptance (opt-in)

By default a draft is accepted only when it matches the token the base model
itself would emit - that exact-match rule is what keeps MTP output
token-identical. At temperature > 0 it is also a ceiling: however good the
draft head, a draft can't match a sampled token more often than the target's
own probabilities allow. `--stochastic-mtp` (run/chat/serve) or
`stochastic_mtp: true` in the server config lifts that ceiling with rejection
sampling: drafts are sampled and accepted with probability `min(1, p/q)`.
This provably preserves the sampling distribution - output remains a true
sample from exactly what non-speculative decoding samples from - but tokens
are no longer bit-identical to a non-speculative run. Greedy requests are
unaffected and stay token-identical.

Measured A/Bs (temp 1.0 unless noted): Qwen3.6-27B coding 73 -> 77%
acceptance; Qwen3.6-35B-A3B +2 to +12 points across coding/chat/creative
profiles; DeepSeek-V4-Flash at IQ2_XXS is the big winner at 55 -> 69% (+11%
decode throughput). The lower the trunk precision and the flatter the text,
the more exact-match leaves on the table. Long-context chat (ultrachat at 4k
depth) gains a smaller +3-5% expected tokens per round. Turn it on when you
sample and want throughput. Leave it off and output is exactly what plain
non-MTP decoding produces.

## The prompt cache

The server keeps a cross-request prompt cache: a request whose prefix was seen
before skips prefill for the cached span. A repeated 32k-token prefix turns tens of
seconds of prompt processing into a sub-second time-to-first-token. Dense-attention
models reuse at block granularity; hybrid and sliding-window models reuse
checkpoint records (under fp16 or kvarn KV alike); pure-recurrent and
CacheList archs reuse verbatim snapshots.

What reuse to expect, per family (tier routing:
[server-config.md](server-config.md#which-tier-serves-which-architecture)):

- **Dense / plain-KV MoE** (block tier): any shared prefix reuses at
  16-token block granularity - identical resends, shared system prompts,
  mid-conversation branches all hit.
- **GDN hybrids** (qwen3.5/3.6 and friends; ckpt tier): an identical
  resend restores all but the final token. The next turn of a
  conversation restores to the render-stable turn boundary snapped down
  to the 2048-token checkpoint grid - at a 9k-token history that is
  ~90% of the prefill. A branch or regenerate restores to the deepest
  interval boundary below the divergence. Recurrent state cannot rewind,
  so the un-restored tail re-prefills. That tail is bounded by the grid,
  never the whole history.
- **Sliding-window models** (gemma-4, gpt-oss; ckpt tier): same shapes
  as GDN, but turn boundaries are exact rather than grid-snapped once
  the prefix clears the attention window - next-turn restore lands
  within a few tokens of the point where the re-rendered history
  actually diverges.
- **CacheList / pure-recurrent** (falcon-h1, deepseek4 - including
  DeepSeek-V4-Flash; exact tier): verbatim-prefix snapshots. Identical
  resends reuse, and so does multi-turn chat: a finished request
  stores prompt plus reply, and the next turn's render extends that
  sequence verbatim, so each turn re-prefills only its new tokens.
  Measured on DeepSeek-V4-Flash (87 GB IQ2_XXS, ~8k-token history):
  cold prefill 44.5 s, identical resend 2.0 s, later turns 1.5-9 s,
  restart from the SSD tier 1.9 s. The boundary to respect: any edit
  to the history is a different sequence and prefills cold. There is
  no partial credit for a merely shared prefix on this tier - the same
  history with one changed line went back to 43 s.
- One stated gap: sliding-window models under `--speculative` retain no
  record of generated tokens (their post-prefill rotating layers decline
  stores by design), so next-turn reuse there comes from the prefill
  boundaries alone. The reply itself re-prefills.

This is the single biggest lever for agent workloads: coding harnesses resend a
large, mostly stable system prompt every turn, and multi-turn chat resends the whole
history. The optional SSD tier (`gmlx init --disk-cache`, or the `cache:` block in
the config) persists entries across restarts and holds more than RAM comfortably
would. Entries are evicted by size budget. Configuration keys:
[server-config.md](server-config.md#cache-keys-cache).

The cache composes with MTP, and a finished request stores its generated
tokens too. Turn N+1 of a conversation warm-starts past the whole of turn N
instead of re-prefilling the previous reply, and a warm hit restores the
draft head's state along with the base model's. Details and switches:
[server-config.md](server-config.md#speculative-decoding--the-prompt-cache).

## Serving concurrent requests

The server decodes all active streams as one batch. Decode is
bandwidth-bound and the batched step reads the weights once for every
stream, so aggregate throughput rises with client count while each stream
gives up less than its proportional share. Steady-state batched decode
measured on an M5 Max (Qwen3.6-35B-A3B Q6_K): three streams deliver 1.3-1.7x
the aggregate of one, the ratio narrowing as context deepens because
attention work is per-stream and does not amortize.

What needs managing is admission: a new request's prompt must prefill while
existing streams are mid-decode. Prefill runs in 2048-token chunks, and a
scheduler that simply alternates one decode step with one chunk lets a long
admission starve live streams, because at depth a chunk costs hundreds of
decode steps' worth of GPU time. Whether pacing admissions helps is decided
by that same quantity. When a chunk costs a live stream many decode steps
(deep context), stock scheduling starves it and pacing rescues it. When
chunks are cheap (shallow prompts, warm prefix hits), pacing only delays
admission, and a delayed admission narrows the decode batch that aggregate
throughput comes from.

`decode_prefill_ratio` (default `auto`) measures this per tick and paces
only when an already-decoding stream that was admitted before the waiters
arrived would otherwise fall below half its batched decode rate. For
simultaneous bursts (no incumbent to protect), cheap chunks, and queued
waiters held behind paced admissions past a deadline it runs stock
scheduling, so one setting serves shallow-burst and deep-second-client
load alike. Paced admission bounds every waiter's time-to-first-token at
twice its unpaced prefill, even when several arrive at once. The
deadline counts only time pacing itself is responsible for: a waiter
blocked by a full decode batch or by the memory admission gate is not
aging toward it, since running unpaced would not admit that waiter any
sooner.

A numeric value pins the static behavior: the decode batch receives that
multiple of each chunk's GPU time before the next chunk is admitted, and at
`1.0` live streams keep roughly half their throughput while a prompt is
admitted. `0` restores strict alternation. Static pacing has two costs
worth naming. A waiter's time-to-first-token stretch compounds with queue
depth, since each waiter also waits out the throttled prefill of everyone
ahead of it: several-fold at moderate bursts, not the single-admission
(1 + ratio)x. And delaying admission keeps the decode batch narrow, which
at burst concurrency can cost aggregate throughput outright. Prefill runs
at full speed whenever nothing is decoding, so single-client serving is
unaffected under every setting.

The deeper the context, the more this matters. In our serve benchmarks on
the same 35B-A3B, a second client arriving at 14k tokens under strict
alternation froze the live stream to 4 percent of its decode rate for the
whole admission; paced, it keeps 80 percent, with the second client's
time-to-first-token unchanged. At 50k tokens the admission is roughly a
minute of prefill and the live stream holds 54 percent instead of 3, a
~26x higher rate through the window. The key is `server.decode_prefill_ratio`
([server-config.md](server-config.md)), the `serve` flag is
`--decode-prefill-ratio`, and the `GMLX_DECODE_PREFILL_RATIO` env is read
per scheduler tick, so it can be changed on a live server.

Pacing bounds the average split, not the stall a single chunk inflicts:
every live stream still hitches by one full chunk whenever one is admitted,
which is 1-2 seconds at deep context and can reach tens of seconds when an
over-RAM model streams weights from disk. `server.prefill_tick_ms` (default
500, flag `--prefill-tick-ms`, env `GMLX_PREFILL_TICK_MS` read per chunk)
bounds that quantum: while streams decode, each chunk is halved until its
predicted wall time, taken from the last observed chunk cost, fits the
budget. The two knobs answer different symptoms: a starved decode batch
needs the ratio, a hitchy stream needs the tick. Smaller chunks cost
a few percent of prefill throughput per halving tier (worst on MoE), so
batch jobs that only care about aggregate throughput can set `0`.

Two interactions to know. Pacing applies to speculative (MTP) serving too,
and the two features divide the work: pacing decides how admissions share
GPU time, while the [width cap](#mtp-speculative-decoding) decides which
decode mode each batch runs in. And
the prompt cache is the strongest admission lever of all: a warm prefix
skips its prefill outright, leaving pacing to govern only the cold suffix.
Agent sessions that resend a cached history and add a few thousand tokens
admit almost for free.

Concurrent streams often share a prefix - the same system prompt, or
histories restored from the prompt cache. The batch cache holds one copy of
that prefix per stream, and a plain batched step re-reads every copy every
token. The server detects the sharing from the streams' token ids when a
batch forms or a stream is admitted (cold prompts, cache-restored
histories, and mixes all count). Such batches decode through a
shared-prefix cascade kernel that reads the prefix once for the whole
batch, so attention traffic per step drops from every stream's full
context to one prefix plus each stream's own suffix. Four streams on a
12k-token system prompt decode about 1.4x faster aggregate. The win grows
with prefix length and stream count, and speculative-verify rounds
cascade the same way (their re-reads are wider, so the saving is larger).
It is on by default and exact - same numbers as the plain step.
`GMLX_CASCADE_SDPA=0` disables it; `GMLX_CASCADE_MIN_P` (default `1024`)
sets the smallest shared prefix worth routing.

## Sparse attention at depth (opt-in)

At deep context, decode attention reads the whole KV cache every token, and
past roughly 16k tokens it comes to dominate the step. `GMLX_SPARSE_ATTN=1`
switches deep decode to top-k sparse attention: the runtime keeps a small
index over the cache (one mean key per 32-token page, maintained as the
cache grows) and each step attends only the best-scoring pages within a
fixed token budget, plus the attention sink and the most recent pages, which
are always kept. Attention cost stops growing with depth.

This is lossy, which is why it is opt-in. Measured on a Llama-3.1-8B Q6_K
at 32k depth with the default 2048-token budget: mean KL divergence against
full attention of 0.008, the same order as the quantization noise of a Q6
checkpoint, at 1.40x end-to-end single-stream decode. At three streams on a
26k shared prompt it runs 1.8x aggregate over the exact cascade route. The
index is good at finding the pages a query actually needs - needle lookups
deep in the context keep working - and the always-resident sink and recency
pages keep the failure mode graceful when it is not.

`GMLX_SPARSE_K` sets the budget (default `2048` tokens: larger tracks full
attention closer, smaller is faster), and `GMLX_SPARSE_MIN_S` (default
`8192`) sets the depth where the route engages - below it full attention is
already cheap and the route stays off. Applies to fp16/bf16 KV caches at
decode width; quantized-KV caches and speculative verify steps run full
attention.

The route only engages on architectures whose quality has been measured,
because the property it trades on is architectural: pure full-attention
stacks (llama-family) concentrate decode attention into a small key set,
while measured SWA hybrids do not. On gemma-4's global layers, even an
exact top-k oracle at the default budget lands an order of magnitude
outside the acceptable divergence band, so gemma-4 is deliberately
excluded and runs full attention regardless of the switch.

## Memory and the KV cache

Weights cost about the GGUF file size. The KV cache, for a standard dense model:

```text
bytes per token = 2 (K and V) x layers x kv_heads x head_dim x 2 (bf16)
```

An 8B-class model (32 layers, 8 KV heads, head dim 128) pays 128 KB per token: 4 GB
for a 32k-token session. A 32B-class dense model (64 layers) pays 256 KB per token:
8 GB at 32k. Long-context agent work can make the cache rival the weights.

Levers, cheapest first:

- KV quantization. `--kv-bits 8` roughly halves the cache at nearly no
  quality cost; `--kv-quant-scheme kvarn` holds that fidelity at 6 bits in
  about half the fp16 cache and stays usable down to 4. [KV cache
  quantization](#kv-cache-quantization) below says what each scheme does,
  which models gain from it, and what the fidelity data shows. With
  speculative decoding on, quantized KV also costs draft acceptance; see
  the interaction note in the speculation section above.
- `--max-kv-size` caps the cache as a rolling window, trading away the oldest
  context. On `run` and `chat` the window quantizes under
  `--kv-quant-scheme kvarn` (the cap must clear the kvarn floor, see
  [cli.md](cli.md#generation)); plain `--kv-bits` cannot quantize a rotating
  window and is refused at start. The server's `max_kv_size` only caps the
  request context budget and builds no rotating cache.
- Long prompts prefill in chunks automatically (2048 tokens), which bounds
  prefill's working memory on top of the cache itself. `--prefill-step-size`
  (on `serve`, `run`, and `chat`; server config `server.prefill_step_size`, or
  a `PREFILL_STEP_SIZE` env) shrinks the chunk to cap peak memory further, at
  some prefill-throughput cost. Applies to speculative (MTP) serving too.

Several families are much cheaper than the formula. Sliding-window layers (gemma)
stop growing at the window size. Hybrid linear-attention models (Qwen3.5/3.6,
Falcon-H1, Granite 4.x, Nemotron-H) keep a small fixed state on most layers and pay
full KV only on their few full-attention layers, which is why a Qwen3.6 at 64k
context is unremarkable on a 64 GB machine. MLA models (DeepSeek-family) store a
compressed cache.

macOS also caps how much RAM the GPU may wire, at a machine-dependent majority
share of total memory. gmlx handles the over-budget MoE case itself
([streaming.md](streaming.md)), and the multi-model server budgets resident weights to a configurable
share of RAM (`--budget-gb`). If a single dense model plus cache sits right at the
cap on a high-RAM Mac, the ceiling can be raised at your own risk with
`sudo sysctl iogpu.wired_limit_mb=<MB>`. It resets at reboot; leave the OS several
GB of headroom.

### KV cache quantization

Two schemes shrink the cache. One policy decides both layer by layer, prints
one `[kv]` line, and reports the result as `kv_quant` on `/v1/models`.

**`--kv-bits N`** (server `kv_bits`) is the affine scheme, mlx-lm's
`QuantizedKVCache`. Each token's K and V rows split into groups of
`--kv-group-size` values (default 64), and each group stores N-bit codes
plus an fp16 scale and bias. Widths 2, 3, 4, 6 and 8. `--quantized-kv-start`
keeps the first stretch of the context fp16.

**`--kv-quant-scheme kvarn`** (server `kv_quant_scheme: kvarn`) is
variance-normalized quantization; `--kv-bits` picks the width (default 6;
2, 3, 4, 5, 6 and 8; split key and value widths through
`GMLX_KVARN_BITS=k6v5`). Every 128-dim slice of a head is Hadamard-rotated,
which spreads outlier channels over the whole slice. K and V are stored in
128-token records. Before rounding, each record is scaled along both axes
(16 alternating row and column normalizations in log space, a Sinkhorn
iteration) so that no token and no channel dominates the code range; three
fp16 axis vectors per record undo the scaling on read. The first 128
tokens (the attention sink) and the newest `--kv-tail-tokens` (default
1024) stay fp16, and a record seals once the tail has moved past it. Decode
and MTP verify read the records in the mlx-kquant kernels and merge the
fp16 tail through one softmax; the prompt cache stores records on its
exact and checkpoint tiers. The scheme needs an mlx-kquant build with the
kvarn ops; without one it drops with a printed reason and the model runs
fp16 KV.

**Which layers.** The rule is by cache shape, not model name. Growing
attention KV quantizes, minus the last layer of a deep stack, which stays
fp16 under either scheme. Recurrent state and sliding windows stay fp16 (a
`--max-kv-size` window on `run` and `chat` is the exception: kvarn
quantizes it, affine refuses it). kvarn takes head_dim 128, 256 and 512
only, so head_dim-64 layers (gpt-oss, falcon-h1) are affine-only. MLA archs
(deepseek4, glm5_next, kimi-k3) keep K and V in one latent store: kvarn
declines them, affine packs the pooled latents at rest. The VLM media path
keeps fp16. A declined model prints the reason and runs fp16.

**When it pays.** Quantization buys memory in proportion to how much of the
cache grows with context, and costs fidelity in proportion to how many
layers it touches. The cache shape decides both:

| cache shape | families | fp16 cache at 32k | verdict |
|---|---|---|---|
| dense full attention on every layer | llama, Mistral, Qwen3 dense | 4 to 8 GB for an 8B to 32B model | the case the schemes exist for: `--kv-bits 8`, or kvarn at 6 (its default) for the same fidelity in less memory, kvarn at 4 when memory is the constraint |
| hybrid recurrent, one attention layer in four | Qwen3.5/3.6/3.8, Nemotron-H, Falcon-H1, Granite 4 | about 2 GB at 27B, plus a fixed recurrent state | worth it only when the context budget is the constraint, 64k and up; the fidelity cost is small too, since three layers in four never quantize |
| sliding-window mix | gemma-4 | the window layers stop growing at the window | modest: only the global layers quantize |
| MLA latent | DeepSeek-V4, GLM-5.3, Kimi-K2 and K3 | already compressed by the architecture | kvarn declines; affine packs the latent pools at rest and is the only lever |
| head_dim 64 | gpt-oss | small per token | affine only |

`/status` reports the live cache size and `/v1/estimate` prices a load in
advance ([server-config.md](server-config.md)).

**Which width.** The measure is teacher-forced logit KLD against an fp16
cache on wikitext, from `scripts/kld_harness.py`, on two legs: prefill
(chunked prefill logits) and decode (token by token from full prefill
depth). KLD in nats, lower is better. The median is the typical position;
the decode p99 is the worst hundredth, where a quantizer's outliers show;
decode top-1 is the share of generated positions whose argmax matches the
fp16 cache.

<!-- kld-tables -->
Qwen3.5-9B Q4_K_M, 16k context (head_dim 256, 7 of 32 layers quantized):

| cache | prefill median | decode median | decode p99 | decode top-1 |
|---|---|---|---|---|
| affine 2 | 0.02778 | 0.02745 | 0.5596 | 89.0% |
| kvarn 2 | 0.01503 | 0.00612 | 0.2301 | 94.7% |
| turboquant 2 | 0.01901 | 0.02094 | 0.4071 | 91.2% |
| affine 3 | 0.00648 | 0.00606 | 0.1057 | 94.3% |
| kvarn 3 | 0.00291 | 0.00139 | 0.0313 | 97.4% |
| turboquant 3 | 0.00464 | 0.00440 | 0.0926 | 96.0% |
| turboquant 3.5 | 0.00340 | 0.00351 | 0.0801 | 96.3% |
| affine 4 | 0.00190 | 0.00183 | 0.0276 | 96.9% |
| kvarn 4 | 0.00117 | 0.00060 | 0.0071 | 98.3% |
| turboquant 4 | 0.00175 | 0.00163 | 0.0265 | 97.8% |
| kvarn 5 | 0.00055 | 0.00029 | 0.0042 | 98.2% |
| kvarn k6 v5 | 0.00046 | 0.00028 | 0.0039 | 98.6% |
| affine 6 | 0.00046 | 0.00038 | 0.0045 | 98.7% |
| kvarn 6 | 0.00036 | 0.00027 | 0.0036 | 98.7% |
| turboquant 6 | 0.00058 | 0.00058 | 0.0060 | 98.7% |
| affine 8 | 0.00029 | 0.00020 | 0.0027 | 98.7% |
| kvarn 8 | 0.00027 | 0.00020 | 0.0030 | 99.2% |
| turboquant 8 | 0.00032 | 0.00024 | 0.0038 | 98.7% |

Qwen3.8-27B Q6_K_XL, 16k context (head_dim 256, 15 of 65 layers quantized):

| cache | prefill median | decode median | decode p99 | decode top-1 |
|---|---|---|---|---|
| affine 2 | 0.01975 | 0.02314 | 0.4697 | 90.3% |
| kvarn 2 | 0.01009 | 0.00491 | 0.1280 | 95.1% |
| affine 3 | 0.00383 | 0.00419 | 0.1034 | 96.1% |
| kvarn 3 | 0.00212 | 0.00113 | 0.0318 | 97.3% |
| affine 4 | 0.00138 | 0.00136 | 0.0268 | 97.5% |
| kvarn 4 | 0.00084 | 0.00045 | 0.0078 | 97.4% |
| kvarn 5 | 0.00041 | 0.00025 | 0.0039 | 98.4% |
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

Qwen3.6-27B Q4_K_S, 32k context (head_dim 256, 15 of 64 layers quantized):

| cache | prefill median | decode median | decode p99 | decode top-1 |
|---|---|---|---|---|
| affine 4 | 0.00158 | 0.00203 | 0.0235 | 97.6% |
| kvarn 4 | 0.00101 | 0.00081 | 0.0095 | 98.5% |
| affine 6 | 0.00041 | 0.00059 | 0.0072 | 98.7% |
| kvarn 6 | 0.00033 | 0.00032 | 0.0038 | 98.5% |
| affine 8 | 0.00028 | 0.00034 | 0.0033 | 99.2% |
| kvarn 8 | 0.00027 | 0.00031 | 0.0047 | 98.9% |

Nemotron-3.5-Lightning-30B-A3B, 16k context (Mamba2 hybrid, head_dim 128):

| cache | prefill median | decode median | decode p99 | decode top-1 |
|---|---|---|---|---|
| kvarn 4 | 0.00270 | 0.00163 | 0.0508 | 98.1% |
| kvarn 6 | 0.00125 | 0.00103 | 0.0266 | 98.8% |
| affine 8 | 0.00123 | 0.00094 | 0.0335 | 98.2% |
| kvarn 8 | 0.00111 | 0.00095 | 0.0298 | 98.6% |
<!-- /kld-tables -->

Reading the tables: at a matched width kvarn beats the affine cache on both
legs at every width below 8, by 2 to 4x on the decode-leg median at 2 to 4
bits, and the two converge at 8. 6-bit kvarn sits between the affine 6-
and 8-bit caches on the 9B and level with affine 8 on the 27Bs, in three
quarters of the 8-bit record: at 32k its decode median can trail affine 8
by a few percent while its p99 and top-1 stay ahead, so the two split the
typical position and the outliers between them. Widths 2 and 3 are for
experiments. The decode leg is the one a long generation accumulates.

Choosing by these numbers. Top-1 is the measure closest to what a greedy
or low-temperature user sees: it is the share of tokens that come out
identical. Median KLD measures how far the whole next-token distribution
moved, which is what sampling at temperature draws from, and it keeps
scoring positions whose argmax never flipped. The p99 is the outlier
budget: one badly wrong position can turn a reasoning chain or a tool call,
and a long generation feeds its own errors back in, so a cache with a lower
p99 drifts less over a thousand tokens even when its median is not the
lowest. When two caches disagree by a few percent on one measure, take the
one with the lower p99 and the higher top-1; when a width buys a large
median gain (2x or more), that gain shows up in every measure. Note that
the ranking does not follow width across schemes: kvarn at 2 bits lands
within a point of affine at 3 on decode top-1 (ahead on the 9B, behind on
Qwen3.8-27B) at a quarter of its median KLD, and kvarn at 4 cuts affine
4's decode median to a third while matching or beating it on top-1. The corpus is wikitext under
teacher forcing, so the tables rank caches against each other; they do not
predict a task score.
TurboQuant is mlx-vlm's scheme (see Origins); gmlx does not enable it, and
it appears here for comparison only.

**What it costs in speed.** Decode cost tracks how much of the step the KV
read is: on GDN hybrids and gemma-4 all three arms sit within run-to-run
spread, while on a KV-bound dense stack (Qwen3-0.6B Q8, 27 of 28 layers)
kvarn6 decode runs 0.81x fp16 and 0.69x kv8 at 16k, 0.98x and 0.75x at 32k.
Prefill is within 10% of both arms. Native MTP composes at batch size 1:
verify rounds (query width block + 1, up to 8) attend the records on the
matrix-unit verify kernels at about a decode step's cost (D=256, 8 queries
per KV head, 16k: 1.0 ms per layer and round against 0.9 for decode);
`GMLX_KVARN_FA=0` keeps verify on the vector decode kernel for an A/B. Pick
kvarn for memory and fidelity, plain `--kv-bits` for peak decode speed on a
KV-bound dense model. Kill switches and the split-width env are in the
[cli.md env table](cli.md#environment-variables).

**Origins.** KVarN is the method of Muller, Bich, Boretti, Chang, Zhuang
and Cavigelli at Huawei: "KVarN: Variance-Normalized KV-Cache Quantization
Mitigates Error Accumulation in Reasoning Tasks",
[arXiv:2606.03458](https://arxiv.org/abs/2606.03458), with a reference vLLM
implementation at [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN)
(Apache-2.0; no code from it is used here). gmlx's cache is an MLX
implementation that follows the record format of
[beellama.cpp](https://github.com/Anbeeld/beellama.cpp) (Anbeeld, MIT), the
llama.cpp fork that first brought the method to GGUF inference: the
`kvarnN` width names, the fp16 precision tail and `--kv-tail-tokens` are
beellama's, and the mlx-kquant kernels are held to its numerics against
fixtures generated from its CPU reference. Notices:
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). TurboQuant is the
scheme of Zandieh, Daliri, Hadian and Mirrokni
([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)), shipped by mlx-vlm
as `kv_quant_scheme: turboquant`.

### The MLX buffer cache at deep context

MLX keeps freed GPU buffers in a wired reuse pool (the buffer cache). That is
normally free performance, but deep-context serving of a near-RAM-size model
retains multi-gigabyte prefill transients in the pool, and the accumulated
wired footprint can exhaust free pages. The failure is a system freeze, not
a clean error. The server therefore always bounds the cache and logs one
`[serve] MLX cache limit: ...` line: when the biggest configured model uses
more than ~60% of the GPU working set the cap is a quarter of the remaining
slack, otherwise 5% of the working set, clamped to 4-12 GiB either way.
MLX's own default is the memory limit (1.5x the working set), and the cache
is wired: the process reads it as free while the kernel counts it against
its free pages, which is how a small model at long context can still walk
the box into a freeze. The runtime governor backs this up by sampling the
kernel's reclaimable pages every tick and going red below a floor
(`GMLX_GOV_KERNEL_FLOOR_GB`); see [cli.md](cli.md#environment-variables).

Override it explicitly when needed: the `server.cache_limit_gb` config key or
the `GMLX_CACHE_LIMIT_GB` env (env wins). A GiB value pins the limit
(benchmarks should pin it for reproducibility); a negative value (or env
`off`/`none`/`unlimited`) forces an unbounded cache and suppresses the auto
policy; `0` disables buffer caching entirely. A bounded cache trades a little
allocator churn for a bounded footprint: transients up to the limit are
recycled in place, larger ones fall through to fresh allocations.

## Bigger than memory: MoE offload

MoE models whose files exceed what the GPU can wire still run, on one of
two placements (`--stream-experts`, `--stream-cpu`), with feeder paths
that keep the disk ahead of the compute and a set of levers - lossless
and lossy - specific to streamed decode. The whole subject has its own
guide: [streaming.md](streaming.md).
