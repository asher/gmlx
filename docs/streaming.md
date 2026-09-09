# Streaming MoE models bigger than memory

This guide is for running a mixture-of-experts model whose file exceeds what
your Mac can hold in memory. It covers the quick start, the two placements,
the fit rule, the lossless settings and the lossy ones.

- [What to expect](#what-to-expect)
- [Quick start](#quick-start)
- [Choosing a placement](#choosing-a-placement)
- [How big a model can this machine
  stream](#how-big-a-model-can-this-machine-stream)
- [The lossless settings](#the-lossless-settings)
- [The lossy settings](#the-lossy-settings)
- [Serving a streamed model](#serving-a-streamed-model)
- [Residency of a streamed model](#residency-of-a-streamed-model)

## What to expect

A [MoE](glossary.md) model activates only a few experts for each token,
which makes the working set of a token a small fraction of the file. gmlx
keeps the parts that all tokens read in memory and streams the routed
experts from disk, which is what lets a 200B-class model run on a 64 GB
machine.

Decode is then bound by the SSD and the CPU rather than the GPU, and
single-digit tokens per second is normal. A model that fits in memory runs
several times faster on the normal GPU path, so streaming is a capacity
feature for the over-budget case, not a speed feature.

Keep the GGUF on the internal SSD, because misses are served at the drive's
random read latency. An external drive works, but decode is slower in
proportion to its latency.

Measured numbers in this guide come from a 14-inch M5 Max MacBook Pro with
128 GB, unless another machine is named. The hardware note under
[Reference numbers](performance.md#reference-numbers) covers other chips.

## Quick start

```sh
# for sharded files, point at the first shard
gmlx run GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf --stream-experts
```

On the server, the placement is the `stream` key of the model entry:

```yaml
models:
  glm:
    path: ~/models/GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf
    stream: experts
    overrides: {load: {kv_bits: 8}}
```

The load maps the file instead of reading it, so generation starts within
seconds whatever the size. The first few dozen tokens are slower while the
expert arena, the wired region that holds the most used experts, fills from
disk. The decode feeder's exit line, printed by `run` and `chat` under `-v`
and always in server logs, shows the hit rate a session reached.

`gmlx validate <file>` says whether this Mac can stream a given file and how
big its arena would be. A quantized KV cache, `--kv-bits 8`, is the usual
addition at long context.

What composes with streaming:

| Combination | Result |
|-------------|--------|
| `--stream-experts` with `--mmproj` | the text tower streams, the vision tower stays on the GPU |
| `--stream-cpu` with `--mmproj` | refused, because the CPU placement would move the vision tower too |
| `--stream-experts` with speculative decoding | composes on the CLI with an explicit `--speculative`. Auto-MTP stays off under streaming |
| `stream:` on a `speculative:` server entry | refused at load. The server loads the drafter after the placement and would leave it unplaced |
| a `--moe-*` lossy setting with speculative decoding | under auto-MTP the setting applies and decode stays plain. With an explicit `--speculative` the command errors |

## Choosing a placement

| Placement | What stays on the GPU | What streams | Use it when |
|-----------|-----------------------|--------------|-------------|
| `--stream-experts`, or `stream: experts` | attention, routers, shared experts, the KV cache | the routed experts, served from the arena and the disk | the default choice, for long context, chat and serving beside other models |
| `--stream-cpu`, or `stream: cpu` | nothing, because the whole model runs on the CPU device from the page cache | everything past the wired budget | a single-model setup where a single device for everything is simpler |

With the decode feeder on, `--stream-experts` matches `--stream-cpu` on
short generations and pulls ahead once the arena has filled. On a server,
`stream: cpu` moves the whole process to the CPU device, so it does not mix
with GPU-resident models.

## How big a model can this machine stream

The file size does not set the limit. The every-token weights do.

A MoE GGUF holds two kinds of tensors. The routed experts are read a few at
a time for each token and stream from disk, so their size sets the decode
speed but not the fit. The every-token weights, which are attention, shared
experts, dense layers, routers, norms, embeddings and the output head, are
read by all tokens and stay resident.

The memory [governor](glossary.md) enforces a ceiling on tracked memory,
which is Metal's recommended working set less 5%, and it never comes closer
to physical RAM than a reserve of 8 GB or 10% of RAM, whichever is larger.
Four things share it, in this order:

1. The every-token weights.
2. The KV room. It holds the KV cache for 32768 tokens by default, capped at
   the trained context, plus a prefill transient and an admission reserve of
   2 GB or 5% of the working set each.
3. The prefill ring. Two copies of the largest layer's expert stacks, sized by
   the model. When it does not fit, prefill falls back to page-cache prefetch
   and the load says so. Decode is unaffected.
4. The decode arena, which takes what is left after a host floor of 5% of RAM,
   at least 4 GB, plus 2.5 GB kept for the page cache. The floor keeps the
   rest of the machine out of swap. A larger arena means a higher hit rate and
   faster decode. Below 1 GB the decode feeder does not start and decode runs
   from the page cache.

A machine therefore streams a model when the every-token weights plus the
KV room fit under the ceiling. Everything else only changes the speed.

### The ceiling by machine size

Default working sets are two thirds of RAM below 36 GB and three quarters
from 36 GB up, and sizes are decimal GB. The KV room floor is the transient
plus the admission reserve with no KV cache in it yet.

| RAM | Working set | Ceiling | KV room floor | Left for weights and KV cache |
|---|---|---|---|---|
| 16 | 11.5 | 9.2 | 4.0 | 5.2 |
| 24 | 17.2 | 16.3 | 4.0 | 12.3 |
| 32 | 22.9 | 21.8 | 4.0 | 17.8 |
| 36 | 29.0 | 27.5 | 4.0 | 23.5 |
| 48 | 38.7 | 36.7 | 4.0 | 32.7 |
| 64 | 51.5 | 49.0 | 5.2 | 43.8 |
| 96 | 77.3 | 73.4 | 7.7 | 65.7 |
| 128 | 103.1 | 97.9 | 10.3 | 87.6 |
| 192 | 154.6 | 146.9 | 15.5 | 131.4 |
| 256 | 206.2 | 195.9 | 20.6 | 175.2 |
| 512 | 412.3 | 391.7 | 41.2 | 350.5 |

`sudo sysctl iogpu.wired_limit_mb=<MB>` raises the working set, although the
reserve still keeps 8 GB or 10% unavailable.

### A worked example

Kimi-K3 UD-Q2_K_XL is an 861 GB file whose every-token weights are 62.2 GB:

| Group | GB |
|---|---|
| attention (MLA) | 31.8 |
| shared experts, 2 per layer | 12.9 |
| dense ffn and routers | 8.2 |
| recurrent (KDA) layers | 6.9 |
| embeddings and output head | 2.4 |

The routed experts are 799 GB, 896 experts in each of 92 layers with 16
read for each token. On a default 128 GB machine the ceiling is 97.9 GB and
the KV room 11.7 GB, leaving 24 GB, of which the 22.7 GB ring fits and the
host floor takes the rest, so there is no arena. The model streams, but
decode reads each expert through the page cache. On a 512 GB machine the
arena is 234 GB, or 29% of the experts.

The quant of the experts does not change the fit. UD-Q4_K_XL is a 1.5 TB
file with the same 62.2 GB of every-token weights, so it streams on the same
machines, although it decodes slower because the arena holds a smaller share
of a larger expert set. UD-Q8_K_XL has 114.7 GB of every-token weights and
fits under no ceiling below 192 GB.

### What changes the limit

- The bits of the every-token tensors. A quant that keeps attention and the
  shared experts at Q8 doubles the resident set. Pick a quant with smaller
  non-expert tensors before you pick a smaller expert quant.
- The working set, raised with `iogpu.wired_limit_mb`.
- The KV room. A smaller room gives a larger arena and a shorter safe context.
  `--kv-bits 8` halves the KV cache part of it. The room's token count is the
  `GMLX_STREAM_KV_CTX` row in [env-vars.md](env-vars.md#runtime).
- Other resident models and other processes. The arena is sized from the RAM
  reclaimable at load, shrinks by a quarter when free RAM later falls under
  the governor's floor and grows again when the RAM is freed.

### The plan gmlx validate prints

`gmlx validate <ref>` prints the plan for a MoE file, local or on Hugging
Face with a header read for each shard:

```text
  streaming: every-token weights 62.2 GB, routed experts 799.1 GB (92 layers, 896 experts, 16 per token), prefill ring 22.7 GB
    every-token by group: attention 31.8 GB, shared experts 12.9 GB, dense ffn and routers 8.2 GB, ...
    this Mac: 128 GB RAM, ceiling 97.9 GB, KV room 11.7 GB at 32768 tokens, host floor 9.6 GB
    => streams, but no decode arena (0.0 GB left after the ring and the host floor), decode runs from the page cache; expect slow decode
```

`gmlx validate --json` carries the same numbers under `stream`, and
`gmlx doctor` adds a clause for each streamed entry to its memory row. A
load prints the live budget as `[stream] memory budget:`, and because that
line includes the reclaimable-RAM clamp its arena can be smaller than the
plan's.

## The lossless settings

These are on by default on all streamed models and change no output. Each
has a flag or env switch for measurement, listed in
[env-vars.md](env-vars.md#runtime).

| Setting | What it does | Off switch |
|-------|--------------|-----------|
| prefill feeder | stages each layer's experts straight from the GGUF into GPU-visible ring slots while the previous layer computes, so each byte is read once | `--no-prefill-feeder` |
| decode feeder | keeps the most routed experts of each layer in the wired arena and reads only the misses from disk at SSD queue depth | `--no-decode-feeder` |
| lookahead prestage | runs the next layer's router early and pre-reads its predicted misses while the current layer computes. It moves bytes only, never routing | `GMLX_DECODE_LOOKAHEAD=0` |
| weight pin | locks the every-token weights in memory so the kernel cannot evict them between tokens on a machine at its free-page floor | `GMLX_PIN_WEIGHTS=0` |
| GPU keep-warm | keeps GPU clocks high through the host and disk gaps between layers with a tiny heartbeat kernel | `--gpu-keepwarm` off, or `GMLX_GPU_KEEPWARM=0` |
| streamable lookup tables | on architectures with a large table in each layer that all tokens read a few rows of, streams the table before the experts | `GMLX_STREAM_PLE=0` |

The prefill feeder stages only the experts the router chose on short prompts,
which is the source of its time-to-first-token gain. Its ring reads bypass
the page cache, since a pass reads each routed expert once and would otherwise
evict the rest of the machine's page cache for pages it never reads again.

The decode feeder's arena starts empty and converges within a few dozen
tokens. Under memory pressure from another model or a build it shrinks,
keeping its most routed experts, and grows again when the pressure ends.
Multi-token expert calls whose routed set exceeds the arena, such as the
next chat turn's prefill, are split along the token axis and served from the
arena rather than falling back to a page-cache gather.

Lookahead prestage keeps its reads off the demand path: they are submitted
only after the current layer's demand misses finish and run at utility disk
priority. A gate in each layer stops submitting prediction ranks that
measure unreliable, and predictions the router then does not route to are
cancelled before they reach the disk.

GPU keep-warm matters because streamed decode alternates sub-millisecond
GPU bursts with host and disk gaps, and the GPU drops to idle clocks in each
gap. The heartbeat costs power only while decoding and stops after a second
of inactivity. Turn it off on battery. It gains nothing on a model that
fits in RAM.

Weight pinning is skipped, with a printed reason, when the every-token set
would exceed 60% of RAM. A tensor the loader converts at load is left out
of the pin, since the runtime reads a converted copy instead of the file
bytes.

Two more properties matter to operators. Because a fork of a process
holding a wired arena would copy it, the arena, ring and pinned weights are
marked not to be inherited, so spawn nothing from a streaming server. When a
larger-than-RAM model is released, at exit or at unload on a running server,
its page cache is released with it.

Native-fp expert tensors, MXFP4 and NVFP4 as in gpt-oss and DeepSeek-V4-Flash
Q4_K_XL quants, are streamed in the same way. Over budget they are served as
zero-copy file bytes instead of being repacked at load, which is what makes
the over-RAM case load in seconds instead of minutes. The `GMLX_NATIVE_FP` row
in [env-vars.md](env-vars.md#runtime) forces either layout.

Measurements of each setting, including the models and machines they were
taken on, are in
[internals/streaming-measurements.md](internals/streaming-measurements.md#lossless-setting-measurements).

## The lossy settings

Four settings trade a bounded amount of output quality for decode speed.
None is on by default, so absent flags and absent config keys mean lossless
routing. All of them act on decode only and on streamed layers only.

A streamed decode token has three costs. Experts that miss the arena are
read from disk at demand latency, which dominates when the hit rate is low,
while those already in the arena cost only a small gather. Each streamed MoE
layer also has a fixed overhead of kernel launches and a host sync that does
not shrink when fewer experts are routed. The settings each reduce a
different cost.

| Setting | Flag | Config key | Reduces | Notes |
|-------|------|------------|------|-------|
| expert cap | `--moe-experts K` | `moe_experts: K` | reads and compute | fixed K experts per token |
| expert mass | `--moe-expert-mass P` | `moe_expert_mass: P` | reads and compute | keeps the smallest set covering share P of the gate mass. Dropped mass is bounded by 1-P and is concentrated on confident tokens |
| miss shed | `--moe-miss-shed P` | `moe_miss_shed: P` | disk stalls | drops only experts that would miss the arena, lowest scores first, keeping share P. Needs the decode feeder |
| keeper prestage | `--moe-prestage keepers` | `moe_prestage: keepers` | residual stalls | applies the miss-shed policy a layer earlier in lookahead. Has no quality parameter and needs miss shed |
| layer shed | `--moe-layer-shed P` | `moe_layer_shed: P` | per-layer overhead | skips a layer's routed experts with probability P on each token. The shared expert still runs |

The two router-side settings combine, so `--moe-experts 6 --moe-expert-mass 0.9`
caps at 6, then drops within the 6. How much expert mass saves is a property
of the router: on a concentrated router most reads disappear for a few
percent of dropped mass, while on a flat router it saves almost nothing.
Measure instead of guessing. `--moe-expert-probe` runs the trained routing
losslessly and prints, for each candidate P, the experts kept and the mass
dropped, decode and prefill separately, and P should be sized against the
decode column. The probe is CLI-only, so run it once before fixing a value
in a config.

Which setting to use depends on a measurement:

1. Read the arena hit rate from the decode feeder's exit line.
2. At a low hit rate, miss shed is the best choice whatever the router's
   distribution. It costs quality only on calls that would stall.
3. At a high hit rate, a concentrated router favors expert mass. A flat
   router leaves only the layer overhead, which only layer shed touches.
4. Before accepting the quality cost of layer shed, confirm keep-warm is on,
   since clock ramp is a large share of that overhead.

Settings that passed on the models they were measured on:

| Model | Setting | Quality verdict |
|-------|---------|-----------------|
| Hy3 IQ4_XS | `moe_layer_shed: 0.10` with `moe_miss_shed: 0.90` | no defects at temperature 0.6. Lower to 0.07 and 0.93 at the card's temperature of 0.9 |
| MiniMax-M3 Q4_K_M | `moe_miss_shed: 0.80` | no defects over two 10k-token generations |
| GLM-5.2 UD-IQ3_XXS | `moe_miss_shed: 0.85` | no defects. 0.80 dropped scene content while keeping valid form |
| Kimi-K3 UD-Q2_K_XL | `moe_miss_shed: 0.65` to `0.80` | working pages throughout, with content drift growing as P falls. 0.60 broke code |

Quality degrades in a consistent order as settings tighten. Multi-step
arithmetic breaks first, well before coherence or code, combined settings
add their losses at the same threshold, and dropped mass degrades content
before form. Certify a setting by rendering its output beside a lossless run
at the same seed and your deployment sampling. The case studies, sample
galleries and the certification procedure are in
[internals/streaming-measurements.md](internals/streaming-measurements.md).

## Serving a streamed model

Send a single request at a time to a server that streams a model. The
wired-memory refresh, lookahead and the GPU-side token path all switch off
when a decode step holds more than one token, so a second concurrent request
puts both on the slow path and the arena's hit rate falls. Prefill is
unaffected.

In a config the feeder opt-outs are `prefill_feeder: false` and
`decode_feeder: false` beside the `stream` key. The lossy settings are the
model keys in the settings table. All of them are listed under
[models](server-config.md#models).

## Residency of a streamed model

A streamed entry counts against the budget as its every-token weights plus
its arena and its prefill ring, while the routed experts stay on disk.
Because the arena fills what the ceiling leaves, a streamed model alone can
use the whole budget, so to keep a second model resident beside it, cap the
arena with the `GMLX_DECODE_ARENA_GB` row in
[env-vars.md](env-vars.md#runtime) so that both fit `budget_gb`. The
streamed load lowers the wired limit for the rest of the process, and a
resident dense model runs unwired from then on.
