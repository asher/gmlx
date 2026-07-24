# Streaming: MoE models bigger than memory

A MoE model whose file exceeds what the GPU can wire - or exceeds RAM
outright - still runs: gmlx streams the expert weights from disk and
keeps what every token needs resident. MoE decode is what makes this
viable: only the routed experts are read per token, so the per-token
working set is a small slice of the file. The levers in this guide
exist to keep that slice cheap - fed from the SSD ahead of demand,
cached where it repeats, and, strictly opt-in, thinned where the
router can spare it.

Set expectations first. When experts stream from disk, decode is bound
by the SSD and CPU, not the GPU, and single-digit tokens per second is
normal - the feeders raise the constant, not the nature of the bound.
This is a capacity feature that makes a 200B-class MoE usable on a
64 GB machine, not a speed feature, and it is strictly for the
over-budget case: a model that fits in memory runs several times
faster on the normal GPU path.

This guide covers launching a streamed model, choosing between the two
placements, the lossless levers (the feeders, lookahead prestage, GPU
keep-warm), and the lossy levers with a measured decision procedure.
General performance topics - measuring, quant choice, speculative
decoding, the prompt cache, memory and the KV cache - are in
[performance.md](performance.md).

## Quick start

```sh
# sharded files: point at the first shard
gmlx run GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf --stream-experts
```

Or served, with the per-model `stream` key:

```yaml
models:
  glm:
    gguf: ~/models/GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf
    stream: experts
    kv_bits: 8
```

The load is a mmap, not a read, so generation starts within seconds
whatever the file size. Prefill stages each layer's experts through the
prefill feeder. Decode starts at the disk's demand rate and improves
over the first few dozen tokens as the expert arena converges on the
model's hot set; the decode feeder's exit stats (printed at `-v` on
`run` and `chat`, always in server logs) show the arena hit rate a
session settled at.

Streaming applies to text models - the server rejects `stream` on VLM
and speculative entries. On the CLI, MTP composes with
`--stream-experts` but defers by default: auto-MTP stays off and an
explicit `--speculative` opts in. What stays resident - the every-token
layers plus the KV cache - follows the normal fit arithmetic in
[getting-started.md](getting-started.md#will-it-fit), and a quantized
KV cache (`--kv-bits 8`) is the usual companion at long context.

## Choosing a placement

Two placements run MoE models whose files exceed what the GPU can wire:

- `--stream-experts` keeps the every-token layers (attention, routers, shared
  experts, KV cache) on the GPU and streams the routed experts, which run on
  the CPU stream. Historically slower than `--stream-cpu` at short context
  because of the per-layer handoff; the decode feeder (below) reverses that,
  and `--stream-experts` keeps its long-context advantage with a quantized KV
  cache, where the large KV stays on GPU.
- `--stream-cpu` runs the whole model on the CPU device, mmap-backed, so the page cache
  streams weights from disk on demand. Past the wired budget the runtime adds
  sequential expert prefetch, advising the kernel a couple of layers ahead so
  prefill reads expert stacks at sequential bandwidth instead of demand-faulting
  them.

With the decode feeder on, `--stream-experts` is the usual choice: it
matches `--stream-cpu` on short generations, pulls ahead once the arena
warms (measured below), and keeps the large KV cache on GPU at depth.
`--stream-cpu` keeps everything on one device; in a server config,
`stream: cpu` switches the whole process to the CPU device, so it suits
a single-model setup rather than mixing with GPU-resident models.

## The feeder paths

Streaming models engage two *feeder* paths by default:

- The **prefill feeder** (`--no-prefill-feeder` disables) stages each layer's
  expert stacks straight from the GGUF into GPU-visible ring slots while the
  previous layer computes, so every byte makes one trip - the page-cache path
  reads each expert byte twice on a machine that is at memory capacity by
  definition. Short prompts stage only the experts the router actually chose
  instead of whole layers (measured on an M3 Max, 162 GB MiniMax Q5_K_M: a
  53-token prompt's time-to-first-token dropped from 19.4 s to 11.4 s).
- The **decode feeder** (`--stream-experts` only; `--no-decode-feeder`
  disables) keeps the most-routed experts of every layer in a wired,
  popularity-managed GPU arena sized to the machine (`GMLX_DECODE_ARENA_GB`
  overrides) and reads only the misses from the GGUF, at SSD queue depth. The
  arena starts empty and converges within a few dozen tokens. The arena is
  wired, so it also polices itself: under system memory pressure (another
  model, a build) it shrinks, keeping its most popular experts, and regrows
  once pressure clears - a long-running model stays a good citizen on a
  machine that is doing other work (`GMLX_DECODE_PRESSURE=0` pins it instead).
  Same model and
  box: decode went from 2.4 tok/s on the page-cache path to 4.0 tok/s averaged
  over a 512-token generation (~4.7 steady, ~90% arena hits), against 3.0
  tok/s for `--stream-cpu` - so `--stream-experts` now matches `--stream-cpu`
  on short generations and pulls ahead roughly 1.5x once the arena warms,
  before the KV-cache advantage at depth.

In server configs the placement is the per-model `stream: experts | cpu`
key and the feeder opt-outs are `prefill_feeder: false` /
`decode_feeder: false`.

## Lookahead prestage

With the decode feeder on, arena misses are also *prestaged by lookahead*
(`GMLX_DECODE_LOOKAHEAD=0` disables): each MoE layer runs the next MoE layer's router
on its own input and pre-reads the predicted misses on a small dedicated pool
while the current layer computes. The residual changes little between
adjacent sublayers, so the prediction lands: measured recall of the next
layer's actual top-k is ~78% on GLM-5.2 (@8) and MiniMax-M3 (@4), against
~35% for previous-token routing reuse. Predictions move bytes and nothing
else - routing and outputs are bit-identical - and speculation is kept off
the demand path three ways: prestage reads are submitted only after the
current layer's demand misses have finished, the read threads run at
utility disk-I/O priority so the kernel services demand misses first
(`GMLX_DECODE_LOOKAHEAD_IOPOL=0` restores default priority), and every
layer settles its in-flight prestages before serving. A per-layer rank
gate watches how often each prediction rank actually lands and stops
submitting ranks that measure below `GMLX_DECODE_LOOKAHEAD_MIN_P` (default
`0.5`); predictions the router then does not route to are cancelled before
they reach the disk when their reads have not started
(`GMLX_DECODE_LOOKAHEAD_CANCEL=0` disables). Together these keep the
wasted-read tax near zero on models where the SSD is the bottleneck.
`GMLX_DECODE_LOOKAHEAD_PROBE=1` prints the per-layer recall table at exit without
issuing reads, the check worth running on a new model family.

When a larger-than-RAM model is released, its page cache is also released, via
`msync(MS_INVALIDATE)` over the shards - at process exit, or at unload on a
running server (`GMLX_RELEASE_PAGECACHE=0` disables).

## GPU keep-warm (`--gpu-keepwarm`)

Streamed decode has a work pattern the GPU's power management punishes:
sub-millisecond compute bursts separated by host and disk gaps every MoE
layer. The GPU races to idle in each gap, its clocks sag, and the next
burst pays the ramp back up - measured as 3-5x inflation of identical
per-layer work (0.3 ms warm vs up to 4+ ms ramp-inflated). The more
per-token host syncs a model's decode path has, the more of its token time
is ramp rather than work.

`--gpu-keepwarm` (env `GMLX_GPU_KEEPWARM=1`; serve: `--gpu-keepwarm` or
config `server.gpu_keepwarm: true`) holds clocks up with a tiny heartbeat
kernel (a 256x256 matmul every 0.5 ms) on its own stream from a background
thread. It moves no model bytes and changes no outputs - the win is purely
clock residency. Measured on the production configs of two over-RAM
models, ABBA-alternated medians over 4 reversed rounds of 512-token
generations:

| model | config under test | without | with | lift |
|---|---|---|---|---|
| GLM-5.2 UD-IQ3_XXS (262 GB, 75 streamed layers) | arena 70 + `--moe-miss-shed 0.85`, lookahead off | 2.51 tok/s | 3.64 tok/s | +45% |
| Hunyuan3 IQ4_XS (159 GB, 79 streamed layers) | `--moe-layer-shed 0.10` + `--moe-miss-shed 0.90` | 4.01 tok/s | 5.29 tok/s | +32% |

The diagnostic signature is worth knowing because it says whether a given
model will benefit: stall time and arena hit rate are unchanged by the
heartbeat (the disk is doing the same work), so if a streamed model's
per-token time is dominated by the eval/sync bucket rather than stalls
(`GMLX_DECODE_PHASE_STATS=1` prints the split), clock sag is a candidate
and keep-warm is the cheap test. Dense in-RAM decode does not have the
gap pattern and gains nothing.

The cost is power, and only while decoding: the heartbeat parks (no GPU
work) after one second without decode activity and wakes on the next
streamed decode call (`GMLX_KEEPWARM_IDLE_S` tunes the window; `0` beats
continuously). An idle server pays nothing; the first token after an idle
gap pays one clock ramp.

The heartbeat kernel itself is nearly free, and measuring it shows the
mechanism plainly (powermetrics, M5 Max, heartbeat alone on an otherwise
idle box): GPU power 199 mW -> 287 mW, while GPU active residency went
from 58% to 99.8% with the clock still at the 338 MHz floor. The beat
does not raise clocks - it removes idleness. During decode the model's
own bursts set the clock level; the heartbeat keeps the GPU from going
idle in the gaps, so the governor holds that level instead of sagging
and re-ramping every layer. The real power cost is therefore holding
the decode-level P-state through the gaps, which scales with how hard
the workload drives the clock - the ~0.1 W kernel is noise against it.
It is opt-in because that trade - more watts for the same output,
faster - is yours to make, not a default, especially on battery.

## The lossy levers

Four levers trade a bounded amount of output quality for decode speed on
streamed MoE models. None is ever on by default: absent flags and absent
config keys mean lossless routing. All are decode-side - a large prefill
chunk routes to nearly every expert either way - and all act only on
streamed layers.

A streamed decode token pays three distinct costs, and each lever cuts a
different one. Experts that miss the decode arena are read from disk at
demand latency - the dominant cost when the hit rate is low. Experts
already resident in the arena cost only gather compute, which is small.
And every streamed MoE layer pays a fixed per-layer overhead (kernel
launches plus a host synchronization) that does not shrink when fewer
experts are routed.

The router-side levers thin the routed set itself, cutting reads and
compute in proportion. `--moe-experts K` caps the router at a fixed K
experts per token. `--moe-expert-mass P` is the adaptive version and
usually the better trade of the two: each token keeps the smallest set of
its routed experts covering share P of the gate mass, so the dropped mass
is bounded by 1-P and lands on tokens where the router was already
confident. A token whose top 3 experts carry 92% of the mass reads 3
experts at P=0.9, while an uncertain token keeps the full fan-out. How
much expert-mass buys is a property of the model's router. On a
concentrated router it is the strongest lever available: most reads
disappear for a few percent of dropped mass. On a flat router it buys
almost nothing - a 299B model we measured keeps 7.1 of its 8 experts at
P=0.90. Measure rather than guess: `--moe-expert-probe` runs the trained
routing losslessly and prints, per candidate P, the experts kept, the
implied read fraction, and the mass actually dropped, with decode and
prefill tabled separately. Size P against the decode table. The two router levers
compose (`--moe-experts 6 --moe-expert-mass 0.9` caps at 6, then drops
within the 6).

The staging levers act at the decode feeder instead of the router.
`--moe-miss-shed P` drops routed experts that would demand-miss the
arena, lowest scores first, keeping at least share P of the token's gate
mass, so its quality budget is spent exactly where the disk stalls are:
an arena-resident or prestage-inflight expert is never dropped, and a
shed expert earns no popularity credit, so the arena keeps its hot set.
It needs the decode feeder and a block that hands router scores to the
expert call; where it engages, it is the most targeted lever per point of
quality spent, and its payoff scales directly with the miss rate.
`--moe-layer-shed P` skips a streamed MoE layer's routed experts entirely
with probability P per token (the layer's shared expert still runs). It
is the blunt end of the scale, and the only lever that also cuts the
per-layer overhead - which makes it the one that still pays when the
arena hit rate is high and misses are rare.

Which to reach for is a measurement, not a doctrine. Run the probe once,
and read the decode feeder's exit stats (arena hit rate; printed by
`run`/`chat` at `-v`, and always in server logs) from a representative
session. The hit rate decides first. A low hit rate - an arena small
relative to the model - makes `--moe-miss-shed` the lead lever whatever
the router looks like: it spends only on calls that would stall, so it
beats expert-mass on cost at equal reads saved, and a probe-attractive
concentration number can still lose to it outright (measured below). At
a healthy hit rate, a concentrated router points to `--moe-expert-mass`,
which removes reads and compute together at minimal dropped mass; a flat
router takes it off the table and leaves the per-layer overhead as the
standing cost, which only `--moe-layer-shed` touches - though a large
share of that overhead is clock ramp, which the lossless keep-warm above
removes first; run it before spending quality here.

One measured point, from the flat-router end of that space: Hy3, a
299B-A21B MoE (161 GB IQ4_XS streaming on a 128 GB machine, decode arena at a ~92%
hit rate; decode-only tok/s from alternated A/B rounds of 512-token
generations; quality scored at temperature 0.6 / top-p 0.95 on a 12-task
goal battery of JSON extraction, constrained format, code with asserts,
multi-step arithmetic, and length control, plus a repetition check):

| setting | decode | quality |
|---|---|---|
| `moe_layer_shed: 0.10` | +8% | clean |
| `moe_miss_shed: 0.90` | +4% | clean |
| both together | +13% | clean |
| the pair softened to 0.07 / 0.93 | +2-4% | clean |
| `moe_expert_mass: 0.90` | ~0%, alone or stacked | clean |

Those are sustained-regime medians. Per the cool-box transient note
under [Measuring](performance.md#measuring), a rested 14-inch machine ran the same arms 15-25%
faster for its first twenty minutes, baseline at 5.0 tok/s and the pair
at 5.6 or better. Note also that softening the pair keeps its quality
margin but not its speed: miss-shed's payoff falls steeply as P rises
(at 0.93 it sheds only a third of the experts it sheds at 0.90), so the
softened pair returned a few percent where the full pair returned +13.
The soft edge is real. In single long-generation checks at this model
card's temperature of 0.9, the pair at 0.09/0.91 emitted a stray token
into code even under top-p 0.97, while 0.07/0.93 ran clean, so the
high-temperature envelope on this model is the softened pair and its
few percent. The larger wins belong to workloads that can run cooler
sampling or accept an occasional stray.

That ordering is this model's, not a law. With a flat router,
expert-mass had nothing cheap to drop; at a 92% hit rate, misses were
rare enough that the per-layer overhead was the standing cost, so
layer-shed led and the shed pair composed (+8% and +4% multiply to
roughly the observed +13%: they cut disjoint costs). On a
concentrated-router model with a healthy hit rate the probe will show
the inversion - most reads removed for a few percent of mass - before
any lossy run needs to be made.

A second measured point, from the low-hit-rate end of the space:
MiniMax-M3, a 4-of-128-expert MoE (264 GB Q4_K_M streaming on the same
128 GB machine, decode arena at an ~87% hit rate; same alternated A/B
method, decode-only medians over 512-token generations). A layer
stalls when any one of its four routed experts misses, so at 87%
per-expert residency roughly half of all token-layer calls stall, and
the miss-targeted lever leads:

| setting | decode | disk stall time |
|---|---|---|
| `moe_miss_shed: 0.85` | +1% | -14% |
| `moe_miss_shed: 0.80` | +6.5% | -31% |
| `moe_expert_mass: 0.85` | ~-3% | -9% |

The probe put this router in the middle of the concentration range
(P=0.85 keeps 3.7 of 4 experts on decode for 4% dropped mass), and
expert-mass did remove reads. But most of the reads it removed were
arena hits that cost nothing, and its router-side filtering cost more
than the stalls it saved. Miss-shed spends the same budget only where
a stall is otherwise certain, which also means its realized cost sits
far below the probe's unconditional number: at P=0.80 the probe
predicts 12% dropped mass, while the residency-aware shed dropped
2.9%, shedding 8% of routed experts across a third of token-layer
calls. Two 10k-token generations at temperature 0.6 / top-p 0.95 ran
clean, producing complete working artifacts with no stray tokens. The
probe sizes expert-mass, but it cannot see residency; when the exit
stats show a low hit rate, reach for miss-shed first.

A third measured point moves one variable: routing width. GLM-5.2
(278 GB UD-IQ3_XXS, 256 experts top-8, sigmoid gating) streams on the
same machine at a per-expert hit rate near 88% - healthier than M3's -
yet stalls more, because a layer stalls when any of eight routed
experts miss, not four: at hit rate h the stall odds are 1 - h^k, so
k = 8 roughly doubles them at the same h. That amplification works in
both directions - every point of hit rate miss-shed buys back is worth
about twice as much - which is why the same lever measured stronger
here (+16.5% decode at P=0.80, stalls halved; +10.7% at P=0.85, both
even-round alternated 512-token medians) and why arena size, flat on
M3, mattered too (each arena GB bought ~0.2pt of hit rate).

Wider routing also concentrates more meaning per expert, and that
moved the quality cliff: P=0.80, clean on M3, broke GLM-5.2 - and
broke it in a way character scans cannot see. A 12k-token
one-page-app generation completed with no stray tokens, valid markup,
and working code, but the page it drew was missing its subject (a sky
with no road and no car, on a prompt asking for a car on a road; the
lossless twin at the same seed drew the full scene). P=0.85 drew the
complete scene. Dropped gate mass degrades content before it degrades
form, so a shed level cannot be certified by scanning the output for
corruption: render the artifact and look at it, at deploy sampling
settings, against a lossless twin at the same seed. Miss-shed's safe
range is per-architecture; re-gate it whenever routing width or
gating changes.

Back on Hy3, quality degraded in a consistent order as the levers
hardened: multi-step arithmetic broke first, well before coherence,
formatting, or code. `moe_layer_shed: 0.20` alone dropped arithmetic
tasks, and so did
`moe_layer_shed: 0.10` combined with `moe_miss_shed: 0.75` even though
each setting is clean alone - stacked levers compound onto the same
cliff, so leave margin on both. On the same battery `moe_miss_shed`
alone stayed clean down to 0.75 and `moe_expert_mass` down to 0.70. If a
workload leans on chained arithmetic, put that in the test set before
sizing any of these.

Past the edge, long generations add a second signature: stray token
substitutions such as wrong-script digits or a bullet character inside
code. The
levers widen the low-probability tail that sampling can reach, so their
safe range depends on the sampling regime: untruncated sampling (top-p
1.0, which some model cards recommend) exposes the whole perturbed tail
that nucleus truncation would mask. Certify at the temperature and top-p
you deploy with - a check run at a lower temperature does not cover a
hotter one. And a short check certifies short answers: a per-token slip
rate too small for it to see still accumulates over a 10k-token
generation, so size levers softer for long-form code than short-form
checks suggest.

### One prompt, four settings

The cost is easier to see than to score. The same one-shot prompt (a
single-file HTML canvas animation of a car driving through parallax
scenery) was run once per setting on the same Hy3 IQ4_XS build, at the
model card's temperature of 0.9 with low reasoning effort, and each generated
page screenshotted. These are single samples at high temperature, so
read them as an illustration, not a certification.

<details>
<summary>The prompt (identical for all four runs)</summary>

> Write a single HTML file with a full-page canvas and no libraries.
> Simulate a realistic side-view of a moving car as the main subject.
> Keep the car visible in the foreground while the background landscape
> scrolls continuously to create the feeling that the car is driving
> forward. Use layered scenery for depth: nearby ground, roadside
> elements, trees, poles, and distant hills or mountains should move at
> different speeds for a natural parallax effect. Animate the wheels
> spinning realistically and add subtle body motion so the car feels
> connected to the road. Let the environment pass smoothly behind it,
> with repeating but varied scenery that makes the movement feel
> believable. Use cinematic lighting and a cohesive sky, such as sunset,
> dusk, or daylight, to enhance atmosphere. The overall motion should
> feel calm, immersive, and realistic, with a seamless looping
> animation.

</details>

| | |
|---|---|
| <a href="assets/perf/lossy-hy3-baseline.html"><img src="assets/perf/lossy-hy3-baseline.png" alt="lossless baseline: detailed sunset scene with streetlight, lane markings, and layered trees"></a><br>lossless, top-p 1.0. 13.2k tokens at 3.0 tok/s. | <a href="assets/perf/lossy-hy3-shed-0.07-0.93.html"><img src="assets/perf/lossy-hy3-shed-0.07-0.93.png" alt="layer-shed 0.07 with miss-shed 0.93: simpler but coherent mountain scene"></a><br>`moe_layer_shed 0.07` + `moe_miss_shed 0.93`, top-p 0.97. 10.6k tokens at 3.5 tok/s. |
| <a href="assets/perf/lossy-hy3-shed-0.10-0.90.html"><img src="assets/perf/lossy-hy3-shed-0.10-0.90.png" alt="layer-shed 0.10 with miss-shed 0.90: flatter, darker scene with simpler shapes"></a><br>`moe_layer_shed 0.10` + `moe_miss_shed 0.90`, top-p 0.95. 11.1k tokens at 3.6 tok/s. | <a href="assets/perf/lossy-hy3-shed-0.20-0.80.html"><img src="assets/perf/lossy-hy3-shed-0.20-0.80.png" alt="layer-shed 0.20 with miss-shed 0.80: black page, the script crashed on a stray token"></a><br>`moe_layer_shed 0.20` + `moe_miss_shed 0.80`, top-p 1.0. 10.0k tokens at 4.2 tok/s. |

The scene simplifies as the levers harden, well before anything breaks.
The first three pages ran clean. The black frame is the past-the-edge
signature in the wild: that page died on its first stray token, a bullet
character where an operator belonged, with CJK characters spliced into
two identifiers further down the same file. The middle setting also
shows the sampling interaction from above. Its page was generated clean
at top-p 0.95, while the same setting sampled untruncated put one
wrong-script token into an 11k-token run. The tok/s figures are
whole-run averages of these single generations at different lengths,
not controlled A/B numbers; the table above is the measured comparison.
Each screenshot links to its generated page, committed beside it in
`docs/assets/perf/` (GitHub shows the page source; download one to
watch the animation).

The sampling interaction also has a constructive reading. The full pair
was rerun once on the same prompt and build with cooled sampling, at
temperature 0.6 and top-p 0.95 instead of the model card's 0.9:

<a href="assets/perf/lossy-hy3-shed-0.10-0.90-cool.html"><img src="assets/perf/lossy-hy3-shed-0.10-0.90-cool.png" alt="layer-shed 0.10 with miss-shed 0.90 at temperature 0.6: layered sunset scene with a red car, lampposts, treeline, and the sun setting behind the hills"></a><br>`moe_layer_shed 0.10` + `moe_miss_shed 0.90`, temperature 0.6, top-p 0.95. 10.2k tokens at 3.8 tok/s.

It ran clean and produced one of the strongest scenes of the whole set,
from the same full pair that needed softening to survive temperature
0.9. This is a single sample like the others, but it points at the
practical recipe on this model: keep the full pair and its entire +13%,
and cool the sampling slightly, rather than giving up most of the speed
win by softening the levers at the card's temperature.

In server configs the lossy levers are the per-model `moe_experts: K` /
`moe_expert_mass: P` / `moe_miss_shed: P` / `moe_layer_shed: P` keys (or
the matching `serve` flags for a single positional model); the probe
stays CLI-only, so size P with a `gmlx run --moe-expert-probe` pass
before pinning a value in a config.

## Native-fp experts (MXFP4/NVFP4)

Models with MXFP4/NVFP4 expert tensors (gpt-oss, DeepSeek-V4-Flash Q4_K_XL
quants) participate in all of the above on equal terms. By default these
tensors are eagerly repacked into MLX's packed layout at load - fine in RAM,
fatal over it (the repack materializes every expert). `GMLX_NATIVE_FP`
picks the layout: `wire` keeps them as zero-copy GGUF wire bytes served by
mlx-kquant's fp4 kernels (loads in seconds, streams like any k-quant),
`packed` forces the repack, and the default `auto` chooses wire whenever a
CPU placement is requested or the file exceeds ~90% of the wired budget.
Wire mode is a hair slower than packed when the model fits in RAM (gpt-oss
decode ~5% at depth 0, converging at depth; prefill is at parity or better)
but is what makes the over-RAM case work at all, and it cuts the load time
from minutes of repack to a mmap.
