# Streaming measurements

The samples and case studies behind
[Models larger than memory](../streaming.md) show what the lossless
over-budget path produces, what each lossy setting did on four models, and
how a setting is certified.

- [What the over-budget case produces](#what-the-over-budget-case-produces)
- [Hy3](#hy3)
- [MiniMax-M3](#minimax-m3)
- [GLM-5.2](#glm-52)
- [Kimi-K3](#kimi-k3)
- [DeepSeek-V4.1-Flash engram tables](#deepseek-v41-flash-engram-tables)
- [Settings that passed](#settings-that-passed)
- [Certifying a setting](#certifying-a-setting)
- [Four settings on Hy3](#four-settings-on-hy3)
- [Four shed levels on Kimi-K3](#four-shed-levels-on-kimi-k3)
- [Lossless setting measurements](#lossless-setting-measurements)
- [Decode feeder defaults](#decode-feeder-defaults)

## What the over-budget case produces

Two single samples came from Kimi-K3, a 2.8T-parameter MoE, generated on an
M5 Max MacBook Pro with 128 GB. Both used `--stream-experts` at streaming
defaults, on the lossless path. The prompt is the one-shot
canvas-animation prompt that the lossy-setting comparisons also use. Each
screenshot links to the generated page, committed beside it in
`docs/assets/perf/`. GitHub shows a page's source, so download one to
watch the animation.

<a href="../assets/perf/kimi-k3-ud-iq2xxs-car.html"><img src="../assets/perf/kimi-k3-ud-iq2xxs-car.png" alt="Kimi-K3 UD-IQ2_XXS: sunset scene, red coupe with a headlight beam under power lines, the sun low between hills"></a><br>The UD-IQ2_XXS file is 662 GB. One generation wrote 30.8k tokens, thinking included, at 1.34 tok/s and temperature 1.0.

<a href="../assets/perf/kimi-k3-ud-q2kxl-car.html"><img src="../assets/perf/kimi-k3-ud-q2kxl-car.png" alt="Kimi-K3 UD-Q2_K_XL: film-grain dusk scene, red sedan with a headlight cone, telegraph poles, layered hills and clouds"></a><br>The UD-Q2_K_XL file is 861 GB, of which 799 GB of experts stay file-backed. One generation wrote 23.7k tokens, thinking included, at 1.15 tok/s and temperature 1.0.

Both pages ran as generated. A model five to seven times the machine's
RAM sustained coherent single-file programs of 24k to 31k tokens at
1.15 to 1.34 tok/s.

## Hy3

Hy3 is the flat-router end of the range. It is a 299B-A21B MoE, streamed as
a 159 GB IQ4_XS file on a 128 GB machine with the decode arena at about a
92% hit rate. Decode-only tok/s come from alternated A/B rounds of
512-token generations. Quality was scored at temperature 0.6 and top-p
0.95 on a 12-task goal battery of JSON extraction, constrained format,
code with asserts, multi-step arithmetic and length control, plus a
repetition check.

| Setting | Decode | Quality |
|---|---|---|
| `moe_layer_shed: 0.10` | +8% | Clean |
| `moe_miss_shed: 0.90` | +4% | Clean |
| Both together | +13% | Clean |
| The pair softened to 0.07 / 0.93 | +2-4% | Clean |
| `moe_expert_mass: 0.90` | ~0%, alone or stacked | Clean |

Those are sustained-regime medians. A rested machine ran the same arms
faster until the chassis throttled, as
[Measuring](../performance.md#measuring) describes. At 0.93, miss-shed
sheds only a third of the experts it sheds at 0.90, so the softened pair
gained a few percent where the full pair gained 13%.

At the model card's temperature of 0.9, the pair at 0.09/0.91
emitted a stray token into code even under top-p 0.97, while 0.07/0.93
ran clean. With a flat router, expert-mass had no low-mass experts to
drop, and at a 92% hit rate the per-layer overhead was the constant cost,
so layer-shed gained more. The two shed settings cut disjoint costs, so
their gains multiply.

## MiniMax-M3

MiniMax-M3 is the low-hit-rate end of the range. It is a 4-of-128-expert
MoE, streamed as a 264 GB Q4_K_M file on the same 128 GB machine with the
decode arena at about an 87% hit rate, measured the same way as Hy3. A
layer stalls when any one of its four routed experts misses, so about 43%
of all token-layer calls stall.

| Setting | Decode | Disk stall time |
|---|---|---|
| `moe_miss_shed: 0.85` | +1% | -14% |
| `moe_miss_shed: 0.80` | +6.5% | -31% |
| `moe_expert_mass: 0.85` | ~-3% | -9% |

Expert-mass removed reads, but most of them were arena hits that cost
nothing, so its router-side filtering cost more than the stalls it saved.
Miss-shed drops mass only where a stall is otherwise certain. At P=0.80
the probe predicts 12% dropped mass, while the residency-aware shed
dropped 2.9%. Two 10k-token generations at temperature 0.6 and top-p 0.95
ran clean.

## GLM-5.2

GLM-5.2 changes the routing width. It is a 282 GB UD-IQ3_XXS file with 256
experts routed top-8 under sigmoid gating. On the same machine it streams
at a per-expert hit rate near 88%, higher than MiniMax-M3, yet it stalls
more, because a layer stalls when any of eight routed experts misses. At
that hit rate the stall odds rise from 40% with four experts to 64% with
eight.

Each point of hit rate that miss-shed recovers therefore removes more
stalls. It gained 16.5% decode at P=0.80 with stalls halved, and 10.7% at
P=0.85, both as alternated 512-token medians. Each GB of arena added about
0.2 points of hit rate.

Wider routing also moved the quality threshold. At P=0.80, which was
clean on MiniMax-M3, a 12k-token one-page-app generation completed with
valid markup and working code. The prompt asked for a car on a road, but
the page showed a sky with no road and no car. The lossless run at the
same seed drew the full scene, and so did P=0.85.

## Kimi-K3

Kimi-K3 UD-Q2_K_XL is 861 GB on the same 128 GB machine, with 896 experts
routed 16 to a token across 92 streamed expert layers. The arena holds a
small fraction of the expert set, the lossless hit rate is about 50%, and
demand stalls take about two thirds of decode wall time. Each shed arm
pairs miss-shed with `--moe-prestage keepers`. Every setting ran one long
generation on the same one-shot prompt as the scale samples, at
temperature 1.0, for 23k to 30k tokens with thinking included. The figures
are whole-run averages, not alternated A/Bs.

| Setting | Dropped mass | Hit rate | Decode |
|---|---|---|---|
| Lossless, ranked prestage | None | 49.9% | 1.15 tok/s |
| `moe_miss_shed: 0.80` | 17.1% | 68.0% | 1.19 tok/s (+3%) |
| `moe_miss_shed: 0.70` | 26.1% | 71.9% | 1.33 tok/s (+16%) |
| `moe_miss_shed: 0.65` | 30.4% | 74.2% | 1.39 tok/s (+21%) |

Memory pressure sized the wired arena from 29 to 33 GB across the runs,
so read the columns as a trend. Shedding raised the remaining experts'
hit rate, because the arena stops loading experts that would be
dropped anyway. All three shed levels produced complete working pages, as
[Four shed levels on Kimi-K3](#four-shed-levels-on-kimi-k3) shows. At
0.60, a code generation produced a program that did not work.

## DeepSeek-V4.1-Flash engram tables

The engram tier is the first case of more than one streamable table, and
of a table read 24 rows deep per token. The figures come from the Q2_K
file, 246 GiB in 7 shards, on an M3 Max with 128 GB, gathering from the
layer-1 table of 384,006,168 rows at 84 bytes, 30.04 GiB in all.

| Case | Page cache | Result |
|---|---|---|
| Decode, 24 rows, 200 steps | Cold | 0.32 ms mean, 0.19 p50, 0.33 p90, 18.09 max |
| Decode, 24 rows, 200 steps | Warm | 0.17 ms mean |
| Prefill, 4096 tokens, 98,304 rows | Cold | 0.02 s, 5.8M rows/s |
| The same with a thread-pool page touch issued ahead | Cold | 6.35 s, of which 6.34 s is the touch |
| The same as plain, with a concurrent F_NOCACHE reader at 15.8 GiB/s | Cold | 0.02 s, 6.1M rows/s |

The gather is not the bottleneck at either width, so the tier needs no
prefill prefetch, and the tables and the expert ring need no I/O
arbitration. Crediting both tables as off-disk lets the planner give the
arena 75.5 GB, 38% of the experts. Without the credit, it prices the
tables as every-token weights and the arena falls to 11 GB.

A table past the size of a 1-D Metal window was once copied into Metal
memory instead of mapped. mlx-kquant now gives such a window a second
dimension. Loading one tensor per file alone gives these figures.

| | Copied | 2-D window |
|---|---|---|
| Q2_K table, zero-copy views | 61 of 62 | 62 of 62 |
| Q2_K table, `IOAccelerator` | 30.0 GiB | 64 KiB |
| Q2_K table, resident mapped file | 30.0 GiB | 528 KiB |
| q8_0 geometry, `IOAccelerator` | 17.7 GiB | 64 KiB |

The whole model ran as follows on the same machine with `--stream-experts`
at streaming defaults.

| | 120 tokens, tables copied | 200 tokens, tables copied | 120 tokens, 2-D windows |
|---|---|---|---|
| Decode | 2.03 tok/s | 2.07 tok/s | 5.97 tok/s |
| Prefill | 2.11 tok/s | 2.24 tok/s | 3.28 tok/s |
| Arena | 13 GB, shed to 3 | 13 GB, shed to 3 | 75.3 GB, no shed |
| Arena hit rate | 42.5% | 44.4% | Not recorded. |
| Expert bytes read per token | 2.6 GB | 2.6 GB | Not recorded. |
| Demand-read stalls | 34.2 s of 77 s | 50.5 s of 113 s | Not recorded. |

The larger arena took the disk stalls from 210 ms per token to 33 ms, in
the phase split that `GMLX_DECODE_PHASE_STATS=1` prints. The GPU sync time
also halved, because a decode that does not wait on the SSD keeps the GPU
off its clock floor.

`GMLX_DECODE_LAYER_PROFILE=2` measured the two hyper-connection routes over
48 greedy tokens after a 16K prompt, with the profile's own syncs included.
This run used `DeepSeek-V4.1-Flash-Q2.gguf`, the single-file ds4 conversion
that [Benchmarks](../benchmarks.md#model-provenance) lists, instead of the
Q2_K file.

| Route | Hyper-connections, ms per token | Total ms per token | Decode |
|---|---|---|---|
| Ops | 93.6 | 250.1 | 4.58 tok/s |
| Fused | 19.4 | 162.6 | 6.85 tok/s |

At 64K the fused route spends 19.0 ms per token on hyper-connections, so
the term does not grow with depth.

## Settings that passed

Four lossy settings passed the
[certification procedure](#certifying-a-setting) on the models they were
measured on.

| Model | Setting | Quality verdict |
|-------|---------|-----------------|
| Hy3 IQ4_XS | `moe_layer_shed: 0.10` with `moe_miss_shed: 0.90` | The pair showed no defects at temperature 0.6. Lower it to 0.07 and 0.93 at the card's temperature of 0.9. |
| MiniMax-M3 Q4_K_M | `moe_miss_shed: 0.80` | Two 10k-token generations showed no defects. |
| GLM-5.2 UD-IQ3_XXS | `moe_miss_shed: 0.85` | The setting showed no defects. At 0.80 the model dropped scene content but kept valid form. |
| Kimi-K3 UD-Q2_K_XL | `moe_miss_shed: 0.65` to `0.80` | Every level produced working pages, and one sample per level cannot rank their content drift. At 0.60 the code broke. |

## Certifying a setting

Quality degrades in a consistent order as the settings become more
aggressive. On Hy3, multi-step arithmetic broke first, well before
coherence, formatting or code. `moe_layer_shed: 0.20` alone dropped
arithmetic tasks, and so did `moe_layer_shed: 0.10` with
`moe_miss_shed: 0.75`, although each half of that pair is clean alone.
Past the quality threshold, long generations also show stray token
substitutions, such as wrong-script digits or a bullet character inside
code.

The procedure works on any model.

1. Generate with the lossless path and with the candidate setting on the
   same prompt at the same seed, at the temperature and top-p you deploy
   with. A check at a lower temperature does not cover a higher one.
2. Score a short goal battery of JSON extraction, constrained format, code
   with asserts, multi-step arithmetic, length control and a repetition
   check. Put chained arithmetic first if the workload depends on it.
3. Generate something long, 10k tokens or more, and scan it for stray
   tokens. A per-token error rate too small to appear in a short check
   still accumulates.
4. Render the artifact and compare it with the lossless run. Dropped gate
   mass degrades content before form, so a page can be valid and complete
   with its subject missing.
5. Leave margin on each stacked setting, since their effects add up against
   a single quality threshold. Certify again whenever routing width or
   gating changes.

## Four settings on Hy3

A one-shot prompt asks for a single-file HTML canvas animation of a car
driving through parallax scenery. It ran once for each setting on the same
Hy3 IQ4_XS build, at the model card's temperature of 0.9 with low
reasoning effort. These are single samples at high temperature, so read
them as an illustration rather than a certification.

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

<a href="../assets/perf/lossy-hy3-baseline.html"><img src="../assets/perf/lossy-hy3-baseline.png" alt="lossless baseline: detailed sunset scene with streetlight, lane markings, and layered trees"></a><br>The lossless run used top-p 1.0 and wrote 13.2k tokens at 3.0 tok/s.

<a href="../assets/perf/lossy-hy3-shed-0.07-0.93.html"><img src="../assets/perf/lossy-hy3-shed-0.07-0.93.png" alt="layer-shed 0.07 with miss-shed 0.93: simpler but coherent mountain scene"></a><br>The run with `moe_layer_shed 0.07` and `moe_miss_shed 0.93` used top-p 0.97 and wrote 10.6k tokens at 3.5 tok/s.

<a href="../assets/perf/lossy-hy3-shed-0.10-0.90.html"><img src="../assets/perf/lossy-hy3-shed-0.10-0.90.png" alt="layer-shed 0.10 with miss-shed 0.90: flatter, darker scene with simpler shapes"></a><br>The run with `moe_layer_shed 0.10` and `moe_miss_shed 0.90` used top-p 0.95 and wrote 11.1k tokens at 3.6 tok/s.

<a href="../assets/perf/lossy-hy3-shed-0.20-0.80.html"><img src="../assets/perf/lossy-hy3-shed-0.20-0.80.png" alt="layer-shed 0.20 with miss-shed 0.80: black page, the script crashed on a stray token"></a><br>The run with `moe_layer_shed 0.20` and `moe_miss_shed 0.80` used top-p 1.0 and wrote 10.0k tokens at 4.2 tok/s.

The scene simplifies as the settings become more aggressive, and the
first three pages ran clean. The black page failed on its first stray
token, a bullet character where an operator belonged. The tok/s figures
are whole-run averages of single generations at different lengths, so for
the measured comparison read the [Hy3](#hy3) table.

On the same prompt and build, the full pair also ran once at temperature
0.6 and top-p 0.95.

<a href="../assets/perf/lossy-hy3-shed-0.10-0.90-cool.html"><img src="../assets/perf/lossy-hy3-shed-0.10-0.90-cool.png" alt="layer-shed 0.10 with miss-shed 0.90 at temperature 0.6: layered sunset scene with a red car, lampposts, treeline, and the sun setting behind the hills"></a><br>The run with `moe_layer_shed 0.10` and `moe_miss_shed 0.90` used temperature 0.6 and top-p 0.95 and wrote 10.2k tokens at 3.8 tok/s.

That generation ran clean and drew one of the strongest scenes of the set, which
suggests keeping the full pair and its 13% at a slightly lower
temperature on this model.

## Four shed levels on Kimi-K3

These are the pages from the four runs in the [Kimi-K3](#kimi-k3) table.

<a href="../assets/perf/kimi-k3-ud-q2kxl-car.html"><img src="../assets/perf/kimi-k3-ud-q2kxl-car.png" alt="lossless: film-grain dusk scene, red sedan with a headlight cone, telegraph poles, layered hills and clouds"></a><br>The lossless run with ranked prestage wrote 23.7k tokens at 1.15 tok/s.

<a href="../assets/perf/lossy-kimi-k3-shed-0.80.html"><img src="../assets/perf/lossy-kimi-k3-shed-0.80.png" alt="miss-shed 0.80 with keeper prestage: bright daylight scene with green fields, mountains and sun, red car with slightly misdrawn body panels"></a><br>The run with `moe_miss_shed 0.80` and keeper prestage wrote 24.2k tokens at 1.19 tok/s.

<a href="../assets/perf/lossy-kimi-k3-shed-0.70.html"><img src="../assets/perf/lossy-kimi-k3-shed-0.70.png" alt="miss-shed 0.70 with keeper prestage: complete but very dark dusk scene, red car with headlights on a dim road, foreground trees as blurred dark shapes"></a><br>The run with `moe_miss_shed 0.70` and keeper prestage wrote 29.6k tokens at 1.33 tok/s.

<a href="../assets/perf/lossy-kimi-k3-shed-0.65.html"><img src="../assets/perf/lossy-kimi-k3-shed-0.65.png" alt="miss-shed 0.65 with keeper prestage: vivid layered sunset with poles, fence and birds, red car with oversized featureless black wheels and a light streak across the body"></a><br>The run with `moe_miss_shed 0.65` and keeper prestage wrote 28.3k tokens at 1.39 tok/s.

All four pages ran as generated, with valid markup, a working animation
loop and no stray tokens. What varies is the scene, and it does not worsen
steadily as P falls. At 0.80 the car body has small geometry glitches, at
0.70 the page renders far darker than its palette intends, and at 0.65 the
car is the most damaged subject. The flaws differ in kind rather than
degree, so a single sample for each setting cannot rank adjacent levels.

## Lossless setting measurements

These numbers back the table in
[The lossless settings](../streaming.md#the-lossless-settings). All are
alternated A/B medians unless noted.

| Setting | Model and machine | Without | With |
|-------|-------------------|---------|------|
| Prefill feeder, short prompt | MiniMax-M2 Q5_K_M 162 GB, M3 Max 128 GB, 53-token prompt | 19.4 s to first token | 11.4 s |
| Decode feeder | Same model and machine, 512-token generation | 2.4 tok/s page cache, 3.0 tok/s `--stream-cpu` | 4.0 tok/s average, 4.7 steady at 90% arena hits |
| Arena token split, second-turn prefill | Kimi-K3 UD-IQ2_XXS, M5 Max 128 GB, 48-token turn | 0.25 tok/s | 2.13 tok/s |
| Weight pin | Kimi-K3 UD-IQ2_XXS 662 GB, 62 GB every-token set, M5 Max 128 GB | 0.10 tok/s decode, 0.62 prefill | 0.38 decode, 0.97 prefill |
| Pin excludes converted tensors | HY4 preview release, F32 output head held as bf16 | 22.6 GB pinned | 19.7 GB pinned, 3.5% fewer expert bytes per token |
| GPU keep-warm | GLM-5.2 UD-IQ3_XXS, arena 70 GB, miss-shed 0.85, lookahead off | 2.51 tok/s | 3.64 tok/s |
| GPU keep-warm | Hy3 IQ4_XS, layer-shed 0.10 with miss-shed 0.90 | 4.01 tok/s | 5.29 tok/s |
| Streamable lookup table | `qwen4exp` Q6 169 GB, short context | 8.4 tok/s, 106 GB wired | 12.6 to 13.4 tok/s, 54 GB wired, converging at 16k depth |

Lookahead prestage recall of the next layer's actual top-k is about
78% on GLM-5.2 at 8 experts and MiniMax-M3 at 4, against about 35% for
reusing the previous token's routing.

Keep-warm changes neither stall time nor arena hit rate, since the disk
does the same work. The gain is clock residency, so keep-warm helps a
streamed model whose per-token time sits in the eval and sync bucket
rather than in stalls.

Without the weight pin, the every-token weights are plain file-backed
pages, which the kernel evicts between uses on a machine at its free-page
minimum. Each token then faults the whole set in again, which shows as
compute time rather than stall time. The symptom is a per-token decode
time of at least the every-token bytes divided by the SSD bandwidth,
whatever the arena hit rate.

## Decode feeder defaults

The decode feeder's defaults rest on these measurements. All are
from the M3 Max with 128 GB unless the row says otherwise, and each was an
A/B on the setting alone.

| Default | Measurement |
|---|---|
| Gathers submitted off the main thread | Submitting off the main thread saved 8 ms of GPU wait per token over the graph-building thread. |
| Fast-disk recipe on `auto` | With the arena seeded from the prefill ring, a 5.7 GB/s drive gained 7% decode throughput on the fast recipe. An M5 Max drive reads 14 GB/s. |
| No background arena seeder | A seeder filling empty slots from a fast drive seeded 1295 slots over 1000 tokens and cost 0.1 tok/s. |
| Demand reads land in a bounce buffer | A `pread` straight into the Metal-shared slot saved 7 ms per token of read wait and cost 18 ms per token of GPU time on the gathers. |
| Expert stacks unmapped after install | With the GPU-visible total past physical RAM, each large command buffer paid 50 to 130 ms of driver time before it ran. |
