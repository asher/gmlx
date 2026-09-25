# Structured read measurements

The timings behind [structured-reads.md](structured-reads.md), for
contributors who change the read engine or the route. They show where the
time of a `/v1/systemone` decision goes on one machine and one GGUF, and
how to measure again.

## Setup

| Item | Value |
|------|-------|
| machine | Apple M3 Max, 128 GB, macOS 26.6 |
| model | diffusiongemma-26B-A4B-it Q4_K_M, a 16.8 GB file |
| canvas | 64, the `server.systemone.canvas` default |
| peak memory | 19.7 GB over the whole bench |

`scripts/structured_read_bench.py` loaded the model in process and timed
each arm with a device synchronisation on both sides. Every arm is the
median of five timed runs after two untimed ones, with the arm order
reversed on every other round and 15 s of idle between blocks. `pmset`
recorded no thermal or performance warning before or after. The schema is
three questions, one of each type, whose answer template is 16 tokens, so
the narrowest canvas it fits is 32.

```sh
python scripts/structured_read_bench.py diffusiongemma-26B-A4B-it-Q4_K_M.gguf
```

## Prefill

The prompt is prefilled once per group, and past a few hundred tokens it is
the largest part of a decision. Chunking at 512 tokens applies only to
longer prompts, where it costs about 5 percent.

| Prompt tokens | Prefill ms | Chunked at 512, ms |
|---------------|------------|--------------------|
| 178 | 217 | 224 |
| 472 | 428 | 415 |
| 955 | 795 | 845 |
| 1900 | 1657 | 1738 |

## Reads and samples

A one-step read on a prefilled 472-token prompt costs about one decoder
pass. Samples run as one batch, so extra samples cost a fraction of a pass
each, while reading them one at a time costs a whole pass each.

| Width | Samples | Batched ms | One at a time, ms |
|-------|---------|------------|-------------------|
| 32 | 1 | 91 | |
| 32 | 2 | 100 | 180 |
| 32 | 4 | 140 | 363 |
| 64 | 1 | 96 | |
| 64 | 2 | 138 | 193 |
| 64 | 4 | 212 | 386 |

## Steps and unembedding

Constrained unembedding saves about 17 ms on a one-sample read and 59 ms on
a four-sample read, since the full-vocabulary product multiplies each slot
row by the whole embedding table.

| Read at width 32 | Constrained ms | Full vocabulary ms |
|------------------|----------------|--------------------|
| 1 sample, 1 step | 98 | 115 |
| 4 samples, 1 step | 144 | 203 |
| 1 sample, 2 steps | 198 | 248 |
| 1 sample, 4 steps | 402 | 243 |
| 1 sample, 8 steps | 776 | 229 |

Past one step, a read costs one pass per step until it converges. On the
bench prompt the full-vocabulary read converged after two steps and the
constrained read ran to the cap. A traced read on the ticket example ran
to the cap in both modes. The convergence rule is the one vLLM uses. It
takes the entropy mean and the argmax stability over every canvas
position, with the entropy over the label set in constrained mode.

## Extension and thoughts

Appending tokens to a prefilled prompt for a later stage costs 12.7 ms for
one token and 33.1 ms for eight.

A 64-token thought on the 472-token prompt took 2068 ms at the median, with
a range of 1341 to 2556 ms, and every run used the whole budget. The same
budget on the ticket example took 6.6 s end to end on the server, so the
cost depends on how many denoise steps each thought block needs.

## Whole decisions

In process, on the 472-token prompt with the three-question schema:

| Samples | Reads | Decision ms |
|---------|-------|-------------|
| 1 | 1 | 708 |
| 4 | 4 | 687 |
| `"auto"` | 4 | 822 |

A decision is the render, the prefill and the reads, so the difference
between one and four samples is within the run-to-run spread. On this prompt `"auto"` extended to four
samples, which adds a second batched pass after the first read.

Over HTTP, `tests/e2e/run_systemone_e2e.py` measured these wall times on
the same machine:

| Request | Reads | Prompt tokens | Wall ms |
|---------|-------|---------------|---------|
| the ticket example, `samples` auto | 4 | 97 | 359 |
| the ticket example, 4 samples | 4 | 97 | 246 |
| four questions in two stages, one skipped | 5 | 214 | 518 |
| twelve yes or no questions, indexed format | 1 | 269 | 395 |
| the ticket example with `think: 64` | 1 | 167 | 6582 |
