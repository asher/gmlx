# Structured read measurements

The timings behind [structured-reads.md](structured-reads.md), for
contributors who change the read engine, the route or its prompt. They show
where the time of a `/v1/systemone` decision goes on one machine, how to
measure again, and how the request options and the wording of a question
change the answers.

- [Setup](#setup)
- [Prefill](#prefill)
- [Reads and samples](#reads-and-samples)
- [Steps and unembedding](#steps-and-unembedding)
- [Extension and thoughts](#extension-and-thoughts)
- [Whole decisions](#whole-decisions)
- [Question wording](#question-wording)
- [Accuracy](#accuracy)
- [Thinking on mixed requests](#thinking-on-mixed-requests)

## Setup

| Item | Value |
|------|-------|
| machine | Apple M3 Max, 128 GB, macOS 26.6 |
| model | diffusiongemma-26B-A4B-it Q4_K_M, a 16.8 GB file, and Q8_0 in the last two sections |
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

## Question wording

The wording of a question changes the answer more than the prompt around
it does. Sixteen facts that need recalled knowledge, such as whether hummus
contains sesame or which currency Bratislava uses, were each asked as a yes
or no question and as its negation, 32 reads in all, at seed 42 with one
sample. Half the true answers are yes, and the table counts how many reads
answered yes.

| Prompt | Right | Yes answers |
|--------|-------|-------------|
| the route's prompt, subject only in the state, such as "Does the dish usually contain sesame?" | 28 of 32 | 20 |
| the same, with the state as plain text instead of JSON | 26 of 32 | 20 |
| the route's prompt, subject named in the question | 31 of 32 | 15 |
| the state first, then the route's system text | 29 of 32 | 17 |
| the route's system prompt, then the state with the question restated | 29 of 32 | 17 |
| a chat prompt with the question alone, read at the first answer position | 31 of 32 | 15 |
| the same chat prompt under the route's system text | 31 of 32 | 15 |

The system text makes no difference on its own, and moving the state ahead
of the questions recovers one read at most. A question that names its
subject gains three and loses the lean toward yes. The route keeps vLLM's
prompt, and [decisions.md](../decisions.md#when-answers-go-wrong) gives
users the wording advice.

These errors come from the model, not from the read. On the same prompt
and canvas, a read matches the first denoise step of mlx-vlm's own
generation loop in every label log-probability to three decimals. A full
generation on the decision prompt writes the same wrong answers, such as
`q: yes` for sesame in pad thai.

## Accuracy

A labeled set measured what each option does to the answers. It holds 38
facts that need recalled knowledge, each asked as a yes or no question and
as its negation, and 26 choice questions on centuries, currencies and
animal classes. Every question names its subject only in the state, as a
fixed question set does, and every read and thought uses seed 42.

| Method | Yes or no right | Yes answers, 38 true | Choice right | Log loss | Seconds per item |
|--------|-----------------|----------------------|--------------|----------|------------------|
| the default, `samples` auto | 66 of 76 | 40 | 23 of 26 | 0.44 | 0.29 |
| `samples: 4` | 67 of 76 | 39 | 23 of 26 | 0.45 | 0.25 |
| full-vocabulary unembedding, 4 samples | 67 of 76 | 39 | 23 of 26 | 0.45 | 0.28 |
| `steps: 4`, 4 samples | 67 of 76 | 35 | 23 of 26 | 0.61 | 0.64 |
| `think: 64` | 71 of 76 | 35 | 25 of 26 | 0.28 | 1.60 |
| `think: "auto"`, threshold 0.8, budget 64 | 69 of 76 | 39 | 26 of 26 | 0.37 | 0.65 |
| 4 samples divided by a read on a content-free state | 59 of 76 | 49 | 20 of 26 | 0.79 | 0.34 |
| 4 samples averaged with the label order reversed | 66 of 76 | 36 | 23 of 26 | 0.44 | 0.50 |

Only a thought changes accuracy. Samples, steps and the full vocabulary
stay within one answer of the default. Dividing by a read on a
content-free state, the calibration used for few-shot classifiers, makes
the answers worse. Reversing the label order trims the lean toward yes at
twice the cost and fixes no answer.

The default's wrong answers are mostly its unsure ones, which is what
`think: "auto"` relies on. It wrote a thought for 15 of the 102 items and
reached 95 right, against 89 for the default and 96 with a thought on
every item, at 40 percent of the time of the full thought. A few answers
stay wrong with a thought, such as sesame in pad thai, so they come from
the model's knowledge and not from the read.

The Q8_0 file, 26.9 GB, gives the same result. It gets 88 right by
default, 93 with `think: "auto"` and 96 with a thought on every item, and
12 of its 14 wrong answers are also wrong on Q4_K_M. A larger quantization
does not fix the answers a read gets wrong.

## Thinking on mixed requests

The labeled set asks one question per request. The 28 requests in the
[decisions.md examples](../decisions.md#examples) and five support tickets
with five questions each show what `think: "auto"` does on requests with
several questions. Each request was decided with `think: 0` and with
`think: "auto"` at seed 42.

| Quantization | Requests that thought | Mean with `think: 0` | Mean with `"auto"` | Mean of a request that thought | Answers changed |
|--------------|-----------------------|----------------------|--------------------|--------------------------------|-----------------|
| Q4_K_M | 4 of 33 | 0.56 s | 0.99 s | 4.4 s | 1 |
| Q8_0 | 5 of 33 | 0.41 s | 0.73 s | 2.7 s | 3 |

Every request that thought had two to five questions, and none of the 21
single-question requests did, because one unsure answer runs the whole
decision again. Most unsure answers were on questions without one right
answer, such as a customer's tone at 0.51 or a ticket's severity, and the
thought moved them between plausible labels. It raised one right answer's
confidence, the euro for Bratislava from 0.42 to 0.99. It also made one
right answer wrong, gluten in risotto alla milanese, from no at 0.68 to yes
at 0.98, so a confident answer after a thought is not proof.

Q8_0 decisions ran about a quarter faster than Q4_K_M decisions in both
modes on this machine.
