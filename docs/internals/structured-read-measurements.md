# Structured read measurements

The timings behind [Structured reads](structured-reads.md) and the
[Letter readout](letter-readout.md) show where a `/v1/systemone` decision's
time goes on one machine, and how the model, the request options and a
question's wording change the answers. The commands in [Setup](#setup)
measure both again.

- [Setup](#setup)
- [Prefill](#prefill)
- [Reads and samples](#reads-and-samples)
- [Steps and unembedding](#steps-and-unembedding)
- [Extension and thoughts](#extension-and-thoughts)
- [Whole decisions](#whole-decisions)
- [Question wording](#question-wording)
- [Accuracy](#accuracy)
- [Thinking on mixed requests](#thinking-on-mixed-requests)
- [Letter readout timings](#letter-readout-timings)
- [Letter readout accuracy](#letter-readout-accuracy)

## Setup

| Item | Value |
|------|-------|
| Machine | The bench ran on an Apple M3 Max with 128 GB on macOS 26.6. |
| Model | The model is diffusiongemma-26B-A4B-it Q4_K_M, a 16.8 GB file, and the Accuracy and Thinking on mixed requests sections also use Q8_0. |
| Letter readout model | The letter readout sections use OpenJev Q4_K_M from `openjev/openjev-GGUF`, a 16.5 GB file, with mlx-kquant 0.4.14 on macOS 26.6.2. |
| Base model | The letter readout accuracy also uses Qwen3.8-27B UD-Q6_K_XL from `unsloth/Qwen3.8-27B-GGUF`, a 25.3 GB file. |
| Canvas | The canvas is 64, the `server.systemone.canvas` default. |
| Peak memory | Peak memory reached 19.7 GB over the whole bench. |

`scripts/structured_read_bench.py` loaded the model in process and timed
each arm with a device synchronization on both sides. Every arm is the
median of five timed runs after two untimed ones, with the arm order
reversed on every other round and 15 s of idle between blocks. `pmset`
recorded no thermal or performance warning before or after. The schema is
three questions, one of each type, whose answer template is 16 tokens, so
the narrowest canvas it fits is 32.

```sh
python scripts/structured_read_bench.py diffusiongemma-26B-A4B-it-Q4_K_M.gguf
```

On any other text model, the same script times the letter readout. It runs
the prefix of a 700-word state and the tails of 1, 5 and 20 questions at
each forward size, one tail by length, 20 tails grouped and one per
forward, a prefix store and lookup, and whole decisions. Its long blocks
rest 15 s after each round, since a few minutes of steady load lower the
GPU clock on some machines. The JSON output keeps every run's time, so a
clock drop shows as a rise across rounds.

```sh
python scripts/structured_read_bench.py OpenJev-Q4_K_M.gguf
```

`scripts/structured_read_accuracy.py` holds the facts, the labeled set and
the requests behind [Question wording](#question-wording),
[Accuracy](#accuracy) and
[Thinking on mixed requests](#thinking-on-mixed-requests), and its `wording`, `labeled`
and `mixed` modes print their tables. Its `thoughts` mode prints the
thought table. Its `cases` mode prints the single answers that
[When answers go wrong](../decisions.md#when-answers-go-wrong) quotes, and the
comparison against mlx-vlm that [Question wording](#question-wording) reports.
Every read uses seed 42, so the answers repeat from run to run while the
times vary.

```sh
python scripts/structured_read_accuracy.py diffusiongemma-26B-A4B-it-Q4_K_M.gguf wording
python scripts/structured_read_accuracy.py diffusiongemma-26B-A4B-it-Q4_K_M.gguf labeled
python scripts/structured_read_accuracy.py diffusiongemma-26B-A4B-it-Q4_K_M.gguf mixed
python scripts/structured_read_accuracy.py diffusiongemma-26B-A4B-it-Q4_K_M.gguf cases
python scripts/structured_read_accuracy.py diffusiongemma-26B-A4B-it-Q4_K_M.gguf thoughts
```

The same script runs the `wording`, `labeled` and `mixed` sets on any
other model through the letter readout. On such a model, `wording` runs the
letter prompt in three layouts, with the subject only in an object state,
in a plain-text state, and named in the question. `labeled` runs the
methods `base` and `s4`, and `mixed` runs without thoughts.

```sh
python scripts/structured_read_accuracy.py OpenJev-Q4_K_M.gguf labeled
```

## Prefill

A group prefills its prompt at most once, since a later stage extends the
earlier prompt when it can. Past a few hundred tokens the prefill is the
largest part of a decision. Chunking at 512 tokens applies only to
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
to the cap in both modes. The stop rule is the one in
[One read](structured-reads.md#one-read), applied to each sample.

## Extension and thoughts

Appending tokens to a prefilled prompt for a later stage costs 12.7 ms for
one token and 33.1 ms for eight.

A 64-token thought on the bench's 472-token prompt took 2068 ms at the
median, with a range of 1341 to 2556 ms. A thought's time follows the
number of denoise steps its canvas takes to converge, at about 150 ms a
step. The `thoughts` mode wrote a 64-token thought three times for each
request in this table.

| Request | Denoise steps | Median ms |
|---------|---------------|-----------|
| The e2e ticket, one question | 48 | 6951 |
| `ticket-1`, the same ticket with five questions | 9 | 1452 |
| `ticket-5` | 11 | 1666 |
| `travel-vienna-bratislava` | 6 | 947 |
| `allergen-pad-thai` | 7 | 1061 |
| `allergen-risotto` | 8 | 1193 |

The thought on the one-question ticket ran to the cap of 48 steps, while
the same ticket with five questions stopped after 9, so the cost depends
on the whole prompt and not on the state alone.

## Whole decisions

In process, on the 472-token prompt with the three-question schema, whole
decisions took these times.

| Samples | Reads | Decision ms |
|---------|-------|-------------|
| 1 | 1 | 708 |
| 4 | 4 | 687 |
| `"auto"` | 4 | 822 |

The difference between one and four samples is within the run-to-run
spread. On this prompt `"auto"` extended to four samples, which adds a
second batched pass after the first read.

Over HTTP, `tests/e2e/run_systemone_e2e.py` measured these wall times on
the same machine. Its ticket asks one question about "Everything is down
and we have a demo at noon."

| Request | Reads | Prompt tokens | Wall ms |
|---------|-------|---------------|---------|
| The ticket, `samples` auto | 4 | 97 | 359 |
| The ticket, 4 samples | 4 | 97 | 246 |
| Four questions in two stages, one skipped | 5 | 214 | 518 |
| Twelve yes or no questions, indexed format | 1 | 269 | 395 |
| The ticket with `think: 64` | 1 | 167 | 6582 |

## Question wording

A question's wording changes the answer more than the prompt around it
does. Sixteen facts that need recalled knowledge, such as whether hummus
contains sesame or which currency Bratislava uses, were each asked as a yes
or no question and as its negation. That makes 32 reads, at seed 42 with
one sample. Half the true answers are yes, and the table counts how many
reads answered yes.

| Prompt | Right | Yes answers |
|--------|-------|-------------|
| The route's prompt, subject only in the state, such as "Does the dish usually contain sesame?" | 28 of 32 | 20 |
| The same, with the state as plain text instead of JSON | 26 of 32 | 20 |
| The route's prompt, subject named in the question | 31 of 32 | 15 |
| The state first, then the route's system text | 29 of 32 | 17 |
| The route's system prompt, then the state with the question restated | 29 of 32 | 17 |
| A chat prompt with the question alone, read at the first answer position | 31 of 32 | 15 |
| The same chat prompt under the opening paragraph of the route's system text | 31 of 32 | 15 |

A question that names its subject gains three and loses the lean toward
yes, while the system text and the order of the state make little
difference. The route keeps vLLM's prompt, and
[When answers go wrong](../decisions.md#when-answers-go-wrong) gives the
wording advice. The wrong answers in the table come from the model and not from the read. On
the same prompt and canvas, a read matches mlx-vlm's own decoder step in
every label log-probability to four decimals.

## Accuracy

A labeled set measured what each option does to the answers. It holds 38
facts that need recalled knowledge, each asked as a yes or no question and
as its negation, and 26 choice questions on centuries, currencies and
animal classes. Every question names its subject only in the state, as a
fixed question set does, and every read and thought uses seed 42.

| Method | Yes or no right | Yes answers, 38 true | Choice right | Log loss | Seconds per item |
|--------|-----------------|----------------------|--------------|----------|------------------|
| The default, `samples` auto | 66 of 76 | 40 | 23 of 26 | 0.44 | 0.29 |
| `samples: 4` | 67 of 76 | 39 | 23 of 26 | 0.45 | 0.25 |
| Full-vocabulary unembedding, 4 samples | 67 of 76 | 39 | 23 of 26 | 0.45 | 0.28 |
| `steps: 4`, 4 samples | 67 of 76 | 35 | 23 of 26 | 0.61 | 0.64 |
| `think: 64` | 71 of 76 | 35 | 25 of 26 | 0.28 | 1.60 |
| `think: "auto"`, threshold 0.8, budget 64 | 69 of 76 | 39 | 26 of 26 | 0.37 | 0.65 |
| 4 samples divided by a read on a content-free state | 59 of 76 | 49 | 20 of 26 | 0.79 | 0.34 |
| 4 samples averaged with the label order reversed | 66 of 76 | 36 | 23 of 26 | 0.44 | 0.50 |

Only a thought improves accuracy. Samples, steps and the full vocabulary
stay within one answer of the default. Dividing by a read on a
content-free state, the calibration used for few-shot classifiers, makes
the answers worse. Reversing the label order trims the lean toward yes at
twice the cost and fixes no answer.

The default's wrong answers are mostly its unsure ones, which is what
`think: "auto"` relies on. With `think: "auto"`, the route wrote a thought
for 15 of the 102 items and reached 95 right. The default reached 89, a
thought on every item reached 96, and `"auto"` took 40 percent of the time
that a thought on every item took. A few answers
stay wrong with a thought, such as sesame in pad thai, so they come from
the model's knowledge and not from the read.

A Q8_0 file of 26.9 GB gives the same result. It gets 88 right by
default, 93 with `think: "auto"` and 96 with a thought on every item, and
12 of its 14 wrong answers are also wrong on Q4_K_M. A larger quantization
does not fix the answers a read gets wrong.

## Thinking on mixed requests

The labeled set asks one question per request. Twenty-eight requests built
on the example questions in
[Structured decisions](../decisions.md#examples) and five support tickets with five
questions each show what `think: "auto"` does on requests with several
questions. Each request was decided with `think: 0` and with `think: "auto"`
at seed 42, and the Q4_K_M times are the mean of one run before and one
after the Q8_0 run.

| Quantization | Requests that thought | Mean with `think: 0` | Mean with `"auto"` | Mean of a request that thought | Answers changed |
|--------------|-----------------------|----------------------|--------------------|--------------------------------|-----------------|
| Q4_K_M | 4 of 33 | 0.36 s | 0.59 s | 2.4 s | 1 |
| Q8_0 | 5 of 33 | 0.37 s | 0.66 s | 2.4 s | 3 |

Every request that thought had two to five questions, and none of the 21
single-question requests did, because one unsure answer runs the whole
decision again. The thought raised the euro for Bratislava from 0.42 to
0.99, and it also turned gluten in risotto alla milanese from no at 0.68
to yes at 0.98, so a confident answer after a thought is not proof. The
slowest request that thought took 3.0 s.

## Letter readout timings

On OpenJev, a decision's time is the sum of its forwards. The state here is
a 710-token support thread, and the questions are a random mix of the
three types with three or four options. The forward sizes ran in the order
128, 64, 128, 256, 128, 64, 128, 256, 128, and the table gives the
medians.

| Forward size | Prefix of 710 tokens, s | Longest forward, ms | Tails of 1, 5 and 20 questions, s |
|--------------|-------------------------|---------------------|-----------------------------------|
| 64 | 4.73 | 450 | 0.46, 2.17, 8.88 |
| 128 | 4.34 | 760 | 0.44, 1.95, 7.98 |
| 256 | 4.22 | 1420 | 0.44, 1.87, 7.53 |

The longest forward is how long a chat stream on the same model waits
between two tokens while a decision runs. At 128, a decision is within 3 to
6 percent of the speed at 256 and holds chat up for half as long. At 64, it
is 9 to 11 percent slower to save about 300 ms per wait. The reader uses
128.

One forward of a single tail on the same prefix took these times. Up to 32
tokens, the Q4_K matrix products take their small-batch route, and above
that the 64-row tile sets the cost, so a forward of 33 tokens costs as
much as one of 64.

| Tokens | 8 | 16 | 24 | 32 | 33 | 40 | 48 | 64 | 96 | 128 |
|--------|---|----|----|----|----|----|----|----|----|-----|
| ms | 105 | 186 | 250 | 250 | 405 | 410 | 415 | 425 | 690 | 715 |

Twenty tails on the same prefix took 7.9 to 9.2 s in 11 grouped forwards,
and 9.5 to 10.2 s in 20 forwards of one tail each, over five alternating
runs whose times rose as the machine warmed. Keeping a new 710-token
prefix in the checkpoint tier took 27 to 52 ms, 29 ms at the median, and
finding it again took 6 to 9 ms.

The mixed set of [Thinking on mixed requests](#thinking-on-mixed-requests)
took 1.43 s per request on OpenJev, from about 1.0 s for one question to
2.6 to 3.1 s for the five-question tickets. DiffusionGemma took 0.32 s per
request without thoughts in the same bench run.

## Letter readout accuracy

The labeled set of [Accuracy](#accuracy) ran through the letter readout on
OpenJev and on the Qwen3.8-27B that OpenJev was trained from, beside a new
run of DiffusionGemma.

| Model and method | Yes or no right | Yes answers, 38 true | Choice right | Log loss | Seconds per item |
|------------------|-----------------|----------------------|--------------|----------|------------------|
| DiffusionGemma Q4_K_M, the default | 66 of 76 | 40 | 23 of 26 | 0.44 | 0.27 |
| DiffusionGemma Q4_K_M, `think: "auto"` | 69 of 76 | 39 | 26 of 26 | 0.37 | 0.51 |
| OpenJev Q4_K_M, one option order | 75 of 76 | 39 | 26 of 26 | 0.05 | 0.82 |
| OpenJev Q4_K_M, `samples: 4` | 75 of 76 | 39 | 26 of 26 | 0.05 | 2.13 |
| Qwen3.8-27B UD-Q6_K_XL, one option order | 74 of 76 | 38 | 26 of 26 | 0.12 | 0.80 |

OpenJev missed only whether tabbouleh is usually free of wheat, at 0.55.
The base model missed two, each at 0.52, and its log loss is more than
twice OpenJev's, since the calibration of the letter readout comes from
OpenJev's training. Four option orders changed no answer and cost 2.6
times as long. On the wording set, OpenJev answered all 32 reads right in
each layout with 16 yes answers, and the base model answered 30, 30 and
31.

