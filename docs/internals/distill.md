# Distillation internals

The teacher pass and the training head of `gmlx distill` keep their memory
bounded in ways the code enforces, and a set of measurements sits behind
the guide's defaults. The user guide is [Distillation](../distill.md),
the flags are under [gmlx distill](../distill-reference.md), and every GB
here is decimal.

- [The teacher pass](#the-teacher-pass)
- [The training head](#the-training-head)
- [The worked run](#the-worked-run)
- [Why the defaults are what they are](#why-the-defaults-are-what-they-are)
- [The cross-tokenizer result](#the-cross-tokenizer-result)

## The teacher pass

The trunk runs over a chunk of `--trunk` tokens and keeps the hidden
states, which are small. The head then runs over those states in
sub-chunks of `step` positions, each of which materializes the logits
across the teacher's whole vocabulary. `step` is the largest size
whose logits and reduction temporaries fit under `--logits-cap-gb`.
`gmlx/distill/teacher.py` holds the budget per vocabulary element.

Every pass measures its first sub-chunk's peak. When the measured
bytes exceed the budget, they replace it, and the step is derived again
against the unchanged cap. When the step changes, the sub-chunk runs again
from the same start at the new step to confirm that the peak fits. A
second miss refuses the pass with the measured figure in the message.

Some families scale their logits after the projection, and the head
carries that scale. To catch a scale the head does not carry, the head
runs beside the model's own forward on a few tokens before the pass,
and again before `train` and `eval` score anything. A difference refuses
the run, so wrong logits never reach the cache.

A streaming MoE teacher reads its expert stacks again on every forward,
so its trunk chunk defaults to a larger size, which divides that traffic.
The manifest's `throughput` block records the forwards, the bytes
read and the stream bandwidth that the pass saw.

## The training head

The student's head is fused into the loss and runs over the gathered
positions in chunks of `--chunk` positions, with a closed-form backward.
A chunk's memory scales linearly with `--chunk` and with the student's
vocabulary, and a chunk that holds tokenizer boundaries
holds more than one that does not.

`train` compares the closed form with the gradient of the head's own
forward on a few tokens, with the head in training mode. A student where
the two differ is refused, so a head that changes its input before the
projection, such as a Hadamard-folded head, never trains on a wrong
cotangent.

No head pass runs inside the trunk's gradient transform. MLX keeps
every intermediate of a transform alive until the outer evaluation, so a
head inside it would pin every chunk's logits at once. The trunk
forward runs first, the head pass computes the loss and the cotangents of
the gathered hidden states outside any transform, and a surrogate loss
carries those cotangents back through the trunk.

The trunk therefore runs twice per step. Both forwards are seeded with the
step's seed right before they run, so LoRA dropout draws the same
mask in both and the cotangents land on the hidden states they came from.
A checkpointed layer replays its seed in the backward recompute. The
replay evaluates an array, which a compiled step cannot do, so
`gmlx train` refuses dropout with checkpointing.

## The worked run

The guide's worked task used a Qwen3.6-27B teacher at UD-Q8_K_XL, a
Qwen3.5-9B student at Q6_K, 615 training questions plus 264 combined
ones, and two rounds.

| Step | Memory | Time | Disk |
|---|---|---|---|
| `gen` | The served teacher takes 36 GB. | The teacher writes 52 tokens per second at `--concurrency` 8, so the training prompts take about 4 hours. | The replies take a few MB. |
| `cache` | The teacher takes about 40 GB, a few GB over its served size. | The teacher scores about 500 tokens per second. | Each position takes 1.6 KB at top-k 256. |
| `align` | Only the tokenizers load. | A conversation row takes about 25 ms on the CPU, and a plain-text row under 1 ms. | The output takes a few MB, unless `--materialize` writes the batch tensors too. |
| `train` | The 9B student peaks at 50.7 GB. | A step of 3 rows takes 9.3 s, so 678 steps take 1.75 hours. | Two checkpoints go under `--ckpt-dir`. |
| `eval` | Only the student loads. | Each slice takes minutes, longer with the adapter attached. | Two report files are written. |

These pass rates came out of the run, with the teacher and student on one
tokenizer:

| Pass rate on | Untouched student | After round one | After round two |
|---|---|---|---|
| Held-out questions of the trained kinds | 0.022 | 0.817 | 0.882 |
| Questions of kinds never trained on | 0.017 | 0.917 | 0.925 |
| Combined held-out questions | 0.000 | 0.767 | 0.783 |

With the schema pasted into its prompt, the untouched student scored
0.946 on the held-out questions, so the adapter reached most of what
pasting the document gives. At the positions the document moved, the
student's nats per token fell from 4.41 to 0.70, against the teacher's
0.41 with the schema in view. The `--chat-sanity`, `--chat-slice` and
`--tasks` measures of `eval` did not move outside their noise.

## Why the defaults are what they are

The guide's settings come from the same schema task, measured by the
served pass rate on held-out questions. Each figure is one served sample
of one adapter. Serving the same adapter again moves a pass rate by three
or four items in a hundred, so differences of that size are sampling.

Combined training rows, which join two kinds of question in one prompt,
raised the pass rate on kinds never trained on from 0.633 to 0.917. Rows
that described the schema in prose instead of querying it lowered that
rate. The second round, in which the student writes the replies and the
teacher is cached over its verified ones with the schema in view, added
about six and a half points on the trained kinds over round one. The
other two slices moved by less than the sampling spread.

The loss defaults are `--dk 1 --alm 1 --ce 0`. A cross-entropy objective
on the teacher's tokens scored level with the sparse KL on the task
pass rates and behind it on every retention measure, so `--ce` stays at 0.
The hidden-state term, `cache --hidden` with `train --hs`, changed neither
the logit terms nor the served pass rates on this task, so it is off by
default.

Whole-reply bits per byte moves by a few thousandths when one
position in twenty gains a nat, which is inside the noise of a run, so
`eval --reply-positions` restricts the reply slice to the positions the
census found.

The loss has no per-token weighting. The tokens that decide a tool call
are its opener and closer, the key names and the tool name, and they are
the most certain positions of a reply. They sit at rank 1 in the cache
with the whole mass, so the student already sees them at full weight.

## The cross-tokenizer result

Aligned onto gemma-4-12b-it at Q6_K from another tokenizer family and
served with thinking off, the same schema cache gave an adapter that
reached 0.296 on the held-out questions against 0.930 with the schema
pasted into its prompt. That is about a third of the gap. It reached 0.050
on the kinds never trained on, where the same-tokenizer student reached
0.925.

The adapter answered the single-table questions and failed the joins on
column names that the schema does not have, so the alignment carried the
replies' shape and only part of the document. The alignment
statistics read an own-group fraction of 0.83, a singleton fraction of
0.20 and a shared-boundary fraction of 0.47. At the positions the document
moved, the student's nats per token fell from 8.21 to 0.80.
