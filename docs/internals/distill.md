# Distillation internals

The `gmlx distill cache` head sub-chunk and the training head's live set
follow simple memory arithmetic, and a set of measurements sits behind the
guide's defaults. The user guide is [Distillation](../distill.md), the
flags are under [gmlx distill](../cli.md#gmlx-distill), and every GB here
is decimal.

- [The teacher pass](#the-teacher-pass)
- [The training head](#the-training-head)
- [The worked run](#the-worked-run)
- [Why the defaults are what they are](#why-the-defaults-are-what-they-are)
- [The cross-tokenizer result](#the-cross-tokenizer-result)

## The teacher pass

The trunk runs over a chunk of `--trunk` tokens with rows stacked on the
batch axis, and keeps the hidden states, which are small. The head then
runs over those states in sub-chunks of `step` positions, and each
sub-chunk materializes a `[step, V]` logits array with V the teacher's
vocabulary width. The reduction per sub-chunk is a float32 log-softmax, a
full-width index from the top-k selection, and one temporary at a time
for the tail mass and the boundary mass. Those run as separate steps, so
at most one full-width temporary is live beyond the log-softmax. That is 14
bytes per V-element, budgeted as 16 without `--floor` and 20 with it. The
sub-chunk's logits are cast to bfloat16 before the reduction whatever the
activation dtype, which is what that budget counts, and the stored top-K
values are float16 in any case.

Some families scale their logits after the projection, and the head
carries that scale beside the softcap. Granite divides by
`logits_scaling`, Cohere multiplies by `logit_scale`, Muse Glimmer
multiplies by `output_multiplier` and MiniCPM scales the hidden states.
As a check that nothing else was missed, the head runs beside the
model's own forward on eight tokens before the pass, and again before
`train` and `eval` score anything. A difference in the
logits refuses the run, so a model that changes its logits in a way the
head does not carry never reaches the cache.

`step` is the largest halving tier of 4096 positions with `step * V * bytes`
under `--logits-cap-gb`, from `gmlx.gen.prefill_plan`. The first sub-chunk
of every pass runs after a peak-memory reset, and the measured bytes per
V-element replace the budget when they exceed it. The step is then
re-derived against the unchanged cap and a second sub-chunk confirms the
peak fits. A second miss refuses the pass with the measured constant in the
message, and both constants land in `progress.json` and the manifest. Bytes
per V-element is a ratio of the head's own live set, so shrinking the cap
alone could never change what the probe measures.

The trunk chunk is bounded by attention activations and by the headroom
`gmlx.gen.prefill_decay` reports after the buffer cache is cleared, with a
sticky shrink and no widening. A streaming MoE teacher re-reads its expert
stacks on every forward, so its trunk chunk defaults to 8192 tokens, which
divides that traffic by sixteen against a 512-token chunk. The manifest's
`throughput` block records forwards, the bytes read and the stream
bandwidth the pass saw.

## The training head

The student's head is fused into the loss and runs over the gathered
compute positions in chunks of `--chunk` positions, with a closed-form
backward and the head parameters passed explicitly. A boundary chunk holds the
float32 logits, the softmax, one logsumexp temporary, the slot map over
the projected groups and the gather of every student token's slot, then
the cotangent in the backward. That is 24 bytes per position and
vocabulary element on a cross-tokenizer pair and 16 on a same-tokenizer
pair. A chunk with no boundaries holds 12. At the default 512 positions
and a 262144-token student vocabulary that is 3.2 GB, which `--chunk`
scales linearly.

This closed form takes the cotangent of the logits back to the hidden
states through the dequantized head weight. A Hadamard-folded head
rotates its input before that weight, so the closed form also takes the
cotangent back through the rotation, in the MLX-op form that has a
backward. Beside the logit check described under the teacher pass,
`train` compares the closed form with the gradient of the head's own
forward on the same eight tokens, with the head in training mode, the
form it trains in. A student where the two differ is refused, so a head
that changes its input before the projection never trains on a wrong
cotangent.

No head pass runs inside the trunk's gradient transform. The trunk
forward is evaluated first, the head pass computes the loss and the
cotangents of the gathered hidden states outside any transform, and a
surrogate loss carries those cotangents back through the trunk. MLX keeps
every intermediate of a transform alive until the outer evaluation, so a
head inside the trunk's transform would pin every chunk's logits at once.

The trunk therefore runs twice per step, once for the head pass and once
under the transform. Both forwards are seeded with the step's seed right
before they run, so LoRA dropout draws the same mask in both and the
cotangents land on the hidden states they were computed from. A
checkpointed layer draws one seed before its forward and replays it in
the backward recompute, so the recompute sees the mask the forward drew.
The draw evaluates an array, which the eager distill loop allows and a
compiled step does not, so `gmlx train` refuses dropout with
checkpointing instead of replaying.

For `--hs`, the hidden-state map draws its initial weights from its own
key, so building it at the first step of a run or a resume leaves the
run's random stream where it was.

## The worked run

The worked task of the guide used a Qwen3.6-27B teacher at UD-Q8_K_XL, a
Qwen3.5-9B student at Q6_K, 615 training questions plus 264 combined
ones, and two rounds. The whole run took about 11 hours. The census check
before it is one `gen` over the held-out prompts, under an hour, plus two
short caches.

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

The guide's settings come from a schema task on a Qwen3.6-27B
teacher at Q8 and a Qwen3.5-9B student at Q6_K, measured by the served
pass rate on held-out questions. Each figure here is one served sample
of one adapter. Serving the same adapter again moves a pass rate by
three or four items in a hundred, so differences of that size between
adapters are sampling.

Combined training rows, which join two kinds of question in one
prompt, raised the pass rate on kinds never trained on from 0.633 to
0.917. Rows that described the schema in prose instead of querying it
lowered that rate. The second round, in which the student writes the
replies and the teacher is cached over its verified ones with the schema
in view, added a few points on every slice over round one.

The loss defaults are `--dk 1 --alm 1 --ce 0`. A cross-entropy
objective on the teacher's tokens scored level with the sparse KL on the
task pass rates and behind it on every retention measure, so `--ce`
stays at 0. The hidden-state term (`cache --hidden` plus `train --hs`)
changed neither the logit terms during training nor the served pass
rates on this task, at 0.5 cosine weight against a 256-wide sketch, so
it is off by default. Whole-reply bits per byte moves by a few
thousandths when one position in twenty gains a nat, inside the noise of
a run, which is why `eval --reply-positions` restricts the reply slice
to the positions the census found.

There is no per-token weighting in the loss. The tokens that decide a
tool call are the call's opener and closer, the key names and the tool
name, and they are the most certain positions of a reply. They sit at
rank 1 in the cache with the whole mass, so a student trained on the
cache sees them at full weight already.

## The cross-tokenizer result

Aligning the same schema cache onto gemma-4-12b-it at Q6_K, a student
from another tokenizer family, gave an adapter trained with the guide's
settings and served with thinking off. It reached 0.296 on the held-out
questions against 0.930 with the schema pasted into its prompt, about a
third of the gap, and 0.050 on the kinds never trained on. The
same-tokenizer student reached 0.882 and 0.925 on the same slices.

The adapter answered the single-table questions and failed the joins on
column names the schema does not have, so the alignment carried the shape of
the replies and only part of the document. The alignment statistics for the
pair read an own-group fraction of 0.83, a singleton fraction of 0.20 and a
shared-boundary fraction of 0.47, which is where the tokenizations diverge.
At the positions the document moved, the student's nats per token fell from
8.21 to 0.80.