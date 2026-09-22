# Distillation internals

This page gives the memory arithmetic of the `gmlx distill cache` head sub-chunk and of the training head's live set, and the measurements behind the guide's defaults. It is for contributors changing either. The user guide is [../distill.md](../distill.md), the flags are under [gmlx distill](../cli.md#gmlx-distill), and every GB here is decimal.

- [The teacher pass](#the-teacher-pass)
- [The training head](#the-training-head)
- [Why the defaults are what they are](#why-the-defaults-are-what-they-are)
- [The cross-tokenizer result](#the-cross-tokenizer-result)

## The teacher pass

The trunk runs over a chunk of `--trunk` tokens with rows stacked on the
batch axis, and keeps the hidden states, which are small. The head then
runs over those states in sub-chunks of `step` positions, and each
sub-chunk materializes a `[step, V]` logits array with V the teacher's
vocabulary width. The reduction per sub-chunk is a float32 log-softmax, a
full-width index from the top-K selection, and one temporary at a time
for the tail mass and the boundary mass. Those run as separate steps, so
at most one full-width temporary is live beyond the log-softmax. That is 14
bytes per V-element, budgeted as 16 without `--floor` and 20 with it.

`step` is the largest halving tier of 4096 positions with `step * V *
bytes` under `--logits-cap-gb`, from `gmlx.gen.prefill_plan`. The first sub-chunk of
every pass runs after a peak-memory reset, and the measured bytes per
V-element replace the budget when they exceed it. The step is then
re-derived against the unchanged cap and a second sub-chunk confirms the
peak fits. A second miss refuses the pass with the measured constant in
the message, and both constants land in `progress.json` and the manifest.
Bytes per V-element is a ratio of the head's own live set, so shrinking
the cap alone could never change what the probe measures.

The trunk chunk is bounded by attention activations and by the headroom
`gmlx.gen.prefill_decay` reports after the buffer cache is cleared, with a
sticky shrink and no widening. A streaming MoE teacher re-reads its expert
stacks on every forward, so its trunk chunk defaults to 8192 tokens, which
divides that traffic by sixteen against a 512-token chunk. The manifest's
`throughput` block records forwards, the bytes read and the stream
bandwidth the pass saw.

## The training head

The student's head is fused into the loss and runs over the gathered
compute positions in chunks of `--chunk` positions inside a checkpoint,
with the head parameters threaded explicitly. A boundary chunk holds the
float32 logits, the softmax, one logsumexp temporary, the slot map over
the projected groups and the gather of every student token's slot, then
the cotangent in the backward. That is 24 bytes per position and
vocabulary element on a cross-tokenizer pair and 16 on an identity pair. A chunk with
no boundaries holds 12. At the default 512 positions and a 262144-token
student vocabulary that is 3.2 GB, which `--chunk` scales linearly.

The head never runs inside the trunk's gradient transform. The trunk
forward is evaluated first, the head pass computes the loss and the
cotangents of the gathered hidden states outside any transform, and a
surrogate loss carries those cotangents back through the trunk. MLX keeps
every intermediate of a transform alive until the outer evaluation, so a
head inside the trunk's transform would pin every chunk's logits at once.

## Why the defaults are what they are

The guide's recipe settings come from a schema task on a Qwen3.6-27B
teacher at Q8 and a Qwen3.5-9B student at Q6_K, measured by the served
pass rate on held-out questions. Each figure below is one served sample
of one adapter. Serving the same adapter again moves a pass rate by
three or four items in a hundred, so differences of that size between
arms are sampling.

Compositional training rows, which join two question families in one
prompt, raised the pass rate on families never trained on from 0.633 to
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
tool call, the call's opener and closer, the key names and the tool
name, are the most certain positions of a reply. They sit at rank 1 in
the cache with the whole mass, so a student trained on the cache sees
them at full weight already.

## The cross-tokenizer result

The same schema cache was aligned onto gemma-4-12b-it at Q6_K, a
student from another tokenizer family, trained with the recipe's
settings and served with thinking off. It reached 0.296 on the held-out
questions against 0.930 with the schema pasted into its prompt, about a
third of the gap, and 0.050 on the families never trained on. The same-tokenizer
student reached 0.882 and 0.925 on the same slices.

The adapter answered the single-table questions and failed the joins on
column names the schema does not have, so the projection carried the
shape of the replies and only part of the document. The census over the
pair read an own-group mass fraction of 0.83 and a shared-boundary
fraction of 0.47, which is where the tokenizations diverge. At the
census positions the student's nats per token fell from 8.21 to 0.80.
