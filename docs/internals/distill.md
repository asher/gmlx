# Distillation memory arithmetic

How `gmlx distill cache` sizes its head sub-chunk and how the training head
bounds its live set, for contributors changing either. The user guide is
[../distill.md](../distill.md) and the flags are under
[gmlx distill](../cli.md#gmlx-distill). Every GB here is decimal.

## The teacher pass

The trunk runs over a chunk of `--trunk` tokens with rows stacked on the
batch axis, and keeps the hidden states, which are small. The head then
runs over those states in sub-chunks of `step` positions, and each
sub-chunk materializes a `[step, V]` logits array with V the teacher's
vocabulary width. The reduction per sub-chunk is a float32 log-softmax, a
full-width index from the top-K selection, and one temporary at a time
for the tail mass and the boundary mass, evaluated as separate steps so at
most one full-width temporary is live beyond the log-softmax. That is 14
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
the cotangent in the backward: 24 bytes per position and vocabulary
element on a cross-tokenizer pair and 16 on an identity pair. A chunk with
no boundaries holds 12. At the default 512 positions and a 262144-token
student vocabulary that is 3.2 GB, which `--chunk` scales linearly.

The head never runs inside the trunk's gradient transform. The trunk
forward is evaluated first, the head pass computes the loss and the
cotangents of the gathered hidden states outside any transform, and a
surrogate loss carries those cotangents back through the trunk. MLX keeps
every intermediate of a transform alive until the outer evaluation, so a
head inside the trunk's transform would pin every chunk's logits at once.
