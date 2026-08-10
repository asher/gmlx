# Speculative decoding under continuous batching

How the server runs MTP speculation and continuous batching together: the
two decode loops, the batch-width cap, and the preempt + resume mechanics
that move requests between them without interrupting any token stream.

For what speculation is and how to enable it, see
[performance.md](performance.md#mtp-speculative-decoding). For the width-cap
config key, see
[server-config.md](server-config.md#speculative_width_cap).

## The two decode loops

A speculative generation runs in one of two loops, chosen by live batch
width:

- The scalar loop (one request decoding). The fastest path: draft and
  target sampler RNG streams are kept coupled, which lets sampled drafts be
  accepted against sampled targets and yields the highest acceptance rates.
  This loop serves the common case of a single stream decoding at full
  speculative speed.
- The batch loop (two or more requests). Tracks per-row state (bonus token,
  KV offset, budget, finished flag), drafts greedily (coupled RNG does not
  extend across rows), and checks a per-model width cap: a batch wider than
  the cap decodes plain, because verification widens every row's weight
  reads and past a measured knee the batch is faster without drafting.

New requests join a running batch between verify rounds: the loop drains an
injection queue, extends the target KV cache and the drafter with the new
rows, and the width cap is re-checked against the widened batch.

## Preempt: joining a scalar generation

The scalar loop has no injection boundary; its speed comes from not being a
batch. Historically that meant a prefilled request arriving while a scalar
speculative generation streamed had to wait for the incumbent to finish
before starting its own decode. The wait is wrong on both axes: the waiter's
time to first token stretches to the incumbent's remaining generation, and
aggregate throughput loses too, because a single speculating stream is
slower than the same hardware decoding several streams plain.

So the server preempts. When waiters queue against a live scalar
speculative generation:

1. The scalar generator is closed at its verify-round boundary. Its cleanup
   path rolls the target KV cache back to exactly the delivered tokens, so
   the boundary state is clean by construction: the next undelivered token
   (the round's bonus token) has no KV entry yet.
2. The generation is rebuilt as a batch-loop generator, restarting from that
   bonus token with its real emitted count, but unarmed: no drafter state,
   no captured hidden. Single-sequence caches are lifted to their batch
   classes on the way.
3. The rebuilt loop's first injection drain admits the waiters. If the new
   width exceeds the cap the batch decodes plain (the common case: any
   second stream trips a cap of 1); otherwise the batch arms itself with a
   capture round (below) and keeps speculating at the new width.

The incumbent's stream continues without a gap. Its rate steps down from
solo-speculative to shared-plain while the batch is wide, which is the
correct trade: total tokens per second across streams goes up.

`GMLX_MTP_PREEMPT=0` restores the old behavior (waiters hold until the
scalar generation drains).

## Resume: re-arming a drained batch

A batch gated to plain decode used to stay plain for the generator's life,
even after finishing rows brought it back under the cap. That latch existed
because re-arming a drafter mid-flight needs fresh hidden state and
shared-KV for every surviving row, and reusing stale per-row state was the
crash seam of an earlier campaign.

The resume path re-arms without touching stale state, by re-running the
generator's own cold-start sequence on fresh captures:

1. When a gated batch drains to the cap or below, the loop first finishes
   consuming its plain-decode double buffer. Gated rounds dispatch the next
   round's forward before reading this round's tokens, and that dispatched
   step has already appended its KV; discarding it would corrupt the cache,
   so one more plain round runs without dispatching a successor.
2. The next round is a capture round: a one-position verify forward of each
   row's pending bonus token, with hidden-state and shared-KV capture on.
   This emits one token per row at plain-decode cost.
3. The drafter is reset and cold-started from the capture: drafters that
   teacher-force a prompt seed from target hidden accept the one-token
   capture (draft quality ramps back over the next rounds), and shared-KV
   drafters get their view re-set from the verify capture through the same
   round tail every armed round uses.
4. Subsequent rounds speculate normally at the drained width.

Rows within a small remaining-budget threshold are not worth the capture
cost and finish plain instead. A new admission landing in the same round
wins over a pending resume: the injection drain runs first and re-trips the
gate, so a batch never arms over the cap.

`GMLX_MTP_RESUME=0` restores the one-way latch.

## Semantics and caveats

- Token streams are continuous across every transition. Preempt restarts
  from the exact rollback boundary; resume consumes the plain lookahead
  before capturing. Nothing is skipped, re-emitted, or re-sampled.
- A preempted request decodes under batch-loop semantics for the rest of
  its generation, including after the batch drains back to a single row:
  greedy drafting instead of the scalar loop's coupled sampling, which
  costs a few points of acceptance at temperature. The next request starts
  scalar again.
- A preempted request drops its prompt-cache retirement context: its prefix
  is not offered back to the APC when it finishes. Waiters and later
  requests retire normally.
- The capture round emits at plain-decode rate; the speculative speedup
  returns on the round after. Resumes are therefore paced by the
  remaining-budget threshold rather than fired for nearly-done rows.

## Longer plays

Two designs that would raise the width caps themselves rather than manage
around them. Documented here for a future pass; neither is built.

### Ragged mixed verify forward

Today every row in a verify round carries the same draft depth, so the
verify forward is a rectangle: batch width times block size. Rows with cold
drafters (fresh joins, fresh resumes) waste verify positions on drafts that
will not be accepted, and MoE targets pay the expert union of every
position in the rectangle.

A ragged verify would give each row its own draft length, packing the
forward as one variable-length sequence batch (the runtime already has
ragged prefill machinery). The MoE win is the interesting one: expert
gather cost scales with the union of experts touched, so trimming wasted
positions trims real bandwidth, and the width-2 loss that currently caps
MoE targets at 1 was measured with rectangular verify. A ragged forward
re-opens that measurement.

### Tree verify

The caps encode a linear-draft trade: each drafted position must beat plain
decode for every row. Verifying a token tree per row instead of a chain
raises acceptance per verify forward (multiple continuations share a
prefix), which shifts the knee outward: the batched verify does more useful
work per unit of bandwidth, so speculation stays profitable at widths that
lose today. This changes the B > 1 verify arithmetic (attention masks over
tree positions, per-row acceptance walks over branches) and the drafter
contract (emit branching drafts), so it is a program, not a patch. The
scalar loop would gain too, but the batch knee is where the cap lives.
