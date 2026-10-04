# Distillation troubleshooting

Each `gmlx distill` step prints progress lines and can write a report. This
page says which figures matter and what to do when a step fails or a result
is poor. The steps themselves are in the [walkthrough](distill-walkthrough.md).

- [The figures](#the-figures)
- [Training lines](#training-lines)
- [The eval report](#the-eval-report)
- [The filter dropped most rows](#the-filter-dropped-most-rows)
- [The checker fails](#the-checker-fails)
- [align warns about the tokenizers](#align-warns-about-the-tokenizers)
- [align drops rows](#align-drops-rows)
- [The census effect is small](#the-census-effect-is-small)
- [The loss does not fall](#the-loss-does-not-fall)
- [The validation loss rises](#the-validation-loss-rises)
- [The pass rate is low](#the-pass-rate-is-low)
- [A step runs out of memory](#a-step-runs-out-of-memory)
- [port 8093 already has a listener](#port-8093-already-has-a-listener)
- [train or eval refuses its input](#train-or-eval-refuses-its-input)

## The figures

- A pass rate is the share of held-out questions the checker accepts. It
  is the one figure that says whether the task works. Served twice, the
  same adapter's pass rate moves by three or four in a hundred, so smaller
  differences between adapters mean nothing.
- A loss is what training drives down. Only its trend matters.
- Nats per token is the student's average surprise at the tokens the
  teacher wrote. A token the student gave probability 0.37 costs one nat.
  Lower is better, and zero means the student would have written the same.
- Bits per byte, `bpb` in the tables, is the same surprise per byte of
  text, so slices with different tokenizers compare.

## Training lines

`train` prints a line every 10 steps, and a validation line every 200 steps
and at the last step:

```text
[train] it 200 loss 0.0465 dk 0.0332 alm 0.0134 ce 0.1682 floored 0 lr 1.43e-05 260 tok/s step 8700 ms load 100 ms peak 49.7 GB active 13.5 cache 8.0
[train] it 200 val 0.0806
```

- `loss` should fall through the first third of the run, then flatten.
- `val` is the loss on the rows `align` held back. Later validation lines
  add the best earlier value in parentheses.
- `peak` is the most memory the run has used, in GB.

`val none` means no held-back row had a position to score. The other
fields are described in [Distillation internals](internals/distill.md#the-train-line).

## The eval report

`eval` writes one Markdown table per kind of measurement. Most tables have
one row per slice. This is the walkthrough's reply table:

| `reply slice` | `bpb after` | `bpb before` | `nats/token after` | `nats/token before` | `se` | `rows` |
|---|---|---|---|---|---|---|
| heldout | 0.2225 | 1.4131 | 0.6946 | 4.4116 | 0.0056 | 173 |

`after` columns score the student with the adapter, and `before` columns
without it, filled in by `--before`. `se` is the standard error of
`bpb after`. Two adapters whose bpb differs by less than twice the `se`
cannot be told apart. Compare `nats/token after` with the teacher's own
figure, the `with` value on the census row
`teacher nats per token on high-delta positions`.

| Table | What it shows |
|---|---|
| `slice` | Plain-text slices. `gate` reads `void` when over 1% of the slice is in the training corpus, so its score does not count. |
| `chat slice` | Conversations, scored on their assistant turns. |
| `task` | Accuracy on the local benchmark task files. |
| `chat sanity` | `compliance`, the share of replies that kept the turn structure, and how often the student refused. |
| `kld vs cache` | The distance from the teacher's stored choices, and `top-1`, how often both pick the same token. |

A reply table that shows `None` scored no row. Every row was longer than
`--chat-max-len` or had nothing to score.

## The filter dropped most rows

The rejects file gives one reason per dropped row:

| Reason | Meaning | Fix |
|---|---|---|
| `length` | The reply did not reach its end of turn. | Raise `--max-tokens` on `gen`. |
| `budget` | The reasoning hit `--thinking-budget`. | Raise the budget, or drop `--thinking` and `--thinking-budget`. |
| `empty` | The answer has fewer than `--min-words` words. | Use `--min-words 1` for short answers. |
| `marker` | A chat template marker leaked into the reply. | None, the reply is unusable. |
| `repeat` | Lines or phrases repeat in the reply or its reasoning. | None, the reply is unusable. |
| `ascii` | Too many non-ASCII characters. Only with `--max-non-ascii`. | Raise `--max-non-ascii`, or leave it off. |
| `tokens` | The reply is over `--max-reply-tokens`. | Size it as the [walkthrough](distill-walkthrough.md#round-one) shows. |
| `verify` | Your checker rejected the reply, with its word under `detail`. | See below. |

Many `verify` rejections mean the teacher gets the task wrong even with the
document in view. A teacher that fails most of a task cannot teach it. Try a
larger teacher, or make the questions and the document clearer.

## The checker fails

`filter` exits 2 when the `--verify` command fails. The command must exit 0
and print exactly one line for each row it reads. Run it by hand on a few
rows: `head -3 r1-replies.jsonl | ./check-sql.py freight.sqlite`.

## align warns about the tokenizers

`align` prints `[align] warn:` when the own-group fraction `a` is under
0.90 or the singleton fraction `s` is under 0.50. The two tokenizers split
text differently, and part of the teacher's output has no direct target in
the student. The run still works, with less to learn from.

With `a` under 0.70, `align` refuses, writes no view and exits 3. Pick a
student from the teacher's family, or pass `--force` and expect a weak
result.

## align drops rows

The `failed to render or pair` line counts rows that the student's chat
template renders differently from the teacher's, and names the first one.
A template that refuses a turn type, such as a tool turn, drops every row
that has one. Pick a student whose template renders the same turns, or
leave those rows out of the corpus.

## The census effect is small

The document changes little of what the teacher says on these prompts.
Check that `gen` ran with `--context`, that the questions need the
document, and that the second `cache` used `--messages-key student_messages`.

## The loss does not fall

If the loss has not fallen by step 40, stop the run. Check that `filter`
kept the rows you expected and that `align` printed `a=1.000`, or a
warning you accepted.

## The validation loss rises

A `val` that rises while `loss` keeps falling means the adapter memorizes
the training rows. Run again with fewer steps or a lower `--lora-rank`.

The adapter file holds the last step. The `best` checkpoint, the one with
the lowest `val`, stays under `--ckpt-dir`, but no action turns it into an
adapter.

## The pass rate is low

A pass rate near zero after training usually has one of three causes:

- The student was served without `--adapter`, or with thinking off.
- The checker cannot parse the student's replies. Read a few by hand.
- The task is beyond the student. The untouched student with the document
  pasted in then scores low too, and no adapter fixes that.

When the pass rate is low but not zero, check the census effect first. A
small effect caps what any adapter can learn. Then add prompts and
phrasings for the kinds that fail, including combined questions, and train
with more steps. If the rate stays low, use a larger student.

## A step runs out of memory

For `train`, use `--batch-size 1` or `--grad-checkpoint`. Both are slower.
A shorter `--max-len` on `cache` makes shorter rows, but drops the replies
that no longer fit.

For `cache` and `gen`, try a smaller quantization of the teacher. `cache`
refuses a dense teacher larger than the memory it can wire. It streams the
experts of a mixture-of-experts teacher from disk when they do not fit, and
`gmlx validate` says whether a file is one.

The document is in the teacher's prompt on every row. A long document needs
a larger `--max-len`, and memory grows with it. To split one, give each
prompt row the section it needs in its own `context` field.

## port 8093 already has a listener

`gen` refuses to start while another server holds its port. A `gen` that
was killed can leave its own server running there, and the message names
the command that stops it.

## train or eval refuses its input

- A refused view means its cache changed, or the student is not the one
  the view was aligned for. Run `align` again.
- A refusal that names fewer train rows than `--batch-size` means the
  corpus is too small for that batch.
- `eval` refuses `--reply-positions` when the census JSON names none of the
  reply rows. Run `census --corpus` with the same file the reply slice
  reads.

Every action's exit codes are in the
[distillation reference](distill-reference.md#exit-codes).
[Troubleshooting](troubleshooting.md) covers problems outside this
pipeline.
