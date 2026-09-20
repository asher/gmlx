# Distill a teacher into a student

This guide is for training a small GGUF model on the knowledge of a larger
one without running both at once, including when the two use different
tokenizers. It walks through the four `gmlx distill` actions on a local
pair and says what each artifact contains.

- [How it works](#how-it-works)
- [Cache the teacher](#cache-the-teacher)
- [Align the cache to the student](#align-the-cache-to-the-student)
- [Train the adapter](#train-the-adapter)
- [Evaluate](#evaluate)
- [Different tokenizers](#different-tokenizers)
- [Limitations](#limitations)

## How it works

Distillation trains a student to reproduce a teacher's next-token
distribution rather than a corpus's next token. Running the teacher inside
the training loop costs its memory and a forward pass on every step, so
`gmlx distill` runs the teacher once. `cache` stores, for every position of
a corpus, the teacher's top-K log-probs, the log-prob of the token that
followed, the mass outside the top-K and the mass on word boundaries. A
cache is a directory of safetensors shards with a manifest, sized at about
6K + 22 bytes per position, so 5 million tokens at K=256 take 7.8 GB.

`align` reads that cache with the student's tokenizer. When the two
tokenizers are the same it records that and nothing else is needed. When
they differ it finds the byte offsets where both tokenizations agree on a
token boundary, and at each one projects the teacher's distribution onto
groups of student tokens. `train` fits a LoRA adapter on the quantized
student against those targets, with the head fused into the loss and
computed in chunks so no full-vocabulary logits array outlives a chunk.
`eval` scores the student before and after in one process.

## Cache the teacher

The corpus is a jsonl file with a `text` field per row, a directory of
text files, or a Hugging Face dataset id, which needs the `datasets`
package. Rows are normalized to NFC and cut into windows of at most
`--max-len` teacher tokens at word boundaries:

```sh
gmlx distill cache --teacher Qwen3-8B-Q6_K.gguf --corpus corpus.jsonl --out cache/ \
    --top-k 256 --max-len 512 --max-disk-gb 20
```

The pass loads the teacher through the same loader as `run`, so a MoE
teacher larger than memory streams its experts from disk. It prints the
size estimate before loading and refuses when the estimate exceeds
`--max-disk-gb` or the free space. `--resume` continues after the last
verified shard. `--validate cache/` checks an existing cache without a
model load.

Chat data is cached with a frame. `--frame chat` reads conversations under
`--messages-key` and makes every assistant turn a target, `--frame reply`
targets only the final turn, and `--frame reply-think` starts the target
at the final turn's reasoning content. `--frame continue` places each plain
text window in a model turn behind a fixed user instruction, so an instruct
teacher sees its own template. The flag table is under
[distill cache](cli.md#distill-cache).

## Align the cache to the student

```sh
gmlx distill align --cache cache/ --student Qwen3-0.6B-Q4_K_M.gguf --out view/
```

The pass runs on the CPU with the tokenizers only. It writes `view.json`
with the row index, the train and validation split, and the census of the
projection: the fraction of the teacher's mass that lands in a student
group of its own, the fraction on one-to-one groups, and the fraction of
student positions that are shared boundaries. It also writes
`tables.safetensors`, which depends only on the tokenizer pair and can be
passed to later runs with `--tables`.

Two numbers gate the pair. A mean own-group mass fraction under 0.90
prints a warning, and under 0.70 the pass refuses unless `--force` is
given, since the projection would redirect too much of the teacher's mass.
On an identity pair both are 1.

## Train the adapter

```sh
gmlx distill train --view view/ --student Qwen3-0.6B-Q4_K_M.gguf \
    --adapter-out qwen3-0.6b-distill.gguf --iters 2000 --batch-size 8 --lr 1e-4
```

The loop is gmlx's own. The batch order comes from `--seed`, the schedule
is a linear warmup into a cosine decay, and `last` and `best` checkpoints
sit under `--ckpt-dir`, where `--resume` picks up at the exact iteration.
The adapter targets the attention and MLP projections of every layer at
rank 16. Training attention and the gated delta scan of hybrid models run
on the memory-saving paths described in the [LoRA guide](lora.md), and
`--grad-checkpoint` adds per-layer recomputation for long rows.

The objective is a sparse KL over the projected top-K plus a tail bucket
for the mass outside it, and on a cross-tokenizer pair a chunk-likelihood
term that matches the probability of whole words between the two
tokenizations. `--dk`, `--alm` and `--ce` weight the three terms. The
adapter is a GGUF that `run`, `chat` and `serve` attach with `--adapter`.

## Evaluate

```sh
gmlx distill eval --student Qwen3-0.6B-Q4_K_M.gguf --adapter qwen3-0.6b-distill.gguf \
    --before --cache cache/ --slice prose=heldout-prose.txt --slice code=heldout-code.txt \
    --kld-cache cache/ --md eval.md --json eval.json
```

`--before` disables the adapter in process for a second pass, so both
numbers come from one loaded model. Each `--slice` is scored in bits per
byte and checked against the cache's corpus, and a slice with more than
one percent of its 64-byte windows in the corpus is marked void. A
same-tokenizer `--kld-cache` gives the sparse KL against the teacher's own
targets. Downstream tasks come from local jsonl files under `--tasks-dir`,
and `--chat-sanity` scores template compliance and drift on a prompt set.
The JSON report carries every per-item result.

## Different tokenizers

A teacher and student that tokenize the same bytes differently agree on
some token boundaries and not others. At a shared boundary the two
distributions are over different vocabularies, so `align` builds, once per
tokenizer pair, a map from each student token to the teacher token that
starts the same bytes. Each teacher token in the top-K then names a group
of student tokens, and the target for that group is the sum of the
teacher's probabilities that map to it. A digit run, a longer merge or a
byte-fallback sequence becomes a constraint on a sum of student
probabilities instead of a wrong one-to-one target.

Between shared boundaries the student is trained on the likelihood of the
whole chunk of bytes, matched to the teacher's likelihood of the same
chunk, which is the ALM term. The census in `view.json` says how much of
the corpus this covers for a pair, and the same-tokenizer case reduces to
plain top-K distillation with nothing projected.

## Limitations

- The student is a K-quant GGUF with a LoRA adapter. Full-parameter
  training and MLX checkpoints are library features without a verb.
- The teacher's routing decisions are not recorded, so a MoE student
  trained from a MoE teacher of the same family learns from the teacher's
  outputs alone.
- Task files for `eval` are read from disk. Nothing is downloaded.
- One cache serves any student, but a view is bound to one cache and one
  tokenizer pair, and `train` refuses a view whose cache changed.
