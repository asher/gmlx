# Distill a teacher into a student

This guide is for training a small GGUF model on the knowledge of a larger
one without running both at once, including when the two use different
tokenizers and when the knowledge is a document the student will never be
shown. It walks through the `gmlx distill` actions, gives the settings that
produced the best results on a local pair, and says what each artifact
contains.

- [How it works](#how-it-works)
- [Choose a corpus](#choose-a-corpus)
- [Generate replies with the teacher](#generate-replies-with-the-teacher)
- [Filter the replies](#filter-the-replies)
- [Cache the teacher](#cache-the-teacher)
- [Align the cache to the student](#align-the-cache-to-the-student)
- [Train the adapter](#train-the-adapter)
- [Evaluate](#evaluate)
- [Recipe: teach a student a document it will not see](#recipe-teach-a-student-a-document-it-will-not-see)
- [What to expect](#what-to-expect)
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

Two more actions serve the case where the corpus is written by the
teacher itself. `gen` runs the teacher through `gmlx serve` over a prompt
set and writes its replies as a conversation corpus, and `filter` drops
the replies a student should not learn from. `census` is a diagnostic
that measures, from two caches of the same replies, how much a context
the student never sees moves the teacher. The flag tables for every
action are under [gmlx distill](cli.md#gmlx-distill).

## Choose a corpus

Fixed text is the simplest corpus. The teacher reads a jsonl file with a
`text` field per row, a directory of text files, or a Hugging Face
dataset id, and the student learns the teacher's distribution over
someone else's words. This transfers general ability and is the right
choice when the goal is a smaller model that behaves like the larger one
on ordinary text.

Generated replies are the corpus when the goal is a behavior: answering
in a format, following a document, reasoning before answering. The
teacher answers a prompt set, its replies are verified and filtered, and
the student learns the teacher's distribution over the teacher's own
words, which are on the teacher's policy and carry none of the noise of
a human corpus. Two rounds work better than one. In the first the teacher
writes the replies. In the second the student, with its first adapter,
writes replies to the same prompts, the teacher scores them, and the
student trains on the teacher's corrections of its own mistakes. The
[recipe](#recipe-teach-a-student-a-document-it-will-not-see) below runs
both rounds.

## Generate replies with the teacher

The prompt set is a jsonl of `{id, messages}` rows whose messages end on
a user turn. A row may add a `context` string, or the run may pass one
file with `--context` for every row without its own. The context goes
into the last user turn on the teacher's side only, and each written row
then carries `student_messages`, the prompt as given, which is what the
student later trains on:

```sh
gmlx distill gen --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --prompts prompts.jsonl \
    --context schema.md --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --concurrency 8 --out replies.jsonl
```

The action serves the teacher on `--port` with `--serve-arg` passed
through to `gmlx serve`, fans out `--concurrency` requests, and stops the
server when it is done. `--base-url` uses a server that is already
running instead, on this machine or elsewhere, so a teacher behind any
OpenAI-compatible endpoint can write the corpus. `--thinking` turns the
teacher's reasoning on and keeps the trace as `reasoning_content` on the
reply. `--thinking-budget` caps the trace per request, and the trace is
counted with the teacher's tokenizer so a reply whose trace was cut is
marked `budget_hit` for the filter. `--max-tokens` is the answer budget
after the trace.

Every reply carries a `gen` block with its finish reason, token counts,
seed and wall time, and the corpus has a sidecar `replies.jsonl.gen.json`
recording the model, the sampling settings and the run. The run is
resumable: prompt ids already in `--out` are skipped, and the exit code
is 1 when some requests failed so a rerun picks up the rest. Without a
prompt set, `--corpus` builds continuation prompts from a text corpus,
quoting the first `--prefix-chars` of each document under an
instruction to continue it.

## Filter the replies

Unfiltered generation distills the teacher's confident errors as readily
as its correct output, and the cache cannot tell them apart. The filter
runs a fixed sequence of checks and names the first one a row fails:
`length` when the reply did not reach its end of turn, `budget` when the
thinking budget cut the trace, `empty`, `marker` for a leaked template
string, `repeat`, `ascii`, `tokens` for a reply over `--max-reply-tokens`,
and `verify` when an external command rejects it:

```sh
gmlx distill filter --in replies.jsonl --out corpus.jsonl \
    --max-reply-tokens 1180 --verify "./check-sql.py --db freight.sqlite" \
    --report filter.json --rejects rejects.jsonl
```

The verify command reads the surviving rows as jsonl on stdin and prints
one line per row, `ok` or a reason word. A test runner, a SQL executor or
a schema checker plugs in there without the filter knowing the task, and
the rows a task checker rejects are the rows that would have taught the
student a wrong answer. `--max-reply-tokens` bounds the trace and answer
together so every training row fits the window the cache will cut.

`--context` prepares the second round. When the student wrote the
replies without a document, the filter puts the document on the
teacher's side of every kept row and keeps the bare prompt under
`student_messages`, so the teacher scores the student's reply with the
document in view while the student trains on the prompt alone. The
sidecar records the filter version, the counts per reason and the
context, and a cache cut from the corpus carries all of it in its
manifest.

## Cache the teacher

Rows are normalized to NFC and cut into windows of at most `--max-len`
teacher tokens at word boundaries:

```sh
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus corpus.jsonl --out cache/ \
    --top-k 256 --max-len 512 --max-disk-gb 20
```

The pass loads the teacher through the same loader as `run`, so a MoE
teacher larger than memory streams its experts from disk. It prints the
size estimate before loading and refuses when the estimate exceeds
`--max-disk-gb` or the free space. `--resume` continues after the last
verified shard. `--validate cache/` checks an existing cache without a
model load.

Conversation rows are cached with a frame. `--frame chat` reads
conversations under `--messages-key` and makes every assistant turn a
target, `--frame reply` targets only the final turn, and `--frame
reply-think` starts the target at the final turn's reasoning content, so
the student learns the trace and the answer behind the thinking-mode
generation prompt. `--frame continue` places each plain text window in a
model turn behind a fixed user instruction, so an instruct teacher sees
its own template. A generated corpus is cached with the reply frame, or
reply-think when it was generated with `--thinking`, and with
`--frame-kwargs '{"enable_thinking": true}'` so the teacher's render
matches the one that wrote the replies. `--max-len` must hold the
longest row with its frame, so a corpus filtered at 1180 reply tokens is
cached at 2560.

K is fixed at cache time and 256 is the default. A cross-tokenizer view
merges teacher tokens into groups at each boundary, so a wider K keeps
more of the teacher's mass in play after merging. The manifest records
the captured-mass histogram, the fraction of the teacher's probability
the top-K holds per position, which says how much the tail bucket
carries. The tokens that decide a tool call, the call's opener and
closer, the key names and the tool name, are the most certain positions
of a reply and sit at rank 1 in the cache with the whole mass, so a
student trained on the cache sees them at full weight and no per-token
weighting exists.

`--routes` stores, for a MoE teacher, the expert ids every layer chose
at every position beside the logits, one byte per expert slot up to 256
experts. A later forward over the same rows can then select the same
experts, which `eval --kld-cache` does when the student is the teacher
itself, so the measured KL is the quantization noise of the weights and
not a different routing of the same tokens. The pass refuses the flag on
a dense teacher and on a MoE block it cannot hook, and names the reason.

`--hidden` stores a sketch of the teacher's final hidden state at every
position beside the logits, for the hidden-state term of `train --hs`.
The sketch is the state projected through a random matrix of
`--hidden-dim` columns drawn from `--hidden-seed`, stored as float16,
so a 256-wide sketch adds 512 bytes per position to the 1.56 KB of a
K=256 cache. The manifest records the width, the seed and the teacher's
hidden size, and a second cache of the same teacher with the same seed
reproduces the same sketch.

## Align the cache to the student

```sh
gmlx distill align --cache cache/ --student Qwen3.5-9B-Q6_K.gguf --out view/
```

The pass runs on the CPU with the tokenizers only. It writes `view.json`
with the row index, the train and validation split, and the census of the
projection: the fraction of the teacher's mass that lands in a student
group of its own, the fraction on one-to-one groups, and the fraction of
student positions that are shared boundaries. It also writes
`tables.safetensors`, which depends only on the tokenizer pair and can be
passed to later runs with `--tables`, so a second cache over the same
pair aligns without rebuilding them.

Two numbers gate the pair. A mean own-group mass fraction under 0.90
prints a warning, and under 0.70 the pass refuses unless `--force` is
given, since the projection would redirect too much of the teacher's mass.
On an identity pair both are 1.

## Train the adapter

```sh
gmlx distill train --view view/ --student Qwen3.5-9B-Q6_K.gguf \
    --adapter-out qwen3.5-9b-distill.gguf --iters 678 --batch-size 3 \
    --lora-rank 128 --lora-alpha 64 --lr 5e-5 --seed 1
```

The loop is gmlx's own. The batch order comes from `--seed`, the schedule
is a linear warmup into a cosine decay, and `last` and `best` checkpoints
sit under `--ckpt-dir`, where `--resume` picks up at the exact iteration.
`--view` repeats to mix views over one tokenizer pair, which is how a
second round trains on the first round's rows and its own together. The
adapter targets the attention and MLP projections of every layer.
Training attention and the gated delta scan of hybrid models run on
gmlx's own memory-saving paths, whose switches are listed in the
[debug switches](internals/debug-switches.md) page, and
`--grad-checkpoint` adds per-layer recomputation for long rows.

`--iters` counts steps, so an epoch is the view's train rows divided by
`--batch-size`, rounded up. Two epochs over the training rows is enough
for a generated corpus, and the row count is in `view.json`:

```sh
python -c 'import json,math,sys; n=sum(e["split"]=="train" for v in sys.argv[1:] for e in json.load(open(v+"/view.json"))["index"]); print(n, 2*math.ceil(n/3))' view/
```

The objective is a sparse KL over the top-K plus a tail bucket for the
mass outside it. The teacher's top-K probabilities are renormalized over
the support, the student's are renormalized over the same support, and
a second term matches the teacher's true tail mass against the student's
mass outside the support, so the student is never asked to put all of its
probability on the top-K and never learns to hide mass in the tail. That
is `--loss bucketed`, the default. `--loss paper` drops the tail term,
which trains the student to match unnormalized top-K probabilities.
`--loss renorm` renormalizes both sides over the support and ignores the
tail, the form that lets a student score well on the support while its
mass outside it drifts. On a cross-tokenizer pair a chunk-likelihood term
matches the probability of whole words between the two tokenizations.
`--dk`, `--alm` and `--ce` weight the three terms, and the adapter is a
GGUF that `run`, `chat` and `serve` attach with `--adapter`.

`--hs W` adds a hidden-state term, off by default. A linear map from the
student's final hidden state to the cache's sketch is trained beside the
adapter, and at every shared boundary the loss adds `W` times one minus
the cosine between the mapped state and the teacher's sketch, or the
squared distance between the two unit vectors with `--hs-loss mse`. The
map lives in the checkpoint directory and never in the adapter. Every
view's cache must carry a hidden block from `cache --hidden` of the same
width, and the pass refuses otherwise. On the recipe in this guide the
term changed neither the logit terms during training nor the served
pass rates, which is why it is off.

The settings that held on a 9B student at Q6_K are rank 128 with alpha
64, a peak rate of 5e-5, batch 3 and two epochs, with `--dk 1 --alm 1
--ce 0`. Rows of up to about 1300 tokens ran at 9.3 seconds per step at
a peak of 50.7 GB on that student. A cross-entropy objective on the
teacher's tokens scored level with the sparse KL on the task endpoints
and behind it on every retention measure, so `--ce` stays at 0 unless a
run shows otherwise.

## Evaluate

```sh
gmlx distill eval --student Qwen3.5-9B-Q6_K.gguf --adapter qwen3.5-9b-distill.gguf \
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

A behavior is measured on replies. `--reply-slice` scores a jsonl of
conversations on the final turn only, or on the trace and the answer
with `--reply-think`, and `--chat-slice` scores every assistant turn of
a conversation set the adapter must not forget. `--reply-positions`
restricts a reply slice to the byte ranges a census found the context
moved, which is the one measure that sees a document being learned:
bits per byte over whole replies moves by a few thousandths when one
position in twenty gains a nat, inside the noise of a run.

The pass rate on a task is measured outside `eval`. Serve the student
with its adapter, generate its replies to the held-out prompts with
`gen`, and run the task checker over them as the filter's `--verify`
command or on its own.

## Recipe: teach a student a document it will not see

The worked case is a database schema. The teacher reads the schema and
answers questions with SQL, the student must answer the same questions
with no schema in its prompt, and a query counts as right when it runs
against the database and returns the reference rows. The same steps
serve any document with a checkable task: an API reference with tests, a
style guide with a linter, a rulebook with a judge.

Prepare three prompt sets before generating anything. Training prompts
cover the document's families of questions, held-out prompts of the same
families measure the trained behavior, and prompts of families never
trained on measure whether the student learned the document or only the
questions. Add compositional training prompts that join two families in
one question. Without them the student answers each family and fails the
joins, and with them the pass rate on the untrained families rose from
0.633 to 0.917 on the worked case. Rows that describe the document in
prose, as opposed to using it, lowered that rate and stay out.

Round one, the teacher writes the corpus:

```sh
gmlx distill gen --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --prompts prompts-train.jsonl \
    --context schema.md --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out r1-replies.jsonl
gmlx distill filter --in r1-replies.jsonl --out r1-corpus.jsonl \
    --max-reply-tokens 1180 --verify "./check-sql.py --db freight.sqlite" --rejects r1-rejects.jsonl
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus r1-corpus.jsonl --out cache-r1/ \
    --frame reply-think --frame-kwargs '{"enable_thinking": true}' --top-k 256 --max-len 2560
gmlx distill align --cache cache-r1/ --student Qwen3.5-9B-Q6_K.gguf --out view-r1/
gmlx distill train --view view-r1/ --student Qwen3.5-9B-Q6_K.gguf --adapter-out r1.gguf \
    --iters 302 --batch-size 3 --lora-rank 128 --lora-alpha 64 --lr 5e-5
```

The thinking budget matters twice. A trace that runs to the budget is a
reply the teacher did not finish, and the filter drops it, so expect to
lose about a third of the verified replies at 1000 tokens and generate
more prompts than rows you need. A trace also makes every row long, so
the reply cap, the cache window and the student's training rows are
sized together: 1180 reply tokens, a 2560-token cache window, and rows
of up to about 1300 student tokens.

Round two, the student writes and the teacher scores:

```sh
gmlx distill gen --teacher Qwen3.5-9B-Q6_K.gguf --serve-arg=--adapter --serve-arg=r1.gguf \
    --prompts prompts-train.jsonl --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out r2-replies.jsonl
gmlx distill filter --in r2-replies.jsonl --out r2-corpus.jsonl --max-reply-tokens 1180 \
    --verify "./check-sql.py --db freight.sqlite" --context schema.md
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus r2-corpus.jsonl --out cache-r2/ \
    --frame reply-think --frame-kwargs '{"enable_thinking": true}' --top-k 256 --max-len 2560
gmlx distill align --cache cache-r2/ --student Qwen3.5-9B-Q6_K.gguf --out view-r2/ --tables view-r1/
gmlx distill train --view view-r1/ --view view-r2/ --student Qwen3.5-9B-Q6_K.gguf \
    --adapter-out r2.gguf --iters 678 --batch-size 3 --lora-rank 128 --lora-alpha 64 --lr 5e-5
```

The student generates without the schema, the checker keeps the replies
that were right, and `filter --context` puts the schema on the teacher's
side so the cache holds the teacher's distribution over the student's
own words with the document in view. The second adapter trains on both
views from the base weights, not on top of the first adapter.

Check that the document moves the teacher before training against it.
Generate the teacher's replies to the held-out prompts with `--context`,
cache that corpus twice with `--frame reply-think`, once as written and
once with `--messages-key student_messages` so the teacher reads the
same replies without the schema, and run the census over the pair:

```sh
gmlx distill census --without cache-heldout-bare/ --with cache-heldout-ctx/ \
    --corpus heldout-ctx.jsonl --out census.json --md census.md
gmlx distill eval --student Qwen3.5-9B-Q6_K.gguf --adapter r2.gguf --before \
    --reply-slice heldout=heldout-ctx.jsonl --reply-think --reply-positions census.json \
    --chat-max-len 1536 --md eval.md --json eval.json
```

The census reports the distillable effect, the mean KL between the
teacher's distributions with and without the document, and the fraction
of positions the document moves by more than a nat. An effect under
about 0.05 nats or a high-delta fraction under one percent means the
document changes little that a student could learn, and the run is not
worth its hours. With two or more contexts the census also reports the
residual, the part of the effect that depends on which context is
present, which no training recovers because the student sees none of
them. The eval then scores the adapter at the positions the census
found, where the teacher's own nats per token with and without the
document are the two ends of the scale.

## What to expect

On the worked case, a 27B teacher at Q8 and a 9B student at Q6_K over a
schema of a few hundred lines, with 615 training questions, 264
compositional ones and two rounds:

| pass rate on | untouched student | after round one | after round two |
|---|---|---|---|
| held-out questions of the trained families | 0.022 | 0.817 | 0.882 |
| questions of families never trained on | 0.017 | 0.917 | 0.925 |
| compositional held-out questions | 0.000 | 0.767 | 0.783 |

The untouched student with the schema pasted into its prompt scores
0.946 on the held-out questions, so the adapter reaches most of what
retrieval gives and needs no schema at inference. At the census
positions the student's nats per token fell from 4.41 to 0.70, against
the teacher's 0.41 with the schema in view. The chat compliance rate
stayed at 1.00, GSM8K moved within its noise on 100 items, and bits per
byte on a code conversation set stayed level, so a rank 128 adapter
trained this way keeps the student's general behavior. Both rounds with
their evaluations took about 11 GPU hours on an M5 Max.

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
- A remote teacher can write the corpus through `gen --base-url`, and
  the trace count then needs `--tokenizer`. It cannot be cached: `cache`
  runs the teacher's forward pass itself and needs a local GGUF.
- Recorded routes are replayed by `eval --kld-cache` on the cache's own
  teacher. `train` does not replay them, so a MoE student trained from a
  MoE teacher of the same family learns from the teacher's outputs alone.
- The hidden-state term reads the teacher's final hidden state only. No
  intermediate layer is stored, and the sketch is fixed at cache time.
- Task files for `eval` are read from disk. Nothing is downloaded.
- One cache serves any student, but a view is bound to its cache and its
  tokenizer pair, and `train` refuses a view whose cache changed.
